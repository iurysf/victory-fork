//! Command queues. Submitted lists run on the host GPU and are waited for,
//! so fence signals complete at once.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const QUEUE_IIDS: [Iid; 4] = [
    guid(
        0x0ec870a6,
        0x5d7e,
        0x4c22,
        [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

/// The title reads the list type right after the header.
#[repr(C)]
struct Queue {
    object: Object,
    kind: u32,
    desc: QueueDesc,
    /// What the queue was given after a wait for a fence that had not been
    /// reached, in order; it runs when the fence gets there.
    #[cfg(windows)]
    held: std::sync::Mutex<Vec<Held>>,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);

/// One thing a waiting queue holds back.
#[cfg(windows)]
enum Held {
    /// The queue waits for the fence (its address) to reach the value.
    Wait(usize, u64),
    /// A list of the next submission waits for a memory value.
    Value(crate::list::ValueWait),
    /// A submission: the lists' host sides, taken from the title's lists
    /// (which it may reset meanwhile), and the order they run in.
    Run(Vec<crate::list::Detached>, Vec<usize>),
    /// A signal of the queue's, due after what it holds before it.
    Signal(usize, u64),
    /// With [`behind`]: the submission at the head of the line is being
    /// prepared and sent by our thread, with the line unlocked, so the
    /// title can add to its end meanwhile; whatever is added stays behind
    /// this (the line is not empty), signals too.
    Running,
    /// With [`present_behind`] or [`present_queued`]: the title's present
    /// of a frame, due after what the queue holds before it: the host
    /// texture shown (none in the tests), the state the title's lists
    /// leave it in, and the frame's number.
    /// With [`present_queued`] also the title's resource whose host state
    /// is read when the frame is shown ([`Lines::present_state`]; 0: the
    /// state given, as [`present_behind`] gives it).
    Present(Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>, u32, u64, usize),
    /// With [`present_queued`]: where another queue's line stood when the
    /// title presented frame `.0`. What is ahead of it ran before that
    /// present when the present ran the lines itself, and still does: the
    /// present waits until every line has reached its mark, and what is
    /// behind a mark waits until the frame has been shown.
    Mark(u64),
}

/// Presents that are in a queue's line and have not run ([`Held::Present`]).
#[cfg(windows)]
static PRESENTS_HELD: AtomicU64 = AtomicU64::new(0);

/// Whether a present goes behind the queue's line and returns at once
/// (`VICTORY_ASYNC_PRESENT=1`; needs [`behind`] and presents that do not
/// settle the flights, `crate::host::late_free_present`).
///
/// Why: with [`behind`] our thread prepares and sends the submissions, but
/// the title's present still ran the line itself first, that is: it stood
/// until our thread had prepared every submission of the frame. Run 225's
/// frame log (the cell, 136 ms a frame): the title's render thread is 77 ms
/// of every frame inside the present waiting for that, using 3 ms of
/// processor itself, and only then begins to record the next frame (about
/// 57 ms), during which our thread has nothing to prepare. The two phases
/// ran one after the other.
///
/// With the switch the present is one more thing in the line
/// ([`Held::Present`]): our thread shows the frame when it gets there, in
/// order, and the title records the next frame meanwhile.
///
/// The title is one frame ahead at most: a present that finds the one
/// before it still in a line runs the old way (the lines first, then
/// itself), and so does a frame that is saved or listed, and a present
/// that finds its queue's line empty. A wait nothing ever meets is
/// therefore still forced, by the present after it.
///
/// What it assumes, beyond what [`behind`] does: title memory a submission
/// reads stays as it was until the submission's fence is signalled, also
/// across the title's present. On the console the card reads that memory
/// when it runs the list, after the present call has returned.
#[cfg(windows)]
fn present_behind() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_ASYNC_PRESENT").is_some_and(|v| v == "1")
            && behind()
            && crate::host::late_free_present();
        if on {
            tracing::info!("presents go behind the queue's line and return at once (VICTORY_ASYNC_PRESENT=1)");
        }
        on
    })
}

/// With [`present_behind`]: puts the present of the next frame behind what
/// the queue `this` holds. False if it has to run the old way.
#[cfg(windows)]
fn send_behind(this: This, planes: u32, plane_parameters: *const Plane) -> bool {
    if this.is_null() || planes == 0 || PRESENTS_HELD.load(Ordering::Relaxed) != 0 {
        return false;
    }
    let frame = FRAMES.load(Ordering::Relaxed) + 1;
    // A frame that is saved or listed is read back at once.
    #[cfg(feature = "diagnostics")]
    if crate::capture::wanted(frame) || crate::capture::listed(frame) {
        return false;
    }
    // SAFETY: the title passes `planes` plane descriptions.
    let Some(plane) = (unsafe { plane_parameters.as_ref() }) else {
        return false;
    };
    if plane.count == 0 || plane.resources.is_null() {
        return false;
    }
    // SAFETY: the plane lists `count` resources.
    let resource = unsafe { *plane.resources };
    let Some(texture) = crate::resource::host_resource(resource) else {
        return false;
    };
    // The state the lists recorded so far leave the texture in: what our
    // copy finds it in when it runs behind them (looked at, not changed).
    const COPY_SOURCE: u32 = 0x800;
    let state = crate::resource::host_transition(resource, COPY_SOURCE).map_or(COPY_SOURCE, |(_, before)| before);
    crate::resource::host_transition(resource, state);
    let texture = texture.clone();
    let sent = hold(this, || {
        PRESENTS_HELD.fetch_add(1, Ordering::Relaxed);
        Held::Present(Some(texture), state, frame, 0)
    });
    if sent {
        PRESENTS_BEHIND.fetch_add(1, Ordering::Relaxed);
        wake();
    }
    sent
}

/// Presents sent behind their queue's line since the start
/// ([`send_behind`]), for the line every 512 of them.
#[cfg(windows)]
static PRESENTS_BEHIND: AtomicU64 = AtomicU64::new(0);

/// How many presents may wait in the lines (`VICTORY_PRESENT_QUEUED=1` or
/// `2`; 0, off, by default; needs [`behind`] and presents that do not
/// settle the flights, `crate::host::late_free_present`).
///
/// Why: with `VICTORY_ASYNC_SUBMIT=1` the title's render thread still
/// stood 42-46 ms of a 96 ms frame inside the present (`present_now` ->
/// `release_held_now` -> the one-releaser lock): the present ran the lines
/// itself, so it stood until our thread had prepared and sent the whole
/// frame, and only then did the title record the next one, while our
/// thread waited for work about as long. The frames went one after the
/// other through two threads that each stood half the time.
///
/// With the switch the present goes into the line of its queue
/// ([`Held::Present`]) and a mark into every other line that holds anything
/// ([`Held::Mark`]); the title's call returns at once and it records the
/// next frame while our thread ([`worker`]) prepares, sends and shows this
/// one. The order is the one the present made when it ran the lines:
/// - everything in the lines ahead of the present and its marks runs
///   before the frame is shown; first what is free to run, and once
///   nothing else moves, a wait ahead of a mark is passed, as the present
///   forced it ([`forces`]: a wait is not kept across frames);
/// - the present runs once every other line has reached its mark;
/// - what the title gives the queues after the present (behind the marks)
///   runs after the frame has been shown;
/// - signals and bottom-of-pipe writes stay in their lines' order and are
///   told when the card has run what is ahead of them, as before.
///
/// At most this many presents wait: the title's next present stands until
/// the oldest has been shown (as a swap chain's frame latency does), so
/// the title is that many frames ahead at most. A frame that is saved or
/// listed still runs the old way. `VICTORY_ASYNC_PRESENT=1` is left as it
/// is: one present behind the line at most, the next one running the lines
/// itself on the title's thread.
#[cfg(windows)]
pub(crate) fn present_queued() -> u64 {
    let run = present_queued_run();
    if !present_possible() {
        return 0;
    }
    // May turn within a run (`crate::ab`): presents queued before a turn
    // off are shown in their turn (the marks stay), and empty lines are
    // opened behind them still (`frame_in_flight`).
    crate::ab::number(crate::ab::Key::PresentQueued, run).min(2)
}

/// [`present_queued`] as the run sets it.
#[cfg(windows)]
fn present_queued_run() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        let asked = match std::env::var("VICTORY_PRESENT_QUEUED").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            _ => 0,
        };
        let most = if asked > 0 && present_possible() { asked } else { 0 };
        if most > 0 {
            tracing::info!(
                most,
                "presents go into the queue's line and return at once; our thread shows the frame in order, and the title's next present waits while {most} wait (VICTORY_PRESENT_QUEUED={most})"
            );
        } else if asked > 0 {
            tracing::warn!("VICTORY_PRESENT_QUEUED needs VICTORY_ASYNC_SUBMIT=1 and presents sent ahead (the relay, or VICTORY_LATE_WAIT=3): off");
        }
        most
    })
}

/// Whether presents may be queued at all: asynchronous submission
/// ([`behind`]) and presents that do not settle the flights.
#[cfg(windows)]
fn present_possible() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| behind() && crate::host::late_free_present())
}

/// The last frame shown from a line ([`Lines::present`]), and the word for
/// a present waiting at [`present_queued`]'s bound.
#[cfg(windows)]
static SHOWN: (std::sync::Mutex<u64>, std::sync::Condvar) = (std::sync::Mutex::new(0), std::sync::Condvar::new());

/// Since the last line ([`present_line`]): presents queued, presents the
/// old way, microseconds the title stood at the bound, presents in the
/// lines summed over the queued ones (at their call), the most at once,
/// microseconds of the title's present calls in all.
#[cfg(windows)]
static QUEUED_STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// With [`present_queued`]: puts the present of the next frame into the
/// line of `this`, marks the other lines, and returns: false if it has to
/// run the old way. Waits first while [`present_queued`] presents wait.
#[cfg(windows)]
fn queue_present(this: This, planes: u32, plane_parameters: *const Plane) -> bool {
    let most = present_queued();
    if most == 0 || this.is_null() || planes == 0 {
        return false;
    }
    let frame = FRAMES.load(Ordering::Relaxed) + 1;
    // A frame that is saved or listed is read back at once.
    #[cfg(feature = "diagnostics")]
    if crate::capture::wanted(frame) || crate::capture::listed(frame) {
        return false;
    }
    // SAFETY: the title passes `planes` plane descriptions.
    let Some(plane) = (unsafe { plane_parameters.as_ref() }) else {
        return false;
    };
    if plane.count == 0 || plane.resources.is_null() {
        return false;
    }
    // SAFETY: the plane lists `count` resources.
    let resource = unsafe { *plane.resources };
    let Some(texture) = crate::resource::host_resource(resource) else {
        return false;
    };
    let texture = texture.clone();
    // The bound: the oldest present waiting is shown first.
    let held = PRESENTS_HELD.load(Ordering::Acquire);
    if held >= most {
        let waiting = std::time::Instant::now();
        let mut shown = lock(&SHOWN.0);
        while PRESENTS_HELD.load(Ordering::Acquire) >= most {
            // (Woken by every present shown; looked at again now and then
            // in case the lines were run the old way meanwhile.)
            shown = SHOWN.1.wait_timeout(shown, std::time::Duration::from_millis(50)).unwrap_or_else(|e| e.into_inner()).0;
            wake();
        }
        drop(shown);
        QUEUED_STATS[2].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
    // The state the texture is in is read when the frame is shown: the
    // submissions ahead of the present in the lines have not been prepared
    // yet, and their transitions are made then (the bench: read here, 0x4
    // where the copy found 0x800).
    let in_lines = PRESENTS_HELD.fetch_add(1, Ordering::AcqRel) + 1;
    QUEUED_FRAME.fetch_max(frame, Ordering::AcqRel);
    queue_present_into(&Queues, this as usize, Some(texture), resource as usize, frame);
    QUEUED_STATS[0].fetch_add(1, Ordering::Relaxed);
    QUEUED_STATS[3].fetch_add(in_lines, Ordering::Relaxed);
    QUEUED_STATS[4].fetch_max(in_lines, Ordering::Relaxed);
    wake();
    true
}

/// [`queue_present`]'s work on the lines: the marks first, then the present,
/// so that a present found in a line has its marks in place.
#[cfg(windows)]
fn queue_present_into<L: Lines>(
    lines: &L,
    q: usize,
    texture: Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    resource: usize,
    frame: u64,
) {
    for other in lines.all_lines() {
        if other == q {
            continue;
        }
        let mut held = lock(lines.line(other));
        if !held.is_empty() {
            held.push(Held::Mark(frame));
        }
    }
    lock(lines.line(q)).push(Held::Present(texture, 0, frame, resource));
    lines.join_waiting(q);
}

/// With [`present_queued`]: the frame of the last present queued
/// ([`queue_present`]); not shown while it is above [`SHOWN`].
#[cfg(windows)]
static QUEUED_FRAME: AtomicU64 = AtomicU64::new(0);

/// With [`present_queued`]: the frame of a present queued and not shown
/// yet, if any.
#[cfg(windows)]
fn frame_in_flight() -> Option<u64> {
    // (Not asked of the switch: one turned off by the A/B leaves presents
    // queued behind it, shown in their turn.)
    let queued = QUEUED_FRAME.load(Ordering::Acquire);
    (queued > *lock(&SHOWN.0)).then_some(queued)
}

/// Before work goes into a line that is empty: with a frame in flight
/// (`in_flight`, [`frame_in_flight`]) the line is opened behind its mark,
/// and true is returned (the line holds something now). A line empty when
/// the present was queued got no mark ([`queue_present_into`]), and what
/// the title gave it afterwards ran before the frame was shown; it ran
/// after, when the present ran the lines itself (the bench's
/// empty_queue_leak).
#[cfg(windows)]
fn open_line(held: &mut Vec<Held>, in_flight: Option<u64>) -> bool {
    match in_flight {
        Some(frame) if held.is_empty() => {
            held.push(Held::Mark(frame));
            true
        }
        _ => false,
    }
}

/// For the frame-rate line, with [`present_queued`], every 256 presents.
#[cfg(windows)]
fn present_line(frames: u64) {
    let s: Vec<u64> = QUEUED_STATS.iter().map(|s| s.swap(0, Ordering::Relaxed)).collect();
    let f = frames.max(1) as f64;
    tracing::info!(
        "presents queued behind the lines (VICTORY_PRESENT_QUEUED={}) over the last {frames} frames: {} queued, {} the old way; \
         the title's present calls took {:.1} ms a frame, {:.1} of them waiting for an earlier frame to be shown; frames in \
         flight when a present was queued {:.2} on average, {} at most",
        present_queued(),
        s[0],
        s[1],
        s[5] as f64 / 1000.0 / f,
        s[2] as f64 / 1000.0 / f,
        if s[0] > 0 { s[3] as f64 / s[0] as f64 } else { 0.0 },
        s[4],
    );
}

/// The queues that hold something (their addresses).
#[cfg(windows)]
static WAITING: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

#[cfg(windows)]
fn lock<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether queues wait for fences. `VICTORY_QUEUE_WAIT=0`: they do not,
/// and every submission runs when it is made (as before).
#[cfg(windows)]
fn waits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_QUEUE_WAIT").is_none_or(|v| v != "0"))
}

/// Whether a submission is held list by list: the lists before the first
/// one whose wait for a memory value is not met run at once, and each list
/// that waits is held with the lists after it, behind its own waits.
/// Holding a submission whole made the title's queues wait for each other:
/// a graphics submission was held for a value a compute list writes, and
/// that compute list for a value an earlier list of the same graphics
/// submission writes. Nothing moved until the present forced it, and the
/// title reached its present only after waiting about 2.5 s, every frame
/// (runs 27-30: 2.4-2.9 s from the last submission to the present call,
/// 13-32 ms before waits were honoured; the frame's main graphics
/// submissions, 88 lists, ran forced, the compute lists before them).
/// On by default; `VICTORY_SPLIT_WAITS=0` holds whole submissions, as
/// before.
#[cfg(windows)]
fn split() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SPLIT_WAITS").is_none_or(|v| v != "0"))
}

/// Whether a list is cut into host lists at its waits for memory values
/// (`crate::list::cut`), and a queue runs a submission part by part: it
/// stands at each wait until the value is in memory, with everything
/// recorded before the wait run and its bottom-of-pipe writes done, as on
/// the console. Held list by list ([`split`]) a wait still held its whole
/// list: the title's lists are chains of jobs, each behind a wait for a
/// value another queue's list writes in the middle of its own chain
/// (frame 858 of run 33: a compute list of 433 commands with 13 waits;
/// graphics lists of the form "341 dispatches, wait, draws"), so two lists
/// each waited for a write behind the other's wait and only the present
/// moved them, in the wrong order.
/// On by default where waits are honoured list by list;
/// `VICTORY_CUT_WAITS=0` holds whole lists, as before.
#[cfg(windows)]
pub(crate) fn cuts() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    waits() && split() && *ON.get_or_init(|| std::env::var_os("VICTORY_CUT_WAITS").is_none_or(|v| v != "0"))
}

/// Whether a present runs what queues still hold (on by default: a wait
/// is not kept across frames). `VICTORY_WAIT_FORCE=0` keeps it held: for
/// finding out whether the title's waits reach across its presents; a
/// wait nothing ever meets then stops the title for good.
#[cfg(windows)]
fn forces() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WAIT_FORCE").is_none_or(|v| v != "0"))
}

/// A submission to a queue whose lists are cut at their waits ([`cuts`]).
/// The lists before the first one that waits run at once if the queue
/// holds nothing; the rest goes behind what the queue holds, part by part
/// (`crate::list::detach_parts`): the waits of a part, then its host
/// lists. [`release_held`] runs the queue's line as far as the values are
/// there, now and whenever a submission has run.
#[cfg(windows)]
fn submit_cut(this: This, lists: &[This]) {
    let Some(host) = crate::host::host() else {
        return;
    };
    let first = if !behind() && lock(&queue(this).held).is_empty() {
        lists
            .iter()
            .position(|&l| crate::list::cut_list(l))
            .unwrap_or(lists.len())
    } else {
        0
    };
    #[cfg(feature = "diagnostics")]
    crate::capture::op(this as usize, || {
        let named: Vec<String> = lists
            .iter()
            .map(|&l| match crate::list::host_list(l) {
                Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                None => format!("title {l:?}"),
            })
            .collect();
        format!(
            "SUBMIT queue kind {:?} held {} lists {named:?} (the first {first} run at once, the rest part by part)",
            Some(queue(this).kind),
            first < lists.len()
        )
    });
    if first > 0 && crate::host::late() {
        // Sent to the card and not waited for: their host sides are taken
        // from the title's lists (which it may reset before they have run)
        // and come back, with their tables' slots, when they are settled.
        let (taken, order) = crate::list::detach(&lists[..first]);
        host.execute_late(taken, order);
    } else if first > 0 {
        let now: Vec<_> = lists[..first]
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.execute(&now);
        // They have run: their descriptor tables' slots are free again, as
        // on the path without cuts ([`execute`]). This path replaced that
        // one without the call, so the lists that run at once kept their
        // slots until the title's reset and the ring ran out as before
        // (run 45: 3,143 of 15,942 tables refused in one frame).
        crate::list::tables_done(&lists[..first]);
    }
    if first < lists.len() {
        let parts = crate::list::detach_parts(&lists[first..]);
        static SEEN: AtomicU64 = AtomicU64::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(
                kind = queue(this).kind,
                lists = lists.len(),
                run_at_once = first,
                parts = parts.len(),
                waits = parts.iter().map(|p| p.0.len()).sum::<usize>(),
                "a submission is given to its queue part by part (its lists wait for memory values)"
            );
        }
        {
            let mut held = lock(&queue(this).held);
            open_line(&mut held, frame_in_flight());
            for (waits, taken, order) in parts {
                held.extend(waits.into_iter().map(Held::Value));
                held.push(Held::Run(taken, order));
            }
        }
        let mut waiting = lock(&WAITING);
        if !waiting.contains(&(this as usize)) {
            waiting.push(this as usize);
        }
    }
    // What ran may be what another queue waits for, and what was just
    // given to the queue may be free to run already.
    release_held(false);
}

/// A submission to a queue that honours waits, list by list ([`split`]).
/// shortcut: a wait still holds its whole list, not only the commands
/// after it; and lists named twice among those held stay in one piece.
#[cfg(windows)]
fn submit_split(this: This, lists: &[This]) {
    let Some(host) = crate::host::host() else {
        return;
    };
    let by = crate::list::waits_by_list(lists);
    // Behind what the queue already holds, nothing runs now.
    let cut = if lock(&queue(this).held).is_empty() && frame_in_flight().is_none() {
        (0..lists.len())
            .find(|&i| by[i].iter().any(|w| !w.met()))
            .unwrap_or(lists.len())
    } else {
        0
    };
    #[cfg(feature = "diagnostics")]
    crate::capture::op(this as usize, || {
        let named: Vec<String> = lists
            .iter()
            .map(|&l| match crate::list::host_list(l) {
                Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                None => format!("title {l:?}"),
            })
            .collect();
        format!(
            "SUBMIT queue kind {:?} held {} lists {named:?} (the first {cut} run at once)",
            Some(queue(this).kind),
            cut < lists.len()
        )
    });
    if cut > 0 {
        let now: Vec<_> = lists[..cut]
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.execute(&now);
    }
    if cut < lists.len() {
        static SEEN: AtomicU64 = AtomicU64::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(
                kind = queue(this).kind,
                lists = lists.len(),
                run_at_once = cut,
                waits = ?by[cut],
                "lists of a submission are held for a memory value"
            );
        }
        let rest = &lists[cut..];
        // A list's host side can be taken once.
        let twice = rest.iter().enumerate().any(|(i, l)| rest[..i].contains(l));
        let mut pieces: Vec<Held> = Vec::new();
        let mut from = cut;
        while from < lists.len() {
            // Up to the next list that waits for anything (met or not: it
            // is looked at again when its turn comes).
            let to = if twice {
                lists.len()
            } else {
                (from + 1..lists.len())
                    .find(|&i| !by[i].is_empty())
                    .unwrap_or(lists.len())
            };
            let waits: Vec<crate::list::ValueWait> = by[from..to].concat();
            pieces.extend(waits.into_iter().map(Held::Value));
            let (taken, order) = crate::list::detach(&lists[from..to]);
            pieces.push(Held::Run(taken, order));
            from = to;
        }
        {
            let mut held = lock(&queue(this).held);
            open_line(&mut held, frame_in_flight());
            held.extend(pieces);
        }
        let mut waiting = lock(&WAITING);
        if !waiting.contains(&(this as usize)) {
            waiting.push(this as usize);
        }
    }
    // What ran may be what another queue waits for, and what was just held
    // may be free to run already.
    release_held(false);
}

/// Adds `what()` to what the queue holds, if it holds anything (it waits
/// for a fence): whether it was added.
#[cfg(windows)]
fn hold(this: This, what: impl FnOnce() -> Held) -> bool {
    if this.is_null() {
        return false;
    }
    let mut held = lock(&queue(this).held);
    if held.is_empty() {
        if !open_line(&mut held, frame_in_flight()) {
            return false;
        }
        held.push(what());
        drop(held);
        let mut waiting = lock(&WAITING);
        if !waiting.contains(&(this as usize)) {
            waiting.push(this as usize);
        }
        return true;
    }
    held.push(what());
    true
}

/// Runs what waiting queues hold as far as their fences have been reached
/// (`all`: all of it), each queue's in the order it was given. Called
/// where a fence may have moved: after a signal and after a submission (a
/// list's bottom-of-pipe write may be into a fence), never from inside
/// one.
#[cfg(windows)]
pub(crate) fn release_held(all: bool) {
    if !all && behind() {
        // A thread of ours runs what is free to run ([`behind`]).
        wake();
        return;
    }
    release_held_now(all);
}

/// Whether a submission is only put in its queue's line by the thread
/// that submits it, and prepared and sent by a thread of ours
/// (`VICTORY_ASYNC_SUBMIT=1`; needs flights and lists cut at their waits,
/// both defaults).
///
/// Why: the profile of run 213 (292 ms a frame) has the title's render
/// thread inside `queue::execute` for 202 ms of every frame: 155 ms in
/// `Host::submit` (preparing: the copies of title memory, the write
/// watch, descriptors, textures; and waiting for flights), while its five
/// recording threads stand idle 78 % of the time and the card 43 %. The
/// title's thread does nothing with the result of that work: it only has
/// to be done before the lists run, in order.
///
/// With the switch a submission's lists are taken from the title's lists
/// (as they already are when the queue holds anything) and put behind
/// the queue's line; one thread of ours ([`worker`]) runs the line in
/// order. A signal of the queue goes behind them, as it does behind
/// anything held. Everything that must see earlier submissions done runs
/// the line itself first: a present, and every submission of ours that
/// is waited for ([`before_sync`]).
///
/// What it assumes: title memory a submission reads is not rewritten by
/// the title between its submit call and the moment our thread prepares
/// it, some milliseconds later. On the console the card reads that
/// memory when it runs the list, later still, so a title that rewrote it
/// earlier would be wrong there too.
#[cfg(windows)]
pub(crate) fn behind() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_ASYNC_SUBMIT").is_some_and(|v| v == "1") && crate::host::late() && cuts();
        if on {
            tracing::info!("submissions are prepared and sent by a thread of ours, not by the thread that submits (VICTORY_ASYNC_SUBMIT=1)");
            let _ = std::thread::Builder::new().name("victory-submit".into()).spawn(worker);
        }
        on
    })
}

/// Whether there is something for [`worker`] to look at.
#[cfg(windows)]
static WORK: (std::sync::Mutex<bool>, std::sync::Condvar) = (std::sync::Mutex::new(false), std::sync::Condvar::new());

#[cfg(windows)]
fn wake() {
    *lock(&WORK.0) = true;
    WORK.1.notify_one();
}

/// The thread of [`behind`]: runs what queues hold as far as it is free
/// to run, each time something was added or a fence moved.
#[cfg(windows)]
fn worker() {
    crate::framelog::keep_on_fast_cores("victory-submit");
    loop {
        {
            let mut work = lock(&WORK.0);
            while !*work {
                work = WORK.1.wait(work).unwrap_or_else(|e| e.into_inner());
            }
            *work = false;
        }
        let started = std::time::Instant::now();
        let (user, kernel) = crate::framelog::thread_times();
        if crate::framelog::on() {
            crate::framelog::add(crate::framelog::T::WorkerLooks, 1);
            if crate::framelog::on_slow_core() {
                crate::framelog::add(crate::framelog::T::WorkerSlow, 1);
            }
        }
        release_held_now(false);
        let (user_now, kernel_now) = crate::framelog::thread_times();
        crate::framelog::add(crate::framelog::T::WorkerUs, started.elapsed().as_micros() as u64);
        crate::framelog::add(crate::framelog::T::WorkerUserUs, user_now.saturating_sub(user));
        crate::framelog::add(crate::framelog::T::WorkerKernelUs, kernel_now.saturating_sub(kernel));
    }
}

#[cfg(windows)]
thread_local! {
    /// This thread is inside [`release_held_now`].
    static RELEASING_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether parts that follow one another in the queues' lines go to the
/// host as one submission (`VICTORY_JOIN_PARTS=1`; `alt`: twenty seconds
/// with, twenty without, in turn, each change said in the log). Off by
/// default. Needs [`behind`].
///
/// Why: the title's lists wait for memory values (19 to 20 waits a frame
/// in the first rooms, each met by the bottom-of-pipe write of the part
/// before it), and every wait cuts a submission into parts: 29 host
/// submissions a frame from about 15 of the title's (run 238). Each part
/// is prepared on its own: 82 askings of the write watch, 666 refresh
/// steps, a heap sync, a merge under the flights in the air, and on the
/// card one list of ours that sends changed bytes and one that compares.
/// The title hands a frame over in one burst at the end of its recording,
/// then stands still until the line has finished it (its fence wait of
/// the next frame: run 238), so the parts are all in the lines when our
/// thread gets to them.
///
/// With the switch the thread that runs the lines takes parts off them
/// in the order they would have run and gives them to the host together:
/// the same host lists in the same order, prepared once. Between two
/// parts a wait passes if memory meets it, if a write of a flight in the
/// air is known to meet it, or if a bottom-of-pipe write of a part taken
/// before it meets it (that part is ahead of it in the joined
/// submission, which is the order the wait asks for). A signal met
/// behind parts not sent yet is delivered right behind the joined
/// submission: never before the part it followed has run. Anything else
/// (a wait not known to be met, a present, the end of what is free to
/// run) sends what was taken first. At most [`join_most`] parts or
/// [`join_lists`] lists are joined, so that the card has work while the
/// next ones are prepared. Each queue with a part taken keeps the mark
/// [`Held::Running`] at the head of its line until the part is sent:
/// what the title gives the queue meanwhile, signals too, stays behind.
#[cfg(windows)]
fn join_parts() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let mode = *MODE.get_or_init(|| {
        let mode = match std::env::var("VICTORY_JOIN_PARTS").as_deref() {
            Ok("1") => 1,
            Ok("alt") => 2,
            _ => 0,
        };
        if mode != 0 {
            tracing::info!(
                "parts that follow one another in the queues' lines go to the host as one submission, at most {} parts or {} lists (VICTORY_JOIN_PARTS=1; alt: in turn)",
                join_most(),
                join_lists()
            );
        }
        mode
    });
    // May turn within a run (`crate::ab`): it is read once each time the
    // lines are run, and every joined submission is sent before they return.
    match mode {
        0 => crate::ab::flag(crate::ab::Key::JoinParts, false),
        1 => crate::ab::flag(crate::ab::Key::JoinParts, true),
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!("parts joined into one host submission from now (VICTORY_JOIN_PARTS=alt): {}", if on { "ON" } else { "off, as usual" });
            }
            on
        }
    }
}

/// The most parts joined into one host submission (`VICTORY_JOIN_MOST`, 4).
#[cfg(windows)]
fn join_most() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_JOIN_MOST").ok().and_then(|v| v.parse().ok()).filter(|&n| n >= 1).unwrap_or(4))
}

/// The most host lists joined into one host submission (`VICTORY_JOIN_LISTS`, 48).
#[cfg(windows)]
fn join_lists() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_JOIN_LISTS").ok().and_then(|v| v.parse().ok()).filter(|&n| n >= 1).unwrap_or(48))
}

/// With [`join_parts`]: parts taken off the queues' lines and not yet
/// given to the host, in the order they run.
#[cfg(windows)]
#[derive(Default)]
struct Joined {
    lists: Vec<crate::list::Detached>,
    order: Vec<usize>,
    parts: usize,
    /// Signals met behind those parts: delivered when the parts are sent.
    signals: Vec<(usize, u64)>,
    /// The queues with a part here: each has [`Held::Running`] at the
    /// head of its line.
    marked: Vec<usize>,
}

#[cfg(windows)]
impl Joined {
    fn is_empty(&self) -> bool {
        self.parts == 0
    }

    /// One more part behind those taken: its lists keep their order.
    fn add(&mut self, lists: Vec<crate::list::Detached>, order: Vec<usize>) {
        let (lists, order) = join(std::mem::take(&mut self.lists), std::mem::take(&mut self.order), lists, order);
        self.lists = lists;
        self.order = order;
        self.parts += 1;
    }

    fn full(&self) -> bool {
        self.parts >= join_most() || self.order.len() >= join_lists()
    }

    /// The last bottom-of-pipe write of the parts taken over `[address,
    /// address + bytes)` (`crate::host::last_write_over`): the newest
    /// part's first.
    fn last_write(&self, address: usize, bytes: usize) -> Option<Option<u64>> {
        self.order.iter().rev().find_map(|&i| self.lists[i].host().last_write(address, bytes))
    }

    /// Whether `next` (a part's lists, in `order`) must go after these have
    /// been sent (`crate::host::join_blocked`).
    fn blocks(&self, next: &[crate::list::Detached], order: &[usize]) -> bool {
        if self.is_empty() {
            return false;
        }
        let taken: Vec<&crate::host::HostList> = self.order.iter().map(|&i| self.lists[i].host()).collect();
        let next: Vec<&crate::host::HostList> = order.iter().map(|&i| next[i].host()).collect();
        crate::host::join_blocked(&taken, &next)
    }
}

#[cfg(windows)]
/// The things of one part behind those of another: `order` names places in
/// `lists`, so the second part's places move up by the first's count.
fn join<T>(mut lists: Vec<T>, mut order: Vec<usize>, more: Vec<T>, more_order: Vec<usize>) -> (Vec<T>, Vec<usize>) {
    let base = lists.len();
    lists.extend(more);
    order.extend(more_order.into_iter().map(|i| i + base));
    (lists, order)
}

/// Parts joined, host submissions they went in, their lists, signals
/// delivered behind them, and waits passed for a write of a part taken.
#[cfg(windows)]
static JOINED: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Gives the host what [`Joined`] holds, as one submission, delivers the
/// signals met behind it, and takes the marks off the lines. Called with
/// no line locked.
#[cfg(windows)]
fn send_joined<L: Lines>(lines: &L, joined: &mut Joined) {
    if joined.parts > 0 {
        let (lists, order) = (std::mem::take(&mut joined.lists), std::mem::take(&mut joined.order));
        JOINED[0].fetch_add(joined.parts as u64, Ordering::Relaxed);
        let sent = JOINED[1].fetch_add(1, Ordering::Relaxed) + 1;
        JOINED[2].fetch_add(order.len() as u64, Ordering::Relaxed);
        JOINED[3].fetch_add(joined.signals.len() as u64, Ordering::Relaxed);
        if sent % 2048 == 0 {
            let n: Vec<u64> = JOINED.iter().map(|n| n.swap(0, Ordering::Relaxed)).collect();
            tracing::info!(
                "parts joined (VICTORY_JOIN_PARTS=1): {} parts went to the host in {} submissions ({:.1} parts, {:.1} lists each), {} signals delivered behind them, {} waits passed for a write of a part not sent yet, {} parts sent apart because they read on the CPU what a part taken writes",
                n[0],
                n[1],
                n[0] as f64 / n[1].max(1) as f64,
                n[2] as f64 / n[1].max(1) as f64,
                n[3],
                n[4],
                n[5]
            );
        }
        joined.parts = 0;
        lines.send(lists, order);
    }
    // Behind what was just sent, in the order they were met.
    for (fence, value) in joined.signals.drain(..) {
        lines.signal(fence, value);
    }
    for q in joined.marked.drain(..) {
        let mut held = lock(lines.line(q));
        if matches!(held.first(), Some(Held::Running)) {
            held.remove(0);
        }
        if held.is_empty() {
            lines.leave(q);
        }
    }
}

/// Before a submission of ours that is waited for, with [`behind`]: what
/// the queues hold and is free to run runs first, on this thread, so the
/// submission keeps its place behind them.
#[cfg(windows)]
pub(crate) fn before_sync() {
    if behind() && !RELEASING_HERE.get() {
        release_held_now(false);
    }
}

#[cfg(windows)]
fn release_held_now(all: bool) {
    // One caller at a time, so a queue's submissions keep their order.
    static RELEASING: std::sync::Mutex<()> = std::sync::Mutex::new(());
    if lock(&WAITING).is_empty() {
        return;
    }
    let _one = lock(&RELEASING);
    let was_here = RELEASING_HERE.replace(true);
    struct Leave(bool);
    impl Drop for Leave {
        fn drop(&mut self) {
            RELEASING_HERE.set(self.0);
        }
    }
    let _leave = Leave(was_here);
    // Parts taken and not sent yet ([`join_parts`]): none when the lines
    // are forced, and none is left when this returns.
    let joining = !all && behind() && crate::host::late() && crate::host::host().is_some() && join_parts();
    // Presents queued and not shown keep their order even when the switch
    // was turned off since (`crate::ab`).
    walk(&Queues, all, joining, present_queued() > 0 || frame_in_flight().is_some());
}

/// What [`walk`] works on: the queues' lines and what is done with what
/// leaves them. [`Queues`] in the layer; the tests' own lines in the
/// tests, which record what was done in which order.
#[cfg(windows)]
trait Lines {
    /// The queues with something in their lines.
    fn waiting(&self) -> Vec<usize>;
    /// `q`'s line is empty: off [`Lines::waiting`].
    fn leave(&self, q: usize);
    /// The queues with something in their lines, looked at without
    /// anything else ([`Lines::waiting`] may do more in the tests).
    fn all_lines(&self) -> Vec<usize>;
    /// `q`'s line holds something now: on [`Lines::waiting`].
    fn join_waiting(&self, q: usize);
    /// The last frame shown from a line ([`Held::Present`]).
    fn shown(&self) -> u64;
    fn line(&self, q: usize) -> &std::sync::Mutex<Vec<Held>>;
    /// Whether the fence (its address) has reached `value`.
    fn reached(&self, fence: usize, value: u64) -> bool;
    /// Whether a queue may pass `wait` ([`value_met`]: may settle).
    fn value_met(&self, wait: &crate::list::ValueWait) -> bool;
    /// With parts taken and not sent (`joined`): whether `wait` holds once
    /// what is ahead of it has run, the newest known write first
    /// (`crate::host::wait_holds_guarded`; `None`: not known without the
    /// card), and whether a part taken's write decided it.
    fn joined_wait(&self, joined: &Joined, wait: &crate::list::ValueWait) -> (Option<bool>, bool);
    /// Whether the part `lists`/`order` must go after the parts in `joined`
    /// have been sent ([`Joined::blocks`]).
    fn joined_blocks(&self, joined: &Joined, lists: &[crate::list::Detached], order: &[usize]) -> bool;
    /// The parts joined go to the host as one submission, not waited for.
    fn send(&self, lists: Vec<crate::list::Detached>, order: Vec<usize>);
    /// A signal behind what was sent.
    fn signal(&self, fence: usize, value: u64);
    /// A part not joined, as it always went: `held` is `line`'s lock,
    /// given back locked.
    fn run<'a>(
        &self,
        line: &'a std::sync::Mutex<Vec<Held>>,
        held: std::sync::MutexGuard<'a, Vec<Held>>,
        lists: Vec<crate::list::Detached>,
        order: Vec<usize>,
        all: bool,
    ) -> std::sync::MutexGuard<'a, Vec<Held>>;
    /// The host state of the title's resource `resource` (its address)
    /// now: what the copy of a queued present finds it in, the submissions
    /// ahead of it prepared (looked at, not changed).
    fn present_state(&self, resource: usize) -> u32;
    /// A present in a line, as [`Lines::run`].
    fn present<'a>(
        &self,
        line: &'a std::sync::Mutex<Vec<Held>>,
        held: std::sync::MutexGuard<'a, Vec<Held>>,
        texture: Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
        state: u32,
        frame: u64,
        all: bool,
    ) -> std::sync::MutexGuard<'a, Vec<Held>>;
}

/// The layer's queues ([`WAITING`]) and the host.
#[cfg(windows)]
struct Queues;

#[cfg(windows)]
impl Lines for Queues {
    fn waiting(&self) -> Vec<usize> {
        lock(&WAITING).clone()
    }

    fn leave(&self, q: usize) {
        lock(&WAITING).retain(|&w| w != q);
    }

    fn line(&self, q: usize) -> &std::sync::Mutex<Vec<Held>> {
        // SAFETY: a live queue: one released is taken off the list.
        unsafe { &(*(q as *const Queue)).held }
    }

    fn all_lines(&self) -> Vec<usize> {
        lock(&WAITING).clone()
    }

    fn join_waiting(&self, q: usize) {
        let mut waiting = lock(&WAITING);
        if !waiting.contains(&q) {
            waiting.push(q);
        }
    }

    fn shown(&self) -> u64 {
        *lock(&SHOWN.0)
    }

    fn reached(&self, fence: usize, value: u64) -> bool {
        crate::fence::completed(fence as This) >= value
    }

    fn value_met(&self, wait: &crate::list::ValueWait) -> bool {
        value_met(wait)
    }

    fn joined_blocks(&self, joined: &Joined, lists: &[crate::list::Detached], order: &[usize]) -> bool {
        joined.blocks(lists, order)
    }

    fn present_state(&self, resource: usize) -> u32 {
        const COPY_SOURCE: u32 = 0x800;
        let resource = resource as This;
        let state = crate::resource::host_transition(resource, COPY_SOURCE).map_or(COPY_SOURCE, |(_, before)| before);
        crate::resource::host_transition(resource, state);
        state
    }

    fn joined_wait(&self, joined: &Joined, wait: &crate::list::ValueWait) -> (Option<bool>, bool) {
        let (address, bytes) = wait.place();
        let mine = joined.last_write(address, bytes);
        let (flights, unsure) = crate::host::late_last_write(address, bytes);
        (crate::host::wait_holds_guarded(wait, mine, flights, unsure, || wait.met()), matches!(mine, Some(Some(_))))
    }

    fn send(&self, lists: Vec<crate::list::Detached>, order: Vec<usize>) {
        match crate::host::host() {
            Some(host) => host.execute_late(lists, order),
            // (Not reached: parts are only taken with a host.)
            None => crate::list::recycle(lists, &order),
        }
    }

    fn signal(&self, fence: usize, value: u64) {
        if !crate::host::late_signal(fence, value) {
            crate::fence::set(fence as This, value);
        }
    }

    fn run<'a>(
        &self,
        line: &'a std::sync::Mutex<Vec<Held>>,
        mut held: std::sync::MutexGuard<'a, Vec<Held>>,
        lists: Vec<crate::list::Detached>,
        order: Vec<usize>,
        all: bool,
    ) -> std::sync::MutexGuard<'a, Vec<Held>> {
        if crate::host::late()
            && let Some(host) = crate::host::host()
        {
            // Not waited for: the host keeps the lists
            // until they are settled.
            if behind() && !all {
                // The line is open while this part is
                // prepared and sent: only this thread takes
                // from its head (one releaser at a time).
                held.insert(0, Held::Running);
                drop(held);
                host.execute_late(lists, order);
                held = lock(line);
                if matches!(held.first(), Some(Held::Running)) {
                    held.remove(0);
                }
            } else {
                host.execute_late(lists, order);
            }
        } else {
            let ran = {
                let runs: Vec<&crate::host::HostList> =
                    order.iter().map(|&i| lists[i].host()).collect();
                match crate::host::host() {
                    Some(host) => {
                        host.execute(&runs);
                        true
                    }
                    None => false,
                }
            };
            // They have run: their host lists are used again.
            if ran {
                crate::list::recycle(lists, &order);
            }
        }
        if all {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Forced, 1);
            static ONCE: AtomicBool = AtomicBool::new(false);
            if !ONCE.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "a queue still held a submission at a present: its fence was not reached"
                );
            }
        }
        held
    }

    fn present<'a>(
        &self,
        line: &'a std::sync::Mutex<Vec<Held>>,
        mut held: std::sync::MutexGuard<'a, Vec<Held>>,
        texture: Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
        state: u32,
        frame: u64,
        all: bool,
    ) -> std::sync::MutexGuard<'a, Vec<Held>> {
        let show = || {
            if let Some(host) = crate::host::host()
                && let Some(texture) = &texture
            {
                let _ = host.present(texture, state, frame);
            }
        };
        if all {
            show();
        } else {
            // The line is open while the frame is shown,
            // as it is while a submission is prepared.
            held.insert(0, Held::Running);
            drop(held);
            show();
            held = lock(line);
            if matches!(held.first(), Some(Held::Running)) {
                held.remove(0);
            }
        }
        PRESENTS_HELD.fetch_sub(1, Ordering::AcqRel);
        {
            let mut shown = lock(&SHOWN.0);
            *shown = (*shown).max(frame);
        }
        SHOWN.1.notify_all();
        held
    }
}

/// Runs what the lines hold as far as it is free to run (`all`: all of
/// it), each queue's in the order it was given; with `joining`
/// ([`join_parts`]) parts that follow one another go to the host
/// together. One caller at a time ([`release_held_now`]).
#[cfg(windows)]
fn walk<L: Lines>(lines: &L, all: bool, joining: bool, marks: bool) {
    let mut joined = Joined::default();
    // With `marks` ([`present_queued`]): a pass in which a wait ahead of a
    // queued present or its mark is passed, as the present forced it when
    // it ran the lines itself; only once nothing else moves.
    let mut forcing = false;
    loop {
        let force_now = std::mem::take(&mut forcing);
        let queues = lines.waiting();
        let mut moved = false;
        // A line stands at a wait with a present or a mark behind it.
        let mut stuck_before_mark = false;
        for q in queues {
            let line = lines.line(q);
            let mut held = lock(line);
            loop {
                // A queue with a part in `joined` has our mark at the head
                // of its line: what is looked at is behind the mark.
                let base = usize::from(joined.marked.contains(&q));
                let marked = marks && held.iter().skip(base + 1).any(|h| matches!(h, Held::Present(..) | Held::Mark(_)));
                // Waits passed: at a present the old way, or ahead of a
                // queued one when nothing else moves.
                let pass = all || (force_now && marked);
                match held.get(base) {
                    None | Some(Held::Running) => break,
                    Some(&Held::Wait(fence, value)) if !pass && !lines.reached(fence, value) => {
                        stuck_before_mark |= marked;
                        break;
                    }
                    // With parts taken and not sent: only what is known
                    // without a wait for the card. A wait that stays is
                    // looked at again when the parts have been sent.
                    Some(Held::Value(wait)) if !pass && !joined.is_empty() => {
                        // The newest known write first: a part taken, then
                        // the flights, memory last (#2611: memory asked
                        // first passed a wait for 7 that a part taken turns
                        // into 8). What is not known without the card waits.
                        match lines.joined_wait(&joined, wait) {
                            (Some(true), taken) => {
                                if taken {
                                    JOINED[4].fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            _ => break,
                        }
                    }
                    Some(Held::Value(wait)) if !pass && !lines.value_met(wait) => {
                        stuck_before_mark |= marked;
                        break;
                    }
                    // The frame is shown behind the parts taken: they are
                    // sent first.
                    Some(Held::Present(..)) if !joined.is_empty() => break,
                    // A queued present: once every other line has reached
                    // its mark of the frame.
                    Some(&Held::Present(_, _, frame, _)) if marks && !all && ahead_of_mark(lines, q, frame) => break,
                    // Behind a mark: once the frame has been shown.
                    Some(&Held::Mark(frame)) if !all && lines.shown() < frame => break,
                    Some(_) => {}
                }
                match held.remove(base) {
                    Held::Wait(..) | Held::Running | Held::Mark(_) => {}
                    Held::Value(wait) => {
                        // Passed at a present without being met: what the
                        // queue stood at.
                        if pass && !lines.value_met(&wait) {
                            static SEEN: AtomicU64 = AtomicU64::new(0);
                            if SEEN.fetch_add(1, Ordering::Relaxed) < 16 {
                                tracing::warn!(
                                    queue = format_args!("{q:#x}"),
                                    ?wait,
                                    "a wait for a memory value was not met at a present: the queue runs on"
                                );
                            }
                        }
                    }
                    Held::Run(lists, order) => {
                        #[cfg(feature = "diagnostics")]
                        crate::capture::op(q, || {
                            format!(
                                "RUN held submission of {} lists (forced {all})",
                                order.len()
                            )
                        });
                        // Diagnostics: the part behind this one stands at
                        // waits that are met already, so the two could have
                        // been one host submission (one wait for the card,
                        // each range shaders write compared once).
                        #[cfg(feature = "diagnostics")]
                        {
                            let waits = held.iter().take_while(|h| matches!(h, Held::Value(_))).count();
                            if waits > 0
                                && matches!(held.get(waits), Some(Held::Run(..)))
                                && held[..waits].iter().all(|h| matches!(h, Held::Value(w) if w.met()))
                            {
                                crate::copystat::add(crate::copystat::C::PartsTogether, 1);
                            }
                        }
                        if joining {
                            // A part whose preparing reads on the CPU what
                            // a part taken writes goes after them (the
                            // bench's case d2): they are sent first.
                            if lines.joined_blocks(&joined, &lists, &order) {
                                JOINED[5].fetch_add(1, Ordering::Relaxed);
                                drop(held);
                                send_joined(lines, &mut joined);
                                held = lock(line);
                            }
                            // Taken, to go with the parts that follow it.
                            // The line stays open behind our mark.
                            if !joined.marked.contains(&q) {
                                held.insert(0, Held::Running);
                                joined.marked.push(q);
                            }
                            joined.add(lists, order);
                            if joined.full() {
                                drop(held);
                                send_joined(lines, &mut joined);
                                held = lock(line);
                            }
                            moved = true;
                            continue;
                        }
                        held = lines.run(line, held, lists, order, all);
                    }
                    Held::Present(texture, state, frame, resource) => {
                        // (3): read now, behind what was ahead of it.
                        let state = if resource != 0 { lines.present_state(resource) } else { state };
                        held = lines.present(line, held, texture, state, frame, all);
                    }
                    // Behind what was sent to the card and not waited
                    // for, if anything was.
                    Held::Signal(fence, value) => {
                        if !joined.is_empty() {
                            // Behind parts not sent yet: with them.
                            joined.signals.push((fence, value));
                        } else {
                            lines.signal(fence, value);
                        }
                    }
                }
                moved = true;
            }
            if held.is_empty() {
                lines.leave(q);
            }
        }
        if !moved {
            if joined.is_empty() {
                if stuck_before_mark && !force_now {
                    // Nothing else moves: what stands ahead of a queued
                    // present is forced, as the present forced it.
                    if forces() {
                        forcing = true;
                        continue;
                    }
                }
                break;
            }
            // Nothing more is free to run behind the parts taken: they go
            // to the host, and what stood at a wait is looked at again.
            send_joined(lines, &mut joined);
        }
    }
}

/// With [`present_queued`]: whether a line other than `q` still holds
/// something ahead of its mark of `frame` (the present of that frame
/// waits). Called with `q`'s line locked: the other lines are locked one
/// at a time, by the one thread that runs the lines.
#[cfg(windows)]
fn ahead_of_mark<L: Lines>(lines: &L, q: usize, frame: u64) -> bool {
    lines.all_lines().into_iter().filter(|&o| o != q).any(|o| {
        let held = lock(lines.line(o));
        held.iter().position(|h| matches!(h, Held::Mark(f) if *f == frame)).is_some_and(|i| i > 0)
    })
}

/// Whether a queue may pass `wait`: memory holds the value, or, with
/// submissions sent to the card without a wait (`crate::host::late`), it
/// will once those in the air have run.
#[cfg(windows)]
fn value_met(wait: &crate::list::ValueWait) -> bool {
    if !crate::host::late() {
        return wait.met();
    }
    if wait_guarded() {
        let (address, bytes) = wait.place();
        let (flights, unsure) = crate::host::late_last_write(address, bytes);
        return match crate::host::wait_holds_guarded(wait, None, flights, unsure, || wait.met()) {
            Some(holds) => holds,
            None => {
                WAIT_EARLY[2].fetch_add(1, Ordering::Relaxed);
                crate::host::settle_for_value();
                wait.met()
            }
        };
    }
    if wait.met() {
        // Memory meets it: is a write ahead of it going to change that
        // (counted; `VICTORY_WAIT_GUARDED=1` asks those writes first)?
        #[cfg(feature = "diagnostics")]
        {
            let (address, bytes) = wait.place();
            let (flights, unsure) = crate::host::late_last_write(address, bytes);
            if flights.is_some() || unsure {
                WAIT_EARLY[0].fetch_add(1, Ordering::Relaxed);
                if crate::host::wait_holds_guarded(wait, None, flights, unsure, || true) == Some(false) {
                    WAIT_EARLY[1].fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        return true;
    }
    crate::host::late_met(wait)
}

/// Whether a queue's wait for a memory value asks the writes ahead of it
/// before memory (`VICTORY_WAIT_GUARDED=1`; off by default): the last
/// bottom-of-pipe write of the flights over the place decides; one the
/// flights' shaders may make, or one of another kind, settles them first;
/// memory only when nothing ahead writes there. As it is, memory is asked
/// first: a wait for 7 that memory meets passes while a flight that stores
/// 8 is in the air, and the part behind it runs early (upstream #2611's
/// class; the bench's oracle, "SeparateLate"). May turn within a run
/// (`crate::ab`): each wait is looked at one way or the other.
#[cfg(windows)]
fn wait_guarded() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let run = *ON.get_or_init(|| std::env::var_os("VICTORY_WAIT_GUARDED").is_some_and(|v| v == "1"));
    crate::ab::flag(crate::ab::Key::WaitGuarded, run)
}

/// Since the last frame line: waits for a memory value passed because
/// memory met them while a write ahead of them (a flight's bottom-of-pipe
/// write or its shaders) could change that, and of those the ones whose
/// last known write does not meet them (diagnostics, the way as it is);
/// and with `VICTORY_WAIT_GUARDED=1`, the settles such waits asked for.
#[cfg(windows)]
pub(crate) static WAIT_EARLY: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

fn queue<'a>(this: This) -> &'a Queue {
    // SAFETY: called from queue methods only.
    unsafe { &*this.cast::<Queue>() }
}

pub(crate) extern "system" fn create(
    _: This,
    desc: *const QueueDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "command queue");
    let queue = Box::new(Queue {
        object: if crate::device::newer() {
            Object::new(&NEWER_FUNCTIONS)
        } else {
            Object::new(&FUNCTIONS)
        },
        kind: desc.kind,
        desc,
        #[cfg(windows)]
        held: Default::default(),
    });
    object::hand_out(QUEUE, queue, iid, &QUEUE_IIDS, out)
}

/// The console's queue description: an engine or pipe and a queue on it
/// in place of PC's priority and node mask.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDescX {
    kind: u32,
    flags: u32,
    engine_or_pipe: u32,
    queue: u32,
}

/// Device method `CreateCommandQueueX`. The host picks engines itself, so
/// the queue is created as on PC.
pub(crate) extern "system" fn create_x(
    this: This,
    desc: *const QueueDescX,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&x) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(desc = ?x, "console command queue");
    let desc = QueueDesc {
        kind: x.kind,
        priority: 0,
        flags: x.flags,
        nodes: 1,
    };
    create(this, &desc, iid, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUEUE,
        iid,
        object::is(iid, &QUEUE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    let left = object::release_boxed::<Queue>(this);
    #[cfg(windows)]
    if left == 0 {
        lock(&WAITING).retain(|&q| q != this as usize);
    }
    left
}

/// Runs the lists' host lists and waits for them (decision 0007). A queue
/// that waits for a fence keeps them until the fence is reached.
extern "system" fn execute(this: This, count: u32, lists: *const This) {
    #[cfg(windows)]
    let started = std::time::Instant::now();
    execute_now(this, count, lists);
    #[cfg(windows)]
    {
        let us = started.elapsed().as_micros() as u64;
        crate::framelog::add(crate::framelog::T::ExecuteUs, us);
        crate::framelog::here(crate::framelog::H::ExecuteUs, us);
    }
}

fn execute_now(this: This, count: u32, lists: *const This) {
    if lists.is_null() {
        return;
    }
    #[cfg(windows)]
    if let Some(host) = crate::host::host() {
        // SAFETY: the title passes `count` lists.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        if cuts() && !this.is_null() {
            submit_cut(this, lists);
            return;
        }
        if waits() && split() && !this.is_null() {
            submit_split(this, lists);
            return;
        }
        // A list that waits for a memory value not there yet: the queue
        // holds the submission, as it does behind a fence.
        if waits() && !this.is_null() {
            let unmet = crate::list::unmet_waits(lists);
            if !unmet.is_empty() {
                static SEEN: AtomicU64 = AtomicU64::new(0);
                if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
                    tracing::info!(kind = queue(this).kind, ?unmet, "a submission is held for a memory value");
                }
                lock(&queue(this).held).extend(unmet.into_iter().map(Held::Value));
                let mut waiting = lock(&WAITING);
                if !waiting.contains(&(this as usize)) {
                    waiting.push(this as usize);
                }
            }
        }
        let kept = hold(this, || {
            let (taken, order) = crate::list::detach(lists);
            Held::Run(taken, order)
        });
        // Diagnostics: the order submissions are made in, next to the
        // commands of the frame's `ops` file (which are in recording order).
        #[cfg(feature = "diagnostics")]
        crate::capture::op(this as usize, || {
            let kind = (!this.is_null()).then(|| queue(this).kind);
            // The lists as the frame's other lines name them (their host sides).
            let named: Vec<String> = lists
                .iter()
                .map(|&l| match crate::list::host_list(l) {
                    Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                    None => format!("title {l:?}"),
                })
                .collect();
            format!("SUBMIT queue kind {kind:?} held {kept} lists {named:?}")
        });
        if kept {
            return;
        }
        let host_lists: Vec<_> = lists
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.execute(&host_lists);
        crate::list::tables_done(lists);
        // What the lists wrote at the bottom of the pipe may be a fence
        // another queue waits for.
        release_held(false);
        return;
    }
    let _ = this;
    static ONCE: AtomicBool = AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        tracing::warn!(count, "command lists submitted; no host GPU runs them");
    }
}

/// The start of a present plane: frame token, then the plane's resources.
#[repr(C)]
pub(crate) struct Plane {
    token: u64,
    count: u32,
    resources: *const This,
}
/// Submission tagged with the frame it belongs to `(count, lists, frame
/// token, flags)`; frames are paced elsewhere, so it runs as a plain one.
extern "system" fn execute_x(this: This, count: u32, lists: *const This, _token: u64, _flags: u32) {
    execute(this, count, lists);
}
/// Signal with sync settings for the top and bottom of the pipe; our
/// submissions are complete when they return, so it signals at once.
extern "system" fn signal2_x(
    this: This,
    fence: This,
    value: u64,
    _bottom: *const u8,
    _top: *const u8,
) -> HRESULT {
    signal(this, fence, value)
}
/// Markers and events only matter to profilers.
extern "system" fn marker(_: This) {}
extern "system" fn signal(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::fence::pace::signal_called(fence as usize, value);
    // A queue that waits signals after what it holds.
    #[cfg(windows)]
    if hold(this, || Held::Signal(fence as usize, value)) {
        return S_OK;
    }
    // Behind the submissions sent to the card and not waited for: the
    // fence is set when they are settled.
    #[cfg(windows)]
    if crate::host::late_signal(fence as usize, value) {
        return S_OK;
    }
    let _ = this;
    crate::fence::signal(fence, value)
}
/// From here on the queue runs nothing it is given until `fence` reaches
/// `value`: how the console orders work between queues (the title's
/// compute queue waits for the depth the graphics queue draws). Our
/// submissions run when they are made, so a wait for a value already
/// reached is nothing; otherwise what the queue is given next is held
/// ([`Held`]) until a signal or a list's bottom-of-pipe write gets the
/// fence there.
extern "system" fn wait(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if !this.is_null() {
        let now = crate::fence::completed(fence);
        // Whether the title uses this at all (the first calls).
        {
            static CALLS: AtomicU64 = AtomicU64::new(0);
            if CALLS.fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::info!(kind = queue(this).kind, now, value, "a queue is told to wait for a fence");
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::capture::op(this as usize, || format!("WAIT queue kind {} for fence {fence:?} value {value} (now {now})", queue(this).kind));
        let behind = {
            let mut held = lock(&queue(this).held);
            let in_flight = frame_in_flight();
            let behind = !held.is_empty() || now < value || in_flight.is_some();
            if behind && waits() {
                open_line(&mut held, in_flight);
                held.push(Held::Wait(fence as usize, value));
            }
            behind
        };
        if behind {
            static SEEN: AtomicU64 = AtomicU64::new(0);
            if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::info!(
                    kind = queue(this).kind,
                    now,
                    value,
                    held = waits(),
                    "a queue waits for a fence not reached yet"
                );
            }
        }
        if behind && waits() {
            {
                let mut waiting = lock(&WAITING);
                if !waiting.contains(&(this as usize)) {
                    waiting.push(this as usize);
                }
            }
            // The fence may have got there meanwhile.
            release_held(false);
        }
    }
    let _ = (this, value);
    S_OK
}
/// Ticks per second of GPU timestamps (see [`Written::GpuTimestamp`]).
extern "system" fn timestamp_frequency(_: This, out: *mut u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(TIMESTAMP_FREQUENCY) };
    S_OK
}
const TIMESTAMP_FREQUENCY: u64 = 100_000_000;

/// What a list's bottom-of-pipe write stores when the list has run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Written {
    Value(u64),
    /// The GPU clock: [`TIMESTAMP_FREQUENCY`] ticks per second since the
    /// process started (our GPU work is done on the CPU's timeline).
    GpuTimestamp,
    /// The CPU's performance counter.
    SystemTimestamp,
}

impl Written {
    /// Stores the low `bytes` (4 or 8) of the value at `at`.
    pub(crate) fn write(self, at: *mut u8, bytes: usize) {
        let value = match self {
            Written::Value(v) => v,
            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64
            }
            Written::SystemTimestamp => system_timestamp(),
        };
        // SAFETY: the title names writable GPU memory (unified with ours).
        unsafe {
            if bytes == 4 {
                at.cast::<u32>().write_unaligned(value as u32);
            } else {
                at.cast::<u64>().write_unaligned(value);
            }
        }
    }
}

fn system_timestamp() -> u64 {
    #[cfg(windows)]
    {
        let mut count = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
        count as u64
    }
    #[cfg(not(windows))]
    0
}

/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "diagnostics")]
pub(crate) fn presented() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// Shows the first plane's first resource in our window.
///
/// shortcut: one plane, no scaling or placement, the source expected in
/// the common state.
extern "system" fn present(
    this: This,
    planes: u32,
    plane_parameters: *const Plane,
    _parameters: *const u8,
) -> HRESULT {
    #[cfg(windows)]
    let presenting = std::time::Instant::now();
    let hr = present_now(this, planes, plane_parameters);
    #[cfg(windows)]
    {
        if present_queued() > 0 {
            QUEUED_STATS[5].fetch_add(presenting.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        crate::framelog::add(crate::framelog::T::PresentUs, presenting.elapsed().as_micros() as u64);
        crate::framelog::frame();
        #[cfg(feature = "diagnostics")]
        crate::fence::pace::frame();
    }
    hr
}

fn present_now(this: This, planes: u32, plane_parameters: *const Plane) -> HRESULT {
    #[cfg(windows)]
    let presenting = std::time::Instant::now();
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::copystat::present_begins();
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::profile::frame();
    // With `present_behind`: the frame is shown by our thread, behind what
    // the queue holds, and nothing is run or waited for here.
    #[cfg(windows)]
    let sent_behind = (present_behind() && send_behind(this, planes, plane_parameters))
        || (present_queued() > 0 && queue_present(this, planes, plane_parameters));
    #[cfg(not(windows))]
    let _ = this;
    // Submissions sent to the card and not waited for are settled first,
    // and what that frees runs in its own order, before anything is forced.
    #[cfg(windows)]
    if sent_behind {
    } else if crate::host::late() && !crate::host::late_free_present() {
        // (With `behind`: what the queues hold runs here first, and what
        // it sent is settled with the rest.)
        if behind() {
            release_held_now(false);
        }
        crate::host::settle_for_present();
        release_held_now(false);
        crate::host::settle_for_present();
    } else if crate::host::late() {
        // The present is sent ahead too: what waits for a value a flight
        // will write runs behind that flight, and nothing is settled here.
        release_held_now(false);
    }
    // shortcut: what a queue still holds at a present runs now, whatever
    // its fence: a wait is not kept across frames.
    #[cfg(windows)]
    if !sent_behind {
        release_held(forces());
    }
    #[cfg(windows)]
    crate::framelog::add(crate::framelog::T::PresentLineUs, presenting.elapsed().as_micros() as u64);
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    #[cfg(windows)]
    if present_queued() > 0 {
        if !sent_behind {
            QUEUED_STATS[1].fetch_add(1, Ordering::Relaxed);
        }
        if frame.is_multiple_of(256) {
            present_line(256);
        }
    }
    #[cfg(windows)]
    if sent_behind && present_queued() == 0 && PRESENTS_BEHIND.load(Ordering::Relaxed).is_multiple_of(512) {
        tracing::info!(
            frame,
            behind = PRESENTS_BEHIND.load(Ordering::Relaxed),
            "presents sent behind their queue's line so far (VICTORY_ASYNC_PRESENT=1); the others ran the old way"
        );
    }
    if frame.is_power_of_two() {
        let (draws, dispatches) = crate::list::take_counts();
        #[cfg(windows)]
        let (drawn, dispatched) = {
            static DRAWN: AtomicU64 = AtomicU64::new(0);
            static DISPATCHED: AtomicU64 = AtomicU64::new(0);
            let since = |last: &AtomicU64, now: u64| now - last.swap(now, Ordering::Relaxed);
            (
                since(&DRAWN, crate::draw::drawn()),
                since(&DISPATCHED, crate::draw::dispatched()),
            )
        };
        #[cfg(not(windows))]
        let (drawn, dispatched) = (0, 0);
        tracing::info!(
            frame,
            draws,
            drawn,
            dispatches,
            dispatched,
            "frame presented (since the last line: draws and dispatches, and those run on the host)"
        );
        #[cfg(windows)]
        if frame >= 256 {
            let (hits, misses) = crate::shader_cache::counts();
            tracing::info!(hits, misses, "shaders from the cache and translated so far");
        }
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    {
        crate::capture::frame_drew(frame, crate::draw::drawn());
        if crate::capture::listed(frame) {
            crate::capture::dump_targets(frame);
        }
    }
    #[cfg(windows)]
    if !sent_behind
        && planes > 0
        // SAFETY: the title passes `planes` plane descriptions.
        && let Some(plane) = unsafe { plane_parameters.as_ref() }
        && plane.count > 0
        && !plane.resources.is_null()
        // SAFETY: the plane lists `count` resources.
        && let resource = unsafe { *plane.resources }
        && let Some(texture) = crate::resource::host_resource(resource)
        && let Some(host) = crate::host::host()
    {
        const COPY_SOURCE: u32 = 0x800;
        let state = crate::resource::host_transition(resource, COPY_SOURCE)
            .map_or(COPY_SOURCE, |(_, before)| before);
        let shown = host.present(texture, state, frame);
        crate::resource::host_transition(resource, state);
        #[cfg(feature = "diagnostics")]
        blinks(frame, resource, state, shown);
        #[cfg(not(feature = "diagnostics"))]
        let _ = shown;
    }
    #[cfg(not(windows))]
    let _ = (planes, plane_parameters);
    S_OK
}
/// Diagnostics: logs a presented frame that is black after one that was
/// not, with what was presented and how many draws ran since.
#[cfg(all(windows, feature = "diagnostics"))]
fn blinks(frame: u64, resource: This, state: u32, shown: Option<bool>) {
    static LAST: std::sync::Mutex<(bool, u64, u64)> = std::sync::Mutex::new((false, 0, 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let drawn = crate::draw::drawn();
    let black = shown.unwrap_or(false);
    let draws = crate::draw::take_frame();
    if black && !last.0 {
        last.2 += 1;
        tracing::warn!(
            frame,
            resource = ?resource,
            state = format_args!("{state:#x}"),
            draws = drawn - last.1,
            copied = shown.is_some(),
            black_frames = last.2,
            levels = ?crate::capture::recent_levels(),
            "a black frame after a frame that was not"
        );
        for d in draws {
            tracing::warn!("black frame {frame}: {d}");
        }
        AFTER.store(frame + 6, Ordering::Relaxed);
    }
    // Six frames on: the levels around it (a fade or a single drop).
    static AFTER: AtomicU64 = AtomicU64::new(0);
    if AFTER.load(Ordering::Relaxed) == frame {
        tracing::warn!(frame, levels = ?crate::capture::recent_levels(), "levels after the black frame");
    }
    *last = (black, drawn, last.2);
}

/// A run of destination pages: how they are mapped (0 pool pages from
/// `start` on, 1 pool page `start` each, 2 unmapped, 3 left as they are)
/// and how many.
#[repr(C)]
pub(crate) struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

/// The ranges mapped from a destination address on.
#[repr(C)]
pub(crate) struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

/// Bytes per mapped page.
const PAGE: u64 = 64 << 10;

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. The title streams textures into pool pages
/// and maps them into its tiled heap this way (title+0xf2c470, ranges of
/// one constant page or of unmapped pages). Done at once: every earlier
/// submission has run.
extern "system" fn copy_page_mappings(
    _: This,
    count: u32,
    batches: *const PageBatch,
    pool: u64,
    pool_pages: u32,
    _flags: u32,
) -> HRESULT {
    let Some(map) = crate::resource::page_mapper() else {
        return E_INVALIDARG;
    };
    if batches.is_null() && count > 0 {
        return E_INVALIDARG;
    }
    let batches = (count > 0).then(|| {
        // SAFETY: the title passes `count` batches.
        unsafe { std::slice::from_raw_parts(batches, count as usize) }
    });
    let mut ok = true;
    for b in batches.into_iter().flatten() {
        if b.ranges.is_null() {
            continue;
        }
        // SAFETY: each batch holds `count` ranges.
        let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
        let mut at = b.destination;
        for r in ranges {
            let len = u64::from(r.count) * PAGE;
            let inside = |pages: u32| r.start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = pool + u64::from(r.start) * PAGE;
            ok &= match r.kind {
                0 if inside(r.count) => map(at, source, len),
                1 if inside(1) => (0..u64::from(r.count)).all(|k| map(at + k * PAGE, source, PAGE)),
                2 => map(at, 0, len),
                3 => true,
                _ => false,
            };
            at += len;
        }
    }
    if !ok {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(pool = format_args!("{pool:#x}"), "a page mapping failed"));
    }
    S_OK
}

extern "system" fn desc(this: This, out: *mut QueueDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(queue(this).desc) };
    }
}

/// The queue methods by their slot in older titles' queues (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, queue 26 (12 as on PC,
/// then console ones).
const METHODS: [(usize, *const ()); 15] = [
    (0, query as *const ()),
    (2, release as *const ()),
    (7, object::parent_device as *const ()), // 0x38
    (10, execute as *const ()),              // 0x50
    (11, marker as *const ()),               // 0x58 set marker
    (12, marker as *const ()),               // 0x60 begin event
    (13, marker as *const ()),               // 0x68 end event
    (14, signal as *const ()),               // 0x70
    (15, wait as *const ()),                 // 0x78
    (16, timestamp_frequency as *const ()),  // 0x80
    (26, desc as *const ()),                 // 0xd0
    (29, copy_page_mappings as *const ()),   // 0xe8 page mapping batch
    (30, present as *const ()),              // 0xf0
    (33, execute_x as *const ()),            // 0x108
    (32, signal2_x as *const ()),            // 0x100
];

/// Newer titles' queues have two more methods before Signal, which moved
/// from 0x70 to 0x80: every slot from 0x70 on is two later (taken here as
/// starting exactly at 0x70, the methods before it kept in place).
/// Inferred from the title's calls: Signal 0x80 `(fence,
/// value)`, timestamp frequency 0x90 (title+0x24841a6), suspend and resume
/// 0xb8/0xc0, PresentX 0x100, the frame-token submission 0x118 `(count,
/// lists, token, flags)` (title+0x24821dd).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 14 { slot + 2 } else { slot }
}

const fn queue_table(newer: bool) -> [*const (); 36] {
    let mut f = object::d3d12_object(object::missing_table::<QUEUE, 36>());
    let mut i = 0;
    while i < METHODS.len() {
        let (slot, method) = METHODS[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    f
}

static FUNCTIONS: Table<36> = Table(queue_table(false));
static NEWER_FUNCTIONS: Table<36> = Table(queue_table(true));

#[cfg(test)]
mod tests {
    use super::*;

    /// Parts joined keep each its lists in its own order, one part after
    /// the other (fork fxceless 39731795d's test).
    #[cfg(windows)]
    #[test]
    fn joined_parts_keep_their_lists_and_their_order() {
        // a part of three lists run as 2, 0, 1; then one of two run as 1, 0; then one of one
        let (lists, order) = join(vec!["a0", "a1", "a2"], vec![2, 0, 1], vec!["b0", "b1"], vec![1, 0]);
        let (lists, order) = join(lists, order, vec!["c0"], vec![0]);
        assert_eq!(lists, ["a0", "a1", "a2", "b0", "b1", "c0"]);
        let run: Vec<&str> = order.iter().map(|&i| lists[i]).collect();
        assert_eq!(run, ["a2", "a0", "a1", "b1", "b0", "c0"], "each part's lists in its own order, the parts one after the other");
        // a part with a list named twice, and an empty one
        let (lists, order) = join(vec!["a0"], vec![0, 0], Vec::new(), Vec::new());
        let (lists, order) = join(lists, order, vec!["b0"], vec![0, 0]);
        assert_eq!(order.iter().map(|&i| lists[i]).collect::<Vec<_>>(), ["a0", "a0", "b0", "b0"]);
    }
    use crate::object::slot;

    /// Where a queue's signal is: newer layouts have it two slots later.
    /// Another test may switch layouts while this one runs, so the queue's
    /// own table decides.
    fn signal_slot(q: This) -> usize {
        // SAFETY: one of our queues; its first field is the table.
        let table = unsafe { *q.cast::<*const *const ()>() };
        if table == NEWER_FUNCTIONS.0.as_ptr() {
            0x80
        } else {
            0x70
        }
    }

    #[test]
    fn queue_signals_complete_at_once() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        // SAFETY: the title reads the type at +0x10.
        assert_eq!(unsafe { q.cast::<u8>().add(0x10).cast::<u32>().read() }, 0);
        let mut fence = null;
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(
            crate::fence::create(null, 0, 0, fence_iid, &mut fence),
            S_OK
        );
        let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(signal(q, fence, 7), S_OK);
        assert_eq!(
            slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence),
            7
        );
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// The title's compute queue: `{COMPUTE, 0, pipe 1, queue 0}`.
    #[test]
    fn console_queues_are_created_as_on_pc() {
        let null = std::ptr::null();
        let desc = QueueDescX {
            kind: 2,
            flags: 0,
            engine_or_pipe: 1,
            queue: 0,
        };
        let mut q = null;
        assert_eq!(create_x(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        assert_eq!(queue(q).desc.kind, 2);
        let none = std::ptr::null();
        assert_eq!(create_x(null, none, &QUEUE_IIDS[0], &mut q), E_INVALIDARG);
    }

    #[test]
    fn newer_queues_have_their_methods_two_slots_later() {
        let (old, new) = (queue_table(false), queue_table(true));
        assert_eq!(old[10], new[10]); // submission stays at 0x50
        assert_eq!(old[14], new[16]); // signal 0x70 → 0x80
        assert_eq!(old[30], new[32]); // present 0xf0 → 0x100
        assert_eq!(new[35], execute_x as *const ()); // 0x118
    }

    #[test]
    fn timestamps_and_values_are_written() {
        let mut at = [0u64; 2];
        Written::Value(0x1_0000_0002).write(at.as_mut_ptr().cast(), 4);
        assert_eq!(at[0], 2);
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        let first = at[1];
        std::thread::sleep(std::time::Duration::from_millis(2));
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        assert!(at[1] > first);
    }
}

/// [`walk`] on lines of its own: what leaves them, in which order, with
/// and without parts joined ([`join_parts`]). Parts carry no lists here:
/// a part's one place in `order` is its name, and joining keeps it
/// (`join` moves places up by the lists before, none here).
#[cfg(all(test, windows))]
mod join_tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::sync::Mutex;

    #[derive(Debug, Clone, PartialEq)]
    enum Done {
        /// One host submission sent ahead, its parts in order.
        Sent(Vec<usize>),
        /// A part not joined ([`Lines::run`]).
        Ran(Vec<usize>),
        Signal(usize, u64),
        /// A frame shown ([`Lines::present`]).
        Present(u64),
    }

    #[derive(Default)]
    struct Fake {
        /// The last frame shown.
        shown: std::sync::atomic::AtomicU64,
        lines: Vec<Mutex<Vec<Held>>>,
        waiting: Mutex<Vec<usize>>,
        fences: Mutex<HashMap<usize, u64>>,
        /// Values (by address) memory holds now.
        met: Mutex<HashSet<usize>>,
        /// Values (by address) a bottom-of-pipe write of a part (its name) stores.
        writes: HashMap<usize, usize>,
        /// Parts sent: in the air, their writes known.
        sent: Mutex<HashSet<usize>>,
        /// Parts that read on the CPU what a part taken before them writes
        /// (`crate::host::join_blocked`): they do not join it.
        blocked: HashSet<usize>,
        /// Host states of resources (by address), as the submissions ahead
        /// leave them: a part named in `transitions` sets one when it runs.
        states: Mutex<HashMap<usize, u32>>,
        transitions: HashMap<usize, (usize, u32)>,
        /// The state each frame was shown with.
        shown_states: Mutex<Vec<(u64, u32)>>,
        done: Mutex<Vec<Done>>,
    }

    fn part(name: usize) -> Held {
        Held::Run(Vec::new(), vec![name])
    }

    fn value(address: usize) -> Held {
        Held::Value(crate::list::ValueWait::equal(address as u64, 1))
    }

    impl Fake {
        fn new(lines: Vec<Vec<Held>>) -> Self {
            let waiting = (0..lines.len()).filter(|&q| !lines[q].is_empty()).collect();
            Self { lines: lines.into_iter().map(Mutex::new).collect(), waiting: Mutex::new(waiting), ..Default::default() }
        }

        fn known(&self, address: usize) -> bool {
            lock(&self.met).contains(&address) || self.writes.get(&address).is_some_and(|p| lock(&self.sent).contains(p))
        }

        fn done(&self) -> Vec<Done> {
            lock(&self.done).clone()
        }
    }

    impl Lines for Fake {
        fn waiting(&self) -> Vec<usize> {
            lock(&self.waiting).clone()
        }
        fn leave(&self, q: usize) {
            lock(&self.waiting).retain(|&w| w != q);
        }
        fn all_lines(&self) -> Vec<usize> {
            lock(&self.waiting).clone()
        }
        fn join_waiting(&self, q: usize) {
            let mut waiting = lock(&self.waiting);
            if !waiting.contains(&q) {
                waiting.push(q);
            }
        }
        fn shown(&self) -> u64 {
            self.shown.load(std::sync::atomic::Ordering::Relaxed)
        }
        fn line(&self, q: usize) -> &Mutex<Vec<Held>> {
            &self.lines[q]
        }
        fn reached(&self, fence: usize, value: u64) -> bool {
            lock(&self.fences).get(&fence).copied().unwrap_or(0) >= value
        }
        fn value_met(&self, wait: &crate::list::ValueWait) -> bool {
            self.known(wait.place().0)
        }
        fn joined_wait(&self, joined: &Joined, wait: &crate::list::ValueWait) -> (Option<bool>, bool) {
            let address = wait.place().0;
            if self.writes.get(&address).is_some_and(|p| joined.order.contains(p)) {
                return (Some(true), true);
            }
            (Some(self.known(address)), false)
        }
        fn joined_blocks(&self, _: &Joined, _: &[crate::list::Detached], order: &[usize]) -> bool {
            order.iter().any(|p| self.blocked.contains(p))
        }
        fn send(&self, _: Vec<crate::list::Detached>, order: Vec<usize>) {
            lock(&self.done).push(Done::Sent(order.clone()));
            lock(&self.sent).extend(order);
        }
        fn signal(&self, fence: usize, value: u64) {
            lock(&self.done).push(Done::Signal(fence, value));
            let mut fences = lock(&self.fences);
            let at = fences.entry(fence).or_insert(0);
            *at = (*at).max(value);
        }
        fn run<'a>(
            &self,
            _: &'a Mutex<Vec<Held>>,
            held: std::sync::MutexGuard<'a, Vec<Held>>,
            _: Vec<crate::list::Detached>,
            order: Vec<usize>,
            _: bool,
        ) -> std::sync::MutexGuard<'a, Vec<Held>> {
            lock(&self.done).push(Done::Ran(order.clone()));
            for p in &order {
                if let Some(&(resource, state)) = self.transitions.get(p) {
                    lock(&self.states).insert(resource, state);
                }
            }
            lock(&self.sent).extend(order);
            held
        }
        fn present_state(&self, resource: usize) -> u32 {
            lock(&self.states).get(&resource).copied().unwrap_or(0x800)
        }
        fn present<'a>(
            &self,
            _: &'a Mutex<Vec<Held>>,
            held: std::sync::MutexGuard<'a, Vec<Held>>,
            _: Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
            state: u32,
            frame: u64,
            _: bool,
        ) -> std::sync::MutexGuard<'a, Vec<Held>> {
            lock(&self.done).push(Done::Present(frame));
            lock(&self.shown_states).push((frame, state));
            self.shown.fetch_max(frame, std::sync::atomic::Ordering::Relaxed);
            held
        }
    }

    const F: usize = 0xF0;
    const G: usize = 0xF8;

    /// The bench's empty_queue_leak: a queue whose line was empty when the
    /// present was queued got no mark, and what the title gave it after the
    /// present ran before the frame was shown. Opened behind the frame in
    /// flight ([`open_line`]), it runs after.
    #[test]
    fn work_given_to_an_empty_line_after_a_queued_present_runs_after_the_frame() {
        let fake = Fake::new(vec![vec![Held::Wait(F, 1), part(1)], vec![]]);
        queue_present_into(&fake, 0, None, 0, 1);
        assert!(lock(&fake.lines[1]).is_empty(), "no mark in an empty line");
        // The title gives queue 1 a part after the present returned.
        {
            let mut held = lock(&fake.lines[1]);
            assert!(open_line(&mut held, Some(1)));
            held.push(part(2));
        }
        fake.join_waiting(1);
        walk(&fake, false, false, true);
        // Nothing may run part 2 before frame 1 is shown.
        let done = fake.done();
        let ran = done.iter().position(|d| *d == Done::Ran(vec![2]));
        let shown = done.iter().position(|d| *d == Done::Present(1));
        assert!(ran.is_none() || shown.is_some_and(|s| s < ran.unwrap()), "{done:?}");
        // The fence comes: part 1, the frame, then part 2.
        lock(&fake.fences).insert(F, 1);
        walk(&fake, false, false, true);
        let done = fake.done();
        assert_eq!(done, [Done::Ran(vec![1]), Done::Present(1), Done::Ran(vec![2])]);
        // A line that is not empty, or no frame in flight: nothing to open.
        let mut held = vec![part(3)];
        assert!(!open_line(&mut held, Some(1)));
        assert!(!open_line(&mut Vec::new(), None));
    }

    /// The bench's stale present state: the texture's host state is what the
    /// submissions ahead of the present leave it in, read when the frame is
    /// shown (0x800 here); read when the present was queued it was the
    /// state before them (0x4).
    #[test]
    fn a_queued_present_shows_the_state_the_submissions_ahead_left() {
        const TEXTURE: usize = 0xABC0;
        let mut fake = Fake::new(vec![vec![part(1)]]);
        lock(&fake.states).insert(TEXTURE, 0x4);
        fake.transitions.insert(1, (TEXTURE, 0x800));
        queue_present_into(&fake, 0, None, TEXTURE, 1);
        assert_eq!(fake.present_state(TEXTURE), 0x4, "as the queue's call found it");
        walk(&fake, false, false, true);
        assert_eq!(fake.done(), [Done::Ran(vec![1]), Done::Present(1)]);
        assert_eq!(lock(&fake.shown_states).clone(), [(1, 0x800)]);
    }

    /// JOIN_PARTS' guards hold in front of a queued present too: a part that
    /// reads what a part taken writes goes after it, and both run before
    /// the frame is shown.
    #[test]
    fn parts_join_or_not_by_the_guards_before_a_queued_present() {
        let mut fake = Fake::new(vec![vec![part(1), part(2), part(3)]]);
        fake.blocked.insert(2);
        queue_present_into(&fake, 0, None, 0, 1);
        walk(&fake, false, true, true);
        assert_eq!(fake.done(), [Done::Sent(vec![1]), Done::Sent(vec![2, 3]), Done::Present(1)]);
    }

    #[test]
    fn a_wait_a_part_taken_meets_joins_and_signals_follow_the_parts_they_followed() {
        // queue 0: part 1, a wait part 1's bottom-of-pipe write meets, part 2, signal F=1
        // queue 1: waits for F=1, part 3
        let mut fake = Fake::new(vec![vec![part(1), value(0x1000), part(2), Held::Signal(F, 1)], vec![Held::Wait(F, 1), part(3)]]);
        fake.writes.insert(0x1000, 1);
        walk(&fake, false, true, false);
        assert_eq!(fake.done(), [Done::Sent(vec![1, 2]), Done::Signal(F, 1), Done::Sent(vec![3])]);
        assert!(fake.lines.iter().all(|l| lock(l).is_empty()), "every line ran out");
        assert!(fake.waiting().is_empty());
        // Without joining: the same parts in the same order, one host submission each.
        let mut fake = Fake::new(vec![vec![part(1), value(0x1000), part(2), Held::Signal(F, 1)], vec![Held::Wait(F, 1), part(3)]]);
        fake.writes.insert(0x1000, 1);
        walk(&fake, false, false, false);
        assert_eq!(fake.done(), [Done::Ran(vec![1]), Done::Ran(vec![2]), Done::Signal(F, 1), Done::Ran(vec![3])]);
    }

    #[test]
    fn a_wait_not_known_to_be_met_sends_what_was_taken_and_holds_the_rest() {
        let fake = Fake::new(vec![vec![part(1), value(0x2000), part(2), Held::Signal(F, 1)]]);
        walk(&fake, false, true, false);
        assert_eq!(fake.done(), [Done::Sent(vec![1])]);
        // The line stands at the wait, with nothing of ours at its head.
        {
            let held = lock(&fake.lines[0]);
            assert_eq!(held.len(), 3);
            assert!(matches!(held[0], Held::Value(_)));
        }
        assert_eq!(fake.waiting(), [0], "still in line");
        // Memory comes to meet it.
        lock(&fake.met).insert(0x2000);
        walk(&fake, false, true, false);
        assert_eq!(fake.done(), [Done::Sent(vec![1]), Done::Sent(vec![2]), Done::Signal(F, 1)]);
        assert!(fake.waiting().is_empty());
    }

    #[test]
    fn a_signal_another_queue_waits_for_is_given_after_its_part_and_never_joins_across() {
        let fake = Fake::new(vec![vec![part(1), Held::Signal(F, 1), part(2)], vec![Held::Wait(F, 1), part(3)]]);
        walk(&fake, false, true, false);
        // Part 2 follows part 1 on its own queue; part 3 waits for the signal
        // behind part 1, which is only given once part 1 has been sent.
        assert_eq!(fake.done(), [Done::Sent(vec![1, 2]), Done::Signal(F, 1), Done::Sent(vec![3])]);
    }

    #[test]
    fn a_fence_reached_and_a_value_in_memory_let_parts_of_two_queues_join() {
        let fake = Fake::new(vec![vec![Held::Wait(G, 5), part(1)], vec![value(0x3000), part(2)]]);
        lock(&fake.fences).insert(G, 7);
        lock(&fake.met).insert(0x3000);
        walk(&fake, false, true, false);
        assert_eq!(fake.done(), [Done::Sent(vec![1, 2])]);
    }

    #[test]
    fn at_most_join_most_parts_go_together() {
        let n = join_most() + 1;
        let fake = Fake::new(vec![(1..=n).map(part).collect()]);
        walk(&fake, false, true, false);
        assert_eq!(fake.done(), [Done::Sent((1..n).collect()), Done::Sent(vec![n])]);
    }

    #[test]
    fn forced_lines_run_every_part_alone_in_order() {
        let fake = Fake::new(vec![vec![part(1), value(0x4000), part(2), Held::Signal(F, 1)], vec![Held::Wait(G, 3), part(3)]]);
        walk(&fake, true, false, false);
        assert_eq!(fake.done(), [Done::Ran(vec![1]), Done::Ran(vec![2]), Done::Signal(F, 1), Done::Ran(vec![3])]);
    }

    /// One thing the title gives a queue, for the present tests.
    #[derive(Clone, Copy, Debug)]
    enum Given {
        Part(usize, usize),
        Signal(usize, usize, u64),
        WaitFence(usize, usize, u64),
        /// Waits for the value part `.1` writes.
        WaitPart(usize, usize),
        /// Waits for a value nothing writes: forced at the next present.
        WaitNever(usize, usize),
        /// The title presents (queue 0).
        Present,
    }

    fn held_of(g: Given) -> (usize, Held) {
        match g {
            Given::Part(q, p) => (q, part(p)),
            Given::Signal(q, f, v) => (q, Held::Signal(f, v)),
            Given::WaitFence(q, f, v) => (q, Held::Wait(f, v)),
            Given::WaitPart(q, by) => (q, value(0x10_0000 + 0x100 * by)),
            Given::WaitNever(q, n) => (q, value(0x20_0000 + 0x100 * n)),
            Given::Present => unreachable!(),
        }
    }

    /// `n` things for `queues` queues (seeded), each waiting only for
    /// things given before it, with a present now and then.
    fn given(seed: u64, queues: usize, n: usize) -> (Vec<Given>, HashMap<usize, usize>) {
        let mut seed = seed;
        let mut next = move |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % m) as usize
        };
        let mut out = Vec::new();
        let mut writes = HashMap::new();
        let (mut parts, mut nevers) = (0, 0);
        let mut signals: Vec<(usize, u64)> = Vec::new();
        let mut fence_values: HashMap<usize, u64> = HashMap::new();
        for _ in 0..n {
            let q = next(queues as u64);
            match next(12) {
                0..=4 => {
                    parts += 1;
                    out.push(Given::Part(q, parts));
                }
                5 if parts > 0 => {
                    let fence = 0x100 + next(3);
                    let v = fence_values.entry(fence).or_insert(0);
                    *v += 1;
                    out.push(Given::Signal(q, fence, *v));
                    signals.push((fence, *v));
                }
                6 if !signals.is_empty() => {
                    let (fence, v) = signals[next(signals.len() as u64)];
                    out.push(Given::WaitFence(q, fence, v));
                }
                7 | 8 if parts > 0 => {
                    let by = 1 + next(parts as u64);
                    writes.insert(0x10_0000 + 0x100 * by, by);
                    out.push(Given::WaitPart(q, by));
                }
                9 => {
                    nevers += 1;
                    out.push(Given::WaitNever(q, nevers));
                }
                10 | 11 => out.push(Given::Present),
                _ => {}
            }
        }
        (out, writes)
    }

    fn fake_lines(queues: usize, writes: &HashMap<usize, usize>) -> Fake {
        let mut fake = Fake::new((0..queues).map(|_| Vec::new()).collect());
        fake.writes = writes.clone();
        fake
    }

    fn give(fake: &Fake, g: Given) {
        let (q, h) = held_of(g);
        lock(&fake.lines[q]).push(h);
        fake.join_waiting(q);
    }

    /// 10,000 things given in rounds, the lines run after each (as our
    /// thread does when it is woken): a present queued into the lines
    /// ([`present_queued`]) gives the same runs, signals and frames shown,
    /// in the same order, as the present that ran the lines itself (free
    /// first, then forced, then shown), and the same swap-chain sequence.
    #[test]
    fn a_queued_present_runs_and_shows_what_the_present_ran_and_showed() {
        let mut total = 0;
        let mut presents = 0;
        for round in 0..110u64 {
            let queues = 1 + (round % 4) as usize;
            let (given, writes) = given(0x9E37_79B9_7F4A_7C15 ^ round.wrapping_mul(0x1000_0001), queues, 100);
            total += given.len();
            // The present runs the lines itself.
            let old = fake_lines(queues, &writes);
            let mut frame = 0;
            for &g in &given {
                if let Given::Present = g {
                    frame += 1;
                    walk(&old, false, false, false);
                    walk(&old, true, false, false);
                    lock(&old.done).push(Done::Present(frame));
                    old.shown.store(frame, std::sync::atomic::Ordering::Relaxed);
                } else {
                    give(&old, g);
                    walk(&old, false, false, false);
                }
            }
            walk(&old, false, false, false);
            walk(&old, true, false, false);
            // Queued.
            let new = fake_lines(queues, &writes);
            let mut frame = 0;
            for &g in &given {
                if let Given::Present = g {
                    frame += 1;
                    queue_present_into(&new, 0, None, 0, frame);
                } else {
                    give(&new, g);
                }
                walk(&new, false, false, true);
            }
            walk(&new, false, false, true);
            walk(&new, true, false, true);
            presents += frame;
            assert_eq!(new.shown(), frame, "round {round}: every frame shown");
            assert_eq!(old.done(), new.done(), "round {round}: queued presents changed the order");
            assert!(new.lines.iter().all(|l| lock(l).is_empty()), "round {round}: every line ran out");
        }
        assert!(total >= 10_000 && presents > 1000, "{total} things, {presents} presents");
    }

    /// With the title ahead (our thread runs the lines only now and then,
    /// with parts joined or not): everything given before a present runs
    /// before its frame is shown, what the present's queue is given after
    /// it runs after, frames are shown in order and each once, every part
    /// runs once in its line's order, and at most `most` presents wait in
    /// the lines (the title stands at the bound until the oldest is shown,
    /// which always comes: no deadlock).
    #[test]
    fn queued_presents_keep_the_frame_order_and_the_bound() {
        let mut seed = 0x1234_5678_9ABC_DEF1u64;
        let mut coin = move |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % m
        };
        for round in 0..200u64 {
            let queues = 1 + (round % 3) as usize;
            let most = 1 + round % 2;
            let joining = round % 4 >= 2;
            let (given, writes) = given(0xC0FF_EE00 ^ round.wrapping_mul(0x9E37_79B9), queues, 60);
            let fake = fake_lines(queues, &writes);
            let mut frame = 0u64;
            // Per part: the frames queued before it was given.
            let mut epoch: HashMap<usize, (usize, u64)> = HashMap::new();
            for &g in &given {
                if let Given::Present = g {
                    // The bound: the title waits while `most` wait.
                    let mut looks = 0;
                    while frame - fake.shown() >= most {
                        walk(&fake, false, joining, true);
                        looks += 1;
                        assert!(looks < 64, "round {round}: the title stood at the bound for good: {:?}", fake.done());
                    }
                    frame += 1;
                    queue_present_into(&fake, 0, None, 0, frame);
                } else {
                    if let Given::Part(q, p) = g {
                        epoch.insert(p, (q, frame));
                    }
                    give(&fake, g);
                }
                assert!(frame - fake.shown() <= most, "round {round}: more presents waiting than the bound");
                if coin(3) == 0 {
                    walk(&fake, false, joining, true);
                }
            }
            walk(&fake, false, joining, true);
            assert_eq!(fake.shown(), frame, "round {round}: every frame shown: {:?}", fake.done());
            walk(&fake, true, joining, true);
            let done = fake.done();
            let shown: Vec<u64> = done.iter().filter_map(|d| if let Done::Present(f) = d { Some(*f) } else { None }).collect();
            assert_eq!(shown, (1..=frame).collect::<Vec<_>>(), "round {round}: frames shown in order, each once");
            // Where each part ran, and how many frames had been shown then.
            let mut ran: HashMap<usize, ((usize, usize), u64)> = HashMap::new();
            let mut shown_now = 0u64;
            for (e, d) in done.iter().enumerate() {
                match d {
                    Done::Present(f) => shown_now = *f,
                    Done::Sent(ps) | Done::Ran(ps) => {
                        for (i, &p) in ps.iter().enumerate() {
                            assert!(ran.insert(p, ((e, i), shown_now)).is_none(), "round {round}: part {p} ran twice");
                        }
                    }
                    Done::Signal(..) => {}
                }
            }
            assert_eq!(ran.len(), epoch.len(), "round {round}: every part ran");
            for (p, &(q, before)) in &epoch {
                let (_, shown_then) = ran[p];
                // Given after `before` frames were queued: it ran before frame
                // before + 1 was shown ...
                assert!(shown_then <= before, "round {round}: part {p} (given after frame {before}) ran after frame {shown_then} was shown");
                // ... and in the present's queue, after frame `before`.
                if q == 0 {
                    assert!(shown_then >= before, "round {round}: part {p} of the present's queue ran before frame {before} was shown");
                }
            }
            // Each line's parts in its order.
            for q in 0..queues {
                let order: Vec<usize> = given
                    .iter()
                    .filter_map(|g| match *g {
                        Given::Part(gq, p) if gq == q => Some(p),
                        _ => None,
                    })
                    .collect();
                assert!(order.windows(2).all(|w| ran[&w[0]].0 < ran[&w[1]].0), "round {round}: queue {q} out of order");
            }
        }
    }

    /// Lines made at random, each thing waiting only for things made
    /// before it; with parts joined and without: every part is sent once,
    /// a queue's parts in its order, a signal after the parts before it, a
    /// part behind a fence wait after the signal, a part behind a value
    /// wait not ahead of the part whose write meets it.
    #[test]
    fn joined_or_not_every_line_keeps_its_order_and_its_waits() {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n) as usize
        };
        for round in 0..400 {
            let queues = 1 + next(4);
            let mut writes = HashMap::new();
            // What each queue's line holds, for the checks.
            #[derive(Clone, Copy)]
            enum Made {
                Part(usize),
                Signal(usize, u64),
                WaitFence(usize, u64),
                WaitPart(usize),
            }
            let mut made: Vec<Vec<Made>> = vec![Vec::new(); queues];
            let mut parts = 0;
            let mut signals: Vec<(usize, u64)> = Vec::new();
            let mut fence_values: HashMap<usize, u64> = HashMap::new();
            for _ in 0..(4 + next(20)) {
                let q = next(queues as u64);
                match next(5) {
                    0 | 1 => {
                        parts += 1;
                        made[q].push(Made::Part(parts));
                    }
                    2 if parts > 0 => {
                        let fence = 0x100 + next(3);
                        let v = fence_values.entry(fence).or_insert(0);
                        *v += 1;
                        made[q].push(Made::Signal(fence, *v));
                        signals.push((fence, *v));
                    }
                    3 if !signals.is_empty() => {
                        let (fence, v) = signals[next(signals.len() as u64)];
                        made[q].push(Made::WaitFence(fence, v));
                    }
                    4 if parts > 0 => {
                        let by = 1 + next(parts as u64);
                        let address = 0x10_0000 + 0x100 * by;
                        writes.insert(address, by);
                        made[q].push(Made::WaitPart(by));
                    }
                    _ => {}
                }
            }
            for joining in [true, false] {
                let lines: Vec<Vec<Held>> = made
                    .iter()
                    .map(|m| {
                        m.iter()
                            .map(|x| match *x {
                                Made::Part(p) => part(p),
                                Made::Signal(f, v) => Held::Signal(f, v),
                                Made::WaitFence(f, v) => Held::Wait(f, v),
                                Made::WaitPart(p) => value(0x10_0000 + 0x100 * p),
                            })
                            .collect()
                    })
                    .collect();
                let mut fake = Fake::new(lines);
                fake.writes = writes.clone();
                walk(&fake, false, joining, false);
                let done = fake.done();
                // Where each part went: (event, place in it).
                let mut at: HashMap<usize, (usize, usize)> = HashMap::new();
                let mut signal_at: HashMap<(usize, u64), usize> = HashMap::new();
                for (e, d) in done.iter().enumerate() {
                    match d {
                        Done::Sent(ps) | Done::Ran(ps) => {
                            assert_eq!(matches!(d, Done::Sent(_)), joining, "round {round}");
                            if !joining {
                                assert_eq!(ps.len(), 1);
                            }
                            assert!(ps.len() <= join_most());
                            for (i, &p) in ps.iter().enumerate() {
                                assert!(at.insert(p, (e, i)).is_none(), "round {round}: part {p} sent twice");
                            }
                        }
                        Done::Signal(f, v) => {
                            signal_at.insert((*f, *v), e);
                        }
                        Done::Present(_) => {}
                    }
                }
                assert_eq!(at.len(), parts, "round {round} (joining {joining}): every part sent: {done:?}");
                for m in &made {
                    let mut last_part: Option<(usize, usize)> = None;
                    let mut gate: usize = 0;
                    let mut gate_part: Option<(usize, usize)> = None;
                    for x in m {
                        match *x {
                            Made::Part(p) => {
                                let here = at[&p];
                                if let Some(before) = last_part {
                                    assert!(here > before, "round {round}: part {p} ahead of the part before it in its line: {done:?}");
                                }
                                assert!(here.0 >= gate, "round {round}: part {p} ahead of the signal it waits for: {done:?}");
                                if let Some(by) = gate_part {
                                    assert!(here > by, "round {round}: part {p} ahead of the write its wait needs: {done:?}");
                                }
                                last_part = Some(here);
                            }
                            Made::Signal(f, v) => {
                                let e = signal_at[&(f, v)];
                                if let Some(before) = last_part {
                                    assert!(e > before.0, "round {round}: signal {f:#x}={v} given before the part ahead of it was sent: {done:?}");
                                }
                            }
                            Made::WaitFence(f, v) => {
                                let e = signal_at.iter().filter(|((sf, sv), _)| *sf == f && *sv >= v).map(|(_, &e)| e).min().expect("signalled");
                                gate = gate.max(e + 1);
                            }
                            Made::WaitPart(by) => {
                                let w = at[&by];
                                gate_part = Some(gate_part.map_or(w, |g| g.max(w)));
                            }
                        }
                    }
                }
                assert!(fake.waiting().is_empty(), "round {round}: every line ran out: {done:?}");
            }
        }
    }
}
