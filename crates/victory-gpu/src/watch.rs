//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! Writes into physical pages through another view are seen too: the
//! runtime tells us the views of the pages it wrote itself ([`writing`],
//! xmem's `write_physical` and DirectStorage's page destinations), and a
//! write by the title through any view of a guarded page faults like one
//! at the guarded address (the guard makes every view of the page
//! read-only, [`open`] gives them all back; not a page of more than
//! [`MANY_VIEWS`] views). A
//! system call writing into a guarded block fails (none is known to).

// With `VICTORY_LIGHT_REFRESH` a block that is read-only by our hand,
// unwritten since, and whose earlier writes are all in the journal has a
// bit (`PUBLISHED`): a range whose blocks all have theirs is answered from
// the journal alone, with no lock of the guards and no kernel call
// (`light`).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

/// A title-memory range whose changed pages another subsystem needs to read.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    seen: Mutex<BTreeMap<usize, u64>>,
}

/// Tracks a runtime GPU-memory range, including physical-page views when
/// `views` is true. Other process memory is not tracked.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    let end = start.checked_add(len)?;
    if len == 0 || writes(start, end, views).is_none() {
        return None;
    }
    Some(Tracked {
        start,
        end,
        seen: Mutex::default(),
    })
}

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Vec<(usize, usize)>,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
    /// With [`journal_unlocked`]: page ranges a [`writes`] is asking the
    /// kernel (or the guards) about with the journal let go ([`Asking`]):
    /// their pages are reset there and not in the journal yet, so nothing
    /// else asks or answers about pages of theirs until they are
    /// ([`unbusy`]). Empty without the switch.
    busy: Vec<(usize, usize)>,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
    ahead: Vec::new(),
    ahead_asked: 0,
    kept: false,
    busy: Vec::new(),
});

/// The journal as [`crate::lockmeter::lock_journal`] takes it.
type JournalHeld = crate::lockmeter::JournalHeld<Journal>;

#[cfg(test)]
thread_local! {
    /// The tests' setting of [`journal_unlocked`] on this thread.
    static JOURNAL_UNLOCKED_IN_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether a [`writes`] asks the kernel, or the guards, with the journal's
/// lock let go (`VICTORY_JOURNAL_UNLOCKED=1`; off by default).
///
/// The asking held the journal for its `GetWriteWatch` call and, where that
/// failed, for the guards' walk. The copies the last flush brings up to
/// date on the `wide` helpers (`VICTORY_WIDE_REFRESH=1`) each ask about
/// their own range and stood in line for it, one asking at a time: the
/// trace of 26/11 (map §14) found the submission's thread 18.5 ms a frame
/// waiting for the journal's lock, held by those helpers. With the switch
/// the range is noted busy ([`Journal::busy`]) and the lock let go while
/// it is asked about; whatever overlaps it waits ([`unbusy`]), ranges that
/// do not are asked at the same time, and the pages found get their
/// asking's number when the lock is taken again: a number can only come
/// later than without the switch, which copies a page again and never
/// misses one. No other lock is taken in its place.
///
/// May turn within a run (`crate::ab`): every reader waits for the busy
/// ranges whichever way the switch is, and there are none while it is off.
/// From JozefDeTremmerie/CCS03 f94625a, which mends what kept fork #1102
/// out (the guards' walk still under the journal; an answer read from the
/// journal while an asking's pages were reset and not noted).
fn journal_unlocked() -> bool {
    #[cfg(test)]
    if let Some(on) = JOURNAL_UNLOCKED_IN_TEST.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let run = *ON.get_or_init(|| std::env::var_os("VICTORY_JOURNAL_UNLOCKED").is_some_and(|v| v == "1"));
    crate::ab::flag(crate::ab::Key::JournalUnlocked, run)
}

/// Told when a range leaves [`Journal::busy`].
static UNBUSY: std::sync::Condvar = std::sync::Condvar::new();

/// The journal once no range a [`writes`] is asking about crosses one of
/// the page ranges `pages` (`[first, last)` each): only those wait, never
/// for a range of other pages. At once when nothing is busy, as always
/// without [`journal_unlocked`].
fn unbusy(j: JournalHeld, pages: &[(usize, usize)]) -> JournalHeld {
    if j.busy.is_empty() {
        return j;
    }
    j.wait_while(&UNBUSY, |j| {
        j.busy.iter().any(|&(a, b)| pages.iter().any(|&(first, last)| a < last && first < b))
    })
}

/// A range in [`Journal::busy`]; leaves it when dropped (a panic in the
/// asking must not leave other threads waiting for it).
struct Asking(usize, usize);

impl Asking {
    /// Takes the range out of `j.busy` and tells the waiters.
    fn done(self, j: &mut Journal) {
        if let Some(at) = j.busy.iter().position(|&r| r == (self.0, self.1)) {
            j.busy.swap_remove(at);
        }
        UNBUSY.notify_all();
        std::mem::forget(self);
    }
}

impl Drop for Asking {
    fn drop(&mut self) {
        let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(at) = j.busy.iter().position(|&r| r == (self.0, self.1)) {
            j.busy.swap_remove(at);
        }
        UNBUSY.notify_all();
    }
}

thread_local! {
    /// Room for the kernel's answer to a [`writes`] made with the journal
    /// let go ([`journal_unlocked`]).
    static UNLOCKED_FOUND: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|r| r.0 < r.1)
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    if pieces.len() < 2 {
        return;
    }
    pieces.sort_unstable();
    let held = crate::lockmeter::lock_journal(&JOURNAL);
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`. Said before any wait.
    if held.kept {
        return;
    }
    // View memory the guards hold is left to `writes`, which answers it
    // from the guards ([`known_view`]): the kernel's write watch can only
    // fail there. Asked here with the others, a run holding such a range
    // failed, was asked again by halves down to single ranges, failing
    // each time, and the range then went to `writes` all the same (after
    // triippywhale d0581a55). `VICTORY_VIEW_ASK=1` asks as before.
    #[cfg(feature = "diagnostics")]
    let all = pieces.len();
    pieces.retain(|&(first, _)| !known_view(first * PAGE));
    #[cfg(feature = "diagnostics")]
    AHEAD_VIEWS.fetch_add((all - pieces.len()) as u64, std::sync::atomic::Ordering::Relaxed);
    // A range a [`writes`] asks about with the journal let go
    // ([`journal_unlocked`]) whose pages these askings would reset is
    // waited for, and only such a range: the spans of the runs are what the
    // kernel is asked about (by halves too, inside them). With nothing busy
    // (always without the switch) nothing is made for it, and the lock is
    // held throughout, as before; else it is let go only while such a range
    // is waited for, and the standing asking is looked at again after.
    let mut held = if held.busy.is_empty() {
        held
    } else {
        let mut spans = Vec::new();
        ahead_runs(&pieces, |_, span| spans.push(span));
        let held = unbusy(held, &spans);
        if held.kept {
            return;
        }
        held
    };
    let j = &mut *held;
    j.asked += 1;
    j.ahead_asked = j.asked;
    j.ahead.clear();
    ahead_runs(&pieces, |run, _| ask_run(j, &pieces[run]));
}

/// [`ask_ahead`]'s runs of `pieces` (page ranges in order), each given to
/// `each` in order: pieces closer than [`AHEAD_GAP`] together, no run over
/// [`AHEAD_MAX`]; the run's pieces and the span asked about.
fn ahead_runs(pieces: &[(usize, usize)], mut each: impl FnMut(std::ops::Range<usize>, (usize, usize))) {
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len()
            && pieces[k].0 <= last + AHEAD_GAP
            && last.max(pieces[k].1) - first <= AHEAD_MAX
        {
            last = last.max(pieces[k].1);
            k += 1;
        }
        each(i..k, (first, last));
        i = k;
    }
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// Since the last frame line: ranges of view memory the guards hold that a
/// submission's askings ahead ([`ask_ahead`], [`ask_once`]) left out. Each
/// was a kernel asking that failed, or several, before.
#[cfg(feature = "diagnostics")]
pub(crate) static AHEAD_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    let mut j = crate::lockmeter::lock_journal(&JOURNAL);
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    // The light refresh: a few of the holes known by their bits are asked
    // of Windows again, a submission at a time ([`look_at_holes`]).
    if light_kept() {
        look_at_holes(HOLES_A_LOOK);
    }
    ask_once_end();
    ask_ahead(ranges);
    // Only a range being asked about with the journal let go that crosses
    // one of the submission's is waited for ([`unbusy`]); with nothing busy
    // (always without the switch) nothing is made for it.
    let mut j = crate::lockmeter::lock_journal(&JOURNAL);
    if !j.busy.is_empty() {
        let pages: Vec<(usize, usize)> =
            ranges.iter().filter(|r| r.0 < r.1).map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE))).collect();
        j = unbusy(j, &pages);
    }
    let j = &mut *j;
    if j.ahead.is_empty() {
        // `ask_ahead` asked nothing (fewer than two ranges).
        j.asked += 1;
        j.ahead_asked = j.asked;
    }
    // `ask_ahead` leaves a range that lies alone to `writes`, which asks
    // once per flush: with one asking for the submission that was every
    // such range asked again at each of its 70 flushes (run 76: 2,300
    // askings a frame with the switch on, as without it). They are asked
    // here, once; what is not write-watched stays with `writes`.
    for &(start, end) in ranges {
        if start >= end {
            continue;
        }
        // View memory the guards hold is not asked about: the asking can
        // only fail there, one failed kernel call for each such copy at
        // every submission, and `writes` answers it from the guards
        // (after triippywhale cc368c7c).
        if known_view(start) {
            #[cfg(feature = "diagnostics")]
            AHEAD_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if !j.ahead.iter().any(|&(a, b)| a <= first && last <= b) && ask_kernel(j, first, last) {
            j.ahead.push((first, last));
        }
    }
    j.kept = !j.ahead.is_empty();
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    let mut j = crate::lockmeter::lock_journal(&JOURNAL);
    if j.kept {
        j.kept = false;
        j.ahead.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    unchanged_gone(at, len);
    let mut j = crate::lockmeter::lock_journal(&JOURNAL);
    if !j.kept || len == 0 {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    // The asking that stands moves with it: `asked` and `writes` give it as
    // "now", and a page noted written after "now" stays stale for every
    // copy that takes it (copied at `now`, written at `now + 1`): copied
    // again at every refresh, and a reason to settle flights
    // (fxceless/victory-fork#410).
    j.ahead_asked = now;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        j.written.insert(page, now);
    }
}

/// We wrote the 64 KiB page numbered `page` of title memory through a
/// view of our own (`crate::inplace::write_own`): no guard was opened and
/// no kernel write-watch bit set, so the journal is told here. The page,
/// and the same physical page at its other title addresses (`others`,
/// page numbers), count as written at a new asking, which every host copy
/// that holds them takes at its next refresh. Returns the asking that had
/// last found `page` written (0: never) and the new one, for the copy
/// that wrote to note itself current.
pub(crate) fn wrote_own(page: usize, others: &[usize]) -> (u64, u64) {
    unchanged_gone(page * PAGE, PAGE);
    for &other in others {
        unchanged_gone(other * PAGE, PAGE);
    }
    let mut j = crate::lockmeter::lock_journal(&JOURNAL);
    let j = &mut *j;
    j.asked += 1;
    let now = j.asked;
    // As in [`wrote`]: an asking that stands for a submission, or for a
    // flush, moves with it, or a copy taking the page at the standing
    // asking would find it written "later" at every refresh.
    if j.kept || !j.ahead.is_empty() {
        j.ahead_asked = now;
    }
    let before = j.written.insert(page, now).unwrap_or(0);
    for &other in others {
        if other != page {
            j.written.insert(other, now);
        }
    }
    (before, now)
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`].
fn ask_run(j: &mut Journal, pieces: &[(usize, usize)]) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if ask_kernel(j, first, last) {
        j.ahead.push((first, last));
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(j, a);
    ask_run(j, b);
}

/// Asks the kernel for the written pages of pages `[first, last)` and notes
/// them in the journal under the ahead asking; false (and nothing changed)
/// if the range is not write-watched memory of one reservation.
fn ask_kernel(j: &mut Journal, first: usize, last: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    if failed {
        return false;
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        crate::framelog::add(crate::framelog::T::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    let now = j.ahead_asked;
    for k in 0..count {
        let page = j.found[k] / PAGE;
        j.written.insert(page, now);
    }
    kernel_answered(first, last, &j.found[..count]);
    true
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = unbusy(crate::lockmeter::lock_journal(&JOURNAL), &[(first, last)]);
    j.ahead
        .iter()
        .any(|&(a, b)| a <= first && last <= b)
        .then(|| (j.ahead_asked, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    let mut written = Vec::new();
    writes_into(start, end, guarded, &mut written).map(|now| (now, written))
}

/// [`writes`] into `written` (emptied first), a list the caller keeps
/// between askings: a new list at each asking showed in the submitting
/// thread's profile (fork #987, cybx: vectors built from tree iterators
/// 3.9 % and 3.1 % of its samples, their growth 3.3 %).
pub(crate) fn writes_into(start: usize, end: usize, guarded: bool, written: &mut Vec<(usize, u64)>) -> Option<u64> {
    // The light refresh: its bits kept, and asked or not, as the switch
    // says now ([`writes_as`]).
    let light = if guarded && light_kept() { 1 + u8::from(light()) } else { 0 };
    writes_as(start, end, guarded, light, written)
}

/// [`writes_into`] with the light refresh as `light` says ([`light`]): 0,
/// as without the switch, no bit is set or asked; 1, the blocks the asking
/// leaves clean get their bits and none is asked (arm A of an A/B on it);
/// 2, a range whose blocks all have their bit is answered from the journal.
fn writes_as(start: usize, end: usize, guarded: bool, light: u8, written: &mut Vec<(usize, u64)>) -> Option<u64> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut held = unbusy(crate::lockmeter::lock_journal(&JOURNAL), &[(first, last)]);
    let j = &mut *held;
    // Asked ahead with the submission's other ranges: the journal has it.
    if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
        written.clear();
        written.extend(j.written.range(first..last).map(|(&p, &w)| (p, w)));
        return Some(j.ahead_asked);
    }
    // The light refresh: every block of the range is read-only by our hand
    // and unwritten since an asking noted it, or a hole noted for good
    // ([`PUBLISHED`]). The guards would find nothing written and this
    // asking would note nothing: its number and the journal as it is are
    // the answer, with the guards' lock not taken and no kernel call.
    if guarded && light == 2 && published(j, first * PAGE, last * PAGE) {
        LIGHT_ASKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        j.asked += 1;
        written.clear();
        written.extend(j.written.range(first..last).map(|(&p, &w)| (p, w)));
        return Some(j.asked);
    }
    // The kernel, or the guards, asked with the journal let go.
    if journal_unlocked() {
        return writes_unlocked(held, (first, last), guarded, light, written);
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // View memory the guards already hold cannot be write-watched: the
    // kernel asking would only fail. Copies of it are flushed many times a
    // frame, each time paying that failed call before the guards answer.
    //
    // Also for a caller that does not take the guards' answer (`guarded`
    // false: the texture refresh asks the write watch first, then the
    // guards): it gets None at once, without the failing call. The
    // watch_remap's 185-191 failed askings a frame were mostly those.
    // `VICTORY_WATCH_VIEW_EARLY=0`: only for askers that take the guards'
    // answer, as before.
    let known_view = (guarded || watch_view_early()) && known_view(first * PAGE);
    if known_view && !guarded {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WatchKnownView, 1);
        return None;
    }
    // SAFETY: the answer buffer holds `count` entries.
    let failed = known_view
        || unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                j.found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        if known_view {
            crate::copystat::C::WatchKnownView
        } else {
            crate::copystat::C::WatchFailed
        },
        failed as u64,
    );
    let guarded = if failed {
        count = 0;
        if !guarded {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = guard(first * PAGE, last * PAGE);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
        written?
    } else {
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        crate::framelog::add(crate::framelog::T::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    if !failed {
        kernel_answered(first, last, &j.found[..count]);
    }
    j.asked += 1;
    let now = j.asked;
    let pages: Vec<usize> = j.found[..count].iter().map(|&at| at / PAGE).chain(guarded).collect();
    for &page in &pages {
        j.written.insert(page, now);
    }
    // The light refresh: the guards were asked, and what they found is
    // noted. The blocks they made clean are known by their bits from now
    // on (the journal's lock is still held: no other asking came between).
    if failed && light != 0 {
        LIGHT_ASKS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        publish(j, &pages, first * PAGE, last * PAGE, told());
    }
    written.clear();
    written.extend(j.written.range(first..last).map(|(&p, &w)| (p, w)));
    Some(now)
}

/// [`writes_as`] past the journal's own answers, with [`journal_unlocked`]:
/// the range is noted busy, the journal let go while the kernel or the
/// guards are asked ([`ask_unlocked`], the same asking as there), then
/// taken again to note what was found, under an asking numbered then.
fn writes_unlocked(
    mut held: JournalHeld,
    (first, last): (usize, usize),
    guarded: bool,
    light: u8,
    written: &mut Vec<(usize, u64)>,
) -> Option<u64> {
    held.busy.push((first, last));
    let asking = Asking(first, last);
    drop(held);
    let found = UNLOCKED_FOUND.with_borrow_mut(|found| ask_unlocked(found, (first, last), guarded));
    let mut held = crate::lockmeter::lock_journal(&JOURNAL);
    let j = &mut *held;
    asking.done(j);
    let (pages, failed) = found?;
    j.asked += 1;
    let now = j.asked;
    for &page in &pages {
        j.written.insert(page, now);
    }
    // A range that crosses one asked ahead ([`ask_ahead`], [`ask_once`])
    // without lying inside it is asked here, and the pages it found in that
    // range now have a later number than the asking that stands: the
    // standing answer moves with it, as [`wrote`] moves it, or a copy
    // answered from the journal would take such a page at the standing
    // number and find it written later at every refresh (the class of
    // fxceless/victory-fork#410). A later number takes a page again, never
    // misses one.
    if pages.iter().any(|&page| j.ahead.iter().any(|&(a, b)| a <= page && page < b)) {
        j.ahead_asked = now;
    }
    // The light refresh, as [`writes_as`] notes it: the waiters for this
    // range are still kept out by the journal's lock, held from here on.
    if failed && light != 0 {
        LIGHT_ASKS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        publish(j, &pages, first * PAGE, last * PAGE, told());
    }
    written.clear();
    written.extend(j.written.range(first..last).map(|(&p, &w)| (p, w)));
    Some(now)
}

/// [`writes_unlocked`]'s asking, with the journal let go: the kernel's write
/// watch about pages `[first, last)` (reset as it answers), or the guards
/// where that fails and `guarded`, as in [`writes_as`]. The pages found
/// written and whether the kernel's asking failed; `None` where neither can
/// tell.
fn ask_unlocked(found: &mut Vec<usize>, (first, last): (usize, usize), guarded: bool) -> Option<(Vec<usize>, bool)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    if found.len() < count {
        found.resize(count, 0);
    }
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let known_view = (guarded || watch_view_early()) && known_view(first * PAGE);
    if known_view && !guarded {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WatchKnownView, 1);
        return None;
    }
    // SAFETY: the answer buffer holds `count` entries.
    let failed = known_view
        || unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        if known_view {
            crate::copystat::C::WatchKnownView
        } else {
            crate::copystat::C::WatchFailed
        },
        failed as u64,
    );
    let guarded = if failed {
        count = 0;
        if !guarded {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = guard(first * PAGE, last * PAGE);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
        written?
    } else {
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        crate::framelog::add(crate::framelog::T::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    if !failed {
        kernel_answered(first, last, &found[..count]);
    }
    Some((found[..count].iter().map(|&at| at / PAGE).chain(guarded).collect(), failed))
}

#[cfg(test)]
mod journal_unlocked_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Acquire, Ordering::Relaxed, Ordering::Release, Ordering::SeqCst};

    /// Private, committed, write-watched memory of `pages` pages of 64 KiB,
    /// as the title's is.
    struct Memory(usize, usize);

    impl Memory {
        fn new(pages: usize) -> Memory {
            use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
            const MEM_WRITE_WATCH: u32 = 0x0020_0000;
            let len = pages * PAGE;
            // SAFETY: a plain allocation of our own.
            let at = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) };
            assert!(!at.is_null() && (at as usize).is_multiple_of(PAGE));
            Memory(at as usize, len)
        }
    }

    impl Drop for Memory {
        fn drop(&mut self) {
            use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
            forget(self.0, self.0 + self.1);
            // SAFETY: our own reservation, which nothing reads any more.
            unsafe { VirtualFree(self.0 as *mut _, 0, MEM_RELEASE) };
        }
    }

    /// The journal's notes of `[start, end)` dropped (the next test that
    /// gets the same addresses finds none).
    fn forget(start: usize, end: usize) {
        let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        let pages: Vec<usize> = j.written.range(start / PAGE..end.div_ceil(PAGE)).map(|(&p, _)| p).collect();
        for page in pages {
            j.written.remove(&page);
        }
    }

    /// A title write of the 8 bytes at `at`, as an atomic store: the test
    /// reads that memory only through the kernel while others write it.
    fn put(at: usize, value: u64) {
        // SAFETY: 8-aligned title memory of the test's, written only so.
        unsafe { (*(at as *const AtomicU64)).store(value, Relaxed) };
    }

    /// A host copy of `[start, start + len)` as `Shadow::update` keeps one:
    /// each page with the asking it was last taken at (0: never), taken
    /// again where the journal has it written later, read through the
    /// kernel (`crate::host::read_into`), as the layer reads title memory.
    struct HostCopy {
        start: usize,
        guarded: bool,
        bytes: Vec<u8>,
        taken: Vec<u64>,
    }

    impl HostCopy {
        fn new(start: usize, len: usize, guarded: bool) -> HostCopy {
            HostCopy { start, guarded, bytes: vec![0; len], taken: vec![0; len / PAGE] }
        }

        /// Brings `[from, to)` (page-aligned offsets) up to date; the pages
        /// copied.
        fn refresh(&mut self, from: usize, to: usize) -> usize {
            let Some((now, written)) = writes(self.start + from, self.start + to, self.guarded) else {
                panic!("watched or guarded memory");
            };
            let base = self.start / PAGE;
            let mut copied = 0;
            for page in (self.start + from) / PAGE..(self.start + to) / PAGE {
                let w = written.iter().find(|&&(p, _)| p == page).map_or(0, |&(_, w)| w);
                let at = &mut self.taken[page - base];
                if *at == 0 || w > *at {
                    *at = now;
                    let off = (page - base) * PAGE;
                    assert!(crate::host::read_into((self.start + off) as *const u8, &mut self.bytes[off..off + PAGE]));
                    copied += 1;
                }
            }
            copied
        }

        /// The 8 bytes of slot `i` (offset `8 * i`).
        fn slot(&self, i: usize) -> u64 {
            u64::from_le_bytes(self.bytes[8 * i..8 * i + 8].try_into().unwrap())
        }
    }

    /// The case of fxceless/victory-fork#410 the review names: a submission's
    /// one asking stands over A; the title writes in A∩B; B, which crosses A
    /// without lying inside it, is asked about; then a copy of A answered
    /// from the journal is brought up to date twice. With the switch the
    /// page is taken once and then not again (the standing answer moved with
    /// B's asking). Off it is taken at every refresh, as before the switch:
    /// that path is left as it was.
    #[test]
    fn a_crossing_asking_moves_the_standing_answer() {
        let _turn = crate::warp_turn();
        for on in [false, true] {
            JOURNAL_UNLOCKED_IN_TEST.set(Some(on));
            let memory = Memory::new(4);
            let m = memory.0;
            ask_once(&[(m, m + 2 * PAGE)]);
            let mut copy = HostCopy::new(m, 2 * PAGE, false);
            assert_eq!(copy.refresh(0, 2 * PAGE), 2, "the first refresh takes all");
            put(m + PAGE + 64, 0x5a5a);
            assert!(writes(m + PAGE, m + 3 * PAGE, false).is_some());
            let taken = [copy.refresh(0, 2 * PAGE), copy.refresh(0, 2 * PAGE)];
            ask_once_end();
            assert_eq!(copy.slot((PAGE + 64) / 8), 0x5a5a, "the title's write is in the copy (on: {on})");
            if on {
                assert_eq!(taken, [1, 0], "taken once, then current");
            } else {
                assert_eq!(taken, [1, 1], "as before the switch");
            }
            drop(memory);
        }
        JOURNAL_UNLOCKED_IN_TEST.set(None);
    }

    /// The concurrent oracle. A writer stores rising numbers at random
    /// 8-byte slots of the memory, and after each store says, per slot, the
    /// last number it stored there (`last`, with release). A submission
    /// thread loops: notes `last` (acquire), makes that its floor, asks
    /// once (`ask_once`) for two of the memory's pages, brings its copy of
    /// them up to date, checks, and ends the asking. Three helpers bring
    /// copies of ranges that cross the submission's and each other's up to
    /// date, each checking against its own note of `last` taken before the
    /// refresh, lowered to the submission's floor where an asking may stand
    /// (an answer from a standing asking is as of that asking). A copy that
    /// holds, at some slot, a number below the floor has lost a write that
    /// ended before its refresh began. Once all have stopped, every copy
    /// brought up to date once more is the memory, byte for byte. Run on
    /// private write-watched memory and on view memory (two views of one
    /// section: written through one, copied through the other, the guards'
    /// way), the switch off and on.
    #[test]
    fn copies_never_hold_a_write_older_than_their_refresh() {
        let _turn = crate::warp_turn();
        for views in [false, true] {
            for on in [false, true] {
                let (refreshes, stores) = oracle(views, on);
                assert!(refreshes > 40 && stores > 1000, "views {views} on {on}: {refreshes} refreshes, {stores} stores");
            }
        }
    }

    fn oracle(views: bool, on: bool) -> (u64, u64) {
        const PAGES: usize = 4;
        let len = PAGES * PAGE;
        let slots = len / 8;
        let (memory, view) = if views {
            let (section, one, two) = super::tests::two_views_for_others(len);
            (None, Some((section, one, two)))
        } else {
            (Some(Memory::new(PAGES)), None)
        };
        let (read, write) = match (&memory, view) {
            (Some(m), _) => (m.0, m.0),
            (None, Some((_, one, two))) => (one, two),
            _ => unreachable!(),
        };
        let last: Vec<AtomicU64> = (0..slots).map(|_| AtomicU64::new(0)).collect();
        let floor: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(vec![0; slots]);
        let stop = AtomicBool::new(false);
        let (refreshes, lost, stores) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
        let note = |from: usize, to: usize| -> Vec<u64> { (from / 8..to / 8).map(|i| last[i].load(Acquire)).collect() };
        let check = |copy: &HostCopy, from: usize, floor: &[u64]| {
            for (k, &f) in floor.iter().enumerate() {
                if copy.slot(from / 8 + k) < f {
                    lost.fetch_add(1, Relaxed);
                }
            }
        };
        let copies: Vec<HostCopy> = std::thread::scope(|s| {
            s.spawn(|| {
                let mut x = 0x9e37_79b9_7f4a_7c15u64 ^ u64::from(on) ^ u64::from(views) << 1;
                let mut seq = 0u64;
                while !stop.load(SeqCst) {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    let i = x as usize % slots;
                    seq += 1;
                    put(write + 8 * i, seq);
                    last[i].store(seq, Release);
                }
                stores.store(seq, SeqCst);
            });
            let submission = s.spawn(|| {
                JOURNAL_UNLOCKED_IN_TEST.set(Some(on));
                let mut copy = HostCopy::new(read, len, views);
                let own = [(0, PAGE), (2 * PAGE, 3 * PAGE)];
                while !stop.load(SeqCst) {
                    let noted = note(0, len);
                    floor.lock().unwrap().clone_from(&noted);
                    ask_once(&own.map(|(a, b)| (read + a, read + b)));
                    for (a, b) in own {
                        copy.refresh(a, b);
                        check(&copy, a, &noted[a / 8..b / 8]);
                    }
                    ask_once_end();
                    refreshes.fetch_add(1, Relaxed);
                }
                copy
            });
            let helpers: Vec<_> = (0..3)
                .map(|k| {
                    let (note, check, floor, stop, refreshes) = (&note, &check, &floor, &stop, &refreshes);
                    s.spawn(move || {
                        JOURNAL_UNLOCKED_IN_TEST.set(Some(on));
                        let mut copy = HostCopy::new(read, len, views);
                        let ranges = [(0, len), (0, 2 * PAGE), (PAGE, 3 * PAGE), (2 * PAGE, len), (PAGE, 2 * PAGE)];
                        let mut turn = k;
                        while !stop.load(SeqCst) {
                            let (a, b) = ranges[turn % ranges.len()];
                            let mut noted = note(a, b);
                            let standing = floor.lock().unwrap();
                            for (n, f) in noted.iter_mut().zip(&standing[a / 8..b / 8]) {
                                *n = (*n).min(*f);
                            }
                            drop(standing);
                            copy.refresh(a, b);
                            check(&copy, a, &noted);
                            turn += 1;
                            refreshes.fetch_add(1, Relaxed);
                        }
                        copy
                    })
                })
                .collect();
            std::thread::sleep(std::time::Duration::from_millis(250));
            stop.store(true, SeqCst);
            let mut copies = vec![submission.join().unwrap()];
            copies.extend(helpers.into_iter().map(|h| h.join().unwrap()));
            copies
        });
        // Every thread has ended: each copy brought up to date once more is
        // the memory as it is.
        JOURNAL_UNLOCKED_IN_TEST.set(Some(on));
        let mut title = vec![0u8; len];
        assert!(crate::host::read_into(read as *const u8, &mut title));
        for (n, mut copy) in copies.into_iter().enumerate() {
            copy.refresh(0, len);
            let differing = copy.bytes.iter().zip(&title).filter(|(a, b)| a != b).count();
            assert_eq!(differing, 0, "copy {n} is not the memory (views {views}, on {on})");
        }
        JOURNAL_UNLOCKED_IN_TEST.set(None);
        assert_eq!(lost.load(SeqCst), 0, "slots below their floor (views {views}, on {on})");
        if let Some((section, one, two)) = view {
            super::tests::drop_views(section, &[one, two], len);
        }
        drop(memory);
        (refreshes.load(SeqCst), stores.load(SeqCst))
    }

    /// A range being asked about with the journal let go holds off an asker
    /// whose pages cross it until its pages are in the journal, and not an
    /// asker of other pages; an asking ahead of other pages does not wait
    /// for it either. A dropped asking leaves its range.
    #[test]
    fn a_busy_range_holds_off_crossing_askers_only() {
        let _turn = crate::warp_turn();
        let memory = Memory::new(6);
        let (m, first) = (memory.0, memory.0 / PAGE);
        // As `writes_unlocked` notes it while it asks.
        let asking = {
            let mut j = JOURNAL.lock().unwrap();
            j.busy.push((first, first + 2));
            Asking(first, first + 2)
        };
        let (crossing, other, ahead) = (AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false));
        // The asking belongs to the scope's closure: an assertion that fails
        // in it drops the asking as it unwinds, before the scope waits for
        // its threads, and the crossing asker goes on (no test hangs).
        std::thread::scope(|s| {
            // (Moved in: captured by value, the rest by reference.)
            let asking = asking;
            s.spawn(|| {
                assert!(writes(m + PAGE, m + 2 * PAGE, false).is_some());
                crossing.store(true, SeqCst);
            });
            s.spawn(|| {
                assert!(writes(m + 3 * PAGE, m + 4 * PAGE, false).is_some());
                other.store(true, SeqCst);
            });
            s.spawn(|| {
                ask_ahead(&[(m + 4 * PAGE, m + 5 * PAGE), (m + 5 * PAGE, m + 6 * PAGE)]);
                ask_ahead_done();
                ahead.store(true, SeqCst);
            });
            let until = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while !(other.load(SeqCst) && ahead.load(SeqCst)) && std::time::Instant::now() < until {
                std::thread::yield_now();
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(other.load(SeqCst), "an asker of other pages went on");
            assert!(ahead.load(SeqCst), "an asking ahead of other pages went on");
            assert!(!crossing.load(SeqCst), "a crossing asker waited");
            asking.done(&mut JOURNAL.lock().unwrap());
        });
        assert!(crossing.load(SeqCst), "and went on once the range was done");
        let dropped = {
            let mut j = JOURNAL.lock().unwrap();
            j.busy.push((first + 2, first + 3));
            Asking(first + 2, first + 3)
        };
        drop(dropped);
        assert!(!JOURNAL.lock().unwrap().busy.contains(&(first + 2, first + 3)));
    }
}

impl Tracked {
    /// Takes changed pages of `[from, to)`, cut to the tracked range. A
    /// partial page stays changed so a later take can read its other bytes.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Vec::new();
        }
        let Some((_, pages)) = writes(from, to, true) else {
            return Vec::new();
        };
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (page, write) in pages {
            if seen.get(&page).is_some_and(|&last| last >= write) {
                continue;
            }
            let (start, end) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
            if start >= end {
                continue;
            }
            // Keep partial pages dirty: the caller has not taken the rest.
            if (page * PAGE).max(self.start) >= from
                && ((page + 1) * PAGE).min(self.end) <= to
            {
                seen.insert(page, write);
            }
            match ranges.last_mut() {
                Some((_, last)) if *last == start => *last = end,
                _ => ranges.push((start, end)),
            }
        }
        ranges
    }
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

/// Whether the block holding `start` is view memory the guards already
/// hold (read-only by our hand, or noted absent): asking the kernel's write
/// watch about it can only fail. `VICTORY_VIEW_ASK=1` (or
/// `VICTORY_GUARD_ASK=1`) asks the kernel anyway, as before.
fn known_view(start: usize) -> bool {
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let ask = *ASK.get_or_init(|| {
        ["VICTORY_VIEW_ASK", "VICTORY_GUARD_ASK"]
            .iter()
            .any(|n| std::env::var_os(n).is_some_and(|v| v == "1"))
    });
    if ask {
        return false;
    }
    let first = start / BLOCK;
    // The light refresh: a block known by its bit is read-only by our hand
    // or a hole the guards noted, which is what they are asked below.
    if light() && PUBLISHED.all(first, first + 1) {
        return true;
    }
    let g = guards(crate::lockmeter::Site::KnownView);
    if g.protected.contains(&first) || g.absent.contains(&first) {
        return true;
    }
    let kept = view_known() && g.viewed.contains(&first);
    drop(g);
    #[cfg(feature = "diagnostics")]
    if kept {
        kept_view_check(first);
    }
    kept
}

/// Of the askings that took a block for view memory only because it is
/// remembered as one ([`view_known`]; after a remapping, the watch_remap
/// keeps it), every 64th: Windows is asked whether it is one (`is_view`).
/// Since the start: askings looked at, those Windows said are no view, and
/// the address of the last such block. A block that is no view any more
/// is not write-watched for it: its refreshes go by the guards alone.
#[cfg(feature = "diagnostics")]
pub(crate) static VIEW_KEPT: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn kept_view_check(block: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    static ASKINGS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if !ASKINGS.fetch_add(1, Relaxed).is_multiple_of(64) {
        return;
    }
    VIEW_KEPT[0].fetch_add(1, Relaxed);
    if !is_view(block * BLOCK, (block + 1) * BLOCK) {
        VIEW_KEPT[1].fetch_add(1, Relaxed);
        VIEW_KEPT[2].store((block * BLOCK) as u64, Relaxed);
    }
}

/// A set of blocks: one by one as before, or as runs of blocks that
/// follow one another with [`guard_intervals`] (chosen once for the run).
struct BlockSet {
    set: BTreeSet<usize>,
    runs: CleanBlocks,
    /// Which of the two holds the blocks now (runs: [`guard_intervals`]).
    runs_mode: bool,
}

impl BlockSet {
    const fn new() -> Self {
        Self { set: BTreeSet::new(), runs: CleanBlocks(BTreeMap::new()), runs_mode: false }
    }

    /// The blocks moved into the other store if `runs` says another (under
    /// the guards' lock, [`guards`]): the switch turned within the run.
    fn store_as(&mut self, runs: bool) {
        if self.runs_mode == runs {
            return;
        }
        if runs {
            let mut at: Option<(usize, usize)> = None;
            for &block in &self.set {
                match at {
                    Some((from, to)) if to == block => at = Some((from, to + 1)),
                    _ => {
                        if let Some((from, to)) = at {
                            self.runs.insert(from, to);
                        }
                        at = Some((block, block + 1));
                    }
                }
            }
            if let Some((from, to)) = at {
                self.runs.insert(from, to);
            }
            self.set.clear();
        } else {
            for (&from, &to) in &self.runs.0 {
                self.set.extend(from..to);
            }
            self.runs.0.clear();
        }
        self.runs_mode = runs;
    }

    fn contains(&self, block: &usize) -> bool {
        if self.runs_mode {
            self.runs.uncovered(*block, *block + 1).is_empty()
        } else {
            self.set.contains(block)
        }
    }

    fn insert(&mut self, block: usize) {
        if self.runs_mode {
            self.runs.insert(block, block + 1);
        } else {
            self.set.insert(block);
        }
    }

    fn remove(&mut self, block: &usize) {
        if self.runs_mode {
            self.runs.remove(*block);
        } else {
            self.set.remove(block);
        }
    }

    fn extend(&mut self, range: std::ops::Range<usize>) {
        if self.runs_mode {
            self.runs.insert(range.start, range.end);
        } else {
            self.set.extend(range);
        }
    }

    /// The blocks of `range` in the set, in order.
    fn blocks_in(&self, range: std::ops::Range<usize>) -> Vec<usize> {
        if !self.runs_mode {
            return self.set.range(range).copied().collect();
        }
        runs_blocks_in(&self.runs, range)
    }

    /// How many blocks of `range` are in the set.
    fn count_in(&self, range: std::ops::Range<usize>) -> usize {
        if !self.runs_mode {
            return self.set.range(range).count();
        }
        runs_count_in(&self.runs, range)
    }
}

/// The blocks of `range` in `runs`, in order ([`BlockSet::blocks_in`]).
fn runs_blocks_in(runs: &CleanBlocks, range: std::ops::Range<usize>) -> Vec<usize> {
    if range.start >= range.end {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut at = range.start;
    for (from, to) in runs.uncovered(range.start, range.end) {
        out.extend(at..from);
        at = to;
    }
    out.extend(at..range.end.max(at));
    out
}

/// How many blocks of `range` are in `runs` ([`BlockSet::count_in`]).
fn runs_count_in(runs: &CleanBlocks, range: std::ops::Range<usize>) -> usize {
    if range.start >= range.end {
        return 0;
    }
    let missing: usize = runs.uncovered(range.start, range.end).iter().map(|(a, b)| b - a).sum();
    range.end - range.start - missing
}

/// Whether the guards keep the blocks they made read-only, and those known
/// as view memory, as runs of blocks that follow one another rather than
/// one by one (`VICTORY_GUARD_INTERVALS=1`; off by default; read once): the
/// title's view memory is guarded in some 150,000 blocks lying in a few
/// runs, each guarding and opening wrote or walked every block. And `guard`
/// takes the guards' lock once for a range all clean. After fork #1067
/// (UwUCammie), adapted: the clean blocks were runs here already (#979),
/// and what `guard` registers for the refresh stays (`register_refreshing`).
/// The bench measured 1.73 -> 0.74 ms in the yard (block 3, B3-1). May
/// turn within a run (`crate::ab`): the guards' lock taken next moves the
/// blocks into the other store ([`guards`]; some 150,000 blocks, once a
/// turn, in a reading the A/B leaves out).
fn guard_intervals() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let run = *ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_INTERVALS").is_some_and(|v| v == "1"));
    crate::ab::flag(crate::ab::Key::GuardRunsStore, run)
}

/// Whether a write fault opens only the view that was written, leaving the
/// page's other views read-only (`PAGE_READONLY`) yet not clean, without a
/// [`PUBLISHED`] bit, and invalidated in [`Unchanged`] watchers
/// (`VICTORY_OPEN_ONE_VIEW=1`; off by default). After avatarmanz 7bc7c07.
fn open_one_view() -> bool {
    #[cfg(test)]
    if let Some(on) = OPEN_ONE_IN_TEST.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let run = *ON.get_or_init(|| std::env::var_os("VICTORY_OPEN_ONE_VIEW").is_some_and(|v| v == "1"));
    crate::ab::flag(crate::ab::Key::OpenOneView, run)
}

/// How long a block the title has just opened by a write is left open by
/// those that take it (`VICTORY_GUARD_REST=<microseconds>`, `1` => 3,000 us,
/// `0` => disabled; off by default). After avatarmanz 7bc7c07.
fn rest() -> std::time::Duration {
    #[cfg(test)]
    if let Some(d) = REST_IN_TEST.get() {
        return d;
    }
    static US: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let run = *US.get_or_init(|| match std::env::var("VICTORY_GUARD_REST") {
        Ok(v) => match v.trim().parse::<u64>() {
            Ok(1) => 3000,
            Ok(n) => n,
            Err(_) => 0,
        },
        Err(_) => 0,
    });
    let us = crate::ab::number(crate::ab::Key::GuardRest, run);
    let us = if us == 1 { 3000 } else { us };
    std::time::Duration::from_micros(us)
}

#[cfg(test)]
thread_local! {
    /// [`open_one_view`] override for unit tests on this thread.
    static OPEN_ONE_IN_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    /// [`rest`] override for unit tests on this thread.
    static REST_IN_TEST: std::cell::Cell<Option<std::time::Duration>> = const { std::cell::Cell::new(None) };
}

struct Guards {
    /// Blocks we made read-only.
    protected: BlockSet,
    /// Of those, the ones not written since a copy took them.
    clean: CleanBlocks,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: BTreeSet<usize>,
    /// Blocks whose physical page was guarded together with its other views
    /// (those views' blocks are in it too): a write to one opens them all.
    /// A write range can be huge (`fread`), so only these are looked up.
    aliased: BTreeSet<usize>,
    /// Per guarded block: the host copies holding title bytes of it as
    /// current ([`watch_unchanged`]); opening or remapping the block takes
    /// those bytes out of what they hold current.
    watchers: BTreeMap<usize, Vec<std::sync::Arc<Unchanged>>>,
    /// With [`view_known`]: blocks made read-only at least once since the
    /// runtime last mapped or unmapped them. They are view memory whether
    /// their guard stands now or a write has opened it.
    viewed: BlockSet,
    /// With `VICTORY_BACK_OWN_CHECK=1` ([`back_check`]): blocks the runtime
    /// mapped or unmapped that were not made read-only since.
    remapped: CleanBlocks,
    /// With [`absent_once`]: for blocks of `absent`, when ([`guard_ms`])
    /// Windows last said they are uncommitted.
    verified: BTreeMap<usize, u64>,
    /// With the light refresh and notes kept for good ([`told`]): the
    /// blocks of `verified` that [`publish`] has looked at. Those of them
    /// that lie in a placeholder reservation ([`placeholder`]) are known by
    /// their bits ([`PUBLISHED`]) until the runtime tells of them; the
    /// others are answered from their notes, which then grow old
    /// ([`note_life`]).
    holes: BTreeSet<usize>,
    /// With notes kept for good ([`told`]): the noted blocks that lie in
    /// no placeholder reservation ([`placeholder`]), ordinary reservations
    /// or free memory, which anybody can commit or take with no word to us.
    /// Their notes live [`UNSURE_MS`] ([`note_life`]).
    unsure: BTreeSet<usize>,
    /// Blocks opened by a write fault, and when ([`rest`]): a block's slot
    /// is its number modulo [`OPENED_SLOTS`], a later block in the slot
    /// taking it (that one then rests no longer). Made once, before the
    /// handler is registered ([`install`]): the handler only writes slots,
    /// it allocates nothing.
    opened: Vec<(usize, Option<std::time::Instant>)>,
}

/// The slots of `Guards::opened`.
const OPENED_SLOTS: usize = 1 << 13;

impl Guards {
    /// Whether `block` is left open for now: the title has just written it.
    fn rests(&self, block: usize, rest: std::time::Duration) -> bool {
        !rest.is_zero()
            && self
                .opened
                .get(block % OPENED_SLOTS)
                .is_some_and(|&(b, at)| b == block && at.is_some_and(|at| at.elapsed() < rest))
    }

    /// A write fault opened `block` (in the handler: no allocation).
    fn opened(&mut self, block: usize) {
        if let Some(slot) = self.opened.get_mut(block % OPENED_SLOTS) {
            *slot = (block, Some(std::time::Instant::now()));
        }
    }

    /// Blocks `[lo, hi)` rest no longer (the runtime remapped them).
    fn forget_opened(&mut self, lo: usize, hi: usize) {
        if hi.saturating_sub(lo) < OPENED_SLOTS {
            for block in lo..hi {
                if let Some(slot) = self.opened.get_mut(block % OPENED_SLOTS)
                    && slot.0 == block
                {
                    *slot = (usize::MAX, None);
                }
            }
        } else {
            for slot in &mut self.opened {
                if (lo..hi).contains(&slot.0) {
                    *slot = (usize::MAX, None);
                }
            }
        }
    }
}

/// Whether `VICTORY_BACK_OWN_CHECK=1` (`Shadow::put_back_own`'s check): the
/// pages it finds differing are sorted by what stands over them
/// ([`differing_kind`]); nothing is kept for it otherwise.
fn back_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BACK_OWN_CHECK").is_some_and(|v| v == "1"))
}

/// What stands over a block of title memory whose bytes differ from a
/// copy that holds them current ([`differing_kind`]).
#[derive(Clone, Copy)]
pub(crate) enum Over {
    /// Guarded and not written since a copy took it.
    Clean = 0,
    /// View memory, guarded once, its guard opened by a write since (or
    /// written through the runtime, still read-only).
    GuardedDirty = 1,
    /// Not view memory: the kernel's write watch, no guard.
    Unguarded = 2,
    /// Noted absent (uncommitted), or mapped or unmapped by the runtime and
    /// not guarded since.
    AbsentOrRemapped = 3,
}

/// For `Shadow::put_back_own`'s check: what stands over `block`. For an
/// unguarded one, the next kernel asking over it says whether its write bit
/// was set ([`BACK_UNGUARDED`]).
pub(crate) fn differing_kind(block: usize) -> Over {
    {
        let g = guards(crate::lockmeter::Site::DifferingKind);
        if g.clean.uncovered(block, block + 1).is_empty() {
            return Over::Clean;
        }
        if g.absent.contains(&block) || g.remapped.uncovered(block, block + 1).is_empty() {
            return Over::AbsentOrRemapped;
        }
        if g.protected.contains(&block) {
            return Over::GuardedDirty;
        }
    }
    if is_view(block * BLOCK, (block + 1) * BLOCK) {
        return Over::GuardedDirty;
    }
    let mut pending = BACK_PENDING.lock().unwrap_or_else(|e| e.into_inner());
    if pending.len() < 4096 {
        pending.push(block * BLOCK / PAGE);
    }
    Over::Unguarded
}

/// Pages of [`Over::Unguarded`] waiting for the next kernel asking over them.
static BACK_PENDING: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// Since the start, of the pages of [`Over::Unguarded`]: those the next
/// kernel asking over them found written (the copy takes them again), and
/// those it did not (the copy keeps what differs: a defect).
pub(crate) static BACK_UNGUARDED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// A kernel asking over pages `[first, last)` found `found` (addresses) written.
fn kernel_answered(first: usize, last: usize, found: &[usize]) {
    if !back_check() {
        return;
    }
    let mut pending = BACK_PENDING.lock().unwrap_or_else(|e| e.into_inner());
    if pending.is_empty() {
        return;
    }
    pending.retain(|&page| {
        if page < first || page >= last {
            return true;
        }
        let written = found.iter().any(|&at| at / PAGE == page);
        BACK_UNGUARDED[usize::from(!written)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        false
    });
}

/// Disjoint, non-touching half-open runs of clean blocks.
struct CleanBlocks(BTreeMap<usize, usize>);

impl CleanBlocks {
    fn insert(&mut self, mut from: usize, mut to: usize) {
        if from >= to {
            return;
        }
        if let Some((&start, &end)) = self.0.range(..=from).next_back()
            && end >= from
        {
            if end >= to {
                return;
            }
            from = start;
        }
        while let Some((&start, &end)) = self.0.range(from..=to).next() {
            self.0.remove(&start);
            to = to.max(end);
        }
        self.0.insert(from, to);
    }

    fn remove(&mut self, block: usize) {
        let Some((&from, &to)) = self.0.range(..=block).next_back() else {
            return;
        };
        if block >= to {
            return;
        }
        if from < block {
            self.0.insert(from, block);
        } else {
            self.0.remove(&from);
        }
        if block + 1 < to {
            self.0.insert(block + 1, to);
        }
    }

    /// Takes blocks `[from, to)` out.
    fn remove_range(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        if let Some((&start, &end)) = self.0.range(..from).next_back()
            && end > from
        {
            self.0.insert(start, from);
            if end > to {
                self.0.insert(to, end);
                return;
            }
        }
        let inside: Vec<(usize, usize)> = self.0.range(from..to).map(|(&s, &e)| (s, e)).collect();
        for (start, end) in inside {
            self.0.remove(&start);
            if end > to {
                self.0.insert(to, end);
            }
        }
    }

    fn uncovered(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let mut next = self.0.range(..=from).next_back().map_or(from, |(_, &end)| from.max(end));
        let mut missing = Vec::new();
        if next >= to {
            return missing;
        }
        for (&start, &end) in self.0.range(from + 1..to) {
            if next < start {
                missing.push((next, start));
            }
            next = end;
        }
        if next < to {
            missing.push((next, to));
        }
        missing
    }
}

/// Whether a block once guarded stays known as view memory while its
/// guard is open (`VICTORY_VIEW_KNOWN=1`; off by default).
///
/// A range whose first block the title wrote since the last check has
/// that block neither read-only nor noted absent, so nothing says it is a
/// view: the kernel's write watch is asked and fails, and then Windows is
/// asked what the memory is (`is_view`), a walk of the page tables to the
/// end of the pages that share the first one's state. In runs 222 and 225
/// that was 338-450 failed askings a frame and 6.5-7.8 ms a frame of the
/// submitting thread in `VirtualQuery`. With the switch the block is
/// remembered from the time it was first made read-only until the runtime
/// maps or unmaps it ([`title_memory_remapped`]).
fn view_known() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let run = *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_KNOWN").is_some_and(|v| v == "1"));
    // May turn within a run (`crate::ab`): off, the blocks remembered are
    // not read, and a remapping forgets its own (`remapped_viewed`), so
    // none stays from before a turn back on; on, they are learnt again.
    crate::ab::flag(crate::ab::Key::ViewKnown, run)
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BlockSet::new(),
    clean: CleanBlocks(BTreeMap::new()),
    absent: BTreeSet::new(),
    aliased: BTreeSet::new(),
    watchers: BTreeMap::new(),
    viewed: BlockSet::new(),
    remapped: CleanBlocks(BTreeMap::new()),
    verified: BTreeMap::new(),
    holes: BTreeSet::new(),
    unsure: BTreeSet::new(),
    opened: Vec::new(),
});

/// Whether host copies are not refreshed again over ranges that nothing
/// wrote since their last refresh (`VICTORY_SKIP_UNCHANGED=1`; off by
/// default). Only guarded view memory counts: every write to it opens a
/// guard first (the title's fault, a kernel write, a remapping, our own
/// writes), and that opening takes the block's bytes out of what each
/// copy holds current. After upstream Marvisak/VIctory#2136 (r4tedx).
pub(crate) fn skip_unchanged() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SKIP_UNCHANGED").is_some_and(|v| v == "1"))
}

/// Of one host copy ([`skip_unchanged`]): the title ranges (addresses, in
/// no order) refreshed into it and not written since, and a count of the
/// takings-out, so that a refresh a write overlapped notes nothing.
#[derive(Default)]
pub(crate) struct Unchanged {
    inner: Mutex<(u64, Vec<(usize, usize)>)>,
}

impl Unchanged {
    fn lock(&self) -> std::sync::MutexGuard<'_, (u64, Vec<(usize, usize)>)> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether `[at, at + len)` is held current.
    pub(crate) fn covers(&self, at: usize, len: usize) -> bool {
        let end = at.saturating_add(len);
        self.lock().1.iter().any(|&(a, b)| a <= at && end <= b)
    }

    /// `[from, to)` was refreshed, begun at count `seen`
    /// ([`watch_unchanged`]): held current unless something was taken out
    /// meanwhile.
    pub(crate) fn note(&self, from: usize, to: usize, seen: u64) {
        let mut inner = self.lock();
        if inner.0 != seen || from >= to || inner.1.iter().any(|&(a, b)| a <= from && to <= b) {
            return;
        }
        if inner.1.len() >= 64 {
            inner.1.clear();
        }
        inner.1.push((from, to));
    }

    /// Nothing is held current any more.
    pub(crate) fn clear(&self) {
        let mut inner = self.lock();
        inner.0 += 1;
        inner.1.clear();
    }

    /// `[from, to)` was written: out of every range held current.
    fn take_out(&self, from: usize, to: usize) {
        let mut inner = self.lock();
        inner.0 += 1;
        let ranges = std::mem::take(&mut inner.1);
        for (a, b) in ranges {
            if b <= from || to <= a {
                inner.1.push((a, b));
                continue;
            }
            if a < from {
                inner.1.push((a, from));
            }
            if to < b {
                inner.1.push((to, b));
            }
        }
    }
}

/// Blocks per bucket of [`Guards::watchers`] (64: 4 MiB): a copy is
/// registered once per bucket its refreshes cover, not once per block (the
/// 174 MB heap's 2,784 blocks, ten times a frame, under the guards' lock
/// in run P5-0).
const BUCKET_SHIFT: usize = 6;

/// Whether any copy was ever registered ([`unchanged_gone`] has nothing to
/// do before).
static WATCHING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The copies watching `block`'s bucket let go of its bytes. They stay
/// registered: the other blocks of the bucket are still theirs.
fn invalidate(g: &mut Guards, block: usize) {
    take_out_range(g, block * BLOCK, (block + 1) * BLOCK);
}

/// The copies registered on the buckets of `[from, to)` (title
/// addresses) let go of those bytes.
fn take_out_range(g: &Guards, from: usize, to: usize) {
    if from >= to || g.watchers.is_empty() {
        return;
    }
    let (first, last) = ((from / BLOCK) >> BUCKET_SHIFT, (to.div_ceil(BLOCK) + (1 << BUCKET_SHIFT) - 1) >> BUCKET_SHIFT);
    for (_, list) in g.watchers.range(first..last) {
        for w in list {
            w.take_out(from, to);
        }
    }
}

/// Registers `w` on the buckets of blocks `[first, last)`, once each.
fn watch_buckets(g: &mut Guards, first: usize, last: usize, w: &std::sync::Arc<Unchanged>) {
    WATCHING.store(true, std::sync::atomic::Ordering::Relaxed);
    for bucket in first >> BUCKET_SHIFT..(last + (1 << BUCKET_SHIFT) - 1) >> BUCKET_SHIFT {
        let list = g.watchers.entry(bucket).or_default();
        if !list.iter().any(|x| std::sync::Arc::ptr_eq(x, w)) {
            list.push(w.clone());
        }
    }
}

/// Before a refresh of `[start, start + len)` into the copy that `w` is
/// of: if every block of the range is guarded now, `w` watches them (any
/// opening from here on takes their bytes out of it) and the count to
/// hand to [`Unchanged::note`] after the refresh; `None` otherwise
/// (write-watched memory is never held current).
#[cfg(test)]
pub(crate) fn watch_unchanged(start: usize, len: usize, w: &std::sync::Arc<Unchanged>) -> Option<u64> {
    let (first, last) = (start / BLOCK, start.saturating_add(len).div_ceil(BLOCK));
    if first >= last {
        return None;
    }
    let mut g = guards(crate::lockmeter::Site::WatchUnchanged);
    if g.protected.count_in(first..last) != last - first {
        return None;
    }
    watch_buckets(&mut g, first, last, w);
    // Read under the guards' lock, which every taking-out holds.
    Some(w.lock().0)
}

thread_local! {
    /// While a host copy's refresh runs on this thread
    /// ([`refresh_watching`]): the copy's [`Unchanged`], the count it had
    /// when [`guard`] first registered it (`None`: not yet, or a guard
    /// found a block of its range open), and the block ranges registered.
    static REFRESHING: std::cell::RefCell<Option<(std::sync::Arc<Unchanged>, Option<u64>, bool, Vec<(usize, usize)>)>> =
        const { std::cell::RefCell::new(None) };
}

/// Runs a refresh of `[from, to)` (title addresses) of the copy that `w`
/// is of; with [`skip_unchanged`], every [`guard`] the refresh makes
/// registers `w` on the blocks it has just guarded, under the same lock,
/// so that a write after the guard (before or during the copy) takes the
/// bytes out again. Afterwards the range is noted current if every block
/// of it was registered so and nothing was taken out since. Registered
/// before the refresh, as it was, a copy whose blocks had been written was
/// never noted until a refresh after a quiet one (14 % of asks skipped in
/// run P4-0).
pub(crate) fn refresh_watching<R>(w: &std::sync::Arc<Unchanged>, from: usize, to: usize, refresh: impl FnOnce() -> R) -> R {
    REFRESHING.with(|r| *r.borrow_mut() = Some((w.clone(), None, false, Vec::new())));
    let out = refresh();
    let taken = REFRESHING.with(|r| r.borrow_mut().take());
    if let Some((_, Some(seen), false, mut registered)) = taken
        && from < to
    {
        registered.sort_unstable();
        let (first, last) = (from / BLOCK, to.div_ceil(BLOCK));
        let mut next = first;
        for (a, b) in registered {
            if a > next {
                break;
            }
            next = next.max(b);
        }
        if next >= last {
            w.note(from, to, seen);
        }
    }
    out
}

/// In [`guard`], holding the guards: registers the refreshing copy, if
/// any, on blocks `[first, last)` if all of them are guarded now (all
/// clean: a clean block is a guarded one, one lookup with #979's runs).
fn register_refreshing(g: &mut Guards, first: usize, last: usize) {
    REFRESHING.with(|r| {
        let mut r = r.borrow_mut();
        let Some((w, seen, failed, registered)) = r.as_mut() else {
            return;
        };
        if *failed || first >= last {
            return;
        }
        if !g.clean.uncovered(first, last).is_empty() {
            *failed = true;
            return;
        }
        watch_buckets(g, first, last, w);
        if seen.is_none() {
            // Under the guards' lock, which every taking-out holds.
            *seen = Some(w.lock().0);
        }
        registered.push((first, last));
    });
}

/// A copy watching `[start, start + len)` goes.
pub(crate) fn unwatch_unchanged(start: usize, len: usize, w: &std::sync::Arc<Unchanged>) {
    let (first, last) = (start / BLOCK, start.saturating_add(len).div_ceil(BLOCK));
    let mut g = guards(crate::lockmeter::Site::UnwatchUnchanged);
    let buckets: Vec<usize> =
        g.watchers.range(first >> BUCKET_SHIFT..(last + (1 << BUCKET_SHIFT) - 1) >> BUCKET_SHIFT).map(|(&b, _)| b).collect();
    for bucket in buckets {
        if let Some(list) = g.watchers.get_mut(&bucket) {
            list.retain(|x| !std::sync::Arc::ptr_eq(x, w));
            if list.is_empty() {
                g.watchers.remove(&bucket);
            }
        }
    }
}

/// Title memory `[at, at + len)` was written by a way that opens no guard
/// (our own views, the journal's own notes): out of every copy's current
/// ranges.
fn unchanged_gone(at: usize, len: usize) {
    if len == 0 || !WATCHING.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let g = guards(crate::lockmeter::Site::UnchangedGone);
    take_out_range(&g, at, at.saturating_add(len));
}


/// The guards, locked for `site`; the waits and holds measured
/// (`crate::lockmeter`).
type GuardsHeld = crate::lockmeter::Held<Guards>;

fn guards(site: crate::lockmeter::Site) -> GuardsHeld {
    use crate::framelog::{T, add};
    let mut g = crate::lockmeter::lock(&GUARDS, site);
    let runs = guard_intervals();
    if g.protected.runs_mode != runs {
        let turning = std::time::Instant::now();
        g.protected.store_as(runs);
        g.viewed.store_as(runs);
        add(T::GuardsStoreUs, turning.elapsed().as_micros() as u64);
    }
    g
}

/// Whether the write faults' new ways may be in use in this session
/// (`VICTORY_OPEN_ONE_VIEW`, `VICTORY_GUARD_REST`): either switch on in the
/// environment, or named by the A/B (`OPEN_ONE_VIEW`, `GUARD_REST`), then in
/// both arms. A block opened alone or left resting in one arm is still so
/// in the next, so the guards keep the new ways' checks (every other view
/// covered before a block is noted clean, the other views cleared before
/// one is opened) for the whole session, whichever arm runs. A session that
/// never names either goes the old way exactly ([`open_classic`],
/// [`protect_classic`], [`views_mapped_classic`]). Read once.
fn guards_kept() -> bool {
    #[cfg(test)]
    if OPEN_ONE_IN_TEST.get().is_some() || REST_IN_TEST.get().is_some() {
        return true;
    }
    static KEPT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *KEPT.get_or_init(|| {
        std::env::var_os("VICTORY_OPEN_ONE_VIEW").is_some_and(|v| v == "1")
            || std::env::var("VICTORY_GUARD_REST").is_ok_and(|v| v.trim().parse::<u64>().is_ok_and(|n| n != 0))
            || std::env::var("VICTORY_AB").is_ok_and(|ab| {
                let ab = ab.to_ascii_uppercase();
                ab.contains("OPEN_ONE_VIEW") || ab.contains("GUARD_REST")
            })
    })
}

/// Whether the bits of the light refresh are kept ([`PUBLISHED`]):
/// `VICTORY_LIGHT_REFRESH=1`, or the A/B names `LIGHT_REFRESH` (then in both
/// arms, so that the arm that asks them finds them true whatever was
/// written in the arm before). Read once.
fn light_kept() -> bool {
    static KEPT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *KEPT.get_or_init(|| {
        light_run()
            || std::env::var("VICTORY_AB").is_ok_and(|ab| ab.to_ascii_uppercase().contains("LIGHT_REFRESH"))
    })
}

/// `VICTORY_LIGHT_REFRESH=1`.
fn light_run() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIGHT_REFRESH").is_some_and(|v| v == "1"))
}

/// Whether an asking about view memory is answered from the bits of its
/// blocks where they can answer (`VICTORY_LIGHT_REFRESH=1`; off by
/// default; the A/B may turn it, `LIGHT_REFRESH`).
///
/// A frame asks the guards about view memory 2,000 to 3,700 times, to hear
/// for 98-99 % that nothing was written; every such asking takes the
/// guards' lock and reads the range's clean blocks: in the yard 1,666
/// takings a frame of `guard_locked`, 30.4 ms held (S-1, ac03e67). With the
/// switch a range whose blocks all have their bit ([`PUBLISHED`]) is
/// answered from the journal alone ([`writes_into`], [`nothing_new`]), and
/// its first block is known for view memory without the lock
/// ([`known_view`]).
///
/// What changes in the answers: nothing. A range is known by its bits
/// exactly while the guards would find it all clean, or clean but for
/// holes noted for good ([`told`]); only who is asked changes. Not with
/// `VICTORY_SKIP_UNCHANGED` (the guards register the copy being refreshed
/// there, [`register_refreshing`], which the bits would skip).
///
/// From fxceless/victory-fork 94ed363 (fxceless), itself after
/// avatarmanz/VIctory-7900XTX 2aa8134b and 74045e46 (avatarmanz), and
/// 689df313 for the other views' bits; its `alt` left out: the A/B turns it
/// here.
pub(crate) fn light() -> bool {
    light_kept() && crate::ab::flag(crate::ab::Key::LightRefresh, light_run()) && !skip_unchanged()
}

/// Since they were last taken ([`light_asks`]), with the light refresh
/// kept: askings of view memory answered by the bits of their blocks,
/// askings that went to the guards, and holes known by their bits that
/// Windows said were filled with nobody telling ([`look_at_holes`]).
static LIGHT_ASKS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// [`LIGHT_ASKS`], taken.
fn light_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| LIGHT_ASKS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// For the frame log's reading of `frames` frames: the light refresh's and
/// [`absent_once`]'s counts a frame, since the last line.
pub(crate) fn askings_line(frames: f64) -> String {
    let f = frames.max(1.0);
    let [bits, guards, filled] = light_asks();
    let [asked, noted, unsure] = absent_looks();
    format!(
        "askings of view memory, a frame: {:.0} answered by the bits of their blocks, {:.0} through the guards (LIGHT_REFRESH {}), \
         {filled} holes found filled with nobody telling; blocks noted uncommitted: {:.0} questions to Windows, {:.0} answered \
         from their note, {:.0} new notes in no placeholder (they live 100 ms with told) (ABSENT_ONCE {}, \
         VICTORY_ABSENT_MS {})",
        bits as f64 / f,
        guards as f64 / f,
        if light() { "on" } else if light_kept() { "off, bits kept" } else { "off" },
        asked as f64 / f,
        noted as f64 / f,
        unsure as f64 / f,
        if absent_once() { "on" } else { "off" },
        match absent_keep_ms() {
            u64::MAX => "told".to_owned(),
            ms => format!("{ms}"),
        }
    )
}

/// The words of a piece of [`Bits`]: the bits of 4 GiB.
const BITS_WORDS: usize = 1 << 10;
/// The pieces of [`Bits`]: 128 TiB, what a process has.
const BITS_PIECES: usize = 1 << 15;

/// A bit per 64 KiB block of the address space, read and cleared with no
/// lock; in pieces made when first needed and never let go.
struct Bits([std::sync::atomic::AtomicPtr<BitsPiece>; BITS_PIECES]);

/// The bits of 4 GiB.
type BitsPiece = [std::sync::atomic::AtomicU64; BITS_WORDS];

impl Bits {
    const fn new() -> Bits {
        Bits([const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; BITS_PIECES])
    }

    /// The piece that holds word `word`, if it was ever made.
    fn piece(&self, word: usize) -> Option<&BitsPiece> {
        let piece = self.0.get(word / BITS_WORDS)?.load(std::sync::atomic::Ordering::Acquire);
        // SAFETY: null, or a piece made by `set`, which is never freed.
        unsafe { piece.as_ref() }
    }

    /// Sets `block`'s bit (nothing past the address space).
    fn set(&self, block: usize) {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, SeqCst};
        let word = block / 64;
        let Some(place) = self.0.get(word / BITS_WORDS) else {
            return;
        };
        if place.load(Acquire).is_null() {
            let made: *mut BitsPiece = Box::into_raw(Box::new([const { std::sync::atomic::AtomicU64::new(0) }; BITS_WORDS]));
            if place.compare_exchange(std::ptr::null_mut(), made, AcqRel, Acquire).is_err() {
                // SAFETY: made above and shown to nobody: another thread's
                // piece is in the place.
                drop(unsafe { Box::from_raw(made) });
            }
        }
        if let Some(piece) = self.piece(word) {
            piece[word % BITS_WORDS].fetch_or(1u64 << (block % 64), SeqCst);
        }
    }

    /// Clears `block`'s bit: ahead of what makes the block writable, and
    /// seen by whoever looks after it.
    fn clear(&self, block: usize) {
        let word = block / 64;
        if let Some(piece) = self.piece(word) {
            piece[word % BITS_WORDS].fetch_and(!(1u64 << (block % 64)), std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Whether the bits of all of `[first, last)` are set (not so of none).
    fn all(&self, first: usize, last: usize) -> bool {
        let mut block = first;
        while block < last {
            let word = block / 64;
            let until = ((word + 1) * 64).min(last);
            let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
            let Some(piece) = self.piece(word) else {
                return false;
            };
            if piece[word % BITS_WORDS].load(std::sync::atomic::Ordering::SeqCst) & mask != mask {
                return false;
            }
            block = until;
        }
        first < last
    }

    /// Whether any bit of `[first, last)` is set.
    fn any(&self, first: usize, last: usize) -> bool {
        let mut block = first;
        while block < last {
            let word = block / 64;
            let until = ((word + 1) * 64).min(last);
            let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
            if let Some(piece) = self.piece(word)
                && piece[word % BITS_WORDS].load(std::sync::atomic::Ordering::SeqCst) & mask != 0
            {
                return true;
            }
            block = until;
        }
        false
    }
}

/// Blocks currently guarded (`PAGE_READONLY`) or about to be made so: set
/// ahead of `VirtualProtect(..., PAGE_READONLY)` and cleared after
/// `VirtualProtect(..., PAGE_READWRITE)` so lock-free fast-paths in the
/// fault handler and `views_mapped` never miss a guarded block.
static PROTECTED: Bits = Bits::new();

/// The blocks an asking can answer for from the journal alone. A block has
/// its bit only while both of these hold: it is read-only by our hand and
/// noted clean (`Guards::clean`), or it is a hole noted for good that lies
/// in a placeholder reservation (`Guards::holes`); and the journal holds
/// every write to it that an asking found.
///
/// Why that is so. A bit is set in one place, [`publish`]: by the asking
/// that found the block written and had it made clean, after it noted that
/// in the journal and with the journal's lock still held, under the
/// guards' lock, and only if the block is clean at that moment. It is
/// cleared under the guards' lock before the block's write access is given
/// back (`open_here`, both of its ways, which the write fault's handler,
/// [`writing`], [`title_memory_written`] and [`views_mapped`] all go
/// through; [`open`] clears every other view of the pages first), and when
/// the runtime tells that what the block shows is new
/// ([`title_memory_remapped`], [`title_memory_committed`]). Clearing does
/// nothing where no bit was ever set, so it is not behind the switch.
///
/// Who asks the bits holds the journal's lock from before the look until
/// it has read the journal ([`published`]), and nobody sets a bit without
/// that lock: a block opened and guarded again between the look and the
/// reading cannot have its bit back in time to be taken for unwritten. A
/// write that comes while the bits are looked at is no different from one
/// that comes just after an asking: its block's bit is gone, and the next
/// asking goes to the guards and finds the block open.
///
/// Our own stores through a view of our own (`crate::inplace::write_own`)
/// open no guard and take no bit: they are noted in the journal
/// ([`wrote_own`]), which is where an asking answered by the bits looks.
static PUBLISHED: Bits = Bits::new();

/// An asking of view memory `[start, end)` found the pages `written` (page
/// numbers) written, had the guards make their blocks read-only and clean
/// ([`guard`]) and has noted them in the journal, whose lock it holds
/// still (`_noted`): the blocks of them that are clean now are known by
/// their bits from here on ([`PUBLISHED`]). With `holes` ([`told`]), so
/// are the blocks of the range noted uncommitted for good that lie in a
/// placeholder reservation; Windows is asked that once for each
/// ([`placeholder`]).
///
/// The blocks an asking leaves clean that have no bit yet are the ones it
/// reports written: every other clean block was left clean by an asking
/// before it, which did this.
fn publish(_noted: &Journal, written: &[usize], start: usize, end: usize, holes: bool) {
    let mut g = guards(crate::lockmeter::Site::Other);
    let g = &mut *g;
    for &page in written {
        let block = page * PAGE / BLOCK;
        // (Not one that a write has opened since the guards were asked,
        // nor one that could not be made read-only.)
        if g.clean.uncovered(block, block + 1).is_empty() {
            PUBLISHED.set(block);
        }
    }
    if !holes {
        return;
    }
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // The notes of the range not looked at yet. Both sets are read in
    // block order, and every block of `holes` has a note.
    let fresh: Vec<usize> = {
        let mut looked = g.holes.range(first..last).copied().peekable();
        g.verified
            .range(first..last)
            .map(|(&block, _)| block)
            .filter(|&block| {
                while looked.next_if(|&hole| hole < block).is_some() {}
                looked.next_if_eq(&block).is_none()
            })
            .collect()
    };
    for block in fresh {
        g.holes.insert(block);
        // (Asked now, not when the note was made: a view the runtime is
        // mapping there, and has not told of yet, is no placeholder.)
        if g.absent.contains(&block) && placeholder(block) {
            PUBLISHED.set(block);
        }
    }
}

/// Whether every block of `[start, end)` has its bit ([`PUBLISHED`]): what
/// [`writes_into`] would answer of the range is in the journal. `_held`:
/// the journal's lock, which the caller holds from before this look until
/// it has read the journal (no bit is set meanwhile: [`publish`]).
fn published(_held: &Journal, start: usize, end: usize) -> bool {
    PUBLISHED.all(start / BLOCK, end.div_ceil(BLOCK))
}

/// Whether a block noted uncommitted stays noted until the runtime tells
/// of it (`VICTORY_ABSENT_ONCE=1` with `VICTORY_ABSENT_MS=told`): then, and
/// only then, can such a block be known by its bit ([`publish`]). With
/// notes that grow old, or none, every asking of a range that holds a hole
/// goes to the guards, which put the question to Windows as they do.
fn told() -> bool {
    absent_once() && absent_keep_ms() == u64::MAX
}

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    /// The kernel's question about a process's memory, `(process, address,
    /// class of the answer, answer out, its room, bytes answered out)`: an
    /// NTSTATUS, negative when it cannot answer. It changes nothing.
    fn NtQueryVirtualMemory(
        process: *mut std::ffi::c_void,
        at: *const std::ffi::c_void,
        class: u32,
        out: *mut std::ffi::c_void,
        len: usize,
        returned: *mut usize,
    ) -> i32;
}

/// Whether `block` lies in a placeholder reservation: memory in which pages
/// come to be only by mapping a view, or making an allocation, in its
/// place, which the runtime does and tells of ([`title_memory_remapped`],
/// [`title_memory_committed`]). An ordinary reservation can be committed
/// by anybody, and free memory taken by anybody, with no word to us:
/// `VirtualQuery` does not tell the three apart, this question does.
fn placeholder(block: usize) -> bool {
    /// `MEMORY_REGION_INFORMATION` (ntddk); only the kinds are read.
    #[repr(C)]
    #[derive(Default)]
    #[allow(dead_code)]
    struct Region {
        allocation_base: usize,
        allocation_protect: u32,
        region_type: u32,
        region_size: usize,
        commit_size: usize,
        partition_id: usize,
        node_preference: usize,
    }
    // (`MemoryRegionInformationEx`; class 3 answers with the kind of memory
    // in that word, not the flags.)
    const MEMORY_REGION_INFORMATION_EX: u32 = 7;
    const PLACEHOLDER_RESERVATION: u32 = 1 << 8;
    let mut region = Region::default();
    // SAFETY: the answer's room and its size; the kernel checks the address.
    let status = unsafe {
        NtQueryVirtualMemory(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            (block * BLOCK) as *const _,
            MEMORY_REGION_INFORMATION_EX,
            (&raw mut region).cast(),
            size_of::<Region>(),
            std::ptr::null_mut(),
        )
    };
    status >= 0 && region.region_type & PLACEHOLDER_RESERVATION != 0
}

/// The note that `block` is uncommitted is gone, or is true no longer: the
/// block is not known by its bit any more ([`publish`]).
fn forget_hole(g: &mut Guards, block: usize) {
    g.unsure.remove(&block);
    if g.holes.remove(&block) {
        PUBLISHED.clear(block);
    }
}

/// How long a note kept for good lives for a hole that lies in no
/// placeholder reservation ([`note_life`]), in milliseconds.
const UNSURE_MS: u64 = 100;

/// How long the note that `block` is uncommitted lives, of `keep`
/// ([`absent_once`]). With the light refresh a note kept for good is kept
/// so only where pages cannot come to be without the runtime's word: in a
/// placeholder reservation, whose holes are known by their bits. A hole
/// that [`publish`] looked at and found in an ordinary reservation or in
/// free memory, which anybody can commit or take with no word to us, is
/// asked of Windows again when its note is [`UNSURE_MS`] old. (Without the
/// light refresh no hole is looked at.) The same with or without the
/// light refresh: a note kept for good of a block in no placeholder
/// reservation (`Guards::unsure`) lives [`UNSURE_MS`].
fn note_life(g: &Guards, block: usize, keep: u64) -> u64 {
    if keep == u64::MAX
        && (g.unsure.contains(&block) || g.holes.contains(&block) && !PUBLISHED.all(block, block + 1))
    {
        UNSURE_MS
    } else {
        keep
    }
}

/// Holes asked of Windows again at each submission ([`look_at_holes`]): a
/// question is a microsecond or so, and the submitting thread is the
/// frame's limit.
const HOLES_A_LOOK: usize = 16;

/// With the light refresh: `most` of the holes known by their bits are
/// asked of Windows again, the next ones each time and round again, in
/// case one was filled with nobody telling ([`title_memory_remapped`],
/// [`title_memory_committed`]), which this title is not known to do. One
/// that was loses its bit and its note, so that the next asking of it goes
/// to the guards, which find it; counted, and said once. One whose
/// reservation is a placeholder no longer (let go, or made anew) loses its
/// bit and keeps its note.
///
/// Not while the runtime changes a mapping: a view it has mapped and not
/// told of yet is no hole filled behind our back, and forgotten here it
/// could be made read-only before the runtime's word, which forgets the
/// guard and leaves the protection.
fn look_at_holes(most: usize) {
    static FROM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let Some(_still) = title_memory_reading_now() else {
        return;
    };
    let part: Vec<usize> = {
        let g = guards(crate::lockmeter::Site::Other);
        if g.holes.is_empty() {
            return;
        }
        let from = FROM.load(std::sync::atomic::Ordering::Relaxed);
        let part: Vec<usize> = g.holes.range(from..).take(most).copied().collect();
        FROM.store(part.last().map_or(0, |last| last + 1), std::sync::atomic::Ordering::Relaxed);
        part
    };
    for block in part {
        // (Not the ones looked at that lie in no placeholder: they have no
        // bit, and their notes are the guards' to keep.)
        if !PUBLISHED.all(block, block + 1) {
            continue;
        }
        let filled = block >= uncommitted_end(block);
        if !filled && placeholder(block) {
            continue;
        }
        let mut g = guards(crate::lockmeter::Site::Other);
        // (Told of since the part was taken: forgotten already.)
        if !g.holes.contains(&block) {
            continue;
        }
        PUBLISHED.clear(block);
        if filled {
            g.holes.remove(&block);
            g.verified.remove(&block);
            LIGHT_ASKS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                tracing::warn!(
                    address = format_args!("{:#x}", block * BLOCK),
                    "a hole of view memory was filled with nobody telling: host copies of it held old bytes until now (VICTORY_LIGHT_REFRESH; said once, counted with the light refresh's askings)"
                );
            });
        }
    }
}

/// Whether nothing of `[start, end)` was written that a copy does not
/// hold, known with nothing asked of the guards or the kernel: `copied` is
/// the copy's note, for each page from page `base` on, of the asking after
/// which it last took the page (0: never), as `crate::shadow` keeps it.
/// True only if [`writes_into`] would answer from the journal (the range
/// was asked ahead, or its blocks are known by their bits) and the answer
/// would have the copy take no page. `false` if something would have to be
/// asked, and always without the light refresh ([`light`]).
pub(crate) fn nothing_new(start: usize, end: usize, base: usize, copied: &[u64]) -> bool {
    light() && nothing_noted(start, end, base, copied)
}

/// [`nothing_new`], whatever the switch says.
fn nothing_noted(start: usize, end: usize, base: usize, copied: &[u64]) -> bool {
    if start >= end {
        return false;
    }
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let Some(taken) = first.checked_sub(base).and_then(|from| copied.get(from..last - base)) else {
        return false;
    };
    // A page never taken is taken at the next refresh (so is, at every
    // refresh, a page the copy's refreshes only ever hold in part).
    if taken.contains(&0) {
        return false;
    }
    let j = unbusy(crate::lockmeter::lock_journal(&JOURNAL), &[(first, last)]);
    // The bits first, then the journal, both under its lock ([`published`]).
    let by_bits = !j.ahead.iter().any(|&(a, b)| a <= first && last <= b);
    if by_bits && !published(&j, first * PAGE, last * PAGE) {
        return false;
    }
    let same = j.written.range(first..last).all(|(&page, &write)| write <= taken[page - first]);
    if same && by_bits {
        LIGHT_ASKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    same
}

/// Whether a block of view memory that Windows said is uncommitted is not
/// asked about again for [`absent_keep_ms`] (`VICTORY_ABSENT_ONCE=1`; off
/// by default; the A/B may turn it, `ABSENT_ONCE`).
///
/// A block noted absent is never "clean", so a range that holds one is
/// walked block by block at every asking, and every such block is a
/// question to Windows (`uncommitted_end`): the title's unmapped pages are
/// placeholders of 64 KiB each, so one question answers for one block. The
/// fork's profile of run 249 had 5.7 ms a frame of the submitting thread in
/// `VirtualQuery` under the guards, with their lock held, which the title's
/// threads wait for in their write faults.
///
/// What the runtime maps there it tells ([`title_memory_remapped`]), and
/// what it commits for the title too ([`title_memory_committed`]): either
/// forgets the block's note at once, with or without the switch. What
/// changes: memory that appears at such an address without the runtime's
/// word is seen up to [`absent_keep_ms`] later than before (never, with
/// `told`, until the runtime says something of it); until then its copy
/// holds zeros, as it did while nothing was there.
///
/// From fxceless/victory-fork 44bff61 and 6d20eed (its `alt` left out: the
/// A/B turns it here).
fn absent_once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let run = *ON.get_or_init(|| std::env::var_os("VICTORY_ABSENT_ONCE").is_some_and(|v| v == "1"));
    crate::ab::flag(crate::ab::Key::AbsentOnce, run)
}

/// How long a block found uncommitted is not asked about again
/// ([`absent_once`]): `VICTORY_ABSENT_MS`, 100 by default (about a frame),
/// 1 to 1000; `told`: until the runtime tells of the block
/// ([`title_memory_remapped`], [`title_memory_committed`]). Read once.
fn absent_keep_ms() -> u64 {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| match std::env::var("VICTORY_ABSENT_MS").as_deref().map(str::trim) {
        // The fork's run 279 with 100 ms: 600,000 to 690,000 questions in
        // twenty seconds without the notes, still 357,000 to 604,000 with
        // them, a frame being as long as a note lives.
        Ok("told") => u64::MAX,
        Ok(v) => v.parse::<u64>().map_or(100, |n| n.clamp(1, 1000)),
        Err(_) => 100,
    })
}

/// Since they were last taken ([`absent_looks`]): questions to Windows
/// about blocks noted absent, and such blocks answered from their note
/// ([`absent_once`]). Counted with the switch off too.
static ABSENT_LOOKS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// [`ABSENT_LOOKS`], taken ([`askings_line`]).
fn absent_looks() -> [u64; 3] {
    [0, 1, 2].map(|i| ABSENT_LOOKS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// Milliseconds since the guards' first asking: the clock of
/// [`absent_once`]'s notes.
fn guard_ms() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    let keep = if absent_once() { absent_keep_ms() } else { 0 };
    guard_at(start, end, if keep != 0 { guard_ms() } else { 0 }, keep)
}

/// [`guard`] at the moment `now` ([`guard_ms`]). `keep`: for how many
/// milliseconds a block Windows said is uncommitted is not asked about
/// again ([`absent_once`]); 0: it is asked about at every asking.
fn guard_at(start: usize, end: usize, now: u64, keep: u64) -> Option<Vec<usize>> {
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: 0.6 ms for the title's 174 MB heap, 16 times a frame, and most
    // of the 68 ms a frame of "guards" (run 58).
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let ask = *ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1"));
    if guard_intervals() && !ask {
        // One lock of the guards ([`guard_intervals`]): a range all clean,
        // nearly every asking, is done with under it; Windows is asked
        // whether the memory is a view only when the guards do not know,
        // with the lock let go. (The handler is registered first: a fault
        // being dispatched holds the loader's lock while its handler waits
        // for the guards'.)
        install();
        let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
        let mut g = guards(crate::lockmeter::Site::GuardClean);
        if first < last && g.clean.uncovered(first, last).is_empty() {
            register_refreshing(&mut g, first, last);
            return Some(Vec::new());
        }
        let known = g.protected.contains(&first) || g.absent.contains(&first) || (view_known() && g.viewed.contains(&first));
        if !known {
            drop(g);
            if !is_view(start, end) {
                return None;
            }
            g = guards(crate::lockmeter::Site::GuardLocked);
        }
        g.now_for(crate::lockmeter::Site::GuardLocked);
        return guard_locked(g, start, end, now, keep);
    }
    let known = !ask && {
        let g = guards(crate::lockmeter::Site::GuardKnown);
        let first = start / BLOCK;
        g.protected.contains(&first) || g.absent.contains(&first) || (view_known() && g.viewed.contains(&first))
    };
    if !known && !is_view(start, end) {
        return None;
    }
    install();
    guard_locked(guards(crate::lockmeter::Site::GuardLocked), start, end, now, keep)
}

/// [`guard`] with the guards locked, the range known to be a view; `now`
/// and `keep` as [`guard_at`] has them.
fn guard_locked(mut g: GuardsHeld, start: usize, end: usize, now: u64, keep: u64) -> Option<Vec<usize>> {
    let mut written = Vec::new();
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // Large heaps are usually clean (#979): one lookup when all of the
    // range is, else only the gaps between clean runs are walked.
    if g.clean.uncovered(first, last).is_empty() {
        // Nothing written, mapped or new since the range was last taken.
        register_refreshing(&mut g, first, last);
        return Some(written);
    }
    let rest = rest();
    let mut uncommitted_until = first;
    for (from, to) in g.clean.uncovered(first, last) {
        let mut run = None;
        for block in from..to {
            let mut open = true;
            if g.absent.contains(&block) {
                // Found uncommitted a moment ago ([`absent_once`]): Windows is
                // not asked again yet. What the runtime maps or commits there
                // it tells, and that takes the note away.
                // (With the light refresh, kept for good only in a placeholder
                // reservation: [`note_life`].)
                let life = note_life(&g, block, keep);
                let noted = life != 0 && g.verified.get(&block).is_some_and(|&at| now < at.saturating_add(life));
                if noted {
                    ABSENT_LOOKS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    open = false;
                } else {
                    // Ordinary commits need no runtime mapping notification. Reuse
                    // this asking's query for the following blocks in the same region.
                    if block >= uncommitted_until {
                        ABSENT_LOOKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        // Ahead of the question: a commit told after it is
                        // looked for ([`title_memory_committed`]).
                        if keep != 0 {
                            NOTED_ABSENT.store(true, std::sync::atomic::Ordering::SeqCst);
                        }
                        uncommitted_until = uncommitted_end(block);
                    }
                    open = block >= uncommitted_until;
                    if open {
                        g.absent.remove(&block);
                        g.verified.remove(&block);
                        forget_hole(&mut g, block);
                    } else if keep != 0 {
                        g.verified.insert(block, now);
                        // Kept for good only where the runtime's word is
                        // sure to come; asked once a block (a placeholder is
                        // made or let go only by the runtime, which tells).
                        if keep == u64::MAX && !g.unsure.contains(&block) && !placeholder(block) {
                            g.unsure.insert(block);
                            ABSENT_LOOKS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
            }
            if open {
                written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
                #[cfg(feature = "diagnostics")]
                if g.aliased.contains(&block) {
                    crate::copystat::add(crate::copystat::C::AliasPages, (BLOCK / PAGE) as u64);
                }
                if g.protected.contains(&block) && !guards_kept() {
                    // Written through the runtime, still read-only.
                    g.clean.insert(block, block + 1);
                } else if g.protected.contains(&block) {
                    // Read-only by way of another view of its page, written
                    // through the runtime, or left unclean because an alias was
                    // opened/resting (`VICTORY_OPEN_ONE_VIEW`/`VICTORY_GUARD_REST`):
                    // only note clean if every alias of its page is covered now.
                    if let Some(begin) = run.take() {
                        protect(&mut g, begin, block, rest);
                    }
                    if cover(&mut g, block, block + 1, rest) {
                        g.clean.insert(block, block + 1);
                    }
                    continue;
                } else if !g.rests(block, rest) {
                    run.get_or_insert(block);
                    continue;
                }
            }
            if let Some(begin) = run.take() {
                protect(&mut g, begin, block, rest);
            }
        }
        if let Some(begin) = run {
            protect(&mut g, begin, to, rest);
        }
    }
    register_refreshing(&mut g, first, last);
    Some(written)
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent. The same physical pages
/// at every other view of them are made read-only too: a write through
/// one of those is this memory written, and only a fault can say so.
/// `VICTORY_VIEW_GUARD=0` leaves the other views as they are, as before.
fn protect(g: &mut Guards, from: usize, to: usize, rest: std::time::Duration) {
    if !guards_kept() {
        protect_classic(g, from, to);
        return;
    }
    protect_here(g, from, to);
    cover(g, from, to, rest);
}

/// [`protect`] in a session that never names the write faults' new ways
/// ([`guards_kept`]): as before them (fx-main da215a9), the other views'
/// blocks made read-only with no look at whether each is covered.
fn protect_classic(g: &mut Guards, from: usize, to: usize) {
    protect_here(g, from, to);
    if !view_guard() {
        return;
    }
    let coalesced = guard_coalesce();
    let metered = crate::lockmeter::on().then(std::time::Instant::now);
    let mut aliases: Vec<usize> = Vec::new();
    for (base, bytes) in crate::inplace::views_between(from.saturating_mul(BLOCK), to.saturating_mul(BLOCK)) {
        let first = (base / BLOCK).max(from);
        let last = (base.saturating_add(bytes).div_ceil(BLOCK)).min(to);
        for block in first..last {
            let shown = crate::inplace::pages_of(block * BLOCK);
            // One view (nothing else to guard), or the page the title shows
            // wherever it has nothing (thousands of views): each view of
            // that one is guarded by itself.
            if shown.len() < 2 || shown.len() > MANY_VIEWS {
                continue;
            }
            #[cfg(feature = "diagnostics")]
            ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for alias in shown {
                let alias = alias / BLOCK;
                g.aliased.insert(alias);
                if alias != block {
                    if coalesced {
                        if !g.protected.contains(&alias) {
                            aliases.push(alias);
                        }
                    } else {
                        protect_alias(g, alias, std::time::Duration::ZERO);
                    }
                }
            }
        }
    }
    // The other views' blocks that follow one another are made read-only
    // in one call ([`guard_coalesce`]); a run the kernel refuses whole (it
    // spans regions it takes one at a time) block by block, as before.
    aliases.sort_unstable();
    aliases.dedup();
    for (first, last) in block_runs(&aliases) {
        if first + 1 == last || !refused_timed(|| protect_run(first, last)) {
            for block in first..last {
                protect_alias(g, block, std::time::Duration::ZERO);
            }
            continue;
        }
        // (The bits the new ways read, kept as they keep them.)
        for block in first..last {
            PROTECTED.set(block);
        }
        g.protected.extend(first..last);
        if view_known() {
            g.viewed.extend(first..last);
        }
    }
    if let Some(started) = metered {
        crate::lockmeter::alias_ns(started.elapsed().as_nanos() as u64);
    }
}

/// Ensures every other view of the physical pages in `[from, to)` is also
/// guarded (unless resting under `rest`). Returns `true` only if every
/// protected block in `[from, to)` has all its aliases covered.
fn cover(g: &mut Guards, from: usize, to: usize, rest: std::time::Duration) -> bool {
    if !view_guard() {
        return true;
    }
    let coalesced = guard_coalesce();
    let metered = crate::lockmeter::on().then(std::time::Instant::now);
    let mut all = true;
    let mut pending: Vec<(usize, usize)> = Vec::new();
    for (base, bytes) in crate::inplace::views_between(from.saturating_mul(BLOCK), to.saturating_mul(BLOCK)) {
        let first = (base / BLOCK).max(from);
        let last = (base.saturating_add(bytes).div_ceil(BLOCK)).min(to);
        for block in first..last {
            if !g.protected.contains(&block) {
                continue;
            }
            let shown = crate::inplace::pages_of(block * BLOCK);
            // One view (nothing else to guard), or the page the title shows
            // wherever it has nothing (thousands of views): each view of
            // that one is guarded by itself.
            if shown.len() < 2 || shown.len() > MANY_VIEWS {
                continue;
            }
            #[cfg(feature = "diagnostics")]
            ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut covered = true;
            for alias in shown {
                let alias = alias / BLOCK;
                g.aliased.insert(alias);
                if alias != block && !g.protected.contains(&alias) {
                    if g.rests(alias, rest) {
                        covered = false;
                    } else if coalesced {
                        pending.push((alias, block));
                    } else if !protect_alias(g, alias, rest) {
                        covered = false;
                    }
                }
            }
            if !covered {
                all = false;
                g.clean.remove(block);
            }
        }
    }
    // The other views' blocks that follow one another are made read-only
    // in one call ([`guard_coalesce`]); a run the kernel refuses whole (it
    // spans regions it takes one at a time) block by block, as before.
    if !pending.is_empty() {
        let mut unique: Vec<usize> = pending.iter().map(|&(a, _)| a).collect();
        unique.sort_unstable();
        unique.dedup();
        for (first, last) in block_runs(&unique) {
            if first + 1 == last {
                protect_alias(g, first, rest);
                continue;
            }
            for block in first..last {
                PROTECTED.set(block);
            }
            if !refused_timed(|| protect_run(first, last)) {
                for block in first..last {
                    if !g.protected.contains(&block) {
                        PROTECTED.clear(block);
                    }
                }
                for block in first..last {
                    protect_alias(g, block, rest);
                }
                continue;
            }
            g.protected.extend(first..last);
            if view_known() {
                g.viewed.extend(first..last);
            }
        }
        for (alias, owner) in pending {
            if !g.protected.contains(&alias) {
                all = false;
                g.clean.remove(owner);
            }
        }
    }
    if let Some(started) = metered {
        crate::lockmeter::alias_ns(started.elapsed().as_nanos() as u64);
    }
    all
}

/// Whether the guards' protections are asked of Windows over runs of
/// blocks that follow one another rather than block by block, where they
/// are given back (`open_here`) and where the other views of guarded pages
/// are guarded (`protect`) (`VICTORY_GUARD_COALESCE=1`; off by default). A
/// run Windows refuses is done block by block, as before; the blocks noted
/// and the copies told are the same. After fork #1059 piece 1 (UwUCammie);
/// the bench measured 4.06 -> 1.68 ms in the yard (block 2, B2-2). May turn
/// within a run (`crate::ab`): it is how one call is made.
fn guard_coalesce() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let run = *ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_COALESCE").is_some_and(|v| v == "1"));
    #[cfg(test)]
    match COALESCE_TEST.load(std::sync::atomic::Ordering::SeqCst) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    crate::ab::flag(crate::ab::Key::GuardCoalesce, run)
}

/// Tests that open the same blocks with [`guard_coalesce`] off (1) and on (2).
#[cfg(test)]
static COALESCE_TEST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The runs `[first, last)` of `blocks` (in order, no two the same).
fn block_runs(blocks: &[usize]) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &block in blocks {
        match runs.last_mut() {
            Some(run) if run.1 == block => run.1 += 1,
            _ => runs.push((block, block + 1)),
        }
    }
    runs
}

/// `call` (a protection asked over a run, [`guard_coalesce`]): its answer;
/// a refusal and its time go to the frame log.
fn refused_timed(call: impl FnOnce() -> bool) -> bool {
    let started = std::time::Instant::now();
    let done = call();
    if !done {
        crate::framelog::add(crate::framelog::T::CoalesceRefused, 1);
        crate::framelog::add(crate::framelog::T::CoalesceRefusedUs, started.elapsed().as_micros() as u64);
    }
    done
}

/// Makes blocks `[first, last)` read-only in one call; false if Windows refused.
fn protect_run(first: usize, last: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let mut old = 0;
    // SAFETY: title view memory; only its protection changes.
    unsafe { VirtualProtect((first * BLOCK) as *const _, (last - first) * BLOCK, PAGE_READONLY, &mut old) != 0 }
}

/// Views of one page past which its views are not guarded together.
pub(crate) const MANY_VIEWS: usize = 64;

/// Whether all the views of a guarded page are guarded together
/// (`VICTORY_VIEW_GUARD`).
fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

/// Diagnostics: blocks guarded together with their other views, blocks
/// opened because another view of theirs was written, and blocks opened
/// because a view of theirs was mapped; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIASES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Makes `block` read-only: another view of pages a copy has just taken
/// through its own view, so that a write through this one faults too. It is
/// not noted clean. The copies of title memory at this view have not taken
/// the pages: noted clean, the block would not be reported written at their
/// next check and they would keep the bytes from before the write (`guard`
/// reports a block that is read-only and not clean as written, then notes it
/// clean). A block that is clean stays so: had the pages been written since
/// its copies took them, `open` would have opened it.
fn protect_alias(g: &mut Guards, block: usize, rest: std::time::Duration) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    if g.protected.contains(&block) {
        return true;
    }
    if g.rests(block, rest) {
        return false;
    }
    // Ahead of the protection.
    PROTECTED.set(block);
    let mut old = 0;
    // SAFETY: title view memory; only its protection changes.
    if unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0 {
        g.protected.insert(block);
        if view_known() {
            g.viewed.insert(block);
        }
        true
    } else {
        PROTECTED.clear(block);
        false
    }
}

/// [`protect`] for one range, without the other views of its pages.
fn protect_here(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |g: &Guards, from: usize, to: usize| {
        for block in from..to {
            PROTECTED.set(block);
        }
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        let done = unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        };
        if !done {
            for block in from..to {
                if !g.protected.contains(&block) {
                    PROTECTED.clear(block);
                }
            }
        }
        done
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.insert(from, to);
        if back_check() {
            g.remapped.remove_range(from, to);
        }
        if view_known() {
            g.viewed.extend(from..to);
        }
    };
    if set(g, from, to) {
        note(g, from, to);
        return;
    }
    // Block by block; a block that is not committed answers for the whole
    // uncommitted run it begins: one question to Windows, no protection
    // asked for the rest of it (fork #987, cybx: 77 runs a frame were
    // refused whole in the world, with 10,264 questions and 1,120
    // protections a frame). The blocks noted absent are the same.
    let mut block = from;
    while block < to {
        if set(g, block, block + 1) {
            note(g, block, block + 1);
            block += 1;
            continue;
        }
        if !guard_runs() {
            // As before: every block asked on its own.
            if block < uncommitted_end(block) {
                g.absent.insert(block);
            }
            block += 1;
            continue;
        }
        let end = uncommitted_end(block).min(to);
        if end > block {
            g.absent.extend(block..end);
            block = end;
        } else {
            block += 1;
        }
    }
}

/// Whether a refused protection skips each uncommitted run whole
/// ([`protect_here`], fork #987; on by default). `VICTORY_GUARD_RUNS=0`:
/// block by block, as before.
fn guard_runs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let run = *ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_RUNS").is_none_or(|v| v != "0"));
    // May turn within a run (`crate::ab`): both ways note the same blocks.
    crate::ab::flag(crate::ab::Key::GuardRuns, run)
}

/// Whether an asking that does not take the guards' answer returns at once
/// for memory known as a view ([`writes_into`]; on by default).
/// `VICTORY_WATCH_VIEW_EARLY=0`: the kernel's write watch is asked first,
/// as before.
fn watch_view_early() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let run = *ON.get_or_init(|| std::env::var_os("VICTORY_WATCH_VIEW_EARLY").is_none_or(|v| v != "0"));
    // May turn within a run (`crate::ab`): both ways answer None.
    crate::ab::flag(crate::ab::Key::WatchViewEarly, run)
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded. The same physical pages at
/// every other view of them are written by that too (the write that faulted,
/// or is ours, `crate::host::copy_mapped`, goes through one view): they are
/// clean no longer, have no [`PUBLISHED`] bit, and their [`Unchanged`]
/// watchers are invalidated, whether they are opened as well
/// ([`open_one_view`]) or left read-only.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    if !guards_kept() {
        return open_classic(g, start, end);
    }
    if start >= end {
        return false;
    }
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let aliased: Vec<usize> = if g.aliased.is_empty() {
        Vec::new()
    } else {
        g.aliased.range(first..last).copied().collect()
    };
    let one = open_one_view();
    let mut others: Vec<usize> = Vec::new();
    // Before write access is given back: clear clean/PUBLISHED/watchers on
    // every alias of the same physical pages.
    for &block in &aliased {
        g.aliased.remove(&block);
        for alias in crate::inplace::pages_of(block * BLOCK) {
            let alias = alias / BLOCK;
            if alias < first || alias >= last {
                g.clean.remove(alias);
                PUBLISHED.clear(alias);
                invalidate(g, alias);
                if !one {
                    g.aliased.remove(&alias);
                    if g.protected.contains(&alias) {
                        others.push(alias);
                    }
                }
            }
        }
    }
    let any = open_here(g, start, end);
    if !one && !others.is_empty() {
        others.sort_unstable();
        others.dedup();
        for alias in others {
            if open_here(g, alias * BLOCK, (alias + 1) * BLOCK) {
                #[cfg(feature = "diagnostics")]
                ALIASES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
    any
}

/// [`open`] in a session that never names the write faults' new ways
/// ([`guards_kept`]): as before them (fx-main da215a9).
fn open_classic(g: &mut Guards, start: usize, end: usize) -> bool {
    // The light refresh: no view of these pages is known by its bit once
    // one of them can be written (who asks the bits takes no lock of ours;
    // after avatarmanz/VIctory-7900XTX 689df313).
    if light_kept() && start < end && !g.aliased.is_empty() {
        for &block in g.aliased.range(start / BLOCK..end.div_ceil(BLOCK)) {
            for alias in crate::inplace::pages_of(block * BLOCK) {
                PUBLISHED.clear(alias / BLOCK);
            }
        }
    }
    let any = open_here(g, start, end);
    if start >= end || g.aliased.is_empty() {
        return any;
    }
    let aliased: Vec<usize> = g.aliased.range(start / BLOCK..end.div_ceil(BLOCK)).copied().collect();
    for block in aliased {
        g.aliased.remove(&block);
        for alias in crate::inplace::pages_of(block * BLOCK) {
            if alias / BLOCK != block && open_here(g, alias, alias.saturating_add(BLOCK)) {
                #[cfg(feature = "diagnostics")]
                ALIASES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            g.aliased.remove(&(alias / BLOCK));
        }
    }
    any
}

/// [`open`] for one range, without the other views of its pages.
fn open_here(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g.protected.blocks_in(start / BLOCK..end.div_ceil(BLOCK));
    if guard_coalesce() {
        // Each run of them given back in one call ([`guard_coalesce`]): a
        // long write (`fread`) opens hundreds of blocks, and every call
        // walks the page tables. Noted and told block by block, as below.
        for (first, last) in block_runs(&blocks) {
            for block in first..last {
                g.protected.remove(&block);
                g.clean.remove(block);
                // Before the block can be written ([`PUBLISHED`]).
                PUBLISHED.clear(block);
                invalidate(g, block);
            }
            let mut old = 0;
            // SAFETY: blocks we made read-only.
            let mut call = || unsafe { VirtualProtect((first * BLOCK) as *const _, (last - first) * BLOCK, PAGE_READWRITE, &mut old) != 0 };
            let set = if last - first > 1 { refused_timed(call) } else { call() };
            if !set && last - first > 1 {
                // A run over regions Windows takes one at a time.
                for block in first..last {
                    let mut old = 0;
                    // SAFETY: a block we made read-only.
                    unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
                }
            }
            for block in first..last {
                PROTECTED.clear(block);
            }
        }
        return !blocks.is_empty();
    }
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(block);
        // Before the block can be written ([`PUBLISHED`]; nothing to do
        // where no bit was ever set).
        PUBLISHED.clear(block);
        invalidate(g, block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
        PROTECTED.clear(block);
    }
    !blocks.is_empty()
}

/// Whether a note that a block is uncommitted may exist ([`absent_once`]):
/// set under the guards' lock before Windows is asked about such a block
/// with notes kept, cleared under it by [`title_memory_committed`] when it
/// finds none. A commit the runtime tells of either finds it set, or came
/// before the question, whose answer then saw it. Without notes no commit
/// takes the guards' lock.
static NOTED_ABSENT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The runtime committed title memory `[at, at + len)` for the title
/// (`XMemVirtualAlloc` with a commit): what [`absent_once`] noted about its
/// blocks is forgotten, so the next asking puts the question to Windows
/// again, as every asking did before the notes. With `VICTORY_ABSENT_MS=told`
/// this and [`title_memory_remapped`] are the only ways a block noted
/// uncommitted is asked about again. From fxceless/victory-fork 6d20eed.
pub fn title_memory_committed(at: usize, len: usize) {
    if len == 0 || !NOTED_ABSENT.load(std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    let mut g = guards(crate::lockmeter::Site::TitleWritten);
    if g.verified.is_empty() {
        // (Under the guards' lock, which every note is made under.)
        NOTED_ABSENT.store(false, std::sync::atomic::Ordering::SeqCst);
        return;
    }
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    let noted: Vec<usize> = g.verified.range(first..last).map(|(&block, _)| block).collect();
    for block in noted {
        g.verified.remove(&block);
        forget_hole(&mut g, block);
    }
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(crate::lockmeter::Site::Writing), at, at.saturating_add(len));
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write the guards do
/// not see once its bytes are there ([`title_memory_written`]), as bits:
/// 1, the guarded blocks under it are opened and count as written (host
/// copies of buffers and textures read through tables take them again);
/// 2, the placed textures picked by index that lie under it are due again;
/// 4, both also for reads to plain addresses when they have finished.
/// Unset or `guards`: bit 1 alone. `0`: nothing, counted only (the default
/// until run 199). `1`: bits 1 and 2. `slots`: bit 2 alone. `all`:
/// everything.
/// (Bit 1 by default: in every run 10,000 to 50,000 guarded blocks of host
/// copies lay under such writes and kept the bytes from before; runs 197
/// and 198 opened them, 9,766 and 10,796 blocks, with nothing worse for
/// it. Bit 2 makes texture slots due and is the one that flooded the
/// queue in an earlier form: still asked for by name.)
pub(crate) fn page_writes() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_PAGE_WRITES").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("1") => 3,
        Ok("slots") => 2,
        Ok("all") => 7,
        _ => 1,
    })
}

/// Since the start: writes into physical pages through the runtime's own
/// views that landed at addresses the title has mapped (one for each run
/// of addresses); their bytes; guarded blocks that lay under them (host
/// copies that would otherwise keep the bytes from before); reads to
/// plain addresses that finished.
pub(crate) static WRITTEN: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The bytes of a write the guards do not see are there: title memory
/// `[at, at + len)`. `pages`: DirectStorage wrote them into physical pages
/// through a view of the runtime's own, and this is an address where the
/// title has those pages mapped (no title address faulted, nothing was
/// mapped or unmapped; the shortcut this file's header names). Otherwise a
/// read to a plain address has finished, of which [`writing`] was told
/// before it began.
///
/// Upstream's c75fdba does the same for its trackers (its `Writing` marks
/// the pages once more when dropped; page destinations are reported at
/// each address showing them). Here it is counted always and acted on as
/// [`page_writes`] says. Called without the runtime's page-table lock;
/// takes the guards' lock, then, alone, the lock of the ranges kept for
/// the texture pass.
pub fn title_memory_written(at: usize, len: usize, pages: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    // Whatever the guards do with it below (by default nothing for a
    // write that is not a page's, `VICTORY_PAGE_WRITES`), a copy holding
    // these bytes as current does not any more: a ReadFile into guarded
    // memory left the bytes from before it current in run P5-0.
    unchanged_gone(at, len);
    let mode = page_writes();
    let end = at.saturating_add(len);
    if pages {
        WRITTEN[0].fetch_add(1, Relaxed);
        WRITTEN[1].fetch_add(len as u64, Relaxed);
    } else {
        WRITTEN[3].fetch_add(1, Relaxed);
        if mode & 4 == 0 {
            return;
        }
    }
    {
        let mut g = guards(crate::lockmeter::Site::TitleWritten);
        let guarded = g.protected.count_in(at / BLOCK..end.div_ceil(BLOCK)) as u64;
        if pages {
            WRITTEN[2].fetch_add(guarded, Relaxed);
        }
        if mode & 1 != 0 && guarded != 0 {
            open(&mut g, at, end);
        }
    }
    // A placed texture there changed with no descriptor written.
    crate::descriptor::host::page_written(at, len);
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again — and are, the pages noted written, which also has the next check
/// guard the new view (`protect`).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    if ranges.is_empty() {
        return;
    }
    if !guards_kept() {
        views_mapped_classic(&ranges);
        return;
    }
    // Check `PROTECTED` without filtering `ranges`: all views of the pages
    // must be cleared under the lock together (avatarmanz 2be0515).
    if !ranges
        .iter()
        .any(|&(at, n)| PROTECTED.any(at / BLOCK, at.saturating_add(n).div_ceil(BLOCK)))
    {
        return;
    }
    let mut g = guards(crate::lockmeter::Site::ViewsMapped);
    // Clear clean, PUBLISHED and watchers on ALL ranges under the lock before
    // any view is made writable by `open_here`.
    for &(at, n) in &ranges {
        let end = at.saturating_add(n);
        let (first, last) = (at / BLOCK, end.div_ceil(BLOCK));
        g.clean.remove_range(first, last);
        take_out_range(&g, at, end);
        for block in g.protected.blocks_in(first..last) {
            PUBLISHED.clear(block);
        }
    }
    for &(at, n) in &ranges {
        if open_here(&mut g, at, at.saturating_add(n)) {
            #[cfg(feature = "diagnostics")]
            ALIASES[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// [`views_mapped`] of the views `ranges` in a session that never names the
/// write faults' new ways ([`guards_kept`]): as before them (fx-main
/// da215a9).
fn views_mapped_classic(ranges: &[(usize, usize)]) {
    let mut g = guards(crate::lockmeter::Site::ViewsMapped);
    // The light refresh: all of them told as written before the first can
    // be written (as in [`open`]).
    if light_kept() {
        for &(at, n) in ranges {
            for block in g.protected.blocks_in(at / BLOCK..at.saturating_add(n).div_ceil(BLOCK)) {
                PUBLISHED.clear(block);
            }
        }
    }
    for &(at, n) in ranges {
        if open_here(&mut g, at, at.saturating_add(n)) {
            #[cfg(feature = "diagnostics")]
            ALIASES[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>, std::time::Instant);

impl Drop for Remapping {
    fn drop(&mut self) {
        // From asking for the lock until it is let go (the frame log).
        crate::framelog::add(crate::framelog::T::RemapUs, self.1.elapsed().as_micros() as u64);
    }
}

/// The runtime is about to map, unmap or move physical pages of title
/// memory: until what it returns is dropped, no read of title memory in
/// place is under way and none begins ([`title_memory_reading`]). Taken
/// around the whole change, the unmapping and the mapping again.
pub fn title_memory_remapping() -> Remapping {
    let asked = std::time::Instant::now();
    let held = MAPPING.write().unwrap_or_else(|e| e.into_inner());
    crate::framelog::add(crate::framelog::T::Remaps, 1);
    crate::framelog::add(crate::framelog::T::RemapWaitUs, asked.elapsed().as_micros() as u64);
    Remapping(held, asked)
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>);

/// For reading title memory in place: until what it returns is dropped,
/// the runtime changes no mapping, so memory found mapped stays mapped.
/// Not to be taken twice by one thread, nor held across anything that
/// maps title memory.
pub fn title_memory_reading() -> Reading {
    Reading(MAPPING.read().unwrap_or_else(|e| e.into_inner()))
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping (or waits to), for a caller that has another
/// way to read and would rather not wait.
pub fn title_memory_reading_now() -> Option<Reading> {
    match MAPPING.try_read() {
        Ok(guard) => Some(Reading(guard)),
        Err(std::sync::TryLockError::Poisoned(e)) => Some(Reading(e.into_inner())),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    let mut g = guards(crate::lockmeter::Site::Remapped);
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    let absent: Vec<usize> = g.absent.range(range.clone()).copied().collect();
    for block in absent {
        g.absent.remove(&block);
        g.verified.remove(&block);
        forget_hole(&mut g, block);
    }
    let blocks: Vec<usize> = g.protected.blocks_in(range.clone());
    for block in blocks {
        g.protected.remove(&block);
        g.clean.remove(block);
        // What it shows is new ([`PUBLISHED`]).
        PUBLISHED.clear(block);
        PROTECTED.clear(block);
    }
    if back_check() {
        g.remapped.insert(range.start, range.end);
    }
    let _ = range;
    take_out_range(&g, at, at.saturating_add(len));
    let aliased: Vec<usize> = g.aliased.range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)).copied().collect();
    for block in aliased {
        g.aliased.remove(&block);
    }
    if !g.opened.is_empty() {
        g.forget_opened(at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    }
    remapped_viewed(&mut g, at, len, view_known());
}

/// Remappings that kept their blocks known as view memory, since the last
/// frame line ([`remapped_viewed`]).
pub(crate) static REMAPS_KEPT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What a remapping of `[at, at + len)` leaves of the blocks known as view
/// memory ([`view_known`]). With it on (fork fxceless 39731795d, its
/// "watch_remap"): the runtime remaps for two things only (`xmem`'s `map`
/// and `unmap`): a range it has just mapped a section's pages into, and a
/// view it is about to make a placeholder again; either way what stands
/// there is view memory or a hole in it, so the blocks stay known as view
/// memory. Forgotten, as before, the next asking about each range went to
/// the kernel's write watch and failed (185-191 a frame in run P6-0), then
/// to Windows' `VirtualQuery` walk (`is_view`). Only that knowledge is
/// kept: the blocks' guards were taken out above, so their bytes are
/// copied again at the next refresh whatever this says.
fn remapped_viewed(g: &mut Guards, at: usize, len: usize, keep: bool) {
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    if keep {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| tracing::info!("remapped title memory stays known as view memory (watch_remap, VICTORY_VIEW_KNOWN=1): on"));
        REMAPS_KEPT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        g.viewed.extend(range);
        return;
    }
    // What the range shows is new: no longer known to be a view.
    let viewed: Vec<usize> = g.viewed.blocks_in(range);
    for block in viewed {
        g.viewed.remove(&block);
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        let block = at / BLOCK;
        let fast = guards_kept() && (open_one_view() || !rest().is_zero());
        // Another thread may have opened the block since this one faulted
        // (then nothing waits for the lock with `fast`), and a copy may have
        // guarded it again since that (then it is ours again, not read-only
        // for some other reason).
        if fast && !PROTECTED.any(block, block + 1) && writable(at) {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards(crate::lockmeter::Site::Handler);
        if open(&mut g, at, at + 1) {
            // (No note with no rest: nothing reads it.)
            if guards_kept() && !rest().is_zero() {
                g.opened(block);
            }
            drop(g);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            crate::framelog::add(crate::framelog::T::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // Switches are read here, not first in the handler ([`open`] and
    // `fast` ask them): reading the environment allocates.
    light_kept();
    guards_kept();
    open_one_view();
    rest();
    guard_coalesce();
    guard_intervals();
    // The rests' slots, made here so that the handler allocates none (it
    // runs inside any thread's fault, the process heap's lock among what
    // that thread may hold).
    static SLOTS: std::sync::Once = std::sync::Once::new();
    SLOTS.call_once(|| {
        // (Tests: whichever thread installs, a later one may rest.)
        if cfg!(test) || guards_kept() {
            guards(crate::lockmeter::Site::Other).opened = vec![(usize::MAX, None); OPENED_SLOTS];
        }
    });
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn clean_runs_match_per_block_tracking() {
        let mut runs = CleanBlocks(BTreeMap::new());
        let mut blocks = BTreeSet::new();
        let check = |runs: &CleanBlocks, blocks: &BTreeSet<usize>| {
            for from in 0..36 {
                for to in from..36 {
                    let actual: Vec<_> = runs.uncovered(from, to).into_iter().flat_map(|(a, b)| a..b).collect();
                    let expected: Vec<_> = (from..to).filter(|block| !blocks.contains(block)).collect();
                    assert_eq!(actual, expected, "{from}..{to}");
                }
            }
        };

        for (from, to) in [(5, 12), (1, 3), (3, 7), (23, 33), (17, 20), (9, 26), (2, 3)] {
            runs.insert(from, to);
            blocks.extend(from..to);
            check(&runs, &blocks);
        }
        for block in [0, 1, 11, 14, 15, 32, 13, 12] {
            runs.remove(block);
            blocks.remove(&block);
            check(&runs, &blocks);
        }
        runs.insert(11, 16);
        blocks.extend(11..16);
        check(&runs, &blocks);
        // Ranges out: inside a run, across runs, at their edges, over none.
        for (from, to) in [(6, 8), (10, 24), (2, 3), (30, 40), (0, 1), (4, 5)] {
            runs.remove_range(from, to);
            for block in from..to {
                blocks.remove(&block);
            }
            check(&runs, &blocks);
        }
    }

    /// `VICTORY_GUARD_INTERVALS` turned within a run: the blocks move from
    /// one store into the other and back with nothing lost or added, and
    /// answer the same in either.
    #[test]
    fn guard_blocks_move_between_stores_whole() {
        let mut set = BlockSet::new();
        for block in [3, 4, 5, 9, 20, 21] {
            set.insert(block);
        }
        set.extend(30..40);
        let want: Vec<usize> = set.blocks_in(0..64);
        set.store_as(true);
        assert!(set.runs_mode && set.set.is_empty());
        assert_eq!(set.blocks_in(0..64), want);
        assert_eq!(set.count_in(0..64), want.len());
        assert!(set.contains(&4) && !set.contains(&6));
        set.remove(&35);
        set.store_as(false);
        assert!(!set.runs_mode && set.runs.0.is_empty());
        let mut less = want.clone();
        less.retain(|&b| b != 35);
        assert_eq!(set.blocks_in(0..64), less);
    }

    /// The guards' blocks as runs answer as one by one
    /// (`VICTORY_GUARD_INTERVALS`): after every step of a mixed sequence of
    /// single blocks, ranges and removals, membership, the blocks of every
    /// range and their count are the same.
    #[test]
    fn guard_blocks_as_runs_answer_as_one_by_one() {
        let mut runs = CleanBlocks(BTreeMap::new());
        let mut set = BTreeSet::new();
        let check = |runs: &CleanBlocks, set: &BTreeSet<usize>| {
            for from in 0..40 {
                for to in from..40 {
                    let want: Vec<usize> = set.range(from..to).copied().collect();
                    assert_eq!(runs_blocks_in(runs, from..to), want, "{from}..{to}");
                    assert_eq!(runs_count_in(runs, from..to), want.len(), "{from}..{to}");
                }
                assert_eq!(runs.uncovered(from, from + 1).is_empty(), set.contains(&from), "{from}");
            }
        };
        let steps: [(u8, usize, usize); 12] = [
            (0, 3, 9), (1, 12, 0), (0, 10, 12), (2, 5, 0), (0, 20, 30), (2, 20, 0),
            (0, 0, 1), (2, 29, 0), (1, 31, 0), (0, 8, 25), (2, 15, 0), (2, 3, 0),
        ];
        for (op, a, b) in steps {
            match op {
                0 => {
                    runs.insert(a, b);
                    set.extend(a..b);
                }
                1 => {
                    runs.insert(a, a + 1);
                    set.insert(a);
                }
                _ => {
                    runs.remove(a);
                    set.remove(&a);
                }
            }
            check(&runs, &set);
        }
        assert_eq!(runs_blocks_in(&runs, 5..3), [] as [usize; 0]);
    }

    /// Opening guarded blocks over runs (`VICTORY_GUARD_COALESCE`) leaves
    /// the same blocks guarded and the same pages read-only as block by
    /// block: eight blocks of memory of the test's, all guarded, then three
    /// ranges opened.
    #[test]
    fn opening_over_runs_leaves_what_block_by_block_leaves() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualAlloc, VirtualFree,
            VirtualQuery,
        };
        let run = |mode: u8| -> (Vec<bool>, Vec<u32>) {
            // SAFETY: a fresh allocation of the test's, freed at the end.
            let p = unsafe { VirtualAlloc(std::ptr::null(), 8 * BLOCK, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
            assert_ne!(p, 0);
            let first = p / BLOCK;
            COALESCE_TEST.store(mode, std::sync::atomic::Ordering::SeqCst);
            let held = {
                let mut g = guards(crate::lockmeter::Site::Other);
                protect_here(&mut g, first, first + 8);
                for (a, b) in [(1, 3), (4, 5), (6, 8)] {
                    let _ = open_here(&mut g, (first + a) * BLOCK, (first + b) * BLOCK);
                }
                let held: Vec<bool> = (first..first + 8).map(|b| g.protected.contains(&b)).collect();
                let protect: Vec<u32> = (0..8)
                    .map(|b| {
                        // SAFETY: plain data filled in by the query.
                        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                        // SAFETY: a valid out pointer of the right size.
                        unsafe { VirtualQuery((p + b * BLOCK) as *const _, &mut info, size_of_val(&info)) };
                        info.Protect
                    })
                    .collect();
                let _ = open_here(&mut g, first * BLOCK, (first + 8) * BLOCK);
                (held, protect)
            };
            COALESCE_TEST.store(0, std::sync::atomic::Ordering::SeqCst);
            let (held, protect) = held;
            // SAFETY: the test's allocation.
            unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
            (held, protect)
        };
        let off = run(1);
        let on = run(2);
        assert_eq!(off, on);
        assert_eq!(off.0, [true, false, false, true, false, true, false, false]);
    }

    /// The runs of blocks one `VirtualProtect` call covers
    /// (`VICTORY_GUARD_COALESCE`).
    #[test]
    fn blocks_are_cut_into_runs_that_follow_one_another() {
        assert_eq!(block_runs(&[]), []);
        assert_eq!(block_runs(&[4]), [(4, 5)]);
        assert_eq!(block_runs(&[1, 2, 3, 7, 8, 10]), [(1, 4), (7, 9), (10, 11)]);
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        forget_journal(p, p + 4 * PAGE);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let (section, view) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (section, MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize)
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        title_memory_remapped(view, 2 * BLOCK);
        forget_journal(view, view + 2 * BLOCK);
        // SAFETY: our view; the section's handle.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            );
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        title_memory_remapped(p, 3 * BLOCK);
        forget_journal(p, p + 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        title_memory_remapped(p, 3 * BLOCK);
        forget_journal(p, p + 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// With a note kept ([`absent_once`]): a block Windows said is
    /// uncommitted is not asked about again until the note is old, so what
    /// appears there without a word is seen that much later; what the
    /// runtime tells of (maps, or commits) is seen at once; and with no note
    /// kept (`keep` 0) everything is seen at the next asking, as before.
    /// From fxceless/victory-fork 44bff61, with the commit told (6d20eed).
    #[test]
    fn a_block_noted_uncommitted_is_asked_about_again_only_when_its_note_is_old() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE_PLACEHOLDER,
            MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // (Nothing else set by hand: the commit's word must reach the notes
        // as in a run without VICTORY_SKIP_UNCHANGED.)
        // SAFETY: a fresh placeholder of five blocks and a pagefile section
        // of five, as the runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                5 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, (5 * BLOCK) as u32, std::ptr::null());
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree((p + block * BLOCK) as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let (from, to) = (p, p + 5 * BLOCK);
        const KEEP: u64 = 100;
        let first = map(0);
        let all: Vec<usize> = (0..5).flat_map(pages).collect();
        assert_eq!(guard_at(from, to, 1_000, KEEP), Some(all), "the first asking takes every block");
        assert_eq!(guard_at(from, to, 1_000, KEEP), Some(Vec::new()), "blocks 1 to 4 are noted uncommitted");
        // Mapped without a word to the guards.
        let second = map(1);
        assert_eq!(guard_at(from, to, 1_050, KEEP), Some(Vec::new()), "the note is 50 ms old: Windows is not asked");
        assert_eq!(guard_at(from, to, 1_100, KEEP), Some(pages(1).collect()), "the note is old: the block is found");
        assert_eq!(guard_at(from, to, 1_100, KEEP), Some(Vec::new()));
        // Mapped and told, as the runtime does: seen at once, note or not.
        let third = map(2);
        title_memory_remapped(p + 2 * BLOCK, BLOCK);
        assert_eq!(guard_at(from, to, 1_101, KEEP), Some(pages(2).collect()));
        // Committed and told (`told`: a note that never gets old).
        let fourth = map(3);
        assert_eq!(guard_at(from, to, 1_101, u64::MAX), Some(Vec::new()), "noted for good: not asked");
        title_memory_committed(p + 3 * BLOCK, BLOCK);
        assert_eq!(guard_at(from, to, 1_101, u64::MAX), Some(pages(3).collect()), "told: asked again, found");
        // No note kept: a block mapped without a word is found at the next
        // asking, though its note is one millisecond old.
        let fifth = map(4);
        assert_eq!(guard_at(from, to, 1_101, 0), Some(pages(4).collect()));
        assert_eq!(guard_at(from, to, 1_101, KEEP), Some(Vec::new()));
        title_memory_remapped(p, 5 * BLOCK);
        forget_journal(p, p + 5 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            for view in [first, second, third, fourth, fifth] {
                UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER);
            }
            for block in 0..5 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// [`writes_as`] of view memory, its answer as a list.
    fn light_ask(start: usize, end: usize, light: u8) -> Option<(u64, Vec<(usize, u64)>)> {
        let mut written = Vec::new();
        writes_as(start, end, true, light, &mut written).map(|now| (now, written))
    }

    /// The tests of the light refresh take turns: they read its counts,
    /// which no other test moves (the switch is not set in tests, so
    /// [`writes_into`] itself keeps no bits and asks none).
    fn light_turn() -> std::sync::MutexGuard<'static, ()> {
        static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
        TURN.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Askings answered by the bits, and askings that went to the guards.
    fn light_counts() -> [u64; 2] {
        [0, 1].map(|i| LIGHT_ASKS[i].load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Whether `[start, end)` is known by its bits.
    fn known(start: usize, end: usize) -> bool {
        published(&JOURNAL.lock().unwrap_or_else(|e| e.into_inner()), start, end)
    }

    /// The pages of an answer of [`writes`] written after the asking `since`.
    fn since(since: u64, written: Vec<(usize, u64)>) -> Vec<usize> {
        written.into_iter().filter(|w| w.1 > since).map(|w| w.0).collect()
    }

    /// The asking that makes blocks clean gives them their bits; the next
    /// one is answered by them, and the guards are not asked. A write takes
    /// the written block's bit, through the guarded view or through another
    /// view of the same pages, and the next asking reports it.
    #[test]
    fn a_range_is_known_by_its_bits_until_a_write() {
        let _turn = light_turn();
        const SIZE: usize = 3 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |block: usize| (one + block * BLOCK) / PAGE;
        assert!(!known(one, one + SIZE), "nothing asked yet");
        // The first asking takes every block in, read-only and clean.
        let (first, all) = light_ask(one, one + SIZE, 2).expect("a view");
        assert_eq!(all, [(pages(0), first), (pages(1), first), (pages(2), first)]);
        assert!(known(one, one + SIZE), "made clean and noted");
        // The second is answered by the bits.
        let before = light_counts();
        let (second, written) = light_ask(one, one + SIZE, 2).expect("a view");
        assert!(second > first);
        assert!(since(first, written).is_empty(), "nothing written");
        assert_eq!(light_counts(), [before[0] + 1, before[1]], "the guards were not asked");
        // A write to the guarded view faults: the handler takes the bit.
        // SAFETY: the first view, read-write once the handler has opened it.
        unsafe { ((one + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert!(!known(one, one + SIZE));
        assert!(known(one, one + BLOCK) && known(one + 2 * BLOCK, one + SIZE), "the blocks beside it keep theirs");
        let before = light_counts();
        let (third, written) = light_ask(one, one + SIZE, 2).expect("a view");
        assert_eq!(since(second, written), [pages(1)]);
        assert_eq!(light_counts(), [before[0], before[1] + 1], "the guards were asked");
        assert!(known(one, one + SIZE), "guarded again and noted");
        // A write through the other view of the same pages takes it too.
        // SAFETY: the second view, as above.
        unsafe { ((two + 2 * BLOCK + 8) as *mut u8).write_volatile(1) };
        assert!(!known(one + 2 * BLOCK, one + SIZE), "written through another view");
        assert!(known(one, one + 2 * BLOCK));
        let (fourth, written) = light_ask(one, one + SIZE, 2).expect("a view");
        assert_eq!(since(third, written), [pages(2)]);
        // A write of ours that is told of (`writing`) does as well.
        writing(one + 8, 8);
        assert!(!known(one, one + BLOCK));
        let (_, written) = light_ask(one, one + SIZE, 2).expect("a view");
        assert_eq!(since(fourth, written), [pages(0)]);
        assert!(known(one, one + SIZE));
        // What the runtime maps anew is not known.
        title_memory_remapped(one + BLOCK, BLOCK);
        assert!(!known(one + BLOCK, one + 2 * BLOCK));
        drop_views(section, &[one, two], SIZE);
    }

    /// `VICTORY_LIGHT_REFRESH=alt`: in the phase without, the bits are not
    /// asked, and they are kept: what is written in that phase, asked about
    /// in it or not, is reported by the first asking of the phase with.
    /// With the switch off no bit is ever set.
    #[test]
    fn bits_kept_through_a_phase_that_does_not_ask_them_are_true_in_the_next() {
        let _turn = light_turn();
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |block: usize| (one + block * BLOCK) / PAGE;
        // With: taken in, then known by its bits.
        let (first, _) = light_ask(one, one + SIZE, 2).expect("a view");
        assert!(known(one, one + SIZE));
        // Without: the bits are there and are not asked.
        let before = light_counts();
        let (second, written) = light_ask(one, one + SIZE, 1).expect("a view");
        assert!(since(first, written).is_empty());
        assert_eq!(light_counts(), [before[0], before[1] + 1], "without: the guards answer");
        // Written in the phase without: the fault takes the bit all the
        // same, and that phase's asking gives it back.
        // SAFETY: the first view, read-write once the handler has opened it.
        unsafe { ((one + 8) as *mut u8).write_volatile(1) };
        assert!(!known(one, one + BLOCK));
        let (third, written) = light_ask(one, one + SIZE, 1).expect("a view");
        assert_eq!(since(second, written), [pages(0)]);
        assert!(known(one, one + SIZE));
        // Written again in the phase without, through the other view, and
        // not asked about until the phase with: its first asking goes to
        // the guards and reports it.
        // SAFETY: the second view, as above.
        unsafe { ((two + BLOCK + 8) as *mut u8).write_volatile(1) };
        let before = light_counts();
        let (fourth, written) = light_ask(one, one + SIZE, 2).expect("a view");
        assert_eq!(since(third, written), [pages(1)], "a bit from before the phase without was trusted");
        assert_eq!(light_counts(), [before[0], before[1] + 1]);
        let before = light_counts();
        let (fifth, written) = light_ask(one, one + SIZE, 2).expect("a view");
        assert!(since(fourth, written).is_empty());
        assert_eq!(light_counts(), [before[0] + 1, before[1]]);
        // The switch off: a write still takes the bit, the asking sets none
        // and counts nothing.
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write_volatile(2) };
        let before = light_counts();
        let (_, written) = light_ask(one, one + SIZE, 0).expect("a view");
        assert_eq!(since(fifth, written), [pages(0)]);
        assert_eq!(light_counts(), before);
        assert!(!known(one, one + BLOCK), "off: no bit is set");
        drop_views(section, &[one, two], SIZE);
    }

    /// A range with unmapped blocks is not known by its bits while its
    /// holes are asked about at every asking. Noted for good, a hole that
    /// lies in a placeholder is; then nothing is asked until the runtime
    /// tells of the block, or the look at the holes finds it filled. A
    /// hole in an ordinary reservation is never known by a bit, and its
    /// note grows old.
    #[test]
    fn holes_are_known_by_their_bits_only_when_noted_for_good() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let _turn = light_turn();
        // SAFETY: a fresh placeholder of four blocks and a pagefile
        // section of four, as the runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                4 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (4 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE;
        let (from, to) = (p, p + 4 * BLOCK);
        let first = map(0);
        assert!(placeholder(p / BLOCK + 1) && placeholder(p / BLOCK + 3), "what is left of the placeholder");
        assert!(!placeholder(p / BLOCK), "a view is none");
        // The first asking takes every block; blocks 1 to 3 are unmapped.
        let (one, all) = light_ask(from, to, 2).expect("a view");
        assert_eq!(since(one - 1, all), [pages(0), pages(1), pages(2), pages(3)]);
        assert!(known(p, p + BLOCK) && !known(from, to));
        // Their notes are not for good (no switch says so here): each
        // asking goes to the guards, which ask Windows.
        let before = light_counts();
        let (two, written) = light_ask(from, to, 2).expect("a view");
        assert!(since(one, written).is_empty());
        assert_eq!(light_counts(), [before[0], before[1] + 1]);
        assert!(!known(from, to), "a hole not noted for good has no bit");
        // Noted for good, as with `VICTORY_ABSENT_MS=told`, and looked at
        // as the asking after the notes does.
        assert_eq!(guard_at(from, to, 1, u64::MAX), Some(Vec::new()));
        publish(&JOURNAL.lock().unwrap_or_else(|e| e.into_inner()), &[], from, to, true);
        assert!(known(from, to), "holes in a placeholder, noted for good");
        let before = light_counts();
        let (three, written) = light_ask(from, to, 2).expect("a view");
        assert!(since(two, written).is_empty());
        assert_eq!(light_counts(), [before[0] + 1, before[1]], "answered by the bits, holes and all");
        // Mapped without a word: not seen, as with the notes alone ...
        let second = map(1);
        let (four, written) = light_ask(from, to, 2).expect("a view");
        assert!(since(three, written).is_empty(), "nothing is asked about a hole known by its bit");
        // ... until the look at the holes comes by (it goes round, and
        // waits its turn while another test changes a mapping).
        for _ in 0..2000 {
            if !known(p + BLOCK, p + 2 * BLOCK) {
                break;
            }
            look_at_holes(usize::MAX);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(!known(p + BLOCK, p + 2 * BLOCK), "found filled");
        assert!(known(p + 2 * BLOCK, to), "the holes beside it are holes still");
        let (five, written) = light_ask(from, to, 2).expect("a view");
        assert_eq!(since(four, written), [pages(1)]);
        assert!(known(from, to));
        // Mapped and told, as the runtime does: the bit goes at once.
        let third = map(2);
        title_memory_remapped(p + 2 * BLOCK, BLOCK);
        assert!(!known(p + 2 * BLOCK, p + 3 * BLOCK));
        let (_, written) = light_ask(from, to, 2).expect("a view");
        assert_eq!(since(five, written), [pages(2)]);
        // A hole in an ordinary reservation, noted for good all the same:
        // anybody can commit it with no word to us, so it gets no bit.
        // SAFETY: a fresh reservation of one block, released below.
        let plain = unsafe { VirtualAlloc(std::ptr::null(), BLOCK, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert_ne!(plain, 0);
        assert!(!placeholder(plain / BLOCK));
        assert_eq!(guard_at(plain, plain + BLOCK, 1, u64::MAX), Some(vec![plain / PAGE]));
        assert_eq!(guard_at(plain, plain + BLOCK, 2, u64::MAX), Some(Vec::new()), "noted");
        publish(&JOURNAL.lock().unwrap_or_else(|e| e.into_inner()), &[], plain, plain + BLOCK, true);
        assert!(!known(plain, plain + BLOCK));
        // Nor does its note live for good: committed with no word, it is
        // found when the note is old.
        // SAFETY: the reservation made above.
        let committed = unsafe { VirtualAlloc(plain as *const _, BLOCK, MEM_COMMIT, PAGE_READWRITE) };
        assert_eq!(committed as usize, plain);
        assert_eq!(guard_at(plain, plain + BLOCK, 50, u64::MAX), Some(Vec::new()), "the note is 48 ms old");
        assert_eq!(guard_at(plain, plain + BLOCK, 2 + UNSURE_MS, u64::MAX), Some(vec![plain / PAGE]), "the note is old");
        title_memory_remapped(plain, BLOCK);
        title_memory_remapped(p, 4 * BLOCK);
        forget_journal(p, p + 4 * BLOCK);
        // SAFETY: our views, placeholder and reservation.
        unsafe {
            VirtualFree(plain as *mut _, 0, MEM_RELEASE);
            for view in [first, second, third] {
                UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER);
            }
            for block in 0..4 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// A copy has nothing new to take of a range known by its bits when it
    /// took every page of it after the page's last noted write, and that is
    /// known with nothing asked.
    #[test]
    fn a_copy_that_holds_every_noted_write_has_nothing_new() {
        let _turn = light_turn();
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let base = one / PAGE;
        assert!(!nothing_noted(one, one + SIZE, base, &[1, 1]), "never asked about: something would have to be asked");
        let (first, _) = light_ask(one, one + SIZE, 2).expect("a view");
        let before = light_counts();
        assert!(nothing_noted(one, one + SIZE, base, &[first, first]), "both pages taken at the asking that found them");
        assert_eq!(light_counts(), [before[0] + 1, before[1]]);
        assert!(!nothing_noted(one, one + SIZE, base, &[first, first - 1]), "a page taken before its write was found");
        assert!(!nothing_noted(one, one + SIZE, base, &[first, 0]), "a page never taken");
        assert!(!nothing_noted(one, one + SIZE, base + 1, &[first, first]), "a range the copy's notes do not hold");
        assert!(!nothing_noted(one, one + SIZE, base, &[first]));
        assert!(!nothing_noted(one, one, base, &[first, first]), "no range");
        // Written: the block's bit is gone, and something must be asked.
        // SAFETY: the first view, read-write once the handler has opened it.
        unsafe { ((one + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert!(!nothing_noted(one, one + SIZE, base, &[first, first]));
        assert!(nothing_noted(one, one + BLOCK, base, &[first, first]), "the block beside it is as it was");
        let (second, _) = light_ask(one, one + SIZE, 2).expect("a view");
        assert!(!nothing_noted(one, one + SIZE, base, &[first, first]), "the journal has the write");
        assert!(nothing_noted(one, one + SIZE, base, &[first, second]));
        drop_views(section, &[one, two], SIZE);
    }

    /// A range's bits are all of its blocks', whatever lies beside them.
    #[test]
    fn bits_of_ranges() {
        static BITS: Bits = Bits::new();
        let blocks = [0usize, 1, 63, 64, 65, 127, 128, 200, 65535, 65536, 65537, 1 << 20, (1 << 31) - 1];
        for &b in &blocks {
            assert!(!BITS.all(b, b + 1));
            BITS.set(b);
            assert!(BITS.all(b, b + 1));
        }
        assert!(BITS.all(0, 2) && BITS.all(63, 66) && BITS.all(127, 129) && BITS.all(65535, 65538));
        assert!(!BITS.all(0, 3) && !BITS.all(62, 64) && !BITS.all(65534, 65536) && !BITS.all(5, 5));
        for b in 300..700 {
            BITS.set(b);
        }
        assert!(BITS.all(300, 700) && !BITS.all(299, 700) && !BITS.all(300, 701));
        BITS.clear(511);
        assert!(!BITS.all(300, 700) && BITS.all(300, 511) && BITS.all(512, 700));
        BITS.set(1 << 40);
        BITS.clear(1 << 40);
        assert!(!BITS.all(1 << 40, (1 << 40) + 1), "outside the address space");
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    pub(super) fn two_views_for_others(bytes: usize) -> (usize, usize, usize) {
        two_views(bytes)
    }

    fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    /// The journal's notes of `[start, end)` dropped: a test that lets its
    /// memory go leaves no written pages for the next one that gets the
    /// same addresses (one asking "nothing written yet" failed so).
    fn forget_journal(start: usize, end: usize) {
        let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        let pages: Vec<usize> = j.written.range(start / PAGE..end.div_ceil(PAGE)).map(|(&p, _)| p).collect();
        for page in pages {
            j.written.remove(&page);
        }
    }

    pub(super) fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            forget_journal(view, view + bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A write of the runtime's own, told at one view of the pages
    /// (`writing`): the other views' next checks report the pages written.
    #[test]
    fn a_write_told_at_one_view_is_reported_at_the_others() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        writing(other + BLOCK / 2, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// Both views of the pages are checked, each for a copy of its own
    /// (the title shows one physical page at several addresses, and each
    /// address has its host copy): a write through either is reported at
    /// both, whichever is checked first. Guarding the pages for the first
    /// view's check must not note the second view clean, or its copy would
    /// keep the bytes from before the write.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0), pages(two, 1)]), "the second view's first check");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()), "nothing written yet");
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 1)]), "written through this view, checked second");
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0)]), "written through this view, checked second");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        drop_views(section, &[one, two], SIZE);
    }

    /// A page shown in more than `MANY_VIEWS` views is guarded at each by
    /// itself: the others stay writable, and nothing looks through them all.
    #[test]
    fn a_page_of_many_views_is_guarded_one_view_at_a_time() {
        let (section, one, two) = two_views(BLOCK);
        let filler = one + (1 << 45);
        for n in 0..MANY_VIEWS {
            crate::inplace::title_memory_view(filler + n * BLOCK, BLOCK, section, 0);
        }
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        assert!(writable(two), "another view of it is left alone");
        // SAFETY: the guarded view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        title_memory_remapped(filler, MANY_VIEWS * BLOCK);
        drop_views(section, &[one, two], BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }

    /// Restores thread-local [`open_one_view`] / [`rest`] test overrides on drop.
    pub(super) struct TestSwitches;

    impl TestSwitches {
        pub(super) fn set(one: Option<bool>, rest: Option<std::time::Duration>) -> Self {
            OPEN_ONE_IN_TEST.set(one);
            REST_IN_TEST.set(rest);
            Self
        }
    }

    impl Drop for TestSwitches {
        fn drop(&mut self) {
            OPEN_ONE_IN_TEST.set(None);
            REST_IN_TEST.set(None);
        }
    }

    /// Maps `VIEWS` views of a fresh `bytes`-sized pagefile section and registers
    /// them with `crate::inplace`.
    fn n_views<const VIEWS: usize>(bytes: usize) -> (usize, [usize; VIEWS]) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        // SAFETY: fresh pagefile section mapped `VIEWS` times; dropped via `drop_views`.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let views = std::array::from_fn(|_| {
                let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
                assert_ne!(view, 0);
                crate::inplace::title_memory_view(view, bytes, section as usize, 0);
                view
            });
            (section as usize, views)
        }
    }

    /// [`Bits::any`] detects any set bit in `[first, last)` across word and
    /// 4 GiB piece boundaries.
    #[test]
    fn bits_any_matches_ranges() {
        static BITS: Bits = Bits::new();
        assert!(!BITS.any(0, 100));
        assert!(!BITS.any(5, 5));
        for b in [63usize, 65536, 65537] {
            BITS.set(b);
        }
        assert!(BITS.any(0, 64));
        assert!(BITS.any(63, 64));
        assert!(!BITS.any(0, 63));
        assert!(!BITS.any(64, 65536));
        assert!(BITS.any(65535, 65537));
        assert!(BITS.any(65537, 65600));
        assert!(!BITS.any(65538, 65600));
        BITS.clear(63);
        assert!(!BITS.any(0, 64));
        BITS.set(1 << 40);
        assert!(!BITS.any(1 << 40, (1 << 40) + 1), "outside the address space");
    }

    /// `VICTORY_OPEN_ONE_VIEW=1`: a write fault opens only the view it came
    /// through; the other view stays `PAGE_READONLY` yet not clean, is still
    /// reported written on `guard` (which then re-closes the faulted view),
    /// accepts its own fault, and is opened by `writing`. From avatarmanz 7bc7c07.
    #[test]
    fn a_write_opens_the_view_it_goes_through() {
        let _sw = TestSwitches::set(Some(true), Some(std::time::Duration::ZERO));
        let (section, one, two) = two_views(BLOCK);
        let page = |view: usize| view / PAGE;
        for view in [one, two] {
            assert_eq!(guard(view, view + BLOCK), Some(vec![page(view)]));
        }
        // SAFETY (all): views mapped above and opened on fault by the handler.
        unsafe { ((two + 8) as *mut u8).write_volatile(1) };
        assert!(writable(two) && !writable(one));
        unsafe { ((one + 8) as *mut u8).write_volatile(1) };
        assert!(writable(two) && writable(one));
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]));
        assert_eq!(guard(two, two + BLOCK), Some(vec![page(two)]));
        // Written at the view that was left read-only, after the other.
        unsafe { ((two + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]), "not clean, though never opened");
        assert!(!writable(two) && !writable(one));
        assert_eq!(guard(two, two + BLOCK), Some(vec![page(two)]));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        // A write told on the still-guarded view finds it and opens it.
        unsafe { ((two + 8) as *mut u8).write_volatile(1) };
        assert!(!writable(one));
        writing(one, 8);
        assert!(writable(one));
        drop_views(section, &[one, two], BLOCK);
    }

    /// `VICTORY_GUARD_REST`: a block just opened by a write fault stays
    /// `PAGE_READWRITE`, yet is reported written and `!PUBLISHED` at every
    /// asking on every view until its rest ends, after which `guard` re-guards
    /// all views and returns empty until a new write. From avatarmanz 7bc7c07.
    #[test]
    fn a_block_just_written_rests() {
        let _turn = light_turn();
        for one_view in [false, true] {
            let _sw = TestSwitches::set(Some(one_view), Some(std::time::Duration::ZERO));
            let (section, one, two) = two_views(BLOCK);
            let page = one / PAGE;
            assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
            assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
            REST_IN_TEST.set(Some(std::time::Duration::from_secs(3600)));
            assert_eq!(guard(one, one + BLOCK), Some(Vec::new()), "nothing written: as it was");
            // SAFETY (all): views mapped above and opened on fault by the handler.
            unsafe { ((two + 8) as *mut u8).write_volatile(1) };
            for _ in 0..3 {
                assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
                assert!(writable(two), "no fault for the next write");
                let mut written = Vec::new();
                assert!(writes_as(two, two + BLOCK, true, 2, &mut written).is_some());
                assert!(!PUBLISHED.all(two / BLOCK, two / BLOCK + 1));
                assert!(!PUBLISHED.all(one / BLOCK, one / BLOCK + 1));
                unsafe { ((two + 8) as *mut u8).write_volatile(1) };
            }
            REST_IN_TEST.set(Some(std::time::Duration::ZERO));
            assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
            assert!(!writable(two) && !writable(one));
            assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
            drop_views(section, &[one, two], BLOCK);
        }
    }

    /// Concurrent writers and askers across 1 and 3 views with `VICTORY_GUARD_REST`
    /// and `VICTORY_OPEN_ONE_VIEW` enabled (plus concurrent `views_mapped` for
    /// 3 views, regression for avatarmanz 2be0515): no write is ever lost.
    #[test]
    fn no_write_to_resting_views_is_lost_between_askers() {
        askers_lose_no_write::<1>(true);
        askers_lose_no_write::<3>(true);
    }

    /// As [`no_write_to_resting_views_is_lost_between_askers`], in a session
    /// that never names the write faults' new ways (`guards_kept` false: the
    /// guards' old paths, `open_classic`, `protect_classic`,
    /// `views_mapped_classic`), asked through the guards as the yard's runs
    /// ask (no light refresh): no write is ever lost.
    #[test]
    fn no_write_through_another_view_is_lost_between_askers_the_old_way() {
        askers_lose_no_write::<1>(false);
        askers_lose_no_write::<3>(false);
    }

    /// Eight threads write their own words through any of `VIEWS` views of
    /// one section and ask through any view, with the runtime's word of new
    /// views and our own told writes among them: every asking after a write
    /// reports it. `new_ways`: with `VICTORY_OPEN_ONE_VIEW` and
    /// `VICTORY_GUARD_REST` (50 us) on, asked by the bits where they can
    /// answer; else neither named, asked through the guards.
    fn askers_lose_no_write<const VIEWS: usize>(new_ways: bool) {
        {
            const PAGES: usize = 8;
            const THREADS: usize = 8;
            let bytes = PAGES * BLOCK;
            let (section, all) = n_views::<VIEWS>(bytes);
            for &view in &all {
                assert!(writes(view, view + bytes, true).is_some());
            }
            let _turn = light_turn();
            std::thread::scope(|s| {
                for t in 0..THREADS {
                    let all_ref = &all;
                    s.spawn(move || {
                        let _sw = new_ways.then(|| TestSwitches::set(Some(true), Some(std::time::Duration::from_micros(50))));
                        assert_eq!(guards_kept(), new_ways, "the session's ways as asked");
                        let light: u8 = if new_ways { 2 } else { 0 };
                        let mut last = [[0u64; PAGES]; VIEWS];
                        let mut seed = (t as u64 + 1).wrapping_mul(6364136223846793005);
                        let mut next = |n: usize| {
                            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                            (seed >> 33) as usize % n
                        };
                        for round in 0..250usize {
                            let page = next(PAGES);
                            let from = page;
                            let to = (page + 1 + next(3)).min(PAGES);
                            let through = all_ref[next(VIEWS)];
                            let asked = next(VIEWS);
                            let at = all_ref[asked];
                            let word = through + page * BLOCK + 8 * (t + 1);
                            let mut written = Vec::new();
                            let before = writes_as(at + from * PAGE, at + to * PAGE, true, light, &mut written)
                                .expect("a view");
                            if round % 19 == 0 && VIEWS > 1 {
                                views_mapped(section, (page * BLOCK) as u64, BLOCK);
                            } else if round % 11 == 0 {
                                writing(word, 8);
                            }
                            // SAFETY: each thread writes its own 8-byte slot inside its view.
                            unsafe { (word as *mut u64).write_volatile((round + 1) as u64) };
                            let (now, written) = {
                                let mut w = Vec::new();
                                let n = writes_as(at + from * PAGE, at + to * PAGE, true, light, &mut w).expect("a view");
                                (n, w)
                            };
                            let number = written.iter().find(|w| w.0 == at / PAGE + page).map_or(0, |w| w.1);
                            assert!(
                                number > before,
                                "VIEWS={VIEWS} page={page}: got={number} <= before={before}"
                            );
                            assert!(number >= last[asked][page], "VIEWS={VIEWS} page={page}: regressed {number} < {}", last[asked][page]);
                            for p in from..to {
                                last[asked][p] = last[asked][p].max(now);
                            }
                        }
                    });
                }
            });
            drop_views(section, &all, bytes);
        }
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{title_memory_reading, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// A change of mapping waits for a read in place that is under way, and
    /// a read in place waits for a change of mapping.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading();
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping();
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping();
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading();
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }
}

#[cfg(test)]
mod unchanged_tests {
    use super::*;
    use std::sync::Arc;

    /// Blocks far past any address the tests map: no other test guards them.
    const FAR: usize = (1 << 46) / BLOCK;

    fn guard_fake(first: usize, n: usize) {
        let mut g = guards(crate::lockmeter::Site::Other);
        for b in first..first + n {
            g.protected.insert(b);
        }
    }

    fn unguard_fake(first: usize, n: usize, w: &Arc<Unchanged>) {
        unwatch_unchanged(first * BLOCK, n * BLOCK, w);
        let mut g = guards(crate::lockmeter::Site::Other);
        for b in first..first + n {
            g.protected.remove(&b);
        }
    }

    /// A refresh of guarded memory is noted current; the next ask for it
    /// is skipped; a write to one block takes only that block's bytes out.
    #[test]
    fn refreshed_guarded_range_is_skipped_until_written() {
        let first = FAR;
        guard_fake(first, 4);
        let w = Arc::new(Unchanged::default());
        let (start, len) = (first * BLOCK, 4 * BLOCK);
        let seen = watch_unchanged(start, len, &w).expect("all blocks guarded");
        w.note(start, start + len, seen);
        assert!(w.covers(start, len), "refreshed and not written: skipped");
        assert!(w.covers(start + BLOCK + 8, 100));
        // The title writes block 2 (the guard opens).
        open_here(&mut guards(crate::lockmeter::Site::Other), (first + 2) * BLOCK, (first + 2) * BLOCK + 1);
        assert!(!w.covers(start, len), "a written block is refreshed again");
        assert!(!w.covers((first + 2) * BLOCK, 16));
        assert!(w.covers(start, 2 * BLOCK), "the blocks before it stay current");
        assert!(w.covers((first + 3) * BLOCK, BLOCK), "the block after it stays current");
        unguard_fake(first, 4, &w);
    }

    /// A write during the refresh: nothing is noted current.
    #[test]
    fn a_write_during_the_refresh_notes_nothing() {
        let first = FAR + 64;
        guard_fake(first, 2);
        let w = Arc::new(Unchanged::default());
        let (start, len) = (first * BLOCK, 2 * BLOCK);
        let seen = watch_unchanged(start, len, &w).expect("all blocks guarded");
        open_here(&mut guards(crate::lockmeter::Site::Other), start, start + 1);
        w.note(start, start + len, seen);
        assert!(!w.covers(start, len));
        assert!(!w.covers(start + BLOCK, 8));
        unguard_fake(first, 2, &w);
    }

    /// A refresh registers its copy in the guard it makes, under the
    /// guards' lock: the range is current right after the first refresh,
    /// a write after it takes it out, and a write during the refresh (after
    /// its guard) leaves nothing noted.
    #[test]
    fn a_refresh_registers_in_its_guard() {
        let bytes = 2 * BLOCK;
        let (section, one, two) = super::tests::two_views_for_others(bytes);
        let w = Arc::new(Unchanged::default());
        // Written before: the first refresh guards it and holds it current.
        // SAFETY: inside our view.
        unsafe { (one as *mut u8).write_volatile(5) };
        refresh_watching(&w, one, one + bytes, || guard(one, one + bytes));
        assert!(w.covers(one, bytes), "current after one refresh");
        // A write through the view faults, opens its block, takes it out.
        // SAFETY: inside our view (the guard's handler opens it).
        unsafe { ((one + BLOCK + 8) as *mut u8).write_volatile(7) };
        assert!(!w.covers(one, bytes));
        assert!(w.covers(one, BLOCK), "the other block stays current");
        // A write during the refresh, after its guard: nothing noted.
        let w2 = Arc::new(Unchanged::default());
        refresh_watching(&w2, one, one + bytes, || {
            let written = guard(one, one + bytes);
            // SAFETY: as above.
            unsafe { ((one + 16) as *mut u8).write_volatile(9) };
            written
        });
        assert!(!w2.covers(one, bytes));
        assert!(!w2.covers(one, 8));
        unwatch_unchanged(one, bytes, &w);
        unwatch_unchanged(one, bytes, &w2);
        super::tests::drop_views(section, &[one, two], bytes);
    }

    /// A write the guards are not told of as a page's (a ReadFile into
    /// title memory, `title_memory_written` with `pages` false, which by
    /// default opens nothing) still takes the bytes out of every copy
    /// holding them current (run P5-0's gap).
    #[test]
    fn a_read_file_into_guarded_memory_takes_its_bytes_out() {
        let first = FAR + 256;
        guard_fake(first, 3);
        let w = Arc::new(Unchanged::default());
        let (start, len) = (first * BLOCK, 3 * BLOCK);
        let seen = watch_unchanged(start, len, &w).unwrap();
        w.note(start, start + len, seen);
        assert!(w.covers(start, len));
        title_memory_written(start + BLOCK + 100, 50, false);
        assert!(!w.covers(start, len));
        assert!(!w.covers(start + BLOCK + 100, 50));
        assert!(w.covers(start, BLOCK), "the rest stays current");
        assert!(w.covers(start + 2 * BLOCK, BLOCK));
        unguard_fake(first, 3, &w);
    }

    /// A copy is registered once per 4 MiB bucket and stays registered:
    /// two writes in the same bucket, one after the other, both take their
    /// bytes out; the copy's registration does not grow with its blocks.
    #[test]
    fn watchers_are_kept_per_bucket() {
        let first = (FAR + 512) & !((1 << BUCKET_SHIFT) - 1);
        let blocks = 2 << BUCKET_SHIFT;
        guard_fake(first, blocks);
        let w = Arc::new(Unchanged::default());
        let (start, len) = (first * BLOCK, blocks * BLOCK);
        let seen = watch_unchanged(start, len, &w).unwrap();
        w.note(start, start + len, seen);
        {
            let g = guards(crate::lockmeter::Site::Other);
            let buckets: usize = g.watchers.range(first >> BUCKET_SHIFT..(first + blocks) >> BUCKET_SHIFT).count();
            assert_eq!(buckets, 2, "one entry per bucket, not per block");
        }
        open_here(&mut guards(crate::lockmeter::Site::Other), start + 3 * BLOCK, start + 3 * BLOCK + 1);
        assert!(!w.covers(start + 3 * BLOCK, 8));
        open_here(&mut guards(crate::lockmeter::Site::Other), start + 5 * BLOCK, start + 5 * BLOCK + 1);
        assert!(!w.covers(start + 5 * BLOCK, 8), "still registered after the first opening");
        assert!(w.covers(start + 4 * BLOCK, BLOCK));
        assert!(w.covers(start + 6 * BLOCK, len - 6 * BLOCK));
        unguard_fake(first, blocks, &w);
    }

    /// A remapping with VICTORY_VIEW_KNOWN keeps its blocks known as view
    /// memory (no failed write-watch asking after it) and still takes
    /// their guards and current bytes out; without the switch it forgets
    /// them, as before.
    #[test]
    fn a_remapping_keeps_its_blocks_known_as_view_memory() {
        let first = FAR + 1024;
        guard_fake(first, 2);
        let w = Arc::new(Unchanged::default());
        let (start, len) = (first * BLOCK, 2 * BLOCK);
        let seen = watch_unchanged(start, len, &w).unwrap();
        w.note(start, start + len, seen);
        {
            let mut g = guards(crate::lockmeter::Site::Other);
            g.protected.remove(&first);
            g.protected.remove(&(first + 1));
            g.clean.remove(first);
            g.clean.remove(first + 1);
            take_out_range(&g, start, start + len);
            remapped_viewed(&mut g, start, len, true);
            assert!(g.viewed.contains(&first) && g.viewed.contains(&(first + 1)), "kept as view memory");
            assert!(!g.protected.contains(&first), "its guard is gone: the bytes are copied again");
        }
        assert!(!w.covers(start, len), "nothing of it is held current");
        {
            let mut g = guards(crate::lockmeter::Site::Other);
            remapped_viewed(&mut g, start, len, false);
            assert!(!g.viewed.contains(&first) && !g.viewed.contains(&(first + 1)), "forgotten without the switch");
        }
        unwatch_unchanged(start, len, &w);
    }

    /// Memory not all guarded (write-watched, or a block open) is never
    /// held current.
    #[test]
    fn unguarded_memory_is_never_held() {
        let first = FAR + 128;
        guard_fake(first, 1);
        let w = Arc::new(Unchanged::default());
        assert!(watch_unchanged(first * BLOCK, 2 * BLOCK, &w).is_none());
        unguard_fake(first, 1, &w);
    }

    /// Shaders that may write the copy, or the copy going: all out.
    #[test]
    fn clear_and_remap_take_everything_out() {
        let first = FAR + 192;
        guard_fake(first, 2);
        let w = Arc::new(Unchanged::default());
        let (start, len) = (first * BLOCK, 2 * BLOCK);
        let seen = watch_unchanged(start, len, &w).unwrap();
        w.note(start, start + len, seen);
        w.clear();
        assert!(!w.covers(start, 8));
        let seen = watch_unchanged(start, len, &w).unwrap();
        w.note(start, start + len, seen);
        assert!(w.covers(start, len));
        // As `title_memory_remapped` does for every watched block of its range.
        {
            let mut g = guards(crate::lockmeter::Site::Other);
            invalidate(&mut g, first + 1);
        }
        assert!(w.covers(start, BLOCK));
        assert!(!w.covers(start + BLOCK, 8));
        unguard_fake(first, 2, &w);
    }

    /// `VICTORY_OPEN_ONE_VIEW=1` keeps the other view `PAGE_READONLY` on a
    /// write fault, yet MUST still invalidate `Unchanged` watchers registered
    /// on that view so `VICTORY_SKIP_UNCHANGED` never keeps stale bytes.
    #[test]
    fn open_one_view_invalidates_watchers_on_other_view() {
        let _sw = super::tests::TestSwitches::set(Some(true), Some(std::time::Duration::ZERO));
        let bytes = 2 * BLOCK;
        let (section, one, two) = super::tests::two_views_for_others(bytes);
        let w = Arc::new(Unchanged::default());
        refresh_watching(&w, one, one + bytes, || guard(one, one + bytes));
        assert!(w.covers(one, bytes), "view one registered as current");
        assert!(guard(two, two + bytes).is_some());
        // SAFETY: `two` is mapped above and opened on fault by the handler.
        unsafe { ((two + BLOCK + 8) as *mut u8).write_volatile(42) };
        assert!(!writable(one), "view one remains PAGE_READONLY under OPEN_ONE_VIEW");
        assert!(
            !w.covers(one + BLOCK, 8),
            "Unchanged watcher on view one MUST be invalidated despite remaining PAGE_READONLY"
        );
        assert!(w.covers(one, BLOCK), "unaffected block on view one stays current");
        assert_eq!(guard(one, one + bytes), Some(vec![(one + BLOCK) / PAGE]));
        unwatch_unchanged(one, bytes, &w);
        super::tests::drop_views(section, &[one, two], bytes);
    }
}
