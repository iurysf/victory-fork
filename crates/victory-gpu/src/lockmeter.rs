//! The guards' lock (`crate::watch`) measured, passively: who waits for it,
//! by the role of the thread and, on the submitting thread, by the step of
//! preparing; who held it meanwhile; and, with `VICTORY_GUARDS_METER`, how
//! long each caller holds it.
//!
//! The first A/A in the yard with the frame log's "preparing by step" (exe
//! 7f3eb89, every new switch off) found 55-62 ms a frame of waits for the
//! guards' lock, summed over the threads, with nothing saying whose. Nothing
//! here changes what the lock does.
//!
//! Cost: the waits are timed only when the lock is taken already (a failed
//! `try_lock` first; the thread's name is read then); every taking stores
//! its caller and context in two atomics. The holds are timed only with
//! `VICTORY_GUARDS_METER=N` (off by default): every Nth taking of a thread
//! (1: all), two clock reads and three atomics each; the line says what
//! that cost a frame, measured.

use std::cell::Cell;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};

/// What a thread is doing, for its waits ([`within`]): the steps of
/// preparing a submission as the frame log names them, and copy-back.
#[derive(Clone, Copy)]
#[repr(u8)]
pub(crate) enum Ctx {
    Ask = 1,
    Steps,
    Flush,
    Snapshots,
    Card,
    CopyBack,
}

thread_local! {
    static CTX: Cell<u8> = const { Cell::new(0) };
    /// Takings by this thread, for `VICTORY_GUARDS_METER`'s one in N.
    static TAKINGS: Cell<u32> = const { Cell::new(0) };
}

/// Until dropped: this thread is doing `ctx`; the one before comes back.
pub(crate) struct Within(u8);

impl Drop for Within {
    fn drop(&mut self) {
        CTX.with(|c| c.set(self.0));
    }
}

/// This thread does `ctx` until the returned value is dropped.
pub(crate) fn within(ctx: Ctx) -> Within {
    Within(CTX.with(|c| c.replace(ctx as u8)))
}

/// Who takes the lock: the call site.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Site {
    KnownView,
    DifferingKind,
    /// Tests only (`watch_unchanged` is theirs).
    #[cfg_attr(not(test), allow(dead_code))]
    WatchUnchanged,
    UnwatchUnchanged,
    UnchangedGone,
    GuardClean,
    GuardKnown,
    GuardLocked,
    Writing,
    TitleWritten,
    ViewsMapped,
    Remapped,
    Handler,
    Other,
}

const SITES: usize = 14;
const SITE_NAMES: [&str; SITES] = [
    "known_view",
    "differing_kind",
    "watch_unchanged",
    "unwatch_unchanged",
    "unchanged_gone",
    "guard, a range all clean",
    "guard, whether the block is known",
    "guard_locked (walk, protect, aliases)",
    "writing (before the runtime's kernel writes)",
    "title_memory_written",
    "views_mapped",
    "title_memory_remapped",
    "the exception handler (open)",
    "other",
];

const ROLES: usize = 12;
const ROLE_NAMES: [&str; ROLES] = [
    "submission, the one asking",
    "submission, the lists' steps",
    "submission, the last flush",
    "submission, snapshots",
    "submission, the card's plan",
    "the submit thread outside the steps",
    "copy-back (settling, the relay)",
    "the exception handler (title threads)",
    "the runtime's calls (kernel writes, maps, remaps)",
    "wide helpers (victory-wide)",
    "other threads of ours",
    "title threads otherwise (recording)",
];

/// The role of this thread taking the lock at `site`.
fn role(site: Site) -> usize {
    if site == Site::Handler {
        return 7;
    }
    match CTX.with(Cell::get) {
        c @ 1..=5 => return usize::from(c) - 1,
        6 => return 6,
        _ => {}
    }
    if matches!(site, Site::Writing | Site::TitleWritten | Site::ViewsMapped | Site::Remapped) {
        return 8;
    }
    match std::thread::current().name() {
        Some("victory-submit") => 5,
        Some("victory-relay" | "victory-late-wait") => 6,
        Some(n) if n.starts_with("victory-wide") => 9,
        Some(n) if n.starts_with("victory") => 10,
        _ => 11,
    }
}

/// Waits by role (µs, count); by the caller holding the lock when the wait
/// began (µs); of those, µs while the holder was preparing a submission.
static WAIT_US: [AtomicU64; ROLES] = [const { AtomicU64::new(0) }; ROLES];
static WAIT_N: [AtomicU64; ROLES] = [const { AtomicU64::new(0) }; ROLES];
static WAIT_BY_HOLDER: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
static WAIT_HOLDER_PREPARING: AtomicU64 = AtomicU64::new(0);
/// The caller holding the lock now, and its context.
static HOLDER: AtomicU8 = AtomicU8::new(Site::Other as u8);
static HOLDER_CTX: AtomicU8 = AtomicU8::new(0);
/// With the meter: takings by caller; their time held, ns (the timed ones
/// times N); the longest single hold, ns; takings timed. The alias part of
/// `guard_locked`'s protections, ns (all, with the meter).
static HOLD_N: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
static HOLD_NS: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
static HOLD_MAX_NS: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
static TIMED: AtomicU64 = AtomicU64::new(0);
static ALIAS_NS: AtomicU64 = AtomicU64::new(0);

/// `VICTORY_GUARDS_METER=N`: every Nth taking's hold timed (1: all); 0, unset: none.
fn every() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_GUARDS_METER").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// Whether the holds are measured.
pub(crate) fn on() -> bool {
    every() > 0
}

/// A lock taken through [`lock`]: the guard, and its hold timed on drop.
pub(crate) struct Held<T: 'static> {
    guard: std::sync::MutexGuard<'static, T>,
    site: Site,
    since: Option<std::time::Instant>,
}

impl<T: 'static> Held<T> {
    /// The same hold goes on for another caller (`guard` into `guard_locked`).
    pub(crate) fn now_for(&mut self, site: Site) {
        self.site = site;
        HOLDER.store(site as u8, Relaxed);
    }
}

impl<T: 'static> std::ops::Deref for Held<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T: 'static> std::ops::DerefMut for Held<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}

impl<T: 'static> Drop for Held<T> {
    fn drop(&mut self) {
        if let Some(since) = self.since {
            let ns = since.elapsed().as_nanos() as u64;
            let i = self.site as usize;
            HOLD_NS[i].fetch_add(ns * u64::from(every()), Relaxed);
            HOLD_MAX_NS[i].fetch_max(ns, Relaxed);
        }
    }
}

/// Takes `mutex` for `site`, as `lock()` would (a poisoned lock is taken
/// as it is), measuring as the module says.
pub(crate) fn lock<T>(mutex: &'static std::sync::Mutex<T>, site: Site) -> Held<T> {
    let guard = match mutex.try_lock() {
        Ok(g) => g,
        Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => {
            let holder = usize::from(HOLDER.load(Relaxed)).min(SITES - 1);
            let preparing = matches!(HOLDER_CTX.load(Relaxed), 1..=5);
            let waiting = std::time::Instant::now();
            let g = mutex.lock().unwrap_or_else(|e| e.into_inner());
            let us = waiting.elapsed().as_micros() as u64;
            crate::framelog::add(crate::framelog::T::GuardsLockUs, us);
            let r = role(site);
            WAIT_US[r].fetch_add(us, Relaxed);
            WAIT_N[r].fetch_add(1, Relaxed);
            WAIT_BY_HOLDER[holder].fetch_add(us, Relaxed);
            if preparing {
                WAIT_HOLDER_PREPARING.fetch_add(us, Relaxed);
            }
            g
        }
    };
    HOLDER.store(site as u8, Relaxed);
    HOLDER_CTX.store(CTX.with(Cell::get), Relaxed);
    let n = every();
    let since = (n > 0).then(|| {
        HOLD_N[site as usize].fetch_add(1, Relaxed);
        TAKINGS.with(|t| {
            let k = t.get().wrapping_add(1);
            t.set(k);
            k % n == 0
        })
    });
    let since = since.unwrap_or(false).then(|| {
        TIMED.fetch_add(1, Relaxed);
        std::time::Instant::now()
    });
    Held { guard, site, since }
}

/// With the meter: `ns` spent guarding the other views of pages, inside a hold.
pub(crate) fn alias_ns(ns: u64) {
    ALIAS_NS.fetch_add(ns, Relaxed);
}

/// The cost of one timed hold's clock reads, ns, measured once.
fn timing_cost_ns() -> f64 {
    static COST: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *COST.get_or_init(|| {
        let started = std::time::Instant::now();
        let mut last = started;
        for _ in 0..10_000 {
            last = std::hint::black_box(std::time::Instant::now());
        }
        // Two reads a timed hold.
        (last - started).as_nanos() as f64 / 10_000.0 * 2.0
    })
}

/// For the frame log's reading of `frames` frames: the line, the counts
/// taken since the last one.
pub(crate) fn line(frames: f64) -> String {
    let f = frames.max(1.0);
    let take = |a: &AtomicU64| a.swap(0, Relaxed) as f64;
    let mut waits: Vec<String> = Vec::new();
    let mut total = 0.0;
    for r in 0..ROLES {
        let us = take(&WAIT_US[r]);
        let n = take(&WAIT_N[r]);
        total += us;
        if n > 0.0 {
            waits.push(format!("{} {:.2} ms in {:.0}", ROLE_NAMES[r], us / 1e3 / f, n / f));
        }
    }
    let mut holders: Vec<String> = Vec::new();
    for s in 0..SITES {
        let us = take(&WAIT_BY_HOLDER[s]);
        if us > 0.0 {
            holders.push(format!("{} {:.2}", SITE_NAMES[s], us / 1e3 / f));
        }
    }
    let preparing = take(&WAIT_HOLDER_PREPARING);
    let mut text = format!(
        "the guards' lock, a frame: waits {:.2} ms; by who waited: {}; by who held it (ms): {}; of the waits, {:.2} ms \
         behind a holder preparing a submission",
        total / 1e3 / f,
        if waits.is_empty() { "none".to_owned() } else { waits.join(", ") },
        if holders.is_empty() { "none".to_owned() } else { holders.join(", ") },
        preparing / 1e3 / f
    );
    let n = every();
    if n == 0 {
        text.push_str("; holds not measured (VICTORY_GUARDS_METER unset)");
        return text;
    }
    let mut holds: Vec<String> = Vec::new();
    for s in 0..SITES {
        let calls = take(&HOLD_N[s]);
        let ns = take(&HOLD_NS[s]);
        let max = HOLD_MAX_NS[s].swap(0, Relaxed) as f64;
        if calls > 0.0 {
            holds.push(format!(
                "{} {:.0} calls {:.2} ms (longest {:.2} ms)",
                SITE_NAMES[s],
                calls / f,
                ns / 1e6 / f,
                max / 1e6
            ));
        }
    }
    let timed = take(&TIMED);
    let alias = take(&ALIAS_NS);
    text.push_str(&format!(
        "; holds by caller, a frame (VICTORY_GUARDS_METER={n}: one taking in {n} timed, times {n}): {}; of them guarding \
         the other views of pages {:.2} ms; the meter's own clock reads {:.3} ms a frame",
        holds.join(", "),
        alias / 1e6 / f,
        timed * timing_cost_ns() / 1e6 / f
    ));
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wait is put down to the waiter's role and to the caller holding
    /// the lock, and to a holder preparing a submission.
    #[test]
    fn a_wait_goes_to_the_waiters_role_and_to_the_holder() {
        static M: std::sync::Mutex<u32> = std::sync::Mutex::new(0);
        let (relay, by_holder, preparing) = (
            WAIT_N[6].load(Relaxed),
            WAIT_BY_HOLDER[Site::GuardLocked as usize].load(Relaxed),
            WAIT_HOLDER_PREPARING.load(Relaxed),
        );
        let (taken_tx, taken_rx) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || {
            let _step = within(Ctx::Steps);
            let mut g = lock(&M, Site::GuardClean);
            g.now_for(Site::GuardLocked);
            *g += 1;
            taken_tx.send(()).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(30));
        });
        taken_rx.recv().unwrap();
        let waiter = std::thread::Builder::new()
            .name("victory-relay".into())
            .spawn(|| {
                let g = lock(&M, Site::KnownView);
                assert_eq!(*g, 1);
            })
            .unwrap();
        holder.join().unwrap();
        waiter.join().unwrap();
        assert!(WAIT_N[6].load(Relaxed) > relay, "the relay's wait is copy-back's");
        assert!(WAIT_BY_HOLDER[Site::GuardLocked as usize].load(Relaxed) > by_holder, "held by guard_locked");
        assert!(WAIT_HOLDER_PREPARING.load(Relaxed) > preparing, "the holder was preparing (the lists' steps)");
        // The context is the thread's own: none here.
        assert_eq!(CTX.with(Cell::get), 0);
    }
}
