//! Title bytes put in under flights' writes, on threads of their own
//! (`VICTORY_UNDER_WIDE=N`; off by default; the A/B may turn it).
//!
//! With flights in the air a submission compares, page by page, title
//! memory with what each copy in the card's memory held when shaders were
//! given it (`Shadow::merge_under_writes`, in the submission's one asking),
//! and sends the changed runs up as dword patches (`Shadow::card_patches_shorter`,
//! in the card's plan). In the yard that is 333 pages, some 530,000 runs and
//! 600,000 dwords a frame; the bench measured 20.75 ms of it on the
//! submitting thread (13.17 compare and merge, 7.58 patches).
//!
//! - `N` = 1: the compare is made 16 bytes at a time with SSE2 masks
//!   ([`changed_runs`]), the same runs as the scalar one.
//! - `N` >= 2: also the pages of a merge are compared and put in on the
//!   submitting thread and `N - 1` threads of a pool of their own
//!   (`victory-under-*`, no core asked for), and the patches of a copy's
//!   runs made in pieces on them, each piece written to its own place.
//!
//! What comes out is the same as on one thread, in the same order: the room
//! a merge may take, the runs noted for the card and the pages noted taken
//! are done in page order on the submitting thread after the compares
//! (`Shadow::merge_under_writes`); the patches' pieces never split a dword.
//!
//! After fxceless/victory-fork c36971b (the compare by masks) and
//! avatarmanz/VIctory-7900XTX b620284 (the merges and the patches on
//! threads of their own, there with AVX2 and pairs of dwords; here SSE2 and
//! the `[dword, bytes, mask]` patches `crate::host::CARD_PATCH` takes, as
//! the bench's arm 1b). Both GPL-3.0.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex};

/// `VICTORY_UNDER_WIDE`: the threads (the submitting one among them), 0 off,
/// at most 8. Read once; the A/B may turn it (`UNDER_WIDE`, B 4 or the value
/// given).
pub(crate) fn threads() -> usize {
    static RUN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let run = *RUN.get_or_init(|| {
        std::env::var("VICTORY_UNDER_WIDE").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0)
    });
    #[cfg(test)]
    if let Some(n) = test_threads() {
        return n;
    }
    crate::ab::number(crate::ab::Key::UnderWide, run).min(8) as usize
}

#[cfg(test)]
thread_local! {
    static TEST_THREADS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn test_threads() -> Option<usize> {
    TEST_THREADS.with(std::cell::Cell::get)
}

/// Tests: [`threads`] gives `n` on this thread until the value is dropped.
#[cfg(test)]
pub(crate) struct TestThreads(Option<usize>);

#[cfg(test)]
impl Drop for TestThreads {
    fn drop(&mut self) {
        TEST_THREADS.with(|c| c.set(self.0));
    }
}

#[cfg(test)]
pub(crate) fn test_with(n: usize) -> TestThreads {
    TestThreads(TEST_THREADS.with(|c| c.replace(Some(n))))
}

/// The runs of bytes where `now` differs from `was` (of one length, the
/// shorter's), as `crate::shadow`'s `changed_runs` gives them: (offset,
/// bytes) in order, every differing byte in a run and no equal one.
#[cfg(target_arch = "x86_64")]
pub(crate) fn changed_runs(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    use core::arch::x86_64::{__m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8};
    let n = now.len().min(was.len());
    let (now, was) = (&now[..n], &was[..n]);
    let mut runs = Vec::new();
    let mut open: Option<usize> = None;
    let mut base = 0;
    while base + 64 <= n {
        let mut differ = 0u64;
        for part in 0..4 {
            let at = base + 16 * part;
            // SAFETY: `at + 16 <= base + 64 <= n` in both slices; SSE2 is
            // part of x86-64.
            let equal = unsafe {
                let a = _mm_loadu_si128(now.as_ptr().add(at).cast::<__m128i>());
                let b = _mm_loadu_si128(was.as_ptr().add(at).cast::<__m128i>());
                _mm_movemask_epi8(_mm_cmpeq_epi8(a, b)) as u32 & 0xffff
            };
            differ |= u64::from(!equal & 0xffff) << (16 * part);
        }
        if differ == 0 {
            if let Some(first) = open.take() {
                runs.push((first, base - first));
            }
        } else if differ != u64::MAX || open.is_none() {
            let mut at = 0u32;
            while at < 64 {
                match open {
                    None => {
                        let skip = (differ >> at).trailing_zeros();
                        if skip >= 64 - at {
                            break;
                        }
                        at += skip;
                        open = Some(base + at as usize);
                    }
                    Some(first) => {
                        let skip = (!differ >> at).trailing_zeros();
                        if skip >= 64 - at {
                            break;
                        }
                        at += skip;
                        runs.push((first, base + at as usize - first));
                        open = None;
                    }
                }
            }
        }
        base += 64;
    }
    for i in base..n {
        match (open, now[i] != was[i]) {
            (None, true) => open = Some(i),
            (Some(first), false) => {
                runs.push((first, i - first));
                open = None;
            }
            _ => {}
        }
    }
    if let Some(first) = open {
        runs.push((first, n - first));
    }
    runs
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn changed_runs(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    let n = now.len().min(was.len());
    let mut runs = Vec::new();
    let mut i = 0;
    while i < n {
        if now[i] != was[i] {
            let first = i;
            while i < n && now[i] != was[i] {
                i += 1;
            }
            runs.push((first, i - first));
        } else {
            i += 1;
        }
    }
    runs
}

/// Runs given to one piece of the patches at least (the bench's 4,096).
pub(crate) const RUNS_A_PIECE: usize = 4096;

/// The dword patches of one piece of runs: how many ([`dword_patches_of`]
/// would give, the piece alone).
fn patches_in(runs: &[(usize, usize)]) -> usize {
    let mut dwords = 0usize;
    let mut last = usize::MAX;
    for &(a, b) in runs {
        if a >= b {
            continue;
        }
        let (first, end) = (a / 4, (b - 1) / 4);
        dwords += end - first + usize::from(first != last);
        last = end;
    }
    dwords
}

/// `crate::shadow::dword_patches_of` of `ranges` (offsets of `data`, in
/// order, apart), appended to `out`, made in pieces on the pool's threads
/// when there are [`RUNS_A_PIECE`] ranges or more and [`threads`] is 2 or
/// more; else on this thread. The same patches in the same order.
pub(crate) fn dword_patches(ranges: &[(usize, usize)], data: &[u8], out: &mut Vec<[u32; 3]>) {
    let n = threads();
    if n < 2 || ranges.len() < RUNS_A_PIECE {
        crate::shadow::dword_patches_of(ranges.iter().copied(), data, out);
        return;
    }
    let started = std::time::Instant::now();
    // Pieces of runs that share no dword with the next piece, and where each
    // piece's patches go.
    let mut pieces: Vec<(usize, usize, usize)> = Vec::new();
    let first = out.len();
    let mut total = 0usize;
    let mut lo = 0usize;
    while lo < ranges.len() {
        let mut hi = (lo + RUNS_A_PIECE).min(ranges.len());
        while hi < ranges.len() && (ranges[hi - 1].1.max(ranges[hi - 1].0 + 1) - 1) / 4 == ranges[hi].0 / 4 {
            hi += 1;
        }
        let count = patches_in(&ranges[lo..hi]);
        if count > 0 {
            pieces.push((lo, hi, total));
            total += count;
        }
        lo = hi;
    }
    out.resize(first + total, [0; 3]);
    let to = out[first..].as_mut_ptr() as usize;
    each(n, pieces.len(), &|i: usize| {
        let (lo, hi, at) = pieces[i];
        let mut piece = Vec::with_capacity(patches_in(&ranges[lo..hi]));
        crate::shadow::dword_patches_of(ranges[lo..hi].iter().copied(), data, &mut piece);
        // SAFETY: the piece's place in `out` is `at..at + its count`, which
        // no other piece writes; `out` outlives the call.
        unsafe { std::ptr::copy_nonoverlapping(piece.as_ptr(), (to as *mut [u32; 3]).add(at), piece.len()) };
    });
    crate::framelog::add(crate::framelog::T::UnderWideUs, started.elapsed().as_micros() as u64);
}

/// Runs `f(0..count)` on this thread and `n - 1` threads of the pool, each
/// index once, and returns when all are done. One caller at a time; a
/// panic of `f` on a pool thread comes back as a panic here.
pub(crate) fn each(n: usize, count: usize, f: &(dyn Fn(usize) + Sync)) {
    if n < 2 || count < 2 {
        for i in 0..count {
            f(i);
        }
        return;
    }
    let pool = pool(n);
    pool.each(count, f);
}

/// The pool for `n` threads (made again when `n` changes, the old one let go
/// when its last caller is done).
fn pool(n: usize) -> Arc<Pool> {
    static POOL: Mutex<Option<Arc<Pool>>> = Mutex::new(None);
    let mut p = POOL.lock().unwrap_or_else(|e| e.into_inner());
    match p.as_ref() {
        Some(pool) if pool.workers + 1 == n => pool.clone(),
        _ => {
            if let Some(old) = p.take() {
                old.stop();
            }
            let pool = Pool::new(n - 1);
            *p = Some(pool.clone());
            pool
        }
    }
}

/// A job handed to the pool's threads: the indices taken in turn.
struct Job {
    count: usize,
    next: AtomicUsize,
    /// `&(dyn Fn(usize) + Sync)` of the caller, alive until every worker
    /// says done.
    f: *const (dyn Fn(usize) + Sync),
    panicked: AtomicBool,
}

// SAFETY: `f` is `Sync` and outlives the job's use on every thread
// (`Pool::each` waits for every worker before returning).
unsafe impl Send for Job {}
// SAFETY: as above; the rest is atomics.
unsafe impl Sync for Job {}

impl Job {
    /// Takes indices until none are left.
    fn work(&self) {
        loop {
            let i = self.next.fetch_add(1, Relaxed);
            if i >= self.count {
                return;
            }
            // SAFETY: see `Job::f`.
            let f = unsafe { &*self.f };
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(i))).is_err() {
                self.panicked.store(true, Relaxed);
            }
        }
    }
}

struct State {
    /// The job of this round, its number, and the workers done with it.
    job: Option<Arc<Job>>,
    round: u64,
    done: usize,
    stop: bool,
}

struct Pool {
    state: Mutex<State>,
    work: Condvar,
    done: Condvar,
    /// The threads asked for, and those made.
    workers: usize,
    alive: AtomicUsize,
    /// One caller at a time.
    caller: Mutex<()>,
}

impl Pool {
    fn new(workers: usize) -> Arc<Pool> {
        let pool = Arc::new(Pool {
            state: Mutex::new(State { job: None, round: 0, done: 0, stop: false }),
            work: Condvar::new(),
            done: Condvar::new(),
            workers,
            alive: AtomicUsize::new(0),
            caller: Mutex::new(()),
        });
        for id in 0..workers {
            let p = pool.clone();
            // A thread that cannot be made leaves its share to the others
            // and the caller: [`Pool::each`] waits for those made.
            if std::thread::Builder::new().name(format!("victory-under-{id}")).spawn(move || p.worker()).is_ok() {
                pool.alive.fetch_add(1, Relaxed);
            }
        }
        pool
    }

    /// Ends the pool's threads (after the caller of the moment is done);
    /// they hold the pool, which goes with the last of them.
    fn stop(&self) {
        let _one = self.caller.lock().unwrap_or_else(|e| e.into_inner());
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.stop = true;
        // A caller that took this pool before it was stopped does the work
        // alone: no thread is waited for.
        self.alive.store(0, Relaxed);
        self.work.notify_all();
    }

    fn worker(self: Arc<Self>) {
        let mut seen = 0u64;
        loop {
            let job = {
                let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
                while !s.stop && (s.round == seen || s.job.is_none()) {
                    s = self.work.wait(s).unwrap_or_else(|e| e.into_inner());
                }
                if s.stop {
                    return;
                }
                seen = s.round;
                s.job.clone()
            };
            if let Some(job) = job {
                job.work();
            }
            let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if s.round == seen {
                s.done += 1;
                if s.done >= self.alive.load(Relaxed) {
                    self.done.notify_one();
                }
            }
        }
    }

    fn each(&self, count: usize, f: &(dyn Fn(usize) + Sync)) {
        let _one = self.caller.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY (of the lifetime's erasure): the job's pointer is used only
        // until every worker says done, below.
        let f: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
        let job = Arc::new(Job { count, next: AtomicUsize::new(0), f, panicked: AtomicBool::new(false) });
        {
            let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
            s.round += 1;
            s.job = Some(job.clone());
            s.done = 0;
            self.work.notify_all();
        }
        job.work();
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while s.done < self.alive.load(Relaxed) {
            s = self.done.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        s.job = None;
        drop(s);
        assert!(!job.panicked.load(Relaxed), "a job of VICTORY_UNDER_WIDE's pool panicked");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
        let n = now.len().min(was.len());
        let mut runs = Vec::new();
        let mut i = 0;
        while i < n {
            if now[i] != was[i] {
                let first = i;
                while i < n && now[i] != was[i] {
                    i += 1;
                }
                runs.push((first, i - first));
            } else {
                i += 1;
            }
        }
        runs
    }

    /// The bench's oracle 1: the compare by masks gives the runs of the
    /// byte-wise compare at every length and alignment around 16 and 64.
    #[test]
    fn the_compare_by_masks_gives_the_bytewise_runs() {
        let mut seed = 7u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed >> 8
        };
        for n in [0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192, 193, 256, 1024, 1025] {
            for (from, to) in [(0, n), (0, 64.min(n)), (63.min(n), 65.min(n)), (64.min(n), n), (1.min(n), n.saturating_sub(1)), (n.saturating_sub(1), n), (0, 1.min(n))] {
                if from > to {
                    continue;
                }
                let was = vec![7u8; n];
                let mut now = was.clone();
                for b in &mut now[from..to] {
                    *b = 9;
                }
                assert_eq!(changed_runs(&now, &was), scalar(&now, &was), "{n} bytes, {from}..{to}");
                let now: Vec<u8> = (0..n).map(|i| if i % 2 == 0 && (from..to).contains(&i) { 9 } else { 7 }).collect();
                assert_eq!(changed_runs(&now, &was), scalar(&now, &was), "{n} bytes, every other of {from}..{to}");
            }
        }
        for round in 0..300 {
            let n = 1 + next() as usize % 70_000;
            let was: Vec<u8> = (0..n).map(|_| next() as u8).collect();
            let mut now = was.clone();
            for _ in 0..next() as usize % (n / 4 + 1) {
                now[next() as usize % n] = next() as u8;
            }
            assert_eq!(changed_runs(&now, &was), scalar(&now, &was), "random {round}, {n} bytes");
        }
        // Slices of two lengths: the shorter's.
        assert_eq!(changed_runs(&[1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 2, 0]), vec![(2, 1)]);
    }

    /// The bench's oracle 3a: the patches made in pieces on 1, 3, 4 and 6
    /// threads are the patches made on one, runs that share dwords among
    /// them.
    #[test]
    fn patches_made_in_pieces_are_the_patches_made_on_one_thread() {
        let mut seed = 77u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed >> 8
        };
        const PAGE: usize = 64 << 10;
        let held: Vec<u8> = (0..5 * PAGE).map(|_| next() as u8).collect();
        let mut title = held.clone();
        for _ in 0..5 * 2200 {
            let at = next() as usize % title.len();
            title[at] ^= ((next() as u8) & 0xFE) | 1;
        }
        let ranges: Vec<(usize, usize)> = scalar(&title, &held).into_iter().map(|(a, n)| (a, a + n)).collect();
        assert!(ranges.len() > 2 * RUNS_A_PIECE, "pieces to make: {}", ranges.len());
        let mut one = vec![[1, 2, 3]];
        crate::shadow::dword_patches_of(ranges.iter().copied(), &title, &mut one);
        for n in [1, 3, 4, 6] {
            let _n = test_with(n);
            let mut wide = vec![[1, 2, 3]];
            dword_patches(&ranges, &title, &mut wide);
            assert_eq!(wide, one, "{n} threads");
        }
        // Runs ending and starting in one dword across a piece's edge.
        let close: Vec<(usize, usize)> = (0..3 * RUNS_A_PIECE).map(|k| (2 * k, 2 * k + 1)).collect();
        let mut one = Vec::new();
        crate::shadow::dword_patches_of(close.iter().copied(), &title, &mut one);
        let _n = test_with(4);
        let mut wide = Vec::new();
        dword_patches(&close, &title, &mut wide);
        assert_eq!(wide, one, "runs two to a dword");
    }

    /// The bench's oracle 4, on the merge's two halves as
    /// `Shadow::merge_under_writes_wide` has them: pages compared on 3, 4
    /// and 6 threads, their runs and bytes kept, then put in page by page
    /// on this thread, leave the copy's bytes and the runs of a compare and
    /// put-in page by page on one thread. (`merge_under_writes` itself
    /// needs write-watched title memory and a copy's pages taken: not here.)
    #[test]
    fn pages_compared_on_threads_and_put_in_in_order_are_the_merge_on_one() {
        const PAGE: usize = 64 << 10;
        let mut seed = 1234u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed >> 8
        };
        let pages: Vec<(Vec<u8>, Vec<u8>)> = (0..7)
            .map(|_| {
                let held: Vec<u8> = (0..PAGE).map(|_| next() as u8).collect();
                let mut title = held.clone();
                for _ in 0..1900 {
                    let at = next() as usize % (PAGE / 4) * 4;
                    for b in 0..4 {
                        title[at + b] = held[at + b] ^ (((next() as u8) & 0xFE) | 1);
                    }
                }
                (title, held)
            })
            .collect();
        let mut one: Vec<Vec<u8>> = pages.iter().map(|p| p.1.clone()).collect();
        let mut one_runs = Vec::new();
        for (held, (title, _)) in one.iter_mut().zip(&pages) {
            let runs = scalar(title, held);
            for &(r, m) in &runs {
                held[r..r + m].copy_from_slice(&title[r..r + m]);
            }
            one_runs.push(runs);
        }
        for n in [3, 4, 6] {
            let found: Vec<Mutex<Option<(Vec<(usize, usize)>, Vec<u8>)>>> = pages.iter().map(|_| Mutex::new(None)).collect();
            each(n, pages.len(), &|i| {
                let (title, held) = &pages[i];
                let runs = changed_runs(title, held);
                let mut bytes = Vec::new();
                for &(r, m) in &runs {
                    bytes.extend_from_slice(&title[r..r + m]);
                }
                *found[i].lock().unwrap() = Some((runs, bytes));
            });
            let mut wide: Vec<Vec<u8>> = pages.iter().map(|p| p.1.clone()).collect();
            let mut wide_runs = Vec::new();
            for (held, slot) in wide.iter_mut().zip(&found) {
                let (runs, bytes) = slot.lock().unwrap().take().unwrap();
                let mut taken = 0;
                for &(r, m) in &runs {
                    held[r..r + m].copy_from_slice(&bytes[taken..taken + m]);
                    taken += m;
                }
                wide_runs.push(runs);
            }
            assert_eq!(wide, one, "the copy's bytes, {n} threads");
            assert_eq!(wide_runs, one_runs, "the runs, {n} threads");
        }
    }

    /// Every index runs once, on the caller and the pool's threads, call
    /// after call, and the pool is made again when the threads change.
    #[test]
    fn the_pool_runs_every_index_once() {
        for n in [2usize, 4, 6, 4] {
            for count in [2usize, 3, 7, 100, 1000] {
                let hits: Vec<AtomicUsize> = (0..count).map(|_| AtomicUsize::new(0)).collect();
                each(n, count, &|i| {
                    hits[i].fetch_add(1, Relaxed);
                });
                assert!(hits.iter().all(|h| h.load(Relaxed) == 1), "{n} threads, {count} indices");
            }
        }
    }
}
