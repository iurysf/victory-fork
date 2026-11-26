//! Every frame's length, and what the long ones were made of.
//!
//! The frame-rate line averages over a second or more, and its counters
//! are reset at each line: a single long frame is lost in it, and so is
//! what that frame was made of. Run 222's dips (150 ms frames with some of
//! 250 to 500 ms among them) could not be placed from those lines.
//!
//! Here every counter runs from the start and is never reset. At the end
//! of each present call the totals are read, and their differences since
//! the present before are that frame's. Every five seconds one line gives
//! the frame times (each one, the shortest, the median, the longest), one
//! gives the usual frame's make-up (the mean of the frames that were not
//! long), and each long frame gets a line of its own in the same form, so
//! a long frame is read against a usual one column by column.
//!
//! On by default in diagnostics builds, off in others;
//! `VICTORY_FRAME_LOG=1` or `0` says otherwise. The counters themselves
//! are plain atomic adds in every build.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Totals since the start (microseconds unless said otherwise).
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum T {
    /// The title's present call, all of it; of it, running the queues'
    /// lines and settling submissions before our copy; and our copy to the
    /// window with the host's present.
    PresentUs,
    PresentLineUs,
    PresentShowUs,
    /// The title's submit calls (`queue::execute`), on every thread.
    ExecuteUs,
    /// The submit thread of ours (`VICTORY_ASYNC_SUBMIT=1`) at work: wall
    /// time running the queues' lines, and its processor time doing so.
    WorkerUs,
    WorkerUserUs,
    WorkerKernelUs,
    /// Submissions sent; preparing them and waiting for the card, on
    /// whichever thread (the frame-rate line's "preparing" and "on the
    /// GPU").
    Submissions,
    PrepareUs,
    CardWaitUs,
    /// The card with nothing of ours queued, and how much of that while
    /// the submitting thread was working.
    CardIdleUs,
    CardIdleBehindUs,
    /// Copies of title memory for submissions (time, bytes), and what
    /// shaders wrote copied back (time, bytes).
    CopyUs,
    CopyBytes,
    CopyBackUs,
    CopyBackBytes,
    /// List calls that record into host lists, on every thread
    /// (nanoseconds, calls). Diagnostics builds only.
    RecordNs,
    RecordCalls,
    /// The runtime changing what title memory shows: calls, the time from
    /// asking for the lock to letting it go, and of that the wait for
    /// readers.
    Remaps,
    RemapUs,
    RemapWaitUs,
    /// Texture refreshes on every thread, and of them the wait for title
    /// memory to hold still; uploads recorded.
    RefreshUs,
    RefreshStillUs,
    Uploads,
    /// Waiting for host pipelines being made.
    PipelineWaitUs,
    /// Writes taken by the guards on view memory.
    GuardFaults,
    /// The title's frame waits: time slept, on every thread.
    PaceUs,
    /// DirectStorage requests finished, their destination bytes, and the
    /// workers' time in them.
    DiskReads,
    DiskBytes,
    DiskUs,
    /// The card's time by kind of list (`crate::cardmeter`), as read a
    /// second and a half after the lists were recorded: our lists that
    /// bring copies in the card's memory up to date and hash their pieces,
    /// our compare lists, and every other list; and how many of each.
    CardSyncUs,
    CardSyncLists,
    CardCheckUs,
    CardCheckLists,
    CardListsUs,
    CardLists,
    /// Looks at which processor a thread is on ([`on_slow_core`]), and how
    /// many found a slow one (an efficiency core of a hybrid processor):
    /// the thread that presents (one look a present), our submit thread
    /// (one a turn), and the recording threads (one every couple of
    /// milliseconds of list calls).
    RenderLooks,
    RenderSlow,
    WorkerLooks,
    WorkerSlow,
    RecordLooks,
    RecordSlow,
    /// The pass over the heap's buffer views (`draw.rs`), `sync_heaps`,
    /// and the write watches' askings of the kernel and the guards
    /// (diagnostics builds only): for `crate::ab`'s parts of a frame.
    HeapViewsUs,
    SyncHeapsUs,
    WatchUs,
    /// "preparing" by step (`crate::host`'s submit): the submission's one
    /// write-watch asking, its lists' steps (of them the deferred work,
    /// the snapshots of ranges shaders write and the textures' refreshes),
    /// the last flush of refreshes, the snapshots taken after it, and the
    /// plan of copies in the card's memory (its merges among them).
    PrepAskUs,
    PrepStepsUs,
    PrepWorkUs,
    PrepGpuWriteUs,
    PrepTextureUs,
    PrepFlushUs,
    PrepSnapshotsUs,
    PrepCardUs,
    /// The switches' own paths, on every thread: waits for the guards'
    /// lock and the turns of their store (`VICTORY_GUARD_INTERVALS`);
    /// protections asked over runs and refused, and the time of those
    /// calls (`VICTORY_GUARD_COALESCE`); in-place copies without the floor
    /// and the faults among them (`VICTORY_COPY_READ_ONLY`); copies of
    /// title buffers and windows made outside the lock
    /// (`VICTORY_SHADOW_SHARED`).
    GuardsLockUs,
    GuardsStoreUs,
    CoalesceRefused,
    CoalesceRefusedUs,
    InPlaceUs,
    InPlaceFaults,
    SharedBuildUs,
    /// The submitting thread in `VICTORY_UNDER_WIDE`'s paths (merges and
    /// patches on threads of their own).
    UnderWideUs,
    /// Title bytes under flights' writes, in both arms of an A/B on
    /// `VICTORY_UNDER_WIDE`: the merges of copies in the card's memory;
    /// of the wide ones, the compare on the pool (wall time) and the put-in
    /// in order; the patches; the pool's jobs and the caller's wait for its
    /// threads after its own share; copies given patches, those of 4,096
    /// runs or more, and their runs.
    MergeCardUs,
    MergeCompareUs,
    MergePutUs,
    PatchesUs,
    UnderJobs,
    UnderWaitUs,
    PatchCopies,
    PatchCopiesBig,
    PatchRuns,
    Count,
}

const N: usize = T::Count as usize;
static TOTAL: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];

/// Adds to a total.
pub fn add(t: T, n: u64) {
    TOTAL[t as usize].fetch_add(n, Relaxed);
}

/// Takes from a total (a time that was summed over threads is corrected
/// to the time it took). Not below 0: the frame line may have taken the
/// total meanwhile, and a wrapped one read as 584,000 years.
pub fn sub(t: T, n: u64) {
    let _ = TOTAL[t as usize].fetch_update(Relaxed, Relaxed, |was| Some(was.saturating_sub(n)));
}

/// What a thread did inside our calls since its last present: for the
/// thread that presents (the title's render thread).
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum H {
    /// List calls, nanoseconds.
    RecordNs,
    /// Submit calls, microseconds.
    ExecuteUs,
    /// Frame waits, microseconds slept.
    PaceUs,
    Count,
}

const HN: usize = H::Count as usize;

thread_local! {
    static HERE: std::cell::Cell<[u64; HN]> = const { std::cell::Cell::new([0; HN]) };
}

/// Adds to this thread's own count.
pub fn here(h: H, n: u64) {
    HERE.with(|c| {
        let mut v = c.get();
        v[h as usize] += n;
        c.set(v);
    });
}

/// Whether the processor this thread runs on right now is a slow one: on a
/// processor with two kinds of cores (the console has eight alike; an Intel
/// hybrid has fast cores and efficiency cores at about half their speed),
/// one of a lower efficiency class than the best. Always false where all
/// cores are alike.
pub fn on_slow_core() -> bool {
    use windows_sys::Win32::System::Threading::GetCurrentProcessorNumber;
    /// By logical processor (of group 0): whether it is slow.
    static SLOW: std::sync::OnceLock<Vec<bool>> = std::sync::OnceLock::new();
    let slow = SLOW.get_or_init(|| {
        let sets = cpu_sets();
        let best = sets.iter().map(|c| c.2).max().unwrap_or(0);
        let mut slow = vec![false; sets.iter().map(|c| c.1 as usize + 1).max().unwrap_or(0)];
        for (_, processor, class) in &sets {
            slow[*processor as usize] = *class < best;
        }
        tracing::info!(
            "processors: {} logical, {} of them slow ones (a lower efficiency class than the best); the frame log counts how \
             often the render, submit and recording threads are found on a slow one",
            slow.len(),
            slow.iter().filter(|&&s| s).count()
        );
        slow
    });
    // SAFETY: a plain query.
    let processor = unsafe { GetCurrentProcessorNumber() } as usize;
    slow.get(processor).copied().unwrap_or(false)
}

/// The system's processor sets of group 0: each one's id, its logical
/// processor and its efficiency class (higher is faster).
fn cpu_sets() -> Vec<(u32, u8, u8)> {
    use windows_sys::Win32::System::SystemInformation::{GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut bytes = 0u32;
    // SAFETY: a size query, then the call into a buffer of that size.
    unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut bytes, GetCurrentProcess(), 0) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as above; the buffer holds `bytes` bytes, aligned for the records.
    let ok = unsafe { GetSystemCpuSetInformation(buffer.as_mut_ptr().cast(), bytes, &mut bytes, GetCurrentProcess(), 0) };
    let mut sets = Vec::new();
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_CPU_SET_INFORMATION>()) };
        if info.Size == 0 {
            break;
        }
        // SAFETY: the record's type is CpuSet (the only one there is).
        let set = unsafe { &info.Anonymous.CpuSet };
        if set.Group == 0 {
            sets.push((set.Id, set.LogicalProcessorIndex, set.EfficiencyClass));
        }
        at += info.Size as usize;
    }
    sets
}

/// The slow logical processors of group 0 as a mask (a lower efficiency
/// class than the best): 0 where all cores are alike.
pub fn slow_processors() -> usize {
    let sets = cpu_sets();
    let best = sets.iter().map(|c| c.2).max().unwrap_or(0);
    sets.iter().filter(|c| c.2 < best && c.1 < usize::BITS as u8).fold(0, |mask, c| mask | 1usize << c.1)
}

/// Asks the system to run the calling thread on the fast cores only, where
/// the processor has two kinds (`VICTORY_FAST_CORES=0`: not asked).
///
/// The title keeps its own threads on fast cores (run 229: its render
/// thread in 3,484 looks of 3,484, its recording threads in 104,254 of
/// 104,254), as on the console, whose eight cores are alike. The layer's
/// submit thread was found on a slow one in 1,529 looks of 24,021, and the
/// frame waits for that thread. A set of processors selected for a thread
/// is a preference the scheduler keeps unless nothing of it is free.
pub fn keep_on_fast_cores(what: &str) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadSelectedCpuSets};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_FAST_CORES").is_none_or(|v| v != "0")) {
        return;
    }
    let sets = cpu_sets();
    let best = sets.iter().map(|c| c.2).max().unwrap_or(0);
    let fast: Vec<u32> = sets.iter().filter(|c| c.2 == best).map(|c| c.0).collect();
    // All alike (or nothing known): nothing to ask for.
    if fast.is_empty() || fast.len() == sets.len() {
        return;
    }
    // SAFETY: our own thread and a list of the system's own set ids.
    let ok = unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), fast.as_ptr(), fast.len() as u32) };
    if ok == 0 {
        tracing::warn!(what, error = %std::io::Error::last_os_error(), "the thread could not be kept on the fast cores");
    } else {
        tracing::info!(what, fast = fast.len(), of = sets.len(), "the thread is kept on the fast cores (VICTORY_FAST_CORES=0: not)");
    }
}

/// Asks the system to run the calling thread on the slow cores only, where
/// the processor has two kinds and at least eight slow logical processors
/// (a helper that must not take a fast core from the title's threads).
/// True if it was asked.
///
/// The title pins its threads as on the console, where disk decompression
/// is the hardware's: measured on a 14700KF, 31 of its threads are tied to
/// logical processors 0-13, the fast cores. The layer decompresses on as
/// many worker threads as there are logical processors (28), free to run
/// anywhere: in the opening cutscene's streaming stretch about ten of them
/// were busy while the title's recording calls took three times as long.
pub fn keep_on_slow_cores(what: &str) -> bool {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadSelectedCpuSets};
    let sets = cpu_sets();
    let best = sets.iter().map(|c| c.2).max().unwrap_or(0);
    let slow: Vec<u32> = sets.iter().filter(|c| c.2 < best).map(|c| c.0).collect();
    if slow.len() < 8 {
        return false;
    }
    // SAFETY: our own thread and a list of the system's own set ids.
    let ok = unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), slow.as_ptr(), slow.len() as u32) } != 0;
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!(what, slow = slow.len(), of = sets.len(), ok, "helper threads are kept on the slow cores: the first one");
    });
    ok
}

/// Whether frames are logged.
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var_os("VICTORY_FRAME_LOG") {
        Some(v) => v != "0",
        None => cfg!(feature = "diagnostics"),
    })
}

fn micros(t: windows_sys::Win32::Foundation::FILETIME) -> u64 {
    (u64::from(t.dwHighDateTime) << 32 | u64::from(t.dwLowDateTime)) / 10
}

/// This thread's processor time so far (user, kernel), in microseconds.
pub fn thread_times() -> (u64, u64) {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: our own thread and four valid out pointers.
    unsafe { GetThreadTimes(GetCurrentThread(), &mut created, &mut exited, &mut kernel, &mut user) };
    (micros(user), micros(kernel))
}

/// The process's processor time so far (user, kernel), in microseconds.
fn process_times() -> (u64, u64) {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: our own process and four valid out pointers.
    unsafe { GetProcessTimes(GetCurrentProcess(), &mut created, &mut exited, &mut kernel, &mut user) };
    (micros(user), micros(kernel))
}

/// The process's page faults so far (soft and hard).
fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: plain data the call fills in.
    let mut c: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    c.cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// The PC's free physical memory, in MiB.
fn free_mb() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    unsafe { GlobalMemoryStatusEx(&mut m) };
    m.ullAvailPhys >> 20
}

/// Values read when a frame ends, all running totals but the last.
const S: usize = 11;
const S_THREAD_USER: usize = 0;
const S_THREAD_KERNEL: usize = 1;
const S_PROCESS_USER: usize = 2;
const S_PROCESS_KERNEL: usize = 3;
const S_PAGE_FAULTS: usize = 4;
const S_DRAWS: usize = 5;
const S_PIPELINES_MADE: usize = 6;
const S_PIPELINES_LOADED: usize = 7;
const S_STAGING_REUSED: usize = 8;
const S_STAGING_MADE: usize = 9;
/// Not a total: the free memory then.
const S_FREE_MB: usize = 10;

fn sample() -> [u64; S] {
    let (tu, tk) = thread_times();
    let (pu, pk) = process_times();
    [
        tu,
        tk,
        pu,
        pk,
        page_faults(),
        crate::draw::drawn(),
        crate::pipeline::frame_log_counts().0,
        crate::pipeline::frame_log_counts().1,
        crate::texture::SPARE_STATS[0].load(Relaxed),
        crate::texture::SPARE_STATS[1].load(Relaxed),
        free_mb(),
    ]
}

/// One frame: its number, when it ended (seconds since the first), its
/// length in microseconds, and the differences of the totals, the sampled
/// values and the presenting thread's own counts over it.
#[derive(Clone)]
struct Frame {
    number: u64,
    at: f64,
    us: f64,
    t: [f64; N],
    s: [f64; S],
    h: [f64; HN],
}

struct State {
    first: std::time::Instant,
    /// The last frame's end: when, the totals and samples then, and the
    /// thread that presented.
    last: Option<(std::time::Instant, [u64; N], [u64; S], u32)>,
    number: u64,
    frames: Vec<Frame>,
    since: std::time::Instant,
    /// The lengths of the last frames (microseconds), for the usual one.
    recent: std::collections::VecDeque<f64>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = values.collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// A frame's make-up as text (`f` may be a mean of frames).
fn text(f: &Frame) -> String {
    let t = |x: T| f.t[x as usize];
    let ms = |x: T| f.t[x as usize] / 1e3;
    let s = |i: usize| f.s[i];
    let h = |x: H| f.h[x as usize];
    format!(
        "{:.0} ms | render thread: ran {:.0} ms (user {:.0}, kernel {:.0}); in the present call {:.1} (lines and settling \
         {:.1}, copy and show {:.1}), in submit calls {:.1}, in list calls {:.1}, paced {:.1} | submit thread: at work \
         {:.1} ms (user {:.1}, kernel {:.1}); {:.0} submissions, preparing {:.1}, waiting for the card {:.1}; copies {:.1} \
         ms {:.1} MB; copy-back {:.1} ms {:.2} MB | card: idle {:.1} ms ({:.1} behind us); by its own clock, of the lists of a moment ago: our \
         send-and-hash lists {:.1} ms in {:.0}, our compare lists {:.1} ms in {:.0}, all other lists {:.1} ms in {:.0} | \
         recording, all threads: {:.1} ms in {:.0} calls; all submit calls {:.1} ms | textures: refresh {:.1} ms ({:.1} waiting for memory to hold \
         still), {:.0} uploads, staging {:.0} made {:.0} reused | remapping: {:.0} calls {:.1} ms ({:.1} waiting for \
         readers) | pipelines: {:.0} made, {:.0} loaded, {:.1} ms waited | disk: {:.0} reads {:.1} MB {:.1} worker ms | \
         faults: {:.0} page, {:.0} guard | process: {:.0} user {:.0} kernel ms | pacing, all threads {:.1} ms | {:.0} \
         draws | {:.0} MB free | found on a slow core: the render thread {:.0} of {:.0} looks, the submit thread {:.0} of {:.0}, recording threads {:.0} of {:.0}",
        f.us / 1e3,
        (s(S_THREAD_USER) + s(S_THREAD_KERNEL)) / 1e3,
        s(S_THREAD_USER) / 1e3,
        s(S_THREAD_KERNEL) / 1e3,
        ms(T::PresentUs),
        ms(T::PresentLineUs),
        ms(T::PresentShowUs),
        h(H::ExecuteUs) / 1e3,
        h(H::RecordNs) / 1e6,
        h(H::PaceUs) / 1e3,
        ms(T::WorkerUs),
        ms(T::WorkerUserUs),
        ms(T::WorkerKernelUs),
        t(T::Submissions),
        ms(T::PrepareUs),
        ms(T::CardWaitUs),
        ms(T::CopyUs),
        t(T::CopyBytes) / 1e6,
        ms(T::CopyBackUs),
        t(T::CopyBackBytes) / 1e6,
        ms(T::CardIdleUs),
        ms(T::CardIdleBehindUs),
        ms(T::CardSyncUs),
        t(T::CardSyncLists),
        ms(T::CardCheckUs),
        t(T::CardCheckLists),
        ms(T::CardListsUs),
        t(T::CardLists),
        t(T::RecordNs) / 1e6,
        t(T::RecordCalls),
        ms(T::ExecuteUs),
        ms(T::RefreshUs),
        ms(T::RefreshStillUs),
        t(T::Uploads),
        s(S_STAGING_MADE),
        s(S_STAGING_REUSED),
        t(T::Remaps),
        ms(T::RemapUs),
        ms(T::RemapWaitUs),
        s(S_PIPELINES_MADE),
        s(S_PIPELINES_LOADED),
        ms(T::PipelineWaitUs),
        t(T::DiskReads),
        t(T::DiskBytes) / 1e6,
        ms(T::DiskUs),
        s(S_PAGE_FAULTS),
        t(T::GuardFaults),
        s(S_PROCESS_USER) / 1e3,
        s(S_PROCESS_KERNEL) / 1e3,
        ms(T::PaceUs),
        s(S_DRAWS),
        s(S_FREE_MB),
        t(T::RenderSlow),
        t(T::RenderLooks),
        t(T::WorkerSlow),
        t(T::WorkerLooks),
        t(T::RecordSlow),
        t(T::RecordLooks),
    )
}

/// The mean of `frames` (not empty).
fn mean(frames: &[&Frame]) -> Frame {
    let n = frames.len() as f64;
    let mut out = Frame { number: 0, at: 0.0, us: 0.0, t: [0.0; N], s: [0.0; S], h: [0.0; HN] };
    for f in frames {
        out.us += f.us / n;
        for i in 0..N {
            out.t[i] += f.t[i] / n;
        }
        for i in 0..S {
            out.s[i] += f.s[i] / n;
        }
        for i in 0..HN {
            out.h[i] += f.h[i] / n;
        }
    }
    out
}

/// Over how many microseconds a frame is long, the usual one being
/// `usual`: half as long again, and at least 40 ms more.
fn long_over(usual: f64) -> f64 {
    (usual * 1.5).max(usual + 40_000.0)
}

/// The lines of a reading: the frame times, the usual frame, the long ones.
fn lines(frames: &[Frame], usual: f64, seconds: f64) -> Vec<String> {
    let over = long_over(usual);
    let (long, plain): (Vec<&Frame>, Vec<&Frame>) = frames.iter().partition(|f| f.us > over);
    let each: Vec<String> = frames.iter().map(|f| format!("{:.0}", f.us / 1e3)).collect();
    let shortest = frames.iter().map(|f| f.us).fold(f64::INFINITY, f64::min);
    let longest = frames.iter().map(|f| f.us).fold(0.0, f64::max);
    let mut out = vec![format!(
        "frame times: {} frames in {seconds:.1} s: shortest {:.0} ms, median {:.0}, longest {:.0}; {} long (over {:.0} ms, \
         the usual frame of the last {RECENT} being {:.0}); each: {}",
        frames.len(),
        shortest / 1e3,
        median(frames.iter().map(|f| f.us)) / 1e3,
        longest / 1e3,
        long.len(),
        over / 1e3,
        usual / 1e3,
        each.join(" ")
    )];
    if !plain.is_empty() {
        out.push(format!("usual frame (the mean of {} not long): {}", plain.len(), text(&mean(&plain))));
    }
    for f in long.iter().take(LONG_MOST) {
        out.push(format!("long frame {} at {:.1} s: {}", f.number, f.at, text(f)));
    }
    if long.len() > LONG_MOST {
        out.push(format!("long frames: {} more in this reading are not listed", long.len() - LONG_MOST));
    }
    out
}

/// Every `VICTORY_FRAME_MARK`th frame (off unless set; 1 to 10,000): one line
/// with the frame's number, when it ended, its length and where it went, so
/// that two runs of a scene that is the same at every opening (the first
/// cutscene) are set side by side frame by frame, by the same shot, not by
/// readings of five seconds that see different shots.
fn mark_every() -> Option<u64> {
    static EVERY: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        std::env::var("VICTORY_FRAME_MARK").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&n| n >= 1).map(|n| n.min(10_000))
    })
}

/// The mark line of frame `f`, if it is one to mark.
fn mark(f: &Frame) -> Option<String> {
    let every = mark_every()?;
    f.number.is_multiple_of(every).then(|| mark_line(f))
}

fn mark_line(f: &Frame) -> String {
    let ms = |t: T| f.t[t as usize] / 1e3;
    format!(
        "frame mark {} at {:.2} s: {:.1} ms; preparing {:.1}, copies {:.1}, heap views {:.1}, sync_heaps {:.1}, watches {:.1}, \
         waiting for the card {:.1}, all other lists {:.1} (the card's clock, lists of a moment ago); {:.0} submissions, {:.0} draws",
        f.number,
        f.at,
        f.us / 1e3,
        ms(T::PrepareUs),
        ms(T::CopyUs),
        ms(T::HeapViewsUs),
        ms(T::SyncHeapsUs),
        ms(T::WatchUs),
        ms(T::CardWaitUs),
        ms(T::CardListsUs),
        f.t[T::Submissions as usize],
        f.s[S_DRAWS]
    )
}

/// The line of where "preparing" goes and of the switches' own paths, for
/// the mean frame `f` of a reading.
fn preparing_line(f: &Frame) -> String {
    let ms = |t: T| f.t[t as usize] / 1e3;
    let listed = ms(T::PrepAskUs) + ms(T::PrepStepsUs) + ms(T::PrepFlushUs) + ms(T::PrepSnapshotsUs) + ms(T::PrepCardUs);
    format!(
        "preparing by step (the mean frame): {:.1} ms; the submission's one asking {:.1}, the lists' steps {:.1} (deferred \
         work {:.1}, snapshots of ranges shaders write {:.1}, textures' refreshes {:.1}, copies and the rest {:.1}), the last \
         flush {:.1}, snapshots {:.1}, the card's plan {:.1}, not in a step {:.1} | the switches' own paths, all threads: \
         waits for the guards' lock {:.2} ms, their store turned {:.2} ms (GUARD_INTERVALS); {:.0} protections over runs \
         refused, {:.2} ms in them (GUARD_COALESCE); in-place copies {:.2} ms, {:.1} faults (COPY_READ_ONLY); copies made \
         outside the lock {:.2} ms (SHADOW_SHARED); merges and patches on threads of their own {:.2} ms (UNDER_WIDE={})",
        ms(T::PrepareUs),
        ms(T::PrepAskUs),
        ms(T::PrepStepsUs),
        ms(T::PrepWorkUs),
        ms(T::PrepGpuWriteUs),
        ms(T::PrepTextureUs),
        ms(T::PrepStepsUs) - ms(T::PrepWorkUs) - ms(T::PrepGpuWriteUs) - ms(T::PrepTextureUs),
        ms(T::PrepFlushUs),
        ms(T::PrepSnapshotsUs),
        ms(T::PrepCardUs),
        ms(T::PrepareUs) - listed,
        ms(T::GuardsLockUs),
        ms(T::GuardsStoreUs),
        f.t[T::CoalesceRefused as usize],
        ms(T::CoalesceRefusedUs),
        ms(T::InPlaceUs),
        f.t[T::InPlaceFaults as usize],
        ms(T::SharedBuildUs),
        ms(T::UnderWideUs),
        crate::underwide::threads(),
    )
}

/// The line of title bytes under flights' writes for the mean frame `f`
/// (`crate::underwide`), the same in both arms of an A/B on it.
fn under_line(f: &Frame) -> String {
    let ms = |t: T| f.t[t as usize] / 1e3;
    let n = |t: T| f.t[t as usize];
    format!(
        "under flights' writes, the mean frame: merges of copies in the card's memory {:.2} ms (on the pool: compare {:.2} \
         wall, put in in order {:.2}), patches {:.2} ms; the pool: {:.0} jobs, {:.2} ms waiting for its threads after the \
         caller's share; patches for {:.0} copies, {:.0} of them with 4,096 runs or more, {:.0} runs (UNDER_WIDE={})",
        ms(T::MergeCardUs),
        ms(T::MergeCompareUs),
        ms(T::MergePutUs),
        ms(T::PatchesUs),
        n(T::UnderJobs),
        ms(T::UnderWaitUs),
        n(T::PatchCopies),
        n(T::PatchCopiesBig),
        n(T::PatchRuns),
        crate::underwide::threads(),
    )
}

/// Frames whose lengths give the usual one, and the long frames listed in
/// a reading at most.
const RECENT: usize = 64;
const LONG_MOST: usize = 8;

/// A frame ends: called at the end of the title's present call, on the
/// thread that made it.
pub fn frame() {
    if !on() {
        return;
    }
    crate::cardmeter::collect();
    add(T::RenderLooks, 1);
    if on_slow_core() {
        add(T::RenderSlow, 1);
    }
    let now = std::time::Instant::now();
    let totals: [u64; N] = std::array::from_fn(|i| TOTAL[i].load(Relaxed));
    let sampled = sample();
    let here = HERE.with(|c| c.replace([0; HN]));
    // SAFETY: a plain query.
    let thread = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let state = state.get_or_insert_with(|| State {
        first: now,
        last: None,
        number: 0,
        frames: Vec::new(),
        since: now,
        recent: std::collections::VecDeque::new(),
    });
    state.number += 1;
    if let Some((then, t0, s0, thread0)) = state.last.replace((now, totals, sampled, thread)) {
        let mut f = Frame {
            number: state.number,
            at: (now - state.first).as_secs_f64(),
            us: (now - then).as_micros() as f64,
            t: std::array::from_fn(|i| totals[i].wrapping_sub(t0[i]) as i64 as f64),
            s: std::array::from_fn(|i| sampled[i].wrapping_sub(s0[i]) as i64 as f64),
            h: here.map(|v| v as f64),
        };
        f.s[S_FREE_MB] = sampled[S_FREE_MB] as f64;
        if thread != thread0 {
            // Another thread presented: its processor time is not this frame's.
            (f.s[S_THREAD_USER], f.s[S_THREAD_KERNEL]) = (0.0, 0.0);
        }
        if let Some(line) = mark(&f) {
            tracing::info!("{line}");
        }
        // The A/B by frames (`VICTORY_AB_FRAMES`): this frame's numbers.
        if crate::ab::by_frames() {
            let usual = median(state.recent.iter().copied());
            let ms = |t: T| f.t[t as usize] / 1e3;
            crate::ab::frame(crate::ab::Reading {
                frame: f.us / 1e3,
                parts: [
                    ms(T::PrepareUs),
                    ms(T::CopyUs),
                    ms(T::HeapViewsUs),
                    ms(T::SyncHeapsUs),
                    ms(T::WatchUs),
                    ms(T::CardWaitUs),
                    ms(T::CardListsUs),
                ],
                draws: f.s[S_DRAWS],
                at: f.at,
                making: crate::pipeline::making(),
                long: usual > 0.0 && f.us > usual * 1.5,
                remap: ms(T::RemapUs),
            });
        }
        state.frames.push(f);
    }
    let elapsed = state.since.elapsed();
    if elapsed < std::time::Duration::from_secs(5) || state.frames.is_empty() {
        return;
    }
    let frames = std::mem::take(&mut state.frames);
    state.since = now;
    // The usual frame: the median of the last frames, this reading's with
    // those before it.
    for f in &frames {
        if state.recent.len() == RECENT {
            state.recent.pop_front();
        }
        state.recent.push_back(f.us);
    }
    let usual = median(state.recent.iter().copied());
    // The A/B within the run (`crate::ab`): this reading's numbers, and
    // its tag at the end of the frame times line.
    let ab = {
        let all: Vec<&Frame> = frames.iter().collect();
        let m = mean(&all);
        let ms = |t: T| m.t[t as usize] / 1e3;
        crate::ab::reading(crate::ab::Reading {
            frame: median(frames.iter().map(|f| f.us)) / 1e3,
            parts: [
                ms(T::PrepareUs),
                ms(T::CopyUs),
                ms(T::HeapViewsUs),
                ms(T::SyncHeapsUs),
                ms(T::WatchUs),
                ms(T::CardWaitUs),
                ms(T::CardListsUs),
            ],
            draws: m.s[S_DRAWS],
            at: frames.last().map_or(0.0, |f| f.at),
            making: crate::pipeline::making(),
            long: false,
            remap: ms(T::RemapUs),
        })
    };
    tracing::info!("{}", crate::background::line(elapsed.as_secs_f64()));
    tracing::info!("{}", preparing_line(&mean(&frames.iter().collect::<Vec<_>>())));
    tracing::info!("{}", crate::lockmeter::line(frames.len() as f64));
    if let Some(line) = crate::lockmeter::journal_line(frames.len() as f64) {
        tracing::info!("{line}");
    }
    tracing::info!("{}", under_line(&mean(&frames.iter().collect::<Vec<_>>())));
    if let Some(line) = crate::shadow::under_by_copy_line(frames.len() as f64) {
        tracing::info!("{line}");
    }
    tracing::info!("{}", crate::watch::askings_line(frames.len() as f64));
    if let Some(line) = crate::shadow::echo_line(frames.len() as f64) {
        tracing::info!("{line}");
    }
    for (i, line) in lines(&frames, usual, elapsed.as_secs_f64()).into_iter().enumerate() {
        if i == 0 {
            tracing::info!("{line}{ab}");
        } else {
            tracing::info!("{line}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(number: u64, ms: f64) -> Frame {
        let mut f = Frame { number, at: number as f64, us: ms * 1e3, t: [0.0; N], s: [0.0; S], h: [0.0; HN] };
        f.t[T::PresentUs as usize] = ms * 100.0;
        f.t[T::Uploads as usize] = ms;
        f
    }

    #[test]
    fn a_long_frame_is_listed_beside_the_usual_one() {
        let frames: Vec<Frame> = [150.0, 160.0, 140.0, 400.0, 150.0].iter().enumerate().map(|(i, &ms)| frame(i as u64, ms)).collect();
        let out = lines(&frames, 150_000.0, 1.0);
        assert_eq!(out.len(), 3, "{out:#?}");
        assert!(out[0].starts_with("frame times: 5 frames in 1.0 s: shortest 140 ms, median 150, longest 400; 1 long (over 225 ms"), "{}", out[0]);
        assert!(out[0].ends_with("each: 150 160 140 400 150"), "{}", out[0]);
        // The usual frame is the mean of the four that are not long.
        assert!(out[1].starts_with("usual frame (the mean of 4 not long): 150 ms |"), "{}", out[1]);
        assert!(out[1].contains("in the present call 15.0 "), "{}", out[1]);
        assert!(out[1].contains(" 150 uploads"), "{}", out[1]);
        assert!(out[2].starts_with("long frame 3 at 3.0 s: 400 ms |"), "{}", out[2]);
        assert!(out[2].contains("in the present call 40.0 "), "{}", out[2]);
        assert!(out[2].contains(" 400 uploads"), "{}", out[2]);
    }

    #[test]
    fn a_frame_mark_names_the_frame_and_where_it_went() {
        let mut f = frame(120, 150.0);
        f.t[T::PrepareUs as usize] = 40_000.0;
        f.t[T::HeapViewsUs as usize] = 12_500.0;
        f.s[S_DRAWS] = 2900.0;
        let line = mark_line(&f);
        assert!(line.starts_with("frame mark 120 at 120.00 s: 150.0 ms; preparing 40.0, copies 0.0, heap views 12.5,"), "{line}");
        assert!(line.ends_with("0 submissions, 2900 draws"), "{line}");
    }

    #[test]
    fn long_is_half_as_long_again_and_at_least_40_ms_more() {
        assert_eq!(long_over(150_000.0), 225_000.0);
        assert_eq!(long_over(30_000.0), 70_000.0);
    }

    #[test]
    fn more_long_frames_than_are_listed_are_counted() {
        let mut frames: Vec<Frame> = (0..12).map(|i| frame(i, 500.0)).collect();
        frames.push(frame(12, 100.0));
        let out = lines(&frames, 100_000.0, 5.0);
        assert_eq!(out.iter().filter(|l| l.starts_with("long frame ")).count(), LONG_MOST);
        assert!(out.last().unwrap().starts_with("long frames: 4 more"), "{out:#?}");
    }

    #[test]
    fn totals_and_a_threads_own_counts_add_up() {
        let before = TOTAL[T::DiskReads as usize].load(Relaxed);
        add(T::DiskReads, 3);
        sub(T::DiskReads, 1);
        assert_eq!(TOTAL[T::DiskReads as usize].load(Relaxed) - before, 2);
        here(H::PaceUs, 5);
        here(H::PaceUs, 6);
        assert_eq!(HERE.with(|c| c.get())[H::PaceUs as usize], 11);
    }
}
