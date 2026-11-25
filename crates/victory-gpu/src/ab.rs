//! A/B within one run: `VICTORY_AB=<KEY>[=<B>][+<KEY>[=<B>]...],<readings>[,<blocks>]`
//! turns one switch (or a group of them, together; the group is refused
//! whole if one of them may not turn) between its value as the run sets it (A) and another
//! (B) in quartets of blocks A B B A, `<readings>` frame-log readings (five
//! seconds each, `crate::framelog`) a block, from the moment the world is
//! steady: three readings in a row of at least 3,000 draws a frame, a
//! median frame under 300 ms and no host pipeline being made, then
//! `VICTORY_AB_AFTER_S` seconds more (0 by default). (In the yard the draws
//! pass 3,000 at 110-160 s while the world still loads, with frames of
//! 300-1,700 ms, and it settles at 161-201 s: runs of c1005e5 at 16:22 and
//! 16:28.) The first reading of each block is left out: the switch turned
//! inside it. The frame drifts within a run (down by 0.2-0.3 ms a second in
//! the yard once settled), and in blocks A B A B that drift goes whole into
//! each B - A; A B B A cancels a linear one: each quartet's difference is
//! the mean of its two B blocks less the mean of its two A blocks.
//!
//! After each quartet, and at the end (`<blocks>` blocks, a multiple of 4,
//! then A for good), one line gives the median frame of A and of B, the
//! difference of each quartet, their median and range, how many quartets
//! agree in sign with the median, the frame's slope over the run (ms a
//! minute, from the readings kept) and within the quartets (the second A
//! block less the first), and the same differences for where a frame goes
//! (preparing submissions, copies, the heap views' pass, `sync_heaps`, the
//! write watches, waiting for the card, and the card's time in all other
//! lists): which part the switch acts on. `NONE` turns nothing (A/A) in
//! the same order: what the method itself finds in the scene.
//!
//! Two rounds of the same exe differ by 4-28 ms a frame (runs P9 and P12),
//! the size of what is measured; one round serving both arms, with the
//! same scene, cache and temperature, is what can tell 5-15 ms apart.
//! After the old fx-main's `crate::ab` (067f795), which named the readings
//! only.
//!
//! A switch takes part only if it may turn between two submissions with
//! nothing to rebuild; the others are refused by name with the reason
//! ([`KEYS`]). A switch that takes part reads [`flag`] or [`number`] over
//! its own setting, each call (an atomic load when the A/B is on another
//! switch or off).

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

/// Switches the A/B knows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(crate) enum Key {
    BackOwnView,
    TextureHashEvery,
    CardQueue,
    GuardRuns,
    WatchViewEarly,
    HeapSweepSlice,
    ViewKnown,
    CopyPlainLeast,
    JoinParts,
    HeapLess,
    HeapLockfree,
    ShadowShared,
    GuardCoalesce,
    GuardRunsStore,
    CopyReadOnly,
    WaitGuarded,
    SkipUnchanged,
    IndirectHoist,
    None,
}

/// Each switch's name and, for one that may not take part, why.
pub(crate) const KEYS: [(Key, &str, Option<&str>); 19] = [
    (Key::BackOwnView, "BACK_OWN_VIEW", None),
    (Key::TextureHashEvery, "TEXTURE_HASH_EVERY", None),
    (Key::CardQueue, "CARD_QUEUE", None),
    (Key::GuardRuns, "GUARD_RUNS", None),
    (Key::WatchViewEarly, "WATCH_VIEW_EARLY", None),
    (Key::HeapSweepSlice, "HEAP_SWEEP_SLICE", None),
    (Key::ViewKnown, "VIEW_KNOWN", None),
    (Key::CopyPlainLeast, "COPY_PLAIN_LEAST", None),
    (Key::JoinParts, "JOIN_PARTS", None),
    (Key::HeapLess, "HEAP_LESS", None),
    (Key::HeapLockfree, "HEAP_LOCKFREE", None),
    (Key::ShadowShared, "SHADOW_SHARED", None),
    (Key::GuardCoalesce, "GUARD_COALESCE", None),
    (
        Key::GuardRunsStore,
        "GUARD_INTERVALS",
        Some(
            "the guards kept as runs of blocks or as single blocks are two stores of the same state; a turn would have              to carry one into the other under the guards' lock",
        ),
    (Key::CopyReadOnly, "COPY_READ_ONLY", None),
    (Key::WaitGuarded, "WAIT_GUARDED", None),
    (
        Key::SkipUnchanged,
        "SKIP_UNCHANGED",
        Some(
            "while off, the ranges each copy holds unchanged are not cleared when shaders may write the copy \
             (`Shadow::gpu_writes`); turned on again, it would trust ranges from before",
        ),
    ),
    (
        Key::IndirectHoist,
        "INDIRECT_HOIST",
        Some(
            "read at each indirect execution recorded and again when the list ends (`list::end_hoist`); a turn \
             between them leaves a list half hoisted",
        ),
    ),
    (Key::None, "NONE", None),
];

/// What `VICTORY_AB` asks for.
#[derive(Debug, PartialEq)]
pub(crate) struct Plan {
    /// The switches turned together, each with B's value for a switch with
    /// a number (`None`: the other usual one).
    keys: Vec<(Key, Option<u64>)>,
    /// Their names, joined with `+`.
    name: String,
    readings: u32,
    /// Blocks in all, a multiple of 4 (`None`: to the end of the run).
    blocks: Option<u32>,
    /// Seconds after the world is steady before the first block
    /// (`VICTORY_AB_AFTER_S`).
    after_s: f64,
    /// `VICTORY_AB_START_S`: the blocks start at the first reading ending
    /// this many seconds after the first frame, steady or not (a scene that
    /// is the same at every opening, as the first cutscene).
    start_s: Option<f64>,
    /// `VICTORY_AB_MIN_DRAWS`: the draws a frame of a steady reading (3,000).
    min_draws: f64,
}

/// `VICTORY_AB`'s plan, or why it is refused.
pub(crate) fn parse(spec: &str) -> Result<Plan, String> {
    let mut parts = spec.split(',').map(str::trim);
    let head = parts.next().unwrap_or("");
    let items: Vec<&str> = head.split('+').map(str::trim).collect();
    let group = items.len() > 1;
    let whole = if group { " (the group is refused whole)" } else { "" };
    let mut keys: Vec<(Key, Option<u64>)> = Vec::new();
    let mut names: Vec<&str> = Vec::new();
    for item in items {
        let (name, b) = match item.split_once('=') {
            Some((name, b)) => {
                (name, Some(b.trim().parse::<u64>().map_err(|_| format!("B's value {b:?} is not a number{whole}"))?))
            }
            None => (item, None),
        };
        let name = name.trim().trim_start_matches("VICTORY_");
        let Some(&(key, name, refused)) = KEYS.iter().find(|k| k.1.eq_ignore_ascii_case(name)) else {
            let known: Vec<&str> = KEYS.iter().filter(|k| k.2.is_none()).map(|k| k.1).collect();
            return Err(format!("{name:?} is not a switch the A/B can turn (it can: {}){whole}", known.join(", ")));
        };
        if let Some(why) = refused {
            return Err(format!("{name} cannot turn within a run: {why}{whole}"));
        }
        if b.is_some() && !matches!(key, Key::TextureHashEvery | Key::CopyPlainLeast) {
            return Err(format!("{name} is on or off: B is the other one, no value{whole}"));
        }
        if keys.iter().any(|k| k.0 == key) {
            return Err(format!("{name} is named twice{whole}"));
        }
        if group && key == Key::None {
            return Err("NONE turns nothing: alone, not in a group".into());
        }
        keys.push((key, b));
        names.push(name);
    }
    let name = names.join("+");
    let readings = match parts.next() {
        None | Some("") => 4,
        Some(n) => n.parse::<u32>().map_err(|_| format!("readings a block {n:?} is not a number"))?,
    };
    if readings < 2 {
        return Err("a block needs two readings at least: the first is left out".into());
    }
    let blocks = match parts.next() {
        None | Some("") => None,
        Some(n) => Some(
            n.parse::<u32>()
                .ok()
                .filter(|&n| n >= 4 && n % 4 == 0)
                .ok_or_else(|| format!("blocks {n:?}: a multiple of 4 (quartets A B B A)"))?,
        ),
    };
    Ok(Plan { keys, name, readings, blocks, after_s: 0.0, start_s: None, min_draws: 3000.0 })
}

/// A number of seconds or draws from the environment: `None` if unset, not
/// a number or below 0.
fn amount(value: Option<&str>) -> Option<f64> {
    value.and_then(|v| v.trim().parse::<f64>().ok()).filter(|s| s.is_finite() && *s >= 0.0)
}

/// `VICTORY_AB_AFTER_S`: seconds, 0 if unset or not a number.
fn after_s(value: Option<&str>) -> f64 {
    amount(value).unwrap_or(0.0)
}

fn plan() -> Option<&'static Plan> {
    static PLAN: OnceLock<Option<Plan>> = OnceLock::new();
    PLAN.get_or_init(|| {
        let spec = std::env::var("VICTORY_AB").ok()?;
        match parse(&spec) {
            Ok(mut plan) => {
                let env = |name: &str| std::env::var(name).ok();
                plan.after_s = after_s(env("VICTORY_AB_AFTER_S").as_deref());
                plan.start_s = amount(env("VICTORY_AB_START_S").as_deref());
                plan.min_draws = amount(env("VICTORY_AB_MIN_DRAWS").as_deref()).unwrap_or(3000.0);
                MASK.store(plan.keys.iter().fold(0, |m, k| m | 1 << k.0 as u32), Relaxed);
                let from = match plan.start_s {
                    Some(at) => format!("from the reading ending {at} s after the first frame (VICTORY_AB_START_S)"),
                    None => format!(
                        "from {} s after the world is steady (3 readings in a row of {} draws or more, a median frame under 300 ms, \
                         no host pipeline being made)",
                        plan.after_s, plan.min_draws
                    ),
                };
                tracing::info!(
                    "A/B within the run (VICTORY_AB={spec}): {} turned in quartets of blocks A B B A, {} readings a block{}, {from}",
                    plan.name,
                    plan.readings,
                    plan.blocks.map_or(String::new(), |n| format!(", {n} blocks"))
                );
                Some(plan)
            }
            Err(why) => {
                tracing::warn!("VICTORY_AB={spec} refused, nothing turns: {why}");
                None
            }
        }
    })
    .as_ref()
}

/// The switches the A/B turns, a bit each ([`Key`] as a number).
static MASK: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Whether the plan was read.
static READ: AtomicBool = AtomicBool::new(false);
/// Whether arm B is on.
static B: AtomicBool = AtomicBool::new(false);

fn turning(key: Key) -> bool {
    if !READ.load(Relaxed) {
        // Read once: the plan sets MASK if it names switches.
        let _ = plan();
        READ.store(true, Relaxed);
    }
    MASK.load(Relaxed) & 1 << key as u32 != 0 && B.load(Relaxed)
}

/// A switch that is on or off: `run` as the run sets it (A), its opposite
/// while the A/B on it is in arm B.
pub(crate) fn flag(key: Key, run: bool) -> bool {
    if turning(key) { !run } else { run }
}

/// A switch with a number: `run` (A), or B's value while the A/B on it is in
/// arm B: the one `VICTORY_AB` gives, else the other usual one (hashing
/// guarded textures every 8 or 64 frames; copies in place from 16 or 256
/// bytes).
pub(crate) fn number(key: Key, run: u64) -> u64 {
    let b = turning(key);
    let given = if b { plan().and_then(|p| p.keys.iter().find(|k| k.0 == key)).and_then(|k| k.1) } else { None };
    arm_value(key, run, b, given)
}

/// A number switch's value in arm A (`run`) or B (`given`, else the other
/// usual one).
fn arm_value(key: Key, run: u64, b: bool, given: Option<u64>) -> u64 {
    if !b {
        return run;
    }
    given.unwrap_or_else(|| other(key, run))
}

fn other(key: Key, run: u64) -> u64 {
    match key {
        Key::TextureHashEvery => if run == 8 { 64 } else { 8 },
        Key::CopyPlainLeast => if run == 256 { 16 } else { 256 },
        _ => run,
    }
}

/// One reading's numbers, ms a frame: the median frame, then the means of
/// where frames went ([`PARTS`]); the draws a frame; when it ended (seconds
/// since the first frame); host pipelines being made then.
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct Reading {
    pub frame: f64,
    pub parts: [f64; 7],
    pub draws: f64,
    pub at: f64,
    pub making: u64,
}

/// The names of [`Reading::parts`].
pub(crate) const PARTS: [&str; 7] =
    ["preparing", "copies", "heap views", "sync_heaps", "watches", "waiting for the card", "all other lists"];

/// The arms of a quartet's blocks: A B B A (true: B).
const ORDER: [bool; 4] = [false, true, true, false];

/// What [`State::step`] goes by.
pub(crate) struct Settings {
    pub readings: u32,
    pub blocks: Option<u32>,
    pub after_s: f64,
    pub start_s: Option<f64>,
    pub min_draws: f64,
}

/// Where the A/B is.
#[derive(Default)]
pub(crate) struct State {
    /// Steady readings in a row, and when the third was.
    steady: u32,
    ready_at: Option<f64>,
    started: bool,
    done: bool,
    /// Blocks closed, readings into the open one, and its kept readings.
    blocks: u32,
    into: u32,
    block: Vec<Reading>,
    /// The open quartet's block medians (`None`: a block that kept no
    /// reading), and every quartet closed whole.
    quartet: Vec<Option<Reading>>,
    quartets: Vec<[Reading; 4]>,
    /// Readings of the blocks: kept; left out as the switch turned in them,
    /// with host pipelines being made, under the draws' floor. Quartets
    /// left out (a block of them kept no reading).
    counts: [u64; 4],
    left_out: u64,
    /// Every reading kept: when, and its frame (for the slope).
    kept: Vec<(f64, f64)>,
}

/// What a reading did to the A/B: its tag (`wait`, `A-`, `A`, `B-`, `B`,
/// `done`), the arm from the next reading on (true: B), and a line for a
/// quartet closed.
pub(crate) struct Step {
    pub tag: &'static str,
    pub b_next: bool,
    pub line: Option<String>,
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn medians(readings: &[Reading]) -> Reading {
    Reading {
        frame: median(readings.iter().map(|r| r.frame).collect()),
        parts: std::array::from_fn(|i| median(readings.iter().map(|r| r.parts[i]).collect())),
        draws: median(readings.iter().map(|r| r.draws).collect()),
        at: median(readings.iter().map(|r| r.at).collect()),
        making: 0,
    }
}

/// The least-squares slope of `points` (seconds, ms), in ms a minute.
fn slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let (mx, my) = (points.iter().map(|p| p.0).sum::<f64>() / n, points.iter().map(|p| p.1).sum::<f64>() / n);
    let sxx: f64 = points.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
    if sxx == 0.0 {
        return 0.0;
    }
    points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>() / sxx * 60.0
}

/// A quartet's B - A of `f`: the mean of its B blocks less that of its A
/// blocks (a drift that is linear in time cancels).
fn quartet_diff(q: &[Reading; 4], f: &dyn Fn(&Reading) -> f64) -> f64 {
    (f(&q[1]) + f(&q[2])) / 2.0 - (f(&q[0]) + f(&q[3])) / 2.0
}

impl State {
    /// One reading.
    pub(crate) fn step(&mut self, s: &Settings, r: Reading, name: &str) -> Step {
        if self.done {
            return Step { tag: "done", b_next: false, line: None };
        }
        if !self.started && let Some(at) = s.start_s {
            self.started = r.at >= at;
            return Step { tag: "wait", b_next: false, line: None };
        }
        if !self.started {
            if r.draws >= s.min_draws && r.frame < 300.0 && r.making == 0 {
                self.steady += 1;
            } else {
                (self.steady, self.ready_at) = (0, None);
            }
            if self.steady >= 3 && self.ready_at.is_none() {
                self.ready_at = Some(r.at);
            }
            self.started = self.ready_at.is_some_and(|at| r.at - at >= s.after_s);
            return Step { tag: "wait", b_next: false, line: None };
        }
        let b = ORDER[self.blocks as usize % 4];
        let first = self.into == 0;
        self.into += 1;
        if first {
            self.counts[1] += 1;
        } else if r.making > 0 {
            // Pipelines being made: the frame waits for them, not the switch.
            self.counts[2] += 1;
        } else if r.draws < s.min_draws {
            // Another scene (a loading screen, a cut).
            self.counts[3] += 1;
        } else {
            self.counts[0] += 1;
            self.block.push(r);
            self.kept.push((r.at, r.frame));
        }
        let tag = match (b, first) {
            (false, true) => "A-",
            (false, false) => "A",
            (true, true) => "B-",
            (true, false) => "B",
        };
        if self.into < s.readings {
            return Step { tag, b_next: b, line: None };
        }
        // The block closes.
        let block = std::mem::take(&mut self.block);
        self.quartet.push((!block.is_empty()).then(|| medians(&block)));
        self.into = 0;
        self.blocks += 1;
        let last = s.blocks.is_some_and(|n| self.blocks >= n);
        let mut line = None;
        if self.quartet.len() == 4 {
            match (self.quartet[0], self.quartet[1], self.quartet[2], self.quartet[3]) {
                (Some(a), Some(b), Some(c), Some(d)) => self.quartets.push([a, b, c, d]),
                _ => self.left_out += 1,
            }
            self.quartet.clear();
            line = Some(self.line(name, last, s.min_draws));
        }
        if last {
            self.done = true;
            return Step { tag, b_next: false, line };
        }
        Step { tag, b_next: ORDER[self.blocks as usize % 4], line }
    }

    /// The line of the quartets so far.
    fn line(&self, name: &str, end: bool, floor: f64) -> String {
        let k = self.quartets.len();
        let readings = format!(
            "readings: {} kept, {} left out as the switch turned in them, {} with host pipelines being made, {} under {floor} \
             draws a frame; {} quartets left out (a block kept no reading)",
            self.counts[0], self.counts[1], self.counts[2], self.counts[3], self.left_out
        );
        if k == 0 {
            return format!("A/B {name}{}: no quartet kept yet; {readings}", if end { " (the end: A from here on)" } else { "" });
        }
        let summary = |d: &[f64]| -> String {
            let m = median(d.to_vec());
            let lo = d.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = d.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let agree = d.iter().filter(|&&x| (x > 0.0) == (m > 0.0) && x != 0.0).count();
            format!("{m:+.1} ({lo:+.1}..{hi:+.1}, {agree} of {} agree in sign)", d.len())
        };
        let diffs = |f: &dyn Fn(&Reading) -> f64| -> Vec<f64> { self.quartets.iter().map(|q| quartet_diff(q, f)).collect() };
        let frames = diffs(&|r| r.frame);
        let each: Vec<String> = frames.iter().map(|d| format!("{d:+.1}")).collect();
        let a = |q: &[Reading; 4]| (q[0].frame + q[3].frame) / 2.0;
        let b = |q: &[Reading; 4]| (q[1].frame + q[2].frame) / 2.0;
        let (qa, qb) = self.quartets.last().map_or((0.0, 0.0), |q| (a(q), b(q)));
        let within: Vec<f64> = self.quartets.iter().map(|q| q[3].frame - q[0].frame).collect();
        let parts: Vec<String> = PARTS.iter().enumerate().map(|(i, p)| format!("{p} {}", summary(&diffs(&|r| r.parts[i])))).collect();
        format!(
            "A/B {name}{}: quartet {k} (blocks A B B A): frame A {qa:.1} B {qb:.1} ms, B - A {:+.1}; all {k} quartets: frame median \
             A {:.1} B {:.1} ms, B - A by quartet {}, median {}; the frame's slope over the run {:+.2} ms a minute, within the \
             quartets (the second A block less the first) {}; B - A of where a frame goes (median, range, sign): {}; {readings}",
            if end { " (the end: A from here on)" } else { "" },
            qb - qa,
            median(self.quartets.iter().map(a).collect()),
            median(self.quartets.iter().map(b).collect()),
            each.join(" "),
            summary(&frames),
            slope(&self.kept),
            summary(&within),
            parts.join("; ")
        )
    }
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// At each frame-log reading: moves the A/B along; the reading's tag for
/// its line (empty with no A/B).
pub(crate) fn reading(r: Reading) -> String {
    let Some(plan) = plan() else {
        return String::new();
    };
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let state = state.get_or_insert_with(State::default);
    let settings = Settings {
        readings: plan.readings,
        blocks: plan.blocks,
        after_s: plan.after_s,
        start_s: plan.start_s,
        min_draws: plan.min_draws,
    };
    let step = state.step(&settings, r, &plan.name);
    if let Some(line) = &step.line {
        tracing::info!("{line}");
    }
    B.store(step.b_next, Relaxed);
    format!("; A/B {} {}", plan.name, step.tag)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plan_names_a_switch_that_may_turn() {
        let p = parse("back_own_view,3").unwrap();
        assert_eq!((p.keys.clone(), p.readings, p.blocks), (vec![(Key::BackOwnView, None)], 3, None));
        let p = parse("VICTORY_TEXTURE_HASH_EVERY=8,4,12").unwrap();
        assert_eq!((p.keys.clone(), p.blocks), (vec![(Key::TextureHashEvery, Some(8))], Some(12)));
        assert_eq!(parse("NONE").unwrap().readings, 4);
        let refused = parse("SKIP_UNCHANGED,4").unwrap_err();
        assert!(refused.contains("cannot turn within a run"), "{refused}");
        assert!(parse("INDIRECT_HOIST").unwrap_err().contains("half hoisted"));
        assert!(parse("RELAY,4").unwrap_err().contains("not a switch"));
        assert!(parse("CARD_QUEUE=3,4").is_err(), "an on/off switch takes no value");
        assert!(parse("CARD_QUEUE,1").is_err(), "a block needs a reading left out and one kept");
        assert!(parse("CARD_QUEUE,4,6").unwrap_err().contains("multiple of 4"));
        assert_eq!(after_s(Some(" 60 ")), 60.0);
        assert_eq!(after_s(Some("x")), 0.0);
        assert_eq!(after_s(None), 0.0);
    }

    /// A group of switches turned together; refused whole if one of them
    /// may not turn, is named twice, or is NONE.
    #[test]
    fn a_group_turns_together_or_not_at_all() {
        let p = parse("HEAP_LESS+heap_lockfree+TEXTURE_HASH_EVERY=8,4,8").unwrap();
        assert_eq!(
            p.keys,
            vec![(Key::HeapLess, None), (Key::HeapLockfree, None), (Key::TextureHashEvery, Some(8))]
        );
        assert_eq!(p.name, "HEAP_LESS+HEAP_LOCKFREE+TEXTURE_HASH_EVERY");
        let refused = parse("HEAP_LESS+SKIP_UNCHANGED,4").unwrap_err();
        assert!(refused.contains("SKIP_UNCHANGED") && refused.contains("refused whole"), "{refused}");
        assert!(parse("HEAP_LESS+HEAP_LESS").unwrap_err().contains("twice"));
        assert!(parse("HEAP_LESS+NONE").unwrap_err().contains("alone"));
    }

    /// The readings of the blocks are kept, or left out as the switch
    /// turned in them, with pipelines being made, or under the draws'
    /// floor; a block that kept none leaves its quartet out.
    #[test]
    fn readings_are_counted_by_why_they_are_left_out() {
        let s = settings(3, Some(8));
        let mut state = State::default();
        let mut lines = Vec::new();
        for i in 0..5 + 24 {
            let mut r = reading(i, 100.0);
            // Second quartet's first block (readings 17 to 19): the first is
            // where the switch turned, the other two are lost.
            if i == 18 {
                r.making = 3;
            }
            if i == 19 {
                r.draws = 10.0;
            }
            lines.extend(state.step(&s, r, "TEST").line);
        }
        assert_eq!(state.counts, [14, 8, 1, 1]);
        assert_eq!((state.quartets.len(), state.left_out), (1, 1));
        let last = lines.last().unwrap();
        assert!(
            last.ends_with("readings: 14 kept, 8 left out as the switch turned in them, 1 with host pipelines being made, 1 under 3000 draws a frame; 1 quartets left out (a block kept no reading)"),
            "{last}"
        );
    }

    #[test]
    fn the_other_value_of_a_number() {
        assert_eq!(other(Key::TextureHashEvery, 64), 8);
        assert_eq!(other(Key::TextureHashEvery, 8), 64);
        assert_eq!(other(Key::CopyPlainLeast, 16), 256);
        assert_eq!(other(Key::CopyPlainLeast, 256), 16);
    }

    /// A reading `i` (five seconds each) of `frame` ms, steady from the
    /// third one on.
    fn reading(i: usize, frame: f64) -> Reading {
        let steady = i >= 2;
        Reading {
            frame,
            parts: [frame / 10.0; 7],
            draws: if steady { 5000.0 } else { 100.0 },
            at: 5.0 * i as f64,
            making: 0,
        }
    }

    /// Runs the A/B over readings made by `ms(b, index)` (b: the arm the
    /// switch is in); returns the tags, the lines and the state.
    fn run(s: &Settings, count: usize, ms: impl Fn(bool, usize) -> f64) -> (Vec<&'static str>, Vec<String>, State) {
        let mut state = State::default();
        let mut b = false;
        let (mut tags, mut lines) = (Vec::new(), Vec::new());
        for i in 0..count {
            let step = state.step(s, reading(i, ms(b, i)), "TEST");
            tags.push(step.tag);
            lines.extend(step.line);
            b = step.b_next;
        }
        (tags, lines, state)
    }

    fn settings(readings: u32, blocks: Option<u32>) -> Settings {
        Settings { readings, blocks, after_s: 0.0, start_s: None, min_draws: 3000.0 }
    }

    #[test]
    fn the_blocks_may_start_at_a_time_or_at_fewer_draws() {
        // At a time: readings end every 5 s; from 20 s on, steady or not.
        let s = Settings { readings: 2, blocks: None, after_s: 0.0, start_s: Some(20.0), min_draws: 3000.0 };
        let mut state = State::default();
        let tags: Vec<&str> = (0..7)
            .map(|i| {
                let mut r = reading(i, 900.0);
                r.draws = 2.0;
                state.step(&s, r, "TEST").tag
            })
            .collect();
        // Reading 4 ends at 20 s: the next one opens the first block.
        assert_eq!(tags, ["wait", "wait", "wait", "wait", "wait", "A-", "A"]);
        // At fewer draws: 1,000 a frame count as steady with a floor of 900.
        let s = Settings { readings: 2, blocks: None, after_s: 0.0, start_s: None, min_draws: 900.0 };
        let mut state = State::default();
        let tags: Vec<&str> = (0..5)
            .map(|i| {
                let mut r = reading(i, 150.0);
                r.draws = 1000.0;
                state.step(&s, r, "TEST").tag
            })
            .collect();
        assert_eq!(tags, ["wait", "wait", "wait", "A-", "A"]);
    }

    #[test]
    fn quartets_run_a_b_b_a_and_leave_out_each_blocks_first_reading() {
        let (tags, lines, _) = run(&settings(3, Some(4)), 20, |_, _| 100.0);
        assert_eq!(
            tags,
            [
                "wait", "wait", "wait", "wait", "wait", "A-", "A", "A", "B-", "B", "B", "B-", "B", "B", "A-", "A", "A", "done",
                "done", "done"
            ]
        );
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with("A/B TEST (the end: A from here on): quartet 1 (blocks A B B A):"), "{}", lines[0]);
    }

    #[test]
    fn the_blocks_wait_for_steady_frames_and_the_delay_after_them() {
        // Readings of 400 ms (still loading) do not count, nor ones with
        // pipelines being made; then three steady ones, then 10 s more.
        let mut state = State::default();
        let s = Settings { readings: 2, blocks: None, after_s: 10.0, start_s: None, min_draws: 3000.0 };
        let mut tags = Vec::new();
        for i in 0..10 {
            let mut r = reading(i, if i < 3 { 400.0 } else { 150.0 });
            if i == 4 {
                r.making = 2;
            }
            tags.push(state.step(&s, r, "TEST").tag);
        }
        // Steady from reading 5 (at 25 s); the third at 7 (35 s); started
        // at 9 (45 s), the next reading being the first block's.
        assert_eq!(tags, ["wait"; 10]);
        assert_eq!(state.step(&s, reading(10, 150.0), "TEST").tag, "A-");
    }

    #[test]
    fn a_linear_drift_cancels_and_a_switch_shows() {
        // The frame falls 0.25 ms a second through the run; B is 10 ms slower.
        let drift = |b: bool, i: usize| 200.0 - 0.25 * 5.0 * i as f64 + if b { 10.0 } else { 0.0 };
        let (_, lines, state) = run(&settings(4, None), 5 + 16 * 3, drift);
        assert_eq!(state.quartets.len(), 3);
        for q in &state.quartets {
            assert!((quartet_diff(q, &|r| r.frame) - 10.0).abs() < 1e-9, "{}", quartet_diff(q, &|r| r.frame));
        }
        assert!(lines.last().unwrap().contains("median +10.0 (+10.0..+10.0, 3 of 3 agree in sign)"), "{}", lines.last().unwrap());
        // The slope over the run is the drift (-15 ms a minute) plus the
        // switch's share, and is said.
        assert!(lines.last().unwrap().contains("the frame's slope over the run -"), "{}", lines.last().unwrap());
    }

    #[test]
    fn the_same_value_in_both_arms_gives_no_difference() {
        // A/A: every quartet's difference is 0, in the frame and in every part.
        let (_, lines, state) = run(&settings(3, None), 5 + 12 * 2, |_, i| 100.0 + (i % 3) as f64);
        assert_eq!(state.quartets.len(), 2);
        for q in &state.quartets {
            assert_eq!(quartet_diff(q, &|r| r.frame), 0.0);
            for i in 0..PARTS.len() {
                assert_eq!(quartet_diff(q, &|r| r.parts[i]), 0.0);
            }
        }
        assert!(lines.last().unwrap().contains("median +0.0 (+0.0..+0.0"), "{}", lines.last().unwrap());
    }

    #[test]
    fn the_reading_the_switch_turned_in_is_left_out() {
        // Each block's first reading is far off; the medians do not see it.
        let (_, _, state) = run(&settings(3, None), 5 + 12, |b, i| {
            let first = i >= 5 && (i - 5) % 3 == 0;
            if first { 1000.0 } else if b { 110.0 } else { 100.0 }
        });
        let q = &state.quartets[0];
        assert_eq!([q[0].frame, q[1].frame, q[2].frame, q[3].frame], [100.0, 110.0, 110.0, 100.0]);
    }

    #[test]
    fn b_given_as_the_runs_own_value_changes_nothing() {
        // VICTORY_AB=TEXTURE_HASH_EVERY=64 on a run with 64: both arms 64.
        for b in [false, true] {
            assert_eq!(arm_value(Key::TextureHashEvery, 64, b, Some(64)), 64);
        }
        assert_eq!(arm_value(Key::TextureHashEvery, 64, true, None), 8);
        assert_eq!(arm_value(Key::TextureHashEvery, 64, false, None), 64);
    }

    #[test]
    fn a_switch_not_turned_keeps_its_value() {
        // No plan in the tests' environment: every switch reads as the run sets it.
        assert!(flag(Key::CardQueue, true));
        assert!(!flag(Key::CardQueue, false));
        assert_eq!(number(Key::TextureHashEvery, 64), 64);
    }

    #[test]
    fn the_slope_is_in_ms_a_minute() {
        assert!((slope(&[(0.0, 100.0), (60.0, 90.0), (120.0, 80.0)]) + 10.0).abs() < 1e-9);
        assert_eq!(slope(&[(0.0, 100.0)]), 0.0);
    }
}
