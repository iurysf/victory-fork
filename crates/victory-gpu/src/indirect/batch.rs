//! Clamps of many executions in one pass (`VICTORY_INDIRECT_BATCH=1`, off
//! by default): the arguments of every indirect execution of one record
//! between two points where they may change are bounded together, at the
//! first of them.
//!
//! After avatarmanz/VIctory-7900XTX 0375962 ("Argument clamps in batches",
//! its `crates/victory-gpu/src/indirect/batch.rs`; the author's replayed
//! frame: 2,355 clamps in 980 passes, 86.1 -> 77.7 ms). Adapted to this
//! tree: the arithmetic is the clamp of one record's here ([`super::SOURCE`],
//! not clamped ahead), and the pass is one group that walks its jobs in
//! order, 64 at a time, so the budget is charged in the order the
//! executions were recorded, as one clamp after another charges it (the
//! fork's pass ran several groups in no defined order). Not taken: its
//! `VICTORY_INDIRECT_BATCH=2` (a dispatch not ending the batch), its wide
//! dispatches and its `REST` switch.
//!
//! The executions that come after the first are not known yet when the
//! pass is recorded, so it reads what to clamp from a table of jobs in
//! upload memory, which the CPU goes on filling while the list is recorded
//! (the list runs after it is closed). An execution that asks for a record
//! already in the table reads the same clamped one, and the job charges the
//! budget once for each execution that reads it.
//!
//! Where it differs from the clamp at each execution: only when a list's
//! budget runs out. There, a record read by several executions fits for
//! all of them or for none, and its charges come at the job's place (the
//! first execution that read it), not at each later one.

use windows::Win32::Graphics::Direct3D12::*;

use super::{Clamp, uav_barrier};
use crate::host::transition;

/// The jobs of one pass.
pub(crate) const JOBS: u32 = 256;
/// The bytes of its table in upload memory: the count, then the jobs.
pub(crate) const TABLE: u64 = 16 + 16 * JOBS as u64;
/// The bytes of a clamped record's place in the scratch buffer.
pub(crate) const SLOT: u32 = 32;
/// The source buffers a pass can read.
pub(crate) const SOURCES: usize = 4;
/// The most executions charged by one job: the most a record costs (2^26)
/// as often still fits 32 bits.
pub(crate) const TIMES: u32 = 31;

pub(crate) const SOURCE: &str = r#"
#define RS "SRV(t0), SRV(t1), SRV(t2), SRV(t3), SRV(t4), UAV(u0), UAV(u1)"
ByteAddressBuffer src0 : register(t0);
ByteAddressBuffer src1 : register(t1);
ByteAddressBuffer src2 : register(t2);
ByteAddressBuffer src3 : register(t3);
// The count, 12 bytes of nothing, then per job: source (bits 0-1), kind
// (4-7), words (8-11) and the executions that read it (16-31); the
// record's offset in the source; its place in the scratch buffer; nothing.
ByteAddressBuffer jobs : register(t4);
RWByteAddressBuffer dst_args : register(u0);
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
static const uint MOST_DRAW = 1u << 23;
static const uint GROUPS = 1u << 20;
groupshared uint costs[64];
uint load(uint source, uint at) {
    if (source == 0) return src0.Load(at);
    if (source == 1) return src1.Load(at);
    if (source == 2) return src2.Load(at);
    return src3.Load(at);
}
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint count = jobs.Load(0);
    // Read by lane 0 only, as the clamp of one reads it before charging.
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget.Load(0), BUDGET);
    }
    for (uint first = 0; first < count; first += 64) {
        uint i = first + lane;
        bool active = i < count;
        uint4 job = uint4(0, 0, 0, 0);
        uint record[5] = {0, 0, 0, 0, 0};
        uint cost = 0;
        uint kind = 0;
        uint words = 0;
        // As the clamp of one record (`super::SOURCE`, not ahead) bounds it.
        if (active) {
            job = jobs.Load4(16 + i * 16);
            kind = (job.x >> 4) & 15;
            words = min((job.x >> 8) & 15, 5);
            for (uint w = 0; w < words; w++) {
                record[w] = load(job.x & 3, job.y + w * 4);
            }
            uint times = job.x >> 16;
            if (kind <= 1) {
                uint vertices = min(record[0], 1u << 24);
                uint instances = min(record[1], (1u << 26) / max(vertices, 1u));
                cost = vertices * instances;
                if (vertices >= MOST_DRAW) {
                    vertices = 0;
                    instances = 0;
                    cost = 0;
                }
                record[0] = vertices;
                record[1] = instances;
            } else if (kind == 2) {
                uint3 g = min(uint3(record[0], record[1], record[2]), uint3(65535u, 65535u, 65535u));
                if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
                    g = uint3(0, 0, 0);
                }
                cost = g.x * g.y * g.z * 64u;
                record[0] = g.x;
                record[1] = g.y;
                record[2] = g.z;
            } else {
                uint asked = record[0];
                // The clamp's notes on mesh records, once per execution.
                uint was;
                budget.InterlockedMax(4, asked, was);
                budget.InterlockedAdd(8, min(asked, 1u << 24) * times, was);
                record[0] = min(record[0], 65535u);
                record[1] = min(record[1], 1u);
                record[2] = min(record[2], 1u);
                cost = record[0] * record[1] * record[2] * 128u;
            }
            cost *= times;
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        // Lane 0 charges the jobs in order, as the clamp's compare and swap
        // does one record after another: one that does not fit runs empty
        // and takes what was left.
        if (lane == 0) {
            uint n = min(64u, count - first);
            for (uint k = 0; k < n; k++) {
                uint c = costs[k];
                if (c == 0) {
                    costs[k] = 1;
                    continue;
                }
                if (c > left) {
                    costs[k] = 0;
                    left = 0;
                } else {
                    costs[k] = 1;
                    left -= c;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (active) {
            if (costs[lane] == 0) {
                record[0] = 0;
                record[1] = 0;
                if (kind > 1) {
                    record[2] = 0;
                }
            }
            for (uint s = 0; s < words; s++) {
                dst_args.Store(job.z + s * 4, record[s]);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0 && count > 0) {
        budget.Store(0, BUDGET - left);
    }
}
"#;

/// The words of a record of `kind` (PC argument numbering).
pub(crate) fn words(kind: u32) -> Option<u32> {
    super::min_stride(kind).map(|bytes| bytes / 4)
}

/// A job's first word.
pub(crate) fn job(source: u32, kind: u32, times: u32) -> u32 {
    source | (kind << 4) | (words(kind).unwrap_or(0) << 8) | (times << 16)
}

/// The jobs of a pass as the CPU fills them ([`SOURCE`]), and where it
/// keeps which record is where.
#[derive(Default)]
pub(crate) struct Table {
    /// The table in upload memory ([`TABLE`] bytes).
    at: usize,
    /// The place of the first clamped record in the scratch buffer.
    base: u32,
    /// The host copies it can read, by number ([`crate::shadow::Shadow::id`];
    /// 0: none).
    sources: [u64; SOURCES],
    /// How many executions each job is charged for so far.
    times: Vec<u32>,
    /// The job of a record (source, offset, kind).
    known: std::collections::HashMap<(u32, u32, u32), u32>,
}

// SAFETY: the table's upload memory is the list's, used under its state.
unsafe impl Send for Table {}

impl Table {
    /// A table at `at`, with no job yet.
    ///
    /// # Safety
    /// `at` is [`TABLE`] bytes of mapped upload memory, 4-byte aligned, that
    /// live until the list that reads them is reset and is not run before
    /// it is closed.
    pub(crate) unsafe fn new(at: *mut u8, base: u32, sources: [u64; SOURCES]) -> Table {
        // SAFETY: the caller's.
        unsafe { at.cast::<u32>().write_volatile(0) };
        Table { at: at as usize, base, sources, ..Table::default() }
    }

    /// The place in the scratch buffer of the record of `kind` at `offset`
    /// of the host copy `id`, once the pass has run: a job of it is added if
    /// there is none. `None` if the pass cannot read that copy or has no
    /// room for another job.
    pub(crate) fn place(&mut self, id: u64, offset: u64, kind: u32) -> Option<u64> {
        let source = self.sources.iter().position(|&s| s == id && s != 0)? as u32;
        let offset = u32::try_from(offset).ok().filter(|o| o.is_multiple_of(4))?;
        offset.checked_add(words(kind)? * 4)?;
        let key = (source, offset, kind);
        let table = self.at as *mut u32;
        let place = |i: u32| u64::from(self.base + i * SLOT);
        if let Some(&i) = self.known.get(&key)
            && self.times[i as usize] < TIMES
        {
            self.times[i as usize] += 1;
            // SAFETY: the job's own word of the table ([`Table::new`]).
            unsafe { table.add(4 + 4 * i as usize).write_volatile(job(source, kind, self.times[i as usize])) };
            return Some(place(i));
        }
        let i = self.times.len() as u32;
        if i == JOBS {
            return None;
        }
        self.times.push(1);
        self.known.insert(key, i);
        // SAFETY: inside the table: fewer than [`JOBS`] jobs before it.
        unsafe {
            let to = table.add(4 + 4 * i as usize);
            to.write_volatile(job(source, kind, 1));
            to.add(1).write_volatile(offset);
            to.add(2).write_volatile(self.base + i * SLOT);
            to.add(3).write_volatile(0);
            table.write_volatile(i + 1);
        }
        Some(place(i))
    }

    /// The jobs so far.
    pub(crate) fn len(&self) -> usize {
        self.times.len()
    }
}

/// Records the pass over the table at `jobs` (a host address) into `dst`,
/// which is in state `before` and left in `INDIRECT_ARGUMENT`: after every
/// earlier shader write (the title's argument passes), before what reads
/// the clamped records.
pub(crate) fn record(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    (dst, before): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: u64,
    sources: [u64; SOURCES],
    jobs: u64,
) {
    // Diagnostics: the pass timed on the card (`VICTORY_CARD_METER_CLAMPS=1`).
    let timed = crate::cardmeter::clamp_begin(l);
    // SAFETY: an open list; our root signature and pipeline; live buffers:
    // the sources and the table are read in the common state (promoted).
    unsafe {
        l.ResourceBarrier(&[uav_barrier(), transition(dst, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        for (i, source) in sources.into_iter().enumerate() {
            l.SetComputeRootShaderResourceView(i as u32, source);
        }
        l.SetComputeRootShaderResourceView(SOURCES as u32, jobs);
        l.SetComputeRootUnorderedAccessView(SOURCES as u32 + 1, dst.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(SOURCES as u32 + 2, budget);
        l.Dispatch(1, 1, 1);
        // The budget word, for the next pass or clamp; the records, for
        // the executions.
        l.ResourceBarrier(&[uav_barrier(), transition(dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)]);
    }
    crate::cardmeter::end(l, timed);
}
