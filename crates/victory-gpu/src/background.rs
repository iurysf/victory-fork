//! The D3D12 runtime's and the driver's work in the background, measured
//! and, as an experiment, set (`VICTORY_D3D_BACKGROUND`).
//!
//! Within one run in the yard the frame went from about 190-200 ms with
//! some 8.5 cores busy to about 150-160 ms with some 5, by itself, at a time
//! that varied (about 193 s in one run, 406-416 s in another; two runs never
//! turned), with none of our pipelines being made. The runtime's "D3D
//! Background Thread"s were seen busy in a profile; the runtime compiles
//! shaders again there for minutes after pipelines are made or loaded.

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

/// `VICTORY_D3D_BACKGROUND=<mode>` (read once, when the device is made; not
/// turned by the A/B): `disable` (D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_BACKGROUND_WORK),
/// `no_profiling` (…_DISABLE_PROFILING_BY_SYSTEM), `intrusive`
/// (…_ALLOW_INTRUSIVE_MEASUREMENTS with D3D12_MEASUREMENTS_ACTION_COMMIT_RESULTS_HIGH_PRIORITY:
/// the work done early rather than dragged out), `allowed` (…_ALLOWED with
/// …_KEEP_ALL, the runtime's own default, asked for). Unset: nothing is
/// asked, as before. The older `VICTORY_D3D12_BACKGROUND=0` is `disable`.
/// The modes are meant for development: the runtime may refuse them
/// outside developer mode, and the log line says what it answered.
fn asked() -> Option<(&'static str, D3D12_BACKGROUND_PROCESSING_MODE, D3D12_MEASUREMENTS_ACTION)> {
    let named = std::env::var("VICTORY_D3D_BACKGROUND").ok();
    let old = std::env::var_os("VICTORY_D3D12_BACKGROUND").is_some_and(|v| v == "0");
    let name = match named.as_deref().map(str::trim) {
        Some(name) => name.to_ascii_lowercase(),
        None if old => "disable".to_owned(),
        None => return None,
    };
    Some(match name.as_str() {
        "disable" => ("disable", D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_BACKGROUND_WORK, D3D12_MEASUREMENTS_ACTION_KEEP_ALL),
        "no_profiling" => (
            "no_profiling",
            D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_PROFILING_BY_SYSTEM,
            D3D12_MEASUREMENTS_ACTION_KEEP_ALL,
        ),
        "intrusive" => (
            "intrusive",
            D3D12_BACKGROUND_PROCESSING_MODE_ALLOW_INTRUSIVE_MEASUREMENTS,
            D3D12_MEASUREMENTS_ACTION_COMMIT_RESULTS_HIGH_PRIORITY,
        ),
        "allowed" => ("allowed", D3D12_BACKGROUND_PROCESSING_MODE_ALLOWED, D3D12_MEASUREMENTS_ACTION_KEEP_ALL),
        other => {
            tracing::warn!(
                asked = other,
                "VICTORY_D3D_BACKGROUND names no mode (disable, no_profiling, intrusive, allowed): nothing is asked"
            );
            return None;
        }
    })
}

/// Right after the device is made: the background mode asked for, if any,
/// and what the runtime answered.
pub(crate) fn set_mode(device: &ID3D12Device) {
    let Some((name, mode, action)) = asked() else {
        return;
    };
    let Ok(device6) = device.cast::<ID3D12Device6>() else {
        tracing::warn!(mode = name, "VICTORY_D3D_BACKGROUND: the device has no ID3D12Device6, nothing asked");
        return;
    };
    let mut further = windows::core::BOOL::default();
    // SAFETY: a live device; the out pointer is ours.
    let answer = unsafe { device6.SetBackgroundProcessingMode(mode, action, None, Some(&raw mut further)) };
    match answer {
        Ok(()) => tracing::info!(
            mode = name,
            mode_value = mode.0,
            action = action.0,
            further_measurements_desired = further.as_bool(),
            "the runtime's background processing mode was set (VICTORY_D3D_BACKGROUND)"
        ),
        Err(e) => tracing::warn!(
            mode = name,
            mode_value = mode.0,
            action = action.0,
            hresult = format_args!("{:#010x}", e.code().0),
            error = %e,
            "the runtime refused the background processing mode (VICTORY_D3D_BACKGROUND; developer mode may be needed)"
        ),
    }
}

/// Processor time of this process's threads named "D3D Background
/// Thread..." and of those whose name says NVIDIA's driver ("nv" at the
/// start, or "nvidia" in it), since the last call: (threads, ms) of each,
/// and the ms since the start.
#[derive(Default, Clone, Copy)]
struct Times {
    d3d: (u32, u64),
    nvidia: (u32, u64),
}

fn times_now() -> Times {
    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME, INVALID_HANDLE_VALUE, LocalFree};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
    };
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcessId, GetThreadDescription, GetThreadTimes, OpenThread, THREAD_QUERY_LIMITED_INFORMATION,
    };
    let mut times = Times::default();
    // SAFETY: a snapshot and thread handles of ours, closed below; plain
    // data filled in by the queries; the name's string is the system's,
    // freed with LocalFree.
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snapshot == INVALID_HANDLE_VALUE {
            return times;
        }
        let process = GetCurrentProcessId();
        let mut entry: THREADENTRY32 = std::mem::zeroed();
        entry.dwSize = size_of::<THREADENTRY32>() as u32;
        let mut more = Thread32First(snapshot, &mut entry) != 0;
        while more {
            if entry.th32OwnerProcessID == process {
                let thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, entry.th32ThreadID);
                if !thread.is_null() {
                    let mut text: *mut u16 = std::ptr::null_mut();
                    let name = if GetThreadDescription(thread, &mut text) >= 0 && !text.is_null() {
                        let len = (0..).take_while(|&i| *text.add(i) != 0).count();
                        let name = String::from_utf16_lossy(std::slice::from_raw_parts(text, len));
                        LocalFree(text.cast());
                        name
                    } else {
                        String::new()
                    };
                    let lower = name.to_ascii_lowercase();
                    let class = if name.starts_with("D3D Background Thread") {
                        Some(&mut times.d3d)
                    } else if lower.starts_with("nv") || lower.contains("nvidia") {
                        Some(&mut times.nvidia)
                    } else {
                        None
                    };
                    if let Some((n, ms)) = class {
                        let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
                        let (mut a, mut b, mut kernel, mut user) = (zero, zero, zero, zero);
                        if GetThreadTimes(thread, &mut a, &mut b, &mut kernel, &mut user) != 0 {
                            let ticks = |t: FILETIME| u64::from(t.dwHighDateTime) << 32 | u64::from(t.dwLowDateTime);
                            *n += 1;
                            *ms += (ticks(kernel) + ticks(user)) / 10_000;
                        }
                    }
                    CloseHandle(thread);
                }
            }
            more = Thread32Next(snapshot, &mut entry) != 0;
        }
        CloseHandle(snapshot);
    }
    times
}

/// For the frame log's reading of `seconds`: one line with the processor
/// time of the runtime's background threads and of the driver's named
/// ones over it (ms a second: 1,000 is one core busy), and since the start.
/// A thread that ended takes its time with it, so a reading may count less.
pub(crate) fn line(seconds: f64) -> String {
    static LAST: std::sync::Mutex<Option<Times>> = std::sync::Mutex::new(None);
    let now = times_now();
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let before = last.replace(now).unwrap_or_default();
    let per_s = |now: u64, before: u64| now.saturating_sub(before) as f64 / seconds.max(0.001);
    format!(
        "background threads: the runtime's \"D3D Background Thread\"s {} with {:.0} ms of processor a second ({} ms since the \
         start), the driver's named ones {} with {:.0} ms a second ({} ms since the start); mode asked: {}",
        now.d3d.0,
        per_s(now.d3d.1, before.d3d.1),
        now.d3d.1,
        now.nvidia.0,
        per_s(now.nvidia.1, before.nvidia.1),
        now.nvidia.1,
        asked().map_or("none (VICTORY_D3D_BACKGROUND unset)", |a| a.0)
    )
}
