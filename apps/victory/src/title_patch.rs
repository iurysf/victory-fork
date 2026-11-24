//! Debugging aids that patch the title's code before it runs, for its
//! FidelityFX path on Series X (with `VICTORY_FIDELITYFX_DIR`):
//!
//! - `VICTORY_UPSCALER=<rva>[,<kind>]` (hex rva, decimal kind, default 7)
//!   makes the title's upscaler picker return `kind`. In Grand Theft Auto
//!   VI 1.0.2.1 the picker is at 7E93F0, reached by 4 direct calls; it
//!   returns 7 (AMD FidelityFX) only on Series S and 6 (the title's own
//!   upscaler) otherwise.
//! - `VICTORY_FFX_SCALE=<scale>[,<rva>]` sets the render scale of that path
//!   (0.5: 1080p for a 4K output, 0.6667: 1440p). The method that picks it
//!   (1.0.2.1: 28EF680, the default) takes 2/3 on Series S and 1.0 on
//!   Series X, so Series X with FidelityFX forced on renders at full 4K and
//!   overflows its memory plan (a 32 MiB surface in its third graphics
//!   pool, then the transient heap).
//!
//! Each checks the code it patches first and patches nothing in other
//! code (another build of the title): the picker must be called, the scale
//! method must have its two loads and its padding where expected.

use victory_loader::LoadedImage;

/// Applies the patches the environment asks for.
pub(crate) fn apply(image: &LoadedImage) {
    upscaler(image);
    ffx_scale(image);
}

fn hex(s: &str) -> Option<usize> {
    usize::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()
}

/// Writes `bytes` at title+`rva`, inside the image (checked), through a
/// writable mapping for the moment. False if Windows refused it.
fn write_code(image: &LoadedImage, rva: usize, bytes: &[u8]) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    if rva.checked_add(bytes.len()).is_none_or(|end| end > image.info().size_of_image as usize) {
        return false;
    }
    let at = (image.base() + rva) as *mut u8;
    let mut old = 0;
    // SAFETY: bytes inside the mapped image; no title code runs yet.
    unsafe {
        if VirtualProtect(at.cast(), bytes.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return false;
        }
        at.copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
        VirtualProtect(at.cast(), bytes.len(), old, &mut old);
    }
    true
}

/// The direct calls and jumps (`E8`/`E9 rel32`) to `rva` in the image's
/// executable sections.
fn calls_to(image: &LoadedImage, rva: usize) -> usize {
    const EXECUTE: u32 = 0x2000_0000;
    let mut calls = 0;
    for s in image.info().sections.iter().filter(|s| s.characteristics & EXECUTE != 0) {
        let start = s.rva as usize;
        // SAFETY: an executable section of the mapped image, readable.
        let code = unsafe { std::slice::from_raw_parts((image.base() + start) as *const u8, s.virtual_size as usize) };
        calls += branches_to(code, start, rva);
    }
    calls
}

/// In `code` (at title+`start`), the `E8`/`E9 rel32` branches whose
/// target is title+`rva`.
fn branches_to(code: &[u8], start: usize, rva: usize) -> usize {
    code.windows(5)
        .enumerate()
        .filter(|(i, w)| {
            let rel = i64::from(i32::from_le_bytes([w[1], w[2], w[3], w[4]]));
            matches!(w[0], 0xE8 | 0xE9) && (start + i + 5) as i64 + rel == rva as i64
        })
        .count()
}

fn upscaler(image: &LoadedImage) {
    let Ok(spec) = std::env::var("VICTORY_UPSCALER") else {
        return;
    };
    let (rva, kind) = spec.split_once(',').unwrap_or((&spec, "7"));
    let (Some(rva), Ok(kind)) = (hex(rva), kind.trim().parse::<u32>()) else {
        eprintln!("VIctory: VICTORY_UPSCALER={spec}: expected <hex rva>[,<kind>]");
        return;
    };
    let calls = calls_to(image, rva);
    if calls == 0 {
        eprintln!("VIctory: upscaler picker at title+{rva:#x} not patched: no direct call to it (wrong rva?)");
        return;
    }
    if write_code(image, rva, &picker_stub(kind)) {
        eprintln!("VIctory: upscaler picker at title+{rva:#x} returns {kind} ({calls} direct calls)");
    } else {
        eprintln!("VIctory: upscaler picker at title+{rva:#x} not patched: not writable");
    }
}

/// `mov eax, kind; ret`.
fn picker_stub(kind: u32) -> [u8; 6] {
    let mut stub = [0xB8, 0, 0, 0, 0, 0xC3];
    stub[1..5].copy_from_slice(&kind.to_le_bytes());
    stub
}

/// The bytes of the scale method [`patch_scale`] reads and writes.
const SCALE_SPAN: usize = 0x68;

fn ffx_scale(image: &LoadedImage) {
    let Ok(spec) = std::env::var("VICTORY_FFX_SCALE") else {
        return;
    };
    let (scale, rva) = spec.split_once(',').unwrap_or((&spec, "28EF680"));
    let scale = scale.trim().parse::<f32>().ok().filter(|s| *s > 0.0 && *s <= 1.0);
    let (Some(scale), Some(rva)) = (scale, hex(rva)) else {
        eprintln!("VIctory: VICTORY_FFX_SCALE={spec}: expected <scale in (0, 1]>[,<hex rva>]");
        return;
    };
    if rva.checked_add(SCALE_SPAN).is_none_or(|end| end > image.info().size_of_image as usize) {
        eprintln!("VIctory: VICTORY_FFX_SCALE: title+{rva:#x} is outside the image");
        return;
    }
    // SAFETY: the method's bytes and its padding, inside the mapped image
    // (checked above), read before any title code runs.
    let mut code = unsafe { std::slice::from_raw_parts((image.base() + rva) as *const u8, SCALE_SPAN) }.to_vec();
    if !patch_scale(&mut code, scale) {
        eprintln!("VIctory: FidelityFX scale not patched: unexpected code at title+{rva:#x} (wrong rva?)");
    } else if write_code(image, rva, &code) {
        eprintln!("VIctory: FidelityFX render scale at title+{rva:#x} set to {scale}");
    } else {
        eprintln!("VIctory: FidelityFX scale at title+{rva:#x} not patched: not writable");
    }
}

/// Points both scale loads of the method `code` starts with
/// (`vmovss xmm0, [rip+d]` at +0x28 and +0x32) at `scale`, written into the
/// method's `int3` padding at +0x64. False, with nothing written, for other
/// code.
fn patch_scale(code: &mut [u8], scale: f32) -> bool {
    const LOADS: [usize; 2] = [0x28, 0x32];
    const SLOT: usize = 0x64;
    const VMOVSS_RIP: [u8; 4] = [0xC5, 0xFA, 0x10, 0x05];
    if code.len() < SCALE_SPAN || LOADS.iter().any(|&l| code[l..l + 4] != VMOVSS_RIP) || code[SLOT..SLOT + 4] != [0xCC; 4] {
        return false;
    }
    code[SLOT..SLOT + 4].copy_from_slice(&scale.to_le_bytes());
    for l in LOADS {
        let disp = (SLOT - (l + 8)) as i32;
        code[l + 4..l + 8].copy_from_slice(&disp.to_le_bytes());
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scale_loads_point_at_the_constant() {
        // Synthetic code: the two loads, filler, and int3 padding.
        let mut code = vec![0x90u8; SCALE_SPAN];
        for at in [0x28, 0x32] {
            code[at..at + 8].copy_from_slice(&[0xC5, 0xFA, 0x10, 0x05, 1, 2, 3, 4]);
        }
        code[0x60..].fill(0xCC);
        assert!(patch_scale(&mut code, 0.5));
        assert_eq!(code[0x64..0x68], 0.5f32.to_le_bytes());
        // Each load's next instruction plus its displacement is the constant.
        for at in [0x28, 0x32] {
            let disp = i32::from_le_bytes(code[at + 4..at + 8].try_into().unwrap());
            assert_eq!(at as i32 + 8 + disp, 0x64);
        }
        // Already patched, so no padding left: nothing is written again.
        let before = code.clone();
        assert!(!patch_scale(&mut code, 0.25));
        assert_eq!(code, before);
    }

    #[test]
    fn other_code_is_left_alone() {
        let mut code = vec![0xCCu8; SCALE_SPAN];
        assert!(!patch_scale(&mut code, 0.5));
        assert!(code.iter().all(|&b| b == 0xCC));
    }

    #[test]
    fn calls_and_jumps_to_the_picker_are_counted() {
        // At title+0x1000: a call to +0x1010, a jump to it, a call elsewhere.
        let mut code = vec![0x90u8; 0x20];
        code[0..5].copy_from_slice(&[0xE8, 0x0B, 0, 0, 0]); // 0x1005 + 0x0b
        code[5..10].copy_from_slice(&[0xE9, 0x06, 0, 0, 0]); // 0x100a + 0x06
        code[10..15].copy_from_slice(&[0xE8, 0x20, 0, 0, 0]);
        assert_eq!(branches_to(&code, 0x1000, 0x1010), 2);
        assert_eq!(branches_to(&code, 0x1000, 0x2000), 0);
    }

    #[test]
    fn the_picker_returns_the_kind() {
        assert_eq!(picker_stub(7), [0xB8, 7, 0, 0, 0, 0xC3]);
    }
}
