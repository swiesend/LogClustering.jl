//! FFI entry points for `LogClustering.jl`.
//!
//! The Julia host owns all input buffers and is responsible for calling the
//! matching `lc_rs_*_free` function on every handle returned from this
//! crate. All strings crossing the boundary are UTF-8 byte slices
//! (`*const u8` + `usize`).
//!
//! Current surface:
//!
//! - `lc_rs_infer_regex` — port of the thesis Algorithm 3.10
//!   ("Log-Key Inferenz"): take a set of tokenised samples and return a
//!   regex that matches every sample, collapsing disagreements into
//!   alternations.
//! - `lc_rs_parse_line` — port of the thesis Algorithm 3.1
//!   ("Event-Log Parser"): recursive regex-cascade descent over a line.

use std::slice;

mod infer;
mod parse_line;

// ---------------------------------------------------------------------------
// Versioning
// ---------------------------------------------------------------------------

/// Monotonically increasing ABI version. Bumped whenever any `#[repr(C)]`
/// layout or function signature changes in a way that breaks callers.
#[no_mangle]
pub extern "C" fn lc_rs_abi_version() -> u32 {
    2
}

// ---------------------------------------------------------------------------
// infer_regex
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct LcBytes {
    pub ptr: *mut u8,
    pub len: usize,
    cap: usize,
}

/// Infer a regex that matches every sample in a packed buffer.
///
/// Wire format of `samples_buf` (all integers little-endian):
///
/// ```text
/// u32 num_samples
/// num_samples × {
///     u32 num_words
///     num_words × { u32 word_len; u8[word_len] word }
/// }
/// ```
///
/// `replacement_tokens_buf` follows the same wire format as *one* sample —
/// a `u32` count followed by length-prefixed tokens. Any word that contains
/// one of these tokens as a substring is replaced in the output by the
/// `wildcard` regex fragment. Pass `num_replacements = 0` for none.
///
/// `wildcard_ptr` / `wildcard_len` gives the regex fragment substituted for
/// replacement-token matches. Pass `""` / `0` to use the default `.*?`.
///
/// `classes_buf` is an optional label → tight-regex map (Stage E″ MDL
/// ladder). Wire format:
///
/// ```text
/// u32 num_classes
/// num_classes × {
///     u32 name_len;    u8[name_len]    name
///     u32 pattern_len; u8[pattern_len] pattern
/// }
/// ```
///
/// When a token exactly equals `%NAME%` and `NAME` is in the map, the
/// corresponding pattern is emitted instead of the default wildcard.
/// Pass `classes_len = 0` to skip.
///
/// Returns a heap-allocated `LcBytes` with the result string, or null on
/// malformed input. The caller must release it with [`lc_rs_free_bytes`].
///
/// # Safety
/// All pointers must be valid for the stated lengths; buffers must be
/// byte-aligned. The function borrows, never retains, caller memory.
#[no_mangle]
pub unsafe extern "C" fn lc_rs_infer_regex(
    samples_ptr: *const u8,
    samples_len: usize,
    replacements_ptr: *const u8,
    replacements_len: usize,
    wildcard_ptr: *const u8,
    wildcard_len: usize,
    classes_ptr: *const u8,
    classes_len: usize,
) -> *mut LcBytes {
    if samples_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let samples = slice::from_raw_parts(samples_ptr, samples_len);
    let replacements = if replacements_ptr.is_null() || replacements_len == 0 {
        &[][..]
    } else {
        slice::from_raw_parts(replacements_ptr, replacements_len)
    };
    let wildcard = if wildcard_ptr.is_null() || wildcard_len == 0 {
        ".*?"
    } else {
        match std::str::from_utf8(slice::from_raw_parts(wildcard_ptr, wildcard_len)) {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };
    let classes = if classes_ptr.is_null() || classes_len == 0 {
        &[][..]
    } else {
        slice::from_raw_parts(classes_ptr, classes_len)
    };

    let out = match infer::infer_regex(samples, replacements, wildcard, classes) {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };

    into_raw_bytes(out.into_bytes())
}

// ---------------------------------------------------------------------------
// parse_line
// ---------------------------------------------------------------------------

/// One span produced by `lc_rs_parse_line`.
///
/// Byte offsets are 0-based, half-open (`[start, end)`). `label_idx` is the
/// 0-based index into the caller's labels array, or `-1` for a raw
/// substring between matches.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LcSpan {
    pub start: u32,
    pub end: u32,
    pub label_idx: i32,
}

#[repr(C)]
pub struct LcSpans {
    pub ptr: *mut LcSpan,
    pub len: usize,
    cap: usize,
}

/// Port of thesis Algorithm 3.1: recursive regex-cascade parser.
///
/// For each `label`, the parser finds all non-overlapping matches of the
/// corresponding `pattern` in the current slice and recurses on the
/// unmatched regions with the remaining labels. Rank order of `labels`
/// matters — complex patterns should come first.
///
/// The returned [`LcSpans`] is an ordered, non-overlapping decomposition
/// of the input, where each entry is either a matched label (`label_idx
/// >= 0`) or a raw substring between matches (`label_idx == -1`).
///
/// Patterns are compiled with the `regex` crate (RE2-flavoured, linear
/// time, no backrefs or lookaround). Invalid patterns cause the whole
/// call to return null.
///
/// # Safety
/// - `line_ptr` must be valid for `line_len` bytes.
/// - `labels_ptr` and `labels_len_ptr` must each point to
///   `num_labels`-element arrays.
/// - Same for `patterns_ptr` / `patterns_len_ptr`.
/// - Each inner pointer must be valid for its paired length.
#[no_mangle]
pub unsafe extern "C" fn lc_rs_parse_line(
    line_ptr: *const u8,
    line_len: usize,
    labels_ptr: *const *const u8,
    labels_len_ptr: *const usize,
    patterns_ptr: *const *const u8,
    patterns_len_ptr: *const usize,
    num_labels: usize,
) -> *mut LcSpans {
    if line_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let line = slice::from_raw_parts(line_ptr, line_len);

    // Collect label names (only used for validation; indices are returned).
    let labels_names = slice_of_strs(labels_ptr, labels_len_ptr, num_labels);
    let pattern_strs = slice_of_strs(patterns_ptr, patterns_len_ptr, num_labels);
    let (labels_names, pattern_strs) = match (labels_names, pattern_strs) {
        (Some(a), Some(b)) if a.len() == b.len() => (a, b),
        _ => return std::ptr::null_mut(),
    };

    let out = match parse_line::parse_line(line, &labels_names, &pattern_strs) {
        Some(v) => v,
        None => return std::ptr::null_mut(),
    };
    into_raw_spans(out)
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Release an `LcBytes` returned by this crate. Accepts null.
///
/// # Safety
/// `ptr` must originate from this crate's allocator (i.e., an earlier
/// `lc_rs_*` call); never call twice on the same pointer.
#[no_mangle]
pub unsafe extern "C" fn lc_rs_free_bytes(ptr: *mut LcBytes) {
    if ptr.is_null() {
        return;
    }
    let b = Box::from_raw(ptr);
    let _ = Vec::from_raw_parts(b.ptr, b.len, b.cap);
}

/// Release an `LcSpans` returned by this crate. Accepts null.
///
/// # Safety
/// `ptr` must originate from this crate's allocator; never call twice.
#[no_mangle]
pub unsafe extern "C" fn lc_rs_free_spans(ptr: *mut LcSpans) {
    if ptr.is_null() {
        return;
    }
    let b = Box::from_raw(ptr);
    let _ = Vec::from_raw_parts(b.ptr, b.len, b.cap);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn into_raw_bytes(mut v: Vec<u8>) -> *mut LcBytes {
    let ptr = v.as_mut_ptr();
    let len = v.len();
    let cap = v.capacity();
    std::mem::forget(v);
    Box::into_raw(Box::new(LcBytes { ptr, len, cap }))
}

fn into_raw_spans(mut v: Vec<LcSpan>) -> *mut LcSpans {
    let ptr = v.as_mut_ptr();
    let len = v.len();
    let cap = v.capacity();
    std::mem::forget(v);
    Box::into_raw(Box::new(LcSpans { ptr, len, cap }))
}

unsafe fn slice_of_strs<'a>(
    ptrs: *const *const u8,
    lens: *const usize,
    n: usize,
) -> Option<Vec<&'a str>> {
    if ptrs.is_null() || lens.is_null() {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let p = *ptrs.add(i);
        let l = *lens.add(i);
        if p.is_null() {
            return None;
        }
        let bytes = slice::from_raw_parts(p, l);
        let s = std::str::from_utf8(bytes).ok()?;
        out.push(s);
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Tests (cargo test)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_matches_crate_constant() {
        // Must equal the version documented in `lc_rs_abi_version`; the
        // Julia side's `Rust.ABI_VERSION` tracks the same integer.
        assert_eq!(lc_rs_abi_version(), 2);
    }
}
