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
mod mdl;
mod parse_line;

// ---------------------------------------------------------------------------
// Versioning
// ---------------------------------------------------------------------------

/// Monotonically increasing ABI version. Bumped whenever any `#[repr(C)]`
/// layout or function signature changes in a way that breaks callers.
#[no_mangle]
pub extern "C" fn lc_rs_abi_version() -> u32 {
    3
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

/// Anti-unification variant of [`lc_rs_infer_regex`]. Samples are
/// aligned pairwise via LCS before position-matching, so variable-
/// length clusters collapse runs of unmatched tokens to a single
/// wildcard instead of padding with alternations. All other wire
/// formats and semantics match `lc_rs_infer_regex`.
///
/// # Safety
/// Same contract as `lc_rs_infer_regex`.
#[no_mangle]
pub unsafe extern "C" fn lc_rs_infer_regex_aligned(
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

    let out = match infer::infer_regex_aligned(samples, replacements, wildcard, classes) {
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

/// One span tagged with its source-line index for the batched parser.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LcLineSpan {
    pub line_idx: u32,
    pub start: u32,
    pub end: u32,
    pub label_idx: i32,
}

#[repr(C)]
pub struct LcLineSpans {
    pub spans_ptr: *mut LcLineSpan,
    pub spans_len: usize,
    spans_cap: usize,
    /// Prefix-sum layout: line `i`'s spans are `spans_ptr[offsets[i]..offsets[i+1]]`.
    /// Has length `num_lines + 1`.
    pub offsets_ptr: *mut u32,
    pub offsets_len: usize,
    offsets_cap: usize,
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

/// Release an `LcLineSpans` returned by `lc_rs_parse_lines`. Accepts null.
///
/// # Safety
/// `ptr` must originate from this crate's allocator; never call twice.
#[no_mangle]
pub unsafe extern "C" fn lc_rs_free_line_spans(ptr: *mut LcLineSpans) {
    if ptr.is_null() {
        return;
    }
    let b = Box::from_raw(ptr);
    let _ = Vec::from_raw_parts(b.spans_ptr, b.spans_len, b.spans_cap);
    let _ = Vec::from_raw_parts(b.offsets_ptr, b.offsets_len, b.offsets_cap);
}

/// Batched form of [`lc_rs_parse_line`]. Compiles each regex once and
/// runs the cascade over every line in `lines_buf`. Returns a
/// prefix-sum view: the spans for line `i` are
/// `spans[offsets[i]..offsets[i+1]]`.
///
/// `lines_buf` wire format: `u32 num_lines` followed by each line as
/// `u32 len` + `len` UTF-8 bytes.
///
/// # Safety
/// - All pointers must be valid for the stated lengths.
/// - `lines_buf` must encode the wire format above.
/// - Free with [`lc_rs_free_line_spans`].
#[no_mangle]
pub unsafe extern "C" fn lc_rs_parse_lines(
    lines_buf_ptr: *const u8,
    lines_buf_len: usize,
    labels_ptr: *const *const u8,
    labels_len_ptr: *const usize,
    patterns_ptr: *const *const u8,
    patterns_len_ptr: *const usize,
    num_labels: usize,
) -> *mut LcLineSpans {
    if lines_buf_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let buf = slice::from_raw_parts(lines_buf_ptr, lines_buf_len);
    let labels_names = slice_of_strs(labels_ptr, labels_len_ptr, num_labels);
    let pattern_strs = slice_of_strs(patterns_ptr, patterns_len_ptr, num_labels);
    let (labels_names, pattern_strs) = match (labels_names, pattern_strs) {
        (Some(a), Some(b)) if a.len() == b.len() => (a, b),
        _ => return std::ptr::null_mut(),
    };
    let (mut spans, mut offsets) = match parse_line::parse_lines(buf, &labels_names, &pattern_strs)
    {
        Some(v) => v,
        None => return std::ptr::null_mut(),
    };
    let spans_ptr = spans.as_mut_ptr();
    let spans_len = spans.len();
    let spans_cap = spans.capacity();
    std::mem::forget(spans);
    let offsets_ptr = offsets.as_mut_ptr();
    let offsets_len = offsets.len();
    let offsets_cap = offsets.capacity();
    std::mem::forget(offsets);
    Box::into_raw(Box::new(LcLineSpans {
        spans_ptr,
        spans_len,
        spans_cap,
        offsets_ptr,
        offsets_len,
        offsets_cap,
    }))
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

// ---------------------------------------------------------------------------
// MDL ladder + RegexSet (Stage E″)
// ---------------------------------------------------------------------------

/// Decode a packed (u32 num + (u32 len, bytes)*) string list into
/// owned-borrow `&str`s.
unsafe fn decode_token_list(buf: &[u8]) -> Option<Vec<&str>> {
    let mut cursor = 0usize;
    if buf.len() < 4 {
        return None;
    }
    let num = u32::from_le_bytes(buf[0..4].try_into().ok()?) as usize;
    cursor += 4;
    let mut out = Vec::with_capacity(num);
    for _ in 0..num {
        if cursor + 4 > buf.len() {
            return None;
        }
        let len = u32::from_le_bytes(buf[cursor..cursor + 4].try_into().ok()?) as usize;
        cursor += 4;
        if cursor + len > buf.len() {
            return None;
        }
        let s = std::str::from_utf8(&buf[cursor..cursor + len]).ok()?;
        cursor += len;
        out.push(s);
    }
    if cursor != buf.len() {
        return None;
    }
    Some(out)
}

/// Decode a packed (label, pattern) pair list matching
/// `infer::decode_classes`' layout.
unsafe fn decode_typed_battery(buf: &[u8]) -> Option<Vec<(&str, &str)>> {
    let mut cursor = 0usize;
    if buf.is_empty() {
        return Some(Vec::new());
    }
    if buf.len() < 4 {
        return None;
    }
    let num = u32::from_le_bytes(buf[0..4].try_into().ok()?) as usize;
    cursor += 4;
    let mut out = Vec::with_capacity(num);
    for _ in 0..num {
        let mut read_str = || -> Option<&str> {
            if cursor + 4 > buf.len() {
                return None;
            }
            let len = u32::from_le_bytes(buf[cursor..cursor + 4].try_into().ok()?) as usize;
            cursor += 4;
            if cursor + len > buf.len() {
                return None;
            }
            let s = std::str::from_utf8(&buf[cursor..cursor + len]).ok()?;
            cursor += len;
            Some(s)
        };
        let label = read_str()?;
        let pattern = read_str()?;
        out.push((label, pattern));
    }
    if cursor != buf.len() {
        return None;
    }
    Some(out)
}

/// Slot-ladder pass: given packed slot realisations and a typed
/// battery, return the tightest regex that matches every value.
///
/// Wire format of both `values_buf` and `typed_buf` follows the
/// `infer_regex` convention. `enum_max` is the maximum distinct-value
/// cardinality at which the ladder emits an enumerated alternation.
/// `wildcard_ptr`/`_len` is the fallback fragment (default `.*?` if
/// null / zero-length).
///
/// # Safety
/// All pointers must be valid for their lengths.
#[no_mangle]
pub unsafe extern "C" fn lc_rs_slot_ladder(
    values_ptr: *const u8,
    values_len: usize,
    typed_ptr: *const u8,
    typed_len: usize,
    enum_max: u32,
    wildcard_ptr: *const u8,
    wildcard_len: usize,
) -> *mut LcBytes {
    if values_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let values_buf = slice::from_raw_parts(values_ptr, values_len);
    let values = match decode_token_list(values_buf) {
        Some(v) => v,
        None => return std::ptr::null_mut(),
    };
    let typed_buf: &[u8] = if typed_ptr.is_null() || typed_len == 0 {
        &[]
    } else {
        slice::from_raw_parts(typed_ptr, typed_len)
    };
    let typed = match decode_typed_battery(typed_buf) {
        Some(v) => v,
        None => return std::ptr::null_mut(),
    };
    let wildcard = if wildcard_ptr.is_null() || wildcard_len == 0 {
        ".*?"
    } else {
        match std::str::from_utf8(slice::from_raw_parts(wildcard_ptr, wildcard_len)) {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };
    let out = mdl::slot_ladder_pattern(&values, &typed, enum_max as usize, wildcard);
    into_raw_bytes(out.into_bytes())
}

/// Alternation-minimiser: return a sorted, dedup'd, escaped
/// alternation group from a packed list of strings.
///
/// # Safety
/// `alts_buf` must follow the standard `u32 num + (u32 len, bytes)*`
/// wire format.
#[no_mangle]
pub unsafe extern "C" fn lc_rs_alt_min(
    alts_ptr: *const u8,
    alts_len: usize,
    wildcard_ptr: *const u8,
    wildcard_len: usize,
) -> *mut LcBytes {
    if alts_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let buf = slice::from_raw_parts(alts_ptr, alts_len);
    let alts = match decode_token_list(buf) {
        Some(v) => v,
        None => return std::ptr::null_mut(),
    };
    let wildcard = if wildcard_ptr.is_null() || wildcard_len == 0 {
        ".*?"
    } else {
        match std::str::from_utf8(slice::from_raw_parts(wildcard_ptr, wildcard_len)) {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };
    let out = mdl::alt_min(&alts, wildcard);
    into_raw_bytes(out.into_bytes())
}

/// Anchored verification. Returns packed `(u32 hits, u32 total)` via
/// the two-slot layout below.
#[repr(C)]
pub struct LcVerify {
    pub hits: u32,
    pub total: u32,
}

/// # Safety
/// Pointers must be valid for their lengths; `samples_buf` uses the
/// standard packed string list.
#[no_mangle]
pub unsafe extern "C" fn lc_rs_verify_pattern(
    pattern_ptr: *const u8,
    pattern_len: usize,
    samples_ptr: *const u8,
    samples_len: usize,
) -> LcVerify {
    if pattern_ptr.is_null() || samples_ptr.is_null() {
        return LcVerify { hits: 0, total: 0 };
    }
    let pattern =
        match std::str::from_utf8(slice::from_raw_parts(pattern_ptr, pattern_len)) {
            Ok(s) => s,
            Err(_) => return LcVerify { hits: 0, total: 0 },
        };
    let buf = slice::from_raw_parts(samples_ptr, samples_len);
    let samples = match decode_token_list(buf) {
        Some(v) => v,
        None => return LcVerify { hits: 0, total: 0 },
    };
    let (h, t) = mdl::verify_pattern(pattern, &samples);
    LcVerify {
        hits: h as u32,
        total: t as u32,
    }
}

/// MDL-style cost estimate in bits for `pattern` over `samples`.
///
/// # Safety
/// Same contract as `lc_rs_verify_pattern`.
#[no_mangle]
pub unsafe extern "C" fn lc_rs_mdl_cost(
    pattern_ptr: *const u8,
    pattern_len: usize,
    samples_ptr: *const u8,
    samples_len: usize,
) -> f64 {
    if pattern_ptr.is_null() || samples_ptr.is_null() {
        return f64::NAN;
    }
    let pattern =
        match std::str::from_utf8(slice::from_raw_parts(pattern_ptr, pattern_len)) {
            Ok(s) => s,
            Err(_) => return f64::NAN,
        };
    let buf = slice::from_raw_parts(samples_ptr, samples_len);
    let samples = match decode_token_list(buf) {
        Some(v) => v,
        None => return f64::NAN,
    };
    mdl::mdl_cost(pattern, &samples)
}

// ---- RegexSet multi-pattern matcher ----

/// Opaque handle for a compiled `regex::RegexSet`.
#[repr(C)]
pub struct LcRegexSet {
    _priv: [u8; 0],
}

/// Compile a list of regexes into a `RegexSet`. Returns null if any
/// pattern fails to parse — all-or-nothing, like Hyperscan.
///
/// # Safety
/// `patterns_buf` must encode the standard packed string list.
#[no_mangle]
pub unsafe extern "C" fn lc_rs_regexset_compile(
    patterns_ptr: *const u8,
    patterns_len: usize,
) -> *mut LcRegexSet {
    if patterns_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let buf = slice::from_raw_parts(patterns_ptr, patterns_len);
    let patterns = match decode_token_list(buf) {
        Some(v) => v,
        None => return std::ptr::null_mut(),
    };
    match mdl::regexset_compile(&patterns) {
        Some(c) => Box::into_raw(Box::new(c)) as *mut LcRegexSet,
        None => std::ptr::null_mut(),
    }
}

/// Match `line` against every pattern in `handle`. Returns a heap
/// `LcBytes` holding a contiguous `u32` array of matched indices
/// (little-endian) — length is `bytes.len / 4`. Empty set → zero-length
/// bytes. Must be freed with [`lc_rs_free_bytes`].
///
/// # Safety
/// `handle` must be a live pointer returned by
/// [`lc_rs_regexset_compile`].
#[no_mangle]
pub unsafe extern "C" fn lc_rs_regexset_match(
    handle: *const LcRegexSet,
    line_ptr: *const u8,
    line_len: usize,
) -> *mut LcBytes {
    if handle.is_null() || line_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let set = &*(handle as *const mdl::CompiledRegexSet);
    let line = match std::str::from_utf8(slice::from_raw_parts(line_ptr, line_len)) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let idxs = mdl::regexset_match(set, line);
    let mut out = Vec::with_capacity(idxs.len() * 4);
    for i in idxs {
        out.extend_from_slice(&i.to_le_bytes());
    }
    into_raw_bytes(out)
}

/// Release a `RegexSet` returned by [`lc_rs_regexset_compile`].
///
/// # Safety
/// Pointer must originate from this crate's allocator; never call
/// twice.
#[no_mangle]
pub unsafe extern "C" fn lc_rs_free_regexset(handle: *mut LcRegexSet) {
    if handle.is_null() {
        return;
    }
    drop(Box::from_raw(handle as *mut mdl::CompiledRegexSet));
}

// ---------------------------------------------------------------------------
// Helpers (cont.)
// ---------------------------------------------------------------------------

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
        assert_eq!(lc_rs_abi_version(), 3);
    }
}
