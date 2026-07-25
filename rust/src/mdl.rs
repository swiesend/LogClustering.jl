//! Stage E″ MDL ladder — plan 001.
//!
//! The companion to `infer.rs`. Given a cluster's per-slot realisations
//! this module picks the tightest regex that still matches every value
//! and reports an MDL cost so the pipeline can prefer the simplest form.
//! Five passes ("the ladder"):
//!
//! 1. **Exact literal** — all realisations identical → escape the value.
//! 2. **Enum** — distinct values ≤ `enum_max` → `(?:v1|v2|…)` (alt-min
//!    applied: sort + dedup).
//! 3. **Typed** — first `(label, pattern)` whose anchored `^pattern$`
//!    matches every value → the class pattern wrapped in a non-capturing
//!    group.
//! 4. **Bounded character class** — same class for every char across all
//!    values (digits, word, alnum) with `min`/`max` length bounds.
//! 5. **Unbounded wildcard** — the default fallback.
//!
//! `alt_min` and `mdl_cost` are exposed as standalone helpers so the
//! Julia side can canonicalise alternations or score a candidate
//! pattern without going through the full ladder. `verify_pattern`
//! (anchored) and `RegexSet` compilation are there so the pipeline can
//! confirm "this regex still matches every sample" and run a
//! multi-pattern matcher that substitutes for Hyperscan on systems
//! without `libhs`.

use regex::{Regex, RegexSet};
use std::collections::BTreeSet;

// ---------------------------------------------------------------------------
// Ladder
// ---------------------------------------------------------------------------

/// Pick the tightest pattern form that matches every value in
/// `values`. `typed` is a ranked `(label, pattern)` battery — typically
/// the `DEFAULT_LABELS` / `DEFAULT_PATTERNS` from `PreProc.Masking`.
/// `enum_max` caps how many distinct values become an enumerated
/// alternation before we fall through to the typed / bounded / wildcard
/// tiers.
pub fn slot_ladder_pattern(
    values: &[&str],
    typed: &[(&str, &str)],
    enum_max: usize,
    wildcard: &str,
) -> String {
    if values.is_empty() {
        return wildcard.to_string();
    }

    // 1. Exact literal.
    let first = values[0];
    if values.iter().all(|v| *v == first) {
        return regex::escape(first);
    }

    // 2. Enum.
    let distinct: BTreeSet<&str> = values.iter().copied().collect();
    if distinct.len() <= enum_max {
        let parts: Vec<String> = distinct.iter().map(|s| regex::escape(s)).collect();
        return format!("(?:{})", parts.join("|"));
    }

    // 3. Typed.
    for (_label, pat) in typed {
        if let Ok(re) = Regex::new(&format!("^(?:{})$", pat)) {
            if values.iter().all(|v| re.is_match(v)) {
                return format!("(?:{})", pat);
            }
        }
    }

    // 4. Bounded character class.
    if let Some(b) = bounded_shape(values) {
        return b;
    }

    // 5. Unbounded wildcard.
    wildcard.to_string()
}

fn bounded_shape(values: &[&str]) -> Option<String> {
    let mut min_len = usize::MAX;
    let mut max_len = 0usize;
    let mut all_digits = true;
    let mut all_word = true;
    let mut all_alnum = true;
    for v in values {
        let l = v.chars().count();
        if l == 0 {
            return None;
        }
        if l < min_len {
            min_len = l;
        }
        if l > max_len {
            max_len = l;
        }
        for c in v.chars() {
            if !c.is_ascii_digit() {
                all_digits = false;
            }
            if !(c.is_ascii_alphanumeric() || c == '_') {
                all_word = false;
            }
            if !c.is_ascii_alphanumeric() {
                all_alnum = false;
            }
        }
    }
    let class = if all_digits {
        r"\d"
    } else if all_word {
        r"\w"
    } else if all_alnum {
        "[A-Za-z0-9]"
    } else {
        return None;
    };
    if min_len == max_len {
        Some(format!("{class}{{{min_len}}}"))
    } else {
        Some(format!("{class}{{{min_len},{max_len}}}"))
    }
}

// ---------------------------------------------------------------------------
// Alternation minimisation
// ---------------------------------------------------------------------------

/// Build a minimised alternation group from an unordered list of
/// alternatives. Duplicates are dropped, the survivors are sorted
/// lexicographically (for determinism), and each is passed through
/// `regex::escape` so metacharacters are safe. Empty input yields the
/// configured `wildcard`.
pub fn alt_min(alternatives: &[&str], wildcard: &str) -> String {
    if alternatives.is_empty() {
        return wildcard.to_string();
    }
    let set: BTreeSet<&str> = alternatives.iter().copied().collect();
    if set.len() == 1 {
        return regex::escape(set.iter().next().unwrap());
    }
    let parts: Vec<String> = set.iter().map(|s| regex::escape(s)).collect();
    format!("(?:{})", parts.join("|"))
}

// ---------------------------------------------------------------------------
// Verification + MDL cost
// ---------------------------------------------------------------------------

/// Anchored verification: compile `^(?:{pattern})$` and return the
/// `(hits, total)` count over `samples`. A malformed pattern returns
/// `(0, total)` rather than panicking — the caller treats that as a
/// total miss.
pub fn verify_pattern(pattern: &str, samples: &[&str]) -> (usize, usize) {
    let re = match Regex::new(&format!("^(?:{})$", pattern)) {
        Ok(r) => r,
        Err(_) => return (0, samples.len()),
    };
    let hits = samples.iter().filter(|s| re.is_match(s)).count();
    (hits, samples.len())
}

/// Two-part MDL-style cost (bits). Rough but monotone: bigger when
/// the pattern is longer AND when it leaves more per-sample residual
/// to encode. Used by the ladder to prefer "tight literal over fat
/// wildcard" at equal coverage.
///
/// `pattern_bits = len(pattern) · log2(128) ≈ 7 · len(pattern)`.
/// `residual_bits ≈ Σᵢ max(0, len(sampleᵢ) − literal_chars) · 7`.
///
/// Not Kolmogorov; the goal is a cheap, monotone score.
pub fn mdl_cost(pattern: &str, samples: &[&str]) -> f64 {
    const BITS_PER_CHAR: f64 = 7.0;
    let pattern_bits = pattern.chars().count() as f64 * BITS_PER_CHAR;
    let literal_chars = pattern
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == ' ' || *c == '.')
        .count();
    let residual_bits: f64 = samples
        .iter()
        .map(|s| (s.chars().count().saturating_sub(literal_chars)) as f64 * BITS_PER_CHAR)
        .sum();
    pattern_bits + residual_bits
}

// ---------------------------------------------------------------------------
// RegexSet multi-pattern matcher
// ---------------------------------------------------------------------------

/// Opaque handle wrapping a compiled `regex::RegexSet`. Returned by
/// [`regexset_compile`] and consumed by [`regexset_match`]; must be
/// freed with `lc_rs_free_regexset`.
pub struct CompiledRegexSet(pub RegexSet);

/// Compile a list of patterns into a `RegexSet`. Returns `None` if
/// any pattern fails to parse — matches the one-bad-regex-fails-the-set
/// behaviour Hyperscan has.
pub fn regexset_compile(patterns: &[&str]) -> Option<CompiledRegexSet> {
    RegexSet::new(patterns).ok().map(CompiledRegexSet)
}

/// Match `line` against every pattern; return the indices (0-based)
/// of those that matched.
pub fn regexset_match(set: &CompiledRegexSet, line: &str) -> Vec<u32> {
    set.0.matches(line).iter().map(|i| i as u32).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn typed_battery<'a>() -> Vec<(&'a str, &'a str)> {
        vec![
            ("IP", r"\d{1,3}(?:\.\d{1,3}){3}"),
            ("INT", r"\d+"),
        ]
    }

    #[test]
    fn ladder_picks_literal_when_all_equal() {
        let out = slot_ladder_pattern(&["foo", "foo", "foo"], &typed_battery(), 8, ".*?");
        assert_eq!(out, "foo");
    }

    #[test]
    fn ladder_picks_enum_for_small_distinct_set() {
        let out = slot_ladder_pattern(&["GET", "POST", "PUT"], &typed_battery(), 8, ".*?");
        // BTreeSet orders lexicographically → GET, POST, PUT.
        assert_eq!(out, "(?:GET|POST|PUT)");
    }

    #[test]
    fn ladder_escapes_enum_members() {
        let out = slot_ladder_pattern(&["a.b", "c.d"], &typed_battery(), 8, ".*?");
        assert_eq!(out, r"(?:a\.b|c\.d)");
    }

    #[test]
    fn ladder_falls_through_to_typed_when_enum_too_large() {
        // 10 distinct IPs, enum_max = 4 → typed match.
        let ips: Vec<String> = (1..=10).map(|i| format!("10.0.0.{i}")).collect();
        let refs: Vec<&str> = ips.iter().map(|s| s.as_str()).collect();
        let out = slot_ladder_pattern(&refs, &typed_battery(), 4, ".*?");
        assert_eq!(out, r"(?:\d{1,3}(?:\.\d{1,3}){3})");
    }

    #[test]
    fn ladder_bounded_shape_all_digits_same_length() {
        // Enum maxed out, no typed matches → bounded digit class.
        let vals: Vec<String> = (1000..=1020).map(|i| i.to_string()).collect();
        let refs: Vec<&str> = vals.iter().map(|s| s.as_str()).collect();
        // Pass an empty typed battery so the ladder can't pick INT first.
        let out = slot_ladder_pattern(&refs, &[], 4, ".*?");
        assert_eq!(out, r"\d{4}");
    }

    #[test]
    fn ladder_bounded_shape_digits_length_range() {
        let vals = ["1", "12", "123", "1234"];
        let out = slot_ladder_pattern(&vals, &[], 2, ".*?");
        assert_eq!(out, r"\d{1,4}");
    }

    #[test]
    fn ladder_falls_through_to_wildcard_on_mixed_shapes() {
        let vals = ["hello", "!@#", "42"];
        let out = slot_ladder_pattern(&vals, &[], 2, ".*?");
        assert_eq!(out, ".*?");
    }

    #[test]
    fn ladder_empty_input_is_wildcard() {
        let out = slot_ladder_pattern(&[], &typed_battery(), 4, "<*>");
        assert_eq!(out, "<*>");
    }

    #[test]
    fn alt_min_dedups_and_sorts() {
        assert_eq!(alt_min(&["c", "a", "b", "a"], ".*?"), "(?:a|b|c)");
    }

    #[test]
    fn alt_min_single_unique_is_just_escaped_literal() {
        assert_eq!(alt_min(&["x.y", "x.y"], ".*?"), r"x\.y");
    }

    #[test]
    fn alt_min_empty_is_wildcard() {
        assert_eq!(alt_min(&[], ".*?"), ".*?");
    }

    #[test]
    fn verify_anchored_match_counts() {
        let samples = ["10.0.0.1", "10.0.0.2", "not an ip"];
        let (hits, total) = verify_pattern(r"\d{1,3}(?:\.\d{1,3}){3}", &samples);
        assert_eq!(hits, 2);
        assert_eq!(total, 3);
    }

    #[test]
    fn verify_malformed_pattern_is_total_miss() {
        let (hits, total) = verify_pattern("[unclosed", &["anything"]);
        assert_eq!(hits, 0);
        assert_eq!(total, 1);
    }

    #[test]
    fn mdl_cost_rewards_tighter_pattern() {
        let samples = ["hello world", "hello world"];
        let tight = mdl_cost("hello world", &samples);
        let loose = mdl_cost(".*?", &samples);
        // A tight literal has no residual; the wildcard does.
        assert!(tight < loose, "tight={tight} loose={loose}");
    }

    #[test]
    fn regexset_compile_and_match() {
        let set = regexset_compile(&[r"^\d+$", r"^[a-z]+$", r"^[A-Z]+$"]).unwrap();
        assert_eq!(regexset_match(&set, "42"), vec![0]);
        assert_eq!(regexset_match(&set, "hello"), vec![1]);
        assert_eq!(regexset_match(&set, "HELLO"), vec![2]);
        assert_eq!(regexset_match(&set, "Hello1"), Vec::<u32>::new());
    }

    #[test]
    fn regexset_compile_rejects_bad_pattern() {
        assert!(regexset_compile(&["[unclosed"]).is_none());
    }
}
