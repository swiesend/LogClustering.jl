//! Log-key regex inference — port of thesis Algorithm 3.10.
//!
//! Given a list of tokenised samples (each a `Vec<String>`), emit a regex
//! that matches every sample exactly. Agreeing tokens become literals;
//! disagreeing tokens at the same position become alternations. Tokens
//! containing any of the configured *replacement* substrings are rewritten
//! to the wildcard fragment so descriptive placeholders like
//! `%RCE_DATETIME%` collapse to `.*?`.
//!
//! Stage E″ extension: if the caller supplies a `label → class regex`
//! map, a `%LABEL%` token that exactly matches one of the keys is
//! rewritten to the *class regex* (e.g. `IP → \d{1,3}(?:\.\d{1,3}){3}`)
//! instead of the default wildcard. This tightens the inferred regex
//! without changing the grouping semantics of Algorithm 3.10.
//!
//! `align = true` (the `infer_regex_aligned` variant) turns on
//! anti-unification — samples of different lengths are first aligned
//! pairwise via LCS, runs of unmatched tokens collapse to a single
//! wildcard, and the resulting template is then emitted as a regex.
//! This is the Stage E″ "anti-unification over aligned tokens" step
//! from plan 001.

use regex::Regex;
use std::collections::{BTreeMap, HashMap};

const LABEL_PATTERN: &str = r"%[0-9A-Z_]*?%";
const EXACT_LABEL: &str = r"^%([0-9A-Z_]*?)%$";

/// Decode a packed samples buffer; see `lib.rs` for the wire format.
fn read_u32(buf: &[u8], cursor: &mut usize) -> Option<usize> {
    if *cursor + 4 > buf.len() {
        return None;
    }
    let n = u32::from_le_bytes([
        buf[*cursor],
        buf[*cursor + 1],
        buf[*cursor + 2],
        buf[*cursor + 3],
    ]) as usize;
    *cursor += 4;
    Some(n)
}

fn read_str<'a>(buf: &'a [u8], cursor: &mut usize) -> Option<&'a str> {
    let len = read_u32(buf, cursor)?;
    if *cursor + len > buf.len() {
        return None;
    }
    let s = std::str::from_utf8(&buf[*cursor..*cursor + len]).ok()?;
    *cursor += len;
    Some(s)
}

fn decode_samples(buf: &[u8]) -> Option<Vec<Vec<&str>>> {
    let mut cursor = 0;
    let num = read_u32(buf, &mut cursor)?;
    let mut out = Vec::with_capacity(num);
    for _ in 0..num {
        let wc = read_u32(buf, &mut cursor)?;
        let mut s = Vec::with_capacity(wc);
        for _ in 0..wc {
            s.push(read_str(buf, &mut cursor)?);
        }
        out.push(s);
    }
    if cursor != buf.len() {
        return None;
    }
    Some(out)
}

fn decode_token_list(buf: &[u8]) -> Option<Vec<&str>> {
    let mut cursor = 0;
    let num = read_u32(buf, &mut cursor)?;
    let mut out = Vec::with_capacity(num);
    for _ in 0..num {
        out.push(read_str(buf, &mut cursor)?);
    }
    if cursor != buf.len() {
        return None;
    }
    Some(out)
}

fn decode_classes(buf: &[u8]) -> Option<HashMap<&str, &str>> {
    if buf.is_empty() {
        return Some(HashMap::new());
    }
    let mut cursor = 0;
    let num = read_u32(buf, &mut cursor)?;
    let mut out = HashMap::with_capacity(num);
    for _ in 0..num {
        let k = read_str(buf, &mut cursor)?;
        let v = read_str(buf, &mut cursor)?;
        out.insert(k, v);
    }
    if cursor != buf.len() {
        return None;
    }
    Some(out)
}

pub fn infer_regex(
    samples_buf: &[u8],
    replacements_buf: &[u8],
    wildcard: &str,
    classes_buf: &[u8],
) -> Option<String> {
    let samples = decode_samples(samples_buf)?;
    let replacements = if replacements_buf.is_empty() {
        Vec::new()
    } else {
        decode_token_list(replacements_buf)?
    };
    let classes = decode_classes(classes_buf)?;
    Some(infer_regex_from(
        &samples,
        &replacements,
        wildcard,
        &classes,
    ))
}

pub fn infer_regex_aligned(
    samples_buf: &[u8],
    replacements_buf: &[u8],
    wildcard: &str,
    classes_buf: &[u8],
) -> Option<String> {
    let samples = decode_samples(samples_buf)?;
    let replacements = if replacements_buf.is_empty() {
        Vec::new()
    } else {
        decode_token_list(replacements_buf)?
    };
    let classes = decode_classes(classes_buf)?;
    Some(anti_unify_regex(
        &samples,
        &replacements,
        wildcard,
        &classes,
    ))
}

/// Core algorithm; pub for unit tests.
pub fn infer_regex_from(
    samples: &[Vec<&str>],
    replacements: &[&str],
    wildcard: &str,
    classes: &HashMap<&str, &str>,
) -> String {
    let label_re = Regex::new(LABEL_PATTERN).expect("static label regex compiles");
    let exact_re = Regex::new(EXACT_LABEL).expect("static exact label regex compiles");

    let max_len = samples.iter().map(|s| s.len()).max().unwrap_or(0);

    let mut groups: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for sample in samples {
        for (w, word) in sample.iter().enumerate() {
            let fragment = encode_word(word, replacements, &label_re, &exact_re, wildcard, classes);
            groups.entry(w).or_default().push(fragment);
        }
    }
    for v in groups.values_mut() {
        dedup_preserving_order(v);
    }

    let mut out = String::new();
    let mut last_emitted: Option<String> = None;
    for w in 0..max_len {
        let Some(group) = groups.get(&w) else {
            continue;
        };
        let fragment = if group.len() > 1 {
            let mut s = String::with_capacity(2 + group.iter().map(|g| g.len() + 1).sum::<usize>());
            s.push('(');
            for (i, g) in group.iter().enumerate() {
                if i > 0 {
                    s.push('|');
                }
                s.push_str(g);
            }
            s.push(')');
            s
        } else if let Some(g) = group.first() {
            g.clone()
        } else {
            continue;
        };
        // Collapse runs of identical wildcards: two adjacent `.*?` match
        // the same language as one, and they dominate the heavy-looking
        // output when several typed slots stand side by side.
        if Some(&fragment) == last_emitted.as_ref() && is_wildcardish(&fragment, wildcard) {
            continue;
        }
        out.push_str(&fragment);
        last_emitted = Some(fragment);
    }
    out.trim().to_string()
}

// ---------------------------------------------------------------------------
// Anti-unification (Stage E″, step 1 of the MDL ladder)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
enum Fragment {
    Literal(String),
    Wildcard,
}

/// Fold the N samples into a single sketch via pairwise LCS. Each pass
/// aligns the running template against the next sample; runs of
/// unmatched tokens on either side collapse to one `Wildcard`.
fn anti_unify(samples: &[Vec<&str>]) -> Vec<Fragment> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut sketch: Vec<Fragment> = samples[0]
        .iter()
        .map(|t| Fragment::Literal((*t).to_string()))
        .collect();
    for sample in &samples[1..] {
        sketch = align_and_merge(&sketch, sample);
    }
    collapse_wildcards(sketch)
}

fn anti_unify_regex(
    samples: &[Vec<&str>],
    replacements: &[&str],
    wildcard: &str,
    classes: &HashMap<&str, &str>,
) -> String {
    let label_re = Regex::new(LABEL_PATTERN).expect("static label regex compiles");
    let exact_re = Regex::new(EXACT_LABEL).expect("static exact label regex compiles");
    let sketch = anti_unify(samples);
    let mut out = String::new();
    for f in &sketch {
        match f {
            Fragment::Literal(s) => out.push_str(&encode_word(
                s,
                replacements,
                &label_re,
                &exact_re,
                wildcard,
                classes,
            )),
            Fragment::Wildcard => out.push_str(wildcard),
        }
    }
    out.trim().to_string()
}

/// LCS-align `template` against `sample`; emit a new fragment list
/// where template literals that match a sample position are kept
/// verbatim, Wildcard slots are kept, and every run of unmatched
/// literals (on either side) becomes one Wildcard.
fn align_and_merge(template: &[Fragment], sample: &[&str]) -> Vec<Fragment> {
    let pairs = lcs_pairs(template, sample);
    let mut out = Vec::with_capacity(template.len() + sample.len());
    let mut ti = 0usize;
    let mut si = 0usize;
    for &(tp, sp) in &pairs {
        if tp > ti || sp > si {
            out.push(Fragment::Wildcard);
        }
        out.push(template[tp].clone());
        ti = tp + 1;
        si = sp + 1;
    }
    if ti < template.len() || si < sample.len() {
        out.push(Fragment::Wildcard);
    }
    out
}

/// Longest common subsequence of template fragments and sample tokens.
/// A Wildcard fragment matches *any* sample token, but with a lower
/// weight (1) than a literal match (2) — so when a literal at position
/// `i` could match the same sample token as a wildcard at position
/// `j`, the alignment keeps the literal and lets the wildcard
/// straddle the unmatched region. Without this weighting the LCS
/// would consume leading literals against downstream wildcards.
fn lcs_pairs(template: &[Fragment], sample: &[&str]) -> Vec<(usize, usize)> {
    let n = template.len();
    let m = sample.len();
    if n == 0 || m == 0 {
        return Vec::new();
    }
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..n {
        for j in 0..m {
            let w = match_weight(&template[i], sample[j]);
            let diag = if w > 0 { dp[i][j] + w } else { 0 };
            dp[i + 1][j + 1] = diag.max(dp[i + 1][j]).max(dp[i][j + 1]);
        }
    }
    let mut i = n;
    let mut j = m;
    let mut out = Vec::new();
    while i > 0 && j > 0 {
        let w = match_weight(&template[i - 1], sample[j - 1]);
        if w > 0 && dp[i][j] == dp[i - 1][j - 1] + w {
            out.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] >= dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    out.reverse();
    out
}

#[inline]
fn match_weight(f: &Fragment, token: &str) -> usize {
    match f {
        Fragment::Literal(s) if s == token => 2,
        Fragment::Wildcard => 1,
        _ => 0,
    }
}

fn collapse_wildcards(frags: Vec<Fragment>) -> Vec<Fragment> {
    let mut out: Vec<Fragment> = Vec::with_capacity(frags.len());
    for f in frags {
        if matches!(f, Fragment::Wildcard) && matches!(out.last(), Some(Fragment::Wildcard)) {
            continue;
        }
        out.push(f);
    }
    out
}

fn encode_word(
    word: &str,
    replacements: &[&str],
    label_re: &Regex,
    exact_re: &Regex,
    wildcard: &str,
    classes: &HashMap<&str, &str>,
) -> String {
    // Exact-match label path: if the whole token is `%LABEL%` and the
    // caller supplied a tight class regex for LABEL, use that.
    if !classes.is_empty() {
        if let Some(cap) = exact_re.captures(word) {
            let name = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            if let Some(cls) = classes.get(name) {
                return (*cls).to_string();
            }
        }
    }

    if !replacements.is_empty() && replacements.iter().any(|r| word.contains(r)) {
        let escaped = regex::escape(word);
        label_re.replace_all(&escaped, wildcard).into_owned()
    } else {
        regex::escape(word)
    }
}

/// A fragment is "wildcardish" for coalescing purposes if it equals the
/// configured wildcard, or is a bare alternation of copies of the
/// wildcard. Literal-bearing fragments must not be coalesced.
fn is_wildcardish(fragment: &str, wildcard: &str) -> bool {
    if fragment == wildcard {
        return true;
    }
    let trimmed = fragment.strip_prefix('(').and_then(|s| s.strip_suffix(')'));
    match trimmed {
        Some(inner) => inner.split('|').all(|p| p == wildcard),
        None => false,
    }
}

fn dedup_preserving_order(v: &mut Vec<String>) {
    let mut seen: Vec<String> = Vec::with_capacity(v.len());
    v.retain(|x| {
        if seen.iter().any(|s| s == x) {
            false
        } else {
            seen.push(x.clone());
            true
        }
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn no_classes<'a>() -> HashMap<&'a str, &'a str> {
        HashMap::new()
    }

    #[test]
    fn thesis_example_3_1() {
        let s1 = vec![
            "%RCE_DATETIME%",
            " ",
            "DEBUG",
            " ",
            "-",
            " ",
            "de",
            ".",
            "rcenvironment",
            ".",
            "core",
            ".",
            "communication",
            ".",
            "transport",
            ".",
            "jms",
            ".",
            "activemq",
        ];
        let s2 = vec![
            "%RCE_DATETIME%",
            " ",
            "DEBUG",
            " ",
            "-",
            " ",
            "de",
            ".",
            "rcenvironment",
            ".",
            "core",
            ".",
            "communication",
            ".",
            "transport",
            ".",
            "jms",
            ".",
            "common",
        ];
        let samples = vec![s1, s2];
        let replacements = vec!["%"];
        let out = infer_regex_from(&samples, &replacements, ".*?", &no_classes());
        assert_eq!(
            out,
            r".*? DEBUG \- de\.rcenvironment\.core\.communication\.transport\.jms\.(activemq|common)"
        );
    }

    #[test]
    fn identical_samples() {
        let s = vec!["foo", " ", "bar"];
        let samples = vec![s.clone(), s];
        let out = infer_regex_from(&samples, &[], ".*?", &no_classes());
        assert_eq!(out, r"foo bar");
    }

    #[test]
    fn disjoint_alternatives_at_multiple_positions() {
        let a = vec!["a", "x"];
        let b = vec!["a", "y"];
        let c = vec!["b", "x"];
        let samples = vec![a, b, c];
        let out = infer_regex_from(&samples, &[], ".*?", &no_classes());
        assert_eq!(out, r"(a|b)(x|y)");
    }

    #[test]
    fn adjacent_wildcards_coalesce() {
        // Four consecutive labels — historically this produced `.*?.*?.*?.*?`.
        let s = vec!["%A%", "%B%", "%C%", "%D%"];
        let samples = vec![s.clone(), s];
        let replacements = vec!["%"];
        let out = infer_regex_from(&samples, &replacements, ".*?", &no_classes());
        assert_eq!(out, ".*?");
    }

    #[test]
    fn class_map_tightens_labels() {
        let s = vec!["%IP%", " ", "done"];
        let samples = vec![s.clone(), s];
        let replacements = vec!["%"];
        let mut classes = HashMap::new();
        classes.insert("IP", r"\d{1,3}(?:\.\d{1,3}){3}");
        let out = infer_regex_from(&samples, &replacements, ".*?", &classes);
        assert_eq!(out, r"\d{1,3}(?:\.\d{1,3}){3} done");
    }

    // ---- anti-unification --------------------------------------------------

    #[test]
    fn anti_unify_identical_samples_returns_literals() {
        let s = vec!["foo", " ", "bar"];
        let samples = vec![s.clone(), s];
        let frags = anti_unify(&samples);
        assert_eq!(
            frags,
            vec![
                Fragment::Literal("foo".into()),
                Fragment::Literal(" ".into()),
                Fragment::Literal("bar".into()),
            ]
        );
    }

    #[test]
    fn anti_unify_length_differ_by_suffix() {
        // samples: [a, b, c]  vs  [a, b, c, d]
        // LCS keeps a, b, c; the trailing d becomes a wildcard.
        let samples = vec![vec!["a", "b", "c"], vec!["a", "b", "c", "d"]];
        let out = anti_unify_regex(&samples, &[], ".*?", &no_classes());
        assert_eq!(out, "abc.*?");
    }

    #[test]
    fn anti_unify_insertion_in_the_middle() {
        // samples: [a, x, b]  vs  [a, y, z, b]
        //                         [a, b]
        // LCS → a, b literal; everything between collapses to a single wildcard.
        let samples = vec![
            vec!["a", "x", "b"],
            vec!["a", "y", "z", "b"],
            vec!["a", "b"],
        ];
        let out = anti_unify_regex(&samples, &[], ".*?", &no_classes());
        assert_eq!(out, "a.*?b");
    }

    #[test]
    fn anti_unify_no_common_prefix_is_all_wildcard() {
        let samples = vec![vec!["x"], vec!["y"], vec!["z"]];
        let out = anti_unify_regex(&samples, &[], ".*?", &no_classes());
        assert_eq!(out, ".*?");
    }
}
