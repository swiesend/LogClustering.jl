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
}
