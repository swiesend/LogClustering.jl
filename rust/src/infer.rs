//! Log-key regex inference — port of thesis Algorithm 3.10.
//!
//! Given a list of tokenised samples (each a `Vec<String>`), emit a regex
//! that matches every sample exactly. Agreeing tokens become literals;
//! disagreeing tokens at the same position become alternations. Tokens
//! containing any of the configured *replacement* substrings are rewritten
//! to the wildcard fragment so descriptive placeholders like
//! `%RCE_DATETIME%` collapse to `.*?`.
//!
//! Thesis pseudocode:
//!
//! ```text
//! function infer(samples; replacements, label=r"%[0-9A-Z_]*?%", asterix=".*?")
//!   for each word position w in range [1, max |sample|]
//!     groups[w] = OrderedSet
//!     for each sample s:
//!       word = s[w]
//!       if any replacement is a substring of word:
//!         groups[w] += replace(escape(word), label, asterix)
//!       else:
//!         groups[w] += escape(word)
//!     end
//!   end
//!   joins[w] = if |groups[w]| > 1 then "(a|b|c)" else "a"
//!   return join(joins)
//! ```

use regex::Regex;
use std::collections::BTreeMap;

const LABEL_PATTERN: &str = r"%[0-9A-Z_]*?%";

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

pub fn infer_regex(samples_buf: &[u8], replacements_buf: &[u8], wildcard: &str) -> Option<String> {
    let samples = decode_samples(samples_buf)?;
    let replacements = if replacements_buf.is_empty() {
        Vec::new()
    } else {
        decode_token_list(replacements_buf)?
    };
    Some(infer_regex_from(&samples, &replacements, wildcard))
}

/// Core algorithm; pub for unit tests.
pub fn infer_regex_from(samples: &[Vec<&str>], replacements: &[&str], wildcard: &str) -> String {
    let label_re = Regex::new(LABEL_PATTERN).expect("static label regex compiles");

    let max_len = samples.iter().map(|s| s.len()).max().unwrap_or(0);

    // Ordered set semantics: preserve first-seen insertion order while
    // de-duplicating. A small Vec is faster than a HashSet for the typical
    // cluster size (<= a few dozen distinct tokens per position).
    let mut groups: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for sample in samples {
        for (w, word) in sample.iter().enumerate() {
            let fragment = encode_word(word, replacements, &label_re, wildcard);
            groups.entry(w).or_default().push(fragment);
        }
    }
    for v in groups.values_mut() {
        dedup_preserving_order(v);
    }

    let mut out = String::new();
    for w in 0..max_len {
        let Some(group) = groups.get(&w) else {
            continue;
        };
        if group.len() > 1 {
            out.push('(');
            for (i, g) in group.iter().enumerate() {
                if i > 0 {
                    out.push('|');
                }
                out.push_str(g);
            }
            out.push(')');
        } else if let Some(g) = group.first() {
            out.push_str(g);
        }
    }
    out.trim().to_string()
}

fn encode_word(word: &str, replacements: &[&str], label_re: &Regex, wildcard: &str) -> String {
    if !replacements.is_empty() && replacements.iter().any(|r| word.contains(r)) {
        let escaped = regex::escape(word);
        label_re.replace_all(&escaped, wildcard).into_owned()
    } else {
        regex::escape(word)
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

    #[test]
    fn thesis_example_3_1() {
        // Given the thesis Beispiel 3.1, expect the documented regex.
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
        let out = infer_regex_from(&samples, &replacements, ".*?");
        assert_eq!(
            out,
            r".*? DEBUG \- de\.rcenvironment\.core\.communication\.transport\.jms\.(activemq|common)"
        );
    }

    #[test]
    fn identical_samples() {
        let s = vec!["foo", " ", "bar"];
        let samples = vec![s.clone(), s];
        let out = infer_regex_from(&samples, &[], ".*?");
        assert_eq!(out, r"foo bar");
    }

    #[test]
    fn disjoint_alternatives_at_multiple_positions() {
        let a = vec!["a", "x"];
        let b = vec!["a", "y"];
        let c = vec!["b", "x"];
        let samples = vec![a, b, c];
        let out = infer_regex_from(&samples, &[], ".*?");
        // Position 0: {a, b} → (a|b); position 1: {x, y} → (x|y).
        assert_eq!(out, r"(a|b)(x|y)");
    }
}
