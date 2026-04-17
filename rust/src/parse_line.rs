//! Recursive regex-cascade parser — port of thesis Algorithm 3.1.
//!
//! `parse_line(line, labels, patterns)` finds every non-overlapping match
//! of `patterns[0]` inside `line`, recurses on the unmatched left and
//! right slices with `patterns[1..]`, and emits an ordered
//! non-overlapping decomposition of `line`. Each span is either a matched
//! label (`label_idx >= 0`) or a raw substring between matches
//! (`label_idx == -1`). Patterns are compiled lazily and anchored via
//! `find_iter`, so complex typed slots (timestamps, IPs, UUIDs) should be
//! listed before simple ones.
//!
//! Compared to the Julia `PreProc/Framing.jl` parser, this one is
//! grammar-driven (ordered regex battery) rather than envelope-driven
//! (first-byte dispatch). They complement each other in the pipeline:
//! framing first, then regex descent on the message body.

use crate::LcSpan;
use regex::Regex;

pub fn parse_line(line: &[u8], _labels: &[&str], patterns: &[&str]) -> Option<Vec<LcSpan>> {
    // Pre-compile every pattern up front; bail on any regex error.
    let compiled: Result<Vec<Regex>, _> = patterns.iter().map(|p| Regex::new(p)).collect();
    let compiled = compiled.ok()?;

    let text = std::str::from_utf8(line).ok()?;
    let mut out = Vec::new();
    descend(text, 0, &compiled, 0, &mut out);
    // The recursion may emit empty raw spans when matches abut; prune.
    out.retain(|s| s.end > s.start || s.label_idx >= 0);
    Some(out)
}

/// Batched form of [`parse_line`]. Compiles the regex battery *once*
/// and runs it over each line in the packed `lines_buf`; returns a
/// flat span list tagged with a `line_idx` so the caller can split
/// back out in O(total_spans).
///
/// `lines_buf` wire format: `u32 num_lines` followed by each line as
/// `u32 len` + `len` UTF-8 bytes.
pub fn parse_lines(
    lines_buf: &[u8],
    _labels: &[&str],
    patterns: &[&str],
) -> Option<(Vec<crate::LcLineSpan>, Vec<u32>)> {
    let compiled: Result<Vec<Regex>, _> = patterns.iter().map(|p| Regex::new(p)).collect();
    let compiled = compiled.ok()?;

    let lines = decode_lines(lines_buf)?;
    let mut spans = Vec::with_capacity(lines.len() * 4);
    // Prefix-sum layout: offsets[i..i+1] brackets the spans for line i.
    let mut offsets = Vec::with_capacity(lines.len() + 1);
    offsets.push(0u32);
    let mut scratch: Vec<LcSpan> = Vec::new();
    for (line_idx, line) in lines.iter().enumerate() {
        scratch.clear();
        descend(line, 0, &compiled, 0, &mut scratch);
        scratch.retain(|s| s.end > s.start || s.label_idx >= 0);
        for s in &scratch {
            spans.push(crate::LcLineSpan {
                line_idx: line_idx as u32,
                start: s.start,
                end: s.end,
                label_idx: s.label_idx,
            });
        }
        offsets.push(spans.len() as u32);
    }
    Some((spans, offsets))
}

/// Parse the `u32 num_lines` + length-prefixed lines wire format into
/// a vector of `&str`s (borrowed from the input buffer).
fn decode_lines(buf: &[u8]) -> Option<Vec<&str>> {
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

fn descend(
    text: &str,
    absolute_offset: u32,
    compiled: &[Regex],
    depth: usize,
    out: &mut Vec<LcSpan>,
) {
    if text.is_empty() {
        return;
    }
    if depth >= compiled.len() {
        out.push(LcSpan {
            start: absolute_offset,
            end: absolute_offset + text.len() as u32,
            label_idx: -1,
        });
        return;
    }
    let re = &compiled[depth];
    let matches: Vec<(usize, usize)> = re.find_iter(text).map(|m| (m.start(), m.end())).collect();
    if matches.is_empty() {
        descend(text, absolute_offset, compiled, depth + 1, out);
        return;
    }
    let mut last_end = 0usize;
    for (ms, me) in &matches {
        if *ms > last_end {
            descend(
                &text[last_end..*ms],
                absolute_offset + last_end as u32,
                compiled,
                depth + 1,
                out,
            );
        }
        out.push(LcSpan {
            start: absolute_offset + *ms as u32,
            end: absolute_offset + *me as u32,
            label_idx: depth as i32,
        });
        last_end = *me;
    }
    if last_end < text.len() {
        descend(
            &text[last_end..],
            absolute_offset + last_end as u32,
            compiled,
            depth + 1,
            out,
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn render(line: &str, spans: &[LcSpan], labels: &[&str]) -> Vec<String> {
        spans
            .iter()
            .map(|s| {
                let slice = &line[s.start as usize..s.end as usize];
                if s.label_idx < 0 {
                    format!("raw({slice:?})")
                } else {
                    let name = labels[s.label_idx as usize];
                    format!("{name}({slice:?})")
                }
            })
            .collect()
    }

    #[test]
    fn decomposes_around_labels_in_rank_order() {
        let line = "2024-01-01T00:00:00Z DEBUG 127.0.0.1 hello";
        let labels = vec!["ts", "ip"];
        let patterns = vec![
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
            r"\d{1,3}(?:\.\d{1,3}){3}",
        ];
        let spans = parse_line(line.as_bytes(), &labels, &patterns).unwrap();
        let rendered = render(line, &spans, &labels);
        assert_eq!(
            rendered,
            vec![
                r#"ts("2024-01-01T00:00:00Z")"#,
                r#"raw(" DEBUG ")"#,
                r#"ip("127.0.0.1")"#,
                r#"raw(" hello")"#,
            ]
        );
    }

    #[test]
    fn empty_line_yields_empty_decomposition() {
        let spans = parse_line(b"", &[], &[]).unwrap();
        assert!(spans.is_empty());
    }

    #[test]
    fn no_patterns_returns_the_whole_line() {
        let line = "just text";
        let spans = parse_line(line.as_bytes(), &[], &[]).unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].label_idx, -1);
        assert_eq!(spans[0].start, 0);
        assert_eq!(spans[0].end, line.len() as u32);
    }
}
