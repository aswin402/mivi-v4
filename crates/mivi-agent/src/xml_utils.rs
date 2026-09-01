//! Single-pass XML escaping utilities for tool formatting.

pub const XML_ESCAPE_CAPACITY_PAD: usize = 16;

#[inline]
fn escape_xml_common(s: &str, escape_quotes: bool) -> String {
    let mut out = String::with_capacity(s.len() + XML_ESCAPE_CAPACITY_PAD);
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' if escape_quotes => out.push_str("&quot;"),
            '\'' if escape_quotes => out.push_str("&apos;"),
            c if c.is_control() && c != '\n' && c != '\r' && c != '\t' => {}
            other => out.push(other),
        }
    }
    out
}

/// Escape attribute value characters (&, <, >, ", ').
pub fn escape_xml_attr(s: &str) -> String {
    escape_xml_common(s, true)
}

/// Escape text content characters (&, <, >).
pub fn escape_xml_content(s: &str) -> String {
    escape_xml_common(s, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xml_escaping() {
        assert_eq!(
            escape_xml_attr("<test & \" ' >"),
            "&lt;test &amp; &quot; &apos; &gt;"
        );
        assert_eq!(
            escape_xml_content("<test & \" ' >"),
            "&lt;test &amp; \" ' &gt;"
        );
    }

    #[test]
    fn test_xml_control_char_filtering() {
        let dirty = "Hello\x00\x08\x1FWorld\n\t!";
        assert_eq!(escape_xml_content(dirty), "HelloWorld\n\t!");
    }
}
