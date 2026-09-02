//! Command output, compiler log, and diff minifier.
//!
//! Inspired by Headroom and RTK (Rust Token Killer).
//! Suppresses noisy passing test cases, download progress bars, and compiler progress lines
//! while preserving failing assertion traces, panic line numbers, error messages, and diff markers.

/// Minifies compiler and test runner output (e.g. `cargo test`, `pytest`, `npm test`).
pub fn minify_test_output(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() / 3);
    let mut in_failure_block = false;

    for line in raw.lines() {
        let trimmed = line.trim();

        // Detect test failures or error headers
        if trimmed.starts_with("failures:")
            || trimmed.starts_with("---- ")
            || trimmed.starts_with("FAIL ")
            || trimmed.starts_with("FAILED")
            || trimmed.contains("FAILED")
        {
            in_failure_block = true;
        }

        // Suppress passing test noise and compilation progress
        if !in_failure_block {
            if trimmed.starts_with("test ") && trimmed.ends_with("... ok") {
                continue;
            }
            if trimmed.starts_with("Compiling ")
                || trimmed.starts_with("Downloaded ")
                || trimmed.starts_with("Updating ")
                || trimmed.starts_with("Fetching ")
            {
                continue;
            }
        }

        // Detect end of failure block / test result line
        if trimmed.starts_with("test result:") {
            in_failure_block = false;
        }

        out.push_str(line);
        out.push('\n');
    }

    out
}

/// Minifies unified git diffs by collapsing large unchanged context blocks.
pub fn minify_git_diff(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut unchanged_line_count = 0;

    for line in raw.lines() {
        if line.starts_with(' ') {
            unchanged_line_count += 1;
            if unchanged_line_count <= 2 {
                out.push_str(line);
                out.push('\n');
            } else if unchanged_line_count == 3 {
                out.push_str("  [... context collapsed ...]\n");
            }
        } else {
            unchanged_line_count = 0;
            out.push_str(line);
            out.push('\n');
        }
    }

    out
}

/// Minifies large homogenous JSON arrays into representative schema headers + samples.
pub fn minify_json_payload(raw: &str, max_array_items: usize) -> String {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(raw) {
        if let Some(arr) = val.as_array() {
            if arr.len() > max_array_items {
                let sample: Vec<&serde_json::Value> = arr.iter().take(max_array_items).collect();
                let summary = serde_json::json!({
                    "_total_count": arr.len(),
                    "_truncated_sample": sample,
                    "_note": format!("Showing first {} of {} items", max_array_items, arr.len())
                });
                return serde_json::to_string_pretty(&summary).unwrap_or_else(|_| raw.to_string());
            }
        }
    }
    raw.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minify_test_output_suppresses_passing_tests() {
        let raw = r#"
   Compiling mivi-core v0.2.11
   Compiling mivi-model v0.2.11
    Finished `test` profile in 1.2s
     Running unittests src/lib.rs

running 4 tests
test math::tests::test_rms_norm ... ok
test math::tests::test_silu ... ok
test math::tests::test_softmax ... ok
test cache::tests::test_overflow ... FAILED

failures:

---- cache::tests::test_overflow stdout ----
thread 'cache::tests::test_overflow' panicked at src/cache.rs:42:
assertion failed: capacity == 64

failures:
    cache::tests::test_overflow

test result: FAILED. 3 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out
"#;
        let minified = minify_test_output(raw);
        assert!(!minified.contains("test math::tests::test_rms_norm ... ok"));
        assert!(!minified.contains("Compiling mivi-core"));
        assert!(minified.contains("test cache::tests::test_overflow ... FAILED"));
        assert!(minified.contains("assertion failed: capacity == 64"));
        assert!(minified.contains("test result: FAILED. 3 passed; 1 failed"));
    }

    #[test]
    fn test_minify_json_payload_large_array() {
        let items: Vec<serde_json::Value> = (0..100)
            .map(|i| serde_json::json!({ "id": i, "name": format!("item_{i}") }))
            .collect();
        let raw = serde_json::to_string(&items).unwrap();

        let minified = minify_json_payload(&raw, 3);
        assert!(minified.contains("_total_count\": 100"));
        assert!(minified.contains("Showing first 3 of 100 items"));
    }
}
