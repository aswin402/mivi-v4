//! Grammar and JSON Schema constrained decoding tracker with syntax error detection.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { schema: serde_json::Value },
}

/// Lightweight stateful JSON syntax validator for token-by-token generation.
#[derive(Debug, Default, Clone)]
pub struct JsonConstraintState {
    pub in_string: bool,
    pub escape: bool,
    pub brace_depth: usize,
    pub bracket_depth: usize,
    pub started: bool,
    pub completed: bool,
    pub has_error: bool,
}

impl JsonConstraintState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance parser state with a decoded token string chunk.
    pub fn feed(&mut self, chunk: &str) -> bool {
        if self.has_error {
            return false;
        }

        for ch in chunk.chars() {
            if self.escape {
                self.escape = false;
                continue;
            }

            if ch == '\\' && self.in_string {
                self.escape = true;
                continue;
            }

            if ch == '"' {
                self.in_string = !self.in_string;
                continue;
            }

            if !self.in_string {
                match ch {
                    '{' => {
                        self.brace_depth += 1;
                        self.started = true;
                    }
                    '}' => {
                        if self.brace_depth > 0 {
                            self.brace_depth -= 1;
                        } else {
                            self.has_error = true;
                            return false;
                        }
                    }
                    '[' => {
                        self.bracket_depth += 1;
                        self.started = true;
                    }
                    ']' => {
                        if self.bracket_depth > 0 {
                            self.bracket_depth -= 1;
                        } else {
                            self.has_error = true;
                            return false;
                        }
                    }
                    _ => {}
                }

                if self.started && self.brace_depth == 0 && self.bracket_depth == 0 {
                    self.completed = true;
                }
            }
        }

        self.is_valid()
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        !self.has_error
    }

    #[inline]
    pub fn is_complete(&self) -> bool {
        !self.has_error
            && self.started
            && self.completed
            && self.brace_depth == 0
            && self.bracket_depth == 0
            && !self.in_string
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_constraint_tracker() {
        let mut tracker = JsonConstraintState::new();
        assert!(tracker.feed(r#"{"name": "#));
        assert!(!tracker.is_complete());
        assert!(tracker.feed(r#""calculator", "#));
        assert!(!tracker.is_complete());
        assert!(tracker.feed(r#""arguments": {"expr": "1+1"}}"#));
        assert!(tracker.is_complete());
        assert!(tracker.is_valid());

        // Error detection on unmatched closing bracket
        let mut broken_tracker = JsonConstraintState::new();
        assert!(!broken_tracker.feed("}"));
        assert!(!broken_tracker.is_valid());
    }
}
