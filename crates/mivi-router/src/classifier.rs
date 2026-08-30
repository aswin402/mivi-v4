use regex::RegexSet;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

const ROUTER_PATTERNS: &[&str] = &[
    r"(?i)\b(fix|bug|error|failing|panic|crash|traceback)\b", // 0: Debug
    r"(?i)\b(search|latest|research|find|lookup|browse|paper)\b", // 1: Research
    r"(?i)\b(test|tests|testing|assert|benchmark|suite)\b",   // 2: Test
    r"(?i)\b(code|function|rust|python|impl|struct|class|fn|def)\b", // 3: Code
];

static ROUTER_SET: LazyLock<RegexSet> =
    LazyLock::new(|| RegexSet::new(ROUTER_PATTERNS).expect("Valid regex patterns for task router"));

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskFamily {
    Chat,
    Agent,
    Code,
    Debug,
    Research,
    General,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeSpecialization {
    Implementation,
    Debugging,
    Testing,
    Frontend,
    Backend,
    Architecture,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteDecision {
    pub primary: TaskFamily,
    pub secondary: Option<CodeSpecialization>,
    pub confidence: f32,
}

pub const CONFIDENCE_DEBUG: f32 = 0.90;
pub const CONFIDENCE_RESEARCH: f32 = 0.85;
pub const CONFIDENCE_TEST: f32 = 0.85;
pub const CONFIDENCE_CODE: f32 = 0.80;
pub const CONFIDENCE_SHORT_CHAT: f32 = 0.95;
pub const CONFIDENCE_DEFAULT_AGENT: f32 = 0.75;
pub const SHORT_PROMPT_CHAR_LIMIT: usize = 20;

pub struct IntentClassifier;

impl IntentClassifier {
    /// Fast single-pass heuristic intent classification (Level 1 + Level 2 routing) using RegexSet
    pub fn classify(prompt: &str) -> RouteDecision {
        let matches = ROUTER_SET.matches(prompt);
        if matches.matched(0) {
            RouteDecision {
                primary: TaskFamily::Debug,
                secondary: Some(CodeSpecialization::Debugging),
                confidence: CONFIDENCE_DEBUG,
            }
        } else if matches.matched(1) {
            RouteDecision {
                primary: TaskFamily::Research,
                secondary: None,
                confidence: CONFIDENCE_RESEARCH,
            }
        } else if matches.matched(2) {
            RouteDecision {
                primary: TaskFamily::Code,
                secondary: Some(CodeSpecialization::Testing),
                confidence: CONFIDENCE_TEST,
            }
        } else if matches.matched(3) {
            RouteDecision {
                primary: TaskFamily::Code,
                secondary: Some(CodeSpecialization::Implementation),
                confidence: CONFIDENCE_CODE,
            }
        } else if prompt.len() < SHORT_PROMPT_CHAR_LIMIT {
            RouteDecision {
                primary: TaskFamily::Chat,
                secondary: None,
                confidence: CONFIDENCE_SHORT_CHAT,
            }
        } else {
            RouteDecision {
                primary: TaskFamily::Agent,
                secondary: None,
                confidence: CONFIDENCE_DEFAULT_AGENT,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_debug() {
        let decision = IntentClassifier::classify("Please fix this bug in my code");
        assert_eq!(decision.primary, TaskFamily::Debug);
        assert_eq!(decision.secondary, Some(CodeSpecialization::Debugging));
    }

    #[test]
    fn test_classifier_research() {
        let decision = IntentClassifier::classify("search the latest papers on transformers");
        assert_eq!(decision.primary, TaskFamily::Research);
    }

    #[test]
    fn test_classifier_test() {
        let decision = IntentClassifier::classify("write a test suite for this module");
        assert_eq!(decision.primary, TaskFamily::Code);
        assert_eq!(decision.secondary, Some(CodeSpecialization::Testing));
    }

    #[test]
    fn test_classifier_short_chat() {
        let decision = IntentClassifier::classify("hello world");
        assert_eq!(decision.primary, TaskFamily::Chat);
    }

    #[test]
    fn test_classifier_default_agent() {
        let decision = IntentClassifier::classify(
            "Please create a comprehensive analysis of the system performance and organize the findings",
        );
        assert_eq!(decision.primary, TaskFamily::Agent);
    }
}
