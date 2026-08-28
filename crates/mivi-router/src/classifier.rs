use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

static RE_DEBUG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(fix|bug|error|failing|panic|crash|traceback)\b").unwrap());
static RE_RESEARCH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(search|latest|research|find|lookup|browse|paper)\b").unwrap());
static RE_TEST: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(test|tests|testing|assert|benchmark|suite)\b").unwrap());
static RE_CODE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(code|function|rust|python|impl|struct|class|fn|def)\b").unwrap());

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteDecision {
    pub primary: TaskFamily,
    pub secondary: Option<CodeSpecialization>,
    pub confidence: f32,
}

pub struct IntentClassifier;

impl IntentClassifier {
    /// Fast heuristic intent classification (Level 1 + Level 2 routing) with word boundary precision
    pub fn classify(prompt: &str) -> RouteDecision {
        let p = prompt.to_lowercase();

        if RE_DEBUG.is_match(&p) {
            RouteDecision {
                primary: TaskFamily::Debug,
                secondary: Some(CodeSpecialization::Debugging),
                confidence: 0.9,
            }
        } else if RE_RESEARCH.is_match(&p) {
            RouteDecision {
                primary: TaskFamily::Research,
                secondary: None,
                confidence: 0.85,
            }
        } else if RE_TEST.is_match(&p) {
            RouteDecision {
                primary: TaskFamily::Code,
                secondary: Some(CodeSpecialization::Testing),
                confidence: 0.85,
            }
        } else if RE_CODE.is_match(&p) {
            RouteDecision {
                primary: TaskFamily::Code,
                secondary: Some(CodeSpecialization::Implementation),
                confidence: 0.8,
            }
        } else if p.len() < 20 {
            RouteDecision {
                primary: TaskFamily::Chat,
                secondary: None,
                confidence: 0.95,
            }
        } else {
            RouteDecision {
                primary: TaskFamily::Agent,
                secondary: None,
                confidence: 0.75,
            }
        }
    }
}
