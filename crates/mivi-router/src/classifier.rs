//! Intent classifier and task domain categorization.

use serde::{Deserialize, Serialize};

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
    /// Fast heuristic intent classification (Level 1 + Level 2 routing)
    pub fn classify(prompt: &str) -> RouteDecision {
        let p = prompt.to_lowercase();

        if p.contains("fix") || p.contains("bug") || p.contains("error") || p.contains("failing") {
            RouteDecision {
                primary: TaskFamily::Debug,
                secondary: Some(CodeSpecialization::Debugging),
                confidence: 0.9,
            }
        } else if p.contains("search") || p.contains("latest") || p.contains("research") {
            RouteDecision {
                primary: TaskFamily::Research,
                secondary: None,
                confidence: 0.85,
            }
        } else if p.contains("test") {
            RouteDecision {
                primary: TaskFamily::Code,
                secondary: Some(CodeSpecialization::Testing),
                confidence: 0.85,
            }
        } else if p.contains("code") || p.contains("function") || p.contains("rust") || p.contains("python") {
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
