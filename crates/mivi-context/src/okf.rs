//! Open Knowledge Format (OKF v0.2) Parser and Progressive Disclosure Navigator.
//!
//! Conforms to the Google Cloud Platform Open Knowledge Format v0.2 specification.
//! Encapsulates vendor-neutral knowledge concepts stored as Markdown with structured YAML frontmatter,
//! featuring first-class provenance, trust tiers, lifecycle freshness, and progressive disclosure.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// OKF v0.2 Concept YAML Frontmatter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OkfFrontmatter {
    #[serde(default)]
    pub id: String,
    #[serde(default = "default_doc_type")]
    pub r#type: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub sources: Vec<String>,
    #[serde(default)]
    pub trust_tier: Option<String>,
    #[serde(default = "default_status")]
    pub status: String,
    #[serde(default)]
    pub stale_after: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_doc_type() -> String {
    "concept".to_string()
}

fn default_status() -> String {
    "active".to_string()
}

impl Default for OkfFrontmatter {
    fn default() -> Self {
        Self {
            id: String::new(),
            r#type: default_doc_type(),
            title: None,
            sources: Vec::new(),
            trust_tier: Some("verified".to_string()),
            status: default_status(),
            stale_after: None,
            tags: Vec::new(),
        }
    }
}

/// A parsed Open Knowledge Format v0.2 Concept document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OkfConcept {
    pub frontmatter: OkfFrontmatter,
    pub title: String,
    pub body: String,
    pub wiki_links: Vec<String>,
}

impl OkfConcept {
    /// Parses an OKF v0.2 Markdown document with YAML frontmatter.
    pub fn parse(raw: &str, default_id: &str) -> Result<Self, String> {
        let trimmed = raw.trim_start();
        if !trimmed.starts_with("---") {
            // Document without frontmatter: default frontmatter with first header as title
            let (title, body) = extract_title_and_body(raw);
            return Ok(Self {
                frontmatter: OkfFrontmatter {
                    id: default_id.to_string(),
                    title: Some(title.clone()),
                    ..Default::default()
                },
                title,
                wiki_links: extract_wiki_links(raw),
                body,
            });
        }

        // Find closing frontmatter delimiter `---`
        let after_start = &trimmed[3..];
        let Some(end_idx) = after_start.find("\n---") else {
            return Err("Malformed OKF document: unclosed YAML frontmatter delimiter '---'".into());
        };

        let yaml_str = &after_start[..end_idx].trim();
        let body_str = &after_start[end_idx + 4..].trim_start();

        let mut frontmatter = parse_simple_yaml(yaml_str);
        if frontmatter.id.is_empty() {
            frontmatter.id = default_id.to_string();
        }

        let (extracted_title, body) = extract_title_and_body(body_str);
        let title = frontmatter
            .title
            .clone()
            .unwrap_or_else(|| extracted_title);

        let wiki_links = extract_wiki_links(body_str);

        Ok(Self {
            frontmatter,
            title,
            body,
            wiki_links,
        })
    }

    /// Check if this concept is still active and not marked stale/deprecated.
    pub fn is_active(&self) -> bool {
        self.frontmatter.status.eq_ignore_ascii_case("active")
    }
}

/// Progressive disclosure directory navigator for OKF Knowledge Bundles.
#[derive(Debug, Clone, Default)]
pub struct OkfBundleNavigator {
    pub concepts: HashMap<String, OkfConcept>,
    pub root_indexes: Vec<String>,
}

impl OkfBundleNavigator {
    pub fn new() -> Self {
        Self {
            concepts: HashMap::new(),
            root_indexes: Vec::new(),
        }
    }

    /// Ingests an OKF concept into the bundle catalog.
    pub fn insert_concept(&mut self, concept: OkfConcept) {
        let id = concept.frontmatter.id.clone();
        if id.ends_with("index") || id == "root" {
            self.root_indexes.push(id.clone());
        }
        self.concepts.insert(id, concept);
    }

    /// Retrieves top-level progressive disclosure summaries without bloating context with full leaf bodies.
    pub fn get_progressive_outline(&self) -> String {
        let mut outline = String::from("# Knowledge Bundle Catalog\n\n");
        for (id, concept) in &self.concepts {
            if concept.is_active() {
                outline.push_str(&format!(
                    "- **[{}]({})** (`{}`): {}\n",
                    concept.title,
                    id,
                    concept.frontmatter.r#type,
                    concept.frontmatter.trust_tier.as_deref().unwrap_or("verified")
                ));
            }
        }
        outline
    }

    /// Retrieve full concept by ID.
    pub fn get_concept(&self, id: &str) -> Option<&OkfConcept> {
        self.concepts.get(id)
    }
}

/// Extract first markdown header `# Title` and return title + remaining body.
fn extract_title_and_body(markdown: &str) -> (String, String) {
    let mut title = "Untitled Concept".to_string();
    let mut body = String::new();
    let mut title_found = false;

    for line in markdown.lines() {
        let trimmed = line.trim();
        if !title_found && trimmed.starts_with('#') {
            title = trimmed.trim_start_matches('#').trim().to_string();
            title_found = true;
        } else {
            body.push_str(line);
            body.push('\n');
        }
    }

    (title, body.trim().to_string())
}

/// Extracts wiki-style internal graph links: `[[concept_id]]`.
fn extract_wiki_links(text: &str) -> Vec<String> {
    let mut links = Vec::new();
    let mut pos = 0;
    while let Some(start) = text[pos..].find("[[") {
        let actual_start = pos + start + 2;
        if let Some(end) = text[actual_start..].find("]]") {
            let actual_end = actual_start + end;
            let link = text[actual_start..actual_end].trim().to_string();
            if !link.is_empty() {
                links.push(link);
            }
            pos = actual_end + 2;
        } else {
            break;
        }
    }
    links
}

/// Lightweight YAML parser for frontmatter metadata without heavy runtime dependencies.
fn parse_simple_yaml(yaml: &str) -> OkfFrontmatter {
    let mut frontmatter = OkfFrontmatter::default();
    let mut active_list_key: Option<&'static str> = None;

    for line in yaml.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Handle multiline list item: - "value" or - value
        if trimmed.starts_with('-') {
            let item = trimmed.trim_start_matches('-').trim().trim_matches('"').trim_matches('\'').to_string();
            if !item.is_empty() {
                match active_list_key {
                    Some("sources") => frontmatter.sources.push(item),
                    Some("tags") => frontmatter.tags.push(item),
                    _ => {}
                }
            }
            continue;
        }

        if let Some((key, val)) = trimmed.split_once(':') {
            let key = key.trim().to_lowercase();
            let val = val.trim().trim_matches('"').trim_matches('\'');
            active_list_key = None;

            match key.as_str() {
                "id" | "concept_id" => frontmatter.id = val.to_string(),
                "type" => frontmatter.r#type = val.to_string(),
                "title" => frontmatter.title = Some(val.to_string()),
                "trust" | "trust_tier" => frontmatter.trust_tier = Some(val.to_string()),
                "status" => frontmatter.status = val.to_string(),
                "stale_after" => frontmatter.stale_after = Some(val.to_string()),
                "sources" => {
                    if val.starts_with('[') && val.ends_with(']') {
                        frontmatter.sources = val[1..val.len() - 1]
                            .split(',')
                            .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                    } else if !val.is_empty() {
                        frontmatter.sources = vec![val.to_string()];
                    } else {
                        active_list_key = Some("sources");
                    }
                }
                "tags" => {
                    if val.starts_with('[') && val.ends_with(']') {
                        frontmatter.tags = val[1..val.len() - 1]
                            .split(',')
                            .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                    } else if !val.is_empty() {
                        frontmatter.tags = vec![val.to_string()];
                    } else {
                        active_list_key = Some("tags");
                    }
                }
                _ => {}
            }
        }
    }

    frontmatter
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_okf_concept_with_frontmatter() {
        let doc = r#"---
id: architecture/turboquant
type: algorithm
title: TurboQuant Vector Quantization
trust_tier: verified
status: active
sources: ["arXiv:2504.19874", "Google Research"]
tags: [quantization, sim_search, slm]
---
# TurboQuant Vector Quantization

TurboQuant performs 4-bit data-oblivious vector quantization using Fast Walsh-Hadamard transforms.
See also [[crates/mivi-core]] and [[crates/mivi-kv]].
"#;

        let concept = OkfConcept::parse(doc, "default_id").unwrap();
        assert_eq!(concept.frontmatter.id, "architecture/turboquant");
        assert_eq!(concept.frontmatter.r#type, "algorithm");
        assert_eq!(concept.title, "TurboQuant Vector Quantization");
        assert_eq!(concept.frontmatter.trust_tier.as_deref(), Some("verified"));
        assert!(concept.is_active());
        assert_eq!(concept.frontmatter.sources.len(), 2);
        assert_eq!(concept.frontmatter.tags.len(), 3);
        assert_eq!(concept.wiki_links, vec!["crates/mivi-core", "crates/mivi-kv"]);
        assert!(concept.body.contains("TurboQuant performs 4-bit"));
    }

    #[test]
    fn test_okf_bundle_navigator_progressive_outline() {
        let mut nav = OkfBundleNavigator::new();
        let concept1 = OkfConcept::parse(
            "---\nid: model/ssm\ntitle: State Space Models\ntype: layer\n---\n# State Space Models\nSSM layers scan in linear time.",
            "model/ssm",
        )
        .unwrap();

        let concept2 = OkfConcept::parse(
            "---\nid: model/attention\ntitle: GQA Attention\ntype: layer\n---\n# GQA Attention\nGQA layers compute KV cache attention.",
            "model/attention",
        )
        .unwrap();

        nav.insert_concept(concept1);
        nav.insert_concept(concept2);

        let outline = nav.get_progressive_outline();
        assert!(outline.contains("Knowledge Bundle Catalog"));
        assert!(outline.contains("State Space Models"));
        assert!(outline.contains("GQA Attention"));
    }

    #[test]
    fn test_parse_okf_concept_with_multiline_yaml_lists() {
        let doc = r#"---
id: algorithms/pld
type: algorithm
title: Prompt Lookup Decoding
sources:
  - "Google Research"
  - "Apoorv Saxena"
tags:
  - speculative_decoding
  - pld
  - latency
---
# Prompt Lookup Decoding
Matches n-grams in the context buffer.
"#;

        let concept = OkfConcept::parse(doc, "default_id").unwrap();
        assert_eq!(concept.frontmatter.id, "algorithms/pld");
        assert_eq!(concept.frontmatter.sources, vec!["Google Research", "Apoorv Saxena"]);
        assert_eq!(concept.frontmatter.tags, vec!["speculative_decoding", "pld", "latency"]);
    }
}
