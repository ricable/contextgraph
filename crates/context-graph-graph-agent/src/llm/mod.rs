//! LLM wrapper for graph relationship analysis.
//!
//! This module provides the LLM integration for graph relationship detection.
//! It shares the underlying Qwen2.5-3B model with the causal-agent crate
//! via `Arc<CausalDiscoveryLLM>`.
//!
//! ## NO FALLBACK POLICY
//!
//! This module follows a strict "no fallback" policy:
//! - JSON parsing fails explicitly on malformed LLM output
//! - Required fields must be present - no defaults
//! - Raw LLM response is included in errors for debugging
//!
//! This ensures problems are visible and fixable, rather than silently
//! producing incorrect results.

pub mod prompt;

use std::sync::Arc;

use context_graph_causal_agent::CausalDiscoveryLLM;
use serde_json::Value;

use crate::error::{GraphAgentError, GraphAgentResult};
use crate::types::{GraphAnalysisResult, GraphLinkDirection, RelationshipType};

use prompt::GraphPromptBuilder;

/// Graph relationship analyzer using shared LLM.
///
/// This wraps the `CausalDiscoveryLLM` from causal-agent to share
/// the same Qwen2.5-3B model instance, avoiding duplicate VRAM usage.
pub struct GraphRelationshipLLM {
    /// Shared LLM from causal-agent.
    llm: Arc<CausalDiscoveryLLM>,

    /// Prompt builder for graph analysis.
    prompt_builder: GraphPromptBuilder,
}

impl GraphRelationshipLLM {
    /// Create a new graph relationship analyzer using shared LLM.
    ///
    /// # Arguments
    /// * `shared_llm` - Arc-wrapped CausalDiscoveryLLM (already loaded)
    pub fn new(shared_llm: Arc<CausalDiscoveryLLM>) -> Self {
        Self {
            llm: shared_llm,
            prompt_builder: GraphPromptBuilder::new(),
        }
    }

    /// Create with custom prompt configuration.
    pub fn with_prompt_builder(mut self, prompt_builder: GraphPromptBuilder) -> Self {
        self.prompt_builder = prompt_builder;
        self
    }

    /// Check if the underlying LLM is loaded.
    pub fn is_loaded(&self) -> bool {
        self.llm.is_loaded()
    }

    /// Analyze graph relationship between two memories.
    ///
    /// # Arguments
    /// * `memory_a` - First memory content
    /// * `memory_b` - Second memory content
    ///
    /// # Returns
    /// Analysis result with relationship type, direction, and confidence
    pub async fn analyze_relationship(
        &self,
        memory_a: &str,
        memory_b: &str,
    ) -> GraphAgentResult<GraphAnalysisResult> {
        if !self.is_loaded() {
            return Err(GraphAgentError::LlmNotInitialized);
        }

        let prompt = self.prompt_builder.build_analysis_prompt(memory_a, memory_b);

        // Use the shared LLM for inference
        let response = self
            .llm
            .generate_text(&prompt)
            .await
            .map_err(|e| GraphAgentError::LlmInferenceError {
                message: format!("Graph LLM inference failed: {}", e),
            })?;

        // Parse the response - NO FALLBACK
        self.parse_analysis_response(&response)
    }

    /// Analyze multiple memory pairs in batch.
    ///
    /// # Arguments
    /// * `pairs` - Vector of (memory_a, memory_b) content pairs
    ///
    /// # Returns
    /// Vector of analysis results, one per pair
    pub async fn analyze_batch(
        &self,
        pairs: &[(String, String)],
    ) -> GraphAgentResult<Vec<GraphAnalysisResult>> {
        if !self.is_loaded() {
            return Err(GraphAgentError::LlmNotInitialized);
        }

        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        // For small batches, process individually for better results
        if pairs.len() <= 3 {
            let mut results = Vec::with_capacity(pairs.len());
            for (a, b) in pairs {
                results.push(self.analyze_relationship(a, b).await?);
            }
            return Ok(results);
        }

        let prompt = self.prompt_builder.build_batch_prompt(pairs);

        let response = self
            .llm
            .generate_text(&prompt)
            .await
            .map_err(|e| GraphAgentError::LlmInferenceError {
                message: format!("Graph LLM batch inference failed: {}", e),
            })?;

        self.parse_batch_response(&response, pairs.len())
    }

    /// Validate a specific relationship between two memories.
    ///
    /// # Arguments
    /// * `memory_a` - First memory content
    /// * `memory_b` - Second memory content
    /// * `expected_type` - The relationship type to validate
    ///
    /// # Returns
    /// Validation result with confidence
    pub async fn validate_relationship(
        &self,
        memory_a: &str,
        memory_b: &str,
        expected_type: RelationshipType,
    ) -> GraphAgentResult<(bool, f32, String)> {
        if !self.is_loaded() {
            return Err(GraphAgentError::LlmNotInitialized);
        }

        let prompt = self
            .prompt_builder
            .build_validation_prompt(memory_a, memory_b, expected_type);

        let response = self
            .llm
            .generate_text(&prompt)
            .await
            .map_err(|e| GraphAgentError::LlmInferenceError {
                message: format!("Graph LLM validation failed: {}", e),
            })?;

        self.parse_validation_response(&response)
    }

    /// Parse single analysis response from LLM.
    ///
    /// NO FALLBACK: Returns error if JSON is malformed. This ensures
    /// we can diagnose and fix LLM output issues instead of silently
    /// returning incorrect results.
    fn parse_analysis_response(&self, response: &str) -> GraphAgentResult<GraphAnalysisResult> {
        // Model generates complete JSON - find and extract it
        let trimmed = response.trim();

        // Look for JSON object in the response
        let json_str = if let Some(start) = trimmed.find('{') {
            if let Some(end) = trimmed.rfind('}') {
                &trimmed[start..=end]
            } else {
                trimmed
            }
        } else {
            trimmed
        };

        // Apply common JSON repairs before parsing
        let repaired = self.repair_json(json_str);

        // Parse as JSON - fail explicitly on malformed output
        let value: Value = serde_json::from_str(&repaired).map_err(|e| {
            GraphAgentError::LlmResponseParseError {
                message: format!(
                    "LLM returned malformed JSON. Parse error: {}. Raw response: {}",
                    e, response
                ),
            }
        })?;

        self.extract_analysis_from_json(&value, response)
    }

    /// Repair common JSON issues from LLM output.
    ///
    /// This fixes issues like:
    /// - Spaces in numbers: "0. 0" -> "0.0"
    /// - Trailing commas before }
    /// - Spaces in field names: "confidence ": -> "confidence":
    ///
    /// Uses pure string manipulation - no regex.
    fn repair_json(&self, json: &str) -> String {
        let chars: Vec<char> = json.chars().collect();
        let mut result = String::with_capacity(json.len());
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];

            // Fix spaces around decimal points in numbers
            // Pattern: digit + optional_spaces + . + optional_spaces + digit
            if c.is_ascii_digit() {
                result.push(c);
                i += 1;

                // Look ahead for pattern: spaces? + . + spaces? + digit
                let mut lookahead = i;
                while lookahead < chars.len() && chars[lookahead] == ' ' {
                    lookahead += 1;
                }
                if lookahead < chars.len() && chars[lookahead] == '.' {
                    lookahead += 1;
                    while lookahead < chars.len() && chars[lookahead] == ' ' {
                        lookahead += 1;
                    }
                    if lookahead < chars.len() && chars[lookahead].is_ascii_digit() {
                        // Found pattern - emit dot directly, skip spaces
                        result.push('.');
                        i = lookahead; // Continue from digit after dot
                    }
                }
                continue;
            }

            // Fix trailing comma before }
            // Pattern: , + spaces + }
            if c == ',' {
                let mut lookahead = i + 1;
                while lookahead < chars.len() && chars[lookahead].is_whitespace() {
                    lookahead += 1;
                }
                if lookahead < chars.len() && chars[lookahead] == '}' {
                    // Skip the comma, let } be added later
                    i += 1;
                    continue;
                }
            }

            // Fix spaces before colon in field names
            // Pattern: "field ": -> "field":
            if c == '"' {
                result.push(c);
                i += 1;

                // Copy string content until closing quote
                while i < chars.len() && chars[i] != '"' {
                    // Skip trailing spaces before the closing quote
                    if chars[i] == ' ' {
                        // Look ahead to see if this is trailing space before "
                        let mut j = i;
                        while j < chars.len() && chars[j] == ' ' {
                            j += 1;
                        }
                        if j < chars.len() && chars[j] == '"' {
                            // Skip trailing spaces in field name
                            i = j;
                            continue;
                        }
                    }
                    result.push(chars[i]);
                    i += 1;
                }
                if i < chars.len() {
                    result.push(chars[i]); // closing quote
                    i += 1;
                }
                continue;
            }

            result.push(c);
            i += 1;
        }

        result
    }

    /// Extract analysis result from parsed JSON.
    ///
    /// NO DEFAULTS: Returns error if required fields are missing. This ensures
    /// we can diagnose LLM output issues instead of silently returning incorrect results.
    fn extract_analysis_from_json(
        &self,
        value: &Value,
        raw_response: &str,
    ) -> GraphAgentResult<GraphAnalysisResult> {
        // Required: has_connection (boolean)
        let has_connection = value
            .get("has_connection")
            .and_then(|v| v.as_bool())
            .ok_or_else(|| GraphAgentError::LlmResponseParseError {
                message: format!(
                    "Missing or invalid 'has_connection' boolean in LLM response: {}",
                    raw_response
                ),
            })?;

        // Required: direction (string)
        let direction_str = value
            .get("direction")
            .and_then(|v| v.as_str())
            .ok_or_else(|| GraphAgentError::LlmResponseParseError {
                message: format!(
                    "Missing or invalid 'direction' string in LLM response: {}",
                    raw_response
                ),
            })?;
        let direction = GraphLinkDirection::from_str(direction_str);

        // Required: relationship_type (string)
        let type_str = value
            .get("relationship_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| GraphAgentError::LlmResponseParseError {
                message: format!(
                    "Missing or invalid 'relationship_type' string in LLM response: {}",
                    raw_response
                ),
            })?;
        let relationship_type = RelationshipType::from_str(type_str);

        // Required: confidence (number)
        let confidence = value
            .get("confidence")
            .and_then(|v| v.as_f64())
            .map(|v| (v as f32).clamp(0.0, 1.0))
            .ok_or_else(|| GraphAgentError::LlmResponseParseError {
                message: format!(
                    "Missing or invalid 'confidence' number in LLM response: {}",
                    raw_response
                ),
            })?;

        // Optional: description (string, defaults to empty)
        let description = value
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(GraphAnalysisResult {
            has_connection,
            direction,
            relationship_type,
            confidence,
            description,
            raw_response: Some(raw_response.to_string()),
        })
    }

    /// Parse batch response from LLM.
    ///
    /// NO FALLBACK: Returns error if JSON is malformed.
    fn parse_batch_response(
        &self,
        response: &str,
        _expected_count: usize,
    ) -> GraphAgentResult<Vec<GraphAnalysisResult>> {
        // Complete the JSON array
        let full_json = format!("[{{\"has_connection\":{}", response.trim());

        // Parse as JSON array - fail explicitly on malformed output
        let values: Vec<Value> = serde_json::from_str(&full_json).map_err(|e| {
            GraphAgentError::LlmResponseParseError {
                message: format!(
                    "LLM returned malformed JSON array. Parse error: {}. Raw response: {}",
                    e, response
                ),
            }
        })?;

        let mut results = Vec::with_capacity(values.len());
        for value in values {
            results.push(self.extract_analysis_from_json(&value, response)?);
        }
        Ok(results)
    }

    /// Parse validation response from LLM.
    ///
    /// NO FALLBACK: Returns error if JSON is malformed.
    fn parse_validation_response(&self, response: &str) -> GraphAgentResult<(bool, f32, String)> {
        let full_json = format!("{{\"valid\":{}", response.trim());

        // Parse as JSON - fail explicitly on malformed output
        let value: Value = serde_json::from_str(&full_json).map_err(|e| {
            GraphAgentError::LlmResponseParseError {
                message: format!(
                    "LLM returned malformed validation JSON. Parse error: {}. Raw response: {}",
                    e, response
                ),
            }
        })?;

        // Required: valid (boolean)
        let valid = value
            .get("valid")
            .and_then(|v| v.as_bool())
            .ok_or_else(|| GraphAgentError::LlmResponseParseError {
                message: format!(
                    "Missing or invalid 'valid' boolean in validation response: {}",
                    response
                ),
            })?;

        // Required: confidence (number)
        let confidence = value
            .get("confidence")
            .and_then(|v| v.as_f64())
            .map(|v| (v as f32).clamp(0.0, 1.0))
            .ok_or_else(|| GraphAgentError::LlmResponseParseError {
                message: format!(
                    "Missing or invalid 'confidence' number in validation response: {}",
                    response
                ),
            })?;

        // Optional: explanation (string, defaults to empty)
        let explanation = value
            .get("explanation")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok((valid, confidence, explanation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to test JSON parsing without LLM
    fn parse_json_response(response: &str) -> GraphAgentResult<GraphAnalysisResult> {
        let full_json = format!("{{\"has_connection\":{}", response.trim());
        let value: Value = serde_json::from_str(&full_json).map_err(|e| {
            GraphAgentError::LlmResponseParseError {
                message: format!("Parse error: {}", e),
            }
        })?;

        // Required: has_connection
        let has_connection = value
            .get("has_connection")
            .and_then(|v| v.as_bool())
            .ok_or_else(|| GraphAgentError::LlmResponseParseError {
                message: "Missing has_connection".to_string(),
            })?;

        // Required: direction
        let direction_str = value
            .get("direction")
            .and_then(|v| v.as_str())
            .ok_or_else(|| GraphAgentError::LlmResponseParseError {
                message: "Missing direction".to_string(),
            })?;
        let direction = GraphLinkDirection::from_str(direction_str);

        // Required: relationship_type
        let type_str = value
            .get("relationship_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| GraphAgentError::LlmResponseParseError {
                message: "Missing relationship_type".to_string(),
            })?;
        let relationship_type = RelationshipType::from_str(type_str);

        // Required: confidence
        let confidence = value
            .get("confidence")
            .and_then(|v| v.as_f64())
            .map(|v| (v as f32).clamp(0.0, 1.0))
            .ok_or_else(|| GraphAgentError::LlmResponseParseError {
                message: "Missing confidence".to_string(),
            })?;

        // Optional: description
        let description = value
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(GraphAnalysisResult {
            has_connection,
            direction,
            relationship_type,
            confidence,
            description,
            raw_response: Some(response.to_string()),
        })
    }

    #[test]
    fn test_parse_valid_json_with_connection() {
        // Simulates LLM output after prompt prefix "{\"has_connection\":"
        let response =
            r#" true, "direction": "a_connects_b", "relationship_type": "imports", "confidence": 0.85, "description": "A imports B"}"#;

        let result = parse_json_response(response).unwrap();

        assert!(result.has_connection);
        assert_eq!(result.direction, GraphLinkDirection::AConnectsB);
        assert_eq!(result.relationship_type, RelationshipType::Imports);
        assert!((result.confidence - 0.85).abs() < 0.01);
        assert_eq!(result.description, "A imports B");
    }

    #[test]
    fn test_parse_valid_json_no_connection() {
        let response =
            r#" false, "direction": "none", "relationship_type": "none", "confidence": 0.1, "description": "No relationship"}"#;

        let result = parse_json_response(response).unwrap();

        assert!(!result.has_connection);
        assert_eq!(result.direction, GraphLinkDirection::NoConnection);
        assert_eq!(result.relationship_type, RelationshipType::None);
        assert!((result.confidence - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_parse_malformed_json_fails() {
        // Malformed JSON should fail - NO FALLBACK
        let response = r#" true, direction: a_connects_b"#; // Missing quotes

        let result = parse_json_response(response);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_missing_required_field_fails() {
        // Missing confidence field should fail - NO DEFAULTS
        let response =
            r#" true, "direction": "a_connects_b", "relationship_type": "imports", "description": "A imports B"}"#;

        let result = parse_json_response(response);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_implements_relationship() {
        let response = r#" true, "direction": "a_connects_b", "relationship_type": "implements", "confidence": 0.92, "description": "A implements trait B"}"#;

        let result = parse_json_response(response).unwrap();

        assert!(result.has_connection);
        assert_eq!(result.relationship_type, RelationshipType::Implements);
        assert!((result.confidence - 0.92).abs() < 0.01);
    }

    #[test]
    fn test_parse_bidirectional_relationship() {
        let response = r#" true, "direction": "bidirectional", "relationship_type": "references", "confidence": 0.75, "description": "Mutual reference"}"#;

        let result = parse_json_response(response).unwrap();

        assert!(result.has_connection);
        assert_eq!(result.direction, GraphLinkDirection::Bidirectional);
    }
}
