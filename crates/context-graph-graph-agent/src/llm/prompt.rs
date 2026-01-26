//! Graph relationship analysis prompts for Hermes 2 Pro.
//!
//! This module provides prompt templates for LLM-based graph relationship
//! detection, using the ChatML format for Hermes 2 Pro Mistral 7B.
//!
//! ## Key Features:
//!
//! 1. **Grammar-constrained output**: GBNF ensures 100% valid JSON
//! 2. **Hermes 2 Pro optimization**: Trained for function calling and structured output
//! 3. **Simplified prompts**: Direct instructions for better accuracy

use crate::types::RelationshipType;

/// Prompt builder for graph relationship analysis.
///
/// Generates prompts in ChatML format compatible with Hermes 2 Pro Mistral 7B.
pub struct GraphPromptBuilder {
    /// Maximum content length per memory (chars).
    max_content_length: usize,
}

impl Default for GraphPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphPromptBuilder {
    /// Create a new prompt builder with default settings.
    pub fn new() -> Self {
        Self {
            max_content_length: 1500,
        }
    }

    /// Create with custom max content length.
    pub fn with_max_content_length(mut self, max_length: usize) -> Self {
        self.max_content_length = max_length;
        self
    }

    /// Build a prompt for analyzing graph relationship between two memories.
    ///
    /// # Arguments
    /// * `memory_a` - First memory content
    /// * `memory_b` - Second memory content
    ///
    /// # Returns
    /// ChatML-formatted prompt string
    pub fn build_analysis_prompt(&self, memory_a: &str, memory_b: &str) -> String {
        let truncated_a = self.truncate_content(memory_a);
        let truncated_b = self.truncate_content(memory_b);

        format!(
            r#"<|im_start|>system
You are a code relationship analyzer. Identify structural relationships between code snippets.

Output JSON with these fields:
- has_connection: true if there is a structural relationship, false otherwise
- direction: "a_to_b", "b_to_a", "bidirectional", or "none"
- relationship_type: "imports", "implements", "calls", "depends_on", "references", "extends", "contains", or "none"
- confidence: 0.0 to 1.0 indicating your confidence
- description: brief explanation of the relationship
<|im_end|>
<|im_start|>user
Analyze if there is a structural relationship between these code snippets:

Code A:
{}

Code B:
{}
<|im_end|>
<|im_start|>assistant
"#,
            truncated_a, truncated_b
        )
    }

    /// Build a prompt for batch analysis of multiple memory pairs.
    ///
    /// # Arguments
    /// * `pairs` - Vector of (memory_a, memory_b) content pairs
    ///
    /// # Returns
    /// ChatML-formatted prompt string expecting JSON array response
    pub fn build_batch_prompt(&self, pairs: &[(String, String)]) -> String {
        let mut pairs_text = String::new();

        for (i, (a, b)) in pairs.iter().enumerate() {
            let truncated_a = self.truncate_content(a);
            let truncated_b = self.truncate_content(b);

            pairs_text.push_str(&format!(
                "Pair {}:\n  A: {}\n  B: {}\n\n",
                i + 1,
                truncated_a,
                truncated_b
            ));
        }

        format!(
            r#"<|im_start|>system
You are a code relationship analyzer. Identify structural relationships between code snippets.
For each pair, output JSON with: has_connection, direction, relationship_type, confidence, description.
<|im_end|>
<|im_start|>user
Analyze these code pairs for structural relationships:

{}
<|im_end|>
<|im_start|>assistant
"#,
            pairs_text
        )
    }

    /// Build a prompt for validating a specific relationship type.
    ///
    /// # Arguments
    /// * `memory_a` - First memory content
    /// * `memory_b` - Second memory content
    /// * `expected_type` - The relationship type to validate
    ///
    /// # Returns
    /// ChatML-formatted prompt string
    pub fn build_validation_prompt(
        &self,
        memory_a: &str,
        memory_b: &str,
        expected_type: RelationshipType,
    ) -> String {
        let truncated_a = self.truncate_content(memory_a);
        let truncated_b = self.truncate_content(memory_b);

        format!(
            r#"<|im_start|>system
You validate code relationships. Determine if the proposed relationship is accurate.

Output JSON with these fields:
- valid: true if the relationship exists, false otherwise
- confidence: 0.0 to 1.0 indicating your confidence
- explanation: brief explanation of your assessment
<|im_end|>
<|im_start|>user
Does code A have a "{}" relationship with code B?

Code A:
{}

Code B:
{}
<|im_end|>
<|im_start|>assistant
"#,
            expected_type.as_str(),
            truncated_a,
            truncated_b
        )
    }

    /// Truncate content to max length at word boundary.
    fn truncate_content(&self, content: &str) -> String {
        // Normalize whitespace
        let content: String = content
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        if content.len() <= self.max_content_length {
            return content;
        }

        // Find last space before max length
        let truncated = &content[..self.max_content_length];
        if let Some(last_space) = truncated.rfind(' ') {
            format!("{}...", &content[..last_space])
        } else {
            format!("{}...", truncated)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_builder_default() {
        let builder = GraphPromptBuilder::new();
        assert_eq!(builder.max_content_length, 1500);
    }

    #[test]
    fn test_analysis_prompt_format() {
        let builder = GraphPromptBuilder::new();
        let prompt = builder.build_analysis_prompt("use crate::foo;", "pub mod foo {}");

        // Verify ChatML tokens present
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_end|>"));
        // Verify user content is included
        assert!(prompt.contains("use crate::foo;"));
        assert!(prompt.contains("pub mod foo"));
        // Verify prompt ends with assistant tag for model to respond
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_content_truncation() {
        let builder = GraphPromptBuilder::new().with_max_content_length(20);
        let long_content = "This is a very long content string that should be truncated";
        let truncated = builder.truncate_content(long_content);

        assert!(truncated.len() <= 25); // 20 + "..."
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_batch_prompt_format() {
        let builder = GraphPromptBuilder::new();
        let pairs = vec![
            ("use foo;".to_string(), "mod foo;".to_string()),
            ("calls bar()".to_string(), "fn bar() {}".to_string()),
        ];

        let prompt = builder.build_batch_prompt(&pairs);
        assert!(prompt.contains("Pair 1:"));
        assert!(prompt.contains("Pair 2:"));
    }

    #[test]
    fn test_validation_prompt_format() {
        let builder = GraphPromptBuilder::new();
        let prompt = builder.build_validation_prompt(
            "use crate::module;",
            "pub mod module {}",
            RelationshipType::Imports,
        );

        assert!(prompt.contains("imports"));
        assert!(prompt.contains("valid"));
    }
}
