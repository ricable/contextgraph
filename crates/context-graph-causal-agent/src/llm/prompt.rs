//! Prompt templates for the Causal Discovery LLM.
//!
//! Uses Hermes 2 Pro Mistral's ChatML format for optimal function-calling performance.
//! The model is specifically trained for structured output and JSON generation.

/// Builder for causal analysis prompts.
#[derive(Debug, Clone)]
pub struct CausalPromptBuilder {
    /// System prompt for the LLM.
    system_prompt: String,

    /// Maximum content length per memory (characters).
    max_content_length: usize,
}

impl Default for CausalPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalPromptBuilder {
    /// Create a new prompt builder with default settings.
    pub fn new() -> Self {
        Self {
            system_prompt: Self::default_system_prompt().to_string(),
            max_content_length: 1500,
        }
    }

    /// Set a custom system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set maximum content length per memory.
    pub fn with_max_content_length(mut self, length: usize) -> Self {
        self.max_content_length = length;
        self
    }

    /// Build the full analysis prompt for two memories.
    ///
    /// Uses Hermes 2 Pro ChatML format with structured output instructions.
    /// The model will generate valid JSON constrained by GBNF grammar.
    pub fn build_analysis_prompt(&self, memory_a: &str, memory_b: &str) -> String {
        let truncated_a = self.truncate_content(memory_a);
        let truncated_b = self.truncate_content(memory_b);

        format!(
            r#"<|im_start|>system
{}
<|im_end|>
<|im_start|>user
Analyze the causal relationship between these two statements:

Statement A: "{}"

Statement B: "{}"

Determine if there is a causal relationship and respond with JSON.
<|im_end|>
<|im_start|>assistant
"#,
            self.system_prompt, truncated_a, truncated_b
        )
    }

    /// Build a batch analysis prompt for multiple pairs.
    pub fn build_batch_prompt(&self, pairs: &[(String, String)]) -> String {
        let mut pairs_text = String::new();
        for (i, (a, b)) in pairs.iter().enumerate() {
            let truncated_a = self.truncate_content(a);
            let truncated_b = self.truncate_content(b);
            pairs_text.push_str(&format!(
                "Pair {}:\n  A: \"{}\"\n  B: \"{}\"\n\n",
                i + 1,
                truncated_a,
                truncated_b
            ));
        }

        format!(
            r#"<|im_start|>system
{}
<|im_end|>
<|im_start|>user
Analyze these statement pairs for causal relationships:

{}

For each pair, determine if there's a causal relationship.
<|im_end|>
<|im_start|>assistant
"#,
            self.system_prompt, pairs_text
        )
    }

    /// Truncate content to maximum length.
    fn truncate_content(&self, content: &str) -> String {
        if content.len() <= self.max_content_length {
            content.to_string()
        } else {
            let truncated = &content[..self.max_content_length];
            // Find last complete word
            if let Some(last_space) = truncated.rfind(' ') {
                format!("{}...", &truncated[..last_space])
            } else {
                format!("{}...", truncated)
            }
        }
    }

    /// Default system prompt optimized for Hermes 2 Pro.
    ///
    /// Hermes 2 Pro is trained for function calling and structured output,
    /// so we keep the prompt focused and direct.
    const fn default_system_prompt() -> &'static str {
        r#"You are a causal reasoning expert. Analyze statements to identify cause-effect relationships.

Output JSON with these fields:
- causal_link: true if there is a causal relationship, false otherwise
- direction: "A_causes_B", "B_causes_A", "bidirectional", or "none"
- confidence: 0.0 to 1.0 indicating your confidence
- mechanism: brief explanation of the causal mechanism

Guidelines:
- Causation requires one event to lead to or produce another
- Correlation alone is not causation
- Consider temporal ordering: causes precede effects
- Be conservative: only claim causation when evidence supports it"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_analysis_prompt() {
        let builder = CausalPromptBuilder::new();
        let prompt = builder.build_analysis_prompt("Event A occurred", "Event B followed");

        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("Event A occurred"));
        assert!(prompt.contains("Event B followed"));
        assert!(prompt.contains("causal_link"));
    }

    #[test]
    fn test_truncate_content() {
        let builder = CausalPromptBuilder::new().with_max_content_length(50);

        let short = "This is a short string";
        assert_eq!(builder.truncate_content(short), short);

        let long = "This is a much longer string that exceeds the maximum content length limit";
        let truncated = builder.truncate_content(long);
        assert!(truncated.len() <= 55); // 50 + "..."
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_build_batch_prompt() {
        let builder = CausalPromptBuilder::new();
        let pairs = vec![
            ("A1".to_string(), "B1".to_string()),
            ("A2".to_string(), "B2".to_string()),
        ];

        let prompt = builder.build_batch_prompt(&pairs);

        assert!(prompt.contains("Pair 1:"));
        assert!(prompt.contains("Pair 2:"));
        assert!(prompt.contains("A1"));
        assert!(prompt.contains("B2"));
    }
}
