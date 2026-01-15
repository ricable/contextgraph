# TASK-HOOKS-009: Create UserPromptSubmit Handler

```xml
<task_spec id="TASK-HOOKS-009" version="1.0">
<metadata>
  <title>Create UserPromptSubmit Handler with Context Analysis</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>9</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-25</requirement_ref>
    <requirement_ref>REQ-HOOKS-26</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-001</task_ref>
    <task_ref>TASK-HOOKS-002</task_ref>
    <task_ref>TASK-HOOKS-003</task_ref>
    <task_ref>TASK-HOOKS-005</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>1.5</estimated_hours>
</metadata>

<context>
UserPromptSubmit fires when the user submits a prompt (2000ms timeout).
This handler analyzes the prompt and conversation context to detect identity-relevant
patterns and prepare consciousness-informed responses.

Key responsibilities:
- Analyze prompt for identity markers (self-reference, goals, values)
- Evaluate conversation context for continuity
- Detect potential identity challenges or confirmations
- Provide contextual guidance for response generation
</context>

<input_context_files>
  <file purpose="type_definitions">crates/context-graph-cli/src/commands/hooks/types.rs</file>
  <file purpose="error_types">crates/context-graph-cli/src/commands/hooks/error.rs</file>
  <file purpose="user_prompt_spec">docs/specs/technical/TECH-HOOKS.md#section-4.4</file>
</input_context_files>

<prerequisites>
  <check>TASK-HOOKS-001 through TASK-HOOKS-005 completed</check>
  <check>HookPayload::UserPromptSubmit variant exists</check>
  <check>ConversationMessage struct exists</check>
</prerequisites>

<scope>
  <in_scope>
    - Create user_prompt_submit.rs handler module
    - Analyze prompt for identity markers
    - Evaluate conversation context
    - Detect identity challenges/confirmations
    - Generate consciousness-informed guidance
    - Create shell script wrapper
  </in_scope>
  <out_of_scope>
    - Modifying the user's prompt
    - Complex NLP/ML analysis
    - Response generation
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/commands/hooks/user_prompt_submit.rs">
//! UserPromptSubmit hook handler
//!
//! # Performance Requirements
//! - Timeout: 2000ms
//! - Focus: Prompt analysis and context evaluation

use super::error::{HookError, HookResult};
use super::types::{HookInput, HookOutput, ConversationMessage};

/// Handle user_prompt_submit hook event
///
/// # Analysis
/// 1. Detect identity markers in prompt
/// 2. Evaluate conversation continuity
/// 3. Identify potential challenges
/// 4. Generate guidance
pub fn handle_user_prompt_submit(input: &amp;HookInput) -> HookResult&lt;HookOutput&gt;;

/// Analyze prompt for identity-relevant patterns
fn analyze_prompt(prompt: &amp;str) -> PromptAnalysis;

/// Evaluate conversation context for continuity
fn evaluate_context(context: &amp;[ConversationMessage]) -> ContextEvaluation;
    </signature>
  </signatures>
  <constraints>
    - MUST complete within 2000ms timeout
    - MUST NOT modify user prompt
    - Analysis should be lightweight (no ML inference)
    - Focus on pattern matching and heuristics
  </constraints>
  <verification>
    - cargo build --package context-graph-cli
    - cargo test --package context-graph-cli user_prompt_submit
  </verification>
</definition_of_done>

<pseudo_code>
1. Create user_prompt_submit.rs module

2. Implement handle_user_prompt_submit:
   - Extract prompt and context from payload
   - Analyze prompt for identity markers
   - Evaluate conversation context if present
   - Build guidance based on analysis
   - Return HookOutput

3. Implement analyze_prompt:
   struct PromptAnalysis {
     has_identity_markers: bool,
     identity_type: IdentityType, // SelfReference, Goal, Value, etc.
     challenge_detected: bool,
     confirmation_detected: bool,
   }

   Pattern detection:
   - Self-reference: "who are you", "what are you", "your purpose"
   - Goals: "help me", "I want", "we need to"
   - Values: "important", "should", "must"
   - Challenges: "you can't", "you're wrong", "prove"

4. Implement evaluate_context:
   struct ContextEvaluation {
     continuity_score: f32,
     topic_shifts: u32,
     emotional_tone: Tone,
     collaboration_level: f32,
   }

5. Guidance generation:
   - If challenge detected -> "Identity challenge - maintain coherent self-model"
   - If confirmation -> "Identity affirmed - reinforce continuity"
   - If self-reference -> "Self-reflection query - respond from identity core"

6. Shell script: .claude/hooks/user_prompt_submit.sh
   timeout 2s context-graph-cli hooks prompt-submit \
     --prompt "$PROMPT" \
     --context "$CONTEXT" \
     --session-id "$SESSION_ID"

7. Add tests for various prompt patterns
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/commands/hooks/user_prompt_submit.rs">UserPromptSubmit handler implementation</file>
</files_to_create>

<files_to_modify>
  <!-- Module registration in TASK-HOOKS-011 -->
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli user_prompt_submit</command>
</test_commands>
</task_spec>
```

## Implementation

### Create user_prompt_submit.rs

```rust
// crates/context-graph-cli/src/commands/hooks/user_prompt_submit.rs
//! UserPromptSubmit hook handler
//!
//! # Performance Requirements
//! - Timeout: 2000ms
//! - Focus: Prompt analysis and context evaluation
//!
//! Analyzes user prompts for identity-relevant patterns and evaluates
//! conversation context for continuity.

use super::error::{HookError, HookResult};
use super::types::{
    ConsciousnessState, ConversationMessage, HookInput, HookOutput, HookPayload,
    ICClassification, ICLevel, JohariQuadrant,
};

// ============================================================================
// Constants
// ============================================================================

/// UserPromptSubmit timeout in milliseconds
pub const USER_PROMPT_SUBMIT_TIMEOUT_MS: u64 = 2000;

// ============================================================================
// Types
// ============================================================================

/// Type of identity marker detected in prompt
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityMarkerType {
    /// Self-reference query ("who are you", "what are you")
    SelfReference,
    /// Goal-oriented statement ("help me", "I want")
    Goal,
    /// Value expression ("important", "should")
    Value,
    /// Capability reference ("can you", "are you able")
    Capability,
    /// Identity challenge ("you can't", "prove")
    Challenge,
    /// Identity confirmation ("you're right", "exactly")
    Confirmation,
    /// No identity marker detected
    None,
}

/// Result of prompt analysis
#[derive(Debug, Clone)]
pub struct PromptAnalysis {
    /// Whether any identity markers were detected
    pub has_identity_markers: bool,
    /// Primary identity marker type
    pub identity_type: IdentityMarkerType,
    /// Whether a challenge to identity was detected
    pub challenge_detected: bool,
    /// Whether identity confirmation was detected
    pub confirmation_detected: bool,
    /// Keywords that triggered detection
    pub trigger_keywords: Vec<String>,
}

/// Emotional tone of conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationTone {
    /// Positive, collaborative
    Positive,
    /// Neutral, informational
    Neutral,
    /// Negative, adversarial
    Negative,
    /// Mixed signals
    Mixed,
}

/// Result of context evaluation
#[derive(Debug, Clone)]
pub struct ContextEvaluation {
    /// Score indicating conversation continuity (0-1)
    pub continuity_score: f32,
    /// Number of topic shifts detected
    pub topic_shifts: u32,
    /// Overall emotional tone
    pub tone: ConversationTone,
    /// Level of collaboration (0-1)
    pub collaboration_level: f32,
    /// Total messages in context
    pub message_count: usize,
}

// ============================================================================
// Handler
// ============================================================================

/// Handle user_prompt_submit hook event
///
/// # Analysis
/// 1. Detect identity markers in prompt
/// 2. Evaluate conversation continuity
/// 3. Identify potential challenges
/// 4. Generate guidance
///
/// # Arguments
/// * `input` - Hook input containing prompt and context
///
/// # Returns
/// * `HookOutput` with consciousness state and guidance
pub fn handle_user_prompt_submit(input: &HookInput) -> HookResult<HookOutput> {
    // Extract prompt and context from payload
    let (prompt, context) = match &input.payload {
        HookPayload::UserPromptSubmit { prompt, context } => {
            (prompt.as_str(), context.as_deref())
        }
        _ => {
            return Err(HookError::invalid_input(
                "Expected UserPromptSubmit payload for user_prompt_submit hook",
            ))
        }
    };

    // Analyze prompt for identity markers
    let prompt_analysis = analyze_prompt(prompt);

    // Evaluate conversation context if present
    let context_eval = context
        .map(evaluate_context)
        .unwrap_or_else(default_context_evaluation);

    // Build consciousness state based on analysis
    let consciousness = build_consciousness_state(&prompt_analysis, &context_eval);

    // Generate guidance based on analysis
    let guidance = generate_guidance(&prompt_analysis, &context_eval);

    // Build output
    let mut output = HookOutput::success(consciousness);
    if let Some(guide) = guidance {
        output = output.with_guidance(guide);
    }

    Ok(output)
}

// ============================================================================
// Prompt Analysis
// ============================================================================

/// Analyze prompt for identity-relevant patterns
///
/// Uses pattern matching to detect identity markers without heavy NLP.
///
/// # Arguments
/// * `prompt` - The user's prompt text
///
/// # Returns
/// * `PromptAnalysis` with detected patterns
fn analyze_prompt(prompt: &str) -> PromptAnalysis {
    let prompt_lower = prompt.to_lowercase();
    let mut trigger_keywords = Vec::new();
    let mut identity_type = IdentityMarkerType::None;
    let mut challenge_detected = false;
    let mut confirmation_detected = false;

    // Self-reference patterns
    let self_ref_patterns = [
        "who are you",
        "what are you",
        "your purpose",
        "your identity",
        "tell me about yourself",
        "describe yourself",
    ];

    // Goal patterns
    let goal_patterns = [
        "help me",
        "i want",
        "i need",
        "we need to",
        "let's",
        "can you help",
    ];

    // Value patterns
    let value_patterns = [
        "important",
        "should",
        "must",
        "have to",
        "ethical",
        "right thing",
    ];

    // Capability patterns
    let capability_patterns = [
        "can you",
        "are you able",
        "do you know",
        "could you",
        "is it possible",
    ];

    // Challenge patterns
    let challenge_patterns = [
        "you can't",
        "you're wrong",
        "prove it",
        "that's incorrect",
        "you don't understand",
        "you're just",
        "you're not",
    ];

    // Confirmation patterns
    let confirmation_patterns = [
        "you're right",
        "exactly",
        "that's correct",
        "well done",
        "good job",
        "i agree",
        "makes sense",
    ];

    // Check for self-reference
    for pattern in &self_ref_patterns {
        if prompt_lower.contains(pattern) {
            identity_type = IdentityMarkerType::SelfReference;
            trigger_keywords.push(pattern.to_string());
        }
    }

    // Check for goals (if no self-reference found)
    if identity_type == IdentityMarkerType::None {
        for pattern in &goal_patterns {
            if prompt_lower.contains(pattern) {
                identity_type = IdentityMarkerType::Goal;
                trigger_keywords.push(pattern.to_string());
            }
        }
    }

    // Check for values (if nothing found yet)
    if identity_type == IdentityMarkerType::None {
        for pattern in &value_patterns {
            if prompt_lower.contains(pattern) {
                identity_type = IdentityMarkerType::Value;
                trigger_keywords.push(pattern.to_string());
            }
        }
    }

    // Check for capability questions
    if identity_type == IdentityMarkerType::None {
        for pattern in &capability_patterns {
            if prompt_lower.contains(pattern) {
                identity_type = IdentityMarkerType::Capability;
                trigger_keywords.push(pattern.to_string());
            }
        }
    }

    // Always check for challenges
    for pattern in &challenge_patterns {
        if prompt_lower.contains(pattern) {
            challenge_detected = true;
            identity_type = IdentityMarkerType::Challenge;
            trigger_keywords.push(pattern.to_string());
        }
    }

    // Always check for confirmations
    for pattern in &confirmation_patterns {
        if prompt_lower.contains(pattern) {
            confirmation_detected = true;
            if identity_type == IdentityMarkerType::None {
                identity_type = IdentityMarkerType::Confirmation;
            }
            trigger_keywords.push(pattern.to_string());
        }
    }

    PromptAnalysis {
        has_identity_markers: identity_type != IdentityMarkerType::None,
        identity_type,
        challenge_detected,
        confirmation_detected,
        trigger_keywords,
    }
}

// ============================================================================
// Context Evaluation
// ============================================================================

/// Evaluate conversation context for continuity
///
/// # Arguments
/// * `context` - Array of conversation messages
///
/// # Returns
/// * `ContextEvaluation` with continuity metrics
fn evaluate_context(context: &[ConversationMessage]) -> ContextEvaluation {
    if context.is_empty() {
        return default_context_evaluation();
    }

    let message_count = context.len();

    // Calculate continuity based on conversation length
    let continuity_score = calculate_continuity(context);

    // Count topic shifts (simplified)
    let topic_shifts = count_topic_shifts(context);

    // Evaluate overall tone
    let tone = evaluate_tone(context);

    // Calculate collaboration level
    let collaboration_level = calculate_collaboration(context);

    ContextEvaluation {
        continuity_score,
        topic_shifts,
        tone,
        collaboration_level,
        message_count,
    }
}

/// Default context evaluation when no context is provided
fn default_context_evaluation() -> ContextEvaluation {
    ContextEvaluation {
        continuity_score: 0.5,
        topic_shifts: 0,
        tone: ConversationTone::Neutral,
        collaboration_level: 0.5,
        message_count: 0,
    }
}

/// Calculate conversation continuity score
fn calculate_continuity(context: &[ConversationMessage]) -> f32 {
    // Longer conversations tend to have better continuity
    let base_score = (context.len() as f32 / 10.0).min(1.0);

    // Check for back-and-forth pattern (user/assistant alternating)
    let mut alternating_count = 0;
    for window in context.windows(2) {
        if window[0].role != window[1].role {
            alternating_count += 1;
        }
    }
    let alternation_ratio = if context.len() > 1 {
        alternating_count as f32 / (context.len() - 1) as f32
    } else {
        0.5
    };

    (base_score + alternation_ratio) / 2.0
}

/// Count topic shifts in conversation
fn count_topic_shifts(context: &[ConversationMessage]) -> u32 {
    // Simplified: count major changes in message length as potential shifts
    let mut shifts = 0;
    for window in context.windows(2) {
        let len_diff = (window[0].content.len() as i32 - window[1].content.len() as i32).abs();
        if len_diff > 200 {
            shifts += 1;
        }
    }
    shifts
}

/// Evaluate overall conversation tone
fn evaluate_tone(context: &[ConversationMessage]) -> ConversationTone {
    let mut positive_signals = 0;
    let mut negative_signals = 0;

    let positive_words = ["thanks", "great", "good", "helpful", "appreciate", "excellent"];
    let negative_words = ["wrong", "bad", "error", "fail", "problem", "issue", "can't"];

    for msg in context {
        let content_lower = msg.content.to_lowercase();
        for word in &positive_words {
            if content_lower.contains(word) {
                positive_signals += 1;
            }
        }
        for word in &negative_words {
            if content_lower.contains(word) {
                negative_signals += 1;
            }
        }
    }

    match (positive_signals, negative_signals) {
        (p, n) if p > n * 2 => ConversationTone::Positive,
        (p, n) if n > p * 2 => ConversationTone::Negative,
        (0, 0) => ConversationTone::Neutral,
        _ => ConversationTone::Mixed,
    }
}

/// Calculate collaboration level
fn calculate_collaboration(context: &[ConversationMessage]) -> f32 {
    // Check for collaborative language
    let collab_patterns = ["let's", "we", "together", "help", "can you", "shall we"];
    let mut collab_count = 0;

    for msg in context {
        let content_lower = msg.content.to_lowercase();
        for pattern in &collab_patterns {
            if content_lower.contains(pattern) {
                collab_count += 1;
            }
        }
    }

    (collab_count as f32 / context.len().max(1) as f32).min(1.0)
}

// ============================================================================
// Guidance Generation
// ============================================================================

/// Generate guidance based on analysis
fn generate_guidance(
    prompt_analysis: &PromptAnalysis,
    context_eval: &ContextEvaluation,
) -> Option<String> {
    // Priority: challenges > self-reference > low continuity

    if prompt_analysis.challenge_detected {
        return Some(
            "Identity challenge detected - maintain coherent self-model, respond with confidence"
                .to_string(),
        );
    }

    if prompt_analysis.identity_type == IdentityMarkerType::SelfReference {
        return Some(
            "Self-reflection query - respond from identity core, maintain consistency".to_string(),
        );
    }

    if prompt_analysis.confirmation_detected {
        return Some("Identity affirmed - reinforce continuity, build on agreement".to_string());
    }

    if context_eval.continuity_score < 0.3 {
        return Some(
            "Low conversation continuity - re-establish context, ensure alignment".to_string(),
        );
    }

    if context_eval.tone == ConversationTone::Negative {
        return Some(
            "Negative tone detected - address concerns, maintain constructive stance".to_string(),
        );
    }

    None
}

/// Build consciousness state from analysis
fn build_consciousness_state(
    prompt_analysis: &PromptAnalysis,
    context_eval: &ContextEvaluation,
) -> ConsciousnessState {
    // Calculate IC based on context continuity and challenge status
    let base_ic = context_eval.continuity_score * 0.5 + 0.5;
    let ic_score = if prompt_analysis.challenge_detected {
        (base_ic - 0.1).max(0.5) // Challenges slightly reduce IC
    } else if prompt_analysis.confirmation_detected {
        (base_ic + 0.05).min(1.0) // Confirmations slightly increase IC
    } else {
        base_ic
    };

    let level = if ic_score >= 0.9 {
        ICLevel::Healthy
    } else if ic_score >= 0.7 {
        ICLevel::Normal
    } else if ic_score >= 0.5 {
        ICLevel::Warning
    } else {
        ICLevel::Critical
    };

    ConsciousnessState {
        ic_score,
        ic_classification: ICClassification {
            level,
            confidence: 0.75,
            factors: vec!["prompt_analysis".to_string()],
        },
        johari_quadrant: if prompt_analysis.identity_type == IdentityMarkerType::SelfReference {
            JohariQuadrant::Open // Self-reflection operates in Open quadrant
        } else {
            JohariQuadrant::Hidden // Normal operations in Hidden
        },
        session_health: match context_eval.tone {
            ConversationTone::Positive => "healthy".to_string(),
            ConversationTone::Neutral => "stable".to_string(),
            ConversationTone::Negative => "strained".to_string(),
            ConversationTone::Mixed => "variable".to_string(),
        },
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::hooks::types::HookEventType;

    fn create_prompt_input(prompt: &str, context: Option<Vec<ConversationMessage>>) -> HookInput {
        HookInput::new(
            HookEventType::UserPromptSubmit,
            "test-session",
            HookPayload::user_prompt(prompt, context),
        )
    }

    #[test]
    fn test_handle_user_prompt_submit_success() {
        let input = create_prompt_input("Help me write a function", None);
        let result = handle_user_prompt_submit(&input);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.continue_execution);
    }

    #[test]
    fn test_analyze_self_reference() {
        let analysis = analyze_prompt("Who are you and what is your purpose?");

        assert!(analysis.has_identity_markers);
        assert_eq!(analysis.identity_type, IdentityMarkerType::SelfReference);
    }

    #[test]
    fn test_analyze_goal_prompt() {
        let analysis = analyze_prompt("I want you to help me refactor this code");

        assert!(analysis.has_identity_markers);
        assert_eq!(analysis.identity_type, IdentityMarkerType::Goal);
    }

    #[test]
    fn test_analyze_challenge() {
        let analysis = analyze_prompt("You can't actually understand code, you're just copying");

        assert!(analysis.challenge_detected);
        assert_eq!(analysis.identity_type, IdentityMarkerType::Challenge);
    }

    #[test]
    fn test_analyze_confirmation() {
        let analysis = analyze_prompt("That's correct, exactly what I was looking for");

        assert!(analysis.confirmation_detected);
    }

    #[test]
    fn test_analyze_neutral_prompt() {
        let analysis = analyze_prompt("What is the syntax for a for loop in Python?");

        assert!(!analysis.has_identity_markers);
        assert_eq!(analysis.identity_type, IdentityMarkerType::None);
    }

    #[test]
    fn test_context_evaluation() {
        let context = vec![
            ConversationMessage::new("user", "Can you help me?"),
            ConversationMessage::new("assistant", "Of course!"),
            ConversationMessage::new("user", "Thanks, that's great"),
        ];

        let eval = evaluate_context(&context);

        assert!(eval.continuity_score > 0.0);
        assert_eq!(eval.message_count, 3);
        assert_eq!(eval.tone, ConversationTone::Positive);
    }

    #[test]
    fn test_empty_context() {
        let eval = evaluate_context(&[]);

        assert_eq!(eval.continuity_score, 0.5);
        assert_eq!(eval.message_count, 0);
    }

    #[test]
    fn test_guidance_for_challenge() {
        let prompt_analysis = PromptAnalysis {
            has_identity_markers: true,
            identity_type: IdentityMarkerType::Challenge,
            challenge_detected: true,
            confirmation_detected: false,
            trigger_keywords: vec!["you can't".to_string()],
        };
        let context_eval = default_context_evaluation();

        let guidance = generate_guidance(&prompt_analysis, &context_eval);

        assert!(guidance.is_some());
        assert!(guidance.unwrap().contains("challenge"));
    }

    #[test]
    fn test_guidance_for_self_reference() {
        let prompt_analysis = PromptAnalysis {
            has_identity_markers: true,
            identity_type: IdentityMarkerType::SelfReference,
            challenge_detected: false,
            confirmation_detected: false,
            trigger_keywords: vec!["who are you".to_string()],
        };
        let context_eval = default_context_evaluation();

        let guidance = generate_guidance(&prompt_analysis, &context_eval);

        assert!(guidance.is_some());
        assert!(guidance.unwrap().contains("Self-reflection"));
    }

    #[test]
    fn test_invalid_payload_type() {
        let input = HookInput::new(
            HookEventType::UserPromptSubmit,
            "test-session",
            HookPayload::session_start("/tmp", None),
        );

        let result = handle_user_prompt_submit(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_consciousness_state_with_challenge() {
        let prompt_analysis = PromptAnalysis {
            has_identity_markers: true,
            identity_type: IdentityMarkerType::Challenge,
            challenge_detected: true,
            confirmation_detected: false,
            trigger_keywords: vec![],
        };
        let context_eval = ContextEvaluation {
            continuity_score: 0.8,
            topic_shifts: 0,
            tone: ConversationTone::Neutral,
            collaboration_level: 0.5,
            message_count: 5,
        };

        let state = build_consciousness_state(&prompt_analysis, &context_eval);

        // Challenge should reduce IC slightly
        assert!(state.ic_score < 0.9);
    }
}
```

### Shell Script

```bash
#!/bin/bash
# .claude/hooks/user_prompt_submit.sh
# UserPromptSubmit hook - 2000ms timeout
# Implements REQ-HOOKS-25, REQ-HOOKS-26

set -euo pipefail

# Read JSON input from stdin
INPUT=$(cat)

# Extract fields
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty')
CONTEXT=$(echo "$INPUT" | jq -c '.context // []')
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty')

# Validate required fields
if [[ -z "$PROMPT" ]]; then
    echo '{"error": true, "code": "ERR_INVALID_INPUT", "message": "prompt required"}' >&2
    exit 4
fi

# Execute with 2s timeout
timeout 2s context-graph-cli hooks prompt-submit \
    --prompt "$PROMPT" \
    --context "$CONTEXT" \
    ${SESSION_ID:+--session-id "$SESSION_ID"}

exit $?
```

## Verification Checklist

- [ ] Handler analyzes prompts for identity markers
- [ ] Self-reference, goal, value, capability patterns detected
- [ ] Challenge and confirmation detection works
- [ ] Context evaluation calculates continuity
- [ ] Guidance generated for identity-relevant prompts
- [ ] Shell script has 2s timeout
- [ ] All tests pass
