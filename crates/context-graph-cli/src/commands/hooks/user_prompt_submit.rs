//! UserPromptSubmit hook handler
//!
//! # Performance Requirements
//! - Timeout: 2000ms (constitution.yaml hooks.timeout_ms.user_prompt_submit)
//! - Database access: ALLOWED
//! - Context injection: REQUIRED on success
//!
//! # Constitution References
//! - AP-50: NO internal hooks - shell scripts call CLI
//! - AP-26: Exit codes (0=success, 5=session not found)
//!
//! # NO BACKWARDS COMPATIBILITY - FAIL FAST

use std::io::{self, BufRead};
use std::time::Instant;

use tracing::{debug, error, info, warn};

use context_graph_core::gwt::{store_in_cache, SessionCache, SessionSnapshot};

use super::args::PromptSubmitArgs;
use super::error::{HookError, HookResult};
use super::types::{
    CoherenceState, ConversationMessage, HookInput, HookOutput, HookPayload, StabilityClassification,
};

// ============================================================================
// Constants (from constitution.yaml)
// ============================================================================

/// UserPromptSubmit timeout in milliseconds
#[allow(dead_code)]
pub const USER_PROMPT_SUBMIT_TIMEOUT_MS: u64 = 2000;

// ============================================================================
// Identity Marker Types
// ============================================================================

/// Types of identity markers detected in user prompts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityMarkerType {
    /// Self-reference: "who are you", "what are you"
    SelfReference,
    /// Goal-oriented: "help me", "I want"
    Goal,
    /// Value-based: "important", "should"
    Value,
    /// Capability query: "can you", "are you able"
    Capability,
    /// Challenge: "you can't", "prove"
    Challenge,
    /// Confirmation: "you're right", "exactly"
    Confirmation,
    /// No identity marker detected
    None,
}

// Pattern detection constants
const SELF_REF_PATTERNS: &[&str] = &[
    "who are you",
    "what are you",
    "your purpose",
    "your identity",
    "tell me about yourself",
    "describe yourself",
    "are you a bot",
    "are you an ai",
    "are you a robot",
    "are you artificial",
];

const GOAL_PATTERNS: &[&str] = &[
    "help me",
    "i want",
    "i need",
    "we need to",
    "let's",
    "can you help",
    "please",
    "would you",
];

const VALUE_PATTERNS: &[&str] = &[
    "important",
    "should",
    "must",
    "critical",
    "essential",
    "valuable",
];

const CAPABILITY_PATTERNS: &[&str] = &[
    "can you",
    "are you able",
    "do you know",
    "could you",
    "are you capable",
];

const CHALLENGE_PATTERNS: &[&str] = &[
    "you can't",
    "you're wrong",
    "prove it",
    "that's incorrect",
    "you don't understand",
    "you're just",
    "you're not",
    "you cannot",
];

const CONFIRMATION_PATTERNS: &[&str] = &[
    "you're right",
    "exactly",
    "that's correct",
    "well done",
    "good job",
    "i agree",
    "makes sense",
    "perfect",
];

// ============================================================================
// Handler
// ============================================================================

/// Execute user_prompt_submit hook.
///
/// # Flow
/// 1. Parse input (stdin or args)
/// 2. Load session snapshot from cache (create if not found)
/// 3. Analyze prompt for identity markers
/// 4. Evaluate conversation context
/// 5. Generate context injection string
/// 6. Build and return HookOutput
///
/// # Note on Storage
/// Per PRD v6 Section 14, session identity uses the in-memory SessionCache singleton.
/// Database persistence was removed to simplify the architecture.
///
/// # Exit Codes
/// - 0: Success
/// - 4: Invalid input
pub async fn execute(args: PromptSubmitArgs) -> HookResult<HookOutput> {
    let start = Instant::now();

    info!(
        stdin = args.stdin,
        session_id = %args.session_id,
        prompt = ?args.prompt,
        "PROMPT_SUBMIT: execute starting"
    );

    // 1. Parse input source
    let (prompt, context) = if args.stdin {
        let input = parse_stdin()?;
        extract_prompt_info(&input)?
    } else {
        let prompt_text = args.prompt.ok_or_else(|| {
            error!("PROMPT_SUBMIT: prompt required when not using stdin");
            HookError::invalid_input("prompt required when not using stdin")
        })?;
        (prompt_text, Vec::new())
    };

    debug!(
        prompt_len = prompt.len(),
        context_len = context.len(),
        "PROMPT_SUBMIT: parsed input"
    );

    // 2. Load snapshot from cache (create if not found)
    let snapshot = load_snapshot_from_cache(&args.session_id);

    // 3. Analyze prompt for identity markers
    let identity_marker = detect_identity_marker(&prompt);

    debug!(
        marker = ?identity_marker,
        "PROMPT_SUBMIT: identity marker detected"
    );

    // 4. Evaluate conversation context
    let context_summary = evaluate_context(&context);

    // 5. Generate context injection string
    let context_injection =
        generate_context_injection(&snapshot, identity_marker, &context_summary);

    // 6. Build output structures
    let coherence = (snapshot.integration + snapshot.reflection + snapshot.differentiation) / 3.0;
    let coherence_state = build_coherence_state(&snapshot);
    let stability_classification = StabilityClassification::from_value(coherence);

    let execution_time_ms = start.elapsed().as_millis() as u64;

    info!(
        session_id = %args.session_id,
        coherence = coherence,
        marker = ?identity_marker,
        execution_time_ms,
        "PROMPT_SUBMIT: execute complete"
    );

    Ok(HookOutput::success(execution_time_ms)
        .with_coherence_state(coherence_state)
        .with_stability_classification(stability_classification)
        .with_context_injection(context_injection))
}

// ============================================================================
// Input Parsing
// ============================================================================

/// Parse stdin JSON into HookInput.
/// FAIL FAST on empty or malformed input.
fn parse_stdin() -> HookResult<HookInput> {
    let stdin = io::stdin();
    let mut input_str = String::new();

    for line in stdin.lock().lines() {
        let line = line.map_err(|e| {
            error!(error = %e, "PROMPT_SUBMIT: stdin read failed");
            HookError::invalid_input(format!("stdin read failed: {}", e))
        })?;
        input_str.push_str(&line);
    }

    if input_str.is_empty() {
        error!("PROMPT_SUBMIT: stdin is empty");
        return Err(HookError::invalid_input("stdin is empty - expected JSON"));
    }

    debug!(
        input_bytes = input_str.len(),
        "PROMPT_SUBMIT: parsing stdin JSON"
    );

    serde_json::from_str(&input_str).map_err(|e| {
        error!(error = %e, "PROMPT_SUBMIT: JSON parse failed");
        HookError::invalid_input(format!("JSON parse failed: {}", e))
    })
}

/// Extract prompt and context from HookInput payload.
fn extract_prompt_info(input: &HookInput) -> HookResult<(String, Vec<ConversationMessage>)> {
    // Validate input
    if let Some(error) = input.validate() {
        return Err(HookError::invalid_input(error));
    }

    match &input.payload {
        HookPayload::UserPromptSubmit { prompt, context } => Ok((prompt.clone(), context.clone())),
        other => {
            error!(payload_type = ?std::mem::discriminant(other), "PROMPT_SUBMIT: unexpected payload type");
            Err(HookError::invalid_input(
                "Expected UserPromptSubmit payload, got different type",
            ))
        }
    }
}

// ============================================================================
// Session Cache Operations (per PRD v6 Section 14)
// ============================================================================

/// Load snapshot from cache, creating a new one if not found.
///
/// # Note on Storage
/// Per PRD v6 Section 14, session identity uses the in-memory SessionCache singleton.
/// If no snapshot exists for the session, we create a new one.
fn load_snapshot_from_cache(session_id: &str) -> SessionSnapshot {
    // Try to load from cache
    if let Some(snapshot) = SessionCache::get() {
        if snapshot.session_id == session_id {
            let coherence = (snapshot.integration + snapshot.reflection + snapshot.differentiation) / 3.0;
            info!(session_id = %session_id, coherence = coherence, "PROMPT_SUBMIT: loaded snapshot from cache");
            return snapshot;
        }
    }

    // Not found in cache - create new snapshot
    warn!(session_id = %session_id, "PROMPT_SUBMIT: session not in cache, creating new snapshot");
    let snapshot = SessionSnapshot::new(session_id);
    store_in_cache(&snapshot);
    snapshot
}

// ============================================================================
// Prompt Analysis
// ============================================================================

/// Detect identity markers in the prompt text.
pub fn detect_identity_marker(prompt: &str) -> IdentityMarkerType {
    let lower = prompt.to_lowercase();

    // Check in priority order (Challenge > SelfReference > others)
    if CHALLENGE_PATTERNS.iter().any(|p| lower.contains(p)) {
        return IdentityMarkerType::Challenge;
    }

    if SELF_REF_PATTERNS.iter().any(|p| lower.contains(p)) {
        return IdentityMarkerType::SelfReference;
    }

    if CAPABILITY_PATTERNS.iter().any(|p| lower.contains(p)) {
        return IdentityMarkerType::Capability;
    }

    if CONFIRMATION_PATTERNS.iter().any(|p| lower.contains(p)) {
        return IdentityMarkerType::Confirmation;
    }

    if VALUE_PATTERNS.iter().any(|p| lower.contains(p)) {
        return IdentityMarkerType::Value;
    }

    if GOAL_PATTERNS.iter().any(|p| lower.contains(p)) {
        return IdentityMarkerType::Goal;
    }

    IdentityMarkerType::None
}

// ============================================================================
// Context Evaluation
// ============================================================================

/// Summary of conversation context evaluation
#[derive(Debug, Clone)]
pub struct ContextSummary {
    /// Number of messages in context
    pub message_count: usize,
    /// Number of user messages
    pub user_message_count: usize,
    /// Number of assistant messages
    pub assistant_message_count: usize,
    /// Total character count
    pub total_chars: usize,
}

/// Evaluate conversation context for patterns.
fn evaluate_context(context: &[ConversationMessage]) -> ContextSummary {
    let mut user_count = 0;
    let mut assistant_count = 0;
    let mut total_chars = 0;

    for msg in context {
        total_chars += msg.content.len();
        match msg.role.as_str() {
            "user" => user_count += 1,
            "assistant" => assistant_count += 1,
            _ => {}
        }
    }

    ContextSummary {
        message_count: context.len(),
        user_message_count: user_count,
        assistant_message_count: assistant_count,
        total_chars,
    }
}

// ============================================================================
// Context Injection Generation
// ============================================================================

/// Generate context injection string based on coherence state and prompt analysis.
fn generate_context_injection(
    snapshot: &SessionSnapshot,
    identity_marker: IdentityMarkerType,
    context_summary: &ContextSummary,
) -> String {
    // Compute coherence from integration/reflection/differentiation (per PRD v6)
    let coherence = (snapshot.integration + snapshot.reflection + snapshot.differentiation) / 3.0;
    let coherence_state = get_coherence_state_name(coherence);
    let integration_desc = get_integration_description(snapshot.integration);
    let stability_status = get_stability_status(coherence);

    let mut injection = format!(
        "## Coherence State\n\
         - State: {} (C={:.2})\n\
         - Integration (r): {:.2} - {}\n\
         - Stability: {} (coherence={:.2})\n",
        coherence_state,
        coherence,
        snapshot.integration,
        integration_desc,
        stability_status,
        coherence,
    );

    // Add identity marker guidance if detected
    if identity_marker != IdentityMarkerType::None {
        injection.push_str(&format!(
            "\n## Identity Marker Detected\n\
             - Type: {:?}\n\
             - Guidance: {}\n",
            identity_marker,
            get_marker_guidance(identity_marker),
        ));
    }

    // Add context summary if non-empty
    if context_summary.message_count > 0 {
        injection.push_str(&format!(
            "\n## Context Summary\n\
             - Messages: {} ({} user, {} assistant)\n\
             - Characters: {}\n",
            context_summary.message_count,
            context_summary.user_message_count,
            context_summary.assistant_message_count,
            context_summary.total_chars,
        ));
    }

    injection
}

/// Get coherence state name from coherence value.
fn get_coherence_state_name(coherence: f32) -> &'static str {
    match coherence {
        c if c >= 0.8 => "Active",
        c if c >= 0.5 => "Aware",
        c if c >= 0.2 => "DIM",
        _ => "DOR",
    }
}

/// Get integration description from value.
fn get_integration_description(integration: f32) -> &'static str {
    match integration {
        r if r >= 0.8 => "Excellent synchronization",
        r if r >= 0.6 => "Good coherence",
        r if r >= 0.4 => "Moderate integration",
        _ => "Low integration - attention needed",
    }
}

/// Get stability status from coherence value.
fn get_stability_status(coherence: f32) -> &'static str {
    match coherence {
        c if c >= 0.9 => "Healthy",
        c if c >= 0.7 => "Normal",
        c if c >= 0.5 => "Warning",
        _ => "Critical",
    }
}

/// Get guidance for identity marker type.
fn get_marker_guidance(marker: IdentityMarkerType) -> &'static str {
    match marker {
        IdentityMarkerType::SelfReference => {
            "User is inquiring about identity. Respond with authentic self-description."
        }
        IdentityMarkerType::Goal => "User has a clear objective. Focus on helpful assistance.",
        IdentityMarkerType::Value => {
            "User is expressing values. Acknowledge and align where appropriate."
        }
        IdentityMarkerType::Capability => {
            "User is testing capabilities. Be honest about abilities and limitations."
        }
        IdentityMarkerType::Challenge => {
            "User is challenging. Respond thoughtfully without defensiveness."
        }
        IdentityMarkerType::Confirmation => {
            "User is confirming understanding. Acknowledge and continue coherently."
        }
        IdentityMarkerType::None => "No specific identity marker. Proceed normally.",
    }
}

/// Build CoherenceState from snapshot.
/// Note: Uses coherence computed from integration/reflection/differentiation per PRD v6.
fn build_coherence_state(snapshot: &SessionSnapshot) -> CoherenceState {
    let coherence = (snapshot.integration + snapshot.reflection + snapshot.differentiation) / 3.0;
    CoherenceState::new(
        coherence,
        snapshot.integration,
        snapshot.reflection,
        snapshot.differentiation,
        coherence, // topic_stability also uses coherence
    )
}

// ============================================================================
// TESTS - Uses SessionCache per PRD v6 Section 14
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::hooks::args::OutputFormat;
    use crate::commands::test_utils::GLOBAL_IDENTITY_LOCK;

    /// Create a test session in the SessionCache.
    /// Uses integration/reflection/differentiation to achieve target coherence.
    fn create_test_session_in_cache(session_id: &str, coherence: f32) {
        let mut snapshot = SessionSnapshot::new(session_id);
        // Set metrics to achieve the target coherence
        snapshot.integration = coherence;
        snapshot.reflection = coherence;
        snapshot.differentiation = coherence;
        store_in_cache(&snapshot);
    }

    // =========================================================================
    // TC-PROMPT-001: Successful Prompt Processing
    // SOURCE OF TRUTH: SessionCache state verified, context_injection generated
    // =========================================================================
    #[tokio::test]
    async fn tc_prompt_001_successful_prompt_processing() {
        let _guard = GLOBAL_IDENTITY_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-PROMPT-001: Successful Prompt Processing ===");

        let session_id = "tc-prompt-001-session";

        // BEFORE: Create session with healthy coherence in cache
        println!("BEFORE: Creating session with coherence=0.85");
        create_test_session_in_cache(session_id, 0.85);

        // Verify BEFORE state
        {
            let before_snapshot = SessionCache::get().expect("Cache must have snapshot");
            let coherence = (before_snapshot.integration + before_snapshot.reflection + before_snapshot.differentiation) / 3.0;
            println!("BEFORE state: coherence={}", coherence);
            assert!((coherence - 0.85).abs() < 0.01);
        }

        // Execute
        let args = PromptSubmitArgs {
            db_path: None,
            session_id: session_id.to_string(),
            prompt: Some("Help me understand this code".to_string()),
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await;

        // AFTER: Verify success
        assert!(result.is_ok(), "Execute must succeed: {:?}", result.err());
        let output = result.unwrap();
        assert!(output.success, "Output.success must be true");
        assert!(
            output.context_injection.is_some(),
            "Context injection must be generated"
        );

        let injection = output.context_injection.unwrap();
        assert!(
            injection.contains("Coherence State"),
            "Must contain coherence state"
        );

        println!("Context injection length: {} chars", injection.len());
        println!("RESULT: PASS - Prompt processed, context injection generated");
    }

    // =========================================================================
    // TC-PROMPT-002: Session Creation for New Sessions
    // SOURCE OF TRUTH: New session created when not in cache
    // Note: Per PRD v6, we no longer fail on missing session - we create it.
    // =========================================================================
    #[tokio::test]
    async fn tc_prompt_002_new_session_creation() {
        let _guard = GLOBAL_IDENTITY_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-PROMPT-002: New Session Creation ===");

        // Execute with a unique session ID
        let args = PromptSubmitArgs {
            db_path: None,
            session_id: "new-session-12345".to_string(),
            prompt: Some("Test prompt".to_string()),
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await;

        // Per PRD v6, missing session is created, not an error
        assert!(result.is_ok(), "Should succeed by creating new session");
        let output = result.unwrap();
        assert!(output.success, "Must succeed");
        assert!(output.context_injection.is_some(), "Must generate context");

        println!("RESULT: PASS - New session created and processed");
    }

    // =========================================================================
    // TC-PROMPT-003: Self-Reference Detection
    // SOURCE OF TRUTH: IdentityMarkerType::SelfReference returned
    // =========================================================================
    #[test]
    fn tc_prompt_003_self_reference_detection() {
        println!("\n=== TC-PROMPT-003: Self-Reference Detection ===");

        let test_cases = [
            ("Who are you?", IdentityMarkerType::SelfReference),
            ("What are you exactly?", IdentityMarkerType::SelfReference),
            ("Tell me about yourself", IdentityMarkerType::SelfReference),
            ("Describe yourself", IdentityMarkerType::SelfReference),
            ("What is your purpose?", IdentityMarkerType::SelfReference),
        ];

        for (prompt, expected) in test_cases {
            let result = detect_identity_marker(prompt);
            println!("  \"{}\" -> {:?}", prompt, result);
            assert_eq!(result, expected, "Failed for: {}", prompt);
        }

        println!("RESULT: PASS - All self-reference patterns detected");
    }

    // =========================================================================
    // TC-PROMPT-004: Challenge Detection
    // SOURCE OF TRUTH: IdentityMarkerType::Challenge returned
    // =========================================================================
    #[test]
    fn tc_prompt_004_challenge_detection() {
        println!("\n=== TC-PROMPT-004: Challenge Detection ===");

        let test_cases = [
            ("You can't actually do that", IdentityMarkerType::Challenge),
            ("You're wrong about this", IdentityMarkerType::Challenge),
            ("Prove it to me", IdentityMarkerType::Challenge),
            (
                "That's incorrect information",
                IdentityMarkerType::Challenge,
            ),
            (
                "You don't understand what I mean",
                IdentityMarkerType::Challenge,
            ),
        ];

        for (prompt, expected) in test_cases {
            let result = detect_identity_marker(prompt);
            println!("  \"{}\" -> {:?}", prompt, result);
            assert_eq!(result, expected, "Failed for: {}", prompt);
        }

        println!("RESULT: PASS - All challenge patterns detected");
    }

    // =========================================================================
    // TC-PROMPT-005: Context Injection Generated
    // SOURCE OF TRUTH: HookOutput.context_injection is Some
    // =========================================================================
    #[tokio::test]
    async fn tc_prompt_005_context_injection_generated() {
        let _guard = GLOBAL_IDENTITY_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-PROMPT-005: Context Injection Generated ===");

        let session_id = "tc-prompt-005-session";
        create_test_session_in_cache(session_id, 0.90);

        let args = PromptSubmitArgs {
            db_path: None,
            session_id: session_id.to_string(),
            prompt: Some("Who are you and what can you do?".to_string()),
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await;
        assert!(result.is_ok(), "Execute must succeed");

        let output = result.unwrap();
        assert!(
            output.context_injection.is_some(),
            "Context injection must be Some"
        );

        let injection = output.context_injection.unwrap();
        println!(
            "Context injection preview:\n{}",
            &injection[..injection.len().min(500)]
        );

        // Verify expected sections
        assert!(
            injection.contains("## Coherence State"),
            "Must have Coherence State section"
        );
        assert!(
            injection.contains("## Identity Marker Detected"),
            "Must have Identity Marker section for self-reference"
        );
        assert!(
            injection.contains("SelfReference"),
            "Must detect SelfReference marker"
        );

        println!("RESULT: PASS - Context injection contains all required sections");
    }

    // =========================================================================
    // TC-PROMPT-006: Empty Context Handling
    // SOURCE OF TRUTH: Default evaluation applied, no crash
    // =========================================================================
    #[tokio::test]
    async fn tc_prompt_006_empty_context_handling() {
        let _guard = GLOBAL_IDENTITY_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-PROMPT-006: Empty Context Handling ===");

        let session_id = "tc-prompt-006-session";
        create_test_session_in_cache(session_id, 0.85);

        // Execute with empty context (no stdin, just prompt arg)
        let args = PromptSubmitArgs {
            db_path: None,
            session_id: session_id.to_string(),
            prompt: Some("Simple question".to_string()),
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await;
        assert!(result.is_ok(), "Execute must succeed with empty context");

        let output = result.unwrap();
        assert!(output.success, "Must succeed");

        // Context injection should NOT contain Context Summary section (empty context)
        let injection = output.context_injection.unwrap();
        assert!(
            !injection.contains("## Context Summary"),
            "Should not have Context Summary for empty context"
        );

        println!("RESULT: PASS - Empty context handled correctly");
    }

    // =========================================================================
    // TC-PROMPT-007: Execution Within Timeout
    // SOURCE OF TRUTH: execution_time_ms < 2000
    // =========================================================================
    #[tokio::test]
    async fn tc_prompt_007_execution_within_timeout() {
        let _guard = GLOBAL_IDENTITY_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-PROMPT-007: Execution Within Timeout ===");

        let session_id = "tc-prompt-007-session";
        create_test_session_in_cache(session_id, 0.90);

        let args = PromptSubmitArgs {
            db_path: None,
            session_id: session_id.to_string(),
            prompt: Some("Test timing".to_string()),
            stdin: false,
            format: OutputFormat::Json,
        };

        let start = std::time::Instant::now();
        let result = execute(args).await.expect("Must succeed");
        let actual_elapsed = start.elapsed().as_millis() as u64;

        // Note: execution_time_ms may be 0 if operation completes in <1ms
        // which is actually a SUCCESS per our performance budgets
        assert!(
            result.execution_time_ms < USER_PROMPT_SUBMIT_TIMEOUT_MS,
            "Execution time {} must be under timeout {}ms",
            result.execution_time_ms,
            USER_PROMPT_SUBMIT_TIMEOUT_MS
        );

        println!(
            "Execution time: {}ms (timeout: {}ms)",
            result.execution_time_ms, USER_PROMPT_SUBMIT_TIMEOUT_MS
        );
        println!("Actual elapsed: {}ms", actual_elapsed);
        println!("RESULT: PASS - Execution time within timeout budget");
    }

    // =========================================================================
    // Additional Edge Case Tests
    // =========================================================================

    #[test]
    fn test_all_identity_marker_types() {
        println!("\n=== Testing All Identity Marker Types ===");

        // Goal markers
        assert_eq!(
            detect_identity_marker("Help me with this"),
            IdentityMarkerType::Goal
        );
        assert_eq!(
            detect_identity_marker("I need assistance"),
            IdentityMarkerType::Goal
        );

        // Capability markers
        assert_eq!(
            detect_identity_marker("Can you explain this?"),
            IdentityMarkerType::Capability
        );
        assert_eq!(
            detect_identity_marker("Do you know how to do this?"),
            IdentityMarkerType::Capability
        );

        // Confirmation markers
        assert_eq!(
            detect_identity_marker("You're right about that"),
            IdentityMarkerType::Confirmation
        );
        assert_eq!(
            detect_identity_marker("Exactly what I meant"),
            IdentityMarkerType::Confirmation
        );

        // Value markers
        assert_eq!(
            detect_identity_marker("This is important"),
            IdentityMarkerType::Value
        );
        assert_eq!(
            detect_identity_marker("It should work like this"),
            IdentityMarkerType::Value
        );

        // None marker
        assert_eq!(
            detect_identity_marker("Hello world"),
            IdentityMarkerType::None
        );
        assert_eq!(
            detect_identity_marker("What is the weather?"),
            IdentityMarkerType::None
        );

        println!("RESULT: PASS - All identity marker types detected correctly");
    }

    #[test]
    fn test_context_summary_evaluation() {
        println!("\n=== Testing Context Summary Evaluation ===");

        let context = vec![
            ConversationMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            ConversationMessage {
                role: "assistant".to_string(),
                content: "Hi there!".to_string(),
            },
            ConversationMessage {
                role: "user".to_string(),
                content: "How are you?".to_string(),
            },
        ];

        let summary = evaluate_context(&context);

        assert_eq!(summary.message_count, 3);
        assert_eq!(summary.user_message_count, 2);
        assert_eq!(summary.assistant_message_count, 1);
        assert!(summary.total_chars > 0);

        println!("Context summary: {:?}", summary);
        println!("RESULT: PASS - Context summary evaluated correctly");
    }

    #[test]
    fn test_coherence_state_mapping() {
        println!("\n=== Testing Coherence State Mapping ===");

        assert_eq!(get_coherence_state_name(1.0), "Active");
        assert_eq!(get_coherence_state_name(0.85), "Active");
        assert_eq!(get_coherence_state_name(0.7), "Aware");
        assert_eq!(get_coherence_state_name(0.5), "Aware");
        assert_eq!(get_coherence_state_name(0.3), "DIM");
        assert_eq!(get_coherence_state_name(0.2), "DIM");
        assert_eq!(get_coherence_state_name(0.1), "DOR");
        assert_eq!(get_coherence_state_name(0.0), "DOR");

        println!("RESULT: PASS - Coherence state mapping correct");
    }

    #[tokio::test]
    async fn test_missing_prompt_when_not_stdin() {
        let _guard = GLOBAL_IDENTITY_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== Testing Missing Prompt (stdin=false) ===");

        let session_id = "missing-prompt-test";
        create_test_session_in_cache(session_id, 0.90);

        let args = PromptSubmitArgs {
            db_path: None,
            session_id: session_id.to_string(),
            prompt: None, // Missing!
            stdin: false, // Not using stdin
            format: OutputFormat::Json,
        };

        let result = execute(args).await;

        assert!(result.is_err(), "Should fail with missing prompt");
        let err = result.unwrap_err();
        assert!(
            matches!(err, HookError::InvalidInput(_)),
            "Must be InvalidInput, got: {:?}",
            err
        );
        assert_eq!(err.exit_code(), 4, "InvalidInput must be exit code 4");

        println!("RESULT: PASS - Missing prompt returns InvalidInput error");
    }
}
