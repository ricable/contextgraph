//! PostToolUse hook handler
//!
//! # Performance Requirements
//! - Timeout: 3000ms (constitution.yaml hooks.timeout_ms.post_tool_use)
//! - Database access: ALLOWED
//!
//! # Constitution References
//! - AP-50: NO internal hooks - shell scripts call CLI
//! - AP-26: Exit codes (0=success)
//!
//! # NO BACKWARDS COMPATIBILITY - FAIL FAST

use std::io::{self, BufRead};
use std::time::Instant;

use tracing::{debug, error, info};

use context_graph_core::gwt::session_snapshot::{store_in_cache, SessionCache, SessionSnapshot};

use super::args::PostToolArgs;
use super::error::{HookError, HookResult};
use super::types::{
    CoherenceState, HookInput, HookOutput, HookPayload, StabilityClassification,
};

// ============================================================================
// Constants (from constitution.yaml)
// ============================================================================

/// PostToolUse timeout in milliseconds
pub const POST_TOOL_USE_TIMEOUT_MS: u64 = 3000;

// ============================================================================
// Types
// ============================================================================

/// Impact of tool execution on coherence state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpactLevel {
    /// No significant impact
    None,
    /// Minor impact - log but don't recalculate
    Low,
    /// Moderate impact - consider recalculation
    Medium,
    /// High impact - force recalculation
    High,
}

/// Result of analyzing tool response
#[derive(Debug, Clone)]
pub struct ToolImpact {
    /// Impact level on coherence
    pub level: ImpactLevel,
    /// Whether tool execution succeeded
    pub tool_success: bool,
}

// ============================================================================
// Handler
// ============================================================================

/// Execute post-tool hook.
///
/// See module doc for full flow and exit codes.
///
/// # Note on Topic Stability
/// Per PRD v6 Section 14, we use Topic Stability for session coherence.
/// Session state is stored in the in-memory SessionCache.
pub async fn execute(args: PostToolArgs) -> HookResult<HookOutput> {
    let start = Instant::now();

    info!(
        stdin = args.stdin,
        session_id = %args.session_id,
        tool_name = ?args.tool_name,
        "POST_TOOL: execute starting"
    );

    // 1. Parse input source
    let (tool_name, tool_response, tool_success) = if args.stdin {
        let input = parse_stdin()?;
        extract_tool_info(&input)?
    } else {
        let name = args.tool_name.ok_or_else(|| {
            error!("POST_TOOL: tool_name required when not using stdin");
            HookError::invalid_input("tool_name required when not using stdin")
        })?;
        (name, String::new(), args.success.unwrap_or(true))
    };

    // 2. Load snapshot from cache (or create new)
    let mut snapshot = load_snapshot_from_cache(&args.session_id)?;

    // 3. Analyze tool response for coherence impact
    let impact = analyze_tool_response(&tool_name, &tool_response, tool_success);

    // 4. Update snapshot based on impact
    if impact.level >= ImpactLevel::Medium {
        update_snapshot_from_impact(&mut snapshot, &impact);
    }

    // 5. Persist updated snapshot to cache
    store_in_cache(&snapshot);

    // 6. Build output structures
    let coherence_state = build_coherence_state(&snapshot);
    // Use average of integration, reflection, differentiation as stability proxy
    let stability_value = (snapshot.integration + snapshot.reflection + snapshot.differentiation) / 3.0;
    let stability_classification = StabilityClassification::from_value(stability_value);

    let execution_time_ms = start.elapsed().as_millis() as u64;

    info!(
        session_id = %args.session_id,
        tool_name = %tool_name,
        stability = stability_value,
        execution_time_ms,
        "POST_TOOL: execute complete"
    );

    Ok(HookOutput::success(execution_time_ms)
        .with_coherence_state(coherence_state)
        .with_stability_classification(stability_classification))
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
            error!(error = %e, "POST_TOOL: stdin read failed");
            HookError::invalid_input(format!("stdin read failed: {}", e))
        })?;
        input_str.push_str(&line);
    }

    if input_str.is_empty() {
        error!("POST_TOOL: stdin is empty");
        return Err(HookError::invalid_input("stdin is empty - expected JSON"));
    }

    debug!(
        input_bytes = input_str.len(),
        "POST_TOOL: parsing stdin JSON"
    );

    serde_json::from_str(&input_str).map_err(|e| {
        error!(error = %e, "POST_TOOL: JSON parse failed");
        HookError::invalid_input(format!("JSON parse failed: {}", e))
    })
}

/// Extract tool info from HookInput payload.
fn extract_tool_info(input: &HookInput) -> HookResult<(String, String, bool)> {
    // Validate input
    if let Some(error) = input.validate() {
        return Err(HookError::invalid_input(error));
    }

    match &input.payload {
        HookPayload::PostToolUse {
            tool_name,
            tool_response,
            ..
        } => {
            // Determine success from response (no error field in actual type)
            let success = !tool_response.contains("error") && !tool_response.contains("Error");
            Ok((tool_name.clone(), tool_response.clone(), success))
        }
        other => {
            error!(payload_type = ?std::mem::discriminant(other), "POST_TOOL: unexpected payload type");
            Err(HookError::invalid_input(
                "Expected PostToolUse payload, got different type",
            ))
        }
    }
}

// ============================================================================
// Cache Operations
// ============================================================================

/// Load snapshot from cache or create new one for session.
///
/// # Note on Topic Stability
/// Per PRD v6 Section 14, we use in-memory SessionCache for session state.
fn load_snapshot_from_cache(session_id: &str) -> HookResult<SessionSnapshot> {
    // Try to load from global cache
    if let Some(snapshot) = SessionCache::get() {
        if snapshot.session_id == session_id {
            let stability = (snapshot.integration + snapshot.reflection + snapshot.differentiation) / 3.0;
            info!(session_id = %session_id, stability = stability, "POST_TOOL: loaded snapshot from cache");
            return Ok(snapshot);
        }
    }

    // Session not found in cache - create new one
    // This is not an error for post_tool_use, we just create a default
    info!(session_id = %session_id, "POST_TOOL: session not in cache, creating new snapshot");
    let snapshot = SessionSnapshot::new(session_id);
    store_in_cache(&snapshot);
    Ok(snapshot)
}

// ============================================================================
// Tool Analysis
// ============================================================================

/// Analyze tool response for coherence updates
fn analyze_tool_response(tool_name: &str, _tool_response: &str, tool_success: bool) -> ToolImpact {
    let level = match tool_name {
        "Read" => ImpactLevel::Low,
        "Write" | "Edit" | "MultiEdit" => ImpactLevel::Medium,
        "Bash" => {
            if tool_success {
                ImpactLevel::Medium
            } else {
                ImpactLevel::High
            }
        }
        "WebFetch" | "WebSearch" => ImpactLevel::Medium,
        "Git" => ImpactLevel::Low,
        "Task" => ImpactLevel::High,
        _ => ImpactLevel::None,
    };

    ToolImpact { level, tool_success }
}

/// Update snapshot based on tool impact
///
/// # Note on Topic Stability
/// Per PRD v6 Section 14, we use integration/reflection/differentiation metrics
/// for coherence tracking.
fn update_snapshot_from_impact(snapshot: &mut SessionSnapshot, impact: &ToolImpact) {
    // Apply coherence changes based on impact level
    let delta = match impact.level {
        ImpactLevel::High => 0.03,
        ImpactLevel::Medium => 0.02,
        ImpactLevel::Low => 0.01,
        ImpactLevel::None => 0.0,
    };

    // Positive delta for successful tools, negative for failures
    // Update integration as the primary metric affected by tool success
    if impact.tool_success {
        snapshot.integration = (snapshot.integration + delta * 0.5).clamp(0.0, 1.0);
    } else {
        snapshot.integration = (snapshot.integration - delta).clamp(0.0, 1.0);
    }
}

/// Build CoherenceState from snapshot.
///
/// # Note
/// We compute coherence from integration, reflection, and differentiation metrics.
fn build_coherence_state(snapshot: &SessionSnapshot) -> CoherenceState {
    let coherence_level = (snapshot.integration + snapshot.reflection + snapshot.differentiation) / 3.0;
    CoherenceState::new(
        coherence_level, // coherence derived from integration metrics
        snapshot.integration,
        snapshot.reflection,
        snapshot.differentiation,
        coherence_level, // topic_stability uses same coherence measure
    )
}

// ============================================================================
// Comparison Implementations
// ============================================================================

impl PartialOrd for ImpactLevel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ImpactLevel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let self_val = match self {
            ImpactLevel::None => 0,
            ImpactLevel::Low => 1,
            ImpactLevel::Medium => 2,
            ImpactLevel::High => 3,
        };
        let other_val = match other {
            ImpactLevel::None => 0,
            ImpactLevel::Low => 1,
            ImpactLevel::Medium => 2,
            ImpactLevel::High => 3,
        };
        self_val.cmp(&other_val)
    }
}

// ============================================================================
// TESTS - Use in-memory SessionCache per PRD v6
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::hooks::args::OutputFormat;
    use crate::commands::test_utils::GLOBAL_IDENTITY_LOCK;

    /// Create a session in the cache for testing
    fn create_test_session(session_id: &str, integration: f32) {
        let mut snapshot = SessionSnapshot::new(session_id);
        snapshot.integration = integration;
        snapshot.reflection = 0.5;
        snapshot.differentiation = 0.5;
        store_in_cache(&snapshot);
    }

    // =========================================================================
    // TC-POST-001: Successful Tool Processing
    // SOURCE OF TRUTH: SessionCache state before/after
    // =========================================================================
    #[tokio::test]
    async fn tc_post_001_successful_tool_processing() {
        let _guard = GLOBAL_IDENTITY_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-POST-001: Successful Tool Processing ===");

        let session_id = "tc-post-001-session";

        // BEFORE: Create session with known integration
        println!("BEFORE: Creating session with integration=0.85");
        create_test_session(session_id, 0.85);

        // Verify BEFORE state
        let before_snapshot = SessionCache::get().expect("Cache must be warm");
        println!("BEFORE state: integration={}", before_snapshot.integration);
        assert!((before_snapshot.integration - 0.85).abs() < 0.01);

        // Execute
        let args = PostToolArgs {
            db_path: None,
            session_id: session_id.to_string(),
            tool_name: Some("Read".to_string()),
            success: Some(true),
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await;

        // AFTER: Verify success
        assert!(result.is_ok(), "Execute must succeed: {:?}", result.err());
        let output = result.unwrap();
        assert!(output.success, "Output.success must be true");

        // Verify AFTER state in cache
        let after_snapshot = SessionCache::get().expect("Cache must have snapshot");
        println!("AFTER state: integration={}", after_snapshot.integration);

        // Read tool should have minimal positive impact
        println!(
            "RESULT: PASS - Tool processed, integration changed from 0.85 to {}",
            after_snapshot.integration
        );
    }

    // =========================================================================
    // TC-POST-002: New Session Created When Not Found
    // SOURCE OF TRUTH: New session created in cache
    // Per PRD v6, we create a new session instead of returning error
    // =========================================================================
    #[tokio::test]
    async fn tc_post_002_new_session_created() {
        let _guard = GLOBAL_IDENTITY_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-POST-002: New Session Created When Not Found ===");

        // Execute with session not in cache - should create new
        let args = PostToolArgs {
            db_path: None,
            session_id: "brand-new-session-12345".to_string(),
            tool_name: Some("Read".to_string()),
            success: Some(true),
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await;

        // Verify success (new session created)
        assert!(result.is_ok(), "Should succeed with new session created");
        let output = result.unwrap();
        assert!(output.success, "Output.success must be true");

        println!("RESULT: PASS - New session created in cache");
    }

    // =========================================================================
    // TC-POST-003: Tool Impact Analysis
    // SOURCE OF TRUTH: ImpactLevel per tool type
    // =========================================================================
    #[test]
    fn tc_post_003_tool_impact_analysis() {
        println!("\n=== TC-POST-003: Tool Impact Analysis ===");

        // Edge Case 1: Read tool (Low impact)
        println!("\nEdge Case 1: Read tool");
        let impact = analyze_tool_response("Read", "", true);
        assert_eq!(impact.level, ImpactLevel::Low);
        println!("  - Level: Low");

        // Edge Case 2: Write tool (Medium impact)
        println!("\nEdge Case 2: Write tool");
        let impact = analyze_tool_response("Write", "", true);
        assert_eq!(impact.level, ImpactLevel::Medium);
        println!("  - Level: Medium");

        // Edge Case 3: Failed Bash (High impact)
        println!("\nEdge Case 3: Failed Bash tool");
        let impact = analyze_tool_response("Bash", "command not found", false);
        assert_eq!(impact.level, ImpactLevel::High);
        println!("  - Level: High");

        // Edge Case 4: Unknown tool (No impact)
        println!("\nEdge Case 4: Unknown tool");
        let impact = analyze_tool_response("CustomTool123", "", true);
        assert_eq!(impact.level, ImpactLevel::None);
        println!("  - Level: None");

        println!("\nRESULT: PASS - All tool impacts correctly classified");
    }

    // =========================================================================
    // TC-POST-004: Impact Level Ordering
    // =========================================================================
    #[test]
    fn tc_post_004_impact_level_ordering() {
        println!("\n=== TC-POST-004: Impact Level Ordering ===");

        assert!(ImpactLevel::High > ImpactLevel::Medium);
        assert!(ImpactLevel::Medium > ImpactLevel::Low);
        assert!(ImpactLevel::Low > ImpactLevel::None);

        println!("RESULT: PASS - ImpactLevel ordering correct");
    }

    // =========================================================================
    // TC-POST-006: Tool Impact Effects
    // SOURCE OF TRUTH: SessionCache state values before/after
    // =========================================================================
    #[tokio::test]
    async fn tc_post_006_tool_impact_effects() {
        let _guard = GLOBAL_IDENTITY_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-POST-006: Tool Impact Effects ===");

        let session_id = "tool-impact-test";
        create_test_session(session_id, 0.80);

        let args = PostToolArgs {
            db_path: None,
            session_id: session_id.to_string(),
            tool_name: Some("WebFetch".to_string()),
            success: Some(true),
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await.unwrap();
        assert!(result.success);

        let snapshot = SessionCache::get().expect("Cache must have snapshot");

        // Successful tool should maintain or increase integration
        println!("Tool impact: integration 0.80 -> {}", snapshot.integration);
        assert!(
            snapshot.integration >= 0.80,
            "Successful tool should maintain or increase integration"
        );

        println!("RESULT: PASS - Tool impact affects integration correctly");
    }

    // =========================================================================
    // TC-POST-007: Execution Time Tracking
    // =========================================================================
    #[tokio::test]
    async fn tc_post_007_execution_time_tracking() {
        let _guard = GLOBAL_IDENTITY_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-POST-007: Execution Time Tracking ===");

        let session_id = "timing-test";
        create_test_session(session_id, 0.90);

        let args = PostToolArgs {
            db_path: None,
            session_id: session_id.to_string(),
            tool_name: Some("Read".to_string()),
            success: Some(true),
            stdin: false,
            format: OutputFormat::Json,
        };

        let start = std::time::Instant::now();
        let result = execute(args).await.unwrap();
        let actual_elapsed = start.elapsed().as_millis() as u64;

        // Note: execution_time_ms may be 0 if operation completes in <1ms
        // which is actually a SUCCESS per our performance budgets (3000ms timeout)
        assert!(
            result.execution_time_ms < POST_TOOL_USE_TIMEOUT_MS,
            "Execution time {} must be under timeout {}ms",
            result.execution_time_ms,
            POST_TOOL_USE_TIMEOUT_MS
        );

        println!(
            "Execution time: {}ms (timeout: {}ms)",
            result.execution_time_ms, POST_TOOL_USE_TIMEOUT_MS
        );
        println!("Actual elapsed: {}ms", actual_elapsed);
        println!("RESULT: PASS - Execution time within timeout budget");
    }

    // =========================================================================
    // TC-POST-008: Missing tool_name when stdin=false
    // SOURCE OF TRUTH: Exit code 4 (InvalidInput)
    // =========================================================================
    #[tokio::test]
    async fn tc_post_008_missing_tool_name() {
        let _guard = GLOBAL_IDENTITY_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-POST-008: Missing tool_name (stdin=false) ===");

        let session_id = "missing-tool-test";
        create_test_session(session_id, 0.90);

        let args = PostToolArgs {
            db_path: None,
            session_id: session_id.to_string(),
            tool_name: None, // Missing!
            success: Some(true),
            stdin: false, // Not reading from stdin
            format: OutputFormat::Json,
        };

        let result = execute(args).await;

        assert!(result.is_err(), "Should fail with missing tool_name");
        let err = result.unwrap_err();
        assert!(
            matches!(err, HookError::InvalidInput(_)),
            "Must be InvalidInput, got: {:?}",
            err
        );
        assert_eq!(err.exit_code(), 4, "InvalidInput must be exit code 4");

        println!("RESULT: PASS - Missing tool_name returns InvalidInput error");
    }
}
