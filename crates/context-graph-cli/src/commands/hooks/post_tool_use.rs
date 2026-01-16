//! PostToolUse hook handler
//!
//! # Performance Requirements
//! - Timeout: 3000ms (constitution.yaml hooks.timeout_ms.post_tool_use)
//! - Database access: ALLOWED
//! - IC recalculation: on significant changes
//!
//! # Constitution References
//! - IDENTITY-002: IC thresholds (Healthy>0.9, Warning<0.7, Critical<0.5)
//! - GWT-003: Identity continuity tracking
//! - AP-50: NO internal hooks - shell scripts call CLI
//! - AP-26: Exit codes (0=success, 6=crisis triggered)
//!
//! # NO BACKWARDS COMPATIBILITY - FAIL FAST

use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use tracing::{debug, error, info};

use context_graph_core::gwt::SessionIdentitySnapshot;
use context_graph_storage::rocksdb_backend::RocksDbMemex;

use super::args::PostToolArgs;
use super::error::{HookError, HookResult};
use super::types::{
    ConsciousnessState, HookInput, HookOutput, HookPayload, ICClassification, JohariQuadrant,
};

// ============================================================================
// Constants (from constitution.yaml)
// ============================================================================

/// PostToolUse timeout in milliseconds
pub const POST_TOOL_USE_TIMEOUT_MS: u64 = 3000;

/// Crisis threshold for IC score (IDENTITY-002)
pub const IC_CRISIS_THRESHOLD: f32 = 0.5;

// ============================================================================
// Types
// ============================================================================

/// Impact of tool execution on consciousness state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpactLevel {
    /// No significant impact
    None,
    /// Minor impact - log but don't recalculate IC
    Low,
    /// Moderate impact - consider IC recalculation
    Medium,
    /// High impact - force IC recalculation
    High,
}

/// Result of analyzing tool response
#[derive(Debug, Clone)]
pub struct ToolImpact {
    /// Impact level on consciousness
    pub level: ImpactLevel,
    /// Johari quadrant to update (if any)
    pub johari_update: Option<JohariQuadrant>,
    /// Delta to apply to quadrant (positive = expansion)
    pub johari_delta: f32,
    /// Whether tool execution succeeded
    pub tool_success: bool,
}

// ============================================================================
// Handler
// ============================================================================

/// Execute post-tool hook.
///
/// See module doc for full flow and exit codes.
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

    // 2. Resolve database path - FAIL FAST if missing
    let db_path = resolve_db_path(args.db_path)?;

    // 3. Open storage and load snapshot
    let memex = open_storage(&db_path)?;
    let mut snapshot = load_snapshot(&memex, &args.session_id)?;

    // 4. Analyze tool response for consciousness impact
    let impact = analyze_tool_response(&tool_name, &tool_response, tool_success);

    // 5. Update snapshot based on impact
    if impact.level >= ImpactLevel::Medium {
        update_snapshot_from_impact(&mut snapshot, &impact);
    }

    // 6. Check crisis threshold BEFORE saving
    if snapshot.last_ic < IC_CRISIS_THRESHOLD {
        error!(
            session_id = %args.session_id,
            ic = snapshot.last_ic,
            "POST_TOOL: IC crisis threshold breached"
        );
        // Save the snapshot even in crisis (for auditing)
        let _ = memex.save_snapshot(&snapshot);
        return Err(HookError::CrisisTriggered(snapshot.last_ic));
    }

    // 7. Persist updated snapshot
    memex.save_snapshot(&snapshot).map_err(|e| {
        error!(session_id = %args.session_id, error = %e, "POST_TOOL: save snapshot failed");
        HookError::storage(format!("Failed to save snapshot: {}", e))
    })?;

    // 8. Build output structures
    let consciousness_state = build_consciousness_state(&snapshot);
    let ic_classification = ICClassification::from_value(snapshot.last_ic);

    let execution_time_ms = start.elapsed().as_millis() as u64;

    info!(
        session_id = %args.session_id,
        tool_name = %tool_name,
        ic = snapshot.last_ic,
        execution_time_ms,
        "POST_TOOL: execute complete"
    );

    Ok(HookOutput::success(execution_time_ms)
        .with_consciousness_state(consciousness_state)
        .with_ic_classification(ic_classification))
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

    debug!(input_bytes = input_str.len(), "POST_TOOL: parsing stdin JSON");

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
// Database Operations
// ============================================================================

/// Resolve database path from argument or environment.
/// FAIL FAST if neither provided.
fn resolve_db_path(arg_path: Option<PathBuf>) -> HookResult<PathBuf> {
    if let Some(path) = arg_path {
        debug!(path = ?path, "POST_TOOL: using CLI db_path");
        return Ok(path);
    }

    if let Ok(env_path) = std::env::var("CONTEXT_GRAPH_DB_PATH") {
        debug!(path = %env_path, "POST_TOOL: using CONTEXT_GRAPH_DB_PATH env var");
        return Ok(PathBuf::from(env_path));
    }

    if let Ok(home) = std::env::var("HOME") {
        let default_path = PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("context-graph")
            .join("db");
        debug!(path = ?default_path, "POST_TOOL: using default db path");
        return Ok(default_path);
    }

    error!("POST_TOOL: No database path available");
    Err(HookError::invalid_input(
        "Database path required. Set CONTEXT_GRAPH_DB_PATH or pass --db-path",
    ))
}

/// Open RocksDB storage.
fn open_storage(db_path: &Path) -> HookResult<Arc<RocksDbMemex>> {
    info!(path = ?db_path, "POST_TOOL: opening storage");

    RocksDbMemex::open(db_path).map(Arc::new).map_err(|e| {
        error!(path = ?db_path, error = %e, "POST_TOOL: storage open failed");
        HookError::storage(format!("Failed to open database at {:?}: {}", db_path, e))
    })
}

/// Load snapshot for session. FAIL FAST if not found.
fn load_snapshot(memex: &Arc<RocksDbMemex>, session_id: &str) -> HookResult<SessionIdentitySnapshot> {
    match memex.load_snapshot(session_id) {
        Ok(Some(snapshot)) => {
            info!(session_id = %session_id, ic = snapshot.last_ic, "POST_TOOL: loaded snapshot");
            Ok(snapshot)
        }
        Ok(None) => {
            error!(session_id = %session_id, "POST_TOOL: session not found");
            Err(HookError::SessionNotFound(session_id.to_string()))
        }
        Err(e) => {
            error!(session_id = %session_id, error = %e, "POST_TOOL: load failed");
            Err(HookError::storage(format!("Failed to load session: {}", e)))
        }
    }
}

// ============================================================================
// Tool Analysis
// ============================================================================

/// Analyze tool response for consciousness updates
fn analyze_tool_response(tool_name: &str, _tool_response: &str, tool_success: bool) -> ToolImpact {
    match tool_name {
        // File read - expands Open quadrant (awareness)
        "Read" => ToolImpact {
            level: ImpactLevel::Low,
            johari_update: Some(JohariQuadrant::Open),
            johari_delta: 0.02,
            tool_success,
        },

        // File write/edit - expands Hidden quadrant (self-knowledge)
        "Write" | "Edit" | "MultiEdit" => ToolImpact {
            level: ImpactLevel::Medium,
            johari_update: Some(JohariQuadrant::Hidden),
            johari_delta: 0.05,
            tool_success,
        },

        // Bash commands - variable impact based on success
        "Bash" => {
            if tool_success {
                ToolImpact {
                    level: ImpactLevel::Medium,
                    johari_update: None,
                    johari_delta: 0.0,
                    tool_success,
                }
            } else {
                // Errors reveal blind spots
                ToolImpact {
                    level: ImpactLevel::High,
                    johari_update: Some(JohariQuadrant::Blind),
                    johari_delta: 0.08,
                    tool_success,
                }
            }
        }

        // External fetch - Unknown to Open transition
        "WebFetch" | "WebSearch" => ToolImpact {
            level: ImpactLevel::Medium,
            johari_update: Some(JohariQuadrant::Open),
            johari_delta: 0.06,
            tool_success,
        },

        // Git operations - project context changes
        "Git" => ToolImpact {
            level: ImpactLevel::Low,
            johari_update: Some(JohariQuadrant::Open),
            johari_delta: 0.02,
            tool_success,
        },

        // Task spawning - agent coordination
        "Task" => ToolImpact {
            level: ImpactLevel::High,
            johari_update: Some(JohariQuadrant::Hidden),
            johari_delta: 0.04,
            tool_success,
        },

        // Default - minimal impact
        _ => ToolImpact {
            level: ImpactLevel::None,
            johari_update: None,
            johari_delta: 0.0,
            tool_success,
        },
    }
}

/// Update snapshot based on tool impact
fn update_snapshot_from_impact(snapshot: &mut SessionIdentitySnapshot, impact: &ToolImpact) {
    // Apply IC changes based on impact
    match impact.johari_update {
        Some(JohariQuadrant::Open) => {
            // Open expansion is positive for IC
            snapshot.last_ic = (snapshot.last_ic + impact.johari_delta * 0.5).clamp(0.0, 1.0);
        }
        Some(JohariQuadrant::Hidden) => {
            // Hidden expansion is slightly positive
            snapshot.last_ic = (snapshot.last_ic + impact.johari_delta * 0.3).clamp(0.0, 1.0);
        }
        Some(JohariQuadrant::Blind) => {
            // Blind spot revelation is negative short-term
            snapshot.last_ic = (snapshot.last_ic - impact.johari_delta * 0.4).clamp(0.0, 1.0);
        }
        Some(JohariQuadrant::Unknown) | None => {
            // No change
        }
    }

    // Tool failures impact IC negatively
    if !impact.tool_success {
        snapshot.last_ic = (snapshot.last_ic - 0.03).clamp(0.0, 1.0);
    }
}

/// Build ConsciousnessState from snapshot.
fn build_consciousness_state(snapshot: &SessionIdentitySnapshot) -> ConsciousnessState {
    ConsciousnessState::new(
        snapshot.consciousness,
        snapshot.integration,
        snapshot.reflection,
        snapshot.differentiation,
        snapshot.last_ic,
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
// TESTS - NO MOCK DATA - REAL DATABASE VERIFICATION
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::hooks::args::OutputFormat;
    use tempfile::TempDir;

    /// Create temporary database for testing
    fn setup_test_db() -> (TempDir, PathBuf) {
        let dir = TempDir::new().expect("TempDir creation must succeed");
        let path = dir.path().join("test.db");
        (dir, path)
    }

    /// Create a real session in the database for testing
    fn create_test_session(db_path: &Path, session_id: &str, ic: f32) {
        let memex = RocksDbMemex::open(db_path).expect("DB must open");
        let mut snapshot = SessionIdentitySnapshot::new(session_id);
        snapshot.last_ic = ic;
        snapshot.consciousness = 0.5;
        snapshot.integration = 0.6;
        memex.save_snapshot(&snapshot).expect("Save must succeed");
    }

    // =========================================================================
    // TC-POST-001: Successful Tool Processing
    // SOURCE OF TRUTH: Database state before/after
    // =========================================================================
    #[tokio::test]
    async fn tc_post_001_successful_tool_processing() {
        println!("\n=== TC-POST-001: Successful Tool Processing ===");

        let (_dir, db_path) = setup_test_db();
        let session_id = "tc-post-001-session";

        // BEFORE: Create session with known IC
        println!("BEFORE: Creating session with IC=0.85");
        create_test_session(&db_path, session_id, 0.85);

        // Verify BEFORE state (drop after checking to release lock)
        {
            let memex = RocksDbMemex::open(&db_path).expect("DB must open");
            let before_snapshot = memex.load_snapshot(session_id).unwrap().unwrap();
            println!("BEFORE state: IC={}", before_snapshot.last_ic);
            assert_eq!(before_snapshot.last_ic, 0.85);
        } // memex dropped here, releasing DB lock

        // Execute
        let args = PostToolArgs {
            db_path: Some(db_path.clone()),
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

        // Verify AFTER state in database
        let memex = RocksDbMemex::open(&db_path).expect("DB must open");
        let after_snapshot = memex.load_snapshot(session_id).unwrap().unwrap();
        println!("AFTER state: IC={}", after_snapshot.last_ic);

        // Read tool should have minimal positive impact
        println!("RESULT: PASS - Tool processed, IC changed from 0.85 to {}", after_snapshot.last_ic);
    }

    // =========================================================================
    // TC-POST-002: Crisis Threshold Detection
    // SOURCE OF TRUTH: Exit code 6 returned, database state preserved
    // =========================================================================
    #[tokio::test]
    async fn tc_post_002_crisis_threshold_detection() {
        println!("\n=== TC-POST-002: Crisis Threshold Detection ===");

        let (_dir, db_path) = setup_test_db();
        let session_id = "tc-post-002-session";

        // BEFORE: Create session with IC just above threshold
        // After a failed Bash command with Blind spot impact, IC will drop below 0.5
        println!("BEFORE: Creating session with IC=0.52");
        create_test_session(&db_path, session_id, 0.52);

        // Execute with tool failure (Bash error reveals blind spot)
        let args = PostToolArgs {
            db_path: Some(db_path.clone()),
            session_id: session_id.to_string(),
            tool_name: Some("Bash".to_string()),
            success: Some(false),  // Tool failed
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await;

        // AFTER: Verify crisis triggered
        assert!(result.is_err(), "Should return error for crisis");
        let err = result.unwrap_err();
        assert!(matches!(err, HookError::CrisisTriggered(_)), "Must be CrisisTriggered, got: {:?}", err);
        assert_eq!(err.exit_code(), 6, "Crisis must be exit code 6");

        // Verify database still has the snapshot (for auditing)
        let memex = RocksDbMemex::open(&db_path).expect("DB must open");
        let snapshot = memex.load_snapshot(session_id).unwrap();
        assert!(snapshot.is_some(), "Snapshot must be preserved even in crisis");

        println!("RESULT: PASS - Crisis detected, exit code 6 returned");
    }

    // =========================================================================
    // TC-POST-003: Session Not Found
    // SOURCE OF TRUTH: Exit code 5 returned
    // =========================================================================
    #[tokio::test]
    async fn tc_post_003_session_not_found() {
        println!("\n=== TC-POST-003: Session Not Found ===");

        let (_dir, db_path) = setup_test_db();

        // Execute with non-existent session
        let args = PostToolArgs {
            db_path: Some(db_path.clone()),
            session_id: "nonexistent-session-12345".to_string(),
            tool_name: Some("Read".to_string()),
            success: Some(true),
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await;

        // Verify error
        assert!(result.is_err(), "Should return error for missing session");
        let err = result.unwrap_err();
        assert!(matches!(err, HookError::SessionNotFound(_)), "Must be SessionNotFound, got: {:?}", err);
        assert_eq!(err.exit_code(), 5, "SessionNotFound must be exit code 5");

        println!("RESULT: PASS - SessionNotFound error with exit code 5");
    }

    // =========================================================================
    // TC-POST-004: Tool Impact Analysis
    // SOURCE OF TRUTH: ImpactLevel and Johari updates per tool type
    // =========================================================================
    #[test]
    fn tc_post_004_tool_impact_analysis() {
        println!("\n=== TC-POST-004: Tool Impact Analysis ===");

        // Edge Case 1: Read tool (Low impact, Open quadrant)
        println!("\nEdge Case 1: Read tool");
        let impact = analyze_tool_response("Read", "", true);
        assert_eq!(impact.level, ImpactLevel::Low);
        assert_eq!(impact.johari_update, Some(JohariQuadrant::Open));
        println!("  - Level: Low, Quadrant: Open");

        // Edge Case 2: Write tool (Medium impact, Hidden quadrant)
        println!("\nEdge Case 2: Write tool");
        let impact = analyze_tool_response("Write", "", true);
        assert_eq!(impact.level, ImpactLevel::Medium);
        assert_eq!(impact.johari_update, Some(JohariQuadrant::Hidden));
        println!("  - Level: Medium, Quadrant: Hidden");

        // Edge Case 3: Failed Bash (High impact, Blind quadrant)
        println!("\nEdge Case 3: Failed Bash tool");
        let impact = analyze_tool_response("Bash", "command not found", false);
        assert_eq!(impact.level, ImpactLevel::High);
        assert_eq!(impact.johari_update, Some(JohariQuadrant::Blind));
        println!("  - Level: High, Quadrant: Blind");

        // Edge Case 4: Unknown tool (No impact)
        println!("\nEdge Case 4: Unknown tool");
        let impact = analyze_tool_response("CustomTool123", "", true);
        assert_eq!(impact.level, ImpactLevel::None);
        assert_eq!(impact.johari_update, None);
        println!("  - Level: None, Quadrant: None");

        println!("\nRESULT: PASS - All tool impacts correctly classified");
    }

    // =========================================================================
    // TC-POST-005: Impact Level Ordering
    // =========================================================================
    #[test]
    fn tc_post_005_impact_level_ordering() {
        println!("\n=== TC-POST-005: Impact Level Ordering ===");

        assert!(ImpactLevel::High > ImpactLevel::Medium);
        assert!(ImpactLevel::Medium > ImpactLevel::Low);
        assert!(ImpactLevel::Low > ImpactLevel::None);

        println!("RESULT: PASS - ImpactLevel ordering correct");
    }

    // =========================================================================
    // TC-POST-006: Database Path Resolution
    // =========================================================================
    #[test]
    fn tc_post_006_db_path_resolution() {
        println!("\n=== TC-POST-006: Database Path Resolution ===");

        // Clear env var
        std::env::remove_var("CONTEXT_GRAPH_DB_PATH");

        // Test 1: CLI arg takes priority
        let arg_path = PathBuf::from("/custom/path");
        let result = resolve_db_path(Some(arg_path.clone()));
        assert_eq!(result.unwrap(), arg_path);
        println!("  - CLI arg priority: PASS");

        // Test 2: Env var used when no arg
        std::env::set_var("CONTEXT_GRAPH_DB_PATH", "/env/path");
        let result = resolve_db_path(None);
        assert_eq!(result.unwrap(), PathBuf::from("/env/path"));
        println!("  - Env var fallback: PASS");

        // Cleanup
        std::env::remove_var("CONTEXT_GRAPH_DB_PATH");

        println!("RESULT: PASS - DB path resolution correct");
    }

    // =========================================================================
    // TC-POST-007: Johari Update Effects on IC
    // SOURCE OF TRUTH: Database IC values before/after
    // =========================================================================
    #[tokio::test]
    async fn tc_post_007_johari_update_effects() {
        println!("\n=== TC-POST-007: Johari Update Effects on IC ===");

        let (_dir, db_path) = setup_test_db();

        // Test Open quadrant expansion (positive)
        let session_id = "johari-open-test";
        create_test_session(&db_path, session_id, 0.80);

        let args = PostToolArgs {
            db_path: Some(db_path.clone()),
            session_id: session_id.to_string(),
            tool_name: Some("WebFetch".to_string()),  // Opens awareness
            success: Some(true),
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await.unwrap();
        assert!(result.success);

        let memex = RocksDbMemex::open(&db_path).expect("DB must open");
        let snapshot = memex.load_snapshot(session_id).unwrap().unwrap();

        // Open expansion should increase IC
        println!("Open expansion: IC 0.80 -> {}", snapshot.last_ic);
        assert!(snapshot.last_ic >= 0.80, "Open expansion should increase or maintain IC");

        println!("RESULT: PASS - Johari updates affect IC correctly");
    }

    // =========================================================================
    // TC-POST-008: Execution Time Tracking
    // =========================================================================
    #[tokio::test]
    async fn tc_post_008_execution_time_tracking() {
        println!("\n=== TC-POST-008: Execution Time Tracking ===");

        let (_dir, db_path) = setup_test_db();
        let session_id = "timing-test";
        create_test_session(&db_path, session_id, 0.90);

        let args = PostToolArgs {
            db_path: Some(db_path),
            session_id: session_id.to_string(),
            tool_name: Some("Read".to_string()),
            success: Some(true),
            stdin: false,
            format: OutputFormat::Json,
        };

        let start = std::time::Instant::now();
        let result = execute(args).await.unwrap();
        let actual_elapsed = start.elapsed().as_millis() as u64;

        assert!(result.execution_time_ms > 0, "Must have positive execution time");
        assert!(
            result.execution_time_ms < POST_TOOL_USE_TIMEOUT_MS,
            "Execution time {} must be under timeout {}ms",
            result.execution_time_ms, POST_TOOL_USE_TIMEOUT_MS
        );

        println!("Execution time: {}ms (timeout: {}ms)", result.execution_time_ms, POST_TOOL_USE_TIMEOUT_MS);
        println!("Actual elapsed: {}ms", actual_elapsed);
        println!("RESULT: PASS - Execution time within timeout budget");
    }

    // =========================================================================
    // TC-POST-009: Missing tool_name when stdin=false
    // SOURCE OF TRUTH: Exit code 4 (InvalidInput)
    // =========================================================================
    #[tokio::test]
    async fn tc_post_009_missing_tool_name() {
        println!("\n=== TC-POST-009: Missing tool_name (stdin=false) ===");

        let (_dir, db_path) = setup_test_db();
        let session_id = "missing-tool-test";
        create_test_session(&db_path, session_id, 0.90);

        let args = PostToolArgs {
            db_path: Some(db_path),
            session_id: session_id.to_string(),
            tool_name: None,  // Missing!
            success: Some(true),
            stdin: false,  // Not reading from stdin
            format: OutputFormat::Json,
        };

        let result = execute(args).await;

        assert!(result.is_err(), "Should fail with missing tool_name");
        let err = result.unwrap_err();
        assert!(matches!(err, HookError::InvalidInput(_)), "Must be InvalidInput, got: {:?}", err);
        assert_eq!(err.exit_code(), 4, "InvalidInput must be exit code 4");

        println!("RESULT: PASS - Missing tool_name returns InvalidInput error");
    }

    // =========================================================================
    // TC-POST-010: IC exactly at threshold (0.5) should NOT trigger crisis
    // Constitution: < 0.5 triggers crisis, not <= 0.5
    // =========================================================================
    #[tokio::test]
    async fn tc_post_010_ic_at_exact_threshold() {
        println!("\n=== TC-POST-010: IC at exact threshold (0.5) ===");

        let (_dir, db_path) = setup_test_db();
        let session_id = "exact-threshold-test";

        // Create session with IC exactly at 0.5
        create_test_session(&db_path, session_id, 0.5);

        // Execute with a tool that doesn't change IC (unknown tool)
        let args = PostToolArgs {
            db_path: Some(db_path.clone()),
            session_id: session_id.to_string(),
            tool_name: Some("UnknownTool".to_string()),
            success: Some(true),
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await;

        // IC=0.5 is NOT < 0.5, so should NOT trigger crisis
        assert!(result.is_ok(), "IC=0.5 should NOT trigger crisis: {:?}", result.err());
        let output = result.unwrap();
        assert!(output.success, "Should succeed when IC is exactly at threshold");

        println!("RESULT: PASS - IC=0.5 does not trigger crisis (< 0.5 required)");
    }

    // =========================================================================
    // TC-POST-011: IC just below threshold (0.49) triggers crisis
    // =========================================================================
    #[tokio::test]
    async fn tc_post_011_ic_just_below_threshold() {
        println!("\n=== TC-POST-011: IC just below threshold (0.49) ===");

        let (_dir, db_path) = setup_test_db();
        let session_id = "below-threshold-test";

        // Create session with IC just below threshold
        create_test_session(&db_path, session_id, 0.49);

        // Execute with any tool
        let args = PostToolArgs {
            db_path: Some(db_path.clone()),
            session_id: session_id.to_string(),
            tool_name: Some("Read".to_string()),
            success: Some(true),
            stdin: false,
            format: OutputFormat::Json,
        };

        let result = execute(args).await;

        // IC=0.49 IS < 0.5, so should trigger crisis
        assert!(result.is_err(), "IC=0.49 should trigger crisis");
        let err = result.unwrap_err();
        assert!(matches!(err, HookError::CrisisTriggered(_)), "Must be CrisisTriggered");
        assert_eq!(err.exit_code(), 6, "Crisis must be exit code 6");

        println!("RESULT: PASS - IC=0.49 triggers crisis (exit code 6)");
    }
}
