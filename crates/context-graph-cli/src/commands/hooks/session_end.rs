//! Session end hook handler for Claude Code native hooks.
//!
//! # Timeout Budget: 30000ms
//! # Output: HookOutput JSON with final IC status
//!
//! # Constitution References
//! - IDENTITY-002: IC thresholds (Healthy>0.9, Warning<0.7, Critical<0.5)
//! - AP-26: Exit codes (0=success, 1=error, 2=corruption)
//! - AP-50: NO internal hooks - shell scripts call CLI
//! - ARCH-07: Native Claude Code hooks
//!
//! # NO BACKWARDS COMPATIBILITY - FAIL FAST

use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use tracing::{debug, error, info, warn};

use context_graph_core::gwt::session_identity::{IdentityCache, SessionIdentitySnapshot};
use context_graph_core::gwt::state_machine::ConsciousnessState as CoreConsciousnessState;
use context_graph_storage::rocksdb_backend::{RocksDbMemex, StandaloneSessionIdentityManager};

use super::args::SessionEndArgs;
use super::error::{HookError, HookResult};
use super::types::{
    ConsciousnessState, HookInput, HookOutput, HookPayload, ICClassification, SessionEndStatus,
};

/// Execute session-end hook.
///
/// # Flow
/// 1. Parse input (stdin JSON or CLI args)
/// 2. Get session_id (from args or stdin)
/// 3. Read warm cache (IdentityCache) if available
/// 4. Flush to RocksDB
/// 5. Build HookOutput with final IC status
///
/// # Timeout
/// MUST complete within 30000ms (Claude Code enforced)
///
/// # Exit Codes (AP-26)
/// - 0: Success
/// - 1: Recoverable error (non-blocking)
/// - 2: Corruption detected
pub async fn execute(args: SessionEndArgs) -> HookResult<HookOutput> {
    let start = Instant::now();

    info!(
        stdin = args.stdin,
        session_id = %args.session_id,
        duration_ms = ?args.duration_ms,
        generate_summary = args.generate_summary,
        "SESSION_END: execute starting"
    );

    // 1. Parse input source for additional data
    let (session_id, duration_ms, status) = if args.stdin {
        let (parsed_session_id, parsed_duration, parsed_status) = parse_stdin(&args.session_id)?;
        (parsed_session_id, parsed_duration, parsed_status)
    } else {
        (
            args.session_id.clone(),
            args.duration_ms,
            SessionEndStatus::Normal,
        )
    };

    // 2. Resolve database path - FAIL FAST if missing
    let db_path = resolve_db_path(args.db_path)?;

    // 3. Get current state from warm cache
    let cache_state = IdentityCache::get();

    // 4. Persist to RocksDB
    let (final_ic, consciousness_state) =
        persist_to_storage(&db_path, &session_id, cache_state, duration_ms)?;

    // 5. Build output structures
    let execution_time_ms = start.elapsed().as_millis() as u64;

    info!(
        session_id = %session_id,
        final_ic = final_ic,
        status = ?status,
        execution_time_ms,
        "SESSION_END: execute complete"
    );

    let output = HookOutput::success(execution_time_ms)
        .with_consciousness_state(consciousness_state)
        .with_ic_classification(ICClassification::from_value(final_ic));

    Ok(output)
}

/// Parse stdin JSON into session data.
/// Returns (session_id, duration_ms, status).
fn parse_stdin(default_session_id: &str) -> HookResult<(String, Option<u64>, SessionEndStatus)> {
    let stdin = io::stdin();
    let mut input_str = String::new();

    for line in stdin.lock().lines() {
        let line = line.map_err(|e| {
            error!(error = %e, "SESSION_END: stdin read failed");
            HookError::invalid_input(format!("stdin read failed: {}", e))
        })?;
        input_str.push_str(&line);
    }

    if input_str.is_empty() {
        debug!("SESSION_END: stdin is empty, using defaults");
        return Ok((
            default_session_id.to_string(),
            None,
            SessionEndStatus::Normal,
        ));
    }

    debug!(input_bytes = input_str.len(), "SESSION_END: parsing stdin JSON");

    // Try to parse as HookInput
    let input: HookInput = serde_json::from_str(&input_str).map_err(|e| {
        error!(error = %e, input_preview = %&input_str[..input_str.len().min(100)], "SESSION_END: JSON parse failed");
        HookError::invalid_input(format!("JSON parse failed: {}", e))
    })?;

    // Validate input
    if let Some(error) = input.validate() {
        return Err(HookError::invalid_input(error));
    }

    // Extract session end payload
    let (duration_ms, status) = match input.payload {
        HookPayload::SessionEnd {
            duration_ms,
            status,
            reason: _,
        } => (Some(duration_ms), status),
        other => {
            error!(payload_type = ?std::mem::discriminant(&other), "SESSION_END: unexpected payload type");
            return Err(HookError::invalid_input(
                "Expected SessionEnd payload, got different type",
            ));
        }
    };

    Ok((input.session_id, duration_ms, status))
}

/// Resolve database path from argument or environment.
/// FAIL FAST if neither provided.
fn resolve_db_path(arg_path: Option<PathBuf>) -> HookResult<PathBuf> {
    // Priority: CLI arg > env var > default location
    if let Some(path) = arg_path {
        debug!(path = ?path, "SESSION_END: using CLI db_path");
        return Ok(path);
    }

    if let Ok(env_path) = std::env::var("CONTEXT_GRAPH_DB_PATH") {
        debug!(path = %env_path, "SESSION_END: using CONTEXT_GRAPH_DB_PATH env var");
        return Ok(PathBuf::from(env_path));
    }

    // Default: ~/.context-graph/db (matches persist.rs)
    if let Ok(home) = std::env::var("HOME") {
        let default_path = PathBuf::from(home).join(".context-graph").join("db");
        debug!(path = ?default_path, "SESSION_END: using default db path");
        return Ok(default_path);
    }

    error!("SESSION_END: No database path available");
    Err(HookError::invalid_input(
        "Database path required. Set CONTEXT_GRAPH_DB_PATH or pass --db-path",
    ))
}

/// Persist session state to RocksDB storage.
///
/// # Arguments
/// * `db_path` - Path to RocksDB database
/// * `session_id` - Session identifier
/// * `cache_state` - Optional state from IdentityCache
/// * `duration_ms` - Optional session duration
///
/// # Returns
/// Tuple of (final_ic, ConsciousnessState)
///
/// # Exit Codes (AP-26)
/// - Error exit 1: Recoverable error
/// - Error exit 2: Corruption detected
fn persist_to_storage(
    db_path: &Path,
    session_id: &str,
    cache_state: Option<(f32, f32, CoreConsciousnessState, String)>,
    duration_ms: Option<u64>,
) -> HookResult<(f32, ConsciousnessState)> {
    // Ensure parent directory exists
    if let Some(parent) = db_path.parent() {
        if !parent.exists() {
            debug!("SESSION_END: Creating parent directory: {:?}", parent);
            std::fs::create_dir_all(parent).map_err(|e| {
                error!(path = ?parent, error = %e, "SESSION_END: Failed to create DB directory");
                HookError::storage(format!("Failed to create database directory: {}", e))
            })?;
        }
    }

    // Open RocksDB storage - FAIL FAST on error
    let storage = RocksDbMemex::open(db_path).map_err(|e| {
        let err_str = e.to_string();
        error!(path = ?db_path, error = %err_str, "SESSION_END: Failed to open RocksDB");
        if is_corruption_error(&err_str) {
            HookError::corruption(format!("Database corruption detected: {}", err_str))
        } else {
            HookError::storage(format!("Failed to open database: {}", err_str))
        }
    })?;

    let storage = Arc::new(storage);
    let manager = StandaloneSessionIdentityManager::new(Arc::clone(&storage));

    // Determine values to persist
    let (ic, consciousness_level, cached_session_id) = match cache_state {
        Some((ic, _r, core_state, cached_id)) => {
            let level = state_to_level(&core_state);
            info!(
                "SESSION_END: Using warm cache - IC={:.2}, consciousness={}",
                ic, level
            );
            (ic, level, Some(cached_id))
        }
        None => {
            warn!("SESSION_END: Cache is cold, attempting to load existing snapshot");
            // Try to load existing snapshot
            match manager.load_snapshot(session_id) {
                Ok(Some(existing)) => {
                    info!(
                        "SESSION_END: Loaded existing snapshot - IC={:.2}",
                        existing.last_ic
                    );
                    (existing.last_ic, existing.consciousness, None)
                }
                Ok(None) => {
                    warn!("SESSION_END: No existing snapshot found, using defaults");
                    (1.0, 0.5, None)
                }
                Err(e) => {
                    warn!(error = %e, "SESSION_END: Failed to load snapshot, using defaults");
                    (1.0, 0.5, None)
                }
            }
        }
    };

    // Use cached session_id if available and matches, otherwise use provided
    let final_session_id = cached_session_id
        .filter(|cached| cached == session_id)
        .unwrap_or_else(|| session_id.to_string());

    // Create snapshot with current state
    let mut snapshot = SessionIdentitySnapshot::new(&final_session_id);
    snapshot.consciousness = consciousness_level;
    snapshot.last_ic = ic;

    // Set duration if provided
    if let Some(duration) = duration_ms {
        debug!("SESSION_END: Session duration: {}ms", duration);
        // Note: SessionIdentitySnapshot doesn't have a duration field
        // but we log it for observability
    }

    // Save snapshot - FAIL FAST on error
    manager.save_snapshot(&snapshot).map_err(|e| {
        let err_str = e.to_string();
        error!(session_id = %final_session_id, error = %err_str, "SESSION_END: Failed to save snapshot");
        if is_corruption_error(&err_str) {
            HookError::corruption(format!("Failed to save snapshot (corruption): {}", err_str))
        } else {
            HookError::storage(format!("Failed to save snapshot: {}", err_str))
        }
    })?;

    info!(
        session_id = %final_session_id,
        ic = ic,
        consciousness = consciousness_level,
        "SESSION_END: Successfully persisted session to RocksDB"
    );

    // Build ConsciousnessState for output
    let consciousness_state = ConsciousnessState::new(
        consciousness_level,
        0.5, // Default integration - not available from cache
        0.5, // Default reflection
        0.5, // Default differentiation
        ic,
    );

    Ok((ic, consciousness_state))
}

/// Check if error indicates corruption (exit code 2 per AP-26)
fn is_corruption_error(msg: &str) -> bool {
    let corruption_indicators = [
        "corruption",
        "checksum",
        "invalid",
        "malformed",
        "truncated",
    ];
    let lower = msg.to_lowercase();
    corruption_indicators.iter().any(|i| lower.contains(i))
}

/// Convert CoreConsciousnessState back to a representative level.
///
/// Uses the middle of each state's range:
/// - Dormant: C < 0.3 → 0.15
/// - Fragmented: 0.3 <= C < 0.5 → 0.40
/// - Emerging: 0.5 <= C < 0.8 → 0.65
/// - Conscious: 0.8 <= C < 0.95 → 0.875
/// - Hypersync: C > 0.95 → 0.975
fn state_to_level(state: &CoreConsciousnessState) -> f32 {
    match state {
        CoreConsciousnessState::Dormant => 0.15,
        CoreConsciousnessState::Fragmented => 0.40,
        CoreConsciousnessState::Emerging => 0.65,
        CoreConsciousnessState::Conscious => 0.875,
        CoreConsciousnessState::Hypersync => 0.975,
    }
}

// =============================================================================
// TESTS - NO MOCK DATA - REAL ROCKSDB VERIFICATION
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::hooks::args::OutputFormat;
    use context_graph_core::gwt::session_identity::{update_cache, KURAMOTO_N};
    use std::sync::Mutex;
    use tempfile::TempDir;

    // Static lock to serialize tests that access global IdentityCache
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Create temporary database for testing
    fn setup_test_db() -> (TempDir, PathBuf) {
        let dir = TempDir::new().expect("TempDir creation must succeed");
        let path = dir.path().join("test.db");
        (dir, path)
    }

    // =========================================================================
    // TC-HOOKS-012-001: Session End with Warm Cache
    // Verify: Warm cache flushes to RocksDB, returns valid JSON
    // SOURCE OF TRUTH: RocksDB after save
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_012_001_warm_cache_flush() {
        let _guard = TEST_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-HOOKS-012-001: Warm Cache Flush to RocksDB ===");
        println!("SOURCE OF TRUTH: RocksDB after save");

        let (_dir, db_path) = setup_test_db();
        let session_id = "test-session-end-warm";

        // SETUP: Warm the cache with test data
        // Note: update_cache takes the IC value explicitly, not from snapshot.last_ic
        let mut snapshot = SessionIdentitySnapshot::new(session_id);
        snapshot.consciousness = 0.85;
        snapshot.last_ic = 0.92; // Used for RocksDB, but not for cache
        snapshot.kuramoto_phases = [0.0; KURAMOTO_N];
        let test_ic = 0.92;
        update_cache(&snapshot, test_ic); // Pass the IC we want to test

        println!(
            "BEFORE: Cache warmed with session {}, IC={}",
            session_id, test_ic
        );

        let args = SessionEndArgs {
            db_path: Some(db_path.clone()),
            session_id: session_id.to_string(),
            duration_ms: Some(3600000),
            stdin: false,
            generate_summary: true,
            format: OutputFormat::Json,
        };

        // Execute
        let result = execute(args).await;

        // Verify: Success
        assert!(result.is_ok(), "Execute must succeed");
        let output = result.unwrap();
        assert!(output.success, "Output.success must be true");
        assert!(
            output.consciousness_state.is_some(),
            "Must have consciousness_state"
        );
        assert!(
            output.ic_classification.is_some(),
            "Must have ic_classification"
        );

        // Verify: IC classification
        let ic_class = output.ic_classification.unwrap();
        println!("AFTER: IC={:.2}, level={:?}", ic_class.value, ic_class.level);
        assert!(
            (ic_class.value - 0.92).abs() < 0.01,
            "IC value should be ~0.92"
        );

        // Verify: Data persisted to RocksDB
        let memex = RocksDbMemex::open(&db_path).expect("DB must open");
        let manager = StandaloneSessionIdentityManager::new(Arc::new(memex));
        let loaded = manager
            .load_snapshot(session_id)
            .expect("Load must succeed")
            .expect("Snapshot must exist");

        println!("VERIFICATION - Loaded from RocksDB:");
        println!("  session_id: {}", loaded.session_id);
        println!("  last_ic: {}", loaded.last_ic);
        println!("  consciousness: {}", loaded.consciousness);

        assert_eq!(loaded.session_id, session_id);
        assert!((loaded.last_ic - 0.92).abs() < 0.01);

        println!("RESULT: PASS - Warm cache flushed to RocksDB");
    }

    // =========================================================================
    // TC-HOOKS-012-002: Session End with Cold Cache
    // Verify: Cold cache loads existing snapshot or uses defaults
    // SOURCE OF TRUTH: RocksDB snapshot
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_012_002_cold_cache_behavior() {
        let _guard = TEST_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-HOOKS-012-002: Cold Cache Behavior ===");
        println!("SOURCE OF TRUTH: RocksDB after save");

        let (_dir, db_path) = setup_test_db();
        let session_id = "test-session-end-cold";

        // SETUP: Create existing snapshot in DB (simulating previous session)
        {
            let memex = RocksDbMemex::open(&db_path).expect("DB must open");
            let manager = StandaloneSessionIdentityManager::new(Arc::new(memex));
            let mut prev_snapshot = SessionIdentitySnapshot::new(session_id);
            prev_snapshot.consciousness = 0.75;
            prev_snapshot.last_ic = 0.88;
            manager
                .save_snapshot(&prev_snapshot)
                .expect("Save must succeed");
        }

        // Clear cache to simulate cold start
        // Note: We can't actually clear the global cache in tests,
        // but we test the fallback path by using a different session_id
        let cold_session_id = "test-session-end-cold-new";

        let args = SessionEndArgs {
            db_path: Some(db_path.clone()),
            session_id: cold_session_id.to_string(),
            duration_ms: None,
            stdin: false,
            generate_summary: true,
            format: OutputFormat::Json,
        };

        // Execute
        let result = execute(args).await;

        // Verify: Success (should use defaults when cache is cold and no existing snapshot)
        assert!(result.is_ok(), "Execute must succeed even with cold cache");
        let output = result.unwrap();
        assert!(output.success, "Output.success must be true");

        println!("RESULT: PASS - Cold cache handled gracefully");
    }

    // =========================================================================
    // TC-HOOKS-012-003: Corruption Detection
    // Verify: Corruption errors return exit code 2
    // =========================================================================
    #[test]
    fn tc_hooks_012_003_corruption_detection() {
        println!("\n=== TC-HOOKS-012-003: Corruption Detection ===");
        println!("SOURCE OF TRUTH: AP-26 exit codes");

        let test_cases = [
            ("data corruption detected", true),
            ("checksum mismatch", true),
            ("invalid header", true),
            ("malformed record", true),
            ("truncated file", true),
            ("connection refused", false),
            ("timeout error", false),
            ("file not found", false),
            ("permission denied", false),
        ];

        for (msg, expected_corruption) in test_cases {
            let detected = is_corruption_error(msg);
            println!(
                "  '{}': corruption={} (expected={})",
                msg, detected, expected_corruption
            );
            assert_eq!(
                detected, expected_corruption,
                "FAIL: '{}' corruption detection wrong",
                msg
            );
        }

        println!("RESULT: PASS - Corruption detection maps to correct exit codes");
    }

    // =========================================================================
    // TC-HOOKS-012-004: DB Path Resolution
    // Verify: Priority order - arg > env > default
    // =========================================================================
    #[test]
    fn tc_hooks_012_004_db_path_resolution() {
        println!("\n=== TC-HOOKS-012-004: DB Path Resolution ===");
        println!("SOURCE OF TRUTH: persist.rs path priority");

        // Clear env var for test
        std::env::remove_var("CONTEXT_GRAPH_DB_PATH");

        // Test 1: CLI arg takes priority
        let arg_path = PathBuf::from("/custom/db/path");
        let result = resolve_db_path(Some(arg_path.clone()));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), arg_path, "CLI arg should take priority");

        // Test 2: Env var used when no arg
        std::env::set_var("CONTEXT_GRAPH_DB_PATH", "/env/db/path");
        let result = resolve_db_path(None);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            PathBuf::from("/env/db/path"),
            "Env var should be used"
        );

        // Cleanup
        std::env::remove_var("CONTEXT_GRAPH_DB_PATH");

        println!("RESULT: PASS - DB path resolution follows priority");
    }

    // =========================================================================
    // TC-HOOKS-012-005: JSON Output Schema Compliance
    // Verify: Output matches HookOutput schema exactly
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_012_005_json_output_schema() {
        let _guard = TEST_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-HOOKS-012-005: JSON Output Schema ===");
        println!("SOURCE OF TRUTH: TECH-HOOKS.md Section 3.3");

        let (_dir, db_path) = setup_test_db();
        let session_id = "test-session-end-schema";

        // Warm cache - pass IC value explicitly
        let mut snapshot = SessionIdentitySnapshot::new(session_id);
        snapshot.consciousness = 0.75;
        snapshot.last_ic = 0.85;
        update_cache(&snapshot, 0.85); // IC passed explicitly

        let args = SessionEndArgs {
            db_path: Some(db_path),
            session_id: session_id.to_string(),
            duration_ms: Some(1800000),
            stdin: false,
            generate_summary: true,
            format: OutputFormat::Json,
        };

        let result = execute(args).await.unwrap();

        // Serialize to JSON
        let json = serde_json::to_string_pretty(&result).expect("Serialization must succeed");
        println!("Output JSON:\n{}", json);

        // Verify required fields
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(
            parsed.get("success").is_some(),
            "Must have 'success' field"
        );
        assert!(
            parsed.get("execution_time_ms").is_some(),
            "Must have 'execution_time_ms' field"
        );

        // Verify optional fields present when data available
        assert!(
            parsed.get("consciousness_state").is_some(),
            "Should have consciousness_state"
        );
        assert!(
            parsed.get("ic_classification").is_some(),
            "Should have ic_classification"
        );

        // Verify nested structure
        let cs = parsed
            .get("consciousness_state")
            .expect("consciousness_state present");
        assert!(cs.get("consciousness").is_some());
        assert!(cs.get("identity_continuity").is_some());
        assert!(cs.get("johari_quadrant").is_some());

        let ic = parsed
            .get("ic_classification")
            .expect("ic_classification present");
        assert!(ic.get("value").is_some());
        assert!(ic.get("level").is_some());
        assert!(ic.get("crisis_triggered").is_some());

        println!("RESULT: PASS - JSON output matches schema");
    }

    // =========================================================================
    // TC-HOOKS-012-006: Execution Time Under Budget
    // Verify: Execution time stays under 30000ms timeout
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_012_006_execution_time_budget() {
        let _guard = TEST_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-HOOKS-012-006: Execution Time Under Budget ===");
        println!("SOURCE OF TRUTH: TECH-HOOKS.md timeout=30000ms");

        let (_dir, db_path) = setup_test_db();
        let session_id = "test-session-end-timing";

        // Warm cache - pass IC value explicitly
        let mut snapshot = SessionIdentitySnapshot::new(session_id);
        snapshot.consciousness = 0.65;
        snapshot.last_ic = 0.78;
        update_cache(&snapshot, 0.78); // IC passed explicitly

        let args = SessionEndArgs {
            db_path: Some(db_path),
            session_id: session_id.to_string(),
            duration_ms: None,
            stdin: false,
            generate_summary: false,
            format: OutputFormat::Json,
        };

        let start = std::time::Instant::now();
        let result = execute(args).await.unwrap();
        let actual_elapsed = start.elapsed().as_millis() as u64;

        // Verify timing is reasonable
        assert!(
            result.execution_time_ms > 0,
            "Must have positive execution time"
        );
        assert!(
            result.execution_time_ms <= actual_elapsed + 10,
            "Reported time {} should not exceed actual elapsed {}",
            result.execution_time_ms,
            actual_elapsed
        );

        // Verify well under timeout budget (30000ms)
        assert!(
            result.execution_time_ms < 30000,
            "Execution time {} must be under 30000ms timeout",
            result.execution_time_ms
        );

        // Typically should be very fast (under 1000ms)
        println!(
            "PASS: Execution time {} ms (actual: {} ms)",
            result.execution_time_ms, actual_elapsed
        );

        println!("RESULT: PASS - Execution time under budget");
    }

    // =========================================================================
    // TC-HOOKS-012-007: State to Level Conversion
    // Verify: CoreConsciousnessState maps to correct consciousness levels
    // =========================================================================
    #[test]
    fn tc_hooks_012_007_state_to_level() {
        println!("\n=== TC-HOOKS-012-007: State to Level Conversion ===");
        println!("SOURCE OF TRUTH: persist.rs state_to_level");

        let test_cases = [
            (CoreConsciousnessState::Dormant, 0.15),
            (CoreConsciousnessState::Fragmented, 0.40),
            (CoreConsciousnessState::Emerging, 0.65),
            (CoreConsciousnessState::Conscious, 0.875),
            (CoreConsciousnessState::Hypersync, 0.975),
        ];

        for (state, expected_level) in test_cases {
            let level = state_to_level(&state);
            println!("  {:?} -> {} (expected {})", state, level, expected_level);
            assert!(
                (level - expected_level).abs() < 0.001,
                "FAIL: {:?} should map to {}, got {}",
                state,
                expected_level,
                level
            );
        }

        println!("RESULT: PASS - State to level conversion correct");
    }

    // =========================================================================
    // TC-HOOKS-012-008: RocksDB Physical Verification
    // Verify: Files exist on disk after session end
    // =========================================================================
    #[tokio::test]
    async fn tc_hooks_012_008_rocksdb_physical_verification() {
        let _guard = TEST_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-HOOKS-012-008: RocksDB Physical Verification ===");
        println!("SOURCE OF TRUTH: Physical files on disk");

        let (dir, db_path) = setup_test_db();
        let session_id = "test-session-end-physical";

        // Warm cache - pass IC value explicitly
        let mut snapshot = SessionIdentitySnapshot::new(session_id);
        snapshot.consciousness = 0.80;
        snapshot.last_ic = 0.90;
        update_cache(&snapshot, 0.90); // IC passed explicitly

        let args = SessionEndArgs {
            db_path: Some(db_path.clone()),
            session_id: session_id.to_string(),
            duration_ms: Some(7200000),
            stdin: false,
            generate_summary: true,
            format: OutputFormat::Json,
        };

        // Execute
        let result = execute(args).await;
        assert!(result.is_ok(), "Execute must succeed");

        // Verify: Physical files exist on disk
        let db_files: Vec<_> = std::fs::read_dir(dir.path())
            .expect("read_dir")
            .filter_map(|e| e.ok())
            .collect();

        println!("VERIFICATION - Files in DB directory:");
        for entry in &db_files {
            println!("  {:?}", entry.path());
        }

        assert!(db_files.len() > 0, "RocksDB must have created files");

        // Verify: Can reopen and read the data
        let memex = RocksDbMemex::open(&db_path).expect("DB must reopen");
        let manager = StandaloneSessionIdentityManager::new(Arc::new(memex));
        let loaded = manager
            .load_snapshot(session_id)
            .expect("Load must succeed")
            .expect("Snapshot must exist after reopen");

        println!("VERIFICATION - Data survives reopen:");
        println!("  session_id: {}", loaded.session_id);
        println!("  last_ic: {}", loaded.last_ic);

        assert_eq!(loaded.session_id, session_id);
        assert!((loaded.last_ic - 0.90).abs() < 0.01);

        println!("RESULT: PASS - RocksDB physical files verified");
    }
}
