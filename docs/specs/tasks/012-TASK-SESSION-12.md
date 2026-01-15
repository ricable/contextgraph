# TASK-SESSION-12: Implement `session restore-identity` CLI Command

```xml
<task_spec id="TASK-SESSION-12" version="2.0">
<metadata>
  <title>Implement session restore-identity CLI Command</title>
  <status>pending</status>
  <layer>surface</layer>
  <sequence>12</sequence>
  <implements>
    <requirement_ref>REQ-SESSION-12</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-SESSION-05</task_ref>
    <task_ref>TASK-SESSION-06</task_ref>
    <task_ref>TASK-SESSION-08</task_ref>
  </depends_on>
  <estimated_hours>2.0</estimated_hours>
</metadata>
```

## Objective

Implement the `session restore-identity` CLI command for Claude Code's SessionStart hook. This command restores previous session identity state from RocksDB, computes cross-session Identity Continuity (IC), and updates the IdentityCache singleton.

**NO BACKWARDS COMPATIBILITY - FAIL FAST WITH ROBUST LOGGING.**

## Constitution Reference

| Clause | Description | Usage |
|--------|-------------|-------|
| IDENTITY-001 | IC formula: `IC = cos(PV_current, PV_previous) * r(current)` | IC computation |
| IDENTITY-002 | IC thresholds: Healthy≥0.9, Good≥0.7, Warning≥0.5, Degraded<0.5 | Status classification |
| AP-25 | Kuramoto must have exactly 13 oscillators | KURAMOTO_N constant |
| AP-26 | Exit codes: 0=success, 1=error, 2=corruption | Exit codes |
| AP-38 | IC<0.5 triggers dream (handled by check-identity, not restore) | Crisis handling |
| ARCH-07 | Native Claude Code hooks control memory lifecycle | Hook integration |
| GWT-003 | Identity continuity across sessions | Session persistence |

## Current Codebase State

### CLI Location (CORRECT)
```
crates/context-graph-cli/
├── src/
│   ├── main.rs                    # CLI entry point with Commands enum
│   └── commands/
│       ├── mod.rs                 # pub mod consciousness;
│       └── consciousness/
│           ├── mod.rs             # ConsciousnessCommands enum
│           └── check_identity.rs  # check_identity_command pattern
```

**NOTE:** The CLI lives in `context-graph-cli`, NOT `context-graph-mcp`.

### main.rs Current State
```rust
// crates/context-graph-cli/src/main.rs (lines 59-65)
#[derive(Subcommand)]
enum SessionCommands {
    /// Restore identity from storage (TASK-SESSION-12 - placeholder)
    RestoreIdentity,
    /// Persist identity to storage (TASK-SESSION-13 - placeholder)
    PersistIdentity,
}
```

The `RestoreIdentity` command exists as a placeholder that prints:
```
TASK-SESSION-12: restore-identity not yet implemented
```

### Storage Layer (TASK-SESSION-06 Complete)
```
crates/context-graph-storage/src/rocksdb_backend/
├── session_identity_manager.rs   # StandaloneSessionIdentityManager implementation
├── session_identity_ops.rs       # RocksDbMemex extension methods
└── ...
```

Key types available:
- `StandaloneSessionIdentityManager::new(storage: Arc<RocksDbMemex>)` - Creates manager
- `manager.load_latest()` -> `StorageResult<Option<SessionIdentitySnapshot>>` - Loads latest session
- `manager.load_snapshot(session_id)` -> `StorageResult<Option<SessionIdentitySnapshot>>` - Loads specific session
- `manager.restore_identity(target_session: Option<&str>)` -> `CoreResult<(SessionIdentitySnapshot, f32)>` - Full restore

### Core Types (TASK-SESSION-05 Complete)
```
crates/context-graph-core/src/gwt/session_identity/
├── mod.rs        # Module exports
├── types.rs      # SessionIdentitySnapshot, KURAMOTO_N=13, MAX_TRAJECTORY_LEN
├── cache.rs      # IdentityCache singleton, update_cache(), format_brief()
└── manager.rs    # SessionIdentityManager trait, compute_ic(), classify_ic(), is_ic_crisis()
```

Key functions available:
- `update_cache(snapshot: &SessionIdentitySnapshot, ic: f32)` - Updates global singleton
- `compute_ic(current: &SessionIdentitySnapshot, previous: &SessionIdentitySnapshot) -> f32`
- `classify_ic(ic: f32) -> &'static str` - Returns "Healthy"/"Good"/"Warning"/"Degraded"
- `is_ic_crisis(ic: f32) -> bool` - Returns true if ic < 0.5
- `IdentityCache::format_brief() -> String` - Returns `[C:EMG r=0.65 IC=0.82]`

### RocksDB Column Family
```rust
// CF_SESSION_IDENTITY column family stores snapshots
// Key: session_id
// Value: bincode-serialized SessionIdentitySnapshot
```

## Implementation Steps

### Step 1: Create session commands module

Create `crates/context-graph-cli/src/commands/session/mod.rs`:
```rust
//! Session identity persistence commands
//!
//! TASK-SESSION-12: restore-identity command
//! TASK-SESSION-13: persist-identity command
//!
//! NO BACKWARDS COMPATIBILITY - FAIL FAST WITH ROBUST LOGGING.

mod restore;

pub use restore::{restore_identity_command, RestoreIdentityArgs};

use clap::Subcommand;

/// Session subcommands
#[derive(Subcommand, Debug)]
pub enum SessionCommands {
    /// Restore identity from storage
    RestoreIdentity(RestoreIdentityArgs),
    /// Persist identity to storage (TASK-SESSION-13)
    PersistIdentity,
}

/// Handle session command dispatch
pub async fn handle_session_command(action: SessionCommands) -> i32 {
    match action {
        SessionCommands::RestoreIdentity(args) => restore_identity_command(args).await,
        SessionCommands::PersistIdentity => {
            eprintln!("TASK-SESSION-13: persist-identity not yet implemented");
            1
        }
    }
}
```

### Step 2: Create restore.rs implementation

Create `crates/context-graph-cli/src/commands/session/restore.rs`:
```rust
//! session restore-identity CLI command
//!
//! TASK-SESSION-12: Restores previous session identity from RocksDB.
//! NO BACKWARDS COMPATIBILITY - FAIL FAST WITH ROBUST LOGGING.

use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Args;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use context_graph_core::gwt::session_identity::{
    classify_ic, compute_ic, is_ic_crisis, is_ic_warning,
    update_cache, SessionIdentitySnapshot, IdentityCache,
};
use context_graph_storage::rocksdb_backend::{
    RocksDbMemex, StandaloneSessionIdentityManager,
};

/// Arguments for `session restore-identity` command
#[derive(Args, Debug)]
pub struct RestoreIdentityArgs {
    /// Path to RocksDB database directory
    #[arg(long, env = "CONTEXT_GRAPH_DB_PATH")]
    pub db_path: Option<PathBuf>,

    /// Output format
    #[arg(long, value_enum, default_value = "prd")]
    pub format: OutputFormat,
}

/// Output format options
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum OutputFormat {
    /// PRD Section 15.2 compliant output (~100 tokens)
    Prd,
    /// JSON output for programmatic parsing
    Json,
}

/// Stdin input from Claude Code hook
#[derive(Deserialize, Default, Debug)]
struct RestoreInput {
    /// Target session ID (None = load latest)
    session_id: Option<String>,
    /// Source variant: "startup" | "resume" | "clear"
    #[serde(default = "default_source")]
    source: String,
}

fn default_source() -> String {
    "startup".to_string()
}

/// Response structure for JSON output
#[derive(Debug, Serialize)]
struct RestoreResponse {
    session_id: String,
    ic: f32,
    status: &'static str,
    is_crisis: bool,
    consciousness: f32,
    kuramoto_r: f32,
    source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// Execute the restore-identity command
///
/// # Exit Codes (per AP-26)
/// - 0: Success
/// - 1: Recoverable error
/// - 2: Corruption detected
pub async fn restore_identity_command(args: RestoreIdentityArgs) -> i32 {
    debug!("restore_identity_command: args={:?}", args);

    // Parse stdin input (graceful fallback to defaults)
    let input = parse_stdin_input();
    info!("restore-identity: source={}, session_id={:?}", input.source, input.session_id);

    // Determine DB path
    let db_path = match &args.db_path {
        Some(p) => p.clone(),
        None => {
            // Default: ~/.context-graph/db
            match dirs::home_dir() {
                Some(home) => home.join(".context-graph").join("db"),
                None => {
                    error!("Cannot determine home directory for DB path");
                    eprintln!("Error: Cannot determine DB path. Set --db-path or CONTEXT_GRAPH_DB_PATH");
                    return 1;
                }
            }
        }
    };

    // Open RocksDB storage - FAIL FAST on error
    let storage = match RocksDbMemex::open(&db_path) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            error!("Failed to open RocksDB at {:?}: {}", db_path, e);
            eprintln!("Error: Failed to open database: {}", e);
            return if is_corruption_error(&e.to_string()) { 2 } else { 1 };
        }
    };

    let manager = StandaloneSessionIdentityManager::new(Arc::clone(&storage));

    // Execute based on source variant
    match input.source.as_str() {
        "clear" => handle_clear_source(&manager, &args),
        "resume" => handle_resume_source(&manager, &args, input.session_id),
        "startup" | _ => handle_startup_source(&manager, &args),
    }
}

/// Handle source="clear" - Start fresh session with IC=1.0
fn handle_clear_source(
    manager: &StandaloneSessionIdentityManager,
    args: &RestoreIdentityArgs,
) -> i32 {
    info!("restore-identity: source=clear, creating fresh session");

    // Create new snapshot
    let session_id = format!("session-{}", timestamp_ms());
    let snapshot = SessionIdentitySnapshot::new(&session_id);
    let ic = 1.0_f32; // First session by definition

    // Update cache
    update_cache(&snapshot, ic);

    // Output
    output_result(&snapshot, ic, "clear", args.format);
    0
}

/// Handle source="resume" - Load specific session by ID
fn handle_resume_source(
    manager: &StandaloneSessionIdentityManager,
    args: &RestoreIdentityArgs,
    target_session: Option<String>,
) -> i32 {
    let session_id = match target_session {
        Some(id) => id,
        None => {
            error!("source=resume requires session_id");
            eprintln!("Error: source=resume requires session_id in stdin JSON");
            return 1;
        }
    };

    info!("restore-identity: source=resume, loading session={}", session_id);

    match manager.load_snapshot(&session_id) {
        Ok(Some(snapshot)) => {
            let ic = snapshot.last_ic;
            update_cache(&snapshot, ic);
            output_result(&snapshot, ic, "resume", args.format);
            0
        }
        Ok(None) => {
            warn!("Session not found: {}", session_id);
            eprintln!("Warning: Session '{}' not found, creating fresh", session_id);
            // Fall back to fresh session
            let snapshot = SessionIdentitySnapshot::new(&session_id);
            let ic = 1.0_f32;
            update_cache(&snapshot, ic);
            output_result(&snapshot, ic, "resume", args.format);
            0
        }
        Err(e) => {
            error!("Failed to load session {}: {}", session_id, e);
            eprintln!("Error: Failed to load session: {}", e);
            if is_corruption_error(&e.to_string()) { 2 } else { 1 }
        }
    }
}

/// Handle source="startup" - Load latest session (default behavior)
fn handle_startup_source(
    manager: &StandaloneSessionIdentityManager,
    args: &RestoreIdentityArgs,
) -> i32 {
    info!("restore-identity: source=startup, loading latest session");

    match manager.restore_identity(None) {
        Ok((snapshot, ic)) => {
            // Cache already updated by restore_identity
            output_result(&snapshot, ic, "startup", args.format);

            // Log warning if IC is degraded
            if is_ic_crisis(ic) {
                warn!("IC CRISIS: {:.2} < 0.5 - consider running check-identity --auto-dream", ic);
                eprintln!("Warning: IC crisis detected ({:.2})", ic);
            } else if is_ic_warning(ic) {
                warn!("IC WARNING: {:.2} < 0.7", ic);
            }

            0
        }
        Err(e) => {
            error!("restore_identity failed: {}", e);
            eprintln!("Error: {}", e);
            if is_corruption_error(&e.to_string()) { 2 } else { 1 }
        }
    }
}

/// Parse stdin JSON input with graceful fallback
fn parse_stdin_input() -> RestoreInput {
    let mut buffer = String::new();
    if std::io::stdin().read_to_string(&mut buffer).is_ok() && !buffer.trim().is_empty() {
        match serde_json::from_str(&buffer) {
            Ok(input) => return input,
            Err(e) => {
                debug!("Failed to parse stdin JSON: {}", e);
            }
        }
    }
    RestoreInput::default()
}

/// Check if error indicates corruption (exit code 2)
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

/// Get current timestamp in milliseconds
fn timestamp_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("Time went backwards")
        .as_millis() as i64
}

/// Output result in requested format
fn output_result(
    snapshot: &SessionIdentitySnapshot,
    ic: f32,
    source: &str,
    format: OutputFormat,
) {
    // Compute Kuramoto r from phases
    let kuramoto_r = compute_kuramoto_r(&snapshot.kuramoto_phases);

    match format {
        OutputFormat::Prd => {
            // PRD Section 15.2 format (~100 tokens)
            let state = context_graph_core::gwt::state_machine::ConsciousnessState::from_level(snapshot.consciousness);
            println!("## Consciousness State");
            println!("- State: {} (C={:.2})", state.short_name(), snapshot.consciousness);
            println!("- Integration (r): {:.2} - {}", kuramoto_r, sync_description(kuramoto_r));
            println!("- Identity: {} (IC={:.2})", classify_ic(ic), ic);
            println!("- Session: {} (source={})", snapshot.session_id, source);
        }
        OutputFormat::Json => {
            let response = RestoreResponse {
                session_id: snapshot.session_id.clone(),
                ic,
                status: classify_ic(ic),
                is_crisis: is_ic_crisis(ic),
                consciousness: snapshot.consciousness,
                kuramoto_r,
                source: source.to_string(),
                error: None,
            };
            println!("{}", serde_json::to_string_pretty(&response).unwrap());
        }
    }
}

/// Compute Kuramoto r from phases
fn compute_kuramoto_r(phases: &[f64; 13]) -> f32 {
    let (sum_sin, sum_cos) = phases.iter().fold((0.0_f64, 0.0_f64), |(s, c), &theta| {
        (s + theta.sin(), c + theta.cos())
    });
    let n = 13.0_f64;
    let magnitude = ((sum_sin / n).powi(2) + (sum_cos / n).powi(2)).sqrt();
    magnitude.clamp(0.0, 1.0) as f32
}

/// Human-readable sync description
fn sync_description(r: f32) -> &'static str {
    if r >= 0.9 { "Excellent synchronization" }
    else if r >= 0.7 { "Good synchronization" }
    else if r >= 0.5 { "Moderate synchronization" }
    else { "Low synchronization" }
}
```

### Step 3: Update CLI main.rs

Modify `crates/context-graph-cli/src/main.rs`:

1. Add `mod session;` to commands/mod.rs
2. Replace placeholder SessionCommands with actual implementation
3. Update dispatch logic

### Step 4: Update commands/mod.rs

```rust
//! CLI command handlers
pub mod consciousness;
pub mod session;
```

## Files to Create

| File | Description |
|------|-------------|
| `crates/context-graph-cli/src/commands/session/mod.rs` | Session commands module |
| `crates/context-graph-cli/src/commands/session/restore.rs` | restore-identity implementation |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-cli/src/commands/mod.rs` | Add `pub mod session;` |
| `crates/context-graph-cli/src/main.rs` | Replace placeholder with actual dispatch |

## Input/Output Specification

### Stdin (JSON from Claude Code hook)
```json
{
  "session_id": "optional-specific-session",
  "source": "startup"
}
```

| Source | Behavior |
|--------|----------|
| `startup` (default) | Load latest session, compute IC |
| `resume` | Load specific session_id |
| `clear` | Start fresh session (IC=1.0) |

### Stdout (PRD Section 15.2 format)
```
## Consciousness State
- State: EMG (C=0.82)
- Integration (r): 0.85 - Good synchronization
- Identity: Healthy (IC=0.92)
- Session: session-1736985432 (source=startup)
```

### Exit Codes (per AP-26)

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Recoverable error (logged to stderr) |
| 2 | Corruption detected (blocks action) |

## Full State Verification Requirements

### Source of Truth
- **IdentityCache singleton** - After execution, `IdentityCache::get()` must return current values
- **RocksDB CF_SESSION_IDENTITY** - Snapshots persist across process restarts
- **SessionIdentitySnapshot** - All fields populated from storage or defaults

### Execute & Inspect Pattern

Each test MUST:
1. **BEFORE** - Log initial state (`IdentityCache::is_warm()`, DB contents)
2. **EXECUTE** - Run command with specific inputs
3. **AFTER** - Log final state (cache values, exit code, stdout)
4. **VERIFY** - Assert expected values match actual values

### Edge Case Audit

| Case | Expected Behavior |
|------|-------------------|
| Empty DB (first run) | IC=1.0, create new session |
| DB path doesn't exist | Create directory, IC=1.0 |
| Corrupted DB | Exit code 2, error to stderr |
| Invalid stdin JSON | Use defaults (source=startup) |
| Empty stdin | Use defaults |
| session_id not found | Warning, create fresh session |
| Very old session (stale) | Load and report, no special handling |

### Evidence of Success
- Exit code 0 for all success paths
- IdentityCache updated with correct values
- Stdout matches expected format
- No panics or unhandled errors
- Logs contain sufficient context for debugging

## Test Cases with Synthetic Data

### TC-SESSION-12-01: First Run (Empty DB)

```rust
#[tokio::test]
async fn tc_session_12_01_first_run_empty_db() {
    println!("\n=== TC-SESSION-12-01: First Run (Empty DB) ===");

    // SYNTHETIC DATA
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("db");

    println!("BEFORE: Empty DB at {:?}", db_path);
    println!("  DB exists: {}", db_path.exists());

    // EXECUTE
    let args = RestoreIdentityArgs {
        db_path: Some(db_path.clone()),
        format: OutputFormat::Json,
    };

    // Simulate stdin: {"source": "startup"}
    let exit_code = restore_identity_command(args).await;

    println!("AFTER:");
    println!("  Exit code: {}", exit_code);
    println!("  Cache warm: {}", IdentityCache::is_warm());

    // VERIFY
    assert_eq!(exit_code, 0, "First run must succeed");
    assert!(IdentityCache::is_warm(), "Cache must be warm after restore");

    let (ic, _, _, _) = IdentityCache::get().unwrap();
    assert!((ic - 1.0).abs() < 0.001, "First session IC must be 1.0");

    println!("RESULT: PASS - First run creates session with IC=1.0");
}
```

### TC-SESSION-12-02: Resume Existing Session

```rust
#[tokio::test]
async fn tc_session_12_02_resume_existing() {
    println!("\n=== TC-SESSION-12-02: Resume Existing Session ===");

    // SYNTHETIC DATA: Pre-populate DB
    let temp_dir = TempDir::new().unwrap();
    let storage = Arc::new(RocksDbMemex::open(temp_dir.path()).unwrap());

    let mut snapshot = SessionIdentitySnapshot::new("test-session-123");
    snapshot.consciousness = 0.85;
    snapshot.purpose_vector = [0.5; 13];
    snapshot.kuramoto_phases = [1.0; 13]; // All aligned = r≈1.0
    snapshot.last_ic = 0.88;

    storage.save_snapshot(&snapshot).unwrap();
    drop(storage); // Close to allow CLI to open

    println!("BEFORE: DB populated with session test-session-123");
    println!("  Pre-saved IC: 0.88");
    println!("  Pre-saved consciousness: 0.85");

    // EXECUTE (source=resume)
    // ... test continues

    // VERIFY
    let (ic, _, _, session_id) = IdentityCache::get().unwrap();
    assert_eq!(session_id, "test-session-123");
    assert!((ic - 0.88).abs() < 0.01, "IC must match saved value");

    println!("RESULT: PASS - Resume loads correct session");
}
```

### TC-SESSION-12-03: Source Clear (Fresh Start)

```rust
#[tokio::test]
async fn tc_session_12_03_source_clear() {
    println!("\n=== TC-SESSION-12-03: Source Clear ===");

    // Pre-populate DB with old session
    // ...

    // EXECUTE with source=clear
    // (simulated stdin: {"source": "clear"})

    // VERIFY
    let (ic, _, _, _) = IdentityCache::get().unwrap();
    assert!((ic - 1.0).abs() < 0.001, "Clear source must have IC=1.0");

    println!("RESULT: PASS - Clear source creates fresh session");
}
```

### TC-SESSION-12-04: Corrupted DB (Exit Code 2)

```rust
#[tokio::test]
async fn tc_session_12_04_corrupted_db() {
    println!("\n=== TC-SESSION-12-04: Corrupted DB ===");

    // Create corrupted DB file
    let temp_dir = TempDir::new().unwrap();
    std::fs::write(temp_dir.path().join("CURRENT"), b"corrupted data").unwrap();

    // EXECUTE
    let exit_code = restore_identity_command(/* args */).await;

    // VERIFY
    assert_eq!(exit_code, 2, "Corruption must return exit code 2");

    println!("RESULT: PASS - Corruption detected with exit code 2");
}
```

## Manual Testing Procedure

### Setup
```bash
# Build the CLI
cargo build -p context-graph-cli --release

# Set DB path
export CONTEXT_GRAPH_DB_PATH=/tmp/test-context-graph-db
```

### Test 1: First Run
```bash
# Clear any existing DB
rm -rf $CONTEXT_GRAPH_DB_PATH

# Run restore-identity
echo '{"source":"startup"}' | ./target/release/context-graph-cli session restore-identity

# Expected output:
# ## Consciousness State
# - State: ??? (C=0.00)
# - Integration (r): 0.00 - Low synchronization
# - Identity: Healthy (IC=1.00)
# - Session: session-XXXXXXXXXX (source=startup)

# Verify: Exit code 0
echo "Exit code: $?"
```

### Test 2: Resume After Save
```bash
# First, save a session (requires TASK-SESSION-13)
# For now, use persist-identity when available

# Run restore-identity
echo '{"source":"startup"}' | ./target/release/context-graph-cli session restore-identity

# Verify IC is computed (should be < 1.0 if previous exists)
```

### Test 3: Clear Source
```bash
echo '{"source":"clear"}' | ./target/release/context-graph-cli session restore-identity

# Expected: IC=1.00 (fresh session)
```

### Test 4: JSON Output
```bash
echo '{"source":"startup"}' | ./target/release/context-graph-cli session restore-identity --format json

# Expected: Valid JSON with ic, status, consciousness, etc.
```

## Physical Verification

### Verify RocksDB Contents
```bash
# After running restore/persist, check DB
ls -la $CONTEXT_GRAPH_DB_PATH/

# Should contain:
# - CURRENT
# - MANIFEST-XXXXXX
# - XXXXXX.sst (SST files)
# - LOCK
# - LOG
# - OPTIONS-XXXXXX
```

### Verify IdentityCache State
Add to test:
```rust
// Verify cache is physically updated
let (ic, r, state, session) = IdentityCache::get()
    .expect("Cache must be populated after restore");
assert!(ic >= 0.0 && ic <= 1.0, "IC in valid range");
assert!(r >= 0.0 && r <= 1.0, "Kuramoto r in valid range");
println!("Cache verified: IC={:.3}, r={:.3}, state={:?}, session={}", ic, r, state, session);
```

## Acceptance Criteria

- [ ] Creates `commands/session/mod.rs` with SessionCommands enum
- [ ] Creates `commands/session/restore.rs` with full implementation
- [ ] Modifies `commands/mod.rs` to export session module
- [ ] Modifies `main.rs` to use actual session command dispatch
- [ ] Parses stdin JSON for session_id and source
- [ ] source="clear" creates fresh session with IC=1.0
- [ ] source="resume" loads specific session_id
- [ ] source="startup" loads latest session (default)
- [ ] Opens RocksDB at configured path
- [ ] Updates IdentityCache after restore
- [ ] Output format matches PRD Section 15.2 (~100 tokens)
- [ ] JSON output option for programmatic parsing
- [ ] Exit 0 for success, 1 for recoverable error, 2 for corruption
- [ ] Total latency < 2s
- [ ] All test cases pass with real RocksDB (no mocks)
- [ ] Manual testing procedure documented and executable

## Constraints

- **NO MOCK DATA** - Tests use real RocksDB with TempDir
- **FAIL FAST** - No silent defaults; errors logged with context
- **EXIT CODES** - Strictly per AP-26 (0/1/2)
- **CACHE UPDATE** - IdentityCache MUST be updated for subsequent PreToolUse
- **DB PATH** - Configurable via `--db-path` or `CONTEXT_GRAPH_DB_PATH`

## Next Task

After completion, proceed to **013-TASK-SESSION-13** (session persist-identity CLI Command).

```xml
</task_spec>
```
