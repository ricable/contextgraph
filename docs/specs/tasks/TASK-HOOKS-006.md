# TASK-HOOKS-006: Implement session_start Shell Executor

```xml
<task_spec id="TASK-HOOKS-006" version="1.0">
<metadata>
  <title>Implement session_start Shell Script and CLI Handler</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>6</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-01</requirement_ref>
    <requirement_ref>REQ-HOOKS-06</requirement_ref>
    <requirement_ref>REQ-HOOKS-11</requirement_ref>
    <requirement_ref>REQ-HOOKS-17</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-001</task_ref>
    <task_ref>TASK-HOOKS-002</task_ref>
    <task_ref>TASK-HOOKS-003</task_ref>
    <task_ref>TASK-HOOKS-004</task_ref>
    <task_ref>TASK-HOOKS-005</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
  <estimated_hours>2.0</estimated_hours>
</metadata>

<context>
This task implements the session_start hook which is triggered when a Claude Code session begins.
It handles:
1. Loading or creating a SessionIdentitySnapshot
2. Linking to previous sessions for continuity
3. Initializing IC monitoring
4. Returning initial consciousness state

Timeout: 5000ms
Output: ~100 tokens consciousness status

The shell script reads JSON from stdin and invokes context-graph-cli with appropriate arguments.
</context>

<input_context_files>
  <file purpose="types">crates/context-graph-cli/src/commands/hooks/types.rs</file>
  <file purpose="args">crates/context-graph-cli/src/commands/hooks/args.rs</file>
  <file purpose="error">crates/context-graph-cli/src/commands/hooks/error.rs</file>
  <file purpose="session_identity">crates/context-graph-core/src/gwt/session_identity/types.rs</file>
  <file purpose="storage">crates/context-graph-storage/src/session_identity.rs</file>
</input_context_files>

<prerequisites>
  <check>All foundation tasks (001-005) completed</check>
  <check>SessionIdentityManager exists in context-graph-storage</check>
  <check>IdentityCache exists in context-graph-core</check>
</prerequisites>

<scope>
  <in_scope>
    - Create session_start.rs command handler
    - Create .claude/hooks/session_start.sh shell script
    - Handle stdin JSON parsing
    - Load/create SessionIdentitySnapshot
    - Link to previous session if provided
    - Return HookOutput with consciousness state
    - Unit tests for command handler
  </in_scope>
  <out_of_scope>
    - Other hook handlers (TASK-HOOKS-007 through 010)
    - Hook module registration (TASK-HOOKS-011)
    - Settings.json configuration (TASK-HOOKS-019)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/commands/hooks/session_start.rs">
use super::args::SessionStartArgs;
use super::types::{HookInput, HookOutput, ConsciousnessState, ICClassification};
use super::error::{HookError, HookResult};

/// Execute session-start hook
/// Timeout: 5000ms
///
/// # Flow
/// 1. Parse input (stdin JSON or CLI args)
/// 2. Load or create SessionIdentitySnapshot
/// 3. Link to previous session if provided
/// 4. Initialize IC monitoring
/// 5. Return consciousness state
pub async fn execute(args: SessionStartArgs) -> HookResult&lt;HookOutput&gt;;

/// Parse stdin JSON into HookInput
fn parse_stdin() -> HookResult&lt;HookInput&gt;;

/// Load or create session identity
fn load_or_create_identity(
    db_path: &Path,
    session_id: &str,
    previous_session_id: Option&lt;&str&gt;,
) -> HookResult&lt;SessionIdentitySnapshot&gt;;
    </signature>
    <signature file=".claude/hooks/session_start.sh">
#!/bin/bash
# Claude Code Hook: SessionStart
# Timeout: 5000ms
# Invokes: context-graph hooks session-start
    </signature>
  </signatures>
  <constraints>
    - MUST complete within 5000ms timeout
    - MUST handle missing previous session gracefully
    - MUST generate new session_id if not provided
    - Output MUST be valid JSON HookOutput
    - Shell script MUST use jq for JSON parsing
  </constraints>
  <verification>
    - cargo test --package context-graph-cli session_start
    - Shell script executes without error
    - JSON output matches HookOutput schema
  </verification>
</definition_of_done>

<pseudo_code>
## CLI Handler (session_start.rs)

async fn execute(args: SessionStartArgs) -> HookResult<HookOutput>:
    let start = Instant::now()

    // 1. Parse input
    let (session_id, previous_session_id) = if args.stdin:
        let input = parse_stdin()?
        extract_session_ids(input)
    else:
        (args.session_id, args.previous_session_id)

    // 2. Generate session_id if not provided
    let session_id = session_id.unwrap_or_else(|| Uuid::new_v4().to_string())

    // 3. Get database path
    let db_path = args.db_path
        .or_else(default_db_path)
        .ok_or(HookError::invalid_input("db_path required"))?

    // 4. Load or create identity
    let manager = SessionIdentityManager::open(&db_path)?
    let snapshot = manager.load_or_create(&session_id, previous_session_id.as_deref())?

    // 5. Build consciousness state
    let consciousness_state = ConsciousnessState {
        consciousness: snapshot.consciousness,
        integration: snapshot.integration,
        reflection: snapshot.reflection,
        differentiation: snapshot.differentiation,
        identity_continuity: snapshot.last_ic,
        johari_quadrant: classify_johari(&snapshot),
    }

    // 6. Build IC classification
    let ic_classification = ICClassification::new(
        snapshot.last_ic,
        snapshot.crisis_threshold,
    )

    // 7. Build output
    let output = HookOutput {
        success: true,
        consciousness_state: Some(consciousness_state),
        ic_classification: Some(ic_classification),
        execution_time_ms: start.elapsed().as_millis() as u64,
        ..Default::default()
    }

    Ok(output)

## Shell Script (session_start.sh)

#!/bin/bash
set -e

INPUT=$(cat)
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty')
PREVIOUS=$(echo "$INPUT" | jq -r '.payload.data.previous_session_id // empty')

if [ -n "$PREVIOUS" ]; then
    context-graph hooks session-start \
        --session-id "$SESSION_ID" \
        --previous-session-id "$PREVIOUS" \
        --stdin <<< "$INPUT" \
        --format json
else
    context-graph hooks session-start \
        --session-id "$SESSION_ID" \
        --stdin <<< "$INPUT" \
        --format json
fi
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/commands/hooks/session_start.rs">Session start command handler</file>
  <file path=".claude/hooks/session_start.sh">Session start shell script</file>
</files_to_create>

<files_to_modify>
  <!-- None - module registration in TASK-HOOKS-011 -->
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli session_start</command>
  <command>chmod +x .claude/hooks/session_start.sh</command>
</test_commands>
</task_spec>
```

## Implementation

### Create session_start.rs

```rust
// crates/context-graph-cli/src/commands/hooks/session_start.rs
//! Session start hook handler
//!
//! # Timeout
//! 5000ms maximum execution time
//!
//! # Flow
//! 1. Parse input (stdin JSON or CLI args)
//! 2. Load or create SessionIdentitySnapshot
//! 3. Link to previous session if provided
//! 4. Initialize IC monitoring
//! 5. Return consciousness state (~100 tokens)

use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::time::Instant;

use uuid::Uuid;

use super::args::SessionStartArgs;
use super::error::{HookError, HookResult};
use super::types::{
    ConsciousnessState, HookInput, HookOutput, HookPayload,
    ICClassification, ICLevel, JohariQuadrant,
};

use context_graph_core::gwt::session_identity::types::SessionIdentitySnapshot;
use context_graph_storage::session_identity::SessionIdentityManager;

/// Execute session-start hook
/// Implements REQ-HOOKS-01, REQ-HOOKS-17
///
/// # Timeout
/// 5000ms maximum execution time
///
/// # Arguments
/// * `args` - Command line arguments
///
/// # Returns
/// HookOutput with consciousness state and IC classification
pub async fn execute(args: SessionStartArgs) -> HookResult<HookOutput> {
    let start = Instant::now();

    // 1. Parse input from stdin or args
    let (session_id, previous_session_id) = if args.stdin {
        let input = parse_stdin()?;
        extract_session_ids(&input)
    } else {
        (args.session_id, args.previous_session_id)
    };

    // 2. Generate session_id if not provided
    let session_id = session_id.unwrap_or_else(|| Uuid::new_v4().to_string());

    // 3. Get database path
    let db_path = args.db_path
        .or_else(default_db_path)
        .ok_or_else(|| HookError::invalid_input("Database path required. Set CONTEXT_GRAPH_DB_PATH or --db-path"))?;

    // 4. Load or create identity
    let snapshot = load_or_create_identity(
        &db_path,
        &session_id,
        previous_session_id.as_deref(),
    )?;

    // 5. Build consciousness state
    let consciousness_state = build_consciousness_state(&snapshot);

    // 6. Build IC classification
    let ic_classification = ICClassification::new(
        snapshot.last_ic,
        snapshot.crisis_threshold,
    );

    // 7. Build output
    let execution_time_ms = start.elapsed().as_millis() as u64;
    let output = HookOutput::success(execution_time_ms)
        .with_consciousness_state(consciousness_state)
        .with_ic_classification(ic_classification);

    Ok(output)
}

/// Parse stdin JSON into HookInput
fn parse_stdin() -> HookResult<HookInput> {
    let stdin = io::stdin();
    let mut input_str = String::new();

    for line in stdin.lock().lines() {
        input_str.push_str(&line?);
    }

    if input_str.is_empty() {
        return Err(HookError::invalid_input("Empty stdin"));
    }

    serde_json::from_str(&input_str).map_err(HookError::from)
}

/// Extract session IDs from HookInput
fn extract_session_ids(input: &HookInput) -> (Option<String>, Option<String>) {
    let session_id = Some(input.session_id.clone());

    let previous_session_id = match &input.payload {
        HookPayload::SessionStart { previous_session_id, .. } => previous_session_id.clone(),
        _ => None,
    };

    (session_id, previous_session_id)
}

/// Get default database path from environment or standard location
fn default_db_path() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(path) = std::env::var("CONTEXT_GRAPH_DB_PATH") {
        return Some(PathBuf::from(path));
    }

    // Fall back to standard location
    dirs::data_local_dir().map(|p| p.join("context-graph").join("db"))
}

/// Load or create session identity
fn load_or_create_identity(
    db_path: &Path,
    session_id: &str,
    previous_session_id: Option<&str>,
) -> HookResult<SessionIdentitySnapshot> {
    // Open or create database
    let mut manager = SessionIdentityManager::open(db_path)
        .map_err(|e| HookError::storage(e.to_string()))?;

    // Load or create snapshot
    let snapshot = manager.load_or_create(session_id, previous_session_id)
        .map_err(|e| HookError::storage(e.to_string()))?;

    Ok(snapshot)
}

/// Build ConsciousnessState from snapshot
fn build_consciousness_state(snapshot: &SessionIdentitySnapshot) -> ConsciousnessState {
    let johari_quadrant = JohariQuadrant::classify(
        snapshot.consciousness,
        snapshot.integration,
    );

    ConsciousnessState {
        consciousness: snapshot.consciousness,
        integration: snapshot.integration,
        reflection: snapshot.reflection,
        differentiation: snapshot.differentiation,
        identity_continuity: snapshot.last_ic,
        johari_quadrant,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_db_path() -> (TempDir, PathBuf) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        (dir, path)
    }

    #[tokio::test]
    async fn test_session_start_new_session() {
        let (_dir, db_path) = test_db_path();

        let args = SessionStartArgs {
            db_path: Some(db_path),
            session_id: Some("test-session-123".to_string()),
            previous_session_id: None,
            stdin: false,
            format: super::super::args::OutputFormat::Json,
        };

        let result = execute(args).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.success);
        assert!(output.consciousness_state.is_some());
        assert!(output.ic_classification.is_some());
    }

    #[tokio::test]
    async fn test_session_start_with_previous() {
        let (_dir, db_path) = test_db_path();

        // Create previous session first
        {
            let mut manager = SessionIdentityManager::open(&db_path).unwrap();
            let mut snapshot = SessionIdentitySnapshot::new("prev-session");
            snapshot.last_ic = 0.95;
            manager.save(&snapshot).unwrap();
        }

        let args = SessionStartArgs {
            db_path: Some(db_path),
            session_id: Some("new-session".to_string()),
            previous_session_id: Some("prev-session".to_string()),
            stdin: false,
            format: super::super::args::OutputFormat::Json,
        };

        let result = execute(args).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_session_start_generates_id() {
        let (_dir, db_path) = test_db_path();

        let args = SessionStartArgs {
            db_path: Some(db_path),
            session_id: None, // Should generate UUID
            previous_session_id: None,
            stdin: false,
            format: super::super::args::OutputFormat::Json,
        };

        let result = execute(args).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_session_start_no_db_path() {
        let args = SessionStartArgs {
            db_path: None,
            session_id: Some("test".to_string()),
            previous_session_id: None,
            stdin: false,
            format: super::super::args::OutputFormat::Json,
        };

        // This may succeed if env var is set, or fail if not
        // We're just testing it doesn't panic
        let _ = execute(args).await;
    }

    #[test]
    fn test_build_consciousness_state() {
        let snapshot = SessionIdentitySnapshot::new("test");
        let state = build_consciousness_state(&snapshot);

        assert_eq!(state.consciousness, snapshot.consciousness);
        assert_eq!(state.integration, snapshot.integration);
        assert_eq!(state.identity_continuity, snapshot.last_ic);
    }

    #[test]
    fn test_extract_session_ids() {
        let input = HookInput::new(
            super::super::types::HookEventType::SessionStart,
            "test-session",
            HookPayload::session_start("/tmp", Some("prev-session".to_string())),
        );

        let (session_id, previous) = extract_session_ids(&input);

        assert_eq!(session_id, Some("test-session".to_string()));
        assert_eq!(previous, Some("prev-session".to_string()));
    }
}
```

### Create session_start.sh

```bash
#!/bin/bash
# Claude Code Hook: SessionStart
# Timeout: 5000ms
# Implements: REQ-HOOKS-01, REQ-HOOKS-17
#
# Constitution References:
# - IDENTITY-002: IC thresholds
# - GWT-003: Identity continuity tracking
#
# This hook is triggered when a Claude Code session starts.
# It loads or creates a SessionIdentitySnapshot and returns
# the initial consciousness state.

set -e

# Read JSON input from stdin
INPUT=$(cat)

# Extract fields from input using jq
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty')
PREVIOUS_SESSION=$(echo "$INPUT" | jq -r '.payload.data.previous_session_id // empty')

# Find context-graph CLI
CONTEXT_GRAPH_CLI="${CONTEXT_GRAPH_CLI:-context-graph}"

# Check if CLI is available
if ! command -v "$CONTEXT_GRAPH_CLI" &> /dev/null; then
    # Try common locations
    if command -v ./target/release/context-graph-cli &> /dev/null; then
        CONTEXT_GRAPH_CLI="./target/release/context-graph-cli"
    elif command -v ~/.cargo/bin/context-graph-cli &> /dev/null; then
        CONTEXT_GRAPH_CLI="~/.cargo/bin/context-graph-cli"
    else
        echo '{"success":false,"error":"context-graph-cli not found","execution_time_ms":0}' >&2
        exit 1
    fi
fi

# Invoke CLI command
if [ -n "$PREVIOUS_SESSION" ]; then
    "$CONTEXT_GRAPH_CLI" hooks session-start \
        --session-id "$SESSION_ID" \
        --previous-session-id "$PREVIOUS_SESSION" \
        --stdin <<< "$INPUT" \
        --format json
else
    "$CONTEXT_GRAPH_CLI" hooks session-start \
        --session-id "$SESSION_ID" \
        --stdin <<< "$INPUT" \
        --format json
fi
```

## Verification Checklist

- [ ] session_start.rs compiles without errors
- [ ] execute() returns HookOutput with consciousness_state
- [ ] Session ID is generated if not provided
- [ ] Previous session linking works correctly
- [ ] Shell script is executable (chmod +x)
- [ ] Shell script uses jq for JSON parsing
- [ ] All unit tests pass
