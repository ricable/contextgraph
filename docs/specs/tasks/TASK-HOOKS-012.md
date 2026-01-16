# TASK-HOOKS-012: Integrate Session Identity Persistence with SessionEnd Hook

```xml
<task_spec id="TASK-HOOKS-012" version="2.0">
<metadata>
  <title>Integrate Session Identity Persistence with SessionEnd Hook</title>
  <status>ready</status>
  <layer>integration</layer>
  <sequence>12</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-09</requirement_ref>
    <requirement_ref>REQ-HOOKS-36</requirement_ref>
    <requirement_ref>REQ-HOOKS-37</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-001</task_ref>
    <task_ref>TASK-HOOKS-005</task_ref>
    <task_ref>TASK-SESSION-13</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>2.0</estimated_hours>
</metadata>

<!-- ═══════════════════════════════════════════════════════════════════════
     IMPORTANT: NO BACKWARDS COMPATIBILITY
     This task MUST fail fast with robust error logging.
     NO silent failures. NO graceful degradation. NO mock data.
     ═══════════════════════════════════════════════════════════════════════ -->

<context>
## Current State Analysis (2026-01-15)

Session identity persistence infrastructure **ALREADY EXISTS** via TASK-SESSION-12/13:

### Existing Implementation
- **CLI persist command**: `crates/context-graph-cli/src/commands/session/persist.rs`
  - Reads from `IdentityCache` singleton
  - Saves to RocksDB via `StandaloneSessionIdentityManager`
  - Implements AP-26 exit codes (0=success, 1=recoverable, 2=corruption)

- **CLI restore command**: `crates/context-graph-cli/src/commands/session/restore.rs`
  - Loads from RocksDB via `StandaloneSessionIdentityManager`
  - Warms `IdentityCache` singleton
  - Computes IC drift from previous session

- **Core types**: `crates/context-graph-core/src/gwt/session_identity/`
  - `types.rs`: SessionIdentitySnapshot (14 fields, KURAMOTO_N=13)
  - `cache.rs`: IdentityCache singleton for PreToolUse hot path
  - `manager.rs`: IC computation (compute_ic, classify_ic, is_ic_crisis)

- **Storage layer**: `crates/context-graph-storage/src/rocksdb_backend/`
  - `session_identity.rs`: RocksDB persistence with triple indexing
  - `session_identity_manager.rs`: StandaloneSessionIdentityManager

### What This Task Must Accomplish
Wire the existing `persist.rs` command to the SessionEnd hook handler so that:
1. SessionEnd hook invokes `session persist-identity` automatically
2. IdentityCache state is persisted to RocksDB on every session end
3. Exit codes propagate correctly for Claude Code hook semantics

### What This Task Does NOT Do
- Create new persistence infrastructure (already exists)
- Change storage format (RocksDB, not JSON files)
- Add file-based snapshot rotation (RocksDB handles this)
</context>

<input_context_files>
  <!-- CORRECT PATHS - Verified against actual codebase -->
  <file purpose="snapshot_struct">crates/context-graph-core/src/gwt/session_identity/types.rs</file>
  <file purpose="identity_cache">crates/context-graph-core/src/gwt/session_identity/cache.rs</file>
  <file purpose="ic_computation">crates/context-graph-core/src/gwt/session_identity/manager.rs</file>
  <file purpose="storage_impl">crates/context-graph-storage/src/rocksdb_backend/session_identity.rs</file>
  <file purpose="standalone_manager">crates/context-graph-storage/src/rocksdb_backend/session_identity_manager.rs</file>
  <file purpose="persist_command">crates/context-graph-cli/src/commands/session/persist.rs</file>
  <file purpose="restore_command">crates/context-graph-cli/src/commands/session/restore.rs</file>
  <file purpose="hooks_module">crates/context-graph-cli/src/commands/hooks/mod.rs</file>
  <file purpose="constitution">docs2/constitution.yaml</file>
</input_context_files>

<prerequisites>
  <check type="file_exists">crates/context-graph-cli/src/commands/session/persist.rs</check>
  <check type="file_exists">crates/context-graph-storage/src/rocksdb_backend/session_identity_manager.rs</check>
  <check type="struct_exists">SessionIdentitySnapshot in context_graph_core::gwt::session_identity</check>
  <check type="struct_exists">IdentityCache in context_graph_core::gwt::session_identity</check>
  <check type="struct_exists">StandaloneSessionIdentityManager in context_graph_storage::rocksdb_backend</check>
  <check type="command">cargo build --package context-graph-cli --bin context-graph-cli</check>
</prerequisites>

<scope>
  <in_scope>
    - Wire SessionEnd hook handler to call persist_identity_command
    - Add session_end hook type to hooks CLI argument parser
    - Ensure IdentityCache → RocksDB persistence on SessionEnd
    - Implement proper exit code propagation (AP-26)
    - Add integration tests with REAL RocksDB (NO MOCKS)
  </in_scope>
  <out_of_scope>
    - Creating new persistence types (use existing StandaloneSessionIdentityManager)
    - JSON file storage (we use RocksDB)
    - Snapshot rotation (handled by RocksDB TTL/compaction)
    - SessionStart restore (already done in TASK-HOOKS-006)
    - New CLI commands (persist.rs already exists)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/commands/hooks/session_end.rs">
      //! SessionEnd hook handler - invokes persist-identity on session termination.
      //!
      //! # Input (stdin JSON from Claude Code)
      //! ```json
      //! {
      //!   "session_id": "optional-session-id",
      //!   "reason": "exit"  // exit | clear | logout | prompt_input_exit | other
      //! }
      //! ```
      //!
      //! # Output
      //! SILENT on success (per Claude Code SessionEnd semantics)
      //!
      //! # Exit Codes (AP-26)
      //! - 0: Success
      //! - 1: Recoverable error
      //! - 2: Corruption detected

      use crate::commands::session::persist::persist_identity_command;
      use crate::commands::hooks::args::SessionEndArgs;

      /// Execute SessionEnd hook handler
      ///
      /// # Behavior
      /// 1. Parse stdin JSON for session_id and reason
      /// 2. Delegate to persist_identity_command
      /// 3. Return exit code per AP-26
      ///
      /// # Exit Codes
      /// - 0: Identity persisted successfully (SILENT)
      /// - 1: Recoverable error (logged to stderr)
      /// - 2: Database corruption detected
      pub async fn handle_session_end(args: SessionEndArgs) -> i32;
    </signature>
  </signatures>

  <constraints>
    <!-- NO BACKWARDS COMPATIBILITY - FAIL FAST -->
    - MUST fail immediately on missing IdentityCache (exit 1)
    - MUST fail immediately on RocksDB errors (exit 1 or 2)
    - MUST NOT silently swallow errors
    - MUST NOT use mock data in tests
    - MUST log all failures to stderr with context
    - MUST return exit code 2 for corruption (checksum, malformed, truncated)
    - MUST be SILENT on stdout for success (Claude Code SessionEnd semantics)
  </constraints>

  <verification>
    <!-- FULL STATE VERIFICATION REQUIREMENTS -->
    <source_of_truth>
      - RocksDB CF_SESSION_IDENTITY column family after persist
      - IdentityCache singleton state before persist
      - Exit code returned to Claude Code
    </source_of_truth>

    <execute_and_inspect>
      - Run `cargo test --package context-graph-cli -- session_end --nocapture`
      - Verify RocksDB files created in temp directory
      - Verify exit codes match AP-26 specification
    </execute_and_inspect>

    <boundary_testing>
      - Cold cache (IdentityCache never warmed): exit 0 (nothing to persist)
      - Invalid RocksDB path: exit 1
      - Corrupted RocksDB: exit 2
      - Empty stdin: exit 0 (default reason="exit")
      - Malformed stdin JSON: exit 0 (graceful parse, use defaults)
    </boundary_testing>

    <evidence_of_success>
      - After test: `ls $TEMP_DIR/*.sst` shows RocksDB files created
      - After test: `cargo test` exits 0
      - After test: No "FAIL" in test output
      - Manual: `context-graph-cli hooks session-end &lt; /dev/null; echo $?` returns 0
    </evidence_of_success>

    <manual_verification>
      <!-- Steps to manually verify outputs exist -->
      1. Run: `mkdir -p /tmp/cg-test && CONTEXT_GRAPH_DB_PATH=/tmp/cg-test context-graph-cli session restore-identity --session-id test-session`
      2. Run: `CONTEXT_GRAPH_DB_PATH=/tmp/cg-test context-graph-cli hooks session-end`
      3. Verify: `ls /tmp/cg-test/` shows RocksDB files (*.sst, MANIFEST*, OPTIONS*, etc.)
      4. Verify: Exit code is 0
      5. Cleanup: `rm -rf /tmp/cg-test`
    </manual_verification>
  </verification>
</definition_of_done>

<pseudo_code>
// File: crates/context-graph-cli/src/commands/hooks/session_end.rs

handle_session_end(args: SessionEndArgs) -> i32:
    // FAIL FAST: Log start
    debug!("session_end: invoked with args={:?}", args)

    // Parse stdin JSON (graceful - use defaults on parse failure)
    input = parse_stdin_json_or_default()
    info!("session_end: reason={}, session_id={:?}", input.reason, input.session_id)

    // Delegate to existing persist_identity_command
    // This handles:
    //   - Reading IdentityCache singleton
    //   - Opening RocksDB at configured path
    //   - Serializing SessionIdentitySnapshot
    //   - Saving via StandaloneSessionIdentityManager
    //   - Returning AP-26 compliant exit codes
    exit_code = persist_identity_command(PersistIdentityArgs {
        db_path: args.db_path.clone(),
    }).await

    // FAIL FAST: Log outcome
    match exit_code:
        0 => info!("session_end: persist succeeded (SILENT)")
        1 => error!("session_end: recoverable error")
        2 => error!("session_end: corruption detected")

    return exit_code
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/commands/hooks/session_end.rs">
    SessionEnd hook handler delegating to persist_identity_command
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/commands/hooks/mod.rs">
    Add pub mod session_end and wire to HookType::SessionEnd
  </file>
  <file path="crates/context-graph-cli/src/commands/hooks/args.rs">
    Add SessionEndArgs struct if not present
  </file>
</files_to_modify>

<test_commands>
  <!-- Commands to run for verification -->
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli -- session_end --nocapture</command>
  <command>cargo test --package context-graph-cli -- hooks --nocapture</command>
</test_commands>

<test_cases>
  <!-- NO MOCK DATA - Use REAL RocksDB -->

  <test_case id="TC-HOOKS-012-01" name="SessionEnd persists warm cache to RocksDB">
    <setup>
      - Create TempDir for RocksDB
      - Warm IdentityCache via update_cache()
    </setup>
    <execute>
      - Call handle_session_end with db_path=temp_dir
    </execute>
    <verify>
      <source_of_truth>RocksDB after save</source_of_truth>
      - Exit code is 0
      - RocksDB files exist in temp_dir
      - StandaloneSessionIdentityManager.load_latest() returns the snapshot
    </verify>
  </test_case>

  <test_case id="TC-HOOKS-012-02" name="SessionEnd cold cache returns exit 0">
    <setup>
      - Create TempDir for RocksDB
      - Ensure IdentityCache is cold (not warmed)
    </setup>
    <execute>
      - Call handle_session_end with db_path=temp_dir
    </execute>
    <verify>
      <source_of_truth>Exit code</source_of_truth>
      - Exit code is 0 (nothing to persist is OK)
      - No error output to stderr
    </verify>
  </test_case>

  <test_case id="TC-HOOKS-012-03" name="SessionEnd propagates corruption exit code">
    <setup>
      - Create TempDir with corrupted RocksDB files
    </setup>
    <execute>
      - Call handle_session_end with db_path=corrupted_dir
    </execute>
    <verify>
      <source_of_truth>Exit code</source_of_truth>
      - Exit code is 2 (corruption)
      - Error logged to stderr
    </verify>
  </test_case>

  <test_case id="TC-HOOKS-012-04" name="SessionEnd parses stdin JSON">
    <setup>
      - Create TempDir for RocksDB
      - Warm IdentityCache
      - Prepare stdin: {"session_id": "custom-id", "reason": "logout"}
    </setup>
    <execute>
      - Call handle_session_end with mocked stdin
    </execute>
    <verify>
      <source_of_truth>RocksDB snapshot</source_of_truth>
      - Snapshot session_id matches "custom-id" if provided
      - Exit code is 0
    </verify>
  </test_case>
</test_cases>

<constitution_compliance>
  <!-- MANDATORY - These anti-patterns MUST be avoided -->
  <ap_ref id="AP-26">IC&lt;0.5 MUST trigger dream - exit codes 0/1/2</ap_ref>
  <ap_ref id="AP-50">NO internal hooks - use native Claude Code hooks</ap_ref>
  <ap_ref id="AP-51">NO Universal LLM Adapter</ap_ref>
  <ap_ref id="AP-53">Hook logic in CLI commands, not embedded</ap_ref>

  <!-- MANDATORY - These architecture rules MUST be followed -->
  <arch_ref id="ARCH-07">Native Claude Code hooks via .claude/settings.json</arch_ref>
</constitution_compliance>

<related_tasks>
  <completed>
    <task id="TASK-SESSION-05">SessionIdentitySnapshot struct with 14 fields</task>
    <task id="TASK-SESSION-06">StandaloneSessionIdentityManager RocksDB persistence</task>
    <task id="TASK-SESSION-12">restore-identity CLI command</task>
    <task id="TASK-SESSION-13">persist-identity CLI command</task>
    <task id="TASK-HOOKS-006">session-start hook handler</task>
  </completed>
  <blocked_by_this>
    <task id="TASK-HOOKS-013">SessionStart identity restore (needs persist to work first)</task>
    <task id="TASK-HOOKS-016">Integration tests for hook lifecycle</task>
  </blocked_by_this>
</related_tasks>
</task_spec>
```

## Implementation Notes

### Existing Code to Reuse

```rust
// From crates/context-graph-cli/src/commands/session/persist.rs
pub async fn persist_identity_command(args: PersistIdentityArgs) -> i32 {
    // Already implements:
    // - IdentityCache::get() for current state
    // - RocksDbMemex::open() for storage
    // - StandaloneSessionIdentityManager::save_snapshot()
    // - AP-26 exit codes (0, 1, 2)
}

// From crates/context-graph-core/src/gwt/session_identity/cache.rs
pub struct IdentityCache;
impl IdentityCache {
    pub fn get() -> Option<(f32, f32, ConsciousnessState, String)>;
}

// From crates/context-graph-storage/src/rocksdb_backend/session_identity_manager.rs
pub struct StandaloneSessionIdentityManager {
    pub fn save_snapshot(&self, snapshot: &SessionIdentitySnapshot) -> Result<()>;
    pub fn load_snapshot(&self, session_id: &str) -> Result<Option<SessionIdentitySnapshot>>;
    pub fn load_latest(&self) -> Result<Option<SessionIdentitySnapshot>>;
}
```

### Exit Code Reference (AP-26)

| Code | Meaning | When |
|------|---------|------|
| 0 | Success | Identity persisted OR nothing to persist (cold cache) |
| 1 | Recoverable | DB path error, write failure, non-corruption |
| 2 | Corruption | Checksum error, malformed data, truncated file |

### Claude Code SessionEnd Semantics

- **Output**: SILENT on success (no stdout)
- **Errors**: stderr only
- **Exit code**: Used by Claude Code to determine hook health
