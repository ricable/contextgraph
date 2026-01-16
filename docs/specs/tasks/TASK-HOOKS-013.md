# TASK-HOOKS-013: Add Drift Metrics Computation to Session Identity Restoration

```xml
<task_spec id="TASK-HOOKS-013" version="2.0">
<metadata>
  <title>Add Drift Metrics Computation to Session Identity Restoration</title>
  <status>ready</status>
  <layer>enhancement</layer>
  <sequence>13</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-12</requirement_ref>
    <requirement_ref>REQ-HOOKS-13</requirement_ref>
    <requirement_ref>REQ-HOOKS-14</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-001</task_ref>
    <task_ref>TASK-HOOKS-006</task_ref>
    <task_ref>TASK-HOOKS-012</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>2.5</estimated_hours>
</metadata>

<!-- ═══════════════════════════════════════════════════════════════════════
     CRITICAL: NO BACKWARDS COMPATIBILITY
     This task MUST fail fast with robust error logging.
     NO silent failures. NO graceful degradation. NO mock data.
     ═══════════════════════════════════════════════════════════════════════ -->

<context>
## Current State Analysis (2026-01-15)

**IMPORTANT**: The original task specification was INCORRECT about file paths and what needs to be created.

### What ALREADY EXISTS (DO NOT RECREATE)

1. **Session Identity Types** (`crates/context-graph-core/src/gwt/session_identity/types.rs`):
   - `SessionIdentitySnapshot` struct with 14 fields
   - `KURAMOTO_N = 13` oscillators
   - `MAX_TRAJECTORY_LEN = 50` entries

2. **IC Computation Functions** (`crates/context-graph-core/src/gwt/session_identity/manager.rs`):
   - `compute_ic(current, previous) -> f32` - Cross-session IC
   - `classify_ic(ic) -> ICClassification` - Threshold classification
   - `is_ic_crisis(ic) -> bool` - Crisis detection (IC < 0.5)
   - `is_ic_warning(ic) -> bool` - Warning detection (IC < 0.7)
   - `compute_kuramoto_r(phases) -> f64` - Synchronization order parameter

3. **Session Start Hook** (`crates/context-graph-cli/src/commands/hooks/session_start.rs`):
   - `execute(args: SessionStartArgs)` - Main handler
   - `load_or_create_snapshot()` - Creates new or loads existing snapshot
   - **ALREADY links previous session** via `previous_session_id`
   - **ALREADY computes cross_session_ic** using `compute_ic()`

4. **Session End Hook** (`crates/context-graph-cli/src/commands/hooks/session_end.rs`):
   - `execute(args: SessionEndArgs)` - Main handler
   - `persist_to_storage()` - Flushes IdentityCache to RocksDB
   - Full test coverage with real RocksDB

5. **CLI Restore Command** (`crates/context-graph-cli/src/commands/session/restore.rs`):
   - `restore_identity_command()` - Standalone restore
   - Handles startup/resume/clear sources
   - Full test coverage (9 test cases)

6. **RocksDB Storage** (`crates/context-graph-storage/src/rocksdb_backend/`):
   - `session_identity.rs` - Column family operations
   - `session_identity_manager.rs` - `StandaloneSessionIdentityManager`

### What This Task ACTUALLY Needs to Implement

The **MISSING** functionality is `DriftMetrics` computation:

```rust
/// Drift metrics computed during session restoration.
/// NOT currently implemented anywhere in the codebase.
pub struct DriftMetrics {
    /// IC delta: current_ic - previous_ic
    pub ic_delta: f32,
    /// Purpose vector cosine distance
    pub purpose_drift: f32,
    /// Time elapsed since previous snapshot
    pub time_since_snapshot_ms: i64,
    /// Kuramoto phase drift (mean absolute difference)
    pub kuramoto_phase_drift: f64,
}
```

This task adds drift metric computation to the existing session_start hook and exposes it in the `HookOutput`.
</context>

<input_context_files>
  <!-- CORRECT PATHS - Verified against actual codebase 2026-01-15 -->
  <file purpose="snapshot_struct">crates/context-graph-core/src/gwt/session_identity/types.rs</file>
  <file purpose="ic_computation">crates/context-graph-core/src/gwt/session_identity/manager.rs</file>
  <file purpose="identity_cache">crates/context-graph-core/src/gwt/session_identity/cache.rs</file>
  <file purpose="session_start_hook">crates/context-graph-cli/src/commands/hooks/session_start.rs</file>
  <file purpose="session_end_hook">crates/context-graph-cli/src/commands/hooks/session_end.rs</file>
  <file purpose="hook_types">crates/context-graph-cli/src/commands/hooks/types.rs</file>
  <file purpose="hook_args">crates/context-graph-cli/src/commands/hooks/args.rs</file>
  <file purpose="restore_command">crates/context-graph-cli/src/commands/session/restore.rs</file>
  <file purpose="storage_impl">crates/context-graph-storage/src/rocksdb_backend/session_identity.rs</file>
  <file purpose="standalone_manager">crates/context-graph-storage/src/rocksdb_backend/session_identity_manager.rs</file>
  <file purpose="constitution">docs2/constitution.yaml</file>
</input_context_files>

<prerequisites>
  <!-- Verify these files exist BEFORE starting -->
  <check type="file_exists">crates/context-graph-core/src/gwt/session_identity/types.rs</check>
  <check type="file_exists">crates/context-graph-core/src/gwt/session_identity/manager.rs</check>
  <check type="file_exists">crates/context-graph-cli/src/commands/hooks/session_start.rs</check>
  <check type="file_exists">crates/context-graph-cli/src/commands/hooks/types.rs</check>
  <check type="struct_exists">SessionIdentitySnapshot in context_graph_core::gwt::session_identity</check>
  <check type="function_exists">compute_ic in context_graph_core::gwt::session_identity::manager</check>
  <check type="function_exists">load_or_create_snapshot in session_start.rs</check>
  <check type="command">cargo build --package context-graph-cli --bin context-graph-cli</check>
</prerequisites>

<scope>
  <in_scope>
    <!-- ONLY these items - nothing else -->
    - Create DriftMetrics struct in types.rs
    - Add compute_drift_metrics() function in session_start.rs
    - Modify load_or_create_snapshot() to compute drift when linking sessions
    - Add drift_metrics field to HookOutput
    - Add tests for drift computation with REAL RocksDB
  </in_scope>
  <out_of_scope>
    <!-- These are EXPLICITLY out of scope - DO NOT implement -->
    - Creating SnapshotRestorer service (original task was wrong - restoration exists)
    - Creating crates/context-graph-cli/src/identity/ directory (doesn't exist, not needed)
    - New CLI commands (restore command already exists in session/)
    - Automatic crisis recovery (different feature)
    - Multi-snapshot merging (future feature)
    - Snapshot freshness validation (already handled by IC thresholds)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <!-- File 1: Add DriftMetrics to types.rs -->
    <signature file="crates/context-graph-cli/src/commands/hooks/types.rs">
      /// Drift metrics computed when linking to a previous session.
      ///
      /// # Constitution References
      /// - IDENTITY-002: IC thresholds determine drift severity
      /// - GWT-003: Identity continuity tracking
      #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
      pub struct DriftMetrics {
          /// IC delta: (current_ic - previous_ic)
          /// Positive = improvement, Negative = degradation
          pub ic_delta: f32,

          /// Purpose vector cosine distance [0.0, 2.0]
          /// 0.0 = identical, 2.0 = opposite
          pub purpose_drift: f32,

          /// Time elapsed since previous snapshot (milliseconds)
          pub time_since_snapshot_ms: i64,

          /// Kuramoto phase drift: mean |phase_i - prev_phase_i|
          /// Range [0.0, π]
          pub kuramoto_phase_drift: f64,
      }

      impl DriftMetrics {
          /// Check if drift indicates identity crisis per IDENTITY-002
          /// Returns true if ic_delta < -0.3 (30% IC drop)
          pub fn is_crisis_drift(&amp;self) -> bool;

          /// Check if drift indicates warning level
          /// Returns true if ic_delta < -0.1 (10% IC drop)
          pub fn is_warning_drift(&amp;self) -> bool;
      }
    </signature>

    <!-- File 2: Add drift computation to session_start.rs -->
    <signature file="crates/context-graph-cli/src/commands/hooks/session_start.rs">
      use super::types::DriftMetrics;
      use std::f64::consts::PI;

      /// Compute drift metrics between current and previous session.
      ///
      /// # Arguments
      /// * `current` - Current session snapshot
      /// * `previous` - Previous session snapshot
      ///
      /// # Returns
      /// DriftMetrics with all four measurements populated
      fn compute_drift_metrics(
          current: &amp;SessionIdentitySnapshot,
          previous: &amp;SessionIdentitySnapshot,
      ) -> DriftMetrics;

      /// Compute cosine distance between two purpose vectors.
      /// Returns value in range [0.0, 2.0].
      #[inline]
      fn cosine_distance(a: &amp;[f32; KURAMOTO_N], b: &amp;[f32; KURAMOTO_N]) -> f32;

      /// Compute mean absolute phase difference.
      /// Returns value in range [0.0, π].
      #[inline]
      fn phase_drift(a: &amp;[f64; KURAMOTO_N], b: &amp;[f64; KURAMOTO_N]) -> f64;
    </signature>

    <!-- File 3: Update HookOutput to include drift -->
    <signature file="crates/context-graph-cli/src/commands/hooks/types.rs">
      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct HookOutput {
          // ... existing fields ...

          /// Drift metrics from session restoration (only present on session_start with previous_session_id)
          #[serde(skip_serializing_if = "Option::is_none")]
          pub drift_metrics: Option&lt;DriftMetrics&gt;,
      }

      impl HookOutput {
          /// Add drift metrics to output
          pub fn with_drift_metrics(self, drift: DriftMetrics) -> Self;
      }
    </signature>
  </signatures>

  <constraints>
    <!-- NO BACKWARDS COMPATIBILITY - FAIL FAST -->
    - MUST compute drift when previous_session_id is provided and snapshot found
    - MUST NOT compute drift for new sessions (no previous)
    - MUST use KURAMOTO_N=13 for all array operations
    - MUST clamp phase drift to [0.0, π]
    - MUST clamp cosine distance to [0.0, 2.0]
    - MUST log drift computation at INFO level
    - MUST log warning if is_warning_drift()
    - MUST log error if is_crisis_drift()
    - MUST NOT silently swallow computation errors
    - MUST NOT use mock data in tests
  </constraints>
</definition_of_done>

<verification>
  <!-- FULL STATE VERIFICATION REQUIREMENTS -->

  <source_of_truth>
    <!-- PRIMARY: Computed DriftMetrics in memory -->
    - DriftMetrics struct populated by compute_drift_metrics()
    - HookOutput.drift_metrics field in JSON response

    <!-- SECONDARY: RocksDB snapshot state -->
    - Previous snapshot loaded from RocksDB CF_SESSION_IDENTITY
    - Current snapshot with cross_session_ic computed

    <!-- TERTIARY: Log output -->
    - INFO log with drift values
    - WARN log if warning_drift
    - ERROR log if crisis_drift
  </source_of_truth>

  <execute_and_inspect>
    <!-- Run these commands and VERIFY output -->
    <step>cargo build --package context-graph-cli</step>
    <step>cargo test --package context-graph-cli -- drift_metrics --nocapture</step>
    <step>cargo test --package context-graph-cli -- session_start --nocapture</step>

    <!-- Inspect these conditions -->
    <inspect>Test output shows DriftMetrics fields populated</inspect>
    <inspect>No "FAIL" strings in test output</inspect>
    <inspect>Exit code is 0 for all commands</inspect>
  </execute_and_inspect>

  <boundary_testing>
    <!-- Edge cases that MUST be tested -->
    | Case | Input | Expected Output |
    |------|-------|-----------------|
    | No previous session | previous_session_id=None | drift_metrics=None |
    | Previous not found | previous_session_id="nonexistent" | drift_metrics=None, warn log |
    | Identical snapshots | same purpose_vector, phases | ic_delta=0, purpose_drift=0, phase_drift=0 |
    | Opposite purposes | cos_distance=2.0 | purpose_drift=2.0 |
    | Large time gap | >7 days | time_since_snapshot_ms>604800000, warn log |
    | Zero vectors | [0.0; 13] | Graceful handling (NaN check) |
    | Maximum drift | all phases π apart | kuramoto_phase_drift ≈ π |
  </boundary_testing>

  <evidence_of_success>
    <!-- Physical verification that task is complete -->
    1. Run: `cargo test --package context-graph-cli -- tc_hooks_013 --nocapture`
       Expect: All tests PASS, no FAIL strings

    2. Run: `grep -r "DriftMetrics" crates/context-graph-cli/src/`
       Expect: Struct defined in types.rs, used in session_start.rs

    3. Run: `grep -r "compute_drift_metrics" crates/context-graph-cli/src/`
       Expect: Function defined and called in session_start.rs

    4. Run: `grep -r "with_drift_metrics" crates/context-graph-cli/src/`
       Expect: Method on HookOutput, called when drift computed

    5. Manual test (see manual_verification section below)
  </evidence_of_success>

  <manual_verification>
    <!-- Steps to manually verify with SYNTHETIC DATA -->

    ## Setup Test Environment
    ```bash
    # Create temp directory for test database
    export TEST_DB="/tmp/cg-drift-test-$(date +%s)"
    mkdir -p "$TEST_DB"
    ```

    ## Step 1: Create Previous Session
    ```bash
    # Start a session to create baseline snapshot
    echo '{"session_id":"prev-session-123","event_type":"session_start","payload":{"session_start":{"cwd":"/tmp","previous_session_id":null}}}' | \
      CONTEXT_GRAPH_DB_PATH="$TEST_DB" \
      ./target/debug/context-graph-cli hooks session-start --stdin

    # Expected output: JSON with success=true, drift_metrics=null (no previous)
    ```

    ## Step 2: End Previous Session (Persist)
    ```bash
    echo '{"session_id":"prev-session-123","event_type":"session_end","payload":{"session_end":{"status":"normal"}}}' | \
      CONTEXT_GRAPH_DB_PATH="$TEST_DB" \
      ./target/debug/context-graph-cli hooks session-end --stdin

    # Expected: JSON with success=true, IC value
    ```

    ## Step 3: Start New Session Linking to Previous
    ```bash
    echo '{"session_id":"new-session-456","event_type":"session_start","payload":{"session_start":{"cwd":"/tmp","previous_session_id":"prev-session-123"}}}' | \
      CONTEXT_GRAPH_DB_PATH="$TEST_DB" \
      ./target/debug/context-graph-cli hooks session-start --stdin

    # Expected output:
    # {
    #   "success": true,
    #   "drift_metrics": {
    #     "ic_delta": <some float>,
    #     "purpose_drift": <0.0 to 2.0>,
    #     "time_since_snapshot_ms": <positive int>,
    #     "kuramoto_phase_drift": <0.0 to π>
    #   },
    #   "consciousness_state": {...},
    #   "ic_classification": {...}
    # }
    ```

    ## Step 4: Verify RocksDB Contains Both Snapshots
    ```bash
    ls -la "$TEST_DB/"
    # Expected: RocksDB files (*.sst, MANIFEST*, OPTIONS*, LOG*, etc.)
    ```

    ## Step 5: Cleanup
    ```bash
    rm -rf "$TEST_DB"
    ```

    ## Expected Synthetic Data Values
    | Field | Expected Range | Notes |
    |-------|---------------|-------|
    | ic_delta | [-1.0, 1.0] | Fresh sessions start at IC=1.0 |
    | purpose_drift | [0.0, 2.0] | Default vectors = 0.0 drift |
    | time_since_snapshot_ms | >0 | At least a few ms elapsed |
    | kuramoto_phase_drift | [0.0, π] | Default phases = 0.0 drift |
  </manual_verification>
</verification>

<pseudo_code>
// ═══════════════════════════════════════════════════════════════════════════
// File: crates/context-graph-cli/src/commands/hooks/types.rs
// ADD this struct after existing type definitions
// ═══════════════════════════════════════════════════════════════════════════

/// Drift metrics computed when linking to a previous session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DriftMetrics {
    pub ic_delta: f32,
    pub purpose_drift: f32,
    pub time_since_snapshot_ms: i64,
    pub kuramoto_phase_drift: f64,
}

impl DriftMetrics {
    pub fn is_crisis_drift(&amp;self) -> bool {
        self.ic_delta &lt; -0.3  // 30% IC drop
    }

    pub fn is_warning_drift(&amp;self) -> bool {
        self.ic_delta &lt; -0.1  // 10% IC drop
    }
}

// ADD to HookOutput struct:
pub drift_metrics: Option&lt;DriftMetrics&gt;,

// ADD method:
impl HookOutput {
    pub fn with_drift_metrics(mut self, drift: DriftMetrics) -> Self {
        self.drift_metrics = Some(drift);
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// File: crates/context-graph-cli/src/commands/hooks/session_start.rs
// MODIFY load_or_create_snapshot to compute and return drift metrics
// ═══════════════════════════════════════════════════════════════════════════

use std::f64::consts::PI;
use super::types::DriftMetrics;

/// Cosine distance = 1 - cos_similarity
/// Range: [0.0, 2.0] where 0=identical, 2=opposite
fn cosine_distance(a: &amp;[f32; KURAMOTO_N], b: &amp;[f32; KURAMOTO_N]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::&lt;f32&gt;().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::&lt;f32&gt;().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;  // Zero vectors are considered identical
    }

    let cos_sim = dot / (norm_a * norm_b);
    1.0 - cos_sim.clamp(-1.0, 1.0)  // Distance in [0.0, 2.0]
}

/// Mean absolute phase difference
/// Range: [0.0, π]
fn phase_drift(a: &amp;[f64; KURAMOTO_N], b: &amp;[f64; KURAMOTO_N]) -> f64 {
    let sum: f64 = a.iter().zip(b.iter())
        .map(|(pa, pb)| {
            let diff = (pa - pb).abs();
            // Wrap to [0, π] (phases are circular)
            if diff > PI { 2.0 * PI - diff } else { diff }
        })
        .sum();

    sum / KURAMOTO_N as f64
}

fn compute_drift_metrics(
    current: &amp;SessionIdentitySnapshot,
    previous: &amp;SessionIdentitySnapshot,
) -> DriftMetrics {
    DriftMetrics {
        ic_delta: current.last_ic - previous.last_ic,
        purpose_drift: cosine_distance(&amp;current.purpose_vector, &amp;previous.purpose_vector),
        time_since_snapshot_ms: current.timestamp_ms - previous.timestamp_ms,
        kuramoto_phase_drift: phase_drift(&amp;current.kuramoto_phases, &amp;previous.kuramoto_phases),
    }
}

// MODIFY load_or_create_snapshot signature:
fn load_or_create_snapshot(
    memex: &amp;Arc&lt;RocksDbMemex&gt;,
    session_id: &amp;str,
    previous_session_id: Option&lt;&amp;str&gt;,
) -> HookResult&lt;(SessionIdentitySnapshot, Option&lt;DriftMetrics&gt;)&gt; {
    // ... existing load logic ...

    let mut drift_metrics = None;

    // Link to previous session if provided
    if let Some(prev_id) = previous_session_id {
        match memex.load_snapshot(prev_id) {
            Ok(Some(prev_snapshot)) => {
                // ... existing linking logic ...

                // NEW: Compute drift metrics
                let drift = compute_drift_metrics(&amp;snapshot, &amp;prev_snapshot);

                // Log drift status
                if drift.is_crisis_drift() {
                    error!(
                        ic_delta = drift.ic_delta,
                        purpose_drift = drift.purpose_drift,
                        "SESSION_START: CRISIS drift detected"
                    );
                } else if drift.is_warning_drift() {
                    warn!(
                        ic_delta = drift.ic_delta,
                        purpose_drift = drift.purpose_drift,
                        "SESSION_START: WARNING drift level"
                    );
                } else {
                    info!(
                        ic_delta = drift.ic_delta,
                        purpose_drift = drift.purpose_drift,
                        time_since_ms = drift.time_since_snapshot_ms,
                        phase_drift = drift.kuramoto_phase_drift,
                        "SESSION_START: drift metrics computed"
                    );
                }

                drift_metrics = Some(drift);
            }
            // ... existing error handling ...
        }
    }

    Ok((snapshot, drift_metrics))
}

// MODIFY execute() to use drift_metrics:
pub async fn execute(args: SessionStartArgs) -> HookResult&lt;HookOutput&gt; {
    // ... existing code ...

    let (snapshot, drift_metrics) = load_or_create_snapshot(...)?;

    // ... existing output building ...

    let mut output = HookOutput::success(execution_time_ms)
        .with_consciousness_state(consciousness_state)
        .with_ic_classification(ic_classification);

    // Add drift metrics if present
    if let Some(drift) = drift_metrics {
        output = output.with_drift_metrics(drift);
    }

    Ok(output)
}
</pseudo_code>

<files_to_create>
  <!-- NONE - Only modifying existing files -->
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/commands/hooks/types.rs">
    Add DriftMetrics struct and HookOutput.drift_metrics field
  </file>
  <file path="crates/context-graph-cli/src/commands/hooks/session_start.rs">
    Add compute_drift_metrics(), cosine_distance(), phase_drift() functions.
    Modify load_or_create_snapshot() to compute and return drift.
    Modify execute() to include drift_metrics in output.
  </file>
</files_to_modify>

<test_cases>
  <!-- NO MOCK DATA - Use REAL RocksDB -->

  <test_case id="TC-HOOKS-013-01" name="Drift metrics computed when linking sessions">
    <setup>
      - Create TempDir for RocksDB
      - Create previous session snapshot via session-start
      - End previous session via session-end to persist
    </setup>
    <execute>
      - Call execute(SessionStartArgs) with previous_session_id
    </execute>
    <verify>
      <source_of_truth>HookOutput.drift_metrics</source_of_truth>
      - drift_metrics is Some(...)
      - ic_delta is a valid f32
      - purpose_drift in [0.0, 2.0]
      - time_since_snapshot_ms > 0
      - kuramoto_phase_drift in [0.0, π]
    </verify>
    <synthetic_data>
      Previous snapshot: default values (IC=1.0, phases=[0.0; 13], purpose=[0.0; 13])
      Expected: ic_delta≈0.0, purpose_drift=0.0, phase_drift=0.0
    </synthetic_data>
  </test_case>

  <test_case id="TC-HOOKS-013-02" name="No drift metrics for new session">
    <setup>
      - Create TempDir for RocksDB
    </setup>
    <execute>
      - Call execute(SessionStartArgs) WITHOUT previous_session_id
    </execute>
    <verify>
      <source_of_truth>HookOutput.drift_metrics</source_of_truth>
      - drift_metrics is None
      - success is true
    </verify>
  </test_case>

  <test_case id="TC-HOOKS-013-03" name="Drift metrics when previous session not found">
    <setup>
      - Create TempDir for RocksDB (empty)
    </setup>
    <execute>
      - Call execute(SessionStartArgs) with previous_session_id="nonexistent"
    </execute>
    <verify>
      <source_of_truth>HookOutput.drift_metrics, log output</source_of_truth>
      - drift_metrics is None
      - WARN log contains "previous session not found"
      - success is true (graceful handling)
    </verify>
  </test_case>

  <test_case id="TC-HOOKS-013-04" name="Cosine distance edge cases">
    <setup>
      Create snapshots with specific purpose vectors
    </setup>
    <execute>
      Call cosine_distance() with various inputs
    </execute>
    <verify>
      | Input A | Input B | Expected Distance |
      |---------|---------|-------------------|
      | [1,0,0,...] | [1,0,0,...] | 0.0 |
      | [1,0,0,...] | [-1,0,0,...] | 2.0 |
      | [0,0,0,...] | [0,0,0,...] | 0.0 |
      | [1,1,1,...] | [0.5,0.5,...] | ~0.0 |
    </verify>
  </test_case>

  <test_case id="TC-HOOKS-013-05" name="Phase drift edge cases">
    <setup>
      Create snapshots with specific Kuramoto phases
    </setup>
    <execute>
      Call phase_drift() with various inputs
    </execute>
    <verify>
      | Input A | Input B | Expected Drift |
      |---------|---------|----------------|
      | [0.0; 13] | [0.0; 13] | 0.0 |
      | [0.0; 13] | [π; 13] | π |
      | [0.0; 13] | [2π; 13] | 0.0 (wrapped) |
      | [π/2; 13] | [0.0; 13] | π/2 |
    </verify>
  </test_case>

  <test_case id="TC-HOOKS-013-06" name="Crisis drift triggers error log">
    <setup>
      - Create previous snapshot with IC=1.0
      - Create current scenario where IC drops to 0.6 (>30% drop)
    </setup>
    <execute>
      - Call compute_drift_metrics() and is_crisis_drift()
    </execute>
    <verify>
      - is_crisis_drift() returns true for ic_delta < -0.3
      - ERROR level log emitted
    </verify>
  </test_case>

  <test_case id="TC-HOOKS-013-07" name="Warning drift triggers warn log">
    <setup>
      - Create previous snapshot with IC=1.0
      - Create current scenario where IC drops to 0.85 (15% drop)
    </setup>
    <execute>
      - Call compute_drift_metrics() and is_warning_drift()
    </execute>
    <verify>
      - is_warning_drift() returns true for ic_delta < -0.1
      - is_crisis_drift() returns false (not below 30%)
      - WARN level log emitted
    </verify>
  </test_case>
</test_cases>

<test_commands>
  <!-- Commands to run for verification -->
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli -- drift --nocapture</command>
  <command>cargo test --package context-graph-cli -- tc_hooks_013 --nocapture</command>
  <command>cargo test --package context-graph-cli -- session_start --nocapture</command>
</test_commands>

<constitution_compliance>
  <!-- MANDATORY - These anti-patterns MUST be avoided -->
  <ap_ref id="AP-25">Kuramoto MUST have exactly N=13 oscillators</ap_ref>
  <ap_ref id="AP-26">Exit codes: 0=success, 1=error, 2=corruption</ap_ref>
  <ap_ref id="AP-50">NO internal hooks - use native Claude Code hooks</ap_ref>
  <ap_ref id="AP-51">NO Universal LLM Adapter</ap_ref>
  <ap_ref id="AP-53">Hook logic in CLI commands, not embedded</ap_ref>

  <!-- MANDATORY - These architecture rules MUST be followed -->
  <arch_ref id="ARCH-07">Native Claude Code hooks via .claude/settings.json</arch_ref>

  <!-- MANDATORY - Identity thresholds from constitution -->
  <identity_ref id="IDENTITY-002">
    IC thresholds: Healthy > 0.9, Warning < 0.7, Critical < 0.5
    Drift thresholds: Warning < -0.1 (10% drop), Crisis < -0.3 (30% drop)
  </identity_ref>
</constitution_compliance>

<related_tasks>
  <completed>
    <task id="TASK-SESSION-05">SessionIdentitySnapshot struct with 14 fields</task>
    <task id="TASK-SESSION-06">StandaloneSessionIdentityManager RocksDB persistence</task>
    <task id="TASK-HOOKS-006">session-start hook handler with load_or_create_snapshot</task>
    <task id="TASK-HOOKS-012">session-end hook with persistence to RocksDB</task>
  </completed>
  <blocked_by_this>
    <task id="TASK-HOOKS-016">Integration tests for full hook lifecycle (needs drift)</task>
  </blocked_by_this>
</related_tasks>
</task_spec>
```

## Implementation Checklist

### Step 1: Add DriftMetrics struct to types.rs
1. Open `crates/context-graph-cli/src/commands/hooks/types.rs`
2. Add `DriftMetrics` struct with 4 fields
3. Add `is_crisis_drift()` and `is_warning_drift()` methods
4. Add `drift_metrics: Option<DriftMetrics>` to `HookOutput`
5. Add `with_drift_metrics()` builder method

### Step 2: Add drift computation to session_start.rs
1. Open `crates/context-graph-cli/src/commands/hooks/session_start.rs`
2. Add `use std::f64::consts::PI;`
3. Add `use super::types::DriftMetrics;`
4. Add `cosine_distance()` function
5. Add `phase_drift()` function
6. Add `compute_drift_metrics()` function
7. Modify `load_or_create_snapshot()` return type to include `Option<DriftMetrics>`
8. Compute drift inside the `Ok(Some(prev_snapshot))` arm
9. Log drift at appropriate level (INFO/WARN/ERROR)
10. Modify `execute()` to use new return type and add drift to output

### Step 3: Run Tests
```bash
cargo build --package context-graph-cli
cargo test --package context-graph-cli -- drift --nocapture
cargo test --package context-graph-cli -- session_start --nocapture
```

### Step 4: Manual Verification
Follow the manual_verification section above with synthetic data.

## Key Differences from Original Task

| Original (WRONG) | Corrected |
|-----------------|-----------|
| Create `crates/context-graph-cli/src/identity/restorer.rs` | Modify existing `session_start.rs` |
| Create `SnapshotRestorer` service | Add `compute_drift_metrics()` function |
| Create new CLI command | Use existing hooks CLI |
| `SnapshotStore` trait | Use existing `RocksDbMemex.load_snapshot()` |
| `GwtClient` integration | Not needed - use existing `compute_ic()` |

The original task was based on an outdated understanding of the codebase. Session restoration already exists via `load_or_create_snapshot()`. This task adds the **missing** drift metrics computation to enhance the existing functionality.
