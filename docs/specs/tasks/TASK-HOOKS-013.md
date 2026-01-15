# TASK-HOOKS-013: Implement Session Identity Snapshot Restoration

```xml
<task_spec id="TASK-HOOKS-013" version="1.0">
<metadata>
  <title>Implement Session Identity Snapshot Restoration</title>
  <status>ready</status>
  <layer>logic</layer>
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
  <estimated_complexity>high</estimated_complexity>
  <estimated_hours>3.0</estimated_hours>
</metadata>

<context>
When session_start fires, the system attempts to restore the previous session's
identity snapshot to maintain continuity. This involves:
1. Loading the latest snapshot from storage
2. Validating snapshot freshness (not too old)
3. Restoring GWT state (Kuramoto phases, workspace)
4. Computing identity drift since last session
</context>

<input_context_files>
  <file purpose="snapshot_store">crates/context-graph-cli/src/identity/snapshot_store.rs</file>
  <file purpose="gwt_state">crates/context-graph-gwt/src/state.rs</file>
  <file purpose="kuramoto">crates/context-graph-gwt/src/kuramoto/mod.rs</file>
  <file purpose="technical_spec">docs/specs/technical/TECH-HOOKS.md#session_restoration</file>
</input_context_files>

<prerequisites>
  <check>SnapshotStore implemented (TASK-HOOKS-012)</check>
  <check>GWT state restoration APIs available</check>
  <check>session_start hook handler exists (TASK-HOOKS-006)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create SnapshotRestorer service
    - Implement snapshot freshness validation (max age configurable)
    - Restore Kuramoto oscillator phases from snapshot
    - Restore Self-Ego node state
    - Compute identity drift metrics (IC delta, purpose drift)
    - Handle restoration failures gracefully (start fresh)
    - CLI command: `identity restore --snapshot-id <id>`
  </in_scope>
  <out_of_scope>
    - Automatic crisis recovery (different feature)
    - Multi-snapshot merging (future feature)
    - Remote snapshot restoration (future feature)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/identity/restorer.rs">
      pub struct SnapshotRestorer {
          store: Arc&lt;dyn SnapshotStore&gt;,
          gwt_client: Arc&lt;GwtClient&gt;,
          max_snapshot_age: Duration,
      }

      impl SnapshotRestorer {
          pub fn new(
              store: Arc&lt;dyn SnapshotStore&gt;,
              gwt_client: Arc&lt;GwtClient&gt;,
              max_snapshot_age: Duration,
          ) -> Self;

          pub async fn restore_latest(&amp;self) -> Result&lt;RestorationResult, RestoreError&gt;;
          pub async fn restore_by_id(&amp;self, id: &amp;SnapshotId) -> Result&lt;RestorationResult, RestoreError&gt;;
          pub async fn compute_drift(&amp;self, snapshot: &amp;SessionIdentitySnapshot) -> DriftMetrics;
      }

      pub struct RestorationResult {
          pub snapshot_id: SnapshotId,
          pub restored_at: DateTime&lt;Utc&gt;,
          pub drift: DriftMetrics,
          pub warnings: Vec&lt;String&gt;,
      }

      pub struct DriftMetrics {
          pub ic_delta: f64,
          pub purpose_drift: f64,
          pub time_since_snapshot: Duration,
          pub kuramoto_phase_drift: f64,
      }
    </signature>
  </signatures>

  <constraints>
    - Snapshots older than max_age (default 7 days) must be rejected
    - Restoration must not corrupt current GWT state on failure
    - Must log all restoration attempts for audit
    - Drift metrics must be computed before restoration
    - Warnings must include: stale snapshot, high drift, missing fields
  </constraints>

  <verification>
    - cargo test --package context-graph-cli restorer
    - Test restoration from fresh snapshot
    - Test rejection of stale snapshot
    - Test graceful failure on corrupt snapshot
    - Test drift computation accuracy
  </verification>
</definition_of_done>

<pseudo_code>
SnapshotRestorer:
  store: SnapshotStore
  gwt_client: GwtClient
  max_snapshot_age: Duration

  restore_latest():
    // Load latest snapshot
    snapshot = store.load_latest()?
    if snapshot.is_none():
      return Err(RestoreError::NoSnapshot)

    return restore_snapshot(snapshot.unwrap())

  restore_by_id(id):
    snapshot = store.load(id)?
    return restore_snapshot(snapshot)

  restore_snapshot(snapshot):
    warnings = []

    // Check freshness
    age = now() - snapshot.captured_at
    if age > max_snapshot_age:
      return Err(RestoreError::SnapshotTooOld(age))

    if age > max_snapshot_age / 2:
      warnings.push("Snapshot is aging, consider refreshing")

    // Compute drift before restoration
    drift = compute_drift(snapshot)

    if drift.ic_delta.abs() > 0.3:
      warnings.push(f"High IC drift detected: {drift.ic_delta}")

    // Restore Kuramoto phases
    gwt_client.set_kuramoto_phases(snapshot.kuramoto_phases)?

    // Restore Self-Ego state
    gwt_client.set_ego_state(EgoState {
      purpose_vector: snapshot.purpose_vector,
      trajectory: snapshot.ego_trajectory,
    })?

    // Restore workspace context
    gwt_client.restore_workspace(snapshot.workspace_context)?

    return Ok(RestorationResult {
      snapshot_id: snapshot.id,
      restored_at: now(),
      drift: drift,
      warnings: warnings,
    })

  compute_drift(snapshot):
    current_state = gwt_client.get_consciousness_state()?
    current_ic = gwt_client.get_identity_continuity()?

    return DriftMetrics {
      ic_delta: current_ic.value - snapshot.identity_continuity,
      purpose_drift: cosine_distance(
        current_state.ego.purpose_vector,
        snapshot.purpose_vector
      ),
      time_since_snapshot: now() - snapshot.captured_at,
      kuramoto_phase_drift: phase_distance(
        current_state.kuramoto.phases,
        snapshot.kuramoto_phases
      ),
    }
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/identity/restorer.rs">
    SnapshotRestorer implementation with drift computation
  </file>
  <file path="crates/context-graph-cli/src/commands/identity/restore.rs">
    CLI restore command implementation
  </file>
  <file path="crates/context-graph-cli/tests/identity/restorer_test.rs">
    Integration tests for restoration (real GWT calls)
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/identity/mod.rs">
    Export restorer module
  </file>
  <file path="crates/context-graph-cli/src/commands/identity/mod.rs">
    Add restore subcommand
  </file>
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli restorer</command>
  <command>./target/debug/context-graph-cli identity restore</command>
</test_commands>
</task_spec>
```
