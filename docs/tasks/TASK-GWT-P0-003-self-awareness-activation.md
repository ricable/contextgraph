# TASK-GWT-P0-003: Self-Awareness Loop Activation

<task_spec id="TASK-GWT-P0-003" version="2.0">
<metadata>
  <title>Activate SelfAwarenessLoop in Production Code Paths</title>
  <status>COMPLETED</status>
  <completed_at>2026-01-11T01:45:00Z</completed_at>
  <last_verified>2026-01-11</last_verified>
  <layer>logic</layer>
  <sequence>3</sequence>
  <implements>
    <item>Constitution v4.0.0 Section gwt.self_ego_node (lines 371-392)</item>
    <item>Self-awareness loop: Retrieve->A(action,PV)->if&lt;0.55 self_reflect->update fingerprint->store evolution</item>
    <item>Identity continuity: IC = cos(PV_t, PV_{t-1}) x r(t); healthy>0.9, warning<0.7, dream<0.5</item>
    <item>SHERLOCK-03 finding: SelfAwarenessLoop::cycle() defined but NEVER CALLED in production</item>
  </implements>
  <depends_on>
    <task_ref status="COMPLETED">TASK-GWT-P0-001</task_ref>
    <!-- VERIFIED 2026-01-11: KuramotoNetwork IS integrated into GwtSystem -->
    <!-- See: crates/context-graph-core/src/gwt/mod.rs lines 88-92, 108, 115-151 -->
    <task_ref status="COMPLETED">TASK-GWT-P0-002</task_ref>
    <!-- VERIFIED 2026-01-11: KuramotoStepper IS implemented -->
    <!-- See: crates/context-graph-mcp/src/handlers/kuramoto_stepper.rs -->
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
</metadata>

---

## CRITICAL CONTEXT FOR AI AGENT

### What This Task Is About

The **SelfAwarenessLoop** is the system's mechanism for monitoring alignment between actions and purpose. It implements the R(t) component of the consciousness equation C(t) = I(t) × R(t) × D(t).

**The Problem:** The `SelfAwarenessLoop::cycle()` method exists with correct algorithms (alignment threshold 0.55, identity continuity formula IC = cos(PV_t, PV_{t-1}) × r(t), status thresholds), but **NO PRODUCTION CODE CALLS cycle()**. The ego node's `purpose_vector` remains `[0.0; 13]` forever because it is never updated from `TeleologicalFingerprint` values.

**The Solution:**
1. Add `process_action_awareness()` to `GwtSystem` that calls `cycle()` on every action
2. Add `update_from_fingerprint()` to `SelfEgoNode` to copy `fingerprint.purpose_vector.alignments` to `purpose_vector`
3. Add `trigger_identity_dream()` to invoke dream consolidation when IC < 0.5 (Critical)

### SHERLOCK-03 Gaps Being Fixed

| Gap | Description | Resolution |
|-----|-------------|------------|
| GAP 1 | `SelfAwarenessLoop::cycle()` NEVER INVOKED | Wire into `process_action_awareness()` |
| GAP 2 | Ego Node NEVER WRITTEN TO | Add `update_from_fingerprint()` |
| GAP 5 | Dream Trigger DISCONNECTED | Add `trigger_identity_dream()` on Critical |

---

## EXACT FILE LOCATIONS (VERIFIED 2026-01-11)

### Files That EXIST (Read These First)

| File | Purpose | Key Lines |
|------|---------|-----------|
| `crates/context-graph-core/src/gwt/ego_node.rs` | SelfEgoNode, SelfAwarenessLoop, IdentityContinuity | Lines 19-32: SelfEgoNode struct, Lines 110-116: SelfAwarenessLoop struct, Lines 191-260: cycle() method |
| `crates/context-graph-core/src/gwt/mod.rs` | GwtSystem orchestrator | Lines 69-93: GwtSystem struct with kuramoto field, Lines 77: `self_ego_node: Arc<RwLock<SelfEgoNode>>` |
| `crates/context-graph-core/src/types/fingerprint/teleological/types.rs` | TeleologicalFingerprint | Lines 24-50: struct with `purpose_vector: PurposeVector` |
| `crates/context-graph-core/src/types/fingerprint/purpose.rs` | PurposeVector | Lines 115-130: struct with `alignments: [f32; 13]` |
| `crates/context-graph-mcp/src/handlers/teleological.rs` | MCP handlers for fingerprints | Action entry point where fingerprints are processed |
| `docs/sherlock-03-self-ego-node.md` | Investigation documentation | Full analysis of the gaps |
| `docs2/constitution.yaml` | Specification | Lines 365-369: self_ego_node spec |

### Files to MODIFY

| File | Modification |
|------|-------------|
| `crates/context-graph-core/src/gwt/mod.rs` | Add `process_action_awareness()`, `trigger_identity_dream()`, add `self_awareness_loop: SelfAwarenessLoop` field |
| `crates/context-graph-core/src/gwt/ego_node.rs` | Add `update_from_fingerprint(&mut self, fingerprint: &TeleologicalFingerprint)` |

### Files to CREATE

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/gwt/tests/action_awareness_tests.rs` | Unit tests for process_action_awareness, update_from_fingerprint, dream trigger |

---

## ARCHITECTURE CONTEXT

### Current State of SelfEgoNode

From `crates/context-graph-core/src/gwt/ego_node.rs`:

```rust
// Lines 19-32
pub struct SelfEgoNode {
    pub id: Uuid,                                       // Fixed: Uuid::nil()
    pub fingerprint: Option<TeleologicalFingerprint>,   // ALWAYS None (never updated)
    pub purpose_vector: [f32; 13],                      // ALWAYS [0.0; 13] (never updated)
    pub coherence_with_actions: f32,                    // ALWAYS 0.0 (never updated)
    pub identity_trajectory: Vec<PurposeSnapshot>,      // ALWAYS empty (never updated)
    pub last_updated: DateTime<Utc>,
}
```

**PROBLEM:** All fields except `id` and `last_updated` remain at their initial values forever.

### Current State of SelfAwarenessLoop

From `crates/context-graph-core/src/gwt/ego_node.rs`:

```rust
// Lines 110-116
pub struct SelfAwarenessLoop {
    continuity: IdentityContinuity,
    alignment_threshold: f32,  // 0.55 per constitution
}

// Lines 208-246 - The cycle() method EXISTS and is CORRECT
pub async fn cycle(
    &mut self,
    ego_node: &mut SelfEgoNode,
    action_embedding: &[f32; 13],
    kuramoto_r: f32,
) -> CoreResult<SelfReflectionResult>
```

**PROBLEM:** The `cycle()` method is implemented correctly but **NEVER CALLED** in production.

### Current State of GwtSystem

From `crates/context-graph-core/src/gwt/mod.rs`:

```rust
// Lines 69-93
pub struct GwtSystem {
    pub consciousness_calc: Arc<ConsciousnessCalculator>,
    pub workspace: Arc<RwLock<GlobalWorkspace>>,
    pub self_ego_node: Arc<RwLock<SelfEgoNode>>,           // EXISTS but never written to
    pub state_machine: Arc<RwLock<StateMachineManager>>,
    pub meta_cognitive: Arc<RwLock<MetaCognitiveLoop>>,
    pub event_broadcaster: Arc<WorkspaceEventBroadcaster>,
    pub kuramoto: Arc<RwLock<KuramotoNetwork>>,            // COMPLETED in P0-001
    // MISSING: self_awareness_loop: SelfAwarenessLoop     // NEEDS TO BE ADDED
}
```

**PROBLEM:** No method exists to call `self_awareness_loop.cycle()`.

### PurposeVector Structure

From `crates/context-graph-core/src/types/fingerprint/purpose.rs`:

```rust
// Lines 115-130
pub struct PurposeVector {
    /// Alignment values for each of 13 embedders. Range: [-1.0, 1.0]
    pub alignments: [f32; 13],  // THIS is the source data
    pub dominant_embedder: u8,
    pub coherence: f32,
    pub stability: f32,
}
```

---

## SCOPE

### In Scope (MUST IMPLEMENT)
- Add `self_awareness_loop: SelfAwarenessLoop` field to `GwtSystem`
- Add `process_action_awareness()` to `GwtSystem` that calls `cycle()`
- Add `update_from_fingerprint()` to `SelfEgoNode`
- Add `trigger_identity_dream()` to `GwtSystem` (with graceful degradation if no dream controller)
- Unit tests for all new methods using REAL data (NO MOCKS)
- Integration test verifying loop executes on action processing

### Out of Scope
- MCP tools for ego state updates (separate task)
- Persistence layer for SelfEgoNode (TASK-GWT-P1-001)
- Full dream controller implementation (assume graceful degradation)
- Kuramoto network integration (COMPLETED in TASK-GWT-P0-001)

---

## DEFINITION OF DONE

### Required Method Signatures

```rust
// File: crates/context-graph-core/src/gwt/ego_node.rs

impl SelfEgoNode {
    /// Update purpose_vector from a TeleologicalFingerprint's purpose alignments.
    ///
    /// Copies fingerprint.purpose_vector.alignments to self.purpose_vector,
    /// updates coherence_with_actions, and sets fingerprint reference.
    ///
    /// # Arguments
    /// * `fingerprint` - The source fingerprint containing purpose_vector.alignments
    ///
    /// # Returns
    /// * `CoreResult<()>` - Ok on success, error on invalid fingerprint
    pub fn update_from_fingerprint(&mut self, fingerprint: &TeleologicalFingerprint) -> CoreResult<()>;
}
```

```rust
// File: crates/context-graph-core/src/gwt/mod.rs

pub struct GwtSystem {
    // ... existing fields ...

    /// Self-awareness loop for identity continuity monitoring
    pub self_awareness_loop: Arc<RwLock<SelfAwarenessLoop>>,
}

impl GwtSystem {
    /// Process an action through the self-awareness loop.
    ///
    /// This method:
    /// 1. Updates self_ego_node.purpose_vector from fingerprint
    /// 2. Computes action_embedding from fingerprint.purpose_vector.alignments
    /// 3. Gets kuramoto_r from internal Kuramoto network
    /// 4. Calls self_awareness_loop.cycle()
    /// 5. Triggers dream if IdentityStatus::Critical
    ///
    /// # Arguments
    /// * `fingerprint` - The action's TeleologicalFingerprint
    ///
    /// # Returns
    /// * `SelfReflectionResult` containing alignment and identity status
    pub async fn process_action_awareness(
        &self,
        fingerprint: &TeleologicalFingerprint,
    ) -> CoreResult<SelfReflectionResult>;

    /// Trigger dream consolidation when identity is Critical (IC < 0.5).
    ///
    /// If dream controller is not available, logs warning and records
    /// purpose snapshot (graceful degradation).
    ///
    /// # Arguments
    /// * `reason` - Description of why dream is triggered
    async fn trigger_identity_dream(&self, reason: &str) -> CoreResult<()>;
}
```

### Required Constraints (FAIL FAST - NO WORKAROUNDS)

| Constraint | Rationale |
|------------|-----------|
| MUST use `tokio::sync::RwLock` for SelfAwarenessLoop | Matches GwtSystem pattern (async methods) |
| MUST call `cycle()` on every action that produces a fingerprint | Constitution requirement |
| MUST update `purpose_vector` from `fingerprint.purpose_vector.alignments` BEFORE `cycle()` | Required for correct alignment calculation |
| MUST trigger dream when `IdentityStatus::Critical` (IC < 0.5) | Constitution line 391: "dream<0.5" |
| MUST preserve alignment_threshold = 0.55 | Constitution line 384: "if<0.55 self_reflect" |
| MUST record purpose snapshot after each cycle | Constitution: "store evolution" |
| MUST NOT use mock data in tests | Verify real behavior |
| MUST NOT create fallbacks or compatibility shims | FAIL FAST with clear error messages |

---

## PSEUDOCODE

### SelfEgoNode::update_from_fingerprint

```rust
pub fn update_from_fingerprint(&mut self, fingerprint: &TeleologicalFingerprint) -> CoreResult<()> {
    // 1. Copy purpose_vector.alignments to self.purpose_vector
    self.purpose_vector = fingerprint.purpose_vector.alignments;

    // 2. Update coherence from fingerprint
    self.coherence_with_actions = fingerprint.purpose_vector.coherence;

    // 3. Store fingerprint reference (clone since we own the data)
    self.fingerprint = Some(fingerprint.clone());

    // 4. Update timestamp
    self.last_updated = Utc::now();

    // 5. Log for debugging
    tracing::debug!(
        "SelfEgoNode updated from fingerprint: purpose_vector[0]={:.4}, coherence={:.4}",
        self.purpose_vector[0],
        self.coherence_with_actions
    );

    Ok(())
}
```

### GwtSystem::process_action_awareness

```rust
pub async fn process_action_awareness(
    &self,
    fingerprint: &TeleologicalFingerprint,
) -> CoreResult<SelfReflectionResult> {
    // 1. Get kuramoto_r from internal network
    let kuramoto_r = self.get_kuramoto_r().await;

    // 2. Extract action_embedding from fingerprint
    let action_embedding = fingerprint.purpose_vector.alignments;

    // 3. Acquire write lock on self_ego_node
    let mut ego_node = self.self_ego_node.write().await;

    // 4. Update purpose_vector from fingerprint
    ego_node.update_from_fingerprint(fingerprint)?;

    // 5. Acquire write lock on self_awareness_loop
    let mut loop_mgr = self.self_awareness_loop.write().await;

    // 6. Execute self-awareness cycle
    let result = loop_mgr.cycle(&mut ego_node, &action_embedding, kuramoto_r).await?;

    // 7. Log the result
    tracing::info!(
        "Self-awareness cycle: alignment={:.4}, identity_status={:?}, identity_coherence={:.4}",
        result.alignment,
        result.identity_status,
        result.identity_coherence
    );

    // 8. Check for Critical identity status - MUST trigger dream
    if result.identity_status == IdentityStatus::Critical {
        drop(ego_node);  // Release lock before async call
        drop(loop_mgr);
        self.trigger_identity_dream("Identity coherence critical").await?;
    }

    // 9. Return result
    Ok(result)
}
```

### GwtSystem::trigger_identity_dream

```rust
async fn trigger_identity_dream(&self, reason: &str) -> CoreResult<()> {
    // 1. Log critical warning
    tracing::warn!("IDENTITY CRITICAL: Triggering dream consolidation. Reason: {}", reason);

    // 2. Record purpose snapshot with dream trigger context
    {
        let mut ego_node = self.self_ego_node.write().await;
        ego_node.record_purpose_snapshot(format!("Dream triggered: {}", reason))?;
    }

    // 3. Broadcast workspace event for dream trigger
    // (DreamController will be wired in TASK-GWT-P1-002)
    self.event_broadcaster.broadcast(WorkspaceEvent::IdentityCritical {
        identity_coherence: {
            let loop_mgr = self.self_awareness_loop.read().await;
            // Access continuity if needed
            0.0  // Will be updated when loop_mgr exposes this
        },
        reason: reason.to_string(),
    }).await;

    // 4. TODO(TASK-GWT-P1-002): Wire to actual DreamController
    // For now, graceful degradation with logging
    tracing::info!("Dream trigger recorded. DreamController integration pending.");

    Ok(())
}
```

---

## FULL STATE VERIFICATION REQUIREMENTS

### 1. Source of Truth Definition

The **Source of Truth** for this task is:

| Source | Location | What It Proves |
|--------|----------|----------------|
| `SelfEgoNode.purpose_vector` | `self_ego_node.read().await.purpose_vector` | Vector updates from [0.0; 13] to fingerprint values |
| `SelfEgoNode.identity_trajectory.len()` | `self_ego_node.read().await.identity_trajectory.len()` | Snapshots are recorded on each cycle |
| `SelfEgoNode.coherence_with_actions` | `self_ego_node.read().await.coherence_with_actions` | Coherence is set from fingerprint |
| `SelfReflectionResult.identity_status` | Return value from `process_action_awareness()` | Correct status computed (Healthy/Warning/Degraded/Critical) |

### 2. Execute & Inspect Protocol

After implementing, you MUST run this test:

```rust
#[tokio::test]
async fn test_full_state_verification_purpose_vector_update() {
    // === SETUP ===
    let gwt = GwtSystem::new().await.expect("GwtSystem creation must succeed");

    // Create a fingerprint with known values
    let fingerprint = create_test_fingerprint_with_alignments([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4]);

    // === STATE BEFORE (Read Source of Truth) ===
    let before = {
        let ego = gwt.self_ego_node.read().await;
        (ego.purpose_vector, ego.identity_trajectory.len(), ego.coherence_with_actions)
    };
    println!("STATE BEFORE:");
    println!("  purpose_vector[0] = {:.4}", before.0[0]);
    println!("  identity_trajectory.len() = {}", before.1);
    println!("  coherence_with_actions = {:.4}", before.2);

    assert_eq!(before.0, [0.0; 13], "Initial purpose_vector must be zeros");
    assert_eq!(before.1, 0, "Initial trajectory must be empty");

    // === EXECUTE ===
    let result = gwt.process_action_awareness(&fingerprint).await
        .expect("process_action_awareness must succeed");

    println!("RESULT:");
    println!("  alignment = {:.4}", result.alignment);
    println!("  identity_status = {:?}", result.identity_status);
    println!("  identity_coherence = {:.4}", result.identity_coherence);

    // === VERIFY VIA SEPARATE READ (Source of Truth) ===
    let after = {
        let ego = gwt.self_ego_node.read().await;
        (ego.purpose_vector, ego.identity_trajectory.len(), ego.coherence_with_actions)
    };
    println!("STATE AFTER:");
    println!("  purpose_vector[0] = {:.4}", after.0[0]);
    println!("  identity_trajectory.len() = {}", after.1);
    println!("  coherence_with_actions = {:.4}", after.2);

    // === ASSERTIONS ===
    assert_ne!(after.0, [0.0; 13], "purpose_vector MUST be updated from fingerprint");
    assert_eq!(after.0[0], 0.8, "purpose_vector[0] must match fingerprint alignments[0]");
    assert!(after.1 >= 1, "identity_trajectory MUST have at least one snapshot");
    assert!(after.2 > 0.0, "coherence_with_actions MUST be non-zero");

    // === EVIDENCE OF SUCCESS ===
    println!("EVIDENCE: purpose_vector updated from {:?} to {:?}", before.0, after.0);
    println!("EVIDENCE: {} snapshots recorded", after.1);
}
```

### 3. Boundary & Edge Case Audit

You MUST implement and run these 3 edge case tests:

#### Edge Case 1: Critical Identity Triggers Dream

```rust
#[tokio::test]
async fn test_critical_identity_triggers_dream() {
    println!("=== EDGE CASE 1: Critical Identity Status ===");

    let gwt = GwtSystem::new().await.unwrap();

    // Set up initial purpose vector
    {
        let mut ego = gwt.self_ego_node.write().await;
        ego.purpose_vector = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2];
        ego.record_purpose_snapshot("Initial state").unwrap();
    }

    // Create fingerprint with VERY DIFFERENT alignments (will cause low cosine similarity)
    let fingerprint = create_test_fingerprint_with_alignments(
        [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2]
    );

    // STATE BEFORE
    let before_trajectory_len = gwt.self_ego_node.read().await.identity_trajectory.len();
    println!("BEFORE: trajectory.len() = {}", before_trajectory_len);

    // EXECUTE
    let result = gwt.process_action_awareness(&fingerprint).await.unwrap();

    println!("RESULT: identity_status = {:?}, identity_coherence = {:.4}",
             result.identity_status, result.identity_coherence);

    // VERIFY
    // With orthogonal/opposite vectors, cosine should be negative
    // IC = cos * r, with low r from fresh Kuramoto, IC should be < 0.5 = Critical

    let after_trajectory_len = gwt.self_ego_node.read().await.identity_trajectory.len();
    println!("AFTER: trajectory.len() = {}", after_trajectory_len);

    // Dream trigger should have recorded an extra snapshot
    assert!(after_trajectory_len > before_trajectory_len,
            "Dream trigger must record snapshot");

    println!("EVIDENCE: Critical status triggered dream consolidation");
}
```

#### Edge Case 2: Low Alignment Triggers Reflection

```rust
#[tokio::test]
async fn test_low_alignment_triggers_reflection() {
    println!("=== EDGE CASE 2: Low Alignment Triggers Reflection ===");

    let gwt = GwtSystem::new().await.unwrap();

    // Set purpose vector to point in one direction
    {
        let mut ego = gwt.self_ego_node.write().await;
        ego.purpose_vector = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    }

    // Create fingerprint pointing in orthogonal direction
    let fingerprint = create_test_fingerprint_with_alignments(
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );

    // STATE BEFORE
    println!("BEFORE: purpose_vector = [1.0, 0.0, ...]");
    println!("ACTION: fingerprint.alignments = [0.0, 1.0, ...]");

    // EXECUTE
    let result = gwt.process_action_awareness(&fingerprint).await.unwrap();

    // VERIFY
    println!("RESULT: alignment = {:.4}, needs_reflection = {}",
             result.alignment, result.needs_reflection);

    // Orthogonal vectors have 0 cosine similarity, which is < 0.55 threshold
    assert!(result.alignment < 0.55, "Orthogonal vectors must have low alignment");
    assert!(result.needs_reflection, "Low alignment must trigger reflection flag");

    println!("EVIDENCE: needs_reflection = true for orthogonal action");
}
```

#### Edge Case 3: High Alignment Does NOT Trigger Reflection

```rust
#[tokio::test]
async fn test_high_alignment_no_reflection() {
    println!("=== EDGE CASE 3: High Alignment No Reflection ===");

    let gwt = GwtSystem::new().await.unwrap();

    // Set purpose vector
    {
        let mut ego = gwt.self_ego_node.write().await;
        ego.purpose_vector = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
    }

    // Create fingerprint with SAME alignments (perfect alignment)
    let fingerprint = create_test_fingerprint_with_alignments(
        [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
    );

    // STATE BEFORE
    println!("BEFORE: purpose_vector matches fingerprint.alignments");

    // EXECUTE
    let result = gwt.process_action_awareness(&fingerprint).await.unwrap();

    // VERIFY
    println!("RESULT: alignment = {:.4}, needs_reflection = {}",
             result.alignment, result.needs_reflection);

    // Same vectors have cosine = 1.0, which is > 0.55 threshold
    assert!(result.alignment > 0.55, "Identical vectors must have high alignment");
    assert!(!result.needs_reflection, "High alignment must NOT trigger reflection");

    println!("EVIDENCE: needs_reflection = false for aligned action");
}
```

### 4. Evidence of Success Log

After running tests, provide output showing:

```
TASK-GWT-P0-003 COMPLETION EVIDENCE:

1. purpose_vector Update:
   BEFORE: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
   AFTER:  [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4]

2. identity_trajectory Growth:
   BEFORE: 0 snapshots
   AFTER:  1 snapshot (recorded on cycle)

3. Critical Identity Dream Trigger:
   identity_status = Critical (IC < 0.5)
   Dream snapshot recorded with reason

4. Reflection Trigger:
   alignment = 0.0 (orthogonal vectors)
   needs_reflection = true (< 0.55 threshold)

5. All Tests Passed:
   cargo test -p context-graph-core gwt::tests:: -- --nocapture
```

---

## TEST COMMANDS

```bash
# Build the package
cd /home/cabdru/contextgraph && cargo build --package context-graph-core

# Run ego_node tests
cd /home/cabdru/contextgraph && cargo test --package context-graph-core ego_node -- --nocapture

# Run all GWT tests
cd /home/cabdru/contextgraph && cargo test --package context-graph-core gwt:: -- --nocapture

# Run clippy (MUST pass with no warnings)
cd /home/cabdru/contextgraph && cargo clippy --package context-graph-core -- -D warnings

# Run integration tests
cd /home/cabdru/contextgraph && cargo test --test gwt_integration -- --nocapture
```

---

## VALIDATION CRITERIA CHECKLIST

| Criterion | Verification Method | Pass/Fail |
|-----------|---------------------|-----------|
| `update_from_fingerprint()` copies alignments to purpose_vector | Unit test: verify `purpose_vector[0] == fingerprint.purpose_vector.alignments[0]` | |
| `process_action_awareness()` calls `cycle()` | Unit test: verify `identity_trajectory.len()` increases | |
| Critical identity triggers dream | Edge case test: verify snapshot recorded with "Dream triggered" | |
| Low alignment triggers reflection | Edge case test: verify `needs_reflection == true` for alignment < 0.55 | |
| High alignment does NOT trigger reflection | Edge case test: verify `needs_reflection == false` for alignment > 0.55 | |
| `purpose_vector` changes from `[0.0; 13]` | Full state verification test | |
| `coherence_with_actions` is non-zero | Full state verification test | |
| All existing tests pass | `cargo test -p context-graph-core gwt::` | |
| Clippy passes | `cargo clippy -p context-graph-core -- -D warnings` | |

---

## ANTI-PATTERNS TO AVOID (FAIL FAST)

| Anti-Pattern | Why It's Wrong | What To Do Instead |
|--------------|----------------|---------------------|
| Using mock TeleologicalFingerprint | Tests pass but don't verify real behavior | Use real `TeleologicalFingerprint` with known values |
| Catching and ignoring errors | Hides bugs, causes silent failures | Propagate errors with `?` or `expect()` with message |
| Using `std::sync::RwLock` for async code | Blocks async runtime | Use `tokio::sync::RwLock` |
| Skipping dream trigger on Critical | Violates constitution spec | Always call `trigger_identity_dream()` |
| Using hardcoded alignment threshold | May drift from constitution | Use constant from constitution (0.55) |
| Not recording purpose snapshots | Loses identity trajectory | Always call `record_purpose_snapshot()` |

---

## CONSTITUTION REFERENCE

From `docs2/constitution.yaml` lines 365-392:

```yaml
self_ego_node:
  id: "SELF_EGO_NODE"
  content: "I am the context graph manager. My purpose is to help humans organize and access their knowledge..."
  fields:
    - fingerprint          # TeleologicalFingerprint (current system state)
    - purpose_vector       # [f32; 13] alignment with north star
    - identity_trajectory  # Vec<PurposeSnapshot> (max 1000)
    - coherence_with_actions  # f32 alignment score
  loop: "Retrieve→A(action,PV)→if<0.55 self_reflect→update fingerprint→store evolution"
  identity_continuity: "IC = cos(PV_t, PV_{t-1}) × r(t); healthy>0.9, warning<0.7, dream<0.5"
  thresholds:
    healthy: 0.9      # IC > 0.9 = stable identity
    warning: 0.7      # 0.7 <= IC <= 0.9 = monitor closely
    degraded: 0.5     # 0.5 <= IC < 0.7 = degraded, needs attention
    critical: 0.5     # IC < 0.5 = TRIGGER DREAM CONSOLIDATION
```

---

## HELPER FUNCTION FOR TESTS

```rust
/// Create a TeleologicalFingerprint with specified alignment values for testing.
/// This is NOT a mock - it creates a real TeleologicalFingerprint with real data.
fn create_test_fingerprint_with_alignments(alignments: [f32; 13]) -> TeleologicalFingerprint {
    use crate::types::fingerprint::purpose::PurposeVector;
    use crate::types::fingerprint::semantic::SemanticFingerprint;
    use crate::types::fingerprint::johari::JohariFingerprint;

    TeleologicalFingerprint {
        id: Uuid::new_v4(),
        semantic: SemanticFingerprint::default(),
        purpose_vector: PurposeVector {
            alignments,
            dominant_embedder: 0,
            coherence: 0.75,  // Non-zero for testing
            stability: 0.8,
        },
        johari: JohariFingerprint::default(),
        purpose_evolution: Vec::new(),
        theta_to_north_star: 0.5,
        content_hash: [0u8; 32],
        created_at: Utc::now(),
        last_updated: Utc::now(),
        access_count: 0,
    }
}
```

---

## RECOMMENDED APPROACH

Based on analysis of the codebase and completed TASK-GWT-P0-002 pattern:

### Pros and Cons

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| Add `self_awareness_loop` to `GwtSystem` | Centralized, consistent with existing pattern | Requires modifying GwtSystem struct | **RECOMMENDED** |
| Create standalone loop | Isolated, easier to test | Coordination complexity, not consistent | Not recommended |
| Use existing `SelfAwarenessLoop::new()` per call | Simple, no struct change | Inefficient, loses continuity state | Not recommended |

### Implementation Order

1. **Add `update_from_fingerprint()` to `SelfEgoNode`** (ego_node.rs)
   - Simplest change, no dependencies
   - Enables purpose_vector updates

2. **Add `self_awareness_loop` field to `GwtSystem`** (mod.rs)
   - Modify struct and `new()` constructor
   - Thread-safe with `Arc<RwLock<SelfAwarenessLoop>>`

3. **Add `trigger_identity_dream()` to `GwtSystem`** (mod.rs)
   - Graceful degradation pattern
   - Log + record snapshot

4. **Add `process_action_awareness()` to `GwtSystem`** (mod.rs)
   - Orchestrates all the above
   - Main integration point

5. **Write tests** (action_awareness_tests.rs or inline in mod.rs tests)
   - Full state verification test
   - 3 edge case tests
   - Evidence logging

---

## VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2026-01-11 | Complete rewrite with verified file paths, Full State Verification requirements, edge cases, FAIL FAST constraints, real data tests, implementation order |
| 1.0 | 2026-01-10 | Initial task specification |

</task_spec>

---

## SUMMARY

This task activates the dormant SelfAwarenessLoop by:

1. **Adding `update_from_fingerprint()` to `SelfEgoNode`** - Copies `fingerprint.purpose_vector.alignments` to `self.purpose_vector`
2. **Adding `self_awareness_loop` field to `GwtSystem`** - Persistent loop instance for continuity tracking
3. **Adding `process_action_awareness()` to `GwtSystem`** - Orchestrates the entire self-awareness cycle
4. **Adding `trigger_identity_dream()`** - Invokes dream consolidation when IC < 0.5 (Critical)

## Dependencies

- **TASK-GWT-P0-001** (COMPLETED): Provides `kuramoto: Arc<RwLock<KuramotoNetwork>>` in GwtSystem
- **TASK-GWT-P0-002** (COMPLETED): Provides background stepper for Kuramoto phases

## Downstream Dependencies

- **TASK-GWT-P1-001**: Ego Node Persistence (depends on this task for write operations)
- **TASK-GWT-P1-002**: Workspace Event Wiring (depends on this task for dream trigger integration)
