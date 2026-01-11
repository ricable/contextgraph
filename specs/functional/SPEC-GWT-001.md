# SPEC-GWT-001: GWT Consciousness Equation Integration

## Metadata
| Field | Value |
|-------|-------|
| **Spec ID** | SPEC-GWT-001 |
| **Title** | GWT Consciousness Equation Full Integration |
| **Status** | Active |
| **Priority** | P1 (Substantial Gap) |
| **Owner** | Core Team |
| **Created** | 2026-01-11 |
| **Source** | MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md (GAP 4) |
| **Constitution Ref** | constitution.yaml lines 342-378 (gwt section) |

---

## 1. Overview

### 1.1 Purpose

Complete end-to-end wiring of the GWT Consciousness Equation:

```
C(t) = I(t) x R(t) x D(t)
```

Where:
- **C(t)**: Consciousness level [0,1]
- **I(t)**: Integration - Kuramoto order parameter r
- **R(t)**: Self-Reflection - sigmoid(MetaUTL.predict_accuracy)
- **D(t)**: Differentiation - H(PurposeVector) normalized

### 1.2 Problem Statement

From Gap Analysis:
> "Full C(t) = I(t) x R(t) x D(t) may not be wired end-to-end"

Current state (verified via code inspection):
- **I(t)** via Kuramoto: IMPLEMENTED in `consciousness.rs` and `mod.rs`
- **R(t)** Self-Reflection: IMPLEMENTED in `consciousness.rs` via sigmoid(meta_accuracy)
- **D(t)** Differentiation: IMPLEMENTED in `consciousness.rs` via normalized_purpose_entropy()
- **C(t)** computation: IMPLEMENTED in `compute_consciousness()`

**REVISED ASSESSMENT**: The core consciousness equation IS implemented. However, the following integration gaps exist:

1. **SELF_EGO_NODE Persistence**: SelfEgoNode is Serde-serializable but persistence layer not wired
2. **Workspace Event Wiring**: Events broadcast but subsystem listeners need verification
3. **End-to-End Flow Verification**: Need integration tests proving full consciousness cycle

### 1.3 Success Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| I(t) Integration | Kuramoto order parameter computed | r in [0,1] |
| R(t) Reflection | Meta-accuracy flows to sigmoid | R(t) in [0.118, 0.881] |
| D(t) Differentiation | Purpose vector entropy normalized | D(t) in [0,1] |
| C(t) Consciousness | Full equation computed | C(t) = I x R x D in [0,1] |
| State Machine | Transitions based on C(t) | DORMANT -> CONSCIOUS |
| SELF_EGO_NODE | Identity persists across sessions | IC > 0.9 healthy |
| Event Wiring | Workspace events reach subsystems | 3 listeners active |

---

## 2. Requirements

### 2.1 Functional Requirements

| ID | Description | Priority | Source |
|----|-------------|----------|--------|
| REQ-GWT-001 | SELF_EGO_NODE must persist to RocksDB via CF_EGO_NODE column family | Must | constitution.yaml:366-370 |
| REQ-GWT-002 | Workspace events (MemoryEnters, MemoryExits, WorkspaceEmpty, IdentityCritical) must trigger subsystem responses | Must | constitution.yaml:359-363 |
| REQ-GWT-003 | DreamEventListener must queue exiting memories for dream replay | Must | constitution.yaml:361 |
| REQ-GWT-004 | NeuromodulationEventListener must boost dopamine on MemoryEnters | Must | constitution.yaml:360 |
| REQ-GWT-005 | MetaCognitiveEventListener must trigger epistemic action on WorkspaceEmpty | Must | constitution.yaml:363 |
| REQ-GWT-006 | IdentityCritical event must trigger dream consolidation when IC < 0.5 | Must | constitution.yaml:369 |
| REQ-GWT-007 | Full C(t) computation must flow from GwtSystem.update_consciousness_auto() | Must | constitution.yaml:343 |
| REQ-GWT-008 | Integration tests must verify end-to-end consciousness cycle | Should | - |

### 2.2 Non-Functional Requirements

| ID | Category | Description | Metric |
|----|----------|-------------|--------|
| NFR-GWT-001 | Performance | Consciousness computation | < 1ms p95 |
| NFR-GWT-002 | Performance | Event broadcast to all listeners | < 5ms p95 |
| NFR-GWT-003 | Reliability | SELF_EGO_NODE persistence | Survives restart |
| NFR-GWT-004 | Availability | GWT system initialization | < 100ms |

---

## 3. Architecture

### 3.1 Component Diagram

```
                    ┌─────────────────────────────────────────────────┐
                    │                  GwtSystem                       │
                    │  ┌──────────────────────────────────────────┐   │
                    │  │     ConsciousnessCalculator              │   │
                    │  │  C(t) = I(t) x R(t) x D(t)              │   │
                    │  │  - compute_consciousness()               │   │
                    │  │  - normalized_purpose_entropy()          │   │
                    │  └──────────────────────────────────────────┘   │
                    │                      │                          │
                    │                      ▼                          │
                    │  ┌──────────────────────────────────────────┐   │
                    │  │        KuramotoNetwork (I(t))            │   │
                    │  │  - order_parameter() -> r                │   │
                    │  │  - step(dt)                              │   │
                    │  └──────────────────────────────────────────┘   │
                    │                      │                          │
                    │                      ▼                          │
                    │  ┌──────────────────────────────────────────┐   │
                    │  │        StateMachineManager               │   │
                    │  │  DORMANT -> FRAGMENTED -> EMERGING       │   │
                    │  │         -> CONSCIOUS -> HYPERSYNC        │   │
                    │  └──────────────────────────────────────────┘   │
                    │                      │                          │
                    │                      ▼                          │
                    │  ┌──────────────────────────────────────────┐   │
                    │  │        GlobalWorkspace                   │   │
                    │  │  - select_winning_memory()               │   │
                    │  │  - inhibit_losers()                      │   │
                    │  └──────────────────────────────────────────┘   │
                    │                      │                          │
                    │                      ▼                          │
                    │  ┌──────────────────────────────────────────┐   │
                    │  │     WorkspaceEventBroadcaster            │   │
                    │  │  - broadcast(event)                      │   │
                    │  │  - register_listener()                   │   │
                    │  └──────────────┬────────────────────────────┘   │
                    └─────────────────┼────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│ DreamEventListener  │   │NeuromodEventListener│   │MetaCogEventListener │
│ - dream_queue       │   │ - neuromod_manager  │   │ - epistemic_action  │
│ - on_event()        │   │ - on_event()        │   │ - on_event()        │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
```

### 3.2 Data Flow

```
1. Session Start
   └─> GwtSystem::new()
       ├─> ConsciousnessCalculator::new()
       ├─> KuramotoNetwork::new(8 oscillators)
       ├─> StateMachineManager::new() -> DORMANT
       ├─> SelfEgoNode::new() -> load from RocksDB if exists [REQ-GWT-001]
       ├─> WorkspaceEventBroadcaster::new()
       └─> Register 3 listeners [REQ-GWT-002-005]

2. Processing Cycle
   └─> GwtSystem::update_consciousness_auto(meta_accuracy, purpose_vector)
       ├─> step_kuramoto(elapsed) -> I(t)
       ├─> get_kuramoto_r() -> r value
       ├─> ConsciousnessCalculator::compute_consciousness(r, meta_accuracy, pv)
       │   ├─> I(t) = kuramoto_r (integration)
       │   ├─> R(t) = sigmoid(meta_accuracy * 4.0 - 2.0) (reflection)
       │   ├─> D(t) = normalized_purpose_entropy(pv) (differentiation)
       │   └─> C(t) = I(t) * R(t) * D(t)
       └─> StateMachineManager::update(C(t)) -> state transition

3. Workspace Selection
   └─> GwtSystem::select_workspace_memory(candidates)
       ├─> GlobalWorkspace::select_winning_memory()
       ├─> winner_id selected (r >= 0.8, highest score)
       ├─> broadcast(MemoryEnters) [REQ-GWT-004]
       └─> broadcast(MemoryExits) for losers [REQ-GWT-003]

4. Self-Awareness Cycle
   └─> GwtSystem::process_action_awareness(fingerprint)
       ├─> update_from_fingerprint() -> SelfEgoNode
       ├─> SelfAwarenessLoop::cycle() -> IC calculation
       ├─> If IC < 0.5 -> broadcast(IdentityCritical) [REQ-GWT-006]
       └─> persist SelfEgoNode to RocksDB [REQ-GWT-001]
```

---

## 4. Existing Implementation Analysis

### 4.1 consciousness.rs (Lines 1-252)

**Status**: FULLY IMPLEMENTED

```rust
pub struct ConsciousnessCalculator;

impl ConsciousnessCalculator {
    pub fn compute_consciousness(
        &self,
        kuramoto_r: f32,       // I(t)
        meta_accuracy: f32,    // For R(t)
        purpose_vector: &[f32; 13], // For D(t)
    ) -> CoreResult<f32> {
        let integration = kuramoto_r;
        let reflection = self.sigmoid(meta_accuracy * 4.0 - 2.0);
        let differentiation = self.normalized_purpose_entropy(purpose_vector)?;
        let consciousness = integration * reflection * differentiation;
        Ok(consciousness.clamp(0.0, 1.0))
    }
}
```

**Verification**: R(t) uses sigmoid transformation, D(t) uses Shannon entropy normalized by log2(13).

### 4.2 mod.rs (Lines 125-303)

**Status**: FULLY IMPLEMENTED

```rust
impl GwtSystem {
    pub async fn update_consciousness_auto(
        &self,
        meta_accuracy: f32,
        purpose_vector: &[f32; 13],
    ) -> crate::CoreResult<f32> {
        let kuramoto_r = self.get_kuramoto_r().await;
        self.update_consciousness(kuramoto_r, meta_accuracy, purpose_vector).await
    }
}
```

**Verification**: update_consciousness_auto() fetches internal Kuramoto r and calls full computation.

### 4.3 ego_node.rs (Lines 1-743)

**Status**: PARTIALLY IMPLEMENTED

- SelfEgoNode: Serde Serialize/Deserialize annotations present
- update_from_fingerprint(): IMPLEMENTED
- record_purpose_snapshot(): IMPLEMENTED
- **GAP**: No RocksDB persistence layer wired

### 4.4 listeners.rs

**Status**: IMPLEMENTED (per mod.rs tests)

- DreamEventListener: Queues exiting memories
- NeuromodulationEventListener: Boosts dopamine on MemoryEnters
- MetaCognitiveEventListener: Triggers epistemic action on WorkspaceEmpty

---

## 5. Gap Analysis Summary

| Component | Status | Gap Description | Task |
|-----------|--------|-----------------|------|
| ConsciousnessCalculator | DONE | None | - |
| KuramotoNetwork | DONE | None | - |
| StateMachineManager | DONE | None | - |
| GlobalWorkspace | DONE | None | - |
| SelfEgoNode | DONE | Persistence layer implemented | TASK-GWT-P1-001 (Completed) |
| Event Listeners | DONE | Integration verified | TASK-GWT-P1-002 (Completed) |
| End-to-End Tests | IN PROGRESS | Chaos/integration tests needed | TASK-GWT-P1-003 (Ready) |

---

## 6. Test Plan

### 6.1 Unit Tests (Existing)

| Test | Location | Status |
|------|----------|--------|
| test_consciousness_equation_* | consciousness.rs | PASSING |
| test_gwt_system_* | mod.rs | PASSING |
| test_identity_continuity_* | ego_node.rs | PASSING |
| test_memory_enters_boosts_dopamine | mod.rs | PASSING |
| test_memory_exits_queues_for_dream | mod.rs | PASSING |
| test_workspace_empty_triggers_epistemic | mod.rs | PASSING |

### 6.2 Integration Tests (Required)

| Test | Description | Priority |
|------|-------------|----------|
| IT-GWT-001 | Full consciousness cycle: DORMANT -> CONSCIOUS | Must |
| IT-GWT-002 | SELF_EGO_NODE persists across system restart | Must |
| IT-GWT-003 | IdentityCritical triggers dream consolidation | Must |
| IT-GWT-004 | C(t) < 0.3 for 10s triggers DORMANT transition | Should |

### 6.3 Chaos Tests

| Test | Scenario | Expected |
|------|----------|----------|
| CH-GWT-001 | RocksDB corruption during persist | Graceful degradation |
| CH-GWT-002 | Concurrent event broadcast | No deadlock |
| CH-GWT-003 | Kuramoto network overflow | Clamp to [0,1] |

---

## 7. Acceptance Criteria

### 7.1 Definition of Done

- [x] SELF_EGO_NODE persists to RocksDB CF_EGO_NODE column family (TASK-GWT-P1-001)
- [x] SELF_EGO_NODE loads on GwtSystem::new() if exists (TASK-GWT-P1-001)
- [x] All 3 event listeners correctly wired and verified (TASK-GWT-P1-002)
- [ ] Integration tests prove full consciousness cycle (TASK-GWT-P1-003)
- [ ] Chaos tests pass without panics (TASK-GWT-P1-003)
- [ ] Documentation updated in mod.rs module docs

### 7.2 Verification Commands

```bash
# Run all GWT tests
cargo test --package context-graph-core gwt:: --no-fail-fast

# Run integration tests
cargo test --package context-graph-core --test gwt_integration

# Run chaos tests
cargo test --package context-graph-core --test gwt_chaos
```

---

## 8. Dependencies

### 8.1 Upstream Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| RocksDB | 0.21+ | SELF_EGO_NODE persistence |
| tokio | 1.35+ | Async event handling |
| serde | 1.0+ | Serialization |
| bincode | 1.3+ | Binary encoding |

### 8.2 Downstream Dependents

| Dependent | Impact |
|-----------|--------|
| Dream Layer | Receives IdentityCritical events |
| Meta-UTL | Provides meta_accuracy for R(t) |
| Neuromodulation | Receives dopamine boost signals |

---

## 9. Related Specifications

| Spec ID | Title | Relationship |
|---------|-------|--------------|
| SPEC-DREAM-001 | Dream Layer Implementation | Receives GWT events |
| SPEC-METAUTL-001 | Meta-UTL Self-Correction | Provides R(t) input |
| SPEC-NEUROMOD-001 | Neuromodulation System | Receives event signals |

---

## Appendix A: Constitution.yaml GWT Section Reference

```yaml
gwt:
  consciousness: "C(t) = I(t) x R(t) x D(t) = r(t) x sigma(MetaUTL.predict_accuracy) x H(PV)"
  components: { C: "Consciousness [0,1]", I: "Integration (Kuramoto r)", R: "Self-Reflection (Meta-UTL)", D: "Differentiation (13D entropy)" }

  kuramoto:
    formula: "dtheta_i/dt = omega_i + (K/N) sum_j sin(theta_j - theta_i)"
    order_param: "r * e^(i*psi) = (1/N) sum_j e^(i*theta_j)"
    thresholds: { coherent: "r>=0.8", fragmented: "r<0.5", hypersync: "r>0.95 (pathological)" }

  self_ego_node:
    id: "SELF_EGO_NODE"
    fields: [fingerprint, purpose_vector, identity_trajectory, coherence_with_actions]
    loop: "Retrieve -> A(action,PV) -> if<0.55 self_reflect -> update fingerprint -> store evolution"
    identity_continuity: "IC = cos(PV_t, PV_{t-1}) x r(t); healthy>0.9, warning<0.7, dream<0.5"

  states: { DORMANT: "r<0.3", FRAGMENTED: "0.3<=r<0.5", EMERGING: "0.5<=r<0.8", CONSCIOUS: "r>=0.8", HYPERSYNC: "r>0.95" }
```
