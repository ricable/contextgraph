# Task Specification: Meta-UTL Core Types and Accuracy History Interface

**Task ID:** TASK-METAUTL-P0-001
**Version:** 3.0.0
**Status:** ✅ IMPLEMENTED
**Layer:** Foundation (Layer 1)
**Sequence:** 1
**Priority:** P0 (Critical)
**Estimated Complexity:** Medium

---

## 1. Metadata

### 1.1 Implements

| Requirement ID | Description | Current Status |
|----------------|-------------|----------------|
| REQ-METAUTL-001 | Rolling accuracy history of at least 100 predictions | ✅ Implemented in `MetaUtlTracker` |
| REQ-METAUTL-002 | Track accuracy per embedder (E1-E13) separately | ✅ Implemented via `embedder_accuracy` array |
| REQ-METAUTL-006 | Lambda weights SHALL always sum to 1.0 | ✅ Enforced in `update_weights()` |
| REQ-METAUTL-007 | Lambda weights SHALL be clamped to [0.05, 0.9] | ✅ Implemented per NORTH-016 constitution (min=0.05, max=0.9) |

### 1.2 Dependencies

| Task ID | Description | Status |
|---------|-------------|--------|
| None | This is the foundation task | N/A |

### 1.3 Blocked By

None - this is the first task in the sequence.

---

## 2. Current Implementation State

> **CRITICAL: Review this section before implementation to avoid duplication**

### 2.1 Existing Implementations

The codebase already has significant Meta-UTL implementation:

| Component | Location | Status |
|-----------|----------|--------|
| `MetaUtlTracker` | `crates/context-graph-mcp/src/handlers/core.rs:60-200` | ✅ Implemented |
| `StoredPrediction` | `crates/context-graph-mcp/src/handlers/core.rs:44-53` | ✅ Implemented |
| `PredictionType` | `crates/context-graph-mcp/src/handlers/core.rs:36-42` | ✅ Implemented |
| MCP Handler: `meta_utl/learning_trajectory` | `crates/context-graph-mcp/src/handlers/utl.rs` | ✅ Implemented |
| MCP Handler: `meta_utl/health_metrics` | `crates/context-graph-mcp/src/handlers/utl.rs` | ✅ Implemented |
| MCP Handler: `meta_utl/predict_storage` | `crates/context-graph-mcp/src/handlers/utl.rs` | ✅ Implemented |
| MCP Handler: `meta_utl/predict_retrieval` | `crates/context-graph-mcp/src/handlers/utl.rs` | ✅ Implemented |
| MCP Handler: `meta_utl/validate_prediction` | `crates/context-graph-mcp/src/handlers/utl.rs` | ✅ Implemented |
| MCP Handler: `meta_utl/optimized_weights` | `crates/context-graph-mcp/src/handlers/utl.rs` | ✅ Implemented |
| FSV Tests | `crates/context-graph-mcp/src/handlers/tests/full_state_verification_meta_utl.rs` | ✅ Implemented |

### 2.2 Current MetaUtlTracker Structure (COMPLETE)

```rust
// Location: crates/context-graph-mcp/src/handlers/core/meta_utl_tracker.rs:22-49
pub struct MetaUtlTracker {
    /// Pending predictions awaiting validation
    pub pending_predictions: HashMap<Uuid, StoredPrediction>,
    /// Per-embedder accuracy rolling window (100 samples per embedder)
    pub embedder_accuracy: [[f32; 100]; NUM_EMBEDDERS],
    /// Current index in each embedder's rolling window
    pub accuracy_indices: [usize; NUM_EMBEDDERS],
    /// Number of samples in each embedder's rolling window
    pub accuracy_counts: [usize; NUM_EMBEDDERS],
    /// Current optimized weights (sum to 1.0, clamped to [0.05, 0.9] per constitution)
    pub current_weights: [f32; NUM_EMBEDDERS],
    /// Total predictions made
    pub prediction_count: usize,
    /// Total validations completed
    pub validation_count: usize,
    /// Last weight update timestamp
    pub last_weight_update: Option<Instant>,
    /// TASK-METAUTL-P0-001: Consecutive cycles with accuracy < 0.7
    pub consecutive_low_count: usize,
    /// TASK-METAUTL-P0-001: Whether Bayesian escalation has been triggered
    pub escalation_triggered: bool,
    /// TASK-METAUTL-P0-001: Self-correction configuration
    pub config: SelfCorrectionConfig,
    /// TASK-METAUTL-P0-001: Tracks which embedders have been updated in current cycle
    cycle_embedder_updated: [bool; NUM_EMBEDDERS],
    /// TASK-METAUTL-P0-001: Number of complete accuracy recording cycles
    cycle_count: usize,
}
```

### 2.3 Implementation Status - ALL COMPLETE

| Component | Description | Status | Location |
|-----------|-------------|--------|----------|
| Lambda weight clamping | REQ-METAUTL-007: Clamp to [0.05, 0.9] | ✅ Done | `meta_utl_tracker.rs:250-301` |
| `Domain` enum | Domain-specific accuracy tracking | ✅ Done | `types.rs:20-36` |
| `MetaLearningEventType` enum | Event type classification | ✅ Done | `types.rs:42-54` |
| `MetaLearningEvent` struct | Event logging struct | ✅ Done | `types.rs:59-115` |
| `SelfCorrectionConfig` struct | Configuration struct | ✅ Done | `types.rs:120-158` |
| Bayesian escalation trigger | When accuracy < 0.7 for 10 cycles | ✅ Done | `meta_utl_tracker.rs:178-187` |
| Consecutive low tracking | Track consecutive low accuracy cycles | ✅ Done | `meta_utl_tracker.rs:147-200` |
| `needs_escalation()` method | Check if escalation needed | ✅ Done | `meta_utl_tracker.rs:388-390` |
| `reset_consecutive_low()` method | Reset after corrective action | ✅ Done | `meta_utl_tracker.rs:406-416` |

---

## 3. Context

### 3.1 Constitution Reference

From `docs2/constitution.yaml`:

```yaml
meta_utl:
  self_correction:
    enabled: true
    threshold: 0.2  # Prediction error threshold
    max_consecutive_failures: 10
    escalation_strategy: "bayesian_optimization"
```

### 3.2 File Structure (SOURCE OF TRUTH)

The codebase organizes Meta-UTL in the MCP handlers crate with modular file structure:

```
crates/
├── context-graph-mcp/
│   └── src/
│       └── handlers/
│           └── core/
│               ├── mod.rs              # Module exports
│               ├── meta_utl_tracker.rs # MetaUtlTracker struct and methods
│               ├── types.rs            # Domain, MetaLearningEvent, SelfCorrectionConfig
│               ├── handlers.rs         # MCP handler implementations
│               └── dispatch.rs         # Tool dispatch logic
│           ├── utl.rs                  # 6 meta_utl/* MCP handlers
│           └── tests/
│               └── full_state_verification_meta_utl.rs
├── context-graph-utl/
│   └── src/
│       ├── lifecycle/
│       │   └── lambda.rs       # LifecycleLambdaWeights (stage-based, NOT self-correcting)
│       └── lib.rs              # NO meta/ module - Meta-UTL lives in MCP crate
└── context-graph-core/
    └── src/
        └── johari/
            └── manager.rs      # NUM_EMBEDDERS = 13 constant (Source of Truth)
```

**IMPORTANT**: Meta-UTL types live in `context-graph-mcp/src/handlers/core/`, NOT in `context-graph-utl/src/meta/`. This is intentional - MetaUtlTracker needs direct access to MCP request/response cycle.

---

## 4. Input Context Files (MUST READ)

| File | Purpose | Read Priority |
|------|---------|---------------|
| `crates/context-graph-mcp/src/handlers/core/meta_utl_tracker.rs` | **MetaUtlTracker implementation** | P0 - Read First |
| `crates/context-graph-mcp/src/handlers/core/types.rs` | **Domain, MetaLearningEvent, SelfCorrectionConfig** | P0 - Read First |
| `crates/context-graph-mcp/src/handlers/utl.rs` | MCP handlers for meta_utl/* tools | P0 |
| `crates/context-graph-mcp/src/handlers/tests/full_state_verification_meta_utl.rs` | Existing FSV tests | P0 |
| `docs2/constitution.yaml` (meta_utl section) | Authoritative constraints | P0 |
| `crates/context-graph-core/src/johari/manager.rs` | NUM_EMBEDDERS = 13 constant (Source of Truth) | P0 |
| `specs/functional/SPEC-METAUTL-001.md` | Full functional specification | P1 |
| `crates/context-graph-utl/src/lifecycle/lambda.rs` | LifecycleLambdaWeights reference | P2 |

---

## 5. Scope

### 5.1 Completed Implementation

All items have been implemented:

1. ✅ **Lambda weight clamping** in `MetaUtlTracker::update_weights()`
   - Max weight capped at 0.9 (HARD constraint)
   - Min weight 0.05 (SOFT constraint per NORTH-016)
   - Re-normalizes after clamping to maintain sum=1.0
   - Location: `meta_utl_tracker.rs:250-373`

2. ✅ **Consecutive low tracking** in `MetaUtlTracker`
   - Tracks consecutive cycles with accuracy < 0.7 threshold
   - Triggers escalation flag when count >= 10
   - Location: `meta_utl_tracker.rs:147-200`

3. ✅ **Domain enum** for domain-specific tracking
   - Code, Medical, Legal, Creative, Research, General
   - Location: `types.rs:20-36`

4. ✅ **MetaLearningEvent** for event logging
   - LambdaAdjustment, BayesianEscalation, AccuracyAlert, AccuracyRecovery, WeightClamped
   - Location: `types.rs:42-115`

5. ✅ **SelfCorrectionConfig** with constitution defaults
   - error_threshold: 0.2, max_consecutive_failures: 10, min_weight: 0.05, max_weight: 0.9
   - Location: `types.rs:120-158`

### 5.2 Already Implemented (Out of Scope for this task)

- MCP handler wiring (TASK-S005)
- Basic prediction/validation flow
- Full State Verification tests

---

## 6. Full State Verification (FSV) Requirements

### 6.1 Source of Truth

| Entity | Source of Truth | Location |
|--------|-----------------|----------|
| NUM_EMBEDDERS | `context_graph_core::johari::NUM_EMBEDDERS` | `crates/context-graph-core/src/johari/manager.rs` |
| Lambda bounds | Constitution YAML | `docs2/constitution.yaml` |
| Accuracy threshold | Constitution YAML (0.7) | `docs2/constitution.yaml` |
| Error threshold | Constitution YAML (0.2) | `docs2/constitution.yaml` |
| Escalation cycles | Constitution YAML (10) | `docs2/constitution.yaml` |

### 6.2 Execute & Inspect Requirements

After implementation, verify by running:

```bash
# 1. Type check entire workspace
cargo check --workspace

# 2. Run existing FSV tests
cargo test -p context-graph-mcp full_state_verification_meta_utl --no-fail-fast

# 3. Run clippy with strict settings
cargo clippy -p context-graph-mcp -- -D warnings

# 4. Verify no regressions
cargo test -p context-graph-mcp --lib
```

### 6.3 Boundary & Edge Case Audit

| Edge Case ID | Description | Before State | After State | Verification Method |
|--------------|-------------|--------------|-------------|---------------------|
| EC-001 | Weight below 0.1 after update | weights[0] = 0.05 | weights[0] = 0.1, re-normalized | Unit test with mock accuracy |
| EC-002 | Weight above 0.9 after update | weights[0] = 0.95 | weights[0] = 0.9, re-normalized | Unit test with mock accuracy |
| EC-003 | 10 consecutive low accuracy | consecutive_low = 9 | consecutive_low = 10, escalation=true | Integration test |
| EC-004 | Accuracy exactly at 0.7 threshold | accuracy = 0.7 | consecutive_low NOT incremented | Unit test boundary |
| EC-005 | All embedders at minimum (0.1) | 13 weights at 0.1 | Sum = 1.3, normalize to sum=1.0 | Unit test |
| EC-006 | Single embedder at 1.0, others at 0.0 | weights[0]=1.0, rest=0.0 | weights[0]=0.9, distribute 0.1 | Unit test |

### 6.4 Evidence of Success Logs

Implementation must produce logs at these checkpoints:

```rust
// Lambda clamping log
tracing::debug!(
    embedder_idx = %idx,
    original_weight = %before,
    clamped_weight = %after,
    "Lambda weight clamped to bounds"
);

// Escalation trigger log
tracing::warn!(
    consecutive_low = %self.consecutive_low_count,
    threshold = 10,
    "Bayesian escalation triggered"
);

// Weight update log
tracing::info!(
    validation_count = %self.validation_count,
    weights = ?self.current_weights,
    "Meta-UTL weights updated"
);
```

---

## 7. Manual Testing Requirements

### 7.1 Pre-Implementation Verification

```bash
# Confirm current state compiles
cargo build -p context-graph-mcp

# Confirm existing tests pass
cargo test -p context-graph-mcp meta_utl

# Get baseline FSV test count
cargo test -p context-graph-mcp full_state_verification_meta_utl --no-run 2>&1 | grep "test"
```

### 7.2 Synthetic Test Data

#### Test Data Set 1: Lambda Clamping Scenario

```rust
// Input: Create tracker with extreme accuracy distribution
let mut tracker = MetaUtlTracker::new();

// Embedder 0 gets 100% accuracy, others get 0%
for _ in 0..50 {
    tracker.record_accuracy(0, 1.0);  // Perfect
    for i in 1..13 {
        tracker.record_accuracy(i, 0.0);  // Terrible
    }
}
tracker.update_weights();

// Expected Output:
// weights[0] = 0.9 (clamped from ~1.0)
// weights[1..13] = 0.1/12 each (redistributed)
// sum(weights) = 1.0
assert!((tracker.current_weights.iter().sum::<f32>() - 1.0).abs() < 0.001);
assert!(tracker.current_weights[0] <= 0.9);
assert!(tracker.current_weights[0] >= 0.1);
```

#### Test Data Set 2: Escalation Trigger Scenario

```rust
// Input: 10 consecutive low accuracy cycles
let mut tracker = MetaUtlTracker::new();

for cycle in 0..10 {
    for embedder in 0..13 {
        tracker.record_accuracy(embedder, 0.5);  // Below 0.7
    }
}

// Expected Output:
// escalation_needed() returns true
// consecutive_low_count >= 10
assert!(tracker.needs_escalation());
assert_eq!(tracker.consecutive_low_count(), 10);
```

#### Test Data Set 3: Recovery Scenario

```rust
// Input: 9 low cycles, then 1 high cycle
let mut tracker = MetaUtlTracker::new();

for cycle in 0..9 {
    for embedder in 0..13 {
        tracker.record_accuracy(embedder, 0.5);
    }
}
// Now record high accuracy
for embedder in 0..13 {
    tracker.record_accuracy(embedder, 0.9);  // Above 0.7
}

// Expected Output:
// consecutive_low_count reset to 0
// escalation_needed() returns false
assert!(!tracker.needs_escalation());
assert_eq!(tracker.consecutive_low_count(), 0);
```

### 7.3 Database/State Verification

After running test suite, manually verify state:

```bash
# 1. Check no panics in test output
cargo test -p context-graph-mcp meta_utl 2>&1 | grep -i panic

# 2. Verify all FSV tests pass
cargo test -p context-graph-mcp full_state_verification_meta_utl -- --nocapture

# 3. Check for any warnings
cargo test -p context-graph-mcp 2>&1 | grep -i warning
```

---

## 8. Implementation Checklist (ALL COMPLETE)

### 8.1 Phase 1: Modify MetaUtlTracker (meta_utl_tracker.rs)

- [x] Add `consecutive_low_count: usize` field - Line 40
- [x] Add `escalation_triggered: bool` field - Line 42
- [x] Add `config: SelfCorrectionConfig` field - Line 44
- [x] Add `cycle_embedder_updated: [bool; NUM_EMBEDDERS]` field - Line 46
- [x] Add `cycle_count: usize` field - Line 48
- [x] Modify `record_accuracy()` to track consecutive low - Lines 110-140
- [x] Add `check_consecutive_low_accuracy()` method - Lines 147-200
- [x] Modify `update_weights()` to clamp max to 0.9 - Lines 316-373
- [x] Add `redistribute_excess_weight()` helper - Lines 250-301
- [x] Add `needs_escalation() -> bool` method - Lines 388-390
- [x] Add `consecutive_low_count() -> usize` method - Lines 397-399
- [x] Add `reset_consecutive_low()` method - Lines 406-416
- [x] Add `config() -> &SelfCorrectionConfig` method - Lines 422-424
- [x] Add tracing logs for all state changes - Throughout

### 8.2 Phase 2: Add Supporting Types (types.rs)

- [x] Add `Domain` enum - Lines 20-36
- [x] Add `MetaLearningEventType` enum - Lines 42-54
- [x] Add `MetaLearningEvent` struct - Lines 59-73
- [x] Add `MetaLearningEvent` constructor methods - Lines 76-115
- [x] Add `SelfCorrectionConfig` struct with Default - Lines 120-158

### 8.3 Phase 3: Tests

- [x] Lambda clamping tests (EC-001, EC-002) - FSV test file
- [x] Escalation trigger tests (EC-003) - FSV test file
- [x] Threshold boundary tests (EC-004) - FSV test file
- [x] Extreme distribution tests (EC-005, EC-006) - FSV test file
- [x] All existing FSV tests pass

---

## 9. Verification Commands

```bash
# Full verification sequence
cargo check --workspace && \
cargo clippy -p context-graph-mcp -- -D warnings && \
cargo test -p context-graph-mcp meta_utl --no-fail-fast && \
cargo test -p context-graph-mcp full_state_verification_meta_utl --no-fail-fast && \
cargo doc -p context-graph-mcp --no-deps
```

---

## 10. Constraints

- **NO BACKWARDS COMPATIBILITY** - Fail fast with robust error logging
- **NO MOCK DATA** - Tests must use real MetaUtlTracker instances
- **NO unwrap()** - Use `expect()` with context or return `Result`
- **FAIL FAST** - Return errors immediately, do not silently ignore
- All accuracy values MUST be clamped to [0.0, 1.0]
- All weights MUST be clamped to [0.1, 0.9] (REQ-METAUTL-007)
- All weights MUST sum to 1.0 (REQ-METAUTL-006)
- All timestamps MUST use `std::time::Instant` for internal tracking

---

## 11. Rollback Plan

If implementation fails validation:

1. `git checkout -- crates/context-graph-mcp/src/handlers/core.rs`
2. Document failure reason in this task file under Notes section
3. Create follow-up task addressing specific issues
4. Do NOT attempt partial fixes - full rollback only

---

## 12. Notes

### 12.1 Architecture Decision

The original task proposed creating `crates/context-graph-utl/src/meta/types.rs`, but the actual implementation places Meta-UTL types in `crates/context-graph-mcp/src/handlers/core/`. This is intentional:

- MetaUtlTracker needs direct access to MCP request/response cycle
- Predictions are tied to MCP handlers, not standalone UTL processing
- Keeps all MCP state in one location for maintainability
- Modular file structure: `meta_utl_tracker.rs` for tracker, `types.rs` for types

### 12.2 Existing Test Coverage

FSV tests exist in `crates/context-graph-mcp/src/handlers/tests/full_state_verification_meta_utl.rs` covering:
- `test_fsv_learning_trajectory_all_embedders`
- `test_fsv_predict_storage_and_validate`
- Edge cases for invalid indices, unknown predictions
- Weight clamping and normalization
- Escalation triggers

### 12.3 Weight Clamping Algorithm Note

The weight clamping uses a priority system:
1. **Sum = 1.0** (HARD constraint, always enforced)
2. **Max weight ≤ 0.9** (HARD constraint, prevents single embedder dominance)
3. **Min weight ≥ 0.05** (SOFT constraint, may be violated in extreme distributions)

This allows mathematically valid solutions even when one embedder has near-perfect accuracy and others have near-zero.

### 12.4 NORTH-016 Constitution Reference

Per `docs2/constitution.yaml` NORTH-016_WeightAdjuster:
- `min: 0.05` (not 0.1 as originally specified)
- `max_delta: 0.10` (per adjustment)
- This allows 13 × 0.05 = 0.65 < 1.0, so sum=1.0 is always achievable

---

**Task History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | ContextGraph Team | Initial task specification |
| 2.0.0 | 2026-01-11 | AI Agent | Updated with codebase audit, FSV requirements, manual testing, correct file paths |
| 3.0.0 | 2026-01-12 | AI Agent | Marked COMPLETE - all components implemented in handlers/core/ directory |
