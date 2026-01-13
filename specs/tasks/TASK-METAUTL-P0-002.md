# Task Specification: Lambda Adjustment Algorithm

**Task ID:** TASK-METAUTL-P0-002
**Version:** 2.0.0
**Status:** NOT STARTED - Ready for Implementation
**Layer:** Logic (Layer 2)
**Sequence:** 2
**Priority:** P0 (Critical)
**Estimated Complexity:** High

---

## 1. Metadata

### 1.1 Implements

| Requirement ID | Description | Current Status |
|----------------|-------------|----------------|
| REQ-METAUTL-003 | Adjust lambda_s and lambda_c when prediction_error > 0.2 | ❌ Not Implemented |
| REQ-METAUTL-004 | Lambda adjustment formula: lambda_new = lambda_old + alpha * (target - actual) | ❌ Not Implemented |
| REQ-METAUTL-005 | Alpha modulated by current ACh level | ❌ Not Implemented |
| REQ-METAUTL-006 | Lambda weights SHALL always sum to 1.0 | ✅ Enforced in LifecycleLambdaWeights |
| REQ-METAUTL-007 | Lambda weights SHALL be clamped to [0.05, 0.9] | ⚠️ Partial - per-embedder weights clamped, not lambda_s/lambda_c |

### 1.2 Dependencies

| Task ID | Description | Status |
|---------|-------------|--------|
| TASK-METAUTL-P0-001 | Core types and accuracy history | ✅ COMPLETE |

### 1.3 Blocked By

- ~~TASK-METAUTL-P0-001~~ ✅ COMPLETE - No longer blocked

---

## 2. Context

### 2.1 Problem Statement

This task implements the core lambda adjustment algorithm that enables the system to self-correct its learning parameters (lambda_s for surprise, lambda_c for coherence) based on prediction errors. This is DISTINCT from the per-embedder weight adjustment already implemented in `MetaUtlTracker`.

**Key distinction:**
- `MetaUtlTracker.current_weights[13]` - Per-embedder weights (implemented in TASK-001)
- `LifecycleLambdaWeights.lambda_s/lambda_c` - Surprise/coherence weights (THIS TASK)

### 2.2 Algorithm Requirements

1. Detect when prediction error exceeds threshold (0.2)
2. Compute adjustment delta using ACh-modulated learning rate
3. Apply adjustment while maintaining sum-to-one invariant
4. Clamp values to valid bounds [0.05, 0.9] per NORTH-016
5. Create `AdaptiveLambdaWeights` wrapper around `LifecycleLambdaWeights`

### 2.3 Current State

The current `LifecycleLambdaWeights` in `lambda.rs` provides:
- Fixed weights per lifecycle stage (Infancy: 0.7/0.3, Growth: 0.5/0.5, Maturity: 0.3/0.7)
- Interpolation between stages
- No self-correction capability

This task adds dynamic adjustment capability via a wrapper struct.

### 2.4 Architecture Decision

**IMPORTANT**: Following the architecture decision from TASK-001, new Meta-UTL types should be placed in `crates/context-graph-mcp/src/handlers/core/` directory, NOT in `crates/context-graph-utl/src/meta/`. The MCP crate owns Meta-UTL state.

---

## 3. Input Context Files (MUST READ)

| File | Purpose | Read Priority |
|------|---------|---------------|
| `crates/context-graph-mcp/src/handlers/core/meta_utl_tracker.rs` | Existing Meta-UTL tracker | P0 |
| `crates/context-graph-mcp/src/handlers/core/types.rs` | SelfCorrectionConfig, MetaLearningEvent | P0 |
| `crates/context-graph-utl/src/lifecycle/lambda.rs` | LifecycleLambdaWeights (643 lines) | P0 |
| `crates/context-graph-core/src/gwt/meta_cognitive/` | ACh level access patterns | P1 |
| `docs2/constitution.yaml` | NORTH-016 weight bounds | P0 |
| `specs/functional/SPEC-METAUTL-001.md` | Algorithm specification | P1 |

---

## 4. Scope

### 4.1 In Scope

- Create `SelfCorrectingLambda` trait
- Implement `AdaptiveLambdaWeights` struct
- Implement adjustment algorithm with normalization
- Implement bounds clamping
- Integrate ACh-based learning rate modulation
- Create adjustment validation logic
- Unit tests for algorithm correctness

### 4.2 Out of Scope

- Escalation to Bayesian optimization (TASK-METAUTL-P0-003)
- Event logging (TASK-METAUTL-P0-004)
- MCP tool integration (TASK-METAUTL-P0-005)
- MetaCognitiveLoop integration (TASK-METAUTL-P0-006)

---

## 5. Prerequisites

| Check | Description | Status |
|-------|-------------|--------|
| [x] | TASK-METAUTL-P0-001 completed | ✅ DONE |
| [x] | `crates/context-graph-mcp/src/handlers/core/types.rs` exists | ✅ EXISTS |
| [x] | SelfCorrectionConfig, MetaLearningEvent types compile | ✅ COMPILES |
| [ ] | Understand LifecycleLambdaWeights in `lifecycle/lambda.rs` | READ REQUIRED |

---

## 6. Definition of Done

### 6.1 Required Signatures

#### File: `crates/context-graph-mcp/src/handlers/core/lambda_correction.rs` (NEW FILE)

```rust
// NOTE: This file goes in MCP crate, not UTL crate, per architecture decision
use context_graph_utl::error::{UtlError, UtlResult};
use context_graph_utl::lifecycle::LifecycleLambdaWeights;
use super::types::{
    LambdaAdjustment, SelfCorrectionConfig, Domain,
};

/// Acetylcholine baseline for normalization
pub const ACH_BASELINE: f32 = 0.001;

/// Acetylcholine maximum
pub const ACH_MAX: f32 = 0.002;

/// Trait for self-correcting lambda weights
pub trait SelfCorrectingLambda {
    /// Adjust lambda weights based on prediction error
    ///
    /// # Arguments
    /// - `prediction_error`: Difference between predicted and actual (L_pred - L_actual)
    /// - `ach_level`: Current acetylcholine level [0.001, 0.002]
    ///
    /// # Returns
    /// - `Some(LambdaAdjustment)` if adjustment was made
    /// - `None` if error below threshold
    fn adjust_lambdas(&mut self, prediction_error: f32, ach_level: f32) -> Option<LambdaAdjustment>;

    /// Get current corrected lambda weights
    fn corrected_weights(&self) -> LifecycleLambdaWeights;

    /// Get base (lifecycle) lambda weights
    fn base_weights(&self) -> LifecycleLambdaWeights;

    /// Reset to base weights
    fn reset_to_base(&mut self);

    /// Record accuracy for tracking
    fn record_accuracy(&mut self, accuracy: f32);

    /// Get rolling accuracy average
    fn rolling_accuracy(&self) -> f32;
}

/// Adaptive lambda weights with self-correction capability
#[derive(Debug, Clone)]
pub struct AdaptiveLambdaWeights {
    /// Base weights from lifecycle stage
    base_weights: LifecycleLambdaWeights,
    /// Current corrected weights
    current_weights: LifecycleLambdaWeights,
    /// Configuration
    config: SelfCorrectionConfig,
    /// Accuracy tracker
    accuracy_tracker: MetaAccuracyHistory,
    /// Total adjustments made
    adjustment_count: u64,
    /// Last adjustment applied
    last_adjustment: Option<LambdaAdjustment>,
}

impl AdaptiveLambdaWeights {
    /// Create new adaptive weights from lifecycle weights
    pub fn new(base_weights: LifecycleLambdaWeights, config: SelfCorrectionConfig) -> Self;

    /// Create with default configuration
    pub fn with_defaults(base_weights: LifecycleLambdaWeights) -> Self;

    /// Compute learning rate modulated by ACh level
    ///
    /// Formula: alpha = base_alpha * (ach_level / ACH_BASELINE)
    /// Clamped to [0.01, 0.1]
    fn compute_alpha(&self, ach_level: f32) -> f32;

    /// Compute raw adjustment deltas
    ///
    /// Formula: delta_s = -alpha * prediction_error
    ///          delta_c = -delta_s (maintain sum invariant)
    fn compute_raw_adjustment(&self, prediction_error: f32, alpha: f32) -> (f32, f32);

    /// Apply bounds and normalize
    ///
    /// 1. Add deltas to current weights
    /// 2. Clamp to [lambda_min, lambda_max]
    /// 3. Renormalize to sum=1.0
    fn apply_and_normalize(&self, delta_s: f32, delta_c: f32) -> UtlResult<LifecycleLambdaWeights>;

    /// Validate adjustment doesn't violate invariants
    fn validate_adjustment(&self, new_weights: &LifecycleLambdaWeights) -> UtlResult<()>;

    /// Get adjustment count
    pub fn adjustment_count(&self) -> u64;

    /// Get last adjustment
    pub fn last_adjustment(&self) -> Option<LambdaAdjustment>;

    /// Check if currently at base weights
    pub fn is_at_base(&self) -> bool;

    /// Get deviation from base weights
    pub fn deviation_from_base(&self) -> (f32, f32);
}

impl SelfCorrectingLambda for AdaptiveLambdaWeights {
    fn adjust_lambdas(&mut self, prediction_error: f32, ach_level: f32) -> Option<LambdaAdjustment>;
    fn corrected_weights(&self) -> LifecycleLambdaWeights;
    fn base_weights(&self) -> LifecycleLambdaWeights;
    fn reset_to_base(&mut self);
    fn record_accuracy(&mut self, accuracy: f32);
    fn rolling_accuracy(&self) -> f32;
}

impl Default for AdaptiveLambdaWeights {
    fn default() -> Self;
}

/// Validate that weights satisfy all invariants
///
/// - sum == 1.0 (within epsilon)
/// - lambda_s in [min, max]
/// - lambda_c in [min, max]
pub fn validate_lambda_invariants(
    lambda_s: f32,
    lambda_c: f32,
    config: &SelfCorrectionConfig,
) -> UtlResult<()>;

/// Normalize weights to sum to 1.0
///
/// Returns normalized (lambda_s, lambda_c) tuple
pub fn normalize_lambdas(lambda_s: f32, lambda_c: f32) -> (f32, f32);

/// Clamp lambdas to valid bounds and renormalize
///
/// Returns clamped and normalized (lambda_s, lambda_c) tuple
pub fn clamp_and_normalize(
    lambda_s: f32,
    lambda_c: f32,
    min: f32,
    max: f32,
) -> (f32, f32);
```

### 6.2 Constraints

- `adjust_lambdas` MUST return `None` if `abs(prediction_error) <= error_threshold`
- Lambda sum MUST equal 1.0 after any adjustment (within epsilon 0.001)
- Lambda values MUST be in [lambda_min, lambda_max] after adjustment
- Alpha MUST be clamped to [0.01, 0.1] regardless of ACh level
- NO panics - all error conditions return `UtlResult::Err`
- Adjustment direction: positive error (over-predicted) reduces lambda_s
- ACh modulation: higher ACh = higher learning rate

### 6.3 Verification Commands

```bash
# Type check
cargo check -p context-graph-mcp

# Unit tests (in MCP crate, not UTL)
cargo test -p context-graph-mcp handlers::core::lambda_correction

# Specific algorithm tests
cargo test -p context-graph-mcp test_adjustment_algorithm

# Clippy
cargo clippy -p context-graph-mcp -- -D warnings

# Verify sum invariant across random inputs
cargo test -p context-graph-mcp test_sum_invariant_maintained
```

---

## 7. Files to Create

| Path | Description |
|------|-------------|
| `crates/context-graph-mcp/src/handlers/core/lambda_correction.rs` | Adjustment algorithm implementation |

---

## 8. Files to Modify

| Path | Modification |
|------|--------------|
| `crates/context-graph-mcp/src/handlers/core/mod.rs` | Add `pub mod lambda_correction;` |
| `crates/context-graph-mcp/src/handlers/core/types.rs` | Add `LambdaAdjustment` struct if not present |

---

## 8.1 Source of Truth

| Data Element | Source of Truth | Location |
|--------------|-----------------|----------|
| Lambda bounds (min/max) | Constitution NORTH-016 | `docs2/constitution.yaml` |
| SelfCorrectionConfig defaults | types.rs Default impl | `handlers/core/types.rs` |
| LifecycleLambdaWeights | UTL lifecycle module | `context-graph-utl/src/lifecycle/lambda.rs` |
| ACh baseline/max constants | THIS FILE defines | `lambda_correction.rs` |
| Prediction error threshold | SelfCorrectionConfig | `handlers/core/types.rs` |

---

## 9. Pseudo-Code

### 9.1 adjust_lambdas Implementation

```
FUNCTION adjust_lambdas(prediction_error: f32, ach_level: f32) -> Option<LambdaAdjustment>:
    // Validate inputs
    IF prediction_error.is_nan() OR prediction_error.is_infinite():
        log_warning("Invalid prediction error")
        RETURN None

    // Check threshold
    IF abs(prediction_error) <= self.config.error_threshold:
        RETURN None

    // Clamp ACh to valid range
    ach_level = clamp(ach_level, ACH_BASELINE, ACH_MAX)

    // Compute modulated learning rate
    alpha = self.compute_alpha(ach_level)

    // Compute raw deltas
    (delta_s, delta_c) = self.compute_raw_adjustment(prediction_error, alpha)

    // Apply bounds and normalize
    new_weights = self.apply_and_normalize(delta_s, delta_c)?

    // Validate invariants
    self.validate_adjustment(new_weights)?

    // Store old weights for event logging
    old_s = self.current_weights.lambda_s()
    old_c = self.current_weights.lambda_c()

    // Apply new weights
    self.current_weights = new_weights
    self.adjustment_count += 1

    // Create adjustment record
    adjustment = LambdaAdjustment {
        delta_lambda_s: new_weights.lambda_s() - old_s,
        delta_lambda_c: new_weights.lambda_c() - old_c,
        alpha: alpha,
        trigger_error: prediction_error,
    }
    self.last_adjustment = Some(adjustment)

    RETURN Some(adjustment)
```

### 9.2 compute_alpha Implementation

```
FUNCTION compute_alpha(ach_level: f32) -> f32:
    // Normalize ACh to [0, 1] range
    ach_normalized = (ach_level - ACH_BASELINE) / (ACH_MAX - ACH_BASELINE)
    ach_normalized = clamp(ach_normalized, 0.0, 1.0)

    // Compute modulated alpha
    // Higher ACh = higher learning rate (faster adaptation)
    alpha = self.config.base_alpha * (1.0 + ach_normalized)

    // Clamp to valid range
    RETURN clamp(alpha, 0.01, 0.1)
```

### 9.3 apply_and_normalize Implementation

```
FUNCTION apply_and_normalize(delta_s: f32, delta_c: f32) -> UtlResult<LifecycleLambdaWeights>:
    // Compute raw new values
    new_s = self.current_weights.lambda_s() + delta_s
    new_c = self.current_weights.lambda_c() + delta_c

    // Clamp to bounds
    new_s = clamp(new_s, self.config.lambda_min, self.config.lambda_max)
    new_c = clamp(new_c, self.config.lambda_min, self.config.lambda_max)

    // Normalize to sum=1.0
    sum = new_s + new_c
    new_s = new_s / sum
    new_c = new_c / sum

    // Final bounds check after normalization
    new_s = clamp(new_s, self.config.lambda_min, self.config.lambda_max)
    new_c = clamp(new_c, self.config.lambda_min, self.config.lambda_max)

    // Re-normalize if clamping affected sum
    sum = new_s + new_c
    IF abs(sum - 1.0) > 0.001:
        new_s = new_s / sum
        new_c = new_c / sum

    // Create and validate new weights
    LifecycleLambdaWeights::new(new_s, new_c)
```

---

## 10. Validation Criteria

| Criterion | Validation Method |
|-----------|-------------------|
| Adjustment only when error > 0.2 | Unit test with error = 0.19 vs 0.21 |
| Lambda sum always 1.0 | Property test with random inputs |
| Bounds respected | Test with extreme deltas |
| ACh modulation works | Compare alpha at ACh=0.001 vs 0.002 |
| Normalization correct | Test edge cases (both near bounds) |
| No panics on invalid input | Fuzz test with NaN, Inf, negative |
| Direction correct | Positive error reduces lambda_s |

---

## 11. Test Cases

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::{LifecycleStage, LifecycleLambdaWeights};

    #[test]
    fn test_no_adjustment_below_threshold() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);

        // Error below threshold (0.2)
        let result = adaptive.adjust_lambdas(0.15, ACH_BASELINE);
        assert!(result.is_none());
        assert!(adaptive.is_at_base());
    }

    #[test]
    fn test_adjustment_above_threshold() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);

        // Error above threshold
        let result = adaptive.adjust_lambdas(0.3, ACH_BASELINE);
        assert!(result.is_some());
        assert!(!adaptive.is_at_base());

        let adj = result.unwrap();
        // Positive error should reduce lambda_s
        assert!(adj.delta_lambda_s < 0.0);
    }

    #[test]
    fn test_sum_invariant_maintained() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);

        // Multiple adjustments
        for i in 0..10 {
            let error = 0.25 * (if i % 2 == 0 { 1.0 } else { -1.0 });
            adaptive.adjust_lambdas(error, ACH_BASELINE);

            let weights = adaptive.corrected_weights();
            let sum = weights.lambda_s() + weights.lambda_c();
            assert!((sum - 1.0).abs() < 0.001, "Sum invariant violated: {}", sum);
        }
    }

    #[test]
    fn test_bounds_respected() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Infancy); // 0.7, 0.3
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);

        // Try to push lambda_s above max with many positive error corrections
        for _ in 0..50 {
            adaptive.adjust_lambdas(-0.5, ACH_MAX); // Negative error increases lambda_s
        }

        let weights = adaptive.corrected_weights();
        assert!(weights.lambda_s() <= 0.9, "lambda_s exceeded max: {}", weights.lambda_s());
        assert!(weights.lambda_c() >= 0.1, "lambda_c below min: {}", weights.lambda_c());
    }

    #[test]
    fn test_ach_modulation() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let adaptive = AdaptiveLambdaWeights::with_defaults(base);

        let alpha_low = adaptive.compute_alpha(ACH_BASELINE);
        let alpha_high = adaptive.compute_alpha(ACH_MAX);

        assert!(alpha_high > alpha_low, "Higher ACh should give higher alpha");
    }

    #[test]
    fn test_reset_to_base() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);

        // Make adjustment
        adaptive.adjust_lambdas(0.3, ACH_BASELINE);
        assert!(!adaptive.is_at_base());

        // Reset
        adaptive.reset_to_base();
        assert!(adaptive.is_at_base());
    }

    #[test]
    fn test_nan_handling() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);

        let result = adaptive.adjust_lambdas(f32::NAN, ACH_BASELINE);
        assert!(result.is_none());

        let result = adaptive.adjust_lambdas(f32::INFINITY, ACH_BASELINE);
        assert!(result.is_none());
    }

    #[test]
    fn test_adjustment_direction() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);

        // Positive error (over-predicted surprise)
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);
        let adj = adaptive.adjust_lambdas(0.3, ACH_BASELINE).unwrap();
        assert!(adj.delta_lambda_s < 0.0, "Positive error should reduce lambda_s");

        // Negative error (under-predicted surprise)
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);
        let adj = adaptive.adjust_lambdas(-0.3, ACH_BASELINE).unwrap();
        assert!(adj.delta_lambda_s > 0.0, "Negative error should increase lambda_s");
    }

    #[test]
    fn test_clamp_and_normalize() {
        let config = SelfCorrectionConfig::default();

        // Test extreme values
        let (s, c) = clamp_and_normalize(2.0, -1.0, config.lambda_min, config.lambda_max);
        assert!(s >= config.lambda_min && s <= config.lambda_max);
        assert!(c >= config.lambda_min && c <= config.lambda_max);
        assert!((s + c - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_accuracy_tracking() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);

        adaptive.record_accuracy(0.8);
        adaptive.record_accuracy(0.9);
        adaptive.record_accuracy(0.7);

        let avg = adaptive.rolling_accuracy();
        assert!((avg - 0.8).abs() < 0.001);
    }
}
```

---

## 12. Rollback Plan

If this task fails validation:

1. Revert files: `git checkout -- crates/context-graph-mcp/src/handlers/core/lambda_correction.rs`
2. Remove mod declaration from `core/mod.rs`
3. Document failure in task notes
4. Types from TASK-001 remain unaffected
5. Existing MetaUtlTracker functionality unaffected

---

## 13. Full State Verification (FSV) Requirements

### 13.1 FSV Pattern

```rust
// DO NOT rely on return values alone - always verify state
let adjustment = adaptive.adjust_lambdas(0.3, ACH_BASELINE);

// VERIFY: Read back actual state after adjustment
let current = adaptive.corrected_weights();
let sum = current.lambda_s() + current.lambda_c();

// FSV assertions
assert!((sum - 1.0).abs() < 0.001, "FSV FAIL: sum invariant violated");
assert!(current.lambda_s() >= 0.05, "FSV FAIL: lambda_s below min");
assert!(current.lambda_s() <= 0.9, "FSV FAIL: lambda_s above max");
```

### 13.2 Edge Case Audit (3 Required)

| Edge Case ID | Condition | Before State | After State | Expected |
|--------------|-----------|--------------|-------------|----------|
| EC-001 | Both lambdas at min bound (impossible) | λ_s=0.05, λ_c=0.05 | N/A | Should reject, sum != 1.0 |
| EC-002 | Lambda_s at max, try to increase | λ_s=0.9, λ_c=0.1 | λ_s=0.9, λ_c=0.1 | No change, clamped |
| EC-003 | Extreme negative error with high ACh | λ_s=0.5, error=-0.9, ACh=max | λ_s<=0.9 | Should not exceed max |

### 13.3 Evidence of Success

Test must print actual values:
```rust
#[test]
fn test_fsv_lambda_adjustment_evidence() {
    let mut adaptive = AdaptiveLambdaWeights::with_defaults(...);

    // BEFORE state
    println!("BEFORE: lambda_s={}, lambda_c={}",
        adaptive.corrected_weights().lambda_s(),
        adaptive.corrected_weights().lambda_c());

    // Execute
    let adj = adaptive.adjust_lambdas(0.35, ACH_MAX);

    // AFTER state
    println!("AFTER: lambda_s={}, lambda_c={}",
        adaptive.corrected_weights().lambda_s(),
        adaptive.corrected_weights().lambda_c());
    println!("ADJUSTMENT: delta_s={:?}", adj.map(|a| a.delta_lambda_s));
}
```

---

## 14. Notes

- The adjustment algorithm is conservative by design (alpha clamped to 0.1 max)
- Direction convention: positive prediction_error means over-prediction
- ACh modulation provides feedback from GWT consciousness level
- Bounds of [0.05, 0.9] per NORTH-016 (not [0.1, 0.9] as originally specified)
- Sum normalization happens AFTER clamping to handle edge cases
- **NO BACKWARDS COMPATIBILITY**: If invariants are violated, FAIL FAST with error

---

## 15. Fail-Fast Error Handling

```rust
// NO silent failures - all errors must be logged and returned
fn adjust_lambdas(&mut self, error: f32, ach: f32) -> Result<Option<LambdaAdjustment>, UtlError> {
    if error.is_nan() || error.is_infinite() {
        tracing::error!("INVALID INPUT: prediction_error={}", error);
        return Err(UtlError::InvalidInput {
            field: "prediction_error".into(),
            reason: format!("NaN or Infinite: {}", error)
        });
    }
    // ... rest of implementation
}
```

---

**Task History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | ContextGraph Team | Initial task specification |
| 2.0.0 | 2026-01-12 | AI Agent | Updated file paths to MCP crate, added FSV sections, fail-fast requirements |
