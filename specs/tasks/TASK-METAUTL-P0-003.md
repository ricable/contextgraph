# Task Specification: Escalation Logic and Bayesian Optimization

**Task ID:** TASK-METAUTL-P0-003
**Version:** 1.0.0
**Status:** Ready
**Layer:** Logic (Layer 2)
**Sequence:** 3
**Priority:** P0 (Critical)
**Estimated Complexity:** High

---

## 1. Metadata

### 1.1 Implements

| Requirement ID | Description |
|----------------|-------------|
| REQ-METAUTL-008 | Escalate to Bayesian optimization when accuracy < 0.7 for 10 cycles |
| REQ-METAUTL-009 | Bayesian optimization SHALL use GP surrogate with EI acquisition |

### 1.2 Dependencies

| Task ID | Description | Status |
|---------|-------------|--------|
| TASK-METAUTL-P0-001 | Core types | Must complete first |
| TASK-METAUTL-P0-002 | Lambda adjustment | Must complete first |

### 1.3 Blocked By

- TASK-METAUTL-P0-002 (adjustment algorithm must exist)

---

## 2. Context

When the gradient-based lambda adjustment fails to improve accuracy (< 0.7 for 10 consecutive cycles), the system must escalate to a more sophisticated optimization strategy. This task implements:

1. Escalation detection logic
2. Bayesian optimization with Gaussian Process surrogate
3. Expected Improvement (EI) acquisition function
4. Fallback mechanisms when BO fails
5. Human escalation trigger after 3 failed BO attempts

This aligns with the 4-level Adaptive Threshold Calibration (ATC) architecture in the constitution:
- L1: EWMA (per-query)
- L2: Temperature scaling (hourly)
- L3: Thompson sampling/UCB (session)
- L4: Bayesian optimization (weekly/escalation)

---

## 3. Input Context Files

| File | Purpose |
|------|---------|
| `crates/context-graph-utl/src/meta/types.rs` | EscalationStatus, config types |
| `crates/context-graph-utl/src/meta/correction.rs` | AdaptiveLambdaWeights |
| `docs2/constitution.yaml` | ATC L4 Bayesian spec |
| `specs/functional/SPEC-METAUTL-001.md` | Escalation requirements |

---

## 4. Scope

### 4.1 In Scope

- Create `EscalationManager` struct
- Implement escalation detection logic
- Implement simple GP surrogate model (1D lambda_s input)
- Implement Expected Improvement acquisition
- Implement proposal-evaluation loop
- Implement fallback to gradient adjustment
- Implement human escalation trigger
- Unit tests for escalation paths

### 4.2 Out of Scope

- Full multi-dimensional Bayesian optimization (Phase 2)
- Integration with external BO libraries (botorch, etc.)
- Event logging integration (TASK-METAUTL-P0-004)
- MCP tool wiring (TASK-METAUTL-P0-005)

---

## 5. Prerequisites

| Check | Description |
|-------|-------------|
| [ ] | TASK-METAUTL-P0-002 completed |
| [ ] | `AdaptiveLambdaWeights` exists and compiles |

---

## 6. Definition of Done

### 6.1 Required Signatures

#### File: `crates/context-graph-utl/src/meta/escalation.rs`

```rust
use crate::error::{UtlError, UtlResult};
use crate::lifecycle::LifecycleLambdaWeights;
use super::types::{
    EscalationStatus, SelfCorrectionConfig, MetaAccuracyHistory,
};
use super::correction::AdaptiveLambdaWeights;

/// Maximum BO iterations per escalation
pub const MAX_BO_ITERATIONS: usize = 10;

/// Number of initial random samples for GP
pub const INITIAL_SAMPLES: usize = 5;

/// Human escalation threshold (consecutive failed escalations)
pub const HUMAN_ESCALATION_THRESHOLD: u32 = 3;

/// Trait for escalation management
pub trait EscalationHandler {
    /// Check if escalation should be triggered
    fn should_escalate(&self) -> bool;

    /// Trigger escalation process
    fn trigger_escalation(&mut self) -> UtlResult<EscalationResult>;

    /// Get current escalation status
    fn status(&self) -> EscalationStatus;

    /// Reset escalation state
    fn reset(&mut self);

    /// Check if human escalation is needed
    fn needs_human_review(&self) -> bool;
}

/// Result of an escalation attempt
#[derive(Debug, Clone)]
pub struct EscalationResult {
    /// Whether escalation succeeded
    pub success: bool,
    /// Proposed new lambda weights
    pub proposed_weights: Option<LifecycleLambdaWeights>,
    /// Expected improvement score
    pub expected_improvement: f32,
    /// Number of BO iterations performed
    pub iterations: usize,
    /// Reason if failed
    pub failure_reason: Option<String>,
}

/// Gaussian Process observation
#[derive(Debug, Clone, Copy)]
pub struct GpObservation {
    /// Lambda_s value (input)
    pub lambda_s: f32,
    /// Observed accuracy (output)
    pub accuracy: f32,
}

/// Simple 1D Gaussian Process for lambda optimization
#[derive(Debug, Clone)]
pub struct SimpleGaussianProcess {
    /// Observations
    observations: Vec<GpObservation>,
    /// Length scale for RBF kernel
    length_scale: f32,
    /// Noise variance
    noise_var: f32,
    /// Signal variance
    signal_var: f32,
}

impl SimpleGaussianProcess {
    /// Create new GP with default hyperparameters
    pub fn new() -> Self;

    /// Create with custom hyperparameters
    pub fn with_params(length_scale: f32, noise_var: f32, signal_var: f32) -> Self;

    /// Add observation
    pub fn add_observation(&mut self, obs: GpObservation);

    /// Predict mean and variance at a point
    pub fn predict(&self, lambda_s: f32) -> (f32, f32);

    /// Compute Expected Improvement at a point
    ///
    /// EI(x) = (mu(x) - f_best) * Phi(Z) + sigma(x) * phi(Z)
    /// where Z = (mu(x) - f_best) / sigma(x)
    pub fn expected_improvement(&self, lambda_s: f32) -> f32;

    /// Find lambda_s that maximizes EI
    ///
    /// Uses grid search over [lambda_min, lambda_max]
    pub fn maximize_ei(&self, lambda_min: f32, lambda_max: f32, grid_size: usize) -> f32;

    /// Get current best observation
    pub fn best_observation(&self) -> Option<GpObservation>;

    /// Get number of observations
    pub fn num_observations(&self) -> usize;

    /// Clear all observations
    pub fn clear(&mut self);

    /// RBF kernel: k(x, x') = signal_var * exp(-0.5 * (x-x')^2 / length_scale^2)
    fn rbf_kernel(&self, x1: f32, x2: f32) -> f32;

    /// Standard normal CDF (approximation)
    fn standard_normal_cdf(&self, x: f32) -> f32;

    /// Standard normal PDF
    fn standard_normal_pdf(&self, x: f32) -> f32;
}

impl Default for SimpleGaussianProcess {
    fn default() -> Self;
}

/// Escalation manager
#[derive(Debug, Clone)]
pub struct EscalationManager {
    /// Configuration
    config: SelfCorrectionConfig,
    /// Current status
    status: EscalationStatus,
    /// Gaussian process for optimization
    gp: SimpleGaussianProcess,
    /// Consecutive failed escalations
    failed_escalations: u32,
    /// Total escalation attempts
    total_attempts: u32,
    /// Successful escalations
    successful_escalations: u32,
    /// Last proposed weights
    last_proposal: Option<LifecycleLambdaWeights>,
}

impl EscalationManager {
    /// Create new escalation manager
    pub fn new(config: SelfCorrectionConfig) -> Self;

    /// Create with default config
    pub fn with_defaults() -> Self;

    /// Check escalation condition against accuracy tracker
    pub fn check_escalation_condition(&self, accuracy_history: &MetaAccuracyHistory) -> bool;

    /// Run Bayesian optimization to find better lambdas
    ///
    /// # Arguments
    /// - `current_weights`: Current lambda weights
    /// - `evaluate_fn`: Function to evaluate accuracy at proposed weights
    ///
    /// # Returns
    /// Best weights found or error
    pub fn run_bayesian_optimization<F>(
        &mut self,
        current_weights: LifecycleLambdaWeights,
        evaluate_fn: F,
    ) -> UtlResult<LifecycleLambdaWeights>
    where
        F: FnMut(LifecycleLambdaWeights) -> f32;

    /// Generate initial samples for GP
    fn generate_initial_samples(
        &self,
        current: LifecycleLambdaWeights,
    ) -> Vec<f32>;

    /// Propose next lambda_s to evaluate
    fn propose_next(&self) -> f32;

    /// Check if BO has converged
    fn has_converged(&self) -> bool;

    /// Record escalation outcome
    pub fn record_outcome(&mut self, success: bool);

    /// Get statistics
    pub fn stats(&self) -> EscalationStats;
}

impl EscalationHandler for EscalationManager {
    fn should_escalate(&self) -> bool;
    fn trigger_escalation(&mut self) -> UtlResult<EscalationResult>;
    fn status(&self) -> EscalationStatus;
    fn reset(&mut self);
    fn needs_human_review(&self) -> bool;
}

impl Default for EscalationManager {
    fn default() -> Self;
}

/// Escalation statistics
#[derive(Debug, Clone, Copy)]
pub struct EscalationStats {
    pub total_attempts: u32,
    pub successful: u32,
    pub failed: u32,
    pub success_rate: f32,
}
```

### 6.2 Constraints

- GP MUST handle empty observation case gracefully
- EI calculation MUST return 0 when sigma is 0
- Lambda proposals MUST be in [lambda_min, lambda_max]
- Human escalation MUST trigger after 3 consecutive BO failures
- BO MUST converge or timeout after MAX_BO_ITERATIONS
- All probability calculations MUST handle edge cases (NaN, Inf)

### 6.3 Verification Commands

```bash
# Type check
cargo check -p context-graph-utl

# Unit tests
cargo test -p context-graph-utl meta::escalation

# Specific GP tests
cargo test -p context-graph-utl test_gaussian_process

# Clippy
cargo clippy -p context-graph-utl -- -D warnings
```

---

## 7. Files to Create

| Path | Description |
|------|-------------|
| `crates/context-graph-utl/src/meta/escalation.rs` | Escalation logic |
| `crates/context-graph-utl/src/meta/tests_escalation.rs` | Unit tests |

---

## 8. Files to Modify

| Path | Modification |
|------|--------------|
| `crates/context-graph-utl/src/meta/mod.rs` | Add `pub mod escalation;` |
| `crates/context-graph-utl/src/lib.rs` | Re-export escalation types |

---

## 9. Pseudo-Code

### 9.1 Expected Improvement Calculation

```
FUNCTION expected_improvement(lambda_s: f32) -> f32:
    IF self.observations.is_empty():
        RETURN 0.0

    // Get prediction
    (mu, var) = self.predict(lambda_s)
    sigma = sqrt(var)

    // Handle zero variance
    IF sigma < 1e-8:
        RETURN 0.0

    // Get current best
    f_best = self.best_observation().accuracy

    // Compute improvement
    z = (mu - f_best) / sigma

    // EI = (mu - f_best) * Phi(z) + sigma * phi(z)
    ei = (mu - f_best) * self.standard_normal_cdf(z) + sigma * self.standard_normal_pdf(z)

    RETURN max(ei, 0.0)
```

### 9.2 Bayesian Optimization Loop

```
FUNCTION run_bayesian_optimization(current_weights, evaluate_fn) -> UtlResult<Weights>:
    // Initialize
    self.gp.clear()
    self.status = EscalationStatus::InProgress

    // Generate initial samples (Latin hypercube or random)
    initial_points = self.generate_initial_samples(current_weights)

    // Evaluate initial samples
    FOR lambda_s IN initial_points:
        weights = LifecycleLambdaWeights::new(lambda_s, 1.0 - lambda_s)?
        accuracy = evaluate_fn(weights)
        self.gp.add_observation(GpObservation { lambda_s, accuracy })

    // BO loop
    FOR iteration IN 0..MAX_BO_ITERATIONS:
        // Find next point to evaluate
        next_lambda_s = self.gp.maximize_ei(config.lambda_min, config.lambda_max, 100)

        // Evaluate
        weights = LifecycleLambdaWeights::new(next_lambda_s, 1.0 - next_lambda_s)?
        accuracy = evaluate_fn(weights)
        self.gp.add_observation(GpObservation { next_lambda_s, accuracy })

        // Check convergence
        IF self.has_converged():
            BREAK

    // Return best found
    best = self.gp.best_observation()
    IF best.is_none():
        RETURN Err(UtlError::EscalationFailed)

    best_obs = best.unwrap()
    LifecycleLambdaWeights::new(best_obs.lambda_s, 1.0 - best_obs.lambda_s)
```

### 9.3 Escalation Decision

```
FUNCTION check_escalation_condition(accuracy_history: &MetaAccuracyHistory) -> bool:
    // Check consecutive low count
    IF accuracy_history.consecutive_low_count() >= self.config.escalation_cycle_count:
        RETURN true

    // Check rolling average
    IF accuracy_history.rolling_average() < self.config.escalation_accuracy_threshold:
        IF accuracy_history.len() >= self.config.escalation_cycle_count:
            RETURN true

    RETURN false
```

---

## 10. Validation Criteria

| Criterion | Validation Method |
|-----------|-------------------|
| GP predicts correctly | Test known function |
| EI is non-negative | Property test |
| BO finds better point | Test with synthetic objective |
| Escalation triggers at 10 cycles | Unit test |
| Human escalation at 3 failures | Unit test |
| Edge cases handled | NaN/empty tests |

---

## 11. Test Cases

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gp_prediction_no_observations() {
        let gp = SimpleGaussianProcess::new();
        let (mu, var) = gp.predict(0.5);
        assert!(var > 0.0, "Variance should be positive with no observations");
    }

    #[test]
    fn test_gp_prediction_with_observations() {
        let mut gp = SimpleGaussianProcess::new();
        gp.add_observation(GpObservation { lambda_s: 0.3, accuracy: 0.6 });
        gp.add_observation(GpObservation { lambda_s: 0.7, accuracy: 0.8 });

        // Prediction near observation should be close to observed value
        let (mu, var) = gp.predict(0.3);
        assert!((mu - 0.6).abs() < 0.1);
        assert!(var < 0.1, "Variance should be low near observation");
    }

    #[test]
    fn test_ei_is_non_negative() {
        let mut gp = SimpleGaussianProcess::new();
        gp.add_observation(GpObservation { lambda_s: 0.5, accuracy: 0.7 });

        for lambda in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let ei = gp.expected_improvement(lambda);
            assert!(ei >= 0.0, "EI should be non-negative at {}", lambda);
        }
    }

    #[test]
    fn test_maximize_ei_in_bounds() {
        let mut gp = SimpleGaussianProcess::new();
        gp.add_observation(GpObservation { lambda_s: 0.5, accuracy: 0.6 });
        gp.add_observation(GpObservation { lambda_s: 0.7, accuracy: 0.8 });

        let best_lambda = gp.maximize_ei(0.1, 0.9, 100);
        assert!(best_lambda >= 0.1 && best_lambda <= 0.9);
    }

    #[test]
    fn test_escalation_triggers_at_threshold() {
        let config = SelfCorrectionConfig::default();
        let manager = EscalationManager::new(config.clone());

        let mut history = MetaAccuracyHistory::with_defaults();

        // Add accuracy values below threshold
        for _ in 0..(config.escalation_cycle_count as usize) {
            history.record(0.5);
        }

        assert!(manager.check_escalation_condition(&history));
    }

    #[test]
    fn test_no_escalation_above_threshold() {
        let config = SelfCorrectionConfig::default();
        let manager = EscalationManager::new(config);

        let mut history = MetaAccuracyHistory::with_defaults();

        // Add accuracy values above threshold
        for _ in 0..15 {
            history.record(0.8);
        }

        assert!(!manager.check_escalation_condition(&history));
    }

    #[test]
    fn test_human_escalation_after_failures() {
        let config = SelfCorrectionConfig::default();
        let mut manager = EscalationManager::new(config);

        // Simulate 3 failed escalations
        for _ in 0..HUMAN_ESCALATION_THRESHOLD {
            manager.record_outcome(false);
        }

        assert!(manager.needs_human_review());
    }

    #[test]
    fn test_bo_finds_better_lambda() {
        let config = SelfCorrectionConfig::default();
        let mut manager = EscalationManager::new(config);

        // Synthetic objective: accuracy = 1.0 - (lambda_s - 0.6)^2
        // Best at lambda_s = 0.6
        let mut evaluate = |weights: LifecycleLambdaWeights| {
            let lambda_s = weights.lambda_s();
            1.0 - (lambda_s - 0.6).powi(2)
        };

        let current = LifecycleLambdaWeights::for_stage(crate::lifecycle::LifecycleStage::Growth);
        let result = manager.run_bayesian_optimization(current, &mut evaluate);

        assert!(result.is_ok());
        let best = result.unwrap();
        // Should find lambda_s close to 0.6
        assert!((best.lambda_s() - 0.6).abs() < 0.15);
    }

    #[test]
    fn test_escalation_stats() {
        let config = SelfCorrectionConfig::default();
        let mut manager = EscalationManager::new(config);

        manager.record_outcome(true);
        manager.record_outcome(true);
        manager.record_outcome(false);

        let stats = manager.stats();
        assert_eq!(stats.total_attempts, 3);
        assert_eq!(stats.successful, 2);
        assert_eq!(stats.failed, 1);
        assert!((stats.success_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_rbf_kernel() {
        let gp = SimpleGaussianProcess::new();

        // Kernel at same point should be signal_var
        let k_same = gp.rbf_kernel(0.5, 0.5);
        assert!((k_same - gp.signal_var).abs() < 0.001);

        // Kernel decays with distance
        let k_close = gp.rbf_kernel(0.5, 0.6);
        let k_far = gp.rbf_kernel(0.5, 0.9);
        assert!(k_close > k_far);
    }
}
```

---

## 12. Rollback Plan

If this task fails validation:

1. Revert `crates/context-graph-utl/src/meta/escalation.rs`
2. Remove mod declaration
3. TASK-002 changes remain unaffected
4. Document failure in task notes

---

## 13. Notes

- The GP implementation is simplified (1D only) for lambda_s optimization
- lambda_c is computed as 1.0 - lambda_s to maintain sum invariant
- Full multi-dimensional BO with botorch/gpytorch is Phase 2
- Human escalation provides safety valve for pathological cases
- Initial samples use Latin hypercube-like spacing for coverage

---

**Task History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | ContextGraph Team | Initial task specification |
