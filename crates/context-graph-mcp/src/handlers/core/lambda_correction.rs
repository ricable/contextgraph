//! Lambda weight self-correction algorithm.
//!
//! TASK-METAUTL-P0-002: Implements adaptive lambda_s/lambda_c weights with
//! self-correction capability based on prediction errors.
//!
//! # Architecture
//!
//! This module provides the `AdaptiveLambdaWeights` struct which wraps
//! `LifecycleLambdaWeights` from the UTL crate with dynamic adjustment
//! capability. This is DISTINCT from per-embedder weights in `MetaUtlTracker`.
//!
//! # Algorithm
//!
//! 1. Detect when prediction error exceeds threshold (0.2)
//! 2. Compute adjustment delta using ACh-modulated learning rate
//! 3. Apply adjustment while maintaining sum-to-one invariant
//! 4. Clamp values to valid bounds [0.05, 0.9] per NORTH-016
//!
//! # Constitution Reference
//!
//! - REQ-METAUTL-003: Adjust lambda_s/lambda_c when prediction_error > 0.2
//! - REQ-METAUTL-004: Formula: lambda_new = lambda_old + alpha * delta
//! - REQ-METAUTL-005: Alpha modulated by current ACh level
//! - REQ-METAUTL-006: Lambda weights SHALL always sum to 1.0
//! - REQ-METAUTL-007: Lambda weights SHALL be clamped to [0.05, 0.9]

// Allow dead_code until integration in TASK-METAUTL-P0-005/006
#![allow(dead_code)]

use context_graph_utl::error::{UtlError, UtlResult};
use context_graph_utl::lifecycle::LifecycleLambdaWeights;

use super::types::{LambdaAdjustment, SelfCorrectionConfig};

// ============================================================================
// Constants
// ============================================================================

/// Acetylcholine baseline level for normalization.
/// SPEC-METAUTL-001: Minimum ACh level used as baseline.
pub const ACH_BASELINE: f32 = 0.001;

/// Acetylcholine maximum level.
/// SPEC-METAUTL-001: Maximum ACh level for modulation ceiling.
pub const ACH_MAX: f32 = 0.002;

/// Minimum alpha (learning rate) value.
/// Prevents adjustments from being too small.
const ALPHA_MIN: f32 = 0.01;

/// Maximum alpha (learning rate) value.
/// Prevents adjustments from being too large.
const ALPHA_MAX: f32 = 0.1;

/// Size of rolling accuracy history buffer.
const ACCURACY_HISTORY_SIZE: usize = 100;

/// Epsilon for floating-point comparison.
const EPSILON: f32 = 0.001;

// ============================================================================
// SelfCorrectingLambda Trait
// ============================================================================

/// Trait for self-correcting lambda weights.
///
/// Provides the interface for lambda weight adjustment based on prediction
/// errors, with ACh-modulated learning rates.
pub trait SelfCorrectingLambda {
    /// Adjust lambda weights based on prediction error.
    ///
    /// # Arguments
    ///
    /// - `prediction_error`: Difference between predicted and actual (L_pred - L_actual)
    /// - `ach_level`: Current acetylcholine level [0.001, 0.002]
    ///
    /// # Returns
    ///
    /// - `Some(LambdaAdjustment)` if adjustment was made
    /// - `None` if error below threshold or invalid input
    ///
    /// # Behavior
    ///
    /// - Positive error (over-predicted) reduces lambda_s
    /// - Negative error (under-predicted) increases lambda_s
    /// - Higher ACh = higher learning rate (faster adaptation)
    fn adjust_lambdas(&mut self, prediction_error: f32, ach_level: f32) -> Option<LambdaAdjustment>;

    /// Get current corrected lambda weights.
    fn corrected_weights(&self) -> LifecycleLambdaWeights;

    /// Get base (lifecycle) lambda weights.
    fn base_weights(&self) -> LifecycleLambdaWeights;

    /// Reset to base weights.
    fn reset_to_base(&mut self);

    /// Record accuracy for tracking.
    fn record_accuracy(&mut self, accuracy: f32);

    /// Get rolling accuracy average.
    fn rolling_accuracy(&self) -> f32;
}

// ============================================================================
// AdaptiveLambdaWeights Struct
// ============================================================================

/// Adaptive lambda weights with self-correction capability.
///
/// Wraps `LifecycleLambdaWeights` with dynamic adjustment capability.
/// Maintains the sum-to-one invariant and respects NORTH-016 bounds.
#[derive(Debug, Clone)]
pub struct AdaptiveLambdaWeights {
    /// Base weights from lifecycle stage
    base_weights: LifecycleLambdaWeights,
    /// Current corrected weights
    current_weights: LifecycleLambdaWeights,
    /// Configuration
    config: SelfCorrectionConfig,
    /// Rolling accuracy history buffer
    accuracy_history: [f32; ACCURACY_HISTORY_SIZE],
    /// Current index in accuracy history (circular buffer)
    accuracy_index: usize,
    /// Number of accuracy values recorded (up to ACCURACY_HISTORY_SIZE)
    accuracy_count: usize,
    /// Total adjustments made
    adjustment_count: u64,
    /// Last adjustment applied
    last_adjustment: Option<LambdaAdjustment>,
}

impl AdaptiveLambdaWeights {
    /// Create new adaptive weights from lifecycle weights.
    ///
    /// # Arguments
    ///
    /// - `base_weights`: Base weights from lifecycle stage
    /// - `config`: Self-correction configuration
    pub fn new(base_weights: LifecycleLambdaWeights, config: SelfCorrectionConfig) -> Self {
        Self {
            base_weights,
            current_weights: base_weights,
            config,
            accuracy_history: [0.0; ACCURACY_HISTORY_SIZE],
            accuracy_index: 0,
            accuracy_count: 0,
            adjustment_count: 0,
            last_adjustment: None,
        }
    }

    /// Create with default configuration.
    ///
    /// # Arguments
    ///
    /// - `base_weights`: Base weights from lifecycle stage
    pub fn with_defaults(base_weights: LifecycleLambdaWeights) -> Self {
        Self::new(base_weights, SelfCorrectionConfig::default())
    }

    /// Compute learning rate modulated by ACh level.
    ///
    /// # Formula
    ///
    /// ```text
    /// ach_normalized = (ach_level - ACH_BASELINE) / (ACH_MAX - ACH_BASELINE)
    /// alpha = base_alpha * (1.0 + ach_normalized)
    /// ```
    ///
    /// Result is clamped to [0.01, 0.1].
    ///
    /// # Arguments
    ///
    /// - `ach_level`: Current acetylcholine level
    ///
    /// # Returns
    ///
    /// Learning rate alpha in range [ALPHA_MIN, ALPHA_MAX]
    pub fn compute_alpha(&self, ach_level: f32) -> f32 {
        // Clamp ACh to valid range
        let ach_clamped = ach_level.clamp(ACH_BASELINE, ACH_MAX);

        // Normalize ACh to [0, 1] range
        let ach_normalized = (ach_clamped - ACH_BASELINE) / (ACH_MAX - ACH_BASELINE);

        // Compute modulated alpha
        // Higher ACh = higher learning rate (faster adaptation)
        let alpha = self.config.base_alpha * (ach_normalized + 1.0);

        // Clamp to valid range
        alpha.clamp(ALPHA_MIN, ALPHA_MAX)
    }

    /// Compute raw adjustment deltas.
    ///
    /// # Formula
    ///
    /// ```text
    /// delta_s = -alpha * prediction_error
    /// delta_c = -delta_s  // maintain sum invariant
    /// ```
    ///
    /// # Arguments
    ///
    /// - `prediction_error`: Difference between predicted and actual
    /// - `alpha`: Learning rate
    ///
    /// # Returns
    ///
    /// Tuple of (delta_s, delta_c)
    pub fn compute_raw_adjustment(&self, prediction_error: f32, alpha: f32) -> (f32, f32) {
        // Negative sign: positive error (over-predicted) should reduce lambda_s
        let delta_s = -alpha * prediction_error;
        let delta_c = -delta_s; // Maintain sum invariant

        (delta_s, delta_c)
    }

    /// Apply bounds and normalize.
    ///
    /// # Algorithm
    ///
    /// 1. Add deltas to current weights
    /// 2. Clamp to [lambda_min, lambda_max]
    /// 3. Renormalize to sum=1.0
    /// 4. Re-clamp if normalization pushed values out of bounds
    /// 5. Final normalization to ensure sum=1.0
    ///
    /// # Arguments
    ///
    /// - `delta_s`: Change in lambda_s
    /// - `delta_c`: Change in lambda_c
    ///
    /// # Returns
    ///
    /// New `LifecycleLambdaWeights` or error if invariants cannot be satisfied
    pub fn apply_and_normalize(&self, delta_s: f32, delta_c: f32) -> UtlResult<LifecycleLambdaWeights> {
        // Compute raw new values
        let mut new_s = self.current_weights.lambda_s() + delta_s;
        let mut new_c = self.current_weights.lambda_c() + delta_c;

        // Clamp to bounds
        new_s = new_s.clamp(self.config.min_weight, self.config.max_weight);
        new_c = new_c.clamp(self.config.min_weight, self.config.max_weight);

        // Normalize to sum=1.0
        let sum = new_s + new_c;
        if sum > EPSILON {
            new_s /= sum;
            new_c /= sum;
        } else {
            // Should not happen with valid bounds, but handle gracefully
            return Err(UtlError::InvalidLambdaWeights {
                novelty: new_s,
                consolidation: new_c,
                reason: "Lambda sum too close to zero".to_string(),
            });
        }

        // Final bounds check after normalization (may have shifted)
        new_s = new_s.clamp(self.config.min_weight, self.config.max_weight);
        new_c = new_c.clamp(self.config.min_weight, self.config.max_weight);

        // Re-normalize if clamping affected sum
        let sum = new_s + new_c;
        if (sum - 1.0).abs() > EPSILON {
            new_s /= sum;
            new_c /= sum;
        }

        // Create and validate new weights
        LifecycleLambdaWeights::new(new_s, new_c)
    }

    /// Validate adjustment doesn't violate invariants.
    ///
    /// # Invariants
    ///
    /// - Sum == 1.0 (within epsilon)
    /// - lambda_s in [min, max]
    /// - lambda_c in [min, max]
    ///
    /// # Arguments
    ///
    /// - `new_weights`: Weights to validate
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, `Err(UtlError)` otherwise
    pub fn validate_adjustment(&self, new_weights: &LifecycleLambdaWeights) -> UtlResult<()> {
        validate_lambda_invariants(
            new_weights.lambda_s(),
            new_weights.lambda_c(),
            &self.config,
        )
    }

    /// Get adjustment count.
    pub fn adjustment_count(&self) -> u64 {
        self.adjustment_count
    }

    /// Get last adjustment.
    pub fn last_adjustment(&self) -> Option<LambdaAdjustment> {
        self.last_adjustment
    }

    /// Check if currently at base weights.
    ///
    /// Returns true if current weights match base weights within epsilon.
    pub fn is_at_base(&self) -> bool {
        let (dev_s, dev_c) = self.deviation_from_base();
        dev_s.abs() < EPSILON && dev_c.abs() < EPSILON
    }

    /// Get deviation from base weights.
    ///
    /// Returns tuple of (lambda_s deviation, lambda_c deviation).
    pub fn deviation_from_base(&self) -> (f32, f32) {
        (
            self.current_weights.lambda_s() - self.base_weights.lambda_s(),
            self.current_weights.lambda_c() - self.base_weights.lambda_c(),
        )
    }
}

impl SelfCorrectingLambda for AdaptiveLambdaWeights {
    fn adjust_lambdas(&mut self, prediction_error: f32, ach_level: f32) -> Option<LambdaAdjustment> {
        // Validate inputs - reject NaN and Infinity
        if prediction_error.is_nan() || prediction_error.is_infinite() {
            tracing::warn!(
                "Invalid prediction_error in adjust_lambdas: {}",
                prediction_error
            );
            return None;
        }

        // Check threshold - no adjustment if error below threshold
        if prediction_error.abs() <= self.config.error_threshold {
            return None;
        }

        // Compute modulated learning rate
        let alpha = self.compute_alpha(ach_level);

        // Compute raw deltas
        let (delta_s, delta_c) = self.compute_raw_adjustment(prediction_error, alpha);

        // Apply bounds and normalize
        let new_weights = match self.apply_and_normalize(delta_s, delta_c) {
            Ok(w) => w,
            Err(e) => {
                tracing::warn!("Failed to apply lambda adjustment: {}", e);
                return None;
            }
        };

        // Validate invariants
        if let Err(e) = self.validate_adjustment(&new_weights) {
            tracing::warn!("Lambda adjustment validation failed: {}", e);
            return None;
        }

        // Store old weights for adjustment record
        let old_s = self.current_weights.lambda_s();
        let old_c = self.current_weights.lambda_c();

        // Apply new weights
        self.current_weights = new_weights;
        self.adjustment_count += 1;

        // Create adjustment record
        let adjustment = LambdaAdjustment {
            delta_lambda_s: new_weights.lambda_s() - old_s,
            delta_lambda_c: new_weights.lambda_c() - old_c,
            alpha,
            trigger_error: prediction_error,
        };
        self.last_adjustment = Some(adjustment);

        Some(adjustment)
    }

    fn corrected_weights(&self) -> LifecycleLambdaWeights {
        self.current_weights
    }

    fn base_weights(&self) -> LifecycleLambdaWeights {
        self.base_weights
    }

    fn reset_to_base(&mut self) {
        self.current_weights = self.base_weights;
        self.last_adjustment = None;
    }

    fn record_accuracy(&mut self, accuracy: f32) {
        // Clamp to valid range
        let accuracy = accuracy.clamp(0.0, 1.0);

        // Store in circular buffer
        self.accuracy_history[self.accuracy_index] = accuracy;
        self.accuracy_index = (self.accuracy_index + 1) % ACCURACY_HISTORY_SIZE;

        if self.accuracy_count < ACCURACY_HISTORY_SIZE {
            self.accuracy_count += 1;
        }
    }

    fn rolling_accuracy(&self) -> f32 {
        if self.accuracy_count == 0 {
            return 0.5; // Default if no data
        }

        let sum: f32 = self.accuracy_history[..self.accuracy_count].iter().sum();
        sum / self.accuracy_count as f32
    }
}

impl Default for AdaptiveLambdaWeights {
    fn default() -> Self {
        Self::with_defaults(LifecycleLambdaWeights::default())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Validate that weights satisfy all invariants.
///
/// # Invariants
///
/// - sum == 1.0 (within epsilon)
/// - lambda_s in [min, max]
/// - lambda_c in [min, max]
///
/// # Arguments
///
/// - `lambda_s`: Surprise weight
/// - `lambda_c`: Coherence weight
/// - `config`: Configuration with bounds
///
/// # Returns
///
/// `Ok(())` if valid, `Err(UtlError)` otherwise
pub fn validate_lambda_invariants(
    lambda_s: f32,
    lambda_c: f32,
    config: &SelfCorrectionConfig,
) -> UtlResult<()> {
    // Check for NaN/Inf
    if lambda_s.is_nan() || lambda_s.is_infinite() {
        return Err(UtlError::InvalidLambdaWeights {
            novelty: lambda_s,
            consolidation: lambda_c,
            reason: "lambda_s is NaN or infinite".to_string(),
        });
    }
    if lambda_c.is_nan() || lambda_c.is_infinite() {
        return Err(UtlError::InvalidLambdaWeights {
            novelty: lambda_s,
            consolidation: lambda_c,
            reason: "lambda_c is NaN or infinite".to_string(),
        });
    }

    // Check sum invariant
    let sum = lambda_s + lambda_c;
    if (sum - 1.0).abs() > EPSILON {
        return Err(UtlError::lambda_sum_error(lambda_s, lambda_c));
    }

    // Check bounds for lambda_s
    if lambda_s < config.min_weight || lambda_s > config.max_weight {
        return Err(UtlError::InvalidLambdaWeights {
            novelty: lambda_s,
            consolidation: lambda_c,
            reason: format!(
                "lambda_s {} out of bounds [{}, {}]",
                lambda_s, config.min_weight, config.max_weight
            ),
        });
    }

    // Check bounds for lambda_c
    if lambda_c < config.min_weight || lambda_c > config.max_weight {
        return Err(UtlError::InvalidLambdaWeights {
            novelty: lambda_s,
            consolidation: lambda_c,
            reason: format!(
                "lambda_c {} out of bounds [{}, {}]",
                lambda_c, config.min_weight, config.max_weight
            ),
        });
    }

    Ok(())
}

/// Normalize weights to sum to 1.0.
///
/// # Arguments
///
/// - `lambda_s`: Surprise weight (may not sum to 1.0)
/// - `lambda_c`: Coherence weight (may not sum to 1.0)
///
/// # Returns
///
/// Normalized (lambda_s, lambda_c) tuple that sums to 1.0.
/// Returns (0.5, 0.5) if input sum is too close to zero.
pub fn normalize_lambdas(lambda_s: f32, lambda_c: f32) -> (f32, f32) {
    let sum = lambda_s + lambda_c;
    if sum.abs() < EPSILON {
        // Handle edge case of zero sum
        return (0.5, 0.5);
    }
    (lambda_s / sum, lambda_c / sum)
}

/// Clamp lambdas to valid bounds and renormalize.
///
/// # Algorithm
///
/// The algorithm ensures both sum=1.0 and bounds [min, max] are satisfied.
/// This may require iterative adjustment when bounds are tight.
///
/// 1. Normalize to sum=1.0 first
/// 2. Clamp to bounds
/// 3. If sum != 1.0 after clamping, redistribute the difference
/// 4. Iterate until convergence
///
/// # Arguments
///
/// - `lambda_s`: Surprise weight
/// - `lambda_c`: Coherence weight
/// - `min`: Minimum bound (e.g., 0.05)
/// - `max`: Maximum bound (e.g., 0.9)
///
/// # Returns
///
/// Clamped and normalized (lambda_s, lambda_c) tuple.
/// Both values will be in [min, max] and sum to 1.0.
pub fn clamp_and_normalize(lambda_s: f32, lambda_c: f32, min: f32, max: f32) -> (f32, f32) {
    // Handle edge cases
    if lambda_s.is_nan() || lambda_c.is_nan() {
        return (0.5, 0.5);
    }

    // First normalize to sum=1.0
    let (mut s, mut c) = normalize_lambdas(lambda_s, lambda_c);

    // Iteratively clamp and redistribute (converges quickly)
    for _ in 0..10 {
        // Clamp to bounds
        let s_clamped = s.clamp(min, max);
        let c_clamped = c.clamp(min, max);

        // Check if sum=1.0 is satisfied
        let sum = s_clamped + c_clamped;
        if (sum - 1.0).abs() <= EPSILON {
            return (s_clamped, c_clamped);
        }

        // If not, we need to redistribute
        // Calculate how much we need to add/subtract
        let diff = 1.0 - sum;

        // Determine which value has room to adjust
        let s_room = if diff > 0.0 {
            max - s_clamped // Room to increase
        } else {
            s_clamped - min // Room to decrease
        };
        let c_room = if diff > 0.0 {
            max - c_clamped // Room to increase
        } else {
            c_clamped - min // Room to decrease
        };

        // Distribute proportionally based on available room
        let total_room = s_room + c_room;
        if total_room > EPSILON {
            s = s_clamped + diff * (s_room / total_room);
            c = c_clamped + diff * (c_room / total_room);
        } else {
            // No room to adjust - return best effort
            // This happens when bounds are too tight (min + max < 1.0)
            // Return values that sum to 1.0, sacrificing bounds if necessary
            return (s_clamped / sum, c_clamped / sum);
        }
    }

    // Final clamp (should be within bounds after iterations)
    let s_final = s.clamp(min, max);
    let c_final = c.clamp(min, max);
    let sum = s_final + c_final;
    if sum > EPSILON {
        (s_final / sum, c_final / sum)
    } else {
        (0.5, 0.5)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use context_graph_utl::lifecycle::LifecycleStage;

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
        assert!(adj.delta_lambda_s < 0.0, "Positive error should reduce lambda_s");
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
            assert!(
                (sum - 1.0).abs() < EPSILON,
                "Sum invariant violated at iteration {}: sum={}",
                i,
                sum
            );
        }
    }

    #[test]
    fn test_bounds_respected() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Infancy); // 0.7, 0.3
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);

        // Try to push lambda_s above max with many negative error corrections
        for _ in 0..50 {
            adaptive.adjust_lambdas(-0.5, ACH_MAX); // Negative error increases lambda_s
        }

        let weights = adaptive.corrected_weights();
        let config = SelfCorrectionConfig::default();
        assert!(
            weights.lambda_s() <= config.max_weight,
            "lambda_s exceeded max: {}",
            weights.lambda_s()
        );
        assert!(
            weights.lambda_c() >= config.min_weight,
            "lambda_c below min: {}",
            weights.lambda_c()
        );
    }

    #[test]
    fn test_ach_modulation() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let adaptive = AdaptiveLambdaWeights::with_defaults(base);

        let alpha_low = adaptive.compute_alpha(ACH_BASELINE);
        let alpha_high = adaptive.compute_alpha(ACH_MAX);

        assert!(
            alpha_high > alpha_low,
            "Higher ACh should give higher alpha: alpha_low={}, alpha_high={}",
            alpha_low,
            alpha_high
        );
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
        assert!(result.is_none(), "NaN input should return None");

        let result = adaptive.adjust_lambdas(f32::INFINITY, ACH_BASELINE);
        assert!(result.is_none(), "Infinity input should return None");

        let result = adaptive.adjust_lambdas(f32::NEG_INFINITY, ACH_BASELINE);
        assert!(result.is_none(), "Negative infinity input should return None");
    }

    #[test]
    fn test_adjustment_direction() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);

        // Positive error (over-predicted surprise)
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);
        let adj = adaptive.adjust_lambdas(0.3, ACH_BASELINE).unwrap();
        assert!(
            adj.delta_lambda_s < 0.0,
            "Positive error should reduce lambda_s"
        );

        // Negative error (under-predicted surprise)
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);
        let adj = adaptive.adjust_lambdas(-0.3, ACH_BASELINE).unwrap();
        assert!(
            adj.delta_lambda_s > 0.0,
            "Negative error should increase lambda_s"
        );
    }

    #[test]
    fn test_clamp_and_normalize() {
        let config = SelfCorrectionConfig::default();

        // Test extreme values
        let (s, c) = clamp_and_normalize(2.0, -1.0, config.min_weight, config.max_weight);
        assert!(
            s >= config.min_weight && s <= config.max_weight,
            "lambda_s out of bounds: {}",
            s
        );
        assert!(
            c >= config.min_weight && c <= config.max_weight,
            "lambda_c out of bounds: {}",
            c
        );
        assert!(
            (s + c - 1.0).abs() < EPSILON,
            "Sum not 1.0: {}",
            s + c
        );
    }

    #[test]
    fn test_accuracy_tracking() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);

        adaptive.record_accuracy(0.8);
        adaptive.record_accuracy(0.9);
        adaptive.record_accuracy(0.7);

        let avg = adaptive.rolling_accuracy();
        assert!(
            (avg - 0.8).abs() < 0.001,
            "Expected avg 0.8, got {}",
            avg
        );
    }

    #[test]
    fn test_validate_lambda_invariants() {
        let config = SelfCorrectionConfig::default();

        // Valid weights
        assert!(validate_lambda_invariants(0.5, 0.5, &config).is_ok());
        assert!(validate_lambda_invariants(0.7, 0.3, &config).is_ok());
        assert!(validate_lambda_invariants(0.3, 0.7, &config).is_ok());

        // Invalid sum
        assert!(validate_lambda_invariants(0.6, 0.6, &config).is_err());

        // Out of bounds
        assert!(validate_lambda_invariants(0.95, 0.05, &config).is_err()); // 0.95 > max
        assert!(validate_lambda_invariants(0.02, 0.98, &config).is_err()); // 0.02 < min

        // NaN
        assert!(validate_lambda_invariants(f32::NAN, 0.5, &config).is_err());
    }

    #[test]
    fn test_normalize_lambdas() {
        let (s, c) = normalize_lambdas(0.6, 0.4);
        assert!((s + c - 1.0).abs() < EPSILON);
        assert!((s - 0.6).abs() < EPSILON);

        let (s, c) = normalize_lambdas(3.0, 7.0);
        assert!((s + c - 1.0).abs() < EPSILON);
        assert!((s - 0.3).abs() < EPSILON);

        // Edge case: zero sum
        let (s, c) = normalize_lambdas(0.0, 0.0);
        assert!((s - 0.5).abs() < EPSILON);
        assert!((c - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_alpha_bounds() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let adaptive = AdaptiveLambdaWeights::with_defaults(base);

        // Test with ACh below baseline
        let alpha = adaptive.compute_alpha(0.0);
        assert!(alpha >= ALPHA_MIN, "Alpha below min: {}", alpha);
        assert!(alpha <= ALPHA_MAX, "Alpha above max: {}", alpha);

        // Test with ACh above max
        let alpha = adaptive.compute_alpha(1.0);
        assert!(alpha >= ALPHA_MIN, "Alpha below min: {}", alpha);
        assert!(alpha <= ALPHA_MAX, "Alpha above max: {}", alpha);
    }

    #[test]
    fn test_adjustment_count() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);

        assert_eq!(adaptive.adjustment_count(), 0);

        adaptive.adjust_lambdas(0.3, ACH_BASELINE);
        assert_eq!(adaptive.adjustment_count(), 1);

        adaptive.adjust_lambdas(0.3, ACH_BASELINE);
        assert_eq!(adaptive.adjustment_count(), 2);

        // Below threshold, no adjustment
        adaptive.adjust_lambdas(0.1, ACH_BASELINE);
        assert_eq!(adaptive.adjustment_count(), 2);
    }

    #[test]
    fn test_last_adjustment() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);

        assert!(adaptive.last_adjustment().is_none());

        adaptive.adjust_lambdas(0.3, ACH_BASELINE);
        let adj = adaptive.last_adjustment().unwrap();
        assert!((adj.trigger_error - 0.3).abs() < EPSILON);
    }

    #[test]
    fn test_deviation_from_base() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);

        let (dev_s, dev_c) = adaptive.deviation_from_base();
        assert!(dev_s.abs() < EPSILON);
        assert!(dev_c.abs() < EPSILON);

        adaptive.adjust_lambdas(0.3, ACH_BASELINE);
        let (dev_s, dev_c) = adaptive.deviation_from_base();
        // Should have some deviation now
        assert!(dev_s != 0.0 || dev_c != 0.0);
    }

    #[test]
    fn test_default_impl() {
        let adaptive = AdaptiveLambdaWeights::default();
        // Default should be Infancy stage weights
        assert!((adaptive.base_weights().lambda_s() - 0.7).abs() < EPSILON);
        assert!((adaptive.base_weights().lambda_c() - 0.3).abs() < EPSILON);
    }

    #[test]
    fn test_rolling_accuracy_empty() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let adaptive = AdaptiveLambdaWeights::with_defaults(base);

        // Default value when no data recorded
        assert!((adaptive.rolling_accuracy() - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_rolling_accuracy_overflow() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);

        // Fill beyond buffer size
        for i in 0..150 {
            adaptive.record_accuracy((i % 10) as f32 / 10.0);
        }

        // Should still work correctly (circular buffer)
        let avg = adaptive.rolling_accuracy();
        assert!(avg >= 0.0 && avg <= 1.0);
    }

    #[test]
    fn test_fsv_lambda_adjustment_evidence() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);

        // BEFORE state
        println!(
            "BEFORE: lambda_s={}, lambda_c={}",
            adaptive.corrected_weights().lambda_s(),
            adaptive.corrected_weights().lambda_c()
        );

        // Execute
        let adj = adaptive.adjust_lambdas(0.35, ACH_MAX);

        // AFTER state
        println!(
            "AFTER: lambda_s={}, lambda_c={}",
            adaptive.corrected_weights().lambda_s(),
            adaptive.corrected_weights().lambda_c()
        );
        println!("ADJUSTMENT: delta_s={:?}", adj.map(|a| a.delta_lambda_s));

        // FSV assertions
        let current = adaptive.corrected_weights();
        let sum = current.lambda_s() + current.lambda_c();
        assert!(
            (sum - 1.0).abs() < EPSILON,
            "FSV FAIL: sum invariant violated"
        );
        assert!(
            current.lambda_s() >= 0.05,
            "FSV FAIL: lambda_s below min"
        );
        assert!(
            current.lambda_s() <= 0.9,
            "FSV FAIL: lambda_s above max"
        );
    }

    #[test]
    fn test_edge_case_max_bound_try_increase() {
        // EC-002: Lambda_s at max, try to increase
        // Start with weights that will push lambda_s near max
        let base = LifecycleLambdaWeights::new(0.85, 0.15).unwrap();
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);

        // Keep trying to increase lambda_s with negative errors
        for _ in 0..100 {
            adaptive.adjust_lambdas(-0.9, ACH_MAX);
        }

        let weights = adaptive.corrected_weights();
        let config = SelfCorrectionConfig::default();

        // Verify bounds are respected
        assert!(
            weights.lambda_s() <= config.max_weight,
            "lambda_s should not exceed max: {}",
            weights.lambda_s()
        );
        assert!(
            weights.lambda_c() >= config.min_weight,
            "lambda_c should not go below min: {}",
            weights.lambda_c()
        );

        // Verify sum invariant
        let sum = weights.lambda_s() + weights.lambda_c();
        assert!(
            (sum - 1.0).abs() < EPSILON,
            "Sum invariant violated: {}",
            sum
        );
    }

    #[test]
    fn test_edge_case_extreme_error_high_ach() {
        // EC-003: Extreme negative error with high ACh
        let base = LifecycleLambdaWeights::new(0.5, 0.5).unwrap();
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);

        // Apply extreme negative error with max ACh
        adaptive.adjust_lambdas(-0.9, ACH_MAX);

        let weights = adaptive.corrected_weights();
        let config = SelfCorrectionConfig::default();

        // Should not exceed max even with extreme values
        assert!(
            weights.lambda_s() <= config.max_weight,
            "lambda_s should not exceed max: {}",
            weights.lambda_s()
        );

        // Sum should still be 1.0
        let sum = weights.lambda_s() + weights.lambda_c();
        assert!(
            (sum - 1.0).abs() < EPSILON,
            "Sum invariant violated: {}",
            sum
        );
    }

    #[test]
    fn test_exact_threshold_boundary() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let mut adaptive = AdaptiveLambdaWeights::with_defaults(base);

        // Exactly at threshold - should NOT adjust
        let result = adaptive.adjust_lambdas(0.2, ACH_BASELINE);
        assert!(result.is_none(), "Exact threshold should not trigger adjustment");

        // Just above threshold - should adjust
        let result = adaptive.adjust_lambdas(0.201, ACH_BASELINE);
        assert!(result.is_some(), "Just above threshold should trigger adjustment");
    }

    #[test]
    fn test_compute_raw_adjustment_symmetry() {
        let base = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let adaptive = AdaptiveLambdaWeights::with_defaults(base);

        let alpha = 0.05;
        let (delta_s, delta_c) = adaptive.compute_raw_adjustment(0.3, alpha);

        // delta_c should be exactly negative of delta_s
        assert!(
            (delta_s + delta_c).abs() < f32::EPSILON,
            "Deltas should sum to zero: {} + {} = {}",
            delta_s,
            delta_c,
            delta_s + delta_c
        );
    }
}
