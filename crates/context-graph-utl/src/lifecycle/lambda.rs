//! Lambda weight computation for lifecycle stages.
//!
//! Implements the Marblestone lambda weights that balance surprise (novelty)
//! and coherence (consolidation) at each lifecycle stage. Lambda weights
//! must always sum to 1.0.
//!
//! # Constitution Reference
//!
//! ```text
//! Infancy:  lambda_s = 0.7, lambda_c = 0.3 (favor novelty)
//! Growth:   lambda_s = 0.5, lambda_c = 0.5 (balanced)
//! Maturity: lambda_s = 0.3, lambda_c = 0.7 (favor coherence)
//! ```
//!
//! # Weight Semantics
//!
//! - `lambda_s` (surprise weight): Higher values prioritize novel information
//! - `lambda_c` (coherence weight): Higher values prioritize knowledge consolidation

use serde::{Deserialize, Serialize};

use crate::config::LifecycleConfig;
use crate::error::{UtlError, UtlResult};

use super::stage::LifecycleStage;

/// Lambda weights for balancing surprise and coherence.
///
/// These weights determine how the UTL formula balances novelty capture
/// against knowledge consolidation. The weights must always sum to 1.0.
///
/// # Invariant
///
/// `lambda_s + lambda_c == 1.0`
///
/// # Example
///
/// ```
/// use context_graph_utl::lifecycle::{LifecycleStage, LifecycleLambdaWeights};
///
/// // Get weights for Growth stage
/// let weights = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
/// assert!((weights.lambda_s() - 0.5).abs() < 0.001);
/// assert!((weights.lambda_c() - 0.5).abs() < 0.001);
///
/// // Verify invariant
/// assert!((weights.lambda_s() + weights.lambda_c() - 1.0).abs() < 0.001);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LifecycleLambdaWeights {
    /// Lambda weight for surprise/novelty (lambda_s).
    lambda_s: f32,

    /// Lambda weight for coherence/consolidation (lambda_c).
    lambda_c: f32,
}

impl LifecycleLambdaWeights {
    /// Tolerance for floating-point comparisons when validating weight sum.
    pub const EPSILON: f32 = 0.001;

    /// Create new lambda weights with validation.
    ///
    /// # Arguments
    ///
    /// * `lambda_s` - Surprise weight in range [0.0, 1.0]
    /// * `lambda_c` - Coherence weight in range [0.0, 1.0]
    ///
    /// # Returns
    ///
    /// `Ok(LifecycleLambdaWeights)` if weights are valid and sum to 1.0,
    /// `Err(UtlError)` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::lifecycle::LifecycleLambdaWeights;
    ///
    /// // Valid weights
    /// let weights = LifecycleLambdaWeights::new(0.6, 0.4).unwrap();
    /// assert!((weights.lambda_s() - 0.6).abs() < 0.001);
    ///
    /// // Invalid weights (don't sum to 1.0)
    /// let result = LifecycleLambdaWeights::new(0.6, 0.6);
    /// assert!(result.is_err());
    /// ```
    pub fn new(lambda_s: f32, lambda_c: f32) -> UtlResult<Self> {
        // Validate non-negative
        if lambda_s < 0.0 || lambda_c < 0.0 {
            return Err(UtlError::negative_lambda(lambda_s, lambda_c));
        }

        // Validate range
        if lambda_s > 1.0 || lambda_c > 1.0 {
            return Err(UtlError::InvalidLambdaWeights {
                novelty: lambda_s,
                consolidation: lambda_c,
                reason: "Lambda weights must be in range [0.0, 1.0]".to_string(),
            });
        }

        // Validate sum
        let sum = lambda_s + lambda_c;
        if (sum - 1.0).abs() > Self::EPSILON {
            return Err(UtlError::lambda_sum_error(lambda_s, lambda_c));
        }

        Ok(Self { lambda_s, lambda_c })
    }

    /// Create lambda weights without validation (unsafe).
    ///
    /// # Safety
    ///
    /// This function does not validate that weights sum to 1.0.
    /// Only use when weights are known to be valid.
    ///
    /// # Arguments
    ///
    /// * `lambda_s` - Surprise weight
    /// * `lambda_c` - Coherence weight
    #[inline]
    pub(crate) fn new_unchecked(lambda_s: f32, lambda_c: f32) -> Self {
        Self { lambda_s, lambda_c }
    }

    /// Get lambda weights for a specific lifecycle stage.
    ///
    /// Returns the canonical Marblestone weights for the given stage.
    ///
    /// # Arguments
    ///
    /// * `stage` - The lifecycle stage
    ///
    /// # Returns
    ///
    /// Lambda weights according to the constitution:
    /// - Infancy: `lambda_s=0.7, lambda_c=0.3`
    /// - Growth: `lambda_s=0.5, lambda_c=0.5`
    /// - Maturity: `lambda_s=0.3, lambda_c=0.7`
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::lifecycle::{LifecycleStage, LifecycleLambdaWeights};
    ///
    /// let infancy = LifecycleLambdaWeights::for_stage(LifecycleStage::Infancy);
    /// assert!((infancy.lambda_s() - 0.7).abs() < 0.001);
    ///
    /// let maturity = LifecycleLambdaWeights::for_stage(LifecycleStage::Maturity);
    /// assert!((maturity.lambda_c() - 0.7).abs() < 0.001);
    /// ```
    #[inline]
    pub fn for_stage(stage: LifecycleStage) -> Self {
        match stage {
            LifecycleStage::Infancy => Self::new_unchecked(0.7, 0.3),
            LifecycleStage::Growth => Self::new_unchecked(0.5, 0.5),
            LifecycleStage::Maturity => Self::new_unchecked(0.3, 0.7),
        }
    }

    /// Get lambda weights for an interaction count.
    ///
    /// Automatically determines the lifecycle stage from the interaction
    /// count and returns the corresponding weights.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of interactions
    ///
    /// # Returns
    ///
    /// Lambda weights for the determined lifecycle stage.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::lifecycle::LifecycleLambdaWeights;
    ///
    /// // Infancy range (0-49)
    /// let weights = LifecycleLambdaWeights::for_interaction_count(25);
    /// assert!((weights.lambda_s() - 0.7).abs() < 0.001);
    ///
    /// // Growth range (50-499)
    /// let weights = LifecycleLambdaWeights::for_interaction_count(250);
    /// assert!((weights.lambda_s() - 0.5).abs() < 0.001);
    ///
    /// // Maturity range (500+)
    /// let weights = LifecycleLambdaWeights::for_interaction_count(1000);
    /// assert!((weights.lambda_s() - 0.3).abs() < 0.001);
    /// ```
    #[inline]
    pub fn for_interaction_count(count: u64) -> Self {
        let stage = LifecycleStage::from_interaction_count(count);
        Self::for_stage(stage)
    }

    /// Compute interpolated lambda weights for smooth stage transitions.
    ///
    /// This method provides smooth weight transitions at stage boundaries,
    /// interpolating between adjacent stage weights within the smoothing window.
    ///
    /// # Arguments
    ///
    /// * `count` - Current interaction count
    /// * `config` - Lifecycle configuration containing smoothing settings
    ///
    /// # Returns
    ///
    /// Interpolated lambda weights if smooth transitions are enabled,
    /// otherwise discrete stage weights.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::config::LifecycleConfig;
    /// use context_graph_utl::lifecycle::LifecycleLambdaWeights;
    ///
    /// let config = LifecycleConfig::default();
    ///
    /// // Near stage boundary, weights are interpolated
    /// let weights = LifecycleLambdaWeights::interpolated(55, &config);
    ///
    /// // Weights should be between Infancy and Growth values
    /// assert!(weights.lambda_s() >= 0.5);
    /// assert!(weights.lambda_s() <= 0.7);
    /// ```
    pub fn interpolated(count: u64, config: &LifecycleConfig) -> Self {
        if !config.smooth_transitions {
            return Self::for_interaction_count(count);
        }

        let stage = LifecycleStage::from_interaction_count(count);
        let base_weights = Self::for_stage(stage);

        // Check if we're within smoothing window of a boundary
        let smoothing_window = config.smoothing_window;

        match stage {
            LifecycleStage::Infancy => {
                // Check if approaching Growth boundary
                let threshold = LifecycleStage::INFANCY_THRESHOLD;
                if count >= threshold.saturating_sub(smoothing_window) && count < threshold {
                    let progress = (threshold - count) as f32 / smoothing_window as f32;
                    let next_weights = Self::for_stage(LifecycleStage::Growth);
                    return Self::lerp(&base_weights, &next_weights, 1.0 - progress);
                }
            }
            LifecycleStage::Growth => {
                // Check if just entered Growth (from Infancy boundary)
                let lower_threshold = LifecycleStage::INFANCY_THRESHOLD;
                if count < lower_threshold + smoothing_window {
                    let progress = (count - lower_threshold) as f32 / smoothing_window as f32;
                    let prev_weights = Self::for_stage(LifecycleStage::Infancy);
                    return Self::lerp(&prev_weights, &base_weights, progress);
                }

                // Check if approaching Maturity boundary
                let upper_threshold = LifecycleStage::GROWTH_THRESHOLD;
                if count >= upper_threshold.saturating_sub(smoothing_window)
                    && count < upper_threshold
                {
                    let progress = (upper_threshold - count) as f32 / smoothing_window as f32;
                    let next_weights = Self::for_stage(LifecycleStage::Maturity);
                    return Self::lerp(&base_weights, &next_weights, 1.0 - progress);
                }
            }
            LifecycleStage::Maturity => {
                // Check if just entered Maturity (from Growth boundary)
                let threshold = LifecycleStage::GROWTH_THRESHOLD;
                if count < threshold + smoothing_window {
                    let progress = (count - threshold) as f32 / smoothing_window as f32;
                    let prev_weights = Self::for_stage(LifecycleStage::Growth);
                    return Self::lerp(&prev_weights, &base_weights, progress);
                }
            }
        }

        base_weights
    }

    /// Linear interpolation between two weight sets.
    ///
    /// # Arguments
    ///
    /// * `a` - Starting weights (t=0)
    /// * `b` - Ending weights (t=1)
    /// * `t` - Interpolation factor in [0.0, 1.0]
    ///
    /// # Returns
    ///
    /// Interpolated weights that preserve the sum-to-one invariant.
    #[inline]
    fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let lambda_s = a.lambda_s * (1.0 - t) + b.lambda_s * t;
        let lambda_c = a.lambda_c * (1.0 - t) + b.lambda_c * t;
        Self::new_unchecked(lambda_s, lambda_c)
    }

    /// Get the surprise (novelty) weight.
    ///
    /// # Returns
    ///
    /// The `lambda_s` weight value.
    #[inline]
    pub fn lambda_s(&self) -> f32 {
        self.lambda_s
    }

    /// Get the coherence (consolidation) weight.
    ///
    /// # Returns
    ///
    /// The `lambda_c` weight value.
    #[inline]
    pub fn lambda_c(&self) -> f32 {
        self.lambda_c
    }

    /// Apply these weights to surprise and coherence values.
    ///
    /// Computes: `lambda_s * delta_s + lambda_c * delta_c`
    ///
    /// # Arguments
    ///
    /// * `delta_s` - Surprise value in [0.0, 1.0]
    /// * `delta_c` - Coherence value in [0.0, 1.0]
    ///
    /// # Returns
    ///
    /// Weighted combination of surprise and coherence.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::lifecycle::{LifecycleStage, LifecycleLambdaWeights};
    ///
    /// let weights = LifecycleLambdaWeights::for_stage(LifecycleStage::Infancy);
    /// let result = weights.apply(0.8, 0.6);
    ///
    /// // 0.7 * 0.8 + 0.3 * 0.6 = 0.56 + 0.18 = 0.74
    /// assert!((result - 0.74).abs() < 0.001);
    /// ```
    #[inline]
    pub fn apply(&self, delta_s: f32, delta_c: f32) -> f32 {
        self.lambda_s * delta_s + self.lambda_c * delta_c
    }

    /// Validate that weights sum to 1.0.
    ///
    /// # Returns
    ///
    /// `true` if weights are valid (sum to 1.0 within epsilon),
    /// `false` otherwise.
    #[inline]
    pub fn is_valid(&self) -> bool {
        let sum = self.lambda_s + self.lambda_c;
        (sum - 1.0).abs() <= Self::EPSILON && self.lambda_s >= 0.0 && self.lambda_c >= 0.0
    }

    /// Get the dominant focus of these weights.
    ///
    /// # Returns
    ///
    /// - `"surprise"` if `lambda_s > lambda_c`
    /// - `"coherence"` if `lambda_c > lambda_s`
    /// - `"balanced"` if weights are equal
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::lifecycle::{LifecycleStage, LifecycleLambdaWeights};
    ///
    /// let infancy = LifecycleLambdaWeights::for_stage(LifecycleStage::Infancy);
    /// assert_eq!(infancy.focus(), "surprise");
    ///
    /// let growth = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
    /// assert_eq!(growth.focus(), "balanced");
    ///
    /// let maturity = LifecycleLambdaWeights::for_stage(LifecycleStage::Maturity);
    /// assert_eq!(maturity.focus(), "coherence");
    /// ```
    pub fn focus(&self) -> &'static str {
        if (self.lambda_s - self.lambda_c).abs() < Self::EPSILON {
            "balanced"
        } else if self.lambda_s > self.lambda_c {
            "surprise"
        } else {
            "coherence"
        }
    }
}

impl Default for LifecycleLambdaWeights {
    /// Returns weights for the Infancy stage (novelty-focused).
    fn default() -> Self {
        Self::for_stage(LifecycleStage::Infancy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid_weights() {
        let weights = LifecycleLambdaWeights::new(0.6, 0.4).unwrap();
        assert!((weights.lambda_s() - 0.6).abs() < 0.001);
        assert!((weights.lambda_c() - 0.4).abs() < 0.001);
        assert!(weights.is_valid());
    }

    #[test]
    fn test_new_invalid_sum() {
        let result = LifecycleLambdaWeights::new(0.6, 0.6);
        assert!(result.is_err());
        match result.unwrap_err() {
            UtlError::InvalidLambdaWeights {
                novelty,
                consolidation,
                ..
            } => {
                assert!((novelty - 0.6).abs() < 0.001);
                assert!((consolidation - 0.6).abs() < 0.001);
            }
            _ => panic!("Expected InvalidLambdaWeights error"),
        }
    }

    #[test]
    fn test_new_negative_weights() {
        let result = LifecycleLambdaWeights::new(-0.3, 1.3);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_weights_exceed_one() {
        let result = LifecycleLambdaWeights::new(1.5, -0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_for_stage_infancy() {
        let weights = LifecycleLambdaWeights::for_stage(LifecycleStage::Infancy);
        assert!((weights.lambda_s() - 0.7).abs() < 0.001);
        assert!((weights.lambda_c() - 0.3).abs() < 0.001);
        assert!(weights.is_valid());
    }

    #[test]
    fn test_for_stage_growth() {
        let weights = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        assert!((weights.lambda_s() - 0.5).abs() < 0.001);
        assert!((weights.lambda_c() - 0.5).abs() < 0.001);
        assert!(weights.is_valid());
    }

    #[test]
    fn test_for_stage_maturity() {
        let weights = LifecycleLambdaWeights::for_stage(LifecycleStage::Maturity);
        assert!((weights.lambda_s() - 0.3).abs() < 0.001);
        assert!((weights.lambda_c() - 0.7).abs() < 0.001);
        assert!(weights.is_valid());
    }

    #[test]
    fn test_for_interaction_count() {
        // Infancy
        let weights = LifecycleLambdaWeights::for_interaction_count(25);
        assert!((weights.lambda_s() - 0.7).abs() < 0.001);

        // Growth
        let weights = LifecycleLambdaWeights::for_interaction_count(250);
        assert!((weights.lambda_s() - 0.5).abs() < 0.001);

        // Maturity
        let weights = LifecycleLambdaWeights::for_interaction_count(1000);
        assert!((weights.lambda_s() - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_interpolated_no_smoothing() {
        let mut config = LifecycleConfig::default();
        config.smooth_transitions = false;

        let weights = LifecycleLambdaWeights::interpolated(55, &config);
        assert!((weights.lambda_s() - 0.5).abs() < 0.001); // Should be Growth weights
    }

    #[test]
    fn test_interpolated_with_smoothing() {
        let config = LifecycleConfig::default();

        // Well within Infancy - should be pure Infancy weights
        let weights = LifecycleLambdaWeights::interpolated(10, &config);
        assert!((weights.lambda_s() - 0.7).abs() < 0.001);

        // Just past Growth threshold - should be interpolated
        let weights = LifecycleLambdaWeights::interpolated(55, &config);
        assert!(weights.lambda_s() > 0.5);
        assert!(weights.lambda_s() < 0.7);

        // Well into Growth - should be pure Growth weights
        let weights = LifecycleLambdaWeights::interpolated(250, &config);
        assert!((weights.lambda_s() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_apply() {
        let weights = LifecycleLambdaWeights::for_stage(LifecycleStage::Infancy);
        let result = weights.apply(0.8, 0.6);

        // 0.7 * 0.8 + 0.3 * 0.6 = 0.56 + 0.18 = 0.74
        assert!((result - 0.74).abs() < 0.001);
    }

    #[test]
    fn test_focus() {
        assert_eq!(
            LifecycleLambdaWeights::for_stage(LifecycleStage::Infancy).focus(),
            "surprise"
        );
        assert_eq!(
            LifecycleLambdaWeights::for_stage(LifecycleStage::Growth).focus(),
            "balanced"
        );
        assert_eq!(
            LifecycleLambdaWeights::for_stage(LifecycleStage::Maturity).focus(),
            "coherence"
        );
    }

    #[test]
    fn test_default() {
        let weights = LifecycleLambdaWeights::default();
        assert!((weights.lambda_s() - 0.7).abs() < 0.001);
        assert!((weights.lambda_c() - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_serialization() {
        let weights = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let json = serde_json::to_string(&weights).unwrap();
        let deserialized: LifecycleLambdaWeights = serde_json::from_str(&json).unwrap();

        assert!((deserialized.lambda_s() - 0.5).abs() < 0.001);
        assert!((deserialized.lambda_c() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_is_valid() {
        let valid = LifecycleLambdaWeights::new_unchecked(0.6, 0.4);
        assert!(valid.is_valid());

        let invalid = LifecycleLambdaWeights::new_unchecked(0.6, 0.6);
        assert!(!invalid.is_valid());

        let negative = LifecycleLambdaWeights::new_unchecked(-0.2, 1.2);
        assert!(!negative.is_valid());
    }

    #[test]
    fn test_lerp() {
        let a = LifecycleLambdaWeights::new_unchecked(0.7, 0.3);
        let b = LifecycleLambdaWeights::new_unchecked(0.5, 0.5);

        // t=0 should give a
        let result = LifecycleLambdaWeights::lerp(&a, &b, 0.0);
        assert!((result.lambda_s() - 0.7).abs() < 0.001);

        // t=1 should give b
        let result = LifecycleLambdaWeights::lerp(&a, &b, 1.0);
        assert!((result.lambda_s() - 0.5).abs() < 0.001);

        // t=0.5 should give midpoint
        let result = LifecycleLambdaWeights::lerp(&a, &b, 0.5);
        assert!((result.lambda_s() - 0.6).abs() < 0.001);
        assert!((result.lambda_c() - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_equality() {
        let a = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let b = LifecycleLambdaWeights::new(0.5, 0.5).unwrap();

        assert_eq!(a, b);
    }

    #[test]
    fn test_clone_and_copy() {
        let weights = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let cloned = weights.clone();
        let copied = weights;

        assert_eq!(weights, cloned);
        assert_eq!(weights, copied);
    }

    #[test]
    fn test_debug() {
        let weights = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let debug = format!("{:?}", weights);
        assert!(debug.contains("lambda_s"));
        assert!(debug.contains("lambda_c"));
    }

    #[test]
    fn test_all_stages_weights_sum_to_one() {
        for stage in LifecycleStage::all() {
            let weights = LifecycleLambdaWeights::for_stage(stage);
            let sum = weights.lambda_s() + weights.lambda_c();
            assert!(
                (sum - 1.0).abs() < LifecycleLambdaWeights::EPSILON,
                "Stage {:?} weights sum to {}, not 1.0",
                stage,
                sum
            );
        }
    }

    #[test]
    fn test_weights_decrease_surprise_with_maturity() {
        let infancy = LifecycleLambdaWeights::for_stage(LifecycleStage::Infancy);
        let growth = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let maturity = LifecycleLambdaWeights::for_stage(LifecycleStage::Maturity);

        assert!(infancy.lambda_s() > growth.lambda_s());
        assert!(growth.lambda_s() > maturity.lambda_s());
    }

    #[test]
    fn test_weights_increase_coherence_with_maturity() {
        let infancy = LifecycleLambdaWeights::for_stage(LifecycleStage::Infancy);
        let growth = LifecycleLambdaWeights::for_stage(LifecycleStage::Growth);
        let maturity = LifecycleLambdaWeights::for_stage(LifecycleStage::Maturity);

        assert!(infancy.lambda_c() < growth.lambda_c());
        assert!(growth.lambda_c() < maturity.lambda_c());
    }
}
