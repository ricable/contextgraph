//! Meta-Learning Callback Trait
//!
//! TASK-METAUTL-P0-006: Defines the cross-crate interface for meta-learning integration.
//!
//! This trait is defined in `context-graph-core` but implemented by `context-graph-mcp`.
//! This pattern avoids circular dependencies between crates while enabling the
//! MetaCognitiveLoop to invoke lambda self-correction.
//!
//! # Architecture
//!
//! ```text
//! context-graph-core (defines trait)
//!          │
//!          ▼
//! context-graph-mcp (implements trait via MetaLearningService)
//! ```

// ============================================================================
// Type Definitions
// ============================================================================

/// Domain classification for domain-specific tracking.
/// TASK-METAUTL-P0-006: Mirrors context-graph-mcp Domain enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MetaDomain {
    Code,
    Medical,
    Legal,
    Creative,
    Research,
    #[default]
    General,
}

/// Lambda weights pair (lambda_s, lambda_c).
/// TASK-METAUTL-P0-006: Simplified representation for cross-crate transfer.
#[derive(Clone, Copy, Debug)]
pub struct LambdaValues {
    pub lambda_s: f32,
    pub lambda_c: f32,
}

impl LambdaValues {
    /// Create new lambda values.
    pub fn new(lambda_s: f32, lambda_c: f32) -> Self {
        Self { lambda_s, lambda_c }
    }

    /// Default infancy weights.
    pub fn infancy() -> Self {
        Self::new(0.7, 0.3)
    }

    /// Validate sum-to-one invariant.
    pub fn is_valid(&self) -> bool {
        (self.lambda_s + self.lambda_c - 1.0).abs() < 0.001
    }
}

impl Default for LambdaValues {
    fn default() -> Self {
        Self::infancy()
    }
}

/// Result of a lambda adjustment operation.
/// TASK-METAUTL-P0-006: Cross-crate safe adjustment result.
#[derive(Clone, Copy, Debug)]
pub struct MetaLambdaAdjustment {
    /// Change in surprise weight
    pub delta_lambda_s: f32,
    /// Change in coherence weight
    pub delta_lambda_c: f32,
    /// Learning rate used
    pub alpha: f32,
    /// Trigger error that caused adjustment
    pub trigger_error: f32,
}

/// Callback status from meta-learning service.
/// TASK-METAUTL-P0-006: Reports state after callback invocation.
#[derive(Clone, Debug)]
pub struct MetaCallbackStatus {
    /// Current accuracy
    pub accuracy: f32,
    /// Whether adjustment was made
    pub adjustment_made: bool,
    /// Adjustment details (if made)
    pub adjustment: Option<MetaLambdaAdjustment>,
    /// Current lambda values
    pub current_lambdas: LambdaValues,
    /// Whether escalation should be triggered
    pub should_escalate: bool,
    /// Total adjustments made
    pub total_adjustments: u64,
}

// ============================================================================
// MetaLearningCallback Trait
// ============================================================================

/// Callback trait for meta-learning service integration.
///
/// TASK-METAUTL-P0-006: This trait enables `MetaCognitiveLoop` (in core crate)
/// to invoke lambda self-correction (in MCP crate) without circular dependencies.
///
/// # Implementation
///
/// The MCP crate implements this trait via `MetaLearningService`, which is then
/// passed to `MetaCognitiveLoop::evaluate_with_correction()`.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` for async integration.
pub trait MetaLearningCallback: Send + Sync {
    /// Record a prediction and potentially trigger adjustment.
    ///
    /// # Arguments
    ///
    /// - `embedder_idx`: Index of embedder (for per-embedder tracking)
    /// - `predicted`: Predicted learning value
    /// - `actual`: Actual learning value
    /// - `domain`: Optional domain context
    /// - `ach_level`: Current acetylcholine level
    ///
    /// # Returns
    ///
    /// Status including any adjustment made.
    fn record_prediction(
        &mut self,
        embedder_idx: usize,
        predicted: f32,
        actual: f32,
        domain: Option<MetaDomain>,
        ach_level: f32,
    ) -> MetaCallbackStatus;

    /// Get current lambda weights.
    fn current_lambdas(&self) -> LambdaValues;

    /// Get current accuracy.
    fn current_accuracy(&self) -> f32;

    /// Check if escalation should be triggered.
    fn should_escalate(&self) -> bool;

    /// Trigger escalation (Bayesian optimization).
    ///
    /// # Returns
    ///
    /// `true` if escalation was successful, `false` otherwise.
    fn trigger_escalation(&mut self) -> bool;

    /// Get total adjustment count.
    fn adjustment_count(&self) -> u64;

    /// Reset to base weights.
    fn reset_to_base(&mut self);

    /// Check if self-correction is enabled.
    fn is_enabled(&self) -> bool;
}

// ============================================================================
// Enhanced MetaCognitiveState
// ============================================================================

/// Enhanced meta-cognitive state including lambda correction info.
/// TASK-METAUTL-P0-006: Extends base MetaCognitiveState with correction details.
#[derive(Clone, Debug)]
pub struct EnhancedMetaCognitiveState {
    /// Current meta-score
    pub meta_score: f32,
    /// Average meta-score over recent history
    pub avg_meta_score: f32,
    /// Whether introspective dream is triggered
    pub dream_triggered: bool,
    /// Current acetylcholine level
    pub acetylcholine: f32,
    /// Lambda adjustment made (if any)
    pub lambda_adjustment: Option<MetaLambdaAdjustment>,
    /// Whether escalation was triggered
    pub escalation_triggered: bool,
    /// Updated lambda weights
    pub current_lambdas: Option<LambdaValues>,
    /// Current accuracy from meta-learning
    pub current_accuracy: Option<f32>,
}

impl EnhancedMetaCognitiveState {
    /// Create from base MetaCognitiveState without correction info.
    ///
    /// Used when no meta-learning service is provided (backward compatibility).
    pub fn from_base(
        meta_score: f32,
        avg_meta_score: f32,
        dream_triggered: bool,
        acetylcholine: f32,
    ) -> Self {
        Self {
            meta_score,
            avg_meta_score,
            dream_triggered,
            acetylcholine,
            lambda_adjustment: None,
            escalation_triggered: false,
            current_lambdas: None,
            current_accuracy: None,
        }
    }

    /// Create with correction info.
    pub fn with_correction(
        meta_score: f32,
        avg_meta_score: f32,
        dream_triggered: bool,
        acetylcholine: f32,
        callback_status: MetaCallbackStatus,
    ) -> Self {
        Self {
            meta_score,
            avg_meta_score,
            dream_triggered,
            acetylcholine,
            lambda_adjustment: callback_status.adjustment,
            escalation_triggered: callback_status.should_escalate,
            current_lambdas: Some(callback_status.current_lambdas),
            current_accuracy: Some(callback_status.accuracy),
        }
    }
}

// ============================================================================
// No-Op Implementation
// ============================================================================

/// No-op implementation for when meta-learning is disabled.
///
/// TASK-METAUTL-P0-006: Provides a default implementation that does nothing,
/// maintaining backward compatibility.
#[derive(Debug, Clone, Default)]
pub struct NoOpMetaLearningCallback {
    lambdas: LambdaValues,
}

impl NoOpMetaLearningCallback {
    pub fn new() -> Self {
        Self {
            lambdas: LambdaValues::infancy(),
        }
    }
}

impl MetaLearningCallback for NoOpMetaLearningCallback {
    fn record_prediction(
        &mut self,
        _embedder_idx: usize,
        _predicted: f32,
        _actual: f32,
        _domain: Option<MetaDomain>,
        _ach_level: f32,
    ) -> MetaCallbackStatus {
        MetaCallbackStatus {
            accuracy: 1.0, // Always perfect when disabled
            adjustment_made: false,
            adjustment: None,
            current_lambdas: self.lambdas,
            should_escalate: false,
            total_adjustments: 0,
        }
    }

    fn current_lambdas(&self) -> LambdaValues {
        self.lambdas
    }

    fn current_accuracy(&self) -> f32 {
        1.0
    }

    fn should_escalate(&self) -> bool {
        false
    }

    fn trigger_escalation(&mut self) -> bool {
        false
    }

    fn adjustment_count(&self) -> u64 {
        0
    }

    fn reset_to_base(&mut self) {
        self.lambdas = LambdaValues::infancy();
    }

    fn is_enabled(&self) -> bool {
        false
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gwt::meta_cognitive::ACH_BASELINE;

    #[test]
    fn test_lambda_values_validity() {
        let valid = LambdaValues::new(0.7, 0.3);
        assert!(valid.is_valid());

        let invalid = LambdaValues::new(0.6, 0.6);
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_lambda_values_default() {
        let default = LambdaValues::default();
        assert!((default.lambda_s - 0.7).abs() < 0.001);
        assert!((default.lambda_c - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_enhanced_state_from_base() {
        let state = EnhancedMetaCognitiveState::from_base(0.5, 0.55, false, ACH_BASELINE);
        assert!((state.meta_score - 0.5).abs() < 0.001);
        assert!(state.lambda_adjustment.is_none());
        assert!(!state.escalation_triggered);
    }

    #[test]
    fn test_no_op_callback() {
        let mut callback = NoOpMetaLearningCallback::new();

        let status = callback.record_prediction(0, 0.5, 0.5, None, ACH_BASELINE);
        assert!(!status.adjustment_made);
        assert!((status.accuracy - 1.0).abs() < 0.001);

        assert!(!callback.is_enabled());
        assert_eq!(callback.adjustment_count(), 0);
    }

    #[test]
    fn test_meta_domain_default() {
        let domain = MetaDomain::default();
        assert_eq!(domain, MetaDomain::General);
    }
}
