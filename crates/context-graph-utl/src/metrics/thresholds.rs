//! Stage-specific threshold configuration for UTL learning events.

use serde::{Deserialize, Serialize};

use crate::lifecycle::LifecycleStage;

/// Thresholds that vary by lifecycle stage.
///
/// These thresholds control when learning events are triggered and
/// how memories are stored/consolidated.
///
/// # Lifecycle Stage Defaults
///
/// - **Infancy**: High entropy trigger (capture novelty), low coherence trigger
/// - **Growth**: Balanced thresholds
/// - **Maturity**: Low entropy trigger, high coherence trigger (prefer consolidation)
///
/// # Example
///
/// ```
/// use context_graph_utl::metrics::StageThresholds;
/// use context_graph_utl::lifecycle::LifecycleStage;
///
/// let thresholds = StageThresholds::for_stage(LifecycleStage::Growth);
/// assert_eq!(thresholds.entropy_trigger, 0.7);
/// assert_eq!(thresholds.coherence_trigger, 0.5);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StageThresholds {
    /// Entropy level that triggers learning [0.0, 1.0]
    pub entropy_trigger: f32,

    /// Coherence level that triggers consolidation [0.0, 1.0]
    pub coherence_trigger: f32,

    /// Minimum importance score to store a memory [0.0, 1.0]
    pub min_importance_store: f32,

    /// Threshold for triggering memory consolidation [0.0, 1.0]
    pub consolidation_threshold: f32,
}

impl Default for StageThresholds {
    /// Returns Growth stage thresholds as default (balanced learning).
    fn default() -> Self {
        Self {
            entropy_trigger: 0.7,
            coherence_trigger: 0.5,
            min_importance_store: 0.3,
            consolidation_threshold: 0.5,
        }
    }
}

impl StageThresholds {
    /// Create thresholds for Infancy stage (novelty-focused).
    pub fn infancy() -> Self {
        Self {
            entropy_trigger: 0.9,
            coherence_trigger: 0.2,
            min_importance_store: 0.1,
            consolidation_threshold: 0.3,
        }
    }

    /// Create thresholds for Growth stage (balanced).
    pub fn growth() -> Self {
        Self::default()
    }

    /// Create thresholds for Maturity stage (coherence-focused).
    pub fn maturity() -> Self {
        Self {
            entropy_trigger: 0.5,
            coherence_trigger: 0.7,
            min_importance_store: 0.5,
            consolidation_threshold: 0.7,
        }
    }

    /// Create thresholds for a specific lifecycle stage.
    pub fn for_stage(stage: LifecycleStage) -> Self {
        match stage {
            LifecycleStage::Infancy => Self::infancy(),
            LifecycleStage::Growth => Self::growth(),
            LifecycleStage::Maturity => Self::maturity(),
        }
    }

    /// Check if entropy exceeds trigger threshold.
    #[inline]
    pub fn should_trigger_learning(&self, entropy: f32) -> bool {
        entropy >= self.entropy_trigger
    }

    /// Check if coherence exceeds trigger threshold.
    #[inline]
    pub fn should_consolidate(&self, coherence: f32) -> bool {
        coherence >= self.coherence_trigger
    }

    /// Check if importance is sufficient to store.
    #[inline]
    pub fn should_store(&self, importance: f32) -> bool {
        importance >= self.min_importance_store
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_thresholds_default() {
        let thresholds = StageThresholds::default();

        // Default is Growth stage thresholds
        assert_eq!(thresholds.entropy_trigger, 0.7);
        assert_eq!(thresholds.coherence_trigger, 0.5);
        assert_eq!(thresholds.min_importance_store, 0.3);
        assert_eq!(thresholds.consolidation_threshold, 0.5);
    }

    #[test]
    fn test_stage_thresholds_infancy() {
        let thresholds = StageThresholds::infancy();

        assert_eq!(thresholds.entropy_trigger, 0.9);
        assert_eq!(thresholds.coherence_trigger, 0.2);
        assert_eq!(thresholds.min_importance_store, 0.1);
        assert_eq!(thresholds.consolidation_threshold, 0.3);
    }

    #[test]
    fn test_stage_thresholds_growth() {
        let thresholds = StageThresholds::growth();

        // Growth is the default
        assert_eq!(thresholds, StageThresholds::default());
    }

    #[test]
    fn test_stage_thresholds_maturity() {
        let thresholds = StageThresholds::maturity();

        assert_eq!(thresholds.entropy_trigger, 0.5);
        assert_eq!(thresholds.coherence_trigger, 0.7);
        assert_eq!(thresholds.min_importance_store, 0.5);
        assert_eq!(thresholds.consolidation_threshold, 0.7);
    }

    #[test]
    fn test_stage_thresholds_for_stage() {
        assert_eq!(
            StageThresholds::for_stage(LifecycleStage::Infancy),
            StageThresholds::infancy()
        );
        assert_eq!(
            StageThresholds::for_stage(LifecycleStage::Growth),
            StageThresholds::growth()
        );
        assert_eq!(
            StageThresholds::for_stage(LifecycleStage::Maturity),
            StageThresholds::maturity()
        );
    }

    #[test]
    fn test_stage_thresholds_should_trigger_learning() {
        let thresholds = StageThresholds::growth(); // entropy_trigger = 0.7

        assert!(!thresholds.should_trigger_learning(0.5)); // Below threshold
        assert!(thresholds.should_trigger_learning(0.7)); // At threshold
        assert!(thresholds.should_trigger_learning(0.9)); // Above threshold
    }

    #[test]
    fn test_stage_thresholds_should_consolidate() {
        let thresholds = StageThresholds::growth(); // coherence_trigger = 0.5

        assert!(!thresholds.should_consolidate(0.3)); // Below threshold
        assert!(thresholds.should_consolidate(0.5)); // At threshold
        assert!(thresholds.should_consolidate(0.8)); // Above threshold
    }

    #[test]
    fn test_stage_thresholds_should_store() {
        let thresholds = StageThresholds::growth(); // min_importance_store = 0.3

        assert!(!thresholds.should_store(0.1)); // Below threshold
        assert!(thresholds.should_store(0.3)); // At threshold
        assert!(thresholds.should_store(0.7)); // Above threshold
    }

    #[test]
    fn test_stage_thresholds_serialization() {
        let thresholds = StageThresholds::maturity();

        let json = serde_json::to_string(&thresholds).expect("serialize");
        let parsed: StageThresholds = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(thresholds, parsed);
    }

    #[test]
    fn test_stage_thresholds_lifecycle_progression() {
        // Verify that entropy trigger decreases as lifecycle progresses
        // (system becomes less novelty-seeking)
        let infancy = StageThresholds::infancy();
        let growth = StageThresholds::growth();
        let maturity = StageThresholds::maturity();

        assert!(infancy.entropy_trigger > growth.entropy_trigger);
        assert!(growth.entropy_trigger > maturity.entropy_trigger);

        // Verify that coherence trigger increases as lifecycle progresses
        // (system becomes more consolidation-focused)
        assert!(infancy.coherence_trigger < growth.coherence_trigger);
        assert!(growth.coherence_trigger < maturity.coherence_trigger);
    }
}
