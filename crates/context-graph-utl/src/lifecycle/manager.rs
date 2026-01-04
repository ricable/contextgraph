//! Lifecycle manager for tracking and transitioning between stages.
//!
//! The `LifecycleManager` tracks the current lifecycle stage of a knowledge
//! base and handles transitions between stages as interactions accumulate.
//! It supports both automatic transitions (based on interaction count) and
//! manual transitions.
//!
//! # Constitution Reference
//!
//! ```text
//! Infancy (n=0-50):   lambda_s=0.7, lambda_c=0.3, stance="capture-novelty"
//! Growth (n=50-500):  lambda_s=0.5, lambda_c=0.5, stance="balanced"
//! Maturity (n=500+):  lambda_s=0.3, lambda_c=0.7, stance="curation-coherence"
//! ```

use serde::{Deserialize, Serialize};

use crate::config::LifecycleConfig;
use crate::error::{UtlError, UtlResult};

use super::lambda::LifecycleLambdaWeights;
use super::stage::LifecycleStage;

/// Manager for lifecycle stage tracking and transitions.
///
/// Tracks the current interaction count and lifecycle stage, providing
/// automatic stage transitions as interactions accumulate. Supports both
/// discrete stage weights and smooth interpolated weights at boundaries.
///
/// # Example
///
/// ```
/// use context_graph_utl::config::LifecycleConfig;
/// use context_graph_utl::lifecycle::{LifecycleManager, LifecycleStage};
///
/// let config = LifecycleConfig::default();
/// let mut manager = LifecycleManager::new(&config);
///
/// // Initial state
/// assert_eq!(manager.current_stage(), LifecycleStage::Infancy);
/// assert_eq!(manager.interaction_count(), 0);
///
/// // Simulate interactions
/// for _ in 0..60 {
///     manager.increment();
/// }
///
/// // Stage should have advanced
/// assert_eq!(manager.current_stage(), LifecycleStage::Growth);
/// assert_eq!(manager.interaction_count(), 60);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleManager {
    /// Current interaction count.
    interaction_count: u64,

    /// Current lifecycle stage.
    current_stage: LifecycleStage,

    /// Whether automatic transitions are enabled.
    auto_transition: bool,

    /// Hysteresis buffer to prevent rapid stage switching.
    transition_hysteresis: u64,

    /// Interaction count at last stage transition.
    last_transition_count: u64,

    /// Enable smooth interpolation between stages.
    smooth_transitions: bool,

    /// Smoothing window size for interpolation.
    smoothing_window: u64,
}

impl LifecycleManager {
    /// Create a new lifecycle manager from configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Lifecycle configuration settings
    ///
    /// # Returns
    ///
    /// A new `LifecycleManager` starting in Infancy stage with 0 interactions.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::config::LifecycleConfig;
    /// use context_graph_utl::lifecycle::LifecycleManager;
    ///
    /// let config = LifecycleConfig::default();
    /// let manager = LifecycleManager::new(&config);
    ///
    /// assert_eq!(manager.interaction_count(), 0);
    /// ```
    pub fn new(config: &LifecycleConfig) -> Self {
        Self {
            interaction_count: 0,
            current_stage: LifecycleStage::Infancy,
            auto_transition: config.auto_transition,
            transition_hysteresis: config.transition_hysteresis,
            last_transition_count: 0,
            smooth_transitions: config.smooth_transitions,
            smoothing_window: config.smoothing_window,
        }
    }

    /// Create a lifecycle manager with a specific initial state.
    ///
    /// # Arguments
    ///
    /// * `config` - Lifecycle configuration settings
    /// * `interaction_count` - Initial interaction count
    ///
    /// # Returns
    ///
    /// A new `LifecycleManager` with the given interaction count and
    /// corresponding stage.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::config::LifecycleConfig;
    /// use context_graph_utl::lifecycle::{LifecycleManager, LifecycleStage};
    ///
    /// let config = LifecycleConfig::default();
    /// let manager = LifecycleManager::with_count(&config, 100);
    ///
    /// assert_eq!(manager.interaction_count(), 100);
    /// assert_eq!(manager.current_stage(), LifecycleStage::Growth);
    /// ```
    pub fn with_count(config: &LifecycleConfig, interaction_count: u64) -> Self {
        let current_stage = LifecycleStage::from_interaction_count(interaction_count);
        Self {
            interaction_count,
            current_stage,
            auto_transition: config.auto_transition,
            transition_hysteresis: config.transition_hysteresis,
            last_transition_count: interaction_count,
            smooth_transitions: config.smooth_transitions,
            smoothing_window: config.smoothing_window,
        }
    }

    /// Get the current lifecycle stage.
    ///
    /// # Returns
    ///
    /// The current `LifecycleStage`.
    #[inline]
    pub fn current_stage(&self) -> LifecycleStage {
        self.current_stage
    }

    /// Get the current interaction count.
    ///
    /// # Returns
    ///
    /// The total number of interactions recorded.
    #[inline]
    pub fn interaction_count(&self) -> u64 {
        self.interaction_count
    }

    /// Get the lambda weights for the current stage.
    ///
    /// Returns discrete stage weights (no interpolation).
    ///
    /// # Returns
    ///
    /// Lambda weights for the current lifecycle stage.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::config::LifecycleConfig;
    /// use context_graph_utl::lifecycle::LifecycleManager;
    ///
    /// let config = LifecycleConfig::default();
    /// let manager = LifecycleManager::new(&config);
    ///
    /// let weights = manager.current_weights();
    /// // Infancy: lambda_s=0.7, lambda_c=0.3
    /// assert!((weights.lambda_s() - 0.7).abs() < 0.001);
    /// ```
    #[inline]
    pub fn current_weights(&self) -> LifecycleLambdaWeights {
        LifecycleLambdaWeights::for_stage(self.current_stage)
    }

    /// Get interpolated lambda weights for smooth transitions.
    ///
    /// If smooth transitions are enabled and the interaction count is near
    /// a stage boundary, returns interpolated weights. Otherwise returns
    /// discrete stage weights.
    ///
    /// # Arguments
    ///
    /// * `config` - Lifecycle configuration (needed for smoothing parameters)
    ///
    /// # Returns
    ///
    /// Lambda weights, possibly interpolated at stage boundaries.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::config::LifecycleConfig;
    /// use context_graph_utl::lifecycle::LifecycleManager;
    ///
    /// let config = LifecycleConfig::default();
    /// let manager = LifecycleManager::with_count(&config, 55);
    ///
    /// let weights = manager.interpolated_weights(&config);
    /// // Near Growth boundary, weights are interpolated
    /// assert!(weights.lambda_s() >= 0.5);
    /// assert!(weights.lambda_s() <= 0.7);
    /// ```
    pub fn interpolated_weights(&self, config: &LifecycleConfig) -> LifecycleLambdaWeights {
        if self.smooth_transitions {
            LifecycleLambdaWeights::interpolated(self.interaction_count, config)
        } else {
            self.current_weights()
        }
    }

    /// Increment the interaction count.
    ///
    /// This method increments the interaction counter and, if automatic
    /// transitions are enabled, checks for and performs stage transitions.
    ///
    /// # Returns
    ///
    /// `true` if a stage transition occurred, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::config::LifecycleConfig;
    /// use context_graph_utl::lifecycle::{LifecycleManager, LifecycleStage};
    ///
    /// // Create config without hysteresis for predictable transitions
    /// let config = LifecycleConfig {
    ///     transition_hysteresis: 0,
    ///     ..LifecycleConfig::default()
    /// };
    /// // Start at count 49 (Infancy), after increment we reach 50 (Growth threshold)
    /// let mut manager = LifecycleManager::with_count(&config, 49);
    /// assert_eq!(manager.current_stage(), LifecycleStage::Infancy);
    ///
    /// // Increment to 50 triggers transition to Growth
    /// let transitioned = manager.increment();
    /// assert!(transitioned);
    /// assert_eq!(manager.current_stage(), LifecycleStage::Growth);
    /// ```
    pub fn increment(&mut self) -> bool {
        self.interaction_count = self.interaction_count.saturating_add(1);

        if self.auto_transition {
            self.check_transition()
        } else {
            false
        }
    }

    /// Increment the interaction count by a specified amount.
    ///
    /// Useful for batch processing where multiple interactions are recorded
    /// at once.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of interactions to add
    ///
    /// # Returns
    ///
    /// `true` if a stage transition occurred, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::config::LifecycleConfig;
    /// use context_graph_utl::lifecycle::{LifecycleManager, LifecycleStage};
    ///
    /// let config = LifecycleConfig::default();
    /// let mut manager = LifecycleManager::new(&config);
    ///
    /// // Add 100 interactions at once
    /// let transitioned = manager.increment_by(100);
    /// assert!(transitioned);
    /// assert_eq!(manager.current_stage(), LifecycleStage::Growth);
    /// ```
    pub fn increment_by(&mut self, count: u64) -> bool {
        self.interaction_count = self.interaction_count.saturating_add(count);

        if self.auto_transition {
            self.check_transition()
        } else {
            false
        }
    }

    /// Check and perform stage transition if warranted.
    ///
    /// This method checks if the current interaction count warrants a
    /// stage transition, respecting the hysteresis buffer to prevent
    /// rapid switching.
    ///
    /// # Returns
    ///
    /// `true` if a transition occurred, `false` otherwise.
    fn check_transition(&mut self) -> bool {
        let expected_stage = LifecycleStage::from_interaction_count(self.interaction_count);

        if expected_stage != self.current_stage {
            // Check hysteresis - only transition if we've moved beyond the buffer
            let interactions_since_last = self
                .interaction_count
                .saturating_sub(self.last_transition_count);

            if interactions_since_last >= self.transition_hysteresis {
                self.current_stage = expected_stage;
                self.last_transition_count = self.interaction_count;
                return true;
            }
        }

        false
    }

    /// Manually transition to a specific lifecycle stage.
    ///
    /// This method allows explicit stage transitions, bypassing automatic
    /// transition logic. The transition must be forward (toward maturity)
    /// or to the same stage.
    ///
    /// # Arguments
    ///
    /// * `stage` - Target lifecycle stage
    ///
    /// # Returns
    ///
    /// `Ok(())` if the transition was successful, `Err(UtlError)` if the
    /// transition is invalid (backward transition).
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::config::LifecycleConfig;
    /// use context_graph_utl::lifecycle::{LifecycleManager, LifecycleStage};
    ///
    /// let config = LifecycleConfig::default();
    /// let mut manager = LifecycleManager::new(&config);
    ///
    /// // Force transition to Growth
    /// manager.transition_to(LifecycleStage::Growth).unwrap();
    /// assert_eq!(manager.current_stage(), LifecycleStage::Growth);
    ///
    /// // Backward transition fails
    /// let result = manager.transition_to(LifecycleStage::Infancy);
    /// assert!(result.is_err());
    /// ```
    pub fn transition_to(&mut self, stage: LifecycleStage) -> UtlResult<()> {
        if !self.current_stage.can_transition_to(stage) {
            return Err(UtlError::invalid_transition(
                self.current_stage.to_string(),
                stage.to_string(),
                "Cannot transition to an earlier lifecycle stage",
            ));
        }

        self.current_stage = stage;
        self.last_transition_count = self.interaction_count;

        // Update interaction count to be at least at the stage minimum
        let (min, _) = stage.interaction_range();
        if self.interaction_count < min {
            self.interaction_count = min;
        }

        Ok(())
    }

    /// Reset the manager to initial state.
    ///
    /// Sets interaction count to 0 and stage to Infancy.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::config::LifecycleConfig;
    /// use context_graph_utl::lifecycle::{LifecycleManager, LifecycleStage};
    ///
    /// let config = LifecycleConfig::default();
    /// let mut manager = LifecycleManager::with_count(&config, 500);
    ///
    /// manager.reset();
    /// assert_eq!(manager.interaction_count(), 0);
    /// assert_eq!(manager.current_stage(), LifecycleStage::Infancy);
    /// ```
    pub fn reset(&mut self) {
        self.interaction_count = 0;
        self.current_stage = LifecycleStage::Infancy;
        self.last_transition_count = 0;
    }

    /// Get the current stance description.
    ///
    /// The stance describes the learning strategy for the current stage.
    ///
    /// # Returns
    ///
    /// Stance string: "capture-novelty", "balanced", or "curation-coherence".
    #[inline]
    pub fn current_stance(&self) -> &'static str {
        self.current_stage.stance()
    }

    /// Check if automatic transitions are enabled.
    #[inline]
    pub fn is_auto_transition_enabled(&self) -> bool {
        self.auto_transition
    }

    /// Enable or disable automatic transitions.
    pub fn set_auto_transition(&mut self, enabled: bool) {
        self.auto_transition = enabled;
    }

    /// Check if smooth transitions are enabled.
    #[inline]
    pub fn is_smooth_transitions_enabled(&self) -> bool {
        self.smooth_transitions
    }

    /// Enable or disable smooth transitions.
    pub fn set_smooth_transitions(&mut self, enabled: bool) {
        self.smooth_transitions = enabled;
    }

    /// Get progress within the current stage as a percentage.
    ///
    /// # Returns
    ///
    /// Progress percentage (0.0 to 1.0) through the current stage.
    /// For Maturity stage (unbounded), returns 1.0 if at or past 500,
    /// with gradual increase beyond.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::config::LifecycleConfig;
    /// use context_graph_utl::lifecycle::LifecycleManager;
    ///
    /// let config = LifecycleConfig::default();
    /// let manager = LifecycleManager::with_count(&config, 25);
    ///
    /// // 25 out of 50 = 50% through Infancy
    /// assert!((manager.stage_progress() - 0.5).abs() < 0.01);
    /// ```
    pub fn stage_progress(&self) -> f64 {
        let (min, max) = self.current_stage.interaction_range();

        if max == u64::MAX {
            // Maturity stage - use asymptotic progress
            let beyond_min = self.interaction_count.saturating_sub(min) as f64;
            1.0 - (1.0 / (1.0 + beyond_min / 500.0))
        } else {
            let range = (max - min) as f64;
            let progress = (self.interaction_count.saturating_sub(min)) as f64;
            (progress / range).min(1.0)
        }
    }

    /// Get the number of interactions until the next stage transition.
    ///
    /// # Returns
    ///
    /// Number of interactions remaining, or `None` if already in Maturity.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::config::LifecycleConfig;
    /// use context_graph_utl::lifecycle::LifecycleManager;
    ///
    /// let config = LifecycleConfig::default();
    /// let manager = LifecycleManager::with_count(&config, 40);
    ///
    /// // 10 interactions until Growth (at 50)
    /// assert_eq!(manager.interactions_until_next_stage(), Some(10));
    /// ```
    pub fn interactions_until_next_stage(&self) -> Option<u64> {
        let (_, max) = self.current_stage.interaction_range();

        if max == u64::MAX {
            None // Already in final stage
        } else {
            Some(max.saturating_sub(self.interaction_count))
        }
    }

    /// Get a summary of the current lifecycle state.
    ///
    /// # Returns
    ///
    /// A human-readable summary string.
    pub fn summary(&self) -> String {
        format!(
            "Stage: {} | Interactions: {} | Stance: {} | Progress: {:.1}%",
            self.current_stage,
            self.interaction_count,
            self.current_stance(),
            self.stage_progress() * 100.0
        )
    }
}

impl Default for LifecycleManager {
    fn default() -> Self {
        Self::new(&LifecycleConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> LifecycleConfig {
        LifecycleConfig::default()
    }

    #[test]
    fn test_new() {
        let config = test_config();
        let manager = LifecycleManager::new(&config);

        assert_eq!(manager.interaction_count(), 0);
        assert_eq!(manager.current_stage(), LifecycleStage::Infancy);
        assert!(manager.is_auto_transition_enabled());
    }

    #[test]
    fn test_with_count() {
        let config = test_config();

        let manager = LifecycleManager::with_count(&config, 100);
        assert_eq!(manager.interaction_count(), 100);
        assert_eq!(manager.current_stage(), LifecycleStage::Growth);

        let manager = LifecycleManager::with_count(&config, 600);
        assert_eq!(manager.current_stage(), LifecycleStage::Maturity);
    }

    #[test]
    fn test_increment() {
        let config = test_config();
        let mut manager = LifecycleManager::new(&config);

        for _ in 0..49 {
            let transitioned = manager.increment();
            assert!(!transitioned);
            assert_eq!(manager.current_stage(), LifecycleStage::Infancy);
        }

        // At 49 interactions, still Infancy
        assert_eq!(manager.interaction_count(), 49);
    }

    #[test]
    fn test_auto_transition_to_growth() {
        let mut config = test_config();
        config.transition_hysteresis = 0; // Disable hysteresis for test
        let mut manager = LifecycleManager::new(&config);

        // Increment to 50
        for _ in 0..50 {
            manager.increment();
        }

        assert_eq!(manager.current_stage(), LifecycleStage::Growth);
        assert_eq!(manager.interaction_count(), 50);
    }

    #[test]
    fn test_auto_transition_to_maturity() {
        let mut config = test_config();
        config.transition_hysteresis = 0;
        let mut manager = LifecycleManager::new(&config);

        manager.increment_by(500);

        assert_eq!(manager.current_stage(), LifecycleStage::Maturity);
        assert_eq!(manager.interaction_count(), 500);
    }

    #[test]
    fn test_increment_by() {
        let mut config = test_config();
        config.transition_hysteresis = 0;
        let mut manager = LifecycleManager::new(&config);

        let transitioned = manager.increment_by(100);
        assert!(transitioned);
        assert_eq!(manager.interaction_count(), 100);
        assert_eq!(manager.current_stage(), LifecycleStage::Growth);
    }

    #[test]
    fn test_manual_transition() {
        let config = test_config();
        let mut manager = LifecycleManager::new(&config);

        // Forward transition should succeed
        assert!(manager.transition_to(LifecycleStage::Growth).is_ok());
        assert_eq!(manager.current_stage(), LifecycleStage::Growth);
        assert!(manager.interaction_count() >= 50); // Count updated to stage minimum

        // Skip to Maturity should also succeed
        assert!(manager.transition_to(LifecycleStage::Maturity).is_ok());
        assert_eq!(manager.current_stage(), LifecycleStage::Maturity);
    }

    #[test]
    fn test_invalid_backward_transition() {
        let config = test_config();
        let mut manager = LifecycleManager::with_count(&config, 100);

        assert_eq!(manager.current_stage(), LifecycleStage::Growth);

        // Backward transition should fail
        let result = manager.transition_to(LifecycleStage::Infancy);
        assert!(result.is_err());

        match result.unwrap_err() {
            UtlError::InvalidLifecycleTransition { from, to, .. } => {
                assert_eq!(from, "Growth");
                assert_eq!(to, "Infancy");
            }
            _ => panic!("Expected InvalidLifecycleTransition error"),
        }
    }

    #[test]
    fn test_same_stage_transition() {
        let config = test_config();
        let mut manager = LifecycleManager::with_count(&config, 100);

        // Transition to same stage should succeed
        assert!(manager.transition_to(LifecycleStage::Growth).is_ok());
    }

    #[test]
    fn test_current_weights() {
        let config = test_config();

        let manager = LifecycleManager::new(&config);
        let weights = manager.current_weights();
        assert!((weights.lambda_s() - 0.7).abs() < 0.001);

        let manager = LifecycleManager::with_count(&config, 100);
        let weights = manager.current_weights();
        assert!((weights.lambda_s() - 0.5).abs() < 0.001);

        let manager = LifecycleManager::with_count(&config, 600);
        let weights = manager.current_weights();
        assert!((weights.lambda_s() - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_interpolated_weights() {
        let config = test_config();
        let manager = LifecycleManager::with_count(&config, 55);

        let weights = manager.interpolated_weights(&config);
        // Should be between Infancy (0.7) and Growth (0.5)
        assert!(weights.lambda_s() >= 0.5);
        assert!(weights.lambda_s() <= 0.7);
    }

    #[test]
    fn test_reset() {
        let config = test_config();
        let mut manager = LifecycleManager::with_count(&config, 600);

        assert_eq!(manager.current_stage(), LifecycleStage::Maturity);

        manager.reset();

        assert_eq!(manager.interaction_count(), 0);
        assert_eq!(manager.current_stage(), LifecycleStage::Infancy);
    }

    #[test]
    fn test_current_stance() {
        let config = test_config();

        let manager = LifecycleManager::new(&config);
        assert_eq!(manager.current_stance(), "capture-novelty");

        let manager = LifecycleManager::with_count(&config, 100);
        assert_eq!(manager.current_stance(), "balanced");

        let manager = LifecycleManager::with_count(&config, 600);
        assert_eq!(manager.current_stance(), "curation-coherence");
    }

    #[test]
    fn test_auto_transition_toggle() {
        let config = test_config();
        let mut manager = LifecycleManager::new(&config);

        assert!(manager.is_auto_transition_enabled());

        manager.set_auto_transition(false);
        assert!(!manager.is_auto_transition_enabled());

        // With auto transition disabled, stage shouldn't change
        manager.increment_by(100);
        assert_eq!(manager.current_stage(), LifecycleStage::Infancy); // Still Infancy
    }

    #[test]
    fn test_smooth_transitions_toggle() {
        let config = test_config();
        let mut manager = LifecycleManager::new(&config);

        assert!(manager.is_smooth_transitions_enabled());

        manager.set_smooth_transitions(false);
        assert!(!manager.is_smooth_transitions_enabled());
    }

    #[test]
    fn test_stage_progress() {
        let config = test_config();

        // 25 out of 50 = 50% through Infancy
        let manager = LifecycleManager::with_count(&config, 25);
        assert!((manager.stage_progress() - 0.5).abs() < 0.01);

        // 0 out of 50 = 0% through Infancy
        let manager = LifecycleManager::new(&config);
        assert!((manager.stage_progress() - 0.0).abs() < 0.01);

        // 275 out of 450 range (50-500) = ~50% through Growth
        let manager = LifecycleManager::with_count(&config, 275);
        assert!((manager.stage_progress() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_interactions_until_next_stage() {
        let config = test_config();

        let manager = LifecycleManager::with_count(&config, 40);
        assert_eq!(manager.interactions_until_next_stage(), Some(10)); // 50 - 40

        let manager = LifecycleManager::with_count(&config, 400);
        assert_eq!(manager.interactions_until_next_stage(), Some(100)); // 500 - 400

        let manager = LifecycleManager::with_count(&config, 600);
        assert_eq!(manager.interactions_until_next_stage(), None); // Already in Maturity
    }

    #[test]
    fn test_summary() {
        let config = test_config();
        let manager = LifecycleManager::with_count(&config, 100);

        let summary = manager.summary();
        assert!(summary.contains("Growth"));
        assert!(summary.contains("100"));
        assert!(summary.contains("balanced"));
    }

    #[test]
    fn test_default() {
        let manager = LifecycleManager::default();

        assert_eq!(manager.interaction_count(), 0);
        assert_eq!(manager.current_stage(), LifecycleStage::Infancy);
    }

    #[test]
    fn test_serialization() {
        let config = test_config();
        let manager = LifecycleManager::with_count(&config, 100);

        let json = serde_json::to_string(&manager).unwrap();
        let deserialized: LifecycleManager = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.interaction_count(), 100);
        assert_eq!(deserialized.current_stage(), LifecycleStage::Growth);
    }

    #[test]
    fn test_hysteresis() {
        let mut config = test_config();
        config.transition_hysteresis = 10;
        let mut manager = LifecycleManager::with_count(&config, 45);

        // At 45, we're in Infancy
        assert_eq!(manager.current_stage(), LifecycleStage::Infancy);

        // Increment to 50 - but hysteresis prevents immediate transition
        manager.increment_by(5);
        assert_eq!(manager.interaction_count(), 50);
        // Due to hysteresis, we might still be in Infancy
        // (depends on last_transition_count)

        // Continue incrementing until past hysteresis
        manager.increment_by(10);
        // Now we should have transitioned
        assert_eq!(manager.current_stage(), LifecycleStage::Growth);
    }

    #[test]
    fn test_clone() {
        let config = test_config();
        let manager = LifecycleManager::with_count(&config, 100);
        let cloned = manager.clone();

        assert_eq!(cloned.interaction_count(), 100);
        assert_eq!(cloned.current_stage(), LifecycleStage::Growth);
    }

    #[test]
    fn test_debug() {
        let config = test_config();
        let manager = LifecycleManager::new(&config);
        let debug = format!("{:?}", manager);

        assert!(debug.contains("LifecycleManager"));
        assert!(debug.contains("interaction_count"));
        assert!(debug.contains("current_stage"));
    }

    #[test]
    fn test_full_lifecycle_progression() {
        let mut config = test_config();
        config.transition_hysteresis = 0;
        let mut manager = LifecycleManager::new(&config);

        // Start in Infancy
        assert_eq!(manager.current_stage(), LifecycleStage::Infancy);

        // Progress through Infancy
        for i in 1..50 {
            manager.increment();
            assert_eq!(manager.interaction_count(), i);
            assert_eq!(manager.current_stage(), LifecycleStage::Infancy);
        }

        // Transition to Growth at 50
        manager.increment();
        assert_eq!(manager.interaction_count(), 50);
        assert_eq!(manager.current_stage(), LifecycleStage::Growth);

        // Progress through Growth
        manager.increment_by(449);
        assert_eq!(manager.interaction_count(), 499);
        assert_eq!(manager.current_stage(), LifecycleStage::Growth);

        // Transition to Maturity at 500
        manager.increment();
        assert_eq!(manager.interaction_count(), 500);
        assert_eq!(manager.current_stage(), LifecycleStage::Maturity);

        // Stay in Maturity indefinitely
        manager.increment_by(1000);
        assert_eq!(manager.interaction_count(), 1500);
        assert_eq!(manager.current_stage(), LifecycleStage::Maturity);
    }
}
