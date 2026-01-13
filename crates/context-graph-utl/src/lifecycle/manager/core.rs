//! Core lifecycle manager functionality.
//!
//! This module contains constructors and basic accessors for the
//! `LifecycleManager`.
//!
//! TASK-METAUTL-P0-006: Added lambda override support for meta-learning integration.

use crate::config::LifecycleConfig;

use super::super::lambda::LifecycleLambdaWeights;
use super::super::stage::LifecycleStage;
use super::types::LifecycleManager;

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
            lambda_override: None, // TASK-METAUTL-P0-006
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
            lambda_override: None, // TASK-METAUTL-P0-006
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

    // ========================================================================
    // TASK-METAUTL-P0-006: Lambda Override Methods
    // ========================================================================

    /// Set lambda weight override from meta-learning correction.
    ///
    /// TASK-METAUTL-P0-006: When set, `get_effective_weights()` returns the
    /// override instead of the lifecycle-determined weights.
    ///
    /// # Arguments
    ///
    /// * `override_weights` - Weights to use instead of lifecycle defaults
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::config::LifecycleConfig;
    /// use context_graph_utl::lifecycle::{LifecycleManager, LifecycleLambdaWeights};
    ///
    /// let config = LifecycleConfig::default();
    /// let mut manager = LifecycleManager::new(&config);
    ///
    /// // Set override weights
    /// let override_weights = LifecycleLambdaWeights::new(0.6, 0.4).unwrap();
    /// manager.set_lambda_override(override_weights);
    ///
    /// assert!(manager.has_lambda_override());
    /// assert!((manager.get_effective_weights().lambda_s() - 0.6).abs() < 0.001);
    /// ```
    pub fn set_lambda_override(&mut self, override_weights: LifecycleLambdaWeights) {
        self.lambda_override = Some(override_weights);
    }

    /// Clear lambda weight override.
    ///
    /// TASK-METAUTL-P0-006: Returns to using lifecycle-determined weights.
    pub fn clear_lambda_override(&mut self) {
        self.lambda_override = None;
    }

    /// Check if override is active.
    ///
    /// TASK-METAUTL-P0-006: Returns `true` if an override is currently set.
    #[inline]
    pub fn has_lambda_override(&self) -> bool {
        self.lambda_override.is_some()
    }

    /// Get current effective weights.
    ///
    /// TASK-METAUTL-P0-006: Returns override weights if set, otherwise
    /// lifecycle weights.
    ///
    /// # Returns
    ///
    /// The currently effective lambda weights.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::config::LifecycleConfig;
    /// use context_graph_utl::lifecycle::{LifecycleManager, LifecycleLambdaWeights};
    ///
    /// let config = LifecycleConfig::default();
    /// let mut manager = LifecycleManager::new(&config);
    ///
    /// // Without override: returns lifecycle weights
    /// let weights = manager.get_effective_weights();
    /// assert!((weights.lambda_s() - 0.7).abs() < 0.001); // Infancy
    ///
    /// // With override: returns override weights
    /// manager.set_lambda_override(LifecycleLambdaWeights::new(0.4, 0.6).unwrap());
    /// let weights = manager.get_effective_weights();
    /// assert!((weights.lambda_s() - 0.4).abs() < 0.001);
    /// ```
    pub fn get_effective_weights(&self) -> LifecycleLambdaWeights {
        self.lambda_override.unwrap_or_else(|| self.current_weights())
    }

    /// Get override deviation from base weights.
    ///
    /// TASK-METAUTL-P0-006: Returns (delta_s, delta_c) if override is active,
    /// (0.0, 0.0) otherwise.
    ///
    /// # Returns
    ///
    /// Tuple of (delta_lambda_s, delta_lambda_c) showing deviation from base.
    pub fn override_deviation(&self) -> (f32, f32) {
        match self.lambda_override {
            Some(override_weights) => {
                let base = self.current_weights();
                (
                    override_weights.lambda_s() - base.lambda_s(),
                    override_weights.lambda_c() - base.lambda_c(),
                )
            }
            None => (0.0, 0.0),
        }
    }
}

impl Default for LifecycleManager {
    fn default() -> Self {
        Self::new(&LifecycleConfig::default())
    }
}
