//! Lifecycle stage transitions.
//!
//! This module handles transitioning between lifecycle stages, including
//! automatic transitions based on interaction count and manual transitions.

use crate::error::{UtlError, UtlResult};

use super::super::stage::LifecycleStage;
use super::types::LifecycleManager;

impl LifecycleManager {
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
    pub(crate) fn check_transition(&mut self) -> bool {
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
}
