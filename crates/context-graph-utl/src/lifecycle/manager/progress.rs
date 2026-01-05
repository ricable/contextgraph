//! Progress tracking for lifecycle stages.
//!
//! This module provides methods for tracking progress through lifecycle
//! stages, including stage progress percentages and time-to-next-stage
//! calculations.

use super::types::LifecycleManager;

impl LifecycleManager {
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
