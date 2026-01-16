//! Autonomous system status types.
//!
//! TASK-P0-005: Renamed north_star_configured to strategic_goal_configured per ARCH-03.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::autonomous::drift::DriftState;
use crate::autonomous::thresholds::AdaptiveThresholdState;

use super::health::AutonomousHealth;

/// Autonomous system status
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutonomousStatus {
    /// Whether autonomous mode is enabled
    pub enabled: bool,
    /// Whether bootstrap has completed
    pub bootstrap_complete: bool,
    /// Whether a Strategic goal is configured
    pub strategic_goal_configured: bool,
    /// Current drift detection state
    pub drift_state: DriftState,
    /// Current adaptive threshold state
    pub threshold_state: AdaptiveThresholdState,
    /// Number of memories pending pruning review
    pub pending_prune_count: u32,
    /// Number of memories in consolidation queue
    pub consolidation_queue_size: u32,
    /// Last optimization timestamp
    pub last_optimization: DateTime<Utc>,
    /// Next scheduled operation
    pub next_scheduled: Option<DateTime<Utc>>,
    /// System health status
    pub health: AutonomousHealth,
}

impl Default for AutonomousStatus {
    fn default() -> Self {
        Self {
            enabled: false,
            bootstrap_complete: false,
            strategic_goal_configured: false,
            drift_state: DriftState::default(),
            threshold_state: AdaptiveThresholdState::default(),
            pending_prune_count: 0,
            consolidation_queue_size: 0,
            last_optimization: Utc::now(),
            next_scheduled: None,
            health: AutonomousHealth::default(),
        }
    }
}

impl AutonomousStatus {
    /// Create a status for a fully initialized system
    pub fn initialized(strategic_goal_configured: bool) -> Self {
        Self {
            enabled: true,
            bootstrap_complete: true,
            strategic_goal_configured,
            drift_state: DriftState::default(),
            threshold_state: AdaptiveThresholdState::default(),
            pending_prune_count: 0,
            consolidation_queue_size: 0,
            last_optimization: Utc::now(),
            next_scheduled: None,
            health: AutonomousHealth::Healthy,
        }
    }

    /// Check if the system is ready for autonomous operations
    pub fn is_ready(&self) -> bool {
        self.enabled && self.bootstrap_complete && self.health.can_continue()
    }

    /// Check if there is pending work
    pub fn has_pending_work(&self) -> bool {
        self.pending_prune_count > 0
            || self.consolidation_queue_size > 0
            || self.drift_state.requires_attention()
    }

    /// Get a summary of the current status
    pub fn summary(&self) -> String {
        let health_str = match &self.health {
            AutonomousHealth::Healthy => "healthy",
            AutonomousHealth::Warning { .. } => "warning",
            AutonomousHealth::Error { recoverable, .. } => {
                if *recoverable {
                    "error (recoverable)"
                } else {
                    "error (fatal)"
                }
            }
        };

        format!(
            "Autonomous[enabled={}, ready={}, health={}, prune={}, consolidate={}]",
            self.enabled,
            self.is_ready(),
            health_str,
            self.pending_prune_count,
            self.consolidation_queue_size
        )
    }
}
