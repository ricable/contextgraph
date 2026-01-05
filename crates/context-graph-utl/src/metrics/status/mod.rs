//! UTL system status and MCP response types.

use serde::{Deserialize, Serialize};

use crate::lifecycle::{LifecycleLambdaWeights, LifecycleStage};
use crate::phase::ConsolidationPhase;

use super::{StageThresholds, UtlComputationMetrics};

/// Complete UTL system status for monitoring and MCP responses.
///
/// Used by the `utl_status` MCP tool and `UtlProcessor::get_status()`.
///
/// # Example
///
/// ```
/// use context_graph_utl::metrics::UtlStatus;
///
/// let status = UtlStatus::new();
/// assert_eq!(status.interaction_count, 0);
/// assert!(status.is_novelty_seeking()); // Default is Infancy stage
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UtlStatus {
    /// Current lifecycle stage (Infancy, Growth, Maturity)
    pub lifecycle_stage: LifecycleStage,

    /// Total interaction count (determines lifecycle stage)
    pub interaction_count: u64,

    /// Current thresholds for the lifecycle stage
    pub current_thresholds: StageThresholds,

    /// Current Marblestone lambda weights
    pub lambda_weights: LifecycleLambdaWeights,

    /// Current phase oscillator angle [0, pi]
    pub phase_angle: f32,

    /// Current consolidation phase (NREM, REM, Wake)
    pub consolidation_phase: ConsolidationPhase,

    /// Accumulated computation metrics
    pub metrics: UtlComputationMetrics,
}

impl Default for UtlStatus {
    fn default() -> Self {
        Self {
            lifecycle_stage: LifecycleStage::default(),
            interaction_count: 0,
            current_thresholds: StageThresholds::default(),
            lambda_weights: LifecycleLambdaWeights::default(),
            phase_angle: 0.0,
            consolidation_phase: ConsolidationPhase::default(),
            metrics: UtlComputationMetrics::default(),
        }
    }
}

impl UtlStatus {
    /// Create new default status.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if system is in encoding phase (NREM).
    pub fn is_encoding(&self) -> bool {
        matches!(self.consolidation_phase, ConsolidationPhase::NREM)
    }

    /// Check if system is in consolidation/integration phase (REM).
    pub fn is_consolidating(&self) -> bool {
        matches!(self.consolidation_phase, ConsolidationPhase::REM)
    }

    /// Check if system is in active wake phase.
    pub fn is_wake(&self) -> bool {
        matches!(self.consolidation_phase, ConsolidationPhase::Wake)
    }

    /// Check if system favors novelty (Infancy stage).
    pub fn is_novelty_seeking(&self) -> bool {
        matches!(self.lifecycle_stage, LifecycleStage::Infancy)
    }

    /// Check if system favors consolidation (Maturity stage).
    pub fn is_consolidation_focused(&self) -> bool {
        matches!(self.lifecycle_stage, LifecycleStage::Maturity)
    }

    /// Check if system is in balanced Growth stage.
    pub fn is_balanced(&self) -> bool {
        matches!(self.lifecycle_stage, LifecycleStage::Growth)
    }

    /// Get a summary string for logging.
    pub fn summary(&self) -> String {
        format!(
            "UTL: stage={:?}, interactions={}, phase={:?}, avg_L={:.3}",
            self.lifecycle_stage,
            self.interaction_count,
            self.consolidation_phase,
            self.metrics.avg_learning_magnitude
        )
    }

    /// Convert to MCP response format.
    pub fn to_mcp_response(&self) -> UtlStatusResponse {
        UtlStatusResponse {
            lifecycle_phase: format!("{:?}", self.lifecycle_stage),
            interaction_count: self.interaction_count,
            entropy: self.metrics.avg_delta_s,
            coherence: self.metrics.avg_delta_c,
            learning_score: self.metrics.avg_learning_magnitude,
            johari_quadrant: format!("{:?}", self.metrics.dominant_quadrant()),
            consolidation_phase: format!("{:?}", self.consolidation_phase),
            phase_angle: self.phase_angle,
            thresholds: ThresholdsResponse::from(&self.current_thresholds),
        }
    }
}

/// MCP response format for utl_status tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UtlStatusResponse {
    /// Current lifecycle phase name
    pub lifecycle_phase: String,

    /// Total interaction count
    pub interaction_count: u64,

    /// Average entropy (delta_s)
    pub entropy: f32,

    /// Average coherence (delta_c)
    pub coherence: f32,

    /// Average learning score
    pub learning_score: f32,

    /// Dominant Johari quadrant name
    pub johari_quadrant: String,

    /// Current consolidation phase name
    pub consolidation_phase: String,

    /// Phase oscillator angle
    pub phase_angle: f32,

    /// Current thresholds
    pub thresholds: ThresholdsResponse,
}

/// Thresholds in MCP response format.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ThresholdsResponse {
    pub entropy_trigger: f32,
    pub coherence_trigger: f32,
    pub min_importance_store: f32,
    pub consolidation_threshold: f32,
}

impl From<&StageThresholds> for ThresholdsResponse {
    fn from(thresholds: &StageThresholds) -> Self {
        Self {
            entropy_trigger: thresholds.entropy_trigger,
            coherence_trigger: thresholds.coherence_trigger,
            min_importance_store: thresholds.min_importance_store,
            consolidation_threshold: thresholds.consolidation_threshold,
        }
    }
}

#[cfg(test)]
mod tests;
