//! Lifecycle stage configurations.
//!
//! Knowledge bases evolve through distinct lifecycle stages, each with
//! different learning dynamics as specified by the Marblestone lambda weights.

use serde::{Deserialize, Serialize};

/// Lifecycle stage configurations.
///
/// Knowledge bases evolve through distinct lifecycle stages, each with
/// different learning dynamics as specified by the Marblestone lambda weights.
///
/// # Constitution Reference
///
/// ```text
/// Infancy (n=0-50):   lambda_dS=0.7, lambda_dC=0.3, stance="capture-novelty"
/// Growth (n=50-500):  lambda_dS=0.5, lambda_dC=0.5, stance="balanced"
/// Maturity (n=500+):  lambda_dS=0.3, lambda_dC=0.7, stance="curation-coherence"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleConfig {
    /// Stage configurations in order: [Infancy, Growth, Maturity].
    pub stages: Vec<StageConfig>,

    /// Enable automatic stage transitions.
    pub auto_transition: bool,

    /// Hysteresis for stage transitions (prevents rapid switching).
    pub transition_hysteresis: u64,

    /// Enable smooth interpolation between stages.
    pub smooth_transitions: bool,

    /// Transition smoothing window (number of interactions).
    pub smoothing_window: u64,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            stages: vec![
                StageConfig::infancy(),
                StageConfig::growth(),
                StageConfig::maturity(),
            ],
            auto_transition: true,
            transition_hysteresis: 10,
            smooth_transitions: true,
            smoothing_window: 25,
        }
    }
}

impl LifecycleConfig {
    /// Create lifecycle config focused on infancy stage.
    pub fn infancy_focused() -> Self {
        let mut config = Self::default();
        config.stages[0].lambda_novelty = 0.8;
        config.stages[0].lambda_consolidation = 0.2;
        config
    }

    /// Create lifecycle config focused on maturity stage.
    pub fn maturity_focused() -> Self {
        let mut config = Self::default();
        config.stages[2].lambda_novelty = 0.2;
        config.stages[2].lambda_consolidation = 0.8;
        config
    }

    /// Get stage configuration for a given interaction count.
    pub fn get_stage(&self, interaction_count: u64) -> Option<&StageConfig> {
        for stage in self.stages.iter().rev() {
            if interaction_count >= stage.min_interactions {
                return Some(stage);
            }
        }
        self.stages.first()
    }

    /// Validate the lifecycle configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.stages.is_empty() {
            return Err("At least one lifecycle stage is required".to_string());
        }
        for (i, stage) in self.stages.iter().enumerate() {
            stage
                .validate()
                .map_err(|e| format!("Stage {}: {}", i, e))?;
        }
        // Verify stages are in order
        for window in self.stages.windows(2) {
            if window[1].min_interactions <= window[0].min_interactions {
                return Err(format!(
                    "Stages must have increasing min_interactions: {} <= {}",
                    window[1].min_interactions, window[0].min_interactions
                ));
            }
        }
        Ok(())
    }
}

/// Individual lifecycle stage configuration.
///
/// Each stage defines different lambda weights for balancing novelty vs. consolidation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageConfig {
    /// Stage name (e.g., "Infancy", "Growth", "Maturity").
    pub name: String,

    /// Minimum interaction count to enter this stage.
    pub min_interactions: u64,

    /// Maximum interaction count (exclusive) for this stage.
    /// Use `u64::MAX` for the final stage.
    pub max_interactions: u64,

    /// Lambda weight for novelty/surprise (lambda_dS).
    /// Range: `[0.0, 1.0]`
    pub lambda_novelty: f32,

    /// Lambda weight for consolidation/coherence (lambda_dC).
    /// Range: `[0.0, 1.0]`
    /// Note: lambda_novelty + lambda_consolidation should equal 1.0
    pub lambda_consolidation: f32,

    /// Surprise trigger threshold (dS_trig).
    /// Range: `[0.0, 1.0]`
    pub surprise_trigger: f32,

    /// Coherence trigger threshold (dC_trig).
    /// Range: `[0.0, 1.0]`
    pub coherence_trigger: f32,

    /// Stage stance description.
    pub stance: String,
}

impl StageConfig {
    /// Create infancy stage configuration (0-50 interactions).
    pub fn infancy() -> Self {
        Self {
            name: "Infancy".to_string(),
            min_interactions: 0,
            max_interactions: 50,
            lambda_novelty: 0.7,
            lambda_consolidation: 0.3,
            surprise_trigger: 0.9,
            coherence_trigger: 0.2,
            stance: "capture-novelty".to_string(),
        }
    }

    /// Create growth stage configuration (50-500 interactions).
    pub fn growth() -> Self {
        Self {
            name: "Growth".to_string(),
            min_interactions: 50,
            max_interactions: 500,
            lambda_novelty: 0.5,
            lambda_consolidation: 0.5,
            surprise_trigger: 0.7,
            coherence_trigger: 0.4,
            stance: "balanced".to_string(),
        }
    }

    /// Create maturity stage configuration (500+ interactions).
    pub fn maturity() -> Self {
        Self {
            name: "Maturity".to_string(),
            min_interactions: 500,
            max_interactions: u64::MAX,
            lambda_novelty: 0.3,
            lambda_consolidation: 0.7,
            surprise_trigger: 0.6,
            coherence_trigger: 0.5,
            stance: "curation-coherence".to_string(),
        }
    }

    /// Validate the stage configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("Stage name cannot be empty".to_string());
        }
        if self.max_interactions <= self.min_interactions {
            return Err(format!(
                "max_interactions ({}) must be > min_interactions ({})",
                self.max_interactions, self.min_interactions
            ));
        }
        if !(0.0..=1.0).contains(&self.lambda_novelty) {
            return Err(format!(
                "lambda_novelty must be in [0, 1], got {}",
                self.lambda_novelty
            ));
        }
        if !(0.0..=1.0).contains(&self.lambda_consolidation) {
            return Err(format!(
                "lambda_consolidation must be in [0, 1], got {}",
                self.lambda_consolidation
            ));
        }
        let lambda_sum = self.lambda_novelty + self.lambda_consolidation;
        if (lambda_sum - 1.0).abs() > 0.001 {
            return Err(format!(
                "lambda_novelty + lambda_consolidation must equal 1.0, got {}",
                lambda_sum
            ));
        }
        if !(0.0..=1.0).contains(&self.surprise_trigger) {
            return Err(format!(
                "surprise_trigger must be in [0, 1], got {}",
                self.surprise_trigger
            ));
        }
        if !(0.0..=1.0).contains(&self.coherence_trigger) {
            return Err(format!(
                "coherence_trigger must be in [0, 1], got {}",
                self.coherence_trigger
            ));
        }
        Ok(())
    }
}
