//! UTL configuration types.
//!
//! This module defines comprehensive configuration types for the UTL (Unified Theory of Learning)
//! computation engine, including surprise, coherence, emotional weighting, phase oscillation,
//! lifecycle stages, Johari classification, and KL divergence settings.
//!
//! # Constitution Reference
//!
//! The UTL formula is: `L = f((delta-S * delta-C) * w_e * cos phi)`
//!
//! Where:
//! - `delta-S`: Entropy/novelty change in range `[0, 1]`
//! - `delta-C`: Coherence/understanding change in range `[0, 1]`
//! - `w_e`: Emotional weight in range `[0.5, 1.5]`
//! - `phi`: Phase synchronization angle in range `[0, pi]`

mod coherence;
mod emotional;
mod johari;
mod kl;
mod lifecycle;
mod phase;
mod surprise;
mod thresholds;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use self::coherence::CoherenceConfig;
pub use self::emotional::EmotionalConfig;
pub use self::johari::JohariConfig;
pub use self::kl::KlConfig;
pub use self::lifecycle::{LifecycleConfig, StageConfig};
pub use self::phase::PhaseConfig;
pub use self::surprise::SurpriseConfig;
pub use self::thresholds::UtlThresholds;

use serde::{Deserialize, Serialize};

/// Main UTL configuration containing all subsystem settings.
///
/// This is the top-level configuration struct that aggregates all UTL-related
/// configuration options. It provides sensible defaults based on the constitution.yaml
/// specifications.
///
/// # Example
///
/// ```
/// use context_graph_utl::config::UtlConfig;
///
/// let config = UtlConfig::default();
/// assert!(config.surprise.entropy_weight > 0.0);
/// assert!(config.lifecycle.stages.len() == 3);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UtlConfig {
    /// Surprise computation settings.
    pub surprise: SurpriseConfig,

    /// Coherence computation settings.
    pub coherence: CoherenceConfig,

    /// Emotional weight settings.
    pub emotional: EmotionalConfig,

    /// Phase oscillation settings.
    pub phase: PhaseConfig,

    /// Johari quadrant classification thresholds.
    pub johari: JohariConfig,

    /// Lifecycle stage configurations.
    pub lifecycle: LifecycleConfig,

    /// KL divergence settings.
    pub kl: KlConfig,

    /// UTL computation thresholds.
    pub thresholds: UtlThresholds,

    /// Enable debug logging for UTL computations.
    #[serde(default)]
    pub debug: bool,
}

impl UtlConfig {
    /// Create a new configuration with custom settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration optimized for high-novelty exploration.
    ///
    /// This preset emphasizes surprise detection over coherence,
    /// suitable for early-stage learning or discovery tasks.
    pub fn exploration_preset() -> Self {
        Self {
            surprise: SurpriseConfig {
                entropy_weight: 0.8,
                novelty_boost: 1.2,
                ..Default::default()
            },
            lifecycle: LifecycleConfig::infancy_focused(),
            ..Default::default()
        }
    }

    /// Create a configuration optimized for coherence-focused curation.
    ///
    /// This preset emphasizes coherence and consolidation over novelty,
    /// suitable for mature knowledge bases.
    pub fn curation_preset() -> Self {
        Self {
            coherence: CoherenceConfig {
                similarity_weight: 0.8,
                consistency_weight: 0.9,
                ..Default::default()
            },
            lifecycle: LifecycleConfig::maturity_focused(),
            ..Default::default()
        }
    }

    /// Validate the configuration, returning an error if invalid.
    pub fn validate(&self) -> Result<(), String> {
        self.surprise.validate()?;
        self.coherence.validate()?;
        self.emotional.validate()?;
        self.phase.validate()?;
        self.johari.validate()?;
        self.lifecycle.validate()?;
        self.kl.validate()?;
        self.thresholds.validate()?;
        Ok(())
    }
}
