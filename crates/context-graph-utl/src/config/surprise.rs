//! Surprise (delta-S) computation settings.
//!
//! Controls how surprise/entropy is computed for knowledge items.
//! Surprise measures the novelty or unexpectedness of information.

use serde::{Deserialize, Serialize};

/// Surprise (delta-S) computation settings.
///
/// Controls how surprise/entropy is computed for knowledge items.
/// Surprise measures the novelty or unexpectedness of information.
///
/// # Constitution Reference
///
/// - `delta-S` range: `[0, 1]` representing entropy/novelty
/// - Higher values indicate more surprising/novel information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseConfig {
    /// Weight applied to entropy component.
    /// Range: `[0.0, 1.0]`
    pub entropy_weight: f32,

    /// Boost factor for novel items.
    /// Range: `[0.5, 2.0]`
    pub novelty_boost: f32,

    /// Decay rate for repeated exposure.
    /// Range: `[0.0, 1.0]`
    pub repetition_decay: f32,

    /// Minimum surprise threshold below which items are considered familiar.
    /// Range: `[0.0, 0.5]`
    pub min_threshold: f32,

    /// Maximum surprise value (for clamping).
    /// Range: `[0.5, 1.0]`
    pub max_value: f32,

    /// Number of samples for entropy estimation.
    pub sample_count: usize,

    /// Use exponential moving average for smoothing.
    pub use_ema: bool,

    /// EMA alpha (smoothing factor).
    /// Range: `[0.0, 1.0]`
    pub ema_alpha: f32,
}

impl Default for SurpriseConfig {
    fn default() -> Self {
        Self {
            entropy_weight: 0.6,
            novelty_boost: 1.0,
            repetition_decay: 0.1,
            min_threshold: 0.05,
            max_value: 1.0,
            sample_count: 100,
            use_ema: true,
            ema_alpha: 0.3,
        }
    }
}

impl SurpriseConfig {
    /// Validate the surprise configuration.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.entropy_weight) {
            return Err(format!(
                "entropy_weight must be in [0, 1], got {}",
                self.entropy_weight
            ));
        }
        if !(0.5..=2.0).contains(&self.novelty_boost) {
            return Err(format!(
                "novelty_boost must be in [0.5, 2.0], got {}",
                self.novelty_boost
            ));
        }
        if !(0.0..=1.0).contains(&self.repetition_decay) {
            return Err(format!(
                "repetition_decay must be in [0, 1], got {}",
                self.repetition_decay
            ));
        }
        if !(0.0..=0.5).contains(&self.min_threshold) {
            return Err(format!(
                "min_threshold must be in [0, 0.5], got {}",
                self.min_threshold
            ));
        }
        if !(0.5..=1.0).contains(&self.max_value) {
            return Err(format!(
                "max_value must be in [0.5, 1.0], got {}",
                self.max_value
            ));
        }
        if self.sample_count == 0 {
            return Err("sample_count must be > 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.ema_alpha) {
            return Err(format!(
                "ema_alpha must be in [0, 1], got {}",
                self.ema_alpha
            ));
        }
        Ok(())
    }
}
