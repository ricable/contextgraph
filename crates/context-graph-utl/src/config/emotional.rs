//! Emotional weight (w_e) settings.
//!
//! Controls how emotional salience affects learning score computation.
//! Emotional weight modulates the overall learning signal.

use serde::{Deserialize, Serialize};

/// Emotional weight (w_e) settings.
///
/// Controls how emotional salience affects learning score computation.
/// Emotional weight modulates the overall learning signal.
///
/// # Constitution Reference
///
/// - `w_e` range: `[0.5, 1.5]` representing emotional weight
/// - Default value is `1.0` (neutral)
/// - Values `> 1.0` amplify learning, values `< 1.0` dampen it
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalConfig {
    /// Minimum emotional weight.
    /// Constitution specifies: `0.5`
    pub min_weight: f32,

    /// Maximum emotional weight.
    /// Constitution specifies: `1.5`
    pub max_weight: f32,

    /// Default/neutral emotional weight.
    pub default_weight: f32,

    /// Decay rate for emotional salience over time.
    /// Range: `[0.0, 1.0]`
    pub decay_rate: f32,

    /// Enable arousal-based modulation.
    pub arousal_modulation: bool,

    /// Arousal sensitivity factor.
    /// Range: `[0.0, 2.0]`
    pub arousal_sensitivity: f32,

    /// Enable valence-based modulation.
    pub valence_modulation: bool,

    /// Valence sensitivity factor.
    /// Range: `[0.0, 2.0]`
    pub valence_sensitivity: f32,
}

impl Default for EmotionalConfig {
    fn default() -> Self {
        Self {
            min_weight: 0.5,
            max_weight: 1.5,
            default_weight: 1.0,
            decay_rate: 0.02,
            arousal_modulation: true,
            arousal_sensitivity: 1.0,
            valence_modulation: true,
            valence_sensitivity: 1.0,
        }
    }
}

impl EmotionalConfig {
    /// Validate the emotional configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.min_weight < 0.0 {
            return Err(format!("min_weight must be >= 0, got {}", self.min_weight));
        }
        if self.max_weight < self.min_weight {
            return Err(format!(
                "max_weight ({}) must be >= min_weight ({})",
                self.max_weight, self.min_weight
            ));
        }
        if !(self.min_weight..=self.max_weight).contains(&self.default_weight) {
            return Err(format!(
                "default_weight must be in [{}, {}], got {}",
                self.min_weight, self.max_weight, self.default_weight
            ));
        }
        if !(0.0..=1.0).contains(&self.decay_rate) {
            return Err(format!(
                "decay_rate must be in [0, 1], got {}",
                self.decay_rate
            ));
        }
        if !(0.0..=2.0).contains(&self.arousal_sensitivity) {
            return Err(format!(
                "arousal_sensitivity must be in [0, 2], got {}",
                self.arousal_sensitivity
            ));
        }
        if !(0.0..=2.0).contains(&self.valence_sensitivity) {
            return Err(format!(
                "valence_sensitivity must be in [0, 2], got {}",
                self.valence_sensitivity
            ));
        }
        Ok(())
    }

    /// Clamp a weight value to the valid range.
    pub fn clamp(&self, weight: f32) -> f32 {
        weight.clamp(self.min_weight, self.max_weight)
    }
}
