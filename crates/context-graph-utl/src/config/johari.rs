//! Johari quadrant classification thresholds.
//!
//! The Johari window model classifies knowledge based on surprise and coherence.

use serde::{Deserialize, Serialize};

/// Johari quadrant classification thresholds.
///
/// The Johari window model classifies knowledge based on surprise and coherence:
/// - **Open**: Low surprise, high coherence (well-known, directly recallable)
/// - **Blind**: High surprise, low coherence (discovery opportunity)
/// - **Hidden**: Low surprise, low coherence (private/unexplored)
/// - **Unknown**: High surprise, high coherence (frontier knowledge)
///
/// # Constitution Reference
///
/// ```text
/// Open:    delta-S < 0.5, delta-C > 0.5 -> direct recall
/// Blind:   delta-S > 0.5, delta-C < 0.5 -> discovery (epistemic_action/dream)
/// Hidden:  delta-S < 0.5, delta-C < 0.5 -> private (get_neighborhood)
/// Unknown: delta-S > 0.5, delta-C > 0.5 -> frontier
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JohariConfig {
    /// Threshold for delta-S classification (surprise).
    /// Items with delta-S < threshold are "low surprise".
    pub surprise_threshold: f32,

    /// Threshold for delta-C classification (coherence).
    /// Items with delta-C > threshold are "high coherence".
    pub coherence_threshold: f32,

    /// Enable fuzzy boundaries for quadrant classification.
    pub fuzzy_boundaries: bool,

    /// Fuzzy boundary width (for smooth transitions).
    /// Range: `[0.0, 0.2]`
    pub boundary_width: f32,

    /// Weight for Open quadrant in composite scores.
    pub open_weight: f32,

    /// Weight for Blind quadrant in composite scores.
    pub blind_weight: f32,

    /// Weight for Hidden quadrant in composite scores.
    pub hidden_weight: f32,

    /// Weight for Unknown quadrant in composite scores.
    pub unknown_weight: f32,
}

impl Default for JohariConfig {
    fn default() -> Self {
        Self {
            surprise_threshold: 0.5,
            coherence_threshold: 0.5,
            fuzzy_boundaries: true,
            boundary_width: 0.1,
            open_weight: 1.0,
            blind_weight: 1.2,
            hidden_weight: 0.8,
            unknown_weight: 1.5,
        }
    }
}

impl JohariConfig {
    /// Validate the Johari configuration.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.surprise_threshold) {
            return Err(format!(
                "surprise_threshold must be in [0, 1], got {}",
                self.surprise_threshold
            ));
        }
        if !(0.0..=1.0).contains(&self.coherence_threshold) {
            return Err(format!(
                "coherence_threshold must be in [0, 1], got {}",
                self.coherence_threshold
            ));
        }
        if !(0.0..=0.2).contains(&self.boundary_width) {
            return Err(format!(
                "boundary_width must be in [0, 0.2], got {}",
                self.boundary_width
            ));
        }
        if self.open_weight < 0.0
            || self.blind_weight < 0.0
            || self.hidden_weight < 0.0
            || self.unknown_weight < 0.0
        {
            return Err("quadrant weights must be >= 0".to_string());
        }
        Ok(())
    }
}
