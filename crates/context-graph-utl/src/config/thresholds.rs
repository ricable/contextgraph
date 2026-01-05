//! UTL computation thresholds.
//!
//! Defines thresholds for various UTL computations and quality checks.

use serde::{Deserialize, Serialize};

/// UTL computation thresholds.
///
/// Defines thresholds for various UTL computations and quality checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtlThresholds {
    /// Minimum valid UTL score.
    pub min_score: f32,

    /// Maximum valid UTL score.
    pub max_score: f32,

    /// Threshold for "high quality" UTL score.
    /// Constitution target: `utl_avg > 0.6`
    pub high_quality: f32,

    /// Threshold for "low quality" requiring attention.
    pub low_quality: f32,

    /// Coherence recovery timeout (seconds).
    /// Constitution target: `< 10s`
    pub coherence_recovery_secs: u64,

    /// Information loss tolerance.
    /// Constitution target: `< 15%`
    pub info_loss_tolerance: f32,

    /// Compression target ratio.
    /// Constitution target: `> 60%`
    pub compression_target: f32,

    /// Numerical tolerance for floating point comparisons.
    pub float_tolerance: f32,

    /// NaN replacement value.
    pub nan_replacement: f32,

    /// Infinity replacement value.
    pub inf_replacement: f32,
}

impl Default for UtlThresholds {
    fn default() -> Self {
        Self {
            min_score: 0.0,
            max_score: 1.0,
            high_quality: 0.6,
            low_quality: 0.3,
            coherence_recovery_secs: 10,
            info_loss_tolerance: 0.15,
            compression_target: 0.6,
            float_tolerance: 1e-6,
            nan_replacement: 0.0,
            inf_replacement: 1.0,
        }
    }
}

impl UtlThresholds {
    /// Validate the thresholds configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.min_score >= self.max_score {
            return Err(format!(
                "min_score ({}) must be < max_score ({})",
                self.min_score, self.max_score
            ));
        }
        if !(self.min_score..=self.max_score).contains(&self.high_quality) {
            return Err(format!(
                "high_quality must be in [{}, {}], got {}",
                self.min_score, self.max_score, self.high_quality
            ));
        }
        if !(self.min_score..=self.max_score).contains(&self.low_quality) {
            return Err(format!(
                "low_quality must be in [{}, {}], got {}",
                self.min_score, self.max_score, self.low_quality
            ));
        }
        if self.low_quality >= self.high_quality {
            return Err(format!(
                "low_quality ({}) must be < high_quality ({})",
                self.low_quality, self.high_quality
            ));
        }
        if !(0.0..=1.0).contains(&self.info_loss_tolerance) {
            return Err(format!(
                "info_loss_tolerance must be in [0, 1], got {}",
                self.info_loss_tolerance
            ));
        }
        if !(0.0..=1.0).contains(&self.compression_target) {
            return Err(format!(
                "compression_target must be in [0, 1], got {}",
                self.compression_target
            ));
        }
        if self.float_tolerance <= 0.0 {
            return Err(format!(
                "float_tolerance must be > 0, got {}",
                self.float_tolerance
            ));
        }
        Ok(())
    }

    /// Clamp a score to valid range, replacing NaN/Inf.
    pub fn clamp_score(&self, score: f32) -> f32 {
        if score.is_nan() {
            self.nan_replacement
        } else if score.is_infinite() {
            if score.is_sign_positive() {
                self.inf_replacement
            } else {
                self.min_score
            }
        } else {
            score.clamp(self.min_score, self.max_score)
        }
    }

    /// Check if a score meets the high quality threshold.
    pub fn is_high_quality(&self, score: f32) -> bool {
        score >= self.high_quality
    }

    /// Check if a score is below the low quality threshold.
    pub fn is_low_quality(&self, score: f32) -> bool {
        score <= self.low_quality
    }
}
