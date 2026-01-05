//! Phase oscillation (phi) settings.
//!
//! Controls phase synchronization between different cognitive processes.
//! Phase alignment affects the coupling of learning signals.

use serde::{Deserialize, Serialize};

/// Phase oscillation (phi) settings.
///
/// Controls phase synchronization between different cognitive processes.
/// Phase alignment affects the coupling of learning signals.
///
/// # Constitution Reference
///
/// - `phi` range: `[0, pi]` representing phase angle
/// - `cos(phi) = 1.0` when fully synchronized (phi = 0)
/// - `cos(phi) = -1.0` when anti-phase (phi = pi)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseConfig {
    /// Minimum phase angle (radians).
    /// Constitution specifies: `0.0`
    pub min_phase: f32,

    /// Maximum phase angle (radians).
    /// Constitution specifies: `pi` (~3.14159)
    pub max_phase: f32,

    /// Default phase angle (radians).
    pub default_phase: f32,

    /// Phase oscillation frequency (Hz).
    /// Constitution reference: L4 operates at 100Hz
    pub frequency_hz: f32,

    /// Phase coupling strength.
    /// Range: `[0.0, 1.0]`
    pub coupling_strength: f32,

    /// Enable adaptive phase adjustment.
    pub adaptive: bool,

    /// Phase adaptation rate.
    /// Range: `[0.0, 1.0]`
    pub adaptation_rate: f32,

    /// Synchronization threshold for coherence.
    /// Range: `[0.0, 1.0]`
    pub sync_threshold: f32,
}

impl Default for PhaseConfig {
    fn default() -> Self {
        Self {
            min_phase: 0.0,
            max_phase: std::f32::consts::PI,
            default_phase: 0.0,
            frequency_hz: 100.0,
            coupling_strength: 0.5,
            adaptive: true,
            adaptation_rate: 0.1,
            sync_threshold: 0.8,
        }
    }
}

impl PhaseConfig {
    /// Validate the phase configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.min_phase < 0.0 {
            return Err(format!("min_phase must be >= 0, got {}", self.min_phase));
        }
        if self.max_phase > std::f32::consts::PI + 0.001 {
            return Err(format!("max_phase must be <= pi, got {}", self.max_phase));
        }
        if self.max_phase < self.min_phase {
            return Err(format!(
                "max_phase ({}) must be >= min_phase ({})",
                self.max_phase, self.min_phase
            ));
        }
        if !(self.min_phase..=self.max_phase).contains(&self.default_phase) {
            return Err(format!(
                "default_phase must be in [{}, {}], got {}",
                self.min_phase, self.max_phase, self.default_phase
            ));
        }
        if self.frequency_hz <= 0.0 {
            return Err(format!(
                "frequency_hz must be > 0, got {}",
                self.frequency_hz
            ));
        }
        if !(0.0..=1.0).contains(&self.coupling_strength) {
            return Err(format!(
                "coupling_strength must be in [0, 1], got {}",
                self.coupling_strength
            ));
        }
        if !(0.0..=1.0).contains(&self.adaptation_rate) {
            return Err(format!(
                "adaptation_rate must be in [0, 1], got {}",
                self.adaptation_rate
            ));
        }
        if !(0.0..=1.0).contains(&self.sync_threshold) {
            return Err(format!(
                "sync_threshold must be in [0, 1], got {}",
                self.sync_threshold
            ));
        }
        Ok(())
    }

    /// Clamp a phase value to the valid range.
    pub fn clamp(&self, phase: f32) -> f32 {
        phase.clamp(self.min_phase, self.max_phase)
    }
}
