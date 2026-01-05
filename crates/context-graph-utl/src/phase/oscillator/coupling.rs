//! Phase coupling and synchronization methods.
//!
//! Contains methods for coupling strength management and phase synchronization detection.

use std::f32::consts::PI;

use super::types::PhaseOscillator;

impl PhaseOscillator {
    /// Get the coupling strength.
    #[inline]
    pub fn coupling_strength(&self) -> f32 {
        self.coupling_strength
    }

    /// Set the coupling strength.
    ///
    /// # Arguments
    ///
    /// * `strength` - Coupling strength in `[0, 1]`
    pub fn set_coupling_strength(&mut self, strength: f32) {
        self.coupling_strength = strength.clamp(0.0, 1.0);
    }

    /// Check if the oscillator is approximately synchronized (φ ≈ 0).
    ///
    /// # Arguments
    ///
    /// * `threshold` - Phase threshold for synchronization (radians)
    ///
    /// # Returns
    ///
    /// `true` if phase is within threshold of 0.
    pub fn is_synchronized(&self, threshold: f32) -> bool {
        self.current_phase < threshold
    }

    /// Check if the oscillator is approximately anti-phase (φ ≈ π).
    ///
    /// # Arguments
    ///
    /// * `threshold` - Phase threshold for anti-phase detection (radians)
    ///
    /// # Returns
    ///
    /// `true` if phase is within threshold of π.
    pub fn is_anti_phase(&self, threshold: f32) -> bool {
        (self.max_phase - self.current_phase) < threshold
    }

    /// Get the phase as a normalized value in `[0, 1]`.
    ///
    /// # Returns
    ///
    /// Normalized phase where 0 = synchronized, 1 = anti-phase.
    pub fn normalized_phase(&self) -> f32 {
        (self.current_phase - self.min_phase) / (self.max_phase - self.min_phase)
    }

    /// Compute phase coupling with another oscillator.
    ///
    /// Returns a coupling factor based on phase difference and coupling strength.
    ///
    /// # Arguments
    ///
    /// * `other` - Another phase oscillator
    ///
    /// # Returns
    ///
    /// Coupling factor in `[0, 1]` where 1.0 means perfect phase alignment.
    pub fn compute_coupling(&self, other: &PhaseOscillator) -> f32 {
        let phase_diff = (self.current_phase - other.current_phase).abs();
        let normalized_diff = phase_diff / PI;
        let base_coupling = 1.0 - normalized_diff;
        base_coupling * self.coupling_strength * other.coupling_strength
    }
}
