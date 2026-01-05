//! Core phase oscillator implementation.
//!
//! Contains the main methods for phase updates, accessors, and configuration.

use std::f32::consts::PI;
use std::time::Duration;

use crate::config::PhaseConfig;
use crate::error::{UtlError, UtlResult};

use super::types::PhaseOscillator;

impl PhaseOscillator {
    /// Create a phase oscillator with a specific starting phase.
    ///
    /// # Arguments
    ///
    /// * `config` - Phase configuration
    /// * `initial_phase` - Initial phase angle in `[0, π]`
    ///
    /// # Errors
    ///
    /// Returns `UtlError::PhaseError` if initial_phase is out of range.
    pub fn with_initial_phase(config: &PhaseConfig, initial_phase: f32) -> UtlResult<Self> {
        if initial_phase < config.min_phase || initial_phase > config.max_phase {
            return Err(UtlError::PhaseError(format!(
                "Initial phase {} out of range [{}, {}]",
                initial_phase, config.min_phase, config.max_phase
            )));
        }

        let mut oscillator = Self::new(config);
        oscillator.current_phase = initial_phase;
        Ok(oscillator)
    }

    /// Create a synchronized oscillator (phase = 0).
    ///
    /// This creates an oscillator at maximum synchronization where `cos(φ) = 1.0`.
    pub fn synchronized(config: &PhaseConfig) -> Self {
        let mut oscillator = Self::new(config);
        oscillator.current_phase = 0.0;
        oscillator
    }

    /// Create an anti-phase oscillator (phase = π).
    ///
    /// This creates an oscillator at minimum synchronization where `cos(φ) = -1.0`.
    pub fn anti_phase(config: &PhaseConfig) -> Self {
        let mut oscillator = Self::new(config);
        oscillator.current_phase = config.max_phase;
        oscillator
    }

    /// Update the phase based on elapsed time.
    ///
    /// The phase oscillates smoothly between `min_phase` and `max_phase` based
    /// on the configured frequency. When the phase reaches a boundary, it
    /// reverses direction.
    ///
    /// # Arguments
    ///
    /// * `elapsed` - Time elapsed since last update
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::phase::PhaseOscillator;
    /// use context_graph_utl::config::PhaseConfig;
    /// use std::time::Duration;
    ///
    /// let config = PhaseConfig::default();
    /// let mut oscillator = PhaseOscillator::new(&config);
    ///
    /// // Update multiple times
    /// for _ in 0..100 {
    ///     oscillator.update(Duration::from_millis(1));
    /// }
    ///
    /// // Phase should still be in valid range
    /// assert!(oscillator.phase() >= 0.0);
    /// assert!(oscillator.phase() <= std::f32::consts::PI);
    /// ```
    pub fn update(&mut self, elapsed: Duration) {
        let dt = elapsed.as_secs_f32();
        self.elapsed_total += dt;

        // Calculate phase change
        let delta_phase = self.angular_velocity * dt * self.direction;

        // Apply adaptive adjustment if enabled and target is set
        let adaptive_adjustment = if self.adaptive {
            if let Some(target) = self.target_phase {
                let error = target - self.current_phase;
                error * self.adaptation_rate * dt
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Update phase
        let new_phase = self.current_phase + delta_phase + adaptive_adjustment;

        // Handle boundary conditions with direction reversal
        if new_phase >= self.max_phase {
            self.current_phase = self.max_phase - (new_phase - self.max_phase);
            self.direction = -1.0;
        } else if new_phase <= self.min_phase {
            self.current_phase = self.min_phase + (self.min_phase - new_phase);
            self.direction = 1.0;
        } else {
            self.current_phase = new_phase;
        }

        // Ensure phase stays in valid range
        self.current_phase = self.current_phase.clamp(self.min_phase, self.max_phase);
    }

    /// Get the current phase angle in radians.
    ///
    /// # Returns
    ///
    /// Phase angle in range `[0, π]`.
    #[inline]
    pub fn phase(&self) -> f32 {
        self.current_phase
    }

    /// Get the cosine of the current phase.
    ///
    /// This is the value used in the UTL formula: `L = f((ΔS × ΔC) · wₑ · cos φ)`
    ///
    /// # Returns
    ///
    /// `cos(φ)` in range `[-1, 1]`:
    /// - `1.0` when φ = 0 (fully synchronized)
    /// - `0.0` when φ = π/2 (orthogonal)
    /// - `-1.0` when φ = π (anti-phase)
    #[inline]
    pub fn cos_phase(&self) -> f32 {
        self.current_phase.cos()
    }

    /// Get the sine of the current phase.
    ///
    /// Useful for computing phase velocity or visualizing phase in 2D.
    ///
    /// # Returns
    ///
    /// `sin(φ)` in range `[0, 1]` for φ in `[0, π]`.
    #[inline]
    pub fn sin_phase(&self) -> f32 {
        self.current_phase.sin()
    }

    /// Set a target phase for adaptive synchronization.
    ///
    /// When adaptive mode is enabled, the oscillator will gradually
    /// adjust toward the target phase.
    ///
    /// # Arguments
    ///
    /// * `target` - Target phase angle in `[0, π]`
    pub fn set_target_phase(&mut self, target: f32) {
        self.target_phase = Some(target.clamp(self.min_phase, self.max_phase));
    }

    /// Clear the target phase.
    ///
    /// The oscillator will return to free oscillation.
    pub fn clear_target_phase(&mut self) {
        self.target_phase = None;
    }

    /// Get the current target phase, if set.
    pub fn target_phase(&self) -> Option<f32> {
        self.target_phase
    }

    /// Set the phase directly (bypasses normal oscillation).
    ///
    /// Use sparingly; prefer `update()` for normal operation.
    ///
    /// # Arguments
    ///
    /// * `phase` - New phase angle (will be clamped to valid range)
    pub fn set_phase(&mut self, phase: f32) {
        self.current_phase = phase.clamp(self.min_phase, self.max_phase);
    }

    /// Reset the oscillator to initial state.
    ///
    /// Sets phase to 0 (fully synchronized) and resets elapsed time.
    pub fn reset(&mut self) {
        self.current_phase = self.min_phase;
        self.direction = 1.0;
        self.elapsed_total = 0.0;
        self.target_phase = None;
    }

    /// Get the oscillation frequency in Hz.
    #[inline]
    pub fn frequency(&self) -> f32 {
        self.frequency_hz
    }

    /// Set the oscillation frequency.
    ///
    /// # Arguments
    ///
    /// * `frequency_hz` - New frequency in Hz (must be > 0)
    ///
    /// # Errors
    ///
    /// Returns `UtlError::PhaseError` if frequency is not positive.
    pub fn set_frequency(&mut self, frequency_hz: f32) -> UtlResult<()> {
        if frequency_hz <= 0.0 {
            return Err(UtlError::PhaseError(format!(
                "Frequency must be positive, got {}",
                frequency_hz
            )));
        }
        self.frequency_hz = frequency_hz;
        self.angular_velocity = PI * frequency_hz;
        Ok(())
    }

    /// Get the total elapsed time since creation or reset.
    pub fn elapsed_total(&self) -> Duration {
        Duration::from_secs_f32(self.elapsed_total)
    }

    /// Get the current oscillation direction.
    ///
    /// # Returns
    ///
    /// `1.0` for increasing phase, `-1.0` for decreasing phase.
    pub fn direction(&self) -> f32 {
        self.direction
    }
}
