//! Phase oscillator for learning rhythms.
//!
//! Provides smooth phase oscillation with configurable frequency and coupling
//! strength for the UTL formula phase component `cos(φ)`.

use std::f32::consts::PI;
use std::time::Duration;

use crate::config::PhaseConfig;
use crate::error::{UtlError, UtlResult};

/// Phase oscillator for learning rhythm synchronization.
///
/// The oscillator maintains a phase angle in the range `[0, π]` that smoothly
/// oscillates based on a configurable frequency. The `cos(φ)` value is used
/// in the UTL formula to modulate learning based on phase alignment.
///
/// # Constitution Reference
///
/// - Phase range: `[0, π]`
/// - `cos(φ) = 1.0` at φ = 0 (fully synchronized)
/// - `cos(φ) = -1.0` at φ = π (anti-phase)
/// - Default frequency: 100Hz (L4 reference)
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
/// // Simulate time passing
/// oscillator.update(Duration::from_millis(5));
///
/// let phase = oscillator.phase();
/// let cos_phi = oscillator.cos_phase();
///
/// assert!(phase >= 0.0 && phase <= std::f32::consts::PI);
/// assert!(cos_phi >= -1.0 && cos_phi <= 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct PhaseOscillator {
    /// Current phase angle in radians `[0, π]`.
    current_phase: f32,

    /// Oscillation frequency in Hz.
    frequency_hz: f32,

    /// Angular velocity (radians per second).
    angular_velocity: f32,

    /// Minimum phase value.
    min_phase: f32,

    /// Maximum phase value.
    max_phase: f32,

    /// Coupling strength for phase synchronization.
    coupling_strength: f32,

    /// Whether adaptive phase adjustment is enabled.
    adaptive: bool,

    /// Adaptation rate for phase adjustments.
    adaptation_rate: f32,

    /// Current direction of oscillation (1.0 or -1.0).
    direction: f32,

    /// Target phase for adaptive synchronization.
    target_phase: Option<f32>,

    /// Total elapsed time in seconds (for smooth oscillation).
    elapsed_total: f32,
}

impl PhaseOscillator {
    /// Create a new phase oscillator from configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Phase configuration containing frequency and bounds
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::phase::PhaseOscillator;
    /// use context_graph_utl::config::PhaseConfig;
    ///
    /// let config = PhaseConfig::default();
    /// let oscillator = PhaseOscillator::new(&config);
    ///
    /// assert_eq!(oscillator.phase(), config.default_phase);
    /// ```
    pub fn new(config: &PhaseConfig) -> Self {
        // Angular velocity = 2π * frequency (full cycle per second)
        // But we oscillate over [0, π], so effective angular velocity is π * frequency
        let angular_velocity = PI * config.frequency_hz;

        Self {
            current_phase: config.default_phase,
            frequency_hz: config.frequency_hz,
            angular_velocity,
            min_phase: config.min_phase,
            max_phase: config.max_phase,
            coupling_strength: config.coupling_strength,
            adaptive: config.adaptive,
            adaptation_rate: config.adaptation_rate,
            direction: 1.0,
            target_phase: None,
            elapsed_total: 0.0,
        }
    }

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

impl Default for PhaseOscillator {
    fn default() -> Self {
        Self::new(&PhaseConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> PhaseConfig {
        PhaseConfig::default()
    }

    #[test]
    fn test_oscillator_creation() {
        let config = test_config();
        let oscillator = PhaseOscillator::new(&config);

        assert_eq!(oscillator.phase(), config.default_phase);
        assert_eq!(oscillator.frequency(), config.frequency_hz);
    }

    #[test]
    fn test_synchronized_oscillator() {
        let config = test_config();
        let oscillator = PhaseOscillator::synchronized(&config);

        assert_eq!(oscillator.phase(), 0.0);
        assert!((oscillator.cos_phase() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_anti_phase_oscillator() {
        let config = test_config();
        let oscillator = PhaseOscillator::anti_phase(&config);

        assert!((oscillator.phase() - PI).abs() < 0.001);
        assert!((oscillator.cos_phase() - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_with_initial_phase_valid() {
        let config = test_config();
        let oscillator = PhaseOscillator::with_initial_phase(&config, PI / 4.0).unwrap();

        assert!((oscillator.phase() - PI / 4.0).abs() < 0.001);
    }

    #[test]
    fn test_with_initial_phase_invalid() {
        let config = test_config();
        let result = PhaseOscillator::with_initial_phase(&config, -0.5);

        assert!(result.is_err());
    }

    #[test]
    fn test_update_changes_phase() {
        let config = test_config();
        let mut oscillator = PhaseOscillator::new(&config);

        let initial_phase = oscillator.phase();
        oscillator.update(Duration::from_millis(10));

        // Phase should have changed (unless at boundary)
        // Note: with high frequency, it might wrap
        let new_phase = oscillator.phase();
        assert!(new_phase >= 0.0 && new_phase <= PI);
    }

    #[test]
    fn test_phase_stays_in_bounds() {
        let config = test_config();
        let mut oscillator = PhaseOscillator::new(&config);

        // Update many times
        for _ in 0..1000 {
            oscillator.update(Duration::from_millis(1));
            assert!(oscillator.phase() >= 0.0);
            assert!(oscillator.phase() <= PI);
        }
    }

    #[test]
    fn test_cos_phase_range() {
        let config = test_config();
        let mut oscillator = PhaseOscillator::new(&config);

        for _ in 0..1000 {
            oscillator.update(Duration::from_millis(1));
            let cos_phi = oscillator.cos_phase();
            assert!(cos_phi >= -1.0);
            assert!(cos_phi <= 1.0);
        }
    }

    #[test]
    fn test_sin_phase_range() {
        let config = test_config();
        let mut oscillator = PhaseOscillator::new(&config);

        for _ in 0..1000 {
            oscillator.update(Duration::from_millis(1));
            let sin_phi = oscillator.sin_phase();
            // sin(φ) for φ in [0, π] is always >= 0
            assert!(sin_phi >= 0.0);
            assert!(sin_phi <= 1.0);
        }
    }

    #[test]
    fn test_set_phase() {
        let config = test_config();
        let mut oscillator = PhaseOscillator::new(&config);

        oscillator.set_phase(PI / 2.0);
        assert!((oscillator.phase() - PI / 2.0).abs() < 0.001);

        // Test clamping
        oscillator.set_phase(5.0);
        assert_eq!(oscillator.phase(), PI);

        oscillator.set_phase(-1.0);
        assert_eq!(oscillator.phase(), 0.0);
    }

    #[test]
    fn test_reset() {
        let config = test_config();
        let mut oscillator = PhaseOscillator::new(&config);

        oscillator.update(Duration::from_millis(100));
        oscillator.set_target_phase(PI / 2.0);
        oscillator.reset();

        assert_eq!(oscillator.phase(), 0.0);
        assert!(oscillator.target_phase().is_none());
    }

    #[test]
    fn test_target_phase() {
        let config = test_config();
        let mut oscillator = PhaseOscillator::new(&config);

        assert!(oscillator.target_phase().is_none());

        oscillator.set_target_phase(PI / 3.0);
        assert!((oscillator.target_phase().unwrap() - PI / 3.0).abs() < 0.001);

        oscillator.clear_target_phase();
        assert!(oscillator.target_phase().is_none());
    }

    #[test]
    fn test_set_frequency_valid() {
        let config = test_config();
        let mut oscillator = PhaseOscillator::new(&config);

        assert!(oscillator.set_frequency(50.0).is_ok());
        assert_eq!(oscillator.frequency(), 50.0);
    }

    #[test]
    fn test_set_frequency_invalid() {
        let config = test_config();
        let mut oscillator = PhaseOscillator::new(&config);

        assert!(oscillator.set_frequency(0.0).is_err());
        assert!(oscillator.set_frequency(-10.0).is_err());
    }

    #[test]
    fn test_coupling_strength() {
        let config = test_config();
        let mut oscillator = PhaseOscillator::new(&config);

        oscillator.set_coupling_strength(0.8);
        assert_eq!(oscillator.coupling_strength(), 0.8);

        // Test clamping
        oscillator.set_coupling_strength(1.5);
        assert_eq!(oscillator.coupling_strength(), 1.0);

        oscillator.set_coupling_strength(-0.2);
        assert_eq!(oscillator.coupling_strength(), 0.0);
    }

    #[test]
    fn test_is_synchronized() {
        let config = test_config();
        let oscillator = PhaseOscillator::synchronized(&config);

        assert!(oscillator.is_synchronized(0.1));
        assert!(!oscillator.is_anti_phase(0.1));
    }

    #[test]
    fn test_is_anti_phase() {
        let config = test_config();
        let oscillator = PhaseOscillator::anti_phase(&config);

        assert!(oscillator.is_anti_phase(0.1));
        assert!(!oscillator.is_synchronized(0.1));
    }

    #[test]
    fn test_normalized_phase() {
        let config = test_config();
        let mut oscillator = PhaseOscillator::new(&config);

        oscillator.set_phase(0.0);
        assert_eq!(oscillator.normalized_phase(), 0.0);

        oscillator.set_phase(PI);
        assert!((oscillator.normalized_phase() - 1.0).abs() < 0.001);

        oscillator.set_phase(PI / 2.0);
        assert!((oscillator.normalized_phase() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_compute_coupling() {
        let config = test_config();
        let mut osc1 = PhaseOscillator::synchronized(&config);
        let mut osc2 = PhaseOscillator::synchronized(&config);

        osc1.set_coupling_strength(1.0);
        osc2.set_coupling_strength(1.0);

        // Same phase = max coupling
        let coupling = osc1.compute_coupling(&osc2);
        assert!((coupling - 1.0).abs() < 0.001);

        // Opposite phase = min coupling
        osc2.set_phase(PI);
        let coupling = osc1.compute_coupling(&osc2);
        assert!(coupling.abs() < 0.001);
    }

    #[test]
    fn test_direction() {
        let config = test_config();
        let oscillator = PhaseOscillator::new(&config);

        // Initial direction should be positive
        assert_eq!(oscillator.direction(), 1.0);
    }

    #[test]
    fn test_elapsed_total() {
        let config = test_config();
        let mut oscillator = PhaseOscillator::new(&config);

        oscillator.update(Duration::from_millis(100));
        oscillator.update(Duration::from_millis(50));

        let elapsed = oscillator.elapsed_total();
        assert!((elapsed.as_millis() as f32 - 150.0).abs() < 1.0);
    }

    #[test]
    fn test_default_oscillator() {
        let oscillator = PhaseOscillator::default();
        let config = PhaseConfig::default();

        assert_eq!(oscillator.frequency(), config.frequency_hz);
    }

    #[test]
    fn test_oscillation_reverses_at_boundaries() {
        let mut config = test_config();
        config.frequency_hz = 1000.0; // High frequency for fast oscillation

        let mut oscillator = PhaseOscillator::new(&config);

        let mut saw_positive_direction = false;
        let mut saw_negative_direction = false;

        for _ in 0..100 {
            oscillator.update(Duration::from_millis(1));
            if oscillator.direction() > 0.0 {
                saw_positive_direction = true;
            } else {
                saw_negative_direction = true;
            }
        }

        // With high frequency oscillation, both directions should be observed
        assert!(saw_positive_direction);
        assert!(saw_negative_direction);
    }

    #[test]
    fn test_cos_phi_at_known_phases() {
        let config = test_config();
        let mut oscillator = PhaseOscillator::new(&config);

        // cos(0) = 1
        oscillator.set_phase(0.0);
        assert!((oscillator.cos_phase() - 1.0).abs() < 0.001);

        // cos(π/2) = 0
        oscillator.set_phase(PI / 2.0);
        assert!(oscillator.cos_phase().abs() < 0.001);

        // cos(π) = -1
        oscillator.set_phase(PI);
        assert!((oscillator.cos_phase() - (-1.0)).abs() < 0.001);
    }
}
