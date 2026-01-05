//! Phase oscillator type definitions.
//!
//! Contains the core `PhaseOscillator` struct for learning rhythm synchronization.

use std::f32::consts::PI;

use crate::config::PhaseConfig;

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
    pub(crate) current_phase: f32,

    /// Oscillation frequency in Hz.
    pub(crate) frequency_hz: f32,

    /// Angular velocity (radians per second).
    pub(crate) angular_velocity: f32,

    /// Minimum phase value.
    pub(crate) min_phase: f32,

    /// Maximum phase value.
    pub(crate) max_phase: f32,

    /// Coupling strength for phase synchronization.
    pub(crate) coupling_strength: f32,

    /// Whether adaptive phase adjustment is enabled.
    pub(crate) adaptive: bool,

    /// Adaptation rate for phase adjustments.
    pub(crate) adaptation_rate: f32,

    /// Current direction of oscillation (1.0 or -1.0).
    pub(crate) direction: f32,

    /// Target phase for adaptive synchronization.
    pub(crate) target_phase: Option<f32>,

    /// Total elapsed time in seconds (for smooth oscillation).
    pub(crate) elapsed_total: f32,
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
}

impl Default for PhaseOscillator {
    fn default() -> Self {
        Self::new(&PhaseConfig::default())
    }
}
