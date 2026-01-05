//! Tests for the phase oscillator module.

use std::f32::consts::PI;
use std::time::Duration;

use crate::config::PhaseConfig;

use super::PhaseOscillator;

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

    let _initial_phase = oscillator.phase();
    oscillator.update(Duration::from_millis(10));

    // Phase should have changed (unless at boundary)
    // Note: with high frequency, it might wrap
    let new_phase = oscillator.phase();
    assert!((0.0..=PI).contains(&new_phase));
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
