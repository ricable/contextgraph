//! Tests for consolidation phase functionality.

use std::f32::consts::PI;
use std::time::Duration;

use crate::config::PhaseConfig;

use super::{ConsolidationPhase, PhaseDetector};

#[test]
fn test_consolidation_phase_recency_bias() {
    assert_eq!(ConsolidationPhase::NREM.recency_bias(), 0.8);
    assert_eq!(ConsolidationPhase::REM.recency_bias(), 0.4);
    assert_eq!(ConsolidationPhase::Wake.recency_bias(), 0.5);
}

#[test]
fn test_consolidation_phase_temperature() {
    assert_eq!(ConsolidationPhase::NREM.temperature(), 0.5);
    assert_eq!(ConsolidationPhase::REM.temperature(), 2.0);
    assert_eq!(ConsolidationPhase::Wake.temperature(), 1.0);
}

#[test]
fn test_consolidation_phase_coupling() {
    assert_eq!(ConsolidationPhase::NREM.coupling_strength(), 0.9);
    assert_eq!(ConsolidationPhase::REM.coupling_strength(), 0.3);
    assert_eq!(ConsolidationPhase::Wake.coupling_strength(), 0.6);
}

#[test]
fn test_consolidation_phase_learning_rate() {
    assert_eq!(ConsolidationPhase::NREM.learning_rate_modifier(), 0.3);
    assert_eq!(ConsolidationPhase::REM.learning_rate_modifier(), 0.5);
    assert_eq!(ConsolidationPhase::Wake.learning_rate_modifier(), 1.0);
}

#[test]
fn test_is_consolidation_phase() {
    assert!(ConsolidationPhase::NREM.is_consolidation_phase());
    assert!(ConsolidationPhase::REM.is_consolidation_phase());
    assert!(!ConsolidationPhase::Wake.is_consolidation_phase());
}

#[test]
fn test_is_wake() {
    assert!(!ConsolidationPhase::NREM.is_wake());
    assert!(!ConsolidationPhase::REM.is_wake());
    assert!(ConsolidationPhase::Wake.is_wake());
}

#[test]
fn test_default_phase_angle() {
    assert_eq!(ConsolidationPhase::NREM.default_phase_angle(), 0.0);
    assert_eq!(ConsolidationPhase::REM.default_phase_angle(), PI);
    assert!((ConsolidationPhase::Wake.default_phase_angle() - PI / 2.0).abs() < 0.001);
}

#[test]
fn test_phase_name_and_description() {
    assert_eq!(ConsolidationPhase::NREM.name(), "NREM");
    assert!(!ConsolidationPhase::NREM.description().is_empty());

    assert_eq!(ConsolidationPhase::REM.name(), "REM");
    assert!(!ConsolidationPhase::REM.description().is_empty());

    assert_eq!(ConsolidationPhase::Wake.name(), "Wake");
    assert!(!ConsolidationPhase::Wake.description().is_empty());
}

#[test]
fn test_phase_display() {
    assert_eq!(format!("{}", ConsolidationPhase::NREM), "NREM");
    assert_eq!(format!("{}", ConsolidationPhase::REM), "REM");
    assert_eq!(format!("{}", ConsolidationPhase::Wake), "Wake");
}

#[test]
fn test_phase_default() {
    assert_eq!(ConsolidationPhase::default(), ConsolidationPhase::Wake);
}

#[test]
fn test_detector_creation() {
    let detector = PhaseDetector::new();
    assert_eq!(detector.current_phase(), ConsolidationPhase::Wake);
}

#[test]
fn test_detector_with_thresholds_valid() {
    let detector = PhaseDetector::with_thresholds(0.2, 0.8).unwrap();
    assert_eq!(detector.nrem_threshold(), 0.2);
    assert_eq!(detector.wake_threshold(), 0.8);
}

#[test]
fn test_detector_with_thresholds_invalid() {
    // NREM threshold too high
    assert!(PhaseDetector::with_thresholds(1.5, 0.8).is_err());

    // Wake threshold too low
    assert!(PhaseDetector::with_thresholds(0.2, -0.1).is_err());

    // NREM >= Wake
    assert!(PhaseDetector::with_thresholds(0.8, 0.3).is_err());

    // Equal thresholds
    assert!(PhaseDetector::with_thresholds(0.5, 0.5).is_err());
}

#[test]
fn test_detect_phase_nrem() {
    let detector = PhaseDetector::new();
    assert_eq!(detector.detect_phase(0.1), ConsolidationPhase::NREM);
    assert_eq!(detector.detect_phase(0.0), ConsolidationPhase::NREM);
    assert_eq!(detector.detect_phase(0.29), ConsolidationPhase::NREM);
}

#[test]
fn test_detect_phase_rem() {
    let detector = PhaseDetector::new();
    assert_eq!(detector.detect_phase(0.5), ConsolidationPhase::REM);
    assert_eq!(detector.detect_phase(0.31), ConsolidationPhase::REM);
    assert_eq!(detector.detect_phase(0.69), ConsolidationPhase::REM);
}

#[test]
fn test_detect_phase_wake() {
    let detector = PhaseDetector::new();
    assert_eq!(detector.detect_phase(0.8), ConsolidationPhase::Wake);
    assert_eq!(detector.detect_phase(0.9), ConsolidationPhase::Wake);
    assert_eq!(detector.detect_phase(1.0), ConsolidationPhase::Wake);
}

#[test]
fn test_detect_phase_clamping() {
    let detector = PhaseDetector::new();

    // Values outside [0, 1] should be clamped
    assert_eq!(detector.detect_phase(-0.5), ConsolidationPhase::NREM);
    assert_eq!(detector.detect_phase(1.5), ConsolidationPhase::Wake);
}

#[test]
fn test_update_smoothing() {
    let mut detector = PhaseDetector::new();
    detector.set_min_phase_duration(Duration::ZERO); // Disable hysteresis for test

    // Start with high activity
    for _ in 0..5 {
        detector.update(0.9);
    }
    assert_eq!(detector.current_phase(), ConsolidationPhase::Wake);

    // Gradually lower activity
    for _ in 0..20 {
        detector.update(0.1);
    }

    // Should eventually transition to NREM
    // (EMA smoothing means it takes multiple updates)
    assert!(detector.smoothed_activity() < 0.3);
}

#[test]
fn test_average_activity() {
    let mut detector = PhaseDetector::new();

    detector.update(0.2);
    detector.update(0.4);
    detector.update(0.6);
    detector.update(0.8);

    let avg = detector.average_activity();
    assert!((avg - 0.5).abs() < 0.01);
}

#[test]
fn test_activity_variance() {
    let mut detector = PhaseDetector::new();

    // Same value multiple times = zero variance
    for _ in 0..5 {
        detector.update(0.5);
    }
    assert!(detector.activity_variance() < 0.01);

    // Different values = non-zero variance
    detector.reset();
    detector.update(0.0);
    detector.update(1.0);
    assert!(detector.activity_variance() > 0.0);
}

#[test]
fn test_force_phase() {
    let mut detector = PhaseDetector::new();

    detector.force_phase(ConsolidationPhase::NREM);
    assert_eq!(detector.current_phase(), ConsolidationPhase::NREM);

    detector.force_phase(ConsolidationPhase::REM);
    assert_eq!(detector.current_phase(), ConsolidationPhase::REM);
}

#[test]
fn test_reset() {
    let mut detector = PhaseDetector::new();

    detector.force_phase(ConsolidationPhase::NREM);
    detector.update(0.1);

    detector.reset();

    assert_eq!(detector.current_phase(), ConsolidationPhase::Wake);
    assert!((detector.smoothed_activity() - 0.5).abs() < 0.01);
}

#[test]
fn test_is_consolidating() {
    let mut detector = PhaseDetector::new();

    detector.force_phase(ConsolidationPhase::Wake);
    assert!(!detector.is_consolidating());

    detector.force_phase(ConsolidationPhase::NREM);
    assert!(detector.is_consolidating());

    detector.force_phase(ConsolidationPhase::REM);
    assert!(detector.is_consolidating());
}

#[test]
fn test_circadian_phase() {
    // Night hours
    assert_eq!(PhaseDetector::circadian_phase(1), ConsolidationPhase::NREM);
    assert_eq!(PhaseDetector::circadian_phase(3), ConsolidationPhase::REM);

    // Day hours
    assert_eq!(PhaseDetector::circadian_phase(12), ConsolidationPhase::Wake);
    assert_eq!(PhaseDetector::circadian_phase(18), ConsolidationPhase::Wake);

    // Evening
    assert_eq!(PhaseDetector::circadian_phase(23), ConsolidationPhase::NREM);
}

#[test]
fn test_set_ema_alpha() {
    let mut detector = PhaseDetector::new();

    detector.set_ema_alpha(0.8);
    // Alpha is internal, just verify no panic

    // Test clamping
    detector.set_ema_alpha(1.5);
    detector.set_ema_alpha(-0.2);
}

#[test]
fn test_from_config() {
    let config = PhaseConfig::default();
    let detector = PhaseDetector::from_config(&config);

    // Should use sync_threshold to derive thresholds
    assert!(detector.wake_threshold() > 0.0);
    assert!(detector.nrem_threshold() > 0.0);
    assert!(detector.nrem_threshold() < detector.wake_threshold());
}

#[test]
fn test_phase_equality() {
    assert_eq!(ConsolidationPhase::NREM, ConsolidationPhase::NREM);
    assert_ne!(ConsolidationPhase::NREM, ConsolidationPhase::REM);
    assert_ne!(ConsolidationPhase::REM, ConsolidationPhase::Wake);
}

#[test]
fn test_phase_clone() {
    let phase = ConsolidationPhase::REM;
    let cloned = phase;
    assert_eq!(phase, cloned);
}

#[test]
fn test_detector_default() {
    let detector = PhaseDetector::default();
    assert_eq!(detector.current_phase(), ConsolidationPhase::Wake);
}
