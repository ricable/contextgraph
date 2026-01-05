//! Tests for SurpriseCalculator.

use crate::config::{KlConfig, SurpriseConfig};
use crate::error::UtlError;
use crate::surprise::SurpriseCalculator;

fn create_calculator() -> SurpriseCalculator {
    SurpriseCalculator::new(&SurpriseConfig::default())
}

#[test]
fn test_calculator_creation() {
    let config = SurpriseConfig::default();
    let calc = SurpriseCalculator::new(&config);

    assert_eq!(calc.entropy_weight(), config.entropy_weight);
    assert_eq!(calc.novelty_boost(), config.novelty_boost);
    assert_eq!(calc.min_threshold(), config.min_threshold);
}

#[test]
fn test_calculator_default() {
    let calc = SurpriseCalculator::default();
    assert!(calc.entropy_weight() > 0.0);
    assert!(calc.max_value() <= 1.0);
}

#[test]
fn test_compute_surprise_empty_current() {
    let calc = create_calculator();
    let empty: Vec<f32> = vec![];
    let history = vec![vec![0.1, 0.2]];

    let surprise = calc.compute_surprise(&empty, &history);
    assert_eq!(surprise, 0.0);
}

#[test]
fn test_compute_surprise_empty_history() {
    let calc = create_calculator();
    let current = vec![0.1, 0.2, 0.3];
    let history: Vec<Vec<f32>> = vec![];

    let surprise = calc.compute_surprise(&current, &history);
    assert_eq!(surprise, 1.0, "Empty history should give maximum surprise");
}

#[test]
fn test_compute_surprise_identical() {
    let calc = create_calculator();
    let current = vec![0.1, 0.2, 0.3, 0.4];
    let history = vec![vec![0.1, 0.2, 0.3, 0.4]];

    let surprise = calc.compute_surprise(&current, &history);
    assert!(
        surprise < 0.1,
        "Identical embedding should have low surprise"
    );
}

#[test]
fn test_compute_surprise_different() {
    let calc = create_calculator();
    let current = vec![0.9, 0.05, 0.03, 0.02];
    let history = vec![vec![0.1, 0.2, 0.3, 0.4]];

    let surprise = calc.compute_surprise(&current, &history);
    assert!(
        surprise > 0.0,
        "Different embeddings should have positive surprise"
    );
    assert!(surprise <= 1.0, "Surprise should be at most 1.0");
}

#[test]
fn test_compute_surprise_range() {
    let calc = create_calculator();
    let current = vec![0.5, 0.3, 0.15, 0.05];
    let history = vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.25, 0.25, 0.25, 0.25]];

    let surprise = calc.compute_surprise(&current, &history);
    assert!(
        (0.0..=1.0).contains(&surprise),
        "Surprise should be in [0, 1]"
    );
}

#[test]
fn test_compute_surprise_checked_error() {
    let calc = create_calculator();
    let empty: Vec<f32> = vec![];
    let history = vec![vec![0.1, 0.2]];

    let result = calc.compute_surprise_checked(&empty, &history);
    assert!(matches!(result, Err(UtlError::EmptyInput)));
}

#[test]
fn test_compute_surprise_checked_success() {
    let calc = create_calculator();
    let current = vec![0.1, 0.2, 0.3];
    let history = vec![vec![0.15, 0.25, 0.35]];

    let result = calc.compute_surprise_checked(&current, &history);
    assert!(result.is_ok());
    let surprise = result.unwrap();
    assert!((0.0..=1.0).contains(&surprise));
}

#[test]
fn test_compute_surprise_smoothed() {
    let mut calc = create_calculator();
    let current = vec![0.1, 0.2, 0.3, 0.4];
    let history = vec![vec![0.5, 0.3, 0.15, 0.05]];

    // First call sets EMA state
    let _first = calc.compute_surprise_smoothed(&current, &history);
    assert!(calc.ema_state().is_some());

    // Second call should be smoothed
    let second = calc.compute_surprise_smoothed(&current, &history);
    assert!((0.0..=1.0).contains(&second));

    // Reset EMA
    calc.reset_ema();
    assert!(calc.ema_state().is_none());
}

#[test]
fn test_compute_kl_surprise() {
    let calc = create_calculator();
    let current = vec![0.25, 0.25, 0.25, 0.25];
    let reference = vec![0.1, 0.2, 0.3, 0.4];

    let result = calc.compute_kl_surprise(&current, &reference);
    assert!(result.is_ok());
    let surprise = result.unwrap();
    assert!((0.0..=1.0).contains(&surprise));
}

#[test]
fn test_compute_combined_surprise() {
    let calc = create_calculator();
    let embedding = vec![0.1, 0.2, 0.3, 0.4];
    let history = vec![vec![0.15, 0.25, 0.35, 0.25]];
    let dist = vec![0.25, 0.25, 0.25, 0.25];
    let ref_dist = vec![0.1, 0.2, 0.3, 0.4];

    // Without distributions
    let surprise1 = calc.compute_combined_surprise(&embedding, &history, None, None);
    assert!((0.0..=1.0).contains(&surprise1));

    // With distributions
    let surprise2 =
        calc.compute_combined_surprise(&embedding, &history, Some(&dist), Some(&ref_dist));
    assert!((0.0..=1.0).contains(&surprise2));
}

#[test]
fn test_repetition_decay() {
    let calc = create_calculator();
    let base = 0.8;

    // No repetitions
    let no_decay = calc.apply_repetition_decay(base, 0);
    assert!((no_decay - base).abs() < 1e-6);

    // With repetitions
    let decayed = calc.apply_repetition_decay(base, 5);
    assert!(decayed < base, "Repeated items should have lower surprise");
    assert!(decayed >= 0.0);
}

#[test]
fn test_no_nan_infinity() {
    let calc = create_calculator();

    // Test with edge case inputs
    let zero = vec![0.0, 0.0, 0.0];
    let normal = vec![0.1, 0.2, 0.7];
    let history = vec![normal.clone()];

    let surprise = calc.compute_surprise(&zero, &history);
    assert!(!surprise.is_nan(), "Should not produce NaN");
    assert!(!surprise.is_infinite(), "Should not produce Infinity");

    // Test clamping
    let result = calc.clamp_result(f32::NAN);
    assert_eq!(result, 0.0);

    let result = calc.clamp_result(f32::INFINITY);
    assert_eq!(result, calc.max_value());

    let result = calc.clamp_result(f32::NEG_INFINITY);
    assert_eq!(result, 0.0);
}

#[test]
fn test_min_threshold() {
    // Create config with higher min threshold
    let config = SurpriseConfig {
        min_threshold: 0.3,
        ..Default::default()
    };
    let calc = SurpriseCalculator::new(&config);

    // Very similar embeddings should produce below-threshold surprise
    let current = vec![0.25, 0.25, 0.25, 0.25];
    let history = vec![vec![0.24, 0.26, 0.25, 0.25]];

    let surprise = calc.compute_surprise(&current, &history);
    // Either 0 (below threshold) or >= min_threshold
    assert!(surprise == 0.0 || surprise >= 0.3);
}

#[test]
fn test_max_value_clamping() {
    let config = SurpriseConfig {
        max_value: 0.8,
        novelty_boost: 2.0,
        ..Default::default()
    };
    let calc = SurpriseCalculator::new(&config);

    let current = vec![1.0, 0.0, 0.0];
    let history = vec![vec![0.0, 1.0, 0.0]];

    let surprise = calc.compute_surprise(&current, &history);
    assert!(surprise <= 0.8, "Surprise should be clamped to max_value");
}

#[test]
fn test_with_kl_config() {
    let config = SurpriseConfig::default();
    let kl_config = KlConfig {
        symmetric: true,
        ..Default::default()
    };

    let calc = SurpriseCalculator::with_kl_config(&config, &kl_config);

    let dist1 = vec![0.5, 0.5];
    let dist2 = vec![0.25, 0.75];

    let kl = calc.compute_kl_surprise(&dist1, &dist2);
    assert!(kl.is_ok());
}
