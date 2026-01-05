//! Tests for UtlComputationMetrics.

use super::*;

#[test]
fn test_computation_metrics_default() {
    let metrics = UtlComputationMetrics::default();

    assert_eq!(metrics.computation_count, 0);
    assert_eq!(metrics.avg_learning_magnitude, 0.0);
    assert_eq!(metrics.avg_delta_s, 0.0);
    assert_eq!(metrics.avg_delta_c, 0.0);
    assert_eq!(metrics.avg_latency_us, 0.0);
    assert_eq!(metrics.p99_latency_us, 0);
}

#[test]
fn test_computation_metrics_is_healthy_default() {
    let metrics = UtlComputationMetrics::default();
    assert!(metrics.is_healthy());
}

#[test]
fn test_computation_metrics_is_healthy_with_good_latency() {
    let metrics = UtlComputationMetrics {
        avg_latency_us: 5000.0, // 5ms
        p99_latency_us: 20000,  // 20ms
        avg_learning_magnitude: 0.5,
        ..Default::default()
    };

    assert!(metrics.is_healthy());
}

#[test]
fn test_computation_metrics_unhealthy_high_avg_latency() {
    let metrics = UtlComputationMetrics {
        avg_latency_us: 15000.0, // 15ms > 10ms threshold
        p99_latency_us: 20000,
        avg_learning_magnitude: 0.5,
        ..Default::default()
    };

    assert!(!metrics.is_healthy());
}

#[test]
fn test_computation_metrics_unhealthy_high_p99_latency() {
    let metrics = UtlComputationMetrics {
        avg_latency_us: 5000.0,
        p99_latency_us: 60000, // 60ms > 50ms threshold
        avg_learning_magnitude: 0.5,
        ..Default::default()
    };

    assert!(!metrics.is_healthy());
}

#[test]
fn test_computation_metrics_unhealthy_nan() {
    let metrics = UtlComputationMetrics {
        avg_learning_magnitude: f32::NAN,
        ..Default::default()
    };

    assert!(!metrics.is_healthy());
}

#[test]
fn test_computation_metrics_unhealthy_infinite() {
    let metrics = UtlComputationMetrics {
        avg_learning_magnitude: f32::INFINITY,
        ..Default::default()
    };

    assert!(!metrics.is_healthy());
}

#[test]
fn test_computation_metrics_learning_efficiency() {
    let metrics = UtlComputationMetrics {
        avg_learning_magnitude: 0.8,
        avg_latency_us: 5000.0,
        ..Default::default()
    };

    let efficiency = metrics.learning_efficiency();
    // (0.8 / 5000.0) * 1000 = 0.16
    assert!((efficiency - 0.16).abs() < 0.001);
}

#[test]
fn test_computation_metrics_learning_efficiency_zero_latency() {
    let metrics = UtlComputationMetrics {
        avg_learning_magnitude: 0.8,
        avg_latency_us: 0.0,
        ..Default::default()
    };

    assert_eq!(metrics.learning_efficiency(), 0.0);
}

#[test]
fn test_computation_metrics_record_first_computation() {
    let mut metrics = UtlComputationMetrics::default();

    metrics.record_computation(0.7, 0.5, 0.6, JohariQuadrant::Open, 1000.0);

    assert_eq!(metrics.computation_count, 1);
    assert_eq!(metrics.avg_learning_magnitude, 0.7);
    assert_eq!(metrics.avg_delta_s, 0.5);
    assert_eq!(metrics.avg_delta_c, 0.6);
    assert_eq!(metrics.avg_latency_us, 1000.0);
    assert_eq!(metrics.quadrant_distribution.open, 1);
}

#[test]
fn test_computation_metrics_record_multiple_computations() {
    let mut metrics = UtlComputationMetrics::default();

    // First computation sets baseline
    metrics.record_computation(0.5, 0.3, 0.4, JohariQuadrant::Open, 1000.0);

    // Second computation uses EMA
    metrics.record_computation(0.9, 0.7, 0.8, JohariQuadrant::Blind, 2000.0);

    assert_eq!(metrics.computation_count, 2);

    // EMA: 0.1 * new + 0.9 * old
    // avg_learning_magnitude = 0.1 * 0.9 + 0.9 * 0.5 = 0.09 + 0.45 = 0.54
    assert!((metrics.avg_learning_magnitude - 0.54).abs() < 0.01);

    assert_eq!(metrics.quadrant_distribution.open, 1);
    assert_eq!(metrics.quadrant_distribution.blind, 1);
}

#[test]
fn test_computation_metrics_dominant_quadrant() {
    let mut metrics = UtlComputationMetrics::default();

    metrics.record_computation(0.5, 0.3, 0.4, JohariQuadrant::Blind, 1000.0);
    metrics.record_computation(0.5, 0.3, 0.4, JohariQuadrant::Blind, 1000.0);
    metrics.record_computation(0.5, 0.3, 0.4, JohariQuadrant::Open, 1000.0);

    assert_eq!(metrics.dominant_quadrant(), JohariQuadrant::Blind);
}

#[test]
fn test_computation_metrics_serialization() {
    let metrics = UtlComputationMetrics {
        computation_count: 100,
        avg_learning_magnitude: 0.65,
        avg_delta_s: 0.4,
        avg_delta_c: 0.5,
        quadrant_distribution: QuadrantDistribution {
            open: 40,
            blind: 30,
            hidden: 20,
            unknown: 10,
        },
        lifecycle_stage: LifecycleStage::Growth,
        lambda_weights: LifecycleLambdaWeights::default(),
        avg_latency_us: 2500.0,
        p99_latency_us: 8000,
    };

    let json = serde_json::to_string(&metrics).expect("serialize");
    let parsed: UtlComputationMetrics = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(metrics, parsed);
}
