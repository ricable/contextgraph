//! Latency budget enforcement tests (TC-GHOST-007).
//!
//! This module contains tests that verify layer latency budget compliance.
//! NOTE: Stubs now return NotImplemented errors per AP-007 (fail-fast).
//! These tests verify that the NotImplemented error is returned quickly.

use std::time::{Duration, Instant};

use crate::error::CoreError;
use crate::traits::NervousLayer;
use crate::types::LayerInput;

use super::{
    StubCoherenceLayer, StubLearningLayer, StubMemoryLayer, StubReflexLayer, StubSensingLayer,
};

/// Helper to create test input.
fn test_input(content: &str) -> LayerInput {
    LayerInput::new("test-request-123".to_string(), content.to_string())
}

#[tokio::test]
async fn test_layer_fails_fast_sensing() {
    let layer = StubSensingLayer::new();
    let input = test_input("test content for sensing layer");

    let budget = layer.latency_budget();
    let start = Instant::now();
    let result = layer.process(input).await;
    let elapsed = start.elapsed();

    // Must fail with NotImplemented (AP-007)
    assert!(result.is_err(), "Sensing layer stub must fail fast");
    assert!(matches!(result, Err(CoreError::NotImplemented(_))));

    // Failure must be fast (well under budget)
    assert!(
        elapsed <= budget,
        "Sensing layer took {:?} but budget is {:?}",
        elapsed,
        budget
    );
}

#[tokio::test]
async fn test_layer_fails_fast_reflex() {
    let layer = StubReflexLayer::new();
    let input = test_input("test content for reflex layer");

    let budget = layer.latency_budget();
    let start = Instant::now();
    let result = layer.process(input).await;
    let elapsed = start.elapsed();

    // Must fail with NotImplemented (AP-007)
    assert!(result.is_err(), "Reflex layer stub must fail fast");
    assert!(matches!(result, Err(CoreError::NotImplemented(_))));

    // Failure must be fast (well under budget)
    assert!(
        elapsed <= budget,
        "Reflex layer took {:?} but budget is {:?}",
        elapsed,
        budget
    );
}

#[tokio::test]
async fn test_layer_fails_fast_memory() {
    let layer = StubMemoryLayer::new();
    let input = test_input("test content for memory layer");

    let budget = layer.latency_budget();
    let start = Instant::now();
    let result = layer.process(input).await;
    let elapsed = start.elapsed();

    // Must fail with NotImplemented (AP-007)
    assert!(result.is_err(), "Memory layer stub must fail fast");
    assert!(matches!(result, Err(CoreError::NotImplemented(_))));

    // Failure must be fast (well under budget)
    assert!(
        elapsed <= budget,
        "Memory layer took {:?} but budget is {:?}",
        elapsed,
        budget
    );
}

#[tokio::test]
async fn test_layer_fails_fast_learning() {
    let layer = StubLearningLayer::new();
    let input = test_input("test content for learning layer");

    let budget = layer.latency_budget();
    let start = Instant::now();
    let result = layer.process(input).await;
    let elapsed = start.elapsed();

    // Must fail with NotImplemented (AP-007)
    assert!(result.is_err(), "Learning layer stub must fail fast");
    assert!(matches!(result, Err(CoreError::NotImplemented(_))));

    // Failure must be fast (well under budget)
    assert!(
        elapsed <= budget,
        "Learning layer took {:?} but budget is {:?}",
        elapsed,
        budget
    );
}

#[tokio::test]
async fn test_layer_fails_fast_coherence() {
    let layer = StubCoherenceLayer::new();
    let input = test_input("test content for coherence layer");

    let budget = layer.latency_budget();
    let start = Instant::now();
    let result = layer.process(input).await;
    let elapsed = start.elapsed();

    // Must fail with NotImplemented (AP-007)
    assert!(result.is_err(), "Coherence layer stub must fail fast");
    assert!(matches!(result, Err(CoreError::NotImplemented(_))));

    // Failure must be fast (well under budget)
    assert!(
        elapsed <= budget,
        "Coherence layer took {:?} but budget is {:?}",
        elapsed,
        budget
    );
}

#[tokio::test]
async fn test_all_layers_fail_fast() {
    let layers: Vec<(Box<dyn NervousLayer>, &str)> = vec![
        (Box::new(StubSensingLayer::new()), "Sensing"),
        (Box::new(StubReflexLayer::new()), "Reflex"),
        (Box::new(StubMemoryLayer::new()), "Memory"),
        (Box::new(StubLearningLayer::new()), "Learning"),
        (Box::new(StubCoherenceLayer::new()), "Coherence"),
    ];

    for (layer, name) in layers {
        let input = LayerInput::new(
            "test-request".to_string(),
            format!("test content for {}", name),
        );

        let budget = layer.latency_budget();
        let start = Instant::now();
        let result = layer.process(input).await;
        let elapsed = start.elapsed();

        // Must fail with NotImplemented (AP-007)
        assert!(result.is_err(), "{} layer stub must fail fast", name);

        // Failure must be fast (well under budget)
        assert!(
            elapsed <= budget,
            "{} layer took {:?} but budget is {:?}",
            name,
            elapsed,
            budget
        );
    }
}

#[tokio::test]
async fn test_stub_layers_report_unhealthy() {
    let layers: Vec<Box<dyn NervousLayer>> = vec![
        Box::new(StubSensingLayer::new()),
        Box::new(StubReflexLayer::new()),
        Box::new(StubMemoryLayer::new()),
        Box::new(StubLearningLayer::new()),
        Box::new(StubCoherenceLayer::new()),
    ];

    for layer in layers {
        let health = layer.health_check().await.unwrap();
        assert!(
            !health,
            "{} stub should report unhealthy",
            layer.layer_name()
        );
    }
}

#[tokio::test]
async fn test_full_pipeline_all_layers_fail() {
    let sensing = StubSensingLayer::new();
    let reflex = StubReflexLayer::new();
    let memory = StubMemoryLayer::new();
    let learning = StubLearningLayer::new();
    let coherence = StubCoherenceLayer::new();

    // Total budget for error return: should be nearly instant
    let total_budget = Duration::from_millis(10);

    let input = test_input("full pipeline test content");
    let start = Instant::now();

    // All should fail with NotImplemented
    assert!(sensing.process(input.clone()).await.is_err());
    assert!(reflex.process(input.clone()).await.is_err());
    assert!(memory.process(input.clone()).await.is_err());
    assert!(learning.process(input.clone()).await.is_err());
    assert!(coherence.process(input).await.is_err());

    let elapsed = start.elapsed();

    assert!(
        elapsed <= total_budget,
        "Full pipeline took {:?} but error returns should be faster than {:?}",
        elapsed,
        total_budget
    );
}
