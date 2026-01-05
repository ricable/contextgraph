//! Edge case tests for modularized layers verification.
//!
//! This module tests boundary conditions to verify the modularization
//! maintains correct behavior in edge cases.

use crate::traits::NervousLayer;
use crate::types::LayerInput;

use super::{
    StubCoherenceLayer, StubLearningLayer, StubMemoryLayer, StubReflexLayer, StubSensingLayer,
};

// =============================================================================
// EDGE CASE 1: Empty Input Handling
// =============================================================================

#[tokio::test]
async fn test_edge_case_empty_input_sensing() {
    println!("\n=== EDGE CASE 1: Empty Input - Sensing Layer ===");
    let layer = StubSensingLayer::new();
    let input = LayerInput::new("test".to_string(), "".to_string());

    println!("BEFORE STATE: input.content = \"{}\" (len={})", input.content, input.content.len());

    let output = layer.process(input).await.expect("Should handle empty input");

    println!("AFTER STATE:");
    println!("  - result.success = {}", output.result.success);
    println!("  - pulse.entropy = {}", output.pulse.entropy);
    println!("  - pulse.coherence = {}", output.pulse.coherence);
    println!("  - duration_us = {}", output.duration_us);

    // SOURCE OF TRUTH VERIFICATION
    assert!(output.result.success, "Empty input should be processed successfully");
    assert!(output.pulse.entropy >= 0.0, "Entropy should be non-negative");
    assert!(output.pulse.coherence >= 0.0, "Coherence should be non-negative");

    println!("VERIFICATION: Empty input handled correctly ✓");
}

#[tokio::test]
async fn test_edge_case_empty_input_all_layers() {
    println!("\n=== EDGE CASE 1b: Empty Input - All Layers ===");

    let layers: Vec<(Box<dyn NervousLayer>, &str)> = vec![
        (Box::new(StubSensingLayer::new()), "Sensing"),
        (Box::new(StubReflexLayer::new()), "Reflex"),
        (Box::new(StubMemoryLayer::new()), "Memory"),
        (Box::new(StubLearningLayer::new()), "Learning"),
        (Box::new(StubCoherenceLayer::new()), "Coherence"),
    ];

    for (layer, name) in layers {
        let input = LayerInput::new("test".to_string(), "".to_string());
        let output = layer.process(input).await.unwrap_or_else(|_| panic!("{} should handle empty input", name));

        println!("{}: success={}, entropy={:.4}, coherence={:.4}",
            name, output.result.success, output.pulse.entropy, output.pulse.coherence);

        assert!(output.result.success, "{} should succeed with empty input", name);
    }

    println!("VERIFICATION: All layers handle empty input correctly ✓");
}

// =============================================================================
// EDGE CASE 2: Large Input Handling
// =============================================================================

#[tokio::test]
async fn test_edge_case_large_input() {
    println!("\n=== EDGE CASE 2: Large Input (10KB) ===");

    let large_content = "x".repeat(10_000);
    let layer = StubSensingLayer::new();
    let input = LayerInput::new("test".to_string(), large_content.clone());

    println!("BEFORE STATE: input.content.len() = {} bytes", input.content.len());

    let output = layer.process(input).await.expect("Should handle large input");

    println!("AFTER STATE:");
    println!("  - result.success = {}", output.result.success);
    println!("  - duration_us = {} (budget: 5000us)", output.duration_us);

    // SOURCE OF TRUTH VERIFICATION
    let budget_us = layer.latency_budget().as_micros() as u64;
    assert!(output.result.success, "Large input should be processed");
    assert!(output.duration_us < budget_us, "Should complete within budget");

    println!("VERIFICATION: Large input processed within budget ✓");
}

// =============================================================================
// EDGE CASE 3: Entropy/Coherence Boundary Values
// =============================================================================

#[tokio::test]
async fn test_edge_case_boundary_values() {
    println!("\n=== EDGE CASE 3: Entropy/Coherence Range Validation ===");

    let layers: Vec<(Box<dyn NervousLayer>, &str)> = vec![
        (Box::new(StubSensingLayer::new()), "Sensing"),
        (Box::new(StubReflexLayer::new()), "Reflex"),
        (Box::new(StubMemoryLayer::new()), "Memory"),
        (Box::new(StubLearningLayer::new()), "Learning"),
        (Box::new(StubCoherenceLayer::new()), "Coherence"),
    ];

    // Test with various input patterns that exercise hash boundaries
    let large_input = "x".repeat(1000);
    let test_inputs = vec!["", "a", "test", "boundary value test", large_input.as_str()];

    for (layer, name) in &layers {
        for test_content in &test_inputs {
            let input = LayerInput::new("test".to_string(), test_content.to_string());
            let output = layer.process(input).await.unwrap();

            // SOURCE OF TRUTH VERIFICATION
            assert!(
                output.pulse.entropy >= 0.0 && output.pulse.entropy <= 1.0,
                "{} entropy {} out of range for input len {}",
                name, output.pulse.entropy, test_content.len()
            );
            assert!(
                output.pulse.coherence >= 0.0 && output.pulse.coherence <= 1.0,
                "{} coherence {} out of range for input len {}",
                name, output.pulse.coherence, test_content.len()
            );
        }
        println!("{}: All boundary values within [0.0, 1.0] range ✓", name);
    }

    println!("\nVERIFICATION: All entropy/coherence values within valid range ✓");
}

// =============================================================================
// SOURCE OF TRUTH VERIFICATION: Module Structure
// =============================================================================

#[test]
fn test_source_of_truth_module_structure() {
    println!("\n=== SOURCE OF TRUTH: Module Re-exports ===");

    // Verify all types are accessible from the expected paths
    let _sensing: StubSensingLayer = StubSensingLayer::new();
    let _reflex: StubReflexLayer = StubReflexLayer::new();
    let _memory: StubMemoryLayer = StubMemoryLayer::new();
    let _learning: StubLearningLayer = StubLearningLayer::new();
    let _coherence: StubCoherenceLayer = StubCoherenceLayer::new();

    println!("StubSensingLayer: accessible ✓");
    println!("StubReflexLayer: accessible ✓");
    println!("StubMemoryLayer: accessible ✓");
    println!("StubLearningLayer: accessible ✓");
    println!("StubCoherenceLayer: accessible ✓");

    println!("\nVERIFICATION: All layer types properly re-exported ✓");
}
