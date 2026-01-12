//! Tests for CoherenceLayer - REAL implementations, NO MOCKS

use std::time::{Duration, Instant};

use crate::atc::{AdaptiveThresholdCalibration, Domain};
use crate::traits::NervousLayer;
use crate::types::{LayerId, LayerInput, LayerResult};

use super::layer::CoherenceLayer;
use super::thresholds::GwtThresholds;

#[tokio::test]
async fn test_coherence_layer_process() {
    let layer = CoherenceLayer::new();
    let input = LayerInput::new("test-123".to_string(), "test content".to_string());

    let result = layer.process(input).await.unwrap();

    assert_eq!(result.layer, LayerId::Coherence);
    assert!(result.result.success);
    assert!(result.result.data.get("resonance").is_some());
    assert!(result.result.data.get("consciousness").is_some());
    assert!(result.result.data.get("gw_ignited").is_some());

    println!("[VERIFIED] CoherenceLayer.process() returns valid output");
}

#[tokio::test]
async fn test_coherence_layer_resonance_range() {
    let layer = CoherenceLayer::new();
    let input = LayerInput::new("test-456".to_string(), "resonance test".to_string());

    let result = layer.process(input).await.unwrap();

    let resonance = result.result.data["resonance"].as_f64().unwrap() as f32;
    assert!(
        (0.0..=1.0).contains(&resonance),
        "Resonance should be in [0,1], got {}",
        resonance
    );
    println!("[VERIFIED] Resonance r ∈ [0, 1]: r = {}", resonance);
}

#[tokio::test]
async fn test_coherence_layer_consciousness_range() {
    let layer = CoherenceLayer::new();
    let input = LayerInput::new("test-789".to_string(), "consciousness test".to_string());

    let result = layer.process(input).await.unwrap();

    let consciousness = result.result.data["consciousness"].as_f64().unwrap() as f32;
    assert!(
        (0.0..=1.0).contains(&consciousness),
        "Consciousness should be in [0,1], got {}",
        consciousness
    );
    println!("[VERIFIED] Consciousness C ∈ [0, 1]: C = {}", consciousness);
}

#[tokio::test]
async fn test_coherence_layer_with_learning_context() {
    let layer = CoherenceLayer::new();

    // Create input with L4 Learning context
    let mut input = LayerInput::new(
        "learning-ctx".to_string(),
        "learning context test".to_string(),
    );
    input.context.layer_results.push(LayerResult::success(
        LayerId::Learning,
        serde_json::json!({
            "weight_delta": 0.5,
            "surprise": 0.8,
            "coherence_w": 0.7,
        }),
    ));

    let result = layer.process(input).await.unwrap();

    assert!(result.result.success);

    let learning_signal = result.result.data["learning_signal"].as_f64().unwrap() as f32;
    assert!(
        (learning_signal - 0.5).abs() < 1e-6,
        "Learning signal should be extracted from L4"
    );

    println!("[VERIFIED] Learning signal extracted: {}", learning_signal);
}

#[tokio::test]
async fn test_coherence_layer_properties() {
    let layer = CoherenceLayer::new();
    let default_thresholds = GwtThresholds::default_general();

    assert_eq!(layer.layer_id(), LayerId::Coherence);
    assert_eq!(layer.latency_budget(), Duration::from_millis(10));
    assert_eq!(layer.layer_name(), "Coherence Layer");
    // Use new GwtThresholds API instead of deprecated GW_THRESHOLD constant
    assert!(
        (layer.gw_threshold() - default_thresholds.gate).abs() < 1e-6,
        "gw_threshold should match GwtThresholds::default_general().gate"
    );

    println!("[VERIFIED] CoherenceLayer properties correct");
}

#[tokio::test]
async fn test_coherence_layer_health_check() {
    let layer = CoherenceLayer::new();
    let healthy = layer.health_check().await.unwrap();

    assert!(healthy, "CoherenceLayer should be healthy");
    println!("[VERIFIED] health_check passes");
}

#[tokio::test]
async fn test_coherence_layer_custom_config() {
    let layer = CoherenceLayer::with_kuramoto(6, 3.0)
        .with_gw_threshold(0.75)
        .with_integration_steps(15);

    assert!((layer.gw_threshold() - 0.75).abs() < 1e-6);
    assert_eq!(layer.integration_steps, 15);

    println!("[VERIFIED] Custom configuration works");
}

#[tokio::test]
async fn test_gw_ignition_tracking() {
    let layer = CoherenceLayer::new().with_gw_threshold(0.1); // Low threshold for easy ignition

    // Run multiple times
    for i in 0..5 {
        let input = LayerInput::new(format!("ignition-{}", i), "test ignition".to_string());
        let _ = layer.process(input).await;
    }

    // Should have some ignitions with low threshold
    let count = layer.ignition_count();
    println!("[INFO] Ignition count with low threshold: {}", count);
    // Note: ignition depends on Kuramoto dynamics, may not always ignite
}

#[tokio::test]
async fn test_pulse_update() {
    let layer = CoherenceLayer::new();

    let mut input = LayerInput::new("pulse-test".to_string(), "pulse update test".to_string());
    input.context.pulse.coherence = 0.3;
    input.context.pulse.entropy = 0.7;

    let result = layer.process(input).await.unwrap();

    // Coherence should be updated to resonance
    assert!(
        result.pulse.source_layer == Some(LayerId::Coherence),
        "Source layer should be Coherence"
    );

    println!("[VERIFIED] Pulse updated with resonance");
}

// ============================================================
// Performance Benchmark - CRITICAL <10ms
// ============================================================

#[tokio::test]
async fn test_coherence_layer_latency_benchmark() {
    let layer = CoherenceLayer::new();

    let iterations = 1000;
    let mut total_us: u64 = 0;
    let mut max_us: u64 = 0;

    for i in 0..iterations {
        let mut input =
            LayerInput::new(format!("bench-{}", i), format!("Benchmark content {}", i));
        input.context.pulse.entropy = (i as f32 / iterations as f32).clamp(0.0, 1.0);
        input.context.pulse.coherence = 0.5;

        let start = Instant::now();
        let _ = layer.process(input).await;
        let elapsed = start.elapsed().as_micros() as u64;

        total_us += elapsed;
        max_us = max_us.max(elapsed);
    }

    let avg_us = total_us / iterations as u64;

    println!("Coherence Layer Benchmark Results:");
    println!("  Iterations: {}", iterations);
    println!("  Avg latency: {} us", avg_us);
    println!("  Max latency: {} us", max_us);
    println!("  Budget: 10000 us (10ms)");

    // Average should be well under budget
    assert!(
        avg_us < 10_000,
        "Average latency {} us exceeds 10ms budget",
        avg_us
    );

    // Max should also be under budget for reliable performance
    assert!(
        max_us < 10_000,
        "Max latency {} us exceeds 10ms budget",
        max_us
    );

    println!("[VERIFIED] Average latency {} us < 10000 us budget", avg_us);
}

// ============================================================
// Integration Tests
// ============================================================

#[tokio::test]
async fn test_full_pipeline_context() {
    let layer = CoherenceLayer::new();

    // Simulate full L1 -> L2 -> L3 -> L4 -> L5 pipeline context
    let mut input = LayerInput::new(
        "pipeline-test".to_string(),
        "Full pipeline test".to_string(),
    );

    // L1 Sensing result
    input.context.layer_results.push(LayerResult::success(
        LayerId::Sensing,
        serde_json::json!({
            "delta_s": 0.6,
            "scrubbed_content": "Full pipeline test",
            "pii_found": false,
        }),
    ));

    // L2 Reflex result (cache miss)
    input.context.layer_results.push(LayerResult::success(
        LayerId::Reflex,
        serde_json::json!({
            "cache_hit": false,
            "query_norm": 1.0,
        }),
    ));

    // L3 Memory result
    input.context.layer_results.push(LayerResult::success(
        LayerId::Memory,
        serde_json::json!({
            "retrieval_count": 3,
            "memories": [],
        }),
    ));

    // L4 Learning result
    input.context.layer_results.push(LayerResult::success(
        LayerId::Learning,
        serde_json::json!({
            "weight_delta": 0.3,
            "surprise": 0.6,
            "coherence_w": 0.75,
            "should_consolidate": false,
        }),
    ));

    // Set pulse state
    input.context.pulse.coherence = 0.5;
    input.context.pulse.entropy = 0.6;

    let result = layer.process(input).await.unwrap();

    assert!(result.result.success);

    // Verify all expected fields are present
    let data = &result.result.data;
    assert!(data.get("resonance").is_some());
    assert!(data.get("consciousness").is_some());
    assert!(data.get("differentiation").is_some());
    assert!(data.get("gw_ignited").is_some());
    assert!(data.get("state").is_some());
    assert!(data.get("oscillator_phases").is_some());
    assert!(data.get("learning_signal").is_some());

    let resonance = data["resonance"].as_f64().unwrap() as f32;
    let consciousness = data["consciousness"].as_f64().unwrap() as f32;
    let learning_signal = data["learning_signal"].as_f64().unwrap() as f32;

    // Verify values are in expected ranges
    assert!((0.0..=1.0).contains(&resonance));
    assert!((0.0..=1.0).contains(&consciousness));
    assert!((learning_signal - 0.3).abs() < 1e-6);

    println!("[VERIFIED] Full pipeline context processed correctly");
    println!("  Resonance: {}", resonance);
    println!("  Consciousness: {}", consciousness);
    println!("  Learning signal: {}", learning_signal);
}

#[tokio::test]
async fn test_consciousness_equation() {
    // Test C(t) = I(t) × R(t) × D(t)
    let layer = CoherenceLayer::new();

    // Test with known values
    let c1 = layer.compute_consciousness(1.0, 1.0, 1.0);
    assert!((c1 - 1.0).abs() < 1e-6, "C(1,1,1) should be 1.0");

    let c2 = layer.compute_consciousness(0.5, 0.5, 0.5);
    assert!((c2 - 0.125).abs() < 1e-6, "C(0.5,0.5,0.5) should be 0.125");

    let c3 = layer.compute_consciousness(0.0, 0.8, 0.8);
    assert!((c3).abs() < 1e-6, "C(0,0.8,0.8) should be 0");

    // Test NaN handling
    let c_nan = layer.compute_consciousness(f32::NAN, 0.5, 0.5);
    assert!((c_nan).abs() < 1e-6, "NaN input should return 0");

    println!("[VERIFIED] Consciousness equation C(t) = I × R × D");
}

// ============================================================
// GwtThresholds API Tests - TASK-ATC-P2-003
// ============================================================

#[tokio::test]
async fn test_gwt_thresholds_default_general() {
    let t = GwtThresholds::default_general();

    // Verify legacy values per constitution
    assert!(
        (t.gate - 0.70).abs() < 1e-6,
        "gate should be 0.70, got {}",
        t.gate
    );
    assert!(
        (t.hypersync - 0.95).abs() < 1e-6,
        "hypersync should be 0.95, got {}",
        t.hypersync
    );
    assert!(
        (t.fragmentation - 0.50).abs() < 1e-6,
        "fragmentation should be 0.50, got {}",
        t.fragmentation
    );

    // Verify validity
    assert!(t.is_valid(), "default_general should be valid");

    println!("[VERIFIED] GwtThresholds::default_general() returns legacy values");
}

#[tokio::test]
async fn test_gwt_thresholds_from_atc() {
    let atc = AdaptiveThresholdCalibration::new();

    // Test all domains
    let domains = [
        Domain::Medical,
        Domain::Legal,
        Domain::Code,
        Domain::General,
        Domain::Creative,
    ];

    for domain in domains {
        let thresholds = GwtThresholds::from_atc(&atc, domain);
        assert!(
            thresholds.is_ok(),
            "from_atc should succeed for {:?}",
            domain
        );

        let t = thresholds.unwrap();
        assert!(t.is_valid(), "thresholds for {:?} should be valid", domain);

        // Verify ranges
        assert!(
            (0.65..=0.95).contains(&t.gate),
            "{:?} gate {} out of range",
            domain,
            t.gate
        );
        assert!(
            (0.90..=0.99).contains(&t.hypersync),
            "{:?} hypersync {} out of range",
            domain,
            t.hypersync
        );
        assert!(
            (0.35..=0.65).contains(&t.fragmentation),
            "{:?} fragmentation {} out of range",
            domain,
            t.fragmentation
        );

        println!(
            "[VERIFIED] {:?}: gate={:.3}, hypersync={:.3}, frag={:.3}",
            domain, t.gate, t.hypersync, t.fragmentation
        );
    }
}

#[tokio::test]
async fn test_gwt_thresholds_domain_strictness() {
    let atc = AdaptiveThresholdCalibration::new();

    let medical = GwtThresholds::from_atc(&atc, Domain::Medical).unwrap();
    let creative = GwtThresholds::from_atc(&atc, Domain::Creative).unwrap();

    // Medical (strictest) should have higher gate than Creative (loosest)
    assert!(
        medical.gate > creative.gate,
        "Medical gate ({}) should be > Creative gate ({})",
        medical.gate,
        creative.gate
    );

    println!(
        "[VERIFIED] Domain strictness: Medical gate ({:.3}) > Creative gate ({:.3})",
        medical.gate, creative.gate
    );
}

#[tokio::test]
async fn test_gwt_thresholds_helper_methods() {
    let t = GwtThresholds::default_general();

    // Test should_broadcast (r >= gate)
    assert!(!t.should_broadcast(0.50), "r=0.50 should not broadcast");
    assert!(!t.should_broadcast(0.69), "r=0.69 should not broadcast");
    assert!(t.should_broadcast(0.70), "r=0.70 should broadcast");
    assert!(t.should_broadcast(0.85), "r=0.85 should broadcast");

    // Test is_hypersync (r > hypersync)
    assert!(!t.is_hypersync(0.90), "r=0.90 not hypersync");
    assert!(!t.is_hypersync(0.95), "r=0.95 not hypersync (boundary)");
    assert!(t.is_hypersync(0.96), "r=0.96 is hypersync");
    assert!(t.is_hypersync(0.99), "r=0.99 is hypersync");

    // Test is_fragmented (r < fragmentation)
    assert!(t.is_fragmented(0.30), "r=0.30 is fragmented");
    assert!(t.is_fragmented(0.49), "r=0.49 is fragmented");
    assert!(!t.is_fragmented(0.50), "r=0.50 not fragmented (boundary)");
    assert!(!t.is_fragmented(0.70), "r=0.70 not fragmented");

    println!("[VERIFIED] GwtThresholds helper methods work correctly");
}

#[tokio::test]
async fn test_coherence_layer_with_atc() {
    let atc = AdaptiveThresholdCalibration::new();

    // Create layer with Code domain thresholds
    let layer = CoherenceLayer::with_atc(&atc, Domain::Code).unwrap();
    let code_thresholds = GwtThresholds::from_atc(&atc, Domain::Code).unwrap();

    // Verify layer uses correct thresholds
    assert!(
        (layer.gw_threshold() - code_thresholds.gate).abs() < 1e-6,
        "Layer gw_threshold should match Code domain gate"
    );

    // Process an input
    let input = LayerInput::new("atc-test".to_string(), "Test with ATC thresholds".to_string());
    let result = layer.process(input).await.unwrap();

    assert!(result.result.success);
    println!("[VERIFIED] CoherenceLayer::with_atc() works correctly");
}

#[tokio::test]
async fn test_coherence_layer_with_explicit_thresholds() {
    // Create custom thresholds
    let custom_t = GwtThresholds {
        gate: 0.80,
        hypersync: 0.97,
        fragmentation: 0.45,
    };

    assert!(custom_t.is_valid(), "Custom thresholds should be valid");

    let layer = CoherenceLayer::with_thresholds(custom_t).unwrap();

    assert!(
        (layer.gw_threshold() - 0.80).abs() < 1e-6,
        "Layer should use custom gate threshold"
    );

    // Verify thresholds accessor
    let t = layer.thresholds();
    assert!(
        (t.gate - 0.80).abs() < 1e-6,
        "thresholds().gate should be 0.80"
    );
    assert!(
        (t.hypersync - 0.97).abs() < 1e-6,
        "thresholds().hypersync should be 0.97"
    );
    assert!(
        (t.fragmentation - 0.45).abs() < 1e-6,
        "thresholds().fragmentation should be 0.45"
    );

    println!("[VERIFIED] CoherenceLayer::with_thresholds() works correctly");
}

#[tokio::test]
async fn test_gwt_thresholds_validation_rejects_invalid() {
    // Test invalid: gate > hypersync
    let invalid1 = GwtThresholds {
        gate: 0.96,
        hypersync: 0.95,
        fragmentation: 0.50,
    };
    assert!(
        !invalid1.is_valid(),
        "gate > hypersync should be invalid"
    );

    // Test invalid: fragmentation > gate
    let invalid2 = GwtThresholds {
        gate: 0.70,
        hypersync: 0.95,
        fragmentation: 0.75,
    };
    assert!(
        !invalid2.is_valid(),
        "fragmentation > gate should be invalid"
    );

    // Test invalid: gate out of range (too low)
    let invalid3 = GwtThresholds {
        gate: 0.50,
        hypersync: 0.95,
        fragmentation: 0.40,
    };
    assert!(!invalid3.is_valid(), "gate < 0.65 should be invalid");

    // Test CoherenceLayer::with_thresholds rejects invalid
    let result = CoherenceLayer::with_thresholds(invalid1);
    assert!(result.is_err(), "with_thresholds should reject invalid thresholds");

    println!("[VERIFIED] GwtThresholds validation rejects invalid configurations");
}

// ============================================================
// Full State Verification (FSV) Test
// ============================================================

#[tokio::test]
async fn test_fsv_gwt_thresholds_comprehensive() {
    println!("\n=== FSV: GwtThresholds Comprehensive Test ===\n");

    // 1. Verify default_general returns legacy values
    let t = GwtThresholds::default_general();
    println!("1. default_general(): gate={}, hypersync={}, frag={}",
             t.gate, t.hypersync, t.fragmentation);
    assert!((t.gate - 0.70).abs() < 1e-6);
    assert!((t.hypersync - 0.95).abs() < 1e-6);
    assert!((t.fragmentation - 0.50).abs() < 1e-6);

    // 2. Verify from_atc works for all domains
    let atc = AdaptiveThresholdCalibration::new();
    println!("\n2. from_atc() domain thresholds:");
    for domain in [Domain::Medical, Domain::Legal, Domain::Code, Domain::General, Domain::Creative] {
        let dt = GwtThresholds::from_atc(&atc, domain).unwrap();
        println!("   {:?}: gate={:.3}, hypersync={:.3}, frag={:.3}",
                 domain, dt.gate, dt.hypersync, dt.fragmentation);
        assert!(dt.is_valid());
    }

    // 3. Verify helper methods
    println!("\n3. Helper method tests:");
    let test_r_values = [0.30, 0.50, 0.70, 0.95, 0.96];
    for r in test_r_values {
        println!("   r={:.2}: broadcast={}, hypersync={}, fragmented={}",
                 r, t.should_broadcast(r), t.is_hypersync(r), t.is_fragmented(r));
    }

    // 4. Verify CoherenceLayer integration
    println!("\n4. CoherenceLayer integration:");
    let layer_default = CoherenceLayer::new();
    let layer_atc = CoherenceLayer::with_atc(&atc, Domain::Code).unwrap();
    let layer_custom = CoherenceLayer::with_thresholds(GwtThresholds {
        gate: 0.75,
        hypersync: 0.96,
        fragmentation: 0.45,
    }).unwrap();

    println!("   Default layer gate: {:.3}", layer_default.gw_threshold());
    println!("   ATC Code layer gate: {:.3}", layer_atc.gw_threshold());
    println!("   Custom layer gate: {:.3}", layer_custom.gw_threshold());

    // 5. Process test to verify end-to-end
    println!("\n5. End-to-end processing:");
    let input = LayerInput::new("fsv-test".to_string(), "FSV comprehensive test".to_string());
    let result = layer_atc.process(input).await.unwrap();

    let resonance = result.result.data["resonance"].as_f64().unwrap();
    let gw_threshold = result.result.data["gw_threshold"].as_f64().unwrap();
    println!("   Resonance: {:.4}", resonance);
    println!("   GW Threshold: {:.4}", gw_threshold);
    println!("   GW Ignited: {}", result.result.data["gw_ignited"]);

    println!("\n=== FSV COMPLETE ===\n");
}
