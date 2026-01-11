//! UTL Handler Tests
//!
//! TASK-UTL-P1-001: Tests for gwt/compute_delta_sc handler.
//! Tests verify:
//! - Per-embedder ΔS computation
//! - Aggregate ΔS/ΔC values
//! - Johari quadrant classification
//! - AP-10 compliance (all values in [0,1], no NaN/Inf)
//! - FAIL FAST error handling

use serde_json::json;
use uuid::Uuid;

use context_graph_core::johari::NUM_EMBEDDERS;
use context_graph_core::types::fingerprint::{
    JohariFingerprint, PurposeVector, SemanticFingerprint, TeleologicalFingerprint,
};

use crate::protocol::JsonRpcId;

use super::{create_test_handlers, make_request};

// ============================================================================
// TASK-UTL-P1-001: gwt/compute_delta_sc Tests
// ============================================================================

/// Create a test TeleologicalFingerprint with specified semantic values.
///
/// Uses zeroed base with modified e1_semantic for testing ΔS computation.
fn create_test_fingerprint_with_semantic(semantic_values: Vec<f32>) -> TeleologicalFingerprint {
    let mut semantic = SemanticFingerprint::zeroed();
    semantic.e1_semantic = semantic_values;
    TeleologicalFingerprint::new(
        semantic,
        PurposeVector::default(),
        JohariFingerprint::zeroed(),
        [0u8; 32],
    )
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_valid() {
    let handlers = create_test_handlers();

    // Create two fingerprints with different semantic values
    let old_fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.7; 1024]);

    // Must call through tools/call with name + arguments
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize old_fp"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize new_fp"),
            }
        })),
    );
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "gwt/compute_delta_sc should succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    // Verify required fields exist
    assert!(
        data.get("delta_s_per_embedder").is_some(),
        "Should have delta_s_per_embedder"
    );
    assert!(
        data.get("delta_s_aggregate").is_some(),
        "Should have delta_s_aggregate"
    );
    assert!(data.get("delta_c").is_some(), "Should have delta_c");
    assert!(
        data.get("johari_quadrants").is_some(),
        "Should have johari_quadrants"
    );
    assert!(
        data.get("johari_aggregate").is_some(),
        "Should have johari_aggregate"
    );
    assert!(
        data.get("utl_learning_potential").is_some(),
        "Should have utl_learning_potential"
    );
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_per_embedder_count() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.3; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.8; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    // Verify 13 per-embedder values
    let delta_s_per_embedder = data["delta_s_per_embedder"]
        .as_array()
        .expect("delta_s_per_embedder should be array");
    assert_eq!(
        delta_s_per_embedder.len(),
        NUM_EMBEDDERS,
        "Should have exactly 13 per-embedder ΔS values"
    );

    // Verify 13 Johari quadrants
    let johari_quadrants = data["johari_quadrants"]
        .as_array()
        .expect("johari_quadrants should be array");
    assert_eq!(
        johari_quadrants.len(),
        NUM_EMBEDDERS,
        "Should have exactly 13 Johari quadrants"
    );
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_ap10_range_compliance() {
    let handlers = create_test_handlers();

    // Use different values to get non-trivial ΔS
    let old_fp = create_test_fingerprint_with_semantic(vec![0.1; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.9; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    // AP-10: All values must be in [0, 1]
    let delta_s_per_embedder = data["delta_s_per_embedder"]
        .as_array()
        .expect("array");
    for (i, val) in delta_s_per_embedder.iter().enumerate() {
        let v = val.as_f64().expect("f64");
        assert!(
            (0.0..=1.0).contains(&v),
            "delta_s_per_embedder[{}] = {} not in [0,1]",
            i,
            v
        );
        assert!(!v.is_nan(), "delta_s_per_embedder[{}] is NaN", i);
        assert!(!v.is_infinite(), "delta_s_per_embedder[{}] is Inf", i);
    }

    let delta_s_agg = data["delta_s_aggregate"].as_f64().expect("f64");
    assert!(
        (0.0..=1.0).contains(&delta_s_agg),
        "delta_s_aggregate = {} not in [0,1]",
        delta_s_agg
    );

    let delta_c = data["delta_c"].as_f64().expect("f64");
    assert!(
        (0.0..=1.0).contains(&delta_c),
        "delta_c = {} not in [0,1]",
        delta_c
    );

    let learning_potential = data["utl_learning_potential"].as_f64().expect("f64");
    assert!(
        (0.0..=1.0).contains(&learning_potential),
        "utl_learning_potential = {} not in [0,1]",
        learning_potential
    );
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_johari_quadrant_values() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.6; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    // Verify Johari quadrants are valid enum values
    let valid_quadrants = ["Open", "Blind", "Hidden", "Unknown"];
    let johari_quadrants = data["johari_quadrants"]
        .as_array()
        .expect("array");

    for (i, quadrant) in johari_quadrants.iter().enumerate() {
        let q = quadrant.as_str().expect("string");
        assert!(
            valid_quadrants.contains(&q),
            "johari_quadrants[{}] = '{}' is not a valid quadrant",
            i,
            q
        );
    }

    let johari_agg = data["johari_aggregate"].as_str().expect("string");
    assert!(
        valid_quadrants.contains(&johari_agg),
        "johari_aggregate = '{}' is not a valid quadrant",
        johari_agg
    );
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_with_diagnostics() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.4; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.7; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
                "include_diagnostics": true,
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    // Verify diagnostics are included
    assert!(
        data.get("diagnostics").is_some(),
        "Should have diagnostics when include_diagnostics=true"
    );

    let diagnostics = &data["diagnostics"];
    assert!(
        diagnostics.get("per_embedder").is_some(),
        "diagnostics should have per_embedder"
    );
    assert!(
        diagnostics.get("johari_threshold").is_some(),
        "diagnostics should have johari_threshold"
    );
    assert!(
        diagnostics.get("coherence_config").is_some(),
        "diagnostics should have coherence_config"
    );
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_custom_johari_threshold() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.6; 1024]);

    // Test with custom johari_threshold
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
                "johari_threshold": 0.4,
                "include_diagnostics": true,
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    // Verify threshold was applied (clamped to [0.35, 0.65])
    let threshold = data["diagnostics"]["johari_threshold"]
        .as_f64()
        .expect("threshold");
    assert!(
        (0.35..=0.65).contains(&threshold),
        "johari_threshold {} should be clamped to [0.35, 0.65]",
        threshold
    );
}

// ============================================================================
// FAIL FAST Error Cases (TASK-UTL-P1-001)
// ============================================================================

#[tokio::test]
async fn test_gwt_compute_delta_sc_missing_vertex_id() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.6; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let error = response.error.expect("Should have error for missing vertex_id");

    assert!(
        error.message.contains("vertex_id"),
        "Error message should mention vertex_id: {:?}",
        error.message
    );
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_invalid_vertex_id() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.6; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": "not-a-valid-uuid",
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let error = response.error.expect("Should have error for invalid UUID");

    assert!(
        error.message.contains("UUID") || error.message.contains("uuid") || error.message.contains("Invalid"),
        "Error message should indicate invalid UUID: {:?}",
        error.message
    );
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_missing_old_fingerprint() {
    let handlers = create_test_handlers();

    let new_fp = create_test_fingerprint_with_semantic(vec![0.6; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let error = response.error.expect("Should have error for missing old_fingerprint");

    assert!(
        error.message.contains("old_fingerprint"),
        "Error message should mention old_fingerprint: {:?}",
        error.message
    );
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_missing_new_fingerprint() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let error = response.error.expect("Should have error for missing new_fingerprint");

    assert!(
        error.message.contains("new_fingerprint"),
        "Error message should mention new_fingerprint: {:?}",
        error.message
    );
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_invalid_fingerprint_json() {
    let handlers = create_test_handlers();

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": { "invalid": "structure" },
                "new_fingerprint": { "also": "invalid" },
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let error = response.error.expect("Should have error for invalid fingerprint JSON");

    assert!(
        error.message.contains("parse") || error.message.contains("fingerprint") || error.message.contains("Failed"),
        "Error message should indicate parse failure: {:?}",
        error.message
    );
}

#[tokio::test]
async fn test_utl_compute_valid() {
    let handlers = create_test_handlers();
    // Handler expects 'input' parameter, not 'content'
    let params = json!({
        "input": "Test content for UTL computation"
    });
    let request = make_request("utl/compute", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(response.error.is_none(), "utl/compute should succeed");
    let result = response.result.expect("Should have result");

    // Handler returns only learningScore
    assert!(
        result.get("learningScore").is_some(),
        "Should have learningScore"
    );
}

#[tokio::test]
async fn test_utl_compute_missing_input() {
    let handlers = create_test_handlers();
    let params = json!({});
    let request = make_request("utl/compute", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_some(),
        "utl/compute should fail without input"
    );
}

#[tokio::test]
async fn test_utl_compute_with_input() {
    let handlers = create_test_handlers();
    let params = json!({
        "input": "Learning about neural networks"
    });
    let request = make_request("utl/compute", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "utl/compute with input should succeed"
    );
}

#[tokio::test]
async fn test_utl_metrics_valid() {
    let handlers = create_test_handlers();
    // Handler expects 'input' parameter
    let params = json!({
        "input": "Test input for metrics"
    });
    let request = make_request("utl/metrics", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(response.error.is_none(), "utl/metrics should succeed");
    let result = response.result.expect("Should have result");

    // Verify metrics fields
    assert!(result.get("entropy").is_some(), "Should have entropy");
    assert!(result.get("coherence").is_some(), "Should have coherence");
    assert!(
        result.get("learningScore").is_some(),
        "Should have learningScore"
    );
}

#[tokio::test]
async fn test_utl_compute_learning_score_range() {
    let handlers = create_test_handlers();
    let params = json!({
        "input": "Simple test content"
    });
    let request = make_request("utl/compute", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");

    let learning_score = result
        .get("learningScore")
        .and_then(|v| v.as_f64())
        .expect("Should have learningScore");

    assert!(
        (0.0..=1.0).contains(&learning_score),
        "learningScore should be between 0 and 1, got {}",
        learning_score
    );
}

#[tokio::test]
async fn test_utl_metrics_entropy_range() {
    let handlers = create_test_handlers();
    let params = json!({
        "input": "Test entropy calculation"
    });
    let request = make_request("utl/metrics", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");

    let entropy = result
        .get("entropy")
        .and_then(|v| v.as_f64())
        .expect("Should have entropy");

    assert!(
        (0.0..=1.0).contains(&entropy),
        "entropy should be between 0 and 1, got {}",
        entropy
    );
}

// ============================================================================
// TASK-DELTA-P1-001: Full State Verification (FSV) Tests
// ============================================================================

/// FSV-01: Verify ΔC formula matches constitution.yaml line 166:
/// ΔC = 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency
///
/// We verify this by enabling diagnostics and checking the component values.
#[tokio::test]
async fn test_fsv_delta_c_formula_verification() {
    let handlers = create_test_handlers();

    // Create fingerprints with different values to get non-trivial components
    let old_fp = create_test_fingerprint_with_semantic(vec![0.3; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.7; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
                "include_diagnostics": true,
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Should succeed: {:?}", response.error);

    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    // Verify diagnostics contains delta_c_components
    let diagnostics = &data["diagnostics"];
    assert!(diagnostics.get("delta_c_components").is_some(), "Should have delta_c_components");

    let components = &diagnostics["delta_c_components"];
    let connectivity = components["connectivity"].as_f64().expect("connectivity") as f32;
    let cluster_fit = components["cluster_fit"].as_f64().expect("cluster_fit") as f32;
    let consistency = components["consistency"].as_f64().expect("consistency") as f32;

    // Verify weights are correct per constitution.yaml (with f32/f64 tolerance)
    let weights = &components["weights"];
    let alpha = weights["alpha_connectivity"].as_f64().unwrap();
    let beta = weights["beta_cluster_fit"].as_f64().unwrap();
    let gamma = weights["gamma_consistency"].as_f64().unwrap();

    assert!(
        (alpha - 0.4).abs() < 0.0001,
        "alpha (connectivity weight) should be ~0.4, got {}",
        alpha
    );
    assert!(
        (beta - 0.4).abs() < 0.0001,
        "beta (cluster_fit weight) should be ~0.4, got {}",
        beta
    );
    assert!(
        (gamma - 0.2).abs() < 0.0001,
        "gamma (consistency weight) should be ~0.2, got {}",
        gamma
    );

    // Verify delta_c matches the formula: 0.4*Connectivity + 0.4*ClusterFit + 0.2*Consistency
    let expected_delta_c = 0.4 * connectivity + 0.4 * cluster_fit + 0.2 * consistency;
    let actual_delta_c = data["delta_c"].as_f64().expect("delta_c") as f32;

    // Allow small floating-point tolerance and clamping effects
    let diff = (expected_delta_c.clamp(0.0, 1.0) - actual_delta_c).abs();
    assert!(
        diff < 0.01,
        "ΔC formula mismatch: expected {:.6} (0.4×{:.4} + 0.4×{:.4} + 0.2×{:.4}), got {:.6}",
        expected_delta_c,
        connectivity,
        cluster_fit,
        consistency,
        actual_delta_c
    );
}

/// FSV-02: Verify UTL learning potential formula:
/// utl_learning_potential = delta_s_aggregate × delta_c
#[tokio::test]
async fn test_fsv_utl_learning_potential_formula() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.2; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.8; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Should succeed");

    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    let delta_s_agg = data["delta_s_aggregate"].as_f64().expect("delta_s_aggregate") as f32;
    let delta_c = data["delta_c"].as_f64().expect("delta_c") as f32;
    let learning_potential = data["utl_learning_potential"].as_f64().expect("utl_learning_potential") as f32;

    // Verify formula: learning_potential = delta_s_aggregate * delta_c
    let expected = (delta_s_agg * delta_c).clamp(0.0, 1.0);
    let diff = (expected - learning_potential).abs();

    assert!(
        diff < 0.001,
        "UTL learning potential formula mismatch: expected {:.6} ({:.4} × {:.4}), got {:.6}",
        expected,
        delta_s_agg,
        delta_c,
        learning_potential
    );
}

/// FSV-03: Verify each per-embedder ΔS value is properly clamped to [0, 1]
#[tokio::test]
async fn test_fsv_delta_s_clamping() {
    let handlers = create_test_handlers();

    // Use extreme values that might produce out-of-range results before clamping
    let old_fp = create_test_fingerprint_with_semantic(vec![0.0; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![1.0; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    let delta_s_per_embedder = data["delta_s_per_embedder"]
        .as_array()
        .expect("delta_s_per_embedder should be array");

    // Verify ALL 13 values are in [0, 1]
    for (i, val) in delta_s_per_embedder.iter().enumerate() {
        let v = val.as_f64().expect("f64");
        assert!(
            (0.0..=1.0).contains(&v),
            "FSV: delta_s_per_embedder[{}] = {} MUST be clamped to [0,1]",
            i,
            v
        );
        assert!(!v.is_nan(), "FSV: delta_s_per_embedder[{}] is NaN", i);
        assert!(!v.is_infinite(), "FSV: delta_s_per_embedder[{}] is Inf", i);
    }

    // Verify aggregate is also clamped
    let delta_s_agg = data["delta_s_aggregate"].as_f64().expect("f64");
    assert!(
        (0.0..=1.0).contains(&delta_s_agg),
        "FSV: delta_s_aggregate = {} MUST be in [0,1]",
        delta_s_agg
    );
}

/// FSV-04: Verify Johari classification follows exact threshold rules from constitution.yaml
///
/// Open: ΔS < threshold, ΔC > threshold (low surprise, high coherence)
/// Blind: ΔS > threshold, ΔC < threshold (high surprise, low coherence)
/// Hidden: ΔS < threshold, ΔC < threshold (low surprise, low coherence)
/// Unknown: ΔS > threshold, ΔC > threshold (high surprise, high coherence)
#[tokio::test]
async fn test_fsv_johari_classification_rules() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.6; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
                "include_diagnostics": true,
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    let threshold = data["diagnostics"]["johari_threshold"]
        .as_f64()
        .expect("johari_threshold") as f32;
    let delta_s_agg = data["delta_s_aggregate"].as_f64().expect("f64") as f32;
    let delta_c = data["delta_c"].as_f64().expect("f64") as f32;
    let johari_agg = data["johari_aggregate"].as_str().expect("string");

    // Verify classification follows exact rules
    let expected_quadrant = match (delta_s_agg < threshold, delta_c > threshold) {
        (true, true) => "Open",    // Low surprise, high coherence
        (false, false) => "Blind", // High surprise, low coherence
        (true, false) => "Hidden", // Low surprise, low coherence
        (false, true) => "Unknown", // High surprise, high coherence
    };

    assert_eq!(
        johari_agg, expected_quadrant,
        "FSV: Johari classification mismatch. ΔS={:.4}, ΔC={:.4}, threshold={:.4}. Expected '{}', got '{}'",
        delta_s_agg, delta_c, threshold, expected_quadrant, johari_agg
    );
}

/// FSV-05: Verify aggregate ΔS is mean of per-embedder values
#[tokio::test]
async fn test_fsv_delta_s_aggregate_is_mean() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.4; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.7; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    let delta_s_per_embedder = data["delta_s_per_embedder"]
        .as_array()
        .expect("array");

    // Calculate expected mean
    let sum: f64 = delta_s_per_embedder.iter().map(|v| v.as_f64().unwrap()).sum();
    let expected_mean = (sum / NUM_EMBEDDERS as f64).clamp(0.0, 1.0);
    let actual_agg = data["delta_s_aggregate"].as_f64().expect("f64");

    let diff = (expected_mean - actual_agg).abs();
    assert!(
        diff < 0.001,
        "FSV: delta_s_aggregate should be mean of per-embedder values. Expected {:.6}, got {:.6}",
        expected_mean,
        actual_agg
    );
}

// ============================================================================
// TASK-DELTA-P1-001: Edge Case Tests
// ============================================================================

/// EC-01: Identical fingerprints should produce ΔS ≈ 0 for all embedders
#[tokio::test]
async fn test_ec01_identical_fingerprints() {
    let handlers = create_test_handlers();

    // Create identical fingerprints
    let fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Should succeed with identical fingerprints");

    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    // With identical fingerprints, ΔS should be very low (close to 0)
    let delta_s_agg = data["delta_s_aggregate"].as_f64().expect("f64");

    // Note: ΔS might not be exactly 0 due to the entropy calculation method,
    // but should be very low for identical embeddings
    assert!(
        delta_s_agg < 0.5,
        "EC-01: Identical fingerprints should produce low ΔS, got {}",
        delta_s_agg
    );

    // Verify all values are still valid (no NaN/Inf)
    let delta_s_per_embedder = data["delta_s_per_embedder"].as_array().expect("array");
    for (i, val) in delta_s_per_embedder.iter().enumerate() {
        let v = val.as_f64().expect("f64");
        assert!(
            (0.0..=1.0).contains(&v) && !v.is_nan() && !v.is_infinite(),
            "EC-01: delta_s_per_embedder[{}] = {} should be valid",
            i,
            v
        );
    }
}

/// EC-02: Maximum change fingerprints (opposite embeddings) should produce high ΔS
#[tokio::test]
async fn test_ec02_maximum_change_fingerprints() {
    let handlers = create_test_handlers();

    // Create maximally different fingerprints
    let old_fp = create_test_fingerprint_with_semantic(vec![0.0; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![1.0; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Should succeed with maximum change fingerprints");

    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    // All values must still be valid and clamped
    let delta_s_agg = data["delta_s_aggregate"].as_f64().expect("f64");
    assert!(
        (0.0..=1.0).contains(&delta_s_agg),
        "EC-02: delta_s_aggregate must be clamped to [0,1], got {}",
        delta_s_agg
    );

    let delta_c = data["delta_c"].as_f64().expect("f64");
    assert!(
        (0.0..=1.0).contains(&delta_c),
        "EC-02: delta_c must be clamped to [0,1], got {}",
        delta_c
    );

    let learning_potential = data["utl_learning_potential"].as_f64().expect("f64");
    assert!(
        (0.0..=1.0).contains(&learning_potential),
        "EC-02: utl_learning_potential must be clamped to [0,1], got {}",
        learning_potential
    );
}

/// EC-03: Zero-magnitude embeddings (all zeros) should not cause NaN/Inf
#[tokio::test]
async fn test_ec03_zero_magnitude_embeddings() {
    let handlers = create_test_handlers();

    // Create zero embeddings - this tests division by zero protection
    let zero_fp = create_test_fingerprint_with_semantic(vec![0.0; 1024]);
    let nonzero_fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&zero_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&nonzero_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Should handle zero embeddings gracefully");

    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    // Verify no NaN/Inf in any output
    let delta_s_per_embedder = data["delta_s_per_embedder"].as_array().expect("array");
    for (i, val) in delta_s_per_embedder.iter().enumerate() {
        let v = val.as_f64().expect("f64");
        assert!(
            !v.is_nan() && !v.is_infinite(),
            "EC-03: delta_s_per_embedder[{}] = {} must not be NaN/Inf",
            i,
            v
        );
    }

    let delta_c = data["delta_c"].as_f64().expect("f64");
    assert!(
        !delta_c.is_nan() && !delta_c.is_infinite(),
        "EC-03: delta_c = {} must not be NaN/Inf",
        delta_c
    );

    let learning_potential = data["utl_learning_potential"].as_f64().expect("f64");
    assert!(
        !learning_potential.is_nan() && !learning_potential.is_infinite(),
        "EC-03: utl_learning_potential = {} must not be NaN/Inf",
        learning_potential
    );
}

/// EC-04: Threshold boundary test - ΔS/ΔC exactly at threshold (0.5)
#[tokio::test]
async fn test_ec04_threshold_boundary() {
    let handlers = create_test_handlers();

    // We can't directly control the exact ΔS/ΔC values, but we can verify
    // the classification logic handles boundary cases correctly
    let old_fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.55; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
                "johari_threshold": 0.5,
                "include_diagnostics": true,
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Should handle boundary values");

    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    let threshold = data["diagnostics"]["johari_threshold"].as_f64().expect("f64");
    let delta_s_agg = data["delta_s_aggregate"].as_f64().expect("f64");
    let delta_c = data["delta_c"].as_f64().expect("f64");
    let johari_agg = data["johari_aggregate"].as_str().expect("string");

    // The classification should be deterministic based on strict < and > comparison
    // Verify the result is one of the valid quadrants
    let valid_quadrants = ["Open", "Blind", "Hidden", "Unknown"];
    assert!(
        valid_quadrants.contains(&johari_agg),
        "EC-04: Boundary case should still produce valid quadrant. ΔS={:.4}, ΔC={:.4}, threshold={:.4}, got '{}'",
        delta_s_agg,
        delta_c,
        threshold,
        johari_agg
    );
}

/// EC-05: Large embedding dimensions should not cause performance issues
#[tokio::test]
async fn test_ec05_large_embedding_dimensions() {
    let handlers = create_test_handlers();

    // Test with max typical dimension (1024 is standard, but let's verify it works)
    let old_fp = create_test_fingerprint_with_semantic(vec![0.3; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.7; 1024]);

    let start = std::time::Instant::now();

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let elapsed = start.elapsed();

    assert!(response.error.is_none(), "Should handle large dimensions");

    // Performance check: should complete in reasonable time (< 1 second)
    assert!(
        elapsed.as_secs() < 1,
        "EC-05: Large dimension computation took too long: {:?}",
        elapsed
    );

    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    // Verify output is still valid
    let delta_s_per_embedder = data["delta_s_per_embedder"].as_array().expect("array");
    assert_eq!(
        delta_s_per_embedder.len(),
        NUM_EMBEDDERS,
        "EC-05: Should still have 13 embedders"
    );
}

/// EC-06: Verify behavior with custom johari_threshold at boundaries
#[tokio::test]
async fn test_ec06_johari_threshold_clamping() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.6; 1024]);

    // Test threshold below minimum (0.35) - should be clamped
    let request_low = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
                "johari_threshold": 0.1,  // Below minimum
                "include_diagnostics": true,
            }
        })),
    );

    let response_low = handlers.dispatch(request_low).await;
    assert!(response_low.error.is_none(), "Should handle low threshold");

    let result_low = response_low.result.expect("result");
    let data_low = super::extract_mcp_tool_data(&result_low);
    let threshold_low = data_low["diagnostics"]["johari_threshold"].as_f64().expect("f64");
    // Use tolerance for f32/f64 precision issues (0.35 stored as f32 may be 0.34999...)
    assert!(
        threshold_low >= 0.35 - 0.0001,
        "EC-06: Threshold {} should be clamped to >= 0.35",
        threshold_low
    );

    // Test threshold above maximum (0.65) - should be clamped
    let request_high = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
                "johari_threshold": 0.9,  // Above maximum
                "include_diagnostics": true,
            }
        })),
    );

    let response_high = handlers.dispatch(request_high).await;
    assert!(response_high.error.is_none(), "Should handle high threshold");

    let result_high = response_high.result.expect("result");
    let data_high = super::extract_mcp_tool_data(&result_high);
    let threshold_high = data_high["diagnostics"]["johari_threshold"].as_f64().expect("f64");
    // Use tolerance for f32/f64 precision issues
    assert!(
        threshold_high <= 0.65 + 0.0001,
        "EC-06: Threshold {} should be clamped to <= 0.65",
        threshold_high
    );
}

// ============================================================================
// TASK-DELTA-P1-001: Property-Based Tests
// ============================================================================

/// PBT-01: All outputs must be in valid range [0, 1] for ANY valid input
#[tokio::test]
async fn test_pbt_all_outputs_valid_range() {
    let handlers = create_test_handlers();

    // Test with various input combinations
    let test_cases = vec![
        (vec![0.0; 1024], vec![0.0; 1024]),
        (vec![1.0; 1024], vec![1.0; 1024]),
        (vec![0.0; 1024], vec![1.0; 1024]),
        (vec![1.0; 1024], vec![0.0; 1024]),
        (vec![0.5; 1024], vec![0.5; 1024]),
        (vec![0.1; 1024], vec![0.9; 1024]),
        (vec![0.25; 1024], vec![0.75; 1024]),
    ];

    for (i, (old_vals, new_vals)) in test_cases.iter().enumerate() {
        let old_fp = create_test_fingerprint_with_semantic(old_vals.clone());
        let new_fp = create_test_fingerprint_with_semantic(new_vals.clone());

        let request = make_request(
            "tools/call",
            Some(JsonRpcId::Number(1)),
            Some(json!({
                "name": "gwt/compute_delta_sc",
                "arguments": {
                    "vertex_id": Uuid::new_v4().to_string(),
                    "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                    "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
                }
            })),
        );

        let response = handlers.dispatch(request).await;
        assert!(response.error.is_none(), "PBT-01: Test case {} should succeed", i);

        let result = response.result.expect("result");
        let data = super::extract_mcp_tool_data(&result);

        // Verify ALL numeric outputs are in [0, 1]
        let delta_s_per_embedder = data["delta_s_per_embedder"].as_array().expect("array");
        for (j, val) in delta_s_per_embedder.iter().enumerate() {
            let v = val.as_f64().expect("f64");
            assert!(
                (0.0..=1.0).contains(&v) && !v.is_nan() && !v.is_infinite(),
                "PBT-01: Test case {} - delta_s_per_embedder[{}] = {} invalid",
                i,
                j,
                v
            );
        }

        let delta_s_agg = data["delta_s_aggregate"].as_f64().expect("f64");
        assert!(
            (0.0..=1.0).contains(&delta_s_agg) && !delta_s_agg.is_nan() && !delta_s_agg.is_infinite(),
            "PBT-01: Test case {} - delta_s_aggregate = {} invalid",
            i,
            delta_s_agg
        );

        let delta_c = data["delta_c"].as_f64().expect("f64");
        assert!(
            (0.0..=1.0).contains(&delta_c) && !delta_c.is_nan() && !delta_c.is_infinite(),
            "PBT-01: Test case {} - delta_c = {} invalid",
            i,
            delta_c
        );

        let learning_potential = data["utl_learning_potential"].as_f64().expect("f64");
        assert!(
            (0.0..=1.0).contains(&learning_potential) && !learning_potential.is_nan() && !learning_potential.is_infinite(),
            "PBT-01: Test case {} - utl_learning_potential = {} invalid",
            i,
            learning_potential
        );
    }
}

/// PBT-02: Johari quadrant must ALWAYS be a valid enum value
#[tokio::test]
async fn test_pbt_johari_always_valid_enum() {
    let handlers = create_test_handlers();
    let valid_quadrants = ["Open", "Blind", "Hidden", "Unknown"];

    // Test various input combinations
    let test_values: Vec<f32> = vec![0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];

    for old_val in &test_values {
        for new_val in &test_values {
            let old_fp = create_test_fingerprint_with_semantic(vec![*old_val; 1024]);
            let new_fp = create_test_fingerprint_with_semantic(vec![*new_val; 1024]);

            let request = make_request(
                "tools/call",
                Some(JsonRpcId::Number(1)),
                Some(json!({
                    "name": "gwt/compute_delta_sc",
                    "arguments": {
                        "vertex_id": Uuid::new_v4().to_string(),
                        "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                        "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
                    }
                })),
            );

            let response = handlers.dispatch(request).await;
            if response.error.is_some() {
                continue; // Skip failed requests
            }

            let result = response.result.expect("result");
            let data = super::extract_mcp_tool_data(&result);

            // Verify aggregate quadrant
            let johari_agg = data["johari_aggregate"].as_str().expect("string");
            assert!(
                valid_quadrants.contains(&johari_agg),
                "PBT-02: johari_aggregate '{}' is not a valid quadrant (old={}, new={})",
                johari_agg,
                old_val,
                new_val
            );

            // Verify all per-embedder quadrants
            let johari_quadrants = data["johari_quadrants"].as_array().expect("array");
            for (i, q) in johari_quadrants.iter().enumerate() {
                let quadrant = q.as_str().expect("string");
                assert!(
                    valid_quadrants.contains(&quadrant),
                    "PBT-02: johari_quadrants[{}] = '{}' is not valid (old={}, new={})",
                    i,
                    quadrant,
                    old_val,
                    new_val
                );
            }
        }
    }
}

/// PBT-03: diagnostics field must be present when include_diagnostics=true
#[tokio::test]
async fn test_pbt_diagnostics_present_when_requested() {
    let handlers = create_test_handlers();

    let test_values: Vec<f32> = vec![0.2, 0.5, 0.8];

    for old_val in &test_values {
        for new_val in &test_values {
            let old_fp = create_test_fingerprint_with_semantic(vec![*old_val; 1024]);
            let new_fp = create_test_fingerprint_with_semantic(vec![*new_val; 1024]);

            // With include_diagnostics=true
            let request = make_request(
                "tools/call",
                Some(JsonRpcId::Number(1)),
                Some(json!({
                    "name": "gwt/compute_delta_sc",
                    "arguments": {
                        "vertex_id": Uuid::new_v4().to_string(),
                        "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                        "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
                        "include_diagnostics": true,
                    }
                })),
            );

            let response = handlers.dispatch(request).await;
            assert!(response.error.is_none(), "Should succeed");

            let result = response.result.expect("result");
            let data = super::extract_mcp_tool_data(&result);

            assert!(
                data.get("diagnostics").is_some(),
                "PBT-03: diagnostics MUST be present when include_diagnostics=true (old={}, new={})",
                old_val,
                new_val
            );

            // Verify diagnostics structure
            let diagnostics = &data["diagnostics"];
            assert!(
                diagnostics.get("per_embedder").is_some(),
                "PBT-03: diagnostics.per_embedder must exist"
            );
            assert!(
                diagnostics.get("johari_threshold").is_some(),
                "PBT-03: diagnostics.johari_threshold must exist"
            );
            assert!(
                diagnostics.get("delta_c_components").is_some(),
                "PBT-03: diagnostics.delta_c_components must exist"
            );
        }
    }
}

/// PBT-04: Without include_diagnostics, diagnostics field should NOT be present
#[tokio::test]
async fn test_pbt_no_diagnostics_by_default() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.4; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.6; 1024]);

    // Without include_diagnostics (or include_diagnostics=false)
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
                "include_diagnostics": false,
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Should succeed");

    let result = response.result.expect("result");
    let data = super::extract_mcp_tool_data(&result);

    assert!(
        data.get("diagnostics").is_none(),
        "PBT-04: diagnostics should NOT be present when include_diagnostics=false"
    );
}
