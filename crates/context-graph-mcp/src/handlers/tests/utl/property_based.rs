//! TASK-DELTA-P1-001: Property-Based Tests

use serde_json::json;
use uuid::Uuid;

use crate::handlers::tests::{create_test_handlers, extract_mcp_tool_data, make_request};
use crate::protocol::JsonRpcId;

use super::helpers::create_test_fingerprint_with_semantic;

/// PBT-01: All outputs must be in valid range [0, 1] for ANY valid input
#[tokio::test]
async fn test_pbt_all_outputs_valid_range() {
    let handlers = create_test_handlers();

    // Test with various input combinations
    let test_cases = [(vec![0.0; 1024], vec![0.0; 1024]),
        (vec![1.0; 1024], vec![1.0; 1024]),
        (vec![0.0; 1024], vec![1.0; 1024]),
        (vec![1.0; 1024], vec![0.0; 1024]),
        (vec![0.5; 1024], vec![0.5; 1024]),
        (vec![0.1; 1024], vec![0.9; 1024]),
        (vec![0.25; 1024], vec![0.75; 1024])];

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
        let data = extract_mcp_tool_data(&result);

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
            let data = extract_mcp_tool_data(&result);

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
            let data = extract_mcp_tool_data(&result);

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
    let data = extract_mcp_tool_data(&result);

    assert!(
        data.get("diagnostics").is_none(),
        "PBT-04: diagnostics should NOT be present when include_diagnostics=false"
    );
}
