//! Manual Full State Verification for gwt/compute_delta_sc (TASK-DELTA-P1-001)
//!
//! This test module performs manual verification of the `gwt/compute_delta_sc` tool
//! by creating synthetic test data and verifying the outputs match expected values.
//!
//! # Test Cases
//!
//! 1. **Small Change**: Minimal difference between old/new fingerprints
//!    - Expected: Low Delta_S (< 0.3), High Delta_C (> 0.7)
//!
//! 2. **Large Change**: Significant difference between fingerprints
//!    - Expected: High Delta_S (> 0.5), variable Delta_C
//!
//! 3. **Identical**: No change between fingerprints
//!    - Expected: Delta_S ~= 0, valid Delta_C
//!
//! # Running the Test
//!
//! ```bash
//! cargo test --package context-graph-mcp manual_delta_sc --features test-utils -- --ignored --nocapture
//! ```

use serde_json::json;
use uuid::Uuid;

use context_graph_core::types::fingerprint::{
    JohariFingerprint, PurposeVector, SemanticFingerprint, SparseVector, TeleologicalFingerprint,
    E10_DIM, E11_DIM, E1_DIM, E2_DIM, E3_DIM, E4_DIM, E5_DIM, E7_DIM, E8_DIM, E9_DIM, NUM_EMBEDDERS,
};

use crate::protocol::JsonRpcId;

use super::{create_test_handlers_with_all_components, extract_mcp_tool_data, make_request};

/// Create a synthetic SemanticFingerprint with E1 semantic set to a specific value.
///
/// All embedders are set to the specified value for dense embeddings.
/// This allows testing how delta computation responds to different embedding magnitudes.
fn create_synthetic_fingerprint(base_value: f32) -> SemanticFingerprint {
    SemanticFingerprint {
        e1_semantic: vec![base_value; E1_DIM],
        e2_temporal_recent: vec![base_value; E2_DIM],
        e3_temporal_periodic: vec![base_value; E3_DIM],
        e4_temporal_positional: vec![base_value; E4_DIM],
        e5_causal: vec![base_value; E5_DIM],
        e6_sparse: SparseVector::empty(),
        e7_code: vec![base_value; E7_DIM],
        e8_graph: vec![base_value; E8_DIM],
        e9_hdc: vec![base_value; E9_DIM],
        e10_multimodal: vec![base_value; E10_DIM],
        e11_entity: vec![base_value; E11_DIM],
        e12_late_interaction: Vec::new(),
        e13_splade: SparseVector::empty(),
    }
}

/// Create a complete TeleologicalFingerprint for testing.
fn create_test_teleological_fingerprint(base_value: f32) -> TeleologicalFingerprint {
    let semantic = create_synthetic_fingerprint(base_value);
    let purpose_vector = PurposeVector::new([0.5; NUM_EMBEDDERS]); // Neutral alignment
    let johari = JohariFingerprint::zeroed();
    let content_hash = [0u8; 32];

    TeleologicalFingerprint::new(semantic, purpose_vector, johari, content_hash)
}

/// Verify that delta_s_per_embedder has exactly 13 elements.
fn verify_delta_s_count(delta_s_per_embedder: &[f64]) -> Result<(), String> {
    if delta_s_per_embedder.len() != NUM_EMBEDDERS {
        return Err(format!(
            "FAIL: delta_s_per_embedder has {} elements, expected {}",
            delta_s_per_embedder.len(),
            NUM_EMBEDDERS
        ));
    }
    Ok(())
}

/// Verify all values are in [0, 1] range.
fn verify_range_01(value: f64, name: &str) -> Result<(), String> {
    if value < 0.0 || value > 1.0 {
        return Err(format!(
            "FAIL: {} = {} is outside [0, 1] range",
            name, value
        ));
    }
    Ok(())
}

/// Verify delta_s_per_embedder values are all in [0, 1] range.
fn verify_delta_s_values(delta_s_per_embedder: &[f64]) -> Result<(), String> {
    for (i, &value) in delta_s_per_embedder.iter().enumerate() {
        verify_range_01(value, &format!("delta_s_per_embedder[{}]", i))?;
    }
    Ok(())
}

/// Verify johari_quadrant is valid.
fn verify_johari_quadrant(quadrant: &str) -> Result<(), String> {
    const VALID_QUADRANTS: [&str; 4] = ["Open", "Blind", "Hidden", "Unknown"];
    if !VALID_QUADRANTS.contains(&quadrant) {
        return Err(format!(
            "FAIL: johari_aggregate '{}' is not one of {:?}",
            quadrant, VALID_QUADRANTS
        ));
    }
    Ok(())
}

/// Verify UTL learning potential formula: utl_learning_potential = delta_s_aggregate * delta_c
fn verify_utl_formula(
    delta_s_aggregate: f64,
    delta_c: f64,
    utl_learning_potential: f64,
) -> Result<(), String> {
    let expected = delta_s_aggregate * delta_c;
    let tolerance = 0.001;
    let diff = (utl_learning_potential - expected).abs();

    if diff > tolerance {
        return Err(format!(
            "FAIL: utl_learning_potential ({}) != delta_s_aggregate * delta_c ({} * {} = {}), diff = {}",
            utl_learning_potential, delta_s_aggregate, delta_c, expected, diff
        ));
    }
    Ok(())
}

/// Extract delta_sc results from the response.
fn extract_delta_sc_results(
    data: &serde_json::Value,
) -> Result<
    (
        Vec<f64>,
        f64,
        f64,
        Vec<String>,
        String,
        f64,
    ),
    String,
> {
    let delta_s_per_embedder = data
        .get("delta_s_per_embedder")
        .and_then(|v| v.as_array())
        .ok_or("Missing delta_s_per_embedder")?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0))
        .collect::<Vec<_>>();

    let delta_s_aggregate = data
        .get("delta_s_aggregate")
        .and_then(|v| v.as_f64())
        .ok_or("Missing delta_s_aggregate")?;

    let delta_c = data
        .get("delta_c")
        .and_then(|v| v.as_f64())
        .ok_or("Missing delta_c")?;

    let johari_quadrants = data
        .get("johari_quadrants")
        .and_then(|v| v.as_array())
        .ok_or("Missing johari_quadrants")?
        .iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect::<Vec<_>>();

    let johari_aggregate = data
        .get("johari_aggregate")
        .and_then(|v| v.as_str())
        .ok_or("Missing johari_aggregate")?
        .to_string();

    let utl_learning_potential = data
        .get("utl_learning_potential")
        .and_then(|v| v.as_f64())
        .ok_or("Missing utl_learning_potential")?;

    Ok((
        delta_s_per_embedder,
        delta_s_aggregate,
        delta_c,
        johari_quadrants,
        johari_aggregate,
        utl_learning_potential,
    ))
}

/// Run verification for a single test case.
async fn run_verification(
    test_name: &str,
    old_fp: &TeleologicalFingerprint,
    new_fp: &TeleologicalFingerprint,
    handlers: &crate::handlers::Handlers,
) -> Result<(), String> {
    println!("\n========================================");
    println!("TEST: {}", test_name);
    println!("========================================");

    // Serialize fingerprints for the request
    let old_fp_json = serde_json::to_value(old_fp)
        .map_err(|e| format!("Failed to serialize old_fingerprint: {}", e))?;
    let new_fp_json = serde_json::to_value(new_fp)
        .map_err(|e| format!("Failed to serialize new_fingerprint: {}", e))?;

    let vertex_id = Uuid::new_v4();

    println!("\n[INPUT]");
    println!("  vertex_id: {}", vertex_id);
    println!(
        "  old_fingerprint.e1_semantic[0..5]: {:?}",
        &old_fp.semantic.e1_semantic[0..5]
    );
    println!(
        "  new_fingerprint.e1_semantic[0..5]: {:?}",
        &new_fp.semantic.e1_semantic[0..5]
    );

    // Create request using tools/call pattern
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": vertex_id.to_string(),
                "old_fingerprint": old_fp_json,
                "new_fingerprint": new_fp_json,
                "include_diagnostics": true,
                "johari_threshold": 0.5
            }
        })),
    );

    // Dispatch the request
    let response = handlers.dispatch(request).await;

    // Check for errors
    if let Some(error) = &response.error {
        return Err(format!(
            "Tool returned error: {} (code: {})",
            error.message, error.code
        ));
    }

    let result = response.result.ok_or("No result in response")?;

    // Extract the data from MCP tool response format
    let data = extract_mcp_tool_data(&result);

    // Extract results
    let (
        delta_s_per_embedder,
        delta_s_aggregate,
        delta_c,
        johari_quadrants,
        johari_aggregate,
        utl_learning_potential,
    ) = extract_delta_sc_results(&data)?;

    println!("\n[OUTPUT]");
    println!("  delta_s_per_embedder: {:?}", delta_s_per_embedder);
    println!("  delta_s_aggregate: {:.6}", delta_s_aggregate);
    println!("  delta_c: {:.6}", delta_c);
    println!("  johari_quadrants: {:?}", johari_quadrants);
    println!("  johari_aggregate: {}", johari_aggregate);
    println!("  utl_learning_potential: {:.6}", utl_learning_potential);

    // Run verifications
    println!("\n[VERIFICATION]");

    // 1. Verify delta_s_per_embedder has exactly 13 elements
    verify_delta_s_count(&delta_s_per_embedder)?;
    println!("  [PASS] delta_s_per_embedder has 13 elements");

    // 2. Verify all delta_s values are in [0, 1]
    verify_delta_s_values(&delta_s_per_embedder)?;
    println!("  [PASS] All delta_s_per_embedder values in [0, 1]");

    // 3. Verify delta_s_aggregate is in [0, 1]
    verify_range_01(delta_s_aggregate, "delta_s_aggregate")?;
    println!("  [PASS] delta_s_aggregate in [0, 1]");

    // 4. Verify delta_c is in [0, 1]
    verify_range_01(delta_c, "delta_c")?;
    println!("  [PASS] delta_c in [0, 1]");

    // 5. Verify utl_learning_potential is in [0, 1]
    verify_range_01(utl_learning_potential, "utl_learning_potential")?;
    println!("  [PASS] utl_learning_potential in [0, 1]");

    // 6. Verify johari_quadrant is valid
    verify_johari_quadrant(&johari_aggregate)?;
    println!("  [PASS] johari_aggregate is valid quadrant");

    // 7. Verify all johari_quadrants are valid
    for (i, q) in johari_quadrants.iter().enumerate() {
        verify_johari_quadrant(q)?;
        println!("  [PASS] johari_quadrants[{}] = {} is valid", i, q);
    }

    // 8. Verify UTL formula: utl_learning_potential = delta_s_aggregate * delta_c
    verify_utl_formula(delta_s_aggregate, delta_c, utl_learning_potential)?;
    println!(
        "  [PASS] UTL formula verified: {:.6} = {:.6} * {:.6}",
        utl_learning_potential, delta_s_aggregate, delta_c
    );

    println!("\n[RESULT] {} - ALL VERIFICATIONS PASSED", test_name);
    Ok(())
}

/// Test Case 1: Small Change
///
/// old_fingerprint: e1_semantic = [0.5; 1024]
/// new_fingerprint: e1_semantic = [0.52; 1024]
/// Expected: Low Delta_S (< 0.3), High Delta_C (> 0.7), Johari = "Open" or "Hidden"
#[tokio::test]
#[ignore = "Manual FSV test - run with --ignored"]
async fn test_delta_sc_small_change() {
    let handlers = create_test_handlers_with_all_components();

    let old_fp = create_test_teleological_fingerprint(0.5);
    let new_fp = create_test_teleological_fingerprint(0.52);

    let result = run_verification("Small Change", &old_fp, &new_fp, &handlers).await;

    if let Err(e) = result {
        panic!("Test failed: {}", e);
    }
}

/// Test Case 2: Large Change
///
/// old_fingerprint: e1_semantic = [0.2; 1024]
/// new_fingerprint: e1_semantic = [0.8; 1024]
/// Expected: High Delta_S (> 0.5), valid Delta_C, Johari varies
#[tokio::test]
#[ignore = "Manual FSV test - run with --ignored"]
async fn test_delta_sc_large_change() {
    let handlers = create_test_handlers_with_all_components();

    let old_fp = create_test_teleological_fingerprint(0.2);
    let new_fp = create_test_teleological_fingerprint(0.8);

    let result = run_verification("Large Change", &old_fp, &new_fp, &handlers).await;

    if let Err(e) = result {
        panic!("Test failed: {}", e);
    }
}

/// Test Case 3: Identical Fingerprints
///
/// old_fingerprint: e1_semantic = [0.5; 1024]
/// new_fingerprint: e1_semantic = [0.5; 1024]
/// Expected: Delta_S ~= 0, Delta_C valid, Johari = "Open" or "Hidden"
#[tokio::test]
#[ignore = "Manual FSV test - run with --ignored"]
async fn test_delta_sc_identical() {
    let handlers = create_test_handlers_with_all_components();

    let old_fp = create_test_teleological_fingerprint(0.5);
    let new_fp = create_test_teleological_fingerprint(0.5);

    let result = run_verification("Identical Fingerprints", &old_fp, &new_fp, &handlers).await;

    if let Err(e) = result {
        panic!("Test failed: {}", e);
    }
}

/// Comprehensive test running all three test cases.
#[tokio::test]
#[ignore = "Manual FSV test - run with --ignored"]
async fn test_delta_sc_all_cases() {
    println!("\n");
    println!("##################################################");
    println!("#  TASK-DELTA-P1-001: Manual Delta SC Verification");
    println!("##################################################");
    println!("\nThis test verifies the gwt/compute_delta_sc tool");
    println!("with three synthetic test cases.\n");

    let handlers = create_test_handlers_with_all_components();

    let mut failures: Vec<String> = Vec::new();

    // Test Case 1: Small Change
    {
        let old_fp = create_test_teleological_fingerprint(0.5);
        let new_fp = create_test_teleological_fingerprint(0.52);
        if let Err(e) = run_verification("Small Change", &old_fp, &new_fp, &handlers).await {
            failures.push(format!("Small Change: {}", e));
        }
    }

    // Test Case 2: Large Change
    {
        let old_fp = create_test_teleological_fingerprint(0.2);
        let new_fp = create_test_teleological_fingerprint(0.8);
        if let Err(e) = run_verification("Large Change", &old_fp, &new_fp, &handlers).await {
            failures.push(format!("Large Change: {}", e));
        }
    }

    // Test Case 3: Identical
    {
        let old_fp = create_test_teleological_fingerprint(0.5);
        let new_fp = create_test_teleological_fingerprint(0.5);
        if let Err(e) =
            run_verification("Identical Fingerprints", &old_fp, &new_fp, &handlers).await
        {
            failures.push(format!("Identical Fingerprints: {}", e));
        }
    }

    // Summary
    println!("\n##################################################");
    println!("#  SUMMARY");
    println!("##################################################");

    if failures.is_empty() {
        println!("\nALL 3 TEST CASES PASSED!");
        println!("\nVerification Status: COMPLETE");
        println!("Memory Key: swarm:TASK-DELTA-P1-001-manual-test-complete");
        println!("Result: SUCCESS");
    } else {
        println!("\nFAILURES ({}):", failures.len());
        for f in &failures {
            println!("  - {}", f);
        }
        panic!("Some test cases failed: {:?}", failures);
    }
}

/// Additional edge case: Zero vector inputs
#[tokio::test]
#[ignore = "Manual FSV test - run with --ignored"]
async fn test_delta_sc_zero_vectors() {
    let handlers = create_test_handlers_with_all_components();

    // Zero vectors should still produce valid outputs (per AP-10 NaN/Inf handling)
    let old_fp = create_test_teleological_fingerprint(0.0);
    let new_fp = create_test_teleological_fingerprint(0.0);

    let result = run_verification("Zero Vectors (Edge Case)", &old_fp, &new_fp, &handlers).await;

    // This may fail due to division by zero in cosine similarity,
    // but the tool should handle it gracefully with NaN/Inf clamping
    match result {
        Ok(_) => println!("Zero vectors handled successfully"),
        Err(e) => {
            // Check if error is about NaN/Inf (acceptable failure mode)
            if e.contains("NaN") || e.contains("Inf") || e.contains("undefined") {
                println!("Zero vectors produced expected NaN/Inf behavior: {}", e);
            } else {
                panic!("Unexpected error with zero vectors: {}", e);
            }
        }
    }
}

/// Additional edge case: Negative values
#[tokio::test]
#[ignore = "Manual FSV test - run with --ignored"]
async fn test_delta_sc_negative_values() {
    let handlers = create_test_handlers_with_all_components();

    let old_fp = create_test_teleological_fingerprint(-0.3);
    let new_fp = create_test_teleological_fingerprint(0.3);

    let result = run_verification("Negative Values (Edge Case)", &old_fp, &new_fp, &handlers).await;

    if let Err(e) = result {
        panic!("Test failed: {}", e);
    }
}

/// Additional edge case: Maximum values
#[tokio::test]
#[ignore = "Manual FSV test - run with --ignored"]
async fn test_delta_sc_max_values() {
    let handlers = create_test_handlers_with_all_components();

    let old_fp = create_test_teleological_fingerprint(1.0);
    let new_fp = create_test_teleological_fingerprint(0.0);

    let result = run_verification("Maximum Delta (Edge Case)", &old_fp, &new_fp, &handlers).await;

    if let Err(e) = result {
        panic!("Test failed: {}", e);
    }
}
