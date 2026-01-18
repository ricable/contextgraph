//! Full State Verification Tests for Search Handlers
//!
//! Complete search workflow tests using STUB implementations for isolated unit testing.

use serde_json::json;

use crate::protocol::JsonRpcId;
use context_graph_core::types::fingerprint::NUM_EMBEDDERS;

use crate::handlers::tests::{create_test_handlers, make_request};

/// FULL STATE VERIFICATION: Complete search workflow test.
///
/// Tests the full search lifecycle with real data:
/// 1. Store multiple fingerprints with different content
/// 2. Search via multi with preset and custom weights
/// 3. Search via single_space targeting specific embedding
/// 4. Search via by_purpose using purpose vector
/// 5. Verify search/weight_profiles returns all 13-weight profiles
///
/// Uses STUB implementations for isolated unit testing.
#[tokio::test]
#[ignore = "Uses memory/store, search/multi, search/single_space, search/by_purpose, search/weight_profiles APIs removed in PRD v6 - use tools/call"]
async fn test_full_state_verification_search_workflow() {
    let handlers = create_test_handlers();

    // =========================================================================
    // STEP 1: Store multiple fingerprints with different content
    // =========================================================================
    let contents = [
        "Machine learning algorithms for predictive analytics",
        "Rust programming language memory safety guarantees",
        "Knowledge graph construction with entity linking",
        "Temporal reasoning for event sequence modeling",
        "Sparse retrieval using inverted indices",
    ];

    let mut stored_ids: Vec<String> = Vec::new();
    for (i, content) in contents.iter().enumerate() {
        let store_params = json!({
            "content": content,
            "importance": 0.5 + (i as f64 * 0.1)
        });
        let store_request = make_request(
            "memory/store",
            Some(JsonRpcId::Number(i as i64 + 1)),
            Some(store_params),
        );
        let response = handlers.dispatch(store_request).await;
        assert!(response.error.is_none(), "Store {} must succeed", i);

        let fingerprint_id = response
            .result
            .unwrap()
            .get("fingerprintId")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        stored_ids.push(fingerprint_id);
    }

    assert_eq!(stored_ids.len(), 5, "Must have stored 5 fingerprints");

    // =========================================================================
    // STEP 2: Search via multi with preset profile
    // =========================================================================
    let multi_preset_params = json!({
        "query": "machine learning data science",
        "query_type": "semantic_search",
        "topK": 10,
        "minSimilarity": 0.0,  // P1-FIX-1: Required parameter for fail-fast
        "include_per_embedder_scores": true
    });
    let multi_preset_request = make_request(
        "search/multi",
        Some(JsonRpcId::Number(10)),
        Some(multi_preset_params),
    );
    let multi_preset_response = handlers.dispatch(multi_preset_request).await;

    assert!(
        multi_preset_response.error.is_none(),
        "search/multi with preset must succeed"
    );
    let multi_preset_result = multi_preset_response.result.expect("Must have result");

    // Verify results found
    let results = multi_preset_result
        .get("results")
        .and_then(|v| v.as_array())
        .expect("Must have results array");
    assert!(
        !results.is_empty(),
        "search/multi must find at least one result"
    );

    // Verify per-embedder scores present (13 scores)
    if !results.is_empty() {
        let per_scores = results[0]
            .get("per_embedder_scores")
            .and_then(|v| v.as_object());
        assert!(per_scores.is_some(), "Must have per_embedder_scores");
        // The scores use space_json_key which may have different naming
        let top_contrib = results[0]
            .get("top_contributing_spaces")
            .and_then(|v| v.as_array());
        assert!(top_contrib.is_some(), "Must have top_contributing_spaces");
    }

    // =========================================================================
    // STEP 3: Search via multi with custom 13-element weights
    // =========================================================================
    let custom_weights: Vec<f64> = vec![
        0.20, 0.05, 0.05, 0.05, 0.05, 0.05, 0.20, 0.10, 0.10, 0.05, 0.05, 0.03, 0.02,
    ];
    let multi_custom_params = json!({
        "query": "programming language memory",
        "query_type": "custom",
        "weights": custom_weights,
        "topK": 5,
        "minSimilarity": 0.0  // P1-FIX-1: Required parameter for fail-fast
    });
    let multi_custom_request = make_request(
        "search/multi",
        Some(JsonRpcId::Number(11)),
        Some(multi_custom_params),
    );
    let multi_custom_response = handlers.dispatch(multi_custom_request).await;

    assert!(
        multi_custom_response.error.is_none(),
        "search/multi with custom weights must succeed"
    );
    let multi_custom_result = multi_custom_response.result.expect("Must have result");
    let custom_metadata = multi_custom_result
        .get("query_metadata")
        .expect("Must have metadata");
    let applied_weights = custom_metadata
        .get("weights_applied")
        .and_then(|v| v.as_array())
        .expect("Must have weights_applied");
    assert_eq!(applied_weights.len(), 13, "Must apply exactly 13 weights");

    // =========================================================================
    // STEP 4: Search via single_space targeting code embedding (index 6)
    // =========================================================================
    let single_space_params = json!({
        "query": "rust programming",
        "space_index": 6,
        "topK": 5,
        "minSimilarity": 0.0  // P1-FIX-1: Required parameter for fail-fast
    });
    let single_space_request = make_request(
        "search/single_space",
        Some(JsonRpcId::Number(12)),
        Some(single_space_params),
    );
    let single_space_response = handlers.dispatch(single_space_request).await;

    assert!(
        single_space_response.error.is_none(),
        "search/single_space must succeed"
    );
    let single_space_result = single_space_response.result.expect("Must have result");
    assert_eq!(
        single_space_result
            .get("space_index")
            .and_then(|v| v.as_u64()),
        Some(6),
        "Must target space index 6"
    );
    assert_eq!(
        single_space_result
            .get("space_name")
            .and_then(|v| v.as_str()),
        Some("E7_Code"),
        "Space 6 must be named E7_Code"
    );

    // =========================================================================
    // STEP 5: Search via by_purpose using purpose vector
    // =========================================================================
    let purpose_vec: Vec<f64> = vec![
        0.3, 0.1, 0.1, 0.0, 0.1, 0.0, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0,
    ];
    let by_purpose_params = json!({
        "purpose_vector": purpose_vec,
        "topK": 5
    });
    let by_purpose_request = make_request(
        "search/by_purpose",
        Some(JsonRpcId::Number(13)),
        Some(by_purpose_params),
    );
    let by_purpose_response = handlers.dispatch(by_purpose_request).await;

    assert!(
        by_purpose_response.error.is_none(),
        "search/by_purpose must succeed"
    );
    let by_purpose_result = by_purpose_response.result.expect("Must have result");
    assert!(
        by_purpose_result.get("results").is_some(),
        "Must have results array"
    );
    assert!(
        by_purpose_result.get("search_time_ms").is_some(),
        "Must report search_time_ms"
    );

    // =========================================================================
    // STEP 6: Verify search/weight_profiles returns all 13-weight profiles
    // =========================================================================
    let profiles_request =
        make_request("search/weight_profiles", Some(JsonRpcId::Number(14)), None);
    let profiles_response = handlers.dispatch(profiles_request).await;

    assert!(
        profiles_response.error.is_none(),
        "search/weight_profiles must succeed"
    );
    let profiles_result = profiles_response.result.expect("Must have result");

    let profiles = profiles_result
        .get("profiles")
        .and_then(|v| v.as_array())
        .expect("Must have profiles array");
    assert!(!profiles.is_empty(), "Must have at least one profile");

    // Verify each profile has 13 weights
    for profile in profiles {
        let weights = profile.get("weights").and_then(|v| v.as_array());
        assert!(weights.is_some(), "Profile must have weights");
        assert_eq!(
            weights.unwrap().len(),
            NUM_EMBEDDERS,
            "Profile must have exactly 13 weights"
        );
    }

    let total_spaces = profiles_result.get("total_spaces").and_then(|v| v.as_u64());
    assert_eq!(total_spaces, Some(13), "total_spaces must be 13");

    // =========================================================================
    // VERIFICATION COMPLETE: All search operations work with real data
    // =========================================================================
}

/// Test error codes are correctly returned for search failures.
#[tokio::test]
#[ignore = "Uses search/multi, search/single_space, search/by_purpose APIs removed in PRD v6 - use tools/call"]
async fn test_full_state_verification_search_error_codes() {
    let handlers = create_test_handlers();

    // Test INVALID_PARAMS for missing query in search/multi
    let missing_query = make_request(
        "search/multi",
        Some(JsonRpcId::Number(1)),
        Some(json!({ "query_type": "semantic_search" })),
    );
    let resp = handlers.dispatch(missing_query).await;
    assert_eq!(
        resp.error.unwrap().code,
        -32602,
        "Missing query must return INVALID_PARAMS"
    );

    // Test INVALID_PARAMS for invalid space_index (13, only 0-12 valid)
    let invalid_space = make_request(
        "search/single_space",
        Some(JsonRpcId::Number(2)),
        Some(json!({ "query": "test", "space_index": 13 })),
    );
    let resp = handlers.dispatch(invalid_space).await;
    assert_eq!(
        resp.error.unwrap().code,
        -32602,
        "Invalid space_index must return INVALID_PARAMS"
    );

    // Test INVALID_PARAMS for 12-element weights (must be 13)
    let twelve_weights: Vec<f64> = vec![0.083; 12]; // Wrong: 12 instead of 13
    let wrong_weights = make_request(
        "search/multi",
        Some(JsonRpcId::Number(3)),
        Some(json!({
            "query": "test",
            "query_type": "custom",
            "weights": twelve_weights
        })),
    );
    let resp = handlers.dispatch(wrong_weights).await;
    assert_eq!(
        resp.error.unwrap().code,
        -32602,
        "12-element weights must return INVALID_PARAMS"
    );

    // Test INVALID_PARAMS for 12-element purpose_vector (must be 13)
    let twelve_purpose: Vec<f64> = vec![0.083; 12]; // Wrong: 12 instead of 13
    let wrong_purpose = make_request(
        "search/by_purpose",
        Some(JsonRpcId::Number(4)),
        Some(json!({ "purpose_vector": twelve_purpose })),
    );
    let resp = handlers.dispatch(wrong_purpose).await;
    assert_eq!(
        resp.error.unwrap().code,
        -32602,
        "12-element purpose_vector must return INVALID_PARAMS"
    );
}
