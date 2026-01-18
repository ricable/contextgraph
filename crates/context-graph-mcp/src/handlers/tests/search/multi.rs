//! Tests for search/multi handler.
//!
//! Tests multi-space search with preset and custom 13-element weights.

use serde_json::json;

use crate::protocol::{error_codes, JsonRpcId};
use context_graph_core::types::fingerprint::NUM_EMBEDDERS;

use crate::handlers::tests::{create_test_handlers, make_request};

/// Test search/multi with semantic_search preset.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_multi_semantic_preset() {
    let handlers = create_test_handlers();

    // First store some content to search
    let store_params = json!({
        "content": "Machine learning enables computers to learn from data",
        "importance": 0.9
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    handlers.dispatch(store_request).await;

    // Now search with semantic_search preset
    let search_params = json!({
        "query": "AI and data science",
        "query_type": "semantic_search",
        "topK": 10,
        "minSimilarity": 0.0  // P1-FIX-1: Required parameter for fail-fast
    });
    let search_request = make_request(
        "search/multi",
        Some(JsonRpcId::Number(2)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(response.error.is_none(), "search/multi should succeed");
    let result = response.result.expect("Should have result");

    // Verify results structure
    assert!(result.get("results").is_some(), "Should have results array");
    assert!(result.get("count").is_some(), "Should have count");
    assert!(
        result.get("query_metadata").is_some(),
        "Should have query_metadata"
    );

    // Verify query_metadata structure
    let metadata = result.get("query_metadata").unwrap();
    assert_eq!(
        metadata.get("query_type_used").and_then(|v| v.as_str()),
        Some("semantic_search"),
        "Should use semantic_search preset"
    );
    assert!(
        metadata.get("weights_applied").is_some(),
        "Should show weights applied"
    );
    assert!(
        metadata.get("aggregation_strategy").is_some(),
        "Should show aggregation strategy"
    );
    assert!(
        metadata.get("search_time_ms").is_some(),
        "Should report search time"
    );
}

/// Test search/multi with custom 13-element weight array.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_multi_custom_weights_13_spaces() {
    let handlers = create_test_handlers();

    // Store content
    let store_params = json!({
        "content": "Graph neural networks for knowledge representation",
        "importance": 0.8
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    handlers.dispatch(store_request).await;

    // Custom weights for 13 spaces (must sum to 1.0)
    let custom_weights: Vec<f64> = vec![
        0.15, // E1 Semantic
        0.05, // E2 Temporal-Recent
        0.05, // E3 Temporal-Periodic
        0.05, // E4 Temporal-Positional
        0.10, // E5 Causal
        0.05, // E6 Sparse
        0.15, // E7 Code
        0.10, // E8 Graph
        0.10, // E9 HDC
        0.05, // E10 Multimodal
        0.05, // E11 Entity
        0.05, // E12 Late Interaction
        0.05, // E13 SPLADE
    ];

    let search_params = json!({
        "query": "knowledge graphs",
        "query_type": "custom",
        "weights": custom_weights,
        "topK": 5,
        "minSimilarity": 0.0,  // P1-FIX-1: Required parameter for fail-fast
        "include_per_embedder_scores": true
    });
    let search_request = make_request(
        "search/multi",
        Some(JsonRpcId::Number(2)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(
        response.error.is_none(),
        "search/multi with custom weights should succeed"
    );
    let result = response.result.expect("Should have result");

    // Verify custom weights were applied
    let metadata = result.get("query_metadata").unwrap();
    let weights_applied = metadata.get("weights_applied").and_then(|v| v.as_array());
    assert!(
        weights_applied.is_some(),
        "Should have weights_applied array"
    );
    assert_eq!(
        weights_applied.unwrap().len(),
        NUM_EMBEDDERS,
        "Must apply exactly 13 weights"
    );
}

/// Test search/multi fails with 12-element weight array (must be 13).
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_multi_invalid_weights_12_spaces_fails() {
    let handlers = create_test_handlers();

    // Only 12 weights (WRONG - must be 13!)
    let invalid_weights: Vec<f64> = vec![
        0.10, 0.08, 0.08, 0.08, 0.10, 0.08, 0.10, 0.10, 0.08, 0.08, 0.08, 0.04,
    ];

    let search_params = json!({
        "query": "test query",
        "query_type": "custom",
        "weights": invalid_weights
    });
    let search_request = make_request(
        "search/multi",
        Some(JsonRpcId::Number(1)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(
        response.error.is_some(),
        "search/multi must fail with 12 weights"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32602,
        "Should return INVALID_PARAMS error code"
    );
    assert!(
        error.message.contains("13") || error.message.contains("weight"),
        "Error should mention weight count issue"
    );
}

/// Test search/multi fails with missing query parameter.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_multi_missing_query_fails() {
    let handlers = create_test_handlers();

    let search_params = json!({
        "query_type": "semantic_search",
        "topK": 10
    });
    let search_request = make_request(
        "search/multi",
        Some(JsonRpcId::Number(1)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(
        response.error.is_some(),
        "search/multi must fail without query"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32602,
        "Should return INVALID_PARAMS error code"
    );
    assert!(
        error.message.contains("query"),
        "Error should mention missing query"
    );
}

/// Test search/multi fails with unknown query_type.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_multi_unknown_query_type_fails() {
    let handlers = create_test_handlers();

    let search_params = json!({
        "query": "test query",
        "query_type": "nonexistent_type"
    });
    let search_request = make_request(
        "search/multi",
        Some(JsonRpcId::Number(1)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(
        response.error.is_some(),
        "search/multi must fail with unknown query_type"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32602,
        "Should return INVALID_PARAMS error code"
    );
    assert!(
        error.message.contains("query_type") || error.message.contains("Available"),
        "Error should mention valid query types"
    );
}

/// Test search/multi with active_spaces array.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_multi_active_spaces_array() {
    let handlers = create_test_handlers();

    // Store content
    let store_params = json!({
        "content": "Rust programming language with memory safety",
        "importance": 0.8
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    handlers.dispatch(store_request).await;

    // Only search in spaces 0, 6, 7 (Semantic, Code, Graph)
    let search_params = json!({
        "query": "programming languages",
        "query_type": "code_search",
        "active_spaces": [0, 6, 7],
        "topK": 5,
        "minSimilarity": 0.0  // P1-FIX-1: Required parameter for fail-fast
    });
    let search_request = make_request(
        "search/multi",
        Some(JsonRpcId::Number(2)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(
        response.error.is_none(),
        "search/multi with active_spaces should succeed"
    );
    let result = response.result.expect("Should have result");

    let metadata = result.get("query_metadata").unwrap();
    assert_eq!(
        metadata.get("spaces_searched").and_then(|v| v.as_u64()),
        Some(3),
        "Should search only 3 spaces"
    );
}

/// Test search/multi with invalid aggregation strategy fails.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_multi_invalid_aggregation_fails() {
    let handlers = create_test_handlers();

    let search_params = json!({
        "query": "test query",
        "aggregation": "invalid_strategy"
    });
    let search_request = make_request(
        "search/multi",
        Some(JsonRpcId::Number(1)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(
        response.error.is_some(),
        "search/multi must fail with invalid aggregation"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32602,
        "Should return INVALID_PARAMS error code"
    );
}

/// Test search/multi with include_pipeline_breakdown=true returns PIPELINE_METRICS_UNAVAILABLE.
///
/// TASK-EMB-024: Pipeline breakdown metrics are NOT YET IMPLEMENTED.
/// NO hardcoded fallback values - must fail fast until real metrics available.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_multi_pipeline_breakdown() {
    let handlers = create_test_handlers();

    // Store content first
    let store_params = json!({
        "content": "Neural network training techniques",
        "importance": 0.9
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    handlers.dispatch(store_request).await;

    let search_params = json!({
        "query": "deep learning",
        "include_pipeline_breakdown": true,
        "minSimilarity": 0.0  // P1-FIX-1: Required parameter (test expects pipeline breakdown error)
    });
    let search_request = make_request(
        "search/multi",
        Some(JsonRpcId::Number(2)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    // VERIFY FAIL-FAST BEHAVIOR
    // TASK-EMB-024: Pipeline breakdown is NOT YET IMPLEMENTED
    assert!(
        response.error.is_some(),
        "search/multi with include_pipeline_breakdown=true MUST return error (TASK-EMB-024)"
    );

    let error = response.error.as_ref().unwrap();
    assert_eq!(
        error.code,
        error_codes::PIPELINE_METRICS_UNAVAILABLE,
        "Should return PIPELINE_METRICS_UNAVAILABLE (-32052)"
    );
    assert!(
        error.message.contains("Pipeline breakdown")
            || error.message.contains("not yet implemented"),
        "Error message should indicate feature is not implemented"
    );
}
