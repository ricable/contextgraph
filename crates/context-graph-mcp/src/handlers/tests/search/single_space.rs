//! Tests for search/single_space handler.
//!
//! Tests single-space search for specific embedding spaces (0-12).

use serde_json::json;

use crate::protocol::JsonRpcId;

use crate::handlers::tests::{create_test_handlers, make_request};

/// Test search/single_space for semantic space (index 0).
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_single_space_semantic() {
    let handlers = create_test_handlers();

    // Store content
    let store_params = json!({
        "content": "Transformers revolutionized natural language processing",
        "importance": 0.9
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    handlers.dispatch(store_request).await;

    // Search in semantic space only (index 0)
    let search_params = json!({
        "query": "NLP transformers",
        "space_index": 0,
        "topK": 5,
        "minSimilarity": 0.0  // P1-FIX-1: Required parameter for fail-fast
    });
    let search_request = make_request(
        "search/single_space",
        Some(JsonRpcId::Number(2)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(
        response.error.is_none(),
        "search/single_space should succeed"
    );
    let result = response.result.expect("Should have result");

    // Verify space info
    assert_eq!(
        result.get("space_index").and_then(|v| v.as_u64()),
        Some(0),
        "Should return space_index 0"
    );
    assert_eq!(
        result.get("space_name").and_then(|v| v.as_str()),
        Some("E1_Semantic"),
        "Should return space_name E1_Semantic"
    );
    assert!(result.get("results").is_some(), "Should have results");
    assert!(
        result.get("search_time_ms").is_some(),
        "Should report search time"
    );
}

/// Test search/single_space for code space (index 6).
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_single_space_code() {
    let handlers = create_test_handlers();

    // Store code-related content
    let store_params = json!({
        "content": "fn main() { println!(\"Hello, world!\"); }",
        "importance": 0.8
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    handlers.dispatch(store_request).await;

    // Search in code space (index 6)
    let search_params = json!({
        "query": "rust main function",
        "space_index": 6,
        "topK": 10,
        "minSimilarity": 0.0  // P1-FIX-1: Required parameter for fail-fast
    });
    let search_request = make_request(
        "search/single_space",
        Some(JsonRpcId::Number(2)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(
        response.error.is_none(),
        "search/single_space for code should succeed"
    );
    let result = response.result.expect("Should have result");

    assert_eq!(
        result.get("space_index").and_then(|v| v.as_u64()),
        Some(6),
        "Should return space_index 6"
    );
    assert_eq!(
        result.get("space_name").and_then(|v| v.as_str()),
        Some("E7_Code"),
        "Should return space_name E7_Code"
    );
}

/// Test search/single_space for SPLADE space (index 12).
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_single_space_splade() {
    let handlers = create_test_handlers();

    // Store content
    let store_params = json!({
        "content": "Sparse lexical and semantic pre-computed embeddings",
        "importance": 0.7
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    handlers.dispatch(store_request).await;

    // Search in SPLADE space (index 12)
    let search_params = json!({
        "query": "sparse embeddings",
        "space_index": 12,
        "topK": 5,
        "minSimilarity": 0.0  // P1-FIX-1: Required parameter for fail-fast
    });
    let search_request = make_request(
        "search/single_space",
        Some(JsonRpcId::Number(2)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(
        response.error.is_none(),
        "search/single_space for SPLADE should succeed"
    );
    let result = response.result.expect("Should have result");

    assert_eq!(
        result.get("space_index").and_then(|v| v.as_u64()),
        Some(12),
        "Should return space_index 12"
    );
    assert_eq!(
        result.get("space_name").and_then(|v| v.as_str()),
        Some("E13_SPLADE"),
        "Should return space_name E13_SPLADE"
    );
}

/// Test search/single_space fails with invalid space_index 13 (only 0-12 valid).
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_single_space_invalid_index_13_fails() {
    let handlers = create_test_handlers();

    let search_params = json!({
        "query": "test query",
        "space_index": 13
    });
    let search_request = make_request(
        "search/single_space",
        Some(JsonRpcId::Number(1)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(
        response.error.is_some(),
        "search/single_space must fail with space_index 13"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32602,
        "Should return INVALID_PARAMS error code"
    );
    assert!(
        error.message.contains("13") || error.message.contains("0-12"),
        "Error should mention valid range 0-12"
    );
}

/// Test search/single_space fails with missing space_index.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_single_space_missing_index_fails() {
    let handlers = create_test_handlers();

    let search_params = json!({
        "query": "test query"
    });
    let search_request = make_request(
        "search/single_space",
        Some(JsonRpcId::Number(1)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(
        response.error.is_some(),
        "search/single_space must fail without space_index"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32602,
        "Should return INVALID_PARAMS error code"
    );
    assert!(
        error.message.contains("space_index"),
        "Error should mention missing space_index"
    );
}

/// Test search/single_space accepts query_text as alias for query.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_single_space_query_text_alias() {
    let handlers = create_test_handlers();

    // Store content
    let store_params = json!({
        "content": "Alternative parameter naming test",
        "importance": 0.5
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    handlers.dispatch(store_request).await;

    // Use query_text instead of query
    let search_params = json!({
        "query_text": "parameter naming",
        "space_index": 0,
        "minSimilarity": 0.0  // P1-FIX-1: Required parameter for fail-fast
    });
    let search_request = make_request(
        "search/single_space",
        Some(JsonRpcId::Number(2)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(
        response.error.is_none(),
        "search/single_space should accept query_text parameter"
    );
}
