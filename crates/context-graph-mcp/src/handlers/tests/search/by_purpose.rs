//! Tests for search/by_purpose handler.
//!
//! Tests purpose-based search with 13D purpose vector alignment.

use serde_json::json;

use crate::protocol::JsonRpcId;

use crate::handlers::tests::{create_test_handlers, make_request};

/// Test search/by_purpose with custom purpose vector.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_by_purpose_custom_vector() {
    let handlers = create_test_handlers();

    // Store content
    let store_params = json!({
        "content": "Purpose-driven content for alignment testing",
        "importance": 0.8
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    handlers.dispatch(store_request).await;

    // Custom 13D purpose vector
    let purpose_vector: Vec<f64> = vec![
        0.3, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0,
    ];

    let search_params = json!({
        "purpose_vector": purpose_vector,
        "topK": 5
    });
    let search_request = make_request(
        "search/by_purpose",
        Some(JsonRpcId::Number(2)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(response.error.is_none(), "search/by_purpose should succeed");
    let result = response.result.expect("Should have result");

    assert!(result.get("results").is_some(), "Should have results array");
    assert!(result.get("count").is_some(), "Should have count");
    assert!(
        result.get("search_time_ms").is_some(),
        "Should report search time"
    );
}

/// Test search/by_purpose without purpose_vector uses default.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_by_purpose_default_vector() {
    let handlers = create_test_handlers();

    // Store content
    let store_params = json!({
        "content": "Content for default purpose search",
        "importance": 0.6
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    handlers.dispatch(store_request).await;

    // No purpose_vector provided - uses default
    let search_params = json!({
        "topK": 10
    });
    let search_request = make_request(
        "search/by_purpose",
        Some(JsonRpcId::Number(2)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(
        response.error.is_none(),
        "search/by_purpose should succeed with default purpose vector"
    );
}

/// Test search/by_purpose with min_alignment filter.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_by_purpose_min_alignment() {
    let handlers = create_test_handlers();

    // Store multiple items
    for i in 0..3 {
        let store_params = json!({
            "content": format!("Purpose alignment test content number {}", i),
            "importance": 0.5 + (i as f64 * 0.1)
        });
        let store_request = make_request(
            "memory/store",
            Some(JsonRpcId::Number(i as i64 + 1)),
            Some(store_params),
        );
        handlers.dispatch(store_request).await;
    }

    // Search with minimum alignment threshold
    let search_params = json!({
        "min_alignment": 0.5,
        "topK": 10
    });
    let search_request = make_request(
        "search/by_purpose",
        Some(JsonRpcId::Number(10)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(response.error.is_none(), "search/by_purpose should succeed");
    let result = response.result.expect("Should have result");

    // Verify min_alignment filter is reported
    let min_filter = result.get("min_alignment_filter").and_then(|v| v.as_f64());
    assert_eq!(min_filter, Some(0.5), "Should report min_alignment_filter");
}

/// Test search/by_purpose fails with wrong-sized purpose vector.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_by_purpose_wrong_size_vector_fails() {
    let handlers = create_test_handlers();

    // Only 12 elements (must be 13!)
    let wrong_vector: Vec<f64> = vec![0.1; 12];

    let search_params = json!({
        "purpose_vector": wrong_vector
    });
    let search_request = make_request(
        "search/by_purpose",
        Some(JsonRpcId::Number(1)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(
        response.error.is_some(),
        "search/by_purpose must fail with 12-element vector"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32602,
        "Should return INVALID_PARAMS error code"
    );
    assert!(
        error.message.contains("13") || error.message.contains("elements"),
        "Error should mention 13 elements required"
    );
}

/// Test search/by_purpose result contains purpose alignment scores.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_search_by_purpose_result_structure() {
    let handlers = create_test_handlers();

    // Store content
    let store_params = json!({
        "content": "Result structure verification content",
        "importance": 0.8
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    handlers.dispatch(store_request).await;

    let search_params = json!({
        "topK": 5
    });
    let search_request = make_request(
        "search/by_purpose",
        Some(JsonRpcId::Number(2)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(response.error.is_none(), "search/by_purpose should succeed");
    let result = response.result.expect("Should have result");

    let results = result.get("results").and_then(|v| v.as_array());
    assert!(results.is_some(), "Should have results array");

    if let Some(results) = results {
        if !results.is_empty() {
            let first = &results[0];
            assert!(
                first.get("fingerprintId").is_some(),
                "Result should have fingerprintId"
            );
            assert!(
                first.get("purpose_alignment").is_some(),
                "Result should have purpose_alignment"
            );
            assert!(
                first.get("alignment_score").is_some(),
                "Result should have alignment_score"
            );
            assert!(
                first.get("purpose_vector").is_some(),
                "Result should have purpose_vector"
            );
        }
    }
}
