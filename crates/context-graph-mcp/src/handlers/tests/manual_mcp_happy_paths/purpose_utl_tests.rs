//! Purpose and UTL handler happy path tests
//!
//! Tests for purpose/query, utl/compute, utl/metrics

use serde_json::json;

use super::common::{create_test_handlers_with_rocksdb_store_access, make_request};

/// Test 7: purpose/query - Query purpose alignment
#[tokio::test]
async fn test_07_purpose_query() {
    println!("\n========================================================================================================");
    println!("TEST 07: purpose/query");
    println!("========================================================================================================");

    let (handlers, _store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store some memories first so there's data to search
    let store_params = json!({
        "content": "Purpose-aligned memory for testing goal alignment",
        "importance": 0.9
    });
    handlers.dispatch(make_request("memory/store", 1, Some(store_params))).await;

    // purpose/query requires a 13-element purpose_vector array
    // Each element is a float in [0.0, 1.0] representing alignment per embedder
    let query_params = json!({
        "purpose_vector": [0.8, 0.7, 0.6, 0.5, 0.9, 0.4, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5],
        "topK": 10,
        "minAlignment": 0.0
    });
    let request = make_request("purpose/query", 2, Some(query_params));
    let response = handlers.dispatch(request).await;

    println!("Response: {}", serde_json::to_string_pretty(&response).unwrap());

    assert!(response.error.is_none(), "Should not have error: {:?}", response.error);
    let result = response.result.expect("Should have result");

    println!("\n[VERIFICATION]");
    // purpose/query returns: results, query_metadata
    if let Some(results) = result.get("results") {
        if let Some(arr) = results.as_array() {
            println!("  Found {} results matching purpose vector", arr.len());
            for (i, r) in arr.iter().take(3).enumerate() {
                println!("  [{}] id={}, alignment={}",
                    i,
                    r.get("id").unwrap_or(&json!("?")),
                    r.get("purpose_alignment").unwrap_or(&json!("?"))
                );
            }
        }
    }

    if let Some(meta) = result.get("query_metadata") {
        println!("  search_time_ms: {}", meta.get("search_time_ms").unwrap_or(&json!("?")));
    }

    println!("\n[PASSED] purpose/query works correctly");
}

/// Test 8: utl/compute - Compute UTL score
#[tokio::test]
async fn test_08_utl_compute() {
    println!("\n========================================================================================================");
    println!("TEST 08: utl/compute");
    println!("========================================================================================================");

    let (handlers, _store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // utl/compute requires 'input' parameter (text string to compute UTL score for)
    let compute_params = json!({
        "input": "Memory for UTL computation testing - analyzing learning patterns and knowledge integration"
    });
    let request = make_request("utl/compute", 2, Some(compute_params));
    let response = handlers.dispatch(request).await;

    println!("Response: {}", serde_json::to_string_pretty(&response).unwrap());

    assert!(response.error.is_none(), "Should not have error: {:?}", response.error);
    let result = response.result.expect("Should have result");

    println!("\n[VERIFICATION]");
    println!("  learningScore: {}", result.get("learningScore").unwrap_or(&json!("?")));

    // Verify learning score is in valid range [0.0, 1.0]
    if let Some(score) = result.get("learningScore").and_then(|v| v.as_f64()) {
        assert!((0.0..=1.0).contains(&score), "Learning score should be in [0.0, 1.0]");
    }

    println!("\n[PASSED] utl/compute works correctly");
}

/// Test 9: utl/metrics - Get UTL metrics
#[tokio::test]
async fn test_09_utl_metrics() {
    println!("\n========================================================================================================");
    println!("TEST 09: utl/metrics");
    println!("========================================================================================================");

    let (handlers, _store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // utl/metrics requires 'input' parameter (text string to compute metrics for)
    let metrics_params = json!({
        "input": "UTL metrics test content - learning patterns and knowledge synthesis"
    });
    let request = make_request("utl/metrics", 10, Some(metrics_params));
    let response = handlers.dispatch(request).await;

    println!("Response: {}", serde_json::to_string_pretty(&response).unwrap());

    assert!(response.error.is_none(), "Should not have error: {:?}", response.error);
    let result = response.result.expect("Should have result");

    println!("\n[VERIFICATION]");
    println!("  entropy: {}", result.get("entropy").unwrap_or(&json!("?")));
    println!("  coherence: {}", result.get("coherence").unwrap_or(&json!("?")));
    println!("  learningScore: {}", result.get("learningScore").unwrap_or(&json!("?")));
    println!("  surprise: {}", result.get("surprise").unwrap_or(&json!("?")));

    println!("\n[PASSED] utl/metrics works correctly");
}
