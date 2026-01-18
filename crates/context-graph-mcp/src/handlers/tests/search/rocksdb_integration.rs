//! RocksDB Integration Tests for Search Handlers
//!
//! These tests use RocksDbTeleologicalStore for REAL persistent storage.
//! This verifies that search operations work correctly with real RocksDB.
//!
//! Components used:
//! - RocksDbTeleologicalStore: REAL (17 column families, persistent storage)
//! - UtlProcessorAdapter: REAL (6-component UTL computation)
//! - StubMultiArrayProvider: STUB (GPU required for real embeddings - TASK-F007)
//!
//! CRITICAL: The `_tempdir` variable MUST be kept alive for the entire test.

use serde_json::json;

use crate::protocol::JsonRpcId;
use crate::weights::WEIGHT_PROFILES;
use context_graph_core::types::fingerprint::NUM_EMBEDDERS;

use crate::handlers::tests::{create_test_handlers_with_rocksdb, make_request};

/// Integration test: search/multi with REAL RocksDB storage.
///
/// Verifies that multi-space search works correctly with real persistent storage.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_rocksdb_integration_search_multi() {
    let (handlers, _tempdir) = create_test_handlers_with_rocksdb().await;

    // Store multiple items to search
    let contents = [
        "Machine learning and artificial intelligence research",
        "Neural network architectures for deep learning",
        "Natural language processing with transformers",
    ];

    for (i, content) in contents.iter().enumerate() {
        let store_params = json!({
            "content": content,
            "importance": 0.7 + (i as f64 * 0.05)
        });
        let store_request = make_request(
            "memory/store",
            Some(JsonRpcId::Number(i as i64 + 1)),
            Some(store_params),
        );
        let store_response = handlers.dispatch(store_request).await;
        assert!(
            store_response.error.is_none(),
            "STORE[{}] must succeed: {:?}",
            i,
            store_response.error
        );
    }

    // Search with semantic_search preset
    let search_params = json!({
        "query": "deep learning neural networks",
        "query_type": "semantic_search",
        "topK": 10,
        "minSimilarity": 0.0,  // P1-FIX-1: Required parameter for fail-fast
        "include_per_embedder_scores": true
    });
    let search_request = make_request(
        "search/multi",
        Some(JsonRpcId::Number(10)),
        Some(search_params),
    );
    let search_response = handlers.dispatch(search_request).await;

    assert!(
        search_response.error.is_none(),
        "search/multi must succeed with RocksDB: {:?}",
        search_response.error
    );
    let search_result = search_response.result.expect("Must have result");

    // Verify results structure
    let results = search_result
        .get("results")
        .and_then(|v| v.as_array())
        .expect("Must have results array");
    let count = search_result
        .get("count")
        .and_then(|v| v.as_u64())
        .expect("Must have count");
    assert_eq!(
        count as usize,
        results.len(),
        "Count must match results length"
    );
    assert!(!results.is_empty(), "Must find at least one result");

    // Verify query_metadata
    let metadata = search_result
        .get("query_metadata")
        .expect("Must have query_metadata");
    assert_eq!(
        metadata.get("query_type_used").and_then(|v| v.as_str()),
        Some("semantic_search"),
        "Must use semantic_search preset"
    );
    let weights_applied = metadata.get("weights_applied").and_then(|v| v.as_array());
    assert!(weights_applied.is_some(), "Must have weights_applied");
    assert_eq!(
        weights_applied.unwrap().len(),
        NUM_EMBEDDERS,
        "Must have 13 weights"
    );

    // Verify per-embedder scores in first result
    if !results.is_empty() {
        let first = &results[0];
        assert!(
            first.get("fingerprintId").is_some(),
            "Result must have fingerprintId"
        );
        assert!(
            first.get("aggregate_similarity").is_some(),
            "Result must have aggregate_similarity"
        );
        assert!(
            first.get("per_embedder_scores").is_some(),
            "Result must have per_embedder_scores"
        );
        assert!(
            first.get("top_contributing_spaces").is_some(),
            "Result must have top_contributing_spaces"
        );
    }
}

/// Integration test: search/multi with custom 13-element weights.
///
/// Verifies that custom weight profiles work with real RocksDB storage.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_rocksdb_integration_search_multi_custom_weights() {
    let (handlers, _tempdir) = create_test_handlers_with_rocksdb().await;

    // Store content
    let store_params = json!({
        "content": "Rust programming language with ownership and borrowing",
        "importance": 0.8
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    let store_response = handlers.dispatch(store_request).await;
    assert!(store_response.error.is_none(), "STORE must succeed");

    // Custom weights emphasizing code embedding (index 6)
    let custom_weights: Vec<f64> = vec![
        0.10, // E1 Semantic
        0.03, // E2 Temporal-Recent
        0.03, // E3 Temporal-Periodic
        0.03, // E4 Temporal-Positional
        0.05, // E5 Causal
        0.03, // E6 Sparse
        0.40, // E7 Code (emphasized)
        0.10, // E8 Graph
        0.10, // E9 HDC
        0.03, // E10 Multimodal
        0.03, // E11 Entity
        0.03, // E12 Late Interaction
        0.04, // E13 SPLADE
    ];

    let search_params = json!({
        "query": "rust ownership",
        "query_type": "custom",
        "weights": custom_weights,
        "topK": 5,
        "minSimilarity": 0.0  // P1-FIX-1: Required parameter for fail-fast
    });
    let search_request = make_request(
        "search/multi",
        Some(JsonRpcId::Number(2)),
        Some(search_params),
    );
    let search_response = handlers.dispatch(search_request).await;

    assert!(
        search_response.error.is_none(),
        "search/multi with custom weights must succeed: {:?}",
        search_response.error
    );
    let search_result = search_response.result.expect("Must have result");

    // Verify custom weights were applied
    let metadata = search_result
        .get("query_metadata")
        .expect("Must have metadata");
    let applied = metadata
        .get("weights_applied")
        .and_then(|v| v.as_array())
        .expect("Must have weights");
    assert_eq!(applied.len(), 13, "Must have exactly 13 weights");

    // Verify code weight (index 6) is 0.40
    let code_weight = applied[6].as_f64().expect("Weight must be f64");
    assert!(
        (code_weight - 0.40).abs() < 0.01,
        "Code weight must be 0.40, got {}",
        code_weight
    );
}

/// Integration test: search/single_space with REAL RocksDB storage.
///
/// Verifies that single-space search targeting specific embedding works.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_rocksdb_integration_search_single_space() {
    let (handlers, _tempdir) = create_test_handlers_with_rocksdb().await;

    // Store content
    let store_params = json!({
        "content": "Knowledge graphs for entity relationship modeling",
        "importance": 0.75
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    let store_response = handlers.dispatch(store_request).await;
    assert!(store_response.error.is_none(), "STORE must succeed");

    // Search in graph embedding space (index 7)
    let search_params = json!({
        "query": "entity relationships",
        "space_index": 7,
        "topK": 5,
        "minSimilarity": 0.0  // P1-FIX-1: Required parameter for fail-fast
    });
    let search_request = make_request(
        "search/single_space",
        Some(JsonRpcId::Number(2)),
        Some(search_params),
    );
    let search_response = handlers.dispatch(search_request).await;

    assert!(
        search_response.error.is_none(),
        "search/single_space must succeed: {:?}",
        search_response.error
    );
    let search_result = search_response.result.expect("Must have result");

    // Verify response structure
    assert_eq!(
        search_result.get("space_index").and_then(|v| v.as_u64()),
        Some(7),
        "Must use space_index 7"
    );
    assert!(
        search_result.get("space_name").is_some(),
        "Must have space_name"
    );
    assert!(
        search_result.get("search_time_ms").is_some(),
        "Must have search_time_ms"
    );

    // Verify results found
    let results = search_result.get("results").and_then(|v| v.as_array());
    assert!(results.is_some(), "Must have results array");
}

/// Integration test: search/by_purpose with REAL RocksDB storage.
///
/// Verifies that purpose-based search works with real storage.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_rocksdb_integration_search_by_purpose() {
    let (handlers, _tempdir) = create_test_handlers_with_rocksdb().await;

    // Store multiple items with different content
    let contents = [
        "Building scalable distributed systems",
        "Optimizing database query performance",
        "Implementing fault-tolerant architectures",
    ];

    for (i, content) in contents.iter().enumerate() {
        let store_params = json!({
            "content": content,
            "importance": 0.6 + (i as f64 * 0.1)
        });
        let store_request = make_request(
            "memory/store",
            Some(JsonRpcId::Number(i as i64 + 1)),
            Some(store_params),
        );
        let store_response = handlers.dispatch(store_request).await;
        assert!(store_response.error.is_none(), "STORE[{}] must succeed", i);
    }

    // Search by purpose vector (13 elements)
    let purpose_vector: Vec<f64> = vec![
        0.9, 0.7, 0.5, 0.4, 0.6, 0.3, 0.8, 0.7, 0.5, 0.4, 0.6, 0.5, 0.4,
    ];
    let search_params = json!({
        "purpose_vector": purpose_vector,
        "topK": 10,
        "threshold": 0.1
    });
    let search_request = make_request(
        "search/by_purpose",
        Some(JsonRpcId::Number(10)),
        Some(search_params),
    );
    let search_response = handlers.dispatch(search_request).await;

    assert!(
        search_response.error.is_none(),
        "search/by_purpose must succeed: {:?}",
        search_response.error
    );
    let search_result = search_response.result.expect("Must have result");

    // Verify results structure
    assert!(search_result.get("results").is_some(), "Must have results");
    assert!(search_result.get("count").is_some(), "Must have count");
    assert!(
        search_result.get("search_time_ms").is_some(),
        "Must have search_time_ms"
    );
    // min_alignment_filter may be None if no threshold provided
}

/// Integration test: search/weight_profiles works with RocksDB handlers.
///
/// Verifies that weight profiles endpoint returns correct 13-weight profiles.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_rocksdb_integration_weight_profiles() {
    let (handlers, _tempdir) = create_test_handlers_with_rocksdb().await;

    let request = make_request("search/weight_profiles", Some(JsonRpcId::Number(1)), None);
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "search/weight_profiles must succeed: {:?}",
        response.error
    );
    let result = response.result.expect("Must have result");

    // Verify profiles array - use WEIGHT_PROFILES.len() for actual count
    let profiles = result
        .get("profiles")
        .and_then(|v| v.as_array())
        .expect("Must have profiles");
    assert_eq!(
        profiles.len(),
        WEIGHT_PROFILES.len(),
        "Must return all {} weight profiles, got {}",
        WEIGHT_PROFILES.len(),
        profiles.len()
    );

    // Verify each profile has exactly 13 weights
    for (i, profile) in profiles.iter().enumerate() {
        let weights = profile.get("weights").and_then(|v| v.as_array());
        assert!(weights.is_some(), "Profile {} must have weights", i);
        assert_eq!(
            weights.unwrap().len(),
            NUM_EMBEDDERS,
            "Profile {} must have exactly 13 weights",
            i
        );
        // Verify profile has name and description
        assert!(
            profile.get("name").is_some(),
            "Profile {} must have name",
            i
        );
        assert!(
            profile.get("primary_spaces").is_some(),
            "Profile {} must have primary_spaces",
            i
        );
    }

    // Verify embedding spaces
    let spaces = result.get("embedding_spaces").and_then(|v| v.as_array());
    assert!(spaces.is_some(), "Must have embedding_spaces");
    assert_eq!(
        spaces.unwrap().len(),
        NUM_EMBEDDERS,
        "Must have 13 embedding spaces"
    );

    // Verify total_spaces
    assert_eq!(
        result.get("total_spaces").and_then(|v| v.as_u64()),
        Some(NUM_EMBEDDERS as u64),
        "total_spaces must be 13"
    );

    // Verify default values
    assert_eq!(
        result.get("default_aggregation").and_then(|v| v.as_str()),
        Some("rrf"),
        "default_aggregation must be rrf"
    );
}

/// Integration test: Error handling with REAL RocksDB.
///
/// Verifies that proper error codes are returned for invalid operations.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_rocksdb_integration_search_error_handling() {
    let (handlers, _tempdir) = create_test_handlers_with_rocksdb().await;

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
        "Missing query must return INVALID_PARAMS with real RocksDB"
    );

    // Test INVALID_PARAMS for invalid space_index
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

    // Test INVALID_PARAMS for 12-element weights
    let twelve_weights: Vec<f64> = vec![0.083; 12];
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
        "12-element weights must return INVALID_PARAMS with real RocksDB"
    );
}
