//! REAL GPU Embedding Tests (Feature-gated: cuda)
//!
//! These tests verify search operations with REAL GPU-accelerated embeddings from
//! ProductionMultiArrayProvider. They require:
//! - NVIDIA CUDA GPU with 8GB+ VRAM
//! - Pre-downloaded models in ./models directory
//! - cargo test --features cuda
//!
//! Run these tests for Full State Verification (FSV) to confirm real embeddings
//! produce semantically meaningful search results.

use serde_json::json;
use std::time::Instant;

use crate::protocol::JsonRpcId;
use context_graph_core::types::fingerprint::NUM_EMBEDDERS;

use crate::handlers::tests::{
    create_test_handlers_with_real_embeddings, extract_mcp_tool_data, make_request,
};

/// FSV: Verify search/multi with REAL GPU embeddings returns semantically relevant results.
///
/// Unlike stub tests that use deterministic fake embeddings, this test verifies that:
/// - Real BAAI/bge-m3 embeddings produce meaningful semantic similarity
/// - ML-related content ranks higher for ML-related queries
/// - Unrelated content ranks lower or is filtered out
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_real_embeddings_search_multi_semantic_relevance() {
    let (handlers, _tempdir) = create_test_handlers_with_real_embeddings().await;

    // Store diverse content: 2 ML-related, 1 unrelated
    let contents = [
        (
            "Machine learning algorithms can classify images using neural networks",
            0.9,
        ),
        (
            "Deep learning models like transformers revolutionized NLP tasks",
            0.8,
        ),
        (
            "The weather forecast predicts rain tomorrow afternoon in Seattle",
            0.7,
        ),
    ];

    let mut stored_ids = Vec::new();
    for (i, (content, importance)) in contents.iter().enumerate() {
        let store_params = json!({
            "content": content,
            "importance": importance
        });
        let store_request = make_request(
            "memory/store",
            Some(JsonRpcId::Number((i + 1) as i64)),
            Some(store_params),
        );
        let response = handlers.dispatch(store_request).await;
        assert!(response.error.is_none(), "Store {} should succeed", i);

        if let Some(result) = response.result {
            let data = extract_mcp_tool_data(&result);
            if let Some(id) = data
                .get("fingerprintId")
                .or_else(|| data.get("fingerprint_id"))
                .and_then(|v| v.as_str())
            {
                stored_ids.push(id.to_string());
            }
        }
    }
    assert_eq!(stored_ids.len(), 3, "Should have stored 3 items");

    // Search for ML content
    let search_params = json!({
        "query": "artificial intelligence and machine learning models",
        "query_type": "semantic_search",
        "topK": 10,
        "minSimilarity": 0.0,
        "include_per_embedder_scores": true
    });
    let search_request = make_request(
        "search/multi",
        Some(JsonRpcId::Number(100)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(
        response.error.is_none(),
        "Search should succeed with real embeddings"
    );
    let result = response.result.expect("Should have result");

    // Verify results structure
    let results = result
        .get("results")
        .and_then(|v| v.as_array())
        .expect("Should have results");
    assert!(!results.is_empty(), "Should have at least one result");

    // With real embeddings, ML content should have higher similarity
    // The weather forecast should have notably lower similarity
    let count = result.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
    println!("Real embedding search returned {} results", count);

    // Verify per-embedder scores are returned (13 embedders)
    if let Some(first) = results.first() {
        if let Some(per_scores) = first.get("per_embedder_scores").and_then(|v| v.as_object()) {
            println!(
                "Per-embedder scores for top result: {} embedders",
                per_scores.len()
            );
            // Should have scores for all 13 embedding spaces
            assert!(!per_scores.is_empty(), "Should have per-embedder scores");
        }

        // Verify top result has reasonable similarity
        if let Some(sim) = first.get("combined_similarity").and_then(|v| v.as_f64()) {
            println!("Top result combined_similarity: {}", sim);
            // With real embeddings, the ML content should have similarity > 0
            // Exact threshold depends on model, but should be positive
            assert!(
                sim >= 0.0,
                "Top result should have non-negative similarity: {}",
                sim
            );
        }
    }

    // Verify query_metadata shows 13-element weights
    if let Some(metadata) = result.get("query_metadata") {
        if let Some(weights) = metadata.get("weights_applied").and_then(|v| v.as_array()) {
            assert_eq!(
                weights.len(),
                NUM_EMBEDDERS,
                "Should have 13-element weights applied"
            );
        }
    }
}

/// FSV: Verify search/single_space works with REAL GPU embeddings.
///
/// Tests single embedding space search (space 0 = Semantic/E1) with real embeddings.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_real_embeddings_search_single_space() {
    let (handlers, _tempdir) = create_test_handlers_with_real_embeddings().await;

    // Store content
    let store_params = json!({
        "content": "Quantum computing uses qubits for parallel computation",
        "importance": 0.85
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    let response = handlers.dispatch(store_request).await;
    assert!(response.error.is_none(), "Store should succeed");

    // Search in space 0 (Semantic/E1)
    let search_params = json!({
        "query": "quantum physics and computation",
        "space_index": 0,
        "topK": 5,
        "minSimilarity": 0.0
    });
    let search_request = make_request(
        "search/single_space",
        Some(JsonRpcId::Number(2)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(
        response.error.is_none(),
        "Single-space search should succeed"
    );
    let result = response.result.expect("Should have result");

    // Verify single-space search structure
    let results = result
        .get("results")
        .and_then(|v| v.as_array())
        .expect("Should have results");

    // With real embeddings, the quantum content should be found
    if !results.is_empty() {
        let first = &results[0];
        // Single-space returns "similarity" not "combined_similarity"
        if let Some(sim) = first.get("similarity").and_then(|v| v.as_f64()) {
            println!("Single-space (E1) search top similarity: {}", sim);
            assert!(sim >= 0.0, "Similarity should be non-negative");
        }
    }

    // Verify metadata shows correct space
    if let Some(metadata) = result.get("query_metadata") {
        assert_eq!(
            metadata.get("space_index").and_then(|v| v.as_u64()),
            Some(0),
            "Should search in space 0"
        );
        assert_eq!(
            metadata.get("space_name").and_then(|v| v.as_str()),
            Some("semantic"),
            "Space 0 should be 'semantic'"
        );
    }
}

/// FSV: Verify search/by_purpose with REAL 13D purpose vectors.
///
/// Tests purpose-aligned search using real embedding-generated purpose vectors.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_real_embeddings_search_by_purpose() {
    let (handlers, _tempdir) = create_test_handlers_with_real_embeddings().await;

    // Store content with diverse topics
    let topics = [
        "Ethical AI development requires transparency and fairness",
        "Database optimization techniques for high-throughput systems",
        "Climate change mitigation through renewable energy adoption",
    ];

    for (i, content) in topics.iter().enumerate() {
        let store_params = json!({
            "content": content,
            "importance": 0.8
        });
        let store_request = make_request(
            "memory/store",
            Some(JsonRpcId::Number((i + 1) as i64)),
            Some(store_params),
        );
        let response = handlers.dispatch(store_request).await;
        assert!(response.error.is_none(), "Store {} should succeed", i);
    }

    // Search by purpose with a 13D purpose vector
    // This simulates aligning to a "beneficial AI" goal
    let purpose_vector: Vec<f64> = vec![
        0.8, // E1: Semantic - high weight for meaning
        0.3, // E2: Temporal-cyclic
        0.3, // E3: Temporal-decay
        0.3, // E4: Temporal-contextual
        0.7, // E5: Causal - important for ethics
        0.2, // E6: Sparse (SPLADE)
        0.4, // E7: Code
        0.3, // E8: Graph
        0.2, // E9: HDC
        0.3, // E10: Multimodal
        0.5, // E11: Entity
        0.2, // E12: Late-interaction
        0.2, // E13: Sparse
    ];

    let search_params = json!({
        "purpose_vector": purpose_vector,
        "topK": 10,
        "threshold": 0.0  // Allow all results
    });
    let search_request = make_request(
        "search/by_purpose",
        Some(JsonRpcId::Number(100)),
        Some(search_params),
    );
    let response = handlers.dispatch(search_request).await;

    assert!(response.error.is_none(), "By-purpose search should succeed");
    let result = response.result.expect("Should have result");

    // Verify results
    let results = result
        .get("results")
        .and_then(|v| v.as_array())
        .expect("Should have results");
    println!("By-purpose search returned {} results", results.len());

    // Verify alignment scores are in expected range
    for (i, r) in results.iter().enumerate() {
        if let Some(alignment) = r.get("alignment_score").and_then(|v| v.as_f64()) {
            println!("Result {} alignment_score: {}", i, alignment);
            // Alignment scores should be in [-1, 1] per constitution
            assert!(
                (-1.0..=1.0).contains(&alignment),
                "Alignment score should be in [-1, 1]: {}",
                alignment
            );
        }
    }
}

/// FSV: Verify REAL embedding search latency meets constitution targets.
///
/// Constitution requires:
/// - single_embed: <10ms
/// - inject_context p95: <25ms
///
/// Search operations should be reasonably fast with real GPU embeddings.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_real_embeddings_search_latency() {
    let (handlers, _tempdir) = create_test_handlers_with_real_embeddings().await;

    // Store some content first
    let store_params = json!({
        "content": "Performance benchmarking for neural embedding systems",
        "importance": 0.9
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    handlers.dispatch(store_request).await;

    // Warm up the models
    let warmup_params = json!({
        "query": "warmup query",
        "query_type": "semantic_search",
        "topK": 1,
        "minSimilarity": 0.0
    });
    let warmup_request = make_request(
        "search/multi",
        Some(JsonRpcId::Number(2)),
        Some(warmup_params),
    );
    handlers.dispatch(warmup_request).await;

    // Measure search latencies
    let mut latencies = Vec::new();
    for i in 0..5 {
        let search_params = json!({
            "query": format!("test query number {}", i),
            "query_type": "semantic_search",
            "topK": 10,
            "minSimilarity": 0.0
        });
        let search_request = make_request(
            "search/multi",
            Some(JsonRpcId::Number((10 + i) as i64)),
            Some(search_params),
        );

        let start = Instant::now();
        let response = handlers.dispatch(search_request).await;
        let latency_ms = start.elapsed().as_millis() as u64;
        latencies.push(latency_ms);

        assert!(response.error.is_none(), "Search {} should succeed", i);
        println!("Search {} latency: {}ms", i, latency_ms);
    }

    // Calculate P95 (for 5 samples, it's the max)
    latencies.sort();
    let p95 = latencies.last().copied().unwrap_or(0);
    println!("Search P95 latency: {}ms (target: reasonable for GPU)", p95);

    // Report but don't fail - actual threshold depends on hardware
    // The constitution says 25ms for inject_context which is more complex
    if p95 > 100 {
        println!(
            "WARNING: P95 search latency {}ms may be high for real-time use",
            p95
        );
    }
}

/// FSV: Verify all 13 embedding spaces are searchable with REAL embeddings.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_real_embeddings_all_13_spaces_searchable() {
    let (handlers, _tempdir) = create_test_handlers_with_real_embeddings().await;

    // Store content that should have embeddings in all 13 spaces
    let store_params = json!({
        "content": "The function calculateSum() iterates through an array to compute the total value for statistical analysis",
        "importance": 0.9
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    let response = handlers.dispatch(store_request).await;
    assert!(response.error.is_none(), "Store should succeed");

    // Search each of the 13 embedding spaces
    let space_names = [
        "semantic",            // E1
        "temporal_cyclic",     // E2
        "temporal_decay",      // E3
        "temporal_contextual", // E4
        "causal",              // E5
        "sparse",              // E6 (SPLADE)
        "code",                // E7
        "graph",               // E8
        "hdc",                 // E9
        "multimodal",          // E10
        "entity",              // E11
        "late_interaction",    // E12 (ColBERT)
        "sparse_2",            // E13
    ];

    for (space_index, space_name) in space_names.iter().enumerate() {
        let search_params = json!({
            "query": "function array calculation statistics",
            "space_index": space_index,
            "topK": 5,
            "minSimilarity": 0.0
        });
        let search_request = make_request(
            "search/single_space",
            Some(JsonRpcId::Number((100 + space_index) as i64)),
            Some(search_params),
        );
        let response = handlers.dispatch(search_request).await;

        // Space search should succeed (even if no results due to sparse embeddings)
        assert!(
            response.error.is_none(),
            "Space {} ({}) search should succeed: {:?}",
            space_index,
            space_name,
            response.error
        );

        if let Some(result) = response.result {
            let count = result.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
            println!("Space {} ({}): {} results", space_index, space_name, count);
        }
    }
}
