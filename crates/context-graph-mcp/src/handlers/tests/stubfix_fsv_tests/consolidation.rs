//! SPEC-STUBFIX-003: trigger_consolidation FSV Tests

use serde_json::json;

use crate::handlers::tests::{
    create_test_handlers_with_rocksdb_store_access, extract_mcp_tool_data, make_request,
};
use crate::protocol::JsonRpcId;

use super::helpers::create_test_fingerprint;

// ============================================================================
// SPEC-STUBFIX-003: trigger_consolidation FSV Tests
// ============================================================================

/// FSV-CONSOLIDATION-001: Empty store returns empty candidates.
#[tokio::test]
#[ignore = "Uses tools not registered in PRD v6"]
async fn test_consolidation_empty_store_returns_empty() {
    println!("\n================================================================================");
    println!("FSV-CONSOLIDATION-001: Empty Store Returns Empty Candidates");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    let count = store.count().await.expect("count works");
    assert_eq!(count, 0, "Store must be empty");

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "trigger_consolidation",
            "arguments": {
                "strategy": "similarity"
            }
        })),
    );
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Handler must succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let statistics = data.get("statistics").expect("Should have statistics");
    let pairs_evaluated = statistics
        .get("pairs_evaluated")
        .and_then(|v| v.as_u64())
        .unwrap_or(999);

    println!("[RESULT]");
    println!("  - pairs_evaluated: {}", pairs_evaluated);

    assert_eq!(pairs_evaluated, 0, "Empty store should evaluate 0 pairs");

    println!("\n[FSV-CONSOLIDATION-001 PASSED] Empty store returns no pairs");
    println!("================================================================================\n");
}

/// FSV-CONSOLIDATION-002: Orthogonal embeddings produce no candidates.
///
/// Stores fingerprints with very different content (different hashes = different embeddings).
/// Since synthetic embeddings are hash-based, different content produces orthogonal vectors.
#[tokio::test]
#[ignore = "Uses tools not registered in PRD v6"]
async fn test_consolidation_orthogonal_no_candidates() {
    println!("\n================================================================================");
    println!("FSV-CONSOLIDATION-002: Orthogonal Embeddings Produce No Candidates");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store fingerprints with very different content (different hashes)
    let test_data = [
        "machine learning optimization with gradient descent",
        "distributed systems consensus with Raft protocol",
        "database indexing using B-trees and hash indexes",
        "rust memory safety with ownership and borrowing",
        "natural language processing with transformers",
    ];

    println!("[SETUP] Storing {} diverse fingerprints:", test_data.len());
    for content in test_data.iter() {
        let fp = create_test_fingerprint(content, 0.7, 5);
        let id = store.store(fp).await.expect("Store must succeed");
        println!(
            "  - {} (content: {}...)",
            id,
            &content[..30.min(content.len())]
        );
    }

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "trigger_consolidation",
            "arguments": {
                "strategy": "similarity",
                "min_similarity": 0.95,
                "max_memories": 10
            }
        })),
    );
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Handler must succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let result_status = data
        .get("consolidation_result")
        .expect("Should have result");
    let candidate_count = result_status
        .get("candidate_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(999);
    let statistics = data.get("statistics").expect("Should have statistics");
    let pairs_evaluated = statistics
        .get("pairs_evaluated")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    println!("\n[RESULT]");
    println!("  - pairs_evaluated: {}", pairs_evaluated);
    println!("  - candidate_count: {}", candidate_count);

    // With 5 items, we have 5*4/2 = 10 pairs
    assert!(
        pairs_evaluated > 0,
        "Should evaluate some pairs, got {}",
        pairs_evaluated
    );

    // Different content should produce low similarity, so no candidates at 0.95 threshold
    assert_eq!(
        candidate_count, 0,
        "Orthogonal embeddings should produce no candidates at 0.95 threshold"
    );

    println!("\n[FSV-CONSOLIDATION-002 PASSED] Orthogonal embeddings produce no candidates");
    println!("================================================================================\n");
}

/// FSV-CONSOLIDATION-003: Identical content produces candidates.
///
/// Stores fingerprints with identical content (same hash = same embeddings).
#[tokio::test]
#[ignore = "Uses tools not registered in PRD v6"]
async fn test_consolidation_identical_produces_candidates() {
    println!("\n================================================================================");
    println!("FSV-CONSOLIDATION-003: Identical Content Produces Candidates");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store fingerprints with IDENTICAL content (same hash = same embeddings)
    let identical_content = "machine learning optimization with neural networks";

    println!("[SETUP] Storing 3 fingerprints with IDENTICAL content:");
    for i in 0..3 {
        let fp = create_test_fingerprint(identical_content, 0.7, 5);
        let id = store.store(fp).await.expect("Store must succeed");
        println!("  - [{}] {} (identical content)", i + 1, id);
    }

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "trigger_consolidation",
            "arguments": {
                "strategy": "similarity",
                "min_similarity": 0.9,
                "max_memories": 10
            }
        })),
    );
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Handler must succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let result_status = data
        .get("consolidation_result")
        .expect("Should have result");
    let candidate_count = result_status
        .get("candidate_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let statistics = data.get("statistics").expect("Should have statistics");
    let pairs_evaluated = statistics
        .get("pairs_evaluated")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    println!("\n[RESULT]");
    println!("  - pairs_evaluated: {}", pairs_evaluated);
    println!("  - candidate_count: {}", candidate_count);

    // Check candidates_sample if available
    if let Some(sample) = data.get("candidates_sample").and_then(|v| v.as_array()) {
        println!("  - candidates_sample count: {}", sample.len());
        for (i, c) in sample.iter().enumerate() {
            let similarity = c.get("similarity").and_then(|v| v.as_f64()).unwrap_or(-1.0);
            println!("    [{}] similarity: {:.4}", i + 1, similarity);
        }
    }

    // With 3 identical items, we have 3 pairs, and all should have similarity = 1.0
    assert!(pairs_evaluated > 0, "Should evaluate some pairs");

    // Identical content should produce candidates
    assert!(
        candidate_count >= 1,
        "Identical content should produce at least 1 consolidation candidate, got {}",
        candidate_count
    );

    println!(
        "\n[FSV-CONSOLIDATION-003 PASSED] Identical content produces consolidation candidates"
    );
    println!("================================================================================\n");
}

/// FSV-CONSOLIDATION-004: Limit parameter is respected.
#[tokio::test]
#[ignore = "Uses tools not registered in PRD v6"]
async fn test_consolidation_limit_respected() {
    println!("\n================================================================================");
    println!("FSV-CONSOLIDATION-004: Limit Parameter Is Respected");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store 10 fingerprints
    println!("[SETUP] Storing 10 fingerprints:");
    for i in 0..10 {
        let content = format!("memory content number {} for testing", i);
        let fp = create_test_fingerprint(&content, 0.7, 5);
        let id = store.store(fp).await.expect("Store must succeed");
        println!("  - [{}] {}", i + 1, id);
    }

    // Request with max_memories = 5 (should only consider 5 memories for pairs)
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "trigger_consolidation",
            "arguments": {
                "strategy": "similarity",
                "min_similarity": 0.5,
                "max_memories": 5
            }
        })),
    );
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Handler must succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let statistics = data.get("statistics").expect("Should have statistics");
    let max_memories_limit = statistics
        .get("max_memories_limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(999);

    println!("\n[RESULT]");
    println!("  - max_memories_limit in response: {}", max_memories_limit);

    assert_eq!(max_memories_limit, 5, "max_memories_limit should be 5");

    println!("\n[FSV-CONSOLIDATION-004 PASSED] Limit parameter is respected");
    println!("================================================================================\n");
}
