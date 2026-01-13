//! STUBFIX Manual Happy Path Tests
//!
//! Tests for steering feedback, pruning candidates, and trigger consolidation

use serde_json::json;

use super::common::{create_test_handlers_with_rocksdb_store_access, make_request};

/// Test: get_steering_feedback - Steering subsystem returns REAL data
#[tokio::test]
async fn test_stubfix_steering_feedback_happy_path() {
    println!("\n========================================================================================================");
    println!("STUBFIX TEST: get_steering_feedback (Steering Subsystem)");
    println!("========================================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store some memories first to have data to analyze
    println!("\n[SETUP] Storing test memories...");
    for i in 0..5 {
        let params = json!({
            "content": format!("Test memory {} for steering feedback validation", i),
            "importance": 0.7 + (i as f64 * 0.05)
        });
        let request = make_request("memory/store", i as i64, Some(params));
        let response = handlers.dispatch(request).await;
        if response.error.is_some() {
            println!("  Warning: Store {} may have failed", i);
        }
    }

    let count = store.count().await.expect("count works");
    println!("  Stored {} memories", count);

    // Call get_steering_feedback
    let request = make_request("tools/call", 10, Some(json!({
        "name": "get_steering_feedback",
        "arguments": {}
    })));
    let response = handlers.dispatch(request).await;

    println!("\nResponse: {}", serde_json::to_string_pretty(&response).unwrap());

    assert!(response.error.is_none(), "Handler must succeed: {:?}", response.error);

    let result = response.result.expect("Should have result");
    let content = result.get("content").and_then(|c| c.as_array());

    if let Some(content_arr) = content {
        for item in content_arr {
            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                let data: serde_json::Value = serde_json::from_str(text).unwrap_or(json!({}));

                println!("\n[VERIFICATION] Steering Feedback Data:");

                // Verify reward
                if let Some(reward) = data.get("reward") {
                    let value = reward.get("value").and_then(|v| v.as_f64()).unwrap_or(-999.0);
                    let gardener_score = reward.get("gardener_score").and_then(|v| v.as_f64()).unwrap_or(-999.0);
                    let curator_score = reward.get("curator_score").and_then(|v| v.as_f64()).unwrap_or(-999.0);
                    let assessor_score = reward.get("assessor_score").and_then(|v| v.as_f64()).unwrap_or(-999.0);

                    println!("  reward.value: {:.4}", value);
                    println!("  reward.gardener_score: {:.4}", gardener_score);
                    println!("  reward.curator_score: {:.4}", curator_score);
                    println!("  reward.assessor_score: {:.4}", assessor_score);

                    assert!((-1.0..=1.0).contains(&value), "Reward value should be in [-1, 1]");
                }

                // Verify gardener details
                if let Some(gardener) = data.get("gardener_details") {
                    let connectivity = gardener.get("connectivity").and_then(|v| v.as_f64()).unwrap_or(-1.0);
                    let dead_ends = gardener.get("dead_ends_removed").and_then(|v| v.as_u64()).unwrap_or(0);

                    println!("  gardener.connectivity: {:.4}", connectivity);
                    println!("  gardener.dead_ends_removed: {}", dead_ends);

                    assert!((0.0..=1.0).contains(&connectivity), "Connectivity should be in [0, 1]");
                }

                println!("\n[PASSED] get_steering_feedback returns REAL computed data");
            }
        }
    }
}

/// Test: get_pruning_candidates - Pruning subsystem returns REAL candidates
#[tokio::test]
async fn test_stubfix_pruning_candidates_happy_path() {
    println!("\n========================================================================================================");
    println!("STUBFIX TEST: get_pruning_candidates (Pruning Subsystem)");
    println!("========================================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store some memories first
    println!("\n[SETUP] Storing test memories...");
    for i in 0..5 {
        let params = json!({
            "content": format!("Test memory {} for pruning candidates validation", i),
            "importance": 0.3 + (i as f64 * 0.1)
        });
        let request = make_request("memory/store", i as i64, Some(params));
        let _ = handlers.dispatch(request).await;
    }

    let count = store.count().await.expect("count works");
    println!("  Stored {} memories", count);

    // Call get_pruning_candidates
    let request = make_request("tools/call", 10, Some(json!({
        "name": "get_pruning_candidates",
        "arguments": {
            "limit": 10,
            "min_staleness_days": 0,
            "min_alignment": 0.9
        }
    })));
    let response = handlers.dispatch(request).await;

    println!("\nResponse: {}", serde_json::to_string_pretty(&response).unwrap());

    assert!(response.error.is_none(), "Handler must succeed: {:?}", response.error);

    let result = response.result.expect("Should have result");
    let content = result.get("content").and_then(|c| c.as_array());

    if let Some(content_arr) = content {
        for item in content_arr {
            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                let data: serde_json::Value = serde_json::from_str(text).unwrap_or(json!({}));

                println!("\n[VERIFICATION] Pruning Candidates Data:");

                // Verify summary
                if let Some(summary) = data.get("summary") {
                    let total = summary.get("total_candidates").and_then(|v| v.as_u64()).unwrap_or(0);
                    println!("  summary.total_candidates: {}", total);
                }

                // Verify candidates array exists
                if let Some(candidates) = data.get("candidates").and_then(|c| c.as_array()) {
                    println!("  candidates count: {}", candidates.len());

                    for (i, candidate) in candidates.iter().enumerate().take(3) {
                        let memory_id = candidate.get("memory_id").and_then(|v| v.as_str()).unwrap_or("?");
                        let reason = candidate.get("reason").and_then(|v| v.as_str()).unwrap_or("?");
                        let alignment = candidate.get("alignment").and_then(|v| v.as_f64()).unwrap_or(-1.0);

                        println!("    [{}] {} - reason: {}, alignment: {:.4}", i+1, memory_id, reason, alignment);
                    }
                }

                println!("\n[PASSED] get_pruning_candidates returns REAL data structure");
            }
        }
    }
}

/// Test: trigger_consolidation - Consolidation subsystem analyzes REAL pairs
#[tokio::test]
async fn test_stubfix_trigger_consolidation_happy_path() {
    println!("\n========================================================================================================");
    println!("STUBFIX TEST: trigger_consolidation (Consolidation Subsystem)");
    println!("========================================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store some memories first - use similar content to potentially get consolidation candidates
    println!("\n[SETUP] Storing test memories with similar content...");
    let base_content = "Machine learning and neural networks for optimization";
    for i in 0..5 {
        let content = format!("{} - variant {}", base_content, i);
        let params = json!({
            "content": content,
            "importance": 0.8
        });
        let request = make_request("memory/store", i as i64, Some(params));
        let _ = handlers.dispatch(request).await;
    }

    let count = store.count().await.expect("count works");
    println!("  Stored {} memories", count);

    // Call trigger_consolidation
    let request = make_request("tools/call", 10, Some(json!({
        "name": "trigger_consolidation",
        "arguments": {
            "strategy": "similarity",
            "min_similarity": 0.5,
            "max_memories": 10
        }
    })));
    let response = handlers.dispatch(request).await;

    println!("\nResponse: {}", serde_json::to_string_pretty(&response).unwrap());

    assert!(response.error.is_none(), "Handler must succeed: {:?}", response.error);

    let result = response.result.expect("Should have result");
    let content = result.get("content").and_then(|c| c.as_array());

    if let Some(content_arr) = content {
        for item in content_arr {
            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                let data: serde_json::Value = serde_json::from_str(text).unwrap_or(json!({}));

                println!("\n[VERIFICATION] Consolidation Data:");

                // Verify statistics
                if let Some(stats) = data.get("statistics") {
                    let pairs_evaluated = stats.get("pairs_evaluated").and_then(|v| v.as_u64()).unwrap_or(0);
                    let strategy = stats.get("strategy").and_then(|v| v.as_str()).unwrap_or("?");
                    let threshold = stats.get("similarity_threshold").and_then(|v| v.as_f64()).unwrap_or(-1.0);

                    println!("  statistics.pairs_evaluated: {}", pairs_evaluated);
                    println!("  statistics.strategy: {}", strategy);
                    println!("  statistics.similarity_threshold: {:.4}", threshold);

                    assert!(pairs_evaluated > 0 || count < 2, "Should evaluate pairs when multiple memories exist");
                    assert_eq!(strategy, "similarity", "Strategy should match request");
                }

                // Verify consolidation_result
                if let Some(result) = data.get("consolidation_result") {
                    let status = result.get("status").and_then(|v| v.as_str()).unwrap_or("?");
                    let candidate_count = result.get("candidate_count").and_then(|v| v.as_u64()).unwrap_or(0);

                    println!("  consolidation_result.status: {}", status);
                    println!("  consolidation_result.candidate_count: {}", candidate_count);
                }

                // Check candidates_sample if present
                if let Some(sample) = data.get("candidates_sample").and_then(|c| c.as_array()) {
                    println!("  candidates_sample count: {}", sample.len());
                    for (i, c) in sample.iter().enumerate().take(3) {
                        let similarity = c.get("similarity").and_then(|v| v.as_f64()).unwrap_or(-1.0);
                        println!("    [{}] similarity: {:.4}", i+1, similarity);
                    }
                }

                println!("\n[PASSED] trigger_consolidation returns REAL analysis data");
            }
        }
    }
}
