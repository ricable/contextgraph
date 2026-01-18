//! SPEC-STUBFIX-002: get_pruning_candidates FSV Tests

use serde_json::json;

use crate::handlers::tests::{
    create_test_handlers_with_rocksdb_store_access, extract_mcp_tool_data, make_request,
};
use crate::protocol::JsonRpcId;

use super::helpers::create_test_fingerprint_with_age;

// ============================================================================
// SPEC-STUBFIX-002: get_pruning_candidates FSV Tests
// ============================================================================

/// FSV-PRUNING-001: Empty store returns empty candidates.
#[tokio::test]
#[ignore = "Uses tools not registered in PRD v6"]
async fn test_pruning_candidates_empty_store_returns_empty() {
    println!("\n================================================================================");
    println!("FSV-PRUNING-001: Empty Store Returns Empty Candidates");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    let count = store.count().await.expect("count works");
    assert_eq!(count, 0, "Store must be empty");

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_pruning_candidates",
            "arguments": {}
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

    let candidates = data.get("candidates").and_then(|v| v.as_array());
    let summary = data.get("summary").expect("Should have summary");
    let total = summary
        .get("total_candidates")
        .and_then(|v| v.as_u64())
        .unwrap_or(999);

    println!("[RESULT]");
    println!("  - candidates: {:?}", candidates.map(|c| c.len()));
    println!("  - total_candidates: {}", total);

    assert_eq!(total, 0, "Empty store should have 0 candidates");

    println!("\n[FSV-PRUNING-001 PASSED] Empty store returns empty candidates");
    println!("================================================================================\n");
}

/// FSV-PRUNING-002: Fresh data produces no candidates.
///
/// Stores fingerprints that are all:
/// - Recently created (< 30 days)
/// - High alignment (> 0.5)
/// - Non-orphan (access_count > 0)
#[tokio::test]
#[ignore = "Uses tools not registered in PRD v6"]
async fn test_pruning_candidates_fresh_data_no_candidates() {
    println!("\n================================================================================");
    println!("FSV-PRUNING-002: Fresh High-Quality Data Produces No Candidates");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store fresh, high-quality fingerprints
    let test_data = [
        ("fresh_memory_1", 0.8_f32, 5_u64, 0_i64),  // 0 days old
        ("fresh_memory_2", 0.75_f32, 3_u64, 1_i64), // 1 day old
        ("fresh_memory_3", 0.9_f32, 10_u64, 5_i64), // 5 days old
    ];

    println!("[SETUP] Storing {} fresh fingerprints:", test_data.len());
    for (content, theta, access_count, days_old) in test_data.iter() {
        let fp = create_test_fingerprint_with_age(content, *theta, *access_count, *days_old);
        let id = store.store(fp).await.expect("Store must succeed");
        println!(
            "  - {} (theta={:.2}, access={}, age={}d)",
            id, theta, access_count, days_old
        );
    }

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_pruning_candidates",
            "arguments": {
                "min_staleness_days": 30,
                "min_alignment": 0.4
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

    let summary = data.get("summary").expect("Should have summary");
    let total = summary
        .get("total_candidates")
        .and_then(|v| v.as_u64())
        .unwrap_or(999);

    println!("\n[RESULT]");
    println!("  - total_candidates: {}", total);

    // Fresh, high-quality data should not be pruning candidates
    assert_eq!(
        total, 0,
        "Fresh high-quality data should produce no candidates"
    );

    println!("\n[FSV-PRUNING-002 PASSED] Fresh data produces no candidates");
    println!("================================================================================\n");
}

/// FSV-PRUNING-003: Stale/low-alignment data produces candidates.
///
/// Stores mix of:
/// - Old, low-alignment fingerprints (should be candidates)
/// - Fresh, high-alignment fingerprints (should NOT be candidates)
#[tokio::test]
#[ignore = "Uses tools not registered in PRD v6"]
async fn test_pruning_candidates_stale_data_produces_candidates() {
    println!("\n================================================================================");
    println!("FSV-PRUNING-003: Stale/Low-Alignment Data Produces Candidates");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store mix of fingerprints
    let test_data = [
        // Should be candidates (stale OR low alignment)
        ("stale_memory_1", 0.2_f32, 0_u64, 100_i64), // Old + low alignment + orphan
        ("stale_memory_2", 0.3_f32, 0_u64, 120_i64), // Old + low alignment + orphan
        ("low_align_memory", 0.1_f32, 1_u64, 50_i64), // Low alignment
        // Should NOT be candidates (fresh + high alignment)
        ("fresh_memory_1", 0.8_f32, 5_u64, 5_i64),
        ("fresh_memory_2", 0.9_f32, 10_u64, 2_i64),
    ];

    println!("[SETUP] Storing {} fingerprints:", test_data.len());
    let mut stored_ids = Vec::new();
    for (content, theta, access_count, days_old) in test_data.iter() {
        let fp = create_test_fingerprint_with_age(content, *theta, *access_count, *days_old);
        let id = store.store(fp).await.expect("Store must succeed");
        stored_ids.push(id);
        let is_candidate = *theta < 0.4 || *days_old > 90 || *access_count == 0;
        println!(
            "  - {} (theta={:.2}, access={}, age={}d) {}",
            id,
            theta,
            access_count,
            days_old,
            if is_candidate { "CANDIDATE" } else { "KEEP" }
        );
    }

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_pruning_candidates",
            "arguments": {
                "min_staleness_days": 90,
                "min_alignment": 0.4,
                "limit": 10
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

    let candidates = data
        .get("candidates")
        .and_then(|v| v.as_array())
        .expect("Should have candidates array");
    let summary = data.get("summary").expect("Should have summary");
    let total = summary
        .get("total_candidates")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    println!("\n[RESULT]");
    println!("  - total_candidates: {}", total);
    println!("  - candidates returned: {}", candidates.len());

    for (i, candidate) in candidates.iter().enumerate() {
        let memory_id = candidate
            .get("memory_id")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let reason = candidate
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let alignment = candidate
            .get("alignment")
            .and_then(|v| v.as_f64())
            .unwrap_or(-1.0);
        let age_days = candidate
            .get("age_days")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        println!(
            "    [{}] {} - reason: {}, alignment: {:.2}, age: {}d",
            i + 1,
            memory_id,
            reason,
            alignment,
            age_days
        );
    }

    // Should have at least some candidates (the stale/low-alignment ones)
    assert!(
        total >= 1,
        "Should have at least 1 pruning candidate, got {}",
        total
    );

    // Verify each returned candidate has a valid reason
    for candidate in candidates {
        let reason = candidate
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(!reason.is_empty(), "Each candidate must have a reason");
    }

    println!(
        "\n[FSV-PRUNING-003 PASSED] Stale/low-alignment data produces candidates with reasons"
    );
    println!("================================================================================\n");
}
