//! SPEC-STUBFIX-001: get_steering_feedback FSV Tests

use serde_json::json;

use crate::handlers::tests::{
    create_test_handlers_with_rocksdb_store_access, extract_mcp_tool_data, make_request,
};
use crate::protocol::JsonRpcId;

use super::helpers::create_test_fingerprint;

// ============================================================================
// SPEC-STUBFIX-001: get_steering_feedback FSV Tests
// ============================================================================

/// FSV-STEERING-001: Empty store returns zero metrics.
///
/// Verifies get_steering_feedback with empty store returns:
/// - orphan_count = 0
/// - connectivity = 0.0 (no nodes to be connected)
#[tokio::test]
#[ignore = "Uses tools not registered in PRD v6"]
async fn test_steering_feedback_empty_store_returns_zero_metrics() {
    println!("\n================================================================================");
    println!("FSV-STEERING-001: Empty Store Returns Zero Metrics");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Verify store is empty
    let count = store.count().await.expect("count works");
    assert_eq!(count, 0, "Store must be empty");
    println!("[BEFORE] Store count: {}", count);

    // Call get_steering_feedback
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_steering_feedback",
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

    // Verify gardener details
    let gardener = data
        .get("gardener_details")
        .expect("Should have gardener_details");
    let connectivity = gardener
        .get("connectivity")
        .and_then(|v| v.as_f64())
        .unwrap_or(-1.0);
    let dead_ends_removed = gardener
        .get("dead_ends_removed")
        .and_then(|v| v.as_u64())
        .unwrap_or(999);

    println!("[RESULT] gardener_details:");
    println!("  - connectivity: {}", connectivity);
    println!("  - dead_ends_removed: {}", dead_ends_removed);

    // With empty store: connectivity should be 0.0 (no aligned nodes)
    assert!(
        (0.0..=0.01).contains(&connectivity),
        "Empty store connectivity should be ~0.0, got {}",
        connectivity
    );

    println!("\n[FSV-STEERING-001 PASSED] Empty store returns zero connectivity");
    println!("================================================================================\n");
}

/// FSV-STEERING-002: All orphans produces low connectivity.
///
/// Stores N fingerprints with access_count=0 (orphan proxy) and theta < 0.5.
/// Verifies:
/// - Handler reads all fingerprints
/// - orphan_count matches stored count
/// - connectivity is low
#[tokio::test]
#[ignore = "Uses tools not registered in PRD v6"]
async fn test_steering_feedback_all_orphans_low_connectivity() {
    println!("\n================================================================================");
    println!("FSV-STEERING-002: All Orphans Produces Low Connectivity");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store 5 fingerprints with access_count=0 (orphan proxy) and low alignment
    let test_data = [
        ("orphan_memory_1", 0.2_f32, 0_u64),
        ("orphan_memory_2", 0.3_f32, 0_u64),
        ("orphan_memory_3", 0.1_f32, 0_u64),
        ("orphan_memory_4", 0.25_f32, 0_u64),
        ("orphan_memory_5", 0.15_f32, 0_u64),
    ];

    println!("[SETUP] Storing {} orphan fingerprints:", test_data.len());
    for (content, theta, access_count) in test_data.iter() {
        let fp = create_test_fingerprint(content, *theta, *access_count);
        let id = store.store(fp).await.expect("Store must succeed");
        println!("  - {} (theta={}, access={})", id, theta, access_count);
    }

    let count = store.count().await.expect("count works");
    assert_eq!(count, test_data.len(), "All fingerprints must be stored");

    // Call get_steering_feedback
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_steering_feedback",
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

    let gardener = data
        .get("gardener_details")
        .expect("Should have gardener_details");
    let connectivity = gardener
        .get("connectivity")
        .and_then(|v| v.as_f64())
        .unwrap_or(-1.0);
    let dead_ends_removed = gardener
        .get("dead_ends_removed")
        .and_then(|v| v.as_u64())
        .unwrap_or(999);

    println!("\n[RESULT] gardener_details:");
    println!("  - connectivity: {}", connectivity);
    println!(
        "  - dead_ends_removed (orphan_count): {}",
        dead_ends_removed
    );

    // All orphans with theta < 0.5 means connectivity = 0 (none aligned to strategic goal)
    assert!(
        (0.0..0.1).contains(&connectivity),
        "All orphans should have low connectivity, got {}",
        connectivity
    );

    // dead_ends_removed should match orphan_count (all 5 have access_count=0)
    assert_eq!(
        dead_ends_removed, 5,
        "dead_ends_removed should equal orphan count"
    );

    println!("\n[FSV-STEERING-002 PASSED] All orphans produces low connectivity and correct orphan count");
    println!("================================================================================\n");
}

/// FSV-STEERING-003: Mixed data produces accurate metrics.
///
/// Stores fingerprints with varied alignment and access patterns:
/// - 3 aligned (theta >= 0.5) with access_count > 0
/// - 2 orphans (access_count = 0) with low alignment
///
/// Verifies connectivity reflects aligned ratio.
#[tokio::test]
#[ignore = "Uses tools not registered in PRD v6"]
async fn test_steering_feedback_mixed_data_accurate_metrics() {
    println!("\n================================================================================");
    println!("FSV-STEERING-003: Mixed Data Produces Accurate Metrics");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store mixed fingerprints
    let test_data = [
        // Aligned and accessed (connected)
        ("aligned_memory_1", 0.8_f32, 5_u64),
        ("aligned_memory_2", 0.75_f32, 3_u64),
        ("aligned_memory_3", 0.6_f32, 2_u64),
        // Orphans (low alignment, never accessed)
        ("orphan_memory_1", 0.2_f32, 0_u64),
        ("orphan_memory_2", 0.3_f32, 0_u64),
    ];

    println!("[SETUP] Storing {} fingerprints:", test_data.len());
    for (content, theta, access_count) in test_data.iter() {
        let fp = create_test_fingerprint(content, *theta, *access_count);
        let id = store.store(fp).await.expect("Store must succeed");
        println!(
            "  - {} (theta={:.2}, access={}) {}",
            id,
            theta,
            access_count,
            if *theta >= 0.5 {
                "ALIGNED"
            } else {
                "NOT ALIGNED"
            }
        );
    }

    // Expected metrics:
    // - 3 out of 5 are aligned (theta >= 0.5) -> connectivity = 0.6
    // - 2 orphans (access_count = 0)
    let expected_aligned = 3;
    let expected_orphans = 2;
    let expected_connectivity = expected_aligned as f64 / test_data.len() as f64;

    println!("\n[EXPECTED]");
    println!("  - Aligned nodes: {}", expected_aligned);
    println!("  - Orphan nodes: {}", expected_orphans);
    println!("  - Connectivity: {:.2}", expected_connectivity);

    // Call get_steering_feedback
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_steering_feedback",
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

    let gardener = data
        .get("gardener_details")
        .expect("Should have gardener_details");
    let connectivity = gardener
        .get("connectivity")
        .and_then(|v| v.as_f64())
        .unwrap_or(-1.0);
    let dead_ends_removed = gardener
        .get("dead_ends_removed")
        .and_then(|v| v.as_u64())
        .unwrap_or(999);

    println!("\n[RESULT] gardener_details:");
    println!("  - connectivity: {:.4}", connectivity);
    println!(
        "  - dead_ends_removed (orphan_count): {}",
        dead_ends_removed
    );

    // Verify connectivity matches expected (with tolerance for floating point)
    let tolerance = 0.05;
    assert!(
        (connectivity - expected_connectivity).abs() < tolerance,
        "Connectivity should be ~{:.2}, got {:.4}",
        expected_connectivity,
        connectivity
    );

    // Verify orphan count
    assert_eq!(
        dead_ends_removed, expected_orphans as u64,
        "Orphan count should match"
    );

    println!(
        "\n[FSV-STEERING-003 PASSED] Mixed data produces accurate connectivity and orphan count"
    );
    println!("================================================================================\n");
}
