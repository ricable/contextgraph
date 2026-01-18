//! Full State Verification - RocksDB Direct Inspection Tests

use serde_json::json;

use crate::handlers::tests::{
    create_test_handlers_with_rocksdb_store_access, extract_mcp_tool_data, make_request,
};
use crate::protocol::JsonRpcId;

use super::helpers::create_test_fingerprint;

// ============================================================================
// Full State Verification - RocksDB Direct Inspection Tests
// ============================================================================

/// FSV-ROCKSDB-STEERING-001: Verify steering data matches direct RocksDB state.
///
/// This test directly inspects the store to verify:
/// - Count matches what handler reports
/// - Each fingerprint's theta and access_count are correctly read
#[tokio::test]
#[ignore = "Uses tools not registered in PRD v6"]
async fn test_fsv_rocksdb_steering_data_matches_store() {
    println!("\n================================================================================");
    println!("FSV-ROCKSDB-STEERING-001: Verify Steering Data Matches RocksDB State");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store fingerprints with known values
    let test_data = [
        ("known_fp_1", 0.8_f32, 5_u64), // aligned, accessed
        ("known_fp_2", 0.3_f32, 0_u64), // not aligned, orphan
        ("known_fp_3", 0.6_f32, 3_u64), // aligned, accessed
    ];

    let mut stored_ids = Vec::new();
    println!("[SETUP] Storing fingerprints with known values:");
    for (content, theta, access_count) in test_data.iter() {
        let fp = create_test_fingerprint(content, *theta, *access_count);
        let id = store.store(fp).await.expect("Store must succeed");
        stored_ids.push(id);
        println!("  - {} (theta={:.2}, access={})", id, theta, access_count);
    }

    // DIRECT ROCKSDB VERIFICATION: Read each fingerprint directly
    println!("\n[DIRECT ROCKSDB VERIFICATION] Reading fingerprints from store:");
    let mut aligned_count = 0;
    let mut orphan_count = 0;

    for id in stored_ids.iter() {
        let fp = store
            .retrieve(*id)
            .await
            .expect("retrieve works")
            .expect("fingerprint exists");

        println!(
            "  - {} theta={:.4}, access={}, aligned={}, orphan={}",
            id,
            fp.alignment_score,
            fp.access_count,
            fp.alignment_score >= 0.5,
            fp.access_count == 0
        );

        if fp.alignment_score >= 0.5 {
            aligned_count += 1;
        }
        if fp.access_count == 0 {
            orphan_count += 1;
        }
    }

    let expected_connectivity = aligned_count as f64 / test_data.len() as f64;
    println!("\n[EXPECTED from direct inspection]");
    println!("  - aligned_count: {}", aligned_count);
    println!("  - orphan_count: {}", orphan_count);
    println!("  - connectivity: {:.4}", expected_connectivity);

    // Call handler and verify it matches
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_steering_feedback",
            "arguments": {}
        })),
    );
    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let gardener = data
        .get("gardener_details")
        .expect("Should have gardener_details");
    let handler_connectivity = gardener
        .get("connectivity")
        .and_then(|v| v.as_f64())
        .unwrap_or(-1.0);
    let handler_orphans = gardener
        .get("dead_ends_removed")
        .and_then(|v| v.as_u64())
        .unwrap_or(999);

    println!("\n[HANDLER RESULT]");
    println!("  - connectivity: {:.4}", handler_connectivity);
    println!("  - dead_ends_removed: {}", handler_orphans);

    // Verify handler matches direct inspection
    let tolerance = 0.01;
    assert!(
        (handler_connectivity - expected_connectivity).abs() < tolerance,
        "Handler connectivity {:.4} must match direct inspection {:.4}",
        handler_connectivity,
        expected_connectivity
    );
    assert_eq!(
        handler_orphans, orphan_count as u64,
        "Handler orphan count must match direct inspection"
    );

    println!("\n[FSV-ROCKSDB-STEERING-001 PASSED] Handler data matches RocksDB state");
    println!("================================================================================\n");
}

/// FSV-ROCKSDB-PRUNING-001: Verify pruning candidates match direct RocksDB state.
#[tokio::test]
#[ignore = "Uses tools not registered in PRD v6"]
async fn test_fsv_rocksdb_pruning_candidates_match_store() {
    println!("\n================================================================================");
    println!("FSV-ROCKSDB-PRUNING-001: Verify Pruning Candidates Match RocksDB State");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store fingerprints - some should be candidates, some not
    let test_data = [
        ("orphan_1", 0.2_f32, 0_u64),  // Candidate: orphan + low alignment
        ("orphan_2", 0.15_f32, 0_u64), // Candidate: orphan + low alignment
        ("good_1", 0.9_f32, 10_u64),   // Not candidate: high alignment + accessed
    ];

    let mut candidate_ids = Vec::new();
    println!("[SETUP] Storing fingerprints:");
    for (content, theta, access_count) in test_data.iter() {
        let fp = create_test_fingerprint(content, *theta, *access_count);
        let id = fp.id;
        store.store(fp).await.expect("Store must succeed");
        // Orphans with low alignment should be candidates
        if *access_count == 0 {
            candidate_ids.push(id);
        }
        println!(
            "  - {} (theta={:.2}, access={}) {}",
            id,
            theta,
            access_count,
            if *access_count == 0 {
                "CANDIDATE"
            } else {
                "KEEP"
            }
        );
    }

    // DIRECT VERIFICATION: Each candidate ID should exist in store
    println!("\n[DIRECT ROCKSDB VERIFICATION] Confirming candidates exist:");
    for id in candidate_ids.iter() {
        let exists = store.retrieve(*id).await.expect("retrieve works").is_some();
        println!("  - {} exists: {}", id, exists);
        assert!(exists, "Candidate must exist in store");
    }

    // Call handler
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_pruning_candidates",
            "arguments": {
                "min_staleness_days": 0,
                "min_alignment": 0.5,
                "limit": 10
            }
        })),
    );
    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let candidates = data
        .get("candidates")
        .and_then(|v| v.as_array())
        .expect("Should have candidates");

    println!(
        "\n[HANDLER RESULT] {} candidates returned:",
        candidates.len()
    );
    let mut handler_ids: Vec<uuid::Uuid> = Vec::new();
    for c in candidates {
        let id_str = c.get("memory_id").and_then(|v| v.as_str()).unwrap_or("?");
        let reason = c.get("reason").and_then(|v| v.as_str()).unwrap_or("?");
        println!("  - {} (reason: {})", id_str, reason);
        if let Ok(id) = uuid::Uuid::parse_str(id_str) {
            handler_ids.push(id);
        }
    }

    // Verify handler returned the expected candidates
    // (handler may return additional candidates due to PruningService logic)
    for expected_id in candidate_ids.iter() {
        // The handler should return orphans
        let found = handler_ids.contains(expected_id);
        println!(
            "\n[VERIFY] Expected candidate {} found in handler result: {}",
            expected_id, found
        );
        assert!(
            found,
            "Expected candidate {} must be in handler result",
            expected_id
        );
    }

    println!("\n[FSV-ROCKSDB-PRUNING-001 PASSED] Pruning candidates match RocksDB state");
    println!("================================================================================\n");
}
