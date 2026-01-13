//! Phase 1: Manual MCP Testing - Store and Verify Synthetic Memories
//!
//! This test stores 5 synthetic memories via MCP handlers and VERIFIES
//! each fingerprint physically exists in the underlying RocksDB store.
//!
//! ## Test Plan
//! 1. Store 5 synthetic memories via handlers.dispatch()
//! 2. Extract fingerprintId from each response
//! 3. Call store.retrieve() to VERIFY each exists
//! 4. Record all fingerprint IDs for Phase 2 agent
//!
//! ## Critical: Full State Verification
//! - Do NOT trust response alone - VERIFY data exists in store
//! - Call store.count() before and after to confirm count increased
//! - Call store.retrieve() for each ID to confirm data is retrievable

use std::sync::Arc;

use serde_json::json;

use context_graph_core::traits::TeleologicalMemoryStore;

use crate::protocol::{JsonRpcId, JsonRpcRequest};

use super::create_test_handlers_with_rocksdb_store_access;

/// Synthetic test data - 5 memories covering different AI/ML domains
const SYNTHETIC_MEMORIES: [&str; 5] = [
    "Machine learning optimization techniques for neural networks including gradient descent, Adam optimizer, and learning rate scheduling",
    "Distributed systems architecture patterns for high availability including load balancing, replication, and consensus protocols",
    "Natural language processing with transformer models covering attention mechanisms, BERT, and GPT architectures",
    "Database indexing strategies for fast retrieval using B-trees, hash indexes, and covering indexes",
    "API design best practices for scalability including REST, GraphQL, rate limiting, and caching strategies",
];

fn make_request(
    method: &str,
    id: Option<JsonRpcId>,
    params: Option<serde_json::Value>,
) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id,
        method: method.to_string(),
        params,
    }
}

/// Helper: Check if fingerprint exists in store (via retrieve).
async fn exists_in_store(store: &Arc<dyn TeleologicalMemoryStore>, id: uuid::Uuid) -> bool {
    store
        .retrieve(id)
        .await
        .map(|opt| opt.is_some())
        .unwrap_or(false)
}

/// Phase 1: Store 5 synthetic memories and verify each physically exists.
///
/// This test:
/// 1. Gets count BEFORE storing (should be 0)
/// 2. Stores all 5 synthetic memories via MCP handlers
/// 3. Extracts fingerprintId from each response
/// 4. Verifies count AFTER storing (should be 5)
/// 5. Calls store.retrieve() for EACH fingerprint to prove it exists
/// 6. Prints evidence of stored data
#[tokio::test]
async fn phase1_manual_store_verify() {
    println!("\n================================================================================");
    println!("PHASE 1: Manual MCP Testing - Store and Verify Synthetic Memories");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // === BEFORE STATE ===
    let count_before = store.count().await.expect("count() should work");
    println!("\n[BEFORE] RocksDB store state:");
    println!("  - Fingerprint count: {}", count_before);
    assert_eq!(count_before, 0, "Store should be empty initially");

    // === STORE ALL 5 SYNTHETIC MEMORIES ===
    println!("\n[STORING] {} synthetic memories:", SYNTHETIC_MEMORIES.len());

    let mut stored_fingerprint_ids: Vec<String> = Vec::new();

    for (i, content) in SYNTHETIC_MEMORIES.iter().enumerate() {
        let params = json!({
            "content": content,
            "importance": 0.85
        });
        let request = make_request(
            "memory/store",
            Some(JsonRpcId::Number((i + 1) as i64)),
            Some(params),
        );
        let response = handlers.dispatch(request).await;

        // Verify no error
        if let Some(error) = &response.error {
            panic!(
                "Store {} failed with error: {} - {}",
                i + 1,
                error.code,
                error.message
            );
        }

        // Extract fingerprint ID
        let result = response.result.expect("Should have result");
        let fingerprint_id_str = result
            .get("fingerprintId")
            .expect("Should have fingerprintId")
            .as_str()
            .expect("Should be string")
            .to_string();

        stored_fingerprint_ids.push(fingerprint_id_str.clone());
        println!(
            "  [{}] Stored: {} -> {}",
            i + 1,
            &content[..50.min(content.len())],
            fingerprint_id_str
        );
    }

    // === AFTER STATE ===
    let count_after = store.count().await.expect("count() should work");
    println!("\n[AFTER] RocksDB store state:");
    println!("  - Fingerprint count: {}", count_after);
    println!("  - Expected count: {}", SYNTHETIC_MEMORIES.len());

    assert_eq!(
        count_after,
        SYNTHETIC_MEMORIES.len(),
        "Count must equal number of stored memories"
    );

    // === CRITICAL: VERIFY EACH FINGERPRINT EXISTS IN STORE ===
    println!("\n[VERIFICATION] Checking each fingerprint exists in store:");

    let mut all_verified = true;
    for (i, id_str) in stored_fingerprint_ids.iter().enumerate() {
        let fingerprint_id =
            uuid::Uuid::parse_str(id_str).expect("Should be valid UUID");

        // Check existence via helper
        let exists = exists_in_store(&store, fingerprint_id).await;

        // Also retrieve the actual fingerprint to prove it has data
        let retrieved = store
            .retrieve(fingerprint_id)
            .await
            .expect("retrieve() should work");

        if let Some(fp) = &retrieved {
            println!(
                "  [{}] {} EXISTS - theta={:.4}, access_count={}, hash={}...",
                i + 1,
                fingerprint_id,
                fp.theta_to_north_star,
                fp.access_count,
                hex::encode(&fp.content_hash[..8])
            );
        } else {
            println!("  [{}] {} MISSING - retrieve returned None!", i + 1, fingerprint_id);
            all_verified = false;
        }

        assert!(exists, "Fingerprint {} must exist in store", fingerprint_id);
        assert!(
            retrieved.is_some(),
            "Fingerprint {} must be retrievable",
            fingerprint_id
        );
    }

    // === SUMMARY ===
    println!("\n================================================================================");
    println!("PHASE 1 RESULTS:");
    println!("================================================================================");
    println!("  store_count_before: {}", count_before);
    println!("  store_count_after: {}", count_after);
    println!("  fingerprint_ids: {:?}", stored_fingerprint_ids);
    println!("  all_verified: {}", all_verified);

    // Print as JSON for Phase 2 agent
    let phase1_results = json!({
        "fingerprint_ids": stored_fingerprint_ids,
        "store_count_before": count_before,
        "store_count_after": count_after,
        "all_verified": all_verified,
        "synthetic_contents": SYNTHETIC_MEMORIES.iter().map(|s| s.to_string()).collect::<Vec<_>>()
    });
    println!("\n[PHASE 1 JSON RESULTS]:");
    println!("{}", serde_json::to_string_pretty(&phase1_results).unwrap());

    assert!(all_verified, "All fingerprints must be verified in store");
    println!("\n[PHASE 1 PASSED] All {} memories stored and verified!", SYNTHETIC_MEMORIES.len());
    println!("================================================================================\n");
}

/// Additional test: Verify retrieval via MCP handler matches store.
///
/// This confirms the MCP handler returns the same data as direct store access.
#[tokio::test]
async fn phase1_verify_mcp_retrieve_matches_store() {
    println!("\n================================================================================");
    println!("PHASE 1 BONUS: Verify MCP Retrieve Matches Direct Store Access");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store one memory
    let content = &SYNTHETIC_MEMORIES[0];
    let params = json!({
        "content": content,
        "importance": 0.9
    });
    let store_request = make_request("memory/store", Some(JsonRpcId::Number(1)), Some(params));
    let store_response = handlers.dispatch(store_request).await;

    let result = store_response.result.expect("Store must succeed");
    let fingerprint_id_str = result.get("fingerprintId").unwrap().as_str().unwrap();
    let fingerprint_id = uuid::Uuid::parse_str(fingerprint_id_str).unwrap();

    println!("\n[SETUP] Stored fingerprint: {}", fingerprint_id);

    // === DIRECT STORE ACCESS ===
    let direct_fp = store
        .retrieve(fingerprint_id)
        .await
        .expect("retrieve() should work")
        .expect("Must exist in store");

    println!("[DIRECT STORE] Retrieved fingerprint:");
    println!("  - ID: {}", direct_fp.id);
    println!("  - theta_to_north_star: {:.4}", direct_fp.theta_to_north_star);
    println!("  - content_hash: {}", hex::encode(direct_fp.content_hash));

    // === MCP HANDLER RETRIEVE ===
    let retrieve_params = json!({ "fingerprintId": fingerprint_id_str });
    let retrieve_request = make_request(
        "memory/retrieve",
        Some(JsonRpcId::Number(2)),
        Some(retrieve_params),
    );
    let retrieve_response = handlers.dispatch(retrieve_request).await;

    assert!(
        retrieve_response.error.is_none(),
        "Retrieve must succeed: {:?}",
        retrieve_response.error
    );

    let result = retrieve_response.result.expect("Should have result");
    let mcp_fp = result.get("fingerprint").expect("Must have fingerprint");

    println!("\n[MCP HANDLER] Retrieved fingerprint:");
    println!("  - ID: {}", mcp_fp.get("id").unwrap());
    println!(
        "  - thetaToNorthStar: {}",
        mcp_fp.get("thetaToNorthStar").unwrap()
    );
    println!(
        "  - contentHashHex: {}",
        mcp_fp.get("contentHashHex").unwrap()
    );

    // === VERIFY MATCH ===
    assert_eq!(
        mcp_fp.get("id").unwrap().as_str().unwrap(),
        direct_fp.id.to_string(),
        "IDs must match"
    );
    assert_eq!(
        mcp_fp.get("contentHashHex").unwrap().as_str().unwrap(),
        hex::encode(direct_fp.content_hash),
        "Content hashes must match"
    );

    println!("\n[VERIFICATION PASSED] MCP handler returns same data as direct store access");
    println!("================================================================================\n");
}
