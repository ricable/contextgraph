//! Edge Case Tests
//!
//! Tests for boundary conditions: minimal, maximum size, and concurrent access.

use std::sync::Arc;

use context_graph_core::traits::TeleologicalMemoryStore;
use context_graph_core::types::fingerprint::SparseVector;
use context_graph_storage::teleological::{
    deserialize_teleological_fingerprint, fingerprint_key, CF_FINGERPRINTS,
};
use tempfile::TempDir;
use uuid::Uuid;

use crate::helpers::{
    create_test_store, generate_real_sparse_vector, generate_real_teleological_fingerprint,
    generate_real_unit_vector,
};

/// Test 4: EDGE CASE - Empty/Minimal Fingerprint
///
/// Tests the boundary condition of minimal valid data.
#[tokio::test]
async fn test_edge_case_minimal_fingerprint() {
    println!("\n================================================================================");
    println!("EDGE CASE TEST: Minimal Valid Fingerprint");
    println!("================================================================================\n");

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let store = create_test_store(&temp_dir);

    // Create minimal fingerprint with minimum required dimensions
    let mut fingerprint = generate_real_teleological_fingerprint(Uuid::new_v4());
    let id = fingerprint.id; // Use the fingerprint's actual ID

    // Test with minimal sparse vectors (1 element each)
    fingerprint.semantic.e6_sparse =
        SparseVector::new(vec![0], vec![1.0]).expect("Failed to create minimal sparse");
    fingerprint.semantic.e13_splade =
        SparseVector::new(vec![0], vec![1.0]).expect("Failed to create minimal sparse");

    println!("[1] Minimal sparse vectors:");
    println!(
        "    E6 sparse nnz: {}",
        fingerprint.semantic.e6_sparse.nnz()
    );
    println!(
        "    E13 SPLADE nnz: {}",
        fingerprint.semantic.e13_splade.nnz()
    );

    store
        .store(fingerprint)
        .await
        .expect("Failed to store minimal fingerprint");

    // Verify physical storage
    let key = fingerprint_key(&id);
    let raw = store
        .get_raw_bytes(CF_FINGERPRINTS, &key)
        .expect("Failed to read");

    assert!(raw.is_some(), "Minimal fingerprint not stored!");
    let bytes = raw.unwrap();

    println!("[2] Stored {} bytes", bytes.len());

    let retrieved = deserialize_teleological_fingerprint(&bytes)
        .expect("Failed to deserialize fingerprint");

    println!("[3] Retrieved sparse vectors:");
    println!("    E6 sparse nnz: {}", retrieved.semantic.e6_sparse.nnz());
    println!(
        "    E13 SPLADE nnz: {}",
        retrieved.semantic.e13_splade.nnz()
    );

    assert_eq!(retrieved.semantic.e6_sparse.nnz(), 1);
    assert_eq!(retrieved.semantic.e13_splade.nnz(), 1);

    println!("\n[PASS] Minimal fingerprint edge case successful");
    println!("================================================================================\n");
}

/// Test 5: EDGE CASE - Maximum Size Fingerprint
///
/// Tests boundary with maximum realistic sparse vector sizes.
#[tokio::test]
async fn test_edge_case_maximum_size_fingerprint() {
    println!("\n================================================================================");
    println!("EDGE CASE TEST: Maximum Size Fingerprint");
    println!("================================================================================\n");

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let store = create_test_store(&temp_dir);

    let mut fingerprint = generate_real_teleological_fingerprint(Uuid::new_v4());
    let id = fingerprint.id; // Use the fingerprint's actual ID

    // Maximum sparse vectors (within 150KB serialization limit)
    // E6 sparse: 2000 elements = 2000 * 6 bytes = ~12KB
    fingerprint.semantic.e6_sparse = generate_real_sparse_vector(2000);
    // E13 SPLADE: 1500 elements = 1500 * 6 bytes = ~9KB
    fingerprint.semantic.e13_splade = generate_real_sparse_vector(1500);

    // Maximum late interaction tokens (100 * 128 * 4 = ~51KB)
    // Total with base (~63KB) + sparse (~21KB) + late (~51KB) = 135KB < 150KB limit
    fingerprint.semantic.e12_late_interaction = vec![generate_real_unit_vector(128); 100];

    println!("[1] Maximum size components (within 150KB limit):");
    println!(
        "    E6 sparse nnz: {}",
        fingerprint.semantic.e6_sparse.nnz()
    );
    println!(
        "    E13 SPLADE nnz: {}",
        fingerprint.semantic.e13_splade.nnz()
    );
    println!(
        "    E12 tokens: {}",
        fingerprint.semantic.e12_late_interaction.len()
    );

    let start = std::time::Instant::now();
    store
        .store(fingerprint.clone())
        .await
        .expect("Failed to store max fingerprint");
    let store_time = start.elapsed();

    println!("[2] Store time: {:?}", store_time);

    // Physical verification
    let key = fingerprint_key(&id);
    let raw = store
        .get_raw_bytes(CF_FINGERPRINTS, &key)
        .expect("Failed to read");

    assert!(raw.is_some(), "Max fingerprint not stored!");
    let bytes = raw.unwrap();

    println!(
        "[3] Stored size: {} bytes ({:.2} KB)",
        bytes.len(),
        bytes.len() as f64 / 1024.0
    );

    let start = std::time::Instant::now();
    let retrieved = deserialize_teleological_fingerprint(&bytes)
        .expect("Failed to deserialize fingerprint");
    let deser_time = start.elapsed();

    println!("[4] Deserialize time: {:?}", deser_time);
    println!("[5] Retrieved components:");
    println!("    E6 sparse nnz: {}", retrieved.semantic.e6_sparse.nnz());
    println!(
        "    E13 SPLADE nnz: {}",
        retrieved.semantic.e13_splade.nnz()
    );
    println!(
        "    E12 tokens: {}",
        retrieved.semantic.e12_late_interaction.len()
    );

    assert_eq!(retrieved.semantic.e6_sparse.nnz(), 2000);
    assert_eq!(retrieved.semantic.e13_splade.nnz(), 1500);
    assert_eq!(retrieved.semantic.e12_late_interaction.len(), 100);

    println!("\n[PASS] Maximum size fingerprint edge case successful");
    println!("================================================================================\n");
}

/// Test 6: EDGE CASE - Concurrent Write/Read Stress
///
/// Tests concurrent access patterns with physical verification.
#[tokio::test]
async fn test_edge_case_concurrent_access() {
    println!("\n================================================================================");
    println!("EDGE CASE TEST: Concurrent Access");
    println!("================================================================================\n");

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let store = Arc::new(create_test_store(&temp_dir));

    let num_tasks = 8;
    let writes_per_task = 10;

    println!(
        "[1] Spawning {} tasks, {} writes each",
        num_tasks, writes_per_task
    );

    let mut handles = Vec::new();

    for task_id in 0..num_tasks {
        let store = Arc::clone(&store);
        handles.push(tokio::spawn(async move {
            let mut ids = Vec::new();
            for i in 0..writes_per_task {
                let fingerprint = generate_real_teleological_fingerprint(Uuid::new_v4());
                let id = fingerprint.id; // Use the fingerprint's actual ID
                store
                    .store(fingerprint)
                    .await
                    .unwrap_or_else(|e| panic!("Task {} write {} failed: {:?}", task_id, i, e));
                ids.push(id);
            }
            ids
        }));
    }

    let results: Vec<_> = futures::future::join_all(handles).await;
    let all_ids: Vec<Uuid> = results
        .into_iter()
        .flat_map(|r| r.expect("Task panicked"))
        .collect();

    println!("[2] Total writes: {}", all_ids.len());

    // Physical verification of ALL writes
    let mut verified = 0;
    for id in &all_ids {
        let key = fingerprint_key(id);
        let raw = store
            .get_raw_bytes(CF_FINGERPRINTS, &key)
            .expect("Read failed");
        assert!(raw.is_some(), "Missing fingerprint: {}", id);
        verified += 1;
    }

    println!("[3] Physically verified {} fingerprints", verified);
    assert_eq!(verified, num_tasks * writes_per_task);

    println!("\n[PASS] Concurrent access edge case successful");
    println!("================================================================================\n");
}
