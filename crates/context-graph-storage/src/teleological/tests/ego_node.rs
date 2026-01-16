//! EGO_NODE persistence tests.
//!
//! TASK-GWT-P1-001: EGO_NODE Persistence Tests
//!
//! CRITICAL: Uses #[tokio::test] to prevent zombie runtime threads.
//! DO NOT use tokio::runtime::Runtime::new() in tests.

use crate::teleological::{RocksDbTeleologicalStore, serialization};
use context_graph_core::gwt::ego_node::{PurposeSnapshot as EgoPurposeSnapshot, SelfEgoNode};
use context_graph_core::traits::TeleologicalMemoryStore;
use chrono::Utc;
use uuid::Uuid;

/// Create a REAL SelfEgoNode for testing.
/// Uses actual struct construction (not mocks).
fn create_real_ego_node() -> SelfEgoNode {
    let now = Utc::now();
    let mut purpose_vector = [0.0f32; 13];
    for (i, val) in purpose_vector.iter_mut().enumerate() {
        *val = (i as f32 + 1.0) * 0.05; // 0.05, 0.10, 0.15, ...
    }

    SelfEgoNode {
        id: Uuid::new_v4(),
        fingerprint: None, // No initial fingerprint
        purpose_vector,
        coherence_with_actions: 0.85,
        identity_trajectory: vec![
            EgoPurposeSnapshot {
                timestamp: now,
                vector: purpose_vector,
                context: "initial_creation".to_string(),
            },
        ],
        last_updated: now,
    }
}

/// Create a SelfEgoNode with identity trajectory for testing.
fn create_ego_node_with_trajectory(snapshot_count: usize) -> SelfEgoNode {
    let now = Utc::now();
    let purpose_vector = [0.75f32; 13];
    let mut trajectory = Vec::with_capacity(snapshot_count);

    for i in 0..snapshot_count {
        let mut snapshot_pv = purpose_vector;
        snapshot_pv[0] = (i as f32) * 0.01; // Vary first dimension
        trajectory.push(EgoPurposeSnapshot {
            timestamp: now - chrono::Duration::hours(i as i64),
            vector: snapshot_pv,
            context: format!("snapshot_{}", i),
        });
    }

    SelfEgoNode {
        id: Uuid::new_v4(),
        fingerprint: None,
        purpose_vector,
        coherence_with_actions: 0.92,
        identity_trajectory: trajectory,
        last_updated: now,
    }
}

#[test]
fn test_serialize_ego_node_roundtrip() {
    println!("=== TEST: SelfEgoNode serialization round-trip (TASK-GWT-P1-001) ===");

    let original = create_real_ego_node();
    println!("BEFORE: Created SelfEgoNode with ID: {}", original.id);
    println!("  - purpose_vector[0..3]: {:?}", &original.purpose_vector[..3]);
    println!("  - coherence_with_actions: {:.4}", original.coherence_with_actions);
    println!("  - identity_trajectory length: {}", original.identity_trajectory.len());

    let serialized = serialization::serialize_ego_node(&original);
    println!("SERIALIZED: {} bytes", serialized.len());
    println!("  - Version byte: {}", serialized[0]);
    println!("  - Payload: {} bytes", serialized.len() - 1);

    let deserialized = serialization::deserialize_ego_node(&serialized);
    println!("AFTER: Deserialized SelfEgoNode ID: {}", deserialized.id);
    println!("  - purpose_vector[0..3]: {:?}", &deserialized.purpose_vector[..3]);
    println!("  - coherence_with_actions: {:.4}", deserialized.coherence_with_actions);

    // Verify all fields match
    assert_eq!(original.id, deserialized.id, "ID mismatch");
    assert_eq!(original.fingerprint.is_none(), deserialized.fingerprint.is_none(), "Fingerprint mismatch");
    for i in 0..13 {
        assert!(
            (original.purpose_vector[i] - deserialized.purpose_vector[i]).abs() < 1e-6,
            "purpose_vector[{}] mismatch", i
        );
    }
    assert!(
        (original.coherence_with_actions - deserialized.coherence_with_actions).abs() < 1e-6,
        "coherence_with_actions mismatch"
    );
    assert_eq!(
        original.identity_trajectory.len(),
        deserialized.identity_trajectory.len(),
        "identity_trajectory length mismatch"
    );

    println!("RESULT: PASS - SelfEgoNode round-trip preserved all fields");
}

#[test]
fn test_serialize_ego_node_with_large_trajectory() {
    println!("=== TEST: SelfEgoNode with 100 identity snapshots ===");

    let original = create_ego_node_with_trajectory(100);
    println!("BEFORE: Created SelfEgoNode with {} snapshots", original.identity_trajectory.len());

    let serialized = serialization::serialize_ego_node(&original);
    println!("SERIALIZED: {} bytes ({:.2}KB)", serialized.len(), serialized.len() as f64 / 1024.0);

    let deserialized = serialization::deserialize_ego_node(&serialized);
    assert_eq!(
        original.identity_trajectory.len(),
        deserialized.identity_trajectory.len(),
        "Trajectory length mismatch"
    );

    // Verify context strings
    for (i, (orig, deser)) in original.identity_trajectory.iter()
        .zip(deserialized.identity_trajectory.iter())
        .enumerate() {
        assert_eq!(orig.context, deser.context, "Context mismatch at snapshot {}", i);
    }

    println!("RESULT: PASS - Large trajectory (100 snapshots) preserved");
}

#[test]
fn test_ego_node_version_constant() {
    println!("=== TEST: EGO_NODE_VERSION constant ===");

    assert_eq!(serialization::EGO_NODE_VERSION, 1, "Version should be 1");

    let ego = create_real_ego_node();
    let serialized = serialization::serialize_ego_node(&ego);
    assert_eq!(serialized[0], serialization::EGO_NODE_VERSION, "First byte should be version");

    println!("RESULT: PASS - EGO_NODE_VERSION is 1");
}

#[tokio::test]
async fn test_ego_node_save_load_roundtrip() {
    println!("=== TEST: save_ego_node / load_ego_node round-trip (TASK-GWT-P1-001) ===");

    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbTeleologicalStore::open(temp_dir.path())
        .expect("Failed to open store");

    // Initially no ego node
    let initial = store.load_ego_node().await.expect("Should query");
    assert!(initial.is_none(), "Initially no ego node should exist");
    println!("BEFORE: No ego node exists");

    // Create and save
    let original = create_real_ego_node();
    let original_id = original.id;
    println!("SAVING: SelfEgoNode id={}", original_id);
    println!("  - purpose_vector[0..3]: {:?}", &original.purpose_vector[..3]);
    println!("  - coherence: {:.4}", original.coherence_with_actions);

    store.save_ego_node(&original).await.expect("Should save ego node");

    // Load and verify
    let loaded = store.load_ego_node().await
        .expect("Should load")
        .expect("Ego node should exist");

    println!("AFTER: Loaded SelfEgoNode id={}", loaded.id);
    println!("  - purpose_vector[0..3]: {:?}", &loaded.purpose_vector[..3]);
    println!("  - coherence: {:.4}", loaded.coherence_with_actions);

    assert_eq!(original_id, loaded.id, "ID mismatch");
    for i in 0..13 {
        assert!(
            (original.purpose_vector[i] - loaded.purpose_vector[i]).abs() < 1e-6,
            "purpose_vector[{}] mismatch", i
        );
    }
    assert_eq!(
        original.identity_trajectory.len(),
        loaded.identity_trajectory.len(),
        "Trajectory length mismatch"
    );

    println!("RESULT: PASS - save/load ego node round-trip successful");
}

#[tokio::test]
async fn test_ego_node_persistence_across_reopen() {
    println!("=== TEST: Ego node persists across store close/reopen (TASK-GWT-P1-001) ===");

    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().to_owned();

    // Create ego node first so we can capture its properties
    let ego = create_ego_node_with_trajectory(5);
    let original_id = ego.id;
    let original_coherence = ego.coherence_with_actions;
    let original_trajectory_len = ego.identity_trajectory.len();

    // Step 1: Open store, save ego node, close
    {
        let store = RocksDbTeleologicalStore::open(&db_path)
            .expect("Failed to open store (first open)");

        println!("STEP 1: Saving ego node id={} with {} snapshots", original_id, original_trajectory_len);

        store.save_ego_node(&ego).await.expect("Should save ego node");
        store.flush().await.expect("Should flush");

        println!("STEP 1: Closing store...");
    } // Store dropped here

    // Step 2: Reopen store, load ego node, verify
    {
        println!("STEP 2: Reopening store...");
        let store = RocksDbTeleologicalStore::open(&db_path)
            .expect("Failed to open store (second open)");

        let loaded = store.load_ego_node().await
            .expect("Should load")
            .expect("Ego node should persist");

        println!("STEP 2: Loaded ego node id={} with {} snapshots", loaded.id, loaded.identity_trajectory.len());

        assert_eq!(original_id, loaded.id, "ID should persist");
        assert!(
            (original_coherence - loaded.coherence_with_actions).abs() < 1e-6,
            "Coherence should persist"
        );
        assert_eq!(
            original_trajectory_len,
            loaded.identity_trajectory.len(),
            "Trajectory length should persist"
        );
    }

    println!("RESULT: PASS - Ego node persists across store close/reopen");
}

#[tokio::test]
async fn test_ego_node_overwrite() {
    println!("=== TEST: save_ego_node overwrites previous value ===");

    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");
    let store = RocksDbTeleologicalStore::open(temp_dir.path())
        .expect("Failed to open store");

    // Save first ego node
    let ego1 = create_real_ego_node();
    let id1 = ego1.id;
    store.save_ego_node(&ego1).await.expect("Should save ego1");
    println!("STEP 1: Saved first ego node id={}", id1);

    // Save second ego node (should overwrite)
    let ego2 = create_ego_node_with_trajectory(10);
    let id2 = ego2.id;
    assert_ne!(id1, id2, "IDs should be different");
    store.save_ego_node(&ego2).await.expect("Should save ego2");
    println!("STEP 2: Saved second ego node id={}", id2);

    // Load - should get ego2
    let loaded = store.load_ego_node().await
        .expect("Should load")
        .expect("Ego node should exist");

    assert_eq!(loaded.id, id2, "Should load the second (latest) ego node");
    assert_eq!(loaded.identity_trajectory.len(), 10, "Should have 10 snapshots from ego2");

    println!("RESULT: PASS - save_ego_node overwrites correctly");
}

#[tokio::test]
async fn test_in_memory_store_ego_node_roundtrip() {
    println!("=== TEST: InMemoryTeleologicalStore ego node round-trip ===");

    use context_graph_core::stubs::InMemoryTeleologicalStore;

    let store = InMemoryTeleologicalStore::new();

    // Initially empty
    let initial = store.load_ego_node().await.expect("Should query");
    assert!(initial.is_none(), "Initially no ego node");

    // Save and load
    let original = create_real_ego_node();
    let original_id = original.id;

    store.save_ego_node(&original).await.expect("Should save");

    let loaded = store.load_ego_node().await
        .expect("Should load")
        .expect("Should exist");

    assert_eq!(original_id, loaded.id, "ID should match");

    println!("RESULT: PASS - InMemoryTeleologicalStore ego node round-trip successful");
}
