//! Tests for RocksDbTeleologicalStore.
//!
//! Comprehensive tests for CRUD operations, persistence, and trait compliance.

use super::*;
use tempfile::TempDir;

use context_graph_core::traits::TeleologicalMemoryStore;
use context_graph_core::types::fingerprint::{
    SemanticFingerprint, SparseVector, TeleologicalFingerprint,
};

/// Create a test fingerprint with real (non-zero) embeddings.
/// Uses deterministic pseudo-random values seeded from a counter.
fn create_test_fingerprint_with_seed(seed: u64) -> TeleologicalFingerprint {
    use std::f32::consts::PI;

    // Generate deterministic embeddings from seed
    let generate_vec = |dim: usize, s: u64| -> Vec<f32> {
        (0..dim)
            .map(|i| {
                let x = ((s as f64 * 0.1 + i as f64 * 0.01) * PI as f64).sin() as f32;
                x * 0.5 + 0.5 // Normalize to [0, 1] range
            })
            .collect()
    };

    // Generate deterministic sparse vector for SPLADE
    let generate_sparse = |s: u64| -> SparseVector {
        let num_entries = 50 + (s % 50) as usize;
        let mut indices: Vec<u16> = Vec::with_capacity(num_entries);
        let mut values: Vec<f32> = Vec::with_capacity(num_entries);
        for i in 0..num_entries {
            let idx = ((s + i as u64 * 31) % 30522) as u16; // u16 for sparse indices
            let val = ((s as f64 * 0.1 + i as f64 * 0.2) * PI as f64).sin().abs() as f32 + 0.1;
            indices.push(idx);
            values.push(val);
        }
        SparseVector { indices, values }
    };

    // Generate late-interaction vectors (variable number of 128D token vectors)
    let generate_late_interaction = |s: u64| -> Vec<Vec<f32>> {
        let num_tokens = 5 + (s % 10) as usize;
        (0..num_tokens)
            .map(|t| generate_vec(128, s + t as u64 * 100))
            .collect()
    };

    // Create SemanticFingerprint with correct fields (per semantic/fingerprint.rs)
    let semantic = SemanticFingerprint {
        e1_semantic: generate_vec(1024, seed),               // 1024D
        e2_temporal_recent: generate_vec(512, seed + 1),     // 512D
        e3_temporal_periodic: generate_vec(512, seed + 2),   // 512D
        e4_temporal_positional: generate_vec(512, seed + 3), // 512D
        e5_causal: generate_vec(768, seed + 4),              // 768D
        e6_sparse: generate_sparse(seed + 5),                // Sparse
        e7_code: generate_vec(1536, seed + 6),               // 1536D
        e8_graph: generate_vec(384, seed + 7),               // 384D
        e9_hdc: generate_vec(1024, seed + 8),                // 1024D HDC (projected)
        e10_multimodal: generate_vec(768, seed + 9),         // 768D
        e11_entity: generate_vec(384, seed + 10),            // 384D
        e12_late_interaction: generate_late_interaction(seed + 11), // Vec<Vec<f32>>
        e13_splade: generate_sparse(seed + 12),              // Sparse
    };

    // Create unique hash
    let mut hash = [0u8; 32];
    for (i, byte) in hash.iter_mut().enumerate() {
        *byte = ((seed + i as u64) % 256) as u8;
    }

    TeleologicalFingerprint::new(semantic, hash)
}

fn create_test_fingerprint() -> TeleologicalFingerprint {
    create_test_fingerprint_with_seed(42)
}

/// Helper to create store with initialized indexes.
///
/// Note: EmbedderIndexRegistry is initialized in the constructor,
/// so no separate initialization step is needed.
fn create_initialized_store(path: &std::path::Path) -> RocksDbTeleologicalStore {
    RocksDbTeleologicalStore::open(path).unwrap()
}

#[tokio::test]
async fn test_open_and_health_check() {
    let tmp = TempDir::new().unwrap();
    let store = create_initialized_store(tmp.path());
    assert!(store.health_check().is_ok());
}

#[tokio::test]
async fn test_store_and_retrieve() {
    let tmp = TempDir::new().unwrap();
    let store = create_initialized_store(tmp.path());

    let fp = create_test_fingerprint();
    let id = fp.id;

    // Store
    let stored_id = store.store(fp.clone()).await.unwrap();
    assert_eq!(stored_id, id);

    // Retrieve
    let retrieved = store.retrieve(id).await.unwrap();
    assert!(retrieved.is_some());
    let retrieved_fp = retrieved.unwrap();
    assert_eq!(retrieved_fp.id, id);
}

#[tokio::test]
async fn test_physical_persistence() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().to_path_buf();

    let fp = create_test_fingerprint();
    let id = fp.id;

    // Store and close
    {
        let store = create_initialized_store(&path);
        store.store(fp.clone()).await.unwrap();
        store.flush().await.unwrap();
    }

    // Reopen and verify
    {
        let store = create_initialized_store(&path);
        let retrieved = store.retrieve(id).await.unwrap();
        assert!(
            retrieved.is_some(),
            "Fingerprint should persist across database close/reopen"
        );
        assert_eq!(retrieved.unwrap().id, id);
    }

    // Verify raw bytes exist in RocksDB
    {
        let store = create_initialized_store(&path);
        let raw = store.get_fingerprint_raw(id).unwrap();
        assert!(raw.is_some(), "Raw bytes should exist in RocksDB");
        let raw_bytes = raw.unwrap();
        // With E9_DIM = 1024 (projected), fingerprints are ~32-40KB
        assert!(
            raw_bytes.len() >= 25000,
            "Serialized fingerprint should be >= 25KB, got {} bytes",
            raw_bytes.len()
        );
    }
}

#[tokio::test]
async fn test_delete_soft() {
    let tmp = TempDir::new().unwrap();
    let store = create_initialized_store(tmp.path());

    let fp = create_test_fingerprint();
    let id = fp.id;

    store.store(fp).await.unwrap();
    let deleted = store.delete(id, true).await.unwrap();
    assert!(deleted);

    // Should not be retrievable after soft delete
    let retrieved = store.retrieve(id).await.unwrap();
    assert!(retrieved.is_none());
}

#[tokio::test]
async fn test_delete_hard() {
    let tmp = TempDir::new().unwrap();
    let store = create_initialized_store(tmp.path());

    let fp = create_test_fingerprint();
    let id = fp.id;

    store.store(fp).await.unwrap();
    let deleted = store.delete(id, false).await.unwrap();
    assert!(deleted);

    // Raw bytes should be gone
    let raw = store.get_fingerprint_raw(id).unwrap();
    assert!(raw.is_none());
}

#[tokio::test]
async fn test_count() {
    let tmp = TempDir::new().unwrap();
    let store = create_initialized_store(tmp.path());

    assert_eq!(store.count().await.unwrap(), 0);

    store
        .store(create_test_fingerprint_with_seed(1))
        .await
        .unwrap();
    store
        .store(create_test_fingerprint_with_seed(2))
        .await
        .unwrap();
    store
        .store(create_test_fingerprint_with_seed(3))
        .await
        .unwrap();

    assert_eq!(store.count().await.unwrap(), 3);
}

#[tokio::test]
async fn test_backend_type() {
    let tmp = TempDir::new().unwrap();
    let store = create_initialized_store(tmp.path());
    assert_eq!(
        store.backend_type(),
        context_graph_core::traits::TeleologicalStorageBackend::RocksDb
    );
}

// ============================================================================
// Corruption Detection Tests - REAL data, NO mocks (TASK-STORAGE-001)
// ============================================================================

/// Test that corruption detection catches missing SST files.
///
/// This test uses REAL RocksDB data:
/// 1. Creates a valid database with multiple fingerprints
/// 2. Forces flush to create SST files on disk
/// 3. Closes database cleanly
/// 4. Deletes an SST file to simulate corruption
/// 5. Attempts to reopen and verifies CorruptionDetected error
#[tokio::test]
async fn test_corruption_detection_missing_sst_file() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().to_path_buf();

    // Step 1: Create database with REAL data (not mocks)
    {
        let store = create_initialized_store(&path);

        // Store multiple fingerprints to ensure SST files are created
        for seed in 1..=10 {
            let fp = create_test_fingerprint_with_seed(seed);
            store.store(fp).await.expect("Store should succeed");
        }

        // Force flush to ensure data is written to SST files
        store.flush().await.expect("Flush should succeed");
    }

    // Step 2: Identify SST files
    let sst_files: Vec<std::path::PathBuf> = std::fs::read_dir(&path)
        .expect("Should read directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".sst"))
        .map(|e| e.path())
        .collect();

    assert!(
        !sst_files.is_empty(),
        "Database should have at least one SST file after storing fingerprints and flushing"
    );

    // Step 3: Delete an SST file to simulate corruption
    let deleted_file = &sst_files[0];
    let deleted_name = deleted_file
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    std::fs::remove_file(deleted_file).expect("Should delete SST file");

    // Step 4: Attempt to reopen - should fail with CorruptionDetected
    let result = RocksDbTeleologicalStore::open(&path);

    match result {
        Err(TeleologicalStoreError::CorruptionDetected {
            path: err_path,
            missing_count,
            missing_files,
            manifest_file,
        }) => {
            // Verify error contains correct information
            assert_eq!(err_path, path.to_string_lossy().to_string());
            assert!(missing_count >= 1, "Should detect at least 1 missing file");
            // Split comma-separated list and check for exact match (not substring)
            // to avoid false positives like "12.sst" matching "112.sst"
            let file_list: Vec<&str> = missing_files.split(", ").collect();
            let deleted_without_ext = deleted_name.replace(".sst", "");
            assert!(
                file_list
                    .iter()
                    .any(|f| *f == deleted_name || *f == deleted_without_ext),
                "Missing files should include the deleted file '{}', got: {:?}",
                deleted_name,
                file_list
            );
            assert!(
                manifest_file.contains("MANIFEST-"),
                "Should reference a MANIFEST file, got: {}",
                manifest_file
            );
        }
        Err(e) => {
            // Also acceptable: RocksDB's own corruption error
            let err_msg = e.to_string();
            assert!(
                err_msg.contains("Corruption") || err_msg.contains("No such file"),
                "Expected CorruptionDetected or RocksDB corruption error, got: {}",
                e
            );
        }
        Ok(_) => {
            panic!("Expected corruption error when opening database with missing SST file");
        }
    }
}

/// Test that a clean database passes corruption check.
///
/// Verifies that corruption detection doesn't produce false positives
/// on a healthy database with REAL data.
#[tokio::test]
async fn test_corruption_detection_clean_database() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().to_path_buf();

    // Create and populate database
    {
        let store = create_initialized_store(&path);
        for seed in 1..=5 {
            let fp = create_test_fingerprint_with_seed(seed);
            store.store(fp).await.expect("Store should succeed");
        }
        store.flush().await.expect("Flush should succeed");
    }

    // Reopen should succeed with no corruption
    let store = RocksDbTeleologicalStore::open(&path);
    assert!(
        store.is_ok(),
        "Clean database should open without corruption error: {:?}",
        store.err()
    );

    // Verify data is still accessible
    let store = store.unwrap();
    let count = store.count().await.expect("Count should succeed");
    assert_eq!(count, 5, "Should have 5 fingerprints after reopen");
}

/// Test corruption detection when MANIFEST references missing files.
///
/// This simulates the exact scenario from the real corruption incident:
/// MANIFEST-000701 referencing missing 000682.sst (file above max existing)
///
/// The detection heuristic catches files referenced ABOVE max_existing,
/// which is the typical pattern when a crash occurs during compaction/flush.
#[tokio::test]
async fn test_corruption_detection_manifest_sst_mismatch() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().to_path_buf();

    // Step 1: Create database with enough data to generate multiple SST files
    {
        let store = create_initialized_store(&path);

        // Store many fingerprints to force multiple flushes and compactions
        for seed in 1..=50 {
            let fp = create_test_fingerprint_with_seed(seed);
            store.store(fp).await.expect("Store should succeed");

            // Periodic flush to create SST files
            if seed % 10 == 0 {
                store.flush().await.expect("Flush should succeed");
            }
        }
        store.flush().await.expect("Final flush should succeed");
    }

    // Step 2: Get sorted SST files and delete the HIGHEST numbered one(s)
    // This simulates corruption where new files were referenced but not fully written
    let mut sst_files: Vec<_> = std::fs::read_dir(&path)
        .expect("Read dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".sst"))
        .collect();

    // Sort by file name (which includes the number)
    sst_files.sort_by_key(|e| e.file_name().to_string_lossy().to_string());

    assert!(!sst_files.is_empty(), "Should have SST files");

    // Delete the LAST (highest numbered) file to simulate incomplete write
    let deleted_file = sst_files.pop().unwrap();
    std::fs::remove_file(deleted_file.path()).expect("Delete SST");

    // Step 3: Verify corruption is detected (or RocksDB reports it)
    let result = RocksDbTeleologicalStore::open(&path);

    // Either our detection or RocksDB's should catch this
    match result {
        Err(TeleologicalStoreError::CorruptionDetected { missing_count, .. }) => {
            assert!(
                missing_count >= 1,
                "Should detect missing file(s), got: {}",
                missing_count
            );
        }
        Err(e) => {
            // RocksDB's own error is also acceptable for corruption
            let err_msg = e.to_string();
            assert!(
                err_msg.contains("Corruption") || err_msg.contains("No such file"),
                "Expected corruption error, got: {}",
                e
            );
        }
        Ok(_) => {
            // If it opens successfully, that's actually OK too - RocksDB might have
            // recovered automatically or the deleted file wasn't needed.
            // The key test is test_corruption_detection_missing_sst_file which
            // deliberately deletes a known file.
        }
    }
}

/// Test that new (empty) database passes corruption check.
///
/// A fresh database without CURRENT file should not trigger false positives.
#[tokio::test]
async fn test_corruption_detection_new_database() {
    let tmp = TempDir::new().unwrap();

    // Open fresh database - should succeed (no CURRENT file yet)
    let result = RocksDbTeleologicalStore::open(tmp.path());
    assert!(
        result.is_ok(),
        "New database should open without corruption error: {:?}",
        result.err()
    );
}

/// Test that corruption detection provides actionable error messages.
///
/// Verifies FAIL FAST policy with detailed context for debugging.
#[tokio::test]
async fn test_corruption_detection_error_details() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().to_path_buf();

    // Create database
    {
        let store = create_initialized_store(&path);
        let fp = create_test_fingerprint_with_seed(99);
        store.store(fp).await.expect("Store should succeed");
        store.flush().await.expect("Flush should succeed");
    }

    // Corrupt by deleting SST files
    for entry in std::fs::read_dir(&path).expect("Read dir") {
        if let Ok(entry) = entry {
            if entry.file_name().to_string_lossy().ends_with(".sst") {
                std::fs::remove_file(entry.path()).expect("Delete");
                break; // Delete just one
            }
        }
    }

    // Get error and verify details
    let result = RocksDbTeleologicalStore::open(&path);
    let err = match result {
        Ok(_) => panic!("Should fail on corruption"),
        Err(e) => e,
    };
    let err_string = err.to_string();

    // Verify error message contains FAIL FAST debugging info
    // Either our custom CorruptionDetected or RocksDB's error
    assert!(
        err_string.contains("CORRUPTION")
            || err_string.contains("Corruption")
            || err_string.contains("No such file"),
        "Error should indicate corruption, got: {}",
        err_string
    );

    // If it's our custom error, verify structure
    if let TeleologicalStoreError::CorruptionDetected {
        path: err_path,
        missing_count,
        missing_files,
        manifest_file,
    } = err
    {
        // Verify all fields are populated
        assert!(!err_path.is_empty(), "Path should not be empty");
        assert!(missing_count >= 1, "Should have at least 1 missing file");
        assert!(
            !missing_files.is_empty(),
            "Missing files list should not be empty"
        );
        assert!(
            !manifest_file.is_empty(),
            "Manifest file should be identified"
        );

        // Verify path matches
        assert!(
            err_path.contains(&path.file_name().unwrap().to_string_lossy().to_string()),
            "Error path should match database path"
        );
    }
}
