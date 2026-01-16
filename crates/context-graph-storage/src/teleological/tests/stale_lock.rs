//! Stale lock detection tests.
//!
//! These tests verify that the RocksDB store correctly handles stale LOCK files
//! that can be left behind when a process crashes or is killed.
//!
//! CRITICAL: Uses #[tokio::test] to prevent zombie runtime threads.
//! DO NOT use tokio::runtime::Runtime::new() in tests.

use crate::teleological::RocksDbTeleologicalStore;
use super::helpers::create_real_fingerprint;
use context_graph_core::traits::TeleologicalMemoryStore;
use tempfile::TempDir;
use std::fs;

#[tokio::test]
async fn test_stale_lock_detection_opens_after_stale_lock() {
    println!("=== STALE LOCK TEST: Database opens after stale LOCK file ===");

    // Create a temporary directory
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("stale_lock_test");
    fs::create_dir_all(&db_path).expect("Failed to create db dir");

    // Step 1: Create a stale LOCK file (simulating crashed process)
    let lock_path = db_path.join("LOCK");
    fs::write(&lock_path, "").expect("Failed to create stale LOCK file");
    println!("BEFORE: Created stale LOCK file at {:?}", lock_path);
    assert!(lock_path.exists(), "LOCK file should exist");

    // Step 2: Open the database - this should detect and remove the stale lock
    println!("OPENING: Attempting to open database with stale LOCK...");
    let store = RocksDbTeleologicalStore::open(&db_path)
        .expect("Should open successfully after stale lock removal");

    println!("AFTER: Database opened successfully");

    // Step 3: Verify the database is usable by performing a basic operation
    let count = store.count().await.expect("Should be able to count");
    println!("VERIFY: Database count = {} (expected 0 for new DB)", count);
    assert_eq!(count, 0, "New database should have 0 entries");

    println!("RESULT: PASS - Stale lock detected and database opened successfully");
}

#[tokio::test]
async fn test_stale_lock_detection_fresh_database() {
    println!("=== STALE LOCK TEST: Fresh database opens without LOCK file ===");

    // Create a temporary directory with no LOCK file
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("fresh_db_test");
    fs::create_dir_all(&db_path).expect("Failed to create db dir");

    let lock_path = db_path.join("LOCK");
    println!("BEFORE: No LOCK file exists at {:?}", lock_path);
    assert!(!lock_path.exists(), "LOCK file should NOT exist");

    // Open the database - should work normally
    println!("OPENING: Opening fresh database...");
    let store = RocksDbTeleologicalStore::open(&db_path)
        .expect("Should open fresh database successfully");

    println!("AFTER: Database opened successfully");

    // Verify the database is usable
    let count = store.count().await.expect("Should be able to count");
    println!("VERIFY: Database count = {} (expected 0 for new DB)", count);
    assert_eq!(count, 0, "New database should have 0 entries");

    println!("RESULT: PASS - Fresh database opened without issues");
}

#[tokio::test]
async fn test_stale_lock_detection_reopen_after_close() {
    println!("=== STALE LOCK TEST: Database reopens after clean close ===");

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("reopen_test");
    fs::create_dir_all(&db_path).expect("Failed to create db dir");

    // Step 1: Open, write, and close the database
    println!("STEP 1: Opening database and writing data...");
    {
        let store = RocksDbTeleologicalStore::open(&db_path)
            .expect("Should open database");

        let fp = create_real_fingerprint();
        store.store(fp).await.expect("Should store fingerprint");
        println!("STEP 1: Stored 1 fingerprint, dropping database handle...");
    } // Database should be closed here, releasing the LOCK

    // Step 2: Verify LOCK file doesn't exist (or will be cleaned if stale)
    let lock_path = db_path.join("LOCK");
    println!("STEP 2: LOCK file exists = {} (may or may not based on RocksDB behavior)", lock_path.exists());

    // Step 3: Reopen the database
    println!("STEP 3: Reopening database...");
    let store = RocksDbTeleologicalStore::open(&db_path)
        .expect("Should reopen database successfully");

    // Step 4: Verify data persisted
    let count = store.count().await.expect("Should be able to count");
    println!("VERIFY: Database count = {} (expected 1)", count);
    assert_eq!(count, 1, "Reopened database should have 1 entry");

    println!("RESULT: PASS - Database reopened and data persisted");
}

#[tokio::test]
async fn test_stale_lock_multiple_stale_lock_files() {
    println!("=== STALE LOCK TEST: Multiple operations with simulated stale locks ===");

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("multi_stale_test");
    fs::create_dir_all(&db_path).expect("Failed to create db dir");

    // Iteration 1: Create stale lock, open, close
    println!("ITERATION 1: Creating stale lock and opening...");
    let lock_path = db_path.join("LOCK");
    fs::write(&lock_path, "").expect("Failed to create stale LOCK");
    {
        let _store = RocksDbTeleologicalStore::open(&db_path)
            .expect("Iteration 1: Should open");
    }
    println!("ITERATION 1: Closed database");

    // Iteration 2: Simulate another crash (create stale lock), reopen
    println!("ITERATION 2: Simulating crash, creating new stale lock...");
    // Note: In real scenario, the LOCK might be released on close.
    // We simulate a crash by re-creating the LOCK file.
    if !lock_path.exists() {
        fs::write(&lock_path, "").expect("Failed to create stale LOCK");
    }
    {
        let _store = RocksDbTeleologicalStore::open(&db_path)
            .expect("Iteration 2: Should open after stale lock");
    }
    println!("ITERATION 2: Closed database");

    // Iteration 3: One more time
    println!("ITERATION 3: Final stale lock test...");
    if !lock_path.exists() {
        fs::write(&lock_path, "").expect("Failed to create stale LOCK");
    }
    let store = RocksDbTeleologicalStore::open(&db_path)
        .expect("Iteration 3: Should open after stale lock");

    // Verify database is functional
    let count = store.count().await.expect("Should be able to count");
    println!("VERIFY: Database opened {} times with stale locks, count = {}", 3, count);

    println!("RESULT: PASS - Multiple stale lock scenarios handled");
}
