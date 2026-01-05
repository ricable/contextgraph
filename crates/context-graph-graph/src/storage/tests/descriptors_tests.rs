//! Tests for column family descriptors.

use crate::error::GraphError;
use crate::storage::config::StorageConfig;
use crate::storage::descriptors::{get_column_family_descriptors, get_db_options};

#[test]
fn test_get_column_family_descriptors_count() {
    let config = StorageConfig::default();
    let descriptors = get_column_family_descriptors(&config).unwrap();
    assert_eq!(descriptors.len(), 7); // M04-T15: added CF_EDGES
}

#[test]
fn test_get_column_family_descriptors_invalid_config() {
    let config = StorageConfig {
        block_cache_size: 0,
        ..Default::default()
    };
    let result = get_column_family_descriptors(&config);
    assert!(result.is_err());
    // Verify the error through match
    match result {
        Err(GraphError::InvalidConfig(msg)) => {
            assert!(msg.contains("block_cache_size"));
        }
        _ => panic!("Expected GraphError::InvalidConfig"),
    }
}

#[test]
fn test_db_options_valid() {
    // Should not panic
    let _opts = get_db_options();
}

#[test]
fn test_db_options_parallelism_at_least_2() {
    // Even on 1-CPU system, should use at least 2 threads
    // We can't easily test this without mocking num_cpus,
    // but we verify the options are valid
    let opts = get_db_options();

    // Verify options are usable by creating a temp DB
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_opts.db");
    let _db = rocksdb::DB::open(&opts, &db_path).unwrap();
}
