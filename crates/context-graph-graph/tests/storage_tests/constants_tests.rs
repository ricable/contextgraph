//! Tests for column family constants and StorageConfig validation.
//!
//! M04-T12: Column family definitions and configuration

use context_graph_graph::error::GraphError;
use context_graph_graph::storage::{
    get_column_family_descriptors, StorageConfig, ALL_COLUMN_FAMILIES, CF_ADJACENCY, CF_CONES,
    CF_EDGES, CF_FAISS_IDS, CF_HYPERBOLIC, CF_METADATA, CF_NODES,
};

// ========== Constants Tests ==========

#[test]
fn test_cf_names() {
    assert_eq!(CF_ADJACENCY, "adjacency");
    assert_eq!(CF_HYPERBOLIC, "hyperbolic");
    assert_eq!(CF_CONES, "entailment_cones");
    assert_eq!(CF_FAISS_IDS, "faiss_ids");
    assert_eq!(CF_NODES, "nodes");
    assert_eq!(CF_METADATA, "metadata");
}

#[test]
fn test_all_column_families_count() {
    // M04-T15: Added CF_EDGES column family
    assert_eq!(ALL_COLUMN_FAMILIES.len(), 7);
}

#[test]
fn test_all_column_families_contains_all() {
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_ADJACENCY));
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_HYPERBOLIC));
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_CONES));
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_FAISS_IDS));
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_NODES));
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_METADATA));
    // M04-T15: Added CF_EDGES column family
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_EDGES));
}

// ========== StorageConfig Tests ==========

#[test]
fn test_storage_config_default() {
    let config = StorageConfig::default();
    assert_eq!(config.block_cache_size, 512 * 1024 * 1024);
    assert!(config.enable_compression);
    assert_eq!(config.bloom_filter_bits, 10);
    assert_eq!(config.write_buffer_size, 64 * 1024 * 1024);
    assert_eq!(config.max_write_buffers, 3);
    assert_eq!(config.target_file_size_base, 64 * 1024 * 1024);
}

#[test]
fn test_storage_config_read_optimized() {
    let config = StorageConfig::read_optimized();
    assert_eq!(config.block_cache_size, 1024 * 1024 * 1024); // 1GB
    assert_eq!(config.bloom_filter_bits, 14);
}

#[test]
fn test_storage_config_write_optimized() {
    let config = StorageConfig::write_optimized();
    assert_eq!(config.write_buffer_size, 128 * 1024 * 1024); // 128MB
    assert_eq!(config.max_write_buffers, 5);
}

#[test]
fn test_storage_config_validate_success() {
    let config = StorageConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_storage_config_validate_block_cache_too_small() {
    let config = StorageConfig {
        block_cache_size: 1024, // Only 1KB
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(matches!(err, GraphError::InvalidConfig(_)));
    assert!(err.to_string().contains("block_cache_size"));
}

#[test]
fn test_storage_config_validate_bloom_filter_invalid() {
    let config = StorageConfig {
        bloom_filter_bits: 0, // Invalid
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(matches!(err, GraphError::InvalidConfig(_)));
    assert!(err.to_string().contains("bloom_filter_bits"));
}

#[test]
fn test_storage_config_validate_write_buffer_too_small() {
    let config = StorageConfig {
        write_buffer_size: 512, // Only 512 bytes
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(matches!(err, GraphError::InvalidConfig(_)));
    assert!(err.to_string().contains("write_buffer_size"));
}

// ========== Column Family Descriptor Tests ==========

#[test]
fn test_get_column_family_descriptors_count() {
    let config = StorageConfig::default();
    let descriptors = get_column_family_descriptors(&config).unwrap();
    // M04-T15: Added CF_EDGES column family
    assert_eq!(descriptors.len(), 7);
}

#[test]
fn test_get_column_family_descriptors_invalid_config() {
    let config = StorageConfig {
        block_cache_size: 0,
        ..Default::default()
    };
    let result = get_column_family_descriptors(&config);
    assert!(result.is_err());
}
