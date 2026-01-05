//! Tests for StorageConfig.

use crate::error::GraphError;
use crate::storage::config::StorageConfig;

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
    // Inherited from default
    assert!(config.enable_compression);
    assert_eq!(config.write_buffer_size, 64 * 1024 * 1024);
}

#[test]
fn test_storage_config_write_optimized() {
    let config = StorageConfig::write_optimized();
    assert_eq!(config.write_buffer_size, 128 * 1024 * 1024); // 128MB
    assert_eq!(config.max_write_buffers, 5);
    // Inherited from default
    assert!(config.enable_compression);
    assert_eq!(config.block_cache_size, 512 * 1024 * 1024);
}

#[test]
fn test_storage_config_validate_success() {
    let config = StorageConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_storage_config_validate_read_optimized() {
    let config = StorageConfig::read_optimized();
    assert!(config.validate().is_ok());
}

#[test]
fn test_storage_config_validate_write_optimized() {
    let config = StorageConfig::write_optimized();
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
fn test_storage_config_validate_block_cache_boundary() {
    // Exactly 1MB should pass
    let config = StorageConfig {
        block_cache_size: 1024 * 1024,
        ..Default::default()
    };
    assert!(config.validate().is_ok());

    // 1 byte less should fail
    let config = StorageConfig {
        block_cache_size: 1024 * 1024 - 1,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_storage_config_validate_bloom_filter_invalid_zero() {
    let config = StorageConfig {
        bloom_filter_bits: 0,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(matches!(err, GraphError::InvalidConfig(_)));
    assert!(err.to_string().contains("bloom_filter_bits"));
}

#[test]
fn test_storage_config_validate_bloom_filter_invalid_high() {
    let config = StorageConfig {
        bloom_filter_bits: 21,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(matches!(err, GraphError::InvalidConfig(_)));
    assert!(err.to_string().contains("bloom_filter_bits"));
}

#[test]
fn test_storage_config_validate_bloom_filter_boundaries() {
    // 1 should pass
    let config = StorageConfig {
        bloom_filter_bits: 1,
        ..Default::default()
    };
    assert!(config.validate().is_ok());

    // 20 should pass
    let config = StorageConfig {
        bloom_filter_bits: 20,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
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
