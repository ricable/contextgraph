//! Tests for IndexConfig.

use crate::config::IndexConfig;

#[test]
fn test_index_config_default_values() {
    let config = IndexConfig::default();
    assert_eq!(config.dimension, 1536);
    assert_eq!(config.nlist, 16384);
    assert_eq!(config.nprobe, 128);
    assert_eq!(config.pq_segments, 64);
    assert_eq!(config.pq_bits, 8);
    assert_eq!(config.gpu_id, 0);
    assert!(config.use_float16);
    assert_eq!(config.min_train_vectors, 4_194_304);
}

#[test]
fn test_index_config_pq_segments_divides_dimension() {
    let config = IndexConfig::default();
    assert_eq!(
        config.dimension % config.pq_segments,
        0,
        "PQ segments must divide dimension evenly"
    );
}

#[test]
fn test_index_config_min_train_vectors_formula() {
    let config = IndexConfig::default();
    assert_eq!(
        config.min_train_vectors,
        256 * config.nlist,
        "min_train_vectors must equal 256 * nlist"
    );
}

#[test]
fn test_factory_string_default() {
    let config = IndexConfig::default();
    assert_eq!(config.factory_string(), "IVF16384,PQ64x8");
}

#[test]
fn test_factory_string_custom() {
    let config = IndexConfig {
        dimension: 768,
        nlist: 4096,
        nprobe: 64,
        pq_segments: 32,
        pq_bits: 4,
        gpu_id: 1,
        use_float16: false,
        min_train_vectors: 256 * 4096,
    };
    assert_eq!(config.factory_string(), "IVF4096,PQ32x4");
}

#[test]
fn test_calculate_min_train_vectors() {
    let config = IndexConfig::default();
    assert_eq!(config.calculate_min_train_vectors(), 4_194_304);

    let custom = IndexConfig {
        nlist: 1024,
        ..Default::default()
    };
    assert_eq!(custom.calculate_min_train_vectors(), 256 * 1024);
}

#[test]
fn test_index_config_serialization_roundtrip() {
    let config = IndexConfig::default();
    let json = serde_json::to_string(&config).expect("Serialization failed");
    let deserialized: IndexConfig = serde_json::from_str(&json).expect("Deserialization failed");
    assert_eq!(config, deserialized);
}

#[test]
fn test_index_config_json_format() {
    let config = IndexConfig::default();
    let json = serde_json::to_string_pretty(&config).expect("Serialization failed");
    assert!(json.contains("\"dimension\": 1536"));
    assert!(json.contains("\"nlist\": 16384"));
    assert!(json.contains("\"nprobe\": 128"));
    assert!(json.contains("\"pq_segments\": 64"));
    assert!(json.contains("\"pq_bits\": 8"));
    assert!(json.contains("\"gpu_id\": 0"));
    assert!(json.contains("\"use_float16\": true"));
    assert!(json.contains("\"min_train_vectors\": 4194304"));
}

#[test]
fn test_pq_bits_type_is_u8() {
    let config = IndexConfig::default();
    // This is a compile-time check - if pq_bits is not u8, this won't compile
    let _: u8 = config.pq_bits;
}
