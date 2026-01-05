//! Advanced RocksDB integration tests.
//!
//! Tests for prefix scan, overwrite, delete, compression, and edge cases.

use context_graph_graph::storage::{
    get_column_family_descriptors, get_db_options, StorageConfig, ALL_COLUMN_FAMILIES,
    CF_ADJACENCY, CF_METADATA, CF_NODES,
};

#[test]
fn test_real_rocksdb_adjacency_prefix_scan() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_prefix_scan.db");

    println!("BEFORE: Testing adjacency prefix scan");

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    // Write multiple edges for the same source node
    let source_node: [u8; 16] = [1; 16];
    let adjacency_cf = db.cf_handle(CF_ADJACENCY).unwrap();

    // Store 3 edges from the same source
    for i in 0..3u8 {
        let mut key = source_node.to_vec();
        key.push(i); // Append edge index
        let value = format!("edge_to_target_{}", i);
        db.put_cf(adjacency_cf, &key, value.as_bytes()).unwrap();
    }

    println!("AFTER WRITE: Stored 3 edges from same source");

    // Use iterator to prefix scan
    let mut count = 0;
    let iter = db.prefix_iterator_cf(adjacency_cf, source_node);
    for item in iter {
        let (key, _value) = item.unwrap();
        if key.starts_with(&source_node) {
            count += 1;
        } else {
            break;
        }
    }

    assert_eq!(count, 3, "Should find 3 edges with same prefix");
    println!("AFTER SCAN: Found {} edges with prefix scan", count);
}

#[test]
fn test_storage_config_with_compression_disabled() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_no_compression.db");

    println!("BEFORE: Testing storage with compression disabled");

    let config = StorageConfig {
        enable_compression: false,
        ..Default::default()
    };

    let db_opts = get_db_options();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    // Write and read to verify it works
    let nodes_cf = db.cf_handle(CF_NODES).unwrap();
    db.put_cf(nodes_cf, b"test_key", b"test_value").unwrap();

    let value = db.get_cf(nodes_cf, b"test_key").unwrap();
    assert_eq!(value, Some(b"test_value".to_vec()));

    println!("AFTER: Verified storage works without compression");
}

#[test]
fn test_storage_multiple_writes_same_key() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_overwrite.db");

    println!("BEFORE: Testing overwrite behavior");

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    let metadata_cf = db.cf_handle(CF_METADATA).unwrap();

    // Write initial value
    db.put_cf(metadata_cf, b"version", b"1").unwrap();
    let v1 = db.get_cf(metadata_cf, b"version").unwrap();
    assert_eq!(v1, Some(b"1".to_vec()));

    // Overwrite
    db.put_cf(metadata_cf, b"version", b"2").unwrap();
    let v2 = db.get_cf(metadata_cf, b"version").unwrap();
    assert_eq!(v2, Some(b"2".to_vec()));

    // Overwrite again
    db.put_cf(metadata_cf, b"version", b"3").unwrap();
    let v3 = db.get_cf(metadata_cf, b"version").unwrap();
    assert_eq!(v3, Some(b"3".to_vec()));

    println!("AFTER: Verified overwrite behavior (1 -> 2 -> 3)");
}

#[test]
fn test_storage_delete_key() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_delete.db");

    println!("BEFORE: Testing delete behavior");

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    let nodes_cf = db.cf_handle(CF_NODES).unwrap();

    // Write
    db.put_cf(nodes_cf, b"to_delete", b"value").unwrap();
    let exists = db.get_cf(nodes_cf, b"to_delete").unwrap();
    assert!(exists.is_some());

    // Delete
    db.delete_cf(nodes_cf, b"to_delete").unwrap();
    let deleted = db.get_cf(nodes_cf, b"to_delete").unwrap();
    assert!(deleted.is_none());

    println!("AFTER: Verified delete behavior");
}

#[test]
fn test_storage_nonexistent_key_returns_none() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_none.db");

    println!("BEFORE: Testing nonexistent key behavior");

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    for cf_name in ALL_COLUMN_FAMILIES {
        let cf = db.cf_handle(cf_name).unwrap();
        let result = db.get_cf(cf, b"nonexistent_key_12345").unwrap();
        assert!(
            result.is_none(),
            "Nonexistent key should return None in {}",
            cf_name
        );
    }

    println!("AFTER: Verified all CFs return None for nonexistent keys");
}

#[test]
fn test_storage_empty_value() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_empty_value.db");

    println!("BEFORE: Testing empty value storage");

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    let metadata_cf = db.cf_handle(CF_METADATA).unwrap();
    db.put_cf(metadata_cf, b"empty_key", b"").unwrap();

    let result = db.get_cf(metadata_cf, b"empty_key").unwrap();
    assert_eq!(result, Some(vec![]));

    println!("AFTER: Verified empty value storage");
}

#[test]
fn test_storage_large_value() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_large_value.db");

    println!("BEFORE: Testing large value storage (1MB)");

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    // 1MB value
    let large_value: Vec<u8> = (0..1024 * 1024).map(|i| (i % 256) as u8).collect();

    let nodes_cf = db.cf_handle(CF_NODES).unwrap();
    db.put_cf(nodes_cf, b"large_key", &large_value).unwrap();

    let result = db.get_cf(nodes_cf, b"large_key").unwrap().unwrap();
    assert_eq!(result.len(), 1024 * 1024);
    assert_eq!(result[0], 0);
    assert_eq!(result[1024 * 1024 - 1], 255);

    println!("AFTER: Verified 1MB value storage");
}
