//! Basic RocksDB integration tests.
//!
//! Tests for column families, open/close, and basic read/write operations.

use context_graph_graph::storage::{
    get_column_family_descriptors, get_db_options, StorageConfig, ALL_COLUMN_FAMILIES, CF_CONES,
    CF_FAISS_IDS, CF_HYPERBOLIC, CF_METADATA, CF_NODES,
};

#[test]
fn test_real_rocksdb_open_with_column_families() {
    // REAL RocksDB - no mocks
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_cf.db");

    println!("BEFORE: Opening RocksDB at {:?}", db_path);

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    // Open REAL database
    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors)
        .expect("Failed to open RocksDB with column families");

    println!("AFTER: RocksDB opened successfully");

    // Verify all CFs exist
    for cf_name in ALL_COLUMN_FAMILIES {
        let cf_handle = db.cf_handle(cf_name);
        assert!(cf_handle.is_some(), "Column family {} must exist", cf_name);
        println!("VERIFIED: Column family '{}' exists", cf_name);
    }
}

#[test]
fn test_real_rocksdb_write_and_read_metadata() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_rw_metadata.db");

    println!(
        "BEFORE: Opening RocksDB for write/read test at {:?}",
        db_path
    );

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    // Write to metadata CF
    let metadata_cf = db.cf_handle(CF_METADATA).unwrap();
    db.put_cf(metadata_cf, b"schema_version", b"1").unwrap();

    println!("AFTER WRITE: Wrote schema_version=1 to metadata CF");

    // Read back
    let value = db.get_cf(metadata_cf, b"schema_version").unwrap();
    assert_eq!(value, Some(b"1".to_vec()));

    println!("AFTER READ: Verified schema_version=1");
}

#[test]
fn test_real_rocksdb_write_to_all_cfs() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_all_cfs.db");

    println!(
        "BEFORE: Opening RocksDB for all-CF write test at {:?}",
        db_path
    );

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    // Write to each CF and verify
    for cf_name in ALL_COLUMN_FAMILIES {
        let cf = db.cf_handle(cf_name).unwrap();
        let key = format!("test_key_{}", cf_name);
        let value = format!("test_value_{}", cf_name);

        db.put_cf(cf, key.as_bytes(), value.as_bytes()).unwrap();

        let result = db.get_cf(cf, key.as_bytes()).unwrap();
        assert_eq!(result, Some(value.as_bytes().to_vec()));

        println!("VERIFIED: CF '{}' write/read successful", cf_name);
    }
}

#[test]
fn test_real_rocksdb_write_hyperbolic_coordinates() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_hyperbolic.db");

    println!("BEFORE: Testing hyperbolic coordinate storage");

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    // Create 64D hyperbolic coordinates (256 bytes as per spec)
    let node_id: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let coordinates: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
    let coords_bytes: Vec<u8> = coordinates.iter().flat_map(|f| f.to_le_bytes()).collect();

    assert_eq!(
        coords_bytes.len(),
        256,
        "Hyperbolic coords must be 256 bytes"
    );

    let hyperbolic_cf = db.cf_handle(CF_HYPERBOLIC).unwrap();
    db.put_cf(hyperbolic_cf, node_id, &coords_bytes).unwrap();

    println!("AFTER WRITE: Stored 64D coordinates (256 bytes)");

    // Read back and verify
    let result = db.get_cf(hyperbolic_cf, node_id).unwrap().unwrap();
    assert_eq!(result.len(), 256);

    // Deserialize and verify first value
    let first_f32 = f32::from_le_bytes([result[0], result[1], result[2], result[3]]);
    assert!((first_f32 - 0.0).abs() < 0.0001);

    println!("AFTER READ: Verified 256-byte hyperbolic coordinates");
}

#[test]
fn test_real_rocksdb_write_entailment_cone() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_cones.db");

    println!("BEFORE: Testing entailment cone storage");

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    // Create entailment cone: 268 bytes (256 coords + 4 aperture + 4 factor + 4 depth)
    let node_id: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let mut cone_data: Vec<u8> = Vec::with_capacity(268);

    // 256 bytes for coordinates (64 f32)
    for i in 0..64 {
        cone_data.extend_from_slice(&(i as f32 * 0.01f32).to_le_bytes());
    }
    // 4 bytes for aperture
    cone_data.extend_from_slice(&0.5f32.to_le_bytes());
    // 4 bytes for factor
    cone_data.extend_from_slice(&1.0f32.to_le_bytes());
    // 4 bytes for depth
    cone_data.extend_from_slice(&3u32.to_le_bytes());

    assert_eq!(cone_data.len(), 268, "Cone data must be 268 bytes");

    let cones_cf = db.cf_handle(CF_CONES).unwrap();
    db.put_cf(cones_cf, node_id, &cone_data).unwrap();

    println!("AFTER WRITE: Stored 268-byte entailment cone");

    // Read back and verify
    let result = db.get_cf(cones_cf, node_id).unwrap().unwrap();
    assert_eq!(result.len(), 268);

    println!("AFTER READ: Verified 268-byte entailment cone");
}

#[test]
fn test_real_rocksdb_write_faiss_id() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_faiss_ids.db");

    println!("BEFORE: Testing FAISS ID mapping storage");

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    // Store FAISS ID mapping (i64 = 8 bytes)
    let node_id: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let faiss_id: i64 = 42_000_000;
    let faiss_id_bytes = faiss_id.to_le_bytes();

    assert_eq!(faiss_id_bytes.len(), 8, "FAISS ID must be 8 bytes");

    let faiss_cf = db.cf_handle(CF_FAISS_IDS).unwrap();
    db.put_cf(faiss_cf, node_id, faiss_id_bytes).unwrap();

    println!("AFTER WRITE: Stored FAISS ID {}", faiss_id);

    // Read back and verify
    let result = db.get_cf(faiss_cf, node_id).unwrap().unwrap();
    let read_id = i64::from_le_bytes([
        result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7],
    ]);
    assert_eq!(read_id, faiss_id);

    println!("AFTER READ: Verified FAISS ID {}", read_id);
}

#[test]
fn test_real_rocksdb_reopen_preserves_data() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_reopen.db");

    println!("BEFORE: Testing data persistence across reopen");

    let db_opts = get_db_options();
    let config = StorageConfig::default();

    // First open: write data
    {
        let cf_descriptors = get_column_family_descriptors(&config).unwrap();
        let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

        let nodes_cf = db.cf_handle(CF_NODES).unwrap();
        db.put_cf(nodes_cf, b"node_id_1", b"node_data_persistent")
            .unwrap();

        println!("AFTER FIRST OPEN: Wrote node data");
    }

    // Second open: verify data persisted
    {
        let cf_descriptors = get_column_family_descriptors(&config).unwrap();
        let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

        let nodes_cf = db.cf_handle(CF_NODES).unwrap();
        let value = db.get_cf(nodes_cf, b"node_id_1").unwrap();
        assert_eq!(value, Some(b"node_data_persistent".to_vec()));

        println!("AFTER SECOND OPEN: Verified data persistence");
    }
}

#[test]
fn test_db_options_parallelism() {
    println!("BEFORE: Testing DB options with parallelism");

    let opts = get_db_options();

    // Verify options are valid by using them
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_opts.db");

    // Should succeed with our options
    let _db = rocksdb::DB::open(&opts, &db_path).unwrap();

    println!("AFTER: DB opened with parallelism options");
}
