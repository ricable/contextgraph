//! Column family descriptor generation for RocksDB.
//!
//! Creates optimized column family descriptors for each data type stored
//! in the graph storage backend.

use rocksdb::{
    BlockBasedOptions, Cache, ColumnFamilyDescriptor, DBCompressionType, Options, SliceTransform,
};

use super::config::StorageConfig;
use super::constants::*;
use crate::error::GraphResult;

/// Get column family descriptors for all graph storage CFs.
///
/// Creates optimized descriptors for each column family based on access patterns:
/// - Adjacency: prefix scans for edge lists
/// - Hyperbolic: point lookups for GPU batch loading
/// - Cones: bloom filter for hierarchy queries
/// - FAISS IDs: point lookups for ID mapping
/// - Nodes: point lookups for node data
/// - Metadata: small, infrequent access
///
/// # Arguments
///
/// * `config` - Storage configuration (validated before use)
///
/// # Returns
///
/// Vector of `ColumnFamilyDescriptor` for all column families.
/// Order matches `ALL_COLUMN_FAMILIES`.
///
/// # Errors
///
/// Returns `GraphError::InvalidConfig` if configuration validation fails.
///
/// # Example
///
/// ```ignore
/// use context_graph_graph::storage::{StorageConfig, get_column_family_descriptors, get_db_options};
///
/// let config = StorageConfig::default();
/// let cf_descriptors = get_column_family_descriptors(&config)?;
/// let db_opts = get_db_options();
/// let db = rocksdb::DB::open_cf_descriptors(&db_opts, "path", cf_descriptors)?;
/// ```
pub fn get_column_family_descriptors(
    config: &StorageConfig,
) -> GraphResult<Vec<ColumnFamilyDescriptor>> {
    // Validate config first - fail fast
    config.validate()?;

    // Create shared LRU cache for memory efficiency
    let cache = Cache::new_lru_cache(config.block_cache_size);

    Ok(vec![
        adjacency_cf_descriptor(config, &cache),
        hyperbolic_cf_descriptor(config, &cache),
        cones_cf_descriptor(config, &cache),
        faiss_ids_cf_descriptor(config, &cache),
        nodes_cf_descriptor(config, &cache),
        metadata_cf_descriptor(&cache),
        edges_cf_descriptor(config, &cache),
    ])
}

/// Get CF descriptor for adjacency column family.
/// Optimized for prefix scans (listing all edges from a node).
fn adjacency_cf_descriptor(config: &StorageConfig, cache: &Cache) -> ColumnFamilyDescriptor {
    let mut opts = Options::default();

    // Write settings
    opts.set_write_buffer_size(config.write_buffer_size);
    opts.set_max_write_buffer_number(config.max_write_buffers);
    opts.set_target_file_size_base(config.target_file_size_base);

    // Compression: LZ4 for fast decompression (GPU batch loading)
    if config.enable_compression {
        opts.set_compression_type(DBCompressionType::Lz4);
    }

    // Block-based table with shared cache
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_block_size(16 * 1024); // 16KB blocks for prefix scans

    // Bloom filter for point lookups within prefix
    block_opts.set_bloom_filter(config.bloom_filter_bits as f64, false);

    opts.set_block_based_table_factory(&block_opts);

    // Optimize for prefix scans (16-byte UUID keys)
    opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(16));

    ColumnFamilyDescriptor::new(CF_ADJACENCY, opts)
}

/// Get CF descriptor for hyperbolic coordinates.
/// Optimized for point lookups (256 bytes per point, GPU batch loading).
fn hyperbolic_cf_descriptor(config: &StorageConfig, cache: &Cache) -> ColumnFamilyDescriptor {
    let mut opts = Options::default();

    opts.set_write_buffer_size(config.write_buffer_size);
    opts.set_max_write_buffer_number(config.max_write_buffers);

    // LZ4 compression (256 bytes of floats compress well)
    if config.enable_compression {
        opts.set_compression_type(DBCompressionType::Lz4);
    }

    // Block-based table optimized for point lookups
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_block_size(4 * 1024); // Smaller blocks for point lookups

    // Strong bloom filter for fast negative lookups
    block_opts.set_bloom_filter(config.bloom_filter_bits as f64, false);

    opts.set_block_based_table_factory(&block_opts);

    // Optimize for point lookups
    opts.optimize_for_point_lookup(64); // 64MB block cache hint

    ColumnFamilyDescriptor::new(CF_HYPERBOLIC, opts)
}

/// Get CF descriptor for entailment cones.
/// Optimized for range scans with bloom filter (268 bytes per cone).
fn cones_cf_descriptor(config: &StorageConfig, cache: &Cache) -> ColumnFamilyDescriptor {
    let mut opts = Options::default();

    opts.set_write_buffer_size(config.write_buffer_size);
    opts.set_max_write_buffer_number(config.max_write_buffers);

    // LZ4 compression
    if config.enable_compression {
        opts.set_compression_type(DBCompressionType::Lz4);
    }

    // Block-based table with bloom filter
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_block_size(8 * 1024); // 8KB blocks

    // Bloom filter enabled for efficient cone lookups (per task spec)
    block_opts.set_bloom_filter(config.bloom_filter_bits as f64, false);
    block_opts.set_whole_key_filtering(true);

    opts.set_block_based_table_factory(&block_opts);

    ColumnFamilyDescriptor::new(CF_CONES, opts)
}

/// Get CF descriptor for FAISS ID mapping.
/// Optimized for point lookups (8 bytes per entry).
fn faiss_ids_cf_descriptor(config: &StorageConfig, cache: &Cache) -> ColumnFamilyDescriptor {
    let mut opts = Options::default();

    // Smaller buffers for small values (8 bytes each)
    opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB
    opts.set_max_write_buffer_number(2);

    // Block-based table
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_block_size(4 * 1024); // 4KB blocks

    // Bloom filter for fast lookups
    block_opts.set_bloom_filter(config.bloom_filter_bits as f64, false);

    opts.set_block_based_table_factory(&block_opts);

    ColumnFamilyDescriptor::new(CF_FAISS_IDS, opts)
}

/// Get CF descriptor for node data.
/// Optimized for point lookups with variable-size values.
fn nodes_cf_descriptor(config: &StorageConfig, cache: &Cache) -> ColumnFamilyDescriptor {
    let mut opts = Options::default();

    opts.set_write_buffer_size(config.write_buffer_size);
    opts.set_max_write_buffer_number(config.max_write_buffers);
    opts.set_target_file_size_base(config.target_file_size_base);

    // LZ4 compression for variable-size node data
    if config.enable_compression {
        opts.set_compression_type(DBCompressionType::Lz4);
    }

    // Block-based table
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_block_size(8 * 1024); // 8KB blocks

    // Bloom filter for point lookups
    block_opts.set_bloom_filter(config.bloom_filter_bits as f64, false);

    opts.set_block_based_table_factory(&block_opts);

    ColumnFamilyDescriptor::new(CF_NODES, opts)
}

/// Get CF descriptor for metadata.
/// Small CF for schema version, statistics, etc.
fn metadata_cf_descriptor(cache: &Cache) -> ColumnFamilyDescriptor {
    let mut opts = Options::default();

    // Minimal write buffer for small metadata
    opts.set_write_buffer_size(4 * 1024 * 1024); // 4MB
    opts.set_max_write_buffer_number(2);

    // Block-based table
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_block_size(4 * 1024);

    opts.set_block_based_table_factory(&block_opts);

    ColumnFamilyDescriptor::new(CF_METADATA, opts)
}

/// Get CF descriptor for full GraphEdge storage (M04-T15).
/// Optimized for point lookups by edge ID.
/// Values are bincode-serialized GraphEdge (~200-300 bytes).
fn edges_cf_descriptor(config: &StorageConfig, cache: &Cache) -> ColumnFamilyDescriptor {
    let mut opts = Options::default();

    opts.set_write_buffer_size(config.write_buffer_size);
    opts.set_max_write_buffer_number(config.max_write_buffers);
    opts.set_target_file_size_base(config.target_file_size_base);

    // LZ4 compression for good compression ratio on serialized edges
    if config.enable_compression {
        opts.set_compression_type(DBCompressionType::Lz4);
    }

    // Block-based table optimized for point lookups
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_block_size(8 * 1024); // 8KB blocks

    // Bloom filter for fast negative lookups
    block_opts.set_bloom_filter(config.bloom_filter_bits as f64, false);

    opts.set_block_based_table_factory(&block_opts);

    ColumnFamilyDescriptor::new(CF_EDGES, opts)
}

/// Get default DB options for opening the database.
///
/// Configures parallelism based on CPU count and sets reasonable defaults
/// for production use. Optimized for systems with high core counts
/// (e.g., Ryzen 9 9950X3D with 16 cores / 32 threads).
///
/// # Constitution Reference
///
/// - stack.lang.rust: 1.75+
/// - AP-004: Avoid blocking I/O in async
#[must_use]
pub fn get_db_options() -> Options {
    let mut opts = Options::default();

    opts.create_if_missing(true);
    opts.create_missing_column_families(true);
    opts.set_max_open_files(1000);
    opts.set_keep_log_file_num(10);

    // Parallelism based on available CPUs
    // Ryzen 9 9950X3D: 16 cores / 32 threads
    let cpu_count = num_cpus::get() as i32;

    // Use at least 2 threads, scale with CPU count
    let parallelism = cpu_count.max(2);
    opts.increase_parallelism(parallelism);

    // Background jobs: min 2, max based on CPU count (cap at reasonable level)
    let bg_jobs = cpu_count.clamp(2, 8);
    opts.set_max_background_jobs(bg_jobs);

    opts
}
