//! Core RocksDbMemex struct and database operations.
//!
//! This module provides the main `RocksDbMemex` struct, which is the primary
//! storage implementation for the Context Graph system. It wraps RocksDB
//! with optimized column families for nodes, edges, embeddings, and indexes.
//!
//! # Architecture
//!
//! ```text
//! RocksDbMemex
//! ├── DB (RocksDB instance)
//! │   ├── CF: nodes        - Primary node storage
//! │   ├── CF: edges        - Graph edge storage
//! │   ├── CF: embeddings   - Vector embeddings (1536D)
//! │   ├── CF: johari_*     - Quadrant indexes (4 CFs)
//! │   ├── CF: temporal     - Time-based index
//! │   ├── CF: tags         - Tag index
//! │   ├── CF: sources      - Source index
//! │   └── CF: system       - Metadata storage
//! └── Cache (LRU block cache, 256MB default)
//! ```
//!
//! # Usage
//!
//! ```rust
//! use context_graph_storage::RocksDbMemex;
//! use tempfile::TempDir;
//!
//! let tmp = TempDir::new().unwrap();
//! let memex = RocksDbMemex::open(tmp.path()).unwrap();
//!
//! // Check health
//! memex.health_check().unwrap();
//!
//! // Flush all data to disk
//! memex.flush_all().unwrap();
//! ```

use rocksdb::{Cache, ColumnFamily, Options, DB};
use std::path::Path;

use crate::column_families::{cf_names, get_all_column_family_descriptors};

use super::config::RocksDbConfig;
use super::error::StorageError;

/// RocksDB-backed storage implementation for the Context Graph system.
///
/// `RocksDbMemex` provides persistent storage for `MemoryNode` and `GraphEdge`
/// entities with optimized column families for different access patterns.
/// It implements the [`Memex`](crate::Memex) trait for abstraction.
///
/// # Thread Safety
///
/// RocksDB's `DB` type is internally thread-safe for concurrent reads and writes.
/// This struct can be safely shared across threads via `Arc<RocksDbMemex>`.
/// All methods take `&self` (not `&mut self`) because RocksDB handles locking internally.
///
/// # Column Families
///
/// Opens 12 column families with optimized settings:
///
/// | Column Family | Purpose | Block Size |
/// |--------------|---------|------------|
/// | `nodes` | Primary node storage | 16KB |
/// | `edges` | Graph edge storage | 16KB |
/// | `embeddings` | Vector embeddings | 64KB (write-optimized) |
/// | `johari_open/blind/hidden/unknown` | Quadrant indexes | 4KB |
/// | `temporal` | Time-based index | 4KB |
/// | `tags` | Tag index | 4KB |
/// | `sources` | Source index | 4KB |
/// | `system` | Metadata storage | 4KB |
///
/// # Performance
///
/// - Block cache: 256MB default (configurable via `RocksDbConfig`)
/// - WAL enabled by default for durability
/// - Bloom filters on index column families
///
/// # Example: Basic Usage
///
/// ```rust
/// use context_graph_storage::{RocksDbMemex, Memex, MemoryNode, JohariQuadrant};
/// use tempfile::TempDir;
///
/// // Create database
/// let tmp = TempDir::new().unwrap();
/// let memex = RocksDbMemex::open(tmp.path()).unwrap();
///
/// // Create and store a node
/// let dim = 1536;
/// let val = 1.0_f32 / (dim as f32).sqrt();
/// let embedding = vec![val; dim];
///
/// let mut node = MemoryNode::new("Hello, RocksDB!".to_string(), embedding);
/// node.quadrant = JohariQuadrant::Open;
/// memex.store_node(&node).unwrap();
///
/// // Retrieve the node
/// let retrieved = memex.get_node(&node.id).unwrap();
/// assert_eq!(retrieved.content, "Hello, RocksDB!");
/// ```
///
/// # Example: Custom Configuration
///
/// ```rust
/// use context_graph_storage::{RocksDbMemex, RocksDbConfig};
/// use tempfile::TempDir;
///
/// let tmp = TempDir::new().unwrap();
/// let config = RocksDbConfig {
///     block_cache_size: 512 * 1024 * 1024, // 512MB cache
///     max_open_files: 2000,
///     enable_wal: true,
///     create_if_missing: true,
/// };
///
/// let memex = RocksDbMemex::open_with_config(tmp.path(), config).unwrap();
/// memex.health_check().unwrap();
/// ```
pub struct RocksDbMemex {
    /// The RocksDB database instance.
    ///
    /// Accessed via `self.db` for all RocksDB operations. Thread-safe
    /// for concurrent reads and writes.
    pub(crate) db: DB,

    /// Shared LRU block cache.
    ///
    /// Kept alive for the database lifetime. The cache is shared across
    /// all column families to optimize memory usage.
    #[allow(dead_code)]
    cache: Cache,

    /// Database path for reference and logging.
    ///
    /// Stores the path as a string for error messages and diagnostics.
    path: String,
}

impl RocksDbMemex {
    /// Opens a RocksDB database at the specified path with default configuration.
    ///
    /// Creates the database directory and all 12 column families if they don't exist.
    /// Uses default configuration: 256MB cache, 1000 max open files, WAL enabled.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the database directory. Will be created if it doesn't exist.
    ///
    /// # Returns
    ///
    /// * `Ok(RocksDbMemex)` - Successfully opened database
    /// * `Err(StorageError::OpenFailed)` - Database could not be opened
    ///
    /// # Errors
    ///
    /// * `StorageError::OpenFailed` - Path is invalid, permissions denied,
    ///   database is locked, or disk is full
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::RocksDbMemex;
    /// use tempfile::TempDir;
    ///
    /// let tmp = TempDir::new().unwrap();
    /// let memex = RocksDbMemex::open(tmp.path()).unwrap();
    ///
    /// // Database is ready for use
    /// memex.health_check().unwrap();
    /// ```
    ///
    /// `Constraint: latency < 100ms for new DB, < 500ms for large existing DB`
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, StorageError> {
        Self::open_with_config(path, RocksDbConfig::default())
    }

    /// Opens a RocksDB database with custom configuration.
    ///
    /// Allows fine-grained control over RocksDB settings like cache size,
    /// file handles, and WAL behavior.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the database directory
    /// * `config` - Custom configuration options (see [`RocksDbConfig`])
    ///
    /// # Returns
    ///
    /// * `Ok(RocksDbMemex)` - Successfully opened database
    /// * `Err(StorageError::OpenFailed)` - Database could not be opened
    ///
    /// # Errors
    ///
    /// * `StorageError::OpenFailed` - Path is invalid, permissions denied,
    ///   database is locked, disk is full, or config is incompatible
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::{RocksDbMemex, RocksDbConfig};
    /// use tempfile::TempDir;
    ///
    /// let tmp = TempDir::new().unwrap();
    ///
    /// // High-performance configuration
    /// let config = RocksDbConfig {
    ///     block_cache_size: 1024 * 1024 * 1024, // 1GB cache
    ///     max_open_files: 5000,
    ///     enable_wal: true, // Durability
    ///     create_if_missing: true,
    /// };
    ///
    /// let memex = RocksDbMemex::open_with_config(tmp.path(), config).unwrap();
    /// ```
    ///
    /// `Constraint: latency < 100ms for new DB, < 500ms for large existing DB`
    pub fn open_with_config<P: AsRef<Path>>(
        path: P,
        config: RocksDbConfig,
    ) -> Result<Self, StorageError> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Create shared block cache
        let cache = Cache::new_lru_cache(config.block_cache_size);

        // Create DB options
        let mut db_opts = Options::default();
        db_opts.create_if_missing(config.create_if_missing);
        db_opts.create_missing_column_families(true);
        db_opts.set_max_open_files(config.max_open_files);

        // WAL configuration
        if !config.enable_wal {
            db_opts.set_manual_wal_flush(true);
        }

        // Get ALL column family descriptors: base (12) + teleological (11+13) + autonomous (7)
        // This ensures CF_SESSION_IDENTITY and other advanced CFs are available
        let cf_descriptors = get_all_column_family_descriptors(&cache);

        // Open database with all column families
        let db = DB::open_cf_descriptors(&db_opts, &path_str, cf_descriptors).map_err(|e| {
            StorageError::OpenFailed {
                path: path_str.clone(),
                message: e.to_string(),
            }
        })?;

        Ok(Self {
            db,
            cache,
            path: path_str,
        })
    }

    /// Gets a reference to a column family by name.
    ///
    /// Used internally to access specific column families for operations.
    /// External callers should use the high-level API methods instead.
    ///
    /// # Arguments
    ///
    /// * `name` - Column family name. Use `cf_names::*` constants for type safety.
    ///
    /// # Returns
    ///
    /// * `Ok(&ColumnFamily)` - Reference to the column family handle
    /// * `Err(StorageError::ColumnFamilyNotFound)` - CF doesn't exist
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::{RocksDbMemex, cf_names};
    /// use tempfile::TempDir;
    ///
    /// let tmp = TempDir::new().unwrap();
    /// let memex = RocksDbMemex::open(tmp.path()).unwrap();
    ///
    /// // Access the nodes column family
    /// let cf = memex.get_cf(cf_names::NODES).unwrap();
    /// ```
    pub fn get_cf(&self, name: &str) -> Result<&ColumnFamily, StorageError> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| StorageError::ColumnFamilyNotFound {
                name: name.to_string(),
            })
    }

    /// Gets the database path.
    ///
    /// Returns the path where the database files are stored, useful for
    /// logging and diagnostics.
    ///
    /// # Returns
    ///
    /// The path string where the database is stored.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::RocksDbMemex;
    /// use tempfile::TempDir;
    ///
    /// let tmp = TempDir::new().unwrap();
    /// let memex = RocksDbMemex::open(tmp.path()).unwrap();
    ///
    /// println!("Database at: {}", memex.path());
    /// ```
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Checks if the database is healthy.
    ///
    /// Verifies that all 12 column families are accessible. This is a
    /// lightweight check that doesn't scan data.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - All column families are accessible
    /// * `Err(StorageError::ColumnFamilyNotFound)` - A column family is missing
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::RocksDbMemex;
    /// use tempfile::TempDir;
    ///
    /// let tmp = TempDir::new().unwrap();
    /// let memex = RocksDbMemex::open(tmp.path()).unwrap();
    ///
    /// // Verify database health before operations
    /// if memex.health_check().is_ok() {
    ///     println!("Database is healthy");
    /// }
    /// ```
    ///
    /// `Constraint: latency < 1ms`
    pub fn health_check(&self) -> Result<(), StorageError> {
        for cf_name in cf_names::ALL {
            self.get_cf(cf_name)?;
        }
        Ok(())
    }

    /// Flushes all column families to disk.
    ///
    /// Forces all buffered writes in the memtable to be persisted to SST files.
    /// Useful before shutdown or when durability is critical.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - All column families flushed successfully
    /// * `Err(StorageError::FlushFailed)` - Flush operation failed
    ///
    /// # Errors
    ///
    /// * `StorageError::FlushFailed` - Disk I/O error, disk full, or
    ///   database is shutting down
    ///
    /// # Performance
    ///
    /// This is an I/O-intensive operation. Avoid calling frequently in
    /// hot paths. RocksDB automatically flushes based on memtable size.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::RocksDbMemex;
    /// use tempfile::TempDir;
    ///
    /// let tmp = TempDir::new().unwrap();
    /// let memex = RocksDbMemex::open(tmp.path()).unwrap();
    ///
    /// // ... perform writes ...
    ///
    /// // Ensure all writes are persisted before shutdown
    /// memex.flush_all().unwrap();
    /// ```
    ///
    /// `Constraint: latency < 500ms for typical workload`
    pub fn flush_all(&self) -> Result<(), StorageError> {
        for cf_name in cf_names::ALL {
            let cf = self.get_cf(cf_name)?;
            self.db
                .flush_cf(cf)
                .map_err(|e| StorageError::FlushFailed(e.to_string()))?;
        }
        Ok(())
    }

    /// Gets a reference to the underlying RocksDB instance.
    ///
    /// Use this for advanced operations not covered by the high-level API.
    /// Be careful not to violate data invariants or column family schemas.
    ///
    /// # Returns
    ///
    /// A reference to the RocksDB `DB` instance.
    ///
    /// # Warning
    ///
    /// Direct DB access bypasses validation and index maintenance.
    /// Only use for operations like statistics gathering or manual repairs.
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_storage::RocksDbMemex;
    /// use tempfile::TempDir;
    ///
    /// let tmp = TempDir::new().unwrap();
    /// let memex = RocksDbMemex::open(tmp.path()).unwrap();
    ///
    /// // Get RocksDB statistics (advanced usage)
    /// let db = memex.db();
    /// // db.property_value("rocksdb.stats") etc.
    /// ```
    /// Get a reference to the underlying RocksDB instance FOR TESTING/DIAGNOSTICS ONLY.
    ///
    /// # Constitution Compliance
    ///
    /// ARCH-06: "All Memory Operations Through MCP Tools - Direct database access forbidden"
    ///
    /// This method is intended ONLY for:
    /// - Integration tests performing Full State Verification
    /// - Diagnostics and debugging
    /// - Health checks
    ///
    /// **DO NOT** use this method in production MCP handlers or application code.
    /// All memory operations should go through the proper MCP tool interface.
    ///
    /// # Warning
    ///
    /// Direct DB access bypasses validation. Improper use violates ARCH-06.
    #[doc(hidden)]
    pub fn db(&self) -> &DB {
        &self.db
    }
}

// DB is automatically closed when RocksDbMemex is dropped (RocksDB's Drop impl)
