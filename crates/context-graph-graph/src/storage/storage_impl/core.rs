//! Core GraphStorage struct and initialization.
//!
//! Provides the main `GraphStorage` struct and open/migration methods.

use std::path::Path;
use std::sync::Arc;

use rocksdb::{ColumnFamily, DB};

use crate::error::{GraphError, GraphResult};
use crate::storage::{
    get_column_family_descriptors, get_db_options, StorageConfig, CF_ADJACENCY, CF_CONES, CF_EDGES,
    CF_HYPERBOLIC, CF_METADATA,
};

/// Graph storage backed by RocksDB.
///
/// Thread-safe via Arc<DB>. Clone is cheap (Arc clone).
///
/// # Column Families
///
/// - `hyperbolic`: Poincare coordinates (256 bytes per node)
/// - `entailment_cones`: Entailment cones (268 bytes per node)
/// - `adjacency`: Edge lists (bincode serialized)
/// - `metadata`: Schema version and statistics
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_graph::storage::{GraphStorage, StorageConfig, PoincarePoint};
///
/// let storage = GraphStorage::open_default("/tmp/graph.db")?;
/// storage.put_hyperbolic(1, &PoincarePoint::origin())?;
/// let point = storage.get_hyperbolic(1)?;
/// assert!(point.is_some());
/// ```
#[derive(Clone)]
pub struct GraphStorage {
    pub(crate) db: Arc<DB>,
}

impl GraphStorage {
    /// Open graph storage at the given path.
    ///
    /// # Arguments
    /// * `path` - Directory path for RocksDB database
    /// * `config` - Storage configuration (use StorageConfig::default())
    ///
    /// # Errors
    /// * `GraphError::StorageOpen` - Failed to open database
    /// * `GraphError::InvalidConfig` - Invalid configuration
    ///
    /// # Example
    /// ```rust,ignore
    /// let storage = GraphStorage::open("/data/graph.db", StorageConfig::default())?;
    /// ```
    pub fn open<P: AsRef<Path>>(path: P, config: StorageConfig) -> GraphResult<Self> {
        let db_opts = get_db_options();
        let cf_descriptors = get_column_family_descriptors(&config)?;

        let db = DB::open_cf_descriptors(&db_opts, path.as_ref(), cf_descriptors).map_err(|e| {
            log::error!("Failed to open GraphStorage at {:?}: {}", path.as_ref(), e);
            GraphError::StorageOpen {
                path: path.as_ref().to_string_lossy().into_owned(),
                cause: e.to_string(),
            }
        })?;

        log::info!("GraphStorage opened at {:?}", path.as_ref());

        Ok(Self { db: Arc::new(db) })
    }

    /// Open with default configuration.
    pub fn open_default<P: AsRef<Path>>(path: P) -> GraphResult<Self> {
        Self::open(path, StorageConfig::default())
    }

    /// Open storage and apply migrations.
    ///
    /// Convenience method that combines open() with migrations.
    /// This is the recommended way to open a database in production.
    ///
    /// # Example
    /// ```rust,ignore
    /// let storage = GraphStorage::open_and_migrate(
    ///     "/data/graph.db",
    ///     StorageConfig::default(),
    /// )?;
    /// // Database is now at latest schema version
    /// ```
    pub fn open_and_migrate<P: AsRef<Path>>(path: P, config: StorageConfig) -> GraphResult<Self> {
        log::info!("Opening storage with migrations at {:?}", path.as_ref());

        let storage = Self::open(path, config)?;

        let before_version = storage.get_schema_version()?;
        let after_version = storage.apply_migrations()?;

        log::info!(
            "Storage ready: migrated from v{} to v{}",
            before_version,
            after_version
        );

        Ok(storage)
    }

    // ========== Column Family Helpers ==========

    pub(crate) fn cf_hyperbolic(&self) -> GraphResult<&ColumnFamily> {
        self.db
            .cf_handle(CF_HYPERBOLIC)
            .ok_or_else(|| GraphError::ColumnFamilyNotFound(CF_HYPERBOLIC.to_string()))
    }

    pub(crate) fn cf_cones(&self) -> GraphResult<&ColumnFamily> {
        self.db
            .cf_handle(CF_CONES)
            .ok_or_else(|| GraphError::ColumnFamilyNotFound(CF_CONES.to_string()))
    }

    pub(crate) fn cf_adjacency(&self) -> GraphResult<&ColumnFamily> {
        self.db
            .cf_handle(CF_ADJACENCY)
            .ok_or_else(|| GraphError::ColumnFamilyNotFound(CF_ADJACENCY.to_string()))
    }

    pub(crate) fn cf_metadata(&self) -> GraphResult<&ColumnFamily> {
        self.db
            .cf_handle(CF_METADATA)
            .ok_or_else(|| GraphError::ColumnFamilyNotFound(CF_METADATA.to_string()))
    }

    /// Get column family handle for edges (M04-T15).
    pub(crate) fn cf_edges(&self) -> GraphResult<&ColumnFamily> {
        self.db
            .cf_handle(CF_EDGES)
            .ok_or_else(|| GraphError::ColumnFamilyNotFound(CF_EDGES.to_string()))
    }
}

impl std::fmt::Debug for GraphStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphStorage")
            .field("hyperbolic_count", &self.hyperbolic_count().unwrap_or(0))
            .field("cone_count", &self.cone_count().unwrap_or(0))
            .field("adjacency_count", &self.adjacency_count().unwrap_or(0))
            .field("edge_count", &self.edge_count().unwrap_or(0))
            .finish()
    }
}
