//! Memory store trait for persistent storage.

use async_trait::async_trait;
use std::path::{Path, PathBuf};

use crate::error::CoreResult;
use crate::types::{JohariQuadrant, MemoryNode, Modality, NodeId};

// =========================================================================
// Storage Backend Types
// =========================================================================

/// Storage backend type indicator.
///
/// Identifies the underlying storage implementation being used.
/// This allows code to make decisions based on backend capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageBackend {
    /// In-memory storage (development/testing only).
    /// Data is lost when the process exits.
    InMemory,
    /// SQLite file-based storage (development).
    /// Single-file database, good for local development.
    SQLite,
    /// RocksDB embedded database (production).
    /// High-performance key-value store with LSM-tree.
    RocksDB,
    /// PostgreSQL with pgvector (production, distributed).
    /// Full SQL database with vector similarity extensions.
    PostgreSQL,
}

/// Configuration for storage backend selection.
///
/// Provides all necessary parameters to initialize a storage backend.
///
/// # Examples
///
/// ```
/// use context_graph_core::traits::StorageConfig;
/// use std::path::PathBuf;
///
/// // Default in-memory config
/// let config = StorageConfig::default();
/// assert_eq!(config.cache_size_mb, 64);
///
/// // RocksDB config
/// let rocks_config = StorageConfig::rocksdb("/var/data/contextgraph");
/// assert!(rocks_config.sync_writes);
/// ```
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Selected storage backend.
    pub backend: StorageBackend,
    /// Path to storage (file path for SQLite/RocksDB, connection string for PostgreSQL).
    pub path: Option<PathBuf>,
    /// Cache size in megabytes.
    pub cache_size_mb: usize,
    /// Whether to synchronize writes immediately.
    /// If true, ensures durability at cost of performance.
    pub sync_writes: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackend::InMemory,
            path: None,
            cache_size_mb: 64,
            sync_writes: false,
        }
    }
}

impl StorageConfig {
    /// Create configuration for RocksDB backend.
    ///
    /// Uses production-appropriate defaults:
    /// - 256 MB cache
    /// - Synchronous writes enabled
    pub fn rocksdb(path: impl Into<PathBuf>) -> Self {
        Self {
            backend: StorageBackend::RocksDB,
            path: Some(path.into()),
            cache_size_mb: 256,
            sync_writes: true,
        }
    }

    /// Create configuration for SQLite backend.
    ///
    /// Uses development-appropriate defaults:
    /// - 64 MB cache
    /// - Synchronous writes disabled for speed
    pub fn sqlite(path: impl Into<PathBuf>) -> Self {
        Self {
            backend: StorageBackend::SQLite,
            path: Some(path.into()),
            cache_size_mb: 64,
            sync_writes: false,
        }
    }

    /// Create configuration for PostgreSQL backend.
    ///
    /// Uses production-appropriate defaults:
    /// - 128 MB cache
    /// - Synchronous writes enabled
    pub fn postgresql(connection_string: impl Into<PathBuf>) -> Self {
        Self {
            backend: StorageBackend::PostgreSQL,
            path: Some(connection_string.into()),
            cache_size_mb: 128,
            sync_writes: true,
        }
    }
}

/// Query options for memory search.
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    /// Maximum results to return
    pub top_k: usize,
    /// Minimum similarity threshold [0.0, 1.0]
    pub min_similarity: Option<f32>,
    /// Filter by Johari quadrant
    pub johari_filter: Option<JohariQuadrant>,
    /// Filter by modality
    pub modality_filter: Option<Modality>,
    /// Include soft-deleted nodes
    pub include_deleted: bool,
}

impl SearchOptions {
    /// Create new search options with the given top_k.
    pub fn new(top_k: usize) -> Self {
        Self {
            top_k,
            ..Default::default()
        }
    }

    /// Set minimum similarity threshold.
    pub fn with_min_similarity(mut self, threshold: f32) -> Self {
        self.min_similarity = Some(threshold);
        self
    }

    /// Filter by Johari quadrant.
    pub fn with_johari_filter(mut self, quadrant: JohariQuadrant) -> Self {
        self.johari_filter = Some(quadrant);
        self
    }
}

/// Persistent memory storage abstraction.
///
/// Provides CRUD operations for memory nodes with vector search capability.
///
/// # Example
///
/// ```
/// use context_graph_core::traits::SearchOptions;
/// use context_graph_core::types::JohariQuadrant;
///
/// // Create search options for memory query
/// let options = SearchOptions::new(10)
///     .with_min_similarity(0.8)
///     .with_johari_filter(JohariQuadrant::Open);
/// assert_eq!(options.top_k, 10);
/// assert_eq!(options.min_similarity, Some(0.8));
/// ```
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Store a memory node, returning its ID.
    async fn store(&self, node: MemoryNode) -> CoreResult<NodeId>;

    /// Retrieve a node by ID, returns None if not found.
    async fn retrieve(&self, id: NodeId) -> CoreResult<Option<MemoryNode>>;

    /// Search for nodes by semantic similarity.
    ///
    /// Returns nodes with their similarity scores.
    async fn search(
        &self,
        query_embedding: &[f32],
        options: SearchOptions,
    ) -> CoreResult<Vec<(MemoryNode, f32)>>;

    /// Search by text query (embedding computed internally).
    async fn search_text(
        &self,
        query: &str,
        options: SearchOptions,
    ) -> CoreResult<Vec<(MemoryNode, f32)>>;

    /// Delete a node (soft or hard delete).
    ///
    /// Returns true if the node was found and deleted.
    async fn delete(&self, id: NodeId, soft: bool) -> CoreResult<bool>;

    /// Update an existing node.
    ///
    /// Returns true if the node was found and updated.
    async fn update(&self, node: MemoryNode) -> CoreResult<bool>;

    /// Get total node count (excluding soft-deleted).
    async fn count(&self) -> CoreResult<usize>;

    /// Compact storage (remove tombstones, optimize indices).
    async fn compact(&self) -> CoreResult<()>;

    // =========================================================================
    // Persistence Operations
    // =========================================================================

    /// Flush any buffered writes to stable storage.
    ///
    /// For in-memory backends, this is a no-op.
    /// For disk-backed backends, ensures all writes are persisted.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::StorageError` if flush fails due to I/O issues.
    async fn flush(&self) -> CoreResult<()>;

    /// Create a checkpoint/snapshot for recovery.
    ///
    /// Returns the path to the checkpoint. For in-memory backends,
    /// serializes to a temp file for testing purposes.
    ///
    /// # Returns
    ///
    /// Path to the checkpoint that can be used with `restore()`.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::StorageError` if checkpoint creation fails.
    async fn checkpoint(&self) -> CoreResult<PathBuf>;

    /// Restore from a previously created checkpoint.
    ///
    /// Replaces current store contents with checkpoint data.
    ///
    /// # Arguments
    ///
    /// * `checkpoint` - Path returned by a previous `checkpoint()` call
    ///
    /// # Errors
    ///
    /// Returns `CoreError::StorageError` if:
    /// - Checkpoint file does not exist
    /// - Checkpoint data is corrupted
    /// - I/O error during restore
    async fn restore(&self, checkpoint: &Path) -> CoreResult<()>;

    // =========================================================================
    // Statistics (Sync for efficiency)
    // =========================================================================

    /// Get the current node count synchronously.
    ///
    /// This is a sync method for efficiency in hot paths.
    /// May return an approximate count for large stores.
    /// Excludes soft-deleted nodes.
    fn node_count_sync(&self) -> usize;

    /// Get the approximate storage size in bytes.
    ///
    /// For in-memory backends, returns estimated memory usage.
    /// For disk backends, returns actual disk usage.
    fn storage_size_bytes(&self) -> usize;

    /// Get the backend type this store uses.
    ///
    /// Allows callers to make decisions based on backend capabilities.
    fn backend_type(&self) -> StorageBackend;
}
