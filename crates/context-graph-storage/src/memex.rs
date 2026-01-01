//! Memex storage trait abstraction.
//!
//! The Memex trait defines the storage contract for MemoryNode and GraphEdge
//! persistence. Named after Vannevar Bush's conceptual memory machine.
//!
//! # Implementors
//! - `RocksDbMemex`: Production RocksDB implementation
//!
//! # Constitution Reference
//! - SEC-06: All delete operations must be soft deletes with 30-day recovery
//! - AP-010: store_memory requires rationale

use context_graph_core::marblestone::EdgeType;
use context_graph_core::types::{
    EmbeddingVector, GraphEdge, JohariQuadrant, MemoryNode, NodeId,
};

use crate::rocksdb_backend::StorageError;

/// Storage health status.
///
/// Returned by `Memex::health_check()` to provide storage metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct StorageHealth {
    /// Whether storage is operational
    pub is_healthy: bool,
    /// Approximate number of nodes (may be estimate)
    pub node_count: u64,
    /// Approximate number of edges (may be estimate)
    pub edge_count: u64,
    /// Approximate storage size in bytes
    pub storage_bytes: u64,
}

impl Default for StorageHealth {
    fn default() -> Self {
        Self {
            is_healthy: true,
            node_count: 0,
            edge_count: 0,
            storage_bytes: 0,
        }
    }
}

/// Storage abstraction trait for the Context Graph system.
///
/// This trait defines the core storage operations required by the system.
/// RocksDbMemex implements this trait, enabling:
/// 1. **Testing**: In-memory implementation for fast unit tests
/// 2. **Flexibility**: Future distributed storage backends
/// 3. **Mocking**: Easy to mock for integration tests
/// 4. **Dependency Injection**: Higher layers depend on trait, not concrete type
///
/// # Object Safety
/// This trait is object-safe and can be used with `dyn Memex`.
/// All methods take `&self` and return concrete types (no generics, no `Self` in return).
///
/// # Thread Safety
/// Implementors MUST be `Send + Sync` for cross-thread usage.
pub trait Memex: Send + Sync {
    // === Node Operations ===

    /// Stores a memory node.
    ///
    /// Validates node before storage. Writes atomically to all relevant
    /// column families (nodes, embeddings, johari, temporal, tags, sources).
    ///
    /// # Errors
    /// - `StorageError::ValidationFailed` if node.validate() fails
    /// - `StorageError::Serialization` if serialization fails
    /// - `StorageError::WriteFailed` if storage write fails
    fn store_node(&self, node: &MemoryNode) -> Result<(), StorageError>;

    /// Retrieves a memory node by ID.
    ///
    /// # Errors
    /// - `StorageError::NotFound` if node doesn't exist
    /// - `StorageError::Serialization` if deserialization fails
    fn get_node(&self, id: &NodeId) -> Result<MemoryNode, StorageError>;

    /// Updates an existing memory node.
    ///
    /// Maintains index consistency when quadrant or tags change.
    /// DOES NOT create if node doesn't exist.
    ///
    /// # Errors
    /// - `StorageError::NotFound` if node doesn't exist
    /// - `StorageError::ValidationFailed` if node.validate() fails
    fn update_node(&self, node: &MemoryNode) -> Result<(), StorageError>;

    /// Deletes a memory node.
    ///
    /// # Arguments
    /// * `id` - Node ID to delete
    /// * `soft_delete` - If true, marks as deleted (SEC-06); if false, permanently removes
    ///
    /// # Errors
    /// - `StorageError::NotFound` if node doesn't exist
    fn delete_node(&self, id: &NodeId, soft_delete: bool) -> Result<(), StorageError>;

    // === Edge Operations ===

    /// Stores a graph edge.
    ///
    /// # Errors
    /// - `StorageError::Serialization` if serialization fails
    /// - `StorageError::WriteFailed` if storage write fails
    fn store_edge(&self, edge: &GraphEdge) -> Result<(), StorageError>;

    /// Retrieves a graph edge by composite key.
    ///
    /// # Errors
    /// - `StorageError::NotFound` if edge doesn't exist
    fn get_edge(
        &self,
        source_id: &NodeId,
        target_id: &NodeId,
        edge_type: EdgeType,
    ) -> Result<GraphEdge, StorageError>;

    /// Gets all outgoing edges from a node.
    ///
    /// Uses prefix scan for efficiency.
    fn get_edges_from(&self, source_id: &NodeId) -> Result<Vec<GraphEdge>, StorageError>;

    /// Gets all incoming edges to a node.
    ///
    /// Note: Full scan - less efficient than get_edges_from.
    fn get_edges_to(&self, target_id: &NodeId) -> Result<Vec<GraphEdge>, StorageError>;

    // === Query Operations ===

    /// Queries nodes by Johari quadrant.
    ///
    /// # Arguments
    /// * `quadrant` - The Johari quadrant to query
    /// * `limit` - Maximum results (None = unlimited)
    fn query_by_quadrant(
        &self,
        quadrant: JohariQuadrant,
        limit: Option<usize>,
    ) -> Result<Vec<NodeId>, StorageError>;

    /// Queries nodes by tag.
    ///
    /// # Arguments
    /// * `tag` - Tag to search for (exact match)
    /// * `limit` - Maximum results (None = unlimited)
    fn query_by_tag(
        &self,
        tag: &str,
        limit: Option<usize>,
    ) -> Result<Vec<NodeId>, StorageError>;

    // === Embedding Operations ===

    /// Retrieves an embedding by node ID.
    ///
    /// # Errors
    /// - `StorageError::NotFound` if no embedding for this node
    fn get_embedding(&self, id: &NodeId) -> Result<EmbeddingVector, StorageError>;

    // === Health ===

    /// Checks storage health and returns metrics.
    ///
    /// Should verify all storage components are accessible.
    fn health_check(&self) -> Result<StorageHealth, StorageError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RocksDbMemex;
    use tempfile::TempDir;

    #[test]
    fn test_trait_object_safe() {
        // Verify trait is object-safe by creating Box<dyn Memex>
        println!("=== TRAIT OBJECT SAFETY TEST ===");
        println!("BEFORE: Attempting to create Box<dyn Memex>");

        let tmp = TempDir::new().expect("create temp dir");
        let db = RocksDbMemex::open(tmp.path()).expect("open db");

        // This line compiles only if trait is object-safe
        let _boxed: Box<dyn Memex> = Box::new(db);

        println!("AFTER: Box<dyn Memex> created successfully");
        println!("RESULT: PASS - Trait is object-safe");
    }

    #[test]
    fn test_storage_health_default() {
        println!("=== STORAGE HEALTH DEFAULT TEST ===");
        let health = StorageHealth::default();

        println!("BEFORE: Creating default StorageHealth");
        println!("AFTER: is_healthy={}, node_count={}, edge_count={}, storage_bytes={}",
            health.is_healthy, health.node_count, health.edge_count, health.storage_bytes);

        assert!(health.is_healthy);
        assert_eq!(health.node_count, 0);
        assert_eq!(health.edge_count, 0);
        assert_eq!(health.storage_bytes, 0);
        println!("RESULT: PASS - StorageHealth::default() works correctly");
    }

    #[test]
    fn test_storage_health_debug_clone() {
        println!("=== STORAGE HEALTH DEBUG/CLONE TEST ===");
        let health = StorageHealth {
            is_healthy: true,
            node_count: 100,
            edge_count: 50,
            storage_bytes: 1024,
        };

        println!("BEFORE: Creating custom StorageHealth");
        let cloned = health.clone();
        assert_eq!(health, cloned);
        println!("AFTER: Cloned successfully, equality verified");

        let debug = format!("{:?}", health);
        assert!(debug.contains("is_healthy: true"));
        assert!(debug.contains("node_count: 100"));
        println!("Debug output: {}", debug);
        println!("RESULT: PASS - Clone and Debug traits work correctly");
    }

    #[test]
    fn test_storage_health_partial_eq() {
        println!("=== STORAGE HEALTH PARTIAL EQ TEST ===");
        let h1 = StorageHealth {
            is_healthy: true,
            node_count: 10,
            edge_count: 5,
            storage_bytes: 512,
        };
        let h2 = StorageHealth {
            is_healthy: true,
            node_count: 10,
            edge_count: 5,
            storage_bytes: 512,
        };
        let h3 = StorageHealth {
            is_healthy: false,
            node_count: 10,
            edge_count: 5,
            storage_bytes: 512,
        };

        println!("BEFORE: Testing PartialEq");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        println!("AFTER: Equal instances matched, different instances differed");
        println!("RESULT: PASS - PartialEq works correctly");
    }
}
