//! GraphEdge operations for GraphStorage (M04-T15).
//!
//! Provides CRUD operations for full GraphEdges with Marblestone NT weights.

use rocksdb::WriteBatch;
use uuid::Uuid;

use super::core::GraphStorage;
use crate::error::{GraphError, GraphResult};
use crate::storage::edges::GraphEdge;

impl GraphStorage {
    // ========== GraphEdge Operations (M04-T15) ==========

    /// Compute edge key from edge ID.
    ///
    /// Key format: 8 bytes (i64 little-endian)
    #[inline]
    fn compute_edge_key(edge_id: i64) -> [u8; 8] {
        edge_id.to_le_bytes()
    }

    /// Get a single GraphEdge by ID.
    ///
    /// # Arguments
    /// * `edge_id` - The edge's unique identifier (i64)
    ///
    /// # Returns
    /// * `Ok(Some(edge))` - Edge found
    /// * `Ok(None)` - Edge not found
    /// * `Err(GraphError::CorruptedData)` - Invalid data in storage
    pub fn get_edge(&self, edge_id: i64) -> GraphResult<Option<GraphEdge>> {
        let cf = self.cf_edges()?;
        let key = Self::compute_edge_key(edge_id);

        match self.db.get_cf(cf, key)? {
            Some(bytes) => {
                let edge: GraphEdge = bincode::deserialize(&bytes).map_err(|e| {
                    log::error!("CORRUPTED: edge edge_id={}: {}", edge_id, e);
                    GraphError::CorruptedData {
                        location: format!("edge edge_id={}", edge_id),
                        details: e.to_string(),
                    }
                })?;
                Ok(Some(edge))
            }
            None => Ok(None),
        }
    }

    /// Store a single GraphEdge.
    ///
    /// Overwrites existing edge if present.
    ///
    /// # Arguments
    /// * `edge` - The edge to store (id field used as key)
    pub fn put_edge(&self, edge: &GraphEdge) -> GraphResult<()> {
        let cf = self.cf_edges()?;
        let key = Self::compute_edge_key(edge.id);
        let value = bincode::serialize(edge)?;

        self.db.put_cf(cf, key, value)?;
        log::trace!("PUT edge edge_id={}", edge.id);
        Ok(())
    }

    /// Delete a GraphEdge by ID.
    pub fn delete_edge(&self, edge_id: i64) -> GraphResult<()> {
        let cf = self.cf_edges()?;
        let key = Self::compute_edge_key(edge_id);

        self.db.delete_cf(cf, key)?;
        log::trace!("DELETE edge edge_id={}", edge_id);
        Ok(())
    }

    /// Get multiple GraphEdges by their IDs.
    ///
    /// Returns edges in the same order as input IDs.
    /// Missing edges are skipped (not included in result).
    ///
    /// # Arguments
    /// * `edge_ids` - Slice of edge IDs to retrieve
    ///
    /// # Returns
    /// Vector of (edge_id, GraphEdge) pairs for found edges.
    pub fn get_edges(&self, edge_ids: &[i64]) -> GraphResult<Vec<(i64, GraphEdge)>> {
        let cf = self.cf_edges()?;
        let mut results = Vec::with_capacity(edge_ids.len());

        for &edge_id in edge_ids {
            let key = Self::compute_edge_key(edge_id);
            if let Some(bytes) = self.db.get_cf(cf, key)? {
                let edge: GraphEdge = bincode::deserialize(&bytes).map_err(|e| {
                    log::error!("CORRUPTED: edge edge_id={}: {}", edge_id, e);
                    GraphError::CorruptedData {
                        location: format!("edge edge_id={}", edge_id),
                        details: e.to_string(),
                    }
                })?;
                results.push((edge_id, edge));
            }
        }

        Ok(results)
    }

    /// Store multiple GraphEdges atomically.
    ///
    /// Uses WriteBatch for atomic commit of all edges.
    ///
    /// # Arguments
    /// * `edges` - Slice of edges to store
    pub fn put_edges(&self, edges: &[GraphEdge]) -> GraphResult<()> {
        if edges.is_empty() {
            return Ok(());
        }

        let cf = self.cf_edges()?;
        let mut batch = WriteBatch::default();

        for edge in edges {
            let key = Self::compute_edge_key(edge.id);
            let value = bincode::serialize(edge)?;
            batch.put_cf(cf, key, value);
        }

        self.db.write(batch)?;
        log::trace!("PUT {} edges", edges.len());
        Ok(())
    }

    /// Delete multiple GraphEdges atomically.
    ///
    /// # Arguments
    /// * `edge_ids` - Slice of edge IDs to delete
    pub fn delete_edges(&self, edge_ids: &[i64]) -> GraphResult<()> {
        if edge_ids.is_empty() {
            return Ok(());
        }

        let cf = self.cf_edges()?;
        let mut batch = WriteBatch::default();

        for &edge_id in edge_ids {
            let key = Self::compute_edge_key(edge_id);
            batch.delete_cf(cf, key);
        }

        self.db.write(batch)?;
        log::trace!("DELETE {} edges", edge_ids.len());
        Ok(())
    }

    /// Get all outgoing edges from a source node (M04-T16 BFS support).
    ///
    /// This method iterates CF_EDGES to find edges where `edge.source`
    /// matches the given source node ID (converted from i64 to UUID).
    ///
    /// # Arguments
    /// * `source_node` - Source node ID (i64 storage format)
    ///
    /// # Returns
    /// Vector of full GraphEdges originating from the source node.
    ///
    /// # Note
    /// This performs a full scan of CF_EDGES. For production use with
    /// large graphs, consider adding a secondary index by source node.
    pub fn get_outgoing_edges(&self, source_node: i64) -> GraphResult<Vec<GraphEdge>> {
        let source_uuid = self.i64_to_uuid(source_node);
        let mut result = Vec::new();

        for edge_result in self.iter_edges()? {
            let edge = edge_result?;
            if edge.source == source_uuid {
                result.push(edge);
            }
        }

        Ok(result)
    }

    // ========== UUID Conversion Helpers ==========

    /// Convert i64 node ID to UUID for comparison with GraphEdge.source/target.
    #[inline]
    fn i64_to_uuid(&self, id: i64) -> Uuid {
        // Use from_u64_pair with the i64 in the first position, 0 in second
        Uuid::from_u64_pair(id as u64, 0)
    }
}
