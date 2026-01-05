//! Adjacency list operations for GraphStorage.
//!
//! Provides operations for legacy graph edges and adjacency lists.

use super::core::GraphStorage;
use super::types::{LegacyGraphEdge, NodeId};
use crate::error::{GraphError, GraphResult};

impl GraphStorage {
    /// Get edges for a node.
    ///
    /// Returns empty Vec if node has no edges.
    pub fn get_adjacency(&self, node_id: NodeId) -> GraphResult<Vec<LegacyGraphEdge>> {
        let cf = self.cf_adjacency()?;
        let key = node_id.to_le_bytes();

        match self.db.get_cf(cf, key)? {
            Some(bytes) => {
                let edges: Vec<LegacyGraphEdge> =
                    bincode::deserialize(&bytes).map_err(|e| GraphError::CorruptedData {
                        location: format!("adjacency node_id={}", node_id),
                        details: e.to_string(),
                    })?;
                Ok(edges)
            }
            None => Ok(Vec::new()),
        }
    }

    /// Store edges for a node.
    pub fn put_adjacency(&self, node_id: NodeId, edges: &[LegacyGraphEdge]) -> GraphResult<()> {
        let cf = self.cf_adjacency()?;
        let key = node_id.to_le_bytes();
        let value = bincode::serialize(edges)?;

        self.db.put_cf(cf, key, value)?;
        log::trace!("PUT adjacency node_id={} edges={}", node_id, edges.len());
        Ok(())
    }

    /// Add a single edge (reads existing, appends, writes back).
    pub fn add_edge(&self, source: NodeId, edge: LegacyGraphEdge) -> GraphResult<()> {
        let mut edges = self.get_adjacency(source)?;
        edges.push(edge);
        self.put_adjacency(source, &edges)
    }

    /// Remove an edge by target node.
    ///
    /// Returns true if edge was found and removed.
    pub fn remove_edge(&self, source: NodeId, target: NodeId) -> GraphResult<bool> {
        let mut edges = self.get_adjacency(source)?;
        let original_len = edges.len();
        edges.retain(|e| e.target != target);

        if edges.len() < original_len {
            self.put_adjacency(source, &edges)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Delete all edges for a node.
    pub fn delete_adjacency(&self, node_id: NodeId) -> GraphResult<()> {
        let cf = self.cf_adjacency()?;
        let key = node_id.to_le_bytes();

        self.db.delete_cf(cf, key)?;
        log::trace!("DELETE adjacency node_id={}", node_id);
        Ok(())
    }
}
