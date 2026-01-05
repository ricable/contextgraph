//! Batch operations for GraphStorage.
//!
//! Provides atomic batch write support for multiple operations.

use rocksdb::WriteBatch;

use super::core::GraphStorage;
use super::serialization::{serialize_cone, serialize_point};
use super::types::{EntailmentCone, LegacyGraphEdge, NodeId, PoincarePoint};
use crate::error::GraphResult;
use crate::storage::edges::GraphEdge;

impl GraphStorage {
    /// Perform multiple operations atomically.
    pub fn write_batch(&self, batch: WriteBatch) -> GraphResult<()> {
        self.db.write(batch)?;
        Ok(())
    }

    /// Create a new write batch.
    #[must_use]
    pub fn new_batch(&self) -> WriteBatch {
        WriteBatch::default()
    }

    /// Add hyperbolic point to batch.
    pub fn batch_put_hyperbolic(
        &self,
        batch: &mut WriteBatch,
        node_id: NodeId,
        point: &PoincarePoint,
    ) -> GraphResult<()> {
        let cf = self.cf_hyperbolic()?;
        let key = node_id.to_le_bytes();
        let value = serialize_point(point);
        batch.put_cf(cf, key, value);
        Ok(())
    }

    /// Add cone to batch.
    pub fn batch_put_cone(
        &self,
        batch: &mut WriteBatch,
        node_id: NodeId,
        cone: &EntailmentCone,
    ) -> GraphResult<()> {
        let cf = self.cf_cones()?;
        let key = node_id.to_le_bytes();
        let value = serialize_cone(cone);
        batch.put_cf(cf, key, value);
        Ok(())
    }

    /// Add adjacency to batch.
    pub fn batch_put_adjacency(
        &self,
        batch: &mut WriteBatch,
        node_id: NodeId,
        edges: &[LegacyGraphEdge],
    ) -> GraphResult<()> {
        let cf = self.cf_adjacency()?;
        let key = node_id.to_le_bytes();
        let value = bincode::serialize(edges)?;
        batch.put_cf(cf, key, value);
        Ok(())
    }

    /// Add edge to batch (M04-T15).
    pub fn batch_put_edge(&self, batch: &mut WriteBatch, edge: &GraphEdge) -> GraphResult<()> {
        let cf = self.cf_edges()?;
        let key = edge.id.to_le_bytes();
        let value = bincode::serialize(edge)?;
        batch.put_cf(cf, key, value);
        Ok(())
    }
}
