//! Iteration operations for GraphStorage.
//!
//! Provides iterators for hyperbolic points, cones, and adjacency lists.

use rocksdb::IteratorMode;

use super::core::GraphStorage;
use super::serialization::{deserialize_cone, deserialize_point};
use super::types::{EntailmentCone, LegacyGraphEdge, NodeId, PoincarePoint};
use crate::error::{GraphError, GraphResult};
use crate::storage::edges::GraphEdge;

impl GraphStorage {
    /// Iterate over all hyperbolic points.
    pub fn iter_hyperbolic(
        &self,
    ) -> GraphResult<impl Iterator<Item = GraphResult<(NodeId, PoincarePoint)>> + '_> {
        let cf = self.cf_hyperbolic()?;
        let iter = self.db.iterator_cf(cf, IteratorMode::Start);

        Ok(iter.map(|result| {
            let (key, value) = result.map_err(GraphError::from)?;
            let node_id = NodeId::from_le_bytes(
                key[..8]
                    .try_into()
                    .expect("NodeId key must be 8 bytes - storage corrupted"),
            );
            let point = deserialize_point(&value)?;
            Ok((node_id, point))
        }))
    }

    /// Iterate over all cones.
    pub fn iter_cones(
        &self,
    ) -> GraphResult<impl Iterator<Item = GraphResult<(NodeId, EntailmentCone)>> + '_> {
        let cf = self.cf_cones()?;
        let iter = self.db.iterator_cf(cf, IteratorMode::Start);

        Ok(iter.map(|result| {
            let (key, value) = result.map_err(GraphError::from)?;
            let node_id = NodeId::from_le_bytes(
                key[..8]
                    .try_into()
                    .expect("NodeId key must be 8 bytes - storage corrupted"),
            );
            let cone = deserialize_cone(&value)?;
            Ok((node_id, cone))
        }))
    }

    /// Iterate over all adjacency lists.
    pub fn iter_adjacency(
        &self,
    ) -> GraphResult<impl Iterator<Item = GraphResult<(NodeId, Vec<LegacyGraphEdge>)>> + '_> {
        let cf = self.cf_adjacency()?;
        let iter = self.db.iterator_cf(cf, IteratorMode::Start);

        Ok(iter.map(|result| {
            let (key, value) = result.map_err(GraphError::from)?;
            let node_id = NodeId::from_le_bytes(
                key[..8]
                    .try_into()
                    .expect("NodeId key must be 8 bytes - storage corrupted"),
            );
            let edges: Vec<LegacyGraphEdge> =
                bincode::deserialize(&value).map_err(|e| GraphError::CorruptedData {
                    location: format!("adjacency node_id={}", node_id),
                    details: e.to_string(),
                })?;
            Ok((node_id, edges))
        }))
    }

    /// Iterate over all GraphEdges.
    pub fn iter_edges(&self) -> GraphResult<impl Iterator<Item = GraphResult<GraphEdge>> + '_> {
        let cf = self.cf_edges()?;
        let iter = self.db.iterator_cf(cf, IteratorMode::Start);

        Ok(iter.map(|result| {
            let (key, value) = result.map_err(GraphError::from)?;
            let edge_id = i64::from_le_bytes(
                key[..8]
                    .try_into()
                    .expect("Edge key must be 8 bytes - storage corrupted"),
            );
            let edge: GraphEdge =
                bincode::deserialize(&value).map_err(|e| GraphError::CorruptedData {
                    location: format!("edge edge_id={}", edge_id),
                    details: e.to_string(),
                })?;
            Ok(edge)
        }))
    }
}
