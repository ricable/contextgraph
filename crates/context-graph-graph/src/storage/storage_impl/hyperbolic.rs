//! Hyperbolic point operations for GraphStorage.
//!
//! Provides get, put, and delete operations for Poincare coordinates.

use super::core::GraphStorage;
use super::serialization::{deserialize_point, serialize_point};
use super::types::{NodeId, PoincarePoint};
use crate::error::GraphResult;

impl GraphStorage {
    /// Get hyperbolic coordinates for a node.
    ///
    /// # Returns
    /// * `Ok(Some(point))` - Point exists
    /// * `Ok(None)` - Node not found
    /// * `Err(GraphError::CorruptedData)` - Invalid data in storage
    pub fn get_hyperbolic(&self, node_id: NodeId) -> GraphResult<Option<PoincarePoint>> {
        let cf = self.cf_hyperbolic()?;
        let key = node_id.to_le_bytes();

        match self.db.get_cf(cf, key)? {
            Some(bytes) => {
                let point = deserialize_point(&bytes).map_err(|e| {
                    log::error!("CORRUPTED: hyperbolic node_id={}: {}", node_id, e);
                    e
                })?;
                Ok(Some(point))
            }
            None => Ok(None),
        }
    }

    /// Store hyperbolic coordinates for a node.
    ///
    /// Overwrites existing coordinates if present.
    pub fn put_hyperbolic(&self, node_id: NodeId, point: &PoincarePoint) -> GraphResult<()> {
        let cf = self.cf_hyperbolic()?;
        let key = node_id.to_le_bytes();
        let value = serialize_point(point);

        self.db.put_cf(cf, key, value)?;
        log::trace!("PUT hyperbolic node_id={}", node_id);
        Ok(())
    }

    /// Delete hyperbolic coordinates for a node.
    pub fn delete_hyperbolic(&self, node_id: NodeId) -> GraphResult<()> {
        let cf = self.cf_hyperbolic()?;
        let key = node_id.to_le_bytes();

        self.db.delete_cf(cf, key)?;
        log::trace!("DELETE hyperbolic node_id={}", node_id);
        Ok(())
    }
}
