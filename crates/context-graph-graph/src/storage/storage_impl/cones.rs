//! Entailment cone operations for GraphStorage.
//!
//! Provides get, put, and delete operations for entailment cones.

use super::core::GraphStorage;
use super::serialization::{deserialize_cone, serialize_cone};
use super::types::{EntailmentCone, NodeId};
use crate::error::GraphResult;

impl GraphStorage {
    /// Get entailment cone for a node.
    pub fn get_cone(&self, node_id: NodeId) -> GraphResult<Option<EntailmentCone>> {
        let cf = self.cf_cones()?;
        let key = node_id.to_le_bytes();

        match self.db.get_cf(cf, key)? {
            Some(bytes) => {
                let cone = deserialize_cone(&bytes).map_err(|e| {
                    log::error!("CORRUPTED: cone node_id={}: {}", node_id, e);
                    e
                })?;
                Ok(Some(cone))
            }
            None => Ok(None),
        }
    }

    /// Store entailment cone for a node.
    pub fn put_cone(&self, node_id: NodeId, cone: &EntailmentCone) -> GraphResult<()> {
        let cf = self.cf_cones()?;
        let key = node_id.to_le_bytes();
        let value = serialize_cone(cone);

        self.db.put_cf(cf, key, value)?;
        log::trace!("PUT cone node_id={}", node_id);
        Ok(())
    }

    /// Delete entailment cone for a node.
    pub fn delete_cone(&self, node_id: NodeId) -> GraphResult<()> {
        let cf = self.cf_cones()?;
        let key = node_id.to_le_bytes();

        self.db.delete_cf(cf, key)?;
        log::trace!("DELETE cone node_id={}", node_id);
        Ok(())
    }
}
