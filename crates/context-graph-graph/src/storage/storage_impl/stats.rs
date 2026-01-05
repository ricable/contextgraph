//! Statistics and schema version operations for GraphStorage.
//!
//! Provides count methods and schema version management.

use rocksdb::IteratorMode;

use super::core::GraphStorage;
use crate::error::{GraphError, GraphResult};

impl GraphStorage {
    // ========== Statistics ==========

    /// Get count of hyperbolic points stored.
    pub fn hyperbolic_count(&self) -> GraphResult<usize> {
        let cf = self.cf_hyperbolic()?;
        Ok(self.db.iterator_cf(cf, IteratorMode::Start).count())
    }

    /// Get count of cones stored.
    pub fn cone_count(&self) -> GraphResult<usize> {
        let cf = self.cf_cones()?;
        Ok(self.db.iterator_cf(cf, IteratorMode::Start).count())
    }

    /// Get count of nodes with adjacency lists.
    pub fn adjacency_count(&self) -> GraphResult<usize> {
        let cf = self.cf_adjacency()?;
        Ok(self.db.iterator_cf(cf, IteratorMode::Start).count())
    }

    /// Get count of GraphEdges stored.
    pub fn edge_count(&self) -> GraphResult<usize> {
        let cf = self.cf_edges()?;
        Ok(self.db.iterator_cf(cf, IteratorMode::Start).count())
    }

    // ========== Schema Version Operations ==========

    /// Get schema version from metadata CF.
    ///
    /// # Returns
    /// * `Ok(version)` - Current schema version (0 if not set)
    /// * `Err(GraphError::CorruptedData)` - Invalid version data
    pub fn get_schema_version(&self) -> GraphResult<u32> {
        let cf = self.cf_metadata()?;

        match self.db.get_cf(cf, b"schema_version")? {
            Some(bytes) => {
                if bytes.len() != 4 {
                    return Err(GraphError::CorruptedData {
                        location: "metadata/schema_version".to_string(),
                        details: format!("Expected 4 bytes, got {}", bytes.len()),
                    });
                }
                let version = u32::from_le_bytes(
                    bytes[..4]
                        .try_into()
                        .expect("verified 4 bytes above - this cannot fail"),
                );
                log::trace!("get_schema_version: {}", version);
                Ok(version)
            }
            None => {
                log::trace!("get_schema_version: 0 (not set)");
                Ok(0) // No version stored = version 0
            }
        }
    }

    /// Set schema version in metadata CF.
    pub fn set_schema_version(&self, version: u32) -> GraphResult<()> {
        let cf = self.cf_metadata()?;
        self.db
            .put_cf(cf, b"schema_version", version.to_le_bytes())?;
        log::debug!("set_schema_version: {}", version);
        Ok(())
    }

    /// Apply all pending migrations.
    ///
    /// Should be called after open() to ensure database is up to date.
    ///
    /// # Returns
    /// * `Ok(version)` - Final schema version
    /// * `Err(GraphError::MigrationFailed)` - Migration failed
    pub fn apply_migrations(&self) -> GraphResult<u32> {
        let migrations = crate::storage::migrations::Migrations::new();
        migrations.migrate(self)
    }

    /// Check if database needs migrations.
    pub fn needs_migrations(&self) -> GraphResult<bool> {
        let migrations = crate::storage::migrations::Migrations::new();
        migrations.needs_migration(self)
    }
}
