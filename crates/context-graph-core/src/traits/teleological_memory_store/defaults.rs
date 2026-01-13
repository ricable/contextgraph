//! Default implementations for optional TeleologicalMemoryStore methods.
//!
//! This module provides default implementations for content storage and ego node
//! storage methods. These are separated from the core trait to keep the main
//! trait file under the 500-line limit while maintaining clear organization.
//!
//! # Content Storage (TASK-CONTENT-003)
//!
//! Content storage allows associating original text content with fingerprints.
//! The default implementations return errors or empty results for backends
//! that don't support content storage.
//!
//! # Ego Node Storage (TASK-GWT-P1-001)
//!
//! Ego node storage persists the singleton SELF_EGO_NODE representing
//! the system's identity. The default implementations provide graceful
//! degradation for backends without ego node support.

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::{CoreError, CoreResult};
use crate::gwt::ego_node::SelfEgoNode;

use super::backend::TeleologicalStorageBackend;

/// Extension trait providing default implementations for optional storage features.
///
/// This trait is automatically implemented for all types implementing
/// `TeleologicalMemoryStore`. It provides default behavior for:
/// - Content storage (store, get, delete, batch get)
/// - Ego node storage (save, load)
///
/// Backends that support these features should override the corresponding
/// methods in `TeleologicalMemoryStore` directly.
#[allow(dead_code)] // Reserved for future extension trait implementations
#[async_trait]
pub trait TeleologicalMemoryStoreDefaults: Send + Sync {
    /// Get the storage backend type for error messages.
    fn backend_type(&self) -> TeleologicalStorageBackend;

    // ==================== Content Storage Defaults ====================

    /// Default: Store content - returns unsupported error.
    ///
    /// Content is stored separately from the fingerprint for efficiency.
    /// This allows large text content to be optionally retrieved without
    /// loading it for every search result.
    ///
    /// # Arguments
    /// * `id` - Fingerprint UUID
    /// * `content` - Original text content (max 1MB)
    ///
    /// # Errors
    /// - `CoreError::Internal` - Content storage not supported by backend
    async fn store_content_default(&self, id: Uuid, content: &str) -> CoreResult<()> {
        let _ = (id, content); // Suppress unused warnings
        Err(CoreError::Internal(format!(
            "Content storage not supported by {} backend",
            self.backend_type()
        )))
    }

    /// Default: Get content - returns None (graceful degradation).
    ///
    /// Returns the original text content that was stored with the fingerprint.
    /// Returns None if content was never stored or backend doesn't support it.
    ///
    /// # Arguments
    /// * `id` - Fingerprint UUID
    ///
    /// # Returns
    /// `None` - Backend does not support content storage.
    async fn get_content_default(&self, id: Uuid) -> CoreResult<Option<String>> {
        let _ = id; // Suppress unused warnings
        Ok(None)
    }

    /// Default: Delete content - returns false (nothing to delete).
    ///
    /// Called automatically when fingerprint is deleted (cascade delete).
    /// Can also be called directly to remove content while keeping the fingerprint.
    ///
    /// # Arguments
    /// * `id` - Fingerprint UUID
    ///
    /// # Returns
    /// `false` - No content existed to delete.
    async fn delete_content_default(&self, id: Uuid) -> CoreResult<bool> {
        let _ = id; // Suppress unused warnings
        Ok(false)
    }

    /// Default: Batch get content - calls get_content sequentially.
    ///
    /// More efficient than individual `get_content` calls for bulk retrieval.
    /// Returns Vec with Some for found content, None for not found.
    ///
    /// # Arguments
    /// * `ids` - Slice of fingerprint UUIDs
    ///
    /// # Returns
    /// Vector of `None` values (backend does not support content storage).
    async fn get_content_batch_default(&self, ids: &[Uuid]) -> CoreResult<Vec<Option<String>>> {
        // Default returns None for all IDs since content storage is not supported
        Ok(vec![None; ids.len()])
    }

    // ==================== Ego Node Storage Defaults ====================

    /// Default: Save ego node - returns unsupported error.
    ///
    /// The SELF_EGO_NODE represents the system's persistent identity across
    /// sessions. Only one ego node ever exists in the database, stored with
    /// a fixed key ("ego_node").
    ///
    /// # Arguments
    /// * `ego_node` - The SelfEgoNode to persist
    ///
    /// # Errors
    /// - `CoreError::Internal` - Ego node storage not supported by backend
    ///
    /// # Constitution Reference
    /// gwt.self_ego_node (lines 371-392): Identity persistence requirements
    async fn save_ego_node_default(&self, ego_node: &SelfEgoNode) -> CoreResult<()> {
        let _ = ego_node; // Suppress unused warnings
        Err(CoreError::Internal(format!(
            "Ego node storage not supported by {} backend",
            self.backend_type()
        )))
    }

    /// Default: Load ego node - returns None (graceful degradation).
    ///
    /// Returns the system's persisted identity. Returns None if no ego node
    /// has been saved yet (first run), indicating the system should initialize
    /// a new identity.
    ///
    /// # Returns
    /// `None` - Backend does not support ego node storage or no ego node saved.
    ///
    /// # Constitution Reference
    /// gwt.self_ego_node (lines 371-392): Identity persistence requirements
    async fn load_ego_node_default(&self) -> CoreResult<Option<SelfEgoNode>> {
        Ok(None)
    }
}
