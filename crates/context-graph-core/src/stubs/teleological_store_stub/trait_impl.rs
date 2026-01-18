//! Complete TeleologicalMemoryStore trait implementation.
//!
//! This module contains the full trait implementation that delegates to
//! the various impl methods in other submodules.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::InMemoryTeleologicalStore;
use crate::error::{CoreError, CoreResult};
use crate::traits::{
    TeleologicalMemoryStore, TeleologicalSearchOptions, TeleologicalSearchResult,
    TeleologicalStorageBackend,
};
use crate::types::fingerprint::{
    PurposeVector, SemanticFingerprint, SparseVector, TeleologicalFingerprint,
};

#[async_trait]
impl TeleologicalMemoryStore for InMemoryTeleologicalStore {
    async fn store(&self, fingerprint: TeleologicalFingerprint) -> CoreResult<Uuid> {
        let id = fingerprint.id;
        let size = Self::estimate_fingerprint_size(&fingerprint);
        debug!("Storing fingerprint {} ({} bytes)", id, size);
        self.data.insert(id, fingerprint);
        self.size_bytes.fetch_add(size, Ordering::Relaxed);
        Ok(id)
    }

    async fn retrieve(&self, id: Uuid) -> CoreResult<Option<TeleologicalFingerprint>> {
        if self.deleted.contains_key(&id) {
            debug!("Fingerprint {} is soft-deleted", id);
            return Ok(None);
        }
        Ok(self.data.get(&id).map(|r| r.clone()))
    }

    async fn update(&self, fingerprint: TeleologicalFingerprint) -> CoreResult<bool> {
        let id = fingerprint.id;
        if !self.data.contains_key(&id) {
            debug!("Update failed: fingerprint {} not found", id);
            return Ok(false);
        }
        let old_size = self
            .data
            .get(&id)
            .map(|r| Self::estimate_fingerprint_size(&r))
            .unwrap_or(0);
        let new_size = Self::estimate_fingerprint_size(&fingerprint);
        self.data.insert(id, fingerprint);
        if new_size > old_size {
            self.size_bytes
                .fetch_add(new_size - old_size, Ordering::Relaxed);
        } else {
            self.size_bytes
                .fetch_sub(old_size - new_size, Ordering::Relaxed);
        }
        debug!("Updated fingerprint {}", id);
        Ok(true)
    }

    async fn delete(&self, id: Uuid, soft: bool) -> CoreResult<bool> {
        if !self.data.contains_key(&id) {
            debug!("Delete failed: fingerprint {} not found", id);
            return Ok(false);
        }
        if soft {
            self.deleted.insert(id, ());
            debug!("Soft-deleted fingerprint {}", id);
        } else {
            if let Some((_, fp)) = self.data.remove(&id) {
                let size = Self::estimate_fingerprint_size(&fp);
                self.size_bytes.fetch_sub(size, Ordering::Relaxed);
            }
            self.deleted.remove(&id);
            self.content.remove(&id);
            debug!("Hard-deleted fingerprint {} (content also removed)", id);
        }
        Ok(true)
    }

    async fn search_semantic(
        &self,
        query: &SemanticFingerprint,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        self.search_semantic_impl(query, options).await
    }

    async fn search_purpose(
        &self,
        query: &PurposeVector,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        self.search_purpose_impl(query, options).await
    }

    async fn search_text(
        &self,
        text: &str,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        self.search_text_impl(text, options).await
    }

    async fn search_sparse(
        &self,
        sparse_query: &SparseVector,
        top_k: usize,
    ) -> CoreResult<Vec<(Uuid, f32)>> {
        self.search_sparse_impl(sparse_query, top_k).await
    }

    async fn store_batch(
        &self,
        fingerprints: Vec<TeleologicalFingerprint>,
    ) -> CoreResult<Vec<Uuid>> {
        debug!("Batch storing {} fingerprints", fingerprints.len());
        let mut ids = Vec::with_capacity(fingerprints.len());
        for fp in fingerprints {
            let id = self.store(fp).await?;
            ids.push(id);
        }
        Ok(ids)
    }

    async fn retrieve_batch(
        &self,
        ids: &[Uuid],
    ) -> CoreResult<Vec<Option<TeleologicalFingerprint>>> {
        debug!("Batch retrieving {} fingerprints", ids.len());
        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            results.push(self.retrieve(*id).await?);
        }
        Ok(results)
    }

    async fn count(&self) -> CoreResult<usize> {
        Ok(self.data.len() - self.deleted.len())
    }

    fn storage_size_bytes(&self) -> usize {
        self.size_bytes.load(Ordering::Relaxed)
    }

    fn backend_type(&self) -> TeleologicalStorageBackend {
        TeleologicalStorageBackend::InMemory
    }

    async fn flush(&self) -> CoreResult<()> {
        debug!("Flush called on in-memory store (no-op)");
        Ok(())
    }

    async fn checkpoint(&self) -> CoreResult<PathBuf> {
        warn!("Checkpoint requested but InMemoryTeleologicalStore does not persist data");
        Err(CoreError::FeatureDisabled {
            feature: "checkpoint".to_string(),
        })
    }

    async fn restore(&self, checkpoint_path: &Path) -> CoreResult<()> {
        error!(
            "Restore from {:?} requested but InMemoryTeleologicalStore does not persist data",
            checkpoint_path
        );
        Err(CoreError::FeatureDisabled {
            feature: "restore".to_string(),
        })
    }

    async fn compact(&self) -> CoreResult<()> {
        let deleted_ids: Vec<Uuid> = self.deleted.iter().map(|r| *r.key()).collect();
        for id in deleted_ids {
            if let Some((_, fp)) = self.data.remove(&id) {
                let size = Self::estimate_fingerprint_size(&fp);
                self.size_bytes.fetch_sub(size, Ordering::Relaxed);
            }
            self.deleted.remove(&id);
        }
        info!(
            "Compaction complete: removed {} soft-deleted entries",
            self.deleted.len()
        );
        Ok(())
    }

    async fn store_content(&self, id: Uuid, content: &str) -> CoreResult<()> {
        self.store_content_impl(id, content).await
    }

    async fn get_content(&self, id: Uuid) -> CoreResult<Option<String>> {
        self.get_content_impl(id).await
    }

    async fn get_content_batch(&self, ids: &[Uuid]) -> CoreResult<Vec<Option<String>>> {
        self.get_content_batch_impl(ids).await
    }

    async fn delete_content(&self, id: Uuid) -> CoreResult<bool> {
        self.delete_content_impl(id).await
    }
}
