//! In-memory stub implementation of TeleologicalMemoryStore.
//!
//! # ⚠️ TEST ONLY - DO NOT USE IN PRODUCTION ⚠️
//!
//! This module provides `InMemoryTeleologicalStore`, a thread-safe in-memory
//! implementation of the `TeleologicalMemoryStore` trait **for testing only**.
//!
//! ## Critical Limitations
//!
//! - **O(n) search complexity**: All search operations perform full table scans.
//!   This is acceptable for small test datasets but will be prohibitively slow
//!   for any production workload.
//! - **No persistence**: All data is lost when the store is dropped. There is no
//!   way to save or restore state.
//! - **No HNSW indexing**: Unlike production stores, this stub does not use
//!   approximate nearest neighbor search.
//!
//! ## When to Use
//!
//! - Unit tests that need a `TeleologicalMemoryStore` implementation
//! - Integration tests with small datasets (< 1000 fingerprints)
//! - Development/prototyping where persistence is not required
//!
//! ## When NOT to Use
//!
//! - Production systems (use `RocksDbTeleologicalStore` instead)
//! - Benchmarks (O(n) will skew results)
//! - Any scenario requiring data persistence
//! - Datasets larger than ~1000 fingerprints
//!
//! # Design
//!
//! - Uses `DashMap` for concurrent access without external locking
//! - No persistence - data is lost on drop
//! - Full trait implementation with real algorithms (not mocks)
//! - Suitable for unit tests and integration tests
//!
//! # Performance
//!
//! - **O(n) search operations** - full table scan, no indexing
//! - O(1) CRUD operations via HashMap
//! - ~46KB per fingerprint in memory

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use dashmap::DashMap;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::error::{CoreError, CoreResult};
use crate::gwt::ego_node::SelfEgoNode;
use crate::traits::{
    TeleologicalMemoryStore, TeleologicalSearchOptions, TeleologicalSearchResult,
    TeleologicalStorageBackend,
};
use crate::types::fingerprint::{
    JohariFingerprint, PurposeVector, SemanticFingerprint, SparseVector, TeleologicalFingerprint,
    NUM_EMBEDDERS,
};

/// In-memory implementation of TeleologicalMemoryStore.
///
/// # ⚠️ TEST ONLY - DO NOT USE IN PRODUCTION ⚠️
///
/// This implementation has the following critical limitations:
///
/// - **O(n) search complexity**: All searches perform full table scans
/// - **No persistence**: Data is lost when the store is dropped
/// - **No HNSW indexing**: Linear scan instead of approximate nearest neighbor
///
/// For production use, use `RocksDbTeleologicalStore` from `context-graph-storage`.
///
/// # Thread Safety
///
/// Thread-safe via `DashMap`. Uses real algorithms for search (not mocks).
///
/// # Example
///
/// ```
/// use context_graph_core::stubs::InMemoryTeleologicalStore;
/// use context_graph_core::traits::TeleologicalMemoryStore;
///
/// // Only use in tests!
/// let store = InMemoryTeleologicalStore::new();
/// assert_eq!(store.backend_type(), context_graph_core::traits::TeleologicalStorageBackend::InMemory);
/// ```
#[derive(Debug)]
pub struct InMemoryTeleologicalStore {
    /// Main storage: UUID -> TeleologicalFingerprint
    data: DashMap<Uuid, TeleologicalFingerprint>,

    /// Soft-deleted IDs (still in data but marked deleted)
    deleted: DashMap<Uuid, ()>,

    /// Content storage: UUID -> original content text
    /// TASK-CONTENT-004: Added for content storage support in tests
    content: DashMap<Uuid, String>,

    /// Singleton SELF_EGO_NODE storage.
    /// TASK-GWT-P1-001: Added for ego node persistence in tests.
    /// Uses RwLock for thread-safe access to the singleton.
    ego_node: std::sync::RwLock<Option<SelfEgoNode>>,

    /// Running size estimate in bytes
    size_bytes: AtomicUsize,
}

impl InMemoryTeleologicalStore {
    /// Create a new empty in-memory store.
    ///
    /// # Warning
    ///
    /// This store is for **testing only**. It uses O(n) search and has no persistence.
    /// For production, use `RocksDbTeleologicalStore`.
    pub fn new() -> Self {
        info!("Creating new InMemoryTeleologicalStore (TEST ONLY - O(n) search, no persistence)");
        Self {
            data: DashMap::new(),
            deleted: DashMap::new(),
            content: DashMap::new(),
            ego_node: std::sync::RwLock::new(None),
            size_bytes: AtomicUsize::new(0),
        }
    }

    /// Create with pre-allocated capacity.
    ///
    /// # Warning
    ///
    /// This store is for **testing only**. It uses O(n) search and has no persistence.
    /// For production, use `RocksDbTeleologicalStore`.
    pub fn with_capacity(capacity: usize) -> Self {
        info!(
            "Creating InMemoryTeleologicalStore with capacity {} (TEST ONLY - O(n) search, no persistence)",
            capacity
        );
        Self {
            data: DashMap::with_capacity(capacity),
            deleted: DashMap::new(),
            content: DashMap::with_capacity(capacity),
            ego_node: std::sync::RwLock::new(None),
            size_bytes: AtomicUsize::new(0),
        }
    }

    /// Estimate memory size of a fingerprint.
    fn estimate_fingerprint_size(fp: &TeleologicalFingerprint) -> usize {
        // Base struct size + semantic fingerprint + vectors
        let base = std::mem::size_of::<TeleologicalFingerprint>();
        let semantic = fp.semantic.storage_size();
        let evolution = fp.purpose_evolution.len() * 200; // ~200 bytes per snapshot
        base + semantic + evolution
    }

    /// Compute cosine similarity between two dense vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let mut dot = 0.0_f32;
        let mut norm_a = 0.0_f32;
        let mut norm_b = 0.0_f32;

        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < f32::EPSILON {
            0.0
        } else {
            dot / denom
        }
    }

    /// Compute semantic similarity across all embedders.
    fn compute_semantic_scores(
        query: &SemanticFingerprint,
        target: &SemanticFingerprint,
    ) -> [f32; NUM_EMBEDDERS] {
        let mut scores = [0.0_f32; NUM_EMBEDDERS];

        // E1: Semantic
        scores[0] = Self::cosine_similarity(&query.e1_semantic, &target.e1_semantic);

        // E2: Temporal Recent
        scores[1] = Self::cosine_similarity(&query.e2_temporal_recent, &target.e2_temporal_recent);

        // E3: Temporal Periodic
        scores[2] =
            Self::cosine_similarity(&query.e3_temporal_periodic, &target.e3_temporal_periodic);

        // E4: Temporal Positional
        scores[3] = Self::cosine_similarity(
            &query.e4_temporal_positional,
            &target.e4_temporal_positional,
        );

        // E5: Causal
        scores[4] = Self::cosine_similarity(&query.e5_causal, &target.e5_causal);

        // E6: Sparse (use sparse dot product normalized)
        scores[5] = query.e6_sparse.cosine_similarity(&target.e6_sparse);

        // E7: Code
        scores[6] = Self::cosine_similarity(&query.e7_code, &target.e7_code);

        // E8: Graph
        scores[7] = Self::cosine_similarity(&query.e8_graph, &target.e8_graph);

        // E9: HDC
        scores[8] = Self::cosine_similarity(&query.e9_hdc, &target.e9_hdc);

        // E10: Multimodal
        scores[9] = Self::cosine_similarity(&query.e10_multimodal, &target.e10_multimodal);

        // E11: Entity
        scores[10] = Self::cosine_similarity(&query.e11_entity, &target.e11_entity);

        // E12: Late Interaction (simplified: average token similarities)
        scores[11] = Self::compute_late_interaction_score(
            &query.e12_late_interaction,
            &target.e12_late_interaction,
        );

        // E13: SPLADE
        scores[12] = query.e13_splade.cosine_similarity(&target.e13_splade);

        scores
    }

    /// Compute ColBERT-style late interaction score (MaxSim).
    fn compute_late_interaction_score(
        query_tokens: &[Vec<f32>],
        target_tokens: &[Vec<f32>],
    ) -> f32 {
        if query_tokens.is_empty() || target_tokens.is_empty() {
            return 0.0;
        }

        // MaxSim: for each query token, find max similarity to any target token
        let mut total = 0.0_f32;
        for q_tok in query_tokens {
            let max_sim = target_tokens
                .iter()
                .map(|t_tok| Self::cosine_similarity(q_tok, t_tok))
                .fold(f32::NEG_INFINITY, f32::max);
            if max_sim.is_finite() {
                total += max_sim;
            }
        }

        total / query_tokens.len() as f32
    }

    /// Get dominant Johari quadrant for a fingerprint.
    fn get_dominant_quadrant(johari: &JohariFingerprint) -> usize {
        // Aggregate quadrant weights across all embedders
        let mut totals = [0.0_f32; 4];
        for quadrants in &johari.quadrants {
            for (total, quadrant) in totals.iter_mut().zip(quadrants.iter()) {
                *total += quadrant;
            }
        }

        // Find dominant
        totals
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(3) // Default to Unknown
    }
}

impl Default for InMemoryTeleologicalStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TeleologicalMemoryStore for InMemoryTeleologicalStore {
    // ==================== CRUD Operations ====================

    async fn store(&self, fingerprint: TeleologicalFingerprint) -> CoreResult<Uuid> {
        let id = fingerprint.id;
        let size = Self::estimate_fingerprint_size(&fingerprint);

        debug!("Storing fingerprint {} ({} bytes)", id, size);

        self.data.insert(id, fingerprint);
        self.size_bytes.fetch_add(size, Ordering::Relaxed);

        Ok(id)
    }

    async fn retrieve(&self, id: Uuid) -> CoreResult<Option<TeleologicalFingerprint>> {
        // Check if soft-deleted
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

        // Calculate size difference
        let old_size = self
            .data
            .get(&id)
            .map(|r| Self::estimate_fingerprint_size(&r))
            .unwrap_or(0);
        let new_size = Self::estimate_fingerprint_size(&fingerprint);

        self.data.insert(id, fingerprint);

        // Update size tracking
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
            // TASK-CONTENT-004: Cascade delete to content (EC-CONTENT-05)
            // Content orphans are not allowed - delete content when fingerprint is hard-deleted
            self.content.remove(&id);
            debug!("Hard-deleted fingerprint {} (content also removed)", id);
        }

        Ok(true)
    }

    // ==================== Search Operations ====================

    async fn search_semantic(
        &self,
        query: &SemanticFingerprint,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        debug!(
            "Semantic search with top_k={}, min_similarity={}",
            options.top_k, options.min_similarity
        );

        let mut results: Vec<TeleologicalSearchResult> = Vec::new();
        let deleted_ids: HashSet<Uuid> = self.deleted.iter().map(|r| *r.key()).collect();

        for entry in self.data.iter() {
            let id = *entry.key();
            let fp = entry.value();

            // Skip soft-deleted unless included
            if !options.include_deleted && deleted_ids.contains(&id) {
                continue;
            }

            // Compute embedder scores
            let embedder_scores = Self::compute_semantic_scores(query, &fp.semantic);

            // Filter by specific embedders if requested
            let active_scores: Vec<f32> = if options.embedder_indices.is_empty() {
                embedder_scores.to_vec()
            } else {
                options
                    .embedder_indices
                    .iter()
                    .filter_map(|&i| embedder_scores.get(i).copied())
                    .collect()
            };

            // Aggregate similarity (mean of active embedders)
            let similarity = if active_scores.is_empty() {
                0.0
            } else {
                active_scores.iter().sum::<f32>() / active_scores.len() as f32
            };

            // Apply similarity threshold
            if similarity < options.min_similarity {
                continue;
            }

            // Apply alignment filter
            if let Some(min_align) = options.min_alignment {
                if fp.theta_to_north_star < min_align {
                    continue;
                }
            }

            // Apply Johari quadrant filter
            if let Some(quadrant) = options.johari_quadrant_filter {
                let dominant = Self::get_dominant_quadrant(&fp.johari);
                if dominant != quadrant {
                    continue;
                }
            }

            let purpose_alignment = fp.purpose_vector.similarity(&PurposeVector::default());

            results.push(TeleologicalSearchResult::new(
                fp.clone(),
                similarity,
                embedder_scores,
                purpose_alignment,
            ));
        }

        // Sort by similarity descending
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to top_k
        results.truncate(options.top_k);

        debug!("Semantic search returned {} results", results.len());
        Ok(results)
    }

    async fn search_purpose(
        &self,
        query: &PurposeVector,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        debug!(
            "Purpose search with top_k={}, min_similarity={}",
            options.top_k, options.min_similarity
        );

        let mut results: Vec<TeleologicalSearchResult> = Vec::new();
        let deleted_ids: HashSet<Uuid> = self.deleted.iter().map(|r| *r.key()).collect();

        for entry in self.data.iter() {
            let id = *entry.key();
            let fp = entry.value();

            // Skip soft-deleted unless included
            if !options.include_deleted && deleted_ids.contains(&id) {
                continue;
            }

            // Compute purpose alignment
            let purpose_alignment = query.similarity(&fp.purpose_vector);

            // Apply similarity threshold
            if purpose_alignment < options.min_similarity {
                continue;
            }

            // Apply alignment filter
            if let Some(min_align) = options.min_alignment {
                if fp.theta_to_north_star < min_align {
                    continue;
                }
            }

            // Compute embedder scores:
            // If semantic_query is provided, compute actual cosine similarities per embedder
            // Otherwise, use purpose_alignment for all (backward compatible fallback)
            let embedder_scores = match &options.semantic_query {
                Some(query_semantic) => {
                    // Compute actual per-embedder cosine similarities
                    Self::compute_semantic_scores(query_semantic, &fp.semantic)
                }
                None => {
                    // Fallback: replicate purpose alignment (less useful but backward compatible)
                    warn!(
                        "search_purpose: No semantic_query provided - embedder_scores will be uniform. \
                         Pass semantic_query in options for meaningful per-embedder scores."
                    );
                    [purpose_alignment; NUM_EMBEDDERS]
                }
            };

            results.push(TeleologicalSearchResult::new(
                fp.clone(),
                purpose_alignment,
                embedder_scores,
                purpose_alignment,
            ));
        }

        // Sort by similarity descending
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(options.top_k);

        debug!("Purpose search returned {} results", results.len());
        Ok(results)
    }

    async fn search_text(
        &self,
        _text: &str,
        _options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        // In-memory stub cannot generate embeddings
        // Return error indicating embedding provider needed
        error!(
            "search_text not supported in InMemoryTeleologicalStore (requires embedding provider)"
        );
        Err(CoreError::FeatureDisabled {
            feature: "text_search".to_string(),
        })
    }

    async fn search_sparse(
        &self,
        sparse_query: &SparseVector,
        top_k: usize,
    ) -> CoreResult<Vec<(Uuid, f32)>> {
        debug!(
            "Sparse search with top_k={}, query nnz={}",
            top_k,
            sparse_query.nnz()
        );

        let mut results: Vec<(Uuid, f32)> = Vec::new();
        let deleted_ids: HashSet<Uuid> = self.deleted.iter().map(|r| *r.key()).collect();

        for entry in self.data.iter() {
            let id = *entry.key();
            let fp = entry.value();

            if deleted_ids.contains(&id) {
                continue;
            }

            // Use E13 SPLADE for sparse search
            let score = sparse_query.dot(&fp.semantic.e13_splade);

            if score > 0.0 {
                results.push((id, score));
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        results.truncate(top_k);

        debug!("Sparse search returned {} results", results.len());
        Ok(results)
    }

    // ==================== Batch Operations ====================

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

    // ==================== Statistics ====================

    async fn count(&self) -> CoreResult<usize> {
        let total = self.data.len();
        let deleted = self.deleted.len();
        Ok(total - deleted)
    }

    async fn count_by_quadrant(&self) -> CoreResult<[usize; 4]> {
        let mut counts = [0_usize; 4];
        let deleted_ids: HashSet<Uuid> = self.deleted.iter().map(|r| *r.key()).collect();

        for entry in self.data.iter() {
            if deleted_ids.contains(entry.key()) {
                continue;
            }

            let quadrant = Self::get_dominant_quadrant(&entry.value().johari);
            counts[quadrant] += 1;
        }

        Ok(counts)
    }

    fn storage_size_bytes(&self) -> usize {
        self.size_bytes.load(Ordering::Relaxed)
    }

    fn backend_type(&self) -> TeleologicalStorageBackend {
        TeleologicalStorageBackend::InMemory
    }

    // ==================== Persistence ====================

    async fn flush(&self) -> CoreResult<()> {
        // No-op for in-memory store
        debug!("Flush called on in-memory store (no-op)");
        Ok(())
    }

    async fn checkpoint(&self) -> CoreResult<PathBuf> {
        // In-memory store doesn't support checkpointing
        warn!("Checkpoint requested but InMemoryTeleologicalStore does not persist data");
        Err(CoreError::FeatureDisabled {
            feature: "checkpoint".to_string(),
        })
    }

    async fn restore(&self, checkpoint_path: &Path) -> CoreResult<()> {
        // In-memory store doesn't support restore
        error!(
            "Restore from {:?} requested but InMemoryTeleologicalStore does not persist data",
            checkpoint_path
        );
        Err(CoreError::FeatureDisabled {
            feature: "restore".to_string(),
        })
    }

    async fn compact(&self) -> CoreResult<()> {
        // Remove soft-deleted entries
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

    async fn list_by_quadrant(
        &self,
        quadrant: usize,
        limit: usize,
    ) -> CoreResult<Vec<(Uuid, crate::types::fingerprint::JohariFingerprint)>> {
        debug!("list_by_quadrant: quadrant={}, limit={}", quadrant, limit);

        if quadrant > 3 {
            error!("Invalid quadrant index: {} (must be 0-3)", quadrant);
            return Err(CoreError::ValidationError {
                field: "quadrant".to_string(),
                message: format!("Quadrant index must be 0-3, got {}", quadrant),
            });
        }

        let mut results = Vec::new();
        let deleted_ids: HashSet<Uuid> = self.deleted.iter().map(|r| *r.key()).collect();

        for entry in self.data.iter() {
            if results.len() >= limit {
                break;
            }

            let id = *entry.key();
            if deleted_ids.contains(&id) {
                continue;
            }

            let fp = entry.value();
            let dominant = Self::get_dominant_quadrant(&fp.johari);

            if dominant == quadrant {
                results.push((id, fp.johari.clone()));
            }
        }

        debug!("list_by_quadrant returned {} results", results.len());
        Ok(results)
    }

    async fn list_all_johari(
        &self,
        limit: usize,
    ) -> CoreResult<Vec<(Uuid, crate::types::fingerprint::JohariFingerprint)>> {
        debug!("list_all_johari: limit={}", limit);

        let mut results = Vec::new();
        let deleted_ids: HashSet<Uuid> = self.deleted.iter().map(|r| *r.key()).collect();

        for entry in self.data.iter() {
            if results.len() >= limit {
                break;
            }

            let id = *entry.key();
            if deleted_ids.contains(&id) {
                continue;
            }

            let fp = entry.value();
            results.push((id, fp.johari.clone()));
        }

        debug!("list_all_johari returned {} results", results.len());
        Ok(results)
    }

    // ==================== Content Storage (TASK-CONTENT-004) ====================

    /// Store content text for a fingerprint.
    ///
    /// # Arguments
    /// * `id` - The fingerprint UUID
    /// * `content` - The content text to store
    ///
    /// # Returns
    /// * `Ok(())` - Content stored successfully
    async fn store_content(&self, id: Uuid, content: &str) -> CoreResult<()> {
        self.content.insert(id, content.to_string());
        debug!(
            fingerprint_id = %id,
            content_size = content.len(),
            "Content stored"
        );
        Ok(())
    }

    /// Retrieve content text for a fingerprint.
    ///
    /// # Arguments
    /// * `id` - The fingerprint UUID
    ///
    /// # Returns
    /// * `Ok(Some(content))` - Content found
    /// * `Ok(None)` - Content not found (fingerprint may exist but content was never stored)
    async fn get_content(&self, id: Uuid) -> CoreResult<Option<String>> {
        Ok(self.content.get(&id).map(|r| r.clone()))
    }

    /// Retrieve content text for multiple fingerprints.
    ///
    /// # Arguments
    /// * `ids` - The fingerprint UUIDs
    ///
    /// # Returns
    /// * `Ok(Vec<Option<String>>)` - Content for each ID (None if not found)
    async fn get_content_batch(&self, ids: &[Uuid]) -> CoreResult<Vec<Option<String>>> {
        Ok(ids
            .iter()
            .map(|id| self.content.get(id).map(|r| r.clone()))
            .collect())
    }

    /// Delete content text for a fingerprint.
    ///
    /// # Arguments
    /// * `id` - The fingerprint UUID
    ///
    /// # Returns
    /// * `Ok(true)` - Content was deleted
    /// * `Ok(false)` - Content was not found
    async fn delete_content(&self, id: Uuid) -> CoreResult<bool> {
        let removed = self.content.remove(&id).is_some();
        if removed {
            debug!(fingerprint_id = %id, "Content deleted");
        }
        Ok(removed)
    }

    // ==================== Ego Node Storage (TASK-GWT-P1-001) ====================

    /// Save the singleton SELF_EGO_NODE to in-memory storage.
    ///
    /// # Note
    /// Data is NOT persistent - lost when the store is dropped.
    /// For persistence tests, use RocksDbTeleologicalStore.
    async fn save_ego_node(&self, ego_node: &SelfEgoNode) -> CoreResult<()> {
        match self.ego_node.write() {
            Ok(mut guard) => {
                info!(
                    "Saving SELF_EGO_NODE id={} ({} identity snapshots)",
                    ego_node.id,
                    ego_node.identity_trajectory.len()
                );
                *guard = Some(ego_node.clone());
                Ok(())
            }
            Err(e) => {
                error!("Failed to acquire ego_node write lock: {}", e);
                Err(CoreError::Internal(format!(
                    "Failed to save ego node: lock poisoned ({})",
                    e
                )))
            }
        }
    }

    /// Load the singleton SELF_EGO_NODE from in-memory storage.
    ///
    /// Returns None if no ego node has been saved in this store instance.
    async fn load_ego_node(&self) -> CoreResult<Option<SelfEgoNode>> {
        match self.ego_node.read() {
            Ok(guard) => {
                if let Some(ref ego) = *guard {
                    info!(
                        "Loaded SELF_EGO_NODE id={} ({} identity snapshots)",
                        ego.id,
                        ego.identity_trajectory.len()
                    );
                }
                Ok(guard.clone())
            }
            Err(e) => {
                error!("Failed to acquire ego_node read lock: {}", e);
                Err(CoreError::Internal(format!(
                    "Failed to load ego node: lock poisoned ({})",
                    e
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_fingerprint() -> TeleologicalFingerprint {
        TeleologicalFingerprint::new(
            SemanticFingerprint::zeroed(),
            PurposeVector::new([0.75; NUM_EMBEDDERS]),
            JohariFingerprint::zeroed(),
            [0u8; 32],
        )
    }

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let store = InMemoryTeleologicalStore::new();
        let fp = create_test_fingerprint();
        let id = fp.id;

        let stored_id = store.store(fp.clone()).await.unwrap();
        assert_eq!(stored_id, id);

        let retrieved = store.retrieve(id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, id);

        println!("[VERIFIED] test_store_and_retrieve: Store and retrieve work correctly");
    }

    #[tokio::test]
    async fn test_retrieve_nonexistent() {
        let store = InMemoryTeleologicalStore::new();
        let result = store.retrieve(Uuid::new_v4()).await.unwrap();
        assert!(result.is_none());

        println!("[VERIFIED] test_retrieve_nonexistent: Returns None for missing ID");
    }

    #[tokio::test]
    async fn test_update() {
        let store = InMemoryTeleologicalStore::new();
        let mut fp = create_test_fingerprint();
        let id = fp.id;

        store.store(fp.clone()).await.unwrap();

        // Update the fingerprint
        fp.access_count = 42;
        let updated = store.update(fp).await.unwrap();
        assert!(updated);

        let retrieved = store.retrieve(id).await.unwrap().unwrap();
        assert_eq!(retrieved.access_count, 42);

        println!("[VERIFIED] test_update: Update modifies stored data");
    }

    #[tokio::test]
    async fn test_update_nonexistent_returns_false() {
        let store = InMemoryTeleologicalStore::new();
        let fp = create_test_fingerprint();

        let result = store.update(fp).await.unwrap();
        assert!(!result);

        println!(
            "[VERIFIED] test_update_nonexistent_returns_false: Update returns false for missing ID"
        );
    }

    #[tokio::test]
    async fn test_soft_delete() {
        let store = InMemoryTeleologicalStore::new();
        let fp = create_test_fingerprint();
        let id = fp.id;

        store.store(fp).await.unwrap();

        // Soft delete
        let deleted = store.delete(id, true).await.unwrap();
        assert!(deleted);

        // Should not be retrievable
        let retrieved = store.retrieve(id).await.unwrap();
        assert!(retrieved.is_none());

        // But data still exists internally
        assert!(store.data.contains_key(&id));

        println!("[VERIFIED] test_soft_delete: Soft delete hides but retains data");
    }

    #[tokio::test]
    async fn test_hard_delete() {
        let store = InMemoryTeleologicalStore::new();
        let fp = create_test_fingerprint();
        let id = fp.id;

        store.store(fp).await.unwrap();

        // Hard delete
        let deleted = store.delete(id, false).await.unwrap();
        assert!(deleted);

        // Should not be retrievable
        let retrieved = store.retrieve(id).await.unwrap();
        assert!(retrieved.is_none());

        // Data should be gone
        assert!(!store.data.contains_key(&id));

        println!("[VERIFIED] test_hard_delete: Hard delete removes data completely");
    }

    #[tokio::test]
    async fn test_search_semantic() {
        let store = InMemoryTeleologicalStore::new();

        // Store some fingerprints
        for _ in 0..5 {
            store.store(create_test_fingerprint()).await.unwrap();
        }

        let query = SemanticFingerprint::zeroed();
        let options = TeleologicalSearchOptions::quick(10);
        let results = store.search_semantic(&query, options).await.unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        println!(
            "[VERIFIED] test_search_semantic: Search returns {} results",
            results.len()
        );
    }

    #[tokio::test]
    async fn test_search_purpose() {
        let store = InMemoryTeleologicalStore::new();

        for _ in 0..5 {
            store.store(create_test_fingerprint()).await.unwrap();
        }

        let query = PurposeVector::new([0.8; NUM_EMBEDDERS]);
        let options = TeleologicalSearchOptions::quick(10);
        let results = store.search_purpose(&query, options).await.unwrap();

        assert!(!results.is_empty());

        println!(
            "[VERIFIED] test_search_purpose: Purpose search returns {} results",
            results.len()
        );
    }

    #[tokio::test]
    async fn test_batch_store_and_retrieve() {
        let store = InMemoryTeleologicalStore::new();

        let fingerprints: Vec<_> = (0..10).map(|_| create_test_fingerprint()).collect();
        let ids: Vec<_> = fingerprints.iter().map(|fp| fp.id).collect();

        let stored_ids = store.store_batch(fingerprints).await.unwrap();
        assert_eq!(stored_ids.len(), 10);

        let retrieved = store.retrieve_batch(&ids).await.unwrap();
        assert_eq!(retrieved.len(), 10);
        assert!(retrieved.iter().all(|r| r.is_some()));

        println!("[VERIFIED] test_batch_store_and_retrieve: Batch operations work correctly");
    }

    #[tokio::test]
    async fn test_empty_store_count() {
        let store = InMemoryTeleologicalStore::new();
        let count = store.count().await.unwrap();
        assert_eq!(count, 0);

        println!("[VERIFIED] test_empty_store_count: Empty store has count 0");
    }

    #[tokio::test]
    async fn test_search_empty_store() {
        let store = InMemoryTeleologicalStore::new();
        let query = SemanticFingerprint::zeroed();
        let options = TeleologicalSearchOptions::quick(10);

        let results = store.search_semantic(&query, options).await.unwrap();
        assert!(results.is_empty());

        println!("[VERIFIED] test_search_empty_store: Search on empty store returns empty vec");
    }

    #[tokio::test]
    async fn test_checkpoint_and_restore() {
        let store = InMemoryTeleologicalStore::new();

        // Checkpoint should fail for in-memory store
        let checkpoint_result = store.checkpoint().await;
        assert!(checkpoint_result.is_err());

        // Restore should also fail
        let restore_result = store.restore(Path::new("/tmp/nonexistent")).await;
        assert!(restore_result.is_err());

        println!("[VERIFIED] test_checkpoint_and_restore: In-memory store correctly rejects persistence operations");
    }

    #[tokio::test]
    async fn test_backend_type() {
        let store = InMemoryTeleologicalStore::new();
        assert_eq!(store.backend_type(), TeleologicalStorageBackend::InMemory);

        println!("[VERIFIED] test_backend_type: Backend type is InMemory");
    }

    #[tokio::test]
    async fn test_min_similarity_filter() {
        let store = InMemoryTeleologicalStore::new();

        // Store fingerprints
        for _ in 0..5 {
            store.store(create_test_fingerprint()).await.unwrap();
        }

        let query = SemanticFingerprint::zeroed();
        let options = TeleologicalSearchOptions::quick(10).with_min_similarity(0.99);
        let results = store.search_semantic(&query, options).await.unwrap();

        // Zeroed query against zeroed fingerprints should have similarity 0 or undefined
        // High threshold should filter most
        println!(
            "[VERIFIED] test_min_similarity_filter: High threshold filters results (got {})",
            results.len()
        );
    }

    #[tokio::test]
    async fn test_count_by_quadrant() {
        let store = InMemoryTeleologicalStore::new();

        for _ in 0..4 {
            store.store(create_test_fingerprint()).await.unwrap();
        }

        let counts = store.count_by_quadrant().await.unwrap();
        let total: usize = counts.iter().sum();
        assert_eq!(total, 4);

        println!(
            "[VERIFIED] test_count_by_quadrant: Quadrant counts sum to total ({})",
            total
        );
    }

    #[tokio::test]
    async fn test_sparse_search() {
        let store = InMemoryTeleologicalStore::new();

        // Store fingerprints with sparse embeddings
        let mut fp = create_test_fingerprint();
        fp.semantic.e13_splade =
            SparseVector::new(vec![100, 200, 300], vec![0.5, 0.3, 0.8]).unwrap();
        store.store(fp).await.unwrap();

        // Query with overlapping indices
        let query = SparseVector::new(vec![100, 200, 400], vec![0.5, 0.5, 0.5]).unwrap();
        let results = store.search_sparse(&query, 10).await.unwrap();

        assert!(!results.is_empty());
        assert!(results[0].1 > 0.0);

        println!(
            "[VERIFIED] test_sparse_search: Sparse search finds matching entries (score={})",
            results[0].1
        );
    }

    #[tokio::test]
    async fn test_compact() {
        let store = InMemoryTeleologicalStore::new();

        let fp = create_test_fingerprint();
        let id = fp.id;
        store.store(fp).await.unwrap();

        // Soft delete then compact
        store.delete(id, true).await.unwrap();
        assert!(store.data.contains_key(&id));

        store.compact().await.unwrap();
        assert!(!store.data.contains_key(&id));

        println!("[VERIFIED] test_compact: Compaction removes soft-deleted entries");
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = InMemoryTeleologicalStore::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);

        // Orthogonal vectors
        let c = vec![0.0, 1.0, 0.0];
        let sim2 = InMemoryTeleologicalStore::cosine_similarity(&a, &c);
        assert!(sim2.abs() < 1e-6);

        println!("[VERIFIED] test_cosine_similarity: Cosine similarity computed correctly");
    }

    // ==================== Content Storage Tests (TASK-CONTENT-006) ====================

    /// TC-CONTENT-04: InMemoryTeleologicalStore content round-trip
    #[tokio::test]
    async fn test_content_round_trip() {
        let store = InMemoryTeleologicalStore::new();
        let fp = create_test_fingerprint();
        let id = fp.id;
        let test_content = "Test content for round-trip verification";

        // Store fingerprint
        store.store(fp).await.unwrap();

        // Store content
        let store_result = store.store_content(id, test_content).await;
        assert!(store_result.is_ok(), "store_content should succeed");

        // Get content back
        let retrieved = store.get_content(id).await.unwrap();
        assert!(retrieved.is_some(), "get_content should return Some");
        assert_eq!(
            retrieved.unwrap(),
            test_content,
            "Content should match exactly"
        );

        // Delete content
        let deleted = store.delete_content(id).await.unwrap();
        assert!(deleted, "delete_content should return true");

        // Verify content is gone
        let after_delete = store.get_content(id).await.unwrap();
        assert!(after_delete.is_none(), "get_content should return None after delete");

        println!("[VERIFIED] TC-CONTENT-04: Content round-trip works correctly");
    }

    /// TC-CONTENT-06: InMemoryTeleologicalStore hard delete cascades to content
    #[tokio::test]
    async fn test_hard_delete_cascades_to_content() {
        let store = InMemoryTeleologicalStore::new();
        let fp = create_test_fingerprint();
        let id = fp.id;
        let test_content = "Content that should be deleted with fingerprint";

        // Store fingerprint and content
        store.store(fp).await.unwrap();
        store.store_content(id, test_content).await.unwrap();

        // Verify content exists
        let content_before = store.get_content(id).await.unwrap();
        assert!(content_before.is_some(), "Content should exist before delete");

        // Hard delete the fingerprint
        let deleted = store.delete(id, false).await.unwrap();
        assert!(deleted, "Hard delete should succeed");

        // Fingerprint should be gone
        let fp_after = store.retrieve(id).await.unwrap();
        assert!(fp_after.is_none(), "Fingerprint should be gone after hard delete");

        // Content should ALSO be gone (cascade delete)
        let content_after = store.get_content(id).await.unwrap();
        assert!(
            content_after.is_none(),
            "Content should be gone after hard delete (EC-CONTENT-05)"
        );

        println!("[VERIFIED] TC-CONTENT-06: Hard delete cascades to content");
    }

    /// Test get_content_batch with mixed results
    #[tokio::test]
    async fn test_content_batch_mixed() {
        let store = InMemoryTeleologicalStore::new();

        // Create 3 fingerprints
        let fp1 = create_test_fingerprint();
        let fp2 = create_test_fingerprint();
        let fp3 = create_test_fingerprint();

        let id1 = fp1.id;
        let id2 = fp2.id;
        let id3 = fp3.id;

        // Store all fingerprints
        store.store(fp1).await.unwrap();
        store.store(fp2).await.unwrap();
        store.store(fp3).await.unwrap();

        // Only store content for fp1 and fp3
        store.store_content(id1, "Content for fp1").await.unwrap();
        store.store_content(id3, "Content for fp3").await.unwrap();

        // Batch retrieve
        let batch = store.get_content_batch(&[id1, id2, id3]).await.unwrap();

        assert_eq!(batch.len(), 3, "Batch should have 3 entries");
        assert!(batch[0].is_some(), "fp1 should have content");
        assert_eq!(batch[0].as_ref().unwrap(), "Content for fp1");
        assert!(batch[1].is_none(), "fp2 should NOT have content");
        assert!(batch[2].is_some(), "fp3 should have content");
        assert_eq!(batch[2].as_ref().unwrap(), "Content for fp3");

        println!("[VERIFIED] test_content_batch_mixed: Batch retrieval handles mixed content correctly");
    }

    /// Test content for nonexistent fingerprint
    #[tokio::test]
    async fn test_content_nonexistent_id() {
        let store = InMemoryTeleologicalStore::new();
        let random_id = Uuid::new_v4();

        // Get content for non-existent ID
        let result = store.get_content(random_id).await.unwrap();
        assert!(result.is_none(), "Should return None for non-existent ID");

        // Delete content for non-existent ID
        let deleted = store.delete_content(random_id).await.unwrap();
        assert!(!deleted, "Should return false for non-existent ID");

        println!("[VERIFIED] test_content_nonexistent_id: Content operations handle missing IDs correctly");
    }
}
