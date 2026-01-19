//! Search operations for TeleologicalMemoryStore trait.
//!
//! Contains semantic, purpose, text, and sparse search implementations.

use std::collections::HashMap;

use tracing::{debug, error, warn};
use uuid::Uuid;

use context_graph_core::error::{CoreError, CoreResult};
use context_graph_core::traits::{TeleologicalSearchOptions, TeleologicalSearchResult};
use context_graph_core::types::fingerprint::{SemanticFingerprint, SparseVector};

use crate::teleological::column_families::CF_E13_SPLADE_INVERTED;
use crate::teleological::indexes::{EmbedderIndex, EmbedderIndexOps};
use crate::teleological::schema::e13_splade_inverted_key;
use crate::teleological::serialization::{
    deserialize_memory_id_list, deserialize_teleological_fingerprint,
};

use super::store::RocksDbTeleologicalStore;
use super::types::TeleologicalStoreError;

impl RocksDbTeleologicalStore {
    /// Search by semantic fingerprint (internal async wrapper).
    pub(crate) async fn search_semantic_async(
        &self,
        query: &SemanticFingerprint,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        debug!(
            "Searching semantic with top_k={}, min_similarity={}",
            options.top_k, options.min_similarity
        );

        // Search E1 Semantic as primary (TODO: parallel multi-space search)
        let entry_embedder = EmbedderIndex::E1Semantic;
        let entry_index = self.index_registry.get(entry_embedder).ok_or_else(|| {
            CoreError::IndexError(format!("Index {:?} not found", entry_embedder))
        })?;

        // Search E1 semantic space with 2x top_k to allow filtering
        let k = (options.top_k * 2).max(20);
        let candidates = entry_index
            .search(&query.e1_semantic, k, None)
            .map_err(|e| {
                error!("Semantic search failed: {}", e);
                CoreError::IndexError(e.to_string())
            })?;

        // Fetch full fingerprints for candidates
        let mut results = Vec::with_capacity(candidates.len());

        for (id, distance) in candidates {
            // Convert distance to similarity (HNSW returns distance, not similarity)
            let similarity = 1.0 - distance.min(1.0);

            // Skip soft-deleted
            if !options.include_deleted && self.is_soft_deleted(&id) {
                continue;
            }

            // Apply min_similarity filter
            if similarity < options.min_similarity {
                continue;
            }

            // Fetch full fingerprint from RocksDB
            if let Some(data) = self.get_fingerprint_raw(id)? {
                let fp = deserialize_teleological_fingerprint(&data);

                // Compute all 13 embedder scores using helper
                let embedder_scores = self.compute_embedder_scores(query, &fp.semantic);

                results.push(TeleologicalSearchResult::new(fp, similarity, embedder_scores));
            }
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

    /// Search by text (internal async wrapper).
    pub(crate) async fn search_text_async(
        &self,
        _text: &str,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>> {
        // Text search requires embedding generation, which is not available in storage layer
        // Return empty results with a warning
        warn!("search_text called but embedding generation not available in storage layer");
        warn!("Use embedding service to generate query embeddings, then call search_semantic");
        Ok(Vec::with_capacity(options.top_k))
    }

    /// Search by sparse vector (internal async wrapper).
    pub(crate) async fn search_sparse_async(
        &self,
        sparse_query: &SparseVector,
        top_k: usize,
    ) -> CoreResult<Vec<(Uuid, f32)>> {
        debug!(
            "Searching sparse with {} active terms, top_k={}",
            sparse_query.nnz(),
            top_k
        );

        let cf = self.get_cf(CF_E13_SPLADE_INVERTED)?;

        // Accumulate scores per document
        let mut doc_scores: HashMap<Uuid, f32> = HashMap::new();

        for (i, &term_id) in sparse_query.indices.iter().enumerate() {
            let term_key = e13_splade_inverted_key(term_id);
            let query_weight = sparse_query.values[i];

            if let Some(data) = self.db.get_cf(cf, term_key).map_err(|e| {
                TeleologicalStoreError::rocksdb_op("get", CF_E13_SPLADE_INVERTED, None, e)
            })? {
                let doc_ids = deserialize_memory_id_list(&data);

                for doc_id in doc_ids {
                    // Skip soft-deleted
                    if self.is_soft_deleted(&doc_id) {
                        continue;
                    }

                    // Simple term frequency scoring
                    // TODO: Implement BM25 or other scoring
                    *doc_scores.entry(doc_id).or_insert(0.0) += query_weight;
                }
            }
        }

        // Sort by score descending
        let mut results: Vec<(Uuid, f32)> = doc_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Truncate to top_k
        results.truncate(top_k);

        debug!("Sparse search returned {} results", results.len());
        Ok(results)
    }
}
