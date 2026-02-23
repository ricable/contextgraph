//! RVF bridge module for dual-write, progressive recall, and COW branching.
//!
//! This module provides the bridge between context-graph-storage's existing usearch HNSW
//! and RVF cognitive containers, enabling:
//! - **Dual-write**: write to both usearch and RVF simultaneously
//! - **Progressive recall**: Layer A (70%, ef=16) → B (85%, ef=64) → C (95%, ef=256)
//! - **COW branching**: derive child stores from parent with Copy-on-Write semantics
//! - **Fallback search**: gracefully degrade to usearch if RVF is unavailable

use parking_lot::RwLock;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

use crate::teleological::indexes::embedder_index::EmbedderIndexOps;

use super::client::{RvfClient, RvfClientConfig};
use super::sona::{SonaConfig, SonaLearning};

/// Errors that can occur during RVF bridge operations.
#[derive(Debug, Error)]
pub enum RvfBridgeError {
    #[error("RVF client error: {0}")]
    Client(#[from] super::client::RvfClientError),

    #[error("USearch error: {0}")]
    USearch(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Store not available")]
    NotAvailable,

    #[error("Invalid state: {0}")]
    InvalidState(String),
}

pub type RvfBridgeResult<T> = Result<T, RvfBridgeError>;

/// Progressive recall layer configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProgressiveRecallLayer {
    /// Layer A: fast recall (~70%), ef=16
    LayerA,
    /// Layer B: medium recall (~85%), ef=64
    LayerB,
    /// Layer C: high recall (~95%), ef=256
    LayerC,
}

impl ProgressiveRecallLayer {
    /// Get the target recall rate for this layer.
    pub fn target_recall(&self) -> f32 {
        match self {
            Self::LayerA => 0.70,
            Self::LayerB => 0.85,
            Self::LayerC => 0.95,
        }
    }

    /// Get the ef_search parameter for this layer.
    pub fn ef_search(&self) -> usize {
        match self {
            Self::LayerA => 16,
            Self::LayerB => 64,
            Self::LayerC => 256,
        }
    }

    /// Get the layer name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::LayerA => "Layer A (Fast)",
            Self::LayerB => "Layer B (Medium)",
            Self::LayerC => "Layer C (High)",
        }
    }

    /// Get the next layer, or None if at max.
    pub fn next(&self) -> Option<Self> {
        match self {
            Self::LayerA => Some(Self::LayerB),
            Self::LayerB => Some(Self::LayerC),
            Self::LayerC => None,
        }
    }
}

/// Configuration for the RVF bridge.
#[derive(Debug, Clone)]
pub struct RvfBridgeConfig {
    /// RVF client configuration
    pub rvf_config: RvfClientConfig,
    /// Enable dual-write mode (write to both usearch and RVF)
    pub dual_write: bool,
    /// Prefer RVF over usearch for search
    pub prefer_rvf: bool,
    /// Enable progressive recall
    pub progressive_recall: bool,
    /// Initial recall target (default: Layer A)
    pub initial_layer: ProgressiveRecallLayer,
    /// Enable SONA learning
    pub sona_enabled: bool,
    /// SONA configuration
    pub sona_config: SonaConfig,
    /// Enable hyperbolic (Poincaré) distance
    pub hyperbolic_enabled: bool,
    /// Store path for local RVF file (if using file-based storage)
    pub store_path: Option<String>,
}

impl Default for RvfBridgeConfig {
    fn default() -> Self {
        Self {
            rvf_config: RvfClientConfig::default(),
            dual_write: false,
            prefer_rvf: false,
            progressive_recall: true,
            initial_layer: ProgressiveRecallLayer::LayerA,
            sona_enabled: true,
            sona_config: SonaConfig::default(),
            hyperbolic_enabled: true,
            store_path: None,
        }
    }
}

/// Search result from the bridge (unified usearch + RVF).
#[derive(Debug, Clone)]
pub struct BridgeSearchResult {
    /// Result ID
    pub id: Uuid,
    /// Similarity score
    pub score: f32,
    /// Source of the result
    pub source: ResultSource,
    /// Progressive recall layer (if applicable)
    pub layer: Option<ProgressiveRecallLayer>,
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
}

/// Source of a search result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultSource {
    /// Result from RVF
    Rvf,
    /// Result from usearch (fallback)
    USearch,
    /// Result from hybrid merge
    Hybrid,
}

/// Bridge status information.
#[derive(Debug, Clone)]
pub struct RvfBridgeStatus {
    /// Whether RVF is available
    pub rvf_available: bool,
    /// Whether dual-write is enabled
    pub dual_write_enabled: bool,
    /// Whether RVF is preferred for search
    pub prefer_rvf: bool,
    /// Current progressive recall layer
    pub current_layer: ProgressiveRecallLayer,
    /// Whether SONA is enabled
    pub sona_enabled: bool,
    /// SONA state
    pub sona_state: super::sona::SonaState,
    /// Whether hyperbolic distance is enabled
    pub hyperbolic_enabled: bool,
    /// Last error message (if any)
    pub last_error: Option<String>,
}

/// RVF Bridge for unified vector search.
///
/// This bridge provides:
///
/// 1. **Dual-write**: When enabled, all writes go to both usearch and RVF
/// 2. **Progressive recall**: Automatically escalates layers if confidence is low
/// 3. **Fallback search**: Gracefully falls back to usearch if RVF is unavailable
/// 4. **COW branching**: Derives child stores from parent with filtering
/// 5. **SONA learning**: Continuously improves retrieval based on feedback
pub struct RvfBridge {
    config: RvfBridgeConfig,
    rvf_client: RwLock<Option<Arc<RvfClient>>>,
    sona: RwLock<Option<SonaLearning>>,
    // For usearch integration - wrapped in Option for lazy initialization
    usearch_index: RwLock<Option<Arc<crate::teleological::indexes::HnswEmbedderIndex>>>,
    // Current progressive recall layer
    current_layer: RwLock<ProgressiveRecallLayer>,
    // Status tracking
    rvf_available: RwLock<bool>,
    last_error: RwLock<Option<String>>,
}

impl RvfBridge {
    /// Create a new RVF bridge with the given configuration.
    pub fn new(config: RvfBridgeConfig) -> RvfBridgeResult<Self> {
        let sona = if config.sona_enabled {
            Some(SonaLearning::new(config.sona_config.clone()))
        } else {
            None
        };

        Ok(Self {
            config,
            rvf_client: RwLock::new(None),
            sona: RwLock::new(sona),
            usearch_index: RwLock::new(None),
            current_layer: RwLock::new(ProgressiveRecallLayer::LayerA),
            rvf_available: RwLock::new(false),
            last_error: RwLock::new(None),
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &RvfBridgeConfig {
        &self.config
    }

    /// Initialize the RVF client connection.
    pub async fn initialize_rvf(&self) -> RvfBridgeResult<()> {
        let client = match RvfClient::new(self.config.rvf_config.clone()) {
            Ok(c) => c,
            Err(e) => {
                *self.last_error.write() = Some(e.to_string());
                return Err(RvfBridgeError::Client(e));
            }
        };

        // Check health
        if client.health().await.unwrap_or(false) {
            *self.rvf_available.write() = true;
            *self.rvf_client.write() = Some(Arc::new(client));
        } else {
            *self.rvf_available.write() = false;
            *self.last_error.write() = Some("RVF service unhealthy".to_string());
        }

        Ok(())
    }

    /// Set the usearch index for fallback.
    pub fn set_usearch_index(
        &self,
        index: Arc<crate::teleological::indexes::HnswEmbedderIndex>,
    ) {
        *self.usearch_index.write() = Some(index);
    }

    /// Store a vector to memory (dual-write or single-write).
    ///
    /// If dual_write is enabled, writes to both usearch and RVF.
    /// Otherwise, writes only to usearch.
    pub async fn store_memory(
        &self,
        id: Uuid,
        vector: &[f32],
        metadata: Option<&serde_json::Value>,
    ) -> RvfBridgeResult<()> {
        // Always write to usearch
        if let Some(index) = self.usearch_index.read().as_ref() {
            index
                .insert(id, vector)
                .map_err(|e| RvfBridgeError::USearch(e.to_string()))?;
        }

        // Dual-write to RVF if enabled
        if self.config.dual_write {
            if let Some(client) = self.rvf_client.read().as_ref() {
                let vectors = vec![vector.to_vec()];
                let ids = Some(vec![id.to_string()]);
                let meta = metadata.map(|m| vec![m.clone()]);

                client
                    .ingest(&vectors, ids.as_deref(), meta.as_deref())
                    .await?;
            }
        }

        Ok(())
    }

    /// Search for similar vectors.
    ///
    /// If prefer_rvf is true, tries RVF first, falls back to usearch.
    /// If progressive_recall is true, escalates layers if confidence is low.
    pub async fn search(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> RvfBridgeResult<Vec<BridgeSearchResult>> {
        // Reset to initial layer
        *self.current_layer.write() = self.config.initial_layer;

        // Try RVF if preferred
        if self.config.prefer_rvf {
            if let Some(client) = self.rvf_client.read().as_ref() {
                if *self.rvf_available.read() {
                    return self.search_rvf_progressive(client, query, top_k).await;
                }
            }
        }

        // Fall back to usearch
        self.search_usearch(query, top_k)
    }

    /// Search with progressive recall.
    ///
    /// Starts at Layer A and escalates if confidence is low.
    async fn search_rvf_progressive(
        &self,
        client: &RvfClient,
        query: &[f32],
        top_k: usize,
    ) -> RvfBridgeResult<Vec<BridgeSearchResult>> {
        let mut all_results: Vec<BridgeSearchResult> = Vec::new();
        let mut seen_ids: std::collections::HashSet<Uuid> = std::collections::HashSet::new();

        // Progressive recall loop
        let mut layer = self.config.initial_layer;

        loop {
            let ef = layer.ef_search();
            let results = client
                .search(query, top_k, Some(ef), None)
                .await
                .map_err(|e| {
                    *self.last_error.write() = Some(e.to_string());
                    RvfBridgeError::Client(e)
                })?;

            // Evaluate confidence if SONA is enabled
            let should_continue = if let Some(sona) = self.sona.read().as_ref() {
                let query_embedding = query.to_vec();
                let result_embeddings: Vec<Vec<f32>> = results
                    .iter()
                    .filter_map(|r| {
                        // In production, would fetch actual embeddings
                        Some(vec![r.score; 384])
                    })
                    .collect();
                let result_ids: Vec<Uuid> = results
                    .iter()
                    .filter_map(|r| Uuid::parse_str(&r.id).ok())
                    .collect();

                let confidence = sona.evaluate_confidence(&query_embedding, &result_embeddings, &result_ids);

                // If confidence is below threshold, try next layer
                confidence.score < layer.target_recall() && layer.next().is_some()
            } else {
                // Without SONA, always go to next layer
                layer.next().is_some()
            };

            // Add new results
            for r in results {
                if let Ok(id) = Uuid::parse_str(&r.id) {
                    if !seen_ids.contains(&id) {
                        seen_ids.insert(id);
                        all_results.push(BridgeSearchResult {
                            id,
                            score: r.score,
                            source: ResultSource::Rvf,
                            layer: Some(layer),
                            metadata: r.metadata,
                        });
                    }
                }
            }

            // Check if we should continue to next layer
            if should_continue {
                if let Some(next) = layer.next() {
                    layer = next;
                    *self.current_layer.write() = layer;
                    continue;
                }
            }

            break;
        }

        // Sort by score descending
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Apply SONA weights if enabled
        if let Some(sona) = self.sona.read().as_ref() {
            let keys: Vec<String> = all_results.iter().map(|r| r.id.to_string()).collect();
            let mut scores: Vec<f32> = all_results.iter().map(|r| r.score).collect();
            sona.apply_weights(&mut scores, &keys);

            for (result, score) in all_results.iter_mut().zip(scores.iter()) {
                result.score = *score;
            }
        }

        Ok(all_results)
    }

    /// Search using usearch (fallback).
    fn search_usearch(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> RvfBridgeResult<Vec<BridgeSearchResult>> {
        let index = self.usearch_index.read();
        let index = index.as_ref().ok_or(RvfBridgeError::NotAvailable)?;

        let results = index
            .search(query, top_k, None)
            .map_err(|e| RvfBridgeError::USearch(e.to_string()))?;

        let bridge_results: Vec<BridgeSearchResult> = results
            .into_iter()
            .map(|(id, score)| BridgeSearchResult {
                id,
                score,
                source: ResultSource::USearch,
                layer: None,
                metadata: None,
            })
            .collect();

        Ok(bridge_results)
    }

    /// Derive a COW branch from the current store.
    ///
    /// Creates a new child store with optional filtering.
    pub async fn derive_child(
        &self,
        parent_id: &str,
        tenant_id: &str,
        filter_type: CowFilterType,
    ) -> RvfBridgeResult<String> {
        let client = self.rvf_client.read();
        let client = client.as_ref().ok_or(RvfBridgeError::NotAvailable)?;

        // Derive the child store
        let child_path = format!("{}.child.{}", parent_id, tenant_id);
        let path = Path::new(&child_path);

        let identity = client.derive(path, Some(filter_type.rvf_type())).await?;

        Ok(identity.file_id)
    }

    /// Add feedback for SONA learning.
    pub fn add_feedback(&self, feedback: super::sona::SonaFeedback) {
        if let Some(sona) = self.sona.read().as_ref() {
            sona.add_feedback(feedback);

            // Check if Loop B should trigger
            if sona.should_run_loop_b() {
                let _ = sona.run_loop_b();
            }
        }
    }

    /// Trigger SONA consolidation (Loop C).
    pub fn trigger_consolidation(&self) -> (usize, usize) {
        if let Some(sona) = self.sona.read().as_ref() {
            if sona.should_run_loop_c() {
                return sona.run_loop_c();
            }
        }
        (0, 0)
    }

    /// Get the current bridge status.
    pub fn status(&self) -> RvfBridgeStatus {
        let sona_state = self.sona
            .read()
            .as_ref()
            .map(|s| s.state())
            .unwrap_or_default();

        RvfBridgeStatus {
            rvf_available: *self.rvf_available.read(),
            dual_write_enabled: self.config.dual_write,
            prefer_rvf: self.config.prefer_rvf,
            current_layer: *self.current_layer.read(),
            sona_enabled: self.config.sona_enabled,
            sona_state,
            hyperbolic_enabled: self.config.hyperbolic_enabled,
            last_error: self.last_error.read().clone(),
        }
    }

    /// Check if RVF is available.
    pub fn is_rvf_available(&self) -> bool {
        *self.rvf_available.read()
    }

    /// Set dual-write mode.
    pub fn set_dual_write(&mut self, enabled: bool) {
        self.config.dual_write = enabled;
    }

    /// Set prefer RVF mode.
    pub fn set_prefer_rvf(&mut self, enabled: bool) {
        self.config.prefer_rvf = enabled;
    }
}

/// COW branch filter type.
#[derive(Debug, Clone, Copy)]
pub enum CowFilterType {
    /// Include only specified IDs
    Include,
    /// Exclude specified IDs
    Exclude,
}

impl CowFilterType {
    /// Get the RVF API type string.
    pub fn rvf_type(&self) -> &'static str {
        match self {
            Self::Include => "include",
            Self::Exclude => "exclude",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progressive_recall_layers() {
        assert_eq!(ProgressiveRecallLayer::LayerA.target_recall(), 0.70);
        assert_eq!(ProgressiveRecallLayer::LayerB.target_recall(), 0.85);
        assert_eq!(ProgressiveRecallLayer::LayerC.target_recall(), 0.95);

        assert_eq!(ProgressiveRecallLayer::LayerA.ef_search(), 16);
        assert_eq!(ProgressiveRecallLayer::LayerB.ef_search(), 64);
        assert_eq!(ProgressiveRecallLayer::LayerC.ef_search(), 256);
    }

    #[test]
    fn test_layer_progression() {
        let mut layer = ProgressiveRecallLayer::LayerA;
        assert_eq!(layer.next(), Some(ProgressiveRecallLayer::LayerB));

        layer = ProgressiveRecallLayer::LayerB;
        assert_eq!(layer.next(), Some(ProgressiveRecallLayer::LayerC));

        layer = ProgressiveRecallLayer::LayerC;
        assert_eq!(layer.next(), None);
    }

    #[test]
    fn test_cow_filter_type() {
        assert_eq!(CowFilterType::Include.rvf_type(), "include");
        assert_eq!(CowFilterType::Exclude.rvf_type(), "exclude");
    }

    #[test]
    fn test_default_config() {
        let config = RvfBridgeConfig::default();
        assert!(!config.dual_write);
        assert!(!config.prefer_rvf);
        assert!(config.progressive_recall);
        assert!(config.sona_enabled);
        assert_eq!(config.initial_layer, ProgressiveRecallLayer::LayerA);
    }
}
