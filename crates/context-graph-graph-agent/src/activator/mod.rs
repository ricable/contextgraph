//! E8 embedder activator for graph relationships.
//!
//! This module generates asymmetric E8 embeddings (source/target) for
//! confirmed graph relationships and stores them in the graph.
//!
//! # E8 Dimension Change
//!
//! E8 uses e5-large-v2 (1024D) instead of MiniLM (384D).
//! This shares the model with E1, avoiding extra VRAM usage.
//!
//! # Asymmetric Embeddings
//!
//! Like E5 (causal), E8 produces two vectors per memory:
//! - **source_vec**: Embedding for outgoing relationships (A -> ?)
//! - **target_vec**: Embedding for incoming relationships (? -> A)

use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use uuid::Uuid;

#[cfg(feature = "test-mode")]
use tracing::debug;

use context_graph_embeddings::models::GraphModel;

use crate::error::{GraphAgentError, GraphAgentResult};
use crate::types::{GraphAnalysisResult, GraphLinkDirection, RelationshipType};

// These are re-exported for tests
#[cfg(test)]
use crate::types::{ContentDomain, RelationshipCategory};

/// Configuration for the E8 activator.
#[derive(Debug, Clone)]
pub struct ActivatorConfig {
    /// Minimum confidence threshold for activation.
    pub min_confidence: f32,

    /// Whether to update existing relationships.
    pub update_existing: bool,

    /// Bidirectional confidence multiplier (default: 0.8).
    pub bidirectional_multiplier: f32,
}

impl Default for ActivatorConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            update_existing: true,
            bidirectional_multiplier: 0.8,
        }
    }
}

/// Statistics for activation operations.
#[derive(Debug, Clone, Default)]
pub struct ActivationStats {
    /// Total relationships processed.
    pub processed: usize,
    /// Embeddings generated.
    pub embeddings_generated: usize,
    /// Graph edges created.
    pub edges_created: usize,
    /// Skipped due to low confidence.
    pub skipped_low_confidence: usize,
    /// Skipped due to existing relationship.
    pub skipped_existing: usize,
    /// Errors encountered.
    pub errors: usize,
}

impl ActivationStats {
    /// Reset all statistics to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Stored graph edge with metadata.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Source memory ID.
    pub source_id: Uuid,
    /// Target memory ID.
    pub target_id: Uuid,
    /// Relationship type.
    pub relationship_type: RelationshipType,
    /// Confidence score.
    pub confidence: f32,
    /// LLM description of the relationship.
    pub description: String,
    /// When the edge was created.
    pub created_at: DateTime<Utc>,
}

impl GraphEdge {
    /// Create a new graph edge.
    pub fn new(
        source_id: Uuid,
        target_id: Uuid,
        relationship_type: RelationshipType,
        confidence: f32,
        description: String,
    ) -> Self {
        Self {
            source_id,
            target_id,
            relationship_type,
            confidence,
            description,
            created_at: Utc::now(),
        }
    }
}

/// Simple in-memory graph storage for discovered relationships.
///
/// In production, this would be backed by persistent storage.
pub struct GraphStorage {
    /// All edges in the graph.
    edges: Vec<GraphEdge>,
}

impl GraphStorage {
    /// Create a new empty graph storage.
    pub fn new() -> Self {
        Self { edges: Vec::new() }
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, edge: GraphEdge) {
        self.edges.push(edge);
    }

    /// Check if a direct edge exists between two nodes.
    pub fn has_edge(&self, source_id: Uuid, target_id: Uuid) -> bool {
        self.edges
            .iter()
            .any(|e| e.source_id == source_id && e.target_id == target_id)
    }

    /// Get all edges from a source node.
    pub fn edges_from(&self, source_id: Uuid) -> Vec<&GraphEdge> {
        self.edges
            .iter()
            .filter(|e| e.source_id == source_id)
            .collect()
    }

    /// Get all edges to a target node.
    pub fn edges_to(&self, target_id: Uuid) -> Vec<&GraphEdge> {
        self.edges
            .iter()
            .filter(|e| e.target_id == target_id)
            .collect()
    }

    /// Get total number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get all edges.
    pub fn all_edges(&self) -> &[GraphEdge] {
        &self.edges
    }
}

impl Default for GraphStorage {
    fn default() -> Self {
        Self::new()
    }
}

/// E8 embedder activator for storing graph relationships.
///
/// This activator:
/// 1. Validates relationship confidence
/// 2. Generates asymmetric E8 embeddings using GraphModel
/// 3. Stores edges in the graph
///
/// # Production vs Test Mode
///
/// In production (without `test-mode` feature), GraphModel is REQUIRED.
/// Use `with_model()` to create activator with real embedding capability.
/// In test mode, placeholder embeddings are allowed when GraphModel is unavailable.
pub struct E8Activator {
    config: ActivatorConfig,
    graph: Arc<RwLock<GraphStorage>>,
    stats: RwLock<ActivationStats>,
    /// Graph model for generating real E8 embeddings.
    /// When None, behavior depends on `test-mode` feature:
    /// - Without feature: fail fast with error
    /// - With feature: use placeholder embeddings
    graph_model: Option<Arc<GraphModel>>,
}

impl E8Activator {
    /// Create a new E8 activator (test mode only, no real embeddings).
    ///
    /// In production (without `test-mode` feature), this will fail when
    /// `activate_relationship` is called. Use `with_model()` instead.
    pub fn new() -> Self {
        Self::with_config(ActivatorConfig::default())
    }

    /// Create with custom configuration (test mode only, no real embeddings).
    ///
    /// In production (without `test-mode` feature), this will fail when
    /// `activate_relationship` is called. Use `with_model()` instead.
    pub fn with_config(config: ActivatorConfig) -> Self {
        Self {
            config,
            graph: Arc::new(RwLock::new(GraphStorage::new())),
            stats: RwLock::new(ActivationStats::default()),
            graph_model: None,
        }
    }

    /// Create with shared graph storage (test mode only, no real embeddings).
    ///
    /// In production (without `test-mode` feature), this will fail when
    /// `activate_relationship` is called. Use `with_model()` instead.
    pub fn with_graph(graph: Arc<RwLock<GraphStorage>>) -> Self {
        Self {
            config: ActivatorConfig::default(),
            graph,
            stats: RwLock::new(ActivationStats::default()),
            graph_model: None,
        }
    }

    /// Create with a real GraphModel for production use.
    ///
    /// This is the recommended constructor for production deployments.
    /// The GraphModel generates real 1024D asymmetric embeddings.
    ///
    /// # Arguments
    /// * `graph_model` - Loaded GraphModel for E8 embeddings
    /// * `config` - Configuration options
    pub fn with_model(graph_model: Arc<GraphModel>, config: ActivatorConfig) -> Self {
        Self {
            config,
            graph: Arc::new(RwLock::new(GraphStorage::new())),
            stats: RwLock::new(ActivationStats::default()),
            graph_model: Some(graph_model),
        }
    }

    /// Create with both GraphModel and shared graph storage.
    pub fn with_model_and_graph(
        graph_model: Arc<GraphModel>,
        graph: Arc<RwLock<GraphStorage>>,
        config: ActivatorConfig,
    ) -> Self {
        Self {
            config,
            graph,
            stats: RwLock::new(ActivationStats::default()),
            graph_model: Some(graph_model),
        }
    }

    /// Activate a confirmed graph relationship.
    ///
    /// # Arguments
    /// * `source_id` - Memory A ID
    /// * `target_id` - Memory B ID
    /// * `source_content` - Memory A content (for embedding)
    /// * `target_content` - Memory B content (for embedding)
    /// * `analysis` - LLM analysis result
    ///
    /// # Returns
    /// Tuple of (source_embedding, target_embedding) - 1024D asymmetric vectors
    ///
    /// # Errors
    /// Returns error if GraphModel is not available and test-mode feature is not enabled.
    pub async fn activate_relationship(
        &self,
        source_id: Uuid,
        target_id: Uuid,
        source_content: &str,
        target_content: &str,
        analysis: &GraphAnalysisResult,
    ) -> GraphAgentResult<(Vec<f32>, Vec<f32>)> {
        // Phase 1: Validation and graph updates (sync, hold locks briefly)
        {
            let mut stats = self.stats.write();
            stats.processed += 1;

            // Check confidence threshold
            if analysis.confidence < self.config.min_confidence {
                stats.skipped_low_confidence += 1;
                return Err(GraphAgentError::ConfigError {
                    message: format!(
                        "Confidence {} below threshold {}",
                        analysis.confidence, self.config.min_confidence
                    ),
                });
            }
        } // Release stats lock

        // Check if relationship already exists and add edges
        let edges_added = {
            let mut graph = self.graph.write();
            if !self.config.update_existing {
                let (effective_source, effective_target) =
                    self.get_effective_source_target(source_id, target_id, analysis.direction);

                if graph.has_edge(effective_source, effective_target) {
                    self.stats.write().skipped_existing += 1;
                    return Err(GraphAgentError::ConfigError {
                        message: "Relationship already exists".to_string(),
                    });
                }
            }

            // Add edges based on direction
            self.add_edges_for_direction(
                &mut graph,
                source_id,
                target_id,
                analysis,
            )
        }; // Release graph lock

        // Update edges_created stat
        self.stats.write().edges_created += edges_added;

        // Phase 2: Generate embeddings (async, no locks held)
        // Combine source and target content for embedding context
        let combined_content = format!("{}\n\nRelated to:\n{}", source_content, target_content);

        let (source_embedding, target_embedding) =
            self.generate_e8_embeddings(&combined_content).await?;

        // Phase 3: Update final stats
        self.stats.write().embeddings_generated += 2;

        Ok((source_embedding, target_embedding))
    }

    /// Generate E8 asymmetric embeddings for source and target roles.
    ///
    /// When a real GraphModel is available, uses embed_dual() for genuine asymmetric embeddings.
    /// In test-mode, falls back to placeholder embeddings for testing without GPU.
    /// In production (without test-mode feature), fails fast if GraphModel unavailable.
    async fn generate_e8_embeddings(
        &self,
        content: &str,
    ) -> GraphAgentResult<(Vec<f32>, Vec<f32>)> {
        match &self.graph_model {
            Some(model) => {
                // Use real GraphModel for production embeddings
                model
                    .embed_dual(content)
                    .await
                    .map_err(|e| GraphAgentError::EmbeddingError {
                        message: format!("Failed to generate E8 embeddings: {}", e),
                    })
            }
            None => {
                // Production mode: fail fast - GraphModel is required
                #[cfg(not(feature = "test-mode"))]
                {
                    return Err(GraphAgentError::ConfigError {
                        message: "GraphModel required in production but not available. \
                                  Use E8Activator::with_model() or enable test-mode feature."
                            .to_string(),
                    });
                }

                // Test mode: allow placeholder embeddings
                #[cfg(feature = "test-mode")]
                {
                    debug!("Using placeholder E8 embeddings (test-mode, no GraphModel)");
                    self.generate_placeholder_embeddings(content)
                }
            }
        }
    }

    /// Generate placeholder embeddings for testing.
    ///
    /// Creates deterministic 1024D vectors based on content hash.
    /// Source and target vectors differ by using different seeds.
    #[cfg(feature = "test-mode")]
    fn generate_placeholder_embeddings(
        &self,
        content: &str,
    ) -> GraphAgentResult<(Vec<f32>, Vec<f32>)> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn hash_to_vec(content: &str, seed: &str, dim: usize) -> Vec<f32> {
            let mut hasher = DefaultHasher::new();
            content.hash(&mut hasher);
            seed.hash(&mut hasher);
            let hash = hasher.finish();

            let mut embedding = Vec::with_capacity(dim);
            let mut current = hash;
            let mut sum_sq = 0.0f32;

            for _ in 0..dim {
                current = current.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let val = ((current >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
                embedding.push(val);
                sum_sq += val * val;
            }

            // L2 normalize
            let norm = sum_sq.sqrt();
            if norm > f32::EPSILON {
                for val in &mut embedding {
                    *val /= norm;
                }
            }
            embedding
        }

        let source_vec = hash_to_vec(content, "source", 1024);
        let target_vec = hash_to_vec(content, "target", 1024);
        Ok((source_vec, target_vec))
    }

    /// Add edges to the graph based on relationship direction.
    fn add_edges_for_direction(
        &self,
        graph: &mut GraphStorage,
        memory_a_id: Uuid,
        memory_b_id: Uuid,
        analysis: &GraphAnalysisResult,
    ) -> usize {
        let mut edges_added = 0;

        match analysis.direction {
            GraphLinkDirection::AConnectsB => {
                // A -> B
                graph.add_edge(GraphEdge::new(
                    memory_a_id,
                    memory_b_id,
                    analysis.relationship_type,
                    analysis.confidence,
                    analysis.description.clone(),
                ));
                edges_added += 1;
            }
            GraphLinkDirection::BConnectsA => {
                // B -> A
                graph.add_edge(GraphEdge::new(
                    memory_b_id,
                    memory_a_id,
                    analysis.relationship_type,
                    analysis.confidence,
                    analysis.description.clone(),
                ));
                edges_added += 1;
            }
            GraphLinkDirection::Bidirectional => {
                // A <-> B (both directions with reduced confidence)
                let reduced_confidence = analysis.confidence * self.config.bidirectional_multiplier;

                graph.add_edge(GraphEdge::new(
                    memory_a_id,
                    memory_b_id,
                    analysis.relationship_type,
                    reduced_confidence,
                    analysis.description.clone(),
                ));
                graph.add_edge(GraphEdge::new(
                    memory_b_id,
                    memory_a_id,
                    analysis.relationship_type,
                    reduced_confidence,
                    analysis.description.clone(),
                ));
                edges_added += 2;
            }
            GraphLinkDirection::NoConnection => {
                // No edges to add
            }
        }

        edges_added
    }

    /// Get effective source and target based on direction.
    fn get_effective_source_target(
        &self,
        memory_a_id: Uuid,
        memory_b_id: Uuid,
        direction: GraphLinkDirection,
    ) -> (Uuid, Uuid) {
        match direction {
            GraphLinkDirection::AConnectsB | GraphLinkDirection::Bidirectional => {
                (memory_a_id, memory_b_id)
            }
            GraphLinkDirection::BConnectsA => (memory_b_id, memory_a_id),
            GraphLinkDirection::NoConnection => (memory_a_id, memory_b_id),
        }
    }

    /// Get current statistics.
    pub fn stats(&self) -> ActivationStats {
        self.stats.read().clone()
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        self.stats.write().reset();
    }

    /// Get the graph storage.
    pub fn graph(&self) -> Arc<RwLock<GraphStorage>> {
        Arc::clone(&self.graph)
    }
}

impl Default for E8Activator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_activator_default_config() {
        let activator = E8Activator::new();
        assert!((activator.config.min_confidence - 0.6).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_activate_low_confidence() {
        let activator = E8Activator::new();
        let analysis = GraphAnalysisResult {
            has_connection: true,
            direction: GraphLinkDirection::AConnectsB,
            relationship_type: RelationshipType::Imports,
            category: RelationshipCategory::Dependency,
            domain: ContentDomain::Code,
            confidence: 0.3, // Below threshold
            description: "A imports B".to_string(),
            raw_response: None,
            llm_provenance: None,
        };

        let result = activator
            .activate_relationship(Uuid::new_v4(), Uuid::new_v4(), "a", "b", &analysis)
            .await;

        assert!(result.is_err());
        assert_eq!(activator.stats().skipped_low_confidence, 1);
    }

    // Test that activation fails fast in production mode without GraphModel
    #[cfg(not(feature = "test-mode"))]
    #[tokio::test]
    async fn test_e8_activator_fails_fast_without_model() {
        let activator = E8Activator::new(); // No model
        let analysis = GraphAnalysisResult {
            has_connection: true,
            direction: GraphLinkDirection::AConnectsB,
            relationship_type: RelationshipType::Imports,
            category: RelationshipCategory::Dependency,
            domain: ContentDomain::Code,
            confidence: 0.85,
            description: "A imports B".to_string(),
            raw_response: None,
            llm_provenance: None,
        };

        let result = activator
            .activate_relationship(Uuid::new_v4(), Uuid::new_v4(), "source", "target", &analysis)
            .await;

        // Should fail because GraphModel is required in production
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("GraphModel required") || err.contains("production"),
            "Error should mention GraphModel requirement: {}",
            err
        );
    }

    // Test that placeholder embeddings work in test-mode
    #[cfg(feature = "test-mode")]
    #[tokio::test]
    async fn test_activate_success_test_mode() {
        let activator = E8Activator::new();
        let analysis = GraphAnalysisResult {
            has_connection: true,
            direction: GraphLinkDirection::AConnectsB,
            relationship_type: RelationshipType::Imports,
            category: RelationshipCategory::Dependency,
            domain: ContentDomain::Code,
            confidence: 0.85,
            description: "A imports B".to_string(),
            raw_response: None,
            llm_provenance: None,
        };

        let result = activator
            .activate_relationship(Uuid::new_v4(), Uuid::new_v4(), "source content", "target content", &analysis)
            .await;

        assert!(result.is_ok(), "Should use placeholder embeddings in test-mode: {:?}", result);
        let (source_emb, target_emb) = result.unwrap();

        // Verify dimensions
        assert_eq!(source_emb.len(), 1024, "E8 embeddings should be 1024D");
        assert_eq!(target_emb.len(), 1024, "E8 embeddings should be 1024D");

        // Source and target should be different (asymmetric)
        assert_ne!(source_emb, target_emb, "Source and target embeddings should differ");

        // Verify non-zero
        let source_sum: f32 = source_emb.iter().map(|x| x.abs()).sum();
        let target_sum: f32 = target_emb.iter().map(|x| x.abs()).sum();
        assert!(source_sum > 0.0, "Source embedding should be non-zero");
        assert!(target_sum > 0.0, "Target embedding should be non-zero");

        let stats = activator.stats();
        assert_eq!(stats.edges_created, 1);
        assert_eq!(stats.embeddings_generated, 2);
    }

    #[cfg(feature = "test-mode")]
    #[tokio::test]
    async fn test_activate_bidirectional_test_mode() {
        let activator = E8Activator::new();
        let analysis = GraphAnalysisResult {
            has_connection: true,
            direction: GraphLinkDirection::Bidirectional,
            relationship_type: RelationshipType::References,
            category: RelationshipCategory::Reference,
            domain: ContentDomain::General,
            confidence: 0.9,
            description: "Mutual reference".to_string(),
            raw_response: None,
        };

        let result = activator
            .activate_relationship(Uuid::new_v4(), Uuid::new_v4(), "a", "b", &analysis)
            .await;

        assert!(result.is_ok());
        assert_eq!(activator.stats().edges_created, 2);
    }

    #[test]
    fn test_graph_storage() {
        let mut graph = GraphStorage::new();
        let source = Uuid::new_v4();
        let target = Uuid::new_v4();

        graph.add_edge(GraphEdge::new(
            source,
            target,
            RelationshipType::Imports,
            0.9,
            "Test".to_string(),
        ));

        assert!(graph.has_edge(source, target));
        assert!(!graph.has_edge(target, source));
        assert_eq!(graph.edge_count(), 1);
    }
}
