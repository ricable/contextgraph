//! CausalRelationship type for storing LLM-generated causal descriptions with provenance.
//!
//! This module defines the [`CausalRelationship`] struct that stores:
//! - LLM-generated 1-3 paragraph causal explanations
//! - E5 dual embeddings (as_cause, as_effect) for asymmetric search (768D each)
//! - E1 semantic embedding for fallback search (1024D)
//! - Full provenance linking back to source content and fingerprint
//!
//! # Storage
//!
//! CausalRelationships are stored in dedicated RocksDB column families:
//! - `CF_CAUSAL_RELATIONSHIPS`: Primary storage by UUID
//! - `CF_CAUSAL_BY_SOURCE`: Secondary index by source fingerprint ID
//! - `CF_CAUSAL_E5_CAUSE_INDEX`: HNSW index for e5_as_cause vectors
//! - `CF_CAUSAL_E5_EFFECT_INDEX`: HNSW index for e5_as_effect vectors
//!
//! # Search
//!
//! E5 dual embeddings enable asymmetric causal search:
//! - "What caused X?" → Query as effect, search cause index
//! - "What are effects of X?" → Query as cause, search effect index
//!
//! E1 embedding provides fallback semantic search for general queries.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A causal relationship identified by LLM analysis.
///
/// Contains the LLM-generated explanation with E5 dual embeddings for
/// asymmetric causal search, plus full provenance linking back to source.
///
/// # Embeddings
///
/// - `e5_as_cause` (768D): Explanation embedded as a cause
/// - `e5_as_effect` (768D): Explanation embedded as an effect
/// - `e1_semantic` (1024D): Fallback semantic embedding
///
/// # Search Strategy
///
/// For "What caused X?":
/// - Query X using `embed_as_effect`
/// - Search `e5_as_cause` index
/// - Returns relationships where explanation acts as a cause of X
///
/// For "What are effects of X?":
/// - Query X using `embed_as_cause`
/// - Search `e5_as_effect` index
/// - Returns relationships where explanation is an effect of X
///
/// # Example
///
/// ```ignore
/// let rel = CausalRelationship::new(
///     "Chronic stress elevates cortisol".to_string(),  // cause
///     "Memory impairment".to_string(),                 // effect
///     "High cortisol causes memory impairment...".to_string(), // explanation
///     e5_cause, e5_effect, e1_semantic,
///     source_id,
///     0.9,
///     "mediated".to_string(),
/// );
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalRelationship {
    /// Unique ID for this causal relationship.
    pub id: Uuid,

    // ===== CAUSAL CONTENT =====

    /// Brief statement of the cause (e.g., "Chronic stress elevates cortisol")
    pub cause_statement: String,

    /// Brief statement of the effect (e.g., "Memory impairment")
    pub effect_statement: String,

    /// LLM-generated explanation (1-2 paragraphs).
    ///
    /// Structure:
    /// - Paragraph 1: What is the causal relationship and why
    /// - Paragraph 2: Mechanism/evidence details
    pub explanation: String,

    // ===== E5 DUAL EMBEDDINGS (768D each) =====

    /// E5 embedding of explanation as a CAUSE (768D).
    ///
    /// Used when searching for effects: "What does X cause?"
    pub e5_as_cause: Vec<f32>,

    /// E5 embedding of explanation as an EFFECT (768D).
    ///
    /// Used when searching for causes: "What caused X?"
    pub e5_as_effect: Vec<f32>,

    // ===== E1 SEMANTIC EMBEDDING (1024D) =====

    /// E1 semantic embedding of the explanation (1024D).
    ///
    /// Provides fallback for general semantic search.
    pub e1_semantic: Vec<f32>,

    // ===== METADATA =====

    /// LLM confidence score [0.0, 1.0].
    pub confidence: f32,

    /// Type of causal mechanism: "direct", "mediated", "feedback", "temporal"
    pub mechanism_type: String,

    // ===== PROVENANCE =====

    /// Original source content this was derived from.
    pub source_content: String,

    /// Fingerprint ID of the source memory.
    pub source_fingerprint_id: Uuid,

    /// Unix timestamp when relationship was identified.
    pub created_at: i64,
}

impl CausalRelationship {
    /// E5 embedding dimension (768D per constitution.yaml).
    pub const E5_DIM: usize = 768;

    /// E1 embedding dimension (1024D per constitution.yaml).
    pub const E1_DIM: usize = 1024;

    /// Create a new causal relationship with all embeddings.
    ///
    /// # Arguments
    ///
    /// * `cause_statement` - Brief statement of the cause
    /// * `effect_statement` - Brief statement of the effect
    /// * `explanation` - LLM-generated 1-2 paragraph explanation
    /// * `e5_as_cause` - E5 768D embedding as cause
    /// * `e5_as_effect` - E5 768D embedding as effect
    /// * `e1_semantic` - E1 1024D semantic embedding
    /// * `source_content` - Original text this was derived from
    /// * `source_fingerprint_id` - UUID of the source memory
    /// * `confidence` - LLM confidence score [0.0, 1.0]
    /// * `mechanism_type` - "direct", "mediated", "feedback", or "temporal"
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cause_statement: String,
        effect_statement: String,
        explanation: String,
        e5_as_cause: Vec<f32>,
        e5_as_effect: Vec<f32>,
        e1_semantic: Vec<f32>,
        source_content: String,
        source_fingerprint_id: Uuid,
        confidence: f32,
        mechanism_type: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            cause_statement,
            effect_statement,
            explanation,
            e5_as_cause,
            e5_as_effect,
            e1_semantic,
            confidence: confidence.clamp(0.0, 1.0),
            mechanism_type,
            source_content,
            source_fingerprint_id,
            created_at: chrono::Utc::now().timestamp(),
        }
    }

    /// Create with a specific ID (for testing or migration).
    #[allow(clippy::too_many_arguments)]
    pub fn with_id(
        id: Uuid,
        cause_statement: String,
        effect_statement: String,
        explanation: String,
        e5_as_cause: Vec<f32>,
        e5_as_effect: Vec<f32>,
        e1_semantic: Vec<f32>,
        source_content: String,
        source_fingerprint_id: Uuid,
        confidence: f32,
        mechanism_type: String,
    ) -> Self {
        Self {
            id,
            cause_statement,
            effect_statement,
            explanation,
            e5_as_cause,
            e5_as_effect,
            e1_semantic,
            confidence: confidence.clamp(0.0, 1.0),
            mechanism_type,
            source_content,
            source_fingerprint_id,
            created_at: chrono::Utc::now().timestamp(),
        }
    }

    /// Check if all embeddings have the expected dimensions.
    ///
    /// - E5 embeddings: 768D
    /// - E1 embedding: 1024D
    pub fn has_valid_embeddings(&self) -> bool {
        self.e5_as_cause.len() == Self::E5_DIM
            && self.e5_as_effect.len() == Self::E5_DIM
            && self.e1_semantic.len() == Self::E1_DIM
    }

    /// Get the E5 cause embedding as a slice.
    pub fn e5_cause_embedding(&self) -> &[f32] {
        &self.e5_as_cause
    }

    /// Get the E5 effect embedding as a slice.
    pub fn e5_effect_embedding(&self) -> &[f32] {
        &self.e5_as_effect
    }

    /// Get the E1 semantic embedding as a slice.
    pub fn e1_embedding(&self) -> &[f32] {
        &self.e1_semantic
    }

    /// Check if this is a high-confidence relationship (>= 0.7).
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.7
    }

    /// Get the mechanism type as a normalized string.
    pub fn normalized_mechanism_type(&self) -> &str {
        match self.mechanism_type.to_lowercase().as_str() {
            "direct" => "direct",
            "mediated" | "indirect" => "mediated",
            "feedback" | "bidirectional" | "loop" => "feedback",
            "temporal" | "sequence" => "temporal",
            _ => "direct",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_embeddings() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            vec![0.0_f32; CausalRelationship::E5_DIM], // e5_as_cause
            vec![0.0_f32; CausalRelationship::E5_DIM], // e5_as_effect
            vec![0.0_f32; CausalRelationship::E1_DIM], // e1_semantic
        )
    }

    #[test]
    fn test_causal_relationship_new() {
        let (e5_cause, e5_effect, e1) = test_embeddings();
        let source_id = Uuid::new_v4();

        let rel = CausalRelationship::new(
            "Chronic stress elevates cortisol".to_string(),
            "Memory impairment".to_string(),
            "High cortisol causes memory impairment...".to_string(),
            e5_cause,
            e5_effect,
            e1,
            "Source content".to_string(),
            source_id,
            0.9,
            "mediated".to_string(),
        );

        assert!(!rel.id.is_nil());
        assert_eq!(rel.cause_statement, "Chronic stress elevates cortisol");
        assert_eq!(rel.effect_statement, "Memory impairment");
        assert_eq!(rel.source_fingerprint_id, source_id);
        assert_eq!(rel.mechanism_type, "mediated");
        assert!((rel.confidence - 0.9).abs() < 0.01);
        assert!(rel.has_valid_embeddings());
    }

    #[test]
    fn test_confidence_clamping() {
        let (e5_cause, e5_effect, e1) = test_embeddings();
        let source_id = Uuid::new_v4();

        let rel = CausalRelationship::new(
            "Cause".to_string(),
            "Effect".to_string(),
            "Test".to_string(),
            e5_cause,
            e5_effect,
            e1,
            "Source".to_string(),
            source_id,
            1.5, // Exceeds max
            "direct".to_string(),
        );

        assert_eq!(rel.confidence, 1.0);
    }

    #[test]
    fn test_normalized_mechanism_type() {
        let (e5_cause, e5_effect, e1) = test_embeddings();
        let source_id = Uuid::new_v4();

        let rel = CausalRelationship::new(
            "Cause".to_string(),
            "Effect".to_string(),
            "Test".to_string(),
            e5_cause,
            e5_effect,
            e1,
            "Source".to_string(),
            source_id,
            0.8,
            "MEDIATED".to_string(), // Different case
        );

        assert_eq!(rel.normalized_mechanism_type(), "mediated");
    }

    #[test]
    fn test_is_high_confidence() {
        let (e5_cause1, e5_effect1, e1_1) = test_embeddings();
        let (e5_cause2, e5_effect2, e1_2) = test_embeddings();
        let source_id = Uuid::new_v4();

        let high = CausalRelationship::new(
            "Cause".to_string(),
            "Effect".to_string(),
            "Test".to_string(),
            e5_cause1,
            e5_effect1,
            e1_1,
            "Source".to_string(),
            source_id,
            0.8,
            "direct".to_string(),
        );

        let low = CausalRelationship::new(
            "Cause".to_string(),
            "Effect".to_string(),
            "Test".to_string(),
            e5_cause2,
            e5_effect2,
            e1_2,
            "Source".to_string(),
            source_id,
            0.5,
            "direct".to_string(),
        );

        assert!(high.is_high_confidence());
        assert!(!low.is_high_confidence());
    }

    #[test]
    fn test_embedding_accessors() {
        let (e5_cause, e5_effect, e1) = test_embeddings();
        let source_id = Uuid::new_v4();

        let rel = CausalRelationship::new(
            "Cause".to_string(),
            "Effect".to_string(),
            "Test".to_string(),
            e5_cause,
            e5_effect,
            e1,
            "Source".to_string(),
            source_id,
            0.8,
            "direct".to_string(),
        );

        assert_eq!(rel.e5_cause_embedding().len(), CausalRelationship::E5_DIM);
        assert_eq!(rel.e5_effect_embedding().len(), CausalRelationship::E5_DIM);
        assert_eq!(rel.e1_embedding().len(), CausalRelationship::E1_DIM);
    }
}
