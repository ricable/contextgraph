//! Memory node representing a stored memory unit in the knowledge graph.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;

use super::{JohariQuadrant, Modality};

/// Unique identifier for memory nodes
pub type NodeId = Uuid;

/// Embedding vector type (1536 dimensions for OpenAI-compatible)
pub type EmbeddingVector = Vec<f32>;

/// Default embedding dimension (OpenAI text-embedding-3-large compatible).
/// Per constitution.yaml: embeddings.models.E7_Code = 1536D
pub const DEFAULT_EMBEDDING_DIM: usize = 1536;

/// Maximum content size in bytes (1MB).
/// Per constitution.yaml: perf.memory constraints
pub const MAX_CONTENT_SIZE: usize = 1_048_576;

/// Errors that occur during MemoryNode validation.
///
/// Each variant provides specific context about what validation failed
/// and what values were involved, enabling actionable error messages.
///
/// # Constitution Compliance
/// - AP-009: Prevents NaN/Infinity by validating before storage
/// - Naming: PascalCase enum, snake_case fields
///
/// # Example
/// ```rust
/// use context_graph_core::types::ValidationError;
///
/// let error = ValidationError::InvalidEmbeddingDimension {
///     expected: 1536,
///     actual: 768,
/// };
/// assert!(error.to_string().contains("expected 1536"));
/// ```
#[derive(Debug, Clone, Error, PartialEq)]
pub enum ValidationError {
    /// Embedding vector has incorrect dimensions.
    /// Expected: 1536 (per constitution.yaml embedding spec)
    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidEmbeddingDimension {
        /// Required dimension (1536)
        expected: usize,
        /// Actual dimension provided
        actual: usize,
    },

    /// A numeric field value is outside its valid range.
    /// Used for importance [0.0, 1.0], valence [-1.0, 1.0], etc.
    #[error("Field '{field}' value {value} is out of bounds [{min}, {max}]")]
    OutOfBounds {
        /// Name of the field that failed validation
        field: String,
        /// The invalid value provided
        value: f64,
        /// Minimum allowed value (inclusive)
        min: f64,
        /// Maximum allowed value (inclusive)
        max: f64,
    },

    /// Content exceeds maximum allowed size.
    /// Limit: 1MB (1,048,576 bytes) per constitution.yaml
    #[error("Content size {size} bytes exceeds maximum allowed {max_size} bytes")]
    ContentTooLarge {
        /// Actual content size in bytes
        size: usize,
        /// Maximum allowed size (1,048,576 bytes)
        max_size: usize,
    },

    /// Embedding vector is not normalized (magnitude should be ~1.0).
    /// Tolerance: magnitude must be in [0.99, 1.01]
    #[error("Embedding not normalized: magnitude is {magnitude:.6}, expected ~1.0")]
    EmbeddingNotNormalized {
        /// Actual magnitude of the embedding vector
        magnitude: f64,
    },
}

/// A memory node representing a single knowledge unit in the Context Graph.
///
/// # Performance Characteristics
/// - Average serialized size: ~6.5KB (with embedding)
/// - Insert latency target: <1ms
/// - Retrieval latency target: <500us
///
/// # PRD Section 4.1 KnowledgeNode Mapping
/// - `id`: UUID v4 unique identifier
/// - `content`: str[<=65536] - actual stored knowledge (max 1MB enforced)
/// - `embedding`: Vec1536 - dense vector representation
/// - `quadrant`: Johari Window classification
/// - `importance`: f32[0,1] - relevance score
/// - `emotional_valence`: f32[-1,1] - emotional charge
/// - `created_at`: Creation timestamp
/// - `accessed_at`: Last access timestamp
/// - `access_count`: Number of accesses
/// - `metadata`: Rich metadata container
///
/// # Constitution Compliance
/// - AP-009: All f32 fields must be validated (no NaN/Infinity)
/// - SEC-06: Soft delete via metadata.deleted flag
/// - Naming: snake_case fields per constitution.yaml
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryNode {
    /// Unique identifier for this node (UUID v4).
    pub id: NodeId,

    /// The content/knowledge stored in this node (max 1MB).
    pub content: String,

    /// Dense embedding vector (1536 dimensions by default).
    pub embedding: EmbeddingVector,

    /// Johari Window quadrant classification.
    pub quadrant: JohariQuadrant,

    /// Importance/relevance score [0.0, 1.0].
    pub importance: f32,

    /// Emotional valence [-1.0, 1.0].
    /// Negative = negative emotion, Positive = positive emotion.
    pub emotional_valence: f32,

    /// Timestamp when this node was created.
    pub created_at: DateTime<Utc>,

    /// Timestamp when this node was last accessed.
    pub accessed_at: DateTime<Utc>,

    /// Number of times this node has been accessed.
    pub access_count: u64,

    /// Rich metadata container.
    pub metadata: NodeMetadata,
}

impl MemoryNode {
    /// Tolerance for embedding normalization check (magnitude must be in [0.99, 1.01])
    const NORMALIZATION_TOLERANCE: f64 = 0.01;

    /// Consolidation threshold score (weighted score >= 0.7 triggers consolidation)
    const CONSOLIDATION_THRESHOLD: f32 = 0.7;

    /// Create a new memory node with the given content and embedding.
    ///
    /// # Arguments
    /// * `content` - The content/knowledge to store
    /// * `embedding` - The embedding vector (should be 1536 dimensions)
    ///
    /// # Default Values
    /// - `importance`: 0.5
    /// - `emotional_valence`: 0.0 (neutral)
    /// - `quadrant`: JohariQuadrant::Open
    /// - `access_count`: 0
    pub fn new(content: String, embedding: EmbeddingVector) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            content,
            embedding,
            quadrant: JohariQuadrant::default(),
            importance: 0.5,
            emotional_valence: 0.0,
            created_at: now,
            accessed_at: now,
            access_count: 0,
            metadata: NodeMetadata::default(),
        }
    }

    /// Create a new MemoryNode with a specific ID.
    ///
    /// # Arguments
    /// * `id` - The specific NodeId (UUID) to use
    /// * `content` - The content to store
    /// * `embedding` - The embedding vector (should be 1536 dimensions)
    ///
    /// # Example
    /// ```rust,ignore
    /// use uuid::Uuid;
    /// let id = Uuid::new_v4();
    /// let node = MemoryNode::with_id(id, "content".to_string(), vec![0.0; 1536]);
    /// assert_eq!(node.id, id);
    /// ```
    pub fn with_id(id: NodeId, content: String, embedding: EmbeddingVector) -> Self {
        let mut node = Self::new(content, embedding);
        node.id = id;
        node
    }

    /// Record an access to this node, updating accessed_at and incrementing access_count.
    ///
    /// Uses saturating_add to prevent overflow (will stay at u64::MAX if at limit).
    ///
    /// # Example
    /// ```rust,ignore
    /// let mut node = MemoryNode::new("test".to_string(), vec![0.0; 1536]);
    /// node.record_access();
    /// assert_eq!(node.access_count, 1);
    /// ```
    pub fn record_access(&mut self) {
        self.accessed_at = Utc::now();
        self.access_count = self.access_count.saturating_add(1);
    }

    /// Get the age of this node in seconds since creation.
    ///
    /// # Returns
    /// Number of seconds since node creation (always >= 0).
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.created_at).num_seconds()
    }

    /// Get the time since last access in seconds.
    ///
    /// # Returns
    /// Number of seconds since last access (always >= 0).
    pub fn time_since_access_seconds(&self) -> i64 {
        (Utc::now() - self.accessed_at).num_seconds()
    }

    /// Compute memory decay using modified Ebbinghaus forgetting curve.
    ///
    /// Formula: R = e^(-t / (S * k * 24))
    /// Where:
    /// - R = retention (0.0 to 1.0)
    /// - t = time since access in hours
    /// - S = memory strength: 1 + ln(access_count + 1)
    /// - k = importance factor: 1 + importance
    /// - 24 = baseline decay period in hours
    ///
    /// # Returns
    /// Retention value between 0.0 (forgotten) and 1.0 (fully retained).
    ///
    /// # Constitution Compliance
    /// - AP-009: Result is clamped to [0.0, 1.0] to prevent NaN/Infinity
    pub fn compute_decay(&self) -> f32 {
        let t_hours = self.time_since_access_seconds() as f64 / 3600.0;
        let strength = 1.0 + ((self.access_count as f64) + 1.0).ln().max(0.0);
        let k = 1.0 + self.importance as f64;
        let decay = (-t_hours / (strength * k * 24.0)).exp();
        decay.clamp(0.0, 1.0) as f32
    }

    /// Determine if this node should be consolidated based on weighted score.
    ///
    /// Score = 0.4 * importance + 0.3 * (1 - decay) + 0.3 * access_frequency
    /// Where access_frequency = accesses per hour, clamped to [0, 1]
    ///
    /// # Returns
    /// `true` if score >= CONSOLIDATION_THRESHOLD (0.7)
    pub fn should_consolidate(&self) -> bool {
        let decay = self.compute_decay();
        let age_hours = (self.age_seconds().max(1) as f32) / 3600.0;
        let access_freq = ((self.access_count as f32) / age_hours).min(1.0);
        let score = 0.4 * self.importance + 0.3 * (1.0 - decay) + 0.3 * access_freq;
        score >= Self::CONSOLIDATION_THRESHOLD
    }

    /// Validate all node constraints.
    ///
    /// # Checks (in order)
    /// 1. Embedding dimension is 1536
    /// 2. Importance is in [0.0, 1.0]
    /// 3. Emotional valence is in [-1.0, 1.0]
    /// 4. Content size is <= 1MB (1,048,576 bytes)
    /// 5. Embedding is normalized (magnitude within +/-0.01 of 1.0)
    ///
    /// # Returns
    /// `Ok(())` if all validations pass, `Err(ValidationError)` on first failure.
    ///
    /// # Constitution Compliance
    /// - AP-009: Validates numeric values to prevent NaN/Infinity propagation
    pub fn validate(&self) -> Result<(), ValidationError> {
        // 1. Check embedding dimension
        if self.embedding.len() != DEFAULT_EMBEDDING_DIM {
            return Err(ValidationError::InvalidEmbeddingDimension {
                expected: DEFAULT_EMBEDDING_DIM,
                actual: self.embedding.len(),
            });
        }

        // 2. Check importance range [0.0, 1.0]
        if self.importance < 0.0 || self.importance > 1.0 || self.importance.is_nan() {
            return Err(ValidationError::OutOfBounds {
                field: "importance".to_string(),
                value: self.importance as f64,
                min: 0.0,
                max: 1.0,
            });
        }

        // 3. Check emotional valence range [-1.0, 1.0]
        if self.emotional_valence < -1.0
            || self.emotional_valence > 1.0
            || self.emotional_valence.is_nan()
        {
            return Err(ValidationError::OutOfBounds {
                field: "emotional_valence".to_string(),
                value: self.emotional_valence as f64,
                min: -1.0,
                max: 1.0,
            });
        }

        // 4. Check content size <= 1MB
        if self.content.len() > MAX_CONTENT_SIZE {
            return Err(ValidationError::ContentTooLarge {
                size: self.content.len(),
                max_size: MAX_CONTENT_SIZE,
            });
        }

        // 5. Check embedding normalization (magnitude ~= 1.0)
        let magnitude: f64 = self
            .embedding
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            .sqrt();

        if (magnitude - 1.0).abs() > Self::NORMALIZATION_TOLERANCE {
            return Err(ValidationError::EmbeddingNotNormalized { magnitude });
        }

        Ok(())
    }
}

impl Default for MemoryNode {
    /// Create a default MemoryNode with empty content and zero-filled embedding.
    ///
    /// NOTE: Default creates a node that will FAIL validation because
    /// zero-filled embedding is not normalized. Use for testing only.
    fn default() -> Self {
        Self::new(String::new(), vec![0.0; DEFAULT_EMBEDDING_DIM])
    }
}

/// Helper function for serde default version value
fn default_version() -> u32 {
    1
}

/// Metadata container for MemoryNode supplementary information.
///
/// # Fields
/// - Source tracking for provenance
/// - Tagging for categorization
/// - Versioning for change tracking
/// - Soft-delete support per SEC-06
/// - Hierarchical relationships (parent/child)
/// - Custom attributes for extensibility
///
/// # Constitution Compliance
/// - SEC-06: Soft delete with 30-day recovery
/// - Naming: snake_case fields
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeMetadata {
    /// Source identifier (e.g., file path, URL, session ID)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// Natural language code (ISO 639-1, e.g., "en", "es")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Content modality type
    pub modality: Modality,

    /// User-defined tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,

    /// Cached UTL learning score [0.0, 1.0]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub utl_score: Option<f32>,

    /// Whether this node has been consolidated
    #[serde(default)]
    pub consolidated: bool,

    /// Timestamp when consolidation occurred
    #[serde(skip_serializing_if = "Option::is_none")]
    pub consolidated_at: Option<DateTime<Utc>>,

    /// Version counter (incremented on updates)
    #[serde(default = "default_version")]
    pub version: u32,

    /// Soft delete flag (SEC-06 compliance)
    #[serde(default)]
    pub deleted: bool,

    /// Timestamp when soft deletion occurred
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deleted_at: Option<DateTime<Utc>>,

    /// Parent node ID for hierarchical relationships
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<Uuid>,

    /// Child node IDs for hierarchical relationships
    #[serde(default)]
    pub child_ids: Vec<Uuid>,

    /// Custom user-defined attributes (JSON-compatible values)
    #[serde(default)]
    pub custom: HashMap<String, serde_json::Value>,

    /// Rationale for storing this memory (required per AP-010)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
}

impl NodeMetadata {
    /// Create new metadata with default values.
    pub fn new() -> Self {
        Self {
            source: None,
            language: None,
            modality: Modality::default(),
            tags: Vec::new(),
            utl_score: None,
            consolidated: false,
            consolidated_at: None,
            version: 1,
            deleted: false,
            deleted_at: None,
            parent_id: None,
            child_ids: Vec::new(),
            custom: HashMap::new(),
            rationale: None,
        }
    }

    /// Set the source identifier. Returns self for builder chaining.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set the language code. Returns self for builder chaining.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set the modality. Returns self for builder chaining.
    pub fn with_modality(mut self, modality: Modality) -> Self {
        self.modality = modality;
        self
    }

    /// Add a tag. Automatically deduplicates (no duplicates stored).
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Remove a tag. Returns true if tag was present and removed.
    pub fn remove_tag(&mut self, tag: &str) -> bool {
        if let Some(pos) = self.tags.iter().position(|t| t == tag) {
            self.tags.remove(pos);
            true
        } else {
            false
        }
    }

    /// Check if a tag exists.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// Set a custom attribute. Overwrites if key already exists.
    pub fn set_custom(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.custom.insert(key.into(), value);
    }

    /// Get a custom attribute by key.
    pub fn get_custom(&self, key: &str) -> Option<&serde_json::Value> {
        self.custom.get(key)
    }

    /// Remove a custom attribute. Returns the removed value if present.
    pub fn remove_custom(&mut self, key: &str) -> Option<serde_json::Value> {
        self.custom.remove(key)
    }

    /// Mark as consolidated with current timestamp.
    pub fn mark_consolidated(&mut self) {
        self.consolidated = true;
        self.consolidated_at = Some(Utc::now());
    }

    /// Mark as deleted (soft delete) with current timestamp.
    /// Per SEC-06: Soft delete with 30-day recovery.
    pub fn mark_deleted(&mut self) {
        self.deleted = true;
        self.deleted_at = Some(Utc::now());
    }

    /// Restore from soft deletion. Clears deleted flag and timestamp.
    pub fn restore(&mut self) {
        self.deleted = false;
        self.deleted_at = None;
    }

    /// Increment version counter. Saturates at u32::MAX (never wraps).
    pub fn increment_version(&mut self) {
        self.version = self.version.saturating_add(1);
    }

    /// Estimate memory size in bytes.
    pub fn estimated_size(&self) -> usize {
        let base = std::mem::size_of::<Self>();

        let source_size = self.source.as_ref().map_or(0, |s| s.len());
        let language_size = self.language.as_ref().map_or(0, |s| s.len());
        let rationale_size = self.rationale.as_ref().map_or(0, |s| s.len());

        let tags_size: usize = self.tags.iter().map(|t| t.len()).sum();
        let child_ids_size = self.child_ids.len() * 16; // UUID is 16 bytes

        // Rough estimate for HashMap: key lengths + value estimate (assume ~32 bytes avg)
        let custom_size: usize = self.custom.keys().map(|k| k.len() + 32).sum();

        base + source_size
            + language_size
            + rationale_size
            + tags_size
            + child_ids_size
            + custom_size
    }
}

impl Default for NodeMetadata {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_node_creation() {
        let embedding = vec![0.1; 1536];
        let node = MemoryNode::new("test content".to_string(), embedding.clone());

        assert_eq!(node.content, "test content");
        assert_eq!(node.embedding.len(), 1536);
        assert_eq!(node.importance, 0.5);
        assert_eq!(node.access_count, 0);
        assert!(!node.metadata.deleted);
    }

    #[test]
    fn test_record_access() {
        let embedding = vec![0.1; 1536];
        let mut node = MemoryNode::new("test".to_string(), embedding);
        let initial_accessed = node.accessed_at;

        std::thread::sleep(std::time::Duration::from_millis(10));
        node.record_access();

        assert_eq!(node.access_count, 1);
        assert!(node.accessed_at > initial_accessed);
    }

    // =========================================================================
    // TC-GHOST-006: Serialization Safety Tests
    // =========================================================================

    #[test]
    fn test_memory_node_json_serialization_round_trip() {
        // TC-GHOST-006: MemoryNode must serialize and deserialize exactly through JSON
        let embedding = vec![0.5; 1536];
        let mut node = MemoryNode::new("Test content for serialization".to_string(), embedding);
        node.importance = 0.85;
        node.access_count = 42;
        node.metadata.source = Some("test-source".to_string());
        node.metadata.language = Some("en".to_string());
        node.metadata.tags = vec!["tag1".to_string(), "tag2".to_string(), "tag3".to_string()];
        node.metadata.utl_score = Some(0.75);
        node.metadata.consolidated = true;
        node.metadata.rationale = Some("Testing serialization round-trip".to_string());

        // Serialize to JSON
        let json_str = serde_json::to_string(&node).expect("MemoryNode must serialize to JSON");

        // Deserialize back
        let restored: MemoryNode =
            serde_json::from_str(&json_str).expect("MemoryNode must deserialize from JSON");

        // Verify exact match using PartialEq
        assert_eq!(
            restored, node,
            "Deserialized node must match original exactly"
        );
    }

    #[test]
    fn test_memory_node_complex_metadata_serialization() {
        // TC-GHOST-006: Complex metadata fields must survive serialization
        let embedding = vec![0.1, 0.2, 0.3]; // Small embedding for test
        let mut node = MemoryNode::new("Complex metadata test".to_string(), embedding);

        // Set all metadata fields
        node.metadata.source = Some("conversation:abc123".to_string());
        node.metadata.language = Some("en-US".to_string());
        node.metadata.tags = vec![
            "important".to_string(),
            "technical".to_string(),
            "machine-learning".to_string(),
            "neural-networks".to_string(),
        ];
        node.metadata.utl_score = Some(0.9876543);
        node.metadata.consolidated = true;
        node.metadata.rationale =
            Some("This is a complex test case with special chars: @#$%^&*()".to_string());

        // Round-trip through JSON
        let json_str = serde_json::to_string(&node).unwrap();
        let restored: MemoryNode = serde_json::from_str(&json_str).unwrap();

        // Verify all metadata fields
        assert_eq!(
            restored.metadata.source,
            Some("conversation:abc123".to_string())
        );
        assert_eq!(restored.metadata.language, Some("en-US".to_string()));
        assert_eq!(restored.metadata.tags.len(), 4);
        assert!(restored
            .metadata
            .tags
            .contains(&"machine-learning".to_string()));
        assert_eq!(restored.metadata.utl_score, Some(0.9876543));
        assert!(restored.metadata.consolidated);
        assert!(restored
            .metadata
            .rationale
            .as_ref()
            .unwrap()
            .contains("special chars"));
    }

    #[test]
    fn test_memory_node_embedding_precision_preserved() {
        // TC-GHOST-006: Embedding float precision must be preserved
        let mut embedding = Vec::with_capacity(1536);
        for i in 0..1536 {
            // Use values that might have precision issues
            let value = (i as f32 / 1536.0) * std::f32::consts::PI;
            embedding.push(value);
        }

        let node = MemoryNode::new("Precision test".to_string(), embedding.clone());

        // Round-trip through JSON
        let json_str = serde_json::to_string(&node).unwrap();
        let restored: MemoryNode = serde_json::from_str(&json_str).unwrap();

        // Verify embedding values are exactly preserved
        assert_eq!(restored.embedding.len(), 1536);
        for (i, (original, restored_val)) in
            embedding.iter().zip(restored.embedding.iter()).enumerate()
        {
            assert_eq!(
                original, restored_val,
                "Embedding value at index {} must be exactly preserved: {} vs {}",
                i, original, restored_val
            );
        }
    }

    #[test]
    fn test_memory_node_timestamps_preserved() {
        // TC-GHOST-006: Timestamps must be preserved through serialization
        let embedding = vec![0.1; 10];
        let node = MemoryNode::new("Timestamp test".to_string(), embedding);
        let original_created_at = node.created_at;
        let original_accessed_at = node.accessed_at;

        // Round-trip through JSON
        let json_str = serde_json::to_string(&node).unwrap();
        let restored: MemoryNode = serde_json::from_str(&json_str).unwrap();

        assert_eq!(
            restored.created_at, original_created_at,
            "created_at must be preserved"
        );
        assert_eq!(
            restored.accessed_at, original_accessed_at,
            "accessed_at must be preserved"
        );
    }

    #[test]
    fn test_memory_node_uuid_preserved() {
        // TC-GHOST-006: UUID must be preserved through serialization
        let embedding = vec![0.1; 10];
        let node = MemoryNode::new("UUID test".to_string(), embedding);
        let original_id = node.id;

        // Round-trip through JSON
        let json_str = serde_json::to_string(&node).unwrap();
        let restored: MemoryNode = serde_json::from_str(&json_str).unwrap();

        assert_eq!(restored.id, original_id, "UUID must be exactly preserved");
    }

    #[test]
    fn test_memory_node_optional_fields_none_serialization() {
        // TC-GHOST-006: Optional None fields must round-trip correctly
        let embedding = vec![0.1; 10];
        let node = MemoryNode::new("Optional fields test".to_string(), embedding);

        // Ensure optional fields are None
        assert!(node.metadata.source.is_none());
        assert!(node.metadata.language.is_none());
        assert!(node.metadata.utl_score.is_none());
        assert!(node.metadata.rationale.is_none());

        // Round-trip
        let json_str = serde_json::to_string(&node).unwrap();
        let restored: MemoryNode = serde_json::from_str(&json_str).unwrap();

        assert!(
            restored.metadata.source.is_none(),
            "None source must remain None"
        );
        assert!(
            restored.metadata.language.is_none(),
            "None language must remain None"
        );
        assert!(
            restored.metadata.utl_score.is_none(),
            "None utl_score must remain None"
        );
        assert!(
            restored.metadata.rationale.is_none(),
            "None rationale must remain None"
        );
    }

    #[test]
    fn test_node_metadata_serialization_isolated() {
        // TC-GHOST-006: NodeMetadata must serialize independently
        let mut metadata = NodeMetadata::default();
        metadata.source = Some("isolated-test".to_string());
        metadata.tags = vec!["a".to_string(), "b".to_string()];
        metadata.utl_score = Some(0.5);

        let json_str = serde_json::to_string(&metadata).unwrap();
        let restored: NodeMetadata = serde_json::from_str(&json_str).unwrap();

        assert_eq!(restored.source, Some("isolated-test".to_string()));
        assert_eq!(restored.tags, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(restored.utl_score, Some(0.5));
    }

    #[test]
    fn test_memory_node_binary_json_equivalence() {
        // TC-GHOST-006: Both pretty and compact JSON must deserialize identically
        let embedding = vec![0.5; 100];
        let mut node = MemoryNode::new("Binary equivalence test".to_string(), embedding);
        node.metadata.tags = vec!["test".to_string()];

        // Compact JSON
        let compact_json = serde_json::to_string(&node).unwrap();
        let from_compact: MemoryNode = serde_json::from_str(&compact_json).unwrap();

        // Pretty JSON
        let pretty_json = serde_json::to_string_pretty(&node).unwrap();
        let from_pretty: MemoryNode = serde_json::from_str(&pretty_json).unwrap();

        // Both must produce identical results
        assert_eq!(
            from_compact, from_pretty,
            "Compact and pretty JSON must deserialize identically"
        );
        assert_eq!(from_compact, node, "Both must match original");
    }

    #[test]
    fn test_memory_node_special_content_serialization() {
        // TC-GHOST-006: Special characters in content must be preserved
        let special_content = r#"Content with "quotes", 'apostrophes', \backslashes\, and
newlines, plus unicode: 日本語 🎉 émojis"#;

        let embedding = vec![0.1; 10];
        let node = MemoryNode::new(special_content.to_string(), embedding);

        let json_str = serde_json::to_string(&node).unwrap();
        let restored: MemoryNode = serde_json::from_str(&json_str).unwrap();

        assert_eq!(
            restored.content, special_content,
            "Special characters must be preserved"
        );
    }

    #[test]
    fn test_memory_node_extreme_values() {
        // TC-GHOST-006: Extreme float values must be handled
        let mut embedding = vec![0.0; 10];
        embedding[0] = f32::MIN_POSITIVE;
        embedding[1] = f32::MAX;
        embedding[2] = f32::MIN;
        embedding[3] = 1e-38;
        embedding[4] = 1e38;

        let mut node = MemoryNode::new("Extreme values test".to_string(), embedding.clone());
        node.importance = 0.0;
        node.access_count = u64::MAX;

        let json_str = serde_json::to_string(&node).unwrap();
        let restored: MemoryNode = serde_json::from_str(&json_str).unwrap();

        assert_eq!(restored.importance, 0.0);
        assert_eq!(restored.access_count, u64::MAX);
        assert_eq!(restored.embedding[0], f32::MIN_POSITIVE);
    }

    // =========================================================================
    // TASK-M02-003: NodeMetadata Tests
    // =========================================================================

    #[test]
    fn test_node_metadata_new_defaults() {
        let meta = NodeMetadata::new();

        assert!(meta.source.is_none());
        assert!(meta.language.is_none());
        assert_eq!(meta.modality, Modality::Text);
        assert!(meta.tags.is_empty());
        assert!(meta.utl_score.is_none());
        assert!(!meta.consolidated);
        assert!(meta.consolidated_at.is_none());
        assert_eq!(meta.version, 1);
        assert!(!meta.deleted);
        assert!(meta.deleted_at.is_none());
        assert!(meta.parent_id.is_none());
        assert!(meta.child_ids.is_empty());
        assert!(meta.custom.is_empty());
        assert!(meta.rationale.is_none());
    }

    #[test]
    fn test_node_metadata_default_equals_new() {
        let from_new = NodeMetadata::new();
        let from_default = NodeMetadata::default();
        assert_eq!(from_new, from_default);
    }

    #[test]
    fn test_node_metadata_builder_with_source() {
        let meta = NodeMetadata::new().with_source("test-source");
        assert_eq!(meta.source, Some("test-source".to_string()));
    }

    #[test]
    fn test_node_metadata_builder_with_language() {
        let meta = NodeMetadata::new().with_language("en-US");
        assert_eq!(meta.language, Some("en-US".to_string()));
    }

    #[test]
    fn test_node_metadata_builder_with_modality() {
        let meta = NodeMetadata::new().with_modality(Modality::Code);
        assert_eq!(meta.modality, Modality::Code);
    }

    #[test]
    fn test_node_metadata_builder_chaining() {
        let meta = NodeMetadata::new()
            .with_source("file.rs")
            .with_language("rust")
            .with_modality(Modality::Code);

        assert_eq!(meta.source, Some("file.rs".to_string()));
        assert_eq!(meta.language, Some("rust".to_string()));
        assert_eq!(meta.modality, Modality::Code);
    }

    #[test]
    fn test_node_metadata_add_tag_single() {
        let mut meta = NodeMetadata::new();
        meta.add_tag("important");
        assert_eq!(meta.tags, vec!["important"]);
    }

    #[test]
    fn test_node_metadata_add_tag_deduplication() {
        let mut meta = NodeMetadata::new();
        meta.add_tag("important");
        meta.add_tag("important");
        meta.add_tag("important");
        assert_eq!(meta.tags.len(), 1);
        assert_eq!(meta.tags, vec!["important"]);
    }

    #[test]
    fn test_node_metadata_add_tag_multiple_unique() {
        let mut meta = NodeMetadata::new();
        meta.add_tag("alpha");
        meta.add_tag("beta");
        meta.add_tag("gamma");
        assert_eq!(meta.tags.len(), 3);
        assert!(meta.tags.contains(&"alpha".to_string()));
        assert!(meta.tags.contains(&"beta".to_string()));
        assert!(meta.tags.contains(&"gamma".to_string()));
    }

    #[test]
    fn test_node_metadata_remove_tag_exists() {
        let mut meta = NodeMetadata::new();
        meta.add_tag("test");
        assert!(meta.remove_tag("test"));
        assert!(meta.tags.is_empty());
    }

    #[test]
    fn test_node_metadata_remove_tag_not_exists() {
        let mut meta = NodeMetadata::new();
        assert!(!meta.remove_tag("nonexistent"));
    }

    #[test]
    fn test_node_metadata_has_tag() {
        let mut meta = NodeMetadata::new();
        meta.add_tag("exists");
        assert!(meta.has_tag("exists"));
        assert!(!meta.has_tag("missing"));
    }

    #[test]
    fn test_node_metadata_custom_attributes() {
        use serde_json::json;

        let mut meta = NodeMetadata::new();
        meta.set_custom("priority", json!(5));
        meta.set_custom("reviewed", json!(true));
        meta.set_custom("tags", json!(["a", "b"]));

        assert_eq!(meta.get_custom("priority"), Some(&json!(5)));
        assert_eq!(meta.get_custom("reviewed"), Some(&json!(true)));
        assert_eq!(meta.get_custom("tags"), Some(&json!(["a", "b"])));
        assert_eq!(meta.get_custom("missing"), None);
    }

    #[test]
    fn test_node_metadata_custom_overwrite() {
        use serde_json::json;

        let mut meta = NodeMetadata::new();
        meta.set_custom("key", json!(1));
        meta.set_custom("key", json!(2));
        assert_eq!(meta.get_custom("key"), Some(&json!(2)));
    }

    #[test]
    fn test_node_metadata_custom_remove() {
        use serde_json::json;

        let mut meta = NodeMetadata::new();
        meta.set_custom("temp", json!("value"));

        let removed = meta.remove_custom("temp");
        assert_eq!(removed, Some(json!("value")));
        assert_eq!(meta.get_custom("temp"), None);

        // Removing again returns None
        assert_eq!(meta.remove_custom("temp"), None);
    }

    #[test]
    fn test_node_metadata_mark_consolidated() {
        let mut meta = NodeMetadata::new();
        assert!(!meta.consolidated);
        assert!(meta.consolidated_at.is_none());

        meta.mark_consolidated();

        assert!(meta.consolidated);
        assert!(meta.consolidated_at.is_some());

        // Timestamp should be recent (within last second)
        let timestamp = meta.consolidated_at.unwrap();
        let now = Utc::now();
        let diff = now.signed_duration_since(timestamp);
        assert!(
            diff.num_seconds() < 1,
            "Consolidated timestamp should be recent"
        );
    }

    #[test]
    fn test_node_metadata_mark_deleted() {
        let mut meta = NodeMetadata::new();
        assert!(!meta.deleted);
        assert!(meta.deleted_at.is_none());

        meta.mark_deleted();

        assert!(meta.deleted);
        assert!(meta.deleted_at.is_some());

        // Timestamp should be recent
        let timestamp = meta.deleted_at.unwrap();
        let now = Utc::now();
        let diff = now.signed_duration_since(timestamp);
        assert!(diff.num_seconds() < 1, "Deleted timestamp should be recent");
    }

    #[test]
    fn test_node_metadata_restore() {
        let mut meta = NodeMetadata::new();
        meta.mark_deleted();
        assert!(meta.deleted);
        assert!(meta.deleted_at.is_some());

        meta.restore();

        assert!(!meta.deleted);
        assert!(meta.deleted_at.is_none());
    }

    #[test]
    fn test_node_metadata_soft_delete_restore_cycle() {
        let mut meta = NodeMetadata::new();

        // Initial state
        assert!(!meta.deleted);

        // Delete
        meta.mark_deleted();
        assert!(meta.deleted);
        assert!(meta.deleted_at.is_some());

        // Restore
        meta.restore();
        assert!(!meta.deleted);
        assert!(meta.deleted_at.is_none());

        // Delete again
        meta.mark_deleted();
        assert!(meta.deleted);
        assert!(meta.deleted_at.is_some());
    }

    #[test]
    fn test_node_metadata_version_increment() {
        let mut meta = NodeMetadata::new();
        assert_eq!(meta.version, 1);

        meta.increment_version();
        assert_eq!(meta.version, 2);

        meta.increment_version();
        assert_eq!(meta.version, 3);
    }

    #[test]
    fn test_node_metadata_version_saturates() {
        let mut meta = NodeMetadata::new();
        meta.version = u32::MAX;

        meta.increment_version();

        // Should NOT wrap to 0, should stay at MAX
        assert_eq!(meta.version, u32::MAX);
    }

    #[test]
    fn test_node_metadata_estimated_size_basic() {
        let meta = NodeMetadata::new();
        let size = meta.estimated_size();

        // Should be at least the base struct size
        assert!(size >= std::mem::size_of::<NodeMetadata>());
    }

    #[test]
    fn test_node_metadata_estimated_size_with_data() {
        use serde_json::json;

        let mut meta = NodeMetadata::new();
        meta.source = Some("very long source string that takes up space".to_string());
        meta.language = Some("en-US".to_string());
        meta.add_tag("tag1");
        meta.add_tag("tag2");
        meta.add_tag("tag3");
        meta.child_ids.push(Uuid::new_v4());
        meta.child_ids.push(Uuid::new_v4());
        meta.set_custom("key1", json!("value1"));
        meta.set_custom("key2", json!(12345));

        let size_with_data = meta.estimated_size();
        let empty_size = NodeMetadata::new().estimated_size();

        // Size with data should be larger
        assert!(
            size_with_data > empty_size,
            "Size with data {} should be > empty size {}",
            size_with_data,
            empty_size
        );
    }

    #[test]
    fn test_node_metadata_serde_roundtrip() {
        use serde_json::json;

        let mut meta = NodeMetadata::new();
        meta.source = Some("test-source".to_string());
        meta.language = Some("en".to_string());
        meta.modality = Modality::Code;
        meta.add_tag("test");
        meta.utl_score = Some(0.75);
        meta.mark_consolidated();
        meta.version = 5;
        meta.parent_id = Some(Uuid::new_v4());
        meta.child_ids.push(Uuid::new_v4());
        meta.set_custom("key", json!("value"));
        meta.rationale = Some("test rationale".to_string());

        let json_str = serde_json::to_string(&meta).expect("serialize failed");
        let restored: NodeMetadata = serde_json::from_str(&json_str).expect("deserialize failed");

        assert_eq!(
            meta, restored,
            "Round-trip serialization must preserve all fields"
        );
    }

    #[test]
    fn test_node_metadata_serde_with_deleted() {
        let mut meta = NodeMetadata::new();
        meta.mark_deleted();

        let json_str = serde_json::to_string(&meta).expect("serialize failed");
        let restored: NodeMetadata = serde_json::from_str(&json_str).expect("deserialize failed");

        assert!(restored.deleted);
        assert!(restored.deleted_at.is_some());
        assert_eq!(meta.deleted_at, restored.deleted_at);
    }

    #[test]
    fn test_node_metadata_hierarchical_relationships() {
        let parent_id = Uuid::new_v4();
        let child1 = Uuid::new_v4();
        let child2 = Uuid::new_v4();

        let mut meta = NodeMetadata::new();
        meta.parent_id = Some(parent_id);
        meta.child_ids.push(child1);
        meta.child_ids.push(child2);

        assert_eq!(meta.parent_id, Some(parent_id));
        assert_eq!(meta.child_ids.len(), 2);
        assert!(meta.child_ids.contains(&child1));
        assert!(meta.child_ids.contains(&child2));
    }

    #[test]
    fn test_node_metadata_clone() {
        use serde_json::json;

        let mut original = NodeMetadata::new();
        original.source = Some("source".to_string());
        original.add_tag("tag");
        original.set_custom("key", json!(1));
        original.mark_consolidated();

        let cloned = original.clone();

        assert_eq!(original, cloned);

        // Verify deep clone (mutating clone doesn't affect original)
        let mut cloned_mut = original.clone();
        cloned_mut.add_tag("new_tag");
        assert_ne!(original.tags.len(), cloned_mut.tags.len());
    }

    // =========================================================================
    // TASK-M02-004: ValidationError Tests
    // =========================================================================

    #[test]
    fn test_validation_error_invalid_embedding_dimension() {
        let error = ValidationError::InvalidEmbeddingDimension {
            expected: 1536,
            actual: 768,
        };
        let msg = error.to_string();

        assert!(
            msg.contains("expected 1536"),
            "Should show expected dimension"
        );
        assert!(msg.contains("got 768"), "Should show actual dimension");
        assert!(
            msg.contains("Invalid embedding dimension"),
            "Should have correct prefix"
        );
    }

    #[test]
    fn test_validation_error_out_of_bounds() {
        let error = ValidationError::OutOfBounds {
            field: "importance".to_string(),
            value: 1.5,
            min: 0.0,
            max: 1.0,
        };
        let msg = error.to_string();

        assert!(msg.contains("importance"), "Should show field name");
        assert!(msg.contains("1.5"), "Should show invalid value");
        assert!(msg.contains("[0, 1]"), "Should show valid range");
    }

    #[test]
    fn test_validation_error_content_too_large() {
        let error = ValidationError::ContentTooLarge {
            size: 2_000_000,
            max_size: 1_048_576,
        };
        let msg = error.to_string();

        assert!(msg.contains("2000000"), "Should show actual size");
        assert!(msg.contains("1048576"), "Should show max size");
        assert!(msg.contains("exceeds maximum"), "Should indicate overflow");
    }

    #[test]
    fn test_validation_error_embedding_not_normalized() {
        let error = ValidationError::EmbeddingNotNormalized { magnitude: 0.85 };
        let msg = error.to_string();

        assert!(
            msg.contains("0.850000"),
            "Should show magnitude with precision"
        );
        assert!(
            msg.contains("not normalized"),
            "Should indicate normalization issue"
        );
        assert!(msg.contains("expected ~1.0"), "Should show expected value");
    }

    #[test]
    fn test_validation_error_implements_std_error() {
        // Verify thiserror properly implements std::error::Error
        let error: Box<dyn std::error::Error> =
            Box::new(ValidationError::InvalidEmbeddingDimension {
                expected: 1536,
                actual: 0,
            });

        // std::error::Error requires Display, which we get from thiserror
        let _ = error.to_string();
    }

    #[test]
    fn test_validation_error_clone() {
        let original = ValidationError::OutOfBounds {
            field: "test".to_string(),
            value: -0.5,
            min: 0.0,
            max: 1.0,
        };
        let cloned = original.clone();

        assert_eq!(original, cloned, "Clone must produce equal value");
    }

    #[test]
    fn test_validation_error_partial_eq() {
        let a = ValidationError::ContentTooLarge {
            size: 100,
            max_size: 50,
        };
        let b = ValidationError::ContentTooLarge {
            size: 100,
            max_size: 50,
        };
        let c = ValidationError::ContentTooLarge {
            size: 101,
            max_size: 50,
        };

        assert_eq!(a, b, "Same values should be equal");
        assert_ne!(a, c, "Different values should not be equal");
    }

    #[test]
    fn test_validation_error_debug_format() {
        let error = ValidationError::InvalidEmbeddingDimension {
            expected: 1536,
            actual: 512,
        };
        let debug_str = format!("{:?}", error);

        assert!(
            debug_str.contains("InvalidEmbeddingDimension"),
            "Debug should show variant"
        );
        assert!(debug_str.contains("1536"), "Debug should show expected");
        assert!(debug_str.contains("512"), "Debug should show actual");
    }

    #[test]
    fn test_validation_error_out_of_bounds_negative_range() {
        // Test for valence which has range [-1.0, 1.0]
        let error = ValidationError::OutOfBounds {
            field: "emotional_valence".to_string(),
            value: -1.5,
            min: -1.0,
            max: 1.0,
        };
        let msg = error.to_string();

        assert!(msg.contains("-1.5"), "Should handle negative values");
        assert!(
            msg.contains("[-1, 1]"),
            "Should show negative range correctly"
        );
    }

    #[test]
    fn test_validation_error_embedding_edge_magnitudes() {
        // Test edge cases for magnitude
        let too_small = ValidationError::EmbeddingNotNormalized { magnitude: 0.0 };
        let too_large = ValidationError::EmbeddingNotNormalized { magnitude: 100.0 };

        assert!(too_small.to_string().contains("0.000000"));
        assert!(too_large.to_string().contains("100.000000"));
    }

    // =========================================================================
    // TASK-M02-005: MemoryNode Struct Tests
    // =========================================================================

    #[test]
    fn test_default_embedding_dim_constant() {
        assert_eq!(DEFAULT_EMBEDDING_DIM, 1536);
    }

    #[test]
    fn test_max_content_size_constant() {
        assert_eq!(MAX_CONTENT_SIZE, 1_048_576);
        assert_eq!(MAX_CONTENT_SIZE, 1024 * 1024); // 1MB
    }

    #[test]
    fn test_memory_node_has_all_required_fields() {
        let embedding = vec![0.1; DEFAULT_EMBEDDING_DIM];
        let node = MemoryNode::new("test".to_string(), embedding);

        // Verify all 10 fields exist and are accessible
        let _id: NodeId = node.id;
        let _content: &String = &node.content;
        let _embedding: &EmbeddingVector = &node.embedding;
        let _quadrant: JohariQuadrant = node.quadrant;
        let _importance: f32 = node.importance;
        let _valence: f32 = node.emotional_valence;
        let _created: DateTime<Utc> = node.created_at;
        let _accessed: DateTime<Utc> = node.accessed_at;
        let _count: u64 = node.access_count;
        let _meta: &NodeMetadata = &node.metadata;
    }

    #[test]
    fn test_memory_node_new_defaults() {
        let embedding = vec![0.0; 1536];
        let node = MemoryNode::new("content".to_string(), embedding);

        assert_eq!(node.content, "content");
        assert_eq!(node.embedding.len(), 1536);
        assert_eq!(node.quadrant, JohariQuadrant::Open);
        assert_eq!(node.importance, 0.5);
        assert_eq!(node.emotional_valence, 0.0);
        assert_eq!(node.access_count, 0);
        assert!(!node.metadata.deleted);
    }

    #[test]
    fn test_memory_node_emotional_valence_range() {
        let embedding = vec![0.0; 10];
        let mut node = MemoryNode::new("test".to_string(), embedding);

        node.emotional_valence = -1.0;
        assert_eq!(node.emotional_valence, -1.0);

        node.emotional_valence = 0.0;
        assert_eq!(node.emotional_valence, 0.0);

        node.emotional_valence = 1.0;
        assert_eq!(node.emotional_valence, 1.0);
    }

    #[test]
    fn test_memory_node_serde_with_emotional_valence() {
        let embedding = vec![0.5; 100];
        let mut node = MemoryNode::new("serde test".to_string(), embedding);
        node.emotional_valence = -0.75;
        node.importance = 0.9;

        let json = serde_json::to_string(&node).expect("serialize");
        let restored: MemoryNode = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.emotional_valence, -0.75);
        assert_eq!(restored.importance, 0.9);
        assert_eq!(restored.content, "serde test");
    }

    #[test]
    fn test_memory_node_record_access_updates_timestamp() {
        let embedding = vec![0.1; 10];
        let mut node = MemoryNode::new("test".to_string(), embedding);
        let initial = node.accessed_at;

        std::thread::sleep(std::time::Duration::from_millis(10));
        node.record_access();

        assert_eq!(node.access_count, 1);
        assert!(node.accessed_at > initial);
    }

    #[test]
    fn test_memory_node_deleted_via_metadata() {
        let embedding = vec![0.1; 10];
        let mut node = MemoryNode::new("test".to_string(), embedding);

        assert!(!node.metadata.deleted);
        node.metadata.mark_deleted();
        assert!(node.metadata.deleted);
    }

    #[test]
    fn test_memory_node_quadrant_field() {
        let embedding = vec![0.1; 10];
        let node = MemoryNode::new("test".to_string(), embedding);
        assert_eq!(node.quadrant, JohariQuadrant::Open);
    }

    #[test]
    fn test_memory_node_accessed_at_field() {
        let embedding = vec![0.1; 10];
        let node = MemoryNode::new("test".to_string(), embedding);
        let _timestamp: DateTime<Utc> = node.accessed_at;
    }

    #[test]
    fn test_memory_node_modality_via_metadata() {
        let embedding = vec![0.1; 10];
        let mut node = MemoryNode::new("test".to_string(), embedding);
        node.metadata.modality = Modality::Code;
        assert_eq!(node.metadata.modality, Modality::Code);
    }

    // =========================================================================
    // TASK-M02-006: MemoryNode Methods Tests
    // =========================================================================

    #[test]
    fn test_with_id_creates_node_with_specific_id() {
        let specific_id = Uuid::new_v4();
        let embedding = vec![0.0; DEFAULT_EMBEDDING_DIM];
        let node = MemoryNode::with_id(specific_id, "test".to_string(), embedding);
        assert_eq!(node.id, specific_id);
    }

    #[test]
    fn test_with_id_preserves_other_defaults() {
        let id = Uuid::new_v4();
        let node = MemoryNode::with_id(id, "content".to_string(), vec![0.1; 1536]);
        assert_eq!(node.importance, 0.5);
        assert_eq!(node.emotional_valence, 0.0);
        assert_eq!(node.access_count, 0);
        assert_eq!(node.quadrant, JohariQuadrant::Open);
    }

    #[test]
    fn test_record_access_uses_saturating_add() {
        let mut node = MemoryNode::new("test".to_string(), vec![0.0; 1536]);
        node.access_count = u64::MAX;
        node.record_access();
        assert_eq!(node.access_count, u64::MAX); // Should NOT wrap to 0
    }

    #[test]
    fn test_age_seconds_positive() {
        let node = MemoryNode::new("test".to_string(), vec![0.0; 1536]);
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(node.age_seconds() >= 0);
    }

    #[test]
    fn test_time_since_access_seconds() {
        let node = MemoryNode::new("test".to_string(), vec![0.0; 1536]);
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(node.time_since_access_seconds() >= 0);
    }

    #[test]
    fn test_compute_decay_recent_access() {
        let node = MemoryNode::new("test".to_string(), vec![0.0; 1536]);
        let decay = node.compute_decay();
        // Just created, decay should be very close to 1.0
        assert!(
            decay >= 0.99,
            "Recent node decay should be ~1.0, got {}",
            decay
        );
    }

    #[test]
    fn test_compute_decay_in_valid_range() {
        let node = MemoryNode::new("test".to_string(), vec![0.0; 1536]);
        let decay = node.compute_decay();
        assert!(
            decay >= 0.0 && decay <= 1.0,
            "Decay {} must be in [0,1]",
            decay
        );
    }

    #[test]
    fn test_compute_decay_handles_high_importance() {
        let mut node = MemoryNode::new("test".to_string(), vec![0.0; 1536]);
        node.importance = 1.0;
        let decay = node.compute_decay();
        assert!(decay >= 0.0 && decay <= 1.0);
    }

    #[test]
    fn test_should_consolidate_high_importance() {
        let mut node = MemoryNode::new("test".to_string(), vec![0.0; 1536]);
        node.importance = 1.0; // High importance should push toward consolidation
                               // Even with just importance=1.0, score = 0.4*1.0 + 0.3*~0 + 0.3*0 = 0.4
                               // Not enough alone, but reasonable behavior
        let _should = node.should_consolidate(); // Just verify it doesn't panic
    }

    #[test]
    fn test_should_consolidate_returns_bool() {
        let node = MemoryNode::new("test".to_string(), vec![0.0; 1536]);
        let result: bool = node.should_consolidate();
        let _ = result; // Type check
    }

    #[test]
    fn test_validate_valid_node() {
        // Create a normalized embedding
        let dim = DEFAULT_EMBEDDING_DIM;
        let val = 1.0 / (dim as f32).sqrt();
        let embedding: Vec<f32> = vec![val; dim];

        let node = MemoryNode::new("valid content".to_string(), embedding);
        assert!(node.validate().is_ok());
    }

    #[test]
    fn test_validate_wrong_embedding_dim() {
        let node = MemoryNode::new("test".to_string(), vec![0.0; 100]); // Wrong dimension
        let result = node.validate();
        assert!(matches!(
            result,
            Err(ValidationError::InvalidEmbeddingDimension { .. })
        ));
    }

    #[test]
    fn test_validate_importance_too_low() {
        let dim = DEFAULT_EMBEDDING_DIM;
        let val = 1.0 / (dim as f32).sqrt();
        let mut node = MemoryNode::new("test".to_string(), vec![val; dim]);
        node.importance = -0.1;
        let result = node.validate();
        assert!(
            matches!(result, Err(ValidationError::OutOfBounds { field, .. }) if field == "importance")
        );
    }

    #[test]
    fn test_validate_importance_too_high() {
        let dim = DEFAULT_EMBEDDING_DIM;
        let val = 1.0 / (dim as f32).sqrt();
        let mut node = MemoryNode::new("test".to_string(), vec![val; dim]);
        node.importance = 1.1;
        let result = node.validate();
        assert!(
            matches!(result, Err(ValidationError::OutOfBounds { field, .. }) if field == "importance")
        );
    }

    #[test]
    fn test_validate_valence_too_low() {
        let dim = DEFAULT_EMBEDDING_DIM;
        let val = 1.0 / (dim as f32).sqrt();
        let mut node = MemoryNode::new("test".to_string(), vec![val; dim]);
        node.emotional_valence = -1.5;
        let result = node.validate();
        assert!(
            matches!(result, Err(ValidationError::OutOfBounds { field, .. }) if field == "emotional_valence")
        );
    }

    #[test]
    fn test_validate_valence_too_high() {
        let dim = DEFAULT_EMBEDDING_DIM;
        let val = 1.0 / (dim as f32).sqrt();
        let mut node = MemoryNode::new("test".to_string(), vec![val; dim]);
        node.emotional_valence = 1.5;
        let result = node.validate();
        assert!(
            matches!(result, Err(ValidationError::OutOfBounds { field, .. }) if field == "emotional_valence")
        );
    }

    #[test]
    fn test_validate_content_too_large() {
        let dim = DEFAULT_EMBEDDING_DIM;
        let val = 1.0 / (dim as f32).sqrt();
        let big_content = "x".repeat(MAX_CONTENT_SIZE + 1);
        let node = MemoryNode::new(big_content, vec![val; dim]);
        let result = node.validate();
        assert!(matches!(
            result,
            Err(ValidationError::ContentTooLarge { .. })
        ));
    }

    #[test]
    fn test_validate_embedding_not_normalized() {
        let node = MemoryNode::new("test".to_string(), vec![0.5; 1536]); // Not normalized
        let result = node.validate();
        assert!(matches!(
            result,
            Err(ValidationError::EmbeddingNotNormalized { .. })
        ));
    }

    #[test]
    fn test_validate_zero_embedding_fails() {
        let node = MemoryNode::new("test".to_string(), vec![0.0; 1536]); // Magnitude = 0
        let result = node.validate();
        assert!(matches!(
            result,
            Err(ValidationError::EmbeddingNotNormalized { .. })
        ));
    }

    #[test]
    fn test_default_creates_node() {
        let node = MemoryNode::default();
        assert!(node.content.is_empty());
        assert_eq!(node.embedding.len(), DEFAULT_EMBEDDING_DIM);
        assert_eq!(node.importance, 0.5);
    }

    #[test]
    fn test_default_embedding_fails_validation() {
        // Default creates zero-filled embedding which is NOT normalized
        let node = MemoryNode::default();
        assert!(node.validate().is_err());
    }

    #[test]
    fn test_validate_boundary_importance_zero() {
        let dim = DEFAULT_EMBEDDING_DIM;
        let val = 1.0 / (dim as f32).sqrt();
        let mut node = MemoryNode::new("test".to_string(), vec![val; dim]);
        node.importance = 0.0; // Boundary
        assert!(node.validate().is_ok());
    }

    #[test]
    fn test_validate_boundary_importance_one() {
        let dim = DEFAULT_EMBEDDING_DIM;
        let val = 1.0 / (dim as f32).sqrt();
        let mut node = MemoryNode::new("test".to_string(), vec![val; dim]);
        node.importance = 1.0; // Boundary
        assert!(node.validate().is_ok());
    }

    #[test]
    fn test_validate_boundary_valence_negative_one() {
        let dim = DEFAULT_EMBEDDING_DIM;
        let val = 1.0 / (dim as f32).sqrt();
        let mut node = MemoryNode::new("test".to_string(), vec![val; dim]);
        node.emotional_valence = -1.0; // Boundary
        assert!(node.validate().is_ok());
    }

    #[test]
    fn test_validate_boundary_valence_positive_one() {
        let dim = DEFAULT_EMBEDDING_DIM;
        let val = 1.0 / (dim as f32).sqrt();
        let mut node = MemoryNode::new("test".to_string(), vec![val; dim]);
        node.emotional_valence = 1.0; // Boundary
        assert!(node.validate().is_ok());
    }

    #[test]
    fn test_validate_content_exactly_max_size() {
        let dim = DEFAULT_EMBEDDING_DIM;
        let val = 1.0 / (dim as f32).sqrt();
        let max_content = "x".repeat(MAX_CONTENT_SIZE); // Exactly at limit
        let node = MemoryNode::new(max_content, vec![val; dim]);
        assert!(node.validate().is_ok());
    }
}
