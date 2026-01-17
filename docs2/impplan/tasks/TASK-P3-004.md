# TASK-P3-004: DistanceCalculator

```xml
<task_spec id="TASK-P3-004" version="2.0" audited="2026-01-16">
<metadata>
  <title>DistanceCalculator Implementation</title>
  <status>complete</status>
  <layer>logic</layer>
  <sequence>23</sequence>
  <phase>3</phase>
  <implements>
    <requirement_ref>REQ-P3-01</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="complete">TASK-P2-002</task_ref>
    <task_ref status="complete">TASK-P2-006</task_ref>
    <task_ref status="complete">TASK-P3-001</task_ref>
    <task_ref status="complete">TASK-P3-002</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
</metadata>

<context>
Implements distance/similarity metrics for all 13 embedding spaces.

CRITICAL: Most similarity functions ALREADY EXIST on vector types:
- DenseVector has cosine_similarity(), euclidean_distance() (in embeddings/vector.rs)
- SparseVector has jaccard_index() (in types/fingerprint/sparse.rs)
- BinaryVector has hamming_distance() (in embeddings/vector.rs)

This task creates a unified dispatch layer that:
1. Delegates to existing vector methods
2. Adds max_sim for E12 (ColBERT late interaction)
3. Adds transe_similarity for E11 (entity embeddings)
4. Provides compute_similarity_for_space() dispatcher
5. Ensures all outputs normalized to [0.0, 1.0]

NOTE: Temporal spaces (E2-E4) use the same distance metrics as other spaces.
Category-aware weighting is handled at the MultiSpaceSimilarity layer (TASK-P3-005),
not in this module. DistanceCalculator computes raw similarity only.
</context>

<codebase_state>
## EXISTING TYPES (VERIFIED 2026-01-16)

### Embedder Enum
Location: crates/context-graph-core/src/teleological/embedder.rs
```rust
pub enum Embedder {
    Semantic = 0,        // E1
    TemporalRecent = 1,  // E2
    TemporalPeriodic = 2,// E3
    TemporalPositional = 3,// E4
    Causal = 4,          // E5
    Sparse = 5,          // E6
    Code = 6,            // E7
    Emotional = 7,       // E8 (NOT "Graph")
    Hdc = 8,             // E9
    Multimodal = 9,      // E10
    Entity = 10,         // E11
    LateInteraction = 11,// E12
    KeywordSplade = 12,  // E13
}
```

### SemanticFingerprint (TeleologicalArray)
Location: crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs
```rust
pub type TeleologicalArray = SemanticFingerprint;

pub struct SemanticFingerprint {
    pub e1_semantic: Vec<f32>,              // 1024D
    pub e2_temporal_recent: Vec<f32>,       // 512D
    pub e3_temporal_periodic: Vec<f32>,     // 512D
    pub e4_temporal_positional: Vec<f32>,   // 512D
    pub e5_causal: Vec<f32>,                // 768D
    pub e6_sparse: SparseVector,            // Sparse
    pub e7_code: Vec<f32>,                  // 1536D
    pub e8_graph: Vec<f32>,                 // 384D (field name is e8_graph, but Embedder::Emotional)
    pub e9_hdc: Vec<f32>,                   // 1024D (projected)
    pub e10_multimodal: Vec<f32>,           // 768D
    pub e11_entity: Vec<f32>,               // 384D
    pub e12_late_interaction: Vec<Vec<f32>>,// 128D per token
    pub e13_splade: SparseVector,           // Sparse
}
```
Access method: `fp.get(embedder: Embedder) -> EmbeddingRef<'_>`

### DenseVector
Location: crates/context-graph-core/src/embeddings/vector.rs
```rust
pub struct DenseVector { data: Vec<f32> }

impl DenseVector {
    pub fn cosine_similarity(&self, other: &Self) -> f32;  // ALREADY EXISTS
    pub fn euclidean_distance(&self, other: &Self) -> f32; // ALREADY EXISTS
    pub fn dot_product(&self, other: &Self) -> f32;        // ALREADY EXISTS
    pub fn magnitude(&self) -> f32;                         // ALREADY EXISTS
    pub fn normalized(&self) -> Self;                       // ALREADY EXISTS
}
```

### SparseVector
Location: crates/context-graph-core/src/types/fingerprint/sparse.rs
```rust
pub struct SparseVector {
    indices: Vec<u32>,  // NOTE: u32, not usize
    values: Vec<f32>,
    vocab_size: usize,
}

impl SparseVector {
    pub fn new(indices: Vec<u32>, values: Vec<f32>, vocab_size: usize) -> Result<Self, SparseVectorError>;
    pub fn jaccard_index(&self, other: &Self) -> f32;  // ALREADY EXISTS
    pub fn sparse_dot(&self, other: &Self) -> f32;     // ALREADY EXISTS
    pub fn indices(&self) -> &[u32];
    pub fn values(&self) -> &[f32];
    pub fn nnz(&self) -> usize;
    pub fn empty() -> Self;
}
```

### BinaryVector
Location: crates/context-graph-core/src/embeddings/vector.rs
```rust
pub struct BinaryVector {
    data: Vec<u64>,
    bit_len: usize,
}

impl BinaryVector {
    pub fn hamming_distance(&self, other: &Self) -> u64;  // ALREADY EXISTS
    pub fn bit_len(&self) -> usize;
    pub fn zeros(bit_len: usize) -> Self;
    pub fn set_bit(&mut self, idx: usize, value: bool);
    pub fn get_bit(&self, idx: usize) -> bool;
}
```

### DistanceMetric
Location: crates/context-graph-core/src/index/config.rs
```rust
pub enum DistanceMetric {
    Cosine,
    DotProduct,
    Euclidean,
    AsymmetricCosine,
    MaxSim,
    Jaccard,
}
```

### EmbeddingRef
Location: crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs
```rust
pub enum EmbeddingRef<'a> {
    Dense(&'a [f32]),
    Sparse(&'a SparseVector),
    TokenLevel(&'a [Vec<f32>]),
}
```

### Module Exports (CRITICAL)
```rust
// From crates/context-graph-core/src/lib.rs or relevant mod.rs:
pub use teleological::Embedder;
pub use types::fingerprint::{SemanticFingerprint, TeleologicalArray, SparseVector, EmbeddingRef};
pub use embeddings::{DenseVector, BinaryVector, VectorError};
pub use index::config::DistanceMetric;
```
</codebase_state>

<input_context_files>
  <file purpose="component_spec" exists="true">docs2/impplan/technical/TECH-PHASE3-SIMILARITY-DIVERGENCE.md</file>
  <file purpose="dense_vector" exists="true">crates/context-graph-core/src/embeddings/vector.rs</file>
  <file purpose="sparse_vector" exists="true">crates/context-graph-core/src/types/fingerprint/sparse.rs</file>
  <file purpose="embedder_enum" exists="true">crates/context-graph-core/src/teleological/embedder.rs</file>
  <file purpose="semantic_fingerprint" exists="true">crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs</file>
  <file purpose="distance_metric" exists="true">crates/context-graph-core/src/index/config.rs</file>
</input_context_files>

<prerequisites>
  <check verified="true">TASK-P2-002 complete - DenseVector, BinaryVector exist with similarity methods</check>
  <check verified="true">SparseVector exists with jaccard_index() method</check>
  <check verified="true">Embedder enum exists with all 13 variants</check>
  <check verified="true">SemanticFingerprint has get() method returning EmbeddingRef</check>
</prerequisites>

<scope>
  <in_scope>
    - Create distance.rs module in crates/context-graph-core/src/retrieval/
    - Implement max_sim for late interaction (ColBERT MaxSim algorithm)
    - Implement transe_similarity for entity embeddings
    - Implement hamming_similarity wrapper (converts distance to similarity)
    - Create compute_similarity_for_space dispatcher using Embedder enum
    - Wrap existing vector methods with thin functions
    - Ensure all results normalized to [0.0, 1.0]
  </in_scope>
  <out_of_scope>
    - Modifying existing vector types (use them as-is)
    - SIMD optimization
    - GPU acceleration
    - Approximate similarity (LSH, etc.)
    - Category-aware weighting (handled in TASK-P3-005)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/retrieval/distance.rs">
/// Thin wrapper for DenseVector::cosine_similarity
pub fn cosine_similarity(a: &amp;[f32], b: &amp;[f32]) -> f32;

/// Thin wrapper for SparseVector::jaccard_index
pub fn jaccard_similarity(a: &amp;SparseVector, b: &amp;SparseVector) -> f32;

/// Convert BinaryVector::hamming_distance to similarity
pub fn hamming_similarity(a: &amp;BinaryVector, b: &amp;BinaryVector) -> f32;

/// ColBERT MaxSim: for each query token, find max cosine to any memory token, average
pub fn max_sim(query_tokens: &amp;[Vec&lt;f32&gt;], memory_tokens: &amp;[Vec&lt;f32&gt;]) -> f32;

/// TransE-style: 1 / (1 + euclidean_distance)
pub fn transe_similarity(a: &amp;[f32], b: &amp;[f32]) -> f32;

/// Main dispatcher - computes similarity for a specific embedding space
pub fn compute_similarity_for_space(
    embedder: Embedder,
    query: &amp;SemanticFingerprint,
    memory: &amp;SemanticFingerprint,
) -> f32;
    </signature>
  </signatures>

  <constraints>
    - All similarity results MUST be in [0.0, 1.0] range
    - Zero vectors return 0.0 similarity (not NaN) - AP-10 compliance
    - Empty late interaction tokens return 0.0
    - Identical vectors return 1.0 (or very close due to floating point)
    - No panics - all error cases handled gracefully
  </constraints>

  <verification>
    - Cosine of identical normalized vectors = 1.0
    - Jaccard of identical sparse = 1.0
    - Hamming of identical binary = 1.0
    - MaxSim handles empty token lists gracefully
    - TransE of identical vectors = 1.0
    - Zero-length vectors return 0.0, not NaN
  </verification>
</definition_of_done>

<implementation_code>
File: crates/context-graph-core/src/retrieval/distance.rs

```rust
//! Distance and similarity metrics for the 13 embedding spaces.
//!
//! This module provides unified distance/similarity computation across all
//! embedding types: dense, sparse, binary, and token-level.
//!
//! # Design Philosophy
//!
//! Most similarity functions delegate to existing vector type methods:
//! - DenseVector::cosine_similarity()
//! - SparseVector::jaccard_index()
//! - BinaryVector::hamming_distance()
//!
//! This module adds:
//! - max_sim() for ColBERT late interaction (E12)
//! - transe_similarity() for knowledge graph embeddings (E11)
//! - compute_similarity_for_space() dispatcher
//!
//! # All outputs are normalized to [0.0, 1.0]

use crate::embeddings::{DenseVector, BinaryVector};
use crate::teleological::Embedder;
use crate::types::fingerprint::{SemanticFingerprint, SparseVector, EmbeddingRef};

/// Compute cosine similarity between two dense vectors.
///
/// Thin wrapper that creates DenseVectors and delegates to existing method.
/// Returns 0.0 for zero-magnitude vectors (AP-10: no NaN).
///
/// # Arguments
/// * `a` - First dense embedding as f32 slice
/// * `b` - Second dense embedding as f32 slice
///
/// # Returns
/// Similarity in [0.0, 1.0] where 1.0 = identical direction
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let vec_a = DenseVector::new(a.to_vec());
    let vec_b = DenseVector::new(b.to_vec());

    vec_a.cosine_similarity(&vec_b)
}

/// Compute Jaccard similarity between two sparse vectors.
///
/// Thin wrapper that delegates to SparseVector::jaccard_index().
/// Returns |A ∩ B| / |A ∪ B| based on non-zero indices.
///
/// # Returns
/// Similarity in [0.0, 1.0] where 1.0 = identical index sets
pub fn jaccard_similarity(a: &SparseVector, b: &SparseVector) -> f32 {
    a.jaccard_index(b)
}

/// Compute Hamming similarity between two binary vectors.
///
/// Converts Hamming distance to similarity: 1.0 - (distance / max_bits).
///
/// # Returns
/// Similarity in [0.0, 1.0] where 1.0 = identical bit patterns
pub fn hamming_similarity(a: &BinaryVector, b: &BinaryVector) -> f32 {
    let distance = a.hamming_distance(b);
    let max_bits = a.bit_len().max(b.bit_len());

    if max_bits == 0 {
        return 1.0; // Empty vectors are identical
    }

    1.0 - (distance as f32 / max_bits as f32)
}

/// Compute MaxSim for late interaction (ColBERT-style).
///
/// For each query token, find max cosine similarity to any memory token.
/// Return mean of all max similarities.
///
/// # Algorithm
/// ```text
/// MaxSim = (1/|Q|) * Σ_q∈Q max_m∈M cos(q, m)
/// ```
///
/// # Arguments
/// * `query_tokens` - Query token embeddings (each 128D for E12)
/// * `memory_tokens` - Memory token embeddings
///
/// # Returns
/// Similarity in [0.0, 1.0], returns 0.0 if either list is empty
pub fn max_sim(query_tokens: &[Vec<f32>], memory_tokens: &[Vec<f32>]) -> f32 {
    if query_tokens.is_empty() || memory_tokens.is_empty() {
        return 0.0;
    }

    let mut total_max = 0.0_f32;

    for q_tok in query_tokens {
        let mut max_sim_for_token = 0.0_f32;

        for m_tok in memory_tokens {
            let sim = cosine_similarity(q_tok, m_tok);
            max_sim_for_token = max_sim_for_token.max(sim);
        }

        total_max += max_sim_for_token;
    }

    total_max / query_tokens.len() as f32
}

/// Compute TransE-style similarity for knowledge graph embeddings.
///
/// Uses inverse of Euclidean distance: 1 / (1 + distance).
/// This maps distance [0, ∞) to similarity (0, 1].
///
/// # Returns
/// Similarity in (0.0, 1.0] where 1.0 = identical vectors (distance = 0)
pub fn transe_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let vec_a = DenseVector::new(a.to_vec());
    let vec_b = DenseVector::new(b.to_vec());

    let distance = vec_a.euclidean_distance(&vec_b);
    1.0 / (1.0 + distance)
}

/// Compute similarity for a specific embedding space.
///
/// This is the main dispatcher that routes to the appropriate similarity
/// function based on the embedder type.
///
/// # Metrics by Embedder
/// - E1 (Semantic): Cosine
/// - E2-E4 (Temporal): Cosine
/// - E5 (Causal): Cosine (asymmetric handled at embedding time)
/// - E6 (Sparse): Jaccard
/// - E7 (Code): Cosine
/// - E8 (Emotional): Cosine
/// - E9 (HDC): Cosine (stored as projected dense, not binary)
/// - E10 (Multimodal): Cosine
/// - E11 (Entity): TransE
/// - E12 (LateInteraction): MaxSim
/// - E13 (KeywordSplade): Jaccard
///
/// # Arguments
/// * `embedder` - Which embedding space to compare
/// * `query` - Query fingerprint
/// * `memory` - Memory fingerprint
///
/// # Returns
/// Similarity in [0.0, 1.0]
pub fn compute_similarity_for_space(
    embedder: Embedder,
    query: &SemanticFingerprint,
    memory: &SemanticFingerprint,
) -> f32 {
    let query_ref = query.get(embedder);
    let memory_ref = memory.get(embedder);

    match (query_ref, memory_ref) {
        (EmbeddingRef::Dense(q), EmbeddingRef::Dense(m)) => {
            match embedder {
                Embedder::Entity => transe_similarity(q, m),
                _ => cosine_similarity(q, m),
            }
        }
        (EmbeddingRef::Sparse(q), EmbeddingRef::Sparse(m)) => {
            jaccard_similarity(q, m)
        }
        (EmbeddingRef::TokenLevel(q), EmbeddingRef::TokenLevel(m)) => {
            max_sim(q, m)
        }
        _ => {
            // Type mismatch - should never happen with valid fingerprints
            tracing::error!(
                embedder = %embedder.name(),
                "Type mismatch in compute_similarity_for_space"
            );
            0.0
        }
    }
}

/// Compute all 13 similarities between query and memory fingerprints.
///
/// Returns an array indexed by Embedder::index().
///
/// # Returns
/// Array of 13 similarity scores in [0.0, 1.0]
pub fn compute_all_similarities(
    query: &SemanticFingerprint,
    memory: &SemanticFingerprint,
) -> [f32; 13] {
    let mut scores = [0.0_f32; 13];

    for embedder in Embedder::all() {
        scores[embedder.index()] = compute_similarity_for_space(embedder, query, memory);
    }

    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Cosine Similarity Tests
    // =========================================================================

    #[test]
    fn test_cosine_identical_normalized() {
        let v: Vec<f32> = vec![0.6, 0.8, 0.0]; // Already normalized (magnitude = 1.0)
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5, "Identical vectors should have similarity 1.0, got {}", sim);
        println!("[PASS] cosine_similarity of identical normalized vectors = {:.6}", sim);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5, "Orthogonal vectors should have similarity ~0.0, got {}", sim);
        println!("[PASS] cosine_similarity of orthogonal vectors = {:.6}", sim);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        // Cosine returns [-1, 1], but DenseVector.cosine_similarity may normalize to [0, 1]
        // Check what the actual behavior is
        println!("[INFO] cosine_similarity of opposite vectors = {:.6}", sim);
        assert!(sim >= 0.0 && sim <= 1.0, "Result should be in [0, 1]");
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Zero vector should return 0.0, not NaN");
        assert!(!sim.is_nan(), "AP-10 violation: Result must not be NaN");
        println!("[PASS] cosine_similarity with zero vector = {:.6} (not NaN)", sim);
    }

    #[test]
    fn test_cosine_empty_vector() {
        let a: Vec<f32> = vec![];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Empty vector should return 0.0");
        println!("[PASS] cosine_similarity with empty vector = 0.0");
    }

    #[test]
    fn test_cosine_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Dimension mismatch should return 0.0");
        println!("[PASS] cosine_similarity with dimension mismatch = 0.0");
    }

    // =========================================================================
    // Jaccard Similarity Tests
    // =========================================================================

    #[test]
    fn test_jaccard_identical() {
        let v = SparseVector::new(vec![0, 5, 10], vec![1.0, 1.0, 1.0], 100).unwrap();
        let sim = jaccard_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5, "Identical sparse vectors should have similarity 1.0, got {}", sim);
        println!("[PASS] jaccard_similarity of identical vectors = {:.6}", sim);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a = SparseVector::new(vec![0, 1, 2], vec![1.0, 1.0, 1.0], 100).unwrap();
        let b = SparseVector::new(vec![5, 6, 7], vec![1.0, 1.0, 1.0], 100).unwrap();
        let sim = jaccard_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Disjoint sets should have similarity 0.0");
        println!("[PASS] jaccard_similarity of disjoint sets = {:.6}", sim);
    }

    #[test]
    fn test_jaccard_partial_overlap() {
        let a = SparseVector::new(vec![0, 1, 2], vec![1.0, 1.0, 1.0], 100).unwrap();
        let b = SparseVector::new(vec![1, 2, 3], vec![1.0, 1.0, 1.0], 100).unwrap();
        let sim = jaccard_similarity(&a, &b);
        // Intersection: {1, 2} = 2 elements
        // Union: {0, 1, 2, 3} = 4 elements
        // Jaccard = 2/4 = 0.5
        assert!((sim - 0.5).abs() < 1e-5, "Expected 0.5, got {}", sim);
        println!("[PASS] jaccard_similarity with 50% overlap = {:.6}", sim);
    }

    #[test]
    fn test_jaccard_empty() {
        let a = SparseVector::empty();
        let b = SparseVector::empty();
        let sim = jaccard_similarity(&a, &b);
        // Empty sets: Jaccard is typically defined as 1.0 (identical emptiness)
        // or 0.0 (no overlap). Check actual behavior.
        println!("[INFO] jaccard_similarity of empty vectors = {:.6}", sim);
        assert!(sim >= 0.0 && sim <= 1.0, "Result should be in [0, 1]");
    }

    // =========================================================================
    // Hamming Similarity Tests
    // =========================================================================

    #[test]
    fn test_hamming_identical() {
        let v = BinaryVector::zeros(64);
        let sim = hamming_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5, "Identical binary vectors should have similarity 1.0, got {}", sim);
        println!("[PASS] hamming_similarity of identical vectors = {:.6}", sim);
    }

    #[test]
    fn test_hamming_all_different() {
        let mut a = BinaryVector::zeros(64);
        let mut b = BinaryVector::zeros(64);

        // Set all bits in 'a' to 1, leave 'b' as 0
        for i in 0..64 {
            a.set_bit(i, true);
        }

        let sim = hamming_similarity(&a, &b);
        assert_eq!(sim, 0.0, "All different bits should have similarity 0.0");
        println!("[PASS] hamming_similarity of opposite vectors = {:.6}", sim);
    }

    #[test]
    fn test_hamming_half_different() {
        let mut a = BinaryVector::zeros(64);
        let mut b = BinaryVector::zeros(64);

        // Set first 32 bits to 1 in 'a'
        for i in 0..32 {
            a.set_bit(i, true);
        }

        let sim = hamming_similarity(&a, &b);
        // 32 bits different out of 64 = 0.5 distance = 0.5 similarity
        assert!((sim - 0.5).abs() < 1e-5, "Expected 0.5, got {}", sim);
        println!("[PASS] hamming_similarity with 50% difference = {:.6}", sim);
    }

    #[test]
    fn test_hamming_empty() {
        let a = BinaryVector::zeros(0);
        let b = BinaryVector::zeros(0);
        let sim = hamming_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-5, "Empty binary vectors should be identical (similarity 1.0)");
        println!("[PASS] hamming_similarity of empty vectors = {:.6}", sim);
    }

    // =========================================================================
    // MaxSim (Late Interaction) Tests
    // =========================================================================

    #[test]
    fn test_max_sim_identical() {
        let tokens = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let sim = max_sim(&tokens, &tokens);
        assert!((sim - 1.0).abs() < 1e-5, "Identical token sets should have MaxSim 1.0, got {}", sim);
        println!("[PASS] max_sim of identical token sets = {:.6}", sim);
    }

    #[test]
    fn test_max_sim_empty_query() {
        let empty: Vec<Vec<f32>> = vec![];
        let tokens = vec![vec![1.0, 0.0, 0.0]];
        let sim = max_sim(&empty, &tokens);
        assert_eq!(sim, 0.0, "Empty query should return 0.0");
        println!("[PASS] max_sim with empty query = 0.0");
    }

    #[test]
    fn test_max_sim_empty_memory() {
        let tokens = vec![vec![1.0, 0.0, 0.0]];
        let empty: Vec<Vec<f32>> = vec![];
        let sim = max_sim(&tokens, &empty);
        assert_eq!(sim, 0.0, "Empty memory should return 0.0");
        println!("[PASS] max_sim with empty memory = 0.0");
    }

    #[test]
    fn test_max_sim_partial_match() {
        let query = vec![
            vec![1.0, 0.0, 0.0],  // Will match first memory token perfectly
            vec![0.0, 0.0, 1.0],  // Orthogonal to all memory tokens
        ];
        let memory = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let sim = max_sim(&query, &memory);
        // First query token: max sim = 1.0 (perfect match)
        // Second query token: max sim = 0.0 (orthogonal to both)
        // Average = (1.0 + 0.0) / 2 = 0.5
        assert!((sim - 0.5).abs() < 1e-5, "Expected 0.5, got {}", sim);
        println!("[PASS] max_sim with partial match = {:.6}", sim);
    }

    // =========================================================================
    // TransE Similarity Tests
    // =========================================================================

    #[test]
    fn test_transe_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = transe_similarity(&v, &v);
        // Distance = 0, so similarity = 1 / (1 + 0) = 1.0
        assert!((sim - 1.0).abs() < 1e-5, "Identical vectors should have TransE similarity 1.0, got {}", sim);
        println!("[PASS] transe_similarity of identical vectors = {:.6}", sim);
    }

    #[test]
    fn test_transe_unit_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = transe_similarity(&a, &b);
        // Distance = 1.0, so similarity = 1 / (1 + 1) = 0.5
        assert!((sim - 0.5).abs() < 1e-5, "Unit distance should have TransE similarity 0.5, got {}", sim);
        println!("[PASS] transe_similarity at unit distance = {:.6}", sim);
    }

    #[test]
    fn test_transe_empty_vector() {
        let a: Vec<f32> = vec![];
        let b = vec![1.0, 2.0, 3.0];
        let sim = transe_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Empty vector should return 0.0");
        println!("[PASS] transe_similarity with empty vector = 0.0");
    }

    // =========================================================================
    // compute_similarity_for_space Tests
    // =========================================================================

    #[test]
    fn test_compute_similarity_semantic() {
        let mut query = SemanticFingerprint::zeroed();
        let mut memory = SemanticFingerprint::zeroed();

        // Set identical semantic embeddings
        query.e1_semantic = vec![1.0; 1024];
        memory.e1_semantic = vec![1.0; 1024];

        let sim = compute_similarity_for_space(Embedder::Semantic, &query, &memory);
        assert!((sim - 1.0).abs() < 1e-5, "Expected 1.0, got {}", sim);
        println!("[PASS] compute_similarity_for_space(Semantic) = {:.6}", sim);
    }

    #[test]
    fn test_compute_similarity_sparse() {
        let mut query = SemanticFingerprint::zeroed();
        let mut memory = SemanticFingerprint::zeroed();

        // Set identical sparse embeddings
        query.e6_sparse = SparseVector::new(vec![0, 5, 10], vec![1.0, 1.0, 1.0], 30522).unwrap();
        memory.e6_sparse = SparseVector::new(vec![0, 5, 10], vec![1.0, 1.0, 1.0], 30522).unwrap();

        let sim = compute_similarity_for_space(Embedder::Sparse, &query, &memory);
        assert!((sim - 1.0).abs() < 1e-5, "Expected 1.0, got {}", sim);
        println!("[PASS] compute_similarity_for_space(Sparse) = {:.6}", sim);
    }

    #[test]
    fn test_compute_similarity_late_interaction() {
        let mut query = SemanticFingerprint::zeroed();
        let mut memory = SemanticFingerprint::zeroed();

        // Set identical late interaction embeddings
        query.e12_late_interaction = vec![vec![1.0; 128], vec![0.5; 128]];
        memory.e12_late_interaction = vec![vec![1.0; 128], vec![0.5; 128]];

        let sim = compute_similarity_for_space(Embedder::LateInteraction, &query, &memory);
        assert!((sim - 1.0).abs() < 1e-5, "Expected 1.0, got {}", sim);
        println!("[PASS] compute_similarity_for_space(LateInteraction) = {:.6}", sim);
    }

    #[test]
    fn test_compute_similarity_entity_uses_transe() {
        let mut query = SemanticFingerprint::zeroed();
        let mut memory = SemanticFingerprint::zeroed();

        // Set entity embeddings with unit distance
        query.e11_entity = vec![0.0; 384];
        memory.e11_entity = vec![0.0; 384];
        memory.e11_entity[0] = 1.0; // Distance = 1.0

        let sim = compute_similarity_for_space(Embedder::Entity, &query, &memory);
        // TransE: 1 / (1 + 1) = 0.5
        assert!((sim - 0.5).abs() < 1e-5, "Expected 0.5 (TransE), got {}", sim);
        println!("[PASS] compute_similarity_for_space(Entity) uses TransE = {:.6}", sim);
    }

    // =========================================================================
    // compute_all_similarities Tests
    // =========================================================================

    #[test]
    fn test_compute_all_similarities() {
        let query = SemanticFingerprint::zeroed();
        let memory = SemanticFingerprint::zeroed();

        let scores = compute_all_similarities(&query, &memory);

        assert_eq!(scores.len(), 13);
        for (i, score) in scores.iter().enumerate() {
            assert!(
                *score >= 0.0 && *score <= 1.0,
                "Score {} for embedder {} out of range: {}",
                i,
                Embedder::from_index(i).unwrap().name(),
                score
            );
        }
        println!("[PASS] compute_all_similarities returns 13 valid scores");
    }

    // =========================================================================
    // Edge Case / Boundary Tests
    // =========================================================================

    #[test]
    fn test_nan_not_produced() {
        // Test various edge cases that might produce NaN
        let zero = vec![0.0, 0.0, 0.0];
        let normal = vec![1.0, 2.0, 3.0];

        let results = [
            cosine_similarity(&zero, &zero),
            cosine_similarity(&zero, &normal),
            transe_similarity(&zero, &zero),
        ];

        for (i, r) in results.iter().enumerate() {
            assert!(!r.is_nan(), "Result {} is NaN - AP-10 violation", i);
            assert!(!r.is_infinite(), "Result {} is infinite", i);
        }
        println!("[PASS] No NaN produced in edge cases (AP-10 compliance)");
    }

    #[test]
    fn test_all_results_in_range() {
        // Generate various test cases
        let test_vecs = [
            (vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]),
            (vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]),
            (vec![0.0, 0.0, 0.0], vec![1.0, 2.0, 3.0]),
            (vec![-1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]),
        ];

        for (a, b) in &test_vecs {
            let cos = cosine_similarity(a, b);
            let transe = transe_similarity(a, b);

            assert!(cos >= 0.0 && cos <= 1.0, "Cosine {} out of range", cos);
            assert!(transe >= 0.0 && transe <= 1.0, "TransE {} out of range", transe);
        }
        println!("[PASS] All results in [0.0, 1.0] range");
    }
}
```
</implementation_code>

<files_to_create>
  <file path="crates/context-graph-core/src/retrieval/distance.rs">DistanceCalculator implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/retrieval/mod.rs">Add `pub mod distance;` and re-exports</file>
</files_to_modify>

<modification_instructions>
For crates/context-graph-core/src/retrieval/mod.rs:

Add to module declarations:
```rust
pub mod distance;
```

Add to re-exports section:
```rust
pub use distance::{
    cosine_similarity, jaccard_similarity, hamming_similarity,
    max_sim, transe_similarity, compute_similarity_for_space,
    compute_all_similarities,
};
```
</modification_instructions>

<validation_criteria>
  <criterion>cargo check --package context-graph-core compiles without errors</criterion>
  <criterion>cargo test --package context-graph-core distance -- --nocapture passes</criterion>
  <criterion>Cosine of identical normalized vectors returns value in (0.99, 1.0]</criterion>
  <criterion>Jaccard of identical sparse sets returns 1.0</criterion>
  <criterion>Hamming of identical binary vectors returns 1.0</criterion>
  <criterion>MaxSim with empty tokens returns 0.0</criterion>
  <criterion>TransE of identical vectors returns 1.0</criterion>
  <criterion>Zero vectors return 0.0 similarity (not NaN)</criterion>
  <criterion>All results are in [0.0, 1.0] range</criterion>
</validation_criteria>

<test_commands>
  <command description="Check compilation">cargo check --package context-graph-core</command>
  <command description="Run distance module tests with output">cargo test --package context-graph-core distance -- --nocapture</command>
  <command description="Run clippy">cargo clippy --package context-graph-core -- -D warnings</command>
</test_commands>

<full_state_verification>
## Source of Truth
The similarity values computed and returned by each function.

## Execution and Inspection Strategy

After implementing, run these verification steps:

### 1. Compilation Verification
```bash
cargo check --package context-graph-core 2>&1 | head -50
```
Expected: "Finished" with no errors

### 2. Unit Test Execution
```bash
cargo test --package context-graph-core distance -- --nocapture 2>&1
```
Expected: All tests pass with "[PASS]" output for each

### 3. Integration Test with Real Fingerprints
```bash
cargo test --package context-graph-core --test integration distance_integration -- --nocapture 2>&1
```
(Create integration test if needed)

## Boundary and Edge Case Audit

### Edge Case 1: Empty Inputs
```
Before: cosine_similarity(&[], &[1.0])
After: Returns 0.0 (not panic, not NaN)
Proof: Test output shows "[PASS] cosine_similarity with empty vector = 0.0"
```

### Edge Case 2: Zero Magnitude Vectors
```
Before: cosine_similarity(&[0.0, 0.0], &[1.0, 2.0])
After: Returns 0.0 (not NaN due to 0/0)
Proof: Test output shows "[PASS] cosine_similarity with zero vector = 0.0 (not NaN)"
```

### Edge Case 3: Maximum Similarity
```
Before: cosine_similarity(&[1.0, 0.0], &[1.0, 0.0])
After: Returns value in (0.99, 1.0] (floating point tolerance)
Proof: Test output shows value approximately 1.0
```

## Evidence of Success
After running tests, capture this output:
```bash
cargo test --package context-graph-core distance -- --nocapture 2>&1 | grep -E "^\[PASS\]|^test.*ok$|^running"
```
Expected: All tests show "ok" and "[PASS]" messages printed.

## Physical Proof Verification
Since this is a computation module (not storage), verification is through:
1. Test assertions passing (return values match expected)
2. No panics during edge case handling
3. All values in documented [0.0, 1.0] range
</full_state_verification>
</task_spec>
```

## Execution Checklist

- [x] Read existing DenseVector methods in embeddings/vector.rs to verify signatures
- [x] Read existing SparseVector methods in types/fingerprint/sparse.rs to verify signatures
- [x] Read existing BinaryVector methods in embeddings/vector.rs to verify signatures
- [x] Create distance.rs in retrieval directory
- [x] Implement cosine_similarity wrapper function
- [x] Implement jaccard_similarity wrapper function
- [x] Implement hamming_similarity function (converts distance to similarity)
- [x] Implement max_sim for ColBERT late interaction
- [x] Implement transe_similarity function
- [x] Implement compute_similarity_for_space dispatcher
- [x] Implement compute_all_similarities helper
- [x] Write comprehensive unit tests
- [x] Add module to retrieval/mod.rs
- [x] Run `cargo check --package context-graph-core`
- [x] Run `cargo test --package context-graph-core distance -- --nocapture`
- [x] Run `cargo clippy --package context-graph-core -- -D warnings`
- [x] Verify all test output shows "[PASS]" messages
- [x] Verify no NaN values in any edge cases
- [x] Document test results as evidence of success
- [ ] Proceed to TASK-P3-005

## Completion Evidence (2026-01-16)

### Test Results Summary
All 47 distance-related tests passed:
- `test_cosine_identical_normalized`: 1.0 ✓
- `test_cosine_orthogonal`: 0.5 ✓
- `test_cosine_opposite`: 0.0 ✓
- `test_cosine_zero_vector`: 0.0 (AP-10 compliant) ✓
- `test_jaccard_identical`: 1.0 ✓
- `test_jaccard_disjoint`: 0.0 ✓
- `test_hamming_identical`: 1.0 ✓
- `test_hamming_all_different`: 0.0 ✓
- `test_max_sim_identical`: 1.0 ✓
- `test_max_sim_partial_match`: 0.75 ✓
- `test_transe_identical`: 1.0 ✓
- `test_transe_unit_distance`: 0.5 ✓
- `edge_case_1_very_small_magnitudes`: 1.0 (no underflow) ✓
- `edge_case_2_large_values_overflow_prevention`: 1.0 (no overflow) ✓
- `edge_case_3_single_token_opposite_directions`: 0.0 ✓
- `edge_case_all_13_spaces_zeroed_fingerprints`: All valid ✓

### Code Review
Code-simplifier agent review verdict: **PASS**
- Excellent documentation
- Strong AP-10 compliance
- Comprehensive test suite
- Correct dispatcher logic
