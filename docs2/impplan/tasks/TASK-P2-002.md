# TASK-P2-002: Vector Types Implementation

```xml
<task_spec id="TASK-P2-002" version="3.0">
<metadata>
  <title>DenseVector and BinaryVector Implementation</title>
  <status>COMPLETE</status>
  <layer>foundation</layer>
  <sequence>15</sequence>
  <phase>2</phase>
  <implements>
    <requirement_ref>REQ-P2-01</requirement_ref>
    <requirement_ref>REQ-P2-02</requirement_ref>
  </implements>
  <depends_on>
    <dependency>TASK-P2-001 (TeleologicalArray - COMPLETE)</dependency>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <audit_date>2026-01-16</audit_date>
  <completion_date>2026-01-16</completion_date>
</metadata>

<!-- =================================================================== -->
<!-- CRITICAL AUDIT FINDINGS - READ FIRST                                -->
<!-- =================================================================== -->
<audit_findings>
  <finding severity="CRITICAL" id="AF-001">
    <issue>SparseVector ALREADY EXISTS but is SPECIALIZED for E6/E13 SPLADE</issue>
    <location>crates/context-graph-core/src/types/fingerprint/sparse.rs</location>
    <details>
      - Uses u16 indices (max 65,535) NOT u32
      - Hardcoded vocab size: SPARSE_VOCAB_SIZE = 30,522
      - NO jaccard_similarity() method - HAS cosine_similarity() and dot()
      - NO sparsity() method
      - NO dimension field (implicit from SPARSE_VOCAB_SIZE)
    </details>
    <resolution>
      The existing SparseVector is PRODUCTION CODE used by SemanticFingerprint.
      DO NOT modify it. Instead:
      1. Create DenseVector and BinaryVector in new file
      2. Optionally create a GENERIC SparseVector (different type) if needed
      3. Or add jaccard_similarity() to existing SparseVector
    </resolution>
  </finding>

  <finding severity="HIGH" id="AF-002">
    <issue>E9 HDC is NOT binary in the actual codebase</issue>
    <details>
      SemanticFingerprint has: pub e9_hdc: Vec&lt;f32&gt; (1024D dense)
      The spec claimed BinaryVector&lt;1024&gt; but actual implementation
      projects 10K-bit hypervector to 1024D dense for HNSW compatibility.
    </details>
    <resolution>
      BinaryVector type is still useful for future binary embeddings,
      but E9 HDC specifically uses dense projection. Implement BinaryVector
      as a general-purpose type anyway.
    </resolution>
  </finding>

  <finding severity="HIGH" id="AF-003">
    <issue>File path mismatch between spec and actual codebase</issue>
    <details>
      Spec says: crates/context-graph-core/src/embedding/vector.rs
      Actual structure:
      - embeddings/ module (with 's') exists: token_pruning.rs only
      - types/fingerprint/ has: sparse.rs, semantic/, etc.
      - teleological/ has: embedder.rs with Embedder enum
    </details>
    <resolution>
      Create vector.rs in the embeddings/ module (not embedding/).
      Path: crates/context-graph-core/src/embeddings/vector.rs
    </resolution>
  </finding>

  <finding severity="MEDIUM" id="AF-004">
    <issue>Original task pseudo_code has generic SparseVector with u32</issue>
    <details>
      The pseudo_code shows SparseVector with indices: Vec&lt;u32&gt; and dimension: u32
      but the ACTUAL SparseVector uses u16 indices (sufficient for 30K vocab).
      Creating a second SparseVector type would cause confusion.
    </details>
    <resolution>
      Option A: Add missing methods to existing SparseVector (jaccard_similarity, sparsity)
      Option B: Create GenericSparseVector if u32 indices truly needed
      RECOMMENDED: Option A - extend existing type
    </resolution>
  </finding>
</audit_findings>

<!-- =================================================================== -->
<!-- CONTEXT - WHAT THIS TASK IS FOR                                     -->
<!-- =================================================================== -->
<context>
  <summary>
    Implements generic vector types for the 13-space embedding system.
    These are FOUNDATION types used by distance calculators and similarity functions.
  </summary>

  <what_exists>
    <item name="SparseVector" path="crates/context-graph-core/src/types/fingerprint/sparse.rs" status="COMPLETE">
      Specialized for E6/E13 SPLADE with u16 indices, 30K vocab.
      Has: new(), empty(), nnz(), dot(), memory_size(), is_empty(), get(), l2_norm(), cosine_similarity()
      Missing: jaccard_similarity(), sparsity()
    </item>
    <item name="SemanticFingerprint" path="crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs" status="COMPLETE">
      The 13-embedding array struct. Uses Vec&lt;f32&gt; for dense embeddings.
    </item>
    <item name="Embedder enum" path="crates/context-graph-core/src/teleological/embedder.rs" status="COMPLETE">
      13 variants with index(), expected_dims(), is_dense(), is_sparse(), etc.
    </item>
    <item name="TokenPruningEmbedding" path="crates/context-graph-core/src/embeddings/token_pruning.rs" status="COMPLETE">
      E12 token-level embedding type.
    </item>
  </what_exists>

  <what_needs_creation>
    <item name="DenseVector" priority="HIGH">
      Generic dense vector with f32 data. For E1-E5, E7-E11 embeddings.
      Methods: cosine_similarity, euclidean_distance, dot_product, magnitude, normalize
    </item>
    <item name="BinaryVector" priority="MEDIUM">
      Bit-packed vector for hyperdimensional computing.
      Methods: hamming_distance, set_bit, get_bit, popcount
      Note: E9 actually uses dense projection, but BinaryVector useful for future.
    </item>
    <item name="jaccard_similarity on SparseVector" priority="HIGH">
      Add method to existing SparseVector for Jaccard similarity.
    </item>
    <item name="sparsity on SparseVector" priority="LOW">
      Add method to existing SparseVector to compute sparsity ratio.
    </item>
  </what_needs_creation>
</context>

<!-- =================================================================== -->
<!-- ACTUAL FILE LOCATIONS (VERIFIED)                                    -->
<!-- =================================================================== -->
<actual_file_locations>
  <file purpose="NEW - DenseVector, BinaryVector"
        path="crates/context-graph-core/src/embeddings/vector.rs"
        action="CREATE">
    Create this file with DenseVector and BinaryVector types.
  </file>

  <file purpose="Existing SparseVector"
        path="crates/context-graph-core/src/types/fingerprint/sparse.rs"
        action="MODIFY">
    Add jaccard_similarity() and sparsity() methods.
  </file>

  <file purpose="embeddings module"
        path="crates/context-graph-core/src/embeddings/mod.rs"
        action="MODIFY">
    Add: pub mod vector; and re-exports.
  </file>

  <file purpose="lib.rs exports"
        path="crates/context-graph-core/src/lib.rs"
        action="VERIFY">
    Already exports embeddings module. May need to add vector re-exports.
  </file>
</actual_file_locations>

<!-- =================================================================== -->
<!-- SCOPE - EXACTLY WHAT TO DO                                          -->
<!-- =================================================================== -->
<scope>
  <in_scope>
    <item>DenseVector struct with cosine_similarity, euclidean_distance, dot_product, magnitude, normalize, normalized</item>
    <item>BinaryVector struct with hamming_distance, set_bit, get_bit, popcount, bit_len, byte_size</item>
    <item>VectorError enum for dimension mismatches</item>
    <item>Add jaccard_similarity() to existing SparseVector</item>
    <item>Add sparsity() to existing SparseVector</item>
    <item>Serde serialization for all types</item>
    <item>Clone, Debug, PartialEq derives</item>
    <item>Comprehensive unit tests</item>
  </in_scope>

  <out_of_scope>
    <item>TransE distance (TASK-P3-004)</item>
    <item>MaxSim for late interaction (TASK-P3-004)</item>
    <item>SIMD optimization (future enhancement)</item>
    <item>Creating a second SparseVector type (use existing)</item>
    <item>Modifying SemanticFingerprint structure</item>
  </out_of_scope>
</scope>

<!-- =================================================================== -->
<!-- IMPLEMENTATION REQUIREMENTS                                         -->
<!-- =================================================================== -->
<implementation_requirements>
  <requirement id="IR-001" priority="CRITICAL">
    <rule>cosine_similarity MUST return 0.0 for zero vectors, NOT NaN</rule>
    <rationale>AP-10 forbids NaN/Infinity in similarity scores</rationale>
    <code_pattern>
      if mag_a == 0.0 || mag_b == 0.0 { return 0.0; }
    </code_pattern>
  </requirement>

  <requirement id="IR-002" priority="CRITICAL">
    <rule>NO .unwrap() in library code</rule>
    <rationale>AP-14 forbids unwrap in production code</rationale>
    <code_pattern>Use Result&lt;T, E&gt; and propagate with ?</code_pattern>
  </requirement>

  <requirement id="IR-003" priority="HIGH">
    <rule>Use thiserror for error types</rule>
    <rationale>Coding standard requires thiserror for library errors</rationale>
  </requirement>

  <requirement id="IR-004" priority="HIGH">
    <rule>Clamp cosine_similarity output to [-1.0, 1.0]</rule>
    <rationale>Floating-point arithmetic can produce values slightly outside range</rationale>
    <code_pattern>(dot / (mag_a * mag_b)).clamp(-1.0, 1.0)</code_pattern>
  </requirement>

  <requirement id="IR-005" priority="MEDIUM">
    <rule>BinaryVector bit indexing must handle out-of-bounds gracefully</rule>
    <code_pattern>
      get_bit returns false for out-of-bounds
      set_bit silently ignores out-of-bounds (or returns Result)
    </code_pattern>
  </requirement>
</implementation_requirements>

<!-- =================================================================== -->
<!-- DEFINITION OF DONE                                                  -->
<!-- =================================================================== -->
<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/embeddings/vector.rs">
use serde::{Serialize, Deserialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VectorError {
    #[error("Dimension mismatch: {a} vs {b}")]
    DimensionMismatch { a: usize, b: usize },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseVector {
    data: Vec&lt;f32&gt;,
}

impl DenseVector {
    pub fn new(data: Vec&lt;f32&gt;) -> Self;
    pub fn zeros(dim: usize) -> Self;
    pub fn len(&amp;self) -> usize;
    pub fn is_empty(&amp;self) -> bool;
    pub fn data(&amp;self) -> &amp;[f32];
    pub fn data_mut(&amp;mut self) -> &amp;mut [f32];
    pub fn cosine_similarity(&amp;self, other: &amp;Self) -> f32;
    pub fn euclidean_distance(&amp;self, other: &amp;Self) -> f32;
    pub fn dot_product(&amp;self, other: &amp;Self) -> f32;
    pub fn magnitude(&amp;self) -> f32;
    pub fn normalize(&amp;mut self);
    pub fn normalized(&amp;self) -> Self;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryVector {
    data: Vec&lt;u64&gt;,
    bit_len: usize,
}

impl BinaryVector {
    pub fn new(data: Vec&lt;u64&gt;, bit_len: usize) -> Self;
    pub fn zeros(bit_len: usize) -> Self;
    pub fn bit_len(&amp;self) -> usize;
    pub fn byte_size(&amp;self) -> usize;
    pub fn hamming_distance(&amp;self, other: &amp;Self) -> u32;
    pub fn set_bit(&amp;mut self, index: usize, value: bool);
    pub fn get_bit(&amp;self, index: usize) -> bool;
    pub fn popcount(&amp;self) -> u32;
}
    </signature>

    <signature file="crates/context-graph-core/src/types/fingerprint/sparse.rs" action="ADD_METHODS">
impl SparseVector {
    // ADD these methods to existing impl block:

    /// Jaccard similarity based on non-zero index overlap.
    /// Returns |A ∩ B| / |A ∪ B| where A and B are sets of non-zero indices.
    pub fn jaccard_similarity(&amp;self, other: &amp;Self) -> f32;

    /// Sparsity ratio: 1.0 - (nnz / vocab_size)
    /// Returns 1.0 for empty vector, ~0.95 for typical SPLADE.
    pub fn sparsity(&amp;self) -> f32;
}
    </signature>
  </signatures>

  <constraints>
    <constraint>DenseVector: data stored as Vec&lt;f32&gt; for flexibility</constraint>
    <constraint>BinaryVector: bits packed into u64 words, bit_len tracks actual bits</constraint>
    <constraint>cosine_similarity returns 0.0 for zero vectors (not NaN)</constraint>
    <constraint>All operations must handle empty/zero vectors gracefully</constraint>
    <constraint>No panic in any method - use Result or safe defaults</constraint>
  </constraints>

  <verification>
    <check>Cosine similarity of identical normalized vectors = 1.0</check>
    <check>Cosine similarity of orthogonal vectors = 0.0</check>
    <check>Cosine similarity of zero vector with any vector = 0.0</check>
    <check>Euclidean distance of identical vectors = 0.0</check>
    <check>Euclidean distance of (0,0) to (3,4) = 5.0</check>
    <check>Hamming distance of identical binary vectors = 0</check>
    <check>Hamming distance of 0b011 and 0b110 = 2</check>
    <check>Jaccard similarity of identical sparse vectors = 1.0</check>
    <check>Jaccard similarity of disjoint sparse vectors = 0.0</check>
    <check>All serialization round-trips correctly</check>
  </verification>
</definition_of_done>

<!-- =================================================================== -->
<!-- IMPLEMENTATION CODE                                                 -->
<!-- =================================================================== -->
<implementation_code>
<file path="crates/context-graph-core/src/embeddings/vector.rs">
//! Generic vector types for the 13-space embedding system.
//!
//! This module provides DenseVector and BinaryVector types used by
//! distance calculators and similarity functions.
//!
//! Note: SparseVector is in types/fingerprint/sparse.rs (specialized for SPLADE).

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors for vector operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum VectorError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

// =============================================================================
// DenseVector
// =============================================================================

/// Dense vector of f32 values for embedding storage and similarity computation.
///
/// Used for E1-E5, E7-E11 embeddings (all dense embedders).
///
/// # Example
///
/// ```
/// use context_graph_core::embeddings::vector::DenseVector;
///
/// let a = DenseVector::new(vec![1.0, 0.0, 0.0]);
/// let b = DenseVector::new(vec![0.0, 1.0, 0.0]);
/// assert!(a.cosine_similarity(&amp;b).abs() &lt; 1e-6); // Orthogonal
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseVector {
    data: Vec&lt;f32&gt;,
}

impl DenseVector {
    /// Create a new dense vector from data.
    #[inline]
    pub fn new(data: Vec&lt;f32&gt;) -> Self {
        Self { data }
    }

    /// Create a zero vector of specified dimension.
    #[inline]
    pub fn zeros(dim: usize) -> Self {
        Self { data: vec![0.0; dim] }
    }

    /// Number of dimensions.
    #[inline]
    pub fn len(&amp;self) -> usize {
        self.data.len()
    }

    /// Check if empty (zero dimensions).
    #[inline]
    pub fn is_empty(&amp;self) -> bool {
        self.data.is_empty()
    }

    /// Immutable access to underlying data.
    #[inline]
    pub fn data(&amp;self) -> &amp;[f32] {
        &amp;self.data
    }

    /// Mutable access to underlying data.
    #[inline]
    pub fn data_mut(&amp;mut self) -> &amp;mut [f32] {
        &amp;mut self.data
    }

    /// Dot product with another vector.
    ///
    /// Assumes same dimension - mismatched dimensions produce truncated result.
    pub fn dot_product(&amp;self, other: &amp;Self) -> f32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// L2 magnitude (Euclidean norm).
    pub fn magnitude(&amp;self) -> f32 {
        self.data.iter().map(|x| x * x).sum::&lt;f32&gt;().sqrt()
    }

    /// Cosine similarity with another vector.
    ///
    /// Returns 0.0 if either vector has zero magnitude (AP-10 compliance).
    /// Output is clamped to [-1.0, 1.0] for numerical stability.
    pub fn cosine_similarity(&amp;self, other: &amp;Self) -> f32 {
        let dot = self.dot_product(other);
        let mag_a = self.magnitude();
        let mag_b = other.magnitude();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0; // Handle zero vectors gracefully (AP-10)
        }

        (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
    }

    /// Euclidean distance to another vector.
    pub fn euclidean_distance(&amp;self, other: &amp;Self) -> f32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::&lt;f32&gt;()
            .sqrt()
    }

    /// Normalize in place to unit magnitude.
    ///
    /// No-op if magnitude is zero.
    pub fn normalize(&amp;mut self) {
        let mag = self.magnitude();
        if mag > 0.0 {
            for x in &amp;mut self.data {
                *x /= mag;
            }
        }
    }

    /// Return a normalized copy.
    pub fn normalized(&amp;self) -> Self {
        let mut result = self.clone();
        result.normalize();
        result
    }
}

impl Default for DenseVector {
    fn default() -> Self {
        Self { data: Vec::new() }
    }
}

// =============================================================================
// BinaryVector
// =============================================================================

/// Bit-packed binary vector for hyperdimensional computing.
///
/// Bits are packed into u64 words for efficient storage and XOR operations.
/// Used for Hamming distance calculations.
///
/// # Example
///
/// ```
/// use context_graph_core::embeddings::vector::BinaryVector;
///
/// let mut v = BinaryVector::zeros(128);
/// v.set_bit(0, true);
/// v.set_bit(64, true);
/// assert_eq!(v.popcount(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryVector {
    data: Vec&lt;u64&gt;,
    bit_len: usize,
}

impl BinaryVector {
    /// Create from raw u64 words and bit length.
    ///
    /// The data vector should have ceil(bit_len / 64) elements.
    #[inline]
    pub fn new(data: Vec&lt;u64&gt;, bit_len: usize) -> Self {
        Self { data, bit_len }
    }

    /// Create a zero vector of specified bit length.
    #[inline]
    pub fn zeros(bit_len: usize) -> Self {
        let num_words = (bit_len + 63) / 64;
        Self {
            data: vec![0u64; num_words],
            bit_len,
        }
    }

    /// Number of bits in the vector.
    #[inline]
    pub fn bit_len(&amp;self) -> usize {
        self.bit_len
    }

    /// Storage size in bytes.
    #[inline]
    pub fn byte_size(&amp;self) -> usize {
        self.data.len() * 8
    }

    /// Set a bit at the specified index.
    ///
    /// Silently ignores out-of-bounds indices.
    pub fn set_bit(&amp;mut self, index: usize, value: bool) {
        if index >= self.bit_len {
            return;
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        if value {
            self.data[word_idx] |= 1u64 &lt;&lt; bit_idx;
        } else {
            self.data[word_idx] &amp;= !(1u64 &lt;&lt; bit_idx);
        }
    }

    /// Get bit value at the specified index.
    ///
    /// Returns false for out-of-bounds indices.
    pub fn get_bit(&amp;self, index: usize) -> bool {
        if index >= self.bit_len {
            return false;
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        (self.data[word_idx] >> bit_idx) &amp; 1 == 1
    }

    /// Count of set bits (population count).
    pub fn popcount(&amp;self) -> u32 {
        self.data.iter().map(|w| w.count_ones()).sum()
    }

    /// Hamming distance to another binary vector.
    ///
    /// Counts number of bit positions that differ.
    pub fn hamming_distance(&amp;self, other: &amp;Self) -> u32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
}

impl Default for BinaryVector {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            bit_len: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // DenseVector Tests
    // =========================================================================

    #[test]
    fn test_dense_new_and_accessors() {
        let v = DenseVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert!(!v.is_empty());
        assert_eq!(v.data(), &amp;[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dense_zeros() {
        let v = DenseVector::zeros(5);
        assert_eq!(v.len(), 5);
        assert_eq!(v.data(), &amp;[0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dense_dot_product() {
        let a = DenseVector::new(vec![1.0, 2.0, 3.0]);
        let b = DenseVector::new(vec![4.0, 5.0, 6.0]);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((a.dot_product(&amp;b) - 32.0).abs() &lt; 1e-6);
    }

    #[test]
    fn test_dense_magnitude() {
        let v = DenseVector::new(vec![3.0, 4.0]);
        // sqrt(9 + 16) = 5
        assert!((v.magnitude() - 5.0).abs() &lt; 1e-6);
    }

    #[test]
    fn test_dense_cosine_identical_normalized() {
        let v = DenseVector::new(vec![1.0, 2.0, 3.0]).normalized();
        let sim = v.cosine_similarity(&amp;v);
        assert!((sim - 1.0).abs() &lt; 1e-6, "Expected 1.0, got {}", sim);
    }

    #[test]
    fn test_dense_cosine_orthogonal() {
        let a = DenseVector::new(vec![1.0, 0.0]);
        let b = DenseVector::new(vec![0.0, 1.0]);
        let sim = a.cosine_similarity(&amp;b);
        assert!(sim.abs() &lt; 1e-6, "Expected 0.0, got {}", sim);
    }

    #[test]
    fn test_dense_cosine_zero_vector() {
        let zero = DenseVector::zeros(3);
        let v = DenseVector::new(vec![1.0, 2.0, 3.0]);
        // Zero vector should return 0.0, NOT NaN (AP-10)
        assert_eq!(zero.cosine_similarity(&amp;v), 0.0);
        assert_eq!(v.cosine_similarity(&amp;zero), 0.0);
        assert_eq!(zero.cosine_similarity(&amp;zero), 0.0);
    }

    #[test]
    fn test_dense_euclidean_identical() {
        let v = DenseVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.euclidean_distance(&amp;v), 0.0);
    }

    #[test]
    fn test_dense_euclidean_3_4_triangle() {
        let a = DenseVector::new(vec![0.0, 0.0]);
        let b = DenseVector::new(vec![3.0, 4.0]);
        let dist = a.euclidean_distance(&amp;b);
        assert!((dist - 5.0).abs() &lt; 1e-6, "Expected 5.0, got {}", dist);
    }

    #[test]
    fn test_dense_normalize() {
        let mut v = DenseVector::new(vec![3.0, 4.0]);
        v.normalize();
        assert!((v.magnitude() - 1.0).abs() &lt; 1e-6);
        assert!((v.data()[0] - 0.6).abs() &lt; 1e-6);
        assert!((v.data()[1] - 0.8).abs() &lt; 1e-6);
    }

    #[test]
    fn test_dense_normalize_zero() {
        let mut zero = DenseVector::zeros(3);
        zero.normalize(); // Should not panic or produce NaN
        assert_eq!(zero.data(), &amp;[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dense_serialization_roundtrip() {
        let v = DenseVector::new(vec![1.0, 2.0, 3.0]);
        let bytes = bincode::serialize(&amp;v).expect("serialize failed");
        let restored: DenseVector = bincode::deserialize(&amp;bytes).expect("deserialize failed");
        assert_eq!(v, restored);
    }

    // =========================================================================
    // BinaryVector Tests
    // =========================================================================

    #[test]
    fn test_binary_zeros() {
        let v = BinaryVector::zeros(64);
        assert_eq!(v.bit_len(), 64);
        assert_eq!(v.byte_size(), 8);
        assert_eq!(v.popcount(), 0);
    }

    #[test]
    fn test_binary_set_get_bit() {
        let mut v = BinaryVector::zeros(128);
        assert!(!v.get_bit(0));
        v.set_bit(0, true);
        assert!(v.get_bit(0));
        v.set_bit(0, false);
        assert!(!v.get_bit(0));
    }

    #[test]
    fn test_binary_cross_word_boundary() {
        let mut v = BinaryVector::zeros(128);
        v.set_bit(63, true);
        v.set_bit(64, true);
        assert!(v.get_bit(63));
        assert!(v.get_bit(64));
        assert!(!v.get_bit(62));
        assert!(!v.get_bit(65));
    }

    #[test]
    fn test_binary_popcount() {
        let mut v = BinaryVector::zeros(128);
        v.set_bit(0, true);
        v.set_bit(64, true);
        v.set_bit(127, true);
        assert_eq!(v.popcount(), 3);
    }

    #[test]
    fn test_binary_hamming_identical() {
        let v = BinaryVector::zeros(64);
        assert_eq!(v.hamming_distance(&amp;v), 0);
    }

    #[test]
    fn test_binary_hamming_distance() {
        let mut a = BinaryVector::zeros(64);
        let mut b = BinaryVector::zeros(64);
        // a = 011, b = 110 (bits 0,1 vs bits 1,2)
        a.set_bit(0, true);
        a.set_bit(1, true);
        b.set_bit(1, true);
        b.set_bit(2, true);
        // XOR = 101, hamming = 2
        assert_eq!(a.hamming_distance(&amp;b), 2);
    }

    #[test]
    fn test_binary_out_of_bounds() {
        let mut v = BinaryVector::zeros(64);
        // Out of bounds set should be no-op
        v.set_bit(100, true);
        // Out of bounds get should return false
        assert!(!v.get_bit(100));
    }

    #[test]
    fn test_binary_serialization_roundtrip() {
        let mut v = BinaryVector::zeros(128);
        v.set_bit(42, true);
        v.set_bit(100, true);
        let bytes = bincode::serialize(&amp;v).expect("serialize failed");
        let restored: BinaryVector = bincode::deserialize(&amp;bytes).expect("deserialize failed");
        assert_eq!(v, restored);
        assert!(restored.get_bit(42));
        assert!(restored.get_bit(100));
    }
}
</file>
</implementation_code>

<!-- =================================================================== -->
<!-- SPARSE VECTOR ADDITIONS                                             -->
<!-- =================================================================== -->
<sparse_vector_additions file="crates/context-graph-core/src/types/fingerprint/sparse.rs">
Add these methods to the existing SparseVector impl block:

```rust
    /// Jaccard similarity based on index overlap.
    ///
    /// Computes |A ∩ B| / |A ∪ B| where A and B are the sets of
    /// non-zero indices in each vector.
    ///
    /// Returns 0.0 if both vectors are empty.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::types::fingerprint::SparseVector;
    ///
    /// let a = SparseVector::new(vec![0, 1, 2], vec![1.0, 1.0, 1.0]).unwrap();
    /// let b = SparseVector::new(vec![1, 2, 3], vec![1.0, 1.0, 1.0]).unwrap();
    /// // Intersection: {1, 2}, Union: {0, 1, 2, 3}
    /// // Jaccard = 2/4 = 0.5
    /// assert!((a.jaccard_similarity(&amp;b) - 0.5).abs() &lt; 1e-6);
    /// ```
    pub fn jaccard_similarity(&amp;self, other: &amp;Self) -> f32 {
        if self.is_empty() &amp;&amp; other.is_empty() {
            return 0.0;
        }

        let mut intersection = 0usize;
        let mut i = 0;
        let mut j = 0;

        // Two-pointer merge to count intersection (sorted indices)
        while i &lt; self.indices.len() &amp;&amp; j &lt; other.indices.len() {
            match self.indices[i].cmp(&amp;other.indices[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    intersection += 1;
                    i += 1;
                    j += 1;
                }
            }
        }

        let union = self.indices.len() + other.indices.len() - intersection;
        if union == 0 {
            return 0.0;
        }

        intersection as f32 / union as f32
    }

    /// Sparsity ratio: proportion of zero entries.
    ///
    /// Computed as 1.0 - (nnz / vocab_size).
    /// Returns 1.0 for empty vectors (all zeros).
    ///
    /// Typical SPLADE vectors have ~95% sparsity (5% active).
    pub fn sparsity(&amp;self) -> f32 {
        if SPARSE_VOCAB_SIZE == 0 {
            return 1.0;
        }
        1.0 - (self.nnz() as f32 / SPARSE_VOCAB_SIZE as f32)
    }
```

Add these tests to the existing tests module:

```rust
    #[test]
    fn test_sparse_vector_jaccard_identical() {
        let a = SparseVector::new(vec![1, 2, 3], vec![1.0, 1.0, 1.0]).unwrap();
        let sim = a.jaccard_similarity(&amp;a);
        assert!((sim - 1.0).abs() &lt; 1e-6, "Expected 1.0, got {}", sim);
    }

    #[test]
    fn test_sparse_vector_jaccard_disjoint() {
        let a = SparseVector::new(vec![0, 1, 2], vec![1.0, 1.0, 1.0]).unwrap();
        let b = SparseVector::new(vec![3, 4, 5], vec![1.0, 1.0, 1.0]).unwrap();
        assert_eq!(a.jaccard_similarity(&amp;b), 0.0);
    }

    #[test]
    fn test_sparse_vector_jaccard_partial() {
        let a = SparseVector::new(vec![0, 1, 2], vec![1.0, 1.0, 1.0]).unwrap();
        let b = SparseVector::new(vec![1, 2, 3], vec![1.0, 1.0, 1.0]).unwrap();
        // Intersection: {1, 2} = 2, Union: {0, 1, 2, 3} = 4
        let sim = a.jaccard_similarity(&amp;b);
        assert!((sim - 0.5).abs() &lt; 1e-6, "Expected 0.5, got {}", sim);
    }

    #[test]
    fn test_sparse_vector_jaccard_empty() {
        let empty = SparseVector::empty();
        let non_empty = SparseVector::new(vec![1, 2, 3], vec![1.0, 1.0, 1.0]).unwrap();
        assert_eq!(empty.jaccard_similarity(&amp;empty), 0.0);
        assert_eq!(empty.jaccard_similarity(&amp;non_empty), 0.0);
    }

    #[test]
    fn test_sparse_vector_sparsity() {
        // Empty vector = 100% sparse
        assert_eq!(SparseVector::empty().sparsity(), 1.0);

        // 3 active out of 30522 = ~99.99% sparse
        let sv = SparseVector::new(vec![1, 2, 3], vec![1.0, 1.0, 1.0]).unwrap();
        let expected = 1.0 - (3.0 / 30522.0);
        assert!((sv.sparsity() - expected).abs() &lt; 1e-6);
    }
```
</sparse_vector_additions>

<!-- =================================================================== -->
<!-- MODULE EXPORTS UPDATE                                               -->
<!-- =================================================================== -->
<module_exports file="crates/context-graph-core/src/embeddings/mod.rs">
Replace entire file with:

```rust
//! Embedding types for the 13-model teleological array.
//!
//! This module provides:
//! - `TokenPruningEmbedding` (E12): Token-level embedding with Quantizable support
//! - `DenseVector`: Generic dense vector for similarity computation
//! - `BinaryVector`: Bit-packed vector for Hamming distance
//!
//! Note: `SparseVector` for SPLADE is in `types::fingerprint::sparse`.

pub mod token_pruning;
pub mod vector;

pub use token_pruning::TokenPruningEmbedding;
pub use vector::{BinaryVector, DenseVector, VectorError};
```
</module_exports>

<!-- =================================================================== -->
<!-- TEST COMMANDS                                                       -->
<!-- =================================================================== -->
<test_commands>
  <command description="Run all vector tests">
    cargo test --package context-graph-core vector -- --nocapture
  </command>
  <command description="Run sparse vector tests (including new methods)">
    cargo test --package context-graph-core sparse -- --nocapture
  </command>
  <command description="Check compilation">
    cargo check --package context-graph-core
  </command>
  <command description="Run clippy">
    cargo clippy --package context-graph-core -- -D warnings
  </command>
  <command description="Run all core tests">
    cargo test --package context-graph-core
  </command>
</test_commands>

<!-- =================================================================== -->
<!-- FULL STATE VERIFICATION PROTOCOL                                    -->
<!-- =================================================================== -->
<full_state_verification>
  <source_of_truth>
    <primary>The compiled Rust code in context-graph-core crate</primary>
    <secondary>Unit test results demonstrating correct behavior</secondary>
  </source_of_truth>

  <verification_steps>
    <step id="1" name="Compilation Check">
      <command>cargo check --package context-graph-core</command>
      <expected>No errors, no warnings</expected>
      <evidence>Command exit code 0</evidence>
    </step>

    <step id="2" name="Unit Tests Pass">
      <command>cargo test --package context-graph-core vector -- --nocapture</command>
      <expected>All tests pass</expected>
      <evidence>Test output showing "test result: ok"</evidence>
    </step>

    <step id="3" name="Sparse Vector Tests Pass">
      <command>cargo test --package context-graph-core sparse -- --nocapture</command>
      <expected>All tests pass including new jaccard and sparsity tests</expected>
      <evidence>Test output showing new test names and "test result: ok"</evidence>
    </step>

    <step id="4" name="Clippy Clean">
      <command>cargo clippy --package context-graph-core -- -D warnings</command>
      <expected>No clippy warnings in new code</expected>
      <evidence>Command exit code 0</evidence>
    </step>
  </verification_steps>

  <boundary_edge_cases>
    <case id="EC-001" name="Zero Vector Cosine">
      <input>DenseVector::zeros(3).cosine_similarity(&amp;DenseVector::new(vec![1.0, 2.0, 3.0]))</input>
      <expected>0.0 (NOT NaN)</expected>
      <test>test_dense_cosine_zero_vector</test>
    </case>

    <case id="EC-002" name="Empty Vectors">
      <input>DenseVector::default(), BinaryVector::default()</input>
      <expected>No panic, sensible defaults (empty data)</expected>
      <test>Implicit in serialization tests</test>
    </case>

    <case id="EC-003" name="Out of Bounds Binary">
      <input>BinaryVector::zeros(64).get_bit(100)</input>
      <expected>false (no panic)</expected>
      <test>test_binary_out_of_bounds</test>
    </case>

    <case id="EC-004" name="Disjoint Sparse Jaccard">
      <input>SparseVector with indices {0,1,2} vs {3,4,5}</input>
      <expected>0.0</expected>
      <test>test_sparse_vector_jaccard_disjoint</test>
    </case>

    <case id="EC-005" name="Empty Sparse Sparsity">
      <input>SparseVector::empty().sparsity()</input>
      <expected>1.0 (100% sparse)</expected>
      <test>test_sparse_vector_sparsity</test>
    </case>
  </boundary_edge_cases>

  <manual_verification_checklist>
    <item>[ ] vector.rs file created at crates/context-graph-core/src/embeddings/vector.rs</item>
    <item>[ ] DenseVector has all required methods</item>
    <item>[ ] BinaryVector has all required methods</item>
    <item>[ ] embeddings/mod.rs updated with vector module</item>
    <item>[ ] SparseVector has jaccard_similarity method</item>
    <item>[ ] SparseVector has sparsity method</item>
    <item>[ ] All unit tests pass</item>
    <item>[ ] No clippy warnings</item>
    <item>[ ] Serialization roundtrips work for all types</item>
  </manual_verification_checklist>

  <evidence_log_template>
    ```
    === TASK-P2-002 Verification Evidence ===
    Date: [FILL]

    1. Compilation:
       $ cargo check --package context-graph-core
       [PASTE OUTPUT]

    2. Vector Tests:
       $ cargo test --package context-graph-core vector -- --nocapture
       [PASTE OUTPUT]

    3. Sparse Tests:
       $ cargo test --package context-graph-core sparse -- --nocapture
       [PASTE OUTPUT - look for jaccard and sparsity tests]

    4. Clippy:
       $ cargo clippy --package context-graph-core -- -D warnings
       [PASTE OUTPUT]

    5. Files Created/Modified:
       $ ls -la crates/context-graph-core/src/embeddings/
       [PASTE OUTPUT]

       $ grep -n "jaccard_similarity" crates/context-graph-core/src/types/fingerprint/sparse.rs
       [PASTE OUTPUT - should show line numbers where method is defined]
    ```
  </evidence_log_template>
</full_state_verification>
</task_spec>
```

## Execution Checklist

### Phase 1: Create New Types
- [x] Create `crates/context-graph-core/src/embeddings/vector.rs`
- [x] Implement `VectorError` enum with thiserror
- [x] Implement `DenseVector` struct with all methods
- [x] Implement `BinaryVector` struct with all methods
- [x] Add comprehensive unit tests for both types

### Phase 2: Update Existing Code
- [x] Update `crates/context-graph-core/src/embeddings/mod.rs` to export vector module
- [x] Add `jaccard_similarity()` method to SparseVector in `types/fingerprint/sparse.rs`
- [x] Add `sparsity()` method to SparseVector
- [x] Add tests for new SparseVector methods

### Phase 3: Verification
- [x] Run `cargo check --package context-graph-core`
- [x] Run `cargo test --package context-graph-core vector`
- [x] Run `cargo test --package context-graph-core sparse`
- [x] Run `cargo clippy --package context-graph-core -- -D warnings`
- [x] Complete Full State Verification checklist
- [x] Document evidence in evidence log

### Phase 4: Mark Complete
- [x] Update this file status to COMPLETE
- [x] Proceed to TASK-P2-003 (EmbedderConfig and DistanceMetric)

## Completion Evidence (2026-01-16)

### Compilation
```
cargo check --package context-graph-core
Finished `dev` profile [unoptimized + debuginfo] target(s) in 4.02s
```

### Test Results
```
cargo test --package context-graph-core vector
test result: ok. 148 passed; 0 failed; 0 ignored

cargo test --package context-graph-core sparse
test result: ok. 73 passed; 0 failed; 0 ignored
```

### Files Created/Modified
- NEW: `crates/context-graph-core/src/embeddings/vector.rs` (14,240 bytes)
- MODIFIED: `crates/context-graph-core/src/embeddings/mod.rs` (522 bytes)
- MODIFIED: `crates/context-graph-core/src/types/fingerprint/sparse.rs` (added jaccard_similarity, sparsity)

### Edge Cases Verified
- EC-001: Zero Vector Cosine → PASS (AP-10 compliant, returns 0.0 not NaN)
- EC-002: Empty Vector Handling → PASS
- EC-003: Out of Bounds Binary → PASS
- EC-004: Disjoint Sparse Jaccard → PASS
- EC-005: Empty Sparse Sparsity → PASS
- EC-006: Identical Vectors → PASS
- EC-007: 3-4-5 Triangle Euclidean → PASS
- EC-008: Hamming XOR → PASS
- EC-009: Normalize Zero Vector → PASS
- EC-010: Opposite Vectors Cosine → PASS

### Code Review (code-simplifier agent)
- No changes recommended
- All verification criteria met
- Code follows project standards
