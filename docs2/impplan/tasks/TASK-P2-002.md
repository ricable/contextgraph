# TASK-P2-002: Vector Types Implementation

```xml
<task_spec id="TASK-P2-002" version="1.0">
<metadata>
  <title>DenseVector, SparseVector, BinaryVector Implementation</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>15</sequence>
  <phase>2</phase>
  <implements>
    <requirement_ref>REQ-P2-01</requirement_ref>
    <requirement_ref>REQ-P2-02</requirement_ref>
  </implements>
  <depends_on>
    <!-- Foundation type - no dependencies -->
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
</metadata>

<context>
Implements the three vector types used in the 13-space embedding system:
- DenseVector: Standard floating-point vectors for most embedders
- SparseVector: Sparse representation for BoW/TF-IDF/SPLADE
- BinaryVector: Bit-packed vectors for hyperdimensional computing

Each type implements appropriate distance/similarity metrics.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE2-EMBEDDING-13SPACE.md#data_models</file>
</input_context_files>

<prerequisites>
  <check>crates/context-graph-core/src/embedding/ directory exists</check>
</prerequisites>

<scope>
  <in_scope>
    - DenseVector with cosine_similarity, euclidean_distance, normalize
    - SparseVector with jaccard_similarity, dot_product, sparsity
    - BinaryVector with hamming_distance, popcount, bit operations
    - Serialization with serde
    - Clone, Debug, PartialEq for all types
    - Unit tests for all distance metrics
  </in_scope>
  <out_of_scope>
    - TransE distance (part of DistanceCalculator in P3-004)
    - MaxSim for late interaction (part of P3-004)
    - SIMD optimization (future enhancement)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/embedding/vector.rs">
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
      pub struct SparseVector {
          indices: Vec&lt;u32&gt;,
          values: Vec&lt;f32&gt;,
          dimension: u32,
      }

      impl SparseVector {
          pub fn new(indices: Vec&lt;u32&gt;, values: Vec&lt;f32&gt;, dimension: u32) -> Result&lt;Self, VectorError&gt;;
          pub fn empty(dimension: u32) -> Self;
          pub fn jaccard_similarity(&amp;self, other: &amp;Self) -> f32;
          pub fn dot_product(&amp;self, other: &amp;Self) -> f32;
          pub fn sparsity(&amp;self) -> f32;
          pub fn nnz(&amp;self) -> usize;
          pub fn byte_size(&amp;self) -> usize;
      }

      #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
      pub struct BinaryVector {
          data: Vec&lt;u64&gt;,
          bit_len: usize,
      }

      impl BinaryVector {
          pub fn new(data: Vec&lt;u64&gt;, bit_len: usize) -> Self;
          pub fn zeros(bit_len: usize) -> Self;
          pub fn hamming_distance(&amp;self, other: &amp;Self) -> u32;
          pub fn set_bit(&amp;mut self, index: usize, value: bool);
          pub fn get_bit(&amp;self, index: usize) -> bool;
          pub fn popcount(&amp;self) -> u32;
          pub fn byte_size(&amp;self) -> usize;
      }
    </signature>
  </signatures>

  <constraints>
    - DenseVector: data stored as Vec&lt;f32&gt; for flexibility
    - SparseVector: indices must be sorted and unique
    - BinaryVector: bits packed into u64 words
    - cosine_similarity returns 0.0 for zero vectors (not NaN)
    - All operations must handle empty/zero vectors gracefully
  </constraints>

  <verification>
    - Cosine similarity of identical normalized vectors = 1.0
    - Euclidean distance of identical vectors = 0.0
    - Jaccard similarity of identical sparse vectors = 1.0
    - Hamming distance of identical binary vectors = 0
    - All serialization round-trips correctly
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/embedding/vector.rs

use serde::{Serialize, Deserialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VectorError {
    #[error("Indices and values must have same length: {indices_len} vs {values_len}")]
    LengthMismatch { indices_len: usize, values_len: usize },
    #[error("Indices must be sorted and unique")]
    UnsortedIndices,
    #[error("Index {index} exceeds dimension {dimension}")]
    IndexOutOfBounds { index: u32, dimension: u32 },
    #[error("Dimension mismatch: {a} vs {b}")]
    DimensionMismatch { a: usize, b: usize },
}

// =============================================================================
// DenseVector
// =============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseVector {
    data: Vec&lt;f32&gt;,
}

impl DenseVector {
    pub fn new(data: Vec&lt;f32&gt;) -> Self {
        Self { data }
    }

    pub fn zeros(dim: usize) -> Self {
        Self { data: vec![0.0; dim] }
    }

    pub fn len(&amp;self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&amp;self) -> bool {
        self.data.is_empty()
    }

    pub fn data(&amp;self) -> &amp;[f32] {
        &amp;self.data
    }

    pub fn data_mut(&amp;mut self) -> &amp;mut [f32] {
        &amp;mut self.data
    }

    pub fn dot_product(&amp;self, other: &amp;Self) -> f32 {
        self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    pub fn magnitude(&amp;self) -> f32 {
        self.data.iter()
            .map(|x| x * x)
            .sum::&lt;f32&gt;()
            .sqrt()
    }

    pub fn cosine_similarity(&amp;self, other: &amp;Self) -> f32 {
        let dot = self.dot_product(other);
        let mag_a = self.magnitude();
        let mag_b = other.magnitude();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0; // Handle zero vectors gracefully
        }

        (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
    }

    pub fn euclidean_distance(&amp;self, other: &amp;Self) -> f32 {
        self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::&lt;f32&gt;()
            .sqrt()
    }

    pub fn normalize(&amp;mut self) {
        let mag = self.magnitude();
        if mag > 0.0 {
            for x in &amp;mut self.data {
                *x /= mag;
            }
        }
    }

    pub fn normalized(&amp;self) -> Self {
        let mut result = self.clone();
        result.normalize();
        result
    }
}

// =============================================================================
// SparseVector
// =============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseVector {
    indices: Vec&lt;u32&gt;,
    values: Vec&lt;f32&gt;,
    dimension: u32,
}

impl SparseVector {
    pub fn new(indices: Vec&lt;u32&gt;, values: Vec&lt;f32&gt;, dimension: u32) -> Result&lt;Self, VectorError&gt; {
        if indices.len() != values.len() {
            return Err(VectorError::LengthMismatch {
                indices_len: indices.len(),
                values_len: values.len(),
            });
        }

        // Verify sorted and unique
        for i in 1..indices.len() {
            if indices[i] &lt;= indices[i - 1] {
                return Err(VectorError::UnsortedIndices);
            }
        }

        // Verify within bounds
        if let Some(&amp;max_idx) = indices.last() {
            if max_idx >= dimension {
                return Err(VectorError::IndexOutOfBounds {
                    index: max_idx,
                    dimension,
                });
            }
        }

        Ok(Self { indices, values, dimension })
    }

    pub fn empty(dimension: u32) -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            dimension,
        }
    }

    pub fn nnz(&amp;self) -> usize {
        self.indices.len()
    }

    pub fn sparsity(&amp;self) -> f32 {
        if self.dimension == 0 {
            return 1.0;
        }
        1.0 - (self.nnz() as f32 / self.dimension as f32)
    }

    pub fn byte_size(&amp;self) -> usize {
        // indices (4 bytes each) + values (4 bytes each) + dimension (4 bytes)
        self.indices.len() * 4 + self.values.len() * 4 + 4
    }

    pub fn dot_product(&amp;self, other: &amp;Self) -> f32 {
        let mut result = 0.0;
        let mut i = 0;
        let mut j = 0;

        while i &lt; self.indices.len() &amp;&amp; j &lt; other.indices.len() {
            if self.indices[i] == other.indices[j] {
                result += self.values[i] * other.values[j];
                i += 1;
                j += 1;
            } else if self.indices[i] &lt; other.indices[j] {
                i += 1;
            } else {
                j += 1;
            }
        }

        result
    }

    pub fn jaccard_similarity(&amp;self, other: &amp;Self) -> f32 {
        // Jaccard = |A ∩ B| / |A ∪ B| (based on non-zero indices)
        let set_a: std::collections::HashSet&lt;_&gt; = self.indices.iter().collect();
        let set_b: std::collections::HashSet&lt;_&gt; = other.indices.iter().collect();

        let intersection = set_a.intersection(&amp;set_b).count();
        let union = set_a.union(&amp;set_b).count();

        if union == 0 {
            return 0.0;
        }

        intersection as f32 / union as f32
    }
}

// =============================================================================
// BinaryVector
// =============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryVector {
    data: Vec&lt;u64&gt;,
    bit_len: usize,
}

impl BinaryVector {
    pub fn new(data: Vec&lt;u64&gt;, bit_len: usize) -> Self {
        Self { data, bit_len }
    }

    pub fn zeros(bit_len: usize) -> Self {
        let num_words = (bit_len + 63) / 64;
        Self {
            data: vec![0u64; num_words],
            bit_len,
        }
    }

    pub fn bit_len(&amp;self) -> usize {
        self.bit_len
    }

    pub fn byte_size(&amp;self) -> usize {
        self.data.len() * 8
    }

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

    pub fn get_bit(&amp;self, index: usize) -> bool {
        if index >= self.bit_len {
            return false;
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        (self.data[word_idx] >> bit_idx) &amp; 1 == 1
    }

    pub fn popcount(&amp;self) -> u32 {
        self.data.iter()
            .map(|w| w.count_ones())
            .sum()
    }

    pub fn hamming_distance(&amp;self, other: &amp;Self) -> u32 {
        self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_cosine_identical() {
        let v = DenseVector::new(vec![1.0, 2.0, 3.0]).normalized();
        let sim = v.cosine_similarity(&amp;v);
        assert!((sim - 1.0).abs() &lt; 1e-6);
    }

    #[test]
    fn test_dense_cosine_orthogonal() {
        let a = DenseVector::new(vec![1.0, 0.0]);
        let b = DenseVector::new(vec![0.0, 1.0]);
        let sim = a.cosine_similarity(&amp;b);
        assert!(sim.abs() &lt; 1e-6);
    }

    #[test]
    fn test_dense_euclidean() {
        let a = DenseVector::new(vec![0.0, 0.0]);
        let b = DenseVector::new(vec![3.0, 4.0]);
        let dist = a.euclidean_distance(&amp;b);
        assert!((dist - 5.0).abs() &lt; 1e-6);
    }

    #[test]
    fn test_sparse_jaccard() {
        let a = SparseVector::new(vec![0, 1, 2], vec![1.0, 1.0, 1.0], 10).unwrap();
        let b = SparseVector::new(vec![1, 2, 3], vec![1.0, 1.0, 1.0], 10).unwrap();
        let sim = a.jaccard_similarity(&amp;b);
        // Intersection: {1, 2}, Union: {0, 1, 2, 3}
        assert!((sim - 0.5).abs() &lt; 1e-6);
    }

    #[test]
    fn test_binary_hamming() {
        let mut a = BinaryVector::zeros(64);
        let mut b = BinaryVector::zeros(64);
        a.set_bit(0, true);
        a.set_bit(1, true);
        b.set_bit(1, true);
        b.set_bit(2, true);
        // a = 011, b = 110, XOR = 101, distance = 2
        let dist = a.hamming_distance(&amp;b);
        assert_eq!(dist, 2);
    }

    #[test]
    fn test_binary_popcount() {
        let mut v = BinaryVector::zeros(128);
        v.set_bit(0, true);
        v.set_bit(64, true);
        v.set_bit(127, true);
        assert_eq!(v.popcount(), 3);
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/embedding/vector.rs">Vector type implementations</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/embedding/mod.rs">Add pub mod vector</file>
</files_to_modify>

<validation_criteria>
  <criterion>DenseVector cosine_similarity works correctly</criterion>
  <criterion>DenseVector euclidean_distance works correctly</criterion>
  <criterion>SparseVector jaccard_similarity works correctly</criterion>
  <criterion>SparseVector validates sorted/unique indices</criterion>
  <criterion>BinaryVector hamming_distance works correctly</criterion>
  <criterion>All types serialize/deserialize correctly</criterion>
  <criterion>Zero vectors handled gracefully (no NaN/panic)</criterion>
</validation_criteria>

<test_commands>
  <command description="Run vector tests">cargo test --package context-graph-core vector</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create vector.rs in embedding directory
- [ ] Implement VectorError enum
- [ ] Implement DenseVector struct
- [ ] Implement SparseVector struct with validation
- [ ] Implement BinaryVector struct
- [ ] Add distance/similarity methods to each type
- [ ] Write comprehensive unit tests
- [ ] Run tests to verify
- [ ] Proceed to TASK-P2-003
