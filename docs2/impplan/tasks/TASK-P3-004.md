# TASK-P3-004: DistanceCalculator

```xml
<task_spec id="TASK-P3-004" version="1.0">
<metadata>
  <title>DistanceCalculator Implementation</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>23</sequence>
  <phase>3</phase>
  <implements>
    <requirement_ref>REQ-P3-01</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P2-002</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
</metadata>

<context>
Implements the DistanceCalculator with all distance/similarity metrics used
across the 13 embedding spaces:
- Cosine similarity for dense vectors
- Jaccard similarity for sparse vectors
- Hamming similarity for binary vectors
- MaxSim for late interaction (ColBERT-style)
- TransE scoring for knowledge graph embeddings

All metrics are normalized to return values in 0.0..1.0 range.

NOTE: Temporal spaces (E2-E4) use the same distance metrics as other spaces,
but downstream components (MultiSpaceSimilarity, DivergenceDetector) may skip
or weight them differently based on EmbedderCategory. The DistanceCalculator
itself computes all distances uniformly - category-aware weighting is handled
at the similarity aggregation layer.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE3-SIMILARITY-DIVERGENCE.md#component_contracts</file>
  <file purpose="vector_types">crates/context-graph-core/src/embedding/vector.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P2-002 complete (vector types exist)</check>
</prerequisites>

<scope>
  <in_scope>
    - Implement cosine_similarity for DenseVector
    - Implement jaccard_similarity for SparseVector
    - Implement hamming_similarity for BinaryVector
    - Implement max_sim for late interaction
    - Implement transe_similarity for entity embeddings
    - Add compute_similarity dispatcher by Embedder
    - Ensure all results normalized to 0.0..1.0
  </in_scope>
  <out_of_scope>
    - SIMD optimization
    - GPU acceleration
    - Approximate similarity (LSH, etc.)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/retrieval/distance.rs">
      pub fn cosine_similarity(a: &amp;DenseVector, b: &amp;DenseVector) -> f32;
      pub fn jaccard_similarity(a: &amp;SparseVector, b: &amp;SparseVector) -> f32;
      pub fn hamming_similarity(a: &amp;BinaryVector, b: &amp;BinaryVector) -> f32;
      pub fn max_sim(query_tokens: &amp;[DenseVector], memory_tokens: &amp;[DenseVector]) -> f32;
      pub fn transe_similarity(a: &amp;DenseVector, b: &amp;DenseVector) -> f32;
      pub fn compute_similarity_for_space(embedder: Embedder, query: &amp;TeleologicalArray, memory: &amp;TeleologicalArray) -> f32;
    </signature>
  </signatures>

  <constraints>
    - All similarity results in 0.0..=1.0 range
    - Zero vectors should return 0.0 similarity (not NaN)
    - Empty late interaction tokens return 0.0
    - Identical vectors should return 1.0 (or very close)
  </constraints>

  <verification>
    - Cosine of identical normalized vectors = 1.0
    - Jaccard of identical sets = 1.0
    - Hamming of identical binary = 1.0
    - MaxSim handles empty token lists
    - TransE of identical vectors = 1.0
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/retrieval/distance.rs

use crate::embedding::{Embedder, TeleologicalArray};
use crate::embedding::vector::{DenseVector, SparseVector, BinaryVector};
use crate::embedding::config::{get_distance_metric, DistanceMetric};

/// Compute cosine similarity between two dense vectors
/// Returns value in [0.0, 1.0] where 1.0 = identical direction
pub fn cosine_similarity(a: &amp;DenseVector, b: &amp;DenseVector) -> f32 {
    // Use the method from DenseVector which handles edge cases
    a.cosine_similarity(b)
}

/// Compute Jaccard similarity between two sparse vectors
/// Returns |A ∩ B| / |A ∪ B| based on non-zero indices
pub fn jaccard_similarity(a: &amp;SparseVector, b: &amp;SparseVector) -> f32 {
    // Use the method from SparseVector
    a.jaccard_similarity(b)
}

/// Compute Hamming similarity between two binary vectors
/// Returns 1.0 - (hamming_distance / total_bits)
pub fn hamming_similarity(a: &amp;BinaryVector, b: &amp;BinaryVector) -> f32 {
    let distance = a.hamming_distance(b);
    let total_bits = a.bit_len().max(b.bit_len()) as f32;

    if total_bits == 0.0 {
        return 1.0; // Empty vectors are identical
    }

    1.0 - (distance as f32 / total_bits)
}

/// Compute MaxSim for late interaction (ColBERT-style)
/// For each query token, find max similarity to any memory token
/// Return mean of all max similarities
pub fn max_sim(query_tokens: &amp;[DenseVector], memory_tokens: &amp;[DenseVector]) -> f32 {
    if query_tokens.is_empty() || memory_tokens.is_empty() {
        return 0.0;
    }

    let mut total_max = 0.0;

    for q_tok in query_tokens {
        let mut max_sim = 0.0f32;
        for m_tok in memory_tokens {
            let sim = cosine_similarity(q_tok, m_tok);
            max_sim = max_sim.max(sim);
        }
        total_max += max_sim;
    }

    total_max / query_tokens.len() as f32
}

/// Compute TransE-style similarity for knowledge graph embeddings
/// Uses inverse of Euclidean distance: 1 / (1 + distance)
pub fn transe_similarity(a: &amp;DenseVector, b: &amp;DenseVector) -> f32 {
    let distance = a.euclidean_distance(b);
    1.0 / (1.0 + distance)
}

/// Compute similarity for a specific embedding space
pub fn compute_similarity_for_space(
    embedder: Embedder,
    query: &amp;TeleologicalArray,
    memory: &amp;TeleologicalArray,
) -> f32 {
    match embedder {
        Embedder::E1Semantic => cosine_similarity(&amp;query.e1_semantic, &amp;memory.e1_semantic),
        Embedder::E2TempRecent => cosine_similarity(&amp;query.e2_temp_recent, &amp;memory.e2_temp_recent),
        Embedder::E3TempPeriodic => cosine_similarity(&amp;query.e3_temp_periodic, &amp;memory.e3_temp_periodic),
        Embedder::E4TempPosition => cosine_similarity(&amp;query.e4_temp_position, &amp;memory.e4_temp_position),
        Embedder::E5Causal => cosine_similarity(&amp;query.e5_causal, &amp;memory.e5_causal),
        Embedder::E6Sparse => jaccard_similarity(&amp;query.e6_sparse, &amp;memory.e6_sparse),
        Embedder::E7Code => cosine_similarity(&amp;query.e7_code, &amp;memory.e7_code),
        Embedder::E8Emotional => cosine_similarity(&amp;query.e8_emotional, &amp;memory.e8_emotional),
        Embedder::E9HDC => hamming_similarity(&amp;query.e9_hdc, &amp;memory.e9_hdc),
        Embedder::E10Multimodal => cosine_similarity(&amp;query.e10_multimodal, &amp;memory.e10_multimodal),
        Embedder::E11Entity => transe_similarity(&amp;query.e11_entity, &amp;memory.e11_entity),
        Embedder::E12LateInteract => max_sim(&amp;query.e12_late_interact, &amp;memory.e12_late_interact),
        Embedder::E13SPLADE => jaccard_similarity(&amp;query.e13_splade, &amp;memory.e13_splade),
    }
}

/// Compute similarity using the appropriate metric based on config
pub fn compute_similarity_by_metric(
    metric: DistanceMetric,
    query_dense: Option&lt;&amp;DenseVector&gt;,
    memory_dense: Option&lt;&amp;DenseVector&gt;,
    query_sparse: Option&lt;&amp;SparseVector&gt;,
    memory_sparse: Option&lt;&amp;SparseVector&gt;,
    query_binary: Option&lt;&amp;BinaryVector&gt;,
    memory_binary: Option&lt;&amp;BinaryVector&gt;,
    query_late: Option&lt;&amp;[DenseVector]&gt;,
    memory_late: Option&lt;&amp;[DenseVector]&gt;,
) -> f32 {
    match metric {
        DistanceMetric::Cosine => {
            if let (Some(q), Some(m)) = (query_dense, memory_dense) {
                cosine_similarity(q, m)
            } else {
                0.0
            }
        }
        DistanceMetric::Euclidean => {
            if let (Some(q), Some(m)) = (query_dense, memory_dense) {
                let dist = q.euclidean_distance(m);
                1.0 / (1.0 + dist) // Convert distance to similarity
            } else {
                0.0
            }
        }
        DistanceMetric::Jaccard => {
            if let (Some(q), Some(m)) = (query_sparse, memory_sparse) {
                jaccard_similarity(q, m)
            } else {
                0.0
            }
        }
        DistanceMetric::Hamming => {
            if let (Some(q), Some(m)) = (query_binary, memory_binary) {
                hamming_similarity(q, m)
            } else {
                0.0
            }
        }
        DistanceMetric::MaxSim => {
            if let (Some(q), Some(m)) = (query_late, memory_late) {
                max_sim(q, m)
            } else {
                0.0
            }
        }
        DistanceMetric::TransE => {
            if let (Some(q), Some(m)) = (query_dense, memory_dense) {
                transe_similarity(q, m)
            } else {
                0.0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let v = DenseVector::new(vec![1.0, 2.0, 3.0]).normalized();
        let sim = cosine_similarity(&amp;v, &amp;v);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = DenseVector::new(vec![1.0, 0.0, 0.0]);
        let b = DenseVector::new(vec![0.0, 1.0, 0.0]);
        let sim = cosine_similarity(&amp;a, &amp;b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = DenseVector::zeros(3);
        let b = DenseVector::new(vec![1.0, 2.0, 3.0]);
        let sim = cosine_similarity(&amp;a, &amp;b);
        assert_eq!(sim, 0.0); // Not NaN
    }

    #[test]
    fn test_jaccard_identical() {
        let v = SparseVector::new(vec![0, 5, 10], vec![1.0, 1.0, 1.0], 100).unwrap();
        let sim = jaccard_similarity(&amp;v, &amp;v);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a = SparseVector::new(vec![0, 1, 2], vec![1.0, 1.0, 1.0], 100).unwrap();
        let b = SparseVector::new(vec![5, 6, 7], vec![1.0, 1.0, 1.0], 100).unwrap();
        let sim = jaccard_similarity(&amp;a, &amp;b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_hamming_identical() {
        let v = BinaryVector::zeros(64);
        let sim = hamming_similarity(&amp;v, &amp;v);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_hamming_opposite() {
        let mut a = BinaryVector::zeros(64);
        let mut b = BinaryVector::zeros(64);

        for i in 0..64 {
            a.set_bit(i, true);
            b.set_bit(i, false);
        }

        let sim = hamming_similarity(&amp;a, &amp;b);
        assert_eq!(sim, 0.0); // All bits different
    }

    #[test]
    fn test_max_sim_identical() {
        let tokens = vec![
            DenseVector::new(vec![1.0, 0.0, 0.0]).normalized(),
            DenseVector::new(vec![0.0, 1.0, 0.0]).normalized(),
        ];
        let sim = max_sim(&amp;tokens, &amp;tokens);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_max_sim_empty() {
        let empty: Vec&lt;DenseVector&gt; = vec![];
        let tokens = vec![DenseVector::new(vec![1.0, 0.0, 0.0])];
        assert_eq!(max_sim(&amp;empty, &amp;tokens), 0.0);
        assert_eq!(max_sim(&amp;tokens, &amp;empty), 0.0);
    }

    #[test]
    fn test_transe_identical() {
        let v = DenseVector::new(vec![1.0, 2.0, 3.0]);
        let sim = transe_similarity(&amp;v, &amp;v);
        assert!((sim - 1.0).abs() < 1e-5); // Distance 0 => 1/(1+0) = 1
    }

    #[test]
    fn test_transe_different() {
        let a = DenseVector::new(vec![0.0, 0.0, 0.0]);
        let b = DenseVector::new(vec![1.0, 0.0, 0.0]);
        let sim = transe_similarity(&amp;a, &amp;b);
        assert!((sim - 0.5).abs() < 1e-5); // Distance 1 => 1/(1+1) = 0.5
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/retrieval/distance.rs">DistanceCalculator implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/retrieval/mod.rs">Add pub mod distance and re-exports</file>
</files_to_modify>

<validation_criteria>
  <criterion>Cosine similarity returns 1.0 for identical normalized vectors</criterion>
  <criterion>Jaccard similarity returns 1.0 for identical sets</criterion>
  <criterion>Hamming similarity returns 1.0 for identical binary vectors</criterion>
  <criterion>MaxSim handles empty token lists gracefully</criterion>
  <criterion>TransE returns 1.0 for identical vectors</criterion>
  <criterion>Zero vectors return 0.0 similarity (not NaN)</criterion>
  <criterion>All results in 0.0..1.0 range</criterion>
</validation_criteria>

<test_commands>
  <command description="Run distance tests">cargo test --package context-graph-core distance</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create distance.rs in retrieval directory
- [ ] Implement cosine_similarity function
- [ ] Implement jaccard_similarity function
- [ ] Implement hamming_similarity function
- [ ] Implement max_sim for late interaction
- [ ] Implement transe_similarity function
- [ ] Implement compute_similarity_for_space dispatcher
- [ ] Write comprehensive unit tests
- [ ] Run tests to verify
- [ ] Proceed to TASK-P3-005
