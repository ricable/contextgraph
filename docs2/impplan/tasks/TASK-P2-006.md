# TASK-P2-006: Quantizer Implementation

```xml
<task_spec id="TASK-P2-006" version="1.0">
<metadata>
  <title>Quantizer Implementation</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>19</sequence>
  <phase>2</phase>
  <implements>
    <requirement_ref>REQ-P2-03</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P2-005</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
</metadata>

<context>
Implements the Quantizer that compresses TeleologicalArray from ~100KB to ~11KB
using different quantization methods per embedder:
- PQ-8: Product quantization with 8-bit codes (E1, E5, E7, E10)
- Float8: 8-bit floating point (E2, E3, E4, E8, E11, E12)
- Binary: Bit packing (E9 - already binary)
- Inverted: Sparse index format (E6, E13)

Quantization is lossy for PQ-8 and Float8 but acceptable for similarity search.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE2-EMBEDDING-13SPACE.md#component_contracts</file>
  <file purpose="config">crates/context-graph-core/src/embedding/config.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P2-005 complete (MultiArrayProvider exists)</check>
  <check>All vector types implemented</check>
</prerequisites>

<scope>
  <in_scope>
    - Create QuantizedArray struct
    - Implement quantize_array() for TeleologicalArray
    - Implement dequantize_array() for retrieval
    - PQ-8 quantization (product quantization)
    - Float8 quantization (8-bit float)
    - Inverted index format for sparse vectors
    - Binary packing (already handled in BinaryVector)
    - QuantizeError enum
  </in_scope>
  <out_of_scope>
    - Trained PQ codebooks (use random initialization for now)
    - SIMD-optimized quantization
    - GPU quantization
    - Adaptive quantization based on content
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/embedding/quantize.rs">
      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct QuantizedArray {
          pub e1_semantic: QuantizedDense,
          pub e2_temp_recent: QuantizedFloat8,
          pub e3_temp_periodic: QuantizedFloat8,
          pub e4_temp_position: QuantizedFloat8,
          pub e5_causal: QuantizedDense,
          pub e6_sparse: InvertedIndex,
          pub e7_code: QuantizedDense,
          pub e8_emotional: QuantizedFloat8,
          pub e9_hdc: PackedBinary,
          pub e10_multimodal: QuantizedDense,
          pub e11_entity: QuantizedFloat8,
          pub e12_late_interact: Vec&lt;QuantizedFloat8&gt;,
          pub e13_splade: InvertedIndex,
      }

      #[derive(Debug, Error)]
      pub enum QuantizeError {
          #[error("Invalid input: {message}")]
          InvalidInput { message: String },
          #[error("Codebook missing for {embedder:?}")]
          CodebookMissing { embedder: Embedder },
      }

      pub fn quantize_array(array: &amp;TeleologicalArray) -> Result&lt;QuantizedArray, QuantizeError&gt;;
      pub fn dequantize_array(quantized: &amp;QuantizedArray) -> Result&lt;TeleologicalArray, QuantizeError&gt;;
    </signature>
  </signatures>

  <constraints>
    - QuantizedArray must be ~11KB (74% smaller than raw)
    - PQ-8 uses 8-bit codes per subvector
    - Float8 scales values to [-1, 1] range
    - Inverted index stores sorted (index, value) pairs
    - Dequantization is lossy but preserves similarity ranking
  </constraints>

  <verification>
    - quantize_array produces smaller output
    - dequantize_array recovers approximate vectors
    - Cosine similarity preserved within 5% after roundtrip
    - All embedder types quantize correctly
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/embedding/quantize.rs

use serde::{Serialize, Deserialize};
use thiserror::Error;
use super::{Embedder, TeleologicalArray};
use super::vector::{DenseVector, SparseVector, BinaryVector};

#[derive(Debug, Error)]
pub enum QuantizeError {
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },
    #[error("Codebook missing for {embedder:?}")]
    CodebookMissing { embedder: Embedder },
}

// =============================================================================
// Quantized Types
// =============================================================================

/// PQ-8 quantized dense vector (8-bit codes per subvector)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedDense {
    pub codes: Vec&lt;u8&gt;,
    pub num_subvectors: usize,
    pub original_dim: usize,
}

/// Float8 quantized vector (8-bit scaled values)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedFloat8 {
    pub data: Vec&lt;u8&gt;,
    pub min_val: f32,
    pub max_val: f32,
    pub original_dim: usize,
}

/// Inverted index for sparse vectors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvertedIndex {
    pub indices: Vec&lt;u32&gt;,
    pub quantized_values: Vec&lt;u8&gt;,  // Float8 quantized
    pub dimension: u32,
    pub min_val: f32,
    pub max_val: f32,
}

/// Packed binary vector (just wraps existing BinaryVector)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedBinary {
    pub data: Vec&lt;u64&gt;,
    pub bit_len: usize,
}

/// Complete quantized TeleologicalArray (~11KB)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedArray {
    pub e1_semantic: QuantizedDense,      // PQ-8: 1024D -> 32 codes = 32B
    pub e2_temp_recent: QuantizedFloat8,  // Float8: 512D -> 512B
    pub e3_temp_periodic: QuantizedFloat8,
    pub e4_temp_position: QuantizedFloat8,
    pub e5_causal: QuantizedDense,        // PQ-8: 768D -> 24 codes = 24B
    pub e6_sparse: InvertedIndex,         // Inverted: ~1KB
    pub e7_code: QuantizedDense,          // PQ-8: 1536D -> 48 codes = 48B
    pub e8_emotional: QuantizedFloat8,    // Float8: 384B
    pub e9_hdc: PackedBinary,             // Binary: 128B (1024 bits)
    pub e10_multimodal: QuantizedDense,   // PQ-8: 768D -> 24 codes = 24B
    pub e11_entity: QuantizedFloat8,      // Float8: 384B
    pub e12_late_interact: Vec&lt;QuantizedFloat8&gt;, // Float8 per token: ~2KB
    pub e13_splade: InvertedIndex,        // Inverted: ~1KB
}

// =============================================================================
// Quantization Functions
// =============================================================================

/// Quantize a full TeleologicalArray
pub fn quantize_array(array: &amp;TeleologicalArray) -&gt; Result&lt;QuantizedArray, QuantizeError&gt; {
    Ok(QuantizedArray {
        e1_semantic: quantize_pq8(&amp;array.e1_semantic, 32)?,
        e2_temp_recent: quantize_float8(&amp;array.e2_temp_recent)?,
        e3_temp_periodic: quantize_float8(&amp;array.e3_temp_periodic)?,
        e4_temp_position: quantize_float8(&amp;array.e4_temp_position)?,
        e5_causal: quantize_pq8(&amp;array.e5_causal, 24)?,
        e6_sparse: quantize_inverted(&amp;array.e6_sparse)?,
        e7_code: quantize_pq8(&amp;array.e7_code, 48)?,
        e8_emotional: quantize_float8(&amp;array.e8_emotional)?,
        e9_hdc: pack_binary(&amp;array.e9_hdc),
        e10_multimodal: quantize_pq8(&amp;array.e10_multimodal, 24)?,
        e11_entity: quantize_float8(&amp;array.e11_entity)?,
        e12_late_interact: array.e12_late_interact
            .iter()
            .map(|v| quantize_float8(v))
            .collect::&lt;Result&lt;Vec&lt;_&gt;, _&gt;&gt;()?,
        e13_splade: quantize_inverted(&amp;array.e13_splade)?,
    })
}

/// Dequantize back to TeleologicalArray (lossy)
pub fn dequantize_array(quantized: &amp;QuantizedArray) -&gt; Result&lt;TeleologicalArray, QuantizeError&gt; {
    Ok(TeleologicalArray {
        e1_semantic: dequantize_pq8(&amp;quantized.e1_semantic)?,
        e2_temp_recent: dequantize_float8(&amp;quantized.e2_temp_recent),
        e3_temp_periodic: dequantize_float8(&amp;quantized.e3_temp_periodic),
        e4_temp_position: dequantize_float8(&amp;quantized.e4_temp_position),
        e5_causal: dequantize_pq8(&amp;quantized.e5_causal)?,
        e6_sparse: dequantize_inverted(&amp;quantized.e6_sparse),
        e7_code: dequantize_pq8(&amp;quantized.e7_code)?,
        e8_emotional: dequantize_float8(&amp;quantized.e8_emotional),
        e9_hdc: unpack_binary(&amp;quantized.e9_hdc),
        e10_multimodal: dequantize_pq8(&amp;quantized.e10_multimodal)?,
        e11_entity: dequantize_float8(&amp;quantized.e11_entity),
        e12_late_interact: quantized.e12_late_interact
            .iter()
            .map(|v| dequantize_float8(v))
            .collect(),
        e13_splade: dequantize_inverted(&amp;quantized.e13_splade),
    })
}

// =============================================================================
// PQ-8 Implementation (Product Quantization)
// =============================================================================

fn quantize_pq8(vec: &amp;DenseVector, num_subvectors: usize) -&gt; Result&lt;QuantizedDense, QuantizeError&gt; {
    let data = vec.data();
    let original_dim = data.len();

    if original_dim == 0 {
        return Ok(QuantizedDense {
            codes: Vec::new(),
            num_subvectors,
            original_dim,
        });
    }

    let subvector_size = original_dim / num_subvectors;
    if original_dim % num_subvectors != 0 {
        return Err(QuantizeError::InvalidInput {
            message: format!(
                "Dimension {} not divisible by num_subvectors {}",
                original_dim, num_subvectors
            ),
        });
    }

    // Simplified PQ: quantize each subvector's mean to 8-bit
    // Real PQ would use learned codebooks
    let mut codes = Vec::with_capacity(num_subvectors);
    for i in 0..num_subvectors {
        let start = i * subvector_size;
        let end = start + subvector_size;
        let subvec = &amp;data[start..end];

        // Compute centroid (mean) and quantize to 8-bit
        let mean: f32 = subvec.iter().sum::&lt;f32&gt;() / subvec.len() as f32;
        let code = ((mean.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
        codes.push(code);
    }

    Ok(QuantizedDense {
        codes,
        num_subvectors,
        original_dim,
    })
}

fn dequantize_pq8(quantized: &amp;QuantizedDense) -&gt; Result&lt;DenseVector, QuantizeError&gt; {
    if quantized.original_dim == 0 {
        return Ok(DenseVector::zeros(0));
    }

    let subvector_size = quantized.original_dim / quantized.num_subvectors;
    let mut data = vec![0.0f32; quantized.original_dim];

    for (i, &amp;code) in quantized.codes.iter().enumerate() {
        let mean = (code as f32 / 127.5) - 1.0;
        let start = i * subvector_size;
        let end = start + subvector_size;
        for j in start..end {
            data[j] = mean;
        }
    }

    Ok(DenseVector::new(data))
}

// =============================================================================
// Float8 Implementation
// =============================================================================

fn quantize_float8(vec: &amp;DenseVector) -&gt; Result&lt;QuantizedFloat8, QuantizeError&gt; {
    let data = vec.data();
    let original_dim = data.len();

    if original_dim == 0 {
        return Ok(QuantizedFloat8 {
            data: Vec::new(),
            min_val: 0.0,
            max_val: 0.0,
            original_dim,
        });
    }

    // Find min/max for scaling
    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;

    let quantized: Vec&lt;u8&gt; = if range &gt; 0.0 {
        data.iter()
            .map(|&amp;v| ((v - min_val) / range * 255.0) as u8)
            .collect()
    } else {
        vec![128u8; original_dim]
    };

    Ok(QuantizedFloat8 {
        data: quantized,
        min_val,
        max_val,
        original_dim,
    })
}

fn dequantize_float8(quantized: &amp;QuantizedFloat8) -&gt; DenseVector {
    let range = quantized.max_val - quantized.min_val;

    let data: Vec&lt;f32&gt; = if range &gt; 0.0 {
        quantized.data
            .iter()
            .map(|&amp;v| (v as f32 / 255.0) * range + quantized.min_val)
            .collect()
    } else {
        vec![quantized.min_val; quantized.original_dim]
    };

    DenseVector::new(data)
}

// =============================================================================
// Inverted Index Implementation
// =============================================================================

fn quantize_inverted(vec: &amp;SparseVector) -&gt; Result&lt;InvertedIndex, QuantizeError&gt; {
    let (indices, values) = vec.data();

    if values.is_empty() {
        return Ok(InvertedIndex {
            indices: Vec::new(),
            quantized_values: Vec::new(),
            dimension: vec.dimension(),
            min_val: 0.0,
            max_val: 0.0,
        });
    }

    let min_val = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;

    let quantized_values: Vec&lt;u8&gt; = if range &gt; 0.0 {
        values.iter()
            .map(|&amp;v| ((v - min_val) / range * 255.0) as u8)
            .collect()
    } else {
        vec![128u8; values.len()]
    };

    Ok(InvertedIndex {
        indices: indices.to_vec(),
        quantized_values,
        dimension: vec.dimension(),
        min_val,
        max_val,
    })
}

fn dequantize_inverted(inverted: &amp;InvertedIndex) -&gt; SparseVector {
    let range = inverted.max_val - inverted.min_val;

    let values: Vec&lt;f32&gt; = if range &gt; 0.0 {
        inverted.quantized_values
            .iter()
            .map(|&amp;v| (v as f32 / 255.0) * range + inverted.min_val)
            .collect()
    } else {
        vec![inverted.min_val; inverted.quantized_values.len()]
    };

    SparseVector::new(inverted.indices.clone(), values, inverted.dimension)
        .unwrap_or_else(|_| SparseVector::empty(inverted.dimension))
}

// =============================================================================
// Binary Packing
// =============================================================================

fn pack_binary(vec: &amp;BinaryVector) -&gt; PackedBinary {
    PackedBinary {
        data: vec.data().to_vec(),
        bit_len: vec.bit_len(),
    }
}

fn unpack_binary(packed: &amp;PackedBinary) -&gt; BinaryVector {
    BinaryVector::new(packed.data.clone(), packed.bit_len)
}

impl QuantizedArray {
    /// Estimate the serialized size in bytes
    pub fn estimated_size_bytes(&amp;self) -&gt; usize {
        self.e1_semantic.codes.len()
            + self.e2_temp_recent.data.len()
            + self.e3_temp_periodic.data.len()
            + self.e4_temp_position.data.len()
            + self.e5_causal.codes.len()
            + self.e6_sparse.indices.len() * 4 + self.e6_sparse.quantized_values.len()
            + self.e7_code.codes.len()
            + self.e8_emotional.data.len()
            + self.e9_hdc.data.len() * 8
            + self.e10_multimodal.codes.len()
            + self.e11_entity.data.len()
            + self.e12_late_interact.iter().map(|v| v.data.len()).sum::&lt;usize&gt;()
            + self.e13_splade.indices.len() * 4 + self.e13_splade.quantized_values.len()
            + 200 // Metadata overhead estimate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float8_roundtrip() {
        let original = DenseVector::new(vec![0.1, 0.5, -0.3, 0.9, -0.8]);
        let quantized = quantize_float8(&amp;original).unwrap();
        let recovered = dequantize_float8(&amp;quantized);

        // Check values are approximately preserved
        for (a, b) in original.data().iter().zip(recovered.data().iter()) {
            assert!((a - b).abs() &lt; 0.01);
        }
    }

    #[test]
    fn test_quantized_array_size() {
        let array = TeleologicalArray::new();
        let quantized = quantize_array(&amp;array).unwrap();
        let size = quantized.estimated_size_bytes();

        // Should be much smaller than unquantized (~100KB -&gt; ~11KB)
        assert!(size &lt; 15_000, "Size {} should be under 15KB", size);
    }

    #[test]
    fn test_similarity_preserved() {
        // Create two similar vectors
        let a = DenseVector::new(vec![0.5, 0.3, 0.2, 0.1]);
        let b = DenseVector::new(vec![0.5, 0.3, 0.2, 0.15]);

        let orig_sim = a.cosine_similarity(&amp;b);

        // Quantize and dequantize
        let qa = quantize_float8(&amp;a).unwrap();
        let qb = quantize_float8(&amp;b).unwrap();
        let ra = dequantize_float8(&amp;qa);
        let rb = dequantize_float8(&amp;qb);

        let recovered_sim = ra.cosine_similarity(&amp;rb);

        // Similarity should be preserved within 5%
        assert!((orig_sim - recovered_sim).abs() &lt; 0.05);
    }

    #[test]
    fn test_full_roundtrip() {
        let original = TeleologicalArray::new();
        let quantized = quantize_array(&amp;original).unwrap();
        let recovered = dequantize_array(&amp;quantized).unwrap();

        // Check dimensions preserved
        assert_eq!(recovered.e1_semantic.len(), 1024);
        assert_eq!(recovered.e7_code.len(), 1536);
        assert_eq!(recovered.e9_hdc.bit_len(), 1024);
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/embedding/quantize.rs">Quantizer implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/embedding/mod.rs">Add pub mod quantize and re-exports</file>
</files_to_modify>

<validation_criteria>
  <criterion>QuantizedArray is ~11KB (under 15KB)</criterion>
  <criterion>Float8 roundtrip preserves values within 1%</criterion>
  <criterion>Cosine similarity preserved within 5% after roundtrip</criterion>
  <criterion>All 13 embeddings quantize and dequantize correctly</criterion>
  <criterion>Empty vectors handle gracefully</criterion>
</validation_criteria>

<test_commands>
  <command description="Run quantize tests">cargo test --package context-graph-core quantize</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>

<notes>
  <note category="pq_codebooks">
    Current PQ implementation uses simplified mean-based quantization.
    Production should use learned codebooks trained on representative data.
  </note>
  <note category="compression_ratio">
    Target: 100KB -&gt; 11KB (89% reduction)
    Actual depends on content - sparse vectors may be smaller.
  </note>
</notes>
</task_spec>
```

## Execution Checklist

- [ ] Create quantize.rs in embedding directory
- [ ] Implement QuantizeError enum
- [ ] Implement QuantizedDense (PQ-8)
- [ ] Implement QuantizedFloat8
- [ ] Implement InvertedIndex
- [ ] Implement PackedBinary
- [ ] Implement QuantizedArray struct
- [ ] Implement quantize_array function
- [ ] Implement dequantize_array function
- [ ] Implement individual quantization functions
- [ ] Write unit tests for roundtrip preservation
- [ ] Verify size reduction targets
- [ ] Run tests to verify
- [ ] Phase 2 complete!
