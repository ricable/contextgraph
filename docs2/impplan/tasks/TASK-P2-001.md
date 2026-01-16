# TASK-P2-001: TeleologicalArray Struct

```xml
<task_spec id="TASK-P2-001" version="1.0">
<metadata>
  <title>TeleologicalArray Struct Implementation</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>14</sequence>
  <phase>2</phase>
  <implements>
    <requirement_ref>REQ-P2-01</requirement_ref>
  </implements>
  <depends_on>
    <!-- Foundation type - no dependencies -->
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<context>
Implements the TeleologicalArray struct that holds all 13 embedding vectors.
This is the core data structure for the 13-space embedding system.

Each memory produces one TeleologicalArray containing embeddings from all 13
specialized embedders with their specific dimensions and vector types.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE2-EMBEDDING-13SPACE.md#data_models</file>
  <file purpose="vector_types">crates/context-graph-core/src/embedding/vector.rs (created in P2-002)</file>
</input_context_files>

<prerequisites>
  <check>crates/context-graph-core/src/embedding/ directory exists</check>
  <check>serde crate available for serialization</check>
</prerequisites>

<scope>
  <in_scope>
    - Create TeleologicalArray struct with all 13 fields
    - Implement Default trait (zero vectors)
    - Implement Clone, Debug, Serialize, Deserialize
    - Create Embedder enum with 13 variants
    - Add helper methods for field access by Embedder
    - Add serialization size estimation method
  </in_scope>
  <out_of_scope>
    - Vector type implementations (TASK-P2-002)
    - Validation logic (TASK-P2-004)
    - Quantization (TASK-P2-006)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/embedding/teleological.rs">
      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct TeleologicalArray {
          pub e1_semantic: DenseVector,
          pub e2_temp_recent: DenseVector,
          pub e3_temp_periodic: DenseVector,
          pub e4_temp_position: DenseVector,
          pub e5_causal: DenseVector,
          pub e6_sparse: SparseVector,
          pub e7_code: DenseVector,
          pub e8_emotional: DenseVector,
          pub e9_hdc: BinaryVector,
          pub e10_multimodal: DenseVector,
          pub e11_entity: DenseVector,
          pub e12_late_interact: Vec&lt;DenseVector&gt;,
          pub e13_splade: SparseVector,
      }

      impl TeleologicalArray {
          pub fn new() -> Self;
          pub fn estimated_size_bytes(&amp;self) -> usize;
      }
    </signature>
    <signature file="crates/context-graph-core/src/embedding/mod.rs">
      #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
      #[repr(u8)]
      pub enum Embedder {
          E1Semantic = 0,
          E2TempRecent = 1,
          E3TempPeriodic = 2,
          E4TempPosition = 3,
          E5Causal = 4,
          E6Sparse = 5,
          E7Code = 6,
          E8Emotional = 7,
          E9HDC = 8,
          E10Multimodal = 9,
          E11Entity = 10,
          E12LateInteract = 11,
          E13SPLADE = 12,
      }

      impl Embedder {
          pub fn all() -> [Embedder; 13];
          pub fn index(&amp;self) -> usize;
          pub fn from_index(index: usize) -> Option&lt;Embedder&gt;;
      }
    </signature>
  </signatures>

  <constraints>
    - All 13 fields must be non-null (use zero vectors for empty)
    - Embedder enum variants have explicit discriminants 0-12
    - Struct must be serializable with serde
    - Must implement Default (all zero vectors)
  </constraints>

  <verification>
    - TeleologicalArray can be created and serialized
    - Embedder enum covers all 13 embedders
    - Default implementation produces valid zero vectors
    - Size estimation is reasonably accurate
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/embedding/mod.rs

use serde::{Serialize, Deserialize};

pub mod teleological;
pub mod vector;
pub mod config;
pub mod validator;
pub mod provider;
pub mod quantize;
pub mod error;

pub use teleological::TeleologicalArray;
pub use vector::{DenseVector, SparseVector, BinaryVector};
pub use config::{EmbedderConfig, DistanceMetric, QuantizationConfig};
pub use error::{EmbedderError, ValidationError, QuantizeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Embedder {
    E1Semantic = 0,
    E2TempRecent = 1,
    E3TempPeriodic = 2,
    E4TempPosition = 3,
    E5Causal = 4,
    E6Sparse = 5,
    E7Code = 6,
    E8Emotional = 7,
    E9HDC = 8,
    E10Multimodal = 9,
    E11Entity = 10,
    E12LateInteract = 11,
    E13SPLADE = 12,
}

impl Embedder {
    pub fn all() -> [Embedder; 13] {
        [
            Embedder::E1Semantic,
            Embedder::E2TempRecent,
            Embedder::E3TempPeriodic,
            Embedder::E4TempPosition,
            Embedder::E5Causal,
            Embedder::E6Sparse,
            Embedder::E7Code,
            Embedder::E8Emotional,
            Embedder::E9HDC,
            Embedder::E10Multimodal,
            Embedder::E11Entity,
            Embedder::E12LateInteract,
            Embedder::E13SPLADE,
        ]
    }

    pub fn index(&amp;self) -> usize {
        *self as usize
    }

    pub fn from_index(index: usize) -> Option&lt;Embedder&gt; {
        match index {
            0 => Some(Embedder::E1Semantic),
            1 => Some(Embedder::E2TempRecent),
            2 => Some(Embedder::E3TempPeriodic),
            3 => Some(Embedder::E4TempPosition),
            4 => Some(Embedder::E5Causal),
            5 => Some(Embedder::E6Sparse),
            6 => Some(Embedder::E7Code),
            7 => Some(Embedder::E8Emotional),
            8 => Some(Embedder::E9HDC),
            9 => Some(Embedder::E10Multimodal),
            10 => Some(Embedder::E11Entity),
            11 => Some(Embedder::E12LateInteract),
            12 => Some(Embedder::E13SPLADE),
            _ => None,
        }
    }
}

---
File: crates/context-graph-core/src/embedding/teleological.rs

use serde::{Serialize, Deserialize};
use super::vector::{DenseVector, SparseVector, BinaryVector};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeleologicalArray {
    /// E1: Semantic embedding (1024D)
    pub e1_semantic: DenseVector,
    /// E2: Temporal recency (512D)
    pub e2_temp_recent: DenseVector,
    /// E3: Temporal periodicity (512D)
    pub e3_temp_periodic: DenseVector,
    /// E4: Temporal position (512D)
    pub e4_temp_position: DenseVector,
    /// E5: Causal embedding (768D, asymmetric)
    pub e5_causal: DenseVector,
    /// E6: Sparse BoW/TF-IDF (~30K)
    pub e6_sparse: SparseVector,
    /// E7: Code/technical embedding (1536D)
    pub e7_code: DenseVector,
    /// E8: Emotional/sentiment (384D)
    pub e8_emotional: DenseVector,
    /// E9: Hyperdimensional computing (1024 bits)
    pub e9_hdc: BinaryVector,
    /// E10: Multimodal embedding (768D)
    pub e10_multimodal: DenseVector,
    /// E11: Entity/knowledge graph (384D)
    pub e11_entity: DenseVector,
    /// E12: Late interaction (128D per token, max 512 tokens)
    pub e12_late_interact: Vec&lt;DenseVector&gt;,
    /// E13: SPLADE sparse (~30K)
    pub e13_splade: SparseVector,
}

impl Default for TeleologicalArray {
    fn default() -> Self {
        Self::new()
    }
}

impl TeleologicalArray {
    pub fn new() -> Self {
        Self {
            e1_semantic: DenseVector::zeros(1024),
            e2_temp_recent: DenseVector::zeros(512),
            e3_temp_periodic: DenseVector::zeros(512),
            e4_temp_position: DenseVector::zeros(512),
            e5_causal: DenseVector::zeros(768),
            e6_sparse: SparseVector::empty(30000),
            e7_code: DenseVector::zeros(1536),
            e8_emotional: DenseVector::zeros(384),
            e9_hdc: BinaryVector::zeros(1024),
            e10_multimodal: DenseVector::zeros(768),
            e11_entity: DenseVector::zeros(384),
            e12_late_interact: Vec::new(),
            e13_splade: SparseVector::empty(30000),
        }
    }

    /// Estimate the serialized size in bytes
    pub fn estimated_size_bytes(&amp;self) -> usize {
        let dense_sizes = [
            self.e1_semantic.len() * 4,      // 1024 * 4 = 4096
            self.e2_temp_recent.len() * 4,   // 512 * 4 = 2048
            self.e3_temp_periodic.len() * 4, // 512 * 4 = 2048
            self.e4_temp_position.len() * 4, // 512 * 4 = 2048
            self.e5_causal.len() * 4,        // 768 * 4 = 3072
            self.e7_code.len() * 4,          // 1536 * 4 = 6144
            self.e8_emotional.len() * 4,     // 384 * 4 = 1536
            self.e10_multimodal.len() * 4,   // 768 * 4 = 3072
            self.e11_entity.len() * 4,       // 384 * 4 = 1536
        ];

        let sparse_sizes = [
            self.e6_sparse.byte_size(),
            self.e13_splade.byte_size(),
        ];

        let binary_size = self.e9_hdc.byte_size();

        let late_interact_size = self.e12_late_interact
            .iter()
            .map(|v| v.len() * 4)
            .sum::&lt;usize&gt;();

        dense_sizes.iter().sum::&lt;usize&gt;()
            + sparse_sizes.iter().sum::&lt;usize&gt;()
            + binary_size
            + late_interact_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_teleological_array_default() {
        let array = TeleologicalArray::new();
        assert_eq!(array.e1_semantic.len(), 1024);
        assert_eq!(array.e2_temp_recent.len(), 512);
        assert_eq!(array.e7_code.len(), 1536);
        assert!(array.e12_late_interact.is_empty());
    }

    #[test]
    fn test_size_estimation() {
        let array = TeleologicalArray::new();
        let size = array.estimated_size_bytes();
        // Should be roughly 25-50KB unquantized for empty content
        assert!(size > 20_000);
        assert!(size < 100_000);
    }

    #[test]
    fn test_serialization() {
        let array = TeleologicalArray::new();
        let serialized = bincode::serialize(&amp;array).unwrap();
        let deserialized: TeleologicalArray = bincode::deserialize(&amp;serialized).unwrap();
        assert_eq!(deserialized.e1_semantic.len(), 1024);
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/embedding/teleological.rs">TeleologicalArray struct</file>
  <file path="crates/context-graph-core/src/embedding/mod.rs">Module with Embedder enum</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/lib.rs">Add pub mod embedding</file>
  <file path="crates/context-graph-core/Cargo.toml">Add serde, bincode dependencies if needed</file>
</files_to_modify>

<validation_criteria>
  <criterion>TeleologicalArray compiles with all 13 fields</criterion>
  <criterion>Default creates valid zero vectors</criterion>
  <criterion>Embedder enum has all 13 variants with correct indices</criterion>
  <criterion>Struct serializes/deserializes with bincode</criterion>
  <criterion>Size estimation returns reasonable value</criterion>
</validation_criteria>

<test_commands>
  <command description="Run teleological tests">cargo test --package context-graph-core teleological</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create embedding directory in context-graph-core/src
- [ ] Create mod.rs with Embedder enum
- [ ] Create teleological.rs with TeleologicalArray struct
- [ ] Implement Default trait
- [ ] Add estimated_size_bytes method
- [ ] Update lib.rs to export embedding module
- [ ] Write unit tests
- [ ] Run tests to verify
- [ ] Proceed to TASK-P2-002
