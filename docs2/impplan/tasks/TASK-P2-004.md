# TASK-P2-004: DimensionValidator

```xml
<task_spec id="TASK-P2-004" version="1.0">
<metadata>
  <title>DimensionValidator Implementation</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>17</sequence>
  <phase>2</phase>
  <implements>
    <requirement_ref>REQ-P2-04</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P2-002</task_ref>
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<context>
Implements the DimensionValidator that validates TeleologicalArray dimensions
against expected values from the EmbedderConfig registry.

Validation is critical for fail-fast behavior - any dimension mismatch indicates
a bug in embedder implementation and must fail immediately.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE2-EMBEDDING-13SPACE.md#component_contracts</file>
  <file purpose="config_registry">crates/context-graph-core/src/embedding/config.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P2-002 complete (vector types exist)</check>
  <check>TASK-P2-003 complete (EmbedderConfig registry exists)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create ValidationError enum
    - Implement validate_teleological_array function
    - Implement validate_single_embedding helper
    - Check all 13 embeddings have correct dimensions
    - Check sparse vectors have reasonable nnz
    - Return detailed error on any validation failure
  </in_scope>
  <out_of_scope>
    - Content validation (empty content is valid)
    - Semantic validation (embeddings can be zero)
    - Runtime dimension changes
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/embedding/validator.rs">
      #[derive(Debug, Error)]
      pub enum ValidationError {
          #[error("Dimension mismatch for {embedder:?}: expected {expected}, got {actual}")]
          DimensionMismatch { embedder: Embedder, expected: usize, actual: usize },
          #[error("Empty embedding for {embedder:?}")]
          EmptyEmbedding { embedder: Embedder },
          #[error("Sparse dimension too large for {embedder:?}: {actual} > {max}")]
          SparseDimensionTooLarge { embedder: Embedder, actual: u32, max: u32 },
      }

      pub fn validate_teleological_array(array: &amp;TeleologicalArray) -> Result&lt;(), ValidationError&gt;;
      pub fn validate_dense_dimension(embedder: Embedder, vec: &amp;DenseVector) -> Result&lt;(), ValidationError&gt;;
      pub fn validate_sparse_dimension(embedder: Embedder, vec: &amp;SparseVector) -> Result&lt;(), ValidationError&gt;;
      pub fn validate_binary_dimension(embedder: Embedder, vec: &amp;BinaryVector) -> Result&lt;(), ValidationError&gt;;
    </signature>
  </signatures>

  <constraints>
    - Validation fails fast on first error
    - All expected dimensions come from EmbedderConfig
    - Sparse dimension validation allows up to 2x config dimension
    - Empty vectors are valid (content might be empty)
    - Binary vectors checked by bit_len not byte_len
  </constraints>

  <verification>
    - Valid TeleologicalArray passes validation
    - Wrong dimension fails with DimensionMismatch
    - Sparse vector exceeding dimension fails
    - All 13 embeddings validated
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/embedding/validator.rs

use thiserror::Error;
use super::{Embedder, TeleologicalArray};
use super::vector::{DenseVector, SparseVector, BinaryVector};
use super::config::{get_dimension, is_sparse};

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Dimension mismatch for {embedder:?}: expected {expected}, got {actual}")]
    DimensionMismatch {
        embedder: Embedder,
        expected: usize,
        actual: usize,
    },
    #[error("Empty embedding for {embedder:?}")]
    EmptyEmbedding {
        embedder: Embedder,
    },
    #[error("Sparse dimension too large for {embedder:?}: {actual} > {max}")]
    SparseDimensionTooLarge {
        embedder: Embedder,
        actual: u32,
        max: u32,
    },
    #[error("Late interaction token count exceeds maximum: {actual} > {max}")]
    TooManyTokens {
        actual: usize,
        max: usize,
    },
}

/// Maximum tokens for late interaction (E12)
const MAX_LATE_INTERACT_TOKENS: usize = 512;

/// Validate a complete TeleologicalArray
pub fn validate_teleological_array(array: &amp;TeleologicalArray) -> Result&lt;(), ValidationError&gt; {
    // E1: Semantic (1024D)
    validate_dense_dimension(Embedder::E1Semantic, &amp;array.e1_semantic)?;

    // E2: Temporal Recent (512D)
    validate_dense_dimension(Embedder::E2TempRecent, &amp;array.e2_temp_recent)?;

    // E3: Temporal Periodic (512D)
    validate_dense_dimension(Embedder::E3TempPeriodic, &amp;array.e3_temp_periodic)?;

    // E4: Temporal Position (512D)
    validate_dense_dimension(Embedder::E4TempPosition, &amp;array.e4_temp_position)?;

    // E5: Causal (768D)
    validate_dense_dimension(Embedder::E5Causal, &amp;array.e5_causal)?;

    // E6: Sparse (~30K)
    validate_sparse_dimension(Embedder::E6Sparse, &amp;array.e6_sparse)?;

    // E7: Code (1536D)
    validate_dense_dimension(Embedder::E7Code, &amp;array.e7_code)?;

    // E8: Emotional (384D)
    validate_dense_dimension(Embedder::E8Emotional, &amp;array.e8_emotional)?;

    // E9: HDC (1024 bits)
    validate_binary_dimension(Embedder::E9HDC, &amp;array.e9_hdc)?;

    // E10: Multimodal (768D)
    validate_dense_dimension(Embedder::E10Multimodal, &amp;array.e10_multimodal)?;

    // E11: Entity (384D)
    validate_dense_dimension(Embedder::E11Entity, &amp;array.e11_entity)?;

    // E12: Late Interaction (128D per token, max 512 tokens)
    validate_late_interaction(&amp;array.e12_late_interact)?;

    // E13: SPLADE (~30K)
    validate_sparse_dimension(Embedder::E13SPLADE, &amp;array.e13_splade)?;

    Ok(())
}

/// Validate a dense vector dimension
pub fn validate_dense_dimension(
    embedder: Embedder,
    vec: &amp;DenseVector,
) -> Result&lt;(), ValidationError&gt; {
    let expected = get_dimension(embedder);
    let actual = vec.len();

    if actual != expected {
        return Err(ValidationError::DimensionMismatch {
            embedder,
            expected,
            actual,
        });
    }

    Ok(())
}

/// Validate a sparse vector dimension
pub fn validate_sparse_dimension(
    embedder: Embedder,
    vec: &amp;SparseVector,
) -> Result&lt;(), ValidationError&gt; {
    let config_dim = get_dimension(embedder) as u32;
    // Allow up to 2x config dimension for sparse vectors (vocabulary can vary)
    let max_dim = config_dim * 2;
    let actual_dim = vec.dimension();

    if actual_dim > max_dim {
        return Err(ValidationError::SparseDimensionTooLarge {
            embedder,
            actual: actual_dim,
            max: max_dim,
        });
    }

    Ok(())
}

/// Validate a binary vector dimension (bit length)
pub fn validate_binary_dimension(
    embedder: Embedder,
    vec: &amp;BinaryVector,
) -> Result&lt;(), ValidationError&gt; {
    let expected = get_dimension(embedder);
    let actual = vec.bit_len();

    if actual != expected {
        return Err(ValidationError::DimensionMismatch {
            embedder,
            expected,
            actual,
        });
    }

    Ok(())
}

/// Validate late interaction embeddings (E12)
fn validate_late_interaction(vecs: &amp;[DenseVector]) -> Result&lt;(), ValidationError&gt; {
    // Check token count
    if vecs.len() > MAX_LATE_INTERACT_TOKENS {
        return Err(ValidationError::TooManyTokens {
            actual: vecs.len(),
            max: MAX_LATE_INTERACT_TOKENS,
        });
    }

    // Check each token embedding dimension
    let expected_dim = get_dimension(Embedder::E12LateInteract);
    for (i, vec) in vecs.iter().enumerate() {
        if vec.len() != expected_dim {
            return Err(ValidationError::DimensionMismatch {
                embedder: Embedder::E12LateInteract,
                expected: expected_dim,
                actual: vec.len(),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_valid_array() {
        let array = TeleologicalArray::new();
        let result = validate_teleological_array(&amp;array);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_wrong_dimension() {
        let mut array = TeleologicalArray::new();
        array.e1_semantic = DenseVector::zeros(512); // Wrong: should be 1024

        let result = validate_teleological_array(&amp;array);
        assert!(matches!(
            result,
            Err(ValidationError::DimensionMismatch {
                embedder: Embedder::E1Semantic,
                expected: 1024,
                actual: 512
            })
        ));
    }

    #[test]
    fn test_validate_sparse_too_large() {
        let mut array = TeleologicalArray::new();
        // Create sparse vector with dimension way too large
        array.e6_sparse = SparseVector::empty(100_000);

        let result = validate_teleological_array(&amp;array);
        assert!(matches!(
            result,
            Err(ValidationError::SparseDimensionTooLarge { .. })
        ));
    }

    #[test]
    fn test_validate_late_interact_too_many_tokens() {
        let mut array = TeleologicalArray::new();
        // Add too many tokens
        for _ in 0..600 {
            array.e12_late_interact.push(DenseVector::zeros(128));
        }

        let result = validate_teleological_array(&amp;array);
        assert!(matches!(
            result,
            Err(ValidationError::TooManyTokens { actual: 600, max: 512 })
        ));
    }

    #[test]
    fn test_validate_binary_dimension() {
        let vec = BinaryVector::zeros(1024);
        assert!(validate_binary_dimension(Embedder::E9HDC, &amp;vec).is_ok());

        let wrong_vec = BinaryVector::zeros(512);
        assert!(validate_binary_dimension(Embedder::E9HDC, &amp;wrong_vec).is_err());
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/embedding/validator.rs">DimensionValidator implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/embedding/mod.rs">Add pub mod validator and re-exports</file>
</files_to_modify>

<validation_criteria>
  <criterion>validate_teleological_array checks all 13 embeddings</criterion>
  <criterion>Dimension mismatch returns detailed error</criterion>
  <criterion>Sparse dimension allows up to 2x configured max</criterion>
  <criterion>Late interaction validates per-token dimension and count</criterion>
  <criterion>Valid TeleologicalArray::new() passes validation</criterion>
</validation_criteria>

<test_commands>
  <command description="Run validator tests">cargo test --package context-graph-core validator</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create validator.rs in embedding directory
- [ ] Implement ValidationError enum
- [ ] Implement validate_teleological_array function
- [ ] Implement validate_dense_dimension helper
- [ ] Implement validate_sparse_dimension helper
- [ ] Implement validate_binary_dimension helper
- [ ] Implement validate_late_interaction helper
- [ ] Write comprehensive unit tests
- [ ] Run tests to verify
- [ ] Proceed to TASK-P2-005
