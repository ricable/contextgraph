# TASK-EMB-004: Create Quantization Structs

<task_spec id="TASK-EMB-004" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Title** | Create Quantization Structs |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 4 |
| **Implements** | REQ-EMB-005 |
| **Depends On** | TASK-EMB-001 |
| **Estimated Complexity** | medium |
| **Parallel Group** | B |

---

## Context

Constitution v4.0.0 specifies quantization strategies per embedder:

| Strategy | Compression | Embedders |
|----------|-------------|-----------|
| PQ-8 | 32x | E1, E5, E7, E10 |
| Float8 | 4x | E2, E3, E4, E8, E11 |
| Binary | 32x | E9 |
| Sparse | native | E6, E13 |
| TokenPruning | ~50% | E12 |

This task creates the foundational types. Logic Layer (TASK-EMB-016/017/018) implements algorithms.

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Constitution | `docs2/constitution.yaml` section `embeddings.quantization` |
| Dimension constants | `crates/context-graph-core/src/config/constants.rs` (TASK-EMB-001) |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-003-quantization.md` |

---

## Prerequisites

- [ ] TASK-EMB-001 completed (dimension constants exist)
- [ ] Read Constitution `embeddings.quantization` section
- [ ] Read Constitution `embeddings.quantization_by_embedder` section

---

## Scope

### In Scope

- Define `QuantizationStrategy` enum
- Define `QuantizedEmbedding` variants for each strategy
- Define `QuantizationConfig` per embedder
- Define `QuantizationError` enum
- Create mapping from ModelId to strategy

### Out of Scope

- Quantization algorithms (TASK-EMB-016/017/018)
- Dequantization (TASK-EMB-020)
- Storage integration (TASK-EMB-022)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-embeddings/src/quantization/types.rs

use crate::config::constants::*;

/// Quantization strategy per Constitution v4.0.0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationStrategy {
    /// Product Quantization with 8 sub-vectors
    /// Compression: 32x, Recall impact: <5%
    /// Used by: E1, E5, E7, E10
    PQ8,

    /// 8-bit floating point
    /// Compression: 4x, Recall impact: <0.3%
    /// Used by: E2, E3, E4, E8, E11
    Float8,

    /// Binary quantization (1-bit per dimension)
    /// Compression: 32x, Recall impact: 5-10%
    /// Used by: E9 (HDC)
    Binary,

    /// Native sparse format (no quantization)
    /// Compression: depends on sparsity
    /// Used by: E6 (SPLADE-like), E13 (SPLADE)
    Sparse,

    /// Token pruning for late interaction
    /// Compression: ~50%, Recall impact: <2%
    /// Used by: E12 (ColBERT-style)
    TokenPruning,

    /// No quantization (full precision)
    /// Used for: testing, golden references
    None,
}

impl QuantizationStrategy {
    /// Get compression ratio for this strategy.
    pub fn compression_ratio(&self) -> f32 {
        match self {
            Self::PQ8 => 32.0,
            Self::Float8 => 4.0,
            Self::Binary => 32.0,
            Self::Sparse => 10.0,  // Approximate, depends on sparsity
            Self::TokenPruning => 2.0,
            Self::None => 1.0,
        }
    }

    /// Get expected recall impact (percentage loss).
    pub fn recall_impact(&self) -> f32 {
        match self {
            Self::PQ8 => 0.05,
            Self::Float8 => 0.003,
            Self::Binary => 0.075,
            Self::Sparse => 0.0,
            Self::TokenPruning => 0.02,
            Self::None => 0.0,
        }
    }
}

/// Quantized embedding storage variants.
#[derive(Debug, Clone)]
pub enum QuantizedEmbedding {
    /// PQ-8: Sub-vector centroids + codes
    PQ8 {
        /// Centroid codes per sub-vector, shape: [num_subvectors]
        codes: Vec<u8>,
        /// Reference to codebook (shared across embeddings)
        codebook_id: u32,
    },

    /// Float8: E5M2 or E4M3 format
    Float8 {
        /// 8-bit float values
        data: Vec<u8>,
        /// Exponent bits (4 or 5)
        exponent_bits: u8,
    },

    /// Binary: 1-bit per dimension
    Binary {
        /// Packed bits, each u64 holds 64 dimensions
        bits: Vec<u64>,
        /// Original dimension count
        original_dim: usize,
    },

    /// Sparse: indices and values
    Sparse {
        /// Active dimension indices (sorted)
        indices: Vec<u32>,
        /// Values at active indices
        values: Vec<f32>,
    },

    /// Token-pruned: subset of token embeddings
    TokenPruned {
        /// Kept token indices
        token_indices: Vec<u16>,
        /// Per-token embedding data
        embeddings: Vec<Vec<f32>>,
    },

    /// Full precision (no quantization)
    Full {
        data: Vec<f32>,
    },
}

impl QuantizedEmbedding {
    /// Get byte size of this quantized embedding.
    pub fn byte_size(&self) -> usize {
        match self {
            Self::PQ8 { codes, .. } => codes.len() + 4,
            Self::Float8 { data, .. } => data.len() + 1,
            Self::Binary { bits, .. } => bits.len() * 8 + 8,
            Self::Sparse { indices, values } => indices.len() * 4 + values.len() * 4,
            Self::TokenPruned { token_indices, embeddings } => {
                token_indices.len() * 2 + embeddings.iter().map(|e| e.len() * 4).sum::<usize>()
            }
            Self::Full { data } => data.len() * 4,
        }
    }

    /// Get the quantization strategy used.
    pub fn strategy(&self) -> QuantizationStrategy {
        match self {
            Self::PQ8 { .. } => QuantizationStrategy::PQ8,
            Self::Float8 { .. } => QuantizationStrategy::Float8,
            Self::Binary { .. } => QuantizationStrategy::Binary,
            Self::Sparse { .. } => QuantizationStrategy::Sparse,
            Self::TokenPruned { .. } => QuantizationStrategy::TokenPruning,
            Self::Full { .. } => QuantizationStrategy::None,
        }
    }
}

/// Configuration for quantizing a specific embedder's output.
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Which strategy to use
    pub strategy: QuantizationStrategy,
    /// Input dimension before quantization
    pub input_dim: usize,
    /// PQ-8 specific: number of sub-vectors
    pub pq_num_subvectors: Option<usize>,
    /// PQ-8 specific: bits per code
    pub pq_bits: Option<u8>,
    /// Float8 specific: exponent bits
    pub float8_exponent_bits: Option<u8>,
    /// Token pruning specific: keep ratio
    pub token_keep_ratio: Option<f32>,
}

impl QuantizationConfig {
    /// Create PQ-8 config for an embedder.
    pub fn pq8(input_dim: usize) -> Self {
        Self {
            strategy: QuantizationStrategy::PQ8,
            input_dim,
            pq_num_subvectors: Some(input_dim / 128),  // 128D per subvector
            pq_bits: Some(8),
            float8_exponent_bits: None,
            token_keep_ratio: None,
        }
    }

    /// Create Float8 config.
    pub fn float8(input_dim: usize) -> Self {
        Self {
            strategy: QuantizationStrategy::Float8,
            input_dim,
            pq_num_subvectors: None,
            pq_bits: None,
            float8_exponent_bits: Some(5),  // E5M2 format
            token_keep_ratio: None,
        }
    }

    /// Create Binary config.
    pub fn binary(input_dim: usize) -> Self {
        Self {
            strategy: QuantizationStrategy::Binary,
            input_dim,
            pq_num_subvectors: None,
            pq_bits: None,
            float8_exponent_bits: None,
            token_keep_ratio: None,
        }
    }

    /// Create Sparse config.
    pub fn sparse() -> Self {
        Self {
            strategy: QuantizationStrategy::Sparse,
            input_dim: 0,  // Variable for sparse
            pq_num_subvectors: None,
            pq_bits: None,
            float8_exponent_bits: None,
            token_keep_ratio: None,
        }
    }

    /// Create TokenPruning config.
    pub fn token_pruning(keep_ratio: f32) -> Self {
        Self {
            strategy: QuantizationStrategy::TokenPruning,
            input_dim: LATE_INTERACTION_DIM,
            pq_num_subvectors: None,
            pq_bits: None,
            float8_exponent_bits: None,
            token_keep_ratio: Some(keep_ratio),
        }
    }
}

/// Embedder to quantization strategy mapping per Constitution.
pub fn get_embedder_strategy(embedder_id: usize) -> QuantizationStrategy {
    match embedder_id {
        1 | 5 | 7 | 10 => QuantizationStrategy::PQ8,      // E1, E5, E7, E10
        2 | 3 | 4 | 8 | 11 => QuantizationStrategy::Float8, // E2-E4, E8, E11
        9 => QuantizationStrategy::Binary,                  // E9 HDC
        6 | 13 => QuantizationStrategy::Sparse,             // E6, E13 SPLADE
        12 => QuantizationStrategy::TokenPruning,           // E12 ColBERT
        _ => QuantizationStrategy::None,
    }
}
```

### Constraints

- Strategy mapping MUST match Constitution `embeddings.quantization_by_embedder`
- Compression ratios MUST match Constitution targets
- All types MUST be serializable for storage

### Verification

- All embedder IDs map to correct strategy
- Compression ratios match Constitution
- Byte size calculations are accurate

---

## Files to Create

| File Path | Description |
|-----------|-------------|
| `crates/context-graph-embeddings/src/quantization/mod.rs` | Module declaration |
| `crates/context-graph-embeddings/src/quantization/types.rs` | Type definitions |

## Files to Modify

| File Path | Change |
|-----------|--------|
| `crates/context-graph-embeddings/src/lib.rs` | Add `pub mod quantization;` |

---

## Validation Criteria

- [ ] `QuantizationStrategy` enum with 6 variants
- [ ] `QuantizedEmbedding` enum with 6 variants
- [ ] `QuantizationConfig` struct with factory methods
- [ ] `get_embedder_strategy()` function matches Constitution
- [ ] `cargo check -p context-graph-embeddings` passes

---

## Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
cargo test -p context-graph-embeddings quantization:: -- --nocapture
```

---

## Pseudo Code

```
types.rs:
  Define QuantizationStrategy enum (PQ8, Float8, Binary, Sparse, TokenPruning, None)
    compression_ratio() method
    recall_impact() method

  Define QuantizedEmbedding enum with storage variants
    byte_size() method
    strategy() method

  Define QuantizationConfig struct
    Factory methods: pq8(), float8(), binary(), sparse(), token_pruning()

  Define get_embedder_strategy(id) mapping function

mod.rs:
  pub mod types;
  pub use types::*;
```

</task_spec>
