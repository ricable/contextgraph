# TASK-EMB-005: Create Storage Types

<task_spec id="TASK-EMB-005" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Title** | Create Storage Types |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 5 |
| **Implements** | REQ-EMB-006 |
| **Depends On** | TASK-EMB-004 |
| **Estimated Complexity** | medium |
| **Parallel Group** | C |

---

## Context

Constitution v4.0.0 `storage` section specifies:
- Layer 1: Primary storage (RocksDB dev, ScyllaDB prod)
- Layer 2a-f: Various indexes (sparse, matryoshka, per-embedder, purpose, goal, late-interaction)
- Total storage per fingerprint: ~17KB quantized

This task creates foundational types for the storage module. Implementation is in Surface Layer (TASK-EMB-022).

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Constitution | `docs2/constitution.yaml` section `storage` |
| Quantization types | `crates/context-graph-embeddings/src/quantization/types.rs` (TASK-EMB-004) |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-004-storage-module.md` |

---

## Prerequisites

- [ ] TASK-EMB-004 completed (QuantizedEmbedding types exist)
- [ ] Read Constitution `storage` section
- [ ] Understand 5-stage retrieval pipeline

---

## Scope

### In Scope

- `TeleologicalFingerprint` complete struct (per Constitution)
- `EmbedderStorage` trait for per-embedder indexes
- `StorageBackend` enum (RocksDB/ScyllaDB)
- `StorageConfig` with layer configurations
- Index query types for each layer

### Out of Scope

- RocksDB/ScyllaDB implementation (TASK-EMB-022)
- Index building (TASK-EMB-022)
- Search algorithms (TASK-EMB-023)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-core/src/storage/types.rs

use uuid::Uuid;
use chrono::{DateTime, Utc};
use crate::quantization::{QuantizedEmbedding, QuantizationStrategy};
use crate::config::constants::*;

/// Complete teleological fingerprint for a memory.
/// Constitution: `storage.layer1_primary.schema`
///
/// This is the COMPLETE representation of a memory's semantic identity
/// across all 13 embedding spaces.
#[derive(Debug, Clone)]
pub struct TeleologicalFingerprint {
    /// Unique memory identifier
    pub id: Uuid,

    /// All 13 embeddings (quantized per Constitution)
    /// Index 0-12 = E1-E13
    pub embeddings: [QuantizedEmbedding; EMBEDDER_COUNT],

    /// Optional raw embeddings (cold storage, full precision)
    /// Used for recomputation if quantization strategy changes
    pub embeddings_raw: Option<[Vec<f32>; EMBEDDER_COUNT]>,

    /// 13D teleological purpose vector
    /// Each dimension = alignment of that embedder to North Star
    pub purpose_vector: [f32; PURPOSE_VECTOR_DIM],

    /// Per-embedder Johari quadrant classification
    /// Values: Open, Blind, Hidden, Unknown
    pub johari_quadrants: [JohariQuadrant; EMBEDDER_COUNT],

    /// Confidence in Johari classification
    pub johari_confidence: [f32; EMBEDDER_COUNT],

    /// Overall alignment to North Star goal
    pub north_star_alignment: f32,

    /// Index of embedder with highest alignment
    pub dominant_embedder: u8,

    /// Kuramoto synchronization score across spaces
    pub coherence_score: f32,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last access timestamp
    pub last_accessed: DateTime<Utc>,
}

impl TeleologicalFingerprint {
    /// Calculate total byte size of this fingerprint.
    pub fn byte_size(&self) -> usize {
        let embedding_size: usize = self.embeddings.iter()
            .map(|e| e.byte_size())
            .sum();

        let raw_size = self.embeddings_raw.as_ref()
            .map(|raw| raw.iter().map(|v| v.len() * 4).sum())
            .unwrap_or(0);

        // Fixed fields: id(16) + purpose(52) + johari(13) + confidence(52) +
        //               alignment(4) + dominant(1) + coherence(4) + timestamps(16)
        let fixed_size = 16 + 52 + 13 + 52 + 4 + 1 + 4 + 16;

        embedding_size + raw_size + fixed_size
    }

    /// Check if fingerprint is under Constitution size budget (~17KB).
    pub fn within_budget(&self) -> bool {
        self.byte_size() <= 17 * 1024
    }

    /// Get quantization strategies used.
    pub fn quantization_strategies(&self) -> [QuantizationStrategy; EMBEDDER_COUNT] {
        let mut strategies = [QuantizationStrategy::None; EMBEDDER_COUNT];
        for (i, emb) in self.embeddings.iter().enumerate() {
            strategies[i] = emb.strategy();
        }
        strategies
    }
}

/// Johari window quadrant classification.
/// Constitution: `delta_sc_computation.johari_classification`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum JohariQuadrant {
    /// ΔS ≤ 0.5, ΔC > 0.5 — Well-understood
    Open = 0,
    /// ΔS > 0.5, ΔC ≤ 0.5 — Discovery opportunity
    Blind = 1,
    /// ΔS ≤ 0.5, ΔC ≤ 0.5 — Dormant
    Hidden = 2,
    /// ΔS > 0.5, ΔC > 0.5 — Frontier
    Unknown = 3,
}

impl JohariQuadrant {
    /// Classify based on ΔS and ΔC values.
    pub fn classify(delta_s: f32, delta_c: f32) -> Self {
        match (delta_s > 0.5, delta_c > 0.5) {
            (false, true) => Self::Open,
            (true, false) => Self::Blind,
            (false, false) => Self::Hidden,
            (true, true) => Self::Unknown,
        }
    }

    /// Get learning priority for this quadrant.
    pub fn learning_priority(&self) -> f32 {
        match self {
            Self::Unknown => 1.0,   // Frontier - highest priority
            Self::Blind => 0.8,     // Discovery opportunity
            Self::Open => 0.3,      // Already understood
            Self::Hidden => 0.2,    // Dormant
        }
    }
}

/// Storage backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageBackend {
    /// Development: RocksDB local
    RocksDB,
    /// Production: ScyllaDB distributed
    ScyllaDB,
}

impl StorageBackend {
    /// Select backend based on environment.
    pub fn from_env() -> Self {
        if std::env::var("PRODUCTION").is_ok() {
            Self::ScyllaDB
        } else {
            Self::RocksDB
        }
    }
}

/// Index layer configuration per Constitution.
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Layer 2a: SPLADE sparse inverted index
    pub sparse_index: SparseIndexConfig,
    /// Layer 2b: Matryoshka 128D HNSW
    pub matryoshka_index: HnswConfig,
    /// Layer 2c: Per-embedder full dimension HNSW
    pub per_embedder_indexes: [HnswConfig; EMBEDDER_COUNT],
    /// Layer 2d: Purpose vector 13D HNSW
    pub purpose_index: HnswConfig,
    /// Layer 2f: Late interaction token index
    pub late_interaction_index: HnswConfig,
}

/// HNSW index configuration.
#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    /// M parameter: max connections per layer
    pub m: usize,
    /// efConstruction: size of dynamic candidate list
    pub ef_construction: usize,
    /// efSearch: size of search candidate list
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
        }
    }
}

impl HnswConfig {
    /// High-recall configuration for Matryoshka fast search.
    pub fn matryoshka() -> Self {
        Self {
            m: 32,
            ef_construction: 256,
            ef_search: 128,
        }
    }

    /// Standard configuration for per-embedder indexes.
    pub fn per_embedder() -> Self {
        Self::default()
    }
}

/// Sparse inverted index configuration.
#[derive(Debug, Clone)]
pub struct SparseIndexConfig {
    /// Vocabulary size (SPLADE)
    pub vocab_size: usize,
    /// Minimum value to index
    pub min_value_threshold: f32,
}

impl Default for SparseIndexConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            min_value_threshold: 0.0,
        }
    }
}

/// Query for storage operations.
#[derive(Debug, Clone)]
pub struct StorageQuery {
    /// Embedding space(s) to search
    pub embedder_ids: Vec<usize>,
    /// Query vector (varies by search type)
    pub vector: Vec<f32>,
    /// Maximum results
    pub k: usize,
    /// Minimum alignment threshold
    pub min_alignment: Option<f32>,
    /// Filter by Johari quadrant
    pub johari_filter: Option<Vec<JohariQuadrant>>,
}
```

### Constraints

- `TeleologicalFingerprint` MUST store all 13 embeddings
- Byte size MUST be calculable for budget enforcement
- `JohariQuadrant` classification MUST match Constitution formula
- Index configs MUST match Constitution `storage.layer2*` sections

### Verification

- Fingerprint byte_size() calculation is accurate
- JohariQuadrant::classify() matches Constitution
- Default HNSW configs match Constitution

---

## Files to Create

| File Path | Description |
|-----------|-------------|
| `crates/context-graph-core/src/storage/types.rs` | Storage type definitions |
| `crates/context-graph-core/src/storage/mod.rs` | Module declaration |

## Files to Modify

| File Path | Change |
|-----------|--------|
| `crates/context-graph-core/src/lib.rs` | Add `pub mod storage;` |

---

## Validation Criteria

- [ ] `TeleologicalFingerprint` struct with all fields per Constitution
- [ ] `JohariQuadrant` enum with classify() method
- [ ] `StorageBackend` enum with env detection
- [ ] `IndexConfig` with all 6 index layers
- [ ] `HnswConfig` and `SparseIndexConfig` with defaults
- [ ] `cargo check -p context-graph-core` passes

---

## Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-core
cargo test -p context-graph-core storage::types:: -- --nocapture
```

---

## Pseudo Code

```
types.rs:
  Define TeleologicalFingerprint struct
    All 13 embeddings as quantized
    Purpose vector [f32; 13]
    Johari quadrants [JohariQuadrant; 13]
    Alignment scores
    Timestamps
    Methods: byte_size(), within_budget(), quantization_strategies()

  Define JohariQuadrant enum (Open, Blind, Hidden, Unknown)
    classify(delta_s, delta_c) factory
    learning_priority() method

  Define StorageBackend enum (RocksDB, ScyllaDB)
    from_env() factory

  Define IndexConfig struct with all layer configs
  Define HnswConfig with M, ef_construction, ef_search
  Define SparseIndexConfig with vocab_size, threshold
  Define StorageQuery for retrieval
```

</task_spec>
