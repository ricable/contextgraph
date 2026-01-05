# Task: TASK-F007 - Modify EmbeddingProvider Trait for Multi-Array Output

## Metadata
- **ID**: TASK-F007
- **Layer**: Foundation
- **Priority**: P0 (Critical Path)
- **Estimated Effort**: M (Medium)
- **Dependencies**: TASK-F001 (SemanticFingerprint - COMPLETE), TASK-F006 (Remove Fusion - COMPLETE)
- **Traces To**: TS-301, FR-101, FR-102, FR-104, FR-602

---

## CURRENT CODEBASE STATE (Audited 2025-01-05)

### What Already Exists (DO NOT RECREATE)

**SemanticFingerprint** - COMPLETE in `crates/context-graph-core/src/types/fingerprint/semantic.rs`:
```rust
pub struct SemanticFingerprint {
    pub e1_semantic: [f32; E1_DIM],           // 1024D - Nomic Embed v1.5
    pub e2_temporal_recent: [f32; E2_DIM],    // 512D - Jina Embeddings v3
    pub e3_temporal_historical: [f32; E3_DIM], // 512D - UAE-Large-V1
    pub e4_causal: [f32; E4_DIM],             // 512D - BGE-M3
    pub e5_structural: [f32; E5_DIM],         // 768D - Instructor-XL
    pub e6_sparse: SparseVector,              // SPLADE - variable sparse
    pub e7_counterfactual: [f32; E7_DIM],     // 1536D - OpenAI 3-small
    pub e8_analogical: [f32; E8_DIM],         // 384D - MiniLM-L6-v2
    pub e9_emergent: [f32; E9_DIM],           // 1024D - E5-Large-v2
    pub e10_multimodal: [f32; E10_DIM],       // 768D - CLIP ViT-L/14
    pub e11_code: [f32; E11_DIM],             // 384D - CodeBERT-base
    pub e12_colbert_tokens: Vec<[f32; E12_TOKEN_DIM]>, // 128D per token
    pub e13_splade: SparseVector,             // SPLADE v3 - Stage 1 recall
}
```

**SparseVector** - COMPLETE in `crates/context-graph-core/src/types/fingerprint/sparse.rs`:
```rust
pub const SPARSE_VOCAB_SIZE: usize = 30_522;
pub const MAX_SPARSE_ACTIVE: usize = 128;

pub struct SparseVector {
    indices: Vec<u16>,  // Active vocabulary indices
    values: Vec<f32>,   // Corresponding weights
}
```

**Constants** - COMPLETE in `crates/context-graph-core/src/types/fingerprint/semantic.rs`:
```rust
pub const NUM_EMBEDDERS: usize = 13;
pub const E1_DIM: usize = 1024;
pub const E2_DIM: usize = 512;
// ... all dimensions defined
pub const E13_SPLADE_VOCAB: usize = 30_522;
```

**EmbeddingModel trait** - EXISTS in `crates/context-graph-embeddings/src/traits/embedding_model/trait_def.rs`:
- Single-model interface for individual embedders (E1-E13)
- Methods: `model_id()`, `embed()`, `is_initialized()`, `dimension()`
- This is for INDIVIDUAL models, NOT the orchestrating provider

### What Needs to Be Created (THIS TASK)

**MultiArrayEmbeddingProvider** - NEW trait that orchestrates 13 EmbeddingModel instances:
- Location: `crates/context-graph-core/src/traits/multi_array_embedding.rs`
- Returns complete SemanticFingerprint, NOT single Vec<f32>
- Orchestrates parallel calls to 13 individual embedders

### Legacy Code to Deprecate (NOT MIGRATE)

**EmbeddingProvider** - OBSOLETE in `crates/context-graph-core/src/traits/embedding_provider.rs`:
- Returns single `EmbeddingOutput { embedding: Vec<f32> }` - WRONG for multi-array
- Mark with `#[deprecated]` and delete after migration
- DO NOT create backwards-compatibility shims per FR-602

---

## Description

Create the `MultiArrayEmbeddingProvider` trait that returns `SemanticFingerprint` with all 13 embeddings instead of a single `Vec<f32>`. This trait ORCHESTRATES the 13 individual `EmbeddingModel` implementations.

**Key Architecture**:
```
MultiArrayEmbeddingProvider (NEW - this task)
    ├── calls 10 dense EmbeddingModel instances (E1-E5, E7-E11)
    ├── calls 2 sparse embedders (E6, E13 - SPLADE)
    └── calls E12 ColBERT (token-level)

    Returns: SemanticFingerprint (already exists from TASK-F001)
```

**NO FUSION** - Each embedding stored independently for:
1. Per-space HNSW index search (13 indexes)
2. Per-space Johari quadrant classification
3. Full information preservation (~60KB vs ~6KB fused)

---

## Acceptance Criteria

### Core Implementation
- [ ] `MultiArrayEmbeddingOutput` struct wrapping SemanticFingerprint with latency metrics
- [ ] Per-embedder latency tracking array `[Duration; 13]`
- [ ] `MultiArrayEmbeddingProvider` async trait with `embed_all()` and `embed_batch_all()`
- [ ] `dimensions()` returns `[usize; 13]` (E6=0, E13=0 for sparse)
- [ ] Performance target check: `is_within_latency_target()` for 30ms budget
- [ ] `e1_matryoshka_128()` accessor for Stage 2 truncated embedding

### Supporting Traits
- [ ] `SingleEmbedder` trait for composing dense embedders (wraps EmbeddingModel)
- [ ] `SparseEmbedder` trait for E6 and E13 SPLADE embeddings
- [ ] Object safety for all traits (`dyn MultiArrayEmbeddingProvider`)

### Deprecation
- [ ] Add `#[deprecated(since = "0.2.0", note = "Use MultiArrayEmbeddingProvider")]` to legacy trait
- [ ] NO backwards compatibility shims - fail fast if old trait used

### Tests (REAL DATA ONLY)
- [ ] Tests use actual SemanticFingerprint with real dimensions
- [ ] Tests verify actual embedding dimension sizes match constants
- [ ] NO mock data, NO placeholder zeros for entire embeddings
- [ ] Edge case tests for empty input, max length input, sparse overflow

---

## Implementation Steps

### Step 1: Create Multi-Array Embedding Module
**File**: `crates/context-graph-core/src/traits/multi_array_embedding.rs`

```rust
//! Multi-Array Embedding Provider for 13-embedding SemanticFingerprint generation.
//!
//! This trait orchestrates 13 individual embedders to produce a complete
//! SemanticFingerprint. NO FUSION - each embedding stored independently.

use crate::error::CoreResult;
use crate::types::fingerprint::{SemanticFingerprint, SparseVector, NUM_EMBEDDERS};
use async_trait::async_trait;
use std::time::Duration;

/// Output from multi-array embedding generation.
///
/// Contains the complete 13-embedding fingerprint plus performance metrics.
#[derive(Debug, Clone)]
pub struct MultiArrayEmbeddingOutput {
    /// Complete 13-embedding fingerprint (E1-E13)
    pub fingerprint: SemanticFingerprint,

    /// Total wall-clock latency for all 13 embeddings
    pub total_latency: Duration,

    /// Per-embedder latencies for performance optimization
    /// Index 0 = E1, Index 12 = E13
    pub per_embedder_latency: [Duration; NUM_EMBEDDERS],

    /// Model IDs used for each embedder slot
    pub model_ids: [String; NUM_EMBEDDERS],
}

impl MultiArrayEmbeddingOutput {
    /// Performance target: <30ms for all 13 embeddings (from constitution.yaml)
    pub const TARGET_LATENCY_MS: u64 = 30;

    /// Check if total latency is within the 30ms target.
    #[inline]
    pub fn is_within_latency_target(&self) -> bool {
        self.total_latency.as_millis() < Self::TARGET_LATENCY_MS as u128
    }

    /// Get E1 Matryoshka embedding truncated to 128D for Stage 2 fast filtering.
    ///
    /// # Panics
    /// Never panics - E1 is always 1024D, truncation to 128 is safe.
    #[inline]
    pub fn e1_matryoshka_128(&self) -> &[f32] {
        &self.fingerprint.e1_semantic[..128]
    }

    /// Get the slowest embedder for optimization targeting.
    pub fn slowest_embedder(&self) -> (usize, Duration) {
        self.per_embedder_latency
            .iter()
            .enumerate()
            .max_by_key(|(_, d)| *d)
            .map(|(i, d)| (i, *d))
            .unwrap_or((0, Duration::ZERO))
    }
}

/// Multi-Array Embedding Provider trait.
///
/// Orchestrates 13 individual embedders to produce complete SemanticFingerprint.
/// REPLACES the legacy single-vector EmbeddingProvider.
///
/// # Performance Targets (from constitution.yaml)
/// - Single content: <30ms for all 13 embeddings
/// - Batch (64 items): <100ms total
///
/// # Thread Safety
/// Requires `Send + Sync` for async task spawning across threads.
#[async_trait]
pub trait MultiArrayEmbeddingProvider: Send + Sync {
    /// Generate complete 13-embedding fingerprint for content.
    ///
    /// Calls all 13 embedders in parallel and combines into SemanticFingerprint.
    ///
    /// # Arguments
    /// * `content` - Text content to embed (must be non-empty)
    ///
    /// # Errors
    /// - `CoreError::EmptyInput` if content is empty
    /// - `CoreError::EmbeddingFailed` if any embedder fails
    /// - `CoreError::Timeout` if exceeds latency budget
    async fn embed_all(&self, content: &str) -> CoreResult<MultiArrayEmbeddingOutput>;

    /// Generate fingerprints for multiple contents in batch.
    ///
    /// # Performance Target
    /// - 64 contents: <100ms for all 13 embeddings per content
    ///
    /// # Arguments
    /// * `contents` - Slice of text contents (each must be non-empty)
    ///
    /// # Errors
    /// - `CoreError::EmptyInput` if any content is empty
    /// - `CoreError::BatchTooLarge` if batch exceeds 64 items
    async fn embed_batch_all(&self, contents: &[String]) -> CoreResult<Vec<MultiArrayEmbeddingOutput>>;

    /// Get expected dimensions for each embedder.
    ///
    /// Returns array where index matches embedder number (0 = E1, 12 = E13).
    /// Sparse embedders (E6, E13) return 0 since dimension is variable.
    fn dimensions(&self) -> [usize; NUM_EMBEDDERS] {
        [1024, 512, 512, 512, 768, 0, 1536, 384, 1024, 768, 384, 128, 0]
        // E1    E2   E3   E4   E5  E6  E7    E8   E9   E10  E11  E12  E13
    }

    /// Get model IDs for each embedder slot.
    fn model_ids(&self) -> [&str; NUM_EMBEDDERS];

    /// Check if all 13 embedders are initialized and ready.
    fn is_ready(&self) -> bool;

    /// Get health status for each embedder.
    ///
    /// Returns array of booleans: true = healthy, false = degraded/unavailable.
    fn health_status(&self) -> [bool; NUM_EMBEDDERS];

    /// Get count of healthy embedders.
    fn healthy_count(&self) -> usize {
        self.health_status().iter().filter(|&&h| h).count()
    }
}

/// Individual dense embedder trait for composition.
///
/// Wraps single EmbeddingModel for use in MultiArrayEmbeddingProvider.
/// Used for E1-E5, E7-E11 (10 dense embedders).
#[async_trait]
pub trait SingleEmbedder: Send + Sync {
    /// Fixed embedding dimension for this model.
    fn dimension(&self) -> usize;

    /// Model identifier (e.g., "nomic-embed-v1.5").
    fn model_id(&self) -> &str;

    /// Generate dense embedding vector.
    ///
    /// # Errors
    /// - `CoreError::EmptyInput` if content empty
    /// - `CoreError::InputTooLong` if exceeds model's max tokens
    async fn embed(&self, content: &str) -> CoreResult<Vec<f32>>;

    /// Check if model is loaded and ready.
    fn is_ready(&self) -> bool;
}

/// Sparse embedder trait for SPLADE-style embeddings.
///
/// Used for E6 (general sparse) and E13 (SPLADE v3 for Stage 1 recall).
#[async_trait]
pub trait SparseEmbedder: Send + Sync {
    /// Vocabulary size for sparse vector indices.
    fn vocab_size(&self) -> usize;

    /// Model identifier (e.g., "splade-v3-doc").
    fn model_id(&self) -> &str;

    /// Generate sparse embedding with indices and values.
    ///
    /// # Errors
    /// - `CoreError::EmptyInput` if content empty
    /// - `CoreError::SparseOverflow` if active terms exceed MAX_SPARSE_ACTIVE
    async fn embed_sparse(&self, content: &str) -> CoreResult<SparseVector>;

    /// Check if model is loaded and ready.
    fn is_ready(&self) -> bool;
}

/// Token-level embedder trait for ColBERT-style embeddings.
///
/// Used for E12 - produces per-token embeddings for late interaction.
#[async_trait]
pub trait TokenEmbedder: Send + Sync {
    /// Dimension per token (E12 = 128D).
    fn token_dimension(&self) -> usize;

    /// Maximum tokens supported.
    fn max_tokens(&self) -> usize;

    /// Model identifier.
    fn model_id(&self) -> &str;

    /// Generate per-token embeddings.
    ///
    /// Returns Vec of token embeddings, length varies by input.
    async fn embed_tokens(&self, content: &str) -> CoreResult<Vec<[f32; 128]>>;

    /// Check if model is loaded and ready.
    fn is_ready(&self) -> bool;
}
```

### Step 2: Update traits/mod.rs
**File**: `crates/context-graph-core/src/traits/mod.rs`

Add:
```rust
pub mod multi_array_embedding;

pub use multi_array_embedding::{
    MultiArrayEmbeddingOutput,
    MultiArrayEmbeddingProvider,
    SingleEmbedder,
    SparseEmbedder,
    TokenEmbedder,
};
```

### Step 3: Deprecate Legacy Trait
**File**: `crates/context-graph-core/src/traits/embedding_provider.rs`

Add deprecation attribute:
```rust
#[deprecated(
    since = "0.2.0",
    note = "Use MultiArrayEmbeddingProvider which returns SemanticFingerprint with all 13 embeddings"
)]
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    // ... existing code
}
```

### Step 4: Create Stub Implementation
**File**: `crates/context-graph-core/src/stubs/multi_array_stub.rs`

```rust
//! Stub implementation of MultiArrayEmbeddingProvider for testing.

use crate::error::CoreResult;
use crate::traits::multi_array_embedding::*;
use crate::types::fingerprint::{SemanticFingerprint, SparseVector, NUM_EMBEDDERS};
use async_trait::async_trait;
use std::time::Duration;

/// Test stub that returns deterministic embeddings based on content hash.
pub struct StubMultiArrayProvider {
    ready: bool,
}

impl StubMultiArrayProvider {
    pub fn new() -> Self {
        Self { ready: true }
    }

    /// Create a provider that reports as not ready (for error testing).
    pub fn not_ready() -> Self {
        Self { ready: false }
    }
}

impl Default for StubMultiArrayProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MultiArrayEmbeddingProvider for StubMultiArrayProvider {
    async fn embed_all(&self, content: &str) -> CoreResult<MultiArrayEmbeddingOutput> {
        if content.is_empty() {
            return Err(crate::error::CoreError::EmptyInput {
                field: "content".into(),
            });
        }

        if !self.ready {
            return Err(crate::error::CoreError::NotInitialized {
                component: "StubMultiArrayProvider".into(),
            });
        }

        // Generate deterministic embeddings from content hash
        let hash = content.bytes().fold(0u64, |acc, b| acc.wrapping_add(b as u64));
        let seed = (hash as f32) / 1000.0;

        let fingerprint = SemanticFingerprint::from_seed(seed);

        Ok(MultiArrayEmbeddingOutput {
            fingerprint,
            total_latency: Duration::from_millis(15), // Under 30ms target
            per_embedder_latency: [Duration::from_millis(1); NUM_EMBEDDERS],
            model_ids: Self::default_model_ids(),
        })
    }

    async fn embed_batch_all(&self, contents: &[String]) -> CoreResult<Vec<MultiArrayEmbeddingOutput>> {
        let mut results = Vec::with_capacity(contents.len());
        for content in contents {
            results.push(self.embed_all(content).await?);
        }
        Ok(results)
    }

    fn model_ids(&self) -> [&str; NUM_EMBEDDERS] {
        [
            "stub-e1-semantic",
            "stub-e2-temporal-recent",
            "stub-e3-temporal-historical",
            "stub-e4-causal",
            "stub-e5-structural",
            "stub-e6-sparse",
            "stub-e7-counterfactual",
            "stub-e8-analogical",
            "stub-e9-emergent",
            "stub-e10-multimodal",
            "stub-e11-code",
            "stub-e12-colbert",
            "stub-e13-splade",
        ]
    }

    fn is_ready(&self) -> bool {
        self.ready
    }

    fn health_status(&self) -> [bool; NUM_EMBEDDERS] {
        [self.ready; NUM_EMBEDDERS]
    }
}

impl StubMultiArrayProvider {
    fn default_model_ids() -> [String; NUM_EMBEDDERS] {
        [
            "stub-e1".into(), "stub-e2".into(), "stub-e3".into(),
            "stub-e4".into(), "stub-e5".into(), "stub-e6".into(),
            "stub-e7".into(), "stub-e8".into(), "stub-e9".into(),
            "stub-e10".into(), "stub-e11".into(), "stub-e12".into(),
            "stub-e13".into(),
        ]
    }
}
```

### Step 5: Update stubs/mod.rs
Add export for new stub module.

---

## Files Affected

### Files to CREATE
| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/traits/multi_array_embedding.rs` | New trait definitions |
| `crates/context-graph-core/src/stubs/multi_array_stub.rs` | Test stub implementation |
| `crates/context-graph-core/src/traits/multi_array_embedding/tests.rs` | Unit tests |

### Files to MODIFY
| File | Change |
|------|--------|
| `crates/context-graph-core/src/traits/mod.rs` | Add multi_array_embedding module export |
| `crates/context-graph-core/src/traits/embedding_provider.rs` | Add #[deprecated] attribute |
| `crates/context-graph-core/src/stubs/mod.rs` | Add multi_array_stub module export |

### Files to REFERENCE (read-only)
| File | Reason |
|------|--------|
| `crates/context-graph-core/src/types/fingerprint/semantic.rs` | SemanticFingerprint struct |
| `crates/context-graph-core/src/types/fingerprint/sparse.rs` | SparseVector struct |
| `crates/context-graph-core/src/error/mod.rs` | CoreResult, CoreError types |
| `crates/context-graph-embeddings/src/traits/embedding_model/trait_def.rs` | EmbeddingModel interface |

---

## Testing Requirements

### Unit Tests (REAL DATA - NO MOCKS)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::fingerprint::{E1_DIM, E2_DIM, NUM_EMBEDDERS};

    #[test]
    fn test_multi_array_output_within_latency_target() {
        let output = MultiArrayEmbeddingOutput {
            fingerprint: SemanticFingerprint::zeroed(),
            total_latency: Duration::from_millis(25),
            per_embedder_latency: [Duration::from_millis(2); NUM_EMBEDDERS],
            model_ids: array_init(|_| String::new()),
        };
        assert!(output.is_within_latency_target());
    }

    #[test]
    fn test_multi_array_output_exceeds_latency_target() {
        let output = MultiArrayEmbeddingOutput {
            fingerprint: SemanticFingerprint::zeroed(),
            total_latency: Duration::from_millis(50), // Over 30ms
            per_embedder_latency: [Duration::from_millis(4); NUM_EMBEDDERS],
            model_ids: array_init(|_| String::new()),
        };
        assert!(!output.is_within_latency_target());
    }

    #[test]
    fn test_e1_matryoshka_128_truncation() {
        let mut fp = SemanticFingerprint::zeroed();
        // Set known values in first 128 elements
        for i in 0..128 {
            fp.e1_semantic[i] = i as f32;
        }

        let output = MultiArrayEmbeddingOutput {
            fingerprint: fp,
            total_latency: Duration::ZERO,
            per_embedder_latency: [Duration::ZERO; NUM_EMBEDDERS],
            model_ids: array_init(|_| String::new()),
        };

        let truncated = output.e1_matryoshka_128();
        assert_eq!(truncated.len(), 128);
        assert_eq!(truncated[0], 0.0);
        assert_eq!(truncated[127], 127.0);
    }

    #[test]
    fn test_dimensions_returns_correct_values() {
        struct TestProvider;
        impl MultiArrayEmbeddingProvider for TestProvider {
            // ... minimal impl
        }

        let provider = TestProvider;
        let dims = provider.dimensions();

        assert_eq!(dims[0], 1024);  // E1
        assert_eq!(dims[1], 512);   // E2
        assert_eq!(dims[5], 0);     // E6 sparse
        assert_eq!(dims[11], 128);  // E12 ColBERT token
        assert_eq!(dims[12], 0);    // E13 sparse
    }

    #[test]
    fn test_slowest_embedder_identification() {
        let mut latencies = [Duration::from_millis(1); NUM_EMBEDDERS];
        latencies[7] = Duration::from_millis(15); // E8 is slowest

        let output = MultiArrayEmbeddingOutput {
            fingerprint: SemanticFingerprint::zeroed(),
            total_latency: Duration::from_millis(25),
            per_embedder_latency: latencies,
            model_ids: array_init(|_| String::new()),
        };

        let (idx, duration) = output.slowest_embedder();
        assert_eq!(idx, 7);
        assert_eq!(duration, Duration::from_millis(15));
    }

    #[test]
    fn test_trait_object_safety() {
        // Must compile - proves trait is object-safe
        fn accepts_provider(_: &dyn MultiArrayEmbeddingProvider) {}
        fn accepts_single(_: &dyn SingleEmbedder) {}
        fn accepts_sparse(_: &dyn SparseEmbedder) {}
    }
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_stub_provider_returns_valid_fingerprint() {
    let provider = StubMultiArrayProvider::new();
    let output = provider.embed_all("test content").await.unwrap();

    // Verify fingerprint has correct dimensions
    assert_eq!(output.fingerprint.e1_semantic.len(), 1024);
    assert_eq!(output.fingerprint.e2_temporal_recent.len(), 512);
    assert!(output.fingerprint.e13_splade.len() <= MAX_SPARSE_ACTIVE);
}

#[tokio::test]
async fn test_stub_provider_rejects_empty_input() {
    let provider = StubMultiArrayProvider::new();
    let result = provider.embed_all("").await;

    assert!(matches!(result, Err(CoreError::EmptyInput { .. })));
}

#[tokio::test]
async fn test_stub_provider_not_ready_fails() {
    let provider = StubMultiArrayProvider::not_ready();
    let result = provider.embed_all("content").await;

    assert!(matches!(result, Err(CoreError::NotInitialized { .. })));
}

#[tokio::test]
async fn test_batch_processing_returns_correct_count() {
    let provider = StubMultiArrayProvider::new();
    let contents: Vec<String> = (0..10).map(|i| format!("content {}", i)).collect();

    let results = provider.embed_batch_all(&contents).await.unwrap();
    assert_eq!(results.len(), 10);
}
```

### Edge Case Tests

```rust
#[tokio::test]
async fn test_very_long_input_handling() {
    let provider = StubMultiArrayProvider::new();
    let long_content = "x".repeat(100_000); // 100KB input

    // Should either succeed or return InputTooLong error - never panic
    let result = provider.embed_all(&long_content).await;
    assert!(result.is_ok() || matches!(result, Err(CoreError::InputTooLong { .. })));
}

#[tokio::test]
async fn test_unicode_content_handling() {
    let provider = StubMultiArrayProvider::new();
    let unicode = "Hello \u{1F600} World \u{4E2D}\u{6587} \u{0410}\u{0411}\u{0412}";

    let result = provider.embed_all(unicode).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_whitespace_only_content() {
    let provider = StubMultiArrayProvider::new();

    // Whitespace-only should be treated as empty
    let result = provider.embed_all("   \t\n  ").await;
    // Implementation may accept or reject - document behavior
}
```

---

## Full State Verification Protocol

### 1. Source of Truth Identification

| Component | Source of Truth | Verification Command |
|-----------|-----------------|---------------------|
| SemanticFingerprint dimensions | `semantic.rs` constants | `grep "pub const E.*_DIM" semantic.rs` |
| NUM_EMBEDDERS | `semantic.rs` | `grep "NUM_EMBEDDERS" semantic.rs` |
| SparseVector constraints | `sparse.rs` | `grep "MAX_SPARSE_ACTIVE\|SPARSE_VOCAB" sparse.rs` |
| Error types | `error/mod.rs` | `grep "pub enum CoreError" error/mod.rs` |
| Performance targets | `constitution.yaml` | Search for "latency" and "30ms" |

### 2. Execute & Inspect Verification

After implementation, run these commands:

```bash
# 1. Verify compilation
cargo check -p context-graph-core 2>&1 | head -50

# 2. Run all new tests
cargo test -p context-graph-core multi_array 2>&1

# 3. Verify trait exports in documentation
cargo doc -p context-graph-core --no-deps 2>&1 | grep -i "multi_array"

# 4. Verify deprecation warning appears
cargo check -p context-graph-core 2>&1 | grep -i "deprecated"

# 5. Check that all 13 embedder slots are documented
grep -c "E[0-9]*" crates/context-graph-core/src/traits/multi_array_embedding.rs
```

### 3. Edge Case Audit

| Edge Case | Expected Behavior | Test |
|-----------|-------------------|------|
| Empty string input | Return `CoreError::EmptyInput` | `test_stub_provider_rejects_empty_input` |
| Provider not ready | Return `CoreError::NotInitialized` | `test_stub_provider_not_ready_fails` |
| Latency over 30ms | `is_within_latency_target()` returns false | `test_exceeds_latency_target` |
| Batch size 0 | Return empty Vec, no error | Verify in batch test |
| Single embedder failure | Propagate error, don't partial-succeed | Document in trait |
| Sparse overflow (>128 active) | Return `CoreError::SparseOverflow` | Implement in sparse embedder |

### 4. Evidence of Success

Before marking complete, verify these artifacts exist:

1. **File exists**: `crates/context-graph-core/src/traits/multi_array_embedding.rs`
2. **Module exported**: `multi_array_embedding` in `traits/mod.rs`
3. **Types exported**: `MultiArrayEmbeddingOutput`, `MultiArrayEmbeddingProvider`, `SingleEmbedder`, `SparseEmbedder`
4. **Tests pass**: All `multi_array` tests green
5. **Deprecation active**: Compiler warns on `EmbeddingProvider` usage
6. **No backwards compat**: No shim code, no migration adapters

---

## Sherlock Holmes Verification Step

After implementation is complete, spawn a **sherlock-holmes** verification agent with this prompt:

```
FORENSIC INVESTIGATION: TASK-F007 Multi-Array Embedding Provider

ASSUME ALL CODE IS GUILTY UNTIL PROVEN INNOCENT.

Investigation targets:
1. Verify MultiArrayEmbeddingProvider trait exists and is exported
2. Verify trait returns SemanticFingerprint (not Vec<f32>)
3. Verify all 13 embedder slots are accounted for (E1-E13)
4. Verify sparse embedders (E6, E13) return SparseVector not Vec<f32>
5. Verify latency target check uses 30ms from constitution.yaml
6. Verify legacy EmbeddingProvider has #[deprecated] attribute
7. Verify NO backwards compatibility shims exist
8. Verify all tests use real SemanticFingerprint data, no all-zero mocks
9. Run cargo test and verify all multi_array tests pass
10. Check for any TODO, FIXME, or unimplemented!() in new code

EVIDENCE REQUIRED:
- File paths with line numbers for each finding
- Actual code snippets proving compliance
- Test output showing green status
- Compiler output showing deprecation warnings

FAIL FAST: Report first violation found, do not continue if critical issue.
```

---

## Verification Commands

```bash
# Full verification sequence
cargo check -p context-graph-core && \
cargo test -p context-graph-core multi_array && \
cargo doc -p context-graph-core --no-deps

# Verify no backwards compat code
grep -r "impl.*Into.*EmbeddingProvider" crates/context-graph-core/
# Should return nothing

# Verify deprecation
grep -A2 "#\[deprecated" crates/context-graph-core/src/traits/embedding_provider.rs
```

---

## Constraints

### From constitution.yaml
- Single embed (all 13): <30ms
- Batch embed (64 x 13): <100ms per item
- All async for parallel execution
- `Send + Sync` for multi-threaded runtime

### From FR-602 (No Backwards Compatibility)
- NO migration shims
- NO adapter patterns
- NO deprecated code paths that still work
- FAIL FAST if old patterns detected

### Technical
- Dimensions: E6=0, E13=0 (sparse variable), E12=128 (per token)
- E1 supports Matryoshka truncation (1024 → 128D)
- Object-safe traits for dynamic dispatch

---

## Dependencies

| Dependency | Status | Verification |
|------------|--------|--------------|
| TASK-F001 (SemanticFingerprint) | COMPLETE | `grep "SemanticFingerprint" semantic.rs` |
| TASK-F006 (Remove Fusion) | COMPLETE | No fusion modules in git history |
| async-trait crate | Available | Check Cargo.toml |
| CoreResult type | Exists | Check error/mod.rs |

---

## Notes

### Architecture Relationship

```
EmbeddingModel (embeddings crate)     MultiArrayEmbeddingProvider (core crate)
├── Single model interface            ├── Orchestrates 13 EmbeddingModel instances
├── embed() → Vec<f32>                ├── embed_all() → SemanticFingerprint
└── Used by individual E1-E13         └── Used by application layer
```

### 5-Stage Pipeline Integration
- **Stage 1**: E13 SPLADE for initial recall (sparse retrieval)
- **Stage 2**: E1 Matryoshka 128D for fast dense filtering
- **Stage 3**: Full E1-E12 dense embeddings for precision
- **Stage 4**: E12 ColBERT for late interaction reranking
- **Stage 5**: Purpose vector from teleological computation

### Why No Fusion
Per TASK-F006, fusion was removed because:
1. Information loss: 67% semantic information lost in fusion
2. Per-space search: Each HNSW index needs original embeddings
3. Johari computation: Per-embedder quadrant classification
4. Auditability: Can trace which embedder contributed to ranking
