# Task: TASK-F007 - Modify EmbeddingProvider Trait for Multi-Array Output

## Metadata
- **ID**: TASK-F007
- **Layer**: Foundation
- **Priority**: P0 (Critical Path)
- **Estimated Effort**: M (Medium)
- **Dependencies**: TASK-F001, TASK-F006
- **Traces To**: TS-301, FR-101, FR-102, FR-104

## Description

Modify the `EmbeddingProvider` trait to return `SemanticFingerprint` with all 12 embeddings instead of a single `Vec<f32>`. This is a fundamental change that enables the multi-array architecture.

Current trait returns `EmbeddingOutput` with single vector. New trait returns `MultiArrayEmbeddingOutput` with complete `SemanticFingerprint`.

**Key Change**: `embed()` returns 12 embeddings, not 1.

## Acceptance Criteria

- [ ] `MultiArrayEmbeddingOutput` struct with SemanticFingerprint
- [ ] Per-embedder latency tracking (for optimization)
- [ ] `MultiArrayEmbeddingProvider` trait replacing legacy `EmbeddingProvider`
- [ ] `embed_all()` method returns complete fingerprint
- [ ] `embed_batch_all()` for batch processing
- [ ] `dimensions()` returns array of 12 dimensions
- [ ] `SingleEmbedder` trait for composing multi-array provider
- [ ] Performance target check method (30ms for all 12)
- [ ] Legacy `EmbeddingProvider` kept but marked deprecated (for gradual migration)
- [ ] Unit tests for new trait structure

## Implementation Steps

1. Read existing `crates/context-graph-core/src/traits/embedding_provider.rs`
2. Create `crates/context-graph-core/src/traits/multi_array_embedding.rs`:
   - Import SemanticFingerprint from fingerprint module
   - Implement MultiArrayEmbeddingOutput struct
   - Implement MultiArrayEmbeddingProvider trait
   - Implement SingleEmbedder trait
3. Update `crates/context-graph-core/src/traits/mod.rs`:
   - Add `pub mod multi_array_embedding;`
   - Export new types
4. Add `#[deprecated]` attribute to old EmbeddingProvider (optional, for gradual migration)
5. Update stub implementation in `crates/context-graph-core/src/stubs/embedding_stub.rs`

## Files Affected

### Files to Create
- `crates/context-graph-core/src/traits/multi_array_embedding.rs` - New trait definition

### Files to Modify
- `crates/context-graph-core/src/traits/mod.rs` - Export new module
- `crates/context-graph-core/src/traits/embedding_provider.rs` - Add deprecation note
- `crates/context-graph-core/src/stubs/embedding_stub.rs` - Implement new trait

## Code Signature (Definition of Done)

```rust
// multi_array_embedding.rs
use crate::error::CoreResult;
use crate::types::fingerprint::SemanticFingerprint;
use async_trait::async_trait;
use std::time::Duration;

/// Output from multi-array embedding generation
#[derive(Debug, Clone)]
pub struct MultiArrayEmbeddingOutput {
    /// The complete 12-embedding fingerprint
    pub fingerprint: SemanticFingerprint,

    /// Total latency for all 12 embeddings
    pub total_latency: Duration,

    /// Per-embedder latencies (for optimization)
    pub per_embedder_latency: [Duration; 12],

    /// Per-embedder model IDs
    pub model_ids: [String; 12],
}

impl MultiArrayEmbeddingOutput {
    /// Expected total latency target: <30ms for all 12 embedders
    pub const TARGET_LATENCY_MS: u64 = 30;

    /// Check if latency is within target
    pub fn is_within_latency_target(&self) -> bool;
}

/// Multi-Array Embedding Provider trait
///
/// REPLACES the legacy EmbeddingProvider that returned Vec<f32>.
/// Returns complete SemanticFingerprint with all 12 embeddings.
///
/// NO FUSION - each embedder output stored independently.
#[async_trait]
pub trait MultiArrayEmbeddingProvider: Send + Sync {
    /// Generate complete 12-embedding fingerprint for content
    ///
    /// # Performance Target
    /// - Single content: <30ms for all 12 embeddings
    async fn embed_all(&self, content: &str) -> CoreResult<MultiArrayEmbeddingOutput>;

    /// Generate fingerprints for multiple contents in batch
    ///
    /// # Performance Target
    /// - 64 contents: <100ms for all 12 embeddings per content
    async fn embed_batch_all(&self, contents: &[String]) -> CoreResult<Vec<MultiArrayEmbeddingOutput>>;

    /// Get expected dimensions for each embedder
    fn dimensions(&self) -> [usize; 12] {
        [1024, 512, 512, 512, 768, 0, 1536, 384, 1024, 768, 384, 128]
    }

    /// Get model IDs for each embedder
    fn model_ids(&self) -> [&str; 12];

    /// Check if all embedders are ready
    fn is_ready(&self) -> bool;

    /// Get health status per embedder
    fn health_status(&self) -> [bool; 12];
}

/// Individual embedder trait for composing MultiArrayEmbeddingProvider
#[async_trait]
pub trait SingleEmbedder: Send + Sync {
    /// Embedding dimension for this embedder
    fn dimension(&self) -> usize;

    /// Model identifier
    fn model_id(&self) -> &str;

    /// Generate single embedding
    async fn embed(&self, content: &str) -> CoreResult<Vec<f32>>;

    /// Check if ready
    fn is_ready(&self) -> bool;
}
```

## Testing Requirements

### Unit Tests
- `test_multi_array_output_latency_check` - Returns true when under 30ms
- `test_multi_array_output_latency_exceeded` - Returns false when over 30ms
- `test_dimensions_default` - Returns correct 12 dimensions
- `test_trait_object_safety` - Can create `dyn MultiArrayEmbeddingProvider`

### Integration Tests
- Test with stub implementation returning valid SemanticFingerprint
- Test batch processing returns correct count

## Verification

```bash
# Compile check
cargo check -p context-graph-core

# Run unit tests
cargo test -p context-graph-core multi_array_embedding

# Verify trait exports
cargo doc -p context-graph-core --open
```

## Constraints

- Performance targets from constitution.yaml:
  - Single embed (all 12): <30ms
  - Batch embed (64 x 12): <100ms
- All methods must be `async` for parallel embedding generation
- `Send + Sync` required for multi-threaded use
- Dimensions array: E6 returns 0 (sparse is variable), E12 returns 128 (per token)

## Migration Notes

The legacy `EmbeddingProvider` trait is kept for backwards compatibility during migration:
1. New code should use `MultiArrayEmbeddingProvider`
2. Existing implementations can implement both traits
3. Gradual migration path: old trait calls can wrap new trait

However, per FR-602 (No Backwards Compatibility), migration shims should NOT be created. Old code using `EmbeddingProvider` should be updated directly.

## Notes

This trait change is fundamental to the multi-array architecture. All embedding generation now returns 12 vectors, not 1.

The `SingleEmbedder` trait allows composition:
```rust
struct CompositeEmbeddingProvider {
    embedders: [Box<dyn SingleEmbedder>; 12],
}

impl MultiArrayEmbeddingProvider for CompositeEmbeddingProvider {
    async fn embed_all(&self, content: &str) -> CoreResult<MultiArrayEmbeddingOutput> {
        // Call all 12 embedders in parallel
        // Combine results into SemanticFingerprint
    }
}
```

Reference implementation in TECH-SPEC-001 Section 3.1 (TS-301).
