# Stub and UTL Removal Plan

## Executive Summary

This document outlines the complete removal of:
1. **All stub implementations** from the codebase
2. **The entire UTL (Unified Theory of Learning) system** - UTL is no longer used

**Philosophy**: NO workarounds, NO fallbacks. The system works or fails fast with robust error logging.

---

## Part 1: UTL Complete Removal

### 1.1 What is UTL?

UTL (Unified Theory of Learning) was an experimental learning framework that computed:
- ΔS (Surprise) - novelty detection via KNN distance
- ΔC (Coherence) - connectivity measurement
- Learning scores via formula: `L = sigmoid(2.0 · ΔS · ΔC · wₑ · cos φ)`

**Status**: UTL is NO LONGER USED. The system now relies purely on the 13-embedder architecture for memory relevance.

### 1.2 UTL Components to Remove

#### A. Entire Crate: `crates/context-graph-utl/`
**Action**: DELETE ENTIRE DIRECTORY

Contains:
- `src/surprise/` - Surprise calculators (embedder_entropy, kl_divergence, etc.)
- `src/coherence/` - Coherence tracking (cluster_fit, structural, tracker, window)
- `src/emotional/` - Emotional state processing
- `src/learning/` - Learning signal computation
- `src/lifecycle/` - Lifecycle management
- `src/phase/` - Phase oscillators and consolidation
- `src/processor/` - UTL processor implementation
- `src/metrics/` - UTL metrics
- `src/config/` - UTL configuration
- `benches/` - UTL benchmarks
- `tests/` - UTL tests

#### B. Core UTL Files
| File | Action |
|------|--------|
| `crates/context-graph-core/src/stubs/utl_stub/` | DELETE DIRECTORY |
| `crates/context-graph-core/src/traits/utl_processor.rs` | DELETE FILE |
| `crates/context-graph-core/src/types/utl.rs` | DELETE FILE |
| `crates/context-graph-mcp/src/adapters/utl_adapter.rs` | DELETE FILE |

#### C. Files Requiring UTL Reference Removal

**High Impact (code changes required):**
| File | Changes |
|------|---------|
| `crates/context-graph-core/src/lib.rs` | Remove `pub mod utl` export |
| `crates/context-graph-core/src/types/mod.rs` | Remove UTL type exports |
| `crates/context-graph-core/src/traits/mod.rs` | Remove `UtlProcessor` trait export |
| `crates/context-graph-core/src/stubs/mod.rs` | Remove `utl_stub` module |
| `crates/context-graph-mcp/src/adapters/mod.rs` | Remove `utl_adapter` module |
| `crates/context-graph-mcp/src/lib.rs` | Remove UTL imports |
| `crates/context-graph-mcp/src/server.rs` | Remove UTL adapter usage |
| `crates/context-graph-mcp/Cargo.toml` | Remove `context-graph-utl` dependency |
| `Cargo.toml` (workspace) | Remove `context-graph-utl` from members |

**Medium Impact (docstring/comment cleanup):**
- `crates/context-graph-core/src/types/memory_node/metadata.rs`
- `crates/context-graph-core/src/types/pulse/cognitive_pulse.rs`
- `crates/context-graph-core/src/similarity/*.rs`
- `crates/context-graph-core/src/neuromod/*.rs`
- `crates/context-graph-core/src/gwt/*.rs`
- `crates/context-graph-core/src/dream/*.rs`
- `crates/context-graph-core/src/clustering/*.rs`
- `crates/context-graph-core/src/config/*.rs`
- `crates/context-graph-mcp/src/handlers/tools/*.rs`
- `crates/context-graph-mcp/src/middleware/*.rs`

**Low Impact (configuration cleanup):**
- `config/default.toml` - Remove UTL config sections
- `config/test.toml` - Remove UTL config sections
- `config/production.toml` - Remove UTL config sections
- `config/development.toml` - Remove UTL config sections
- `.github/workflows/ci.yml` - Remove UTL crate from CI

---

## Part 2: Stub Removal

### 2.1 Stub Components to Remove

#### A. Core Stubs Directory: `crates/context-graph-core/src/stubs/`

| Component | Production Replacement | Action |
|-----------|----------------------|--------|
| `multi_array_stub.rs` | `ProductionMultiArrayProvider` in `context-graph-embeddings` | DELETE |
| `teleological_store_stub/` | `RocksDbTeleologicalStore` in `context-graph-storage` | DELETE |
| `graph_index.rs` | FAISS GPU index in `context-graph-graph` | DELETE |
| `utl_stub/` | N/A (UTL removed entirely) | DELETE |
| `mod.rs` | N/A | DELETE (entire stubs module) |

#### B. CUDA Stub: `crates/context-graph-cuda/src/stub.rs`

| Component | Production Replacement | Action |
|-----------|----------------------|--------|
| `StubVectorOps` | Real CUDA vector ops | DELETE |

### 2.2 Stub Reference Removal

Files referencing stubs that need updating:
- `crates/context-graph-core/src/lib.rs` - Remove `pub mod stubs` export
- `crates/context-graph-cuda/src/lib.rs` - Remove stub module
- All test files using `StubMultiArrayProvider`, `InMemoryTeleologicalStore`, `InMemoryGraphIndex`

### 2.3 Test Migration Strategy

**Current State**: Tests use stub implementations for deterministic behavior.

**Target State**: Tests use REAL implementations with REAL data.

**Migration Approach**:
1. Tests requiring embedding MUST use `ProductionMultiArrayProvider` (requires GPU)
2. Tests requiring storage MUST use `RocksDbTeleologicalStore` (requires disk)
3. Tests MUST be marked `#[ignore]` if hardware unavailable, NOT stubbed
4. Synthetic test data with KNOWN inputs/outputs for verification

---

## Part 3: Implementation Order

### Phase 1: UTL Removal (Clean Break)

1. **Remove workspace member**
   - Edit `Cargo.toml` to remove `context-graph-utl` from members
   - Edit dependent `Cargo.toml` files to remove dependency

2. **Delete UTL crate**
   - `rm -rf crates/context-graph-utl/`

3. **Remove UTL from core**
   - Delete `crates/context-graph-core/src/traits/utl_processor.rs`
   - Delete `crates/context-graph-core/src/types/utl.rs`
   - Delete `crates/context-graph-core/src/stubs/utl_stub/`
   - Update `mod.rs` files to remove exports

4. **Remove UTL from MCP**
   - Delete `crates/context-graph-mcp/src/adapters/utl_adapter.rs`
   - Update adapter module
   - Update server.rs

5. **Clean up references**
   - Remove UTL docstrings/comments from all files
   - Remove UTL configuration sections

6. **Verify build**
   - `cargo build --all-features`
   - `cargo test --all-features`

### Phase 2: Stub Removal

1. **Verify production implementations exist**
   - Confirm `ProductionMultiArrayProvider` is complete
   - Confirm `RocksDbTeleologicalStore` is complete
   - Confirm FAISS GPU index is complete
   - Confirm CUDA vector ops are complete

2. **Migrate tests**
   - Update tests to use production implementations
   - Mark hardware-dependent tests appropriately
   - Create synthetic test datasets

3. **Delete stub implementations**
   - Delete entire `crates/context-graph-core/src/stubs/` directory
   - Delete `crates/context-graph-cuda/src/stub.rs`

4. **Remove stub feature flags**
   - Remove `test-utils` feature that gates stubs
   - Update Cargo.toml files

5. **Verify build**
   - `cargo build --all-features`
   - `cargo test --all-features` (with GPU)

---

## Part 4: Error Handling Strategy

### 4.1 Fail Fast Principle

When production implementations are unavailable:

```rust
// WRONG - Don't fallback to stubs
let provider = match ProductionMultiArrayProvider::new(...).await {
    Ok(p) => p,
    Err(_) => StubMultiArrayProvider::new(), // NO!
};

// RIGHT - Fail fast with clear error
let provider = ProductionMultiArrayProvider::new(...)
    .await
    .map_err(|e| {
        tracing::error!(
            error = %e,
            models_dir = ?models_dir,
            "FATAL: Failed to initialize embedding provider. \
             Ensure all 13 models exist and CUDA GPU is available."
        );
        e
    })?;
```

### 4.2 Required Error Messages

All errors MUST include:
1. **What failed** - Component name
2. **Why it failed** - Root cause
3. **What to fix** - Actionable resolution
4. **Context** - Relevant paths, IDs, configurations

---

## Part 5: Verification Checklist

### UTL Removal Verification
- [ ] `context-graph-utl` crate deleted
- [ ] No `utl` or `UTL` references in codebase (except this doc)
- [ ] `cargo build` succeeds
- [ ] `cargo test` succeeds
- [ ] No UTL types in public API

### Stub Removal Verification
- [ ] `stubs/` directory deleted from core
- [ ] `stub.rs` deleted from cuda
- [ ] No `Stub` types in codebase
- [ ] No `test-utils` feature flag
- [ ] All tests use production implementations
- [ ] `cargo build` succeeds
- [ ] `cargo test` succeeds (with appropriate hardware)

---

## Part 6: Files Summary

### Files to DELETE (UTL)
```
crates/context-graph-utl/                          # ENTIRE CRATE
crates/context-graph-core/src/stubs/utl_stub/      # UTL stub directory
crates/context-graph-core/src/traits/utl_processor.rs
crates/context-graph-core/src/types/utl.rs
crates/context-graph-mcp/src/adapters/utl_adapter.rs
```

### Files to DELETE (Stubs)
```
crates/context-graph-core/src/stubs/               # ENTIRE DIRECTORY
crates/context-graph-cuda/src/stub.rs
```

### Files to MODIFY (Remove UTL/Stub references)
```
Cargo.toml                                         # Remove utl from workspace
crates/context-graph-core/Cargo.toml
crates/context-graph-core/src/lib.rs
crates/context-graph-core/src/types/mod.rs
crates/context-graph-core/src/traits/mod.rs
crates/context-graph-core/src/error/*.rs
crates/context-graph-core/src/config/*.rs
crates/context-graph-mcp/Cargo.toml
crates/context-graph-mcp/src/lib.rs
crates/context-graph-mcp/src/server.rs
crates/context-graph-mcp/src/adapters/mod.rs
crates/context-graph-mcp/src/handlers/tools/*.rs
crates/context-graph-cuda/src/lib.rs
config/*.toml
.github/workflows/ci.yml
```

---

## Approval

**This plan requires approval before implementation.**

Once approved, implementation will proceed in the order specified above with full state verification at each phase boundary.
