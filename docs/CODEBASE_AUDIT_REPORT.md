# Codebase Forensic Audit Report

**Date:** 2026-02-11
**Branch:** casetrack
**Scope:** All crates in `/home/cabdru/contextgraph/crates/` (~1,510 .rs files, ~468K lines)

---

## Executive Summary

| Category | Findings | Severity | Est. Lines Affected |
|----------|----------|----------|---------------------|
| Dead/Unreachable Code | 10 major items | HIGH | ~36,600 |
| Broken/Fake Implementations | 10 items | HIGH | ~2,000 |
| Redundant/Duplicate Code | 9 patterns | MEDIUM-HIGH | ~1,100 |
| Test Quality Issues | 7 categories | HIGH | ~3,000 untested |

**Bottom line:** The codebase compiles, all 3,435 tests pass, and the 55 MCP tools function correctly. However, beneath that surface are ~36K lines of dead code, fake health monitoring that always reports "active", phantom configuration options that do nothing, audit logging that silently fails, and critical code paths with zero test coverage.

---

## Section 1: Dead / Unreachable Code

### 1.1 CRITICAL: Entire `context-graph-graph` Crate Is Isolated (~29,741 lines)

**Location:** `crates/context-graph-graph/` (192 files)
**Contains:** FAISS GPU index, hyperbolic geometry, entailment cones, graph traversal, contradiction detection

No other crate in the workspace depends on `context-graph-graph`. All 63 files referencing `context_graph_graph::` are within the crate itself. It compiles as a workspace member but is an isolated island with zero consumers.

**Impact:** ~29,741 lines compiled on every build that serve no purpose.

### 1.2 HIGH: Dead Core Modules (~5,236 lines)

| Module | Files | Lines | Evidence |
|--------|-------|-------|----------|
| `core::steering` (Gardener/Curator/Assessor) | 5 | 1,444 | Zero references outside its own `#[cfg(test)]` blocks |
| `core::atc` (Adaptive Temperature Control) | 8 | 3,789 | Only used by 1 integration test, never by MCP/CLI/storage |
| `core::session` | 1 | 3 | Empty placeholder with "reserved for future" comment |

### 1.3 HIGH: Training Tool Defined But Never Registered

**Location:** `crates/context-graph-mcp/src/tools/definitions/training.rs:10`
**Constant:** `TRAIN_CAUSAL_EMBEDDER` in `names.rs:128` -- zero references

The `training` module is declared in `definitions/mod.rs:35` and compiles, but `training::definitions()` is never called in `get_tool_definitions()`. The `train_causal_embedder` MCP tool is defined but invisible to clients.

### 1.4 MEDIUM: Dead Functions (Compiler-Confirmed)

**30+ dead functions** across the codebase, primarily in:
- **Benchmark binaries** (lower priority): `gpu_bench.rs` (4 functions), `mcp_bench.rs` (8 functions), `sparse_bench.rs` (3 functions), `unified_realdata.rs` (4 functions)
- **MCP production code** (higher priority):
  - `causal_hint.rs:88` -- `with_min_confidence()` builder never called
  - `causal_hint.rs:97` -- `load_model()` never called
  - `entity_dtos.rs:391` -- `string_to_entity_type()` only used in its own test
  - `entity_dtos.rs:143` -- `EntityBoostConfig` methods only used in own tests

### 1.5 MEDIUM: Dead Struct Fields

| File | Field | Issue |
|------|-------|-------|
| `embedder_dtos.rs:485` | `include_details` on `ListEmbedderIndexesRequest` | Deserialized but never read by handler |
| `entity_dtos.rs:461` | `entity_types` on request struct | Never read |
| `entity_dtos.rs:961` | `entity_types`, `include_memory_counts` | Never read |
| `user_prompt_submit.rs:618` | `id` on `RecentConversationTurn` | Set but never read |
| Multiple benchmark structs | `config`, `rng`, `checkpoint_interval`, etc. | 12+ fields never read |

### 1.6 LOW: Dead Feature Flags

| Crate | Feature | Status |
|-------|---------|--------|
| `context-graph-embeddings` | `onnx` | Defined, zero `cfg` gates in code |
| `context-graph-cuda` | `cudnn` | Defined, zero `cfg` gates in code |
| `context-graph-mcp` | `integration` | Defined, zero `cfg` gates in code |
| `context-graph-cuda` | `faiss-working` | Defined, never enabled by any dependent |
| `context-graph-graph` | `faiss-gpu` | Default + `compile_error!` without it = can never be disabled |
| `context-graph-causal-agent` | `test-mode` | Code-gated but never enabled in any Cargo.toml |
| `context-graph-graph-agent` | `test-mode` | Code-gated but never enabled in any Cargo.toml |

### 1.7 LOW: Public Functions Never Called Outside Own Crate

| File | Function |
|------|----------|
| `causal/chain.rs:146` | `compute_chain_score_raw()` |
| `causal/chain.rs:260` | `score_causal_chain_attenuated()` |
| `causal/chain.rs:384` | `rank_causes_by_abduction_raw()` |
| `causal/chain.rs:476` | `rank_effects_by_prediction_raw()` |
| `causal/asymmetric.rs:532` | `compute_e5_asymmetric_full()` |
| `graph/asymmetric.rs:467` | `compute_e8_asymmetric_full()` |
| `graph/asymmetric.rs:619` | `rank_by_connectivity()` |

9 additional `pub` methods in MCP DTOs (`is_target()`, `with_content()`, `empty()`, `compute_total_score()`, `is_incoming()`, etc.) are only called by their own unit tests.

---

## Section 2: Broken / Fake Implementations

### 2.1 CRITICAL: Health Monitoring Always Reports "Active" (Production Default)

**Location:** `crates/context-graph-core/src/monitoring.rs:530-601`
**Used at:** `crates/context-graph-mcp/src/server.rs:465-466`

`StubLayerStatusProvider` is hardcoded to return `LayerStatus::Active` for ALL four layers (L1_Sensing, L3_Memory, L4_Learning, L5_Coherence) and `health_check_passed: Some(true)` for all. It performs **zero actual health checks**.

This stub is the **production default** -- the MCP server creates it directly:
```rust
let layer_status_provider: Arc<dyn LayerStatusProvider> =
    Arc::new(StubLayerStatusProvider::new());
```

The `get_memetic_status` MCP tool reports these results to users. Users see "active" for every layer regardless of actual state -- even if embedding models fail to load, the store is corrupted, or the learning loop crashes.

### 2.2 CRITICAL: InMemory Store Silently Drops All Audit/Merge/Importance Data

**Location:** `crates/context-graph-core/src/stubs/teleological_store_stub/trait_impl.rs:885-981`

The `InMemoryTeleologicalStore` (used as the default test backend) implements these methods as no-ops:
- `append_audit_record` -- returns `Ok(())`, data discarded
- `get_audit_by_target` -- always returns empty Vec
- `append_merge_record` -- returns `Ok(())`, data discarded
- `get_merge_history` -- always returns empty
- `append_importance_change` -- returns `Ok(())`, data discarded
- `store_embedding_version` -- returns `Ok(())`, data discarded
- `store_custom_weight_profile` -- returns `Ok(())`, data discarded

**Impact:** Any test using `InMemoryTeleologicalStore` will silently succeed for all audit/merge/importance operations. The data goes into a black hole. Tests cannot detect regressions in these subsystems.

### 2.3 HIGH: Audit Log Errors Silently Swallowed in 18+ MCP Handlers

**Pattern:** `if let Err(e) = self.teleological_store.append_audit_record(...) { warn!(...); }`

Found in: `consolidation_tools.rs:403`, `causal_discovery_tools.rs:735`, `curation_tools.rs:105,270`, `maintenance_tools.rs:54`, `embedder_tools.rs:260,641,1299`, `file_watcher_tools.rs:412,564`, `memory_tools.rs:405,1658`, `graph_tools.rs:807,954,1025`, `topic_tools.rs:549`, `merge.rs:411,441`

**Impact:** If the RocksDB audit column family is corrupted or out of disk space, the system happily continues with gaps in the audit trail and no error surfaced to callers.

### 2.4 HIGH: SteeringSystem and FeedbackLearner Are Complete Dead Code

**SteeringSystem** (`crates/context-graph-core/src/steering/`): Full implementation with Gardener/Curator/Assessor, reward computation, pruning threshold checks. Never used outside `#[cfg(test)]` blocks.

**FeedbackLearner** (`crates/context-graph-core/src/teleological/services/feedback_learner/`): Gradient-based learning with momentum, per-embedder adjustments, event buffering across 5 source + 5 test files. Never referenced from MCP or any production code.

Both give the impression of active functionality that does not exist.

### 2.5 HIGH: SystemMonitor Infrastructure Built But Never Wired

**Location:** `crates/context-graph-core/src/monitoring.rs:7-12, 450-505`

The module's own documentation acknowledges hardcoded values that need replacement:
```
- coherence_recovery_time_ms: 8500
- attack_detection_rate: 0.97
- false_positive_rate: 0.015
```

The `SystemMonitor` trait exists with error types and a stub, but NO real implementation exists. The trait is never constructed, never injected, never used in production.

### 2.6 MEDIUM: Stale Doc Comments Cite Wrong Thresholds

**Location:** `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs:1694-1698`

Doc comment says:
```
/// - E5 >= 0.94 -> "definitely causal" -> 1.05x boost
/// - E5 <= 0.92 -> "definitely non-causal" -> 0.90x demotion
```

Actual constants (after LoRA training recalibration):
- `CAUSAL_THRESHOLD = 0.30` (not 0.94)
- `NON_CAUSAL_THRESHOLD = 0.22` (not 0.92)
- `CAUSAL_BOOST = 1.10` (not 1.05)
- `NON_CAUSAL_DEMOTION = 0.85` (not 0.90)

A future developer reading these docs will have a fundamentally wrong understanding of gate behavior.

### 2.7 LOW: Gardener.evaluate() Ignores edge_count Parameter

**Location:** `crates/context-graph-core/src/steering/gardener.rs:67-78`

Accepts `_edge_count: usize` (leading underscore = intentionally unused), hardcodes `edges_pruned: 0`. Combined with the steering system being dead code, this is doubly inert.

### 2.8 LOW: compact() Log Message Reports Wrong Count

**Location:** `crates/context-graph-core/src/stubs/teleological_store_stub/trait_impl.rs:177-191`

After removing entries, logs `self.deleted.len()` as "removed count" -- but this is the **remaining** count after removal, not the removed count.

---

## Section 3: Redundant / Duplicate Code

### 3.1 HIGH: Cosine Similarity Implemented 15+ Times

The `cosine_similarity` function exists as **15+ independent implementations** across the codebase, each with subtly different behavior:

| Location | Return | Error Handling |
|----------|--------|---------------|
| `core/similarity/dense/primitives.rs:111` | `Result<f32>` | Typed errors (canonical) |
| `core/similarity/dense/simd.rs:40` | `Result<f32>` | AVX2 SIMD optimized |
| `core/retrieval/distance.rs:35` | `f32` | Normalizes [-1,1] to [0,1] |
| `core/embeddings/vector.rs:99` | `f32` | Method on DenseVector |
| `benchmark/util.rs:9` | `f32` | Raw standalone |
| `benchmark/causal_bench/metrics.rs:194` | `f32` | Clamps [-1,1] |
| `benchmark/metrics/e1_semantic.rs:484` | `f64` | Different precision |
| `embeddings/warm/inference/validation.rs:188` | `f32` | Production code! |
| `embeddings/storage/types/index_entry.rs:78` | `f32` | Panics on mismatch |
| `embeddings/types/embedding/operations.rs:101` | `EmbeddingResult` | Wrapped result |
| `storage/teleological/indexes/metrics.rs:137` | `f32` | Panics on mismatch |
| `mcp/handlers/merge.rs:1090` | `f32` | Local copy in production handler |

**Behavioral inconsistency:** Zero-vector handling varies (return 0.0 vs Err vs divide-by-zero). The `distance.rs` version normalizes to [0,1] while others return raw [-1,1]. The benchmark uses f64 while production uses f32.

### 3.2 MEDIUM-HIGH: MaxSim Implemented 8 Times

The MaxSim (ColBERT late interaction) algorithm has 8 independent implementations. Critically, `default_engine.rs:121` uses **dot product** instead of cosine similarity -- a behavioral divergence that produces different results for non-normalized vectors.

### 3.3 MEDIUM: Phantom Configuration Options (9 Fields That Do Nothing)

These config fields are parsed from user configuration but **never checked by production code**:

| Config Field | Default | Actual Effect |
|-------------|---------|---------------|
| `feature_flags.dream_enabled` | false | None -- never checked in production |
| `feature_flags.neuromodulation_enabled` | false | None |
| `feature_flags.active_inference_enabled` | false | None |
| `feature_flags.immune_enabled` | false | None |
| `feature_flags.utl_enabled` | true | None |
| `cuda.enabled` | false | None -- contradicts compile-time enforcement |
| `embedding.dimension` | 1536 | None -- doesn't match any embedder (E1=1024) |
| `embedding.max_input_length` | 8191 | None |
| `utl.default_emotional_weight` | 0.5 | None |

A user could set `dream_enabled = true` in their config and nothing would change. The config appears to offer functionality that does not exist.

### 3.4 MEDIUM: Deprecated Code Still Compiled (~250 lines)

| Item | Location | Deprecated Since |
|------|----------|-----------------|
| `Embedder::Graph` alias | `core/teleological/embedder.rs:127` | 0.9.0 |
| `fuse()` method (~100 lines) | `core/teleological/services/fusion_engine.rs:203` | 5.0.0 |
| `fuse_with_profile()` | `core/teleological/services/fusion_engine.rs:280` | 5.0.0 |
| `recency_boost` field + method | `core/traits/teleological_memory_store/options.rs:1250` | 6.1.0 |
| Entire `projections.rs` module | `embeddings/models/pretrained/contextual/projections.rs` | 0.2.0 |
| `StubVectorOps` | `cuda/src/stub.rs:43` | 0.1.0 |

### 3.5 MEDIUM: Normalize Function Duplicated 9 Times

Similar to cosine similarity, `normalize` for vectors exists in 9 independent copies across core, embeddings, storage, and MCP crates.

### 3.6 LOW: Redundant Re-Exports

`{Domain, EdgeType, NeurotransmitterWeights}` re-exported 7 times across the module hierarchy. `NodeId` has similar re-export chains. Within `context-graph-graph` alone, the same types are re-exported at 4 different levels.

### 3.7 LOW: 50+ `#[allow(dead_code)]` Suppressions

Many in production code paths including `graph/storage/migrations.rs`, `graph/query/graph.rs`, `core/embeddings/provider.rs`, and `cli/commands/hooks/mod.rs`. These suppress warnings about genuinely dead code rather than removing it.

---

## Section 4: Test Quality Issues

### 4.1 CRITICAL: 3 Critical Files (2,978 Lines) With Zero Tests

| File | Lines | Functionality |
|------|-------|---------------|
| `mcp/handlers/tools/consolidation.rs` | 452 | Core `trigger_consolidation` MCP tool -- causal-aware memory merging |
| `mcp/handlers/tools/causal_discovery_tools.rs` | 927 | LLM-powered causal discovery with Hermes-2-Pro-Mistral-7B |
| `storage/teleological/rocksdb_store/search.rs` | 1,599 | Core multi-space search execution against RocksDB |

These files implement some of the most complex logic in the system and have zero unit tests.

### 4.2 HIGH: Validation Tests That Don't Test the Validator

**Location:** `mcp/handlers/tools/memory_tools.rs:2164, 2184`

`rationale_validation_boundary_cases` tests `empty.len() < MIN_RATIONALE_LEN` -- this tests **Rust's string length function**, not the actual validation in `call_store_memory`. If someone removes validation from the handler, these tests still pass.

Similarly, `topk_validation_boundary_cases` asserts `0 < MIN_TOP_K` (i.e., `0 < 1`), testing arithmetic.

### 4.3 HIGH: Circular Constant-Value Tests

Tests at `memory_tools.rs:2150-2157` assert `MIN_RATIONALE_LEN == 1` and `MAX_TOP_K == 100` -- asserting constants equal their own values. These cannot detect behavioral regressions; they only fail if someone changes a constant without updating the test.

Tests at `asymmetric.rs:2065` assert `CAUSAL_THRESHOLD == 0.30` -- same pattern.

### 4.4 MEDIUM: Tests With Only `is_ok()` Assertions (11+ Tests)

These tests call functions and only verify `assert!(result.is_ok())` without inspecting the returned value:

- `quantization/batch.rs:57,69` -- `test_batch_quantize`/`test_batch_dequantize` never verify round-trip preserves values
- `embeddings/models/factory/tests/` -- 3 tests verify model creation succeeded but never check dimensions, type, or config
- `similarity/tests.rs:405,455` -- score-in-range tests that would pass for any function returning [0,1]
- `middleware/validation.rs:310,332` -- validate returns Ok without checking the validated value

### 4.5 MEDIUM: Tests That Test Nothing (No Assertions)

| Test | Issue |
|------|-------|
| `topic.rs:441` `test_print_portfolio_empty` | Comment: "Just ensure it doesn't panic" -- no assertions |
| `topic.rs:450` `test_print_stability_dream_recommended` | Same |
| `divergence.rs:210` `test_print_divergence_aligned` | Same |
| `divergence.rs:222` `test_print_divergence_divergent_verbose` | Same |
| `retriever.rs:360` `test_retriever_component_access` | `let _` bindings, no assertions |

### 4.6 MEDIUM: Missing Serialization Round-Trip Tests for Asymmetric Fields

The asymmetric E5 fields (`e5_causal_as_cause`, `e5_causal_as_effect`, `e8_graph_as_source`, `e8_graph_as_target`, `e10_multimodal_as_context`) are newer additions to `SemanticFingerprint`. Given the documented lesson that "bincode + skip_serializing_if = BROKEN", these fields have no dedicated round-trip tests to verify they survive serialization.

### 4.7 MEDIUM: No Causal Gate Integration Test

Unit tests verify the gate function in isolation, but no integration test:
1. Stores a memory with a known causal E5 score
2. Searches with a causal query
3. Verifies the gate correctly boosts/demotes the result in the actual search pipeline

---

## Section 5: Prioritized Remediation Plan

### Priority 1 (Highest Impact, Lowest Risk)

| # | Action | Lines Saved | Risk |
|---|--------|-------------|------|
| 1a | Replace `StubLayerStatusProvider` with real checks or remove `get_memetic_status` tool | 0 | LOW |
| 1b | Fix stale doc comments at `memory_tools.rs:1694-1698` | 5 | NONE |
| 1c | Remove `TRAIN_CAUSAL_EMBEDDER` constant and `training.rs` definitions module | ~50 | NONE |
| 1d | Remove dead feature flags (`onnx`, `cudnn`, `integration`) from Cargo.toml | ~5 | NONE |

### Priority 2 (Add Missing Tests)

| # | Action | Coverage Gap |
|---|--------|-------------|
| 2a | Add unit tests for `consolidation.rs` | 452 lines untested |
| 2b | Add unit tests for `causal_discovery_tools.rs` | 927 lines untested |
| 2c | Add unit tests for `search.rs` (storage layer) | 1,599 lines untested |
| 2d | Add serialization round-trip tests for asymmetric fields | Data loss risk |
| 2e | Add causal gate integration test | End-to-end gap |
| 2f | Fix validation tests to test actual handler, not string length | False confidence |

### Priority 3 (Dead Code Removal)

| # | Action | Lines Removed |
|---|--------|---------------|
| 3a | Remove `core::steering` module | ~1,444 |
| 3b | Remove `core::atc` module | ~3,789 |
| 3c | Remove `core::session` placeholder | ~3 |
| 3d | Evaluate whether `context-graph-graph` should be removed or integrated | ~29,741 |
| 3e | Remove dead functions in benchmark binaries | ~800 |

### Priority 4 (Consolidate Duplicates)

| # | Action | Benefit |
|---|--------|---------|
| 4a | Consolidate cosine_similarity to single canonical implementation | Eliminate 15 copies, fix behavioral inconsistency |
| 4b | Consolidate MaxSim to single implementation | Eliminate 8 copies, fix dot-product vs cosine divergence |
| 4c | Remove deprecated code (fusion methods, projections, etc.) | ~250 lines |
| 4d | Remove or implement phantom config options | Reduce user confusion |

### Priority 5 (Improve Existing Tests)

| # | Action |
|---|--------|
| 5a | Add assertions to no-panic-only tests |
| 5b | Replace `is_ok()` assertions with value verification |
| 5c | Add round-trip assertions to quantization batch tests |
| 5d | Remove duplicate constant-value tests across crates |

---

## Appendix: Compiler Warnings (Current)

The following warnings are emitted by `cargo check`:

| File | Warning |
|------|---------|
| `device.rs:19` | Unused import `CUresult` |
| `benchmark_causal_large.rs:27,34` | Fields `title`, `source_dataset` never read |
| `benchmark_graph_code.rs:13,16,19` | Unused imports (chrono, MemoryForGraphAnalysis, uuid) |
| `benchmark_graph_code.rs:469` | Use of deprecated `with_config` |
| `benchmark_causal_enhanced.rs:18,31` | Unused import, unused field `id` |
| `e1_semantic.rs:560,644,663,877,879` | Unused variables, unnecessary `mut` |
| `mod.rs:137` | Use of deprecated `with_config` |
| `sparse.rs:192,329` | Unused variables |
| `causal_hint.rs:88,97` | Dead methods `with_min_confidence`, `load_model` |
| `names.rs:128` | Unused constant `TRAIN_CAUSAL_EMBEDDER` |
| `embedder_dtos.rs:485` | Dead field `include_details` |
