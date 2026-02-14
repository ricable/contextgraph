# Context Graph Optimization & Integration Report

**Date**: 2026-02-14
**Branch**: casetrack
**Scope**: Full codebase analysis across 10 crates, 1,318 files, ~431K LOC

## Executive Summary

The Context Graph system is architecturally sound with clear DDD boundaries, a sophisticated 13-embedder fingerprinting pipeline, and a well-tested MCP interface (55 tools). The core innovations — multi-space fusion, asymmetric embedders, causal gating — are solid and well-integrated.

However, 6 significant optimization opportunities exist that could **eliminate ~3,500+ LOC of duplication**, **reduce binary size by ~50%**, **cut startup time**, and **improve runtime performance** without any architectural changes. These are all integration/consolidation wins on existing code.

---

## 1. MCP Handler Boilerplate (~2,400 LOC recoverable)

**Problem**: Every MCP tool handler repeats the same 3-phase pattern:
1. Parse request DTO from JSON (`serde_json::from_value` + error handling = ~8 lines)
2. Validate request (`request.validate()` + error handling = ~6 lines)
3. Embed query + search (`multi_array_provider.embed_all` + options + error = ~16 lines)

This is copy-pasted across 30+ tool handlers in `handlers/tools/*.rs`.

**Impact**: ~2,400 LOC of pure duplication across `memory_tools.rs` (2,590 lines), `entity_tools.rs` (1,802 lines), `embedder_tools.rs` (1,362 lines), etc. Three files exceed the 500-line guideline by 2.7x-5.2x.

**Solution**: Extract 3 helper methods on `Handlers`:

```rust
// 1. Parse + validate (replaces 14 lines × 30 tools = 420 LOC)
async fn parse_request<T: DeserializeOwned + Validatable>(
    &self, id: Option<JsonRpcId>, args: Value
) -> Result<T, JsonRpcResponse>

// 2. Embed + search (replaces 16 lines × 15 tools = 240 LOC)
async fn embed_and_search(
    &self, query: &str, strategy: SearchStrategy, profile: &str, top_k: usize
) -> Result<Vec<SearchResult>, String>

// 3. Apply boost + collect (replaces 8 lines × 8 tools = 64 LOC)
fn boost_and_collect<T: From<SearchResult>>(
    results: Vec<SearchResult>, boost: BoostStrategy, min_score: f32, top_k: usize
) -> Vec<T>
```

**Effort**: 2-3 days | **LOC saved**: ~2,000 | **Risk**: Low

---

## 2. Binary Size: 587MB MCP Server (50%+ reducible)

**Problem**: The release binary is **587MB**. Benchmark binaries are 550-585MB each. The `target/` directory is **232GB** (184G debug + 48G release).

**Root causes**:
- LLM inference libraries (llama-cpp-2) statically linked into every binary
- All 13 embedder models compiled into every binary (even benchmarks that test 1 embedder)
- No `strip` or LTO optimization visible in benchmarks
- 40+ benchmark binaries, each linking the full stack

**Solutions**:

| Action | Binary Reduction | Effort |
|--------|-----------------|--------|
| Feature-gate LLM (llama-cpp-2) behind `llm` feature | ~500MB → ~80MB for non-LLM binaries | 1 day |
| Split benchmark crate into per-domain sub-crates | Each bench drops from 550MB to ~30MB | 2 days |
| `cargo clean` debug artifacts | Recovers 184GB disk | 1 min |
| Verify `strip = true` and `lto = "thin"` in `[profile.release]` | ~10-20% further reduction | 30 min |

**Effort**: 2-3 days | **Impact**: Massive disk + deploy improvement

---

## 3. Embedder Model Duplication (~1,500 LOC abstractable)

**Problem**: 10 pretrained embedder models (Semantic, Causal, Code, Graph, Multimodal, Entity, Kepler, LateInteraction, Sparse×2) each implement `EmbeddingModel` with nearly identical boilerplate:
- `embed()` with batch pipeline → ~50 lines each
- `is_loaded()` / `unload()` → ~10 lines each
- Factory match arm → ~3 lines each

The `models/pretrained/` directory has 143 files with repeated patterns.

**Solution**: Generic `HfEmbedder<C: EmbedderConfig>` that handles the common pipeline:

```rust
pub struct HfEmbedder<C: EmbedderConfig> {
    model: Option<CandleModel>,
    tokenizer: Tokenizer,
    config: C,
}

impl<C: EmbedderConfig> EmbeddingModel for HfEmbedder<C> {
    fn embed(&self, inputs: &[ModelInput]) -> Result<Vec<Embedding>> {
        // Generic: tokenize → batch → forward → pool → project
        // Config trait provides: pool_strategy, projection_dim, max_tokens
    }
}
```

Each embedder becomes a `type SemanticEmbedder = HfEmbedder<SemanticConfig>;` with only the config-specific differences (pooling strategy, dimension, projection).

**Effort**: 3-5 days | **LOC saved**: ~1,200 | **Risk**: Medium (needs careful testing)

---

## 4. Storage Layer: Missing Garbage Collection + Startup Bottleneck

### 4a. Soft-Delete Garbage Collection (MISSING)

**Problem**: Soft-deleted memories persist in RocksDB forever. The in-memory `HashMap<Uuid, bool>` grows unbounded. All 50 column families still contain the deleted data. Inverted index posting lists (E6/E13) still reference soft-deleted UUIDs.

**Solution**: Background GC task on a configurable interval (e.g., hourly):
1. Scan `CF_SYSTEM` soft-delete markers older than retention period
2. Hard-delete from all CFs (`CF_FINGERPRINTS`, `CF_EMB_0`..`CF_EMB_12`, etc.)
3. Remove from inverted index posting lists
4. Trigger RocksDB compaction for reclaimed CFs

**Effort**: 2 days | **Impact**: Prevents unbounded storage growth

### 4b. HNSW Index Rebuild on Startup

**Problem**: Every `store.open()` does a full O(n) scan of `CF_FINGERPRINTS` to rebuild 12 HNSW indexes. At 100K fingerprints this is ~1-5 seconds; at 1M it's ~10-30 seconds. During rebuild, search returns empty results.

**Solution**: Persist HNSW graph structures to dedicated CFs (`CF_HNSW_GRAPH_E1`, etc.). On startup, deserialize instead of rebuild. Only rebuild on version mismatch or corruption.

**Effort**: 3-5 days | **Impact**: Startup goes from seconds to milliseconds

### 4c. Inverted Index O(n) Contains Check

**Problem**: When inserting a UUID into a posting list, the code does `if !ids.contains(id)` which is O(n) linear scan. For popular terms with 1000+ entries, this becomes measurable.

**Solution**: Use `HashSet<Uuid>` for posting lists internally, serialize to `Vec<Uuid>` on write. Or maintain sorted `Vec` with binary search.

**Effort**: 0.5 days | **Impact**: Measurable write throughput improvement at scale

---

## 5. Causal + Graph Agent Unification

**Problem**: `context-graph-causal-agent` (5,026 LOC) and `context-graph-graph-agent` (3,997 LOC) share ~70% of their structure:
- Both use llama-cpp-2 with GBNF grammars for structured JSON output
- Both have `Service`, `Activator`, `LLM`, `Scanner` modules with identical patterns
- Both load Hermes-2-Pro-Mistral-7B
- Graph agent already depends on causal agent

**Duplication**: ~2,000 LOC of shared LLM integration, service lifecycle, and activation patterns.

**Solution**: Extract shared `LlmAgentFramework` into `context-graph-core` or a new `context-graph-llm-common` crate:

```rust
pub trait LlmAgent {
    type Input;
    type Output;
    fn grammar(&self) -> &str;           // GBNF grammar for structured output
    fn prompt(&self, input: &Self::Input) -> String;
    fn parse(&self, raw: &str) -> Result<Self::Output>;
}

pub struct LlmAgentRunner<A: LlmAgent> {
    model: LlamaModel,      // Shared model instance
    agent: A,
}
```

Both causal and graph agents become thin `impl LlmAgent` with just their prompt templates and parsing logic. The model can be shared (single load of Hermes-2-Pro) instead of loaded twice.

**Effort**: 3-4 days | **LOC saved**: ~1,500 | **Bonus**: Single model load saves ~4GB VRAM

---

## 6. Search Strategy: Missing Unified Pipeline

**Problem**: The 3 search strategies (E1Only, MultiSpace, Pipeline) are implemented as large `match` arms in `search.rs` (~600 LOC). Adding a new strategy or modifying pipeline stages requires touching this monolithic function.

Additionally, the graph edge rebuild happens on-demand (first graph_link_tools call after restart) rather than at startup, creating a race with search availability.

**Solution**: `SearchStage` trait with composable pipeline:

```rust
pub trait SearchStage {
    async fn execute(&self, ctx: SearchContext) -> Result<SearchContext>;
}

// E1Only = [HnswSearch(E1)]
// MultiSpace = [MultiSpaceHnsw([E1,E5,E7,E8,E10,E11]), RRFFusion]
// Pipeline = [SparseRecall(E13), MultiSpaceScore, ColBertRerank(E12)]
```

Also: move graph edge rebuild to startup (alongside HNSW rebuild) instead of on-demand in `ensure_graph_edges_built()`.

**Effort**: 3-5 days | **Impact**: Extensibility + eliminates first-query delay

---

## What's Already Well-Integrated

These areas are solid and need no changes:

| Area | Assessment |
|------|-----------|
| **13-Embedder Fingerprinting** | Clean architecture. Each embedder has clear purpose. Asymmetric handling (E5/E8/E10) is correct. |
| **Weight Profile System** | 14 profiles well-tuned. Custom profile support works. AP-71 (temporal=0 in semantic) enforced. |
| **RRF Fusion** | Research-backed (k=60). Score-weighted variant for E5 magnitude. Correct implementation. |
| **Causal Gate** | Thresholds calibrated (TPR=83.9%, TNR=84.3%). Boost/demotion applied correctly. |
| **Error Handling** | Consistent FAIL FAST philosophy. Detailed error types with RocksDB operation context. |
| **MCP Protocol** | JSON-RPC 2.0 compliant. 55 tools well-categorized. Tool registry with O(1) lookup. |
| **Soft-Delete Persistence** | Fixed (BUG-1). Persists to CF_SYSTEM. Loaded at startup. |
| **Column Family Organization** | 50 CFs logically grouped. Schema documented. Version-prefixed serialization. |
| **Provenance System** | Audit logs, merge history, importance history all append-only. JSON serialization. |
| **Test Infrastructure** | 777 test files. 84 benchmark runs. Criterion.rs integration. |

---

## Priority Ranking

| # | Optimization | LOC Impact | Performance Impact | Effort | Risk |
|---|-------------|-----------|-------------------|--------|------|
| 1 | MCP handler helpers | -2,000 | Maintainability | 2-3d | Low |
| 2 | Feature-gate LLM for binary size | — | -500MB per binary | 1-2d | Low |
| 3 | Soft-delete GC | +200 | Prevents storage leak | 2d | Low |
| 4 | Inverted index HashSet | ~0 | Write throughput | 0.5d | Low |
| 5 | Causal/Graph agent unification | -1,500 | -4GB VRAM | 3-4d | Medium |
| 6 | Generic HfEmbedder | -1,200 | Maintainability | 3-5d | Medium |
| 7 | HNSW persistence | +500 | Startup time | 3-5d | Medium |
| 8 | SearchStage pipeline | ~0 | Extensibility | 3-5d | Medium |

**Total recoverable LOC**: ~4,700
**Total effort**: ~18-27 days
**Recommended first sprint**: Items 1-4 (6.5 days, low risk, high impact)

---

## Compiler Warnings to Clean Up

From the current build diagnostics:

| File | Warning | Fix |
|------|---------|-----|
| `mod.rs:983,1000,1018` | Deprecated `CausalDiscoveryService::new()` | Use `with_models()` |
| `benchmark_causal_enhanced.rs:18` | Unused import `CausalLinkDirection` | Remove |
| `benchmark_causal.rs:12` | Unused import `Duration` | Remove |
| `benchmark_causal.rs:28-34` | Dead code: `id`, `doc_id`, `word_count` fields | Prefix with `_` or use |
| `stubs.rs:70` | Deprecated `GraphDiscoveryService::with_config` | Use `with_models()` |
| `runner.rs:740` | Unused import `super::*` | Remove |
| `entity_dtos.rs:59` | Unused constant `DEFAULT_E11_WEIGHT` | Remove or use |

These are all trivial fixes (< 30 min total).
