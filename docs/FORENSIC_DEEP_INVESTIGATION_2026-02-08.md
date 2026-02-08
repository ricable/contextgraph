# Forensic Deep Investigation Report

**Date**: 2026-02-08
**Branch**: casetrack
**Scope**: Full codebase investigation for code that appears to work but is actually broken
**Agents Deployed**: 7 parallel investigation agents
**Total Findings**: 73

---

## Executive Summary

A comprehensive forensic investigation of the entire context-graph codebase revealed **73 findings** across 6 investigation domains. The most critical discoveries fall into three categories:

1. **Facade Services** (2 CRITICAL): Both background discovery services (causal + graph) are elaborate facades -- they start, report status, accept configuration, handle graceful shutdown, but their core loops are empty. No autonomous discovery ever occurs.

2. **Data Integrity** (2 CRITICAL): `store_batch_async()` skips HNSW indexing (making batch-stored memories unsearchable), and `compact_async()` resurrects soft-deleted data by draining the tracking HashMap without hard-deleting from RocksDB.

3. **Phantom Parameters**: Multiple MCP tool parameters are parsed, validated, echoed in responses, but have zero effect on behavior -- creating an illusion of user control.

### Severity Distribution

| Severity | Count | Description |
|----------|-------|-------------|
| CRITICAL | 6 | System-breaking: no-op services, data loss, unsearchable data |
| HIGH | 16 | Significant: silent failures, race conditions, DoS vectors |
| MEDIUM | 28 | Moderate: phantom params, biased queries, degraded quality |
| LOW | 20 | Minor: dead code, stale docs, unnecessary unsafe |
| INFO | 3 | Confirmed working / previously fixed |

---

## Table of Contents

1. [Critical Findings](#1-critical-findings)
2. [MCP Tool Handler Findings](#2-mcp-tool-handler-findings)
3. [Storage Layer Findings](#3-storage-layer-findings)
4. [Core Logic & Trait Findings](#4-core-logic--trait-findings)
5. [Agent & Service Findings](#5-agent--service-findings)
6. [Test Suite Findings](#6-test-suite-findings)
7. [Build, Config & Safety Findings](#7-build-config--safety-findings)
8. [Remediation Priority](#8-remediation-priority)

---

## 1. Critical Findings

These 6 findings represent code that fundamentally does not do what it claims.

### CRIT-01: Causal Agent Background Discovery Loop is a No-Op

- **File**: `crates/context-graph-causal-agent/src/service/mod.rs:480-484`
- **Impact**: The entire background causal discovery feature is non-functional
- **Details**: `CausalDiscoveryService::start()` spawns a tokio task with an interval loop. The loop body only emits `debug!("Discovery cycle tick (no memories provided)")` and returns to sleep. It never calls `run_discovery_cycle()`, never fetches memories, never invokes the scanner or LLM. The service reports `ServiceStatus::Running` while doing nothing.

### CRIT-02: Graph Agent Background Discovery Loop is a No-Op

- **File**: `crates/context-graph-graph-agent/src/service/mod.rs:443-447`
- **Impact**: The entire background graph discovery feature is non-functional
- **Details**: Identical pattern to CRIT-01. Loop body only emits `debug!("Background discovery cycle would run here")`. Comment explicitly admits: "In background mode, we would load memories from storage."

### CRIT-03: `store_batch_async` Skips HNSW Index Insertion

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/persistence.rs:40-56`
- **Impact**: Batch-stored fingerprints are invisible to all search strategies until server restart
- **Details**: `store_batch_async` calls `store_fingerprint_internal` (RocksDB write) but never calls `add_to_indexes` (HNSW insertion). Compare with `store_async` in `crud.rs:34-49` which correctly calls both. Batch-stored memories persist in RocksDB but are unsearchable until `rebuild_indexes_from_store()` runs on next restart.

### CRIT-04: `compact_async` Resurrects Soft-Deleted Data

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/persistence.rs:330-333`
- **Impact**: Soft-deleted data reappears after compaction
- **Details**: Soft delete (`crud.rs:138`) only inserts into the in-memory `soft_deleted` HashMap -- it does NOT delete from RocksDB. When `compact_async` drains the HashMap (`self.soft_deleted.write().drain()`), it removes the ONLY record that these entries are deleted. After compaction, soft-deleted fingerprints reappear in search results. Compaction should either hard-delete from RocksDB or NOT drain the HashMap.

### CRIT-05: Non-Deterministic RRF Ranks in Causal Multi-Embedder Search (Stub)

- **File**: `crates/context-graph-core/src/stubs/teleological_store_stub/trait_impl.rs:789-821`
- **Impact**: Multi-embedder causal search returns non-deterministic results
- **Details**: Score tuples are collected into `HashMap<Uuid, f32>`, destroying sort order. RRF ranks are then assigned via `HashMap::enumerate()`, but HashMap iteration order is undefined in Rust (random seed). Results vary between runs for identical input.

### CRIT-06: `with_embedders()` is a No-Op in Production (RocksDB) Search

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/search.rs` (entire file)
- **Impact**: All embedder-specific searches actually search E1 only
- **Details**: The `embedder_indices` field is set on `TeleologicalSearchOptions` but the RocksDB search implementation never reads it. All three search functions (`search_e1_only_sync`, `search_multi_space_sync`, `search_pipeline_sync`) hardcode which embedders they query. When `search_by_embedder` requests E11-only search, the actual execution hits E1 HNSW. Results are then re-scored by E11 via `compute_embedder_scores_sync`, but this misses candidates that E11 would have found but E1 didn't.

---

## 2. MCP Tool Handler Findings

### MCP-01 (HIGH): `sessionScope` Parameter Parsed But Never Used to Filter

- **File**: `crates/context-graph-mcp/src/handlers/tools/causal_discovery_tools.rs:79-143`
- **Details**: `sessionScope` ("current", "all", "recent") is parsed, validated, logged, and echoed in response. But memory collection at lines 122-143 iterates `indexed_files` unconditionally. The parameter has zero effect on behavior.

### MCP-02 (HIGH): "Semantic" Consolidation Strategy Selects ALL Pairs

- **File**: `crates/context-graph-mcp/src/handlers/tools/consolidation.rs:261, 316-332`
- **Details**: `alignment` is hardcoded to `1.0` for every memory (line 261). The "semantic" strategy checks `alignment >= 0.5` which is always true. For 100 memories, this produces 4,950 pairs regardless of actual semantic similarity.

### MCP-03 (HIGH): Custom Weight Profile Strategy Override Bug

- **File**: `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs:1083-1091`
- **Details**: After applying custom weights via `with_custom_weights()`, code applies the ORIGINAL strategy variable instead of forcing `MultiSpace`. If user specified `e1_only` plus a custom profile, the custom weights are silently ignored because E1Only only searches E1 space.

### MCP-04 (MEDIUM): `averageConfidence` is Fabricated

- **File**: `crates/context-graph-mcp/src/handlers/tools/causal_discovery_tools.rs:284-291`
- **Details**: Formula: `min_confidence + (1.0 - min_confidence) * 0.5`. Always returns midpoint between min_confidence and 1.0 (default: 0.85). Not computed from actual relationship data.

### MCP-05 (MEDIUM): Consolidation Uses Dot Product, Not Cosine Similarity

- **File**: `crates/context-graph-mcp/src/handlers/tools/consolidation.rs:100-108, 280-285`
- **Details**: Raw dot product used instead of cosine similarity. For non-normalized embeddings, the 0.85 threshold becomes meaningless (dot products can exceed 1.0 for 1024D vectors).

### MCP-06 (MEDIUM): `filterGraphDirection` Parsed But Never Filters

- **File**: `crates/context-graph-mcp/src/handlers/tools/graph_tools.rs:243, 259-261`
- **Details**: `graph_direction` is always `None` (hardcoded at line 183). Filter condition is never satisfied. Code comment: "Direction inference requires fingerprint retrieval which is not yet implemented."

### MCP-07 (MEDIUM): `minConfidence` Ignored in "pairs" Mode

- **File**: `crates/context-graph-mcp/src/handlers/tools/causal_discovery_tools.rs:78, 268`
- **Details**: Parsed and validated but never passed to `service.run_discovery_cycle()`. Only affects "extract" mode (line 529).

### MCP-08 (MEDIUM): Hardcoded "conversation context" Query Biases Retrieval

- **File**: `crates/context-graph-mcp/src/handlers/tools/sequence_tools.rs:167`
- **Details**: When no query parameter provided, uses literal string `"conversation context"` as search query. Biases retrieval toward memories semantically similar to that phrase rather than providing unbiased sequence-based retrieval.

### MCP-09 (MEDIUM): Divergence Detection Biased by Hardcoded Query

- **File**: `crates/context-graph-mcp/src/handlers/tools/topic_tools.rs:579`
- **Details**: Uses `"context memory"` as initial query to find recent memories. Memories semantically distant from this phrase may not appear, causing missed divergence alerts.

### MCP-10 (MEDIUM): Consolidation Never Actually Merges

- **File**: `crates/context-graph-mcp/src/handlers/tools/consolidation.rs:390-393`
- **Details**: `trigger_consolidation` only REPORTS candidate pairs but never calls any merge/delete operation. `max_daily_merges: 50` is misleading since zero merges ever happen. It's a read-only analysis tool masquerading as a write operation.

### MCP-11 (LOW): Embedding Version Records Are Static Strings

- **File**: `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs:491-533`
- **Details**: Comment admits: "These are static descriptors, not dynamic model versions." Every fingerprint gets identical strings like `"pretrained-semantic-1024d"`. Stale embedding detection is impossible.

### MCP-12 (LOW): Hardcoded Query Bias in Memory Chain Traversal

- **File**: `crates/context-graph-mcp/src/handlers/tools/sequence_tools.rs:553`
- **Details**: Uses `"memory chain traversal"` as default semantic filter. Same class as MCP-08.

### MCP-13 (LOW): `entropy` Always None in Stability Response

- **File**: `crates/context-graph-mcp/src/handlers/tools/topic_tools.rs:295`
- **Details**: `entropy: None, // Not currently computed`. Feature advertised in schema but never implemented.

---

## 3. Storage Layer Findings

### STG-01 (HIGH): Deserialization Functions Panic on Corruption

- **File**: `crates/context-graph-storage/src/teleological/serialization.rs:126-336`
- **Details**: Every deserialization function panics on any corruption (no Result return). One corrupted fingerprint in RocksDB will crash the ENTIRE process during iteration. Functions like `list_fingerprints_unbiased_async` and `compact_async` iterate ALL fingerprints.

### STG-02 (HIGH): HNSW Index Grows Monotonically (Ghost Vectors)

- **File**: `crates/context-graph-storage/src/teleological/indexes/hnsw_impl/ops.rs:74-87`
- **Details**: usearch does not support vector deletion. `remove()` only removes the UUID-to-key mapping. The vector remains in the HNSW graph forever. Over time: increasing memory usage, degrading search quality (ghost node traversal), fewer results than requested (k*2 buffer may be insufficient).

### STG-03 (HIGH): Lost-Update Race in Causal Relationship Secondary Index

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/causal_relationships.rs:110-160`
- **Details**: Read-modify-write on `causal_by_source` index is non-atomic. Two concurrent stores for the same `source_fingerprint_id` will race; the loser's ID is silently dropped from the index.

### STG-04 (HIGH): Lost-Update Race in Inverted Indexes (E13/E6)

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/inverted_index.rs:38-53, 118-134`
- **Details**: Same race condition as STG-03 but affects many terms per fingerprint. Concurrent fingerprint creation can lose inverted index entries for shared terms.

### STG-05 (HIGH): Pipeline Stage 3 (E12 MaxSim Reranking) is Completely Skipped

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/search.rs:617`
- **Details**: Line 617: `// STAGE 3: Optional E12 re-ranking (skip in sync version for now)`. Despite E12 tokens being stored and MaxSim functions existing, Stage 3 is never executed. Pipeline is functionally identical to MultiSpace with extra overhead.

### STG-06 (HIGH): Silent 0.0 Return on Dimension Mismatch in Release Mode

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/helpers.rs:20-49`
- **Details**: `compute_cosine_similarity` uses `debug_assert_eq!` for dimension check. In release mode, mismatched dimensions silently return 0.0 (maximally dissimilar), suppressing legitimate results and hiding upstream bugs.

### STG-07 (MEDIUM): E12 Tokens Still Serialized with Bincode

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/store.rs:667`
- **Details**: Every other field was migrated to JSON per project policy (bincode + skip_serializing_if = BROKEN). E12 is the last holdout. Not broken today but violates the established standard.

### STG-08 (MEDIUM): `search_sparse_async` Computes `total_docs` Twice

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/search.rs:1146-1169 vs 715-781`
- **Details**: `count_async()` result is never passed to `search_sparse_sync`, which recomputes via full O(n) RocksDB iteration. Two full-table scans instead of one; soft-deleted count discrepancy between the two methods.

### STG-09 (MEDIUM): Duplicate Fingerprint Insert Creates Ghost in usearch

- **File**: `crates/context-graph-storage/src/teleological/indexes/hnsw_impl/ops.rs:34-39`
- **Details**: On duplicate UUID insert, old vector remains in usearch. Every `update_async()` call creates one ghost vector per embedder index (up to 15 ghost vectors per update).

### STG-10 (MEDIUM): Lost-Update Race in File Index

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/file_index.rs:170-227`
- **Details**: Same read-modify-write race as STG-03/04 but without even a WriteBatch. Read and write are entirely separate RocksDB operations, making the race window wider.

### STG-11 (MEDIUM): Pipeline Loads Fingerprints Twice Per Result

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/search.rs:519-527, 625-629`
- **Details**: Stage 2 deserializes full fingerprints for scoring, discards them. Final Results section loads and deserializes them again. Double I/O + TOCTOU race.

### STG-12 (MEDIUM): RRF Score Scaling Uses Magic Number 10.0

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/search.rs:389, 594`
- **Details**: RRF scores (typically 0.0-0.1) multiplied by 10.0. Not mathematically justified. `min_similarity` has different semantics depending on fusion strategy. With many embedders, scaled scores can exceed 1.0.

### STG-13 (MEDIUM): Checkpoint Restore is Permanently Broken

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/persistence.rs:296-313`
- **Details**: After validating path exists, unconditionally returns `Err("In-place restore not supported")`. Checkpoint creation works but restore is a permanent no-op.

### STG-14 (MEDIUM): Silent Search Degradation When Embedder Indexes Fail

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/search.rs:443-501`
- **Details**: In Pipeline Stage 1, E5/E7/E8/E11 failures are silently swallowed by `if let Ok(...)` without even a warning. Only E1 failure is treated as fatal. Corrupted indexes cause silent quality degradation.

### STG-15 (MEDIUM): Inconsistent Error Handling Between Rebuild Methods

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/store.rs:480-607`
- **Details**: `rebuild_indexes_from_store` fails the entire store open on any fingerprint error. `rebuild_causal_e11_index` only logs `warn!` and returns `Ok(())` even with errors. Silent partial index.

### STG-16 (LOW): `search_text_async` Always Returns Error

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/search.rs:1103-1119`
- **Details**: Always returns `Err(CoreError::NotImplemented(...))`. Well-documented but runtime-only discovery.

### STG-17 (LOW): Pipeline `multi_search` Field is Dead Code

- **File**: `crates/context-graph-storage/src/teleological/search/pipeline/execution.rs:33-34`
- **Details**: `#[allow(dead_code)]` field. Stage 3 RRF uses single E1 ranking only. "Multi-space RRF" is misleading.

---

## 4. Core Logic & Trait Findings

### CORE-01 (HIGH): E5/E8/E10 Stubs Use Identical Vectors for Asymmetric Fields

- **File**: `crates/context-graph-core/src/stubs/multi_array_stub.rs:288-302`
- **Details**: `e5_causal_as_cause` and `e5_causal_as_effect` filled with the exact same vector. Same for E8 source/target and E10 local/global. Tests using stubs cannot verify asymmetric search logic.

### CORE-02 (HIGH): Silent No-Op Stubs for Audit, Merge History, Weight Profiles

- **File**: `crates/context-graph-core/src/stubs/teleological_store_stub/trait_impl.rs:877-975`
- **Details**: `append_audit_record`, `append_merge_record`, etc. return `Ok(())`. Getters return `Ok(Vec::new())`. No way to distinguish "persisted" from "silently discarded". Integration tests using stubs cannot detect broken write paths.

### CORE-03 (HIGH): Clustering Support Defaults Return Empty Without Warning

- **File**: `crates/context-graph-core/src/traits/teleological_memory_store/defaults.rs:326-340`
- **Details**: `scan_fingerprints_for_clustering_default` and `list_fingerprints_unbiased_default` return `Ok(Vec::new())`. Any backend that fails to override these gets zero clustering input. HDBSCAN receives empty matrix, returns zero clusters -- no errors, no warnings.

### CORE-04 (HIGH): Entity Search Union is Fake (Consequence of CRIT-06)

- **File**: `crates/context-graph-mcp/src/handlers/tools/entity_tools.rs:511-541`
- **Details**: `search_by_entities` claims to UNION E1 and E11 candidate sets. Due to CRIT-06, both searches execute E1-based search. "E11 discovered X candidates E1 missed" metric always reports 0.

### CORE-05 (MEDIUM): Entity Extraction is Heuristic-Only, No NER

- **File**: `crates/context-graph-mcp/src/handlers/tools/entity_tools.rs:90-123`
- **Details**: Despite docs mentioning "KB-based entity detection" and "KEPLER's capabilities", actual implementation splits on whitespace, checks for uppercase first character, and filters common English words. All entities typed as `EntityType::Unknown`.

### CORE-06 (MEDIUM): Unweighted Average Similarity in Stub Search

- **File**: `crates/context-graph-core/src/stubs/teleological_store_stub/search.rs:52-56`
- **Details**: Stub averages all 13 embedder scores equally. Production uses category-weighted scoring. Temporal embedders E2-E4 (should have weight 0.0) get equal weight. Stub-based tests are unreliable indicators of real behavior.

### CORE-07 (MEDIUM): Compaction Log Always Reports Zero Deletions (Stub)

- **File**: `crates/context-graph-core/src/stubs/teleological_store_stub/trait_impl.rs:177-189`
- **Details**: Loop drains `self.deleted`, then logs `self.deleted.len()` which is always 0 after draining.

### CORE-08 (MEDIUM): Inconsistent Error Strategies in Trait Defaults

- **File**: `crates/context-graph-core/src/traits/teleological_memory_store/defaults.rs:55-211`
- **Details**: Write operations return `Err(Internal)`, but corresponding read operations return `Ok(None)` or `Ok(vec![])`. Readers cannot distinguish "no data" from "not supported".

### CORE-09 (LOW): Hardcoded Query Bias in Entity Graph

- **File**: `crates/context-graph-mcp/src/handlers/tools/entity_tools.rs:1431-1432`
- **Details**: When no `center_entity` provided, defaults to `"code programming framework database"`. Biases toward software-related memories.

### CORE-10 (LOW): Retrieval Tests Only Exercise Stubs

- **File**: `crates/context-graph-core/src/retrieval/tests.rs:13-14`
- **Details**: All tests use `InMemoryTeleologicalStore` + `StubMultiArrayProvider`. Given stub divergences (CORE-01, CORE-06, CRIT-05), these tests create false confidence.

---

## 5. Agent & Service Findings

### AGT-01 (HIGH): Scanner Marks Pairs as Analyzed Before LLM Confirmation

- **File**: `crates/context-graph-causal-agent/src/scanner/mod.rs:245`
- **Details**: Pair inserted into `analyzed_pairs` as soon as it scores above `min_initial_score`, BEFORE LLM analysis. If LLM fails (timeout, error, unavailable), pair is permanently skipped. No retry mechanism.

### AGT-02 (HIGH): CausalGraph is In-Memory Only, No Persistence

- **File**: `crates/context-graph-core/src/causal/scm.rs:230-239`
- **Details**: `CausalGraph` is a pure in-memory `HashMap<Uuid, CausalNode>` + `Vec<CausalEdge>`. The activator adds edges but never writes to RocksDB. All causal relationships lost on restart.

### AGT-03 (HIGH): GraphStorage is In-Memory Only, No Persistence

- **File**: `crates/context-graph-graph-agent/src/activator/mod.rs:119-170`
- **Details**: `GraphStorage` is a plain `Vec<GraphEdge>` with no persistence. Source comment: "In production, this would be backed by persistent storage." All graph relationships lost on restart.

### AGT-04 (HIGH): No Line Length Limit on read_line (OOM DoS)

- **File**: `crates/context-graph-mcp/src/server.rs:625, 1487`
- **Details**: Both stdio and TCP handlers use `reader.read_line()` with no size limit. A client can send a multi-gigabyte line (no newline) causing unbounded memory allocation. TCP path is externally exploitable.

### AGT-05 (MEDIUM): LLM Read Lock Held During Entire Token Generation

- **File**: `crates/context-graph-causal-agent/src/llm/mod.rs:880-998`
- **Details**: parking_lot read lock held for entire generation loop (seconds to minutes). Blocks any write operations on LLM state.

### AGT-06 (MEDIUM): Unicode Panic in `truncate_name`

- **File**: `crates/context-graph-causal-agent/src/activator/mod.rs:577`
- **Details**: `&trimmed[..max_len - 3]` uses byte slicing. Panics if offset falls in middle of multi-byte UTF-8 character. Chinese/Japanese/Korean/emoji content will crash.

### AGT-07 (MEDIUM): FAISS Index Hardcoded to L2 Distance

- **File**: `crates/context-graph-graph/src/index/gpu_index/index.rs:127`
- **Details**: `MetricType::L2` hardcoded. Rankings are correct for normalized vectors, but raw distance VALUES are L2 (0-2 range), not cosine (-1 to 1). Downstream thresholding may be wrong.

### AGT-08 (MEDIUM): Missing Request Timeout for Stdio Transport

- **File**: `crates/context-graph-mcp/src/server.rs:643`
- **Details**: TCP handler has `tokio::time::timeout()`. Stdio handler has none. Hung request blocks forever.

### AGT-09 (MEDIUM): Background Worker Double-Start Race

- **File**: `crates/context-graph-storage/src/graph_edges/builder.rs:683-686`
- **Details**: `swap(true)` logs warning but doesn't prevent continued execution. Second `tokio::spawn` creates duplicate worker processing the same queue.

### AGT-10 (LOW): Chaos Tests Test Accounting, Not Real GPU

- **File**: `crates/context-graph-graph/tests/chaos_tests/*.rs`
- **Details**: All 16 tests use `GpuMemoryManager` (software budget tracker), not real CUDA. Test names like `test_gpu_oom_recovery` imply GPU-level validation that doesn't occur.

---

## 6. Test Suite Findings

### TST-01 (CRITICAL): `evidence_of_e2e_test_coverage` Has Zero Assertions

- **File**: `crates/context-graph-mcp/src/handlers/tests/mcp_protocol_e2e_test.rs:883-909`
- **Details**: Test function contains ONLY `println!()` statements. Prints "EVIDENCE OF SUCCESS" with zero assertions. Always passes regardless of system state.

### TST-02 (CRITICAL): All GPU/CUDA E2E Tests Silently Skipped

- **Files**: `mcp_protocol_e2e_test.rs`, `gpu_embedding_verification.rs`, `search_periodic_test.rs`
- **Details**: All tests gated by `#[cfg(feature = "cuda")]`. In standard `cargo test`, entire test suites are compiled away. Only the `evidence_of_e2e_test_coverage` function (TST-01) runs, printing "ALL VERIFIED" while zero actual tests executed.

### TST-03 (HIGH): `merge_concepts` Auto-Credited Without Testing

- **File**: `crates/context-graph-mcp/src/handlers/tests/mcp_protocol_e2e_test.rs:858`
- **Details**: `merge_concepts` is skipped entirely and hardcoded into `successful_tools` set: `successful_tools.insert("merge_concepts"); // Tested in dedicated test`. Then all 11 tools asserted as successful.

### TST-04 (HIGH): Content Storage Round-Trip Test Always Passes

- **File**: `crates/context-graph-mcp/src/handlers/tests/content_storage_verification.rs:264-313`
- **Details**: If fingerprint NOT found in search results, test prints a note and passes. No `assert!` verifies content was actually found. Test always passes even if content storage is completely broken.

### TST-05 (MEDIUM): Multiple Search Tests Skip Assertions on Empty Results

- **Files**: `content_storage_verification.rs:116-132`, `semantic_search_skill_verification.rs:84,210,274`
- **Details**: Assertions wrapped in `if !results.is_empty()` guards. With stub embeddings, search returns empty results, no assertions run, test prints "PASSED".

### TST-06 (MEDIUM): `minSimilarity` Test Acknowledges It Tests Nothing

- **File**: `crates/context-graph-mcp/src/handlers/tests/semantic_search_skill_verification.rs:352-401`
- **Details**: Comment: "With stub embeddings, all vectors are the same, so similarity is always 1.0". Test asserts nothing about filtering behavior. Always passes.

### TST-07 (MEDIUM): Layer Status Tests Verify Stub Hardcoded Values

- **Files**: `manual_fsv_verification.rs:387-411`, `task_emb_024_verification.rs:50`
- **Details**: Tests verify `StubLayerStatusProvider` which ALWAYS returns "active" for all layers. Tests pass even if real layer status detection is broken.

### TST-08 (MEDIUM): Stub Search Tests with Zeroed Vectors

- **File**: `crates/context-graph-core/src/stubs/teleological_store_stub/tests.rs:84-94`
- **Details**: Queries with zeroed vector against zeroed fingerprints. Only asserts results non-empty. Doesn't verify similarity scores are meaningful.

### TST-09 (LOW): `min_similarity_filter` Test Has No Assertion

- **File**: `crates/context-graph-core/src/stubs/teleological_store_stub/tests.rs:140-148`
- **Details**: Result assigned to `_results` (underscore prefix). No assertion. Test only verifies function doesn't panic.

### TST-10 (LOW): Search Periodic Tests Have sleep()-Based Races

- **File**: `crates/context-graph-mcp/src/handlers/tests/search_periodic_test.rs:118,331,379,439`
- **Details**: `sleep(Duration::from_millis(500))` as indexing wait. Assertions wrapped in `if !results.is_empty()`, so lost races still pass.

### TST-11 (LOW): Edge Case Test Accepts All Outcomes

- **File**: `crates/context-graph-mcp/src/handlers/tests/manual_fsv_verification.rs:239-289`
- **Details**: Empty content test accepts both success and failure as "VERIFIED". Does not enforce expected behavior.

### TST-12 (LOW): Benchmarks Test Formula Math, Not System Performance

- **Files**: `crates/context-graph-benchmark/src/metrics/e1_semantic.rs:648+`, `e4_hybrid_session.rs:891+`
- **Details**: Tests construct metrics manually with values that pass all targets. Proves the formula works, not that the system meets targets.

---

## 7. Build, Config & Safety Findings

### BLD-01 (HIGH): `GpuResources` Marked `Sync` Without Thread-Safety Guarantee

- **File**: `crates/context-graph-cuda/src/ffi/faiss.rs:565-567`
- **Details**: FAISS `StandardGpuResources` is not documented as thread-safe. `unsafe impl Sync` allows `&GpuResources` shared across threads. If multiple indices share `Arc<GpuResources>` and operate from different threads, this is a data race.

### BLD-02 (HIGH): Missing i32 Bounds Check in `compute_pairwise_distances_gpu`

- **File**: `crates/context-graph-cuda/src/ffi/knn.rs:506-507`
- **Details**: `n_points` and `dimension` cast to `i32` without bounds check. Sibling function `compute_core_distances_gpu` (line 384) correctly validates. Inconsistency proves this is an oversight.

### BLD-03 (MEDIUM): `static mut` for CUDA Init Result (Deprecated Pattern)

- **Files**: `crates/context-graph-cuda/src/safe/device.rs:31`, `crates/context-graph-cuda/src/ffi/knn.rs:82`
- **Details**: `static mut` is deprecated and will become a hard error under Rust edition 2024. The same fix was applied elsewhere (HIGH-21: `static mut` -> `AtomicU32`) but missed these two production files.

### BLD-04 (MEDIUM): `num_pairs` Overflow Risk in Pairwise Distance

- **File**: `crates/context-graph-cuda/src/ffi/knn.rs:481`
- **Details**: `n_points * (n_points - 1) / 2` can overflow `usize` on 32-bit platforms. No `checked_mul` protection. CRIT-07 bounds check at line 523 happens AFTER potentially-overflowed value is already used.

### BLD-05 (MEDIUM): Systemic `unsafe impl Send/Sync` (10+ Model Types)

- **Files**: `crates/context-graph-embeddings/src/models/pretrained/*/types.rs` (10+ occurrences)
- **Details**: All use superficial safety comments like "safe due to RwLock". `LlmState` wrapping C++ FFI objects (`LlamaBackend`, `LlamaModel`) is highest risk -- thread safety depends on llama.cpp internals.

### BLD-06 (MEDIUM): `test-utils` Feature in Production Dependencies

- **File**: `crates/context-graph-benchmark/Cargo.toml:21`
- **Details**: `context-graph-core` declared with `features = ["test-utils"]` in `[dependencies]` (not dev-dependencies). Feature is documented as "WARNING: Never enable this in production builds!"

### BLD-07 (MEDIUM): CausalE11Index Memory Leak on Duplicate Insertions

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/causal_hnsw_index.rs:178-183`
- **Details**: usearch doesn't support deletion. Old vector remains permanently. Each re-insertion leaks 3072 bytes (768 * 4). No automatic rebuild trigger based on ghost count.

### BLD-08 (MEDIUM): Inconsistent Bounds Checking in CUDA Kernel Params

- **File**: `crates/context-graph-cuda/src/ffi/knn.rs:384-395`
- **Details**: `compute_core_distances_gpu` validates `n_points` but not `dimension` or `k` for i32 overflow. Asymmetry in validation.

### BLD-09 (LOW): Lock Ordering Inversion in CausalE11Index

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/causal_hnsw_index.rs:172-175 vs 360-363`
- **Details**: `insert()` acquires locks [1,2,3,4], `rebuild()` acquires [3,1,2,4]. ABBA pattern. parking_lot hangs on deadlock (no panic).

### BLD-10 (LOW): Unbounded `soft_deleted` HashMap

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/store.rs:84`
- **Details**: Only drained during compaction. Millions of deletions could consume significant memory. Uses `HashMap<Uuid, bool>` where `bool` is always `true` (should be HashSet).

### BLD-11 (LOW): Duplicate Logging Frameworks (log + tracing)

- **File**: `crates/context-graph-graph/Cargo.toml:44`
- **Details**: Declares both `tracing` and `log` as dependencies. Every other crate uses only `tracing`.

### BLD-12 (LOW): Workspace Dependency Version Inconsistencies

- **Files**: Multiple Cargo.toml files
- **Details**: `context-graph-embeddings` declares `uuid = { version = "1.0" }` vs workspace `uuid = { version = "1.6" }`. Missing `serde` feature (works via transitive dep, but fragile).

### BLD-13 (LOW): Dead Feature Flags (onnx, cudnn)

- **Files**: `crates/context-graph-embeddings/Cargo.toml:98`, `crates/context-graph-cuda/Cargo.toml:46`
- **Details**: `onnx = []` enables nothing, no `#[cfg(feature = "onnx")]` blocks exist. `cudnn = ["cuda"]` has zero code gated on it.

### BLD-14 (LOW): 153 `#[allow(dead_code)]` Annotations

- **Scope**: 80 files across codebase
- **Details**: Each suppressed warning is a potential indicator of incomplete features, abandoned refactoring, or over-engineering.

### BLD-15 (LOW): Stale CF Count in Doc Comments (17 vs 52)

- **File**: `crates/context-graph-storage/src/teleological/rocksdb_store/store.rs:49,102,158`
- **Details**: Comments reference "17 column families" and "39 total". Actual count is 52.

### BLD-16 (LOW): `half` Crate Version/Feature Mismatch

- **Files**: `crates/context-graph-core/Cargo.toml:51` vs `crates/context-graph-cuda/Cargo.toml:31`
- **Details**: Core uses `half = "2.3"`, CUDA uses `half = "2.4"` with `num-traits` feature. Core lacks `num-traits`.

### BLD-17 (LOW): Likely Unused `num_cpus` Dependency

- **File**: `crates/context-graph-graph/Cargo.toml:51`
- **Details**: `std::thread::available_parallelism()` and `rayon` provide same functionality.

---

## 8. Remediation Priority

### Batch 1: Critical Data Integrity (Immediate)

| ID | Fix | Effort |
|----|-----|--------|
| CRIT-03 | Add `add_to_indexes()` call in `store_batch_async` | Small |
| CRIT-04 | Hard-delete from RocksDB in `compact_async` before draining HashMap | Medium |
| STG-01 | Wrap deserialization in `Result`, skip corrupted records with `warn!` | Medium |
| AGT-04 | Add `read_line` size limit (e.g., `take(10MB)`) for both stdio and TCP | Small |

### Batch 2: Phantom Parameters & Logic Bugs (High Impact)

| ID | Fix | Effort |
|----|-----|--------|
| CRIT-06 | Route search to correct HNSW index based on `embedder_indices` | Large |
| MCP-01 | Wire `sessionScope` to filter memories by session/recency | Medium |
| MCP-02 | Compute real semantic alignment for consolidation strategy | Medium |
| MCP-03 | Force `MultiSpace` strategy when custom weight profile is loaded | Small |
| STG-03/04/10 | Use RocksDB merge operator or transactions for read-modify-write | Large |

### Batch 3: Incomplete Features (Significant)

| ID | Fix | Effort |
|----|-----|--------|
| CRIT-01/02 | Implement memory loading and discovery cycle in background loops | Large |
| AGT-02/03 | Add RocksDB persistence for CausalGraph and GraphStorage | Large |
| STG-05 | Implement E12 MaxSim reranking in Pipeline Stage 3 | Medium |
| MCP-10 | Implement actual merge operation in consolidation | Large |

### Batch 4: Test Suite Integrity

| ID | Fix | Effort |
|----|-----|--------|
| TST-01 | Replace println-only test with real assertions | Small |
| TST-02 | Create non-CUDA E2E tests or document CUDA requirement | Medium |
| TST-04/05 | Add `assert!(!results.is_empty())` before conditional blocks | Small |
| CORE-01 | Fix stub asymmetric vectors (different seeds for cause/effect) | Small |

### Batch 5: Safety & Correctness

| ID | Fix | Effort |
|----|-----|--------|
| BLD-01 | Remove `unsafe impl Sync` from `GpuResources` or add Mutex wrapper | Small |
| BLD-02/08 | Add bounds checks for dimension/k in CUDA functions | Small |
| BLD-03 | Replace `static mut` with `OnceLock<CUresult>` | Small |
| AGT-06 | Use `char_indices()` for UTF-8 safe truncation | Small |
| CRIT-05 | Sort score vectors before RRF rank assignment | Small |

### Batch 6: Cleanup & Hygiene

| ID | Fix | Effort |
|----|-----|--------|
| STG-02 | Replace ghost vectors with periodic `rebuild()` trigger | Medium |
| BLD-05 | Audit `unsafe impl Send/Sync` -- remove where auto-derivation works | Medium |
| BLD-14 | Audit 153 `#[allow(dead_code)]` annotations | Large |
| BLD-15 | Update stale doc comments | Small |

---

## Appendix: Cross-Cutting Themes

### Theme 1: The Facade Pattern

Both background discovery services and several MCP parameters follow the same anti-pattern: build a complete interface (parse, validate, log, echo back) but leave the core logic unimplemented. The external API contract is fulfilled at every level except the actual behavior.

### Theme 2: Stub Divergence

The InMemoryTeleologicalStore and StubMultiArrayProvider diverge from production behavior in at least 5 ways: unweighted averaging, identical asymmetric vectors, non-deterministic RRF, silent no-ops for provenance, and missing index operations. Tests passing against stubs provide limited confidence about production correctness.

### Theme 3: Read-Modify-Write Without Atomicity

Four separate locations (causal index, E13 inverted index, E6 inverted index, file index) share the same pattern: read from RocksDB, modify in memory, write back. None are protected against concurrent modification. This is the most pervasive structural bug in the storage layer.

### Theme 4: Silent Degradation

Multiple error paths convert failures to empty results (`Ok(Vec::new())`, `Ok(None)`, `0.0`) rather than propagating errors. This makes debugging extremely difficult -- the system appears to work but produces subtly wrong results.
