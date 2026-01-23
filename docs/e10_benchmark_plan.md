# E10 Intent/Context MCP Tool Integration Benchmark Plan

## Overview

This document outlines the benchmark plan for validating the E10 dual intent/context embedding integration into the MCP tool system. The E10 system adds two new tools (`search_by_intent`, `find_contextual_matches`) that ENHANCE the E1 semantic foundation with intent-aware retrieval.

**Version**: 1.0.0
**Date**: 2026-01-23
**Related Plan**: E10 Dual Intent/Context Integration Plan

---

## Benchmark Objectives

1. **Performance**: Measure latency and throughput of E10 tools
2. **Quality**: Validate retrieval accuracy with intent/context awareness
3. **Integration**: Confirm E10 enhances (not competes with) E1 semantic search
4. **Asymmetry**: Verify intent→context vs context→intent direction modifiers work correctly
5. **Blending**: Validate `blendWithSemantic` parameter behavior

---

## Test Environment

### Hardware Requirements
- GPU: NVIDIA RTX 5090 (32GB VRAM)
- CPU: 16+ cores
- RAM: 64GB+
- Storage: NVMe SSD

### Software Requirements
- CUDA 13.1+
- Rust 1.75+
- All 13 embedding models loaded

### Database State
- RocksDB with pre-populated test memories
- Minimum 1,000 diverse memories for meaningful benchmarks

---

## Benchmark Categories

### 1. Latency Benchmarks

#### 1.1 Single Request Latency

| Test Case | Tool | Expected p50 | Expected p95 | Expected p99 |
|-----------|------|--------------|--------------|--------------|
| L1.1 | `search_by_intent` (cold) | <500ms | <1000ms | <1500ms |
| L1.2 | `search_by_intent` (warm) | <200ms | <400ms | <600ms |
| L1.3 | `find_contextual_matches` (cold) | <500ms | <1000ms | <1500ms |
| L1.4 | `find_contextual_matches` (warm) | <200ms | <400ms | <600ms |
| L1.5 | Baseline `search_graph` (warm) | <150ms | <300ms | <500ms |

**Method**:
```bash
# Run 100 requests, measure latency distribution
for i in {1..100}; do
  echo '{"jsonrpc":"2.0","id":'$i',"method":"tools/call","params":{"name":"search_by_intent","arguments":{"query":"find performance optimization work","topK":10}}}' | nc 127.0.0.1 3190
done
```

#### 1.2 Comparative Latency (E10 vs E1)

| Scenario | Metric | Target |
|----------|--------|--------|
| E10 overhead vs pure E1 | Additional latency | <100ms |
| Blend computation | Time for score combination | <10ms |
| Direction modifier application | Time to apply asymmetry | <5ms |

### 2. Throughput Benchmarks

#### 2.1 Requests Per Second

| Test Case | Tool | Concurrent Clients | Target RPS |
|-----------|------|-------------------|------------|
| T2.1 | `search_by_intent` | 1 | >5 |
| T2.2 | `search_by_intent` | 4 | >15 |
| T2.3 | `search_by_intent` | 8 | >25 |
| T2.4 | `find_contextual_matches` | 1 | >5 |
| T2.5 | `find_contextual_matches` | 4 | >15 |

**Method**:
```bash
# Use wrk or custom load generator
wrk -t4 -c8 -d30s --script=intent_search.lua http://127.0.0.1:3190/
```

### 3. Quality Benchmarks

#### 3.1 Retrieval Accuracy

Pre-define ground truth datasets with known intent/context relationships:

| Dataset | Description | Size | Expected MRR |
|---------|-------------|------|--------------|
| Q3.1 | Intent queries with known answers | 50 queries | >0.7 |
| Q3.2 | Context matching scenarios | 50 queries | >0.7 |
| Q3.3 | Mixed intent+semantic queries | 100 queries | >0.65 |

**Metrics**:
- Mean Reciprocal Rank (MRR)
- Precision@1, Precision@5, Precision@10
- Recall@10
- NDCG@10

#### 3.2 Intent vs Semantic Differentiation

Validate that E10 finds results that pure E1 would miss:

| Test Case | Scenario | Expected Behavior |
|-----------|----------|-------------------|
| Q3.4 | Same content, different intent | E10 should differentiate |
| Q3.5 | Similar intent, different content | E10 should find related |
| Q3.6 | E1 high match, E10 low match | `blendWithSemantic` should balance |

**Example Test Data**:
```json
{
  "memory_1": {
    "content": "Implemented caching for API responses",
    "intent": "performance optimization"
  },
  "memory_2": {
    "content": "Implemented caching for API responses",
    "intent": "reduce external API calls"
  }
}
```
Query: "find work aimed at making the system faster"
- E1 alone: Both memories equal similarity
- E10: memory_1 should rank higher (performance intent match)

### 4. Asymmetry Benchmarks

#### 4.1 Direction Modifier Validation

| Test Case | Direction | Modifier | Validation |
|-----------|-----------|----------|------------|
| A4.1 | intent→context | 1.2x | Score boost confirmed |
| A4.2 | context→intent | 0.8x | Score dampening confirmed |
| A4.3 | Symmetric comparison | 1.0x | No modifier when same role |

**Method**:
```python
# Store memory with known embedding
# Query as intent → check score
# Query as context → check score
# Verify ratio matches 1.2/0.8 = 1.5x difference
```

#### 4.2 Asymmetry Impact on Rankings

| Scenario | Query Role | Memory Role | Expected Effect |
|----------|------------|-------------|-----------------|
| A4.4 | Intent query | Context memory | 1.2x boost (intended use) |
| A4.5 | Context query | Intent memory | 0.8x dampen (reverse direction) |

### 5. Blending Benchmarks

#### 5.1 `blendWithSemantic` Parameter

| Test Case | Blend Value | E1 Weight | E10 Weight | Expected Behavior |
|-----------|-------------|-----------|------------|-------------------|
| B5.1 | 0.0 | 100% | 0% | Pure E1 results |
| B5.2 | 0.3 (default) | 70% | 30% | E1 dominant, E10 enhances |
| B5.3 | 0.5 | 50% | 50% | Equal contribution |
| B5.4 | 1.0 | 0% | 100% | Pure E10 results |

**Validation**:
```python
# For each blend value:
# 1. Run search with known scores
# 2. Calculate expected blended score: final = e1*(1-blend) + e10*blend
# 3. Verify actual matches expected within epsilon (0.001)
```

#### 5.2 Blend Edge Cases

| Test Case | Scenario | Expected Handling |
|-----------|----------|-------------------|
| B5.5 | E1 strong (>0.8), E10 weak (<0.3) | E1 should dominate at low blend |
| B5.6 | E1 weak (<0.3), E10 strong (>0.8) | E10 should lift ranking at high blend |
| B5.7 | Both weak (<0.3) | Combined score still weak |
| B5.8 | Both strong (>0.8) | Combined score very strong |

### 6. Integration Benchmarks

#### 6.1 Weight Profile Validation

Verify `intent_search` weight profile is used correctly:

| Embedder | Expected Weight | Purpose |
|----------|-----------------|---------|
| E1 | 0.40 | Foundation semantic |
| E5 | 0.10 | Causal reasoning |
| E6 | 0.05 | Keyword precision |
| E7 | 0.10 | Code patterns |
| E8 | 0.05 | Graph structure |
| E10 | 0.25 | Intent/context (PRIMARY) |
| E11 | 0.05 | Entity relationships |
| E2-E4 | 0.0 | Temporal excluded |
| E9 | 0.0 | HDC excluded |
| E12-E13 | 0.0 | Pipeline-only |

#### 6.2 Tool Dispatch Verification

| Test Case | Request | Expected Handler |
|-----------|---------|------------------|
| I6.1 | `search_by_intent` | `call_search_by_intent` |
| I6.2 | `find_contextual_matches` | `call_find_contextual_matches` |
| I6.3 | Unknown tool | Error -32601 |

### 7. Edge Case Benchmarks

#### 7.1 Input Validation

| Test Case | Input | Expected Response |
|-----------|-------|-------------------|
| E7.1 | Empty query | Error: query required |
| E7.2 | Query > 10,000 chars | Truncation or error |
| E7.3 | topK = 0 | Error: invalid topK |
| E7.4 | topK > 50 | Clamped to 50 |
| E7.5 | minScore = -1 | Error: invalid range |
| E7.6 | minScore = 2.0 | Error: invalid range |
| E7.7 | blendWithSemantic = -0.5 | Error: invalid range |
| E7.8 | blendWithSemantic = 1.5 | Error: invalid range |

#### 7.2 Empty/Minimal Database

| Test Case | Database State | Expected Behavior |
|-----------|----------------|-------------------|
| E7.9 | 0 memories | Empty results, no error |
| E7.10 | 1 memory | Single result or empty |
| E7.11 | Memories without E10 embeddings | Graceful degradation |

#### 7.3 Model Loading Failures

| Test Case | Scenario | Expected Behavior |
|-----------|----------|-------------------|
| E7.12 | E10 model not loaded | Error with clear message |
| E7.13 | E1 model not loaded | Error (required) |
| E7.14 | Partial model loading | Tools that need missing model fail fast |

### 8. Stress Benchmarks

#### 8.1 Sustained Load

| Test Case | Duration | Concurrent | Target |
|-----------|----------|------------|--------|
| S8.1 | 5 minutes | 4 clients | No degradation |
| S8.2 | 30 minutes | 2 clients | Memory stable |
| S8.3 | 1 hour | 1 client | No leaks |

#### 8.2 Burst Load

| Test Case | Burst Size | Expected Recovery |
|-----------|------------|-------------------|
| S8.4 | 50 requests in 1s | <5s to normal latency |
| S8.5 | 100 requests in 1s | <10s to normal latency |

---

## Test Data Requirements

### Synthetic Test Dataset

Create a dataset with controlled intent/context properties:

```json
{
  "dataset_name": "e10_benchmark_v1",
  "size": 1000,
  "categories": [
    {
      "name": "performance_optimization",
      "count": 100,
      "intent_keywords": ["optimize", "faster", "performance", "speed"],
      "content_types": ["caching", "indexing", "algorithm improvement"]
    },
    {
      "name": "bug_fixing",
      "count": 100,
      "intent_keywords": ["fix", "resolve", "debug", "error"],
      "content_types": ["null pointer", "race condition", "validation"]
    },
    {
      "name": "feature_development",
      "count": 100,
      "intent_keywords": ["implement", "add", "create", "build"],
      "content_types": ["API endpoint", "UI component", "database schema"]
    }
    // ... 7 more categories to reach 1000
  ]
}
```

### Ground Truth Queries

```json
{
  "queries": [
    {
      "id": "Q001",
      "query": "find work focused on making the system faster",
      "expected_category": "performance_optimization",
      "expected_top3_ids": ["mem_001", "mem_023", "mem_045"]
    },
    {
      "id": "Q002",
      "query": "what was done to fix the authentication issue",
      "expected_category": "bug_fixing",
      "expected_top3_ids": ["mem_102", "mem_115", "mem_131"]
    }
    // ... 98 more queries
  ]
}
```

---

## Benchmark Execution Plan

### Phase 1: Setup (Day 1)

1. [ ] Create synthetic test dataset (1000 memories)
2. [ ] Generate ground truth queries (100 queries)
3. [ ] Set up benchmark harness
4. [ ] Verify all 13 models loaded
5. [ ] Baseline `search_graph` performance

### Phase 2: Latency & Throughput (Day 2)

1. [ ] Run latency benchmarks (L1.1 - L1.5)
2. [ ] Run throughput benchmarks (T2.1 - T2.5)
3. [ ] Compare E10 overhead vs baseline
4. [ ] Document any bottlenecks

### Phase 3: Quality & Accuracy (Day 3)

1. [ ] Run retrieval accuracy benchmarks (Q3.1 - Q3.6)
2. [ ] Calculate MRR, Precision, Recall, NDCG
3. [ ] Validate intent differentiation
4. [ ] Compare with pure E1 results

### Phase 4: Asymmetry & Blending (Day 4)

1. [ ] Run asymmetry benchmarks (A4.1 - A4.5)
2. [ ] Validate direction modifier ratios
3. [ ] Run blending benchmarks (B5.1 - B5.8)
4. [ ] Verify weight calculations

### Phase 5: Integration & Edge Cases (Day 5)

1. [ ] Run integration benchmarks (I6.1 - I6.3)
2. [ ] Verify weight profile usage
3. [ ] Run edge case benchmarks (E7.1 - E7.14)
4. [ ] Document error handling

### Phase 6: Stress Testing (Day 6)

1. [ ] Run sustained load tests (S8.1 - S8.3)
2. [ ] Monitor memory usage
3. [ ] Run burst tests (S8.4 - S8.5)
4. [ ] Document recovery behavior

### Phase 7: Analysis & Report (Day 7)

1. [ ] Compile all benchmark results
2. [ ] Generate performance charts
3. [ ] Write analysis report
4. [ ] Identify optimization opportunities

---

## Success Criteria

### Must Pass (Blocking)

- [ ] All latency benchmarks meet p95 targets
- [ ] MRR > 0.6 on quality benchmarks
- [ ] Asymmetry modifiers applied correctly (within 1% tolerance)
- [ ] No crashes under stress testing
- [ ] All edge case inputs handled gracefully

### Should Pass (Non-Blocking)

- [ ] Throughput meets targets at 8 concurrent clients
- [ ] E10 overhead < 50ms vs pure E1
- [ ] Memory stable over 1 hour sustained load

### Nice to Have

- [ ] MRR > 0.75 on intent-specific queries
- [ ] p99 latency < 500ms for warm requests
- [ ] Linear scaling up to 16 concurrent clients

---

## Benchmark Tooling

### Required Tools

1. **Load Generator**: Custom Rust binary or `wrk` with Lua scripts
2. **Metrics Collection**: Prometheus + Grafana or custom logging
3. **Analysis**: Python with pandas, matplotlib for charts
4. **Database Seeding**: CLI tool or test fixtures

### Benchmark Harness

```rust
// benchmark_harness/src/main.rs
struct BenchmarkResult {
    test_id: String,
    latency_p50_ms: f64,
    latency_p95_ms: f64,
    latency_p99_ms: f64,
    throughput_rps: f64,
    error_count: u64,
}

async fn run_benchmark(config: BenchmarkConfig) -> BenchmarkResult {
    // Implementation
}
```

---

## Appendix: JSON-RPC Test Payloads

### search_by_intent

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search_by_intent",
    "arguments": {
      "query": "find performance optimization work",
      "topK": 10,
      "minScore": 0.2,
      "blendWithSemantic": 0.3,
      "includeContent": false
    }
  }
}
```

### find_contextual_matches

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "find_contextual_matches",
    "arguments": {
      "context": "debugging a memory leak in the caching layer",
      "topK": 10,
      "minScore": 0.2,
      "blendWithSemantic": 0.3,
      "includeContent": true
    }
  }
}
```

### Baseline search_graph

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "search_graph",
    "arguments": {
      "query": "performance optimization",
      "topK": 10,
      "weightProfile": "balanced"
    }
  }
}
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-23 | Claude | Initial benchmark plan |
