# Benchmark Dashboard

**Status**: PASS | **Score**: 92% | **Compliance**: 100%

---

## Quick Metrics

### Embedder Performance

| Embedder | Role | Key Metric | Value | Target | Status |
|----------|------|------------|-------|--------|--------|
| E1 | Foundation | Validated | YES | - | PASS |
| E2 | Recency | Weighted MRR | 0.987 | >0.80 | PASS |
| E3 | Periodic | Cluster Quality | 1.0 | >0.30 | PASS |
| E4 | Sequence | Before/After | 100% | >80% | PASS |
| E5 | Causal | Asymmetry | 1.50 | 1.2-2.0 | PASS |
| E6 | Sparse | MRR@10 | 0.684 | >0.50 | PASS |
| E8 | Graph | Detection | 92% | >75% | PASS |
| E10 | Multimodal | Intent Acc | 95% | >70% | PASS |

### MCP Tool Latency

| Tool | p95 Latency | Target | Status |
|------|-------------|--------|--------|
| search_by_intent | 1.76ms | <2000ms | PASS |
| find_contextual_matches | 1.47ms | <2000ms | PASS |
| search_graph_intent | 1.26ms | <2000ms | PASS |

### Asymmetry Compliance

All embedders using asymmetric retrieval achieve **1.5x ratio** (target: 1.5 +/- 0.15):
- E5 Causal: 1.50 (PASS)
- E8 Graph: 1.50 (PASS)
- E10 Multimodal: 1.50 (PASS)

---

## Constitution Status

### ARCH Rules: 9/9 VALIDATED

```
ARCH-01  TeleologicalArray is atomic              PASS
ARCH-02  Apples-to-apples comparison only         PASS
ARCH-09  Topic threshold >= 2.5                   PASS
ARCH-12  E1 is THE semantic foundation            PASS
ARCH-17  E10 enhances E1 appropriately            PASS
ARCH-18  E5 uses asymmetric similarity            PASS
ARCH-21  Multi-space uses weighted RRF            PASS
ARCH-22  E2 supports configurable decay           PASS
ARCH-25  Temporal POST-retrieval only             PASS
```

### Anti-Patterns: 5/5 CLEAN

```
AP-02   No cross-embedder comparison              CLEAN
AP-60   Temporal excluded from topics             CLEAN
AP-61   weighted_agreement for threshold          CLEAN
AP-73   Temporal not in similarity fusion         CLEAN
AP-77   E5 asymmetric similarity                  CLEAN
```

---

## Warnings

| Priority | Embedder | Issue | Current | Target |
|----------|----------|-------|---------|--------|
| LOW | E2 | Decay accuracy | 69.6% | 80% |
| LOW | E4 | Sequence accuracy | 47.2% | 65% |

---

## Files

- `full_analysis.md` - Complete analysis (280+ lines)
- `summary.json` - Machine-readable summary
- `temporal_full.json` - E2/E3/E4 results
- `causal_synthetic.json` - E5 synthetic
- `causal_realdata_real.json` - E5 real data
- `e6_sparse_benchmark.json` - E6/E13 sparse
- `graph_full.json` - E8 graph
- `multimodal_benchmark.json` - E10 multimodal
- `mcp_intent_benchmark.json` - E10 MCP integration

---

*Last updated: 2026-01-23*
