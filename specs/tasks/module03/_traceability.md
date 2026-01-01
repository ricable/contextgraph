# Module 3: Embedding Pipeline - PRD Traceability Matrix

```yaml
metadata:
  module_id: M03
  module_name: 12-Model Embedding Pipeline
  version: 2.0.0
  generated: 2026-01-01
  prd_sources:
    - docs2/contextprd.md
    - docs2/constitution.yaml
    - docs2/implementationplan.md
    - docs2/CUDA-13-1-RTX-5090-Report.md
  coverage_target: 100%
```

---

## PRD Coverage Summary

| PRD Section | Requirements | Tasks Covering | Coverage |
|-------------|-------------|----------------|----------|
| 12-Model Ensemble | 12 | 12 | 100% |
| FuseMoE Fusion | 5 | 7 | 100% |
| Batch Processing | 4 | 4 | 100% |
| Cache System | 5 | 4 | 100% |
| Performance Targets | 8 | 2 | 100% |
| API Compatibility | 2 | 2 | 100% |
| GPU/CUDA | 6 | 4 | 100% |
| Configuration | 5 | 6 | 100% |
| Error Handling | 3 | 1 | 100% |
| **TOTAL** | **50** | **52** | **100%** |

---

## Detailed Traceability Matrix

### PRD-EMB-001: Semantic Embedding Model

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| Semantic embedding using E5-large-v2 | contextprd.md | M03-L03 | ☐ |
| 1024D output dimension | constitution.yaml | M03-L03, M03-F02 | ☐ |
| <5ms latency target | constitution.yaml | M03-S11 | ☐ |
| Support instruction prefix | contextprd.md | M03-L03 | ☐ |

### PRD-EMB-002: Temporal-Recent Model

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| Exponential decay weighting | contextprd.md | M03-L04 | ☐ |
| 512D output dimension | constitution.yaml | M03-L04, M03-F02 | ☐ |
| Custom implementation (no pretrained) | implementationplan.md | M03-L04 | ☐ |

### PRD-EMB-003: Temporal-Periodic Model

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| Fourier basis functions | contextprd.md | M03-L05 | ☐ |
| 512D output dimension | constitution.yaml | M03-L05, M03-F02 | ☐ |
| Encode hour/day/week/month/year | contextprd.md | M03-L05 | ☐ |

### PRD-EMB-004: Temporal-Positional Model

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| Sinusoidal positional encoding | contextprd.md | M03-L06 | ☐ |
| 512D output dimension | constitution.yaml | M03-L06, M03-F02 | ☐ |
| Transformer-style PE | implementationplan.md | M03-L06 | ☐ |

### PRD-EMB-005: Causal Model

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| Longformer-base-4096 | models_config.toml | M03-L07 | ☐ |
| 768D output dimension | constitution.yaml | M03-L07, M03-F02 | ☐ |
| 4096 token context length | contextprd.md | M03-L07 | ☐ |
| <8ms latency target | constitution.yaml | M03-S11 | ☐ |

### PRD-EMB-006: Sparse Model

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| SPLADE-cocondenser | models_config.toml | M03-L08 | ☐ |
| ~30K sparse (5% active) | contextprd.md | M03-L08 | ☐ |
| Project to 1536D | implementationplan.md | M03-L08 | ☐ |
| <3ms latency target | constitution.yaml | M03-S11 | ☐ |

### PRD-EMB-007: Code Model

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| CodeBERT-base | models_config.toml | M03-L09 | ☐ |
| 768D output dimension | constitution.yaml | M03-L09, M03-F02 | ☐ |
| Language-aware tokenization | contextprd.md | M03-L09 | ☐ |
| <10ms latency target | constitution.yaml | M03-S11 | ☐ |

### PRD-EMB-008: Graph Model

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| paraphrase-MiniLM-L6-v2 | models_config.toml | M03-L10 | ✅ |
| 384D output dimension | constitution.yaml | M03-L10, M03-F02 | ✅ |
| Message passing for relations | contextprd.md | M03-L10 | ✅ |
| <5ms latency target | constitution.yaml | M03-S11 | ☐ |

### PRD-EMB-009: HDC Model

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| Hyperdimensional computing | contextprd.md | M03-L11 | ☐ |
| 10K-bit → 1024D projection | constitution.yaml | M03-L11, M03-F02 | ☐ |
| XOR binding, majority bundling | contextprd.md | M03-L11 | ☐ |
| Custom implementation | implementationplan.md | M03-L11 | ☐ |

### PRD-EMB-010: Multimodal Model

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| CLIP-vit-large-patch14 | models_config.toml | M03-L12 | ☐ |
| 768D output dimension | constitution.yaml | M03-L12, M03-F02 | ☐ |
| Support text AND image inputs | contextprd.md | M03-L12, M03-F06 | ☐ |
| 77 token limit | contextprd.md | M03-L12 | ☐ |

### PRD-EMB-011: Entity Model

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| all-MiniLM-L6-v2 | models_config.toml | M03-L13 | ☐ |
| 384D output dimension | constitution.yaml | M03-L13, M03-F02 | ☐ |
| TransE-style (h + r ≈ t) | contextprd.md | M03-L13 | ☐ |
| <2ms latency target | constitution.yaml | M03-S11 | ☐ |

### PRD-EMB-012: Late-Interaction Model

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| ColBERTv2.0 | models_config.toml | M03-L14 | ☐ |
| 128D per-token embeddings | constitution.yaml | M03-L14, M03-F02 | ☐ |
| MaxSim scoring | contextprd.md | M03-L14 | ☐ |
| <8ms latency target | constitution.yaml | M03-S11 | ☐ |

---

### PRD-FUSION: FuseMoE Requirements

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| 8 expert networks | constitution.yaml | M03-L21, M03-F14 | ☐ |
| Top-k=2 routing | constitution.yaml | M03-L22, M03-F14 | ☐ |
| Laplace-smoothed gating | contextprd.md | M03-L20, M03-F14 | ☐ |
| 1536D output dimension | constitution.yaml | M03-L23, M03-F05 | ☐ |
| <3ms fusion latency | constitution.yaml | M03-S11 | ☐ |
| Expert weights stored | contextprd.md | M03-F05, M03-L23 | ☐ |
| Capacity factor 1.25x | constitution.yaml | M03-L22, M03-F14 | ☐ |
| Load balance loss | contextprd.md | M03-L22 | ☐ |
| CAME-AB optional | implementationplan.md | M03-L24 | ☐ |

---

### PRD-BATCH: Batch Processing Requirements

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| Dynamic batching | constitution.yaml | M03-L17, M03-F13 | ☐ |
| Max batch size 32 | constitution.yaml | M03-F13 | ☐ |
| Sort by sequence length | contextprd.md | M03-L17, M03-F13 | ☐ |
| >100 items/sec throughput | constitution.yaml | M03-S11 | ☐ |
| Padding strategies | contextprd.md | M03-F13 | ☐ |
| Per-model queues | implementationplan.md | M03-L16, M03-L17 | ☐ |
| 50ms max wait timeout | constitution.yaml | M03-F13 | ☐ |

---

### PRD-CACHE: Embedding Cache Requirements

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| LRU eviction policy | constitution.yaml | M03-L19, M03-F15 | ✅ |
| 100K max entries | constitution.yaml | M03-F15 | ✅ |
| <100μs cache hit latency | constitution.yaml | M03-S11 | ☐ |
| >80% hit rate target | constitution.yaml | M03-S10 | ☐ |
| Disk persistence option | contextprd.md | M03-L19, M03-F15 | ✅ |
| TTL-based expiration | constitution.yaml | M03-F15 | ✅ |
| Content hashing (xxhash64) | contextprd.md | M03-L18 | ☐ |

---

### PRD-API: API Compatibility Requirements

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| EmbeddingProvider trait compat | implementationplan.md | M03-S03 | ☐ |
| 1536D output (matches MemoryNode) | constitution.yaml | M03-S03, M03-F05 | ☐ |
| embed() method | contextprd.md | M03-S03 | ☐ |
| embed_batch() method | contextprd.md | M03-S03 | ☐ |
| dimension() returns 1536 | constitution.yaml | M03-S03 | ☐ |
| Thread-safe (Arc) | implementationplan.md | M03-S03 | ☐ |

---

### PRD-GPU: GPU/CUDA Requirements

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| CUDA 13.1 support | CUDA-Report.md | M03-S04, M03-F15 | ✅ |
| Green Contexts | CUDA-Report.md | M03-F15 | ✅ |
| GPU memory pool | CUDA-Report.md | M03-S05 | ☐ |
| <24GB total memory | constitution.yaml | M03-L02, M03-S10 | ☐ |
| Mixed precision (FP16/BF16) | CUDA-Report.md | M03-F12, M03-F15 | ✅ |
| CUDA Graphs | CUDA-Report.md | M03-F15 | ✅ |
| Compute capability 12.0 | CUDA-Report.md | M03-S04 | ☐ |
| Grouped GEMM for MoE | CUDA-Report.md | M03-S06 | ☐ |

---

### PRD-CONFIG: Configuration Requirements

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| TOML configuration files | constitution.yaml | M03-S08 | ☐ |
| models_config.toml support | implementationplan.md | M03-S08 | ☐ |
| Environment override | contextprd.md | M03-S08 | ☐ |
| Per-model device placement | constitution.yaml | M03-F12 | ☐ |
| Quantization modes | constitution.yaml | M03-F12 | ☐ |
| Lazy loading option | implementationplan.md | M03-F12, M03-L01 | ☐ |
| Preload models list | constitution.yaml | M03-F12 | ☐ |

---

### PRD-ERROR: Error Handling Requirements

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| Comprehensive error enum | implementationplan.md | M03-F08 | ☐ |
| Model-specific errors | contextprd.md | M03-F08 | ☐ |
| GPU/CUDA errors | CUDA-Report.md | M03-F08 | ☐ |
| Timeout errors | constitution.yaml | M03-F08 | ☐ |
| Validation errors | contextprd.md | M03-F08 | ☐ |
| No fallbacks/workarounds | implementationplan.md | M03-F08 | ☐ |

---

### PRD-PERF: Performance Requirements

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| <200ms single embed E2E | constitution.yaml | M03-S11 | ☐ |
| >100 items/sec batch | constitution.yaml | M03-S11 | ☐ |
| <3ms FuseMoE fusion | constitution.yaml | M03-S11 | ☐ |
| <100μs cache hit | constitution.yaml | M03-S11 | ☐ |
| P95 latency metrics | constitution.yaml | M03-S02, M03-S11 | ☐ |
| Throughput metrics | constitution.yaml | M03-S02, M03-S11 | ☐ |
| Memory usage tracking | constitution.yaml | M03-L02, M03-S02 | ☐ |
| GPU utilization | CUDA-Report.md | M03-S02 | ☐ |

---

### PRD-TEST: Testing Requirements

| PRD Requirement | Source | Task ID | Status |
|-----------------|--------|---------|--------|
| >90% unit test coverage | implementationplan.md | M03-S09 | ☐ |
| >80% integration coverage | implementationplan.md | M03-S10 | ☐ |
| Real models (no stubs) | implementationplan.md | M03-S10 | ☐ |
| Benchmark regression CI | implementationplan.md | M03-S11 | ☐ |
| E2E pipeline tests | implementationplan.md | M03-S10 | ☐ |

---

## Gap Analysis

### Fully Covered (No Gaps)

- ✅ All 12 embedding models mapped to tasks
- ✅ FuseMoE architecture fully specified
- ✅ Batch processing with dynamic batching
- ✅ Cache system with LRU and persistence
- ✅ API compatibility with EmbeddingProvider
- ✅ GPU/CUDA integration points defined
- ✅ Configuration loading and overrides
- ✅ Comprehensive error handling
- ✅ Performance targets with benchmarks
- ✅ Testing strategy with coverage targets

### Potential Risks

| Risk | Mitigation | Task |
|------|------------|------|
| Model loading OOM | MemoryTracker with budgets | M03-L02 |
| Latency target miss | Per-model benchmarking | M03-S11 |
| Cache hit rate low | Tune TTL and capacity | M03-F15, M03-L19 |
| CUDA compat issues | Stub interfaces first | M03-S06 |

---

## Verification Checklist

### Pre-Implementation

- [ ] All PRD requirements mapped to at least one task
- [ ] No orphan requirements (unmapped)
- [ ] No orphan tasks (no PRD backing)
- [ ] Dependencies correctly captured
- [ ] Performance targets assigned to benchmarks

### Post-Implementation

- [ ] All ☐ boxes checked (requirements met)
- [ ] Unit tests cover types/traits
- [ ] Integration tests cover E2E flow
- [ ] Benchmarks validate performance
- [ ] Documentation complete

---

## Requirement Status Legend

| Symbol | Meaning |
|--------|---------|
| ☐ | Not started |
| ◐ | In progress |
| ☑ | Complete, not verified |
| ✅ | Complete and verified |
| ❌ | Blocked/Failed |

---

*Traceability Matrix Generated: 2026-01-01*
*Module: 03 - 12-Model Embedding Pipeline*
*Version: 2.0.0*
*Coverage: 100% PRD requirements mapped*
