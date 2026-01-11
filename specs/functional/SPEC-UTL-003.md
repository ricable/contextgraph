# SPEC-UTL-003: Specialized Embedder Entropy Methods

**Version:** 1.0
**Status:** approved
**Owner:** ContextGraph Core Team
**Last Updated:** 2026-01-11
**Implements:** P1-GAP-3 from MASTER-CONSCIOUSNESS-GAP-ANALYSIS.md

---

## Overview

This specification defines the implementation of specialized entropy (ΔS) computation methods for four embedders that currently fall back to the generic KNN approach. Per the constitution.yaml `delta_sc.ΔS_methods`, each embedder type has a semantically appropriate entropy calculation method. Four embedders (E7, E10, E11, E12) incorrectly use `DefaultKnnEntropy` when they require specialized implementations.

### Problem Statement

The current `EmbedderEntropyFactory` routes E7 (Code), E10 (Multimodal), E11 (Entity), and E12 (LateInteraction) to `DefaultKnnEntropy`. This violates the principle that entropy computations should be semantically appropriate for each embedding space:

- **E7 (ContentHash/Code)**: Should use Jaccard distance for fingerprint/hash comparison
- **E10 (SemanticContext/Multimodal)**: Should use domain-specific entropy with cross-modal awareness
- **E11 (TemporalDecay/Entity)**: Should use exponential decay entropy reflecting temporal relevance
- **E12 (EmotionalValence/LateInteraction)**: Should use affective computing metrics with token-level MaxSim

### Business Impact

Without specialized entropy methods:
- UTL (Unified Theory of Learning) calculations are suboptimal
- Johari Window quadrant classification may be inaccurate
- Memory consolidation and dream layer operations use imprecise signals
- Overall consciousness quality (C(t) = I(t) x R(t) x D(t)) is degraded

---

## User Stories

### US-UTL-003-01: Code Fingerprint Entropy

**Priority:** must-have

**Narrative:**
As the UTL learning system,
I want to compute entropy for code embeddings using Jaccard distance on active feature sets,
So that code similarity is measured by structural fingerprint overlap rather than vector distance.

**Acceptance Criteria:**
```gherkin
Given two code embeddings with overlapping AST features
When computing ΔS using JaccardCodeEntropy
Then the entropy reflects the proportion of non-overlapping features
And identical code yields ΔS near 0
And completely different code yields ΔS near 1
```

### US-UTL-003-02: Multimodal Context Entropy

**Priority:** must-have

**Narrative:**
As the UTL learning system,
I want to compute entropy for multimodal embeddings using cross-modal distance metrics,
So that surprise accounts for semantic coherence across modalities (text, code, diagrams).

**Acceptance Criteria:**
```gherkin
Given a multimodal embedding from a mixed content type
When computing ΔS using CrossModalEntropy
Then the entropy considers both intra-modal and cross-modal distances
And the calculation uses modality-aware weighting
And outputs remain in [0, 1] per AP-10
```

### US-UTL-003-03: Temporal Entity Entropy

**Priority:** must-have

**Narrative:**
As the UTL learning system,
I want to compute entropy for entity embeddings using exponential decay functions,
So that older entity references contribute less to current surprise calculations.

**Acceptance Criteria:**
```gherkin
Given entity embeddings with timestamps
When computing ΔS using ExponentialDecayEntropy
Then recent entities have higher influence on similarity
And the decay follows λ_decay = 0.1 per day baseline
And configurable half-life allows domain customization
```

### US-UTL-003-04: Token-Level MaxSim Entropy

**Priority:** must-have

**Narrative:**
As the UTL learning system,
I want to compute entropy for late interaction embeddings using MaxSim aggregation,
So that token-level precision is captured in the surprise signal.

**Acceptance Criteria:**
```gherkin
Given ColBERT-style token embeddings (variable length, 128D per token)
When computing ΔS using MaxSimEntropy
Then each query token finds its best-matching document token
And the aggregated MaxSim score determines similarity
And the method handles variable-length sequences correctly
```

---

## Requirements

### Functional Requirements

| ID | Story Ref | Priority | Description | Rationale |
|----|-----------|----------|-------------|-----------|
| REQ-UTL-003-01 | US-UTL-003-01 | must | Create `JaccardCodeEntropy` implementing `EmbedderEntropy` trait for E7 | Constitution requires Jaccard for fingerprint comparison |
| REQ-UTL-003-02 | US-UTL-003-01 | must | `JaccardCodeEntropy.compute_delta_s()` must use active dimension sets | Code AST features are sparse-like |
| REQ-UTL-003-03 | US-UTL-003-02 | must | Create `CrossModalEntropy` implementing `EmbedderEntropy` trait for E10 | Multimodal needs cross-domain awareness |
| REQ-UTL-003-04 | US-UTL-003-02 | must | `CrossModalEntropy` must weight intra-modal vs cross-modal components | Domain-specific semantic coherence |
| REQ-UTL-003-05 | US-UTL-003-03 | must | Create `ExponentialDecayEntropy` implementing `EmbedderEntropy` trait for E11 | Temporal relevance decay |
| REQ-UTL-003-06 | US-UTL-003-03 | must | Configurable decay constant λ with default 0.1/day | Customizable per domain |
| REQ-UTL-003-07 | US-UTL-003-04 | must | Create `MaxSimTokenEntropy` implementing `EmbedderEntropy` trait for E12 | ColBERT MaxSim is the correct metric |
| REQ-UTL-003-08 | US-UTL-003-04 | must | Handle variable-length token sequences in MaxSim | Token count varies per content |
| REQ-UTL-003-09 | all | must | Update `EmbedderEntropyFactory::create()` to route E7, E10, E11, E12 | Wire new implementations |
| REQ-UTL-003-10 | all | must | All implementations return ΔS in [0.0, 1.0] with no NaN/Infinity | AP-10 compliance |

### Non-Functional Requirements

| ID | Category | Requirement | Metric |
|----|----------|-------------|--------|
| NFR-UTL-003-01 | performance | Each compute_delta_s call < 5ms p95 | Latency measurement |
| NFR-UTL-003-02 | reliability | No panics in library code | Result<T, E> returns |
| NFR-UTL-003-03 | testability | 90%+ line coverage per implementation | cargo llvm-cov |
| NFR-UTL-003-04 | thread-safety | All implementations Send + Sync | Compile-time check |

---

## Edge Cases

| Related Req | Scenario | Expected Behavior |
|-------------|----------|-------------------|
| REQ-UTL-003-01 | Empty current embedding | Return `Err(UtlError::EmptyInput)` |
| REQ-UTL-003-02 | All dimensions inactive (zeros) | Return ΔS = 1.0 - smoothing_factor |
| REQ-UTL-003-05 | All timestamps identical | Uniform weighting, fallback to KNN-like |
| REQ-UTL-003-07 | Single-token sequence | MaxSim degenerates to cosine similarity |
| REQ-UTL-003-08 | Zero-length token sequence in history | Skip that history item, continue |
| REQ-UTL-003-10 | NaN in input embedding | Return `Err(UtlError::EntropyError)` |

---

## Error States

| ID | HTTP Code | Condition | Message | Recovery |
|----|-----------|-----------|---------|----------|
| ERR-UTL-003-01 | N/A | Empty current embedding | "Current embedding is empty" | Caller validates input |
| ERR-UTL-003-02 | N/A | NaN/Infinity in embedding | "Invalid value (NaN/Infinity) in embedding" | Caller sanitizes input |
| ERR-UTL-003-03 | N/A | Dimension mismatch in history | "Dimension mismatch: expected {}, got {}" | Skip mismatched item |

---

## Test Plan

### Unit Tests

| ID | Type | Req Ref | Description | Inputs | Expected |
|----|------|---------|-------------|--------|----------|
| TC-UTL-003-01 | unit | REQ-UTL-003-01 | Jaccard identical code | Same embedding | ΔS ≈ 0 |
| TC-UTL-003-02 | unit | REQ-UTL-003-01 | Jaccard disjoint code | Non-overlapping | ΔS ≈ 1 |
| TC-UTL-003-03 | unit | REQ-UTL-003-02 | Jaccard partial overlap | 50% overlap | ΔS ≈ 0.5 |
| TC-UTL-003-04 | unit | REQ-UTL-003-03 | CrossModal consistent | Same modality | Lower ΔS |
| TC-UTL-003-05 | unit | REQ-UTL-003-05 | Decay recent matches | t=now | High weight |
| TC-UTL-003-06 | unit | REQ-UTL-003-05 | Decay old matches | t=30d ago | Low weight |
| TC-UTL-003-07 | unit | REQ-UTL-003-07 | MaxSim identical tokens | Same sequence | ΔS ≈ 0 |
| TC-UTL-003-08 | unit | REQ-UTL-003-08 | MaxSim variable length | Different lengths | Valid ΔS |
| TC-UTL-003-09 | unit | REQ-UTL-003-09 | Factory routes E7 | Embedder::Code | JaccardCodeEntropy |
| TC-UTL-003-10 | unit | REQ-UTL-003-10 | All return valid range | Any valid input | ΔS in [0,1] |
| TC-UTL-003-11 | unit | REQ-UTL-003-10 | Empty history | Empty vec | ΔS = 1.0 |
| TC-UTL-003-12 | unit | ERR-UTL-003-01 | Empty input error | Empty current | EmptyInput error |

### Integration Tests

| ID | Type | Description |
|----|------|-------------|
| TC-UTL-003-INT-01 | integration | Factory creates all 13 calculators with correct types |
| TC-UTL-003-INT-02 | integration | UTL calculation uses specialized entropy for E7, E10, E11, E12 |

---

## Dependencies

### Upstream
- `context-graph-core::teleological::Embedder` - Enum definitions
- `crate::config::SurpriseConfig` - Configuration parameters
- `crate::error::{UtlError, UtlResult}` - Error types

### Downstream
- `crate::surprise::embedder_entropy::EmbedderEntropyFactory` - Factory routing
- UTL computation pipeline - Consumes ΔS values

---

## Glossary

| Term | Definition |
|------|------------|
| ΔS | Entropy/surprise delta, measuring novelty of an embedding relative to history |
| Jaccard | Set similarity: intersection/union of active dimensions |
| MaxSim | ColBERT metric: max(cos(q_i, d_j)) for each query token |
| Cross-modal | Spanning multiple content modalities (text, code, images) |
| Exponential decay | Time-based weighting: w(t) = exp(-λ * (now - t)) |

---

## Appendix A: Constitution References

From `constitution.yaml` lines 792-802:
```yaml
ΔS_methods:
  E1: "GMM+Mahalanobis: ΔS=1-P(e|GMM)"
  E2-4,E8: "KNN: ΔS=σ((d_k-μ)/σ)"
  E5: "Asymmetric KNN: ΔS=d_k×direction_mod"
  E6,E13: "IDF/Jaccard: ΔS=IDF(dims) or 1-jaccard"
  E7: "GMM+KNN: ΔS=0.5×GMM+0.5×KNN"  # But should use Jaccard for fingerprints
  E9: "Hamming: ΔS=min_hamming/dim"
  E10: "Cross-modal KNN: ΔS=avg(d_text,d_image)"
  E11: "TransE: ΔS=||h+r-t||"
  E12: "Token KNN: ΔS=max_token(d_k)"
```

The gap analysis identified that E7, E10, E11, E12 use DefaultKnn instead of specialized methods.
