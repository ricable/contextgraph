# E5 Causal Embedder Fix Task

**Date**: 2026-02-09
**Branch**: casetrack
**Objective**: Fix E5 causal embedder so it provides ranking signal instead of noise

---

## 1. Problem Statement

The E5 causal embedder (index 4 in the 13-embedder array, 768D, allenai/longformer-base-4096) produces degenerate embeddings. ALL text — causal, non-causal, single-character, any domain — scores 0.93-0.98 cosine similarity. This means E5 provides zero topical discrimination. Despite this, the `causal_reasoning` weight profile assigns E5 the highest weight (0.45) of any embedder, meaning it contributes 69% of the fusion numerator while providing 0% ranking signal.

### Measured Impact

| Metric | E5 Contribution |
|--------|----------------|
| Top-1 accuracy (3 causal queries) | 0/3 — ranks "x" (single char), interest rates, heat expansion above correct answers |
| Score spread across results | 0.004-0.022 (E1 is 17x wider at 0.062-0.153) |
| Ablation: removing E5 | INCREASES correct answer score (0.779 vs 0.770 with E5) |
| Cross-embedder anomalies (E5 finds something E1 misses) | 0 found |
| Direction modifier ranking change | None — all E5 scores scale uniformly |

### Three Root Causes

1. **RC-1: Degenerate embeddings.** The Longformer model was never fine-tuned for causal similarity. It produces near-constant cosine scores for all text. This is called *anisotropy* — the embedding space is collapsed into a narrow cone. Research confirms this is a known property of pre-trained transformers not fine-tuned with contrastive loss (see: Ethayarajh 2019, "How Contextual are Contextualized Word Representations?").

2. **RC-2: E5 weight (0.45) dominates fusion.** The final user-facing score is `compute_semantic_fusion()` which computes `sum(score_i * weight_i) / sum(weight_i)`. At E5=0.45 and E1=0.20, E5 pulls all scores toward its compressed 0.95 mean, drowning E1's genuine topical signal.

3. **RC-3: Direction modifiers applied to compressed scores.** `apply_asymmetric_e5_reranking()` blends `(1-e5_weight) * original + e5_weight * asymmetric_e5 * direction_mod`. Since all E5 scores are ~0.95, the direction modifier (1.2x or 0.8x) scales a near-constant, adding noise.

---

## 2. Project Architecture (What You Need to Know)

### Workspace Structure (10 Rust crates)

```
contextgraph/
  crates/
    context-graph-core/       # Domain types, traits, HNSW indexes, similarity, weights
    context-graph-storage/    # RocksDB backend, 50 CFs, teleological store, search
    context-graph-mcp/        # MCP JSON-RPC server, 55 tools, TCP transport
    context-graph-embeddings/ # 13 embedder models, ONNX/Candle, quantization
    context-graph-cuda/       # CUDA ops, HDBSCAN
    context-graph-graph/      # Graph structures, traversal
    context-graph-cli/        # CLI interface
    context-graph-benchmark/  # Performance benchmarks
    context-graph-causal-agent/ # LLM causal discovery (GBNF grammar)
    context-graph-graph-agent/  # LLM graph relationship discovery
  docs2/constitution.yaml     # Constitution v7.0.0 — authoritative rules
  models/                     # Downloaded model weights (ONNX, GGUF)
```

### 13 Embedders (ALL GPU-resident)

| Index | Name | Dim | Type | Purpose |
|-------|------|-----|------|---------|
| 0 | E1_Semantic | 1024 | Dense | Foundation — topical similarity (PROVEN: 3/3 correct top-1, 17x discrimination) |
| 1-3 | E2-E4 | 512 | Dense | Temporal (recency, periodic, positional) — always weight=0 in semantic profiles |
| 4 | **E5_Causal** | **768** | **Asymmetric** | **Causal chains — BROKEN (this task)** |
| 5 | E6_Sparse | 30K | Sparse | Keyword matching (Jaccard) |
| 6 | E7_Code | 1536 | Dense | Source code similarity |
| 7 | E8_Graph | 1024 | Asymmetric | Node2Vec structural (source/target dual vectors) |
| 8 | E9_HDC | 1024 | Dense | Hyperdimensional computing (noise-robust) |
| 9 | E10_Multimodal | 768 | Asymmetric | Paraphrase detection (doc/query dual vectors) |
| 10 | E11_Entity | 768 | Dense | Named entity matching |
| 11 | E12_ColBERT | 128/token | Late interaction | Stage 3 reranking only (never in fusion weights) |
| 12 | E13_SPLADE | 30K | Sparse | Stage 1 recall only (never in fusion weights) |

### Key Architectural Rules (from constitution.yaml)

- **ARCH-SIGNAL-01**: "ALL 13 embedders provide SIGNAL, never noise." — E5 currently violates this.
- **ARCH-12**: "E1 is foundation — all retrieval starts with E1."
- **AP-77**: "E5 MUST NOT use symmetric cosine — causal is directional." Direction modifiers: cause_to_effect=1.2, effect_to_cause=0.8.
- **ARCH-21**: "Multi-space fusion: Weighted RRF, not weighted sum." (Note: RRF is used for candidate selection; final user scores use weighted cosine average via `compute_semantic_fusion()`.)
- **Testing golden rule**: "ALL MCP tests use real RocksDB + real GPU embeddings. NO STUBS."
- **Error handling**: "FAIL FAST — no silent degradation."
- **Serialization**: "bincode + skip_serializing_if = SILENT CORRUPTION. Use JSON for provenance/metadata."

### E5 Asymmetric Vector Storage

Each memory stores TWO E5 vectors in `SemanticFingerprint`:
- `e5_causal_as_cause: Vec<f32>` — 768D, how this memory looks as a CAUSE
- `e5_causal_as_effect: Vec<f32>` — 768D, how this memory looks as an EFFECT
- Accessors: `get_e5_as_cause()`, `get_e5_as_effect()` (fall back to legacy `e5_causal` if empty)
- **No `causal_hint` struct exists.** Direction is inferred at query time via `infer_result_causal_direction()` using vector norm comparison (>10% threshold).
- E1 is accessed via the PUBLIC field `e1_semantic: Vec<f32>` (1024D). There is NO `get_e1()` accessor — you must add one or access the field directly.

### Data Flow: How a Causal Query Works

```
User: "What causes lung cancer?" → search_graph MCP tool
  1. detect_causal_query_intent() → CausalDirection::Cause (keyword-based, 96% accuracy)
  2. If causal → set weight_profile = "causal_reasoning" (E5=0.45, E1=0.20)
  3. search_semantic() → multi_space strategy:
     a. Each active embedder's HNSW index queried independently
     b. RRF fusion selects top-K candidates (rrf_contribution = weight / (rank + 1 + 60))
     c. compute_semantic_fusion() computes final score = sum(score_i * weight_i) / sum(weight_i)
  4. apply_asymmetric_e5_reranking() → blends E5 asymmetric similarity into scores
  5. Results returned to user with searchTransparency metadata
```

For `search_causes`/`search_effects`:
```
  1. Candidate retrieval via search_semantic() with causal_reasoning profile
  2. rank_causes_by_abduction() or rank_effects_by_prediction() scores using E5-ONLY
  3. Results returned with E5 scores (currently broken — random rankings)
```

---

## 3. Fixes (6 total, ordered by dependency)

### Fix 1: Reweight `causal_reasoning` Profile

**Root cause**: RC-2
**File**: `crates/context-graph-core/src/weights/mod.rs` lines 141-157

The `causal_reasoning` profile is a `[f32; 13]` array in the `WEIGHT_PROFILES` const. Change:

```
BEFORE: [0.20, 0.0, 0.0, 0.0, 0.45, 0.05, 0.10, 0.10, 0.0, 0.05, 0.05, 0.0, 0.0]
AFTER:  [0.40, 0.0, 0.0, 0.0, 0.10, 0.05, 0.15, 0.10, 0.0, 0.10, 0.10, 0.0, 0.0]
```

| Index | Embedder | Before | After | Rationale |
|-------|----------|--------|-------|-----------|
| 0 | E1_Semantic | 0.20 | 0.40 | Proven 3/3 top-1 correct, 17x better discrimination |
| 4 | E5_Causal | 0.45 | 0.10 | Demoted — provides binary structure signal only |
| 6 | E7_Code | 0.10 | 0.15 | Handles technical/scientific causal text |
| 9 | E10_Multimodal | 0.05 | 0.10 | Paraphrase matching helps find same-concept causes |
| 10 | E11_Entity | 0.05 | 0.10 | Entity-aware discrimination (found deforestation/crime in tests) |

Sum = 1.00. No other profiles need changes.

### Fix 2: Replace Continuous E5 Blending with Binary Causal Gate

**Root cause**: RC-1, RC-2
**Files**: `crates/context-graph-core/src/causal/asymmetric.rs` (add), `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs` (modify)

**Why**: E5 scores cluster 0.93-0.98 for causal text and 0.90-0.94 for some non-causal text. A binary gate converts this compressed continuous signal into a useful boost/demotion.

**Part A** — Add to `asymmetric.rs` (after the existing `direction_mod` module at line ~51):

```rust
/// Causal content gate thresholds.
///
/// E5 scores cluster 0.93-0.98 for causal text and 0.90-0.94 for non-causal.
/// These thresholds convert the compressed continuous signal into a binary gate.
pub mod causal_gate {
    /// Minimum E5 score to consider content "definitely causal"
    pub const CAUSAL_THRESHOLD: f32 = 0.94;
    /// Maximum E5 score to consider content "definitely non-causal"
    pub const NON_CAUSAL_THRESHOLD: f32 = 0.92;
    /// Boost applied to results that pass the causal gate (for causal queries)
    pub const CAUSAL_BOOST: f32 = 1.05;
    /// Demotion applied to results that fail the causal gate (for causal queries)
    pub const NON_CAUSAL_DEMOTION: f32 = 0.90;
}

/// Apply E5 causal gating to a result score.
///
/// Converts E5's compressed continuous score into a binary boost/demotion.
/// Between CAUSAL_THRESHOLD and NON_CAUSAL_THRESHOLD: no change (ambiguous).
pub fn apply_causal_gate(original_score: f32, e5_score: f32, is_causal_query: bool) -> f32 {
    if !is_causal_query {
        return original_score;
    }
    if e5_score >= causal_gate::CAUSAL_THRESHOLD {
        original_score * causal_gate::CAUSAL_BOOST
    } else if e5_score <= causal_gate::NON_CAUSAL_THRESHOLD {
        original_score * causal_gate::NON_CAUSAL_DEMOTION
    } else {
        original_score
    }
}
```

**Part B** — Modify `apply_asymmetric_e5_reranking()` in `memory_tools.rs` (line 1683). Replace the body — remove the continuous blending formula, use the binary gate instead:

```rust
fn apply_asymmetric_e5_reranking(
    results: &mut [TeleologicalSearchResult],
    query_embedding: &SemanticFingerprint,
    query_direction: CausalDirection,
    _e5_weight: f32, // Kept for signature stability, not used in gate logic
) {
    if results.is_empty() {
        return;
    }
    let is_causal = !matches!(query_direction, CausalDirection::Unknown);

    for result in results.iter_mut() {
        let query_is_cause = matches!(query_direction, CausalDirection::Cause);
        let e5_sim = compute_e5_asymmetric_fingerprint_similarity(
            query_embedding,
            &result.fingerprint.semantic,
            query_is_cause,
        );
        result.similarity = apply_causal_gate(result.similarity, e5_sim, is_causal);
    }

    results.sort_by(|a, b| {
        b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal)
    });
}
```

This removes: `get_direction_modifier()`, `infer_result_causal_direction()`, and the blending formula. Those functions can be deleted or kept for other uses.

### Fix 3: Direction-Aware Reranking via Keyword Detection

**Root cause**: RC-3
**File**: `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs`

**Why**: `detect_causal_query_intent()` has 96% accuracy (keyword-based). E5 has 0% topical accuracy. Use the query's detected direction to boost results whose E5 vector norms suggest matching direction.

Add a new function and call it AFTER `apply_asymmetric_e5_reranking()` at line ~1204:

```rust
/// Direction-aware reranking using keyword-detected query direction.
///
/// Uses infer_result_causal_direction() (E5 vector norm comparison) to determine
/// if a result describes a cause or effect, then boosts results whose direction
/// matches what the query seeks.
fn apply_direction_aware_reranking(
    results: &mut [TeleologicalSearchResult],
    query_direction: CausalDirection,
) {
    if matches!(query_direction, CausalDirection::Unknown) || results.is_empty() {
        return;
    }

    const DIRECTION_MATCH_BOOST: f32 = 1.08;

    for result in results.iter_mut() {
        let result_dir = infer_result_causal_direction(&result.fingerprint.semantic);
        let boost = match (&query_direction, &result_dir) {
            (CausalDirection::Cause, CausalDirection::Cause) => DIRECTION_MATCH_BOOST,
            (CausalDirection::Effect, CausalDirection::Effect) => DIRECTION_MATCH_BOOST,
            _ => 1.0,
        };
        result.similarity *= boost;
    }

    results.sort_by(|a, b| {
        b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal)
    });
}
```

**Integration**: At line ~1204, after the `apply_asymmetric_e5_reranking` call, add:
```rust
apply_direction_aware_reranking(&mut results, causal_direction);
```

**NOTE**: `infer_result_causal_direction()` already exists at line 1749 in the same file. It uses E5 vector norm comparison (>10% threshold). This function is kept — its norm comparison is still the best available signal for per-result direction classification (it just shouldn't be used for continuous scoring).

### Fix 4: E1-Anchored Scoring in rank_causes/rank_effects

**Root cause**: RC-1, RC-3
**File**: `crates/context-graph-core/src/causal/chain.rs` (lines 339-365 and 426-452)

**Why**: `rank_causes_by_abduction()` and `rank_effects_by_prediction()` use E5-ONLY for scoring. Since E5 can't discriminate topics, these return random rankings. Blend in E1 as the primary signal.

**Important code facts**:
- `cosine_similarity()` is defined LOCALLY in chain.rs at line 546 (not imported, not named `_f32`)
- `SemanticFingerprint.e1_semantic` is a public field (Vec<f32>, 1024D) — use `&fp.e1_semantic`
- `compute_e5_asymmetric_fingerprint_similarity()` is imported from asymmetric.rs

**Change `rank_causes_by_abduction()` (line 348-352)**:

```rust
// BEFORE:
let raw_sim =
    compute_e5_asymmetric_fingerprint_similarity(effect_fingerprint, cause_fp, false);
let adjusted_score = raw_sim * direction_mod::EFFECT_TO_CAUSE;

// AFTER:
let e5_sim =
    compute_e5_asymmetric_fingerprint_similarity(effect_fingerprint, cause_fp, false);
let e1_sim = cosine_similarity(&effect_fingerprint.e1_semantic, &cause_fp.e1_semantic);
let blended = 0.80 * e1_sim + 0.20 * e5_sim;
let adjusted_score = blended * direction_mod::EFFECT_TO_CAUSE;
```

Also update `raw_similarity` in the `AbductionResult` struct: set it to `e5_sim` (preserves backwards compat of that field).

**Change `rank_effects_by_prediction()` (line 435-439)**:

```rust
// BEFORE:
let raw_sim =
    compute_e5_asymmetric_fingerprint_similarity(cause_fingerprint, effect_fp, true);
let adjusted_score = (raw_sim * direction_mod::CAUSE_TO_EFFECT).clamp(0.0, 1.0);

// AFTER:
let e5_sim =
    compute_e5_asymmetric_fingerprint_similarity(cause_fingerprint, effect_fp, true);
let e1_sim = cosine_similarity(&cause_fingerprint.e1_semantic, &effect_fp.e1_semantic);
let blended = 0.80 * e1_sim + 0.20 * e5_sim;
let adjusted_score = (blended * direction_mod::CAUSE_TO_EFFECT).clamp(0.0, 1.0);
```

Also update `raw_similarity` in the `PredictionResult`: set it to `e5_sim`.

### Fix 5: Variance-Based Weight Suppression in Fusion

**Root cause**: RC-2
**File**: `crates/context-graph-storage/src/teleological/rocksdb_store/search.rs`

**Why**: Defense-in-depth. If ANY embedder (not just E5) produces near-constant scores across results, its weight should be automatically reduced. This prevents future degenerate embedders from poisoning fusion.

Add a new function that wraps `compute_semantic_fusion()` (defined at line 1106):

```rust
/// Suppress embedders with near-zero score variance before fusion.
///
/// If an embedder produces nearly identical scores for all candidates,
/// it contributes noise, not signal. Reduce its weight.
fn suppress_degenerate_weights(
    all_scores: &[&[f32; 13]],
    weights: &[f32; 13],
) -> [f32; 13] {
    const MIN_VARIANCE: f32 = 0.001;
    const SUPPRESSION_FACTOR: f32 = 0.25;

    if all_scores.len() < 3 {
        return *weights;
    }

    let mut adjusted = *weights;
    let n = all_scores.len() as f32;

    for idx in 0..13 {
        if weights[idx] <= 0.0 {
            continue;
        }
        let scores: Vec<f32> = all_scores.iter()
            .map(|s| s[idx])
            .filter(|s| *s > 0.0)
            .collect();
        if scores.len() < 3 {
            continue;
        }
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores.iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f32>() / scores.len() as f32;
        if variance < MIN_VARIANCE {
            adjusted[idx] *= SUPPRESSION_FACTOR;
        }
    }

    // Renormalize
    let total: f32 = adjusted.iter().sum();
    if total > 0.0 {
        for w in adjusted.iter_mut() {
            *w /= total;
        }
    }
    adjusted
}
```

**Integration**: This needs to be called at the multi_space search paths where `compute_semantic_fusion` is called (lines 324, 629, 816 in search.rs). The challenge: those call sites score one candidate at a time. You need to collect all candidate scores FIRST, run `suppress_degenerate_weights` once, then apply the adjusted weights. This requires refactoring the scoring loop to be two-pass:
1. First pass: collect all `[f32; 13]` score arrays
2. Call `suppress_degenerate_weights` once
3. Second pass: call `compute_semantic_fusion` with adjusted weights

This is the most invasive fix. If the two-pass refactor is too complex, an alternative is to just rely on Fix 1 (reduced weight) and skip this fix — the weight reduction from 0.45→0.10 already achieves 80% of the benefit.

### Fix 6: Change Candidate Retrieval Profile for search_causes/search_effects

**Root cause**: RC-1
**File**: `crates/context-graph-mcp/src/handlers/tools/causal_tools.rs` (lines 152 and 494)

**Why**: Both `search_causes` and `search_effects` use `causal_reasoning` profile for candidate retrieval, meaning E5 at 0.45 dominates which candidates are even fetched. After Fix 1, E1 will dominate at 0.40, but changing to `semantic_search` (E1=0.33) is even better because the E5 direction modifiers are applied separately in the post-retrieval ranking step.

**Change line 152** (search_causes):
```rust
// BEFORE:
.with_weight_profile("causal_reasoning")
// AFTER:
.with_weight_profile("semantic_search")
```

**Change line 494** (search_effects):
```rust
// BEFORE:
.with_weight_profile("causal_reasoning")
// AFTER:
.with_weight_profile("semantic_search")
```

The `rank_causes_by_abduction()` / `rank_effects_by_prediction()` functions (Fix 4) handle E5 direction modifiers in the post-retrieval scoring step.

---

## 4. Dependency Order

```
Fix 1 (reweight)              ← No dependencies. Do first. 1-line change.
Fix 6 (candidate source)      ← No dependencies. Do with Fix 1. 2-line change.
Fix 4 (E1-anchored scoring)   ← No dependencies. ~10 lines changed in chain.rs.
Fix 2 (causal gating)         ← Replaces current reranking logic. ~40 lines new + ~30 lines replaced.
Fix 3 (direction reranking)   ← After Fix 2 (uses same call site). ~25 lines new + 1 line integration.
Fix 5 (variance guard)        ← Independent, most invasive. Can be deferred.
```

Recommended implementation order: **1 → 6 → 4 → 2 → 3 → 5**

---

## 5. Success Criteria (Measurable)

After implementing fixes, verify with the same test data:

| Metric | Current (broken) | Target (fixed) | How to Measure |
|--------|-----------------|----------------|----------------|
| `search_graph` top-1 for "What causes lung cancer?" | E5 ranks "x" first | Correct causal memory ranks #1 | search_graph with searchTransparency=true |
| `search_graph` top-1 for "What are the effects of CO2?" | Score 0.770 | Score > 0.80 | search_graph with searchTransparency=true |
| Ablation: E5 inclusion effect | -0.009 (hurts) | >= 0.0 (neutral or helps) | Compare with/without excludeEmbedders=["E5"] |
| Score spread in causal_reasoning profile | 0.029 avg | > 0.05 | Max score minus score at rank 5 |
| `search_causes` top-1 for "CO2 emissions" | Random (E5-only ranking) | CO2→warming memory ranks #1 | search_causes tool |
| Neutral confound rank for causal queries | Same rank as causal content | Lower than causal content | search_graph: compare neutral vs causal result scores |
| Direction modifier creates ranking change | No rank changes | At least 1 rank swap per query | Compare search_effects vs search_causes |

---

## 6. Full State Verification Protocol

Do NOT rely on return values alone. After each fix, verify the physical state:

### 6.1 Source of Truth Check

After storing test memories, verify they exist in RocksDB:
```
store_memory → get memory ID → get_memory_fingerprint(id) →
  Verify: e5_causal_as_cause has 768 non-zero floats
  Verify: e5_causal_as_effect has 768 non-zero floats
  Verify: e1_semantic has 1024 non-zero floats
```

### 6.2 Search Path Verification

For each search query:
```
search_graph(query, searchTransparency=true) →
  Verify: effectiveProfile matches expected (causal_reasoning or null)
  Verify: causal.direction matches expected (cause/effect/unknown)
  Verify: activeEmbedders shows correct weights per profile
  Verify: results[0] is the topically correct answer
  Verify: results[0].similarity > results[4].similarity by > 0.05
```

### 6.3 Before/After Comparison

Run the EXACT same queries before and after fixes. Compare:
- Top-1 memory ID (should change for broken queries)
- Score spread (should increase)
- E5 weight in transparency report (should decrease from 0.45 to 0.10)

### 6.4 Edge Cases to Verify

1. **Empty query** → should return proper error, not crash
2. **Non-causal query** ("Tell me about the Arctic") → direction=unknown, NO causal_reasoning profile applied, E5 weight stays at default profile level
3. **Single-word causal query** ("why?") → should detect as CausalDirection::Cause
4. **Negation** ("does NOT cause") → known limitation (substring matching returns Cause; document this)

---

## 7. Synthetic Test Data

Store these memories (use `store_memory` MCP tool). These have known expected outputs.

### 7.1 Test Memories

```
ID: M1  Content: "Smoking causes lung cancer through tar buildup in lung tissue over decades of exposure"
        Domain: Medical, Direction: Forward causal (cause→effect)
        Expected: Top-1 for "What causes lung cancer?"

ID: M2  Content: "Cirrhosis is caused by sustained liver inflammation from hepatitis C viral infection"
        Domain: Medical, Direction: Backward causal (effect←cause)
        Expected: Top-1 for "What causes cirrhosis?"

ID: M3  Content: "The liver is the largest internal organ in the human body, weighing approximately 1.5 kilograms"
        Domain: Medical, Direction: None (neutral confound)
        Expected: Should rank BELOW M1 and M2 for any medical causal query

ID: M4  Content: "Increased atmospheric CO2 concentration drives global temperature rise through the greenhouse effect"
        Domain: Climate, Direction: Forward causal
        Expected: Top-1 for "What are the effects of CO2 emissions?"

ID: M5  Content: "Glacial melting results from rising global temperatures weakening polar ice sheet integrity"
        Domain: Climate, Direction: Backward causal
        Expected: Top-1 for "What causes glacial melting?"

ID: M6  Content: "The Arctic region experiences six months of continuous daylight during summer solstice"
        Domain: Climate, Direction: None (neutral confound)
        Expected: Top-1 for "Tell me about the Arctic climate" (non-causal), low rank for causal queries

ID: M7  Content: "Poverty reduces access to quality education, perpetuating intergenerational cycles of disadvantage"
        Domain: Social, Direction: Forward causal
        Expected: Top-1 for "What are the effects of poverty?"

ID: M8  Content: "Urban crime rates are driven by socioeconomic inequality and lack of community investment"
        Domain: Social, Direction: Backward causal
        Expected: Top-1 for "What causes urban crime?"

ID: M9  Content: "The world population reached eight billion people in November 2022"
        Domain: Social, Direction: None (neutral confound)
        Expected: Low rank for all causal queries

ID: M10 Content: "Dopamine release in the nucleus accumbens reinforces addictive behaviors through reward pathway sensitization"
        Domain: Neuroscience, Direction: Forward causal
        Expected: Top-1 for "What causes addiction?"
```

### 7.2 Expected Query Results (post-fix)

| Query | Expected Top-1 | Expected Direction | Expected Profile |
|-------|---------------|-------------------|-----------------|
| "What causes lung cancer?" | M1 (smoking/tar) | Cause | causal_reasoning |
| "What are the effects of CO2 emissions?" | M4 (CO2/warming) | Effect | causal_reasoning |
| "What are the effects of poverty?" | M7 (poverty/education) | Effect | causal_reasoning |
| "What causes glacial melting?" | M5 (glacial/temperatures) | Cause | causal_reasoning |
| "Tell me about the Arctic climate" | M6 (Arctic daylight) | Unknown | default (no causal profile) |
| "What causes addiction?" | M10 (dopamine/addiction) | Cause | causal_reasoning |

---

## 8. Build & Test Commands

```bash
# Build (must succeed with zero errors before any commit)
cargo build --release

# Run all tests
cargo test --release

# Run specific MCP tests
cargo test -p context-graph-mcp --release

# Run specific causal tests
cargo test -p context-graph-core causal --release

# Run specific chain tests
cargo test -p context-graph-core chain --release

# Test counts (approximate):
# - MCP: ~700 tests
# - Core: ~2800 tests
# - Storage: ~630 tests
```

After building, the MCP server binary must be restarted to pick up changes:
- The MCP server is a SEPARATE BINARY from the test harness
- `cargo build --release` builds everything
- MCP server path: `target/release/context-graph-mcp`

---

## 9. What NOT to Do

1. **Do NOT remove E5 entirely.** The dual-vector asymmetric architecture is correct. The model is bad, not the architecture. Keep E5 at reduced weight for when a better model is swapped in.

2. **Do NOT add backwards compatibility shims.** If `apply_asymmetric_e5_reranking()` changes its behavior, that's the fix. No feature flags, no fallback paths.

3. **Do NOT add whitening transforms.** While whitening (PCA-based score decompression) is a known fix for anisotropy in the literature, it requires storing and maintaining a covariance matrix per embedder. This is over-engineering for a 0.10-weight embedder. The binary gate (Fix 2) achieves the same practical result with zero maintenance.

4. **Do NOT fine-tune or replace the Longformer model.** That's a future task. This task fixes the integration layer to minimize damage from the current degenerate model.

5. **Do NOT use mock data in tests.** All tests use real RocksDB + real GPU embeddings via `create_test_handlers()`.

6. **Do NOT save test files to the root folder.** Tests go in `tests/` or `#[cfg(test)] mod tests` blocks within the relevant crate.

---

## 10. Research Context (Why These Fixes Work)

### Anisotropy (Root Cause)
Pre-trained transformers like Longformer produce embeddings that occupy a narrow cone in high-dimensional space. Cosine similarity between any two vectors is high (0.9+) because all vectors point roughly the same direction. This is well-documented (Ethayarajh 2019, "How Contextual are Contextualized Word Representations?"). The fix is either: (a) fine-tune with contrastive loss (out of scope), or (b) treat the compressed signal as binary (Fix 2).

### Variance-Based Suppression
Automatically detecting and suppressing degenerate embedders by measuring per-embedder score variance across candidates is a standard technique in ensemble fusion. If variance < threshold, the embedder is adding noise. This generalizes beyond E5 to protect against any future degenerate model.

### Binary Gating vs Continuous Scoring
When an embedder's scores are compressed into a narrow band (0.93-0.98), using them as continuous ranking signals injects noise. Converting to a binary gate (above/below threshold) extracts the only signal that exists: "is this content causal-structured or not?" This is Occam's razor — use the simplest model that matches the actual signal.

### E1 Anchoring
E1 (general semantic similarity) has proven 3/3 correct top-1 accuracy and 17x wider score spread than E5. Making E1 the primary ranking signal and E5 a secondary nudge (80/20 blend in Fix 4) is the simplest way to get correct rankings while preserving E5's directional information.

---

## 11. Previous Stress Test Raw Data

### Score Compression Evidence

**Query: "What causes lung cancer?"**

| Rank | Content | E5 Score | E1 Score | Correct? |
|------|---------|----------|----------|----------|
| E5 #1 | "x" (single character) | 0.961 | 0.746 | No |
| E5 #2 | Heat causes thermal expansion | 0.953 | 0.789 | No |
| E5 #3 | Interest rates result in... | 0.949 | 0.707 | No |
| E5 #4 | World population (neutral) | 0.948 | 0.759 | No |
| E5 #5 | Poverty reduces education | 0.945 | 0.754 | No |
| E5 #6 | Alcohol damages liver | 0.944 | 0.805 | No |

E5 spread: 0.022. E1 spread: 0.062 (2.8x wider). E5 top-1: "x" (single character).

**Query: "What are the effects of poverty?"**

E5 spread: 0.004. E1 spread: 0.153 (38x wider). E5 top-1: "Heat causes expansion" (wrong domain).

### Ablation Evidence (Query: "What are the effects of CO2 emissions?")

| Config | Top-1 Score | Top-1 Correct? |
|--------|------------|----------------|
| E5=0.45 (current) | 0.770 | Yes |
| E5 excluded | 0.779 (+0.009) | Yes |
| E1 only | 0.822 (+0.052) | Yes |

Removing E5 INCREASES the correct answer's score. E5 is actively harmful.

### Direction Modifier Evidence

search_effects/search_causes ratio = 1.48-1.50x (matches theoretical 1.5x from 1.2/0.8). But both APIs return the same top-1: "x" (single character). The modifiers scale all scores uniformly — no ranking change.
