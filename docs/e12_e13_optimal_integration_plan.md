# E12 (ColBERT) and E13 (SPLADE) Optimal Integration Plan

**Date:** 2026-01-25
**Status:** PROPOSED
**Philosophy:** E12/E13 ENHANCE E1 - they find what semantic search MISSES

---

## Executive Summary

E12 and E13 are properly implemented but **narrowly exposed**. Only `search_graph` can use them. This plan proposes expanding their reach to all specialized tools while maintaining the "enhancer" philosophy where E12/E13 find what E1 misses, not compete with it.

---

## 1. Core Philosophy: Enhancers, Not Competitors

### What E1 (Semantic) Does Well
- General semantic similarity
- Conceptual matching ("database" ↔ "data storage")
- Context understanding

### What E1 MISSES (Where E12/E13 Add Value)

| E1 Blind Spot | E12 (ColBERT) Solution | E13 (SPLADE) Solution |
|---------------|------------------------|------------------------|
| **Exact phrase matching** | Token-level MaxSim finds "connect_db" exactly | - |
| **Technical jargon** | Preserves exact function/API names | Term expansion: "postgres" → "postgresql" |
| **Averaged-out keywords** | Token-level precision prevents dilution | BM25 IDF boosts rare terms |
| **Long queries** | Each token matched independently | Sparse recall handles many terms |
| **Acronyms** | "API" matched as tokens, not semantics | "REST" expands to related terms |

### Example: "What databases work with Rust?"

```
E1 Semantic: Finds memories with "database" OR "Rust" (semantic match)
   → MISSES: "Diesel ORM provides type-safe queries" (no "database" word)

E13 SPLADE Stage 1: Expands query terms, recalls 10x candidates
   → FINDS: "Diesel" through term expansion (ORM → database domain)

E12 ColBERT Stage 3: Token-level rerank of candidates
   → BOOSTS: Memories with exact "Rust" + database-related tokens

Combined: Better answer because E13 found Diesel, E12 ranked it higher
```

---

## 2. Current State Gap Analysis

### Tools WITH E12/E13 Access (1 tool)

| Tool | E13 Recall | E12 Rerank | How |
|------|------------|------------|-----|
| `search_graph` | ✅ | ✅ | `strategy: "pipeline"`, `enableRerank: true` |

### Tools WITHOUT E12/E13 Access (6 tools)

| Tool | Current Strategy | E12/E13 Benefit |
|------|------------------|-----------------|
| `search_code` | MultiSpace (fixed) | E12 would find exact function names E1 averages out |
| `search_by_entities` | MultiSpace (fixed) | E12 would match exact entity names ("PostgreSQL" vs "postgres") |
| `search_causes` | MultiSpace (fixed) | E12 would rerank causal chains for phrase precision |
| `search_by_keywords` | MultiSpace (fixed) | E13 already expands queries; E12 would add phrase precision |
| `search_connections` | MultiSpace (fixed) | E12 would match exact relationship phrases |
| `search_robust` | MultiSpace (fixed) | E12 would rerank typo-corrected results |

---

## 3. Proposed Integration Strategy

### Phase 1: Enable Pipeline Mode for All Specialized Tools (HIGH PRIORITY)

**Goal:** Allow specialized tools to use E13 Stage 1 recall + E12 Stage 3 reranking.

**Implementation:**

1. **Add `strategy` parameter to specialized tools:**

```typescript
// For search_code, search_causes, search_by_keywords, etc.
{
  "strategy": {
    "enum": ["default", "pipeline"],
    "default": "default",
    "description": "Search strategy. 'pipeline' enables E13 recall + E12 reranking for maximum precision."
  }
}
```

2. **Modify tool handlers to accept strategy:**

```rust
// In code_tools.rs, causal_tools.rs, etc.
let strategy = match params.strategy.as_deref() {
    Some("pipeline") => SearchStrategy::Pipeline,
    _ => SearchStrategy::MultiSpace, // Preserve backward compatibility
};

let options = TeleologicalSearchOptions::default()
    .with_strategy(strategy)
    .with_rerank(strategy == SearchStrategy::Pipeline); // Auto-enable E12 for pipeline
```

**Files to Modify:**
- `crates/context-graph-mcp/src/handlers/tools/code_tools.rs`
- `crates/context-graph-mcp/src/handlers/tools/causal_tools.rs`
- `crates/context-graph-mcp/src/handlers/tools/keyword_tools.rs`
- `crates/context-graph-mcp/src/handlers/tools/entity_tools.rs`
- `crates/context-graph-mcp/src/handlers/tools/intent_tools.rs`
- `crates/context-graph-mcp/src/handlers/tools/robustness_tools.rs`

---

### Phase 2: Implement E13 BM25 Scoring (HIGH PRIORITY)

**Goal:** Replace naive term frequency with BM25 for better E13 recall quality.

**Current Code (suboptimal):**
```rust
// rocksdb_store/search.rs line 1162
*doc_scores.entry(doc_id).or_insert(0.0) += query_weight;
// TODO: Implement BM25 or other scoring
```

**Proposed Implementation:**
```rust
/// BM25 scoring parameters (standard values)
const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

fn bm25_score(
    term_freq: f32,
    doc_len: f32,
    avg_doc_len: f32,
    idf: f32,
    query_weight: f32,
) -> f32 {
    let numerator = term_freq * (BM25_K1 + 1.0);
    let denominator = term_freq + BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / avg_doc_len));
    idf * query_weight * (numerator / denominator)
}

// In search_sparse_async():
for &doc_id in &doc_ids {
    let doc_len = self.get_doc_length(doc_id)?; // Need to track
    let idf = (total_docs as f32 / doc_ids.len() as f32).ln();
    let score = bm25_score(term_freq, doc_len, avg_doc_len, idf, query_weight);
    *doc_scores.entry(doc_id).or_insert(0.0) += score;
}
```

**Benefits:**
- Rare terms get higher IDF boost (finds unique content E1 averages)
- Document length normalization prevents long documents from dominating
- Standard scoring used by Elasticsearch, Vespa, etc.

---

### Phase 3: Expose E12 Blend Ratio (MEDIUM PRIORITY)

**Goal:** Allow users to tune E12's contribution in Stage 3 reranking.

**Current Code (hardcoded):**
```rust
// rocksdb_store/search.rs line 850
let blended = original_fusion * 0.6 + maxsim_score * 0.4;
```

**Proposed Implementation:**

1. **Add parameter to search_graph:**
```typescript
{
  "e12BlendRatio": {
    "type": "number",
    "minimum": 0.0,
    "maximum": 1.0,
    "default": 0.4,
    "description": "E12 ColBERT weight in final reranking. Higher = more token-level precision. Only used when enableRerank=true."
  }
}
```

2. **Pass through to storage layer:**
```rust
// In TeleologicalSearchOptions
pub struct TeleologicalSearchOptions {
    // ... existing fields
    pub e12_blend_ratio: f32, // New field, default 0.4
}

// In rerank_with_colbert():
let blended = original_fusion * (1.0 - options.e12_blend_ratio)
            + maxsim_score * options.e12_blend_ratio;
```

**Use Cases:**
- **Legal/compliance queries:** `e12BlendRatio: 0.8` (need exact phrases)
- **Brainstorming:** `e12BlendRatio: 0.2` (prefer semantic breadth)
- **Code search:** `e12BlendRatio: 0.6` (balance function names + semantics)

---

### Phase 4: Auto-Upgrade to Pipeline for Precision Queries (MEDIUM PRIORITY)

**Goal:** Automatically use E13 recall + E12 reranking when query patterns suggest precision is needed.

**Detection Heuristics:**
```rust
fn should_upgrade_to_pipeline(query: &str, strategy: SearchStrategy) -> bool {
    // Only upgrade from E1Only, not from user-specified strategies
    if strategy != SearchStrategy::E1Only {
        return false;
    }

    // Precision indicators
    let has_quotes = query.contains('"');
    let has_exact_keyword = query.contains("exact") || query.contains("specific");
    let is_long_query = query.split_whitespace().count() > 10;
    let has_code_pattern = query.contains("()") || query.contains("fn ") || query.contains("::");
    let has_technical_term = has_camel_case(query) || has_snake_case(query);

    has_quotes || has_exact_keyword || is_long_query || has_code_pattern || has_technical_term
}
```

**Implementation:**
```rust
// In call_search_graph():
let effective_strategy = if should_upgrade_to_pipeline(&query, strategy) {
    info!("Auto-upgrading to pipeline strategy for precision query");
    SearchStrategy::Pipeline
} else {
    strategy
};

let effective_rerank = enable_rerank || (effective_strategy == SearchStrategy::Pipeline);
```

---

### Phase 5: Create Pipeline-Aware Weight Profiles (LOW PRIORITY)

**Goal:** Optimize E1-E13 weights for each pipeline stage.

**Current Problem:** All 15 weight profiles set E12=0.0, E13=0.0 (correct for fusion, but no pipeline-specific profiles).

**Proposed Profiles:**

```rust
// weights.rs - Add new profiles

/// Stage 1: Maximize recall (E13-heavy)
const PIPELINE_STAGE1_WEIGHTS: [f32; 13] = [
    0.30,  // E1_Semantic (reduced - E13 handles recall)
    0.0,   // E2-E4 temporal (never in fusion)
    0.0, 0.0, 0.0,
    0.10,  // E5_Causal (causal recall)
    0.15,  // E6_Sparse (keyword recall)
    0.10,  // E7_Code (code pattern recall)
    0.05,  // E8_Graph
    0.0,   // E9_HDC
    0.05,  // E10_Multimodal
    0.05,  // E11_Entity
    0.0,   // E12_LateInteraction (Stage 3 only)
    0.20,  // E13_SPLADE (boosted for recall)
];

/// Stage 2: Semantic fusion (E1-heavy)
const PIPELINE_STAGE2_WEIGHTS: [f32; 13] = [
    0.50,  // E1_Semantic (dominant)
    // ... rest similar to semantic_search profile
];

/// Stage 3: Precision reranking (E12 via MaxSim, not in fusion)
// No weights needed - E12 is applied directly via MaxSim
```

---

### Phase 6: Enrichment Pipeline Integration (LOW PRIORITY)

**Goal:** When `enrichMode: "full"` is set, auto-use pipeline for maximum E12/E13 benefit.

**Implementation:**
```rust
// In enrichment_pipeline.rs

impl EnrichmentPipeline {
    fn determine_search_options(&self, enrich_mode: EnrichMode) -> TeleologicalSearchOptions {
        match enrich_mode {
            EnrichMode::Off => TeleologicalSearchOptions::default()
                .with_strategy(SearchStrategy::E1Only),
            EnrichMode::Light => TeleologicalSearchOptions::default()
                .with_strategy(SearchStrategy::MultiSpace),
            EnrichMode::Full => TeleologicalSearchOptions::default()
                .with_strategy(SearchStrategy::Pipeline)  // E13 recall
                .with_rerank(true)                        // E12 rerank
                .with_e12_blend_ratio(0.4),              // Standard blend
        }
    }
}
```

---

## 4. Implementation Priority Matrix

| Phase | Priority | Effort | Impact | Dependencies |
|-------|----------|--------|--------|--------------|
| Phase 1: Pipeline for specialized tools | HIGH | Medium | HIGH | None |
| Phase 2: E13 BM25 scoring | HIGH | Medium | MEDIUM | None |
| Phase 3: E12 blend ratio parameter | MEDIUM | Low | MEDIUM | None |
| Phase 4: Auto-upgrade precision queries | MEDIUM | Low | LOW | Phase 1 |
| Phase 5: Pipeline weight profiles | LOW | Low | LOW | Phase 1 |
| Phase 6: Enrichment integration | LOW | Low | LOW | Phase 1, 3 |

---

## 5. Constitution Compliance Checklist

All proposed changes maintain compliance with CLAUDE.md:

| Rule | Compliance |
|------|------------|
| ARCH-12: E1 is THE semantic foundation | ✅ E12/E13 enhance E1, never replace it |
| ARCH-13: E12/E13 excluded from fusion | ✅ Pipeline stages only, not weighted sum |
| AP-73: E12 reranking only | ✅ Only in Stage 3 after E1 scoring |
| AP-74: E13 recall only | ✅ Only in Stage 1 for candidate generation |
| AP-76: 3-stage pipeline | ✅ E13 → E1 → E12 preserved |
| ARCH-17: Strong E1 = refine, weak E1 = broaden | ✅ E12 refines strong E1 matches |

---

## 6. Success Metrics

### Quantitative
- **E12 Phrase Precision:** Exact phrase queries should score 20%+ higher with E12 reranking
- **E13 Recall:** Pipeline mode should recall 15%+ more relevant documents than E1Only
- **Combined Accuracy:** Pipeline + E12 should improve MRR@10 by 10-15% vs E1Only

### Qualitative
- Users can find exact function names with `search_code strategy: "pipeline"`
- Technical jargon queries return more relevant results
- Long, specific queries auto-upgrade to precision mode

---

## 7. Example Use Cases After Integration

### Use Case 1: Finding Exact Function

**Before:**
```json
{"tool": "search_code", "query": "connect_db function", "mode": "hybrid"}
// E1 semantic averages "connect_db" into general "database connection" semantics
// MISSES exact function with that name
```

**After:**
```json
{"tool": "search_code", "query": "connect_db function", "strategy": "pipeline"}
// E13 Stage 1: Recalls all memories with "connect" or "db" terms
// E1 Stage 2: Scores by semantic similarity
// E12 Stage 3: Boosts exact "connect_db" token matches
// FINDS the exact function
```

### Use Case 2: Entity Precision

**Before:**
```json
{"tool": "search_by_entities", "entities": ["PostgreSQL"]}
// E1 + E11 union finds "database" related memories
// MISSES memories saying "postgres" without "PostgreSQL"
```

**After:**
```json
{"tool": "search_by_entities", "entities": ["PostgreSQL"], "strategy": "pipeline"}
// E13 Stage 1: Expands "PostgreSQL" → "postgres", "pg", etc.
// E12 Stage 3: Boosts exact "PostgreSQL" matches
// FINDS both variants with correct ranking
```

### Use Case 3: Causal Chain Precision

**Before:**
```json
{"tool": "search_causes", "query": "what caused the authentication timeout error"}
// E5 causal finds "caused" relationships
// MISSES specific "authentication timeout" phrase matches
```

**After:**
```json
{"tool": "search_causes", "query": "what caused the authentication timeout error", "strategy": "pipeline"}
// E13 Stage 1: Expands "authentication", "timeout", "error"
// E5 + E1 Stage 2: Scores causal + semantic similarity
// E12 Stage 3: Boosts exact "authentication timeout" phrase
// FINDS more precise causal chains
```

---

## 8. Implementation Order

### Week 1: Phase 1 + Phase 2
- Add `strategy` parameter to 6 specialized tools
- Implement E13 BM25 scoring
- Add integration tests

### Week 2: Phase 3 + Phase 4
- Add `e12BlendRatio` parameter
- Implement precision query auto-detection
- Add unit tests

### Week 3: Phase 5 + Phase 6
- Create pipeline-aware weight profiles
- Integrate with enrichment pipeline
- End-to-end testing

---

## 9. Rollback Strategy

All changes are backward compatible:
- New parameters have sensible defaults
- `strategy: "default"` preserves current behavior
- No breaking changes to existing API contracts

If issues arise:
1. Disable auto-upgrade (Phase 4) first
2. Revert to hardcoded E12 blend if ratio causes issues
3. Keep E13 term frequency if BM25 causes regressions

---

## 10. Conclusion

E12 and E13 are powerful enhancers that find what E1 misses. Current integration is **correct but limited**. This plan expands their reach while maintaining the constitutional principle that they enhance E1, never compete with it.

**Expected Outcome:** Users get better answers because:
- E13 recalls more relevant candidates (term expansion)
- E12 ranks exact matches higher (token precision)
- Combined with E1's semantic understanding = superior retrieval
