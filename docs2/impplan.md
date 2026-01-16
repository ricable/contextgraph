# Dynamic Multi-Purpose Vector Implementation Plan

## Executive Summary

Transform contextgraph into a multi-space emergent topic system:

1. Works from 0 memories (graceful degradation)
2. Discovers topics via 13 parallel HDBSCAN+BIRCH clustering spaces
3. Captures Claude's descriptions + responses + MD file content as memories
4. Uses frequency/recency importance (EWMA + BM25 + Wilson)
5. Operates autonomously — NO North Star, NO IC, NO SELF_EGO_NODE
6. Detects BOTH similarity clusters AND divergence from recent activity
7. Integrates via Claude Code hooks for autonomous context injection

---

## Part 0: North Star System Removal

### Components to Remove

**MCP Tools:** `auto_bootstrap_north_star`, `get_alignment_drift`, `get_drift_history`, `trigger_drift_correction`, `get_identity_continuity`, `get_ego_state`

**Data Structures:** `NorthStar`, `GoalHierarchy`, `SelfEgoNode`, `IdentityContinuityMonitor`, `DriftDetector`, `DriftCorrector`, `north_star_alignment` field

**Constitution Rules:** AP-01, AP-26, AP-37, AP-38, AP-40, `gwt.self_ego_node` section

**Files to Delete:**
- `crates/context-graph-core/src/autonomous/north_star.rs`
- `crates/context-graph-core/src/autonomous/goal_hierarchy.rs`
- `crates/context-graph-core/src/gwt/ego_node/` (entire directory)
- `crates/context-graph-mcp/src/handlers/autonomous/bootstrap.rs`
- `crates/context-graph-mcp/src/handlers/autonomous/drift.rs`

### Replacements

| Removed | Replacement |
|---------|-------------|
| North Star | Emergent Topic Portfolio |
| SELF_EGO_NODE | Topic Profile (13D) |
| Identity Continuity | Topic Stability (churn tracking) |
| IC < 0.5 triggers dream | entropy > 0.7 + churn > 0.5 |

---

## Part 1: Memory Sources & Capture

### Three Memory Sources

| Source | What Gets Captured | Trigger |
|--------|-------------------|---------|
| Hook Descriptions | Claude's description of what it's doing | Every hook event |
| Claude Responses | End-of-session answers, significant responses | SessionEnd, Stop hooks |
| MD File Watcher | Content from created/modified .md files | File system events |

### Hook Lifecycle

| Hook | Captures | Injects | CLI Command |
|------|----------|---------|-------------|
| SessionStart | - | Portfolio summary + recent divergences | `session start` |
| UserPromptSubmit | Prompt text | Similar memories + divergence alerts | `inject-context` |
| PreToolUse | Tool description | Brief relevant context | `inject-brief` |
| PostToolUse | Tool description + any output summary | - | `capture-memory` |
| Stop | Claude's response summary | - | `capture-response` |
| SessionEnd | Session summary | - | `session end` |

### Hook Configuration (`.claude/settings.json`)

```json
{
  "hooks": {
    "SessionStart": [{"hooks": [{"type": "command", "command": "./hooks/session-start.sh", "timeout": 5000}]}],
    "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "./hooks/user-prompt-submit.sh", "timeout": 2000}]}],
    "PreToolUse": [{"matcher": "Edit|Write|Bash", "hooks": [{"type": "command", "command": "./hooks/pre-tool-use.sh", "timeout": 500}]}],
    "PostToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": "./hooks/post-tool-use.sh", "timeout": 3000}]}],
    "Stop": [{"hooks": [{"type": "command", "command": "./hooks/stop.sh", "timeout": 3000}]}],
    "SessionEnd": [{"hooks": [{"type": "command", "command": "./hooks/session-end.sh", "timeout": 30000}]}]
  }
}
```

### Memory Schema

```
Memory {
  id: UUID,
  content: String,
  source: HookDescription | ClaudeResponse | MDFileChunk,
  created_at: Timestamp,
  session_id: String,
  teleological_array: [E1..E13],  // All 13 embeddings
  chunk_metadata: Option<ChunkMetadata>
}

ChunkMetadata {
  file_path: String,
  chunk_index: u32,
  total_chunks: u32,
  word_offset: u32
}
```

### MD File Watcher

**Watch:** All `.md` files in project directory (configurable)

**Chunking Strategy:**
- Chunk size: 200 words
- Overlap: 50 words (25%)
- Preserve sentence boundaries when possible
- Each chunk becomes a separate Memory with ChunkMetadata

**On file create/modify:**
1. Read file content
2. Split into 200-word chunks with 50-word overlap
3. Embed each chunk with all 13 embedders
4. Store as Memory with source=MDFileChunk

---

## Part 2: 13-Space Embedding & Comparison

### Embedder Categories

The 13 embedders (from PRD v5 Section 3) are divided into **4 categories** based on their semantic role:

| Category | Embedders | Purpose Vectors | Role | Counts for Topics? |
|----------|-----------|-----------------|------|-------------------|
| **Semantic** | E1, E5, E6, E7, E10, E12, E13 | V_meaning, V_causality, V_selectivity, V_correctness, V_multimodality, V_precision, V_keyword | Capture meaning, concepts, intent, code logic | ✅ YES (Primary) |
| **Temporal** | E2, E3, E4 | V_freshness, V_periodicity, V_ordering | Capture WHEN things happened | ❌ NO (Metadata only) |
| **Relational** | E8, E11 | V_connectivity, V_factuality | Capture entity/graph relationships | ⚠️ SUPPORTING (weight 0.5×) |
| **Structural** | E9 | V_robustness | Capture structural patterns | ⚠️ SUPPORTING (weight 0.5×) |

**Rationale:** Temporal embedders cluster things that happened at similar times, but temporal proximity ≠ semantic relationship. Working on 3 unrelated tasks in the same hour creates temporal clusters that are NOT topics. Temporal data is valuable metadata for context enrichment but must NOT trigger topic detection.

### Category-Specific Rules

#### Semantic Embedders (E1, E5, E6, E7, E10, E12, E13) — 7 spaces
- **Primary trigger spaces** for topic detection
- Clustering here indicates actual conceptual relationship
- Count fully (1.0×) toward "agreeing spaces" threshold
- Used for: similarity detection, divergence alerts, topic formation
- Kuramoto frequencies: γ/β bands (high-frequency semantic processing)

#### Temporal Embedders (E2, E3, E4) — 3 spaces
- **Metadata only** — NEVER count for topic detection
- NEVER trigger divergence alerts (working at different times is not divergence)
- NEVER count toward "≥N spaces agreeing" topic threshold
- Kuramoto frequencies: α band (8Hz) — rhythmic temporal processing
- Used for:
  - **E2 Temporal-Recent (V_freshness):** How recently something occurred
  - **E3 Temporal-Periodic (V_periodicity):** Cyclical patterns (daily, weekly, etc.)
  - **E4 Temporal-Positional (V_ordering):** Sequence/order relationships
  - Recency weighting in importance scoring
  - Session correlation (what happened in same session)
  - Timeline enrichment in injected context ("📅 From same session")

#### Relational Embedders (E8 Graph, E11 Entity) — 2 spaces
- **Supporting spaces** — can reinforce topics but not define them alone
- Count at 0.5× weight toward agreeing spaces
- Rationale: Same entities/relationships may appear in unrelated contexts
- Used for:
  - **E8 Graph/GNN (V_connectivity):** Graph structure relationships
  - **E11 Entity/TransE (V_factuality):** Named entity relationships
  - Entity-based retrieval ("other memories mentioning [entity]")
  - Relationship enrichment in context

#### Structural Embedders (E9 HDC) — 1 space
- **Supporting space** — captures structural/format patterns
- Count at 0.5× weight toward agreeing spaces
- Rationale: Structural similarity (e.g., list formats) doesn't imply topic similarity
- Used for:
  - **E9 HDC (V_robustness):** Hyperdimensional computing for structure
  - Format-aware retrieval
  - Noise-robust pattern matching

### Per-Embedder Specifications (from PRD v5)

| ID | Name | Dim | Purpose Vector | Category | Distance | Topic Weight | Kuramoto ω |
|----|------|-----|----------------|----------|----------|--------------|------------|
| E1 | Semantic (Matryoshka) | 1024D | V_meaning | Semantic | Cosine | 1.0 | 40Hz γ |
| E2 | Temporal-Recent | 512D | V_freshness | Temporal | Cosine | 0.0 | 8Hz α |
| E3 | Temporal-Periodic | 512D | V_periodicity | Temporal | Cosine | 0.0 | 8Hz α |
| E4 | Temporal-Positional | 512D | V_ordering | Temporal | Cosine | 0.0 | 8Hz α |
| E5 | Causal (asymmetric) | 768D | V_causality | Semantic | Asymmetric KNN | 1.0 | 25Hz β |
| E6 | Sparse | ~30K (5% active) | V_selectivity | Semantic | Jaccard | 1.0 | 4Hz θ |
| E7 | Code (AST) | 1536D | V_correctness | Semantic | Cosine | 1.0 | 25Hz β |
| E8 | Graph/GNN | 384D | V_connectivity | Relational | TransE | 0.5 | 12Hz α-β |
| E9 | HDC | 10K→1024D | V_robustness | Structural | Hamming | 0.5 | 80Hz γ+ |
| E10 | Multimodal | 768D | V_multimodality | Semantic | Cosine | 1.0 | 40Hz γ |
| E11 | Entity/TransE | 384D | V_factuality | Relational | TransE | 0.5 | 15Hz β |
| E12 | Late-Interaction | 128D/token | V_precision | Semantic | MaxSim | 1.0 | 60Hz γ+ |
| E13 | SPLADE | ~30K sparse | V_keyword | Semantic | Jaccard | 1.0 | 4Hz θ |

### ΔS Computation Methods (from PRD Section 12)

| Space | Method | Notes |
|-------|--------|-------|
| E1 Semantic | GMM + Mahalanobis | ΔS = 1 - P(e\|GMM) |
| E2-E4 Temporal | KNN | ΔS = σ((d_k - μ) / σ_d) |
| E5 Causal | Asymmetric KNN | ΔS = d_k × direction_mod |
| E6 Sparse | IDF | ΔS = IDF(active_dims) |
| E7 Code | GMM + KNN hybrid | ΔS = 0.5×GMM + 0.5×KNN |
| E9 HDC | Hamming | ΔS = min_hamming / dim |
| E11 Entity | TransE | ΔS = \|\|h + r - t\|\| |
| E12 Late | Token KNN | ΔS = max_token(d_k) |
| E13 SPLADE | Jaccard | ΔS = 1 - jaccard(active) |

### Similarity Detection (What to Inject)

A memory is **relevant** if it shows high similarity in **ANY semantic or supporting** embedding space.

**IMPORTANT:** Temporal embedders (E2-E4) are EXCLUDED from relevance detection — temporal proximity alone does not make a memory relevant.

```
relevant(query, memory) = ANY(
  // Semantic spaces (primary)
  similarity(query.E1, memory.E1) > threshold_E1,
  similarity(query.E5, memory.E5) > threshold_E5,
  similarity(query.E6, memory.E6) > threshold_E6,
  similarity(query.E7, memory.E7) > threshold_E7,
  similarity(query.E10, memory.E10) > threshold_E10,
  similarity(query.E12, memory.E12) > threshold_E12,
  similarity(query.E13, memory.E13) > threshold_E13,
  // Supporting spaces (relational + structural)
  similarity(query.E8, memory.E8) > threshold_E8,
  similarity(query.E9, memory.E9) > threshold_E9,
  similarity(query.E11, memory.E11) > threshold_E11
)
// NOTE: E2, E3, E4 (temporal) deliberately excluded
```

**Per-space thresholds** (tunable):
| Space | Category | High Similarity | Low Similarity (Divergence) |
|-------|----------|-----------------|----------------------------|
| E1 Semantic | Semantic | > 0.75 | < 0.3 |
| E2-E4 Temporal | Temporal | N/A (metadata only) | N/A (no divergence) |
| E5 Causal | Semantic | > 0.70 | < 0.25 |
| E6 Sparse | Semantic | > 0.60 | < 0.2 |
| E7 Code | Semantic | > 0.80 | < 0.35 |
| E8 Graph | Relational | > 0.65 | N/A |
| E9 HDC | Structural | > 0.70 | N/A |
| E10 Multimodal | Semantic | > 0.70 | < 0.3 |
| E11 Entity | Relational | > 0.65 | N/A |
| E12 Late-Interaction | Semantic | > 0.70 | < 0.3 |
| E13 SPLADE | Semantic | > 0.60 | < 0.2 |

### Divergence Detection (Contradiction Alert)

Compare current activity to **recent** memories (last 2 hours / current session).

**CRITICAL:** Only **Semantic embedders** (E1, E5, E6, E7, E10, E12, E13) participate in divergence detection.

- Temporal embedders EXCLUDED: Working at different times is not divergence
- Relational embedders EXCLUDED: Different entities/relationships is not inherently divergent
- Structural embedders EXCLUDED: Different structure is not semantic divergence

```
divergent(current, recent_memory) = ANY(
  similarity(current.Ei, recent_memory.Ei) < low_threshold_Ei
  for i in SEMANTIC_EMBEDDERS  // Only E1, E5, E6, E7, E10, E12, E13
)
```

**Divergence signals:**
- Current query has LOW similarity to recent activity in **semantic** spaces
- Indicates: context switch, contradictory approach, or working on something different

**Injection format:**
```
⚠️ DIVERGENCE DETECTED
Recent activity in [semantic_space]: "[memory content summary]"
Current appears different - similarity: 0.23
```

### Temporal Context Enrichment

While temporal embedders don't trigger similarity/divergence, they provide valuable context:

```
temporal_context(memory) = {
  same_session: similarity(current.E2, memory.E2) > 0.8,
  same_day: similarity(current.E3, memory.E3) > 0.7,
  same_period: similarity(current.E4, memory.E4) > 0.6
}
```

**Injection uses:**
- "📅 From same session" badge on related memories
- "🕐 You also worked on X around this time" enrichment
- Session-based grouping in context window

### Multi-Space Relevance Score

```
relevance_score = Σ (category_weight_i × embedder_weight_i × max(0, similarity_i - threshold_i))

Category weights:
  SEMANTIC (E1, E5, E6, E7, E10, E12, E13): 1.0
  TEMPORAL (E2, E3, E4): 0.0  // Excluded from relevance scoring
  RELATIONAL (E8, E11): 0.5
  STRUCTURAL (E9): 0.5

Embedder weights (within category):
  E1 Semantic: 1.0
  E5 Causal: 0.9
  E7 Code: 0.85
  E6 Sparse: 0.7
  E10 Multimodal: 0.8
  E12 Late-Interaction: 0.75
  E13 SPLADE: 0.7
  E8 Graph: 0.6
  E9 HDC: 0.5
  E11 Entity: 0.6
```

**Note:** Temporal embedders contribute 0.0 to relevance score but are still computed and stored for context enrichment.

---

## Part 3: Injection Strategy

### What Gets Injected

| Priority | Type | Condition | Token Budget |
|----------|------|-----------|--------------|
| 1 | Divergence Alerts | Low similarity to recent in **semantic spaces** | ~200 |
| 2 | High-Relevance Topics | weighted_agreement ≥ 2.5 (topic clusters) | ~400 |
| 3 | Related Memories | weighted_agreement ∈ [1.0, 2.5) | ~300 |
| 4 | Recent Context | Last session summary | ~200 |
| 5 | Temporal Enrichment | Same-session/same-period badges (metadata) | ~50 |

**Note:** Temporal embedders (E2-E4) NEVER appear in Priority 1-3. They only contribute to Priority 5 as contextual metadata badges.

### Injection Priority Algorithm

```
priority = relevance_score × recency_factor × weighted_diversity_bonus

recency_factor:
  < 1 hour = 1.3x
  < 1 day = 1.2x
  < 7 days = 1.1x
  < 30 days = 1.0x
  > 90 days = 0.8x

weighted_diversity_bonus (based on weighted_agreement, not raw space count):
  weighted_agreement ≥ 5.0 = 1.5x  (strong topic signal)
  weighted_agreement ∈ [2.5, 5.0) = 1.2x  (topic threshold met)
  weighted_agreement ∈ [1.0, 2.5) = 1.0x  (related)
  weighted_agreement < 1.0 = 0.8x  (weak)

# NOTE: Temporal spaces (E2-E4) do NOT contribute to weighted_agreement
# Temporal proximity is captured separately for badge enrichment
```

### Context Window Format

```
## Relevant Context

### Recent Related Work
[High-relevance memories from clusters]

### Potentially Related
[Single-space matches]

### ⚠️ Note: Activity Shift Detected
[Divergence alerts if any]
```

---

## Part 4: Multi-Space Clustering

### Cluster Formation

- HDBSCAN for batch clustering per embedding space
- BIRCH CF-trees for online updates
- min_cluster_size = 3
- silhouette_score > 0.3

### Cross-Space Topic Synthesis

Topics are formed when memories cluster together in **semantic** embedding spaces. Temporal clustering is explicitly excluded.

**Topic Formation Rule:**
```
weighted_agreement = Σ (topic_weight_i × is_clustered_i)

Where:
  SEMANTIC embedders (E1, E5, E6, E7, E10, E12, E13): topic_weight = 1.0
  TEMPORAL embedders (E2, E3, E4): topic_weight = 0.0  // NEVER count
  RELATIONAL embedders (E8, E11): topic_weight = 0.5
  STRUCTURAL embedder (E9): topic_weight = 0.5

Max possible weighted_agreement = 7×1.0 + 2×0.5 + 1×0.5 = 8.5
```

**Topic Detection Threshold:**
```
is_topic = weighted_agreement >= 2.5

// Examples:
// 3 semantic spaces agreeing = 3.0 ✅ TOPIC
// 2 semantic + 1 relational = 2.5 ✅ TOPIC
// 2 semantic spaces only = 2.0 ❌ NOT TOPIC
// 5 temporal spaces = 0.0 ❌ NOT TOPIC (temporal excluded)
// 1 semantic + 3 relational = 1.0 + 1.5 = 2.5 ✅ TOPIC
```

**Topic Confidence:**
```
topic_confidence = weighted_agreement / 8.5  // Normalized to [0, 1]
```

**Topic Profile (13D):**
```
topic_profile = [strength_E1, strength_E2, ..., strength_E13]
// All 13 dimensions stored, but only semantic/relational/structural used for detection
// Temporal dimensions stored for context enrichment only
```

### Cluster-Based Retrieval

1. Identify which clusters the current query belongs to (per space)
2. Retrieve other members of those clusters
3. Rank by multi-space overlap (member of same cluster in more spaces = higher rank)

---

## Part 5: Progressive Feature Activation

| Tier | Memories | Features |
|------|----------|----------|
| 0 | 0 | Storage, basic retrieval |
| 1 | 1-2 | Pairwise similarity |
| 2 | 3-9 | Basic clustering |
| 3 | 10-29 | Multiple clusters, divergence detection |
| 4 | 30-99 | Reliable statistics |
| 5 | 100-499 | Sub-clustering, trend analysis |
| 6 | 500+ | Full personalization |

**Tier 0-2 Defaults:** Cluster=-1, TopicProfile=[0.5;13], Stability=1.0

---

## Part 6: Importance Scoring

```
Importance = Frequency_Score × Recency_Weight

Frequency_Score = BM25_saturated(log(1+access_count))
  BM25: tf_saturated = (freq × (k1+1)) / (freq + k1), k1=1.2

Recency_Weight = e^(-λ × days)
  λ = ln(2) / 45 (45-day half-life)

Burst_Damping = min(freq, μ + 1.5σ)
```

---

## Part 7: Bias Mitigation

### Thompson Sampling (15% exploration)
- Beta(α, β) per cluster; new clusters start Beta(1,1)
- Sample θᵢ ~ Beta(αᵢ, βᵢ), select highest
- Update: accessed → α+=1, not accessed → β+=1

### MMR Diversity (λ=0.7)
- MMR = λ × relevance - (1-λ) × max_similarity_to_selected

### Inverse Propensity Scoring
- weight = 1 / propensity
- propensity = exposure_count / total_queries (floor 0.1)

---

## Part 8: Topic Stability

```
TopicStabilityTracker {
  topics: HashMap<TopicId, TopicMetrics>,
  history: VecDeque<TopicSnapshot>
}

TopicMetrics {
  id, age, membership_stability, centroid_stability,
  access_frequency, last_accessed,
  phase: Emerging|Stable|Declining|Merging
}

churn_rate: f32 // 0.0=stable, 1.0=completely new
```

**Dream Triggers:**
- entropy > 0.7 for 5+ min → MAY trigger
- churn > 0.5 AND entropy > 0.7 → MAY trigger

---

## Part 9: Implementation Roadmap

### Phase 0: North Star Removal
- Delete MCP tools, data structures, constitution rules
- Files: See Part 0

### Phase 1: Memory Capture System
- Hook description capture
- Claude response capture
- MD file watcher with chunking
- Files: `memory/sources.rs`, `memory/chunker.rs`, `memory/watcher.rs`

### Phase 2: 13-Space Embedding
- Embed all memories with all 13 embedders
- Store teleological arrays
- Files: `embedding/multi_space.rs`

### Phase 3: Similarity & Divergence Detection
- Per-space similarity thresholds
- Divergence detection against recent memories
- Files: `retrieval/similarity.rs`, `retrieval/divergence.rs`

### Phase 4: Multi-Space Clustering
- HDBSCAN + BIRCH per space
- Cross-space topic synthesis
- Files: `clustering/multi_space_manager.rs`, `clustering/hdbscan.rs`, `clustering/birch.rs`

### Phase 5: Injection Pipeline
- Priority ranking
- Token budgeting
- Context formatting
- Files: `injection/pipeline.rs`, `injection/formatter.rs`

### Phase 6: CLI & Hooks
- `context-graph-cli` binary
- Hook shell scripts
- Files: `cli/main.rs`, `cli/commands/*.rs`, `hooks/*.sh`

---

## Part 10: Validation Checklist

### North Star Removal
- [ ] All North Star MCP tools removed
- [ ] No `north_star_alignment`, `SELF_EGO_NODE`, IC references
- [ ] System boots and stores memories

### Memory Capture
- [ ] Hook descriptions captured and embedded
- [ ] Claude responses captured at session end
- [ ] MD file watcher chunks and embeds files
- [ ] 200-word chunks with 50-word overlap

### 13-Space Operations
- [ ] All memories have 13 embeddings
- [ ] Per-space similarity thresholds work
- [ ] Divergence detection finds low-similarity recent memories
- [ ] Multi-space relevance scoring
- [ ] **Temporal exclusion:** E2-E4 never count toward topic detection
- [ ] **Temporal exclusion:** E2-E4 never trigger divergence alerts
- [ ] **Relational/Structural weighting:** E8, E9, E11 count at 0.5× weight
- [ ] **Category weights:** weighted_agreement formula correctly implemented
- [ ] **Temporal enrichment:** Same-session badges appear on temporally-close memories

### Injection
- [ ] Divergence alerts surface when activity shifts
- [ ] Cluster matches inject related context
- [ ] Token budgets respected
- [ ] Recency weighting applied

### Clustering
- [ ] HDBSCAN forms clusters per space
- [ ] Cross-space topic synthesis (≥3 spaces)
- [ ] Progressive activation by tier

---

## Part 11: Summary

### Core Architecture
| Component | Description |
|-----------|-------------|
| Memory Sources | Hook descriptions, Claude responses, MD file chunks |
| Embedding | All 13 embedders for every memory |
| Similarity | Relevant if similar in ANY **semantic or supporting** space (excludes temporal) |
| Divergence | Alert if LOW similarity to recent in **semantic spaces only** |
| Clustering | HDBSCAN+BIRCH per space, topics from weighted agreement ≥2.5 |
| Injection | Clusters + divergence alerts + temporal context enrichment |

### Embedder Category Summary
| Category | Embedders | Topic Weight | Role |
|----------|-----------|--------------|------|
| Semantic | E1, E5, E6, E7, E10, E12, E13 (7) | 1.0× each | Primary topic triggers |
| Temporal | E2, E3, E4 (3) | 0.0× (excluded) | Metadata/context enrichment only |
| Relational | E8, E11 (2) | 0.5× each | Supporting evidence |
| Structural | E9 (1) | 0.5× | Supporting evidence |

**Max weighted agreement:** 7×1.0 + 2×0.5 + 1×0.5 = **8.5**

### Key Thresholds
| Metric | Value |
|--------|-------|
| Chunk size | 200 words |
| Chunk overlap | 50 words (25%) |
| High similarity (semantic) | > 0.70-0.80 (varies by space) |
| Low similarity (divergence) | < 0.25-0.35 (semantic spaces only) |
| Cluster min size | 3 |
| Topic threshold | weighted_agreement ≥ 2.5 |
| Exploration budget | 15% |
| Recency half-life | 45 days |

### Injection Logic
```
# Topic Detection (semantic + supporting only, temporal excluded)
IF weighted_agreement ≥ 2.5 → inject as topic cluster match
IF weighted_agreement ∈ [1.0, 2.5) → inject as related
IF LOW similarity to recent in SEMANTIC spaces → inject divergence alert

# Temporal Enrichment (never triggers, only enriches)
IF temporal similarity high → add "📅 From same session" badge
IF temporal similarity high → add "🕐 Around same time as X" context
```

### What Changed from Original PRD
| Original PRD | Implementation Plan Change | Rationale |
|--------------|---------------------------|-----------|
| All 13 spaces equal for topics | Temporal (E2-E4) excluded | Temporal proximity ≠ semantic relationship |
| Topics from ≥3 agreeing spaces | Topics from weighted_agreement ≥2.5 | Weighted by category |
| Divergence in any space | Divergence in semantic spaces only | Different time ≠ divergence |
| Temporal same as semantic | Temporal = metadata enrichment | Clusters in time ≠ topics |
