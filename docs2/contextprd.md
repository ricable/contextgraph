# Context Graph PRD v6 (Multi-Space Emergent Topic System)

**Platform**: Claude Code CLI ONLY | **Paradigm**: 13-embedding Multi-Space Clustering + Emergent Topic Discovery

**Architecture**: Hook-based memory capture + 13 parallel HDBSCAN/BIRCH clustering spaces + weighted topic synthesis + divergence detection

---

## 0. EXECUTIVE SUMMARY

Transform contextgraph into a multi-space emergent topic system:

1. Works from 0 memories (graceful degradation with progressive feature activation)
2. Discovers topics via 13 parallel HDBSCAN+BIRCH clustering spaces
3. Captures Claude's descriptions + responses + MD file content as memories
4. Uses frequency/recency importance scoring (EWMA + BM25 + Wilson)
5. Operates autonomously via Claude Code hooks — NO manual goal setting
6. Detects BOTH similarity clusters AND divergence from recent activity
7. Integrates via native Claude Code hooks for autonomous context injection

**Key Principle**: Temporal proximity does NOT equal semantic relationship. Temporal embedders (E2-E4) provide metadata enrichment but NEVER trigger topic detection or divergence alerts.

---

## 1. QUICK START

**Your Role**: You are a librarian, not an archivist. The system stores automatically via hooks. Your job is ensuring what's stored is findable, coherent, and useful.

**First Contact**: System initializes at SessionStart hook — portfolio summary + recent divergences injected automatically.

**Memory Sources**:
| Source | What Gets Captured | Trigger |
|--------|-------------------|---------|
| Hook Descriptions | Claude's description of what it's doing | Every hook event |
| Claude Responses | End-of-session answers, significant responses | SessionEnd, Stop hooks |
| MD File Watcher | Content from created/modified .md files | File system events |

**Progressive Feature Activation**:
| Tier | Memories | Features |
|------|----------|----------|
| 0 | 0 | Storage, basic retrieval |
| 1 | 1-2 | Pairwise similarity |
| 2 | 3-9 | Basic clustering |
| 3 | 10-29 | Multiple clusters, divergence detection |
| 4 | 30-99 | Reliable statistics |
| 5 | 100-499 | Sub-clustering, trend analysis |
| 6 | 500+ | Full personalization |

---

## 2. ARCHITECTURE

### 2.1 Memory Schema

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

### 2.2 MD File Watcher

**Watch**: All `.md` files in project directory (configurable)

**Chunking Strategy**:
- Chunk size: 200 words
- Overlap: 50 words (25%)
- Preserve sentence boundaries when possible
- Each chunk becomes a separate Memory with ChunkMetadata

### 2.3 Topic Portfolio

Topics emerge autonomously from clustering, not from manual goal setting.

```
TopicPortfolio {
  topics: Vec<Topic>,
  stability_tracker: TopicStabilityTracker,
  last_updated: Timestamp
}

Topic {
  id: TopicId,
  profile: [f32; 13],  // Strength in each embedding space
  members: Vec<MemoryId>,
  weighted_agreement: f32,  // 0.0 - 8.5
  confidence: f32,  // weighted_agreement / 8.5
  phase: Emerging | Stable | Declining | Merging
}
```

### 2.4 Topic Stability

```
TopicStabilityTracker {
  topics: HashMap<TopicId, TopicMetrics>,
  history: VecDeque<TopicSnapshot>
}

TopicMetrics {
  id: TopicId,
  age: Duration,
  membership_stability: f32,
  centroid_stability: f32,
  access_frequency: f32,
  last_accessed: Timestamp,
  phase: Emerging | Stable | Declining | Merging
}

churn_rate: f32  // 0.0=stable, 1.0=completely new topics
```

**Consolidation Triggers** (replaces IC-based dream triggers):
- entropy > 0.7 for 5+ min → MAY trigger consolidation
- churn > 0.5 AND entropy > 0.7 → MAY trigger consolidation

---

## 3. 13-EMBEDDER SYSTEM

### 3.1 Embedder Categories

The 13 embedders are divided into **4 categories** based on their semantic role:

| Category | Embedders | Topic Weight | Role |
|----------|-----------|--------------|------|
| **Semantic** | E1, E5, E6, E7, E10, E12, E13 (7) | 1.0x each | Primary topic triggers |
| **Temporal** | E2, E3, E4 (3) | 0.0x (EXCLUDED) | Metadata/context enrichment ONLY |
| **Relational** | E8, E11 (2) | 0.5x each | Supporting evidence |
| **Structural** | E9 (1) | 0.5x | Supporting evidence |

**Critical**: Temporal embedders (E2-E4) NEVER count toward topic detection. Working on 3 unrelated tasks in the same hour creates temporal clusters that are NOT topics.

### 3.2 Per-Embedder Specifications

| ID | Name | Dim | Purpose Vector | Category | Distance | Topic Weight |
|----|------|-----|----------------|----------|----------|--------------|
| E1 | Semantic (Matryoshka) | 1024D | V_meaning | Semantic | Cosine | 1.0 |
| E2 | Temporal-Recent | 512D | V_freshness | Temporal | Cosine | 0.0 |
| E3 | Temporal-Periodic | 512D | V_periodicity | Temporal | Cosine | 0.0 |
| E4 | Temporal-Positional | 512D | V_ordering | Temporal | Cosine | 0.0 |
| E5 | Causal (asymmetric) | 768D | V_causality | Semantic | Asymmetric KNN | 1.0 |
| E6 | Sparse | ~30K (5% active) | V_selectivity | Semantic | Jaccard | 1.0 |
| E7 | Code (AST) | 1536D | V_correctness | Semantic | Cosine | 1.0 |
| E8 | Graph/GNN | 384D | V_connectivity | Relational | TransE | 0.5 |
| E9 | HDC | 10K→1024D | V_robustness | Structural | Hamming | 0.5 |
| E10 | Multimodal | 768D | V_multimodality | Semantic | Cosine | 1.0 |
| E11 | Entity/TransE | 384D | V_factuality | Relational | TransE | 0.5 |
| E12 | Late-Interaction | 128D/token | V_precision | Semantic | MaxSim | 1.0 |
| E13 | SPLADE | ~30K sparse | V_keyword | Semantic | Jaccard | 1.0 |

### 3.3 ΔS Computation Methods

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

---

## 4. SIMILARITY & DIVERGENCE DETECTION

### 4.1 Similarity Detection (What to Inject)

A memory is **relevant** if it shows high similarity in **ANY semantic or supporting** embedding space.

**IMPORTANT**: Temporal embedders (E2-E4) are EXCLUDED from relevance detection.

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

### 4.2 Per-Space Thresholds

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

### 4.3 Divergence Detection (Contradiction Alert)

Compare current activity to **recent** memories (last 2 hours / current session).

**CRITICAL**: Only **Semantic embedders** (E1, E5, E6, E7, E10, E12, E13) participate in divergence detection.

- Temporal embedders EXCLUDED: Working at different times is not divergence
- Relational embedders EXCLUDED: Different entities/relationships is not inherently divergent
- Structural embedders EXCLUDED: Different structure is not semantic divergence

```
divergent(current, recent_memory) = ANY(
  similarity(current.Ei, recent_memory.Ei) < low_threshold_Ei
  for i in SEMANTIC_EMBEDDERS  // Only E1, E5, E6, E7, E10, E12, E13
)
```

**Injection format**:
```
DIVERGENCE DETECTED
Recent activity in [semantic_space]: "[memory content summary]"
Current appears different - similarity: 0.23
```

### 4.4 Temporal Context Enrichment

While temporal embedders don't trigger similarity/divergence, they provide valuable context:

```
temporal_context(memory) = {
  same_session: similarity(current.E2, memory.E2) > 0.8,
  same_day: similarity(current.E3, memory.E3) > 0.7,
  same_period: similarity(current.E4, memory.E4) > 0.6
}
```

**Injection uses**:
- "From same session" badge on related memories
- "You also worked on X around this time" enrichment
- Session-based grouping in context window

### 4.5 Multi-Space Relevance Score

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

---

## 5. MULTI-SPACE CLUSTERING & TOPIC SYNTHESIS

### 5.1 Cluster Formation

- HDBSCAN for batch clustering per embedding space
- BIRCH CF-trees for online updates
- min_cluster_size = 3
- silhouette_score > 0.3

### 5.2 Cross-Space Topic Synthesis

Topics are formed when memories cluster together in **semantic** embedding spaces. Temporal clustering is explicitly excluded.

**Topic Formation Rule**:
```
weighted_agreement = Σ (topic_weight_i × is_clustered_i)

Where:
  SEMANTIC embedders (E1, E5, E6, E7, E10, E12, E13): topic_weight = 1.0
  TEMPORAL embedders (E2, E3, E4): topic_weight = 0.0  // NEVER count
  RELATIONAL embedders (E8, E11): topic_weight = 0.5
  STRUCTURAL embedder (E9): topic_weight = 0.5

Max possible weighted_agreement = 7×1.0 + 2×0.5 + 1×0.5 = 8.5
```

**Topic Detection Threshold**:
```
is_topic = weighted_agreement >= 2.5

// Examples:
// 3 semantic spaces agreeing = 3.0 -> TOPIC
// 2 semantic + 1 relational = 2.5 -> TOPIC
// 2 semantic spaces only = 2.0 -> NOT TOPIC
// 5 temporal spaces = 0.0 -> NOT TOPIC (temporal excluded)
// 1 semantic + 3 relational = 1.0 + 1.5 = 2.5 -> TOPIC
```

**Topic Confidence**:
```
topic_confidence = weighted_agreement / 8.5  // Normalized to [0, 1]
```

**Topic Profile (13D)**:
```
topic_profile = [strength_E1, strength_E2, ..., strength_E13]
// All 13 dimensions stored, but only semantic/relational/structural used for detection
// Temporal dimensions stored for context enrichment only
```

### 5.3 Cluster-Based Retrieval

1. Identify which clusters the current query belongs to (per space)
2. Retrieve other members of those clusters
3. Rank by multi-space overlap (member of same cluster in more spaces = higher rank)

---

## 6. INJECTION STRATEGY

### 6.1 What Gets Injected

| Priority | Type | Condition | Token Budget |
|----------|------|-----------|--------------|
| 1 | Divergence Alerts | Low similarity to recent in **semantic spaces** | ~200 |
| 2 | High-Relevance Topics | weighted_agreement >= 2.5 (topic clusters) | ~400 |
| 3 | Related Memories | weighted_agreement in [1.0, 2.5) | ~300 |
| 4 | Recent Context | Last session summary | ~200 |
| 5 | Temporal Enrichment | Same-session/same-period badges (metadata) | ~50 |

**Note**: Temporal embedders (E2-E4) NEVER appear in Priority 1-3. They only contribute to Priority 5 as contextual metadata badges.

### 6.2 Injection Priority Algorithm

```
priority = relevance_score × recency_factor × weighted_diversity_bonus

recency_factor:
  < 1 hour = 1.3x
  < 1 day = 1.2x
  < 7 days = 1.1x
  < 30 days = 1.0x
  > 90 days = 0.8x

weighted_diversity_bonus (based on weighted_agreement, not raw space count):
  weighted_agreement >= 5.0 = 1.5x  (strong topic signal)
  weighted_agreement in [2.5, 5.0) = 1.2x  (topic threshold met)
  weighted_agreement in [1.0, 2.5) = 1.0x  (related)
  weighted_agreement < 1.0 = 0.8x  (weak)

# NOTE: Temporal spaces (E2-E4) do NOT contribute to weighted_agreement
# Temporal proximity is captured separately for badge enrichment
```

### 6.3 Context Window Format

```
## Relevant Context

### Recent Related Work
[High-relevance memories from clusters]

### Potentially Related
[Single-space matches]

### Note: Activity Shift Detected
[Divergence alerts if any]
```

---

## 7. IMPORTANCE SCORING

```
Importance = Frequency_Score × Recency_Weight

Frequency_Score = BM25_saturated(log(1+access_count))
  BM25: tf_saturated = (freq × (k1+1)) / (freq + k1), k1=1.2

Recency_Weight = e^(-λ × days)
  λ = ln(2) / 45 (45-day half-life)

Burst_Damping = min(freq, μ + 1.5σ)
```

---

## 8. BIAS MITIGATION

### 8.1 Thompson Sampling (15% exploration)
- Beta(α, β) per cluster; new clusters start Beta(1,1)
- Sample θ_i ~ Beta(α_i, β_i), select highest
- Update: accessed → α+=1, not accessed → β+=1

### 8.2 MMR Diversity (λ=0.7)
- MMR = λ × relevance - (1-λ) × max_similarity_to_selected

### 8.3 Inverse Propensity Scoring
- weight = 1 / propensity
- propensity = exposure_count / total_queries (floor 0.1)

---

## 9. NATIVE CLAUDE CODE HOOK INTEGRATION

**CRITICAL ARCHITECTURE DECISION**: This system uses **NATIVE Claude Code hooks** configured through `.claude/settings.json` — NOT internal/built-in hooks or custom middleware.

### 9.1 Hook Lifecycle

| Hook | Captures | Injects | CLI Command |
|------|----------|---------|-------------|
| SessionStart | - | Portfolio summary + recent divergences | `session start` |
| UserPromptSubmit | Prompt text | Similar memories + divergence alerts | `inject-context` |
| PreToolUse | Tool description | Brief relevant context | `inject-brief` |
| PostToolUse | Tool description + output summary | - | `capture-memory` |
| Stop | Claude's response summary | - | `capture-response` |
| SessionEnd | Session summary | - | `session end` |

### 9.2 Hook Configuration (`.claude/settings.json`)

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

### 9.3 CLI Commands

```
context-graph-cli
├── session
│   ├── start           # Initialize session, inject portfolio summary
│   └── end             # Persist session summary, trigger consolidation if needed
├── memory
│   ├── capture         # Store memory with embeddings
│   └── inject-context  # Inject relevant context for query
├── topic
│   ├── portfolio       # Get current topic portfolio
│   ├── stability       # Get topic stability metrics
│   └── detect          # Run topic detection
└── divergence
    └── check           # Check for divergence from recent activity
```

### 9.4 Shell Script Executors

**SessionStart** (`hooks/session-start.sh`):
```bash
#!/bin/bash
context-graph-cli session start
context-graph-cli topic portfolio --format brief
```

**UserPromptSubmit** (`hooks/user-prompt-submit.sh`):
```bash
#!/bin/bash
context-graph-cli memory inject-context "$PROMPT"
context-graph-cli divergence check --recent 2h
```

**PostToolUse** (`hooks/post-tool-use.sh`):
```bash
#!/bin/bash
context-graph-cli memory capture --source hook_description --content "$DESCRIPTION"
```

**SessionEnd** (`hooks/session-end.sh`):
```bash
#!/bin/bash
context-graph-cli memory capture --source session_summary
context-graph-cli session end
```

---

## 10. MCP TOOLS

### 10.1 Core Tools

| Tool | WHEN | WHY | Key Params |
|------|------|-----|------------|
| `inject_context` | Starting task | Primary retrieval+distillation | query, max_tokens, verbosity |
| `search_graph` | Need specific nodes | Raw vector search | query, top_k, filters |
| `store_memory` | User shares novel info | Requires rationale | content, importance, rationale, link_to |
| `get_topic_portfolio` | Understand current topics | View emergent topics | format |
| `get_topic_stability` | Monitor topic health | Track churn | → churn_rate, entropy |
| `trigger_consolidation` | churn>0.5 + entropy>0.7 | Consolidate topics | blocking |
| `get_memetic_status` | Start, periodically | Health + curation_tasks | → entropy, coherence, tasks |

### 10.2 Topic Tools

| Tool | Purpose | Returns |
|------|---------|---------|
| `get_topic_portfolio` | All current topics | {topics[], stability, churn_rate} |
| `get_topic_stability` | Stability metrics | {churn_rate, entropy, phases} |
| `detect_topics` | Run topic detection | {new_topics[], merged_topics[]} |
| `get_divergence_alerts` | Recent divergences | {alerts[], severity} |

### 10.3 Curation Tools

| Tool | Purpose | Key Params |
|------|---------|------------|
| `merge_concepts` | Merge duplicate memories | source_node_ids[], merge_strategy |
| `forget_concept` | Soft delete memory | node_id, soft_delete=true |
| `boost_importance` | Increase memory importance | node_id, delta |

---

## 11. DATA MODEL

### 11.1 KnowledgeNode

```
KnowledgeNode {
  id: UUID,
  content: String (max 65536),
  fingerprint: {
    embeddings: [E1..E13],  // All 13 embeddings
    dominant_embedder: EmbedderId,
    coherence_score: f32
  },
  timestamps: {created_at, updated_at, last_accessed},
  importance: f32 (0-1),
  access_count: u32,
  cluster_assignments: HashMap<EmbedderId, ClusterId>,
  source: HookDescription | ClaudeResponse | MDFileChunk,
  chunk_metadata: Option<ChunkMetadata>
}
```

### 11.2 GraphEdge

```
GraphEdge {
  source: NodeId,
  target: NodeId,
  edge_type: Semantic | Temporal | Causal | Hierarchical | Relational,
  weight: f32,
  confidence: f32
}
```

### 11.3 TeleologicalFingerprint

```
TeleologicalFingerprint {
  semantic_fingerprint: [E1..E13],  // Raw embeddings
  topic_profile: [f32; 13],  // Per-space topic membership strength
  dominant_embedder: EmbedderId,
  coherence_score: f32
}
```

---

## 12. PERFORMANCE BUDGETS

| Operation | Target |
|-----------|--------|
| All 13 embed | <35ms |
| Batch 64×13 | <120ms |
| Per-space HNSW | <2ms |
| Topic profile search (13D) | <1ms |
| inject_context P95 | <40ms |
| Any tool P99 | <60ms |
| Cluster update (BIRCH) | <5ms |
| Topic detection | <50ms |

**Quality Gates**:
- Unit test coverage >= 90%
- Integration test coverage >= 80%
- Info loss from chunking < 15%

---

## 13. KEY THRESHOLDS SUMMARY

| Metric | Value |
|--------|-------|
| Chunk size | 200 words |
| Chunk overlap | 50 words (25%) |
| High similarity (semantic) | > 0.70-0.80 (varies by space) |
| Low similarity (divergence) | < 0.25-0.35 (semantic spaces only) |
| Cluster min size | 3 |
| Topic threshold | weighted_agreement >= 2.5 |
| Max weighted agreement | 8.5 (7 semantic × 1.0 + 2 relational × 0.5 + 1 structural × 0.5) |
| Exploration budget | 15% |
| Recency half-life | 45 days |
| Consolidation trigger | entropy > 0.7 AND churn > 0.5 |

---

## 14. DESIGN PRINCIPLES

This architecture is built on the following core principles:

| Principle | Implementation |
|-----------|----------------|
| Emergent Topics | Topics emerge from clustering, not manual goal setting |
| Topic Stability | Churn and entropy metrics track stability over time |
| Consolidation Triggers | entropy > 0.7 AND churn > 0.5 triggers dream consolidation |
| Coherence Scoring | Topic-based coherence from weighted agreement |
| Per-Space Clustering | Independent clustering in each embedding space |
| Category Weights | Semantic (1.0), Relational/Structural (0.5), Temporal (0.0) |

**Key Design Decisions**:
1. Topics emerge from data clustering, supporting multi-topic work
2. Temporal proximity is metadata only, not semantic similarity
3. Weighted agreement (threshold >= 2.5) defines topic membership
4. Explicit separation of temporal metadata from semantic clustering

---

## 15. REFERENCES

**Internal**: Memory Schema (2.1), Topic Portfolio (2.3), Embedder Categories (3.1), Similarity Detection (4.1), Divergence Detection (4.3), Topic Synthesis (5.2), Hook Integration (9)