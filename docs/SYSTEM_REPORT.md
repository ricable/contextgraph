# ContextGraph System Report
## Version 6.5.0 | 13-Perspectives Collaboration | E1 Foundation

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [The 13-Embedder System](#the-13-embedder-system)
4. [Code Embedding Pipeline](#code-embedding-pipeline)
5. [Storage Architecture](#storage-architecture)
6. [Topic Detection System](#topic-detection-system)
7. [Retrieval Pipeline](#retrieval-pipeline)
8. [MCP Tools Reference](#mcp-tools-reference)
9. [Configuration](#configuration)
10. [Architectural Rules](#architectural-rules)
11. [Anti-Patterns](#anti-patterns)

---

## Executive Summary

**ContextGraph** is a production-grade 13-embedder semantic memory retrieval system designed for Claude Code (Anthropic's CLI). It implements a "13-perspectives collaboration" model where each of 13 different embedding models captures unique semantic dimensions that others miss.

### Core Philosophy

```
13 embedders = 13 unique perspectives on every memory
Each finds what OTHERS MISS. Combined = superior answers.

Example: Query "What databases work with Rust?"
- E1 finds: "database" or "Rust" semantically
- E11 finds: "Diesel" (knows Diesel IS a database ORM - E1 missed this)
- E7 finds: code using sqlx, diesel crates
- Combined: Better answer than any single embedder
```

### Key Capabilities

- **Multi-perspective Understanding**: 13 independent embedding models each capturing different semantic aspects
- **Asymmetric Similarity**: Directional relationships (cause→effect, source→target) for E5 and E8
- **Code-aware Embeddings**: Tree-sitter AST parsing with Qodo-Embed-1-1.5B (E7)
- **5-stage Retrieval Pipeline**: Optimized for <60ms @ 1M memories
- **Topic Clustering**: Weighted agreement scoring across embedding spaces
- **Native Claude Code Integration**: MCP hooks for session management
- **GPU Acceleration**: Candle/CUDA for RTX 5090 Blackwell architecture
- **RocksDB Persistence**: 39 column families with fail-fast error handling

---

## System Architecture

### Project Structure

Monorepo with 11 crates under `/crates/`:

| Crate | Purpose |
|-------|---------|
| `context-graph-core` | Core domain types, traits, teleological fusion |
| `context-graph-storage` | RocksDB persistence layer |
| `context-graph-embeddings` | Embedding model provider (Candle/CUDA) |
| `context-graph-mcp` | Model Context Protocol (MCP) server |
| `context-graph-cuda` | GPU acceleration utilities |
| `context-graph-graph` | Graph linking and K-NN construction |
| `context-graph-causal-agent` | Causal reasoning agent |
| `context-graph-graph-agent` | Graph relationship activator |
| `context-graph-benchmark` | Performance benchmarking suite |
| `context-graph-cli` | Command-line interface |
| `context-graph-test-utils` | Testing utilities |

### High-Level Flow

```
User Query
    ↓
[MCP Server] (stdio/TCP)
    ↓
[Tool Handler] (50 tools)
    ↓
[Retrieval Pipeline] (5 stages)
    ├── S1: E13 SPLADE Sparse Pre-filter
    ├── S2: E1 Matryoshka 128D Fast ANN
    ├── S3: Multi-Space HNSW Dense Search
    ├── S4: Teleological Alignment Filter
    └── S5: E12 ColBERT Late Interaction Rerank
    ↓
[Results] → Claude Code
```

---

## The 13-Embedder System

Each embedder captures a different semantic lens. The system maintains **full information preservation** (~46KB per memory) with no lossy compression.

### Embedder Summary Table

| # | Name | Model | Dims | Category | Topic Weight | Purpose |
|---|------|-------|------|----------|--------------|---------|
| **E1** | Semantic | e5-large-v2 | 1024 | SEMANTIC | 1.0 | Foundation: semantic similarity |
| **E2** | TemporalRecent | Exponential Decay | 512 | TEMPORAL | 0.0 | Recency weighting |
| **E3** | TemporalPeriodic | Fourier | 512 | TEMPORAL | 0.0 | Cyclical patterns (time-of-day) |
| **E4** | TemporalPositional | Sinusoidal PE | 512 | TEMPORAL | 0.0 | Sequence ordering (before/after) |
| **E5** | Causal | Longformer SCM | 768 (dual) | SEMANTIC | 1.0 | Causal chains (why/cause-effect) **ASYMMETRIC** |
| **E6** | Sparse | SPLADE | 30,522 vocab | SEMANTIC | 1.0 | Exact keyword matches |
| **E7** | Code | Qodo-Embed-1.5B | 1536 | SEMANTIC | 1.0 | Code patterns, function signatures |
| **E8** | Graph | e5-large-v2 | 1024 (dual) | RELATIONAL | 0.5 | Graph connectivity **ASYMMETRIC** |
| **E9** | HDC | Hyperdimensional | 1024 | STRUCTURAL | 0.5 | Noise-robust structure |
| **E10** | Multimodal | CLIP | 768 (dual) | SEMANTIC | 1.0 | Same-goal work (different words) |
| **E11** | Entity | KEPLER | 768 | RELATIONAL | 0.5 | Named entities, knowledge graphs |
| **E12** | LateInteraction | ColBERT | 128/token | SEMANTIC | 1.0 | Per-token embeddings (MaxSim) |
| **E13** | KeywordSplade | SPLADE v3 | 30,522 vocab | SEMANTIC | 1.0 | Term expansions (Stage 1 recall) |

### Detailed Embedder Specifications

#### E1: Semantic (Foundation)
- **Model**: `intfloat/e5-large-v2`
- **Dimensions**: 1024D dense
- **Quantization**: PQ8 (32 subvectors, 8-bit codes)
- **Purpose**: General semantic understanding - the foundation all retrieval starts with
- **Key Rule**: ARCH-12 - E1 is foundation, all retrieval starts with E1

#### E2: Temporal Recent
- **Model**: Custom exponential decay function
- **Dimensions**: 512D dense
- **Quantization**: Float8
- **Purpose**: Captures recency of memories with exponential decay weighting
- **Key Rule**: ARCH-25 - Temporal boosts POST-retrieval only, NOT in similarity fusion

#### E3: Temporal Periodic
- **Model**: Custom Fourier encoding
- **Dimensions**: 512D dense
- **Quantization**: Float8
- **Purpose**: Detects periodic/cyclical patterns (daily, weekly, seasonal)

#### E4: Temporal Positional
- **Model**: Custom sinusoidal positional encoding
- **Dimensions**: 512D dense
- **Quantization**: Float8
- **Purpose**: Encodes absolute position/order in conversation sequences

#### E5: Causal (Asymmetric)
- **Model**: Longformer SCM (Structured Causal Model)
- **Dimensions**: 768D dense (dual vectors)
- **Vectors**: `e5_causal_as_cause` and `e5_causal_as_effect`
- **Quantization**: PQ8 (24 subvectors, 8-bit)
- **Purpose**: Detects causal chains with directional awareness
- **Key Rule**: ARCH-18 - E5 uses asymmetric similarity (cause→effect ≠ effect→cause)
- **Direction Modifiers**: cause→effect 1.2x boost, effect→cause 0.8x dampening

#### E6: Sparse Lexical
- **Model**: `naver/splade-cocondenser-ensembledistil`
- **Dimensions**: Sparse (30,522 vocabulary size, ~1500 active per document)
- **Quantization**: Inverted index
- **Purpose**: Exact keyword/lexical matching via term expansion

#### E7: Code
- **Model**: `Qodo/Qodo-Embed-1-1.5B`
- **Dimensions**: 1536D dense
- **Quantization**: PQ8 (48 subvectors, 8-bit)
- **VRAM**: ~3GB (separate from main 13-embedder stack)
- **Purpose**: Detects code patterns, function signatures, syntax
- **Key Rule**: ARCH-CODE-02 - E7 is PRIMARY embedder for code

#### E8: Graph (Asymmetric)
- **Model**: `intfloat/e5-large-v2` (shares E1 model for VRAM efficiency)
- **Dimensions**: 1024D dense (dual vectors)
- **Vectors**: `e8_graph_as_source` and `e8_graph_as_target`
- **Quantization**: Float8
- **Purpose**: Detects graph structure (X imports Y, A references B)
- **Direction Modifiers**: source→target 1.2x, target→source 0.8x

#### E9: HDC (Hyperdimensional Computing)
- **Model**: Custom HDC with character trigrams
- **Dimensions**: 1024D dense (projected from 10,000-bit native)
- **Quantization**: PQ8 (32 subvectors)
- **Purpose**: Noise-robust structure via hyperdimensional encoding

#### E10: Multimodal (Multiplicative Boost)
- **Model**: CLIP (Contrastive Language-Image Pre-training)
- **Dimensions**: 768D dense (dual vectors)
- **Vectors**: `e10_multimodal_as_intent` and `e10_multimodal_as_context`
- **Quantization**: PQ8 (24 subvectors, 8-bit)
- **Purpose**: Finds same-goal work using different words
- **Key Rule**: ARCH-28 - E10 uses multiplicative boost: `E1 * (1 + boost)`
- **Boost Calculation**:
  - Strong E1 (>0.8): 5% boost
  - Medium E1 (0.5-0.8): 10% boost
  - Weak E1 (<0.5): 15% boost
  - Multiplier clamped to [0.8, 1.2]

#### E11: Entity (KEPLER)
- **Model**: `facebook/kepler` (RoBERTa-base + TransE)
- **Dimensions**: 768D dense
- **Quantization**: Float8
- **Purpose**: Entity knowledge - understands that Diesel IS a database ORM
- **TransE Operations**: `r̂ = t - h` for relationship inference

#### E12: Late Interaction (ColBERT)
- **Model**: ColBERT-style late-interaction model
- **Dimensions**: 128D per token (variable-length sequence, max 512 tokens)
- **Quantization**: Float8 per token
- **Purpose**: Token-level MaxSim scoring for high-precision reranking
- **Key Rule**: AP-74 - E12 ColBERT: reranking ONLY, not initial retrieval

#### E13: Keyword SPLADE
- **Model**: SPLADE v3
- **Dimensions**: Sparse (30,522 vocabulary size)
- **Quantization**: Inverted index
- **Purpose**: Term expansion for Stage 1 recall (fast→quick)
- **Key Rule**: AP-75 - E13 SPLADE: Stage 1 recall ONLY, not final ranking

### TeleologicalArray Structure

```rust
pub struct SemanticFingerprint {
    // E1-E4 (dense)
    pub e1_semantic: Vec<f32>,              // 1024D
    pub e2_temporal_recent: Vec<f32>,       // 512D
    pub e3_temporal_periodic: Vec<f32>,     // 512D
    pub e4_temporal_positional: Vec<f32>,   // 512D

    // E5 (dual asymmetric)
    pub e5_causal_as_cause: Vec<f32>,       // 768D
    pub e5_causal_as_effect: Vec<f32>,      // 768D

    // E6 (sparse)
    pub e6_sparse: SparseVector,            // ~1500 active / 30522 vocab

    // E7 (dense code)
    pub e7_code: Vec<f32>,                  // 1536D

    // E8 (dual asymmetric)
    pub e8_graph_as_source: Vec<f32>,       // 1024D
    pub e8_graph_as_target: Vec<f32>,       // 1024D

    // E9-E11 (dense)
    pub e9_hdc: Vec<f32>,                   // 1024D (projected)
    pub e10_multimodal_as_intent: Vec<f32>, // 768D
    pub e10_multimodal_as_context: Vec<f32>, // 768D
    pub e11_entity: Vec<f32>,               // 768D

    // E12 (token-level)
    pub e12_late_interaction: Vec<Vec<f32>>, // 128D per token

    // E13 (sparse)
    pub e13_splade: SparseVector,           // 30522 vocab
}

// Type alias
pub type TeleologicalArray = SemanticFingerprint;
```

**Storage**: ~46KB per fingerprint (NO compression, NO fusion - full information preservation)

---

## Code Embedding Pipeline

Code entities are stored **SEPARATELY** from the 13-embedder teleological system with dedicated storage.

### Pipeline Flow

```
Source Code Files
      ↓ [tree-sitter parser]
   AST Parse
      ↓ [cAST methodology: respect syntactic boundaries]
   Code Chunks (functions, structs, traits as atomic units)
      ↓ [AST-aware chunking]
   Target: ~500 non-whitespace chars (Qodo recommendation)
   Min: 100 chars (avoid tiny fragments)
   Max: 1000 chars (prevent semantic dilution)
      ↓
   CodeEntity (with rich metadata)
      ↓ [Full 13-embedder treatment]
   SemanticFingerprint (all 13 embeddings)
      ↓
   CodeStore (separate from TeleologicalStore)
```

### CodeEntity Structure

```rust
struct CodeEntity {
    id: Uuid,
    entity_type: CodeEntityType,  // Function/Struct/Trait/Method/Impl/Enum/Const/Static/Macro/Module
    name: String,
    code: String,
    language: String,             // "rust" for now
    file_path: PathBuf,
    line_start: u32,
    line_end: u32,
    module_path: Option<String>,
    signature: Option<String>,
    parent_type: Option<String>,  // For methods
    visibility: Visibility,
    doc_comment: Option<String>,
    attributes: Vec<String>,
    content_hash: String,         // SHA256 for change detection
}
```

### Code Architecture Rules

| Rule | Description |
|------|-------------|
| ARCH-CODE-01 | Code entities stored SEPARATELY from teleological memories |
| ARCH-CODE-02 | E7 is PRIMARY embedder for code (Qodo-Embed 1536D) |
| ARCH-CODE-03 | AST chunking preserves syntactic boundaries |
| ARCH-CODE-04 | Tree-sitter for parsing (NOT regex) |

### Code Anti-Patterns

| Rule | Forbidden Action |
|------|------------------|
| AP-CODE-01 | NEVER chunk code by word count - use AST boundaries |
| AP-CODE-02 | NEVER store code in teleological memory graph |
| AP-CODE-03 | NEVER use E1 alone for code search - E7 is primary |

---

## Storage Architecture

### RocksDB Column Families (39 total)

**Base (11)**:
- nodes, edges, embeddings, metadata, temporal, tags, sources, system, embedder_edges, typed_edges, typed_edges_by_type

**Teleological (15)**:
- fingerprints, topic_profiles, e13_splade_inverted, e1_matryoshka_128, synergy_matrix, teleological_profiles, teleological_vectors, + backups

**Quantized Embeddings (13)**:
- emb_0 through emb_12 (post-retrieval fingerprint compression)

**Code (5)**:
- code_entities, code_e7_embeddings, code_file_index, code_name_index, code_signature_index

### Memory Structure

```rust
struct Memory {
    id: Uuid,
    content: String,
    teleological_array: SemanticFingerprint,  // ALL 13 embeddings
    source: MemorySource,
    session_id: String,
    created_at: DateTime<Utc>,
    importance: f32,
    // ... metadata
}
```

### Memory Sources

| Source | Description |
|--------|-------------|
| HookDescription | Claude's description of tool use |
| ClaudeResponse | Session summaries, significant responses |
| MDFileChunk | Markdown file chunks (200 words, 50 overlap) |
| CodeEntity | AST-parsed code entities |
| CausalExplanation | LLM-generated causal insights |

---

## Topic Detection System

### TopicProfile (13D vector)

Each topic has a 13D alignment profile showing strength in each embedding space:

```rust
struct TopicProfile {
    strengths: [f32; 13],  // 0.0 to 1.0, one per embedder
}
```

### Weighted Agreement Formula

```
weighted_agreement = Sum(topic_weight_i × is_clustered_i)

Category Weights:
- SEMANTIC (E1, E5, E6, E7, E10, E12, E13):  1.0
- RELATIONAL (E8, E11):                       0.5
- STRUCTURAL (E9):                            0.5
- TEMPORAL (E2, E3, E4):                      0.0 (NEVER counts!)

Maximum Possible: 7×1.0 + 2×0.5 + 1×0.5 = 8.5

Topic Detection Threshold: weighted_agreement >= 2.5
Topic Validation: silhouette_score >= 0.3
```

### Topic Lifecycle

| Phase | Criteria |
|-------|----------|
| Emerging | <1 hour old, high churn (>0.3) |
| Stable | 24+ hours, low churn (<0.1) |
| Declining | High churn (>0.5), members leaving |
| Merging | Being absorbed into another topic |

### Divergence Detection

Uses **SEMANTIC EMBEDDERS ONLY** (E1, E5, E6, E7, E10, E12, E13):
- Lookback: 2 hours (RECENT_LOOKBACK_SECS)
- Max recent: 50 memories (MAX_RECENT_MEMORIES)
- Triggers context re-injection if significant topic drift detected
- Temporal proximity NEVER triggers divergence (AP-63)

---

## Retrieval Pipeline

**Target Performance**: <60ms @ 1M memories

### Stage 1: SPLADE Sparse Pre-filter (<5ms)
- **Embedder**: E13 (SPLADE v3)
- **Method**: Sparse inverted index with 30K vocabulary
- **Output**: ~10K candidates
- **Scoring**: BM25+SPLADE hybrid

### Stage 2: Matryoshka 128D Fast ANN (<10ms)
- **Embedder**: E1 first 128 dimensions
- **Method**: HNSW approximate nearest neighbors
- **Output**: ~1K candidates
- **Scaling**: Matryoshka 1024D → 128D

### Stage 3: Multi-Space HNSW Dense Search (<20ms)
- **Embedders**: 10 HNSW indexes (E1-E5, E7-E11)
- **Method**: Parallel search with RRF fusion
- **Output**: ~100 candidates
- **Fusion**: Weighted RRF (NOT weighted sum per ARCH-21)

### Stage 4: Teleological Alignment Filter (<10ms)
- **Method**: Topic-based filtering (weighted_agreement >= 2.5)
- **Validation**: Semantic coherence check
- **Output**: ~50 candidates

### Stage 5: Late Interaction Reranking (<15ms)
- **Embedder**: E12 ColBERT
- **Method**: Token-level MaxSim scoring
- **Output**: Final ranked results (10-20)

### Index Architecture

| Index Type | Count | Embedders | Stage | Purpose |
|------------|-------|-----------|-------|---------|
| HNSW | 10 | E1-E5, E7-E11 | S3 | Dense similarity |
| HNSW | 1 | E1[..128] | S2 | Fast 128D pre-filter |
| Inverted | 1 | E13 SPLADE | S1 | Sparse pre-filter |
| MaxSim | 1 | E12 ColBERT | S5 | Token-level reranking |

### When to Use Which Enhancer

| Query Type | Primary Enhancer |
|------------|------------------|
| Causal queries (why, what caused) | E5 |
| Code queries (implementations, functions) | E7 |
| Intent queries (same goal, similar purpose) | E10 |
| Entity queries (specific named things) | E11 |
| Keyword queries (exact terms, jargon) | E6, E13 |

---

## MCP Tools Reference

### Overview

The system provides **50 MCP tools** organized into 17 categories.

### Category Summary

| Category | Count | Tools |
|----------|-------|-------|
| Core | 4 | store_memory, get_memetic_status, search_graph, trigger_consolidation |
| Topic | 4 | get_topic_portfolio, get_topic_stability, detect_topics, get_divergence_alerts |
| Curation | 2 | forget_concept, boost_importance |
| Merge | 1 | merge_concepts |
| Sequence | 4 | get_conversation_context, get_session_timeline, traverse_memory_chain, compare_session_states |
| Causal | 4 | search_causal_relationships, search_causes, search_effects, get_causal_chain |
| Causal Discovery | 2 | trigger_causal_discovery, get_causal_discovery_status |
| Keyword | 1 | search_by_keywords |
| Code | 1 | search_code |
| Graph | 4 | search_connections, get_graph_path, discover_graph_relationships, validate_graph_link |
| Robustness | 1 | search_robust |
| Intent | 1 | search_by_intent |
| Entity | 6 | extract_entities, search_by_entities, infer_relationship, find_related_entities, validate_knowledge, get_entity_graph |
| Embedder-First | 4 | search_by_embedder, get_embedder_clusters, compare_embedder_views, list_embedder_indexes |
| Temporal | 2 | search_recent, search_periodic |
| Graph Linking | 4 | get_memory_neighbors, get_typed_edges, traverse_graph, get_unified_neighbors |
| File Watcher | 4 | list_watched_files, get_file_watcher_stats, delete_file_content, reconcile_files |
| Maintenance | 1 | repair_causal_relationships |

---

### Core Tools

#### store_memory
Store a memory node directly in the knowledge graph.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| content | string | Yes | - | The content to store |
| rationale | string | No | - | Why this context is relevant (1-1024 chars) |
| importance | number | No | 0.5 | Importance score [0.0-1.0] |
| modality | enum | No | text | Content type (text/code/image/audio/structured/mixed) |
| tags | array | No | [] | Categorization tags |
| sessionId | string | No | - | Session ID for scoped storage |

**Returns:** Confirmation with stored memory ID

---

#### get_memetic_status
Get current system state including fingerprint count, embedder count, storage backend/size, and layer status.

**Parameters:** None

**Returns:** System metrics and status

---

#### search_graph
Multi-space semantic search with automatic asymmetric E5 causal and E10 intent enhancements.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| query | string | Yes | - | Search query |
| topK | integer | No | 10 | Results to return (1-100) |
| minSimilarity | number | No | 0 | Similarity threshold [0-1] |
| modality | enum | No | - | Filter by content type |
| includeContent | boolean | No | false | Include full content |
| strategy | enum | No | e1_only | Search strategy (e1_only/multi_space/pipeline) |
| weightProfile | enum | No | - | Weight profile for multi-space fusion |
| enableRerank | boolean | No | false | Enable ColBERT E12 reranking |
| enableAsymmetricE5 | boolean | No | true | Asymmetric causal reranking |
| causalDirection | enum | No | auto | Causal direction (auto/cause/effect/none) |
| enableQueryExpansion | boolean | No | - | Query expansion for causal |
| intentMode | enum | No | none | E10 mode (none/seeking_intent/seeking_context/auto) |
| intentBlend | number | No | 0.3 | E10 blend weight [0-1] |
| enableIntentGate | boolean | No | - | Intent filtering in pipeline |
| intentGateThreshold | number | No | 0.3 | Intent filter threshold [0-1] |
| temporalWeight | number | No | 0 | Temporal boost [0-1] |
| conversationContext | object | No | - | Sequence-based retrieval config |
| sessionScope | enum | No | current | Session scope (current/all/recent) |

**Returns:** Ranked memories with relevance scores

**Implementation Notes:**
- E1 foundation with optional E5 asymmetric similarity (cause→effect 1.2x, effect→cause 0.8x)
- E10 multiplicative boosting per ARCH-28

---

#### trigger_consolidation
Trigger memory consolidation to merge similar memories and reduce redundancy.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| strategy | enum | No | similarity | Consolidation approach (similarity/temporal/semantic) |
| min_similarity | number | No | 0.85 | Similarity threshold [0-1] |
| max_memories | integer | No | 100 | Batch size (1-10000) |

**Returns:** Consolidation results with merge statistics

---

### Topic Tools

#### get_topic_portfolio
Get all discovered topics with profiles and stability metrics.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| format | enum | No | standard | Output format (brief/standard/verbose) |

**Returns:** Topics with names, spaces, tiers, and stability info

**Implementation Notes:** Uses weighted multi-space clustering (threshold ≥ 2.5). Excludes temporal embedders (E2-E4).

---

#### get_topic_stability
Get portfolio-level stability metrics including churn rate and entropy.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| hours | integer | No | 6 | Lookback period (1-168) |

**Returns:** Stability metrics, entropy, churn rate, phase breakdown

---

#### detect_topics
Force topic detection recalculation using HDBSCAN.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| force | boolean | No | false | Force even if recently computed |

**Returns:** Detected topics with weighted_agreement scores

**Implementation Notes:** HDBSCAN clustering requiring min 3 memories, threshold ≥ 2.5

---

#### get_divergence_alerts
Check for divergence from recent activity using SEMANTIC embedders only.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| lookback_hours | integer | No | 2 | Lookback period (1-48) |

**Returns:** Divergence alerts with semantic embedder scores

**Implementation Notes:** Uses semantic embedders only (E1, E5, E6, E7, E10, E12, E13) per AP-62, AP-63

---

### Curation Tools

#### forget_concept
Soft-delete a memory with 30-day recovery window.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| node_id | string (UUID) | Yes | - | Memory to forget |
| soft_delete | boolean | No | true | Use soft delete (30-day recovery per SEC-06) |

**Returns:** Deleted timestamp for recovery tracking

---

#### boost_importance
Adjust memory importance score by delta with clamping.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| node_id | string (UUID) | Yes | - | Memory to boost |
| delta | number | Yes | - | Change value [-1.0 to 1.0] |

**Returns:** Old value, delta, and new value (clamped to [0.0, 1.0])

---

### Merge Tool

#### merge_concepts
Merge two or more related concept nodes into unified node.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| source_ids | array (UUIDs) | Yes | - | Concepts to merge (2-10) |
| target_name | string | Yes | - | Name for merged concept (1-256 chars) |
| merge_strategy | enum | No | union | Merge approach (union/intersection/weighted_average) |
| rationale | string | Yes | - | Merge justification (1-1024 chars) |
| force_merge | boolean | No | false | Force despite conflicts |

**Returns:** Merged node with reversal_hash for 30-day undo (SEC-06)

---

### Sequence/Temporal Navigation Tools

#### get_conversation_context
Get memories around current conversation turn with auto-anchoring using E4.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| direction | enum | No | before | Direction to search (before/after/both) |
| windowSize | integer | No | 10 | Number of turns (1-50) |
| sessionOnly | boolean | No | true | Only current session |
| includeContent | boolean | No | true | Include full content |
| query | string | No | - | Semantic filter |
| minSimilarity | number | No | 0 | Semantic threshold [0-1] |

**Returns:** Memories ordered by sequence with position labels

---

#### get_session_timeline
Get ordered timeline of all session memories with sequence numbers.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| sessionId | string | No | current | Session ID |
| limit | integer | No | 50 | Max memories (1-200) |
| offset | integer | No | 0 | Pagination offset |
| sourceTypes | array | No | - | Filter sources (HookDescription/ClaudeResponse/Manual/MDFileChunk) |
| includeContent | boolean | No | false | Include full content |

**Returns:** Ordered timeline with position labels like "2 turns ago"

---

#### traverse_memory_chain
Navigate through chain of memories from anchor point.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| anchorId | string (UUID) | Yes | - | Starting memory |
| direction | enum | No | backward | Traversal direction (forward/backward/bidirectional) |
| hops | integer | No | 5 | Maximum hops (1-20) |
| semanticFilter | string | No | - | Topic filter |
| minSimilarity | number | No | 0.3 | Semantic similarity threshold [0-1] |
| includeContent | boolean | No | true | Include full content |

**Returns:** Memory chain with hop distances

---

#### compare_session_states
Compare memory state at different sequence points in session.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| beforeSequence | integer or "start" | Yes | - | Starting point |
| afterSequence | integer or "current" | Yes | - | Ending point |
| topicFilter | string | No | - | Optional topic focus |
| sessionId | string | No | current | Session ID |

**Returns:** Topics, memory counts, and differences

---

### Causal Tools

#### search_causal_relationships
Search for causal relationships using E5 asymmetric similarity with LLM-generated descriptions.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| query | string | Yes | - | Natural language causal query |
| direction | enum | No | all | Filter by direction (cause/effect/all) |
| topK | integer | No | 10 | Results to return (1-100) |
| includeSource | boolean | No | true | Include source content |

**Returns:** LLM-generated 1-3 paragraph descriptions with provenance links

**Implementation Notes:** E5 asymmetric with direction modifiers (cause→effect 1.2x, effect→cause 0.8x per AP-77)

---

#### search_causes
Abductive reasoning to find likely causes of observed effects.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| query | string | Yes | - | Observed effect to explain |
| topK | integer | No | 10 | Causes to return (1-50) |
| minScore | number | No | 0.1 | Abductive score threshold [0-1] |
| includeContent | boolean | No | false | Include full content |
| filterCausalDirection | enum | No | - | Filter by persisted direction (cause/effect/unknown) |

**Returns:** Ranked causes with abductive scores

**Implementation Notes:** E5 asymmetric with 0.8x effect→cause dampening (AP-77)

---

#### search_effects
Find effects/consequences of a given cause.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| query | string | Yes | - | Cause to find effects for |
| topK | integer | No | 10 | Effects to return (1-50) |
| minScore | number | No | 0.1 | Predictive score threshold [0-1] |
| includeContent | boolean | No | false | Include full content |
| filterCausalDirection | enum | No | - | Filter by direction (cause/effect/unknown) |

**Returns:** Ranked effects with predictive scores

**Implementation Notes:** E5 asymmetric with 1.2x cause→effect boost (AP-77)

---

#### get_causal_chain
Build and visualize transitive causal chains from anchor point.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| anchorId | string (UUID) | Yes | - | Starting memory |
| direction | enum | No | forward | Traversal direction (cause→effect or effect→cause) |
| maxHops | integer | No | 5 | Maximum hops (1-10) |
| minSimilarity | number | No | 0.3 | Similarity threshold [0-1] |
| includeContent | boolean | No | false | Include full content |

**Returns:** Causal chain visualization with hop-attenuated scores (0.9^hop)

---

### Causal Discovery Tools

#### trigger_causal_discovery
Manually trigger causal discovery agent using Qwen2.5 LLM.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| maxPairs | integer | No | 50 | Candidate pairs per run (1-200) |
| minConfidence | number | No | 0.7 | LLM confidence threshold [0.5-1.0] |
| sessionScope | enum | No | all | Memory scope (current/all/recent) |
| similarityThreshold | number | No | 0.5 | E1 similarity for candidates [0.3-0.9] |
| skipAnalyzed | boolean | No | true | Skip previously analyzed pairs |
| dryRun | boolean | No | false | Test without creating embeddings/edges |

**Returns:** Pairs analyzed, relationships found, VRAM usage

**Implementation Notes:** Qwen2.5-3B with FP16 on RTX 5090 (3B≈6GB, 7B≈14GB)

---

#### get_causal_discovery_status
Get causal discovery agent status and statistics.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| includeLastResult | boolean | No | true | Include last cycle results |
| includeGraphStats | boolean | No | true | Include causal graph stats |

**Returns:** Agent status, last results, VRAM usage, cumulative statistics

---

### Keyword Tools

#### search_by_keywords
Find memories matching specific keywords using E6 sparse embeddings.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| query | string | Yes | - | Keyword query |
| topK | integer | No | 10 | Results to return (1-50) |
| minScore | number | No | 0.1 | Blended score threshold [0-1] |
| blendWithSemantic | number | No | 0.3 | E6 weight in blend (0=pure E1, 1=pure E6) |
| useSpladeExpansion | boolean | No | true | Use E13 SPLADE term expansion |
| includeContent | boolean | No | false | Include full content |

**Returns:** Ranked memories with blended scores

---

### Code Tools

#### search_code
Find memories containing code patterns using E7 dense embeddings.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| query | string | Yes | - | Code query (describes functionality or patterns) |
| topK | integer | No | 10 | Results to return (1-50) |
| minScore | number | No | 0.2 | Blended score threshold [0-1] |
| blendWithSemantic | number | No | 0.4 | E7 weight (0=pure E1, 1=pure E7) |
| includeContent | boolean | No | false | Include full content |

**Returns:** Ranked memories with detected language info

**Implementation Notes:** E7 (Qodo-Embed-1-1.5B, 1536D) via tree-sitter AST chunking

---

### Graph Tools

#### search_connections
Find memories connected to concept using asymmetric E8 similarity.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| query | string | Yes | - | Concept to find connections for |
| direction | enum | No | both | Connection direction (source/target/both) |
| topK | integer | No | 10 | Connections to return (1-50) |
| minScore | number | No | 0.1 | Connection score threshold [0-1] |
| includeContent | boolean | No | false | Include full content |
| filterGraphDirection | enum | No | - | Filter by persisted direction (source/target/unknown) |

**Returns:** Connected memories with scores

**Implementation Notes:** E8 asymmetric with 1.2x/0.8x direction modifiers

---

#### get_graph_path
Build multi-hop graph paths from anchor point with visualization.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| anchorId | string (UUID) | Yes | - | Starting memory |
| direction | enum | No | forward | Traversal direction |
| maxHops | integer | No | 5 | Maximum hops (1-10) |
| minSimilarity | number | No | 0.3 | Similarity threshold [0-1] |
| includeContent | boolean | No | false | Include full content |

**Returns:** Graph paths with hop-attenuated scores (0.9^hop)

---

#### discover_graph_relationships
Discover graph relationships using LLM analysis with asymmetric E8 embeddings.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| memory_ids | array (UUIDs) | Yes | - | Memories to analyze (2-50) |
| relationship_types | array | No | - | Filter by types (imports/calls/cites/extends/etc.) |
| relationship_categories | array | No | - | Filter by categories (containment/dependency/reference/implementation/extension/invocation) |
| content_domain | enum | No | general | Domain hint (code/legal/academic/general) |
| min_confidence | number | No | 0.7 | Confidence threshold [0-1] |
| batch_size | integer | No | 50 | Candidate pairs per batch (1-100) |

**Returns:** Discovered relationships with confidence scores

**Implementation Notes:** Qwen2.5-3B with 20 relationship types across 4 domains

---

#### validate_graph_link
Validate proposed graph link using LLM analysis.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| source_id | string (UUID) | Yes | - | Source memory (points to) |
| target_id | string (UUID) | Yes | - | Target memory (pointed to) |
| expected_relationship_type | enum | No | - | Expected relationship to validate |

**Returns:** Validation result with confidence score, detected type, category, direction

---

### Robustness Tools

#### search_robust
Find memories using E9 noise-robust structural matching with typo tolerance.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| query | string | Yes | - | Query (typos OK, minLength 3) |
| topK | integer | No | 10 | Results to return (1-50) |
| minScore | number | No | 0.1 | Blended score threshold [0-1] |
| e9DiscoveryThreshold | number | No | 0.7 | E9 score for blind-spot marking [0-1] |
| e1WeaknessThreshold | number | No | 0.5 | E1 score for "missed" marking [0-1] |
| includeContent | boolean | No | false | Include full content |
| includeE9Score | boolean | No | true | Include per-embedder scores |

**Returns:** Ranked memories with E9/E1 blind-spot analysis

**Implementation Notes:** E9 HDC (10,000-bit binary hypervectors) with character trigrams, projected to 1024D

---

### Intent Tools

#### search_by_intent
Find memories with similar intent/goal using E10 asymmetric retrieval.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| query | string | Yes | - | Intent or goal to search for |
| context | string | No | - | Alternative to query for situation-based search |
| topK | integer | No | 10 | Results to return (1-50) |
| minScore | number | No | 0.2 | Score threshold [0-1] |
| blendWithSemantic | number | No | 0.1 | Legacy parameter |
| includeContent | boolean | No | false | Include full content |
| weightProfile | enum | No | - | Weight profile (intent_search/intent_enhanced/balanced/etc.) |

**Returns:** Ranked memories with intent alignment scores

**Implementation Notes:** E10 multiplicative boost per ARCH-17: strong E1→5% boost, medium→10%, weak→15%. Clamped [0.8, 1.2]

---

### Entity Tools

#### extract_entities
Extract and canonicalize entities from text with KB lookup.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| text | string | Yes | - | Text to extract from |
| includeUnknown | boolean | No | true | Include heuristic-detected entities |
| groupByType | boolean | No | false | Group by type |

**Returns:** Entities with canonical forms and types

**Implementation Notes:** Pattern matching with KB lookup (postgres→postgresql, k8s→kubernetes)

---

#### search_by_entities
Find memories containing specific entities with entity-aware ranking.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| entities | array | Yes | - | Entity names to search |
| entityTypes | array | No | - | Filter by types |
| matchMode | enum | No | any | Any or all entities |
| topK | integer | No | 10 | Results to return (1-50) |
| minScore | number | No | 0.2 | Threshold [0-1] |
| includeContent | boolean | No | false | Include full content |
| boostExactMatch | number | No | 1.3 | Exact match multiplier [1.0-3.0] |

**Returns:** Ranked memories with entity scores

**Implementation Notes:** E11 embeddings + entity Jaccard similarity hybrid scoring

---

#### infer_relationship
Infer relationship between two entities using TransE knowledge graph operations.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| headEntity | string | Yes | - | Subject entity |
| tailEntity | string | Yes | - | Object entity |
| headType | enum | No | - | Optional type hint |
| tailType | enum | No | - | Optional type hint |
| topK | integer | No | 5 | Relation candidates (1-20) |
| includeScore | boolean | No | true | Include TransE scores |

**Returns:** Ranked relation candidates with scores

**Implementation Notes:** TransE formula: r̂ = t - h, match against known relations

---

#### find_related_entities
Find entities with given relationship using TransE.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| entity | string | Yes | - | Source entity |
| relation | string | Yes | - | Relationship type |
| direction | enum | No | outgoing | Direction (outgoing/incoming) |
| entityType | enum | No | - | Filter results to type |
| topK | integer | No | 10 | Results to return (1-50) |
| minScore | number | No | - | TransE score threshold (negative values) |
| searchMemories | boolean | No | true | Filter to entities in memories |

**Returns:** Ranked entities with TransE scores

**Implementation Notes:** TransE: outgoing t̂ = h + r, incoming ĥ = t - r

---

#### validate_knowledge
Score (subject, predicate, object) triple validity using TransE.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| subject | string | Yes | - | Head entity |
| predicate | string | Yes | - | Relation |
| object | string | Yes | - | Tail entity |
| subjectType | enum | No | - | Optional type hint |
| objectType | enum | No | - | Optional type hint |

**Returns:** Validation result (valid/uncertain/unlikely) with supporting/contradicting memories

**Implementation Notes:** TransE score = -||h + r - t||₂

---

#### get_entity_graph
Build and visualize entity relationship graph in memories.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| centerEntity | string | No | - | Optional focal entity |
| maxNodes | integer | No | 50 | Max nodes (1-500) |
| maxDepth | integer | No | 2 | Max hops from center (1-5) |
| entityTypes | array | No | - | Filter by types |
| minRelationScore | number | No | 0.3 | Edge score threshold [0-1] |
| includeMemoryCounts | boolean | No | true | Include reference counts |

**Returns:** Graph with entity nodes and relationship edges

---

### Embedder-First Search Tools

#### search_by_embedder
Search using any embedder (E1-E13) as primary perspective.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| embedder | enum (E1-E13) | Yes | - | Primary embedder |
| query | string | Yes | - | Search query |
| topK | integer | No | 10 | Results to return (1-100) |
| minSimilarity | number | No | 0 | Similarity threshold [0-1] |
| includeContent | boolean | No | false | Include full content |
| includeAllScores | boolean | No | false | Include all 13 embedder scores |

**Returns:** Results ranked by selected embedder

---

#### get_embedder_clusters
Explore clusters in specific embedder's space.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| embedder | enum (E1-E13) | Yes | - | Embedder space |
| minClusterSize | integer | No | 3 | Min memories per cluster (2-50) |
| topClusters | integer | No | 10 | Max clusters (1-50) |
| includeSamples | boolean | No | true | Include sample memories |
| samplesPerCluster | integer | No | 3 | Samples per cluster (1-10) |

**Returns:** Clusters with sample memories

---

#### compare_embedder_views
Compare how different embedders rank same query side-by-side.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| query | string | Yes | - | Search query |
| embedders | array (E1-E13) | Yes | - | Embedders to compare (2-5) |
| topK | integer | No | 5 | Top results per embedder (1-20) |
| includeContent | boolean | No | false | Include full content |

**Returns:** Rankings side-by-side with agreement/unique finds

---

#### list_embedder_indexes
List all 13 embedder indexes with statistics.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| includeDetails | boolean | No | true | Include detailed stats |

**Returns:** Index stats (dimension, type, vector count, size, GPU residency)

---

### Temporal Tools

#### search_recent
Search with E2 temporal freshness boost for recent memories.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| query | string | Yes | - | Search query |
| topK | integer | No | 10 | Results to return (1-100) |
| temporalWeight | number | No | 0.3 | Boost weight [0.1-1.0] |
| decayFunction | enum | No | exponential | Decay curve (linear/exponential/step) |
| temporalScale | enum | No | meso | Time horizon (micro/meso/macro/long) |
| includeContent | boolean | No | true | Include full content |
| minSimilarity | number | No | 0.1 | Semantic threshold before boost [0-1] |

**Returns:** Ranked memories with recency prioritization

**Implementation Notes:** E2 POST-retrieval boost only (per ARCH-25, not in fusion)

---

#### search_periodic
Search for memories matching periodic time patterns (E3).

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| query | string | Yes | - | Search query |
| topK | integer | No | 10 | Results to return (1-100) |
| targetHour | integer | No | - | Target hour 0-23 (auto-detected if not provided) |
| targetDayOfWeek | integer | No | - | Target day 0-6 (auto-detected if not provided) |
| autoDetect | boolean | No | false | Auto-detect from current time |
| periodicWeight | number | No | 0.3 | Boost weight [0.1-1.0] |
| includeContent | boolean | No | true | Include full content |
| minSimilarity | number | No | 0.1 | Semantic threshold [0-1] |

**Returns:** Ranked memories with periodic pattern matching

**Implementation Notes:** E3 POST-retrieval boost for time-of-day/day-of-week patterns

---

### Graph Linking Tools

#### get_memory_neighbors
Get K nearest neighbors of memory in specific embedder space.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| memory_id | string (UUID) | Yes | - | Memory to find neighbors for |
| embedder_id | integer | No | 0 | Embedder space (0=E1, 4=E5, 6=E7, 7=E8, 9=E10, 10=E11) |
| top_k | integer | No | 10 | Neighbors to return (1-50) |
| min_similarity | number | No | 0 | Similarity threshold [0-1] |
| include_content | boolean | No | false | Include full content |

**Returns:** Neighbors sorted by similarity in selected space

---

#### get_typed_edges
Get typed edges from memory representing relationships from embedder agreement.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| memory_id | string (UUID) | Yes | - | Source memory |
| edge_type | enum | No | - | Filter by type (semantic_similar/code_related/entity_shared/causal_chain/graph_connected/intent_aligned/keyword_overlap/multi_agreement) |
| direction | enum | No | outgoing | Edge direction (outgoing/incoming/both) |
| min_weight | number | No | 0 | Weight threshold [0-1] |
| include_content | boolean | No | false | Include full content |

**Returns:** Typed edges with weights and target memories

---

#### traverse_graph
Multi-hop graph traversal from starting memory.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| start_memory_id | string (UUID) | Yes | - | Starting memory |
| max_hops | integer | No | 2 | Traversal depth (1-5) |
| edge_type | enum | No | - | Filter by edge type |
| min_weight | number | No | 0.3 | Weight threshold [0-1] |
| max_results | integer | No | 20 | Max paths (1-100) |
| include_content | boolean | No | false | Include full content |

**Returns:** Paths through knowledge graph

---

#### get_unified_neighbors
Find neighbors via Weighted RRF fusion across all 13 embedders (excluding temporal E2-E4).

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| memory_id | string (UUID) | Yes | - | Memory to find neighbors for |
| weight_profile | enum | No | semantic_search | Weight profile (semantic_search/code_search/causal_reasoning/fact_checking/intent_search/intent_enhanced/graph_reasoning/category_weighted) |
| top_k | integer | No | 10 | Neighbors to return (1-50) |
| min_score | number | No | 0 | RRF score threshold [0-1] |
| include_content | boolean | No | false | Include full content |
| include_embedder_breakdown | boolean | No | true | Include per-embedder scores/ranks |

**Returns:** Unified neighbors with embedder breakdown

**Implementation Notes:** Per ARCH-21 Weighted RRF (not weighted sum). Per AP-60 excludes E2-E4 temporal.

---

### File Watcher Tools

#### list_watched_files
List all files with embeddings from file watcher.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| include_counts | boolean | No | true | Include chunk counts |
| path_filter | string | No | - | Glob pattern filter |

**Returns:** File paths with chunk counts and last update times

---

#### get_file_watcher_stats
Get statistics about file watcher content.

**Parameters:** None

**Returns:** Total files, chunks, avg chunks/file, min/max values

---

#### delete_file_content
Delete all embeddings for specific file path.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| file_path | string | Yes | - | Absolute file path |
| soft_delete | boolean | No | true | Use soft delete (30-day recovery per SEC-06) |

**Returns:** Deletion confirmation

---

#### reconcile_files
Find orphaned files (embeddings exist but file missing on disk) and optionally delete.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| dry_run | boolean | No | true | Only report without deleting |
| base_path | string | No | - | Optional base path to limit scope |

**Returns:** Orphaned files list with deletion status

---

### Maintenance Tools

#### repair_causal_relationships
Repair corrupted causal relationship entries by removing deserialization failures.

**Parameters:** None

**Returns:** (deleted_count, total_scanned) statistics

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| RUST_LOG | warn | Logging level |
| CONTEXT_GRAPH_STORAGE_PATH | ./data | RocksDB path |
| CONTEXT_GRAPH_MODELS_PATH | ./models | Model directory |
| MCP_TIMEOUT | 60000 | MCP timeout (ms) |

### Default Configuration (`config/default.toml`)

```toml
[general]
phase = "ghost"  # Development phase

[server]
transport = "stdio"  # MCP transport
tcp_port = 8080

[storage]
backend = "memory"  # Dev: memory; Prod: RocksDB

[embeddings]
provider = "stub"  # Dev: stub; Prod: candle+CUDA

[cuda]
enabled = false  # GPU acceleration

[watcher]
enabled = true  # File watching
watch_paths = ["./docs"]
extensions = ["md"]

[watcher.code]
enabled = false
use_ast_chunker = true
target_chunk_size = 500
min_chunk_size = 100
max_chunk_size = 1000
```

### Native Hooks (.claude/settings.json)

| Hook | Timeout | Purpose |
|------|---------|---------|
| SessionStart | 5000ms | Load topic portfolio, warm indexes |
| UserPromptSubmit | 2000ms | Embed prompt, search, detect divergence, inject context |
| PreToolUse | 500ms | Inject brief relevant context |
| PostToolUse | 3000ms | Capture + embed as HookDescription |
| SessionEnd | 30000ms | Persist state, run HDBSCAN |

---

## Architectural Rules

### Core Rules (MUST Follow)

| Rule | Description |
|------|-------------|
| ARCH-01 | TeleologicalArray is atomic - all 13 embeddings or nothing |
| ARCH-02 | Apples-to-apples only - compare E1↔E1, never E1↔E5 |
| ARCH-04 | Temporal (E2-E4) NEVER count toward topics |
| ARCH-05 | All 13 embedders required - missing = fatal |
| ARCH-06 | All memory ops through MCP tools only |
| ARCH-09 | Topic threshold: weighted_agreement >= 2.5 |
| ARCH-10 | Divergence detection: SEMANTIC embedders only (E1,E5,E6,E7,E10,E12,E13) |
| ARCH-12 | E1 is foundation - all retrieval starts with E1 |
| ARCH-13 | Strategies: E1Only (default), MultiSpace (E1+enhancers), Pipeline (E13→E1→E12) |
| ARCH-17 | Strong E1 (>0.8): enhancers refine. Weak E1 (<0.4): enhancers broaden |
| ARCH-18 | E5 Causal: asymmetric similarity (cause→effect direction matters) |
| ARCH-21 | Multi-space fusion: use Weighted RRF, not weighted sum |
| ARCH-25 | Temporal boosts POST-retrieval only, NOT in similarity fusion |
| ARCH-28 | E10 uses multiplicative boost: E1 * (1 + boost), NOT linear blending |
| ARCH-29 | E10 boost adapts: strong E1=5%, medium=10%, weak=15% |
| ARCH-30 | E10 alignment: >0.5=boost, <0.5=reduce, =0.5=neutral |
| ARCH-33 | E10 multiplier clamped to [0.8, 1.2] |

### Code-Specific Rules

| Rule | Description |
|------|-------------|
| ARCH-CODE-01 | Code entities stored separately from teleological memories |
| ARCH-CODE-02 | E7 is primary embedder for code, not part of 13-embedder array |
| ARCH-CODE-03 | AST chunking preserves syntactic boundaries |
| ARCH-CODE-04 | Tree-sitter for parsing, NOT regex extraction |

---

## Anti-Patterns

### Forbidden Actions

| Rule | Forbidden Action |
|------|------------------|
| AP-02 | No cross-embedder comparison (E1↔E5) |
| AP-04 | No partial TeleologicalArray |
| AP-05 | No embedding fusion into single vector |
| AP-60 | Temporal (E2-E4) MUST NOT count toward topics |
| AP-73 | Temporal MUST NOT be used in similarity fusion |
| AP-74 | E12 ColBERT: reranking ONLY, not initial retrieval |
| AP-75 | E13 SPLADE: Stage 1 recall ONLY, not final ranking |
| AP-77 | E5 MUST NOT use symmetric cosine - causal is directional |
| AP-79 | MUST NOT use simple weighted sum - use Weighted RRF |
| AP-80 | E10 MUST NOT use linear blending - makes E10 compete with E1 |
| AP-84 | E10 MUST NOT override E1 - when E1=0, result=0 |

### Code Anti-Patterns

| Rule | Forbidden Action |
|------|------------------|
| AP-CODE-01 | NEVER chunk code by word count - use AST boundaries |
| AP-CODE-02 | NEVER store code in teleological memory graph |
| AP-CODE-03 | NEVER use E1 alone for code search - E7 is primary |

---

## Key Thresholds

| Threshold | Value | Purpose |
|-----------|-------|---------|
| Topic Detection | 2.5 | weighted_agreement for topic detection |
| High Similarity | 0.75 | High similarity threshold |
| Low Similarity | 0.30 | Divergence threshold |
| Duplicate Detection | 0.90 | Duplicate detection threshold |
| Topic Validation | 0.30 | silhouette_score minimum |

---

## GPU Acceleration

**Target**: RTX 5090 Blackwell architecture

**Dependencies**:
- candle-core 0.9.2-alpha with CUDA 13.x support
- Parallel agent access via `spawn_blocking`

**VRAM Requirements**:
| Component | VRAM |
|-----------|------|
| E7 (Qodo-Embed-1.5B) | ~3GB |
| Qwen2.5-3B (Causal) | ~6GB |
| Qwen2.5-7B (optional) | ~14GB |

---

*Generated from ContextGraph v6.5.0 codebase analysis*
