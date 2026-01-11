# Context Graph MCP Complete Guide

A comprehensive reference for the Context Graph MCP (Model Context Protocol) server - all 35 tools, 40+ JSON-RPC methods, the 13-embedder teleological system, GWT consciousness, and complete protocol details.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [The 13-Embedder Teleological System](#the-13-embedder-teleological-system)
4. [The 5-Layer Bio-Nervous System](#the-5-layer-bio-nervous-system)
5. [MCP Protocol & Transport](#mcp-protocol--transport)
6. [All 35 MCP Tools Reference](#all-35-mcp-tools-reference)
7. [JSON-RPC Methods Reference](#json-rpc-methods-reference)
8. [Weight Profiles](#weight-profiles)
9. [Cognitive Pulse System](#cognitive-pulse-system)
10. [Global Workspace Theory (GWT)](#global-workspace-theory-gwt)
11. [Johari Window Classification](#johari-window-classification)
12. [Dream Consolidation System](#dream-consolidation-system)
13. [Autonomous North Star System](#autonomous-north-star-system)
14. [Error Codes Reference](#error-codes-reference)
15. [Claude Code Integration](#claude-code-integration)
16. [Performance Targets](#performance-targets)

---

## Quick Start

### Add MCP Server to Claude Code

```bash
# Using cargo (development)
claude mcp add context-graph -- cargo run --manifest-path /path/to/contextgraph/crates/context-graph-mcp/Cargo.toml

# Using pre-built binary
claude mcp add context-graph -- /path/to/context-graph-mcp
```

### Basic Usage Flow

```
1. Store memories     → store_memory / inject_context
2. Bootstrap system   → auto_bootstrap_north_star
3. Search & retrieve  → search_graph / search_teleological
4. Monitor state      → get_memetic_status (check _cognitive_pulse)
5. Maintain health    → trigger_dream / trigger_consolidation
```

---

## Architecture Overview

The Context Graph MCP server is a bio-inspired knowledge management system implementing:

- **13 specialized embedding models** (teleological system)
- **Global Workspace Theory (GWT)** for computational consciousness
- **Kuramoto oscillator networks** for neural synchronization
- **Johari Window classification** per embedder for introspective awareness
- **Dream consolidation cycles** (NREM + REM phases)
- **Neuromodulation system** (Dopamine, Serotonin, Noradrenaline, Acetylcholine)
- **Autonomous goal alignment** with drift detection and correction

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONTEXT GRAPH MCP SERVER                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐           │
│  │  Multi-Array    │   │  Teleological   │   │   Johari        │           │
│  │  Embedding      │   │  Memory Store   │   │   Transition    │           │
│  │  Provider       │   │  (RocksDB)      │   │   Manager       │           │
│  │  (13 Embedders) │   │                 │   │                 │           │
│  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘           │
│           │                     │                     │                     │
│           └──────────┬──────────┴──────────┬──────────┘                     │
│                      │                     │                                │
│  ┌───────────────────▼─────────────────────▼───────────────────┐           │
│  │                    MCP Request Handlers                      │           │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐│           │
│  │  │ Memory  │ │ Search  │ │  GWT    │ │ Dream   │ │Teleolog.││           │
│  │  │ Handler │ │ Handler │ │ Handler │ │ Handler │ │ Handler ││           │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘│           │
│  └──────────────────────────────────────────────────────────────┘           │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │               GWT/Consciousness System                        │           │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ │           │
│  │  │  Kuramoto  │ │  Global    │ │  Self-Ego  │ │   Meta     │ │           │
│  │  │  Network   │ │  Workspace │ │   Node     │ │ Cognitive  │ │           │
│  │  │ (13 osc.)  │ │  (WTA)     │ │            │ │   Loop     │ │           │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘ │           │
│  └──────────────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The 13-Embedder Teleological System

The teleological system uses 13 specialized embedding models, each capturing a different semantic dimension. Together they form a "teleological fingerprint" for each memory.

### Embedder Specifications

| Index | ID | Name | Model | Dimension | Purpose (V_goal) | Quantization |
|-------|-----|------|-------|-----------|------------------|--------------|
| 0 | E1 | Semantic | e5-large-v2 | 1024D (Matryoshka) | V_meaning | PQ-8 |
| 1 | E2 | Temporal-Recent | Exponential decay | 512D | V_freshness | Float8 |
| 2 | E3 | Temporal-Periodic | Fourier | 512D | V_periodicity | Float8 |
| 3 | E4 | Temporal-Positional | Sinusoidal PE | 512D | V_ordering | Float8 |
| 4 | E5 | Causal | Longformer (SCM) | 768D | V_causality | PQ-8 |
| 5 | E6 | Sparse | SPLADE | ~30K (5% active) | V_selectivity | Sparse |
| 6 | E7 | Code | Qodo-Embed (AST) | 1536D | V_correctness | PQ-8 |
| 7 | E8 | Graph | MiniLM (GNN) | 384D | V_connectivity | Float8 |
| 8 | E9 | HDC | Hyperdimensional | 1024D | V_robustness | Binary |
| 9 | E10 | Multimodal | CLIP | 768D | V_multimodality | PQ-8 |
| 10 | E11 | Entity | MiniLM (TransE) | 384D | V_factuality | Float8 |
| 11 | E12 | Late-Interaction | ColBERT + MaxSim | 128D/token | V_precision | Token pruning + SIMD |
| 12 | E13 | SPLADE v3 | SPLADE | ~30K sparse | V_keyword_precision | Sparse |

### Embedding Groups (6)

| Group | Embedders | Purpose |
|-------|-----------|---------|
| **Factual** | E1 (Semantic), E11 (Entity) | Core meaning and factual grounding |
| **Temporal** | E2 (Recent), E3 (Periodic), E4 (Positional) | Time-aware representations |
| **Causal** | E5 (Causal) | Cause-effect relationships |
| **Relational** | E8 (Graph), E10 (Multimodal) | Structural and cross-modal connections |
| **Qualitative** | E9 (HDC), E12 (Late-Interaction) | Robust and precise representations |
| **Implementation** | E6 (Sparse), E7 (Code), E13 (SPLADE) | Keyword and code understanding |

### MaxSim Late Interaction (E12)

E12 now implements **Stage 5 MaxSim scoring** with SIMD acceleration for ColBERT-style late interaction:

```
MaxSim(Q, D) = (1/|Q|) × Σᵢ max_j cos(qᵢ, dⱼ)
```

**Performance Targets**:
- Score 50 candidates: <15ms
- Single MaxSim (50×50 tokens): <300μs
- SIMD vs scalar speedup: >4x

The MaxSim scorer uses AVX2 intrinsics for 4-8x speedup on 128D token vectors, with parallel dot product computation via rayon.

### Per-Embedder Entropy Methods (ΔS)

Each embedder type now has specialized entropy computation per constitution.yaml delta_sc.ΔS_methods:

| Embedder | Method | Description |
|----------|--------|-------------|
| E1 (Semantic) | GMM + Mahalanobis | Gaussian mixture model distance |
| E2-E4, E8 | Normalized KNN | Standard k-nearest-neighbor entropy |
| E5 (Causal) | Asymmetric KNN | Direction-aware distance modifiers |
| E6, E13 (Sparse/SPLADE) | Jaccard Active | 1 - Jaccard(active_dims) |
| E7 (Code) | GMM + KNN Hybrid | Combined approach for AST embeddings |
| E9 (HDC) | Hamming Prototype | Distance to learned binary prototypes |
| E10-E12 | Default KNN | Fallback normalized KNN |

All outputs are clamped to [0.0, 1.0] per AP-10 constitution requirement

### Kuramoto Natural Frequencies (Hz)

Each embedder has a natural oscillation frequency for synchronization:

| E1 | E2-E4 | E5 | E6 | E7 | E8 | E9 | E10 | E11 | E12 | E13 |
|----|-------|----|----|----|----|----|----|-----|-----|-----|
| 40 | 8 | 25 | 4 | 25 | 12 | 80 | 40 | 15 | 60 | 4 |

### Teleological Vector Components

Each memory has three key components:

1. **Purpose Vector (13D)**: Normalized activation across all 13 embedders
2. **Cross-Correlations (78D)**: Pairwise correlations between all C(13,2) = 78 embedder pairs
3. **Group Alignments (6D)**: Aggregated alignment scores per functional group

---

## The 5-Layer Bio-Nervous System

### Layer 1: Perception Layer
- **Function**: Input processing and initial embedding
- **Components**: Multi-array embedding provider, content preprocessing
- **Output**: Raw 13-embedder outputs

### Layer 2: Memory Layer
- **Function**: Storage and retrieval
- **Components**: RocksDB teleological store, HNSW indices (usearch)
- **HNSW**: O(log n) graph traversal via usearch crate (NOT brute force)
- **Output**: Teleological fingerprints, search results

### Layer 3: Reasoning Layer
- **Function**: Inference and analysis
- **Components**: Causal inference engine, UTL processor
- **Output**: Causal relationships, learning metrics

### Layer 4: Meta Layer
- **Function**: Self-monitoring and consciousness
- **Components**: GWT system, Kuramoto network, Meta-UTL tracker
- **Output**: Consciousness state, cognitive pulse

### Layer 5: Action Layer
- **Function**: Output generation and steering
- **Components**: Steering system (Gardener, Curator, Assessor)
- **Output**: Actions, recommendations, feedback

---

## MCP Protocol & Transport

### Protocol Version
MCP 2024-11-05 specification

### Transport Options
- **STDIO** (default): stdin/stdout JSON-RPC
- **TCP**: Newline-delimited JSON (NDJSON) on configurable port

### Request Format
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "tool_name",
    "arguments": { ... }
  }
}
```

### Response Format
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": { ... },
  "X-Cognitive-Pulse": {
    "entropy": 0.42,
    "coherence": 0.78,
    "learning_score": 0.55,
    "quadrant": "Open",
    "suggested_action": "direct_recall"
  }
}
```

---

## All 35 MCP Tools Reference

### Tool Categories Summary

| Category | Count | Tools |
|----------|-------|-------|
| Core Memory | 3 | inject_context, store_memory, search_graph |
| System Status | 3 | get_memetic_status, get_graph_manifest, utl_status |
| GWT/Consciousness | 6 | get_consciousness_state, get_kuramoto_sync, get_workspace_status, get_ego_state, trigger_workspace_broadcast, adjust_coupling |
| ATC | 3 | get_threshold_status, get_calibration_metrics, trigger_recalibration |
| Dream | 4 | trigger_dream, get_dream_status, abort_dream, get_amortized_shortcuts |
| Neuromodulation | 2 | get_neuromodulation_state, adjust_neuromodulator |
| Steering | 1 | get_steering_feedback |
| Causal | 1 | omni_infer |
| Teleological | 5 | search_teleological, compute_teleological_vector, fuse_embeddings, update_synergy_matrix, manage_teleological_profile |
| Autonomous | 7 | auto_bootstrap_north_star, get_alignment_drift, trigger_drift_correction, get_pruning_candidates, trigger_consolidation, discover_sub_goals, get_autonomous_status |
| **Total** | **35** | |

---

### Core Memory Tools (3)

#### 1. `inject_context`
Inject context into the knowledge graph with UTL (Unified Theory of Learning) processing. Analyzes content for learning potential and stores with computed metrics.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | Yes | - | The content to inject |
| `rationale` | string | Yes | - | Why this context is relevant |
| `modality` | enum | No | "text" | `text`, `code`, `image`, `audio`, `structured`, `mixed` |
| `importance` | number | No | 0.5 | Importance score [0.0, 1.0] |

**Returns**: `fingerprintId`, UTL metrics (entropy, coherence, learning_score, surprise)

**Example**:
```json
{
  "name": "inject_context",
  "arguments": {
    "content": "The Kuramoto model describes synchronization of coupled oscillators",
    "rationale": "Core concept for consciousness modeling",
    "modality": "text",
    "importance": 0.8
  }
}
```

---

#### 2. `store_memory`
Store a memory node directly without UTL processing. Use for raw storage when learning analysis is not needed.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | Yes | - | The content to store |
| `importance` | number | No | 0.5 | Importance score [0.0, 1.0] |
| `modality` | enum | No | "text" | Content modality |
| `tags` | array[string] | No | - | Optional tags for categorization |

**Example**:
```json
{
  "name": "store_memory",
  "arguments": {
    "content": "function fibonacci(n) { return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2); }",
    "modality": "code",
    "tags": ["algorithm", "recursion", "math"]
  }
}
```

---

#### 3. `search_graph`
Search the knowledge graph using semantic similarity. Returns nodes matching the query with relevance scores.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | The search query text |
| `topK` | integer | No | 10 | Maximum results (1-100) |
| `minSimilarity` | number | No | 0.0 | Minimum similarity threshold [0.0, 1.0] |
| `modality` | enum | No | - | Filter results by modality |

**Example**:
```json
{
  "name": "search_graph",
  "arguments": {
    "query": "neural oscillator synchronization",
    "topK": 5,
    "minSimilarity": 0.7
  }
}
```

---

### System Status Tools (3)

#### 4. `get_memetic_status`
Get current system status with LIVE UTL metrics: entropy (novelty), coherence (understanding), learning score (magnitude), Johari quadrant, consolidation phase, and suggested action. Also returns node count and 5-layer status.

**Parameters**: None required

**Returns**: Phase, fingerprint count, UTL metrics, 5-layer bio-nervous system status

---

#### 5. `get_graph_manifest`
Get the 5-layer bio-nervous system architecture description and current layer statuses.

**Parameters**: None required

---

#### 6. `utl_status`
Query current UTL (Unified Theory of Learning) system state including lifecycle phase, entropy, coherence, learning score, Johari quadrant, and consolidation phase.

**Parameters**: None required

---

### GWT/Consciousness Tools (6)

#### 7. `get_consciousness_state`
Get current consciousness state from the GWT system.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | No | default | Session ID for tracking |

**Returns**: Kuramoto r, consciousness level C, meta-cognitive score, differentiation, workspace status, identity coherence

---

#### 8. `get_kuramoto_sync`
Get Kuramoto oscillator network synchronization state.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | No | default | Session ID |

**Returns**:
- `r` - Order parameter [0,1] (1 = fully synchronized)
- `psi` - Mean phase
- `phases` - Array of 13 oscillator phases
- `natural_freqs` - Natural frequencies per oscillator
- `coupling` - Current coupling strength K

---

#### 9. `get_workspace_status`
Get Global Workspace status (Winner-Take-All selection details).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | No | default | Session ID |

**Returns**: Active memory, broadcasting state, coherence threshold, competing candidates

---

#### 10. `get_ego_state`
Get Self-Ego Node state for identity monitoring.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | No | default | Session ID |

**Returns**: 13D purpose vector, identity status, coherence with actions, trajectory length

**Persistence**: The Self-Ego Node is persisted to RocksDB via `SELF_EGO_NODE` column family. On restart, identity is restored automatically, enabling continuity across sessions

---

#### 11. `trigger_workspace_broadcast`
Trigger winner-take-all workspace broadcast with a specific memory.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | string (UUID) | Yes | - | Memory UUID to broadcast |
| `importance` | number | No | 0.8 | Importance score [0.0, 1.0] |
| `alignment` | number | No | 0.8 | North Star alignment [0.0, 1.0] |
| `force` | boolean | No | false | Force broadcast below threshold |

---

#### 12. `adjust_coupling`
Adjust Kuramoto oscillator coupling strength K. Higher K = faster synchronization.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `new_K` | number | Yes | - | New coupling strength [0, 10] |

**Returns**: old_K, new_K, predicted_r

---

### ATC (Adaptive Threshold Calibration) Tools (3)

#### 13. `get_threshold_status`
Get current ATC threshold status including all thresholds, calibration state, and adaptation metrics.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `domain` | enum | No | "General" | `Code`, `Medical`, `Legal`, `Creative`, `Research`, `General` |
| `embedder_id` | integer | No | - | Specific embedder (1-13) for detailed info |

---

#### 14. `get_calibration_metrics`
Get calibration quality metrics: ECE (Expected Calibration Error), MCE (Maximum Calibration Error), Brier Score.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `timeframe` | enum | No | "24h" | `1h`, `24h`, `7d`, `30d` |

**Targets**: ECE < 0.05 (excellent), < 0.10 (good)

---

#### 15. `trigger_recalibration`
Manually trigger recalibration at a specific ATC level.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `level` | integer | Yes | - | ATC level (1-4) |
| `domain` | enum | No | "General" | Domain context |

**ATC Levels**:
- **Level 1**: EWMA drift adjustment
- **Level 2**: Temperature scaling
- **Level 3**: Thompson Sampling exploration
- **Level 4**: Bayesian meta-optimization

---

### Dream Tools (4)

#### 16. `trigger_dream`
Manually trigger a dream consolidation cycle. System must be idle (activity < 0.15).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `force` | boolean | No | false | Force dream even if activity > threshold (not recommended) |

**Phases**: NREM (3 min) + REM (2 min)
**Constitution mandate**: Wake latency < 100ms on external query

---

#### 17. `get_dream_status`
Get current dream system status.

**Parameters**: None required

**Returns**: State (Awake/NREM/REM/Waking), GPU usage, activity level, time since last dream

---

#### 18. `abort_dream`
Abort current dream cycle. Must complete within 100ms.

**Parameters**: None required

**Returns**: Wake latency, partial dream report

---

#### 19. `get_amortized_shortcuts`
Get shortcut candidates from amortized learning. Returns paths traversed 5+ times with 3+ hops.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min_confidence` | number | No | 0.7 | Minimum confidence threshold [0,1] |
| `limit` | integer | No | 20 | Maximum shortcuts (1-100) |

---

### Neuromodulation Tools (2)

#### 20. `get_neuromodulation_state`
Get all 4 neuromodulator levels.

**Parameters**: None required

**Modulators and Ranges**:
| Modulator | Controls | Range |
|-----------|----------|-------|
| Dopamine | hopfield.beta | [1, 5] |
| Serotonin | space_weights | [0, 1] |
| Noradrenaline | attention.temp | [0.5, 2] |
| Acetylcholine | utl.lr | [0.001, 0.002] (read-only) |

---

#### 21. `adjust_neuromodulator`
Adjust a specific neuromodulator level. ACh is read-only (managed by GWT).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `modulator` | enum | Yes | - | `dopamine`, `serotonin`, `noradrenaline` |
| `delta` | number | Yes | - | Amount to add/subtract |

---

### Steering Tools (1)

#### 22. `get_steering_feedback`
Get steering feedback from Gardener (graph health), Curator (memory quality), and Assessor (performance).

**Parameters**: None required

**Returns**: SteeringReward [-1, 1], component scores, recommendations

---

### Causal Inference Tools (1)

#### 23. `omni_infer`
Perform omni-directional causal inference.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source` | string (UUID) | Yes | - | Source node UUID |
| `target` | string (UUID) | No | - | Target node UUID |
| `direction` | enum | No | "forward" | Inference direction |

**Directions**:
- `forward` - A→B effect
- `backward` - B→A cause
- `bidirectional` - A↔B mutual
- `bridge` - Cross-domain inference
- `abduction` - Best hypothesis

---

### Teleological Tools (5)

#### 24. `search_teleological`
Search across all 13 embedder dimensions with configurable strategy and scope.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query_content` | string | No | - | Content to search for (will be embedded) |
| `query_vector_id` | string | No | - | Existing vector ID to use as query |
| `strategy` | enum | No | "adaptive" | Search strategy |
| `scope` | enum | No | "full" | Comparison scope |
| `specific_groups` | array[enum] | No | - | Specific groups to compare |
| `specific_embedder` | integer | No | - | Single embedder index (0-12) |
| `weight_purpose` | number | No | 0.4 | Purpose vector weight [0,1] |
| `weight_correlations` | number | No | 0.35 | Cross-correlation weight [0,1] |
| `weight_groups` | number | No | 0.15 | Group alignments weight [0,1] |
| `min_similarity` | number | No | 0.3 | Minimum similarity threshold |
| `max_results` | integer | No | 20 | Maximum results (1-1000) |
| `include_breakdown` | boolean | No | true | Include per-component breakdown |

**Strategies**: `cosine`, `euclidean`, `synergy_weighted`, `group_hierarchical`, `cross_correlation_dominant`, `tucker_compressed`, `adaptive`

**Scopes**: `full`, `purpose_vector_only`, `cross_correlations_only`, `group_alignments_only`

**Groups**: `factual`, `temporal`, `causal`, `relational`, `qualitative`, `implementation`

---

#### 25. `compute_teleological_vector`
Compute a complete 13-embedder teleological vector from content.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | Yes | - | Content to compute vector for |
| `profile_id` | string | No | - | Profile ID for task-specific weighting |
| `compute_tucker` | boolean | No | false | Compute Tucker decomposition |
| `tucker_ranks` | array[int] | No | [4,4,128] | Tucker ranks [r1, r2, r3] |
| `include_per_embedder` | boolean | No | false | Include raw per-embedder outputs |

**Returns**: Purpose vector (13D), cross-correlations (78D), group alignments (6D), optional Tucker core

---

#### 26. `fuse_embeddings`
Fuse embeddings using synergy matrix and optional profile weights.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | string | Yes | - | Memory ID to fuse |
| `fusion_method` | enum | No | "hierarchical" | `linear`, `attention`, `gated`, `hierarchical`, `tucker` |
| `profile_id` | string | No | - | Profile ID |
| `custom_weights` | array[number] | No | - | Custom weights [E1..E13] (13 values) |
| `apply_synergy` | boolean | No | true | Apply synergy matrix |
| `store_result` | boolean | No | true | Store fused vector |

---

#### 27. `update_synergy_matrix`
Update synergy matrix from retrieval feedback. Implements online learning.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query_vector_id` | string | Yes | - | Query vector ID |
| `result_vector_id` | string | Yes | - | Result vector ID |
| `feedback` | enum | Yes | - | `relevant`, `not_relevant`, `partially_relevant` |
| `relevance_score` | number | No | - | Fine-grained relevance [0,1] |
| `learning_rate` | number | No | 0.01 | Learning rate [0.001, 0.5] |
| `update_scope` | enum | No | "contributing_pairs" | `all_pairs`, `high_synergy_only`, `contributing_pairs` |

---

#### 28. `manage_teleological_profile`
CRUD operations for task-specific teleological profiles.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | enum | Yes | - | `create`, `read`, `update`, `delete`, `list` |
| `profile_id` | string | No | - | Profile ID (required for read/update/delete) |
| `name` | string | No | - | Human-readable name |
| `task_type` | enum | No | - | `code_implementation`, `research`, `creative`, `analysis`, `debugging`, `documentation`, `custom` |
| `embedder_weights` | array[number] | No | - | Per-embedder weights [E1..E13] |
| `group_priorities` | object | No | - | Group priority weights |
| `fusion_strategy` | enum | No | "hierarchical" | Default fusion strategy |

---

### Autonomous Tools (7)

> **Note**: Manual North Star tools have been **REMOVED**. They created single 1024D embeddings that cannot be meaningfully compared to 13-embedder teleological arrays. Use the autonomous system which discovers purpose from stored fingerprints.

#### 29. `auto_bootstrap_north_star`
Bootstrap the autonomous North Star system from existing teleological embeddings. Discovers emergent purpose patterns and initializes drift detection, pruning, consolidation, and sub-goal discovery.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `confidence_threshold` | number | No | 0.7 | Minimum confidence [0,1] |
| `max_candidates` | integer | No | 10 | Maximum candidates (1-100) |

---

#### 30. `get_alignment_drift`
Get current alignment drift state including severity, trend, and recommendations.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `timeframe` | enum | No | "24h" | `1h`, `24h`, `7d`, `30d` |
| `include_history` | boolean | No | false | Include full drift history |

**Returns**: Severity, trend, recommendations

---

#### 31. `trigger_drift_correction`
Manually trigger drift correction cycle.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `force` | boolean | No | false | Force correction even if low severity |
| `target_alignment` | number | No | - | Target alignment to achieve |

**Strategies by severity**: Threshold adjustment, weight rebalancing, goal reinforcement, emergency intervention

---

#### 32. `get_pruning_candidates`
Identify memories eligible for pruning based on staleness, low alignment, redundancy, or orphaned status.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | integer | No | 20 | Maximum candidates (1-1000) |
| `min_staleness_days` | integer | No | 30 | Minimum age for staleness |
| `min_alignment` | number | No | 0.4 | Below this = candidate |

---

#### 33. `trigger_consolidation`
Trigger memory consolidation to merge similar memories.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `max_memories` | integer | No | 100 | Maximum to process (1-10000) |
| `strategy` | enum | No | "similarity" | `similarity`, `temporal`, `semantic` |
| `min_similarity` | number | No | 0.85 | Minimum similarity threshold |

---

#### 34. `discover_sub_goals`
Discover potential sub-goals from memory clusters.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min_confidence` | number | No | 0.6 | Minimum confidence [0,1] |
| `max_goals` | integer | No | 5 | Maximum to discover (1-20) |
| `parent_goal_id` | string | No | - | Parent goal (defaults to North Star) |

---

#### 35. `get_autonomous_status`
Get comprehensive autonomous system status.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `include_metrics` | boolean | No | false | Detailed per-service metrics |
| `include_history` | boolean | No | false | Recent operation history |
| `history_count` | integer | No | 10 | History entries (1-100) |

**Services**: Bootstrap, drift detector, drift corrector, pruning, consolidation, sub-goal discovery

---

## JSON-RPC Methods Reference

Beyond tools, the server exposes direct JSON-RPC methods:

### Lifecycle Methods
| Method | Description |
|--------|-------------|
| `initialize` | Initialize MCP server connection |
| `shutdown` | Graceful shutdown |

### Tools Protocol
| Method | Description |
|--------|-------------|
| `tools/list` | List all available tools with schemas |
| `tools/call` | Call a tool with arguments |

### Memory Operations
| Method | Description |
|--------|-------------|
| `memory/store` | Direct memory storage |
| `memory/retrieve` | Retrieve memory by ID |
| `memory/search` | Search memories |
| `memory/delete` | Delete memory |
| `memory/inject` | Inject with 13-embedder fingerprint |
| `memory/inject_batch` | Batch injection with parallel embedding |
| `memory/search_multi_perspective` | Multi-embedder search with RRF fusion |
| `memory/compare` | Single pair comparison |
| `memory/batch_compare` | 1-to-N comparison |
| `memory/similarity_matrix` | N×N similarity matrix |

### Search Operations
| Method | Description |
|--------|-------------|
| `search/multi` | Multi-space search |
| `search/single_space` | Single embedding space search |
| `search/by_purpose` | Purpose-aligned search |
| `search/weight_profiles` | Get available weight profiles |

### Purpose/Goal Operations
| Method | Description |
|--------|-------------|
| `purpose/query` | Query by 13D purpose vector similarity |
| `goal/hierarchy_query` | Navigate goal hierarchy |
| `goal/aligned_memories` | Find memories aligned to goal |
| `purpose/drift_check` | Detect alignment drift |

### Johari Operations
| Method | Description |
|--------|-------------|
| `johari/get_distribution` | Get per-embedder quadrant distribution |
| `johari/find_by_quadrant` | Find memories by quadrant |
| `johari/transition` | Execute single transition |
| `johari/transition_batch` | Atomic batch transitions |
| `johari/cross_space_analysis` | Cross-space analysis |
| `johari/transition_probabilities` | Get transition probability matrix |

### Meta-UTL Operations
| Method | Description |
|--------|-------------|
| `meta_utl/learning_trajectory` | Per-embedder learning trajectory |
| `meta_utl/health_metrics` | System health with targets |
| `meta_utl/predict_storage` | Predict storage impact |
| `meta_utl/predict_retrieval` | Predict retrieval quality |
| `meta_utl/validate_prediction` | Validate prediction against outcome |
| `meta_utl/optimized_weights` | Get meta-learned optimized weights |

### GWT/Consciousness Operations
| Method | Description |
|--------|-------------|
| `gwt/kuramoto_status` | Kuramoto network status |
| `gwt/consciousness_level` | Consciousness level and metrics |
| `gwt/workspace_status` | Workspace status and active memory |
| `gwt/state_status` | State machine status |
| `gwt/meta_cognitive_status` | Meta-cognitive loop status |
| `gwt/self_ego_status` | Self-ego node status |
| `consciousness/get_state` | Full consciousness state |
| `consciousness/sync_level` | Lightweight sync level for health checks |

---

## Weight Profiles

Predefined profiles for multi-embedding search (13 weights each):

### `semantic_search`
Heavy E1 (Semantic), moderate E7 (Code), E5 (Causal)
```
[0.28, 0.05, 0.05, 0.05, 0.10, 0.04, 0.18, 0.05, 0.05, 0.05, 0.03, 0.05, 0.02]
```

### `causal_reasoning`
Heavy E5 (Causal), moderate E1, E8 (Graph)
```
[0.15, 0.03, 0.03, 0.03, 0.40, 0.03, 0.10, 0.08, 0.03, 0.05, 0.03, 0.02, 0.02]
```

### `code_search`
Heavy E7 (Code), E4 (Positional), E1
```
[0.15, 0.02, 0.02, 0.15, 0.05, 0.05, 0.35, 0.02, 0.02, 0.05, 0.05, 0.05, 0.02]
```

### `temporal_navigation`
Heavy E2, E3, E4 (all temporal)
```
[0.12, 0.22, 0.22, 0.22, 0.03, 0.02, 0.03, 0.02, 0.03, 0.03, 0.02, 0.02, 0.02]
```

### `fact_checking`
Heavy E11 (Entity), E5 (Causal), E6 (Sparse)
```
[0.10, 0.02, 0.02, 0.02, 0.18, 0.10, 0.05, 0.05, 0.02, 0.05, 0.35, 0.02, 0.02]
```

### `balanced`
Equal weights across all 13 spaces
```
[0.077, 0.077, 0.077, 0.077, 0.077, 0.077, 0.077, 0.077, 0.077, 0.077, 0.077, 0.077, 0.076]
```

---

## Cognitive Pulse System

**Every MCP response includes a `X-Cognitive-Pulse` header** with live UTL metrics for real-time cognitive state visibility.

### Response Structure

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": { ... },
  "X-Cognitive-Pulse": {
    "entropy": 0.42,
    "coherence": 0.78,
    "learning_score": 0.55,
    "quadrant": "Open",
    "suggested_action": "direct_recall"
  }
}
```

### Pulse Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `entropy` | float | [0, 1] | Novelty/surprise level (ΔS) |
| `coherence` | float | [0, 1] | Understanding/integration level (ΔC) |
| `learning_score` | float | [0, 1] | UTL learning magnitude |
| `quadrant` | string | - | Johari quadrant classification |
| `suggested_action` | string | - | Recommended next action |

### Johari Quadrant → Action Mapping

| Entropy | Coherence | Quadrant | Suggested Action |
|---------|-----------|----------|------------------|
| < 0.5 | > 0.5 | Open | `direct_recall` |
| > 0.5 | < 0.5 | Blind | `trigger_dream` |
| < 0.5 | < 0.5 | Hidden | `get_neighborhood` |
| > 0.5 | > 0.5 | Unknown | `epistemic_action` |

### Agent Response Pattern

```
1. Call any MCP tool
2. Check X-Cognitive-Pulse in response
3. If suggested_action != current plan:
   - Consider triggering the suggested action
   - High entropy (>0.7) for 5+ min → trigger_dream
   - Low coherence (<0.4) → get_neighborhood or epistemic_action
4. Continue with task
```

---

## Global Workspace Theory (GWT)

The GWT system implements computational consciousness:

### Components

1. **Kuramoto Network** (13 oscillators)
   - Each oscillator corresponds to one embedder
   - Order parameter r ∈ [0,1] measures synchronization
   - Target: r > 0.8 for coherent consciousness
   - **Background Stepper**: 100Hz tokio task for continuous phase updates

2. **Global Workspace** (Winner-Take-All)
   - Memories compete for workspace access
   - Winner broadcasts to all modules
   - Coherence threshold gates access
   - **Event Listeners**: Wire events to subsystems (see below)

3. **Self-Ego Node**
   - 13D purpose vector represents identity
   - Trajectory tracking for continuity
   - Coherence with actions measured
   - **RocksDB Persistence**: Identity survives restarts

4. **Meta-Cognitive Loop**
   - Monitors system state
   - Evaluates consciousness quality
   - Triggers corrective actions

### Kuramoto Background Stepper

A tokio background task continuously steps the Kuramoto oscillator network at 100Hz (10ms interval) to enable temporal dynamics for consciousness emergence:

```rust
KuramotoStepperConfig {
    step_interval_ms: 10  // 100Hz satisfies Nyquist for all brain wave frequencies
}
```

Without continuous stepping:
- Order parameter `r` remains static
- Consciousness emergence via `C(t) = I(t) × R(t) × D(t)` is impossible
- The system appears frozen in time

### Workspace Event Listeners

Three listeners wire workspace events to subsystems:

| Listener | Event | Action |
|----------|-------|--------|
| **DreamEventListener** | `MemoryExits` (r < 0.7) | Queue memory for dream replay |
| **NeuromodulationEventListener** | `MemoryEnters` (r > 0.8) | Boost dopamine by 0.2 |
| **MetaCognitiveEventListener** | `WorkspaceEmpty` | Trigger epistemic action |

These implement constitution.yaml requirements:
- `neuromod.Dopamine.trigger: "memory_enters_workspace"`
- `gwt.workspace_events: memory_exits → dream replay, workspace_empty → epistemic action`

### Consciousness Level Formula

```
C(t) = r(t) × Φ(t) × M(t)
```

Where:
- r(t) = Kuramoto order parameter
- Φ(t) = Information integration (IIT-inspired)
- M(t) = Meta-cognitive evaluation

---

## Johari Window Classification

Each memory has a Johari quadrant classification **per embedder** (13 classifications total):

| Quadrant | Description | Characteristics |
|----------|-------------|-----------------|
| **Open** | Known to self and others | High confidence, well-integrated |
| **Hidden** | Known to self, not others | Private knowledge, potential to share |
| **Blind** | Known to others, not self | Needs attention, potential insight |
| **Unknown** | Unknown to both | Novel territory, exploration needed |

### Valid Transitions
- **Open ↔ Hidden**: via explicit sharing/concealing
- **Blind → Open**: via feedback acceptance
- **Unknown → Any**: via discovery

---

## Dream Consolidation System

Sleep-inspired memory consolidation:

### Phases

1. **NREM Phase** (3 minutes)
   - Replay and strengthen connections
   - Consolidate similar memories
   - Prune weak connections

2. **REM Phase** (2 minutes)
   - Create novel associations
   - Cross-domain linking
   - Amortized shortcut learning

### Constitution Mandates

- Activity threshold: < 0.15 to start
- GPU usage: < 30%
- Wake latency: < 100ms on external query
- Automatic abort on incoming requests

---

## Autonomous North Star System

The system discovers and maintains purpose autonomously from teleological embeddings:

### Services

1. **Bootstrap Service**: Discovers purpose from stored 13-embedder fingerprints
2. **Drift Detector**: Monitors alignment deviation
3. **Drift Corrector**: Applies correction strategies
4. **Pruning Service**: Identifies stale/misaligned memories
5. **Consolidation Service**: Merges similar memories
6. **Sub-goal Discovery**: Finds emergent themes from clusters

### Why Manual North Star Was Removed

Manual North Star tools created single 1024D embeddings that cannot be meaningfully compared to 13-embedder teleological arrays:
- **Manual**: ONE vector (1024D from text-embedding-3-large)
- **Teleological**: 13 DIFFERENT embeddings with different dimensions (384D to 30K sparse)

The autonomous system works entirely within the teleological space for apples-to-apples comparisons.

---

## Error Codes Reference

### Standard JSON-RPC Errors

| Code | Name | Description |
|------|------|-------------|
| -32700 | PARSE_ERROR | Invalid JSON |
| -32600 | INVALID_REQUEST | Invalid request structure |
| -32601 | METHOD_NOT_FOUND | Unknown method |
| -32602 | INVALID_PARAMS | Invalid parameters |
| -32603 | INTERNAL_ERROR | Server error |

### Context Graph Errors (-32001 to -32009)

| Code | Name | Description |
|------|------|-------------|
| -32001 | FEATURE_DISABLED | Feature not enabled |
| -32002 | NODE_NOT_FOUND | Node doesn't exist |
| -32003 | PAYLOAD_TOO_LARGE | Request too large |
| -32004 | STORAGE_ERROR | Storage operation failed |
| -32005 | EMBEDDING_ERROR | Embedding computation failed |
| -32006 | TOOL_NOT_FOUND | Tool doesn't exist |
| -32007 | LAYER_TIMEOUT | Layer operation timed out |
| -32008 | INDEX_ERROR | HNSW/index operation failed |
| -32009 | GPU_ERROR | GPU/CUDA operation failed |

### Teleological Errors (-32010 to -32019)

| Code | Name | Description |
|------|------|-------------|
| -32010 | FINGERPRINT_NOT_FOUND | Teleological fingerprint not found |
| -32011 | EMBEDDER_NOT_READY | 13-embedder provider not ready |
| -32012 | PURPOSE_COMPUTATION_ERROR | Purpose vector computation failed |
| -32013 | JOHARI_CLASSIFICATION_ERROR | Johari classification failed |
| -32014 | SPARSE_SEARCH_ERROR | SPLADE search failed |
| -32015 | SEMANTIC_SEARCH_ERROR | Semantic search failed |
| -32016 | PURPOSE_SEARCH_ERROR | Purpose alignment search failed |
| -32017 | CHECKPOINT_ERROR | Checkpoint/restore failed |
| -32018 | BATCH_OPERATION_ERROR | Batch operation failed |
| -32019 | TOOL_NOT_IMPLEMENTED | Tool not implemented (FAIL FAST) |

### Goal/Alignment Errors (-32020 to -32029)

| Code | Name | Description |
|------|------|-------------|
| -32020 | GOAL_NOT_FOUND | Goal not in hierarchy |
| -32022 | ALIGNMENT_COMPUTATION_ERROR | Alignment computation failed |
| -32023 | GOAL_HIERARCHY_ERROR | Goal hierarchy operation failed |

### Johari Errors (-32030 to -32039)

| Code | Name | Description |
|------|------|-------------|
| -32030 | JOHARI_INVALID_EMBEDDER_INDEX | Index must be 0-12 |
| -32031 | JOHARI_INVALID_QUADRANT | Invalid quadrant string |
| -32032 | JOHARI_INVALID_SOFT_CLASSIFICATION | Weights don't sum to 1.0 |
| -32033 | JOHARI_TRANSITION_ERROR | Transition validation failed |
| -32034 | JOHARI_BATCH_ERROR | Batch transition failed |

### Meta-UTL Errors (-32040 to -32049)

| Code | Name | Description |
|------|------|-------------|
| -32040 | META_UTL_PREDICTION_NOT_FOUND | Prediction not found |
| -32041 | META_UTL_NOT_INITIALIZED | Meta-UTL not initialized |
| -32042 | META_UTL_INSUFFICIENT_DATA | Not enough data for prediction |
| -32043 | META_UTL_INVALID_OUTCOME | Invalid outcome format |
| -32044 | META_UTL_TRAJECTORY_ERROR | Trajectory computation failed |
| -32045 | META_UTL_HEALTH_ERROR | Health metrics failed |

### GWT/Kuramoto Errors (-32060 to -32069)

| Code | Name | Description |
|------|------|-------------|
| -32060 | GWT_NOT_INITIALIZED | GWT system not initialized |
| -32061 | KURAMOTO_ERROR | Kuramoto network error |
| -32062 | CONSCIOUSNESS_COMPUTATION_FAILED | Consciousness computation failed |
| -32063 | WORKSPACE_ERROR | Workspace operation failed |
| -32064 | STATE_TRANSITION_ERROR | State machine transition failed |
| -32065 | META_COGNITIVE_ERROR | Meta-cognitive evaluation failed |
| -32066 | SELF_EGO_ERROR | Self-ego node operation failed |
| -32067 | IDENTITY_CONTINUITY_ERROR | Identity continuity check failed |

### Dream Errors (-32070 to -32079)

| Code | Name | Description |
|------|------|-------------|
| -32070 | DREAM_NOT_INITIALIZED | Dream controller not initialized |
| -32071 | DREAM_CYCLE_ERROR | Dream cycle start/trigger failed |
| -32072 | DREAM_ABORT_ERROR | Dream abort failed |
| -32073 | AMORTIZED_LEARNING_ERROR | Amortized learning error |

### Neuromodulation Errors (-32080 to -32089)

| Code | Name | Description |
|------|------|-------------|
| -32080 | NEUROMOD_NOT_INITIALIZED | Neuromod manager not initialized |
| -32081 | NEUROMOD_ADJUSTMENT_ERROR | Adjustment failed |
| -32082 | NEUROMOD_ACH_READ_ONLY | Acetylcholine is read-only |

### Steering Errors (-32090 to -32099)

| Code | Name | Description |
|------|------|-------------|
| -32090 | STEERING_NOT_INITIALIZED | Steering system not initialized |
| -32091 | STEERING_FEEDBACK_ERROR | Feedback computation failed |
| -32092 | GARDENER_ERROR | Gardener component error |
| -32093 | CURATOR_ERROR | Curator component error |
| -32094 | ASSESSOR_ERROR | Assessor component error |

### Causal Errors (-32100 to -32109)

| Code | Name | Description |
|------|------|-------------|
| -32100 | CAUSAL_NOT_INITIALIZED | Causal engine not initialized |
| -32101 | CAUSAL_INVALID_DIRECTION | Invalid inference direction |
| -32102 | CAUSAL_INFERENCE_ERROR | Inference failed |
| -32103 | CAUSAL_TARGET_REQUIRED | Target node required |
| -32104 | CAUSAL_GRAPH_ERROR | Graph operation failed |

### TCP Transport Errors (-32110 to -32119)

| Code | Name | Description |
|------|------|-------------|
| -32110 | TCP_BIND_FAILED | Address/port unavailable |
| -32111 | TCP_CONNECTION_ERROR | Stream read/write failed |
| -32112 | TCP_MAX_CONNECTIONS_REACHED | Connection limit reached |
| -32113 | TCP_FRAME_ERROR | Invalid NDJSON framing |
| -32114 | TCP_CLIENT_TIMEOUT | Request processing timeout |

---

## Claude Code Integration

### MCP Server Configuration

```bash
# Add to Claude Code
claude mcp add context-graph -- cargo run --manifest-path /path/to/contextgraph/crates/context-graph-mcp/Cargo.toml
```

### Hook Integration Patterns

#### Pre-Task Hook (Inject Relevant Context)

```yaml
# .claude/hooks/pre-task.yaml
hooks:
  pre-task:
    - name: context-inject
      command: |
        STATE=$(mcp call context-graph get_memetic_status)
        CONTEXT=$(mcp call context-graph search_graph '{"query": "$TASK_DESCRIPTION", "topK": 5}')
        echo "$CONTEXT"
```

#### Post-Edit Hook (Store Code Changes)

```yaml
# .claude/hooks/post-edit.yaml
hooks:
  post-edit:
    - name: context-store
      command: |
        mcp call context-graph inject_context '{
          "content": "'"$FILE_CHANGES"'",
          "rationale": "Code changes from edit to '"$FILE_PATH"'",
          "modality": "code",
          "importance": 0.7
        }'
```

#### Session End Hook (Dream Consolidation)

```yaml
# .claude/hooks/session-end.yaml
hooks:
  session-end:
    - name: context-consolidate
      command: |
        STATUS=$(mcp call context-graph get_memetic_status)
        ENTROPY=$(echo "$STATUS" | jq -r '.utl.entropy')
        if (( $(echo "$ENTROPY > 0.7" | bc -l) )); then
          mcp call context-graph trigger_dream '{"force": false}'
        fi
```

### Autonomous Operation Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS CONTEXT FLOW                          │
├─────────────────────────────────────────────────────────────────────┤
│  1. Task Start                                                      │
│     └─ pre-task hook → get_memetic_status → check suggested_action │
│     └─ search_graph → inject relevant 13-embedding fingerprints    │
│                                                                     │
│  2. During Work                                                     │
│     └─ Every tool response includes X-Cognitive-Pulse              │
│     └─ Monitor entropy/coherence for drift                         │
│     └─ post-edit hook → inject_context with code changes           │
│                                                                     │
│  3. Periodic Check                                                  │
│     └─ get_consciousness_state → check C(t) level                  │
│     └─ get_alignment_drift → detect goal misalignment              │
│     └─ If drift detected → trigger_drift_correction                │
│                                                                     │
│  4. Session End                                                     │
│     └─ session-end hook → trigger_dream if entropy > 0.7           │
│     └─ trigger_consolidation for memory hygiene                    │
│     └─ discover_sub_goals from session learnings                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Performance Targets

| Operation | Target Latency |
|-----------|----------------|
| Single Embed (all 13) | < 35ms |
| inject_context P95 | < 40ms |
| search_graph P95 | < 30ms |
| Any tool P99 | < 60ms |
| Cognitive Pulse | < 1ms |
| Dream wake | < 100ms |
| TCP connection setup | < 10ms |
| Maximum concurrent connections | Configurable (default: 100) |
| **HNSW Search** | O(log n) via usearch graph traversal |
| **MaxSim (50×50 tokens)** | < 300μs with SIMD |
| **Kuramoto Step** | 10ms interval (100Hz) |
| **Workspace Event Dispatch** | Non-blocking (try_write) |

---

## Summary

The Context Graph MCP server provides:

- **35 tools** across 10 categories
- **40+ JSON-RPC methods** for direct access
- **13 specialized embedders** forming teleological fingerprints
- **5-layer bio-nervous architecture**
- **GWT consciousness system** with Kuramoto synchronization
  - 100Hz background stepper for continuous phase evolution
  - Workspace event listeners for subsystem wiring
  - Persistent Self-Ego Node (RocksDB backed)
- **Dream consolidation** for memory hygiene
- **Autonomous goal alignment** with drift detection
- **Cognitive Pulse** in every response for adaptive behavior
- **O(log n) HNSW search** via usearch graph traversal
- **MaxSim Stage 5** with SIMD acceleration for E12 late interaction
- **Per-embedder ΔS entropy** with specialized methods per embedding type

The system is designed for seamless integration with Claude Code via MCP protocol, enabling autonomous context management that learns and adapts over time.

---

*Generated from source code analysis. Last updated: 2026-01-10*

*Key source files: crates/context-graph-mcp/src/tools.rs, protocol.rs, weights.rs, handlers/kuramoto_stepper.rs; crates/context-graph-core/src/gwt/listeners.rs, ego_node.rs; crates/context-graph-storage/src/teleological/search/maxsim.rs, indexes/hnsw_impl.rs; crates/context-graph-utl/src/surprise/embedder_entropy/*
