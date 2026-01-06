# TASK-S003: New MCP Handlers for Purpose/Goal Operations

```yaml
metadata:
  id: "TASK-S003"
  title: "New MCP Handlers for Purpose/Goal Operations"
  layer: "surface"
  priority: "P0"
  estimated_hours: 10
  created: "2026-01-04"
  updated: "2026-01-05"
  status: "pending"
  dependencies:
    - "TASK-L002"  # Purpose Vector Computation - COMPLETE
    - "TASK-L003"  # Goal Alignment Calculator - COMPLETE
    - "TASK-L006"  # Purpose Pattern Index - COMPLETE
  traces_to:
    - "FR-201"  # PurposeVector (13D Alignment Signature)
    - "FR-202"  # North Star Alignment Thresholds
    - "FR-303"  # Goal Hierarchy Index
```

## CRITICAL: Read These Files First

Before implementing this task, you MUST read and understand:

1. **constitution.yaml** - Core specification defining 13 embedders, alignment thresholds
2. **contextprd.md** - Product requirements document
3. **docs2/projection/specs/tasks/_index.md** - Task hierarchy and dependencies
4. **module-6-mvv/_index.md** - Minimum Viable Version requirements
5. **TASK-S002-mcp-search-handlers.md** - REFERENCE PATTERN for handler implementation

## CRITICAL: Codebase Audit Results

**THE PREVIOUS VERSION OF THIS TASK HAD ERRORS. THIS VERSION CORRECTS THEM.**

### Discrepancy #1: 13 Embedders NOT 12

The previous task specified 12D purpose vectors. **WRONG**. The system uses **13 embedders (E1-E13)**:

| Index | Name | Description |
|-------|------|-------------|
| 0 | E1_Semantic | General semantic similarity |
| 1 | E2_Temporal_Recent | Recent time proximity |
| 2 | E3_Temporal_Periodic | Recurring patterns |
| 3 | E4_Temporal_Positional | Document position encoding |
| 4 | E5_Causal | Cause-effect relationships |
| 5 | E6_Sparse | Keyword-level matching |
| 6 | E7_Code | Source code similarity |
| 7 | E8_Graph | Node2Vec structural |
| 8 | E9_HDC | Hyperdimensional computing |
| 9 | E10_Multimodal | Cross-modal alignment |
| 10 | E11_Entity | Named entity matching |
| 11 | E12_Late_Interaction | ColBERT-style token matching |
| 12 | E13_SPLADE | Sparse learned expansion |

**Source**: `context_graph_core::types::fingerprint::NUM_EMBEDDERS = 13`

### Discrepancy #2: Files That Already Exist (DO NOT CREATE)

These types/modules already exist and MUST be used:

| Type | Location | Purpose |
|------|----------|---------|
| `PurposeVector` | `crates/context-graph-core/src/types/fingerprint/purpose.rs:107` | 13D alignment vector |
| `AlignmentThreshold` | `crates/context-graph-core/src/types/fingerprint/purpose.rs:19` | Optimal/Acceptable/Warning/Critical |
| `AlignmentLevel` | `crates/context-graph-core/src/retrieval/teleological_result.rs:276` | Same classification for results |
| `GoalAlignmentCalculator` | `crates/context-graph-core/src/alignment/calculator.rs:87` | Trait for alignment computation |
| `DefaultAlignmentCalculator` | `crates/context-graph-core/src/alignment/calculator.rs:139` | Default implementation |
| `GoalHierarchy` | `crates/context-graph-core/src/purpose/goals.rs` | Goal tree structure |
| `GoalNode` | `crates/context-graph-core/src/purpose/goals.rs:114` | Individual goal nodes |
| `GoalLevel` | `crates/context-graph-core/src/purpose/goals.rs:64` | NorthStar/Strategic/Tactical/Immediate |
| `GoalId` | `crates/context-graph-core/src/purpose/goals.rs:21` | Goal identifier type |
| `TeleologicalMemoryStore` | `crates/context-graph-core/src/traits/teleological_memory_store.rs:233` | Storage trait |

**DO NOT RECREATE THESE TYPES. USE THEM.**

### Discrepancy #3: No router.rs File

The previous task referenced `router.rs`. **WRONG**. Handler dispatch is in `core.rs:51-95`.

### Current Handler Structure

```
crates/context-graph-mcp/src/handlers/
├── mod.rs          # Module exports (add purpose.rs here)
├── core.rs         # Handlers struct with dispatch() method (lines 51-95)
├── lifecycle.rs    # initialize, shutdown
├── memory.rs       # TASK-S001 memory handlers
├── search.rs       # TASK-S002 search handlers (REFERENCE PATTERN)
├── system.rs       # status, health
├── tools.rs        # tools/list, tools/call
├── utl.rs          # UTL compute handlers
└── tests/          # Test modules (add tests/purpose.rs)
```

### Current Handlers Struct (core.rs:18-29)

```rust
pub struct Handlers {
    /// TeleologicalMemoryStore - NOT legacy MemoryStore
    pub(super) teleological_store: Arc<dyn TeleologicalMemoryStore>,

    /// UTL processor for learning metrics
    pub(super) utl_processor: Arc<dyn UtlProcessor>,

    /// MultiArrayEmbeddingProvider - 13 embeddings per content
    pub(super) multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
}
```

**NOTE**: You will need to ADD a `GoalAlignmentCalculator` field to this struct.

### Existing Error Codes (protocol.rs:87-123)

Use these EXISTING error codes:

```rust
pub const INVALID_PARAMS: i32 = -32602;
pub const FINGERPRINT_NOT_FOUND: i32 = -32010;
pub const PURPOSE_COMPUTATION_ERROR: i32 = -32012;
pub const PURPOSE_SEARCH_ERROR: i32 = -32016;
```

You will need to ADD these new codes to `protocol.rs`:

```rust
// Goal/alignment specific error codes (-32020 to -32029)
pub const GOAL_NOT_FOUND: i32 = -32020;
pub const NORTH_STAR_NOT_CONFIGURED: i32 = -32021;
pub const ALIGNMENT_COMPUTATION_ERROR: i32 = -32022;
pub const GOAL_HIERARCHY_ERROR: i32 = -32023;
```

## Problem Statement

Create new MCP handlers for teleological purpose and goal operations including:
1. Purpose vector queries (find memories with similar 13D purpose signatures)
2. North Star alignment checking with threshold classification
3. Goal hierarchy navigation (traverse and query the goal tree)
4. Alignment drift detection (identify memories becoming misaligned)

**These are NEW handlers - no legacy equivalents exist.**

## Technical Specification

### NO BACKWARDS COMPATIBILITY

This implementation:
- Uses `TeleologicalMemoryStore` (NOT legacy MemoryStore)
- Uses 13 embedding spaces (NOT 12)
- Uses existing `GoalAlignmentCalculator` trait
- Uses existing `PurposeVector` and `AlignmentThreshold` types
- FAILS FAST on any error with robust logging
- NO MOCK DATA in tests

### MCP Method Registration

Add these methods to `protocol.rs` `methods` module:

```rust
// Purpose/goal operations (TASK-S003)
pub const PURPOSE_QUERY: &str = "purpose/query";
pub const PURPOSE_NORTH_STAR_ALIGNMENT: &str = "purpose/north_star_alignment";
pub const GOAL_HIERARCHY_QUERY: &str = "goal/hierarchy_query";
pub const GOAL_ALIGNED_MEMORIES: &str = "goal/aligned_memories";
pub const PURPOSE_DRIFT_CHECK: &str = "purpose/drift_check";
pub const NORTH_STAR_UPDATE: &str = "purpose/north_star_update";
```

### MCP Handler Function Signatures

```rust
// File: crates/context-graph-mcp/src/handlers/purpose.rs

use std::sync::Arc;
use tracing::{debug, error, instrument, warn};

use context_graph_core::alignment::{
    AlignmentConfig, AlignmentResult, GoalAlignmentCalculator,
};
use context_graph_core::purpose::{GoalHierarchy, GoalId, GoalLevel, GoalNode};
use context_graph_core::traits::TeleologicalMemoryStore;
use context_graph_core::types::fingerprint::{
    AlignmentThreshold, PurposeVector, NUM_EMBEDDERS,  // NUM_EMBEDDERS = 13
};

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

impl Handlers {
    /// Query memories by purpose vector similarity.
    ///
    /// # Request Parameters
    /// - `purpose_vector` (required): 13-element array of alignment values [-1.0, 1.0]
    /// - `top_k` (optional): Maximum results, default 10
    /// - `min_similarity` (optional): Minimum purpose similarity threshold
    /// - `filter_by_dominant_embedder` (optional): Only return memories where this embedder (0-12) is dominant
    ///
    /// # Response
    /// - `results`: Array of memories with purpose similarity scores
    /// - `query_time_ms`: Search latency
    ///
    /// # Error Codes
    /// - INVALID_PARAMS (-32602): Invalid purpose vector (not 13 elements, values out of range)
    /// - PURPOSE_SEARCH_ERROR (-32016): Purpose search failed
    #[instrument(skip(self, params), fields(method = "purpose/query"))]
    pub(super) async fn handle_purpose_query(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        // Implementation here
    }

    /// Check alignment to North Star for multiple memories.
    ///
    /// # Request Parameters
    /// - `memory_ids` (required): Array of UUIDs to check (1-100)
    /// - `include_per_space_alignment` (optional): Include 13-element breakdown, default false
    /// - `include_threshold_classification` (optional): Include Optimal/Acceptable/Warning/Critical, default true
    ///
    /// # Response
    /// - `alignments`: Array of alignment results per memory
    /// - `summary`: Aggregate statistics (optimal_count, warning_count, etc.)
    ///
    /// # Error Codes
    /// - INVALID_PARAMS (-32602): Invalid memory IDs
    /// - FINGERPRINT_NOT_FOUND (-32010): Memory not found
    /// - NORTH_STAR_NOT_CONFIGURED (-32021): No North Star goal set
    /// - ALIGNMENT_COMPUTATION_ERROR (-32022): Computation failed
    #[instrument(skip(self, params), fields(method = "purpose/north_star_alignment"))]
    pub(super) async fn handle_north_star_alignment(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        // Implementation here
    }

    /// Navigate goal hierarchy.
    ///
    /// # Request Parameters
    /// - `goal_id` (optional): Goal UUID (null for North Star root)
    /// - `operation` (required): "get_children", "get_ancestors", "get_subtree", "get_aligned_memories"
    /// - `depth` (optional): Depth for subtree operation, default 1, max 4
    /// - `include_alignment_stats` (optional): Include memory alignment statistics, default false
    ///
    /// # Response
    /// - `goal`: The requested goal node
    /// - `operation`: Operation performed
    /// - `children` / `ancestors`: Resulting goal array (operation-dependent)
    /// - `alignment_stats`: Optional statistics
    ///
    /// # Error Codes
    /// - INVALID_PARAMS (-32602): Invalid goal_id or operation
    /// - GOAL_NOT_FOUND (-32020): Goal does not exist
    /// - GOAL_HIERARCHY_ERROR (-32023): Hierarchy traversal failed
    #[instrument(skip(self, params), fields(method = "goal/hierarchy_query"))]
    pub(super) async fn handle_goal_hierarchy_query(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        // Implementation here
    }

    /// Find memories aligned to a specific goal.
    ///
    /// # Request Parameters
    /// - `goal_id` (required): Goal UUID
    /// - `min_alignment` (optional): Minimum alignment threshold, default 0.55
    /// - `top_k` (optional): Maximum results, default 10
    /// - `include_classification` (optional): Include threshold classification, default true
    ///
    /// # Response
    /// - `results`: Array of aligned memories with scores
    /// - `goal`: The goal being queried
    ///
    /// # Error Codes
    /// - GOAL_NOT_FOUND (-32020): Goal does not exist
    /// - PURPOSE_SEARCH_ERROR (-32016): Search failed
    #[instrument(skip(self, params), fields(method = "goal/aligned_memories"))]
    pub(super) async fn handle_find_aligned_to_goal(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        // Implementation here
    }

    /// Detect alignment drift in memories.
    ///
    /// # Request Parameters
    /// - `memory_ids` (optional): Specific memories to check (null for all with history)
    /// - `drift_threshold` (optional): Alignment delta threshold, default -0.15
    /// - `time_window_hours` (optional): Look back window, default 24
    ///
    /// # Response
    /// - `drifting_memories`: Array of memories showing drift
    /// - `summary`: total_checked, drifting_count, average_drift, worst_drift
    ///
    /// # Error Codes
    /// - INVALID_PARAMS (-32602): Invalid parameters
    /// - PURPOSE_COMPUTATION_ERROR (-32012): Drift computation failed
    #[instrument(skip(self, params), fields(method = "purpose/drift_check"))]
    pub(super) async fn handle_alignment_drift_check(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        // Implementation here
    }

    /// Update the North Star goal.
    ///
    /// # Request Parameters
    /// - `description` (required): New goal description
    /// - `embedding` (optional): 1024D embedding (will be generated if not provided)
    /// - `keywords` (optional): Keywords for SPLADE matching
    ///
    /// # Response
    /// - `goal`: Updated North Star goal
    /// - `affected_memories`: Count of memories that need realignment
    ///
    /// # Error Codes
    /// - INVALID_PARAMS (-32602): Invalid parameters
    /// - GOAL_HIERARCHY_ERROR (-32023): Update failed
    #[instrument(skip(self, params), fields(method = "purpose/north_star_update"))]
    pub(super) async fn handle_north_star_update(
        &self,
        id: Option<JsonRpcId>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        // Implementation here
    }
}
```

### Request/Response JSON Schemas (CORRECTED FOR 13 SPACES)

#### PurposeQueryRequest

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["purpose_vector"],
  "properties": {
    "purpose_vector": {
      "type": "array",
      "items": { "type": "number", "minimum": -1.0, "maximum": 1.0 },
      "minItems": 13,
      "maxItems": 13,
      "description": "13D purpose vector (alignment per embedder to North Star)"
    },
    "top_k": {
      "type": "integer",
      "minimum": 1,
      "maximum": 1000,
      "default": 10
    },
    "min_similarity": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "default": 0.5
    },
    "filter_by_dominant_embedder": {
      "type": "integer",
      "minimum": 0,
      "maximum": 12,
      "description": "Only return memories where this embedder (0-12) is dominant"
    }
  }
}
```

#### PurposeQueryResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["results", "query_time_ms"],
  "properties": {
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "purpose_similarity", "purpose_vector"],
        "properties": {
          "id": { "type": "string", "format": "uuid" },
          "purpose_similarity": { "type": "number" },
          "purpose_vector": {
            "type": "object",
            "properties": {
              "alignments": {
                "type": "array",
                "items": { "type": "number" },
                "minItems": 13,
                "maxItems": 13
              },
              "dominant_embedder": { "type": "integer", "minimum": 0, "maximum": 12 },
              "coherence": { "type": "number" }
            }
          },
          "theta_to_north_star": { "type": "number" },
          "threshold_classification": {
            "type": "string",
            "enum": ["optimal", "acceptable", "warning", "critical"]
          }
        }
      }
    },
    "query_time_ms": { "type": "number" }
  }
}
```

#### NorthStarAlignmentRequest

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["memory_ids"],
  "properties": {
    "memory_ids": {
      "type": "array",
      "items": { "type": "string", "format": "uuid" },
      "minItems": 1,
      "maxItems": 100
    },
    "include_per_space_alignment": {
      "type": "boolean",
      "default": false,
      "description": "Include 13-element alignment breakdown per embedding space"
    },
    "include_threshold_classification": {
      "type": "boolean",
      "default": true
    }
  }
}
```

#### NorthStarAlignmentResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["alignments", "summary"],
  "properties": {
    "alignments": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "theta", "classification"],
        "properties": {
          "id": { "type": "string", "format": "uuid" },
          "theta": {
            "type": "number",
            "description": "Aggregate alignment to North Star [0, 1]"
          },
          "classification": {
            "type": "string",
            "enum": ["optimal", "acceptable", "warning", "critical"]
          },
          "per_space_alignment": {
            "type": "array",
            "items": { "type": "number" },
            "minItems": 13,
            "maxItems": 13,
            "description": "Alignment per embedding space (if requested)"
          },
          "dominant_space": {
            "type": "integer",
            "minimum": 0,
            "maximum": 12
          },
          "is_misaligned": { "type": "boolean" }
        }
      }
    },
    "summary": {
      "type": "object",
      "properties": {
        "optimal_count": { "type": "integer" },
        "acceptable_count": { "type": "integer" },
        "warning_count": { "type": "integer" },
        "critical_count": { "type": "integer" },
        "average_theta": { "type": "number" }
      }
    }
  }
}
```

#### GoalHierarchyRequest

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["operation"],
  "properties": {
    "goal_id": {
      "type": "string",
      "description": "Goal ID (null/omitted for North Star root)"
    },
    "operation": {
      "type": "string",
      "enum": ["get_children", "get_ancestors", "get_subtree", "get_aligned_memories"]
    },
    "depth": {
      "type": "integer",
      "minimum": 1,
      "maximum": 4,
      "default": 1,
      "description": "Depth for subtree operation (4 levels: NorthStar, Strategic, Tactical, Immediate)"
    },
    "include_alignment_stats": {
      "type": "boolean",
      "default": false
    }
  }
}
```

#### GoalHierarchyResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["goal", "operation"],
  "properties": {
    "goal": {
      "type": "object",
      "required": ["id", "level", "description"],
      "properties": {
        "id": { "type": "string" },
        "description": { "type": "string" },
        "level": {
          "type": "string",
          "enum": ["north_star", "strategic", "tactical", "immediate"]
        },
        "level_depth": {
          "type": "integer",
          "description": "0=north_star, 1=strategic, 2=tactical, 3=immediate"
        },
        "parent_id": { "type": "string" },
        "weight": { "type": "number", "minimum": 0, "maximum": 1 },
        "propagation_weight": {
          "type": "number",
          "description": "1.0 for NorthStar, 0.7 for Strategic, 0.4 for Tactical, 0.2 for Immediate"
        }
      }
    },
    "operation": { "type": "string" },
    "children": {
      "type": "array",
      "items": { "$ref": "#/definitions/Goal" }
    },
    "ancestors": {
      "type": "array",
      "items": { "$ref": "#/definitions/Goal" }
    },
    "alignment_stats": {
      "type": "object",
      "properties": {
        "aligned_memory_count": { "type": "integer" },
        "average_alignment": { "type": "number" },
        "min_alignment": { "type": "number" },
        "max_alignment": { "type": "number" },
        "optimal_percentage": { "type": "number" }
      }
    }
  }
}
```

#### AlignmentDriftRequest

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "memory_ids": {
      "type": "array",
      "items": { "type": "string", "format": "uuid" },
      "description": "Specific memories to check (null/omitted for all with evolution history)"
    },
    "drift_threshold": {
      "type": "number",
      "default": -0.15,
      "description": "Alignment delta threshold for drift warning (negative = degradation)"
    },
    "time_window_hours": {
      "type": "integer",
      "minimum": 1,
      "maximum": 720,
      "default": 24
    }
  }
}
```

#### AlignmentDriftResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["drifting_memories", "summary"],
  "properties": {
    "drifting_memories": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "current_theta", "previous_theta", "delta"],
        "properties": {
          "id": { "type": "string", "format": "uuid" },
          "current_theta": { "type": "number" },
          "previous_theta": { "type": "number" },
          "delta": { "type": "number", "description": "Negative = degradation" },
          "current_classification": {
            "type": "string",
            "enum": ["optimal", "acceptable", "warning", "critical"]
          },
          "previous_classification": {
            "type": "string",
            "enum": ["optimal", "acceptable", "warning", "critical"]
          },
          "time_since_previous_hours": { "type": "number" },
          "per_space_deltas": {
            "type": "array",
            "items": { "type": "number" },
            "minItems": 13,
            "maxItems": 13
          }
        }
      }
    },
    "summary": {
      "type": "object",
      "properties": {
        "total_checked": { "type": "integer" },
        "drifting_count": { "type": "integer" },
        "average_drift": { "type": "number" },
        "worst_drift": { "type": "number" },
        "memories_needing_intervention": { "type": "integer" }
      }
    }
  }
}
```

### Alignment Threshold Classification

From `purpose.rs:19-28` and `teleological_result.rs:276-294`:

```rust
// Thresholds from constitution.yaml (Royse 2026 research)
fn classify_theta(theta: f32) -> &'static str {
    if theta >= 0.75 {
        "optimal"      // Strong alignment
    } else if theta >= 0.70 {
        "acceptable"   // Good, monitor for drift
    } else if theta >= 0.55 {
        "warning"      // Degrading, intervention recommended
    } else {
        "critical"     // Immediate action required
    }
}
```

### Goal Level Propagation Weights

From `goals.rs:87-99`:

| Level | Propagation Weight | Depth |
|-------|-------------------|-------|
| NorthStar | 1.0 | 0 |
| Strategic | 0.7 | 1 |
| Tactical | 0.4 | 2 |
| Immediate | 0.2 | 3 |

## Implementation Requirements

### Prerequisites (ALL COMPLETED)

- [x] TASK-L002 complete (Purpose Vector Computation) - `crates/context-graph-core/src/purpose/`
- [x] TASK-L003 complete (Goal Alignment Calculator) - `crates/context-graph-core/src/alignment/`
- [x] TASK-L006 complete (Purpose Pattern Index) - `crates/context-graph-core/src/index/purpose/`

### Scope

#### In Scope

- Purpose vector similarity search (13D)
- North Star alignment checking with threshold classification
- Goal hierarchy navigation (children, ancestors, subtree)
- Alignment drift detection
- North Star update operation
- Find memories aligned to specific goals

#### Out of Scope

- Semantic content search (TASK-S002 - COMPLETE)
- Johari operations (TASK-S004 - separate task)
- Memory store/retrieve (TASK-S001 - COMPLETE)

### Constraints

- Purpose vector MUST be exactly 13D (NUM_EMBEDDERS = 13)
- Alignment values in [-1.0, 1.0]
- Goal hierarchy limited to 4 levels (NorthStar, Strategic, Tactical, Immediate)
- Drift detection requires `purpose_evolution` history in TeleologicalFingerprint
- FAIL FAST on any error - NO fallbacks

## Definition of Done

### Implementation Checklist

- [ ] Add `GoalAlignmentCalculator` to Handlers struct
- [ ] Add new method constants to `protocol.rs methods` module
- [ ] Add new error codes to `protocol.rs error_codes` module
- [ ] `handle_purpose_query` for 13D purpose similarity search
- [ ] `handle_north_star_alignment` with threshold classification
- [ ] `handle_goal_hierarchy_query` with all 4 operations
- [ ] `handle_find_aligned_to_goal` for goal-based retrieval
- [ ] `handle_alignment_drift_check` with configurable threshold
- [ ] `handle_north_star_update` for goal management
- [ ] Add method dispatch cases to `core.rs dispatch()`
- [ ] All error cases with specific error codes

### Testing Requirements

**NO MOCK DATA. REAL DATA ONLY.**

Tests MUST:
1. Use real TeleologicalFingerprint with 13 embeddings
2. Use real GoalAlignmentCalculator (DefaultAlignmentCalculator)
3. Print BEFORE/AFTER state for debugging
4. Log actual database contents after operations
5. Follow patterns from `handlers/tests/search.rs`

```rust
// File: crates/context-graph-mcp/src/handlers/tests/purpose.rs

use super::*;
use context_graph_core::alignment::DefaultAlignmentCalculator;
use context_graph_core::purpose::{GoalHierarchy, GoalLevel, GoalNode};
use context_graph_core::stubs::InMemoryTeleologicalStore;
use context_graph_core::types::fingerprint::{
    AlignmentThreshold, PurposeVector, NUM_EMBEDDERS,
};
use uuid::Uuid;

/// Create test handlers with REAL store and calculator.
fn create_test_handlers() -> Handlers {
    // REAL store, not mock
    let store = Arc::new(InMemoryTeleologicalStore::new());
    let utl = Arc::new(DefaultUtlProcessor::new());
    let provider = Arc::new(DefaultMultiArrayProvider::new());
    let calculator = Arc::new(DefaultAlignmentCalculator::new());

    Handlers::new_with_calculator(store, utl, provider, calculator)
}

/// Create a test goal hierarchy with North Star and sub-goals.
fn create_test_goal_hierarchy() -> GoalHierarchy {
    let north_star = GoalNode::north_star(
        "master_ml",
        "Master machine learning fundamentals",
        vec![0.5; 1024],
        vec!["machine".into(), "learning".into(), "ai".into()],
    );

    let strategic = GoalNode::child(
        "understand_neural_nets",
        "Understand neural network architectures",
        GoalLevel::Strategic,
        "master_ml",
        vec![0.5; 1024],
        0.8,
        vec!["neural".into(), "network".into()],
    );

    let mut hierarchy = GoalHierarchy::new(north_star);
    hierarchy.add_goal(strategic).unwrap();
    hierarchy
}

#[tokio::test]
async fn test_purpose_query_13d_vector() {
    let handlers = create_test_handlers();

    // Create test fingerprint with known purpose vector
    let test_pv = PurposeVector::new([
        0.8, 0.2, 0.3, 0.1, 0.6, 0.1, 0.7, 0.2, 0.1, 0.4, 0.2, 0.3, 0.1
    ]); // 13 elements

    println!("BEFORE: Creating test fingerprint with 13D purpose vector");
    println!("  Purpose vector: {:?}", test_pv.alignments);
    println!("  Dominant embedder: {}", test_pv.dominant_embedder);
    println!("  Coherence: {:.4}", test_pv.coherence);

    // Store test data
    let fp = create_test_fingerprint_with_purpose(test_pv.clone());
    handlers.teleological_store.store(fp).await.unwrap();

    // Query with similar purpose vector
    let query_pv: [f32; NUM_EMBEDDERS] = [
        0.75, 0.25, 0.35, 0.15, 0.55, 0.15, 0.65, 0.25, 0.15, 0.35, 0.25, 0.35, 0.15
    ];

    let request = serde_json::json!({
        "purpose_vector": query_pv,
        "top_k": 5,
        "min_similarity": 0.7
    });

    let response = handlers
        .handle_purpose_query(Some(JsonRpcId::Number(1)), Some(request))
        .await;

    // VERIFY
    assert!(response.error.is_none(), "Purpose query should succeed");
    let result = response.result.unwrap();

    println!("AFTER: Query returned {} results", result["results"].as_array().unwrap().len());

    for res in result["results"].as_array().unwrap() {
        let similarity = res["purpose_similarity"].as_f64().unwrap();
        let alignments = res["purpose_vector"]["alignments"].as_array().unwrap();

        println!("  Result: similarity={:.4}", similarity);
        assert_eq!(alignments.len(), 13, "Purpose vector must have 13 elements");
        assert!(similarity >= 0.7, "Similarity must meet threshold");
    }

    println!("[VERIFIED] Purpose query with 13D vector works correctly");
}

#[tokio::test]
async fn test_purpose_query_invalid_12d_vector_fails() {
    let handlers = create_test_handlers();

    // WRONG: Only 12 elements (legacy format)
    let bad_pv = [0.5, 0.2, 0.3, 0.1, 0.6, 0.1, 0.7, 0.2, 0.1, 0.4, 0.2, 0.3];

    println!("EDGE CASE: Testing 12D vector rejection");
    println!("  Input length: {}", bad_pv.len());

    let request = serde_json::json!({
        "purpose_vector": bad_pv
    });

    let response = handlers
        .handle_purpose_query(Some(JsonRpcId::Number(1)), Some(request))
        .await;

    assert!(response.error.is_some(), "12D vector must fail");
    let error = response.error.unwrap();
    assert_eq!(error.code, error_codes::INVALID_PARAMS);
    assert!(error.message.contains("13"), "Error must mention 13 elements required");

    println!("[VERIFIED] 12D vector rejected with clear error: {}", error.message);
}

#[tokio::test]
async fn test_north_star_alignment_thresholds() {
    let handlers = create_test_handlers();

    // Create memories with varying alignment values
    let test_cases = [
        (0.80, "optimal"),
        (0.72, "acceptable"),
        (0.60, "warning"),
        (0.40, "critical"),
    ];

    let mut memory_ids = Vec::new();
    for (theta, _expected_class) in &test_cases {
        let fp = create_fingerprint_with_theta(*theta);
        let id = handlers.teleological_store.store(fp).await.unwrap();
        memory_ids.push(id);
    }

    println!("BEFORE: Created {} memories with varying alignment", memory_ids.len());

    let request = serde_json::json!({
        "memory_ids": memory_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
        "include_threshold_classification": true,
        "include_per_space_alignment": true
    });

    let response = handlers
        .handle_north_star_alignment(Some(JsonRpcId::Number(1)), Some(request))
        .await;

    assert!(response.error.is_none());
    let result = response.result.unwrap();

    println!("AFTER: Alignment check completed");

    let alignments = result["alignments"].as_array().unwrap();
    for (i, alignment) in alignments.iter().enumerate() {
        let theta = alignment["theta"].as_f64().unwrap();
        let classification = alignment["classification"].as_str().unwrap();
        let per_space = alignment["per_space_alignment"].as_array().unwrap();

        println!("  Memory {}: theta={:.2}, class={}, per_space_len={}",
            i, theta, classification, per_space.len());

        // Verify per-space has 13 elements
        assert_eq!(per_space.len(), 13, "Per-space alignment must have 13 elements");

        // Verify classification matches theta
        let expected = test_cases[i].1;
        assert_eq!(classification, expected, "Classification mismatch for theta={}", theta);
    }

    // Verify summary
    let summary = &result["summary"];
    println!("Summary: optimal={}, acceptable={}, warning={}, critical={}",
        summary["optimal_count"], summary["acceptable_count"],
        summary["warning_count"], summary["critical_count"]);

    assert_eq!(summary["optimal_count"].as_i64().unwrap(), 1);
    assert_eq!(summary["acceptable_count"].as_i64().unwrap(), 1);
    assert_eq!(summary["warning_count"].as_i64().unwrap(), 1);
    assert_eq!(summary["critical_count"].as_i64().unwrap(), 1);

    println!("[VERIFIED] Threshold classification: Optimal≥0.75, Acceptable≥0.70, Warning≥0.55, Critical<0.55");
}

#[tokio::test]
async fn test_goal_hierarchy_navigation() {
    let handlers = create_test_handlers();
    let hierarchy = create_test_goal_hierarchy();

    // Set up hierarchy in store (implementation detail)
    handlers.set_goal_hierarchy(hierarchy).await;

    println!("BEFORE: Testing goal hierarchy navigation");

    // Test get_children of North Star
    let request = serde_json::json!({
        "goal_id": null,  // North Star
        "operation": "get_children"
    });

    let response = handlers
        .handle_goal_hierarchy_query(Some(JsonRpcId::Number(1)), Some(request))
        .await;

    assert!(response.error.is_none());
    let result = response.result.unwrap();

    let goal = &result["goal"];
    let level = goal["level"].as_str().unwrap();
    assert_eq!(level, "north_star");

    let children = result["children"].as_array().unwrap();
    println!("AFTER: North Star has {} children", children.len());

    for child in children {
        let child_level = child["level"].as_str().unwrap();
        assert_eq!(child_level, "strategic", "Children of NorthStar must be Strategic");
        println!("  Child: {} ({})", child["id"], child["level"]);
    }

    println!("[VERIFIED] Goal hierarchy navigation works correctly");
}

#[tokio::test]
async fn test_alignment_drift_detection() {
    let handlers = create_test_handlers();

    // Create memory with evolution history showing drift
    let current_theta = 0.60;
    let previous_theta = 0.80;
    let delta = current_theta - previous_theta; // -0.20

    let fp = create_fingerprint_with_evolution(current_theta, previous_theta);
    let id = handlers.teleological_store.store(fp).await.unwrap();

    println!("BEFORE: Created memory with drift");
    println!("  Current theta: {:.2}", current_theta);
    println!("  Previous theta: {:.2}", previous_theta);
    println!("  Delta: {:.2}", delta);

    let request = serde_json::json!({
        "memory_ids": [id.to_string()],
        "drift_threshold": -0.15,  // Trigger on >= 15% degradation
        "time_window_hours": 24
    });

    let response = handlers
        .handle_alignment_drift_check(Some(JsonRpcId::Number(1)), Some(request))
        .await;

    assert!(response.error.is_none());
    let result = response.result.unwrap();

    let drifting = result["drifting_memories"].as_array().unwrap();
    assert_eq!(drifting.len(), 1, "Should detect 1 drifting memory");

    let drift = &drifting[0];
    let detected_delta = drift["delta"].as_f64().unwrap();
    assert!(detected_delta < -0.15, "Delta should exceed threshold");

    println!("AFTER: Detected drift");
    println!("  Delta detected: {:.4}", detected_delta);
    println!("  Current: {}, Previous: {}",
        drift["current_classification"], drift["previous_classification"]);

    let summary = &result["summary"];
    println!("Summary: total={}, drifting={}, worst_drift={}",
        summary["total_checked"], summary["drifting_count"], summary["worst_drift"]);

    println!("[VERIFIED] Alignment drift detection works correctly");
}

#[tokio::test]
async fn edge_case_north_star_not_configured() {
    let handlers = create_test_handlers();
    // DO NOT set up goal hierarchy

    println!("EDGE CASE: North Star not configured");

    let request = serde_json::json!({
        "memory_ids": [Uuid::new_v4().to_string()]
    });

    let response = handlers
        .handle_north_star_alignment(Some(JsonRpcId::Number(1)), Some(request))
        .await;

    assert!(response.error.is_some());
    let error = response.error.unwrap();
    assert_eq!(error.code, error_codes::NORTH_STAR_NOT_CONFIGURED);

    println!("[VERIFIED] Returns NORTH_STAR_NOT_CONFIGURED error: {}", error.message);
}

#[tokio::test]
async fn edge_case_purpose_vector_out_of_range() {
    let handlers = create_test_handlers();

    // Values outside [-1.0, 1.0]
    let bad_pv = [1.5, 0.2, 0.3, 0.1, 0.6, -1.5, 0.7, 0.2, 0.1, 0.4, 0.2, 0.3, 0.1];

    println!("EDGE CASE: Purpose vector values out of range");
    println!("  Input: {:?}", bad_pv);

    let request = serde_json::json!({
        "purpose_vector": bad_pv
    });

    let response = handlers
        .handle_purpose_query(Some(JsonRpcId::Number(1)), Some(request))
        .await;

    assert!(response.error.is_some());
    let error = response.error.unwrap();
    assert_eq!(error.code, error_codes::INVALID_PARAMS);
    assert!(error.message.contains("-1.0") || error.message.contains("1.0") || error.message.contains("range"));

    println!("[VERIFIED] Out of range values rejected: {}", error.message);
}
```

### Verification Commands

```bash
# Run purpose handler tests
cargo test -p context-graph-mcp purpose --no-fail-fast -- --nocapture

# Verify goal hierarchy
cargo test -p context-graph-mcp goal_hierarchy --no-fail-fast -- --nocapture

# Test drift detection
cargo test -p context-graph-mcp alignment_drift --no-fail-fast -- --nocapture

# Run all handler tests
cargo test -p context-graph-mcp handlers --no-fail-fast -- --nocapture
```

## Files to Create

| File | Description |
|------|-------------|
| `crates/context-graph-mcp/src/handlers/purpose.rs` | Purpose and goal handlers |
| `crates/context-graph-mcp/src/handlers/tests/purpose.rs` | Tests for purpose handlers |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-mcp/src/handlers/mod.rs` | Add `mod purpose;` |
| `crates/context-graph-mcp/src/handlers/core.rs` | Add `GoalAlignmentCalculator` to Handlers struct, add dispatch cases |
| `crates/context-graph-mcp/src/protocol.rs` | Add method constants and error codes |
| `crates/context-graph-mcp/src/handlers/tests/mod.rs` | Add `mod purpose;` |

---

## Full State Verification Requirements

After completing implementation, you MUST perform Full State Verification.

### Source of Truth Definition

| Component | Source of Truth | How to Verify |
|-----------|-----------------|---------------|
| Purpose vector dimension | `NUM_EMBEDDERS = 13` | Assert array length == 13 |
| Alignment thresholds | `AlignmentThreshold::classify()` | Verify θ≥0.75=Optimal, θ≥0.70=Acceptable, θ≥0.55=Warning, θ<0.55=Critical |
| Goal levels | `GoalLevel` enum | Verify 4 levels with correct propagation weights |
| Error codes | `protocol.rs error_codes` | Verify exact code returned matches constant |
| Goal hierarchy | `GoalHierarchy` | Verify NorthStar has no parent, children have correct levels |

### Execute & Inspect Protocol

For each test, you MUST:

1. **BEFORE**: Log initial state
```rust
println!("BEFORE: Store fingerprint count = {}", store.count().await.unwrap());
println!("BEFORE: Purpose vector = {:?}", pv.alignments);
```

2. **EXECUTE**: Run the operation

3. **AFTER**: Perform SEPARATE read operation on Source of Truth
```rust
// SEPARATE verification query
let retrieved = store.retrieve(result_id).await.unwrap().unwrap();
println!("AFTER: Retrieved fingerprint {} - theta = {:.4}",
    retrieved.id, retrieved.theta_to_north_star);
println!("AFTER: Classification = {:?}",
    AlignmentThreshold::classify(retrieved.theta_to_north_star));
```

### Boundary & Edge Case Audit

These edge cases MUST be tested:

1. **Empty memory_ids array** - Returns INVALID_PARAMS
2. **Non-existent memory ID** - Returns FINGERPRINT_NOT_FOUND
3. **Non-existent goal ID** - Returns GOAL_NOT_FOUND
4. **North Star not configured** - Returns NORTH_STAR_NOT_CONFIGURED
5. **Purpose vector with 12 elements** - Returns INVALID_PARAMS with "13 elements required"
6. **Purpose vector values outside [-1.0, 1.0]** - Returns INVALID_PARAMS
7. **goal_id=null for get_children** - Returns North Star's children
8. **Memory with no evolution history for drift check** - Skipped, not error
9. **drift_threshold=0.0** - All changes flagged as drift

### Evidence of Success

Your implementation is complete when you can show:

1. **13D purpose vector verification**:
```
Purpose query: input vector has 13 elements
Response: all results have 13-element alignment arrays
weights_applied = [0.8, 0.2, 0.3, 0.1, 0.6, 0.1, 0.7, 0.2, 0.1, 0.4, 0.2, 0.3, 0.1]
```

2. **Threshold classification verification**:
```
theta=0.80 -> classification=optimal ✓
theta=0.72 -> classification=acceptable ✓
theta=0.60 -> classification=warning ✓
theta=0.40 -> classification=critical ✓
```

3. **Goal hierarchy verification**:
```
NorthStar (depth=0, weight=1.0) -> 2 Strategic children
Strategic (depth=1, weight=0.7) -> 3 Tactical children
Tactical (depth=2, weight=0.4) -> 5 Immediate children
```

4. **Drift detection verification**:
```
Memory abc123: theta 0.80 -> 0.60, delta=-0.20 (>threshold -0.15)
Classification: optimal -> warning
DRIFT DETECTED ✓
```

### Manual Verification Checklist

After running tests, manually verify:

- [ ] `cargo test -p context-graph-mcp purpose` passes with all `[VERIFIED]` messages
- [ ] No test uses mock data (grep for "mock" should return nothing in test files)
- [ ] All error codes match `protocol.rs` constants exactly
- [ ] Purpose vectors have exactly 13 elements (not 12)
- [ ] AlignmentThreshold thresholds match constitution.yaml (0.75, 0.70, 0.55)
- [ ] GoalLevel propagation weights match (1.0, 0.7, 0.4, 0.2)
- [ ] All JSON responses have correct structure per schemas above

---

## Traceability

| Requirement | Source | Coverage |
|-------------|--------|----------|
| FR-201 | constitution.yaml | 13D purpose vector query |
| FR-202 | constitution.yaml | North Star thresholds (Optimal≥0.75, etc.) |
| FR-303 | constitution.yaml | Goal hierarchy navigation (4 levels) |
| Propagation weights | constitution.yaml | NorthStar=1.0, Strategic=0.7, Tactical=0.4, Immediate=0.2 |
| Drift detection | Royse 2026 | -0.15 threshold default |

---

*Task created: 2026-01-04*
*Task updated: 2026-01-05 - CORRECTED for 13 embedders, actual file paths, Full State Verification*
*Layer: Surface*
*Priority: P0 - Core teleological purpose API*
