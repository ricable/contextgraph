# TASK-S003: New MCP Handlers for Purpose/Goal Operations

```yaml
metadata:
  id: "TASK-S003"
  title: "New MCP Handlers for Purpose/Goal Operations"
  layer: "surface"
  priority: "P0"
  estimated_hours: 8
  created: "2026-01-04"
  status: "pending"
  dependencies:
    - "TASK-L002"  # Purpose Vector Computation
    - "TASK-L003"  # Goal Alignment Calculator
    - "TASK-L006"  # Purpose Pattern Index
  traces_to:
    - "FR-201"  # PurposeVector (12D Alignment Signature)
    - "FR-202"  # North Star Alignment Thresholds
    - "FR-303"  # Goal Hierarchy Index
```

## Problem Statement

Create new MCP handlers for teleological purpose and goal operations including purpose vector queries, North Star alignment checking, and goal hierarchy navigation.

## Context

Purpose-driven operations are central to the teleological architecture. These handlers expose:
1. Purpose vector similarity search (find memories with similar purpose signatures)
2. North Star alignment checking (verify if memories serve the system's goals)
3. Goal hierarchy navigation (traverse and query the goal tree)
4. Alignment drift detection (identify memories becoming misaligned)

**These are NEW handlers - no legacy equivalents exist.**

## Technical Specification

### MCP Handler Function Signatures

```rust
/// Query memories by purpose vector similarity
pub async fn handle_purpose_query(
    request: PurposeQueryRequest,
    store: Arc<dyn TeleologicalMemoryStore>,
) -> Result<PurposeQueryResponse, McpError>;

/// Check alignment to North Star
pub async fn handle_north_star_alignment(
    request: NorthStarAlignmentRequest,
    alignment_calc: Arc<dyn GoalAlignmentCalculator>,
) -> Result<NorthStarAlignmentResponse, McpError>;

/// Navigate goal hierarchy
pub async fn handle_goal_hierarchy_query(
    request: GoalHierarchyRequest,
    store: Arc<dyn TeleologicalMemoryStore>,
) -> Result<GoalHierarchyResponse, McpError>;

/// Find memories aligned to a specific goal
pub async fn handle_find_aligned_to_goal(
    request: GoalAlignedMemoriesRequest,
    store: Arc<dyn TeleologicalMemoryStore>,
) -> Result<GoalAlignedMemoriesResponse, McpError>;

/// Detect alignment drift in memories
pub async fn handle_alignment_drift_check(
    request: AlignmentDriftRequest,
    store: Arc<dyn TeleologicalMemoryStore>,
) -> Result<AlignmentDriftResponse, McpError>;

/// Update the North Star goal
pub async fn handle_north_star_update(
    request: NorthStarUpdateRequest,
    store: Arc<dyn TeleologicalMemoryStore>,
) -> Result<NorthStarUpdateResponse, McpError>;
```

### Request/Response JSON Schemas

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
      "minItems": 12,
      "maxItems": 12,
      "description": "12D purpose vector (alignment per embedder to North Star)"
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
      "minimum": 1,
      "maximum": 12,
      "description": "Only return memories where this embedder is dominant"
    }
  }
}
```

#### PurposeQueryResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["results"],
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
              "alignments": { "type": "array", "items": { "type": "number" }, "minItems": 12, "maxItems": 12 },
              "dominant_embedder": { "type": "integer" },
              "coherence": { "type": "number" }
            }
          },
          "theta_to_north_star": { "type": "number" }
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
      "maxItems": 100,
      "description": "Memory IDs to check alignment for"
    },
    "include_per_space_alignment": {
      "type": "boolean",
      "default": false,
      "description": "Include alignment breakdown per embedding space"
    },
    "include_threshold_classification": {
      "type": "boolean",
      "default": true,
      "description": "Include Optimal/Acceptable/Warning/Critical classification"
    }
  }
}
```

#### NorthStarAlignmentResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["alignments"],
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
            "enum": ["optimal", "acceptable", "warning", "critical"],
            "description": "Alignment threshold classification"
          },
          "per_space_alignment": {
            "type": "array",
            "items": { "type": "number" },
            "minItems": 12,
            "maxItems": 12,
            "description": "Alignment per embedding space (if requested)"
          },
          "dominant_space": {
            "type": "integer",
            "minimum": 1,
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
  "properties": {
    "goal_id": {
      "type": "string",
      "format": "uuid",
      "description": "Goal to query (null for North Star root)"
    },
    "operation": {
      "type": "string",
      "enum": ["get_children", "get_ancestors", "get_subtree", "get_aligned_memories"],
      "default": "get_children"
    },
    "depth": {
      "type": "integer",
      "minimum": 1,
      "maximum": 10,
      "default": 1,
      "description": "Depth for subtree operation"
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
      "required": ["id", "level", "embedding"],
      "properties": {
        "id": { "type": "string", "format": "uuid" },
        "name": { "type": "string" },
        "level": {
          "type": "integer",
          "description": "0=north_star, 1=mid, 2=local"
        },
        "embedding": {
          "type": "array",
          "items": { "type": "number" },
          "minItems": 1536,
          "maxItems": 1536
        },
        "parent_id": { "type": "string", "format": "uuid" }
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
        "max_alignment": { "type": "number" }
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
      "description": "Specific memories to check (null for all)"
    },
    "drift_threshold": {
      "type": "number",
      "default": -0.15,
      "description": "Alignment delta threshold for drift warning (default: -0.15)"
    },
    "time_window_hours": {
      "type": "integer",
      "default": 24,
      "description": "Look back window for drift detection"
    }
  }
}
```

#### AlignmentDriftResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["drifting_memories"],
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
          "delta": { "type": "number" },
          "time_since_previous_hours": { "type": "number" },
          "per_space_deltas": {
            "type": "array",
            "items": { "type": "number" },
            "minItems": 12,
            "maxItems": 12
          },
          "trigger_event": { "type": "string" }
        }
      }
    },
    "summary": {
      "type": "object",
      "properties": {
        "total_checked": { "type": "integer" },
        "drifting_count": { "type": "integer" },
        "average_drift": { "type": "number" },
        "worst_drift": { "type": "number" }
      }
    }
  }
}
```

### Error Handling

```rust
#[derive(Debug, Clone, Serialize)]
pub enum McpPurposeError {
    /// Invalid purpose vector dimension
    InvalidPurposeVectorDimension { expected: usize, actual: usize },
    /// Purpose vector values out of range [-1, 1]
    PurposeValueOutOfRange { index: usize, value: f32 },
    /// Goal not found
    GoalNotFound { id: Uuid },
    /// Memory not found
    MemoryNotFound { id: Uuid },
    /// Invalid goal hierarchy operation
    InvalidHierarchyOperation { operation: String },
    /// North Star not configured
    NorthStarNotConfigured,
}
```

## Implementation Requirements

### Prerequisites

- [ ] TASK-L002 complete (Purpose Vector Computation)
- [ ] TASK-L003 complete (Goal Alignment Calculator)
- [ ] TASK-L006 complete (Purpose Pattern Index)

### Scope

#### In Scope

- Purpose vector similarity search
- North Star alignment checking with thresholds
- Goal hierarchy navigation (children, ancestors, subtree)
- Alignment drift detection
- North Star update operation
- Find memories aligned to specific goals

#### Out of Scope

- Semantic content search (TASK-S002)
- Johari operations (TASK-S004)
- Memory store/retrieve (TASK-S001)

### Constraints

- Purpose vector must be exactly 12D
- Alignment values must be in [-1, 1]
- Goal hierarchy limited to 3 levels (north_star, mid, local)
- Drift detection requires purpose evolution history

## Definition of Done

### Implementation Checklist

- [ ] `handle_purpose_query` for purpose similarity search
- [ ] `handle_north_star_alignment` with threshold classification
- [ ] `handle_goal_hierarchy_query` with all operations
- [ ] `handle_find_aligned_to_goal` for goal-based retrieval
- [ ] `handle_alignment_drift_check` with configurable threshold
- [ ] `handle_north_star_update` for goal management
- [ ] All error cases with context

### Testing Requirements

Tests MUST use REAL purpose vectors computed from REAL embeddings.

```rust
#[cfg(test)]
mod tests {
    use crate::test_fixtures::{load_real_purpose_vectors, create_test_goal_hierarchy};

    #[tokio::test]
    async fn test_purpose_query() {
        let query_pv = load_real_purpose_vectors("ml_concept_purpose.json");

        let request = PurposeQueryRequest {
            purpose_vector: query_pv.alignments,
            top_k: 5,
            min_similarity: 0.7,
            ..Default::default()
        };

        let response = handle_purpose_query(request, store.clone()).await.unwrap();

        // Verify results have similar purpose signatures
        for result in &response.results {
            assert!(result.purpose_similarity >= 0.7);
            assert_eq!(result.purpose_vector.alignments.len(), 12);
        }
    }

    #[tokio::test]
    async fn test_north_star_alignment_thresholds() {
        let memories = store_test_memories_with_varying_alignment();

        let request = NorthStarAlignmentRequest {
            memory_ids: memories.iter().map(|m| m.id).collect(),
            include_threshold_classification: true,
            include_per_space_alignment: true,
        };

        let response = handle_north_star_alignment(request, calc.clone()).await.unwrap();

        // Verify threshold classifications
        for alignment in &response.alignments {
            match alignment.theta {
                t if t >= 0.75 => assert_eq!(alignment.classification, "optimal"),
                t if t >= 0.70 => assert_eq!(alignment.classification, "acceptable"),
                t if t >= 0.55 => assert_eq!(alignment.classification, "warning"),
                _ => assert_eq!(alignment.classification, "critical"),
            }
        }

        // Verify per-space alignment
        for alignment in &response.alignments {
            assert_eq!(alignment.per_space_alignment.as_ref().unwrap().len(), 12);
        }
    }

    #[tokio::test]
    async fn test_goal_hierarchy_navigation() {
        let hierarchy = create_test_goal_hierarchy();

        // Get children of North Star
        let request = GoalHierarchyRequest {
            goal_id: None, // North Star
            operation: "get_children".into(),
            ..Default::default()
        };

        let response = handle_goal_hierarchy_query(request, store.clone()).await.unwrap();
        assert_eq!(response.goal.level, 0); // North Star level
        assert!(response.children.is_some());

        // Children should be level 1 (mid)
        for child in response.children.as_ref().unwrap() {
            assert_eq!(child.level, 1);
        }
    }

    #[tokio::test]
    async fn test_alignment_drift_detection() {
        // Create memory with purpose evolution showing drift
        let drifting_memory = create_drifting_memory(-0.20); // 20% drift

        let request = AlignmentDriftRequest {
            memory_ids: Some(vec![drifting_memory.id]),
            drift_threshold: -0.15,
            time_window_hours: 24,
        };

        let response = handle_alignment_drift_check(request, store.clone()).await.unwrap();

        assert_eq!(response.drifting_memories.len(), 1);
        assert!(response.drifting_memories[0].delta < -0.15);
    }
}
```

### Verification Commands

```bash
# Run purpose handler tests
cargo test -p context-graph-mcp purpose_handlers

# Verify goal hierarchy
cargo test -p context-graph-mcp goal_hierarchy

# Test drift detection
cargo test -p context-graph-mcp alignment_drift
```

## Files to Create

| File | Description |
|------|-------------|
| `crates/context-graph-mcp/src/handlers/purpose.rs` | Purpose vector handlers |
| `crates/context-graph-mcp/src/handlers/goals.rs` | Goal hierarchy handlers |
| `crates/context-graph-mcp/src/schemas/purpose_query.json` | Purpose query schema |
| `crates/context-graph-mcp/src/schemas/goal_hierarchy.json` | Goal hierarchy schema |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-mcp/src/handlers/mod.rs` | Export purpose and goal handlers |
| `crates/context-graph-mcp/src/router.rs` | Register new routes |
| `crates/context-graph-mcp/src/error.rs` | Add McpPurposeError |

## Traceability

| Requirement | Source | Coverage |
|-------------|--------|----------|
| FR-201 | FUNC-SPEC-001 | Purpose vector query |
| FR-202 | FUNC-SPEC-001 | North Star thresholds |
| FR-303 | FUNC-SPEC-001 | Goal hierarchy navigation |

---

*Task created: 2026-01-04*
*Layer: Surface*
*Priority: P0 - Core teleological API*
