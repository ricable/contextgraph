# TASK-S004: New MCP Handlers for Johari Quadrant Operations

```yaml
metadata:
  id: "TASK-S004"
  title: "New MCP Handlers for Johari Quadrant Operations"
  layer: "surface"
  priority: "P1"
  estimated_hours: 6
  created: "2026-01-04"
  status: "pending"
  dependencies:
    - "TASK-F003"  # JohariFingerprint struct
    - "TASK-L004"  # Johari Transition Manager
  traces_to:
    - "FR-203"  # JohariFingerprint Per Embedder
```

## Problem Statement

Create new MCP handlers for Johari Window operations including per-embedder quadrant queries, quadrant distribution analysis, and transition state updates.

## Context

The Johari Window model classifies each embedding space into four quadrants:
- **Open**: Low entropy, High coherence (aware in this space)
- **Blind**: High entropy, Low coherence (discovery opportunity)
- **Hidden**: Low entropy, Low coherence (latent in this space)
- **Unknown**: High entropy, High coherence (frontier in this space)

Cross-space Johari analysis enables targeted learning. A memory can be Open(semantic) but Blind(causal), indicating an opportunity for causal understanding.

**These are NEW handlers - no legacy equivalents exist.**

## Technical Specification

### MCP Handler Function Signatures

```rust
/// Get Johari quadrant distribution for a memory
pub async fn handle_johari_get_distribution(
    request: JohariDistributionRequest,
    store: Arc<dyn TeleologicalMemoryStore>,
) -> Result<JohariDistributionResponse, McpError>;

/// Find memories by Johari quadrant for specific embedder
pub async fn handle_johari_find_by_quadrant(
    request: JohariFindByQuadrantRequest,
    store: Arc<dyn TeleologicalMemoryStore>,
) -> Result<JohariFindByQuadrantResponse, McpError>;

/// Update Johari classification for a memory
pub async fn handle_johari_update(
    request: JohariUpdateRequest,
    transition_mgr: Arc<dyn JohariTransitionManager>,
) -> Result<JohariUpdateResponse, McpError>;

/// Get cross-space Johari analysis (blind spots, opportunities)
pub async fn handle_johari_cross_space_analysis(
    request: JohariCrossSpaceRequest,
    store: Arc<dyn TeleologicalMemoryStore>,
) -> Result<JohariCrossSpaceResponse, McpError>;

/// Get Johari transition probabilities for embedder
pub async fn handle_johari_transition_probabilities(
    request: JohariTransitionRequest,
    transition_mgr: Arc<dyn JohariTransitionManager>,
) -> Result<JohariTransitionResponse, McpError>;
```

### Request/Response JSON Schemas

#### JohariDistributionRequest

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["memory_id"],
  "properties": {
    "memory_id": {
      "type": "string",
      "format": "uuid"
    },
    "include_confidence": {
      "type": "boolean",
      "default": true,
      "description": "Include confidence scores per classification"
    },
    "include_transition_predictions": {
      "type": "boolean",
      "default": false,
      "description": "Include predicted next quadrant per embedder"
    }
  }
}
```

#### JohariDistributionResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["memory_id", "per_embedder_quadrants"],
  "properties": {
    "memory_id": { "type": "string", "format": "uuid" },
    "per_embedder_quadrants": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["embedder_index", "embedder_name", "quadrant"],
        "properties": {
          "embedder_index": { "type": "integer", "minimum": 0, "maximum": 11 },
          "embedder_name": { "type": "string" },
          "quadrant": {
            "type": "string",
            "enum": ["open", "hidden", "blind", "unknown"]
          },
          "soft_classification": {
            "type": "object",
            "properties": {
              "open": { "type": "number" },
              "hidden": { "type": "number" },
              "blind": { "type": "number" },
              "unknown": { "type": "number" }
            },
            "description": "Probability distribution across quadrants"
          },
          "confidence": { "type": "number" },
          "predicted_next_quadrant": {
            "type": "string",
            "enum": ["open", "hidden", "blind", "unknown"]
          }
        }
      },
      "minItems": 12,
      "maxItems": 12
    },
    "summary": {
      "type": "object",
      "properties": {
        "open_count": { "type": "integer" },
        "hidden_count": { "type": "integer" },
        "blind_count": { "type": "integer" },
        "unknown_count": { "type": "integer" },
        "average_confidence": { "type": "number" }
      }
    }
  }
}
```

#### JohariFindByQuadrantRequest

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["embedder_index", "quadrant"],
  "properties": {
    "embedder_index": {
      "type": "integer",
      "minimum": 0,
      "maximum": 11,
      "description": "Which embedding space to filter by"
    },
    "quadrant": {
      "type": "string",
      "enum": ["open", "hidden", "blind", "unknown"]
    },
    "min_confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "default": 0.5
    },
    "top_k": {
      "type": "integer",
      "minimum": 1,
      "maximum": 1000,
      "default": 100
    }
  }
}
```

#### JohariFindByQuadrantResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["embedder_index", "quadrant", "memories"],
  "properties": {
    "embedder_index": { "type": "integer" },
    "embedder_name": { "type": "string" },
    "quadrant": { "type": "string" },
    "memories": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string", "format": "uuid" },
          "confidence": { "type": "number" },
          "soft_classification": {
            "type": "array",
            "items": { "type": "number" },
            "minItems": 4,
            "maxItems": 4
          }
        }
      }
    },
    "total_count": { "type": "integer" }
  }
}
```

#### JohariUpdateRequest

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["memory_id"],
  "properties": {
    "memory_id": {
      "type": "string",
      "format": "uuid"
    },
    "updates": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["embedder_index"],
        "properties": {
          "embedder_index": { "type": "integer", "minimum": 0, "maximum": 11 },
          "new_quadrant": {
            "type": "string",
            "enum": ["open", "hidden", "blind", "unknown"]
          },
          "soft_classification": {
            "type": "object",
            "properties": {
              "open": { "type": "number" },
              "hidden": { "type": "number" },
              "blind": { "type": "number" },
              "unknown": { "type": "number" }
            }
          },
          "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
        }
      },
      "description": "Updates per embedder (partial update allowed)"
    },
    "trigger_event": {
      "type": "string",
      "description": "What triggered this update (for evolution tracking)"
    }
  }
}
```

#### JohariCrossSpaceRequest

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "memory_ids": {
      "type": "array",
      "items": { "type": "string", "format": "uuid" },
      "description": "Specific memories to analyze (null for system-wide)"
    },
    "analysis_type": {
      "type": "string",
      "enum": ["blind_spots", "learning_opportunities", "quadrant_correlation", "all"],
      "default": "all"
    }
  }
}
```

#### JohariCrossSpaceResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "blind_spots": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "memory_id": { "type": "string", "format": "uuid" },
          "aware_space": { "type": "integer" },
          "blind_space": { "type": "integer" },
          "description": { "type": "string" },
          "learning_suggestion": { "type": "string" }
        }
      },
      "description": "Memories that are Open in one space but Blind in another"
    },
    "learning_opportunities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "memory_id": { "type": "string", "format": "uuid" },
          "unknown_spaces": { "type": "array", "items": { "type": "integer" } },
          "potential": { "type": "string", "enum": ["high", "medium", "low"] }
        }
      },
      "description": "Memories with Unknown quadrants (frontiers)"
    },
    "quadrant_correlation": {
      "type": "object",
      "description": "Correlation matrix between embedder quadrant assignments",
      "additionalProperties": {
        "type": "object",
        "additionalProperties": { "type": "number" }
      }
    }
  }
}
```

#### JohariTransitionRequest

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["embedder_index"],
  "properties": {
    "embedder_index": {
      "type": "integer",
      "minimum": 0,
      "maximum": 11
    },
    "memory_id": {
      "type": "string",
      "format": "uuid",
      "description": "Get memory-specific probabilities (null for global)"
    }
  }
}
```

#### JohariTransitionResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["embedder_index", "transition_matrix"],
  "properties": {
    "embedder_index": { "type": "integer" },
    "embedder_name": { "type": "string" },
    "transition_matrix": {
      "type": "object",
      "properties": {
        "from_open": {
          "type": "object",
          "properties": {
            "to_open": { "type": "number" },
            "to_hidden": { "type": "number" },
            "to_blind": { "type": "number" },
            "to_unknown": { "type": "number" }
          }
        },
        "from_hidden": { "$ref": "#/definitions/TransitionRow" },
        "from_blind": { "$ref": "#/definitions/TransitionRow" },
        "from_unknown": { "$ref": "#/definitions/TransitionRow" }
      },
      "description": "4x4 transition probability matrix"
    },
    "sample_size": {
      "type": "integer",
      "description": "Number of transitions used to compute probabilities"
    }
  }
}
```

### Error Handling

```rust
#[derive(Debug, Clone, Serialize)]
pub enum McpJohariError {
    /// Invalid embedder index
    InvalidEmbedderIndex { index: usize, max: usize },
    /// Invalid quadrant name
    InvalidQuadrant { quadrant: String },
    /// Soft classification doesn't sum to 1.0
    InvalidSoftClassification { sum: f32 },
    /// Memory not found
    MemoryNotFound { id: Uuid },
    /// Confidence out of range
    ConfidenceOutOfRange { value: f32 },
}
```

## Implementation Requirements

### Prerequisites

- [ ] TASK-F003 complete (JohariFingerprint struct)
- [ ] TASK-L004 complete (Johari Transition Manager)

### Scope

#### In Scope

- Per-memory Johari distribution queries
- Find memories by quadrant per embedder
- Johari classification updates with evolution tracking
- Cross-space analysis (blind spots, opportunities)
- Transition probability queries
- Soft classification support (probability distributions)

#### Out of Scope

- Purpose operations (TASK-S003)
- Search operations (TASK-S002)
- Storage operations (TASK-S001)

### Constraints

- All 12 embedders must have quadrant data
- Soft classifications must sum to 1.0
- Confidence must be in [0, 1]
- Transition updates trigger evolution snapshots

## Definition of Done

### Implementation Checklist

- [ ] `handle_johari_get_distribution` with soft classification
- [ ] `handle_johari_find_by_quadrant` with confidence filter
- [ ] `handle_johari_update` with evolution tracking
- [ ] `handle_johari_cross_space_analysis` for insights
- [ ] `handle_johari_transition_probabilities` for prediction
- [ ] All error cases with context

### Testing Requirements

Tests MUST use REAL Johari classifications from actual entropy/coherence measurements.

```rust
#[cfg(test)]
mod tests {
    use crate::test_fixtures::load_real_johari_fingerprints;

    #[tokio::test]
    async fn test_johari_distribution() {
        let memory = store_memory_with_real_johari();

        let request = JohariDistributionRequest {
            memory_id: memory.id,
            include_confidence: true,
            include_transition_predictions: true,
        };

        let response = handle_johari_get_distribution(request, store.clone()).await.unwrap();

        assert_eq!(response.per_embedder_quadrants.len(), 12);

        // Verify all quadrants are valid
        for eq in &response.per_embedder_quadrants {
            assert!(["open", "hidden", "blind", "unknown"].contains(&eq.quadrant.as_str()));
            assert!(eq.confidence >= 0.0 && eq.confidence <= 1.0);
        }

        // Verify summary
        let sum = response.summary.open_count
            + response.summary.hidden_count
            + response.summary.blind_count
            + response.summary.unknown_count;
        assert_eq!(sum, 12);
    }

    #[tokio::test]
    async fn test_johari_find_by_quadrant() {
        // Store memories with known Johari patterns
        let semantic_open_memories = store_semantic_open_memories(5);

        let request = JohariFindByQuadrantRequest {
            embedder_index: 0, // E1 semantic
            quadrant: "open".into(),
            min_confidence: 0.7,
            top_k: 10,
        };

        let response = handle_johari_find_by_quadrant(request, store.clone()).await.unwrap();

        // All returned memories should have E1 in Open quadrant
        for mem in &response.memories {
            assert!(mem.confidence >= 0.7);
        }
        assert!(response.total_count >= 5);
    }

    #[tokio::test]
    async fn test_johari_cross_space_blind_spots() {
        // Memory that is Open(E1) but Blind(E5)
        let blind_spot_memory = create_semantic_open_causal_blind_memory();

        let request = JohariCrossSpaceRequest {
            memory_ids: Some(vec![blind_spot_memory.id]),
            analysis_type: "blind_spots".into(),
        };

        let response = handle_johari_cross_space_analysis(request, store.clone()).await.unwrap();

        assert!(!response.blind_spots.is_empty());
        let spot = &response.blind_spots[0];
        assert_eq!(spot.aware_space, 0);  // E1 semantic
        assert_eq!(spot.blind_space, 4);  // E5 causal
    }

    #[tokio::test]
    async fn test_johari_update_triggers_evolution() {
        let memory = store_memory_with_real_johari();
        let initial_evolution_count = memory.purpose_evolution.len();

        let request = JohariUpdateRequest {
            memory_id: memory.id,
            updates: vec![JohariEmbedderUpdate {
                embedder_index: 2,
                new_quadrant: Some("open".into()),
                confidence: Some(0.9),
                ..Default::default()
            }],
            trigger_event: Some("awareness_gained".into()),
        };

        let response = handle_johari_update(request, mgr.clone()).await.unwrap();

        // Verify evolution snapshot was created
        let updated = store.retrieve(memory.id).await.unwrap().unwrap();
        assert!(updated.purpose_evolution.len() > initial_evolution_count);
    }
}
```

### Verification Commands

```bash
# Run Johari handler tests
cargo test -p context-graph-mcp johari_handlers

# Verify cross-space analysis
cargo test -p context-graph-mcp cross_space_analysis

# Test transition matrix
cargo test -p context-graph-mcp johari_transitions
```

## Files to Create

| File | Description |
|------|-------------|
| `crates/context-graph-mcp/src/handlers/johari.rs` | Johari quadrant handlers |
| `crates/context-graph-mcp/src/schemas/johari_distribution.json` | Distribution schema |
| `crates/context-graph-mcp/src/schemas/johari_cross_space.json` | Cross-space schema |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-mcp/src/handlers/mod.rs` | Export johari handlers |
| `crates/context-graph-mcp/src/router.rs` | Register johari routes |
| `crates/context-graph-mcp/src/error.rs` | Add McpJohariError |

## Traceability

| Requirement | Source | Coverage |
|-------------|--------|----------|
| FR-203 | FUNC-SPEC-001 | Johari per embedder |
| AC-203.1 | FUNC-SPEC-001 | All 12 spaces classified |
| AC-203.2 | FUNC-SPEC-001 | Confidence scores |
| AC-203.3 | FUNC-SPEC-001 | Transition probabilities |
| AC-203.4 | FUNC-SPEC-001 | Bitmap index queries |

---

*Task created: 2026-01-04*
*Layer: Surface*
*Priority: P1 - Awareness features*
