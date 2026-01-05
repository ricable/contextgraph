# TASK-S002: Update MCP Search Handlers for Weighted Multi-Array Queries

```yaml
metadata:
  id: "TASK-S002"
  title: "Update MCP Search Handlers for Weighted Multi-Array Queries"
  layer: "surface"
  priority: "P0"
  estimated_hours: 8
  created: "2026-01-04"
  status: "pending"
  dependencies:
    - "TASK-L001"  # Multi-Embedding Query Executor
    - "TASK-L006"  # Purpose Pattern Index
    - "TASK-L007"  # Cross-Space Similarity Engine
  traces_to:
    - "FR-401"  # Weighted Similarity Across 12 Embedders
    - "FR-402"  # Per-Embedder Weight Configuration
```

## Problem Statement

Update MCP search handlers to accept per-embedder weight parameters, execute multi-space queries, and return results with per-embedder similarity scores for explainability.

## Context

Search operations must now:
1. Accept configurable weights for each of 12 embedding spaces
2. Support query type presets (semantic, causal, code, temporal, fact-checking)
3. Return per-embedder scores for explainability
4. Support single-space searches for targeted retrieval
5. Enable purpose-aligned search with North Star weighting

## Technical Specification

### MCP Handler Function Signatures

```rust
/// Multi-embedding semantic search
pub async fn handle_search_multi(
    request: MultiSearchRequest,
    executor: Arc<dyn MultiEmbeddingQueryExecutor>,
) -> Result<MultiSearchResponse, McpError>;

/// Single-space targeted search
pub async fn handle_search_single_space(
    request: SingleSpaceSearchRequest,
    executor: Arc<dyn MultiEmbeddingQueryExecutor>,
) -> Result<SingleSpaceSearchResponse, McpError>;

/// Search by purpose vector similarity
pub async fn handle_search_by_purpose(
    request: PurposeSearchRequest,
    store: Arc<dyn TeleologicalMemoryStore>,
) -> Result<PurposeSearchResponse, McpError>;

/// Get configured weight profiles
pub async fn handle_get_weight_profiles(
) -> Result<WeightProfilesResponse, McpError>;
```

### Request/Response JSON Schemas

#### MultiSearchRequest

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["query"],
  "properties": {
    "query": {
      "oneOf": [
        { "type": "string", "description": "Query text to embed" },
        {
          "type": "object",
          "description": "Pre-computed query embeddings",
          "required": ["embeddings"],
          "properties": {
            "embeddings": { "$ref": "#/definitions/SemanticFingerprint" }
          }
        }
      ]
    },
    "query_type": {
      "type": "string",
      "enum": ["semantic_search", "causal_reasoning", "code_search", "temporal_navigation", "fact_checking", "balanced", "custom"],
      "default": "semantic_search",
      "description": "Predefined weight profile to use"
    },
    "weights": {
      "type": "array",
      "items": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
      "minItems": 12,
      "maxItems": 12,
      "description": "Custom per-embedder weights (required if query_type is 'custom')"
    },
    "active_spaces": {
      "type": "array",
      "items": { "type": "integer", "minimum": 0, "maximum": 11 },
      "description": "Which embedding spaces to search (default: all)"
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
    "include_per_embedder_scores": {
      "type": "boolean",
      "default": true,
      "description": "Include similarity breakdown per embedding space"
    },
    "include_purpose_alignment": {
      "type": "boolean",
      "default": false,
      "description": "Include alignment to North Star in results"
    }
  }
}
```

#### MultiSearchResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["results", "query_metadata"],
  "properties": {
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "aggregate_similarity"],
        "properties": {
          "id": { "type": "string", "format": "uuid" },
          "aggregate_similarity": {
            "type": "number",
            "description": "Weighted average similarity across active spaces"
          },
          "per_embedder_scores": {
            "type": "object",
            "description": "Similarity scores per embedding space",
            "properties": {
              "e1_semantic": { "type": "number" },
              "e2_temporal_recent": { "type": "number" },
              "e3_temporal_periodic": { "type": "number" },
              "e4_temporal_positional": { "type": "number" },
              "e5_causal": { "type": "number" },
              "e6_sparse": { "type": "number" },
              "e7_code": { "type": "number" },
              "e8_graph": { "type": "number" },
              "e9_hdc": { "type": "number" },
              "e10_multimodal": { "type": "number" },
              "e11_entity": { "type": "number" },
              "e12_late_interaction": { "type": "number" }
            }
          },
          "top_contributing_spaces": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "space_index": { "type": "integer" },
                "space_name": { "type": "string" },
                "weighted_contribution": { "type": "number" }
              }
            },
            "maxItems": 3
          },
          "purpose_alignment": {
            "type": "number",
            "description": "Alignment to North Star (if requested)"
          }
        }
      }
    },
    "query_metadata": {
      "type": "object",
      "properties": {
        "query_type_used": { "type": "string" },
        "weights_applied": {
          "type": "array",
          "items": { "type": "number" },
          "minItems": 12,
          "maxItems": 12
        },
        "spaces_searched": { "type": "integer" },
        "total_candidates_scanned": { "type": "integer" },
        "search_time_ms": { "type": "number" }
      }
    }
  }
}
```

#### SingleSpaceSearchRequest

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["space_index", "query_embedding"],
  "properties": {
    "space_index": {
      "type": "integer",
      "minimum": 0,
      "maximum": 11,
      "description": "Which embedding space to search (0=E1, 11=E12)"
    },
    "query_embedding": {
      "type": "array",
      "items": { "type": "number" },
      "description": "Query embedding vector for the specified space"
    },
    "top_k": { "type": "integer", "default": 10 },
    "min_similarity": { "type": "number", "default": 0.0 }
  }
}
```

### Weight Profile Configuration

```rust
/// Predefined weight profiles per query type
pub const WEIGHT_PROFILES: &[(&str, [f32; 12])] = &[
    ("semantic_search", [0.30, 0.05, 0.05, 0.05, 0.10, 0.05, 0.20, 0.05, 0.05, 0.05, 0.03, 0.02]),
    ("causal_reasoning", [0.15, 0.03, 0.03, 0.03, 0.45, 0.03, 0.10, 0.03, 0.03, 0.05, 0.05, 0.02]),
    ("code_search", [0.15, 0.02, 0.02, 0.35, 0.05, 0.03, 0.25, 0.02, 0.02, 0.03, 0.03, 0.03]),
    ("temporal_navigation", [0.15, 0.20, 0.20, 0.20, 0.05, 0.02, 0.05, 0.02, 0.03, 0.03, 0.03, 0.02]),
    ("fact_checking", [0.10, 0.02, 0.02, 0.02, 0.20, 0.05, 0.05, 0.02, 0.02, 0.05, 0.43, 0.02]),
    ("balanced", [0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.087]),
];

/// Validate that weights sum to ~1.0
pub fn validate_weights(weights: &[f32; 12]) -> Result<(), McpError> {
    let sum: f32 = weights.iter().sum();
    if (sum - 1.0).abs() > 0.01 {
        return Err(McpError::validation_error(format!(
            "Weights must sum to 1.0, got {}. Weights: {:?}",
            sum, weights
        )));
    }
    Ok(())
}
```

### Error Handling

```rust
#[derive(Debug, Clone, Serialize)]
pub enum McpSearchError {
    /// Weights don't sum to 1.0
    InvalidWeights { sum: f32, weights: Vec<f32> },
    /// Custom query_type requires weights
    MissingCustomWeights,
    /// Query embedding dimension mismatch for space
    QueryDimensionMismatch {
        space_index: usize,
        expected: usize,
        actual: usize,
    },
    /// Invalid space index
    InvalidSpaceIndex { index: usize, max: usize },
    /// Search execution error
    ExecutionError { message: String },
}
```

## Implementation Requirements

### Prerequisites

- [ ] TASK-L001 complete (MultiEmbeddingQueryExecutor)
- [ ] TASK-L006 complete (Purpose Pattern Index for purpose search)
- [ ] TASK-L007 complete (Cross-Space Similarity Engine)

### Scope

#### In Scope

- Multi-space search with configurable weights
- Query type presets (semantic, causal, code, etc.)
- Custom weight support with validation
- Per-embedder score breakdown in results
- Single-space targeted search
- Purpose vector similarity search
- Top contributing spaces identification

#### Out of Scope

- Goal hierarchy navigation (TASK-S003)
- Johari quadrant filtering (TASK-S004)
- Storage/retrieval (TASK-S001)

### Constraints

- Query latency target: < 50ms (warm indexes)
- Weights MUST sum to 1.0 (validated)
- Dimension validation per space
- Fail fast on invalid input

## Definition of Done

### Implementation Checklist

- [ ] `handle_search_multi` with weight profiles
- [ ] Weight validation (sum to 1.0)
- [ ] Per-embedder score breakdown
- [ ] Top contributing spaces identification
- [ ] `handle_search_single_space` for targeted queries
- [ ] `handle_search_by_purpose` for purpose-aligned search
- [ ] `handle_get_weight_profiles` for discovery
- [ ] All error cases with context

### Testing Requirements

Tests MUST use REAL embeddings from actual models.

```rust
#[cfg(test)]
mod tests {
    use crate::test_fixtures::load_real_query_embeddings;

    #[tokio::test]
    async fn test_multi_search_semantic_preset() {
        let query_embeddings = load_real_query_embeddings("what is machine learning");

        let request = MultiSearchRequest {
            query: QueryInput::Embeddings(query_embeddings),
            query_type: Some("semantic_search".into()),
            weights: None,
            top_k: 10,
            include_per_embedder_scores: true,
            ..Default::default()
        };

        let response = handle_search_multi(request, executor.clone()).await.unwrap();

        // Verify semantic_search weights applied
        let weights = response.query_metadata.weights_applied;
        assert!((weights[0] - 0.30).abs() < 0.01); // E1 semantic weight

        // Verify per-embedder scores present
        for result in &response.results {
            assert!(result.per_embedder_scores.is_some());
            let scores = result.per_embedder_scores.as_ref().unwrap();
            assert!(scores.e1_semantic >= 0.0 && scores.e1_semantic <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_multi_search_custom_weights() {
        let custom_weights = [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let request = MultiSearchRequest {
            query: QueryInput::Text("causal query".into()),
            query_type: Some("custom".into()),
            weights: Some(custom_weights),
            ..Default::default()
        };

        let response = handle_search_multi(request, executor.clone()).await.unwrap();
        assert_eq!(response.query_metadata.weights_applied, custom_weights);
    }

    #[tokio::test]
    async fn test_multi_search_invalid_weights() {
        let bad_weights = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // Sum = 1.5

        let request = MultiSearchRequest {
            query: QueryInput::Text("test".into()),
            query_type: Some("custom".into()),
            weights: Some(bad_weights),
            ..Default::default()
        };

        let result = handle_search_multi(request, executor.clone()).await;
        assert!(matches!(
            result.unwrap_err(),
            McpError { code, .. } if code.contains("invalid_weights")
        ));
    }

    #[tokio::test]
    async fn test_single_space_search() {
        let e1_query = load_real_embedding_for_space(0, "test query");

        let request = SingleSpaceSearchRequest {
            space_index: 0,
            query_embedding: e1_query,
            top_k: 5,
            min_similarity: 0.5,
        };

        let response = handle_search_single_space(request, executor.clone()).await.unwrap();
        assert!(response.results.len() <= 5);
        for result in &response.results {
            assert!(result.similarity >= 0.5);
        }
    }

    #[tokio::test]
    async fn test_top_contributing_spaces() {
        let response = handle_search_multi(/* ... */).await.unwrap();

        for result in &response.results {
            let top = &result.top_contributing_spaces;
            assert!(top.len() <= 3);
            // Verify sorted by contribution
            for i in 1..top.len() {
                assert!(top[i-1].weighted_contribution >= top[i].weighted_contribution);
            }
        }
    }
}
```

### Verification Commands

```bash
# Run search handler tests
cargo test -p context-graph-mcp search_handlers

# Verify weight validation
cargo test -p context-graph-mcp weight_validation

# Performance test
cargo bench -p context-graph-mcp search_performance
```

## Files to Create

| File | Description |
|------|-------------|
| `crates/context-graph-mcp/src/handlers/search.rs` | Multi-embedding search handlers |
| `crates/context-graph-mcp/src/weights.rs` | Weight profiles and validation |
| `crates/context-graph-mcp/src/schemas/search_request.json` | JSON schema for search |
| `crates/context-graph-mcp/src/schemas/search_response.json` | JSON schema for response |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-mcp/src/handlers/mod.rs` | Export search handlers |
| `crates/context-graph-mcp/src/router.rs` | Register search routes |
| `crates/context-graph-mcp/src/error.rs` | Add McpSearchError |

## Files to Delete

| File | Reason |
|------|--------|
| `crates/context-graph-mcp/src/handlers/fused_search.rs` | Legacy fusion search |

## Traceability

| Requirement | Source | Coverage |
|-------------|--------|----------|
| FR-401 | FUNC-SPEC-001 | Weighted similarity |
| FR-402 | FUNC-SPEC-001 | Per-embedder weights |
| FR-403 | FUNC-SPEC-001 | Distance metrics |

---

*Task created: 2026-01-04*
*Layer: Surface*
*Priority: P0 - Core search API*
