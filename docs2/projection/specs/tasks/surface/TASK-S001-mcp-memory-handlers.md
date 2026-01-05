# TASK-S001: Update MCP Memory Handlers for TeleologicalFingerprint

```yaml
metadata:
  id: "TASK-S001"
  title: "Update MCP Memory Handlers for TeleologicalFingerprint"
  layer: "surface"
  priority: "P0"
  estimated_hours: 6
  created: "2026-01-04"
  status: "pending"
  dependencies:
    - "TASK-F001"  # SemanticFingerprint struct
    - "TASK-F002"  # TeleologicalFingerprint struct
    - "TASK-F008"  # TeleologicalMemoryStore trait
    - "TASK-L008"  # Teleological Retrieval Pipeline
  traces_to:
    - "FR-101"  # 12-Embedding Array Storage
    - "FR-301"  # Primary Storage in RocksDB
```

## Problem Statement

Update all MCP memory handlers to operate on TeleologicalFingerprint with 12-embedding arrays instead of legacy single-vector storage. Remove all fusion-related handler code.

## Context

The MCP (Model Context Protocol) handlers are the external API surface for memory operations. They must be updated to:
1. Accept 12-array embeddings on store operations
2. Return complete TeleologicalFingerprint on retrieve operations
3. Handle per-embedder metadata (dimensions, model IDs)
4. Completely remove all fusion-related handlers

**NO BACKWARDS COMPATIBILITY** - clients must update to new API.

## Technical Specification

### MCP Handler Function Signatures

```rust
/// Store memory with TeleologicalFingerprint
pub async fn handle_memory_store(
    request: MemoryStoreRequest,
    store: Arc<dyn TeleologicalMemoryStore>,
) -> Result<MemoryStoreResponse, McpError>;

/// Retrieve memory by ID
pub async fn handle_memory_retrieve(
    request: MemoryRetrieveRequest,
    store: Arc<dyn TeleologicalMemoryStore>,
) -> Result<MemoryRetrieveResponse, McpError>;

/// Delete memory by ID
pub async fn handle_memory_delete(
    request: MemoryDeleteRequest,
    store: Arc<dyn TeleologicalMemoryStore>,
) -> Result<MemoryDeleteResponse, McpError>;

/// Batch store multiple memories
pub async fn handle_memory_batch_store(
    request: MemoryBatchStoreRequest,
    store: Arc<dyn TeleologicalMemoryStore>,
) -> Result<MemoryBatchStoreResponse, McpError>;
```

### Request/Response JSON Schemas

#### MemoryStoreRequest

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["content", "embeddings"],
  "properties": {
    "content": {
      "type": "string",
      "description": "Raw content to store"
    },
    "embeddings": {
      "type": "object",
      "description": "12-array embeddings from all embedders",
      "required": ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10", "e11", "e12"],
      "properties": {
        "e1": { "type": "array", "items": { "type": "number" }, "minItems": 1024, "maxItems": 1024 },
        "e2": { "type": "array", "items": { "type": "number" }, "minItems": 512, "maxItems": 512 },
        "e3": { "type": "array", "items": { "type": "number" }, "minItems": 512, "maxItems": 512 },
        "e4": { "type": "array", "items": { "type": "number" }, "minItems": 512, "maxItems": 512 },
        "e5": {
          "type": "object",
          "required": ["query", "doc"],
          "properties": {
            "query": { "type": "array", "items": { "type": "number" }, "minItems": 768, "maxItems": 768 },
            "doc": { "type": "array", "items": { "type": "number" }, "minItems": 768, "maxItems": 768 }
          }
        },
        "e6": {
          "type": "object",
          "required": ["indices", "values"],
          "properties": {
            "indices": { "type": "array", "items": { "type": "integer", "minimum": 0, "maximum": 29999 } },
            "values": { "type": "array", "items": { "type": "number" } }
          }
        },
        "e7": { "type": "array", "items": { "type": "number" }, "minItems": 1536, "maxItems": 1536 },
        "e8": { "type": "array", "items": { "type": "number" }, "minItems": 384, "maxItems": 384 },
        "e9": { "type": "array", "items": { "type": "number" }, "minItems": 1024, "maxItems": 1024 },
        "e10": { "type": "array", "items": { "type": "number" }, "minItems": 768, "maxItems": 768 },
        "e11": { "type": "array", "items": { "type": "number" }, "minItems": 384, "maxItems": 384 },
        "e12": { "type": "array", "items": { "type": "array", "items": { "type": "number" }, "minItems": 128, "maxItems": 128 } }
      }
    },
    "purpose_vector": {
      "type": "array",
      "items": { "type": "number" },
      "minItems": 12,
      "maxItems": 12,
      "description": "12D alignment to North Star per embedding space"
    },
    "johari_quadrants": {
      "type": "array",
      "items": { "type": "string", "enum": ["open", "hidden", "blind", "unknown"] },
      "minItems": 12,
      "maxItems": 12,
      "description": "Per-embedder Johari classification"
    },
    "metadata": {
      "type": "object",
      "properties": {
        "source_type": { "type": "string" },
        "tags": { "type": "array", "items": { "type": "string" } }
      }
    }
  }
}
```

#### MemoryStoreResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["id", "storage_size_bytes", "embedder_dimensions"],
  "properties": {
    "id": { "type": "string", "format": "uuid" },
    "storage_size_bytes": { "type": "integer" },
    "embedder_dimensions": {
      "type": "array",
      "items": { "type": "integer" },
      "minItems": 12,
      "maxItems": 12
    },
    "north_star_alignment": { "type": "number" },
    "created_at": { "type": "string", "format": "date-time" }
  }
}
```

#### MemoryRetrieveResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["id", "embeddings", "purpose_vector", "johari"],
  "properties": {
    "id": { "type": "string", "format": "uuid" },
    "embeddings": {
      "type": "object",
      "description": "Complete 12-array SemanticFingerprint"
    },
    "purpose_vector": {
      "type": "object",
      "required": ["alignments", "dominant_embedder", "coherence"],
      "properties": {
        "alignments": { "type": "array", "items": { "type": "number" }, "minItems": 12, "maxItems": 12 },
        "dominant_embedder": { "type": "integer", "minimum": 1, "maximum": 12 },
        "coherence": { "type": "number" }
      }
    },
    "johari": {
      "type": "object",
      "required": ["quadrants", "confidence"],
      "properties": {
        "quadrants": { "type": "array", "items": { "type": "string" }, "minItems": 12, "maxItems": 12 },
        "confidence": { "type": "array", "items": { "type": "number" }, "minItems": 12, "maxItems": 12 }
      }
    },
    "theta_to_north_star": { "type": "number" },
    "purpose_evolution_count": { "type": "integer" },
    "created_at": { "type": "string", "format": "date-time" },
    "last_updated": { "type": "string", "format": "date-time" },
    "access_count": { "type": "integer" }
  }
}
```

### Error Handling

```rust
/// MCP Error types for memory operations
#[derive(Debug, Clone, Serialize)]
pub enum McpMemoryError {
    /// Embedding dimension mismatch
    DimensionMismatch {
        embedder: usize,
        expected: usize,
        actual: usize,
    },
    /// Missing required embedder
    MissingEmbedder { embedder: usize },
    /// Invalid sparse indices
    InvalidSparseIndices { max_allowed: usize, found: usize },
    /// Memory not found
    NotFound { id: Uuid },
    /// Storage error
    StorageError { message: String },
    /// Serialization error
    SerializationError { message: String },
}

impl McpMemoryError {
    /// Convert to MCP error response with context
    pub fn to_mcp_error(&self) -> McpError {
        McpError {
            code: self.error_code(),
            message: self.to_string(),
            data: Some(serde_json::to_value(self).unwrap()),
        }
    }
}
```

## Implementation Requirements

### Prerequisites

- [ ] TASK-F001 complete (SemanticFingerprint)
- [ ] TASK-F002 complete (TeleologicalFingerprint)
- [ ] TASK-F008 complete (TeleologicalMemoryStore trait)
- [ ] TASK-L008 complete (Retrieval pipeline for internal use)

### Scope

#### In Scope

- Update `handle_memory_store` for 12-array embeddings
- Update `handle_memory_retrieve` to return full fingerprint
- Update `handle_memory_delete` (minimal changes)
- Add `handle_memory_batch_store` for efficiency
- Dimension validation for all 12 embedders
- Sparse vector validation (E6)
- Late-interaction validation (E12)
- Comprehensive error responses with context

#### Out of Scope

- Search handlers (TASK-S002)
- Purpose/goal handlers (TASK-S003)
- Johari handlers (TASK-S004)

### Constraints

- Fail fast on dimension mismatch
- NO silent fallbacks to defaults
- NO backwards compatibility with Vec<f32> API
- All errors include full context for debugging

## Definition of Done

### Implementation Checklist

- [ ] `handle_memory_store` accepts 12-array embeddings
- [ ] All embedder dimensions validated (E1-E12)
- [ ] Sparse vector indices validated (0-29999)
- [ ] `handle_memory_retrieve` returns complete fingerprint
- [ ] `handle_memory_batch_store` implemented
- [ ] `McpMemoryError` with detailed context
- [ ] All fusion-related handler code removed
- [ ] JSON schemas documented in API docs

### Testing Requirements

All tests MUST use REAL embedding data from actual models, NOT mock vectors.

```rust
#[cfg(test)]
mod tests {
    use crate::test_fixtures::load_real_embeddings;

    #[tokio::test]
    async fn test_memory_store_valid_fingerprint() {
        // Load REAL embeddings from test fixtures
        let embeddings = load_real_embeddings("sample_text_embeddings.json");

        let request = MemoryStoreRequest {
            content: "Test content".into(),
            embeddings,
            purpose_vector: Some(compute_purpose_vector(&embeddings)),
            johari_quadrants: None,
            metadata: None,
        };

        let response = handle_memory_store(request, store.clone()).await.unwrap();
        assert!(response.storage_size_bytes > 40_000); // ~46KB expected
        assert_eq!(response.embedder_dimensions, [1024, 512, 512, 512, 768, 0, 1536, 384, 1024, 768, 384, 128]);
    }

    #[tokio::test]
    async fn test_memory_store_dimension_mismatch() {
        let mut embeddings = load_real_embeddings("sample_text_embeddings.json");
        embeddings.e1 = vec![0.0; 512]; // Wrong dimension

        let request = MemoryStoreRequest {
            content: "Test content".into(),
            embeddings,
            ..Default::default()
        };

        let result = handle_memory_store(request, store.clone()).await;
        match result {
            Err(McpError { code, data, .. }) => {
                assert!(code.contains("dimension_mismatch"));
                let err_data: McpMemoryError = serde_json::from_value(data.unwrap()).unwrap();
                assert!(matches!(err_data, McpMemoryError::DimensionMismatch { embedder: 0, expected: 1024, actual: 512 }));
            }
            Ok(_) => panic!("Should have failed with dimension mismatch"),
        }
    }

    #[tokio::test]
    async fn test_memory_retrieve_full_fingerprint() {
        // Store first
        let store_response = /* ... */;

        let request = MemoryRetrieveRequest { id: store_response.id };
        let response = handle_memory_retrieve(request, store.clone()).await.unwrap();

        // Verify all 12 embeddings present
        assert_eq!(response.embeddings.e1.len(), 1024);
        assert_eq!(response.embeddings.e2.len(), 512);
        // ... verify all 12

        // Verify purpose vector
        assert_eq!(response.purpose_vector.alignments.len(), 12);
        assert!(response.purpose_vector.dominant_embedder >= 1 && response.purpose_vector.dominant_embedder <= 12);
    }

    #[tokio::test]
    async fn test_memory_store_sparse_validation() {
        let mut embeddings = load_real_embeddings("sample_text_embeddings.json");
        embeddings.e6.indices = vec![30001]; // Out of bounds
        embeddings.e6.values = vec![1.0];

        let result = handle_memory_store(/* ... */).await;
        assert!(matches!(
            result.unwrap_err(),
            McpError { code, .. } if code.contains("invalid_sparse")
        ));
    }
}
```

### Verification Commands

```bash
# Run MCP handler tests
cargo test -p context-graph-mcp memory_handlers

# Verify no fusion handlers remain
rg -l "fuse|fusion" crates/context-graph-mcp/src/handlers/

# Verify dimension validation
cargo test -p context-graph-mcp dimension_mismatch

# Integration test with real embeddings
cargo test -p context-graph-mcp --features integration_tests memory_integration
```

## Files to Create

| File | Description |
|------|-------------|
| `crates/context-graph-mcp/src/handlers/memory.rs` | Updated memory handlers |
| `crates/context-graph-mcp/src/schemas/memory_store.json` | JSON schema for store |
| `crates/context-graph-mcp/src/schemas/memory_retrieve.json` | JSON schema for retrieve |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-mcp/src/handlers/mod.rs` | Export new memory handlers, remove fusion |
| `crates/context-graph-mcp/src/error.rs` | Add McpMemoryError type |
| `crates/context-graph-mcp/src/router.rs` | Update handler registration |

## Files to Delete

| File | Reason |
|------|--------|
| `crates/context-graph-mcp/src/handlers/fused_memory.rs` | Legacy fusion handler |
| `crates/context-graph-mcp/src/handlers/vector_store.rs` | Legacy single-vector |

## Traceability

| Requirement | Source | Coverage |
|-------------|--------|----------|
| FR-101 | FUNC-SPEC-001 | 12-array storage via MCP |
| FR-301 | FUNC-SPEC-001 | Primary storage integration |
| FR-603 | FUNC-SPEC-001 | Fail fast with context |
| TS-101 | TECH-SPEC-001 | SemanticFingerprint in handlers |

---

*Task created: 2026-01-04*
*Layer: Surface*
*Priority: P0 - Core API surface*
