# Task 07: Implement Curation Tool Handlers

## Metadata
- **Task ID**: TASK-GAP-007
- **Phase**: 2 (MCP Infrastructure)
- **Priority**: High
- **Complexity**: Medium
- **Dependencies**: task05 (TASK-GAP-005 - DTOs COMPLETED in `curation_dtos.rs`)
- **Status**: **COMPLETED** (2026-01-18)

## Completion Summary

### Implementation Complete
| File | Status | Notes |
|------|--------|-------|
| `crates/context-graph-mcp/src/handlers/tools/curation_dtos.rs` | COMPLETE | 680+ lines, DTOs + 30 tests + validation methods |
| `crates/context-graph-mcp/src/tools/names.rs` | COMPLETE | `FORGET_CONCEPT`, `BOOST_IMPORTANCE` constants (dead_code removed) |
| `crates/context-graph-mcp/src/handlers/tools/helpers.rs` | COMPLETE | `tool_result_with_pulse()`, `tool_error_with_pulse()` |
| `crates/context-graph-mcp/src/handlers/tools/mod.rs` | COMPLETE | `mod curation_tools;` added |
| `crates/context-graph-mcp/src/handlers/tools/dispatch.rs` | COMPLETE | `FORGET_CONCEPT` and `BOOST_IMPORTANCE` dispatch cases added |
| `crates/context-graph-mcp/src/handlers/tools/curation_tools.rs` | COMPLETE | 140 lines, both handlers implemented |
| `crates/context-graph-mcp/src/handlers/tests/curation_tools_fsv.rs` | COMPLETE | 13 FSV tests for full state verification |

### Test Results
- **Total Tests**: 45 curation-related tests
  - 30 DTO unit tests (curation_dtos.rs)
  - 2 handler unit tests (curation_tools.rs)
  - 13 Full State Verification tests (curation_tools_fsv.rs)
- **All tests PASS**

### Constitution Compliance Verified
| Requirement | Implementation | Status |
|-------------|----------------|--------|
| SEC-06: 30-day recovery | `SOFT_DELETE_RECOVERY_DAYS = 30`, `compute_recovery_deadline()` | COMPLIANT |
| BR-MCP-001: soft_delete defaults to true | `default_soft_delete() -> true`, `Default` impl | COMPLIANT |
| BR-MCP-002: clamp to [0.0, 1.0] | `MIN_IMPORTANCE/MAX_IMPORTANCE` constants, `clamp()` | COMPLIANT |
| AP-10: No NaN/Infinity | Explicit check in `BoostImportanceRequest::validate()` | COMPLIANT |

---

## Original Specification (for reference)

## Current State Assessment (2026-01-18)

### What Exists
| File | Status | Notes |
|------|--------|-------|
| `crates/context-graph-mcp/src/handlers/tools/curation_dtos.rs` | COMPLETE | 680+ lines, DTOs + tests + validation methods |
| `crates/context-graph-mcp/src/tools/names.rs` | COMPLETE | `FORGET_CONCEPT`, `BOOST_IMPORTANCE` constants defined (marked `#[allow(dead_code)]`) |
| `crates/context-graph-mcp/src/handlers/tools/helpers.rs` | COMPLETE | `tool_result_with_pulse()`, `tool_error_with_pulse()` implemented |
| `crates/context-graph-mcp/src/handlers/tools/mod.rs` | PARTIAL | Lists `curation_tools` in docstring, has `pub mod curation_dtos;`, but `mod curation_tools;` NOT YET ADDED |
| `crates/context-graph-mcp/src/handlers/tools/dispatch.rs` | PARTIAL | Missing `FORGET_CONCEPT` and `BOOST_IMPORTANCE` dispatch cases |
| `crates/context-graph-mcp/src/handlers/tools/curation_tools.rs` | MISSING | **THIS IS WHAT NEEDS TO BE CREATED** |

### Key Interfaces (Verified)

#### TeleologicalMemoryStore Trait
**Location**: `crates/context-graph-core/src/traits/teleological_memory_store/store.rs`
```rust
async fn retrieve(&self, id: Uuid) -> CoreResult<Option<TeleologicalFingerprint>>;
async fn update(&self, fingerprint: TeleologicalFingerprint) -> CoreResult<bool>;
async fn delete(&self, id: Uuid, soft: bool) -> CoreResult<bool>;  // soft=true for SEC-06
async fn count(&self) -> CoreResult<usize>;
```

#### TeleologicalFingerprint Struct
**Location**: `crates/context-graph-core/src/types/fingerprint/teleological/types.rs`
```rust
pub struct TeleologicalFingerprint {
    pub id: Uuid,
    pub alignment_score: f32,  // THIS IS THE IMPORTANCE FIELD (per TASK-P0-001)
    pub semantic: SemanticFingerprint,
    pub purpose_vector: PurposeVector,
    pub purpose_evolution: Vec<PurposeSnapshot>,
    pub content_hash: [u8; 32],
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub access_count: u64,
}
```

#### Handlers Struct Members
**Location**: `crates/context-graph-mcp/src/handlers/core/handlers.rs`
```rust
pub(in crate::handlers) teleological_store: Arc<dyn TeleologicalMemoryStore>,
pub(in crate::handlers) utl_processor: Arc<dyn UtlProcessor>,
pub(in crate::handlers) multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
pub(in crate::handlers) goal_hierarchy: Arc<RwLock<GoalHierarchy>>,
pub(in crate::handlers) layer_status_provider: Arc<dyn LayerStatusProvider>,
```

#### Helper Methods
**Location**: `crates/context-graph-mcp/src/handlers/tools/helpers.rs`
```rust
pub(crate) fn tool_result_with_pulse(&self, id: Option<JsonRpcId>, data: serde_json::Value) -> JsonRpcResponse;
pub(crate) fn tool_error_with_pulse(&self, id: Option<JsonRpcId>, message: &str) -> JsonRpcResponse;
```

#### Curation DTOs (ALREADY IMPLEMENTED)
**Location**: `crates/context-graph-mcp/src/handlers/tools/curation_dtos.rs`
```rust
// Request DTOs with validation
pub struct ForgetConceptRequest { pub node_id: String, pub soft_delete: bool }
pub struct BoostImportanceRequest { pub node_id: String, pub delta: f32 }

// Response DTOs with factory methods
pub struct ForgetConceptResponse { pub forgotten_id: Uuid, pub soft_deleted: bool, pub recoverable_until: Option<DateTime<Utc>> }
pub struct BoostImportanceResponse { pub node_id: Uuid, pub old_importance: f32, pub new_importance: f32, pub clamped: bool }

// Factory methods available:
ForgetConceptResponse::soft_deleted(id: Uuid) -> Self
ForgetConceptResponse::hard_deleted(id: Uuid) -> Self
BoostImportanceResponse::new(node_id: Uuid, old_importance: f32, delta: f32) -> Self

// Validation methods:
ForgetConceptRequest::validate() -> Result<Uuid, String>
BoostImportanceRequest::validate() -> Result<Uuid, String>
BoostImportanceRequest::apply_delta(current: f32) -> (f32, bool)  // Returns (new_value, was_clamped)
```

## Objective

Implement the 2 curation-related MCP tool handlers: `forget_concept` and `boost_importance` in a new file `curation_tools.rs`.

**Constitution Requirements**:
- SEC-06: Soft delete 30-day recovery (default behavior)
- BR-MCP-001: forget_concept uses soft delete by default
- BR-MCP-002: boost_importance clamps final value to [0.0, 1.0]

## Files to Create/Modify

### 1. CREATE: `crates/context-graph-mcp/src/handlers/tools/curation_tools.rs`

```rust
//! Curation tool handlers.
//!
//! Per PRD Section 10.3, implements:
//! - forget_concept: Soft-delete a memory (30-day recovery per SEC-06)
//! - boost_importance: Adjust memory importance score
//!
//! Constitution Compliance:
//! - SEC-06: Soft delete 30-day recovery
//! - BR-MCP-001: forget_concept uses soft delete by default
//! - BR-MCP-002: boost_importance clamps final value to [0.0, 1.0]

use chrono::Utc;
use tracing::{debug, error, info, warn};

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::super::Handlers;
use super::curation_dtos::{
    BoostImportanceRequest, BoostImportanceResponse, ForgetConceptRequest, ForgetConceptResponse,
};

impl Handlers {
    /// Handle forget_concept tool call.
    ///
    /// Soft-deletes a memory with 30-day recovery window per SEC-06.
    ///
    /// # Arguments
    /// * `id` - JSON-RPC request ID
    /// * `arguments` - Tool arguments (node_id, soft_delete)
    ///
    /// # Returns
    /// JsonRpcResponse with ForgetConceptResponse
    ///
    /// # Constitution Compliance
    /// - SEC-06: 30-day recovery for soft delete
    /// - BR-MCP-001: soft_delete defaults to true
    pub(crate) async fn call_forget_concept(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling forget_concept");

        // Parse request
        let request: ForgetConceptRequest = match serde_json::from_value(arguments) {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, "forget_concept: Failed to parse request");
                return self.tool_error_with_pulse(id, &format!("Invalid params: {}", e));
            }
        };

        // Validate and parse UUID using DTO's validate method
        let node_id = match request.validate() {
            Ok(uuid) => uuid,
            Err(validation_error) => {
                error!(error = %validation_error, "forget_concept: Validation failed");
                return self.tool_error_with_pulse(id, &format!("Invalid params: {}", validation_error));
            }
        };

        // Check if memory exists - FAIL FAST if not found
        let exists = match self.teleological_store.retrieve(node_id).await {
            Ok(Some(_)) => true,
            Ok(None) => {
                warn!(node_id = %node_id, "forget_concept: Memory not found");
                return JsonRpcResponse::error(
                    id,
                    error_codes::FINGERPRINT_NOT_FOUND,
                    format!("Memory {} not found", node_id),
                );
            }
            Err(e) => {
                error!(error = %e, node_id = %node_id, "forget_concept: Failed to check memory existence");
                return self.tool_error_with_pulse(
                    id,
                    &format!("Storage error: Failed to check memory: {}", e),
                );
            }
        };

        debug!(node_id = %node_id, exists = exists, soft_delete = request.soft_delete, "forget_concept: Memory exists, proceeding with delete");

        // Perform delete operation
        let delete_result = self.teleological_store.delete(node_id, request.soft_delete).await;

        match delete_result {
            Ok(true) => {
                // Build response using DTO factory methods
                let response = if request.soft_delete {
                    info!(node_id = %node_id, "forget_concept: Soft deleted memory (30-day recovery per SEC-06)");
                    ForgetConceptResponse::soft_deleted(node_id)
                } else {
                    warn!(node_id = %node_id, "forget_concept: HARD deleted memory (no recovery)");
                    ForgetConceptResponse::hard_deleted(node_id)
                };

                self.tool_result_with_pulse(
                    id,
                    serde_json::to_value(response).expect("ForgetConceptResponse should serialize"),
                )
            }
            Ok(false) => {
                // Store returned false - memory not found (race condition)
                warn!(node_id = %node_id, "forget_concept: Delete returned false - memory may have been deleted concurrently");
                JsonRpcResponse::error(
                    id,
                    error_codes::FINGERPRINT_NOT_FOUND,
                    format!("Memory {} not found (may have been deleted concurrently)", node_id),
                )
            }
            Err(e) => {
                error!(error = %e, node_id = %node_id, "forget_concept: Delete operation failed");
                self.tool_error_with_pulse(
                    id,
                    &format!("Storage error: Delete failed: {}", e),
                )
            }
        }
    }

    /// Handle boost_importance tool call.
    ///
    /// Adjusts memory importance by delta, clamping to [0.0, 1.0].
    ///
    /// # Arguments
    /// * `id` - JSON-RPC request ID
    /// * `arguments` - Tool arguments (node_id, delta)
    ///
    /// # Returns
    /// JsonRpcResponse with BoostImportanceResponse
    ///
    /// # Constitution Compliance
    /// - BR-MCP-002: Importance clamped to [0.0, 1.0]
    /// - AP-10: No NaN/Infinity in values
    pub(crate) async fn call_boost_importance(
        &self,
        id: Option<JsonRpcId>,
        arguments: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling boost_importance");

        // Parse request
        let request: BoostImportanceRequest = match serde_json::from_value(arguments) {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, "boost_importance: Failed to parse request");
                return self.tool_error_with_pulse(id, &format!("Invalid params: {}", e));
            }
        };

        // Validate delta range and UUID using DTO's validate method
        // This checks: NaN/Infinity (AP-10), delta range [-1.0, 1.0], UUID format
        let node_id = match request.validate() {
            Ok(uuid) => uuid,
            Err(validation_error) => {
                error!(error = %validation_error, "boost_importance: Validation failed");
                return self.tool_error_with_pulse(id, &format!("Invalid params: {}", validation_error));
            }
        };

        debug!(node_id = %node_id, delta = request.delta, "boost_importance: Processing request");

        // Get current memory to read importance (alignment_score)
        let mut fingerprint = match self.teleological_store.retrieve(node_id).await {
            Ok(Some(fp)) => fp,
            Ok(None) => {
                warn!(node_id = %node_id, "boost_importance: Memory not found");
                return JsonRpcResponse::error(
                    id,
                    error_codes::FINGERPRINT_NOT_FOUND,
                    format!("Memory {} not found", node_id),
                );
            }
            Err(e) => {
                error!(error = %e, node_id = %node_id, "boost_importance: Failed to retrieve memory");
                return self.tool_error_with_pulse(
                    id,
                    &format!("Storage error: Failed to retrieve memory: {}", e),
                );
            }
        };

        // Get current importance (alignment_score field per TASK-P0-001)
        let old_importance = fingerprint.alignment_score;

        // Calculate new importance with clamping using DTO helper
        let (new_importance, clamped) = request.apply_delta(old_importance);

        debug!(
            node_id = %node_id,
            old_importance = old_importance,
            delta = request.delta,
            new_importance = new_importance,
            clamped = clamped,
            "boost_importance: Computed new importance"
        );

        // Update fingerprint with new importance
        fingerprint.alignment_score = new_importance;
        fingerprint.last_updated = Utc::now();

        // Persist the updated fingerprint
        match self.teleological_store.update(fingerprint).await {
            Ok(true) => {
                info!(
                    node_id = %node_id,
                    old = old_importance,
                    new = new_importance,
                    clamped = clamped,
                    "boost_importance: Updated memory importance"
                );

                // Build response using DTO factory method
                let response = BoostImportanceResponse::new(node_id, old_importance, request.delta);

                self.tool_result_with_pulse(
                    id,
                    serde_json::to_value(response).expect("BoostImportanceResponse should serialize"),
                )
            }
            Ok(false) => {
                // Update returned false - memory not found (race condition)
                warn!(node_id = %node_id, "boost_importance: Update returned false - memory may have been deleted concurrently");
                JsonRpcResponse::error(
                    id,
                    error_codes::FINGERPRINT_NOT_FOUND,
                    format!("Memory {} not found (may have been deleted concurrently)", node_id),
                )
            }
            Err(e) => {
                error!(error = %e, node_id = %node_id, "boost_importance: Update operation failed");
                self.tool_error_with_pulse(
                    id,
                    &format!("Storage error: Update failed: {}", e),
                )
            }
        }
    }
}
```

### 2. MODIFY: `crates/context-graph-mcp/src/handlers/tools/mod.rs`

Add the module declaration. Current file has `pub mod curation_dtos;` but needs `mod curation_tools;`:

```rust
//! MCP tool call handlers.
//!
//! PRD v6 Section 10 MCP Tools:
//! - inject_context, store_memory, search_graph (memory_tools.rs)
//! - get_memetic_status (status_tools.rs)
//! - trigger_consolidation (consolidation.rs)
//! - merge_concepts (../merge.rs)
//! - get_topic_portfolio, get_topic_stability, detect_topics, get_divergence_alerts (topic_tools.rs)
//! - forget_concept, boost_importance (curation_tools.rs)

mod consolidation;
mod curation_tools;  // ADD THIS LINE
mod dispatch;
mod helpers;
mod memory_tools;
mod status_tools;
mod topic_tools;

// DTOs for PRD v6 gap tools (TASK-GAP-005)
pub mod curation_dtos;
pub mod topic_dtos;
```

### 3. MODIFY: `crates/context-graph-mcp/src/handlers/tools/dispatch.rs`

Add dispatch cases for the new tools. Add to the match statement after `MERGE_CONCEPTS`:

```rust
// ========== CURATION TOOLS (PRD Section 10.3) ==========
tool_names::MERGE_CONCEPTS => self.call_merge_concepts(id, arguments).await,
tool_names::FORGET_CONCEPT => self.call_forget_concept(id, arguments).await,  // ADD
tool_names::BOOST_IMPORTANCE => self.call_boost_importance(id, arguments).await,  // ADD
```

### 4. MODIFY: `crates/context-graph-mcp/src/tools/names.rs`

Remove `#[allow(dead_code)]` from the constants since they're now used:

```rust
// ========== CURATION TOOLS (PRD Section 10.3) ==========
pub const MERGE_CONCEPTS: &str = "merge_concepts";
pub const FORGET_CONCEPT: &str = "forget_concept";  // Remove #[allow(dead_code)]
pub const BOOST_IMPORTANCE: &str = "boost_importance";  // Remove #[allow(dead_code)]
```

## Implementation Steps

### Step 1: Create curation_tools.rs
Create the file with the exact content shown above.

### Step 2: Update mod.rs
Add `mod curation_tools;` after `mod consolidation;`.

### Step 3: Update dispatch.rs
Add the two dispatch cases for FORGET_CONCEPT and BOOST_IMPORTANCE.

### Step 4: Update names.rs
Remove the `#[allow(dead_code)]` attributes from the constants.

### Step 5: Build and Verify
```bash
cargo check -p context-graph-mcp
cargo clippy -p context-graph-mcp -- -D warnings
cargo test -p context-graph-mcp -- curation
```

## Full State Verification Requirements

After completing the logic, you MUST perform Full State Verification (FSV). Do not rely on return values alone.

### Source of Truth
- **forget_concept**: The `TeleologicalMemoryStore` - verify memory is soft-deleted or hard-deleted
- **boost_importance**: The `TeleologicalFingerprint.alignment_score` field in storage

### Execute & Inspect Protocol

For each handler, after calling through MCP:
1. Call the handler with synthetic test data
2. Immediately perform a SEPARATE read operation on the store to verify
3. Print/log the actual state before and after

### Required FSV Tests

Create test file: `crates/context-graph-mcp/src/handlers/tests/curation_tools_fsv.rs`

```rust
//! Full State Verification tests for curation tools.
//!
//! These tests verify that operations actually persist to storage,
//! not just that handlers return success responses.

use super::{create_test_handlers_with_rocksdb_store_access, extract_mcp_tool_data};
use crate::protocol::JsonRpcId;
use context_graph_core::types::fingerprint::TeleologicalFingerprint;
use serde_json::json;
use uuid::Uuid;

/// FSV Test 1: forget_concept soft delete actually marks memory as deleted
#[tokio::test]
async fn test_fsv_forget_concept_soft_delete() {
    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // SETUP: Create and store a test fingerprint
    let test_id = Uuid::new_v4();
    let fingerprint = create_test_fingerprint(test_id);
    store.store(fingerprint).await.expect("setup: store fingerprint");

    // BEFORE STATE: Verify memory exists
    let before = store.retrieve(test_id).await.expect("retrieve before");
    assert!(before.is_some(), "FSV BEFORE: Memory MUST exist before delete");
    println!("FSV BEFORE: Memory {} exists: true", test_id);

    // EXECUTE: Call forget_concept via handler
    let response = handlers.call_forget_concept(
        Some(JsonRpcId::Number(1)),
        json!({"node_id": test_id.to_string(), "soft_delete": true}),
    ).await;

    // RESPONSE CHECK
    assert!(response.error.is_none(), "Handler should not return error");
    let data = extract_mcp_tool_data(response.result.as_ref().unwrap());
    assert_eq!(data["soft_deleted"], true);
    println!("FSV RESPONSE: soft_deleted={}", data["soft_deleted"]);

    // AFTER STATE: Verify memory is actually soft-deleted in storage
    let after = store.retrieve(test_id).await.expect("retrieve after");
    assert!(after.is_none(), "FSV AFTER: Memory MUST be inaccessible after soft delete");
    println!("FSV AFTER: Memory {} accessible: false (soft deleted)", test_id);

    // EVIDENCE: Print storage state
    let count = store.count().await.expect("count");
    println!("FSV EVIDENCE: Store count after soft delete: {}", count);
}

/// FSV Test 2: boost_importance actually updates alignment_score in storage
#[tokio::test]
async fn test_fsv_boost_importance_updates_storage() {
    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // SETUP: Create fingerprint with known importance
    let test_id = Uuid::new_v4();
    let mut fingerprint = create_test_fingerprint(test_id);
    fingerprint.alignment_score = 0.5;
    store.store(fingerprint).await.expect("setup: store fingerprint");

    // BEFORE STATE: Verify initial importance
    let before = store.retrieve(test_id).await.expect("retrieve before").unwrap();
    assert!((before.alignment_score - 0.5).abs() < f32::EPSILON, "FSV BEFORE: alignment_score must be 0.5");
    println!("FSV BEFORE: alignment_score = {}", before.alignment_score);

    // EXECUTE: Call boost_importance via handler
    let response = handlers.call_boost_importance(
        Some(JsonRpcId::Number(1)),
        json!({"node_id": test_id.to_string(), "delta": 0.3}),
    ).await;

    // RESPONSE CHECK
    assert!(response.error.is_none(), "Handler should not return error");
    let data = extract_mcp_tool_data(response.result.as_ref().unwrap());
    assert!((data["old_importance"].as_f64().unwrap() - 0.5).abs() < 0.001);
    assert!((data["new_importance"].as_f64().unwrap() - 0.8).abs() < 0.001);
    println!("FSV RESPONSE: old={}, new={}, clamped={}",
             data["old_importance"], data["new_importance"], data["clamped"]);

    // AFTER STATE: Verify importance actually changed in storage
    let after = store.retrieve(test_id).await.expect("retrieve after").unwrap();
    assert!((after.alignment_score - 0.8).abs() < f32::EPSILON,
            "FSV AFTER: alignment_score MUST be 0.8 in storage, got {}", after.alignment_score);
    println!("FSV AFTER: alignment_score = {} (verified from storage)", after.alignment_score);

    // EVIDENCE: Show the actual stored value
    println!("FSV EVIDENCE: Direct storage read confirms alignment_score = {}", after.alignment_score);
}

/// FSV Test 3: boost_importance clamping at upper bound
#[tokio::test]
async fn test_fsv_boost_importance_clamps_at_max() {
    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // SETUP: Create fingerprint at 0.9 importance
    let test_id = Uuid::new_v4();
    let mut fingerprint = create_test_fingerprint(test_id);
    fingerprint.alignment_score = 0.9;
    store.store(fingerprint).await.expect("setup: store fingerprint");

    // BEFORE STATE
    println!("FSV BEFORE: alignment_score = 0.9");

    // EXECUTE: Boost by 0.5 (should clamp to 1.0, not 1.4)
    let response = handlers.call_boost_importance(
        Some(JsonRpcId::Number(1)),
        json!({"node_id": test_id.to_string(), "delta": 0.5}),
    ).await;

    // RESPONSE CHECK
    let data = extract_mcp_tool_data(response.result.as_ref().unwrap());
    assert_eq!(data["clamped"], true, "FSV: clamped flag must be true");
    assert!((data["new_importance"].as_f64().unwrap() - 1.0).abs() < 0.001);
    println!("FSV RESPONSE: clamped=true, new_importance=1.0");

    // AFTER STATE: Verify storage has clamped value
    let after = store.retrieve(test_id).await.expect("retrieve after").unwrap();
    assert!((after.alignment_score - 1.0).abs() < f32::EPSILON,
            "FSV AFTER: alignment_score MUST be clamped to 1.0, got {}", after.alignment_score);
    println!("FSV AFTER: alignment_score = {} (clamped at max per BR-MCP-002)", after.alignment_score);
}

/// Helper to create a test fingerprint
fn create_test_fingerprint(id: Uuid) -> TeleologicalFingerprint {
    use chrono::Utc;
    use context_graph_core::types::fingerprint::{PurposeVector, SemanticFingerprint};

    TeleologicalFingerprint {
        id,
        semantic: SemanticFingerprint::zeroed(),
        purpose_vector: PurposeVector::neutral(),
        purpose_evolution: vec![],
        alignment_score: 0.5,
        content_hash: [0u8; 32],
        created_at: Utc::now(),
        last_updated: Utc::now(),
        access_count: 0,
    }
}
```

## Edge Case & Boundary Testing

### Edge Cases to Test

| Case | Input | Expected Behavior |
|------|-------|-------------------|
| **Empty UUID** | `node_id: ""` | Error: "Invalid UUID format" |
| **Invalid UUID** | `node_id: "not-a-uuid"` | Error: "Invalid UUID format" |
| **Non-existent memory** | Valid UUID that doesn't exist | Error: FINGERPRINT_NOT_FOUND (-32010) |
| **Delta at +1.0** | `delta: 1.0` with importance 0.5 | new=1.0, clamped=true (from 1.5) |
| **Delta at -1.0** | `delta: -1.0` with importance 0.5 | new=0.0, clamped=true (from -0.5) |
| **Delta = 0.0** | `delta: 0.0` | No change, clamped=false |
| **Delta NaN** | `delta: NaN` | Error: "delta must be a finite number" |
| **Delta Infinity** | `delta: Infinity` | Error: "delta must be a finite number" |
| **Delta > 1.0** | `delta: 1.5` | Error: "delta must be between -1.0 and 1.0" |
| **Delta < -1.0** | `delta: -1.5` | Error: "delta must be between -1.0 and 1.0" |
| **Hard delete** | `soft_delete: false` | Permanent deletion, recoverable_until=null |
| **Concurrent delete** | Delete same ID twice | Second call gets FINGERPRINT_NOT_FOUND |

### Manual Testing Protocol

After implementing, manually test using the MCP server:

```bash
# Start the MCP server
cargo run -p context-graph-mcp --bin context-graph-mcp

# In another terminal, send test requests via netcat or curl
# Test 1: Store a memory first (to have something to delete/boost)
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"store_memory","arguments":{"content":"Test memory for curation","importance":0.5}}}' | nc localhost 8080

# Note the returned ID, then test forget_concept
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"forget_concept","arguments":{"node_id":"<UUID>","soft_delete":true}}}' | nc localhost 8080

# Test boost_importance with another memory
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"boost_importance","arguments":{"node_id":"<UUID>","delta":0.3}}}' | nc localhost 8080
```

## Definition of Done

### Code Requirements
- [x] File `curation_tools.rs` exists at specified path
- [x] `call_forget_concept` implemented with full error handling
- [x] `call_forget_concept` uses `store.delete(id, soft)` (not `.get()`)
- [x] `call_forget_concept` defaults to soft_delete=true per BR-MCP-001
- [x] `call_boost_importance` implemented with full error handling
- [x] `call_boost_importance` uses `store.retrieve()` then `store.update()`
- [x] `call_boost_importance` modifies `fingerprint.alignment_score`
- [x] All handlers use `tool_result_with_pulse()` / `tool_error_with_pulse()`
- [x] Uses `error_codes::FINGERPRINT_NOT_FOUND` for missing memories
- [x] `mod.rs` includes `mod curation_tools;`
- [x] `dispatch.rs` includes both dispatch cases
- [x] `names.rs` has `#[allow(dead_code)]` removed from constants

### Build Requirements
- [x] `cargo check -p context-graph-mcp` passes
- [x] `cargo clippy -p context-graph-mcp -- -D warnings` passes
- [x] No new warnings introduced

### Test Requirements
- [x] FSV tests created in `curation_tools_fsv.rs`
- [x] FSV test for soft delete verifies storage state
- [x] FSV test for boost_importance verifies alignment_score in storage
- [x] FSV test for clamping verifies clamped value in storage
- [x] All edge cases tested (empty UUID, invalid UUID, non-existent, NaN, bounds)
- [x] `cargo test -p context-graph-mcp -- curation` passes

### Verification Commands

```bash
cd /home/cabdru/contextgraph

# Verify file exists
test -f crates/context-graph-mcp/src/handlers/tools/curation_tools.rs && echo "curation_tools.rs exists"

# Verify mod.rs has module declaration
grep "mod curation_tools;" crates/context-graph-mcp/src/handlers/tools/mod.rs && echo "mod.rs updated"

# Verify dispatch.rs has both cases
grep "FORGET_CONCEPT\|BOOST_IMPORTANCE" crates/context-graph-mcp/src/handlers/tools/dispatch.rs | wc -l
# Expected: 2

# Verify names.rs constants are used (no dead_code warning)
cargo clippy -p context-graph-mcp -- -D warnings 2>&1 | grep -c "dead_code.*FORGET_CONCEPT\|dead_code.*BOOST_IMPORTANCE"
# Expected: 0

# Verify compilation
cargo check -p context-graph-mcp

# Verify no clippy warnings
cargo clippy -p context-graph-mcp -- -D warnings

# Run all tests including FSV
cargo test -p context-graph-mcp -- curation

# Verify both handlers exist
grep -c "pub(crate) async fn call_" crates/context-graph-mcp/src/handlers/tools/curation_tools.rs
# Expected: 2
```

## Reference Patterns

Follow existing patterns in:
- `topic_tools.rs` (lines 47-126) - Complete handler pattern with validation
- `memory_tools.rs` - DTO validation and error handling
- `helpers.rs` (lines 37-75) - `tool_result_with_pulse` behavior

## CRITICAL: No Backwards Compatibility

- NO fallbacks for missing data
- NO stub returns - FAIL FAST with proper errors
- NO mock data in tests - use real storage (InMemoryTeleologicalStore or RocksDbTeleologicalStore)
- If something doesn't work, it MUST error with detailed logging
