# TASK-S004: MCP Handlers for Johari Quadrant Operations

```yaml
metadata:
  id: "TASK-S004"
  title: "MCP Handlers for Johari Quadrant Operations"
  layer: "surface"
  priority: "P1"
  estimated_hours: 6
  created: "2026-01-04"
  updated: "2026-01-05"
  status: "completed"
  dependencies:
    - "TASK-F003"  # JohariFingerprint struct - COMPLETED
    - "TASK-L004"  # Johari Transition Manager - COMPLETED
  traces_to:
    - "FR-203"  # JohariFingerprint Per Embedder
```

## Critical Context for Implementation

### What Already Exists (DO NOT RECREATE)

The following components are **fully implemented** and ready to use:

1. **`JohariFingerprint`** (`crates/context-graph-core/src/types/fingerprint/johari/`)
   - `core.rs`: 13-embedder quadrant weights, confidence, transition matrices
   - `analysis.rs`: `find_blind_spots()`, `predict_transition()`, `openness()`, `is_aware()`
   - `classify.rs`: `classify_quadrant()`, `dominant_quadrant()`, `set_quadrant()`
   - Constants: `NUM_EMBEDDERS = 13`, `OPEN_IDX=0`, `HIDDEN_IDX=1`, `BLIND_IDX=2`, `UNKNOWN_IDX=3`

2. **`JohariTransitionManager` Trait** (`crates/context-graph-core/src/johari/manager.rs`)
   - `classify()`: Classify SemanticFingerprint → JohariFingerprint
   - `transition()`: Single embedder transition with validation
   - `transition_batch()`: Atomic multi-embedder transitions
   - `find_by_quadrant()`: Query memories by quadrant pattern
   - `discover_blind_spots()`: Find blind spots from external signals
   - `get_transition_stats()`, `get_transition_history()`

3. **`DefaultJohariManager`** (`crates/context-graph-core/src/johari/default_manager.rs`)
   - Full implementation using `TeleologicalMemoryStore`
   - All-or-nothing batch semantics
   - Validates transitions via `JohariQuadrant::can_transition_to()`

4. **`JohariQuadrant`** (`crates/context-graph-core/src/types/johari/quadrant.rs`)
   - `Open`, `Hidden`, `Blind`, `Unknown` variants
   - `valid_transitions()`, `can_transition_to()`, `transition_to()`
   - `is_self_aware()`, `is_other_aware()`, `default_retrieval_weight()`

5. **`TransitionTrigger`** (`crates/context-graph-core/src/types/johari/transition.rs`)
   - `ExplicitShare`, `SelfRecognition`, `PatternDiscovery`, `Privatize`, `ExternalObservation`, `DreamConsolidation`

6. **`JohariError`** (`crates/context-graph-core/src/johari/error.rs`)
   - `NotFound`, `InvalidTransition`, `InvalidEmbedderIndex`, `StorageError`, `BatchValidationFailed`, `InvalidTrigger`

7. **MCP Handler Infrastructure** (`crates/context-graph-mcp/src/handlers/`)
   - `Handlers` struct in `core.rs` with `dispatch()` method
   - Purpose handlers in `purpose.rs` (TASK-S003) - use as template
   - Full state verification tests in `tests/full_state_verification_purpose.rs`

### Established Patterns (MUST FOLLOW)

#### Handler Pattern (from `purpose.rs`)
```rust
#[instrument(skip(self, params), fields(method = "johari/get_distribution"))]
pub(super) async fn handle_johari_get_distribution(
    &self,
    id: Option<JsonRpcId>,
    params: Option<serde_json::Value>,
) -> JsonRpcResponse {
    let params = match params {
        Some(p) => p,
        None => {
            error!("johari/get_distribution: Missing parameters");
            return JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                "Missing parameters - memory_id required",
            );
        }
    };
    // ... implementation
}
```

#### Error Code Range (from `protocol.rs`)
```rust
// Johari-specific error codes (-32030 to -32039) - TASK-S004
pub const JOHARI_INVALID_EMBEDDER_INDEX: i32 = -32030;
pub const JOHARI_INVALID_QUADRANT: i32 = -32031;
pub const JOHARI_INVALID_SOFT_CLASSIFICATION: i32 = -32032;
pub const JOHARI_TRANSITION_ERROR: i32 = -32033;
pub const JOHARI_BATCH_ERROR: i32 = -32034;
```

---

## Problem Statement

Create MCP handlers exposing `JohariTransitionManager` operations via JSON-RPC. These handlers provide per-embedder quadrant queries, distribution analysis, transition execution, and cross-space blind spot discovery.

## UTL-Johari Quadrant Mapping (from constitution.yaml)

```
         Coherence (ΔC)
             High
              │
   Hidden     │    Open
   (ΔS<0.5,   │   (ΔS<0.5,
    ΔC<0.5)   │    ΔC>0.5)
              │
Low ──────────┼────────── High Entropy (ΔS)
              │
   Blind      │   Unknown
   (ΔS>0.5,   │   (ΔS>0.5,
    ΔC<0.5)   │    ΔC>0.5)
              │
             Low
```

- **Open**: Low entropy (ΔS<0.5), High coherence (ΔC>0.5) → Direct recall
- **Hidden**: Low entropy (ΔS<0.5), Low coherence (ΔC<0.5) → Private knowledge
- **Blind**: High entropy (ΔS>0.5), Low coherence (ΔC<0.5) → Discovery opportunity
- **Unknown**: High entropy (ΔS>0.5), High coherence (ΔC>0.5) → Frontier

---

## Technical Specification

### New File: `crates/context-graph-mcp/src/handlers/johari.rs`

```rust
//! Johari quadrant handlers.
//!
//! TASK-S004: MCP handlers for JohariTransitionManager operations.
//!
//! # Methods
//!
//! - `johari/get_distribution`: Get per-embedder quadrant distribution for a memory
//! - `johari/find_by_quadrant`: Find memories by quadrant for specific embedder
//! - `johari/transition`: Execute validated transition with trigger
//! - `johari/transition_batch`: Atomic multi-embedder transitions
//! - `johari/cross_space_analysis`: Blind spots, learning opportunities
//! - `johari/transition_probabilities`: Get transition matrix for embedder
//!
//! # Error Handling
//!
//! FAIL FAST: All errors return immediately with detailed error codes.
//! NO fallbacks, NO default values, NO mock data.

use serde_json::json;
use tracing::{debug, error, instrument};

use context_graph_core::johari::{
    DefaultJohariManager, JohariTransitionManager, ClassificationContext,
    QuadrantPattern, TimeRange, NUM_EMBEDDERS,
};
use context_graph_core::types::{JohariQuadrant, TransitionTrigger};
use context_graph_core::types::fingerprint::JohariFingerprint;

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::Handlers;
```

### Handler 1: `johari/get_distribution`

**Purpose**: Get complete Johari quadrant distribution for a memory's 13 embedders.

**Request**:
```json
{
  "method": "johari/get_distribution",
  "params": {
    "memory_id": "uuid-string",
    "include_confidence": true,
    "include_transition_predictions": false
  }
}
```

**Response**:
```json
{
  "memory_id": "uuid-string",
  "per_embedder_quadrants": [
    {
      "embedder_index": 0,
      "embedder_name": "E1_semantic",
      "quadrant": "open",
      "soft_classification": { "open": 1.0, "hidden": 0.0, "blind": 0.0, "unknown": 0.0 },
      "confidence": 0.95,
      "predicted_next_quadrant": "hidden"
    },
    // ... 12 more embedders (indices 1-12)
  ],
  "summary": {
    "open_count": 5,
    "hidden_count": 3,
    "blind_count": 2,
    "unknown_count": 3,
    "average_confidence": 0.82
  }
}
```

**Implementation Notes**:
- Retrieve `TeleologicalFingerprint` via `self.teleological_store.retrieve(uuid)`
- Access `fingerprint.johari` for quadrant data
- Use `johari.dominant_quadrant(embedder_idx)` for hard classification
- Use `johari.quadrants[embedder_idx]` for soft weights
- Use `johari.confidence[embedder_idx]` for confidence
- Use `johari.predict_transition(embedder_idx, current)` for predictions

**Error Codes**:
- `-32602` (INVALID_PARAMS): Missing `memory_id`
- `-32010` (FINGERPRINT_NOT_FOUND): UUID not found

### Handler 2: `johari/find_by_quadrant`

**Purpose**: Find memories where a specific embedder is in a target quadrant.

**Request**:
```json
{
  "method": "johari/find_by_quadrant",
  "params": {
    "embedder_index": 0,
    "quadrant": "blind",
    "min_confidence": 0.7,
    "top_k": 100
  }
}
```

**Response**:
```json
{
  "embedder_index": 0,
  "embedder_name": "E1_semantic",
  "quadrant": "blind",
  "memories": [
    {
      "id": "uuid-1",
      "confidence": 0.92,
      "soft_classification": [0.0, 0.0, 0.92, 0.08]
    }
  ],
  "total_count": 42
}
```

**Implementation Notes**:
- Create `QuadrantPattern::Exact` or use `QuadrantPattern::AtLeast`
- Call `johari_manager.find_by_quadrant(pattern, top_k)`
- Filter results by `min_confidence` in post-processing

**Error Codes**:
- `-32030` (JOHARI_INVALID_EMBEDDER_INDEX): `embedder_index >= 13`
- `-32031` (JOHARI_INVALID_QUADRANT): Invalid quadrant string

### Handler 3: `johari/transition`

**Purpose**: Execute a single validated Johari transition.

**Request**:
```json
{
  "method": "johari/transition",
  "params": {
    "memory_id": "uuid-string",
    "embedder_index": 5,
    "to_quadrant": "open",
    "trigger": "dream_consolidation"
  }
}
```

**Response**:
```json
{
  "memory_id": "uuid-string",
  "embedder_index": 5,
  "from_quadrant": "unknown",
  "to_quadrant": "open",
  "trigger": "dream_consolidation",
  "success": true,
  "updated_johari": {
    "quadrants": [[...], ...],
    "confidence": [...]
  }
}
```

**Valid Transition Rules** (from `JohariQuadrant::valid_transitions()`):
- `Open → Hidden` via `Privatize`
- `Hidden → Open` via `ExplicitShare`
- `Blind → Open` via `SelfRecognition`
- `Blind → Hidden` via `SelfRecognition`
- `Unknown → Open` via `DreamConsolidation` or `PatternDiscovery`
- `Unknown → Hidden` via `DreamConsolidation`
- `Unknown → Blind` via `ExternalObservation`

**Implementation Notes**:
- Parse `trigger` string to `TransitionTrigger` enum
- Call `johari_manager.transition(uuid, embedder_idx, to_quadrant, trigger)`
- Return error if transition is invalid (no fallback!)

**Error Codes**:
- `-32033` (JOHARI_TRANSITION_ERROR): Invalid transition or trigger

### Handler 4: `johari/transition_batch`

**Purpose**: Execute multiple transitions atomically (all-or-nothing).

**Request**:
```json
{
  "method": "johari/transition_batch",
  "params": {
    "memory_id": "uuid-string",
    "transitions": [
      { "embedder_index": 0, "to_quadrant": "open", "trigger": "dream_consolidation" },
      { "embedder_index": 5, "to_quadrant": "blind", "trigger": "external_observation" }
    ]
  }
}
```

**Response**:
```json
{
  "memory_id": "uuid-string",
  "success": true,
  "transitions_applied": 2,
  "updated_johari": { ... }
}
```

**Implementation Notes**:
- Convert request to `Vec<(usize, JohariQuadrant, TransitionTrigger)>`
- Call `johari_manager.transition_batch(uuid, transitions)`
- If ANY fails, return error and apply NONE

**Error Codes**:
- `-32034` (JOHARI_BATCH_ERROR): Batch validation failed at index N

### Handler 5: `johari/cross_space_analysis`

**Purpose**: Analyze cross-space patterns (blind spots, learning opportunities).

**Request**:
```json
{
  "method": "johari/cross_space_analysis",
  "params": {
    "memory_ids": ["uuid-1", "uuid-2"],
    "analysis_type": "blind_spots"
  }
}
```

**Response**:
```json
{
  "blind_spots": [
    {
      "memory_id": "uuid-1",
      "aware_space": 0,
      "aware_space_name": "E1_semantic",
      "blind_space": 4,
      "blind_space_name": "E5_causal",
      "description": "Semantic understanding without causal insight",
      "learning_suggestion": "Explore causal relationships via dream consolidation"
    }
  ],
  "learning_opportunities": [
    {
      "memory_id": "uuid-2",
      "unknown_spaces": [7, 8, 9],
      "potential": "high"
    }
  ],
  "quadrant_correlation": {
    "E1_semantic_vs_E5_causal": 0.42
  }
}
```

**Implementation Notes**:
- Use `johari.find_blind_spots()` which compares E1 Open weight × other embedder Blind weight
- For learning opportunities, find memories with >5 Unknown embedders
- Quadrant correlation requires scanning all memories and computing correlation matrix

### Handler 6: `johari/transition_probabilities`

**Purpose**: Get transition probability matrix for an embedder.

**Request**:
```json
{
  "method": "johari/transition_probabilities",
  "params": {
    "embedder_index": 0,
    "memory_id": "uuid-string"
  }
}
```

**Response**:
```json
{
  "embedder_index": 0,
  "embedder_name": "E1_semantic",
  "transition_matrix": {
    "from_open": { "to_open": 0.7, "to_hidden": 0.3, "to_blind": 0.0, "to_unknown": 0.0 },
    "from_hidden": { "to_open": 0.6, "to_hidden": 0.4, "to_blind": 0.0, "to_unknown": 0.0 },
    "from_blind": { "to_open": 0.5, "to_hidden": 0.3, "to_blind": 0.2, "to_unknown": 0.0 },
    "from_unknown": { "to_open": 0.25, "to_hidden": 0.25, "to_blind": 0.25, "to_unknown": 0.25 }
  },
  "sample_size": 150
}
```

**Implementation Notes**:
- Access `fingerprint.johari.transition_probs[embedder_idx]` for the 4x4 matrix
- Each row corresponds to a source quadrant, each column to a target

---

## Files to Modify

### 1. `crates/context-graph-mcp/src/handlers/mod.rs`

Add:
```rust
mod johari;
```

### 2. `crates/context-graph-mcp/src/handlers/core.rs`

Add to `Handlers` struct:
```rust
/// Johari transition manager - manages Johari quadrant transitions.
/// TASK-S004: Required for johari/* handlers.
pub(super) johari_manager: Arc<dyn JohariTransitionManager>,
```

Add to `new()` and `with_shared_hierarchy()`:
```rust
use context_graph_core::johari::DefaultJohariManager;

// Create Johari manager
let johari_manager: Arc<dyn JohariTransitionManager> =
    Arc::new(DefaultJohariManager::new(teleological_store.clone()));
```

Add to `dispatch()`:
```rust
// Johari operations (TASK-S004)
methods::JOHARI_GET_DISTRIBUTION => {
    self.handle_johari_get_distribution(request.id, request.params).await
}
methods::JOHARI_FIND_BY_QUADRANT => {
    self.handle_johari_find_by_quadrant(request.id, request.params).await
}
methods::JOHARI_TRANSITION => {
    self.handle_johari_transition(request.id, request.params).await
}
methods::JOHARI_TRANSITION_BATCH => {
    self.handle_johari_transition_batch(request.id, request.params).await
}
methods::JOHARI_CROSS_SPACE_ANALYSIS => {
    self.handle_johari_cross_space_analysis(request.id, request.params).await
}
methods::JOHARI_TRANSITION_PROBABILITIES => {
    self.handle_johari_transition_probabilities(request.id, request.params).await
}
```

### 3. `crates/context-graph-mcp/src/protocol.rs`

Add to `error_codes` module:
```rust
// Johari-specific error codes (-32030 to -32039) - TASK-S004
/// Invalid embedder index (must be 0-12)
pub const JOHARI_INVALID_EMBEDDER_INDEX: i32 = -32030;
/// Invalid quadrant string (must be open/hidden/blind/unknown)
pub const JOHARI_INVALID_QUADRANT: i32 = -32031;
/// Soft classification weights don't sum to 1.0
pub const JOHARI_INVALID_SOFT_CLASSIFICATION: i32 = -32032;
/// Transition validation failed
pub const JOHARI_TRANSITION_ERROR: i32 = -32033;
/// Batch transition failed (all-or-nothing)
pub const JOHARI_BATCH_ERROR: i32 = -32034;
```

Add to `methods` module:
```rust
// Johari operations (TASK-S004)
/// Get per-embedder Johari quadrant distribution
pub const JOHARI_GET_DISTRIBUTION: &str = "johari/get_distribution";
/// Find memories by quadrant for specific embedder
pub const JOHARI_FIND_BY_QUADRANT: &str = "johari/find_by_quadrant";
/// Execute single Johari transition
pub const JOHARI_TRANSITION: &str = "johari/transition";
/// Execute batch Johari transitions (atomic)
pub const JOHARI_TRANSITION_BATCH: &str = "johari/transition_batch";
/// Cross-space Johari analysis (blind spots, opportunities)
pub const JOHARI_CROSS_SPACE_ANALYSIS: &str = "johari/cross_space_analysis";
/// Get transition probability matrix
pub const JOHARI_TRANSITION_PROBABILITIES: &str = "johari/transition_probabilities";
```

---

## Test File: `crates/context-graph-mcp/src/handlers/tests/full_state_verification_johari.rs`

### Required Test Structure

Tests MUST follow the Full State Verification pattern from `full_state_verification_purpose.rs`:

```rust
//! Full State Verification Tests for Johari Handlers
//!
//! TASK-S004: Comprehensive verification that directly inspects the Source of Truth.
//!
//! ## Verification Methodology
//!
//! 1. Define Source of Truth: InMemoryTeleologicalStore + JohariTransitionManager
//! 2. Execute & Inspect: Run handlers, then directly query stores to verify
//! 3. Edge Case Audit: Test 3+ edge cases with BEFORE/AFTER state logging
//! 4. Evidence of Success: Print actual data residing in the system

use std::sync::Arc;
use context_graph_core::johari::DefaultJohariManager;
use context_graph_core::stubs::InMemoryTeleologicalStore;
// ... additional imports

/// Create test handlers with SHARED access for direct verification.
fn create_verifiable_handlers() -> (
    Handlers,
    Arc<InMemoryTeleologicalStore>,
    Arc<dyn JohariTransitionManager>,
) {
    let store = Arc::new(InMemoryTeleologicalStore::new());
    let johari_manager: Arc<dyn JohariTransitionManager> =
        Arc::new(DefaultJohariManager::new(store.clone()));
    // ... setup handlers with shared johari_manager
    (handlers, store, johari_manager)
}
```

### Required Tests

1. **`test_full_state_verification_distribution_cycle`**
   - Store memory → Get distribution → Verify all 13 embedders present
   - BEFORE/AFTER state logging
   - Direct `store.retrieve(uuid)` to verify

2. **`test_full_state_verification_transition_persistence`**
   - Store memory with Unknown quadrant
   - Execute transition Unknown→Open via `DreamConsolidation`
   - Verify via `store.retrieve(uuid)` that quadrant changed
   - Print BEFORE/AFTER johari state

3. **`test_full_state_verification_batch_all_or_nothing`**
   - Store memory with multiple Unknown embedders
   - Submit batch with one INVALID transition
   - Verify ALL embedders unchanged (all-or-nothing)
   - Direct store inspection

4. **Edge Case Tests** (3 minimum):
   - `edge_case_1_embedder_index_13`: Invalid index must return `-32030`
   - `edge_case_2_invalid_quadrant_string`: Must return `-32031`
   - `edge_case_3_boundary_classification_0_5_0_5`: At threshold (Blind per spec)

---

## Full State Verification Requirements

After completing the logic, you MUST perform Full State Verification:

### 1. Define the Source of Truth

For Johari handlers, the Source of Truth is:
- **`InMemoryTeleologicalStore`**: Contains `TeleologicalFingerprint` with `johari: JohariFingerprint`
- **`JohariFingerprint.quadrants[embedder_idx]`**: The actual soft weights
- **`JohariFingerprint.confidence[embedder_idx]`**: Confidence per embedder

### 2. Execute & Inspect

For EVERY handler test:
```rust
// Execute handler
let response = handlers.dispatch(request).await;

// MUST verify in Source of Truth (not just return value)
let stored = store.retrieve(uuid).await.unwrap().unwrap();
assert_eq!(
    stored.johari.dominant_quadrant(embedder_idx),
    JohariQuadrant::Open,
    "[VERIFICATION FAILED] Transition not persisted to store"
);

println!("[VERIFIED] Transition persisted: {:?} → {:?}",
    before_quadrant, after_quadrant);
```

### 3. Boundary & Edge Case Audit

Simulate and verify these 3 edge cases:

**Edge Case 1: Empty transitions array**
```rust
// BEFORE: Print current johari state
let before = store.retrieve(id).await.unwrap().unwrap();
println!("[STATE BEFORE] quadrants: {:?}", before.johari.quadrants);

// ACTION: Submit empty transitions array
let response = handlers.dispatch(batch_with_empty).await;

// AFTER: Verify state unchanged
let after = store.retrieve(id).await.unwrap().unwrap();
assert_eq!(before.johari.quadrants, after.johari.quadrants);
println!("[EDGE CASE 1 PASSED] Empty transitions handled correctly");
```

**Edge Case 2: Maximum embedder index (12 = valid, 13 = invalid)**
```rust
// Valid max index
let valid = manager.transition(id, 12, Open, DreamConsolidation).await;
assert!(valid.is_ok());

// Invalid index
let invalid = manager.transition(id, 13, Open, DreamConsolidation).await;
assert!(matches!(invalid, Err(JohariError::InvalidEmbedderIndex(13))));
println!("[EDGE CASE 2 PASSED] Boundary embedder index validated");
```

**Edge Case 3: Soft classification sum check (must equal 1.0)**
```rust
// Verify all stored soft classifications sum to 1.0
let stored = store.retrieve(id).await.unwrap().unwrap();
for (i, weights) in stored.johari.quadrants.iter().enumerate() {
    let sum: f32 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 0.001,
        "E{} weights sum to {} (expected 1.0)", i+1, sum);
}
println!("[EDGE CASE 3 PASSED] All soft classifications sum to 1.0");
```

### 4. Evidence of Success

Every test MUST print a summary:
```rust
println!("======================================================================");
println!("EVIDENCE OF SUCCESS - Johari Handler Verification");
println!("======================================================================");
println!("Source of Truth: InMemoryTeleologicalStore");
println!("  - Fingerprint ID: {}", uuid);
println!("  - E1 quadrant: {:?}", stored.johari.dominant_quadrant(0));
println!("  - E5 quadrant: {:?}", stored.johari.dominant_quadrant(4));
println!("Physical Evidence:");
println!("  - Transition executed: {:?} → {:?}", from, to);
println!("  - Persisted to store: YES (verified via retrieve)");
println!("======================================================================");
```

---

## Verification Commands

```bash
# Run all Johari handler tests
cargo test -p context-graph-mcp johari --nocapture

# Run full state verification tests only
cargo test -p context-graph-mcp full_state_verification_johari --nocapture

# Run edge case tests
cargo test -p context-graph-mcp edge_case --nocapture

# Verify transition persistence
cargo test -p context-graph-mcp test_full_state_verification_transition_persistence --nocapture
```

---

## Definition of Done

### Implementation Checklist

- [ ] `johari.rs` handler file created with all 6 handlers
- [ ] `mod.rs` updated with `mod johari;`
- [ ] `core.rs` updated with `johari_manager` field and dispatch cases
- [ ] `protocol.rs` updated with error codes and method constants
- [ ] All handlers use `tracing::instrument` for logging
- [ ] All handlers return proper error codes (no generic errors)

### Testing Checklist (Full State Verification)

- [ ] `test_full_state_verification_distribution_cycle` - Store→Get→Verify
- [ ] `test_full_state_verification_transition_persistence` - Transition persists to store
- [ ] `test_full_state_verification_batch_all_or_nothing` - Failed batch preserves state
- [ ] `edge_case_1_embedder_index_13` - Returns `-32030`
- [ ] `edge_case_2_invalid_quadrant_string` - Returns `-32031`
- [ ] `edge_case_3_boundary_classification` - Correct behavior at threshold
- [ ] All tests print BEFORE/AFTER state
- [ ] All tests verify via direct store access (not just handler response)
- [ ] Evidence of success printed for each test

### NO Backwards Compatibility

- NO fallback to default values on error
- NO mock data in tests (use real `InMemoryTeleologicalStore`)
- NO workarounds for invalid inputs (fail fast with error code)
- NO tests that pass when the system is broken

---

## Embedder Name Reference

| Index | Name | Domain |
|-------|------|--------|
| 0 | E1_semantic | General semantic |
| 1 | E2_episodic | Episode/temporal |
| 2 | E3_procedural | Procedures/how-to |
| 3 | E4_conceptual | Concepts/taxonomy |
| 4 | E5_causal | Cause-effect |
| 5 | E6_emotional | Emotional context |
| 6 | E7_metacognitive | Meta-awareness |
| 7 | E8_spatial | Spatial/structural |
| 8 | E9_linguistic | Language patterns |
| 9 | E10_social | Social relationships |
| 10 | E11_contextual | Context awareness |
| 11 | E12_predictive | Predictions/forecasts |
| 12 | E13_sparse | SPLADE sparse vectors |

---

## Traceability

| Requirement | Source | Coverage |
|-------------|--------|----------|
| FR-203 | FUNC-SPEC-001 | Johari per embedder |
| AC-203.1 | FUNC-SPEC-001 | All 13 spaces classified |
| AC-203.2 | FUNC-SPEC-001 | Confidence scores |
| AC-203.3 | FUNC-SPEC-001 | Transition probabilities |
| AC-203.4 | FUNC-SPEC-001 | Pattern-based queries |

---

*Task updated: 2026-01-05*
*Layer: Surface*
*Priority: P1 - Awareness features*
