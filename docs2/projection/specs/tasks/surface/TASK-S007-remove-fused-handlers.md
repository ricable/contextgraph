# TASK-S007: Remove All Fusion-Related MCP Handlers

```yaml
metadata:
  id: "TASK-S007"
  title: "Remove All Fusion-Related MCP Handlers"
  layer: "surface"
  priority: "P0"
  estimated_hours: 4
  created: "2026-01-04"
  status: "pending"
  dependencies:
    - "TASK-F006"  # Remove fusion files (core)
  traces_to:
    - "FR-601"  # Complete Removal of 36 Fusion Files
    - "FR-602"  # No Backwards Compatibility
```

## Problem Statement

Remove all fusion-related MCP handlers from the codebase, clean up handler registry, and update all handler exports. No backwards compatibility.

## Context

The MCP handler layer contains legacy fusion-related handlers that must be completely removed:
- Fused search handlers
- Fused memory handlers
- Single-vector similarity handlers
- Gating-related handlers
- Legacy Vector1536 handlers

**NO backwards compatibility layers. Clients must update to new API immediately.**

## Technical Specification

### Handlers to Remove

```rust
// ALL OF THESE MUST BE DELETED - NOT DEPRECATED, DELETED

/// REMOVE: Legacy fused search handler
pub async fn handle_fused_search(...) // DELETE

/// REMOVE: Legacy single-vector store
pub async fn handle_vector_store(...) // DELETE

/// REMOVE: Legacy Vector1536 similarity
pub async fn handle_vector_similarity(...) // DELETE

/// REMOVE: Legacy gating query
pub async fn handle_gating_query(...) // DELETE

/// REMOVE: Legacy fusion config
pub async fn handle_fusion_config_get(...) // DELETE
pub async fn handle_fusion_config_set(...) // DELETE

/// REMOVE: Legacy expert selection
pub async fn handle_expert_selection(...) // DELETE
```

### Files to Delete

```
crates/context-graph-mcp/src/handlers/
├── fused_search.rs           # DELETE
├── fused_memory.rs           # DELETE
├── vector_store.rs           # DELETE
├── vector_similarity.rs      # DELETE
├── gating.rs                 # DELETE
├── fusion_config.rs          # DELETE
├── expert_selection.rs       # DELETE
└── legacy_compat.rs          # DELETE

crates/context-graph-mcp/src/schemas/
├── fused_search_request.json     # DELETE
├── fused_search_response.json    # DELETE
├── vector_store_request.json     # DELETE
├── fusion_config.json            # DELETE
└── gating_config.json            # DELETE

crates/context-graph-mcp/tests/
├── fused_search_tests.rs     # DELETE
├── vector_store_tests.rs     # DELETE
└── fusion_config_tests.rs    # DELETE
```

### Handler Registry Updates

```rust
// BEFORE (to be removed)
pub fn register_handlers(router: &mut Router) {
    // Legacy fusion handlers - REMOVE ALL OF THESE
    router.register("fused_search", handle_fused_search);
    router.register("vector_store", handle_vector_store);
    router.register("vector_similarity", handle_vector_similarity);
    router.register("fusion_config_get", handle_fusion_config_get);
    router.register("fusion_config_set", handle_fusion_config_set);
    router.register("gating_query", handle_gating_query);
    router.register("expert_selection", handle_expert_selection);

    // New handlers - KEEP
    router.register("memory_store", handle_memory_store);
    router.register("memory_retrieve", handle_memory_retrieve);
    // ...
}

// AFTER (clean registry)
pub fn register_handlers(router: &mut Router) {
    // Memory handlers (TASK-S001)
    router.register("memory_store", handle_memory_store);
    router.register("memory_retrieve", handle_memory_retrieve);
    router.register("memory_delete", handle_memory_delete);
    router.register("memory_batch_store", handle_memory_batch_store);

    // Search handlers (TASK-S002)
    router.register("search_multi", handle_search_multi);
    router.register("search_single_space", handle_search_single_space);
    router.register("search_by_purpose", handle_search_by_purpose);
    router.register("get_weight_profiles", handle_get_weight_profiles);

    // Purpose handlers (TASK-S003)
    router.register("purpose_query", handle_purpose_query);
    router.register("north_star_alignment", handle_north_star_alignment);
    router.register("goal_hierarchy_query", handle_goal_hierarchy_query);
    router.register("find_aligned_to_goal", handle_find_aligned_to_goal);
    router.register("alignment_drift_check", handle_alignment_drift_check);
    router.register("north_star_update", handle_north_star_update);

    // Johari handlers (TASK-S004)
    router.register("johari_get_distribution", handle_johari_get_distribution);
    router.register("johari_find_by_quadrant", handle_johari_find_by_quadrant);
    router.register("johari_update", handle_johari_update);
    router.register("johari_cross_space_analysis", handle_johari_cross_space_analysis);
    router.register("johari_transition_probabilities", handle_johari_transition_probabilities);

    // Meta-UTL handlers (TASK-S005)
    router.register("meta_utl_learning_trajectory", handle_meta_utl_learning_trajectory);
    router.register("meta_utl_health_metrics", handle_meta_utl_health_metrics);
    router.register("meta_utl_predict_storage", handle_meta_utl_predict_storage);
    router.register("meta_utl_predict_retrieval", handle_meta_utl_predict_retrieval);
    router.register("meta_utl_validate_prediction", handle_meta_utl_validate_prediction);
    router.register("meta_utl_get_optimized_weights", handle_meta_utl_get_optimized_weights);
}
```

### Module Exports Update

```rust
// crates/context-graph-mcp/src/handlers/mod.rs

// REMOVE these exports
// pub mod fused_search;        // DELETE
// pub mod fused_memory;        // DELETE
// pub mod vector_store;        // DELETE
// pub mod vector_similarity;   // DELETE
// pub mod gating;              // DELETE
// pub mod fusion_config;       // DELETE
// pub mod expert_selection;    // DELETE
// pub mod legacy_compat;       // DELETE

// KEEP/ADD these exports
pub mod memory;
pub mod search;
pub mod purpose;
pub mod goals;
pub mod johari;
pub mod meta_utl;

// Re-exports
pub use memory::*;
pub use search::*;
pub use purpose::*;
pub use goals::*;
pub use johari::*;
pub use meta_utl::*;
```

### Error Response for Legacy Calls

```rust
/// Error for legacy handler calls (should never be reached if properly removed)
/// This is a COMPILE-TIME safety net, not runtime compatibility
#[derive(Debug)]
pub struct LegacyHandlerError;

impl std::fmt::Display for LegacyHandlerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CRITICAL: Legacy fusion handler invoked. This should be impossible if removal was complete. Check handler registry.")
    }
}

// If any code tries to call legacy handlers at runtime, fail immediately
macro_rules! fail_on_legacy_call {
    ($name:expr) => {
        panic!(
            "CRITICAL ERROR: Legacy handler '{}' was called. \
            All fusion handlers should have been removed. \
            This indicates incomplete removal in TASK-S007. \
            Stack trace will help identify the caller.",
            $name
        );
    };
}
```

### Verification Script

```bash
#!/bin/bash
# verify_no_fusion_handlers.sh

echo "Verifying no fusion handlers remain..."

# Check for fusion-related files
FUSION_FILES=$(find crates/context-graph-mcp -name "*fuse*" -o -name "*fusion*" -o -name "*gating*" -o -name "*vector_store*" -o -name "*expert*" 2>/dev/null)

if [ -n "$FUSION_FILES" ]; then
    echo "ERROR: Found fusion-related files:"
    echo "$FUSION_FILES"
    exit 1
fi

# Check for fusion-related code patterns in handlers
FUSION_PATTERNS="fuse_moe|FuseMoE|fusion|gating|expert_select|Vector1536|handle_fused|handle_vector_store"
FOUND=$(rg -l "$FUSION_PATTERNS" crates/context-graph-mcp/src/handlers/ 2>/dev/null)

if [ -n "$FOUND" ]; then
    echo "ERROR: Found fusion-related code in:"
    echo "$FOUND"
    rg "$FUSION_PATTERNS" crates/context-graph-mcp/src/handlers/
    exit 1
fi

# Check handler registry for legacy registrations
REGISTRY_CHECK=$(rg "register.*fused|register.*gating|register.*expert" crates/context-graph-mcp/src/router.rs 2>/dev/null)

if [ -n "$REGISTRY_CHECK" ]; then
    echo "ERROR: Found legacy handler registrations:"
    echo "$REGISTRY_CHECK"
    exit 1
fi

# Check for legacy imports
LEGACY_IMPORTS=$(rg "use.*fused|use.*fusion|use.*gating" crates/context-graph-mcp/ 2>/dev/null)

if [ -n "$LEGACY_IMPORTS" ]; then
    echo "ERROR: Found legacy imports:"
    echo "$LEGACY_IMPORTS"
    exit 1
fi

echo "SUCCESS: No fusion handlers found"
exit 0
```

## Implementation Requirements

### Prerequisites

- [ ] TASK-F006 complete (Core fusion file removal)

### Scope

#### In Scope

- Delete all fusion-related handler files
- Delete all fusion-related schema files
- Delete all fusion-related test files
- Update handler registry
- Update module exports
- Verify no fusion code remains

#### Out of Scope

- Creating new handlers (TASK-S001 through TASK-S005)
- Core fusion removal (TASK-F006)

### Constraints

- NO deprecation warnings - DELETE immediately
- NO compatibility shims
- NO migration helpers
- Compilation must fail if any fusion code remains

## Definition of Done

### Implementation Checklist

- [ ] All fusion handler files deleted
- [ ] All fusion schema files deleted
- [ ] All fusion test files deleted
- [ ] Handler registry cleaned
- [ ] Module exports updated
- [ ] No fusion patterns in MCP crate
- [ ] Verification script passes
- [ ] Crate compiles without fusion

### Verification Commands

```bash
# Run verification script
./scripts/verify_no_fusion_handlers.sh

# Check for any fusion patterns
rg "fuse|fusion|gating|expert" crates/context-graph-mcp/

# Compile to verify
cargo build -p context-graph-mcp

# Run handler tests (should have no fusion tests)
cargo test -p context-graph-mcp
```

## Files to Delete

| File | Reason |
|------|--------|
| `handlers/fused_search.rs` | Legacy fusion search |
| `handlers/fused_memory.rs` | Legacy fusion memory |
| `handlers/vector_store.rs` | Legacy Vector1536 store |
| `handlers/vector_similarity.rs` | Legacy single-vector sim |
| `handlers/gating.rs` | Legacy gating handlers |
| `handlers/fusion_config.rs` | Legacy fusion config |
| `handlers/expert_selection.rs` | Legacy expert selection |
| `handlers/legacy_compat.rs` | Compatibility shims |
| `schemas/fused_*.json` | Legacy schemas |
| `tests/fused_*.rs` | Legacy tests |
| `tests/vector_*.rs` | Legacy tests |
| `tests/fusion_*.rs` | Legacy tests |

## Files to Modify

| File | Change |
|------|--------|
| `handlers/mod.rs` | Remove fusion exports |
| `router.rs` | Remove legacy registrations |
| `lib.rs` | Update public API |
| `error.rs` | Remove fusion error types |

## Traceability

| Requirement | Source | Coverage |
|-------------|--------|----------|
| FR-601 | FUNC-SPEC-001 | Handler file deletion |
| FR-602 | FUNC-SPEC-001 | No compat layer |
| AC-601.1 | FUNC-SPEC-001 | No fuse/fusion files |
| AC-601.2 | FUNC-SPEC-001 | No imports |
| AC-602.3 | FUNC-SPEC-001 | No compatibility shims |

---

*Task created: 2026-01-04*
*Layer: Surface*
*Priority: P0 - Architectural integrity*
