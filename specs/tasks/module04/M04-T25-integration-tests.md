---
id: "M04-T25"
title: "Complete Module Integration Tests"
description: |
  VERIFY and COMPLETE comprehensive integration tests for Module 4 Knowledge Graph.
  Test files ALREADY EXIST at crates/context-graph-graph/tests/ - you must audit, fix, and complete them.
  Tests exercise complete workflows: storage lifecycle, hyperbolic geometry, entailment cones,
  graph traversal with Marblestone modulation, domain-aware search, and contradiction detection.
  ALL tests use REAL implementations - NO MOCKS.
layer: "surface"
status: "in_progress"
priority: "critical"
estimated_hours: 8
sequence: 35
depends_on:
  - "M04-T18"  # Semantic search (COMPLETE - f044c84)
  - "M04-T19"  # Domain-aware search (COMPLETE - f891496)
  - "M04-T20"  # Entailment query (COMPLETE - 4fd5052)
  - "M04-T21"  # Contradiction detection (COMPLETE - 11a4bb8)
  - "M04-T22"  # get_modulated_weight (COMPLETE - 4536e42)
  - "M04-T23"  # Poincare CUDA kernel (COMPLETE - f303e17)
  - "M04-T24"  # Cone CUDA kernel (COMPLETE - 6274f5e)
  - "M04-T26"  # EdgeType::Contradicts (COMPLETE - 11a4bb8)
spec_refs:
  - "TECH-GRAPH-004 Section 11"
  - "constitution.yaml testing"
  - "All NFR-KG requirements"
files_already_created:
  - path: "crates/context-graph-graph/tests/integration_tests.rs"
    description: "ALREADY EXISTS - Audit and complete"
  - path: "crates/context-graph-graph/tests/common/mod.rs"
    description: "ALREADY EXISTS - Shared test utilities module"
  - path: "crates/context-graph-graph/tests/common/fixtures.rs"
    description: "ALREADY EXISTS - Test data fixtures"
  - path: "crates/context-graph-graph/tests/common/helpers.rs"
    description: "ALREADY EXISTS - Helper functions for test setup"
files_to_modify:
  - path: "crates/context-graph-graph/tests/integration_tests.rs"
    description: "Complete all test categories, ensure compilation, verify all tests pass"
test_file: "crates/context-graph-graph/tests/integration_tests.rs"
---

# M04-T25: Complete Module Integration Tests

## CRITICAL: READ THIS FIRST

**YOU ARE AN AI AGENT WITH A FRESH CONTEXT WINDOW.**

### What Already Exists (DO NOT RECREATE)

The test infrastructure ALREADY EXISTS:
```
crates/context-graph-graph/tests/
├── integration_tests.rs     # ALREADY EXISTS - ~800 lines, PARTIAL
├── common/
│   ├── mod.rs               # ALREADY EXISTS
│   ├── fixtures.rs          # ALREADY EXISTS
│   └── helpers.rs           # ALREADY EXISTS
├── nt_integration_tests.rs  # ALREADY EXISTS
├── nt_validation_tests.rs   # ALREADY EXISTS
└── storage_tests.rs         # ALREADY EXISTS
```

### Your Task

1. **AUDIT** existing `integration_tests.rs` for compilation errors
2. **FIX** any imports that don't match actual module exports
3. **COMPLETE** any missing test categories
4. **VERIFY** all tests pass with `cargo test -p context-graph-graph --test integration_tests`

### Mandatory Rules

1. **NO BACKWARDS COMPATIBILITY** - System must work or fail fast
2. **NO WORKAROUNDS/FALLBACKS** - If something doesn't work, error out with detailed diagnostics
3. **NO MOCK DATA** - Use REAL storage, REAL data operations
4. **VERIFY OUTPUTS PHYSICALLY** - After each operation, query the actual data source to prove it worked
5. **USE sherlock-holmes AGENT** at the end for forensic verification

---

## Current Codebase State (Audited 2026-01-04)

### Verified Crate Structure

```
crates/context-graph-graph/
├── Cargo.toml
├── src/
│   ├── lib.rs                     # Main exports - READ THIS FOR IMPORTS
│   ├── config.rs                  # IndexConfig, HyperbolicConfig, ConeConfig
│   ├── error.rs                   # GraphError enum (32 variants)
│   ├── hyperbolic/
│   │   ├── mod.rs
│   │   ├── poincare.rs            # PoincarePoint (64D)
│   │   └── mobius.rs              # PoincareBall operations
│   ├── entailment/
│   │   ├── mod.rs                 # Re-exports
│   │   ├── cones.rs               # EntailmentCone struct
│   │   └── query.rs               # entailment_query(), EntailmentResult
│   ├── index/
│   │   ├── mod.rs
│   │   ├── faiss_ffi.rs           # FAISS FFI bindings
│   │   ├── gpu_index.rs           # FaissGpuIndex wrapper
│   │   └── search_result.rs       # SearchResult, SearchResultItem
│   ├── storage/
│   │   ├── mod.rs                 # Column family constants, re-exports
│   │   ├── storage_impl.rs        # GraphStorage backend
│   │   ├── edges.rs               # GraphEdge, EdgeId
│   │   └── migrations.rs          # Schema migrations
│   ├── search/
│   │   ├── mod.rs                 # semantic_search(), semantic_search_simple()
│   │   ├── domain_search.rs       # domain_aware_search(), DomainSearchResult
│   │   ├── filters.rs             # SearchFilters
│   │   └── result.rs              # SemanticSearchResult, SemanticSearchResultItem
│   ├── traversal/
│   │   ├── mod.rs                 # Re-exports BFS, DFS, A*
│   │   ├── bfs.rs                 # bfs_traverse(), BfsParams, BfsResult
│   │   ├── dfs.rs                 # dfs_traverse(), DfsParams, DfsResult
│   │   └── astar.rs               # astar_search(), AstarParams, AstarResult
│   ├── contradiction/
│   │   ├── mod.rs                 # Re-exports
│   │   └── detector.rs            # contradiction_detect(), ContradictionResult
│   ├── marblestone/
│   │   ├── mod.rs                 # get_modulated_weight(), DOMAIN_MATCH_BONUS
│   │   └── validation.rs          # NT weight validation
│   └── query/                     # High-level query operations
└── tests/
    ├── integration_tests.rs       # THIS TASK
    ├── common/                    # Test utilities
    ├── nt_integration_tests.rs
    ├── nt_validation_tests.rs
    └── storage_tests.rs
```

### CUDA Crate Structure

```
crates/context-graph-cuda/
├── src/
│   ├── lib.rs                     # Exports: poincare, cone modules
│   ├── poincare.rs                # poincare_distance_cpu(), poincare_distance_batch_cpu()
│   │                              # (GPU: poincare_distance_batch_gpu with cuda feature)
│   ├── cone.rs                    # cone_check_batch_cpu(), cone_membership_score_cpu()
│   │                              # (GPU: cone_check_batch_gpu with cuda feature)
│   ├── error.rs                   # CudaError enum
│   ├── ops.rs                     # VectorOps trait
│   └── stub.rs                    # StubVectorOps for CPU fallback
└── tests/
    └── (test files)
```

---

## Correct Import Patterns

### From `context_graph_graph` Crate Root

```rust
// Config types
use context_graph_graph::{IndexConfig, HyperbolicConfig, ConeConfig};

// Error types
use context_graph_graph::{GraphError, GraphResult};

// Hyperbolic
use context_graph_graph::{PoincareBall, PoincarePoint};

// Entailment - CORRECT exports from lib.rs
use context_graph_graph::{
    EntailmentCone, entailment_query, EntailmentDirection,
    EntailmentQueryParams, EntailmentResult, BatchEntailmentResult,
    entailment_check_batch, entailment_score, is_entailed_by,
    lowest_common_ancestor, LcaResult,
};

// Index
use context_graph_graph::{FaissGpuIndex, GpuResources, MetricType, SearchResult, SearchResultItem};

// Search
use context_graph_graph::{
    semantic_search, semantic_search_simple, semantic_search_batch, semantic_search_batch_simple,
    SearchFilters, SemanticSearchResult, SemanticSearchResultItem,
    BatchSemanticSearchResult, SearchStats,
};

// Domain search (from search module)
use context_graph_graph::search::{domain_aware_search, DomainSearchResult, DomainSearchResults};

// Contradiction
use context_graph_graph::{
    contradiction_detect, check_contradiction, get_contradictions, mark_contradiction,
    ContradictionParams, ContradictionResult, ContradictionType,
};

// Core types (re-exported from context_graph_core)
use context_graph_graph::{Domain, EdgeType, NeurotransmitterWeights};
use context_graph_graph::{EmbeddingVector, NodeId, DEFAULT_EMBEDDING_DIM};

// Storage - IMPORTANT: These come from storage module
use context_graph_graph::storage::{
    GraphStorage, StorageConfig, PoincarePoint as StoragePoincarePoint,
    EntailmentCone as StorageCone, LegacyGraphEdge, NodeId as StorageNodeId,
    GraphEdge, EdgeId, Domain as StorageDomain, EdgeType as StorageEdgeType,
    NeurotransmitterWeights as StorageNT,
    SCHEMA_VERSION, MigrationInfo, Migrations,
    CF_ADJACENCY, CF_HYPERBOLIC, CF_CONES, CF_FAISS_IDS, CF_NODES, CF_METADATA,
    ALL_COLUMN_FAMILIES, get_column_family_descriptors, get_db_options,
};

// Traversal
use context_graph_graph::traversal::{
    bfs_traverse, bfs_shortest_path, bfs_neighborhood, bfs_domain_neighborhood,
    BfsParams, BfsResult,
    dfs_traverse, dfs_neighborhood, dfs_domain_neighborhood,
    DfsParams, DfsResult, DfsIterator,
    astar_search, astar_bidirectional, astar_path, astar_domain_path,
    AstarParams, AstarResult,
};

// Marblestone
use context_graph_graph::marblestone::DOMAIN_MATCH_BONUS;
// Note: get_modulated_weight is in marblestone::mod.rs
```

### From `context_graph_cuda` Crate

```rust
use context_graph_cuda::{
    // Poincare CPU operations
    PoincareCudaConfig, poincare_distance_cpu, poincare_distance_batch_cpu,

    // Cone CPU operations
    ConeCudaConfig, ConeData, ConeKernelInfo,
    cone_check_batch_cpu, cone_membership_score_cpu,
    is_cone_gpu_available, get_cone_kernel_info,
    CONE_DATA_DIM, POINT_DIM,

    // Error types
    CudaError, CudaResult,

    // Stub for testing
    StubVectorOps, VectorOps,
};

// GPU operations (only with cuda feature)
#[cfg(feature = "cuda")]
use context_graph_cuda::{
    poincare_distance_batch_gpu, poincare_distance_single_gpu,
    cone_check_batch_gpu, cone_check_single_gpu,
};
```

---

## Constitution Performance Targets (NFR)

| Operation | Target | Test Method |
|-----------|--------|-------------|
| FAISS k=100 search (1M vectors) | <2ms | Time 100 iterations, verify avg <2ms |
| Poincare distance (GPU, 1K×1K) | <1ms | Use context-graph-cuda, verify <1ms |
| Cone containment (GPU, 1K×1K) | <2ms | Use context-graph-cuda, verify <2ms |
| BFS depth=6 (10M nodes) | <100ms | Create graph, measure traversal |
| Domain-aware search | <10ms | Include NT modulation in timing |
| Entailment query | <1ms/cone | Measure per-check latency |

---

## Test Categories Required

### 1. Storage Lifecycle Tests ✓ (EXIST)
- `test_storage_lifecycle_complete` - Create, migrate, CRUD
- `test_storage_batch_operations` - Batch writes with timing

### 2. Hyperbolic Geometry Tests ✓ (EXIST)
- `test_poincare_point_invariants` - Origin, norm, boundary
- `test_poincare_distance_properties` - Symmetry, triangle inequality

### 3. Entailment Cone Tests (AUDIT NEEDED)
- `test_entailment_cone_creation` - Apex, aperture validation
- `test_cone_containment_logic` - Membership scoring
- `test_entailment_hierarchy_query` - Ancestor/descendant queries

### 4. Graph Traversal Tests (AUDIT NEEDED)
- `test_bfs_depth_limits` - max_depth enforcement
- `test_bfs_with_nt_modulation` - Domain preference affects order
- `test_dfs_vs_bfs_coverage` - Both reach same nodes
- `test_astar_optimal_path` - Finds shortest path

### 5. Search Operation Tests (AUDIT NEEDED)
- `test_semantic_search_basic` - k-NN returns k results
- `test_domain_aware_search` - NT modulation affects ranking

### 6. Contradiction Detection Tests (AUDIT NEEDED)
- `test_explicit_contradiction` - EdgeType::Contradicts edges
- `test_semantic_contradiction` - High similarity + opposite meaning

### 7. End-to-End Workflow Tests (AUDIT NEEDED)
- `test_complete_knowledge_graph_workflow` - All operations integrated

### 8. Edge Case Tests (AUDIT NEEDED)
- `test_empty_inputs` - Empty queries, missing nodes
- `test_boundary_values` - Max norm, zero aperture
- `test_nan_infinity_handling` - Invalid float handling

---

## Full State Verification Protocol

### MANDATORY: After completing any implementation, you MUST perform:

#### 1. Define the Source of Truth

| Operation | Source of Truth | How to Verify |
|-----------|-----------------|---------------|
| Storage put | `storage.get_*()` | Retrieve what was stored |
| Storage count | `storage.*_count()` | Compare counts before/after |
| FAISS add | `index.ntotal()` | Assert equals vectors added |
| Traversal | `BfsResult.visited_nodes` | Verify all nodes exist in storage |
| Entailment query | Query results | Cross-check with storage cones |
| Contradiction | `ContradictionResult` | Verify EdgeType::Contradicts in storage |

#### 2. Execute & Inspect Pattern

For EVERY test operation:
```rust
// 1. Capture state BEFORE
let state_before = storage.hyperbolic_count().expect("Count failed");
println!("BEFORE: count={}", state_before);

// 2. Execute operation
storage.put_hyperbolic(id, &point).expect("Put failed");

// 3. Read Source of Truth IMMEDIATELY
let state_after = storage.hyperbolic_count().expect("Count failed");
println!("AFTER: count={}", state_after);

// 4. Assert expected change
assert_eq!(state_after, state_before + 1, "Count should increase by 1");

// 5. Physical verification - read back the actual data
let retrieved = storage.get_hyperbolic(id).expect("Get failed").expect("Should exist");
assert_eq!(point.coords, retrieved.coords, "Data should match");
println!("VERIFIED: Data stored and retrieved correctly");
```

#### 3. Boundary & Edge Case Audit (3 Required)

You MUST test these edge cases with state logging:

**Edge Case 1: Empty Query**
```rust
println!("EDGE CASE 1: Empty query");
println!("  BEFORE: storage has {} nodes", storage.hyperbolic_count()?);
let result = bfs_traverse(&storage, nonexistent_node, &params);
println!("  AFTER: result = {:?}", result);
// Expected: Error or empty result, NOT panic
```

**Edge Case 2: Boundary Values**
```rust
println!("EDGE CASE 2: Point at max_norm boundary");
let boundary_point = generate_poincare_point(seed, 0.99999);
println!("  BEFORE: norm = {}", boundary_point.norm());
storage.put_hyperbolic(id, &boundary_point)?;
let retrieved = storage.get_hyperbolic(id)?.expect("Should exist");
println!("  AFTER: retrieved norm = {}", retrieved.norm());
assert!(retrieved.norm() < 1.0, "Point must be inside unit ball");
```

**Edge Case 3: Invalid Input Rejection**
```rust
println!("EDGE CASE 3: Invalid NT weights");
let invalid_nt = NeurotransmitterWeights::new(f32::NAN, 0.5, 0.5);
println!("  BEFORE: weights = {:?}", invalid_nt);
let result = invalid_nt.validate();
println!("  AFTER: validation = {:?}", result);
assert!(result.is_err(), "NaN must be rejected");
```

#### 4. Evidence of Success Log

Every test MUST print:
```
=== TEST: <test_name> ===
SOURCE OF TRUTH: <what is being verified>
BEFORE: <initial state>
OPERATION: <what was executed>
AFTER: <final state>
PHYSICAL VERIFICATION: <proof data exists>
RESULT: PASS
=== END TEST ===
```

---

## Test Commands

```bash
# Compile check (DO THIS FIRST)
cargo build -p context-graph-graph --test integration_tests

# Run all integration tests with output
cargo test -p context-graph-graph --test integration_tests -- --nocapture

# Run specific test
cargo test -p context-graph-graph --test integration_tests test_storage_lifecycle_complete -- --nocapture

# Run with verbose logging
RUST_LOG=debug cargo test -p context-graph-graph --test integration_tests -- --nocapture

# Run NFR performance tests (ignored by default)
cargo test -p context-graph-graph --test integration_tests -- --ignored --nocapture

# Check for compilation errors in all tests
cargo test -p context-graph-graph --no-run
```

---

## Acceptance Criteria

- [ ] `cargo build -p context-graph-graph --test integration_tests` compiles without errors
- [ ] All CPU tests pass: `cargo test -p context-graph-graph --test integration_tests`
- [ ] Every test prints state verification evidence (BEFORE/AFTER)
- [ ] No mock data used (per REQ-KG-TEST)
- [ ] No clippy warnings: `cargo clippy -p context-graph-graph --test integration_tests`
- [ ] Edge cases handled with proper errors (no panics on invalid input)
- [ ] 3 boundary edge cases tested and logged

---

## Sherlock-Holmes Forensic Verification

**MANDATORY: After completing all fixes, spawn sherlock-holmes with this prompt:**

```
FORENSIC VERIFICATION TASK: M04-T25 Integration Tests

EVIDENCE TO EXAMINE:
1. Files exist at correct paths:
   - crates/context-graph-graph/tests/integration_tests.rs
   - crates/context-graph-graph/tests/common/mod.rs
   - crates/context-graph-graph/tests/common/fixtures.rs
   - crates/context-graph-graph/tests/common/helpers.rs

2. Compilation succeeds:
   cargo build -p context-graph-graph --test integration_tests 2>&1

3. Tests pass:
   cargo test -p context-graph-graph --test integration_tests -- --nocapture 2>&1

VERIFICATION CHECKLIST:
- [ ] All imports resolve correctly (no unresolved imports)
- [ ] Tests use real storage (GraphStorage::new or similar)
- [ ] Tests verify state BEFORE and AFTER operations
- [ ] Tests print evidence of physical verification
- [ ] No unwrap() without expect() context
- [ ] No mock data generators (check fixtures.rs)
- [ ] Edge cases covered (empty, boundary, invalid)

CRITICAL ISSUES TO FLAG:
- Compilation errors
- Import resolution failures
- Tests that always pass (no assertions)
- Missing state verification
- Panics instead of proper error handling

Report ALL issues found with file:line references.
```

---

## Common Errors and Fixes

### Error: `unresolved import`
**Cause**: Import path doesn't match actual module exports
**Fix**: Check `lib.rs` re-exports, use correct path

### Error: `type mismatch`
**Cause**: Storage types vs hyperbolic types are different structs
**Fix**: Use `storage::PoincarePoint` for storage ops, `hyperbolic::PoincarePoint` for math

### Error: `no method named X`
**Cause**: Method exists on different struct or in different module
**Fix**: Check actual struct definition in source file

### Error: Tests compile but always pass
**Cause**: No assertions or assertions that can't fail
**Fix**: Add `assert!()` with meaningful conditions, verify state changed

---

## Git History Context (Recent Commits)

```
6274f5e feat(cuda): complete M04-T24 cone membership CUDA kernel
f303e17 feat(cuda): complete M04-T23 Poincare distance CUDA kernel
4536e42 feat(graph): complete M04-T22 standalone modulation utilities
11a4bb8 feat(graph): complete M04-T21 contradiction detection and M04-T26 EdgeType::Contradicts
4fd5052 feat(graph): complete M04-T20 entailment query with full state verification
f891496 feat(graph): complete M04-T19 domain-aware search with NT modulation
f044c84 feat(graph): complete M04-T18 semantic search with FAISS GPU integration
```

All dependencies for M04-T25 are complete. The integration tests need to verify these implementations work together.

---

## Task NOT Complete Until

1. `cargo test -p context-graph-graph --test integration_tests` passes
2. sherlock-holmes confirms all evidence checks pass
3. Every test prints BEFORE/AFTER state verification
4. 3 edge cases are tested and logged
