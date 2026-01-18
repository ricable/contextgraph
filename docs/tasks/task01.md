# Task 01: Fix MCP Test Suite Compilation Errors

## Metadata
- **Task ID**: TASK-GAP-001
- **Phase**: 1 (Foundation)
- **Priority**: Critical (Blocking all test execution)
- **Dependencies**: None
- **Branch**: multistar
- **Status**: ✅ COMPLETED (2026-01-18)

## Completion Summary

### Results
- **Compilation**: ✅ PASSING - `cargo test --no-run -p context-graph-mcp` exits 0
- **Clippy**: ✅ PASSING - `cargo clippy -p context-graph-mcp -- -D warnings` exits 0
- **Tests**: ✅ PASSING - 211 passed, 0 failed, 137 ignored

### Changes Made
1. Updated 7 helper functions in `mod.rs` to use `Handlers::with_all()` instead of removed `Handlers::new()`
2. Removed dead GWT/MetaUtl code sections that referenced deleted modules
3. Updated `task_emb_024_verification.rs`, `manual_fsv_verification.rs`, and `full_state_verification_search.rs`
4. Marked 137 tests as `#[ignore]` that use APIs removed in PRD v6 refactor:
   - Tests using `memory/store`, `memory/retrieve`, `memory/delete` (now `tools/call` with `store_memory`)
   - Tests using `search/multi`, `search/single_space`, `search/by_purpose` (now `tools/call` with `search_graph`)
   - Tests expecting 54-58 tools (PRD v6 has 6 core tools)
   - Tests for tools not registered in PRD v6 (`utl_status`, `get_pruning_candidates`, etc.)

### Future Work (TASK-GAP-002)
Tests marked with `#[ignore = "... - TASK-GAP-002"]` should be reimplemented when:
- CognitivePulse adds `quadrant` and `suggested_action` fields
- Tool count increases as PRD v6 modules are implemented
- Removed tools are re-added to the registry

---

## Problem Statement

The MCP test suite fails to compile due to imports referencing modules that were **deleted in commit `fab0622`** (PRD v6 compliance refactor). This commit removed ~50 handler modules, but the test files still reference deleted types.

**Compilation currently fails with 11 errors preventing `cargo test -p context-graph-mcp` from running.**

---

## Current Codebase State (Verified 2026-01-18)

### Handlers Struct Definition

**File**: `crates/context-graph-mcp/src/handlers/core/handlers.rs`

```rust
pub struct Handlers {
    pub(in crate::handlers) teleological_store: Arc<dyn TeleologicalMemoryStore>,
    pub(in crate::handlers) utl_processor: Arc<dyn UtlProcessor>,
    pub(in crate::handlers) multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
    pub(in crate::handlers) goal_hierarchy: Arc<RwLock<GoalHierarchy>>,
    pub(in crate::handlers) layer_status_provider: Arc<dyn LayerStatusProvider>,
}
```

**Available Constructor** (line 51):
```rust
pub fn with_all(
    teleological_store: Arc<dyn TeleologicalMemoryStore>,
    utl_processor: Arc<dyn UtlProcessor>,
    multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
    goal_hierarchy: Arc<RwLock<GoalHierarchy>>,
    layer_status_provider: Arc<dyn LayerStatusProvider>,
) -> Self
```

**What Exports from `handlers/core/mod.rs`**:
```rust
pub use self::handlers::Handlers;
```

### Deleted Modules (commit fab0622)

These modules NO LONGER EXIST:
- `handlers/gwt_providers.rs` - GWT workspace providers
- `handlers/gwt_traits.rs` - GWT trait definitions
- `handlers/core/meta_utl_tracker.rs` - Meta-UTL tracking
- `handlers/autonomous/*` - Autonomous goal/drift/health handlers
- `handlers/dream/*` - Dream consolidation handlers
- `handlers/purpose/*` - Purpose/alignment handlers
- `handlers/session/*` - Session handlers
- `handlers/atc.rs`, `handlers/causal.rs`, `handlers/neuromod.rs`, etc.

### Deleted Constructor Methods

These methods were removed from `Handlers`:
- `Handlers::new()` - OLD constructor that accepted 5-6 args
- `Handlers::with_gwt()` - Constructor with GWT components (11 args)
- `Handlers::with_gwt_and_subsystems()` - Constructor with all subsystems
- `Handlers::with_meta_utl_tracker()` - Constructor with MetaUtlTracker

**The ONLY valid constructor is `Handlers::with_all()` which requires 5 args.**

---

## Compilation Errors (11 Total)

### Error Group 1: Missing MetaUtlTracker (3 errors)

| File | Line | Error |
|------|------|-------|
| `handlers/tests/mod.rs` | 700 | `use crate::handlers::core::MetaUtlTracker` |
| `handlers/tests/task_emb_024_verification.rs` | 28 | `use crate::handlers::core::MetaUtlTracker` |
| `handlers/tests/manual_fsv_verification.rs` | 21 | `use crate::handlers::core::MetaUtlTracker` |

### Error Group 2: Missing gwt_providers Module (2 errors)

| File | Line | Error |
|------|------|-------|
| `handlers/tests/mod.rs` | 701-703 | `use crate::handlers::gwt_providers::{...}` |
| `handlers/tests/mod.rs` | 827 | `use super::gwt_providers::{...}` |

### Error Group 3: Missing gwt_traits Module (1 error)

| File | Line | Error |
|------|------|-------|
| `handlers/tests/mod.rs` | 704 | `use crate::handlers::gwt_traits::{...}` |

### Error Group 4: Missing Handlers::new() (5 errors)

| File | Line | Function |
|------|------|----------|
| `handlers/tests/mod.rs` | 221 | `create_test_handlers()` |
| `handlers/tests/mod.rs` | 240 | `create_test_handlers_no_goals()` |
| `handlers/tests/mod.rs` | 318 | `create_test_handlers_with_rocksdb()` |
| `handlers/tests/mod.rs` | 370 | `create_test_handlers_with_rocksdb_no_goals()` |
| `handlers/tests/mod.rs` | 502 | `create_test_handlers_with_rocksdb_store_access()` |
| `handlers/tests/mod.rs` | 588 | `create_test_handlers_with_real_embeddings()` |
| `handlers/tests/mod.rs` | 638 | `create_test_handlers_with_real_embeddings_store_access()` |

---

## Implementation Steps

### Step 1: Fix mod.rs Helper Functions (Lines 212-647)

**Change all calls from `Handlers::new(...)` to `Handlers::with_all(...)`**

The signature changes from:
```rust
// OLD (5 args, no RwLock wrapper, no layer_status_provider)
Handlers::new(
    teleological_store,
    utl_processor,
    multi_array_provider,
    alignment_calculator,  // <-- This was removed
    goal_hierarchy,        // <-- Was GoalHierarchy, now Arc<RwLock<GoalHierarchy>>
)
```

To:
```rust
// NEW (5 args, requires RwLock wrapper and layer_status_provider)
Handlers::with_all(
    teleological_store,
    utl_processor,
    multi_array_provider,
    Arc::new(RwLock::new(goal_hierarchy)),
    layer_status_provider,  // NEW required arg
)
```

**Required imports to add**:
```rust
use context_graph_core::monitoring::StubLayerStatusProvider;
use parking_lot::RwLock;
```

### Step 2: Remove Dead GWT/MetaUtl Code (Lines 690-893)

**DELETE these lines entirely** (not needed for PRD v6):

1. Lines 694-698 - Remove imports:
```rust
// DELETE:
use parking_lot::RwLock as ParkingRwLock;
use tokio::sync::RwLock as TokioRwLock;
use context_graph_core::monitoring::{StubLayerStatusProvider, StubSystemMonitor};
use context_graph_core::{LayerStatusProvider, SystemMonitor};
```

2. Lines 700-704 - Remove broken imports:
```rust
// DELETE:
use crate::handlers::core::MetaUtlTracker;
use crate::handlers::gwt_providers::{
    GwtSystemProviderImpl, MetaCognitiveProviderImpl, WorkspaceProviderImpl,
};
use crate::handlers::gwt_traits::{GwtSystemProvider, MetaCognitiveProvider, WorkspaceProvider};
```

3. Lines 729-761 - DELETE `create_test_handlers_with_warm_gwt()` function
4. Lines 771-811 - DELETE `create_test_handlers_with_warm_gwt_rocksdb()` function
5. Lines 826-893 - DELETE `create_test_handlers_with_all_components()` function

### Step 3: Fix task_emb_024_verification.rs

**File**: `crates/context-graph-mcp/src/handlers/tests/task_emb_024_verification.rs`

1. **Line 28** - Remove broken import:
```rust
// DELETE:
use crate::handlers::core::MetaUtlTracker;
```

2. **Lines 35-51** - Fix `create_handlers_with_stub_monitors()`:
```rust
fn create_handlers_with_stub_monitors() -> Handlers {
    use context_graph_core::monitoring::StubLayerStatusProvider;
    use parking_lot::RwLock;

    let store: Arc<dyn TeleologicalMemoryStore> = Arc::new(InMemoryTeleologicalStore::new());
    let utl_processor = Arc::new(StubUtlProcessor::new());
    let multi_array = Arc::new(StubMultiArrayProvider::new());
    let goal_hierarchy = GoalHierarchy::default();
    let layer_status: Arc<dyn LayerStatusProvider> = Arc::new(StubLayerStatusProvider);

    Handlers::with_all(
        store,
        utl_processor,
        multi_array,
        Arc::new(RwLock::new(goal_hierarchy)),
        layer_status,
    )
}
```

3. **Lines 53-75** - DELETE `create_handlers_with_tracker()` function (uses deleted MetaUtlTracker)

4. **Any tests using `create_handlers_with_tracker()`** - Mark with `#[ignore]`:
```rust
#[tokio::test]
#[ignore = "MetaUtlTracker removed in fab0622 - blocked until Meta-UTL reimplemented"]
async fn test_name() {
    // Test body removed - cannot compile
}
```

### Step 4: Fix manual_fsv_verification.rs

**File**: `crates/context-graph-mcp/src/handlers/tests/manual_fsv_verification.rs`

1. **Line 21** - Remove broken import:
```rust
// DELETE:
use crate::handlers::core::MetaUtlTracker;
```

2. **Add required imports after line 19**:
```rust
use context_graph_core::monitoring::{LayerStatusProvider, StubLayerStatusProvider};
use parking_lot::RwLock;
```

3. **Lines 71-93** - Fix `manual_fsv_memory_store_physical_verification()`:

Replace lines 78-93:
```rust
// OLD (broken):
let hierarchy = Arc::new(RwLock::new(create_test_hierarchy()));
let tracker = Arc::new(RwLock::new(MetaUtlTracker::new()));
let handlers = Handlers::with_meta_utl_tracker(
    store.clone(), utl_processor, multi_array, alignment, hierarchy, tracker,
);

// NEW (working):
let hierarchy = Arc::new(RwLock::new(create_test_hierarchy()));
let layer_status: Arc<dyn LayerStatusProvider> = Arc::new(StubLayerStatusProvider);
let handlers = Handlers::with_all(
    store.clone(), utl_processor, multi_array, hierarchy, layer_status,
);
```

4. **All other test functions using MetaUtlTracker or with_meta_utl_tracker** - Mark with `#[ignore]`:
```rust
#[tokio::test]
#[ignore = "MetaUtlTracker removed in fab0622 - blocked until Meta-UTL reimplemented"]
async fn manual_fsv_meta_utl_verification() {
    // Original test cannot compile
}
```

---

## Code Templates

### Working create_test_handlers()

```rust
pub(crate) fn create_test_handlers() -> Handlers {
    use context_graph_core::monitoring::StubLayerStatusProvider;
    use parking_lot::RwLock;

    let teleological_store: Arc<dyn TeleologicalMemoryStore> =
        Arc::new(InMemoryTeleologicalStore::new());
    let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());
    let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> =
        Arc::new(StubMultiArrayProvider::new());
    let goal_hierarchy = create_test_hierarchy();
    let layer_status: Arc<dyn LayerStatusProvider> = Arc::new(StubLayerStatusProvider);

    Handlers::with_all(
        teleological_store,
        utl_processor,
        multi_array_provider,
        Arc::new(RwLock::new(goal_hierarchy)),
        layer_status,
    )
}
```

### Working create_test_handlers_with_rocksdb()

```rust
pub(crate) async fn create_test_handlers_with_rocksdb() -> (Handlers, TempDir) {
    use context_graph_core::monitoring::StubLayerStatusProvider;
    use parking_lot::RwLock;

    let tempdir = TempDir::new().expect("Failed to create temp directory for RocksDB test");
    let db_path = tempdir.path().join("test_rocksdb");

    let rocksdb_store = RocksDbTeleologicalStore::open(&db_path)
        .expect("Failed to open RocksDbTeleologicalStore in test");

    let teleological_store: Arc<dyn TeleologicalMemoryStore> = Arc::new(rocksdb_store);
    let utl_processor: Arc<dyn UtlProcessor> = Arc::new(UtlProcessorAdapter::with_defaults());
    let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> =
        Arc::new(StubMultiArrayProvider::new());
    let goal_hierarchy = create_test_hierarchy();
    let layer_status: Arc<dyn LayerStatusProvider> = Arc::new(StubLayerStatusProvider);

    let handlers = Handlers::with_all(
        teleological_store,
        utl_processor,
        multi_array_provider,
        Arc::new(RwLock::new(goal_hierarchy)),
        layer_status,
    );

    (handlers, tempdir)
}
```

---

## Definition of Done

### Compilation Gates (MUST PASS)
- [ ] `cargo test --no-run -p context-graph-mcp` exits 0 (no compilation errors)
- [ ] `cargo clippy -p context-graph-mcp -- -D warnings` exits 0 (no warnings)

### Reference Cleanup (MUST VERIFY)
- [ ] Zero grep hits for `MetaUtlTracker` in `crates/context-graph-mcp/src/handlers/tests/`
- [ ] Zero grep hits for `gwt_providers` in `crates/context-graph-mcp/src/handlers/tests/`
- [ ] Zero grep hits for `gwt_traits` in `crates/context-graph-mcp/src/handlers/tests/`
- [ ] Zero grep hits for `Handlers::new` in `crates/context-graph-mcp/src/handlers/tests/`
- [ ] Zero grep hits for `with_meta_utl_tracker` in `crates/context-graph-mcp/src/handlers/tests/`
- [ ] Zero grep hits for `with_gwt` in `crates/context-graph-mcp/src/handlers/tests/`

### Test Execution (MUST PASS)
- [ ] `cargo test -p context-graph-mcp --lib` runs without panic
- [ ] Tests using `create_test_handlers()` execute successfully
- [ ] Tests using `create_test_handlers_with_rocksdb()` execute successfully

---

## Full State Verification Requirements

After completing the implementation, you MUST perform Full State Verification:

### 1. Define Source of Truth

| Component | Source of Truth Location |
|-----------|-------------------------|
| Compilation | `cargo test --no-run -p context-graph-mcp` exit code |
| Clippy | `cargo clippy -p context-graph-mcp -- -D warnings` exit code |
| Dead references | `grep -r "pattern" path` output (must be empty) |
| Test execution | `cargo test -p context-graph-mcp --lib` output |

### 2. Execute & Inspect

After each fix, verify:
```bash
# Source of Truth 1: Compilation status
cargo test --no-run -p context-graph-mcp 2>&1 | tail -20
echo "Exit code: $?"

# Source of Truth 2: Clippy warnings
cargo clippy -p context-graph-mcp -- -D warnings 2>&1 | tail -20
echo "Exit code: $?"

# Source of Truth 3: Dead reference check
echo "=== Checking for dead references ==="
grep -r "MetaUtlTracker" crates/context-graph-mcp/src/handlers/tests/ || echo "PASS: No MetaUtlTracker references"
grep -r "gwt_providers" crates/context-graph-mcp/src/handlers/tests/ || echo "PASS: No gwt_providers references"
grep -r "gwt_traits" crates/context-graph-mcp/src/handlers/tests/ || echo "PASS: No gwt_traits references"
grep -r "Handlers::new\(" crates/context-graph-mcp/src/handlers/tests/ || echo "PASS: No Handlers::new references"

# Source of Truth 4: Test execution
cargo test -p context-graph-mcp --lib -- --nocapture 2>&1 | tail -50
echo "Exit code: $?"
```

### 3. Boundary & Edge Case Audit

Test these 3 edge cases after fixes:

**Edge Case 1: Empty goal hierarchy**
```bash
# Run test that uses create_test_handlers_no_goals()
cargo test -p context-graph-mcp test_missing_strategic --nocapture
# Expected: Should compile and run (may fail assertion, but must not panic on creation)
```

**Edge Case 2: RocksDB initialization**
```bash
# Run test that creates RocksDB
cargo test -p context-graph-mcp create_test_handlers_with_rocksdb --nocapture
# Expected: Should create temp directory and open RocksDB without error
```

**Edge Case 3: Multiple handler creations**
```bash
# Run multiple tests in sequence
cargo test -p context-graph-mcp --lib -- test_tools test_initialize --nocapture
# Expected: Each test should get isolated handler instances
```

### 4. Evidence of Success

Provide a log showing:
```
=== FINAL VERIFICATION LOG ===
Timestamp: [date]

1. Compilation:
   $ cargo test --no-run -p context-graph-mcp
   [output]
   Exit code: 0  # MUST be 0

2. Clippy:
   $ cargo clippy -p context-graph-mcp -- -D warnings
   [output]
   Exit code: 0  # MUST be 0

3. Dead References:
   $ grep -r "MetaUtlTracker" crates/context-graph-mcp/src/handlers/tests/
   [empty - no matches]  # MUST be empty

   $ grep -r "gwt_providers" crates/context-graph-mcp/src/handlers/tests/
   [empty - no matches]  # MUST be empty

4. Test Execution:
   $ cargo test -p context-graph-mcp --lib
   running X tests
   test ... ok
   test ... ok
   test result: ok. X passed; 0 failed  # MUST show 0 failed
```

---

## CRITICAL RULES

1. **NO BACKWARDS COMPATIBILITY** - Do not create workarounds or fallbacks
2. **FAIL FAST** - If something doesn't work, it must error with clear message
3. **NO MOCK DATA IN TESTS** - Use real stubs that fail fast on unimplemented paths
4. **NO COVERING UP FAILURES** - If a test cannot be fixed, mark it `#[ignore]` with reason

---

## Verification Commands

```bash
cd /home/cabdru/contextgraph

# Step 1: Verify tests compile (Source of Truth: exit code)
cargo test --no-run -p context-graph-mcp
echo "Compilation exit code: $?"

# Step 2: Run clippy (Source of Truth: exit code)
cargo clippy -p context-graph-mcp -- -D warnings
echo "Clippy exit code: $?"

# Step 3: Verify no references to deleted modules (Source of Truth: grep output)
echo "=== Dead Reference Check ==="
grep -rn "MetaUtlTracker" crates/context-graph-mcp/src/handlers/tests/ && echo "FAIL" || echo "PASS"
grep -rn "gwt_providers" crates/context-graph-mcp/src/handlers/tests/ && echo "FAIL" || echo "PASS"
grep -rn "gwt_traits" crates/context-graph-mcp/src/handlers/tests/ && echo "FAIL" || echo "PASS"
grep -rn "Handlers::new\(" crates/context-graph-mcp/src/handlers/tests/ && echo "FAIL" || echo "PASS"
grep -rn "with_meta_utl_tracker" crates/context-graph-mcp/src/handlers/tests/ && echo "FAIL" || echo "PASS"
grep -rn "with_gwt" crates/context-graph-mcp/src/handlers/tests/ && echo "FAIL" || echo "PASS"

# Step 4: Run tests to ensure nothing regressed (Source of Truth: test output)
cargo test -p context-graph-mcp --lib -- --nocapture 2>&1 | tee /tmp/test_output.log
tail -20 /tmp/test_output.log

# Step 5: Count ignored tests (informational)
grep -rn "#\[ignore" crates/context-graph-mcp/src/handlers/tests/ | wc -l
```
