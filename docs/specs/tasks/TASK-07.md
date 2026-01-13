# TASK-07: Convert WorkspaceProvider to async

## STATUS: COMPLETE

**Completed**: 2026-01-13
**Verified By**: Full state verification protocol with before/after logs

**Task ID**: TASK-07 (Original: TASK-PERF-002)
**Layer**: Foundation | **Phase**: 1
**Sequence**: 7
**Estimated Hours**: 2
**Depends On**: TASK-06 (COMPLETE - async-trait added)
**Blocks**: TASK-16 (Remove block_on from gwt_providers)

---

## CRITICAL CONTEXT FOR AI AGENTS

**READ BEFORE IMPLEMENTING**: This task document was audited 2026-01-13 against actual codebase state.

### What This Task ACTUALLY Requires

The `WorkspaceProvider` trait in `gwt_traits.rs` is currently **PARTIALLY async**:
- `select_winning_memory` is ALREADY async (correct)
- These 5 methods are STILL sync and MUST be converted to async:
  - `get_active_memory`
  - `is_broadcasting`
  - `has_conflict`
  - `get_conflict_details`
  - `coherence_threshold`

### Why This Matters (Constitution AP-08)

The sync methods cause the implementation (`WorkspaceProviderImpl` in `gwt_providers.rs`) to use `futures::executor::block_on()` which causes **DEADLOCKS on single-threaded tokio runtime**. This violates Constitution AP-08: "No sync I/O in async context".

### Key Files (Verified Paths)

| File | Purpose |
|------|---------|
| `crates/context-graph-mcp/src/handlers/gwt_traits.rs` | Trait definition to modify |
| `crates/context-graph-mcp/src/handlers/gwt_providers.rs` | Implementation (DO NOT MODIFY IN THIS TASK) |
| `crates/context-graph-core/src/gwt/workspace/global.rs` | GlobalWorkspace (underlying type) |

---

## CURRENT STATE (Verified 2026-01-13)

### gwt_traits.rs Lines 162-194 (Current)

```rust
/// Provider trait for workspace selection operations.
///
/// Handles winner-take-all memory selection for global workspace.
/// TASK-GWT-001: Required for workspace broadcast operations.
#[async_trait]
pub trait WorkspaceProvider: Send + Sync {
    /// Select winning memory via winner-take-all algorithm.
    async fn select_winning_memory(
        &self,
        candidates: Vec<(Uuid, f32, f32, f32)>,
    ) -> CoreResult<Option<Uuid>>;

    /// Get currently active (conscious) memory if broadcasting.
    fn get_active_memory(&self) -> Option<Uuid>;  // <-- SYNC (MUST FIX)

    /// Check if broadcast window is still active.
    fn is_broadcasting(&self) -> bool;  // <-- SYNC (MUST FIX)

    /// Check for workspace conflict (multiple memories with r > 0.8).
    fn has_conflict(&self) -> bool;  // <-- SYNC (MUST FIX)

    /// Get conflicting memory IDs if present.
    fn get_conflict_details(&self) -> Option<Vec<Uuid>>;  // <-- SYNC (MUST FIX)

    /// Get coherence threshold for workspace entry.
    fn coherence_threshold(&self) -> f32;  // <-- SYNC (MUST FIX)
}
```

### gwt_providers.rs Lines 340-365 (Current - Uses block_on)

```rust
fn get_active_memory(&self) -> Option<Uuid> {
    // PROBLEM: block_on causes deadlock
    let workspace = futures::executor::block_on(self.workspace.read());
    workspace.get_active_memory()
}

fn is_broadcasting(&self) -> bool {
    let workspace = futures::executor::block_on(self.workspace.read());
    workspace.is_broadcasting()
}
// ... same pattern for all 5 sync methods
```

---

## DEFINITION OF DONE

### Target Signature (gwt_traits.rs)

```rust
// crates/context-graph-mcp/src/handlers/gwt_traits.rs
use async_trait::async_trait;
use uuid::Uuid;
use context_graph_core::error::CoreResult;

/// Workspace provider trait for GWT integration.
///
/// All methods are async to prevent deadlock with single-threaded runtimes.
/// Constitution: AP-08 ("No sync I/O in async context")
#[async_trait]
pub trait WorkspaceProvider: Send + Sync {
    /// Select winning memory via winner-take-all algorithm.
    ///
    /// # Arguments
    /// - candidates: Vec of (memory_id, order_parameter_r, importance, alignment)
    ///
    /// # Returns
    /// UUID of winning memory, or None if no candidates pass coherence threshold (0.8)
    async fn select_winning_memory(
        &self,
        candidates: Vec<(Uuid, f32, f32, f32)>,
    ) -> CoreResult<Option<Uuid>>;

    /// Get currently active (conscious) memory if broadcasting.
    async fn get_active_memory(&self) -> Option<Uuid>;

    /// Check if broadcast window is still active.
    async fn is_broadcasting(&self) -> bool;

    /// Check for workspace conflict (multiple memories with r > 0.8).
    async fn has_conflict(&self) -> bool;

    /// Get conflicting memory IDs if present.
    async fn get_conflict_details(&self) -> Option<Vec<Uuid>>;

    /// Get coherence threshold for workspace entry.
    async fn coherence_threshold(&self) -> f32;
}
```

### Constraints (MUST ALL PASS)

- [x] Trait MUST have `#[async_trait]` attribute (ALREADY PRESENT)
- [x] ALL 6 methods MUST be `async fn` (NOW 6/6)
- [x] Trait MUST require `Send + Sync` (ALREADY PRESENT)
- [x] Documentation MUST reference AP-08 (Line 168)

---

## IMPLEMENTATION STEPS

### Step 1: Modify gwt_traits.rs (Lines 180-194)

Change the 5 sync methods to async:

```rust
// BEFORE (line 181)
fn get_active_memory(&self) -> Option<Uuid>;

// AFTER
async fn get_active_memory(&self) -> Option<Uuid>;
```

Repeat for: `is_broadcasting`, `has_conflict`, `get_conflict_details`, `coherence_threshold`

### Step 2: Verify Compilation Fails on Implementation

After modifying the trait, `cargo check -p context-graph-mcp` MUST fail with:
```
error[E0706]: functions in traits cannot be declared `async`
   --> crates/context-graph-mcp/src/handlers/gwt_providers.rs:340:5
    |
340 |     fn get_active_memory(&self) -> Option<Uuid> {
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ `async` required
```

This proves the trait change was applied correctly. The implementation fix is TASK-16.

### Step 3: DO NOT FIX gwt_providers.rs

**CRITICAL**: This task ONLY modifies the trait definition. The implementation fix (removing `block_on()`) is TASK-16. If you fix the implementation here, you violate task boundaries.

---

## VERIFICATION COMMANDS

### Primary Verification

```bash
# Step 1: Check trait has all async methods
grep -A 30 "pub trait WorkspaceProvider" crates/context-graph-mcp/src/handlers/gwt_traits.rs \
  | grep -E "^\s*async fn" | wc -l
# Expected: 6 (all 6 methods async)

# Step 2: Verify no sync fn remains in trait
grep -A 30 "pub trait WorkspaceProvider" crates/context-graph-mcp/src/handlers/gwt_traits.rs \
  | grep -E "^\s*fn [a-z]" | wc -l
# Expected: 0 (no sync methods)

# Step 3: Verify async-trait attribute present
grep -B 1 "pub trait WorkspaceProvider" crates/context-graph-mcp/src/handlers/gwt_traits.rs \
  | grep -q "#\[async_trait\]" && echo "PASS: async_trait present" || echo "FAIL"

# Step 4: Verify compilation FAILS (expected - implementation not yet updated)
cargo check -p context-graph-mcp 2>&1 | grep -q "E0706\|async" && echo "EXPECTED: Compilation fails until TASK-16" || echo "WARNING: Check output"
```

### AP-08 Compliance Verification

```bash
# Verify documentation mentions AP-08
grep -q "AP-08" crates/context-graph-mcp/src/handlers/gwt_traits.rs \
  && echo "PASS: AP-08 documented" || echo "FAIL: Add AP-08 reference"
```

---

## FULL STATE VERIFICATION PROTOCOL

### Source of Truth

The source of truth is the `gwt_traits.rs` file trait definition and compilation behavior.

### Execute & Inspect Protocol

1. **Before State Capture**:
```bash
echo "=== BEFORE STATE ===" > /tmp/task07-verification.log
grep -n "fn " crates/context-graph-mcp/src/handlers/gwt_traits.rs \
  | grep -A1 "WorkspaceProvider" >> /tmp/task07-verification.log
cargo check -p context-graph-mcp 2>&1 | head -20 >> /tmp/task07-verification.log
```

2. **After Modification**:
```bash
echo "=== AFTER STATE ===" >> /tmp/task07-verification.log
grep -n "async fn" crates/context-graph-mcp/src/handlers/gwt_traits.rs \
  | grep "get_active_memory\|is_broadcasting\|has_conflict\|coherence_threshold" >> /tmp/task07-verification.log
```

3. **Evidence of Success**:
```bash
cat /tmp/task07-verification.log
# Must show:
# - BEFORE: sync methods (fn get_active_memory, etc.)
# - AFTER: async methods (async fn get_active_memory, etc.)
```

---

## EDGE CASE TESTING

### Edge Case 1: Trait Object Compatibility

```rust
// This MUST still compile after changes
fn accepts_dyn_workspace(provider: &dyn WorkspaceProvider) {
    // async_trait handles dyn compatibility
}
```

**Verification**:
```bash
grep -r "dyn WorkspaceProvider" crates/context-graph-mcp/src/ 2>/dev/null
# If matches exist, verify they still compile after trait change
```

### Edge Case 2: Send + Sync Bounds

```rust
// Verify trait still requires Send + Sync
fn spawn_task<P: WorkspaceProvider + 'static>(p: P) {
    tokio::spawn(async move {
        p.is_broadcasting().await;
    });
}
```

**Verification**: The trait definition MUST include `: Send + Sync`

### Edge Case 3: Return Type Preservation

Verify return types unchanged:
- `get_active_memory` -> `Option<Uuid>` (not `CoreResult<Option<Uuid>>`)
- `is_broadcasting` -> `bool`
- `has_conflict` -> `bool`
- `get_conflict_details` -> `Option<Vec<Uuid>>`
- `coherence_threshold` -> `f32`

---

## MANUAL TESTING WITH SYNTHETIC DATA

### Test 1: Trait Signature Inspection

```bash
# Extract exact trait signature
rust-analyzer --query "WorkspaceProvider" crates/context-graph-mcp/src/handlers/gwt_traits.rs 2>/dev/null || \
grep -A 40 "pub trait WorkspaceProvider" crates/context-graph-mcp/src/handlers/gwt_traits.rs
```

**Expected Output**: All methods show `async fn`

### Test 2: Compilation Behavior

```bash
# Before TASK-16, this SHOULD fail
cargo check -p context-graph-mcp 2>&1 | tee /tmp/check-output.log

# Verify error is about async mismatch
grep -q "method .* should be async" /tmp/check-output.log || \
grep -q "expected async" /tmp/check-output.log || \
grep -q "E0706" /tmp/check-output.log
```

### Test 3: No Backward Compatibility Hacks

```bash
# Verify no compat shims added
grep -r "compat\|legacy\|deprecated\|#\[allow(deprecated)\]" \
  crates/context-graph-mcp/src/handlers/gwt_traits.rs && exit 1 || echo "PASS: No compat hacks"
```

---

## BREAKING CHANGES (INTENTIONAL)

This is a **breaking API change**. The following will fail to compile until TASK-16:

1. `WorkspaceProviderImpl` in `gwt_providers.rs`
2. Any tests calling sync methods without `.await`
3. Any code using `dyn WorkspaceProvider` with sync calls

**THIS IS EXPECTED**. Fixing these is TASK-16, not this task.

---

## DEPENDENCY CHAIN

```
TASK-06 (COMPLETE)     TASK-07 (THIS TASK)     TASK-16 (NEXT)
async-trait added  -->  Trait made async   -->  Implementation async
                        (breaks compilation)    (fixes compilation)
```

---

## COMMON MISTAKES TO AVOID

1. **DO NOT** modify `gwt_providers.rs` - that's TASK-16
2. **DO NOT** add return type wrappers (keep `-> bool` not `-> Result<bool>`)
3. **DO NOT** add default implementations
4. **DO NOT** add backward compatibility shims
5. **DO NOT** suppress compiler errors - they are expected

---

## ROLLBACK PROCEDURE

If task must be reverted:

```bash
git checkout HEAD -- crates/context-graph-mcp/src/handlers/gwt_traits.rs
cargo check -p context-graph-mcp  # Should pass (original state)
```

---

## SUCCESS CRITERIA CHECKLIST

- [x] All 6 WorkspaceProvider methods are `async fn`
- [x] Trait still has `#[async_trait]` attribute
- [x] Trait still requires `Send + Sync`
- [x] Documentation includes AP-08 reference
- [x] `cargo check` fails with async mismatch error (expected until TASK-16)
- [x] No implementation changes in this task
- [x] Verification log captured in /tmp/task07-verification.log
- [x] No backward compatibility hacks added

---

## NEXT TASK

After TASK-07 is complete:
- **TASK-08**: Convert MetaCognitiveProvider to async (parallel, no dependency on TASK-07)
- **TASK-16**: Remove block_on from gwt_providers (depends on TASK-07)

---

*Task Specification v3.0 - Audited 2026-01-13 against actual codebase state*
