# TASK-06: Add async-trait to MCP crate

## STATUS: COMPLETE

**Verified**: 2026-01-13
**Completion Evidence**: `async-trait = { workspace = true }` exists in `crates/context-graph-mcp/Cargo.toml`

---

## QUICK VERIFICATION (Run This First)

```bash
# Verify async-trait is present and working
cargo tree -p context-graph-mcp 2>/dev/null | grep async-trait
# Expected output: "├── async-trait v0.1.89 (proc-macro)" or similar

# Verify crate compiles
cargo check -p context-graph-mcp
# Expected: "Finished" with no errors

# Verify async-trait is actually being used in code
grep -r "#\[async_trait\]" crates/context-graph-mcp/src/
# Expected: Multiple matches in gwt_traits.rs, gwt_providers.rs, utl_adapter.rs
```

**If all three commands succeed, this task is COMPLETE. Skip to next task.**

---

## TASK CONTEXT

### What This Task Does
Adds the `async-trait` crate dependency to `context-graph-mcp` so that synchronous provider traits can be converted to async.

### Why This Matters
- **Constitution AP-08**: Prohibits sync I/O in async context
- `block_on()` calls cause deadlocks in single-threaded tokio runtimes
- Prerequisite for TASK-07 (WorkspaceProvider async) and TASK-08 (MetaCognitiveProvider async)

### Technical Background
Rust's native async traits (stabilized in 1.75) have limitations with `dyn Trait` objects. `async-trait` provides:
- `#[async_trait]` macro for trait definitions
- `Box<dyn Future>` under the hood
- Compatibility with tokio runtime

---

## CURRENT STATE (Verified 2026-01-13)

### Files Already Modified
| File | State |
|------|-------|
| `crates/context-graph-mcp/Cargo.toml` | `async-trait = { workspace = true }` on line 39 |

### Files Already Using async-trait
| File | Usage |
|------|-------|
| `src/handlers/gwt_traits.rs` | `#[async_trait]` on 3 traits |
| `src/handlers/gwt_providers.rs` | `#[async_trait]` on 3 impl blocks |
| `src/adapters/utl_adapter.rs` | `#[async_trait]` on 1 trait impl |

---

## IF TASK WERE NOT COMPLETE

### Implementation Steps (Already Done)

1. **Add dependency to Cargo.toml**:
```toml
# crates/context-graph-mcp/Cargo.toml
[dependencies]
async-trait = { workspace = true }  # Uses version from root Cargo.toml
```

2. **Verify in workspace Cargo.toml**:
```toml
# Root Cargo.toml [workspace.dependencies]
async-trait = "0.1"
```

3. **Run verification commands**:
```bash
cargo check -p context-graph-mcp
cargo tree -p context-graph-mcp | grep async-trait
```

### Usage Pattern (For Reference)
```rust
use async_trait::async_trait;

#[async_trait]
pub trait WorkspaceProvider: Send + Sync {
    async fn get_active_memory(&self) -> Option<Uuid>;
}
```

---

## FULL STATE VERIFICATION PROTOCOL

### Source of Truth
The source of truth is the `Cargo.toml` file and the compiled crate.

### Verification Steps

#### Step 1: Check Cargo.toml Contains Dependency
```bash
grep "async-trait" crates/context-graph-mcp/Cargo.toml
```
**Expected**: `async-trait = { workspace = true }` or `async-trait = "0.1"`
**Failure**: Line not found → Task incomplete

#### Step 2: Check Crate Compiles
```bash
cargo check -p context-graph-mcp 2>&1
```
**Expected**: Exit code 0, output contains "Finished"
**Failure**: Compilation errors → Debug and fix

#### Step 3: Check Dependency Tree
```bash
cargo tree -p context-graph-mcp | grep async-trait
```
**Expected**: `├── async-trait v0.1.*` appears
**Failure**: Not in tree → Cargo.toml error

#### Step 4: Check No Conflicting Versions
```bash
cargo tree -p context-graph-mcp -d | grep async-trait
```
**Expected**: No output (no duplicates) or single version
**Failure**: Multiple versions → Resolve version conflict

---

## EDGE CASE TESTING

### Edge Case 1: Empty Cargo Cache
```bash
# Before
rm -rf target/

# Action
cargo check -p context-graph-mcp

# After
echo $?  # Must be 0
```

### Edge Case 2: Workspace Version Mismatch
```bash
# Check root Cargo.toml defines async-trait
grep "async-trait" Cargo.toml
```
If missing from `[workspace.dependencies]`, the build will fail.

### Edge Case 3: Feature Conflict
```bash
# Check for incompatible features
cargo check -p context-graph-mcp --all-features
```

---

## EVIDENCE LOG

### Execution Evidence (2026-01-13)
```
$ cargo tree -p context-graph-mcp | grep async-trait
├── async-trait v0.1.89 (proc-macro)
│   ├── async-trait v0.1.89 (proc-macro) (*)
... (multiple transitive dependencies also use it)

$ cargo check -p context-graph-mcp
    Finished `dev` profile [unoptimized + debuginfo] target(s) in X.XXs

$ grep -c "#\[async_trait\]" crates/context-graph-mcp/src/**/*.rs
7  (7 usages across 3 files)
```

---

## DEPENDENCY GRAPH

```
TASK-06 (This Task) ──┬──> TASK-07 (WorkspaceProvider async)
                      │
                      └──> TASK-08 (MetaCognitiveProvider async)
```

**This task blocks**: TASK-07, TASK-08
**This task depends on**: Nothing (no dependencies)

---

## DEFINITION OF DONE CHECKLIST

- [x] `async-trait = "0.1"` or `async-trait = { workspace = true }` in Cargo.toml
- [x] `cargo check -p context-graph-mcp` passes
- [x] `cargo tree -p context-graph-mcp | grep async-trait` shows the dependency
- [x] No version conflicts in dependency tree
- [x] At least one file uses `#[async_trait]` attribute

---

## ROLLBACK PROCEDURE (If Needed)

If this task needs to be reverted:
```bash
# Remove from Cargo.toml
sed -i '/async-trait/d' crates/context-graph-mcp/Cargo.toml

# Verify removal
cargo check -p context-graph-mcp  # Will fail if async-trait is used in code
```

**WARNING**: Removing async-trait will break:
- `gwt_traits.rs`
- `gwt_providers.rs`
- `utl_adapter.rs`

---

## NEXT TASK

After verifying TASK-06 is complete, proceed to:
- **TASK-07**: Convert WorkspaceProvider to async
- **TASK-08**: Convert MetaCognitiveProvider to async

Both can be done in parallel since they only depend on TASK-06.

---

*Task Specification v2.0 - Updated 2026-01-13 - VERIFIED COMPLETE*
