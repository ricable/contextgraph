# TASK-CORE-005: Update GoalNode Structure to Use TeleologicalArray

```xml
<task_spec id="TASK-CORE-005" version="6.0" last_audit="2026-01-09">
<metadata>
  <title>Update GoalNode to Use TeleologicalArray (SemanticFingerprint)</title>
  <status>IN_PROGRESS</status><!-- Core implementation DONE, MCP tests FAILING -->
  <layer>foundation</layer>
  <sequence>5</sequence>
  <implements>
    <requirement_ref>REQ-GOAL-REFACTOR-01</requirement_ref>
    <requirement_ref>ARCH-02: Apples-to-apples comparison only</requirement_ref>
    <requirement_ref>ARCH-03: Autonomous-first (no manual goals)</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="COMPLETED">TASK-CORE-001</task_ref>
    <task_ref status="COMPLETED">TASK-CORE-002</task_ref>
    <task_ref status="COMPLETED">TASK-CORE-003</task_ref>
    <task_ref status="COMPLETED">TASK-CORE-004</task_ref>
  </depends_on>
  <blocks>
    <task_ref>TASK-CORE-006</task_ref>
    <task_ref>TASK-LOGIC-009</task_ref>
    <task_ref>TASK-LOGIC-010</task_ref>
    <task_ref>TASK-INTEG-002</task_ref>
  </blocks>
</metadata>

<current_status>
## STATUS: CORE IMPLEMENTATION COMPLETE, MCP TESTS FAILING

### What Is Done (DO NOT REDO)
1. **GoalNode struct updated** (`crates/context-graph-core/src/purpose/goals.rs`)
   - `embedding: Vec<f32>` REMOVED
   - `teleological_array: TeleologicalArray` ADDED
   - Uses `Uuid` for `id` field (not String)
   - Has `parent_id: Option<Uuid>` and `child_ids: Vec<Uuid>`

2. **New types added to goals.rs**:
   - `GoalNodeError` enum with 3 variants
   - `DiscoveryMethod` enum with 4 variants
   - `GoalDiscoveryMetadata` struct with validation

3. **Constructor pattern enforced**:
   - `autonomous_goal()` - primary constructor, validates TeleologicalArray
   - `child_goal()` - for sub-goals with parent reference
   - Old `north_star()` constructor REMOVED

4. **Core tests pass**: `cargo test -p context-graph-core` = 2760 passed, 0 failed

### What Is Broken (FIX REQUIRED)
**35 tests failing in `context-graph-mcp`**

Root cause: MCP handler tests use STRING-based goal IDs like `"ns_knowledge"`,
but GoalNode now uses `Uuid` for IDs.

Failing test files:
- `crates/context-graph-mcp/src/handlers/tests/north_star.rs`
- `crates/context-graph-mcp/src/handlers/tests/purpose.rs`
- `crates/context-graph-mcp/src/handlers/tests/integration_e2e.rs`
- `crates/context-graph-mcp/src/handlers/tests/full_state_verification_purpose.rs`
- `crates/context-graph-mcp/src/handlers/tests/full_state_verification_gwt.rs`
- `crates/context-graph-mcp/src/handlers/tests/full_state_verification.rs`
- `crates/context-graph-mcp/src/handlers/tests/manual_fsv_purpose.rs`
- `crates/context-graph-mcp/src/handlers/tests/manual_fsv_verification.rs`
- `crates/context-graph-mcp/src/handlers/tests/memory.rs`
- `crates/context-graph-mcp/src/handlers/tests/search.rs`
</current_status>

<remaining_work>
## TASK: Fix MCP Handler Tests

### Problem 1: String Goal IDs → Uuid
Tests create goals with string IDs but GoalNode uses Uuid.

**Before (BROKEN)**:
```rust
let goal = GoalNode::autonomous_goal(...);
// Test uses: "goal_id": "ns_knowledge" (string)
```

**After (REQUIRED)**:
```rust
let goal = GoalNode::autonomous_goal(...);
let goal_id = goal.id; // This is a Uuid
// Test uses: "goal_id": goal_id.to_string()
```

### Problem 2: MCP Handler Goal ID Parsing
The MCP handlers need to parse goal_id strings as Uuids.

File: `crates/context-graph-mcp/src/handlers/purpose.rs`

Check the `hierarchy_query` handler - it receives `goal_id` as a string from JSON
and must parse it to Uuid before looking up in GoalHierarchy.

**Fix pattern**:
```rust
let goal_id_str = params.get("goal_id").and_then(|v| v.as_str())
    .ok_or_else(|| ...)?;
let goal_id = Uuid::parse_str(goal_id_str)
    .map_err(|e| RpcError::invalid_params(format!("Invalid goal_id UUID: {}", e)))?;
```

### Problem 3: Test Fixtures Use Real Fingerprints
Tests must NOT mock fingerprints. Use:
- `SemanticFingerprint::zeroed()` for valid empty fingerprints
- Real `GoalDiscoveryMetadata::bootstrap()` for discovery metadata

### Files to Fix

| File | Issue |
|------|-------|
| `handlers/tests/north_star.rs` | Uses string goal IDs, needs Uuid conversion |
| `handlers/tests/purpose.rs` | Same issue |
| `handlers/tests/integration_e2e.rs` | Same issue |
| `handlers/tests/full_state_verification_*.rs` | Same issue |
| `handlers/tests/manual_fsv_*.rs` | Same issue |
| `handlers/purpose.rs` | May need Uuid parsing for goal_id params |
</remaining_work>

<verification_commands>
## Verification Steps

### Step 1: Check Core (should already pass)
```bash
cargo test -p context-graph-core
# Expected: 2760 passed, 0 failed
```

### Step 2: Check MCP after fixes
```bash
cargo test -p context-graph-mcp
# Expected: All tests pass (currently 35 failing)
```

### Step 3: Verify no old patterns
```bash
# Must return NO matches
rg "embedding: Vec<f32>" crates/context-graph-core/src/purpose/goals.rs

# Must return NO matches
rg 'goal_id.*=.*"[a-z_]+"' crates/context-graph-mcp/src/handlers/tests/

# Should find the new field
rg "teleological_array: TeleologicalArray" crates/context-graph-core/src/purpose/goals.rs
```

### Step 4: Full test suite
```bash
cargo test --workspace
# Expected: All tests pass
```
</verification_commands>

<source_of_truth>
## Source of Truth Locations

| Component | Location | Expected State |
|-----------|----------|----------------|
| GoalNode struct | `src/purpose/goals.rs:230-261` | Has teleological_array, id is Uuid |
| GoalNodeError | `src/purpose/goals.rs:42-60` | 3 variants |
| DiscoveryMethod | `src/purpose/goals.rs:62-76` | 4 variants |
| GoalDiscoveryMetadata | `src/purpose/goals.rs:78-139` | Has validation |
| autonomous_goal() | `src/purpose/goals.rs:291-310` | Returns Result, validates array |
| child_goal() | `src/purpose/goals.rs:323-347` | Returns Result, validates array |
| GoalHierarchy | `src/purpose/goals.rs:387+` | Uses Uuid keys |
</source_of_truth>

<full_state_verification_protocol>
## FSV Protocol (Post-Fix)

### 1. Execute All Tests
```bash
cargo test --workspace 2>&1 | tee /tmp/test_results.txt
grep -E "(passed|failed)" /tmp/test_results.txt
```

### 2. Verify GoalNode Structure
```bash
echo "=== GoalNode Fields ===" > /tmp/fsv.txt
rg -A 15 "pub struct GoalNode" crates/context-graph-core/src/purpose/goals.rs >> /tmp/fsv.txt

echo "=== No Old Embedding Field ===" >> /tmp/fsv.txt
rg "embedding: Vec<f32>" crates/context-graph-core/src/purpose/goals.rs >> /tmp/fsv.txt || \
  echo "CONFIRMED: No old embedding field" >> /tmp/fsv.txt

cat /tmp/fsv.txt
```

### 3. Edge Case Tests (Must Exist and Pass)

**Edge Case 1: Incomplete fingerprint rejected**
```rust
#[test]
fn test_incomplete_fingerprint_rejected() {
    let mut fp = SemanticFingerprint::zeroed();
    fp.e1_semantic = vec![]; // Invalid
    let discovery = GoalDiscoveryMetadata::bootstrap();
    let result = GoalNode::autonomous_goal(
        "Test".into(), GoalLevel::NorthStar, fp, discovery
    );
    assert!(matches!(result, Err(GoalNodeError::InvalidArray(_))));
}
```

**Edge Case 2: Invalid confidence rejected**
```rust
#[test]
fn test_invalid_confidence_rejected() {
    let result = GoalDiscoveryMetadata::new(
        DiscoveryMethod::Clustering, 1.5, 10, 0.8 // confidence > 1.0
    );
    assert!(matches!(result, Err(GoalNodeError::InvalidConfidence(_))));
}
```

**Edge Case 3: Multiple North Stars rejected**
```rust
#[test]
fn test_multiple_north_stars_rejected() {
    let fp = SemanticFingerprint::zeroed();
    let discovery = GoalDiscoveryMetadata::bootstrap();
    let mut hierarchy = GoalHierarchy::new();

    let ns1 = GoalNode::autonomous_goal("NS1".into(), GoalLevel::NorthStar, fp.clone(), discovery.clone()).unwrap();
    let ns2 = GoalNode::autonomous_goal("NS2".into(), GoalLevel::NorthStar, fp, discovery).unwrap();

    hierarchy.add_goal(ns1).unwrap();
    let result = hierarchy.add_goal(ns2);
    assert!(matches!(result, Err(GoalHierarchyError::MultipleNorthStars)));
}
```
</full_state_verification_protocol>

<critical_rules>
## CRITICAL RULES

### NO BACKWARDS COMPATIBILITY
- Do NOT add fallback to old `embedding: Vec<f32>` field
- Do NOT add migration shims for string-based goal IDs
- Do NOT create default implementations that mask errors
- FAIL FAST if anything is wrong

### NO MOCK DATA IN TESTS
- Use `SemanticFingerprint::zeroed()` for valid empty fingerprints
- Use `GoalDiscoveryMetadata::bootstrap()` for discovery metadata
- Real validation must run - tests must prove the system works

### ERROR LOGGING
All errors must be specific and actionable:
```
GoalNodeError::InvalidArray(DimensionMismatch { embedder: Semantic, expected: 1024, actual: 0 })
// Actionable: "E1_semantic must have 1024 dimensions, got 0"
```
</critical_rules>

<constitution_alignment>
## Constitution Alignment Verification

| Rule | Status | Evidence |
|------|--------|----------|
| ARCH-02: Apples-to-apples | ENABLED | GoalNode has TeleologicalArray |
| ARCH-03: Autonomous-first | ENFORCED | Only autonomous_goal() constructor exists |
| ARCH-05: All 13 embedders | ENFORCED | validate_strict() called in constructors |
| AP-14: No .unwrap() | FOLLOWED | All constructors return Result |
</constitution_alignment>

<files_modified_in_working_tree>
## Files Already Modified (Uncommitted)

These files have been changed from HEAD:

### Core Crate (ALL PASSING)
- `crates/context-graph-core/src/purpose/goals.rs` - GoalNode refactored
- `crates/context-graph-core/src/purpose/mod.rs` - Re-exports updated
- `crates/context-graph-core/src/purpose/tests.rs` - Tests updated
- `crates/context-graph-core/src/purpose/default_computer.rs` - Uses goal.array()
- `crates/context-graph-core/src/alignment/calculator.rs` - Per-embedder comparison
- `crates/context-graph-core/src/alignment/tests.rs` - Uses new API
- `crates/context-graph-core/src/alignment/config.rs` - Uses new API
- `crates/context-graph-core/src/alignment/error.rs` - Updated
- `crates/context-graph-core/src/alignment/misalignment.rs` - Updated
- `crates/context-graph-core/src/alignment/pattern.rs` - Updated
- `crates/context-graph-core/src/alignment/score.rs` - Updated
- `crates/context-graph-core/src/index/purpose/*.rs` - Updated
- `crates/context-graph-core/src/retrieval/pipeline.rs` - Uses new API

### MCP Crate (35 TESTS FAILING)
- `crates/context-graph-mcp/src/handlers/purpose.rs` - May need Uuid parsing
- `crates/context-graph-mcp/src/handlers/autonomous.rs` - Updated
- `crates/context-graph-mcp/src/handlers/tools.rs` - Updated
- `crates/context-graph-mcp/src/handlers/tests/*.rs` - NEED FIXES

### Storage Crate (PASSING)
- `crates/context-graph-storage/tests/purpose_vector_integration.rs` - Updated
</files_modified_in_working_tree>

<next_steps>
## Immediate Next Steps

1. **Read the failing test files** in `crates/context-graph-mcp/src/handlers/tests/`
2. **Identify goal_id usage patterns** - find all places using string goal IDs
3. **Update tests** to use Uuid-based goal IDs:
   - Create goals with `GoalNode::autonomous_goal()`
   - Use `goal.id.to_string()` when passing to MCP handlers
   - Parse goal_id as Uuid in handler code if needed
4. **Run tests incrementally**: `cargo test -p context-graph-mcp handlers::tests::north_star`
5. **Verify all 367 MCP tests pass** before committing
</next_steps>

</task_spec>
```
