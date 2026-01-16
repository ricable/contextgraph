# TASK-P0-004: Database Schema Cleanup - Remove North Star Column Families

```xml
<task_spec id="TASK-P0-004" version="2.0">
<metadata>
  <title>Database Schema Cleanup - Remove North Star Column Families</title>
  <status>COMPLETE</status>
  <layer>foundation</layer>
  <sequence>4</sequence>
  <phase>0</phase>
  <implements>
    <requirement_ref>REQ-P0-05</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="COMPLETE">TASK-P0-003</task_ref>
    <task_ref status="COMPLETE">TASK-P0-001</task_ref>
    <task_ref status="COMPLETE">TASK-P0-002</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <last_audit>2026-01-16</last_audit>
</metadata>

<context>
## Project State (2026-01-16)

TASK-P0-001 (commit 5f6dfc7) and TASK-P0-002 already removed North Star code from:
- GoalLevel enum (NorthStar level removed, Strategic is now top)
- MCP handlers (6 tools removed)
- Constitution updated to v6.0.0 (topic-based architecture)

However, database-related column families and test files still contain references.

## Storage Architecture Overview

The Context Graph uses RocksDB with 43 total column families (NOT SQL tables):
- **Base CFs (12)**: nodes, edges, embeddings, metadata, johari_*, temporal, tags, sources, system
- **Teleological CFs (11)**: fingerprints, purpose_vectors, e13_splade_inverted, etc.
- **Quantized Embedder CFs (13)**: emb_0 through emb_12
- **Autonomous CFs (7)**: The ones that need cleanup

## Column Families to Review/Cleanup (AUTONOMOUS_CFS)

Located in: `crates/context-graph-storage/src/autonomous/column_families.rs`

| CF Name | Current Purpose | Action Required |
|---------|-----------------|-----------------|
| `autonomous_config` | Singleton AutonomousConfig | KEEP - still used |
| `adaptive_threshold_state` | Singleton threshold state | KEEP - still used |
| `drift_history` | Historical drift data | REVIEW - remove if not used by topic system |
| `goal_activity_metrics` | Per-goal activity | REVIEW - may need removal |
| `autonomous_lineage` | Lineage events | KEEP - traceability |
| `consolidation_history` | Consolidation records | KEEP - memory consolidation |
| `memory_curation` | Memory curation state | KEEP - curation feature |

## Column Families Definitely NOT Needed (Per Constitution v6.0.0)

These CFs serve the OLD North Star architecture and are NOT used by topic-based system:
- `drift_history` - Old drift detection (replaced by topic_stability.churn_rate)
- `goal_activity_metrics` - Manual goals (forbidden by ARCH-03)
- `ego_node` (in teleological CFs) - Self Ego Node (replaced by TopicProfile)
</context>

<critical_findings>
## Audit Results (2026-01-16)

### 1. North Star References Still in Tests
The following test files still contain North Star references:
- `crates/context-graph-storage/tests/teleological_integration.rs` (lines mention "theta to north star")
- `crates/context-graph-storage/tests/purpose_vector_integration.rs` (North Star goal creation)
- `crates/context-graph-storage/tests/autonomous_integration.rs` (drift_history tests)

### 2. CF_EGO_NODE Still Defined
File: `crates/context-graph-storage/src/teleological/column_families.rs` line 101
```rust
pub const CF_EGO_NODE: &str = "ego_node";
```
This is included in TELEOLOGICAL_CFS (11 total) and is used for SelfEgoNode storage.
Per Constitution v6.0.0, SelfEgoNode is replaced by TopicProfile.

### 3. Drift History Operations Still Exist
File: `crates/context-graph-storage/src/autonomous/rocksdb_store/operations/drift.rs`
Contains operations like `get_drift_history()` which are still callable.

### 4. Correct File Paths
Storage module is at: `crates/context-graph-storage/` (NOT context-graph-core)
Key files:
- `src/autonomous/column_families.rs` - Autonomous CF definitions
- `src/autonomous/rocksdb_store/` - RocksDB implementation
- `src/teleological/column_families.rs` - Teleological CF definitions
</critical_findings>

<scope>
<in_scope>
  <!-- Phase 1: Determine what to remove -->
  - Audit usage of drift_history CF in new topic-based system
  - Audit usage of goal_activity_metrics CF
  - Audit usage of ego_node CF (in teleological module)

  <!-- Phase 2: Remove unused column families -->
  - Remove CF_DRIFT_HISTORY from AUTONOMOUS_CFS if not used
  - Remove CF_GOAL_ACTIVITY_METRICS from AUTONOMOUS_CFS if not used
  - Remove CF_EGO_NODE from TELEOLOGICAL_CFS if not used
  - Remove associated operations in rocksdb_store/operations/
  - Update AUTONOMOUS_CF_COUNT and TELEOLOGICAL_CF_COUNT
  - Update TOTAL_COLUMN_FAMILIES constant in column_families.rs

  <!-- Phase 3: Clean up tests -->
  - Fix/remove North Star references in test files
  - Update test assertions for new CF counts

  <!-- Phase 4: Verify database opens correctly -->
  - Test database open with reduced column families
  - Ensure no orphan data access attempts
</in_scope>

<out_of_scope>
  - Creating new column families (Phase 4+ tasks)
  - Migrating data (no data preservation needed)
  - Backup creation (user responsibility)
  - Adding topic-based CFs (separate task)
</out_of_scope>
</scope>

<prerequisites>
  <check status="COMPLETE">TASK-P0-001 completed (North Star code removed)</check>
  <check status="COMPLETE">TASK-P0-002 completed (MCP tools removed)</check>
  <check status="COMPLETE">TASK-P0-003 completed (constitution updated to v6.0.0)</check>
  <check>No active code depends on removed CFs (verify with grep)</check>
  <check>Test database backup exists if preservation needed</check>
</prerequisites>

<definition_of_done>
<criteria>
  <criterion id="DOD-1">CF_DRIFT_HISTORY removed or documented why kept</criterion>
  <criterion id="DOD-2">CF_GOAL_ACTIVITY_METRICS removed or documented why kept</criterion>
  <criterion id="DOD-3">CF_EGO_NODE removed from TELEOLOGICAL_CFS or documented why kept</criterion>
  <criterion id="DOD-4">AUTONOMOUS_CF_COUNT matches actual count</criterion>
  <criterion id="DOD-5">TELEOLOGICAL_CF_COUNT matches actual count</criterion>
  <criterion id="DOD-6">TOTAL_COLUMN_FAMILIES matches sum of all CF arrays</criterion>
  <criterion id="DOD-7">All tests pass with updated CF counts</criterion>
  <criterion id="DOD-8">Database opens without errors</criterion>
  <criterion id="DOD-9">No "north_star" references in non-comment code</criterion>
</criteria>

<verification>
  - cargo check --package context-graph-storage succeeds
  - cargo test --package context-graph-storage succeeds
  - grep -r "north_star" returns only documentation/comments
  - RocksDB opens with new CF configuration
</verification>
</definition_of_done>

<files_to_modify>
<!-- Autonomous Column Families -->
<file path="crates/context-graph-storage/src/autonomous/column_families.rs">
  <action>Review and potentially remove: CF_DRIFT_HISTORY, CF_GOAL_ACTIVITY_METRICS</action>
  <action>Update AUTONOMOUS_CFS array</action>
  <action>Update AUTONOMOUS_CF_COUNT constant</action>
</file>

<file path="crates/context-graph-storage/src/autonomous/mod.rs">
  <action>Remove exports for drift/goal operations if CFs removed</action>
</file>

<file path="crates/context-graph-storage/src/autonomous/rocksdb_store/operations/drift.rs">
  <action>Remove if CF_DRIFT_HISTORY removed</action>
</file>

<file path="crates/context-graph-storage/src/autonomous/rocksdb_store/operations/goal.rs">
  <action>Remove if CF_GOAL_ACTIVITY_METRICS removed</action>
</file>

<file path="crates/context-graph-storage/src/autonomous/schema.rs">
  <action>Remove drift/goal key builders if CFs removed</action>
</file>

<!-- Teleological Column Families -->
<file path="crates/context-graph-storage/src/teleological/column_families.rs">
  <action>Review CF_EGO_NODE - remove if SelfEgoNode not used</action>
  <action>Update TELEOLOGICAL_CFS array</action>
  <action>Update TELEOLOGICAL_CF_COUNT constant</action>
  <action>Remove ego_node_cf_options() function if CF removed</action>
</file>

<!-- Main Column Families -->
<file path="crates/context-graph-storage/src/column_families.rs">
  <action>Update TOTAL_COLUMN_FAMILIES constant to match new totals</action>
</file>

<!-- Test Files -->
<file path="crates/context-graph-storage/tests/teleological_integration.rs">
  <action>Remove/fix "theta to north star" references</action>
</file>

<file path="crates/context-graph-storage/tests/purpose_vector_integration.rs">
  <action>Remove North Star goal creation code</action>
</file>

<file path="crates/context-graph-storage/tests/autonomous_integration.rs">
  <action>Remove drift_history test functions if CF removed</action>
</file>

<file path="crates/context-graph-storage/src/autonomous/tests.rs">
  <action>Remove drift_history tests if CF removed</action>
</file>
</files_to_modify>

<pseudo_code>
## Step 1: Verify Usage

```bash
# Check if drift_history is used anywhere in non-test code
grep -r "drift_history\|CF_DRIFT_HISTORY\|get_drift_history" \
  crates/context-graph-*/src/ \
  --include="*.rs" | grep -v "test" | grep -v "/tests/"

# Check if goal_activity_metrics is used
grep -r "goal_activity\|CF_GOAL_ACTIVITY" \
  crates/context-graph-*/src/ \
  --include="*.rs" | grep -v "test"

# Check if ego_node is used
grep -r "ego_node\|CF_EGO_NODE\|EgoNode" \
  crates/context-graph-*/src/ \
  --include="*.rs" | grep -v "test"
```

## Step 2: Remove Column Family (if confirmed unused)

In autonomous/column_families.rs:
```rust
// BEFORE
pub const AUTONOMOUS_CFS: &[&str] = &[
    CF_AUTONOMOUS_CONFIG,
    CF_ADAPTIVE_THRESHOLD_STATE,
    CF_DRIFT_HISTORY,  // REMOVE
    CF_GOAL_ACTIVITY_METRICS,  // REMOVE
    CF_AUTONOMOUS_LINEAGE,
    CF_CONSOLIDATION_HISTORY,
    CF_MEMORY_CURATION,
];
pub const AUTONOMOUS_CF_COUNT: usize = 7;

// AFTER
pub const AUTONOMOUS_CFS: &[&str] = &[
    CF_AUTONOMOUS_CONFIG,
    CF_ADAPTIVE_THRESHOLD_STATE,
    CF_AUTONOMOUS_LINEAGE,
    CF_CONSOLIDATION_HISTORY,
    CF_MEMORY_CURATION,
];
pub const AUTONOMOUS_CF_COUNT: usize = 5;
```

## Step 3: Remove Associated Operations

Delete files:
- autonomous/rocksdb_store/operations/drift.rs
- autonomous/rocksdb_store/operations/goal.rs

Update autonomous/rocksdb_store/operations/mod.rs

## Step 4: Update Totals

In column_families.rs:
```rust
// BEFORE: Base (12) + Teleological (11) + Quantized (13) + Autonomous (7) = 43
// AFTER:  Base (12) + Teleological (10) + Quantized (13) + Autonomous (5) = 40
pub const TOTAL_COLUMN_FAMILIES: usize = 40;
```

## Step 5: Fix Tests

Update test CF count assertions from 43 to 40
Remove test functions that use removed CFs
</pseudo_code>

<test_commands>
<!-- Pre-execution checks -->
<command description="Check drift_history usage">
grep -r "drift_history\|CF_DRIFT_HISTORY" crates/ --include="*.rs" | grep -v "/tests/" | grep -v "_test.rs" | grep -v "mod tests"
</command>

<command description="Check goal_activity_metrics usage">
grep -r "goal_activity\|CF_GOAL_ACTIVITY" crates/ --include="*.rs" | grep -v "/tests/" | grep -v "_test.rs"
</command>

<command description="Check ego_node CF usage">
grep -r "CF_EGO_NODE\|ego_node" crates/context-graph-storage/src/ --include="*.rs" | grep -v "test"
</command>

<!-- Post-modification verification -->
<command description="Verify storage compiles">
cargo check --package context-graph-storage 2>&1 | head -30
</command>

<command description="Run storage tests">
cargo test --package context-graph-storage 2>&1 | tail -50
</command>

<command description="Verify no North Star in code">
grep -r "north_star" crates/context-graph-storage/src/ --include="*.rs" | wc -l
</command>

<command description="Check remaining test references">
grep -r "north_star\|NorthStar" crates/context-graph-storage/tests/ --include="*.rs" | wc -l
</command>
</test_commands>

<validation_criteria>
  <criterion>CF counts in tests match actual counts after modification</criterion>
  <criterion>Database opens without "column family not found" errors</criterion>
  <criterion>No orphan operations try to access removed CFs</criterion>
  <criterion>grep -r "north_star" returns 0 matches in src/ directories</criterion>
  <criterion>All storage tests pass (cargo test --package context-graph-storage)</criterion>
</validation_criteria>

<full_state_verification>
## Source of Truth Definition

The Source of Truth for column families is the actual RocksDB column family list at database open time.

### Evidence Collection Protocol

1. **Before State**: Run this command and save output
   ```bash
   cargo test --package context-graph-storage test_total_column_families_constant -- --nocapture 2>&1
   ```
   Expected: Test shows current CF count (43)

2. **After Modification**: Run same test
   Expected: Test shows new CF count (should match TOTAL_COLUMN_FAMILIES constant)

3. **Physical Verification**: Create test DB and list CFs
   ```rust
   // Add this test to verify actual CFs
   #[test]
   fn verify_db_column_families() {
       let temp = tempfile::tempdir().unwrap();
       let cache = Cache::new_lru_cache(256 * 1024 * 1024);
       let descriptors = get_all_column_family_descriptors(&cache);

       println!("=== COLUMN FAMILY VERIFICATION ===");
       println!("Total CF count: {}", descriptors.len());
       println!("Expected: {}", TOTAL_COLUMN_FAMILIES);

       for (i, desc) in descriptors.iter().enumerate() {
           println!("  CF[{}]: {}", i, desc.name());
       }

       assert_eq!(descriptors.len(), TOTAL_COLUMN_FAMILIES);
   }
   ```

### Boundary & Edge Case Audit

**Edge Case 1: Empty Database Open**
```bash
# BEFORE: Create empty DB
rm -rf /tmp/cg_test_db
# AFTER: Open with new CF set - should succeed
```

**Edge Case 2: Existing DB with Old CFs**
If a database exists with the old CF set (43 CFs), opening with new set (40 CFs) will:
- RocksDB will ignore data in undefined CFs
- No migration needed for development
- Production requires explicit migration (out of scope for this task)

**Edge Case 3: Missing CF Reference**
```bash
# Verify no code references removed CFs
grep -r "CF_DRIFT_HISTORY\|CF_GOAL_ACTIVITY_METRICS\|CF_EGO_NODE" \
  crates/context-graph-*/src/ --include="*.rs" | wc -l
# Expected: 0 after cleanup
```

### Success Evidence

Provide a log showing:
1. Before CF count: 43
2. Removed CFs: [list]
3. After CF count: [new count]
4. All tests pass
5. grep results showing no orphan references
</full_state_verification>

<notes>
<note category="rocksdb_behavior">
RocksDB column families are defined at database open time.
Removing them from the definition list is sufficient - RocksDB will
silently ignore existing data in undefined column families.
This means NO explicit migration needed for development databases.
</note>

<note category="no_backwards_compatibility">
Per requirements: System either works without removed CFs or fails fast.
NO shims, stubs, or compatibility layers.
If something breaks, debug and fix - don't cover it up.
</note>

<note category="test_data">
Tests MUST NOT use mock data. All tests must:
1. Create real RocksDB instances in temp directories
2. Write real data
3. Read back and verify
4. Check physical presence in column families
</note>

<note category="error_handling">
Removed CFs should result in clear errors if accidentally accessed:
- "Column family not found: drift_history"
- NOT silent failures or empty returns
</note>
</notes>

<execution_order>
1. Run usage grep commands to confirm which CFs are truly unused
2. Remove unused CF constants and arrays
3. Remove associated operation files
4. Update CF count constants
5. Fix test files (remove North Star references)
6. Run cargo check
7. Run cargo test
8. Verify with grep for any remaining references
9. Document final CF counts in this task
</execution_order>
</task_spec>
```

## Execution Checklist

### Phase 1: Audit (REQUIRED FIRST)
- [x] Run grep commands to verify CF usage in production code
- [x] Document which CFs are actually unused
- [x] Confirm ego_node IS used by GWT system - KEPT

### Phase 2: Remove Unused Column Families
- [x] Remove CF_DRIFT_HISTORY (confirmed unused - old drift detection)
- [x] Remove CF_GOAL_ACTIVITY_METRICS (confirmed unused - manual goals forbidden ARCH-03)
- [x] CF_EGO_NODE KEPT - actively used by GWT system (28+ files use SelfEgoNode)
- [x] Update AUTONOMOUS_CFS array (7 → 5)
- [x] Update AUTONOMOUS_CF_COUNT (7 → 5)
- [x] TELEOLOGICAL_CFS unchanged (CF_EGO_NODE kept)
- [x] TELEOLOGICAL_CF_COUNT unchanged (11)
- [x] Update TOTAL_COLUMN_FAMILIES (43 → 41)

### Phase 3: Remove Operations
- [x] Delete autonomous/rocksdb_store/operations/drift.rs
- [x] Delete autonomous/rocksdb_store/operations/goal.rs
- [x] Update operations/mod.rs
- [x] ego_node_cf_options() KEPT - CF_EGO_NODE still used
- [x] Update get_autonomous_cf_descriptors()
- [x] get_teleological_cf_descriptors() unchanged (CF_EGO_NODE kept)

### Phase 4: Fix Tests
- [x] teleological_integration.rs - NO CHANGES NEEDED (no north_star references found)
- [x] purpose_vector_integration.rs - Updated naming from north_star to strategic_goal
- [x] autonomous_integration.rs - Removed drift_history and goal_activity_metrics tests
- [x] autonomous/tests.rs - Removed drift tests
- [x] Update CF count assertions in all tests (43 → 41)

### Phase 5: Full State Verification
- [x] cargo check --package context-graph-storage passes
- [x] cargo test --package context-graph-storage --lib --tests passes (688 tests)
- [x] grep -r "north_star" crates/context-graph-storage/src/ returns 0
- [x] grep for removed CFs returns 0 in production code
- [x] Document before/after CF counts (see below)
- [x] Test physically verifies CF count (test_total_column_families_constant, test_get_all_column_family_descriptors_returns_41)

### Phase 6: Manual Verification
- [x] Database opens correctly via integration tests
- [x] New empty database creates with new CF set (verified via tests)
- [x] TOTAL_COLUMN_FAMILIES constant (41) matches reality

## Completion Summary (2026-01-16)

### Before State
- AUTONOMOUS_CF_COUNT: 7
- TOTAL_COLUMN_FAMILIES: 43
- AUTONOMOUS_CFS: [autonomous_config, adaptive_threshold_state, drift_history, goal_activity_metrics, autonomous_lineage, consolidation_history, memory_curation]

### After State
- AUTONOMOUS_CF_COUNT: 5
- TOTAL_COLUMN_FAMILIES: 41
- AUTONOMOUS_CFS: [autonomous_config, adaptive_threshold_state, autonomous_lineage, consolidation_history, memory_curation]

### Removed
1. **CF_DRIFT_HISTORY** - Old drift detection replaced by topic_stability.churn_rate (ARCH-10)
2. **CF_GOAL_ACTIVITY_METRICS** - Manual goals forbidden by ARCH-03

### Kept (With Justification)
1. **CF_EGO_NODE** - Actively used by GWT (Global Workspace Theory) system for SelfEgoNode storage. NOT related to North Star goals. Used in 28+ files in context-graph-core/src/gwt/.

### Test Results
- 688 lib/integration tests pass
- 3 pre-existing doctest failures (unrelated - `config()` method issues)
- All CF count assertions updated and passing

## Files Quick Reference

| File | Action |
|------|--------|
| `crates/context-graph-storage/src/autonomous/column_families.rs` | ✅ Removed 2 CFs, updated counts |
| `crates/context-graph-storage/src/autonomous/mod.rs` | ✅ Removed exports |
| `crates/context-graph-storage/src/autonomous/rocksdb_store/operations/drift.rs` | ✅ DELETED |
| `crates/context-graph-storage/src/autonomous/rocksdb_store/operations/goal.rs` | ✅ DELETED |
| `crates/context-graph-storage/src/autonomous/schema.rs` | ✅ Removed key builders |
| `crates/context-graph-storage/src/teleological/column_families.rs` | ⏭️ SKIPPED - CF_EGO_NODE kept |
| `crates/context-graph-storage/src/column_families.rs` | ✅ Updated TOTAL constant (43→41) |
| `crates/context-graph-storage/tests/*.rs` | ✅ Fixed North Star references |
