# TASK-P0-004: Database Migrations (Drop Tables)

```xml
<task_spec id="TASK-P0-004" version="1.0">
<metadata>
  <title>Database Migrations - Drop North Star Tables</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>4</sequence>
  <phase>0</phase>
  <implements>
    <requirement_ref>REQ-P0-05</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P0-003</task_ref>
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<context>
The North Star system stores data in database tables that are no longer needed.
This task creates and runs migrations to drop these tables.

CRITICAL: This is a destructive operation. All data in these tables will be lost.
There is no backwards compatibility - the system either works without these
tables or fails fast.

Uses RocksDB column families which need to be dropped.
</context>

<input_context_files>
  <file purpose="tables_to_drop">docs2/impplan/technical/TECH-PHASE0-NORTHSTAR-REMOVAL.md#database_migrations</file>
  <file purpose="db_schema">crates/context-graph-core/src/storage/</file>
</input_context_files>

<prerequisites>
  <check>TASK-P0-003 completed (constitution updated)</check>
  <check>Database backup created if data preservation needed</check>
  <check>RocksDB storage module location identified</check>
</prerequisites>

<scope>
  <in_scope>
    - Drop/remove north_star_goals column family
    - Drop/remove drift_history column family
    - Drop/remove identity_continuity column family
    - Drop/remove pruning_candidates column family
    - Drop/remove sub_goals column family
    - Remove migration files if they exist for creating these
    - Update schema initialization code
  </in_scope>
  <out_of_scope>
    - Creating new tables (later phases)
    - Migrating data to new schema
    - Backup creation (user responsibility)
  </out_of_scope>
</scope>

<definition_of_done>
  <tables_dropped>
    <table name="north_star_goals">Primary North Star goals storage</table>
    <table name="drift_history">Drift detection history</table>
    <table name="identity_continuity">Identity continuity snapshots</table>
    <table name="pruning_candidates">Pruning candidate tracking</table>
    <table name="sub_goals">Sub-goal hierarchy</table>
    <table name="ego_state">Self-ego node state</table>
  </tables_dropped>

  <constraints>
    - No data migration - pure deletion
    - Column family removal in RocksDB
    - Update schema initialization to not create these
    - Migration must be idempotent (safe to run multiple times)
  </constraints>

  <verification>
    - Database opens without errors
    - Querying dropped column families returns appropriate error
    - Storage tests pass (excluding North Star tests)
  </verification>
</definition_of_done>

<pseudo_code>
Migration sequence:
1. Locate RocksDB column family definitions
   - Likely in crates/context-graph-core/src/storage/schema.rs or similar
2. Remove column family declarations for:
   - north_star_goals
   - drift_history
   - identity_continuity
   - pruning_candidates
   - sub_goals
   - ego_state
3. Update database initialization code:
   - Remove creation of these column families
   - Remove any accessor methods for these tables
4. If existing migration system:
   - Create migration to drop column families
5. If no migration system (direct schema):
   - Simply remove from schema definition
6. Test database opens correctly
</pseudo_code>

<files_to_modify>
  <file path="crates/context-graph-core/src/storage/schema.rs">Remove column family definitions</file>
  <file path="crates/context-graph-core/src/storage/mod.rs">Remove table accessor methods</file>
  <file path="crates/context-graph-core/src/storage/migrations/">Add drop migration if system exists</file>
</files_to_modify>

<files_to_delete>
  <file path="crates/context-graph-core/src/storage/north_star.rs">If exists as standalone</file>
  <file path="crates/context-graph-core/src/storage/drift.rs">If exists as standalone</file>
</files_to_delete>

<validation_criteria>
  <criterion>Database schema no longer includes North Star tables</criterion>
  <criterion>Database opens without errors</criterion>
  <criterion>No column family references to removed tables</criterion>
  <criterion>Storage module compiles without errors</criterion>
</validation_criteria>

<test_commands>
  <command description="Check storage compiles">cargo check --package context-graph-core 2>&1 | grep -E "(error|warning)" | head -20</command>
  <command description="Verify schema changes">grep -r "north_star" crates/context-graph-core/src/storage/ || echo "Clean"</command>
  <command description="Run storage tests">cargo test --package context-graph-core storage:: 2>&1 | tail -20</command>
</test_commands>

<notes>
  <note category="rocksdb">
    RocksDB column families are defined at database open time.
    Removing them from the definition list is sufficient - RocksDB will
    ignore existing data in undefined column families.
  </note>
  <note category="data_loss">
    All data in these column families will become inaccessible.
    This is intentional per the "no backwards compatibility" requirement.
  </note>
</notes>
</task_spec>
```

## Execution Checklist

- [ ] Locate RocksDB schema/column family definitions
- [ ] Remove north_star_goals column family
- [ ] Remove drift_history column family
- [ ] Remove identity_continuity column family
- [ ] Remove pruning_candidates column family
- [ ] Remove sub_goals column family
- [ ] Remove ego_state column family
- [ ] Update database initialization code
- [ ] Remove standalone storage files if present
- [ ] Verify compilation
- [ ] Proceed to TASK-P0-005
