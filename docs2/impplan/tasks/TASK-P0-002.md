# TASK-P0-002: Remove MCP Handlers and Routes

```xml
<task_spec id="TASK-P0-002" version="1.0">
<metadata>
  <title>Remove MCP Handlers and Routes</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>2</sequence>
  <phase>0</phase>
  <implements>
    <requirement_ref>REQ-P0-03</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P0-001</task_ref>
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<context>
After deleting North Star source files, this task removes all MCP tool handlers
and routes that exposed North Star functionality. These handlers will cause
compilation errors since their dependencies no longer exist.

The MCP server exposes tools via JSON-RPC. Each North Star tool has a handler
function that must be removed, plus the tool registration/routing.
</context>

<input_context_files>
  <file purpose="handlers_to_remove">docs2/impplan/technical/TECH-PHASE0-NORTHSTAR-REMOVAL.md#mcp_handlers</file>
  <file purpose="mcp_structure">crates/context-graph-mcp/src/lib.rs</file>
  <file purpose="tool_registry">crates/context-graph-mcp/src/tools/mod.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P0-001 completed (source files deleted)</check>
  <check>MCP crate location identified</check>
</prerequisites>

<scope>
  <in_scope>
    - Remove North Star MCP tool handlers
    - Remove tool registrations from router/dispatcher
    - Remove North Star tool schema definitions
    - Remove imports referencing deleted modules
    - Update mod.rs files to remove deleted modules
  </in_scope>
  <out_of_scope>
    - Constitution.yaml updates (TASK-P0-003)
    - Database schema changes (TASK-P0-004)
    - Adding new tools
  </out_of_scope>
</scope>

<definition_of_done>
  <handlers_removed>
    <handler name="auto_bootstrap_north_star">Remove handler and registration</handler>
    <handler name="get_alignment_drift">Remove handler and registration</handler>
    <handler name="get_drift_history">Remove handler and registration</handler>
    <handler name="trigger_drift_correction">Remove handler and registration</handler>
    <handler name="get_pruning_candidates">Remove handler and registration</handler>
    <handler name="trigger_consolidation">Remove handler and registration</handler>
    <handler name="discover_sub_goals">Remove handler and registration</handler>
    <handler name="get_autonomous_status">Remove handler and registration</handler>
    <handler name="get_ego_state">Remove handler and registration</handler>
    <handler name="get_identity_continuity">Remove handler and registration</handler>
  </handlers_removed>

  <constraints>
    - Remove entire handler functions, not just disable
    - Remove from tool schema/registry
    - Update all affected mod.rs files
    - Ensure no dangling imports remain
  </constraints>

  <verification>
    - grep for removed handler names returns no results
    - cargo check on MCP crate passes (no missing imports)
  </verification>
</definition_of_done>

<pseudo_code>
Handler removal sequence:
1. Locate MCP tools directory (crates/context-graph-mcp/src/tools/)
2. For each North Star handler:
   a. Delete handler file if standalone (e.g., north_star.rs, drift.rs)
   b. Or remove handler functions from combined file
3. Update tools/mod.rs:
   a. Remove module declarations (mod north_star;)
   b. Remove use statements
   c. Remove from tool registry match arms
4. Update main router/dispatcher:
   a. Remove tool name -> handler mappings
5. Remove any schema definitions for removed tools
6. Verify with grep that handler names no longer exist
</pseudo_code>

<files_to_delete>
  <file path="crates/context-graph-mcp/src/tools/north_star.rs">If exists as standalone</file>
  <file path="crates/context-graph-mcp/src/tools/drift.rs">If exists as standalone</file>
  <file path="crates/context-graph-mcp/src/tools/ego.rs">Identity/ego handlers if standalone</file>
</files_to_delete>

<files_to_modify>
  <file path="crates/context-graph-mcp/src/tools/mod.rs">Remove module declarations and re-exports</file>
  <file path="crates/context-graph-mcp/src/lib.rs">Remove tool registrations if present</file>
  <file path="crates/context-graph-mcp/src/router.rs">Remove handler routing if present</file>
</files_to_modify>

<validation_criteria>
  <criterion>No MCP handlers reference north_star module</criterion>
  <criterion>No MCP handlers reference drift module</criterion>
  <criterion>No MCP handlers reference identity_continuity</criterion>
  <criterion>cargo check --package context-graph-mcp succeeds</criterion>
  <criterion>No dangling imports in MCP crate</criterion>
</validation_criteria>

<test_commands>
  <command description="Check no north_star refs in MCP">grep -r "north_star" crates/context-graph-mcp/ || echo "Clean"</command>
  <command description="Check no drift refs in MCP">grep -r "drift" crates/context-graph-mcp/ || echo "Clean"</command>
  <command description="Verify MCP crate compiles">cargo check --package context-graph-mcp 2>&1 | head -50</command>
</test_commands>

<notes>
  <note category="partial_files">
    If handlers are in combined files (not standalone), carefully remove only
    the specific handler functions and their related code, preserving other handlers.
  </note>
  <note category="schema">
    MCP tools have JSON schema definitions. Ensure these are also removed
    so the tool doesn't appear in capability listings.
  </note>
</notes>
</task_spec>
```

## Execution Checklist

- [ ] Locate MCP tools directory structure
- [ ] Identify all North Star handler files/functions
- [ ] Delete standalone handler files
- [ ] Remove handler functions from combined files
- [ ] Update mod.rs to remove module declarations
- [ ] Update router/dispatcher to remove tool mappings
- [ ] Remove schema definitions
- [ ] Verify with grep commands
- [ ] Run cargo check to verify compilation
- [ ] Proceed to TASK-P0-003
