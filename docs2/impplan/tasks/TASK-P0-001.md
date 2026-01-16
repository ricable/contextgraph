# TASK-P0-001: Delete North Star Source Files

```xml
<task_spec id="TASK-P0-001" version="1.0">
<metadata>
  <title>Delete North Star Source Files</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>1</sequence>
  <phase>0</phase>
  <implements>
    <requirement_ref>REQ-P0-01</requirement_ref>
    <requirement_ref>REQ-P0-02</requirement_ref>
  </implements>
  <depends_on><!-- None - first task --></depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<context>
First task in the North Star removal phase. Deletes all source files related to North Star,
SelfEgoNode, IdentityContinuity, DriftDetector, and DriftCorrector functionality.
This is pure file deletion - no code modification required. Subsequent tasks will handle
references, handlers, and migrations.

CRITICAL: This removes ~13,000 lines of code. No backwards compatibility maintained.
System must function after removal or fail fast.
</context>

<input_context_files>
  <file purpose="files_to_delete">docs2/impplan/technical/TECH-PHASE0-NORTHSTAR-REMOVAL.md#files_to_delete</file>
  <file purpose="verification">docs2/impplan/functional/SPEC-PHASE0-NORTHSTAR-REMOVAL.md</file>
</input_context_files>

<prerequisites>
  <check>Git repository clean (no uncommitted changes)</check>
  <check>Backup branch created if desired</check>
</prerequisites>

<scope>
  <in_scope>
    - Delete src/north_star/ directory and all contents
    - Delete src/drift/ directory (DriftDetector, DriftCorrector)
    - Delete src/ego/identity_continuity.rs if exists
    - Delete src/ego/self_ego_node.rs if exists
    - Delete tests/north_star/ directory if exists
    - Delete tests/drift/ directory if exists
  </in_scope>
  <out_of_scope>
    - MCP handler removal (TASK-P0-002)
    - Constitution updates (TASK-P0-003)
    - Database migrations (TASK-P0-004)
    - Fixing any compile errors from deleted imports
  </out_of_scope>
</scope>

<definition_of_done>
  <deletions>
    <!-- Files/directories that must be deleted -->
    <deletion path="src/north_star/">Complete directory removal</deletion>
    <deletion path="src/drift/">Complete directory removal if exists</deletion>
    <deletion path="src/ego/identity_continuity.rs">File removal if exists</deletion>
    <deletion path="src/ego/self_ego_node.rs">File removal if exists</deletion>
    <deletion path="tests/north_star/">Test directory removal if exists</deletion>
    <deletion path="tests/drift/">Test directory removal if exists</deletion>
  </deletions>

  <constraints>
    - Delete entire directories, not selective files
    - Use git rm for proper tracking
    - Do NOT attempt to fix compilation errors in this task
    - Create a single commit with descriptive message
  </constraints>

  <verification>
    - ls src/north_star/ returns "No such file or directory"
    - ls src/drift/ returns "No such file or directory"
    - Git status shows deleted files
  </verification>
</definition_of_done>

<pseudo_code>
Deletion sequence:
1. cd /home/cabdru/contextgraph
2. git rm -rf src/north_star/  (if exists)
3. git rm -rf src/drift/       (if exists)
4. git rm -f src/ego/identity_continuity.rs  (if exists)
5. git rm -f src/ego/self_ego_node.rs        (if exists)
6. git rm -rf tests/north_star/  (if exists)
7. git rm -rf tests/drift/       (if exists)
8. Verify deletions with ls commands
9. Do NOT commit yet (will be part of P0 completion)
</pseudo_code>

<files_to_delete>
  <directory path="src/north_star/">North Star core implementation</directory>
  <directory path="src/drift/">Drift detection and correction</directory>
  <file path="src/ego/identity_continuity.rs">Identity continuity tracking</file>
  <file path="src/ego/self_ego_node.rs">Self-ego node implementation</file>
  <directory path="tests/north_star/">North Star tests</directory>
  <directory path="tests/drift/">Drift tests</directory>
</files_to_delete>

<files_to_modify>
  <!-- None in this task - modifications handled in subsequent tasks -->
</files_to_modify>

<validation_criteria>
  <criterion>All North Star directories removed from src/</criterion>
  <criterion>All drift directories removed from src/</criterion>
  <criterion>Related test directories removed</criterion>
  <criterion>Git tracks all deletions</criterion>
  <criterion>No North Star files remain in codebase</criterion>
</validation_criteria>

<test_commands>
  <command description="Verify north_star gone">! test -d src/north_star</command>
  <command description="Verify drift gone">! test -d src/drift</command>
  <command description="Find any remaining north_star refs">grep -r "north_star" src/ || echo "No references found"</command>
</test_commands>

<notes>
  <note category="compilation">
    Compilation WILL fail after this task due to removed imports.
    This is expected. Do not attempt to fix - subsequent tasks handle this.
  </note>
  <note category="git">
    Do not create commit until P0-005 verifies complete removal.
    All P0 tasks should be committed together as atomic removal.
  </note>
</notes>
</task_spec>
```

## Execution Checklist

- [ ] Verify git repository is clean
- [ ] Delete src/north_star/ directory
- [ ] Delete src/drift/ directory
- [ ] Delete identity continuity and self-ego files
- [ ] Delete related test directories
- [ ] Verify all deletions with ls commands
- [ ] Stage deletions with git
- [ ] Proceed to TASK-P0-002
