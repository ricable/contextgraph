# TASK-P0-003: Update Constitution.yaml

```xml
<task_spec id="TASK-P0-003" version="1.0">
<metadata>
  <title>Update Constitution.yaml</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>3</sequence>
  <phase>0</phase>
  <implements>
    <requirement_ref>REQ-P0-04</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P0-002</task_ref>
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<context>
The constitution.yaml file defines system rules, thresholds, and behavior contracts.
It contains sections related to North Star, identity continuity, drift detection,
and pruning that must be removed.

This is a configuration file update - remove obsolete sections while preserving
the overall structure and any sections needed for the new implementation.
</context>

<input_context_files>
  <file purpose="current_constitution">constitution.yaml</file>
  <file purpose="sections_to_remove">docs2/impplan/technical/TECH-PHASE0-NORTHSTAR-REMOVAL.md#constitution_changes</file>
</input_context_files>

<prerequisites>
  <check>TASK-P0-002 completed (MCP handlers removed)</check>
  <check>constitution.yaml file located</check>
</prerequisites>

<scope>
  <in_scope>
    - Remove north_star section entirely
    - Remove identity section (identity_continuity thresholds)
    - Remove drift section (drift detection parameters)
    - Remove pruning section (if North Star specific)
    - Remove sub_goals section
    - Remove autonomous section
    - Keep core system configuration
    - Keep embedding/similarity sections (used by new system)
  </in_scope>
  <out_of_scope>
    - Adding new configuration sections
    - Database changes (TASK-P0-004)
    - Code changes to read new config
  </out_of_scope>
</scope>

<definition_of_done>
  <sections_removed>
    <section path="north_star">Entire North Star configuration</section>
    <section path="identity">Identity continuity thresholds</section>
    <section path="drift">Drift detection parameters</section>
    <section path="pruning">If North Star specific pruning config</section>
    <section path="sub_goals">Sub-goal discovery config</section>
    <section path="autonomous">Autonomous operation config</section>
    <section path="self_ego">Self-ego node configuration</section>
  </sections_removed>

  <sections_preserved>
    <section path="utl">UTL metrics (may be repurposed)</section>
    <section path="embedding">Embedding configurations</section>
    <section path="storage">Storage configurations</section>
    <section path="security">Security rules</section>
  </sections_preserved>

  <constraints>
    - Maintain valid YAML syntax
    - Preserve file structure (comments, indentation style)
    - Do not add new sections in this task
    - Ensure no dangling references to removed sections
  </constraints>

  <verification>
    - YAML parses without errors
    - grep for "north_star" returns nothing
    - grep for "identity_continuity" returns nothing
    - grep for "drift" returns nothing (in config context)
  </verification>
</definition_of_done>

<pseudo_code>
Constitution update sequence:
1. Read constitution.yaml
2. Identify all North Star related sections:
   - Look for: north_star, identity, drift, pruning, sub_goals, autonomous, self_ego
3. Remove each identified section entirely
4. Verify remaining YAML is valid
5. Check for any cross-references to removed sections
6. Write updated constitution.yaml
7. Validate with YAML parser
</pseudo_code>

<files_to_modify>
  <file path="constitution.yaml">Remove North Star related sections</file>
</files_to_modify>

<validation_criteria>
  <criterion>constitution.yaml is valid YAML</criterion>
  <criterion>No north_star section exists</criterion>
  <criterion>No identity section exists</criterion>
  <criterion>No drift section exists</criterion>
  <criterion>No self_ego section exists</criterion>
  <criterion>No sub_goals section exists</criterion>
  <criterion>No autonomous section exists</criterion>
  <criterion>Core system configuration preserved</criterion>
</validation_criteria>

<test_commands>
  <command description="Validate YAML syntax">python3 -c "import yaml; yaml.safe_load(open('constitution.yaml'))"</command>
  <command description="Check no north_star">! grep -q "north_star:" constitution.yaml</command>
  <command description="Check no identity section">! grep -q "^identity:" constitution.yaml</command>
  <command description="Check no drift section">! grep -q "^drift:" constitution.yaml</command>
</test_commands>

<notes>
  <note category="yaml_structure">
    YAML is whitespace-sensitive. When removing sections, ensure proper
    indentation is maintained for surrounding content.
  </note>
  <note category="comments">
    Preserve any useful comments that document the file structure.
    Remove comments that reference deleted functionality.
  </note>
</notes>
</task_spec>
```

## Execution Checklist

- [ ] Read current constitution.yaml
- [ ] Identify all North Star related sections
- [ ] Remove north_star section
- [ ] Remove identity section
- [ ] Remove drift section
- [ ] Remove self_ego section
- [ ] Remove sub_goals section
- [ ] Remove autonomous section
- [ ] Verify YAML syntax validity
- [ ] Verify no references to removed sections
- [ ] Proceed to TASK-P0-004
