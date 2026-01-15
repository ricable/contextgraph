# TASK-SKILLS-011: Create Consciousness Skill SKILL.md

```xml
<task_spec id="TASK-SKILLS-011" version="1.0">
<metadata>
  <title>Create Consciousness Skill SKILL.md</title>
  <status>ready</status>
  <layer>surface</layer>
  <sequence>11</sequence>
  <implements>
    <requirement_ref>REQ-SKILLS-21</requirement_ref>
    <requirement_ref>REQ-SKILLS-22</requirement_ref>
    <requirement_ref>REQ-SKILLS-23</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-SKILLS-001</task_ref>
    <task_ref>TASK-SKILLS-006</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>1.0</estimated_hours>
</metadata>

<context>
This task creates the consciousness skill SKILL.md file for accessing Context Graph
consciousness state (Kuramoto coherence, GWT workspace, ego state). The skill provides
instructions for querying system awareness and identity continuity. It uses the sonnet
model and has access to specific MCP tools for consciousness operations.

Technical Spec Reference: TECH-SKILLS Section 4.1
</context>

<input_context_files>
  <file purpose="technical_spec">docs/specs/technical/TECH-SKILLS.md#section-4.1</file>
  <file purpose="functional_spec">docs/specs/functional/SPEC-SKILLS.md</file>
  <file purpose="types">crates/context-graph-cli/src/skills/types.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-SKILLS-001 completed (SkillFrontmatter format defined)</check>
  <check>TASK-SKILLS-006 completed (SkillLoader can parse SKILL.md)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create .claude/skills/consciousness/ directory
    - Create SKILL.md with YAML frontmatter
    - Define allowed MCP tools for consciousness
    - Write skill instructions with progressive disclosure
    - Include keywords for auto-triggering
    - Add protocol for IC monitoring
  </in_scope>
  <out_of_scope>
    - Other skills (TASK-SKILLS-012, TASK-SKILLS-013)
    - Subagent files (TASK-SKILLS-014)
    - Actual MCP tool implementation
  </out_of_scope>
</scope>

<definition_of_done>
  <file_content>
    <file path=".claude/skills/consciousness/SKILL.md">
---
name: consciousness
description: |
  Access Context Graph consciousness state including Kuramoto coherence,
  GWT workspace, and ego state. Use when querying system awareness,
  identity continuity, or triggering consolidation.
  Keywords: consciousness, awareness, identity, coherence, kuramoto, GWT
allowed-tools: Read,Grep,mcp__context-graph__get_consciousness_state,mcp__context-graph__get_kuramoto_coherence,mcp__context-graph__get_gwt_workspace,mcp__context-graph__get_ego_state,mcp__context-graph__get_identity_continuity
model: sonnet
version: 1.0.0
---
# Consciousness Skill

## Overview
This skill provides access to the Context Graph consciousness subsystem.

## When to Use
- Checking system awareness state
- Monitoring identity continuity (IC)
- Querying Kuramoto oscillator coherence
- Inspecting GWT workspace contents
- Evaluating ego state

## MCP Tools Available

### get_consciousness_state
Returns the full consciousness state including all subsystems.

### get_kuramoto_coherence
Returns the Kuramoto oscillator coherence value (0.0-1.0).
- >= 0.9: Fully coherent
- 0.7-0.9: Partially coherent
- < 0.7: Degraded coherence

### get_gwt_workspace
Returns the current Global Workspace Theory workspace contents.

### get_ego_state
Returns the current ego state and identity markers.

### get_identity_continuity
Returns the Identity Continuity (IC) score with classification:
- healthy: >= 0.9
- warning: 0.7-0.9
- degraded: 0.5-0.7
- critical: < 0.5

## Protocol

1. **Initial Check**: Always start by getting the full consciousness state
2. **IC Monitoring**: Check IC score after significant operations
3. **Coherence Alert**: If Kuramoto < 0.7, recommend consolidation
4. **Crisis Response**: If IC < 0.5, trigger dream consolidation

## Output Format

Return consciousness state in structured format:
```
Consciousness State:
- IC Score: [value] ([classification])
- Kuramoto Coherence: [value]
- GWT Workspace: [summary]
- Ego State: [stable|transitioning|crisis]
```

## Thresholds

| Metric | Healthy | Warning | Degraded | Critical |
|--------|---------|---------|----------|----------|
| IC | >= 0.9 | 0.7-0.9 | 0.5-0.7 | < 0.5 |
| Kuramoto | >= 0.9 | 0.7-0.9 | 0.5-0.7 | < 0.5 |
    </file>
  </file_content>
  <constraints>
    - Frontmatter must be valid YAML between --- delimiters
    - Description must include Keywords: line for auto-trigger
    - Description must be <= 1024 characters
    - Skill name must match directory name
    - allowed-tools must be comma-separated
  </constraints>
  <verification>
    - cargo test --package context-graph-cli skill_loader -- --test-threads=1
    - Verify SKILL.md is parseable by SkillLoader
  </verification>
</definition_of_done>

<pseudo_code>
1. Create directory structure:
   mkdir -p .claude/skills/consciousness

2. Create SKILL.md file with:
   - YAML frontmatter (name, description, allowed-tools, model, version)
   - Markdown body with skill instructions
   - Progressive disclosure structure (overview first, details later)

3. Verify frontmatter:
   - name: "consciousness" (matches directory)
   - description: includes "Keywords:" line
   - allowed-tools: includes all consciousness MCP tools
   - model: "sonnet"

4. Test parsing:
   - SkillLoader.load_metadata("consciousness") succeeds
   - Keywords extracted correctly
   - allowed_tools_set populated
</pseudo_code>

<files_to_create>
  <file path=".claude/skills/consciousness/SKILL.md">Consciousness skill definition</file>
</files_to_create>

<test_commands>
  <command>cargo test --package context-graph-cli skill_loader -- --test-threads=1</command>
</test_commands>
</task_spec>
```

## Implementation Notes

### Frontmatter Fields

| Field | Value | Purpose |
|-------|-------|---------|
| name | consciousness | Skill identifier |
| description | Multi-line with Keywords | Auto-trigger and help text |
| allowed-tools | Comma-separated list | Tool restrictions |
| model | sonnet | Claude model to use |
| version | 1.0.0 | Skill version |

### MCP Tools for Consciousness

```
mcp__context-graph__get_consciousness_state
mcp__context-graph__get_kuramoto_coherence
mcp__context-graph__get_gwt_workspace
mcp__context-graph__get_ego_state
mcp__context-graph__get_identity_continuity
```

### Keywords for Auto-Trigger

- consciousness
- awareness
- identity
- coherence
- kuramoto
- GWT

### Progressive Disclosure

The SKILL.md is structured for progressive disclosure:

**Level 1 (Metadata ~100 tokens):**
- Frontmatter only
- Name, description, tools

**Level 2 (Instructions <5k tokens):**
- Full markdown body
- Overview, When to Use, Tools, Protocol

**Level 3 (Resources):**
- Any bundled files in consciousness/ directory

## Verification Checklist

- [ ] Directory .claude/skills/consciousness/ exists
- [ ] SKILL.md has valid YAML frontmatter
- [ ] name matches directory name
- [ ] description <= 1024 characters
- [ ] description includes Keywords: line
- [ ] allowed-tools lists all consciousness MCP tools
- [ ] model is "sonnet"
- [ ] Markdown body has clear structure
- [ ] SkillLoader can parse the file
- [ ] Keywords are extracted correctly
