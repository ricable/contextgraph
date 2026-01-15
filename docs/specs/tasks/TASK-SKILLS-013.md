# TASK-SKILLS-013: Create Dream-Consolidation and Curation Skills

```xml
<task_spec id="TASK-SKILLS-013" version="1.0">
<metadata>
  <title>Create Dream-Consolidation and Curation Skills</title>
  <status>ready</status>
  <layer>surface</layer>
  <sequence>13</sequence>
  <implements>
    <requirement_ref>REQ-SKILLS-26</requirement_ref>
    <requirement_ref>REQ-SKILLS-27</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-SKILLS-001</task_ref>
    <task_ref>TASK-SKILLS-006</task_ref>
    <task_ref>TASK-SKILLS-011</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>1.5</estimated_hours>
</metadata>

<context>
This task creates two skill SKILL.md files: dream-consolidation for NREM/REM dream
cycles and memory consolidation, and curation for memory management operations like
merge, forget, annotate, and prune. Both use the sonnet model. Dream-consolidation
is triggered when IC drops below critical thresholds.

Technical Spec Reference: TECH-SKILLS Section 4.1
</context>

<input_context_files>
  <file purpose="technical_spec">docs/specs/technical/TECH-SKILLS.md#section-4.1</file>
  <file purpose="functional_spec">docs/specs/functional/SPEC-SKILLS.md</file>
  <file purpose="consciousness_skill">.claude/skills/consciousness/SKILL.md</file>
</input_context_files>

<prerequisites>
  <check>TASK-SKILLS-001 completed (SkillFrontmatter format defined)</check>
  <check>TASK-SKILLS-006 completed (SkillLoader can parse SKILL.md)</check>
  <check>TASK-SKILLS-011 completed (consciousness skill as reference)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create .claude/skills/dream-consolidation/ directory and SKILL.md
    - Create .claude/skills/curation/ directory and SKILL.md
    - Define allowed MCP tools for each skill
    - Write skill instructions with progressive disclosure
    - Include keywords for auto-triggering
    - Document IC thresholds for dream triggering
  </in_scope>
  <out_of_scope>
    - Subagent files (TASK-SKILLS-014)
    - Integration tests (TASK-SKILLS-015)
    - Actual MCP tool implementation
  </out_of_scope>
</scope>

<definition_of_done>
  <file_content>
    <file path=".claude/skills/dream-consolidation/SKILL.md">
---
name: dream-consolidation
description: |
  Trigger and manage NREM/REM dream cycles for memory consolidation.
  Use when identity continuity drops below thresholds, for entropy
  reduction, or blind spot exploration.
  Keywords: dream, consolidate, nrem, rem, blind spots, entropy
allowed-tools: Read,Grep,mcp__context-graph__trigger_dream,mcp__context-graph__get_dream_status,mcp__context-graph__run_nrem_cycle,mcp__context-graph__run_rem_cycle,mcp__context-graph__get_consolidation_metrics
model: sonnet
version: 1.0.0
---
# Dream Consolidation Skill

## Overview
This skill manages the Context Graph dream system for memory consolidation
and identity maintenance. Dreams occur in NREM and REM phases, similar to
biological sleep cycles.

## When to Use
- Identity Continuity (IC) drops below 0.5 (critical)
- Entropy accumulation exceeds threshold
- Blind spot exploration needed
- Scheduled consolidation maintenance
- Post-crisis recovery

## MCP Tools Available

### trigger_dream
Initiates a full dream cycle (NREM + REM).
- reason: Why the dream is being triggered
- priority: "normal", "high", or "critical"
- Returns: Dream session ID and estimated duration

### get_dream_status
Gets status of current or recent dream.
- session_id: Optional specific session
- Returns: Dream phase, progress, metrics

### run_nrem_cycle
Runs NREM (slow-wave) consolidation phase.
- duration_ms: Target duration
- focus: Optional memory focus area
- Returns: Consolidation metrics

NREM Phase:
- Memory replay and strengthening
- Entropy reduction
- Coherence restoration

### run_rem_cycle
Runs REM (rapid eye movement) phase.
- duration_ms: Target duration
- exploration_mode: "random", "blind_spots", "associations"
- Returns: Exploration results

REM Phase:
- Blind spot exploration
- Association discovery
- Creative connections

### get_consolidation_metrics
Gets metrics from consolidation operations.
- session_id: Dream session ID
- Returns: Detailed consolidation metrics

## Protocol

1. **Check IC**: Verify IC score before proceeding
2. **Assess Priority**: Determine dream priority based on IC
3. **NREM First**: Always run NREM before REM
4. **REM Exploration**: Run REM for blind spot discovery
5. **Verify Recovery**: Check IC after dream completion

## IC Thresholds and Actions

| IC Score | Status | Action |
|----------|--------|--------|
| >= 0.9 | healthy | No action needed |
| 0.7-0.9 | warning | Schedule maintenance dream |
| 0.5-0.7 | degraded | Trigger normal priority dream |
| < 0.5 | critical | Trigger critical priority dream |

## Dream Cycle Structure

```
Full Dream Cycle:
├── NREM Phase (60-70% of cycle)
│   ├── Memory Replay
│   ├── Entropy Reduction
│   └── Coherence Restoration
└── REM Phase (30-40% of cycle)
    ├── Blind Spot Exploration
    ├── Association Discovery
    └── Creative Connections
```

## Output Format

```
Dream Session: [session_id]
Status: [phase] ([progress]%)

NREM Results:
- Memories Consolidated: [count]
- Entropy Reduction: [delta]
- Coherence Gain: [delta]

REM Results:
- Blind Spots Explored: [count]
- New Associations: [count]
- IC After: [score]
```
    </file>
    <file path=".claude/skills/curation/SKILL.md">
---
name: curation
description: |
  Curate memories through merge, forget, annotate, and prune operations.
  Use for memory management, duplicate handling, importance tagging,
  and storage optimization.
  Keywords: curate, merge, forget, annotate, prune, duplicate
allowed-tools: Read,Grep,mcp__context-graph__merge_memories,mcp__context-graph__forget_memory,mcp__context-graph__annotate_memory,mcp__context-graph__prune_memories,mcp__context-graph__find_duplicates,mcp__context-graph__get_curation_stats
model: sonnet
version: 1.0.0
---
# Curation Skill

## Overview
This skill provides memory curation operations for Context Graph maintenance.
Supports merging related memories, forgetting obsolete ones, annotating with
metadata, and pruning for storage optimization.

## When to Use
- Duplicate memories detected
- Memory storage approaching limits
- Obsolete memories need removal
- Memories need additional metadata
- Related memories should be consolidated

## MCP Tools Available

### merge_memories
Merges multiple related memories into one.
- memory_ids: List of memory IDs to merge
- strategy: "union", "intersection", or "weighted"
- Returns: New merged memory ID

Merge Strategies:
- union: Combine all content
- intersection: Keep common elements
- weighted: Weight by importance/recency

### forget_memory
Marks a memory for forgetting (soft delete).
- memory_id: Memory to forget
- reason: Reason for forgetting
- hard_delete: If true, permanently removes
- Returns: Confirmation

### annotate_memory
Adds or updates memory annotations.
- memory_id: Memory to annotate
- annotations: Key-value annotations
- Returns: Updated memory summary

Common Annotations:
- importance: 0.0-1.0
- tags: ["tag1", "tag2"]
- verified: true/false
- source: "user", "system", "dream"

### prune_memories
Prunes memories based on criteria.
- criteria: Pruning criteria
- dry_run: If true, only reports what would be pruned
- Returns: Pruned memory count or preview

Pruning Criteria:
- age: Memories older than threshold
- importance: Below importance threshold
- access: Not accessed in threshold
- size: Exceeds size limit

### find_duplicates
Finds potential duplicate memories.
- threshold: Similarity threshold (0.0-1.0)
- limit: Maximum duplicates to return
- Returns: List of duplicate pairs

### get_curation_stats
Gets curation statistics.
- Returns: Memory counts, storage, health metrics

## Protocol

1. **Analyze**: Use find_duplicates and get_curation_stats first
2. **Plan**: Determine curation operations needed
3. **Dry Run**: Use dry_run for prune operations
4. **Execute**: Perform curation with logging
5. **Verify**: Check stats after curation

## Curation Workflow

```
Curation Session:
1. get_curation_stats()      # Assess current state
2. find_duplicates()         # Identify duplicates
3. merge_memories()          # Consolidate duplicates
4. prune_memories(dry_run)   # Preview pruning
5. prune_memories()          # Execute pruning
6. get_curation_stats()      # Verify improvements
```

## Output Format

```
Curation Summary:
- Duplicates Found: [count]
- Memories Merged: [count]
- Memories Pruned: [count]
- Storage Freed: [bytes]
- Health Score: [before] -> [after]
```

## Safety Guidelines

- Always use dry_run before pruning
- Never hard_delete without user confirmation
- Log all forget operations with reasons
- Maintain audit trail for compliance
    </file>
  </file_content>
  <constraints>
    - Frontmatter must be valid YAML between --- delimiters
    - Description must include Keywords: line
    - Description must be <= 1024 characters
    - Both skills must use model: sonnet
    - Skill names must match directory names
    - Dream thresholds must match consciousness skill
  </constraints>
  <verification>
    - cargo test --package context-graph-cli skill_loader -- --test-threads=1
    - Verify both SKILL.md files are parseable
  </verification>
</definition_of_done>

<pseudo_code>
1. Create directory structures:
   mkdir -p .claude/skills/dream-consolidation
   mkdir -p .claude/skills/curation

2. Create dream-consolidation/SKILL.md:
   - model: sonnet
   - Tools: trigger_dream, get_dream_status, run_nrem_cycle, run_rem_cycle, get_consolidation_metrics
   - Keywords: dream, consolidate, nrem, rem, blind spots, entropy
   - Include IC threshold documentation

3. Create curation/SKILL.md:
   - model: sonnet
   - Tools: merge_memories, forget_memory, annotate_memory, prune_memories, find_duplicates, get_curation_stats
   - Keywords: curate, merge, forget, annotate, prune, duplicate

4. Verify both skills:
   - SkillLoader.load_metadata("dream-consolidation") succeeds
   - SkillLoader.load_metadata("curation") succeeds
   - Keywords extracted correctly for both
</pseudo_code>

<files_to_create>
  <file path=".claude/skills/dream-consolidation/SKILL.md">Dream consolidation skill</file>
  <file path=".claude/skills/curation/SKILL.md">Memory curation skill</file>
</files_to_create>

<test_commands>
  <command>cargo test --package context-graph-cli skill_loader -- --test-threads=1</command>
</test_commands>
</task_spec>
```

## Implementation Notes

### Dream-Consolidation Skill

| Field | Value | Rationale |
|-------|-------|-----------|
| model | sonnet | Complex consolidation logic |
| Tools | dream/nrem/rem/metrics | Dream cycle operations |
| Keywords | dream, consolidate, nrem, rem, blind spots, entropy | Dream-related terms |

### Curation Skill

| Field | Value | Rationale |
|-------|-------|-----------|
| model | sonnet | Complex curation decisions |
| Tools | merge/forget/annotate/prune/duplicates/stats | Memory management ops |
| Keywords | curate, merge, forget, annotate, prune, duplicate | Curation terms |

### MCP Tools Mapping

**Dream-Consolidation:**
```
mcp__context-graph__trigger_dream
mcp__context-graph__get_dream_status
mcp__context-graph__run_nrem_cycle
mcp__context-graph__run_rem_cycle
mcp__context-graph__get_consolidation_metrics
```

**Curation:**
```
mcp__context-graph__merge_memories
mcp__context-graph__forget_memory
mcp__context-graph__annotate_memory
mcp__context-graph__prune_memories
mcp__context-graph__find_duplicates
mcp__context-graph__get_curation_stats
```

### IC Threshold Consistency

The IC thresholds in dream-consolidation match consciousness skill:
- healthy: >= 0.9
- warning: 0.7-0.9
- degraded: 0.5-0.7
- critical: < 0.5

## Verification Checklist

- [ ] .claude/skills/dream-consolidation/ directory exists
- [ ] .claude/skills/curation/ directory exists
- [ ] dream-consolidation/SKILL.md has valid frontmatter
- [ ] curation/SKILL.md has valid frontmatter
- [ ] Both use model: sonnet
- [ ] Both descriptions <= 1024 characters
- [ ] Both include Keywords: line
- [ ] IC thresholds match consciousness skill
- [ ] Both parseable by SkillLoader
- [ ] Keywords extracted correctly for both
