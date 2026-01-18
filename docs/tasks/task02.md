# Task 02: Create Skills Directory Structure and Placeholder SKILL.md Files

## Metadata
- **Task ID**: TASK-GAP-003
- **Phase**: 1 (Foundation)
- **Priority**: Critical (Skills are 0% implemented, blocking user access to context-graph functionality)
- **Complexity**: Low (Directory creation and file scaffolding only)
- **Dependencies**: None (This is a standalone task)
- **Branch**: multistar
- **Status**: COMPLETED

---

## CRITICAL CONTEXT FOR IMPLEMENTING AGENT

**READ THIS FIRST**: The skills directory exists but is EMPTY. No skills have been created. You are NOT updating existing files - you are creating them from scratch.

### Current State (Verified 2026-01-18)

| Component | Expected Path | Actual State |
|-----------|--------------|--------------|
| Skills directory | `.claude/skills/` | EXISTS but EMPTY |
| topic-explorer | `.claude/skills/topic-explorer/SKILL.md` | DOES NOT EXIST |
| memory-inject | `.claude/skills/memory-inject/SKILL.md` | DOES NOT EXIST |
| semantic-search | `.claude/skills/semantic-search/SKILL.md` | DOES NOT EXIST |
| dream-consolidation | `.claude/skills/dream-consolidation/SKILL.md` | DOES NOT EXIST |
| curation | `.claude/skills/curation/SKILL.md` | DOES NOT EXIST |

### Verification Command
```bash
ls -la /home/cabdru/contextgraph/.claude/skills/
# Expected output: Empty directory (only . and ..)
```

---

## Objective

Create the 5 skill directories and placeholder SKILL.md files required by PRD v6 Section 9.3. Each SKILL.md must contain valid YAML frontmatter so Claude Code can discover and load the skill.

**This task creates SCAFFOLDING ONLY** - full skill implementations are in tasks 09-13.

---

## Constitutional Rules (MUST FOLLOW)

From `/home/cabdru/contextgraph/docs2/constitution.yaml`:
- **ARCH-07**: Native Claude Code hooks (.claude/settings.json) control memory lifecycle
- **AP-50**: NO internal/built-in hooks - NATIVE hooks via .claude/settings.json ONLY
- **AP-53**: Hook logic MUST be in shell scripts calling context-graph-cli

From `/home/cabdru/contextgraph/docs2/claudeskills.md`:
- Skills location: `.claude/skills/*/SKILL.md`
- Frontmatter: MUST start with `---`, use spaces (not tabs), avoid multiline descriptions
- Required fields: `name`, `description`
- Optional fields: `allowed-tools`, `model`, `version`, `user-invocable`

---

## Files to Create

### Directory Structure

```
/home/cabdru/contextgraph/.claude/skills/
├── topic-explorer/
│   └── SKILL.md
├── memory-inject/
│   └── SKILL.md
├── semantic-search/
│   └── SKILL.md
├── dream-consolidation/
│   └── SKILL.md
└── curation/
    └── SKILL.md
```

---

## Implementation Steps

### Step 1: Create skill directories

```bash
mkdir -p /home/cabdru/contextgraph/.claude/skills/topic-explorer
mkdir -p /home/cabdru/contextgraph/.claude/skills/memory-inject
mkdir -p /home/cabdru/contextgraph/.claude/skills/semantic-search
mkdir -p /home/cabdru/contextgraph/.claude/skills/dream-consolidation
mkdir -p /home/cabdru/contextgraph/.claude/skills/curation
```

### Step 2: Create SKILL.md files

Each file must have valid YAML frontmatter. Placeholder body indicates unimplemented status.

#### File 1: `.claude/skills/topic-explorer/SKILL.md`

```markdown
---
name: topic-explorer
description: Explore emergent topic portfolio, stability metrics, and weighted agreement scores. Use for topic queries, stability checks, divergence analysis. Keywords: topics, portfolio, stability, churn, weighted agreement, divergence.
allowed-tools: Read,Glob,Grep
model: sonnet
version: 0.1.0
user-invocable: true
---
# Topic Explorer

**STATUS: PLACEHOLDER - Full implementation in TASK-GAP-009**

## Overview

Provides access to the emergent topic system with weighted multi-space clustering.

## Capabilities (Planned)

- Query current topic portfolio
- Check topic stability metrics (entropy, churn)
- Analyze weighted agreement scores across 13 embedding spaces
- Detect divergence from recent activity

## MCP Tools Used

- `get_memetic_status` - Current UTL metrics and topic state
- `search_graph` - Find memories by topic similarity

## Not Yet Implemented

This is a placeholder. Run `/topic-explorer` to see this message.
```

#### File 2: `.claude/skills/memory-inject/SKILL.md`

```markdown
---
name: memory-inject
description: Retrieve and inject contextual memories for current task. Automatically distills content to fit token budget. Use for starting tasks, needing background, or restoring context. Keywords: memory, context, inject, retrieve, recall, background.
allowed-tools: Read,Glob
model: haiku
version: 0.1.0
user-invocable: true
---
# Memory Inject

**STATUS: PLACEHOLDER - Full implementation in TASK-GAP-010**

## Overview

Retrieves relevant memories from the knowledge graph and injects them into the current context.

## Capabilities (Planned)

- Search for memories semantically similar to current task
- Rank by relevance score across 13 embedding spaces
- Apply temporal enrichment badges (same-session, same-day)
- Distill to fit token budget (~1200 tokens max)

## MCP Tools Used

- `search_graph` - Multi-space semantic search
- `inject_context` - Store new context with rationale

## Not Yet Implemented

This is a placeholder. Run `/memory-inject` to see this message.
```

#### File 3: `.claude/skills/semantic-search/SKILL.md`

```markdown
---
name: semantic-search
description: Search the knowledge graph using multi-space retrieval. Supports semantic, causal, code, and entity search modes. Keywords: search, find, query, lookup, semantic, causal, code.
allowed-tools: Read,Glob
model: haiku
version: 0.1.0
user-invocable: true
---
# Semantic Search

**STATUS: PLACEHOLDER - Full implementation in TASK-GAP-011**

## Overview

Provides multi-space search across the 13-embedding knowledge graph.

## Capabilities (Planned)

- Semantic search (E1 - general meaning)
- Causal search (E5 - why/because relationships)
- Code search (E7 - technical/code content)
- Entity search (E11 - named entity relationships)
- Combined multi-space RRF ranking

## MCP Tools Used

- `search_graph` - Primary search interface

## Embedding Spaces

| Space | Purpose | Weight |
|-------|---------|--------|
| E1 | Semantic meaning | 1.0 |
| E5 | Causal relationships | 0.9 |
| E7 | Code/technical | 0.85 |
| E10 | Multimodal intent | 0.8 |

## Not Yet Implemented

This is a placeholder. Run `/semantic-search` to see this message.
```

#### File 4: `.claude/skills/dream-consolidation/SKILL.md`

```markdown
---
name: dream-consolidation
description: Trigger memory consolidation via dream phases. NREM replays high-importance patterns. REM discovers blind spots via hyperbolic random walk. Use when entropy high or churn high. Keywords: dream, consolidate, nrem, rem, blind spots, entropy, churn.
allowed-tools: Read,Glob,Bash
model: sonnet
version: 0.1.0
user-invocable: true
---
# Dream Consolidation

**STATUS: PLACEHOLDER - Full implementation in TASK-GAP-012**

## Overview

Triggers memory consolidation through simulated dream phases.

## Dream Phases

### NREM Phase (3 min)
- Purpose: Hebbian learning replay
- Formula: `Delta_w_ij = eta x phi_i x phi_j`
- Strengthens high-importance memory connections

### REM Phase (2 min)
- Purpose: Blind spot discovery
- Model: Poincare ball hyperbolic random walk
- Discovers unexpected semantic connections

## Trigger Conditions

| Metric | Threshold |
|--------|-----------|
| Entropy | > 0.7 for 5+ min |
| Churn | > 0.5 AND entropy > 0.7 |
| GPU Usage | < 80% (constraint) |
| Activity | < 0.15 (idle required) |

## MCP Tools Used

- `get_memetic_status` - Check entropy/churn metrics
- `trigger_consolidation` - Execute consolidation

## Not Yet Implemented

This is a placeholder. Run `/dream-consolidation` to see this message.
```

#### File 5: `.claude/skills/curation/SKILL.md`

```markdown
---
name: curation
description: Curate the knowledge graph by merging, annotating, or forgetting concepts. Process curation tasks from get_memetic_status. Keywords: curate, merge, forget, annotate, prune, duplicate.
allowed-tools: Read,Glob,Bash
model: sonnet
version: 0.1.0
user-invocable: true
---
# Curation

**STATUS: PLACEHOLDER - Full implementation in TASK-GAP-013**

## Overview

Provides tools for curating the knowledge graph through merging, forgetting, and annotating concepts.

## Capabilities (Planned)

- Merge related concepts with rationale
- Identify and remove duplicates
- Prune low-importance memories
- Annotate concepts with metadata

## MCP Tools Used

- `get_memetic_status` - Get curation suggestions
- `merge_concepts` - Merge 2-10 concepts with rationale
- `trigger_consolidation` - Run consolidation after curation

## Merge Strategies

| Strategy | Description |
|----------|-------------|
| `union` | Combine all properties from source concepts |
| `intersection` | Keep only common properties |
| `weighted_average` | Weight by importance scores |

## Not Yet Implemented

This is a placeholder. Run `/curation` to see this message.
```

---

## Definition of Done

### Directory Structure Verification

- [x] Directory `/home/cabdru/contextgraph/.claude/skills/topic-explorer/` exists
- [x] Directory `/home/cabdru/contextgraph/.claude/skills/memory-inject/` exists
- [x] Directory `/home/cabdru/contextgraph/.claude/skills/semantic-search/` exists
- [x] Directory `/home/cabdru/contextgraph/.claude/skills/dream-consolidation/` exists
- [x] Directory `/home/cabdru/contextgraph/.claude/skills/curation/` exists

### SKILL.md File Verification

- [x] File `.claude/skills/topic-explorer/SKILL.md` exists and has valid frontmatter
- [x] File `.claude/skills/memory-inject/SKILL.md` exists and has valid frontmatter
- [x] File `.claude/skills/semantic-search/SKILL.md` exists and has valid frontmatter
- [x] File `.claude/skills/dream-consolidation/SKILL.md` exists and has valid frontmatter
- [x] File `.claude/skills/curation/SKILL.md` exists and has valid frontmatter

### Frontmatter Validation

- [x] All SKILL.md files start with `---` (no blank lines before)
- [x] All SKILL.md files have `name` field matching directory name
- [x] All SKILL.md files have `description` field (single line, < 1024 chars)
- [x] All SKILL.md files have `user-invocable: true`

---

## Full State Verification (MANDATORY)

After completing implementation, you MUST perform Full State Verification.

### 1. Define Source of Truth

| Check | Source of Truth | Expected Result |
|-------|-----------------|-----------------|
| Directories exist | `ls -d .claude/skills/*/` | 5 directories |
| Files exist | `ls .claude/skills/*/SKILL.md` | 5 files |
| Frontmatter valid | `head -1 .claude/skills/*/SKILL.md` | All show `---` |
| Names correct | `grep -h "^name:" .claude/skills/*/SKILL.md` | 5 unique names |
| User invocable | `grep -h "user-invocable:" .claude/skills/*/SKILL.md` | All show `true` |

### 2. Execute & Inspect

Run these commands and verify output:

```bash
cd /home/cabdru/contextgraph

# Source of Truth 1: Directory count
echo "=== Directory Count ==="
ls -d .claude/skills/*/ 2>/dev/null | wc -l
# EXPECTED: 5

# Source of Truth 2: File existence
echo "=== SKILL.md Files ==="
ls .claude/skills/*/SKILL.md 2>/dev/null
# EXPECTED: 5 files listed

# Source of Truth 3: Frontmatter validation
echo "=== Frontmatter First Line ==="
for f in .claude/skills/*/SKILL.md; do
    echo -n "$f: "
    head -1 "$f"
done
# EXPECTED: All show "---"

# Source of Truth 4: Name extraction
echo "=== Skill Names ==="
grep -h "^name:" .claude/skills/*/SKILL.md
# EXPECTED: 5 unique names matching directory names

# Source of Truth 5: User invocable status
echo "=== User Invocable ==="
grep -h "user-invocable:" .claude/skills/*/SKILL.md
# EXPECTED: All show "user-invocable: true"
```

### 3. Boundary & Edge Case Audit

Test these 3 edge cases:

**Edge Case 1: Empty description check**
```bash
# Verify no empty descriptions
for f in .claude/skills/*/SKILL.md; do
    desc=$(grep "^description:" "$f" | cut -d: -f2-)
    if [ -z "$desc" ]; then
        echo "FAIL: Empty description in $f"
        exit 1
    fi
    echo "PASS: $f has description"
done
```

**Edge Case 2: YAML syntax validation**
```bash
# Check for tabs (should be spaces only)
for f in .claude/skills/*/SKILL.md; do
    if grep -P "^\t" "$f" >/dev/null 2>&1; then
        echo "FAIL: Tabs found in $f (must use spaces)"
        exit 1
    fi
    echo "PASS: $f uses spaces"
done
```

**Edge Case 3: Duplicate names check**
```bash
# Verify all names are unique
names=$(grep -h "^name:" .claude/skills/*/SKILL.md | sort)
unique_names=$(echo "$names" | uniq)
if [ "$names" != "$unique_names" ]; then
    echo "FAIL: Duplicate skill names found"
    exit 1
fi
echo "PASS: All skill names unique"
```

### 4. Evidence of Success

Provide a verification log in this format:

```
=== TASK-02 VERIFICATION LOG ===
Timestamp: [ISO 8601 timestamp]

1. Directory Count:
   $ ls -d .claude/skills/*/ | wc -l
   5  # MUST be 5

2. SKILL.md Files:
   $ ls .claude/skills/*/SKILL.md
   .claude/skills/curation/SKILL.md
   .claude/skills/dream-consolidation/SKILL.md
   .claude/skills/memory-inject/SKILL.md
   .claude/skills/semantic-search/SKILL.md
   .claude/skills/topic-explorer/SKILL.md  # MUST list all 5

3. Frontmatter Valid:
   $ head -1 .claude/skills/*/SKILL.md
   [All show "---"]  # MUST start with ---

4. Names Match Directories:
   topic-explorer -> topic-explorer ✓
   memory-inject -> memory-inject ✓
   semantic-search -> semantic-search ✓
   dream-consolidation -> dream-consolidation ✓
   curation -> curation ✓

5. User Invocable:
   All skills have user-invocable: true ✓

=== VERIFICATION COMPLETE ===
```

---

## CRITICAL RULES

1. **NO BACKWARDS COMPATIBILITY** - These are new files, nothing to preserve
2. **FAIL FAST** - If frontmatter is invalid, Claude Code will silently ignore the skill
3. **NO WORKAROUNDS** - Use exact frontmatter format from specification
4. **YAML SYNTAX** - Spaces only (NO TABS), no blank lines before `---`
5. **SINGLE-LINE DESCRIPTIONS** - Multiline descriptions cause silent failures

---

## Verification Commands (Copy-Paste Ready)

```bash
cd /home/cabdru/contextgraph

# Full verification script
echo "=== TASK-02 VERIFICATION ==="
echo ""
echo "1. Directory Count:"
ls -d .claude/skills/*/ 2>/dev/null | wc -l

echo ""
echo "2. SKILL.md Files:"
ls .claude/skills/*/SKILL.md 2>/dev/null || echo "FAIL: No SKILL.md files found"

echo ""
echo "3. Frontmatter First Lines:"
for f in .claude/skills/*/SKILL.md; do
    echo -n "$(basename $(dirname $f)): "
    head -1 "$f"
done

echo ""
echo "4. Skill Names:"
grep -h "^name:" .claude/skills/*/SKILL.md

echo ""
echo "5. User Invocable Status:"
grep -h "user-invocable:" .claude/skills/*/SKILL.md

echo ""
echo "6. Description Length Check:"
for f in .claude/skills/*/SKILL.md; do
    desc=$(grep "^description:" "$f" | cut -d: -f2-)
    len=${#desc}
    echo "$(basename $(dirname $f)): $len chars"
    if [ $len -gt 1024 ]; then
        echo "  WARNING: Exceeds 1024 char limit"
    fi
done

echo ""
echo "=== VERIFICATION COMPLETE ==="
```

---

## Related Tasks

| Task | Description | Dependency |
|------|-------------|------------|
| task01.md | Fix MCP test compilation | COMPLETED |
| **task02.md** | **Create skills directory structure** | **THIS TASK** |
| task09.md | Implement topic-explorer skill | Depends on task02 |
| task10.md | Implement memory-inject skill | Depends on task02 |
| task11.md | Implement semantic-search skill | Depends on task02 |
| task12.md | Implement dream-consolidation skill | Depends on task02 |
| task13.md | Implement curation skill | Depends on task02 |

---

## Reference Files

Before implementing, read these files:
- `/home/cabdru/contextgraph/docs2/claudeskills.md` - Claude Code skills specification
- `/home/cabdru/contextgraph/docs2/constitution.yaml` - PRD v6 constitution (Section 9.3)
- `/home/cabdru/contextgraph/.claude/settings.json` - Current hook configuration (for context)

---

## Existing Infrastructure (For Context)

The following already exists and works:
- `.claude/settings.json` - Hook configuration (5 hooks active)
- `.claude/hooks/session_start.sh` - Session start hook (103 lines)
- `.claude/hooks/pre_tool_use.sh` - Pre tool use hook (66 lines)
- `.claude/hooks/post_tool_use.sh` - Post tool use hook (62 lines)
- `.claude/hooks/user_prompt_submit.sh` - User prompt hook (85 lines)
- `.claude/hooks/session_end.sh` - Session end hook (61 lines)
- `context-graph-cli` - CLI binary with hooks/memory/session commands

The skills will eventually invoke the CLI and MCP tools, but for this task you are ONLY creating the directory structure and placeholder SKILL.md files.
