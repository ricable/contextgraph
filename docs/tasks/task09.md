# Task 09: Implement topic-explorer SKILL.md

## Metadata
- **Task ID**: TASK-GAP-009
- **Phase**: 3 (Skills Framework)
- **Priority**: High
- **Complexity**: Low
- **Dependencies**: NONE - All required backend infrastructure is COMPLETE
- **Branch**: multistar

## Current State Assessment (2026-01-18)

### COMPLETED BACKEND INFRASTRUCTURE

| Component | Status | Location | Lines |
|-----------|--------|----------|-------|
| `get_topic_portfolio` handler | DONE | `src/handlers/tools/topic_tools.rs:47-126` | 80 |
| `get_topic_stability` handler | DONE | `src/handlers/tools/topic_tools.rs:144-207` | 64 |
| `detect_topics` handler | DONE | `src/handlers/tools/topic_tools.rs:226-304` | 79 |
| `get_divergence_alerts` handler | DONE | `src/handlers/tools/topic_tools.rs:324-376` | 53 |
| Topic DTOs | DONE | `src/handlers/tools/topic_dtos.rs` | 700+ |
| Tool definitions | DONE | `src/tools/definitions/topic.rs` | 380 |
| Tool constants | DONE | `src/tools/names.rs:22-25` | 4 |
| Dispatch routing | DONE | `src/handlers/tools/dispatch.rs:76-86` | 4 |
| Handler tests | DONE | `src/handlers/tests/topic_tools.rs` | 568 |

### WHAT EXISTS - SKILL FILE PLACEHOLDER

**File**: `/home/cabdru/contextgraph/.claude/skills/topic-explorer/SKILL.md`

Current placeholder (31 lines) contains:
- Correct frontmatter (`model: sonnet`, `user_invocable: true`)
- "STATUS: PLACEHOLDER" header
- **WRONG** MCP tools listed (`get_memetic_status`, `search_graph`)
- No usage instructions
- No output format documentation
- No edge case handling

### WHAT'S NEEDED

Replace the placeholder SKILL.md with full implementation that documents:
1. The **CORRECT** MCP tools: `get_topic_portfolio`, `get_topic_stability`
2. Step-by-step usage instructions
3. Output format examples (brief, standard, verbose)
4. Edge cases (Tier 0, no topics, high churn, dream recommendation)
5. Example user queries and responses

## Objective

Create the topic-explorer skill SKILL.md file with complete implementation instructions so Claude can use the topic system MCP tools effectively.

## Input Context Files (MUST READ)

```bash
# 1. SKILL FORMAT REFERENCE - How skills are structured
/home/cabdru/contextgraph/docs2/claudeskills.md

# 2. CONSTITUTION - Topic system rules and thresholds
/home/cabdru/contextgraph/docs2/constitution.yaml
# Key sections: topic_system, embedder_categories, weighted_agreement

# 3. TOOL DTOs - Request/response formats
/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/tools/topic_dtos.rs
# Contains: GetTopicPortfolioRequest, TopicPortfolioResponse, GetTopicStabilityRequest, TopicStabilityResponse
```

## MCP TOOLS - EXACT SPECIFICATIONS

### Tool 1: `get_topic_portfolio`

**Purpose**: Get all discovered topics with profiles, stability metrics, and tier info.

**Request Parameters**:
```json
{
  "format": "brief" | "standard" | "verbose"
}
```
- `format` (optional, default: "standard"):
  - `brief`: Topic names and confidence only
  - `standard`: Includes contributing spaces and member counts
  - `verbose`: Full topic profiles with all 13 strengths

**Response Structure**:
```json
{
  "topics": [
    {
      "id": "uuid",
      "name": "optional-string",
      "confidence": 0.0-1.0,
      "weighted_agreement": 0.0-8.5,
      "member_count": number,
      "contributing_spaces": ["Semantic", "Causal", "Code"],
      "phase": "Emerging" | "Stable" | "Declining" | "Merging"
    }
  ],
  "stability": {
    "churn_rate": 0.0-1.0,
    "entropy": 0.0-1.0,
    "is_stable": boolean
  },
  "total_topics": number,
  "tier": 0-6
}
```

**Tier Meanings** (per `topic_dtos.rs:267-277`):
| Tier | Memory Count | Features |
|------|--------------|----------|
| 0 | 0 | Storage only, no retrieval |
| 1 | 1-2 | Pairwise similarity |
| 2 | 3-9 | Basic clustering |
| 3 | 10-29 | Multiple clusters, divergence detection |
| 4 | 30-99 | Reliable statistics |
| 5 | 100-499 | Sub-clustering, trend analysis |
| 6 | 500+ | Full personalization |

### Tool 2: `get_topic_stability`

**Purpose**: Get portfolio-level stability metrics including churn, entropy, phase breakdown.

**Request Parameters**:
```json
{
  "hours": 1-168
}
```
- `hours` (optional, default: 6): Lookback period for computing averages

**Response Structure**:
```json
{
  "churn_rate": 0.0-1.0,
  "entropy": 0.0-1.0,
  "phases": {
    "emerging": number,
    "stable": number,
    "declining": number,
    "merging": number
  },
  "dream_recommended": boolean,
  "high_churn_warning": boolean,
  "average_churn": 0.0-1.0
}
```

**Constitution Thresholds** (per `topic_dtos.rs:44-51`):
- Topic threshold: `weighted_agreement >= 2.5` (ARCH-09)
- Max weighted agreement: `8.5` (7 semantic×1.0 + 2 relational×0.5 + 1 structural×0.5)
- Healthy churn: `< 0.3`
- Warning churn: `[0.3, 0.5)`
- Unstable/High churn: `>= 0.5`
- Dream trigger: `entropy > 0.7 AND churn > 0.5` (AP-70)

## CONSTITUTION RULES - CRITICAL

These rules MUST be reflected in the skill instructions:

| Rule ID | Rule | Impact on Skill |
|---------|------|-----------------|
| ARCH-09 | Topic threshold >= 2.5 weighted agreement | Explain to users what makes a valid topic |
| AP-60 | Temporal embedders (E2-E4) weight = 0.0 | Topics based on meaning, not timing |
| AP-70 | Dream when entropy > 0.7 AND churn > 0.5 | Recommend `/dream-consolidation` skill |
| AP-62 | Divergence uses SEMANTIC only (E1,E5,E6,E7,E10,E12,E13) | Explain what triggers divergence |

## File to Create/Modify

**Replace**: `/home/cabdru/contextgraph/.claude/skills/topic-explorer/SKILL.md`

## Implementation Specification

Create SKILL.md with this EXACT content:

```markdown
---
name: topic-explorer
description: Explore emergent topic portfolio, stability metrics, and weighted agreement scores. Use for topic queries, stability checks, understanding topic relationships. Keywords: topics, portfolio, stability, churn, weighted agreement, divergence.
allowed-tools: Read,Glob,Grep
model: sonnet
version: 1.0.0
user-invocable: true
---
# Topic Explorer

Explore the emergent topic portfolio discovered via weighted multi-space clustering.

## Overview

Topics emerge autonomously from clustering across 13 embedding spaces. This skill queries the current topic state using MCP tools. Topics are NOT manually defined - they emerge when memories cluster in 3+ semantic spaces with weighted_agreement >= 2.5.

**Embedder Categories for Topic Detection**:
- SEMANTIC (weight 1.0): E1, E5, E6, E7, E10, E12, E13 - Primary topic triggers
- RELATIONAL (weight 0.5): E8, E11 - Supporting
- STRUCTURAL (weight 0.5): E9 - Supporting
- TEMPORAL (weight 0.0): E2, E3, E4 - NEVER count toward topics (metadata only)

**Max weighted_agreement**: 8.5 (7×1.0 + 2×0.5 + 1×0.5)

## Instructions

When the user asks about topics, topic stability, or knowledge graph structure:

### Query Topic Portfolio

1. Call `get_topic_portfolio` with appropriate format:
   - `{"format": "brief"}` - Quick overview (names + confidence)
   - `{"format": "standard"}` - Normal view (includes contributing spaces)
   - `{"format": "verbose"}` - Detailed view (full profiles)

2. Interpret the response:
   - Check `tier` field to understand system maturity (0-6)
   - Present topics sorted by confidence
   - Show contributing embedding spaces
   - Note lifecycle phase (Emerging, Stable, Declining, Merging)

### Query Topic Stability

1. Call `get_topic_stability` with lookback period:
   - `{"hours": 6}` - Default, recent stability
   - `{"hours": 24}` - Daily stability
   - `{"hours": 168}` - Weekly stability

2. Interpret stability metrics:
   - `churn_rate < 0.3`: Healthy (topics are stable)
   - `churn_rate >= 0.3 and < 0.5`: Warning (some instability)
   - `churn_rate >= 0.5`: Unstable (high topic turnover)
   - `entropy > 0.7`: High novelty/unfamiliarity

3. Check `dream_recommended` flag:
   - If `true`: Suggest running `/dream-consolidation` skill
   - Trigger condition: entropy > 0.7 AND churn > 0.5

## MCP Tools

### get_topic_portfolio
Get all discovered topics with profiles.

**Parameters**:
- `format` (optional): "brief" | "standard" | "verbose" (default: "standard")

**Returns**:
- `topics`: Array of discovered topics with id, name, confidence, weighted_agreement, member_count, contributing_spaces, phase
- `stability`: Portfolio-level churn_rate, entropy, is_stable
- `total_topics`: Count of topics meeting threshold
- `tier`: Current progressive tier (0-6)

### get_topic_stability
Get portfolio-level stability metrics.

**Parameters**:
- `hours` (optional): Lookback period 1-168 (default: 6)

**Returns**:
- `churn_rate`: Topic turnover rate [0.0-1.0]
- `entropy`: Distribution entropy [0.0-1.0]
- `phases`: Breakdown by lifecycle (emerging, stable, declining, merging)
- `dream_recommended`: Whether consolidation is needed
- `high_churn_warning`: Whether churn exceeds threshold

## Output Formats

### Brief Format
```
Topics (N discovered, Tier X):
1. [Topic Name] - confidence: X.XX (Stable)
2. [Topic Name] - confidence: X.XX (Emerging)
...
Stability: churn=0.XX, entropy=0.XX
```

### Standard Format
```
Topics (N discovered, Tier X):

1. [Topic Name]
   Confidence: X.XX (weighted_agreement: X.X/8.5)
   Members: N memories
   Contributing: E1_Semantic, E5_Causal, E7_Code
   Phase: Stable

2. [Topic Name]
   ...

Portfolio Stability:
- Churn Rate: 0.XX (healthy < 0.3)
- Entropy: 0.XX
- Phase Distribution: N emerging, N stable, N declining
```

## Edge Cases

### Tier 0 (No Memories)
Response: "System at Tier 0 - no memories stored yet. Topics will emerge once memories are injected and cluster across semantic spaces."

### No Topics Discovered
Response: "No topics discovered yet. Topics emerge when memories cluster in 3+ embedding spaces with weighted_agreement >= 2.5. Current tier: X with N memories."

### High Churn Warning
If `churn_rate >= 0.5`, include: "High churn detected (X.XX) - topic structure is unstable. Consider running `/dream-consolidation` to stabilize."

### Dream Recommended
If `dream_recommended: true`, include: "Dream consolidation recommended (entropy: X.XX, churn: X.XX). Run `/dream-consolidation` to consolidate memories and stabilize topics."

## Example Usage

**User**: "What topics have emerged?"
**Action**: Call `get_topic_portfolio({"format": "standard"})`
**Response**: Format topics with contributing spaces and stability summary.

**User**: "Is my topic structure stable?"
**Action**: Call `get_topic_stability({"hours": 6})`
**Response**: Interpret metrics and provide health assessment.

**User**: "Give me a quick overview of topics"
**Action**: Call `get_topic_portfolio({"format": "brief"})`
**Response**: Compact list with names and confidence.

**User**: "Show detailed topic profiles"
**Action**: Call `get_topic_portfolio({"format": "verbose"})`
**Response**: Full profiles with all 13 embedding space strengths.
```

## Definition of Done

### File Verification
- [ ] File exists at `/home/cabdru/contextgraph/.claude/skills/topic-explorer/SKILL.md`
- [ ] Frontmatter contains `model: sonnet`
- [ ] Frontmatter contains `user-invocable: true`
- [ ] Frontmatter contains `version: 1.0.0` (upgraded from 0.1.0)
- [ ] No "PLACEHOLDER" text remains

### Content Verification
- [ ] MCP tools documented: `get_topic_portfolio`, `get_topic_stability` (NOT get_memetic_status)
- [ ] Output formats documented (brief, standard, verbose)
- [ ] Edge cases documented (Tier 0, no topics, high churn, dream recommended)
- [ ] Constitution thresholds correctly stated (2.5, 8.5, 0.3, 0.5, 0.7)
- [ ] Embedder categories explained (SEMANTIC, RELATIONAL, STRUCTURAL, TEMPORAL)

### Manual Testing
- [ ] File is valid Markdown (parseable)
- [ ] Frontmatter is valid YAML
- [ ] File can be discovered by Claude Code skills system

## Verification Commands

```bash
cd /home/cabdru/contextgraph

# 1. Verify file exists and is not placeholder
test -f .claude/skills/topic-explorer/SKILL.md && \
  ! grep -q "PLACEHOLDER" .claude/skills/topic-explorer/SKILL.md && \
  echo "PASS: File exists and not placeholder" || echo "FAIL"

# 2. Verify frontmatter
head -8 .claude/skills/topic-explorer/SKILL.md
# Expected:
# ---
# name: topic-explorer
# description: Explore emergent topic portfolio...
# allowed-tools: Read,Glob,Grep
# model: sonnet
# version: 1.0.0
# user-invocable: true
# ---

# 3. Verify correct MCP tools documented
grep "get_topic_portfolio\|get_topic_stability" .claude/skills/topic-explorer/SKILL.md | wc -l
# Expected: >= 6 occurrences

# 4. Verify WRONG tools are NOT documented
grep -c "get_memetic_status" .claude/skills/topic-explorer/SKILL.md
# Expected: 0

# 5. Verify edge cases section exists
grep -c "Tier 0\|No Topics\|High Churn\|Dream Recommended" .claude/skills/topic-explorer/SKILL.md
# Expected: >= 4

# 6. Verify thresholds documented
grep -c "2.5\|8.5\|0.3\|0.5\|0.7" .claude/skills/topic-explorer/SKILL.md
# Expected: >= 5

# 7. Verify embedder categories
grep -c "SEMANTIC\|RELATIONAL\|STRUCTURAL\|TEMPORAL" .claude/skills/topic-explorer/SKILL.md
# Expected: >= 4
```

## Full State Verification (FSV) Protocol

After completing the implementation:

### 1. Source of Truth Identification
- **Source**: Filesystem at `/home/cabdru/contextgraph/.claude/skills/topic-explorer/SKILL.md`
- **Verification Method**: Direct file read and content comparison

### 2. Execute & Inspect
```bash
# Read the actual file content
cat /home/cabdru/contextgraph/.claude/skills/topic-explorer/SKILL.md

# Verify file size (should be ~4-5KB, not 1KB placeholder)
wc -c /home/cabdru/contextgraph/.claude/skills/topic-explorer/SKILL.md
# Expected: >= 4000 bytes

# Verify line count (should be ~150+ lines, not 31)
wc -l /home/cabdru/contextgraph/.claude/skills/topic-explorer/SKILL.md
# Expected: >= 140 lines
```

### 3. Boundary & Edge Case Audit

**Edge Case 1: Empty/Malformed YAML Frontmatter**
```bash
# Before: Check current frontmatter
head -8 .claude/skills/topic-explorer/SKILL.md

# After: Verify frontmatter parses correctly
python3 -c "
import yaml
with open('.claude/skills/topic-explorer/SKILL.md') as f:
    content = f.read()
    frontmatter = content.split('---')[1]
    data = yaml.safe_load(frontmatter)
    assert data['model'] == 'sonnet', 'Model must be sonnet'
    assert data['user-invocable'] == True, 'Must be user-invocable'
    print('PASS: Frontmatter valid')
"
```

**Edge Case 2: Missing Required Sections**
```bash
# Verify all required sections exist
for section in "Instructions" "MCP Tools" "Output Formats" "Edge Cases" "Example Usage"; do
  grep -q "## $section" .claude/skills/topic-explorer/SKILL.md && \
    echo "PASS: Found '$section'" || echo "FAIL: Missing '$section'"
done
```

**Edge Case 3: Incorrect Tool Names**
```bash
# Verify correct tools mentioned, wrong tools NOT mentioned
grep -q "get_topic_portfolio" .claude/skills/topic-explorer/SKILL.md && echo "PASS: Correct tool" || echo "FAIL"
grep -q "get_topic_stability" .claude/skills/topic-explorer/SKILL.md && echo "PASS: Correct tool" || echo "FAIL"
! grep -q "get_memetic_status" .claude/skills/topic-explorer/SKILL.md && echo "PASS: Wrong tool absent" || echo "FAIL: Wrong tool present"
```

### 4. Evidence of Success

Provide a log showing:
1. File content summary (first 20 lines, last 10 lines)
2. File size and line count
3. Results of all verification commands
4. Screenshot or copy of the Output Formats section to prove formatting is correct

## NO BACKWARDS COMPATIBILITY

- Delete the existing placeholder content entirely
- Do NOT preserve any "PLACEHOLDER" text
- Do NOT keep references to wrong MCP tools (get_memetic_status, search_graph)
- If something doesn't work, it should error with clear logging
- NO mock data - use real MCP tool schemas from the handlers

## FAILURE MODES

If any of these occur, the task is NOT complete:

| Failure | How to Detect | Fix |
|---------|---------------|-----|
| Placeholder text remains | `grep PLACEHOLDER` returns results | Remove all placeholder content |
| Wrong MCP tools documented | `grep get_memetic_status` returns results | Replace with correct tools |
| Missing edge cases | Edge case section incomplete | Add all 4 edge cases |
| Invalid YAML frontmatter | Python yaml.safe_load fails | Fix frontmatter syntax |
| File too small | `wc -c` < 4000 bytes | Content is incomplete |
| Missing sections | grep for section headers fails | Add missing sections |
