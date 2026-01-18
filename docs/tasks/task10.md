# Task 10: Implement memory-inject SKILL.md

## Metadata
- **Task ID**: TASK-GAP-010
- **Phase**: 3 (Skills Framework)
- **Priority**: Medium
- **Complexity**: Medium
- **Dependencies**:
  - task02 (directory created at `.claude/skills/memory-inject/`)
  - MCP tools registered (search_graph, inject_context)
- **Last Audited**: 2026-01-18

---

## Critical Corrections from Original Task

**ORIGINAL SPEC WAS INCORRECT.** The original task specified `inject_context` as a retrieval tool. This is **WRONG**.

| Tool | Actual Purpose | Parameters |
|------|----------------|------------|
| `search_graph` | **RETRIEVAL** - Searches memories by semantic similarity | `query` (required), `topK`, `minSimilarity`, `modality` |
| `inject_context` | **STORAGE** - Stores new context with UTL processing | `content` (required), `rationale` (required), `modality`, `importance` |

The memory-inject skill must use `search_graph` for retrieval, NOT `inject_context`.

---

## Objective

Replace the placeholder SKILL.md at `.claude/skills/memory-inject/SKILL.md` with a fully functional skill that:
1. Uses `search_graph` MCP tool to retrieve relevant memories
2. Supports three verbosity levels (compact, standard, verbose)
3. Manages token budgets for context injection
4. Handles edge cases properly

---

## Current State (Source of Truth)

### File Location
```
/home/cabdru/contextgraph/.claude/skills/memory-inject/SKILL.md
```

### Current Content (Placeholder - 32 lines)
```markdown
---
name: memory-inject
description: Retrieve and inject contextual memories for current task...
allowed-tools: Read,Glob
model: haiku
version: 0.1.0
user-invocable: true
---
# Memory Inject

**STATUS: PLACEHOLDER - Full implementation in TASK-GAP-010**
[... placeholder content ...]
```

### Reference Implementation
Use `/home/cabdru/contextgraph/.claude/skills/topic-explorer/SKILL.md` as the format reference (150 lines, fully implemented).

---

## MCP Tools - Accurate Signatures

### search_graph (PRIMARY - Use for retrieval)
**Location**: `crates/context-graph-mcp/src/tools/definitions/core.rs:94-127`

```json
{
  "name": "search_graph",
  "description": "Search the knowledge graph using semantic similarity. Returns nodes matching the query with relevance scores.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query text"
      },
      "topK": {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
        "default": 10,
        "description": "Maximum number of results to return"
      },
      "minSimilarity": {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
        "default": 0.0,
        "description": "Minimum similarity threshold [0.0, 1.0]"
      },
      "modality": {
        "type": "string",
        "enum": ["text", "code", "image", "audio", "structured", "mixed"],
        "description": "Filter results by modality"
      }
    },
    "required": ["query"]
  }
}
```

### inject_context (SECONDARY - Use for storage only)
**Location**: `crates/context-graph-mcp/src/tools/definitions/core.rs:12-43`

```json
{
  "name": "inject_context",
  "description": "Inject context into the knowledge graph with UTL processing. Analyzes content for learning potential and stores with computed metrics.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "content": {
        "type": "string",
        "description": "The content to inject into the knowledge graph"
      },
      "rationale": {
        "type": "string",
        "description": "Why this context is relevant and should be stored"
      },
      "modality": {
        "type": "string",
        "enum": ["text", "code", "image", "audio", "structured", "mixed"],
        "default": "text"
      },
      "importance": {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
        "default": 0.5
      }
    },
    "required": ["content", "rationale"]
  }
}
```

---

## Implementation Steps

### Step 1: Read Current Files
```bash
# Verify current placeholder exists
cat /home/cabdru/contextgraph/.claude/skills/memory-inject/SKILL.md

# Read reference implementation for format
cat /home/cabdru/contextgraph/.claude/skills/topic-explorer/SKILL.md
```

### Step 2: Write SKILL.md

Create `/home/cabdru/contextgraph/.claude/skills/memory-inject/SKILL.md` with this exact content:

```markdown
---
name: memory-inject
description: "Retrieve and inject contextual memories for current task. Uses search_graph for semantic retrieval across 13 embedding spaces. Supports compact/standard/verbose formats with token budgeting. Keywords: memory, context, inject, retrieve, recall, background."
allowed-tools: Read,Glob,Grep
model: haiku
version: 1.0.0
user-invocable: true
---
# Memory Inject

Retrieve relevant memories from the knowledge graph and format them for context injection.

## Overview

This skill searches the knowledge graph using `search_graph` to find semantically similar memories. Results are ranked by relevance across the 13 embedding spaces (weighted by category: SEMANTIC=1.0, RELATIONAL=0.5, STRUCTURAL=0.5, TEMPORAL=0.0 for metadata only).

**Primary Tool**: `search_graph` (retrieval)
**Secondary Tool**: `inject_context` (storage - use only when explicitly saving new context)

## Instructions

When the user needs context, background, or wants to recall previous work:

### Basic Retrieval

1. Extract the query from user input or current task context
2. Call `search_graph` with appropriate parameters:
   ```json
   {
     "query": "<user query or task context>",
     "topK": 10,
     "minSimilarity": 0.3
   }
   ```
3. Format results based on requested verbosity
4. Track token usage and truncate if budget exceeded

### Verbosity Levels

| Level | Token Budget | Content |
|-------|--------------|---------|
| compact | ~300 tokens | Titles + relevance scores only |
| standard | ~800 tokens | Full content + source + timestamp |
| verbose | ~1200 tokens | Add per-embedder similarity scores |

### Code Context Retrieval

For code-related queries, add modality filter:
```json
{
  "query": "authentication implementation",
  "topK": 10,
  "minSimilarity": 0.3,
  "modality": "code"
}
```

## MCP Tools

### search_graph (Primary)

Retrieves memories by semantic similarity.

**Parameters**:
- `query` (required): Search query text
- `topK` (optional): Max results 1-100 (default: 10)
- `minSimilarity` (optional): Threshold 0.0-1.0 (default: 0.0)
- `modality` (optional): Filter by type (text, code, image, audio, structured, mixed)

**Returns**:
```json
{
  "results": [
    {
      "id": "uuid",
      "content": "memory content",
      "similarity": 0.85,
      "modality": "text",
      "importance": 0.7,
      "created_at": "2026-01-15T10:30:00Z"
    }
  ],
  "total_found": 15,
  "query_time_ms": 12
}
```

### inject_context (Secondary - Storage Only)

Store new context into the knowledge graph. Use ONLY when user explicitly wants to save something.

**Parameters**:
- `content` (required): Content to store
- `rationale` (required): Why this should be stored
- `modality` (optional): Content type (default: "text")
- `importance` (optional): Score 0.0-1.0 (default: 0.5)

**Returns**:
```json
{
  "id": "uuid",
  "learning_score": 0.65,
  "entropy": 0.42,
  "coherence": 0.78
}
```

## Output Formats

### Compact Format
```
Relevant Memories (N found):
- [Memory title/summary] (relevance: 0.XX)
- [Memory title/summary] (relevance: 0.XX)
[N results, ~M tokens]
```

### Standard Format
```
Retrieved Memories (N found):

1. [Memory content]
   - Source: text | Created: 2026-01-15 10:30
   - Relevance: 0.XX | Importance: 0.XX

2. [Memory content]
   - Source: code | Created: 2026-01-14 15:45
   - Relevance: 0.XX | Importance: 0.XX

Token usage: M/budget
```

### Verbose Format
Standard format plus per-embedder similarity breakdown:
```
1. [Memory content]
   - Source: text | Created: 2026-01-15 10:30
   - Relevance: 0.XX | Importance: 0.XX
   - Embedder Similarities:
     E1_Semantic: 0.XX (SEMANTIC, weight: 1.0)
     E5_Causal: 0.XX (SEMANTIC, weight: 1.0)
     E7_Code: 0.XX (SEMANTIC, weight: 1.0)
     E8_Graph: 0.XX (RELATIONAL, weight: 0.5)
     [E2-E4 temporal excluded from scoring]
```

## Embedder Categories Reference

| Category | Embedders | Weight | Role |
|----------|-----------|--------|------|
| SEMANTIC | E1, E5, E6, E7, E10, E12, E13 | 1.0 | Primary relevance |
| RELATIONAL | E8, E11 | 0.5 | Supporting |
| STRUCTURAL | E9 | 0.5 | Supporting |
| TEMPORAL | E2, E3, E4 | 0.0 | Metadata only |

## Edge Cases

| Condition | Response |
|-----------|----------|
| No query provided | "Please provide a search query to retrieve relevant memories." |
| No memories found | "No relevant memories found for '[query]'. The knowledge graph may not have related content yet." |
| Zero results above threshold | "No memories exceeded the similarity threshold (0.XX). Try lowering minSimilarity or broadening your query." |
| Token budget exceeded | "[N results truncated to fit budget. Showing top M most relevant.]" |
| Empty knowledge graph | "Knowledge graph is empty. No memories have been stored yet." |

## Example Usage

| User Request | Action | Parameters |
|--------------|--------|------------|
| "What do we know about authentication?" | `search_graph` | `{"query": "authentication", "topK": 10}` |
| "Get code context for the API" | `search_graph` | `{"query": "API implementation", "modality": "code"}` |
| "Brief overview of recent work" | `search_graph` | `{"query": "recent work tasks", "topK": 5}` |
| "Detailed context with scores" | `search_graph` + verbose format | `{"query": "...", "topK": 15}` |
| "Save this finding for later" | `inject_context` | `{"content": "...", "rationale": "..."}` |

## Token Budgeting

1. **Estimate tokens**: ~4 chars per token average
2. **Reserve header**: ~50 tokens for format headers
3. **Per-result cost**:
   - Compact: ~20 tokens/result
   - Standard: ~80 tokens/result
   - Verbose: ~150 tokens/result
4. **Truncation**: When approaching budget, stop adding results and note truncation

## Do NOT

- Use `inject_context` for retrieval (it's for STORAGE)
- Guess memory content - only show actual results
- Exceed token budget without noting truncation
- Include temporal embedder scores in relevance calculations (weight = 0.0)
```

---

## Definition of Done

- [ ] File exists at `/home/cabdru/contextgraph/.claude/skills/memory-inject/SKILL.md`
- [ ] Frontmatter has `model: haiku` (per PRD Section 9.3)
- [ ] Frontmatter has `user-invocable: true`
- [ ] Frontmatter has `version: 1.0.0`
- [ ] Documents `search_graph` as PRIMARY retrieval tool (not inject_context)
- [ ] Documents `inject_context` as SECONDARY storage tool
- [ ] Keywords documented: memory, context, inject, retrieve, recall, background
- [ ] Three verbosity levels documented (compact, standard, verbose)
- [ ] Edge cases documented (no query, no results, token exceeded, empty graph)
- [ ] Embedder categories reference table included
- [ ] Token budgeting section included
- [ ] File is valid Markdown with valid YAML frontmatter

---

## Full State Verification

### Source of Truth
The file `/home/cabdru/contextgraph/.claude/skills/memory-inject/SKILL.md`

### Verification Commands

```bash
cd /home/cabdru/contextgraph

# 1. Verify file exists and is not a placeholder
test -f .claude/skills/memory-inject/SKILL.md && echo "FILE EXISTS"
grep -c "PLACEHOLDER" .claude/skills/memory-inject/SKILL.md  # Should be 0

# 2. Verify frontmatter
head -10 .claude/skills/memory-inject/SKILL.md
# Expected:
# ---
# name: memory-inject
# description: "Retrieve and inject contextual memories..."
# allowed-tools: Read,Glob,Grep
# model: haiku
# version: 1.0.0
# user-invocable: true
# ---

# 3. Verify search_graph is documented as primary
grep -c "search_graph" .claude/skills/memory-inject/SKILL.md  # Should be >= 10

# 4. Verify inject_context is documented as secondary/storage
grep -A2 "inject_context" .claude/skills/memory-inject/SKILL.md | grep -i "storage\|secondary"

# 5. Verify verbosity levels
grep -E "compact|standard|verbose" .claude/skills/memory-inject/SKILL.md | wc -l  # Should be >= 6

# 6. Verify embedder categories table
grep -E "SEMANTIC|TEMPORAL|RELATIONAL|STRUCTURAL" .claude/skills/memory-inject/SKILL.md | wc -l  # Should be >= 4

# 7. Verify line count (should be ~180-220 lines, not 32 like placeholder)
wc -l .claude/skills/memory-inject/SKILL.md  # Should be >= 150

# 8. Validate YAML frontmatter syntax
python3 -c "
import yaml
with open('.claude/skills/memory-inject/SKILL.md', 'r') as f:
    content = f.read()
    # Extract frontmatter between ---
    parts = content.split('---')
    if len(parts) >= 3:
        frontmatter = yaml.safe_load(parts[1])
        assert frontmatter['name'] == 'memory-inject'
        assert frontmatter['model'] == 'haiku'
        assert frontmatter['user-invocable'] == True
        assert frontmatter['version'] == '1.0.0'
        print('FRONTMATTER VALID')
    else:
        print('ERROR: Invalid frontmatter structure')
"
```

---

## Synthetic Test Data

### Test Case 1: Basic Query
**Input**:
```json
{"query": "authentication implementation", "topK": 5}
```
**Expected search_graph call**: Valid
**Expected output format**: Standard (5 results max)

### Test Case 2: Code Modality Filter
**Input**:
```json
{"query": "API endpoints", "topK": 10, "modality": "code", "minSimilarity": 0.4}
```
**Expected search_graph call**: Valid with modality filter
**Expected output**: Only code-type memories with similarity >= 0.4

### Test Case 3: Verbose Format Request
**Input**: User asks "Give me detailed context with similarity scores"
**Expected**: Call search_graph, format with verbose output including E1, E5, E6, E7, E10, E12, E13 scores

### Test Case 4: Empty Query (Edge Case)
**Input**: User says "/memory-inject" with no query
**Expected**: Prompt for query: "Please provide a search query..."

### Test Case 5: Storage Request (Secondary Function)
**Input**: User says "Save this: The login system uses JWT tokens"
**Expected inject_context call**:
```json
{
  "content": "The login system uses JWT tokens",
  "rationale": "User explicitly requested to save this information",
  "modality": "text",
  "importance": 0.5
}
```

---

## Boundary & Edge Case Audit

### Edge Case 1: Empty Knowledge Graph
**State Before**: No memories in graph (node_count = 0)
**Action**: Call search_graph with any query
**Expected State After**: Return empty results with message "Knowledge graph is empty"
**Verification**: `get_memetic_status` returns `node_count: 0`

### Edge Case 2: Maximum topK (100)
**State Before**: Graph has 500+ memories
**Action**: `search_graph({"query": "test", "topK": 100})`
**Expected State After**: Returns exactly 100 results
**Verification**: Response `results.length === 100`

### Edge Case 3: minSimilarity = 1.0 (Exact Match Only)
**State Before**: Graph has memories
**Action**: `search_graph({"query": "unique phrase xyz", "minSimilarity": 1.0})`
**Expected State After**: Returns 0 results (exact match unlikely)
**Verification**: Response `results.length === 0` with appropriate message

---

## Evidence of Success Checklist

After implementation, provide logs showing:

1. **File content verification**:
   ```bash
   cat .claude/skills/memory-inject/SKILL.md | head -50
   ```

2. **Line count verification**:
   ```bash
   wc -l .claude/skills/memory-inject/SKILL.md
   # Expected: >= 150 lines
   ```

3. **No placeholder markers**:
   ```bash
   grep -i "placeholder\|not.*implemented\|todo" .claude/skills/memory-inject/SKILL.md
   # Expected: No output
   ```

4. **Tool documentation verification**:
   ```bash
   grep -c "search_graph" .claude/skills/memory-inject/SKILL.md
   # Expected: >= 10 occurrences
   ```

5. **Frontmatter validation**:
   ```bash
   head -10 .claude/skills/memory-inject/SKILL.md
   # Expected: Valid YAML with model: haiku, version: 1.0.0
   ```

---

## Related Files (DO NOT MODIFY)

These files exist and should NOT be changed by this task:

| File | Status | Notes |
|------|--------|-------|
| `crates/context-graph-mcp/src/tools/definitions/core.rs` | IMPLEMENTED | Contains search_graph, inject_context |
| `crates/context-graph-mcp/src/tools/definitions/topic.rs` | IMPLEMENTED | Topic tools |
| `crates/context-graph-mcp/src/tools/definitions/curation.rs` | IMPLEMENTED | Curation tools |
| `.claude/skills/topic-explorer/SKILL.md` | IMPLEMENTED | Reference format |
| `.claude/settings.json` | CONFIGURED | Native hooks |

---

## Anti-Patterns to Avoid

1. **AP-WRONG-TOOL**: Using `inject_context` for retrieval (it's for STORAGE)
2. **AP-MOCK-DATA**: Using fake/mock data in tests - use real search_graph calls
3. **AP-SILENT-FAIL**: Catching errors without logging - all errors must be visible
4. **AP-BACKWARDS-COMPAT**: No fallbacks or workarounds - fail fast if broken
5. **AP-GUESS-PARAMS**: Do not guess tool parameters - use exact signatures from core.rs

---

## MANDATORY: Physical Verification Protocol

### Principle
In computing, there's a trigger event that initiates process X, which leads to outcome Y. The trigger can be identified when it occurs, and whatever Y produces can be tracked. **You MUST manually verify outputs exist.**

### Verification Steps After Implementation

#### Step 1: Verify File Physically Exists
```bash
# Check file exists and is not empty
ls -la /home/cabdru/contextgraph/.claude/skills/memory-inject/SKILL.md
stat /home/cabdru/contextgraph/.claude/skills/memory-inject/SKILL.md
```

**Expected Output**: File exists with size > 5000 bytes

#### Step 2: Verify Content Is Correct
```bash
# Verify not a placeholder
grep -c "PLACEHOLDER" /home/cabdru/contextgraph/.claude/skills/memory-inject/SKILL.md
# Expected: 0

# Verify search_graph is primary tool
grep -c "search_graph" /home/cabdru/contextgraph/.claude/skills/memory-inject/SKILL.md
# Expected: >= 10

# Verify inject_context documented as storage
grep -B1 -A1 "inject_context" /home/cabdru/contextgraph/.claude/skills/memory-inject/SKILL.md | grep -i "storage\|secondary\|only"
# Expected: matches showing it's for storage
```

#### Step 3: Validate YAML Frontmatter
```bash
# Extract and validate frontmatter
python3 << 'EOF'
import yaml
import sys

with open('/home/cabdru/contextgraph/.claude/skills/memory-inject/SKILL.md', 'r') as f:
    content = f.read()

# Split on --- to get frontmatter
parts = content.split('---', 2)
if len(parts) < 3:
    print("ERROR: Invalid frontmatter structure")
    sys.exit(1)

try:
    fm = yaml.safe_load(parts[1])

    # Required fields
    checks = [
        ('name', 'memory-inject'),
        ('model', 'haiku'),
        ('version', '1.0.0'),
    ]

    errors = []
    for key, expected in checks:
        actual = fm.get(key)
        if actual != expected:
            errors.append(f"{key}: expected '{expected}', got '{actual}'")

    # Check user-invocable (can be user_invocable or user-invocable)
    if not (fm.get('user-invocable') == True or fm.get('user_invocable') == True):
        errors.append("user-invocable: expected True")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("FRONTMATTER VALIDATION: PASSED")
    print(f"  name: {fm.get('name')}")
    print(f"  model: {fm.get('model')}")
    print(f"  version: {fm.get('version')}")
    print(f"  user-invocable: {fm.get('user-invocable', fm.get('user_invocable'))}")

except yaml.YAMLError as e:
    print(f"YAML PARSE ERROR: {e}")
    sys.exit(1)
EOF
```

#### Step 4: MCP Tool Integration Test

**Prerequisite**: MCP server must be running

```bash
# Test that search_graph tool is callable
# This verifies the tool the skill documents actually exists

# Option A: Use context-graph-cli (if available)
context-graph-cli tools list | grep search_graph

# Option B: Check tool is registered in MCP
grep -r "search_graph" /home/cabdru/contextgraph/crates/context-graph-mcp/src/tools/definitions/
```

**Expected**: `search_graph` tool exists with correct signature

---

## Manual Testing Protocol

### Test 1: Skill Invocation in Claude Code

**Setup**: Start Claude Code session in `/home/cabdru/contextgraph`

**Action**: Type `/memory-inject` or ask "recall authentication discussions"

**Verification**:
1. Check Claude Code transcript for tool calls
2. Verify `search_graph` is called (NOT `inject_context`)
3. Verify response follows documented output format

**Evidence Required**: Screenshot or copy of Claude Code response showing search_graph call

### Test 2: Synthetic Data Happy Path

**Input Data**:
```json
{
  "query": "authentication flow OAuth",
  "topK": 5,
  "minSimilarity": 0.3
}
```

**Expected Behavior**:
1. Skill should call `search_graph` with these parameters
2. Results should be formatted per Output Formats section
3. Token budget should be tracked

**State Before**: Note memory count via `get_memetic_status`
**State After**: Memory count unchanged (retrieval doesn't add memories)

### Test 3: Edge Case - Empty Query

**Input**: User invokes skill without providing a query

**Expected**:
- Skill prompts: "Please provide a search query..."
- No `search_graph` call made with empty query
- No error thrown

**Verification**: Check Claude Code transcript shows prompt, not error

### Test 4: Edge Case - Zero Results

**Input**:
```json
{
  "query": "xyznonexistentquery12345",
  "topK": 10
}
```

**Expected**:
- `search_graph` returns `{"results": [], "count": 0}`
- Skill displays: "No relevant memories found for 'xyznonexistentquery12345'"
- No error thrown

**Verification**: Response matches edge case documentation

### Test 5: Modality Filter

**Input**:
```json
{
  "query": "function implementation",
  "modality": "code",
  "topK": 10
}
```

**Expected**:
- Only code-modality memories returned
- If no code memories exist, appropriate message shown

**Verification**: All returned results have modality="code" (or empty if none exist)

---

## Database/Storage Verification

After any memory injection (via `inject_context`), verify physical storage:

```bash
# Check RocksDB has the data (development)
ls -la /home/cabdru/contextgraph/data/rocksdb/

# Use MCP tool to verify
# Call get_memetic_status and check node_count increased

# For search results, verify they match actual stored content
# by comparing fingerprintId from search_graph with stored data
```

---

## Failure Recovery

If tests fail, debug using:

```bash
# 1. Check MCP server logs
tail -f /home/cabdru/contextgraph/logs/mcp.log

# 2. Verify tool registration
grep -r "TOOL_SEARCH_GRAPH\|search_graph" /home/cabdru/contextgraph/crates/context-graph-mcp/src/

# 3. Check skill is discoverable
ls -la /home/cabdru/contextgraph/.claude/skills/

# 4. Validate skill frontmatter syntax
head -10 /home/cabdru/contextgraph/.claude/skills/memory-inject/SKILL.md
```

**DO NOT**:
- Create workarounds for failing tests
- Use mock data to make tests pass
- Silence errors

**DO**:
- Log exact error messages
- Trace to root cause
- Fix the actual issue

---

## Success Criteria Summary

| Criterion | Verification Method | Expected Result |
|-----------|---------------------|-----------------|
| File exists | `test -f .claude/skills/memory-inject/SKILL.md` | Exit code 0 |
| Not placeholder | `grep -c PLACEHOLDER ...` | Count = 0 |
| Correct model | `grep "model: haiku"` | Match found |
| Correct version | `grep "version: 1.0.0"` | Match found |
| search_graph documented | `grep -c search_graph ...` | Count >= 10 |
| inject_context = storage | `grep inject_context ... \| grep storage` | Match found |
| Valid YAML | Python yaml.safe_load | No exception |
| Line count | `wc -l` | >= 150 lines |
| Claude Code test | Manual invocation | search_graph called |

---

## Appendix: Actual Tool Implementations

### search_graph Handler
**File**: `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs:338-427`

Key implementation details:
- Generates query embedding via `multi_array_provider.embed_all(query)`
- Uses `TeleologicalSearchOptions::quick(top_k)` for search
- Returns `fingerprintId`, `similarity`, `purposeAlignment`, `dominantEmbedder`
- Optionally includes `content` if `includeContent: true`

### inject_context Handler
**File**: `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs:22-190`

Key implementation details:
- **STORES** content, does NOT retrieve
- Requires `content` and `rationale` parameters
- Computes UTL metrics (entropy, coherence, learning_score)
- Generates all 13 embeddings
- Returns `fingerprintId`, `utl` metrics

**NEVER use inject_context for retrieval in this skill.**
