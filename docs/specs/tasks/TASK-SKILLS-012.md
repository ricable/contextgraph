# TASK-SKILLS-012: Create Memory-Inject and Semantic-Search Skills

```xml
<task_spec id="TASK-SKILLS-012" version="1.0">
<metadata>
  <title>Create Memory-Inject and Semantic-Search Skills</title>
  <status>ready</status>
  <layer>surface</layer>
  <sequence>12</sequence>
  <implements>
    <requirement_ref>REQ-SKILLS-24</requirement_ref>
    <requirement_ref>REQ-SKILLS-25</requirement_ref>
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
This task creates two skill SKILL.md files: memory-inject for context injection/retrieval
and semantic-search for querying memory with semantic and causal search. Memory-inject
uses the haiku model for fast latency (<500ms target), while semantic-search uses sonnet
for more complex query understanding.

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
    - Create .claude/skills/memory-inject/ directory and SKILL.md
    - Create .claude/skills/semantic-search/ directory and SKILL.md
    - Define allowed MCP tools for each skill
    - Write skill instructions with progressive disclosure
    - Include keywords for auto-triggering
  </in_scope>
  <out_of_scope>
    - Dream and curation skills (TASK-SKILLS-013)
    - Subagent files (TASK-SKILLS-014)
    - Actual MCP tool implementation
  </out_of_scope>
</scope>

<definition_of_done>
  <file_content>
    <file path=".claude/skills/memory-inject/SKILL.md">
---
name: memory-inject
description: |
  Inject and retrieve context memories with background loading support.
  Use for storing new memories, retrieving existing context, and
  background memory operations.
  Keywords: memory, context, inject, retrieve, recall, background
allowed-tools: Read,Grep,mcp__context-graph__inject_memory,mcp__context-graph__retrieve_memory,mcp__context-graph__get_memory_stats,mcp__context-graph__list_memories
model: haiku
version: 1.0.0
---
# Memory Inject Skill

## Overview
This skill provides fast memory injection and retrieval for Context Graph.
Uses haiku model for <500ms latency target.

## When to Use
- Storing new memories or context
- Retrieving previously stored memories
- Background memory operations
- Checking memory statistics

## MCP Tools Available

### inject_memory
Stores a new memory with content and metadata.
- content: The memory content to store
- metadata: Optional metadata (tags, importance, etc.)
- Returns: Memory ID and injection timestamp

### retrieve_memory
Retrieves a specific memory by ID.
- memory_id: The ID of the memory to retrieve
- Returns: Memory content and metadata

### get_memory_stats
Returns memory system statistics.
- total_memories: Count of stored memories
- storage_used: Storage space used
- last_injection: Timestamp of last injection

### list_memories
Lists memories with optional filtering.
- limit: Maximum memories to return
- filter: Optional filter criteria
- Returns: List of memory summaries

## Protocol

1. **Injection**: Always include relevant metadata for searchability
2. **Retrieval**: Use memory IDs from previous operations or search
3. **Background**: For bulk operations, use background mode
4. **Stats Check**: Monitor memory stats periodically

## Output Format

For injection:
```
Memory Injected:
- ID: [memory_id]
- Timestamp: [timestamp]
- Size: [bytes]
```

For retrieval:
```
Memory Retrieved:
- ID: [memory_id]
- Content: [content_preview]
- Metadata: [metadata_summary]
```

## Performance

Target latency: <500ms for single operations
Use background mode for operations >10 memories
    </file>
    <file path=".claude/skills/semantic-search/SKILL.md">
---
name: semantic-search
description: |
  Search memories using semantic similarity and causal graph traversal.
  Use for finding relevant context, exploring memory connections,
  and causal relationship queries.
  Keywords: search, find, query, lookup, semantic, causal
allowed-tools: Read,Grep,mcp__context-graph__semantic_search,mcp__context-graph__causal_search,mcp__context-graph__get_related_memories,mcp__context-graph__traverse_graph
model: sonnet
version: 1.0.0
---
# Semantic Search Skill

## Overview
This skill provides semantic and causal search across Context Graph memories.
Uses sonnet model for complex query understanding.

## When to Use
- Finding semantically similar memories
- Exploring causal relationships
- Traversing the memory graph
- Complex multi-hop queries

## MCP Tools Available

### semantic_search
Searches memories by semantic similarity.
- query: Natural language search query
- limit: Maximum results to return
- threshold: Minimum similarity score (0.0-1.0)
- Returns: Ranked list of similar memories

### causal_search
Searches memories by causal relationships.
- query: The effect or cause to search for
- direction: "causes" or "effects"
- depth: Maximum causal chain depth
- Returns: Causal chain results

### get_related_memories
Gets memories related to a given memory.
- memory_id: Source memory ID
- relation_types: Types of relations to include
- Returns: List of related memories with relation types

### traverse_graph
Traverses the memory graph from a starting point.
- start_id: Starting memory ID
- direction: "outgoing", "incoming", or "both"
- max_depth: Maximum traversal depth
- Returns: Graph traversal results

## Protocol

1. **Semantic First**: Start with semantic search for initial results
2. **Causal Exploration**: Use causal search for cause/effect queries
3. **Graph Traversal**: Explore connections with traverse_graph
4. **Combine Results**: Merge results from multiple search types

## Query Patterns

### Simple Lookup
```
semantic_search(query="authentication implementation")
```

### Causal Chain
```
causal_search(query="login failure", direction="causes", depth=3)
```

### Graph Exploration
```
traverse_graph(start_id="mem-123", direction="both", max_depth=2)
```

## Output Format

```
Search Results ([count] matches):
1. [memory_id] (score: [similarity])
   [content_preview]
2. ...

Causal Chain:
[cause] -> [effect1] -> [effect2]
```

## Ranking

Results are ranked by:
1. Semantic similarity score
2. Recency (more recent = higher)
3. Importance (from metadata)
4. Access frequency
    </file>
  </file_content>
  <constraints>
    - Frontmatter must be valid YAML between --- delimiters
    - Description must include Keywords: line
    - Description must be <= 1024 characters
    - memory-inject must use model: haiku
    - semantic-search must use model: sonnet
    - Skill names must match directory names
  </constraints>
  <verification>
    - cargo test --package context-graph-cli skill_loader -- --test-threads=1
    - Verify both SKILL.md files are parseable
  </verification>
</definition_of_done>

<pseudo_code>
1. Create directory structures:
   mkdir -p .claude/skills/memory-inject
   mkdir -p .claude/skills/semantic-search

2. Create memory-inject/SKILL.md:
   - model: haiku (for <500ms latency)
   - Tools: inject_memory, retrieve_memory, get_memory_stats, list_memories
   - Keywords: memory, context, inject, retrieve, recall, background

3. Create semantic-search/SKILL.md:
   - model: sonnet (for complex queries)
   - Tools: semantic_search, causal_search, get_related_memories, traverse_graph
   - Keywords: search, find, query, lookup, semantic, causal

4. Verify both skills:
   - SkillLoader.load_metadata("memory-inject") succeeds
   - SkillLoader.load_metadata("semantic-search") succeeds
   - Keywords extracted correctly for both
</pseudo_code>

<files_to_create>
  <file path=".claude/skills/memory-inject/SKILL.md">Memory injection skill</file>
  <file path=".claude/skills/semantic-search/SKILL.md">Semantic search skill</file>
</files_to_create>

<test_commands>
  <command>cargo test --package context-graph-cli skill_loader -- --test-threads=1</command>
</test_commands>
</task_spec>
```

## Implementation Notes

### Memory-Inject Skill

| Field | Value | Rationale |
|-------|-------|-----------|
| model | haiku | Fast latency <500ms target |
| Tools | inject/retrieve/stats/list | Core memory CRUD operations |
| Keywords | memory, context, inject, retrieve, recall, background | Common memory terms |

### Semantic-Search Skill

| Field | Value | Rationale |
|-------|-------|-----------|
| model | sonnet | Complex query understanding |
| Tools | semantic/causal/related/traverse | Search and graph operations |
| Keywords | search, find, query, lookup, semantic, causal | Search-related terms |

### MCP Tools Mapping

**Memory-Inject:**
```
mcp__context-graph__inject_memory
mcp__context-graph__retrieve_memory
mcp__context-graph__get_memory_stats
mcp__context-graph__list_memories
```

**Semantic-Search:**
```
mcp__context-graph__semantic_search
mcp__context-graph__causal_search
mcp__context-graph__get_related_memories
mcp__context-graph__traverse_graph
```

### Keyword Separation

The skills have non-overlapping primary keywords to avoid trigger conflicts:

- memory-inject: focuses on "inject", "retrieve", "recall"
- semantic-search: focuses on "search", "find", "query"

Both may trigger on "memory" but with different confidence based on context.

## Verification Checklist

- [ ] .claude/skills/memory-inject/ directory exists
- [ ] .claude/skills/semantic-search/ directory exists
- [ ] memory-inject/SKILL.md has valid frontmatter
- [ ] semantic-search/SKILL.md has valid frontmatter
- [ ] memory-inject uses model: haiku
- [ ] semantic-search uses model: sonnet
- [ ] Both descriptions <= 1024 characters
- [ ] Both include Keywords: line
- [ ] Both parseable by SkillLoader
- [ ] Keywords extracted correctly for both
