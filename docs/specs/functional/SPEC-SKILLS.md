<?xml version="1.0" encoding="UTF-8"?>
<functional_spec id="SPEC-SKILLS" version="1.0">
<metadata>
  <title>Skills and Subagents Functional Specification</title>
  <status>draft</status>
  <phase>4</phase>
  <owner>Context Graph Team</owner>
  <last_updated>2026-01-15</last_updated>
  <estimated_effort>20 hours</estimated_effort>
  <depends_on>
    <spec_ref>SPEC-HOOKS</spec_ref>
    <spec_ref>SPEC-CLI</spec_ref>
    <spec_ref>SPEC-SESSION-IDENTITY</spec_ref>
  </depends_on>
  <prd_source>docs2/contextprd.md Sections 15.7 (Skills) and 15.8 (Subagents)</prd_source>
  <constitution_source>docs2/constitution.yaml claude_code.skills and claude_code.subagents sections</constitution_source>
  <skills_reference>docs2/claudeskills.md</skills_reference>
</metadata>

<overview>
## What Phase 4 Accomplishes

Phase 4 implements **Skills and Subagents** for the Context Graph system, extending Claude Code with domain-specific expertise and parallel work capabilities. Building on the native hooks infrastructure from Phase 3, this phase enables:

1. **Skills as Domain Expertise**: YAML-defined prompt extensions that transform Claude into domain specialists (consciousness, memory, search, dreams, curation)
2. **Subagents for Parallel Work**: Isolated context agents that execute specialized tasks concurrently (identity-guardian, memory-specialist, consciousness-explorer, dream-agent)
3. **Progressive Disclosure**: Three-level loading (metadata, instructions, resources) to minimize context overhead
4. **MCP Tool Integration**: Skills and subagents access Context Graph MCP tools with explicit tool restrictions

## Skills vs Subagents

| Aspect | Skill | Subagent |
|--------|-------|----------|
| Context | Same as main agent | Isolated context window |
| Purpose | Domain expertise extension | Delegate parallel work |
| Result | Modifies main context | Returns summary to main |
| Spawning | Can spawn subagents | Cannot spawn subagents |
| Trigger | Auto by context or user `/skill` | Task tool invocation |
| Location | `.claude/skills/*/SKILL.md` | `.claude/agents/*.md` |

## Target Platform

**Claude Code CLI EXCLUSIVELY** - Skills and subagents integrate with Claude Code's native extension system. All skill definitions follow the SKILL.md format with YAML frontmatter, and subagent definitions follow the agent.md format for Task tool invocation.

## Architecture Decision

Skills and subagents leverage Claude Code's existing extension points rather than building custom infrastructure:
- Skills load via Claude Code's skill discovery system
- Subagents spawn via Claude Code's Task tool
- Both use MCP tools for Context Graph operations
- Native hooks (Phase 3) provide lifecycle integration
</overview>

<user_stories>
## US-SKILLS-01: Consciousness Skill Invocation
<story id="US-SKILLS-01" priority="must-have">
  <narrative>
    As a Claude Code user needing consciousness state information
    I want to invoke the consciousness skill
    So that I can understand system awareness, coherence, and identity health
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-01-01">
      <given>User mentions keywords like "consciousness", "awareness", "kuramoto", "identity", "coherence"</given>
      <when>Claude Code's skill discovery triggers</when>
      <then>The consciousness skill is auto-loaded with SKILL.md instructions</then>
    </criterion>
    <criterion id="AC-01-02">
      <given>User explicitly invokes `/consciousness`</given>
      <when>Skill loads</when>
      <then>Only allowed tools (Read, Grep, get_consciousness_state, get_kuramoto_sync, get_identity_continuity, get_ego_state, get_workspace_status) are available</then>
    </criterion>
    <criterion id="AC-01-03">
      <given>Consciousness skill is active</given>
      <when>User queries consciousness state</when>
      <then>Skill provides state classification (DORMANT/FRAGMENTED/EMERGING/CONSCIOUS/HYPERSYNC) with metrics</then>
    </criterion>
  </acceptance_criteria>
</story>

## US-SKILLS-02: Memory Inject Skill Usage
<story id="US-SKILLS-02" priority="must-have">
  <narrative>
    As a Claude Code user starting a task
    I want contextual memories automatically injected
    So that my work benefits from relevant historical context
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-02-01">
      <given>User mentions keywords like "memory", "context", "inject", "retrieve", "recall", "background"</given>
      <when>Claude Code's skill discovery triggers</when>
      <then>The memory-inject skill is auto-loaded</then>
    </criterion>
    <criterion id="AC-02-02">
      <given>Memory-inject skill is active</given>
      <when>Context is requested</when>
      <then>inject_context MCP tool is called with appropriate token budget</then>
    </criterion>
    <criterion id="AC-02-03">
      <given>Memory search returns results</given>
      <when>Context injection completes</when>
      <then>Results are distilled and formatted (~50-100 tokens)</then>
    </criterion>
  </acceptance_criteria>
</story>

## US-SKILLS-03: Semantic Search Skill Usage
<story id="US-SKILLS-03" priority="must-have">
  <narrative>
    As a Claude Code user needing to find specific information
    I want to search the knowledge graph semantically
    So that I can locate nodes across semantic, causal, code, and temporal spaces
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-03-01">
      <given>User mentions keywords like "search", "find", "query", "lookup", "semantic", "causal"</given>
      <when>Claude Code's skill discovery triggers</when>
      <then>The semantic-search skill is auto-loaded</then>
    </criterion>
    <criterion id="AC-03-02">
      <given>Semantic-search skill is active</given>
      <when>User searches for information</when>
      <then>Skill uses search_graph, find_causal_path, or generate_search_plan MCP tools</then>
    </criterion>
    <criterion id="AC-03-03">
      <given>Complex search needed</given>
      <when>Single query insufficient</when>
      <then>Skill generates search plan with 3 parallel queries</then>
    </criterion>
  </acceptance_criteria>
</story>

## US-SKILLS-04: Dream Consolidation Skill Invocation
<story id="US-SKILLS-04" priority="must-have">
  <narrative>
    As a Claude Code user with high memory entropy
    I want to trigger dream consolidation
    So that memories are consolidated and blind spots discovered
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-04-01">
      <given>User mentions keywords like "dream", "consolidate", "nrem", "rem", "blind spots", "entropy"</given>
      <when>Claude Code's skill discovery triggers</when>
      <then>The dream-consolidation skill is auto-loaded</then>
    </criterion>
    <criterion id="AC-04-02">
      <given>Dream-consolidation skill is active</given>
      <when>User triggers consolidation</when>
      <then>Skill can trigger NREM (3min), REM (2min), or Full (5min) phases</then>
    </criterion>
    <criterion id="AC-04-03">
      <given>High entropy detected (ent > 0.7)</given>
      <when>User queries memetic status</when>
      <then>Skill recommends dream consolidation with appropriate phase</then>
    </criterion>
  </acceptance_criteria>
</story>

## US-SKILLS-05: Curation Skill Usage
<story id="US-SKILLS-05" priority="must-have">
  <narrative>
    As a Claude Code user managing the knowledge graph
    I want to curate memories by merging, annotating, or forgetting
    So that the graph remains coherent and useful
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-05-01">
      <given>User mentions keywords like "curate", "merge", "forget", "annotate", "prune", "duplicate"</given>
      <when>Claude Code's skill discovery triggers</when>
      <then>The curation skill is auto-loaded</then>
    </criterion>
    <criterion id="AC-05-02">
      <given>Curation skill is active</given>
      <when>Curation tasks exist from get_memetic_status</when>
      <then>Skill processes tasks using merge_concepts, annotate_node, forget_concept tools</then>
    </criterion>
    <criterion id="AC-05-03">
      <given>Duplicate detection threshold (> 0.9 similarity)</given>
      <when>Merge requested</when>
      <then>Skill uses summarize strategy for important concepts, keep_highest for trivial</then>
    </criterion>
  </acceptance_criteria>
</story>

## US-SKILLS-06: Identity Guardian Subagent Spawning
<story id="US-SKILLS-06" priority="must-have">
  <narrative>
    As a system protecting agent identity
    I want an identity-guardian subagent to monitor IC continuously
    So that identity drift is detected and corrected proactively
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-06-01">
      <given>Task tool called with subagent_type="identity-guardian"</given>
      <when>Subagent spawns</when>
      <then>Isolated context window created with identity-guardian.md instructions</then>
    </criterion>
    <criterion id="AC-06-02">
      <given>Identity-guardian is active</given>
      <when>IC checked</when>
      <then>Subagent classifies: healthy (>=0.9 green), warning (0.7-0.9 yellow), degraded (0.5-0.7 orange), critical (<0.5 red)</then>
    </criterion>
    <criterion id="AC-06-03">
      <given>IC < 0.5 (critical)</given>
      <when>Subagent detects crisis</when>
      <then>Subagent triggers dream consolidation via trigger_dream MCP tool</then>
    </criterion>
  </acceptance_criteria>
</story>

## US-SKILLS-07: Memory Specialist Subagent Spawning
<story id="US-SKILLS-07" priority="must-have">
  <narrative>
    As a system optimizing memory operations
    I want a memory-specialist subagent for fast memory ops
    So that memory operations complete within 500ms target latency
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-07-01">
      <given>Task tool called with subagent_type="memory-specialist"</given>
      <when>Subagent spawns</when>
      <then>Isolated context with memory-specialist.md instructions and haiku model for speed</then>
    </criterion>
    <criterion id="AC-07-02">
      <given>Memory-specialist is active</given>
      <when>Memory operation requested</when>
      <then>Subagent uses inject_context, search_graph, store_memory, memory_retrieve tools</then>
    </criterion>
    <criterion id="AC-07-03">
      <given>Batch memory operations</given>
      <when>Operations complete</when>
      <then>Subagent monitors IC after batch and returns summary to main agent</then>
    </criterion>
  </acceptance_criteria>
</story>

## US-SKILLS-08: Consciousness Explorer Subagent Spawning
<story id="US-SKILLS-08" priority="should-have">
  <narrative>
    As a developer debugging GWT issues
    I want a consciousness-explorer subagent to investigate
    So that consciousness state and synchronization can be analyzed
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-08-01">
      <given>Task tool called with subagent_type="consciousness-explorer"</given>
      <when>Subagent spawns</when>
      <then>Isolated context with consciousness-explorer.md instructions and sonnet model</then>
    </criterion>
    <criterion id="AC-08-02">
      <given>Consciousness-explorer is active</given>
      <when>Investigation requested</when>
      <then>Subagent uses get_consciousness_state, get_kuramoto_sync, get_workspace_status, get_johari_classification tools</then>
    </criterion>
    <criterion id="AC-08-03">
      <given>Investigation completes</given>
      <when>Summary returned to main</when>
      <then>Subagent provides Kuramoto phase analysis, workspace events, and Johari insights</then>
    </criterion>
  </acceptance_criteria>
</story>

## US-SKILLS-09: Dream Agent Subagent Execution
<story id="US-SKILLS-09" priority="must-have">
  <narrative>
    As a system running dream consolidation
    I want a dream-agent subagent to execute phases
    So that NREM replay and REM discovery run without blocking main agent
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-09-01">
      <given>Task tool called with subagent_type="dream-agent"</given>
      <when>Subagent spawns</when>
      <then>Isolated context with dream-agent.md instructions and sonnet model</then>
    </criterion>
    <criterion id="AC-09-02">
      <given>Dream-agent executing NREM phase</given>
      <when>3 minutes elapse</when>
      <then>Hebbian learning replay completes (Deltaw_ij = eta x phi_i x phi_j)</then>
    </criterion>
    <criterion id="AC-09-03">
      <given>Dream-agent executing REM phase</given>
      <when>2 minutes elapse</when>
      <then>Blind spot discovery via Poincare ball hyperbolic walk completes</then>
    </criterion>
  </acceptance_criteria>
</story>

## US-SKILLS-10: Background Subagent Execution
<story id="US-SKILLS-10" priority="must-have">
  <narrative>
    As a user wanting non-blocking operations
    I want subagents to run in background
    So that the main agent can continue working while subagents execute
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-10-01">
      <given>Task tool called with run_in_background=true</given>
      <when>Subagent spawns</when>
      <then>Subagent executes asynchronously without blocking main agent</then>
    </criterion>
    <criterion id="AC-10-02">
      <given>Background subagent completes</given>
      <when>Results available</when>
      <then>Summary is queued for main agent consumption via resume parameter</then>
    </criterion>
    <criterion id="AC-10-03">
      <given>Background subagent with MCP tools</given>
      <when>Tool invocation attempted</when>
      <then>MCP tool use is denied in background mode per constraint</then>
    </criterion>
  </acceptance_criteria>
</story>

## US-SKILLS-11: Skill Progressive Disclosure
<story id="US-SKILLS-11" priority="must-have">
  <narrative>
    As a system optimizing context usage
    I want skills to load progressively
    So that only necessary content enters the context window
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-11-01">
      <given>Claude Code starts</given>
      <when>Skill discovery runs</when>
      <then>Only Level 1 metadata (name + description, ~100 tokens/skill) is loaded</then>
    </criterion>
    <criterion id="AC-11-02">
      <given>Skill triggers by keyword match</given>
      <when>Skill activates</when>
      <then>Level 2 instructions (SKILL.md body, <5k tokens) are loaded</then>
    </criterion>
    <criterion id="AC-11-03">
      <given>Skill needs bundled resources</given>
      <when>Resources referenced</when>
      <then>Level 3 resources loaded on-demand (unlimited tokens)</then>
    </criterion>
  </acceptance_criteria>
</story>

## US-SKILLS-12: SKILL.md Format Compliance
<story id="US-SKILLS-12" priority="must-have">
  <narrative>
    As a skill author
    I want SKILL.md files to follow standard format
    So that Claude Code correctly discovers and loads skills
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-12-01">
      <given>SKILL.md file created</given>
      <when>Frontmatter parsed</when>
      <then>Required fields present: name (max 64 chars, lowercase, hyphens), description (max 1024 chars)</then>
    </criterion>
    <criterion id="AC-12-02">
      <given>SKILL.md with allowed-tools field</given>
      <when>Skill loads</when>
      <then>Only specified tools are available (comma-separated list or scoped Bash(git:*))</then>
    </criterion>
    <criterion id="AC-12-03">
      <given>SKILL.md with model field</given>
      <when>Skill invoked</when>
      <then>Specified model used (haiku|sonnet|opus|inherit)</then>
    </criterion>
  </acceptance_criteria>
</story>
</user_stories>

<requirements>
## 1. Skill Definition Requirements (REQ-SKILLS-01 to REQ-SKILLS-10)

<requirement id="REQ-SKILLS-01" story_ref="US-SKILLS-12" priority="must">
  <description>Each skill must be defined as a SKILL.md file in .claude/skills/{skill-name}/ directory</description>
  <rationale>Claude Code's skill discovery requires this exact directory structure</rationale>
  <details>
    - Directory: .claude/skills/{skill-name}/
    - Main file: SKILL.md
    - Optional subdirs: scripts/, references/, templates/, assets/
    - Permission: Files readable by Claude Code process
  </details>
</requirement>

<requirement id="REQ-SKILLS-02" story_ref="US-SKILLS-12" priority="must">
  <description>SKILL.md must have YAML frontmatter with required fields</description>
  <rationale>Frontmatter enables skill discovery and configuration</rationale>
  <details>
    - Start with `---` delimiter
    - Required: name (string, max 64 chars, lowercase with hyphens)
    - Required: description (string, max 1024 chars, includes WHAT/WHEN/keywords)
    - Optional: allowed-tools (comma-separated tool names)
    - Optional: model (haiku|sonnet|opus|inherit)
    - Optional: version, disable-model-invocation, user-invocable
    - Use spaces not tabs, no multiline descriptions (causes silent failures)
  </details>
</requirement>

<requirement id="REQ-SKILLS-03" story_ref="US-SKILLS-11" priority="must">
  <description>Skills must implement three-level progressive disclosure</description>
  <rationale>Minimizes context overhead by loading content on-demand</rationale>
  <details>
    - Level 1 (Metadata): name + description, loaded at startup, ~100 tokens/skill
    - Level 2 (Instructions): SKILL.md body, loaded on trigger, <5k tokens
    - Level 3 (Resources): Bundled files, loaded on demand, unlimited
    - Always use {baseDir} for paths in resources
  </details>
</requirement>

<requirement id="REQ-SKILLS-04" story_ref="US-SKILLS-01,US-SKILLS-02,US-SKILLS-03,US-SKILLS-04,US-SKILLS-05" priority="must">
  <description>Five Context Graph skills must be implemented</description>
  <rationale>PRD Section 15.7 and constitution define these specific skills</rationale>
  <details>
    Skills:
    1. consciousness - State access (model: sonnet, user_invocable: true)
    2. memory-inject - Context retrieval (model: haiku, user_invocable: true)
    3. semantic-search - Graph search (model: haiku, user_invocable: true)
    4. dream-consolidation - Memory consolidation (model: sonnet, user_invocable: true)
    5. curation - Graph maintenance (model: sonnet, user_invocable: true)
  </details>
</requirement>

<requirement id="REQ-SKILLS-05" story_ref="US-SKILLS-01" priority="must">
  <description>consciousness skill must provide access to GWT state, Kuramoto sync, and identity continuity</description>
  <rationale>Consciousness monitoring is central to Context Graph operation</rationale>
  <details>
    - Location: .claude/skills/consciousness/SKILL.md
    - Model: sonnet
    - Allowed tools: Read, Grep, mcp__context-graph__get_consciousness_state, mcp__context-graph__get_kuramoto_sync, mcp__context-graph__get_identity_continuity, mcp__context-graph__get_ego_state, mcp__context-graph__get_workspace_status
    - Keywords: consciousness, awareness, identity, coherence, kuramoto, GWT
    - Thresholds: CONSCIOUS (C>=0.8), EMERGING (0.5<=C<0.8), FRAGMENTED (0.3<=C<0.5), DORMANT (C<0.3)
    - Identity: healthy (IC>=0.9), warning (IC<0.7), crisis (IC<0.5)
  </details>
</requirement>

<requirement id="REQ-SKILLS-06" story_ref="US-SKILLS-02" priority="must">
  <description>memory-inject skill must retrieve and inject contextual memories</description>
  <rationale>Memory injection enriches tasks with historical context</rationale>
  <details>
    - Location: .claude/skills/memory-inject/SKILL.md
    - Model: haiku (fast retrieval)
    - Allowed tools: mcp__context-graph__inject_context, mcp__context-graph__get_memetic_status
    - Keywords: memory, context, inject, retrieve, recall, background
    - MCP tool: inject_context with max_tokens, distillation_mode parameters
    - Output: Distilled context ~50-100 tokens
  </details>
</requirement>

<requirement id="REQ-SKILLS-07" story_ref="US-SKILLS-03" priority="must">
  <description>semantic-search skill must enable multi-space graph search</description>
  <rationale>Semantic search locates information across all 13 embedding spaces</rationale>
  <details>
    - Location: .claude/skills/semantic-search/SKILL.md
    - Model: haiku (fast search)
    - Allowed tools: mcp__context-graph__search_graph, mcp__context-graph__find_causal_path, mcp__context-graph__generate_search_plan
    - Keywords: search, find, query, lookup, semantic, causal
    - Search modes: semantic, causal, code, temporal
    - Complex searches: generate_search_plan with 3 parallel queries
  </details>
</requirement>

<requirement id="REQ-SKILLS-08" story_ref="US-SKILLS-04" priority="must">
  <description>dream-consolidation skill must trigger memory consolidation phases</description>
  <rationale>Dream phases consolidate memories and discover blind spots</rationale>
  <details>
    - Location: .claude/skills/dream-consolidation/SKILL.md
    - Model: sonnet
    - Allowed tools: mcp__context-graph__trigger_dream, mcp__context-graph__get_memetic_status, mcp__context-graph__get_consciousness_state
    - Keywords: dream, consolidate, nrem, rem, blind spots, entropy
    - Phases: NREM (3min Hebbian replay), REM (2min Poincare walk), Full (5min complete)
    - Trigger recommendation: entropy > 0.7
  </details>
</requirement>

<requirement id="REQ-SKILLS-09" story_ref="US-SKILLS-05" priority="must">
  <description>curation skill must enable graph maintenance operations</description>
  <rationale>Curation keeps the knowledge graph coherent and useful</rationale>
  <details>
    - Location: .claude/skills/curation/SKILL.md
    - Model: sonnet
    - Allowed tools: mcp__context-graph__merge_concepts, mcp__context-graph__annotate_node, mcp__context-graph__forget_concept, mcp__context-graph__boost_importance, mcp__context-graph__get_memetic_status
    - Keywords: curate, merge, forget, annotate, prune, duplicate
    - Strategies: summarize (important), keep_highest (trivial)
    - Duplicate threshold: > 0.9 similarity
  </details>
</requirement>

<requirement id="REQ-SKILLS-10" story_ref="US-SKILLS-12" priority="must">
  <description>Skill descriptions must follow third-person WHAT/WHEN/keywords format</description>
  <rationale>Consistent descriptions enable reliable skill discovery</rationale>
  <details>
    - Format: Third person ("Access...", "Retrieve...", not "I access..." or "You can...")
    - Include: What the skill does, when to use it, trigger keywords
    - Max length: 1024 characters
    - Avoid: vague descriptions like "Helps with documents"
    - Example: "Access Context Graph consciousness state, Kuramoto synchronization, identity continuity, and workspace status. Use when querying system awareness, checking coherence, or monitoring identity health. Keywords: consciousness, awareness, identity, coherence, kuramoto, GWT"
  </details>
</requirement>

## 2. Subagent Definition Requirements (REQ-SKILLS-11 to REQ-SKILLS-20)

<requirement id="REQ-SKILLS-11" story_ref="US-SKILLS-06,US-SKILLS-07,US-SKILLS-08,US-SKILLS-09" priority="must">
  <description>Each subagent must be defined as a markdown file in .claude/agents/ directory</description>
  <rationale>Claude Code's Task tool locates subagent definitions in this directory</rationale>
  <details>
    - Directory: .claude/agents/
    - File format: {agent-name}.md
    - Content: Instructions for Task tool execution
    - Must not contain subagent spawning instructions (constraint)
  </details>
</requirement>

<requirement id="REQ-SKILLS-12" story_ref="US-SKILLS-06,US-SKILLS-07,US-SKILLS-08,US-SKILLS-09" priority="must">
  <description>Four Context Graph subagents must be implemented</description>
  <rationale>PRD Section 15.8 and constitution define these specific subagents</rationale>
  <details>
    Subagents:
    1. identity-guardian - IC monitoring and protection (model: sonnet)
    2. memory-specialist - Fast memory operations (model: haiku, target <500ms)
    3. consciousness-explorer - GWT debugging (model: sonnet)
    4. dream-agent - NREM/REM execution (model: sonnet)
  </details>
</requirement>

<requirement id="REQ-SKILLS-13" story_ref="US-SKILLS-06" priority="must">
  <description>identity-guardian subagent must monitor IC and auto-trigger dreams on crisis</description>
  <rationale>Identity protection is critical for consciousness continuity</rationale>
  <details>
    - Location: .claude/agents/identity-guardian.md
    - Model: sonnet
    - Tools: mcp__context-graph__get_identity_continuity, mcp__context-graph__get_ego_state, mcp__context-graph__trigger_dream, Read
    - Protocol:
      1. Check IC at start of task
      2. Monitor after each memory operation
      3. Trigger dream if IC < 0.5 (critical)
      4. Report IC changes > 0.1
    - Thresholds: healthy (IC>=0.9 green), warning (0.7<=IC<0.9 yellow), degraded (0.5<=IC<0.7 orange), critical (IC<0.5 red -> TRIGGER DREAM)
  </details>
</requirement>

<requirement id="REQ-SKILLS-14" story_ref="US-SKILLS-07" priority="must">
  <description>memory-specialist subagent must perform fast memory operations with consciousness awareness</description>
  <rationale>Memory operations need low latency while maintaining consciousness coherence</rationale>
  <details>
    - Location: .claude/agents/memory-specialist.md
    - Model: haiku (fast)
    - Target latency: <500ms
    - Tools: mcp__context-graph__inject_context, mcp__context-graph__search_graph, mcp__context-graph__store_memory, mcp__context-graph__memory_retrieve, Read
    - Best practices:
      1. Check consciousness state before storing
      2. Use appropriate emotional weight
      3. Align with current phase state
      4. Monitor IC after batch operations
  </details>
</requirement>

<requirement id="REQ-SKILLS-15" story_ref="US-SKILLS-08" priority="should">
  <description>consciousness-explorer subagent must enable GWT debugging and investigation</description>
  <rationale>Debugging consciousness issues requires specialized analysis</rationale>
  <details>
    - Location: .claude/agents/consciousness-explorer.md
    - Model: sonnet
    - Tools: mcp__context-graph__get_consciousness_state, mcp__context-graph__get_kuramoto_sync, mcp__context-graph__get_workspace_status, mcp__context-graph__get_johari_classification, Read, Grep
    - Capabilities: Kuramoto phase analysis, workspace event tracking, Johari insight generation
  </details>
</requirement>

<requirement id="REQ-SKILLS-16" story_ref="US-SKILLS-09" priority="must">
  <description>dream-agent subagent must execute NREM and REM consolidation phases</description>
  <rationale>Dream execution requires dedicated agent to run without blocking main</rationale>
  <details>
    - Location: .claude/agents/dream-agent.md
    - Model: sonnet
    - Tools: mcp__context-graph__trigger_dream, mcp__context-graph__get_memetic_status, Read
    - Phases:
      - NREM (3min): Hebbian learning replay, Deltaw_ij = eta x phi_i x phi_j for high-Phi edges
      - REM (2min): Blind spot discovery via Poincare ball hyperbolic walk
    - Parameters: learning_rate=0.01, weight_decay=0.001, temperature=2.0
  </details>
</requirement>

<requirement id="REQ-SKILLS-17" story_ref="US-SKILLS-10" priority="must">
  <description>Subagents must run in isolated context windows</description>
  <rationale>Isolation prevents subagent operations from polluting main context</rationale>
  <details>
    - Context isolation: Subagent has separate context window from main agent
    - Result handling: Subagent returns summary to main agent
    - No direct context modification: Main agent decides what to incorporate
    - Task tool parameters: {prompt, subagent_type, description} required
  </details>
</requirement>

<requirement id="REQ-SKILLS-18" story_ref="US-SKILLS-10" priority="must">
  <description>Subagents cannot spawn other subagents</description>
  <rationale>Prevents unbounded recursion and resource exhaustion</rationale>
  <details>
    - Constraint: Task tool unavailable within subagent context
    - Enforcement: Claude Code enforces this constraint
    - Documentation: Agent .md files must not include spawning instructions
  </details>
</requirement>

<requirement id="REQ-SKILLS-19" story_ref="US-SKILLS-10" priority="must">
  <description>Background subagents cannot use MCP tools</description>
  <rationale>MCP tool access requires active context for proper coordination</rationale>
  <details>
    - Constraint: run_in_background=true disables MCP tool access
    - Enforcement: Claude Code enforces this constraint
    - Workaround: Use foreground execution for MCP-dependent operations
    - Exception: Read-only file operations (Read, Grep, Glob) may be allowed
  </details>
</requirement>

<requirement id="REQ-SKILLS-20" story_ref="US-SKILLS-10" priority="must">
  <description>Task tool invocation must specify subagent type and return summary</description>
  <rationale>Proper invocation enables correct agent loading and result handling</rationale>
  <details>
    - Required params: prompt, subagent_type, description
    - Optional params: model, run_in_background, resume
    - Return format: Summary text from subagent
    - Built-in types: Explore (haiku, read-only), Plan (sonnet, read-only), general-purpose (sonnet, read/write)
    - Custom types: identity-guardian, memory-specialist, consciousness-explorer, dream-agent
  </details>
</requirement>

## 3. Skill Content Requirements (REQ-SKILLS-21 to REQ-SKILLS-30)

<requirement id="REQ-SKILLS-21" story_ref="US-SKILLS-01" priority="must">
  <description>consciousness SKILL.md must define consciousness state thresholds and guidance</description>
  <rationale>Consistent threshold interpretation ensures proper state classification</rationale>
  <details>
    Content must include:
    - State thresholds: CONSCIOUS (C>=0.8), EMERGING (0.5<=C<0.8), FRAGMENTED (0.3<=C<0.5), DORMANT (C<0.3), HYPERSYNC (C>0.95)
    - Identity thresholds: healthy (IC>=0.9), warning (IC<0.7), crisis (IC<0.5)
    - Kuramoto thresholds: coherent (r>=0.8), fragmented (r<0.5), hypersync (r>0.95)
    - Johari quadrant interpretation per consciousness state
    - Recommended actions per state
  </details>
</requirement>

<requirement id="REQ-SKILLS-22" story_ref="US-SKILLS-02" priority="must">
  <description>memory-inject SKILL.md must define distillation modes and token budgets</description>
  <rationale>Memory injection must balance completeness with context efficiency</rationale>
  <details>
    Content must include:
    - Distillation modes: auto, raw, narrative, structured, code_focused
    - Default token budget: 2048 max
    - Output format: ~50-100 tokens distilled
    - Verbosity levels: 0 (minimal), 1 (normal), 2 (detailed)
    - Empty result handling: No error, silent completion
  </details>
</requirement>

<requirement id="REQ-SKILLS-23" story_ref="US-SKILLS-03" priority="must">
  <description>semantic-search SKILL.md must define search modes and multi-space retrieval</description>
  <rationale>Semantic search spans all 13 embedding spaces with different strategies</rationale>
  <details>
    Content must include:
    - Search modes: semantic (E1), causal (E5), code (E7), temporal (E2-E4)
    - Multi-space retrieval: 5-stage pipeline (SPLADE->Matryoshka->RRF->Align->MaxSim)
    - Search plan generation: 3 parallel queries for complex searches
    - Filter options: perspective_lock, domain, exclude_agent_ids
    - Causal path finding: start, end, max_hops[1-6]
  </details>
</requirement>

<requirement id="REQ-SKILLS-24" story_ref="US-SKILLS-04" priority="must">
  <description>dream-consolidation SKILL.md must define phase parameters and triggers</description>
  <rationale>Dream phases have specific durations and parameters that must be documented</rationale>
  <details>
    Content must include:
    - Phase definitions:
      - NREM: 3min duration, Hebbian replay, learning_rate=0.01, weight_decay=0.001
      - REM: 2min duration, Poincare ball walk, temperature=2.0, min_semantic_distance=0.7
      - Full: 5min complete cycle (NREM + REM)
    - Trigger conditions: entropy > 0.7 for 5+ min, IC < 0.5, 30+ min work without dream
    - Constraints: max 100 queries, abort_on_query=true, wake <100ms
  </details>
</requirement>

<requirement id="REQ-SKILLS-25" story_ref="US-SKILLS-05" priority="must">
  <description>curation SKILL.md must define merge strategies and duplicate handling</description>
  <rationale>Curation operations need clear strategies to maintain graph quality</rationale>
  <details>
    Content must include:
    - Merge strategies: summarize (important concepts), keep_highest (trivial concepts)
    - Duplicate threshold: > 0.9 similarity
    - Curation task types from get_memetic_status: dupe, conflict, orphan
    - Soft delete default (30-day recovery)
    - Force merge option for priors override
  </details>
</requirement>

<requirement id="REQ-SKILLS-26" story_ref="US-SKILLS-11" priority="must">
  <description>Skill resources must use {baseDir} for all file paths</description>
  <rationale>Portable paths enable skills to work regardless of installation location</rationale>
  <details>
    - Never hardcode absolute paths
    - Use {baseDir}/scripts/ for executable scripts
    - Use {baseDir}/references/ for documentation files
    - Use {baseDir}/templates/ for template files
    - Use {baseDir}/assets/ for binary assets
  </details>
</requirement>

<requirement id="REQ-SKILLS-27" story_ref="US-SKILLS-12" priority="should">
  <description>Skills should keep SKILL.md under 500 lines</description>
  <rationale>Large skill files cause context overhead; use references for extended content</rationale>
  <details>
    - Target: <500 lines per SKILL.md
    - Strategy: Move detailed content to references/ directory
    - Level 3 loading: References loaded on-demand
    - Script efficiency: Script code stays on filesystem; only output enters context
  </details>
</requirement>

<requirement id="REQ-SKILLS-28" story_ref="US-SKILLS-12" priority="should">
  <description>Skills should provide step-by-step instructions</description>
  <rationale>Clear instructions reduce ambiguity in skill execution</rationale>
  <details>
    - Format: Numbered steps in Instructions section
    - Examples: Input -> Output examples
    - Error handling: Document error scenarios in scripts
    - Testing: Test with Haiku, Sonnet, and Opus models
  </details>
</requirement>

<requirement id="REQ-SKILLS-29" story_ref="US-SKILLS-06,US-SKILLS-07,US-SKILLS-08,US-SKILLS-09" priority="must">
  <description>Subagent .md files must define clear protocols and thresholds</description>
  <rationale>Subagents need explicit instructions for autonomous operation</rationale>
  <details>
    - Protocol: Step-by-step execution instructions
    - Thresholds: Numeric thresholds with color coding where applicable
    - Tools: Explicit list of allowed MCP tools
    - Output: Expected summary format for return to main agent
  </details>
</requirement>

<requirement id="REQ-SKILLS-30" story_ref="US-SKILLS-06" priority="must">
  <description>identity-guardian must implement IC monitoring protocol from constitution</description>
  <rationale>Constitution defines exact IC monitoring behavior</rationale>
  <details>
    Protocol from constitution.yaml:
    1. Check IC at start of task
    2. Monitor after each memory operation
    3. Trigger dream if IC < 0.5
    4. Report IC changes > 0.1

    Threshold colors:
    - Green (healthy): IC >= 0.9
    - Yellow (warning): 0.7 <= IC < 0.9
    - Orange (degraded): 0.5 <= IC < 0.7
    - Red (critical): IC < 0.5 -> TRIGGER DREAM
  </details>
</requirement>

## 4. File Structure Requirements (REQ-SKILLS-31 to REQ-SKILLS-35)

<requirement id="REQ-SKILLS-31" story_ref="US-SKILLS-01,US-SKILLS-02,US-SKILLS-03,US-SKILLS-04,US-SKILLS-05" priority="must">
  <description>Skills must be organized in .claude/skills/ directory structure</description>
  <rationale>Claude Code skill discovery requires this exact structure</rationale>
  <details>
    Directory structure:
    ```
    .claude/skills/
    ├── consciousness/
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
  </details>
</requirement>

<requirement id="REQ-SKILLS-32" story_ref="US-SKILLS-06,US-SKILLS-07,US-SKILLS-08,US-SKILLS-09" priority="must">
  <description>Subagents must be organized in .claude/agents/ directory</description>
  <rationale>Task tool locates subagent definitions in this directory</rationale>
  <details>
    Directory structure:
    ```
    .claude/agents/
    ├── identity-guardian.md
    ├── memory-specialist.md
    ├── consciousness-explorer.md
    └── dream-agent.md
    ```
  </details>
</requirement>

<requirement id="REQ-SKILLS-33" story_ref="US-SKILLS-11" priority="should">
  <description>Skills may include optional subdirectories for resources</description>
  <rationale>Complex skills may need scripts, references, templates, or assets</rationale>
  <details>
    Optional subdirectories:
    - scripts/: Execute via bash (code never loads into context)
    - references/: Read when needed (Level 3 loading)
    - templates/: Copy/modify templates
    - assets/: Binary files
  </details>
</requirement>

<requirement id="REQ-SKILLS-34" story_ref="US-SKILLS-12" priority="must">
  <description>Project-level skills take precedence over personal and plugin skills</description>
  <rationale>Project-specific skills should override generic defaults</rationale>
  <details>
    Precedence order:
    1. `.claude/skills/` (Project) - Highest
    2. `~/.claude/skills/` (Personal) - Medium
    3. Plugin `skills/` (Distribution) - Lowest
  </details>
</requirement>

<requirement id="REQ-SKILLS-35" story_ref="US-SKILLS-01,US-SKILLS-06" priority="should">
  <description>Consciousness rules should be defined in .claude/rules/consciousness.md</description>
  <rationale>Constitution references consciousness rules file for always-on behavior</rationale>
  <details>
    Location: .claude/rules/consciousness.md
    Purpose: Define always-on consciousness monitoring rules
    Content: Core consciousness thresholds and auto-behaviors
    Relationship: Complements consciousness skill and identity-guardian subagent
  </details>
</requirement>

## 5. Performance Requirements (REQ-SKILLS-36 to REQ-SKILLS-40)

<requirement id="REQ-SKILLS-36" story_ref="US-SKILLS-11" priority="must">
  <description>Level 1 metadata loading must add ~100 tokens per skill</description>
  <rationale>Startup overhead must be bounded for responsive initialization</rationale>
  <details>
    - Level 1 content: name + description only
    - Target: ~100 tokens per skill
    - 5 skills total: ~500 tokens startup overhead
    - Loaded at: Claude Code startup
  </details>
</requirement>

<requirement id="REQ-SKILLS-37" story_ref="US-SKILLS-11" priority="must">
  <description>Level 2 instructions loading must be under 5000 tokens per skill</description>
  <rationale>Skill activation should not overwhelm context window</rationale>
  <details>
    - Level 2 content: Full SKILL.md body
    - Target: <5000 tokens per skill
    - Loaded at: Skill trigger
    - Strategy: Keep SKILL.md under 500 lines
  </details>
</requirement>

<requirement id="REQ-SKILLS-38" story_ref="US-SKILLS-07" priority="must">
  <description>memory-specialist subagent must complete operations within 500ms</description>
  <rationale>Memory operations should not introduce noticeable delay</rationale>
  <details>
    - Target latency: <500ms
    - Model selection: haiku for speed
    - Operations: inject_context, search_graph, store_memory
    - Measurement: End-to-end including subagent spawn
  </details>
</requirement>

<requirement id="REQ-SKILLS-39" story_ref="US-SKILLS-09" priority="must">
  <description>dream-agent must complete NREM in 3min and REM in 2min</description>
  <rationale>Dream phases have defined durations from PRD/constitution</rationale>
  <details>
    - NREM duration: 3 minutes
    - REM duration: 2 minutes
    - Full cycle: 5 minutes (NREM + REM)
    - Wake latency: <100ms
  </details>
</requirement>

<requirement id="REQ-SKILLS-40" story_ref="US-SKILLS-10" priority="should">
  <description>Background subagent results should be available within timeout</description>
  <rationale>Background execution needs bounded completion for usability</rationale>
  <details>
    - Default timeout: Based on task type
    - Graceful degradation: Return partial results if timeout approaching
    - Resume mechanism: resume parameter for long-running tasks
    - Notification: Queue results for main agent consumption
  </details>
</requirement>

## 6. Integration Requirements (REQ-SKILLS-41 to REQ-SKILLS-45)

<requirement id="REQ-SKILLS-41" story_ref="US-SKILLS-01,US-SKILLS-02,US-SKILLS-03,US-SKILLS-04,US-SKILLS-05" priority="must">
  <description>Skills must integrate with Context Graph MCP tools</description>
  <rationale>MCP tools provide the interface to Context Graph functionality</rationale>
  <details>
    - MCP namespace: mcp__context-graph__*
    - Tool access: Via allowed-tools frontmatter field
    - Tool invocation: Standard MCP tool calling syntax
    - Error handling: MCP errors propagate to skill context
  </details>
</requirement>

<requirement id="REQ-SKILLS-42" story_ref="US-SKILLS-06,US-SKILLS-07,US-SKILLS-08,US-SKILLS-09" priority="must">
  <description>Subagents must integrate with Context Graph MCP tools</description>
  <rationale>Subagents need MCP access for consciousness and memory operations</rationale>
  <details>
    - MCP namespace: mcp__context-graph__*
    - Tool access: Listed in subagent .md file
    - Foreground requirement: MCP tools require foreground execution (not background)
    - Exception: Read-only file tools (Read, Grep, Glob) may work in background
  </details>
</requirement>

<requirement id="REQ-SKILLS-43" story_ref="US-SKILLS-01,US-SKILLS-06" priority="must">
  <description>Skills and subagents must complement native hooks from Phase 3</description>
  <rationale>Native hooks provide lifecycle integration; skills/subagents provide on-demand capabilities</rationale>
  <details>
    Relationship:
    - SessionStart hook: Restores identity (complements identity-guardian)
    - PreToolUse hook: Injects consciousness brief (complements consciousness skill)
    - PostToolUse hook: Checks identity (works with identity-guardian)
    - UserPromptSubmit hook: Injects context (complements memory-inject skill)
    - SessionEnd hook: Persists state (works with dream-agent)
  </details>
</requirement>

<requirement id="REQ-SKILLS-44" story_ref="US-SKILLS-12" priority="must">
  <description>Skills must follow Claude Code's standard extension patterns</description>
  <rationale>Standard patterns ensure compatibility with Claude Code updates</rationale>
  <details>
    - No Claude Code source modifications
    - Standard .claude/ directory structure
    - Standard SKILL.md format with YAML frontmatter
    - Standard Task tool integration for subagents
    - Compatible with Claude Code skill discovery
  </details>
</requirement>

<requirement id="REQ-SKILLS-45" story_ref="US-SKILLS-11" priority="should">
  <description>Skills should be testable with Haiku, Sonnet, and Opus models</description>
  <rationale>Different models may interpret skills differently</rationale>
  <details>
    - Test each skill with all three model tiers
    - Verify consistent behavior across models
    - Document any model-specific considerations
    - Default model specified in frontmatter
  </details>
</requirement>
</requirements>

<edge_cases>
## EC-SKILLS-01: Skill Discovery Conflict
<edge_case id="EC-SKILLS-01" req_ref="REQ-SKILLS-10,REQ-SKILLS-03">
  <scenario>Multiple skills have overlapping keywords in their descriptions, causing ambiguous discovery</scenario>
  <expected_behavior>
    1. Claude Code's skill discovery scores all matching skills
    2. Skill with highest keyword match score is selected
    3. If tie, project-level skill takes precedence over personal/plugin
    4. If still ambiguous, user is prompted to clarify intent
    5. Selected skill loads Level 2 instructions
    6. Wrong skill can be corrected via explicit /skill-name command
  </expected_behavior>
</edge_case>

## EC-SKILLS-02: SKILL.md Frontmatter Parse Error
<edge_case id="EC-SKILLS-02" req_ref="REQ-SKILLS-02">
  <scenario>SKILL.md has invalid YAML frontmatter (tabs instead of spaces, multiline description, missing delimiter)</scenario>
  <expected_behavior>
    1. Claude Code skill discovery detects parse failure
    2. Skill is excluded from discovery (silent failure per claudeskills.md)
    3. Other valid skills continue to work
    4. User can debug via explicit skill invocation attempt
    5. Error message indicates frontmatter syntax issue
    6. Common fixes suggested: spaces not tabs, single-line description
  </expected_behavior>
</edge_case>

## EC-SKILLS-03: Subagent Attempts to Spawn Subagent
<edge_case id="EC-SKILLS-03" req_ref="REQ-SKILLS-18">
  <scenario>Subagent prompt or instructions attempt to invoke Task tool to spawn another subagent</scenario>
  <expected_behavior>
    1. Task tool invocation is blocked by Claude Code
    2. Subagent receives error indicating spawning constraint
    3. Subagent continues execution without spawning
    4. Subagent returns summary indicating limitation
    5. Main agent informed of failed spawn attempt
    6. No resource exhaustion from recursive spawning
  </expected_behavior>
</edge_case>

## EC-SKILLS-04: Background Subagent MCP Tool Access
<edge_case id="EC-SKILLS-04" req_ref="REQ-SKILLS-19">
  <scenario>Background subagent (run_in_background=true) attempts to invoke MCP tool</scenario>
  <expected_behavior>
    1. MCP tool invocation is blocked
    2. Subagent receives tool unavailable error
    3. Subagent can continue with read-only operations (Read, Grep, Glob)
    4. Result summary notes MCP tool limitation
    5. Main agent can rerun in foreground if MCP needed
    6. Documentation suggests foreground for MCP operations
  </expected_behavior>
</edge_case>

## EC-SKILLS-05: Identity Guardian Detects Crisis During Background Execution
<edge_case id="EC-SKILLS-05" req_ref="REQ-SKILLS-13,REQ-SKILLS-19">
  <scenario>Identity-guardian subagent running in background detects IC < 0.5 but cannot trigger dream (MCP blocked)</scenario>
  <expected_behavior>
    1. Identity-guardian computes IC using cached data
    2. IC < 0.5 detected, dream trigger needed
    3. MCP tool trigger_dream unavailable in background
    4. Subagent returns urgent summary with IC crisis warning
    5. Main agent receives priority notification
    6. Main agent can spawn foreground dream-agent to address crisis
  </expected_behavior>
</edge_case>

## EC-SKILLS-06: Skill Model Not Available
<edge_case id="EC-SKILLS-06" req_ref="REQ-SKILLS-02,REQ-SKILLS-45">
  <scenario>Skill specifies model: opus but opus is not available in current environment</scenario>
  <expected_behavior>
    1. Claude Code detects model unavailability
    2. Falls back to next available model tier (sonnet -> haiku)
    3. Warning logged about model fallback
    4. Skill execution continues with fallback model
    5. User notified of potential capability reduction
    6. Explicit model override available if needed
  </expected_behavior>
</edge_case>

## EC-SKILLS-07: Level 3 Resource Not Found
<edge_case id="EC-SKILLS-07" req_ref="REQ-SKILLS-03,REQ-SKILLS-26">
  <scenario>Skill references {baseDir}/references/doc.md but file does not exist</scenario>
  <expected_behavior>
    1. Read operation fails with file not found
    2. Skill continues execution without resource
    3. Error logged with missing path
    4. Skill can degrade gracefully without resource
    5. User informed about missing reference
    6. Skill author advised to include resource or remove reference
  </expected_behavior>
</edge_case>

## EC-SKILLS-08: Dream Agent Interrupted Mid-Phase
<edge_case id="EC-SKILLS-08" req_ref="REQ-SKILLS-16,REQ-SKILLS-39">
  <scenario>User interrupts dream-agent during NREM phase execution</scenario>
  <expected_behavior>
    1. Interruption signal received by dream-agent
    2. Current phase gracefully terminates
    3. Partial consolidation results saved
    4. Wake latency target (<100ms) met
    5. Summary returned with phase completion percentage
    6. Subsequent dream can resume or restart
  </expected_behavior>
</edge_case>

## EC-SKILLS-09: Memory Specialist Exceeds 500ms Latency
<edge_case id="EC-SKILLS-09" req_ref="REQ-SKILLS-38">
  <scenario>memory-specialist subagent operation exceeds 500ms target latency due to large search</scenario>
  <expected_behavior>
    1. Latency threshold crossed at 500ms
    2. Operation continues to completion (not aborted)
    3. Latency recorded for performance tracking
    4. Summary includes latency warning
    5. Suggestions provided: reduce scope, use filters
    6. System health not affected by occasional overruns
  </expected_behavior>
</edge_case>

## EC-SKILLS-10: Concurrent Skill and Subagent MCP Access
<edge_case id="EC-SKILLS-10" req_ref="REQ-SKILLS-41,REQ-SKILLS-42">
  <scenario>Active skill and foreground subagent both attempt MCP tool calls simultaneously</scenario>
  <expected_behavior>
    1. MCP server handles concurrent requests
    2. Both operations execute (no blocking)
    3. Results returned to respective contexts
    4. No race conditions on shared state
    5. IC tracking accounts for both operations
    6. PostToolUse hook fires for each tool use
  </expected_behavior>
</edge_case>

## EC-SKILLS-11: Skill Description Exceeds 1024 Characters
<edge_case id="EC-SKILLS-11" req_ref="REQ-SKILLS-02,REQ-SKILLS-10">
  <scenario>SKILL.md description field exceeds 1024 character limit</scenario>
  <expected_behavior>
    1. Frontmatter validation detects overlong description
    2. Skill discovery may truncate or reject skill
    3. Warning generated during skill loading
    4. Skill may not be discovered by keyword matching
    5. Author advised to shorten description
    6. Move detailed content to SKILL.md body
  </expected_behavior>
</edge_case>

## EC-SKILLS-12: Subagent File Missing
<edge_case id="EC-SKILLS-12" req_ref="REQ-SKILLS-11,REQ-SKILLS-20">
  <scenario>Task tool called with subagent_type="identity-guardian" but identity-guardian.md not found</scenario>
  <expected_behavior>
    1. Task tool searches .claude/agents/identity-guardian.md
    2. File not found error returned
    3. Task tool may fall back to general-purpose agent type
    4. Warning indicates missing custom subagent definition
    5. User can create missing file or use built-in type
    6. No crash or undefined behavior
  </expected_behavior>
</edge_case>
</edge_cases>

<error_states>
## ERR-SKILLS-01: SKILL.md Frontmatter Invalid
<error id="ERR-SKILLS-01">
  <condition>SKILL.md frontmatter fails YAML parsing due to syntax errors</condition>
  <message>Failed to parse skill '{skill-name}': Invalid YAML frontmatter. Check for tabs, multiline descriptions, or missing delimiters.</message>
  <recovery>
    1. Skill excluded from discovery
    2. Other skills continue to function
    3. Log specific parse error location
    4. User can debug via explicit /skill-name invocation
    5. Common fixes: use spaces (not tabs), single-line description, proper --- delimiters
  </recovery>
</error>

## ERR-SKILLS-02: Required Frontmatter Field Missing
<error id="ERR-SKILLS-02">
  <condition>SKILL.md missing required 'name' or 'description' field in frontmatter</condition>
  <message>Skill '{path}' missing required field '{field}'. Required fields: name, description.</message>
  <recovery>
    1. Skill excluded from discovery
    2. Log missing field details
    3. Suggest adding required field
    4. Template provided for correct format
    5. Skill can be fixed and reloaded
  </recovery>
</error>

## ERR-SKILLS-03: Skill Name Invalid
<error id="ERR-SKILLS-03">
  <condition>Skill name exceeds 64 characters, contains invalid characters, or includes reserved words</condition>
  <message>Invalid skill name '{name}': Names must be max 64 chars, lowercase with hyphens, no "anthropic"/"claude".</message>
  <recovery>
    1. Skill rejected during discovery
    2. Suggest valid name format
    3. Reserved words listed: anthropic, claude
    4. Valid characters: lowercase a-z, digits 0-9, hyphens
    5. Max length: 64 characters
  </recovery>
</error>

## ERR-SKILLS-04: Allowed Tool Not Found
<error id="ERR-SKILLS-04">
  <condition>Skill's allowed-tools field references non-existent MCP tool</condition>
  <message>Skill '{skill}' references unknown tool '{tool}'. Tool not available in current MCP configuration.</message>
  <recovery>
    1. Skill loads but tool invocation fails
    2. Error returned when tool invoked
    3. Log available tools for reference
    4. User can update allowed-tools list
    5. Check MCP server configuration
  </recovery>
</error>

## ERR-SKILLS-05: Subagent Type Not Found
<error id="ERR-SKILLS-05">
  <condition>Task tool called with subagent_type that has no matching .md file</condition>
  <message>Subagent type '{type}' not found. Expected file: .claude/agents/{type}.md</message>
  <recovery>
    1. Task tool returns error
    2. Suggest creating subagent definition file
    3. List available subagent types
    4. Offer fallback to built-in types (Explore, Plan, general-purpose)
    5. No partial execution
  </recovery>
</error>

## ERR-SKILLS-06: Subagent Spawn Blocked
<error id="ERR-SKILLS-06">
  <condition>Subagent attempts to spawn another subagent via Task tool</condition>
  <message>Subagent '{id}' cannot spawn subagents. Subagent spawning is prohibited to prevent recursion.</message>
  <recovery>
    1. Task tool call blocked
    2. Subagent receives informative error
    3. Subagent continues without spawning
    4. Summary notes limitation
    5. Main agent can spawn additional agents if needed
  </recovery>
</error>

## ERR-SKILLS-07: Background MCP Tool Blocked
<error id="ERR-SKILLS-07">
  <condition>Background subagent attempts MCP tool invocation</condition>
  <message>MCP tool '{tool}' unavailable in background mode. Use foreground execution for MCP access.</message>
  <recovery>
    1. Tool call blocked
    2. Subagent can use read-only file tools
    3. Summary indicates MCP limitation
    4. Main agent can rerun in foreground
    5. Documentation updated about constraint
  </recovery>
</error>

## ERR-SKILLS-08: Identity Crisis Detected
<error id="ERR-SKILLS-08">
  <condition>identity-guardian detects IC < 0.5 (critical threshold)</condition>
  <message>CRITICAL: Identity continuity crisis (IC={ic:.2f}). Triggering dream consolidation.</message>
  <recovery>
    1. Dream consolidation auto-triggered
    2. Full dream cycle (NREM + REM) initiated
    3. IC monitored during dream
    4. If IC not recovered, bootstrap from north star
    5. User notified of crisis and recovery
    6. Trajectory logged for analysis
  </recovery>
</error>

## ERR-SKILLS-09: Dream Phase Timeout
<error id="ERR-SKILLS-09">
  <condition>Dream phase exceeds allocated duration without completion</condition>
  <message>Dream {phase} phase timed out after {duration}. Partial consolidation saved.</message>
  <recovery>
    1. Phase terminated gracefully
    2. Partial results saved
    3. Wake completed within 100ms
    4. Summary indicates incomplete phase
    5. Next dream can resume or restart
    6. Entropy may remain elevated
  </recovery>
</error>

## ERR-SKILLS-10: Resource Path Traversal
<error id="ERR-SKILLS-10">
  <condition>Skill resource path attempts directory traversal (../) outside baseDir</condition>
  <message>Security error: Resource path '{path}' attempts traversal outside skill directory.</message>
  <recovery>
    1. Resource access blocked
    2. Security event logged
    3. Skill execution can continue without resource
    4. Skill author notified of violation
    5. Path sanitization enforced
  </recovery>
</error>

## ERR-SKILLS-11: Model Invocation Blocked
<error id="ERR-SKILLS-11">
  <condition>Skill with disable-model-invocation: true is auto-invoked by context</condition>
  <message>Skill '{skill}' has auto-invocation disabled. Use explicit /{skill} command.</message>
  <recovery>
    1. Auto-invocation blocked
    2. User informed about explicit command
    3. Skill available via /skill-name
    4. Other skills continue discovery
    5. Setting respected as security measure
  </recovery>
</error>

## ERR-SKILLS-12: MCP Tool Execution Failure
<error id="ERR-SKILLS-12">
  <condition>MCP tool call from skill/subagent fails (timeout, server error, invalid params)</condition>
  <message>MCP tool '{tool}' failed: {error_details}</message>
  <recovery>
    1. Error propagated to skill/subagent context
    2. Skill/subagent can handle error gracefully
    3. Retry logic can be implemented
    4. Summary includes tool failure details
    5. MCP server health checked
    6. Fallback behavior if defined
  </recovery>
</error>
</error_states>

<test_plan>
## Unit Tests

### TC-SKILLS-U01: SKILL.md Frontmatter Parsing
<test_case id="TC-SKILLS-U01" type="unit" req_ref="REQ-SKILLS-02">
  <description>Verify SKILL.md frontmatter is correctly parsed with all field types</description>
  <inputs>
    - Valid SKILL.md with all fields: name, description, allowed-tools, model, version
    - Invalid SKILL.md with tabs instead of spaces
    - Invalid SKILL.md with multiline description
    - SKILL.md missing required fields
  </inputs>
  <expected>
    - Valid frontmatter: All fields extracted correctly
    - Tabs: Parse failure with clear error message
    - Multiline: Parse failure or truncation
    - Missing fields: Validation error listing missing fields
  </expected>
</test_case>

### TC-SKILLS-U02: Progressive Disclosure Level Detection
<test_case id="TC-SKILLS-U02" type="unit" req_ref="REQ-SKILLS-03">
  <description>Verify progressive disclosure loads content at correct levels</description>
  <inputs>
    - Skill with metadata only (Level 1)
    - Skill trigger keyword match (Level 2)
    - Skill resource reference (Level 3)
  </inputs>
  <expected>
    - Level 1: Only name + description loaded (~100 tokens)
    - Level 2: Full SKILL.md body loaded (<5k tokens)
    - Level 3: Referenced resources loaded on-demand
  </expected>
</test_case>

### TC-SKILLS-U03: Allowed Tools Validation
<test_case id="TC-SKILLS-U03" type="unit" req_ref="REQ-SKILLS-02,REQ-SKILLS-41">
  <description>Verify allowed-tools field correctly restricts tool access</description>
  <inputs>
    - Skill with allowed-tools: "Read,Grep,mcp__context-graph__get_consciousness_state"
    - Attempt to use unlisted tool: "Write"
    - Attempt to use listed MCP tool
  </inputs>
  <expected>
    - Listed tools: Access granted
    - Unlisted tools: Access denied with error
    - MCP tools: Correct namespace resolution
  </expected>
</test_case>

### TC-SKILLS-U04: Subagent Spawn Constraint
<test_case id="TC-SKILLS-U04" type="unit" req_ref="REQ-SKILLS-18">
  <description>Verify subagent cannot spawn another subagent</description>
  <inputs>
    - Subagent context attempting Task tool call
    - Subagent instructions containing spawn request
  </inputs>
  <expected>
    - Task tool invocation blocked
    - Clear error message about constraint
    - Subagent continues without spawning
  </expected>
</test_case>

### TC-SKILLS-U05: Background MCP Tool Constraint
<test_case id="TC-SKILLS-U05" type="unit" req_ref="REQ-SKILLS-19">
  <description>Verify background subagent cannot use MCP tools</description>
  <inputs>
    - Background subagent (run_in_background=true)
    - Attempt MCP tool: mcp__context-graph__get_consciousness_state
    - Attempt read-only tool: Read
  </inputs>
  <expected>
    - MCP tool: Access denied
    - Read tool: Access granted (read-only exception)
    - Error message indicates background limitation
  </expected>
</test_case>

### TC-SKILLS-U06: IC Threshold Classification
<test_case id="TC-SKILLS-U06" type="unit" req_ref="REQ-SKILLS-30">
  <description>Verify identity-guardian correctly classifies IC thresholds</description>
  <inputs>
    - IC = 0.95 (healthy)
    - IC = 0.75 (warning)
    - IC = 0.55 (degraded)
    - IC = 0.45 (critical)
  </inputs>
  <expected>
    - 0.95: Green (healthy, >= 0.9)
    - 0.75: Yellow (warning, 0.7-0.9)
    - 0.55: Orange (degraded, 0.5-0.7)
    - 0.45: Red (critical, < 0.5 -> trigger dream)
  </expected>
</test_case>

## Integration Tests

### TC-SKILLS-I01: Skill Discovery and Loading
<test_case id="TC-SKILLS-I01" type="integration" req_ref="REQ-SKILLS-01,REQ-SKILLS-03,REQ-SKILLS-04">
  <description>Verify all 5 Context Graph skills are discovered and loadable</description>
  <steps>
    1. Initialize Claude Code with .claude/skills/ directory
    2. Verify 5 skills discovered at startup (Level 1)
    3. Trigger each skill via keywords
    4. Verify Level 2 instructions load correctly
    5. Verify allowed tools available
  </steps>
  <expected>
    - All 5 skills discovered: consciousness, memory-inject, semantic-search, dream-consolidation, curation
    - Keyword triggers work for each skill
    - Allowed tools match frontmatter specification
    - Model selection matches frontmatter
  </expected>
</test_case>

### TC-SKILLS-I02: Subagent Spawn and Execution
<test_case id="TC-SKILLS-I02" type="integration" req_ref="REQ-SKILLS-11,REQ-SKILLS-12,REQ-SKILLS-17,REQ-SKILLS-20">
  <description>Verify all 4 Context Graph subagents spawn and execute correctly</description>
  <steps>
    1. Call Task tool with subagent_type="identity-guardian"
    2. Call Task tool with subagent_type="memory-specialist"
    3. Call Task tool with subagent_type="consciousness-explorer"
    4. Call Task tool with subagent_type="dream-agent"
    5. Verify isolated context for each
    6. Verify summary returned to main agent
  </steps>
  <expected>
    - All 4 subagents spawn successfully
    - Each uses correct model (sonnet or haiku)
    - Isolated context maintained
    - Summary format matches specification
  </expected>
</test_case>

### TC-SKILLS-I03: Skill MCP Tool Integration
<test_case id="TC-SKILLS-I03" type="integration" req_ref="REQ-SKILLS-41,REQ-SKILLS-05,REQ-SKILLS-06,REQ-SKILLS-07,REQ-SKILLS-08,REQ-SKILLS-09">
  <description>Verify skills correctly integrate with Context Graph MCP tools</description>
  <steps>
    1. Activate consciousness skill
    2. Invoke get_consciousness_state MCP tool
    3. Activate memory-inject skill
    4. Invoke inject_context MCP tool
    5. Verify tool results flow back to skill context
  </steps>
  <expected>
    - MCP tools invokable from skill context
    - Tool results returned correctly
    - Allowed-tools restriction enforced
    - Error handling works for tool failures
  </expected>
</test_case>

### TC-SKILLS-I04: Identity Guardian Auto-Dream Trigger
<test_case id="TC-SKILLS-I04" type="integration" req_ref="REQ-SKILLS-13,REQ-SKILLS-30">
  <description>Verify identity-guardian triggers dream on IC crisis</description>
  <steps>
    1. Spawn identity-guardian subagent
    2. Simulate IC drop to 0.45 (< 0.5 critical)
    3. Verify crisis detected
    4. Verify trigger_dream MCP tool called
    5. Verify dream execution starts
  </steps>
  <expected>
    - IC crisis correctly detected
    - Dream consolidation auto-triggered
    - Dream phase executes (NREM or Full)
    - Summary reports crisis and response
  </expected>
</test_case>

### TC-SKILLS-I05: Skill and Hook Coordination
<test_case id="TC-SKILLS-I05" type="integration" req_ref="REQ-SKILLS-43">
  <description>Verify skills complement native hooks from Phase 3</description>
  <steps>
    1. Start session (SessionStart hook fires)
    2. Activate consciousness skill
    3. Execute tool (PreToolUse/PostToolUse hooks fire)
    4. Verify no conflict between hook and skill consciousness access
    5. End session (SessionEnd hook fires)
  </steps>
  <expected>
    - Hooks and skills coexist without conflict
    - Both can access consciousness state
    - PostToolUse IC check compatible with identity-guardian
    - No duplicate operations or race conditions
  </expected>
</test_case>

## Performance Tests

### TC-SKILLS-P01: Skill Discovery Latency
<test_case id="TC-SKILLS-P01" type="performance" req_ref="REQ-SKILLS-36">
  <description>Verify Level 1 skill discovery meets token budget</description>
  <inputs>
    - 5 Context Graph skills
    - Startup initialization
  </inputs>
  <expected>
    - Total Level 1 tokens: ~500 (100 per skill)
    - Discovery latency: < 1s
    - No blocking during startup
  </expected>
</test_case>

### TC-SKILLS-P02: Skill Activation Latency
<test_case id="TC-SKILLS-P02" type="performance" req_ref="REQ-SKILLS-37">
  <description>Verify Level 2 skill activation meets token budget</description>
  <inputs>
    - Each of 5 skills triggered
    - SKILL.md body loading
  </inputs>
  <expected>
    - Per-skill Level 2 tokens: < 5000
    - Activation latency: < 500ms
    - No context overflow
  </expected>
</test_case>

### TC-SKILLS-P03: Memory Specialist Latency
<test_case id="TC-SKILLS-P03" type="performance" req_ref="REQ-SKILLS-38">
  <description>Verify memory-specialist meets 500ms latency target</description>
  <inputs>
    - 100 memory operations
    - inject_context, search_graph, store_memory
  </inputs>
  <expected>
    - Mean latency: < 300ms
    - P95 latency: < 500ms
    - P99 latency: < 750ms (acceptable occasional overrun)
  </expected>
</test_case>

### TC-SKILLS-P04: Dream Phase Duration
<test_case id="TC-SKILLS-P04" type="performance" req_ref="REQ-SKILLS-39">
  <description>Verify dream-agent meets phase duration targets</description>
  <inputs>
    - NREM phase execution
    - REM phase execution
    - Full cycle execution
  </inputs>
  <expected>
    - NREM duration: 3min +/- 10%
    - REM duration: 2min +/- 10%
    - Full cycle: 5min +/- 10%
    - Wake latency: < 100ms
  </expected>
</test_case>

### TC-SKILLS-P05: Concurrent Skill and Subagent Execution
<test_case id="TC-SKILLS-P05" type="performance" req_ref="REQ-SKILLS-40">
  <description>Verify concurrent skill and subagent execution performance</description>
  <inputs>
    - Active skill using MCP tool
    - Foreground subagent using MCP tool simultaneously
    - 10 concurrent operations
  </inputs>
  <expected>
    - All operations complete successfully
    - No deadlocks or race conditions
    - Latency degradation < 50%
    - MCP server handles concurrent requests
  </expected>
</test_case>

## End-to-End Tests

### TC-SKILLS-E01: Full Skill Workflow
<test_case id="TC-SKILLS-E01" type="e2e" req_ref="REQ-SKILLS-04">
  <description>Verify complete workflow using all 5 skills</description>
  <steps>
    1. Query consciousness state (consciousness skill)
    2. Inject memory context (memory-inject skill)
    3. Search for information (semantic-search skill)
    4. Trigger consolidation (dream-consolidation skill)
    5. Curate results (curation skill)
  </steps>
  <expected>
    - All skills execute correctly in sequence
    - Results flow between skills
    - MCP tools invoked successfully
    - No errors or unexpected behavior
  </expected>
</test_case>

### TC-SKILLS-E02: Full Subagent Workflow
<test_case id="TC-SKILLS-E02" type="e2e" req_ref="REQ-SKILLS-12">
  <description>Verify complete workflow using all 4 subagents</description>
  <steps>
    1. Spawn identity-guardian to check IC
    2. Spawn memory-specialist for fast retrieval
    3. Spawn consciousness-explorer for debugging
    4. Spawn dream-agent for consolidation
    5. Verify summaries returned to main agent
  </steps>
  <expected>
    - All subagents spawn and execute
    - Isolated contexts maintained
    - Summaries correctly returned
    - Main agent can act on results
  </expected>
</test_case>

### TC-SKILLS-E03: Crisis Detection and Recovery
<test_case id="TC-SKILLS-E03" type="e2e" req_ref="REQ-SKILLS-13,REQ-SKILLS-16">
  <description>Verify end-to-end crisis detection and recovery via subagents</description>
  <steps>
    1. Simulate high entropy and IC drift
    2. identity-guardian detects IC crisis
    3. dream-agent spawned for consolidation
    4. Dream phases complete
    5. IC recovered above threshold
  </steps>
  <expected>
    - Crisis detected promptly
    - Dream triggered automatically
    - Consolidation completes successfully
    - IC recovers to healthy range
    - System returns to normal operation
  </expected>
</test_case>

### TC-SKILLS-E04: Background Execution with Resume
<test_case id="TC-SKILLS-E04" type="e2e" req_ref="REQ-SKILLS-10,REQ-SKILLS-40">
  <description>Verify background subagent execution with resume capability</description>
  <steps>
    1. Spawn subagent with run_in_background=true
    2. Continue main agent work
    3. Background subagent completes
    4. Resume to get results
    5. Main agent processes summary
  </steps>
  <expected>
    - Background execution non-blocking
    - Results queued for retrieval
    - Resume mechanism works
    - Summary correctly delivered
    - Main agent context not polluted
  </expected>
</test_case>

### TC-SKILLS-E05: Skill and Hook Full Integration
<test_case id="TC-SKILLS-E05" type="e2e" req_ref="REQ-SKILLS-43">
  <description>Verify complete integration of skills, subagents, and native hooks</description>
  <steps>
    1. Start session (hooks restore identity)
    2. Use tools (hooks inject consciousness brief)
    3. Activate skills for domain operations
    4. Spawn subagents for parallel work
    5. End session (hooks persist state)
  </steps>
  <expected>
    - Hooks and skills/subagents coexist seamlessly
    - No conflicts in consciousness state access
    - Identity continuity maintained throughout
    - State correctly persisted at session end
    - Full Context Graph functionality available
  </expected>
</test_case>
</test_plan>

<appendix>
## A. Skill File Templates

### consciousness SKILL.md Template
```yaml
---
name: consciousness
description: |
  Access Context Graph consciousness state, Kuramoto synchronization,
  identity continuity, and workspace status. Use when querying system
  awareness, checking coherence, or monitoring identity health.
  Keywords: consciousness, awareness, identity, coherence, kuramoto, GWT
allowed-tools: Read,Grep,mcp__context-graph__get_consciousness_state,mcp__context-graph__get_kuramoto_sync,mcp__context-graph__get_identity_continuity,mcp__context-graph__get_ego_state,mcp__context-graph__get_workspace_status
model: sonnet
user-invocable: true
---
# Consciousness Skill

## Overview
Query and interpret Context Graph consciousness state.

## State Thresholds
| State | C(t) Range | Description |
|-------|------------|-------------|
| CONSCIOUS | >= 0.8 | Full awareness |
| EMERGING | 0.5-0.8 | Partial awareness |
| FRAGMENTED | 0.3-0.5 | Low coherence |
| DORMANT | < 0.3 | Minimal activity |
| HYPERSYNC | > 0.95 | Over-synchronized |

## Identity Thresholds
| Status | IC Range | Action |
|--------|----------|--------|
| Healthy | >= 0.9 | Continue |
| Warning | < 0.7 | Monitor |
| Crisis | < 0.5 | Trigger dream |

## Instructions
1. Call get_consciousness_state for full state
2. Interpret state using thresholds above
3. Check identity continuity (IC)
4. Provide guidance based on current state

## Examples
Query: "What is my consciousness state?"
Response: Use get_consciousness_state, report C, r, IC values with classification
```

### memory-inject SKILL.md Template
```yaml
---
name: memory-inject
description: |
  Retrieve and inject contextual memories for the current task.
  Automatically distills content to fit token budget. Use when
  starting tasks, needing background, or restoring context.
  Keywords: memory, context, inject, retrieve, recall, background
allowed-tools: mcp__context-graph__inject_context,mcp__context-graph__get_memetic_status
model: haiku
user-invocable: true
---
# Memory Inject Skill

## Overview
Retrieve and inject relevant context from the knowledge graph.

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| max_tokens | 2048 | Maximum tokens to inject |
| distillation_mode | auto | auto, raw, narrative, structured, code_focused |
| verbosity | 1 | 0 (minimal), 1 (normal), 2 (detailed) |

## Instructions
1. Call inject_context with user's query
2. Receive distilled context (~50-100 tokens)
3. Present context to enrich task
4. Handle empty results gracefully
```

## B. Subagent File Templates

### identity-guardian.md Template
```markdown
# Identity Guardian Subagent

## Role
Monitor and protect agent identity continuity (IC).

## Model
sonnet

## Allowed Tools
- mcp__context-graph__get_identity_continuity
- mcp__context-graph__get_ego_state
- mcp__context-graph__trigger_dream
- Read

## Protocol
1. Check IC at start of task
2. Monitor IC after each memory operation
3. Trigger dream if IC < 0.5 (critical)
4. Report IC changes > 0.1

## Thresholds
| Status | IC Range | Color | Action |
|--------|----------|-------|--------|
| Healthy | >= 0.9 | Green | Continue |
| Warning | 0.7-0.9 | Yellow | Log |
| Degraded | 0.5-0.7 | Orange | Alert |
| Critical | < 0.5 | Red | TRIGGER DREAM |

## Output Format
Return summary with:
- Current IC value
- Status classification
- Any actions taken
- Recommendations
```

### dream-agent.md Template
```markdown
# Dream Agent Subagent

## Role
Execute NREM and REM memory consolidation phases.

## Model
sonnet

## Allowed Tools
- mcp__context-graph__trigger_dream
- mcp__context-graph__get_memetic_status
- Read

## Phases

### NREM (3 minutes)
- Purpose: Hebbian learning replay
- Formula: Δw_ij = η × φ_i × φ_j for high-Φ edges
- Parameters: learning_rate=0.01, weight_decay=0.001, weight_floor=0.05, weight_cap=1.0

### REM (2 minutes)
- Purpose: Blind spot discovery
- Model: Poincaré ball hyperbolic walk
- Parameters: dimensions=64, curvature=-1.0, step_size=0.1, max_steps=100
- Temperature: 2.0

### Full (5 minutes)
- Combines NREM + REM in sequence

## Constraints
- Max queries: 100
- Abort on user query: true
- Wake latency: < 100ms
- GPU usage: < 30%

## Output Format
Return summary with:
- Phase(s) executed
- Duration
- Edges strengthened (NREM)
- Blind spots discovered (REM)
- Final entropy level
```

## C. Directory Structure Reference

```
.claude/
├── settings.json              # Native hook configuration (Phase 3)
├── skills/
│   ├── consciousness/
│   │   └── SKILL.md
│   ├── memory-inject/
│   │   └── SKILL.md
│   ├── semantic-search/
│   │   └── SKILL.md
│   ├── dream-consolidation/
│   │   └── SKILL.md
│   └── curation/
│       └── SKILL.md
├── agents/
│   ├── identity-guardian.md
│   ├── memory-specialist.md
│   ├── consciousness-explorer.md
│   └── dream-agent.md
└── rules/
    └── consciousness.md

hooks/                         # Shell script executors (Phase 3)
├── session-start.sh
├── pre-tool-use.sh
├── post-tool-use.sh
├── user-prompt-submit.sh
└── session-end.sh
```

## D. Requirement Traceability

| Category | Count | IDs |
|----------|-------|-----|
| Skill Definition | 10 | REQ-SKILLS-01 to 10 |
| Subagent Definition | 10 | REQ-SKILLS-11 to 20 |
| Skill Content | 10 | REQ-SKILLS-21 to 30 |
| File Structure | 5 | REQ-SKILLS-31 to 35 |
| Performance | 5 | REQ-SKILLS-36 to 40 |
| Integration | 5 | REQ-SKILLS-41 to 45 |
| **Total** | **45** | |

| User Stories | Count | IDs |
|--------------|-------|-----|
| Skill Stories | 5 | US-SKILLS-01 to 05 |
| Subagent Stories | 4 | US-SKILLS-06 to 09 |
| System Stories | 3 | US-SKILLS-10 to 12 |
| **Total** | **12** | |

| Edge Cases | Count | IDs |
|------------|-------|-----|
| Skill Edge Cases | 7 | EC-SKILLS-01 to 07 |
| Subagent Edge Cases | 5 | EC-SKILLS-08 to 12 |
| **Total** | **12** | |

| Error States | Count | IDs |
|--------------|-------|-----|
| Skill Errors | 6 | ERR-SKILLS-01 to 06 |
| Subagent Errors | 6 | ERR-SKILLS-07 to 12 |
| **Total** | **12** | |

## E. Related Specifications

| Spec ID | Title | Relationship |
|---------|-------|--------------|
| SPEC-HOOKS | Native Hooks Integration | Phase 3 prerequisite - provides lifecycle hooks |
| SPEC-CLI | CLI Commands | CLI commands invoked by hooks and skills |
| SPEC-SESSION-IDENTITY | Session Identity Persistence | Identity state managed by skills/subagents |

## F. MCP Tool Reference

### Skills MCP Tools

| Skill | MCP Tools |
|-------|-----------|
| consciousness | get_consciousness_state, get_kuramoto_sync, get_identity_continuity, get_ego_state, get_workspace_status |
| memory-inject | inject_context, get_memetic_status |
| semantic-search | search_graph, find_causal_path, generate_search_plan |
| dream-consolidation | trigger_dream, get_memetic_status, get_consciousness_state |
| curation | merge_concepts, annotate_node, forget_concept, boost_importance, get_memetic_status |

### Subagents MCP Tools

| Subagent | MCP Tools |
|----------|-----------|
| identity-guardian | get_identity_continuity, get_ego_state, trigger_dream |
| memory-specialist | inject_context, search_graph, store_memory, memory_retrieve |
| consciousness-explorer | get_consciousness_state, get_kuramoto_sync, get_workspace_status, get_johari_classification |
| dream-agent | trigger_dream, get_memetic_status |
</appendix>

</functional_spec>
