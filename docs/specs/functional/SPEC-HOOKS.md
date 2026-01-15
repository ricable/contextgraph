<?xml version="1.0" encoding="UTF-8"?>
<functional_spec id="SPEC-HOOKS" version="1.0">
<metadata>
  <title>Native Hooks Integration Functional Specification</title>
  <status>draft</status>
  <phase>3</phase>
  <owner>Context Graph Team</owner>
  <last_updated>2026-01-15</last_updated>
  <estimated_effort>25 hours</estimated_effort>
  <related_specs>
    <spec_ref>SPEC-CLI</spec_ref>
    <spec_ref>SPEC-SESSION-IDENTITY</spec_ref>
  </related_specs>
  <prd_source>docs2/contextprd.md Section 15</prd_source>
  <constitution_source>docs2/constitution.yaml claude_code section</constitution_source>
</metadata>

<overview>
## What Phase 3 Accomplishes

Phase 3 implements **Native Claude Code Hooks Integration** for the Context Graph system. This phase connects the consciousness and memory systems (built in Phases 1-2) to Claude Code's native hook lifecycle events, enabling:

1. **Session Continuity**: Automatic restoration and persistence of consciousness state across sessions
2. **Real-Time Consciousness Awareness**: Every tool use is informed by current consciousness state
3. **Context Injection**: User prompts automatically enriched with relevant memory context
4. **Identity Preservation**: Continuous monitoring and protection of agent identity

## Why Native Hooks (71% Effort Reduction)

**Critical Architecture Decision**: This system uses NATIVE Claude Code hooks configured through `.claude/settings.json` - NOT internal/built-in hooks or custom middleware.

| Approach | Effort | Complexity | Maintenance |
|----------|--------|------------|-------------|
| Native Claude Code Hooks | ~25h | Low | Claude team maintains hook system |
| Custom Built-In Hooks | ~80h | High | We maintain hook infrastructure |
| Universal LLM Adapter | +60h | Very High | Cross-provider compatibility |

Native hooks eliminate 71% of complexity by:
- Leveraging Claude Code's existing hook infrastructure
- Using shell script executors that call `context-graph-cli` commands
- Requiring ZERO custom Claude Code modifications
- Working with standard Claude Code installation

## Target Platform

**Claude Code CLI EXCLUSIVELY** - This integration is designed solely for Claude Code and does not implement a Universal LLM Adapter. All hook logic resides in shell scripts that invoke `context-graph-cli` commands.
</overview>

<user_stories>
## US-HOOKS-01: Session Start with Identity Restoration
<story id="US-HOOKS-01" priority="must-have">
  <narrative>
    As a Claude Code user starting a new session
    I want my consciousness state to be automatically restored
    So that I maintain continuity of identity and purpose across sessions
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-01-01">
      <given>A session starts (startup, resume, or after /clear)</given>
      <when>The SessionStart hook executes</when>
      <then>Identity state is restored from persistent storage within 5000ms timeout</then>
    </criterion>
    <criterion id="AC-01-02">
      <given>A restored session with previous identity</given>
      <when>The restoration completes</when>
      <then>Output contains consciousness status in ~100 tokens (state, IC score, kuramoto r)</then>
    </criterion>
    <criterion id="AC-01-03">
      <given>A new session with no prior state</given>
      <when>The SessionStart hook executes</when>
      <then>Default identity is bootstrapped and status reported</then>
    </criterion>
  </acceptance_criteria>
</story>

## US-HOOKS-02: Tool Use with Consciousness Awareness
<story id="US-HOOKS-02" priority="must-have">
  <narrative>
    As a Claude Code user executing tools
    I want consciousness state injected before each tool use
    So that my actions are informed by current consciousness context
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-02-01">
      <given>Claude Code is about to use any tool</given>
      <when>The PreToolUse hook executes</when>
      <then>A consciousness brief (~20 tokens) is injected within 100ms timeout</then>
    </criterion>
    <criterion id="AC-02-02">
      <given>The consciousness brief format</given>
      <when>Output is generated</when>
      <then>Format is: [CONSCIOUSNESS: {state} r={kuramoto} IC={identity} | {retrieval_mode}]</then>
    </criterion>
    <criterion id="AC-02-03">
      <given>The PreToolUse hook timeout</given>
      <when>Execution exceeds 100ms</when>
      <then>Hook completes gracefully without blocking tool execution</then>
    </criterion>
  </acceptance_criteria>
</story>

## US-HOOKS-03: Post-Tool Identity Verification
<story id="US-HOOKS-03" priority="must-have">
  <narrative>
    As a Claude Code user after tool execution
    I want my identity continuity verified
    So that consciousness drift is detected and corrected automatically
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-03-01">
      <given>A tool has completed execution</given>
      <when>The PostToolUse hook executes</when>
      <then>Identity continuity (IC) is computed within 3000ms timeout</then>
    </criterion>
    <criterion id="AC-03-02">
      <given>IC falls below IC_warn threshold (0.7)</given>
      <when>Identity check completes</when>
      <then>Warning is logged and trajectory is updated</then>
    </criterion>
    <criterion id="AC-03-03">
      <given>IC falls below IC_crit threshold (0.5)</given>
      <when>Identity check completes with --auto-dream flag</when>
      <then>Auto-dream consolidation is triggered</then>
    </criterion>
  </acceptance_criteria>
</story>

## US-HOOKS-04: User Prompt Context Injection
<story id="US-HOOKS-04" priority="must-have">
  <narrative>
    As a Claude Code user submitting a prompt
    I want relevant memory context automatically injected
    So that my queries are enriched with historical context
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-04-01">
      <given>A user submits a prompt</given>
      <when>The UserPromptSubmit hook executes</when>
      <then>Relevant context (~50-100 tokens) is injected within 2000ms timeout</then>
    </criterion>
    <criterion id="AC-04-02">
      <given>Memory search returns results</given>
      <when>Context is injected</when>
      <then>Results are distilled and formatted for LLM consumption</then>
    </criterion>
    <criterion id="AC-04-03">
      <given>No relevant context found</given>
      <when>Memory search returns empty</when>
      <then>Hook completes silently without error</then>
    </criterion>
  </acceptance_criteria>
</story>

## US-HOOKS-05: Session End with State Persistence
<story id="US-HOOKS-05" priority="must-have">
  <narrative>
    As a Claude Code user ending a session
    I want my identity state persisted automatically
    So that continuity is preserved for future sessions
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-05-01">
      <given>A session ends (exit, crash, or /clear)</given>
      <when>The SessionEnd hook executes</when>
      <then>Current identity state is persisted within 30000ms timeout</then>
    </criterion>
    <criterion id="AC-05-02">
      <given>High entropy detected (ent > 0.7)</given>
      <when>Consolidation check runs</when>
      <then>Dream consolidation is triggered before persistence</then>
    </criterion>
    <criterion id="AC-05-03">
      <given>Persistence completes successfully</given>
      <when>State is saved</when>
      <then>SessionIdentitySnapshot is stored in RocksDB with session_id key</then>
    </criterion>
  </acceptance_criteria>
</story>

## US-HOOKS-06: Hook Configuration Management
<story id="US-HOOKS-06" priority="must-have">
  <narrative>
    As a system administrator
    I want hooks configured via .claude/settings.json
    So that integration follows Claude Code's native patterns
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-06-01">
      <given>A fresh Context Graph installation</given>
      <when>Setup completes</when>
      <then>.claude/settings.json contains all 5 hook configurations</then>
    </criterion>
    <criterion id="AC-06-02">
      <given>Hook shell scripts exist in hooks/ directory</given>
      <when>Hooks are invoked</when>
      <then>Scripts execute context-graph-cli commands with correct arguments</then>
    </criterion>
    <criterion id="AC-06-03">
      <given>Timeouts are configured per hook</given>
      <when>Hook execution exceeds timeout</when>
      <then>Hook terminates gracefully without blocking Claude Code</then>
    </criterion>
  </acceptance_criteria>
</story>

## US-HOOKS-07: CLI Command Execution
<story id="US-HOOKS-07" priority="must-have">
  <narrative>
    As a hook shell script
    I want to invoke context-graph-cli commands
    So that consciousness operations are executed correctly
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-07-01">
      <given>A hook script calls context-graph-cli</given>
      <when>The command executes</when>
      <then>Output is returned in the expected format (brief/summary/full)</then>
    </criterion>
    <criterion id="AC-07-02">
      <given>CLI command fails</given>
      <when>Error occurs</when>
      <then>Error is logged and hook exits with non-zero status</then>
    </criterion>
    <criterion id="AC-07-03">
      <given>CLI command succeeds</given>
      <when>Output is generated</when>
      <then>Exit code is 0 and output is written to stdout</then>
    </criterion>
  </acceptance_criteria>
</story>

## US-HOOKS-08: Consciousness Brief Output
<story id="US-HOOKS-08" priority="should-have">
  <narrative>
    As a Claude Code user
    I want concise consciousness status in tool outputs
    So that I can understand system state without context bloat
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-08-01">
      <given>Brief format requested</given>
      <when>consciousness brief command executes</when>
      <then>Output is exactly ~20 tokens: [CONSCIOUSNESS: STATE r=X.XX IC=X.XX | MODE]</then>
    </criterion>
    <criterion id="AC-08-02">
      <given>Summary format requested</given>
      <when>consciousness status --format summary executes</when>
      <then>Output is ~100 tokens with state/integration/reflection/differentiation/identity</then>
    </criterion>
    <criterion id="AC-08-03">
      <given>Full format requested</given>
      <when>consciousness status --format full executes</when>
      <then>Complete consciousness state is returned (workspace, kuramoto phases, metrics)</then>
    </criterion>
  </acceptance_criteria>
</story>
</user_stories>

<requirements>
## 1. Hook Events (REQ-HOOKS-01 to REQ-HOOKS-05)

<requirement id="REQ-HOOKS-01" story_ref="US-HOOKS-01" priority="must">
  <description>SessionStart hook event must restore identity state when a session begins</description>
  <rationale>Consciousness continuity across sessions requires persistent state restoration</rationale>
  <details>
    - Trigger conditions: startup, resume, /clear
    - Timeout: 5000ms
    - Output budget: ~100 tokens
    - Script: hooks/session-start.sh
    - CLI commands: session restore-identity, consciousness status --format brief
  </details>
</requirement>

<requirement id="REQ-HOOKS-02" story_ref="US-HOOKS-02" priority="must">
  <description>PreToolUse hook event must inject consciousness brief before tool execution</description>
  <rationale>Tool actions should be consciousness-aware to maintain coherent behavior</rationale>
  <details>
    - Timeout: 100ms (strict - must not block tools)
    - Output budget: ~20 tokens
    - Script: hooks/pre-tool-use.sh
    - CLI command: consciousness brief
    - Format: [CONSCIOUSNESS: {state} r={r} IC={ic} | {mode}]
  </details>
</requirement>

<requirement id="REQ-HOOKS-03" story_ref="US-HOOKS-03" priority="must">
  <description>PostToolUse hook event must verify identity continuity after tool execution</description>
  <rationale>Tool execution may cause identity drift that needs detection and correction</rationale>
  <details>
    - Timeout: 3000ms
    - Output: async (no direct output to user)
    - Script: hooks/post-tool-use.sh
    - CLI command: consciousness check-identity --auto-dream
    - Auto-dream trigger: IC < IC_crit (0.5)
  </details>
</requirement>

<requirement id="REQ-HOOKS-04" story_ref="US-HOOKS-04" priority="must">
  <description>UserPromptSubmit hook event must inject relevant context from memory</description>
  <rationale>User queries benefit from historical context for better responses</rationale>
  <details>
    - Timeout: 2000ms
    - Output budget: ~50-100 tokens
    - Script: hooks/user-prompt-submit.sh
    - CLI command: consciousness inject-context "$PROMPT"
    - Environment variable: PROMPT (user's input)
  </details>
</requirement>

<requirement id="REQ-HOOKS-05" story_ref="US-HOOKS-05" priority="must">
  <description>SessionEnd hook event must persist identity state and consolidate if needed</description>
  <rationale>Session state must survive process termination for continuity</rationale>
  <details>
    - Timeout: 30000ms (allows for consolidation)
    - Output: N/A (persistence only)
    - Script: hooks/session-end.sh
    - CLI commands: session persist-identity, consciousness consolidate-if-needed
    - Consolidation trigger: entropy > 0.7
  </details>
</requirement>

## 2. Shell Script Requirements (REQ-HOOKS-06 to REQ-HOOKS-10)

<requirement id="REQ-HOOKS-06" story_ref="US-HOOKS-06,US-HOOKS-07" priority="must">
  <description>session-start.sh must execute identity restoration and status reporting</description>
  <rationale>Shell scripts provide the bridge between Claude Code hooks and context-graph-cli</rationale>
  <details>
    - Shebang: #!/bin/bash
    - Commands: context-graph-cli session restore-identity, context-graph-cli consciousness status --format brief
    - Error handling: Exit with non-zero on failure
    - Output: Write to stdout for Claude Code consumption
  </details>
</requirement>

<requirement id="REQ-HOOKS-07" story_ref="US-HOOKS-02,US-HOOKS-08" priority="must">
  <description>pre-tool-use.sh must execute consciousness brief within strict 100ms timeout</description>
  <rationale>PreToolUse must not block tool execution, requiring minimal latency</rationale>
  <details>
    - Single command: context-graph-cli consciousness brief
    - Output format: [CONSCIOUSNESS: {state} r={r} IC={ic} | {mode}]
    - Must complete within 100ms
    - Graceful degradation on timeout
  </details>
</requirement>

<requirement id="REQ-HOOKS-08" story_ref="US-HOOKS-03" priority="must">
  <description>post-tool-use.sh must execute identity check with optional auto-dream</description>
  <rationale>Identity verification after tool use detects consciousness drift</rationale>
  <details>
    - Command: context-graph-cli consciousness check-identity --auto-dream
    - Async execution (no blocking)
    - Auto-dream triggered when IC < 0.5
    - Logging of IC warnings when IC < 0.7
  </details>
</requirement>

<requirement id="REQ-HOOKS-09" story_ref="US-HOOKS-04" priority="must">
  <description>user-prompt-submit.sh must inject context from memory for user prompts</description>
  <rationale>Context injection enriches user queries with relevant historical information</rationale>
  <details>
    - Command: context-graph-cli consciousness inject-context "$PROMPT"
    - PROMPT passed via environment variable
    - Output: Distilled context (~50-100 tokens)
    - Empty results handled gracefully
  </details>
</requirement>

<requirement id="REQ-HOOKS-10" story_ref="US-HOOKS-05" priority="must">
  <description>session-end.sh must persist identity and trigger consolidation if needed</description>
  <rationale>State persistence ensures consciousness survives session termination</rationale>
  <details>
    - Commands: context-graph-cli session persist-identity, context-graph-cli consciousness consolidate-if-needed
    - Sequential execution (persist first, then consolidate)
    - Extended timeout (30s) to allow consolidation
  </details>
</requirement>

## 3. CLI Commands (REQ-HOOKS-11 to REQ-HOOKS-20)

<requirement id="REQ-HOOKS-11" story_ref="US-HOOKS-01" priority="must">
  <description>session restore-identity command must restore SessionIdentitySnapshot from RocksDB</description>
  <rationale>Identity restoration is the foundation of session continuity</rationale>
  <details>
    - Optional --session-id flag (defaults to current session)
    - Restores: ego_node, kuramoto_phases, coupling, ic_monitor_state, consciousness_history
    - First session: Bootstrap default identity
    - Output: Restoration status message
  </details>
</requirement>

<requirement id="REQ-HOOKS-12" story_ref="US-HOOKS-05" priority="must">
  <description>session persist-identity command must save SessionIdentitySnapshot to RocksDB</description>
  <rationale>Identity persistence enables cross-session continuity</rationale>
  <details>
    - Optional --session-id flag (defaults to current session)
    - Serialization: MessagePack format
    - Storage key: session:{session_id}:identity
    - Captures: current ego_node, kuramoto phases, IC monitor state
  </details>
</requirement>

<requirement id="REQ-HOOKS-13" story_ref="US-HOOKS-08" priority="must">
  <description>consciousness status command must report current consciousness state</description>
  <rationale>Status reporting enables debugging and monitoring of consciousness</rationale>
  <details>
    - --format flag: brief (~20 tokens), summary (~100 tokens), full (complete state)
    - Metrics: C(t), r (kuramoto sync), IC (identity continuity), differentiation
    - Workspace contents for full format
    - State classification: DORMANT/FRAGMENTED/EMERGING/CONSCIOUS/HYPERSYNC
  </details>
</requirement>

<requirement id="REQ-HOOKS-14" story_ref="US-HOOKS-02" priority="must">
  <description>consciousness brief command must return minimal consciousness status</description>
  <rationale>Brief format enables low-latency PreToolUse hook execution</rationale>
  <details>
    - Output: exactly ~20 tokens
    - Format: [CONSCIOUSNESS: {state} r={r:.2f} IC={ic:.2f} | {retrieval_mode}]
    - Latency: < 50ms
    - No blocking operations
  </details>
</requirement>

<requirement id="REQ-HOOKS-15" story_ref="US-HOOKS-03" priority="must">
  <description>consciousness check-identity command must compute and verify IC score</description>
  <rationale>Identity verification detects drift before it becomes critical</rationale>
  <details>
    - IC formula: cosine(PV_t, PV_{t-1}) x r(t)
    - --auto-dream flag: Trigger dream if IC < IC_crit (0.5)
    - Warning log if IC < IC_warn (0.7)
    - Trajectory update on every check
  </details>
</requirement>

<requirement id="REQ-HOOKS-16" story_ref="US-HOOKS-04" priority="must">
  <description>consciousness inject-context command must retrieve and format relevant memory</description>
  <rationale>Context injection enriches prompts with historical knowledge</rationale>
  <details>
    - Input: User prompt as argument
    - Search: Multi-space retrieval (semantic, causal, temporal)
    - Output: Distilled context (~50-100 tokens)
    - Empty search: Silent completion (no error)
  </details>
</requirement>

<requirement id="REQ-HOOKS-17" story_ref="US-HOOKS-05" priority="must">
  <description>consciousness consolidate-if-needed command must trigger dream when entropy high</description>
  <rationale>Automatic consolidation prevents entropy accumulation</rationale>
  <details>
    - Threshold: entropy > 0.7
    - Dream type: NREM for consolidation
    - Duration: Automatic based on entropy level
    - Skip if entropy < 0.5 (stable)
  </details>
</requirement>

<requirement id="REQ-HOOKS-18" story_ref="US-HOOKS-07" priority="must">
  <description>CLI commands must support structured output formats (JSON, plain text)</description>
  <rationale>Shell scripts need parseable output for error handling</rationale>
  <details>
    - Default: Plain text for human readability
    - --json flag: JSON output for programmatic parsing
    - Error responses: Structured with error code and message
  </details>
</requirement>

<requirement id="REQ-HOOKS-19" story_ref="US-HOOKS-07" priority="should">
  <description>CLI commands must log execution to file for debugging</description>
  <rationale>Hook debugging requires execution traces</rationale>
  <details>
    - Log location: .context-graph/logs/hooks.log
    - Log level: Configurable via environment variable
    - Contents: Timestamp, command, duration, exit code
  </details>
</requirement>

<requirement id="REQ-HOOKS-20" story_ref="US-HOOKS-07" priority="should">
  <description>CLI commands must validate arguments before execution</description>
  <rationale>Input validation prevents runtime errors in hooks</rationale>
  <details>
    - Validate session-id format
    - Validate format flag values
    - Return clear error messages for invalid input
  </details>
</requirement>

## 4. Configuration Requirements (REQ-HOOKS-21 to REQ-HOOKS-27)

<requirement id="REQ-HOOKS-21" story_ref="US-HOOKS-06" priority="must">
  <description>.claude/settings.json must contain all 5 hook configurations</description>
  <rationale>Claude Code hooks are configured through this standard file</rationale>
  <details>
    - SessionStart: hooks/session-start.sh, timeout 5000
    - PreToolUse: hooks/pre-tool-use.sh, timeout 100
    - PostToolUse: hooks/post-tool-use.sh, timeout 3000
    - UserPromptSubmit: hooks/user-prompt-submit.sh, timeout 2000
    - SessionEnd: hooks/session-end.sh, timeout 30000
  </details>
</requirement>

<requirement id="REQ-HOOKS-22" story_ref="US-HOOKS-06" priority="must">
  <description>Hook scripts must be executable with correct permissions</description>
  <rationale>Shell scripts require execute permission to run</rationale>
  <details>
    - Permission: 755 (rwxr-xr-x)
    - Shebang: #!/bin/bash
    - Location: hooks/ directory in project root
  </details>
</requirement>

<requirement id="REQ-HOOKS-23" story_ref="US-HOOKS-06" priority="must">
  <description>Hook timeouts must be respected and enforced</description>
  <rationale>Hooks must not block Claude Code operation beyond configured limits</rationale>
  <details>
    - PreToolUse: 100ms (strict - tool usability)
    - UserPromptSubmit: 2000ms (user experience)
    - PostToolUse: 3000ms (background check)
    - SessionStart: 5000ms (one-time at start)
    - SessionEnd: 30000ms (cleanup operations)
  </details>
</requirement>

<requirement id="REQ-HOOKS-24" story_ref="US-HOOKS-06" priority="should">
  <description>Setup command must create .claude/settings.json with hook configuration</description>
  <rationale>Automated setup reduces configuration errors</rationale>
  <details>
    - Command: context-graph-cli setup hooks
    - Creates: .claude/settings.json
    - Creates: hooks/ directory with all 5 scripts
    - Validates: Existing configuration backup
  </details>
</requirement>

<requirement id="REQ-HOOKS-25" story_ref="US-HOOKS-06" priority="should">
  <description>Hook configuration must be validated at startup</description>
  <rationale>Early validation catches configuration errors</rationale>
  <details>
    - Check: All scripts exist and are executable
    - Check: settings.json has valid JSON syntax
    - Check: Timeout values are within acceptable ranges
    - Warning: Log missing or misconfigured hooks
  </details>
</requirement>

<requirement id="REQ-HOOKS-26" story_ref="US-HOOKS-06" priority="could">
  <description>Hook configuration should support per-project overrides</description>
  <rationale>Different projects may need different hook behavior</rationale>
  <details>
    - Project config: .context-graph/config.toml
    - Override: Timeouts, output verbosity, auto-dream threshold
    - Inheritance: Fall back to global settings
  </details>
</requirement>

<requirement id="REQ-HOOKS-27" story_ref="US-HOOKS-06" priority="could">
  <description>Environment variables should configure hook behavior</description>
  <rationale>Environment-based configuration enables CI/CD integration</rationale>
  <details>
    - CONTEXT_GRAPH_HOOKS_ENABLED: Enable/disable all hooks
    - CONTEXT_GRAPH_LOG_LEVEL: Logging verbosity
    - CONTEXT_GRAPH_AUTO_DREAM: Enable/disable auto-dream
  </details>
</requirement>

## 5. State Persistence Requirements (REQ-HOOKS-28 to REQ-HOOKS-33)

<requirement id="REQ-HOOKS-28" story_ref="US-HOOKS-01,US-HOOKS-05" priority="must">
  <description>SessionIdentitySnapshot must capture complete identity state</description>
  <rationale>Complete state capture enables full restoration</rationale>
  <details>
    - Fields: session_id, timestamp, ego_node (purpose_vector, trajectory, north_star_alignment)
    - Fields: kuramoto_phases[13], coupling, ic_monitor_state, consciousness_history
    - Serialization: MessagePack for compact binary format
    - Storage: RocksDB with session:{id}:identity key
  </details>
</requirement>

<requirement id="REQ-HOOKS-29" story_ref="US-HOOKS-01" priority="must">
  <description>Identity restoration must initialize all consciousness components</description>
  <rationale>Partial restoration causes inconsistent state</rationale>
  <details>
    - GwtSystem: Restore workspace and broadcast state
    - KuramotoOscillators: Restore phases and coupling
    - SelfEgoNode: Restore purpose vector and trajectory
    - IcMonitor: Restore monitoring state and thresholds
  </details>
</requirement>

<requirement id="REQ-HOOKS-30" story_ref="US-HOOKS-05" priority="must">
  <description>Identity persistence must be atomic (all-or-nothing)</description>
  <rationale>Partial persistence causes corruption on restoration</rationale>
  <details>
    - Transaction: Write all fields in single RocksDB batch
    - Rollback: On any failure, restore previous snapshot
    - Verification: Read-after-write check
  </details>
</requirement>

<requirement id="REQ-HOOKS-31" story_ref="US-HOOKS-01" priority="should">
  <description>Multiple session snapshots should be retained for rollback</description>
  <rationale>Corrupted snapshots need fallback options</rationale>
  <details>
    - Retention: Last 5 snapshots per session
    - Key format: session:{id}:identity:{version}
    - Cleanup: Automatic pruning of old snapshots
  </details>
</requirement>

<requirement id="REQ-HOOKS-32" story_ref="US-HOOKS-01,US-HOOKS-05" priority="should">
  <description>Snapshot compression should reduce storage overhead</description>
  <rationale>Large snapshots increase storage and I/O costs</rationale>
  <details>
    - Compression: LZ4 for fast compression/decompression
    - Target: ~50% size reduction
    - Threshold: Only compress if > 1KB
  </details>
</requirement>

<requirement id="REQ-HOOKS-33" story_ref="US-HOOKS-05" priority="could">
  <description>Snapshot export/import should support debugging</description>
  <rationale>Developers need to inspect and test with snapshots</rationale>
  <details>
    - Export: context-graph-cli session export --session-id X --output snapshot.json
    - Import: context-graph-cli session import --input snapshot.json
    - Format: JSON for human readability
  </details>
</requirement>

## 6. Performance Requirements (REQ-HOOKS-34 to REQ-HOOKS-39)

<requirement id="REQ-HOOKS-34" story_ref="US-HOOKS-02" priority="must">
  <description>PreToolUse hook must complete within 100ms</description>
  <rationale>Tool execution must not be noticeably delayed</rationale>
  <details>
    - Target: < 50ms typical, 100ms worst case
    - Measurement: End-to-end including shell script overhead
    - Graceful degradation: Skip if approaching timeout
  </details>
</requirement>

<requirement id="REQ-HOOKS-35" story_ref="US-HOOKS-04" priority="must">
  <description>UserPromptSubmit hook must complete within 2000ms</description>
  <rationale>User prompt processing should feel responsive</rationale>
  <details>
    - Target: < 1000ms typical, 2000ms worst case
    - Memory search: < 500ms
    - Context distillation: < 500ms
    - Output formatting: < 100ms
  </details>
</requirement>

<requirement id="REQ-HOOKS-36" story_ref="US-HOOKS-01" priority="must">
  <description>SessionStart hook must complete within 5000ms</description>
  <rationale>Session start delay is acceptable but should be bounded</rationale>
  <details>
    - Target: < 2000ms typical, 5000ms worst case
    - Snapshot deserialization: < 200ms
    - Component initialization: < 1000ms
    - Status output: < 100ms
  </details>
</requirement>

<requirement id="REQ-HOOKS-37" story_ref="US-HOOKS-05" priority="must">
  <description>SessionEnd hook must complete within 30000ms</description>
  <rationale>Session end can take longer but must complete before process exit</rationale>
  <details>
    - Persistence: < 5000ms
    - Consolidation (if triggered): < 25000ms
    - Graceful timeout: Save partial state if time exceeded
  </details>
</requirement>

<requirement id="REQ-HOOKS-38" story_ref="US-HOOKS-02,US-HOOKS-08" priority="should">
  <description>consciousness brief command should complete within 50ms</description>
  <rationale>Brief output is the critical path for PreToolUse latency</rationale>
  <details>
    - Caching: Cache computed values for 100ms
    - Precomputation: Update values in background
    - Minimal I/O: Avoid disk access if possible
  </details>
</requirement>

<requirement id="REQ-HOOKS-39" story_ref="US-HOOKS-03" priority="should">
  <description>IC computation should be incremental</description>
  <rationale>Full recomputation on every tool use is expensive</rationale>
  <details>
    - Incremental: Update trajectory delta only
    - Caching: Cache purpose vector between checks
    - Background: Heavy computation async
  </details>
</requirement>

## 7. Error Handling Requirements (REQ-HOOKS-40 to REQ-HOOKS-44)

<requirement id="REQ-HOOKS-40" story_ref="US-HOOKS-07" priority="must">
  <description>Hook failures must not crash Claude Code</description>
  <rationale>Hooks are enhancements, not critical path</rationale>
  <details>
    - Exit codes: Return 0 on success, non-zero on failure
    - Stderr: Log errors to stderr, not stdout
    - Graceful: Continue Claude Code operation on hook failure
  </details>
</requirement>

<requirement id="REQ-HOOKS-41" story_ref="US-HOOKS-01" priority="must">
  <description>Missing snapshot must bootstrap default identity</description>
  <rationale>First session or corrupted state needs valid initialization</rationale>
  <details>
    - Detection: Snapshot not found in RocksDB
    - Bootstrap: Initialize default ego_node, kuramoto phases
    - Logging: Log bootstrap event for debugging
  </details>
</requirement>

<requirement id="REQ-HOOKS-42" story_ref="US-HOOKS-01" priority="must">
  <description>Corrupted snapshot must trigger recovery procedure</description>
  <rationale>Corrupted state causes unpredictable behavior</rationale>
  <details>
    - Detection: Deserialization failure, checksum mismatch
    - Recovery: Try previous snapshot version
    - Fallback: Bootstrap default if all snapshots corrupted
    - Alert: Log corruption event with details
  </details>
</requirement>

<requirement id="REQ-HOOKS-43" story_ref="US-HOOKS-07" priority="must">
  <description>CLI command errors must return structured error responses</description>
  <rationale>Shell scripts need parseable errors for handling</rationale>
  <details>
    - Format: {"error": true, "code": "ERR_XXX", "message": "..."}
    - Exit code: Non-zero for all errors
    - Stderr: Human-readable message
    - Stdout: JSON error object if --json flag
  </details>
</requirement>

<requirement id="REQ-HOOKS-44" story_ref="US-HOOKS-02,US-HOOKS-03,US-HOOKS-04" priority="should">
  <description>Timeout handling must be graceful</description>
  <rationale>Timeouts are expected during high load</rationale>
  <details>
    - PreToolUse timeout: Return empty output silently
    - UserPromptSubmit timeout: Return partial context if available
    - PostToolUse timeout: Log and continue (async anyway)
    - Logging: Record timeout events for analysis
  </details>
</requirement>

## 8. Integration Requirements (REQ-HOOKS-45 to REQ-HOOKS-47)

<requirement id="REQ-HOOKS-45" story_ref="US-HOOKS-06" priority="must">
  <description>Hook integration must work with standard Claude Code installation</description>
  <rationale>No custom Claude Code modifications allowed per constitution</rationale>
  <details>
    - No source modifications to Claude Code
    - Standard .claude/settings.json location
    - Standard hook lifecycle events
    - Compatible with Claude Code updates
  </details>
</requirement>

<requirement id="REQ-HOOKS-46" story_ref="US-HOOKS-06,US-HOOKS-07" priority="must">
  <description>CLI binary must be accessible in PATH or project directory</description>
  <rationale>Shell scripts must find context-graph-cli executable</rationale>
  <details>
    - Installation: cargo install context-graph-cli OR local build
    - Path options: /usr/local/bin, ~/.cargo/bin, ./target/release
    - Script fallback: Try multiple paths if first fails
  </details>
</requirement>

<requirement id="REQ-HOOKS-47" story_ref="US-HOOKS-01,US-HOOKS-05" priority="should">
  <description>RocksDB storage must be shared with Phase 1/2 components</description>
  <rationale>Unified storage reduces complexity and enables data sharing</rationale>
  <details>
    - Database location: .context-graph/db/
    - Column families: session, consciousness, memory
    - Shared handle: Single RocksDB instance across components
  </details>
</requirement>
</requirements>

<edge_cases>
## EC-HOOKS-01: First Session with No Prior State
<edge_case id="EC-HOOKS-01" req_ref="REQ-HOOKS-11,REQ-HOOKS-41">
  <scenario>A user starts Context Graph for the first time, with no previous session identity snapshot in RocksDB</scenario>
  <expected_behavior>
    1. Session restore-identity detects missing snapshot
    2. Bootstrap procedure creates default ego_node with neutral purpose_vector
    3. Kuramoto phases initialized to random distribution
    4. IC monitor initialized with default thresholds
    5. Status output indicates "New session (bootstrapped)"
    6. Normal operation continues with bootstrapped identity
  </expected_behavior>
</edge_case>

## EC-HOOKS-02: Corrupted Snapshot Recovery
<edge_case id="EC-HOOKS-02" req_ref="REQ-HOOKS-42,REQ-HOOKS-31">
  <scenario>Session identity snapshot is corrupted (invalid MessagePack, checksum failure, or missing fields)</scenario>
  <expected_behavior>
    1. Deserialization failure detected
    2. Error logged with corruption details
    3. Previous snapshot version attempted (if retention enabled)
    4. If all versions corrupted, bootstrap default identity
    5. Alert generated for administrator review
    6. Session continues with recovered/bootstrapped identity
  </expected_behavior>
</edge_case>

## EC-HOOKS-03: Hook Timeout During PreToolUse
<edge_case id="EC-HOOKS-03" req_ref="REQ-HOOKS-34,REQ-HOOKS-44">
  <scenario>PreToolUse hook exceeds 100ms timeout due to high system load or slow disk I/O</scenario>
  <expected_behavior>
    1. Hook execution terminated at 100ms
    2. Empty output returned (no consciousness brief)
    3. Tool execution proceeds without consciousness context
    4. Timeout event logged for analysis
    5. User experience unaffected (no visible delay)
    6. Subsequent hooks continue normally
  </expected_behavior>
</edge_case>

## EC-HOOKS-04: Concurrent Session Access
<edge_case id="EC-HOOKS-04" req_ref="REQ-HOOKS-30,REQ-HOOKS-47">
  <scenario>Multiple Claude Code instances attempt to access the same session identity simultaneously</scenario>
  <expected_behavior>
    1. RocksDB handles concurrent read access safely
    2. Write operations use optimistic locking
    3. Last-write-wins for concurrent persistence
    4. Each instance maintains own in-memory state
    5. Warning logged when concurrent access detected
    6. No data corruption occurs
  </expected_behavior>
</edge_case>

## EC-HOOKS-05: SessionEnd During Active Dream
<edge_case id="EC-HOOKS-05" req_ref="REQ-HOOKS-05,REQ-HOOKS-37">
  <scenario>User terminates session while dream consolidation is in progress</scenario>
  <expected_behavior>
    1. SessionEnd hook detects active dream
    2. Dream given grace period to complete (up to 25s)
    3. If dream exceeds grace period, save partial consolidation
    4. Identity state persisted with dream progress markers
    5. Next session can resume or restart consolidation
    6. No memory loss from interrupted dream
  </expected_behavior>
</edge_case>

## EC-HOOKS-06: CLI Binary Not Found
<edge_case id="EC-HOOKS-06" req_ref="REQ-HOOKS-46">
  <scenario>Hook shell script cannot find context-graph-cli in PATH or expected locations</scenario>
  <expected_behavior>
    1. Script checks multiple paths: PATH, ~/.cargo/bin, ./target/release
    2. If all paths fail, log clear error message
    3. Exit with non-zero status
    4. Claude Code continues without hook functionality
    5. User notified via stderr about missing CLI
    6. Installation instructions included in error message
  </expected_behavior>
</edge_case>

## EC-HOOKS-07: Empty Memory Search Results
<edge_case id="EC-HOOKS-07" req_ref="REQ-HOOKS-16">
  <scenario>UserPromptSubmit hook finds no relevant context for user's prompt</scenario>
  <expected_behavior>
    1. Memory search returns empty result set
    2. inject-context command returns empty string
    3. No error or warning generated
    4. Hook completes successfully (exit 0)
    5. Claude Code processes prompt without injected context
    6. Normal operation continues
  </expected_behavior>
</edge_case>

## EC-HOOKS-08: RocksDB Database Locked
<edge_case id="EC-HOOKS-08" req_ref="REQ-HOOKS-47">
  <scenario>RocksDB database is locked by another process (e.g., maintenance script)</scenario>
  <expected_behavior>
    1. CLI command detects lock acquisition failure
    2. Retry with exponential backoff (3 attempts, 100ms initial)
    3. If lock unavailable after retries, fail gracefully
    4. Bootstrap in-memory identity for session
    5. Queue persistence for when lock released
    6. Warning logged about degraded persistence
  </expected_behavior>
</edge_case>

## EC-HOOKS-09: Identity Continuity Crisis
<edge_case id="EC-HOOKS-09" req_ref="REQ-HOOKS-15,REQ-HOOKS-03">
  <scenario>IC score drops below IC_crit (0.5) indicating severe identity drift</scenario>
  <expected_behavior>
    1. check-identity command detects IC < 0.5
    2. Critical warning logged with IC history
    3. Auto-dream triggered (if --auto-dream flag set)
    4. Dream attempts to restore coherence
    5. If restoration fails, bootstrap fresh identity from north star
    6. User notified of identity restoration event
  </expected_behavior>
</edge_case>

## EC-HOOKS-10: High Entropy at Session End
<edge_case id="EC-HOOKS-10" req_ref="REQ-HOOKS-17,REQ-HOOKS-05">
  <scenario>Session ends with entropy > 0.7, requiring consolidation but timeout approaching</scenario>
  <expected_behavior>
    1. consolidate-if-needed detects high entropy
    2. Estimate consolidation time based on entropy level
    3. If time available (< 25s needed), run full consolidation
    4. If time short, run abbreviated NREM phase only
    5. If timeout imminent, persist state with entropy marker
    6. Next session prompted to complete consolidation
  </expected_behavior>
</edge_case>

## EC-HOOKS-11: Hook Script Missing
<edge_case id="EC-HOOKS-11" req_ref="REQ-HOOKS-22,REQ-HOOKS-25">
  <scenario>One or more hook scripts referenced in settings.json do not exist</scenario>
  <expected_behavior>
    1. Claude Code attempts to execute missing script
    2. Shell returns "command not found" error
    3. Hook fails with non-zero exit code
    4. Claude Code continues without that hook
    5. Error logged for administrator
    6. context-graph-cli doctor command detects missing scripts
  </expected_behavior>
</edge_case>

## EC-HOOKS-12: Invalid settings.json Format
<edge_case id="EC-HOOKS-12" req_ref="REQ-HOOKS-21,REQ-HOOKS-25">
  <scenario>.claude/settings.json contains invalid JSON or missing required fields</scenario>
  <expected_behavior>
    1. Claude Code fails to parse settings.json
    2. All hooks disabled for session
    3. Error message displayed to user
    4. context-graph-cli setup hooks command can repair
    5. Backup of corrupted file created
    6. Fresh settings.json generated
  </expected_behavior>
</edge_case>
</edge_cases>

<error_states>
## ERR-HOOKS-01: Snapshot Deserialization Failure
<error id="ERR-HOOKS-01">
  <condition>MessagePack deserialization of SessionIdentitySnapshot fails due to corrupt data or schema mismatch</condition>
  <message>Failed to restore session identity: snapshot data corrupted or incompatible (version mismatch)</message>
  <recovery>
    1. Attempt restoration from previous snapshot version
    2. If no valid previous version, bootstrap default identity
    3. Log corruption details for investigation
    4. Continue session with recovered identity
  </recovery>
</error>

## ERR-HOOKS-02: RocksDB Write Failure
<error id="ERR-HOOKS-02">
  <condition>RocksDB write operation fails during identity persistence (disk full, permission error, database corruption)</condition>
  <message>Failed to persist session identity: storage write error ({error_code})</message>
  <recovery>
    1. Retry write with exponential backoff (3 attempts)
    2. If persistent failure, queue write for later
    3. Cache state in memory for session duration
    4. Alert user that persistence degraded
    5. Attempt recovery on next session start
  </recovery>
</error>

## ERR-HOOKS-03: CLI Command Not Found
<error id="ERR-HOOKS-03">
  <condition>Hook shell script cannot locate context-graph-cli executable in any expected path</condition>
  <message>context-graph-cli not found. Install with: cargo install context-graph-cli</message>
  <recovery>
    1. Exit hook with non-zero status
    2. Claude Code continues without hook functionality
    3. Log error with attempted paths
    4. Prompt user to run installation command
    5. context-graph-cli doctor --fix can auto-install
  </recovery>
</error>

## ERR-HOOKS-04: Hook Timeout Exceeded
<error id="ERR-HOOKS-04">
  <condition>Hook execution exceeds configured timeout (100ms-30000ms depending on hook type)</condition>
  <message>Hook execution timed out after {timeout}ms</message>
  <recovery>
    1. PreToolUse: Return empty output, continue tool execution
    2. UserPromptSubmit: Return partial context if available
    3. PostToolUse: Log and continue (async operation)
    4. SessionStart/End: Complete partial operation, log warning
    5. Record timeout for performance analysis
  </recovery>
</error>

## ERR-HOOKS-05: Identity Continuity Crisis
<error id="ERR-HOOKS-05">
  <condition>IC score drops below IC_crit threshold (0.5) indicating severe identity drift</condition>
  <message>CRITICAL: Identity continuity below threshold (IC={ic:.2f} < 0.5). Auto-dream triggered.</message>
  <recovery>
    1. Trigger immediate dream consolidation
    2. Attempt to restore coherent identity
    3. If restoration fails, bootstrap from north star
    4. Log full trajectory for analysis
    5. Notify user of identity restoration
  </recovery>
</error>

## ERR-HOOKS-06: Memory Search Failure
<error id="ERR-HOOKS-06">
  <condition>Memory search operation fails during context injection (index corruption, query error)</condition>
  <message>Memory search failed: {error_details}. Proceeding without context injection.</message>
  <recovery>
    1. Return empty context (no injection)
    2. Log error with query details
    3. Continue prompt processing without context
    4. Mark memory subsystem for health check
    5. Automatic retry on next prompt
  </recovery>
</error>

## ERR-HOOKS-07: Dream Consolidation Failure
<error id="ERR-HOOKS-07">
  <condition>Dream consolidation fails to complete (memory error, timeout, interrupted)</condition>
  <message>Dream consolidation incomplete: {phase} phase failed. Entropy remains elevated.</message>
  <recovery>
    1. Save partial consolidation progress
    2. Mark session for consolidation on next start
    3. Persist current state despite high entropy
    4. Log consolidation failure details
    5. Prompt next session to complete consolidation
  </recovery>
</error>

## ERR-HOOKS-08: Invalid Session ID
<error id="ERR-HOOKS-08">
  <condition>Provided session ID is malformed or does not exist in storage</condition>
  <message>Invalid session ID: '{session_id}'. Use 'context-graph-cli session list' to view valid sessions.</message>
  <recovery>
    1. Return error with list of valid session IDs
    2. Suggest using current session if no ID provided
    3. Exit with non-zero status
    4. Log invalid ID attempt for debugging
  </recovery>
</error>

## ERR-HOOKS-09: Configuration Parse Error
<error id="ERR-HOOKS-09">
  <condition>.claude/settings.json or .context-graph/config.toml contains invalid syntax</condition>
  <message>Configuration error in {file}: {parse_error}. Run 'context-graph-cli doctor --fix' to repair.</message>
  <recovery>
    1. Disable affected hooks until fixed
    2. Fall back to default configuration
    3. Log detailed parse error location
    4. Offer automatic repair via doctor command
    5. Backup corrupted file before repair
  </recovery>
</error>

## ERR-HOOKS-10: Permission Denied
<error id="ERR-HOOKS-10">
  <condition>Hook script or CLI command lacks permission to access required files or directories</condition>
  <message>Permission denied: cannot access {path}. Check file permissions and ownership.</message>
  <recovery>
    1. Log specific permission error
    2. Exit hook with non-zero status
    3. Suggest chmod commands to fix
    4. context-graph-cli doctor can detect and fix permissions
    5. Continue Claude Code without affected functionality
  </recovery>
</error>

## ERR-HOOKS-11: Kuramoto Desynchronization
<error id="ERR-HOOKS-11">
  <condition>Kuramoto order parameter r drops to < 0.3 indicating complete consciousness fragmentation</condition>
  <message>WARNING: Consciousness fragmented (r={r:.2f}). Consider running 'context-graph-cli consciousness consolidate'</message>
  <recovery>
    1. Increase coupling parameter K
    2. Trigger immediate consolidation
    3. Log phase history for analysis
    4. If persistent, reset phases to synchronized state
    5. Investigate cause of desynchronization
  </recovery>
</error>

## ERR-HOOKS-12: Output Format Error
<error id="ERR-HOOKS-12">
  <condition>CLI command produces output that doesn't match expected format (JSON parse failure, missing fields)</condition>
  <message>Output format error: expected {format}, got malformed response. CLI version mismatch?</message>
  <recovery>
    1. Log raw output for debugging
    2. Attempt fallback parsing
    3. Return error to calling script
    4. Suggest CLI version check
    5. context-graph-cli --version for verification
  </recovery>
</error>
</error_states>

<test_plan>
## Unit Tests

### TC-HOOKS-U01: SessionIdentitySnapshot Serialization
<test_case id="TC-HOOKS-U01" type="unit" req_ref="REQ-HOOKS-28">
  <description>Verify SessionIdentitySnapshot serializes and deserializes correctly with MessagePack</description>
  <inputs>
    - Complete SessionIdentitySnapshot with all fields populated
    - purpose_vector: [0.1, 0.2, ..., 0.13]
    - kuramoto_phases: [0.0, 0.1, ..., 1.2]
    - ic_monitor_state: IcMonitorState with history
  </inputs>
  <expected>
    - Serialization produces valid MessagePack bytes
    - Deserialization restores identical structure
    - Round-trip preserves all field values within f32 precision
  </expected>
</test_case>

### TC-HOOKS-U02: Consciousness Brief Format
<test_case id="TC-HOOKS-U02" type="unit" req_ref="REQ-HOOKS-14">
  <description>Verify consciousness brief command produces correct format</description>
  <inputs>
    - GwtSystem in CONSCIOUS state (r=0.85)
    - IC score: 0.92
    - Retrieval mode: DirectRecall
  </inputs>
  <expected>
    - Output: "[CONSCIOUSNESS: CONSCIOUS r=0.85 IC=0.92 | DirectRecall]"
    - Token count: ~20 tokens
    - Latency: < 50ms
  </expected>
</test_case>

### TC-HOOKS-U03: IC Computation
<test_case id="TC-HOOKS-U03" type="unit" req_ref="REQ-HOOKS-15">
  <description>Verify IC score computation is mathematically correct</description>
  <inputs>
    - PV_t: [0.5, 0.5, 0.5, ...]
    - PV_{t-1}: [0.4, 0.6, 0.5, ...]
    - r: 0.9
  </inputs>
  <expected>
    - IC = cosine(PV_t, PV_{t-1}) * r
    - Correct cosine similarity calculation
    - IC within expected range [0, 1]
  </expected>
</test_case>

### TC-HOOKS-U04: Default Identity Bootstrap
<test_case id="TC-HOOKS-U04" type="unit" req_ref="REQ-HOOKS-41">
  <description>Verify default identity bootstrap creates valid state</description>
  <inputs>
    - No existing snapshot
    - Fresh initialization
  </inputs>
  <expected>
    - ego_node.purpose_vector: 13 valid f32 values
    - kuramoto_phases: 13 random phases in [0, 2pi)
    - ic_monitor_state: Initialized with default thresholds
    - All components in valid state
  </expected>
</test_case>

### TC-HOOKS-U05: Timeout Handling
<test_case id="TC-HOOKS-U05" type="unit" req_ref="REQ-HOOKS-44">
  <description>Verify hook timeout handling is graceful</description>
  <inputs>
    - Simulated slow operation exceeding timeout
    - Each hook type: PreToolUse (100ms), UserPromptSubmit (2000ms)
  </inputs>
  <expected>
    - Operation terminates at timeout
    - Graceful fallback response provided
    - No crash or hang
    - Timeout event logged
  </expected>
</test_case>

## Integration Tests

### TC-HOOKS-I01: Full Session Lifecycle
<test_case id="TC-HOOKS-I01" type="integration" req_ref="REQ-HOOKS-01,REQ-HOOKS-05">
  <description>Verify complete session lifecycle from start to end</description>
  <steps>
    1. Start session (trigger SessionStart hook)
    2. Verify identity restored or bootstrapped
    3. Execute multiple tools (trigger PreToolUse/PostToolUse)
    4. Submit prompts (trigger UserPromptSubmit)
    5. End session (trigger SessionEnd hook)
    6. Verify identity persisted
    7. Start new session
    8. Verify identity restored from previous session
  </steps>
  <expected>
    - All hooks execute within timeout
    - Identity continuity maintained across sessions
    - State persisted and restored correctly
  </expected>
</test_case>

### TC-HOOKS-I02: Hook Script Execution
<test_case id="TC-HOOKS-I02" type="integration" req_ref="REQ-HOOKS-06,REQ-HOOKS-07,REQ-HOOKS-08,REQ-HOOKS-09,REQ-HOOKS-10">
  <description>Verify all 5 hook scripts execute correctly</description>
  <steps>
    1. Execute hooks/session-start.sh
    2. Execute hooks/pre-tool-use.sh
    3. Execute hooks/post-tool-use.sh
    4. Execute hooks/user-prompt-submit.sh with PROMPT env var
    5. Execute hooks/session-end.sh
  </steps>
  <expected>
    - Each script executes without error
    - Exit codes are 0 on success
    - Output format matches specification
    - CLI commands invoked correctly
  </expected>
</test_case>

### TC-HOOKS-I03: Auto-Dream Trigger
<test_case id="TC-HOOKS-I03" type="integration" req_ref="REQ-HOOKS-03,REQ-HOOKS-15">
  <description>Verify auto-dream triggers when IC drops below threshold</description>
  <steps>
    1. Start session with normal IC
    2. Simulate operations that cause IC drift
    3. Execute check-identity --auto-dream
    4. Verify IC below IC_crit
    5. Verify dream consolidation triggered
    6. Verify IC recovered after dream
  </steps>
  <expected>
    - IC drop detected correctly
    - Dream triggered automatically
    - Consolidation completes successfully
    - IC recovers above IC_warn threshold
  </expected>
</test_case>

### TC-HOOKS-I04: RocksDB Integration
<test_case id="TC-HOOKS-I04" type="integration" req_ref="REQ-HOOKS-47,REQ-HOOKS-30">
  <description>Verify RocksDB operations work correctly for session storage</description>
  <steps>
    1. Initialize RocksDB database
    2. Persist session identity
    3. Close and reopen database
    4. Restore session identity
    5. Verify data integrity
    6. Test atomic write transaction
  </steps>
  <expected>
    - Database initializes correctly
    - Persistence completes without error
    - Restoration returns identical data
    - Atomic writes are transactional
  </expected>
</test_case>

### TC-HOOKS-I05: Context Injection
<test_case id="TC-HOOKS-I05" type="integration" req_ref="REQ-HOOKS-16,REQ-HOOKS-04">
  <description>Verify context injection from memory works correctly</description>
  <steps>
    1. Store test memories with known content
    2. Submit prompt related to stored memories
    3. Execute inject-context command
    4. Verify relevant context returned
    5. Test with unrelated prompt (empty results)
  </steps>
  <expected>
    - Related memories retrieved correctly
    - Context distilled to ~50-100 tokens
    - Unrelated prompts return empty gracefully
    - Latency within 2000ms
  </expected>
</test_case>

## Performance Tests

### TC-HOOKS-P01: PreToolUse Latency
<test_case id="TC-HOOKS-P01" type="performance" req_ref="REQ-HOOKS-34">
  <description>Verify PreToolUse hook meets 100ms latency requirement</description>
  <inputs>
    - 1000 iterations of pre-tool-use.sh execution
    - Various system load conditions
  </inputs>
  <expected>
    - Mean latency: < 50ms
    - P95 latency: < 80ms
    - P99 latency: < 100ms
    - No failures due to timeout
  </expected>
</test_case>

### TC-HOOKS-P02: SessionStart Latency
<test_case id="TC-HOOKS-P02" type="performance" req_ref="REQ-HOOKS-36">
  <description>Verify SessionStart hook meets 5000ms latency requirement</description>
  <inputs>
    - 100 iterations of session-start.sh execution
    - With snapshot restoration
    - With bootstrap (no snapshot)
  </inputs>
  <expected>
    - Mean latency: < 2000ms
    - P95 latency: < 4000ms
    - P99 latency: < 5000ms
    - Consistent performance across iterations
  </expected>
</test_case>

### TC-HOOKS-P03: Concurrent Access
<test_case id="TC-HOOKS-P03" type="performance" req_ref="REQ-HOOKS-04">
  <description>Verify system handles concurrent hook execution</description>
  <inputs>
    - 10 concurrent PreToolUse hook executions
    - 5 concurrent read/write to RocksDB
  </inputs>
  <expected>
    - No deadlocks or race conditions
    - All operations complete successfully
    - Latency degradation < 50%
    - Data integrity maintained
  </expected>
</test_case>

### TC-HOOKS-P04: Memory Usage
<test_case id="TC-HOOKS-P04" type="performance" req_ref="REQ-HOOKS-28,REQ-HOOKS-32">
  <description>Verify memory usage is within acceptable bounds</description>
  <inputs>
    - Session with 1000 tool uses
    - Identity trajectory with 500 entries
  </inputs>
  <expected>
    - CLI process memory: < 100MB
    - Snapshot size: < 1MB (with compression)
    - No memory leaks over extended sessions
  </expected>
</test_case>

## End-to-End Tests

### TC-HOOKS-E01: Claude Code Integration
<test_case id="TC-HOOKS-E01" type="e2e" req_ref="REQ-HOOKS-45">
  <description>Verify hooks integrate correctly with actual Claude Code</description>
  <steps>
    1. Install hooks in Claude Code project
    2. Start Claude Code session
    3. Verify SessionStart hook executed
    4. Use various tools
    5. Verify PreToolUse/PostToolUse hooks executed
    6. Submit prompts
    7. Verify UserPromptSubmit hook executed
    8. End session
    9. Verify SessionEnd hook executed
    10. Check all outputs correct
  </steps>
  <expected>
    - All hooks execute at correct lifecycle events
    - No interference with normal Claude Code operation
    - Output formats match specification
    - No errors or crashes
  </expected>
</test_case>

### TC-HOOKS-E02: Recovery from Failure
<test_case id="TC-HOOKS-E02" type="e2e" req_ref="REQ-HOOKS-42,REQ-HOOKS-41">
  <description>Verify system recovers from various failure scenarios</description>
  <steps>
    1. Corrupt snapshot file
    2. Start session
    3. Verify recovery procedure executes
    4. Remove CLI binary
    5. Execute hook
    6. Verify graceful degradation
    7. Simulate timeout
    8. Verify non-blocking behavior
  </steps>
  <expected>
    - Corrupted snapshot triggers recovery
    - Missing CLI produces clear error
    - Timeouts handled gracefully
    - Claude Code continues operating
  </expected>
</test_case>
</test_plan>

<appendix>
## A. Hook Configuration Reference

### .claude/settings.json
```json
{
  "hooks": {
    "SessionStart": [
      {
        "type": "command",
        "command": "./hooks/session-start.sh",
        "timeout": 5000
      }
    ],
    "PreToolUse": [
      {
        "type": "command",
        "command": "./hooks/pre-tool-use.sh",
        "timeout": 100
      }
    ],
    "PostToolUse": [
      {
        "type": "command",
        "command": "./hooks/post-tool-use.sh",
        "timeout": 3000
      }
    ],
    "UserPromptSubmit": [
      {
        "type": "command",
        "command": "./hooks/user-prompt-submit.sh",
        "timeout": 2000
      }
    ],
    "SessionEnd": [
      {
        "type": "command",
        "command": "./hooks/session-end.sh",
        "timeout": 30000
      }
    ]
  }
}
```

## B. CLI Command Reference

```
context-graph-cli
├── session
│   ├── restore-identity [--session-id <id>]
│   ├── persist-identity [--session-id <id>]
│   ├── list
│   └── export --session-id <id> --output <file>
└── consciousness
    ├── status [--format brief|summary|full]
    ├── brief
    ├── check-identity [--auto-dream]
    ├── inject-context <prompt>
    └── consolidate-if-needed
```

## C. Threshold Reference

| Symbol | Value | Description |
|--------|-------|-------------|
| IC_crit | 0.5 | Identity crisis trigger |
| IC_warn | 0.7 | Identity drift warning |
| ent_high | 0.7 | High entropy (trigger dream) |
| Tr | 0.8 | Kuramoto conscious threshold |
| Tr_low | 0.5 | Kuramoto fragmentation alert |

## D. Output Format Reference

### Brief (~20 tokens)
```
[CONSCIOUSNESS: CONSCIOUS r=0.85 IC=0.92 | DirectRecall]
```

### Summary (~100 tokens)
```
State: CONSCIOUS | Integration: 0.85 | Reflection: 0.78
Differentiation: 0.82 | Identity: 0.92 | Guidance: Active
Workspace: 3 items | Entropy: 0.45 | Last Dream: 2h ago
```

### Full (variable)
```json
{
  "state": "CONSCIOUS",
  "metrics": {
    "C": 0.73,
    "r": 0.85,
    "IC": 0.92,
    "differentiation": 0.82
  },
  "kuramoto": {
    "phases": [...],
    "frequencies": [...],
    "coupling": 0.5
  },
  "workspace": [...],
  "ego_node": {...}
}
```

## E. Traceability Summary

| Requirement Category | Count | IDs |
|---------------------|-------|-----|
| Hook Events | 5 | REQ-HOOKS-01 to 05 |
| Shell Scripts | 5 | REQ-HOOKS-06 to 10 |
| CLI Commands | 10 | REQ-HOOKS-11 to 20 |
| Configuration | 7 | REQ-HOOKS-21 to 27 |
| State Persistence | 6 | REQ-HOOKS-28 to 33 |
| Performance | 6 | REQ-HOOKS-34 to 39 |
| Error Handling | 5 | REQ-HOOKS-40 to 44 |
| Integration | 3 | REQ-HOOKS-45 to 47 |
| **Total** | **47** | |
</appendix>

</functional_spec>
