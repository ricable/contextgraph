# Functional Specification: Phase 6 - CLI & Hooks Integration

```xml
<functional_spec id="SPEC-PHASE6" version="1.0">
<metadata>
  <title>CLI & Hooks Integration</title>
  <status>approved</status>
  <owner>Context Graph Team</owner>
  <created>2026-01-16</created>
  <last_updated>2026-01-16</last_updated>
  <implements>impplan.md Part 1 (Hook Lifecycle), Part 9 (Phase 6)</implements>
  <depends_on>
    <spec_ref>SPEC-PHASE0</spec_ref>
    <spec_ref>SPEC-PHASE1</spec_ref>
    <spec_ref>SPEC-PHASE2</spec_ref>
    <spec_ref>SPEC-PHASE3</spec_ref>
    <spec_ref>SPEC-PHASE4</spec_ref>
    <spec_ref>SPEC-PHASE5</spec_ref>
  </depends_on>
</metadata>

<overview>
Provide a CLI binary (`context-graph-cli`) and shell script hooks that integrate contextgraph with Claude Code's hook system. This is the user-facing interface that ties together all previous phases.

The CLI exposes commands that the hook scripts invoke. The hooks are configured in `.claude/settings.json` and execute at specific points in Claude Code's lifecycle. This architecture provides:

1. **Capture** - Memory creation from hook descriptions, Claude responses, and session events
2. **Injection** - Context injection via stdout that Claude Code reads as <system-reminder> content
3. **Session Management** - Session start/end tracking for scoping recent memories

**Problem Solved**: Claude Code has no persistent memory between sessions. The hook system enables autonomous context injection without manual user intervention.

**Who Benefits**: End users who want Claude Code to automatically remember and surface relevant context across sessions without explicit memory management.
</overview>

<user_stories>
<story id="US-P6-01" priority="must-have">
  <narrative>
    As a Claude Code user
    I want hooks to automatically capture my work context
    So that the system builds memory without manual intervention
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P6-01-01">
      <given>Hook scripts are configured in .claude/settings.json</given>
      <when>Claude Code triggers any hook event</when>
      <then>The corresponding hook script executes and calls context-graph-cli</then>
    </criterion>
    <criterion id="AC-P6-01-02">
      <given>A PostToolUse hook fires</given>
      <when>The hook script runs</when>
      <then>A memory is created from the tool description</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P6-02" priority="must-have">
  <narrative>
    As a Claude Code session
    I want relevant context injected automatically
    So that Claude has access to related memories without user prompting
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P6-02-01">
      <given>User submits a prompt</given>
      <when>UserPromptSubmit hook fires</when>
      <then>Related memories and divergence alerts are printed to stdout</then>
    </criterion>
    <criterion id="AC-P6-02-02">
      <given>CLI prints context to stdout</given>
      <when>Claude Code reads the hook output</when>
      <then>Context appears in Claude's system prompt as reminder</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P6-03" priority="must-have">
  <narrative>
    As a Claude Code session
    I want session boundaries tracked
    So that "recent" memories are scoped correctly
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P6-03-01">
      <given>A new Claude Code session starts</given>
      <when>SessionStart hook fires</when>
      <then>New session_id is generated and stored</then>
    </criterion>
    <criterion id="AC-P6-03-02">
      <given>A Claude Code session ends</given>
      <when>SessionEnd hook fires</when>
      <then>Session is marked complete with end timestamp</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P6-04" priority="must-have">
  <narrative>
    As a user installing contextgraph
    I want a single setup command
    So that I don't have to manually configure hooks
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P6-04-01">
      <given>context-graph-cli is installed</given>
      <when>User runs `context-graph-cli setup`</when>
      <then>.claude/settings.json is created/updated with hook configuration</then>
    </criterion>
    <criterion id="AC-P6-04-02">
      <given>Setup command runs</given>
      <when>Hook scripts don't exist</when>
      <then>Hook scripts are created in ./hooks/ directory</then>
    </criterion>
  </acceptance_criteria>
</story>
</user_stories>

<requirements>
<requirement id="REQ-P6-01" story_ref="US-P6-01" priority="must">
  <description>Implement context-graph-cli binary with subcommands</description>
  <rationale>CLI is the interface between hook scripts and the contextgraph system</rationale>
  <commands>
    <command name="session">
      <subcommand name="start">Start new session, output session_id</subcommand>
      <subcommand name="end">End current session, capture summary</subcommand>
    </command>
    <command name="inject-context">
      <description>Query memories and output injection context to stdout</description>
      <args>--query TEXT, --session-id UUID, --budget INT</args>
      <output>Formatted context for Claude Code</output>
    </command>
    <command name="inject-brief">
      <description>Quick context injection for PreToolUse (lower latency)</description>
      <args>--query TEXT, --session-id UUID, --budget INT</args>
      <output>Brief context (max 200 tokens)</output>
    </command>
    <command name="capture-memory">
      <description>Create memory from hook description</description>
      <args>--content TEXT, --source hook|response, --session-id UUID</args>
    </command>
    <command name="capture-response">
      <description>Capture Claude's response at Stop event</description>
      <args>--content TEXT, --session-id UUID</args>
    </command>
    <command name="setup">
      <description>Configure hooks in .claude/settings.json</description>
      <args>--force (overwrite existing)</args>
    </command>
    <command name="status">
      <description>Show system status (memory count, active session, etc.)</description>
    </command>
  </commands>
</requirement>

<requirement id="REQ-P6-02" story_ref="US-P6-01" priority="must">
  <description>Create hook shell scripts that invoke CLI commands</description>
  <rationale>Shell scripts bridge Claude Code hook events to CLI commands</rationale>
  <scripts>
    <script name="session-start.sh">
      <triggers>SessionStart</triggers>
      <actions>
        1. Call `context-graph-cli session start`
        2. Call `context-graph-cli inject-context --query "$CLAUDE_CONTEXT"`
        3. Output portfolio summary + recent divergences
      </actions>
    </script>
    <script name="user-prompt-submit.sh">
      <triggers>UserPromptSubmit</triggers>
      <actions>
        1. Read user prompt from environment
        2. Call `context-graph-cli inject-context --query "$USER_PROMPT"`
        3. Output similar memories + divergence alerts
      </actions>
    </script>
    <script name="pre-tool-use.sh">
      <triggers>PreToolUse (matcher: Edit|Write|Bash)</triggers>
      <actions>
        1. Read tool info from environment
        2. Call `context-graph-cli inject-brief --query "$TOOL_DESCRIPTION"`
        3. Output brief relevant context
      </actions>
    </script>
    <script name="post-tool-use.sh">
      <triggers>PostToolUse (matcher: *)</triggers>
      <actions>
        1. Read tool description from environment
        2. Call `context-graph-cli capture-memory --content "$TOOL_DESCRIPTION" --source hook`
        3. No stdout output (capture only)
      </actions>
    </script>
    <script name="stop.sh">
      <triggers>Stop</triggers>
      <actions>
        1. Read Claude's response summary from environment
        2. Call `context-graph-cli capture-response --content "$RESPONSE_SUMMARY"`
        3. No stdout output (capture only)
      </actions>
    </script>
    <script name="session-end.sh">
      <triggers>SessionEnd</triggers>
      <actions>
        1. Read session summary from environment
        2. Call `context-graph-cli capture-memory --content "$SESSION_SUMMARY" --source hook`
        3. Call `context-graph-cli session end`
        4. No stdout output (capture only)
      </actions>
    </script>
  </scripts>
</requirement>

<requirement id="REQ-P6-03" story_ref="US-P6-02" priority="must">
  <description>Output format must be Claude Code compatible</description>
  <rationale>Claude Code reads hook stdout and injects as system-reminder</rationale>
  <format>
    <output_rules>
      - Plain text output (no XML tags in output itself)
      - Markdown formatting allowed
      - Claude Code wraps output in &lt;system-reminder&gt; tags automatically
      - Empty output = no injection
    </output_rules>
    <example_output>
## Relevant Context

### Recent Related Work
- Yesterday: Implemented HDBSCAN clustering in multi_space.rs
- 3 days ago: Fixed embedding dimension validation

### Potentially Related
- Rust error handling patterns (2 weeks ago)

### Note: Activity Shift Detected
Your current query about "testing" has low similarity to recent work on "clustering"
Recent context: HDBSCAN implementation and BIRCH integration
    </example_output>
  </format>
</requirement>

<requirement id="REQ-P6-04" story_ref="US-P6-03" priority="must">
  <description>Session management with persistence</description>
  <rationale>Sessions scope "recent" memories and track activity boundaries</rationale>
  <behavior>
    <session_start>
      1. Generate UUID for session_id
      2. Store session record: {id, start_time, status: "active"}
      3. Write session_id to ~/.contextgraph/current_session
      4. Output session_id to stdout
    </session_start>
    <session_end>
      1. Read current session_id from file
      2. Update session record: {end_time, status: "completed"}
      3. Clear current_session file
      4. Mark all session memories as belonging to this session
    </session_end>
  </behavior>
</requirement>

<requirement id="REQ-P6-05" story_ref="US-P6-04" priority="must">
  <description>Setup command configures .claude/settings.json</description>
  <rationale>Single command for user onboarding</rationale>
  <settings_json>
{
  "hooks": {
    "SessionStart": [{"hooks": [{"type": "command", "command": "./hooks/session-start.sh", "timeout": 5000}]}],
    "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "./hooks/user-prompt-submit.sh", "timeout": 2000}]}],
    "PreToolUse": [{"matcher": "Edit|Write|Bash", "hooks": [{"type": "command", "command": "./hooks/pre-tool-use.sh", "timeout": 500}]}],
    "PostToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": "./hooks/post-tool-use.sh", "timeout": 3000}]}],
    "Stop": [{"hooks": [{"type": "command", "command": "./hooks/stop.sh", "timeout": 3000}]}],
    "SessionEnd": [{"hooks": [{"type": "command", "command": "./hooks/session-end.sh", "timeout": 30000}]}]
  }
}
  </settings_json>
  <behavior>
    1. Check if .claude/settings.json exists
    2. If exists, merge hooks config (preserve other settings)
    3. If not exists, create with hooks config
    4. Create ./hooks/ directory if not exists
    5. Write all hook scripts with executable permissions
    6. Verify scripts are executable
  </behavior>
</requirement>

<requirement id="REQ-P6-06" story_ref="US-P6-01" priority="must">
  <description>Hook timeouts must be respected</description>
  <rationale>Claude Code kills hooks that exceed timeout</rationale>
  <timeouts>
    | Hook | Timeout | Design Constraint |
    |------|---------|-------------------|
    | SessionStart | 5000ms | Can do full retrieval |
    | UserPromptSubmit | 2000ms | Must be fast for UX |
    | PreToolUse | 500ms | Minimal latency allowed |
    | PostToolUse | 3000ms | Capture can be slower |
    | Stop | 3000ms | Capture can be slower |
    | SessionEnd | 30000ms | Can do full processing |
  </timeouts>
  <implementation>
    - PreToolUse uses inject-brief with smaller budget
    - UserPromptSubmit limits retrieval to top-10
    - SessionEnd can batch process pending captures
  </implementation>
</requirement>

<requirement id="REQ-P6-07" story_ref="US-P6-02" priority="must">
  <description>CLI reads environment variables set by Claude Code</description>
  <rationale>Claude Code passes context via environment variables</rationale>
  <env_vars>
    | Variable | Hook | Content |
    |----------|------|---------|
    | CLAUDE_SESSION_ID | All | Current session identifier |
    | CLAUDE_CONTEXT | SessionStart | Initial context |
    | USER_PROMPT | UserPromptSubmit | User's prompt text |
    | TOOL_NAME | PreToolUse, PostToolUse | Tool being used |
    | TOOL_DESCRIPTION | PreToolUse, PostToolUse | Description of tool action |
    | TOOL_INPUT | PostToolUse | Tool input parameters |
    | TOOL_OUTPUT | PostToolUse | Tool output (truncated) |
    | RESPONSE_SUMMARY | Stop | Claude's response summary |
    | SESSION_SUMMARY | SessionEnd | Full session summary |
  </env_vars>
</requirement>

<requirement id="REQ-P6-08" story_ref="US-P6-01" priority="should">
  <description>CLI provides debug/verbose mode</description>
  <rationale>Troubleshooting hook issues requires visibility</rationale>
  <behavior>
    - --verbose flag logs to stderr (not stdout to avoid injection)
    - Logs: command received, query embedding time, retrieval time, results found
    - Logs to ~/.contextgraph/logs/cli.log
    - Log rotation: 7 days, max 10MB per file
  </behavior>
</requirement>
</requirements>

<edge_cases>
<edge_case id="EC-P6-01" req_ref="REQ-P6-05">
  <scenario>.claude/settings.json has invalid JSON</scenario>
  <expected_behavior>Setup fails with error: "Invalid JSON in .claude/settings.json at line X". Suggest manual fix or --force to overwrite.</expected_behavior>
</edge_case>

<edge_case id="EC-P6-02" req_ref="REQ-P6-06">
  <scenario>Hook times out during retrieval</scenario>
  <expected_behavior>CLI catches SIGTERM, outputs partial results if available, logs "Timeout after Xms, returning partial results".</expected_behavior>
</edge_case>

<edge_case id="EC-P6-03" req_ref="REQ-P6-04">
  <scenario>Session end called without active session</scenario>
  <expected_behavior>Warning logged: "No active session to end". Command completes successfully (idempotent).</expected_behavior>
</edge_case>

<edge_case id="EC-P6-04" req_ref="REQ-P6-07">
  <scenario>Required environment variable missing</scenario>
  <expected_behavior>CLI continues with empty value. Warning logged: "ENV_VAR not set, using empty". Hook should not fail for missing optional context.</expected_behavior>
</edge_case>

<edge_case id="EC-P6-05" req_ref="REQ-P6-02">
  <scenario>Hook script file has wrong permissions</scenario>
  <expected_behavior>Setup command chmod +x on all scripts. If chmod fails, error: "Cannot set execute permission on hooks/X.sh: [error]".</expected_behavior>
</edge_case>

<edge_case id="EC-P6-06" req_ref="REQ-P6-01">
  <scenario>CLI invoked without database initialized</scenario>
  <expected_behavior>CLI auto-initializes database on first use. Logs: "Initializing contextgraph database at ~/.contextgraph/db".</expected_behavior>
</edge_case>

<edge_case id="EC-P6-07" req_ref="REQ-P6-03">
  <scenario>Multiple concurrent sessions (rare but possible)</scenario>
  <expected_behavior>Each session has its own ID. Memories tagged with their session_id. CLI uses CLAUDE_SESSION_ID env var to distinguish.</expected_behavior>
</edge_case>
</edge_cases>

<error_states>
<error id="ERR-P6-01" exit_code="1">
  <condition>Database connection fails</condition>
  <message>Failed to connect to contextgraph database: [error]</message>
  <recovery>Check ~/.contextgraph/db exists and has correct permissions.</recovery>
</error>

<error id="ERR-P6-02" exit_code="1">
  <condition>Embedding service unavailable</condition>
  <message>Embedding service not responding: [error]</message>
  <recovery>Ensure contextgraph service is running. Run: context-graph-cli status</recovery>
</error>

<error id="ERR-P6-03" exit_code="1">
  <condition>Setup cannot write to .claude directory</condition>
  <message>Cannot write to .claude/settings.json: [error]</message>
  <recovery>Check directory permissions. Create .claude/ manually if needed.</recovery>
</error>

<error id="ERR-P6-04" exit_code="0">
  <condition>Retrieval returns no results (not an error)</condition>
  <message>(no output - empty stdout)</message>
  <recovery>Normal operation. No relevant context found.</recovery>
</error>

<error id="ERR-P6-05" exit_code="1">
  <condition>Invalid command line arguments</condition>
  <message>Invalid argument: [details]. Run --help for usage.</message>
  <recovery>Correct the command line invocation.</recovery>
</error>
</error_states>

<test_plan>
<test_case id="TC-P6-01" type="unit" req_ref="REQ-P6-01">
  <description>CLI parses all subcommands correctly</description>
  <inputs>["session start", "session end", "inject-context --query 'test'", "capture-memory --content 'test'"]</inputs>
  <expected>Each command parsed without error</expected>
</test_case>

<test_case id="TC-P6-02" type="integration" req_ref="REQ-P6-02">
  <description>Hook scripts execute and call CLI</description>
  <inputs>["source hooks/session-start.sh with mock env vars"]</inputs>
  <expected>CLI invoked with correct arguments, output produced</expected>
</test_case>

<test_case id="TC-P6-03" type="integration" req_ref="REQ-P6-03">
  <description>inject-context outputs valid markdown</description>
  <inputs>["inject-context --query 'implement clustering' with memories in DB"]</inputs>
  <expected>Markdown output with ## headers and - bullet points</expected>
</test_case>

<test_case id="TC-P6-04" type="integration" req_ref="REQ-P6-04">
  <description>Session lifecycle tracked correctly</description>
  <inputs>["session start", "capture-memory x3", "session end"]</inputs>
  <expected>All 3 memories associated with session, session marked completed</expected>
</test_case>

<test_case id="TC-P6-05" type="integration" req_ref="REQ-P6-05">
  <description>Setup creates valid .claude/settings.json</description>
  <inputs>["setup command in fresh directory"]</inputs>
  <expected>.claude/settings.json exists with valid JSON, hooks/ directory with 6 scripts</expected>
</test_case>

<test_case id="TC-P6-06" type="unit" req_ref="REQ-P6-06">
  <description>inject-brief completes within 500ms budget</description>
  <inputs>["inject-brief --query 'test' --budget 200"]</inputs>
  <expected>Execution completes in <400ms (buffer for timeout)</expected>
</test_case>

<test_case id="TC-P6-07" type="integration" req_ref="REQ-P6-07">
  <description>CLI reads environment variables</description>
  <inputs>["USER_PROMPT='test query' context-graph-cli inject-context"]</inputs>
  <expected>Query uses 'test query' from environment</expected>
</test_case>

<test_case id="TC-P6-08" type="e2e" req_ref="REQ-P6-02">
  <description>Full hook lifecycle integration test</description>
  <inputs>[
    "Simulate SessionStart",
    "Simulate UserPromptSubmit with query",
    "Simulate PostToolUse x3",
    "Simulate Stop",
    "Simulate SessionEnd"
  ]</inputs>
  <expected>
    - Session created with ID
    - Context injected at SessionStart and UserPromptSubmit
    - 3 memories captured from PostToolUse
    - Response captured at Stop
    - Session completed with all memories associated
  </expected>
</test_case>

<test_case id="TC-P6-09" type="unit" req_ref="REQ-P6-05">
  <description>Setup merges with existing settings</description>
  <inputs>["Existing .claude/settings.json with custom keys, run setup"]</inputs>
  <expected>Custom keys preserved, hooks key added/updated</expected>
</test_case>

<test_case id="TC-P6-10" type="performance" req_ref="REQ-P6-06">
  <description>Hook latency meets timeout requirements</description>
  <inputs>["100 sequential UserPromptSubmit calls with 50 memories in DB"]</inputs>
  <expected>P99 latency < 1500ms (within 2000ms timeout with buffer)</expected>
</test_case>
</test_plan>

<validation_criteria>
  <criterion>CLI binary compiles and runs on Linux/macOS</criterion>
  <criterion>All 6 hook scripts created by setup command</criterion>
  <criterion>Hook scripts are executable (chmod +x)</criterion>
  <criterion>Session lifecycle correctly tracked in database</criterion>
  <criterion>Context injection outputs valid markdown to stdout</criterion>
  <criterion>Capture commands create memories with correct source type</criterion>
  <criterion>All hooks complete within their timeout budgets</criterion>
  <criterion>Setup correctly merges with existing .claude/settings.json</criterion>
  <criterion>CLI handles missing env vars gracefully</criterion>
  <criterion>Verbose mode logs to stderr, not stdout</criterion>
</validation_criteria>
</functional_spec>
```

## CLI Command Reference

| Command | Description | Args | Output |
|---------|-------------|------|--------|
| `session start` | Begin new session | none | session_id to stdout |
| `session end` | End current session | none | none |
| `inject-context` | Full context injection | --query, --budget | Markdown to stdout |
| `inject-brief` | Quick context injection | --query, --budget | Brief markdown to stdout |
| `capture-memory` | Create memory from text | --content, --source | none |
| `capture-response` | Capture Claude response | --content | none |
| `setup` | Configure hooks | --force | Setup status |
| `status` | Show system status | none | Status info |

## Hook Script Flow

```
SessionStart Hook:
┌──────────────────────────────────────────┐
│ 1. context-graph-cli session start       │
│ 2. context-graph-cli inject-context      │
│ 3. Output: Portfolio + Recent Divergences│
└──────────────────────────────────────────┘

UserPromptSubmit Hook:
┌──────────────────────────────────────────┐
│ 1. Read USER_PROMPT from env             │
│ 2. context-graph-cli inject-context      │
│ 3. Output: Similar memories + Divergence │
└──────────────────────────────────────────┘

PreToolUse Hook:
┌──────────────────────────────────────────┐
│ 1. Read TOOL_DESCRIPTION from env        │
│ 2. context-graph-cli inject-brief        │
│ 3. Output: Brief relevant context        │
└──────────────────────────────────────────┘

PostToolUse Hook:
┌──────────────────────────────────────────┐
│ 1. Read TOOL_DESCRIPTION from env        │
│ 2. context-graph-cli capture-memory      │
│ 3. (No output - capture only)            │
└──────────────────────────────────────────┘

Stop Hook:
┌──────────────────────────────────────────┐
│ 1. Read RESPONSE_SUMMARY from env        │
│ 2. context-graph-cli capture-response    │
│ 3. (No output - capture only)            │
└──────────────────────────────────────────┘

SessionEnd Hook:
┌──────────────────────────────────────────┐
│ 1. Read SESSION_SUMMARY from env         │
│ 2. context-graph-cli capture-memory      │
│ 3. context-graph-cli session end         │
│ 4. (No output - capture only)            │
└──────────────────────────────────────────┘
```

## Timeout Budget Allocation

| Hook | Total Timeout | CLI Operations | Per-Op Budget |
|------|---------------|----------------|---------------|
| SessionStart | 5000ms | start + inject | 2500ms each |
| UserPromptSubmit | 2000ms | inject only | 1800ms |
| PreToolUse | 500ms | inject-brief | 400ms |
| PostToolUse | 3000ms | capture | 2500ms |
| Stop | 3000ms | capture | 2500ms |
| SessionEnd | 30000ms | capture + end | 15000ms each |
