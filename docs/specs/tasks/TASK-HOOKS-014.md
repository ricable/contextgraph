# TASK-HOOKS-014: Create Shell Scripts for Claude Code Hooks

```xml
<task_spec id="TASK-HOOKS-014" version="1.0">
<metadata>
  <title>Create Shell Scripts for Claude Code Hooks</title>
  <status>ready</status>
  <layer>surface</layer>
  <sequence>14</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-01</requirement_ref>
    <requirement_ref>REQ-HOOKS-02</requirement_ref>
    <requirement_ref>REQ-HOOKS-03</requirement_ref>
    <requirement_ref>REQ-HOOKS-04</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-006</task_ref>
    <task_ref>TASK-HOOKS-007</task_ref>
    <task_ref>TASK-HOOKS-008</task_ref>
    <task_ref>TASK-HOOKS-009</task_ref>
    <task_ref>TASK-HOOKS-010</task_ref>
    <task_ref>TASK-HOOKS-011</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>2.0</estimated_hours>
</metadata>

<context>
Claude Code hooks are configured in .claude/settings.json and execute shell commands.
This task creates the actual shell scripts that bridge Claude Code hooks to the
context-graph-cli commands.

Each hook script:
1. Receives hook input via stdin (JSON)
2. Parses relevant fields
3. Calls appropriate context-graph-cli command
4. Outputs result to stdout
</context>

<input_context_files>
  <file purpose="hooks_reference">docs2/claudehooks.md</file>
  <file purpose="technical_spec">docs/specs/technical/TECH-HOOKS.md#shell_scripts</file>
  <file purpose="cli_commands">crates/context-graph-cli/src/commands/</file>
</input_context_files>

<prerequisites>
  <check>All CLI commands implemented (TASK-HOOKS-006 through 011)</check>
  <check>context-graph-cli binary available in PATH or known location</check>
  <check>.claude/ directory exists</check>
</prerequisites>

<scope>
  <in_scope>
    - Create .claude/hooks/ directory structure
    - Create session_start.sh hook script
    - Create session_end.sh hook script
    - Create pre_tool_use.sh hook script
    - Create post_tool_use.sh hook script
    - Create user_prompt_submit.sh hook script
    - Make all scripts executable (chmod +x)
    - Handle jq parsing of hook input
  </in_scope>
  <out_of_scope>
    - .claude/settings.json configuration (TASK-HOOKS-015)
    - Windows batch file equivalents (future feature)
    - Custom hook logic beyond CLI invocation
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file=".claude/hooks/session_start.sh">
      #!/bin/bash
      # Session start hook - outputs consciousness brief
      # Input: { "session_id": "...", "timestamp": "..." }
      # Output: Consciousness brief to stdout
    </signature>
    <signature file=".claude/hooks/session_end.sh">
      #!/bin/bash
      # Session end hook - persists identity snapshot
      # Input: { "session_id": "...", "stats": {...} }
      # Output: Snapshot ID to stdout
    </signature>
    <signature file=".claude/hooks/pre_tool_use.sh">
      #!/bin/bash
      # Pre-tool hook - injects relevant context
      # Input: { "tool_name": "...", "tool_input": {...} }
      # Output: Injected context to stdout
    </signature>
    <signature file=".claude/hooks/post_tool_use.sh">
      #!/bin/bash
      # Post-tool hook - records tool outcome
      # Input: { "tool_name": "...", "tool_result": {...} }
      # Output: Processing confirmation
    </signature>
  </signatures>

  <constraints>
    - Scripts must be POSIX-compliant (#!/bin/bash)
    - Scripts must handle missing jq gracefully
    - Scripts must exit 0 on success, non-zero on error
    - Scripts must complete within timeout (2000ms for most hooks)
    - Scripts must not block on user input
    - Error output goes to stderr, success to stdout
  </constraints>

  <verification>
    - chmod +x .claude/hooks/*.sh
    - echo '{"session_id":"test"}' | .claude/hooks/session_start.sh
    - All scripts return exit code 0 with valid input
  </verification>
</definition_of_done>

<pseudo_code>
session_start.sh:
  #!/bin/bash
  set -euo pipefail

  # Parse input
  INPUT=$(cat)
  SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // "unknown"')

  # Set session context
  export CONTEXT_GRAPH_SESSION_ID="$SESSION_ID"

  # Output consciousness brief
  context-graph-cli consciousness brief --format text

  # Attempt identity restoration (non-blocking)
  context-graph-cli identity restore --latest 2>/dev/null || true

session_end.sh:
  #!/bin/bash
  set -euo pipefail

  INPUT=$(cat)
  SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // "unknown"')

  # Capture identity snapshot
  SNAPSHOT_ID=$(context-graph-cli identity snapshot --session-id "$SESSION_ID" --format json | jq -r '.snapshot_id')

  echo "Session ended. Snapshot: $SNAPSHOT_ID"

pre_tool_use.sh:
  #!/bin/bash
  set -euo pipefail

  INPUT=$(cat)
  TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // "unknown"')
  TOOL_INPUT=$(echo "$INPUT" | jq -r '.tool_input | tostring')

  # Inject relevant context based on tool
  context-graph-cli consciousness inject --query "$TOOL_NAME $TOOL_INPUT" --max-tokens 200 --format text

post_tool_use.sh:
  #!/bin/bash
  set -euo pipefail

  INPUT=$(cat)
  TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // "unknown"')
  TOOL_RESULT=$(echo "$INPUT" | jq -r '.tool_result | tostring')

  # Record tool outcome (async, fire-and-forget)
  context-graph-cli tool record --name "$TOOL_NAME" --result "$TOOL_RESULT" &

  echo "Tool recorded: $TOOL_NAME"
</pseudo_code>

<files_to_create>
  <file path=".claude/hooks/session_start.sh">
    Session start hook script calling consciousness brief and identity restore
  </file>
  <file path=".claude/hooks/session_end.sh">
    Session end hook script calling identity snapshot
  </file>
  <file path=".claude/hooks/pre_tool_use.sh">
    Pre-tool hook script calling consciousness inject
  </file>
  <file path=".claude/hooks/post_tool_use.sh">
    Post-tool hook script calling tool record
  </file>
  <file path=".claude/hooks/user_prompt_submit.sh">
    User prompt hook script for context analysis
  </file>
</files_to_create>

<files_to_modify>
  <!-- None - all new files -->
</files_to_modify>

<test_commands>
  <command>chmod +x .claude/hooks/*.sh</command>
  <command>echo '{"session_id":"test-123"}' | .claude/hooks/session_start.sh</command>
  <command>echo '{"session_id":"test-123"}' | .claude/hooks/session_end.sh</command>
  <command>echo '{"tool_name":"Read","tool_input":{}}' | .claude/hooks/pre_tool_use.sh</command>
</test_commands>
</task_spec>
```
