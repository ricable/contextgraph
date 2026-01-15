# TASK-HOOKS-015: Configure .claude/settings.json Hook Registrations

```xml
<task_spec id="TASK-HOOKS-015" version="1.0">
<metadata>
  <title>Configure .claude/settings.json Hook Registrations</title>
  <status>ready</status>
  <layer>surface</layer>
  <sequence>15</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-05</requirement_ref>
    <requirement_ref>REQ-HOOKS-06</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-014</task_ref>
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
  <estimated_hours>1.0</estimated_hours>
</metadata>

<context>
Claude Code discovers hooks through .claude/settings.json configuration.
This task creates the settings file that registers all Context Graph hooks
with appropriate timeouts and matchers.

Per constitution AP-50: Only native Claude Code hooks via settings.json.
No custom hook infrastructure.
</context>

<input_context_files>
  <file purpose="hooks_reference">docs2/claudehooks.md</file>
  <file purpose="technical_spec">docs/specs/technical/TECH-HOOKS.md#settings_json</file>
  <file purpose="shell_scripts">.claude/hooks/</file>
</input_context_files>

<prerequisites>
  <check>Shell scripts exist in .claude/hooks/ (TASK-HOOKS-014)</check>
  <check>Scripts are executable</check>
</prerequisites>

<scope>
  <in_scope>
    - Create .claude/settings.json with hooks array
    - Configure PreToolUse hook with tool matchers
    - Configure PostToolUse hook
    - Configure SessionStart hook
    - Configure SessionEnd hook
    - Configure UserPromptSubmit hook
    - Set appropriate timeouts per hook type
  </in_scope>
  <out_of_scope>
    - Hook script implementation (TASK-HOOKS-014)
    - Custom hook types not in Claude Code spec
    - MCP server configuration (separate config)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file=".claude/settings.json">
      {
        "hooks": [
          {
            "type": "PreToolUse",
            "command": ".claude/hooks/pre_tool_use.sh",
            "timeout": 2000,
            "matcher": { "tool_name": "*" }
          },
          {
            "type": "PostToolUse",
            "command": ".claude/hooks/post_tool_use.sh",
            "timeout": 2000
          },
          {
            "type": "SessionStart",
            "command": ".claude/hooks/session_start.sh",
            "timeout": 5000
          },
          {
            "type": "SessionEnd",
            "command": ".claude/hooks/session_end.sh",
            "timeout": 5000
          },
          {
            "type": "UserPromptSubmit",
            "command": ".claude/hooks/user_prompt_submit.sh",
            "timeout": 2000
          }
        ]
      }
    </signature>
  </signatures>

  <constraints>
    - File must be valid JSON (parseable by jq)
    - Hook commands must use relative paths from project root
    - Timeouts must match Claude Code limits (2000ms for tool hooks, 5000ms for session)
    - Must not include unsupported hook types
    - No absolute paths (portability)
  </constraints>

  <verification>
    - jq . .claude/settings.json (valid JSON)
    - Claude Code recognizes hooks on startup
    - Hook commands execute when triggered
  </verification>
</definition_of_done>

<pseudo_code>
settings.json structure:
{
  "hooks": [
    // Session lifecycle hooks
    {
      "type": "SessionStart",
      "command": ".claude/hooks/session_start.sh",
      "timeout": 5000  // 5s for startup
    },
    {
      "type": "SessionEnd",
      "command": ".claude/hooks/session_end.sh",
      "timeout": 5000  // 5s for cleanup
    },

    // Tool hooks (2s timeout per Claude Code spec)
    {
      "type": "PreToolUse",
      "command": ".claude/hooks/pre_tool_use.sh",
      "timeout": 2000,
      "matcher": {
        "tool_name": "*"  // All tools
      }
    },
    {
      "type": "PostToolUse",
      "command": ".claude/hooks/post_tool_use.sh",
      "timeout": 2000
    },

    // User interaction hooks
    {
      "type": "UserPromptSubmit",
      "command": ".claude/hooks/user_prompt_submit.sh",
      "timeout": 2000
    }
  ],

  // Optional: MCP server configuration
  "mcpServers": {
    "context-graph": {
      "command": "context-graph-mcp",
      "args": ["--port", "9315"]
    }
  }
}
</pseudo_code>

<files_to_create>
  <file path=".claude/settings.json">
    Complete Claude Code settings with hook registrations
  </file>
</files_to_create>

<files_to_modify>
  <!-- None - creating new settings file -->
</files_to_modify>

<test_commands>
  <command>jq . .claude/settings.json</command>
  <command>cat .claude/settings.json | python3 -m json.tool</command>
</test_commands>
</task_spec>
```
