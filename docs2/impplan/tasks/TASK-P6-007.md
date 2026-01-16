# Task: TASK-P6-007 - Setup Command

```xml
<task_spec id="TASK-P6-007" version="1.0">
<metadata>
  <title>Setup Command</title>
  <phase>6</phase>
  <sequence>49</sequence>
  <layer>surface</layer>
  <estimated_loc>250</estimated_loc>
  <dependencies>
    <dependency task="TASK-P6-001">CLI infrastructure</dependency>
    <dependency task="TASK-P6-002">Session commands (used in hooks)</dependency>
    <dependency task="TASK-P6-003">Inject command (used in hooks)</dependency>
    <dependency task="TASK-P6-005">Capture command (used in hooks)</dependency>
  </dependencies>
  <produces>
    <artifact type="function">handle_setup</artifact>
    <artifact type="const">SETTINGS_JSON_TEMPLATE</artifact>
    <artifact type="const">HOOK_SCRIPT_TEMPLATES</artifact>
  </produces>
</metadata>

<context>
  <background>
    The setup command configures Claude Code to use context-graph by creating
    or updating .claude/settings.json with hook configurations and creating
    the hook shell scripts in ./hooks/ directory.
  </background>
  <business_value>
    One-command setup for context-graph integration with Claude Code.
    Makes onboarding trivial for new users.
  </business_value>
  <technical_context>
    Creates .claude/settings.json with hooks configuration. Creates ./hooks/
    directory with 6 shell scripts (session-start, user-prompt-submit,
    pre-tool-use, post-tool-use, stop, session-end). Preserves existing
    non-hook settings in settings.json.
  </technical_context>
</context>

<prerequisites>
  <prerequisite type="code">crates/context-graph-cli/src/main.rs with Setup command</prerequisite>
  <prerequisite type="knowledge">Claude Code hook system format per TECH-PHASE6</prerequisite>
</prerequisites>

<scope>
  <includes>
    <item>handle_setup() function</item>
    <item>Settings.json template and merge logic</item>
    <item>Hook script templates (6 scripts)</item>
    <item>Directory creation (.claude, hooks)</item>
    <item>chmod +x on scripts</item>
    <item>--force flag for overwriting</item>
  </includes>
  <excludes>
    <item>Hook script execution (Claude Code handles this)</item>
    <item>Hook timeout configuration validation</item>
  </excludes>
</scope>

<definition_of_done>
  <criterion id="DOD-1">
    <description>setup creates .claude/settings.json with hooks config</description>
    <verification>File exists and contains hooks key</verification>
  </criterion>
  <criterion id="DOD-2">
    <description>setup creates ./hooks/ directory with 6 scripts</description>
    <verification>ls ./hooks/ shows all 6 scripts</verification>
  </criterion>
  <criterion id="DOD-3">
    <description>Hook scripts are executable</description>
    <verification>stat -c %a shows 755</verification>
  </criterion>
  <criterion id="DOD-4">
    <description>--force overwrites existing configuration</description>
    <verification>Running setup --force recreates files</verification>
  </criterion>
  <criterion id="DOD-5">
    <description>Existing settings.json keys are preserved</description>
    <verification>Non-hook keys remain after setup</verification>
  </criterion>

  <signatures>
    <signature name="handle_setup">
      <code>
pub async fn handle_setup(force: bool) -> Result&lt;(), CliError&gt;
      </code>
    </signature>
  </signatures>

  <constraints>
    <constraint type="behavior">Without --force, fail if hooks already configured</constraint>
    <constraint type="behavior">Preserve non-hook keys in existing settings.json</constraint>
    <constraint type="permissions">Scripts must be chmod 755</constraint>
  </constraints>
</definition_of_done>

<pseudo_code>
```rust
// crates/context-graph-cli/src/commands/setup.rs

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use serde_json::{json, Value};
use tracing::info;
use crate::error::CliError;

/// Hook configuration for settings.json
const SETTINGS_JSON_HOOKS: &str = r#"{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "./hooks/session-start.sh",
            "timeout": 5000
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "./hooks/user-prompt-submit.sh",
            "timeout": 2000
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Edit|Write|Bash",
        "hooks": [
          {
            "type": "command",
            "command": "./hooks/pre-tool-use.sh",
            "timeout": 500
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "./hooks/post-tool-use.sh",
            "timeout": 3000
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "./hooks/stop.sh",
            "timeout": 3000
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "./hooks/session-end.sh",
            "timeout": 30000
          }
        ]
      }
    ]
  }
}"#;

/// Hook script templates
const HOOK_SESSION_START: &str = r#"#!/bin/bash
set -e

# Start session and get context
SESSION_ID=$(context-graph-cli session start)
export CLAUDE_SESSION_ID="$SESSION_ID"

# Inject portfolio summary
context-graph-cli inject-context --session-id "$SESSION_ID"
"#;

const HOOK_USER_PROMPT_SUBMIT: &str = r#"#!/bin/bash
set -e

# Read session ID from file or env
SESSION_ID="${CLAUDE_SESSION_ID:-$(cat ~/.contextgraph/current_session 2>/dev/null || echo '')}"

# Inject context based on user prompt
context-graph-cli inject-context \
  --query "${USER_PROMPT:-}" \
  --session-id "$SESSION_ID"
"#;

const HOOK_PRE_TOOL_USE: &str = r#"#!/bin/bash
set -e

# Quick context for tool use
context-graph-cli inject-brief \
  --query "${TOOL_DESCRIPTION:-${TOOL_NAME:-}}" \
  --budget 200
"#;

const HOOK_POST_TOOL_USE: &str = r#"#!/bin/bash
set -e

SESSION_ID="${CLAUDE_SESSION_ID:-$(cat ~/.contextgraph/current_session 2>/dev/null || echo '')}"

# Capture tool description as memory
context-graph-cli capture-memory \
  --content "${TOOL_DESCRIPTION:-}" \
  --source hook \
  --hook-type PostToolUse \
  --tool-name "${TOOL_NAME:-}" \
  --session-id "$SESSION_ID"
"#;

const HOOK_STOP: &str = r#"#!/bin/bash
set -e

SESSION_ID="${CLAUDE_SESSION_ID:-$(cat ~/.contextgraph/current_session 2>/dev/null || echo '')}"

# Capture Claude's response
context-graph-cli capture-response \
  --content "${RESPONSE_SUMMARY:-}" \
  --session-id "$SESSION_ID"
"#;

const HOOK_SESSION_END: &str = r#"#!/bin/bash
set -e

SESSION_ID="${CLAUDE_SESSION_ID:-$(cat ~/.contextgraph/current_session 2>/dev/null || echo '')}"

# Capture session summary
if [ -n "${SESSION_SUMMARY:-}" ]; then
  context-graph-cli capture-memory \
    --content "$SESSION_SUMMARY" \
    --source hook \
    --hook-type SessionEnd \
    --session-id "$SESSION_ID"
fi

# End session
context-graph-cli session end
"#;

/// Handle setup command.
/// Creates .claude/settings.json and ./hooks/ directory with scripts.
pub async fn handle_setup(force: bool) -> Result<(), CliError> {
    let settings_path = Path::new(".claude/settings.json");
    let hooks_dir = Path::new("./hooks");

    // Check for existing configuration
    if settings_path.exists() && !force {
        let existing: Value = serde_json::from_str(&fs::read_to_string(settings_path)?)?;
        if existing.get("hooks").is_some() {
            return Err(CliError::ConfigError {
                message: "Hooks already configured. Use --force to overwrite.".to_string(),
            });
        }
    }

    info!("Setting up context-graph integration");

    // Create directories
    fs::create_dir_all(".claude")?;
    fs::create_dir_all("./hooks")?;

    // Create or update settings.json
    let hooks_config: Value = serde_json::from_str(SETTINGS_JSON_HOOKS)?;

    let final_settings = if settings_path.exists() {
        let mut existing: Value = serde_json::from_str(&fs::read_to_string(settings_path)?)?;
        merge_settings(&mut existing, hooks_config);
        existing
    } else {
        hooks_config
    };

    fs::write(settings_path, serde_json::to_string_pretty(&final_settings)?)?;
    info!("Created .claude/settings.json");

    // Create hook scripts
    create_hook_script(hooks_dir.join("session-start.sh"), HOOK_SESSION_START)?;
    create_hook_script(hooks_dir.join("user-prompt-submit.sh"), HOOK_USER_PROMPT_SUBMIT)?;
    create_hook_script(hooks_dir.join("pre-tool-use.sh"), HOOK_PRE_TOOL_USE)?;
    create_hook_script(hooks_dir.join("post-tool-use.sh"), HOOK_POST_TOOL_USE)?;
    create_hook_script(hooks_dir.join("stop.sh"), HOOK_STOP)?;
    create_hook_script(hooks_dir.join("session-end.sh"), HOOK_SESSION_END)?;

    info!("Created hook scripts in ./hooks/");

    // Print summary
    println!("✓ Created .claude/settings.json with hooks configuration");
    println!("✓ Created hook scripts:");
    println!("  - ./hooks/session-start.sh");
    println!("  - ./hooks/user-prompt-submit.sh");
    println!("  - ./hooks/pre-tool-use.sh");
    println!("  - ./hooks/post-tool-use.sh");
    println!("  - ./hooks/stop.sh");
    println!("  - ./hooks/session-end.sh");
    println!();
    println!("Setup complete! Context-graph is now integrated with Claude Code.");

    Ok(())
}

fn create_hook_script(path: impl AsRef<Path>, content: &str) -> Result<(), CliError> {
    let path = path.as_ref();
    fs::write(path, content)?;

    // Make executable (chmod 755)
    let mut perms = fs::metadata(path)?.permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms)?;

    Ok(())
}

fn merge_settings(existing: &mut Value, hooks_config: Value) {
    if let Value::Object(ref mut map) = existing {
        if let Value::Object(hooks_map) = hooks_config {
            for (key, value) in hooks_map {
                map.insert(key, value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_merge_settings_preserves_keys() {
        let mut existing = json!({
            "customKey": "value",
            "another": 123
        });

        let hooks = json!({
            "hooks": {
                "SessionStart": []
            }
        });

        merge_settings(&mut existing, hooks);

        assert!(existing.get("customKey").is_some());
        assert!(existing.get("another").is_some());
        assert!(existing.get("hooks").is_some());
    }

    #[test]
    fn test_create_hook_script_executable() {
        let temp_dir = TempDir::new().unwrap();
        let script_path = temp_dir.path().join("test.sh");

        create_hook_script(&script_path, "#!/bin/bash\necho test").unwrap();

        let perms = fs::metadata(&script_path).unwrap().permissions();
        assert_eq!(perms.mode() & 0o777, 0o755);
    }
}
```
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/commands/setup.rs">
    Setup command handler with templates
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/commands/mod.rs">
    Add pub mod setup;
  </file>
</files_to_modify>

<validation_criteria>
  <criterion type="compilation">cargo build --package context-graph-cli compiles</criterion>
  <criterion type="test">cargo test commands::setup --package context-graph-cli -- all tests pass</criterion>
  <criterion type="cli">./context-graph-cli setup creates all files correctly</criterion>
</validation_criteria>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test commands::setup --package context-graph-cli</command>
  <command>cd /tmp/test-project && /path/to/context-graph-cli setup</command>
  <command>ls -la ./hooks/</command>
  <command>cat .claude/settings.json | jq .hooks</command>
</test_commands>
</task_spec>
```
