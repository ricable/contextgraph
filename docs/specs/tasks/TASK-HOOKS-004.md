# TASK-HOOKS-004: Create CLI Argument Types

```xml
<task_spec id="TASK-HOOKS-004" version="1.0">
<metadata>
  <title>Create CLI Argument Types for Hooks Commands</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>4</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-17</requirement_ref>
    <requirement_ref>REQ-HOOKS-18</requirement_ref>
    <requirement_ref>REQ-HOOKS-19</requirement_ref>
    <requirement_ref>REQ-HOOKS-20</requirement_ref>
    <requirement_ref>REQ-HOOKS-21</requirement_ref>
    <requirement_ref>REQ-HOOKS-22</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-001</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>1.0</estimated_hours>
</metadata>

<context>
This task creates the clap argument definitions for all hooks CLI subcommands.
These define the command-line interface that shell scripts use to invoke context-graph-cli.
Each hook command has specific arguments matching Claude Code's hook data.

Commands defined:
- hooks session-start: Initialize session identity
- hooks pre-tool: Fast path consciousness check
- hooks post-tool: Update IC after tool execution
- hooks prompt-submit: Inject context for user prompt
- hooks session-end: Persist final session state
- hooks generate-config: Generate hook configuration files
</context>

<input_context_files>
  <file purpose="technical_spec">docs/specs/technical/TECH-HOOKS.md#section-2.3</file>
  <file purpose="existing_cli_structure">crates/context-graph-cli/src/main.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-HOOKS-001 completed (HookEventType exists)</check>
  <check>clap is a workspace dependency</check>
</prerequisites>

<scope>
  <in_scope>
    - Create HooksCommands enum with 6 subcommands
    - Create SessionStartArgs struct
    - Create PreToolArgs struct (with fast_path flag)
    - Create PostToolArgs struct
    - Create PromptSubmitArgs struct
    - Create SessionEndArgs struct
    - Create GenerateConfigArgs struct
    - Create OutputFormat enum
    - Create HookType enum (for generate-config)
    - Create ShellType enum
  </in_scope>
  <out_of_scope>
    - Command handler implementations (TASK-HOOKS-012 through 017)
    - Module registration (TASK-HOOKS-011)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/commands/hooks/args.rs">
use clap::{Args, Subcommand, ValueEnum};
use std::path::PathBuf;

/// Hook commands for Claude Code native integration
#[derive(Subcommand, Debug)]
pub enum HooksCommands {
    #[command(name = "session-start")]
    SessionStart(SessionStartArgs),
    #[command(name = "pre-tool")]
    PreTool(PreToolArgs),
    #[command(name = "post-tool")]
    PostTool(PostToolArgs),
    #[command(name = "prompt-submit")]
    PromptSubmit(PromptSubmitArgs),
    #[command(name = "session-end")]
    SessionEnd(SessionEndArgs),
    #[command(name = "generate-config")]
    GenerateConfig(GenerateConfigArgs),
}

#[derive(Args, Debug)]
pub struct SessionStartArgs {
    #[arg(long, env = "CONTEXT_GRAPH_DB_PATH")]
    pub db_path: Option&lt;PathBuf&gt;,
    #[arg(long)]
    pub session_id: Option&lt;String&gt;,
    #[arg(long)]
    pub previous_session_id: Option&lt;String&gt;,
    #[arg(long, default_value = "false")]
    pub stdin: bool,
    #[arg(long, value_enum, default_value = "json")]
    pub format: OutputFormat,
}

#[derive(Args, Debug)]
pub struct PreToolArgs {
    #[arg(long)]
    pub session_id: String,
    #[arg(long)]
    pub tool_name: Option&lt;String&gt;,
    #[arg(long, default_value = "false")]
    pub stdin: bool,
    #[arg(long, default_value = "true")]
    pub fast_path: bool,
    #[arg(long, value_enum, default_value = "json")]
    pub format: OutputFormat,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    Json,
    JsonCompact,
    Text,
}
    </signature>
  </signatures>
  <constraints>
    - All commands MUST use kebab-case names (session-start, not session_start)
    - db_path MUST support CONTEXT_GRAPH_DB_PATH env variable
    - PreToolArgs MUST have fast_path defaulting to true (no DB access)
    - stdin flag MUST default to false
    - format MUST default to json
  </constraints>
  <verification>
    - cargo build --package context-graph-cli
    - Verify --help output for each subcommand
  </verification>
</definition_of_done>

<pseudo_code>
1. Create args.rs file with clap imports

2. Create OutputFormat enum:
   - Json (default, pretty)
   - JsonCompact (single line)
   - Text (human readable)

3. Create HookType enum for generate-config:
   - SessionStart
   - PreToolUse
   - PostToolUse
   - UserPromptSubmit
   - SessionEnd

4. Create ShellType enum:
   - Bash (default)
   - Zsh
   - Fish
   - Powershell

5. Create SessionStartArgs:
   - db_path: Option<PathBuf> with env
   - session_id: Option<String>
   - previous_session_id: Option<String>
   - stdin: bool (default false)
   - format: OutputFormat (default json)

6. Create PreToolArgs (FAST PATH):
   - session_id: String (required)
   - tool_name: Option<String>
   - stdin: bool (default false)
   - fast_path: bool (default true)
   - format: OutputFormat

7. Create PostToolArgs:
   - db_path: Option<PathBuf>
   - session_id: String (required)
   - tool_name: Option<String>
   - success: Option<bool>
   - stdin: bool (default false)
   - format: OutputFormat

8. Create PromptSubmitArgs:
   - db_path: Option<PathBuf>
   - session_id: String (required)
   - prompt: Option<String>
   - stdin: bool (default false)
   - format: OutputFormat

9. Create SessionEndArgs:
   - db_path: Option<PathBuf>
   - session_id: String (required)
   - duration_ms: Option<u64>
   - stdin: bool (default false)
   - generate_summary: bool (default true)
   - format: OutputFormat

10. Create GenerateConfigArgs:
    - output_dir: PathBuf (default .claude/hooks)
    - force: bool (default false)
    - hooks: Option<Vec<HookType>>
    - shell: ShellType (default bash)

11. Create HooksCommands enum with all 6 subcommands
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/commands/hooks/args.rs">CLI argument definitions for hooks commands</file>
</files_to_create>

<files_to_modify>
  <!-- None - module registration in TASK-HOOKS-011 -->
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
</test_commands>
</task_spec>
```

## Implementation

### Create args.rs

```rust
// crates/context-graph-cli/src/commands/hooks/args.rs
//! CLI argument definitions for hooks commands
//!
//! # Commands
//! - `hooks session-start`: Initialize session identity
//! - `hooks pre-tool`: Fast path consciousness check (100ms timeout)
//! - `hooks post-tool`: Update IC after tool execution
//! - `hooks prompt-submit`: Inject context for user prompt
//! - `hooks session-end`: Persist final session state
//! - `hooks generate-config`: Generate hook configuration files

use clap::{Args, Subcommand, ValueEnum};
use std::path::PathBuf;

// ============================================================================
// Output Format
// ============================================================================

/// Output format for hook responses
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum OutputFormat {
    /// JSON format (default for hook integration)
    #[default]
    Json,
    /// Compact JSON (single line, no whitespace)
    JsonCompact,
    /// Human-readable text
    Text,
}

// ============================================================================
// Hook Type (for generate-config)
// ============================================================================

/// Hook types for generation
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum HookType {
    /// Session initialization hook
    SessionStart,
    /// Pre-tool execution hook
    PreToolUse,
    /// Post-tool execution hook
    PostToolUse,
    /// User prompt submission hook
    UserPromptSubmit,
    /// Session termination hook
    SessionEnd,
}

// ============================================================================
// Shell Type
// ============================================================================

/// Shell type for script generation
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum ShellType {
    /// Bash shell (default)
    #[default]
    Bash,
    /// Zsh shell
    Zsh,
    /// Fish shell
    Fish,
    /// PowerShell (Windows)
    Powershell,
}

// ============================================================================
// Session Start Arguments
// ============================================================================

/// Session start command arguments
/// Implements REQ-HOOKS-17
#[derive(Args, Debug)]
pub struct SessionStartArgs {
    /// Database path for session storage
    #[arg(long, env = "CONTEXT_GRAPH_DB_PATH")]
    pub db_path: Option<PathBuf>,

    /// Session ID (auto-generated if not provided)
    #[arg(long)]
    pub session_id: Option<String>,

    /// Previous session ID for continuity linking
    #[arg(long)]
    pub previous_session_id: Option<String>,

    /// Read input from stdin (JSON HookInput)
    #[arg(long, default_value = "false")]
    pub stdin: bool,

    /// Output format
    #[arg(long, value_enum, default_value = "json")]
    pub format: OutputFormat,
}

// ============================================================================
// Pre-Tool Arguments (FAST PATH)
// ============================================================================

/// Pre-tool command arguments (FAST PATH - 100ms timeout)
/// Implements REQ-HOOKS-18
///
/// # Performance
/// This command MUST complete within 100ms.
/// When `fast_path` is true (default), no database access occurs.
#[derive(Args, Debug)]
pub struct PreToolArgs {
    /// Session ID
    #[arg(long)]
    pub session_id: String,

    /// Tool name being invoked
    #[arg(long)]
    pub tool_name: Option<String>,

    /// Read input from stdin
    #[arg(long, default_value = "false")]
    pub stdin: bool,

    /// Skip database access for faster response
    /// When true, uses cached state only (default: true)
    #[arg(long, default_value = "true")]
    pub fast_path: bool,

    /// Output format
    #[arg(long, value_enum, default_value = "json")]
    pub format: OutputFormat,
}

// ============================================================================
// Post-Tool Arguments
// ============================================================================

/// Post-tool command arguments
/// Implements REQ-HOOKS-19
#[derive(Args, Debug)]
pub struct PostToolArgs {
    /// Database path
    #[arg(long, env = "CONTEXT_GRAPH_DB_PATH")]
    pub db_path: Option<PathBuf>,

    /// Session ID
    #[arg(long)]
    pub session_id: String,

    /// Tool name that was executed
    #[arg(long)]
    pub tool_name: Option<String>,

    /// Tool execution succeeded
    #[arg(long)]
    pub success: Option<bool>,

    /// Read input from stdin
    #[arg(long, default_value = "false")]
    pub stdin: bool,

    /// Output format
    #[arg(long, value_enum, default_value = "json")]
    pub format: OutputFormat,
}

// ============================================================================
// Prompt Submit Arguments
// ============================================================================

/// Prompt submit command arguments
/// Implements REQ-HOOKS-20
#[derive(Args, Debug)]
pub struct PromptSubmitArgs {
    /// Database path
    #[arg(long, env = "CONTEXT_GRAPH_DB_PATH")]
    pub db_path: Option<PathBuf>,

    /// Session ID
    #[arg(long)]
    pub session_id: String,

    /// User prompt text
    #[arg(long)]
    pub prompt: Option<String>,

    /// Read input from stdin
    #[arg(long, default_value = "false")]
    pub stdin: bool,

    /// Output format
    #[arg(long, value_enum, default_value = "json")]
    pub format: OutputFormat,
}

// ============================================================================
// Session End Arguments
// ============================================================================

/// Session end command arguments
/// Implements REQ-HOOKS-21
#[derive(Args, Debug)]
pub struct SessionEndArgs {
    /// Database path
    #[arg(long, env = "CONTEXT_GRAPH_DB_PATH")]
    pub db_path: Option<PathBuf>,

    /// Session ID
    #[arg(long)]
    pub session_id: String,

    /// Session duration in milliseconds
    #[arg(long)]
    pub duration_ms: Option<u64>,

    /// Read input from stdin
    #[arg(long, default_value = "false")]
    pub stdin: bool,

    /// Generate session summary
    #[arg(long, default_value = "true")]
    pub generate_summary: bool,

    /// Output format
    #[arg(long, value_enum, default_value = "json")]
    pub format: OutputFormat,
}

// ============================================================================
// Generate Config Arguments
// ============================================================================

/// Generate config command arguments
/// Implements REQ-HOOKS-22
#[derive(Args, Debug)]
pub struct GenerateConfigArgs {
    /// Output directory for hook scripts
    #[arg(long, default_value = ".claude/hooks")]
    pub output_dir: PathBuf,

    /// Overwrite existing files
    #[arg(long, default_value = "false")]
    pub force: bool,

    /// Hook types to generate (all if not specified)
    #[arg(long, value_delimiter = ',')]
    pub hooks: Option<Vec<HookType>>,

    /// Shell to target for script generation
    #[arg(long, value_enum, default_value = "bash")]
    pub shell: ShellType,
}

// ============================================================================
// Hooks Commands Enum
// ============================================================================

/// Hook commands for Claude Code native integration
/// Implements REQ-HOOKS-17 through REQ-HOOKS-22
#[derive(Subcommand, Debug)]
pub enum HooksCommands {
    /// Handle session start event
    /// Timeout: 5000ms
    #[command(name = "session-start")]
    SessionStart(SessionStartArgs),

    /// Handle pre-tool-use event (FAST PATH)
    /// Timeout: 100ms - NO DATABASE ACCESS
    #[command(name = "pre-tool")]
    PreTool(PreToolArgs),

    /// Handle post-tool-use event
    /// Timeout: 3000ms
    #[command(name = "post-tool")]
    PostTool(PostToolArgs),

    /// Handle user prompt submit event
    /// Timeout: 2000ms
    #[command(name = "prompt-submit")]
    PromptSubmit(PromptSubmitArgs),

    /// Handle session end event
    /// Timeout: 30000ms
    #[command(name = "session-end")]
    SessionEnd(SessionEndArgs),

    /// Generate hook configuration files
    #[command(name = "generate-config")]
    GenerateConfig(GenerateConfigArgs),
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct TestCli {
        #[command(subcommand)]
        command: HooksCommands,
    }

    #[test]
    fn test_session_start_args_parsing() {
        let cli = TestCli::parse_from([
            "test",
            "session-start",
            "--session-id",
            "test-123",
            "--stdin",
        ]);
        if let HooksCommands::SessionStart(args) = cli.command {
            assert_eq!(args.session_id, Some("test-123".to_string()));
            assert!(args.stdin);
        } else {
            panic!("Expected SessionStart command");
        }
    }

    #[test]
    fn test_pre_tool_args_defaults() {
        let cli = TestCli::parse_from([
            "test",
            "pre-tool",
            "--session-id",
            "test-123",
        ]);
        if let HooksCommands::PreTool(args) = cli.command {
            assert_eq!(args.session_id, "test-123");
            assert!(args.fast_path); // Default true
            assert!(!args.stdin); // Default false
        } else {
            panic!("Expected PreTool command");
        }
    }

    #[test]
    fn test_generate_config_args() {
        let cli = TestCli::parse_from([
            "test",
            "generate-config",
            "--output-dir",
            "/custom/path",
            "--force",
            "--shell",
            "zsh",
        ]);
        if let HooksCommands::GenerateConfig(args) = cli.command {
            assert_eq!(args.output_dir, PathBuf::from("/custom/path"));
            assert!(args.force);
            assert!(matches!(args.shell, ShellType::Zsh));
        } else {
            panic!("Expected GenerateConfig command");
        }
    }

    #[test]
    fn test_output_format_values() {
        // Verify all format variants exist
        let _json = OutputFormat::Json;
        let _compact = OutputFormat::JsonCompact;
        let _text = OutputFormat::Text;
    }

    #[test]
    fn test_hook_type_values() {
        // Verify all hook types exist for generate-config
        let _types = [
            HookType::SessionStart,
            HookType::PreToolUse,
            HookType::PostToolUse,
            HookType::UserPromptSubmit,
            HookType::SessionEnd,
        ];
        assert_eq!(_types.len(), 5);
    }
}
```

## Verification Checklist

- [ ] All 6 subcommands defined with kebab-case names
- [ ] SessionStartArgs has db_path with env support
- [ ] PreToolArgs.fast_path defaults to true
- [ ] All args.stdin defaults to false
- [ ] All args.format defaults to json
- [ ] GenerateConfigArgs.output_dir defaults to ".claude/hooks"
- [ ] All argument structs derive Args, Debug
