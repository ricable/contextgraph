# Task: TASK-P6-001 - CLI Arg Parsing with Clap

```xml
<task_spec id="TASK-P6-001" version="1.0">
<metadata>
  <title>CLI Argument Parsing with Clap</title>
  <phase>6</phase>
  <sequence>43</sequence>
  <layer>foundation</layer>
  <estimated_loc>250</estimated_loc>
  <dependencies>
    <!-- No phase dependencies - foundation task -->
  </dependencies>
  <produces>
    <artifact type="struct">CliConfig</artifact>
    <artifact type="struct">CliContext</artifact>
    <artifact type="enum">Commands</artifact>
    <artifact type="enum">CliError</artifact>
    <artifact type="binary">context-graph-cli entry point</artifact>
  </produces>
</metadata>

<context>
  <background>
    The CLI binary is the primary interface between Claude Code hooks and the
    context-graph-core library. It parses command-line arguments, initializes
    configuration, sets up logging, and dispatches to appropriate command handlers.
  </background>
  <business_value>
    Provides the executable that Claude Code hooks invoke. Without this CLI,
    there is no way to integrate with Claude Code's hook system.
  </business_value>
  <technical_context>
    Uses clap v4 for argument parsing with derive macros. Supports both command-line
    arguments and environment variable fallbacks. Logs to stderr in verbose mode
    to avoid stdout pollution that would interfere with context injection.
  </technical_context>
</context>

<prerequisites>
  <prerequisite type="crate">clap = "4.4" with derive feature</prerequisite>
  <prerequisite type="crate">tokio = "1.0" with full feature</prerequisite>
  <prerequisite type="crate">tracing = "0.1"</prerequisite>
  <prerequisite type="crate">tracing-subscriber = "0.3"</prerequisite>
  <prerequisite type="crate">thiserror = "1.0"</prerequisite>
  <prerequisite type="crate">context-graph-core (workspace dependency)</prerequisite>
</prerequisites>

<scope>
  <includes>
    <item>CliConfig struct with all configuration fields</item>
    <item>CliContext struct for shared state across commands</item>
    <item>Commands enum with all CLI subcommands</item>
    <item>CliError enum for CLI-specific errors</item>
    <item>main.rs entry point with clap setup</item>
    <item>Logging initialization (file and stderr)</item>
    <item>Config loading from ~/.contextgraph/</item>
    <item>Environment variable reading utility</item>
  </includes>
  <excludes>
    <item>Individual command implementations (TASK-P6-002 through P6-007)</item>
    <item>Hook scripts (TASK-P6-008)</item>
  </excludes>
</scope>

<definition_of_done>
  <criterion id="DOD-1">
    <description>CLI binary compiles and shows help</description>
    <verification>cargo build --package context-graph-cli && ./target/debug/context-graph-cli --help</verification>
  </criterion>
  <criterion id="DOD-2">
    <description>All subcommands defined with correct arguments</description>
    <verification>Each subcommand shows proper help text</verification>
  </criterion>
  <criterion id="DOD-3">
    <description>CliError covers all error variants</description>
    <verification>Unit tests for each variant</verification>
  </criterion>
  <criterion id="DOD-4">
    <description>Logging works in both verbose and normal modes</description>
    <verification>Run with --verbose shows stderr output</verification>
  </criterion>

  <signatures>
    <signature name="CliConfig">
      <code>
#[derive(Debug, Clone)]
pub struct CliConfig {
    pub db_path: PathBuf,
    pub log_path: PathBuf,
    pub current_session_file: PathBuf,
    pub verbose: bool,
    pub timeout_ms: u64,
}
      </code>
    </signature>
    <signature name="CliContext">
      <code>
pub struct CliContext {
    pub config: CliConfig,
    pub db: Arc&lt;Database&gt;,
}
      </code>
    </signature>
    <signature name="Commands">
      <code>
#[derive(Debug, Subcommand)]
pub enum Commands {
    Session { #[command(subcommand)] action: SessionAction },
    InjectContext { #[arg(long)] query: Option&lt;String&gt;, #[arg(long)] session_id: Option&lt;String&gt;, #[arg(long, default_value = "1200")] budget: u32 },
    InjectBrief { #[arg(long)] query: Option&lt;String&gt;, #[arg(long, default_value = "200")] budget: u32 },
    CaptureMemory { #[arg(long)] content: Option&lt;String&gt;, #[arg(long)] source: String, #[arg(long)] session_id: Option&lt;String&gt;, #[arg(long)] hook_type: Option&lt;String&gt;, #[arg(long)] tool_name: Option&lt;String&gt; },
    CaptureResponse { #[arg(long)] content: Option&lt;String&gt;, #[arg(long)] session_id: Option&lt;String&gt; },
    Setup { #[arg(long)] force: bool },
    Status,
}
      </code>
    </signature>
    <signature name="CliError">
      <code>
#[derive(Debug, Error)]
pub enum CliError {
    #[error("Config error: {message}")]
    ConfigError { message: String },
    #[error("Database error: {source}")]
    DatabaseError { #[from] source: StorageError },
    #[error("Embedding error: {source}")]
    EmbeddingError { #[from] source: EmbedderError },
    #[error("Capture error: {source}")]
    CaptureError { #[from] source: CaptureError },
    #[error("Injection error: {source}")]
    InjectionError { #[from] source: InjectionError },
    #[error("IO error: {source}")]
    IoError { #[from] source: std::io::Error },
    #[error("JSON error: {source}")]
    JsonError { #[from] source: serde_json::Error },
    #[error("Timeout: {operation} exceeded {duration_ms}ms")]
    TimeoutError { operation: String, duration_ms: u64 },
}
      </code>
    </signature>
  </signatures>

  <constraints>
    <constraint type="output">Verbose logging goes to stderr only</constraint>
    <constraint type="output">Command output goes to stdout only</constraint>
    <constraint type="config">Default db_path = ~/.contextgraph/db</constraint>
    <constraint type="config">Default log_path = ~/.contextgraph/logs</constraint>
    <constraint type="config">Default timeout_ms = 5000</constraint>
  </constraints>
</definition_of_done>

<pseudo_code>
```rust
// crates/context-graph-cli/src/main.rs

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

mod commands;
mod config;
mod error;
mod env;

use config::CliConfig;
use error::CliError;

/// Context Graph CLI - Memory management for Claude Code
#[derive(Parser, Debug)]
#[command(name = "context-graph-cli")]
#[command(version, about, long_about = None)]
pub struct Cli {
    /// Enable verbose logging to stderr
    #[arg(long, global = true)]
    verbose: bool,

    /// Override database path
    #[arg(long, global = true)]
    db_path: Option<PathBuf>,

    /// Timeout in milliseconds
    #[arg(long, global = true, default_value = "5000")]
    timeout: u64,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Manage sessions
    Session {
        #[command(subcommand)]
        action: SessionAction,
    },

    /// Inject full context (for SessionStart, UserPromptSubmit)
    InjectContext {
        /// Query text to find relevant context
        #[arg(long)]
        query: Option<String>,

        /// Session ID
        #[arg(long)]
        session_id: Option<String>,

        /// Token budget
        #[arg(long, default_value = "1200")]
        budget: u32,
    },

    /// Inject brief context (for PreToolUse)
    InjectBrief {
        /// Query text
        #[arg(long)]
        query: Option<String>,

        /// Token budget
        #[arg(long, default_value = "200")]
        budget: u32,
    },

    /// Capture memory from hook
    CaptureMemory {
        /// Content to capture
        #[arg(long)]
        content: Option<String>,

        /// Source: hook or response
        #[arg(long)]
        source: String,

        /// Session ID
        #[arg(long)]
        session_id: Option<String>,

        /// Hook type (for source=hook)
        #[arg(long)]
        hook_type: Option<String>,

        /// Tool name (for PostToolUse)
        #[arg(long)]
        tool_name: Option<String>,
    },

    /// Capture Claude response
    CaptureResponse {
        /// Response content
        #[arg(long)]
        content: Option<String>,

        /// Session ID
        #[arg(long)]
        session_id: Option<String>,
    },

    /// Setup hooks and configuration
    Setup {
        /// Force overwrite existing config
        #[arg(long)]
        force: bool,
    },

    /// Show system status
    Status,
}

#[derive(Debug, Subcommand)]
pub enum SessionAction {
    /// Start a new session
    Start,
    /// End the current session
    End,
}

#[tokio::main]
async fn main() -> Result<(), CliError> {
    let cli = Cli::parse();

    // Initialize logging
    init_logging(cli.verbose)?;

    // Load config
    let config = CliConfig::load(cli.db_path, cli.timeout, cli.verbose)?;

    // Initialize database
    let db = config.init_database().await?;
    let ctx = CliContext { config, db: Arc::new(db) };

    // Dispatch command
    match cli.command {
        Commands::Session { action } => match action {
            SessionAction::Start => commands::session::handle_start(&ctx).await,
            SessionAction::End => commands::session::handle_end(&ctx).await,
        },
        Commands::InjectContext { query, session_id, budget } => {
            commands::inject::handle_inject_context(&ctx, query, session_id, budget).await
        }
        Commands::InjectBrief { query, budget } => {
            commands::inject::handle_inject_brief(&ctx, query, budget).await
        }
        Commands::CaptureMemory { content, source, session_id, hook_type, tool_name } => {
            commands::capture::handle_capture_memory(&ctx, content, source, session_id, hook_type, tool_name).await
        }
        Commands::CaptureResponse { content, session_id } => {
            commands::capture::handle_capture_response(&ctx, content, session_id).await
        }
        Commands::Setup { force } => {
            commands::setup::handle_setup(force).await
        }
        Commands::Status => {
            commands::status::handle_status(&ctx).await
        }
    }
}

fn init_logging(verbose: bool) -> Result<(), CliError> {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let filter = if verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("warn")
    };

    if verbose {
        // Log to stderr in verbose mode
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt::layer().with_writer(std::io::stderr))
            .init();
    } else {
        // Log to file in normal mode
        let log_dir = dirs::home_dir()
            .ok_or_else(|| CliError::ConfigError {
                message: "Could not find home directory".to_string(),
            })?
            .join(".contextgraph/logs");

        std::fs::create_dir_all(&log_dir)?;

        let file_appender = tracing_appender::rolling::daily(&log_dir, "cli.log");
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt::layer().with_writer(file_appender))
            .init();
    }

    Ok(())
}

// crates/context-graph-cli/src/config.rs

use std::path::PathBuf;
use crate::error::CliError;

#[derive(Debug, Clone)]
pub struct CliConfig {
    pub db_path: PathBuf,
    pub log_path: PathBuf,
    pub current_session_file: PathBuf,
    pub verbose: bool,
    pub timeout_ms: u64,
}

impl CliConfig {
    pub fn load(
        db_path_override: Option<PathBuf>,
        timeout: u64,
        verbose: bool,
    ) -> Result<Self, CliError> {
        let home = dirs::home_dir().ok_or_else(|| CliError::ConfigError {
            message: "Could not find home directory".to_string(),
        })?;

        let base_dir = home.join(".contextgraph");

        let db_path = db_path_override.unwrap_or_else(|| base_dir.join("db"));
        let log_path = base_dir.join("logs");
        let current_session_file = base_dir.join("current_session");

        // Create directories if they don't exist
        std::fs::create_dir_all(&db_path)?;
        std::fs::create_dir_all(&log_path)?;

        Ok(Self {
            db_path,
            log_path,
            current_session_file,
            verbose,
            timeout_ms: timeout,
        })
    }

    pub async fn init_database(&self) -> Result<Database, CliError> {
        use context_graph_core::storage::Database;

        Database::open(&self.db_path)
            .await
            .map_err(|e| CliError::DatabaseError { source: e })
    }
}

pub struct CliContext {
    pub config: CliConfig,
    pub db: Arc<Database>,
}

// crates/context-graph-cli/src/error.rs

use thiserror::Error;
use context_graph_core::storage::StorageError;
use context_graph_core::embedding::EmbedderError;
use context_graph_core::memory::CaptureError;
use context_graph_core::injection::InjectionError;

#[derive(Debug, Error)]
pub enum CliError {
    #[error("Config error: {message}")]
    ConfigError { message: String },

    #[error("Database error: {source}")]
    DatabaseError {
        #[from]
        source: StorageError,
    },

    #[error("Embedding error: {source}")]
    EmbeddingError {
        #[from]
        source: EmbedderError,
    },

    #[error("Capture error: {source}")]
    CaptureError {
        #[from]
        source: CaptureError,
    },

    #[error("Injection error: {source}")]
    InjectionError {
        #[from]
        source: InjectionError,
    },

    #[error("IO error: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },

    #[error("JSON error: {source}")]
    JsonError {
        #[from]
        source: serde_json::Error,
    },

    #[error("Timeout: {operation} exceeded {duration_ms}ms")]
    TimeoutError { operation: String, duration_ms: u64 },
}

impl CliError {
    pub fn exit_code(&self) -> i32 {
        match self {
            CliError::ConfigError { .. } => 1,
            CliError::DatabaseError { .. } => 2,
            CliError::EmbeddingError { .. } => 3,
            CliError::CaptureError { .. } => 4,
            CliError::InjectionError { .. } => 5,
            CliError::IoError { .. } => 6,
            CliError::JsonError { .. } => 7,
            CliError::TimeoutError { .. } => 8,
        }
    }
}

// crates/context-graph-cli/src/env.rs

use tracing::debug;

/// Get value from environment variable or argument.
/// Prefers argument if provided, falls back to env var.
pub fn get_env_or_arg(env_name: &str, arg_value: Option<String>) -> Option<String> {
    if let Some(value) = arg_value {
        debug!("Using argument value for {}", env_name);
        return Some(value);
    }

    match std::env::var(env_name) {
        Ok(value) if !value.is_empty() => {
            debug!("Using environment variable {}", env_name);
            Some(value)
        }
        _ => None,
    }
}

/// Get required value from environment variable or argument.
/// Returns error if neither is available.
pub fn require_env_or_arg(
    env_name: &str,
    arg_value: Option<String>,
    description: &str,
) -> Result<String, String> {
    get_env_or_arg(env_name, arg_value)
        .ok_or_else(|| format!("{} required (use --{} or set {})", description, env_name.to_lowercase().replace('_', "-"), env_name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arg_takes_priority() {
        std::env::set_var("TEST_VAR", "env_value");
        let result = get_env_or_arg("TEST_VAR", Some("arg_value".to_string()));
        assert_eq!(result, Some("arg_value".to_string()));
        std::env::remove_var("TEST_VAR");
    }

    #[test]
    fn test_env_fallback() {
        std::env::set_var("TEST_VAR_2", "env_value");
        let result = get_env_or_arg("TEST_VAR_2", None);
        assert_eq!(result, Some("env_value".to_string()));
        std::env::remove_var("TEST_VAR_2");
    }

    #[test]
    fn test_neither_returns_none() {
        std::env::remove_var("NONEXISTENT_VAR");
        let result = get_env_or_arg("NONEXISTENT_VAR", None);
        assert_eq!(result, None);
    }
}
```
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/Cargo.toml">
    CLI crate manifest with dependencies
  </file>
  <file path="crates/context-graph-cli/src/main.rs">
    Entry point with clap setup
  </file>
  <file path="crates/context-graph-cli/src/config.rs">
    CliConfig and CliContext
  </file>
  <file path="crates/context-graph-cli/src/error.rs">
    CliError enum
  </file>
  <file path="crates/context-graph-cli/src/env.rs">
    Environment variable utilities
  </file>
  <file path="crates/context-graph-cli/src/commands/mod.rs">
    Command module exports
  </file>
</files_to_create>

<files_to_modify>
  <file path="Cargo.toml">
    Add context-graph-cli to workspace members
  </file>
</files_to_modify>

<validation_criteria>
  <criterion type="compilation">cargo build --package context-graph-cli compiles</criterion>
  <criterion type="test">cargo test --package context-graph-cli -- all tests pass</criterion>
  <criterion type="cli">./target/debug/context-graph-cli --help shows all commands</criterion>
</validation_criteria>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli</command>
  <command>./target/debug/context-graph-cli --help</command>
  <command>./target/debug/context-graph-cli session --help</command>
  <command>./target/debug/context-graph-cli inject-context --help</command>
</test_commands>
</task_spec>
```
