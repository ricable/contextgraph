# Task: TASK-P6-002 - Session Start/End Commands

```xml
<task_spec id="TASK-P6-002" version="1.0">
<metadata>
  <title>Session Start/End Commands</title>
  <phase>6</phase>
  <sequence>44</sequence>
  <layer>surface</layer>
  <estimated_loc>120</estimated_loc>
  <dependencies>
    <dependency task="TASK-P6-001">CLI infrastructure (CliContext, CliError)</dependency>
    <dependency task="TASK-P1-006">SessionManager from core</dependency>
  </dependencies>
  <produces>
    <artifact type="function">handle_session_start</artifact>
    <artifact type="function">handle_session_end</artifact>
  </produces>
</metadata>

<context>
  <background>
    Session commands manage the lifecycle of Claude Code sessions. session start
    creates a new session and outputs its ID (which gets captured by the shell
    script). session end finalizes the session and clears the current session file.
  </background>
  <business_value>
    Sessions group related memories together and enable session-based context
    retrieval and topic tracking.
  </business_value>
  <technical_context>
    Session ID is written to ~/.contextgraph/current_session for shell scripts
    to read. The ID is also printed to stdout for direct capture in scripts.
  </technical_context>
</context>

<prerequisites>
  <prerequisite type="code">crates/context-graph-cli/src/main.rs with Commands enum</prerequisite>
  <prerequisite type="code">crates/context-graph-core/src/memory/session.rs with SessionManager</prerequisite>
</prerequisites>

<scope>
  <includes>
    <item>handle_session_start() function</item>
    <item>handle_session_end() function</item>
    <item>Current session file management</item>
    <item>stdout output of session ID</item>
    <item>Logging of session lifecycle events</item>
  </includes>
  <excludes>
    <item>Session summary capture (TASK-P6-005)</item>
    <item>Context injection (TASK-P6-003)</item>
  </excludes>
</scope>

<definition_of_done>
  <criterion id="DOD-1">
    <description>session start outputs new session ID to stdout</description>
    <verification>./context-graph-cli session start outputs UUID format</verification>
  </criterion>
  <criterion id="DOD-2">
    <description>Session ID written to current_session file</description>
    <verification>cat ~/.contextgraph/current_session matches stdout</verification>
  </criterion>
  <criterion id="DOD-3">
    <description>session end clears current_session file</description>
    <verification>File is empty after session end</verification>
  </criterion>
  <criterion id="DOD-4">
    <description>session end with no active session logs warning, returns Ok</description>
    <verification>Exit code 0, warning in logs</verification>
  </criterion>

  <signatures>
    <signature name="handle_session_start">
      <code>
pub async fn handle_session_start(ctx: &amp;CliContext) -> Result&lt;(), CliError&gt;
      </code>
    </signature>
    <signature name="handle_session_end">
      <code>
pub async fn handle_session_end(ctx: &amp;CliContext) -> Result&lt;(), CliError&gt;
      </code>
    </signature>
  </signatures>

  <constraints>
    <constraint type="output">Only session ID to stdout, nothing else</constraint>
    <constraint type="behavior">session end is idempotent (safe to call multiple times)</constraint>
    <constraint type="file">current_session file uses UTF-8 encoding</constraint>
  </constraints>
</definition_of_done>

<pseudo_code>
```rust
// crates/context-graph-cli/src/commands/session.rs

use std::fs;
use tracing::{info, warn};
use crate::config::CliContext;
use crate::error::CliError;
use context_graph_core::memory::SessionManager;

/// Handle session start command.
/// Creates new session, writes ID to file, prints to stdout.
pub async fn handle_session_start(ctx: &CliContext) -> Result<(), CliError> {
    info!("Starting new session");

    // Create session manager
    let session_manager = SessionManager::new(ctx.db.clone());

    // Start new session
    let session = session_manager.start_session().await?;
    let session_id = session.id.to_string();

    // Write session ID to current_session file
    fs::write(&ctx.config.current_session_file, &session_id)?;

    info!(session_id = %session_id, "Session started");

    // Print session ID to stdout (for shell script capture)
    println!("{}", session_id);

    Ok(())
}

/// Handle session end command.
/// Ends the current session if one exists, clears current_session file.
pub async fn handle_session_end(ctx: &CliContext) -> Result<(), CliError> {
    info!("Ending session");

    // Read current session ID
    let session_id = match fs::read_to_string(&ctx.config.current_session_file) {
        Ok(id) if !id.trim().is_empty() => id.trim().to_string(),
        Ok(_) => {
            warn!("No active session (current_session file is empty)");
            return Ok(());
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            warn!("No active session (current_session file not found)");
            return Ok(());
        }
        Err(e) => return Err(e.into()),
    };

    // Parse session ID
    let session_uuid = match uuid::Uuid::parse_str(&session_id) {
        Ok(id) => id,
        Err(_) => {
            warn!(session_id = %session_id, "Invalid session ID in current_session file");
            // Clear the file anyway
            fs::write(&ctx.config.current_session_file, "")?;
            return Ok(());
        }
    };

    // Create session manager
    let session_manager = SessionManager::new(ctx.db.clone());

    // End session
    match session_manager.end_session(session_uuid).await {
        Ok(_) => {
            info!(session_id = %session_id, "Session ended successfully");
        }
        Err(e) => {
            warn!(session_id = %session_id, error = %e, "Error ending session");
            // Don't fail - just clear the file
        }
    }

    // Clear current_session file
    fs::write(&ctx.config.current_session_file, "")?;

    Ok(())
}

/// Read current session ID from file.
/// Returns None if no active session.
pub fn read_current_session(ctx: &CliContext) -> Option<String> {
    match fs::read_to_string(&ctx.config.current_session_file) {
        Ok(id) if !id.trim().is_empty() => Some(id.trim().to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_config(temp_dir: &TempDir) -> CliConfig {
        CliConfig {
            db_path: temp_dir.path().join("db"),
            log_path: temp_dir.path().join("logs"),
            current_session_file: temp_dir.path().join("current_session"),
            verbose: false,
            timeout_ms: 5000,
        }
    }

    #[test]
    fn test_read_current_session_none() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir);
        let ctx = CliContext {
            config,
            db: Arc::new(/* mock db */),
        };

        // File doesn't exist
        assert!(read_current_session(&ctx).is_none());
    }

    #[test]
    fn test_read_current_session_empty() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir);

        // Create empty file
        fs::write(&config.current_session_file, "").unwrap();

        let ctx = CliContext {
            config,
            db: Arc::new(/* mock db */),
        };

        assert!(read_current_session(&ctx).is_none());
    }

    #[test]
    fn test_read_current_session_valid() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir);
        let session_id = "550e8400-e29b-41d4-a716-446655440000";

        // Create file with session ID
        fs::write(&config.current_session_file, session_id).unwrap();

        let ctx = CliContext {
            config,
            db: Arc::new(/* mock db */),
        };

        assert_eq!(read_current_session(&ctx), Some(session_id.to_string()));
    }
}
```
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/commands/session.rs">
    Session start/end command handlers
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/commands/mod.rs">
    Add pub mod session;
  </file>
</files_to_modify>

<validation_criteria>
  <criterion type="compilation">cargo build --package context-graph-cli compiles</criterion>
  <criterion type="test">cargo test commands::session --package context-graph-cli -- all tests pass</criterion>
  <criterion type="cli">./context-graph-cli session start outputs valid UUID</criterion>
</validation_criteria>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test commands::session --package context-graph-cli</command>
  <command>./target/debug/context-graph-cli session start</command>
  <command>cat ~/.contextgraph/current_session</command>
  <command>./target/debug/context-graph-cli session end</command>
</test_commands>
</task_spec>
```
