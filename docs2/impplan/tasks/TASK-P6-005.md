# Task: TASK-P6-005 - Capture Memory Command

```xml
<task_spec id="TASK-P6-005" version="1.0">
<metadata>
  <title>Capture Memory Command</title>
  <phase>6</phase>
  <sequence>47</sequence>
  <layer>surface</layer>
  <estimated_loc>120</estimated_loc>
  <dependencies>
    <dependency task="TASK-P6-001">CLI infrastructure</dependency>
    <dependency task="TASK-P1-007">MemoryCaptureService</dependency>
  </dependencies>
  <produces>
    <artifact type="function">handle_capture_memory</artifact>
  </produces>
</metadata>

<context>
  <background>
    The capture-memory command is called by PostToolUse and SessionEnd hooks.
    It captures tool descriptions and session summaries as memories, embedding
    them and storing them for future retrieval.
  </background>
  <business_value>
    Enables automatic memory capture from Claude Code tool usage, building
    the knowledge base that powers context injection.
  </business_value>
  <technical_context>
    Content comes from --content flag or TOOL_DESCRIPTION env var.
    Source must be "hook" for hook-originating content.
    Hook type and tool name provide additional context for categorization.
  </technical_context>
</context>

<prerequisites>
  <prerequisite type="code">crates/context-graph-cli/src/main.rs with CaptureMemory command</prerequisite>
  <prerequisite type="code">crates/context-graph-core/src/memory/capture.rs with MemoryCaptureService</prerequisite>
</prerequisites>

<scope>
  <includes>
    <item>handle_capture_memory() function</item>
    <item>TOOL_DESCRIPTION env var reading</item>
    <item>Hook type parsing (PostToolUse, SessionEnd, etc.)</item>
    <item>Session ID retrieval from file/env</item>
    <item>Silent operation (no stdout output)</item>
  </includes>
  <excludes>
    <item>Response capture (TASK-P6-006)</item>
    <item>Context injection (TASK-P6-003, P6-004)</item>
  </excludes>
</scope>

<definition_of_done>
  <criterion id="DOD-1">
    <description>capture-memory creates memory in database</description>
    <verification>Memory count increases after capture</verification>
  </criterion>
  <criterion id="DOD-2">
    <description>No stdout output (silent capture)</description>
    <verification>stdout is empty</verification>
  </criterion>
  <criterion id="DOD-3">
    <description>TOOL_DESCRIPTION env var is used</description>
    <verification>TOOL_DESCRIPTION=test ./context-graph-cli capture-memory works</verification>
  </criterion>
  <criterion id="DOD-4">
    <description>Empty content is silently ignored</description>
    <verification>Exit code 0, no error for empty content</verification>
  </criterion>

  <signatures>
    <signature name="handle_capture_memory">
      <code>
pub async fn handle_capture_memory(
    ctx: &amp;CliContext,
    content: Option&lt;String&gt;,
    source: String,
    session_id: Option&lt;String&gt;,
    hook_type: Option&lt;String&gt;,
    tool_name: Option&lt;String&gt;,
) -> Result&lt;(), CliError&gt;
      </code>
    </signature>
  </signatures>

  <constraints>
    <constraint type="output">No stdout output</constraint>
    <constraint type="behavior">Empty content is silently ignored (not error)</constraint>
    <constraint type="performance">Complete in &lt;2700ms (within hook timeout)</constraint>
  </constraints>
</definition_of_done>

<pseudo_code>
```rust
// crates/context-graph-cli/src/commands/capture.rs

use tracing::{info, debug, warn};
use crate::config::CliContext;
use crate::error::CliError;
use crate::env::get_env_or_arg;
use crate::commands::session::read_current_session;

use context_graph_core::memory::{MemoryCaptureService, MemorySource, HookType};

/// Handle capture-memory command.
/// Captures tool descriptions and hook content as memories.
pub async fn handle_capture_memory(
    ctx: &CliContext,
    content_arg: Option<String>,
    source: String,
    session_id_arg: Option<String>,
    hook_type_arg: Option<String>,
    tool_name: Option<String>,
) -> Result<(), CliError> {
    // Get content from arg or environment
    let content = get_env_or_arg("TOOL_DESCRIPTION", content_arg)
        .or_else(|| get_env_or_arg("SESSION_SUMMARY", None));

    let content = match content {
        Some(c) if !c.trim().is_empty() => c,
        _ => {
            debug!("No content to capture");
            return Ok(());
        }
    };

    // Get session ID
    let session_id = get_env_or_arg("CLAUDE_SESSION_ID", session_id_arg)
        .or_else(|| read_current_session(ctx));

    let session_id = match session_id {
        Some(id) => {
            uuid::Uuid::parse_str(&id).ok()
        }
        None => {
            warn!("No session ID available for capture");
            None
        }
    };

    // Parse source
    let memory_source = match source.to_lowercase().as_str() {
        "hook" => MemorySource::Hook,
        "response" => MemorySource::Response,
        other => {
            warn!(source = other, "Unknown source, defaulting to Hook");
            MemorySource::Hook
        }
    };

    // Parse hook type if provided
    let hook_type = hook_type_arg.and_then(|ht| match ht.to_lowercase().as_str() {
        "posttooluse" | "post_tool_use" => Some(HookType::PostToolUse),
        "sessionend" | "session_end" => Some(HookType::SessionEnd),
        "userpromptsubmit" | "user_prompt_submit" => Some(HookType::UserPromptSubmit),
        "sessionstart" | "session_start" => Some(HookType::SessionStart),
        _ => {
            warn!(hook_type = ht, "Unknown hook type");
            None
        }
    });

    info!(
        content_len = content.len(),
        source = ?memory_source,
        hook_type = ?hook_type,
        tool_name = ?tool_name,
        "Capturing memory"
    );

    // Create capture service
    let capture_service = MemoryCaptureService::new(ctx.db.clone()).await?;

    // Capture the memory
    let metadata = CaptureMetadata {
        source: memory_source,
        hook_type,
        tool_name,
        session_id,
    };

    capture_service.capture_hook_description(&content, metadata).await?;

    info!("Memory captured successfully");

    // No stdout output - silent capture
    Ok(())
}

/// Handle capture-response command.
/// Captures Claude's response summaries.
pub async fn handle_capture_response(
    ctx: &CliContext,
    content_arg: Option<String>,
    session_id_arg: Option<String>,
) -> Result<(), CliError> {
    // Get content from arg or environment
    let content = get_env_or_arg("RESPONSE_SUMMARY", content_arg);

    let content = match content {
        Some(c) if !c.trim().is_empty() => c,
        _ => {
            debug!("No response content to capture");
            return Ok(());
        }
    };

    // Get session ID
    let session_id = get_env_or_arg("CLAUDE_SESSION_ID", session_id_arg)
        .or_else(|| read_current_session(ctx));

    let session_id = match session_id {
        Some(id) => uuid::Uuid::parse_str(&id).ok(),
        None => {
            warn!("No session ID available for response capture");
            None
        }
    };

    info!(content_len = content.len(), "Capturing response");

    // Create capture service
    let capture_service = MemoryCaptureService::new(ctx.db.clone()).await?;

    // Capture the response
    let metadata = CaptureMetadata {
        source: MemorySource::Response,
        hook_type: Some(HookType::Stop),
        tool_name: None,
        session_id,
    };

    capture_service.capture_claude_response(&content, metadata).await?;

    info!("Response captured successfully");

    // No stdout output
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_content_returns_ok() {
        // Empty content should not cause error
        let result = handle_capture_memory(
            &test_ctx(),
            None,
            "hook".to_string(),
            None,
            None,
            None,
        );
        // Would need async test setup
    }

    #[test]
    fn test_hook_type_parsing() {
        let ht = "PostToolUse";
        let parsed = match ht.to_lowercase().as_str() {
            "posttooluse" | "post_tool_use" => Some(HookType::PostToolUse),
            _ => None,
        };
        assert_eq!(parsed, Some(HookType::PostToolUse));
    }

    #[test]
    fn test_source_parsing() {
        let source = "hook";
        let parsed = match source.to_lowercase().as_str() {
            "hook" => MemorySource::Hook,
            "response" => MemorySource::Response,
            _ => MemorySource::Hook,
        };
        assert_eq!(parsed, MemorySource::Hook);
    }
}
```
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/commands/capture.rs">
    capture-memory and capture-response command handlers
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/commands/mod.rs">
    Add pub mod capture;
  </file>
</files_to_modify>

<validation_criteria>
  <criterion type="compilation">cargo build --package context-graph-cli compiles</criterion>
  <criterion type="test">cargo test commands::capture --package context-graph-cli -- all tests pass</criterion>
  <criterion type="cli">./context-graph-cli capture-memory --content "test" --source hook exits 0</criterion>
</validation_criteria>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test commands::capture --package context-graph-cli</command>
  <command>./target/debug/context-graph-cli capture-memory --content "Implemented HDBSCAN" --source hook --hook-type PostToolUse</command>
  <command>TOOL_DESCRIPTION="test" ./target/debug/context-graph-cli capture-memory --source hook</command>
</test_commands>
</task_spec>
```
