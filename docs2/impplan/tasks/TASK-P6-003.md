# Task: TASK-P6-003 - Inject Context Command

```xml
<task_spec id="TASK-P6-003" version="1.0">
<metadata>
  <title>Inject Context Command</title>
  <phase>6</phase>
  <sequence>45</sequence>
  <layer>surface</layer>
  <estimated_loc>150</estimated_loc>
  <dependencies>
    <dependency task="TASK-P6-001">CLI infrastructure</dependency>
    <dependency task="TASK-P5-007">InjectionPipeline</dependency>
    <dependency task="TASK-P2-005">MultiArrayProvider for embedding</dependency>
  </dependencies>
  <produces>
    <artifact type="function">handle_inject_context</artifact>
  </produces>
</metadata>

<context>
  <background>
    The inject-context command is called by SessionStart and UserPromptSubmit hooks.
    It embeds the query, finds relevant memories, computes divergence, and outputs
    formatted context markdown to stdout for Claude Code to inject.
  </background>
  <business_value>
    This is the primary context injection mechanism. It surfaces relevant past
    work and divergence alerts to help Claude Code understand the user's context.
  </business_value>
  <technical_context>
    Query comes from --query flag or USER_PROMPT environment variable.
    Session ID comes from --session-id flag or CLAUDE_SESSION_ID env var.
    Output goes directly to stdout for Claude Code hook capture.
  </technical_context>
</context>

<prerequisites>
  <prerequisite type="code">crates/context-graph-cli/src/main.rs with InjectContext command</prerequisite>
  <prerequisite type="code">crates/context-graph-core/src/injection/pipeline.rs with InjectionPipeline</prerequisite>
  <prerequisite type="code">crates/context-graph-core/src/embedding/provider.rs with MultiArrayProvider</prerequisite>
</prerequisites>

<scope>
  <includes>
    <item>handle_inject_context() function</item>
    <item>Query embedding via MultiArrayProvider</item>
    <item>Context generation via InjectionPipeline</item>
    <item>Environment variable fallbacks</item>
    <item>Empty output handling (no relevant context)</item>
  </includes>
  <excludes>
    <item>Brief context injection (TASK-P6-004)</item>
    <item>Memory capture (TASK-P6-005, P6-006)</item>
  </excludes>
</scope>

<definition_of_done>
  <criterion id="DOD-1">
    <description>inject-context outputs formatted markdown to stdout</description>
    <verification>Output contains "## Relevant Context" header</verification>
  </criterion>
  <criterion id="DOD-2">
    <description>Empty query returns empty output (not error)</description>
    <verification>Exit code 0, empty stdout</verification>
  </criterion>
  <criterion id="DOD-3">
    <description>No relevant context returns empty output</description>
    <verification>Exit code 0, empty stdout</verification>
  </criterion>
  <criterion id="DOD-4">
    <description>Environment variable fallbacks work</description>
    <verification>USER_PROMPT env var used when --query not provided</verification>
  </criterion>

  <signatures>
    <signature name="handle_inject_context">
      <code>
pub async fn handle_inject_context(
    ctx: &amp;CliContext,
    query: Option&lt;String&gt;,
    session_id: Option&lt;String&gt;,
    budget: u32,
) -> Result&lt;(), CliError&gt;
      </code>
    </signature>
  </signatures>

  <constraints>
    <constraint type="output">Only context markdown to stdout</constraint>
    <constraint type="output">Empty string if no relevant context (not error)</constraint>
    <constraint type="performance">Complete in &lt;4500ms (within hook timeout)</constraint>
    <constraint type="budget">Default budget 1200 tokens per TECH-PHASE5</constraint>
  </constraints>
</definition_of_done>

<pseudo_code>
```rust
// crates/context-graph-cli/src/commands/inject.rs

use tracing::{info, debug, warn};
use crate::config::CliContext;
use crate::error::CliError;
use crate::env::get_env_or_arg;
use crate::commands::session::read_current_session;

use context_graph_core::embedding::MultiArrayProvider;
use context_graph_core::injection::{InjectionPipeline, TokenBudget, DEFAULT_TOKEN_BUDGET};

/// Handle inject-context command.
/// Embeds query, retrieves context, outputs markdown to stdout.
pub async fn handle_inject_context(
    ctx: &CliContext,
    query_arg: Option<String>,
    session_id_arg: Option<String>,
    budget: u32,
) -> Result<(), CliError> {
    // Get query from arg or environment
    let query = get_env_or_arg("USER_PROMPT", query_arg);

    let query = match query {
        Some(q) if !q.trim().is_empty() => q,
        _ => {
            // No query = no context to inject
            debug!("No query provided, returning empty context");
            return Ok(());
        }
    };

    // Get session ID from arg, environment, or file
    let session_id = get_env_or_arg("CLAUDE_SESSION_ID", session_id_arg)
        .or_else(|| read_current_session(ctx));

    let session_id = session_id.unwrap_or_else(|| {
        warn!("No session ID available");
        "unknown".to_string()
    });

    info!(query_len = query.len(), session_id = %session_id, "Injecting context");

    // Create embedding provider
    let provider = MultiArrayProvider::new(ctx.db.clone()).await?;

    // Embed the query
    let query_embedding = provider.embed(&query).await?;

    // Create injection pipeline
    let pipeline = InjectionPipeline::new(ctx.db.clone()).await?;

    // Build token budget
    let token_budget = if budget != 1200 {
        TokenBudget::with_total(budget)
    } else {
        DEFAULT_TOKEN_BUDGET
    };

    // Generate context
    let result = pipeline
        .generate_context(&query_embedding, &session_id, &token_budget)
        .await?;

    // Output result
    if result.is_empty() {
        debug!("No relevant context found");
        // Empty stdout = no injection
    } else {
        info!(
            memories = result.memory_count(),
            tokens = result.tokens_used,
            "Context generated"
        );
        // Print context to stdout
        print!("{}", result.formatted_context);
    }

    Ok(())
}

/// Handle inject-brief command.
/// Simplified context for PreToolUse hook.
pub async fn handle_inject_brief(
    ctx: &CliContext,
    query_arg: Option<String>,
    budget: u32,
) -> Result<(), CliError> {
    // Get query from arg or environment
    let query = get_env_or_arg("TOOL_DESCRIPTION", query_arg)
        .or_else(|| get_env_or_arg("TOOL_NAME", None));

    let query = match query {
        Some(q) if !q.trim().is_empty() => q,
        _ => {
            debug!("No query provided for brief context");
            return Ok(());
        }
    };

    info!(query_len = query.len(), budget, "Injecting brief context");

    // Create embedding provider
    let provider = MultiArrayProvider::new(ctx.db.clone()).await?;

    // Embed the query
    let query_embedding = provider.embed(&query).await?;

    // Create injection pipeline
    let pipeline = InjectionPipeline::new(ctx.db.clone()).await?;

    // Generate brief context
    let result = pipeline
        .generate_brief_context(&query_embedding, budget)
        .await?;

    // Output result
    if result.is_empty() {
        debug!("No relevant context for brief injection");
    } else {
        info!(result_len = result.len(), "Brief context generated");
        print!("{}", result);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_query_returns_ok() {
        // Test that empty query doesn't cause error
        // This requires a mock CliContext
    }

    #[test]
    fn test_env_var_fallback() {
        std::env::set_var("USER_PROMPT", "test query");
        let query = get_env_or_arg("USER_PROMPT", None);
        assert_eq!(query, Some("test query".to_string()));
        std::env::remove_var("USER_PROMPT");
    }

    #[test]
    fn test_arg_priority_over_env() {
        std::env::set_var("USER_PROMPT", "env query");
        let query = get_env_or_arg("USER_PROMPT", Some("arg query".to_string()));
        assert_eq!(query, Some("arg query".to_string()));
        std::env::remove_var("USER_PROMPT");
    }
}
```
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/commands/inject.rs">
    inject-context and inject-brief command handlers
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/commands/mod.rs">
    Add pub mod inject;
  </file>
</files_to_modify>

<validation_criteria>
  <criterion type="compilation">cargo build --package context-graph-cli compiles</criterion>
  <criterion type="test">cargo test commands::inject --package context-graph-cli -- all tests pass</criterion>
  <criterion type="cli">./context-graph-cli inject-context --query "test" outputs markdown or empty</criterion>
</validation_criteria>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test commands::inject --package context-graph-cli</command>
  <command>./target/debug/context-graph-cli inject-context --query "implement HDBSCAN clustering"</command>
  <command>USER_PROMPT="test" ./target/debug/context-graph-cli inject-context</command>
</test_commands>
</task_spec>
```
