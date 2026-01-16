# Task: TASK-P6-004 - Inject Brief Command

```xml
<task_spec id="TASK-P6-004" version="1.0">
<metadata>
  <title>Inject Brief Command</title>
  <phase>6</phase>
  <sequence>46</sequence>
  <layer>surface</layer>
  <estimated_loc>50</estimated_loc>
  <dependencies>
    <dependency task="TASK-P6-001">CLI infrastructure</dependency>
    <dependency task="TASK-P6-003">Inject command infrastructure (shared code)</dependency>
    <dependency task="TASK-P5-007">InjectionPipeline.generate_brief_context()</dependency>
  </dependencies>
  <produces>
    <artifact type="function">handle_inject_brief (in inject.rs)</artifact>
  </produces>
</metadata>

<context>
  <background>
    The inject-brief command is called by PreToolUse hook. It provides a very
    short context summary (max 200 tokens) to give Claude quick context before
    executing a tool, without the overhead of full context generation.
  </background>
  <business_value>
    Enables quick, lightweight context injection for tool use without slowing
    down tool execution with full context retrieval.
  </business_value>
  <technical_context>
    Uses TOOL_DESCRIPTION or TOOL_NAME environment variables as query.
    Default budget is 200 tokens. Output is a simple "Related: ..." format.
  </technical_context>
</context>

<prerequisites>
  <prerequisite type="code">crates/context-graph-cli/src/commands/inject.rs (from TASK-P6-003)</prerequisite>
  <prerequisite type="code">InjectionPipeline.generate_brief_context()</prerequisite>
</prerequisites>

<scope>
  <includes>
    <item>handle_inject_brief() function</item>
    <item>TOOL_DESCRIPTION/TOOL_NAME env var reading</item>
    <item>Brief budget (200 tokens)</item>
  </includes>
  <excludes>
    <item>Full context injection (TASK-P6-003)</item>
  </excludes>
</scope>

<definition_of_done>
  <criterion id="DOD-1">
    <description>inject-brief outputs brief format to stdout</description>
    <verification>Output starts with "Related:" or is empty</verification>
  </criterion>
  <criterion id="DOD-2">
    <description>Output fits within budget (200 tokens default)</description>
    <verification>Word count × 1.3 ≤ budget</verification>
  </criterion>
  <criterion id="DOD-3">
    <description>TOOL_DESCRIPTION env var is read</description>
    <verification>TOOL_DESCRIPTION=test ./context-graph-cli inject-brief works</verification>
  </criterion>
  <criterion id="DOD-4">
    <description>Completes in &lt;400ms</description>
    <verification>time ./context-graph-cli inject-brief shows fast execution</verification>
  </criterion>

  <signatures>
    <signature name="handle_inject_brief">
      <code>
pub async fn handle_inject_brief(
    ctx: &amp;CliContext,
    query: Option&lt;String&gt;,
    budget: u32,
) -> Result&lt;(), CliError&gt;
      </code>
    </signature>
  </signatures>

  <constraints>
    <constraint type="performance">Complete in &lt;400ms (hook timeout is 500ms)</constraint>
    <constraint type="budget">Default budget 200 tokens</constraint>
    <constraint type="output">Brief single-line or short paragraph format</constraint>
  </constraints>
</definition_of_done>

<pseudo_code>
```rust
// Already included in TASK-P6-003 inject.rs
// This task validates the brief-specific behavior

// Additional tests for brief context:

#[cfg(test)]
mod brief_tests {
    use super::*;

    #[test]
    fn test_tool_description_env() {
        std::env::set_var("TOOL_DESCRIPTION", "Running cargo test");
        let query = get_env_or_arg("TOOL_DESCRIPTION", None)
            .or_else(|| get_env_or_arg("TOOL_NAME", None));
        assert_eq!(query, Some("Running cargo test".to_string()));
        std::env::remove_var("TOOL_DESCRIPTION");
    }

    #[test]
    fn test_tool_name_fallback() {
        std::env::remove_var("TOOL_DESCRIPTION");
        std::env::set_var("TOOL_NAME", "Bash");
        let query = get_env_or_arg("TOOL_DESCRIPTION", None)
            .or_else(|| get_env_or_arg("TOOL_NAME", None));
        assert_eq!(query, Some("Bash".to_string()));
        std::env::remove_var("TOOL_NAME");
    }

    #[test]
    fn test_budget_default() {
        // Default should be 200 per BRIEF_BUDGET constant
        use context_graph_core::injection::BRIEF_BUDGET;
        assert_eq!(BRIEF_BUDGET, 200);
    }
}
```
</pseudo_code>

<files_to_modify>
  <file path="crates/context-graph-cli/src/commands/inject.rs">
    Add/update handle_inject_brief tests
  </file>
</files_to_modify>

<validation_criteria>
  <criterion type="compilation">cargo build --package context-graph-cli compiles</criterion>
  <criterion type="test">cargo test commands::inject::brief_tests -- all tests pass</criterion>
  <criterion type="cli">./context-graph-cli inject-brief --query "test" outputs brief format</criterion>
  <criterion type="performance">Brief injection completes in &lt;400ms</criterion>
</validation_criteria>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test commands::inject --package context-graph-cli</command>
  <command>time ./target/debug/context-graph-cli inject-brief --query "test"</command>
  <command>TOOL_DESCRIPTION="Writing file" ./target/debug/context-graph-cli inject-brief</command>
</test_commands>
</task_spec>
```
