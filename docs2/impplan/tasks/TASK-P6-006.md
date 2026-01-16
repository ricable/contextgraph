# Task: TASK-P6-006 - Capture Response Command

```xml
<task_spec id="TASK-P6-006" version="1.0">
<metadata>
  <title>Capture Response Command</title>
  <phase>6</phase>
  <sequence>48</sequence>
  <layer>surface</layer>
  <estimated_loc>50</estimated_loc>
  <dependencies>
    <dependency task="TASK-P6-001">CLI infrastructure</dependency>
    <dependency task="TASK-P6-005">Capture command infrastructure (shared code)</dependency>
    <dependency task="TASK-P1-007">MemoryCaptureService</dependency>
  </dependencies>
  <produces>
    <artifact type="function">handle_capture_response (in capture.rs)</artifact>
  </produces>
</metadata>

<context>
  <background>
    The capture-response command is called by the Stop hook. It captures
    Claude's response summaries as memories, enabling the system to learn
    from what Claude actually said/did, not just what tools were used.
  </background>
  <business_value>
    Captures Claude's actual responses, enabling more nuanced context retrieval
    that understands outcomes, not just actions.
  </business_value>
  <technical_context>
    Content comes from --content flag or RESPONSE_SUMMARY env var.
    This is a simpler variant of capture-memory focused on response capture.
  </technical_context>
</context>

<prerequisites>
  <prerequisite type="code">crates/context-graph-cli/src/commands/capture.rs (from TASK-P6-005)</prerequisite>
  <prerequisite type="code">MemoryCaptureService.capture_claude_response()</prerequisite>
</prerequisites>

<scope>
  <includes>
    <item>handle_capture_response() function</item>
    <item>RESPONSE_SUMMARY env var reading</item>
    <item>Silent operation (no stdout)</item>
  </includes>
  <excludes>
    <item>Memory capture from hooks (TASK-P6-005)</item>
  </excludes>
</scope>

<definition_of_done>
  <criterion id="DOD-1">
    <description>capture-response creates memory in database</description>
    <verification>Memory count increases after capture</verification>
  </criterion>
  <criterion id="DOD-2">
    <description>RESPONSE_SUMMARY env var is used</description>
    <verification>RESPONSE_SUMMARY=test ./context-graph-cli capture-response works</verification>
  </criterion>
  <criterion id="DOD-3">
    <description>Empty content is silently ignored</description>
    <verification>Exit code 0, no error</verification>
  </criterion>

  <signatures>
    <signature name="handle_capture_response">
      <code>
pub async fn handle_capture_response(
    ctx: &amp;CliContext,
    content: Option&lt;String&gt;,
    session_id: Option&lt;String&gt;,
) -> Result&lt;(), CliError&gt;
      </code>
    </signature>
  </signatures>

  <constraints>
    <constraint type="output">No stdout output</constraint>
    <constraint type="behavior">Empty content is silently ignored</constraint>
    <constraint type="performance">Complete in &lt;2700ms (within Stop hook timeout)</constraint>
  </constraints>
</definition_of_done>

<pseudo_code>
```rust
// Already included in TASK-P6-005 capture.rs
// This task validates the response-specific behavior

// Additional tests for response capture:

#[cfg(test)]
mod response_tests {
    use super::*;

    #[test]
    fn test_response_summary_env() {
        std::env::set_var("RESPONSE_SUMMARY", "Created new file");
        let content = get_env_or_arg("RESPONSE_SUMMARY", None);
        assert_eq!(content, Some("Created new file".to_string()));
        std::env::remove_var("RESPONSE_SUMMARY");
    }

    #[test]
    fn test_empty_response_ignored() {
        std::env::remove_var("RESPONSE_SUMMARY");
        let content = get_env_or_arg("RESPONSE_SUMMARY", None);
        assert_eq!(content, None);
        // Empty content should return Ok without creating memory
    }
}
```
</pseudo_code>

<files_to_modify>
  <file path="crates/context-graph-cli/src/commands/capture.rs">
    Add/update handle_capture_response tests
  </file>
</files_to_modify>

<validation_criteria>
  <criterion type="compilation">cargo build --package context-graph-cli compiles</criterion>
  <criterion type="test">cargo test commands::capture::response_tests -- all tests pass</criterion>
  <criterion type="cli">./context-graph-cli capture-response --content "test" exits 0</criterion>
</validation_criteria>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test commands::capture --package context-graph-cli</command>
  <command>./target/debug/context-graph-cli capture-response --content "File created successfully"</command>
  <command>RESPONSE_SUMMARY="test" ./target/debug/context-graph-cli capture-response</command>
</test_commands>
</task_spec>
```
