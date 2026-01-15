# TASK-HOOKS-017: Create End-to-End Tests with Real MCP Calls

```xml
<task_spec id="TASK-HOOKS-017" version="1.0">
<metadata>
  <title>Create End-to-End Tests with Real MCP Calls</title>
  <status>ready</status>
  <layer>surface</layer>
  <sequence>17</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-43</requirement_ref>
    <requirement_ref>REQ-HOOKS-44</requirement_ref>
    <requirement_ref>REQ-HOOKS-45</requirement_ref>
    <requirement_ref>REQ-HOOKS-46</requirement_ref>
    <requirement_ref>REQ-HOOKS-47</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-014</task_ref>
    <task_ref>TASK-HOOKS-015</task_ref>
    <task_ref>TASK-HOOKS-016</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
  <estimated_hours>4.0</estimated_hours>
</metadata>

<context>
End-to-end tests verify the complete hook system works with real shell scripts,
real Claude Code settings, and real MCP server. These tests simulate actual
Claude Code sessions.

Test flow:
1. Start MCP server
2. Execute shell scripts as Claude Code would
3. Verify MCP tool calls occurred
4. Verify state changes in GWT
5. Verify identity persistence
</context>

<input_context_files>
  <file purpose="shell_scripts">.claude/hooks/</file>
  <file purpose="settings">.claude/settings.json</file>
  <file purpose="mcp_server">crates/context-graph-mcp/src/server.rs</file>
  <file purpose="integration_tests">crates/context-graph-cli/tests/integration/</file>
</input_context_files>

<prerequisites>
  <check>Shell scripts created (TASK-HOOKS-014)</check>
  <check>settings.json configured (TASK-HOOKS-015)</check>
  <check>Integration tests passing (TASK-HOOKS-016)</check>
  <check>MCP server operational</check>
</prerequisites>

<scope>
  <in_scope>
    - Full session simulation (start → work → end)
    - Shell script execution verification
    - MCP tool call verification (audit log)
    - GWT state verification after hooks
    - Identity persistence verification
    - Error scenario E2E tests
    - Timeout behavior verification
  </in_scope>
  <out_of_scope>
    - Actual Claude Code process spawning (not possible in test)
    - UI interaction testing
    - Cross-platform testing (Linux-only for now)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/tests/e2e/full_session_test.rs">
      #[tokio::test]
      async fn test_e2e_full_session_workflow();

      #[tokio::test]
      async fn test_e2e_identity_continuity_across_sessions();

      #[tokio::test]
      async fn test_e2e_consciousness_state_updates();

      #[tokio::test]
      async fn test_e2e_hook_error_recovery();

      #[tokio::test]
      async fn test_e2e_shell_script_timeout();
    </signature>
    <signature file="crates/context-graph-cli/tests/e2e/helpers.rs">
      pub async fn start_test_mcp_server() -> McpServerHandle;
      pub async fn execute_hook_script(script: &amp;str, input: &amp;str) -> Result&lt;String, E2EError&gt;;
      pub async fn verify_mcp_tool_called(tool_name: &amp;str) -> bool;
      pub async fn get_gwt_state() -> GwtState;
    </signature>
  </signatures>

  <constraints>
    - NO MOCKS - real MCP server, real shell execution
    - Tests must clean up all state (snapshots, temp files)
    - Tests must be idempotent (runnable multiple times)
    - Tests must work in fresh environment (no prior state)
    - Shell scripts must be tested as Claude Code would execute them
    - Audit trail must be verifiable
  </constraints>

  <verification>
    - cargo test --package context-graph-cli --test e2e
    - Tests pass with real MCP server running
    - State changes visible in GWT after tests
    - Snapshots created in .claude/identity/snapshots/
  </verification>
</definition_of_done>

<pseudo_code>
test_e2e_full_session_workflow():
  // Setup real environment
  mcp_handle = start_test_mcp_server().await
  ensure_clean_state()

  // 1. Execute session_start.sh as Claude Code would
  start_input = json!({
    "session_id": "e2e-test-001",
    "timestamp": now().to_rfc3339()
  })
  start_output = execute_hook_script(
    ".claude/hooks/session_start.sh",
    start_input.to_string()
  ).await?

  // Verify consciousness brief was output
  assert start_output.contains("Kuramoto")
  assert start_output.contains("Identity")

  // Verify MCP tool was called
  assert verify_mcp_tool_called("get_consciousness_state").await

  // 2. Execute pre_tool_use.sh
  pre_input = json!({
    "tool_name": "Read",
    "tool_input": { "file_path": "/test/file.rs" }
  })
  pre_output = execute_hook_script(
    ".claude/hooks/pre_tool_use.sh",
    pre_input.to_string()
  ).await?

  // Verify inject_context was called
  assert verify_mcp_tool_called("inject_context").await

  // 3. Execute post_tool_use.sh
  post_input = json!({
    "tool_name": "Read",
    "tool_result": { "content": "test content" }
  })
  execute_hook_script(
    ".claude/hooks/post_tool_use.sh",
    post_input.to_string()
  ).await?

  // 4. Execute session_end.sh
  end_input = json!({
    "session_id": "e2e-test-001",
    "stats": { "tool_count": 1, "duration_ms": 5000 }
  })
  end_output = execute_hook_script(
    ".claude/hooks/session_end.sh",
    end_input.to_string()
  ).await?

  // Verify snapshot was created
  assert end_output.contains("Snapshot")
  snapshots = list_snapshots(".claude/identity/snapshots/")
  assert snapshots.iter().any(|s| s.session_id == "e2e-test-001")

  // Cleanup
  mcp_handle.stop().await


test_e2e_identity_continuity_across_sessions():
  mcp_handle = start_test_mcp_server().await

  // Session 1: Create identity state
  session1_id = "e2e-session-001"
  execute_session(session1_id).await

  // Get IC after session 1
  ic_after_session1 = get_gwt_state().await.identity_continuity

  // Session 2: Should restore from session 1
  session2_id = "e2e-session-002"
  start_output = execute_hook_script(
    ".claude/hooks/session_start.sh",
    json!({ "session_id": session2_id }).to_string()
  ).await?

  // Verify restoration happened
  ic_after_restore = get_gwt_state().await.identity_continuity

  // IC should be continuous (within drift tolerance)
  assert (ic_after_restore - ic_after_session1).abs() < 0.1

  mcp_handle.stop().await


execute_hook_script(script_path, input):
  // Use std::process::Command to execute shell script
  let output = Command::new("bash")
    .arg(script_path)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()?

  // Write input to stdin
  output.stdin.write_all(input.as_bytes())?

  // Wait with timeout
  let result = tokio::time::timeout(
    Duration::from_millis(5000),
    output.wait_with_output()
  ).await??

  if !result.status.success() {
    return Err(E2EError::ScriptFailed(
      String::from_utf8_lossy(&result.stderr).to_string()
    ))
  }

  Ok(String::from_utf8_lossy(&result.stdout).to_string())
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/tests/e2e/full_session_test.rs">
    Full session workflow E2E tests
  </file>
  <file path="crates/context-graph-cli/tests/e2e/identity_continuity_test.rs">
    Identity continuity across sessions E2E tests
  </file>
  <file path="crates/context-graph-cli/tests/e2e/error_recovery_test.rs">
    Error recovery and timeout E2E tests
  </file>
  <file path="crates/context-graph-cli/tests/e2e/helpers.rs">
    E2E test helper functions
  </file>
  <file path="crates/context-graph-cli/tests/e2e/mod.rs">
    E2E test module exports
  </file>
</files_to_create>

<files_to_modify>
  <!-- None - all new test files -->
</files_to_modify>

<test_commands>
  <command>cargo test --package context-graph-cli --test e2e -- --test-threads=1</command>
  <command>cargo test --package context-graph-cli e2e_full_session</command>
  <command>cargo test --package context-graph-cli e2e_identity_continuity</command>
</test_commands>
</task_spec>
```
