# TASK-HOOKS-016: Create Integration Tests for Hook Lifecycle

```xml
<task_spec id="TASK-HOOKS-016" version="1.0">
<metadata>
  <title>Create Integration Tests for Hook Lifecycle</title>
  <status>ready</status>
  <layer>surface</layer>
  <sequence>16</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-40</requirement_ref>
    <requirement_ref>REQ-HOOKS-41</requirement_ref>
    <requirement_ref>REQ-HOOKS-42</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-006</task_ref>
    <task_ref>TASK-HOOKS-007</task_ref>
    <task_ref>TASK-HOOKS-008</task_ref>
    <task_ref>TASK-HOOKS-009</task_ref>
    <task_ref>TASK-HOOKS-012</task_ref>
    <task_ref>TASK-HOOKS-013</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
  <estimated_hours>3.5</estimated_hours>
</metadata>

<context>
Integration tests verify that individual hook components work together correctly.
These tests use REAL MCP calls and REAL file operations - NO MOCKS.

Test categories:
1. Session lifecycle: start → tool use → end
2. Identity persistence: snapshot → restore → verify
3. Consciousness integration: brief → inject → verify state
</context>

<input_context_files>
  <file purpose="hook_handlers">crates/context-graph-cli/src/hooks/</file>
  <file purpose="identity_module">crates/context-graph-cli/src/identity/</file>
  <file purpose="mcp_client">crates/context-graph-mcp/src/client.rs</file>
  <file purpose="test_utilities">crates/context-graph-test-utils/src/</file>
</input_context_files>

<prerequisites>
  <check>All hook handlers implemented</check>
  <check>Snapshot store implemented (TASK-HOOKS-012)</check>
  <check>Restorer implemented (TASK-HOOKS-013)</check>
  <check>MCP server running for tests</check>
</prerequisites>

<scope>
  <in_scope>
    - Session lifecycle integration tests
    - Identity snapshot/restore integration tests
    - Hook timeout verification tests
    - Error propagation tests
    - Concurrent hook execution tests
    - Test fixtures with real data (not mocks)
  </in_scope>
  <out_of_scope>
    - End-to-end Claude Code tests (TASK-HOOKS-017)
    - Performance benchmarks (separate task)
    - UI/UX tests (not applicable)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/tests/integration/hook_lifecycle_test.rs">
      #[tokio::test]
      async fn test_session_lifecycle_full_flow();

      #[tokio::test]
      async fn test_identity_snapshot_and_restore();

      #[tokio::test]
      async fn test_consciousness_state_injection();

      #[tokio::test]
      async fn test_hook_timeout_handling();

      #[tokio::test]
      async fn test_concurrent_tool_hooks();
    </signature>
    <signature file="crates/context-graph-cli/tests/integration/mod.rs">
      mod hook_lifecycle_test;
      mod identity_integration_test;
      mod consciousness_integration_test;
    </signature>
  </signatures>

  <constraints>
    - NO MOCK DATA - all tests use real MCP server
    - Tests must clean up after themselves (temp directories)
    - Tests must be independent (no shared state between tests)
    - Tests must complete within 30 seconds each
    - Failed tests must provide detailed diagnostics
    - Tests must run in CI environment
  </constraints>

  <verification>
    - cargo test --package context-graph-cli --test integration
    - All tests pass with real MCP server
    - No test pollution (can run in any order)
  </verification>
</definition_of_done>

<pseudo_code>
test_session_lifecycle_full_flow():
  // Setup
  mcp_server = start_mcp_server()
  temp_dir = create_temp_dir()

  // 1. Simulate session start
  session_id = "test-session-001"
  start_result = session_start_handler(HookInput {
    session_id: session_id,
    timestamp: now()
  })
  assert start_result.is_ok()
  assert start_result.consciousness_brief.contains("Kuramoto")

  // 2. Simulate tool use
  tool_input = HookInput {
    tool_name: "Read",
    tool_input: { file_path: "/some/file.rs" }
  }
  pre_result = pre_tool_use_handler(tool_input)
  assert pre_result.is_ok()

  post_input = HookInput {
    tool_name: "Read",
    tool_result: { content: "file contents" }
  }
  post_result = post_tool_use_handler(post_input)
  assert post_result.is_ok()

  // 3. Simulate session end
  end_result = session_end_handler(HookInput {
    session_id: session_id,
    stats: { tool_count: 1 }
  })
  assert end_result.is_ok()
  assert end_result.snapshot_id.is_some()

  // Cleanup
  cleanup_temp_dir(temp_dir)
  stop_mcp_server(mcp_server)


test_identity_snapshot_and_restore():
  // Setup with real store
  store = FileSnapshotStore::new(temp_dir, 10)
  restorer = SnapshotRestorer::new(store, gwt_client, Duration::days(7))

  // Create and save snapshot
  snapshot = SessionIdentitySnapshot {
    session_id: "test-001",
    identity_continuity: 0.85,
    kuramoto_phases: [0.1, 0.2, ...],
    purpose_vector: [0.5, 0.3, ...],
    captured_at: now()
  }
  save_result = store.save(&snapshot).await
  assert save_result.is_ok()

  // Restore and verify
  restore_result = restorer.restore_latest().await
  assert restore_result.is_ok()
  assert restore_result.drift.ic_delta.abs() < 0.01  // Should be same session


test_hook_timeout_handling():
  // Test that hooks respect timeout constraints
  slow_handler = || {
    sleep(Duration::ms(3000))  // Exceeds 2000ms limit
  }

  result = with_timeout(Duration::ms(2000), slow_handler)
  assert result.is_err()
  assert result.error_kind() == TimeoutError
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/tests/integration/hook_lifecycle_test.rs">
    Session lifecycle integration tests
  </file>
  <file path="crates/context-graph-cli/tests/integration/identity_integration_test.rs">
    Identity snapshot/restore integration tests
  </file>
  <file path="crates/context-graph-cli/tests/integration/consciousness_integration_test.rs">
    Consciousness brief/inject integration tests
  </file>
  <file path="crates/context-graph-cli/tests/integration/mod.rs">
    Integration test module exports
  </file>
</files_to_create>

<files_to_modify>
  <!-- None - all new test files -->
</files_to_modify>

<test_commands>
  <command>cargo test --package context-graph-cli --test integration -- --test-threads=1</command>
  <command>cargo test --package context-graph-cli hook_lifecycle</command>
  <command>cargo test --package context-graph-cli identity_integration</command>
</test_commands>
</task_spec>
```
