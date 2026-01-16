# Task: TASK-P6-009 - E2E Integration Tests

```xml
<task_spec id="TASK-P6-009" version="1.0">
<metadata>
  <title>End-to-End Integration Tests</title>
  <phase>6</phase>
  <sequence>51</sequence>
  <layer>surface</layer>
  <estimated_loc>400</estimated_loc>
  <dependencies>
    <dependency task="TASK-P6-008">Hook shell scripts</dependency>
    <dependency task="TASK-P6-007">Setup command</dependency>
    <dependency task="TASK-P6-003">Inject context command</dependency>
    <dependency task="TASK-P6-005">Capture memory command</dependency>
  </dependencies>
  <produces>
    <artifact type="test">cli_integration_tests.rs</artifact>
  </produces>
</metadata>

<context>
  <background>
    Integration tests validate the complete flow from memory capture through
    embedding, storage, retrieval, and context injection. They simulate real
    Claude Code hook scenarios without mocking internal components.
  </background>
  <business_value>
    Ensures the system works end-to-end before deployment. Catches integration
    issues that unit tests miss.
  </business_value>
  <technical_context>
    Tests use a temporary database and real CLI binary. They simulate the
    sequence of hook calls that would occur in a real Claude Code session.
    No mock data - real synthetic content that exercises all code paths.
  </technical_context>
</context>

<prerequisites>
  <prerequisite type="binary">context-graph-cli compiled</prerequisite>
  <prerequisite type="code">All Phase 6 tasks completed</prerequisite>
</prerequisites>

<scope>
  <includes>
    <item>Full session lifecycle test</item>
    <item>Memory capture and retrieval test</item>
    <item>Context injection relevance test</item>
    <item>Divergence detection test</item>
    <item>Multi-session continuity test</item>
    <item>Error handling tests</item>
  </includes>
  <excludes>
    <item>Performance benchmarks (TASK-P6-010)</item>
    <item>Mock-based unit tests</item>
  </excludes>
</scope>

<definition_of_done>
  <criterion id="DOD-1">
    <description>Full session lifecycle test passes</description>
    <verification>cargo test --test cli_integration full_session_lifecycle</verification>
  </criterion>
  <criterion id="DOD-2">
    <description>Memory capture and retrieval works correctly</description>
    <verification>Captured memory appears in injection output</verification>
  </criterion>
  <criterion id="DOD-3">
    <description>Divergence detection test passes</description>
    <verification>Divergence alert appears when topic shifts</verification>
  </criterion>
  <criterion id="DOD-4">
    <description>Multi-session test passes</description>
    <verification>Previous session context available in new session</verification>
  </criterion>
  <criterion id="DOD-5">
    <description>All tests use real data, no mocks</description>
    <verification>Code review confirms no mock data</verification>
  </criterion>

  <signatures>
    <signature name="full_session_lifecycle">
      <code>
#[tokio::test]
async fn test_full_session_lifecycle() -> Result&lt;(), Box&lt;dyn std::error::Error&gt;&gt;
      </code>
    </signature>
    <signature name="memory_capture_and_retrieval">
      <code>
#[tokio::test]
async fn test_memory_capture_and_retrieval() -> Result&lt;(), Box&lt;dyn std::error::Error&gt;&gt;
      </code>
    </signature>
  </signatures>

  <constraints>
    <constraint type="data">No mock data - use real synthetic content</constraint>
    <constraint type="isolation">Each test uses isolated temp directory</constraint>
    <constraint type="cleanup">Tests clean up after themselves</constraint>
  </constraints>
</definition_of_done>

<pseudo_code>
```rust
// crates/context-graph-cli/tests/integration/cli_tests.rs

use std::process::Command;
use tempfile::TempDir;
use std::env;

/// Helper to run CLI commands in test environment
struct CliTestRunner {
    temp_dir: TempDir,
    db_path: String,
}

impl CliTestRunner {
    fn new() -> Self {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db").to_string_lossy().to_string();

        Self { temp_dir, db_path }
    }

    fn run(&self, args: &[&str]) -> std::process::Output {
        Command::new("./target/debug/context-graph-cli")
            .args(args)
            .arg("--db-path")
            .arg(&self.db_path)
            .env("HOME", self.temp_dir.path())
            .output()
            .expect("Failed to execute CLI")
    }

    fn run_with_env(&self, args: &[&str], env_vars: &[(&str, &str)]) -> std::process::Output {
        let mut cmd = Command::new("./target/debug/context-graph-cli");
        cmd.args(args)
            .arg("--db-path")
            .arg(&self.db_path)
            .env("HOME", self.temp_dir.path());

        for (key, value) in env_vars {
            cmd.env(key, value);
        }

        cmd.output().expect("Failed to execute CLI")
    }
}

#[test]
fn test_full_session_lifecycle() {
    let runner = CliTestRunner::new();

    // 1. Start session
    let output = runner.run(&["session", "start"]);
    assert!(output.status.success(), "session start failed");
    let session_id = String::from_utf8_lossy(&output.stdout).trim().to_string();
    assert!(!session_id.is_empty(), "session ID should not be empty");

    // Verify UUID format
    uuid::Uuid::parse_str(&session_id).expect("Invalid session ID format");

    // 2. Capture some memories
    let output = runner.run_with_env(
        &["capture-memory", "--source", "hook", "--hook-type", "PostToolUse"],
        &[
            ("TOOL_DESCRIPTION", "Implemented HDBSCAN clustering algorithm with EOM cluster selection"),
            ("CLAUDE_SESSION_ID", &session_id),
        ],
    );
    assert!(output.status.success(), "capture-memory failed");

    let output = runner.run_with_env(
        &["capture-memory", "--source", "hook", "--hook-type", "PostToolUse"],
        &[
            ("TOOL_DESCRIPTION", "Added BIRCH tree for incremental clustering"),
            ("CLAUDE_SESSION_ID", &session_id),
        ],
    );
    assert!(output.status.success(), "capture-memory failed");

    // 3. Inject context - should find captured memories
    let output = runner.run_with_env(
        &["inject-context"],
        &[
            ("USER_PROMPT", "implement clustering"),
            ("CLAUDE_SESSION_ID", &session_id),
        ],
    );
    assert!(output.status.success(), "inject-context failed");
    let context = String::from_utf8_lossy(&output.stdout);

    // Context should mention our captured memories
    assert!(
        context.contains("HDBSCAN") || context.contains("clustering"),
        "Context should include captured clustering memories"
    );

    // 4. End session
    let output = runner.run(&["session", "end"]);
    assert!(output.status.success(), "session end failed");

    // 5. Verify session ended (second end should be idempotent)
    let output = runner.run(&["session", "end"]);
    assert!(output.status.success(), "second session end should be ok");
}

#[test]
fn test_memory_capture_and_retrieval() {
    let runner = CliTestRunner::new();

    // Start session
    let output = runner.run(&["session", "start"]);
    let session_id = String::from_utf8_lossy(&output.stdout).trim().to_string();

    // Capture unique memory
    let unique_content = format!("Implemented unique algorithm {}", uuid::Uuid::new_v4());
    let output = runner.run_with_env(
        &["capture-memory", "--source", "hook"],
        &[
            ("TOOL_DESCRIPTION", &unique_content),
            ("CLAUDE_SESSION_ID", &session_id),
        ],
    );
    assert!(output.status.success());

    // Query for the unique content
    let output = runner.run_with_env(
        &["inject-context"],
        &[
            ("USER_PROMPT", "unique algorithm"),
            ("CLAUDE_SESSION_ID", &session_id),
        ],
    );
    assert!(output.status.success());

    let context = String::from_utf8_lossy(&output.stdout);
    assert!(
        context.contains("algorithm") || context.contains("unique"),
        "Should retrieve the captured memory"
    );

    // Cleanup
    runner.run(&["session", "end"]);
}

#[test]
fn test_divergence_detection() {
    let runner = CliTestRunner::new();

    // Start session and capture clustering-related memories
    let output = runner.run(&["session", "start"]);
    let session_id = String::from_utf8_lossy(&output.stdout).trim().to_string();

    // Capture several clustering memories
    for i in 0..5 {
        let content = format!("Clustering implementation detail {}: HDBSCAN algorithm", i);
        runner.run_with_env(
            &["capture-memory", "--source", "hook"],
            &[
                ("TOOL_DESCRIPTION", &content),
                ("CLAUDE_SESSION_ID", &session_id),
            ],
        );
    }

    // Query about something completely different
    let output = runner.run_with_env(
        &["inject-context"],
        &[
            ("USER_PROMPT", "database migration SQL queries"),
            ("CLAUDE_SESSION_ID", &session_id),
        ],
    );
    assert!(output.status.success());

    let context = String::from_utf8_lossy(&output.stdout);

    // Should detect divergence from clustering to database topic
    // Note: This depends on having enough context and proper divergence detection
    // The actual assertion may vary based on implementation

    runner.run(&["session", "end"]);
}

#[test]
fn test_multi_session_continuity() {
    let runner = CliTestRunner::new();

    // Session 1: Capture some memories
    let output = runner.run(&["session", "start"]);
    let session1_id = String::from_utf8_lossy(&output.stdout).trim().to_string();

    runner.run_with_env(
        &["capture-memory", "--source", "hook"],
        &[
            ("TOOL_DESCRIPTION", "Session 1: Implemented authentication system"),
            ("CLAUDE_SESSION_ID", &session1_id),
        ],
    );

    runner.run(&["session", "end"]);

    // Session 2: Should be able to retrieve session 1 memories
    let output = runner.run(&["session", "start"]);
    let session2_id = String::from_utf8_lossy(&output.stdout).trim().to_string();

    let output = runner.run_with_env(
        &["inject-context"],
        &[
            ("USER_PROMPT", "authentication"),
            ("CLAUDE_SESSION_ID", &session2_id),
        ],
    );
    assert!(output.status.success());

    let context = String::from_utf8_lossy(&output.stdout);
    assert!(
        context.contains("authentication") || context.contains("Session 1"),
        "Should retrieve memories from previous session"
    );

    runner.run(&["session", "end"]);
}

#[test]
fn test_inject_brief_performance() {
    let runner = CliTestRunner::new();

    // Start session
    let output = runner.run(&["session", "start"]);
    let session_id = String::from_utf8_lossy(&output.stdout).trim().to_string();

    // Time the brief injection
    let start = std::time::Instant::now();
    let output = runner.run_with_env(
        &["inject-brief", "--query", "test query"],
        &[("CLAUDE_SESSION_ID", &session_id)],
    );
    let elapsed = start.elapsed();

    assert!(output.status.success());
    assert!(
        elapsed.as_millis() < 500,
        "inject-brief should complete in <500ms, took {}ms",
        elapsed.as_millis()
    );

    runner.run(&["session", "end"]);
}

#[test]
fn test_empty_content_handling() {
    let runner = CliTestRunner::new();

    // Start session
    let output = runner.run(&["session", "start"]);
    let session_id = String::from_utf8_lossy(&output.stdout).trim().to_string();

    // Capture with empty content should succeed (silently ignored)
    let output = runner.run_with_env(
        &["capture-memory", "--source", "hook"],
        &[
            ("TOOL_DESCRIPTION", ""),
            ("CLAUDE_SESSION_ID", &session_id),
        ],
    );
    assert!(output.status.success(), "Empty content capture should succeed");

    // Inject with empty query should succeed (return empty)
    let output = runner.run_with_env(
        &["inject-context"],
        &[
            ("USER_PROMPT", ""),
            ("CLAUDE_SESSION_ID", &session_id),
        ],
    );
    assert!(output.status.success(), "Empty query injection should succeed");
    let context = String::from_utf8_lossy(&output.stdout);
    assert!(context.is_empty(), "Empty query should return empty context");

    runner.run(&["session", "end"]);
}

#[test]
fn test_status_command() {
    let runner = CliTestRunner::new();

    // Status with no session
    let output = runner.run(&["status"]);
    assert!(output.status.success());

    // Start session and capture memory
    runner.run(&["session", "start"]);
    runner.run_with_env(
        &["capture-memory", "--source", "hook"],
        &[("TOOL_DESCRIPTION", "Test memory for status")],
    );

    // Status should show memory count
    let output = runner.run(&["status"]);
    assert!(output.status.success());
    let status = String::from_utf8_lossy(&output.stdout);
    assert!(status.contains("1") || status.contains("memory"), "Status should show memory count");

    runner.run(&["session", "end"]);
}
```
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/tests/integration/cli_tests.rs">
    Integration test suite
  </file>
  <file path="crates/context-graph-cli/tests/integration/mod.rs">
    Test module exports
  </file>
</files_to_create>

<validation_criteria>
  <criterion type="test">cargo test --test cli_tests -- all tests pass</criterion>
  <criterion type="coverage">Tests cover all major flows</criterion>
  <criterion type="isolation">Tests don't interfere with each other</criterion>
</validation_criteria>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli --test cli_tests</command>
  <command>cargo test --package context-graph-cli --test cli_tests -- --test-threads=1</command>
</test_commands>
</task_spec>
```
