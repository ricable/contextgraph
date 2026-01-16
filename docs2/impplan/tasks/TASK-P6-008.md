# Task: TASK-P6-008 - Hook Shell Scripts

```xml
<task_spec id="TASK-P6-008" version="1.0">
<metadata>
  <title>Hook Shell Scripts</title>
  <phase>6</phase>
  <sequence>50</sequence>
  <layer>surface</layer>
  <estimated_loc>100</estimated_loc>
  <dependencies>
    <dependency task="TASK-P6-007">Setup command (creates scripts)</dependency>
    <dependency task="TASK-P6-002">Session commands (used by scripts)</dependency>
    <dependency task="TASK-P6-003">Inject command (used by scripts)</dependency>
    <dependency task="TASK-P6-005">Capture command (used by scripts)</dependency>
  </dependencies>
  <produces>
    <artifact type="script">session-start.sh</artifact>
    <artifact type="script">user-prompt-submit.sh</artifact>
    <artifact type="script">pre-tool-use.sh</artifact>
    <artifact type="script">post-tool-use.sh</artifact>
    <artifact type="script">stop.sh</artifact>
    <artifact type="script">session-end.sh</artifact>
  </produces>
</metadata>

<context>
  <background>
    The hook shell scripts bridge Claude Code's hook system to the context-graph
    CLI. They read environment variables set by Claude Code, invoke the appropriate
    CLI commands, and output results that Claude Code captures.
  </background>
  <business_value>
    Scripts are the actual integration point with Claude Code. They must be
    reliable, handle errors gracefully, and respect timeout constraints.
  </business_value>
  <technical_context>
    Scripts use set -e for fail-fast behavior. They read from environment
    variables (USER_PROMPT, TOOL_NAME, etc.) and write to stdout for injection
    or silently capture memories. Each script has a timeout in settings.json.
  </technical_context>
</context>

<prerequisites>
  <prerequisite type="code">crates/context-graph-cli/src/commands/setup.rs with script templates</prerequisite>
  <prerequisite type="binary">context-graph-cli must be in PATH</prerequisite>
</prerequisites>

<scope>
  <includes>
    <item>Script template verification and testing</item>
    <item>Environment variable handling validation</item>
    <item>Error handling behavior validation</item>
    <item>Timeout compliance validation</item>
  </includes>
  <excludes>
    <item>Script generation code (TASK-P6-007)</item>
    <item>CLI command implementations</item>
  </excludes>
</scope>

<definition_of_done>
  <criterion id="DOD-1">
    <description>session-start.sh outputs session ID and context</description>
    <verification>Manual test shows expected output</verification>
  </criterion>
  <criterion id="DOD-2">
    <description>user-prompt-submit.sh injects context based on USER_PROMPT</description>
    <verification>USER_PROMPT=test ./hooks/user-prompt-submit.sh works</verification>
  </criterion>
  <criterion id="DOD-3">
    <description>pre-tool-use.sh injects brief context</description>
    <verification>TOOL_NAME=Bash ./hooks/pre-tool-use.sh works</verification>
  </criterion>
  <criterion id="DOD-4">
    <description>post-tool-use.sh captures silently</description>
    <verification>No stdout, memory created</verification>
  </criterion>
  <criterion id="DOD-5">
    <description>All scripts complete within timeout</description>
    <verification>time shows execution under timeout limit</verification>
  </criterion>

  <signatures>
    <!-- Shell scripts don't have Rust signatures -->
    <signature name="session-start.sh">
      <code>
#!/bin/bash
set -e
SESSION_ID=$(context-graph-cli session start)
export CLAUDE_SESSION_ID="$SESSION_ID"
context-graph-cli inject-context --session-id "$SESSION_ID"
      </code>
    </signature>
    <signature name="user-prompt-submit.sh">
      <code>
#!/bin/bash
set -e
SESSION_ID="${CLAUDE_SESSION_ID:-$(cat ~/.contextgraph/current_session 2>/dev/null || echo '')}"
context-graph-cli inject-context --query "${USER_PROMPT:-}" --session-id "$SESSION_ID"
      </code>
    </signature>
  </signatures>

  <constraints>
    <constraint type="behavior">set -e for fail-fast</constraint>
    <constraint type="behavior">Graceful handling of missing env vars</constraint>
    <constraint type="timeout">Each script must complete within its timeout</constraint>
  </constraints>
</definition_of_done>

<pseudo_code>
```bash
# Script Testing Procedures

# Test session-start.sh
echo "=== Testing session-start.sh ==="
./hooks/session-start.sh
# Expected: Outputs session ID line, then context markdown

# Verify session file was created
cat ~/.contextgraph/current_session
# Expected: UUID matching session ID output

# Test user-prompt-submit.sh
echo "=== Testing user-prompt-submit.sh ==="
USER_PROMPT="implement HDBSCAN clustering" ./hooks/user-prompt-submit.sh
# Expected: Context markdown related to clustering

# Test pre-tool-use.sh
echo "=== Testing pre-tool-use.sh ==="
TOOL_NAME="Bash" TOOL_DESCRIPTION="Running tests" ./hooks/pre-tool-use.sh
# Expected: Brief context or empty

# Verify timing
time ./hooks/pre-tool-use.sh
# Expected: real < 0.5s

# Test post-tool-use.sh
echo "=== Testing post-tool-use.sh ==="
TOOL_DESCRIPTION="Implemented HDBSCAN clustering algorithm" \
TOOL_NAME="Write" \
./hooks/post-tool-use.sh
# Expected: No stdout, memory captured

# Test stop.sh
echo "=== Testing stop.sh ==="
RESPONSE_SUMMARY="Created clustering implementation" ./hooks/stop.sh
# Expected: No stdout, response captured

# Test session-end.sh
echo "=== Testing session-end.sh ==="
SESSION_SUMMARY="Completed HDBSCAN implementation" ./hooks/session-end.sh
# Expected: No stdout, session ended

# Verify session file cleared
cat ~/.contextgraph/current_session
# Expected: Empty

# Timeout Compliance Test
echo "=== Timeout Compliance ==="
time ./hooks/session-start.sh    # Should be < 5s
time ./hooks/user-prompt-submit.sh # Should be < 2s
time ./hooks/pre-tool-use.sh     # Should be < 0.5s
time ./hooks/post-tool-use.sh    # Should be < 3s
time ./hooks/stop.sh             # Should be < 3s
time ./hooks/session-end.sh      # Should be < 30s

# Error Handling Test
echo "=== Error Handling ==="
# Test with missing session
rm -f ~/.contextgraph/current_session
./hooks/user-prompt-submit.sh
# Expected: Should not fail, uses empty session

# Test with empty env vars
USER_PROMPT="" ./hooks/user-prompt-submit.sh
# Expected: Should not fail, returns empty
```
</pseudo_code>

<files_to_modify>
  <file path="crates/context-graph-cli/src/commands/setup.rs">
    Verify script templates match specs in TECH-PHASE6
  </file>
</files_to_modify>

<validation_criteria>
  <criterion type="test">All 6 scripts execute without error</criterion>
  <criterion type="timeout">Scripts complete within timeout limits</criterion>
  <criterion type="behavior">Missing env vars handled gracefully</criterion>
</validation_criteria>

<test_commands>
  <command>./hooks/session-start.sh</command>
  <command>USER_PROMPT="test" ./hooks/user-prompt-submit.sh</command>
  <command>TOOL_NAME="Bash" ./hooks/pre-tool-use.sh</command>
  <command>TOOL_DESCRIPTION="test" ./hooks/post-tool-use.sh</command>
  <command>RESPONSE_SUMMARY="test" ./hooks/stop.sh</command>
  <command>./hooks/session-end.sh</command>
  <command>time ./hooks/pre-tool-use.sh</command>
</test_commands>
</task_spec>
```
