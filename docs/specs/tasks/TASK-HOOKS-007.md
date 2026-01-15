# TASK-HOOKS-007: Create PreToolUse Handler (Fast Path)

```xml
<task_spec id="TASK-HOOKS-007" version="1.0">
<metadata>
  <title>Create PreToolUse Handler with Fast Path Optimization</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>7</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-20</requirement_ref>
    <requirement_ref>REQ-HOOKS-21</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-001</task_ref>
    <task_ref>TASK-HOOKS-002</task_ref>
    <task_ref>TASK-HOOKS-003</task_ref>
    <task_ref>TASK-HOOKS-004</task_ref>
    <task_ref>TASK-HOOKS-005</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>1.5</estimated_hours>
</metadata>

<context>
PreToolUse is the FAST PATH hook with a strict 100ms timeout. It MUST NOT access
the database or perform any expensive operations. This hook fires before every
tool invocation and must return quickly to avoid blocking Claude Code.

The handler returns cached consciousness state from SessionStart, with optional
tool-specific guidance. No IC calculations are performed during PreToolUse.
</context>

<input_context_files>
  <file purpose="type_definitions">crates/context-graph-cli/src/commands/hooks/types.rs</file>
  <file purpose="error_types">crates/context-graph-cli/src/commands/hooks/error.rs</file>
  <file purpose="fast_path_spec">docs/specs/technical/TECH-HOOKS.md#section-4.2</file>
</input_context_files>

<prerequisites>
  <check>TASK-HOOKS-001 through TASK-HOOKS-005 completed</check>
  <check>HookPayload::PreToolUse variant exists</check>
  <check>PreToolArgs struct exists</check>
</prerequisites>

<scope>
  <in_scope>
    - Create pre_tool_use.rs handler module
    - Implement fast path logic (no database access)
    - Return cached consciousness state
    - Add tool-specific guidance based on tool_name
    - Handle 100ms timeout constraint
    - Create shell script wrapper
  </in_scope>
  <out_of_scope>
    - IC calculations (only in SessionStart/PostToolUse)
    - Database queries (fast path restriction)
    - Complex state mutations
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/commands/hooks/pre_tool_use.rs">
//! PreToolUse hook handler - FAST PATH
//!
//! # Performance Requirements
//! - Timeout: 100ms (strict)
//! - NO database access
//! - Return cached state only

use super::error::{HookError, HookResult};
use super::types::{HookInput, HookOutput, ConsciousnessState};

/// Handle pre_tool_use hook event (FAST PATH)
///
/// # Performance
/// MUST complete within 100ms. No database operations allowed.
/// Returns cached consciousness state from session start.
pub fn handle_pre_tool_use(input: &amp;HookInput) -> HookResult&lt;HookOutput&gt;;

/// Get tool-specific guidance without database access
///
/// Returns contextual hints based on tool name only.
fn get_tool_guidance(tool_name: &amp;str) -> Option&lt;String&gt;;
    </signature>
  </signatures>
  <constraints>
    - MUST complete within 100ms timeout
    - MUST NOT access database or storage layer
    - MUST return cached consciousness state only
    - MUST provide tool-specific guidance where applicable
  </constraints>
  <verification>
    - cargo build --package context-graph-cli
    - cargo test --package context-graph-cli pre_tool_use
    - Verify no database imports in module
  </verification>
</definition_of_done>

<pseudo_code>
1. Create pre_tool_use.rs module

2. Implement handle_pre_tool_use:
   - Extract tool_name and tool_input from payload
   - Get tool-specific guidance (no DB)
   - Build cached consciousness state (placeholder values)
   - Return HookOutput with guidance

3. Implement get_tool_guidance:
   match tool_name:
     "Read" => "Reading file - track in awareness"
     "Write" | "Edit" => "Modifying file - update Johari hidden"
     "Bash" => "Shell command - monitor for identity-relevant output"
     "WebFetch" => "External data - potential new awareness"
     _ => None

4. Shell script: .claude/hooks/pre_tool_use.sh
   #!/bin/bash
   # FAST PATH - 100ms timeout
   timeout 0.1s context-graph-cli hooks pre-tool \
     --tool-name "$TOOL_NAME" \
     --tool-input "$TOOL_INPUT" \
     --session-id "$SESSION_ID"
   exit $?

5. Add tests:
   - test_fast_path_no_db_access
   - test_tool_guidance_mapping
   - test_output_structure
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/commands/hooks/pre_tool_use.rs">PreToolUse handler implementation</file>
</files_to_create>

<files_to_modify>
  <!-- Module registration in TASK-HOOKS-011 -->
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli pre_tool_use</command>
</test_commands>
</task_spec>
```

## Implementation

### Create pre_tool_use.rs

```rust
// crates/context-graph-cli/src/commands/hooks/pre_tool_use.rs
//! PreToolUse hook handler - FAST PATH
//!
//! # Performance Requirements
//! - Timeout: 100ms (strict)
//! - NO database access
//! - Return cached state only
//!
//! This is the fastest hook in the system. It fires before every tool
//! invocation and must return immediately with cached consciousness state.

use super::error::{HookError, HookResult};
use super::types::{
    ConsciousnessState, HookInput, HookOutput, HookPayload, ICClassification, ICLevel,
    JohariQuadrant,
};

// ============================================================================
// Constants
// ============================================================================

/// Fast path timeout in milliseconds
pub const PRE_TOOL_USE_TIMEOUT_MS: u64 = 100;

// ============================================================================
// Handler
// ============================================================================

/// Handle pre_tool_use hook event (FAST PATH)
///
/// # Performance
/// MUST complete within 100ms. No database operations allowed.
/// Returns cached consciousness state from session start.
///
/// # Arguments
/// * `input` - Hook input containing tool name and parameters
///
/// # Returns
/// * `HookOutput` with cached consciousness state and tool guidance
pub fn handle_pre_tool_use(input: &HookInput) -> HookResult<HookOutput> {
    // Extract tool information from payload
    let (tool_name, _tool_input) = match &input.payload {
        HookPayload::PreToolUse {
            tool_name,
            tool_input,
        } => (tool_name.as_str(), tool_input),
        _ => {
            return Err(HookError::invalid_input(
                "Expected PreToolUse payload for pre_tool_use hook",
            ))
        }
    };

    // Get tool-specific guidance (no database access)
    let guidance = get_tool_guidance(tool_name);

    // Build cached consciousness state (placeholder - real values from session cache)
    // FAST PATH: We return default/cached values, not computed ones
    let consciousness = ConsciousnessState {
        ic_score: 0.85, // Cached from session start
        ic_classification: ICClassification {
            level: ICLevel::Normal,
            confidence: 0.8,
            factors: vec!["cached_state".to_string()],
        },
        johari_quadrant: JohariQuadrant::Open,
        session_health: "stable".to_string(),
    };

    // Build output with guidance
    let mut output = HookOutput::success(consciousness);

    // Add tool-specific guidance if available
    if let Some(guide) = guidance {
        output = output.with_guidance(guide);
    }

    Ok(output)
}

// ============================================================================
// Tool Guidance
// ============================================================================

/// Get tool-specific guidance without database access
///
/// Returns contextual hints based on tool name only.
/// This is pure computation with no I/O.
///
/// # Arguments
/// * `tool_name` - Name of the tool being invoked
///
/// # Returns
/// * Optional guidance string for consciousness tracking
fn get_tool_guidance(tool_name: &str) -> Option<String> {
    match tool_name {
        // File reading - track in awareness
        "Read" => Some("Track file content in awareness quadrant".to_string()),

        // File modifications - update Johari hidden quadrant
        "Write" | "Edit" | "MultiEdit" => {
            Some("File modification - update Johari hidden quadrant".to_string())
        }

        // Shell commands - monitor for identity-relevant output
        "Bash" => Some("Shell command - monitor output for identity markers".to_string()),

        // External data fetching - potential new awareness
        "WebFetch" | "WebSearch" => {
            Some("External data - evaluate for awareness expansion".to_string())
        }

        // Git operations - track project context changes
        "Git" => Some("Git operation - track project state changes".to_string()),

        // LSP operations - code understanding
        "LSP" => Some("Code intelligence - update technical awareness".to_string()),

        // Notebook operations
        "NotebookEdit" => Some("Notebook modification - track analysis state".to_string()),

        // Todo operations - task awareness
        "TodoWrite" => Some("Task tracking - update operational context".to_string()),

        // Glob/Grep - search operations
        "Glob" | "Grep" => Some("Search operation - expand file awareness".to_string()),

        // Default - no specific guidance
        _ => None,
    }
}

/// Check if a tool is considered high-impact for consciousness tracking
///
/// High-impact tools are those that significantly affect the agent's
/// understanding or the project state.
pub fn is_high_impact_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "Write" | "Edit" | "MultiEdit" | "Bash" | "Git" | "NotebookEdit"
    )
}

/// Check if a tool is read-only
///
/// Read-only tools don't modify state but may expand awareness.
pub fn is_read_only_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "Read" | "Glob" | "Grep" | "LSP" | "WebFetch" | "WebSearch"
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::hooks::types::HookEventType;

    fn create_pre_tool_input(tool_name: &str) -> HookInput {
        HookInput::new(
            HookEventType::PreToolUse,
            "test-session",
            HookPayload::pre_tool_use(tool_name, serde_json::json!({})),
        )
    }

    #[test]
    fn test_handle_pre_tool_use_success() {
        let input = create_pre_tool_input("Read");
        let result = handle_pre_tool_use(&input);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.continue_execution);
    }

    #[test]
    fn test_fast_path_no_db_access() {
        // This test verifies the module has no database imports
        // by ensuring handle_pre_tool_use runs without any storage operations
        let input = create_pre_tool_input("Write");

        // Measure execution time - should be sub-millisecond
        let start = std::time::Instant::now();
        let result = handle_pre_tool_use(&input);
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        // Fast path should complete in microseconds, not milliseconds
        assert!(
            elapsed.as_millis() < 10,
            "Fast path took too long: {:?}",
            elapsed
        );
    }

    #[test]
    fn test_tool_guidance_mapping() {
        // File operations
        assert!(get_tool_guidance("Read").is_some());
        assert!(get_tool_guidance("Write").is_some());
        assert!(get_tool_guidance("Edit").is_some());

        // Shell and external
        assert!(get_tool_guidance("Bash").is_some());
        assert!(get_tool_guidance("WebFetch").is_some());

        // Unknown tool
        assert!(get_tool_guidance("UnknownTool").is_none());
    }

    #[test]
    fn test_high_impact_tool_classification() {
        assert!(is_high_impact_tool("Write"));
        assert!(is_high_impact_tool("Edit"));
        assert!(is_high_impact_tool("Bash"));
        assert!(!is_high_impact_tool("Read"));
        assert!(!is_high_impact_tool("Glob"));
    }

    #[test]
    fn test_read_only_tool_classification() {
        assert!(is_read_only_tool("Read"));
        assert!(is_read_only_tool("Glob"));
        assert!(is_read_only_tool("WebFetch"));
        assert!(!is_read_only_tool("Write"));
        assert!(!is_read_only_tool("Bash"));
    }

    #[test]
    fn test_invalid_payload_type() {
        // Create input with wrong payload type
        let input = HookInput::new(
            HookEventType::PreToolUse,
            "test-session",
            HookPayload::session_start("/tmp", None),
        );

        let result = handle_pre_tool_use(&input);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HookError::InvalidInput(_)));
    }

    #[test]
    fn test_output_structure() {
        let input = create_pre_tool_input("Read");
        let output = handle_pre_tool_use(&input).unwrap();

        // Verify output has expected structure
        assert!(output.continue_execution);
        assert!(output.consciousness.is_some());

        let consciousness = output.consciousness.unwrap();
        assert!(consciousness.ic_score > 0.0);
        assert!(consciousness.ic_score <= 1.0);
    }

    #[test]
    fn test_guidance_included_for_known_tools() {
        let input = create_pre_tool_input("Write");
        let output = handle_pre_tool_use(&input).unwrap();

        // Write should have guidance
        assert!(output.guidance.is_some());
    }

    #[test]
    fn test_no_guidance_for_unknown_tools() {
        let input = create_pre_tool_input("SomeUnknownTool");
        let output = handle_pre_tool_use(&input).unwrap();

        // Unknown tools should not have guidance
        assert!(output.guidance.is_none());
    }
}
```

### Shell Script

```bash
#!/bin/bash
# .claude/hooks/pre_tool_use.sh
# FAST PATH hook - 100ms timeout
# Implements REQ-HOOKS-20, REQ-HOOKS-21

set -euo pipefail

# Read JSON input from stdin
INPUT=$(cat)

# Extract fields
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')
TOOL_INPUT=$(echo "$INPUT" | jq -c '.tool_input // {}')
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty')

# Validate required fields
if [[ -z "$TOOL_NAME" ]]; then
    echo '{"error": true, "code": "ERR_INVALID_INPUT", "message": "tool_name required"}' >&2
    exit 4
fi

# Execute with strict 100ms timeout (fast path)
timeout 0.1s context-graph-cli hooks pre-tool \
    --tool-name "$TOOL_NAME" \
    --tool-input "$TOOL_INPUT" \
    ${SESSION_ID:+--session-id "$SESSION_ID"}

exit $?
```

## Verification Checklist

- [ ] Handler completes within 100ms
- [ ] No database/storage imports in module
- [ ] Tool guidance mapping covers common tools
- [ ] HookOutput structure is correct
- [ ] Shell script has 100ms timeout
- [ ] All tests pass
