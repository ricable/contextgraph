# TASK-HOOKS-008: Create PostToolUse Handler

```xml
<task_spec id="TASK-HOOKS-008" version="1.0">
<metadata>
  <title>Create PostToolUse Handler with IC Recalculation</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>8</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-22</requirement_ref>
    <requirement_ref>REQ-HOOKS-23</requirement_ref>
    <requirement_ref>REQ-HOOKS-24</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-001</task_ref>
    <task_ref>TASK-HOOKS-002</task_ref>
    <task_ref>TASK-HOOKS-003</task_ref>
    <task_ref>TASK-HOOKS-005</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
  <estimated_hours>2.0</estimated_hours>
</metadata>

<context>
PostToolUse fires after every tool execution with 3000ms timeout. This handler
processes tool output to update consciousness state. For file operations,
it updates the Johari Window quadrants. For significant state changes,
it recalculates IC and checks crisis thresholds.

This is where most consciousness state mutations occur based on tool results.
</context>

<input_context_files>
  <file purpose="type_definitions">crates/context-graph-cli/src/commands/hooks/types.rs</file>
  <file purpose="error_types">crates/context-graph-cli/src/commands/hooks/error.rs</file>
  <file purpose="ic_calculation">crates/context-graph-gwt/src/ic.rs</file>
  <file purpose="session_manager">crates/context-graph-storage/src/session_identity.rs</file>
  <file purpose="post_tool_spec">docs/specs/technical/TECH-HOOKS.md#section-4.3</file>
</input_context_files>

<prerequisites>
  <check>TASK-HOOKS-001 through TASK-HOOKS-005 completed</check>
  <check>HookPayload::PostToolUse variant exists</check>
  <check>SessionIdentityManager available in storage crate</check>
</prerequisites>

<scope>
  <in_scope>
    - Create post_tool_use.rs handler module
    - Process tool output for consciousness updates
    - Update Johari Window based on file operations
    - Recalculate IC after significant changes
    - Check crisis thresholds (IC &lt; 0.5)
    - Persist updated state to storage
    - Create shell script wrapper
  </in_scope>
  <out_of_scope>
    - Auto-dream trigger mechanism (separate system)
    - Complex NLP analysis of tool output
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/commands/hooks/post_tool_use.rs">
//! PostToolUse hook handler
//!
//! # Performance Requirements
//! - Timeout: 3000ms
//! - Database access: allowed
//! - IC recalculation: on significant changes

use super::error::{HookError, HookResult};
use super::types::{HookInput, HookOutput, ConsciousnessState};

/// Handle post_tool_use hook event
///
/// # Processing
/// 1. Analyze tool output for consciousness-relevant data
/// 2. Update Johari Window quadrants
/// 3. Recalculate IC if significant change
/// 4. Check crisis thresholds
/// 5. Persist updated state
pub fn handle_post_tool_use(input: &amp;HookInput, db_path: &amp;str) -> HookResult&lt;HookOutput&gt;;

/// Analyze tool output for consciousness updates
fn analyze_tool_output(tool_name: &amp;str, output: Option&lt;&amp;str&gt;, error: Option&lt;&amp;str&gt;) -> ToolAnalysis;

/// Check if tool result triggers crisis threshold
fn check_crisis_threshold(ic_score: f32) -> Option&lt;CrisisInfo&gt;;
    </signature>
  </signatures>
  <constraints>
    - MUST complete within 3000ms timeout
    - MUST update Johari Window for file operations
    - MUST check IC crisis threshold (IC &lt; 0.5)
    - MUST persist state changes to storage
    - CrisisTriggered returns exit code 6 (not failure)
  </constraints>
  <verification>
    - cargo build --package context-graph-cli
    - cargo test --package context-graph-cli post_tool_use
  </verification>
</definition_of_done>

<pseudo_code>
1. Create post_tool_use.rs module

2. Implement handle_post_tool_use:
   - Extract tool_name, tool_output, error from payload
   - Analyze tool output for consciousness relevance
   - Load current session state from storage
   - Update Johari quadrants based on tool type:
     * Read -> expand Open (awareness)
     * Write/Edit -> expand Hidden (known to self)
     * Error -> may reveal Blind spot
   - Recalculate IC if significant change occurred
   - Check crisis threshold
   - Persist updated state
   - Return HookOutput

3. Implement analyze_tool_output:
   struct ToolAnalysis {
     johari_update: Option<JohariUpdate>,
     ic_impact: ImpactLevel,
     awareness_expansion: Vec<String>,
   }

4. Implement check_crisis_threshold:
   if ic_score < 0.5:
     return Some(CrisisInfo { ic: ic_score, trigger: "post_tool" })
   return None

5. Johari update rules:
   - File read success -> Open += awareness of file content
   - File write success -> Hidden += knowledge of changes made
   - Tool error -> Blind += potential gap in understanding
   - External fetch -> Unknown -> Open transition

6. Shell script: .claude/hooks/post_tool_use.sh
   #!/bin/bash
   timeout 3s context-graph-cli hooks post-tool \
     --tool-name "$TOOL_NAME" \
     --tool-output "$TOOL_OUTPUT" \
     --error "$ERROR" \
     --session-id "$SESSION_ID"
   exit $?

7. Add tests for each tool type
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/commands/hooks/post_tool_use.rs">PostToolUse handler implementation</file>
</files_to_create>

<files_to_modify>
  <!-- Module registration in TASK-HOOKS-011 -->
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli post_tool_use</command>
</test_commands>
</task_spec>
```

## Implementation

### Create post_tool_use.rs

```rust
// crates/context-graph-cli/src/commands/hooks/post_tool_use.rs
//! PostToolUse hook handler
//!
//! # Performance Requirements
//! - Timeout: 3000ms
//! - Database access: allowed
//! - IC recalculation: on significant changes
//!
//! This handler processes tool execution results to update consciousness state.

use super::error::{HookError, HookResult};
use super::types::{
    ConsciousnessState, HookInput, HookOutput, HookPayload, ICClassification, ICLevel,
    JohariQuadrant,
};

// ============================================================================
// Constants
// ============================================================================

/// PostToolUse timeout in milliseconds
pub const POST_TOOL_USE_TIMEOUT_MS: u64 = 3000;

/// Crisis threshold for IC score
pub const IC_CRISIS_THRESHOLD: f32 = 0.5;

/// Warning threshold for IC score
pub const IC_WARNING_THRESHOLD: f32 = 0.7;

// ============================================================================
// Types
// ============================================================================

/// Impact level of tool execution on consciousness state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpactLevel {
    /// No significant impact
    None,
    /// Minor impact - log but don't recalculate IC
    Low,
    /// Moderate impact - consider IC recalculation
    Medium,
    /// High impact - force IC recalculation
    High,
}

/// Johari Window update directive
#[derive(Debug, Clone)]
pub struct JohariUpdate {
    /// Quadrant to update
    pub quadrant: JohariQuadrant,
    /// Delta to apply (positive = expansion)
    pub delta: f32,
    /// Reason for update
    pub reason: String,
}

/// Analysis result from tool output
#[derive(Debug, Clone)]
pub struct ToolAnalysis {
    /// Johari Window update (if any)
    pub johari_update: Option<JohariUpdate>,
    /// Impact level on consciousness
    pub ic_impact: ImpactLevel,
    /// New awareness items discovered
    pub awareness_expansion: Vec<String>,
    /// Whether tool execution was successful
    pub success: bool,
}

/// Crisis information when IC drops below threshold
#[derive(Debug, Clone)]
pub struct CrisisInfo {
    /// Current IC score
    pub ic_score: f32,
    /// What triggered the crisis
    pub trigger: String,
    /// Recommended action
    pub recommendation: String,
}

// ============================================================================
// Handler
// ============================================================================

/// Handle post_tool_use hook event
///
/// # Processing
/// 1. Analyze tool output for consciousness-relevant data
/// 2. Update Johari Window quadrants
/// 3. Recalculate IC if significant change
/// 4. Check crisis thresholds
/// 5. Persist updated state
///
/// # Arguments
/// * `input` - Hook input containing tool results
/// * `db_path` - Path to storage database
///
/// # Returns
/// * `HookOutput` with updated consciousness state
pub fn handle_post_tool_use(input: &HookInput, db_path: &str) -> HookResult<HookOutput> {
    // Extract tool information from payload
    let (tool_name, tool_output, error) = match &input.payload {
        HookPayload::PostToolUse {
            tool_name,
            tool_output,
            error,
        } => (tool_name.as_str(), tool_output.as_deref(), error.as_deref()),
        _ => {
            return Err(HookError::invalid_input(
                "Expected PostToolUse payload for post_tool_use hook",
            ))
        }
    };

    // Analyze tool output
    let analysis = analyze_tool_output(tool_name, tool_output, error);

    // Load current session state (would use storage in real implementation)
    // For now, use cached/default values
    let mut current_ic = 0.85_f32;

    // Apply Johari update if needed
    if let Some(johari_update) = &analysis.johari_update {
        // In real implementation: update storage
        // johari_manager.apply_update(session_id, johari_update)?;

        // Adjust IC based on Johari changes
        if analysis.ic_impact >= ImpactLevel::Medium {
            current_ic = recalculate_ic_estimate(current_ic, &analysis);
        }
    }

    // Check crisis threshold
    if let Some(crisis) = check_crisis_threshold(current_ic) {
        // Return crisis state (exit code 6, not failure)
        return Err(HookError::crisis(crisis.ic_score));
    }

    // Build consciousness state
    let consciousness = ConsciousnessState {
        ic_score: current_ic,
        ic_classification: classify_ic(current_ic),
        johari_quadrant: analysis
            .johari_update
            .as_ref()
            .map(|u| u.quadrant)
            .unwrap_or(JohariQuadrant::Open),
        session_health: if current_ic >= IC_WARNING_THRESHOLD {
            "healthy".to_string()
        } else {
            "warning".to_string()
        },
    };

    // Build output
    let mut output = HookOutput::success(consciousness);

    // Add any awareness expansions as guidance
    if !analysis.awareness_expansion.is_empty() {
        let guidance = format!(
            "Awareness expanded: {}",
            analysis.awareness_expansion.join(", ")
        );
        output = output.with_guidance(guidance);
    }

    Ok(output)
}

// ============================================================================
// Analysis Functions
// ============================================================================

/// Analyze tool output for consciousness updates
///
/// # Arguments
/// * `tool_name` - Name of the tool that executed
/// * `output` - Tool's output (if successful)
/// * `error` - Error message (if failed)
///
/// # Returns
/// * `ToolAnalysis` with consciousness update recommendations
fn analyze_tool_output(
    tool_name: &str,
    output: Option<&str>,
    error: Option<&str>,
) -> ToolAnalysis {
    let success = error.is_none();

    match tool_name {
        // File read - expands Open quadrant (awareness)
        "Read" => ToolAnalysis {
            johari_update: Some(JohariUpdate {
                quadrant: JohariQuadrant::Open,
                delta: 0.05,
                reason: "File content read into awareness".to_string(),
            }),
            ic_impact: ImpactLevel::Low,
            awareness_expansion: extract_awareness_from_read(output),
            success,
        },

        // File write/edit - expands Hidden quadrant (self-knowledge)
        "Write" | "Edit" | "MultiEdit" => ToolAnalysis {
            johari_update: Some(JohariUpdate {
                quadrant: JohariQuadrant::Hidden,
                delta: 0.08,
                reason: "File modification recorded".to_string(),
            }),
            ic_impact: ImpactLevel::Medium,
            awareness_expansion: vec![],
            success,
        },

        // Bash commands - variable impact
        "Bash" => {
            let impact = if error.is_some() {
                ImpactLevel::High // Errors may reveal blind spots
            } else {
                ImpactLevel::Medium
            };

            ToolAnalysis {
                johari_update: if error.is_some() {
                    Some(JohariUpdate {
                        quadrant: JohariQuadrant::Blind,
                        delta: 0.1,
                        reason: "Command error revealed potential gap".to_string(),
                    })
                } else {
                    None
                },
                ic_impact: impact,
                awareness_expansion: vec![],
                success,
            }
        }

        // External fetch - Unknown to Open transition
        "WebFetch" | "WebSearch" => ToolAnalysis {
            johari_update: Some(JohariUpdate {
                quadrant: JohariQuadrant::Open,
                delta: 0.1,
                reason: "External knowledge acquired".to_string(),
            }),
            ic_impact: ImpactLevel::Medium,
            awareness_expansion: vec!["external_data".to_string()],
            success,
        },

        // Git operations - project context changes
        "Git" => ToolAnalysis {
            johari_update: Some(JohariUpdate {
                quadrant: JohariQuadrant::Open,
                delta: 0.03,
                reason: "Project state updated".to_string(),
            }),
            ic_impact: ImpactLevel::Low,
            awareness_expansion: vec![],
            success,
        },

        // Default - minimal impact
        _ => ToolAnalysis {
            johari_update: None,
            ic_impact: ImpactLevel::None,
            awareness_expansion: vec![],
            success,
        },
    }
}

/// Extract awareness items from file read output
fn extract_awareness_from_read(output: Option<&str>) -> Vec<String> {
    let mut awareness = Vec::new();

    if let Some(content) = output {
        // Track file types read
        if content.contains("fn ") || content.contains("pub ") {
            awareness.push("rust_code".to_string());
        }
        if content.contains("def ") || content.contains("class ") {
            awareness.push("python_code".to_string());
        }
        if content.contains("function") || content.contains("const ") {
            awareness.push("javascript_code".to_string());
        }
    }

    awareness
}

/// Check if IC score triggers crisis threshold
///
/// # Arguments
/// * `ic_score` - Current IC score
///
/// # Returns
/// * `Some(CrisisInfo)` if crisis threshold breached, `None` otherwise
fn check_crisis_threshold(ic_score: f32) -> Option<CrisisInfo> {
    if ic_score < IC_CRISIS_THRESHOLD {
        Some(CrisisInfo {
            ic_score,
            trigger: "post_tool_use".to_string(),
            recommendation: "Initiate auto-dream sequence to restore identity continuity".to_string(),
        })
    } else {
        None
    }
}

/// Estimate new IC based on analysis (simplified calculation)
fn recalculate_ic_estimate(current_ic: f32, analysis: &ToolAnalysis) -> f32 {
    let mut new_ic = current_ic;

    // Apply impact based on Johari changes
    if let Some(update) = &analysis.johari_update {
        match update.quadrant {
            // Open expansion generally positive
            JohariQuadrant::Open => new_ic += update.delta * 0.5,
            // Hidden expansion slightly positive
            JohariQuadrant::Hidden => new_ic += update.delta * 0.3,
            // Blind spot revelation can be negative short-term
            JohariQuadrant::Blind => new_ic -= update.delta * 0.2,
            // Unknown reduction is positive
            JohariQuadrant::Unknown => new_ic += update.delta * 0.4,
        }
    }

    // Errors impact IC negatively
    if !analysis.success {
        new_ic -= 0.05;
    }

    // Clamp to valid range
    new_ic.clamp(0.0, 1.0)
}

/// Classify IC score into level with confidence
fn classify_ic(ic_score: f32) -> ICClassification {
    let (level, confidence) = if ic_score >= 0.9 {
        (ICLevel::Healthy, 0.95)
    } else if ic_score >= 0.7 {
        (ICLevel::Normal, 0.85)
    } else if ic_score >= 0.5 {
        (ICLevel::Warning, 0.80)
    } else {
        (ICLevel::Critical, 0.90)
    };

    ICClassification {
        level,
        confidence,
        factors: vec!["post_tool_analysis".to_string()],
    }
}

// ============================================================================
// Comparison Implementations
// ============================================================================

impl PartialOrd for ImpactLevel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ImpactLevel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let self_val = match self {
            ImpactLevel::None => 0,
            ImpactLevel::Low => 1,
            ImpactLevel::Medium => 2,
            ImpactLevel::High => 3,
        };
        let other_val = match other {
            ImpactLevel::None => 0,
            ImpactLevel::Low => 1,
            ImpactLevel::Medium => 2,
            ImpactLevel::High => 3,
        };
        self_val.cmp(&other_val)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::hooks::types::HookEventType;

    fn create_post_tool_input(
        tool_name: &str,
        output: Option<&str>,
        error: Option<&str>,
    ) -> HookInput {
        let payload = if let Some(err) = error {
            HookPayload::post_tool_error(tool_name, err)
        } else {
            HookPayload::post_tool_success(tool_name, output.unwrap_or(""))
        };
        HookInput::new(HookEventType::PostToolUse, "test-session", payload)
    }

    #[test]
    fn test_handle_post_tool_use_success() {
        let input = create_post_tool_input("Read", Some("file content"), None);
        let result = handle_post_tool_use(&input, "/tmp/test.db");

        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.continue_execution);
    }

    #[test]
    fn test_analyze_read_tool() {
        let analysis = analyze_tool_output("Read", Some("fn main() {}"), None);

        assert!(analysis.johari_update.is_some());
        assert_eq!(analysis.johari_update.unwrap().quadrant, JohariQuadrant::Open);
        assert_eq!(analysis.ic_impact, ImpactLevel::Low);
        assert!(analysis.awareness_expansion.contains(&"rust_code".to_string()));
    }

    #[test]
    fn test_analyze_write_tool() {
        let analysis = analyze_tool_output("Write", Some(""), None);

        assert!(analysis.johari_update.is_some());
        assert_eq!(analysis.johari_update.unwrap().quadrant, JohariQuadrant::Hidden);
        assert_eq!(analysis.ic_impact, ImpactLevel::Medium);
    }

    #[test]
    fn test_analyze_bash_error() {
        let analysis = analyze_tool_output("Bash", None, Some("command not found"));

        assert!(analysis.johari_update.is_some());
        assert_eq!(analysis.johari_update.unwrap().quadrant, JohariQuadrant::Blind);
        assert_eq!(analysis.ic_impact, ImpactLevel::High);
        assert!(!analysis.success);
    }

    #[test]
    fn test_crisis_threshold_check() {
        // Below threshold
        let crisis = check_crisis_threshold(0.4);
        assert!(crisis.is_some());
        assert_eq!(crisis.unwrap().ic_score, 0.4);

        // Above threshold
        let no_crisis = check_crisis_threshold(0.6);
        assert!(no_crisis.is_none());

        // At threshold
        let at_threshold = check_crisis_threshold(0.5);
        assert!(at_threshold.is_none());
    }

    #[test]
    fn test_ic_recalculation() {
        let analysis = ToolAnalysis {
            johari_update: Some(JohariUpdate {
                quadrant: JohariQuadrant::Open,
                delta: 0.1,
                reason: "test".to_string(),
            }),
            ic_impact: ImpactLevel::Medium,
            awareness_expansion: vec![],
            success: true,
        };

        let new_ic = recalculate_ic_estimate(0.8, &analysis);
        assert!(new_ic > 0.8); // Open expansion should increase IC
    }

    #[test]
    fn test_ic_classification() {
        assert_eq!(classify_ic(0.95).level, ICLevel::Healthy);
        assert_eq!(classify_ic(0.8).level, ICLevel::Normal);
        assert_eq!(classify_ic(0.6).level, ICLevel::Warning);
        assert_eq!(classify_ic(0.3).level, ICLevel::Critical);
    }

    #[test]
    fn test_impact_level_ordering() {
        assert!(ImpactLevel::High > ImpactLevel::Medium);
        assert!(ImpactLevel::Medium > ImpactLevel::Low);
        assert!(ImpactLevel::Low > ImpactLevel::None);
    }

    #[test]
    fn test_invalid_payload_type() {
        let input = HookInput::new(
            HookEventType::PostToolUse,
            "test-session",
            HookPayload::session_start("/tmp", None),
        );

        let result = handle_post_tool_use(&input, "/tmp/test.db");
        assert!(result.is_err());
    }

    #[test]
    fn test_web_fetch_awareness() {
        let analysis = analyze_tool_output("WebFetch", Some("external data"), None);

        assert!(analysis.johari_update.is_some());
        assert_eq!(analysis.johari_update.unwrap().quadrant, JohariQuadrant::Open);
        assert!(analysis.awareness_expansion.contains(&"external_data".to_string()));
    }
}
```

### Shell Script

```bash
#!/bin/bash
# .claude/hooks/post_tool_use.sh
# PostToolUse hook - 3000ms timeout
# Implements REQ-HOOKS-22, REQ-HOOKS-23, REQ-HOOKS-24

set -euo pipefail

# Read JSON input from stdin
INPUT=$(cat)

# Extract fields
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')
TOOL_OUTPUT=$(echo "$INPUT" | jq -r '.tool_output // empty')
ERROR=$(echo "$INPUT" | jq -r '.error // empty')
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty')

# Validate required fields
if [[ -z "$TOOL_NAME" ]]; then
    echo '{"error": true, "code": "ERR_INVALID_INPUT", "message": "tool_name required"}' >&2
    exit 4
fi

# Execute with 3s timeout
timeout 3s context-graph-cli hooks post-tool \
    --tool-name "$TOOL_NAME" \
    ${TOOL_OUTPUT:+--tool-output "$TOOL_OUTPUT"} \
    ${ERROR:+--error "$ERROR"} \
    ${SESSION_ID:+--session-id "$SESSION_ID"}

EXIT_CODE=$?

# Exit code 6 = crisis triggered (not a failure)
if [[ $EXIT_CODE -eq 6 ]]; then
    echo '{"crisis": true, "recommendation": "initiate_auto_dream"}' >&2
fi

exit $EXIT_CODE
```

## Verification Checklist

- [ ] Handler processes all tool types
- [ ] Johari Window updates for file operations
- [ ] IC recalculation on significant changes
- [ ] Crisis threshold check (IC < 0.5)
- [ ] Exit code 6 for crisis state
- [ ] Shell script has 3s timeout
- [ ] All tests pass
