# TASK-HOOKS-003: Create HookPayload Variants

```xml
<task_spec id="TASK-HOOKS-003" version="1.0">
<metadata>
  <title>Create HookPayload Enum with Event-Specific Variants</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>3</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-10</requirement_ref>
    <requirement_ref>REQ-HOOKS-11</requirement_ref>
    <requirement_ref>REQ-HOOKS-12</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-001</task_ref>
    <task_ref>TASK-HOOKS-002</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>1.0</estimated_hours>
</metadata>

<context>
This task creates the HookPayload enum with typed variants for each hook event type.
Each variant contains the specific data Claude Code passes to that hook type.
This replaces the temporary serde_json::Value payload in HookInput.

Payload types:
- SessionStart: cwd, previous_session_id
- PreToolUse: tool_name, tool_input
- PostToolUse: tool_name, tool_output, error
- UserPromptSubmit: prompt, context
- SessionEnd: duration_ms, status
</context>

<input_context_files>
  <file purpose="type_definitions">crates/context-graph-cli/src/commands/hooks/types.rs</file>
  <file purpose="technical_spec">docs/specs/technical/TECH-HOOKS.md#section-2.2</file>
</input_context_files>

<prerequisites>
  <check>TASK-HOOKS-001 completed (HookEventType exists)</check>
  <check>TASK-HOOKS-002 completed (HookInput/HookOutput exist)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create HookPayload enum with 5 typed variants
    - Create ConversationMessage struct
    - Create SessionEndStatus enum
    - Update HookInput to use typed HookPayload
    - Add unit tests
  </in_scope>
  <out_of_scope>
    - CLI argument types (TASK-HOOKS-004)
    - Error handling types (TASK-HOOKS-005)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/commands/hooks/types.rs">
/// Event-specific payload data
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum HookPayload {
    /// SessionStart payload
    SessionStart {
        cwd: String,
        previous_session_id: Option&lt;String&gt;,
    },
    /// PreToolUse payload
    PreToolUse {
        tool_name: String,
        tool_input: serde_json::Value,
    },
    /// PostToolUse payload
    PostToolUse {
        tool_name: String,
        tool_output: Option&lt;String&gt;,
        error: Option&lt;String&gt;,
    },
    /// UserPromptSubmit payload
    UserPromptSubmit {
        prompt: String,
        context: Option&lt;Vec&lt;ConversationMessage&gt;&gt;,
    },
    /// SessionEnd payload
    SessionEnd {
        duration_ms: u64,
        status: SessionEndStatus,
    },
}

/// Conversation message for context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
}

/// Session end status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionEndStatus {
    Normal,
    Timeout,
    Error,
    UserAbort,
}
    </signature>
  </signatures>
  <constraints>
    - HookPayload MUST use tagged enum serialization (#[serde(tag = "type", content = "data")])
    - tool_input MUST remain serde_json::Value (arbitrary tool parameters)
    - All string fields MUST be owned (String, not &str)
    - SessionEndStatus MUST use snake_case serialization
  </constraints>
  <verification>
    - cargo test --package context-graph-cli hook_payload
    - Verify JSON serialization matches Claude Code format
  </verification>
</definition_of_done>

<pseudo_code>
1. Create SessionEndStatus enum:
   - Normal (clean exit)
   - Timeout (session timed out)
   - Error (error occurred)
   - UserAbort (user interrupted)

2. Create ConversationMessage struct:
   - role: String (user/assistant/system)
   - content: String

3. Create HookPayload enum with serde tag="type", content="data":
   - SessionStart { cwd, previous_session_id }
   - PreToolUse { tool_name, tool_input }
   - PostToolUse { tool_name, tool_output, error }
   - UserPromptSubmit { prompt, context }
   - SessionEnd { duration_ms, status }

4. Update HookInput struct:
   - Change payload from serde_json::Value to HookPayload

5. Add helper methods to HookPayload:
   - is_session_start() -> bool
   - is_pre_tool_use() -> bool
   - etc.

6. Add tests for JSON serialization format
</pseudo_code>

<files_to_create>
  <!-- No new files -->
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/commands/hooks/types.rs">Add HookPayload, ConversationMessage, SessionEndStatus; update HookInput</file>
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli hook_payload</command>
</test_commands>
</task_spec>
```

## Implementation

### Add to types.rs (before HookInput)

```rust
// ============================================================================
// Session End Status
// ============================================================================

/// Session end status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionEndStatus {
    /// Clean session termination
    Normal,
    /// Session timed out
    Timeout,
    /// Error occurred during session
    Error,
    /// User interrupted/aborted session
    UserAbort,
}

impl Default for SessionEndStatus {
    fn default() -> Self {
        Self::Normal
    }
}

// ============================================================================
// Conversation Message
// ============================================================================

/// Conversation message for context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    /// Message role: user, assistant, or system
    pub role: String,
    /// Message content
    pub content: String,
}

impl ConversationMessage {
    /// Create new conversation message
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

// ============================================================================
// Hook Payload
// ============================================================================

/// Event-specific payload data
/// Implements REQ-HOOKS-10, REQ-HOOKS-11, REQ-HOOKS-12
///
/// # Serialization
/// Uses tagged enum format: {"type": "session_start", "data": {...}}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum HookPayload {
    /// SessionStart payload - session initialization
    SessionStart {
        /// Working directory path
        cwd: String,
        /// Previous session ID for continuity (optional)
        previous_session_id: Option<String>,
    },

    /// PreToolUse payload - before tool execution
    PreToolUse {
        /// Tool name being invoked
        tool_name: String,
        /// Tool input parameters (arbitrary JSON)
        tool_input: serde_json::Value,
    },

    /// PostToolUse payload - after tool execution
    PostToolUse {
        /// Tool name that was executed
        tool_name: String,
        /// Tool execution result (if successful)
        tool_output: Option<String>,
        /// Error message (if failed)
        error: Option<String>,
    },

    /// UserPromptSubmit payload - user prompt submitted
    UserPromptSubmit {
        /// User's prompt text
        prompt: String,
        /// Conversation context (recent messages)
        context: Option<Vec<ConversationMessage>>,
    },

    /// SessionEnd payload - session termination
    SessionEnd {
        /// Session duration in milliseconds
        duration_ms: u64,
        /// Final status
        status: SessionEndStatus,
    },
}

impl HookPayload {
    // ========================================================================
    // Type Checking Methods
    // ========================================================================

    /// Check if this is a SessionStart payload
    #[inline]
    pub const fn is_session_start(&self) -> bool {
        matches!(self, Self::SessionStart { .. })
    }

    /// Check if this is a PreToolUse payload
    #[inline]
    pub const fn is_pre_tool_use(&self) -> bool {
        matches!(self, Self::PreToolUse { .. })
    }

    /// Check if this is a PostToolUse payload
    #[inline]
    pub const fn is_post_tool_use(&self) -> bool {
        matches!(self, Self::PostToolUse { .. })
    }

    /// Check if this is a UserPromptSubmit payload
    #[inline]
    pub const fn is_user_prompt_submit(&self) -> bool {
        matches!(self, Self::UserPromptSubmit { .. })
    }

    /// Check if this is a SessionEnd payload
    #[inline]
    pub const fn is_session_end(&self) -> bool {
        matches!(self, Self::SessionEnd { .. })
    }

    // ========================================================================
    // Constructor Methods
    // ========================================================================

    /// Create SessionStart payload
    pub fn session_start(cwd: impl Into<String>, previous_session_id: Option<String>) -> Self {
        Self::SessionStart {
            cwd: cwd.into(),
            previous_session_id,
        }
    }

    /// Create PreToolUse payload
    pub fn pre_tool_use(tool_name: impl Into<String>, tool_input: serde_json::Value) -> Self {
        Self::PreToolUse {
            tool_name: tool_name.into(),
            tool_input,
        }
    }

    /// Create PostToolUse payload for successful tool execution
    pub fn post_tool_success(tool_name: impl Into<String>, output: impl Into<String>) -> Self {
        Self::PostToolUse {
            tool_name: tool_name.into(),
            tool_output: Some(output.into()),
            error: None,
        }
    }

    /// Create PostToolUse payload for failed tool execution
    pub fn post_tool_error(tool_name: impl Into<String>, error: impl Into<String>) -> Self {
        Self::PostToolUse {
            tool_name: tool_name.into(),
            tool_output: None,
            error: Some(error.into()),
        }
    }

    /// Create UserPromptSubmit payload
    pub fn user_prompt(prompt: impl Into<String>, context: Option<Vec<ConversationMessage>>) -> Self {
        Self::UserPromptSubmit {
            prompt: prompt.into(),
            context,
        }
    }

    /// Create SessionEnd payload
    pub fn session_end(duration_ms: u64, status: SessionEndStatus) -> Self {
        Self::SessionEnd {
            duration_ms,
            status,
        }
    }

    // ========================================================================
    // Accessor Methods
    // ========================================================================

    /// Get tool name if this is a tool-related payload
    pub fn tool_name(&self) -> Option<&str> {
        match self {
            Self::PreToolUse { tool_name, .. } => Some(tool_name),
            Self::PostToolUse { tool_name, .. } => Some(tool_name),
            _ => None,
        }
    }

    /// Check if PostToolUse indicates an error
    pub fn is_tool_error(&self) -> bool {
        matches!(self, Self::PostToolUse { error: Some(_), .. })
    }
}

// ============================================================================
// Update HookInput (replace the existing definition)
// ============================================================================

/// Input received from Claude Code hook system via stdin
/// Implements REQ-HOOKS-03, REQ-HOOKS-10
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookInput {
    /// Hook event type
    pub hook_type: HookEventType,
    /// Session identifier from Claude Code
    pub session_id: String,
    /// Unix timestamp in milliseconds
    pub timestamp_ms: i64,
    /// Event-specific payload
    pub payload: HookPayload,
}

impl HookInput {
    /// Create new hook input
    pub fn new(
        hook_type: HookEventType,
        session_id: impl Into<String>,
        payload: HookPayload,
    ) -> Self {
        Self {
            hook_type,
            session_id: session_id.into(),
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
            payload,
        }
    }

    /// Parse hook input from JSON string (stdin)
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod hook_payload_tests {
    use super::*;

    #[test]
    fn test_session_start_payload_serialization() {
        let payload = HookPayload::session_start("/home/user/project", None);
        let json = serde_json::to_value(&payload).unwrap();

        assert_eq!(json["type"], "session_start");
        assert_eq!(json["data"]["cwd"], "/home/user/project");
    }

    #[test]
    fn test_pre_tool_use_payload_serialization() {
        let payload = HookPayload::pre_tool_use(
            "Read",
            serde_json::json!({"file_path": "/tmp/test.rs"}),
        );
        let json = serde_json::to_value(&payload).unwrap();

        assert_eq!(json["type"], "pre_tool_use");
        assert_eq!(json["data"]["tool_name"], "Read");
    }

    #[test]
    fn test_post_tool_use_success_serialization() {
        let payload = HookPayload::post_tool_success("Read", "file contents here");
        let json = serde_json::to_value(&payload).unwrap();

        assert_eq!(json["type"], "post_tool_use");
        assert_eq!(json["data"]["tool_output"], "file contents here");
        assert!(json["data"]["error"].is_null());
    }

    #[test]
    fn test_post_tool_use_error_serialization() {
        let payload = HookPayload::post_tool_error("Read", "File not found");
        let json = serde_json::to_value(&payload).unwrap();

        assert_eq!(json["type"], "post_tool_use");
        assert!(json["data"]["tool_output"].is_null());
        assert_eq!(json["data"]["error"], "File not found");
    }

    #[test]
    fn test_user_prompt_submit_serialization() {
        let context = vec![
            ConversationMessage::new("user", "Hello"),
            ConversationMessage::new("assistant", "Hi there!"),
        ];
        let payload = HookPayload::user_prompt("How do I read a file?", Some(context));
        let json = serde_json::to_value(&payload).unwrap();

        assert_eq!(json["type"], "user_prompt_submit");
        assert_eq!(json["data"]["prompt"], "How do I read a file?");
        assert_eq!(json["data"]["context"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_session_end_serialization() {
        let payload = HookPayload::session_end(3600000, SessionEndStatus::Normal);
        let json = serde_json::to_value(&payload).unwrap();

        assert_eq!(json["type"], "session_end");
        assert_eq!(json["data"]["duration_ms"], 3600000);
        assert_eq!(json["data"]["status"], "normal");
    }

    #[test]
    fn test_hook_input_full_roundtrip() {
        let input = HookInput::new(
            HookEventType::SessionStart,
            "test-session-123",
            HookPayload::session_start("/home/user", Some("prev-session".to_string())),
        );

        let json = serde_json::to_string(&input).unwrap();
        let parsed: HookInput = serde_json::from_str(&json).unwrap();

        assert_eq!(input.hook_type, parsed.hook_type);
        assert_eq!(input.session_id, parsed.session_id);
    }

    #[test]
    fn test_payload_type_checking() {
        let payload = HookPayload::session_start("/tmp", None);
        assert!(payload.is_session_start());
        assert!(!payload.is_pre_tool_use());

        let payload = HookPayload::pre_tool_use("Read", serde_json::Value::Null);
        assert!(payload.is_pre_tool_use());
        assert!(!payload.is_session_start());
    }

    #[test]
    fn test_tool_name_accessor() {
        let payload = HookPayload::pre_tool_use("Read", serde_json::Value::Null);
        assert_eq!(payload.tool_name(), Some("Read"));

        let payload = HookPayload::session_start("/tmp", None);
        assert_eq!(payload.tool_name(), None);
    }

    #[test]
    fn test_is_tool_error() {
        let success = HookPayload::post_tool_success("Read", "content");
        assert!(!success.is_tool_error());

        let error = HookPayload::post_tool_error("Read", "failed");
        assert!(error.is_tool_error());
    }
}
```

## Verification Checklist

- [ ] HookPayload enum has 5 variants matching hook event types
- [ ] Tagged enum serialization works: {"type": "...", "data": {...}}
- [ ] ConversationMessage has role and content fields
- [ ] SessionEndStatus has 4 variants (Normal, Timeout, Error, UserAbort)
- [ ] HookInput uses typed HookPayload (not serde_json::Value)
- [ ] All JSON round-trip tests pass
