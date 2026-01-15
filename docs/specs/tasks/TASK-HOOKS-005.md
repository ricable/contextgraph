# TASK-HOOKS-005: Create HookError Enum

```xml
<task_spec id="TASK-HOOKS-005" version="1.0">
<metadata>
  <title>Create HookError Enum with Exit Codes</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>5</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-40</requirement_ref>
    <requirement_ref>REQ-HOOKS-43</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-001</task_ref>
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
  <estimated_hours>0.5</estimated_hours>
</metadata>

<context>
This task creates the HookError enum that defines all error types for hook operations.
Each error variant maps to a specific exit code following the specification in TECH-HOOKS.md.
Error handling must be robust since hooks run in shell scripts where exit codes are critical.

Exit Code Mapping:
- 0: Success
- 1: General Error
- 2: Timeout
- 3: Database Error
- 4: Invalid Input
- 5: Session Not Found
- 6: Crisis Triggered (not failure, but special state)
</context>

<input_context_files>
  <file purpose="exit_code_spec">docs/specs/technical/TECH-HOOKS.md#section-3.2</file>
  <file purpose="error_handling">docs/specs/technical/TECH-HOOKS.md#section-6.4</file>
</input_context_files>

<prerequisites>
  <check>TASK-HOOKS-001 completed (HookEventType exists)</check>
  <check>thiserror is a workspace dependency</check>
</prerequisites>

<scope>
  <in_scope>
    - Create HookError enum with all error variants
    - Implement exit_code() method for each variant
    - Implement From conversions for common errors
    - Add comprehensive documentation
    - Create unit tests
  </in_scope>
  <out_of_scope>
    - Error recovery logic (handled in command implementations)
    - Logging infrastructure
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/commands/hooks/error.rs">
use thiserror::Error;

/// Hook-specific error types
/// Implements REQ-HOOKS-40, REQ-HOOKS-43
#[derive(Debug, Error)]
pub enum HookError {
    #[error("Hook timeout after {0}ms")]
    Timeout(u64),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Database error: {0}")]
    Storage(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Crisis threshold breached: IC={0}")]
    CrisisTriggered(f32),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    General(String),
}

impl HookError {
    /// Convert to exit code per TECH-HOOKS.md section 3.2
    pub fn exit_code(&self) -> i32;
}
    </signature>
  </signatures>
  <constraints>
    - Exit codes MUST match specification exactly
    - thiserror MUST be used for Error derive
    - CrisisTriggered is exit code 6 (special state, not failure)
    - From implementations for serde_json::Error and std::io::Error
  </constraints>
  <verification>
    - cargo test --package context-graph-cli hook_error
    - Verify exit codes match spec
  </verification>
</definition_of_done>

<pseudo_code>
1. Create error.rs file

2. Define HookError enum with thiserror:
   - Timeout(u64) - exit 2
   - InvalidInput(String) - exit 4
   - Storage(String) - exit 3
   - Serialization(serde_json::Error) - exit 4
   - SessionNotFound(String) - exit 5
   - CrisisTriggered(f32) - exit 6
   - Io(std::io::Error) - exit 1
   - General(String) - exit 1

3. Implement exit_code() method:
   match self:
     Timeout(_) => 2
     Storage(_) => 3
     InvalidInput(_) | Serialization(_) => 4
     SessionNotFound(_) => 5
     CrisisTriggered(_) => 6
     Io(_) | General(_) => 1

4. Implement From conversions:
   - From<serde_json::Error> (automatic via #[from])
   - From<std::io::Error> (automatic via #[from])
   - From<String> for General variant

5. Add helper methods:
   - is_recoverable() -> bool
   - to_json_error() -> serde_json::Value

6. Add tests for exit code mapping
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/commands/hooks/error.rs">HookError enum and related implementations</file>
</files_to_create>

<files_to_modify>
  <!-- None - module registration in TASK-HOOKS-011 -->
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli hook_error</command>
</test_commands>
</task_spec>
```

## Implementation

### Create error.rs

```rust
// crates/context-graph-cli/src/commands/hooks/error.rs
//! Error types for hook commands
//!
//! # Exit Codes
//! | Code | Meaning | Description |
//! |------|---------|-------------|
//! | 0 | Success | Hook executed successfully |
//! | 1 | General Error | Unspecified error |
//! | 2 | Timeout | Operation exceeded timeout |
//! | 3 | Database Error | Storage operation failed |
//! | 4 | Invalid Input | Malformed input data |
//! | 5 | Session Not Found | Referenced session doesn't exist |
//! | 6 | Crisis Triggered | IC dropped below crisis threshold |

use thiserror::Error;

/// Hook-specific error types
/// Implements REQ-HOOKS-40, REQ-HOOKS-43
#[derive(Debug, Error)]
pub enum HookError {
    /// Hook execution timed out
    /// Exit code: 2
    #[error("Hook timeout after {0}ms")]
    Timeout(u64),

    /// Invalid or malformed input data
    /// Exit code: 4
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Database/storage operation failed
    /// Exit code: 3
    #[error("Database error: {0}")]
    Storage(String),

    /// JSON serialization/deserialization error
    /// Exit code: 4
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Referenced session does not exist
    /// Exit code: 5
    #[error("Session not found: {0}")]
    SessionNotFound(String),

    /// Identity continuity crisis triggered (IC below threshold)
    /// Exit code: 6 (special state, not failure)
    #[error("Crisis threshold breached: IC={0}")]
    CrisisTriggered(f32),

    /// IO operation failed
    /// Exit code: 1
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// General/unspecified error
    /// Exit code: 1
    #[error("{0}")]
    General(String),
}

impl HookError {
    // ========================================================================
    // Exit Code Mapping
    // ========================================================================

    /// Convert to exit code per TECH-HOOKS.md section 3.2
    ///
    /// # Exit Codes
    /// - 0: Success (not an error)
    /// - 1: General Error
    /// - 2: Timeout
    /// - 3: Database Error
    /// - 4: Invalid Input
    /// - 5: Session Not Found
    /// - 6: Crisis Triggered
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Timeout(_) => 2,
            Self::Storage(_) => 3,
            Self::InvalidInput(_) | Self::Serialization(_) => 4,
            Self::SessionNotFound(_) => 5,
            Self::CrisisTriggered(_) => 6,
            Self::Io(_) | Self::General(_) => 1,
        }
    }

    // ========================================================================
    // Error Classification
    // ========================================================================

    /// Check if this error is recoverable
    ///
    /// Recoverable errors may succeed on retry or with different input.
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Timeouts may succeed with retry
            Self::Timeout(_) => true,
            // Storage errors may be transient
            Self::Storage(_) => true,
            // IO errors may be transient
            Self::Io(_) => true,
            // Crisis triggered is a state, not a failure
            Self::CrisisTriggered(_) => true,
            // These require fixing the input/code
            Self::InvalidInput(_)
            | Self::Serialization(_)
            | Self::SessionNotFound(_)
            | Self::General(_) => false,
        }
    }

    /// Check if this error indicates a crisis state (not a failure)
    ///
    /// Crisis errors require special handling (e.g., triggering auto-dream).
    pub fn is_crisis(&self) -> bool {
        matches!(self, Self::CrisisTriggered(_))
    }

    /// Check if this is a timeout error
    pub fn is_timeout(&self) -> bool {
        matches!(self, Self::Timeout(_))
    }

    // ========================================================================
    // JSON Error Format
    // ========================================================================

    /// Convert to structured JSON error for shell script consumption
    ///
    /// # Returns
    /// JSON object with error code, message, and metadata
    pub fn to_json_error(&self) -> serde_json::Value {
        serde_json::json!({
            "error": true,
            "code": self.error_code(),
            "exit_code": self.exit_code(),
            "message": self.to_string(),
            "recoverable": self.is_recoverable(),
            "crisis": self.is_crisis(),
        })
    }

    /// Get error code string (e.g., "ERR_TIMEOUT")
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::Timeout(_) => "ERR_TIMEOUT",
            Self::Storage(_) => "ERR_DATABASE",
            Self::InvalidInput(_) => "ERR_INVALID_INPUT",
            Self::Serialization(_) => "ERR_SERIALIZATION",
            Self::SessionNotFound(_) => "ERR_SESSION_NOT_FOUND",
            Self::CrisisTriggered(_) => "ERR_CRISIS",
            Self::Io(_) => "ERR_IO",
            Self::General(_) => "ERR_GENERAL",
        }
    }

    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create timeout error with duration
    pub fn timeout(timeout_ms: u64) -> Self {
        Self::Timeout(timeout_ms)
    }

    /// Create invalid input error
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }

    /// Create storage error
    pub fn storage(message: impl Into<String>) -> Self {
        Self::Storage(message.into())
    }

    /// Create session not found error
    pub fn session_not_found(session_id: impl Into<String>) -> Self {
        Self::SessionNotFound(session_id.into())
    }

    /// Create crisis triggered error
    pub fn crisis(ic_value: f32) -> Self {
        Self::CrisisTriggered(ic_value)
    }

    /// Create general error
    pub fn general(message: impl Into<String>) -> Self {
        Self::General(message.into())
    }
}

// ============================================================================
// From Implementations
// ============================================================================

impl From<String> for HookError {
    fn from(s: String) -> Self {
        Self::General(s)
    }
}

impl From<&str> for HookError {
    fn from(s: &str) -> Self {
        Self::General(s.to_string())
    }
}

// ============================================================================
// Result Type Alias
// ============================================================================

/// Result type for hook operations
pub type HookResult<T> = Result<T, HookError>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exit_codes_match_spec() {
        // Verify exit codes match TECH-HOOKS.md section 3.2
        assert_eq!(HookError::timeout(100).exit_code(), 2);
        assert_eq!(HookError::storage("db error").exit_code(), 3);
        assert_eq!(HookError::invalid_input("bad data").exit_code(), 4);
        assert_eq!(HookError::session_not_found("test-123").exit_code(), 5);
        assert_eq!(HookError::crisis(0.45).exit_code(), 6);
        assert_eq!(HookError::general("something").exit_code(), 1);
    }

    #[test]
    fn test_serialization_error_exit_code() {
        let json_err = serde_json::from_str::<String>("invalid json");
        if let Err(e) = json_err {
            let hook_err = HookError::from(e);
            assert_eq!(hook_err.exit_code(), 4);
        }
    }

    #[test]
    fn test_io_error_exit_code() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let hook_err = HookError::from(io_err);
        assert_eq!(hook_err.exit_code(), 1);
    }

    #[test]
    fn test_is_recoverable() {
        assert!(HookError::timeout(100).is_recoverable());
        assert!(HookError::storage("db").is_recoverable());
        assert!(HookError::crisis(0.4).is_recoverable());
        assert!(!HookError::invalid_input("bad").is_recoverable());
        assert!(!HookError::session_not_found("test").is_recoverable());
    }

    #[test]
    fn test_is_crisis() {
        assert!(HookError::crisis(0.4).is_crisis());
        assert!(!HookError::timeout(100).is_crisis());
        assert!(!HookError::general("err").is_crisis());
    }

    #[test]
    fn test_error_codes() {
        assert_eq!(HookError::timeout(100).error_code(), "ERR_TIMEOUT");
        assert_eq!(HookError::storage("db").error_code(), "ERR_DATABASE");
        assert_eq!(HookError::invalid_input("x").error_code(), "ERR_INVALID_INPUT");
        assert_eq!(HookError::session_not_found("x").error_code(), "ERR_SESSION_NOT_FOUND");
        assert_eq!(HookError::crisis(0.4).error_code(), "ERR_CRISIS");
    }

    #[test]
    fn test_to_json_error() {
        let err = HookError::timeout(100);
        let json = err.to_json_error();

        assert_eq!(json["error"], true);
        assert_eq!(json["code"], "ERR_TIMEOUT");
        assert_eq!(json["exit_code"], 2);
        assert_eq!(json["recoverable"], true);
        assert_eq!(json["crisis"], false);
    }

    #[test]
    fn test_from_string() {
        let err: HookError = "test error".into();
        assert!(matches!(err, HookError::General(_)));
        assert_eq!(err.exit_code(), 1);
    }

    #[test]
    fn test_error_display() {
        assert_eq!(
            HookError::timeout(100).to_string(),
            "Hook timeout after 100ms"
        );
        assert_eq!(
            HookError::crisis(0.45).to_string(),
            "Crisis threshold breached: IC=0.45"
        );
    }
}
```

## Verification Checklist

- [ ] All 8 error variants defined
- [ ] Exit codes match TECH-HOOKS.md section 3.2 exactly
- [ ] From implementations for serde_json::Error and std::io::Error
- [ ] is_recoverable() returns correct values
- [ ] to_json_error() produces valid JSON structure
- [ ] All tests pass
