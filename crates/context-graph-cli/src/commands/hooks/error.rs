//! Error types for hook commands
//!
//! # Exit Codes (TECH-HOOKS.md Section 3.2)
//!
//! | Code | Meaning | Description |
//! |------|---------|-------------|
//! | 0 | Success | Hook executed successfully |
//! | 1 | General Error | Unspecified error |
//! | 2 | Timeout | Operation exceeded timeout |
//! | 3 | Database Error | Storage operation failed |
//! | 4 | Invalid Input | Malformed input data |
//! | 5 | Session Not Found | Referenced session doesn't exist |
//! | 6 | Crisis Triggered | IC dropped below crisis threshold |
//!
//! # Constitution References
//! - IDENTITY-002: IC thresholds (Healthy>0.9, Warning<0.7, Critical<0.5)
//! - AP-26: Exit codes (0=success, 1=error, 2=corruption)
//!
//! # NO BACKWARDS COMPATIBILITY
//! This module FAILS FAST on any error. Do not add fallback logic.

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

    /// Database corruption detected (AP-26)
    /// Exit code: 2
    #[error("Corruption detected: {0}")]
    Corruption(String),

    /// Topic stability crisis triggered (stability below threshold)
    /// Exit code: 6 (special state, not failure)
    /// Constitution: topic_stability defines < 0.5 as crisis
    #[error("Crisis threshold breached: stability={0}")]
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
    // Exit Code Mapping (TECH-HOOKS.md Section 3.2)
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
    #[inline]
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Corruption(_) => 2, // AP-26: corruption = exit code 2
            Self::Timeout(_) => 2,    // Legacy: also mapped to 2
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
    #[inline]
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
            // Corruption is NOT recoverable - requires database repair
            Self::Corruption(_) => false,
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
    /// Constitution: IDENTITY-002 defines IC < 0.5 as crisis threshold.
    #[inline]
    pub fn is_crisis(&self) -> bool {
        matches!(self, Self::CrisisTriggered(_))
    }

    /// Check if this is a timeout error
    #[inline]
    pub fn is_timeout(&self) -> bool {
        matches!(self, Self::Timeout(_))
    }

    // ========================================================================
    // Error Code Strings
    // ========================================================================

    /// Get error code string (e.g., "ERR_TIMEOUT")
    #[inline]
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::Timeout(_) => "ERR_TIMEOUT",
            Self::Storage(_) => "ERR_DATABASE",
            Self::InvalidInput(_) => "ERR_INVALID_INPUT",
            Self::Serialization(_) => "ERR_SERIALIZATION",
            Self::SessionNotFound(_) => "ERR_SESSION_NOT_FOUND",
            Self::CrisisTriggered(_) => "ERR_CRISIS",
            Self::Corruption(_) => "ERR_CORRUPTION",
            Self::Io(_) => "ERR_IO",
            Self::General(_) => "ERR_GENERAL",
        }
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

    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create timeout error with duration
    #[inline]
    pub fn timeout(timeout_ms: u64) -> Self {
        Self::Timeout(timeout_ms)
    }

    /// Create invalid input error
    #[inline]
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }

    /// Create storage error
    #[inline]
    pub fn storage(message: impl Into<String>) -> Self {
        Self::Storage(message.into())
    }

    /// Create session not found error
    #[inline]
    pub fn session_not_found(session_id: impl Into<String>) -> Self {
        Self::SessionNotFound(session_id.into())
    }

    /// Create crisis triggered error
    #[inline]
    pub fn crisis(ic_value: f32) -> Self {
        Self::CrisisTriggered(ic_value)
    }

    /// Create general error
    #[inline]
    pub fn general(message: impl Into<String>) -> Self {
        Self::General(message.into())
    }

    /// Create corruption error (AP-26: exit code 2)
    #[inline]
    pub fn corruption(message: impl Into<String>) -> Self {
        Self::Corruption(message.into())
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

    // ========================================================================
    // Exit Code Tests (TC-HOOKS-005-001)
    // ========================================================================

    #[test]
    fn test_exit_codes_match_spec() {
        // Verify exit codes match TECH-HOOKS.md section 3.2 EXACTLY
        assert_eq!(
            HookError::timeout(100).exit_code(),
            2,
            "Timeout must be exit code 2"
        );
        assert_eq!(
            HookError::storage("db error").exit_code(),
            3,
            "Storage must be exit code 3"
        );
        assert_eq!(
            HookError::invalid_input("bad data").exit_code(),
            4,
            "InvalidInput must be exit code 4"
        );
        assert_eq!(
            HookError::session_not_found("test-123").exit_code(),
            5,
            "SessionNotFound must be exit code 5"
        );
        assert_eq!(
            HookError::crisis(0.45).exit_code(),
            6,
            "CrisisTriggered must be exit code 6"
        );
        assert_eq!(
            HookError::general("something").exit_code(),
            1,
            "General must be exit code 1"
        );
    }

    #[test]
    fn test_serialization_error_exit_code() {
        let json_err = serde_json::from_str::<String>("invalid json");
        if let Err(e) = json_err {
            let hook_err = HookError::from(e);
            assert_eq!(
                hook_err.exit_code(),
                4,
                "Serialization errors must be exit code 4"
            );
        } else {
            panic!("Expected JSON parse error");
        }
    }

    #[test]
    fn test_io_error_exit_code() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let hook_err = HookError::from(io_err);
        assert_eq!(hook_err.exit_code(), 1, "IO errors must be exit code 1");
    }

    // ========================================================================
    // Error Code String Tests (TC-HOOKS-005-002)
    // ========================================================================

    #[test]
    fn test_error_codes() {
        assert_eq!(HookError::timeout(100).error_code(), "ERR_TIMEOUT");
        assert_eq!(HookError::storage("db").error_code(), "ERR_DATABASE");
        assert_eq!(
            HookError::invalid_input("x").error_code(),
            "ERR_INVALID_INPUT"
        );
        assert_eq!(
            HookError::session_not_found("x").error_code(),
            "ERR_SESSION_NOT_FOUND"
        );
        assert_eq!(HookError::crisis(0.4).error_code(), "ERR_CRISIS");
        assert_eq!(HookError::general("x").error_code(), "ERR_GENERAL");
    }

    #[test]
    fn test_serialization_error_code() {
        let json_err = serde_json::from_str::<String>("{}");
        if let Err(e) = json_err {
            let hook_err = HookError::from(e);
            assert_eq!(hook_err.error_code(), "ERR_SERIALIZATION");
        }
    }

    #[test]
    fn test_io_error_code() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let hook_err = HookError::from(io_err);
        assert_eq!(hook_err.error_code(), "ERR_IO");
    }

    // ========================================================================
    // Recoverable Tests (TC-HOOKS-005-003)
    // ========================================================================

    #[test]
    fn test_is_recoverable() {
        // Recoverable
        assert!(
            HookError::timeout(100).is_recoverable(),
            "Timeout should be recoverable"
        );
        assert!(
            HookError::storage("db").is_recoverable(),
            "Storage should be recoverable"
        );
        assert!(
            HookError::crisis(0.4).is_recoverable(),
            "Crisis should be recoverable"
        );

        // Not recoverable
        assert!(
            !HookError::invalid_input("bad").is_recoverable(),
            "InvalidInput should not be recoverable"
        );
        assert!(
            !HookError::session_not_found("test").is_recoverable(),
            "SessionNotFound should not be recoverable"
        );
        assert!(
            !HookError::general("x").is_recoverable(),
            "General should not be recoverable"
        );
    }

    #[test]
    fn test_io_is_recoverable() {
        let io_err = std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout");
        let hook_err = HookError::from(io_err);
        assert!(hook_err.is_recoverable(), "IO errors should be recoverable");
    }

    // ========================================================================
    // Crisis Detection Tests (TC-HOOKS-005-007)
    // ========================================================================

    #[test]
    fn test_is_crisis() {
        assert!(HookError::crisis(0.4).is_crisis());
        assert!(HookError::crisis(0.0).is_crisis());
        assert!(HookError::crisis(0.49).is_crisis());
        assert!(!HookError::timeout(100).is_crisis());
        assert!(!HookError::general("err").is_crisis());
        assert!(!HookError::storage("db").is_crisis());
    }

    // ========================================================================
    // Timeout Detection Tests (TC-HOOKS-005-008)
    // ========================================================================

    #[test]
    fn test_is_timeout() {
        assert!(HookError::timeout(100).is_timeout());
        assert!(HookError::timeout(0).is_timeout());
        assert!(HookError::timeout(u64::MAX).is_timeout());
        assert!(!HookError::crisis(0.4).is_timeout());
        assert!(!HookError::general("timeout").is_timeout());
    }

    // ========================================================================
    // JSON Error Tests (TC-HOOKS-005-004)
    // ========================================================================

    #[test]
    fn test_to_json_error() {
        let err = HookError::timeout(100);
        let json = err.to_json_error();

        assert_eq!(json["error"], true);
        assert_eq!(json["code"], "ERR_TIMEOUT");
        assert_eq!(json["exit_code"], 2);
        assert_eq!(json["message"], "Hook timeout after 100ms");
        assert_eq!(json["recoverable"], true);
        assert_eq!(json["crisis"], false);
    }

    #[test]
    fn test_crisis_json_error() {
        let err = HookError::crisis(0.45);
        let json = err.to_json_error();

        assert_eq!(json["error"], true);
        assert_eq!(json["code"], "ERR_CRISIS");
        assert_eq!(json["exit_code"], 6);
        assert_eq!(json["recoverable"], true);
        assert_eq!(json["crisis"], true);
    }

    // ========================================================================
    // From Implementation Tests (TC-HOOKS-005-005)
    // ========================================================================

    #[test]
    fn test_from_string() {
        let err: HookError = String::from("test error").into();
        assert!(matches!(err, HookError::General(_)));
        assert_eq!(err.exit_code(), 1);
        assert_eq!(err.to_string(), "test error");
    }

    #[test]
    fn test_from_str() {
        let err: HookError = "test error".into();
        assert!(matches!(err, HookError::General(_)));
        assert_eq!(err.exit_code(), 1);
    }

    // ========================================================================
    // Display Tests (TC-HOOKS-005-006)
    // ========================================================================

    #[test]
    fn test_error_display() {
        assert_eq!(
            HookError::timeout(100).to_string(),
            "Hook timeout after 100ms"
        );
        assert_eq!(
            HookError::crisis(0.45).to_string(),
            "Crisis threshold breached: stability=0.45"
        );
        assert_eq!(
            HookError::session_not_found("abc-123").to_string(),
            "Session not found: abc-123"
        );
        assert_eq!(
            HookError::storage("connection failed").to_string(),
            "Database error: connection failed"
        );
        assert_eq!(
            HookError::invalid_input("missing field").to_string(),
            "Invalid input: missing field"
        );
    }

    // ========================================================================
    // Boundary Tests
    // ========================================================================

    #[test]
    fn test_timeout_zero() {
        let err = HookError::timeout(0);
        assert_eq!(err.exit_code(), 2);
        assert!(err.is_timeout());
        assert_eq!(err.to_string(), "Hook timeout after 0ms");
    }

    #[test]
    fn test_timeout_max() {
        let err = HookError::timeout(u64::MAX);
        assert_eq!(err.exit_code(), 2);
        assert!(err.is_timeout());
    }

    #[test]
    fn test_crisis_threshold() {
        // IC = 0.5 is still critical (< threshold means < 0.5)
        let err = HookError::crisis(0.5);
        assert_eq!(err.exit_code(), 6);
        assert!(err.is_crisis());
    }

    #[test]
    fn test_crisis_below_threshold() {
        let err = HookError::crisis(0.49);
        assert_eq!(err.exit_code(), 6);
        assert!(err.is_crisis());
    }

    #[test]
    fn test_empty_session_id() {
        let err = HookError::session_not_found("");
        assert_eq!(err.exit_code(), 5);
        assert_eq!(err.to_string(), "Session not found: ");
    }

    #[test]
    fn test_unicode_error() {
        let err = HookError::general("错误消息");
        assert_eq!(err.exit_code(), 1);
        assert_eq!(err.to_string(), "错误消息");
    }

    #[test]
    fn test_empty_string_error() {
        let err = HookError::general("");
        assert_eq!(err.exit_code(), 1);
        assert_eq!(err.to_string(), "");
    }

    // ========================================================================
    // Constructor Tests
    // ========================================================================

    #[test]
    fn test_constructors() {
        // Verify all constructors create correct variants
        assert!(matches!(HookError::timeout(100), HookError::Timeout(100)));
        assert!(matches!(
            HookError::invalid_input("x"),
            HookError::InvalidInput(_)
        ));
        assert!(matches!(HookError::storage("x"), HookError::Storage(_)));
        assert!(matches!(
            HookError::session_not_found("x"),
            HookError::SessionNotFound(_)
        ));
        assert!(matches!(
            HookError::crisis(0.4),
            HookError::CrisisTriggered(_)
        ));
        assert!(matches!(HookError::general("x"), HookError::General(_)));
    }

    // ========================================================================
    // HookResult Type Alias Test
    // ========================================================================

    #[test]
    fn test_hook_result_type() {
        fn returns_result() -> HookResult<i32> {
            Ok(42)
        }

        fn returns_error() -> HookResult<i32> {
            Err(HookError::timeout(100))
        }

        assert_eq!(returns_result().unwrap(), 42);
        assert!(returns_error().is_err());
    }
}
