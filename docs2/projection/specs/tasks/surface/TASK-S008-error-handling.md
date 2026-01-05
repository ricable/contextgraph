# TASK-S008: Fail-Fast Error Handling Throughout

```yaml
metadata:
  id: "TASK-S008"
  title: "Fail-Fast Error Handling Throughout"
  layer: "surface"
  priority: "P0"
  estimated_hours: 6
  created: "2026-01-04"
  status: "pending"
  dependencies:
    - "TASK-S001"  # Memory handlers
    - "TASK-S002"  # Search handlers
    - "TASK-S003"  # Purpose handlers
    - "TASK-S004"  # Johari handlers
    - "TASK-S005"  # Meta-UTL handlers
    - "TASK-S006"  # Integration tests
    - "TASK-S007"  # Remove fusion handlers
  traces_to:
    - "FR-603"  # Fail Fast with Robust Error Logging
```

## Problem Statement

Implement comprehensive fail-fast error handling throughout all MCP handlers with detailed contextual logging. No silent failures, no fallbacks to defaults.

## Context

The system MUST fail immediately and loudly when:
- Dimension mismatches occur
- Legacy data formats are encountered
- Invalid embeddings are received
- Required data is missing
- Constraints are violated

Every error MUST include:
- What failed
- Why it failed
- What the expected value was
- What the actual value was
- Stack context for debugging

**NO silent failures. NO fallbacks. NO defaults for critical data.**

## Technical Specification

### Error Type Hierarchy

```rust
use std::backtrace::Backtrace;
use uuid::Uuid;

/// Top-level MCP error with full context
#[derive(Debug)]
pub struct McpError {
    /// Error code for client handling
    pub code: McpErrorCode,
    /// Human-readable message
    pub message: String,
    /// Structured error details
    pub details: ErrorDetails,
    /// Captured backtrace
    pub backtrace: Backtrace,
    /// Timestamp of error
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Request ID if available
    pub request_id: Option<Uuid>,
}

/// Error codes for categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum McpErrorCode {
    // Validation errors (4xx equivalent)
    DimensionMismatch,
    MissingRequiredField,
    InvalidValue,
    InvalidFormat,
    ConstraintViolation,

    // Data errors
    NotFound,
    AlreadyExists,
    DataCorruption,
    LegacyFormatDetected,

    // System errors (5xx equivalent)
    StorageError,
    IndexError,
    ComputationError,
    Timeout,
    ResourceExhausted,

    // Critical errors (panic-worthy)
    InvariantViolation,
    FusionCodeDetected,
    InternalInconsistency,
}

/// Structured error details with all context
#[derive(Debug, Serialize)]
pub struct ErrorDetails {
    /// Component where error occurred
    pub component: String,
    /// Operation being performed
    pub operation: String,
    /// Expected value/state (if applicable)
    pub expected: Option<serde_json::Value>,
    /// Actual value/state (if applicable)
    pub actual: Option<serde_json::Value>,
    /// Additional context
    pub context: std::collections::HashMap<String, serde_json::Value>,
    /// Suggestions for resolution
    pub suggestions: Vec<String>,
}

impl McpError {
    /// Create dimension mismatch error with full context
    pub fn dimension_mismatch(
        embedder_index: usize,
        embedder_name: &str,
        expected: usize,
        actual: usize,
        request_id: Option<Uuid>,
    ) -> Self {
        Self {
            code: McpErrorCode::DimensionMismatch,
            message: format!(
                "Embedding dimension mismatch for {} (E{}): expected {} dimensions, got {}",
                embedder_name, embedder_index + 1, expected, actual
            ),
            details: ErrorDetails {
                component: "embedding_validation".into(),
                operation: "validate_dimensions".into(),
                expected: Some(serde_json::json!({
                    "embedder_index": embedder_index,
                    "embedder_name": embedder_name,
                    "dimension": expected
                })),
                actual: Some(serde_json::json!({
                    "dimension": actual
                })),
                context: std::collections::HashMap::new(),
                suggestions: vec![
                    format!("Ensure {} model outputs {}-dimensional vectors", embedder_name, expected),
                    "Check embedding pipeline configuration".into(),
                    "Verify model version matches expected output dimensions".into(),
                ],
            },
            backtrace: Backtrace::capture(),
            timestamp: chrono::Utc::now(),
            request_id,
        }
    }

    /// Create missing field error
    pub fn missing_required_field(
        field_name: &str,
        parent_type: &str,
        request_id: Option<Uuid>,
    ) -> Self {
        Self {
            code: McpErrorCode::MissingRequiredField,
            message: format!(
                "Required field '{}' missing in {}",
                field_name, parent_type
            ),
            details: ErrorDetails {
                component: "request_validation".into(),
                operation: "validate_request".into(),
                expected: Some(serde_json::json!({
                    "field": field_name,
                    "required": true
                })),
                actual: Some(serde_json::json!({
                    "field": field_name,
                    "present": false
                })),
                context: [
                    ("parent_type".into(), serde_json::json!(parent_type))
                ].into_iter().collect(),
                suggestions: vec![
                    format!("Add '{}' field to request", field_name),
                    format!("Check API documentation for {} schema", parent_type),
                ],
            },
            backtrace: Backtrace::capture(),
            timestamp: chrono::Utc::now(),
            request_id,
        }
    }

    /// Create legacy format detected error (CRITICAL)
    pub fn legacy_format_detected(
        format_type: &str,
        location: &str,
        request_id: Option<Uuid>,
    ) -> Self {
        Self {
            code: McpErrorCode::LegacyFormatDetected,
            message: format!(
                "CRITICAL: Legacy {} format detected at {}. \
                All fusion/legacy formats have been removed and are not supported.",
                format_type, location
            ),
            details: ErrorDetails {
                component: "format_validation".into(),
                operation: "detect_legacy".into(),
                expected: Some(serde_json::json!({
                    "format": "TeleologicalFingerprint",
                    "version": "2.0"
                })),
                actual: Some(serde_json::json!({
                    "format": format_type,
                    "location": location
                })),
                context: std::collections::HashMap::new(),
                suggestions: vec![
                    "Re-index data using new 12-array fingerprint format".into(),
                    "There is NO migration path - data must be re-embedded".into(),
                    "Contact system administrator if this is unexpected".into(),
                ],
            },
            backtrace: Backtrace::capture(),
            timestamp: chrono::Utc::now(),
            request_id,
        }
    }

    /// Create invariant violation error (CRITICAL - should never happen)
    pub fn invariant_violation(
        invariant: &str,
        details: &str,
        request_id: Option<Uuid>,
    ) -> Self {
        Self {
            code: McpErrorCode::InvariantViolation,
            message: format!(
                "CRITICAL INVARIANT VIOLATION: {} - {}. \
                This indicates a bug in the system.",
                invariant, details
            ),
            details: ErrorDetails {
                component: "invariant_check".into(),
                operation: "verify_invariant".into(),
                expected: Some(serde_json::json!({
                    "invariant": invariant,
                    "should_hold": true
                })),
                actual: Some(serde_json::json!({
                    "invariant": invariant,
                    "violated": true,
                    "details": details
                })),
                context: std::collections::HashMap::new(),
                suggestions: vec![
                    "This is a bug - please report with full stack trace".into(),
                    "Include request_id in bug report".into(),
                ],
            },
            backtrace: Backtrace::capture(),
            timestamp: chrono::Utc::now(),
            request_id,
        }
    }
}
```

### Validation Macros

```rust
/// Validate embedding dimension - fails fast with full context
macro_rules! validate_dimension {
    ($embedding:expr, $embedder_idx:expr, $embedder_name:expr, $expected:expr, $request_id:expr) => {
        {
            let actual = $embedding.len();
            if actual != $expected {
                return Err(McpError::dimension_mismatch(
                    $embedder_idx,
                    $embedder_name,
                    $expected,
                    actual,
                    $request_id,
                ));
            }
        }
    };
}

/// Validate required field - fails fast with context
macro_rules! require_field {
    ($value:expr, $field:expr, $parent:expr, $request_id:expr) => {
        match $value {
            Some(v) => v,
            None => return Err(McpError::missing_required_field($field, $parent, $request_id)),
        }
    };
}

/// Ensure no fusion code paths - fails immediately
macro_rules! ensure_no_fusion {
    ($context:expr) => {
        if cfg!(feature = "fusion") {
            panic!(
                "CRITICAL: Fusion feature flag detected in {}. \
                This should be impossible in the new architecture.",
                $context
            );
        }
    };
}

/// Validate value range - fails with context
macro_rules! validate_range {
    ($value:expr, $min:expr, $max:expr, $name:expr, $request_id:expr) => {
        if $value < $min || $value > $max {
            return Err(McpError::invalid_value(
                $name,
                format!("[{}, {}]", $min, $max),
                $value.to_string(),
                $request_id,
            ));
        }
    };
}
```

### Logging Integration

```rust
use tracing::{error, warn, info, debug, instrument, Span};

/// Log error with full context and return
#[instrument(skip_all, fields(error_code = ?err.code, request_id = ?err.request_id))]
pub fn log_and_return_error(err: McpError) -> McpError {
    error!(
        code = ?err.code,
        message = %err.message,
        component = %err.details.component,
        operation = %err.details.operation,
        expected = ?err.details.expected,
        actual = ?err.details.actual,
        timestamp = %err.timestamp,
        "MCP handler error"
    );

    // For critical errors, also capture backtrace
    if matches!(err.code,
        McpErrorCode::InvariantViolation |
        McpErrorCode::FusionCodeDetected |
        McpErrorCode::InternalInconsistency
    ) {
        error!(backtrace = %err.backtrace, "Critical error backtrace");
    }

    err
}

/// Structured error logging for failed operations
pub fn log_operation_failure(
    operation: &str,
    component: &str,
    err: &McpError,
    duration_ms: u64,
) {
    error!(
        operation = %operation,
        component = %component,
        error_code = ?err.code,
        message = %err.message,
        duration_ms = %duration_ms,
        request_id = ?err.request_id,
        "Operation failed"
    );
}
```

### Handler Error Patterns

```rust
/// Example handler with proper error handling
#[instrument(skip(store), fields(request_id = %request.request_id))]
pub async fn handle_memory_store(
    request: MemoryStoreRequest,
    store: Arc<dyn TeleologicalMemoryStore>,
) -> Result<MemoryStoreResponse, McpError> {
    let request_id = Some(request.request_id);

    // Validate all dimensions - fail fast
    validate_dimension!(request.embeddings.e1, 0, "E1_text_general", 1024, request_id);
    validate_dimension!(request.embeddings.e2, 1, "E2_text_small", 512, request_id);
    validate_dimension!(request.embeddings.e3, 2, "E3_multilingual", 512, request_id);
    validate_dimension!(request.embeddings.e4, 3, "E4_code", 512, request_id);
    validate_dimension!(request.embeddings.e5.query, 4, "E5_query", 768, request_id);
    validate_dimension!(request.embeddings.e5.doc, 4, "E5_doc", 768, request_id);
    // E6 sparse - validate indices
    for &idx in &request.embeddings.e6.indices {
        if idx as usize >= 30_000 {
            return Err(McpError::invalid_value(
                "E6_sparse_index",
                "[0, 29999]",
                idx.to_string(),
                request_id,
            ));
        }
    }
    validate_dimension!(request.embeddings.e7, 6, "E7_openai_ada", 1536, request_id);
    validate_dimension!(request.embeddings.e8, 7, "E8_minilm", 384, request_id);
    validate_dimension!(request.embeddings.e9, 8, "E9_simhash", 1024, request_id);
    validate_dimension!(request.embeddings.e10, 9, "E10_instructor", 768, request_id);
    validate_dimension!(request.embeddings.e11, 10, "E11_fast", 384, request_id);
    // E12 late-interaction - validate per-token dimension
    for (i, token_emb) in request.embeddings.e12.iter().enumerate() {
        if token_emb.len() != 128 {
            return Err(McpError::dimension_mismatch(
                11,
                &format!("E12_token_{}", i),
                128,
                token_emb.len(),
                request_id,
            ));
        }
    }

    // Validate purpose vector if provided
    if let Some(ref pv) = request.purpose_vector {
        if pv.len() != 12 {
            return Err(McpError::dimension_mismatch(
                0,
                "purpose_vector",
                12,
                pv.len(),
                request_id,
            ));
        }
        for (i, &alignment) in pv.iter().enumerate() {
            validate_range!(alignment, -1.0, 1.0, &format!("purpose_vector[{}]", i), request_id);
        }
    }

    // Store with error context
    let fingerprint = TeleologicalFingerprint::try_from_request(&request)
        .map_err(|e| {
            log_and_return_error(McpError::invalid_format(
                "TeleologicalFingerprint conversion",
                e.to_string(),
                request_id,
            ))
        })?;

    let id = store.store(fingerprint).await.map_err(|e| {
        log_and_return_error(McpError::storage_error(
            "store",
            e.to_string(),
            request_id,
        ))
    })?;

    Ok(MemoryStoreResponse {
        id,
        storage_size_bytes: 46_000, // Approximate
        embedder_dimensions: [1024, 512, 512, 512, 768, 0, 1536, 384, 1024, 768, 384, 128],
        north_star_alignment: fingerprint.theta_to_north_star,
        created_at: fingerprint.created_at,
    })
}
```

### JSON Error Response Format

```json
{
  "error": {
    "code": "dimension_mismatch",
    "message": "Embedding dimension mismatch for E1_text_general (E1): expected 1024 dimensions, got 512",
    "details": {
      "component": "embedding_validation",
      "operation": "validate_dimensions",
      "expected": {
        "embedder_index": 0,
        "embedder_name": "E1_text_general",
        "dimension": 1024
      },
      "actual": {
        "dimension": 512
      },
      "context": {},
      "suggestions": [
        "Ensure E1_text_general model outputs 1024-dimensional vectors",
        "Check embedding pipeline configuration",
        "Verify model version matches expected output dimensions"
      ]
    },
    "timestamp": "2026-01-04T12:00:00Z",
    "request_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

## Implementation Requirements

### Prerequisites

All other Surface Layer tasks must be complete (S001-S007).

### Scope

#### In Scope

- Comprehensive error type hierarchy
- Validation macros for common checks
- Structured error logging
- JSON error response format
- Error context in all handlers
- Backtrace capture for critical errors

#### Out of Scope

- Specific handler implementations (other S tasks)
- Error monitoring infrastructure
- Error analytics

### Constraints

- NO `unwrap()` in handler code
- NO `expect()` without descriptive message
- NO silent error swallowing
- NO fallback to defaults for critical data
- ALL errors must include context

## Definition of Done

### Implementation Checklist

- [ ] McpError type with full context
- [ ] McpErrorCode enum for all categories
- [ ] ErrorDetails struct with suggestions
- [ ] Validation macros implemented
- [ ] Logging integration complete
- [ ] All handlers use proper error handling
- [ ] JSON error format implemented
- [ ] No unwrap() in handler code
- [ ] Backtrace capture for critical errors

### Testing Requirements

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_dimension_mismatch_error_context() {
        let err = McpError::dimension_mismatch(0, "E1_text_general", 1024, 512, Some(Uuid::new_v4()));

        assert_eq!(err.code, McpErrorCode::DimensionMismatch);
        assert!(err.message.contains("1024"));
        assert!(err.message.contains("512"));
        assert!(err.details.suggestions.len() > 0);
        assert!(err.backtrace.status() == BacktraceStatus::Captured);
    }

    #[test]
    fn test_legacy_format_error_is_critical() {
        let err = McpError::legacy_format_detected("Vector1536", "storage", None);

        assert_eq!(err.code, McpErrorCode::LegacyFormatDetected);
        assert!(err.message.contains("CRITICAL"));
        assert!(err.details.suggestions.iter().any(|s| s.contains("NO migration")));
    }

    #[tokio::test]
    async fn test_handler_fails_fast_on_dimension_mismatch() {
        let mut request = valid_memory_store_request();
        request.embeddings.e1 = vec![0.0; 512]; // Wrong dimension

        let result = handle_memory_store(request, store.clone()).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, McpErrorCode::DimensionMismatch);
    }

    #[test]
    fn test_no_unwrap_in_handlers() {
        // This test uses grep to verify no unwrap() calls in handlers
        let output = std::process::Command::new("rg")
            .args(&["-c", r"\.unwrap\(\)", "crates/context-graph-mcp/src/handlers/"])
            .output()
            .expect("Failed to run rg");

        let count: usize = String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|l| l.split(':').last())
            .filter_map(|n| n.parse().ok())
            .sum();

        assert_eq!(count, 0, "Found {} unwrap() calls in handlers", count);
    }
}
```

### Verification Commands

```bash
# Verify no unwrap() in handlers
rg "\.unwrap\(\)" crates/context-graph-mcp/src/handlers/ && echo "FAIL: Found unwrap()" || echo "PASS: No unwrap()"

# Verify all errors include request_id
rg "McpError::" crates/context-graph-mcp/src/handlers/ | grep -v "request_id"

# Run error handling tests
cargo test -p context-graph-mcp error_handling
```

## Files to Create

| File | Description |
|------|-------------|
| `crates/context-graph-mcp/src/error.rs` | McpError and related types |
| `crates/context-graph-mcp/src/validation.rs` | Validation macros |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-mcp/src/handlers/*.rs` | Use proper error handling |
| `crates/context-graph-mcp/src/lib.rs` | Export error types |

## Traceability

| Requirement | Source | Coverage |
|-------------|--------|----------|
| FR-603 | FUNC-SPEC-001 | Fail fast with context |
| AC-603.1 | FUNC-SPEC-001 | Invalid dimension triggers error |
| AC-603.2 | FUNC-SPEC-001 | Legacy format logs with trace |
| AC-603.3 | FUNC-SPEC-001 | All errors include context |
| AC-603.4 | FUNC-SPEC-001 | No unwrap() in production |

---

*Task created: 2026-01-04*
*Layer: Surface*
*Priority: P0 - System reliability*
