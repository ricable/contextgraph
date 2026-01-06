# TASK-EMB-003: Create ProjectionError Enum

<task_spec id="TASK-EMB-003" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Title** | Create ProjectionError Enum |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 3 |
| **Implements** | REQ-EMB-001 |
| **Depends On** | TASK-EMB-002 |
| **Estimated Complexity** | low |
| **Parallel Group** | C |

---

## Context

The ProjectionMatrix requires dedicated error types for:
- Missing weight files (no fallback allowed)
- Dimension mismatches
- GPU/CUDA failures
- Checksum validation failures

Constitution rule `AP-007` forbids stub data in prod, so these errors MUST cause failures, not fallbacks.

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| ProjectionMatrix | `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` (TASK-EMB-002) |
| Constitution | `docs2/constitution.yaml` sections `forbidden`, `rules` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-001-sparse-projection.md` |

---

## Prerequisites

- [ ] TASK-EMB-002 completed (ProjectionMatrix struct exists)
- [ ] `thiserror` crate available

---

## Scope

### In Scope

- Create `ProjectionError` enum with all failure variants
- Implement `std::error::Error` via thiserror
- Define error codes matching Constitution pattern

### Out of Scope

- Error handling logic (Logic Layer)
- Recovery strategies (Surface Layer)
- Logging (handled by callers)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/error.rs

use std::path::PathBuf;
use thiserror::Error;

/// Errors for sparse projection matrix operations.
///
/// # Constitution Compliance
/// - NO fallback allowed for any error (AP-007)
/// - Missing weights MUST cause hard failure
/// - GPU unavailable MUST cause hard failure
///
/// # Error Codes
/// - EMB-E001: Matrix file missing
/// - EMB-E002: Dimension mismatch
/// - EMB-E003: GPU operation failed
/// - EMB-E004: Checksum validation failed
/// - EMB-E005: Sparse input invalid
#[derive(Error, Debug)]
pub enum ProjectionError {
    /// Projection matrix file not found at expected path.
    ///
    /// CRITICAL: Hash fallback is FORBIDDEN.
    /// System MUST fail if weights are missing.
    #[error("[EMB-E001] MATRIX_MISSING: Projection matrix not found at {path}")]
    MatrixMissing {
        path: PathBuf,
    },

    /// Loaded matrix has wrong dimensions.
    ///
    /// Expected: [30522, 1536]
    /// This indicates corrupted weights or wrong model version.
    #[error("[EMB-E002] DIMENSION_MISMATCH: Expected [30522, 1536], got [{actual_rows}, {actual_cols}] at {path}")]
    DimensionMismatch {
        path: PathBuf,
        actual_rows: usize,
        actual_cols: usize,
    },

    /// GPU operation failed.
    ///
    /// Could be: CUDA unavailable, OOM, kernel failure.
    /// CPU fallback is FORBIDDEN.
    #[error("[EMB-E003] GPU_ERROR: {operation} failed: {details}")]
    GpuError {
        operation: String,
        details: String,
    },

    /// Weight file checksum mismatch.
    ///
    /// Indicates corrupted download or tampering.
    #[error("[EMB-E004] CHECKSUM_MISMATCH: Expected {expected}, got {actual}")]
    ChecksumMismatch {
        expected: String,
        actual: String,
    },

    /// Input sparse vector is invalid.
    ///
    /// Could be: empty, wrong dimension, invalid indices.
    #[error("[EMB-E005] INVALID_SPARSE: {reason}")]
    InvalidSparse {
        reason: String,
    },

    /// SafeTensors parsing failed.
    #[error("[EMB-E006] PARSE_ERROR: Failed to parse SafeTensors: {details}")]
    ParseError {
        details: String,
    },

    /// IO error reading weight file.
    #[error("[EMB-E007] IO_ERROR: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },
}

impl ProjectionError {
    /// Get numeric error code for logging/metrics.
    pub fn code(&self) -> u32 {
        match self {
            Self::MatrixMissing { .. } => 1,
            Self::DimensionMismatch { .. } => 2,
            Self::GpuError { .. } => 3,
            Self::ChecksumMismatch { .. } => 4,
            Self::InvalidSparse { .. } => 5,
            Self::ParseError { .. } => 6,
            Self::IoError { .. } => 7,
        }
    }

    /// Check if error is recoverable (spoiler: none are).
    ///
    /// Constitution AP-007 forbids fallback for projection errors.
    pub fn is_recoverable(&self) -> bool {
        false  // ALL projection errors are fatal
    }

    /// Get error category for metrics.
    pub fn category(&self) -> &'static str {
        match self {
            Self::MatrixMissing { .. } => "configuration",
            Self::DimensionMismatch { .. } => "data_corruption",
            Self::GpuError { .. } => "hardware",
            Self::ChecksumMismatch { .. } => "integrity",
            Self::InvalidSparse { .. } => "input_validation",
            Self::ParseError { .. } => "data_corruption",
            Self::IoError { .. } => "io",
        }
    }
}
```

### Constraints

- MUST use `thiserror` for derive
- Error codes MUST match Constitution `mcp.errors` pattern
- `is_recoverable()` MUST return `false` for all variants
- Error messages MUST include error code prefix

### Verification

- All 7 error variants compile
- Error codes are unique and sequential
- `is_recoverable()` returns false for all

---

## Files to Create

| File Path | Description |
|-----------|-------------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/error.rs` | ProjectionError enum |

## Files to Modify

| File Path | Change |
|-----------|--------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/mod.rs` | Add `pub mod error; pub use error::ProjectionError;` |
| `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` | Use `super::ProjectionError` |

---

## Validation Criteria

- [ ] `ProjectionError` enum with 7 variants
- [ ] All variants have `[EMB-E0XX]` error code prefix
- [ ] `code()`, `is_recoverable()`, `category()` methods implemented
- [ ] `#[from]` for std::io::Error
- [ ] `cargo check -p context-graph-embeddings` passes

---

## Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
cargo test -p context-graph-embeddings error:: -- --nocapture
```

---

## Pseudo Code

```
error.rs:
  Import thiserror::Error
  Import std::path::PathBuf

  Define ProjectionError enum:
    MatrixMissing { path }
    DimensionMismatch { path, actual_rows, actual_cols }
    GpuError { operation, details }
    ChecksumMismatch { expected, actual }
    InvalidSparse { reason }
    ParseError { details }
    IoError { source }

  Impl ProjectionError:
    fn code() -> u32  // 1-7
    fn is_recoverable() -> bool  // always false
    fn category() -> &'static str

mod.rs:
  Add pub mod error;
  Add pub use error::ProjectionError;
```

</task_spec>
