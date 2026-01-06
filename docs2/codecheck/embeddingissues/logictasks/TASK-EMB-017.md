# TASK-EMB-007: Create Consolidated Errors

<task_spec id="TASK-EMB-007" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Title** | Create Consolidated Errors |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 7 |
| **Implements** | All requirements |
| **Depends On** | TASK-EMB-003, TASK-EMB-005, TASK-EMB-006 |
| **Estimated Complexity** | medium |
| **Parallel Group** | D |

---

## Context

Multiple Foundation tasks create domain-specific error types:
- TASK-EMB-003: `ProjectionError`
- TASK-EMB-005: Storage errors (implicit)
- TASK-EMB-006: `WarmLoadError`

This task consolidates these into a unified `EmbeddingPipelineError` that follows Constitution `mcp.errors` conventions and enables proper error propagation.

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| ProjectionError | `crates/context-graph-embeddings/src/models/pretrained/sparse/error.rs` (TASK-EMB-003) |
| WarmLoadError | `crates/context-graph-embeddings/src/models/pretrained/warm/types.rs` (TASK-EMB-006) |
| Constitution | `docs2/constitution.yaml` section `mcp.errors` |

---

## Prerequisites

- [ ] TASK-EMB-003 completed (ProjectionError exists)
- [ ] TASK-EMB-005 completed (Storage types exist)
- [ ] TASK-EMB-006 completed (WarmLoadError exists)

---

## Scope

### In Scope

- Unified `EmbeddingPipelineError` enum
- MCP error code mapping per Constitution
- `From` implementations for wrapped errors
- Error categorization for metrics
- Recovery hints (even though AP-007 forbids recovery)

### Out of Scope

- Error handling logic (Logic/Surface Layer)
- Logging infrastructure
- Metrics collection

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-embeddings/src/error.rs

use std::path::PathBuf;
use thiserror::Error;

use crate::models::pretrained::sparse::ProjectionError;
use crate::models::pretrained::warm::types::WarmLoadError;

/// Unified error type for the embedding pipeline.
///
/// # Constitution Compliance
/// - Error codes match `mcp.errors` section
/// - No recovery fallback (AP-007)
/// - All errors are fatal for embedding operations
///
/// # Error Code Ranges
/// - EMB-E001 to EMB-E007: Projection errors
/// - EMB-E008 to EMB-E012: Warm loading errors
/// - EMB-E020 to EMB-E029: Quantization errors
/// - EMB-E030 to EMB-E039: Storage errors
/// - EMB-E040 to EMB-E049: Inference errors
/// - EMB-E050 to EMB-E059: Pipeline errors
#[derive(Error, Debug)]
pub enum EmbeddingPipelineError {
    // =========== Projection Errors (EMB-E001 to EMB-E007) ===========

    /// Projection matrix errors
    #[error(transparent)]
    Projection(#[from] ProjectionError),

    // =========== Warm Loading Errors (EMB-E008 to EMB-E012) ===========

    /// Model warm loading errors
    #[error(transparent)]
    WarmLoad(#[from] WarmLoadError),

    // =========== Quantization Errors (EMB-E020 to EMB-E029) ===========

    /// Quantization encoding failed
    #[error("[EMB-E020] QUANTIZE_FAILED: Failed to quantize {embedder_id}: {details}")]
    QuantizationFailed {
        embedder_id: usize,
        details: String,
    },

    /// Dequantization failed
    #[error("[EMB-E021] DEQUANTIZE_FAILED: Failed to dequantize {embedder_id}: {details}")]
    DequantizationFailed {
        embedder_id: usize,
        details: String,
    },

    /// Codebook not found for PQ
    #[error("[EMB-E022] CODEBOOK_MISSING: PQ codebook {codebook_id} not found")]
    CodebookMissing {
        codebook_id: u32,
    },

    // =========== Storage Errors (EMB-E030 to EMB-E039) ===========

    /// Storage backend unavailable
    #[error("[EMB-E030] STORAGE_UNAVAILABLE: {backend} connection failed: {details}")]
    StorageUnavailable {
        backend: String,
        details: String,
    },

    /// Fingerprint not found
    #[error("[EMB-E031] FINGERPRINT_NOT_FOUND: Memory {id} not in storage")]
    FingerprintNotFound {
        id: uuid::Uuid,
    },

    /// Index query failed
    #[error("[EMB-E032] INDEX_QUERY_FAILED: Query on {index_type} failed: {details}")]
    IndexQueryFailed {
        index_type: String,
        details: String,
    },

    /// Storage size budget exceeded
    #[error("[EMB-E033] SIZE_BUDGET_EXCEEDED: Fingerprint is {actual_kb}KB, budget is 17KB")]
    SizeBudgetExceeded {
        actual_kb: usize,
    },

    // =========== Inference Errors (EMB-E040 to EMB-E049) ===========

    /// Model not loaded
    #[error("[EMB-E040] MODEL_NOT_LOADED: {model_id:?} must be warmed before inference")]
    ModelNotLoaded {
        model_id: crate::ModelId,
    },

    /// Inference timeout
    #[error("[EMB-E041] INFERENCE_TIMEOUT: {model_id:?} took {elapsed_ms}ms, budget was {budget_ms}ms")]
    InferenceTimeout {
        model_id: crate::ModelId,
        elapsed_ms: u64,
        budget_ms: u64,
    },

    /// Batch size exceeded
    #[error("[EMB-E042] BATCH_TOO_LARGE: Requested {requested}, max is {max}")]
    BatchTooLarge {
        requested: usize,
        max: usize,
    },

    // =========== Pipeline Errors (EMB-E050 to EMB-E059) ===========

    /// Pipeline stage failed
    #[error("[EMB-E050] STAGE_FAILED: Stage {stage} failed: {details}")]
    StageFailed {
        stage: String,
        details: String,
    },

    /// Constitution violation detected
    #[error("[EMB-E051] CONSTITUTION_VIOLATION: {rule} violated: {details}")]
    ConstitutionViolation {
        rule: String,
        details: String,
    },

    /// CUDA not available
    #[error("[EMB-E052] CUDA_REQUIRED: GPU required but not available")]
    CudaRequired,

    /// Configuration error
    #[error("[EMB-E053] CONFIG_ERROR: {details}")]
    ConfigError {
        details: String,
    },
}

impl EmbeddingPipelineError {
    /// Get MCP error code (Constitution mcp.errors format).
    pub fn mcp_code(&self) -> i32 {
        match self {
            // Map to Constitution error code ranges
            Self::Projection(_) => -32003,  // CausalInferenceError (closest)
            Self::WarmLoad(_) => -32002,    // StorageError (loading)
            Self::QuantizationFailed { .. } => -32003,
            Self::DequantizationFailed { .. } => -32003,
            Self::CodebookMissing { .. } => -32002,
            Self::StorageUnavailable { .. } => -32002,
            Self::FingerprintNotFound { .. } => -32000,  // SessionNotFound
            Self::IndexQueryFailed { .. } => -32001,     // GraphQueryError
            Self::SizeBudgetExceeded { .. } => -32002,
            Self::ModelNotLoaded { .. } => -32002,
            Self::InferenceTimeout { .. } => -32003,
            Self::BatchTooLarge { .. } => -32602,  // Invalid params
            Self::StageFailed { .. } => -32603,    // Internal error
            Self::ConstitutionViolation { .. } => -32603,
            Self::CudaRequired => -32603,
            Self::ConfigError { .. } => -32602,
        }
    }

    /// Get error category for metrics.
    pub fn category(&self) -> &'static str {
        match self {
            Self::Projection(_) => "projection",
            Self::WarmLoad(_) => "warm_loading",
            Self::QuantizationFailed { .. } | Self::DequantizationFailed { .. } | Self::CodebookMissing { .. } => "quantization",
            Self::StorageUnavailable { .. } | Self::FingerprintNotFound { .. } | Self::IndexQueryFailed { .. } | Self::SizeBudgetExceeded { .. } => "storage",
            Self::ModelNotLoaded { .. } | Self::InferenceTimeout { .. } | Self::BatchTooLarge { .. } => "inference",
            Self::StageFailed { .. } | Self::ConstitutionViolation { .. } | Self::CudaRequired | Self::ConfigError { .. } => "pipeline",
        }
    }

    /// Check if error is recoverable.
    ///
    /// Constitution AP-007: NO embedding errors are recoverable.
    /// Fallback/retry is FORBIDDEN.
    pub fn is_recoverable(&self) -> bool {
        false  // All embedding pipeline errors are fatal
    }

    /// Get recovery hint (informational only - actual recovery forbidden).
    pub fn recovery_hint(&self) -> &'static str {
        match self {
            Self::Projection(_) => "Check projection matrix weights exist and are valid",
            Self::WarmLoad(_) => "Verify GPU availability and weight files",
            Self::QuantizationFailed { .. } => "Check input dimensions match embedder config",
            Self::DequantizationFailed { .. } => "Verify quantized data integrity",
            Self::CodebookMissing { .. } => "Regenerate PQ codebooks",
            Self::StorageUnavailable { .. } => "Check database connectivity",
            Self::FingerprintNotFound { .. } => "Memory may have been deleted",
            Self::IndexQueryFailed { .. } => "Check index health, may need rebuild",
            Self::SizeBudgetExceeded { .. } => "Review quantization strategy",
            Self::ModelNotLoaded { .. } => "Call warm_model() before inference",
            Self::InferenceTimeout { .. } => "Reduce batch size or check GPU load",
            Self::BatchTooLarge { .. } => "Use smaller batches",
            Self::StageFailed { .. } => "Check stage input/output compatibility",
            Self::ConstitutionViolation { .. } => "Review Constitution compliance",
            Self::CudaRequired => "GPU with CUDA 13.1+ required",
            Self::ConfigError { .. } => "Check configuration file syntax",
        }
    }
}

/// Result type alias for embedding pipeline operations.
pub type EmbeddingResult<T> = Result<T, EmbeddingPipelineError>;
```

### Constraints

- MUST include `From` for ProjectionError and WarmLoadError
- Error codes MUST be in designated ranges
- `is_recoverable()` MUST return `false` for all variants
- `mcp_code()` MUST map to Constitution codes

### Verification

- All error variants have unique codes in correct ranges
- `#[from]` attributes compile for wrapped errors
- `mcp_code()` returns valid Constitution codes

---

## Files to Create

| File Path | Description |
|-----------|-------------|
| `crates/context-graph-embeddings/src/error.rs` | Consolidated error types |

## Files to Modify

| File Path | Change |
|-----------|--------|
| `crates/context-graph-embeddings/src/lib.rs` | Add `pub mod error; pub use error::*;` |

---

## Validation Criteria

- [ ] `EmbeddingPipelineError` enum with all variant categories
- [ ] `From<ProjectionError>` and `From<WarmLoadError>` implemented via `#[from]`
- [ ] `mcp_code()` returns Constitution-compliant codes
- [ ] `category()` returns metric-friendly categories
- [ ] `is_recoverable()` returns false for all
- [ ] `recovery_hint()` provides informational messages
- [ ] `EmbeddingResult<T>` type alias defined
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
  Import ProjectionError and WarmLoadError

  Define EmbeddingPipelineError enum:
    Projection(#[from] ProjectionError)
    WarmLoad(#[from] WarmLoadError)
    QuantizationFailed, DequantizationFailed, CodebookMissing
    StorageUnavailable, FingerprintNotFound, IndexQueryFailed, SizeBudgetExceeded
    ModelNotLoaded, InferenceTimeout, BatchTooLarge
    StageFailed, ConstitutionViolation, CudaRequired, ConfigError

  Impl EmbeddingPipelineError:
    mcp_code() -> i32 (Constitution codes)
    category() -> &str (for metrics)
    is_recoverable() -> bool (always false)
    recovery_hint() -> &str (informational)

  Type alias: EmbeddingResult<T> = Result<T, EmbeddingPipelineError>
```

</task_spec>
