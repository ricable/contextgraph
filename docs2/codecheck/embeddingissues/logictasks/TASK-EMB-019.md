# TASK-EMB-009: Create Weight File Spec

<task_spec id="TASK-EMB-009" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Title** | Create Weight File Spec |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 9 |
| **Implements** | REQ-EMB-001 |
| **Depends On** | TASK-EMB-002 |
| **Estimated Complexity** | low |
| **Parallel Group** | C |

---

## Context

TECH-EMB-001 specifies loading a learned projection matrix from SafeTensors format. This task documents the exact file format specification so that:
1. Weight files can be validated before loading
2. Training pipelines produce compatible files
3. Tests can generate valid fixtures

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| ProjectionMatrix | `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` (TASK-EMB-002) |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-001-sparse-projection.md` |
| SafeTensors docs | https://huggingface.co/docs/safetensors |

---

## Prerequisites

- [ ] TASK-EMB-002 completed (ProjectionMatrix constants defined)
- [ ] Read SafeTensors format specification
- [ ] Understand dimension requirements

---

## Scope

### In Scope

- Document weight file format in code
- Create `WeightFileSpec` struct with validation
- Define expected metadata fields
- Create file validation function
- Add integrity checking utilities

### Out of Scope

- Actual weight generation (training process)
- Loading implementation (TASK-EMB-011)
- GPU transfer (TASK-EMB-011)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/weight_spec.rs

use std::path::Path;
use sha2::{Sha256, Digest};
use safetensors::SafeTensors;

use crate::config::constants::{SPARSE_VOCAB_SIZE, SPARSE_PROJECTED_DIMENSION};
use super::projection::{PROJECTION_WEIGHT_FILE, PROJECTION_TENSOR_NAME};

/// Weight file format specification for sparse projection matrix.
///
/// # File Format: SafeTensors
///
/// SafeTensors is a safe, fast, and portable tensor serialization format
/// developed by Hugging Face. It provides:
/// - Memory-mapped loading (fast)
/// - No arbitrary code execution (safe)
/// - Cross-platform compatibility (portable)
///
/// # Required Contents
///
/// ## Tensor: "projection_matrix"
/// - **Shape**: [30522, 1536]
/// - **DType**: F32
/// - **Description**: Learned projection weights from vocab to dense space
///
/// ## Metadata (Optional but Recommended)
/// - `version`: Weight file version (e.g., "1.0.0")
/// - `training_date`: ISO 8601 timestamp
/// - `training_dataset`: Dataset used for training
/// - `training_steps`: Number of training steps
/// - `constitution_version`: Constitution version (e.g., "4.0.0")
///
/// # File Path Convention
///
/// Weights should be stored at:
/// ```
/// {model_dir}/sparse_projection.safetensors
/// ```
///
/// # Example Training Output
///
/// The training pipeline should produce:
/// ```python
/// import torch
/// from safetensors.torch import save_file
///
/// weights = torch.randn(30522, 1536)  # Or trained weights
/// tensors = {"projection_matrix": weights}
/// metadata = {
///     "version": "1.0.0",
///     "constitution_version": "4.0.0",
///     "training_date": "2026-01-06T00:00:00Z",
/// }
/// save_file(tensors, "sparse_projection.safetensors", metadata=metadata)
/// ```
pub struct WeightFileSpec;

impl WeightFileSpec {
    /// Expected file name.
    pub const FILE_NAME: &'static str = PROJECTION_WEIGHT_FILE;

    /// Expected tensor name.
    pub const TENSOR_NAME: &'static str = PROJECTION_TENSOR_NAME;

    /// Expected tensor shape.
    pub const EXPECTED_SHAPE: [usize; 2] = [SPARSE_VOCAB_SIZE, SPARSE_PROJECTED_DIMENSION];

    /// Expected data type (F32).
    pub const EXPECTED_DTYPE: &'static str = "F32";

    /// Minimum expected file size (bytes).
    /// 30522 × 1536 × 4 bytes = ~187MB uncompressed
    pub const MIN_FILE_SIZE: usize = SPARSE_VOCAB_SIZE * SPARSE_PROJECTED_DIMENSION * 4;

    /// Validate a weight file without loading to GPU.
    ///
    /// # Checks
    /// 1. File exists and is readable
    /// 2. File parses as valid SafeTensors
    /// 3. Contains tensor with correct name
    /// 4. Tensor has correct shape [30522, 1536]
    /// 5. Tensor has correct dtype (F32)
    ///
    /// # Returns
    /// `Ok(ValidationResult)` with metadata if valid.
    /// `Err(ValidationError)` with specific failure reason.
    pub fn validate(path: &Path) -> Result<ValidationResult, ValidationError> {
        // Check file exists
        if !path.exists() {
            return Err(ValidationError::FileNotFound {
                path: path.to_path_buf()
            });
        }

        // Read file bytes
        let bytes = std::fs::read(path).map_err(|e| ValidationError::IoError {
            path: path.to_path_buf(),
            details: e.to_string(),
        })?;

        // Check minimum size
        if bytes.len() < Self::MIN_FILE_SIZE {
            return Err(ValidationError::FileTooSmall {
                path: path.to_path_buf(),
                actual: bytes.len(),
                minimum: Self::MIN_FILE_SIZE,
            });
        }

        // Compute SHA256
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let checksum: [u8; 32] = hasher.finalize().into();

        // Parse SafeTensors
        let tensors = SafeTensors::deserialize(&bytes).map_err(|e| {
            ValidationError::ParseError {
                path: path.to_path_buf(),
                details: e.to_string(),
            }
        })?;

        // Get tensor
        let tensor = tensors.tensor(Self::TENSOR_NAME).map_err(|_| {
            ValidationError::TensorMissing {
                path: path.to_path_buf(),
                tensor_name: Self::TENSOR_NAME.to_string(),
            }
        })?;

        // Check shape
        let shape = tensor.shape();
        if shape != Self::EXPECTED_SHAPE {
            return Err(ValidationError::ShapeMismatch {
                path: path.to_path_buf(),
                expected: Self::EXPECTED_SHAPE.to_vec(),
                actual: shape.to_vec(),
            });
        }

        // Check dtype
        let dtype = format!("{:?}", tensor.dtype());
        if !dtype.contains("F32") {
            return Err(ValidationError::DtypeMismatch {
                path: path.to_path_buf(),
                expected: Self::EXPECTED_DTYPE.to_string(),
                actual: dtype,
            });
        }

        // Extract metadata
        let metadata = Self::extract_metadata(&tensors);

        Ok(ValidationResult {
            path: path.to_path_buf(),
            checksum,
            file_size: bytes.len(),
            metadata,
        })
    }

    /// Extract metadata from SafeTensors file.
    fn extract_metadata(tensors: &SafeTensors) -> WeightMetadata {
        // SafeTensors metadata is stored in the header
        // Note: actual API may vary by safetensors version
        WeightMetadata {
            version: None,  // Extract from header if available
            training_date: None,
            training_dataset: None,
            training_steps: None,
            constitution_version: None,
        }
    }

    /// Generate a test fixture file (for testing only).
    ///
    /// # Warning
    /// This generates RANDOM weights, not trained weights.
    /// Only use for testing file format handling.
    #[cfg(test)]
    pub fn generate_test_fixture(output_path: &Path) -> std::io::Result<()> {
        use std::io::Write;

        // This would use safetensors::serialize in practice
        // Simplified for specification purposes
        unimplemented!("Use Python safetensors to generate fixtures")
    }
}

/// Result of weight file validation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Path to validated file.
    pub path: std::path::PathBuf,

    /// SHA256 checksum.
    pub checksum: [u8; 32],

    /// File size in bytes.
    pub file_size: usize,

    /// Extracted metadata.
    pub metadata: WeightMetadata,
}

impl ValidationResult {
    /// Get checksum as hex string.
    pub fn checksum_hex(&self) -> String {
        hex::encode(&self.checksum)
    }

    /// Check if file size is reasonable.
    pub fn size_reasonable(&self) -> bool {
        // Should be close to MIN_FILE_SIZE (slight variation for header)
        let expected = WeightFileSpec::MIN_FILE_SIZE;
        self.file_size >= expected && self.file_size <= expected * 2
    }
}

/// Metadata from weight file.
#[derive(Debug, Clone, Default)]
pub struct WeightMetadata {
    /// Weight file version.
    pub version: Option<String>,

    /// Training date (ISO 8601).
    pub training_date: Option<String>,

    /// Dataset used for training.
    pub training_dataset: Option<String>,

    /// Number of training steps.
    pub training_steps: Option<u64>,

    /// Constitution version weights were trained for.
    pub constitution_version: Option<String>,
}

/// Validation error types.
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Weight file not found: {path}")]
    FileNotFound { path: std::path::PathBuf },

    #[error("IO error reading {path}: {details}")]
    IoError { path: std::path::PathBuf, details: String },

    #[error("File too small: {path} is {actual} bytes, minimum is {minimum}")]
    FileTooSmall { path: std::path::PathBuf, actual: usize, minimum: usize },

    #[error("Failed to parse SafeTensors {path}: {details}")]
    ParseError { path: std::path::PathBuf, details: String },

    #[error("Tensor '{tensor_name}' not found in {path}")]
    TensorMissing { path: std::path::PathBuf, tensor_name: String },

    #[error("Shape mismatch in {path}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { path: std::path::PathBuf, expected: Vec<usize>, actual: Vec<usize> },

    #[error("Dtype mismatch in {path}: expected {expected}, got {actual}")]
    DtypeMismatch { path: std::path::PathBuf, expected: String, actual: String },
}
```

### Constraints

- Shape MUST be exactly [30522, 1536]
- DType MUST be F32
- File MUST be valid SafeTensors format
- Validation MUST work without GPU

### Verification

- `validate()` returns Ok for valid files
- `validate()` returns specific errors for invalid files
- Checksum is computed correctly

---

## Files to Create

| File Path | Description |
|-----------|-------------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/weight_spec.rs` | Weight file specification and validation |

## Files to Modify

| File Path | Change |
|-----------|--------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/mod.rs` | Add `pub mod weight_spec;` |

---

## Validation Criteria

- [ ] `WeightFileSpec` struct with all constants
- [ ] `validate()` function with 5+ validation checks
- [ ] `ValidationResult` with checksum and metadata
- [ ] `WeightMetadata` for training provenance
- [ ] `ValidationError` enum with specific error types
- [ ] `cargo check -p context-graph-embeddings` passes

---

## Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
cargo test -p context-graph-embeddings weight_spec:: -- --nocapture
```

---

## Pseudo Code

```
weight_spec.rs:
  Define WeightFileSpec struct with constants:
    FILE_NAME = "sparse_projection.safetensors"
    TENSOR_NAME = "projection_matrix"
    EXPECTED_SHAPE = [30522, 1536]
    EXPECTED_DTYPE = "F32"
    MIN_FILE_SIZE = ~187MB

  Impl WeightFileSpec:
    validate(path) -> Result<ValidationResult, ValidationError>
      Check file exists
      Read bytes
      Check minimum size
      Compute SHA256 checksum
      Parse SafeTensors
      Get tensor by name
      Validate shape
      Validate dtype
      Extract metadata
      Return ValidationResult

  Define ValidationResult:
    path, checksum, file_size, metadata
    checksum_hex() method

  Define WeightMetadata:
    version, training_date, training_dataset, training_steps, constitution_version

  Define ValidationError enum:
    FileNotFound, IoError, FileTooSmall, ParseError, TensorMissing, ShapeMismatch, DtypeMismatch
```

</task_spec>
