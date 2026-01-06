# TASK-EMB-002: Create ProjectionMatrix Struct

<task_spec id="TASK-EMB-002" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Title** | Create ProjectionMatrix Struct |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 2 |
| **Implements** | REQ-EMB-001 |
| **Depends On** | TASK-EMB-001 |
| **Estimated Complexity** | medium |
| **Parallel Group** | B |

---

## Context

TECH-EMB-001 specifies a learned projection matrix for sparse-to-dense transformation. This replaces the broken hash-based projection (`idx % projected_dim`) with a real neural network weight matrix loaded from SafeTensors.

The current implementation uses hash modulo which DESTROYS semantic information. Constitution AP-007 forbids stub data in prod.

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Dimension constants | `crates/context-graph-core/src/config/constants.rs` (TASK-EMB-001) |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-001-sparse-projection.md` |
| Constitution | `docs2/constitution.yaml` section `embeddings.E6_Sparse` |

---

## Prerequisites

- [ ] TASK-EMB-001 completed (dimension constants exist)
- [ ] `candle-core` crate with CUDA feature available
- [ ] Rust 1.75+ with 2021 edition

---

## Scope

### In Scope

- Create `ProjectionMatrix` struct with GPU tensor storage
- Define fields: weights tensor, device, checksum
- Create associated constants for file paths
- Placeholder for `load()` and `project()` methods (implemented in Logic Layer)

### Out of Scope

- `load()` implementation (TASK-EMB-011)
- `project()` implementation (TASK-EMB-012)
- Error types (TASK-EMB-003)
- Weight file creation (separate training process)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs

use candle_core::{Device, Tensor};
use crate::config::constants::{SPARSE_VOCAB_SIZE, SPARSE_PROJECTED_DIMENSION};

/// Weight file name (SafeTensors format)
pub const PROJECTION_WEIGHT_FILE: &str = "sparse_projection.safetensors";

/// Tensor name within SafeTensors file
pub const PROJECTION_TENSOR_NAME: &str = "projection_matrix";

/// Learned projection matrix for sparse-to-dense transformation.
///
/// # Constitution Compliance
/// - Implements E6_Sparse learned projection per TECH-EMB-001
/// - Replaces forbidden hash modulo approach (AP-007 violation)
/// - Loaded from SafeTensors file, NOT generated at runtime
///
/// # Shape
/// - `weights`: [SPARSE_VOCAB_SIZE, SPARSE_PROJECTED_DIMENSION] = [30522, 1536]
///
/// # GPU Requirement
/// - Matrix stored on CUDA device
/// - CPU fallback is FORBIDDEN
#[derive(Debug)]
pub struct ProjectionMatrix {
    /// Weight tensor on GPU: [30522, 1536]
    /// Loaded from SafeTensors, NOT randomly initialized
    pub weights: Tensor,

    /// CUDA device holding the weights
    /// MUST be Device::Cuda, NOT Device::Cpu
    pub device: Device,

    /// SHA256 checksum of weight file
    /// Used for cache invalidation and integrity verification
    /// MUST be real checksum, NOT 0xDEAD_BEEF_CAFE_BABE
    pub weight_checksum: [u8; 32],
}

impl ProjectionMatrix {
    /// Load projection matrix from SafeTensors file.
    /// Implementation in TASK-EMB-011.
    pub fn load(model_dir: &std::path::Path) -> Result<Self, super::ProjectionError>;

    /// Project sparse vector to dense using learned weights.
    /// Implementation in TASK-EMB-012.
    pub fn project(&self, sparse: &super::SparseVector) -> Result<Vec<f32>, super::ProjectionError>;

    /// Batch project multiple sparse vectors.
    /// Implementation in TASK-EMB-012.
    pub fn project_batch(&self, batch: &[super::SparseVector]) -> Result<Vec<Vec<f32>>, super::ProjectionError>;

    /// Get weight matrix dimensions for validation.
    pub fn dimensions(&self) -> (usize, usize) {
        (SPARSE_VOCAB_SIZE, SPARSE_PROJECTED_DIMENSION)
    }

    /// Verify matrix is on correct device.
    pub fn verify_device(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }

    /// Get checksum as hex string for logging.
    pub fn checksum_hex(&self) -> String {
        hex::encode(&self.weight_checksum)
    }
}
```

### Constraints

- `weights` tensor MUST be on CUDA device
- `device` MUST be `Device::Cuda`, not `Device::Cpu`
- `weight_checksum` MUST be real SHA256, NOT magic bytes
- Shape MUST be `[30522, 1536]`

### Verification

- Struct compiles with candle-core CUDA feature
- Associated constants have correct values
- Method signatures match specification

---

## Files to Create

| File Path | Description |
|-----------|-------------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs` | ProjectionMatrix struct and constants |

## Files to Modify

| File Path | Change |
|-----------|--------|
| `crates/context-graph-embeddings/src/models/pretrained/sparse/mod.rs` | Add `pub mod projection;` |
| `crates/context-graph-embeddings/Cargo.toml` | Add `hex` crate for checksum display |

---

## Validation Criteria

- [ ] `ProjectionMatrix` struct with 3 fields defined
- [ ] Constants `PROJECTION_WEIGHT_FILE` and `PROJECTION_TENSOR_NAME` defined
- [ ] Method signatures for `load()`, `project()`, `project_batch()` declared
- [ ] Helper methods `dimensions()`, `verify_device()`, `checksum_hex()` implemented
- [ ] Uses dimension constants from TASK-EMB-001
- [ ] `cargo check -p context-graph-embeddings --features cuda` passes

---

## Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings --features cuda
cargo doc -p context-graph-embeddings --no-deps
```

---

## Pseudo Code

```
projection.rs:
  Import candle_core::{Device, Tensor}
  Import dimension constants from config

  Define PROJECTION_WEIGHT_FILE = "sparse_projection.safetensors"
  Define PROJECTION_TENSOR_NAME = "projection_matrix"

  Define ProjectionMatrix struct:
    weights: Tensor (GPU)
    device: Device (CUDA only)
    weight_checksum: [u8; 32] (SHA256)

  Impl ProjectionMatrix:
    fn load(...) -> Result<Self, ProjectionError>  // placeholder
    fn project(...) -> Result<Vec<f32>, ProjectionError>  // placeholder
    fn project_batch(...) -> Result<Vec<Vec<f32>>, ProjectionError>  // placeholder
    fn dimensions() -> (usize, usize)  // returns (30522, 1536)
    fn verify_device() -> bool  // checks Device::Cuda
    fn checksum_hex() -> String  // hex encode checksum
```

</task_spec>
