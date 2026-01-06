# TASK-EMB-006: Create WarmLoadResult Struct

<task_spec id="TASK-EMB-006" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Title** | Create WarmLoadResult Struct |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 6 |
| **Implements** | REQ-EMB-003 |
| **Depends On** | TASK-EMB-001 |
| **Estimated Complexity** | medium |
| **Parallel Group** | B |

---

## Context

TECH-EMB-002 specifies real model warm loading with:
- Actual GPU VRAM allocation (verified via nvidia-smi)
- Real weight checksums (not 0xDEAD_BEEF_CAFE_BABE)
- Validated model inference outputs

The current implementation uses `simulate_weight_loading()` which returns fake data. This task creates the result types for real loading.

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Current warm loader | `crates/context-graph-embeddings/src/models/pretrained/warm/loader/operations.rs` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-002-warm-loading.md` |
| Constitution | `docs2/constitution.yaml` sections `stack.gpu`, `AP-007` |

---

## Prerequisites

- [ ] TASK-EMB-001 completed (dimension constants exist)
- [ ] Understand current `simulate_weight_loading()` behavior
- [ ] Read AP-007 prohibition on stub data

---

## Scope

### In Scope

- `WarmLoadResult` struct with real validation data
- `VramAllocation` struct for GPU memory tracking
- `ModelChecksum` type for integrity verification
- `InferenceValidation` struct for test output verification
- Error types for loading failures

### Out of Scope

- Actual loading implementation (TASK-EMB-013)
- VRAM allocation logic (TASK-EMB-014)
- Inference execution (TASK-EMB-015)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-embeddings/src/models/pretrained/warm/types.rs

use std::path::PathBuf;
use chrono::{DateTime, Utc};

/// Result of warming up a model (loading weights to GPU).
///
/// # Constitution Compliance
/// - Real VRAM allocation (nvidia-smi verifiable)
/// - Real checksums (NOT 0xDEAD_BEEF_CAFE_BABE)
/// - Real inference outputs (NOT sin wave fake data)
///
/// # CRITICAL
/// AP-007 forbids stub data in production.
/// If any field contains simulated data, this is a Constitution violation.
#[derive(Debug, Clone)]
pub struct WarmLoadResult {
    /// Which model was loaded
    pub model_id: crate::ModelId,

    /// VRAM allocation details
    pub vram: VramAllocation,

    /// Weight file checksums for all loaded files
    pub checksums: Vec<ModelChecksum>,

    /// Inference validation result
    pub inference_validation: InferenceValidation,

    /// Total load time
    pub load_duration: std::time::Duration,

    /// Timestamp of successful load
    pub loaded_at: DateTime<Utc>,

    /// GPU device index used
    pub device_index: u32,
}

impl WarmLoadResult {
    /// Verify this result contains real (not simulated) data.
    ///
    /// Returns false if any indicator of simulation is detected.
    pub fn is_real(&self) -> bool {
        // Check for known fake checksums
        let has_fake_checksum = self.checksums.iter().any(|c| {
            c.sha256.starts_with(&[0xDE, 0xAD, 0xBE, 0xEF])
        });

        // Check for fake VRAM pointer
        let has_fake_vram = self.vram.base_ptr == 0x7f80_0000_0000u64;

        // Check for sin wave output pattern
        let has_fake_output = self.inference_validation.sample_output
            .windows(2)
            .all(|w| (w[1] - w[0]).abs() < 0.01);

        !has_fake_checksum && !has_fake_vram && !has_fake_output
    }

    /// Panic if this result contains simulated data.
    ///
    /// Call this in production to enforce Constitution AP-007.
    pub fn assert_real(&self) {
        if !self.is_real() {
            panic!(
                "[EMB-E010] SIMULATION_DETECTED: WarmLoadResult contains fake data. \
                 Constitution AP-007 violation. Model: {:?}",
                self.model_id
            );
        }
    }
}

/// GPU VRAM allocation tracking.
///
/// CRITICAL: All values MUST come from real CUDA API calls.
/// Fake values (like 0x7f80_0000_0000) are FORBIDDEN.
#[derive(Debug, Clone)]
pub struct VramAllocation {
    /// Base pointer on GPU (from cudaMalloc)
    /// MUST NOT be 0x7f80_0000_0000 or similar fake value
    pub base_ptr: u64,

    /// Total bytes allocated
    pub size_bytes: usize,

    /// VRAM used before loading (from cudaMemGetInfo)
    pub vram_before_mb: u64,

    /// VRAM used after loading (from cudaMemGetInfo)
    pub vram_after_mb: u64,

    /// Actual delta (should match size_bytes approximately)
    pub vram_delta_mb: u64,
}

impl VramAllocation {
    /// Check if allocation looks real.
    pub fn is_real(&self) -> bool {
        // Fake pointer check
        if self.base_ptr == 0x7f80_0000_0000u64 {
            return false;
        }

        // Zero allocation is suspicious
        if self.size_bytes == 0 {
            return false;
        }

        // VRAM delta should roughly match size
        let expected_delta_mb = (self.size_bytes / (1024 * 1024)) as u64;
        let delta_diff = (self.vram_delta_mb as i64 - expected_delta_mb as i64).abs();

        // Allow 50MB tolerance for GPU overhead
        delta_diff < 50
    }

    /// Get VRAM delta as human-readable string.
    pub fn delta_display(&self) -> String {
        format!("{} MB ({} -> {} MB)",
            self.vram_delta_mb,
            self.vram_before_mb,
            self.vram_after_mb
        )
    }
}

/// Weight file checksum for integrity verification.
#[derive(Debug, Clone)]
pub struct ModelChecksum {
    /// Path to weight file
    pub file_path: PathBuf,

    /// SHA256 hash of file contents
    /// MUST NOT be 0xDEAD_BEEF_CAFE_BABE or similar
    pub sha256: [u8; 32],

    /// File size in bytes
    pub file_size: usize,
}

impl ModelChecksum {
    /// Get checksum as hex string.
    pub fn hex(&self) -> String {
        hex::encode(&self.sha256)
    }

    /// Check if checksum looks real (not a magic constant).
    pub fn is_real(&self) -> bool {
        // Check for known fake patterns
        let fake_patterns: [[u8; 4]; 3] = [
            [0xDE, 0xAD, 0xBE, 0xEF],
            [0xCA, 0xFE, 0xBA, 0xBE],
            [0x00, 0x00, 0x00, 0x00],
        ];

        !fake_patterns.iter().any(|p| self.sha256.starts_with(p))
    }
}

/// Inference validation result.
///
/// Used to verify model produces meaningful output, not random/fake data.
#[derive(Debug, Clone)]
pub struct InferenceValidation {
    /// Sample input used for validation
    pub sample_input: String,

    /// Sample output (embedding vector)
    /// MUST NOT be sin wave pattern: (i * 0.001).sin()
    pub sample_output: Vec<f32>,

    /// L2 norm of output (should be ~1.0 for normalized)
    pub output_norm: f32,

    /// Inference latency
    pub latency: std::time::Duration,

    /// Whether output matches golden reference
    pub matches_golden: bool,

    /// Cosine similarity to golden reference
    pub golden_similarity: f32,
}

impl InferenceValidation {
    /// Check if output looks like real inference (not fake pattern).
    pub fn is_real(&self) -> bool {
        // Sin wave pattern detection
        let is_sin_wave = self.sample_output.windows(10).all(|w| {
            let diffs: Vec<f32> = w.windows(2).map(|p| (p[1] - p[0]).abs()).collect();
            let variance = diffs.iter().map(|d| d * d).sum::<f32>() / diffs.len() as f32;
            variance < 0.0001  // Suspiciously smooth
        });

        // All zeros detection
        let is_zeros = self.sample_output.iter().all(|&v| v.abs() < 1e-6);

        // Check golden similarity (should be high for real model)
        let has_good_golden = self.golden_similarity > 0.95;

        !is_sin_wave && !is_zeros && has_good_golden
    }

    /// Panic if output looks fake.
    pub fn assert_real(&self) {
        if !self.is_real() {
            panic!(
                "[EMB-E011] FAKE_INFERENCE: Output pattern indicates simulation. \
                 Golden similarity: {:.4}. Constitution AP-007 violation.",
                self.golden_similarity
            );
        }
    }
}

/// Error during warm loading.
#[derive(Debug, thiserror::Error)]
pub enum WarmLoadError {
    #[error("[EMB-E008] WEIGHTS_MISSING: Weight file not found at {path}")]
    WeightsMissing { path: PathBuf },

    #[error("[EMB-E009] VRAM_INSUFFICIENT: Need {required_mb}MB, only {available_mb}MB available")]
    InsufficientVram { required_mb: u64, available_mb: u64 },

    #[error("[EMB-E010] SIMULATION_DETECTED: {details}")]
    SimulationDetected { details: String },

    #[error("[EMB-E011] INFERENCE_FAILED: {details}")]
    InferenceFailed { details: String },

    #[error("[EMB-E012] CUDA_ERROR: {operation} failed: {details}")]
    CudaError { operation: String, details: String },
}
```

### Constraints

- All result types MUST have `is_real()` validation methods
- Known fake patterns MUST be detected and rejected
- Error codes MUST match Constitution `mcp.errors` pattern

### Verification

- `is_real()` returns false for known fake patterns
- `assert_real()` panics on simulation detection
- Error messages include Constitution error codes

---

## Files to Create

| File Path | Description |
|-----------|-------------|
| `crates/context-graph-embeddings/src/models/pretrained/warm/types.rs` | Warm loading result types |

## Files to Modify

| File Path | Change |
|-----------|--------|
| `crates/context-graph-embeddings/src/models/pretrained/warm/mod.rs` | Add `pub mod types;` |

---

## Validation Criteria

- [ ] `WarmLoadResult` struct with all validation fields
- [ ] `VramAllocation` with real/fake detection
- [ ] `ModelChecksum` with hex display
- [ ] `InferenceValidation` with golden comparison
- [ ] `WarmLoadError` enum with error codes
- [ ] All `is_real()` methods detect known fake patterns
- [ ] `cargo check -p context-graph-embeddings` passes

---

## Test Commands

```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings
cargo test -p context-graph-embeddings warm::types:: -- --nocapture
```

---

## Pseudo Code

```
types.rs:
  Define WarmLoadResult struct
    model_id, vram, checksums, inference_validation, timing
    is_real() checks all components
    assert_real() panics if simulated

  Define VramAllocation struct
    base_ptr, size_bytes, vram_before/after/delta
    is_real() checks for fake pointer pattern

  Define ModelChecksum struct
    file_path, sha256, file_size
    hex() for display
    is_real() checks for magic bytes

  Define InferenceValidation struct
    sample_input/output, norms, golden comparison
    is_real() checks for sin wave pattern

  Define WarmLoadError enum with Constitution error codes
```

</task_spec>
