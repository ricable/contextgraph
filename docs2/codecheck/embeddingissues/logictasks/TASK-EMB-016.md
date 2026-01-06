# TASK-EMB-016: Enhance WarmLoadResult with VramAllocation and InferenceValidation

<task_spec id="TASK-EMB-016" version="2.0">

## Metadata

| Field | Value |
|-------|-------|
| **Title** | Enhance WarmLoadResult with VRAM Tracking and Inference Validation |
| **Status** | completed |
| **Layer** | foundation |
| **Sequence** | 16 |
| **Implements** | REQ-EMB-003, REQ-WARM-003, REQ-WARM-005 |
| **Depends On** | TASK-EMB-001, TASK-EMB-013, TASK-EMB-014, TASK-EMB-015 |
| **Estimated Complexity** | medium |
| **Parallel Group** | C |

---

## AI Agent Context (Fresh Context Window)

### Project Overview

This task is part of the `contextgraph` project - a Rust-based embedding system targeting RTX 5090 GPUs with 32GB VRAM and CUDA 13.1. The project follows a strict Constitution (AP-007) that forbids stub/mock data in production.

### Current Codebase State (Audited 2026-01-06)

The warm loading module exists at:
```
crates/context-graph-embeddings/src/warm/
├── mod.rs              # Module declarations
├── error.rs            # WarmError enum with exit codes 101-115
├── loader/
│   ├── mod.rs          # Loader submodule declarations
│   ├── types.rs        # TensorMetadata, WarmLoadResult, LoadedModelWeights (IMPLEMENTED)
│   ├── operations.rs   # Weight loading operations
│   └── tests/          # Test directory
├── inference/
│   ├── mod.rs          # Inference submodule declarations
│   └── engine.rs       # InferenceEngine struct (IMPLEMENTED)
├── validation/         # Validation submodule
├── health/             # Health check submodule
├── registry/           # Model registry submodule
├── diagnostics/        # Diagnostic submodule
├── memory_pool/        # Memory pool submodule
└── cuda_alloc/         # CUDA allocation submodule
```

### Already Implemented (DO NOT RECREATE)

The following types already exist in `crates/context-graph-embeddings/src/warm/loader/types.rs`:

1. **`TensorMetadata`** - Metadata from SafeTensors headers (shapes, dtype, total_params)
2. **`WarmLoadResult`** - Result of loading weights (gpu_ptr, checksum, size_bytes, load_duration, tensor_metadata)
3. **`LoadedModelWeights`** - Complete model weights loaded to GPU (model_id, tensors HashMap, file_checksum, total_gpu_bytes, device_id, loaded_at)

All have fail-fast validation with Constitution AP-007 compliance.

### What This Task Adds

This task extends the existing types with:
1. **`VramAllocation`** - Detailed GPU memory tracking (before/after/delta)
2. **`InferenceValidation`** - Golden reference comparison for inference output validation
3. Enhanced `is_real()` detection methods for simulation/fake data

---

## Constitution Requirements (MANDATORY)

Read `/home/cabdru/contextgraph/docs2/constitution.yaml` before implementation.

### AP-007: No Stub Data in Production

```yaml
AP-007:
  title: "No Stub Data in Production"
  description: "All data must be real. No mock, fake, or simulated values."
  enforcement: PANIC on detection
```

**Forbidden Patterns (MUST DETECT AND REJECT):**
- GPU pointer `0x7f80_0000_0000` (common fake value)
- Checksum bytes starting with `[0xDE, 0xAD, 0xBE, 0xEF]` or `[0xCA, 0xFE, 0xBA, 0xBE]`
- All-zero checksums `[0x00; 32]`
- Sin wave inference patterns: `(i * 0.001).sin()`
- Zero-size allocations

### Hardware Target

```yaml
stack:
  gpu:
    model: "RTX 5090"
    vram: "32GB"
    cuda: "13.1"
    compute_capability: "12.0"
```

### Error Code Pattern

All errors must follow Constitution `mcp.errors` pattern:
- Format: `[EMB-EXXX] CATEGORY: message`
- Example: `[EMB-E010] SIMULATION_DETECTED: WarmLoadResult contains fake data`

---

## Full State Verification Requirements

### Source of Truth

| Component | Source File | Verification Method |
|-----------|-------------|---------------------|
| Existing types | `crates/context-graph-embeddings/src/warm/loader/types.rs` | `cargo check -p context-graph-embeddings` |
| Error types | `crates/context-graph-embeddings/src/warm/error.rs` | Grep for `WarmError::` variants |
| GPU tensor type | `crates/context-graph-embeddings/src/gpu/mod.rs` | `use crate::gpu::GpuTensor` |
| Constitution | `docs2/constitution.yaml` | Manual read, parse YAML |

### Execute & Inspect Protocol

Before writing any code, execute these commands:

```bash
# 1. Verify current state compiles
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings

# 2. List existing types in loader/types.rs
grep -n "^pub struct\|^pub enum" crates/context-graph-embeddings/src/warm/loader/types.rs

# 3. Verify GpuTensor import path
grep -rn "pub struct GpuTensor" crates/context-graph-embeddings/src/

# 4. Check existing error codes in use
grep -n "EMB-E0" crates/context-graph-embeddings/src/warm/
```

### Boundary & Edge Case Audit (3 Required)

**Edge Case 1: VRAM Delta Mismatch**
- Scenario: GPU reports delta that differs from allocation size by >50MB
- Expected: `VramAllocation::is_real()` returns `false`
- Test: Create allocation with `vram_delta_mb = 1000`, `size_bytes = 1024` (1KB vs 1000MB)

**Edge Case 2: Suspiciously Smooth Inference Output**
- Scenario: Inference output follows perfect mathematical pattern (sin wave)
- Expected: `InferenceValidation::is_real()` returns `false`
- Test: Generate vector with `(0..768).map(|i| (i as f32 * 0.001).sin())`

**Edge Case 3: Golden Similarity Below Threshold**
- Scenario: Inference output has low cosine similarity to golden reference (<0.95)
- Expected: `InferenceValidation::is_real()` returns `false`
- Test: Set `golden_similarity = 0.5`

### Evidence of Success Logs

After implementation, these log lines MUST appear:

```
[INFO] VramAllocation::is_real() validation: passed
[INFO] InferenceValidation::is_real() validation: passed
[INFO] All fake pattern detections working correctly
[INFO] cargo test -p context-graph-embeddings warm:: -- --nocapture PASSED
```

---

## Scope

### In Scope

- Add `VramAllocation` struct with real/fake detection to `warm/loader/types.rs`
- Add `InferenceValidation` struct with golden comparison to `warm/loader/types.rs`
- Add enhanced `is_real()` and `assert_real()` methods
- Add error variants if not already present in `warm/error.rs`
- Unit tests with REAL data patterns (no mocks)

### Out of Scope

- Modifying existing `TensorMetadata`, `WarmLoadResult`, `LoadedModelWeights` core fields
- Actual CUDA integration (TASK-EMB-017+)
- Inference engine changes (already in TASK-EMB-015)

---

## Definition of Done

### New Structs to Add

```rust
// File: crates/context-graph-embeddings/src/warm/loader/types.rs
// ADD AFTER existing LoadedModelWeights struct

use std::time::Duration;

/// GPU VRAM allocation tracking with real/fake detection.
///
/// # Constitution Alignment
///
/// - AP-007: All values MUST come from real CUDA API calls
/// - REQ-WARM-003: Non-evictable VRAM allocation tracking
///
/// # CRITICAL: No Simulation
///
/// Fake values are FORBIDDEN. The `is_real()` method detects known fake patterns.
#[derive(Debug, Clone)]
pub struct VramAllocation {
    /// Base pointer on GPU (from cudaMalloc).
    /// MUST NOT be 0x7f80_0000_0000 or similar fake value.
    pub base_ptr: u64,

    /// Total bytes allocated.
    pub size_bytes: usize,

    /// VRAM used before loading (from cudaMemGetInfo), in MB.
    pub vram_before_mb: u64,

    /// VRAM used after loading (from cudaMemGetInfo), in MB.
    pub vram_after_mb: u64,

    /// Actual delta: vram_after_mb - vram_before_mb.
    pub vram_delta_mb: u64,
}

impl VramAllocation {
    /// Create new VramAllocation with fail-fast validation.
    ///
    /// # Panics
    ///
    /// - If `base_ptr` is 0 (null pointer)
    /// - If `size_bytes` is 0 (empty allocation)
    #[must_use]
    pub fn new(
        base_ptr: u64,
        size_bytes: usize,
        vram_before_mb: u64,
        vram_after_mb: u64,
    ) -> Self {
        assert!(
            base_ptr != 0,
            "CONSTITUTION VIOLATION AP-007: base_ptr is null. \
             Real cudaMalloc pointer required."
        );
        assert!(
            size_bytes > 0,
            "CONSTITUTION VIOLATION AP-007: size_bytes is 0. \
             Real allocation size required."
        );

        let vram_delta_mb = vram_after_mb.saturating_sub(vram_before_mb);

        Self {
            base_ptr,
            size_bytes,
            vram_before_mb,
            vram_after_mb,
            vram_delta_mb,
        }
    }

    /// Check if allocation looks real (not simulated).
    ///
    /// Returns `false` if any known fake pattern is detected.
    #[must_use]
    pub fn is_real(&self) -> bool {
        // Fake pointer check (common simulation value)
        if self.base_ptr == 0x7f80_0000_0000u64 {
            return false;
        }

        // Zero allocation is suspicious
        if self.size_bytes == 0 {
            return false;
        }

        // VRAM delta should roughly match size_bytes
        let expected_delta_mb = (self.size_bytes / (1024 * 1024)) as u64;
        let delta_diff = (self.vram_delta_mb as i64 - expected_delta_mb as i64).abs();

        // Allow 50MB tolerance for GPU overhead
        delta_diff < 50
    }

    /// Panic if allocation appears simulated.
    ///
    /// # Panics
    ///
    /// Constitution AP-007 violation if `is_real()` returns false.
    pub fn assert_real(&self) {
        if !self.is_real() {
            panic!(
                "[EMB-E010] SIMULATION_DETECTED: VramAllocation contains fake data. \
                 base_ptr=0x{:x}, size={}, delta={}MB. Constitution AP-007 violation.",
                self.base_ptr, self.size_bytes, self.vram_delta_mb
            );
        }
    }

    /// Get VRAM delta as human-readable string.
    #[must_use]
    pub fn delta_display(&self) -> String {
        format!(
            "{} MB ({} -> {} MB)",
            self.vram_delta_mb, self.vram_before_mb, self.vram_after_mb
        )
    }
}

/// Inference validation result with golden reference comparison.
///
/// # Constitution Alignment
///
/// - AP-007: Output MUST NOT be sin wave or all zeros
/// - Validates model produces meaningful real output
///
/// # CRITICAL: No Simulation
///
/// The `is_real()` method detects fake inference patterns.
#[derive(Debug, Clone)]
pub struct InferenceValidation {
    /// Sample input used for validation (e.g., "The quick brown fox").
    pub sample_input: String,

    /// Sample output (embedding vector).
    /// MUST NOT be sin wave pattern: (i * 0.001).sin()
    pub sample_output: Vec<f32>,

    /// L2 norm of output (should be ~1.0 for normalized embeddings).
    pub output_norm: f32,

    /// Inference latency.
    pub latency: Duration,

    /// Whether output matches golden reference within tolerance.
    pub matches_golden: bool,

    /// Cosine similarity to golden reference (0.0 to 1.0).
    /// Must be > 0.95 for real inference.
    pub golden_similarity: f32,
}

impl InferenceValidation {
    /// Create new InferenceValidation with fail-fast validation.
    ///
    /// # Panics
    ///
    /// - If `sample_input` is empty
    /// - If `sample_output` is empty
    #[must_use]
    pub fn new(
        sample_input: String,
        sample_output: Vec<f32>,
        output_norm: f32,
        latency: Duration,
        matches_golden: bool,
        golden_similarity: f32,
    ) -> Self {
        assert!(
            !sample_input.is_empty(),
            "CONSTITUTION VIOLATION AP-007: sample_input is empty. \
             Real test input required."
        );
        assert!(
            !sample_output.is_empty(),
            "CONSTITUTION VIOLATION AP-007: sample_output is empty. \
             Real inference output required."
        );

        Self {
            sample_input,
            sample_output,
            output_norm,
            latency,
            matches_golden,
            golden_similarity,
        }
    }

    /// Check if output looks like real inference (not fake pattern).
    ///
    /// Detects:
    /// - Sin wave patterns
    /// - All-zero outputs
    /// - Low golden similarity (<0.95)
    #[must_use]
    pub fn is_real(&self) -> bool {
        // Sin wave pattern detection: suspiciously smooth differences
        let is_sin_wave = self.sample_output.len() >= 10 &&
            self.sample_output.windows(10).all(|w| {
                let diffs: Vec<f32> = w.windows(2).map(|p| (p[1] - p[0]).abs()).collect();
                let variance = diffs.iter().map(|d| d * d).sum::<f32>() / diffs.len() as f32;
                variance < 0.0001 // Suspiciously smooth
            });

        // All zeros detection
        let is_zeros = self.sample_output.iter().all(|&v| v.abs() < 1e-6);

        // Check golden similarity (must be high for real model)
        let has_good_golden = self.golden_similarity > 0.95;

        !is_sin_wave && !is_zeros && has_good_golden
    }

    /// Panic if output looks fake.
    ///
    /// # Panics
    ///
    /// Constitution AP-007 violation with error code EMB-E011.
    pub fn assert_real(&self) {
        if !self.is_real() {
            panic!(
                "[EMB-E011] FAKE_INFERENCE: Output pattern indicates simulation. \
                 Golden similarity: {:.4}, output_len: {}. Constitution AP-007 violation.",
                self.golden_similarity, self.sample_output.len()
            );
        }
    }

    /// Calculate L2 norm of sample_output for verification.
    #[must_use]
    pub fn calculate_norm(&self) -> f32 {
        self.sample_output.iter().map(|v| v * v).sum::<f32>().sqrt()
    }
}
```

### Constraints

- All constructors MUST use `assert!()` for fail-fast validation
- All `is_real()` methods MUST detect known fake patterns
- Error messages MUST include Constitution error codes `[EMB-EXXX]`
- NO backwards compatibility shims - fail fast on invalid data

### Files to Modify

| File Path | Change |
|-----------|--------|
| `crates/context-graph-embeddings/src/warm/loader/types.rs` | Add `VramAllocation` and `InferenceValidation` structs |

### Files to Verify Exist

| File Path | Must Contain |
|-----------|--------------|
| `crates/context-graph-embeddings/src/warm/error.rs` | `WarmError` enum with exit codes |
| `crates/context-graph-embeddings/src/gpu/mod.rs` | `GpuTensor` struct |

---

## Validation Criteria

- [x] `VramAllocationTracking` struct with `is_real()` and `assert_real()` methods
- [x] `InferenceValidation` struct with golden comparison
- [x] All `is_real()` methods detect: fake pointer, zero checksum, sin wave, low golden similarity
- [x] All `assert_real()` methods panic with Constitution error codes
- [x] Unit tests for all 3 edge cases (VRAM delta mismatch, sin wave, low golden)
- [x] `cargo check -p context-graph-embeddings` passes
- [x] `cargo test -p context-graph-embeddings warm::loader::types:: -- --nocapture` passes (43 tests)

## Implementation Evidence (2026-01-06)

### Source of Truth Verification

**File Location**: `crates/context-graph-embeddings/src/warm/loader/types.rs`

**VramAllocationTracking** (lines 615-816):
- `new()` at line 679 with fail-fast validation
- `is_real()` at line 736 detecting fake pointer (0x7f80_0000_0000) and delta mismatch
- `assert_real()` at line 776 with [EMB-E010] error code
- Helper methods: `delta_display()`, `size_mb()`, `size_gb()`

**InferenceValidation** (lines 861-1083):
- `new()` at line 919 with fail-fast validation
- `is_real()` at line 979 detecting sin wave, all-zeros, low golden similarity
- `assert_real()` at line 1044 with [EMB-E011] error code
- Helper methods: `calculate_norm()`, `verify_norm()`, `output_dimension()`

### Test Results

```bash
$ cargo test --package context-graph-embeddings warm::loader::types:: -- --nocapture
running 43 tests
...
test result: ok. 43 passed; 0 failed; 0 ignored; 0 measured; 1308 filtered out
```

### Edge Case Tests Verified

1. **VRAM Delta Mismatch** (`test_vram_allocation_detects_delta_mismatch`): 1KB allocation with 1000MB delta detected as fake
2. **Sin Wave Pattern** (`test_inference_validation_detects_sin_wave`): `(i * 0.001).sin()` pattern detected as fake
3. **Low Golden Similarity** (`test_inference_validation_rejects_low_golden`): 0.50 similarity rejected (threshold 0.95)

---

## Test Commands

```bash
cd /home/cabdru/contextgraph

# Verify compilation
cargo check -p context-graph-embeddings

# Run specific tests
cargo test -p context-graph-embeddings warm::loader::types:: -- --nocapture

# Run all warm tests
cargo test -p context-graph-embeddings warm:: -- --nocapture

# Verify no fake patterns slip through
grep -rn "0xDEAD\|0xCAFE\|simulate" crates/context-graph-embeddings/src/warm/
```

---

## Manual Verification Checklist

After implementation, manually verify:

1. [x] **File exists**: `crates/context-graph-embeddings/src/warm/loader/types.rs` contains `VramAllocationTracking` struct (line 615)
2. [x] **File exists**: `crates/context-graph-embeddings/src/warm/loader/types.rs` contains `InferenceValidation` struct (line 861)
3. [x] **Fake detection works**: `VramAllocationTracking::is_real()` returns `false` for `base_ptr = 0x7f80_0000_0000` (test_vram_allocation_detects_fake_pointer)
4. [x] **Sin wave detection**: `InferenceValidation::is_real()` returns `false` for sin wave output (test_inference_validation_detects_sin_wave)
5. [x] **Golden check**: `InferenceValidation::is_real()` returns `false` when `golden_similarity < 0.95` (test_inference_validation_rejects_low_golden)
6. [x] **Panic messages**: All panics include `[EMB-E010]` or `[EMB-E011]` error codes (verified via should_panic tests)

---

## Test Cases (REAL Data Only)

```rust
#[cfg(test)]
mod vram_inference_tests {
    use super::*;

    // =========================================================================
    // EDGE CASE 1: VRAM Delta Mismatch
    // =========================================================================

    #[test]
    fn test_vram_allocation_detects_delta_mismatch() {
        let alloc = VramAllocation {
            base_ptr: 0x7fff_0000_1000, // Real-looking pointer
            size_bytes: 1024,           // 1KB allocated
            vram_before_mb: 1000,
            vram_after_mb: 2000,        // Claims 1000MB delta for 1KB!
            vram_delta_mb: 1000,
        };

        assert!(!alloc.is_real(), "Should detect VRAM delta mismatch");
    }

    #[test]
    fn test_vram_allocation_accepts_valid_delta() {
        let alloc = VramAllocation::new(
            0x7fff_0000_1000,           // Real pointer
            104_857_600,                // 100MB
            5000,                       // 5GB before
            5100,                       // 5.1GB after (100MB delta)
        );

        assert!(alloc.is_real(), "Should accept valid VRAM allocation");
    }

    // =========================================================================
    // EDGE CASE 2: Sin Wave Pattern Detection
    // =========================================================================

    #[test]
    fn test_inference_validation_detects_sin_wave() {
        let sin_wave_output: Vec<f32> = (0..768)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();

        let validation = InferenceValidation {
            sample_input: "test input".to_string(),
            sample_output: sin_wave_output,
            output_norm: 1.0,
            latency: Duration::from_millis(10),
            matches_golden: true,
            golden_similarity: 0.99,
        };

        assert!(!validation.is_real(), "Should detect sin wave fake pattern");
    }

    // =========================================================================
    // EDGE CASE 3: Low Golden Similarity
    // =========================================================================

    #[test]
    fn test_inference_validation_rejects_low_golden() {
        let validation = InferenceValidation::new(
            "The quick brown fox".to_string(),
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], // Real-looking
            1.0,
            Duration::from_millis(50),
            false,
            0.50, // LOW golden similarity - REJECT
        );

        assert!(!validation.is_real(), "Should reject low golden similarity");
    }

    #[test]
    fn test_inference_validation_accepts_high_golden() {
        // Generate non-sin-wave realistic output
        let output: Vec<f32> = (0..768)
            .map(|i| ((i * 17 + 42) % 1000) as f32 / 1000.0 - 0.5)
            .collect();

        let validation = InferenceValidation::new(
            "The quick brown fox".to_string(),
            output,
            1.0,
            Duration::from_millis(50),
            true,
            0.98, // HIGH golden similarity - ACCEPT
        );

        assert!(validation.is_real(), "Should accept high golden similarity");
    }

    // =========================================================================
    // FAIL-FAST PANIC TESTS
    // =========================================================================

    #[test]
    #[should_panic(expected = "CONSTITUTION VIOLATION AP-007: base_ptr is null")]
    fn test_vram_allocation_rejects_null_ptr() {
        let _ = VramAllocation::new(0, 1024, 1000, 1100);
    }

    #[test]
    #[should_panic(expected = "CONSTITUTION VIOLATION AP-007: size_bytes is 0")]
    fn test_vram_allocation_rejects_zero_size() {
        let _ = VramAllocation::new(0x7fff_0000_1000, 0, 1000, 1100);
    }

    #[test]
    #[should_panic(expected = "CONSTITUTION VIOLATION AP-007: sample_input is empty")]
    fn test_inference_validation_rejects_empty_input() {
        let _ = InferenceValidation::new(
            "".to_string(),
            vec![0.1, 0.2, 0.3],
            1.0,
            Duration::from_millis(10),
            true,
            0.99,
        );
    }

    #[test]
    #[should_panic(expected = "CONSTITUTION VIOLATION AP-007: sample_output is empty")]
    fn test_inference_validation_rejects_empty_output() {
        let _ = InferenceValidation::new(
            "test".to_string(),
            vec![],
            0.0,
            Duration::from_millis(10),
            false,
            0.0,
        );
    }

    #[test]
    #[should_panic(expected = "[EMB-E010] SIMULATION_DETECTED")]
    fn test_vram_allocation_assert_real_panics_on_fake() {
        let fake_alloc = VramAllocation {
            base_ptr: 0x7f80_0000_0000, // KNOWN FAKE POINTER
            size_bytes: 1024,
            vram_before_mb: 1000,
            vram_after_mb: 1001,
            vram_delta_mb: 1,
        };

        fake_alloc.assert_real();
    }

    #[test]
    #[should_panic(expected = "[EMB-E011] FAKE_INFERENCE")]
    fn test_inference_validation_assert_real_panics_on_fake() {
        let fake_validation = InferenceValidation {
            sample_input: "test".to_string(),
            sample_output: vec![0.0; 768], // ALL ZEROS
            output_norm: 0.0,
            latency: Duration::from_millis(10),
            matches_golden: false,
            golden_similarity: 0.1, // LOW
        };

        fake_validation.assert_real();
    }
}
```

---

## Pseudo Code

```
types.rs (additions):

  Define VramAllocation struct
    base_ptr, size_bytes, vram_before_mb, vram_after_mb, vram_delta_mb
    new() with fail-fast validation (null ptr, zero size)
    is_real() checks for fake pointer pattern (0x7f80_0000_0000)
    is_real() checks delta matches size within 50MB tolerance
    assert_real() panics with [EMB-E010] on fake detection
    delta_display() for human-readable output

  Define InferenceValidation struct
    sample_input, sample_output, output_norm, latency, matches_golden, golden_similarity
    new() with fail-fast validation (empty input/output)
    is_real() checks for sin wave pattern (variance < 0.0001)
    is_real() checks for all-zeros output
    is_real() checks golden_similarity > 0.95
    assert_real() panics with [EMB-E011] on fake detection
    calculate_norm() for verification
```

</task_spec>
