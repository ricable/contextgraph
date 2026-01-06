# TASK-EMB-017: CUDA Integration for Warm Loading Pipeline

<task_spec id="TASK-EMB-017" version="2.0">

## Metadata

| Field | Value |
|-------|-------|
| **Title** | CUDA Integration for Warm Loading Pipeline |
| **Status** | ready |
| **Layer** | logic |
| **Sequence** | 17 |
| **Implements** | REQ-EMB-001, REQ-EMB-002, Constitution AP-007 |
| **Depends On** | TASK-EMB-016 (VramAllocationTracking, InferenceValidation) |
| **Estimated Complexity** | high |
| **Parallel Group** | E |

---

## AI Agent Context

### Current Codebase State (as of 2025-01-06)

**IMPORTANT**: This section provides the actual state of the codebase. Do NOT assume files exist without verification.

#### Verified Existing Files

| File | Status | Contains |
|------|--------|----------|
| `crates/context-graph-embeddings/src/warm/mod.rs` | EXISTS | Module exports: `pub mod error`, `pub mod loader`, `pub mod inference` |
| `crates/context-graph-embeddings/src/warm/error.rs` | EXISTS | `WarmError` enum with exit codes 101-115, `WarmResult<T>` alias |
| `crates/context-graph-embeddings/src/warm/loader/mod.rs` | EXISTS | Module structure with `types`, `operations`, `engine` |
| `crates/context-graph-embeddings/src/warm/loader/types.rs` | EXISTS | `TensorMetadata`, `WarmLoadResult`, `LoadedModelWeights`, `VramAllocationTracking`, `InferenceValidation` (1900+ lines) |
| `crates/context-graph-embeddings/src/warm/loader/operations.rs` | EXISTS | `load_single_model()`, `allocate_model_vram()`, `load_weights()`, `validate_model()` |
| `crates/context-graph-embeddings/src/warm/inference/mod.rs` | EXISTS | Inference module exports |
| `crates/context-graph-embeddings/src/warm/inference/engine.rs` | EXISTS | `InferenceEngine` struct with `run_test_inference()` |

#### Key Types from TASK-EMB-016 (COMPLETED)

```rust
// From warm/loader/types.rs - ALREADY IMPLEMENTED
pub struct VramAllocationTracking {
    pub total_bytes_requested: u64,
    pub total_bytes_allocated: u64,
    pub allocation_timestamps: Vec<std::time::Instant>,
    pub cuda_device_id: i32,
    pub base_address: u64,  // 0x7f80_0000_0000 = FAKE detection
    pub allocation_pattern: AllocationPattern,
}

pub struct InferenceValidation {
    pub test_inference_run: bool,
    pub output_dimensions_correct: bool,
    pub output_range_valid: bool,
    pub latency_within_budget: bool,
    pub validation_timestamp: std::time::Instant,
    pub golden_similarity: f32,  // Must be > 0.99
    pub sin_wave_detected: bool, // TRUE = fake/mock data
}
```

#### Key Functions from operations.rs

```rust
// From warm/loader/operations.rs - ALREADY IMPLEMENTED
pub fn load_single_model(
    model_path: &Path,
    config: &WarmConfig,
    allocator: &mut WarmCudaAllocator,
) -> WarmResult<WarmLoadResult>;

pub fn allocate_model_vram(
    allocator: &mut WarmCudaAllocator,
    weights: &LoadedModelWeights,
    config: &WarmConfig,
) -> WarmResult<VramAllocationTracking>;

pub fn validate_model(
    weights: &LoadedModelWeights,
    vram_tracking: &VramAllocationTracking,
    engine: &InferenceEngine,
) -> WarmResult<InferenceValidation>;
```

#### Hardware Target (from constitution.yaml)

```yaml
hardware:
  target_gpu: "NVIDIA RTX 5090"
  vram_budget_gb: 32
  cuda_version: "13.1"
  tensor_cores: "5th generation"
  sm_count: 170
```

#### Error Format (Constitution mcp.errors)

All errors MUST use format: `[EMB-EXXX] CATEGORY: message`

Exit codes for warm loading (from warm/error.rs):
- 101: SAFETENSORS_PARSE
- 102: TENSOR_DTYPE_MISMATCH
- 103: TENSOR_SHAPE_MISMATCH
- 104: CUDA_ALLOC_FAILED
- 105: CUDA_COPY_FAILED
- 106: VRAM_EXHAUSTED
- 107: MODEL_VALIDATION_FAILED
- 108: INFERENCE_TIMEOUT
- 109: FAKE_ALLOCATION_DETECTED
- 110: SIN_WAVE_OUTPUT_DETECTED
- 115: CONSTITUTION_VIOLATION

---

## Context

TASK-EMB-016 implemented `VramAllocationTracking` and `InferenceValidation` types with robust fake detection. However, the actual CUDA integration that connects these types to real GPU operations was marked as "Out of Scope".

This task implements the actual CUDA binding layer that:
1. Performs real CUDA memory allocations (not fakes)
2. Copies SafeTensor weights to GPU
3. Executes test inference to validate model loading
4. Fails fast if ANY fake/mock patterns detected

### Constitution Requirements

**AP-007 (NO FALLBACK)**:
- Mock/stub data is FORBIDDEN in production
- Fake allocations MUST cause immediate process exit
- Sin wave outputs MUST cause immediate process exit
- No graceful degradation - fail loud and clear

---

## Prerequisites

- [x] TASK-EMB-016 completed (VramAllocationTracking, InferenceValidation exist)
- [ ] CUDA 13.1+ toolkit installed
- [ ] cuDNN 9.x available
- [ ] RTX 5090 or compatible GPU available
- [ ] SafeTensors test weights available at `tests/fixtures/weights/`

---

## Scope

### In Scope

1. **WarmCudaAllocator CUDA Bindings**
   - Real `cuMemAlloc` calls via cuda-rs or cudarc
   - Allocation tracking with base address verification
   - VRAM budget enforcement (32GB max)

2. **Weight Transfer Operations**
   - `cuMemcpyHtoD` for SafeTensor → GPU
   - Async transfer with stream management
   - Transfer validation (byte count verification)

3. **Inference Execution**
   - Forward pass execution for test inference
   - Output validation against golden embeddings
   - Latency measurement and budget enforcement

4. **Fail-Fast Validation**
   - Base address pattern check (0x7f80_0000_0000 = FAKE)
   - Output sin wave pattern detection
   - Golden similarity threshold (> 0.99)

### Out of Scope

- Model architecture implementation (separate task)
- Quantization (TASK-EMB-012)
- Multi-GPU coordination (future task)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-embeddings/src/warm/cuda/allocator.rs

use crate::warm::error::{WarmError, WarmResult};
use crate::warm::loader::types::{VramAllocationTracking, AllocationPattern};

/// Real CUDA memory allocator - NO FAKES ALLOWED.
///
/// # Constitution Compliance
/// - AP-007: No mock allocations permitted
/// - Fake base address (0x7f80_0000_0000) causes exit(109)
///
/// # Hardware Target
/// - RTX 5090 with 32GB VRAM
/// - CUDA 13.1+
pub struct WarmCudaAllocator {
    device_id: i32,
    total_allocated: u64,
    vram_budget: u64,
    allocations: Vec<CudaAllocation>,
}

/// Individual CUDA allocation record.
pub struct CudaAllocation {
    pub device_ptr: u64,
    pub size_bytes: u64,
    pub timestamp: std::time::Instant,
    pub tensor_name: String,
}

impl WarmCudaAllocator {
    /// Create allocator for specified CUDA device.
    ///
    /// # Panics
    /// - If CUDA device not available
    /// - If device doesn't meet RTX 5090 requirements
    pub fn new(device_id: i32, vram_budget_gb: u64) -> WarmResult<Self>;

    /// Allocate VRAM for tensor.
    ///
    /// # Returns
    /// - Device pointer on success
    /// - WarmError::CudaAllocFailed if allocation fails
    /// - WarmError::VramExhausted if budget exceeded
    ///
    /// # Panics (Constitution AP-007)
    /// - If returned pointer matches fake pattern 0x7f80_0000_0000
    pub fn allocate(&mut self, size_bytes: u64, tensor_name: &str) -> WarmResult<u64>;

    /// Copy data from host to device.
    ///
    /// # Panics
    /// - If copy fails (no retry, immediate exit)
    pub fn copy_to_device(&self, host_ptr: *const u8, device_ptr: u64, size: u64) -> WarmResult<()>;

    /// Free device memory.
    pub fn free(&mut self, device_ptr: u64) -> WarmResult<()>;

    /// Get allocation tracking for validation.
    pub fn get_tracking(&self) -> VramAllocationTracking;

    /// Verify all allocations are real (not fake).
    ///
    /// # Panics (exit 109)
    /// - If ANY allocation has fake base address pattern
    pub fn verify_real_allocations(&self);
}

// File: crates/context-graph-embeddings/src/warm/cuda/transfer.rs

use crate::warm::error::WarmResult;
use crate::warm::loader::types::LoadedModelWeights;

/// Transfer SafeTensor weights to GPU.
///
/// # Constitution Compliance
/// - No partial transfers - all or nothing
/// - Transfer verification required
pub fn transfer_weights_to_gpu(
    allocator: &mut WarmCudaAllocator,
    weights: &LoadedModelWeights,
) -> WarmResult<GpuWeights>;

/// GPU-resident model weights.
pub struct GpuWeights {
    pub device_ptrs: HashMap<String, u64>,
    pub total_bytes: u64,
    pub transfer_time_ms: u64,
}

// File: crates/context-graph-embeddings/src/warm/cuda/inference.rs

use crate::warm::error::WarmResult;
use crate::warm::loader::types::InferenceValidation;

/// Execute test inference to validate model loading.
///
/// # Constitution Compliance
/// - AP-007: Sin wave output = exit(110)
/// - Golden similarity < 0.99 = exit(107)
///
/// # Validation Steps
/// 1. Run forward pass with test input
/// 2. Check output dimensions match expected
/// 3. Check output values in valid range [-1, 1]
/// 4. Compare against golden embeddings (similarity > 0.99)
/// 5. Detect sin wave pattern (FORBIDDEN)
pub fn run_validation_inference(
    gpu_weights: &GpuWeights,
    test_input: &[f32],
    golden_output: &[f32],
) -> WarmResult<InferenceValidation>;

/// Detect sin wave pattern in output (indicates fake/mock).
///
/// # Algorithm
/// - Compute FFT of output
/// - Check for dominant single frequency
/// - If >80% energy in one frequency band → sin wave detected
fn detect_sin_wave_pattern(output: &[f32]) -> bool;

/// Compute cosine similarity between output and golden.
fn compute_golden_similarity(output: &[f32], golden: &[f32]) -> f32;
```

### Constraints

- MUST use real CUDA allocations (cuMemAlloc or equivalent)
- MUST fail immediately (exit with code) on fake detection
- MUST NOT retry failed operations
- MUST verify all allocations before returning success
- MUST use golden embeddings for validation (no random data)

### Verification

```bash
# Run with real GPU - MUST pass
cargo test -p context-graph-embeddings cuda::allocator:: -- --nocapture

# Verify exit codes on fake detection
cargo test -p context-graph-embeddings cuda::fake_detection:: -- --nocapture
```

---

## Files to Create

| File Path | Description |
|-----------|-------------|
| `crates/context-graph-embeddings/src/warm/cuda/mod.rs` | CUDA module exports |
| `crates/context-graph-embeddings/src/warm/cuda/allocator.rs` | Real CUDA allocator |
| `crates/context-graph-embeddings/src/warm/cuda/transfer.rs` | Weight transfer operations |
| `crates/context-graph-embeddings/src/warm/cuda/inference.rs` | Validation inference |

## Files to Modify

| File Path | Change |
|-----------|--------|
| `crates/context-graph-embeddings/src/warm/mod.rs` | Add `pub mod cuda;` |
| `crates/context-graph-embeddings/Cargo.toml` | Add cuda-rs or cudarc dependency |

---

## Full State Verification

### Source of Truth

| Verification | Source | Expected State |
|--------------|--------|----------------|
| CUDA device available | `nvidia-smi` | Shows RTX 5090 or compatible |
| CUDA toolkit version | `nvcc --version` | CUDA 13.1+ |
| Allocator creates real memory | `cuMemGetInfo` | Free VRAM decreases after allocation |
| Weights transfer completes | Device pointer valid | Non-zero, not fake pattern |
| Inference produces output | Forward pass | 768-dim embedding vector |

### Execute & Inspect

```bash
# 1. Verify CUDA environment
nvidia-smi
nvcc --version

# 2. Run allocator test with memory inspection
RUST_LOG=debug cargo test -p context-graph-embeddings cuda::allocator::test_real_allocation -- --nocapture 2>&1 | grep -E "(allocated|device_ptr|base_address)"

# 3. Verify VRAM actually used
nvidia-smi --query-gpu=memory.used --format=csv -l 1 &
cargo test -p context-graph-embeddings cuda::transfer::test_weight_transfer -- --nocapture
# Memory.used should increase during test

# 4. Run inference validation
cargo test -p context-graph-embeddings cuda::inference::test_validation -- --nocapture
```

### Edge Cases (3 Required)

#### Edge Case 1: Fake Allocation Detection

**Scenario**: Allocator returns mock pointer (0x7f80_0000_0000 pattern)

```rust
#[test]
fn test_fake_allocation_causes_exit() {
    // This test MUST exit(109) if fake allocation detected
    let mut allocator = WarmCudaAllocator::new(0, 32).unwrap();

    // Inject fake pointer for testing
    #[cfg(test)]
    allocator.inject_fake_pointer(0x7f80_0000_0000_u64);

    // This MUST panic/exit, not return Ok
    let result = std::panic::catch_unwind(|| {
        allocator.verify_real_allocations();
    });

    assert!(result.is_err(), "Fake allocation MUST cause panic");
}
```

#### Edge Case 2: Sin Wave Output Detection

**Scenario**: Model outputs sin wave pattern (indicates mock inference)

```rust
#[test]
fn test_sin_wave_output_causes_exit() {
    // Generate obvious sin wave
    let fake_output: Vec<f32> = (0..768)
        .map(|i| (i as f32 * 0.1).sin())
        .collect();

    let detected = detect_sin_wave_pattern(&fake_output);
    assert!(detected, "Sin wave pattern MUST be detected");

    // Full validation should fail
    let result = run_validation_inference(
        &gpu_weights,
        &test_input,
        &golden_output,
    );

    assert!(matches!(result, Err(WarmError::SinWaveOutputDetected)));
}
```

#### Edge Case 3: VRAM Budget Exhaustion

**Scenario**: Allocation exceeds 32GB budget

```rust
#[test]
fn test_vram_budget_enforcement() {
    let mut allocator = WarmCudaAllocator::new(0, 32).unwrap(); // 32GB budget

    // Try to allocate more than budget
    let result = allocator.allocate(35 * 1024 * 1024 * 1024, "oversized_tensor");

    assert!(matches!(result, Err(WarmError::VramExhausted { .. })));
}
```

### Evidence of Success

After implementation, these artifacts MUST exist:

| Artifact | Location | Verification Command |
|----------|----------|---------------------|
| CUDA module | `src/warm/cuda/mod.rs` | `test -f crates/context-graph-embeddings/src/warm/cuda/mod.rs` |
| Allocator impl | `src/warm/cuda/allocator.rs` | `grep "impl WarmCudaAllocator" crates/context-graph-embeddings/src/warm/cuda/allocator.rs` |
| Real allocation test | test output | `cargo test cuda::allocator::test_real -- 2>&1 \| grep "PASSED"` |
| Fake detection test | test output | `cargo test cuda::fake_detection -- 2>&1 \| grep "PASSED"` |
| VRAM decrease during test | nvidia-smi | Manual observation during test run |

---

## Manual Verification Requirements

### Physical Output Checks

**REQUIRED**: After running tests, manually verify:

1. **VRAM Usage**: Run `nvidia-smi` before and during tests. VRAM usage MUST increase.
2. **No Mock Warnings**: Test output MUST NOT contain "mock", "fake", "stub" warnings
3. **Exit Codes**: Failed fake detection tests MUST exit with codes 109 or 110
4. **Allocation Addresses**: Log output device pointers MUST NOT match 0x7f80_0000_0000

### Inspection Checklist

```bash
# Run this checklist after implementation

# 1. Check module structure exists
ls -la crates/context-graph-embeddings/src/warm/cuda/

# 2. Verify no mock keywords in production code
grep -r "mock\|fake\|stub" crates/context-graph-embeddings/src/warm/cuda/ && echo "FAIL: Mock keywords found" || echo "PASS: No mock keywords"

# 3. Run tests and capture exit codes
cargo test -p context-graph-embeddings cuda:: -- --nocapture; echo "Exit code: $?"

# 4. Verify CUDA calls are real (check for cuda-sys or cudarc usage)
grep -r "cuMemAlloc\|cuda::\|cudarc::" crates/context-graph-embeddings/src/warm/cuda/

# 5. Check error handling uses exit codes
grep -r "exit\|std::process::exit" crates/context-graph-embeddings/src/warm/cuda/
```

---

## Test Commands

```bash
cd /home/cabdru/contextgraph

# Build with CUDA support
cargo build -p context-graph-embeddings --features cuda

# Run all CUDA tests
cargo test -p context-graph-embeddings cuda:: -- --nocapture

# Run specific test categories
cargo test -p context-graph-embeddings cuda::allocator:: -- --nocapture
cargo test -p context-graph-embeddings cuda::transfer:: -- --nocapture
cargo test -p context-graph-embeddings cuda::inference:: -- --nocapture

# Check for compilation errors
cargo check -p context-graph-embeddings --features cuda
```

---

## Pseudo Code

```
cuda/mod.rs:
  pub mod allocator;
  pub mod transfer;
  pub mod inference;

cuda/allocator.rs:
  struct WarmCudaAllocator {
    device_id, total_allocated, vram_budget, allocations
  }

  impl WarmCudaAllocator:
    new(device_id, vram_budget_gb):
      - Initialize CUDA context
      - Query device properties
      - Assert RTX 5090 compatible
      - Return allocator

    allocate(size_bytes, tensor_name):
      - Check budget not exceeded
      - Call cuMemAlloc (REAL, not mock)
      - Record allocation with timestamp
      - ASSERT device_ptr != FAKE_PATTERN (0x7f80_0000_0000)
      - Return device_ptr or exit(109)

    copy_to_device(host_ptr, device_ptr, size):
      - Call cuMemcpyHtoD (REAL)
      - Verify bytes copied == size
      - Return Ok or exit(105)

    verify_real_allocations():
      - For each allocation:
        - Check base address != FAKE_PATTERN
        - If fake detected: exit(109)

cuda/transfer.rs:
  transfer_weights_to_gpu(allocator, weights):
    - For each tensor in weights:
      - Allocate GPU memory
      - Copy tensor data to GPU
      - Record device pointer
    - Return GpuWeights with all pointers

cuda/inference.rs:
  run_validation_inference(gpu_weights, test_input, golden_output):
    - Execute forward pass on GPU
    - Get output embeddings
    - Check dimensions (768)
    - Check value range [-1, 1]
    - Detect sin wave pattern → exit(110) if found
    - Compute golden similarity
    - If similarity < 0.99 → exit(107)
    - Return InferenceValidation

  detect_sin_wave_pattern(output):
    - Compute FFT
    - Find dominant frequency
    - If >80% energy in single band → return true
    - Return false
```

---

## Dependencies

### Cargo.toml Additions

```toml
[dependencies]
# Choose ONE of these CUDA bindings
cudarc = { version = "0.11", optional = true }  # Preferred: safe Rust bindings
# OR
cuda-sys = { version = "0.3", optional = true }  # Alternative: raw FFI

[features]
cuda = ["cudarc"]  # or ["cuda-sys"]
```

### System Requirements

- CUDA Toolkit 13.1+
- cuDNN 9.x
- NVIDIA driver 550+
- RTX 5090 or compatible GPU

</task_spec>
