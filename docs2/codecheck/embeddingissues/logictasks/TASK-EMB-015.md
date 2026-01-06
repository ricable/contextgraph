# TASK-EMB-015: Replace Fake Inference Validation with Real GPU Model Inference

<task_spec id="TASK-EMB-015" version="2.0">

## Metadata

| Field | Value |
|-------|-------|
| **Title** | Replace Fake Inference Validation with Real GPU Model Inference |
| **Status** | ready |
| **Layer** | warm-loading |
| **Sequence** | 15 |
| **Implements** | Constitution AP-007 (No stub data in production) |
| **Depends On** | TASK-EMB-013 (COMPLETE), TASK-EMB-014 (COMPLETE) |
| **Estimated Complexity** | high |
| **Parallel Group** | None (sequential after EMB-014) |

---

## ⚠️ CRITICAL CONSTITUTION VIOLATION TO FIX

**Current Code Contains FORBIDDEN Fake Data:**

```rust
// FILE: crates/context-graph-embeddings/src/warm/loader/operations.rs
// LINES 405-408 - THIS CODE MUST BE DELETED:

// Simulate test inference output   ← FORBIDDEN COMMENT
let output: Vec<f32> = (0..expected_dimension)
    .map(|i| (i as f32 * 0.001).sin())   // ← FAKE SIN-WAVE PATTERN - VIOLATES AP-007
    .collect();
```

**Constitution AP-007 States:**
> "Stub data in prod -> use tests/fixtures/"

The sin-wave pattern is stub data masquerading as inference output. This MUST be replaced with real model inference on GPU-resident weights.

---

## Context

### Dependency Chain (Source of Truth)

| Task | Status | What It Does |
|------|--------|--------------|
| **TASK-EMB-013** | ✅ COMPLETE | Real weight loading from SafeTensors files with SHA256 checksums |
| **TASK-EMB-014** | ✅ COMPLETE | Real CUDA allocation via `WarmCudaAllocator::allocate_protected()` |
| **TASK-EMB-015** | 🔴 THIS TASK | Replace fake sin-wave validation with real inference |

After TASK-EMB-013 and TASK-EMB-014, weights are:
1. Loaded from real `.safetensors` files
2. Allocated in real VRAM via `cudaMalloc`
3. But validation uses FAKE sin-wave output instead of running the model

### Hardware Target (from constitution.yaml)

```yaml
hardware:
  gpu: RTX 5090
  vram: 32GB
  cuda_version: "13.1"
```

### Stack Requirements (from constitution.yaml)

```yaml
stack:
  rust: "1.75+"
  cuda: "13.1"
  inference: "candle-core"  # GPU inference library
```

---

## Input Context Files (MUST READ)

| Purpose | File Path | What to Extract |
|---------|-----------|-----------------|
| **Constitution** | `docs2/constitution.yaml` | AP-007, hardware specs, stack requirements |
| **Fake Code Location** | `crates/context-graph-embeddings/src/warm/loader/operations.rs` | Lines 388-418, specifically line 407 |
| **Validator Interface** | `crates/context-graph-embeddings/src/warm/validation/validator.rs` | `WarmValidator` struct and methods |
| **Test Config** | `crates/context-graph-embeddings/src/warm/validation/config.rs` | `TestInferenceConfig` struct |
| **Model Handle** | `crates/context-graph-embeddings/src/warm/handle.rs` | `ModelHandle` with VRAM pointer |
| **CUDA Allocator** | `crates/context-graph-embeddings/src/warm/cuda_alloc.rs` | `WarmCudaAllocator` interface |

---

## Prerequisites

- [x] TASK-EMB-013 completed (real SafeTensors weight loading)
- [x] TASK-EMB-014 completed (real CUDA allocation)
- [ ] Read Constitution section `principles.AP-007`
- [ ] Verify `candle-core` and `candle-nn` are in Cargo.toml
- [ ] Understand how ModelHandle provides VRAM pointer

---

## Scope

### In Scope

1. **DELETE** the fake sin-wave validation code at `operations.rs:405-408`
2. **IMPLEMENT** real inference using Candle on GPU-resident weights
3. **CREATE** `InferenceEngine` struct that wraps Candle model
4. **UPDATE** `validate_model()` to call real inference
5. **ADD** comprehensive error handling with EMB-E011 error codes
6. **WRITE** integration tests using real model files (NO MOCK DATA)

### Out of Scope

- Creating new model architectures (use existing embedder configs)
- Changing the WarmValidator public API
- Modifying TASK-EMB-013 or TASK-EMB-014 code

---

## Definition of Done

### Code to DELETE (operations.rs:405-408)

```rust
// DELETE THIS ENTIRE BLOCK:
// Simulate test inference output
let output: Vec<f32> = (0..expected_dimension)
    .map(|i| (i as f32 * 0.001).sin())
    .collect();
```

### Code to IMPLEMENT

#### 1. InferenceEngine Struct (NEW FILE)

```rust
// File: crates/context-graph-embeddings/src/warm/inference/engine.rs

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use crate::warm::error::{WarmError, WarmResult};
use crate::warm::handle::ModelHandle;

/// Real GPU inference engine using Candle.
///
/// Constitution AP-007 Compliance: This performs REAL inference
/// on GPU-resident weights. NO fake data patterns allowed.
pub struct InferenceEngine {
    device: Device,
    model_id: String,
}

impl InferenceEngine {
    /// Create inference engine for a loaded model.
    ///
    /// # Arguments
    /// * `model_id` - Model identifier (e.g., "E1_Semantic")
    /// * `handle` - ModelHandle with VRAM pointer from TASK-EMB-014
    ///
    /// # Errors
    /// - `WarmError::InferenceInitFailed` - Candle initialization failed
    pub fn new(model_id: &str, handle: &ModelHandle) -> WarmResult<Self> {
        // Initialize CUDA device
        let device = Device::cuda_if_available(handle.device_id() as usize)
            .map_err(|e| WarmError::InferenceInitFailed {
                model_id: model_id.to_string(),
                reason: format!("CUDA device init failed: {}", e),
            })?;

        tracing::info!(
            "[EMB-I015] InferenceEngine initialized for {} on device {}",
            model_id,
            handle.device_id()
        );

        Ok(Self {
            device,
            model_id: model_id.to_string(),
        })
    }

    /// Run test inference with a sample input.
    ///
    /// # CRITICAL: Real Inference
    /// This runs actual GPU computation. The output vector contains
    /// real embedding values, NOT synthetic patterns.
    ///
    /// # Arguments
    /// * `handle` - ModelHandle with VRAM pointer
    /// * `test_input` - Sample text for inference
    /// * `expected_dim` - Expected output dimension
    ///
    /// # Returns
    /// Real embedding vector from GPU inference
    ///
    /// # Errors
    /// - `WarmError::InferenceFailed` - Forward pass failed
    /// - `WarmError::ModelDimensionMismatch` - Output dimension wrong
    pub fn run_test_inference(
        &self,
        handle: &ModelHandle,
        test_input: &str,
        expected_dim: usize,
    ) -> WarmResult<Vec<f32>> {
        let start = std::time::Instant::now();

        // Tokenize input (placeholder - integrate with actual tokenizer)
        let input_ids = self.tokenize(test_input)?;

        // Create input tensor on GPU
        let input_tensor = Tensor::new(&input_ids[..], &self.device)
            .map_err(|e| WarmError::InferenceFailed {
                model_id: self.model_id.clone(),
                reason: format!("Input tensor creation failed: {}", e),
                input_hash: Self::hash_input(test_input),
            })?;

        // Run forward pass using weights at handle.vram_ptr()
        // This is where real GPU computation happens
        let output = self.forward_pass(handle, &input_tensor, expected_dim)?;

        let duration = start.elapsed();
        tracing::info!(
            "[EMB-I015] Test inference for {} completed in {:?}, output dim={}",
            self.model_id,
            duration,
            output.len()
        );

        Ok(output)
    }

    /// Tokenize input text.
    fn tokenize(&self, input: &str) -> WarmResult<Vec<u32>> {
        // Use model-specific tokenizer
        // For now, simple byte-level tokenization
        Ok(input.bytes().map(|b| b as u32).collect())
    }

    /// Execute forward pass on GPU.
    fn forward_pass(
        &self,
        handle: &ModelHandle,
        input: &Tensor,
        expected_dim: usize,
    ) -> WarmResult<Vec<f32>> {
        // Load weights from VRAM pointer
        // handle.vram_ptr() points to real cudaMalloc'd memory

        // For embedding models: simple matrix multiply
        // weights @ input -> output

        // This is a simplified implementation
        // Real implementation needs model architecture specifics
        let output_size = expected_dim;
        let mut output = vec![0.0f32; output_size];

        // TODO: Implement actual Candle forward pass using handle.vram_ptr()
        // This requires integrating Candle's unsafe CUDA tensor from raw pointer

        // TEMPORARY: Return zeros to indicate "not yet implemented"
        // This is DIFFERENT from fake sin-wave - it's an honest placeholder
        // that will fail validation until properly implemented

        tracing::warn!(
            "[EMB-W015] Forward pass returning zeros - IMPLEMENT CANDLE INTEGRATION"
        );

        Ok(output)
    }

    /// Hash input for error logging.
    fn hash_input(input: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        input.hash(&mut hasher);
        hasher.finish()
    }
}
```

#### 2. Update validate_model() in operations.rs

```rust
// File: crates/context-graph-embeddings/src/warm/loader/operations.rs
// REPLACE the validate_model function (lines 388-418)

use crate::warm::inference::InferenceEngine;

/// Validate a model after loading using REAL inference.
///
/// # CRITICAL: Constitution AP-007 Compliance
/// This function runs REAL GPU inference. The previous sin-wave
/// fake data has been DELETED as it violated AP-007.
///
/// # Arguments
/// * `model_id` - Model identifier
/// * `expected_dimension` - Expected embedding dimension
/// * `config` - Warm loading configuration
/// * `validator` - Model validator
/// * `handle` - ModelHandle with VRAM pointer (from TASK-EMB-014)
pub fn validate_model(
    model_id: &str,
    expected_dimension: usize,
    config: &WarmConfig,
    validator: &WarmValidator,
    handle: &ModelHandle,
) -> WarmResult<()> {
    if !config.enable_test_inference {
        tracing::info!(
            "[EMB-I015] Skipping validation for {} (disabled in config)",
            model_id
        );
        return Ok(());
    }

    tracing::info!("[EMB-I015] Validating model {} with REAL inference", model_id);

    // Create inference engine
    let engine = InferenceEngine::new(model_id, handle)?;

    // Run REAL test inference
    let test_input = "The quick brown fox jumps over the lazy dog.";
    let output = engine.run_test_inference(handle, test_input, expected_dimension)?;

    // Validate dimensions
    validator.validate_dimensions(model_id, expected_dimension, output.len())?;

    // Validate no NaN/Inf in output
    validator.validate_weights_finite_for_model(model_id, &output)?;

    // Validate output is not all zeros (catches unimplemented forward pass)
    let non_zero_count = output.iter().filter(|&&v| v != 0.0).count();
    if non_zero_count == 0 {
        return Err(WarmError::ModelValidationFailed {
            model_id: model_id.to_string(),
            reason: "Inference output is all zeros - forward pass not implemented".to_string(),
            expected_output: Some("non-zero embedding values".to_string()),
            actual_output: Some("all zeros".to_string()),
        });
    }

    // Validate output is normalized (for embedding models)
    let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
    if (norm - 1.0).abs() > 0.1 {
        tracing::warn!(
            "[EMB-W015] Output norm {} differs from expected 1.0 for {}",
            norm,
            model_id
        );
    }

    tracing::info!(
        "[EMB-I015] Model {} validation PASSED (dim={}, norm={:.4})",
        model_id,
        output.len(),
        norm
    );

    Ok(())
}
```

#### 3. Update Error Types

```rust
// Add to: crates/context-graph-embeddings/src/warm/error.rs

/// Inference initialization failed.
/// Error code: EMB-E011
#[error("[EMB-E011] Inference initialization failed for {model_id}: {reason}")]
InferenceInitFailed {
    model_id: String,
    reason: String,
},

/// Inference execution failed.
/// Error code: EMB-E011
#[error("[EMB-E011] Inference failed for {model_id}: {reason} (input_hash=0x{input_hash:016x})")]
InferenceFailed {
    model_id: String,
    reason: String,
    input_hash: u64,
},
```

### Constraints

- **NO BACKWARDS COMPATIBILITY** - The fake sin-wave code must be completely removed
- **FAIL FAST** - Any inference failure must return an error immediately
- **NO MOCK DATA** - Tests must use real `.safetensors` model files
- **REAL CUDA** - Must use weights from `ModelHandle::vram_ptr()` allocated by TASK-EMB-014

---

## Full State Verification

### Source of Truth

| Component | Source File | Verification Method |
|-----------|-------------|---------------------|
| Model weights loaded | `loader/operations.rs:load_weights()` | Check SHA256 checksum matches |
| VRAM allocated | `cuda_alloc.rs:allocate_protected()` | Check `vram_ptr != 0` |
| Inference runs | `inference/engine.rs:run_test_inference()` | Check output non-zero |
| Validation passes | `validation/validator.rs` | Check all assertions pass |

### Execute & Inspect Protocol

After each operation, you MUST verify:

```bash
# 1. Verify weights loaded
cargo test -p context-graph-embeddings warm::loader::tests::test_load_weights -- --nocapture
# INSPECT: Log shows "Loaded weights for E1_Semantic ... checksum XX..."

# 2. Verify CUDA allocation
cargo test -p context-graph-embeddings warm::cuda_alloc::tests::test_allocate_protected -- --nocapture
# INSPECT: Log shows "Allocated X bytes at 0x... (REAL cudaMalloc)"

# 3. Verify inference runs
cargo test -p context-graph-embeddings warm::inference::tests::test_inference_engine -- --nocapture
# INSPECT: Log shows "Test inference completed in Xms, output dim=1024"

# 4. Full integration test
cargo test -p context-graph-embeddings warm::loader::tests::test_load_and_validate -- --nocapture
# INSPECT: Log shows "Model E1_Semantic validation PASSED"
```

### Boundary & Edge Case Audit

#### Edge Case 1: Empty Input

```rust
#[test]
fn test_inference_empty_input() {
    let engine = create_test_engine();
    let result = engine.run_test_inference(&handle, "", 1024);

    // EXPECTED: Error, not crash
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, WarmError::InferenceFailed { .. }));
}
```

#### Edge Case 2: Dimension Mismatch

```rust
#[test]
fn test_inference_wrong_dimension() {
    let engine = create_test_engine();
    // Request 512-dim from 1024-dim model
    let result = engine.run_test_inference(&handle, "test", 512);

    // EXPECTED: Dimension mismatch error
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, WarmError::ModelDimensionMismatch { .. }));
}
```

#### Edge Case 3: Invalid VRAM Pointer

```rust
#[test]
fn test_inference_null_vram_ptr() {
    let bad_handle = ModelHandle::new(0, 1024, 0, 0); // null ptr
    let engine = InferenceEngine::new("E1_Semantic", &bad_handle);

    // EXPECTED: Error on inference, not panic
    assert!(engine.is_err() || {
        let eng = engine.unwrap();
        eng.run_test_inference(&bad_handle, "test", 1024).is_err()
    });
}
```

### Evidence of Success Logs

When this task is complete, running the test suite MUST produce:

```
[EMB-I015] InferenceEngine initialized for E1_Semantic on device 0
[EMB-I015] Test inference for E1_Semantic completed in 12ms, output dim=1024
[EMB-I015] Validating model E1_Semantic with REAL inference
[EMB-I015] Model E1_Semantic validation PASSED (dim=1024, norm=0.9998)
```

The following log patterns MUST NOT appear:

```
# FORBIDDEN - indicates fake data:
"Simulate test inference"
".sin()"
"0x7f80"  # fake pointer pattern
"DEAD_BEEF"
```

---

## Files to Create

| File Path | Description |
|-----------|-------------|
| `crates/context-graph-embeddings/src/warm/inference/mod.rs` | Module declaration |
| `crates/context-graph-embeddings/src/warm/inference/engine.rs` | InferenceEngine implementation |
| `crates/context-graph-embeddings/tests/inference_integration.rs` | Integration tests |

## Files to Modify

| File Path | Change |
|-----------|--------|
| `crates/context-graph-embeddings/src/warm/mod.rs` | Add `pub mod inference;` |
| `crates/context-graph-embeddings/src/warm/loader/operations.rs` | DELETE lines 405-408, UPDATE validate_model() |
| `crates/context-graph-embeddings/src/warm/error.rs` | Add `InferenceInitFailed`, `InferenceFailed` variants |
| `crates/context-graph-embeddings/Cargo.toml` | Add `candle-core`, `candle-nn` dependencies |

---

## Validation Criteria

- [ ] Lines 405-408 in operations.rs are DELETED (no sin-wave)
- [ ] `InferenceEngine` struct exists and compiles
- [ ] `validate_model()` takes `ModelHandle` parameter
- [ ] Tests use real `.safetensors` files, not mock data
- [ ] `cargo test -p context-graph-embeddings` passes
- [ ] Logs contain `[EMB-I015]` success messages
- [ ] Logs do NOT contain forbidden patterns

---

## Test Commands

```bash
cd /home/cabdru/contextgraph

# Verify fake code is deleted
! grep -n "\.sin()" crates/context-graph-embeddings/src/warm/loader/operations.rs

# Build check
cargo check -p context-graph-embeddings

# Run all warm loader tests
cargo test -p context-graph-embeddings warm:: -- --nocapture

# Run inference-specific tests
cargo test -p context-graph-embeddings inference:: -- --nocapture

# Integration test
cargo test -p context-graph-embeddings --test inference_integration -- --nocapture
```

---

## Pseudo Code

```
inference/engine.rs:
  struct InferenceEngine { device, model_id }

  new(model_id, handle):
    device = Device::cuda(handle.device_id())
    return Self { device, model_id }

  run_test_inference(handle, test_input, expected_dim):
    tokens = tokenize(test_input)
    input_tensor = Tensor::new(tokens, device)
    output = forward_pass(handle.vram_ptr(), input_tensor)
    return output.to_vec()

operations.rs validate_model():
  DELETE: output = (0..dim).map(|i| (i * 0.001).sin()).collect()

  ADD:
    engine = InferenceEngine::new(model_id, handle)
    output = engine.run_test_inference(handle, "test text", expected_dim)
    validator.validate_dimensions(model_id, expected_dim, output.len())
    validator.validate_weights_finite_for_model(model_id, &output)
    assert non_zero_count > 0
```

---

## Manual Verification Checklist

After implementation, manually verify:

- [ ] Open `operations.rs` - confirm lines 405-408 are gone
- [ ] Open `inference/engine.rs` - confirm `InferenceEngine` exists
- [ ] Run `cargo test` - confirm all tests pass
- [ ] Check test output logs - confirm `[EMB-I015]` messages appear
- [ ] Search for `.sin()` in warm/ directory - confirm zero matches
- [ ] Run integration test - confirm model validation succeeds with real inference

</task_spec>
