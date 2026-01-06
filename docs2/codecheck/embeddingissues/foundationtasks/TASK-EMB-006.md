# TASK-EMB-006: Create WarmLoadResult Struct

<task_spec id="TASK-EMB-006" version="2.0">

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-006 |
| **Title** | Create Warm Loading Data Types |
| **Status** | complete |
| **Layer** | foundation |
| **Sequence** | 6 |
| **Implements** | REQ-EMB-003 |
| **Depends On** | TASK-EMB-001 |
| **Estimated Complexity** | medium |
| **Constitution Ref** | `stack.gpu`, `AP-007` |
| **Last Audited** | 2026-01-06 |

---

## Context

TECH-EMB-002 specifies real weight loading to replace simulated operations. This task creates the data structures for loaded weights and loading results. Actual loading implementation is Logic Layer.

**Why This Matters:**
- Current warm loading is SIMULATED (fake pointers, fake checksums)
- Constitution AP-007 forbids stub data in production
- Real VRAM allocation is required for inference

---

## Constitution Requirements

**MUST READ:** `/home/cabdru/contextgraph/constitution.yaml`

### Relevant Constitution Sections

```yaml
# From stack.gpu (Line ~45-55)
stack:
  gpu:
    model: RTX_5090
    vram: 32GB
    cuda: "13.1"
    tensor_cores: 512
    memory_bandwidth: "1.8TB/s"

# From AP-007 (Line ~180-185)
AP-007:
  name: "No Stub Data in Production"
  description: "All data structures must contain real, validated data"
  violation: "CRITICAL - System cannot operate with fake data"

# From security (Line ~200-210)
security:
  checksums: SHA256
  validation: "fail-fast on corruption"
```

---

## Codebase Audit (2026-01-06)

### CRITICAL: Existing Types - DO NOT DUPLICATE

| Type | Location | Status |
|------|----------|--------|
| `GpuTensor` | `crates/context-graph-embeddings/src/gpu/tensor/core.rs` | **EXISTS - USE THIS** |
| `DType` | `candle_core::DType` | **EXISTS - USE THIS** |

**IMPORTANT:** The original task specified creating `GpuTensor` and `DType`. These already exist. This task creates ONLY the warm-loading specific types that don't exist.

### Simulated Data Patterns to Replace (Current Violations)

| File | Line | Pattern | Problem |
|------|------|---------|---------|
| `warm/loader/operations.rs` | 126-128 | `0x7f80_0000_0000u64` | FAKE GPU pointer |
| `warm/loader/operations.rs` | 150-156 | `0xDEAD_BEEF_CAFE_BABE` | FAKE checksum |
| `warm/loader/preflight.rs` | 78-85 | `stub_mode: true` | Simulated GPU info |

### Current Warm Loader Structure

```
crates/context-graph-embeddings/src/warm/loader/
├── mod.rs           # Module exports
├── operations.rs    # Contains simulated loading (TO BE REPLACED)
├── preflight.rs     # Contains stub GPU detection (TO BE REPLACED)
└── types.rs         # DOES NOT EXIST - CREATE THIS
```

---

## Input Context Files

| Purpose | File Path | Must Read |
|---------|-----------|-----------|
| Constitution | `/home/cabdru/contextgraph/constitution.yaml` | **YES** |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-002-warm-loading.md` | YES |
| Existing GpuTensor | `crates/context-graph-embeddings/src/gpu/tensor/core.rs` | YES |
| Current operations | `crates/context-graph-embeddings/src/warm/loader/operations.rs` | YES |

---

## Prerequisites

- [ ] TASK-EMB-001 completed (SPARSE_PROJECTED_DIMENSION = 1536)
- [ ] Read constitution.yaml for AP-007 requirements
- [ ] Verify `crate::gpu::GpuTensor` exists (DO NOT DUPLICATE)
- [ ] Verify `candle_core::DType` exists (DO NOT DUPLICATE)

---

## Scope

### In Scope (CREATE NEW)
- `WarmLoadResult` struct - Result of weight loading operation
- `TensorMetadata` struct - SafeTensors header metadata
- `LoadedModelWeights` struct - Complete loaded model state

### Out of Scope (ALREADY EXISTS - DO NOT CREATE)
- `GpuTensor` - Use `crate::gpu::GpuTensor`
- `DType` - Use `candle_core::DType`

### Out of Scope (Logic Layer)
- Actual loading implementations (TASK-EMB-013)
- CUDA calls (TASK-EMB-014)
- SafeTensors parsing

---

## Definition of Done

### Exact Signatures

```rust
// File: crates/context-graph-embeddings/src/warm/loader/types.rs

use std::collections::HashMap;
use std::time::{Duration, Instant};
use candle_core::DType;
use crate::gpu::GpuTensor;

// =============================================================================
// CRITICAL: NO SIMULATION - ALL DATA MUST BE REAL
// Constitution AP-007: "No Stub Data in Production"
// =============================================================================

/// Result of loading a model's weights into GPU memory.
///
/// # Constitution Alignment
/// - REQ-WARM-003: Non-evictable VRAM allocation
/// - REQ-WARM-005: Weight integrity verification
/// - AP-007: No stub data in production
///
/// # CRITICAL: No Simulation
/// All fields contain REAL data from actual loading operations.
/// This struct will PANIC if constructed with invalid data.
#[derive(Debug)]
pub struct WarmLoadResult {
    /// Real GPU device pointer from cudaMalloc.
    /// MUST be non-zero. Zero pointer = PANIC.
    pub gpu_ptr: u64,

    /// Real SHA256 checksum of the weight file.
    /// MUST be non-zero. All-zero checksum = PANIC.
    pub checksum: [u8; 32],

    /// Actual size of weights in GPU memory.
    /// MUST be > 0. Zero size = PANIC.
    pub size_bytes: usize,

    /// Loading duration for performance monitoring.
    pub load_duration: Duration,

    /// Tensor metadata from SafeTensors header.
    pub tensor_metadata: TensorMetadata,
}

impl WarmLoadResult {
    /// Create a new WarmLoadResult with validation.
    ///
    /// # Panics
    /// - If `gpu_ptr` is 0 (null pointer)
    /// - If `checksum` is all zeros (invalid checksum)
    /// - If `size_bytes` is 0 (empty allocation)
    /// - If `tensor_metadata.total_params` is 0 (empty model)
    ///
    /// # Constitution: Fail-Fast
    /// Per AP-007, we panic immediately on invalid data rather than
    /// propagating corruption through the system.
    pub fn new(
        gpu_ptr: u64,
        checksum: [u8; 32],
        size_bytes: usize,
        load_duration: Duration,
        tensor_metadata: TensorMetadata,
    ) -> Self {
        // FAIL-FAST: Null GPU pointer
        assert!(
            gpu_ptr != 0,
            "CONSTITUTION VIOLATION AP-007: gpu_ptr is null (0x0). \
             Real cudaMalloc pointer required."
        );

        // FAIL-FAST: Zero checksum (impossible for real SHA256)
        assert!(
            checksum != [0u8; 32],
            "CONSTITUTION VIOLATION AP-007: checksum is all zeros. \
             Real SHA256 checksum required."
        );

        // FAIL-FAST: Zero size
        assert!(
            size_bytes > 0,
            "CONSTITUTION VIOLATION AP-007: size_bytes is 0. \
             Real allocation size required."
        );

        // FAIL-FAST: Empty model
        assert!(
            tensor_metadata.total_params > 0,
            "CONSTITUTION VIOLATION AP-007: total_params is 0. \
             Real model weights required."
        );

        Self {
            gpu_ptr,
            checksum,
            size_bytes,
            load_duration,
            tensor_metadata,
        }
    }

    /// Verify checksum matches expected value.
    pub fn verify_checksum(&self, expected: &[u8; 32]) -> bool {
        self.checksum == *expected
    }
}

/// Metadata extracted from SafeTensors file header.
///
/// # CRITICAL: No Simulation
/// All fields must reflect actual SafeTensors header content.
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Tensor name -> shape mapping.
    /// Example: {"embeddings.weight": [30522, 768]}
    pub shapes: HashMap<String, Vec<usize>>,

    /// Data type of tensors (from candle_core).
    pub dtype: DType,

    /// Total number of parameters across all tensors.
    /// MUST be > 0 for valid models.
    pub total_params: usize,
}

impl TensorMetadata {
    /// Create new TensorMetadata with validation.
    ///
    /// # Panics
    /// - If `shapes` is empty (no tensors)
    /// - If `total_params` is 0
    pub fn new(
        shapes: HashMap<String, Vec<usize>>,
        dtype: DType,
        total_params: usize,
    ) -> Self {
        assert!(
            !shapes.is_empty(),
            "CONSTITUTION VIOLATION AP-007: shapes is empty. \
             SafeTensors must contain at least one tensor."
        );

        assert!(
            total_params > 0,
            "CONSTITUTION VIOLATION AP-007: total_params is 0. \
             Model must have parameters."
        );

        Self { shapes, dtype, total_params }
    }

    /// Calculate total params from shapes (for verification).
    pub fn calculate_total_params(&self) -> usize {
        self.shapes.values()
            .map(|shape| shape.iter().product::<usize>())
            .sum()
    }

    /// Verify total_params matches calculated value.
    pub fn verify_params(&self) -> bool {
        self.total_params == self.calculate_total_params()
    }
}

/// Complete set of weights for a model loaded into GPU memory.
///
/// # CRITICAL: No Simulation
/// This represents REAL weights loaded into REAL GPU memory.
#[derive(Debug)]
pub struct LoadedModelWeights {
    /// Model identifier (e.g., "E1_Semantic").
    pub model_id: String,

    /// Named tensors loaded to GPU.
    /// Uses existing GpuTensor from crate::gpu module.
    pub tensors: HashMap<String, GpuTensor>,

    /// SHA256 checksum of source weight file.
    pub file_checksum: [u8; 32],

    /// Total GPU memory used (bytes).
    pub total_gpu_bytes: usize,

    /// CUDA device where weights are loaded.
    pub device_id: u32,

    /// Timestamp when weights were loaded.
    pub loaded_at: Instant,
}

impl LoadedModelWeights {
    /// Create new LoadedModelWeights with validation.
    ///
    /// # Panics
    /// - If `model_id` is empty
    /// - If `tensors` is empty
    /// - If `file_checksum` is all zeros
    /// - If `total_gpu_bytes` is 0
    pub fn new(
        model_id: String,
        tensors: HashMap<String, GpuTensor>,
        file_checksum: [u8; 32],
        total_gpu_bytes: usize,
        device_id: u32,
    ) -> Self {
        assert!(
            !model_id.is_empty(),
            "CONSTITUTION VIOLATION AP-007: model_id is empty."
        );

        assert!(
            !tensors.is_empty(),
            "CONSTITUTION VIOLATION AP-007: tensors is empty. \
             Model must have at least one tensor."
        );

        assert!(
            file_checksum != [0u8; 32],
            "CONSTITUTION VIOLATION AP-007: file_checksum is all zeros."
        );

        assert!(
            total_gpu_bytes > 0,
            "CONSTITUTION VIOLATION AP-007: total_gpu_bytes is 0."
        );

        Self {
            model_id,
            tensors,
            file_checksum,
            total_gpu_bytes,
            device_id,
            loaded_at: Instant::now(),
        }
    }

    /// Get a specific tensor by name.
    pub fn get_tensor(&self, name: &str) -> Option<&GpuTensor> {
        self.tensors.get(name)
    }

    /// Check if a tensor exists.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Get all tensor names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }
}

// =============================================================================
// COMPILE-TIME ASSERTIONS
// =============================================================================

/// Compile-time check: Checksum size must be 32 bytes (SHA256)
const _: () = assert!(
    std::mem::size_of::<[u8; 32]>() == 32,
    "Checksum must be exactly 32 bytes for SHA256"
);
```

### mod.rs Modifications

```rust
// File: crates/context-graph-embeddings/src/warm/loader/mod.rs
// ADD this line after existing module declarations:

pub mod types;

pub use types::{
    WarmLoadResult,
    TensorMetadata,
    LoadedModelWeights,
};
```

---

## Files to Create

| File | Content |
|------|---------|
| `crates/context-graph-embeddings/src/warm/loader/types.rs` | Warm loading types (code above) |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-embeddings/src/warm/loader/mod.rs` | Add `pub mod types;` and re-exports |

---

## Full State Verification Protocol

### Source of Truth Locations

| Data | Location | Expected |
|------|----------|----------|
| GpuTensor definition | `crate::gpu::tensor::core::GpuTensor` | EXISTS |
| DType definition | `candle_core::DType` | EXISTS (F32, F16, BF16) |
| WarmLoadResult | `warm/loader/types.rs` | CREATED by this task |
| AP-007 requirement | `constitution.yaml` | No stub data |

### Execute & Inspect Verification

**Step 1: Verify existing types are accessible**
```bash
# GpuTensor should exist
grep -rn "pub struct GpuTensor" crates/context-graph-embeddings/src/gpu/
# Expected: crates/context-graph-embeddings/src/gpu/tensor/core.rs

# DType should be from candle
grep -rn "use candle_core::DType" crates/context-graph-embeddings/src/
```

**Step 2: Compile check**
```bash
cd /home/cabdru/contextgraph
cargo check -p context-graph-embeddings 2>&1 | head -50
# Expected: no errors
```

**Step 3: Verify no simulation patterns in new code**
```bash
# Should return EMPTY (no fake data in types.rs)
grep -E "0x7f80|DEAD_BEEF|simulate|stub" crates/context-graph-embeddings/src/warm/loader/types.rs
```

**Step 4: Run unit tests**
```bash
cargo test -p context-graph-embeddings warm::loader::types --nocapture
```

---

## Edge Case Tests (REQUIRED)

Create these tests in `types.rs` (add `#[cfg(test)]` module):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // FAIL-FAST VALIDATION TESTS
    // =========================================================================

    #[test]
    #[should_panic(expected = "CONSTITUTION VIOLATION AP-007: gpu_ptr is null")]
    fn test_warm_load_result_rejects_null_pointer() {
        let metadata = TensorMetadata::new(
            [("test".to_string(), vec![100, 768])].into_iter().collect(),
            DType::F32,
            76800,
        );
        let _ = WarmLoadResult::new(
            0,  // NULL POINTER - MUST PANIC
            [1u8; 32],
            1024,
            Duration::from_millis(100),
            metadata,
        );
    }

    #[test]
    #[should_panic(expected = "CONSTITUTION VIOLATION AP-007: checksum is all zeros")]
    fn test_warm_load_result_rejects_zero_checksum() {
        let metadata = TensorMetadata::new(
            [("test".to_string(), vec![100, 768])].into_iter().collect(),
            DType::F32,
            76800,
        );
        let _ = WarmLoadResult::new(
            0x7fff_0000_1000,
            [0u8; 32],  // ZERO CHECKSUM - MUST PANIC
            1024,
            Duration::from_millis(100),
            metadata,
        );
    }

    #[test]
    #[should_panic(expected = "CONSTITUTION VIOLATION AP-007: size_bytes is 0")]
    fn test_warm_load_result_rejects_zero_size() {
        let metadata = TensorMetadata::new(
            [("test".to_string(), vec![100, 768])].into_iter().collect(),
            DType::F32,
            76800,
        );
        let _ = WarmLoadResult::new(
            0x7fff_0000_1000,
            [1u8; 32],
            0,  // ZERO SIZE - MUST PANIC
            Duration::from_millis(100),
            metadata,
        );
    }

    #[test]
    #[should_panic(expected = "CONSTITUTION VIOLATION AP-007: shapes is empty")]
    fn test_tensor_metadata_rejects_empty_shapes() {
        let _ = TensorMetadata::new(
            HashMap::new(),  // EMPTY - MUST PANIC
            DType::F32,
            100,
        );
    }

    #[test]
    #[should_panic(expected = "CONSTITUTION VIOLATION AP-007: total_params is 0")]
    fn test_tensor_metadata_rejects_zero_params() {
        let _ = TensorMetadata::new(
            [("test".to_string(), vec![100, 768])].into_iter().collect(),
            DType::F32,
            0,  // ZERO PARAMS - MUST PANIC
        );
    }

    #[test]
    #[should_panic(expected = "CONSTITUTION VIOLATION AP-007: model_id is empty")]
    fn test_loaded_model_weights_rejects_empty_model_id() {
        let tensor = GpuTensor::new(0x7fff_0000_1000, vec![100, 768], 0);
        let _ = LoadedModelWeights::new(
            "".to_string(),  // EMPTY - MUST PANIC
            [("test".to_string(), tensor)].into_iter().collect(),
            [1u8; 32],
            1024,
            0,
        );
    }

    // =========================================================================
    // VALID DATA TESTS
    // =========================================================================

    #[test]
    fn test_warm_load_result_accepts_valid_data() {
        let metadata = TensorMetadata::new(
            [("embeddings.weight".to_string(), vec![30522, 768])].into_iter().collect(),
            DType::F32,
            23_440_896,  // 30522 * 768
        );

        let result = WarmLoadResult::new(
            0x7fff_0000_1000,  // Real-looking pointer
            [0xAB; 32],        // Non-zero checksum
            93_763_584,        // 23M params * 4 bytes
            Duration::from_millis(150),
            metadata,
        );

        assert!(result.gpu_ptr != 0);
        assert!(result.size_bytes > 0);
    }

    #[test]
    fn test_tensor_metadata_calculates_params() {
        let metadata = TensorMetadata::new(
            [
                ("layer1".to_string(), vec![768, 768]),   // 589,824
                ("layer2".to_string(), vec![768, 3072]),  // 2,359,296
            ].into_iter().collect(),
            DType::F32,
            2_949_120,  // Sum of above
        );

        assert!(metadata.verify_params());
        assert_eq!(metadata.calculate_total_params(), 2_949_120);
    }

    #[test]
    fn test_checksum_verification() {
        let expected = [0xAB; 32];
        let metadata = TensorMetadata::new(
            [("test".to_string(), vec![100])].into_iter().collect(),
            DType::F32,
            100,
        );

        let result = WarmLoadResult::new(
            0x7fff_0000_1000,
            expected,
            400,
            Duration::from_millis(10),
            metadata,
        );

        assert!(result.verify_checksum(&expected));
        assert!(!result.verify_checksum(&[0xCD; 32]));
    }
}
```

---

## Evidence of Success

After implementation, provide this verification log:

```
[VERIFICATION LOG - TASK-EMB-006]
Date: YYYY-MM-DD HH:MM:SS
Implementer: [agent-id]

1. CONSTITUTION READ:
   - constitution.yaml parsed: YES
   - AP-007 requirement identified: "No Stub Data in Production"
   - stack.gpu requirements noted: RTX_5090, 32GB VRAM

2. EXISTING TYPES AUDIT:
   - GpuTensor found at: crates/context-graph-embeddings/src/gpu/tensor/core.rs
   - DType source: candle_core::DType
   - NO DUPLICATES CREATED: CONFIRMED

3. NEW TYPES CREATED:
   - WarmLoadResult: YES (with fail-fast validation)
   - TensorMetadata: YES (with fail-fast validation)
   - LoadedModelWeights: YES (with fail-fast validation)

4. COMPILATION:
   $ cargo check -p context-graph-embeddings
   Result: SUCCESS (no errors)

5. TESTS:
   $ cargo test -p context-graph-embeddings warm::loader::types
   - test_warm_load_result_rejects_null_pointer: PASSED (panic as expected)
   - test_warm_load_result_rejects_zero_checksum: PASSED (panic as expected)
   - test_warm_load_result_rejects_zero_size: PASSED (panic as expected)
   - test_tensor_metadata_rejects_empty_shapes: PASSED (panic as expected)
   - test_tensor_metadata_rejects_zero_params: PASSED (panic as expected)
   - test_loaded_model_weights_rejects_empty_model_id: PASSED (panic as expected)
   - test_warm_load_result_accepts_valid_data: PASSED
   - test_tensor_metadata_calculates_params: PASSED
   - test_checksum_verification: PASSED

6. NO SIMULATION PATTERNS:
   $ grep -E "0x7f80|DEAD_BEEF|simulate|stub" warm/loader/types.rs
   Result: EMPTY (no fake data)

7. MODULE EXPORTS:
   - pub mod types; added to mod.rs: YES
   - Re-exports configured: YES
```

---

## Anti-Patterns to Avoid

| Pattern | Problem | Correct Approach |
|---------|---------|------------------|
| `0x7f80_0000_0000` | Fake GPU pointer | Panic on 0x0, accept any non-zero |
| `0xDEAD_BEEF_CAFE_BABE` | Fake checksum | Panic on all-zero, accept any non-zero |
| `simulate_weight_loading()` | No actual loading | Types that REQUIRE real data |
| Creating duplicate `GpuTensor` | Wastes code, causes confusion | Use `crate::gpu::GpuTensor` |
| Creating duplicate `DType` | Wastes code, causes confusion | Use `candle_core::DType` |
| Accepting empty HashMap | Silent corruption | Panic immediately |
| Accepting zero size | Silent corruption | Panic immediately |

---

## Traceability

| Requirement | Tech Spec | Constitution | Code Location |
|-------------|-----------|--------------|---------------|
| REQ-EMB-003 | TECH-EMB-002 | stack.gpu | `WarmLoadResult.gpu_ptr` |
| REQ-WARM-003 | TECH-EMB-002 | stack.gpu.vram | `LoadedModelWeights.total_gpu_bytes` |
| REQ-WARM-005 | TECH-EMB-002 | security.checksums | `WarmLoadResult.checksum` |
| AP-007 | - | AP-007 | All fail-fast assertions |

---

## Post-Implementation Checklist

- [ ] Read `constitution.yaml` for AP-007 requirements
- [ ] Verified `crate::gpu::GpuTensor` exists (not duplicated)
- [ ] Verified `candle_core::DType` exists (not duplicated)
- [ ] `types.rs` created with fail-fast validation
- [ ] All panic tests pass (6 panic tests)
- [ ] All valid data tests pass (3+ tests)
- [ ] `cargo check -p context-graph-embeddings` succeeds
- [ ] No simulation patterns in new code
- [ ] Module exports added to `mod.rs`
- [ ] Verification log completed with timestamps
- [ ] Edge cases documented with before/after state

</task_spec>
