# TASK-EMB-014: Replace Fake VRAM Pointers with Real CUDA Allocations

<task_spec id="TASK-EMB-014" version="3.0">

## Metadata

| Field | Value |
|-------|-------|
| **Title** | Replace Fake VRAM Pointers with Real CUDA cudaMalloc Allocations |
| **Status** | **COMPLETE** |
| **Layer** | logic |
| **Sequence** | 14 |
| **Implements** | REQ-WARM-005 (VRAM Allocation) |
| **Depends On** | TASK-EMB-013 (COMPLETE - Real weight loading exists) |
| **Estimated Complexity** | high |
| **Updated** | 2026-01-06 |
| **Completed** | 2026-01-06 |
| **Codebase Audit** | VERIFIED |

---

## CRITICAL: Codebase Audit Summary (2026-01-06)

**This section documents the ACTUAL current state of the codebase.**

### What EXISTS and Must Be DELETED

| Component | File Path | Line | What To Delete |
|-----------|-----------|------|----------------|
| Fake pointer generation | `src/warm/loader/operations.rs` | 149-151 | `0x7f80_0000_0000 + offset` pattern |
| Fake validation output | `src/warm/loader/operations.rs` | 331-332 | `(i as f32 * 0.001).sin()` fake inference output |

### Current Broken Code (VERIFIED - Lines 147-164)

```rust
// FILE: crates/context-graph-embeddings/src/warm/loader/operations.rs
// The allocate_model_vram function at lines 128-164

pub fn allocate_model_vram(
    model_id: &str,
    size_bytes: usize,
    memory_pools: &mut WarmMemoryPools,
) -> WarmResult<u64> {
    // ... capacity check is good ...

    // PROBLEM: These 3 lines generate FAKE pointers
    let base_ptr = 0x7f80_0000_0000u64;  // LINE 149 - FAKE!
    let offset = memory_pools.list_model_allocations().len() as u64 * 0x1_0000_0000;  // LINE 150
    let vram_ptr = base_ptr + offset;  // LINE 151 - FAKE pointer!

    // This records the FAKE pointer
    memory_pools.allocate_model(model_id, size_bytes, vram_ptr)?;
    // ...
}
```

### What EXISTS and Must Be USED

| Component | File Path | Status |
|-----------|-----------|--------|
| WarmCudaAllocator struct | `src/warm/cuda_alloc/allocator.rs` | EXISTS - Has `allocate_protected()` method |
| VramAllocation struct | `src/warm/cuda_alloc/allocation.rs` | EXISTS - Tracks real CUDA pointers |
| allocate_protected() | `src/warm/cuda_alloc/allocator_cuda.rs` | EXISTS - Uses candle_core Tensor allocation |
| GpuInfo struct | `src/warm/cuda_alloc/gpu_info.rs` | EXISTS |
| ModelMemoryPool | `src/warm/memory_pool/model_pool.rs` | EXISTS - Records allocations |
| load_weights() | `src/warm/loader/operations.rs` | EXISTS (TASK-EMB-013 COMPLETE) |
| WarmError variants | `src/warm/error.rs` | EXISTS - CudaAllocFailed, VramAllocationFailed |

---

## Context

### Why This Task Exists

Constitution AP-007 FORBIDS stub data in production. The current code generates fake VRAM pointers instead of calling real CUDA allocation APIs.

**Fake Pattern Being Removed:**
```rust
let base_ptr = 0x7f80_0000_0000u64;
let offset = memory_pools.list_model_allocations().len() as u64 * 0x1_0000_0000;
let vram_ptr = base_ptr + offset;  // NOT A REAL GPU POINTER!
```

**Real Pattern Required:**
```rust
let allocation = cuda_allocator.allocate_protected(size_bytes)?;
let vram_ptr = allocation.ptr;  // REAL cudaMalloc pointer!
```

### What This Task Does

1. **UPDATE** `allocate_model_vram()` to use `WarmCudaAllocator::allocate_protected()`
2. **ADD** `WarmCudaAllocator` parameter to the loading pipeline
3. **DELETE** fake pointer generation lines
4. **VERIFY** allocations return real CUDA pointers

### What This Task Does NOT Do

- Weight file loading (TASK-EMB-013 - COMPLETE)
- Inference validation fix (TASK-EMB-015 - next task)
- Stub mode removal from preflight (TASK-EMB-019)

---

## Input Context Files

| Purpose | File Path | What To Read |
|---------|-----------|--------------|
| Fake pointer code | `crates/context-graph-embeddings/src/warm/loader/operations.rs` | Lines 128-164 |
| CUDA allocator | `crates/context-graph-embeddings/src/warm/cuda_alloc/allocator_cuda.rs` | `allocate_protected()` |
| Allocation struct | `crates/context-graph-embeddings/src/warm/cuda_alloc/allocation.rs` | `VramAllocation` |
| Memory pools | `crates/context-graph-embeddings/src/warm/memory_pool/mod.rs` | Pool interface |
| WarmError | `crates/context-graph-embeddings/src/warm/error.rs` | Error variants |
| Constitution | `docs2/constitution.yaml` | AP-007, stack.gpu, perf.memory |

---

## Prerequisites

- [x] TASK-EMB-013 completed (`load_weights()` reads real files)
- [x] `WarmCudaAllocator` struct exists at `src/warm/cuda_alloc/allocator.rs`
- [x] `allocate_protected()` method exists at `src/warm/cuda_alloc/allocator_cuda.rs`
- [x] `VramAllocation` struct exists at `src/warm/cuda_alloc/allocation.rs`
- [ ] CUDA-capable GPU available (RTX 5090 or compatible for testing)

---

## Scope

### In Scope

1. **UPDATE** `allocate_model_vram()` to accept a `WarmCudaAllocator` reference
2. **DELETE** fake pointer generation (lines 149-151)
3. **CALL** `cuda_allocator.allocate_protected(size_bytes)`
4. **RETURN** real `VramAllocation.ptr` value
5. **UPDATE** `load_single_model()` to pass allocator
6. **UPDATE** `WarmLoader` engine to hold allocator instance
7. **UPDATE** tests to verify real CUDA pointers

### Out of Scope

- Weight loading (TASK-EMB-013 - COMPLETE)
- Inference validation (TASK-EMB-015)
- Working memory pool allocation (separate concern)

---

## Definition of Done

### Step 1: Update allocate_model_vram Function Signature

**File**: `crates/context-graph-embeddings/src/warm/loader/operations.rs`

Change the function signature to accept a CUDA allocator:

```rust
// OLD signature (DELETE):
pub fn allocate_model_vram(
    model_id: &str,
    size_bytes: usize,
    memory_pools: &mut WarmMemoryPools,
) -> WarmResult<u64>

// NEW signature (ADD):
use crate::warm::cuda_alloc::WarmCudaAllocator;

pub fn allocate_model_vram(
    model_id: &str,
    size_bytes: usize,
    memory_pools: &mut WarmMemoryPools,
    cuda_allocator: &mut WarmCudaAllocator,
) -> WarmResult<u64>
```

### Step 2: DELETE Fake Pointer Generation

**File**: `crates/context-graph-embeddings/src/warm/loader/operations.rs`

Delete lines 149-151:
```rust
// DELETE THESE LINES:
let base_ptr = 0x7f80_0000_0000u64;
let offset = memory_pools.list_model_allocations().len() as u64 * 0x1_0000_0000;
let vram_ptr = base_ptr + offset;
```

### Step 3: ADD Real CUDA Allocation

**File**: `crates/context-graph-embeddings/src/warm/loader/operations.rs`

Replace with real allocation:

```rust
/// Allocate VRAM for a model from the model pool using CUDA.
///
/// # CRITICAL: Real cudaMalloc
/// This function uses real CUDA memory allocation via WarmCudaAllocator.
/// Fake pointers (0x7f80...) are FORBIDDEN per Constitution AP-007.
///
/// # Arguments
/// * `model_id` - Model identifier for error messages
/// * `size_bytes` - Size to allocate in bytes
/// * `memory_pools` - Pool tracking for accounting
/// * `cuda_allocator` - CUDA allocator for real GPU memory
///
/// # Returns
/// * `Ok(vram_ptr)` - Real CUDA device pointer from cudaMalloc
/// * `Err(WarmError)` - If allocation fails
///
/// # Errors
/// - `WarmError::VramAllocationFailed` - Pool capacity exceeded
/// - `WarmError::CudaAllocFailed` - CUDA allocation failed
pub fn allocate_model_vram(
    model_id: &str,
    size_bytes: usize,
    memory_pools: &mut WarmMemoryPools,
    cuda_allocator: &mut WarmCudaAllocator,
) -> WarmResult<u64> {
    // Check if we have enough space in the model pool
    if memory_pools.available_model_bytes() < size_bytes {
        return Err(WarmError::VramAllocationFailed {
            requested_bytes: size_bytes,
            available_bytes: memory_pools.available_model_bytes(),
            error: format!(
                "Model pool exhausted: {} bytes requested, {} bytes available",
                size_bytes,
                memory_pools.available_model_bytes()
            ),
        });
    }

    // Allocate REAL VRAM via cudaMalloc (non-evictable)
    let allocation = cuda_allocator.allocate_protected(size_bytes).map_err(|e| {
        tracing::error!(
            "[EMB-E008] CUDA allocation failed for {}: {} bytes - {}",
            model_id,
            size_bytes,
            e
        );
        e
    })?;

    let vram_ptr = allocation.ptr;

    // Verify allocation is valid (non-null pointer)
    if vram_ptr == 0 {
        return Err(WarmError::CudaAllocFailed {
            requested_bytes: size_bytes,
            cuda_error: "cudaMalloc returned null pointer".to_string(),
            vram_free: cuda_allocator.query_available_vram().ok(),
            allocation_history: cuda_allocator.allocation_history().to_vec(),
        });
    }

    // Record allocation in memory pool for accounting
    memory_pools.allocate_model(model_id, size_bytes, vram_ptr)?;

    tracing::info!(
        "Allocated {} bytes for {} at 0x{:016x} (REAL cudaMalloc)",
        size_bytes,
        model_id,
        vram_ptr
    );

    Ok(vram_ptr)
}
```

### Step 4: Update load_single_model to Pass Allocator

**File**: `crates/context-graph-embeddings/src/warm/loader/operations.rs`

Update the `load_single_model` function signature and call:

```rust
// OLD signature:
pub fn load_single_model(
    model_id: &str,
    config: &WarmConfig,
    registry: &SharedWarmRegistry,
    memory_pools: &mut WarmMemoryPools,
    validator: &WarmValidator,
) -> WarmResult<()>

// NEW signature:
pub fn load_single_model(
    model_id: &str,
    config: &WarmConfig,
    registry: &SharedWarmRegistry,
    memory_pools: &mut WarmMemoryPools,
    cuda_allocator: &mut WarmCudaAllocator,
    validator: &WarmValidator,
) -> WarmResult<()>
```

Update the call to `allocate_model_vram` inside `load_single_model`:

```rust
// OLD call (line 57):
let vram_ptr = allocate_model_vram(model_id, expected_bytes, memory_pools)?;

// NEW call:
let vram_ptr = allocate_model_vram(model_id, expected_bytes, memory_pools, cuda_allocator)?;
```

### Step 5: Update WarmLoader Engine

**File**: `crates/context-graph-embeddings/src/warm/loader/engine.rs`

Add `WarmCudaAllocator` field to the loader and update initialization.

Check current engine structure and add allocator field:

```rust
// Add import at top:
use crate::warm::cuda_alloc::WarmCudaAllocator;

// In WarmLoader struct, add field:
pub struct WarmLoader {
    // ... existing fields ...
    cuda_allocator: Option<WarmCudaAllocator>,
}

// In new() or init(), initialize allocator:
let cuda_allocator = WarmCudaAllocator::new(config.cuda_device_id)?;
```

Update any methods that call `load_single_model` to pass the allocator.

### Step 6: Update mod.rs Exports

**File**: `crates/context-graph-embeddings/src/warm/loader/mod.rs`

Ensure the updated function signature is re-exported correctly.

### Step 7: Update Tests

**File**: `crates/context-graph-embeddings/src/warm/loader/tests/loader_tests.rs`

Delete any tests that use fake pointers and add real tests:

```rust
#[test]
fn test_allocate_model_vram_real_cuda() {
    // This test requires CUDA hardware
    // Skip if not available
    let cuda_result = WarmCudaAllocator::new(0);
    if cuda_result.is_err() {
        eprintln!("Skipping test: CUDA not available");
        return;
    }
    let mut cuda_allocator = cuda_result.unwrap();

    let mut memory_pools = WarmMemoryPools::new(
        24 * 1024 * 1024 * 1024,  // 24GB model pool
        8 * 1024 * 1024 * 1024,   // 8GB working pool
    );

    println!("=== BEFORE: VRAM allocation test ===");
    println!("Model pool available: {} bytes", memory_pools.available_model_bytes());
    println!("CUDA allocator total: {} bytes", cuda_allocator.total_allocated());

    let result = allocate_model_vram(
        "test_model",
        100_000_000,  // 100MB
        &mut memory_pools,
        &mut cuda_allocator,
    );

    println!("=== AFTER: VRAM allocation test ===");
    println!("Result: {:?}", result);

    assert!(result.is_ok());
    let ptr = result.unwrap();

    // CRITICAL: Verify this is a REAL pointer, not fake
    assert_ne!(ptr, 0, "Pointer must not be null");
    assert!(ptr < 0x7f80_0000_0000 || ptr > 0x7fff_ffff_ffff,
        "Pointer 0x{:x} appears to be a fake pattern", ptr);

    // Verify allocation was recorded
    assert!(memory_pools.model_pool_contains("test_model"));

    println!("Allocated REAL pointer: 0x{:016x}", ptr);
}

#[test]
fn test_allocate_model_vram_insufficient_capacity() {
    let cuda_result = WarmCudaAllocator::new(0);
    if cuda_result.is_err() {
        eprintln!("Skipping test: CUDA not available");
        return;
    }
    let mut cuda_allocator = cuda_result.unwrap();

    // Create a pool with only 1MB capacity
    let mut memory_pools = WarmMemoryPools::new(
        1024 * 1024,  // 1MB model pool
        1024 * 1024,  // 1MB working pool
    );

    let result = allocate_model_vram(
        "large_model",
        100_000_000,  // Request 100MB (exceeds 1MB pool)
        &mut memory_pools,
        &mut cuda_allocator,
    );

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), WarmError::VramAllocationFailed { .. }));
}
```

---

## Files to Modify

| File | Change |
|------|--------|
| `src/warm/loader/operations.rs` | UPDATE `allocate_model_vram` signature and body |
| `src/warm/loader/operations.rs` | UPDATE `load_single_model` to pass allocator |
| `src/warm/loader/engine.rs` | ADD `WarmCudaAllocator` field, UPDATE methods |
| `src/warm/loader/mod.rs` | UPDATE exports if needed |
| `src/warm/loader/tests/loader_tests.rs` | UPDATE tests to use real allocator |

---

## Validation Criteria

### Compilation Checks

- [ ] `cargo check -p context-graph-embeddings` passes
- [ ] `grep -rn "0x7f80_0000_0000" crates/context-graph-embeddings/src/warm/loader/` returns ZERO results
- [ ] `grep -rn "base_ptr.*offset" crates/context-graph-embeddings/src/warm/loader/` returns ZERO results

### Functional Checks

- [ ] `allocate_model_vram()` accepts `WarmCudaAllocator` parameter
- [ ] `allocate_model_vram()` calls `cuda_allocator.allocate_protected()`
- [ ] Returned pointer is from CUDA, not fake pattern
- [ ] Memory pool accounting works correctly
- [ ] Errors include CUDA diagnostic information

### Test Commands

```bash
cd /home/cabdru/contextgraph

# Step 1: Verify fake pointers are gone
grep -rn "0x7f80_0000_0000" crates/context-graph-embeddings/src/warm/loader/
# Expected: NO OUTPUT (zero matches)

# Step 2: Verify fake offset pattern is gone
grep -rn "base_ptr.*offset" crates/context-graph-embeddings/src/warm/loader/
# Expected: NO OUTPUT (zero matches)

# Step 3: Run compilation check
cargo check -p context-graph-embeddings

# Step 4: Run tests (requires CUDA)
cargo test -p context-graph-embeddings warm::loader::allocate_model_vram -- --nocapture
```

---

## CRITICAL: Full State Verification

### Source of Truth

The source of truth for this task is:
1. **CUDA Device Memory**: Real GPU memory allocated via cudaMalloc
2. **WarmCudaAllocator.total_allocated()**: Tracks allocation count
3. **Memory Pool Allocations**: `memory_pools.list_model_allocations()`
4. **Pointer Value**: Must be real CUDA pointer, not fake pattern

### Execute & Inspect Protocol

After implementing, you MUST:

1. **Initialize CUDA allocator**:
```rust
let mut cuda_allocator = WarmCudaAllocator::new(0)?;
println!("GPU: {:?}", cuda_allocator.get_gpu_info()?);
```

2. **Allocate memory**:
```rust
let ptr = allocate_model_vram(
    "E1_Semantic",
    1_000_000_000,  // 1GB
    &mut memory_pools,
    &mut cuda_allocator,
)?;
```

3. **Verify the pointer is REAL**:
```rust
// Fake pattern was: 0x7f80_0000_0000 + offset
// Real pointers will be different
assert!(ptr != 0);
assert!(ptr < 0x7f80_0000_0000 || ptr > 0x7fff_ffff_ffff,
    "Pointer appears to be fake pattern");
```

4. **Verify CUDA tracking**:
```rust
// Check allocator tracked the allocation
assert!(cuda_allocator.total_allocated() >= 1_000_000_000);
```

5. **Verify pointer is valid with CUDA query**:
```rust
// The allocator has a verify method
assert!(cuda_allocator.verify_allocation(ptr)?);
```

### Boundary & Edge Case Audit

You MUST manually test these 3 edge cases and print state before/after:

#### Edge Case 1: Zero-byte Allocation (Should Fail)

```rust
#[test]
fn test_edge_case_zero_allocation() {
    let mut cuda_allocator = WarmCudaAllocator::new(0).unwrap();
    let mut memory_pools = WarmMemoryPools::new(24 * GB, 8 * GB);

    println!("=== BEFORE: Zero allocation test ===");
    println!("Total allocated: {}", cuda_allocator.total_allocated());

    let result = allocate_model_vram(
        "zero_model",
        0,
        &mut memory_pools,
        &mut cuda_allocator,
    );

    println!("=== AFTER: Zero allocation test ===");
    println!("Result: {:?}", result);

    // Zero allocation should either fail or return valid minimal allocation
    // Document actual behavior
}
```

#### Edge Case 2: Maximum Single Allocation (Near VRAM Limit)

```rust
#[test]
fn test_edge_case_large_allocation() {
    let mut cuda_allocator = WarmCudaAllocator::new(0).unwrap();
    let gpu_info = cuda_allocator.get_gpu_info().unwrap();
    let total_vram = gpu_info.total_memory_bytes;

    // Try to allocate 90% of VRAM
    let request_size = (total_vram as f64 * 0.9) as usize;
    let mut memory_pools = WarmMemoryPools::new(total_vram, 0);

    println!("=== BEFORE: Large allocation test ===");
    println!("Total VRAM: {} bytes", total_vram);
    println!("Requesting: {} bytes (90%)", request_size);
    println!("Available: {} bytes", cuda_allocator.query_available_vram().unwrap());

    let result = allocate_model_vram(
        "large_model",
        request_size,
        &mut memory_pools,
        &mut cuda_allocator,
    );

    println!("=== AFTER: Large allocation test ===");
    println!("Result: {:?}", result);
    println!("Now allocated: {} bytes", cuda_allocator.total_allocated());
}
```

#### Edge Case 3: Allocation After Pool Exhaustion

```rust
#[test]
fn test_edge_case_pool_exhaustion() {
    let mut cuda_allocator = WarmCudaAllocator::new(0).unwrap();
    // Create pool with exactly 1GB
    let mut memory_pools = WarmMemoryPools::new(1 * GB, 0);

    println!("=== BEFORE: Pool exhaustion test ===");
    println!("Pool capacity: {} bytes", memory_pools.available_model_bytes());

    // First allocation: 800MB (should succeed)
    let result1 = allocate_model_vram("model1", 800 * MB, &mut memory_pools, &mut cuda_allocator);
    println!("First allocation (800MB): {:?}", result1.is_ok());

    // Second allocation: 300MB (should FAIL - only 200MB left)
    let result2 = allocate_model_vram("model2", 300 * MB, &mut memory_pools, &mut cuda_allocator);

    println!("=== AFTER: Pool exhaustion test ===");
    println!("Second allocation (300MB): {:?}", result2);

    assert!(result1.is_ok());
    assert!(result2.is_err());
    assert!(matches!(result2.unwrap_err(), WarmError::VramAllocationFailed { .. }));
}
```

### Evidence of Success

After running all tests, you MUST provide a log showing:

```
=== PHYSICAL EVIDENCE: allocate_model_vram() Implementation ===

1. Fake pointers DELETED:
   $ grep -rn "0x7f80_0000_0000" crates/context-graph-embeddings/src/warm/loader/
   (no output)

2. Real CUDA allocation verified:
   Model: E1_Semantic
   Size: 1,000,000,000 bytes
   Pointer: 0x7f1234567890  (NOT in fake range 0x7f80_xxxx_xxxx)

3. CUDA tracking verified:
   cuda_allocator.total_allocated() = 1,000,000,000 bytes
   cuda_allocator.verify_allocation(ptr) = true

4. Memory pool accounting verified:
   memory_pools.model_pool_contains("E1_Semantic") = true
   memory_pools.get_model_allocation("E1_Semantic").size_bytes = 1,000,000,000

5. Tests pass:
   $ cargo test -p context-graph-embeddings warm::loader::allocate
   test warm::loader::test_allocate_model_vram_real_cuda ... ok
   test warm::loader::test_allocate_model_vram_insufficient_capacity ... ok
   test warm::loader::test_edge_case_zero_allocation ... ok
   test warm::loader::test_edge_case_large_allocation ... ok
   test warm::loader::test_edge_case_pool_exhaustion ... ok
```

---

## Constraints

### MUST DO

- Use `WarmCudaAllocator::allocate_protected()` for all model VRAM
- Return REAL CUDA pointers from cudaMalloc
- Verify pointer is non-null before returning
- Log allocation details including pointer value
- Handle CUDA errors with detailed diagnostics
- Update all callers to pass allocator

### MUST NOT DO

- Generate fake pointers (0x7f80...)
- Use offset calculations for pointer generation
- Fall back to fake allocation if CUDA fails
- Create backwards compatibility shims
- Mock CUDA in production code

---

## Related Tasks

| Task | Relationship |
|------|--------------|
| TASK-EMB-013 | Depends on (provides real weight loading) |
| TASK-EMB-015 | Next task (inference validation fix) |
| TASK-EMB-019 | Related (stub mode removal) |

---

## Troubleshooting

### "CudaInitFailed" when running tests

**Cause**: No CUDA-capable GPU detected or CUDA drivers not installed.

**Fix**:
```bash
# Check CUDA availability
nvidia-smi
# If not available, tests requiring CUDA should be skipped gracefully
```

### "CudaAllocFailed" with sufficient VRAM

**Cause**: VRAM fragmentation or concurrent allocations.

**Fix**:
```bash
# Clear GPU memory
nvidia-smi --gpu-reset
# Or check for other processes using GPU
nvidia-smi -q -d MEMORY
```

### Pointer verification fails (looks like fake pattern)

**Cause**: Old code path still being called.

**Fix**: Verify ALL call sites of `allocate_model_vram` pass the `cuda_allocator` parameter. Search for any remaining calls with old signature.

---

## CRITICAL: Manual Verification Requirement

**Trigger Event**: Call to `allocate_model_vram()`

**Process X**: CUDA allocation via `allocate_protected()` → Candle tensor creation → Memory recorded in allocator

**Outcome Y**: Real CUDA pointer stored in memory pool

**Physical Verification**:
1. The pointer value MUST NOT match the fake pattern `0x7f80_xxxx_xxxx + offset`
2. `cuda_allocator.total_allocated()` MUST increase by the allocation size
3. `memory_pools.get_model_allocation(model_id)` MUST return the allocation
4. On systems with CUDA, `nvidia-smi` should show increased memory usage

</task_spec>
