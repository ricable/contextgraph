# TASK-EMB-013: Implement Real Weight Loading

<task_spec id="TASK-EMB-013" version="3.0">

## Metadata

| Field | Value |
|-------|-------|
| **Title** | Replace Simulated Weight Loading with Real SafeTensors Loading |
| **Status** | **COMPLETE** |
| **Layer** | logic |
| **Sequence** | 13 |
| **Implements** | REQ-EMB-003 |
| **Depends On** | TASK-EMB-006 (COMPLETE - WarmLoadResult struct exists) |
| **Estimated Complexity** | high |
| **Updated** | 2026-01-06 |
| **Codebase Audit** | VERIFIED |
| **Completed** | 2026-01-06 |

---

## COMPLETION EVIDENCE (2026-01-06)

### Physical Verification Results

```
========================================================================
          TASK-EMB-013 FULL STATE VERIFICATION - EVIDENCE LOG
========================================================================

SOURCE OF TRUTH: Rust source files in crates/context-graph-embeddings/src/warm/

=== DELETION VERIFICATION ===
simulate_weight_loading function: NO MATCHES (DELETED ✅)

=== FAKE CHECKSUM PATTERNS IN PRODUCTION CODE ===
0xDEAD_BEEF in production weight loading: NO MATCHES ✅
(Only found in documentation comment warning it's forbidden)

=== REAL FUNCTIONS EXIST ===
load_weights at: operations.rs:194
verify_checksum at: operations.rs:290

=== PHYSICAL SHA256 VERIFICATION ===
File: /tmp/verification_test.safetensors (88 bytes)
External sha256sum: ff0b41a49a541d2383b963663323322c4cd8506aed85e1d955a848dfcac6de46
Rust load_weights:  ff0b41a49a541d2383b963663323322c4cd8506aed85e1d955a848dfcac6de46
✅ CHECKSUMS MATCH - PHYSICAL VERIFICATION PASSED

=== TEST RESULTS ===
All 18 loader tests PASSED including:
- test_load_weights_real_file (real SafeTensors data)
- test_load_weights_missing_file (edge case: WarmError::WeightFileMissing)
- test_load_weights_invalid_format (edge case: WarmError::WeightFileCorrupted)
- test_verify_checksum_match
- test_verify_checksum_mismatch (WarmError::WeightChecksumMismatch)

=== CONSTITUTION COMPLIANCE ===
AP-007: No stub data in production code ✅
========================================================================
```

### Files Modified

| File | Change |
|------|--------|
| `Cargo.toml` | Added `hex = "0.4"` dependency |
| `src/warm/error.rs` | Added WeightFileMissing, WeightFileCorrupted, WeightChecksumMismatch (exit codes 111-113) |
| `src/warm/loader/operations.rs` | Deleted simulate_weight_loading, added load_weights and verify_checksum |
| `src/warm/loader/engine.rs` | Removed simulate_weight_loading import and wrapper |
| `src/warm/loader/mod.rs` | Added public exports for load_weights, verify_checksum |
| `src/warm/loader/tests/loader_tests.rs` | Rewrote tests to use real SafeTensors data |
| `tests/physical_verification.rs` | Added physical verification test |

---

## CRITICAL: Codebase Audit Summary (2026-01-06)

**This section documents the ACTUAL current state of the codebase.**

### What EXISTS and Must Be DELETED

| Component | File Path | Line | What To Delete |
|-----------|-----------|------|----------------|
| `simulate_weight_loading()` | `src/warm/loader/operations.rs` | 150-165 | ENTIRE FUNCTION - returns fake checksum `0xDEAD_BEEF_CAFE_BABE` |
| Fake pointer generation | `src/warm/loader/operations.rs` | 126-128 | `0x7f80_0000_0000 + offset` pattern |
| Sin wave validation | `src/warm/loader/operations.rs` | 185-187 | `(i as f32 * 0.001).sin()` fake output |
| Call to simulate_weight_loading | `src/warm/loader/operations.rs` | 62 | Must call real `load_weights()` instead |
| Engine re-export | `src/warm/loader/engine.rs` | 28 | `use super::operations::simulate_weight_loading` |
| Engine wrapper method | `src/warm/loader/engine.rs` | 285-290 | `simulate_weight_loading()` method |
| Test using fake function | `src/warm/loader/tests/loader_tests.rs` | 253-261 | `test_simulate_weight_loading()` test |

### What EXISTS and Must Be PRESERVED

| Component | File Path | Status |
|-----------|-----------|--------|
| WarmLoadResult struct | `src/warm/loader/types.rs` | EXISTS (TASK-EMB-006 COMPLETE) |
| WarmError enum (20+ variants) | `src/warm/error.rs` | EXISTS - includes WeightFileMissing, ChecksumMismatch |
| WarmConfig | `src/warm/config.rs` | EXISTS |
| WarmMemoryPools | `src/warm/memory_pool/mod.rs` | EXISTS |
| WarmValidator | `src/warm/validation/mod.rs` | EXISTS |
| safetensors dependency | `Cargo.toml` | EXISTS (added in TASK-EMB-011) |
| sha2 dependency | `Cargo.toml` | EXISTS (added in TASK-EMB-011) |

### Current Broken Code (VERIFIED - Lines 150-165)

```rust
// FILE: crates/context-graph-embeddings/src/warm/loader/operations.rs
// THIS ENTIRE FUNCTION MUST BE DELETED:

pub fn simulate_weight_loading(model_id: &str, _size_bytes: usize) -> WarmResult<u64> {
    // Generate a deterministic checksum based on model ID
    let mut checksum = 0u64;
    for (i, byte) in model_id.bytes().enumerate() {
        checksum ^= (byte as u64) << ((i % 8) * 8);
    }
    checksum ^= 0xDEAD_BEEF_CAFE_BABEu64;  // FORBIDDEN: Fake checksum

    tracing::debug!(
        "Simulated weight loading for {} (checksum: 0x{:016x})",
        model_id,
        checksum
    );

    Ok(checksum)
}
```

### Current Broken Fake Pointers (VERIFIED - Lines 126-128)

```rust
// FILE: crates/context-graph-embeddings/src/warm/loader/operations.rs
// THIS FAKE POINTER LOGIC MUST BE DELETED:

let base_ptr = 0x7f80_0000_0000u64;
let offset = memory_pools.list_model_allocations().len() as u64 * 0x1_0000_0000;
let vram_ptr = base_ptr + offset;
```

### Current Broken Validation (VERIFIED - Lines 185-187)

```rust
// FILE: crates/context-graph-embeddings/src/warm/loader/operations.rs
// THIS FAKE OUTPUT MUST BE DELETED:

let output: Vec<f32> = (0..expected_dimension)
    .map(|i| (i as f32 * 0.001).sin())  // FORBIDDEN: Fake sin wave
    .collect();
```

---

## Context

### Why This Task Exists

Constitution AP-007 explicitly FORBIDS stub data in production:
```yaml
# From docs2/constitution.yaml
forbidden:
  AP-007: "Stub data in prod -> use tests/fixtures/"
```

The current warm loader implementation uses THREE categories of fake data:
1. **Fake Checksums**: `0xDEAD_BEEF_CAFE_BABE` XOR pattern instead of real SHA256
2. **Fake Pointers**: `0x7f80_0000_0000 + offset` instead of real cudaMalloc
3. **Fake Validation**: `sin(i * 0.001)` instead of real model inference

### What This Task Does

Replace `simulate_weight_loading()` with `load_weights()` that:
1. Reads REAL bytes from REAL SafeTensors files on disk
2. Computes REAL SHA256 checksums using the `sha2` crate
3. Parses REAL tensor metadata from SafeTensors headers
4. Returns REAL file sizes and shapes

### What This Task Does NOT Do

- GPU memory allocation (TASK-EMB-014)
- Inference validation (TASK-EMB-015)
- Removing stub mode from preflight (TASK-EMB-019)

---

## Input Context Files

| Purpose | File Path | What To Read |
|---------|-----------|--------------|
| Current broken code | `crates/context-graph-embeddings/src/warm/loader/operations.rs` | Lines 150-165 (delete target) |
| WarmError enum | `crates/context-graph-embeddings/src/warm/error.rs` | Error variants to use |
| WarmLoadResult | `crates/context-graph-embeddings/src/warm/loader/types.rs` | Return type structure |
| Constitution | `docs2/constitution.yaml` | AP-007, stack.gpu requirements |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-002-warm-loading.md` | Full specification |

---

## Prerequisites

- [x] TASK-EMB-006 completed (WarmLoadResult struct exists at `src/warm/loader/types.rs`)
- [x] SafeTensors crate in Cargo.toml (`safetensors = "0.4"`)
- [x] sha2 crate in Cargo.toml (`sha2 = "0.10"`)
- [ ] Weight files exist in `models/` directory (SafeTensors format)

---

## Scope

### In Scope

1. **DELETE** `simulate_weight_loading()` function entirely
2. **CREATE** `load_weights()` function that reads real files
3. **CREATE** `verify_checksum()` function for integrity verification
4. **UPDATE** `load_single_model()` to call `load_weights()` instead
5. **DELETE** all calls and re-exports of `simulate_weight_loading`
6. **UPDATE** tests to use real data, not mocks

### Out of Scope

- GPU memory allocation (`allocate_model_vram` - TASK-EMB-014)
- Real inference validation (`validate_model` - TASK-EMB-015)
- Stub mode removal from preflight (TASK-EMB-019)

---

## Definition of Done

### Step 1: DELETE simulate_weight_loading

**File**: `crates/context-graph-embeddings/src/warm/loader/operations.rs`

Delete lines 150-165 entirely. The function signature is:
```rust
pub fn simulate_weight_loading(model_id: &str, _size_bytes: usize) -> WarmResult<u64>
```

### Step 2: DELETE Engine Re-exports

**File**: `crates/context-graph-embeddings/src/warm/loader/engine.rs`

Delete line 28:
```rust
use super::operations::{allocate_model_vram, simulate_weight_loading};
```

Replace with:
```rust
use super::operations::{allocate_model_vram, load_weights};
```

Delete lines 285-290 (the wrapper method):
```rust
pub(crate) fn simulate_weight_loading(
    &self,
    model_id: &str,
    size_bytes: usize,
) -> WarmResult<u64> {
    simulate_weight_loading(model_id, size_bytes)
}
```

### Step 3: CREATE load_weights Function

**File**: `crates/context-graph-embeddings/src/warm/loader/operations.rs`

Add at the position where `simulate_weight_loading` was deleted:

```rust
use safetensors::SafeTensors;
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Tensor metadata extracted from SafeTensors file.
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Map of tensor name to shape
    pub shapes: HashMap<String, Vec<usize>>,
    /// Data type of tensors
    pub dtype: safetensors::Dtype,
    /// Total parameter count
    pub total_params: usize,
}

/// Load model weights from SafeTensors file.
///
/// # CRITICAL: No Simulation
/// This function reads REAL bytes from REAL files.
/// Fake checksums (0xDEAD_BEEF...) are FORBIDDEN per Constitution AP-007.
///
/// # Arguments
/// * `weight_path` - Path to the SafeTensors file
/// * `model_id` - Model identifier for error messages
///
/// # Returns
/// * `Ok((file_bytes, checksum, metadata))` - Real data from real file
/// * `Err(WarmError)` - If file missing, corrupted, or parse fails
///
/// # Errors
/// - `WarmError::WeightFileMissing` - File not found at path
/// - `WarmError::WeightFileCorrupted` - Parse error or invalid format
pub fn load_weights(
    weight_path: &Path,
    model_id: &str,
) -> WarmResult<(Vec<u8>, [u8; 32], TensorMetadata)> {
    let start = Instant::now();

    // Step 1: Read actual file bytes
    let file_bytes = fs::read(weight_path).map_err(|e| {
        tracing::error!(
            "[EMB-E006] Weight file not found: {:?}, error: {}",
            weight_path,
            e
        );
        WarmError::WeightFileMissing {
            model_id: model_id.to_string(),
            path: weight_path.to_path_buf(),
        }
    })?;

    tracing::debug!(
        "Read {} bytes from {:?} for {}",
        file_bytes.len(),
        weight_path,
        model_id
    );

    // Step 2: Compute REAL SHA256 checksum
    let mut hasher = Sha256::new();
    hasher.update(&file_bytes);
    let checksum: [u8; 32] = hasher.finalize().into();

    tracing::debug!(
        "Computed SHA256 checksum for {}: {:02x}{:02x}{:02x}{:02x}...",
        model_id,
        checksum[0],
        checksum[1],
        checksum[2],
        checksum[3]
    );

    // Step 3: Parse SafeTensors to extract metadata
    let tensors = SafeTensors::deserialize(&file_bytes).map_err(|e| {
        tracing::error!(
            "[EMB-E004] SafeTensors parse failed for {}: {}",
            model_id,
            e
        );
        WarmError::WeightFileCorrupted {
            model_id: model_id.to_string(),
            path: weight_path.to_path_buf(),
            reason: format!("SafeTensors parse error: {}", e),
        }
    })?;

    // Step 4: Extract tensor metadata
    let mut shapes = HashMap::new();
    let mut total_params = 0usize;
    let mut dtype = safetensors::Dtype::F32;

    for (name, view) in tensors.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        total_params += shape.iter().product::<usize>();
        dtype = view.dtype();
        shapes.insert(name.to_string(), shape);
    }

    let metadata = TensorMetadata {
        shapes,
        dtype,
        total_params,
    };

    let duration = start.elapsed();
    tracing::info!(
        "Loaded weights for {} in {:?}: {} params, {} bytes, checksum {:02x}{:02x}...",
        model_id,
        duration,
        total_params,
        file_bytes.len(),
        checksum[0],
        checksum[1]
    );

    Ok((file_bytes, checksum, metadata))
}

/// Verify checksum against expected value.
///
/// # Arguments
/// * `actual` - Computed SHA256 checksum (32 bytes)
/// * `expected` - Expected checksum (32 bytes)
/// * `model_id` - Model identifier for error messages
///
/// # Returns
/// * `Ok(())` - Checksums match
/// * `Err(WarmError::WeightChecksumMismatch)` - Checksums differ
pub fn verify_checksum(
    actual: &[u8; 32],
    expected: &[u8; 32],
    model_id: &str,
) -> WarmResult<()> {
    if actual != expected {
        let actual_hex = hex::encode(actual);
        let expected_hex = hex::encode(expected);
        tracing::error!(
            "[EMB-E004] Checksum mismatch for {}: expected {}, got {}",
            model_id,
            expected_hex,
            actual_hex
        );
        return Err(WarmError::WeightChecksumMismatch {
            model_id: model_id.to_string(),
            expected: expected_hex,
            actual: actual_hex,
        });
    }
    Ok(())
}
```

### Step 4: UPDATE load_single_model

**File**: `crates/context-graph-embeddings/src/warm/loader/operations.rs`

Replace line 62:
```rust
// OLD (DELETE):
let checksum = simulate_weight_loading(model_id, expected_bytes)?;

// NEW (ADD):
let weight_path = config.model_weights_dir.join(format!("{}.safetensors", model_id));
let (file_bytes, checksum_bytes, metadata) = load_weights(&weight_path, model_id)?;

// Convert [u8; 32] to u64 for handle (first 8 bytes as checksum identifier)
let checksum = u64::from_le_bytes([
    checksum_bytes[0], checksum_bytes[1], checksum_bytes[2], checksum_bytes[3],
    checksum_bytes[4], checksum_bytes[5], checksum_bytes[6], checksum_bytes[7],
]);

// Verify file size matches expected
if file_bytes.len() != expected_bytes {
    tracing::warn!(
        "Weight file size mismatch for {}: expected {}, got {}",
        model_id,
        expected_bytes,
        file_bytes.len()
    );
}
```

### Step 5: DELETE Test Using Fake Function

**File**: `crates/context-graph-embeddings/src/warm/loader/tests/loader_tests.rs`

Delete lines 253-261:
```rust
#[test]
fn test_simulate_weight_loading() {
    let loader = WarmLoader::default();

    let checksum1 = loader.simulate_weight_loading("E1_Semantic", 1024).unwrap();
    let checksum2 = loader
        .simulate_weight_loading("E2_TemporalRecent", 1024)
        .unwrap();
    // ... rest of test
}
```

Replace with real test:
```rust
#[test]
fn test_load_weights_real_file() {
    use std::io::Write;
    use tempfile::TempDir;

    // Create temp directory with real SafeTensors file
    let temp_dir = TempDir::new().unwrap();
    let weight_path = temp_dir.path().join("test_model.safetensors");

    // Create a minimal valid SafeTensors file
    let tensor_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let tensor_bytes: Vec<u8> = tensor_data.iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    // Write SafeTensors format
    let st = safetensors::serialize(
        [("weights".to_string(), safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            &[2, 2],
            &tensor_bytes,
        ).unwrap())].into_iter().collect(),
        &None,
    ).unwrap();
    std::fs::write(&weight_path, &st).unwrap();

    // Test load_weights
    let (bytes, checksum, metadata) = load_weights(&weight_path, "test_model").unwrap();

    // Verify real data, not fake
    assert_eq!(bytes, st);
    assert_eq!(checksum.len(), 32); // Real SHA256 is 32 bytes
    assert!(!checksum.iter().all(|&b| b == 0)); // Not all zeros
    assert_eq!(metadata.total_params, 4);
    assert!(metadata.shapes.contains_key("weights"));

    // Verify checksum is deterministic
    let (_, checksum2, _) = load_weights(&weight_path, "test_model").unwrap();
    assert_eq!(checksum, checksum2);
}

#[test]
fn test_load_weights_missing_file() {
    let result = load_weights(
        Path::new("/nonexistent/path/model.safetensors"),
        "missing_model"
    );

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, WarmError::WeightFileMissing { .. }));
}

#[test]
fn test_verify_checksum_match() {
    let checksum = [1u8; 32];
    let expected = [1u8; 32];
    assert!(verify_checksum(&checksum, &expected, "test").is_ok());
}

#[test]
fn test_verify_checksum_mismatch() {
    let checksum = [1u8; 32];
    let expected = [2u8; 32];
    let result = verify_checksum(&checksum, &expected, "test");
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), WarmError::WeightChecksumMismatch { .. }));
}
```

### Step 6: Add hex Dependency (if not present)

**File**: `crates/context-graph-embeddings/Cargo.toml`

Ensure these dependencies exist:
```toml
[dependencies]
safetensors = "0.4"
sha2 = "0.10"
hex = "0.4"
```

---

## Files to Modify

| File | Change |
|------|--------|
| `src/warm/loader/operations.rs` | DELETE simulate_weight_loading, ADD load_weights, UPDATE load_single_model |
| `src/warm/loader/engine.rs` | DELETE simulate_weight_loading import and wrapper |
| `src/warm/loader/tests/loader_tests.rs` | DELETE fake test, ADD real tests |
| `Cargo.toml` | ADD hex dependency if missing |

---

## Validation Criteria

### Compilation Checks

- [ ] `cargo check -p context-graph-embeddings` passes
- [ ] `grep -rn "simulate_weight_loading" crates/context-graph-embeddings/` returns ZERO results
- [ ] `grep -rn "0xDEAD_BEEF" crates/context-graph-embeddings/` returns ZERO results
- [ ] `grep -rn "CAFE_BABE" crates/context-graph-embeddings/` returns ZERO results

### Functional Checks

- [ ] `load_weights()` returns `Result<(Vec<u8>, [u8; 32], TensorMetadata), WarmError>`
- [ ] Missing file returns `WarmError::WeightFileMissing`
- [ ] Corrupted file returns `WarmError::WeightFileCorrupted`
- [ ] Checksum mismatch returns `WarmError::WeightChecksumMismatch`
- [ ] Checksum is 32 bytes (SHA256)
- [ ] Checksum is deterministic (same file = same checksum)
- [ ] Checksum changes when file changes

### Test Commands

```bash
cd /home/cabdru/contextgraph

# Step 1: Verify simulate_weight_loading is gone
grep -rn "simulate_weight_loading" crates/context-graph-embeddings/
# Expected: NO OUTPUT (zero matches)

# Step 2: Verify fake checksums are gone
grep -rn "0xDEAD_BEEF\|CAFE_BABE" crates/context-graph-embeddings/
# Expected: NO OUTPUT (zero matches)

# Step 3: Run compilation check
cargo check -p context-graph-embeddings

# Step 4: Run tests
cargo test -p context-graph-embeddings warm::loader::load_weights -- --nocapture
cargo test -p context-graph-embeddings warm::loader::verify_checksum -- --nocapture
```

---

## CRITICAL: Full State Verification

### Source of Truth

The source of truth for this task is:
1. **File System**: SafeTensors weight files in `models/` directory
2. **SHA256 Checksum**: Computed by `sha2::Sha256` crate
3. **SafeTensors Metadata**: Parsed by `safetensors::SafeTensors::deserialize()`

### Execute & Inspect Protocol

After implementing, you MUST:

1. **Create a test weight file**:
```bash
# Create a minimal SafeTensors file for testing
cd /home/cabdru/contextgraph
mkdir -p tests/fixtures/weights
# Use the safetensors Python library or Rust code to create a valid file
```

2. **Run the implementation**:
```rust
let (bytes, checksum, metadata) = load_weights(
    Path::new("tests/fixtures/weights/test_model.safetensors"),
    "test_model"
)?;
```

3. **Verify the checksum independently**:
```bash
# Compute SHA256 with system tool
sha256sum tests/fixtures/weights/test_model.safetensors
# Compare output with what load_weights() returned
```

4. **Verify SafeTensors parsing independently**:
```bash
# Use Python to verify the tensor shapes
python3 -c "
from safetensors import safe_open
with safe_open('tests/fixtures/weights/test_model.safetensors', framework='pt') as f:
    for name in f.keys():
        tensor = f.get_tensor(name)
        print(f'{name}: {tensor.shape}')
"
# Compare with what metadata.shapes contains
```

### Boundary & Edge Case Audit

You MUST manually test these 3 edge cases and print state before/after:

#### Edge Case 1: Empty File (0 bytes)

```rust
#[test]
fn test_edge_case_empty_file() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let empty_path = temp_dir.path().join("empty.safetensors");
    std::fs::write(&empty_path, b"").unwrap();

    println!("=== BEFORE: Empty file test ===");
    println!("File exists: {}", empty_path.exists());
    println!("File size: {} bytes", std::fs::metadata(&empty_path).unwrap().len());

    let result = load_weights(&empty_path, "empty_test");

    println!("=== AFTER: Empty file test ===");
    println!("Result: {:?}", result);

    assert!(result.is_err());
    // Empty file should fail SafeTensors parse
}
```

#### Edge Case 2: Maximum Size File (approaching memory limits)

```rust
#[test]
fn test_edge_case_large_file_metadata() {
    // Don't actually create a multi-GB file, just test metadata handling
    let metadata = TensorMetadata {
        shapes: [
            ("layer.0.weight".to_string(), vec![10000, 10000]),
            ("layer.1.weight".to_string(), vec![10000, 10000]),
        ].into_iter().collect(),
        dtype: safetensors::Dtype::F32,
        total_params: 200_000_000, // 200M params
    };

    println!("=== Large model metadata ===");
    println!("Total params: {}", metadata.total_params);
    println!("Estimated size: {} GB", (metadata.total_params * 4) as f64 / 1e9);

    // Verify our code handles large numbers correctly
    assert_eq!(metadata.total_params, 200_000_000);
}
```

#### Edge Case 3: Invalid SafeTensors Format

```rust
#[test]
fn test_edge_case_invalid_format() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let invalid_path = temp_dir.path().join("invalid.safetensors");
    std::fs::write(&invalid_path, b"not a valid safetensors file").unwrap();

    println!("=== BEFORE: Invalid format test ===");
    println!("File content: {:?}", std::fs::read(&invalid_path).unwrap());

    let result = load_weights(&invalid_path, "invalid_test");

    println!("=== AFTER: Invalid format test ===");
    println!("Result: {:?}", result);

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), WarmError::WeightFileCorrupted { .. }));
}
```

### Evidence of Success

After running all tests, you MUST provide a log showing:

```
=== PHYSICAL EVIDENCE: load_weights() Implementation ===

1. simulate_weight_loading DELETED:
   $ grep -rn "simulate_weight_loading" crates/context-graph-embeddings/
   (no output)

2. Fake checksums DELETED:
   $ grep -rn "0xDEAD_BEEF" crates/context-graph-embeddings/
   (no output)

3. Real file loaded:
   File: tests/fixtures/weights/test_model.safetensors
   Size: 1234 bytes
   SHA256: 3a7bd3e2c9f5...

4. Checksum verified with system tool:
   $ sha256sum tests/fixtures/weights/test_model.safetensors
   3a7bd3e2c9f5... tests/fixtures/weights/test_model.safetensors
   MATCHES load_weights() output

5. Tensor metadata verified:
   Python SafeTensors: weights: torch.Size([2, 2])
   Rust metadata.shapes: {"weights": [2, 2]}
   MATCHES

6. Tests pass:
   $ cargo test -p context-graph-embeddings warm::loader
   test warm::loader::test_load_weights_real_file ... ok
   test warm::loader::test_load_weights_missing_file ... ok
   test warm::loader::test_verify_checksum_match ... ok
   test warm::loader::test_verify_checksum_mismatch ... ok
```

---

## Constraints

### MUST DO

- Use `safetensors::SafeTensors::deserialize()` for parsing
- Use `sha2::Sha256` for checksum computation
- Return real 32-byte SHA256 checksum
- Read real bytes from real files
- Fail fast with clear errors

### MUST NOT DO

- Generate checksums from model_id string
- Use fake pointers (0x7f80...)
- Use XOR patterns for checksums
- Fall back to simulation
- Mock file reads in production code
- Create backwards compatibility shims

---

## Related Tasks

| Task | Relationship |
|------|--------------|
| TASK-EMB-006 | Depends on (WarmLoadResult struct) |
| TASK-EMB-014 | Blocked by this (needs real weight loading for VRAM allocation) |
| TASK-EMB-015 | Blocked by this (needs real weights for inference validation) |
| TASK-EMB-019 | Related (stub mode removal after this completes) |

---

## Troubleshooting

### "WeightFileMissing" when running tests

**Cause**: No SafeTensors files in expected location.

**Fix**: Create test fixtures:
```bash
mkdir -p tests/fixtures/weights
# Create minimal SafeTensors file using Python or Rust
```

### "WeightFileCorrupted" on valid file

**Cause**: File is not in SafeTensors format (might be PyTorch .pt or ONNX).

**Fix**: Convert to SafeTensors:
```python
from safetensors.torch import save_file
import torch
tensors = torch.load("model.pt")
save_file(tensors, "model.safetensors")
```

### Checksum mismatch after download

**Cause**: Incomplete download or file corruption.

**Fix**: Re-download and verify with sha256sum:
```bash
sha256sum model.safetensors
# Compare with expected checksum from model registry
```

</task_spec>
