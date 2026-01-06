# TASK-EMB-010: Create Golden Reference Fixtures

<task_spec id="TASK-EMB-010" version="1.0">

## Metadata

| Field | Value |
|-------|-------|
| **Title** | Create Golden Reference Fixtures |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 10 |
| **Implements** | REQ-EMB-003 |
| **Depends On** | None |
| **Estimated Complexity** | low |
| **Parallel Group** | A (start immediately) |

---

## Context

TECH-EMB-002 specifies validating inference output against golden reference files. This prevents fake/simulated output from passing tests. Golden references are pre-computed outputs from known inputs that inference results must match.

Constitution AP-007 forbids stub data in prod - golden references are the authoritative truth for validation.

---

## Input Context Files

| Purpose | File Path |
|---------|-----------|
| Constitution | `docs2/constitution.yaml` sections `AP-007`, `testing.validation` |
| Tech spec | `docs2/codecheck/embeddingissues/TECH-EMB-002-warm-loading.md` |
| Test fixtures dir | `tests/fixtures/` |

---

## Prerequisites

- [ ] Understand what "golden reference" means in testing
- [ ] Access to produce real model outputs (for reference generation)
- [ ] Test fixtures directory exists

---

## Scope

### In Scope

- Define golden reference file format specification
- Create `GoldenReference` struct for loading/comparing
- Define tolerance thresholds for floating point comparison
- Create fixture directory structure
- Document how to regenerate goldens

### Out of Scope

- Actual golden reference generation (requires trained models)
- Inference validation logic (TASK-EMB-015)
- Model loading (TASK-EMB-013)

---

## Definition of Done

### Signatures

```rust
// File: crates/context-graph-embeddings/src/testing/golden.rs

use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// Golden reference file for validating model inference output.
///
/// # Purpose
/// Golden references are pre-computed outputs from known inputs.
/// They serve as the "ground truth" for inference validation.
///
/// # Constitution Compliance
/// - AP-007: No stub data in prod - goldens ARE the authoritative truth
/// - Tests MUST compare against goldens, not generated fake data
///
/// # File Format
/// JSON file with structure:
/// ```json
/// {
///   "version": "1.0.0",
///   "model_id": "E6_Sparse",
///   "generated_at": "2026-01-06T00:00:00Z",
///   "input": {
///     "text": "The quick brown fox",
///     "tokens": [1996, 4248, 2829, 4419]
///   },
///   "output": {
///     "embedding": [0.123, -0.456, ...],
///     "norm": 1.0,
///     "dimension": 1536
///   },
///   "metadata": {
///     "model_version": "1.0.0",
///     "constitution_version": "4.0.0"
///   }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenReference {
    /// Schema version.
    pub version: String,

    /// Which model/embedder this golden is for.
    pub model_id: String,

    /// When the golden was generated (ISO 8601).
    pub generated_at: String,

    /// Input that produced this output.
    pub input: GoldenInput,

    /// Expected output.
    pub output: GoldenOutput,

    /// Generation metadata.
    pub metadata: GoldenMetadata,
}

/// Input specification for a golden reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenInput {
    /// Original text input.
    pub text: String,

    /// Tokenized input (optional, depends on model).
    #[serde(default)]
    pub tokens: Vec<u32>,

    /// Raw input vector (for projection tests).
    #[serde(default)]
    pub vector: Vec<f32>,
}

/// Expected output from inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenOutput {
    /// Expected embedding vector.
    pub embedding: Vec<f32>,

    /// Expected L2 norm (typically 1.0 for normalized).
    pub norm: f32,

    /// Output dimension.
    pub dimension: usize,

    /// Checksum of embedding (for quick comparison).
    #[serde(default)]
    pub checksum: String,
}

/// Metadata about golden generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenMetadata {
    /// Model version used to generate.
    pub model_version: String,

    /// Constitution version.
    pub constitution_version: String,

    /// Hardware used (for reproducibility tracking).
    #[serde(default)]
    pub hardware: Option<String>,

    /// Any notes about this golden.
    #[serde(default)]
    pub notes: Option<String>,
}

impl GoldenReference {
    /// Load a golden reference from JSON file.
    pub fn load(path: &Path) -> Result<Self, GoldenError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            GoldenError::IoError {
                path: path.to_path_buf(),
                details: e.to_string(),
            }
        })?;

        serde_json::from_str(&content).map_err(|e| {
            GoldenError::ParseError {
                path: path.to_path_buf(),
                details: e.to_string(),
            }
        })
    }

    /// Save golden reference to JSON file.
    pub fn save(&self, path: &Path) -> Result<(), GoldenError> {
        let content = serde_json::to_string_pretty(self).map_err(|e| {
            GoldenError::SerializeError {
                details: e.to_string(),
            }
        })?;

        std::fs::write(path, content).map_err(|e| {
            GoldenError::IoError {
                path: path.to_path_buf(),
                details: e.to_string(),
            }
        })
    }

    /// Compare actual output against golden with tolerance.
    ///
    /// # Arguments
    /// * `actual` - Actual inference output
    /// * `tolerance` - Maximum allowed absolute difference per element
    ///
    /// # Returns
    /// `Ok(similarity)` if within tolerance, `Err(GoldenError)` with details.
    pub fn compare(&self, actual: &[f32], tolerance: f32) -> Result<f32, GoldenError> {
        if actual.len() != self.output.embedding.len() {
            return Err(GoldenError::DimensionMismatch {
                expected: self.output.embedding.len(),
                actual: actual.len(),
            });
        }

        // Compute differences
        let max_diff = actual.iter()
            .zip(self.output.embedding.iter())
            .map(|(a, g)| (a - g).abs())
            .fold(0.0f32, f32::max);

        if max_diff > tolerance {
            return Err(GoldenError::ToleranceExceeded {
                max_diff,
                tolerance,
                golden_id: self.model_id.clone(),
            });
        }

        // Compute cosine similarity
        let dot: f32 = actual.iter()
            .zip(self.output.embedding.iter())
            .map(|(a, g)| a * g)
            .sum();

        let norm_actual: f32 = actual.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_golden: f32 = self.output.embedding.iter().map(|v| v * v).sum::<f32>().sqrt();

        let similarity = if norm_actual > 1e-8 && norm_golden > 1e-8 {
            dot / (norm_actual * norm_golden)
        } else {
            0.0
        };

        Ok(similarity)
    }

    /// Get default tolerance for this model type.
    pub fn default_tolerance(&self) -> f32 {
        // Different models may need different tolerances
        // due to GPU non-determinism
        match self.model_id.as_str() {
            "E6_Sparse" | "E13_SPLADE" => 1e-4,  // Sparse: stricter
            "E9_HDC" => 1e-2,                     // Binary: more tolerance
            _ => 1e-5,                            // Default: very strict
        }
    }

    /// Assert actual output matches golden (panics on failure).
    pub fn assert_matches(&self, actual: &[f32]) {
        let tolerance = self.default_tolerance();
        match self.compare(actual, tolerance) {
            Ok(similarity) => {
                assert!(similarity > 0.99,
                    "[GOLDEN-FAIL] Cosine similarity {:.6} below threshold 0.99 for {}",
                    similarity, self.model_id);
            }
            Err(e) => {
                panic!("[GOLDEN-FAIL] {}", e);
            }
        }
    }
}

/// Standard golden reference file paths.
pub struct GoldenPaths;

impl GoldenPaths {
    /// Base directory for golden fixtures.
    pub const BASE_DIR: &'static str = "tests/fixtures/golden";

    /// Get path for a specific model's golden.
    pub fn for_model(model_id: &str) -> PathBuf {
        PathBuf::from(Self::BASE_DIR).join(format!("{}.json", model_id.to_lowercase()))
    }

    /// Get all available goldens.
    pub fn available() -> Vec<PathBuf> {
        let base = PathBuf::from(Self::BASE_DIR);
        if !base.exists() {
            return Vec::new();
        }

        std::fs::read_dir(base)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().map_or(false, |ext| ext == "json"))
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Tolerance presets for different validation scenarios.
pub struct TolerancePresets;

impl TolerancePresets {
    /// Strict: for deterministic CPU operations.
    pub const STRICT: f32 = 1e-6;

    /// Normal: for GPU operations (slight non-determinism).
    pub const NORMAL: f32 = 1e-5;

    /// Relaxed: for operations with known variance.
    pub const RELAXED: f32 = 1e-4;

    /// Binary: for binary quantized outputs.
    pub const BINARY: f32 = 1e-2;

    /// Select tolerance based on model type.
    pub fn for_model(model_id: &str) -> f32 {
        match model_id {
            "E9_HDC" => Self::BINARY,
            "E6_Sparse" | "E13_SPLADE" => Self::RELAXED,
            _ => Self::NORMAL,
        }
    }
}

/// Errors during golden reference operations.
#[derive(Debug, thiserror::Error)]
pub enum GoldenError {
    #[error("Failed to read golden file {path}: {details}")]
    IoError { path: PathBuf, details: String },

    #[error("Failed to parse golden file {path}: {details}")]
    ParseError { path: PathBuf, details: String },

    #[error("Failed to serialize golden: {details}")]
    SerializeError { details: String },

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Output exceeds tolerance {tolerance} (max diff: {max_diff}) for {golden_id}")]
    ToleranceExceeded { max_diff: f32, tolerance: f32, golden_id: String },

    #[error("Golden not found for model: {model_id}")]
    GoldenNotFound { model_id: String },
}
```

### Directory Structure

```
tests/fixtures/golden/
├── e1_semantic.json        # E1 Semantic embedding golden
├── e5_causal.json          # E5 Causal embedding golden
├── e6_sparse.json          # E6 Sparse embedding golden
├── e6_sparse_projection.json  # Sparse projection specifically
├── e13_splade.json         # E13 SPLADE golden
├── README.md               # How to regenerate goldens
└── generate_goldens.py     # Script to regenerate (runs real models)
```

### Constraints

- Golden files MUST be generated from real model inference
- Tolerance MUST account for GPU non-determinism
- Comparison MUST fail if dimension mismatches
- All goldens MUST include generation metadata

### Verification

- `GoldenReference::load()` parses sample golden file
- `compare()` returns Ok for matching output
- `compare()` returns Err for mismatched output
- Directory structure exists

---

## Files to Create

| File Path | Description |
|-----------|-------------|
| `crates/context-graph-embeddings/src/testing/mod.rs` | Testing module declaration |
| `crates/context-graph-embeddings/src/testing/golden.rs` | Golden reference types |
| `tests/fixtures/golden/README.md` | Instructions for golden generation |

## Files to Modify

| File Path | Change |
|-----------|--------|
| `crates/context-graph-embeddings/src/lib.rs` | Add `pub mod testing;` |

---

## Validation Criteria

- [ ] `GoldenReference` struct with all fields
- [ ] `load()` and `save()` for JSON serialization
- [ ] `compare()` with tolerance checking
- [ ] `assert_matches()` for test assertions
- [ ] `GoldenPaths` for standard locations
- [ ] `TolerancePresets` for different scenarios
- [ ] `tests/fixtures/golden/` directory created
- [ ] README.md documents regeneration process
- [ ] `cargo check -p context-graph-embeddings` passes

---

## Test Commands

```bash
cd /home/cabdru/contextgraph
mkdir -p tests/fixtures/golden
cargo check -p context-graph-embeddings
cargo test -p context-graph-embeddings golden:: -- --nocapture
```

---

## Pseudo Code

```
golden.rs:
  Define GoldenReference struct (Serialize, Deserialize):
    version, model_id, generated_at
    input: GoldenInput (text, tokens, vector)
    output: GoldenOutput (embedding, norm, dimension, checksum)
    metadata: GoldenMetadata (model_version, constitution_version, hardware, notes)

  Impl GoldenReference:
    load(path) -> Result<Self, GoldenError>
    save(path) -> Result<(), GoldenError>
    compare(actual, tolerance) -> Result<f32, GoldenError>
    default_tolerance() -> f32
    assert_matches(actual)

  Define GoldenPaths struct:
    BASE_DIR = "tests/fixtures/golden"
    for_model(model_id) -> PathBuf
    available() -> Vec<PathBuf>

  Define TolerancePresets struct:
    STRICT = 1e-6
    NORMAL = 1e-5
    RELAXED = 1e-4
    BINARY = 1e-2
    for_model(model_id) -> f32

  Define GoldenError enum:
    IoError, ParseError, SerializeError
    DimensionMismatch, ToleranceExceeded, GoldenNotFound

tests/fixtures/golden/README.md:
  Document how to run generate_goldens.py
  List required models and hardware
  Explain when to regenerate
```

</task_spec>
