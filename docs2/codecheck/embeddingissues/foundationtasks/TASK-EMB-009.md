# TASK-EMB-009: Create Weight File Specification Documentation

<task_spec id="TASK-EMB-009" version="2.0" updated="2026-01-06">

## CRITICAL CONTEXT FOR AI AGENT

**You are creating documentation for a weight file that DOES NOT EXIST YET.**

The `sparse_projection.safetensors` file must be trained separately. This task documents:
1. The exact file format specification
2. Validation requirements
3. Download/training instructions
4. No-fallback policy

**DO NOT confuse this with:**
- `models/sparse/model.safetensors` (508MB) - This is the SPLADE backbone model (already exists)
- The projection matrix is a SEPARATE file for converting 30522D sparse → 1536D dense

---

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-EMB-009 |
| **Title** | Create Weight File Specification Documentation |
| **Status** | ready |
| **Layer** | foundation |
| **Sequence** | 9 |
| **Implements** | REQ-EMB-001 |
| **Depends On** | TASK-EMB-002 (ProjectionMatrix struct - COMPLETE) |
| **Estimated Complexity** | low |
| **Constitution Ref** | `AP-007` (no stub data in production) |

---

## Problem Statement

The `ProjectionMatrix` struct exists (TASK-EMB-002) but there's no documentation explaining:
1. What weight file is needed
2. Where to download it
3. How to validate it
4. What happens if it's missing

**Without this documentation:**
- Developers don't know what file to download
- No validation can be performed
- Silent failures may occur
- Users cannot self-diagnose missing weight issues

---

## Current Codebase State (Verified 2026-01-06)

### Models Directory Structure

```
/home/cabdru/contextgraph/models/
├── models_config.toml          # Auto-generated config
├── sparse/                     # SPLADE backbone model (EXISTS)
│   ├── model.safetensors       # 508MB - BERT + MLM head (NOT the projection)
│   ├── README.md               # HuggingFace README (from download)
│   ├── config.json
│   ├── tokenizer.json
│   └── vocab.txt
├── semantic/                   # E1 semantic model
├── code/                       # E7 code model
├── causal/                     # E5 causal model
├── entity/                     # E11 entity model
├── graph/                      # E8 graph model
├── multimodal/                 # E10 multimodal model
├── late-interaction/           # E12 ColBERT model
├── contextual/                 # Additional contextual model
├── splade-v3/                  # Alternative SPLADE
├── hdc/                        # E9 HDC (empty - custom)
├── hyperbolic/                 # (empty - custom)
└── temporal/                   # E2-E4 temporal (empty - custom)
```

### What DOES NOT Exist Yet

| File | Size | Purpose | Status |
|------|------|---------|--------|
| `models/sparse_projection.safetensors` | ~187MB | Learned projection matrix [30522, 1536] | **DOES NOT EXIST** |

### ProjectionMatrix Struct (TASK-EMB-002 - COMPLETE)

**File:** `crates/context-graph-embeddings/src/models/pretrained/sparse/projection.rs`

```rust
// Verified existing constants
pub const PROJECTION_WEIGHT_FILE: &str = "sparse_projection.safetensors";
pub const PROJECTION_TENSOR_NAME: &str = "projection.weight";

pub struct ProjectionMatrix {
    weights: Tensor,           // [30522, 1536] on GPU
    device: Device,            // CUDA for production
    weight_checksum: [u8; 32], // SHA256 verification
}

impl ProjectionMatrix {
    pub const EXPECTED_SHAPE: (usize, usize) = (30522, 1536);
    pub const EXPECTED_FILE_SIZE: usize = 187_527_168; // 30522 * 1536 * 4 bytes
}
```

### Sparse Types (TASK-EMB-001 - COMPLETE)

**File:** `crates/context-graph-embeddings/src/models/pretrained/sparse/types.rs`

```rust
pub const SPARSE_VOCAB_SIZE: usize = 30522;          // BERT vocabulary
pub const SPARSE_PROJECTED_DIMENSION: usize = 1536;  // Output dimension
```

---

## Scope

### In Scope
1. Create `models/README.md` documenting ALL weight files
2. Document sparse projection matrix specification in detail
3. Document checksum verification process
4. Document download/training instructions
5. Emphasize no-fallback policy (AP-007)

### Out of Scope
- Actual weight training (separate process)
- Weight file creation (separate training task)
- Loading implementation (TASK-EMB-011)
- Projection implementation (TASK-EMB-012)

---

## Definition of Done

### File to Create

**Path:** `/home/cabdru/contextgraph/models/README.md`

**Content:**

```markdown
# Model Weight Files

This directory contains pre-trained model weights required for the Context Graph embedding pipeline.

## Directory Structure

```
models/
├── README.md                      # This file
├── models_config.toml             # Auto-generated model paths
├── sparse_projection.safetensors  # REQUIRED: Sparse projection weights [30522, 1536]
├── sparse/                        # SPLADE backbone (naver/splade-cocondenser-ensembledistil)
├── semantic/                      # E1: intfloat/e5-large-v2
├── code/                          # E7: microsoft/codebert-base
├── causal/                        # E5: allenai/longformer-base-4096
├── entity/                        # E11: sentence-transformers/all-MiniLM-L6-v2
├── graph/                         # E8: sentence-transformers/paraphrase-MiniLM-L6-v2
├── multimodal/                    # E10: openai/clip-vit-large-patch14
├── late-interaction/              # E12: colbert-ir/colbertv2.0
├── contextual/                    # sentence-transformers/all-mpnet-base-v2
└── temporal/                      # E2-E4: Custom temporal embeddings
```

## Sparse Projection Matrix

### CRITICAL: This File is REQUIRED

The sparse projection matrix converts 30522-dimensional sparse vectors (from SPLADE)
to 1536-dimensional dense vectors for multi-array storage.

**If this file is missing, the system WILL FAIL. There is NO fallback.**

### File Specification

| Property | Value |
|----------|-------|
| **File Name** | `sparse_projection.safetensors` |
| **Location** | `models/sparse_projection.safetensors` |
| **Format** | SafeTensors v0.4+ |
| **Tensor Name** | `projection.weight` |
| **Shape** | [30522, 1536] |
| **Data Type** | float32 |
| **Size** | ~187 MB (30522 × 1536 × 4 bytes = 187,527,168 bytes) |
| **Download URL** | https://huggingface.co/contextgraph/sparse-projection |

### Shape Breakdown

```
Input:  30522 (BERT vocabulary size - sparse dimension)
Output: 1536  (Constitution E6_Sparse projected dimension)

Matrix: [30522 rows × 1536 columns]
        Each row i contains the dense representation for vocabulary token i
```

### Checksum Verification

The embedding system verifies weight file integrity via SHA-256 checksum.

```
Expected checksum: <TBD_AFTER_TRAINING>
Verification: SHA256(sparse_projection.safetensors) must match
```

**Verification command:**
```bash
sha256sum models/sparse_projection.safetensors
```

### Training Details

| Property | Value |
|----------|-------|
| **Training Dataset** | MS MARCO passages |
| **Training Objective** | Contrastive learning with semantic preservation |
| **Semantic Preservation Score** | >0.85 (required) |
| **Constitution Version** | 4.0.0 |
| **Training Script** | `scripts/train_sparse_projection.py` (TBD) |

### Usage in Code

The projection matrix is loaded automatically during model initialization:

```rust
use crate::models::pretrained::sparse::{ProjectionMatrix, PROJECTION_WEIGHT_FILE};

// Load projection matrix (TASK-EMB-011)
let model_path = PathBuf::from("models");
let projection = ProjectionMatrix::load(&model_path)?;

// Project sparse to dense (TASK-EMB-012)
let sparse_vector = model.embed_sparse(&input).await?;
let dense_vector = projection.project(&sparse_vector)?;

// Result: 1536D dense vector for multi-array storage
assert_eq!(dense_vector.len(), 1536);
```

### CRITICAL: No Hash Fallback

**Constitution AP-007 prohibits stub data in production.**

The previous hash-based projection (`idx % projected_dim`) has been **REMOVED** because:

1. **Hash collisions destroy semantics**:
   - Token "machine" (idx 3057) and "learning" (idx 4593) could map to the same dimension
   - `3057 % 1536 = 481` and `4593 % 1536 = 481` (collision!)

2. **No learned representation**:
   - Hash modulo is random noise, not semantic structure

3. **Violates AP-007**:
   - Hash fallback is stub/mock behavior masquerading as real functionality

**If `sparse_projection.safetensors` is missing, the system MUST panic with:**
```
[EMB-E006] PROJECTION_MATRIX_MISSING: Weight file not found at models/sparse_projection.safetensors
  Expected: models/sparse_projection.safetensors
  Remediation: Download projection weights from https://huggingface.co/contextgraph/sparse-projection
```

## Other Model Weights

### Pre-trained Models (Downloaded from HuggingFace)

| Embedder | Directory | HuggingFace Repo | Size |
|----------|-----------|------------------|------|
| E1 Semantic | `semantic/` | intfloat/e5-large-v2 | ~1.3GB |
| E5 Causal | `causal/` | allenai/longformer-base-4096 | ~717MB |
| E6 SPLADE | `sparse/` | naver/splade-cocondenser-ensembledistil | ~508MB |
| E7 Code | `code/` | microsoft/codebert-base | ~513MB |
| E8 Graph | `graph/` | sentence-transformers/paraphrase-MiniLM-L6-v2 | ~87MB |
| E10 Multimodal | `multimodal/` | openai/clip-vit-large-patch14 | ~1.6GB |
| E11 Entity | `entity/` | sentence-transformers/all-MiniLM-L6-v2 | ~87MB |
| E12 Late Interaction | `late-interaction/` | colbert-ir/colbertv2.0 | ~419MB |

### Custom Models (Require Training)

| Embedder | Directory | Status |
|----------|-----------|--------|
| E2-E4 Temporal | `temporal/` | Custom implementation |
| E9 HDC | `hdc/` | Custom hyperdimensional computing |
| E13 SPLADE (v3) | `splade-v3/` | Alternative SPLADE version |

## Downloading Models

### Using the Download Script

```bash
# Download all pre-trained models
python scripts/download_models.py

# Download specific model
python scripts/download_models.py --model sparse
```

### Manual Download

Each model can be downloaded manually from HuggingFace:

```bash
# Example: Download SPLADE model
git lfs install
git clone https://huggingface.co/naver/splade-cocondenser-ensembledistil models/sparse
```

## Validation

### Verify All Required Weights Exist

```bash
#!/bin/bash
# Check all required weight files

MODELS_DIR="models"
REQUIRED_FILES=(
    "sparse_projection.safetensors"
    "sparse/model.safetensors"
    "semantic/model.safetensors"
    "code/model.safetensors"
    "causal/model.safetensors"
    "entity/model.safetensors"
    "graph/model.safetensors"
    "multimodal/model.safetensors"
    "late-interaction/model.safetensors"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$MODELS_DIR/$file" ]]; then
        echo "✓ $file exists"
    else
        echo "✗ $file MISSING"
    fi
done
```

### Verify Sparse Projection Shape

```bash
# Using Python to verify safetensors file
python -c "
from safetensors import safe_open
with safe_open('models/sparse_projection.safetensors', framework='pt') as f:
    tensor = f.get_tensor('projection.weight')
    print(f'Shape: {tensor.shape}')
    assert tensor.shape == (30522, 1536), f'Wrong shape: {tensor.shape}'
    print('✓ Shape verified: [30522, 1536]')
"
```

## Troubleshooting

### Error: PROJECTION_MATRIX_MISSING

**Symptom:**
```
[EMB-E006] PROJECTION_MATRIX_MISSING: Weight file not found
```

**Solution:**
1. Download the projection weights:
   ```bash
   wget https://huggingface.co/contextgraph/sparse-projection/resolve/main/sparse_projection.safetensors -O models/sparse_projection.safetensors
   ```
2. Or train your own (see training documentation)

### Error: WEIGHT_CHECKSUM_MISMATCH

**Symptom:**
```
[EMB-E004] WEIGHT_CHECKSUM_MISMATCH: Corrupted weight file
```

**Solution:**
1. Delete the corrupted file
2. Re-download from trusted source
3. Verify checksum matches

### Error: DIMENSION_MISMATCH

**Symptom:**
```
[EMB-E005] DIMENSION_MISMATCH: Projection matrix has wrong shape
```

**Solution:**
1. Verify you downloaded the correct file
2. Shape MUST be [30522, 1536]
3. Older versions may have [30522, 768] - these are INCOMPATIBLE

## Constitution References

- **E6_Sparse**: `dim: "~30K 5%active"` → 1536D output via learned projection
- **AP-007**: No stub data in production - hash fallback is FORBIDDEN
- **E13_SPLADE**: Same projection architecture as E6

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-06 | Initial specification |
```

---

## Verification Commands

Execute ALL commands after creating the file:

```bash
cd /home/cabdru/contextgraph

# 1. Verify file exists
ls -la models/README.md
# Expected: README.md file exists

# 2. Verify file content includes key sections
grep -c "sparse_projection.safetensors" models/README.md
# Expected: Multiple occurrences (>5)

# 3. Verify shape documentation
grep "30522.*1536" models/README.md
# Expected: Shape [30522, 1536] documented

# 4. Verify no-fallback policy documented
grep -c "AP-007\|No.*fallback\|FORBIDDEN" models/README.md
# Expected: Multiple occurrences (>3)

# 5. Verify download URL documented
grep "huggingface.co/contextgraph" models/README.md
# Expected: Download URL present

# 6. Verify checksum documentation
grep -c "SHA-256\|checksum" models/README.md
# Expected: Multiple occurrences (>2)
```

---

## Full State Verification Protocol

### Source of Truth

| Data | Location | Verification Method |
|------|----------|---------------------|
| README content | `models/README.md` | `cat models/README.md` |
| File existence | Filesystem | `ls -la models/README.md` |
| Content completeness | File content | `grep` for required sections |

### Execute & Inspect

After creating the file:

```bash
# 1. Verify file was created
stat models/README.md
# Must show: regular file, non-zero size

# 2. Verify content structure (check for headers)
grep "^##" models/README.md | head -10
# Expected output:
# ## Directory Structure
# ## Sparse Projection Matrix
# ## Other Model Weights
# ## Downloading Models
# ## Validation
# ## Troubleshooting
# ## Constitution References
# ## Version History

# 3. Count total lines
wc -l models/README.md
# Expected: >200 lines
```

### Boundary & Edge Case Audit

**Edge Case 1: Empty models directory**
```bash
# Simulate: What if models/ doesn't exist?
# The README should document this clearly
grep -A5 "PROJECTION_MATRIX_MISSING" models/README.md
# VERIFY: Error message and remediation are documented
```

**Edge Case 2: Wrong shape file**
```bash
# Simulate: What if someone downloads wrong version?
grep -A5 "DIMENSION_MISMATCH" models/README.md
# VERIFY: Error message explains [30522, 768] incompatibility
```

**Edge Case 3: Corrupted download**
```bash
# Simulate: What if file is corrupted?
grep -A5 "WEIGHT_CHECKSUM_MISMATCH" models/README.md
# VERIFY: Remediation steps documented
```

### Evidence of Success Log

```
========================================
TASK-EMB-009 VERIFICATION LOG
========================================

1. FILE CREATION CHECK:
   models/README.md exists: YES
   File size: XXX bytes (non-zero)

2. REQUIRED SECTIONS:
   Directory Structure: PRESENT
   Sparse Projection Matrix: PRESENT
   File Specification Table: PRESENT
   Checksum Verification: PRESENT
   No Hash Fallback Warning: PRESENT
   Other Model Weights: PRESENT
   Troubleshooting: PRESENT

3. KEY CONTENT VERIFICATION:
   Shape [30522, 1536]: DOCUMENTED
   Size ~187MB: DOCUMENTED
   Tensor name "projection.weight": DOCUMENTED
   SafeTensors format: DOCUMENTED
   Download URL: DOCUMENTED

4. CONSTITUTION COMPLIANCE:
   AP-007 referenced: YES
   No-fallback policy emphasized: YES
   E6_Sparse referenced: YES

5. ERROR DOCUMENTATION:
   EMB-E006 (MISSING): DOCUMENTED
   EMB-E004 (CHECKSUM): DOCUMENTED
   EMB-E005 (DIMENSION): DOCUMENTED

STATUS: COMPLETE
========================================
```

---

## What NOT to Modify

| Item | Reason |
|------|--------|
| `models/sparse/README.md` | HuggingFace README for SPLADE model |
| `models/models_config.toml` | Auto-generated, not manually edited |
| Any `model.safetensors` files | Binary weight files |
| Existing model subdirectories | Downloaded from HuggingFace |

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why It's Wrong |
|--------------|----------------|
| Document hash fallback as option | Violates AP-007 |
| Skip checksum documentation | Prevents integrity verification |
| Omit error codes | Makes debugging harder |
| Use wrong file path | `crates/.../models/` is wrong, use `models/` |
| Document `model.safetensors` as projection | That's the SPLADE backbone, not projection matrix |

---

## Notes for Implementing Agent

1. **Create file at `models/README.md`** - NOT in `crates/context-graph-embeddings/`
2. The `sparse_projection.safetensors` file does NOT exist yet - document requirements only
3. Existing `models/sparse/README.md` is from HuggingFace - don't modify it
4. Constitution AP-007 requires explicit "no fallback" documentation
5. All error codes (EMB-E004, E005, E006) must be documented
6. Include both automated and manual verification methods

---

## Dependencies

### Prerequisites (All COMPLETE)
- **TASK-EMB-001**: `SPARSE_PROJECTED_DIMENSION = 1536` ✓
- **TASK-EMB-002**: `ProjectionMatrix` struct exists ✓

### This Task Enables
- **TASK-EMB-011**: Loading implementation needs documented spec
- **TASK-EMB-012**: Projection implementation references this spec
- User self-service: Developers can diagnose missing weight issues

---

## Traceability

| Requirement | Constitution Section | Code Location |
|-------------|---------------------|---------------|
| E6 Projection | `embeddings.models.E6_Sparse` | `projection.rs` |
| No Stub Data | `forbidden.AP-007` | README documentation |
| Checksum Verification | Security best practice | README documentation |

</task_spec>
