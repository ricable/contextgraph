# TASK-05: Add CI gate for FFI consolidation

**Original ID**: TASK-ARCH-005
**Status**: READY
**Layer**: Foundation
**Sequence**: 5
**Implements**: REQ-ARCH-005
**Dependencies**: TASK-04 (GpuDevice RAII wrapper - ✅ COMPLETE)
**Blocks**: None
**Estimated Hours**: 1

---

## 🚨 CRITICAL: READ THIS ENTIRE DOCUMENT BEFORE IMPLEMENTING

This document contains everything you need to implement this task. **ASSUME NOTHING. VERIFY EVERYTHING.**

---

## EXECUTIVE SUMMARY

**Goal**: Create a CI gate script that FAILS if any CUDA/FAISS FFI declarations exist outside the `context-graph-cuda` crate.

**Why**: Constitution rule `ARCH-06` mandates all CUDA FFI in `context-graph-cuda` only. This enables focused security audits and prevents FFI scatter.

**Deliverables**:
1. `scripts/check-ffi-consolidation.sh` - POSIX-compliant script
2. Addition of FFI check job to `.github/workflows/ci.yml`

---

## CURRENT CODEBASE STATE (Verified 2026-01-13)

### Directory Structure
```
crates/
├── context-graph-cuda/       # ✅ ONLY crate with CUDA/FAISS FFI
│   └── src/
│       ├── ffi/
│       │   ├── mod.rs        # FFI module root
│       │   ├── cuda_driver.rs # CUDA Driver API FFI (cuInit, cuDeviceGet, etc.)
│       │   └── faiss.rs      # FAISS FFI (faiss_index_factory, etc.)
│       └── safe/
│           └── device.rs     # GpuDevice RAII wrapper (TASK-04)
├── context-graph-core/       # No FFI - only business logic
├── context-graph-mcp/        # No FFI - MCP server
├── context-graph-embeddings/ # No FFI - re-exports only
├── context-graph-graph/      # No FFI - re-exports only (faiss_ffi/mod.rs re-exports)
└── context-graph-storage/    # No FFI
```

### Existing CI Pipeline (`.github/workflows/ci.yml`)
- **Lines 1-404**: Complete CI workflow with check, test, coverage, docs, audit, benchmarks
- **NO FFI CONSOLIDATION CHECK EXISTS** - this is what TASK-05 creates

### Existing Scripts (`scripts/`)
- `benchmark-check.sh` (5191 bytes) - Reference for script style
- `check-ffi-consolidation.sh` - **DOES NOT EXIST** - this is what TASK-05 creates

### FFI Consolidation Status (CURRENT)
- All `extern "C"` blocks with CUDA/FAISS keywords are in `context-graph-cuda`
- Other crates have re-export modules (not FFI declarations):
  - `crates/context-graph-graph/src/index/faiss_ffi/mod.rs` - Re-exports from `context_graph_cuda::ffi::faiss`
  - This is ALLOWED - re-exports are not FFI declarations

---

## IMPLEMENTATION SPECIFICATION

### File 1: `scripts/check-ffi-consolidation.sh`

**Location**: `/home/cabdru/contextgraph/scripts/check-ffi-consolidation.sh`

**EXACT CONTENT** (copy verbatim - do NOT modify):

```bash
#!/usr/bin/env bash
#
# TASK-05: FFI Consolidation Gate
#
# Verifies that all CUDA/FAISS FFI declarations are in context-graph-cuda only.
# Constitution Reference: ARCH-06
#
# Exit codes:
#   0 - All FFI consolidated correctly
#   1 - FFI violation found outside context-graph-cuda
#
# Usage:
#   ./scripts/check-ffi-consolidation.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "  FFI Consolidation Check (ARCH-06)"
echo "========================================"
echo ""

# Change to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Verify crates directory exists
if [ ! -d "crates" ]; then
    echo -e "${RED}ERROR: crates/ directory not found${NC}"
    echo "Expected repository structure with crates/ at root"
    exit 1
fi

echo "Scanning for CUDA/FAISS FFI outside context-graph-cuda..."
echo ""

# Find all .rs files with extern "C" blocks that contain CUDA/FAISS keywords
# EXCLUDING context-graph-cuda crate
#
# Logic:
# 1. Find all .rs files NOT in context-graph-cuda
# 2. Filter to files containing 'extern "C"' literally
# 3. Further filter to files containing CUDA/FAISS identifiers
#
# CUDA identifiers: cuInit, cuDevice, cuCtx, cuMem, CUresult, CUdevice, CUcontext
# FAISS identifiers: faiss_, FaissIndex, FaissGpu, FAISS_

VIOLATIONS=""

# Step 1: Find all .rs files outside context-graph-cuda
while IFS= read -r -d '' file; do
    # Skip non-.rs files (shouldn't happen but be safe)
    [[ "$file" != *.rs ]] && continue

    # Check if file contains 'extern "C"' block
    if grep -q 'extern "C"' "$file" 2>/dev/null; then
        # Check if file contains CUDA or FAISS FFI identifiers
        # These patterns match actual FFI declarations, not re-exports or comments
        if grep -qE '(fn\s+cu[A-Z][a-zA-Z_]+\s*\(|fn\s+faiss_[a-z_]+\s*\(|type\s+CU[a-z]+\s*=|type\s+Faiss[A-Za-z]+\s*=)' "$file" 2>/dev/null; then
            VIOLATIONS="${VIOLATIONS}${file}\n"
            echo -e "${RED}VIOLATION:${NC} $file"
            echo "  Contains extern \"C\" block with CUDA/FAISS function or type declarations"
            # Show the offending lines for debugging
            echo "  Offending patterns:"
            grep -nE '(fn\s+cu[A-Z][a-zA-Z_]+\s*\(|fn\s+faiss_[a-z_]+\s*\(|type\s+CU[a-z]+\s*=|type\s+Faiss[A-Za-z]+\s*=)' "$file" 2>/dev/null | head -5 | while read -r line; do
                echo "    $line"
            done
            echo ""
        fi
    fi
done < <(find crates -name "*.rs" -not -path "*/context-graph-cuda/*" -print0 2>/dev/null)

echo "========================================"
echo "  Results"
echo "========================================"
echo ""

if [ -n "$VIOLATIONS" ]; then
    echo -e "${RED}FAILED: CUDA/FAISS FFI found outside context-graph-cuda${NC}"
    echo ""
    echo "The following files violate ARCH-06:"
    echo -e "$VIOLATIONS"
    echo ""
    echo "Action Required:"
    echo "  1. Move all extern \"C\" declarations to context-graph-cuda/src/ffi/"
    echo "  2. Use re-exports (pub use) in other crates instead of FFI declarations"
    echo ""
    echo "See: docs/specs/tasks/TASK-02.md (CUDA FFI)"
    echo "See: docs/specs/tasks/TASK-03.md (FAISS FFI)"
    exit 1
else
    echo -e "${GREEN}PASSED: All CUDA/FAISS FFI consolidated in context-graph-cuda${NC}"
    echo ""
    echo "Verified locations:"
    echo "  ✅ crates/context-graph-cuda/src/ffi/cuda_driver.rs"
    echo "  ✅ crates/context-graph-cuda/src/ffi/faiss.rs"
    exit 0
fi
```

### File 2: Addition to `.github/workflows/ci.yml`

**Location**: `/home/cabdru/contextgraph/.github/workflows/ci.yml`

**Action**: ADD the following job AFTER the `check` job (around line 60):

```yaml
  # =============================================================================
  # FFI Consolidation Gate (TASK-05)
  # =============================================================================
  # Enforces ARCH-06: All CUDA/FAISS FFI must be in context-graph-cuda only
  ffi-consolidation:
    name: FFI Consolidation
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Check FFI consolidation
        run: |
          chmod +x scripts/check-ffi-consolidation.sh
          ./scripts/check-ffi-consolidation.sh
```

---

## CONSTRAINTS (MUST FOLLOW)

| ID | Constraint | Rationale |
|----|------------|-----------|
| C1 | Script MUST be POSIX-compliant (works in sh, bash, dash) | CI runners may have different shells |
| C2 | Script MUST exit 0 on success, 1 on violation | Standard CI gate behavior |
| C3 | Script MUST provide clear file paths in errors | Enables quick debugging |
| C4 | Script MUST NOT flag re-exports as violations | `pub use` is allowed; `extern "C"` with FFI declarations is not |
| C5 | CI job MUST run before test job | Fail fast on architecture violations |
| C6 | Script MUST handle missing crates/ directory | Graceful error for corrupt checkouts |

---

## SOURCE OF TRUTH

The final result is verified by:

1. **Script file exists**: `/home/cabdru/contextgraph/scripts/check-ffi-consolidation.sh`
2. **Script is executable**: `chmod +x` must be applied
3. **CI job exists**: `.github/workflows/ci.yml` contains `ffi-consolidation` job
4. **Script passes**: Running `./scripts/check-ffi-consolidation.sh` exits with code 0

---

## FULL STATE VERIFICATION PROTOCOL

After completing the implementation logic, you MUST perform Full State Verification:

### Step 1: Define Source of Truth

| Artifact | Location | Verification Method |
|----------|----------|---------------------|
| Script file | `scripts/check-ffi-consolidation.sh` | `ls -la scripts/check-ffi-consolidation.sh` |
| Script executable | File permissions | `stat -c %a scripts/check-ffi-consolidation.sh` should show 755 |
| CI job | `.github/workflows/ci.yml` | `grep -A10 'ffi-consolidation:' .github/workflows/ci.yml` |
| Script passes | Exit code 0 | `./scripts/check-ffi-consolidation.sh && echo "PASS"` |

### Step 2: Execute & Inspect

Run the script and manually verify:

```bash
# Step 2a: Run the script
./scripts/check-ffi-consolidation.sh
echo "Exit code: $?"

# Step 2b: Verify exit code is 0
if [ $? -eq 0 ]; then
    echo "✅ FFI consolidation check PASSED"
else
    echo "❌ FFI consolidation check FAILED"
fi

# Step 2c: Verify CI job exists
grep -q "ffi-consolidation:" .github/workflows/ci.yml && echo "✅ CI job exists" || echo "❌ CI job missing"
```

### Step 3: Boundary & Edge Case Audit

Execute these 3 edge cases and print state before/after:

#### Edge Case 1: No Violations (Current State)

**Before State**:
```bash
find crates -name "*.rs" -not -path "*/context-graph-cuda/*" -exec grep -l 'extern "C"' {} \; | wc -l
# Expected: 0 or only re-export files (which don't match the FFI pattern)
```

**Action**: Run `./scripts/check-ffi-consolidation.sh`

**After State**:
```
Exit code: 0
Output: "PASSED: All CUDA/FAISS FFI consolidated in context-graph-cuda"
```

#### Edge Case 2: Simulate Violation

**Before State**:
```bash
# Create temporary violation file
mkdir -p /tmp/violation_test
cat > /tmp/violation_test/bad_ffi.rs << 'EOF'
extern "C" {
    fn cuInit(flags: u32) -> i32;
}
EOF
```

**Action**: Copy to crates and run script (then clean up)
```bash
cp /tmp/violation_test/bad_ffi.rs crates/context-graph-core/src/
./scripts/check-ffi-consolidation.sh
echo "Exit code: $?"
rm crates/context-graph-core/src/bad_ffi.rs
```

**After State**:
```
Exit code: 1
Output contains: "VIOLATION:" and "crates/context-graph-core/src/bad_ffi.rs"
```

#### Edge Case 3: Re-export Files (Should NOT Trigger)

**Before State**:
```bash
# This file exists and re-exports FFI (allowed)
cat crates/context-graph-graph/src/index/faiss_ffi/mod.rs | head -5
```

**Action**: Run script

**After State**:
```
Exit code: 0
No violation reported for faiss_ffi/mod.rs (because it uses `pub use`, not `extern "C"` declarations)
```

### Step 4: Evidence of Success

Provide a log showing the actual state after execution:

```bash
echo "=== EVIDENCE OF SUCCESS ==="
echo ""
echo "1. Script file exists:"
ls -la scripts/check-ffi-consolidation.sh

echo ""
echo "2. Script is executable:"
file scripts/check-ffi-consolidation.sh

echo ""
echo "3. Script output:"
./scripts/check-ffi-consolidation.sh

echo ""
echo "4. CI job in workflow:"
grep -A15 "ffi-consolidation:" .github/workflows/ci.yml

echo ""
echo "5. Exit codes verified:"
./scripts/check-ffi-consolidation.sh && echo "PASS: Exit 0" || echo "FAIL: Exit non-zero"
```

---

## MANUAL TESTING PROTOCOL

### Test 1: Happy Path - Clean Repository

**Synthetic Input**: Current repository state (no violations)

**Expected Output**:
- Exit code: 0
- Stdout contains: "PASSED: All CUDA/FAISS FFI consolidated in context-graph-cuda"

**Verification**:
```bash
./scripts/check-ffi-consolidation.sh
echo "Exit code: $?"
# MUST show: Exit code: 0
```

### Test 2: Violation Detection

**Synthetic Input**: Add temporary FFI violation

**Expected Output**:
- Exit code: 1
- Stdout contains: "VIOLATION:" with file path
- Stdout contains: "FAILED: CUDA/FAISS FFI found outside context-graph-cuda"

**Verification**:
```bash
# Create violation
echo 'extern "C" { fn cuInit(flags: u32) -> i32; }' > crates/context-graph-core/src/violation_test.rs

# Run check
./scripts/check-ffi-consolidation.sh
EXIT_CODE=$?

# Clean up
rm crates/context-graph-core/src/violation_test.rs

# Verify
[ $EXIT_CODE -eq 1 ] && echo "✅ Violation detected correctly" || echo "❌ Should have detected violation"
```

### Test 3: Re-export Files Not Flagged

**Synthetic Input**: Existing re-export file `crates/context-graph-graph/src/index/faiss_ffi/mod.rs`

**Expected Output**:
- Exit code: 0
- Re-export file NOT listed as violation (it uses `pub use`, not `extern "C"` declarations)

**Verification**:
```bash
# Verify re-export file exists
cat crates/context-graph-graph/src/index/faiss_ffi/mod.rs | grep "pub use"
# Should show: pub use context_graph_cuda::ffi::faiss::...

# Run check
./scripts/check-ffi-consolidation.sh
# MUST exit 0 and NOT list faiss_ffi/mod.rs as violation
```

---

## FILES TO CREATE

| Path | Action | Size |
|------|--------|------|
| `scripts/check-ffi-consolidation.sh` | CREATE | ~2.5KB |

## FILES TO MODIFY

| Path | Action | Lines Changed |
|------|--------|---------------|
| `.github/workflows/ci.yml` | ADD `ffi-consolidation` job after line 60 | +15 lines |

---

## VERIFICATION COMMANDS

Run ALL of these after implementation:

```bash
# 1. Verify script exists and is executable
ls -la scripts/check-ffi-consolidation.sh
# Expected: -rwxr-xr-x ... check-ffi-consolidation.sh

# 2. Run script locally
./scripts/check-ffi-consolidation.sh
# Expected: Exit 0, "PASSED" message

# 3. Verify CI job exists
grep -A10 "ffi-consolidation:" .github/workflows/ci.yml
# Expected: Shows job definition with checkout and script run

# 4. Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo "YAML valid" || echo "YAML invalid"
# Expected: "YAML valid"

# 5. Test violation detection (temporary)
echo 'extern "C" { fn cuInit(flags: u32) -> i32; }' > /tmp/test_violation.rs
cp /tmp/test_violation.rs crates/context-graph-core/src/
./scripts/check-ffi-consolidation.sh && echo "UNEXPECTED PASS" || echo "CORRECTLY DETECTED VIOLATION"
rm crates/context-graph-core/src/test_violation.rs
```

---

## ANTI-PATTERNS (DO NOT DO)

| Anti-Pattern | Why It's Wrong | Correct Approach |
|--------------|----------------|------------------|
| Flagging re-export files | `pub use` is allowed | Only flag `extern "C"` with FFI declarations |
| Using grep without `extern "C"` check | Would flag imports/comments | Require both `extern "C"` AND FFI patterns |
| Hardcoding file paths | Fragile to renames | Use `find` with patterns |
| Silent failures | CI needs clear output | Always print what was checked |
| Non-zero exit on warnings | Breaks CI unnecessarily | Only exit 1 on actual violations |

---

## REFERENCES

- Constitution `arch_rules.ARCH-06`: CUDA FFI only in context-graph-cuda
- TASK-02 (completed): CUDA Driver FFI consolidation
- TASK-03 (completed): FAISS FFI consolidation
- TASK-04 (completed): GpuDevice RAII wrapper (dependency)

---

## COMPLETION CHECKLIST

- [ ] `scripts/check-ffi-consolidation.sh` created
- [ ] Script is executable (`chmod +x`)
- [ ] Script exits 0 on clean repository
- [ ] Script exits 1 when violation added
- [ ] CI job added to `.github/workflows/ci.yml`
- [ ] YAML syntax is valid
- [ ] All verification commands pass
- [ ] Edge cases tested (no violations, violation detection, re-exports)
- [ ] Evidence of success logged

---

## ERROR HANDLING REQUIREMENTS

**NO BACKWARDS COMPATIBILITY. FAIL FAST.**

If the script encounters any of these conditions, it MUST:

| Condition | Action | Exit Code |
|-----------|--------|-----------|
| `crates/` directory missing | Print error, exit | 1 |
| FFI violation found | Print file path + line numbers, exit | 1 |
| No `.rs` files found | Print warning, exit success (empty is valid) | 0 |
| `find` command fails | Error message, exit | 1 |

The script MUST NOT:
- Silently ignore errors
- Create workarounds for missing directories
- Continue after detecting violations
- Use mock data or stubs

---

## DEPENDENCY VERIFICATION

Before starting this task, verify TASK-04 is complete:

```bash
# TASK-04 created these files:
ls -la crates/context-graph-cuda/src/safe/device.rs
# Expected: File exists with GpuDevice implementation

ls -la crates/context-graph-cuda/src/safe/mod.rs
# Expected: File exists with module exports

# TASK-04 FFI additions:
grep "cuCtxCreate_v2" crates/context-graph-cuda/src/ffi/cuda_driver.rs
# Expected: Function declaration found
```

If any of these checks fail, TASK-04 is NOT complete and you cannot proceed with TASK-05.
