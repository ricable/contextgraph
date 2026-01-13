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
        # FFI patterns include versioned functions (e.g., cuDeviceTotalMem_v2)
        if grep -qE '(fn\s+cu[A-Z][a-zA-Z_0-9]+\s*\(|fn\s+faiss_[a-z_0-9]+\s*\(|type\s+CU[a-z]+\s*=|type\s+Faiss[A-Za-z]+\s*=)' "$file" 2>/dev/null; then
            VIOLATIONS="${VIOLATIONS}${file}\n"
            echo -e "${RED}VIOLATION:${NC} $file"
            echo "  Contains extern \"C\" block with CUDA/FAISS function or type declarations"
            # Show the offending lines for debugging
            echo "  Offending patterns:"
            grep -nE '(fn\s+cu[A-Z][a-zA-Z_0-9]+\s*\(|fn\s+faiss_[a-z_0-9]+\s*\(|type\s+CU[a-z]+\s*=|type\s+Faiss[A-Za-z]+\s*=)' "$file" 2>/dev/null | head -5 | while read -r line; do
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
