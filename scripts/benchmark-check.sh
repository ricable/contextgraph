#!/bin/bash
#
# TASK-TEST-P2-002: Benchmark regression detection script
#
# This script compares Criterion benchmark results against a baseline
# and fails if any benchmark regresses by more than REGRESSION_THRESHOLD.
#
# Usage:
#   ./scripts/benchmark-check.sh [baseline_dir] [current_dir]
#
# If no arguments provided, uses default Criterion output locations.
#
# Exit codes:
#   0 - No regression detected
#   1 - Regression detected (>5%)
#   2 - Baseline not found (first run)
#   3 - Error during comparison

set -euo pipefail

REGRESSION_THRESHOLD=5.0  # percent
CRITERION_DIR="${CRITERION_DIR:-target/criterion}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "  Benchmark Regression Check"
echo "  Threshold: ${REGRESSION_THRESHOLD}%"
echo "========================================"
echo ""

# Check if bc is available for floating point arithmetic
if ! command -v bc &> /dev/null; then
    echo -e "${YELLOW}WARNING: 'bc' not found. Installing may be required for full functionality.${NC}"
    echo "Falling back to integer comparisons only."
    USE_BC=false
else
    USE_BC=true
fi

# Check if jq is available for JSON parsing
if ! command -v jq &> /dev/null; then
    echo -e "${RED}ERROR: 'jq' is required but not installed.${NC}"
    echo "Install it with: apt-get install jq (Debian/Ubuntu) or brew install jq (macOS)"
    exit 3
fi

# Check if Criterion results exist
if [ ! -d "$CRITERION_DIR" ]; then
    echo -e "${YELLOW}WARNING: No benchmark results found at $CRITERION_DIR${NC}"
    echo "Run 'cargo bench' first to generate results."
    exit 2
fi

# Find all benchmark estimates.json files
ESTIMATES=$(find "$CRITERION_DIR" -name "estimates.json" -type f 2>/dev/null || true)

if [ -z "$ESTIMATES" ]; then
    echo -e "${YELLOW}WARNING: No estimates.json files found${NC}"
    echo "Run 'cargo bench' to generate benchmark data."
    exit 2
fi

# Track regression status
REGRESSIONS_FOUND=0
TOTAL_BENCHMARKS=0
IMPROVEMENTS_FOUND=0
SUMMARY=""

# Parse each benchmark
while IFS= read -r estimate_file; do
    # Skip if this is a change estimate (we want the base estimate)
    if [[ "$estimate_file" == *"/change/"* ]]; then
        continue
    fi

    # Extract benchmark name from path
    BENCH_NAME=$(dirname "$estimate_file" | sed "s|$CRITERION_DIR/||" | sed 's|/new||' | sed 's|/base||')

    # Try to get the mean value (in nanoseconds)
    MEAN_NS=$(jq -r '.mean.point_estimate // empty' "$estimate_file" 2>/dev/null || echo "")

    if [ -z "$MEAN_NS" ]; then
        continue
    fi

    TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))

    # Convert to milliseconds for display
    if $USE_BC; then
        MEAN_MS=$(echo "scale=4; $MEAN_NS / 1000000" | bc)
    else
        MEAN_MS="$(echo "$MEAN_NS" | cut -d'.' -f1) ns"
    fi

    # Check for change file (Criterion's regression detection)
    CHANGE_FILE=$(dirname "$estimate_file")/change/estimates.json
    if [ -f "$CHANGE_FILE" ]; then
        CHANGE_PCT=$(jq -r '.mean.point_estimate // 0' "$CHANGE_FILE" 2>/dev/null || echo "0")

        if $USE_BC; then
            CHANGE_PCT=$(echo "scale=2; $CHANGE_PCT * 100" | bc)

            # Check if regression exceeds threshold
            IS_REGRESSION=$(echo "$CHANGE_PCT > $REGRESSION_THRESHOLD" | bc -l)
            IS_IMPROVEMENT=$(echo "$CHANGE_PCT < -$REGRESSION_THRESHOLD" | bc -l)

            if [ "$IS_REGRESSION" -eq 1 ]; then
                echo -e "${RED}REGRESSION:${NC} $BENCH_NAME"
                echo "  Mean: ${MEAN_MS}ms (${CHANGE_PCT}% slower)"
                SUMMARY="${SUMMARY}FAIL: $BENCH_NAME (+${CHANGE_PCT}%)\n"
                REGRESSIONS_FOUND=$((REGRESSIONS_FOUND + 1))
            elif [ "$IS_IMPROVEMENT" -eq 1 ]; then
                echo -e "${GREEN}IMPROVEMENT:${NC} $BENCH_NAME"
                echo "  Mean: ${MEAN_MS}ms (${CHANGE_PCT}% faster)"
                SUMMARY="${SUMMARY}PASS: $BENCH_NAME (${CHANGE_PCT}%)\n"
                IMPROVEMENTS_FOUND=$((IMPROVEMENTS_FOUND + 1))
            else
                echo -e "${GREEN}OK:${NC} $BENCH_NAME"
                echo "  Mean: ${MEAN_MS}ms (${CHANGE_PCT}% change)"
                SUMMARY="${SUMMARY}PASS: $BENCH_NAME\n"
            fi
        else
            echo "INFO: $BENCH_NAME"
            echo "  Mean: ${MEAN_MS}"
            SUMMARY="${SUMMARY}INFO: $BENCH_NAME\n"
        fi
    else
        echo "NEW: $BENCH_NAME"
        echo "  Mean: ${MEAN_MS}ms (no baseline)"
        SUMMARY="${SUMMARY}NEW: $BENCH_NAME\n"
    fi
done <<< "$ESTIMATES"

echo ""
echo "========================================"
echo "  Summary"
echo "========================================"
echo ""
echo "Total benchmarks: $TOTAL_BENCHMARKS"
echo "Regressions found: $REGRESSIONS_FOUND"
echo "Improvements found: $IMPROVEMENTS_FOUND"
echo ""

if [ $REGRESSIONS_FOUND -gt 0 ]; then
    echo -e "${RED}FAILED: $REGRESSIONS_FOUND benchmark(s) regressed by more than ${REGRESSION_THRESHOLD}%${NC}"
    echo ""
    echo "Regressions:"
    echo -e "$SUMMARY" | grep "^FAIL" || true
    exit 1
else
    echo -e "${GREEN}PASSED: No regressions detected${NC}"
    exit 0
fi
