# TASK-ATC-P2-002: Extend DomainThresholds Struct with New Threshold Fields

**Version:** 4.0
**Status:** COMPLETED
**Layer:** Foundation
**Sequence:** 2
**Implements:** REQ-ATC-006, REQ-ATC-007
**Depends On:** TASK-ATC-P2-001 (COMPLETE)
**Estimated Complexity:** Medium
**Priority:** P2
**Completion Date:** 2026-01-11
**Last Verified:** 2026-01-12

---

## Metadata

```yaml
id: TASK-ATC-P2-002
title: Extend DomainThresholds Struct with New Threshold Fields
status: completed
layer: foundation
sequence: 2
implements:
  - REQ-ATC-006
  - REQ-ATC-007
depends_on:
  - TASK-ATC-P2-001  # COMPLETE
estimated_complexity: medium
completion_date: "2026-01-11"
```

---

## Completion Summary

This task is **COMPLETE**. All deliverables exist and all 77 ATC tests pass.

### What Was Implemented

| Component | Location | Status |
|-----------|----------|--------|
| `DomainThresholds` struct | `crates/context-graph-core/src/atc/domain.rs:96-129` | 19 fields (18 f32 + 1 Domain) |
| `DomainThresholds::new()` | `domain.rs:139-196` | Domain-aware initialization |
| `DomainThresholds::is_valid()` | `domain.rs:242-280` | Range + monotonicity validation |
| `DomainThresholds::clamp()` | `domain.rs:286-316` | Range clamping for all fields |
| `DomainThresholds::blend_with_similar()` | `domain.rs:201-236` | Transfer learning blend |
| `ThresholdAccessor` trait | `accessor.rs:39-52` | Dynamic threshold lookup |
| `impl ThresholdAccessor for ATC` | `accessor.rs:54-106` | Full implementation |
| `THRESHOLD_NAMES` constant | `accessor.rs:9-34` | 18 threshold names |
| Unit tests (domain) | `domain.rs:434-849` | 23 tests |
| Unit tests (accessor) | `accessor.rs:108-264` | 11 tests |

### Test Verification

```bash
$ cargo test -p context-graph-core atc::
# Result: 77 passed, 0 failed
```

---

## Critical Rules

1. **NO BACKWARDS COMPATIBILITY** - System must work after changes or fail fast
2. **NO WORKAROUNDS OR FALLBACKS** - Errors must surface immediately with robust logging
3. **NO MOCK DATA IN TESTS** - Use real computations, verify actual system state
4. **FAIL FAST** - Invalid thresholds MUST panic or return `Result::Err`

---

## Current Codebase State

### DomainThresholds Struct (19 fields total)

**File:** `crates/context-graph-core/src/atc/domain.rs` lines 96-129

```rust
pub struct DomainThresholds {
    pub domain: Domain,

    // === Existing fields (6) ===
    pub theta_opt: f32,           // [0.60, 0.90] Optimal alignment
    pub theta_acc: f32,           // [0.55, 0.85] Acceptable alignment
    pub theta_warn: f32,          // [0.40, 0.70] Warning alignment
    pub theta_dup: f32,           // [0.80, 0.98] Duplicate detection
    pub theta_edge: f32,          // [0.50, 0.85] Edge creation
    pub confidence_bias: f32,     // Domain confidence adjustment

    // === GWT thresholds (3) ===
    pub theta_gate: f32,          // [0.65, 0.95] GW broadcast gate
    pub theta_hypersync: f32,     // [0.90, 0.99] Hypersync detection
    pub theta_fragmentation: f32, // [0.35, 0.65] Fragmentation warning

    // === Layer thresholds (3) ===
    pub theta_memory_sim: f32,    // [0.35, 0.75] Memory similarity
    pub theta_reflex_hit: f32,    // [0.70, 0.95] Reflex cache hit
    pub theta_consolidation: f32, // [0.05, 0.30] Consolidation trigger

    // === Dream thresholds (3) ===
    pub theta_dream_activity: f32,  // [0.05, 0.30] Dream trigger
    pub theta_semantic_leap: f32,   // [0.50, 0.90] REM exploration
    pub theta_shortcut_conf: f32,   // [0.50, 0.85] Shortcut confidence

    // === Classification thresholds (2) ===
    pub theta_johari: f32,          // [0.35, 0.65] Johari boundary
    pub theta_blind_spot: f32,      // [0.35, 0.65] Blind spot detection

    // === Autonomous thresholds (2) ===
    pub theta_obsolescence_low: f32,  // [0.20, 0.50] Low relevance
    pub theta_obsolescence_high: f32, // [0.65, 0.90] High confidence
}
```

### ThresholdAccessor Trait

**File:** `crates/context-graph-core/src/atc/accessor.rs`

```rust
pub const THRESHOLD_NAMES: &[&str] = &[
    "theta_opt", "theta_acc", "theta_warn", "theta_dup", "theta_edge",
    "theta_gate", "theta_hypersync", "theta_fragmentation",
    "theta_memory_sim", "theta_reflex_hit", "theta_consolidation",
    "theta_dream_activity", "theta_semantic_leap", "theta_shortcut_conf",
    "theta_johari", "theta_blind_spot",
    "theta_obsolescence_low", "theta_obsolescence_high",
];

pub trait ThresholdAccessor {
    fn get_threshold(&self, name: &str, domain: Domain) -> Option<f32>;
    fn get_threshold_or_general(&self, name: &str, domain: Domain) -> f32;
    fn observe_threshold_usage(&mut self, name: &str, value: f32);
    fn list_threshold_names() -> &'static [&'static str];
}
```

### Domain Strictness Values

| Domain | Strictness | Description |
|--------|------------|-------------|
| Code | 0.9 | Strict thresholds, low FP tolerance |
| Medical | 1.0 | Very strict, high causal weight |
| Legal | 0.8 | Moderate, high semantic precision |
| Creative | 0.2 | Loose thresholds, exploration encouraged |
| Research | 0.5 | Balanced, novelty valued |
| General | 0.5 | Default priors |

---

## Full State Verification (FSV) - COMPLETED

### 1. Source of Truth

| Data | Source | Verification Method |
|------|--------|---------------------|
| DomainThresholds fields | `domain.rs` struct definition | Compile + field count |
| Field ranges | `is_valid()` method | Unit tests with boundary values |
| Domain strictness scaling | `new()` method | Unit tests comparing domains |
| ThresholdAccessor | `accessor.rs` | Unit tests for all 18 names |

### 2. Execute & Inspect - VERIFIED

```bash
# Compile check
$ cargo check -p context-graph-core
# Result: PASSED (with unrelated warnings)

# Run ATC tests
$ cargo test -p context-graph-core atc:: 2>&1 | tail -20
# Result: 77 passed; 0 failed

# Verify field count (grep for pub theta_ fields)
$ rg "pub theta_" crates/context-graph-core/src/atc/domain.rs -c
# Result: 18 (excluding domain field which is Domain type)

# Verify THRESHOLD_NAMES count
$ rg "\"theta_" crates/context-graph-core/src/atc/accessor.rs -c
# Result: 18
```

### 3. Boundary & Edge Case Audit - VERIFIED

| Edge Case | Input | Expected | Test Name | Result |
|-----------|-------|----------|-----------|--------|
| Min strictness domain | `Domain::Creative` (0.2) | theta_gate ≈ 0.78 | `test_creative_loosest_thresholds` | PASS |
| Max strictness domain | `Domain::Medical` (1.0) | theta_gate ≈ 0.90 | `test_medical_strictest_thresholds` | PASS |
| Invalid theta_gate | 0.50 (below min) | `is_valid() == false` | `test_is_valid_fails_invalid_gate_below_min` | PASS |
| Invalid monotonicity | obsolescence_high < low | `is_valid() == false` | `test_is_valid_fails_invalid_obsolescence_monotonicity` | PASS |
| Blend 50/50 | alpha=0.5 | Values between both | `test_blend_includes_new_fields` | PASS |
| Unknown threshold | "theta_xyz" | `get_threshold() == None` | `test_get_threshold_unknown_returns_none` | PASS |

### 4. Evidence of Success

```
=== ATC Test Results (2026-01-12) ===
Total tests: 77
Passed: 77
Failed: 0

Key validations:
  ✓ test_extended_fields_exist - All 13 new fields in valid ranges
  ✓ test_domain_strictness_affects_new_thresholds - Code > Creative
  ✓ test_obsolescence_monotonicity - high > low for all 6 domains
  ✓ test_clamp_all_new_fields - All fields clamp correctly
  ✓ test_blend_includes_new_fields - Transfer learning works
  ✓ test_all_domains_valid_on_creation - All 6 domains valid
  ✓ test_field_count_is_19 - Exactly 19 f32 threshold fields
  ✓ test_get_threshold_known_names - All 18 names return values
  ✓ test_all_domains_have_all_thresholds - 6 domains × 18 thresholds
```

---

## Manual Testing Procedure

### Synthetic Test: Domain Comparison

Run this test to see actual computed values:

```bash
cargo test -p context-graph-core test_print_all_domain_thresholds -- --nocapture
```

**Expected Output Pattern:**

```
=== Domain Threshold Values (TASK-ATC-P2-002) ===

Medical (strictness=1.0):
  theta_gate: 0.900
  theta_hypersync: 0.970
  theta_fragmentation: 0.400
  theta_memory_sim: 0.650
  theta_reflex_hit: 0.900
  theta_consolidation: 0.200
  theta_dream_activity: 0.100
  theta_semantic_leap: 0.600
  theta_shortcut_conf: 0.800
  theta_johari: 0.500
  theta_blind_spot: 0.500
  theta_obsolescence_low: 0.400
  theta_obsolescence_high: 0.850

Creative (strictness=0.2):
  theta_gate: 0.780
  theta_hypersync: 0.938
  theta_fragmentation: 0.480
  theta_memory_sim: 0.530
  theta_reflex_hit: 0.820
  theta_consolidation: 0.120
  theta_dream_activity: 0.140
  theta_semantic_leap: 0.680
  theta_shortcut_conf: 0.720
  theta_johari: 0.500
  theta_blind_spot: 0.500
  theta_obsolescence_low: 0.320
  theta_obsolescence_high: 0.770
```

### Synthetic Test: ThresholdAccessor Roundtrip

```bash
cargo test -p context-graph-core atc::accessor::tests -- --nocapture
```

**Verifies:**
1. All 18 threshold names return `Some(f32)` for all 6 domains
2. Unknown names return `None`
3. Domain strictness affects threshold values correctly
4. Obsolescence monotonicity (high > low) holds for all domains

---

## Validation Commands

```bash
# Full ATC test suite
cargo test -p context-graph-core atc:: -- --nocapture

# Specific test categories
cargo test -p context-graph-core atc::domain::tests
cargo test -p context-graph-core atc::accessor::tests

# Clippy check
cargo clippy -p context-graph-core -- -D warnings

# Doc tests
cargo test -p context-graph-core --doc
```

---

## Acceptance Criteria Checklist

### Struct Extension
- [x] Struct has 19 fields (18 f32 + 1 Domain)
- [x] All domains initialize successfully
- [x] `is_valid()` checks all 19 fields
- [x] `clamp()` respects all ranges
- [x] `blend_with_similar()` includes all fields

### ThresholdAccessor
- [x] `THRESHOLD_NAMES` has 18 entries
- [x] `get_threshold()` returns correct values
- [x] `get_threshold_or_general()` provides fallback
- [x] Unknown threshold names return None

### Domain Behavior
- [x] Domain strictness scaling correct
- [x] Medical has strictest thresholds
- [x] Creative has loosest thresholds
- [x] Obsolescence monotonicity enforced

### Tests
- [x] All 77 ATC tests pass
- [x] No mock data used
- [x] Real computations verified

---

## Constitution Reference

### GWT Thresholds (`docs2/constitution.yaml` lines 220-236)

```yaml
gwt:
  kuramoto:
    thresholds: { coherent: "r≥0.8", fragmented: "r<0.5", hypersync: "r>0.95" }
  workspace:
    coherence_threshold: 0.8
```

### Dream Thresholds (`docs2/constitution.yaml` lines 254-280)

```yaml
dream:
  trigger: { activity: "<0.15", idle: "10min", entropy: ">0.7 for 5min" }
  phases:
    rem:
      blind_spot: { min_semantic_distance: 0.7 }
  amortized:
    confidence_threshold: 0.7
```

### Adaptive Thresholds (`docs2/constitution.yaml` lines 309-326)

```yaml
adaptive_thresholds:
  priors:
    θ_opt: [0.75, "[0.60,0.90]"]
    θ_acc: [0.70, "[0.55,0.85]"]
    θ_warn: [0.55, "[0.40,0.70]"]
    θ_dup: [0.90, "[0.80,0.98]"]
    θ_kur: [0.80, "[0.65,0.95]"]
```

---

## Next Steps

This task is COMPLETE. Subsequent tasks in the ATC migration sequence:

| Task | Description | Status |
|------|-------------|--------|
| TASK-ATC-P2-003 | GWT Thresholds Migration | Ready |
| TASK-ATC-P2-004 | Bio-Nervous Thresholds Migration | Ready |
| TASK-ATC-P2-005 | Dream Thresholds Migration | Ready |
| TASK-ATC-P2-006 | Johari Thresholds Migration | Ready |
| TASK-ATC-P2-007 | Autonomous Thresholds Migration | Ready |
| TASK-ATC-P2-008 | Validation Tests | Ready |

These tasks will modify the calling code to use `ATC.get_domain_thresholds(domain).theta_*` instead of hardcoded constants.

---

## Notes

- All threshold ranges derived from `constitution.yaml` and `threshold-inventory.yaml`
- Domain strictness scaling follows existing `theta_opt/acc/warn` logic
- `theta_johari` and `theta_blind_spot` are fixed at 0.5 per constitution but stored for future flexibility
- The accessor trait enables both static (field access) and dynamic (name-based) threshold retrieval
- **NO MOCK DATA** - All tests use actual `DomainThresholds` instances
- **FAIL FAST** - Invalid configurations cause test failures, not silent fallbacks

---

**Created:** 2026-01-11
**Updated:** 2026-01-12
**Author:** AI Coding Agent
**Status:** COMPLETED
