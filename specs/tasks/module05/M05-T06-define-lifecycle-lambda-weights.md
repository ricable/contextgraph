---
id: "M05-T06"
title: "Define LifecycleLambdaWeights Struct (Marblestone)"
status: "COMPLETE"
priority: "critical"
layer: "foundation"
sequence: 6
depends_on: ["M05-T05"]
verified: "2026-01-04"
git_commit: "f521803"
test_count: 27
---

# STATUS: COMPLETE

**This task is FULLY IMPLEMENTED and VERIFIED.** The implementation exists at the paths below with 27 passing tests. Do NOT re-implement.

## Source of Truth

| What | Location |
|------|----------|
| Implementation | `crates/context-graph-utl/src/lifecycle/lambda.rs` (642 lines) |
| Module Export | `crates/context-graph-utl/src/lifecycle/mod.rs` (line 52) |
| Crate Re-export | `crates/context-graph-utl/src/lib.rs` (line 58) |
| Error Types | `crates/context-graph-utl/src/error.rs` (lines 27-36, 155-172) |
| Stage Enum | `crates/context-graph-utl/src/lifecycle/stage.rs` |
| Config | `crates/context-graph-utl/src/config.rs` (`LifecycleConfig`) |

## Verification Commands

```bash
# VERIFY 1: Run all lifecycle tests (expect 27+ pass)
cargo test -p context-graph-utl lifecycle -- --nocapture 2>&1 | grep -E "(test |passed|failed)"

# VERIFY 2: Confirm lambda.rs exists with real implementation
wc -l crates/context-graph-utl/src/lifecycle/lambda.rs
# Expected: 642 lines

# VERIFY 3: Run lambda-specific tests
cargo test -p context-graph-utl test_for_stage -- --nocapture
cargo test -p context-graph-utl test_apply -- --nocapture
cargo test -p context-graph-utl test_lerp -- --nocapture

# VERIFY 4: Confirm re-exports work
cargo test -p context-graph-utl test_lifecycle_re_exports -- --nocapture

# VERIFY 5: Doc tests pass
cargo test -p context-graph-utl --doc -- lifecycle

# VERIFY 6: Clippy passes
cargo clippy -p context-graph-utl -- -D warnings
```

## What Actually Exists

### Struct: `LifecycleLambdaWeights`

**Location:** `crates/context-graph-utl/src/lifecycle/lambda.rs:49-56`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LifecycleLambdaWeights {
    /// Lambda weight for surprise/novelty (lambda_s).
    lambda_s: f32,
    /// Lambda weight for coherence/consolidation (lambda_c).
    lambda_c: f32,
}
```

### Implemented Methods

| Method | Signature | Line |
|--------|-----------|------|
| `new` | `(lambda_s: f32, lambda_c: f32) -> UtlResult<Self>` | 87 |
| `new_unchecked` | `(lambda_s: f32, lambda_c: f32) -> Self` | 123 |
| `for_stage` | `(stage: LifecycleStage) -> Self` | 154 |
| `for_interaction_count` | `(count: u64) -> Self` | 193 |
| `interpolated` | `(count: u64, config: &LifecycleConfig) -> Self` | 228 |
| `lerp` (private) | `(a: &Self, b: &Self, t: f32) -> Self` | 294 |
| `lambda_s` | `(&self) -> f32` | 307 |
| `lambda_c` | `(&self) -> f32` | 315 |
| `apply` | `(&self, delta_s: f32, delta_c: f32) -> f32` | 346 |
| `is_valid` | `(&self) -> bool` | 357 |
| `focus` | `(&self) -> &'static str` | 384 |

### Trait Implementations

- `Default` (returns Infancy weights) - line 395
- `Clone`, `Copy`, `PartialEq` - line 49
- `Debug` - line 49
- `Serialize`, `Deserialize` - line 49

### Weight Values (Constitution Compliant)

| Stage | lambda_s | lambda_c | Stance |
|-------|----------|----------|--------|
| Infancy | 0.7 | 0.3 | capture-novelty |
| Growth | 0.5 | 0.5 | balanced |
| Maturity | 0.3 | 0.7 | curation-coherence |

### Error Type

**Location:** `crates/context-graph-utl/src/error.rs:27-36`

```rust
#[error("Invalid lambda weights: novelty={novelty}, consolidation={consolidation}. {reason}")]
InvalidLambdaWeights {
    novelty: f32,
    consolidation: f32,
    reason: String,
}
```

Helper methods at lines 155-172:
- `UtlError::lambda_sum_error(novelty, consolidation)`
- `UtlError::negative_lambda(novelty, consolidation)`

## Mathematical Foundation

From `constitution.yaml` lines 164-167:

```yaml
lifecycle:  # Marblestone lambda weights
  infancy:  { n: "0-50",   lambda_s: 0.7, lambda_c: 0.3, stance: "capture-novelty" }
  growth:   { n: "50-500", lambda_s: 0.5, lambda_c: 0.5, stance: "balanced" }
  maturity: { n: "500+",  lambda_s: 0.3, lambda_c: 0.7, stance: "curation-coherence" }
```

Weight application formula (REQ-UTL-034):
```
L_weighted = lambda_s * delta_s + lambda_c * delta_c
```

Invariant: `lambda_s + lambda_c = 1.0` (enforced in `new()` with EPSILON=0.001)

## Tests (27 total in lambda.rs)

All tests use **REAL DATA** - no mocks:

| Test | Purpose |
|------|---------|
| `test_new_valid_weights` | Valid construction |
| `test_new_invalid_sum` | Rejects sum != 1.0 |
| `test_new_negative_weights` | Rejects negative values |
| `test_new_weights_exceed_one` | Rejects > 1.0 values |
| `test_for_stage_infancy` | Infancy: 0.7/0.3 |
| `test_for_stage_growth` | Growth: 0.5/0.5 |
| `test_for_stage_maturity` | Maturity: 0.3/0.7 |
| `test_for_interaction_count` | Stage from count |
| `test_interpolated_no_smoothing` | Discrete transitions |
| `test_interpolated_with_smoothing` | Smooth transitions |
| `test_apply` | Weight application formula |
| `test_focus` | Dominance detection |
| `test_default` | Default = Infancy |
| `test_serialization` | JSON roundtrip |
| `test_is_valid` | Invariant check |
| `test_lerp` | Interpolation |
| `test_equality` | PartialEq |
| `test_clone_and_copy` | Copy semantics |
| `test_debug` | Debug formatting |
| `test_all_stages_weights_sum_to_one` | Invariant for all stages |
| `test_weights_decrease_surprise_with_maturity` | Monotonic decrease |
| `test_weights_increase_coherence_with_maturity` | Monotonic increase |

## Full State Verification Protocol

### Step 1: Execute & Inspect Source of Truth

```bash
# Run tests and capture output
cargo test -p context-graph-utl lifecycle --no-fail-fast 2>&1 | tee /tmp/lambda_tests.log

# Verify specific test output
grep "test_all_stages_weights_sum_to_one" /tmp/lambda_tests.log
grep "test_apply" /tmp/lambda_tests.log
```

### Step 2: Manual Edge Case Audit

**Case 1: Empty/Zero Inputs**
```bash
cargo test -p context-graph-utl test_apply -- --nocapture 2>&1
# Verify output: weights.apply(0.0, 0.0) == 0.0
```

**Case 2: Maximum Values**
```bash
cargo test -p context-graph-utl test_new_valid_weights -- --nocapture 2>&1
# Verify: lambda_s=1.0, lambda_c=0.0 is valid
```

**Case 3: Invalid Sum**
```bash
cargo test -p context-graph-utl test_new_invalid_sum -- --nocapture 2>&1
# Verify: Returns UtlError::InvalidLambdaWeights
```

### Step 3: Evidence of Success

After running verification, check:
```bash
# Show test summary
cargo test -p context-graph-utl lifecycle 2>&1 | tail -3
# Expected: "test result: ok. XX passed; 0 failed"

# Verify file exists and has content
ls -la crates/context-graph-utl/src/lifecycle/lambda.rs
# Expected: 21600+ bytes
```

## Sherlock Holmes Final Verification

**MANDATORY:** After any changes to this module, run sherlock-holmes verification:

```
Use Task tool with subagent_type="sherlock-holmes" and prompt:
"Forensic audit of LifecycleLambdaWeights implementation:
1. Verify lambda.rs at crates/context-graph-utl/src/lifecycle/lambda.rs exists
2. Run: cargo test -p context-graph-utl lifecycle --no-fail-fast
3. Confirm 27+ tests pass with 0 failures
4. Verify invariant: all stage weights sum to 1.0
5. Check re-export in lib.rs line 58
6. Verify UtlError::InvalidLambdaWeights exists in error.rs
Report: PASS/FAIL with evidence for each check"
```

## Dependencies

- **Uses:** `LifecycleStage` (M05-T05)
- **Uses:** `UtlError` (M05-T23)
- **Uses:** `LifecycleConfig` (M05-T07)
- **Used By:** `LifecycleManager` (M05-T19), `compute_learning_magnitude` (lib.rs)

## Anti-Patterns to Avoid (From Constitution)

- AP-009: Never allow NaN/Infinity - clamp to valid range (IMPLEMENTED via `is_valid()`)
- AP-003: No magic numbers - constants `EPSILON=0.001` and stage thresholds are defined

## If You Need to Modify

1. **DO NOT** change weight values without updating constitution.yaml
2. **DO NOT** remove invariant validation in `new()`
3. **DO NOT** add mock data to tests - use real `LifecycleStage` values
4. **ALWAYS** run full test suite after changes
5. **ALWAYS** verify re-exports still work

---

*Task Verified: 2026-01-04*
*Implementation Status: COMPLETE*
*Test Coverage: 27 tests passing*
*Git Reference: f521803 feat(utl): complete context-graph-utl crate with 453 tests passing*
