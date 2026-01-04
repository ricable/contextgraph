---
id: "M05-T07"
title: "Define LifecycleConfig and StageConfig Structs"
description: |
  TASK ALREADY COMPLETE. LifecycleConfig and StageConfig exist in config.rs.
  This document serves as a verification reference.
layer: "foundation"
status: "COMPLETE"
priority: "high"
estimated_hours: 0
sequence: 7
depends_on:
  - "M05-T05"
  - "M05-T06"
spec_refs:
  - "constitution.yaml lines 164-167"
  - "contextprd.md Section 2.4"
actual_implementation:
  - path: "crates/context-graph-utl/src/config.rs"
    lines: "643-848"
    description: "LifecycleConfig (line 643) and StageConfig (line 730)"
  - path: "crates/context-graph-utl/src/lifecycle/mod.rs"
    description: "Re-exports LifecycleStage, LifecycleLambdaWeights, LifecycleManager"
  - path: "crates/context-graph-utl/src/lifecycle/manager.rs"
    description: "LifecycleManager uses LifecycleConfig"
  - path: "crates/context-graph-utl/src/lifecycle/lambda.rs"
    description: "LifecycleLambdaWeights.interpolated() uses LifecycleConfig"
---

# ⚠️ STATUS: COMPLETE - VERIFICATION ONLY

**This task was completed in commit `f521803`**. The structures exist and are fully tested with 369 passing tests.

## Actual Implementation Location

### LifecycleConfig (config.rs:643-674)

```rust
// crates/context-graph-utl/src/config.rs:643
pub struct LifecycleConfig {
    pub stages: Vec<StageConfig>,       // [Infancy, Growth, Maturity]
    pub auto_transition: bool,           // Enable automatic stage transitions
    pub transition_hysteresis: u64,      // Prevents rapid switching (default: 10)
    pub smooth_transitions: bool,        // Enable lambda interpolation
    pub smoothing_window: u64,           // Window size (default: 25)
}
```

### StageConfig (config.rs:730-760)

```rust
// crates/context-graph-utl/src/config.rs:730
pub struct StageConfig {
    pub name: String,                    // "Infancy", "Growth", "Maturity"
    pub min_interactions: u64,           // Stage entry threshold
    pub max_interactions: u64,           // Stage exit threshold (u64::MAX for Maturity)
    pub lambda_novelty: f32,             // λ_ΔS weight [0,1]
    pub lambda_consolidation: f32,       // λ_ΔC weight [0,1] (must sum to 1.0 with above)
    pub surprise_trigger: f32,           // ΔS_trig threshold
    pub coherence_trigger: f32,          // ΔC_trig threshold
    pub stance: String,                  // "capture-novelty", "balanced", "curation-coherence"
}
```

## Constitution Values (VERIFIED)

From `constitution.yaml:164-167`:

| Stage    | n Range   | λ_ΔS | λ_ΔC | ΔS_trig | ΔC_trig | Stance              |
|----------|-----------|------|------|---------|---------|---------------------|
| Infancy  | 0-50      | 0.7  | 0.3  | 0.9     | 0.2     | capture-novelty     |
| Growth   | 50-500    | 0.5  | 0.5  | 0.7     | 0.4     | balanced            |
| Maturity | 500+      | 0.3  | 0.7  | 0.6     | 0.5     | curation-coherence  |

## Verification Commands

```bash
# 1. Verify struct exists
grep -n "pub struct LifecycleConfig" crates/context-graph-utl/src/config.rs
# Expected: 643:pub struct LifecycleConfig {

# 2. Verify StageConfig exists
grep -n "pub struct StageConfig" crates/context-graph-utl/src/config.rs
# Expected: 730:pub struct StageConfig {

# 3. Run lifecycle tests (68 tests)
cargo test -p context-graph-utl lifecycle -- --nocapture 2>&1 | tail -10
# Expected: test result: ok. 68 passed; 0 failed;

# 4. Run all UTL tests (369 tests)
cargo test -p context-graph-utl --lib 2>&1 | tail -3
# Expected: test result: ok. 369 passed; 0 failed;

# 5. Verify config validation
cargo test -p context-graph-utl config -- --nocapture 2>&1 | grep -E "(test_.*|passed|failed)"
```

## Source of Truth Verification

The implementation lives in:
- **Primary**: `crates/context-graph-utl/src/config.rs` lines 643-848
- **Usage**: `crates/context-graph-utl/src/lifecycle/manager.rs` (consumes LifecycleConfig)
- **Usage**: `crates/context-graph-utl/src/lifecycle/lambda.rs` (interpolated() method)

### Evidence of Correct Implementation

```bash
# Verify default values match constitution
cargo test -p context-graph-utl test_stage_config --nocapture 2>&1
```

Expected output showing constitution-compliant values:
- `StageConfig::infancy()` → λ_novelty=0.7, λ_consolidation=0.3
- `StageConfig::growth()` → λ_novelty=0.5, λ_consolidation=0.5
- `StageConfig::maturity()` → λ_novelty=0.3, λ_consolidation=0.7

## Key Methods Available

### StageConfig

| Method | Description |
|--------|-------------|
| `StageConfig::infancy()` | Create Infancy config (0-50, λs=0.7, λc=0.3) |
| `StageConfig::growth()` | Create Growth config (50-500, λs=0.5, λc=0.5) |
| `StageConfig::maturity()` | Create Maturity config (500+, λs=0.3, λc=0.7) |
| `validate(&self) -> Result<(), String>` | Validate all fields including λ sum |

### LifecycleConfig

| Method | Description |
|--------|-------------|
| `LifecycleConfig::default()` | Standard 3-stage config |
| `LifecycleConfig::infancy_focused()` | Higher novelty weights |
| `LifecycleConfig::maturity_focused()` | Higher coherence weights |
| `stage_for_count(count: u64) -> Option<&StageConfig>` | Get stage by interaction count |
| `validate(&self) -> Result<(), Vec<String>>` | Validate all stages |

## Edge Cases Handled

1. **λ weight invariant**: `lambda_novelty + lambda_consolidation = 1.0` (validated with 0.001 epsilon)
2. **Stage boundaries**: Exclusive max prevents overlap (Infancy max=50 means 50 is Growth)
3. **Maturity unbounded**: Uses `u64::MAX` for max_interactions
4. **Hysteresis**: Prevents rapid stage switching (default 10 interactions)
5. **Smooth transitions**: Interpolated lambda weights near boundaries

## Related Components

- **LifecycleStage** (`lifecycle/stage.rs`): Enum with Infancy/Growth/Maturity
- **LifecycleLambdaWeights** (`lifecycle/lambda.rs`): Weight computation and interpolation
- **LifecycleManager** (`lifecycle/manager.rs`): State machine that uses LifecycleConfig

## Full State Verification Protocol

After any changes, execute:

```bash
# Step 1: Build clean
cargo build -p context-graph-utl 2>&1

# Step 2: Run ALL tests - NO MOCKS, REAL DATA
cargo test -p context-graph-utl --lib 2>&1 | tail -5
# MUST SEE: test result: ok. 369 passed; 0 failed;

# Step 3: Verify specific lifecycle tests
cargo test -p context-graph-utl lifecycle 2>&1 | grep -E "test result"
# MUST SEE: test result: ok. 68 passed; 0 failed;

# Step 4: Verify clippy passes
cargo clippy -p context-graph-utl -- -D warnings 2>&1

# Step 5: Verify doc tests
cargo test -p context-graph-utl --doc 2>&1 | tail -3
```

## Boundary & Edge Case Audit

### Case 1: Empty Stages Vector
```rust
let mut config = LifecycleConfig::default();
config.stages = vec![];
assert!(config.validate().is_err());
// Error: "No stages configured"
```

### Case 2: Invalid Lambda Sum
```rust
let mut stage = StageConfig::growth();
stage.lambda_novelty = 0.6;
stage.lambda_consolidation = 0.6; // Sum = 1.2
assert!(stage.validate().is_err());
// Error: "lambda_novelty + lambda_consolidation must equal 1.0"
```

### Case 3: Overlapping Stage Ranges
```rust
let mut config = LifecycleConfig::default();
config.stages[0].max_interactions = 100; // Infancy 0-100
config.stages[1].min_interactions = 50;   // Growth 50-500 (OVERLAP!)
assert!(config.validate().is_err());
```

## Sherlock Holmes Final Verification Checklist

- [ ] `cargo build -p context-graph-utl` succeeds
- [ ] `cargo test -p context-graph-utl --lib` shows 369 passed
- [ ] `cargo test -p context-graph-utl lifecycle` shows 68 passed
- [ ] `cargo clippy -p context-graph-utl -- -D warnings` no warnings
- [ ] `grep -n "pub struct LifecycleConfig" config.rs` returns line 643
- [ ] `grep -n "pub struct StageConfig" config.rs` returns line 730
- [ ] Default values match constitution.yaml lines 164-167

## Manual Output Verification

Run the following to prove the structs work correctly:

```bash
# Create a test file to verify outputs
cat > /tmp/verify_lifecycle.rs << 'EOF'
use context_graph_utl::config::{LifecycleConfig, StageConfig};

fn main() {
    // Verify StageConfig values
    let infancy = StageConfig::infancy();
    println!("Infancy: λ_novelty={}, λ_consolidation={}, ΔS_trig={}, ΔC_trig={}",
        infancy.lambda_novelty, infancy.lambda_consolidation,
        infancy.surprise_trigger, infancy.coherence_trigger);

    let growth = StageConfig::growth();
    println!("Growth: λ_novelty={}, λ_consolidation={}, ΔS_trig={}, ΔC_trig={}",
        growth.lambda_novelty, growth.lambda_consolidation,
        growth.surprise_trigger, growth.coherence_trigger);

    let maturity = StageConfig::maturity();
    println!("Maturity: λ_novelty={}, λ_consolidation={}, ΔS_trig={}, ΔC_trig={}",
        maturity.lambda_novelty, maturity.lambda_consolidation,
        maturity.surprise_trigger, maturity.coherence_trigger);

    // Verify LifecycleConfig
    let config = LifecycleConfig::default();
    println!("Stages: {}", config.stages.len());
    println!("Auto transition: {}", config.auto_transition);
    println!("Hysteresis: {}", config.transition_hysteresis);
}
EOF
```

Expected output:
```
Infancy: λ_novelty=0.7, λ_consolidation=0.3, ΔS_trig=0.9, ΔC_trig=0.2
Growth: λ_novelty=0.5, λ_consolidation=0.5, ΔS_trig=0.7, ΔC_trig=0.4
Maturity: λ_novelty=0.3, λ_consolidation=0.7, ΔS_trig=0.6, ΔC_trig=0.5
Stages: 3
Auto transition: true
Hysteresis: 10
```

---

*Task Version: 2.0.0 (Updated to reflect actual implementation)*
*Verified: 2026-01-04*
*Git Reference: f521803*
*Status: COMPLETE - 369 tests passing*
