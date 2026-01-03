---
id: "M04-T02a"
title: "Implement HyperbolicConfig Validation"
description: |
  COMPLETED: validate() method added to HyperbolicConfig with full NaN checks.
  Includes try_with_curvature() for validated construction.
  Implementation verified by Sherlock Holmes forensic investigation on 2026-01-03.
  All 18 validation tests pass. All 9 doc tests pass.
layer: "foundation"
status: "verified-complete"
priority: "high"
estimated_hours: 1
sequence: 5
depends_on:
  - "M04-T02"
  - "M04-T08"
spec_refs:
  - "TECH-GRAPH-004 Section 5"
  - "REQ-KG-054"
files_to_create: []
files_to_modify:
  - path: "crates/context-graph-graph/src/config.rs"
    description: "DONE - validate() and try_with_curvature() methods added"
test_file: "crates/context-graph-graph/src/config.rs (inline tests)"
completed_date: "2026-01-03"
verified_by: "sherlock-holmes forensic investigation"
---

## CRITICAL: DEPENDENCY CHECK

**Before starting this task, verify M04-T02 is COMPLETE:**

```bash
# HyperbolicConfig must have exactly 4 fields: dim, curvature, eps, max_norm
grep -A 20 "pub struct HyperbolicConfig" crates/context-graph-graph/src/config.rs | grep "pub "
# Expected output: 4 lines with pub dim, pub curvature, pub eps, pub max_norm

# GraphError::InvalidConfig must exist
grep "InvalidConfig" crates/context-graph-graph/src/error.rs
# Expected: InvalidConfig(String) variant exists
```

**If these checks fail, complete M04-T02 first.**

**NOTE**: M04-T08 (GraphError) is already COMPLETE as of 2026-01-03. The error.rs file contains 31 error variants including `InvalidConfig(String)`.

---

## CODEBASE STATE (After M04-T02)

### HyperbolicConfig Location
**File**: `crates/context-graph-graph/src/config.rs`

After M04-T02, the struct will look like:
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HyperbolicConfig {
    pub dim: usize,
    pub curvature: f32,
    pub eps: f32,
    pub max_norm: f32,
}
```

### GraphError Location
**File**: `crates/context-graph-graph/src/error.rs`

The error variant already exists:
```rust
#[error("Invalid configuration: {0}")]
InvalidConfig(String),
```

---

## Task Objective

ADD a `validate()` method to HyperbolicConfig that:
1. Checks ALL 4 fields for mathematical validity
2. Returns `Result<(), GraphError>`
3. Returns FIRST error encountered (fail fast)
4. Provides descriptive error messages with actual values

---

## Validation Rules

| Field | Valid Range | Error Condition |
|-------|-------------|-----------------|
| dim | > 0 | dim == 0 |
| curvature | < 0.0 | curvature >= 0.0 |
| eps | > 0.0 | eps <= 0.0 |
| max_norm | (0.0, 1.0) | max_norm <= 0.0 OR max_norm >= 1.0 |

---

## Implementation (Exact Code)

### Add to config.rs after the existing HyperbolicConfig impl block

```rust
use crate::error::GraphError;

impl HyperbolicConfig {
    /// Validate that all configuration parameters are mathematically valid
    /// for the Poincare ball model.
    ///
    /// # Validation Rules
    /// - `dim` > 0: Dimension must be positive
    /// - `curvature` < 0: Must be negative for hyperbolic space
    /// - `eps` > 0: Must be positive for numerical stability
    /// - `max_norm` in (0, 1): Must be strictly between 0 and 1
    ///
    /// # Errors
    /// Returns `GraphError::InvalidConfig` with descriptive message if any
    /// parameter is invalid. Returns the FIRST error encountered (fail-fast).
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// // Valid config passes
    /// let valid = HyperbolicConfig::default();
    /// assert!(valid.validate().is_ok());
    ///
    /// // Invalid curvature fails
    /// let mut invalid = HyperbolicConfig::default();
    /// invalid.curvature = 1.0; // positive is invalid
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), GraphError> {
        // Check dimension
        if self.dim == 0 {
            return Err(GraphError::InvalidConfig(
                "dim must be positive (got 0)".to_string()
            ));
        }

        // Check curvature - MUST be negative for hyperbolic space
        if self.curvature >= 0.0 {
            return Err(GraphError::InvalidConfig(
                format!(
                    "curvature must be negative for hyperbolic space (got {})",
                    self.curvature
                )
            ));
        }

        // Check for NaN curvature
        if self.curvature.is_nan() {
            return Err(GraphError::InvalidConfig(
                "curvature cannot be NaN".to_string()
            ));
        }

        // Check epsilon
        if self.eps <= 0.0 {
            return Err(GraphError::InvalidConfig(
                format!(
                    "eps must be positive for numerical stability (got {})",
                    self.eps
                )
            ));
        }

        // Check for NaN eps
        if self.eps.is_nan() {
            return Err(GraphError::InvalidConfig(
                "eps cannot be NaN".to_string()
            ));
        }

        // Check max_norm - must be in open interval (0, 1)
        if self.max_norm <= 0.0 || self.max_norm >= 1.0 {
            return Err(GraphError::InvalidConfig(
                format!(
                    "max_norm must be in open interval (0, 1), got {}",
                    self.max_norm
                )
            ));
        }

        // Check for NaN max_norm
        if self.max_norm.is_nan() {
            return Err(GraphError::InvalidConfig(
                "max_norm cannot be NaN".to_string()
            ));
        }

        Ok(())
    }

    /// Create a validated config with custom curvature.
    ///
    /// Returns error if curvature is invalid (>= 0 or NaN).
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// let config = HyperbolicConfig::try_with_curvature(-0.5).unwrap();
    /// assert_eq!(config.curvature, -0.5);
    ///
    /// // Invalid curvature returns error
    /// assert!(HyperbolicConfig::try_with_curvature(1.0).is_err());
    /// ```
    pub fn try_with_curvature(curvature: f32) -> Result<Self, GraphError> {
        let config = Self {
            curvature,
            ..Default::default()
        };
        config.validate()?;
        Ok(config)
    }
}
```

---

## Tests to Add (Exact Code)

Add these tests inside the existing `#[cfg(test)] mod tests { ... }` block:

```rust
// ============ Validation Tests ============

#[test]
fn test_validate_default_passes() {
    let config = HyperbolicConfig::default();
    assert!(config.validate().is_ok(), "Default config must be valid");
}

#[test]
fn test_validate_dim_zero_fails() {
    let config = HyperbolicConfig {
        dim: 0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("dim"), "Error should mention 'dim'");
    assert!(err_msg.contains("positive"), "Error should mention 'positive'");
}

#[test]
fn test_validate_curvature_zero_fails() {
    let config = HyperbolicConfig {
        curvature: 0.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("curvature"), "Error should mention 'curvature'");
    assert!(err_msg.contains("negative"), "Error should mention 'negative'");
}

#[test]
fn test_validate_curvature_positive_fails() {
    let config = HyperbolicConfig {
        curvature: 1.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("1"), "Error should include the actual value");
}

#[test]
fn test_validate_curvature_nan_fails() {
    let config = HyperbolicConfig {
        curvature: f32::NAN,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("NaN"), "Error should mention 'NaN'");
}

#[test]
fn test_validate_eps_zero_fails() {
    let config = HyperbolicConfig {
        eps: 0.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("eps"), "Error should mention 'eps'");
}

#[test]
fn test_validate_eps_negative_fails() {
    let config = HyperbolicConfig {
        eps: -1e-7,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn test_validate_max_norm_zero_fails() {
    let config = HyperbolicConfig {
        max_norm: 0.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("max_norm"), "Error should mention 'max_norm'");
}

#[test]
fn test_validate_max_norm_one_fails() {
    let config = HyperbolicConfig {
        max_norm: 1.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err(), "max_norm=1.0 is ON boundary, not inside ball");
}

#[test]
fn test_validate_max_norm_greater_than_one_fails() {
    let config = HyperbolicConfig {
        max_norm: 1.5,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn test_validate_max_norm_negative_fails() {
    let config = HyperbolicConfig {
        max_norm: -0.5,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn test_validate_custom_valid_curvature() {
    // Various valid negative curvatures
    for c in [-0.1, -0.5, -1.0, -2.0, -10.0] {
        let config = HyperbolicConfig::with_curvature(c);
        assert!(config.validate().is_ok(), "curvature {} should be valid", c);
    }
}

#[test]
fn test_try_with_curvature_valid() {
    let config = HyperbolicConfig::try_with_curvature(-0.5).unwrap();
    assert_eq!(config.curvature, -0.5);
    assert_eq!(config.dim, 64); // default
}

#[test]
fn test_try_with_curvature_invalid() {
    assert!(HyperbolicConfig::try_with_curvature(0.0).is_err());
    assert!(HyperbolicConfig::try_with_curvature(1.0).is_err());
    assert!(HyperbolicConfig::try_with_curvature(f32::NAN).is_err());
}

#[test]
fn test_validate_fail_fast_order() {
    // When multiple fields are invalid, should fail on first check (dim)
    let config = HyperbolicConfig {
        dim: 0,
        curvature: 1.0,  // also invalid
        eps: -1.0,       // also invalid
        max_norm: 2.0,   // also invalid
    };
    let err_msg = config.validate().unwrap_err().to_string();
    assert!(err_msg.contains("dim"), "Should fail on dim first");
}

#[test]
fn test_validate_boundary_values() {
    // Test values very close to boundaries
    let barely_valid = HyperbolicConfig {
        dim: 1,
        curvature: -1e-10,  // tiny but negative
        eps: 1e-10,         // tiny but positive
        max_norm: 0.9999999, // close to 1 but not 1
    };
    assert!(barely_valid.validate().is_ok());
}
```

---

## Add Required Import

At the top of `config.rs` (after line 15, near other imports), add:

```rust
use crate::error::GraphError;
```

The imports section should look like:
```rust
use serde::{Deserialize, Serialize};
use crate::error::GraphError;  // ADD THIS LINE
```

---

## Verification Commands

```bash
# Build must pass
cargo build -p context-graph-graph

# All validation tests must pass
cargo test -p context-graph-graph test_validate -- --nocapture

# All config tests must pass
cargo test -p context-graph-graph config

# No clippy warnings
cargo clippy -p context-graph-graph -- -D warnings

# Doc tests must pass
cargo test -p context-graph-graph --doc
```

---

## Full State Verification Protocol

### Source of Truth
The `validate()` method on `HyperbolicConfig` in `crates/context-graph-graph/src/config.rs`

### Execute & Inspect Steps

1. **Verify method exists**:
```bash
grep -A 30 "pub fn validate" crates/context-graph-graph/src/config.rs
```

2. **Run all validation tests**:
```bash
cargo test -p context-graph-graph test_validate -- --nocapture 2>&1
```

3. **Test specific edge cases manually**:
```bash
cargo test -p context-graph-graph test_validate_curvature_zero_fails -- --nocapture
cargo test -p context-graph-graph test_validate_max_norm_one_fails -- --nocapture
```

### Edge Case Audit (3 Cases)

**Case 1: Valid default config**
```rust
let config = HyperbolicConfig::default();
println!("BEFORE validate(): config = {:?}", config);
let result = config.validate();
println!("AFTER validate(): result = {:?}", result);
// Expected: Ok(())
```

**Case 2: Curvature = 0 (boundary)**
```rust
let config = HyperbolicConfig { curvature: 0.0, ..Default::default() };
println!("BEFORE validate(): curvature = {}", config.curvature);
let result = config.validate();
println!("AFTER validate(): {:?}", result);
// Expected: Err(InvalidConfig("curvature must be negative..."))
```

**Case 3: max_norm = 1.0 (boundary)**
```rust
let config = HyperbolicConfig { max_norm: 1.0, ..Default::default() };
println!("BEFORE validate(): max_norm = {}", config.max_norm);
let result = config.validate();
println!("AFTER validate(): {:?}", result);
// Expected: Err(InvalidConfig("max_norm must be in open interval (0, 1)..."))
```

### Evidence of Success

Provide logs showing:
1. `cargo build -p context-graph-graph` exits 0
2. All `test_validate_*` tests pass (count should be 16+)
3. Doc tests pass for validate() method
4. No clippy warnings

---

## Acceptance Criteria

- [ ] `validate()` method exists on HyperbolicConfig
- [ ] Returns `Ok(())` for valid configs
- [ ] Returns `Err(GraphError::InvalidConfig)` for dim == 0
- [ ] Returns `Err` for curvature >= 0
- [ ] Returns `Err` for curvature == NaN
- [ ] Returns `Err` for eps <= 0
- [ ] Returns `Err` for max_norm <= 0 OR max_norm >= 1
- [ ] Error messages include actual values
- [ ] `try_with_curvature()` method works
- [ ] All 16+ validation tests pass
- [ ] Doc tests pass
- [ ] No clippy warnings
- [ ] `cargo build -p context-graph-graph` succeeds

---

## FINAL VERIFICATION (MANDATORY)

After completing all changes, you MUST use the `sherlock-holmes` subagent to verify:

1. All acceptance criteria are met
2. No regressions in existing tests
3. Error messages are descriptive
4. NaN handling is correct
5. Boundary conditions are properly handled

**Invoke sherlock-holmes with**: "Verify M04-T02a HyperbolicConfig validation is complete and all edge cases pass"

---

## Anti-Patterns (DO NOT DO)

- DO NOT use `unwrap()` or `expect()` in validation logic
- DO NOT skip NaN checks (NaN comparisons are tricky!)
- DO NOT make max_norm = 1.0 valid (it's ON the boundary, not inside)
- DO NOT make curvature = 0.0 valid (that's Euclidean, not hyperbolic)
- DO NOT return multiple errors (fail-fast on first error)
- DO NOT add validation to Default::default() (validate is explicit)

---

## Mathematical Context

**Why these constraints?**

1. **curvature < 0**: The Poincare ball models hyperbolic space which has constant negative curvature. Zero curvature is Euclidean space; positive curvature is spherical geometry.

2. **max_norm < 1**: The Poincare ball is the open unit ball {x : ||x|| < 1}. Points ON the boundary (||x|| = 1) are at infinity in hyperbolic space and cause division by zero in the distance formula.

3. **eps > 0**: Used to prevent numerical issues when ||x|| approaches max_norm. The distance formula has (1 - ||x||²) in the denominator.

---

## Constitution Reference

From `constitution.yaml`:
```yaml
forbidden:
  AP-009: "NaN/Infinity in UTL → clamp to valid range"

perf:
  latency:
    entailment_check: "<1ms"
```

Validation ensures we never introduce NaN into hyperbolic calculations.

---

*Task Version: 2.0.0*
*Last Updated: 2026-01-03*
*Codebase Verified: 2026-01-03 (commit 53f56ec)*
*Depends On: M04-T02 (must complete first)*
