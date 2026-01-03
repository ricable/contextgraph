---
id: "M04-T03"
title: "Complete ConeConfig for Entailment Cones"
description: |
  MODIFY existing ConeConfig struct in config.rs to match specification.
  Current implementation has 3 fields with WRONG defaults. Spec requires 5 fields.
  This is a MODIFICATION task - the struct EXISTS but is INCOMPLETE.
layer: "foundation"
status: "pending"
priority: "high"
estimated_hours: 1.5
sequence: 6
depends_on:
  - "M04-T00"
spec_refs:
  - "TECH-GRAPH-004 Section 6"
  - "REQ-KG-052"
files_to_create: []
files_to_modify:
  - path: "crates/context-graph-graph/src/config.rs"
    description: "Complete ConeConfig struct with 5 fields and compute_aperture method"
test_file: "crates/context-graph-graph/src/config.rs (inline tests)"
---

## CRITICAL CODEBASE STATE (Read First)

### What Already Exists (DO NOT RECREATE)

**File**: `crates/context-graph-graph/src/config.rs`
**Lines**: 332-359 (struct at 341-349, Default impl at 351-358)

```rust
// CURRENT IMPLEMENTATION (INCOMPLETE - AS OF COMMIT 7306023):
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConeConfig {
    /// Base aperture angle in radians (default: PI/4 = 45 degrees)
    pub base_aperture: f32,      // WRONG DEFAULT - spec says 1.0 rad (~57°)
    /// Aperture decay factor per depth level (default: 0.9)
    pub aperture_decay: f32,     // WRONG DEFAULT - spec says 0.85
    /// Minimum aperture angle (default: 0.1 radians)
    pub min_aperture: f32,       // CORRECT
}
// MISSING: max_aperture (1.5 rad)
// MISSING: membership_threshold (0.7)
// MISSING: validate() method
// MISSING: compute_aperture(depth) method

impl Default for ConeConfig {
    fn default() -> Self {
        Self {
            base_aperture: std::f32::consts::FRAC_PI_4,  // WRONG! Spec says 1.0
            aperture_decay: 0.9,                         // WRONG! Spec says 0.85
            min_aperture: 0.1,                           // CORRECT
        }
    }
}
```

### What Is Missing (Per Spec REQ-KG-052 & TECH-GRAPH-004 Section 6)

1. Field `max_aperture: f32` - Maximum aperture (1.5 rad = ~85.9°)
2. Field `membership_threshold: f32` - Soft membership score threshold (0.7)
3. Method `compute_aperture(depth: u32) -> f32` - Formula: `base * decay^depth` clamped to [min, max]
4. Method `validate() -> Result<(), GraphError>` - Parameter validation
5. Fix `base_aperture` default: 1.0 rad (NOT PI/4)
6. Fix `aperture_decay` default: 0.85 (NOT 0.9)
7. Add `PartialEq` derive for tests

### Crate Structure Context

The `context-graph-graph` crate exists at `crates/context-graph-graph/`:
- `src/lib.rs` - Re-exports config types
- `src/config.rs` - Contains IndexConfig, HyperbolicConfig, ConeConfig
- `src/error.rs` - GraphError enum with InvalidConfig variant

**Import already exists**: `use crate::error::GraphError;` at line 17

---

## Task Objective

EDIT `crates/context-graph-graph/src/config.rs` to:
1. Add 2 missing fields to ConeConfig
2. Fix 2 incorrect default values
3. Add compute_aperture(depth) method
4. Add validate() method
5. Add comprehensive tests

---

## Exact Signatures (MUST Match)

```rust
use serde::{Deserialize, Serialize};

/// Configuration for entailment cones in hyperbolic space.
///
/// Entailment cones enable O(1) IS-A hierarchy queries. A concept's cone
/// contains all concepts it subsumes. Aperture narrows with depth,
/// creating increasingly specific cones for child concepts.
///
/// # Mathematics
///
/// - Aperture at depth d: `aperture(d) = base_aperture * decay^d`
/// - Result clamped to `[min_aperture, max_aperture]`
/// - Cone A contains point P iff angle(P - apex, axis) <= aperture
///
/// # Constitution Reference
///
/// - perf.latency.entailment_check: <1ms
/// - Section 9 "HYPERBOLIC ENTAILMENT CONES" in contextprd.md
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConeConfig {
    /// Minimum cone aperture in radians.
    /// Prevents cones from becoming too narrow at deep levels.
    /// Default: 0.1 rad (~5.7 degrees)
    pub min_aperture: f32,

    /// Maximum cone aperture in radians.
    /// Prevents cones from becoming too wide at root level.
    /// Default: 1.5 rad (~85.9 degrees)
    pub max_aperture: f32,

    /// Base aperture for depth 0 nodes (root concepts).
    /// This is the starting aperture before decay is applied.
    /// Default: 1.0 rad (~57.3 degrees)
    pub base_aperture: f32,

    /// Decay factor applied per hierarchy level.
    /// Must be in open interval (0, 1).
    /// Default: 0.85 (15% narrower per level)
    pub aperture_decay: f32,

    /// Threshold for soft membership scoring.
    /// Points with membership score >= threshold are considered contained.
    /// Must be in open interval (0, 1).
    /// Default: 0.7
    pub membership_threshold: f32,
}

impl Default for ConeConfig {
    fn default() -> Self {
        Self {
            min_aperture: 0.1,      // ~5.7 degrees
            max_aperture: 1.5,      // ~85.9 degrees
            base_aperture: 1.0,     // ~57.3 degrees
            aperture_decay: 0.85,   // 15% narrower per level
            membership_threshold: 0.7,
        }
    }
}

impl ConeConfig {
    /// Compute aperture for a node at given depth.
    ///
    /// # Formula
    /// `aperture = base_aperture * aperture_decay^depth`
    /// Result is clamped to `[min_aperture, max_aperture]`.
    ///
    /// # Arguments
    /// * `depth` - Depth in hierarchy (0 = root)
    ///
    /// # Returns
    /// Aperture in radians, clamped to valid range.
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::ConeConfig;
    ///
    /// let config = ConeConfig::default();
    /// assert_eq!(config.compute_aperture(0), 1.0);  // base at root
    /// assert!((config.compute_aperture(1) - 0.85).abs() < 1e-6);  // 1.0 * 0.85
    /// assert_eq!(config.compute_aperture(100), 0.1);  // clamped to min
    /// ```
    pub fn compute_aperture(&self, depth: u32) -> f32 {
        let raw = self.base_aperture * self.aperture_decay.powi(depth as i32);
        raw.clamp(self.min_aperture, self.max_aperture)
    }

    /// Validate configuration parameters.
    ///
    /// # Validation Rules
    /// - `min_aperture` > 0: Must be positive
    /// - `max_aperture` > `min_aperture`: Max must exceed min
    /// - `base_aperture` in [`min_aperture`, `max_aperture`]: Base must be in valid range
    /// - `aperture_decay` in (0, 1): Must be strictly between 0 and 1
    /// - `membership_threshold` in (0, 1): Must be strictly between 0 and 1
    ///
    /// # Errors
    /// Returns `GraphError::InvalidConfig` with descriptive message if any
    /// parameter is invalid. Fails fast on first error.
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::ConeConfig;
    ///
    /// let valid = ConeConfig::default();
    /// assert!(valid.validate().is_ok());
    ///
    /// let mut invalid = ConeConfig::default();
    /// invalid.aperture_decay = 1.5;  // must be < 1
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), crate::error::GraphError> {
        // Check min_aperture is positive
        if self.min_aperture <= 0.0 {
            return Err(crate::error::GraphError::InvalidConfig(
                format!("min_aperture must be positive (got {})", self.min_aperture)
            ));
        }

        // Check for NaN in min_aperture
        if self.min_aperture.is_nan() {
            return Err(crate::error::GraphError::InvalidConfig(
                "min_aperture cannot be NaN".to_string()
            ));
        }

        // Check max_aperture > min_aperture
        if self.max_aperture <= self.min_aperture {
            return Err(crate::error::GraphError::InvalidConfig(
                format!(
                    "max_aperture ({}) must be greater than min_aperture ({})",
                    self.max_aperture, self.min_aperture
                )
            ));
        }

        // Check for NaN in max_aperture
        if self.max_aperture.is_nan() {
            return Err(crate::error::GraphError::InvalidConfig(
                "max_aperture cannot be NaN".to_string()
            ));
        }

        // Check base_aperture is in valid range
        if self.base_aperture < self.min_aperture || self.base_aperture > self.max_aperture {
            return Err(crate::error::GraphError::InvalidConfig(
                format!(
                    "base_aperture ({}) must be in range [{}, {}]",
                    self.base_aperture, self.min_aperture, self.max_aperture
                )
            ));
        }

        // Check for NaN in base_aperture
        if self.base_aperture.is_nan() {
            return Err(crate::error::GraphError::InvalidConfig(
                "base_aperture cannot be NaN".to_string()
            ));
        }

        // Check aperture_decay in (0, 1)
        if self.aperture_decay <= 0.0 || self.aperture_decay >= 1.0 {
            return Err(crate::error::GraphError::InvalidConfig(
                format!(
                    "aperture_decay must be in open interval (0, 1), got {}",
                    self.aperture_decay
                )
            ));
        }

        // Check for NaN in aperture_decay
        if self.aperture_decay.is_nan() {
            return Err(crate::error::GraphError::InvalidConfig(
                "aperture_decay cannot be NaN".to_string()
            ));
        }

        // Check membership_threshold in (0, 1)
        if self.membership_threshold <= 0.0 || self.membership_threshold >= 1.0 {
            return Err(crate::error::GraphError::InvalidConfig(
                format!(
                    "membership_threshold must be in open interval (0, 1), got {}",
                    self.membership_threshold
                )
            ));
        }

        // Check for NaN in membership_threshold
        if self.membership_threshold.is_nan() {
            return Err(crate::error::GraphError::InvalidConfig(
                "membership_threshold cannot be NaN".to_string()
            ));
        }

        Ok(())
    }
}
```

---

## Constraints (MUST Follow)

1. **Field Types**: All 5 fields are `f32`
2. **Field Order**: min_aperture, max_aperture, base_aperture, aperture_decay, membership_threshold
3. **Derive Macros**: MUST include `PartialEq` for equality tests
4. **Default Values** (EXACT):
   - min_aperture: 0.1
   - max_aperture: 1.5
   - base_aperture: 1.0
   - aperture_decay: 0.85
   - membership_threshold: 0.7
5. **compute_aperture** uses `powi()` not `powf()` for integer exponent
6. **NO BACKWARDS COMPATIBILITY**: Delete old fields with wrong defaults
7. **Import**: `use crate::error::GraphError;` already exists at line 17 - DO NOT add again

---

## Implementation Steps

1. **Locate ConeConfig** at lines 332-359 in `config.rs`
2. **Replace struct** with new 5-field definition including PartialEq
3. **Replace Default impl** with correct values
4. **Add impl ConeConfig block** with compute_aperture and validate methods
5. **Replace existing tests** with comprehensive new tests
6. **Run verification commands**

---

## Required Test Cases

Add these tests to the `#[cfg(test)] mod tests` section:

```rust
// ============ ConeConfig Tests ============

#[test]
fn test_cone_config_default_values() {
    let config = ConeConfig::default();

    // Verify all 5 fields match spec
    assert_eq!(config.min_aperture, 0.1, "min_aperture must be 0.1");
    assert_eq!(config.max_aperture, 1.5, "max_aperture must be 1.5");
    assert_eq!(config.base_aperture, 1.0, "base_aperture must be 1.0");
    assert_eq!(config.aperture_decay, 0.85, "aperture_decay must be 0.85");
    assert_eq!(config.membership_threshold, 0.7, "membership_threshold must be 0.7");
}

#[test]
fn test_cone_config_field_constraints() {
    let config = ConeConfig::default();

    // Verify logical relationships
    assert!(config.min_aperture > 0.0, "min_aperture must be positive");
    assert!(config.max_aperture > config.min_aperture, "max > min");
    assert!(config.base_aperture >= config.min_aperture, "base >= min");
    assert!(config.base_aperture <= config.max_aperture, "base <= max");
    assert!(config.aperture_decay > 0.0 && config.aperture_decay < 1.0, "decay in (0,1)");
    assert!(config.membership_threshold > 0.0 && config.membership_threshold < 1.0, "threshold in (0,1)");
}

#[test]
fn test_compute_aperture_depth_zero() {
    let config = ConeConfig::default();
    // depth=0: base_aperture * 0.85^0 = 1.0 * 1 = 1.0
    assert_eq!(config.compute_aperture(0), 1.0);
}

#[test]
fn test_compute_aperture_depth_one() {
    let config = ConeConfig::default();
    // depth=1: 1.0 * 0.85^1 = 0.85
    let result = config.compute_aperture(1);
    assert!((result - 0.85).abs() < 1e-6, "Expected 0.85, got {}", result);
}

#[test]
fn test_compute_aperture_depth_two() {
    let config = ConeConfig::default();
    // depth=2: 1.0 * 0.85^2 = 0.7225
    let result = config.compute_aperture(2);
    assert!((result - 0.7225).abs() < 1e-6, "Expected 0.7225, got {}", result);
}

#[test]
fn test_compute_aperture_clamps_to_min() {
    let config = ConeConfig::default();
    // Very deep: should clamp to min_aperture = 0.1
    // 1.0 * 0.85^100 ≈ 3.6e-8, clamped to 0.1
    assert_eq!(config.compute_aperture(100), 0.1);
}

#[test]
fn test_compute_aperture_clamps_to_max() {
    // Config where base > max (shouldn't happen in practice but test clamping)
    let config = ConeConfig {
        min_aperture: 0.1,
        max_aperture: 0.5,
        base_aperture: 1.0,  // Exceeds max
        aperture_decay: 0.85,
        membership_threshold: 0.7,
    };
    // depth=0: raw=1.0, clamped to max=0.5
    assert_eq!(config.compute_aperture(0), 0.5);
}

#[test]
fn test_validate_default_passes() {
    let config = ConeConfig::default();
    assert!(config.validate().is_ok(), "Default config must be valid");
}

#[test]
fn test_validate_min_aperture_zero_fails() {
    let config = ConeConfig {
        min_aperture: 0.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("min_aperture"));
}

#[test]
fn test_validate_min_aperture_negative_fails() {
    let config = ConeConfig {
        min_aperture: -0.1,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_max_less_than_min_fails() {
    let config = ConeConfig {
        min_aperture: 1.0,
        max_aperture: 0.5,  // Less than min
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("max_aperture"));
}

#[test]
fn test_validate_max_equals_min_fails() {
    let config = ConeConfig {
        min_aperture: 0.5,
        max_aperture: 0.5,  // Equal, should fail (must be greater)
        base_aperture: 0.5,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_decay_zero_fails() {
    let config = ConeConfig {
        aperture_decay: 0.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("aperture_decay"));
}

#[test]
fn test_validate_decay_one_fails() {
    let config = ConeConfig {
        aperture_decay: 1.0,  // Boundary, excluded
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_decay_greater_than_one_fails() {
    let config = ConeConfig {
        aperture_decay: 1.5,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_threshold_zero_fails() {
    let config = ConeConfig {
        membership_threshold: 0.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("membership_threshold"));
}

#[test]
fn test_validate_threshold_one_fails() {
    let config = ConeConfig {
        membership_threshold: 1.0,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_nan_fields_fail() {
    // Test each field with NaN
    let configs = [
        ConeConfig { min_aperture: f32::NAN, ..Default::default() },
        ConeConfig { max_aperture: f32::NAN, ..Default::default() },
        ConeConfig { base_aperture: f32::NAN, ..Default::default() },
        ConeConfig { aperture_decay: f32::NAN, ..Default::default() },
        ConeConfig { membership_threshold: f32::NAN, ..Default::default() },
    ];

    for (i, config) in configs.iter().enumerate() {
        assert!(
            config.validate().is_err(),
            "Config {} with NaN should fail validation", i
        );
    }
}

#[test]
fn test_cone_config_serialization_roundtrip() {
    let config = ConeConfig::default();
    let json = serde_json::to_string(&config).expect("Serialization failed");
    let deserialized: ConeConfig = serde_json::from_str(&json).expect("Deserialization failed");
    assert_eq!(config, deserialized);
}

#[test]
fn test_cone_config_json_has_all_fields() {
    let config = ConeConfig::default();
    let json = serde_json::to_string_pretty(&config).expect("Serialization failed");

    // Verify all 5 fields appear in JSON
    assert!(json.contains("\"min_aperture\":"), "JSON must contain min_aperture");
    assert!(json.contains("\"max_aperture\":"), "JSON must contain max_aperture");
    assert!(json.contains("\"base_aperture\":"), "JSON must contain base_aperture");
    assert!(json.contains("\"aperture_decay\":"), "JSON must contain aperture_decay");
    assert!(json.contains("\"membership_threshold\":"), "JSON must contain membership_threshold");
}

#[test]
fn test_cone_config_equality() {
    let a = ConeConfig::default();
    let b = ConeConfig::default();
    assert_eq!(a, b, "Two default configs must be equal");

    let c = ConeConfig {
        min_aperture: 0.2,  // Different
        ..Default::default()
    };
    assert_ne!(a, c, "Different configs must not be equal");
}
```

---

## Verification Commands

```bash
# Build must pass
cargo build -p context-graph-graph

# All config tests must pass
cargo test -p context-graph-graph config -- --nocapture

# Specific ConeConfig tests
cargo test -p context-graph-graph cone_config -- --nocapture

# No clippy warnings
cargo clippy -p context-graph-graph -- -D warnings

# Verify exact field count (must show 5)
grep -A 30 "pub struct ConeConfig" crates/context-graph-graph/src/config.rs | grep "pub " | wc -l
# Expected output: 5
```

---

## Full State Verification Protocol

### Source of Truth
The ConeConfig struct in `crates/context-graph-graph/src/config.rs`

### Execute & Inspect Steps

1. **After editing, verify struct fields**:
```bash
grep -A 35 "pub struct ConeConfig" crates/context-graph-graph/src/config.rs
```

2. **Verify default values in test output**:
```bash
cargo test -p context-graph-graph test_cone_config_default_values -- --nocapture
```

3. **Verify compute_aperture behavior**:
```bash
cargo test -p context-graph-graph compute_aperture -- --nocapture
```

### Edge Case Audit (3 Cases)

**Case 1: depth = 0 (root concept)**
```rust
// State BEFORE: Not applicable (compute_aperture didn't exist)
// State AFTER:
let config = ConeConfig::default();
let aperture = config.compute_aperture(0);
println!("BEFORE: N/A (method didn't exist)");
println!("AFTER depth=0: aperture={}", aperture);
// Expected: aperture=1.0 (base_aperture, no decay applied)
```

**Case 2: depth = 100 (very deep, clamped to min)**
```rust
let config = ConeConfig::default();
let raw = 1.0 * 0.85_f32.powi(100);
let clamped = config.compute_aperture(100);
println!("BEFORE: N/A");
println!("AFTER depth=100: raw={}, clamped={}", raw, clamped);
// Expected: raw≈3.6e-8, clamped=0.1 (min_aperture)
```

**Case 3: Invalid aperture_decay = 1.5**
```rust
let config = ConeConfig {
    aperture_decay: 1.5,
    ..Default::default()
};
println!("BEFORE validate(): config created with decay=1.5");
let result = config.validate();
println!("AFTER validate(): result={:?}", result);
// Expected: Err(InvalidConfig("aperture_decay must be in open interval (0, 1)..."))
```

### Evidence of Success Log

Provide a log showing:
```
$ cargo build -p context-graph-graph
   Compiling context-graph-graph v0.1.0
    Finished dev [unoptimized + debuginfo] target(s)

$ cargo test -p context-graph-graph cone_config -- --nocapture
running 18 tests
test config::tests::test_cone_config_default_values ... ok
test config::tests::test_cone_config_field_constraints ... ok
test config::tests::test_compute_aperture_depth_zero ... ok
test config::tests::test_compute_aperture_depth_one ... ok
test config::tests::test_compute_aperture_depth_two ... ok
test config::tests::test_compute_aperture_clamps_to_min ... ok
test config::tests::test_compute_aperture_clamps_to_max ... ok
test config::tests::test_validate_default_passes ... ok
...
test result: ok. 18 passed; 0 failed

$ grep -A 30 "pub struct ConeConfig" crates/context-graph-graph/src/config.rs | grep "pub " | wc -l
5
```

---

## Acceptance Criteria

- [ ] ConeConfig has exactly 5 fields: min_aperture, max_aperture, base_aperture, aperture_decay, membership_threshold
- [ ] All fields are f32 type
- [ ] PartialEq is derived
- [ ] Default values match spec: 0.1, 1.5, 1.0, 0.85, 0.7
- [ ] compute_aperture(0) returns base_aperture (1.0)
- [ ] compute_aperture(1) returns 0.85 (1.0 * 0.85)
- [ ] compute_aperture(100) returns 0.1 (clamped to min)
- [ ] validate() returns Ok for default config
- [ ] validate() returns Err for aperture_decay >= 1.0
- [ ] validate() returns Err for aperture_decay <= 0.0
- [ ] validate() returns Err for membership_threshold outside (0, 1)
- [ ] validate() returns Err for any NaN value
- [ ] JSON serialization includes all 5 fields
- [ ] All tests pass
- [ ] No clippy warnings
- [ ] `cargo build -p context-graph-graph` succeeds

---

## Manual Output Verification (MANDATORY)

After completing implementation, you MUST manually verify these outputs exist:

1. **Run the build and capture output**:
```bash
cargo build -p context-graph-graph 2>&1 | tee /tmp/cone_config_build.log
```

2. **Run tests and capture output**:
```bash
cargo test -p context-graph-graph cone_config -- --nocapture 2>&1 | tee /tmp/cone_config_test.log
```

3. **Verify the struct has exactly 5 fields**:
```bash
grep -c "pub [a-z_]*:" crates/context-graph-graph/src/config.rs | grep -A30 "struct ConeConfig"
```

4. **Print sample output to prove compute_aperture works**:
```bash
cargo test -p context-graph-graph test_compute_aperture_depth_zero -- --nocapture
```

The test output should show `aperture = 1.0` for depth 0.

---

## FINAL VERIFICATION (MANDATORY)

After completing all changes, you MUST use the `sherlock-holmes` subagent to verify:

1. All 5 fields exist with correct types
2. All default values match spec exactly
3. compute_aperture() produces correct results at depths 0, 1, 2, 100
4. validate() correctly rejects invalid configurations
5. JSON serialization includes all 5 fields
6. No regressions in existing IndexConfig or HyperbolicConfig tests

**Invoke sherlock-holmes with**: "Verify M04-T03 ConeConfig implementation is complete and correct per spec REQ-KG-052. Check that struct has 5 fields, defaults are 0.1/1.5/1.0/0.85/0.7, compute_aperture works for depths 0/1/100, and validate rejects invalid configs."

---

## Anti-Patterns (DO NOT DO)

- DO NOT keep old field order (use spec order)
- DO NOT keep wrong defaults (PI/4, 0.9)
- DO NOT use powf() - use powi() for integer exponent
- DO NOT add a new GraphError variant - use existing InvalidConfig
- DO NOT forget PartialEq derive
- DO NOT add duplicate import for GraphError
- DO NOT skip validation of base_aperture range
- DO NOT create backwards compatibility shims

---

## Relationship to Other Tasks

- **Depends on**: M04-T00 (crate exists) - COMPLETE ✅
- **Required by**: M04-T06 (EntailmentCone uses ConeConfig for aperture calculation)
- **Required by**: M04-T07 (Containment logic uses compute_aperture)
- **Related to**: M04-T02, M04-T02a (similar config pattern with validation)

---

*Task Version: 2.0.0*
*Last Updated: 2026-01-03*
*Codebase Verified: 2026-01-03 (commit 7306023)*
*Prior Tasks Complete: M04-T00 ✅, M04-T01 ✅, M04-T01a ✅, M04-T02 ✅, M04-T02a ✅*
