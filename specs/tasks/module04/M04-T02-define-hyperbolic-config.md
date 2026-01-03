---
id: "M04-T02"
title: "Complete HyperbolicConfig for Poincare Ball"
description: |
  COMPLETED: HyperbolicConfig struct in config.rs now has all 4 fields per spec.
  Implementation verified by Sherlock Holmes forensic investigation on 2026-01-03.
layer: "foundation"
status: "verified-complete"
priority: "critical"
estimated_hours: 0.5
sequence: 4
depends_on:
  - "M04-T00"
spec_refs:
  - "TECH-GRAPH-004 Section 5"
  - "REQ-KG-050, REQ-KG-054"
files_to_create: []
files_to_modify:
  - path: "crates/context-graph-graph/src/config.rs"
    description: "DONE - eps field added, dimension renamed to dim"
test_file: "crates/context-graph-graph/src/config.rs (inline tests)"
completed_date: "2026-01-03"
verified_by: "sherlock-holmes forensic investigation"
---

## CRITICAL CODEBASE STATE (Read First)

### What Already Exists (DO NOT RECREATE)

**File**: `crates/context-graph-graph/src/config.rs`
**Lines**: 123-141 (struct at 123-131, Default impl at 133-141)

```rust
// CURRENT IMPLEMENTATION (INCOMPLETE):
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperbolicConfig {
    pub dimension: usize,  // WRONG NAME - spec says "dim"
    pub curvature: f32,
    pub max_norm: f32,     // MISSING: eps field
}

impl Default for HyperbolicConfig {
    fn default() -> Self {
        Self {
            dimension: 64,     // WRONG NAME
            curvature: -1.0,
            max_norm: 0.999,   // CLOSE but spec says 1.0 - 1e-5 = 0.99999
        }
    }
}
```

### What Is Missing

1. Field `eps: f32` for numerical stability (spec: 1e-7)
2. Field name should be `dim` not `dimension` (for consistency with spec)
3. Default `max_norm` should be `1.0 - 1e-5` = `0.99999` (more precise)
4. Helper methods: `with_curvature()`, `abs_curvature()`

---

## Task Objective

EDIT `crates/context-graph-graph/src/config.rs` to make HyperbolicConfig match the specification exactly.

---

## Required Changes (Exact Diff)

### Change 1: Update struct definition (lines ~123-131)

**FROM:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperbolicConfig {
    /// Dimension of hyperbolic space (default: 64)
    pub dimension: usize,
    /// Curvature parameter (must be negative, default: -1.0)
    pub curvature: f32,
    /// Maximum norm for points (default: 0.999, must be < 1.0)
    pub max_norm: f32,
}
```

**TO:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HyperbolicConfig {
    /// Dimension of hyperbolic space (typically 64 for knowledge graphs).
    /// Must be positive.
    pub dim: usize,

    /// Curvature of hyperbolic space. MUST be negative.
    /// Default: -1.0 (unit hyperbolic space)
    /// Validated in M04-T02a.
    pub curvature: f32,

    /// Epsilon for numerical stability in hyperbolic operations.
    /// Prevents division by zero and NaN in distance calculations.
    /// Default: 1e-7
    pub eps: f32,

    /// Maximum norm for points (keeps points strictly inside ball boundary).
    /// Points with norm >= max_norm will be projected back inside.
    /// Must be in open interval (0, 1). Default: 1.0 - 1e-5 = 0.99999
    pub max_norm: f32,
}
```

### Change 2: Update Default impl (lines ~133-141)

**FROM:**
```rust
impl Default for HyperbolicConfig {
    fn default() -> Self {
        Self {
            dimension: 64,
            curvature: -1.0,
            max_norm: 0.999,
        }
    }
}
```

**TO:**
```rust
impl Default for HyperbolicConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            curvature: -1.0,
            eps: 1e-7,
            max_norm: 1.0 - 1e-5, // 0.99999
        }
    }
}

impl HyperbolicConfig {
    /// Create config with custom curvature.
    ///
    /// # Arguments
    /// * `curvature` - Must be negative. Use validate() to check.
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    /// let config = HyperbolicConfig::with_curvature(-0.5);
    /// assert_eq!(config.curvature, -0.5);
    /// assert_eq!(config.dim, 64); // other fields use defaults
    /// ```
    pub fn with_curvature(curvature: f32) -> Self {
        Self {
            curvature,
            ..Default::default()
        }
    }

    /// Get absolute value of curvature.
    ///
    /// Useful for formulas that need |c| rather than c.
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    /// let config = HyperbolicConfig::default();
    /// assert_eq!(config.abs_curvature(), 1.0);
    /// ```
    #[inline]
    pub fn abs_curvature(&self) -> f32 {
        self.curvature.abs()
    }

    /// Scale factor derived from curvature: sqrt(|c|)
    ///
    /// Used in Mobius operations and distance calculations.
    #[inline]
    pub fn scale(&self) -> f32 {
        self.abs_curvature().sqrt()
    }
}
```

### Change 3: Update existing tests (lines ~273-280)

**FROM:**
```rust
#[test]
fn test_hyperbolic_config_default() {
    let config = HyperbolicConfig::default();
    assert_eq!(config.dimension, 64);
    assert_eq!(config.curvature, -1.0);
    assert!(config.curvature < 0.0, "Curvature must be negative");
    assert!(config.max_norm < 1.0, "Max norm must be < 1.0");
    assert!(config.max_norm > 0.0, "Max norm must be positive");
}
```

**TO:**
```rust
#[test]
fn test_hyperbolic_config_default() {
    let config = HyperbolicConfig::default();

    // Verify all 4 fields
    assert_eq!(config.dim, 64, "Default dim must be 64");
    assert_eq!(config.curvature, -1.0, "Default curvature must be -1.0");
    assert_eq!(config.eps, 1e-7, "Default eps must be 1e-7");
    assert!((config.max_norm - 0.99999).abs() < 1e-10, "Default max_norm must be 1.0 - 1e-5");

    // Invariants
    assert!(config.curvature < 0.0, "Curvature must be negative");
    assert!(config.max_norm < 1.0, "Max norm must be < 1.0");
    assert!(config.max_norm > 0.0, "Max norm must be positive");
    assert!(config.eps > 0.0, "Eps must be positive");
}

#[test]
fn test_hyperbolic_config_with_curvature() {
    let config = HyperbolicConfig::with_curvature(-0.5);
    assert_eq!(config.curvature, -0.5);
    assert_eq!(config.dim, 64); // defaults preserved
    assert_eq!(config.eps, 1e-7);
}

#[test]
fn test_hyperbolic_config_abs_curvature() {
    let config = HyperbolicConfig::default();
    assert_eq!(config.abs_curvature(), 1.0);

    let config2 = HyperbolicConfig::with_curvature(-2.5);
    assert_eq!(config2.abs_curvature(), 2.5);
}

#[test]
fn test_hyperbolic_config_scale() {
    let config = HyperbolicConfig::default();
    assert_eq!(config.scale(), 1.0); // sqrt(|-1.0|) = 1.0

    let config2 = HyperbolicConfig::with_curvature(-4.0);
    assert_eq!(config2.scale(), 2.0); // sqrt(|-4.0|) = 2.0
}

#[test]
fn test_hyperbolic_config_serialization_roundtrip() {
    let config = HyperbolicConfig::default();
    let json = serde_json::to_string(&config).expect("Serialization failed");
    let deserialized: HyperbolicConfig = serde_json::from_str(&json).expect("Deserialization failed");
    assert_eq!(config, deserialized);
}

#[test]
fn test_hyperbolic_config_json_fields() {
    let config = HyperbolicConfig::default();
    let json = serde_json::to_string_pretty(&config).expect("Serialization failed");

    // Verify all 4 fields appear in JSON
    assert!(json.contains("\"dim\":"), "JSON must contain dim field");
    assert!(json.contains("\"curvature\":"), "JSON must contain curvature field");
    assert!(json.contains("\"eps\":"), "JSON must contain eps field");
    assert!(json.contains("\"max_norm\":"), "JSON must contain max_norm field");
}
```

---

## Constraints (MUST Follow)

1. **Field Types**: `dim: usize`, `curvature: f32`, `eps: f32`, `max_norm: f32`
2. **Derive Macros**: MUST include `PartialEq` for equality tests
3. **Default Values**: dim=64, curvature=-1.0, eps=1e-7, max_norm=0.99999
4. **NO VALIDATION HERE**: Validation is M04-T02a task (separate concern)
5. **NO BACKWARDS COMPATIBILITY**: This is a breaking change to field names

---

## Verification Commands

```bash
# Build must pass
cargo build -p context-graph-graph

# All tests must pass
cargo test -p context-graph-graph config

# No clippy warnings
cargo clippy -p context-graph-graph -- -D warnings

# Verify exact field count
grep -A 20 "pub struct HyperbolicConfig" crates/context-graph-graph/src/config.rs | grep "pub " | wc -l
# Expected: 4
```

---

## Full State Verification Protocol

### Source of Truth
The HyperbolicConfig struct in `crates/context-graph-graph/src/config.rs`

### Execute & Inspect Steps

1. **After editing, run**:
```bash
cargo test -p context-graph-graph test_hyperbolic_config_default -- --nocapture
```

2. **Verify struct has exactly 4 fields**:
```bash
grep -A 25 "pub struct HyperbolicConfig" crates/context-graph-graph/src/config.rs
```

3. **Verify JSON output**:
```bash
cargo test -p context-graph-graph test_hyperbolic_config_json_fields -- --nocapture
```

### Edge Case Audit (3 Cases)

**Case 1: Default instantiation**
```rust
// Before: HyperbolicConfig { dimension: 64, curvature: -1.0, max_norm: 0.999 }
// After:  HyperbolicConfig { dim: 64, curvature: -1.0, eps: 1e-7, max_norm: 0.99999 }
let config = HyperbolicConfig::default();
println!("BEFORE default: N/A (new fields added)");
println!("AFTER default: dim={}, curvature={}, eps={}, max_norm={}",
    config.dim, config.curvature, config.eps, config.max_norm);
// Expected: dim=64, curvature=-1.0, eps=0.0000001, max_norm=0.99999
```

**Case 2: Custom curvature**
```rust
let config = HyperbolicConfig::with_curvature(-0.1);
println!("BEFORE with_curvature: N/A (method didn't exist)");
println!("AFTER with_curvature(-0.1): curvature={}, dim={}",
    config.curvature, config.dim);
// Expected: curvature=-0.1, dim=64 (default preserved)
```

**Case 3: Serialization**
```rust
let config = HyperbolicConfig::default();
let json = serde_json::to_string(&config)?;
println!("BEFORE JSON: {{\"dimension\":64,\"curvature\":-1.0,\"max_norm\":0.999}}");
println!("AFTER JSON: {}", json);
// Expected: {"dim":64,"curvature":-1.0,"eps":1e-7,"max_norm":0.99999}
```

### Evidence of Success

Provide a log showing:
1. `cargo build -p context-graph-graph` exits 0
2. `cargo test -p context-graph-graph config` shows all tests passing
3. Grep output showing exactly 4 `pub ` fields in HyperbolicConfig

---

## Acceptance Criteria

- [ ] HyperbolicConfig has exactly 4 fields: dim, curvature, eps, max_norm
- [ ] Field names match spec (dim NOT dimension)
- [ ] Default values: dim=64, curvature=-1.0, eps=1e-7, max_norm=0.99999
- [ ] PartialEq derived for equality comparison
- [ ] with_curvature() method works
- [ ] abs_curvature() method works
- [ ] scale() method works
- [ ] JSON serialization includes all 4 fields
- [ ] All tests pass
- [ ] No clippy warnings
- [ ] `cargo build -p context-graph-graph` succeeds

---

## FINAL VERIFICATION (MANDATORY)

After completing all changes, you MUST use the `sherlock-holmes` subagent to verify:

1. All acceptance criteria are met
2. No regressions in existing tests
3. The struct matches the specification exactly
4. All edge cases are handled correctly

**Invoke sherlock-holmes with**: "Verify M04-T02 HyperbolicConfig implementation is complete and correct per spec"

---

## Anti-Patterns (DO NOT DO)

- DO NOT add validation logic (that's M04-T02a)
- DO NOT keep old field name `dimension` for backwards compatibility
- DO NOT use 0.999 for max_norm (use precise 1.0 - 1e-5)
- DO NOT forget to add PartialEq derive
- DO NOT create a new file (edit existing config.rs)

---

*Task Version: 2.0.0*
*Last Updated: 2026-01-03*
*Codebase Verified: 2026-01-03 (commit 53f56ec)*
