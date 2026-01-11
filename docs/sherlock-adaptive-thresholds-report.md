# Sherlock Holmes Forensic Report: Adaptive Threshold Calibration System

**Case ID**: SHERLOCK-ATC-001
**Date**: 2026-01-10
**Investigator**: Sherlock Holmes (Forensic Code Detective)
**Subject**: Adaptive Threshold Calibration (ATC) Implementation

---

## Executive Summary

*"The game is afoot!"*

After exhaustive forensic examination of the ContextGraph codebase, I present my findings on the Adaptive Threshold Calibration (ATC) system implementation. The evidence reveals a **comprehensive 4-level ATC architecture** that closely aligns with PRD requirements, though some areas require attention.

**VERDICT: LARGELY INNOCENT with MINOR VIOLATIONS**

The codebase demonstrates sophisticated adaptive threshold mechanisms, but contains residual hardcoded values in peripheral modules that should be migrated to the ATC system.

---

## Evidence Log

### Source of Truth Verification

| Location | Status | Notes |
|----------|--------|-------|
| `/crates/context-graph-core/src/atc/` | VERIFIED | Complete 4-level ATC implementation |
| `/crates/context-graph-core/src/atc/level1_ewma.rs` | VERIFIED | EWMA drift tracking implemented |
| `/crates/context-graph-core/src/atc/level2_temperature.rs` | VERIFIED | Per-embedder temperature scaling |
| `/crates/context-graph-core/src/atc/level3_bandit.rs` | VERIFIED | Thompson Sampling + UCB |
| `/crates/context-graph-core/src/atc/level4_bayesian.rs` | VERIFIED | GP surrogate + Expected Improvement |
| `/crates/context-graph-core/src/atc/domain.rs` | VERIFIED | Domain-specific thresholds (6 domains) |
| `/crates/context-graph-core/src/atc/calibration.rs` | VERIFIED | ECE, MCE, Brier metrics |
| `/crates/context-graph-mcp/src/handlers/atc.rs` | VERIFIED | MCP tools implemented |
| `/crates/context-graph-core/src/autonomous/services/threshold_learner.rs` | VERIFIED | NORTH-009 ThresholdLearner |

---

## Finding 1: 4-Level ATC Architecture

**STATUS: IMPLEMENTED**

The constitution-mandated 4-level adaptive threshold calibration is fully implemented:

### Level 1: EWMA Drift Tracker (Per-Query)
**File**: `/crates/context-graph-core/src/atc/level1_ewma.rs`

```rust
// Formula: theta_ewma(t) = alpha * theta_observed(t) + (1 - alpha) * theta_ewma(t-1)
// Drift detection: |theta_ewma - theta_baseline| / sigma_baseline
// Triggers: >2sigma -> Level 2, >3sigma -> Level 3
```

**Evidence**:
- `EwmaState` struct with `ewma_value`, `baseline`, `baseline_std`, `alpha`
- `DriftTracker` managing multiple threshold types
- `get_level2_triggers()` and `get_level3_triggers()` methods
- Alpha clamped to [0.1, 0.3] range

### Level 2: Temperature Scaling (Hourly)
**File**: `/crates/context-graph-core/src/atc/level2_temperature.rs`

```rust
// Formula: calibrated_confidence = sigmoid(logit(raw_confidence) / T)
// Per-embedder temperatures:
// - E1_Semantic: T=1.0 (baseline)
// - E5_Causal: T=1.2 (overconfident)
// - E7_Code: T=0.9 (needs precision)
// - E9_HDC: T=1.5 (noisy)
// - E13_SPLADE: T=1.1 (sparse = variable)
```

**Evidence**:
- `TemperatureCalibration` per embedder
- `TemperatureScaler` managing all embedders
- Grid search calibration over temperature range
- `embedder_temperature_range()` function with per-embedder bounds
- `should_recalibrate()` checks hourly interval

### Level 3: Thompson Sampling Bandit (Session)
**File**: `/crates/context-graph-core/src/atc/level3_bandit.rs`

```rust
// Algorithms:
// 1. Thompson Sampling: sample from Beta(alpha, beta) per arm
// 2. UCB: theta = argmax[mu(theta) + c * sqrt(ln(N)/n(theta))]
// 3. Budgeted UCB: violation_budget(t) = B_0 * exp(-lambda * t)
```

**Evidence**:
- `ThresholdArm` and `ArmStats` structs
- `ThresholdBandit` with `select_thompson()` and `select_ucb()`
- `get_violation_budget()` with exponential decay
- Beta distribution sampling using `rand_distr::Beta`
- Comprehensive test for exploration behavior

### Level 4: Bayesian Meta-Optimizer (Weekly)
**File**: `/crates/context-graph-core/src/atc/level4_bayesian.rs`

```rust
// Algorithm:
// 1. Fit GP to (threshold, performance) observations
// 2. Maximize EI (Expected Improvement) to select next config
// 3. Evaluate system with new thresholds
// 4. Update GP with observation
```

**Evidence**:
- `GaussianProcessTracker` with RBF kernel
- `BayesianOptimizer` with `suggest_next()` using EI
- `ThresholdConstraints` enforcing monotonicity
- `should_optimize()` checking weekly interval
- Kernel function: `k(x,x') = sigma^2 * exp(-||x-x'||^2 / (2*l^2))`

---

## Finding 2: Threshold Categories

**STATUS: IMPLEMENTED**

The PRD-specified threshold categories are defined:

| Threshold | PRD Requirement | Implementation Location | Status |
|-----------|-----------------|------------------------|--------|
| theta_opt | Optimal alignment | `DomainThresholds.theta_opt` | PRESENT |
| theta_acc | Acceptable alignment | `DomainThresholds.theta_acc` | PRESENT |
| theta_warn | Warning alignment | `DomainThresholds.theta_warn` | PRESENT |
| theta_dup | Duplicate detection | `DomainThresholds.theta_dup` | PRESENT |
| theta_edge | Edge case handling | `DomainThresholds.theta_edge` | PRESENT |
| theta_joh | Johari classification | `UtlConfig` / `JohariConfig` | PRESENT |
| theta_kur | Kuramoto sync | `PhaseSyncConfig` | PRESENT |
| theta_ent_h | Entropy high | `SurpriseConfig` | PRESENT |
| theta_ent_l | Entropy low | `SurpriseConfig` | PRESENT |
| theta_gate | Gating threshold | `StageThresholds` | PRESENT |

**Constraint Enforcement**:
```rust
// ThresholdConstraints (level4_bayesian.rs lines 224-250):
theta_opt_range: (0.60, 0.90)
theta_acc_range: (0.55, 0.85)
theta_warn_range: (0.40, 0.70)
theta_dup_range: (0.80, 0.98)
theta_edge_range: (0.50, 0.85)
enforce_monotonicity: true  // theta_opt > theta_acc > theta_warn
```

---

## Finding 3: Per-Embedder Temperature Scaling

**STATUS: IMPLEMENTED**

**File**: `/crates/context-graph-core/src/atc/level2_temperature.rs`

```rust
pub fn embedder_temperature_range(embedder: Embedder) -> (f32, f32) {
    match embedder {
        Embedder::Causal => (0.8, 2.5),      // E5 overconfident
        Embedder::Code => (0.5, 1.5),         // E7 needs precision
        Embedder::Hdc => (1.0, 3.0),          // E9 noisy
        Embedder::KeywordSplade => (0.7, 2.0), // E13 sparse
        _ => (0.5, 2.0),                      // Default range
    }
}
```

**Default Temperatures** (from `Embedder::default_temperature()` in teleological module):
- Semantic (E1): 1.0
- Causal (E5): 1.2
- Code (E7): 0.9
- HDC (E9): 1.5

---

## Finding 4: Domain-Specific Thresholds

**STATUS: IMPLEMENTED**

**File**: `/crates/context-graph-core/src/atc/domain.rs`

All 6 PRD-mandated domains are present:

| Domain | Description | Strictness | Status |
|--------|-------------|------------|--------|
| Code | Low FP tolerance | 0.9 | IMPLEMENTED |
| Medical | Very strict, high causal | 1.0 | IMPLEMENTED |
| Legal | Moderate, semantic precision | 0.8 | IMPLEMENTED |
| Creative | Loose, exploration encouraged | 0.2 | IMPLEMENTED |
| Research | Balanced, novelty valued | 0.5 | IMPLEMENTED |
| General | Default priors | 0.5 | IMPLEMENTED |

**Transfer Learning**:
```rust
// Formula: theta_new = alpha * theta_similar_domain + (1 - alpha) * theta_general
pub fn blend_with_similar(&mut self, similar: &DomainThresholds, alpha: f32)
```

**Similarity Chain**:
- Code -> Research -> General
- Medical <-> Legal
- Creative -> Research -> General

---

## Finding 5: Calibration Metrics

**STATUS: IMPLEMENTED**

**File**: `/crates/context-graph-core/src/atc/calibration.rs`

| Metric | Target | Implementation | Status |
|--------|--------|----------------|--------|
| ECE (Expected Calibration Error) | < 0.05 | `compute_ece()` | IMPLEMENTED |
| MCE (Maximum Calibration Error) | < 0.10 | `compute_mce()` | IMPLEMENTED |
| Brier Score | < 0.10 | `compute_brier()` | IMPLEMENTED |

**Quality Status Levels**:
```rust
pub enum CalibrationStatus {
    Excellent,  // ECE < 0.05
    Good,       // 0.05 <= ECE < 0.10
    Acceptable, // 0.10 <= ECE < 0.15
    Poor,       // 0.15 <= ECE < 0.25
    Critical,   // ECE >= 0.25
}
```

**Self-Correction Protocol**:
- Minor (ECE [0.05, 0.10]): Increase EWMA alpha
- Moderate (ECE [0.10, 0.15]): Thompson exploration + temperature recalibration
- Major (ECE > 0.15): Reset to domain priors + Bayesian optimization
- Critical (ECE > 0.25): Fallback to conservative static

---

## Finding 6: NORTH-009 ThresholdLearner Service

**STATUS: IMPLEMENTED**

**File**: `/crates/context-graph-core/src/autonomous/services/threshold_learner.rs`

The ThresholdLearner service implements all 4 ATC levels:

```rust
pub struct ThresholdLearner {
    config: AdaptiveThresholdConfig,
    state: AdaptiveThresholdState,
    embedder_states: [EmbedderLearningState; NUM_EMBEDDERS],
    ewma_alpha: f32,
    total_observations: u32,
    total_successes: u32,
    bayesian_history: Vec<BayesianObservation>,
    best_performance: f32,
    last_recalibration_check: DateTime<Utc>,
    created_at: DateTime<Utc>,
}
```

**Key Methods**:
- `learn_from_feedback()`: Main entry point orchestrating all 4 levels
- `update_ewma()`: Level 1 drift tracking
- `temperature_scale()`: Level 2 calibration
- `thompson_sample()`: Level 3 exploration
- `bayesian_update()`: Level 4 meta-optimization

---

## Finding 7: MCP Tools

**STATUS: IMPLEMENTED**

**File**: `/crates/context-graph-mcp/src/handlers/atc.rs`

All 3 PRD-mandated MCP tools are present:

| Tool | Description | Status |
|------|-------------|--------|
| `get_threshold_status` | Current threshold config and drift | IMPLEMENTED |
| `get_calibration_metrics` | ECE, MCE, Brier and quality status | IMPLEMENTED |
| `trigger_recalibration` | Manual Level 1-4 recalibration | IMPLEMENTED |

**Tool Registration** (tools.rs):
```rust
pub const GET_THRESHOLD_STATUS: &str = "get_threshold_status";
pub const GET_CALIBRATION_METRICS: &str = "get_calibration_metrics";
pub const TRIGGER_RECALIBRATION: &str = "trigger_recalibration";
```

---

## Finding 8: Residual Hardcoded Thresholds

**STATUS: REQUIRES ATTENTION**

While the ATC system is comprehensive, some modules contain hardcoded thresholds that should be migrated:

### UTL Module
**File**: `/crates/context-graph-utl/src/config/thresholds.rs`

```rust
impl Default for UtlThresholds {
    fn default() -> Self {
        Self {
            min_score: 0.0,
            max_score: 1.0,
            high_quality: 0.6,        // HARDCODED
            low_quality: 0.3,         // HARDCODED
            coherence_recovery_secs: 10,
            info_loss_tolerance: 0.15, // HARDCODED
            compression_target: 0.6,   // HARDCODED
            ...
        }
    }
}
```

### Stage Thresholds
**File**: `/crates/context-graph-utl/src/metrics/thresholds.rs`

```rust
impl Default for StageThresholds {
    fn default() -> Self {
        Self {
            entropy_trigger: 0.7,        // HARDCODED
            coherence_trigger: 0.5,      // HARDCODED
            min_importance_store: 0.3,   // HARDCODED
            consolidation_threshold: 0.5, // HARDCODED
        }
    }
}
```

### Johari Classifier
**File**: `/crates/context-graph-utl/src/johari/classifier.rs`

```rust
const DEFAULT_THRESHOLD: f32 = 0.5;  // HARDCODED
```

### Dream Scheduler
**File**: `/crates/context-graph-core/src/dream/mod.rs`

```rust
pub const ACTIVITY_THRESHOLD: f32 = 0.15;  // HARDCODED
```

### Quality Gates Tests
**File**: `/crates/context-graph-utl/tests/quality_gates.rs`

```rust
const UTL_THRESHOLD: f64 = 0.6;               // HARDCODED
const ATTACK_DETECTION_THRESHOLD: f64 = 0.95; // HARDCODED
```

---

## Verification Matrix

| Check | Method | Expected | Actual | Verdict |
|-------|--------|----------|--------|---------|
| Level 1 EWMA | Code review | Drift detection | Present | INNOCENT |
| Level 2 Temperature | Code review | Per-embedder T | Present | INNOCENT |
| Level 3 Bandit | Code review | Thompson + UCB | Present | INNOCENT |
| Level 4 Bayesian | Code review | GP + EI | Present | INNOCENT |
| Domain Thresholds | Code review | 6 domains | 6 found | INNOCENT |
| Calibration Metrics | Code review | ECE/MCE/Brier | All present | INNOCENT |
| MCP Tools | Code review | 3 tools | 3 found | INNOCENT |
| No Hardcoded | Grep search | Zero matches | Several found | MINOR VIOLATION |

---

## Test Coverage

The ATC system has comprehensive test coverage:

**File**: `/crates/context-graph-core/tests/atc_integration.rs`

| Test | Purpose | Status |
|------|---------|--------|
| `test_ewma_drift_detection` | Level 1 verification | PASSES |
| `test_temperature_scaling_per_embedder` | Level 2 verification | PASSES |
| `test_thompson_sampling_convergence` | Level 3 verification | PASSES |
| `test_bayesian_gp_optimization` | Level 4 verification | PASSES |
| `test_domain_transfer_learning` | Domain blending | PASSES |
| `test_calibration_quality_monitoring` | ECE/MCE/Brier | PASSES |
| `test_fail_fast_on_missing_config` | Constraint validation | PASSES |
| `test_full_atc_system_integration` | End-to-end | PASSES |
| `test_edge_cases` | Boundary conditions | PASSES |
| `test_convergence_speed` | Performance regression | PASSES |

**MCP Tool Tests**: `/crates/context-graph-mcp/src/handlers/tests/exhaustive_mcp_tools.rs`

- 10+ tests for ATC tool handlers
- Full State Verification (FSV) tests for real data verification

---

## Recommendations

### Priority 1: Migrate Hardcoded Thresholds

The following modules should integrate with the ATC system:

1. **UTL Thresholds** (`utl/config/thresholds.rs`):
   - Add `AtcProvider` dependency
   - Replace hardcoded defaults with ATC lookups
   - Use domain-specific thresholds based on content type

2. **Stage Thresholds** (`utl/metrics/thresholds.rs`):
   - Integrate lifecycle stage thresholds with ATC
   - Allow ATC to adapt based on lifecycle phase

3. **Dream Scheduler** (`core/dream/mod.rs`):
   - Replace `ACTIVITY_THRESHOLD` constant with ATC-managed value
   - Allow adaptive activity detection

### Priority 2: Add theta_joh, theta_kur Integration

Currently these thresholds exist in separate config files but are not managed by the unified ATC system:

- Connect `JohariConfig` thresholds to ATC Level 2
- Connect `PhaseSyncConfig` thresholds to ATC Level 2

### Priority 3: Add Persistent Calibration State

The current implementation loses calibration state on restart:

```rust
// RECOMMENDATION: Add persistence methods
impl AdaptiveThresholdCalibration {
    pub fn save_state(&self, path: &Path) -> Result<(), Error>;
    pub fn load_state(path: &Path) -> Result<Self, Error>;
}
```

### Priority 4: Add Observability

Consider adding:
- Prometheus metrics for ECE/MCE/Brier
- Tracing spans for calibration operations
- Alert integration for Critical calibration status

---

## Conclusion

*"When you have eliminated the impossible, whatever remains, however improbable, must be the truth."*

The ContextGraph codebase demonstrates a **sophisticated and constitution-compliant Adaptive Threshold Calibration system**. The 4-level architecture (EWMA, Temperature, Bandit, Bayesian) is fully implemented with:

- Proper drift detection and escalation triggers
- Per-embedder temperature scaling with calibration
- Thompson Sampling and UCB exploration strategies
- Bayesian optimization with GP surrogate and Expected Improvement
- Domain-specific thresholds for 6 content domains
- Complete calibration metrics (ECE, MCE, Brier)
- MCP tool integration for runtime management

The system is **NOT GUILTY** of the primary charge of missing adaptive thresholds. However, it is guilty of **MINOR VIOLATIONS** - residual hardcoded thresholds in peripheral modules that should be migrated to the unified ATC system.

**CASE STATUS**: INVESTIGATION COMPLETE

**RECOMMENDED ACTION**: Migrate remaining hardcoded thresholds to ATC system within next sprint cycle.

---

*"The world is full of obvious things which nobody by any chance ever observes."*

**Signed**: Sherlock Holmes, Forensic Code Detective
**Date**: 2026-01-10
