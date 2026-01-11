# SPEC-ATC-001: Hardcoded Threshold Migration to Adaptive Threshold Calibration (ATC)

**Version:** 1.0
**Status:** Approved
**Priority:** P2 (Minor Refinement)
**Owner:** Context Graph Core Team
**Last Updated:** 2026-01-11
**Related Specs:** SPEC-CORE-001, SPEC-GWT-001

---

## 1. Overview

This specification defines the systematic migration of legacy hardcoded threshold constants scattered across the codebase to the centralized 4-level Adaptive Threshold Calibration (ATC) system. The ATC system provides domain-aware, self-calibrating thresholds that adapt to usage patterns.

### 1.1 Problem Statement

The ContextGraph codebase contains approximately 50+ hardcoded threshold constants across multiple modules. These thresholds:

- Bypass the 4-level ATC system defined in constitution.yaml (lines 809-837)
- Cannot adapt to different domains (Code, Medical, Legal, Creative, Research, General)
- Fail to benefit from EWMA drift tracking, temperature scaling, Thompson sampling, or Bayesian optimization
- Create maintenance burden when threshold tuning is required
- Violate architectural rule AP-12: "No magic numbers; define named constants"

### 1.2 Solution

Migrate all relevant hardcoded thresholds to retrieve values through the `AdaptiveThresholdCalibration` API:

```rust
// BEFORE (hardcoded)
const PHI_THRESHOLD: f32 = 0.8;
if phi > PHI_THRESHOLD { ... }

// AFTER (ATC-managed)
let threshold = atc.get_domain_thresholds(domain)
    .map(|t| t.theta_gate)
    .unwrap_or(DomainThresholds::default().theta_gate);
if phi > threshold { ... }
```

---

## 2. User Stories

### US-ATC-001: Developer Threshold Visibility
**Priority:** Must-Have

```
As a system developer
I want all behavioral thresholds managed through ATC
So that I can tune them per-domain without code changes
```

**Acceptance Criteria:**
- GIVEN a hardcoded threshold in the codebase
- WHEN I trace its usage
- THEN it MUST resolve to an ATC-managed value OR be documented as intentionally static

### US-ATC-002: Domain-Adaptive Behavior
**Priority:** Must-Have

```
As the consciousness system
I want thresholds that adapt to domain context
So that Medical domain uses stricter thresholds than Creative domain
```

**Acceptance Criteria:**
- GIVEN a threshold used in coherence calculation
- WHEN the active domain is Medical
- THEN the threshold MUST be stricter than General domain defaults

### US-ATC-003: Threshold Drift Tracking
**Priority:** Should-Have

```
As the ATC system
I want to observe all threshold usage
So that Level 1 EWMA can detect distribution drift
```

**Acceptance Criteria:**
- GIVEN a threshold retrieved from ATC
- WHEN the value is used in a decision
- THEN the usage MUST be observable by the drift tracker

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Validation |
|----|-------------|----------|------------|
| REQ-ATC-001 | All GWT thresholds (Phi, Kuramoto, coherence) MUST use ATC | Must | Code audit |
| REQ-ATC-002 | All UTL thresholds (entropy, coherence bounds) MUST use ATC | Must | Code audit |
| REQ-ATC-003 | All dream layer thresholds MUST use ATC | Must | Code audit |
| REQ-ATC-004 | All neuromodulation bounds MUST use ATC | Should | Code audit |
| REQ-ATC-005 | All layer thresholds (reflex, memory, learning) MUST use ATC | Must | Code audit |
| REQ-ATC-006 | Threshold retrieval MUST support domain context | Must | Unit tests |
| REQ-ATC-007 | Threshold retrieval MUST fallback to General domain if unspecified | Must | Unit tests |
| REQ-ATC-008 | Threshold usage MUST be observable for EWMA tracking | Should | Integration tests |

### 3.2 Non-Functional Requirements

| ID | Category | Requirement | Metric |
|----|----------|-------------|--------|
| NFR-ATC-001 | Performance | Threshold retrieval MUST be < 1us | Benchmark |
| NFR-ATC-002 | Memory | No additional allocations per threshold access | Heap profiling |
| NFR-ATC-003 | Backward Compat | Existing behavior MUST be preserved for General domain | Regression tests |

---

## 4. Threshold Inventory

### 4.1 Critical Thresholds (Must Migrate)

| File | Constant | Current Value | Target ATC Field | Domain Sensitivity |
|------|----------|---------------|------------------|-------------------|
| `layers/coherence.rs:60` | `GW_THRESHOLD` | 0.7 | `theta_gate` | High |
| `layers/coherence.rs:69` | `HYPERSYNC_THRESHOLD` | 0.95 | `theta_hypersync` | Medium |
| `layers/coherence.rs:72` | `FRAGMENTATION_THRESHOLD` | 0.5 | `theta_fragmentation` | Medium |
| `layers/memory.rs:52` | `MIN_MEMORY_SIMILARITY` | 0.5 | `theta_memory_sim` | High |
| `layers/reflex.rs:52` | `MIN_HIT_SIMILARITY` | 0.85 | `theta_reflex_hit` | Low |
| `layers/learning.rs:47` | `DEFAULT_CONSOLIDATION_THRESHOLD` | 0.1 | `theta_consolidation` | Medium |
| `dream/mod.rs:74` | `ACTIVITY_THRESHOLD` | 0.15 | `theta_dream_activity` | Low |
| `dream/mod.rs:83` | `MIN_SEMANTIC_LEAP` | 0.7 | `theta_semantic_leap` | Medium |
| `dream/mod.rs:108` | `SHORTCUT_CONFIDENCE_THRESHOLD` | 0.7 | `theta_shortcut_conf` | Medium |
| `config/constants.rs:37-55` | `AlignmentThresholds::*` | Various | `theta_opt/acc/warn` | Critical |
| `config/constants.rs:114` | `BLIND_SPOT_THRESHOLD` | 0.5 | `theta_blind_spot` | Medium |
| `types/fingerprint/johari/core.rs:47-50` | `ENTROPY/COHERENCE_THRESHOLD` | 0.5 | `theta_johari` | Medium |
| `utl/johari/classifier.rs:55` | `DEFAULT_THRESHOLD` | 0.5 | `theta_johari` | Medium |
| `autonomous/services/obsolescence_detector.rs:17-21` | `*_THRESHOLD` | Various | `theta_obsolescence_*` | Medium |
| `autonomous/services/drift_detector.rs:283` | `SLOPE_THRESHOLD` | 0.005 | `theta_drift_slope` | Low |

### 4.2 Neuromodulation Thresholds (Should Migrate)

| File | Constant | Current Value | Notes |
|------|----------|---------------|-------|
| `neuromod/dopamine.rs:32-35` | `DA_DECAY_RATE`, `DA_WORKSPACE_INCREMENT` | 0.05, 0.2 | Bio-inspired, less domain-sensitive |
| `neuromod/serotonin.rs:29-38` | `SEROTONIN_*` | Various | Bio-inspired constants |
| `neuromod/acetylcholine.rs:30-36` | `ACH_*` | Various | Bio-inspired constants |
| `neuromod/noradrenaline.rs:31-40` | `NE_*` | Various | Bio-inspired constants |

### 4.3 Embedder-Specific Thresholds (Evaluate)

| File | Constant | Current Value | Notes |
|------|----------|---------------|-------|
| `warm/cuda_alloc/constants.rs:54` | `SIN_WAVE_ENERGY_THRESHOLD` | 0.80 | Validation threshold |
| `warm/cuda_alloc/constants.rs:59` | `GOLDEN_SIMILARITY_THRESHOLD` | 0.99 | Validation threshold |
| `warm/loader/types.rs:890-896` | `MIN_GOLDEN_SIMILARITY`, etc. | Various | Validation thresholds |

### 4.4 Intentionally Static (Do Not Migrate)

| File | Constant | Value | Rationale |
|------|----------|-------|-----------|
| `atc/level4_bayesian.rs:54-56` | `LENGTH_SCALE`, `NOISE_VARIANCE` | 0.1, 0.01 | GP hyperparameters |
| `similarity/multi_utl.rs:293` | `MIN_SIGNAL_THRESHOLD` | 0.001 | Numerical stability |
| `graph/entailment/cones/operations.rs:32` | `MIN_APERTURE` | 1e-6 | Numerical stability |

---

## 5. Architecture Changes

### 5.1 ATC Extension

Extend `DomainThresholds` struct with new fields:

```rust
pub struct DomainThresholds {
    pub domain: Domain,
    // Existing
    pub theta_opt: f32,
    pub theta_acc: f32,
    pub theta_warn: f32,
    pub theta_dup: f32,
    pub theta_edge: f32,
    pub confidence_bias: f32,

    // NEW - GWT thresholds
    pub theta_gate: f32,           // GW broadcast gate (default: 0.7-0.8)
    pub theta_hypersync: f32,      // Hypersync detection (default: 0.95)
    pub theta_fragmentation: f32,  // Fragmentation warning (default: 0.5)

    // NEW - Layer thresholds
    pub theta_memory_sim: f32,     // Memory similarity (default: 0.5)
    pub theta_reflex_hit: f32,     // Reflex cache hit (default: 0.85)
    pub theta_consolidation: f32,  // Consolidation trigger (default: 0.1)

    // NEW - Dream thresholds
    pub theta_dream_activity: f32,  // Dream trigger (default: 0.15)
    pub theta_semantic_leap: f32,   // REM exploration (default: 0.7)
    pub theta_shortcut_conf: f32,   // Shortcut confidence (default: 0.7)

    // NEW - Classification thresholds
    pub theta_johari: f32,          // Johari quadrant boundary (default: 0.5)
    pub theta_blind_spot: f32,      // Blind spot detection (default: 0.5)

    // NEW - Autonomous thresholds
    pub theta_obsolescence_low: f32,  // Low relevance (default: 0.3)
    pub theta_obsolescence_mid: f32,  // Medium confidence (default: 0.6)
    pub theta_obsolescence_high: f32, // High confidence (default: 0.8)
}
```

### 5.2 Threshold Accessor Trait

Create unified accessor interface:

```rust
pub trait ThresholdAccessor {
    fn get_threshold(&self, name: &str, domain: Domain) -> f32;
    fn observe_threshold_usage(&mut self, name: &str, value: f32, outcome: bool);
}
```

---

## 6. Edge Cases

| ID | Scenario | Expected Behavior |
|----|----------|-------------------|
| EC-ATC-001 | Domain not specified | Use General domain thresholds |
| EC-ATC-002 | Threshold name not found | Return documented default, log warning |
| EC-ATC-003 | ATC not initialized | Fall back to static defaults |
| EC-ATC-004 | Threshold outside valid range after adaptation | Clamp to constitution-defined range |

---

## 7. Error States

| ID | HTTP Code | Condition | Message | Recovery |
|----|-----------|-----------|---------|----------|
| ERR-ATC-001 | N/A | Unknown threshold name | "Unknown threshold: {name}" | Use default |
| ERR-ATC-002 | N/A | Invalid domain string | "Invalid domain: {domain}" | Use General |

---

## 8. Test Plan

### 8.1 Unit Tests

| ID | Type | Description | Input | Expected |
|----|------|-------------|-------|----------|
| TC-ATC-001 | Unit | Extended DomainThresholds creation | Domain::Code | All fields populated |
| TC-ATC-002 | Unit | Domain-specific threshold differences | Code vs Creative | Code thresholds stricter |
| TC-ATC-003 | Unit | Threshold accessor trait | "theta_gate", Code | Returns domain value |
| TC-ATC-004 | Unit | Unknown threshold fallback | "unknown", Code | Returns default, logs warning |

### 8.2 Integration Tests

| ID | Type | Description |
|----|------|-------------|
| TC-ATC-010 | Integration | GWT coherence uses ATC thresholds |
| TC-ATC-011 | Integration | Dream layer uses ATC thresholds |
| TC-ATC-012 | Integration | Layer modules use ATC thresholds |
| TC-ATC-013 | Integration | Threshold usage observable by EWMA |

### 8.3 Regression Tests

| ID | Type | Description |
|----|------|-------------|
| TC-ATC-020 | Regression | General domain behavior unchanged |
| TC-ATC-021 | Regression | Existing tests pass with ATC |

---

## 9. Migration Strategy

### Phase 1: Discovery (TASK-ATC-P2-001)
- Scan codebase for all hardcoded thresholds
- Classify into migrate/evaluate/static categories
- Document in threshold inventory

### Phase 2: ATC Extension (TASK-ATC-P2-002)
- Extend DomainThresholds struct
- Add domain-specific initialization
- Create threshold accessor trait

### Phase 3: GWT Migration (TASK-ATC-P2-003)
- Migrate GW_THRESHOLD
- Migrate HYPERSYNC_THRESHOLD
- Migrate FRAGMENTATION_THRESHOLD

### Phase 4: Layer Migration (TASK-ATC-P2-004)
- Migrate memory layer thresholds
- Migrate reflex layer thresholds
- Migrate learning layer thresholds

### Phase 5: Dream Migration (TASK-ATC-P2-005)
- Migrate dream layer thresholds
- Migrate semantic leap threshold
- Migrate shortcut confidence threshold

### Phase 6: Johari Migration (TASK-ATC-P2-006)
- Migrate Johari classification thresholds
- Migrate blind spot threshold
- Update classifier to use ATC

### Phase 7: Autonomous Services (TASK-ATC-P2-007)
- Migrate obsolescence detector thresholds
- Migrate drift detector thresholds

### Phase 8: Validation (TASK-ATC-P2-008)
- Run full test suite
- Verify domain-specific behavior
- Performance benchmarks

---

## 10. Success Criteria

1. **Coverage:** 100% of critical thresholds use ATC
2. **Tests:** All existing tests pass
3. **Performance:** No measurable latency regression
4. **Documentation:** Each migrated threshold documented

---

## 11. Dependencies

- ATC module fully implemented (COMPLETED)
- Domain enum and DomainThresholds struct (COMPLETED)
- Level 1-4 calibration infrastructure (COMPLETED)

---

## 12. Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Behavior change for General domain | Low | High | Ensure defaults match current values |
| Performance regression | Low | Medium | Threshold retrieval is O(1) HashMap lookup |
| Missing threshold during migration | Medium | Low | Automated scanning + code review |

---

## Appendix A: Constitution References

From `constitution.yaml` lines 809-837:

```yaml
adaptive_thresholds:
  rationale: "Different domains/users/drift require different thresholds"
  priors:
    θ_opt: [0.75, "[0.60,0.90]", session]
    θ_acc: [0.70, "[0.55,0.85]", session]
    θ_warn: [0.55, "[0.40,0.70]", session]
    θ_dup: [0.90, "[0.80,0.98]", hourly]
    θ_edge: [0.70, "[0.50,0.85]", hourly]
    θ_joh: [0.50, "[0.35,0.65]", per-embedder]
    θ_kur: [0.80, "[0.65,0.95]", daily]
```

---

**Document History:**
- 2026-01-11: Initial version (v1.0)
