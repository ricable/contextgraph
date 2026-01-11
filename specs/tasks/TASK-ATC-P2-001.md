# TASK-ATC-P2-001: Discover All Hardcoded Thresholds via Codebase Scan

**Version:** 2.0
**Status:** Ready
**Layer:** Foundation
**Sequence:** 1
**Implements:** REQ-ATC-001 through REQ-ATC-005
**Depends On:** None
**Estimated Complexity:** Medium
**Priority:** P2

---

## Metadata

```yaml
id: TASK-ATC-P2-001
title: Discover All Hardcoded Thresholds via Codebase Scan
status: ready
layer: foundation
sequence: 1
implements:
  - REQ-ATC-001
  - REQ-ATC-002
  - REQ-ATC-003
  - REQ-ATC-004
  - REQ-ATC-005
depends_on: []
estimated_complexity: medium
```

---

## Context

This is the foundational task for the ATC threshold migration. Before any migration can occur, we must have a complete inventory of all hardcoded thresholds categorized by:

1. **Critical (Must Migrate):** Behavioral thresholds affecting consciousness, learning, or decision-making
2. **Should Migrate:** Thresholds benefiting from domain adaptation
3. **Evaluate:** Thresholds requiring further analysis
4. **Intentionally Static:** Numerical constants that should not be adaptive

The goal is to generate a YAML inventory file that subsequent tasks (TASK-ATC-P2-002 through TASK-ATC-P2-008) will use for actual migration.

---

## Critical Rules

1. **NO BACKWARDS COMPATIBILITY** - The system must work after changes or fail fast for debugging
2. **NO WORKAROUNDS OR FALLBACKS** - If something doesn't work, it must error with robust logging
3. **NO MOCK DATA IN TESTS** - Use real data and test actual functionality
4. **FAIL FAST** - Errors must surface immediately with clear diagnostic information

---

## Current State of the Codebase

### ATC Module Already Exists

The ATC system is already implemented at:
- **Main module:** `crates/context-graph-core/src/atc/mod.rs`
- **Domain thresholds:** `crates/context-graph-core/src/atc/domain.rs`
- **Level 1 EWMA:** `crates/context-graph-core/src/atc/level1_ewma.rs`
- **Level 2 Temperature:** `crates/context-graph-core/src/atc/level2_temperature.rs`
- **Level 3 Bandit:** `crates/context-graph-core/src/atc/level3_bandit.rs`
- **Level 4 Bayesian:** `crates/context-graph-core/src/atc/level4_bayesian.rs`
- **Calibration:** `crates/context-graph-core/src/atc/calibration.rs`

### DomainThresholds Structure (domain.rs)

```rust
pub struct DomainThresholds {
    pub domain: Domain,
    pub theta_opt: f32,      // Optimal alignment [0.60, 0.90]
    pub theta_acc: f32,      // Acceptable alignment [0.55, 0.85]
    pub theta_warn: f32,     // Warning threshold [0.40, 0.70]
    pub theta_dup: f32,      // Duplicate detection [0.80, 0.98]
    pub theta_edge: f32,     // Edge weight threshold [0.50, 0.85]
    pub confidence_bias: f32, // Domain calibration
}
```

**Domains:** Code, Medical, Legal, Creative, Research, General

---

## Verified Threshold Locations (Current Codebase)

### Bio-Nervous Layers (`crates/context-graph-core/src/layers/`)

| File | Constant | Value | Line | Purpose |
|------|----------|-------|------|---------|
| `coherence.rs` | `GW_THRESHOLD` | 0.7 | 60 | Global Workspace broadcast gate |
| `coherence.rs` | `HYPERSYNC_THRESHOLD` | 0.95 | 69 | Pathological hypersynchronization |
| `coherence.rs` | `FRAGMENTATION_THRESHOLD` | 0.5 | 72 | Fragmented state detection |
| `coherence.rs` | `KURAMOTO_K` | 2.0 | 53 | Coupling strength |
| `memory.rs` | `MIN_MEMORY_SIMILARITY` | 0.5 | 52 | Memory relevance threshold |
| `memory.rs` | `DEFAULT_MHN_BETA` | 2.0 | 46 | Hopfield temperature |
| `reflex.rs` | `MIN_HIT_SIMILARITY` | 0.85 | 52 | Cache hit threshold |
| `reflex.rs` | `DEFAULT_BETA` | 1.0 | 48 | Reflex Hopfield beta |
| `learning.rs` | `DEFAULT_CONSOLIDATION_THRESHOLD` | 0.1 | 47 | Weight consolidation trigger |
| `learning.rs` | `DEFAULT_LEARNING_RATE` | 0.0005 | 44 | UTL learning rate |
| `learning.rs` | `GRADIENT_CLIP` | 1.0 | 50 | Gradient clipping |

### Dream Layer (`crates/context-graph-core/src/dream/`)

| File | Constant | Value | Purpose |
|------|----------|-------|---------|
| `mod.rs` | `ACTIVITY_THRESHOLD` | 0.15 | Dream trigger threshold |
| `mod.rs` | `MIN_SEMANTIC_LEAP` | 0.7 | REM exploration distance |
| `mod.rs` | `SHORTCUT_CONFIDENCE_THRESHOLD` | 0.7 | Amortized learning confidence |
| `mod.rs` | `NREM_COUPLING` | 0.9 | Hebbian replay coupling |
| `mod.rs` | `REM_TEMPERATURE` | 2.0 | Exploration temperature |
| `mod.rs` | `NREM_RECENCY_BIAS` | 0.8 | Recent memory bias |
| `mod.rs` | `MAX_GPU_USAGE` | 0.30 | GPU budget during dream |
| `mod.rs` | `MIN_SHORTCUT_HOPS` | 3 | Shortcut minimum path length |
| `mod.rs` | `MIN_SHORTCUT_TRAVERSALS` | 5 | Shortcut minimum traversals |

### Config Constants (`crates/context-graph-core/src/config/constants.rs`)

| Module | Constant | Value | Purpose |
|--------|----------|-------|---------|
| `alignment::` | `OPTIMAL` | 0.75 | Constitution teleological.thresholds.optimal |
| `alignment::` | `ACCEPTABLE` | 0.70 | Constitution teleological.thresholds.acceptable |
| `alignment::` | `WARNING` | 0.55 | Constitution teleological.thresholds.warning |
| `alignment::` | `CRITICAL` | 0.55 | Constitution teleological.thresholds.critical |
| `johari::` | `BOUNDARY` | 0.5 | Johari quadrant boundary |
| `johari::` | `BLIND_SPOT_THRESHOLD` | 0.5 | Blind spot detection |

### Neuromodulation (`crates/context-graph-core/src/neuromod/`)

| File | Constant | Value | Purpose |
|------|----------|-------|---------|
| `dopamine.rs` | `DA_DECAY_RATE` | 0.05 | Dopamine decay |
| `dopamine.rs` | `DA_WORKSPACE_INCREMENT` | 0.2 | GW broadcast boost |
| `serotonin.rs` | `SEROTONIN_BASELINE` | 0.5 | Serotonin baseline |
| `serotonin.rs` | `SEROTONIN_DECAY_RATE` | 0.02 | Decay rate |
| `noradrenaline.rs` | `NE_MIN` | 0.5 | Minimum attention temp |
| `noradrenaline.rs` | `NE_DECAY_RATE` | 0.1 | Decay rate |
| `noradrenaline.rs` | `NE_THREAT_SPIKE` | 0.5 | Threat response |
| `acetylcholine.rs` | `ACH_BASELINE` | 0.001 | Learning rate baseline |
| `acetylcholine.rs` | `ACH_MAX` | 0.002 | Maximum learning rate |
| `acetylcholine.rs` | `ACH_DECAY_RATE` | 0.1 | Decay rate |

### Autonomous Services (`crates/context-graph-core/src/autonomous/services/`)

| File | Constant | Value | Purpose |
|------|----------|-------|---------|
| `obsolescence_detector.rs` | `DEFAULT_RELEVANCE_THRESHOLD` | 0.3 | Goal relevance |
| `obsolescence_detector.rs` | `HIGH_CONFIDENCE_THRESHOLD` | 0.8 | Action confidence |
| `obsolescence_detector.rs` | `MEDIUM_CONFIDENCE_THRESHOLD` | 0.6 | Medium confidence |
| `drift_detector.rs` | `MIN_SAMPLES_DEFAULT` | 10 | Drift detection samples |
| `drift_detector.rs` | `SLOPE_THRESHOLD` | 0.005 | Trend detection |

### UTL Config (`crates/context-graph-utl/src/`)

| File | Constant | Value | Purpose |
|------|----------|-------|---------|
| `config/thresholds.rs` | `high_quality` | 0.6 | UTL quality threshold |
| `config/thresholds.rs` | `low_quality` | 0.3 | Low quality threshold |
| `config/thresholds.rs` | `info_loss_tolerance` | 0.15 | Information loss limit |
| `lifecycle/stage.rs` | `INFANCY_THRESHOLD` | 50 | Lifecycle stage boundary |
| `lifecycle/stage.rs` | `GROWTH_THRESHOLD` | 500 | Growth stage boundary |
| `johari/classifier.rs` | `DEFAULT_THRESHOLD` | 0.5 | Johari default |

### MCP Handlers (`crates/context-graph-mcp/src/handlers/`)

| File | Constant | Value | Purpose |
|------|----------|-------|---------|
| `utl.rs` | `LEARNING_SCORE_TARGET` | 0.6 | Quality gate target |
| `utl.rs` | `ATTACK_DETECTION_TARGET` | 0.95 | Security detection |
| `utl.rs` | `FALSE_POSITIVE_TARGET` | 0.02 | FP rate target |
| `utl.rs` | `ALPHA` | 0.4 | Connectivity weight in ΔC |
| `utl.rs` | `BETA` | 0.4 | ClusterFit weight in ΔC |
| `utl.rs` | `GAMMA` | 0.2 | Consistency weight in ΔC |
| `core.rs` | `TREND_THRESHOLD` | 0.02 | Trend detection |

### GWT (`crates/context-graph-core/src/gwt/`)

| File | Constant | Value | Purpose |
|------|----------|-------|---------|
| `workspace.rs` | `DA_INHIBITION_FACTOR` | 0.1 | DA inhibition on exit |

### Storage (`crates/context-graph-storage/src/`)

| File | Constant | Value | Purpose |
|------|----------|-------|---------|
| `teleological/serialization.rs` | `MIN_FINGERPRINT_SIZE` | 5000 | Validation |
| `teleological/serialization.rs` | `MAX_FINGERPRINT_SIZE` | 150000 | Validation |
| `teleological/search/token_storage.rs` | `MAX_TOKENS_PER_MEMORY` | 512 | Token limit |

---

## Scope

### In Scope

1. **Systematic scan** using ripgrep patterns:
   - `const.*THRESHOLD`
   - `const.*MIN_`
   - `const.*MAX_`
   - `: f32 = 0.` or `: f64 = 0.`
   - Comments containing "threshold"

2. **Classification** of each threshold:
   - Category (critical/should/evaluate/static)
   - Current value and type
   - File:line location
   - Semantic purpose
   - Domain sensitivity (high/medium/low/none)
   - Proposed ATC field name

3. **Generate** `specs/tasks/threshold-inventory.yaml`

4. **Identify** thresholds already managed by ATC

### Out of Scope

- Actual code migration (TASK-ATC-P2-002+)
- ATC struct modifications
- Test updates
- Documentation beyond inventory

---

## Definition of Done

### Deliverable

**File:** `specs/tasks/threshold-inventory.yaml`

```yaml
# Threshold Inventory - TASK-ATC-P2-001
version: "2.0"
generated_at: "2026-01-11T00:00:00Z"
total_count: <number>

categories:
  critical:
    count: <number>
    thresholds:
      - name: "GW_THRESHOLD"
        file: "crates/context-graph-core/src/layers/coherence.rs"
        line: 60
        current_value: 0.7
        type: "f32"
        purpose: "Global Workspace broadcast gate"
        domain_sensitivity: "high"
        atc_field: "theta_gate"
        rationale: "Determines when memories enter consciousness"
        constitution_ref: "gwt.workspace.coherence_threshold"

  should_migrate:
    count: <number>
    thresholds: [...]

  evaluate:
    count: <number>
    thresholds: [...]

  static:
    count: <number>
    thresholds: [...]
    rationale_required: true

summary:
  total_thresholds: <number>
  requires_migration: <number>
  already_using_atc: <number>
  intentionally_static: <number>
```

### Constraints

- MUST scan all `crates/` subdirectories
- MUST NOT modify any source files
- MUST categorize ALL discovered thresholds (no gaps)
- MUST provide rationale for static classification
- MUST identify domain sensitivity for migratable thresholds

---

## Full State Verification (FSV) Requirements

After completing the discovery logic, you MUST perform FSV:

### 1. Define Source of Truth

The source of truth is `specs/tasks/threshold-inventory.yaml`. After generation, this file must:
- Be valid YAML (parseable by Python yaml.safe_load)
- Have total_count matching actual threshold entries
- Have no duplicate (file, line) tuples

### 2. Execute & Inspect

```bash
# Generate inventory
# ... (your script/command)

# Verify YAML syntax
python3 -c "import yaml; data=yaml.safe_load(open('specs/tasks/threshold-inventory.yaml')); print(f'Loaded {data[\"total_count\"]} thresholds')"

# Count entries
grep -c "^      - name:" specs/tasks/threshold-inventory.yaml

# Cross-check with ripgrep
rg "const.*THRESHOLD.*f(32|64)" crates/ --count-matches | wc -l
```

### 3. Boundary & Edge Case Audit

Run these 3 edge cases and document state before/after:

**Edge Case 1: Empty Pattern Match**
```bash
# Before: Check if pattern exists in test file
rg "NONEXISTENT_THRESHOLD_XYZ" crates/
# After: Should return 0 matches, inventory should not include it
```

**Edge Case 2: Multi-line Constant Declaration**
```bash
# Find constants spread across lines
rg -U "pub const.*\n.*f32" crates/ --count-matches
# Verify these are captured in inventory
```

**Edge Case 3: Already-ATC-Managed Thresholds**
```bash
# Check DomainThresholds fields
rg "theta_opt|theta_acc|theta_warn" crates/context-graph-core/src/atc/
# These should be marked as "already_using_atc" in inventory
```

### 4. Evidence of Success

Provide a log showing:
1. Total thresholds discovered by ripgrep
2. Total thresholds in inventory YAML
3. Breakdown by category (critical/should/evaluate/static)
4. Verification that counts match

---

## Manual Testing Procedures

### Happy Path Tests

**Test 1: Basic Threshold Discovery**
- Input: Run ripgrep `const.*THRESHOLD` on crates/
- Expected: All known thresholds from "Verified Threshold Locations" appear
- Verification: Compare output against table above

**Test 2: YAML Generation**
- Input: Generate threshold-inventory.yaml
- Expected: Valid YAML, parseable, all fields present
- Verification: `python3 -c "import yaml; yaml.safe_load(open('specs/tasks/threshold-inventory.yaml'))"`

**Test 3: Category Assignment**
- Input: Review GW_THRESHOLD entry
- Expected: Category="critical", domain_sensitivity="high", atc_field present
- Verification: Manual inspection of YAML

### Synthetic Test Data

Use these known thresholds to verify discovery:

| Synthetic Input | Expected Category | Expected atc_field |
|-----------------|-------------------|--------------------|
| `GW_THRESHOLD: f32 = 0.7` | critical | theta_gate |
| `MIN_MEMORY_SIMILARITY: f32 = 0.5` | should_migrate | theta_memory_relevance |
| `GRADIENT_CLIP: f32 = 1.0` | static | N/A (numerical stability) |
| `KURAMOTO_K: f32 = 2.0` | evaluate | (bio-inspired constant) |

### Edge Case Tests

**Edge 1: Zero-value thresholds**
```bash
rg ": f32 = 0\.0" crates/
# Verify these aren't classified as thresholds unless semantically meaningful
```

**Edge 2: Negative thresholds**
```bash
rg "const.*= -" crates/ --type rust
# Expect: FAILURE_PREDICTION_DELTA = -0.15 (should be evaluate/static)
```

**Edge 3: Integer thresholds**
```bash
rg "const.*THRESHOLD.*usize|u64|i32" crates/
# Expect: MIN_SHORTCUT_HOPS, MIN_SHORTCUT_TRAVERSALS
```

---

## Validation Criteria

| Criterion | Validation Method |
|-----------|-------------------|
| Inventory is valid YAML | Python yaml.safe_load |
| All thresholds categorized | total_count == sum of category counts |
| Critical thresholds have ATC field | grep for atc_field in critical section |
| Static thresholds have rationale | grep for rationale in static section |
| No duplicate entries | sort unique check on file:line |
| Domain sensitivity for migratable | grep for domain_sensitivity |

---

## Test Commands

```bash
# 1. Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('specs/tasks/threshold-inventory.yaml'))"

# 2. Count total thresholds discovered
grep -c "^      - name:" specs/tasks/threshold-inventory.yaml

# 3. Verify known thresholds exist
rg "GW_THRESHOLD|HYPERSYNC_THRESHOLD|MIN_MEMORY_SIMILARITY" crates/ --count

# 4. Check for completeness against ripgrep
GREP_COUNT=$(rg "const.*THRESHOLD.*f(32|64)" crates/ -c | awk -F: '{sum+=$2} END {print sum}')
echo "Ripgrep found: $GREP_COUNT"

# 5. Verify no duplicates
grep "file:" specs/tasks/threshold-inventory.yaml | sort | uniq -d | wc -l
# Should output 0
```

---

## Acceptance Criteria Checklist

### Discovery Completeness
- [ ] All `crates/context-graph-*` directories scanned
- [ ] Pattern `const.*THRESHOLD` found and documented
- [ ] Pattern `const.*MIN_` found and documented
- [ ] Pattern `const.*MAX_` found and documented
- [ ] Pattern `: f32 = 0.` found and documented (behavioral ones)
- [ ] All results deduplicated by (file, line) tuple

### Classification Accuracy
- [ ] Every threshold assigned exactly one category
- [ ] Critical thresholds have `atc_field` mapping
- [ ] Critical thresholds have `domain_sensitivity` rating
- [ ] Static thresholds have `rationale` explaining why
- [ ] Evaluate thresholds have `notes` for domain experts

### Inventory Quality
- [ ] YAML file passes syntax validation
- [ ] total_count matches sum of category counts
- [ ] No duplicate entries
- [ ] All file paths are relative to project root
- [ ] Line numbers verified against current codebase

### FSV Evidence
- [ ] Source of truth defined (threshold-inventory.yaml)
- [ ] Execute & inspect completed with logs
- [ ] 3 edge cases audited with before/after state
- [ ] Evidence log provided showing actual data in system

---

## Constitution Reference

From `/docs2/constitution.yaml` lines 309-326:

```yaml
adaptive_thresholds:
  priors:
    θ_opt: [0.75, "[0.60,0.90]"]
    θ_acc: [0.70, "[0.55,0.85]"]
    θ_warn: [0.55, "[0.40,0.70]"]
    θ_dup: [0.90, "[0.80,0.98]"]
    θ_kur: [0.80, "[0.65,0.95]"]

  levels:
    L1_EWMA: { freq: "per-query", formula: "θ_ewma=α×θ_obs+(1-α)×θ_ewma" }
    L2_Temp: { freq: hourly, formula: "σ(logit(raw)/T)" }
    L3_Bandit: { freq: session, method: "Thompson sampling Beta(α,β)" }
    L4_Bayesian: { freq: weekly, surrogate: "GP", acquisition: "EI" }

  calibration: { ECE: "<0.05", MCE: "<0.10", Brier: "<0.10" }
```

---

## Notes

- This task is READ-ONLY and produces documentation only
- Subsequent tasks (ATC-P2-002+) will use this inventory for actual migration
- Pay special attention to thresholds in `atc/` module - these are internal to ATC
- Neuromodulation thresholds are bio-inspired and may warrant "evaluate" classification
- The ATC system already exists - this task discovers what STILL needs migration

---

**Created:** 2026-01-11
**Updated:** 2026-01-11
**Author:** AI Coding Agent
**Status:** Ready for implementation
