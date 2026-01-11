# TASK-ATC-P2-008: Validation and Integration Testing

**Version:** 1.0
**Status:** Ready
**Layer:** Surface
**Sequence:** 8
**Implements:** REQ-ATC-008, NFR-ATC-001, NFR-ATC-002, NFR-ATC-003
**Depends On:** TASK-ATC-P2-003, TASK-ATC-P2-004, TASK-ATC-P2-005, TASK-ATC-P2-006, TASK-ATC-P2-007
**Estimated Complexity:** Medium
**Priority:** P2

---

## Metadata

```yaml
id: TASK-ATC-P2-008
title: Validation and Integration Testing
status: ready
layer: surface
sequence: 8
implements:
  - REQ-ATC-008
  - NFR-ATC-001
  - NFR-ATC-002
  - NFR-ATC-003
depends_on:
  - TASK-ATC-P2-003
  - TASK-ATC-P2-004
  - TASK-ATC-P2-005
  - TASK-ATC-P2-006
  - TASK-ATC-P2-007
estimated_complexity: medium
```

---

## Context

This is the final task in the ATC threshold migration sequence. All threshold migrations have been completed in tasks 003-007. This task validates the complete migration by:

1. Running full regression test suite
2. Verifying domain-specific behavior end-to-end
3. Measuring performance impact
4. Validating EWMA drift tracking observability
5. Ensuring backward compatibility for General domain

---

## Input Context Files

| File | Purpose |
|------|---------|
| `/home/cabdru/contextgraph/specs/tasks/threshold-inventory.yaml` | Original inventory |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/atc/` | All ATC modules |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/layers/` | Migrated layer thresholds |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/dream/` | Migrated dream thresholds |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/types/fingerprint/johari/` | Migrated Johari thresholds |
| `/home/cabdru/contextgraph/crates/context-graph-core/src/autonomous/` | Migrated autonomous thresholds |
| `/home/cabdru/contextgraph/tests/` | Existing test suites |

---

## Prerequisites

- [x] TASK-ATC-P2-003 completed (GWT thresholds)
- [x] TASK-ATC-P2-004 completed (Layer thresholds)
- [x] TASK-ATC-P2-005 completed (Dream thresholds)
- [x] TASK-ATC-P2-006 completed (Johari thresholds)
- [x] TASK-ATC-P2-007 completed (Autonomous thresholds)

---

## Scope

### In Scope

1. Full regression test suite execution
2. Integration tests for ATC with all migrated thresholds
3. Performance benchmarks for threshold retrieval
4. Domain behavior verification tests
5. EWMA observability validation
6. Backward compatibility verification
7. Documentation of test results

### Out of Scope

- New feature development
- Threshold value tuning (future task)
- MCP tool updates for threshold exposure

---

## Definition of Done

### Deliverables

1. **Integration Test File:** `/home/cabdru/contextgraph/tests/integration/atc_migration_tests.rs`

```rust
//! ATC Migration Integration Tests
//!
//! Validates the complete hardcoded threshold migration to ATC.
//! Part of TASK-ATC-P2-008.

use context_graph_core::{
    atc::{AdaptiveThresholdCalibration, Domain, ThresholdAccessor},
    layers::{GwtThresholds, LayerThresholds},
    dream::DreamThresholds,
    types::fingerprint::johari::JohariThresholds,
    autonomous::AutonomousThresholds,
};

mod domain_behavior {
    use super::*;

    #[test]
    fn test_domain_strictness_ordering() {
        let atc = AdaptiveThresholdCalibration::new();

        // Strictness order: Medical > Code > Legal > Research > General > Creative
        let medical = atc.get_threshold("theta_gate", Domain::Medical);
        let code = atc.get_threshold("theta_gate", Domain::Code);
        let creative = atc.get_threshold("theta_gate", Domain::Creative);

        assert!(medical > code);
        assert!(code > creative);
    }

    #[test]
    fn test_all_domains_valid() {
        let atc = AdaptiveThresholdCalibration::new();

        for domain in [
            Domain::Code,
            Domain::Medical,
            Domain::Legal,
            Domain::Creative,
            Domain::Research,
            Domain::General,
        ] {
            let thresholds = atc.get_domain_thresholds(domain).unwrap();
            assert!(thresholds.is_valid(), "Domain {:?} has invalid thresholds", domain);
        }
    }
}

mod backward_compatibility {
    use super::*;

    #[test]
    fn test_general_domain_matches_old_defaults() {
        let atc = AdaptiveThresholdCalibration::new();

        // GWT thresholds
        let gwt = GwtThresholds::from_atc(&atc, Domain::General);
        assert_eq!(gwt.gate, 0.70);
        assert_eq!(gwt.hypersync, 0.95);
        assert_eq!(gwt.fragmentation, 0.50);

        // Layer thresholds
        let layers = LayerThresholds::from_atc(&atc, Domain::General);
        assert_eq!(layers.memory_similarity, 0.50);
        assert_eq!(layers.reflex_hit, 0.85);
        assert_eq!(layers.consolidation, 0.10);

        // Dream thresholds
        let dream = DreamThresholds::from_atc(&atc, Domain::General);
        assert_eq!(dream.activity, 0.15);
        assert_eq!(dream.semantic_leap, 0.70);
        assert_eq!(dream.shortcut_confidence, 0.70);

        // Johari thresholds
        let johari = JohariThresholds::from_atc(&atc, Domain::General);
        assert_eq!(johari.entropy, 0.50);
        assert_eq!(johari.coherence, 0.50);

        // Autonomous thresholds
        let auto = AutonomousThresholds::from_atc(&atc, Domain::General);
        assert_eq!(auto.obsolescence_low, 0.30);
        assert_eq!(auto.obsolescence_mid, 0.60);
        assert_eq!(auto.obsolescence_high, 0.80);
    }
}

mod performance {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_threshold_retrieval_latency() {
        let atc = AdaptiveThresholdCalibration::new();
        let iterations = 10_000;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = atc.get_threshold("theta_gate", Domain::Code);
        }
        let elapsed = start.elapsed();

        let per_lookup_ns = elapsed.as_nanos() / iterations as u128;

        // NFR-ATC-001: Threshold retrieval MUST be < 1us
        assert!(
            per_lookup_ns < 1_000,
            "Threshold retrieval took {}ns, expected < 1000ns",
            per_lookup_ns
        );
    }
}

mod ewma_observability {
    use super::*;

    #[test]
    fn test_threshold_usage_observable() {
        let mut atc = AdaptiveThresholdCalibration::new();

        // Register and observe threshold
        atc.register_threshold("theta_gate", 0.75, 0.05, 0.2);
        atc.observe_threshold("theta_gate", 0.80);
        atc.observe_threshold("theta_gate", 0.82);

        // Drift should be non-zero after observations
        let drift = atc.get_drift_status();
        assert!(drift.get("theta_gate").is_some());
    }
}

mod integration_scenarios {
    use super::*;

    #[test]
    fn test_consciousness_with_medical_thresholds() {
        let atc = AdaptiveThresholdCalibration::new();
        let gwt = GwtThresholds::from_atc(&atc, Domain::Medical);

        // Medical domain should have stricter gate
        assert!(gwt.gate > 0.70);

        // At coherence 0.75, should NOT broadcast in Medical
        let coherence = 0.75;
        let should_broadcast = coherence >= gwt.gate;
        assert!(!should_broadcast);

        // But WOULD broadcast in Creative
        let creative_gwt = GwtThresholds::from_atc(&atc, Domain::Creative);
        let creative_broadcast = coherence >= creative_gwt.gate;
        assert!(creative_broadcast);
    }

    #[test]
    fn test_dream_behavior_by_domain() {
        let atc = AdaptiveThresholdCalibration::new();
        let creative = DreamThresholds::from_atc(&atc, Domain::Creative);
        let code = DreamThresholds::from_atc(&atc, Domain::Code);

        // Creative dreams more aggressively
        assert!(creative.activity < code.activity);
        assert!(creative.semantic_leap < code.semantic_leap);
    }
}
```

2. **Performance Benchmark:** `/home/cabdru/contextgraph/tests/benchmarks/atc_thresholds.rs`

```rust
//! ATC Threshold Retrieval Benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use context_graph_core::atc::{AdaptiveThresholdCalibration, Domain, ThresholdAccessor};

fn threshold_retrieval_benchmark(c: &mut Criterion) {
    let atc = AdaptiveThresholdCalibration::new();

    c.bench_function("get_threshold single", |b| {
        b.iter(|| {
            black_box(atc.get_threshold("theta_gate", Domain::Code))
        });
    });

    c.bench_function("get_threshold all names", |b| {
        b.iter(|| {
            for name in AdaptiveThresholdCalibration::list_threshold_names() {
                black_box(atc.get_threshold(name, Domain::General));
            }
        });
    });

    c.bench_function("get_threshold all domains", |b| {
        b.iter(|| {
            for domain in [Domain::Code, Domain::Medical, Domain::Legal,
                          Domain::Creative, Domain::Research, Domain::General] {
                black_box(atc.get_threshold("theta_gate", domain));
            }
        });
    });
}

criterion_group!(benches, threshold_retrieval_benchmark);
criterion_main!(benches);
```

3. **Validation Report:** `/home/cabdru/contextgraph/specs/tasks/ATC-MIGRATION-VALIDATION.md`

### Constraints

- MUST pass all existing tests (0 regressions)
- MUST meet NFR-ATC-001 performance budget (< 1us per retrieval)
- MUST verify all 6 domains behave correctly
- MUST document any deprecation warnings

### Verification

```bash
# Full regression suite
cargo test --all

# Specific ATC migration tests
cargo test --test atc_migration_tests

# Performance benchmarks
cargo bench --bench atc_thresholds

# Check for deprecation warnings
cargo build 2>&1 | grep -i "deprecated"
```

---

## Pseudo Code

```
FUNCTION validate_atc_migration():
    // 1. Run full regression suite
    result = cargo test --all
    ASSERT result.exit_code == 0
    RECORD test_count, pass_count, fail_count

    // 2. Run integration tests
    integration_result = cargo test --test atc_migration_tests
    ASSERT integration_result.exit_code == 0

    // 3. Run performance benchmarks
    bench_result = cargo bench --bench atc_thresholds
    EXTRACT per_lookup_ns from bench_result
    ASSERT per_lookup_ns < 1000  // < 1us

    // 4. Verify domain behavior
    FOR domain IN [Code, Medical, Legal, Creative, Research, General]:
        thresholds = atc.get_domain_thresholds(domain)
        ASSERT thresholds.is_valid()
        IF domain in [Code, Medical]:
            ASSERT thresholds stricter than General
        IF domain == Creative:
            ASSERT thresholds looser than General

    // 5. Check deprecation warnings
    build_output = cargo build 2>&1
    deprecated_lines = grep "deprecated" build_output
    RECORD deprecated_items

    // 6. Generate validation report
    WRITE ATC-MIGRATION-VALIDATION.md with:
        - Test results
        - Performance numbers
        - Domain behavior verification
        - Deprecated items list
        - Migration status: PASS/FAIL
```

---

## Files to Create

| Path | Description |
|------|-------------|
| `/home/cabdru/contextgraph/tests/integration/atc_migration_tests.rs` | Integration tests |
| `/home/cabdru/contextgraph/tests/benchmarks/atc_thresholds.rs` | Performance benchmarks |
| `/home/cabdru/contextgraph/specs/tasks/ATC-MIGRATION-VALIDATION.md` | Validation report |

---

## Files to Modify

| Path | Changes |
|------|---------|
| `/home/cabdru/contextgraph/tests/integration/mod.rs` | Add atc_migration_tests module |
| `/home/cabdru/contextgraph/Cargo.toml` | Add criterion dev-dependency if needed |

---

## Validation Criteria

| Criterion | Validation Method |
|-----------|-------------------|
| All tests pass | `cargo test --all` exit code 0 |
| No regressions | Compare test count before/after |
| Performance < 1us | Benchmark measurement |
| All domains valid | Integration test |
| Backward compatible | General domain test |
| EWMA observable | Integration test |
| Documentation complete | Validation report exists |

---

## Test Commands

```bash
# Full test suite
cargo test --all

# Integration tests only
cargo test --test atc_migration_tests

# Performance benchmarks
cargo bench --bench atc_thresholds

# Check deprecations
cargo build 2>&1 | grep -c "deprecated"

# Generate coverage
cargo llvm-cov --package context-graph-core --ignore-filename-regex "tests"
```

---

## Validation Report Template

```markdown
# ATC Migration Validation Report

**Date:** YYYY-MM-DD
**Version:** 1.0
**Task:** TASK-ATC-P2-008

## 1. Test Results

### Regression Suite
- Total Tests: XXX
- Passed: XXX
- Failed: 0
- Skipped: X

### Integration Tests
- Domain Behavior: PASS
- Backward Compatibility: PASS
- EWMA Observability: PASS

## 2. Performance Results

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Single lookup | < 1000ns | XXXns | PASS |
| All names lookup | < 20000ns | XXXXns | PASS |
| All domains lookup | < 6000ns | XXXXns | PASS |

## 3. Domain Behavior Verification

| Domain | Strictness | theta_gate | theta_memory_sim | Status |
|--------|------------|------------|------------------|--------|
| Medical | 1.0 | 0.XX | 0.XX | PASS |
| Code | 0.9 | 0.XX | 0.XX | PASS |
| Legal | 0.8 | 0.XX | 0.XX | PASS |
| Research | 0.5 | 0.XX | 0.XX | PASS |
| General | 0.5 | 0.70 | 0.50 | PASS |
| Creative | 0.2 | 0.XX | 0.XX | PASS |

## 4. Deprecated Items

| Item | Location | Replacement |
|------|----------|-------------|
| GW_THRESHOLD | coherence.rs:60 | GwtThresholds.gate |
| MIN_MEMORY_SIMILARITY | memory.rs:52 | LayerThresholds.memory_similarity |
| ... | ... | ... |

## 5. Migration Status

**OVERALL: PASS**

All hardcoded thresholds have been successfully migrated to the ATC system.
Domain-specific behavior is verified. Performance targets are met.

## 6. Recommendations

1. Remove deprecated constants in next major version
2. Consider adding per-embedder thresholds for Johari
3. Monitor drift patterns in production

---

Validated by: [Agent Name]
Date: YYYY-MM-DD
```

---

## Notes

- This is a validation task, not implementation
- All prior migrations must be complete before running
- Performance benchmarks require release mode for accurate results
- Deprecation warnings are expected and documented
- Final report serves as migration completion certificate

---

**Created:** 2026-01-11
**Author:** Specification Agent
