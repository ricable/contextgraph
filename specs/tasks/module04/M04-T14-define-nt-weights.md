---
id: "M04-T14"
title: "Integrate NeurotransmitterWeights with Graph Layer"
description: |
  NeurotransmitterWeights and Domain ALREADY EXIST in context-graph-core.
  This task verifies integration with context-graph-graph and documents the API.
  NO NEW TYPES TO CREATE - verify re-exports and write integration tests.

  CANONICAL FORMULA (from context-graph-core):
  w_eff = ((base * excitatory - base * inhibitory) * (1 + (modulatory - 0.5) * 0.4)).clamp(0.0, 1.0)
layer: "logic"
status: "complete"
priority: "high"
estimated_hours: 1
sequence: 19
depends_on: []
spec_refs:
  - "TECH-GRAPH-004 Section 4.1"
  - "REQ-KG-065"
  - "constitution.yaml edge_model.nt_weights"
files_to_create: []
files_to_modify: []
test_file: "crates/context-graph-graph/tests/nt_integration_tests.rs"
completed: "2026-01-03"
commit: "a4dbcd9"
---

## ✅ TASK STATUS: COMPLETE

**Completed**: 2026-01-03
**Commit**: a4dbcd9
**Verified by**: sherlock-holmes subagent

### Implementation Summary

| Component | Location | Status |
|-----------|----------|--------|
| `NeurotransmitterWeights` | `context-graph-core/src/marblestone/neurotransmitter_weights.rs` | ✅ EXISTS |
| `Domain` | `context-graph-core/src/marblestone/domain.rs` | ✅ EXISTS |
| Re-export | `context-graph-graph/src/lib.rs:60` | ✅ VERIFIED |
| Marblestone module | `context-graph-graph/src/marblestone/mod.rs` | ✅ VERIFIED |
| Integration tests | `crates/context-graph-graph/tests/nt_integration_tests.rs` | ✅ 22 TESTS PASS |

---

## Context

The Marblestone-inspired neurotransmitter system exists in `context-graph-core`. This task verified integration with `context-graph-graph` through re-exports.

### Constitution Reference
- `edge_model.nt_weights`: Defines weight structure and domain profiles
- `AP-009`: All weights must be in [0.0, 1.0]

---

## Existing API (REFERENCE ONLY)

```rust
// In context-graph-core/src/marblestone/neurotransmitter_weights.rs
pub struct NeurotransmitterWeights {
    pub excitatory: f32,  // [0.0, 1.0]
    pub inhibitory: f32,  // [0.0, 1.0]
    pub modulatory: f32,  // [0.0, 1.0]
}

impl NeurotransmitterWeights {
    pub fn new(excitatory: f32, inhibitory: f32, modulatory: f32) -> Self;
    pub fn for_domain(domain: Domain) -> Self;
    pub fn compute_effective_weight(&self, base_weight: f32) -> f32;
    pub fn validate(&self) -> bool;  // Returns bool, NOT Result
}

// In context-graph-core/src/marblestone/domain.rs
pub enum Domain { Code, Legal, Medical, Creative, Research, General }
```

### Domain-Specific Profiles

| Domain | excitatory | inhibitory | modulatory |
|--------|------------|------------|------------|
| Code | 0.6 | 0.3 | 0.4 |
| Legal | 0.4 | 0.4 | 0.2 |
| Medical | 0.5 | 0.3 | 0.5 |
| Creative | 0.8 | 0.1 | 0.6 |
| Research | 0.6 | 0.2 | 0.5 |
| General | 0.5 | 0.2 | 0.3 |

---

## Verification Evidence

### Re-export Verification
```bash
$ grep -n "pub use context_graph_core::marblestone" crates/context-graph-graph/src/lib.rs
60:pub use context_graph_core::marblestone::{Domain, EdgeType, NeurotransmitterWeights};
```

### Test Results
```bash
$ cargo test -p context-graph-graph nt_integration
running 22 tests
test test_code_domain_values ... ok
test test_creative_domain_values ... ok
test test_general_domain_values ... ok
test test_legal_domain_values ... ok
test test_medical_domain_values ... ok
test test_research_domain_values ... ok
[...more tests...]
test result: ok. 22 passed; 0 failed
```

### Formula Verification (General Domain)
```
Input: base_weight = 1.0, Domain::General (e=0.5, i=0.2, m=0.3)
Formula: ((1.0*0.5 - 1.0*0.2) * (1 + (0.3-0.5)*0.4)).clamp(0,1)
       = (0.3 * 0.92).clamp(0,1)
       = 0.276
Result: VERIFIED ✅
```

---

## Definition of Done (ALL MET)

### Acceptance Criteria
- [x] `use context_graph_graph::NeurotransmitterWeights;` compiles
- [x] `use context_graph_graph::Domain;` compiles
- [x] `NeurotransmitterWeights::for_domain(Domain::Code)` returns correct values
- [x] `weights.compute_effective_weight(1.0)` computes correctly
- [x] `weights.validate()` returns true for valid weights
- [x] Integration tests pass in context-graph-graph
- [x] No clippy warnings

### Verification Commands (All Pass)
```bash
cargo build -p context-graph-graph  # ✅ PASS
cargo test -p context-graph-graph nt_integration  # ✅ 22/22 PASS
cargo clippy -p context-graph-graph -- -D warnings  # ✅ NO WARNINGS
```

---

## Related Tasks
- M04-T14a: Result-returning validation wrapper (COMPLETE)
- M04-T15: Integrate NT weights into GraphEdge struct (NEXT)
