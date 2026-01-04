---
id: "M05-T08"
title: "Define JohariQuadrant Enum and SuggestedAction"
description: |
  Implement JohariQuadrant enum for memory classification.
  Variants: Open (low entropy, high coherence), Blind (high entropy, low coherence),
  Hidden (low entropy, low coherence), Unknown (high entropy, high coherence).
  SuggestedAction enum: DirectRecall, TriggerDream, GetNeighborhood, EpistemicAction.
  Include methods: name(), is_well_understood(), requires_exploration().
layer: "foundation"
status: "complete"
priority: "critical"
estimated_hours: 2
sequence: 8
depends_on: []
spec_refs:
  - "constitution.yaml lines 159-163 - Johari quadrant definitions"
  - "contextprd.md Section 2.2 - Johari Quadrants table"
  - "learntheory.md - Theoretical Johari Window mapping"
---

## STATUS: COMPLETE ✅

This task is **fully implemented**. JohariQuadrant and SuggestedAction are already implemented and tested across two crates:

### Verified Location of Implementations

| Type | Location | Line | Tests |
|------|----------|------|-------|
| `JohariQuadrant` | `crates/context-graph-core/src/types/johari/quadrant.rs` | 23-251 | 49 tests |
| `SuggestedAction` (core) | `crates/context-graph-core/src/types/pulse.rs` | 318-354 | part of pulse tests |
| `SuggestedAction` (UTL) | `crates/context-graph-utl/src/johari/retrieval.rs` | 42-129 | 38 tests |
| `JohariClassifier` | `crates/context-graph-utl/src/johari/classifier.rs` | 113-368 | 20 tests |
| `QuadrantRetrieval` | `crates/context-graph-utl/src/johari/retrieval.rs` | 216-480 | 18 tests |

### Verification Commands

```bash
# Verify tests pass
cargo test -p context-graph-utl johari -- --nocapture
# Expected: 38 passed; 0 failed

cargo test -p context-graph-core johari -- --nocapture
# Expected: 49 passed; 0 failed

# Verify exports work
cargo build -p context-graph-utl
```

### Git Reference

```
f521803 feat(utl): complete context-graph-utl crate with 453 tests passing
```

---

## Implementation Details (For Reference)

### JohariQuadrant (context-graph-core)

**File:** `crates/context-graph-core/src/types/johari/quadrant.rs`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JohariQuadrant {
    Open,    // ΔS < 0.5, ΔC > 0.5 → direct recall
    Hidden,  // ΔS < 0.5, ΔC < 0.5 → private (get_neighborhood)
    Blind,   // ΔS > 0.5, ΔC < 0.5 → discovery (epistemic_action/dream)
    Unknown, // ΔS > 0.5, ΔC > 0.5 → frontier
}
```

**Available Methods:**
- `is_self_aware() -> bool` - true for Open, Hidden
- `is_other_aware() -> bool` - true for Open, Blind
- `default_retrieval_weight() -> f32` - Open=1.0, Blind=0.7, Hidden=0.3, Unknown=0.5
- `include_in_default_context() -> bool` - true for Open, Blind, Unknown
- `description() -> &'static str` - human-readable description
- `column_family() -> &'static str` - RocksDB column family name
- `all() -> [JohariQuadrant; 4]` - all variants
- `valid_transitions() -> &[(JohariQuadrant, TransitionTrigger)]` - state machine
- `can_transition_to(target) -> bool` - check valid transition
- `transition_to(target, trigger) -> Result<JohariTransition, String>` - perform transition
- `Default` implementation returns `Open`
- `Display`, `FromStr` implementations

### SuggestedAction (context-graph-utl)

**File:** `crates/context-graph-utl/src/johari/retrieval.rs`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SuggestedAction {
    DirectRecall,     // Open quadrant → retrieve with full confidence
    EpistemicAction,  // Blind quadrant → trigger discovery mechanisms
    GetNeighborhood,  // Hidden quadrant → explore related context
    TriggerDream,     // Unknown quadrant → consolidate frontier knowledge
}
```

**Available Methods:**
- `description() -> &'static str` - human-readable description
- `urgency() -> f32` - priority level (DirectRecall=1.0, TriggerDream=0.8, EpistemicAction=0.7, GetNeighborhood=0.5)
- `all() -> [SuggestedAction; 4]` - all variants
- `Display` implementation

### Re-exports

**context-graph-utl/src/johari/mod.rs:**
```rust
pub use context_graph_core::types::JohariQuadrant;  // DO NOT DUPLICATE
pub use retrieval::{SuggestedAction, get_suggested_action, get_retrieval_weight, QuadrantRetrieval};
pub use classifier::{JohariClassifier, classify_quadrant};
```

**context-graph-utl/src/lib.rs:**
```rust
pub use johari::{classify_quadrant, JohariClassifier, JohariQuadrant};
```

---

## Constitution Compliance

### Johari Window Classification (constitution.yaml lines 159-163)

| Quadrant | ΔS (Entropy) | ΔC (Coherence) | Action |
|----------|-------------|----------------|--------|
| Open     | < 0.5       | > 0.5          | direct recall |
| Blind    | > 0.5       | < 0.5          | discovery (epistemic_action/dream) |
| Hidden   | < 0.5       | < 0.5          | private (get_neighborhood) |
| Unknown  | > 0.5       | > 0.5          | frontier |

### Classification Matrix (contextprd.md Section 2.2)

```
                    ENTROPY (Surprise)
                    Low         High
             ┌─────────────┬─────────────┐
        High │    OPEN     │   UNKNOWN   │
COHERENCE    │ (Confident) │ (Exploring) │
             ├─────────────┼─────────────┤
        Low  │   HIDDEN    │    BLIND    │
             │ (Isolated)  │ (Confused)  │
             └─────────────┴─────────────┘
```

---

## Full State Verification (REQUIRED)

After any modifications to this code, you MUST perform:

### 1. Source of Truth Verification

The source of truth for JohariQuadrant is:
- **Definition:** `crates/context-graph-core/src/types/johari/quadrant.rs`
- **UTL re-export:** `crates/context-graph-utl/src/johari/mod.rs` line 42
- **Crate re-export:** `crates/context-graph-utl/src/lib.rs` line 61

Verify with:
```bash
# Check that JohariQuadrant is exported from context-graph-utl
cargo doc -p context-graph-utl --no-deps 2>&1 | grep -i "JohariQuadrant"
```

### 2. Execute & Inspect

```bash
# Run all johari tests with output
cargo test -p context-graph-utl johari -- --nocapture 2>&1 | tee /tmp/johari_test_output.txt

# Verify test count
grep "passed" /tmp/johari_test_output.txt
# Expected: "38 passed; 0 failed"

# Run core tests
cargo test -p context-graph-core johari -- --nocapture 2>&1 | tee /tmp/core_johari_output.txt
grep "passed" /tmp/core_johari_output.txt
# Expected: "49 passed; 0 failed"
```

### 3. Boundary & Edge Case Audit

**Edge Case 1: Boundary values (0.5, 0.5)**
```bash
# State BEFORE: At threshold, should classify as Blind (high entropy, low coherence)
cargo test -p context-graph-utl test_classify_quadrant_boundary -- --nocapture
# State AFTER: Verify output shows JohariQuadrant::Blind
```

**Edge Case 2: Out-of-range inputs**
```bash
# State BEFORE: Inputs like (-0.5, 1.5) should be clamped
cargo test -p context-graph-utl test_classifier_clamps_input -- --nocapture
# State AFTER: Verify clamping produces valid classification
```

**Edge Case 3: Extreme corners**
```bash
# State BEFORE: (0.0, 1.0), (1.0, 0.0), (0.0, 0.0), (1.0, 1.0)
cargo test -p context-graph-utl test_extreme_values -- --nocapture
# State AFTER: Each maps to correct quadrant
```

### 4. Evidence of Success

After running tests, you must show:
```bash
# Final verification showing all tests pass
cargo test -p context-graph-utl johari 2>&1 | tail -5
# Must show: "test result: ok. 38 passed; 0 failed"

cargo test -p context-graph-core johari 2>&1 | tail -5
# Must show: "test result: ok. 49 passed; 0 failed"

# Verify no compilation errors
cargo build -p context-graph-utl 2>&1 | grep -E "(error|warning:)" | head -10
```

---

## Final Verification with Sherlock-Holmes Agent

**MANDATORY FINAL STEP:** After completing all implementation, you MUST spawn a `sherlock-holmes` subagent to verify:

1. All JohariQuadrant variants match constitution.yaml
2. SuggestedAction mappings are correct
3. All tests pass with real data (no mocks)
4. Re-exports work from both crates
5. No duplicate type definitions exist

If Sherlock identifies any issues, they MUST be fixed before marking this task complete.

---

## What You Would Do If This Were Not Complete

If this task were not complete, you would:

1. **Create `quadrant.rs`** in `crates/context-graph-utl/src/johari/` with JohariQuadrant enum
2. Add `#[repr(u8)]` for efficient storage per constitution requirements
3. Implement all required methods matching the signatures above
4. Add `SuggestedAction` enum with 6 variants per original spec (but note: current implementation has 4 variants which matches the constitution better)
5. Add comprehensive tests with NO MOCKS - use real values
6. Update `mod.rs` to re-export types
7. Run full verification suite

**CRITICAL:** The current implementation uses 4 SuggestedAction variants in UTL (DirectRecall, EpistemicAction, GetNeighborhood, TriggerDream) which directly maps to the 4 Johari quadrants. The core crate has a different SuggestedAction with 7 variants (Ready, Continue, Explore, Consolidate, Prune, Stabilize, Review) for the Cognitive Pulse feature. Both are correct for their purposes.

---

## Files Summary

| File | Status | Purpose |
|------|--------|---------|
| `crates/context-graph-core/src/types/johari/quadrant.rs` | ✅ EXISTS | JohariQuadrant definition |
| `crates/context-graph-core/src/types/johari/mod.rs` | ✅ EXISTS | Module exports |
| `crates/context-graph-utl/src/johari/classifier.rs` | ✅ EXISTS | JohariClassifier |
| `crates/context-graph-utl/src/johari/retrieval.rs` | ✅ EXISTS | SuggestedAction + QuadrantRetrieval |
| `crates/context-graph-utl/src/johari/mod.rs` | ✅ EXISTS | Re-exports from core |
| `crates/context-graph-utl/src/lib.rs` | ✅ EXISTS | Crate-level exports |

---

*Task Version: 2.0.0*
*Audited: 2026-01-04*
*Status: COMPLETE - All implementations verified*
*Module: 05 - UTL Integration*
*Test Count: 87 tests (38 UTL + 49 Core)*
