# Task Specification: Meta-UTL Core Types and Accuracy History Interface

**Task ID:** TASK-METAUTL-P0-001
**Version:** 1.0.0
**Status:** Ready
**Layer:** Foundation (Layer 1)
**Sequence:** 1
**Priority:** P0 (Critical)
**Estimated Complexity:** Medium

---

## 1. Metadata

### 1.1 Implements

| Requirement ID | Description |
|----------------|-------------|
| REQ-METAUTL-001 | Rolling accuracy history of at least 100 predictions |
| REQ-METAUTL-002 | Track accuracy per embedder (E1-E13) separately |
| REQ-METAUTL-006 | Lambda weights SHALL always sum to 1.0 |
| REQ-METAUTL-007 | Lambda weights SHALL be clamped to [0.1, 0.9] |

### 1.2 Dependencies

| Task ID | Description | Status |
|---------|-------------|--------|
| None | This is the foundation task | N/A |

### 1.3 Blocked By

None - this is the first task in the sequence.

---

## 2. Context

This task establishes the foundational types and data structures for the Meta-UTL self-correction protocol. These types will be used by all subsequent tasks in the implementation.

The current codebase has:
- `LifecycleLambdaWeights` in `crates/context-graph-utl/src/lifecycle/lambda.rs` - Fixed by stage
- `MetaCognitiveLoop` in `crates/context-graph-core/src/gwt/meta_cognitive.rs` - No lambda adjustment
- No accuracy history tracking

---

## 3. Input Context Files

| File | Purpose |
|------|---------|
| `crates/context-graph-utl/src/lifecycle/lambda.rs` | Existing lambda weights implementation |
| `crates/context-graph-utl/src/lifecycle/stage.rs` | Lifecycle stage definitions |
| `crates/context-graph-core/src/types/utl.rs` | Core UTL types |
| `crates/context-graph-core/src/gwt/meta_cognitive.rs` | MetaCognitiveLoop for integration patterns |
| `docs2/constitution.yaml` | Authoritative constraints |
| `specs/functional/SPEC-METAUTL-001.md` | Functional specification |

---

## 4. Scope

### 4.1 In Scope

- Create `MetaAccuracyHistory` struct with FIFO buffer
- Create `EmbedderAccuracyTracker` with 13-element array
- Create `LambdaAdjustment` result struct
- Create `MetaLearningEvent` struct for event logging
- Create `MetaLearningEventType` enum
- Create `SelfCorrectionConfig` configuration struct
- Create `SelfCorrectionState` state container
- Create `Domain` enum for domain-specific tracking
- Add necessary traits for serialization

### 4.2 Out of Scope

- Lambda adjustment logic (TASK-METAUTL-P0-002)
- Escalation logic (TASK-METAUTL-P0-003)
- Event logging implementation (TASK-METAUTL-P0-004)
- MCP tool wiring (TASK-METAUTL-P0-005)
- Integration with MetaCognitiveLoop (TASK-METAUTL-P0-006)

---

## 5. Prerequisites

| Check | Description |
|-------|-------------|
| [x] | Rust toolchain installed (1.75+) |
| [x] | Workspace compiles successfully |
| [x] | chrono crate available in workspace |
| [x] | serde crate available in workspace |

---

## 6. Definition of Done

### 6.1 Required Signatures

#### File: `crates/context-graph-utl/src/meta/types.rs`

```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Number of embedders in the teleological fingerprint
pub const NUM_EMBEDDERS: usize = 13;

/// Default accuracy history buffer size
pub const DEFAULT_HISTORY_SIZE: usize = 100;

/// Domain types for domain-specific accuracy tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Domain {
    Code,
    Medical,
    Legal,
    Creative,
    Research,
    General,
}

/// Rolling accuracy history for self-correction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaAccuracyHistory {
    /// Rolling buffer of accuracy values [0.0, 1.0]
    history: VecDeque<f32>,
    /// Maximum history size
    max_size: usize,
    /// Consecutive low accuracy count (< 0.7)
    consecutive_low_count: u32,
    /// Last update timestamp
    last_update: DateTime<Utc>,
}

impl MetaAccuracyHistory {
    /// Create new accuracy history with specified max size
    pub fn new(max_size: usize) -> Self;

    /// Create with default size (100)
    pub fn with_defaults() -> Self;

    /// Record a new accuracy value
    /// Returns true if value was below escalation threshold
    pub fn record(&mut self, accuracy: f32) -> bool;

    /// Get rolling average accuracy
    pub fn rolling_average(&self) -> f32;

    /// Get consecutive low count
    pub fn consecutive_low_count(&self) -> u32;

    /// Reset consecutive low count
    pub fn reset_consecutive_low(&mut self);

    /// Get history length
    pub fn len(&self) -> usize;

    /// Check if history is empty
    pub fn is_empty(&self) -> bool;

    /// Get last update timestamp
    pub fn last_update(&self) -> DateTime<Utc>;

    /// Clear history
    pub fn clear(&mut self);
}

/// Per-embedder accuracy tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderAccuracyTracker {
    /// Accuracy history for each of 13 embedders
    embedder_history: [MetaAccuracyHistory; NUM_EMBEDDERS],
    /// Per-domain accuracy
    domain_accuracy: HashMap<Domain, MetaAccuracyHistory>,
    /// Global accuracy (weighted average)
    global_accuracy: MetaAccuracyHistory,
}

impl EmbedderAccuracyTracker {
    /// Create new tracker with default history sizes
    pub fn new() -> Self;

    /// Record accuracy for a specific embedder
    pub fn record_embedder_accuracy(&mut self, embedder_idx: usize, accuracy: f32);

    /// Record accuracy for a domain
    pub fn record_domain_accuracy(&mut self, domain: Domain, accuracy: f32);

    /// Record global accuracy
    pub fn record_global_accuracy(&mut self, accuracy: f32);

    /// Get embedder rolling average
    pub fn embedder_average(&self, embedder_idx: usize) -> f32;

    /// Get domain rolling average
    pub fn domain_average(&self, domain: Domain) -> f32;

    /// Get global rolling average
    pub fn global_average(&self) -> f32;

    /// Check if any embedder needs escalation
    pub fn any_needs_escalation(&self) -> bool;

    /// Get embedder with lowest accuracy
    pub fn lowest_accuracy_embedder(&self) -> (usize, f32);
}

impl Default for EmbedderAccuracyTracker {
    fn default() -> Self;
}

/// Lambda adjustment result
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LambdaAdjustment {
    /// Change to lambda_s (surprise weight)
    pub delta_lambda_s: f32,
    /// Change to lambda_c (coherence weight)
    pub delta_lambda_c: f32,
    /// Learning rate used
    pub alpha: f32,
    /// Prediction error that triggered adjustment
    pub trigger_error: f32,
}

impl LambdaAdjustment {
    /// Create new adjustment
    pub fn new(delta_s: f32, delta_c: f32, alpha: f32, error: f32) -> Self;

    /// Check if adjustment is significant
    pub fn is_significant(&self) -> bool;
}

/// Meta-learning event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetaLearningEventType {
    /// Lambda adjustment triggered by prediction error
    LambdaAdjustment,
    /// Escalation to Bayesian optimization
    BayesianEscalation,
    /// Accuracy threshold crossed
    AccuracyAlert,
    /// Self-healing intervention
    SelfHealing,
    /// Human escalation requested
    HumanEscalation,
}

/// Meta-learning event for introspection log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event type
    pub event_type: MetaLearningEventType,
    /// Prediction error that triggered event
    pub prediction_error: f32,
    /// Lambda values before change
    pub lambda_before: (f32, f32),
    /// Lambda values after change
    pub lambda_after: (f32, f32),
    /// Current accuracy rolling average
    pub accuracy_avg: f32,
    /// Whether escalation was triggered
    pub escalated: bool,
    /// Domain context (if applicable)
    pub domain: Option<Domain>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl MetaLearningEvent {
    /// Create new lambda adjustment event
    pub fn lambda_adjustment(
        error: f32,
        before: (f32, f32),
        after: (f32, f32),
        accuracy: f32,
        domain: Option<Domain>,
    ) -> Self;

    /// Create new escalation event
    pub fn escalation(accuracy: f32, domain: Option<Domain>) -> Self;

    /// Create new accuracy alert event
    pub fn accuracy_alert(accuracy: f32, threshold: f32, domain: Option<Domain>) -> Self;
}

/// Self-correction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfCorrectionConfig {
    /// Enable/disable self-correction
    pub enabled: bool,
    /// Prediction error threshold to trigger adjustment
    pub error_threshold: f32,
    /// Base learning rate for lambda adjustment
    pub base_alpha: f32,
    /// Accuracy threshold for escalation
    pub escalation_accuracy_threshold: f32,
    /// Consecutive cycles below threshold for escalation
    pub escalation_cycle_count: u32,
    /// Maximum history size per embedder
    pub max_history_size: usize,
    /// Lambda bounds [min, max]
    pub lambda_min: f32,
    pub lambda_max: f32,
    /// Event log retention days
    pub event_retention_days: u32,
}

impl Default for SelfCorrectionConfig {
    fn default() -> Self;
}

impl SelfCorrectionConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String>;
}

/// Escalation status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EscalationStatus {
    /// No escalation needed
    None,
    /// Escalation pending
    Pending,
    /// Escalation in progress
    InProgress,
    /// Escalation completed successfully
    Completed,
    /// Escalation failed, human review needed
    Failed,
}
```

### 6.2 Constraints

- All structs MUST derive `Debug`, `Clone`, `Serialize`, `Deserialize`
- `MetaAccuracyHistory::record` MUST maintain FIFO eviction when over max_size
- `EmbedderAccuracyTracker` MUST have exactly 13 embedder histories
- `SelfCorrectionConfig::default()` values MUST match constitution.yaml
- NO `unwrap()` in library code - use `expect()` with context or return `Result`
- All accuracy values MUST be clamped to [0.0, 1.0]
- All timestamps MUST use `chrono::Utc`

### 6.3 Verification Commands

```bash
# Type check
cargo check -p context-graph-utl

# Unit tests
cargo test -p context-graph-utl meta::types

# Documentation
cargo doc -p context-graph-utl --no-deps
```

---

## 7. Files to Create

| Path | Description |
|------|-------------|
| `crates/context-graph-utl/src/meta/mod.rs` | Module declaration |
| `crates/context-graph-utl/src/meta/types.rs` | Core types as specified above |
| `crates/context-graph-utl/src/meta/tests.rs` | Unit tests for types |

---

## 8. Files to Modify

| Path | Modification |
|------|--------------|
| `crates/context-graph-utl/src/lib.rs` | Add `pub mod meta;` and re-exports |
| `crates/context-graph-utl/Cargo.toml` | Ensure `chrono`, `serde`, `serde_json` dependencies |

---

## 9. Pseudo-Code

### 9.1 MetaAccuracyHistory::record

```
FUNCTION record(accuracy: f32) -> bool:
    // Clamp input
    accuracy = clamp(accuracy, 0.0, 1.0)

    // Add to history
    self.history.push_back(accuracy)

    // FIFO eviction
    WHILE self.history.len() > self.max_size:
        self.history.pop_front()

    // Update timestamp
    self.last_update = Utc::now()

    // Track consecutive low
    IF accuracy < ESCALATION_THRESHOLD (0.7):
        self.consecutive_low_count += 1
        RETURN true
    ELSE:
        self.consecutive_low_count = 0
        RETURN false
```

### 9.2 EmbedderAccuracyTracker::new

```
FUNCTION new() -> Self:
    // Initialize 13 embedder histories
    embedder_history = []
    FOR i in 0..NUM_EMBEDDERS:
        embedder_history.push(MetaAccuracyHistory::with_defaults())

    // Initialize domain map (empty, lazy population)
    domain_accuracy = HashMap::new()

    // Initialize global tracker
    global_accuracy = MetaAccuracyHistory::with_defaults()

    RETURN Self { embedder_history, domain_accuracy, global_accuracy }
```

### 9.3 SelfCorrectionConfig::default

```
FUNCTION default() -> Self:
    Self {
        enabled: true,
        error_threshold: 0.2,           // constitution: correction.threshold
        base_alpha: 0.05,               // conservative learning rate
        escalation_accuracy_threshold: 0.7,  // constitution: accuracy<0.7
        escalation_cycle_count: 10,     // 10 consecutive cycles
        max_history_size: 100,          // REQ-METAUTL-001
        lambda_min: 0.1,                // REQ-METAUTL-007
        lambda_max: 0.9,                // REQ-METAUTL-007
        event_retention_days: 7,
    }
```

---

## 10. Validation Criteria

| Criterion | Validation Method |
|-----------|-------------------|
| All types compile without errors | `cargo check -p context-graph-utl` |
| Unit tests pass | `cargo test -p context-graph-utl meta::types` |
| Documentation generates | `cargo doc -p context-graph-utl --no-deps` |
| Serde roundtrip works | Test JSON serialize/deserialize |
| Accuracy clamping works | Test with values outside [0, 1] |
| FIFO eviction works | Add 101 values, verify len=100 |
| Default config matches constitution | Assert field values |
| No clippy warnings | `cargo clippy -p context-graph-utl` |

---

## 11. Test Cases

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_history_fifo() {
        let mut history = MetaAccuracyHistory::new(10);
        for i in 0..15 {
            history.record(i as f32 / 15.0);
        }
        assert_eq!(history.len(), 10);
    }

    #[test]
    fn test_accuracy_history_consecutive_low() {
        let mut history = MetaAccuracyHistory::with_defaults();
        for _ in 0..5 {
            history.record(0.5); // Below 0.7
        }
        assert_eq!(history.consecutive_low_count(), 5);

        history.record(0.8); // Above 0.7
        assert_eq!(history.consecutive_low_count(), 0);
    }

    #[test]
    fn test_accuracy_clamping() {
        let mut history = MetaAccuracyHistory::with_defaults();
        history.record(1.5); // Should clamp to 1.0
        history.record(-0.5); // Should clamp to 0.0
        assert!(history.rolling_average() >= 0.0);
        assert!(history.rolling_average() <= 1.0);
    }

    #[test]
    fn test_embedder_tracker_13_embedders() {
        let tracker = EmbedderAccuracyTracker::new();
        // Should have exactly 13 embedders
        for i in 0..13 {
            assert!(tracker.embedder_average(i) >= 0.0 || tracker.embedder_average(i).is_nan());
        }
    }

    #[test]
    fn test_config_defaults() {
        let config = SelfCorrectionConfig::default();
        assert_eq!(config.error_threshold, 0.2);
        assert_eq!(config.escalation_accuracy_threshold, 0.7);
        assert_eq!(config.escalation_cycle_count, 10);
        assert_eq!(config.lambda_min, 0.1);
        assert_eq!(config.lambda_max, 0.9);
    }

    #[test]
    fn test_lambda_adjustment_serde() {
        let adj = LambdaAdjustment::new(0.05, -0.05, 0.02, 0.25);
        let json = serde_json::to_string(&adj).unwrap();
        let parsed: LambdaAdjustment = serde_json::from_str(&json).unwrap();
        assert_eq!(adj.delta_lambda_s, parsed.delta_lambda_s);
    }

    #[test]
    fn test_meta_learning_event_creation() {
        let event = MetaLearningEvent::lambda_adjustment(
            0.25,
            (0.7, 0.3),
            (0.65, 0.35),
            0.75,
            Some(Domain::Code),
        );
        assert_eq!(event.event_type, MetaLearningEventType::LambdaAdjustment);
        assert_eq!(event.lambda_before, (0.7, 0.3));
        assert_eq!(event.lambda_after, (0.65, 0.35));
    }

    #[test]
    fn test_domain_enum_serde() {
        let domain = Domain::Code;
        let json = serde_json::to_string(&domain).unwrap();
        assert_eq!(json, "\"code\"");
        let parsed: Domain = serde_json::from_str(&json).unwrap();
        assert_eq!(domain, parsed);
    }
}
```

---

## 12. Rollback Plan

If this task fails validation:

1. Revert files: `git checkout -- crates/context-graph-utl/src/meta/`
2. Remove mod declaration from `lib.rs`
3. Document failure reason in task notes
4. Create follow-up task addressing issues

---

## 13. Notes

- This task creates NO external dependencies beyond existing workspace crates
- Types are designed for zero-copy serialization where possible
- `Domain` enum matches constitution.yaml domain categories
- History size of 100 balances memory vs statistical significance
- Escalation threshold of 10 consecutive cycles prevents oscillation

---

**Task History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | ContextGraph Team | Initial task specification |
