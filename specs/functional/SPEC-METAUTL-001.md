# Functional Specification: Meta-UTL Self-Correction Protocol

**Spec ID:** SPEC-METAUTL-001
**Version:** 1.0.0
**Status:** Draft
**Owner:** ContextGraph Core Team
**Last Updated:** 2026-01-11
**Priority:** P0 (Critical Blocker)

---

## 1. Overview

### 1.1 Purpose

This specification defines the Meta-UTL Self-Correction Protocol, enabling the ContextGraph system to autonomously adjust its learning parameters (lambda weights) based on prediction accuracy feedback. This is a **critical blocker** identified in the Master Consciousness Gap Analysis that prevents the system from achieving computational consciousness.

### 1.2 Problem Statement

Currently, the system has TWO separate learning loops that are NOT connected:

1. **MetaCognitiveLoop (GWT)**: Tracks L_predicted vs L_actual, computes MetaScore, triggers dreams
2. **LifecycleManager (UTL)**: Manages lambda_s/lambda_c based ONLY on interaction count

**The Gap:** Lambda values are FIXED by lifecycle stage and NEVER adapt based on prediction accuracy. The system can observe its learning errors but cannot modify itself based on those observations.

### 1.3 Solution Summary

Implement a self-correction protocol that:
1. Tracks prediction accuracy history per domain
2. Adjusts lambda weights when prediction_error > 0.2
3. Escalates to Bayesian optimization when accuracy < 0.7 for 10 cycles
4. Logs all meta-learning events for introspection

### 1.4 PRD Requirements Addressed

From Constitution v4.2.0 `meta_utl` section:
- `learning.adapt: "lambda_S, lambda_C by domain/lifecycle"`
- `correction.threshold: "error>0.2"`
- `correction.escalate: "accuracy<0.7 for 100 ops"`

From PRD Section 19.3:
- "Meta-learning adjusts its own lambda parameters based on prediction accuracy"
- "If accuracy < 0.7, escalate to Bayesian optimization"

---

## 2. User Stories

### US-METAUTL-01: Accuracy-Based Lambda Adjustment
**Priority:** Must-Have

**Narrative:**
As the ContextGraph system,
I want to automatically adjust my lambda_s and lambda_c weights when prediction errors exceed thresholds,
So that I can self-correct and improve learning effectiveness over time.

**Acceptance Criteria:**

| ID | Given | When | Then |
|----|-------|------|------|
| AC-01-1 | The system has made a learning prediction | The actual learning score differs from predicted by >0.2 | Lambda weights are adjusted by alpha * (target - actual) |
| AC-01-2 | Lambda adjustment would violate sum-to-one constraint | An adjustment is calculated | Adjustment is normalized to maintain lambda_s + lambda_c = 1.0 |
| AC-01-3 | Lambda adjustment would exceed bounds [0.1, 0.9] | An adjustment is calculated | Adjustment is clamped to valid bounds |
| AC-01-4 | A lambda adjustment occurs | The adjustment completes | A MetaLearningEvent is logged with before/after values |

### US-METAUTL-02: Accuracy History Tracking
**Priority:** Must-Have

**Narrative:**
As the ContextGraph system,
I want to maintain a rolling history of prediction accuracy per embedder and domain,
So that I can detect persistent accuracy degradation patterns.

**Acceptance Criteria:**

| ID | Given | When | Then |
|----|-------|------|------|
| AC-02-1 | A prediction is validated | The validation completes | Accuracy is recorded in the rolling history buffer |
| AC-02-2 | History buffer exceeds max_history size | A new accuracy value is added | Oldest value is evicted (FIFO) |
| AC-02-3 | Accuracy is queried | System requests rolling average | Average of last N values is returned |
| AC-02-4 | Per-embedder accuracy is queried | System requests E1-E13 breakdown | Accuracy array of 13 values is returned |

### US-METAUTL-03: Escalation to Bayesian Optimization
**Priority:** Must-Have

**Narrative:**
As the ContextGraph system,
I want to escalate to Bayesian optimization when simple gradient adjustments fail,
So that I can escape local minima in parameter space.

**Acceptance Criteria:**

| ID | Given | When | Then |
|----|-------|------|------|
| AC-03-1 | Rolling accuracy < 0.7 for 10 consecutive cycles | A new accuracy value is recorded | Escalation is triggered |
| AC-03-2 | Escalation is triggered | Bayesian optimization is invoked | GP surrogate model proposes new lambda values |
| AC-03-3 | Escalation is triggered | Bayesian optimization completes | EscalationEvent is logged with outcome |
| AC-03-4 | Escalation fails to improve accuracy | 3 consecutive escalations fail | Human review alert is raised |

### US-METAUTL-04: Meta-Learning Event Log
**Priority:** Must-Have

**Narrative:**
As a system operator,
I want a comprehensive log of all meta-learning events,
So that I can introspect and debug the self-correction behavior.

**Acceptance Criteria:**

| ID | Given | When | Then |
|----|-------|------|------|
| AC-04-1 | A meta-learning event occurs | Event is generated | Event is persisted with timestamp, type, and context |
| AC-04-2 | Event log is queried | Time range is specified | Matching events are returned in chronological order |
| AC-04-3 | Event log exceeds retention limit | New event is logged | Old events are archived (not deleted) |
| AC-04-4 | MCP tool `get_meta_learning_log` is called | Parameters specify filters | Filtered events are returned |

### US-METAUTL-05: Integration with MetaCognitiveLoop
**Priority:** Must-Have

**Narrative:**
As the ContextGraph system,
I want the MetaCognitiveLoop to trigger lambda adjustments when MetaScore is persistently low,
So that GWT-level meta-cognition drives parameter optimization.

**Acceptance Criteria:**

| ID | Given | When | Then |
|----|-------|------|------|
| AC-05-1 | MetaScore < 0.5 for 5 consecutive operations | Dream is triggered | Lambda self-correction is also invoked |
| AC-05-2 | MetaCognitiveState is returned | State includes prediction error | Correction decision is made based on error |
| AC-05-3 | ACh level is increased | Dream triggers | Lambda learning rate (alpha) is also boosted |

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Description | Story Ref | Priority |
|----|-------------|-----------|----------|
| REQ-METAUTL-001 | System SHALL maintain a rolling accuracy history of at least 100 predictions | US-02 | Must |
| REQ-METAUTL-002 | System SHALL track accuracy per embedder (E1-E13) separately | US-02 | Must |
| REQ-METAUTL-003 | System SHALL adjust lambda_s and lambda_c when prediction_error > 0.2 | US-01 | Must |
| REQ-METAUTL-004 | Lambda adjustment SHALL use formula: lambda_new = lambda_old + alpha * (target - actual) | US-01 | Must |
| REQ-METAUTL-005 | Alpha (learning rate) SHALL be modulated by current ACh level | US-05 | Must |
| REQ-METAUTL-006 | Lambda weights SHALL always sum to 1.0 after adjustment | US-01 | Must |
| REQ-METAUTL-007 | Lambda weights SHALL be clamped to range [0.1, 0.9] | US-01 | Must |
| REQ-METAUTL-008 | System SHALL escalate to Bayesian optimization when accuracy < 0.7 for 10 cycles | US-03 | Must |
| REQ-METAUTL-009 | Bayesian optimization SHALL use GP surrogate with EI acquisition | US-03 | Should |
| REQ-METAUTL-010 | System SHALL log all meta-learning events with full context | US-04 | Must |
| REQ-METAUTL-011 | Event log SHALL support time-range queries | US-04 | Must |
| REQ-METAUTL-012 | MetaCognitiveLoop.evaluate SHALL trigger lambda correction when dream_triggered=true | US-05 | Must |
| REQ-METAUTL-013 | System SHALL provide MCP tool `get_meta_learning_status` | US-04 | Must |
| REQ-METAUTL-014 | System SHALL provide MCP tool `trigger_lambda_recalibration` | US-01 | Should |
| REQ-METAUTL-015 | System SHALL track domain-specific accuracy (Code, Medical, Legal, etc.) | US-02 | Should |

### 3.2 Non-Functional Requirements

| ID | Category | Description | Metric |
|----|----------|-------------|--------|
| NFR-METAUTL-001 | Performance | Lambda adjustment latency | < 1ms p95 |
| NFR-METAUTL-002 | Performance | Accuracy history lookup | < 100us p95 |
| NFR-METAUTL-003 | Performance | Event log query (100 events) | < 10ms p95 |
| NFR-METAUTL-004 | Memory | Accuracy history buffer | < 10KB per embedder |
| NFR-METAUTL-005 | Memory | Event log retention | 7 days hot, 90 days cold |
| NFR-METAUTL-006 | Reliability | Lambda invariant (sum=1.0) | 100% maintained |
| NFR-METAUTL-007 | Accuracy | Prediction accuracy after self-correction | > 0.7 within 50 cycles |

---

## 4. Data Model

### 4.1 Core Types

```rust
/// Prediction accuracy history for self-correction
pub struct MetaAccuracyHistory {
    /// Rolling buffer of accuracy values [0.0, 1.0]
    pub history: VecDeque<f32>,
    /// Maximum history size
    pub max_size: usize,
    /// Current rolling average
    pub rolling_average: f32,
    /// Consecutive low accuracy count (< 0.7)
    pub consecutive_low_count: u32,
    /// Last update timestamp
    pub last_update: DateTime<Utc>,
}

/// Per-embedder accuracy tracking
pub struct EmbedderAccuracyTracker {
    /// Accuracy history for each of 13 embedders
    pub embedder_history: [MetaAccuracyHistory; 13],
    /// Per-domain accuracy (Code, Medical, etc.)
    pub domain_accuracy: HashMap<Domain, MetaAccuracyHistory>,
}

/// Lambda adjustment parameters
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

/// Meta-learning event for introspection log
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

/// Event types for meta-learning log
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

/// Self-correction protocol state
pub struct SelfCorrectionState {
    /// Whether self-correction is enabled
    pub enabled: bool,
    /// Current lambda values (override lifecycle defaults)
    pub current_lambdas: Option<LifecycleLambdaWeights>,
    /// Accuracy tracker
    pub accuracy_tracker: EmbedderAccuracyTracker,
    /// Event log
    pub event_log: MetaLearningEventLog,
    /// Escalation status
    pub escalation_status: EscalationStatus,
}
```

### 4.2 Configuration

```rust
/// Self-correction protocol configuration
pub struct SelfCorrectionConfig {
    /// Enable/disable self-correction
    pub enabled: bool,
    /// Prediction error threshold to trigger adjustment
    pub error_threshold: f32,  // Default: 0.2
    /// Base learning rate for lambda adjustment
    pub base_alpha: f32,  // Default: 0.05
    /// Accuracy threshold for escalation
    pub escalation_accuracy_threshold: f32,  // Default: 0.7
    /// Consecutive cycles below threshold for escalation
    pub escalation_cycle_count: u32,  // Default: 10
    /// Maximum history size per embedder
    pub max_history_size: usize,  // Default: 100
    /// Lambda bounds [min, max]
    pub lambda_bounds: (f32, f32),  // Default: (0.1, 0.9)
    /// Event log retention days
    pub event_retention_days: u32,  // Default: 7
}
```

---

## 5. Edge Cases and Error States

### 5.1 Edge Cases

| ID | Scenario | Expected Behavior |
|----|----------|-------------------|
| EC-01 | Lambda adjustment would make lambda_s + lambda_c != 1.0 | Normalize adjustment to maintain invariant |
| EC-02 | Lambda adjustment would exceed bounds | Clamp to [0.1, 0.9] then renormalize |
| EC-03 | First prediction (no history) | Use lifecycle defaults, skip adjustment |
| EC-04 | All 13 embedders have different accuracy profiles | Per-embedder weighting in aggregation |
| EC-05 | Bayesian optimization suggests invalid lambdas | Reject and use fallback gradient adjustment |
| EC-06 | Accuracy oscillates around 0.7 threshold | Use hysteresis (0.65 to arm, 0.75 to disarm) |
| EC-07 | Domain changes mid-session | Maintain per-domain accuracy separately |
| EC-08 | System restart mid-correction cycle | Persist state, resume on restart |

### 5.2 Error States

| ID | Error Condition | HTTP Code | Message | Recovery |
|----|-----------------|-----------|---------|----------|
| ERR-01 | Invalid prediction error (NaN/Inf) | 400 | "Invalid prediction error value" | Skip adjustment, log warning |
| ERR-02 | Lambda invariant violated | 500 | "Lambda sum invariant violated" | Reset to lifecycle defaults |
| ERR-03 | Accuracy history corrupted | 500 | "Accuracy history corrupted" | Clear history, restart tracking |
| ERR-04 | Bayesian optimization timeout | 504 | "Bayesian optimization timeout" | Use gradient fallback |
| ERR-05 | Event log write failure | 500 | "Failed to persist meta-learning event" | Retry with exponential backoff |
| ERR-06 | ACh level out of range | 400 | "ACh level out of valid range" | Clamp to [0.001, 0.002] |

---

## 6. API Contracts

### 6.1 Internal Rust API

```rust
/// Trait for self-correcting lambda weights
pub trait SelfCorrectingLambda {
    /// Adjust lambda weights based on prediction error
    fn adjust_lambdas(&mut self, prediction_error: f32, ach_level: f32) -> UtlResult<LambdaAdjustment>;

    /// Get current corrected lambda weights
    fn corrected_weights(&self) -> LifecycleLambdaWeights;

    /// Check if escalation is needed
    fn should_escalate(&self) -> bool;

    /// Trigger Bayesian optimization
    fn trigger_bayesian_optimization(&mut self) -> UtlResult<LifecycleLambdaWeights>;

    /// Record accuracy for history
    fn record_accuracy(&mut self, embedder_idx: usize, accuracy: f32);

    /// Get rolling accuracy average
    fn rolling_accuracy(&self) -> f32;
}

/// Trait for meta-learning event logging
pub trait MetaLearningLogger {
    /// Log a meta-learning event
    fn log_event(&mut self, event: MetaLearningEvent) -> CoreResult<()>;

    /// Query events by time range
    fn query_events(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> CoreResult<Vec<MetaLearningEvent>>;

    /// Get recent events
    fn recent_events(&self, count: usize) -> Vec<MetaLearningEvent>;
}
```

### 6.2 MCP Tool Contracts

```yaml
tools:
  get_meta_learning_status:
    description: "Get current meta-UTL self-correction status"
    parameters: {}
    returns:
      enabled: boolean
      current_accuracy: number
      consecutive_low_count: number
      current_lambdas: { lambda_s: number, lambda_c: number }
      escalation_status: string
      recent_events_count: number

  trigger_lambda_recalibration:
    description: "Manually trigger lambda recalibration"
    parameters:
      force_bayesian: boolean  # Optional, force Bayesian optimization
    returns:
      success: boolean
      adjustment: { delta_s: number, delta_c: number }
      new_lambdas: { lambda_s: number, lambda_c: number }

  get_meta_learning_log:
    description: "Query meta-learning event log"
    parameters:
      start_time: string  # ISO 8601
      end_time: string    # ISO 8601
      event_type: string  # Optional filter
      limit: number       # Max events to return
    returns:
      events: array
      total_count: number
```

---

## 7. Test Plan

### 7.1 Unit Tests

| ID | Test Case | Input | Expected Output | REQ Ref |
|----|-----------|-------|-----------------|---------|
| TC-01 | Lambda adjustment with error > 0.2 | error=0.3, ach=0.001 | Adjustment applied | REQ-003 |
| TC-02 | Lambda adjustment with error <= 0.2 | error=0.15, ach=0.001 | No adjustment | REQ-003 |
| TC-03 | Lambda normalization after adjustment | delta_s=0.1 | sum=1.0 maintained | REQ-006 |
| TC-04 | Lambda clamping at bounds | delta_s=0.5 (would exceed) | Clamped to 0.9 | REQ-007 |
| TC-05 | ACh modulates learning rate | ach=0.002 vs ach=0.001 | Higher alpha with higher ACh | REQ-005 |
| TC-06 | Accuracy history FIFO eviction | 101 values added | Size = 100 | REQ-001 |
| TC-07 | Per-embedder accuracy tracking | E1=0.9, E7=0.6 | Separate histories | REQ-002 |
| TC-08 | Escalation trigger at 10 cycles | 10 consecutive < 0.7 | escalation_triggered=true | REQ-008 |
| TC-09 | Event logging captures context | Lambda adjustment | Event contains before/after | REQ-010 |
| TC-10 | Time-range query returns correct events | range=[t1, t2] | Events in range | REQ-011 |

### 7.2 Integration Tests

| ID | Test Case | Description | Validation |
|----|-----------|-------------|------------|
| IT-01 | MetaCognitiveLoop triggers correction | Dream triggers lambda adjustment | Lambda values change |
| IT-02 | End-to-end prediction validation | Predict -> Validate -> Correct | Accuracy improves over 50 cycles |
| IT-03 | MCP tool integration | Call get_meta_learning_status | Valid response structure |
| IT-04 | Persistence across restart | Adjust lambdas, restart, verify | State preserved |
| IT-05 | Bayesian escalation path | Force 10 low accuracy cycles | GP optimization invoked |

### 7.3 Chaos Tests

| ID | Test Case | Injection | Expected Recovery |
|----|-----------|-----------|-------------------|
| CT-01 | NaN in prediction error | Inject NaN | Skip adjustment, log warning |
| CT-02 | Concurrent lambda adjustments | Race condition | Mutex protects state |
| CT-03 | Event log disk full | Fill disk | Graceful degradation, in-memory buffer |
| CT-04 | Accuracy history corruption | Corrupt buffer | Reset and rebuild |

---

## 8. Implementation Notes

### 8.1 Algorithm: Lambda Adjustment

```
ALGORITHM: adjust_lambdas(prediction_error, ach_level)

INPUT:
  - prediction_error: f32 in [-1.0, 1.0]
  - ach_level: f32 in [0.001, 0.002]

OUTPUT:
  - LambdaAdjustment or None

1. IF abs(prediction_error) <= ERROR_THRESHOLD (0.2):
     RETURN None

2. Compute learning rate:
     alpha = BASE_ALPHA * (ach_level / ACH_BASELINE)
     alpha = clamp(alpha, 0.01, 0.1)

3. Compute raw adjustment:
     // Positive error means over-predicted surprise, reduce lambda_s
     // Negative error means under-predicted, increase lambda_s
     delta_s = -alpha * prediction_error
     delta_c = -delta_s  // Maintain sum invariant

4. Apply bounds:
     new_s = clamp(current_s + delta_s, LAMBDA_MIN, LAMBDA_MAX)
     new_c = clamp(current_c + delta_c, LAMBDA_MIN, LAMBDA_MAX)

5. Renormalize:
     sum = new_s + new_c
     new_s = new_s / sum
     new_c = new_c / sum

6. Log event:
     log(LambdaAdjustment, before=(current_s, current_c), after=(new_s, new_c))

7. RETURN LambdaAdjustment { delta_s, delta_c, alpha, prediction_error }
```

### 8.2 Algorithm: Escalation Check

```
ALGORITHM: should_escalate()

INPUT:
  - accuracy_history: VecDeque<f32>
  - consecutive_low_count: u32

OUTPUT:
  - bool

1. IF consecutive_low_count >= ESCALATION_CYCLE_COUNT (10):
     RETURN true

2. Compute rolling average:
     avg = sum(accuracy_history) / len(accuracy_history)

3. IF avg < ESCALATION_ACCURACY_THRESHOLD (0.7):
     INCREMENT consecutive_low_count
   ELSE:
     RESET consecutive_low_count = 0

4. RETURN false
```

### 8.3 Integration Points

1. **UtlProcessor.compute_learning()**: After computing LearningSignal, call self-correction
2. **MetaCognitiveLoop.evaluate()**: When dream_triggered, invoke lambda adjustment
3. **MCP handlers/utl.rs**: Add new handler functions for MCP tools
4. **LifecycleManager**: Add override mechanism for corrected lambdas

---

## 9. Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Lambda oscillation | Medium | Medium | Add momentum term, use exponential moving average |
| Over-correction | High | Low | Conservative alpha, bounds checking |
| Escalation loops | Medium | Low | Escalation cooldown period |
| State corruption | High | Very Low | Atomic updates, persistence checksums |
| Performance regression | Medium | Low | Lazy computation, caching |

---

## 10. Dependencies

### 10.1 Internal Dependencies

- `context-graph-core::gwt::meta_cognitive` - MetaCognitiveLoop
- `context-graph-utl::lifecycle` - LifecycleManager, LifecycleLambdaWeights
- `context-graph-core::neuromod` - Acetylcholine levels
- `context-graph-mcp::handlers` - MCP tool registration

### 10.2 External Dependencies

- `chrono` - Timestamp handling
- `serde` - Event serialization
- (Optional) `botorch` or `gpytorch` - Bayesian optimization (Phase 2)

---

## 11. Appendix

### 11.1 Constitution References

```yaml
# From constitution.yaml v4.2.0

meta_utl:
  awareness: [storage_prediction, retrieval_prediction, parameter_optimization]
  learning: { track: "success_rate, prediction_accuracy, parameter_drift", adapt: "lambda_S, lambda_C by domain/lifecycle" }
  predictors:
    storage_impact: { input: "fingerprint+context", output: "delta_L", accuracy: ">0.85" }
    retrieval_quality: { input: "query+top_k", output: "relevance", accuracy: ">0.80" }
    alignment_drift: { input: "fingerprint+time", output: "future alignment", window: "24h" }
  correction: { threshold: "error>0.2", escalate: "accuracy<0.7 for 100 ops" }
```

### 11.2 Related Specifications

- SPEC-GWT-001: Global Workspace Theory (C(t) = I(t) * R(t) * D(t))
- SPEC-NEUROMOD-001: Neuromodulation System (ACh, DA, etc.)
- SPEC-DREAM-001: Dream Layer (NREM/REM phases)

---

**Document History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | ContextGraph Team | Initial specification |
