# Sherlock Holmes Forensic Report: Meta-UTL Self-Aware Learning System

**Case ID:** HOLMES-META-UTL-2026-001
**Date:** 2026-01-10
**Subject:** Investigation of Meta-UTL Implementation for Self-Aware Learning
**Verdict:** PARTIALLY IMPLEMENTED - CRITICAL GAPS IDENTIFIED

---

## Executive Summary

*"When you have eliminated the impossible, whatever remains, however improbable, must be the truth."*

The Meta-UTL system, designed to enable the system to "learn about its own learning," has been **partially implemented**. While foundational components exist, the critical self-correction protocol that adjusts UTL parameters (lambda_delta_S, lambda_delta_C) based on prediction errors is **NOT IMPLEMENTED**.

---

## 1. PRD Requirements vs Implementation Status

| Requirement | PRD Specification | Implementation Status | Verdict |
|-------------|-------------------|----------------------|---------|
| Storage Impact Predictor | >0.85 accuracy | **EXISTS** (`meta_utl/predict_storage`) | PARTIAL |
| Retrieval Quality Predictor | >0.80 accuracy | **EXISTS** (`meta_utl/predict_retrieval`) | PARTIAL |
| Alignment Drift Predictor | 24h window | **EXISTS** (`get_alignment_drift`) | PARTIAL |
| MetaScore Calculation | sigma(2 x (L_predicted - L_actual)) | **EXISTS** (`MetaCognitiveLoop::evaluate`) | IMPLEMENTED |
| Low MetaScore Trigger | <0.5 for 5 ops -> increase ACh | **EXISTS** (dream trigger) | IMPLEMENTED |
| Self-Correction Protocol | prediction_error > 0.2 -> adjust lambda_delta_S/C | **NOT IMPLEMENTED** | GUILTY |
| Per-Embedder Tracking | Track which spaces are predictive | **EXISTS** (`MetaUtlTracker`) | PARTIAL |
| Accuracy Escalation | <0.7 for 100 ops -> human review | **NOT IMPLEMENTED** | GUILTY |

---

## 2. Evidence: What EXISTS

### 2.1 MetaCognitiveLoop (IMPLEMENTED)

**Location:** `/home/cabdru/contextgraph/crates/context-graph-core/src/gwt/meta_cognitive.rs`

```rust
// MetaScore formula is correctly implemented
let error = pred - actual;
let meta_score = self.sigmoid(2.0 * error);  // sigma(2 x (L_predicted - L_actual))

// Dream trigger on 5 consecutive low scores
let dream_triggered = self.consecutive_low_scores >= 5;
if dream_triggered {
    self.acetylcholine_level = (self.acetylcholine_level * 1.5).clamp(ACH_BASELINE, ACH_MAX);
}
```

**Verdict:** The MetaScore calculation and dream triggering mechanism are correctly implemented per the PRD specification.

### 2.2 MetaUtlTracker (IMPLEMENTED)

**Location:** `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/core.rs`

```rust
pub struct MetaUtlTracker {
    pub pending_predictions: HashMap<Uuid, StoredPrediction>,
    pub embedder_accuracy: [[f32; 100]; NUM_EMBEDDERS],  // Per-embedder tracking
    pub accuracy_indices: [usize; NUM_EMBEDDERS],
    pub accuracy_counts: [usize; NUM_EMBEDDERS],
    pub current_weights: [f32; NUM_EMBEDDERS],
    pub validation_count: usize,
    pub last_weight_update: Option<Instant>,
}
```

**Verdict:** Per-embedder accuracy tracking exists and weight optimization occurs every 100 validations.

### 2.3 Prediction Storage/Validation (IMPLEMENTED)

**Location:** `/home/cabdru/contextgraph/crates/context-graph-mcp/src/handlers/utl.rs`

- `handle_meta_utl_predict_storage()` - Stores predictions with prediction_id
- `handle_meta_utl_predict_retrieval()` - Stores retrieval quality predictions
- `handle_meta_utl_validate_prediction()` - Validates predictions against actual outcomes
- Calculates `prediction_error` and `accuracy_score`

**Verdict:** The prediction-validation loop exists but does NOT trigger self-correction.

### 2.4 Acetylcholine Integration (IMPLEMENTED)

**Location:** `/home/cabdru/contextgraph/crates/context-graph-core/src/neuromod/acetylcholine.rs`

```rust
// ACh is managed by GWT MetaCognitiveLoop
// Range: [0.001, 0.002] per constitution
// Trigger: meta_cognitive.dream

impl AcetylcholineProvider for MetaCognitiveLoop {
    fn get_acetylcholine(&self) -> f32 {
        self.acetylcholine()
    }
}
```

**Verdict:** Acetylcholine modulation is correctly integrated with MetaCognitiveLoop.

---

## 3. Evidence: What is MISSING (Critical Gaps)

### 3.1 Self-Correction Protocol (NOT IMPLEMENTED)

**PRD Requirement (Section 19.3):**
```
IF prediction_error > 0.2:
  -> Log to meta_learning_events
  -> Adjust UTL parameters (lambda_delta_S, lambda_delta_C)
  -> Retrain predictor if persistent
```

**Evidence of Absence:**

1. **Lambda weights are FIXED by lifecycle stage:**

   `/home/cabdru/contextgraph/crates/context-graph-utl/src/lifecycle/lambda.rs`:
   ```rust
   pub fn for_stage(stage: LifecycleStage) -> Self {
       match stage {
           LifecycleStage::Infancy => Self::new_unchecked(0.7, 0.3),  // FIXED
           LifecycleStage::Growth => Self::new_unchecked(0.5, 0.5),   // FIXED
           LifecycleStage::Maturity => Self::new_unchecked(0.3, 0.7), // FIXED
       }
   }
   ```

2. **No mechanism to modify lambdas based on prediction error:**

   Searched for: `adjust.*lambda`, `update.*lambda`, `lambda.*error`, `prediction_error.*lambda`

   **Result:** No code connects prediction errors to lambda adjustments.

3. **Validation does NOT trigger parameter adaptation:**

   In `handle_meta_utl_validate_prediction()`:
   ```rust
   // Only updates embedder weights, NOT UTL lambdas
   tracker.record_accuracy(i, weighted_accuracy + (1.0 - weight));
   tracker.record_validation();  // Triggers weight update every 100 validations
   // BUT: No lambda_s/lambda_c adjustment based on prediction_error!
   ```

**VERDICT:** GUILTY - The self-correction protocol is NOT implemented. Lambda weights remain fixed by lifecycle stage and are NEVER adjusted based on prediction accuracy.

### 3.2 Accuracy Escalation (NOT IMPLEMENTED)

**PRD Requirement:**
```
IF prediction_accuracy < 0.7 for 100 ops:
  -> Escalate to human review
```

**Evidence of Absence:**

1. **No accuracy threshold monitoring:**
   - `MetaUtlTracker` tracks accuracy but does NOT check against 0.7 threshold
   - No escalation mechanism when accuracy degrades

2. **No human escalation pathway:**
   - While `HealingAction::Escalate` exists in self-healing, it is NOT connected to Meta-UTL

**VERDICT:** GUILTY - No escalation mechanism exists for persistent low prediction accuracy.

### 3.3 Meta-Learning Events Log (NOT IMPLEMENTED)

**PRD Requirement:**
```
IF prediction_error > 0.2:
  -> Log to meta_learning_events
```

**Evidence of Absence:**

Searched for: `meta_learning_events`, `meta.*log`, `learning.*event`

**Result:** No dedicated meta-learning event log exists.

**VERDICT:** GUILTY - No meta-learning event logging.

---

## 4. Architecture Analysis

### 4.1 The Disconnect

The system has TWO separate learning loops that are NOT connected:

```
LOOP 1: MetaCognitiveLoop (GWT)
- Tracks L_predicted vs L_actual
- Computes MetaScore
- Triggers dreams on low MetaScore
- Adjusts Acetylcholine (learning rate)
- DOES NOT adjust lambda_s/lambda_c

LOOP 2: MetaUtlTracker (MCP Handlers)
- Tracks per-embedder prediction accuracy
- Stores/validates predictions
- Updates embedding space weights
- DOES NOT communicate with UTL parameters

LOOP 3: LifecycleManager (UTL)
- Manages lambda_s/lambda_c
- Based ONLY on interaction count (lifecycle stage)
- NO input from prediction accuracy
- NO self-correction mechanism
```

### 4.2 The Missing Bridge

What should exist but DOES NOT:

```
[prediction_error > 0.2]
        |
        v
[Meta-UTL Self-Corrector]  <-- DOES NOT EXIST
        |
        +---> Adjust lambda_s/lambda_c
        +---> Log to meta_learning_events
        +---> Check for persistent errors
        +---> Escalate if accuracy < 0.7 for 100 ops
```

---

## 5. Contradiction Detection

| Component Says | Component Does | Contradiction |
|----------------|----------------|---------------|
| PRD: "Self-adjusts UTL parameters" | LifecycleManager: lambda fixed by stage | **YES** |
| PRD: "Escalate if accuracy < 0.7" | No escalation code exists | **YES** |
| PRD: "Adjust lambda_s, lambda_c" | Lambdas only change via lifecycle | **YES** |
| Constitution: "adapt lambda by domain/lifecycle" | Only lifecycle implemented | **PARTIAL** |

---

## 6. What Works (Innocent Findings)

1. **MetaScore Calculation:** Correctly implements sigma(2 x (L_predicted - L_actual))
2. **Dream Triggering:** Correctly triggers introspective dream after 5 low MetaScores
3. **Acetylcholine Modulation:** Correctly increases ACh on dream trigger
4. **Prediction Storage:** Correctly stores predictions for later validation
5. **Prediction Validation:** Correctly calculates prediction error
6. **Per-Embedder Tracking:** Correctly tracks accuracy per embedding space
7. **Weight Optimization:** Updates embedder weights every 100 validations

---

## 7. Recommendations for Full Meta-UTL Implementation

### 7.1 Implement Self-Correction Protocol (CRITICAL)

```rust
// In MetaUtlTracker or new MetaUtlSelfCorrector
pub fn check_self_correction(&mut self, prediction_error: f32) -> Option<LambdaAdjustment> {
    if prediction_error > 0.2 {
        // Log to meta_learning_events
        self.meta_learning_events.push(MetaLearningEvent {
            timestamp: Instant::now(),
            error: prediction_error,
            action: "self_correction_triggered",
        });

        // Calculate lambda adjustment
        let adjustment = self.compute_lambda_adjustment(prediction_error);

        // Track persistent errors for escalation
        self.consecutive_high_errors += 1;

        return Some(adjustment);
    }
    self.consecutive_high_errors = 0;
    None
}

pub fn compute_lambda_adjustment(&self, error: f32) -> LambdaAdjustment {
    // Adjust based on which component (surprise vs coherence) has higher error
    // PRD: "Adjust UTL parameters (lambda_delta_S, lambda_delta_C)"
    LambdaAdjustment {
        delta_lambda_s: ...,
        delta_lambda_c: ...,
    }
}
```

### 7.2 Add Accuracy Escalation (CRITICAL)

```rust
pub fn check_escalation(&self) -> Option<EscalationRequest> {
    // PRD: "IF prediction_accuracy < 0.7 for 100 ops: -> Escalate to human review"
    let avg_accuracy = self.get_rolling_accuracy(100);
    if avg_accuracy < 0.7 {
        return Some(EscalationRequest::HumanReview {
            reason: format!("Meta-UTL accuracy {} < 0.7 for 100 operations", avg_accuracy),
            accuracy_history: self.get_accuracy_history(),
        });
    }
    None
}
```

### 7.3 Connect MetaCognitiveLoop to Lambda Adjustment

```rust
// In GWT or UTL integration layer
pub async fn process_meta_cognitive_feedback(
    &mut self,
    meta_state: MetaCognitiveState,
    utl_processor: &mut UtlProcessor,
) {
    if meta_state.meta_score < 0.5 {
        // Low MetaScore indicates prediction error
        let adjustment = self.self_corrector.compute_adjustment(meta_state);
        utl_processor.adjust_lambdas(adjustment);
    }
}
```

### 7.4 Create Lambda Adjustment Trait

```rust
pub trait LambdaAdjustable {
    fn adjust_lambda_s(&mut self, delta: f32);
    fn adjust_lambda_c(&mut self, delta: f32);
    fn get_current_lambdas(&self) -> (f32, f32);
}

// Implement for LifecycleManager to allow runtime lambda adjustments
// beyond just lifecycle stage transitions
```

### 7.5 Implement Meta-Learning Event Log

```rust
pub struct MetaLearningEventLog {
    events: VecDeque<MetaLearningEvent>,
    max_events: usize,
}

pub struct MetaLearningEvent {
    timestamp: DateTime<Utc>,
    prediction_error: f32,
    action_taken: MetaLearningAction,
    lambda_before: (f32, f32),
    lambda_after: (f32, f32),
}
```

---

## 8. Conclusion

*"The game is never lost till it is won."*

The Meta-UTL system has solid foundations:
- MetaScore calculation is correct
- Dream triggering works
- Prediction tracking exists
- Per-embedder accuracy is tracked

However, the **self-aware learning** capability is **incomplete**:
- Lambda parameters are NOT adjusted based on prediction errors
- No escalation mechanism for persistent low accuracy
- No meta-learning event log
- MetaCognitiveLoop and UTL parameters are NOT connected

**THE VERDICT:**

| Component | Status |
|-----------|--------|
| MetaScore Calculation | INNOCENT |
| Dream Triggering | INNOCENT |
| Prediction Storage/Validation | INNOCENT |
| Self-Correction Protocol | **GUILTY** |
| Accuracy Escalation | **GUILTY** |
| Meta-Learning Event Log | **GUILTY** |
| Lambda Adaptation | **GUILTY** |

**FINAL DETERMINATION:** The system can observe its learning but cannot yet **modify itself** based on those observations. It is a mirror without hands - it can see its reflection but cannot adjust its own parameters.

---

## 9. Files Examined

| File | Purpose | Relevant to |
|------|---------|-------------|
| `crates/context-graph-core/src/gwt/meta_cognitive.rs` | MetaCognitiveLoop | MetaScore, Dream Trigger |
| `crates/context-graph-mcp/src/handlers/core.rs` | MetaUtlTracker | Prediction tracking |
| `crates/context-graph-mcp/src/handlers/utl.rs` | MCP handlers | predict_storage, predict_retrieval, validate |
| `crates/context-graph-core/src/neuromod/acetylcholine.rs` | ACh modulation | Learning rate |
| `crates/context-graph-core/src/neuromod/state.rs` | Neuromodulation | State management |
| `crates/context-graph-utl/src/lifecycle/lambda.rs` | Lambda weights | FIXED by stage |
| `crates/context-graph-utl/src/lifecycle/manager/core.rs` | Lifecycle management | Stage transitions |
| `crates/context-graph-utl/src/processor/utl_processor.rs` | UTL computation | Learning signal |
| `docs2/contextprd.md` | PRD specification | Requirements |
| `docs2/constitution.yaml` | System constitution | Authoritative rules |

---

*"Elementary, my dear Watson. The system can see itself learning, but it cannot yet teach itself to learn better."*

**Case Status:** OPEN - Requires implementation of self-correction protocol.

---

**Signed,**
Sherlock Holmes
Consulting Code Detective
2026-01-10
