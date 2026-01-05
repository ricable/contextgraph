# TASK-L004: Johari Transition Manager

```yaml
metadata:
  id: "TASK-L004"
  title: "Johari Transition Manager"
  layer: "logic"
  priority: "P0"
  estimated_hours: 8
  created: "2026-01-04"
  status: "pending"
  dependencies:
    - "TASK-F003"  # JohariFingerprint struct
    - "TASK-F002"  # TeleologicalFingerprint struct
  spec_refs:
    - "projectionplan1.md:johari-quadrants"
    - "projectionplan2.md:awareness-evolution"
```

## Problem Statement

Implement a manager that tracks and facilitates transitions between Johari quadrants (Open, Hidden, Blind, Unknown) for each embedding space, enabling awareness-based retrieval and self-reflection.

## Context

The Johari Window model classifies each embedding space's content into four quadrants based on self-awareness and disclosure. This enables:
- Awareness-based retrieval (prioritize open/disclosed content)
- Blind spot discovery (surface hidden patterns)
- Self-reflection prompts (move hidden to open)
- Unknown exploration (novel discovery)

The 12-space Multi-Array architecture applies Johari classification per embedding space, allowing fine-grained awareness tracking.

## Technical Specification

### Data Structures

```rust
/// Johari quadrant classification
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JohariQuadrant {
    /// Known to self and others - openly used
    Open,
    /// Known to self but not disclosed - intentionally hidden
    Hidden,
    /// Known to others but not self - blind spots
    Blind,
    /// Unknown to both - unexplored territory
    Unknown,
}

/// Johari classification for all 12 embedding spaces
#[derive(Clone, Debug, PartialEq)]
pub struct JohariFingerprint {
    /// Quadrant classification per space
    pub quadrants: [JohariQuadrant; 12],

    /// Confidence in each classification
    pub confidence: [f32; 12],

    /// Disclosure intent flags per space
    pub disclosure_intent: [bool; 12],

    /// When classifications were last updated
    pub updated_at: Timestamp,

    /// Transition history (last N transitions)
    pub transition_history: Vec<JohariTransition>,
}

/// A transition between quadrants
#[derive(Clone, Debug)]
pub struct JohariTransition {
    pub space_index: usize,
    pub from: JohariQuadrant,
    pub to: JohariQuadrant,
    pub trigger: TransitionTrigger,
    pub timestamp: Timestamp,
}

/// What triggered a quadrant transition
#[derive(Clone, Debug)]
pub enum TransitionTrigger {
    /// User explicitly disclosed information
    UserDisclosure,
    /// Feedback from others revealed blind spot
    ExternalFeedback,
    /// System discovered pattern through analysis
    PatternDiscovery,
    /// Time-based decay (open -> hidden if not used)
    TemporalDecay,
    /// Manual classification by user
    ManualClassification,
}

/// Configuration for Johari management
pub struct JohariConfig {
    /// Maximum transition history to retain
    pub max_history: usize,

    /// Decay period before open -> hidden (if unused)
    pub decay_period: Duration,

    /// Minimum confidence to classify as non-Unknown
    pub confidence_threshold: f32,

    /// Whether to auto-discover blind spots
    pub auto_discover_blind: bool,
}
```

### Core Trait

```rust
/// Manages Johari quadrant transitions
#[async_trait]
pub trait JohariTransitionManager: Send + Sync {
    /// Classify a fingerprint's Johari quadrants
    async fn classify(
        &self,
        fingerprint: &SemanticFingerprint,
        context: &ClassificationContext,
    ) -> Result<JohariFingerprint, JohariError>;

    /// Request transition from one quadrant to another
    async fn transition(
        &self,
        memory_id: MemoryId,
        space_index: usize,
        to_quadrant: JohariQuadrant,
        trigger: TransitionTrigger,
    ) -> Result<JohariFingerprint, JohariError>;

    /// Batch transition multiple spaces
    async fn transition_batch(
        &self,
        memory_id: MemoryId,
        transitions: Vec<(usize, JohariQuadrant, TransitionTrigger)>,
    ) -> Result<JohariFingerprint, JohariError>;

    /// Find memories in a specific quadrant configuration
    async fn find_by_quadrant(
        &self,
        pattern: QuadrantPattern,
        limit: usize,
    ) -> Result<Vec<(MemoryId, JohariFingerprint)>, JohariError>;

    /// Discover potential blind spots
    async fn discover_blind_spots(
        &self,
        memory_id: MemoryId,
        external_signals: &[ExternalSignal],
    ) -> Result<Vec<BlindSpotCandidate>, JohariError>;

    /// Get transition statistics
    async fn get_transition_stats(
        &self,
        time_range: TimeRange,
    ) -> Result<TransitionStats, JohariError>;
}

/// Context for classification decisions
pub struct ClassificationContext {
    /// User's known disclosure preferences
    pub disclosure_preferences: DisClosurePreferences,

    /// Recent access patterns for this memory
    pub access_history: Vec<AccessEvent>,

    /// External feedback signals
    pub external_signals: Vec<ExternalSignal>,
}

/// Pattern for querying by quadrant
pub enum QuadrantPattern {
    /// All spaces in a single quadrant
    AllIn(JohariQuadrant),

    /// At least N spaces in a quadrant
    AtLeast { quadrant: JohariQuadrant, count: usize },

    /// Specific configuration
    Exact([JohariQuadrant; 12]),

    /// Mixed (some open, some hidden)
    Mixed { open_min: usize, hidden_max: usize },
}
```

### Transition State Machine

```rust
impl JohariQuadrant {
    /// Valid transitions from this quadrant
    pub fn valid_transitions(&self) -> &[JohariQuadrant] {
        match self {
            // Open can become Hidden (withdrawal) or Unknown (forget)
            Self::Open => &[Self::Hidden, Self::Unknown],

            // Hidden can become Open (disclosure) or Unknown (forget)
            Self::Hidden => &[Self::Open, Self::Unknown],

            // Blind can become Open (awareness) or Unknown (dismiss)
            Self::Blind => &[Self::Open, Self::Unknown],

            // Unknown can transition to any (discovery/classification)
            Self::Unknown => &[Self::Open, Self::Hidden, Self::Blind],
        }
    }

    /// Check if transition is valid
    pub fn can_transition_to(&self, target: JohariQuadrant) -> bool {
        self.valid_transitions().contains(&target)
    }
}

/// State machine for quadrant transitions
pub struct JohariStateMachine {
    config: JohariConfig,
}

impl JohariStateMachine {
    pub fn apply_transition(
        &self,
        current: JohariQuadrant,
        target: JohariQuadrant,
        trigger: TransitionTrigger,
    ) -> Result<JohariQuadrant, TransitionError> {
        // Validate transition is allowed
        if !current.can_transition_to(target) {
            return Err(TransitionError::InvalidTransition {
                from: current,
                to: target,
            });
        }

        // Validate trigger is appropriate for transition
        match (&current, &target, &trigger) {
            // Open -> Hidden requires intent
            (JohariQuadrant::Open, JohariQuadrant::Hidden, TransitionTrigger::TemporalDecay) |
            (JohariQuadrant::Open, JohariQuadrant::Hidden, TransitionTrigger::ManualClassification) => Ok(target),

            // Hidden -> Open requires disclosure
            (JohariQuadrant::Hidden, JohariQuadrant::Open, TransitionTrigger::UserDisclosure) |
            (JohariQuadrant::Hidden, JohariQuadrant::Open, TransitionTrigger::ManualClassification) => Ok(target),

            // Blind -> Open requires feedback awareness
            (JohariQuadrant::Blind, JohariQuadrant::Open, TransitionTrigger::ExternalFeedback) |
            (JohariQuadrant::Blind, JohariQuadrant::Open, TransitionTrigger::PatternDiscovery) => Ok(target),

            // Unknown -> any is discovery
            (JohariQuadrant::Unknown, _, _) => Ok(target),

            // Any -> Unknown is valid (forgetting/dismissing)
            (_, JohariQuadrant::Unknown, _) => Ok(target),

            _ => Err(TransitionError::InvalidTrigger {
                from: current,
                to: target,
                trigger,
            }),
        }
    }
}
```

## Implementation Requirements

### Prerequisites

- [ ] TASK-F003 complete (JohariFingerprint struct defined)
- [ ] TASK-F002 complete (TeleologicalFingerprint with Johari field)

### Scope

#### In Scope

- Johari classification per embedding space
- Quadrant transition state machine
- Transition history tracking
- Blind spot discovery algorithm
- Quadrant-based queries
- Transition statistics

#### Out of Scope

- Johari index storage (TASK-F004)
- UI for disclosure management
- External feedback collection

### Constraints

- Classification < 5ms per fingerprint
- Transition operations atomic
- History limited to prevent unbounded growth
- Thread-safe for concurrent transitions

## Pseudo Code

```
FUNCTION classify_johari(fingerprint, context):
    johari = JohariFingerprint::default()

    FOR space_idx IN 0..12:
        embedding = fingerprint.embeddings[space_idx]
        IF embedding IS NULL:
            johari.quadrants[space_idx] = Unknown
            johari.confidence[space_idx] = 0.0
            CONTINUE

        // Check disclosure preferences
        IF context.disclosure_preferences.is_disclosed(space_idx):
            johari.quadrants[space_idx] = Open
            johari.disclosure_intent[space_idx] = true
        ELSE IF context.disclosure_preferences.is_hidden(space_idx):
            johari.quadrants[space_idx] = Hidden
            johari.disclosure_intent[space_idx] = false
        ELSE:
            // Check access patterns for implicit classification
            access_count = count_recent_accesses(context.access_history, space_idx)
            external_count = count_external_references(context.external_signals, space_idx)

            IF access_count > 0 AND external_count > 0:
                johari.quadrants[space_idx] = Open
            ELSE IF access_count > 0 AND external_count == 0:
                johari.quadrants[space_idx] = Hidden
            ELSE IF access_count == 0 AND external_count > 0:
                johari.quadrants[space_idx] = Blind
            ELSE:
                johari.quadrants[space_idx] = Unknown

        // Compute confidence based on signal strength
        johari.confidence[space_idx] = compute_confidence(
            access_count, external_count, embedding
        )

    johari.updated_at = now()
    RETURN johari

FUNCTION transition(memory_id, space_idx, to_quadrant, trigger):
    // Load current Johari state
    current = storage.get_johari(memory_id)
    current_quadrant = current.quadrants[space_idx]

    // Validate transition via state machine
    state_machine.apply_transition(current_quadrant, to_quadrant, trigger)?

    // Record transition
    transition = JohariTransition {
        space_index: space_idx,
        from: current_quadrant,
        to: to_quadrant,
        trigger,
        timestamp: now()
    }

    // Update fingerprint
    current.quadrants[space_idx] = to_quadrant
    current.transition_history.push(transition)
    current.transition_history.truncate(config.max_history)
    current.updated_at = now()

    // Persist
    storage.update_johari(memory_id, current)

    RETURN current

FUNCTION discover_blind_spots(memory_id, external_signals):
    fingerprint = storage.get_fingerprint(memory_id)
    johari = storage.get_johari(memory_id)
    candidates = []

    FOR space_idx IN 0..12:
        IF johari.quadrants[space_idx] == Unknown OR johari.quadrants[space_idx] == Hidden:
            // Check if external signals reference this space
            signal_strength = 0.0
            FOR signal IN external_signals:
                IF signal.references_space(space_idx):
                    signal_strength += signal.strength

            IF signal_strength > config.blind_spot_threshold:
                candidates.push(BlindSpotCandidate {
                    space_index: space_idx,
                    current_quadrant: johari.quadrants[space_idx],
                    signal_strength,
                    suggested_transition: Blind
                })

    RETURN candidates
```

## Definition of Done

### Implementation Checklist

- [ ] `JohariQuadrant` enum with transition rules
- [ ] `JohariFingerprint` struct with per-space quadrants
- [ ] `JohariTransition` history tracking
- [ ] `JohariStateMachine` for valid transitions
- [ ] `JohariTransitionManager` trait
- [ ] Default implementation with classification
- [ ] Blind spot discovery algorithm
- [ ] Quadrant-based query support

### Testing Requirements

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_transitions() {
        assert!(JohariQuadrant::Open.can_transition_to(JohariQuadrant::Hidden));
        assert!(JohariQuadrant::Hidden.can_transition_to(JohariQuadrant::Open));
        assert!(JohariQuadrant::Blind.can_transition_to(JohariQuadrant::Open));
        assert!(JohariQuadrant::Unknown.can_transition_to(JohariQuadrant::Open));

        // Invalid transitions
        assert!(!JohariQuadrant::Open.can_transition_to(JohariQuadrant::Blind));
        assert!(!JohariQuadrant::Hidden.can_transition_to(JohariQuadrant::Blind));
    }

    #[tokio::test]
    async fn test_classification() {
        let manager = DefaultJohariManager::new();
        let fingerprint = create_test_fingerprint();
        let context = ClassificationContext {
            disclosure_preferences: DisClosurePreferences::all_open(),
            access_history: vec![],
            external_signals: vec![],
        };

        let johari = manager.classify(&fingerprint, &context).await.unwrap();

        // All should be Open since preferences say all_open
        for q in johari.quadrants {
            assert_eq!(q, JohariQuadrant::Open);
        }
    }

    #[tokio::test]
    async fn test_transition() {
        let manager = DefaultJohariManager::new();
        let memory_id = MemoryId::new();

        // Store initial state as Open
        manager.initialize(memory_id, JohariFingerprint::all_open()).await.unwrap();

        // Transition space 0 from Open to Hidden
        let result = manager.transition(
            memory_id,
            0,
            JohariQuadrant::Hidden,
            TransitionTrigger::ManualClassification,
        ).await.unwrap();

        assert_eq!(result.quadrants[0], JohariQuadrant::Hidden);
        assert_eq!(result.transition_history.len(), 1);
    }

    #[tokio::test]
    async fn test_blind_spot_discovery() {
        let manager = DefaultJohariManager::new();
        let memory_id = MemoryId::new();

        // Initialize with some Unknown spaces
        let mut johari = JohariFingerprint::default();
        johari.quadrants[5] = JohariQuadrant::Unknown;
        manager.initialize(memory_id, johari).await.unwrap();

        // External signal referencing space 5
        let signals = vec![ExternalSignal {
            space_index: 5,
            strength: 0.8,
            source: "external_system".into(),
        }];

        let candidates = manager.discover_blind_spots(memory_id, &signals).await.unwrap();

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].space_index, 5);
    }
}
```

### Verification Commands

```bash
# Run unit tests
cargo test -p context-graph-core johari

# Run with coverage
cargo llvm-cov -p context-graph-core --test johari

# Benchmark operations
cargo bench -p context-graph-core -- johari
```

## Files to Create

| File | Description |
|------|-------------|
| `crates/context-graph-core/src/johari/mod.rs` | Johari module |
| `crates/context-graph-core/src/johari/quadrant.rs` | JohariQuadrant enum |
| `crates/context-graph-core/src/johari/fingerprint.rs` | JohariFingerprint struct |
| `crates/context-graph-core/src/johari/state_machine.rs` | Transition state machine |
| `crates/context-graph-core/src/johari/manager.rs` | JohariTransitionManager trait and impl |
| `crates/context-graph-core/src/johari/discovery.rs` | Blind spot discovery |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-core/src/lib.rs` | Add `pub mod johari` |

## Traceability

| Requirement | Source | Coverage |
|-------------|--------|----------|
| Johari quadrants | projectionplan1.md:johari | Complete |
| Per-space classification | projectionplan1.md:multi-array | Complete |
| Transition tracking | projectionplan2.md:evolution | Complete |
| Blind spot discovery | projectionplan2.md:awareness | Complete |

---

*Task created: 2026-01-04*
*Layer: Logic*
*Priority: P0 - Core awareness tracking*
