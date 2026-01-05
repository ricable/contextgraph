# Task: TASK-F003 - Implement JohariFingerprint Struct

## Metadata
- **ID**: TASK-F003
- **Layer**: Foundation
- **Priority**: P0 (Critical Path)
- **Estimated Effort**: M (Medium)
- **Dependencies**: TASK-F001
- **Traces To**: TS-103, FR-203

## Description

Implement the `JohariFingerprint` struct that provides per-embedder Johari Window classification. Unlike the existing `JohariQuadrant` enum which classifies a single memory, this struct classifies each of the 12 embedding spaces independently.

A memory can be:
- **Open** in semantic space (E1) - we understand its meaning
- **Blind** in causal space (E5) - we don't understand why it matters
- **Hidden** in code space (E7) - latent technical knowledge
- **Unknown** in entity space (E11) - frontier discovery

This enables cross-space Johari analysis: "Find memories that are Open(semantic) but Blind(causal)" to surface knowledge gaps.

## Acceptance Criteria

- [ ] `JohariFingerprint` struct with per-embedder quadrant weights
- [ ] Soft classification: 4 weights per embedder (not just dominant)
- [ ] Confidence score per embedder classification
- [ ] Transition probability matrix for evolution prediction
- [ ] Integration with existing `JohariQuadrant` enum from `types/johari/`
- [ ] `classify_quadrant(entropy, coherence)` static method
- [ ] `dominant_quadrant(embedder_idx)` method
- [ ] `find_by_quadrant(quadrant)` method returns embedder indices
- [ ] `find_blind_spots()` method for cross-space analysis
- [ ] Compact byte encoding (2 bits per quadrant = 3 bytes for 12)
- [ ] Unit tests with varied Johari distributions

## Implementation Steps

1. Read existing `crates/context-graph-core/src/types/johari/quadrant.rs` for `JohariQuadrant` enum
2. Create `crates/context-graph-core/src/types/fingerprint/johari.rs`:
   - Import `JohariQuadrant` from existing module
   - Define `NUM_EMBEDDERS = 12` constant
   - Define `ENTROPY_THRESHOLD = 0.5` and `COHERENCE_THRESHOLD = 0.5`
   - Implement `JohariFingerprint` struct
   - Implement classification and query methods
   - Implement compact byte encoding/decoding
3. Update `crates/context-graph-core/src/types/fingerprint/mod.rs` to export

## Files Affected

### Files to Create
- `crates/context-graph-core/src/types/fingerprint/johari.rs` - JohariFingerprint implementation

### Files to Modify
- `crates/context-graph-core/src/types/fingerprint/mod.rs` - Export JohariFingerprint

### Existing Files to Reference (READ ONLY)
- `crates/context-graph-core/src/types/johari/quadrant.rs` - JohariQuadrant enum definition

## Code Signature (Definition of Done)

```rust
// johari.rs
use crate::types::johari::JohariQuadrant;

pub const NUM_EMBEDDERS: usize = 12;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JohariFingerprint {
    /// Quadrant weights per embedder: [Open, Hidden, Blind, Unknown]
    /// Each inner array sums to 1.0 (soft classification)
    pub quadrants: [[f32; 4]; NUM_EMBEDDERS],

    /// Confidence of classification per embedder [0.0, 1.0]
    pub confidence: [f32; NUM_EMBEDDERS],

    /// Transition probability matrix
    /// transitions[embedder][from_quadrant][to_quadrant]
    pub transition_probs: [[[f32; 4]; 4]; NUM_EMBEDDERS],
}

impl JohariFingerprint {
    pub const ENTROPY_THRESHOLD: f32 = 0.5;
    pub const COHERENCE_THRESHOLD: f32 = 0.5;

    /// Create zeroed fingerprint with uniform transition priors
    pub fn zeroed() -> Self;

    /// Classify based on entropy and coherence metrics
    /// Open: Low entropy, High coherence (aware)
    /// Hidden: Low entropy, Low coherence (latent)
    /// Blind: High entropy, Low coherence (discovery opportunity)
    /// Unknown: High entropy, High coherence (frontier)
    pub fn classify_quadrant(entropy: f32, coherence: f32) -> JohariQuadrant;

    /// Get dominant quadrant for an embedder
    pub fn dominant_quadrant(&self, embedder_idx: usize) -> JohariQuadrant;

    /// Set quadrant weights for an embedder (normalizes to sum=1.0)
    pub fn set_quadrant(&mut self, embedder_idx: usize,
                        open: f32, hidden: f32, blind: f32, unknown: f32,
                        confidence: f32);

    /// Find embedders in a specific quadrant
    pub fn find_by_quadrant(&self, quadrant: JohariQuadrant) -> Vec<usize>;

    /// Find blind spots: high semantic awareness but low causal awareness
    pub fn find_blind_spots(&self) -> Vec<(usize, f32)>;

    /// Predict next quadrant given current state
    pub fn predict_transition(&self, embedder_idx: usize, current: JohariQuadrant) -> JohariQuadrant;

    /// Encode quadrants as compact bytes (2 bits per quadrant = 3 bytes)
    pub fn to_compact_bytes(&self) -> [u8; 3];

    /// Decode quadrants from compact bytes
    pub fn from_compact_bytes(bytes: [u8; 3]) -> Self;
}
```

## Testing Requirements

### Unit Tests
- `test_johari_fingerprint_zeroed` - All weights zero, uniform transition priors
- `test_classify_quadrant_open` - Low entropy + high coherence
- `test_classify_quadrant_hidden` - Low entropy + low coherence
- `test_classify_quadrant_blind` - High entropy + low coherence
- `test_classify_quadrant_unknown` - High entropy + high coherence
- `test_dominant_quadrant` - Returns highest weighted quadrant
- `test_set_quadrant_normalizes` - Weights sum to 1.0 after set
- `test_find_by_quadrant` - Returns correct embedder indices
- `test_find_blind_spots` - Detects semantic-open but causal-blind
- `test_compact_bytes_roundtrip` - Encode then decode matches
- `test_predict_transition` - Uses transition matrix correctly

### Test Fixtures
- Create `tests/fixtures/johari/sample_johari_fingerprint.json`
- Include varied distributions across embedders

## Verification

```bash
# Compile check
cargo check -p context-graph-core

# Run unit tests
cargo test -p context-graph-core johari

# Verify integration with existing JohariQuadrant
cargo test -p context-graph-core types::johari
```

## Constraints

- Must use existing `JohariQuadrant` enum (Open, Hidden, Blind, Unknown)
- Quadrant weights must always sum to 1.0 per embedder
- Transition probabilities must be valid (rows sum to 1.0)
- Compact encoding uses 2 bits per quadrant (4 possible values)
- Entropy/coherence thresholds from constitution.yaml (0.5)

## Notes

The JohariFingerprint enables powerful queries like:
- "Find memories where I'm Open(E1) but Blind(E5)" - semantic understanding without causal insight
- "Find memories with high Unknown(E11)" - frontier entity knowledge
- "Track transition from Hidden to Open" - knowledge becoming conscious

This is required by TASK-F002 (TeleologicalFingerprint). Can be developed in parallel with TASK-F001.

Reference implementation in TECH-SPEC-001 Section 1.3 (TS-103).
