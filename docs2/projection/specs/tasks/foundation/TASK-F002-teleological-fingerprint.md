# Task: TASK-F002 - Implement TeleologicalFingerprint Struct

## Metadata
- **ID**: TASK-F002
- **Layer**: Foundation
- **Priority**: P0 (Critical Path)
- **Estimated Effort**: L (Large)
- **Dependencies**: TASK-F001, TASK-F003
- **Traces To**: TS-102, FR-201, FR-202, FR-203, FR-204

## Description

Implement the `TeleologicalFingerprint` struct that wraps `SemanticFingerprint` with purpose-aware metadata enabling goal-aligned retrieval. This is the complete node representation combining:

1. **SemanticFingerprint**: The raw 12-embedding array (from TASK-F001)
2. **PurposeVector**: 12D alignment signature to North Star goal
3. **JohariFingerprint**: Per-embedder awareness classification (from TASK-F003)
4. **Purpose Evolution**: Time-series tracking of alignment changes

The TeleologicalFingerprint enables queries like "find memories similar to X that serve the same purpose" rather than just "find memories similar to X".

## Acceptance Criteria

- [ ] `PurposeVector` struct with 12-element alignment array
- [ ] `AlignmentThreshold` enum (Optimal, Acceptable, Warning, Critical)
- [ ] `EvolutionTrigger` enum for tracking purpose changes
- [ ] `PurposeSnapshot` struct for time-series data
- [ ] `TeleologicalFingerprint` struct integrating all components
- [ ] North Star alignment computation methods
- [ ] Purpose evolution recording with snapshot limit
- [ ] Misalignment warning detection (delta_A < -0.15)
- [ ] Integration with existing `JohariQuadrant` from `types/johari/`
- [ ] Unit tests with realistic alignment data

## Implementation Steps

1. Create `crates/context-graph-core/src/types/fingerprint/purpose.rs`:
   - Define `NUM_EMBEDDERS = 12` constant
   - Implement `AlignmentThreshold` enum with `classify(theta)` method
   - Implement `PurposeVector` struct with alignment methods
2. Create `crates/context-graph-core/src/types/fingerprint/evolution.rs`:
   - Implement `EvolutionTrigger` enum (Created, Accessed, GoalChanged, Recalibration, MisalignmentDetected)
   - Implement `PurposeSnapshot` struct
3. Create `crates/context-graph-core/src/types/fingerprint/teleological.rs`:
   - Import `SemanticFingerprint`, `PurposeVector`, `JohariFingerprint`
   - Implement `TeleologicalFingerprint` with all fields
   - Implement `new()`, `record_snapshot()`, `compute_alignment_delta()`
   - Implement `check_misalignment_warning()`
4. Update `crates/context-graph-core/src/types/fingerprint/mod.rs` to export new types
5. Add tests in same files or separate test modules

## Files Affected

### Files to Create
- `crates/context-graph-core/src/types/fingerprint/purpose.rs` - PurposeVector and AlignmentThreshold
- `crates/context-graph-core/src/types/fingerprint/evolution.rs` - EvolutionTrigger and PurposeSnapshot
- `crates/context-graph-core/src/types/fingerprint/teleological.rs` - TeleologicalFingerprint

### Files to Modify
- `crates/context-graph-core/src/types/fingerprint/mod.rs` - Export new types
- `crates/context-graph-core/Cargo.toml` - Add `uuid` and `chrono` dependencies if not present

## Code Signature (Definition of Done)

```rust
// purpose.rs
pub const NUM_EMBEDDERS: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlignmentThreshold {
    Optimal,    // theta >= 0.75
    Acceptable, // theta in [0.70, 0.75)
    Warning,    // theta in [0.55, 0.70)
    Critical,   // theta < 0.55
}

impl AlignmentThreshold {
    pub fn classify(theta: f32) -> Self;
    pub fn is_misaligned(&self) -> bool;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurposeVector {
    pub alignments: [f32; NUM_EMBEDDERS],
    pub dominant_embedder: u8,
    pub coherence: f32,
    pub stability: f32,
}

impl PurposeVector {
    pub fn aggregate_alignment(&self) -> f32;
    pub fn threshold_status(&self) -> AlignmentThreshold;
    pub fn find_dominant(&self) -> u8;
    pub fn similarity(&self, other: &Self) -> f32;
}

// evolution.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvolutionTrigger {
    Created,
    Accessed { query_context: String },
    GoalChanged { old_goal: Uuid, new_goal: Uuid },
    Recalibration,
    MisalignmentDetected { delta_a: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurposeSnapshot {
    pub timestamp: DateTime<Utc>,
    pub purpose: PurposeVector,
    pub johari: JohariFingerprint,
    pub trigger: EvolutionTrigger,
}

// teleological.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeleologicalFingerprint {
    pub id: Uuid,
    pub semantic: SemanticFingerprint,
    pub purpose_vector: PurposeVector,
    pub johari: JohariFingerprint,
    pub purpose_evolution: Vec<PurposeSnapshot>,
    pub theta_to_north_star: f32,
    pub content_hash: [u8; 32],
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub access_count: u64,
}

impl TeleologicalFingerprint {
    pub const EXPECTED_SIZE_BYTES: usize = 46_000;
    pub const MAX_EVOLUTION_SNAPSHOTS: usize = 100;

    pub fn new(semantic: SemanticFingerprint, purpose_vector: PurposeVector,
               johari: JohariFingerprint, content_hash: [u8; 32]) -> Self;
    pub fn record_snapshot(&mut self, trigger: EvolutionTrigger);
    pub fn compute_alignment_delta(&self) -> f32;
    pub fn check_misalignment_warning(&self) -> Option<f32>;
    pub fn alignment_status(&self) -> AlignmentThreshold;
}
```

## Testing Requirements

### Unit Tests
- `test_alignment_threshold_classify_optimal` - theta >= 0.75 returns Optimal
- `test_alignment_threshold_classify_acceptable` - theta 0.70-0.75 returns Acceptable
- `test_alignment_threshold_classify_warning` - theta 0.55-0.70 returns Warning
- `test_alignment_threshold_classify_critical` - theta < 0.55 returns Critical
- `test_purpose_vector_aggregate` - Correct mean of 12 alignments
- `test_purpose_vector_similarity` - Cosine similarity in 12D
- `test_purpose_vector_find_dominant` - Returns highest alignment index
- `test_teleological_new` - Creates with initial snapshot
- `test_teleological_record_snapshot` - Adds to evolution, respects limit
- `test_teleological_alignment_delta` - Computes difference from previous
- `test_teleological_misalignment_warning` - Triggers at delta < -0.15

### Integration Tests
- Test with real SemanticFingerprint from TASK-F001
- Test with real JohariFingerprint from TASK-F003

## Verification

```bash
# Compile check
cargo check -p context-graph-core

# Run unit tests
cargo test -p context-graph-core teleological

# Run all fingerprint tests
cargo test -p context-graph-core fingerprint
```

## Constraints

- Alignment thresholds from Royse 2026 research (constitution.yaml)
- MAX_EVOLUTION_SNAPSHOTS = 100 (older snapshots go to TimescaleDB in production)
- Purpose vector is 12D - one alignment per embedder
- UUID v4 for fingerprint ID
- SHA-256 for content_hash (32 bytes)
- All timestamps in UTC

## Notes

This task depends on both TASK-F001 (SemanticFingerprint) and TASK-F003 (JohariFingerprint). The three together form the complete teleological data model.

Reference implementation in TECH-SPEC-001 Section 1.2 (TS-102).
