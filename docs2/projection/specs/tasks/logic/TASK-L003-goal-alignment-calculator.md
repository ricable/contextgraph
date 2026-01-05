# TASK-L003: Goal Alignment Calculator

```yaml
metadata:
  id: "TASK-L003"
  title: "Goal Alignment Calculator"
  layer: "logic"
  priority: "P0"
  estimated_hours: 6
  created: "2026-01-04"
  status: "pending"
  dependencies:
    - "TASK-L002"  # Purpose Vector Computation
    - "TASK-F002"  # TeleologicalFingerprint struct
  spec_refs:
    - "projectionplan1.md:goal-hierarchy"
    - "projectionplan2.md:alignment-scoring"
```

## Problem Statement

Implement a calculator that scores how well a memory (via its TeleologicalFingerprint) aligns with goals at each level of the hierarchy, enabling goal-aware retrieval and misalignment detection.

## Context

The Goal Alignment Calculator traverses the goal hierarchy (North Star -> Strategic -> Tactical -> Immediate) and computes alignment scores at each level. This enables:
- Multi-level alignment scoring
- Goal drift detection (aligned with tactical but not strategic)
- Misalignment flagging for review
- Priority-based retrieval ordering

## Technical Specification

### Data Structures

```rust
/// Alignment scores across the goal hierarchy
#[derive(Clone, Debug, PartialEq)]
pub struct GoalAlignmentScore {
    /// Alignment with North Star (top-level) goal
    pub north_star_alignment: f32,

    /// Alignment with strategic (mid-level) goals
    pub strategic_alignment: f32,

    /// Alignment with tactical (near-term) goals
    pub tactical_alignment: f32,

    /// Alignment with immediate context
    pub immediate_alignment: f32,

    /// Weighted composite score
    pub composite_score: f32,

    /// Detailed per-goal breakdown
    pub goal_breakdown: Vec<GoalScore>,

    /// Flags for misalignment patterns
    pub misalignment_flags: MisalignmentFlags,
}

/// Score for an individual goal
#[derive(Clone, Debug)]
pub struct GoalScore {
    pub goal_id: GoalId,
    pub goal_level: GoalLevel,
    pub alignment: f32,
    pub contributing_spaces: Vec<usize>,
}

/// Flags indicating misalignment patterns
#[derive(Clone, Debug, Default)]
pub struct MisalignmentFlags {
    /// Aligned with lower-level but not higher-level goals
    pub tactical_without_strategic: bool,

    /// Aligned with child goals but not parent
    pub divergent_hierarchy: bool,

    /// Below minimum threshold for all goals
    pub below_threshold: bool,

    /// High variance in alignment across spaces
    pub inconsistent_alignment: bool,
}

/// Configuration for alignment calculation
pub struct AlignmentConfig {
    /// Weights for each goal level in composite score
    pub level_weights: LevelWeights,

    /// Minimum alignment to consider "aligned"
    pub alignment_threshold: f32,

    /// Whether to flag misalignments
    pub detect_misalignment: bool,

    /// Space weights for alignment computation
    pub space_weights: Option<[f32; 12]>,
}

#[derive(Clone, Debug)]
pub struct LevelWeights {
    pub north_star: f32,
    pub strategic: f32,
    pub tactical: f32,
    pub immediate: f32,
}

impl Default for LevelWeights {
    fn default() -> Self {
        Self {
            north_star: 0.4,
            strategic: 0.3,
            tactical: 0.2,
            immediate: 0.1,
        }
    }
}
```

### Core Trait

```rust
/// Calculates alignment between memories and goals
#[async_trait]
pub trait GoalAlignmentCalculator: Send + Sync {
    /// Calculate alignment for a single fingerprint
    async fn calculate_alignment(
        &self,
        fingerprint: &TeleologicalFingerprint,
        goals: &[GoalNode],
        config: &AlignmentConfig,
    ) -> Result<GoalAlignmentScore, AlignmentError>;

    /// Batch calculate alignments
    async fn calculate_alignment_batch(
        &self,
        fingerprints: &[TeleologicalFingerprint],
        goals: &[GoalNode],
        config: &AlignmentConfig,
    ) -> Result<Vec<GoalAlignmentScore>, AlignmentError>;

    /// Find memories with specific alignment patterns
    async fn find_by_alignment(
        &self,
        pattern: AlignmentPattern,
        goals: &[GoalNode],
        limit: usize,
    ) -> Result<Vec<(MemoryId, GoalAlignmentScore)>, AlignmentError>;

    /// Detect goal drift for a memory over time
    async fn detect_goal_drift(
        &self,
        memory_id: MemoryId,
        time_range: TimeRange,
    ) -> Result<GoalDriftReport, AlignmentError>;
}

/// Pattern for querying alignment
pub enum AlignmentPattern {
    /// Well-aligned with North Star
    NorthStarAligned { min_score: f32 },

    /// Misaligned (tactical without strategic)
    Misaligned,

    /// High alignment variance
    Inconsistent { min_variance: f32 },

    /// Below threshold for all goals
    Unaligned { threshold: f32 },
}
```

### Computation Logic

```rust
impl GoalAlignmentCalculator for DefaultAlignmentCalculator {
    async fn calculate_alignment(
        &self,
        fingerprint: &TeleologicalFingerprint,
        goals: &[GoalNode],
        config: &AlignmentConfig,
    ) -> Result<GoalAlignmentScore, AlignmentError> {
        let purpose = &fingerprint.purpose_vector;
        let mut goal_breakdown = Vec::new();

        // Group goals by level
        let by_level = goals.iter()
            .fold(HashMap::new(), |mut acc, g| {
                acc.entry(g.level).or_insert_with(Vec::new).push(g);
                acc
            });

        // Calculate per-level alignment
        let north_star_alignment = self.calculate_level_alignment(
            purpose,
            by_level.get(&GoalLevel::NorthStar).unwrap_or(&vec![]),
            config,
            &mut goal_breakdown,
        )?;

        let strategic_alignment = self.calculate_level_alignment(
            purpose,
            by_level.get(&GoalLevel::Strategic).unwrap_or(&vec![]),
            config,
            &mut goal_breakdown,
        )?;

        let tactical_alignment = self.calculate_level_alignment(
            purpose,
            by_level.get(&GoalLevel::Tactical).unwrap_or(&vec![]),
            config,
            &mut goal_breakdown,
        )?;

        let immediate_alignment = self.calculate_level_alignment(
            purpose,
            by_level.get(&GoalLevel::Immediate).unwrap_or(&vec![]),
            config,
            &mut goal_breakdown,
        )?;

        // Compute composite score
        let weights = &config.level_weights;
        let composite_score =
            north_star_alignment * weights.north_star +
            strategic_alignment * weights.strategic +
            tactical_alignment * weights.tactical +
            immediate_alignment * weights.immediate;

        // Detect misalignment patterns
        let misalignment_flags = if config.detect_misalignment {
            self.detect_misalignment_patterns(
                north_star_alignment,
                strategic_alignment,
                tactical_alignment,
                immediate_alignment,
                config.alignment_threshold,
            )
        } else {
            MisalignmentFlags::default()
        };

        Ok(GoalAlignmentScore {
            north_star_alignment,
            strategic_alignment,
            tactical_alignment,
            immediate_alignment,
            composite_score,
            goal_breakdown,
            misalignment_flags,
        })
    }
}
```

## Implementation Requirements

### Prerequisites

- [ ] TASK-L002 complete (PurposeVector available)
- [ ] TASK-F002 complete (TeleologicalFingerprint with purpose field)

### Scope

#### In Scope

- Multi-level alignment calculation
- Composite score computation
- Misalignment detection
- Per-goal breakdown
- Batch processing

#### Out of Scope

- Goal hierarchy storage (TASK-F004)
- Alignment-based retrieval (TASK-L008)
- UI/API for goal management

### Constraints

- Calculation < 5ms per fingerprint
- Support 4 goal levels + N goals per level
- Thread-safe for concurrent calculations
- Deterministic for same inputs

## Pseudo Code

```
FUNCTION calculate_goal_alignment(fingerprint, goals, config):
    purpose = fingerprint.purpose_vector
    goal_breakdown = []

    // Group goals by level
    by_level = group_by(goals, g => g.level)

    // Calculate alignment per level
    FOR level IN [NorthStar, Strategic, Tactical, Immediate]:
        level_goals = by_level.get(level) OR []

        IF level_goals.is_empty():
            level_alignment = 0.0
        ELSE:
            // For each goal, compute alignment using purpose vector
            goal_alignments = []
            FOR goal IN level_goals:
                // Dot product of purpose vector with goal's space weights
                alignment = 0.0
                contributing_spaces = []
                FOR space_idx IN 0..12:
                    IF purpose.alignment[space_idx] > 0:
                        space_align = purpose.alignment[space_idx] * goal.space_weights[space_idx]
                        alignment += space_align
                        IF space_align > 0.1:
                            contributing_spaces.push(space_idx)

                goal_alignments.push((goal, alignment, contributing_spaces))
                goal_breakdown.push(GoalScore {
                    goal_id: goal.id,
                    goal_level: level,
                    alignment,
                    contributing_spaces
                })

            // Level alignment = weighted average of goal alignments
            level_alignment = weighted_average(goal_alignments)

        SET level_alignment_map[level] = level_alignment

    // Composite score
    composite =
        level_alignment_map[NorthStar] * config.level_weights.north_star +
        level_alignment_map[Strategic] * config.level_weights.strategic +
        level_alignment_map[Tactical] * config.level_weights.tactical +
        level_alignment_map[Immediate] * config.level_weights.immediate

    // Detect misalignment
    misalignment = detect_misalignment(level_alignment_map, config.threshold)

    RETURN GoalAlignmentScore {
        north_star_alignment: level_alignment_map[NorthStar],
        strategic_alignment: level_alignment_map[Strategic],
        tactical_alignment: level_alignment_map[Tactical],
        immediate_alignment: level_alignment_map[Immediate],
        composite_score: composite,
        goal_breakdown,
        misalignment_flags: misalignment
    }

FUNCTION detect_misalignment(levels, threshold):
    flags = MisalignmentFlags::default()

    // Tactical without strategic
    IF levels[Tactical] > threshold AND levels[Strategic] < threshold:
        flags.tactical_without_strategic = true

    // Divergent hierarchy (child aligned, parent not)
    IF levels[Strategic] > threshold AND levels[NorthStar] < threshold * 0.5:
        flags.divergent_hierarchy = true

    // Below threshold for all
    IF all(levels.values(), v => v < threshold):
        flags.below_threshold = true

    // Inconsistent (high variance)
    variance = compute_variance(levels.values())
    IF variance > 0.3:
        flags.inconsistent_alignment = true

    RETURN flags
```

## Definition of Done

### Implementation Checklist

- [ ] `GoalAlignmentScore` struct
- [ ] `MisalignmentFlags` struct
- [ ] `LevelWeights` configuration
- [ ] `GoalAlignmentCalculator` trait
- [ ] Default implementation
- [ ] Per-level alignment calculation
- [ ] Composite score computation
- [ ] Misalignment pattern detection
- [ ] Batch processing support

### Testing Requirements

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_alignment_calculation() {
        let calculator = DefaultAlignmentCalculator::new();
        let fingerprint = create_test_teleological_fingerprint();
        let goals = create_test_goal_hierarchy();
        let config = AlignmentConfig::default();

        let score = calculator.calculate_alignment(&fingerprint, &goals, &config).await.unwrap();

        assert!(score.composite_score >= 0.0 && score.composite_score <= 1.0);
        assert!(!score.goal_breakdown.is_empty());
    }

    #[tokio::test]
    async fn test_misalignment_detection() {
        let calculator = DefaultAlignmentCalculator::new();
        let fingerprint = create_misaligned_fingerprint(); // Tactical high, strategic low
        let goals = create_test_goal_hierarchy();
        let config = AlignmentConfig {
            detect_misalignment: true,
            alignment_threshold: 0.5,
            ..Default::default()
        };

        let score = calculator.calculate_alignment(&fingerprint, &goals, &config).await.unwrap();

        assert!(score.misalignment_flags.tactical_without_strategic);
    }

    #[tokio::test]
    async fn test_level_weights() {
        let calculator = DefaultAlignmentCalculator::new();
        let fingerprint = create_test_teleological_fingerprint();
        let goals = create_test_goal_hierarchy();

        let config1 = AlignmentConfig {
            level_weights: LevelWeights {
                north_star: 1.0,
                strategic: 0.0,
                tactical: 0.0,
                immediate: 0.0,
            },
            ..Default::default()
        };

        let config2 = AlignmentConfig {
            level_weights: LevelWeights {
                north_star: 0.0,
                strategic: 0.0,
                tactical: 0.0,
                immediate: 1.0,
            },
            ..Default::default()
        };

        let score1 = calculator.calculate_alignment(&fingerprint, &goals, &config1).await.unwrap();
        let score2 = calculator.calculate_alignment(&fingerprint, &goals, &config2).await.unwrap();

        // Different weights should produce different composite scores
        assert_ne!(score1.composite_score, score2.composite_score);
    }

    #[tokio::test]
    async fn test_batch_calculation() {
        let calculator = DefaultAlignmentCalculator::new();
        let fingerprints = vec![create_test_teleological_fingerprint(); 5];
        let goals = create_test_goal_hierarchy();
        let config = AlignmentConfig::default();

        let scores = calculator.calculate_alignment_batch(&fingerprints, &goals, &config).await.unwrap();

        assert_eq!(scores.len(), 5);
    }
}
```

### Verification Commands

```bash
# Run unit tests
cargo test -p context-graph-core goal_alignment

# Run with coverage
cargo llvm-cov -p context-graph-core --test goal_alignment

# Benchmark calculation
cargo bench -p context-graph-core -- alignment
```

## Files to Create

| File | Description |
|------|-------------|
| `crates/context-graph-core/src/alignment/mod.rs` | Alignment module |
| `crates/context-graph-core/src/alignment/calculator.rs` | GoalAlignmentCalculator trait and impl |
| `crates/context-graph-core/src/alignment/score.rs` | GoalAlignmentScore struct |
| `crates/context-graph-core/src/alignment/misalignment.rs` | Misalignment detection |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-core/src/lib.rs` | Add `pub mod alignment` |

## Traceability

| Requirement | Source | Coverage |
|-------------|--------|----------|
| Multi-level alignment | projectionplan1.md:goal-hierarchy | Complete |
| Composite scoring | projectionplan2.md:alignment | Complete |
| Misalignment detection | projectionplan2.md:drift | Complete |
| Level weights | projectionplan1.md:weights | Complete |

---

*Task created: 2026-01-04*
*Layer: Logic*
*Priority: P0 - Core goal alignment*
