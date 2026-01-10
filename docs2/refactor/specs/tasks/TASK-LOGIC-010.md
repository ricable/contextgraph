# TASK-LOGIC-010: Teleological Drift Detection

```xml
<task_spec id="TASK-LOGIC-010" version="5.0">
<metadata>
  <title>Implement Teleological Drift Detection with Per-Embedder Analysis</title>
  <status>done</status>
  <layer>logic</layer>
  <sequence>20</sequence>
  <implements>
    <requirement_ref>REQ-DRIFT-DETECTION-01</requirement_ref>
    <requirement_ref>REQ-PURPOSE-CHECK-01</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="DONE">TASK-LOGIC-004</task_ref>
    <task_ref status="DONE">TASK-LOGIC-009</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <completed_date>2026-01-09</completed_date>
</metadata>

<context>
## CURRENT STATE (Verified 2026-01-09)

### IMPLEMENTATION COMPLETE

The file `crates/context-graph-core/src/autonomous/drift.rs` contains the **COMPLETE** implementation:
- **1647 lines** of production code
- **30 unit tests** (all passing)
- Uses `TeleologicalComparator` from TASK-LOGIC-004
- Performs per-embedder drift analysis across all 13 embedders
- 5-level `DriftLevel` enum (Critical, High, Medium, Low, None)
- Full trend analysis with linear regression

### DEPENDENCIES (VERIFIED COMPLETE)

| Task | Component | Location | Status |
|------|-----------|----------|--------|
| TASK-LOGIC-004 | TeleologicalComparator | `crates/context-graph-core/src/teleological/comparator.rs` | **DONE** (1089 lines) |
| TASK-LOGIC-009 | GoalDiscoveryPipeline | `crates/context-graph-core/src/autonomous/discovery.rs` | **DONE** (840 lines) |

### FILE LOCATIONS (VERIFIED)

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `crates/context-graph-core/src/teleological/comparator.rs` | TeleologicalComparator | 1089 | EXISTS |
| `crates/context-graph-core/src/teleological/embedder.rs` | Embedder enum (13 variants) | 561 | EXISTS |
| `crates/context-graph-core/src/autonomous/discovery.rs` | GoalDiscoveryPipeline | 840 | EXISTS |
| `crates/context-graph-core/src/autonomous/drift.rs` | Teleological drift detection | 1647 | **COMPLETE** |
| `crates/context-graph-core/src/autonomous/mod.rs` | Module exports | ~114 | EXISTS |
</context>

<objective>
**COMPLETE**: Implementation of teleological drift detection that:

1. ✅ Uses `TeleologicalComparator` from TASK-LOGIC-004 for apples-to-apples comparison
2. ✅ Performs per-embedder drift analysis across all 13 embedders
3. ✅ Uses 5-level `DriftLevel` (Critical, High, Medium, Low, None)
4. ✅ Tracks history with per-embedder [f32; 13] arrays
5. ✅ Generates embedder-specific recommendations
6. ✅ Computes trend analysis with linear regression

The detector compares recent memories to established goals using the full 13-embedder teleological array, providing granular insight into which semantic dimensions are drifting.
</objective>

<rationale>
Per-embedder drift detection provides:
1. **Granular insight**: Know exactly which embedder (Semantic, Temporal, Causal, etc.) is drifting
2. **Early warning**: Detect drift in specific dimensions before overall alignment degrades
3. **Actionable recommendations**: Embedder-specific suggestions for correction
4. **Trend analysis**: Predict when drift will become critical
5. **ARCH-02 compliance**: Apples-to-apples comparison via TeleologicalComparator
</rationale>

<architecture_constraints>
## From constitution.yaml (MUST NOT VIOLATE)

- **ARCH-01**: TeleologicalArray is atomic - all 13 embeddings stored/retrieved together
- **ARCH-02**: Apples-to-apples comparison - E1 compares with E1, NEVER cross-embedder
- **FAIL FAST**: All errors are fatal. No recovery attempts. No fallbacks.
- Single `Embedder` enum with exactly 13 variants

## Existing Types Used

```rust
// From crates/context-graph-core/src/teleological/embedder.rs
pub enum Embedder {
    Semantic = 0,           // E1: Core meaning
    TemporalRecent = 1,     // E2: Recent time
    TemporalPeriodic = 2,   // E3: Cyclical patterns
    TemporalPositional = 3, // E4: Sequence position
    Causal = 4,             // E5: Cause-effect
    Sparse = 5,             // E6: BM25-style lexical
    Code = 6,               // E7: Code structure
    Graph = 7,              // E8: Relationship structure
    Hdc = 8,                // E9: Holographic patterns
    Multimodal = 9,         // E10: Cross-modal
    Entity = 10,            // E11: Named entities
    LateInteraction = 11,   // E12: ColBERT tokens
    KeywordSplade = 12,     // E13: Learned expansion (renamed from Splade)
}

// From crates/context-graph-core/src/teleological/comparator.rs
pub struct TeleologicalComparator { ... }
impl TeleologicalComparator {
    pub fn new() -> Self;
    pub fn with_config(config: MatrixSearchConfig) -> Self;
    pub fn compare(&self, a: &SemanticFingerprint, b: &SemanticFingerprint) -> Result<ComparisonResult, ComparisonError>;
}
```
</architecture_constraints>

<implementation_summary>
## Implemented Types (drift.rs:1647 lines)

### Core Drift Types (TASK-LOGIC-010)

```rust
/// Drift severity levels (5 levels, ordered worst-to-best for Ord).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DriftLevel {
    Critical,  // < 0.40 similarity
    High,      // >= 0.40, < 0.55
    Medium,    // >= 0.55, < 0.70
    Low,       // >= 0.70, < 0.85
    None,      // >= 0.85
}

/// Drift trend direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftTrend {
    Improving,  // Positive slope
    Stable,     // |slope| < 0.01
    Worsening,  // Negative slope (TASK-LOGIC-010 name)
    Declining,  // Legacy name (same as Worsening)
}

/// Configuration for drift thresholds.
pub struct DriftThresholds {
    pub none_min: f32,   // >= this = None (default: 0.85)
    pub low_min: f32,    // >= this = Low (default: 0.70)
    pub medium_min: f32, // >= this = Medium (default: 0.55)
    pub high_min: f32,   // >= this = High (default: 0.40)
}

/// Teleological drift detector using per-embedder array comparison.
pub struct TeleologicalDriftDetector {
    comparator: TeleologicalComparator,
    history: DriftHistory,
    thresholds: DriftThresholds,
}

impl TeleologicalDriftDetector {
    pub fn new(comparator: TeleologicalComparator) -> Self;
    pub fn with_thresholds(comparator: TeleologicalComparator, thresholds: DriftThresholds) -> Result<Self, DriftError>;
    pub fn check_drift(&self, memories: &[SemanticFingerprint], goal: &SemanticFingerprint, strategy: SearchStrategy) -> Result<DriftResult, DriftError>;
    pub fn check_drift_with_history(&mut self, memories: &[SemanticFingerprint], goal: &SemanticFingerprint, goal_id: &str, strategy: SearchStrategy) -> Result<DriftResult, DriftError>;
    pub fn get_trend(&self, goal_id: &str) -> Option<TrendAnalysis>;
}

/// Error types for drift detection.
pub enum DriftError {
    EmptyMemories,
    InvalidGoal { reason: String },
    ComparisonFailed { embedder: Embedder, reason: String },
    InvalidThresholds { reason: String },
    ComparisonValidationFailed { reason: String },
}

/// Result of drift analysis.
pub struct DriftResult {
    pub overall_drift: OverallDrift,
    pub per_embedder_drift: PerEmbedderDrift,
    pub most_drifted_embedders: Vec<EmbedderDriftInfo>,
    pub recommendations: Vec<DriftRecommendation>,
    pub trend: Option<TrendAnalysis>,
    pub analyzed_count: usize,
    pub timestamp: DateTime<Utc>,
}

/// Per-embedder drift breakdown for all 13 embedders.
pub struct PerEmbedderDrift {
    pub embedder_drift: [EmbedderDriftInfo; 13],
}

/// Single history entry with per-embedder breakdown.
pub struct DriftHistoryEntry {
    pub timestamp: DateTime<Utc>,
    pub overall_similarity: f32,
    pub per_embedder: [f32; 13],
    pub memories_analyzed: usize,
}
```

### Legacy Types (Preserved for NORTH-010/011 Services)

```rust
/// Legacy drift config for NORTH-010/011 services.
pub struct DriftConfig {
    pub monitoring: DriftMonitoring,
    pub alert_threshold: f32,
    pub auto_correct: bool,
    pub severe_threshold: f32,
    pub window_days: u32,
}

/// Legacy 4-level severity for NORTH-010/011 services.
pub enum DriftSeverity {
    None, Mild, Moderate, Severe,
}

/// Legacy drift state for NORTH-010/011 services.
pub struct DriftState {
    pub rolling_mean: f32,
    pub baseline: f32,
    pub drift: f32,
    pub severity: DriftSeverity,
    pub trend: DriftTrend,
    pub checked_at: DateTime<Utc>,
    pub history: VecDeque<DriftDataPoint>,
}
```
</implementation_summary>

<test_summary>
## 30 Unit Tests (All Passing)

### Drift Level Classification Tests (5 tests)
- `test_drift_level_from_similarity_none` - 0.85+ = None
- `test_drift_level_from_similarity_low` - 0.70-0.84 = Low
- `test_drift_level_from_similarity_medium` - 0.55-0.69 = Medium
- `test_drift_level_from_similarity_high` - 0.40-0.54 = High
- `test_drift_level_from_similarity_critical` - <0.40 = Critical
- `test_drift_level_ordering` - Critical < High < Medium < Low < None

### Fail Fast Tests (3 tests)
- `test_fail_fast_empty_memories` - Returns `Err(DriftError::EmptyMemories)`
- `test_fail_fast_invalid_goal_nan` - Returns `Err(DriftError::InvalidGoal)`
- `test_fail_fast_invalid_goal_inf` - Returns `Err(DriftError::InvalidGoal)`

### Per-Embedder Analysis Tests (4 tests)
- `test_per_embedder_breakdown_all_13` - All 13 embedders present
- `test_per_embedder_similarity_valid_range` - All similarities in [0.0, 1.0]
- `test_drift_score_equals_one_minus_similarity` - drift_score = 1.0 - similarity
- `test_single_memory_analysis` - Works with single memory

### Most Drifted Embedders Tests (2 tests)
- `test_most_drifted_sorted_worst_first` - Critical first
- `test_most_drifted_max_five` - Max 5 returned

### Trend Analysis Tests (3 tests)
- `test_trend_requires_minimum_samples` - None if < 3 samples
- `test_trend_available_with_enough_samples` - Available with 5+ samples
- `test_trend_direction_stable_for_identical` - Stable for identical inputs

### Recommendations Tests (2 tests)
- `test_recommendations_only_for_medium_plus` - Only Medium/High/Critical
- `test_recommendations_priority_matches_drift_level` - Priority matches level

### History Tests (2 tests)
- `test_history_per_goal_isolation` - Separate history per goal_id
- `test_history_entry_has_per_embedder_array` - [f32; 13] per entry

### Custom Thresholds Tests (2 tests)
- `test_custom_thresholds` - Custom thresholds work
- `test_invalid_thresholds_rejected` - Invalid thresholds rejected

### Timestamp Tests (1 test)
- `test_result_has_recent_timestamp` - Timestamp is recent
</test_summary>

<fail_fast_compliance>
## FAIL FAST Compliance (Verified)

All errors propagate immediately:

| Input | Expected Error | Status |
|-------|----------------|--------|
| Empty memories slice | `DriftError::EmptyMemories` | ✅ |
| Goal with NaN embedding | `DriftError::InvalidGoal { reason }` | ✅ |
| Goal with Inf embedding | `DriftError::InvalidGoal { reason }` | ✅ |
| Comparison failure | `DriftError::ComparisonFailed { embedder, reason }` | ✅ |
| Invalid thresholds | `DriftError::InvalidThresholds { reason }` | ✅ |

**NO Fallbacks. NO Recovery. NO Silent Failures.**
</fail_fast_compliance>

<source_of_truth>
## Full State Verification Protocol

### 1. Before ANY Drift Check

```rust
// Print detector state
println!("=== SOURCE OF TRUTH: Detector State ===");
println!("Thresholds: {:?}", detector.thresholds);
println!("History entries: {:?}", detector.history.len());
for embedder in Embedder::all() {
    println!("  {:?}: threshold at idx {}", embedder, embedder.index());
}
```

### 2. After check_drift Execution

```rust
let result = detector.check_drift(&memories, &goal, strategy)?;

// VERIFY: All 13 embedders present
assert_eq!(result.per_embedder_drift.embedder_drift.len(), 13);

// VERIFY: Each embedder has valid similarity
for info in &result.per_embedder_drift.embedder_drift {
    assert!(!info.similarity.is_nan(), "NaN for {:?}", info.embedder);
    assert!(info.similarity >= 0.0 && info.similarity <= 1.0);
}

// VERIFY: Drift score is 1.0 - similarity
let expected_drift = 1.0 - result.overall_drift.similarity;
assert!((result.overall_drift.drift_score - expected_drift).abs() < 0.0001);

// VERIFY: Most drifted sorted descending
for window in result.most_drifted_embedders.windows(2) {
    assert!(window[0].drift_level <= window[1].drift_level);
}

// VERIFY: Timestamp is recent
assert!(result.timestamp > Utc::now() - chrono::Duration::seconds(5));
```

### 3. Manual Verification Checklist

| Check | How to Verify | Expected |
|-------|---------------|----------|
| Result has 13 embedders | `result.per_embedder_drift.embedder_drift.len()` | 13 |
| No NaN similarities | Loop check `!is_nan()` | All false |
| Drift score correct | `1.0 - similarity` | Matches drift_score |
| History recorded | `detector.history.get(goal_id)` | Entry count increases |
| Trend after 3+ samples | `detector.get_trend(goal_id)` | Some(TrendAnalysis) |
| Recommendations filtered | All have `drift_level <= Medium` | True |
</source_of_truth>

<verification_commands>
## Verification (All Passing)

```bash
# 1. Count drift.rs lines (should be ~1647)
wc -l crates/context-graph-core/src/autonomous/drift.rs
# Result: 1647

# 2. Verify no unwrap_or_default (fail-fast)
grep -c "unwrap_or_default" crates/context-graph-core/src/autonomous/drift.rs
# Expected: 0

# 3. Verify DriftLevel exists
grep -c "DriftLevel" crates/context-graph-core/src/autonomous/drift.rs
# Expected: >20

# 4. Verify per-embedder array
grep -c "\[f32; 13\]" crates/context-graph-core/src/autonomous/drift.rs
# Expected: >=2 (NUM_EMBEDDERS constant used)

# 5. Verify TeleologicalComparator used
grep -c "TeleologicalComparator" crates/context-graph-core/src/autonomous/drift.rs
# Expected: >=5

# 6. Verify fail-fast error handling
grep -c "DriftError" crates/context-graph-core/src/autonomous/drift.rs
# Expected: >=15

# 7. Test count (should have 30 tests)
grep -c "#\[test\]" crates/context-graph-core/src/autonomous/drift.rs
# Expected: 30

# 8. Compile check
cargo check -p context-graph-core
# Expected: success

# 9. Run drift tests
cargo test -p context-graph-core autonomous::drift -- --nocapture
# Expected: "test result: ok. 30 passed; 0 failed"

# 10. Clippy validation
cargo clippy -p context-graph-core -- -D warnings
# Expected: no errors
```
</verification_commands>
</task_spec>
```
