# Task: TASK-P5-003b - TemporalEnrichmentProvider for Priority 5 Badges

```xml
<task_spec id="TASK-P5-003b" version="1.0">
<metadata>
  <title>TemporalEnrichmentProvider for Priority 5 Badges</title>
  <phase>5</phase>
  <sequence>38.5</sequence>
  <layer>logic</layer>
  <estimated_loc>100</estimated_loc>
  <dependencies>
    <dependency task="TASK-P2-001">TeleologicalArray with temporal embeddings</dependency>
    <dependency task="TASK-P5-001">InjectionCandidate type</dependency>
  </dependencies>
  <produces>
    <artifact type="struct">TemporalEnrichmentProvider</artifact>
    <artifact type="struct">TemporalBadge</artifact>
    <artifact type="enum">TemporalBadgeType</artifact>
  </produces>
</metadata>

<context>
  <background>
    While temporal embedders (E2, E3, E4) are excluded from topic detection and
    divergence alerts, they provide valuable context enrichment metadata. This
    component computes temporal badges for injection Priority 5 ("Temporal Enrichment").

    Temporal embedders capture:
    - E2 (Temporal-Recent): How recently something occurred
    - E3 (Temporal-Periodic): Cyclical patterns (daily, weekly)
    - E4 (Temporal-Position): Sequence/order relationships
  </background>
  <business_value>
    Enables context-aware badges like "From same session" and "You worked on X
    around this time" without incorrectly treating temporal proximity as semantic
    similarity.
  </business_value>
  <technical_context>
    Temporal embedders are stored in the TeleologicalArray but have weight 0.0 in
    topic detection. This component uses them exclusively for metadata enrichment
    in the injection pipeline's Priority 5 slot (~50 token budget).
  </technical_context>
</context>

<prerequisites>
  <prerequisite type="code">crates/context-graph-core/src/embedding/teleological.rs with E2, E3, E4</prerequisite>
  <prerequisite type="code">crates/context-graph-core/src/injection/candidate.rs</prerequisite>
</prerequisites>

<scope>
  <includes>
    <item>TemporalBadgeType enum (SameSession, SameDay, SamePeriod, SameSequence)</item>
    <item>TemporalBadge struct with badge type and display text</item>
    <item>TemporalEnrichmentProvider::compute_badges() method</item>
    <item>Similarity thresholds for each temporal embedding space</item>
    <item>Unit tests for badge computation</item>
  </includes>
  <excludes>
    <item>Topic detection (temporal embedders have weight 0.0)</item>
    <item>Divergence detection (temporal embedders excluded)</item>
    <item>Relevance scoring (temporal embedders excluded)</item>
    <item>Rendering badges to string (handled by ContextFormatter)</item>
  </excludes>
</scope>

<definition_of_done>
  <criterion id="DOD-1">
    <description>TemporalEnrichmentProvider computes badges based on E2, E3, E4 similarity</description>
    <verification>cargo build --package context-graph-core</verification>
  </criterion>
  <criterion id="DOD-2">
    <description>Badges computed only when similarity exceeds threshold</description>
    <verification>Unit tests verify thresholds</verification>
  </criterion>
  <criterion id="DOD-3">
    <description>No contribution to topic detection or relevance scoring</description>
    <verification>Code review confirms temporal embedders isolated to badges only</verification>
  </criterion>

  <signatures>
    <signature name="TemporalBadgeType">
      <code>
/// Type of temporal enrichment badge.
/// These are metadata-only and do NOT affect relevance or topic detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalBadgeType {
    /// From same session (E2 similarity > 0.8)
    SameSession,
    /// From same day/period (E3 similarity > 0.7)
    SameDay,
    /// From same time period/cycle (E3 similarity > 0.6)
    SamePeriod,
    /// In same sequence/order (E4 similarity > 0.6)
    SameSequence,
}
      </code>
    </signature>
    <signature name="TemporalBadge">
      <code>
/// A temporal enrichment badge for context injection.
/// These appear in Priority 5 slot (~50 tokens) and provide
/// contextual metadata without affecting relevance scoring.
#[derive(Debug, Clone)]
pub struct TemporalBadge {
    pub badge_type: TemporalBadgeType,
    pub similarity: f32,
}

impl TemporalBadge {
    pub fn display_emoji(&self) -> &'static str;
    pub fn display_text(&self) -> &'static str;
}
      </code>
    </signature>
    <signature name="TemporalEnrichmentProvider">
      <code>
/// Computes temporal enrichment badges from E2, E3, E4 embeddings.
/// These badges provide contextual metadata without affecting
/// topic detection or relevance scoring.
pub struct TemporalEnrichmentProvider {
    /// E2 (Temporal-Recent) similarity threshold for "Same Session"
    same_session_threshold: f32,  // default 0.8
    /// E3 (Temporal-Periodic) similarity threshold for "Same Day"
    same_day_threshold: f32,      // default 0.7
    /// E3 (Temporal-Periodic) similarity threshold for "Same Period"
    same_period_threshold: f32,   // default 0.6
    /// E4 (Temporal-Position) similarity threshold for "Same Sequence"
    same_sequence_threshold: f32, // default 0.6
}

impl TemporalEnrichmentProvider {
    pub fn new() -> Self;
    pub fn with_thresholds(session: f32, day: f32, period: f32, sequence: f32) -> Self;

    /// Compute temporal badges for a candidate memory relative to current context.
    /// Uses E2 (recent), E3 (periodic), E4 (positional) embeddings only.
    pub fn compute_badges(
        &self,
        current: &TeleologicalArray,
        candidate: &TeleologicalArray,
    ) -> Vec<TemporalBadge>;
}
      </code>
    </signature>
  </signatures>

  <constraints>
    <constraint type="exclusion">Temporal embedders (E2, E3, E4) MUST NOT contribute to topic detection</constraint>
    <constraint type="exclusion">Temporal embedders MUST NOT contribute to relevance scoring</constraint>
    <constraint type="exclusion">Temporal embedders MUST NOT trigger divergence alerts</constraint>
    <constraint type="threshold">SameSession: E2 similarity > 0.8</constraint>
    <constraint type="threshold">SameDay: E3 similarity > 0.7</constraint>
    <constraint type="threshold">SamePeriod: E3 similarity > 0.6 (but < 0.7)</constraint>
    <constraint type="threshold">SameSequence: E4 similarity > 0.6</constraint>
    <constraint type="budget">Priority 5 badges budget: ~50 tokens</constraint>
  </constraints>
</definition_of_done>

<pseudo_code>
```rust
// crates/context-graph-core/src/injection/temporal_enrichment.rs

use crate::embedding::TeleologicalArray;

/// Default thresholds for temporal badges
const DEFAULT_SAME_SESSION_THRESHOLD: f32 = 0.8;   // E2
const DEFAULT_SAME_DAY_THRESHOLD: f32 = 0.7;       // E3
const DEFAULT_SAME_PERIOD_THRESHOLD: f32 = 0.6;    // E3
const DEFAULT_SAME_SEQUENCE_THRESHOLD: f32 = 0.6;  // E4

/// Type of temporal enrichment badge.
/// These are metadata-only and do NOT affect relevance or topic detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalBadgeType {
    /// From same session (E2 Temporal-Recent similarity > 0.8)
    SameSession,
    /// From same day (E3 Temporal-Periodic similarity > 0.7)
    SameDay,
    /// From same time period/cycle (E3 similarity > 0.6)
    SamePeriod,
    /// In same sequence/order (E4 Temporal-Position similarity > 0.6)
    SameSequence,
}

impl TemporalBadgeType {
    pub fn emoji(&self) -> &'static str {
        match self {
            Self::SameSession => "📅",
            Self::SameDay => "🕐",
            Self::SamePeriod => "🔄",
            Self::SameSequence => "📊",
        }
    }

    pub fn display_text(&self) -> &'static str {
        match self {
            Self::SameSession => "From same session",
            Self::SameDay => "From same day",
            Self::SamePeriod => "From similar time period",
            Self::SameSequence => "In same sequence",
        }
    }
}

/// A temporal enrichment badge for context injection.
#[derive(Debug, Clone)]
pub struct TemporalBadge {
    pub badge_type: TemporalBadgeType,
    pub similarity: f32,
}

impl TemporalBadge {
    pub fn new(badge_type: TemporalBadgeType, similarity: f32) -> Self {
        Self { badge_type, similarity }
    }

    pub fn display_emoji(&self) -> &'static str {
        self.badge_type.emoji()
    }

    pub fn display_text(&self) -> &'static str {
        self.badge_type.display_text()
    }
}

/// Computes temporal enrichment badges from E2, E3, E4 embeddings.
///
/// IMPORTANT: These badges are metadata-only for Priority 5 injection.
/// Temporal embedders have weight 0.0 in topic detection and relevance scoring.
pub struct TemporalEnrichmentProvider {
    same_session_threshold: f32,
    same_day_threshold: f32,
    same_period_threshold: f32,
    same_sequence_threshold: f32,
}

impl Default for TemporalEnrichmentProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalEnrichmentProvider {
    pub fn new() -> Self {
        Self {
            same_session_threshold: DEFAULT_SAME_SESSION_THRESHOLD,
            same_day_threshold: DEFAULT_SAME_DAY_THRESHOLD,
            same_period_threshold: DEFAULT_SAME_PERIOD_THRESHOLD,
            same_sequence_threshold: DEFAULT_SAME_SEQUENCE_THRESHOLD,
        }
    }

    pub fn with_thresholds(
        session: f32,
        day: f32,
        period: f32,
        sequence: f32,
    ) -> Self {
        Self {
            same_session_threshold: session,
            same_day_threshold: day,
            same_period_threshold: period,
            same_sequence_threshold: sequence,
        }
    }

    /// Compute temporal badges for a candidate memory relative to current context.
    ///
    /// Uses ONLY temporal embedders:
    /// - E2 (Temporal-Recent): "Same Session" badge
    /// - E3 (Temporal-Periodic): "Same Day" or "Same Period" badge
    /// - E4 (Temporal-Position): "Same Sequence" badge
    pub fn compute_badges(
        &self,
        current: &TeleologicalArray,
        candidate: &TeleologicalArray,
    ) -> Vec<TemporalBadge> {
        let mut badges = Vec::new();

        // E2: Temporal-Recent -> Same Session
        let e2_similarity = cosine_similarity(
            current.e2_temp_recent.as_slice(),
            candidate.e2_temp_recent.as_slice(),
        );
        if e2_similarity > self.same_session_threshold {
            badges.push(TemporalBadge::new(TemporalBadgeType::SameSession, e2_similarity));
        }

        // E3: Temporal-Periodic -> Same Day or Same Period
        let e3_similarity = cosine_similarity(
            current.e3_temp_periodic.as_slice(),
            candidate.e3_temp_periodic.as_slice(),
        );
        if e3_similarity > self.same_day_threshold {
            badges.push(TemporalBadge::new(TemporalBadgeType::SameDay, e3_similarity));
        } else if e3_similarity > self.same_period_threshold {
            badges.push(TemporalBadge::new(TemporalBadgeType::SamePeriod, e3_similarity));
        }

        // E4: Temporal-Position -> Same Sequence
        let e4_similarity = cosine_similarity(
            current.e4_temp_position.as_slice(),
            candidate.e4_temp_position.as_slice(),
        );
        if e4_similarity > self.same_sequence_threshold {
            badges.push(TemporalBadge::new(TemporalBadgeType::SameSequence, e4_similarity));
        }

        badges
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_array_with_temporal(e2: &[f32], e3: &[f32], e4: &[f32]) -> TeleologicalArray {
        // Create a test TeleologicalArray with specified temporal embeddings
        let mut array = TeleologicalArray::default();
        array.e2_temp_recent = e2.to_vec().into();
        array.e3_temp_periodic = e3.to_vec().into();
        array.e4_temp_position = e4.to_vec().into();
        array
    }

    #[test]
    fn test_same_session_badge() {
        let provider = TemporalEnrichmentProvider::new();

        let current = make_test_array_with_temporal(
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
        );

        // Very similar E2 -> Same Session
        let candidate = make_test_array_with_temporal(
            &[0.95, 0.05, 0.0],  // High E2 similarity
            &[0.0, 0.0, 1.0],    // Low E3 similarity
            &[1.0, 0.0, 0.0],    // Low E4 similarity
        );

        let badges = provider.compute_badges(&current, &candidate);

        assert!(badges.iter().any(|b| b.badge_type == TemporalBadgeType::SameSession));
        assert!(!badges.iter().any(|b| b.badge_type == TemporalBadgeType::SameDay));
    }

    #[test]
    fn test_same_day_badge() {
        let provider = TemporalEnrichmentProvider::new();

        let current = make_test_array_with_temporal(
            &[1.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0],
        );

        // High E3 similarity -> Same Day
        let candidate = make_test_array_with_temporal(
            &[0.0, 1.0, 0.0],    // Low E2 similarity
            &[0.9, 0.1, 0.0],    // High E3 similarity
            &[1.0, 0.0, 0.0],    // Low E4 similarity
        );

        let badges = provider.compute_badges(&current, &candidate);

        assert!(badges.iter().any(|b| b.badge_type == TemporalBadgeType::SameDay));
    }

    #[test]
    fn test_no_badges_below_threshold() {
        let provider = TemporalEnrichmentProvider::new();

        let current = make_test_array_with_temporal(
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
        );

        // All temporal similarities below threshold
        let candidate = make_test_array_with_temporal(
            &[0.0, 1.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
        );

        let badges = provider.compute_badges(&current, &candidate);

        assert!(badges.is_empty());
    }

    #[test]
    fn test_badge_display() {
        let badge = TemporalBadge::new(TemporalBadgeType::SameSession, 0.9);

        assert_eq!(badge.display_emoji(), "📅");
        assert_eq!(badge.display_text(), "From same session");
    }
}
```
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/injection/temporal_enrichment.rs">
    TemporalEnrichmentProvider, TemporalBadge, TemporalBadgeType
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/injection/mod.rs">
    Add pub mod temporal_enrichment; pub use temporal_enrichment::*;
  </file>
</files_to_modify>

<validation_criteria>
  <criterion type="compilation">cargo build --package context-graph-core compiles without errors</criterion>
  <criterion type="test">cargo test injection::temporal_enrichment::tests -- all tests pass</criterion>
  <criterion type="isolation">Temporal embedders only used for badges, not relevance/topics</criterion>
</validation_criteria>

<test_commands>
  <command>cargo build --package context-graph-core</command>
  <command>cargo test injection::temporal_enrichment::tests --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create temporal_enrichment.rs
- [ ] Define TemporalBadgeType enum (SameSession, SameDay, SamePeriod, SameSequence)
- [ ] Implement TemporalBadge struct with display methods
- [ ] Implement TemporalEnrichmentProvider with configurable thresholds
- [ ] Implement compute_badges() using E2, E3, E4 similarities
- [ ] Verify temporal embedders NOT used for relevance/topic detection
- [ ] Add to injection/mod.rs exports
- [ ] Write unit tests for badge computation
- [ ] Write unit tests for threshold boundaries
- [ ] Run tests to verify
- [ ] Proceed to TASK-P5-004
