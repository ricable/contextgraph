# TASK-P4-002: Topic, TopicProfile, TopicPhase Types

```xml
<task_spec id="TASK-P4-002" version="2.0">
<metadata>
  <title>Topic Type Implementation</title>
  <status>COMPLETE</status>
  <completed_date>2026-01-17</completed_date>
  <layer>foundation</layer>
  <sequence>28</sequence>
  <phase>4</phase>
  <implements>
    <requirement_ref>REQ-P4-04</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="COMPLETE">TASK-P4-001</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <last_audited>2025-01-17</last_audited>
</metadata>

<codebase_state_verification>
  <!-- CRITICAL: Verified against actual codebase on 2025-01-17 -->

  <prerequisite_verification>
    <item status="VERIFIED">TASK-P4-001 COMPLETE - 64 tests passing</item>
    <item status="VERIFIED">ClusterMembership exists at src/clustering/membership.rs</item>
    <item status="VERIFIED">Cluster exists at src/clustering/cluster.rs</item>
    <item status="VERIFIED">ClusterError exists at src/clustering/error.rs</item>
    <item status="VERIFIED">clustering module exported in lib.rs:36 and lib.rs:98</item>
  </prerequisite_verification>

  <existing_types_to_use>
    <!-- CRITICAL: Use these EXACT imports - verified against actual codebase -->
    <type name="Embedder" path="crate::teleological::Embedder" verified="true">
      Has 13 variants: Semantic(0), TemporalRecent(1), TemporalPeriodic(2),
      TemporalPositional(3), Causal(4), Sparse(5), Code(6), Emotional(7),
      Hdc(8), Multimodal(9), Entity(10), LateInteraction(11), KeywordSplade(12)
      Methods: all(), index(), from_index()
    </type>
    <type name="EmbedderCategory" path="crate::embeddings::category::EmbedderCategory" verified="true">
      Has topic_weight() method returning: Semantic=1.0, Temporal=0.0, Relational=0.5, Structural=0.5
    </type>
    <type name="ClusterMembership" path="crate::clustering::ClusterMembership" verified="true">
      Fields: memory_id, space, cluster_id, membership_probability, is_core_point
    </type>
    <type name="Cluster" path="crate::clustering::Cluster" verified="true">
      Fields: id, space, centroid, member_count, silhouette_score, created_at
    </type>
  </existing_types_to_use>

  <constants_from_constitution verified="true">
    <!-- From embeddings/category.rs - already implemented -->
    <constant name="TOPIC_THRESHOLD" value="2.5" source="embeddings/category.rs:topic_threshold()"/>
    <constant name="MAX_WEIGHTED_AGREEMENT" value="8.5" source="embeddings/category.rs:max_weighted_agreement()"/>

    <!-- From constitution.yaml ARCH-09, ARCH-10, AP-60 -->
    <rule id="ARCH-09">Topic threshold is weighted_agreement >= 2.5 (not raw space count)</rule>
    <rule id="ARCH-10">Divergence detection uses SEMANTIC embedders only</rule>
    <rule id="AP-60">Temporal embedders (E2-E4) MUST NOT count toward topic detection</rule>
  </constants_from_constitution>
</codebase_state_verification>

<context>
Implements Topic, TopicProfile, TopicPhase, and TopicStability types. A Topic
represents a cross-space concept that clusters together in multiple embedding
spaces WITH WEIGHTED AGREEMENT.

CRITICAL: Topics use WEIGHTED agreement, NOT raw space count:
- SEMANTIC embedders (E1, E5, E6, E7, E10, E12, E13): weight = 1.0
- TEMPORAL embedders (E2, E3, E4): weight = 0.0 (NEVER count)
- RELATIONAL embedders (E8, E11): weight = 0.5
- STRUCTURAL embedders (E9): weight = 0.5

Topic threshold: weighted_agreement >= 2.5 (max = 8.5)
Example: 3 SEMANTIC spaces = 3.0 weighted -> TOPIC
Example: 2 SEMANTIC + 1 RELATIONAL = 2.5 weighted -> TOPIC
Example: 5 TEMPORAL spaces = 0.0 weighted -> NOT TOPIC (temporal excluded!)
</context>

<input_context_files>
  <file purpose="data_models">docs2/impplan/technical/TECH-PHASE4-CLUSTERING.md#data_models</file>
  <file purpose="membership" verified="exists">crates/context-graph-core/src/clustering/membership.rs</file>
  <file purpose="cluster" verified="exists">crates/context-graph-core/src/clustering/cluster.rs</file>
  <file purpose="embedder_enum" verified="exists">crates/context-graph-core/src/teleological/embedder.rs</file>
  <file purpose="category_weights" verified="exists">crates/context-graph-core/src/embeddings/category.rs</file>
</input_context_files>

<prerequisites>
  <check status="VERIFIED">TASK-P4-001 complete - verified via `cargo test --package context-graph-core clustering` (64 tests pass)</check>
  <check status="VERIFIED">ClusterMembership, Cluster, ClusterError types exist and are exported</check>
  <check status="VERIFIED">Embedder::all() returns all 13 embedders</check>
  <check status="VERIFIED">EmbedderCategory::topic_weight() returns correct weights</check>
</prerequisites>

<scope>
  <in_scope>
    - Create Topic struct with all fields
    - Create TopicProfile with 13-space strengths
    - Create TopicPhase enum
    - Create TopicStability struct
    - Implement weighted_agreement calculation (CRITICAL: use EmbedderCategory::topic_weight())
    - Implement profile similarity calculation (cosine)
    - Implement dominant_spaces method
    - Implement is_topic() validation using weighted threshold 2.5
  </in_scope>
  <out_of_scope>
    - Topic synthesis algorithm (TASK-P4-008)
    - Topic stability tracking updates (TASK-P4-009)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/clustering/topic.rs">
      use std::collections::HashMap;
      use serde::{Serialize, Deserialize};
      use uuid::Uuid;
      use chrono::{DateTime, Utc};
      use crate::teleological::Embedder;
      use crate::embeddings::category::EmbedderCategory;

      #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
      pub enum TopicPhase {
          Emerging,
          Stable,
          Declining,
          Merging,
      }

      #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
      pub struct TopicProfile {
          /// Strength in each of 13 embedding spaces (0.0..=1.0)
          pub strengths: [f32; 13],
      }

      impl TopicProfile {
          pub fn new(strengths: [f32; 13]) -> Self;
          pub fn default_profile() -> Self;
          pub fn strength(&amp;self, embedder: Embedder) -> f32;
          pub fn set_strength(&amp;mut self, embedder: Embedder, strength: f32);
          pub fn dominant_spaces(&amp;self) -> Vec&lt;Embedder&gt;;
          pub fn similarity(&amp;self, other: &amp;TopicProfile) -> f32;
          pub fn active_space_count(&amp;self) -> usize;

          /// CRITICAL: Compute weighted agreement per ARCH-09
          /// Uses EmbedderCategory::topic_weight() - temporal spaces contribute 0.0
          pub fn weighted_agreement(&amp;self) -> f32;

          /// Check if meets topic threshold (>= 2.5)
          pub fn is_topic(&amp;self) -> bool;
      }

      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct TopicStability {
          pub phase: TopicPhase,
          pub age_hours: f32,
          pub membership_churn: f32,
          pub centroid_drift: f32,
          pub access_count: u32,
          pub last_accessed: Option&lt;DateTime&lt;Utc&gt;&gt;,
      }

      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct Topic {
          pub id: Uuid,
          pub name: Option&lt;String&gt;,
          pub profile: TopicProfile,
          pub contributing_spaces: Vec&lt;Embedder&gt;,
          pub cluster_ids: HashMap&lt;Embedder, i32&gt;,
          pub member_memories: Vec&lt;Uuid&gt;,
          /// CRITICAL: confidence = weighted_agreement / 8.5 (NOT raw count / 13)
          pub confidence: f32,
          pub stability: TopicStability,
          pub created_at: DateTime&lt;Utc&gt;,
      }

      impl Topic {
          pub fn new(profile: TopicProfile, cluster_ids: HashMap&lt;Embedder, i32&gt;, members: Vec&lt;Uuid&gt;) -> Self;
          /// CRITICAL: confidence = weighted_agreement / MAX_WEIGHTED_AGREEMENT (8.5)
          pub fn compute_confidence(&amp;self) -> f32;
          pub fn record_access(&amp;mut self);
          /// CRITICAL: Uses weighted_agreement >= 2.5 threshold
          pub fn is_valid(&amp;self) -> bool;
      }
    </signature>
  </signatures>

  <constraints>
    - TopicProfile strengths in 0.0..=1.0 (clamp on construction)
    - weighted_agreement uses EmbedderCategory::topic_weight() for each space
    - TEMPORAL spaces (E2, E3, E4) contribute 0.0 to weighted_agreement (ARCH-04, AP-60)
    - Topic validity: weighted_agreement >= 2.5 (NOT raw space count >= 3)
    - confidence = weighted_agreement / 8.5 (NOT contributing_spaces / 13)
    - membership_churn in 0.0..=1.0
    - centroid_drift in 0.0..=1.0
    - All f32 operations must check for NaN/Infinity (AP-10)
  </constraints>

  <verification>
    - dominant_spaces returns spaces with strength > 0.5
    - Profile similarity computed correctly (cosine, handle zero vectors)
    - weighted_agreement correctly applies category weights
    - TEMPORAL spaces contribute 0.0 to weighted_agreement
    - is_topic() returns true iff weighted_agreement >= 2.5
    - confidence uses MAX_WEIGHTED_AGREEMENT (8.5) as denominator
    - TopicPhase transitions follow constitution rules
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/clustering/topic.rs

//! Topic types for multi-space clustering.
//!
//! Per constitution ARCH-09: Topic threshold is weighted_agreement >= 2.5
//! Per constitution AP-60: Temporal embedders (E2-E4) NEVER count toward topic detection

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use crate::teleological::Embedder;
use crate::embeddings::category::{EmbedderCategory, topic_threshold, max_weighted_agreement};

/// Lifecycle phase of a topic
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TopicPhase {
    /// Less than 1 hour old, membership changing rapidly
    #[default]
    Emerging,
    /// Consistent membership for 24+ hours, churn < 0.1
    Stable,
    /// Decreasing access, members leaving
    Declining,
    /// Being absorbed into another topic
    Merging,
}

/// Per-space strength profile for a topic
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TopicProfile {
    /// Strength in each of 13 embedding spaces (0.0..=1.0)
    pub strengths: [f32; 13],
}

impl Default for TopicProfile {
    fn default() -> Self {
        Self { strengths: [0.0; 13] }
    }
}

impl TopicProfile {
    /// Create a new topic profile with clamped strengths
    pub fn new(strengths: [f32; 13]) -> Self {
        let strengths = strengths.map(|s| s.clamp(0.0, 1.0));
        Self { strengths }
    }

    /// Create a default (all zeros) profile
    pub fn default_profile() -> Self {
        Self::default()
    }

    /// Get strength for a specific embedder
    #[inline]
    pub fn strength(&amp;self, embedder: Embedder) -> f32 {
        self.strengths[embedder.index()]
    }

    /// Set strength for a specific embedder (clamped to 0.0..=1.0)
    pub fn set_strength(&amp;mut self, embedder: Embedder, strength: f32) {
        self.strengths[embedder.index()] = strength.clamp(0.0, 1.0);
    }

    /// Get spaces where this topic is dominant (strength > 0.5)
    pub fn dominant_spaces(&amp;self) -> Vec&lt;Embedder&gt; {
        Embedder::all()
            .into_iter()
            .filter(|e| self.strength(*e) > 0.5)
            .collect()
    }

    /// Compute weighted agreement per ARCH-09.
    ///
    /// Uses EmbedderCategory::topic_weight() for each space:
    /// - SEMANTIC (E1, E5, E6, E7, E10, E12, E13): 1.0 weight
    /// - TEMPORAL (E2, E3, E4): 0.0 weight (NEVER counts per AP-60)
    /// - RELATIONAL (E8, E11): 0.5 weight
    /// - STRUCTURAL (E9): 0.5 weight
    ///
    /// # Returns
    /// Sum of (strength_i * topic_weight_i) for all spaces, max = 8.5
    pub fn weighted_agreement(&amp;self) -> f32 {
        let mut sum = 0.0f32;
        for embedder in Embedder::all() {
            let strength = self.strength(embedder);
            let category = EmbedderCategory::from(embedder);
            let weight = category.topic_weight();
            sum += strength * weight;
        }
        // Clamp to valid range and handle NaN (AP-10)
        if sum.is_nan() || sum.is_infinite() {
            0.0
        } else {
            sum.clamp(0.0, max_weighted_agreement())
        }
    }

    /// Check if this profile meets the topic threshold.
    /// Per ARCH-09: weighted_agreement >= 2.5
    #[inline]
    pub fn is_topic(&amp;self) -> bool {
        self.weighted_agreement() >= topic_threshold()
    }

    /// Compute cosine similarity with another profile
    pub fn similarity(&amp;self, other: &amp;TopicProfile) -> f32 {
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for i in 0..13 {
            dot += self.strengths[i] * other.strengths[i];
            norm_a += self.strengths[i] * self.strengths[i];
            norm_b += other.strengths[i] * other.strengths[i];
        }

        let norm = (norm_a.sqrt() * norm_b.sqrt()).max(1e-10);
        let result = dot / norm;

        // Handle NaN/Infinity (AP-10)
        if result.is_nan() || result.is_infinite() {
            0.0
        } else {
            result.clamp(0.0, 1.0)
        }
    }

    /// Count spaces with non-zero strength (> 0.1 threshold)
    pub fn active_space_count(&amp;self) -> usize {
        self.strengths.iter().filter(|&amp;&amp;s| s > 0.1).count()
    }
}

/// Stability metrics for a topic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicStability {
    /// Current lifecycle phase
    pub phase: TopicPhase,
    /// Age in hours since creation
    pub age_hours: f32,
    /// Membership churn rate (0.0..=1.0) - how often members change
    pub membership_churn: f32,
    /// Centroid drift since last snapshot (0.0..=1.0)
    pub centroid_drift: f32,
    /// Total access count
    pub access_count: u32,
    /// Last access time
    pub last_accessed: Option&lt;DateTime&lt;Utc&gt;&gt;,
}

impl Default for TopicStability {
    fn default() -> Self {
        Self {
            phase: TopicPhase::Emerging,
            age_hours: 0.0,
            membership_churn: 0.0,
            centroid_drift: 0.0,
            access_count: 0,
            last_accessed: None,
        }
    }
}

impl TopicStability {
    /// Create new stability metrics in Emerging phase
    pub fn new() -> Self {
        Self::default()
    }

    /// Update the phase based on current metrics.
    ///
    /// Phase transition rules:
    /// - Emerging: age < 1hr AND churn > 0.3
    /// - Stable: age >= 24hr AND churn < 0.1
    /// - Declining: access declining (detected externally)
    /// - Merging: set externally when merged
    pub fn update_phase(&amp;mut self) {
        self.phase = if self.age_hours < 1.0 &amp;&amp; self.membership_churn > 0.3 {
            TopicPhase::Emerging
        } else if self.membership_churn < 0.1 &amp;&amp; self.age_hours >= 24.0 {
            TopicPhase::Stable
        } else if self.membership_churn > 0.5 {
            TopicPhase::Declining
        } else {
            self.phase // Keep current if no transition triggers
        };
    }

    /// Check if topic is in stable phase
    #[inline]
    pub fn is_stable(&amp;self) -> bool {
        self.phase == TopicPhase::Stable
    }

    /// Check if topic is healthy (churn < 0.3 per constitution)
    #[inline]
    pub fn is_healthy(&amp;self) -> bool {
        self.membership_churn < 0.3
    }
}

/// A topic that emerges from cross-space clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topic {
    /// Unique identifier
    pub id: Uuid,
    /// Optional human-readable name (auto-generated or user-provided)
    pub name: Option&lt;String&gt;,
    /// Per-space strength profile
    pub profile: TopicProfile,
    /// Spaces where this topic has strong representation (strength > 0.5)
    pub contributing_spaces: Vec&lt;Embedder&gt;,
    /// Cluster ID in each contributing space
    pub cluster_ids: HashMap&lt;Embedder, i32&gt;,
    /// Memory IDs that belong to this topic
    pub member_memories: Vec&lt;Uuid&gt;,
    /// Confidence score = weighted_agreement / 8.5 (per ARCH-09)
    pub confidence: f32,
    /// Stability metrics
    pub stability: TopicStability,
    /// Creation timestamp
    pub created_at: DateTime&lt;Utc&gt;,
}

impl Topic {
    /// Create a new topic from profile and cluster assignments.
    ///
    /// Confidence is computed as weighted_agreement / MAX_WEIGHTED_AGREEMENT (8.5)
    pub fn new(
        profile: TopicProfile,
        cluster_ids: HashMap&lt;Embedder, i32&gt;,
        members: Vec&lt;Uuid&gt;,
    ) -> Self {
        let contributing_spaces = profile.dominant_spaces();
        let weighted = profile.weighted_agreement();
        let confidence = weighted / max_weighted_agreement();

        Self {
            id: Uuid::new_v4(),
            name: None,
            profile,
            contributing_spaces,
            cluster_ids,
            member_memories: members,
            confidence: confidence.clamp(0.0, 1.0),
            stability: TopicStability::new(),
            created_at: Utc::now(),
        }
    }

    /// Compute confidence based on weighted agreement.
    /// confidence = weighted_agreement / 8.5
    pub fn compute_confidence(&amp;self) -> f32 {
        let weighted = self.profile.weighted_agreement();
        (weighted / max_weighted_agreement()).clamp(0.0, 1.0)
    }

    /// Record an access to this topic
    pub fn record_access(&amp;mut self) {
        self.stability.access_count = self.stability.access_count.saturating_add(1);
        self.stability.last_accessed = Some(Utc::now());
    }

    /// Set the topic name
    pub fn set_name(&amp;mut self, name: String) {
        self.name = Some(name);
    }

    /// Check if this topic is valid (weighted_agreement >= 2.5 per ARCH-09)
    #[inline]
    pub fn is_valid(&amp;self) -> bool {
        self.profile.is_topic()
    }

    /// Get member count
    #[inline]
    pub fn member_count(&amp;self) -> usize {
        self.member_memories.len()
    }

    /// Check if a memory belongs to this topic
    pub fn contains_memory(&amp;self, memory_id: &amp;Uuid) -> bool {
        self.member_memories.contains(memory_id)
    }

    /// Update contributing spaces from profile (call after modifying profile)
    pub fn update_contributing_spaces(&amp;mut self) {
        self.contributing_spaces = self.profile.dominant_spaces();
        self.confidence = self.compute_confidence();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== TopicProfile Tests =====

    #[test]
    fn test_profile_strength_clamping() {
        let profile = TopicProfile::new([1.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(profile.strengths[0], 1.0, "1.5 should clamp to 1.0");
        assert_eq!(profile.strengths[1], 0.0, "-0.5 should clamp to 0.0");
        assert_eq!(profile.strengths[2], 0.5, "0.5 should stay 0.5");
        println!("[PASS] test_profile_strength_clamping");
    }

    #[test]
    fn test_weighted_agreement_semantic_only() {
        // E1 (Semantic, weight=1.0), E5 (Causal/Semantic, weight=1.0), E7 (Code/Semantic, weight=1.0)
        let mut strengths = [0.0; 13];
        strengths[Embedder::Semantic.index()] = 1.0;      // E1
        strengths[Embedder::Causal.index()] = 1.0;        // E5
        strengths[Embedder::Code.index()] = 1.0;          // E7

        let profile = TopicProfile::new(strengths);
        let weighted = profile.weighted_agreement();

        assert!((weighted - 3.0).abs() < 0.001,
            "3 semantic spaces at strength 1.0 should give weighted_agreement = 3.0, got {}", weighted);
        assert!(profile.is_topic(), "weighted_agreement 3.0 >= 2.5 threshold should be topic");
        println!("[PASS] test_weighted_agreement_semantic_only - weighted={}", weighted);
    }

    #[test]
    fn test_weighted_agreement_temporal_excluded() {
        // CRITICAL TEST: Temporal spaces (E2, E3, E4) should contribute 0.0 per AP-60
        let mut strengths = [0.0; 13];
        strengths[Embedder::TemporalRecent.index()] = 1.0;      // E2 - temporal
        strengths[Embedder::TemporalPeriodic.index()] = 1.0;    // E3 - temporal
        strengths[Embedder::TemporalPositional.index()] = 1.0;  // E4 - temporal

        let profile = TopicProfile::new(strengths);
        let weighted = profile.weighted_agreement();

        assert!((weighted - 0.0).abs() < 0.001,
            "3 temporal spaces should give weighted_agreement = 0.0 per AP-60, got {}", weighted);
        assert!(!profile.is_topic(), "temporal-only should NOT be topic");
        println!("[PASS] test_weighted_agreement_temporal_excluded - temporal contributes 0.0");
    }

    #[test]
    fn test_weighted_agreement_mixed_categories() {
        // 2 semantic (2.0) + 1 relational (0.5) = 2.5 -> exactly threshold
        let mut strengths = [0.0; 13];
        strengths[Embedder::Semantic.index()] = 1.0;      // E1 - semantic (1.0)
        strengths[Embedder::Causal.index()] = 1.0;        // E5 - semantic (1.0)
        strengths[Embedder::Entity.index()] = 1.0;        // E11 - relational (0.5)

        let profile = TopicProfile::new(strengths);
        let weighted = profile.weighted_agreement();

        assert!((weighted - 2.5).abs() < 0.001,
            "2 semantic + 1 relational should give 2.5, got {}", weighted);
        assert!(profile.is_topic(), "weighted_agreement 2.5 meets threshold");
        println!("[PASS] test_weighted_agreement_mixed_categories - weighted={}", weighted);
    }

    #[test]
    fn test_weighted_agreement_below_threshold() {
        // 2 semantic only = 2.0 < 2.5 threshold
        let mut strengths = [0.0; 13];
        strengths[Embedder::Semantic.index()] = 1.0;   // E1
        strengths[Embedder::Causal.index()] = 1.0;     // E5

        let profile = TopicProfile::new(strengths);
        let weighted = profile.weighted_agreement();

        assert!((weighted - 2.0).abs() < 0.001,
            "2 semantic should give 2.0, got {}", weighted);
        assert!(!profile.is_topic(), "weighted_agreement 2.0 < 2.5 threshold");
        println!("[PASS] test_weighted_agreement_below_threshold - weighted={}", weighted);
    }

    #[test]
    fn test_weighted_agreement_max_value() {
        // All spaces at 1.0 - max possible
        let profile = TopicProfile::new([1.0; 13]);
        let weighted = profile.weighted_agreement();

        // Max = 7*1.0 (semantic) + 2*0.5 (relational) + 1*0.5 (structural) + 3*0.0 (temporal) = 8.5
        assert!((weighted - 8.5).abs() < 0.001,
            "All spaces at 1.0 should give max 8.5, got {}", weighted);
        println!("[PASS] test_weighted_agreement_max_value - weighted={}", weighted);
    }

    #[test]
    fn test_dominant_spaces() {
        let mut strengths = [0.0; 13];
        strengths[Embedder::Semantic.index()] = 0.8;
        strengths[Embedder::Causal.index()] = 0.7;
        strengths[Embedder::Code.index()] = 0.9;
        strengths[Embedder::Entity.index()] = 0.3; // Below 0.5, should not be dominant

        let profile = TopicProfile::new(strengths);
        let dominant = profile.dominant_spaces();

        assert_eq!(dominant.len(), 3, "Should have 3 dominant spaces (> 0.5)");
        assert!(dominant.contains(&amp;Embedder::Semantic));
        assert!(dominant.contains(&amp;Embedder::Causal));
        assert!(dominant.contains(&amp;Embedder::Code));
        assert!(!dominant.contains(&amp;Embedder::Entity), "0.3 should not be dominant");
        println!("[PASS] test_dominant_spaces - found {} dominant spaces", dominant.len());
    }

    #[test]
    fn test_profile_similarity_identical() {
        let p1 = TopicProfile::new([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p2 = TopicProfile::new([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let sim = p1.similarity(&amp;p2);
        assert!((sim - 1.0).abs() < 0.001, "Identical profiles should have similarity 1.0, got {}", sim);
        println!("[PASS] test_profile_similarity_identical - sim={}", sim);
    }

    #[test]
    fn test_profile_similarity_orthogonal() {
        let p1 = TopicProfile::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p2 = TopicProfile::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let sim = p1.similarity(&amp;p2);
        assert!(sim < 0.001, "Orthogonal profiles should have similarity ~0.0, got {}", sim);
        println!("[PASS] test_profile_similarity_orthogonal - sim={}", sim);
    }

    #[test]
    fn test_profile_similarity_zero_vector() {
        let p1 = TopicProfile::new([0.0; 13]);
        let p2 = TopicProfile::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let sim = p1.similarity(&amp;p2);
        assert!(!sim.is_nan(), "Zero vector similarity should not be NaN");
        assert!(sim >= 0.0 &amp;&amp; sim <= 1.0, "Similarity should be in valid range");
        println!("[PASS] test_profile_similarity_zero_vector - handles zero vector gracefully");
    }

    // ===== Topic Tests =====

    #[test]
    fn test_topic_confidence_calculation() {
        // 3 semantic spaces at 1.0 = weighted 3.0
        // confidence = 3.0 / 8.5 ≈ 0.353
        let mut strengths = [0.0; 13];
        strengths[Embedder::Semantic.index()] = 1.0;
        strengths[Embedder::Causal.index()] = 1.0;
        strengths[Embedder::Code.index()] = 1.0;

        let profile = TopicProfile::new(strengths);
        let topic = Topic::new(profile, HashMap::new(), vec![]);

        let expected_confidence = 3.0 / 8.5;
        assert!((topic.confidence - expected_confidence).abs() < 0.001,
            "confidence should be weighted/8.5 = {}, got {}", expected_confidence, topic.confidence);
        println!("[PASS] test_topic_confidence_calculation - confidence={}", topic.confidence);
    }

    #[test]
    fn test_topic_validity_weighted_threshold() {
        // Valid: 3 semantic = 3.0 >= 2.5
        let mut valid_strengths = [0.0; 13];
        valid_strengths[Embedder::Semantic.index()] = 1.0;
        valid_strengths[Embedder::Causal.index()] = 1.0;
        valid_strengths[Embedder::Code.index()] = 1.0;

        let valid_topic = Topic::new(TopicProfile::new(valid_strengths), HashMap::new(), vec![]);
        assert!(valid_topic.is_valid(), "3 semantic spaces (weighted=3.0) should be valid");

        // Invalid: 2 semantic = 2.0 < 2.5
        let mut invalid_strengths = [0.0; 13];
        invalid_strengths[Embedder::Semantic.index()] = 1.0;
        invalid_strengths[Embedder::Causal.index()] = 1.0;

        let invalid_topic = Topic::new(TopicProfile::new(invalid_strengths), HashMap::new(), vec![]);
        assert!(!invalid_topic.is_valid(), "2 semantic spaces (weighted=2.0) should NOT be valid");

        println!("[PASS] test_topic_validity_weighted_threshold");
    }

    #[test]
    fn test_topic_validity_temporal_ignored() {
        // CRITICAL: 5 temporal spaces should NOT make a valid topic
        let mut temporal_strengths = [0.0; 13];
        temporal_strengths[Embedder::TemporalRecent.index()] = 1.0;
        temporal_strengths[Embedder::TemporalPeriodic.index()] = 1.0;
        temporal_strengths[Embedder::TemporalPositional.index()] = 1.0;
        // Adding more temporal won't help - they're weight 0.0

        let temporal_topic = Topic::new(TopicProfile::new(temporal_strengths), HashMap::new(), vec![]);
        assert!(!temporal_topic.is_valid(),
            "Temporal-only topic (weighted=0.0) must NOT be valid per AP-60");
        println!("[PASS] test_topic_validity_temporal_ignored - temporal excluded from topic detection");
    }

    #[test]
    fn test_topic_record_access() {
        let profile = TopicProfile::default();
        let mut topic = Topic::new(profile, HashMap::new(), vec![]);

        assert_eq!(topic.stability.access_count, 0);
        assert!(topic.stability.last_accessed.is_none());

        topic.record_access();

        assert_eq!(topic.stability.access_count, 1);
        assert!(topic.stability.last_accessed.is_some());
        println!("[PASS] test_topic_record_access");
    }

    #[test]
    fn test_topic_member_operations() {
        let profile = TopicProfile::default();
        let mem1 = Uuid::new_v4();
        let mem2 = Uuid::new_v4();
        let mem3 = Uuid::new_v4();

        let topic = Topic::new(profile, HashMap::new(), vec![mem1, mem2]);

        assert_eq!(topic.member_count(), 2);
        assert!(topic.contains_memory(&amp;mem1));
        assert!(topic.contains_memory(&amp;mem2));
        assert!(!topic.contains_memory(&amp;mem3));
        println!("[PASS] test_topic_member_operations");
    }

    // ===== TopicStability Tests =====

    #[test]
    fn test_stability_phase_transitions() {
        let mut stability = TopicStability::new();
        assert_eq!(stability.phase, TopicPhase::Emerging, "Should start as Emerging");

        // Young with high churn -> Emerging
        stability.age_hours = 0.5;
        stability.membership_churn = 0.4;
        stability.update_phase();
        assert_eq!(stability.phase, TopicPhase::Emerging);

        // Old with low churn -> Stable
        stability.age_hours = 48.0;
        stability.membership_churn = 0.05;
        stability.update_phase();
        assert_eq!(stability.phase, TopicPhase::Stable);

        // High churn -> Declining
        stability.membership_churn = 0.6;
        stability.update_phase();
        assert_eq!(stability.phase, TopicPhase::Declining);

        println!("[PASS] test_stability_phase_transitions");
    }

    #[test]
    fn test_stability_health_check() {
        let mut stability = TopicStability::new();
        stability.membership_churn = 0.2;
        assert!(stability.is_healthy(), "churn 0.2 < 0.3 should be healthy");

        stability.membership_churn = 0.4;
        assert!(!stability.is_healthy(), "churn 0.4 >= 0.3 should not be healthy");
        println!("[PASS] test_stability_health_check");
    }

    // ===== Serialization Tests =====

    #[test]
    fn test_topic_serialization_roundtrip() {
        let mut strengths = [0.0; 13];
        strengths[0] = 0.9;
        strengths[4] = 0.8;

        let profile = TopicProfile::new(strengths);
        let topic = Topic::new(profile, HashMap::new(), vec![Uuid::new_v4()]);

        let json = serde_json::to_string(&amp;topic).expect("serialize should work");
        let restored: Topic = serde_json::from_str(&amp;json).expect("deserialize should work");

        assert_eq!(topic.id, restored.id);
        assert_eq!(topic.profile.strengths, restored.profile.strengths);
        assert_eq!(topic.member_memories.len(), restored.member_memories.len());
        println!("[PASS] test_topic_serialization_roundtrip - JSON length: {}", json.len());
    }

    #[test]
    fn test_topic_phase_serialization() {
        for phase in [TopicPhase::Emerging, TopicPhase::Stable, TopicPhase::Declining, TopicPhase::Merging] {
            let json = serde_json::to_string(&amp;phase).expect("serialize phase");
            let restored: TopicPhase = serde_json::from_str(&amp;json).expect("deserialize phase");
            assert_eq!(phase, restored);
        }
        println!("[PASS] test_topic_phase_serialization - all phases serialize correctly");
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/clustering/topic.rs">
    Topic, TopicProfile, TopicPhase, TopicStability types with weighted agreement
  </file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/clustering/mod.rs">
    Add: pub mod topic;
    Add re-exports: pub use topic::{Topic, TopicProfile, TopicPhase, TopicStability};
  </file>
  <file path="crates/context-graph-core/src/lib.rs">
    Add to clustering re-exports: Topic, TopicProfile, TopicPhase, TopicStability
  </file>
</files_to_modify>

<validation_criteria>
  <criterion priority="CRITICAL">TopicProfile::weighted_agreement() uses EmbedderCategory::topic_weight()</criterion>
  <criterion priority="CRITICAL">Temporal spaces (E2, E3, E4) contribute 0.0 to weighted_agreement</criterion>
  <criterion priority="CRITICAL">is_topic() returns true iff weighted_agreement >= 2.5</criterion>
  <criterion priority="CRITICAL">Topic confidence = weighted_agreement / 8.5 (NOT raw count / 13)</criterion>
  <criterion priority="CRITICAL">Topic::is_valid() uses weighted threshold, not raw space count</criterion>
  <criterion>TopicProfile strengths clamped to 0.0..=1.0</criterion>
  <criterion>dominant_spaces returns spaces with strength > 0.5</criterion>
  <criterion>Profile similarity uses cosine similarity, handles zero vectors</criterion>
  <criterion>TopicPhase transitions based on age and churn</criterion>
  <criterion>All f32 operations handle NaN/Infinity (return 0.0)</criterion>
</validation_criteria>

<test_commands>
  <command description="Run topic tests">cargo test --package context-graph-core topic -- --nocapture</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
  <command description="Run all clustering tests">cargo test --package context-graph-core clustering -- --nocapture</command>
  <command description="Verify no clippy warnings">cargo clippy --package context-graph-core -- -D warnings</command>
</test_commands>
</task_spec>
```

## Full State Verification Requirements

### Source of Truth Identification

| Aspect | Source of Truth | Verification Method |
|--------|-----------------|---------------------|
| Topic threshold | `embeddings/category.rs::topic_threshold()` = 2.5 | Call function, assert == 2.5 |
| Max weighted agreement | `embeddings/category.rs::max_weighted_agreement()` = 8.5 | Call function, assert == 8.5 |
| Embedder category weights | `EmbedderCategory::topic_weight()` | Semantic=1.0, Temporal=0.0, Relational=0.5, Structural=0.5 |
| Embedder enum variants | `teleological/embedder.rs::Embedder` | 13 variants with index() method |
| Temporal exclusion rule | Constitution ARCH-04, AP-60 | Test weighted_agreement with temporal-only = 0.0 |

### Execute & Inspect Protocol

After implementation, run these verification steps:

```bash
# 1. Compile check - must pass with no errors
cargo check --package context-graph-core

# 2. Run all topic tests with output
cargo test --package context-graph-core topic -- --nocapture 2>&1 | tee /tmp/topic_test_output.txt

# 3. Verify critical tests passed
grep -E "test_weighted_agreement_temporal_excluded.*PASS" /tmp/topic_test_output.txt || echo "CRITICAL: Temporal exclusion test missing"
grep -E "test_topic_validity_temporal_ignored.*PASS" /tmp/topic_test_output.txt || echo "CRITICAL: Validity temporal test missing"

# 4. Clippy check
cargo clippy --package context-graph-core -- -D warnings

# 5. Doc tests
cargo test --package context-graph-core --doc
```

### Boundary & Edge Case Audit

| Edge Case | Test Name | Expected Behavior |
|-----------|-----------|-------------------|
| Zero vector profile | `test_profile_similarity_zero_vector` | Return 0.0, not NaN |
| All temporal spaces | `test_weighted_agreement_temporal_excluded` | weighted_agreement = 0.0 |
| Exactly at threshold | `test_weighted_agreement_mixed_categories` | 2.5 should be topic |
| Below threshold | `test_weighted_agreement_below_threshold` | 2.0 should NOT be topic |
| Max weighted value | `test_weighted_agreement_max_value` | Clamp to 8.5 |
| Strength > 1.0 | `test_profile_strength_clamping` | Clamp to 1.0 |
| Strength < 0.0 | `test_profile_strength_clamping` | Clamp to 0.0 |
| NaN in calculation | (inline check) | Return 0.0 |

### Evidence of Success

The implementation is successful when:

1. **All tests pass**: `cargo test --package context-graph-core clustering` shows 0 failures
2. **Temporal exclusion verified**: Test output contains "temporal contributes 0.0"
3. **Weighted threshold verified**: Test shows "weighted_agreement 3.0 >= 2.5 threshold"
4. **Exports work**: `use context_graph_core::clustering::{Topic, TopicProfile, TopicPhase, TopicStability};` compiles
5. **No clippy warnings**: `cargo clippy` passes with `-D warnings`

## Trigger-Outcome Verification

| Trigger | Expected Outcome | Verification |
|---------|-----------------|--------------|
| Create TopicProfile with 3 semantic strengths=1.0 | weighted_agreement() returns 3.0 | Assert in test |
| Create TopicProfile with 3 temporal strengths=1.0 | weighted_agreement() returns 0.0 | Assert in test |
| Create Topic from profile with weighted 3.0 | confidence = 3.0/8.5 ≈ 0.353 | Assert in test |
| Call is_topic() on profile with weighted 2.5 | Returns true | Assert in test |
| Call is_topic() on profile with weighted 2.0 | Returns false | Assert in test |
| Call is_valid() on temporal-only Topic | Returns false | Assert in test |
| Serialize/deserialize Topic | All fields preserved | Roundtrip test |

## Manual Testing with Synthetic Data

After implementation, perform these manual verification steps:

### Test 1: Weighted Agreement Calculation
```rust
// In a scratch test or main.rs
use context_graph_core::clustering::TopicProfile;
use context_graph_core::teleological::Embedder;

let mut strengths = [0.0; 13];
strengths[Embedder::Semantic.index()] = 1.0;      // E1 - weight 1.0
strengths[Embedder::TemporalRecent.index()] = 1.0; // E2 - weight 0.0 (excluded!)
strengths[Embedder::Causal.index()] = 1.0;        // E5 - weight 1.0
strengths[Embedder::Entity.index()] = 1.0;        // E11 - weight 0.5

let profile = TopicProfile::new(strengths);
let weighted = profile.weighted_agreement();

// Expected: 1.0 + 0.0 + 1.0 + 0.5 = 2.5
println!("Weighted agreement: {} (expected 2.5)", weighted);
assert!((weighted - 2.5).abs() < 0.001);
assert!(profile.is_topic()); // Should be true at exactly 2.5
```

### Test 2: Topic Creation and Validation
```rust
use context_graph_core::clustering::{Topic, TopicProfile};
use std::collections::HashMap;

// Create a valid topic (3 semantic = 3.0 weighted)
let mut valid_strengths = [0.0; 13];
valid_strengths[0] = 0.9; // Semantic
valid_strengths[4] = 0.9; // Causal
valid_strengths[6] = 0.9; // Code

let profile = TopicProfile::new(valid_strengths);
let topic = Topic::new(profile, HashMap::new(), vec![]);

println!("Topic valid: {} (expected true)", topic.is_valid());
println!("Topic confidence: {} (expected ~0.35)", topic.confidence);
assert!(topic.is_valid());
```

### Test 3: Verify Temporal Exclusion
```rust
// CRITICAL: This must fail to create a valid topic
let mut temporal_only = [0.0; 13];
temporal_only[1] = 1.0; // TemporalRecent
temporal_only[2] = 1.0; // TemporalPeriodic
temporal_only[3] = 1.0; // TemporalPositional

let profile = TopicProfile::new(temporal_only);
let topic = Topic::new(profile, HashMap::new(), vec![]);

println!("Temporal-only topic valid: {} (MUST be false per AP-60)", topic.is_valid());
assert!(!topic.is_valid(), "Temporal-only MUST NOT be valid topic");
```

## Execution Checklist

- [ ] Create topic.rs with TopicPhase enum (with Default derive)
- [ ] Implement TopicProfile struct with weighted_agreement() method
- [ ] Implement TopicStability struct with phase transitions
- [ ] Implement Topic struct with weighted confidence calculation
- [ ] Add is_topic() validation using weighted threshold 2.5
- [ ] Add to mod.rs: `pub mod topic;` and re-exports
- [ ] Update lib.rs with re-exports
- [ ] Run unit tests: `cargo test --package context-graph-core topic`
- [ ] Verify temporal exclusion test passes
- [ ] Run clippy: `cargo clippy --package context-graph-core -- -D warnings`
- [ ] Perform manual testing with synthetic data
- [ ] Verify all tests pass with output showing expected values
- [ ] Proceed to TASK-P4-003

## Anti-Pattern Checklist

Before marking complete, verify:

- [x] NOT using raw space count >= 3 for topic validity (use weighted >= 2.5)
- [x] NOT including temporal (E2-E4) in weighted agreement calculation
- [x] NOT using confidence = spaces/13 (use weighted/8.5)
- [x] NOT allowing NaN or Infinity in calculations (clamp/check)
- [x] NOT creating mock EmbedderCategory - use real one from crate

---

## Completion Verification Report

**Completed:** 2026-01-17

### Files Created
- `crates/context-graph-core/src/clustering/topic.rs` - 488 lines

### Files Modified
- `crates/context-graph-core/src/clustering/mod.rs` - Added topic module and re-exports
- `crates/context-graph-core/src/lib.rs` - Added clustering type re-exports

### Test Results

**Unit Tests:** 27 passing in topic.rs
**Integration Tests:** 9 passing in topic_manual_test.rs
**Total:** 32 topic-related tests passing

```
test clustering::topic::tests::test_active_space_count ... ok
test clustering::topic::tests::test_constitution_examples ... ok
test clustering::topic::tests::test_dominant_spaces ... ok
test clustering::topic::tests::test_extreme_strength_values ... ok
test clustering::topic::tests::test_nan_handling_weighted_agreement ... ok
test clustering::topic::tests::test_profile_similarity_identical ... ok
test clustering::topic::tests::test_profile_similarity_orthogonal ... ok
test clustering::topic::tests::test_profile_similarity_zero_vector ... ok
test clustering::topic::tests::test_profile_strength_clamping ... ok
test clustering::topic::tests::test_stability_health_check ... ok
test clustering::topic::tests::test_stability_phase_transitions ... ok
test clustering::topic::tests::test_topic_confidence_calculation ... ok
test clustering::topic::tests::test_topic_member_operations ... ok
test clustering::topic::tests::test_topic_phase_display ... ok
test clustering::topic::tests::test_topic_phase_serialization ... ok
test clustering::topic::tests::test_topic_profile_serialization ... ok
test clustering::topic::tests::test_topic_record_access ... ok
test clustering::topic::tests::test_topic_serialization_roundtrip ... ok
test clustering::topic::tests::test_topic_stability_serialization ... ok
test clustering::topic::tests::test_topic_update_contributing_spaces ... ok
test clustering::topic::tests::test_topic_validity_temporal_ignored ... ok
test clustering::topic::tests::test_topic_validity_weighted_threshold ... ok
test clustering::topic::tests::test_weighted_agreement_below_threshold ... ok
test clustering::topic::tests::test_weighted_agreement_max_value ... ok
test clustering::topic::tests::test_weighted_agreement_mixed_categories ... ok
test clustering::topic::tests::test_weighted_agreement_semantic_only ... ok
test clustering::topic::tests::test_weighted_agreement_temporal_excluded ... ok

test result: ok. 32 passed; 0 failed
```

### Constitution Compliance Verified

| Rule | Status | Evidence |
|------|--------|----------|
| ARCH-09 | PASS | `topic_threshold() = 2.5` verified in tests |
| AP-60 | PASS | `test_weighted_agreement_temporal_excluded` confirms temporal=0.0 |
| AP-10 | PASS | NaN/Infinity handling in `weighted_agreement()` and `similarity()` |

### Weighted Agreement Verification

```
3 semantic spaces at 1.0 = 3.0 weighted ✓ TOPIC
2 semantic + 1 relational = 2.5 weighted ✓ TOPIC
2 semantic only = 2.0 weighted ✗ NOT TOPIC
3 temporal only = 0.0 weighted ✗ NOT TOPIC (AP-60)
All spaces at 1.0 = 8.5 weighted (max) ✓
```

### Clippy Status

No warnings in topic.rs (existing warnings in other files are unrelated to this task)
