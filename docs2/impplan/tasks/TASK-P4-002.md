# TASK-P4-002: Topic, TopicProfile, TopicPhase Types

```xml
<task_spec id="TASK-P4-002" version="1.0">
<metadata>
  <title>Topic Type Implementation</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>28</sequence>
  <phase>4</phase>
  <implements>
    <requirement_ref>REQ-P4-04</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P4-001</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
</metadata>

<context>
Implements Topic, TopicProfile, TopicPhase, and TopicStability types. A Topic
represents a cross-space concept that clusters together in multiple embedding
spaces. TopicProfile captures per-space strength. TopicPhase tracks lifecycle
(Emerging, Stable, Declining, Merging).

Topics emerge from finding memories that cluster together in ≥3 spaces.
</context>

<input_context_files>
  <file purpose="data_models">docs2/impplan/technical/TECH-PHASE4-CLUSTERING.md#data_models</file>
  <file purpose="membership">crates/context-graph-core/src/clustering/membership.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P4-001 complete (ClusterMembership exists)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create Topic struct with all fields
    - Create TopicProfile with 13-space strengths
    - Create TopicPhase enum
    - Create TopicStability struct
    - Implement profile similarity calculation
    - Implement dominant_spaces method
  </in_scope>
  <out_of_scope>
    - Topic synthesis algorithm (TASK-P4-008)
    - Topic stability tracking (TASK-P4-009)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/clustering/topic.rs">
      #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
      pub enum TopicPhase {
          Emerging,
          Stable,
          Declining,
          Merging,
      }

      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct TopicProfile {
          pub strengths: [f32; 13],
      }

      impl TopicProfile {
          pub fn new(strengths: [f32; 13]) -> Self;
          pub fn dominant_spaces(&amp;self) -> Vec&lt;Embedder&gt;;
          pub fn similarity(&amp;self, other: &amp;TopicProfile) -> f32;
          pub fn strength(&amp;self, embedder: Embedder) -> f32;
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
          pub confidence: f32,
          pub stability: TopicStability,
          pub created_at: DateTime&lt;Utc&gt;,
      }

      impl Topic {
          pub fn new(profile: TopicProfile, cluster_ids: HashMap&lt;Embedder, i32&gt;, members: Vec&lt;Uuid&gt;) -> Self;
          pub fn compute_confidence(&amp;self) -> f32;
          pub fn record_access(&amp;mut self);
      }
    </signature>
  </signatures>

  <constraints>
    - TopicProfile strengths in 0.0..=1.0
    - contributing_spaces.len() >= 3 for valid topic
    - confidence = agreeing_spaces / 13
    - membership_churn in 0.0..=1.0
    - centroid_drift in 0.0..=1.0
  </constraints>

  <verification>
    - dominant_spaces returns spaces with strength > 0.5
    - Profile similarity computed correctly (cosine)
    - Topic confidence calculated from contributing spaces
    - TopicPhase transitions make sense
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/clustering/topic.rs

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use crate::embedding::Embedder;

/// Lifecycle phase of a topic
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopicPhase {
    /// Less than 1 hour old, membership changing
    Emerging,
    /// Consistent membership for 24+ hours
    Stable,
    /// Decreasing access, members leaving
    Declining,
    /// Being absorbed into another topic
    Merging,
}

impl Default for TopicPhase {
    fn default() -> Self {
        TopicPhase::Emerging
    }
}

/// Per-space strength profile for a topic
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TopicProfile {
    /// Strength in each of 13 embedding spaces (0.0..=1.0)
    pub strengths: [f32; 13],
}

impl TopicProfile {
    /// Create a new topic profile
    pub fn new(strengths: [f32; 13]) -> Self {
        let strengths = strengths.map(|s| s.clamp(0.0, 1.0));
        Self { strengths }
    }

    /// Create a default (all zeros) profile
    pub fn default_profile() -> Self {
        Self { strengths: [0.0; 13] }
    }

    /// Get strength for a specific embedder
    pub fn strength(&amp;self, embedder: Embedder) -> f32 {
        self.strengths[embedder.index()]
    }

    /// Set strength for a specific embedder
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

    /// Compute cosine similarity with another profile
    pub fn similarity(&amp;self, other: &amp;TopicProfile) -> f32 {
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for i in 0..13 {
            dot += self.strengths[i] * other.strengths[i];
            norm_a += self.strengths[i] * self.strengths[i];
            norm_b += other.strengths[i] * other.strengths[i];
        }

        let norm = (norm_a.sqrt() * norm_b.sqrt()).max(1e-10);
        (dot / norm).clamp(0.0, 1.0)
    }

    /// Count spaces with non-zero strength
    pub fn active_space_count(&amp;self) -> usize {
        self.strengths.iter().filter(|&&s| s > 0.1).count()
    }
}

/// Stability metrics for a topic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicStability {
    /// Current lifecycle phase
    pub phase: TopicPhase,
    /// Age in hours since creation
    pub age_hours: f32,
    /// Membership churn rate (0.0..=1.0)
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
    /// Create new stability metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Update the phase based on current metrics
    pub fn update_phase(&amp;mut self) {
        self.phase = if self.age_hours < 1.0 &amp;&amp; self.membership_churn > 0.3 {
            TopicPhase::Emerging
        } else if self.membership_churn < 0.1 &amp;&amp; self.age_hours >= 24.0 {
            TopicPhase::Stable
        } else if self.access_count > 0 &amp;&amp; self.membership_churn > 0.2 {
            // Access declining check would need more history
            TopicPhase::Declining
        } else {
            self.phase
        };
    }

    /// Check if topic is stable
    pub fn is_stable(&amp;self) -> bool {
        self.phase == TopicPhase::Stable
    }
}

/// A topic that emerges from cross-space clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topic {
    /// Unique identifier
    pub id: Uuid,
    /// Optional human-readable name
    pub name: Option&lt;String&gt;,
    /// Per-space strength profile
    pub profile: TopicProfile,
    /// Spaces where this topic has strong representation
    pub contributing_spaces: Vec&lt;Embedder&gt;,
    /// Cluster ID in each contributing space
    pub cluster_ids: HashMap&lt;Embedder, i32&gt;,
    /// Memory IDs that belong to this topic
    pub member_memories: Vec&lt;Uuid&gt;,
    /// Confidence score (0.0..=1.0)
    pub confidence: f32,
    /// Stability metrics
    pub stability: TopicStability,
    /// Creation timestamp
    pub created_at: DateTime&lt;Utc&gt;,
}

impl Topic {
    /// Create a new topic
    pub fn new(
        profile: TopicProfile,
        cluster_ids: HashMap&lt;Embedder, i32&gt;,
        members: Vec&lt;Uuid&gt;,
    ) -> Self {
        let contributing_spaces = profile.dominant_spaces();
        let confidence = contributing_spaces.len() as f32 / 13.0;

        Self {
            id: Uuid::new_v4(),
            name: None,
            profile,
            contributing_spaces,
            cluster_ids,
            member_memories: members,
            confidence,
            stability: TopicStability::new(),
            created_at: Utc::now(),
        }
    }

    /// Compute confidence based on contributing spaces
    pub fn compute_confidence(&amp;self) -> f32 {
        self.contributing_spaces.len() as f32 / 13.0
    }

    /// Record an access to this topic
    pub fn record_access(&amp;mut self) {
        self.stability.access_count += 1;
        self.stability.last_accessed = Some(Utc::now());
    }

    /// Set the topic name
    pub fn set_name(&amp;mut self, name: String) {
        self.name = Some(name);
    }

    /// Check if this topic is valid (has minimum contributing spaces)
    pub fn is_valid(&amp;self) -> bool {
        self.contributing_spaces.len() >= 3
    }

    /// Get member count
    pub fn member_count(&amp;self) -> usize {
        self.member_memories.len()
    }

    /// Check if a memory belongs to this topic
    pub fn contains_memory(&amp;self, memory_id: &amp;Uuid) -> bool {
        self.member_memories.contains(memory_id)
    }

    /// Update contributing spaces from profile
    pub fn update_contributing_spaces(&amp;mut self) {
        self.contributing_spaces = self.profile.dominant_spaces();
        self.confidence = self.compute_confidence();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_profile_dominant_spaces() {
        let mut strengths = [0.0; 13];
        strengths[0] = 0.8; // E1
        strengths[4] = 0.7; // E5
        strengths[6] = 0.9; // E7

        let profile = TopicProfile::new(strengths);
        let dominant = profile.dominant_spaces();

        assert_eq!(dominant.len(), 3);
        assert!(dominant.contains(&amp;Embedder::E1Semantic));
        assert!(dominant.contains(&amp;Embedder::E5Causal));
        assert!(dominant.contains(&amp;Embedder::E7Code));
    }

    #[test]
    fn test_profile_similarity() {
        let p1 = TopicProfile::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p2 = TopicProfile::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p3 = TopicProfile::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        assert!((p1.similarity(&amp;p2) - 1.0).abs() < 1e-5);
        assert!(p1.similarity(&amp;p3) < 0.1);
    }

    #[test]
    fn test_topic_confidence() {
        let mut strengths = [0.0; 13];
        strengths[0] = 0.8;
        strengths[4] = 0.7;
        strengths[6] = 0.9;
        strengths[7] = 0.6;

        let profile = TopicProfile::new(strengths);
        let topic = Topic::new(profile, HashMap::new(), vec![]);

        // 4 contributing spaces out of 13
        assert!((topic.confidence - 4.0 / 13.0).abs() < 1e-5);
    }

    #[test]
    fn test_topic_validity() {
        let mut strengths = [0.0; 13];
        strengths[0] = 0.8;
        strengths[4] = 0.7;
        strengths[6] = 0.9;

        let profile = TopicProfile::new(strengths);
        let topic = Topic::new(profile, HashMap::new(), vec![]);

        assert!(topic.is_valid()); // 3 contributing spaces

        let weak_profile = TopicProfile::new([0.3; 13]); // None above 0.5
        let weak_topic = Topic::new(weak_profile, HashMap::new(), vec![]);
        assert!(!weak_topic.is_valid());
    }

    #[test]
    fn test_topic_phase_update() {
        let mut stability = TopicStability::new();

        stability.age_hours = 0.5;
        stability.membership_churn = 0.4;
        stability.update_phase();
        assert_eq!(stability.phase, TopicPhase::Emerging);

        stability.age_hours = 48.0;
        stability.membership_churn = 0.05;
        stability.update_phase();
        assert_eq!(stability.phase, TopicPhase::Stable);
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/clustering/topic.rs">Topic, TopicProfile, TopicPhase types</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/clustering/mod.rs">Add pub mod topic and re-exports</file>
</files_to_modify>

<validation_criteria>
  <criterion>TopicProfile strengths clamped to 0.0..=1.0</criterion>
  <criterion>dominant_spaces returns spaces with strength > 0.5</criterion>
  <criterion>Profile similarity uses cosine similarity</criterion>
  <criterion>Topic requires >= 3 contributing spaces for validity</criterion>
  <criterion>confidence = contributing_spaces.len() / 13</criterion>
  <criterion>TopicPhase transitions based on age and churn</criterion>
</validation_criteria>

<test_commands>
  <command description="Run topic tests">cargo test --package context-graph-core topic</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create topic.rs with TopicPhase enum
- [ ] Implement TopicProfile struct
- [ ] Implement TopicStability struct
- [ ] Implement Topic struct
- [ ] Implement profile similarity calculation
- [ ] Implement dominant_spaces method
- [ ] Add to mod.rs
- [ ] Write unit tests
- [ ] Run tests to verify
- [ ] Proceed to TASK-P4-003
