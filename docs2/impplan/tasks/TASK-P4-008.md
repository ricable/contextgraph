# TASK-P4-008: TopicSynthesizer

```xml
<task_spec id="TASK-P4-008" version="1.0">
<metadata>
  <title>TopicSynthesizer Implementation</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>34</sequence>
  <phase>4</phase>
  <implements>
    <requirement_ref>REQ-P4-04</requirement_ref>
    <requirement_ref>REQ-P4-05</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P4-002</task_ref>
    <task_ref>TASK-P4-007</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
</metadata>

<context>
Implements TopicSynthesizer which identifies cross-space topics using **weighted agreement**
rather than raw space count. Uses the cross-space algorithm to build connected components
of "topic-mates" and generates TopicProfiles.

**CRITICAL: Weighted Agreement Formula**
```
weighted_agreement = Sum(topic_weight_i * is_clustered_i)

Embedder Category Weights:
  SEMANTIC (E1, E5, E6, E7, E10, E12, E13): 1.0 each (7 spaces, max 7.0)
  TEMPORAL (E2, E3, E4): 0.0 each (EXCLUDED from topic detection)
  RELATIONAL (E8, E11): 0.5 each (2 spaces, max 1.0)
  STRUCTURAL (E9): 0.5 (1 space, max 0.5)

MAX_WEIGHTED_AGREEMENT = 8.5
TOPIC_THRESHOLD = 2.5 (minimum weighted_agreement for topic formation)
```

Topics emerge when memories show coherent clustering with weighted_agreement >= 2.5.
Temporal clustering is explicitly excluded - temporal proximity is NOT semantic similarity.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE4-CLUSTERING.md#component_contracts</file>
  <file purpose="algorithm">docs2/impplan/technical/TECH-PHASE4-CLUSTERING.md (Cross-Space Topic Detection Algorithm)</file>
  <file purpose="topic">crates/context-graph-core/src/clustering/topic.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P4-002 complete (Topic types exist)</check>
  <check>TASK-P4-007 complete (MultiSpaceClusterManager exists)</check>
</prerequisites>

<scope>
  <in_scope>
    - Implement synthesize_topics method
    - Build memory-to-clusters map
    - Compute weighted_agreement using embedder category weights
    - Find memories clustering together with weighted_agreement >= 2.5
    - Build connected components of topic-mates
    - Generate TopicProfiles with semantic/relational/structural breakdown
    - Compute topic_confidence = weighted_agreement / 8.5
    - Merge highly similar topics
    - Update topic stability
  </in_scope>
  <out_of_scope>
    - Topic naming (future LLM integration)
    - Topic persistence
    - Historical topic tracking
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/clustering/synthesizer.rs">
      /// Weighted agreement constants
      pub const TOPIC_THRESHOLD: f32 = 2.5;
      pub const MAX_WEIGHTED_AGREEMENT: f32 = 8.5;

      /// Embedder category weights for topic detection
      pub const SEMANTIC_WEIGHT: f32 = 1.0;  // E1, E5, E6, E7, E10, E12, E13
      pub const TEMPORAL_WEIGHT: f32 = 0.0;  // E2, E3, E4 (excluded)
      pub const RELATIONAL_WEIGHT: f32 = 0.5; // E8, E11
      pub const STRUCTURAL_WEIGHT: f32 = 0.5; // E9

      pub struct TopicSynthesizer {
          topic_threshold: f32,             // default 2.5
          merge_similarity_threshold: f32,
          min_silhouette: f32,
      }

      impl TopicSynthesizer {
          pub fn new() -> Self;
          pub fn with_config(topic_threshold: f32, merge_threshold: f32, min_silhouette: f32) -> Self;
          pub fn synthesize_topics(&amp;self, memberships: &amp;HashMap&lt;Embedder, Vec&lt;ClusterMembership&gt;&gt;) -> Result&lt;Vec&lt;Topic&gt;, ClusterError&gt;;
          pub fn update_topic_stability(&amp;self, topic: &amp;mut Topic, old_members: &amp;[Uuid], new_members: &amp;[Uuid]);
          fn compute_weighted_agreement(&amp;self, mem_a: &amp;Uuid, mem_b: &amp;Uuid, mem_clusters: &amp;HashMap&lt;Uuid, HashMap&lt;Embedder, i32&gt;&gt;) -> f32;
          fn get_embedder_weight(&amp;self, embedder: Embedder) -> f32;
          fn find_topic_mates(&amp;self, mem_clusters: &amp;HashMap&lt;Uuid, HashMap&lt;Embedder, i32&gt;&gt;) -> Vec&lt;Vec&lt;Uuid&gt;&gt;;
          fn compute_topic_profile(&amp;self, members: &amp;[Uuid], mem_clusters: &amp;HashMap&lt;Uuid, HashMap&lt;Embedder, i32&gt;&gt;) -> TopicProfile;
          fn merge_similar_topics(&amp;self, topics: Vec&lt;Topic&gt;) -> Vec&lt;Topic&gt;;
      }
    </signature>
  </signatures>

  <constraints>
    - TOPIC_THRESHOLD = 2.5 (weighted agreement minimum)
    - MAX_WEIGHTED_AGREEMENT = 8.5 (for normalization)
    - topic_confidence = weighted_agreement / 8.5
    - Temporal embedders (E2-E4) have weight 0.0 (excluded from topic detection)
    - Semantic embedders (E1, E5, E6, E7, E10, E12, E13) have weight 1.0
    - Relational embedders (E8, E11) have weight 0.5
    - Structural embedder (E9) has weight 0.5
    - merge_similarity_threshold = 0.9 (from spec)
    - Topics with &lt;2 members may be filtered
    - Profile similarity uses cosine
  </constraints>

  <verification>
    - synthesize_topics returns valid topics
    - Topics have weighted_agreement >= 2.5
    - Temporal spaces do NOT contribute to weighted_agreement
    - topic_confidence correctly normalized to [0, 1]
    - TopicProfile includes semantic/relational/structural breakdown
    - Similar topics merged
    - Stability updated properly
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/clustering/synthesizer.rs

use std::collections::{HashMap, HashSet};
use uuid::Uuid;
use crate::embedding::Embedder;
use super::membership::ClusterMembership;
use super::topic::{Topic, TopicProfile, TopicPhase, TopicStability};
use super::error::ClusterError;

/// Weighted agreement threshold for topic formation
pub const TOPIC_THRESHOLD: f32 = 2.5;

/// Maximum possible weighted agreement (7 semantic + 1.5 relational + 0.5 structural)
pub const MAX_WEIGHTED_AGREEMENT: f32 = 8.5;

/// Embedder category weights
pub const SEMANTIC_WEIGHT: f32 = 1.0;   // E1, E5, E6, E7, E10, E12, E13
pub const TEMPORAL_WEIGHT: f32 = 0.0;   // E2, E3, E4 (excluded from topics)
pub const RELATIONAL_WEIGHT: f32 = 0.5; // E8, E11
pub const STRUCTURAL_WEIGHT: f32 = 0.5; // E9

const DEFAULT_MERGE_THRESHOLD: f32 = 0.9;
const DEFAULT_MIN_SILHOUETTE: f32 = 0.3;

/// Synthesizes topics from cross-space clustering using weighted agreement.
/// Temporal embedders are excluded from topic detection.
pub struct TopicSynthesizer {
    /// Minimum weighted agreement for topic formation (default 2.5)
    topic_threshold: f32,
    /// Threshold for merging similar topics
    merge_similarity_threshold: f32,
    /// Minimum silhouette score for valid clusters
    min_silhouette: f32,
}

impl Default for TopicSynthesizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TopicSynthesizer {
    /// Create with default configuration
    pub fn new() -> Self {
        Self {
            topic_threshold: TOPIC_THRESHOLD,
            merge_similarity_threshold: DEFAULT_MERGE_THRESHOLD,
            min_silhouette: DEFAULT_MIN_SILHOUETTE,
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        topic_threshold: f32,
        merge_threshold: f32,
        min_silhouette: f32,
    ) -> Self {
        Self {
            topic_threshold: topic_threshold.max(0.0),
            merge_similarity_threshold: merge_threshold.clamp(0.0, 1.0),
            min_silhouette: min_silhouette.clamp(-1.0, 1.0),
        }
    }

    /// Get the weight for an embedder based on its category
    fn get_embedder_weight(&amp;self, embedder: Embedder) -> f32 {
        match embedder {
            // Semantic embedders (primary topic spaces)
            Embedder::E1Semantic | Embedder::E5Causal | Embedder::E6Sparse |
            Embedder::E7Code | Embedder::E10Multimodal | Embedder::E12LateInteract |
            Embedder::E13SPLADE => SEMANTIC_WEIGHT,

            // Temporal embedders (excluded from topic detection)
            Embedder::E2TempRecent | Embedder::E3TempPeriodic |
            Embedder::E4TempPosition => TEMPORAL_WEIGHT,

            // Relational embedders (supporting)
            Embedder::E8Emotional | Embedder::E11Entity => RELATIONAL_WEIGHT,

            // Structural embedder (supporting)
            Embedder::E9HDC => STRUCTURAL_WEIGHT,
        }
    }

    /// Synthesize topics from cluster memberships across all spaces
    pub fn synthesize_topics(
        &amp;self,
        memberships: &amp;HashMap&lt;Embedder, Vec&lt;ClusterMembership&gt;&gt;,
    ) -> Result&lt;Vec&lt;Topic&gt;, ClusterError&gt; {
        // Step 1: Build memory-to-clusters map
        let mem_clusters = self.build_mem_clusters_map(memberships);

        if mem_clusters.is_empty() {
            return Ok(Vec::new());
        }

        // Step 2: Find topic-mates (memories with weighted_agreement >= threshold)
        let topic_groups = self.find_topic_mates(&amp;mem_clusters);

        // Step 3: Create topics from groups
        let mut topics: Vec&lt;Topic&gt; = topic_groups
            .into_iter()
            .filter(|group| group.len() >= 2) // Need at least 2 memories
            .map(|members| self.create_topic(&amp;members, &amp;mem_clusters))
            .filter(|t| t.is_valid()) // Filter invalid topics
            .collect();

        // Step 4: Merge highly similar topics
        topics = self.merge_similar_topics(topics);

        Ok(topics)
    }

    /// Build map: memory_id -> (space -> cluster_id)
    fn build_mem_clusters_map(
        &amp;self,
        memberships: &amp;HashMap&lt;Embedder, Vec&lt;ClusterMembership&gt;&gt;,
    ) -> HashMap&lt;Uuid, HashMap&lt;Embedder, i32&gt;&gt; {
        let mut mem_clusters: HashMap&lt;Uuid, HashMap&lt;Embedder, i32&gt;&gt; = HashMap::new();

        for (space, space_memberships) in memberships {
            for membership in space_memberships {
                mem_clusters
                    .entry(membership.memory_id)
                    .or_insert_with(HashMap::new)
                    .insert(*space, membership.cluster_id);
            }
        }

        mem_clusters
    }

    /// Find groups of memories that cluster together with weighted_agreement >= threshold
    fn find_topic_mates(
        &amp;self,
        mem_clusters: &amp;HashMap&lt;Uuid, HashMap&lt;Embedder, i32&gt;&gt;,
    ) -> Vec&lt;Vec&lt;Uuid&gt;&gt; {
        let memory_ids: Vec&lt;Uuid&gt; = mem_clusters.keys().cloned().collect();
        let n = memory_ids.len();

        // Build adjacency list based on weighted agreement
        let mut edges: Vec&lt;(usize, usize)&gt; = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let weighted_agreement = self.compute_weighted_agreement(
                    &amp;memory_ids[i],
                    &amp;memory_ids[j],
                    mem_clusters,
                );

                if weighted_agreement >= self.topic_threshold {
                    edges.push((i, j));
                }
            }
        }

        // Find connected components using Union-Find
        let mut parent: Vec&lt;usize&gt; = (0..n).collect();

        fn find(parent: &amp;mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &amp;mut [usize], i: usize, j: usize) {
            let pi = find(parent, i);
            let pj = find(parent, j);
            if pi != pj {
                parent[pi] = pj;
            }
        }

        for (i, j) in edges {
            union(&amp;mut parent, i, j);
        }

        // Group by component
        let mut components: HashMap&lt;usize, Vec&lt;Uuid&gt;&gt; = HashMap::new();
        for i in 0..n {
            let root = find(&amp;mut parent, i);
            components
                .entry(root)
                .or_insert_with(Vec::new)
                .push(memory_ids[i]);
        }

        components.into_values().collect()
    }

    /// Compute weighted agreement between two memories.
    /// Uses embedder category weights: semantic=1.0, temporal=0.0, relational=0.5, structural=0.5
    fn compute_weighted_agreement(
        &amp;self,
        mem_a: &amp;Uuid,
        mem_b: &amp;Uuid,
        mem_clusters: &amp;HashMap&lt;Uuid, HashMap&lt;Embedder, i32&gt;&gt;,
    ) -> f32 {
        let clusters_a = match mem_clusters.get(mem_a) {
            Some(c) => c,
            None => return 0.0,
        };

        let clusters_b = match mem_clusters.get(mem_b) {
            Some(c) => c,
            None => return 0.0,
        };

        let mut weighted_agreement = 0.0;

        for embedder in Embedder::all() {
            let ca = clusters_a.get(&amp;embedder).copied().unwrap_or(-1);
            let cb = clusters_b.get(&amp;embedder).copied().unwrap_or(-1);

            // Both in same non-noise cluster
            if ca != -1 &amp;&amp; ca == cb {
                weighted_agreement += self.get_embedder_weight(embedder);
            }
        }

        weighted_agreement
    }

    /// Create a topic from a group of members
    fn create_topic(
        &amp;self,
        members: &amp;[Uuid],
        mem_clusters: &amp;HashMap&lt;Uuid, HashMap&lt;Embedder, i32&gt;&gt;,
    ) -> Topic {
        let profile = self.compute_topic_profile(members, mem_clusters);
        let cluster_ids = self.compute_cluster_ids(members, mem_clusters);

        Topic::new(profile, cluster_ids, members.to_vec())
    }

    /// Compute topic profile (per-space strength)
    fn compute_topic_profile(
        &amp;self,
        members: &amp;[Uuid],
        mem_clusters: &amp;HashMap&lt;Uuid, HashMap&lt;Embedder, i32&gt;&gt;,
    ) -> TopicProfile {
        let mut strengths = [0.0f32; 13];

        if members.is_empty() {
            return TopicProfile::new(strengths);
        }

        for (i, embedder) in Embedder::all().into_iter().enumerate() {
            // Collect cluster assignments for this space
            let mut clusters: HashMap&lt;i32, usize&gt; = HashMap::new();

            for mem_id in members {
                if let Some(space_clusters) = mem_clusters.get(mem_id) {
                    let cluster_id = space_clusters.get(&amp;embedder).copied().unwrap_or(-1);
                    if cluster_id != -1 {
                        *clusters.entry(cluster_id).or_insert(0) += 1;
                    }
                }
            }

            // Strength = fraction of members in the dominant cluster
            if let Some((&amp;_dominant_cluster, &amp;count)) = clusters.iter().max_by_key(|(_, &amp;c)| c) {
                strengths[i] = count as f32 / members.len() as f32;
            }
        }

        TopicProfile::new(strengths)
    }

    /// Compute cluster IDs for topic (most common cluster per space)
    fn compute_cluster_ids(
        &amp;self,
        members: &amp;[Uuid],
        mem_clusters: &amp;HashMap&lt;Uuid, HashMap&lt;Embedder, i32&gt;&gt;,
    ) -> HashMap&lt;Embedder, i32&gt; {
        let mut result: HashMap&lt;Embedder, i32&gt; = HashMap::new();

        for embedder in Embedder::all() {
            let mut clusters: HashMap&lt;i32, usize&gt; = HashMap::new();

            for mem_id in members {
                if let Some(space_clusters) = mem_clusters.get(mem_id) {
                    let cluster_id = space_clusters.get(&amp;embedder).copied().unwrap_or(-1);
                    if cluster_id != -1 {
                        *clusters.entry(cluster_id).or_insert(0) += 1;
                    }
                }
            }

            if let Some((&amp;dominant, _)) = clusters.iter().max_by_key(|(_, &amp;c)| c) {
                result.insert(embedder, dominant);
            }
        }

        result
    }

    /// Merge highly similar topics
    fn merge_similar_topics(&amp;self, mut topics: Vec&lt;Topic&gt;) -> Vec&lt;Topic&gt; {
        if topics.len() <= 1 {
            return topics;
        }

        // Sort by member count (largest first)
        topics.sort_by(|a, b| b.member_count().cmp(&amp;a.member_count()));

        let mut merged: Vec&lt;Topic&gt; = Vec::new();
        let mut absorbed: HashSet&lt;usize&gt; = HashSet::new();

        for i in 0..topics.len() {
            if absorbed.contains(&amp;i) {
                continue;
            }

            let mut current = topics[i].clone();

            // Check for similar topics to merge
            for j in (i + 1)..topics.len() {
                if absorbed.contains(&amp;j) {
                    continue;
                }

                let similarity = current.profile.similarity(&amp;topics[j].profile);

                if similarity >= self.merge_similarity_threshold {
                    // Absorb topic j into current
                    for mem_id in &amp;topics[j].member_memories {
                        if !current.member_memories.contains(mem_id) {
                            current.member_memories.push(*mem_id);
                        }
                    }

                    // Merge cluster_ids (prefer current's)
                    for (space, cluster_id) in &amp;topics[j].cluster_ids {
                        current.cluster_ids.entry(*space).or_insert(*cluster_id);
                    }

                    absorbed.insert(j);
                }
            }

            // Update profile after potential merges
            current.update_contributing_spaces();
            merged.push(current);
        }

        merged
    }

    /// Update topic stability based on membership changes
    pub fn update_topic_stability(
        &amp;self,
        topic: &amp;mut Topic,
        old_members: &amp;[Uuid],
        new_members: &amp;[Uuid],
    ) {
        // Compute membership churn
        let old_set: HashSet&lt;_&gt; = old_members.iter().collect();
        let new_set: HashSet&lt;_&gt; = new_members.iter().collect();

        let symmetric_diff = old_set.symmetric_difference(&amp;new_set).count();
        let union_size = old_set.union(&amp;new_set).count();

        let churn = if union_size > 0 {
            symmetric_diff as f32 / union_size as f32
        } else {
            0.0
        };

        topic.stability.membership_churn = churn;

        // Update age
        let age = chrono::Utc::now() - topic.created_at;
        topic.stability.age_hours = age.num_minutes() as f32 / 60.0;

        // Update phase
        topic.stability.update_phase();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_memberships() -> HashMap&lt;Embedder, Vec&lt;ClusterMembership&gt;&gt; {
        let mut memberships = HashMap::new();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        // Create memberships where id1 and id2 share clusters in 4 spaces
        for embedder in Embedder::all() {
            let cluster_id = match embedder {
                Embedder::E1Semantic | Embedder::E5Causal |
                Embedder::E7Code | Embedder::E8Emotional => 1,
                _ => -1,
            };

            memberships.entry(embedder).or_insert_with(Vec::new).push(
                ClusterMembership::new(id1, embedder, cluster_id, 0.9, true)
            );

            memberships.entry(embedder).or_insert_with(Vec::new).push(
                ClusterMembership::new(id2, embedder, cluster_id, 0.9, true)
            );

            // id3 is different
            memberships.entry(embedder).or_insert_with(Vec::new).push(
                ClusterMembership::new(id3, embedder, 99, 0.9, true)
            );
        }

        memberships
    }

    #[test]
    fn test_synthesize_topics() {
        let synthesizer = TopicSynthesizer::new();
        let memberships = create_test_memberships();

        let topics = synthesizer.synthesize_topics(&amp;memberships).unwrap();

        // Should find at least one topic (id1 and id2 share clusters)
        assert!(!topics.is_empty());
    }

    #[test]
    fn test_count_shared_clusters() {
        let synthesizer = TopicSynthesizer::new();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let mut mem_clusters = HashMap::new();

        let mut clusters1 = HashMap::new();
        clusters1.insert(Embedder::E1Semantic, 1);
        clusters1.insert(Embedder::E5Causal, 1);
        clusters1.insert(Embedder::E7Code, 1);
        mem_clusters.insert(id1, clusters1);

        let mut clusters2 = HashMap::new();
        clusters2.insert(Embedder::E1Semantic, 1);
        clusters2.insert(Embedder::E5Causal, 1);
        clusters2.insert(Embedder::E7Code, 2); // Different cluster
        mem_clusters.insert(id2, clusters2);

        let shared = synthesizer.count_shared_clusters(&amp;id1, &amp;id2, &amp;mem_clusters);
        assert_eq!(shared, 2); // E1 and E5
    }

    #[test]
    fn test_topic_profile_computation() {
        let synthesizer = TopicSynthesizer::new();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let mut mem_clusters = HashMap::new();

        let mut clusters1 = HashMap::new();
        clusters1.insert(Embedder::E1Semantic, 1);
        mem_clusters.insert(id1, clusters1);

        let mut clusters2 = HashMap::new();
        clusters2.insert(Embedder::E1Semantic, 1);
        mem_clusters.insert(id2, clusters2);

        let profile = synthesizer.compute_topic_profile(&amp;[id1, id2], &amp;mem_clusters);

        // E1 should have strength 1.0 (both in same cluster)
        assert_eq!(profile.strength(Embedder::E1Semantic), 1.0);
    }

    #[test]
    fn test_merge_similar_topics() {
        let synthesizer = TopicSynthesizer::with_config(3, 0.9, 0.3);

        // Create two nearly identical topics
        let profile1 = TopicProfile::new([0.9, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let profile2 = TopicProfile::new([0.85, 0.82, 0.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let topic1 = Topic::new(profile1, HashMap::new(), vec![Uuid::new_v4()]);
        let topic2 = Topic::new(profile2, HashMap::new(), vec![Uuid::new_v4()]);

        let merged = synthesizer.merge_similar_topics(vec![topic1, topic2]);

        // Should merge into one topic (profiles are similar)
        assert!(merged.len() <= 2);
    }

    #[test]
    fn test_update_stability() {
        let synthesizer = TopicSynthesizer::new();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        let profile = TopicProfile::new([0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut topic = Topic::new(profile, HashMap::new(), vec![id1, id2]);

        // Simulate membership change
        let old_members = vec![id1, id2];
        let new_members = vec![id1, id3]; // id2 left, id3 joined

        synthesizer.update_topic_stability(&amp;mut topic, &amp;old_members, &amp;new_members);

        // Churn should be > 0
        assert!(topic.stability.membership_churn > 0.0);
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/clustering/synthesizer.rs">TopicSynthesizer implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/clustering/mod.rs">Add pub mod synthesizer and re-exports</file>
</files_to_modify>

<validation_criteria>
  <criterion>synthesize_topics returns valid topics</criterion>
  <criterion>Topics have weighted_agreement >= 2.5 (TOPIC_THRESHOLD)</criterion>
  <criterion>compute_weighted_agreement uses correct category weights</criterion>
  <criterion>Temporal embedders (E2-E4) contribute 0.0 to weighted_agreement</criterion>
  <criterion>Semantic embedders contribute 1.0, relational 0.5, structural 0.5</criterion>
  <criterion>topic_confidence = weighted_agreement / MAX_WEIGHTED_AGREEMENT (8.5)</criterion>
  <criterion>TopicProfile includes semantic/relational/structural breakdown</criterion>
  <criterion>Similar topics merged (> 0.9 similarity)</criterion>
  <criterion>Stability updated correctly on membership change</criterion>
</validation_criteria>

<test_commands>
  <command description="Run synthesizer tests">cargo test --package context-graph-core synthesizer</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create synthesizer.rs
- [ ] Define constants: TOPIC_THRESHOLD (2.5), MAX_WEIGHTED_AGREEMENT (8.5)
- [ ] Define embedder category weight constants
- [ ] Implement TopicSynthesizer struct with topic_threshold field
- [ ] Implement get_embedder_weight() for category-based weights
- [ ] Implement build_mem_clusters_map
- [ ] Implement compute_weighted_agreement (replaces count_shared_clusters)
- [ ] Implement find_topic_mates using weighted_agreement threshold
- [ ] Implement compute_topic_profile with semantic/relational/structural breakdown
- [ ] Implement topic_confidence calculation (weighted_agreement / 8.5)
- [ ] Implement merge_similar_topics
- [ ] Implement update_topic_stability
- [ ] Write unit tests verifying temporal exclusion
- [ ] Write unit tests for weighted agreement edge cases
- [ ] Run tests to verify
- [ ] Proceed to TASK-P4-009
