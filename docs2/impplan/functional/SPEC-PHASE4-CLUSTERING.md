# Functional Specification: Phase 4 - Multi-Space Clustering

```xml
<functional_spec id="SPEC-PHASE4" version="1.0">
<metadata>
  <title>Multi-Space Clustering (HDBSCAN + BIRCH)</title>
  <status>approved</status>
  <owner>Context Graph Team</owner>
  <created>2026-01-16</created>
  <last_updated>2026-01-16</last_updated>
  <revision_notes>Updated to use weighted_agreement formula instead of raw space counts for topic synthesis</revision_notes>
  <implements>impplan.md Part 4, Part 8</implements>
  <depends_on>
    <spec_ref>SPEC-PHASE0</spec_ref>
    <spec_ref>SPEC-PHASE1</spec_ref>
    <spec_ref>SPEC-PHASE2</spec_ref>
    <spec_ref>SPEC-PHASE3</spec_ref>
  </depends_on>
  <related_specs>
    <spec_ref>SPEC-PHASE5</spec_ref>
  </related_specs>
</metadata>

<overview>
Implement density-based clustering using HDBSCAN for batch clustering and BIRCH CF-trees for online updates. The system maintains **13 parallel clustering spaces** (one per embedder), then synthesizes cross-space **Topics** from memories using weighted agreement across non-temporal spaces.

This replaces the current K-means clustering with:
1. **HDBSCAN**: Hierarchical Density-Based Spatial Clustering for batch re-clustering
2. **BIRCH**: Balanced Iterative Reducing and Clustering using Hierarchies for incremental updates
3. **Topic Synthesis**: Cross-space topic discovery using weighted agreement (threshold >= 2.5)

**Clustering Scope**: HDBSCAN and BIRCH clustering operates on all 13 embedding spaces. However, topic synthesis only considers non-temporal weighted agreement, excluding E2-E4 (temporal embedders) from the topic threshold calculation.

**Problem Solved**: Current K-means clustering requires specifying cluster count upfront. HDBSCAN discovers natural cluster structure. BIRCH enables real-time updates without full re-clustering.

**Who Benefits**: The injection system which can retrieve cluster-related memories; the stability tracking system which monitors topic evolution.
</overview>

<user_stories>
<story id="US-P4-01" priority="must-have">
  <narrative>
    As a clustering system
    I want to maintain separate clusters in each of the 13 embedding spaces
    So that I can identify memories similar in specific dimensions
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P4-01-01">
      <given>100 memories with all 13 embeddings</given>
      <when>Running batch clustering</when>
      <then>13 separate HDBSCAN results are produced, one per embedder</then>
    </criterion>
    <criterion id="AC-P4-01-02">
      <given>A cluster in E7 (Code) space</given>
      <when>Querying its membership</when>
      <then>Only memories with similar code patterns are in the cluster</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P4-02" priority="must-have">
  <narrative>
    As a real-time system
    I want to incrementally update clusters when new memories arrive
    So that I don't have to re-cluster the entire database
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P4-02-01">
      <given>Existing clusters from 1000 memories</given>
      <when>A new memory is captured</when>
      <then>Memory is assigned to existing clusters via BIRCH in &lt;10ms</then>
    </criterion>
    <criterion id="AC-P4-02-02">
      <given>BIRCH tree with existing clusters</given>
      <when>100 new memories arrive in batch</when>
      <then>All are assigned or form new clusters without full re-clustering</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P4-03" priority="must-have">
  <narrative>
    As a topic discovery system
    I want to synthesize topics from memories with weighted_agreement >= 2.5
    So that I can identify robust cross-dimensional themes weighted by embedder importance
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P4-03-01">
      <given>Memories A, B, C clustering together in E1, E5, E7, and E10 (all semantic, weight=1.0 each)</given>
      <when>Running topic synthesis</when>
      <then>A Topic is created with weighted_agreement=4.0, confidence=4.0/8.5≈0.47 and topic_profile showing strength in those 4 spaces</then>
    </criterion>
    <criterion id="AC-P4-03-02">
      <given>Memories clustering together only in E8 (relational, weight=0.5) and E9 (structural, weight=0.5)</given>
      <when>Running topic synthesis</when>
      <then>No Topic is created (weighted_agreement=1.0, below 2.5 threshold)</then>
    </criterion>
    <criterion id="AC-P4-03-03">
      <given>Memories clustering together in E2, E3, E4 (temporal, weight=0.0 each)</given>
      <when>Running topic synthesis</when>
      <then>No Topic is created (weighted_agreement=0.0, temporal embedders excluded)</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P4-04" priority="must-have">
  <narrative>
    As a stability monitor
    I want to track topic churn over time
    So that I can detect when the knowledge structure is changing rapidly
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P4-04-01">
      <given>Topics T1, T2, T3 existed 1 hour ago</given>
      <when>Current topics are T1, T4, T5 (T2, T3 gone; T4, T5 new)</when>
      <then>Churn rate = 4/5 = 0.8 (4 changes out of 5 total)</then>
    </criterion>
  </acceptance_criteria>
</story>
</user_stories>

<requirements>
<requirement id="REQ-P4-01" story_ref="US-P4-01" priority="must">
  <description>Implement HDBSCAN for batch clustering per embedding space</description>
  <rationale>HDBSCAN discovers natural clusters without specifying count</rationale>
  <params>
    min_cluster_size: 3
    min_samples: 2
    cluster_selection_method: "eom" (Excess of Mass)
    metric: per-embedder (cosine/jaccard/hamming/maxsim/transe)
  </params>
  <algorithm>
    1. Extract embeddings for one space from all memories
    2. Compute pairwise distances using appropriate metric
    3. Build minimum spanning tree
    4. Build cluster hierarchy
    5. Extract flat clusters using EOM
    6. Return cluster labels (-1 = noise)
  </algorithm>
</requirement>

<requirement id="REQ-P4-02" story_ref="US-P4-02" priority="must">
  <description>Implement BIRCH CF-trees for online cluster updates</description>
  <rationale>BIRCH enables incremental clustering without full re-computation</rationale>
  <params>
    branching_factor: 50
    threshold: adaptive (starts at 0.3, adjusts based on cluster quality)
    leaf_capacity: 100
  </params>
  <algorithm>
    1. Maintain 13 CF-trees (one per embedder)
    2. On new memory: insert into each tree
    3. If leaf overflows: split or increase threshold
    4. Periodically: run HDBSCAN on CF entries (not raw data) for global structure
  </algorithm>
</requirement>

<requirement id="REQ-P4-03" story_ref="US-P4-03" priority="must">
  <description>Implement cross-space Topic synthesis using weighted agreement</description>
  <rationale>Topics must emerge from weighted multi-space agreement, prioritizing semantic embedders</rationale>
  <weighted_agreement_formula>
    ```
    weighted_agreement = Σ(topic_weight_i × is_clustered_i)

    Where topic_weight by embedder category:
    - Semantic (E1, E5-E7, E10, E12-E13): 1.0
    - Relational (E8, E11): 0.5
    - Structural (E9): 0.5
    - Temporal (E2-E4): 0.0 (excluded from topic synthesis)

    Topic threshold: weighted_agreement >= 2.5
    Max possible weighted_agreement: 7×1.0 + 2×0.5 + 1×0.5 = 8.5
    ```
  </weighted_agreement_formula>
  <algorithm>
    ```
    const TOPIC_WEIGHTS: [f32; 13] = [
      1.0,  // E1: Semantic
      0.0,  // E2: Temporal (excluded)
      0.0,  // E3: Temporal (excluded)
      0.0,  // E4: Temporal (excluded)
      1.0,  // E5: Semantic
      1.0,  // E6: Semantic
      1.0,  // E7: Semantic
      0.5,  // E8: Relational
      0.5,  // E9: Structural
      1.0,  // E10: Semantic
      0.5,  // E11: Relational
      1.0,  // E12: Semantic
      1.0,  // E13: Semantic
    ];
    const MAX_WEIGHTED_AGREEMENT: f32 = 8.5;
    const TOPIC_THRESHOLD: f32 = 2.5;

    fn synthesize_topics(clusters: &amp;[PerSpaceClusters]) -&gt; Vec&lt;Topic&gt; {
      let mut topics = vec![];
      for memory_set in find_memory_groups_in_same_cluster() {
        let weighted_agreement = compute_weighted_agreement(memory_set, clusters);
        if weighted_agreement &gt;= TOPIC_THRESHOLD {
          let confidence = weighted_agreement / MAX_WEIGHTED_AGREEMENT;
          let profile = compute_topic_profile(memory_set, clusters);
          topics.push(Topic { members: memory_set, confidence, profile });
        }
      }
      topics
    }

    fn compute_weighted_agreement(memory_set: &amp;[MemoryId], clusters: &amp;[PerSpaceClusters]) -&gt; f32 {
      let mut weighted_sum = 0.0;
      for (i, space_clusters) in clusters.iter().enumerate() {
        if memories_cluster_together(memory_set, space_clusters) {
          weighted_sum += TOPIC_WEIGHTS[i];
        }
      }
      weighted_sum
    }
    ```
  </algorithm>
</requirement>

<requirement id="REQ-P4-04" story_ref="US-P4-03" priority="must">
  <description>Define Topic and TopicProfile structures</description>
  <rationale>Need structured representation of emergent topics</rationale>
  <schema>
    Topic {
      id: TopicId,
      members: Vec&lt;MemoryId&gt;,
      weighted_agreement: f32,   // Sum of topic_weights for agreeing spaces
      confidence: f32,           // weighted_agreement / 8.5
      profile: TopicProfile,
      created_at: DateTime&lt;Utc&gt;,
      last_updated: DateTime&lt;Utc&gt;,
      access_count: u64,
      phase: TopicPhase,
    }

    TopicProfile {
      strengths: [f32; 13],      // Per-space cluster tightness
      centroid: TeleologicalArray, // Representative embedding
    }

    TopicPhase {
      Emerging,   // &lt;1 hour old, &lt;5 members
      Stable,     // &gt;1 hour old, silhouette &gt; 0.3
      Declining,  // No new members in 24h
      Merging,    // Being absorbed by another topic
    }
  </schema>
</requirement>

<requirement id="REQ-P4-TOPIC-01" story_ref="US-P4-03" priority="must">
  <description>Topics require weighted_agreement >= 2.5</description>
  <rationale>Ensures topics have sufficient semantic support, not just temporal or weak relational clustering</rationale>
  <threshold>2.5</threshold>
  <examples>
    - 3 semantic spaces (3×1.0 = 3.0) → PASS
    - 2 semantic + 1 relational (2×1.0 + 0.5 = 2.5) → PASS
    - 2 semantic only (2×1.0 = 2.0) → FAIL
    - 5 relational/structural (5×0.5 = 2.5) → PASS
    - All 3 temporal (3×0.0 = 0.0) → FAIL
  </examples>
</requirement>

<requirement id="REQ-P4-TOPIC-02" story_ref="US-P4-03" priority="must">
  <description>Temporal embedders (E2-E4) excluded from topic synthesis</description>
  <rationale>Temporal clustering indicates co-occurrence in time, not semantic relatedness. Topics should represent conceptual themes, not temporal coincidence.</rationale>
  <excluded_embedders>
    - E2: Temporal context
    - E3: Temporal patterns
    - E4: Temporal decay
  </excluded_embedders>
  <note>HDBSCAN and BIRCH still cluster in temporal spaces for other purposes (e.g., session detection), but temporal clustering does not contribute to topic formation.</note>
</requirement>

<requirement id="REQ-P4-TOPIC-03" story_ref="US-P4-03" priority="must">
  <description>Topic confidence = weighted_agreement / 8.5</description>
  <rationale>Normalizes confidence to [0, 1] range based on maximum possible weighted agreement</rationale>
  <formula>
    confidence = weighted_agreement / MAX_WEIGHTED_AGREEMENT
    where MAX_WEIGHTED_AGREEMENT = 7×1.0 + 2×0.5 + 1×0.5 = 8.5
  </formula>
  <examples>
    - weighted_agreement = 8.5 → confidence = 1.0 (perfect agreement in all weighted spaces)
    - weighted_agreement = 4.0 → confidence = 0.47 (moderate agreement)
    - weighted_agreement = 2.5 → confidence = 0.29 (minimum threshold)
  </examples>
</requirement>

<requirement id="REQ-P4-05" story_ref="US-P4-04" priority="must">
  <description>Implement TopicStabilityTracker for churn monitoring</description>
  <rationale>Need to detect when knowledge structure is unstable</rationale>
  <schema>
    TopicStabilityTracker {
      topics: HashMap&lt;TopicId, TopicMetrics&gt;,
      history: VecDeque&lt;TopicSnapshot&gt;,
      max_history: 100,
    }

    TopicMetrics {
      id: TopicId,
      age: Duration,
      membership_stability: f32,  // How stable is member set
      centroid_stability: f32,    // How stable is centroid
      access_frequency: f32,
      last_accessed: DateTime&lt;Utc&gt;,
      phase: TopicPhase,
    }
  </schema>
  <formula>
    churn_rate = |topics_added ∪ topics_removed| / |current_topics ∪ previous_topics|
  </formula>
</requirement>

<requirement id="REQ-P4-06" story_ref="US-P4-01" priority="must">
  <description>Implement MultiSpaceClusterManager orchestration</description>
  <rationale>Need unified interface for 13 parallel clustering systems</rationale>
  <methods>
    - batch_cluster(memories: &amp;[Memory]) -&gt; MultiSpaceClusterResult
    - insert_memory(memory: &amp;Memory) -&gt; Vec&lt;(Embedder, ClusterId)&gt;
    - get_cluster_members(embedder: Embedder, cluster_id: ClusterId) -&gt; Vec&lt;MemoryId&gt;
    - synthesize_topics() -&gt; Vec&lt;Topic&gt;
    - get_memory_clusters(memory_id: MemoryId) -&gt; HashMap&lt;Embedder, Option&lt;ClusterId&gt;&gt;
    - compute_silhouette_scores() -&gt; [f32; 13]
  </methods>
</requirement>

<requirement id="REQ-P4-07" story_ref="US-P4-01" priority="must">
  <description>Define silhouette score threshold for cluster quality</description>
  <rationale>Reject low-quality clusters</rationale>
  <threshold>silhouette_score &gt; 0.3</threshold>
  <behavior>
    Clusters with silhouette &lt;= 0.3 are marked as "weak" and not used for topic synthesis.
    Weak clusters are still stored but flagged.
  </behavior>
</requirement>

<requirement id="REQ-P4-08" story_ref="US-P4-04" priority="must">
  <description>Define dream triggers based on stability metrics</description>
  <rationale>High entropy + high churn should trigger consolidation</rationale>
  <triggers>
    - entropy &gt; 0.7 for 5+ min → MAY trigger dream
    - churn &gt; 0.5 AND entropy &gt; 0.7 → MAY trigger dream
  </triggers>
  <note>These replace the IC-based dream triggers from the removed North Star system.</note>
</requirement>
</requirements>

<edge_cases>
<edge_case id="EC-P4-01" req_ref="REQ-P4-01">
  <scenario>Fewer than 3 memories (min_cluster_size)</scenario>
  <expected_behavior>All memories assigned to noise cluster (-1). No topics formed. This is valid state for new system.</expected_behavior>
</edge_case>

<edge_case id="EC-P4-02" req_ref="REQ-P4-01">
  <scenario>All memories identical in one space</scenario>
  <expected_behavior>Single cluster formed in that space with all members. Silhouette undefined (single cluster), marked as weak.</expected_behavior>
</edge_case>

<edge_case id="EC-P4-03" req_ref="REQ-P4-03">
  <scenario>Memory is noise (-1) in all 13 spaces</scenario>
  <expected_behavior>Memory belongs to no topic. This is valid for outlier content.</expected_behavior>
</edge_case>

<edge_case id="EC-P4-04" req_ref="REQ-P4-02">
  <scenario>BIRCH tree exceeds memory limit</scenario>
  <expected_behavior>Increase threshold to merge CF entries. Log warning: "BIRCH threshold increased to [value] for [embedder]"</expected_behavior>
</edge_case>

<edge_case id="EC-P4-05" req_ref="REQ-P4-03">
  <scenario>Topic members all deleted</scenario>
  <expected_behavior>Topic marked as Declining then removed on next synthesis. Not an error.</expected_behavior>
</edge_case>

<edge_case id="EC-P4-06" req_ref="REQ-P4-05">
  <scenario>Churn rate = 1.0 (all topics changed)</scenario>
  <expected_behavior>Warning logged: "High topic churn detected (1.0)". Dream trigger condition met if entropy also high.</expected_behavior>
</edge_case>
</edge_cases>

<error_states>
<error id="ERR-P4-01" http_code="500">
  <condition>HDBSCAN fails to converge</condition>
  <message>HDBSCAN failed for embedder [name]: [error]</message>
  <recovery>Use previous cluster state for that embedder. Log error. Retry on next batch.</recovery>
</error>

<error id="ERR-P4-02" http_code="500">
  <condition>BIRCH tree corruption detected</condition>
  <message>BIRCH tree corrupted for embedder [name]: CF entries invalid</message>
  <recovery>Rebuild tree from scratch using batch HDBSCAN. Log error with diagnostic info.</recovery>
</error>

<error id="ERR-P4-03" http_code="400">
  <condition>min_cluster_size set to &lt;2</condition>
  <message>Invalid config: min_cluster_size must be &gt;= 2</message>
  <recovery>Use default value (3). Log warning.</recovery>
</error>
</error_states>

<test_plan>
<test_case id="TC-P4-01" type="unit" req_ref="REQ-P4-01">
  <description>HDBSCAN produces 13 separate clustering results</description>
  <inputs>["100 memories with varied content"]</inputs>
  <expected>13 PerSpaceClusters structures returned</expected>
</test_case>

<test_case id="TC-P4-02" type="unit" req_ref="REQ-P4-01">
  <description>HDBSCAN respects min_cluster_size=3</description>
  <inputs>["10 memories forming 2 clusters of size 2"]</inputs>
  <expected>All memories assigned to noise (-1)</expected>
</test_case>

<test_case id="TC-P4-03" type="unit" req_ref="REQ-P4-02">
  <description>BIRCH assigns new memory to existing cluster</description>
  <inputs>["Existing clusters from 100 memories", "New memory similar to cluster centroid"]</inputs>
  <expected>Memory assigned to nearest cluster, not noise</expected>
</test_case>

<test_case id="TC-P4-04" type="unit" req_ref="REQ-P4-03">
  <description>Topic synthesized from semantic space agreement (weighted >= 2.5)</description>
  <inputs>["5 memories clustering together in E1, E5, E7, E10 (all semantic, weight=1.0)"]</inputs>
  <expected>Topic created with weighted_agreement=4.0, confidence=4.0/8.5≈0.47, profile shows E1,E5,E7,E10 strengths</expected>
</test_case>

<test_case id="TC-P4-05" type="unit" req_ref="REQ-P4-TOPIC-01">
  <description>No topic from insufficient weighted agreement</description>
  <inputs>["5 memories clustering together only in E1, E5 (2×1.0=2.0)"]</inputs>
  <expected>No topic created (weighted_agreement=2.0 below 2.5 threshold)</expected>
</test_case>

<test_case id="TC-P4-09" type="unit" req_ref="REQ-P4-TOPIC-02">
  <description>Temporal embedders excluded from topic synthesis</description>
  <inputs>["5 memories clustering together in E2, E3, E4 (all temporal, weight=0.0)"]</inputs>
  <expected>No topic created (weighted_agreement=0.0, temporal embedders excluded)</expected>
</test_case>

<test_case id="TC-P4-10" type="unit" req_ref="REQ-P4-TOPIC-03">
  <description>Topic confidence calculation uses weighted formula</description>
  <inputs>["Memories with weighted_agreement=4.25 (3 semantic + 1 relational + 1 structural)"]</inputs>
  <expected>confidence = 4.25/8.5 = 0.5</expected>
</test_case>

<test_case id="TC-P4-11" type="unit" req_ref="REQ-P4-TOPIC-01">
  <description>Relational/structural spaces can form topics at threshold</description>
  <inputs>["5 memories clustering in E8, E9, E11 (relational/structural) plus E1, E5 (semantic)"]</inputs>
  <expected>weighted_agreement = 2×1.0 + 3×0.5 = 3.5 >= 2.5, topic created</expected>
</test_case>

<test_case id="TC-P4-06" type="unit" req_ref="REQ-P4-05">
  <description>Churn rate computed correctly</description>
  <inputs>["Previous: [T1,T2,T3], Current: [T1,T4,T5]"]</inputs>
  <expected>churn_rate = 4/5 = 0.8</expected>
</test_case>

<test_case id="TC-P4-07" type="integration" req_ref="REQ-P4-02">
  <description>BIRCH insertion completes in &lt;10ms</description>
  <inputs>["1000 existing memories", "1 new memory"]</inputs>
  <expected>insertion_time &lt; 10ms</expected>
</test_case>

<test_case id="TC-P4-08" type="unit" req_ref="REQ-P4-07">
  <description>Silhouette threshold filters weak clusters</description>
  <inputs>["Cluster with silhouette=0.25"]</inputs>
  <expected>Cluster marked as weak, not used for topic synthesis</expected>
</test_case>
</test_plan>

<validation_criteria>
  <criterion>13 parallel HDBSCAN clustering spaces operational</criterion>
  <criterion>BIRCH incremental updates complete in &lt;10ms</criterion>
  <criterion>Topics synthesized from weighted_agreement >= 2.5</criterion>
  <criterion>Temporal embedders (E2-E4) excluded from topic synthesis (weight=0.0)</criterion>
  <criterion>Topic confidence = weighted_agreement / 8.5</criterion>
  <criterion>Semantic embedders contribute weight=1.0, relational/structural contribute weight=0.5</criterion>
  <criterion>Topic profiles show per-space strengths</criterion>
  <criterion>Churn rate computed correctly</criterion>
  <criterion>Silhouette threshold (0.3) filters weak clusters</criterion>
  <criterion>Dream triggers based on entropy + churn</criterion>
</validation_criteria>
</functional_spec>
```

## Clustering Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MultiSpaceClusterManager                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐       ┌─────────────┐                │
│  │ E1 Semantic │  │ E2 Temporal │  ...  │ E13 SPLADE  │                │
│  │ HDBSCAN     │  │ HDBSCAN     │       │ HDBSCAN     │                │
│  │ BIRCH       │  │ BIRCH       │       │ BIRCH       │                │
│  └─────────────┘  └─────────────┘       └─────────────┘                │
│        │               │                      │                         │
│        └───────────────┼──────────────────────┘                         │
│                        │                                                │
│                        ▼                                                │
│              ┌─────────────────────┐                                    │
│              │  Topic Synthesizer  │                                    │
│              │ (weighted >= 2.5)   │                                    │
│              │ (excludes E2-E4)    │                                    │
│              └─────────────────────┘                                    │
│                        │                                                │
│                        ▼                                                │
│              ┌─────────────────────┐                                    │
│              │ TopicStabilityTracker│                                   │
│              │ (churn monitoring)   │                                   │
│              └─────────────────────┘                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Topic Synthesis Example

**Input**: 5 memories (M1-M5) with cluster assignments:

| Memory | E1 (1.0) | E2 (0.0) | E3 (0.0) | E4 (0.0) | E5 (1.0) | E6 (1.0) | E7 (1.0) | E8 (0.5) | E9 (0.5) | E10 (1.0) | E11 (0.5) | E12 (1.0) | E13 (1.0) |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|-----------|-----------|-----------|
| M1 | C1 | C1 | C1 | C1 | C1 | C2 | C1 | C3 | C4 | C1 | C5 | C2 | C2 |
| M2 | C1 | C1 | C1 | C1 | C1 | C2 | C1 | C3 | C4 | C1 | C5 | C2 | C2 |
| M3 | C1 | C1 | C1 | C1 | C1 | C2 | C1 | C3 | C4 | C1 | C5 | C2 | C2 |
| M4 | C2 | C1 | C1 | C1 | C1 | C3 | C2 | C3 | C5 | C2 | C5 | C3 | C3 |
| M5 | C2 | C2 | C2 | C2 | C2 | C3 | C2 | C4 | C5 | C2 | C6 | C3 | C3 |

**Weighted Agreement Calculation for M1, M2, M3**:
- E1 (Semantic): clustered together, weight=1.0 -> +1.0
- E2 (Temporal): clustered together, weight=0.0 -> +0.0 (excluded)
- E3 (Temporal): clustered together, weight=0.0 -> +0.0 (excluded)
- E4 (Temporal): clustered together, weight=0.0 -> +0.0 (excluded)
- E5 (Semantic): clustered together, weight=1.0 -> +1.0
- E6 (Semantic): clustered together, weight=1.0 -> +1.0
- E7 (Semantic): clustered together, weight=1.0 -> +1.0
- E8 (Relational): clustered together, weight=0.5 -> +0.5
- E9 (Structural): clustered together, weight=0.5 -> +0.5
- E10 (Semantic): clustered together, weight=1.0 -> +1.0
- E11 (Relational): clustered together, weight=0.5 -> +0.5
- E12 (Semantic): clustered together, weight=1.0 -> +1.0
- E13 (Semantic): clustered together, weight=1.0 -> +1.0

**Total weighted_agreement** = 1.0+1.0+1.0+1.0+0.5+0.5+1.0+0.5+1.0+1.0 = **8.5** (max possible)

**Topic Synthesis**:
- M1, M2, M3: weighted_agreement=8.5 >= 2.5 -> **Topic T1**
- M4, M5: weighted_agreement = E5(1.0) + E8(0.5) + E11(0.5) = 2.0 < 2.5 -> **No Topic**

**Topic T1**:
```
Topic {
  id: "topic-001",
  members: [M1, M2, M3],
  weighted_agreement: 8.5,
  confidence: 8.5 / 8.5 = 1.0,
  profile: TopicProfile {
    strengths: [0.85, 0.0, 0.0, 0.0, 0.80, 0.75, 0.90, 0.60, 0.55, 0.75, 0.50, 0.70, 0.65],
    centroid: average(M1.teleological_array, M2.teleological_array, M3.teleological_array)
  },
  phase: Emerging
}
```

**Note**: Although M1-M3 cluster together in E2-E4 (temporal spaces), these contribute 0.0 to weighted_agreement. HDBSCAN/BIRCH clustering operates on all 13 spaces, but topic synthesis only considers non-temporal weighted agreement.

## Progressive Feature Activation by Tier

| Tier | Memory Count | Features |
|------|--------------|----------|
| 0 | 0 | Storage, basic retrieval |
| 1 | 1-2 | Pairwise similarity |
| 2 | 3-9 | Basic HDBSCAN (may produce no clusters) |
| 3 | 10-29 | Multiple clusters possible, divergence detection |
| 4 | 30-99 | Reliable statistics, topic synthesis starts |
| 5 | 100-499 | Sub-clustering, trend analysis |
| 6 | 500+ | Full personalization |

**Tier 0-2 Defaults**: Cluster=-1, TopicProfile=[0.5;13], Stability=1.0
