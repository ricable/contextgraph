# Functional Specification: Phase 3 - Similarity & Divergence Detection

```xml
<functional_spec id="SPEC-PHASE3" version="1.0">
<metadata>
  <title>Similarity and Divergence Detection</title>
  <status>approved</status>
  <owner>Context Graph Team</owner>
  <created>2026-01-16</created>
  <last_updated>2026-01-16</last_updated>
  <implements>impplan.md Part 2 (Similarity Detection, Divergence Detection, Multi-Space Relevance)</implements>
  <depends_on>
    <spec_ref>SPEC-PHASE0</spec_ref>
    <spec_ref>SPEC-PHASE1</spec_ref>
    <spec_ref>SPEC-PHASE2</spec_ref>
  </depends_on>
  <related_specs>
    <spec_ref>SPEC-PHASE4</spec_ref>
    <spec_ref>SPEC-PHASE5</spec_ref>
  </related_specs>
</metadata>

<overview>
Implement per-space similarity thresholds and divergence detection for the 13-embedding system. The key insight is that a memory is **relevant** if it shows high similarity in **ANY NON-TEMPORAL** embedding space (not all). Conversely, **divergence** is detected when current activity has **LOW** similarity to recent memories in SEMANTIC spaces only.

**Important**: Temporal spaces (E2-TempRecent, E3-TempPeriodic, E4-TempPosition) are EXCLUDED from similarity and divergence detection. These spaces encode when something happened, not what it contains, making them unsuitable for content-based similarity matching.

This enables:
1. Multi-perspective retrieval (find memories similar in code space OR semantic space OR causal space)
2. Activity shift detection (alert when working on something very different from recent context)
3. Weighted relevance scoring across 10 non-temporal spaces

**Problem Solved**: Current retrieval uses single-space similarity. This misses memories that are relevant in non-semantic ways (e.g., similar code patterns, similar causal structure).

**Who Benefits**: Claude instances that get more comprehensive context from memories similar in any dimension; users who are alerted when activities diverge from recent work.
</overview>

<user_stories>
<story id="US-P3-01" priority="must-have">
  <narrative>
    As a retrieval system
    I want to find memories similar in ANY non-temporal embedding space
    So that I can surface relevant context even if only one dimension matches
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P3-01-01">
      <given>A query and a memory that are 0.9 similar in E7 (Code) but only 0.3 similar in E1 (Semantic)</given>
      <when>Running similarity detection</when>
      <then>Memory is marked as relevant because E7 &gt; threshold_E7 (0.80)</then>
    </criterion>
    <criterion id="AC-P3-01-02">
      <given>A query and a memory that are below threshold in ALL 10 non-temporal spaces</given>
      <when>Running similarity detection</when>
      <then>Memory is marked as NOT relevant (temporal spaces E2-E4 are excluded from consideration)</then>
    </criterion>
    <criterion id="AC-P3-01-03">
      <given>A query and a memory that are 0.95 similar in E2 (TempRecent) but below threshold in all non-temporal spaces</given>
      <when>Running similarity detection</when>
      <then>Memory is marked as NOT relevant because temporal spaces are excluded</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P3-02" priority="must-have">
  <narrative>
    As a context system
    I want to detect when current activity diverges from recent work in SEMANTIC spaces
    So that I can alert about potential context switches
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P3-02-01">
      <given>Current query with 0.15 similarity to recent memories in E1 (Semantic)</given>
      <when>Running divergence detection</when>
      <then>Divergence alert generated: "Low semantic similarity to recent activity"</then>
    </criterion>
    <criterion id="AC-P3-02-02">
      <given>Current query with &gt;0.3 similarity to recent memories in ALL semantic spaces (E1, E5-E7, E10, E12-E13)</given>
      <when>Running divergence detection</when>
      <then>No divergence alert generated</then>
    </criterion>
    <criterion id="AC-P3-02-03">
      <given>Current query with 0.05 similarity to recent memories in E2 (TempRecent)</given>
      <when>Running divergence detection</when>
      <then>No divergence alert generated because temporal spaces (E2-E4) are excluded from divergence detection</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P3-03" priority="must-have">
  <narrative>
    As a ranking system
    I want to compute multi-space relevance scores using only NON-TEMPORAL spaces
    So that I can rank memories by how many spaces they match and by what margin
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P3-03-01">
      <given>Memory A matching in 5 non-temporal spaces, Memory B matching in 2 non-temporal spaces</given>
      <when>Computing relevance scores</when>
      <then>Memory A has higher relevance score than Memory B</then>
    </criterion>
    <criterion id="AC-P3-03-02">
      <given>Memory with 0.95 similarity in E1 (weight 1.0)</given>
      <when>Computing relevance score</when>
      <then>Score includes (1.0 × (0.95 - 0.75)) = 0.20 from E1</then>
    </criterion>
    <criterion id="AC-P3-03-03">
      <given>Memory with 0.99 similarity in E2 (TempRecent, weight 0.0)</given>
      <when>Computing relevance score</when>
      <then>E2 contributes 0.0 to the score because temporal spaces have weight 0.0</then>
    </criterion>
    <criterion id="AC-P3-03-04">
      <given>multi_space_relevance calculation</given>
      <when>Computing weighted sum</when>
      <then>multi_space_relevance = weighted_sum / active_space_count where temporal spaces (E2-E4) are excluded</then>
    </criterion>
  </acceptance_criteria>
</story>
</user_stories>

<requirements>
<requirement id="REQ-P3-01" story_ref="US-P3-01" priority="must">
  <description>Define per-space similarity thresholds (High Similarity) - Temporal spaces EXCLUDED</description>
  <rationale>Different spaces have different natural similarity distributions. Temporal spaces (E2-E4) encode WHEN something happened, not WHAT it contains, making them unsuitable for content-based similarity matching.</rationale>
  <thresholds>
    <!-- SEMANTIC SPACES (used for similarity detection) -->
    E1_Semantic:     high_threshold = 0.75
    E5_Causal:       high_threshold = 0.70
    E6_Sparse:       high_threshold = 0.60
    E7_Code:         high_threshold = 0.80
    E10_Multimodal:  high_threshold = 0.70
    E12_LateInteract:high_threshold = 0.70
    E13_SPLADE:      high_threshold = 0.60

    <!-- RELATIONAL SPACES (used for similarity detection) -->
    E8_Emotional:    high_threshold = 0.70
    E11_Entity:      high_threshold = 0.70

    <!-- STRUCTURAL SPACE (used for similarity detection) -->
    E9_HDC:          high_threshold = 0.70

    <!-- TEMPORAL SPACES (EXCLUDED from similarity detection) -->
    E2_TempRecent:   EXCLUDED (weight = 0.0)
    E3_TempPeriodic: EXCLUDED (weight = 0.0)
    E4_TempPosition: EXCLUDED (weight = 0.0)
  </thresholds>
  <note>Only 10 non-temporal spaces are evaluated for similarity. A memory matching threshold in E2-E4 ONLY is NOT considered relevant.</note>
</requirement>

<requirement id="REQ-P3-02" story_ref="US-P3-02" priority="must">
  <description>Define per-space divergence thresholds (Low Similarity) - Temporal spaces EXCLUDED</description>
  <rationale>Low similarity to recent activity indicates context switch. Divergence is detected when query similarity drops significantly in SEMANTIC spaces only. Temporal spaces (E2-E4) are NOT considered for divergence alerts as they encode time, not content.</rationale>
  <thresholds>
    <!-- SEMANTIC SPACES (used for divergence detection) -->
    E1_Semantic:     low_threshold = 0.30
    E5_Causal:       low_threshold = 0.25
    E6_Sparse:       low_threshold = 0.20
    E7_Code:         low_threshold = 0.35
    E10_Multimodal:  low_threshold = 0.30
    E12_LateInteract:low_threshold = 0.30
    E13_SPLADE:      low_threshold = 0.20

    <!-- RELATIONAL SPACES (used for divergence detection) -->
    E8_Emotional:    low_threshold = 0.30
    E11_Entity:      low_threshold = 0.30

    <!-- STRUCTURAL SPACE (used for divergence detection) -->
    E9_HDC:          low_threshold = 0.30

    <!-- TEMPORAL SPACES (EXCLUDED from divergence detection) -->
    E2_TempRecent:   EXCLUDED (not evaluated)
    E3_TempPeriodic: EXCLUDED (not evaluated)
    E4_TempPosition: EXCLUDED (not evaluated)
  </thresholds>
  <note>Divergence threshold applies only to weighted category scores. Temporal spaces are excluded because low temporal similarity is expected and meaningful (memories SHOULD be from different times).</note>
</requirement>

<requirement id="REQ-P3-03" story_ref="US-P3-01" priority="must">
  <description>Implement ANY() similarity logic - excluding temporal spaces</description>
  <rationale>Memory is relevant if similar in ANY non-temporal space. Temporal spaces (E2-E4) are excluded because they encode WHEN, not WHAT.</rationale>
  <algorithm>
    ```
    // Non-temporal embedder indices (excludes E2=1, E3=2, E4=3)
    const NON_TEMPORAL_EMBEDDERS: [usize; 10] = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    // Maps to: E1, E5, E6, E7, E8, E9, E10, E11, E12, E13

    fn is_relevant(query: &amp;TeleologicalArray, memory: &amp;TeleologicalArray) -&gt; bool {
      for &amp;embedder in NON_TEMPORAL_EMBEDDERS.iter() {
        let sim = similarity(query.get(embedder), memory.get(embedder), embedder);
        if sim &gt; HIGH_THRESHOLDS[embedder] {
          return true;  // Similar in at least one NON-TEMPORAL space
        }
      }
      false  // Not similar in any non-temporal space
    }
    ```
  </algorithm>
  <note>Memory must be similar in at least 3 NON-TEMPORAL spaces (excludes E2, E3, E4) to be considered highly relevant.</note>
</requirement>

<requirement id="REQ-P3-04" story_ref="US-P3-02" priority="must">
  <description>Implement divergence detection against recent memories - excluding temporal spaces</description>
  <rationale>Detect activity shifts to alert user and adjust context. Temporal spaces (E2-E4) are NOT considered for divergence alerts because low temporal similarity is expected and meaningful.</rationale>
  <algorithm>
    ```
    // Embedders used for divergence detection (SEMANTIC spaces only)
    // Excludes temporal spaces E2, E3, E4
    const DIVERGENCE_EMBEDDERS: [usize; 10] = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    // Maps to: E1, E5, E6, E7, E8, E9, E10, E11, E12, E13

    fn detect_divergence(current: &amp;TeleologicalArray, recent: &amp;[Memory]) -&gt; Vec&lt;DivergenceAlert&gt; {
      let mut alerts = vec![];
      // Only check non-temporal spaces for divergence
      for &amp;embedder in DIVERGENCE_EMBEDDERS.iter() {
        let max_sim = recent.iter()
          .map(|m| similarity(current.get(embedder), m.teleological_array.get(embedder), embedder))
          .max();
        if max_sim &lt; LOW_THRESHOLDS[embedder] {
          alerts.push(DivergenceAlert {
            embedder,
            similarity: max_sim,
            threshold: LOW_THRESHOLDS[embedder],
          });
        }
      }
      alerts
    }
    ```
  </algorithm>
  <params>
    recent_window: 2 hours OR current session (whichever is smaller)
  </params>
  <exclusions>
    - E2 (TempRecent): Excluded - temporal similarity is irrelevant to content divergence
    - E3 (TempPeriodic): Excluded - periodic patterns do not indicate topic shift
    - E4 (TempPosition): Excluded - position in session is not content-based
  </exclusions>
</requirement>

<requirement id="REQ-P3-05" story_ref="US-P3-03" priority="must">
  <description>Implement multi-space relevance scoring with category-weighted scoring - temporal spaces EXCLUDED</description>
  <rationale>Rank memories by how strongly they match across multiple non-temporal spaces. Temporal spaces contribute 0.0 weight.</rationale>
  <algorithm>
    ```
    fn relevance_score(query: &amp;TeleologicalArray, memory: &amp;TeleologicalArray) -&gt; f32 {
      // Category-weighted scoring formula:
      // weighted_similarity = SUM(category_weight_i * space_similarity_i) / SUM(category_weight_i)
      //
      // Where:
      // - Semantic spaces (E1, E5-E7, E10, E12-E13): weight = 1.0
      // - Relational spaces (E8, E11): weight = 0.5
      // - Structural space (E9): weight = 0.5
      // - Temporal spaces (E2-E4): weight = 0.0 (EXCLUDED)

      // Weights by embedder index: [E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13]
      let weights = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 1.0];
      let mut weighted_sum = 0.0;
      let mut total_weight = 0.0;

      for (i, &amp;weight) in weights.iter().enumerate() {
        if weight > 0.0 {  // Skip temporal spaces with 0 weight
          let sim = similarity(query.get(i), memory.get(i), i);
          let margin = (sim - HIGH_THRESHOLDS[i]).max(0.0);
          weighted_sum += weight * margin;
          total_weight += weight;
        }
      }

      // multi_space_relevance = weighted_sum / active_space_count
      if total_weight > 0.0 {
        weighted_sum / total_weight
      } else {
        0.0
      }
    }
    ```
  </algorithm>
  <weights>
    <!-- SEMANTIC SPACES (weight = 1.0) -->
    E1_Semantic:      1.0 (primary relevance signal)
    E5_Causal:        1.0 (cause-effect relationships)
    E6_Sparse:        1.0 (keyword matching)
    E7_Code:          1.0 (code patterns)
    E10_Multimodal:   1.0 (cross-modal content)
    E12_LateInteract: 1.0 (token-level matching)
    E13_SPLADE:       1.0 (term expansion)

    <!-- RELATIONAL SPACES (weight = 0.5) -->
    E8_Emotional:     0.5 (emotional valence)
    E11_Entity:       0.5 (entity relationships)

    <!-- STRUCTURAL SPACE (weight = 0.5) -->
    E9_HDC:           0.5 (compositional structure)

    <!-- TEMPORAL SPACES (weight = 0.0 - EXCLUDED) -->
    E2_TempRecent:    0.0 (excluded from scoring)
    E3_TempPeriodic:  0.0 (excluded from scoring)
    E4_TempPosition:  0.0 (excluded from scoring)
  </weights>
  <formula>
    multi_space_relevance = weighted_sum / active_space_count
    where temporal spaces (E2-E4) are excluded (active_space_count = 10)
  </formula>
</requirement>

<requirement id="REQ-P3-06" story_ref="US-P3-02" priority="must">
  <description>Define DivergenceAlert structure</description>
  <rationale>Need structured output for divergence detection</rationale>
  <schema>
    DivergenceAlert {
      embedder: Embedder,
      embedder_name: String,
      similarity: f32,
      threshold: f32,
      recent_memory_summary: String,  // Brief summary of what recent activity was
      divergence_magnitude: f32,      // How far below threshold
    }
  </schema>
</requirement>

<requirement id="REQ-P3-07" story_ref="US-P3-01" priority="must">
  <description>Implement SimilarityConfig for threshold management with temporal exclusion</description>
  <rationale>Thresholds should be configurable and loadable from config file. Temporal spaces (E2-E4) have weight 0.0 by default.</rationale>
  <schema>
    SimilarityConfig {
      high_thresholds: [f32; 13],       // Thresholds for all 13 spaces (temporal thresholds unused)
      low_thresholds: [f32; 13],        // Divergence thresholds (temporal thresholds unused)
      weights: [f32; 13],               // Category weights (E2-E4 = 0.0)
      recent_window_hours: f32,
      excluded_embedders: [bool; 13],   // true for E2, E3, E4; false for others
    }
  </schema>
  <defaults>
    excluded_embedders: [false, true, true, true, false, false, false, false, false, false, false, false, false]
    // E1=include, E2=exclude, E3=exclude, E4=exclude, E5-E13=include
  </defaults>
  <methods>
    - load_from_file(path: &amp;Path) -&gt; Result&lt;SimilarityConfig&gt;
    - default() -&gt; SimilarityConfig  // Returns values from spec with temporal exclusion
    - is_excluded(embedder: usize) -&gt; bool  // Returns true for E2, E3, E4
    - active_embedder_count() -&gt; usize  // Returns 10 (excludes temporal)
  </methods>
</requirement>

<requirement id="REQ-P3-08" story_ref="US-P3-01" priority="must">
  <description>Implement per-embedder similarity functions</description>
  <rationale>Different embedders use different similarity metrics</rationale>
  <functions>
    - cosine_similarity(a: &amp;[f32], b: &amp;[f32]) -&gt; f32
    - jaccard_similarity(a: &amp;SparseVector, b: &amp;SparseVector) -&gt; f32
    - hamming_similarity(a: &amp;[u8], b: &amp;[u8]) -&gt; f32
    - maxsim_similarity(a: &amp;[Vec&lt;f32&gt;], b: &amp;[Vec&lt;f32&gt;]) -&gt; f32
    - transe_similarity(h: &amp;[f32], r: &amp;[f32], t: &amp;[f32]) -&gt; f32
    - similarity(a: EmbeddingRef, b: EmbeddingRef, embedder: Embedder) -&gt; f32  // Dispatcher
  </functions>
</requirement>
</requirements>

<edge_cases>
<edge_case id="EC-P3-01" req_ref="REQ-P3-03">
  <scenario>Memory has NaN or Infinity in embedding</scenario>
  <expected_behavior>Similarity returns 0.0 for that embedder with warning: "Invalid embedding value in [embedder]". Memory is NOT marked as relevant in that space.</expected_behavior>
</edge_case>

<edge_case id="EC-P3-02" req_ref="REQ-P3-04">
  <scenario>No recent memories exist (new session, empty DB)</scenario>
  <expected_behavior>No divergence alerts generated. Log: "Skipping divergence detection: no recent memories"</expected_behavior>
</edge_case>

<edge_case id="EC-P3-03" req_ref="REQ-P3-04">
  <scenario>All recent memories are from a different session</scenario>
  <expected_behavior>Use time-based window (2 hours) instead of session-based. If no memories in 2 hours, no divergence alerts.</expected_behavior>
</edge_case>

<edge_case id="EC-P3-04" req_ref="REQ-P3-05">
  <scenario>Memory matches in ALL 10 non-temporal spaces above threshold</scenario>
  <expected_behavior>Very high relevance score. This is the ideal match. Temporal spaces (E2-E4) do not contribute to score.</expected_behavior>
</edge_case>

<edge_case id="EC-P3-07" req_ref="REQ-P3-03">
  <scenario>Memory matches ONLY in temporal spaces (E2, E3, E4) but not in any non-temporal space</scenario>
  <expected_behavior>Memory is NOT relevant. Temporal similarity alone does not constitute relevance.</expected_behavior>
</edge_case>

<edge_case id="EC-P3-08" req_ref="REQ-P3-04">
  <scenario>Current query has 0.0 similarity to recent memories in E2 (TempRecent)</scenario>
  <expected_behavior>No divergence alert for E2. Temporal spaces are excluded from divergence detection.</expected_behavior>
</edge_case>

<edge_case id="EC-P3-05" req_ref="REQ-P3-08">
  <scenario>Sparse vector is completely empty (no terms)</scenario>
  <expected_behavior>Jaccard similarity = 0.0. Not an error, just no overlap.</expected_behavior>
</edge_case>

<edge_case id="EC-P3-06" req_ref="REQ-P3-08">
  <scenario>Late-interaction has different token counts</scenario>
  <expected_behavior>MaxSim handles asymmetric token counts correctly by taking max over shorter sequence.</expected_behavior>
</edge_case>
</edge_cases>

<error_states>
<error id="ERR-P3-01" http_code="500">
  <condition>Similarity computation produces NaN (division by zero)</condition>
  <message>Similarity computation failed: division by zero in [embedder]</message>
  <recovery>Return 0.0 similarity for that embedder. Log error with embedding magnitudes.</recovery>
</error>

<error id="ERR-P3-02" http_code="400">
  <condition>Config file has invalid threshold values</condition>
  <message>Invalid threshold in config: [embedder] threshold [value] must be in [0.0, 1.0]</message>
  <recovery>Use default thresholds. Log warning about config issue.</recovery>
</error>
</error_states>

<test_plan>
<test_case id="TC-P3-01" type="unit" req_ref="REQ-P3-03">
  <description>ANY() logic returns true when ONE space matches</description>
  <inputs>["Query and memory with E7 similarity=0.85, all others &lt;0.5"]</inputs>
  <expected>is_relevant returns true (0.85 &gt; 0.80 threshold for E7)</expected>
</test_case>

<test_case id="TC-P3-02" type="unit" req_ref="REQ-P3-03">
  <description>ANY() logic returns false when NO space matches</description>
  <inputs>["Query and memory with all similarities below threshold"]</inputs>
  <expected>is_relevant returns false</expected>
</test_case>

<test_case id="TC-P3-03" type="unit" req_ref="REQ-P3-04">
  <description>Divergence detected when below low threshold</description>
  <inputs>["Current with 0.15 similarity to recent in E1"]</inputs>
  <expected>DivergenceAlert for E1 with similarity=0.15, threshold=0.30</expected>
</test_case>

<test_case id="TC-P3-04" type="unit" req_ref="REQ-P3-04">
  <description>No divergence when all above low threshold</description>
  <inputs>["Current with all similarities &gt;0.35"]</inputs>
  <expected>Empty divergence alerts vector</expected>
</test_case>

<test_case id="TC-P3-05" type="unit" req_ref="REQ-P3-05">
  <description>Relevance score weighted correctly with category weights</description>
  <inputs>["Memory with E1 sim=0.95 (margin=0.20), E5 sim=0.85 (margin=0.15)"]</inputs>
  <expected>Score includes 1.0×0.20 + 1.0×0.15 = 0.35 (both semantic category, weight 1.0)</expected>
</test_case>

<test_case id="TC-P3-09" type="unit" req_ref="REQ-P3-03">
  <description>Temporal spaces excluded from similarity detection</description>
  <inputs>["Query and memory with E2 similarity=0.99, all non-temporal spaces &lt;0.5"]</inputs>
  <expected>is_relevant returns false (E2 is temporal, excluded from relevance check)</expected>
</test_case>

<test_case id="TC-P3-10" type="unit" req_ref="REQ-P3-04">
  <description>Temporal spaces excluded from divergence detection</description>
  <inputs>["Current with 0.05 similarity to recent in E2 (TempRecent)"]</inputs>
  <expected>No DivergenceAlert for E2 (temporal spaces excluded)</expected>
</test_case>

<test_case id="TC-P3-11" type="unit" req_ref="REQ-P3-05">
  <description>Temporal spaces contribute 0 weight to relevance score</description>
  <inputs>["Memory with E2 sim=0.99 (margin=0.29), E3 sim=0.99, E4 sim=0.99"]</inputs>
  <expected>Score = 0.0 from temporal spaces (weight=0.0 for E2, E3, E4)</expected>
</test_case>

<test_case id="TC-P3-12" type="unit" req_ref="REQ-P3-05">
  <description>Category-weighted scoring uses correct category weights</description>
  <inputs>["E8 (Emotional) sim=0.90 (margin=0.20), E9 (HDC) sim=0.90 (margin=0.20)"]</inputs>
  <expected>Score includes 0.5×0.20 + 0.5×0.20 = 0.20 (relational/structural categories, weight 0.5)</expected>
</test_case>

<test_case id="TC-P3-06" type="unit" req_ref="REQ-P3-08">
  <description>Cosine similarity computed correctly</description>
  <inputs>["[1,0,0] and [1,0,0]"]</inputs>
  <expected>Similarity = 1.0</expected>
</test_case>

<test_case id="TC-P3-07" type="unit" req_ref="REQ-P3-08">
  <description>Jaccard similarity computed correctly</description>
  <inputs>["SparseVector {a:1, b:1} and SparseVector {a:1, c:1}"]</inputs>
  <expected>Similarity = 1/3 = 0.333 (intersection=1, union=3)</expected>
</test_case>

<test_case id="TC-P3-08" type="integration" req_ref="REQ-P3-04">
  <description>Recent window respects 2-hour limit</description>
  <inputs>["Memories from 3 hours ago, 1 hour ago"]</inputs>
  <expected>Only 1-hour-ago memory used for divergence detection</expected>
</test_case>
</test_plan>

<validation_criteria>
  <criterion>Per-space high thresholds configured as specified for 10 non-temporal spaces</criterion>
  <criterion>Per-space low thresholds configured as specified for 10 non-temporal spaces</criterion>
  <criterion>ANY() logic correctly identifies relevant memories using ONLY non-temporal spaces</criterion>
  <criterion>Divergence detection finds low-similarity recent memories in SEMANTIC spaces only</criterion>
  <criterion>Relevance score uses correct category weights (semantic=1.0, relational=0.5, structural=0.5, temporal=0.0)</criterion>
  <criterion>All similarity functions implemented (cosine, jaccard, hamming, maxsim, transe)</criterion>
  <criterion>Recent window respects 2-hour OR session limit</criterion>
  <criterion>Temporal spaces (E2-E4) are EXCLUDED from all similarity and divergence calculations</criterion>
  <criterion>Multi-space relevance computed as weighted_sum / active_space_count (10 spaces)</criterion>
</validation_criteria>
</functional_spec>
```

## Threshold Summary Tables

### High Similarity Thresholds (Memory is Relevant) - NON-TEMPORAL SPACES ONLY

| Embedder | Category | Threshold | Rationale |
|----------|----------|-----------|-----------|
| E1 Semantic | Semantic | 0.75 | Primary relevance signal |
| E5 Causal | Semantic | 0.70 | Cause-effect chains need flexibility |
| E6 Sparse | Semantic | 0.60 | Keyword overlap naturally lower |
| E7 Code | Semantic | 0.80 | Code similarity should be high |
| E10 Multimodal | Semantic | 0.70 | Cross-modal content matching |
| E12 LateInteract | Semantic | 0.70 | Token-level matching |
| E13 SPLADE | Semantic | 0.60 | Term expansion overlap naturally lower |
| E8 Emotional | Relational | 0.70 | Emotional valence matching |
| E11 Entity | Relational | 0.70 | Entity relationship matching |
| E9 HDC | Structural | 0.70 | Compositional structure matching |
| **E2 TempRecent** | **EXCLUDED** | N/A | Temporal - not used for similarity |
| **E3 TempPeriodic** | **EXCLUDED** | N/A | Temporal - not used for similarity |
| **E4 TempPosition** | **EXCLUDED** | N/A | Temporal - not used for similarity |

### Low Similarity Thresholds (Divergence Detected) - NON-TEMPORAL SPACES ONLY

| Embedder | Category | Threshold | Rationale |
|----------|----------|-----------|-----------|
| E1 Semantic | Semantic | 0.30 | Below 30% semantic overlap = different topic |
| E5 Causal | Semantic | 0.25 | Causal chains more diverse |
| E6 Sparse | Semantic | 0.20 | Keywords can vary significantly |
| E7 Code | Semantic | 0.35 | Code patterns more distinct |
| E10 Multimodal | Semantic | 0.30 | Cross-modal divergence threshold |
| E12 LateInteract | Semantic | 0.30 | Token-level divergence threshold |
| E13 SPLADE | Semantic | 0.20 | Term expansion varies |
| E8 Emotional | Relational | 0.30 | Emotional divergence threshold |
| E11 Entity | Relational | 0.30 | Entity divergence threshold |
| E9 HDC | Structural | 0.30 | Structural divergence threshold |
| **E2 TempRecent** | **EXCLUDED** | N/A | Temporal - not used for divergence |
| **E3 TempPeriodic** | **EXCLUDED** | N/A | Temporal - not used for divergence |
| **E4 TempPosition** | **EXCLUDED** | N/A | Temporal - not used for divergence |

### Category-Weighted Scoring

```
weighted_similarity = SUM(category_weight_i * space_similarity_i) / SUM(category_weight_i)

Where:
- Semantic spaces (E1, E5-E7, E10, E12-E13): weight = 1.0
- Relational spaces (E8, E11): weight = 0.5
- Structural space (E9): weight = 0.5
- Temporal spaces (E2-E4): weight = 0.0 (EXCLUDED)

Total active weight = 7*1.0 + 2*0.5 + 1*0.5 = 8.5
Active space count = 10 (excludes E2, E3, E4)
```

| Embedder | Category | Weight | Rationale |
|----------|----------|--------|-----------|
| E1 Semantic | Semantic | 1.0 | Primary relevance signal |
| E5 Causal | Semantic | 1.0 | Cause-effect relationships |
| E6 Sparse | Semantic | 1.0 | Keyword matching |
| E7 Code | Semantic | 1.0 | Code patterns |
| E10 Multimodal | Semantic | 1.0 | Cross-modal content |
| E12 LateInteract | Semantic | 1.0 | Token-level matching |
| E13 SPLADE | Semantic | 1.0 | Term expansion |
| E8 Emotional | Relational | 0.5 | Emotional valence |
| E11 Entity | Relational | 0.5 | Entity relationships |
| E9 HDC | Structural | 0.5 | Compositional structure |
| **E2 TempRecent** | **Temporal** | **0.0** | **EXCLUDED** |
| **E3 TempPeriodic** | **Temporal** | **0.0** | **EXCLUDED** |
| **E4 TempPosition** | **Temporal** | **0.0** | **EXCLUDED** |

## Temporal Exclusion Rationale

Temporal embedders (E2, E3, E4) are excluded from similarity and divergence detection because:

1. **They encode WHEN, not WHAT**: Temporal embeddings capture when a memory was created, not its content
2. **Similar times != similar content**: Two memories from the same time period may be completely unrelated
3. **Expected low similarity**: Memories from different times SHOULD have low temporal similarity - this is not divergence
4. **Content-based retrieval focus**: Similarity detection aims to find semantically related content, not temporally co-located content

Temporal embeddings remain useful for:
- Time-based filtering (e.g., "memories from last week")
- Recency boosting in ranking (applied separately from similarity)
- Session-based grouping

## Divergence Alert Format

```
DIVERGENCE DETECTED
Recent activity in Semantic space: "Discussing authentication implementation"
Current appears different - similarity: 0.23 (threshold: 0.30)
This may indicate a context switch.

Note: Divergence is only detected in semantic spaces (E1, E5-E7, E10, E12-E13).
Temporal spaces (E2-E4) are not considered for divergence alerts.
```
