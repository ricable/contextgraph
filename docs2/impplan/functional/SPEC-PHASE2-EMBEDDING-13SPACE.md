# Functional Specification: Phase 2 - 13-Space Embedding System

```xml
<functional_spec id="SPEC-PHASE2" version="1.0">
<metadata>
  <title>13-Space Embedding System</title>
  <status>approved</status>
  <owner>Context Graph Team</owner>
  <created>2026-01-16</created>
  <last_updated>2026-01-16</last_updated>
  <implements>impplan.md Part 2</implements>
  <depends_on>
    <spec_ref>SPEC-PHASE0</spec_ref>
    <spec_ref>SPEC-PHASE1</spec_ref>
  </depends_on>
  <related_specs>
    <spec_ref>SPEC-PHASE3</spec_ref>
    <spec_ref>SPEC-PHASE4</spec_ref>
  </related_specs>
</metadata>

<overview>
Ensure all Memory objects are embedded with all 13 embedders and stored with the complete TeleologicalArray. This phase verifies the existing 13-embedder infrastructure is properly integrated with the new Memory capture system from Phase 1.

The existing codebase already has the 13 embedders defined (E1-E13). This phase focuses on:
1. Ensuring Memory.teleological_array is populated for ALL memories
2. Configuring per-embedder semantics and distance metrics
3. Verifying atomic storage (all 13 or nothing)

**Problem Solved**: The existing MemoryNode uses a single 1536D embedding. The new system requires all 13 embeddings for every memory to enable multi-space retrieval and divergence detection.

**Who Benefits**: The retrieval system which can now find memories similar in ANY embedding space; the clustering system which can form clusters per space.
</overview>

<user_stories>
<story id="US-P2-01" priority="must-have">
  <narrative>
    As a memory capture system
    I want to generate all 13 embeddings atomically for each memory
    So that no memory exists with partial embedding data
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P2-01-01">
      <given>Content to be captured as memory</given>
      <when>Memory capture completes</when>
      <then>Memory.teleological_array contains non-empty vectors for all 13 embedders</then>
    </criterion>
    <criterion id="AC-P2-01-02">
      <given>Embedding generation fails for any of the 13 embedders</given>
      <when>Memory capture is attempted</when>
      <then>Memory is NOT stored and error is returned</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P2-02" priority="must-have">
  <narrative>
    As a retrieval system
    I want each embedder to have defined semantics and distance metrics
    So that I can perform appropriate comparisons in each space
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P2-02-01">
      <given>The embedder configuration</given>
      <when>Querying for embedder E6 (Sparse/SPLADE)</when>
      <then>Distance metric is Jaccard, not Cosine</then>
    </criterion>
    <criterion id="AC-P2-02-02">
      <given>The embedder configuration</given>
      <when>Querying for embedder E9 (HDC)</when>
      <then>Distance metric is Hamming</then>
    </criterion>
    <criterion id="AC-P2-02-03">
      <given>The embedder configuration</given>
      <when>Querying for embedder E12 (Late-Interaction)</when>
      <then>Distance metric is MaxSim (max similarity across tokens)</then>
    </criterion>
  </acceptance_criteria>
</story>

<story id="US-P2-03" priority="must-have">
  <narrative>
    As a storage system
    I want to store all 13 embeddings efficiently
    So that memory usage is reasonable (~17KB quantized per memory)
  </narrative>
  <acceptance_criteria>
    <criterion id="AC-P2-03-01">
      <given>A memory with all 13 embeddings</given>
      <when>Serialized for storage</when>
      <then>Total size is &lt;20KB when quantized</then>
    </criterion>
  </acceptance_criteria>
</story>
</user_stories>

<requirements>
<requirement id="REQ-P2-01" story_ref="US-P2-01" priority="must">
  <description>Implement atomic 13-embedding generation via MultiArrayProvider</description>
  <rationale>Partial embeddings would break multi-space retrieval</rationale>
  <behavior>
    1. Call MultiArrayProvider.embed_all(content)
    2. If ANY embedder fails, return Err immediately
    3. On success, return TeleologicalArray with all 13 embeddings
    4. Never store partial results
  </behavior>
</requirement>

<requirement id="REQ-P2-02" story_ref="US-P2-02" priority="must">
  <description>Define per-embedder distance metrics in configuration</description>
  <rationale>Different embedding spaces require different similarity measures</rationale>
  <config>
    E1_Semantic:     { dim: 1024D, distance: Cosine }
    E2_TempRecent:   { dim: 512D,  distance: Cosine }
    E3_TempPeriodic: { dim: 512D,  distance: Cosine }
    E4_TempPosition: { dim: 512D,  distance: Cosine }
    E5_Causal:       { dim: 768D,  distance: Cosine, asymmetric: true }
    E6_Sparse:       { dim: ~30K,  distance: Jaccard }
    E7_Code:         { dim: 1536D, distance: Cosine }
    E8_Emotional:    { dim: 384D,  distance: Cosine }
    E9_HDC:          { dim: 1024D, distance: Hamming }
    E10_Multimodal:  { dim: 768D,  distance: Cosine }
    E11_Entity:      { dim: 384D,  distance: TransE }
    E12_LateInteract:{ dim: 128D/tok, distance: MaxSim }
    E13_SPLADE:      { dim: ~30K,  distance: Jaccard }
  </config>
</requirement>

<requirement id="REQ-P2-03" story_ref="US-P2-03" priority="must">
  <description>Implement quantization for efficient storage</description>
  <rationale>Full float32 storage would use ~100KB per memory</rationale>
  <quantization>
    Dense vectors: PQ-8 (product quantization) for E1, E5, E7, E10
    512D vectors: Float8 for E2, E3, E4
    384D vectors: Float8 for E8, E11
    HDC: Binary (1024 bits = 128 bytes)
    Sparse: Inverted index (only store non-zero terms)
  </quantization>
</requirement>

<requirement id="REQ-P2-04" story_ref="US-P2-01" priority="must">
  <description>Validate TeleologicalArray dimensions on storage</description>
  <rationale>Catch dimension mismatches before they corrupt the database</rationale>
  <behavior>
    1. Before storing, validate each embedding has correct dimension
    2. E1: exactly 1024, E7: exactly 1536, etc.
    3. If validation fails, return Err with details
  </behavior>
</requirement>

<requirement id="REQ-P2-05" story_ref="US-P2-02" priority="must">
  <description>Implement EmbedderConfig registry for runtime lookup</description>
  <rationale>Need to look up embedder properties when computing similarity</rationale>
  <methods>
    - get_distance_metric(embedder: Embedder) -> DistanceMetric
    - get_dimension(embedder: Embedder) -> usize
    - get_quantization(embedder: Embedder) -> QuantizationConfig
    - is_asymmetric(embedder: Embedder) -> bool
    - is_sparse(embedder: Embedder) -> bool
  </methods>
</requirement>
</requirements>

<edge_cases>
<edge_case id="EC-P2-01" req_ref="REQ-P2-01">
  <scenario>One embedder model is unavailable (e.g., GPU OOM)</scenario>
  <expected_behavior>Entire embedding generation fails with clear error: "Embedder E7 (Code) failed: CUDA out of memory". Memory is NOT stored.</expected_behavior>
</edge_case>

<edge_case id="EC-P2-02" req_ref="REQ-P2-01">
  <scenario>Content is empty string</scenario>
  <expected_behavior>All embedders produce zero vectors. Warning logged: "Embedding empty content produces zero vectors". Memory CAN be stored (valid state).</expected_behavior>
</edge_case>

<edge_case id="EC-P2-03" req_ref="REQ-P2-04">
  <scenario>Embedder returns wrong dimension (bug)</scenario>
  <expected_behavior>Validation fails: "E1_Semantic returned 512D, expected 1024D". Memory NOT stored. This indicates embedder bug requiring investigation.</expected_behavior>
</edge_case>

<edge_case id="EC-P2-04" req_ref="REQ-P2-02">
  <scenario>Asymmetric distance requested for symmetric embedder</scenario>
  <expected_behavior>Symmetric distance used with warning: "E1_Semantic is symmetric; ignoring asymmetric flag"</expected_behavior>
</edge_case>
</edge_cases>

<error_states>
<error id="ERR-P2-01" http_code="500">
  <condition>Embedder model fails to load</condition>
  <message>Failed to initialize embedder [name]: [error]</message>
  <recovery>System cannot start. Fix model path or GPU allocation.</recovery>
</error>

<error id="ERR-P2-02" http_code="500">
  <condition>Embedding generation timeout (&gt;500ms for batch)</condition>
  <message>Embedding generation timed out after 500ms</message>
  <recovery>Retry with smaller batch. Check GPU utilization.</recovery>
</error>

<error id="ERR-P2-03" http_code="500">
  <condition>Quantization produces invalid output</condition>
  <message>Quantization failed for [embedder]: [error]</message>
  <recovery>Store unquantized temporarily. Investigate quantization bug.</recovery>
</error>
</error_states>

<test_plan>
<test_case id="TC-P2-01" type="unit" req_ref="REQ-P2-01">
  <description>All 13 embeddings generated for sample content</description>
  <inputs>["Hello world, this is a test."]</inputs>
  <expected>TeleologicalArray with 13 non-empty embeddings</expected>
</test_case>

<test_case id="TC-P2-02" type="unit" req_ref="REQ-P2-04">
  <description>Dimension validation catches wrong size</description>
  <inputs>["TeleologicalArray with E1 = 512D (wrong)"]</inputs>
  <expected>Validation error returned</expected>
</test_case>

<test_case id="TC-P2-03" type="unit" req_ref="REQ-P2-02">
  <description>Distance metrics correctly assigned</description>
  <inputs>["Query embedder config for E6, E9, E12"]</inputs>
  <expected>E6=Jaccard, E9=Hamming, E12=MaxSim</expected>
</test_case>

<test_case id="TC-P2-04" type="integration" req_ref="REQ-P2-03">
  <description>Quantized storage size is &lt;20KB</description>
  <inputs>["Store memory with all 13 embeddings"]</inputs>
  <expected>Serialized size &lt; 20480 bytes</expected>
</test_case>

<test_case id="TC-P2-05" type="unit" req_ref="REQ-P2-01">
  <description>Partial embedding failure prevents storage</description>
  <inputs>["Mock E7 to fail during embedding"]</inputs>
  <expected>Error returned, no memory stored</expected>
</test_case>

<test_case id="TC-P2-06" type="integration" req_ref="REQ-P2-05">
  <description>EmbedderConfig registry returns correct values</description>
  <inputs>["Query all 13 embedders"]</inputs>
  <expected>All return valid config matching spec</expected>
</test_case>
</test_plan>

<validation_criteria>
  <criterion>All captured memories have all 13 embeddings populated</criterion>
  <criterion>No partial embeddings exist in storage</criterion>
  <criterion>Each embedder has defined distance metric</criterion>
  <criterion>Quantized memory size &lt;20KB</criterion>
  <criterion>Dimension validation catches mismatches</criterion>
  <criterion>EmbedderConfig registry provides runtime lookup</criterion>
</validation_criteria>
</functional_spec>
```

## 13-Embedder Summary Table

| ID | Name | Category | Dimension | Distance | Quantization | Purpose |
|----|------|----------|-----------|----------|--------------|---------|
| E1 | Semantic | Semantic | 1024D | Cosine | PQ-8 | Conceptual meaning |
| E2 | TemporalRecent | Temporal | 512D | Cosine | Float8 | Recency patterns |
| E3 | TemporalPeriodic | Temporal | 512D | Cosine | Float8 | Cyclical patterns |
| E4 | TemporalPositional | Temporal | 512D | Cosine | Float8 | Sequence position |
| E5 | Causal | Semantic | 768D | Cosine (asymmetric) | PQ-8 | Cause-effect |
| E6 | Sparse | Semantic | ~30K | Jaccard | Inverted | Keywords |
| E7 | Code | Semantic | 1536D | Cosine | PQ-8 | Code patterns |
| E8 | Emotional | Relational | 384D | Cosine | Float8 | Sentiment |
| E9 | HDC | Structural | 1024D | Hamming | Binary | Structure |
| E10 | Multimodal | Semantic | 768D | Cosine | PQ-8 | Intent |
| E11 | Entity | Relational | 384D | TransE | Float8 | Entity relations |
| E12 | LateInteraction | Semantic | 128D/token | MaxSim | Dense | Precision |
| E13 | SPLADE | Semantic | ~30K | Jaccard | Inverted | Term expansion |

## Embedder Categories

Each embedder is classified into a category that determines its role in topic detection:

| Category | Embedders | Topic Weight | Role |
|----------|-----------|--------------|------|
| Semantic | E1, E5, E6, E7, E10, E12, E13 | 1.0x | Primary topic triggers - capture WHAT |
| Temporal | E2, E3, E4 | 0.0x | Metadata only - capture WHEN, excluded from topic detection |
| Relational | E8, E11 | 0.5x | Supporting evidence - capture WHO/WHAT relationships |
| Structural | E9 | 0.5x | Supporting evidence - captures code structure patterns |

### Key Rules

1. **Temporal Exclusion**: E2 (Hour-of-Day), E3 (Day-of-Week), E4 (Recency) are NEVER used for:
   - Topic detection (weighted_agreement calculation)
   - Divergence detection
   - Multi-space relevance scoring

2. **Temporal embedders ARE used for**:
   - Recency weighting in injection priority
   - Session correlation (same time patterns -> likely same session)
   - Timeline enrichment in context display

3. **Weighted Agreement**: Topic synthesis uses `weighted_agreement = sum(topic_weight_i * is_clustered_i)` where threshold >= 2.5

## Storage Size Breakdown (Quantized)

| Embedder | Raw Size | Quantized Size |
|----------|----------|----------------|
| E1 (1024D) | 4096B | ~1024B (PQ-8) |
| E2-E4 (3×512D) | 6144B | ~1536B (Float8) |
| E5 (768D) | 3072B | ~768B (PQ-8) |
| E6 (~30K sparse) | ~1KB avg | ~1KB (inverted) |
| E7 (1536D) | 6144B | ~1536B (PQ-8) |
| E8 (384D) | 1536B | ~384B (Float8) |
| E9 (1024D binary) | 4096B | 128B |
| E10 (768D) | 3072B | ~768B (PQ-8) |
| E11 (384D) | 1536B | ~384B (Float8) |
| E12 (~10 tokens) | ~5KB | ~2.5KB |
| E13 (~30K sparse) | ~1KB avg | ~1KB (inverted) |
| **Total** | ~36KB | **~11KB** |
