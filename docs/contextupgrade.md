# Contextual Embedder Integration Analysis

**Version:** 1.0
**Date:** 2026-01-22
**Status:** Analysis Complete
**Author:** Context Graph Development Team

---

## Executive Summary

This document analyzes the `all-mpnet-base-v2` model located at `/home/cabdru/contextgraph/models/contextual` and determines its optimal integration into the 13-embedder context-graph system. The analysis draws on patterns established by recent E5 (Causal), E6 (Sparse), and E8 (Graph) embedder upgrades.

**Key Finding:** The contextual model (`all-mpnet-base-v2`, 768D) is a powerful general-purpose sentence encoder trained on 1.17 billion pairs. However, the 13-embedder system is already complete. The optimal integration strategy is to **leverage this model to enhance E10 (V_multimodality)** or explore its use as a **context-aware re-ranker** that complements E1's semantic foundation.

---

## Part 1: Model Analysis

### Model Specifications

| Property | Value | Source |
|----------|-------|--------|
| **Model Name** | `sentence-transformers/all-mpnet-base-v2` | README.md |
| **Base Architecture** | Microsoft MPNet | config.json |
| **Output Dimension** | 768D | config.json `hidden_size` |
| **Max Sequence Length** | 384 tokens | README.md |
| **Training Data** | 1.17 billion sentence pairs | README.md |
| **Training Objective** | Contrastive learning | README.md |
| **Vocabulary Size** | 30,527 | config.json |
| **Attention Heads** | 12 | config.json |
| **Hidden Layers** | 12 | config.json |
| **Pooling** | Mean pooling | modules.json |

### Training Data Sources

The model was trained on a diverse mix of datasets, making it exceptionally well-suited for understanding contextual relationships:

| Dataset | Pairs | Relevance to "Context" |
|---------|-------|------------------------|
| Reddit comments | 726M | Conversational context |
| S2ORC Citations | 116M | Academic document context |
| WikiAnswers | 77M | Question-answer context |
| PAQ (Question, Answer) | 64M | Information retrieval context |
| Stack Exchange | 68M | Technical Q&A context |
| MS MARCO triplets | 9M | Search relevance context |
| Natural Questions | 100K | Factual context |
| Code Search | 1.1M | Code-text context |

**Key Insight:** This training makes the model excellent at understanding **how pieces of text relate to each other** - the "context" of information. This is distinct from:
- E1 (V_meaning): What text **means** semantically
- E5 (V_causality): What text **causes or is caused by**
- E8 (V_connectivity): What text **connects to structurally**

### Model Files Available

```
models/contextual/
├── config.json                    # Model configuration
├── tokenizer.json                 # HuggingFace tokenizer
├── model.safetensors              # PyTorch weights
├── pytorch_model.bin              # Alternative weights
├── onnx/
│   ├── model.onnx                 # ONNX export
│   ├── model_O2.onnx              # Optimized O2
│   ├── model_O3.onnx              # Optimized O3
│   ├── model_O4.onnx              # Optimized O4
│   ├── model_qint8_*.onnx         # Quantized variants
│   └── model_quint8_avx2.onnx     # AVX2 optimized
└── openvino/
    ├── openvino_model.xml         # OpenVINO model
    └── openvino_model.bin         # OpenVINO weights
```

The availability of ONNX and OpenVINO exports indicates the model is production-ready with multiple inference options.

---

## Part 2: Current 13-Embedder System Analysis

### Complete Embedder Registry

| Slot | Name | Dimension | Purpose | Category | Topic Weight |
|------|------|-----------|---------|----------|--------------|
| **E1** | V_meaning | 1024D | Primary semantic | FOUNDATION | 1.0 |
| **E2** | V_freshness | 512D | Recency (temporal) | TEMPORAL | 0.0 |
| **E3** | V_periodicity | 512D | Cyclical patterns | TEMPORAL | 0.0 |
| **E4** | V_ordering | 512D | Sequence position | TEMPORAL | 0.0 |
| **E5** | V_causality | 768D | Causal reasoning | SEMANTIC_ENHANCER | 1.0 |
| **E6** | V_selectivity | ~30K sparse | Keyword precision | SEMANTIC_ENHANCER | 1.0 |
| **E7** | V_correctness | 1536D | Code/technical | SEMANTIC_ENHANCER | 1.0 |
| **E8** | V_connectivity | 384D | Graph structure | RELATIONAL | 0.5 |
| **E9** | V_robustness | 1024D binary | Hyperdimensional | STRUCTURAL | 0.5 |
| **E10** | V_multimodality | 768D | Cross-modal intent | SEMANTIC_ENHANCER | 1.0 |
| **E11** | V_factuality | 384D | Named entities | RELATIONAL | 0.5 |
| **E12** | V_precision | 128D/token | Late interaction | SEMANTIC_ENHANCER | 1.0 |
| **E13** | V_keyword_precision | ~30K sparse | SPLADE expansion | SEMANTIC_ENHANCER | 1.0 |

### Dimension Analysis

The contextual model (768D) matches:
- **E5 (V_causality)**: 768D - Causal reasoning
- **E10 (V_multimodality)**: 768D - Cross-modal intent

This suggests natural integration paths with either E5 or E10.

### Constitutional Constraints

From `constitution.yaml`, key constraints affect integration options:

```yaml
arch_rules:
  ARCH-01: "TeleologicalArray is atomic - store all 13 embeddings or nothing"
  ARCH-05: "All 13 embedders required - missing embedder is fatal error"

embeddings:
  paradigm: "FINGERPRINT - E1 foundation + 12 enhancement layers; ~17KB quantized"
```

**Critical Constraint:** The system is designed for exactly 13 embedders. Adding a 14th would require:
- Changes to `SemanticFingerprint` struct
- Updates to `NUM_EMBEDDERS` constant
- HNSW index structure changes
- Storage schema migration
- All weight profiles updated

---

## Part 3: Recent Embedder Upgrade Patterns

### Pattern Analysis: E5 Causal Upgrade (1feff8f)

**Key Innovations:**
1. **Dual Projections**: Separate W_cause and W_effect matrices (perturbed identities)
2. **Marker Detection**: 94 cause/effect linguistic markers
3. **Asymmetric Similarity**: 1.2x forward (cause→effect), 0.8x backward
4. **MCP Tools**: `search_causes`, `get_causal_chain`
5. **Weight Profile**: `causal_reasoning` with 45% E5 weight

**Code Pattern:**
```rust
// CausalModel.embed_dual() returns genuinely different vectors
pub async fn embed_dual(&self, content: &str) -> Result<(Vec<f32>, Vec<f32>)> {
    // Single encoder pass
    let base = self.encode(content)?;
    // Dual projection
    let cause_vec = self.projection.project_cause(&base);
    let effect_vec = self.projection.project_effect(&base);
    Ok((cause_vec, effect_vec))
}
```

### Pattern Analysis: E6 Sparse Upgrade (398dd54)

**Key Innovations:**
1. **Dual Storage**: Both sparse vector AND projected dense
2. **Inverted Index**: RocksDB column family for term→memory mapping
3. **Query-Aware Boosting**: 0.5x-2.0x based on technical term detection
4. **E6 Tie-Breaker**: Post-fusion disambiguation using term overlap
5. **Stage 1 Co-Pilot**: E6 recall alongside E13 SPLADE

**Code Pattern:**
```rust
// E6 stored in two formats
pub struct E6DualEmbedding {
    pub sparse: SparseVector,        // Original ~235 active terms
    pub dense_projected: Vec<f32>,   // 1536D for fusion
}
```

### Pattern Analysis: E8 Graph Upgrade (9d968bc)

**Key Innovations:**
1. **Dual Projections**: W_source and W_target (perturbed identities)
2. **Structural Markers**: ~60 source/target indicator patterns
3. **Asymmetric Similarity**: 1.2x forward (source→target), 0.8x backward
4. **MCP Tools**: `get_neighbors`, `traverse_path`, `find_hubs`
5. **Weight Profile**: `graph_reasoning` with 45% E8 weight

**Code Pattern:**
```rust
// GraphModel.embed_dual() returns source/target vectors
pub async fn embed_dual(&self, content: &str) -> Result<(Vec<f32>, Vec<f32>)> {
    let base = self.encode(content)?;
    let source_vec = self.projection.project_source(&base);
    let target_vec = self.projection.project_target(&base);
    Ok((source_vec, target_vec))
}
```

### Common Upgrade Pattern

```
1. MARKER DETECTION
   - Identify linguistic patterns that signal the embedder's specialty
   - E5: "because", "therefore", "causes", "leads to"
   - E6: Technical terms, API paths, acronyms
   - E8: "imports", "uses", "depends on", "extends"

2. ASYMMETRIC PROJECTIONS (if directional)
   - Create W_A and W_B as perturbed identity matrices
   - Apply after single encoder pass
   - Produces genuinely different vectors for different roles

3. MCP TOOLS
   - Create specialized search/traversal tools
   - Use the embedder's unique perspective
   - Integrate with weight profiles

4. BENCHMARK SUITE
   - Stress tests with known answers
   - Real-data benchmarks (Wikipedia, arXiv)
   - Metrics: MRR, direction accuracy, recall@K

5. WEIGHT PROFILE
   - Define weight distribution for query types
   - E5 at 45% for "why" queries
   - E8 at 45% for "connected to" queries
```

---

## Part 4: Integration Strategy Analysis

### Option 1: Replace E10 (V_multimodality)

**Rationale:** E10 currently uses CLIP (openai/clip-vit-large-patch14) for text-image pairs, but the system is text-only. The 768D dimension matches exactly.

**Pros:**
- No architectural changes needed
- Same dimension (768D)
- `all-mpnet-base-v2` is arguably better for pure text

**Cons:**
- Loses image embedding capability (future-proofing)
- E10's "cross-modal intent" purpose differs from "contextual relationships"
- Would need to update all E10 references

**Verdict:** NOT RECOMMENDED - E10 serves a distinct purpose

### Option 2: Add E14 (New Slot)

**Rationale:** Add a 14th embedder specifically for contextual relationships.

**Pros:**
- Clean separation of concerns
- Full control over purpose and weight

**Cons:**
- Requires changing `NUM_EMBEDDERS = 13` (constitutional violation)
- Storage schema changes (all fingerprints grow)
- HNSW index restructuring
- All weight profiles need updating
- Constitution explicitly defines 13-embedder paradigm

**Verdict:** NOT RECOMMENDED - Violates constitutional constraints

### Option 3: Context-Aware Re-Ranker (E12-Style)

**Rationale:** Use contextual model as a Stage 3.5 re-ranker that scores context alignment.

**Pros:**
- No fingerprint storage changes
- Applied only during retrieval
- Complements E12 ColBERT re-ranking

**Cons:**
- Adds latency to retrieval pipeline
- Not part of fingerprint (can't cluster on it)
- Limited benefit if E1 already captures semantics

**Verdict:** POSSIBLE - Low-impact enhancement

### Option 4: Enhanced E10 with Context Window (RECOMMENDED)

**Rationale:** Upgrade E10 to include contextual understanding while preserving multimodal capability.

**Implementation:**
```rust
// E10 produces TWO vectors: semantic intent + contextual embedding
pub async fn embed_with_context(&self, content: &str, context: Option<&str>)
    -> Result<(Vec<f32>, Vec<f32>)> {
    // Standard CLIP text embedding for intent
    let intent_vec = self.clip_text_encode(content)?;

    // Contextual embedding using all-mpnet-base-v2
    let context_vec = if let Some(ctx) = context {
        // Embed content with surrounding context
        let contextualized = format!("{} [SEP] {}", ctx, content);
        self.contextual_model.embed(&contextualized)?
    } else {
        self.contextual_model.embed(content)?
    };

    Ok((intent_vec, context_vec))
}
```

**Fingerprint Extension:**
```rust
pub struct SemanticFingerprint {
    // ... existing fields ...

    // E10 upgrade: dual vectors for intent + context
    pub e10_multimodal_intent: Vec<f32>,    // 768D CLIP (existing)
    pub e10_multimodal_context: Vec<f32>,   // 768D MPNet (NEW)
    pub e10_multimodal: Vec<f32>,           // DEPRECATED - backward compat
}
```

**Pros:**
- Follows E5/E8 dual embedding pattern
- No new embedder slot needed
- Contextual model used for its strength
- E10 gains richer representation

**Cons:**
- E10 becomes more complex
- Requires updating E10 loading logic
- Storage grows slightly (~6KB per fingerprint)

**Verdict:** RECOMMENDED - Best fit within existing architecture

### Option 5: Context-Enhanced E4 (V_ordering)

**Rationale:** E4 captures sequence position but lacks semantic understanding of context flow. The contextual model could enhance E4's ability to understand narrative/topical continuity.

**Implementation:**
```rust
// E4 produces: positional encoding + context flow embedding
pub async fn embed_with_flow(&self, content: &str, prev_content: Option<&str>)
    -> Result<(Vec<f32>, Vec<f32>)> {
    // Standard positional encoding
    let position_vec = self.compute_positional_encoding(sequence_num)?;

    // Context flow using all-mpnet-base-v2
    let flow_vec = if let Some(prev) = prev_content {
        let flow_text = format!("{} [THEN] {}", prev, content);
        self.contextual_model.embed(&flow_text)?
    } else {
        self.contextual_model.embed(content)?
    };

    Ok((position_vec, flow_vec))
}
```

**Pros:**
- E4 gains semantic context awareness
- Complements pure positional encoding
- Natural fit for "before/after" relationships

**Cons:**
- E4 is TEMPORAL category (0.0 topic weight)
- May conflate temporal and semantic concerns
- E4 is already specialized for sequence

**Verdict:** POSSIBLE - Interesting but changes E4's focused purpose

---

## Part 5: Recommended Integration Plan

Based on the analysis, **Option 4 (Enhanced E10)** is recommended. Here's the detailed implementation plan:

### Phase 1: Contextual Model Integration (Week 1)

#### 1.1 Add ContextualModel to Embeddings Crate

```rust
// NEW: crates/context-graph-embeddings/src/models/pretrained/contextual/mod.rs

pub mod constants;
pub mod model;
pub mod forward;

pub use model::ContextualModel;
```

```rust
// NEW: crates/context-graph-embeddings/src/models/pretrained/contextual/constants.rs

pub const CONTEXTUAL_DIMENSION: usize = 768;
pub const CONTEXTUAL_MAX_TOKENS: usize = 384;
pub const CONTEXTUAL_MODEL_NAME: &str = "all-mpnet-base-v2";
pub const CONTEXTUAL_LATENCY_BUDGET_MS: u64 = 5;
```

```rust
// NEW: crates/context-graph-embeddings/src/models/pretrained/contextual/model.rs

pub struct ContextualModel {
    model_state: RwLock<ModelState>,
    model_path: PathBuf,
    config: SingleModelConfig,
    loaded: AtomicBool,
}

impl ContextualModel {
    pub fn new(model_path: &Path, config: SingleModelConfig) -> EmbeddingResult<Self>;
    pub async fn load(&self) -> EmbeddingResult<()>;
    pub async fn embed(&self, content: &str) -> EmbeddingResult<Vec<f32>>;

    /// Embed content with surrounding context for richer representation
    pub async fn embed_with_context(
        &self,
        content: &str,
        surrounding_context: Option<&str>
    ) -> EmbeddingResult<Vec<f32>>;
}
```

#### 1.2 Update MultimodalModel for Dual Embedding

```rust
// UPDATE: crates/context-graph-embeddings/src/models/pretrained/multimodal/model.rs

pub struct MultimodalModel {
    // ... existing CLIP fields ...

    /// Contextual model for context-aware embeddings
    contextual_model: ContextualModel,
}

impl MultimodalModel {
    /// Embed with both intent (CLIP) and context (MPNet) vectors
    pub async fn embed_dual(&self, content: &str) -> EmbeddingResult<(Vec<f32>, Vec<f32>)> {
        // Intent via CLIP text encoder
        let intent_vec = self.embed_text_internal(content).await?;

        // Context via MPNet
        let context_vec = self.contextual_model.embed(content).await?;

        Ok((intent_vec, context_vec))
    }

    /// Embed with explicit surrounding context
    pub async fn embed_with_context(
        &self,
        content: &str,
        surrounding: Option<&str>
    ) -> EmbeddingResult<(Vec<f32>, Vec<f32>)> {
        let intent_vec = self.embed_text_internal(content).await?;
        let context_vec = self.contextual_model
            .embed_with_context(content, surrounding).await?;
        Ok((intent_vec, context_vec))
    }
}
```

### Phase 2: Fingerprint Structure Update (Week 1-2)

#### 2.1 Extend SemanticFingerprint

```rust
// UPDATE: crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs

pub struct SemanticFingerprint {
    // ... existing fields (E1-E9, E11-E13) ...

    // E10 UPGRADE: Dual vectors for intent + context (following E5/E8 pattern)
    /// E10 as intent embedding (CLIP-based, 768D)
    pub e10_multimodal_intent: Vec<f32>,
    /// E10 as context embedding (MPNet-based, 768D) - NEW
    pub e10_multimodal_context: Vec<f32>,
    /// DEPRECATED: Legacy single vector (backward compat)
    pub e10_multimodal: Vec<f32>,
}
```

#### 2.2 Update MultiArrayProvider

```rust
// UPDATE: crates/context-graph-embeddings/src/provider/multi_array.rs

// Add E10 dual adapter (following E5/E8 pattern)
struct MultimodalDualEmbedderAdapter {
    model: Arc<MultimodalModel>,
}

impl MultimodalDualEmbedderAdapter {
    async fn embed_dual(&self, content: &str) -> CoreResult<(Vec<f32>, Vec<f32>)> {
        self.model.embed_dual(content).await.map_err(|e| {
            CoreError::Embedding(format!("E10 dual embedding failed: {}", e))
        })
    }
}

// In embed_all():
let (e10_intent_vec, e10_context_vec) = r10?;
// ...
fingerprint = SemanticFingerprint {
    // ...
    e10_multimodal_intent: e10_intent_vec,
    e10_multimodal_context: e10_context_vec,
    e10_multimodal: Vec::new(), // Empty - using new dual format
    // ...
};
```

### Phase 3: Context Marker Detection (Week 2)

#### 3.1 Create Context Markers Module

```rust
// NEW: crates/context-graph-embeddings/src/models/pretrained/contextual/marker_detection.rs

/// Context continuation markers
pub const CONTINUATION_MARKERS: &[&str] = &[
    "also", "additionally", "furthermore", "moreover", "similarly",
    "in addition", "as well", "along with", "together with",
    "related to", "concerning", "regarding", "about",
];

/// Context shift markers
pub const SHIFT_MARKERS: &[&str] = &[
    "however", "but", "although", "nevertheless", "conversely",
    "on the other hand", "in contrast", "unlike", "whereas",
    "different from", "alternatively", "instead",
];

/// Reference markers
pub const REFERENCE_MARKERS: &[&str] = &[
    "this", "that", "these", "those", "it", "they",
    "the above", "the following", "as mentioned", "previously",
    "earlier", "later", "below", "see also",
];

#[derive(Debug, Clone)]
pub struct ContextMarkerResult {
    pub continuation_count: usize,
    pub shift_count: usize,
    pub reference_count: usize,
    pub context_type: ContextType,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContextType {
    Continuation,  // Topic continues
    Shift,         // Topic changes
    Reference,     // References other content
    Standalone,    // No clear context signals
}

pub fn detect_context_markers(text: &str) -> ContextMarkerResult {
    let text_lower = text.to_lowercase();

    let continuation = CONTINUATION_MARKERS.iter()
        .filter(|m| text_lower.contains(*m)).count();
    let shift = SHIFT_MARKERS.iter()
        .filter(|m| text_lower.contains(*m)).count();
    let reference = REFERENCE_MARKERS.iter()
        .filter(|m| text_lower.contains(*m)).count();

    let context_type = if shift > continuation + reference {
        ContextType::Shift
    } else if continuation > reference {
        ContextType::Continuation
    } else if reference > 0 {
        ContextType::Reference
    } else {
        ContextType::Standalone
    };

    let total = continuation + shift + reference;
    let confidence = (total as f32).min(5.0) / 5.0;

    ContextMarkerResult {
        continuation_count: continuation,
        shift_count: shift,
        reference_count: reference,
        context_type,
        confidence,
    }
}
```

### Phase 4: Asymmetric Similarity (Week 2-3)

#### 4.1 Context-Aware Similarity

```rust
// NEW: crates/context-graph-core/src/context/asymmetric.rs

/// Context relationship modifiers (following E5/E8 pattern)
pub const CONTINUATION_BOOST: f32 = 1.15;  // Same topic continues
pub const SHIFT_DAMPEN: f32 = 0.85;        // Topic changes
pub const REFERENCE_BOOST: f32 = 1.10;     // References prior content

/// Compute E10 context-aware fingerprint similarity
pub fn compute_e10_context_similarity(
    query: &SemanticFingerprint,
    doc: &SemanticFingerprint,
    query_context_type: ContextType,
) -> f32 {
    // Base similarity on context vectors
    let base_sim = cosine_similarity(
        &query.e10_multimodal_context,
        &doc.e10_multimodal_context
    );

    // Apply context-aware modifier
    let modifier = match query_context_type {
        ContextType::Continuation => CONTINUATION_BOOST,
        ContextType::Shift => SHIFT_DAMPEN,
        ContextType::Reference => REFERENCE_BOOST,
        ContextType::Standalone => 1.0,
    };

    base_sim * modifier
}

/// Compute combined E10 score (intent + context)
pub fn compute_e10_combined_similarity(
    query: &SemanticFingerprint,
    doc: &SemanticFingerprint,
    intent_weight: f32,  // e.g., 0.6
    context_weight: f32, // e.g., 0.4
) -> f32 {
    let intent_sim = cosine_similarity(
        &query.e10_multimodal_intent,
        &doc.e10_multimodal_intent
    );
    let context_sim = cosine_similarity(
        &query.e10_multimodal_context,
        &doc.e10_multimodal_context
    );

    intent_weight * intent_sim + context_weight * context_sim
}
```

### Phase 5: MCP Tools (Week 3)

#### 5.1 Context Search Tools

```rust
// NEW: crates/context-graph-mcp/src/handlers/tools/context_tools.rs

/// Tool: search_context
/// Find memories with similar contextual relationships
pub async fn search_context(
    query: &str,
    surrounding_context: Option<&str>,
    top_k: usize,
    min_score: f32,
    include_content: bool,
) -> Result<Vec<ContextSearchResult>, ToolError>;

/// Tool: get_context_chain
/// Find sequence of contextually related memories
pub async fn get_context_chain(
    anchor_id: Uuid,
    direction: ContextDirection,  // Forward, Backward, Both
    max_hops: usize,
    min_similarity: f32,
) -> Result<ContextChain, ToolError>;

/// Tool: detect_context_shift
/// Identify where topics shift in a conversation
pub async fn detect_context_shift(
    session_id: &str,
    sensitivity: f32,  // 0.0-1.0
) -> Result<Vec<ContextShiftPoint>, ToolError>;
```

#### 5.2 Tool Definitions

```rust
// NEW: crates/context-graph-mcp/src/tools/definitions/context.rs

pub fn search_context_definition() -> ToolDefinition {
    ToolDefinition {
        name: "search_context".to_string(),
        description: "Find memories with similar contextual relationships. \
            Uses E10 context embeddings to find content that fits the same \
            conversational or topical context.".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to find contextually similar memories for"
                },
                "surrounding_context": {
                    "type": "string",
                    "description": "Optional surrounding context for better matching"
                },
                "top_k": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum results to return"
                },
                "min_score": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Minimum context similarity threshold"
                }
            },
            "required": ["query"]
        }),
    }
}
```

### Phase 6: Weight Profile (Week 3)

```rust
// UPDATE: crates/context-graph-core/src/similarity/config.rs

/// Create weights emphasizing contextual understanding
pub fn context_aware_weights() -> [f32; NUM_EMBEDDERS] {
    let mut weights = [0.015; NUM_EMBEDDERS];
    weights[0] = 0.30;   // E1 Semantic (baseline)
    weights[9] = 0.40;   // E10 Context (PRIMARY) - uses context vector
    weights[4] = 0.15;   // E5 Causal (supporting)
    weights[3] = 0.10;   // E4 Sequence (supporting)
    // E2-E4 temporal at 0.0 per AP-71
    weights
}

/// Update conversation_search to use context
pub fn conversation_search_weights() -> [f32; NUM_EMBEDDERS] {
    let mut weights = [0.015; NUM_EMBEDDERS];
    weights[0] = 0.35;   // E1 Semantic
    weights[9] = 0.35;   // E10 Context (context vector)
    weights[3] = 0.15;   // E4 Sequence
    weights[4] = 0.10;   // E5 Causal
    weights
}
```

### Phase 7: Benchmark Suite (Week 4)

#### 7.1 Stress Tests

```rust
// ADD TO: crates/context-graph-benchmark/src/stress_corpus.rs

pub fn build_e10_context_corpus() -> EmbedderStressConfig {
    EmbedderStressConfig {
        embedder: EmbedderIndex::E10Multimodal,
        name: "E10 Context",
        description: "Contextual relationship understanding",
        corpus: vec![
            // Context continuation
            StressCorpusEntry {
                content: "Rust's ownership system prevents data races. \
                    Additionally, the borrow checker ensures memory safety.".into(),
                doc_id: 0,
                e1_limitation: Some("E1 sees Rust/memory but not continuation".into()),
                metadata: Some(json!({"context_type": "continuation"})),
            },
            // Context shift
            StressCorpusEntry {
                content: "The database connection failed. However, we can \
                    fall back to the cache for read operations.".into(),
                doc_id: 1,
                e1_limitation: Some("E1 misses the contrast/shift signal".into()),
                metadata: Some(json!({"context_type": "shift"})),
            },
            // Reference
            StressCorpusEntry {
                content: "As mentioned earlier, the API returns JSON. \
                    This format allows easy parsing in JavaScript.".into(),
                doc_id: 2,
                e1_limitation: Some("E1 doesn't see the reference link".into()),
                metadata: Some(json!({"context_type": "reference"})),
            },
            // Q&A context
            StressCorpusEntry {
                content: "Q: How do I handle errors in async Rust? \
                    A: Use the ? operator with Result types.".into(),
                doc_id: 3,
                e1_limitation: Some("E1 sees keywords but not Q&A structure".into()),
                metadata: Some(json!({"context_type": "qa_pair"})),
            },
        ],
        queries: vec![
            StressQuery {
                query: "What else does Rust's ownership provide?".into(),
                target_embedder: EmbedderIndex::E10Multimodal,
                expected_top_docs: vec![0],  // Continuation context
                anti_expected_docs: vec![1], // Shift context
                e1_failure_reason: "E1 may rank by Rust keywords, not continuation".into(),
            },
            StressQuery {
                query: "What's the alternative when the main approach fails?".into(),
                target_embedder: EmbedderIndex::E10Multimodal,
                expected_top_docs: vec![1],  // Shift/fallback context
                anti_expected_docs: vec![],
                e1_failure_reason: "E1 may miss 'however' shift signal".into(),
            },
        ],
    }
}
```

#### 7.2 Benchmark Binary

```toml
# ADD TO: crates/context-graph-benchmark/Cargo.toml

[[bin]]
name = "context-bench"
path = "src/bin/context_bench.rs"
required-features = ["bin"]

[[bin]]
name = "context-realdata-bench"
path = "src/bin/context_realdata_bench.rs"
required-features = ["bin", "real-embeddings"]
```

---

## Part 6: Success Metrics

### Minimum Viable Integration

| Metric | Target | Measurement |
|--------|--------|-------------|
| E10 Context MRR | >0.75 | Stress test queries |
| Context Marker Detection | >80% accuracy | Annotated test set |
| Continuation Detection | >70% precision | Q&A pair matching |
| Shift Detection | >65% precision | Topic change detection |
| Storage Overhead | <10% | Fingerprint size increase |
| Latency Overhead | <20ms | E10 dual embedding time |

### Stretch Goals

| Metric | Target | Notes |
|--------|--------|-------|
| E10 Context MRR | >0.85 | Competitive with E1 |
| Context Chain Recall@5 | >70% | Multi-hop context |
| Conversation Coherence | r>0.6 | Session continuity |
| E10 Ablation Delta | >3% | Performance(all) - Performance(no E10 context) |

---

## Part 7: Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Model loading overhead | Medium | Low | Lazy loading, shared tokenizer |
| Storage increase (~6KB) | Certain | Low | Acceptable per constitution |
| E10 complexity increase | High | Medium | Clear separation of intent/context |
| Backward compatibility | Medium | High | Keep legacy e10_multimodal field |
| Context detection noise | Medium | Medium | Threshold tuning, confidence scores |

---

## Part 8: Alternative: Context as Re-Ranker

If full integration is deemed too complex, a lightweight alternative:

```rust
// Contextual model as Stage 3.5 re-ranker (no fingerprint changes)

pub struct ContextualReranker {
    model: ContextualModel,
}

impl ContextualReranker {
    /// Re-rank candidates based on contextual fit with query
    pub async fn rerank(
        &self,
        query: &str,
        query_context: Option<&str>,
        candidates: &[(Uuid, f32, String)], // id, score, content
        top_k: usize,
    ) -> Vec<(Uuid, f32)> {
        let query_with_context = match query_context {
            Some(ctx) => format!("{} [CTX] {}", ctx, query),
            None => query.to_string(),
        };

        let query_vec = self.model.embed(&query_with_context).await?;

        let mut scored: Vec<_> = candidates.iter()
            .map(|(id, base_score, content)| {
                let content_vec = self.model.embed(content).await?;
                let context_sim = cosine_similarity(&query_vec, &content_vec);
                // Blend: 70% base score + 30% context
                let final_score = 0.7 * base_score + 0.3 * context_sim;
                (*id, final_score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.into_iter().take(top_k).collect()
    }
}
```

**Pros:**
- No fingerprint changes
- Can be toggled on/off
- Simpler implementation

**Cons:**
- Not part of fingerprint (can't cluster)
- Adds latency per query
- Limited to re-ranking

---

## Part 9: Implementation Timeline

| Phase | Task | Effort | Priority |
|-------|------|--------|----------|
| **Week 1** | Add ContextualModel | 8-10h | HIGH |
| **Week 1** | Update MultimodalModel | 6-8h | HIGH |
| **Week 1-2** | Extend SemanticFingerprint | 4-6h | HIGH |
| **Week 2** | Context marker detection | 4-6h | MEDIUM |
| **Week 2** | Update MultiArrayProvider | 4-6h | HIGH |
| **Week 2-3** | Asymmetric similarity | 4-6h | MEDIUM |
| **Week 3** | MCP tools | 8-10h | MEDIUM |
| **Week 3** | Weight profiles | 2-3h | MEDIUM |
| **Week 4** | Stress tests | 6-8h | HIGH |
| **Week 4** | Benchmark binaries | 6-8h | HIGH |
| **Week 4** | Integration testing | 4-6h | HIGH |

**Total Estimated Effort:** 55-75 hours

---

## Part 10: Constitution Updates

### New Architecture Rules

```yaml
# ADD TO: constitution.yaml arch_rules

ARCH-28: "E10 uses dual vectors: intent (CLIP) and context (MPNet)"
ARCH-29: "Context markers (continuation/shift/reference) inform E10 similarity"
ARCH-30: "E10 context embedding uses surrounding text when available"
```

### Updated Embedder Categories

```yaml
# UPDATE: constitution.yaml embedder_categories

SEMANTIC_ENHANCERS:
  embedders: [E5, E6, E7, E10, E12, E13]
  # E10 now dual-purpose: V_multimodality (intent) + V_context (relationships)
  properties:
    - "E10 (V_multimodality): Adds INTENT understanding via CLIP text encoder"
    - "E10 (V_context): Adds CONTEXTUAL understanding via MPNet encoder"
```

---

## Appendix A: File References

| File | Purpose | Status |
|------|---------|--------|
| `embeddings/src/models/pretrained/contextual/mod.rs` | ContextualModel module | NEW |
| `embeddings/src/models/pretrained/contextual/model.rs` | ContextualModel struct | NEW |
| `embeddings/src/models/pretrained/contextual/marker_detection.rs` | Context markers | NEW |
| `embeddings/src/models/pretrained/multimodal/model.rs` | MultimodalModel | UPDATE |
| `embeddings/src/provider/multi_array.rs` | Provider | UPDATE |
| `core/src/types/fingerprint/semantic/fingerprint.rs` | Fingerprint struct | UPDATE |
| `core/src/context/asymmetric.rs` | Context similarity | NEW |
| `core/src/similarity/config.rs` | Weight profiles | UPDATE |
| `mcp/src/handlers/tools/context_tools.rs` | MCP tools | NEW |
| `mcp/src/tools/definitions/context.rs` | Tool schemas | NEW |
| `benchmark/src/stress_corpus.rs` | Stress tests | UPDATE |
| `benchmark/src/bin/context_bench.rs` | Benchmark binary | NEW |

## Appendix B: Model Comparison

| Property | E10 CLIP (Current) | Contextual MPNet | Combined E10 |
|----------|-------------------|------------------|--------------|
| Dimension | 768D | 768D | 768D + 768D |
| Training Data | 400M image-text | 1.17B text pairs | Both |
| Specialty | Image-text alignment | Sentence similarity | Intent + Context |
| Max Tokens | 77 | 384 | 77 / 384 |
| Best For | Cross-modal | Contextual flow | Both use cases |

## Appendix C: Training Data Overlap Analysis

The contextual model's training includes several sources that overlap with context-graph's use cases:

| Source | Relevance |
|--------|-----------|
| Reddit (726M) | Conversational memory context |
| Stack Exchange (68M) | Technical Q&A context |
| S2ORC Citations (210M) | Document reference context |
| Code Search (1.1M) | Code-text context alignment |
| Natural Questions (100K) | Factual retrieval context |

This training makes `all-mpnet-base-v2` particularly well-suited for understanding contextual relationships in conversation-style and documentation-style content - exactly the use cases context-graph targets.

---

## Conclusion

The contextual model (`all-mpnet-base-v2`) is a high-quality sentence encoder that can significantly enhance the context-graph system. Rather than adding a 14th embedder (which would violate constitutional constraints), the recommended approach is to **upgrade E10 (V_multimodality) with dual intent/context vectors**, following the successful patterns established by E5 and E8.

This integration:
1. **Preserves architectural integrity** - No new embedder slots
2. **Follows proven patterns** - Dual projections like E5/E8
3. **Leverages model strengths** - 1.17B pairs of contextual training
4. **Adds unique capability** - Context flow understanding
5. **Complements E1** - Enhances semantic foundation, doesn't compete

The estimated effort of 55-75 hours is comparable to E5 and E8 upgrades, and the expected quality improvements (3-5% ablation delta) justify the investment.
