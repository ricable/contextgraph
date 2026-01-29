# PRD 05: 4-Embedder Stack

**Version**: 4.0.0 | **Parent**: [PRD 01 Overview](PRD_01_OVERVIEW.md) | **Language**: Rust

---

## 1. Design Philosophy

The embedder stack is designed for **consumer hardware**:

- **4 embedders** (not 13-15): Reduced from research system for practical use
- **384D max**: Smaller dimensions = less RAM, faster search
- **ONNX format**: CPU-optimized, cross-platform
- **Quantized (INT8)**: 50% smaller, nearly same quality
- **No LLM inference**: Removed causal/reasoning embedders that need GPUs
- **Tiered loading**: Free tier loads 2 models; Pro loads 3 (BM25 is algorithmic)

---

## 2. Embedder Specifications

### E1: Semantic Similarity (PRIMARY)

| Property | Value |
|----------|-------|
| Model | bge-small-en-v1.5 (BAAI) |
| Dimension | 384 |
| Size | 65MB (INT8 ONNX) |
| Speed | 50ms/chunk (M1), 100ms/chunk (Intel i5) |
| Tier | FREE |
| Purpose | Core semantic search |

**What it finds**: "quarterly revenue decline" matches "Q3 earnings drop"
**Role in pipeline**: Foundation embedder. All search queries start here. Stage 2 ranking.

### E6: Keyword Expansion (SPLADE)

| Property | Value |
|----------|-------|
| Model | SPLADE-cocondenser-selfdistil (Naver) |
| Dimension | Sparse (30K vocabulary) |
| Size | 55MB (INT8 ONNX) |
| Speed | 30ms/chunk |
| Tier | FREE |
| Purpose | Exact term matching + expansion |

**What it finds**: "Q3 earnings" also matches "third quarter revenue", "Q3 financial results"
**Role in pipeline**: Stage 2 ranking alongside E1. Catches exact terminology E1 misses.

### E12: Precision Reranking (ColBERT)

| Property | Value |
|----------|-------|
| Model | ColBERT-v2-small |
| Dimension | 64 per token |
| Size | 110MB (INT8 ONNX) |
| Speed | 100ms for top 50 candidates |
| Tier | PRO |
| Purpose | Final reranking for exact phrase matches |

**What it finds**: "revenue increased significantly" ranks higher than "revenue did not increase"
**Role in pipeline**: Stage 3 (final rerank). Token-level MaxSim scoring. Only runs on top 50 candidates.

### E13: Fast Recall (BM25)

| Property | Value |
|----------|-------|
| Model | None (algorithmic -- BM25/TF-IDF) |
| Dimension | N/A (inverted index) |
| Size | ~2MB index per 1000 documents |
| Speed | <5ms for any query |
| Tier | FREE |
| Purpose | Fast initial candidate retrieval |

**What it finds**: Exact keyword matches for terms like "invoice", "contract", "deadline"
**Role in pipeline**: Stage 1. Retrieves initial 500 candidates from inverted index.

---

## 3. Footprint Summary

| Metric | Free Tier | Pro Tier |
|--------|-----------|----------|
| Models to download | 2 (E1, E6) | 3 (+ E12) |
| Model disk space | ~120MB | ~230MB |
| RAM at runtime | ~600MB | ~1.0GB |
| Per-chunk embed time | ~80ms | ~180ms |
| Search latency | <100ms | <200ms |

---

## 4. Provenance Linkage

**Every embedding vector is traceable back to its source document, page, and paragraph.** The chain is: `embedding key (e1:{chunk_uuid})` -> `ChunkData` (text + full `Provenance`) -> source file on disk. No embedding is stored without its chunk existing first; the ingestion pipeline (PRD 06) creates ChunkData with full Provenance before calling `embed_chunk()`.

For the canonical Provenance struct fields, storage layout, and complete chain specification, see [PRD 04 Section 5.2](PRD_04_STORAGE_ARCHITECTURE.md#52-the-provenance-chain-how-embeddings-trace-back-to-source).

---

## 5. Embedding Engine Implementation

```rust
use ort::{Session, Environment, GraphOptimizationLevel, ExecutionProvider};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Consumer-optimized embedding engine
pub struct EmbeddingEngine {
    env: Arc<Environment>,
    models: HashMap<EmbedderId, Option<Session>>,
    tier: LicenseTier,
    model_dir: PathBuf,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum EmbedderId {
    E1,     // Semantic (FREE)
    E6,     // Keywords (FREE)
    E12,    // ColBERT (PRO)
    // E13 is BM25, not a neural model
}

impl EmbedderId {
    pub fn model_dir_name(&self) -> &'static str {
        match self {
            Self::E1 => "bge-small-en-v1.5",
            Self::E6 => "splade-distil",
            Self::E12 => "colbert-small",
        }
    }

    pub fn dimension(&self) -> usize {
        match self {
            Self::E1 => 384,
            Self::E6 => 0,    // Sparse
            Self::E12 => 64,  // Per token
        }
    }

    pub fn is_sparse(&self) -> bool {
        matches!(self, Self::E6)
    }

    pub fn is_free_tier(&self) -> bool {
        matches!(self, Self::E1 | Self::E6)
    }
}

impl EmbeddingEngine {
    pub fn new(model_dir: &Path, tier: LicenseTier) -> Result<Self> {
        let env = Environment::builder()
            .with_name("casetrack")
            .with_execution_providers([
                #[cfg(target_os = "macos")]
                ExecutionProvider::CoreML(Default::default()),
                #[cfg(target_os = "windows")]
                ExecutionProvider::DirectML(Default::default()),
                ExecutionProvider::CPU(Default::default()),
            ])
            .build()?;

        let mut engine = Self {
            env: Arc::new(env),
            models: HashMap::new(),
            tier,
            model_dir: model_dir.to_path_buf(),
        };

        // Load models based on tier
        for id in Self::models_for_tier(tier) {
            engine.load_model(id)?;
        }

        Ok(engine)
    }

    fn load_model(&mut self, id: EmbedderId) -> Result<()> {
        let path = self.model_dir
            .join(id.model_dir_name())
            .join("model.onnx");

        if path.exists() {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(2)?  // Limit threads for consumer hardware
                .with_model_from_file(&path)?;
            self.models.insert(id, Some(session));
        } else {
            self.models.insert(id, None);  // Will download on demand
        }
        Ok(())
    }

    fn models_for_tier(tier: LicenseTier) -> Vec<EmbedderId> {
        match tier {
            LicenseTier::Free => vec![
                EmbedderId::E1,
                EmbedderId::E6,
            ],
            _ => vec![
                EmbedderId::E1,
                EmbedderId::E6,
                EmbedderId::E12,
            ],
        }
    }

    /// Embed a chunk with all active models
    pub fn embed_chunk(&self, text: &str) -> Result<ChunkEmbeddings> {
        let mut embeddings = ChunkEmbeddings::default();

        for (id, session) in &self.models {
            if let Some(session) = session {
                match id {
                    EmbedderId::E6 => {
                        embeddings.e6 = Some(self.run_sparse_inference(session, text)?);
                    }
                    EmbedderId::E12 => {
                        embeddings.e12 = Some(self.run_token_inference(session, text)?);
                    }
                    _ => {
                        let vec = self.run_dense_inference(session, text)?;
                        match id {
                            EmbedderId::E1 => embeddings.e1 = Some(vec),
                            _ => {}
                        }
                    }
                }
            }
        }

        Ok(embeddings)
    }

    /// Embed a query (same models, but may use query-specific prefixes)
    pub fn embed_query(&self, query: &str, embedder: EmbedderId) -> Result<QueryEmbedding> {
        let session = self.models.get(&embedder)
            .ok_or(CaseTrackError::EmbedderNotLoaded(embedder))?
            .as_ref()
            .ok_or(CaseTrackError::ModelNotDownloaded(embedder))?;

        match embedder {
            EmbedderId::E6 => {
                Ok(QueryEmbedding::Sparse(self.run_sparse_inference(session, query)?))
            }
            EmbedderId::E12 => {
                Ok(QueryEmbedding::Token(self.run_token_inference(session, query)?))
            }
            _ => {
                Ok(QueryEmbedding::Dense(self.run_dense_inference(session, query)?))
            }
        }
    }

    fn run_dense_inference(&self, session: &Session, text: &str) -> Result<Vec<f32>> {
        let tokens = self.tokenize(text, 512)?;  // Max 512 tokens

        let outputs = session.run(ort::inputs![
            "input_ids" => tokens.input_ids,
            "attention_mask" => tokens.attention_mask,
        ]?)?;

        let hidden = outputs["last_hidden_state"].extract_tensor::<f32>()?;
        Ok(mean_pool(&hidden, &tokens.attention_mask))
    }

    fn run_sparse_inference(&self, session: &Session, text: &str) -> Result<SparseVec> {
        let tokens = self.tokenize(text, 512)?;

        let outputs = session.run(ort::inputs![
            "input_ids" => tokens.input_ids,
            "attention_mask" => tokens.attention_mask,
        ]?)?;

        let logits = outputs["logits"].extract_tensor::<f32>()?;
        Ok(splade_max_pool(&logits, &tokens.attention_mask))
    }

    fn run_token_inference(&self, session: &Session, text: &str) -> Result<TokenEmbeddings> {
        let tokens = self.tokenize(text, 512)?;

        let outputs = session.run(ort::inputs![
            "input_ids" => tokens.input_ids,
            "attention_mask" => tokens.attention_mask,
        ]?)?;

        let hidden = outputs["last_hidden_state"].extract_tensor::<f32>()?;
        Ok(extract_token_embeddings(&hidden, &tokens.attention_mask))
    }
}

/// Embeddings for a single chunk
#[derive(Default)]
pub struct ChunkEmbeddings {
    pub e1: Option<Vec<f32>>,           // 384D
    pub e6: Option<SparseVec>,          // Sparse
    pub e12: Option<TokenEmbeddings>,   // 64D per token
}

pub enum QueryEmbedding {
    Dense(Vec<f32>),
    Sparse(SparseVec),
    Token(TokenEmbeddings),
}
```

---

## 6. Model Management

### 6.1 Lazy Loading

Models not needed for the current operation are not loaded:

```rust
/// Load a model on demand if not already loaded
pub fn ensure_model_loaded(&mut self, id: EmbedderId) -> Result<&Session> {
    if let Some(Some(session)) = self.models.get(&id) {
        return Ok(session);
    }

    // Check if model files exist
    let model_path = self.model_dir
        .join(id.model_dir_name())
        .join("model.onnx");

    if !model_path.exists() {
        return Err(CaseTrackError::ModelNotDownloaded(id));
    }

    // Load model
    tracing::info!("Lazy-loading model {:?}", id);
    self.load_model(id)?;

    self.models.get(&id)
        .and_then(|opt| opt.as_ref())
        .ok_or(CaseTrackError::ModelLoadFailed(id))
}
```

### 6.2 Memory Pressure Handling

```rust
/// Unload least-recently-used models when memory is constrained
pub fn handle_memory_pressure(&mut self) {
    let available_mb = sysinfo::System::new_all()
        .available_memory() / (1024 * 1024);

    if available_mb < 1024 {  // Less than 1GB free
        tracing::warn!("Low memory ({} MB free). Unloading Pro models.", available_mb);

        // Unload Pro-tier models (keep Free tier loaded)
        for id in &[EmbedderId::E12] {
            if let Some(slot) = self.models.get_mut(id) {
                *slot = None;
            }
        }
    }
}
```

---

## 7. ONNX Model Conversion Notes

For the fresh project build, models must be converted from PyTorch to ONNX:

```python
# Example: Convert bge-small-en-v1.5 to ONNX
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

dummy_input = tokenizer("hello world", return_tensors="pt")

torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"},
    },
    opset_version=14,
)

# Quantize to INT8
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "model.onnx",
    "model_int8.onnx",
    weight_type=QuantType.QInt8,
)
```

A `scripts/convert_models.py` script should be included in the repository to automate this for all 3 neural models. Pre-converted ONNX models should be hosted on Hugging Face under a `casetrack/` organization.

---

*CaseTrack PRD v4.0.0 -- Document 5 of 10*
