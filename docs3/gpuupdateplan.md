# GPU Acceleration Update Plan
## Context Graph Embeddings - RTX 5090 Migration

```yaml
metadata:
  version: 1.0.0
  created: 2026-01-01
  target_hardware: NVIDIA RTX 5090 32GB (Blackwell GB202)
  cuda_version: 13.1
  driver_version: 591.44
  framework: Candle 0.9.2-alpha (HuggingFace)
  estimated_total_effort: 120-160 hours
  phases: 5
  priority: CRITICAL
```

---

## Executive Summary

### Current State
- **CPU-only implementation** using `Vec<f32>` operations
- Tests taking **60+ seconds** for FuseMoE (323M parameters)
- **234 CPU-bound operations** identified across 20 files
- **84 specific functions** requiring GPU migration

### Target State
- **Full GPU acceleration** on RTX 5090 (21,760 CUDA cores, 680 tensor cores)
- Expected **30-100x speedup** for matrix operations
- Sub-second inference for FuseMoE layer
- CUDA 13.1 with Blackwell architecture optimizations

### Hardware Specifications
| Spec | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 5090 |
| VRAM | 32GB GDDR7 |
| Bandwidth | 1,792 GB/s |
| CUDA Cores | 21,760 |
| Tensor Cores | 680 (5th gen) |
| Compute Capability | 12.0 |
| FP16 TFLOPS | 209.5 |
| INT8 TOPS | 3,352 |

---

## Phase 0: Foundation Setup (COMPLETED)

### 0.1 Dependencies Configuration

**Status: DONE**

```toml
# Cargo.toml (workspace)
[workspace.dependencies]
candle-core = { version = "0.9.2-alpha", features = ["cuda"] }
candle-nn = { version = "0.9.2-alpha", features = ["cuda"] }
candle-transformers = { version = "0.9.2-alpha", features = ["cuda"] }
```

```toml
# crates/context-graph-embeddings/Cargo.toml
[dependencies]
candle-core = { workspace = true, optional = true }
candle-nn = { workspace = true, optional = true }

[features]
default = ["stub"]  # Temporary: stub while GPU implementations are built
cuda = ["candle"]
candle = ["dep:candle-core", "dep:candle-nn"]
```

### 0.2 CUDA 13.1 Compatibility

**Status: DONE**

- cudarc 0.17.8+ supports CUDA 13.x
- Candle 0.9.2-alpha includes CUDA 13 support via PR #3089
- Environment: `CUDA_PATH=/usr/local/cuda-13.1`

---

## Phase 1: Core Tensor Infrastructure
**Estimated: 16-24 hours**

### 1.1 Create GPU Device Manager

**File:** `crates/context-graph-embeddings/src/gpu/mod.rs`

```rust
//! GPU device management for RTX 5090 acceleration.

use candle_core::{Device, DType};
use std::sync::OnceLock;

static GPU_DEVICE: OnceLock<Device> = OnceLock::new();

/// Initialize GPU device (call once at startup).
pub fn init_gpu() -> Result<&'static Device, candle_core::Error> {
    GPU_DEVICE.get_or_try_init(|| {
        Device::new_cuda(0)
    })
}

/// Get the active GPU device.
pub fn device() -> &'static Device {
    GPU_DEVICE.get().expect("GPU not initialized - call init_gpu() first")
}

/// Default dtype for embeddings (f32 for accuracy, f16 for speed).
pub fn default_dtype() -> DType {
    DType::F32
}
```

**Tasks:**
| ID | Task | Complexity | Hours |
|----|------|------------|-------|
| P1-01 | Create `src/gpu/mod.rs` module structure | Simple | 1 |
| P1-02 | Implement `GpuDevice` singleton with CUDA 0 | Simple | 2 |
| P1-03 | Add dtype configuration (F32/F16/BF16) | Simple | 1 |
| P1-04 | Implement device memory tracking | Medium | 3 |
| P1-05 | Add error handling for GPU initialization | Simple | 2 |
| P1-06 | Create `GpuTensor` wrapper type | Medium | 4 |
| P1-07 | Unit tests for GPU initialization | Simple | 2 |

### 1.2 Core Type GPU Conversions

**Files to modify:**
- `src/types/embedding.rs`
- `src/types/fused.rs`
- `src/types/concatenated.rs`

**Pattern to implement:**

```rust
impl ModelEmbedding {
    /// Convert to GPU tensor.
    #[cfg(feature = "candle")]
    pub fn to_tensor(&self, device: &Device) -> candle_core::Result<Tensor> {
        Tensor::from_slice(&self.vector, (self.vector.len(),), device)
    }

    /// Create from GPU tensor.
    #[cfg(feature = "candle")]
    pub fn from_tensor(tensor: &Tensor, model_id: ModelId) -> candle_core::Result<Self> {
        let vector = tensor.to_vec1::<f32>()?;
        Ok(Self::new(model_id, vector))
    }
}
```

**Tasks:**
| ID | Task | File | Complexity | Hours |
|----|------|------|------------|-------|
| P1-08 | `ModelEmbedding::to_tensor()` | embedding.rs | Simple | 1 |
| P1-09 | `ModelEmbedding::from_tensor()` | embedding.rs | Simple | 1 |
| P1-10 | `FusedEmbedding::to_tensor()` | fused.rs | Simple | 1 |
| P1-11 | `FusedEmbedding::from_tensor()` | fused.rs | Simple | 1 |
| P1-12 | `ConcatenatedEmbedding::to_tensor()` | concatenated.rs | Medium | 2 |
| P1-13 | `ConcatenatedEmbedding::from_tensor()` | concatenated.rs | Medium | 2 |

---

## Phase 2: Core Operations GPU Migration
**Estimated: 24-32 hours**

### 2.1 Simple Operations (42 total)

These are direct 1:1 replacements with Candle tensor operations.

#### L2 Normalization Pattern

**Before (CPU):**
```rust
fn l2_norm(&self) -> f32 {
    self.values.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn normalize(&mut self) {
    let norm = self.l2_norm();
    for v in &mut self.values { *v /= norm; }
}
```

**After (GPU):**
```rust
#[cfg(feature = "candle")]
fn l2_norm_gpu(&self, tensor: &Tensor) -> candle_core::Result<f32> {
    tensor.sqr()?.sum_all()?.sqrt()?.to_vec0()
}

#[cfg(feature = "candle")]
fn normalize_gpu(tensor: &Tensor) -> candle_core::Result<Tensor> {
    let norm = tensor.sqr()?.sum_all()?.sqrt()?;
    tensor.broadcast_div(&norm)
}
```

#### Cosine Similarity Pattern

**Before (CPU):**
```rust
fn cosine_similarity(&self, other: &Self) -> f32 {
    let dot: f32 = self.values.iter()
        .zip(other.values.iter())
        .map(|(a, b)| a * b)
        .sum();
    let norm_a = self.l2_norm();
    let norm_b = other.l2_norm();
    dot / (norm_a * norm_b + 1e-8)
}
```

**After (GPU):**
```rust
#[cfg(feature = "candle")]
fn cosine_similarity_gpu(a: &Tensor, b: &Tensor) -> candle_core::Result<f32> {
    let dot = a.mul(b)?.sum_all()?;
    let norm_a = a.sqr()?.sum_all()?.sqrt()?;
    let norm_b = b.sqr()?.sum_all()?.sqrt()?;
    let denom = (norm_a * norm_b)? + 1e-8;
    dot.broadcast_div(&denom)?.to_vec0()
}
```

**Tasks:**
| ID | Task | Location | Complexity | Hours |
|----|------|----------|------------|-------|
| P2-01 | `l2_norm()` GPU implementation | embedding.rs:164 | Simple | 0.5 |
| P2-02 | `normalize()` GPU implementation | embedding.rs:183 | Simple | 0.5 |
| P2-03 | `magnitude()` GPU implementation | fused.rs:448 | Simple | 0.5 |
| P2-04 | `cosine_similarity()` GPU | embedding.rs:269 | Simple | 1 |
| P2-05 | `dot_product()` GPU | embedding.rs:286 | Simple | 0.5 |
| P2-06 | `validate()` NaN/Inf check GPU | embedding.rs:141 | Simple | 1 |
| P2-07 | `SparseVector::l2_norm()` GPU | sparse.rs:169 | Simple | 0.5 |
| P2-08 | `SparseVector::sum()` GPU | sparse.rs:165 | Simple | 0.5 |
| P2-09 | `to_flat_vector()` as tensor cat | concatenated.rs:280 | Medium | 2 |

### 2.2 Activation Functions

**File:** `src/fusion/experts.rs`

**Before (CPU):**
```rust
impl Activation {
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Activation::Gelu => {
                const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                const COEF: f32 = 0.044715;
                let inner = SQRT_2_OVER_PI * (x + COEF * x * x * x);
                x * 0.5 * (1.0 + inner.tanh())
            }
            Activation::Relu => x.max(0.0),
            Activation::Silu => x * (1.0 / (1.0 + (-x).exp())),
        }
    }
}
```

**After (GPU):**
```rust
#[cfg(feature = "candle")]
impl Activation {
    pub fn apply_tensor(&self, tensor: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Activation::Gelu => tensor.gelu(),
            Activation::Relu => tensor.relu(),
            Activation::Silu => tensor.silu(),
        }
    }
}
```

**Tasks:**
| ID | Task | Complexity | Hours |
|----|------|------------|-------|
| P2-10 | `Activation::apply_tensor()` for GELU | Simple | 1 |
| P2-11 | `Activation::apply_tensor()` for ReLU | Simple | 0.5 |
| P2-12 | `Activation::apply_tensor()` for SiLU | Simple | 0.5 |
| P2-13 | Unit tests for GPU activations | Simple | 1 |

---

## Phase 3: Fusion Layer GPU Migration
**Estimated: 32-48 hours**

This is the **highest priority** phase - the 60+ second tests are here.

### 3.1 LayerNorm GPU Implementation

**File:** `src/fusion/gating.rs`

**Before (CPU):**
```rust
impl LayerNorm {
    pub fn forward(&self, input: &[f32], batch_size: usize) -> EmbeddingResult<Vec<f32>> {
        let mut output = Vec::with_capacity(input.len());

        for b in 0..batch_size {
            let start = b * self.dim;
            let end = start + self.dim;
            let sample = &input[start..end];

            // Compute mean
            let mean: f32 = sample.iter().sum::<f32>() / self.dim as f32;

            // Compute variance
            let variance: f32 = sample.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / self.dim as f32;

            // Normalize
            let std_inv = 1.0 / (variance + self.eps).sqrt();
            for i in 0..self.dim {
                let normalized = (sample[i] - mean) * std_inv;
                output.push(normalized * self.gamma[i] + self.beta[i]);
            }
        }
        Ok(output)
    }
}
```

**After (GPU) using candle_nn::LayerNorm:**
```rust
#[cfg(feature = "candle")]
use candle_nn::{LayerNorm as CandleLayerNorm, LayerNormConfig};

#[cfg(feature = "candle")]
pub struct GpuLayerNorm {
    inner: CandleLayerNorm,
    dim: usize,
}

#[cfg(feature = "candle")]
impl GpuLayerNorm {
    pub fn new(dim: usize, device: &Device) -> candle_core::Result<Self> {
        let config = LayerNormConfig::default();
        let vb = candle_nn::VarBuilder::zeros(DType::F32, device);
        let inner = candle_nn::layer_norm(dim, config, vb)?;
        Ok(Self { inner, dim })
    }

    pub fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        self.inner.forward(input)
    }
}
```

**Tasks:**
| ID | Task | Complexity | Hours |
|----|------|------------|-------|
| P3-01 | Create `GpuLayerNorm` wrapper | Medium | 3 |
| P3-02 | Implement `forward()` with candle_nn | Medium | 2 |
| P3-03 | Add weight initialization from CPU | Medium | 2 |
| P3-04 | Unit tests for GPU LayerNorm | Simple | 2 |
| P3-05 | Benchmark CPU vs GPU LayerNorm | Simple | 1 |

### 3.2 Linear Layer GPU Implementation

**File:** `src/fusion/gating.rs`

**Before (CPU) - Triple nested loop:**
```rust
impl Linear {
    pub fn forward(&self, input: &[f32], batch_size: usize) -> EmbeddingResult<Vec<f32>> {
        let mut output = Vec::with_capacity(batch_size * self.out_features);

        for b in 0..batch_size {
            let input_offset = b * self.in_features;
            for o in 0..self.out_features {
                let mut sum = self.bias[o];
                let weight_offset = o * self.in_features;
                for i in 0..self.in_features {
                    sum += input[input_offset + i] * self.weights[weight_offset + i];
                }
                output.push(sum);
            }
        }
        Ok(output)
    }
}
```

**After (GPU) - cuBLAS GEMM:**
```rust
#[cfg(feature = "candle")]
use candle_nn::Linear as CandleLinear;

#[cfg(feature = "candle")]
pub struct GpuLinear {
    weight: Tensor,  // [out_features, in_features]
    bias: Tensor,    // [out_features]
}

#[cfg(feature = "candle")]
impl GpuLinear {
    pub fn new(in_features: usize, out_features: usize, device: &Device) -> candle_core::Result<Self> {
        let weight = Tensor::randn(0.0, 1.0, (out_features, in_features), device)?;
        let bias = Tensor::zeros((out_features,), DType::F32, device)?;
        Ok(Self { weight, bias })
    }

    pub fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        // input: [batch_size, in_features]
        // weight: [out_features, in_features]
        // output: [batch_size, out_features]
        input.matmul(&self.weight.t())?.broadcast_add(&self.bias)
    }
}
```

**Expected Speedup:** 50-100x for large matrices (8320 -> 4096)

**Tasks:**
| ID | Task | Complexity | Hours |
|----|------|------------|-------|
| P3-06 | Create `GpuLinear` with weight tensors | Medium | 3 |
| P3-07 | Implement `forward()` with matmul | Medium | 2 |
| P3-08 | Xavier initialization on GPU | Simple | 1 |
| P3-09 | Weight transfer from CPU Linear | Medium | 2 |
| P3-10 | Unit tests comparing CPU/GPU outputs | Medium | 2 |
| P3-11 | Benchmark CPU vs GPU Linear | Simple | 1 |

### 3.3 Softmax with Temperature GPU

**File:** `src/fusion/gating.rs`

**Before (CPU):**
```rust
fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = scaled.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp_values.iter().sum();
    exp_values.iter().map(|&x| x / sum_exp).collect()
}
```

**After (GPU):**
```rust
#[cfg(feature = "candle")]
fn softmax_with_temperature_gpu(logits: &Tensor, temperature: f32) -> candle_core::Result<Tensor> {
    let scaled = (logits / temperature as f64)?;
    scaled.softmax(candle_core::D::Minus1)
}
```

**Tasks:**
| ID | Task | Complexity | Hours |
|----|------|------------|-------|
| P3-12 | Implement `softmax_with_temperature_gpu()` | Simple | 1 |
| P3-13 | Implement `apply_laplace_smoothing_gpu()` | Simple | 1 |
| P3-14 | Implement `select_top_k_gpu()` | Medium | 3 |
| P3-15 | Unit tests for GPU softmax | Simple | 1 |

### 3.4 GatingNetwork Full GPU

**File:** `src/fusion/gating.rs`

```rust
#[cfg(feature = "candle")]
pub struct GpuGatingNetwork {
    layer_norm: GpuLayerNorm,
    projection: GpuLinear,
    num_experts: usize,
    temperature: f32,
    laplace_alpha: f32,
    device: Device,
}

#[cfg(feature = "candle")]
impl GpuGatingNetwork {
    pub fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        // input: [batch_size, TOTAL_CONCATENATED=8320]
        let normalized = self.layer_norm.forward(input)?;
        let logits = self.projection.forward(&normalized)?;
        let probs = softmax_with_temperature_gpu(&logits, self.temperature)?;
        apply_laplace_smoothing_gpu(&probs, self.laplace_alpha, self.num_experts)
    }

    pub fn forward_topk(&self, input: &Tensor, top_k: usize)
        -> candle_core::Result<(Tensor, Tensor)>
    {
        let probs = self.forward(input)?;
        // Returns (indices, weights) for top-k experts
        probs.topk(top_k)
    }
}
```

**Tasks:**
| ID | Task | Complexity | Hours |
|----|------|------------|-------|
| P3-16 | Create `GpuGatingNetwork` struct | Medium | 2 |
| P3-17 | Implement GPU `forward()` pipeline | Complex | 4 |
| P3-18 | Implement GPU `forward_topk()` | Medium | 3 |
| P3-19 | Integration tests | Medium | 2 |

### 3.5 Expert and ExpertPool GPU

**File:** `src/fusion/experts.rs`

```rust
#[cfg(feature = "candle")]
pub struct GpuExpert {
    input_to_hidden: GpuLinear,  // 8320 -> 4096
    hidden_to_output: GpuLinear, // 4096 -> 1536
    activation: Activation,
    expert_id: usize,
}

#[cfg(feature = "candle")]
impl GpuExpert {
    pub fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let hidden = self.input_to_hidden.forward(input)?;
        let activated = self.activation.apply_tensor(&hidden)?;
        self.hidden_to_output.forward(&activated)
    }
}

#[cfg(feature = "candle")]
pub struct GpuExpertPool {
    experts: Vec<GpuExpert>,
    device: Device,
}

#[cfg(feature = "candle")]
impl GpuExpertPool {
    /// Forward pass through top-k experts with weighted combination.
    pub fn forward_topk(
        &self,
        input: &Tensor,      // [batch_size, 8320]
        indices: &Tensor,    // [batch_size, top_k] expert indices
        weights: &Tensor,    // [batch_size, top_k] routing weights
    ) -> candle_core::Result<Tensor> {
        // Batched expert execution on GPU
        let batch_size = input.dim(0)?;
        let top_k = indices.dim(1)?;

        let mut output = Tensor::zeros(
            (batch_size, FUSED_OUTPUT),
            DType::F32,
            &self.device
        )?;

        // Execute each expert and accumulate weighted outputs
        for k in 0..top_k {
            let expert_indices = indices.i((.., k))?;
            let expert_weights = weights.i((.., k))?;

            // TODO: Optimize with batched gather for same-expert samples
            for b in 0..batch_size {
                let idx: usize = expert_indices.i(b)?.to_vec0()?;
                let weight: f32 = expert_weights.i(b)?.to_vec0()?;

                let sample_input = input.i(b)?;
                let expert_output = self.experts[idx].forward(&sample_input)?;

                let weighted = (expert_output * weight)?;
                output = output.slice_assign(&[b..b+1, ..], &weighted.unsqueeze(0)?)?;
            }
        }

        Ok(output)
    }
}
```

**Tasks:**
| ID | Task | Complexity | Hours |
|----|------|------------|-------|
| P3-20 | Create `GpuExpert` struct | Medium | 2 |
| P3-21 | Implement `GpuExpert::forward()` | Medium | 2 |
| P3-22 | Create `GpuExpertPool` struct | Medium | 2 |
| P3-23 | Implement naive `forward_topk()` | Complex | 4 |
| P3-24 | Optimize with batched gather | Complex | 6 |
| P3-25 | Integration tests with GatingNetwork | Complex | 4 |
| P3-26 | Benchmark: expect 60s -> <1s | Simple | 2 |

---

## Phase 4: Pretrained Models GPU Migration
**Estimated: 40-56 hours**

### 4.1 Model Loading Infrastructure

**File:** `src/models/pretrained/loader.rs` (new)

```rust
#[cfg(feature = "candle")]
pub struct ModelLoader {
    device: Device,
    dtype: DType,
}

#[cfg(feature = "candle")]
impl ModelLoader {
    pub fn load_safetensors<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> candle_core::Result<HashMap<String, Tensor>> {
        candle_core::safetensors::load(path, &self.device)
    }

    pub fn load_bert_model<P: AsRef<Path>>(
        &self,
        model_path: P,
        config_path: P,
    ) -> candle_core::Result<BertModel> {
        let config = BertConfig::from_file(config_path)?;
        let vb = VarBuilder::from_safetensors(model_path, self.dtype, &self.device)?;
        BertModel::load(vb, &config)
    }
}
```

**Tasks:**
| ID | Task | Complexity | Hours |
|----|------|------------|-------|
| P4-01 | Create `ModelLoader` struct | Medium | 2 |
| P4-02 | Implement safetensors loading | Medium | 2 |
| P4-03 | Implement BERT model loading | Complex | 4 |
| P4-04 | Implement MiniLM loading | Complex | 4 |
| P4-05 | Implement SPLADE loading | Complex | 4 |
| P4-06 | Implement ColBERT loading | Complex | 4 |

### 4.2 SemanticModel GPU

**File:** `src/models/pretrained/semantic.rs`

```rust
#[cfg(feature = "candle")]
pub struct GpuSemanticModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    config: SemanticConfig,
}

#[cfg(feature = "candle")]
impl GpuSemanticModel {
    pub async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding> {
        let text = input.to_text()?;

        // Tokenize
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| EmbeddingError::TokenizationError(e.to_string()))?;

        let input_ids = Tensor::from_slice(
            encoding.get_ids(),
            (1, encoding.len()),
            &self.device,
        )?;

        let attention_mask = Tensor::from_slice(
            encoding.get_attention_mask(),
            (1, encoding.len()),
            &self.device,
        )?;

        let token_type_ids = Tensor::zeros_like(&input_ids)?;

        // Forward pass
        let output = self.model.forward(&input_ids, &attention_mask, &token_type_ids)?;

        // Mean pooling
        let pooled = mean_pool(&output, &attention_mask)?;

        // Normalize
        let normalized = normalize_l2(&pooled)?;

        // Convert back to CPU
        let vector = normalized.to_vec1::<f32>()?;

        Ok(ModelEmbedding::new(ModelId::Semantic, vector))
    }
}

#[cfg(feature = "candle")]
fn mean_pool(hidden_states: &Tensor, attention_mask: &Tensor) -> candle_core::Result<Tensor> {
    let mask = attention_mask.unsqueeze(2)?.to_dtype(hidden_states.dtype())?;
    let masked = (hidden_states * &mask)?;
    let sum = masked.sum(1)?;
    let count = mask.sum(1)?;
    sum.broadcast_div(&count)
}

#[cfg(feature = "candle")]
fn normalize_l2(tensor: &Tensor) -> candle_core::Result<Tensor> {
    let norm = tensor.sqr()?.sum_keepdim(1)?.sqrt()?;
    tensor.broadcast_div(&(norm + 1e-12)?)
}
```

**Tasks:**
| ID | Task | Complexity | Hours |
|----|------|------------|-------|
| P4-07 | Create `GpuSemanticModel` struct | Medium | 2 |
| P4-08 | Implement GPU `load()` | Complex | 4 |
| P4-09 | Implement GPU `embed()` | Complex | 4 |
| P4-10 | Implement `mean_pool()` helper | Simple | 1 |
| P4-11 | Integration tests | Medium | 2 |

### 4.3 SparseModel GPU

**File:** `src/models/pretrained/sparse.rs`

```rust
#[cfg(feature = "candle")]
pub struct GpuSparseModel {
    model: BertForMaskedLM,  // SPLADE uses MLM head
    tokenizer: Tokenizer,
    device: Device,
    config: SparseConfig,
}

#[cfg(feature = "candle")]
impl GpuSparseModel {
    pub async fn embed(&self, input: &ModelInput) -> EmbeddingResult<SparseVector> {
        let text = input.to_text()?;

        // Tokenize
        let encoding = self.tokenizer.encode(text, true)?;
        let input_ids = Tensor::from_slice(encoding.get_ids(), (1, encoding.len()), &self.device)?;
        let attention_mask = Tensor::from_slice(encoding.get_attention_mask(), (1, encoding.len()), &self.device)?;

        // Forward through MLM head
        let logits = self.model.forward(&input_ids, &attention_mask)?;

        // SPLADE: log(1 + ReLU(logits)) with max pooling over sequence
        let activated = logits.relu()?.log1p()?;  // log(1 + ReLU(x))
        let pooled = activated.max(1)?;  // Max over sequence dimension

        // Convert to sparse (keep top-k or threshold)
        self.to_sparse_vector(&pooled)
    }

    fn to_sparse_vector(&self, dense: &Tensor) -> EmbeddingResult<SparseVector> {
        let values = dense.to_vec1::<f32>()?;
        let threshold = 0.0;  // SPLADE threshold

        let mut indices = Vec::new();
        let mut sparse_values = Vec::new();

        for (idx, &val) in values.iter().enumerate() {
            if val > threshold {
                indices.push(idx as u32);
                sparse_values.push(val);
            }
        }

        Ok(SparseVector::new(indices, sparse_values, values.len()))
    }
}
```

**Tasks:**
| ID | Task | Complexity | Hours |
|----|------|------------|-------|
| P4-12 | Create `GpuSparseModel` struct | Medium | 2 |
| P4-13 | Implement SPLADE forward pass | Complex | 4 |
| P4-14 | Implement log-saturation activation | Simple | 1 |
| P4-15 | Implement sparse conversion | Medium | 2 |
| P4-16 | Integration tests | Medium | 2 |

### 4.4 GraphModel GPU

**Tasks:**
| ID | Task | Complexity | Hours |
|----|------|------------|-------|
| P4-17 | Create `GpuGraphModel` struct | Medium | 2 |
| P4-18 | Implement MiniLM forward pass | Complex | 4 |
| P4-19 | Integration tests | Medium | 2 |

### 4.5 LateInteractionModel GPU (ColBERT)

```rust
#[cfg(feature = "candle")]
pub struct GpuLateInteractionModel {
    model: ColBertModel,
    tokenizer: Tokenizer,
    device: Device,
}

#[cfg(feature = "candle")]
impl GpuLateInteractionModel {
    /// Compute MaxSim score between query and document token embeddings.
    pub fn maxsim_score_gpu(
        query_embs: &Tensor,   // [num_query_tokens, dim]
        doc_embs: &Tensor,     // [num_doc_tokens, dim]
    ) -> candle_core::Result<f32> {
        // Compute all pairwise similarities
        let similarities = query_embs.matmul(&doc_embs.t())?;  // [Q, D]

        // Max over document tokens for each query token
        let max_sims = similarities.max(1)?;  // [Q]

        // Mean over query tokens
        let score = max_sims.mean_all()?.to_vec0::<f32>()?;

        Ok(score)
    }
}
```

**Tasks:**
| ID | Task | Complexity | Hours |
|----|------|------------|-------|
| P4-20 | Create `GpuLateInteractionModel` struct | Medium | 2 |
| P4-21 | Implement ColBERT forward pass | Complex | 4 |
| P4-22 | Implement `maxsim_score_gpu()` | Medium | 2 |
| P4-23 | Implement batched MaxSim | Complex | 4 |
| P4-24 | Integration tests | Medium | 2 |

---

## Phase 5: Integration and Optimization
**Estimated: 16-24 hours**

### 5.1 Feature Flag Cleanup

**Tasks:**
| ID | Task | Complexity | Hours |
|----|------|------------|-------|
| P5-01 | Remove all `unimplemented!()` stubs | Simple | 2 |
| P5-02 | Update `#[cfg(feature = "candle")]` guards | Simple | 2 |
| P5-03 | Change default feature to `cuda` | Simple | 1 |
| P5-04 | Update documentation | Simple | 2 |

### 5.2 Memory Optimization

**Tasks:**
| ID | Task | Complexity | Hours |
|----|------|------------|-------|
| P5-05 | Implement tensor memory pooling | Complex | 4 |
| P5-06 | Add GPU memory monitoring | Medium | 2 |
| P5-07 | Implement batch size auto-tuning | Medium | 3 |

### 5.3 Performance Benchmarks

**File:** `benches/gpu_benchmarks.rs`

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_fusion_layer(c: &mut Criterion) {
    let device = Device::new_cuda(0).unwrap();
    let config = FusionConfig::default();

    // Create GPU components
    let gating = GpuGatingNetwork::new(&config, &device).unwrap();
    let experts = GpuExpertPool::new(&config, &device).unwrap();

    // Generate test input
    let input = Tensor::randn(0.0, 1.0, (32, 8320), &device).unwrap();

    c.bench_function("fusion_forward_gpu", |b| {
        b.iter(|| {
            let (indices, weights) = gating.forward_topk(&input, 4).unwrap();
            experts.forward_topk(&input, &indices, &weights).unwrap()
        })
    });
}

criterion_group!(benches, benchmark_fusion_layer);
criterion_main!(benches);
```

**Tasks:**
| ID | Task | Complexity | Hours |
|----|------|------------|-------|
| P5-08 | Create benchmark suite | Medium | 3 |
| P5-09 | Benchmark fusion layer CPU vs GPU | Simple | 1 |
| P5-10 | Benchmark pretrained models | Simple | 1 |
| P5-11 | Document performance results | Simple | 2 |

### 5.4 CI/CD Updates

**Tasks:**
| ID | Task | Complexity | Hours |
|----|------|------------|-------|
| P5-12 | Add CUDA build to CI | Medium | 2 |
| P5-13 | Add GPU test runner | Complex | 3 |
| P5-14 | Update release workflow | Simple | 1 |

---

## Migration Verification Checklist

### Per-Phase Verification

#### Phase 1 Completion Criteria
- [ ] `cargo check -p context-graph-embeddings --features cuda` passes
- [ ] GPU device initialization works
- [ ] Core types convert to/from tensors

#### Phase 2 Completion Criteria
- [ ] All simple operations have GPU implementations
- [ ] Unit tests pass for GPU operations
- [ ] Output values match CPU within f32 epsilon

#### Phase 3 Completion Criteria
- [ ] Fusion layer tests pass on GPU
- [ ] **Test time reduced from 60+ seconds to <1 second**
- [ ] Memory usage within 8GB for batch_size=32

#### Phase 4 Completion Criteria
- [ ] All pretrained models load on GPU
- [ ] Inference produces same embeddings as CPU (within epsilon)
- [ ] No `unimplemented!()` stubs remain

#### Phase 5 Completion Criteria
- [ ] `default = ["cuda"]` set in Cargo.toml
- [ ] All benchmarks documented
- [ ] CI passes with GPU tests

---

## Expected Performance Improvements

| Component | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Linear 8320→4096 | ~200ms | ~4ms | 50x |
| LayerNorm 8320 | ~10ms | ~0.2ms | 50x |
| Expert FFN (1 expert) | ~500ms | ~8ms | 60x |
| ExpertPool top-4 | ~2s | ~25ms | 80x |
| GatingNetwork | ~300ms | ~5ms | 60x |
| **Full FuseMoE forward** | **60+ sec** | **<1 sec** | **60-100x** |
| SemanticModel embed | ~100ms | ~5ms | 20x |
| SparseModel embed | ~150ms | ~8ms | 19x |
| MaxSim scoring | ~500ms | ~10ms | 50x |

---

## Risk Mitigation

### Risk 1: CUDA Version Compatibility
- **Mitigation:** Pin cudarc 0.17.8+, test on CUDA 13.1 specifically
- **Fallback:** Use `--features stub` for CPU-only mode

### Risk 2: Memory Overflow for Large Batches
- **Mitigation:** Implement batch size auto-tuning based on available VRAM
- **Fallback:** Reduce batch size dynamically

### Risk 3: Precision Differences CPU vs GPU
- **Mitigation:** Use F32 dtype, validate against CPU outputs
- **Acceptable:** Differences < 1e-5 for normalized embeddings

### Risk 4: cuDNN/cuBLAS Availability
- **Mitigation:** Candle handles library loading
- **Fallback:** Use pure CUDA kernels if cuDNN unavailable

---

## Dependencies and Prerequisites

### Required Crates
```toml
[dependencies]
candle-core = { version = "0.9.2-alpha", features = ["cuda"] }
candle-nn = { version = "0.9.2-alpha", features = ["cuda"] }
candle-transformers = { version = "0.9.2-alpha", features = ["cuda"] }
tokenizers = "0.15"
safetensors = "0.4"
```

### System Requirements
- NVIDIA Driver 591.44+
- CUDA Toolkit 13.1
- cuDNN 8.9+
- 32GB+ VRAM (RTX 5090)
- Rust 1.75+

### Environment Variables
```bash
export CUDA_PATH=/usr/local/cuda-13.1
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

---

## Task Summary

| Phase | Tasks | Hours (Est.) | Priority |
|-------|-------|--------------|----------|
| Phase 0: Foundation Setup | 7 | DONE | Critical |
| Phase 1: Core Infrastructure | 13 | 16-24 | Critical |
| Phase 2: Core Operations | 13 | 24-32 | High |
| Phase 3: Fusion Layer | 26 | 32-48 | **HIGHEST** |
| Phase 4: Pretrained Models | 24 | 40-56 | High |
| Phase 5: Integration | 14 | 16-24 | Medium |
| **Total** | **97** | **128-184** | - |

---

## Quick Start for Implementer

```bash
# 1. Ensure CUDA environment
export CUDA_PATH=/usr/local/cuda-13.1
nvcc --version  # Should show 13.1

# 2. Build with CUDA
cargo build -p context-graph-embeddings --features cuda

# 3. Run GPU tests
cargo test -p context-graph-embeddings --features cuda -- --nocapture

# 4. Benchmark (after implementation)
cargo bench -p context-graph-embeddings --features cuda
```

---

*Document Version: 1.0.0*
*Created: 2026-01-01*
*Author: Claude Code (Opus 4.5)*
*Hardware Target: RTX 5090 32GB + CUDA 13.1*
