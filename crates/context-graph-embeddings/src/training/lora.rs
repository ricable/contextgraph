//! LoRA (Low-Rank Adaptation) adapters for NomicBERT attention.
//!
//! Rank-16 LoRA on Q, V attention projections:
//! - ~2.4M additional trainable parameters
//! - Teaches encoder to produce better causal representations before projection
//! - Phase 2b: only after projection training (Phase 2a) proves the approach
//!
//! # Architecture
//!
//! For each attention layer:
//! ```text
//! x → [W_q + B_q × A_q] → query   (original + low-rank update)
//! x → [W_v + B_v × A_v] → value   (original + low-rank update)
//! ```
//!
//! Where A is [hidden_size, rank] and B is [rank, hidden_size].
//! Total params per layer: 2 × (hidden_size × rank + rank × hidden_size)
//! = 2 × 2 × 768 × 16 = 98,304 per layer
//! × 12 layers = 1,179,648 total LoRA params

use candle_core::{DType, Device, Tensor, Var};

use crate::error::{EmbeddingError, EmbeddingResult};

/// LoRA adapter configuration.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Rank of the low-rank decomposition (default: 16).
    pub rank: usize,
    /// Scaling factor: alpha / rank (default: alpha=16, so scale=1.0).
    pub alpha: f32,
    /// Dropout rate for LoRA (default: 0.1).
    pub dropout: f32,
    /// Hidden size of the base model (default: 768 for NomicBERT).
    pub hidden_size: usize,
    /// Number of encoder layers (default: 12 for NomicBERT).
    pub num_layers: usize,
    /// Whether to apply LoRA to query projections.
    pub apply_query: bool,
    /// Whether to apply LoRA to value projections.
    pub apply_value: bool,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 16.0,
            dropout: 0.1,
            hidden_size: 768,
            num_layers: 12,
            apply_query: true,
            apply_value: true,
        }
    }
}

impl LoraConfig {
    /// Compute the scaling factor.
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Total number of LoRA parameters.
    pub fn total_params(&self) -> usize {
        let per_adapter = 2 * self.hidden_size * self.rank; // A + B
        let adapters_per_layer =
            self.apply_query as usize + self.apply_value as usize;
        per_adapter * adapters_per_layer * self.num_layers
    }
}

/// A single LoRA adapter pair (A down-projection + B up-projection).
pub struct LoraAdapter {
    /// Down-projection: [hidden_size, rank] (initialized with Kaiming)
    pub a: Var,
    /// Up-projection: [rank, hidden_size] (initialized to zero)
    pub b: Var,
    /// Scaling factor
    scale: f32,
}

impl LoraAdapter {
    /// Create a new LoRA adapter.
    ///
    /// A is initialized with Kaiming uniform, B is initialized to zero.
    /// This ensures the LoRA contribution starts at zero (identity behavior).
    pub fn new(
        hidden_size: usize,
        rank: usize,
        scale: f32,
        device: &Device,
    ) -> EmbeddingResult<Self> {
        // A: Kaiming uniform initialization for stable gradients
        let std_dev = (2.0 / hidden_size as f64).sqrt() as f32;
        let a_data: Vec<f32> = (0..hidden_size * rank)
            .map(|i| {
                // Deterministic pseudo-random based on position
                let x = ((i as f32 * 0.618033988 + 0.31415926) % 1.0) * 2.0 - 1.0;
                x * std_dev
            })
            .collect();

        let a_tensor = Tensor::from_slice(&a_data, (hidden_size, rank), device)
            .map_err(map_candle)?;
        let a = Var::from_tensor(&a_tensor).map_err(map_candle)?;

        // B: initialized to zero (LoRA starts as identity)
        let b_tensor =
            Tensor::zeros((rank, hidden_size), DType::F32, device).map_err(map_candle)?;
        let b = Var::from_tensor(&b_tensor).map_err(map_candle)?;

        Ok(Self { a, b, scale })
    }

    /// Apply LoRA: output = scale * (x @ A @ B)
    pub fn forward(&self, x: &Tensor) -> EmbeddingResult<Tensor> {
        let low_rank = x
            .matmul(self.a.as_tensor())
            .map_err(map_candle)?
            .matmul(self.b.as_tensor())
            .map_err(map_candle)?;

        low_rank
            .affine(self.scale as f64, 0.0)
            .map_err(map_candle)
    }

    /// Get trainable variables for optimizer registration.
    pub fn trainable_vars(&self) -> Vec<&Var> {
        vec![&self.a, &self.b]
    }

    /// Total number of parameters.
    pub fn num_params(&self) -> usize {
        let a_shape = self.a.as_tensor().shape();
        let b_shape = self.b.as_tensor().shape();
        a_shape.elem_count() + b_shape.elem_count()
    }
}

/// Collection of LoRA adapters for all encoder layers.
pub struct LoraLayers {
    /// Per-layer LoRA adapters for query projections.
    pub query_adapters: Vec<LoraAdapter>,
    /// Per-layer LoRA adapters for value projections.
    pub value_adapters: Vec<LoraAdapter>,
    /// Configuration.
    pub config: LoraConfig,
}

impl LoraLayers {
    /// Create LoRA adapters for all layers.
    pub fn new(config: LoraConfig, device: &Device) -> EmbeddingResult<Self> {
        let scale = config.scale();

        let mut query_adapters = Vec::new();
        let mut value_adapters = Vec::new();

        for _layer in 0..config.num_layers {
            if config.apply_query {
                query_adapters.push(LoraAdapter::new(
                    config.hidden_size,
                    config.rank,
                    scale,
                    device,
                )?);
            }
            if config.apply_value {
                value_adapters.push(LoraAdapter::new(
                    config.hidden_size,
                    config.rank,
                    scale,
                    device,
                )?);
            }
        }

        Ok(Self {
            query_adapters,
            value_adapters,
            config,
        })
    }

    /// Get all trainable variables across all layers.
    pub fn all_trainable_vars(&self) -> Vec<&Var> {
        let mut vars = Vec::new();
        for adapter in &self.query_adapters {
            vars.extend(adapter.trainable_vars());
        }
        for adapter in &self.value_adapters {
            vars.extend(adapter.trainable_vars());
        }
        vars
    }

    /// Total number of LoRA parameters.
    pub fn total_params(&self) -> usize {
        self.query_adapters
            .iter()
            .chain(self.value_adapters.iter())
            .map(|a| a.num_params())
            .sum()
    }

    /// Apply LoRA to a query projection output at a given layer.
    pub fn apply_query(&self, layer: usize, x: &Tensor) -> EmbeddingResult<Tensor> {
        if layer < self.query_adapters.len() {
            self.query_adapters[layer].forward(x)
        } else {
            Tensor::zeros_like(x).map_err(map_candle)
        }
    }

    /// Apply LoRA to a value projection output at a given layer.
    pub fn apply_value(&self, layer: usize, x: &Tensor) -> EmbeddingResult<Tensor> {
        if layer < self.value_adapters.len() {
            self.value_adapters[layer].forward(x)
        } else {
            Tensor::zeros_like(x).map_err(map_candle)
        }
    }
}

impl LoraLayers {
    /// Load LoRA adapters from a safetensors checkpoint.
    ///
    /// Expects tensor names like `lora.query.{layer}.a`, `lora.query.{layer}.b`,
    /// `lora.value.{layer}.a`, `lora.value.{layer}.b` — matching the save format
    /// in `CausalTrainingPipeline::save_lora()`.
    pub fn load_from_safetensors(
        path: &std::path::Path,
        config: LoraConfig,
        device: &Device,
    ) -> EmbeddingResult<Self> {
        let data = std::fs::read(path).map_err(|e| EmbeddingError::InternalError {
            message: format!("Failed to read LoRA checkpoint: {}", e),
        })?;

        let safetensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| EmbeddingError::InternalError {
                message: format!("Failed to deserialize LoRA checkpoint: {}", e),
            })?;

        let scale = config.scale();

        let load_tensor = |name: &str| -> EmbeddingResult<Tensor> {
            let view = safetensors.tensor(name).map_err(|e| {
                EmbeddingError::InternalError {
                    message: format!("Missing LoRA tensor '{}': {}", name, e),
                }
            })?;
            let shape: Vec<usize> = view.shape().to_vec();
            let float_data: &[f32] = bytemuck::cast_slice(view.data());
            Tensor::from_slice(float_data, shape, device).map_err(|e| {
                EmbeddingError::GpuError {
                    message: format!("Failed to create LoRA tensor '{}': {}", name, e),
                }
            })
        };

        let mut query_adapters = Vec::new();
        let mut value_adapters = Vec::new();

        for i in 0..config.num_layers {
            if config.apply_query {
                let a_tensor = load_tensor(&format!("lora.query.{}.a", i))?;
                let b_tensor = load_tensor(&format!("lora.query.{}.b", i))?;
                let a = Var::from_tensor(&a_tensor).map_err(map_candle)?;
                let b = Var::from_tensor(&b_tensor).map_err(map_candle)?;
                query_adapters.push(LoraAdapter { a, b, scale });
            }
            if config.apply_value {
                let a_tensor = load_tensor(&format!("lora.value.{}.a", i))?;
                let b_tensor = load_tensor(&format!("lora.value.{}.b", i))?;
                let a = Var::from_tensor(&a_tensor).map_err(map_candle)?;
                let b = Var::from_tensor(&b_tensor).map_err(map_candle)?;
                value_adapters.push(LoraAdapter { a, b, scale });
            }
        }

        tracing::info!(
            "Loaded LoRA weights from {}: {} query + {} value adapters, {} total params",
            path.display(),
            query_adapters.len(),
            value_adapters.len(),
            query_adapters.iter().chain(value_adapters.iter()).map(|a| a.num_params()).sum::<usize>(),
        );

        Ok(Self {
            query_adapters,
            value_adapters,
            config,
        })
    }
}

/// Map candle errors to EmbeddingError.
fn map_candle(e: candle_core::Error) -> EmbeddingError {
    EmbeddingError::GpuError {
        message: format!("LoRA error: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_defaults() {
        let config = LoraConfig::default();
        assert_eq!(config.rank, 16);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 12);
        assert!((config.scale() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lora_param_count() {
        let config = LoraConfig::default();
        // 2 adapters (Q, V) × 12 layers × 2 × 768 × 16
        let expected = 2 * 12 * 2 * 768 * 16;
        assert_eq!(config.total_params(), expected);
    }

    #[test]
    fn test_lora_adapter_zero_init() {
        let device = Device::Cpu;
        let adapter = LoraAdapter::new(8, 4, 1.0, &device).unwrap();

        // B initialized to zero → LoRA output should be zero
        let x = Tensor::ones((2, 8), DType::F32, &device).unwrap();
        let output = adapter.forward(&x).unwrap();
        let vals: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();

        for v in vals {
            assert!(
                v.abs() < 1e-6,
                "LoRA output should be ~0 at init (B=0), got {}",
                v
            );
        }
    }

    #[test]
    fn test_lora_layers_creation() {
        let config = LoraConfig {
            num_layers: 2,
            hidden_size: 8,
            rank: 4,
            ..Default::default()
        };
        let layers = LoraLayers::new(config, &Device::Cpu).unwrap();

        assert_eq!(layers.query_adapters.len(), 2);
        assert_eq!(layers.value_adapters.len(), 2);
        assert_eq!(layers.all_trainable_vars().len(), 8); // 2 vars × 2 adapters × 2 layers
    }

    #[test]
    fn test_lora_forward_shape() {
        let device = Device::Cpu;
        let adapter = LoraAdapter::new(8, 4, 1.0, &device).unwrap();

        let x = Tensor::ones((3, 8), DType::F32, &device).unwrap();
        let output = adapter.forward(&x).unwrap();

        assert_eq!(output.dims(), &[3, 8]); // Same shape as input
    }
}
