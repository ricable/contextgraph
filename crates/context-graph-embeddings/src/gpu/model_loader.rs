//! GPU ModelLoader for loading pretrained BERT models from safetensors.
//!
//! # Architecture
//!
//! This module provides GPU-accelerated model loading via Candle's VarBuilder.
//! It loads safetensors files from local model directories and constructs
//! complete BERT architecture components for embedding generation.
//!
//! # Supported Architectures
//!
//! | Model Type | Architecture | Example |
//! |------------|--------------|---------|
//! | BERT | BertModel | e5-large-v2, all-MiniLM-L6-v2 |
//! | MPNet | MPNetModel | all-mpnet-base-v2 |
//!
//! # Usage
//!
//! ```rust,ignore
//! use context_graph_embeddings::gpu::{GpuModelLoader, BertWeights};
//!
//! let loader = GpuModelLoader::new()?;
//! let weights = loader.load_bert_weights(Path::new("/models/semantic"))?;
//! // Use weights for inference
//! ```

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use std::path::Path;
use thiserror::Error;

use super::init_gpu;

/// Error type for model loading operations.
///
/// Provides detailed context for debugging loading failures:
/// - Model path that failed
/// - Specific layer/weight that failed
/// - Expected vs actual tensor shapes
#[derive(Debug, Error)]
pub enum ModelLoadError {
    /// GPU initialization failed.
    #[error("GPU initialization failed: {message}")]
    GpuInitError { message: String },

    /// Model directory does not exist or is not accessible.
    #[error("Model directory not found: {path}")]
    ModelDirectoryNotFound { path: String },

    /// config.json file missing or unreadable.
    #[error("Config file not found at {path}: {source}")]
    ConfigNotFound {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// config.json parsing failed.
    #[error("Config parse error for {path}: {message}")]
    ConfigParseError { path: String, message: String },

    /// model.safetensors file missing.
    #[error("Safetensors file not found at {path}")]
    SafetensorsNotFound { path: String },

    /// Safetensors file loading failed.
    #[error("Failed to load safetensors from {path}: {message}")]
    SafetensorsLoadError { path: String, message: String },

    /// Specific weight tensor not found in safetensors.
    #[error("Weight not found: {weight_name} in {model_path}")]
    WeightNotFound { weight_name: String, model_path: String },

    /// Weight tensor has unexpected shape.
    #[error("Shape mismatch for {weight_name}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        weight_name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Candle tensor operation failed.
    #[error("Tensor operation failed for {operation}: {message}")]
    TensorError { operation: String, message: String },

    /// Unsupported model architecture.
    #[error("Unsupported architecture: {architecture} (supported: BERT, MPNet)")]
    UnsupportedArchitecture { architecture: String },
}

impl From<candle_core::Error> for ModelLoadError {
    fn from(err: candle_core::Error) -> Self {
        ModelLoadError::TensorError {
            operation: "candle".to_string(),
            message: err.to_string(),
        }
    }
}

/// BERT model configuration parsed from config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct BertConfig {
    /// Vocabulary size (e.g., 30522 for BERT).
    pub vocab_size: usize,
    /// Hidden layer size (e.g., 768 for BERT-base, 1024 for BERT-large).
    pub hidden_size: usize,
    /// Number of hidden layers (e.g., 12 for BERT-base, 24 for BERT-large).
    pub num_hidden_layers: usize,
    /// Number of attention heads (e.g., 12 for BERT-base, 16 for BERT-large).
    pub num_attention_heads: usize,
    /// Intermediate FFN size (usually 4x hidden_size).
    pub intermediate_size: usize,
    /// Hidden activation function (gelu, relu, etc.).
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// Dropout probability for hidden layers.
    #[serde(default = "default_dropout")]
    pub hidden_dropout_prob: f64,
    /// Dropout probability for attention.
    #[serde(default = "default_dropout")]
    pub attention_probs_dropout_prob: f64,
    /// Maximum sequence length.
    #[serde(default = "default_max_position")]
    pub max_position_embeddings: usize,
    /// Token type vocabulary size (usually 2).
    #[serde(default = "default_type_vocab")]
    pub type_vocab_size: usize,
    /// Layer normalization epsilon.
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    /// Padding token ID.
    #[serde(default)]
    pub pad_token_id: usize,
    /// Model type string (bert, mpnet, etc.).
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Architecture list.
    #[serde(default)]
    pub architectures: Vec<String>,
}

fn default_hidden_act() -> String {
    "gelu".to_string()
}

fn default_dropout() -> f64 {
    0.1
}

fn default_max_position() -> usize {
    512
}

fn default_type_vocab() -> usize {
    2
}

fn default_layer_norm_eps() -> f64 {
    1e-12
}

fn default_model_type() -> String {
    "bert".to_string()
}

/// Embedding weights (word, position, token_type, LayerNorm).
#[derive(Debug)]
pub struct EmbeddingWeights {
    /// Word embeddings: [vocab_size, hidden_size]
    pub word_embeddings: Tensor,
    /// Position embeddings: [max_position, hidden_size]
    pub position_embeddings: Tensor,
    /// Token type embeddings: [type_vocab_size, hidden_size]
    pub token_type_embeddings: Tensor,
    /// LayerNorm weight: [hidden_size]
    pub layer_norm_weight: Tensor,
    /// LayerNorm bias: [hidden_size]
    pub layer_norm_bias: Tensor,
}

/// Self-attention weights for a single layer.
#[derive(Debug)]
pub struct AttentionWeights {
    /// Query projection: [hidden_size, hidden_size]
    pub query_weight: Tensor,
    /// Query bias: [hidden_size]
    pub query_bias: Tensor,
    /// Key projection: [hidden_size, hidden_size]
    pub key_weight: Tensor,
    /// Key bias: [hidden_size]
    pub key_bias: Tensor,
    /// Value projection: [hidden_size, hidden_size]
    pub value_weight: Tensor,
    /// Value bias: [hidden_size]
    pub value_bias: Tensor,
    /// Output projection: [hidden_size, hidden_size]
    pub output_weight: Tensor,
    /// Output bias: [hidden_size]
    pub output_bias: Tensor,
    /// Attention output LayerNorm weight: [hidden_size]
    pub layer_norm_weight: Tensor,
    /// Attention output LayerNorm bias: [hidden_size]
    pub layer_norm_bias: Tensor,
}

/// Feed-forward network weights for a single layer.
#[derive(Debug)]
pub struct FfnWeights {
    /// Intermediate (up) projection: [hidden_size, intermediate_size]
    pub intermediate_weight: Tensor,
    /// Intermediate bias: [intermediate_size]
    pub intermediate_bias: Tensor,
    /// Output (down) projection: [intermediate_size, hidden_size]
    pub output_weight: Tensor,
    /// Output bias: [hidden_size]
    pub output_bias: Tensor,
    /// Output LayerNorm weight: [hidden_size]
    pub layer_norm_weight: Tensor,
    /// Output LayerNorm bias: [hidden_size]
    pub layer_norm_bias: Tensor,
}

/// Complete weights for a single encoder layer.
#[derive(Debug)]
pub struct EncoderLayerWeights {
    /// Self-attention weights.
    pub attention: AttentionWeights,
    /// Feed-forward network weights.
    pub ffn: FfnWeights,
}

/// Pooler weights for [CLS] token projection.
#[derive(Debug)]
pub struct PoolerWeights {
    /// Dense projection: [hidden_size, hidden_size]
    pub dense_weight: Tensor,
    /// Dense bias: [hidden_size]
    pub dense_bias: Tensor,
}

/// Complete BERT model weights loaded from safetensors.
#[derive(Debug)]
pub struct BertWeights {
    /// Model configuration.
    pub config: BertConfig,
    /// Embedding layer weights.
    pub embeddings: EmbeddingWeights,
    /// Encoder layer weights (one per layer).
    pub encoder_layers: Vec<EncoderLayerWeights>,
    /// Pooler weights (optional, may not exist in some models).
    pub pooler: Option<PoolerWeights>,
    /// Device the weights are loaded on.
    device: &'static Device,
}

impl BertWeights {
    /// Get the device these weights are loaded on.
    pub fn device(&self) -> &'static Device {
        self.device
    }

    /// Get total parameter count.
    pub fn param_count(&self) -> usize {
        let embedding_params = self.embeddings.word_embeddings.elem_count()
            + self.embeddings.position_embeddings.elem_count()
            + self.embeddings.token_type_embeddings.elem_count()
            + self.embeddings.layer_norm_weight.elem_count()
            + self.embeddings.layer_norm_bias.elem_count();

        let layer_params: usize = self
            .encoder_layers
            .iter()
            .map(|layer| {
                layer.attention.query_weight.elem_count()
                    + layer.attention.query_bias.elem_count()
                    + layer.attention.key_weight.elem_count()
                    + layer.attention.key_bias.elem_count()
                    + layer.attention.value_weight.elem_count()
                    + layer.attention.value_bias.elem_count()
                    + layer.attention.output_weight.elem_count()
                    + layer.attention.output_bias.elem_count()
                    + layer.attention.layer_norm_weight.elem_count()
                    + layer.attention.layer_norm_bias.elem_count()
                    + layer.ffn.intermediate_weight.elem_count()
                    + layer.ffn.intermediate_bias.elem_count()
                    + layer.ffn.output_weight.elem_count()
                    + layer.ffn.output_bias.elem_count()
                    + layer.ffn.layer_norm_weight.elem_count()
                    + layer.ffn.layer_norm_bias.elem_count()
            })
            .sum();

        let pooler_params = self
            .pooler
            .as_ref()
            .map(|p| p.dense_weight.elem_count() + p.dense_bias.elem_count())
            .unwrap_or(0);

        embedding_params + layer_params + pooler_params
    }

    /// Get estimated VRAM usage in bytes (F32).
    pub fn vram_bytes(&self) -> usize {
        self.param_count() * std::mem::size_of::<f32>()
    }
}

/// GPU model loader for loading pretrained models from safetensors.
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_embeddings::gpu::GpuModelLoader;
/// use std::path::Path;
///
/// let loader = GpuModelLoader::new()?;
/// let weights = loader.load_bert_weights(Path::new("/models/semantic"))?;
/// println!("Loaded {} parameters", weights.param_count());
/// ```
pub struct GpuModelLoader {
    /// Reference to the GPU device singleton.
    device: &'static Device,
    /// Default dtype for loading weights.
    dtype: DType,
}

impl GpuModelLoader {
    /// Create a new GPU model loader.
    ///
    /// Initializes the GPU device if not already initialized.
    ///
    /// # Errors
    ///
    /// Returns error if GPU initialization fails (no CUDA, no GPU hardware).
    pub fn new() -> Result<Self, ModelLoadError> {
        let device = init_gpu().map_err(|e| ModelLoadError::GpuInitError {
            message: e.to_string(),
        })?;

        Ok(Self {
            device,
            dtype: DType::F32,
        })
    }

    /// Create a loader with a specific dtype.
    pub fn with_dtype(dtype: DType) -> Result<Self, ModelLoadError> {
        let device = init_gpu().map_err(|e| ModelLoadError::GpuInitError {
            message: e.to_string(),
        })?;

        Ok(Self { device, dtype })
    }

    /// Get the GPU device.
    pub fn device(&self) -> &'static Device {
        self.device
    }

    /// Get the dtype used for loading.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Load BERT model configuration from config.json.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Path to model directory containing config.json
    ///
    /// # Errors
    ///
    /// Returns error if config.json is missing or malformed.
    pub fn load_config(&self, model_dir: &Path) -> Result<BertConfig, ModelLoadError> {
        let config_path = model_dir.join("config.json");

        if !config_path.exists() {
            return Err(ModelLoadError::ConfigNotFound {
                path: config_path.display().to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "config.json not found",
                ),
            });
        }

        let config_content =
            std::fs::read_to_string(&config_path).map_err(|e| ModelLoadError::ConfigNotFound {
                path: config_path.display().to_string(),
                source: e,
            })?;

        let config: BertConfig =
            serde_json::from_str(&config_content).map_err(|e| ModelLoadError::ConfigParseError {
                path: config_path.display().to_string(),
                message: e.to_string(),
            })?;

        // Validate architecture
        let is_bert = config.model_type == "bert"
            || config
                .architectures
                .iter()
                .any(|a| a.contains("Bert") || a.contains("bert"));
        let is_mpnet = config.model_type == "mpnet"
            || config
                .architectures
                .iter()
                .any(|a| a.contains("MPNet") || a.contains("mpnet"));

        if !is_bert && !is_mpnet {
            return Err(ModelLoadError::UnsupportedArchitecture {
                architecture: config.model_type.clone(),
            });
        }

        tracing::info!(
            "Loaded config for {} model: hidden_size={}, layers={}, heads={}",
            config.model_type,
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads
        );

        Ok(config)
    }

    /// Load complete BERT weights from safetensors.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Path to model directory containing:
    ///   - config.json
    ///   - model.safetensors
    ///
    /// # Returns
    ///
    /// Complete BertWeights struct with all layers loaded to GPU.
    ///
    /// # Errors
    ///
    /// Returns detailed error if:
    /// - Model directory doesn't exist
    /// - config.json is missing or malformed
    /// - model.safetensors is missing
    /// - Any weight tensor is missing
    /// - Any tensor has unexpected shape
    pub fn load_bert_weights(&self, model_dir: &Path) -> Result<BertWeights, ModelLoadError> {
        self.load_bert_weights_with_prefix(model_dir, "")
    }

    /// Load BERT weights with a custom weight key prefix.
    /// Some models (like SPLADE, ColBERT) use `bert.` as a prefix for all weights.
    ///
    /// # Arguments
    /// * `model_dir` - Path to model directory containing config.json and model.safetensors
    /// * `prefix` - Prefix for weight keys (e.g., "bert." for models with that prefix)
    pub fn load_bert_weights_with_prefix(
        &self,
        model_dir: &Path,
        prefix: &str,
    ) -> Result<BertWeights, ModelLoadError> {
        // Validate model directory exists
        if !model_dir.exists() {
            return Err(ModelLoadError::ModelDirectoryNotFound {
                path: model_dir.display().to_string(),
            });
        }

        // Load config
        let config = self.load_config(model_dir)?;

        // Check for safetensors file
        let safetensors_path = model_dir.join("model.safetensors");
        if !safetensors_path.exists() {
            return Err(ModelLoadError::SafetensorsNotFound {
                path: safetensors_path.display().to_string(),
            });
        }

        tracing::info!(
            "Loading safetensors from: {} (prefix: '{}')",
            safetensors_path.display(),
            prefix
        );

        // Create VarBuilder from safetensors
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&safetensors_path], self.dtype, self.device)
                .map_err(|e| ModelLoadError::SafetensorsLoadError {
                    path: safetensors_path.display().to_string(),
                    message: e.to_string(),
                })?
        };

        // Load embeddings
        let embeddings = self.load_embeddings_with_prefix(&vb, &config, model_dir, prefix)?;

        // Load encoder layers
        let mut encoder_layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let layer = self.load_encoder_layer_with_prefix(&vb, &config, layer_idx, model_dir, prefix)?;
            encoder_layers.push(layer);
        }

        // Try to load pooler (optional - some models don't have it)
        let pooler = self.load_pooler_with_prefix(&vb, &config, model_dir, prefix).ok();

        let weights = BertWeights {
            config,
            embeddings,
            encoder_layers,
            pooler,
            device: self.device,
        };

        tracing::info!(
            "Loaded BERT weights: {} parameters, {:.2} MB VRAM",
            weights.param_count(),
            weights.vram_bytes() as f64 / (1024.0 * 1024.0)
        );

        Ok(weights)
    }

    /// Load embedding layer weights.
    fn load_embeddings(
        &self,
        vb: &VarBuilder,
        config: &BertConfig,
        model_dir: &Path,
    ) -> Result<EmbeddingWeights, ModelLoadError> {
        self.load_embeddings_with_prefix(vb, config, model_dir, "")
    }

    /// Load embedding layer weights with optional model prefix.
    fn load_embeddings_with_prefix(
        &self,
        vb: &VarBuilder,
        config: &BertConfig,
        model_dir: &Path,
        model_prefix: &str,
    ) -> Result<EmbeddingWeights, ModelLoadError> {
        let prefix = if model_prefix.is_empty() {
            "embeddings".to_string()
        } else {
            format!("{}embeddings", model_prefix)
        };
        let model_path = model_dir.display().to_string();

        // Word embeddings: [vocab_size, hidden_size]
        let word_embeddings = self.get_tensor(
            vb,
            &format!("{}.word_embeddings.weight", prefix),
            &[config.vocab_size, config.hidden_size],
            &model_path,
        )?;

        // Position embeddings: [max_position_embeddings, hidden_size]
        let position_embeddings = self.get_tensor(
            vb,
            &format!("{}.position_embeddings.weight", prefix),
            &[config.max_position_embeddings, config.hidden_size],
            &model_path,
        )?;

        // Token type embeddings: [type_vocab_size, hidden_size]
        let token_type_embeddings = self.get_tensor(
            vb,
            &format!("{}.token_type_embeddings.weight", prefix),
            &[config.type_vocab_size, config.hidden_size],
            &model_path,
        )?;

        // LayerNorm
        let layer_norm_weight = self.get_tensor(
            vb,
            &format!("{}.LayerNorm.weight", prefix),
            &[config.hidden_size],
            &model_path,
        )?;
        let layer_norm_bias = self.get_tensor(
            vb,
            &format!("{}.LayerNorm.bias", prefix),
            &[config.hidden_size],
            &model_path,
        )?;

        Ok(EmbeddingWeights {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm_weight,
            layer_norm_bias,
        })
    }

    /// Load a single encoder layer.
    fn load_encoder_layer(
        &self,
        vb: &VarBuilder,
        config: &BertConfig,
        layer_idx: usize,
        model_dir: &Path,
    ) -> Result<EncoderLayerWeights, ModelLoadError> {
        self.load_encoder_layer_with_prefix(vb, config, layer_idx, model_dir, "")
    }

    /// Load a single encoder layer with optional model prefix.
    fn load_encoder_layer_with_prefix(
        &self,
        vb: &VarBuilder,
        config: &BertConfig,
        layer_idx: usize,
        model_dir: &Path,
        model_prefix: &str,
    ) -> Result<EncoderLayerWeights, ModelLoadError> {
        let attention = self.load_attention_weights_with_prefix(vb, config, layer_idx, model_dir, model_prefix)?;
        let ffn = self.load_ffn_weights_with_prefix(vb, config, layer_idx, model_dir, model_prefix)?;

        Ok(EncoderLayerWeights { attention, ffn })
    }

    /// Load self-attention weights for a layer.
    fn load_attention_weights(
        &self,
        vb: &VarBuilder,
        config: &BertConfig,
        layer_idx: usize,
        model_dir: &Path,
    ) -> Result<AttentionWeights, ModelLoadError> {
        self.load_attention_weights_with_prefix(vb, config, layer_idx, model_dir, "")
    }

    /// Load self-attention weights for a layer with optional model prefix.
    fn load_attention_weights_with_prefix(
        &self,
        vb: &VarBuilder,
        config: &BertConfig,
        layer_idx: usize,
        model_dir: &Path,
        model_prefix: &str,
    ) -> Result<AttentionWeights, ModelLoadError> {
        let prefix = format!("{}encoder.layer.{}.attention", model_prefix, layer_idx);
        let model_path = model_dir.display().to_string();

        // Self attention Q/K/V
        let query_weight = self.get_tensor(
            vb,
            &format!("{}.self.query.weight", prefix),
            &[config.hidden_size, config.hidden_size],
            &model_path,
        )?;
        let query_bias = self.get_tensor(
            vb,
            &format!("{}.self.query.bias", prefix),
            &[config.hidden_size],
            &model_path,
        )?;

        let key_weight = self.get_tensor(
            vb,
            &format!("{}.self.key.weight", prefix),
            &[config.hidden_size, config.hidden_size],
            &model_path,
        )?;
        let key_bias = self.get_tensor(
            vb,
            &format!("{}.self.key.bias", prefix),
            &[config.hidden_size],
            &model_path,
        )?;

        let value_weight = self.get_tensor(
            vb,
            &format!("{}.self.value.weight", prefix),
            &[config.hidden_size, config.hidden_size],
            &model_path,
        )?;
        let value_bias = self.get_tensor(
            vb,
            &format!("{}.self.value.bias", prefix),
            &[config.hidden_size],
            &model_path,
        )?;

        // Output projection
        let output_weight = self.get_tensor(
            vb,
            &format!("{}.output.dense.weight", prefix),
            &[config.hidden_size, config.hidden_size],
            &model_path,
        )?;
        let output_bias = self.get_tensor(
            vb,
            &format!("{}.output.dense.bias", prefix),
            &[config.hidden_size],
            &model_path,
        )?;

        // LayerNorm
        let layer_norm_weight = self.get_tensor(
            vb,
            &format!("{}.output.LayerNorm.weight", prefix),
            &[config.hidden_size],
            &model_path,
        )?;
        let layer_norm_bias = self.get_tensor(
            vb,
            &format!("{}.output.LayerNorm.bias", prefix),
            &[config.hidden_size],
            &model_path,
        )?;

        Ok(AttentionWeights {
            query_weight,
            query_bias,
            key_weight,
            key_bias,
            value_weight,
            value_bias,
            output_weight,
            output_bias,
            layer_norm_weight,
            layer_norm_bias,
        })
    }

    /// Load feed-forward network weights for a layer.
    fn load_ffn_weights(
        &self,
        vb: &VarBuilder,
        config: &BertConfig,
        layer_idx: usize,
        model_dir: &Path,
    ) -> Result<FfnWeights, ModelLoadError> {
        self.load_ffn_weights_with_prefix(vb, config, layer_idx, model_dir, "")
    }

    /// Load feed-forward network weights for a layer with optional model prefix.
    fn load_ffn_weights_with_prefix(
        &self,
        vb: &VarBuilder,
        config: &BertConfig,
        layer_idx: usize,
        model_dir: &Path,
        model_prefix: &str,
    ) -> Result<FfnWeights, ModelLoadError> {
        let prefix = format!("{}encoder.layer.{}", model_prefix, layer_idx);
        let model_path = model_dir.display().to_string();

        // Intermediate (up projection)
        let intermediate_weight = self.get_tensor(
            vb,
            &format!("{}.intermediate.dense.weight", prefix),
            &[config.intermediate_size, config.hidden_size],
            &model_path,
        )?;
        let intermediate_bias = self.get_tensor(
            vb,
            &format!("{}.intermediate.dense.bias", prefix),
            &[config.intermediate_size],
            &model_path,
        )?;

        // Output (down projection)
        let output_weight = self.get_tensor(
            vb,
            &format!("{}.output.dense.weight", prefix),
            &[config.hidden_size, config.intermediate_size],
            &model_path,
        )?;
        let output_bias = self.get_tensor(
            vb,
            &format!("{}.output.dense.bias", prefix),
            &[config.hidden_size],
            &model_path,
        )?;

        // LayerNorm
        let layer_norm_weight = self.get_tensor(
            vb,
            &format!("{}.output.LayerNorm.weight", prefix),
            &[config.hidden_size],
            &model_path,
        )?;
        let layer_norm_bias = self.get_tensor(
            vb,
            &format!("{}.output.LayerNorm.bias", prefix),
            &[config.hidden_size],
            &model_path,
        )?;

        Ok(FfnWeights {
            intermediate_weight,
            intermediate_bias,
            output_weight,
            output_bias,
            layer_norm_weight,
            layer_norm_bias,
        })
    }

    /// Load pooler weights.
    fn load_pooler(
        &self,
        vb: &VarBuilder,
        config: &BertConfig,
        model_dir: &Path,
    ) -> Result<PoolerWeights, ModelLoadError> {
        self.load_pooler_with_prefix(vb, config, model_dir, "")
    }

    /// Load pooler weights with optional model prefix.
    fn load_pooler_with_prefix(
        &self,
        vb: &VarBuilder,
        config: &BertConfig,
        model_dir: &Path,
        model_prefix: &str,
    ) -> Result<PoolerWeights, ModelLoadError> {
        let model_path = model_dir.display().to_string();

        let dense_weight = self.get_tensor(
            vb,
            &format!("{}pooler.dense.weight", model_prefix),
            &[config.hidden_size, config.hidden_size],
            &model_path,
        )?;
        let dense_bias = self.get_tensor(
            vb,
            &format!("{}pooler.dense.bias", model_prefix),
            &[config.hidden_size],
            &model_path,
        )?;

        Ok(PoolerWeights {
            dense_weight,
            dense_bias,
        })
    }

    /// Get a tensor from VarBuilder with shape validation.
    fn get_tensor(
        &self,
        vb: &VarBuilder,
        name: &str,
        expected_shape: &[usize],
        model_path: &str,
    ) -> Result<Tensor, ModelLoadError> {
        let tensor = vb.get(expected_shape, name).map_err(|e| {
            // Check if it's a shape error or missing tensor
            let err_str = e.to_string();
            if err_str.contains("shape") || err_str.contains("Shape") {
                ModelLoadError::ShapeMismatch {
                    weight_name: name.to_string(),
                    expected: expected_shape.to_vec(),
                    actual: vec![], // We don't have access to actual shape in error
                }
            } else {
                ModelLoadError::WeightNotFound {
                    weight_name: name.to_string(),
                    model_path: model_path.to_string(),
                }
            }
        })?;

        // Verify shape matches exactly
        let actual_shape: Vec<usize> = tensor.dims().to_vec();
        if actual_shape != expected_shape {
            return Err(ModelLoadError::ShapeMismatch {
                weight_name: name.to_string(),
                expected: expected_shape.to_vec(),
                actual: actual_shape,
            });
        }

        Ok(tensor)
    }

    /// Load multiple models from a models config.
    ///
    /// # Arguments
    ///
    /// * `models_config_path` - Path to models_config.toml
    ///
    /// # Returns
    ///
    /// Map of model name to loaded BertWeights.
    pub fn load_models_from_config(
        &self,
        models_config_path: &Path,
    ) -> Result<std::collections::HashMap<String, BertWeights>, ModelLoadError> {
        let content = std::fs::read_to_string(models_config_path).map_err(|e| {
            ModelLoadError::ConfigNotFound {
                path: models_config_path.display().to_string(),
                source: e,
            }
        })?;

        #[derive(Deserialize)]
        struct ModelsConfig {
            models: std::collections::HashMap<String, ModelEntry>,
        }

        #[derive(Deserialize)]
        struct ModelEntry {
            path: String,
            #[allow(dead_code)]
            repo: String,
        }

        let config: ModelsConfig =
            toml::from_str(&content).map_err(|e| ModelLoadError::ConfigParseError {
                path: models_config_path.display().to_string(),
                message: e.to_string(),
            })?;

        let mut loaded = std::collections::HashMap::new();

        for (name, entry) in config.models {
            let model_dir = Path::new(&entry.path);

            // Check if model.safetensors exists
            let safetensors_path = model_dir.join("model.safetensors");
            if !safetensors_path.exists() {
                tracing::warn!(
                    "Skipping model '{}': no model.safetensors at {}",
                    name,
                    safetensors_path.display()
                );
                continue;
            }

            match self.load_bert_weights(model_dir) {
                Ok(weights) => {
                    tracing::info!(
                        "Loaded model '{}': {} params",
                        name,
                        weights.param_count()
                    );
                    loaded.insert(name, weights);
                }
                Err(e) => {
                    tracing::warn!("Failed to load model '{}': {}", name, e);
                }
            }
        }

        Ok(loaded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_load_error_display() {
        let err = ModelLoadError::WeightNotFound {
            weight_name: "encoder.layer.0.attention.self.query.weight".to_string(),
            model_path: "/models/semantic".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("encoder.layer.0.attention.self.query.weight"));
        assert!(msg.contains("/models/semantic"));
    }

    #[test]
    fn test_shape_mismatch_error() {
        let err = ModelLoadError::ShapeMismatch {
            weight_name: "embeddings.word_embeddings.weight".to_string(),
            expected: vec![30522, 768],
            actual: vec![30522, 1024],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("30522"));
        assert!(msg.contains("768"));
        assert!(msg.contains("1024"));
    }

    #[test]
    fn test_bert_config_defaults() {
        let json = r#"{
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072
        }"#;

        let config: BertConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_act, "gelu");
        assert_eq!(config.max_position_embeddings, 512);
        assert_eq!(config.type_vocab_size, 2);
        assert!((config.layer_norm_eps - 1e-12).abs() < 1e-15);
    }

    #[test]
    fn test_bert_config_full() {
        let json = r#"{
            "vocab_size": 30522,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "layer_norm_eps": 1e-12,
            "pad_token_id": 0,
            "model_type": "bert",
            "architectures": ["BertModel"]
        }"#;

        let config: BertConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.vocab_size, 30522);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.intermediate_size, 4096);
        assert_eq!(config.model_type, "bert");
        assert!(config.architectures.contains(&"BertModel".to_string()));
    }

    #[test]
    fn test_gpu_init_error_conversion() {
        let err = candle_core::Error::Msg("CUDA not available".to_string());
        let load_err: ModelLoadError = err.into();
        match load_err {
            ModelLoadError::TensorError { operation, message } => {
                assert_eq!(operation, "candle");
                assert!(message.contains("CUDA"));
            }
            _ => panic!("Expected TensorError"),
        }
    }

    // Integration tests require GPU - skip in CI
    #[test]
    #[ignore = "Requires GPU"]
    fn test_load_semantic_model() {
        let loader = GpuModelLoader::new().expect("GPU init failed");
        let model_dir = Path::new("/home/cabdru/contextgraph/models/semantic");
        let weights = loader.load_bert_weights(model_dir).expect("Load failed");

        // e5-large-v2 specs
        assert_eq!(weights.config.hidden_size, 1024);
        assert_eq!(weights.config.num_hidden_layers, 24);
        assert_eq!(weights.config.num_attention_heads, 16);
        assert!(weights.param_count() > 300_000_000); // ~335M params
    }

    #[test]
    #[ignore = "Requires GPU"]
    fn test_load_entity_model() {
        let loader = GpuModelLoader::new().expect("GPU init failed");
        let model_dir = Path::new("/home/cabdru/contextgraph/models/entity");
        let weights = loader.load_bert_weights(model_dir).expect("Load failed");

        // all-MiniLM-L6-v2 specs
        assert_eq!(weights.config.hidden_size, 384);
        assert_eq!(weights.config.num_hidden_layers, 6);
        assert_eq!(weights.config.num_attention_heads, 12);
    }
}
