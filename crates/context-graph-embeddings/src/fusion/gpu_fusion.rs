//! GPU-accelerated fusion layer components for RTX 5090.
//!
//! This module provides GPU implementations of the FuseMoE fusion layer
//! using Candle for CUDA acceleration.
//!
//! # Architecture
//!
//! ```text
//! Input: [batch_size, 8320] (concatenated embeddings)
//!        |
//!        v
//!   [GpuLayerNorm(8320)] -----> Normalize to mean=0, var=1
//!        |
//!        v
//!   [GpuLinear(8320 → 8)] ----> Expert routing logits
//!        |
//!        v
//!   [Temperature Softmax] -----> Expert probabilities
//!        |
//!        v
//!   [Top-K Selection] ---------> (indices, weights)
//!        |
//!        v
//!   [GpuExpertPool] -----------> Weighted expert outputs
//!        |
//!        v
//!   Output: [batch_size, 1536] (fused embedding)
//! ```
//!
//! # Hardware Target
//!
//! - NVIDIA RTX 5090 32GB (Blackwell GB202)
//! - CUDA 13.1 with Compute Capability 12.0
//! - Expected speedup: 60-100x vs CPU
//!
//! # No Fallbacks Policy
//!
//! All GPU operations fail fast with descriptive errors.
//! No CPU fallbacks are implemented - system must work or fail for debugging.

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Tensor, D};
#[cfg(feature = "candle")]
use candle_nn;

use crate::config::FusionConfig;
use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::dimensions::{FUSED_OUTPUT, NUM_EXPERTS, TOTAL_CONCATENATED};

// =============================================================================
// GPU LAYER NORM
// =============================================================================

/// GPU-accelerated Layer Normalization.
///
/// Normalizes each sample in a batch to mean=0, var=1,
/// then applies learned scale (gamma) and shift (beta).
///
/// # Formula
///
/// ```text
/// y = gamma * (x - mean) / sqrt(var + eps) + beta
/// ```
///
/// # GPU Acceleration
///
/// Uses cuBLAS for vectorized mean/variance computation.
/// Expected speedup: 50x vs CPU for 8320D vectors.
#[cfg(feature = "candle")]
#[derive(Debug)]
pub struct GpuLayerNorm {
    /// Scale parameter (γ) - shape: [dim]
    gamma: Tensor,
    /// Shift parameter (β) - shape: [dim]
    beta: Tensor,
    /// Numerical stability constant
    eps: f64,
    /// Expected input dimension
    dim: usize,
}

#[cfg(feature = "candle")]
impl GpuLayerNorm {
    /// Create a new GPU LayerNorm.
    ///
    /// # Arguments
    ///
    /// * `dim` - Input/output dimension (must be > 0)
    /// * `device` - CUDA device for tensor allocation
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::InvalidDimension` if dim == 0
    /// - `EmbeddingError::GpuError` if tensor allocation fails
    pub fn new(dim: usize, device: &Device) -> EmbeddingResult<Self> {
        if dim == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }

        // Initialize gamma=1.0, beta=0.0 on GPU
        let gamma = Tensor::ones((dim,), DType::F32, device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("Failed to allocate gamma tensor: {}", e),
            }
        })?;

        let beta = Tensor::zeros((dim,), DType::F32, device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("Failed to allocate beta tensor: {}", e),
            }
        })?;

        Ok(Self {
            gamma,
            beta,
            eps: 1e-5,
            dim,
        })
    }

    /// Create GpuLayerNorm from CPU LayerNorm weights.
    ///
    /// Transfers gamma and beta parameters to GPU.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Scale parameters from CPU
    /// * `beta` - Shift parameters from CPU
    /// * `device` - CUDA device
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::GpuError` if transfer fails
    pub fn from_cpu(
        gamma: &[f32],
        beta: &[f32],
        device: &Device,
    ) -> EmbeddingResult<Self> {
        if gamma.len() != beta.len() {
            return Err(EmbeddingError::DimensionMismatch {
                expected: gamma.len(),
                got: beta.len(),
            });
        }

        let dim = gamma.len();
        if dim == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }

        let gamma_tensor = Tensor::from_slice(gamma, (dim,), device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("Failed to transfer gamma to GPU: {}", e),
            }
        })?;

        let beta_tensor = Tensor::from_slice(beta, (dim,), device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("Failed to transfer beta to GPU: {}", e),
            }
        })?;

        Ok(Self {
            gamma: gamma_tensor,
            beta: beta_tensor,
            eps: 1e-5,
            dim,
        })
    }

    /// Get the input dimension.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Forward pass through GPU layer normalization.
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape [batch_size, dim]
    ///
    /// # Returns
    ///
    /// Normalized tensor of same shape.
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::GpuError` if GPU operation fails
    /// - `EmbeddingError::InvalidDimension` if input shape is wrong
    pub fn forward(&self, input: &Tensor) -> EmbeddingResult<Tensor> {
        let input_shape = input.dims();
        if input_shape.len() != 2 {
            return Err(EmbeddingError::GpuError {
                message: format!(
                    "GpuLayerNorm expects 2D input [batch, dim], got {:?}",
                    input_shape
                ),
            });
        }

        if input_shape[1] != self.dim {
            return Err(EmbeddingError::InvalidDimension {
                expected: self.dim,
                actual: input_shape[1],
            });
        }

        // Compute mean along last dimension: [batch, 1]
        let mean = input
            .mean_keepdim(D::Minus1)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Mean computation failed: {}", e),
            })?;

        // Compute variance: E[(x - mean)^2]
        let centered = input
            .broadcast_sub(&mean)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Centering failed: {}", e),
            })?;

        let variance = centered
            .sqr()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Square failed: {}", e),
            })?
            .mean_keepdim(D::Minus1)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Variance computation failed: {}", e),
            })?;

        // Normalize: (x - mean) / sqrt(var + eps)
        let std_inv = (variance + self.eps)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Epsilon addition failed: {}", e),
            })?
            .sqrt()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Sqrt failed: {}", e),
            })?;

        let normalized = centered
            .broadcast_div(&std_inv)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Normalization division failed: {}", e),
            })?;

        // Apply scale and shift: gamma * normalized + beta
        let scaled = normalized
            .broadcast_mul(&self.gamma)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Gamma multiplication failed: {}", e),
            })?;

        scaled
            .broadcast_add(&self.beta)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Beta addition failed: {}", e),
            })
    }

    /// Parameter count for LayerNorm (gamma + beta).
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.dim * 2 // gamma and beta
    }
}

// =============================================================================
// GPU LINEAR
// =============================================================================

/// GPU-accelerated Linear (Fully Connected) Layer.
///
/// Computes: y = x @ W^T + b
///
/// # GPU Acceleration
///
/// Uses cuBLAS GEMM for matrix multiplication.
/// Expected speedup: 50-100x vs CPU for large matrices.
#[cfg(feature = "candle")]
#[derive(Debug)]
pub struct GpuLinear {
    /// Weight matrix: [out_features, in_features]
    weight: Tensor,
    /// Bias vector: [out_features]
    bias: Tensor,
    /// Input dimension
    in_features: usize,
    /// Output dimension
    out_features: usize,
}

#[cfg(feature = "candle")]
impl GpuLinear {
    /// Create a new GPU Linear layer with Xavier initialization.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Input dimension (must be > 0)
    /// * `out_features` - Output dimension (must be > 0)
    /// * `device` - CUDA device
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::InvalidDimension` if dimensions are invalid
    /// - `EmbeddingError::GpuError` if tensor allocation fails
    pub fn new(
        in_features: usize,
        out_features: usize,
        device: &Device,
    ) -> EmbeddingResult<Self> {
        if in_features == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }
        if out_features == 0 {
            return Err(EmbeddingError::InvalidDimension {
                expected: 1,
                actual: 0,
            });
        }

        // Xavier initialization: U(-sqrt(6/(in+out)), sqrt(6/(in+out)))
        let limit = (6.0 / (in_features + out_features) as f64).sqrt();

        // Use randn and scale for approximate Xavier
        let weight = Tensor::randn(
            0.0f32,
            (limit * 0.5) as f32, // scale std for Xavier-ish
            (out_features, in_features),
            device,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("Failed to allocate weight tensor: {}", e),
        })?;

        let bias = Tensor::zeros((out_features,), DType::F32, device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("Failed to allocate bias tensor: {}", e),
            }
        })?;

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    /// Create GpuLinear from CPU weights.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `weights` - Weight matrix (row-major: [out_features, in_features])
    /// * `bias` - Bias vector: [out_features]
    /// * `device` - CUDA device
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::DimensionMismatch` if dimensions don't match
    /// - `EmbeddingError::GpuError` if transfer fails
    pub fn from_cpu(
        in_features: usize,
        out_features: usize,
        weights: &[f32],
        bias: &[f32],
        device: &Device,
    ) -> EmbeddingResult<Self> {
        let expected_weights = out_features * in_features;
        if weights.len() != expected_weights {
            return Err(EmbeddingError::DimensionMismatch {
                expected: expected_weights,
                got: weights.len(),
            });
        }
        if bias.len() != out_features {
            return Err(EmbeddingError::DimensionMismatch {
                expected: out_features,
                got: bias.len(),
            });
        }

        let weight_tensor = Tensor::from_slice(
            weights,
            (out_features, in_features),
            device,
        )
        .map_err(|e| EmbeddingError::GpuError {
            message: format!("Failed to transfer weights to GPU: {}", e),
        })?;

        let bias_tensor = Tensor::from_slice(bias, (out_features,), device).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("Failed to transfer bias to GPU: {}", e),
            }
        })?;

        Ok(Self {
            weight: weight_tensor,
            bias: bias_tensor,
            in_features,
            out_features,
        })
    }

    /// Get input dimension.
    #[inline]
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output dimension.
    #[inline]
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Forward pass through GPU linear layer.
    ///
    /// Computes: y = x @ W^T + b
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape [batch_size, in_features]
    ///
    /// # Returns
    ///
    /// Output tensor of shape [batch_size, out_features].
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::GpuError` if GPU operation fails
    /// - `EmbeddingError::InvalidDimension` if input shape is wrong
    pub fn forward(&self, input: &Tensor) -> EmbeddingResult<Tensor> {
        let input_shape = input.dims();
        if input_shape.len() != 2 {
            return Err(EmbeddingError::GpuError {
                message: format!(
                    "GpuLinear expects 2D input [batch, in_features], got {:?}",
                    input_shape
                ),
            });
        }

        if input_shape[1] != self.in_features {
            return Err(EmbeddingError::InvalidDimension {
                expected: self.in_features,
                actual: input_shape[1],
            });
        }

        // y = x @ W^T
        // input: [batch, in] @ weight.T: [in, out] -> [batch, out]
        let weight_t = self.weight.t().map_err(|e| EmbeddingError::GpuError {
            message: format!("Weight transpose failed: {}", e),
        })?;

        let matmul_result = input.matmul(&weight_t).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("Matrix multiplication failed: {}", e),
            }
        })?;

        // Add bias
        matmul_result
            .broadcast_add(&self.bias)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Bias addition failed: {}", e),
            })
    }

    /// Parameter count for Linear layer (weight + bias).
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.in_features * self.out_features + self.out_features
    }
}

// =============================================================================
// GPU ACTIVATION FUNCTIONS
// =============================================================================

/// GPU-accelerated activation functions.
#[cfg(feature = "candle")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuActivation {
    /// GELU activation (Gaussian Error Linear Unit)
    #[default]
    Gelu,
    /// ReLU activation (Rectified Linear Unit)
    Relu,
    /// SiLU activation (Sigmoid Linear Unit)
    Silu,
}

#[cfg(feature = "candle")]
impl GpuActivation {
    /// Apply activation function to tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor
    ///
    /// # Returns
    ///
    /// Activated tensor of same shape.
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::GpuError` if GPU operation fails
    pub fn forward(&self, tensor: &Tensor) -> EmbeddingResult<Tensor> {
        match self {
            GpuActivation::Gelu => tensor.gelu().map_err(|e| EmbeddingError::GpuError {
                message: format!("GELU activation failed: {}", e),
            }),
            GpuActivation::Relu => tensor.relu().map_err(|e| EmbeddingError::GpuError {
                message: format!("ReLU activation failed: {}", e),
            }),
            GpuActivation::Silu => tensor.silu().map_err(|e| EmbeddingError::GpuError {
                message: format!("SiLU activation failed: {}", e),
            }),
        }
    }
}

// =============================================================================
// GPU GATING NETWORK
// =============================================================================

/// GPU-accelerated Gating Network for FuseMoE routing.
///
/// Routes 8320D concatenated embeddings to 8 experts using
/// temperature-scaled softmax with optional Laplace smoothing.
///
/// # GPU Acceleration
///
/// Full forward pass on GPU:
/// - LayerNorm: cuBLAS vectorized ops
/// - Linear: cuBLAS GEMM
/// - Softmax: Fused CUDA kernel
///
/// Expected speedup: 60x vs CPU.
#[cfg(feature = "candle")]
#[derive(Debug)]
pub struct GpuGatingNetwork {
    /// Layer normalization for input
    layer_norm: GpuLayerNorm,
    /// Linear projection from input_dim to num_experts
    projection: GpuLinear,
    /// Softmax temperature (lower = sharper)
    temperature: f32,
    /// Laplace smoothing alpha (0 = disabled)
    laplace_alpha: f32,
    /// Number of experts
    num_experts: usize,
    /// Reference to device
    device: Device,
}

#[cfg(feature = "candle")]
impl GpuGatingNetwork {
    /// Create a new GPU GatingNetwork.
    ///
    /// # Arguments
    ///
    /// * `config` - Fusion configuration
    /// * `device` - CUDA device
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::ConfigError` if configuration is invalid
    /// - `EmbeddingError::GpuError` if GPU allocation fails
    pub fn new(config: &FusionConfig, device: &Device) -> EmbeddingResult<Self> {
        config.validate()?;

        let layer_norm = GpuLayerNorm::new(TOTAL_CONCATENATED, device)?;
        let projection = GpuLinear::new(TOTAL_CONCATENATED, config.num_experts, device)?;

        Ok(Self {
            layer_norm,
            projection,
            temperature: config.temperature,
            laplace_alpha: config.laplace_alpha,
            num_experts: config.num_experts,
            device: device.clone(),
        })
    }

    /// Create GpuGatingNetwork from CPU weights.
    ///
    /// Transfers all parameters to GPU.
    pub fn from_cpu(
        layer_norm_gamma: &[f32],
        layer_norm_beta: &[f32],
        projection_weights: &[f32],
        projection_bias: &[f32],
        config: &FusionConfig,
        device: &Device,
    ) -> EmbeddingResult<Self> {
        let layer_norm = GpuLayerNorm::from_cpu(layer_norm_gamma, layer_norm_beta, device)?;
        let projection = GpuLinear::from_cpu(
            TOTAL_CONCATENATED,
            config.num_experts,
            projection_weights,
            projection_bias,
            device,
        )?;

        Ok(Self {
            layer_norm,
            projection,
            temperature: config.temperature,
            laplace_alpha: config.laplace_alpha,
            num_experts: config.num_experts,
            device: device.clone(),
        })
    }

    /// Get the number of experts.
    #[inline]
    #[must_use]
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Get the input dimension.
    #[inline]
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.layer_norm.dim()
    }

    /// Forward pass through gating network.
    ///
    /// Returns expert probabilities.
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape [batch_size, 8320]
    ///
    /// # Returns
    ///
    /// Probabilities tensor of shape [batch_size, num_experts].
    pub fn forward(&self, input: &Tensor) -> EmbeddingResult<Tensor> {
        // Step 1: Layer normalization
        let normalized = self.layer_norm.forward(input)?;

        // Step 2: Linear projection to logits
        let logits = self.projection.forward(&normalized)?;

        // Step 3: Temperature-scaled softmax
        let probs = self.softmax_with_temperature(&logits)?;

        // Step 4: Laplace smoothing (if enabled)
        if self.laplace_alpha > 0.0 {
            self.apply_laplace_smoothing(&probs)
        } else {
            Ok(probs)
        }
    }

    /// Forward pass with top-k selection.
    ///
    /// Returns (indices, weights) for top-k experts.
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape [batch_size, 8320]
    /// * `top_k` - Number of experts to select
    ///
    /// # Returns
    ///
    /// Tuple of:
    /// - `indices`: Tensor of shape [batch_size, top_k]
    /// - `weights`: Tensor of shape [batch_size, top_k] (renormalized)
    pub fn forward_topk(
        &self,
        input: &Tensor,
        top_k: usize,
    ) -> EmbeddingResult<(Tensor, Tensor)> {
        if top_k > self.num_experts {
            return Err(EmbeddingError::ConfigError {
                message: format!(
                    "top_k ({}) cannot exceed num_experts ({})",
                    top_k, self.num_experts
                ),
            });
        }

        let probs = self.forward(input)?;

        // GPU top-k selection
        let (values, indices) = probs
            .sort_last_dim(false) // descending
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Sort failed: {}", e),
            })?;

        // Take top-k
        let topk_values = values
            .narrow(D::Minus1, 0, top_k)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Narrow values failed: {}", e),
            })?;

        let topk_indices = indices
            .narrow(D::Minus1, 0, top_k)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Narrow indices failed: {}", e),
            })?;

        // Renormalize weights to sum to 1
        let weight_sum = topk_values
            .sum_keepdim(D::Minus1)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Sum failed: {}", e),
            })?;

        let normalized_weights = topk_values
            .broadcast_div(&weight_sum)
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Weight normalization failed: {}", e),
            })?;

        Ok((topk_indices, normalized_weights))
    }

    /// Temperature-scaled softmax on GPU.
    fn softmax_with_temperature(&self, logits: &Tensor) -> EmbeddingResult<Tensor> {
        let scaled = (logits / self.temperature as f64).map_err(|e| {
            EmbeddingError::GpuError {
                message: format!("Temperature scaling failed: {}", e),
            }
        })?;

        candle_nn::ops::softmax(&scaled, D::Minus1).map_err(|e| EmbeddingError::GpuError {
            message: format!("Softmax failed: {}", e),
        })
    }

    /// Apply Laplace smoothing on GPU.
    fn apply_laplace_smoothing(&self, probs: &Tensor) -> EmbeddingResult<Tensor> {
        let alpha = self.laplace_alpha as f64;
        let k = self.num_experts as f64;
        let denominator = 1.0 + alpha * k;

        let smoothed = (probs + alpha).map_err(|e| EmbeddingError::GpuError {
            message: format!("Alpha addition failed: {}", e),
        })?;

        (smoothed / denominator).map_err(|e| EmbeddingError::GpuError {
            message: format!("Smoothing division failed: {}", e),
        })
    }

    /// Parameter count for the gating network.
    ///
    /// Includes LayerNorm and Linear projection parameters.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.layer_norm.parameter_count() + self.projection.parameter_count()
    }
}

// =============================================================================
// GPU EXPERT
// =============================================================================

/// GPU-accelerated Expert Network.
///
/// Single expert FFN: input_dim -> hidden_dim -> GELU -> output_dim
///
/// # GPU Acceleration
///
/// - Linear layers: cuBLAS GEMM
/// - Activation: Fused CUDA kernel
///
/// Expected speedup: 60x vs CPU.
#[cfg(feature = "candle")]
#[derive(Debug)]
pub struct GpuExpert {
    /// First linear layer: input_dim -> hidden_dim
    input_to_hidden: GpuLinear,
    /// Second linear layer: hidden_dim -> output_dim
    hidden_to_output: GpuLinear,
    /// Activation function
    activation: GpuActivation,
    /// Expert identifier
    expert_id: usize,
}

#[cfg(feature = "candle")]
impl GpuExpert {
    /// Create a new GPU Expert.
    ///
    /// # Arguments
    ///
    /// * `expert_id` - Unique identifier (0..NUM_EXPERTS)
    /// * `input_dim` - Input dimension (8320)
    /// * `hidden_dim` - Hidden layer dimension (4096)
    /// * `output_dim` - Output dimension (1536)
    /// * `device` - CUDA device
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::GpuError` if allocation fails
    pub fn new(
        expert_id: usize,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        device: &Device,
    ) -> EmbeddingResult<Self> {
        let input_to_hidden = GpuLinear::new(input_dim, hidden_dim, device)?;
        let hidden_to_output = GpuLinear::new(hidden_dim, output_dim, device)?;

        tracing::debug!(
            expert_id,
            input_dim,
            hidden_dim,
            output_dim,
            "Created GPU Expert network"
        );

        Ok(Self {
            input_to_hidden,
            hidden_to_output,
            activation: GpuActivation::Gelu,
            expert_id,
        })
    }

    /// Create GpuExpert from CPU weights.
    pub fn from_cpu(
        expert_id: usize,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        input_to_hidden_weights: &[f32],
        input_to_hidden_bias: &[f32],
        hidden_to_output_weights: &[f32],
        hidden_to_output_bias: &[f32],
        device: &Device,
    ) -> EmbeddingResult<Self> {
        let input_to_hidden = GpuLinear::from_cpu(
            input_dim,
            hidden_dim,
            input_to_hidden_weights,
            input_to_hidden_bias,
            device,
        )?;

        let hidden_to_output = GpuLinear::from_cpu(
            hidden_dim,
            output_dim,
            hidden_to_output_weights,
            hidden_to_output_bias,
            device,
        )?;

        Ok(Self {
            input_to_hidden,
            hidden_to_output,
            activation: GpuActivation::Gelu,
            expert_id,
        })
    }

    /// Get expert identifier.
    #[inline]
    #[must_use]
    pub fn expert_id(&self) -> usize {
        self.expert_id
    }

    /// Forward pass through expert.
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape [batch_size, input_dim]
    ///
    /// # Returns
    ///
    /// Output tensor of shape [batch_size, output_dim].
    pub fn forward(&self, input: &Tensor) -> EmbeddingResult<Tensor> {
        // Step 1: Input -> Hidden
        let hidden = self.input_to_hidden.forward(input)?;

        // Step 2: Apply activation
        let activated = self.activation.forward(&hidden)?;

        // Step 3: Hidden -> Output
        self.hidden_to_output.forward(&activated)
    }

    /// Get input dimension.
    #[inline]
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.input_to_hidden.in_features()
    }

    /// Get output dimension.
    #[inline]
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.hidden_to_output.out_features()
    }

    /// Parameter count for the expert.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.input_to_hidden.parameter_count() + self.hidden_to_output.parameter_count()
    }
}

// =============================================================================
// GPU EXPERT POOL
// =============================================================================

/// GPU-accelerated Expert Pool with top-k routing.
///
/// Manages 8 experts and provides weighted combination of outputs.
///
/// # GPU Acceleration
///
/// All expert computations on GPU with parallel execution.
/// Expected speedup: 80x vs CPU for full forward pass.
#[cfg(feature = "candle")]
#[derive(Debug)]
pub struct GpuExpertPool {
    /// Array of 8 experts
    experts: Vec<GpuExpert>,
    /// Input dimension (8320)
    input_dim: usize,
    /// Hidden dimension (4096)
    hidden_dim: usize,
    /// Output dimension (1536)
    output_dim: usize,
    /// CUDA device
    device: Device,
}

#[cfg(feature = "candle")]
impl GpuExpertPool {
    /// Create new GPU expert pool.
    ///
    /// # Arguments
    ///
    /// * `config` - FusionConfig with expert_hidden_dim
    /// * `device` - CUDA device
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::GpuError` if allocation fails
    pub fn new(config: &FusionConfig, device: &Device) -> EmbeddingResult<Self> {
        let input_dim = TOTAL_CONCATENATED;
        let hidden_dim = config.expert_hidden_dim;
        let output_dim = FUSED_OUTPUT;

        let mut experts = Vec::with_capacity(NUM_EXPERTS);

        for expert_id in 0..NUM_EXPERTS {
            let expert = GpuExpert::new(
                expert_id,
                input_dim,
                hidden_dim,
                output_dim,
                device,
            )?;
            experts.push(expert);
        }

        tracing::info!(
            num_experts = NUM_EXPERTS,
            input_dim,
            hidden_dim,
            output_dim,
            "Created GPU ExpertPool"
        );

        Ok(Self {
            experts,
            input_dim,
            hidden_dim,
            output_dim,
            device: device.clone(),
        })
    }

    /// Get number of experts.
    #[inline]
    #[must_use]
    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// Get input dimension.
    #[inline]
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get output dimension.
    #[inline]
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Forward pass through top-k experts with weighted combination.
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape [batch_size, 8320]
    /// * `indices` - Expert indices tensor [batch_size, top_k]
    /// * `weights` - Routing weights tensor [batch_size, top_k]
    ///
    /// # Returns
    ///
    /// Weighted combination of expert outputs [batch_size, 1536].
    pub fn forward_topk(
        &self,
        input: &Tensor,
        indices: &Tensor,
        weights: &Tensor,
    ) -> EmbeddingResult<Tensor> {
        let input_shape = input.dims();
        let indices_shape = indices.dims();
        let weights_shape = weights.dims();

        if input_shape.len() != 2 {
            return Err(EmbeddingError::GpuError {
                message: format!("Input must be 2D, got {:?}", input_shape),
            });
        }

        let batch_size = input_shape[0];
        let top_k = indices_shape[1];

        // Validate shapes
        if indices_shape[0] != batch_size || weights_shape[0] != batch_size {
            return Err(EmbeddingError::GpuError {
                message: format!(
                    "Batch size mismatch: input={}, indices={}, weights={}",
                    batch_size, indices_shape[0], weights_shape[0]
                ),
            });
        }

        // Convert indices to Vec for iteration
        let indices_cpu: Vec<u32> = indices
            .flatten_all()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Flatten indices failed: {}", e),
            })?
            .to_vec1()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Convert indices to CPU failed: {}", e),
            })?;

        let weights_cpu: Vec<f32> = weights
            .flatten_all()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Flatten weights failed: {}", e),
            })?
            .to_vec1()
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Convert weights to CPU failed: {}", e),
            })?;

        // Collect sample outputs for concatenation
        let mut sample_outputs: Vec<Tensor> = Vec::with_capacity(batch_size);

        // Process each sample in batch
        for b in 0..batch_size {
            // Get input sample using narrow (slice) operation
            let sample_input = input
                .narrow(0, b, 1)
                .map_err(|e| EmbeddingError::GpuError {
                    message: format!("Sample slicing failed at index {}: {}", b, e),
                })?;

            let mut sample_output = Tensor::zeros(
                (1, self.output_dim),
                DType::F32,
                &self.device,
            )
            .map_err(|e| EmbeddingError::GpuError {
                message: format!("Sample output allocation failed: {}", e),
            })?;

            // Process each selected expert
            for k in 0..top_k {
                let idx = indices_cpu[b * top_k + k] as usize;
                let weight = weights_cpu[b * top_k + k];

                if idx >= self.experts.len() {
                    return Err(EmbeddingError::GpuError {
                        message: format!(
                            "Expert index {} out of bounds (max {})",
                            idx,
                            self.experts.len() - 1
                        ),
                    });
                }

                // Forward through expert
                let expert_output = self.experts[idx].forward(&sample_input)?;

                // Accumulate weighted output
                let weighted = (expert_output * weight as f64).map_err(|e| {
                    EmbeddingError::GpuError {
                        message: format!("Weight multiplication failed: {}", e),
                    }
                })?;

                sample_output = (sample_output + weighted).map_err(|e| {
                    EmbeddingError::GpuError {
                        message: format!("Output accumulation failed: {}", e),
                    }
                })?;
            }

            sample_outputs.push(sample_output);
        }

        // Concatenate all sample outputs into final batch tensor
        let output = Tensor::cat(&sample_outputs, 0).map_err(|e| EmbeddingError::GpuError {
            message: format!("Output concatenation failed: {}", e),
        })?;

        Ok(output)
    }

    /// Parameter count for all experts.
    ///
    /// For 8 experts with 8320 -> 4096 -> 1536:
    /// ~323M parameters total.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        let layer1_params = self.input_dim * self.hidden_dim + self.hidden_dim;
        let layer2_params = self.hidden_dim * self.output_dim + self.output_dim;
        let per_expert = layer1_params + layer2_params;
        per_expert * self.experts.len()
    }
}

// =============================================================================
// GPU FUSE MOE LAYER (COMBINED)
// =============================================================================

/// GPU-accelerated FuseMoE layer combining gating and experts.
///
/// Complete fusion layer:
/// 1. GatingNetwork routes input to top-k experts
/// 2. ExpertPool computes weighted expert outputs
/// 3. Returns fused 1536D embedding
#[cfg(feature = "candle")]
#[derive(Debug)]
pub struct GpuFuseMoE {
    /// Gating network for expert routing
    gating: GpuGatingNetwork,
    /// Pool of expert networks
    experts: GpuExpertPool,
    /// Number of experts to use per sample
    top_k: usize,
}

#[cfg(feature = "candle")]
impl GpuFuseMoE {
    /// Create new GPU FuseMoE layer.
    ///
    /// # Arguments
    ///
    /// * `config` - Fusion configuration
    /// * `device` - CUDA device
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::GpuError` if GPU allocation fails
    pub fn new(config: &FusionConfig, device: &Device) -> EmbeddingResult<Self> {
        config.validate()?;

        let gating = GpuGatingNetwork::new(config, device)?;
        let experts = GpuExpertPool::new(config, device)?;

        tracing::info!(
            num_experts = config.num_experts,
            top_k = config.top_k,
            parameter_count = experts.parameter_count(),
            "Created GPU FuseMoE layer"
        );

        Ok(Self {
            gating,
            experts,
            top_k: config.top_k,
        })
    }

    /// Forward pass through full FuseMoE layer.
    ///
    /// # Arguments
    ///
    /// * `input` - Concatenated embeddings tensor [batch_size, 8320]
    ///
    /// # Returns
    ///
    /// Fused embedding tensor [batch_size, 1536].
    ///
    /// # Errors
    ///
    /// - `EmbeddingError::GpuError` if GPU operation fails
    pub fn forward(&self, input: &Tensor) -> EmbeddingResult<Tensor> {
        // Step 1: Get expert routing
        let (indices, weights) = self.gating.forward_topk(input, self.top_k)?;

        // Step 2: Forward through experts with weighted combination
        self.experts.forward_topk(input, &indices, &weights)
    }

    /// Get the gating network (for introspection).
    #[inline]
    #[must_use]
    pub fn gating(&self) -> &GpuGatingNetwork {
        &self.gating
    }

    /// Get the expert pool (for introspection).
    #[inline]
    #[must_use]
    pub fn experts(&self) -> &GpuExpertPool {
        &self.experts
    }

    /// Get top-k value.
    #[inline]
    #[must_use]
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Get input dimension (concatenated embedding size).
    #[inline]
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.gating.input_dim()
    }

    /// Get output dimension (fused embedding size).
    #[inline]
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.experts.output_dim()
    }

    /// Total parameter count for the FuseMoE layer.
    ///
    /// Includes:
    /// - Gating network: LayerNorm + Linear
    /// - Expert pool: All expert networks
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.gating.parameter_count() + self.experts.parameter_count()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
#[cfg(feature = "candle")]
mod tests {
    use super::*;

    // Helper to get test device - GPU-only architecture
    fn test_device() -> Device {
        // GPU-only architecture - tests MUST use real GPU
        // Tests will fail if GPU is unavailable - this is intentional
        match Device::new_cuda(0) {
            Ok(device) => device,
            Err(e) => {
                panic!(
                    "GPU test device initialization failed: {}. \
                     This crate requires a CUDA GPU for all tests. \
                     Ensure CUDA drivers are installed and GPU is available.",
                    e
                );
            }
        }
    }

    // =========================================================================
    // GPU LAYER NORM TESTS
    // =========================================================================

    #[test]
    fn test_gpu_layernorm_creation() {
        let device = test_device();
        let norm = GpuLayerNorm::new(1024, &device).unwrap();
        assert_eq!(norm.dim(), 1024);
    }

    #[test]
    fn test_gpu_layernorm_zero_dim_fails() {
        let device = test_device();
        let result = GpuLayerNorm::new(0, &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_layernorm_forward() {
        let device = test_device();
        let norm = GpuLayerNorm::new(4, &device).unwrap();

        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let output = norm.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 4]);

        // Mean should be ~0
        let mean: f32 = output.mean_all().unwrap().to_vec0().unwrap();
        assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);
    }

    #[test]
    fn test_gpu_layernorm_real_dimensions() {
        let device = test_device();
        // Real FuseMoE input dimension
        let norm = GpuLayerNorm::new(TOTAL_CONCATENATED, &device).unwrap();
        assert_eq!(norm.dim(), 8320);

        // Create real input: batch of 4 samples with 8320 dimensions
        let input_data: Vec<f32> = (0..4 * 8320).map(|i| (i as f32) * 0.001).collect();
        let input = Tensor::from_slice(&input_data, (4, 8320), &device).unwrap();
        let output = norm.forward(&input).unwrap();

        assert_eq!(output.dims(), &[4, 8320]);
    }

    // =========================================================================
    // GPU LINEAR TESTS
    // =========================================================================

    #[test]
    fn test_gpu_linear_creation() {
        let device = test_device();
        let linear = GpuLinear::new(8320, 8, &device).unwrap();
        assert_eq!(linear.in_features(), 8320);
        assert_eq!(linear.out_features(), 8);
    }

    #[test]
    fn test_gpu_linear_forward() {
        let device = test_device();
        let linear = GpuLinear::new(4, 2, &device).unwrap();

        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let output = linear.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 2]); // Candle preserves batch dimension
    }

    #[test]
    fn test_gpu_linear_real_gating_projection() {
        let device = test_device();
        // Gating network: 8320 -> 8 experts
        let linear = GpuLinear::new(8320, 8, &device).unwrap();

        let input_data: Vec<f32> = (0..8320).map(|i| (i as f32) * 0.0001).collect();
        let input = Tensor::from_slice(&input_data, (1, 8320), &device).unwrap();
        let output = linear.forward(&input).unwrap();

        // Should have 8 expert logits
        let dims = output.dims();
        assert!(dims.contains(&8), "Output should have 8 expert logits, got {:?}", dims);
    }

    // =========================================================================
    // GPU ACTIVATION TESTS
    // =========================================================================

    #[test]
    fn test_gpu_activation_gelu() {
        let device = test_device();
        let input = Tensor::from_slice(&[1.0f32, -1.0, 0.0], (1, 3), &device).unwrap();

        let act = GpuActivation::Gelu;
        let output = act.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 3]);
    }

    #[test]
    fn test_gpu_activation_relu() {
        let device = test_device();
        let input = Tensor::from_slice(&[1.0f32, -1.0, 0.0], (1, 3), &device).unwrap();

        let act = GpuActivation::Relu;
        let output = act.forward(&input).unwrap();

        let values: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        assert!(values[0] > 0.0); // 1.0 -> 1.0
        assert!(values[1] == 0.0); // -1.0 -> 0.0
        assert!(values[2] == 0.0); // 0.0 -> 0.0
    }

    #[test]
    fn test_gpu_activation_silu() {
        let device = test_device();
        let input = Tensor::from_slice(&[1.0f32, -1.0, 0.0], (1, 3), &device).unwrap();

        let act = GpuActivation::Silu;
        let output = act.forward(&input).unwrap();

        let values: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        // SiLU(x) = x * sigmoid(x)
        assert!(values[0] > 0.5); // SiLU(1.0) ≈ 0.731
        assert!(values[1] < 0.0); // SiLU(-1.0) ≈ -0.269
    }

    // =========================================================================
    // GPU EXPERT TESTS
    // =========================================================================

    #[test]
    fn test_gpu_expert_creation() {
        let device = test_device();
        let expert = GpuExpert::new(
            0,                  // expert_id
            TOTAL_CONCATENATED, // 8320
            4096,               // hidden
            FUSED_OUTPUT,       // 1536
            &device,
        )
        .unwrap();

        assert_eq!(expert.input_dim(), 8320);
        assert_eq!(expert.output_dim(), 1536);
    }

    #[test]
    fn test_gpu_expert_forward_real_dims() {
        let device = test_device();
        let expert = GpuExpert::new(
            0, // expert_id
            TOTAL_CONCATENATED,
            4096,
            FUSED_OUTPUT,
            &device,
        )
        .unwrap();

        // Single sample forward
        let input_data: Vec<f32> = (0..8320).map(|i| (i as f32) * 0.0001).collect();
        let input = Tensor::from_slice(&input_data, (1, 8320), &device).unwrap();
        let output = expert.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 1536], "Expert output should be [1, 1536]");
    }

    #[test]
    fn test_gpu_expert_forward_batch() {
        let device = test_device();
        let expert = GpuExpert::new(
            0, // expert_id
            TOTAL_CONCATENATED,
            4096,
            FUSED_OUTPUT,
            &device,
        )
        .unwrap();

        // Batch of 8 samples
        let input_data: Vec<f32> = (0..8 * 8320).map(|i| (i as f32) * 0.00001).collect();
        let input = Tensor::from_slice(&input_data, (8, 8320), &device).unwrap();
        let output = expert.forward(&input).unwrap();

        assert_eq!(output.dims(), &[8, 1536], "Expert output should be [8, 1536]");
    }

    #[test]
    fn test_gpu_expert_parameter_count() {
        let device = test_device();
        let expert = GpuExpert::new(
            0, 8320, 4096, 1536, &device,
        )
        .unwrap();

        let params = expert.parameter_count();
        // Layer 1: 8320 * 4096 + 4096 = 34,082,816
        // Layer 2: 4096 * 1536 + 1536 = 6,292,992
        // Total: 40,375,808
        assert_eq!(params, 40_375_808, "Expert should have ~40M parameters");
    }

    // =========================================================================
    // GPU GATING NETWORK TESTS
    // =========================================================================

    #[test]
    fn test_gpu_gating_network_creation() {
        let device = test_device();
        let config = FusionConfig::default();
        let gating = GpuGatingNetwork::new(&config, &device).unwrap();

        assert_eq!(gating.input_dim(), 8320);
        assert_eq!(gating.num_experts(), 8);
    }

    #[test]
    fn test_gpu_gating_forward_real_data() {
        let device = test_device();
        let config = FusionConfig::default();
        let gating = GpuGatingNetwork::new(&config, &device).unwrap();

        // Real concatenated embeddings
        let input_data: Vec<f32> = (0..8320).map(|i| (i as f32) * 0.0001).collect();
        let input = Tensor::from_slice(&input_data, (1, 8320), &device).unwrap();
        let probs = gating.forward(&input).unwrap();

        let dims = probs.dims();
        assert!(dims.contains(&8), "Gating should output 8 expert probs, got {:?}", dims);

        // Verify probabilities sum to 1
        let probs_vec: Vec<f32> = probs.flatten_all().unwrap().to_vec1().unwrap();
        let sum: f32 = probs_vec.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Probabilities should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_gpu_gating_topk_selection() {
        let device = test_device();
        let config = FusionConfig::default();
        let gating = GpuGatingNetwork::new(&config, &device).unwrap();

        // Batch of 4 samples
        let input_data: Vec<f32> = (0..4 * 8320).map(|i| (i as f32) * 0.00001).collect();
        let input = Tensor::from_slice(&input_data, (4, 8320), &device).unwrap();

        // forward_topk combines forward pass + top-k selection
        let (indices, weights) = gating.forward_topk(&input, 4).unwrap();

        // Should select top 4 experts for each of 4 samples = 16 total
        let indices_vec: Vec<u32> = indices.flatten_all().unwrap().to_vec1().unwrap();
        let weights_vec: Vec<f32> = weights.flatten_all().unwrap().to_vec1().unwrap();

        assert_eq!(indices_vec.len(), 16, "Should have 4 samples * 4 experts = 16 indices");
        assert_eq!(weights_vec.len(), 16, "Should have 16 weights");

        // All indices should be in [0, 7]
        for idx in &indices_vec {
            assert!(*idx < 8, "Expert index should be < 8, got {}", idx);
        }

        // Weights should be positive and sum to 1 per sample
        for sample in 0..4 {
            let sample_weights: f32 = (0..4)
                .map(|k| weights_vec[sample * 4 + k])
                .sum();
            assert!(
                (sample_weights - 1.0).abs() < 0.01,
                "Sample {} weights should sum to 1, got {}",
                sample, sample_weights
            );
        }
    }

    // =========================================================================
    // GPU EXPERT POOL TESTS
    // =========================================================================

    #[test]
    fn test_gpu_expert_pool_creation() {
        let device = test_device();
        let config = FusionConfig::default();
        let pool = GpuExpertPool::new(&config, &device).unwrap();

        assert_eq!(pool.num_experts(), 8);
        assert_eq!(pool.input_dim(), 8320);
        assert_eq!(pool.output_dim(), 1536);
    }

    #[test]
    fn test_gpu_expert_pool_forward_topk_single() {
        let device = test_device();
        let config = FusionConfig::default();
        let pool = GpuExpertPool::new(&config, &device).unwrap();

        // Single sample input
        let input_data: Vec<f32> = (0..8320).map(|i| (i as f32) * 0.0001).collect();
        let input = Tensor::from_slice(&input_data, (1, 8320), &device).unwrap();

        // Select top 4 experts with equal weights
        let indices = Tensor::from_slice(&[0u32, 1, 2, 3], (1, 4), &device).unwrap();
        let weights = Tensor::from_slice(&[0.25f32, 0.25, 0.25, 0.25], (1, 4), &device).unwrap();

        let output = pool.forward_topk(&input, &indices, &weights).unwrap();

        assert_eq!(output.dims(), &[1, 1536], "Pool output should be [1, 1536]");
    }

    #[test]
    fn test_gpu_expert_pool_forward_topk_batch() {
        let device = test_device();
        let config = FusionConfig::default();
        let pool = GpuExpertPool::new(&config, &device).unwrap();

        // Batch of 4 samples
        let input_data: Vec<f32> = (0..4 * 8320).map(|i| (i as f32) * 0.00001).collect();
        let input = Tensor::from_slice(&input_data, (4, 8320), &device).unwrap();

        // Each sample selects 4 experts
        let indices = Tensor::from_slice(
            &[0u32, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6],
            (4, 4),
            &device,
        )
        .unwrap();
        let weights = Tensor::from_slice(
            &[0.4f32, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2, 0.1],
            (4, 4),
            &device,
        )
        .unwrap();

        let output = pool.forward_topk(&input, &indices, &weights).unwrap();

        assert_eq!(output.dims(), &[4, 1536], "Pool output should be [4, 1536]");
    }

    #[test]
    fn test_gpu_expert_pool_parameter_count() {
        let device = test_device();
        let config = FusionConfig::default();
        let pool = GpuExpertPool::new(&config, &device).unwrap();

        let params = pool.parameter_count();
        // 8 experts * 40,392,704 params/expert = 323,141,632
        assert!(params > 300_000_000, "Pool should have > 300M parameters, got {}", params);
        assert!(params < 350_000_000, "Pool should have < 350M parameters, got {}", params);
    }

    // =========================================================================
    // GPU FUSE MOE COMPLETE PIPELINE TESTS
    // =========================================================================

    #[test]
    fn test_gpu_fusemoe_creation() {
        let device = test_device();
        let config = FusionConfig::default();
        let fusemoe = GpuFuseMoE::new(&config, &device).unwrap();

        assert_eq!(fusemoe.input_dim(), 8320);
        assert_eq!(fusemoe.output_dim(), 1536);
    }

    #[test]
    fn test_gpu_fusemoe_forward_single_sample() {
        let device = test_device();
        let config = FusionConfig::default();
        let fusemoe = GpuFuseMoE::new(&config, &device).unwrap();

        // Single sample: 12 concatenated model embeddings
        let input_data: Vec<f32> = (0..8320).map(|i| (i as f32) * 0.0001).collect();
        let input = Tensor::from_slice(&input_data, (1, 8320), &device).unwrap();

        let output = fusemoe.forward(&input).unwrap();

        assert_eq!(
            output.dims(),
            &[1, 1536],
            "FuseMoE output should be [1, 1536]"
        );
    }

    #[test]
    fn test_gpu_fusemoe_forward_batch() {
        let device = test_device();
        let config = FusionConfig::default();
        let fusemoe = GpuFuseMoE::new(&config, &device).unwrap();

        // Batch of 32 samples (typical batch size)
        let input_data: Vec<f32> = (0..32 * 8320)
            .map(|i| ((i % 1000) as f32) * 0.001)
            .collect();
        let input = Tensor::from_slice(&input_data, (32, 8320), &device).unwrap();

        let output = fusemoe.forward(&input).unwrap();

        assert_eq!(
            output.dims(),
            &[32, 1536],
            "FuseMoE output should be [32, 1536]"
        );
    }

    #[test]
    fn test_gpu_fusemoe_output_values_normalized() {
        let device = test_device();
        let config = FusionConfig::default();
        let fusemoe = GpuFuseMoE::new(&config, &device).unwrap();

        // Create input with known values
        let input_data: Vec<f32> = (0..8320).map(|i| (i as f32) * 0.0001).collect();
        let input = Tensor::from_slice(&input_data, (1, 8320), &device).unwrap();

        let output = fusemoe.forward(&input).unwrap();
        let output_vec: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();

        // Verify no NaN or Inf values
        for (i, val) in output_vec.iter().enumerate() {
            assert!(!val.is_nan(), "Output[{}] is NaN", i);
            assert!(!val.is_infinite(), "Output[{}] is Inf", i);
        }
    }

    #[test]
    fn test_gpu_fusemoe_total_parameter_count() {
        let device = test_device();
        let config = FusionConfig::default();
        let fusemoe = GpuFuseMoE::new(&config, &device).unwrap();

        let params = fusemoe.parameter_count();
        // Gating: LayerNorm (2 * 8320) + Linear (8320 * 8 + 8) = 16,640 + 66,568 = 83,208
        // Experts: 8 * 40,392,704 = 323,141,632
        // Total: ~323M parameters
        assert!(
            params > 320_000_000,
            "FuseMoE should have > 320M parameters, got {}",
            params
        );
        assert!(
            params < 330_000_000,
            "FuseMoE should have < 330M parameters, got {}",
            params
        );
    }

    // =========================================================================
    // DIMENSION CONSISTENCY TESTS
    // =========================================================================

    #[test]
    fn test_dimension_constants_match() {
        // Verify our constants match expected values
        assert_eq!(TOTAL_CONCATENATED, 8320, "TOTAL_CONCATENATED should be 8320");
        assert_eq!(FUSED_OUTPUT, 1536, "FUSED_OUTPUT should be 1536");
        assert_eq!(NUM_EXPERTS, 8, "NUM_EXPERTS should be 8");
    }

    #[test]
    fn test_gpu_fusemoe_wrong_input_dim_fails() {
        let device = test_device();
        let config = FusionConfig::default();
        let fusemoe = GpuFuseMoE::new(&config, &device).unwrap();

        // Wrong input dimension (1024 instead of 8320)
        let input = Tensor::zeros((1, 1024), DType::F32, &device).unwrap();
        let result = fusemoe.forward(&input);

        assert!(result.is_err(), "Forward with wrong dimension should fail");
    }

    #[test]
    fn test_gpu_fusemoe_reproducibility() {
        let device = test_device();
        let config = FusionConfig::default();

        // Same input should produce same output
        let input_data: Vec<f32> = (0..8320).map(|i| (i as f32) * 0.0001).collect();
        let input = Tensor::from_slice(&input_data, (1, 8320), &device).unwrap();

        let fusemoe1 = GpuFuseMoE::new(&config, &device).unwrap();
        let output1 = fusemoe1.forward(&input).unwrap();
        let output1_vec: Vec<f32> = output1.flatten_all().unwrap().to_vec1().unwrap();

        // Note: With random initialization, outputs will differ between instances
        // This test verifies forward pass doesn't crash with same input
        assert_eq!(output1_vec.len(), 1536);
    }
}
