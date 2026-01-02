//! FuseMoE fusion layer components.
//!
//! This module implements the Mixture-of-Experts fusion for combining
//! 12 model embeddings into a unified 1536D representation.
//!
//! # Components
//!
//! ## CPU Components
//! - [`GatingNetwork`]: Routes 8320D concatenated embeddings to 8 experts
//! - [`LayerNorm`]: Input normalization for stability
//! - [`Linear`]: Projection layer for the gating network
//! - [`Expert`]: Single expert FFN (8320 -> 4096 -> 1536)
//! - [`ExpertPool`]: Pool of 8 experts with top-k routing
//! - [`Activation`]: Activation functions for experts (GELU, ReLU, SiLU)
//!
//! ## GPU Components (feature = "candle")
//! - [`GpuLayerNorm`]: GPU-accelerated layer normalization
//! - [`GpuLinear`]: GPU-accelerated linear layer with cuBLAS GEMM
//! - [`GpuGatingNetwork`]: GPU-accelerated gating network
//! - [`GpuExpert`]: GPU-accelerated expert network
//! - [`GpuExpertPool`]: GPU-accelerated expert pool with top-k routing
//! - [`GpuFuseMoE`]: Complete GPU fusion layer (60-100x speedup)
//!
//! # Example (CPU)
//!
//! ```rust,ignore
//! use context_graph_embeddings::fusion::{GatingNetwork, ExpertPool};
//! use context_graph_embeddings::config::FusionConfig;
//! use context_graph_embeddings::types::dimensions::TOP_K_EXPERTS;
//!
//! let config = FusionConfig::default();
//! let gating = GatingNetwork::new(&config)?;
//! let experts = ExpertPool::new(&config)?;
//!
//! let input = vec![0.5f32; 8320];
//! let probs = gating.forward(&input, 1)?;
//! let (indices, weights) = gating.select_top_k(&probs, 1, TOP_K_EXPERTS)?;
//! let output = experts.forward_topk(&input, 1, &indices, &weights, TOP_K_EXPERTS)?;
//! assert_eq!(output.len(), 1536);
//! ```
//!
//! # Example (GPU)
//!
//! ```rust,ignore
//! use context_graph_embeddings::fusion::GpuFuseMoE;
//! use context_graph_embeddings::gpu::init_gpu;
//! use context_graph_embeddings::config::FusionConfig;
//!
//! let device = init_gpu()?;
//! let config = FusionConfig::default();
//! let fusion = GpuFuseMoE::new(&config, device)?;
//!
//! let input = Tensor::randn(0.0, 1.0, (32, 8320), device)?;
//! let output = fusion.forward(&input)?;  // [32, 1536]
//! ```

pub mod experts;
pub mod gating;
#[cfg(feature = "candle")]
pub mod gpu_fusion;

pub use experts::{Activation, Expert, ExpertPool};
pub use gating::{GatingNetwork, LayerNorm, Linear};

// GPU exports (available with cuda feature)
#[cfg(feature = "candle")]
pub use gpu_fusion::{
    GpuActivation, GpuExpert, GpuExpertPool, GpuFuseMoE, GpuGatingNetwork,
    GpuLayerNorm, GpuLinear,
};
