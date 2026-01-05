//! Native and projected dimension constants for the 12-model embedding pipeline.
//!
//! This module defines the exact dimensions for each model in the ensemble:
//! - Native dimensions: Raw model output sizes
//! - Projected dimensions: Target sizes for Multi-Array Storage

// =============================================================================
// NATIVE OUTPUT DIMENSIONS (before any projection)
// =============================================================================

/// E1: Semantic embedding native dimension (e5-large-v2)
pub const SEMANTIC_NATIVE: usize = 1024;

/// E2: Temporal-Recent native dimension (custom exponential decay)
pub const TEMPORAL_RECENT_NATIVE: usize = 512;

/// E3: Temporal-Periodic native dimension (custom Fourier basis)
pub const TEMPORAL_PERIODIC_NATIVE: usize = 512;

/// E4: Temporal-Positional native dimension (custom sinusoidal PE)
pub const TEMPORAL_POSITIONAL_NATIVE: usize = 512;

/// E5: Causal embedding native dimension (Longformer)
pub const CAUSAL_NATIVE: usize = 768;

/// E6: Sparse lexical native dimension (SPLADE vocab size, ~5% active)
pub const SPARSE_NATIVE: usize = 30522;

/// E7: Code embedding native dimension (CodeT5p embed_dim)
pub const CODE_NATIVE: usize = 256;

/// E8: Graph embedding native dimension (paraphrase-MiniLM-L6-v2)
pub const GRAPH_NATIVE: usize = 384;

/// E9: Hyperdimensional computing native dimension (10K-bit vector)
pub const HDC_NATIVE: usize = 10000;

/// E10: Multimodal embedding native dimension (CLIP)
pub const MULTIMODAL_NATIVE: usize = 768;

/// E11: Entity embedding native dimension (all-MiniLM-L6-v2)
pub const ENTITY_NATIVE: usize = 384;

/// E12: Late-interaction native dimension per token (ColBERT)
pub const LATE_INTERACTION_NATIVE: usize = 128;

// =============================================================================
// PROJECTED DIMENSIONS (for Multi-Array Storage)
// =============================================================================

/// E1: Semantic projected dimension (no projection needed)
pub const SEMANTIC: usize = 1024;

/// E2: Temporal-Recent projected dimension (no projection needed)
pub const TEMPORAL_RECENT: usize = 512;

/// E3: Temporal-Periodic projected dimension (no projection needed)
pub const TEMPORAL_PERIODIC: usize = 512;

/// E4: Temporal-Positional projected dimension (no projection needed)
pub const TEMPORAL_POSITIONAL: usize = 512;

/// E5: Causal projected dimension (no projection needed)
pub const CAUSAL: usize = 768;

/// E6: Sparse projected dimension (30K sparse -> 1536D via learned projection)
pub const SPARSE: usize = 1536;

/// E7: Code projected dimension (256 embed -> 768D via projection to match CodeT5p d_model)
pub const CODE: usize = 768;

/// E8: Graph projected dimension (no projection needed)
pub const GRAPH: usize = 384;

/// E9: HDC projected dimension (10K-bit -> 1024D via learned projection)
pub const HDC: usize = 1024;

/// E10: Multimodal projected dimension (no projection needed)
pub const MULTIMODAL: usize = 768;

/// E11: Entity projected dimension (no projection needed)
pub const ENTITY: usize = 384;

/// E12: Late-interaction projected dimension (pooled to single vector)
pub const LATE_INTERACTION: usize = 128;
