//! Configuration types for Knowledge Graph components.
//!
//! This module provides configuration structures for:
//! - FAISS IVF-PQ vector index (IndexConfig)
//! - Hyperbolic/Poincare ball geometry (HyperbolicConfig)
//! - Entailment cones for IS-A queries (ConeConfig)
//!
//! # Constitution Reference
//!
//! - perf.latency.faiss_1M_k100: <2ms (drives nlist/nprobe defaults)
//! - embeddings.models.E7_Code: 1536D (default dimension)

mod cone;
mod hyperbolic;
mod index;

// Re-export all public types for backwards compatibility
pub use self::cone::ConeConfig;
pub use self::hyperbolic::HyperbolicConfig;
pub use self::index::IndexConfig;

#[cfg(test)]
mod tests;
