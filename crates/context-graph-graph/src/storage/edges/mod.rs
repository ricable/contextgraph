//! Graph edge types optimized for RocksDB storage.
//!
//! This module provides GraphEdge with Marblestone neurotransmitter modulation
//! and steering-based weight adjustment for domain-aware graph traversal.
//!
//! # CANONICAL FORMULA for get_modulated_weight()
//!
//! ```text
//! net_activation = excitatory - inhibitory + (modulatory * 0.5)
//! domain_bonus = 0.1 if edge_domain == query_domain else 0.0
//! steering_factor = 0.5 + steering_reward  // Range [0.5, 1.5]
//! w_eff = base_weight * (1.0 + net_activation + domain_bonus) * steering_factor
//! ```
//! Result clamped to [0.0, 1.0]
//!
//! # Constitution Reference
//!
//! - edge_model.nt_weights: Definition and formula
//! - edge_model.steering_reward: [-1,1] range
//! - AP-009: NaN/Infinity clamped to valid range
//!
//! # Module Structure
//!
//! - [`types`]: Core type definitions (EdgeId, GraphEdge)
//! - [`impl_core`]: Constructors and factory methods
//! - [`impl_modulation`]: Weight modulation and traversal methods
//! - [`traits`]: PartialEq, Eq, Hash implementations

// ========== Submodules ==========

mod impl_core;
mod impl_modulation;
mod traits;
mod types;

#[cfg(test)]
mod tests;

// ========== Re-exports for 100% backwards compatibility ==========

// Re-export from core - DO NOT REDEFINE
pub use context_graph_core::marblestone::{Domain, EdgeType, NeurotransmitterWeights};
pub use context_graph_core::types::NodeId;

// Core types
pub use types::{EdgeId, GraphEdge};
