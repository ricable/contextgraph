//! Graph edge types optimized for RocksDB storage.
//!
//! This module defines the core GraphEdge struct with Marblestone neurotransmitter
//! modulation and steering-based weight adjustment for domain-aware graph traversal.
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

use serde::{Deserialize, Serialize};

// Re-export from core - DO NOT REDEFINE
pub use context_graph_core::marblestone::{Domain, EdgeType, NeurotransmitterWeights};
pub use context_graph_core::types::NodeId;

/// Edge identifier for graph crate (i64 for RocksDB key efficiency).
///
/// Uses i64 instead of UUID for:
/// - Efficient RocksDB key encoding (8 bytes vs 16)
/// - FAISS ID compatibility
/// - Simpler range scans
pub type EdgeId = i64;

/// Graph edge with Marblestone neuro-modulation and steering support.
///
/// Optimized for RocksDB storage with bincode serialization.
/// Uses u64 Unix timestamps for efficiency (not chrono DateTime).
///
/// # CANONICAL FORMULA for modulated weight
///
/// ```text
/// net_activation = excitatory - inhibitory + (modulatory * 0.5)
/// domain_bonus = 0.1 if edge_domain == query_domain else 0.0
/// steering_factor = 0.5 + steering_reward
/// w_eff = base_weight * (1.0 + net_activation + domain_bonus) * steering_factor
/// ```
/// Result is clamped to [0.0, 1.0]
///
/// # Fields (13 total per PRD Section 4.2)
///
/// 1. id - Unique edge identifier (i64)
/// 2. source - Source node ID (UUID)
/// 3. target - Target node ID (UUID)
/// 4. edge_type - Relationship classification
/// 5. weight - Base edge weight [0.0, 1.0]
/// 6. confidence - Certainty score [0.0, 1.0]
/// 7. domain - Cognitive domain
/// 8. neurotransmitter_weights - NT modulation weights
/// 9. is_amortized_shortcut - Dream-learned shortcut flag
/// 10. steering_reward - Steering subsystem feedback [0.0, 1.0]
/// 11. traversal_count - Access count
/// 12. created_at - Unix timestamp of creation
/// 13. last_traversed_at - Unix timestamp of last access (0 = never)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Unique edge identifier (i64 for RocksDB efficiency).
    pub id: EdgeId,

    /// Source node ID (UUID).
    pub source: NodeId,

    /// Target node ID (UUID).
    pub target: NodeId,

    /// Edge type classification.
    pub edge_type: EdgeType,

    /// Base edge weight [0.0, 1.0].
    /// Higher = stronger relationship.
    pub weight: f32,

    /// Confidence score [0.0, 1.0].
    /// How certain we are about this edge.
    pub confidence: f32,

    /// Cognitive domain of this edge.
    pub domain: Domain,

    /// Neurotransmitter weights for modulation.
    pub neurotransmitter_weights: NeurotransmitterWeights,

    /// Whether this is an amortized inference shortcut
    /// (learned pattern that bypasses intermediate steps).
    pub is_amortized_shortcut: bool,

    /// Steering reward from traversal history [0.0, 1.0].
    /// Updated via EMA when edge leads to successful retrieval.
    /// NOTE: Constitution says [-1,1] but we use [0,1] for steering_factor math.
    pub steering_reward: f32,

    /// Number of times this edge has been traversed.
    pub traversal_count: u64,

    /// Unix timestamp when edge was created.
    pub created_at: u64,

    /// Unix timestamp of last traversal (0 if never).
    pub last_traversed_at: u64,
}
