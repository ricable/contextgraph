//! Constants for UTL computation handlers.
//!
//! TASK-S005: Embedder names and constitution.yaml targets.
//! TASK-FIX-003: Fixed to match canonical Embedder::name() values.

use context_graph_core::johari::NUM_EMBEDDERS;

/// Embedder names for trajectory reporting.
/// 13 embedders: E1-E13.
///
/// CANONICAL SOURCE: `Embedder::name()` in `context-graph-core/src/teleological/embedder.rs`
/// These names MUST match exactly. See TASK-FIX-003 for rationale.
pub(super) const EMBEDDER_NAMES: [&str; NUM_EMBEDDERS] = [
    "E1_Semantic",          // E1
    "E2_Temporal_Recent",   // E2
    "E3_Temporal_Periodic", // E3
    "E4_Temporal_Positional", // E4
    "E5_Causal",            // E5
    "E6_Sparse_Lexical",    // E6
    "E7_Code",              // E7
    "E8_Emotional",         // E8
    "E9_HDC",               // E9
    "E10_Multimodal",       // E10
    "E11_Entity",           // E11
    "E12_Late_Interaction", // E12
    "E13_SPLADE",           // E13
];

/// Constitution.yaml targets (hardcoded per TASK-S005 spec).
pub(super) const LEARNING_SCORE_TARGET: f32 = 0.6;
pub(super) const COHERENCE_RECOVERY_TARGET_MS: u64 = 10000;
pub(super) const ATTACK_DETECTION_TARGET: f32 = 0.95;
pub(super) const FALSE_POSITIVE_TARGET: f32 = 0.02;

/// ΔC formula weights per constitution.yaml line 166.
pub(super) const ALPHA: f32 = 0.4; // Connectivity weight
pub(super) const BETA: f32 = 0.4;  // ClusterFit weight
pub(super) const GAMMA: f32 = 0.2; // Consistency weight
