//! SELF_EGO_NODE - System Identity and Self-Awareness
//!
//! Implements persistent system identity node as specified in Constitution v4.0.0
//! Section gwt.self_ego_node (lines 371-392).
//!
//! The SELF_EGO_NODE represents the system's understanding of itself:
//! - Current system state (TeleologicalFingerprint)
//! - System's purpose alignment (PurposeVector)
//! - Identity history (trajectory of purpose evolution)
//! - Alignment between actions and self-model
//!
//! # Persistence (TASK-GWT-P1-001)
//!
//! SelfEgoNode and related types implement Serde Serialize/Deserialize for
//! persistent storage in RocksDB via the CF_EGO_NODE column family.
//!
//! # Module Structure
//!
//! - `types`: Constants, basic types (PurposeSnapshot, IdentityStatus)
//! - `self_ego_node`: SelfEgoNode struct and implementation
//! - `purpose_vector_history`: PurposeVectorHistory and provider trait
//! - `identity_continuity`: IdentityContinuity tracking
//! - `cosine`: Cosine similarity computation
//! - `monitor`: IdentityContinuityMonitor
//! - `awareness_loop`: SelfAwarenessLoop implementation

mod awareness_loop;
mod cosine;
mod crisis_protocol;
mod identity_continuity;
mod monitor;
mod purpose_vector_history;
mod self_ego_node;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use awareness_loop::SelfAwarenessLoop;
pub use cosine::cosine_similarity_13d;
pub use crisis_protocol::{
    CrisisAction, CrisisProtocol, CrisisProtocolResult, IdentityCrisisEvent,
};
pub use identity_continuity::IdentityContinuity;
pub use monitor::{CrisisDetectionResult, IdentityContinuityMonitor};
pub use purpose_vector_history::{PurposeVectorHistory, PurposeVectorHistoryProvider};
pub use self_ego_node::SelfEgoNode;
pub use types::{
    IdentityStatus, PurposeSnapshot, SelfReflectionResult, CRISIS_EVENT_COOLDOWN,
    IC_CRISIS_THRESHOLD, IC_CRITICAL_THRESHOLD, MAX_PV_HISTORY_SIZE,
};
