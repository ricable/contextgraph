//! Type definitions for SELF_EGO_NODE
//!
//! Contains constants, basic types, and enums used across the ego_node module.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Maximum purpose vector history size per constitution
/// Reference: constitution.yaml line 390 (identity_trajectory: 1000)
pub const MAX_PV_HISTORY_SIZE: usize = 1000;

/// Default crisis threshold per constitution.yaml line 369
/// IC < 0.7 indicates identity drift (warning/degraded state)
pub const IC_CRISIS_THRESHOLD: f32 = 0.7;

/// Critical threshold triggering dream consolidation per constitution.yaml line 369
/// IC < 0.5 triggers introspective dream (critical state)
pub const IC_CRITICAL_THRESHOLD: f32 = 0.5;

/// Epsilon for numerical stability in magnitude comparisons
/// Prevents division by zero in cosine similarity calculation
pub(crate) const COSINE_EPSILON: f32 = 1e-8;

/// Minimum time between crisis event emissions (30 seconds)
/// Prevents event spam during IC fluctuations
/// Per constitution.yaml: throttle workspace events
/// TASK-IDENTITY-P0-004: Crisis Detection
pub const CRISIS_EVENT_COOLDOWN: Duration = Duration::from_secs(30);

/// Snapshot of purpose vector at a point in time
///
/// # Persistence (TASK-GWT-P1-001)
///
/// Serializable component of SelfEgoNode's identity_trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurposeSnapshot {
    /// Purpose vector at this moment
    pub vector: [f32; 13],
    /// Timestamp of snapshot
    pub timestamp: DateTime<Utc>,
    /// Context (brief description of system state)
    pub context: String,
}

/// Identity status enum for SELF_EGO_NODE state tracking.
///
/// # Persistence (TASK-GWT-P1-001)
///
/// Serializable component of IdentityContinuity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentityStatus {
    /// IC > 0.9: Healthy identity continuity
    Healthy,
    /// 0.7 <= IC <= 0.9: Warning state, monitor closely
    Warning,
    /// IC < 0.7: Degraded identity, may need intervention
    Degraded,
    /// IC < 0.5: Critical, trigger introspective dream
    Critical,
}

/// Result from a self-awareness cycle
#[derive(Debug, Clone)]
pub struct SelfReflectionResult {
    /// Alignment between action and purpose
    pub alignment: f32,
    /// Whether self-reflection should be triggered
    pub needs_reflection: bool,
    /// Current identity status
    pub identity_status: IdentityStatus,
    /// Current identity coherence value
    pub identity_coherence: f32,
}
