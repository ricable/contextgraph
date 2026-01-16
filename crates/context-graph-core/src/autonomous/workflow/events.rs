//! Optimization event types.
//!
//! TASK-P0-005: Removed NorthStarUpdated variant per ARCH-03 (autonomous operation).

use serde::{Deserialize, Serialize};

use crate::autonomous::bootstrap::GoalId;
use crate::autonomous::curation::MemoryId;

use super::schedule::ScheduledCheckType;

/// Optimization event trigger
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OptimizationEvent {
    /// A new memory was stored
    MemoryStored { memory_id: MemoryId },
    /// A memory was retrieved
    MemoryRetrieved { memory_id: MemoryId, query: String },
    // REMOVED: NorthStarUpdated per TASK-P0-005 (ARCH-03)
    /// A new goal was added
    GoalAdded { goal_id: GoalId },
    /// Consciousness level dropped below threshold
    ConsciousnessDropped { level: f32 },
    /// A scheduled check is due
    ScheduledCheck { check_type: ScheduledCheckType },
}

impl OptimizationEvent {
    /// Get a descriptive name for this event type
    pub fn event_type_name(&self) -> &'static str {
        match self {
            Self::MemoryStored { .. } => "memory_stored",
            Self::MemoryRetrieved { .. } => "memory_retrieved",
            // REMOVED: NorthStarUpdated per TASK-P0-005 (ARCH-03)
            Self::GoalAdded { .. } => "goal_added",
            Self::ConsciousnessDropped { .. } => "consciousness_dropped",
            Self::ScheduledCheck { .. } => "scheduled_check",
        }
    }

    /// Check if this event requires immediate processing
    pub fn is_urgent(&self) -> bool {
        // TASK-P0-005: Removed NorthStarUpdated from urgent events
        matches!(self, Self::ConsciousnessDropped { .. })
    }
}
