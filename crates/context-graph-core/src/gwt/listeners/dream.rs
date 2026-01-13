//! Dream Event Listener
//!
//! Queues exiting memories for dream replay consolidation.

use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::gwt::workspace::{WorkspaceEvent, WorkspaceEventListener};

/// Listener that queues exiting memories for dream replay
///
/// When a memory exits the workspace (r dropped below 0.7), it is queued
/// for offline dream replay consolidation.
pub struct DreamEventListener {
    dream_queue: Arc<RwLock<Vec<Uuid>>>,
}

impl DreamEventListener {
    /// Create a new dream event listener with the given queue
    pub fn new(dream_queue: Arc<RwLock<Vec<Uuid>>>) -> Self {
        Self { dream_queue }
    }

    /// Get a clone of the dream queue arc for external access
    pub fn queue(&self) -> Arc<RwLock<Vec<Uuid>>> {
        Arc::clone(&self.dream_queue)
    }
}

impl WorkspaceEventListener for DreamEventListener {
    fn on_event(&self, event: &WorkspaceEvent) {
        match event {
            WorkspaceEvent::MemoryExits {
                id,
                order_parameter,
                timestamp: _,
            } => {
                // Queue memory for dream replay - non-blocking acquire
                match self.dream_queue.try_write() {
                    Ok(mut queue) => {
                        queue.push(*id);
                        tracing::debug!(
                            "Queued memory {:?} for dream replay (r={:.3})",
                            id,
                            order_parameter
                        );
                    }
                    Err(e) => {
                        tracing::error!(
                            "CRITICAL: Failed to acquire dream_queue lock: {:?}",
                            e
                        );
                        panic!("DreamEventListener: Lock poisoned or deadlocked");
                    }
                }
            }
            WorkspaceEvent::IdentityCritical {
                identity_coherence,
                previous_status,
                current_status,
                reason,
                timestamp: _,
            } => {
                // Log identity critical - DreamController handles separately via direct wiring
                tracing::warn!(
                    "Identity critical (IC={:.3}): {} (transition: {} -> {})",
                    identity_coherence,
                    reason,
                    previous_status,
                    current_status,
                );
            }
            // No-op for other events
            WorkspaceEvent::MemoryEnters { .. } => {}
            WorkspaceEvent::WorkspaceConflict { .. } => {}
            WorkspaceEvent::WorkspaceEmpty { .. } => {}
        }
    }
}

impl std::fmt::Debug for DreamEventListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DreamEventListener").finish()
    }
}
