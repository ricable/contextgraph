//! Neuromodulation Event Listener
//!
//! Boosts dopamine on memory entry to the workspace.

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::gwt::workspace::{WorkspaceEvent, WorkspaceEventListener};
use crate::neuromod::NeuromodulationManager;

/// Listener that boosts dopamine on memory entry
///
/// When a memory enters the workspace (r crossed 0.8 upward), dopamine
/// is increased by DA_WORKSPACE_INCREMENT (0.2).
pub struct NeuromodulationEventListener {
    neuromod_manager: Arc<RwLock<NeuromodulationManager>>,
}

impl NeuromodulationEventListener {
    /// Create a new neuromodulation event listener
    pub fn new(neuromod_manager: Arc<RwLock<NeuromodulationManager>>) -> Self {
        Self { neuromod_manager }
    }

    /// Get a reference to the neuromod manager arc
    pub fn neuromod(&self) -> Arc<RwLock<NeuromodulationManager>> {
        Arc::clone(&self.neuromod_manager)
    }
}

impl WorkspaceEventListener for NeuromodulationEventListener {
    fn on_event(&self, event: &WorkspaceEvent) {
        match event {
            WorkspaceEvent::MemoryEnters {
                id,
                order_parameter,
                timestamp: _,
                fingerprint: _, // TASK-IDENTITY-P0-006: Not used by neuromod
            } => {
                // Boost dopamine on workspace entry - non-blocking acquire
                match self.neuromod_manager.try_write() {
                    Ok(mut mgr) => {
                        mgr.on_workspace_entry();
                        tracing::debug!(
                            "Dopamine boosted for memory {:?} entering workspace (r={:.3})",
                            id,
                            order_parameter
                        );
                    }
                    Err(e) => {
                        tracing::error!(
                            "CRITICAL: Failed to acquire neuromod_manager lock: {:?}",
                            e
                        );
                        panic!("NeuromodulationEventListener: Lock poisoned or deadlocked");
                    }
                }
            }
            // No-op for other events
            WorkspaceEvent::MemoryExits { .. } => {}
            WorkspaceEvent::WorkspaceConflict { .. } => {}
            WorkspaceEvent::WorkspaceEmpty { .. } => {}
            WorkspaceEvent::IdentityCritical { .. } => {}
        }
    }
}

impl std::fmt::Debug for NeuromodulationEventListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuromodulationEventListener").finish()
    }
}
