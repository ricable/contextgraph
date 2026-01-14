//! Meta-Cognitive Event Listener
//!
//! Triggers epistemic action on workspace empty.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::gwt::meta_cognitive::MetaCognitiveLoop;
use crate::gwt::workspace::{WorkspaceEvent, WorkspaceEventListener};

/// Minimum duration (ms) workspace must be empty before triggering epistemic action.
/// PRD Section 2.5.3: "workspace_empty: No memory r > 0.8 for 5s"
/// Constitution Rule: gwt.workspace.events.empty_5s
pub const WORKSPACE_EMPTY_THRESHOLD_MS: u64 = 5000;

/// Listener that triggers epistemic action on workspace empty
///
/// When the workspace is empty for an extended period, an epistemic action
/// flag is set to trigger exploratory behavior.
pub struct MetaCognitiveEventListener {
    meta_cognitive: Arc<RwLock<MetaCognitiveLoop>>,
    epistemic_action_triggered: Arc<AtomicBool>,
}

impl MetaCognitiveEventListener {
    /// Create a new meta-cognitive event listener
    pub fn new(
        meta_cognitive: Arc<RwLock<MetaCognitiveLoop>>,
        epistemic_action_triggered: Arc<AtomicBool>,
    ) -> Self {
        Self {
            meta_cognitive,
            epistemic_action_triggered,
        }
    }

    /// Check if epistemic action has been triggered
    pub fn is_epistemic_action_triggered(&self) -> bool {
        self.epistemic_action_triggered.load(Ordering::SeqCst)
    }

    /// Reset the epistemic action flag
    pub fn reset_epistemic_action(&self) {
        self.epistemic_action_triggered.store(false, Ordering::SeqCst);
    }

    /// Get a reference to the meta-cognitive loop arc
    pub fn meta_cognitive(&self) -> Arc<RwLock<MetaCognitiveLoop>> {
        Arc::clone(&self.meta_cognitive)
    }
}

impl WorkspaceEventListener for MetaCognitiveEventListener {
    fn on_event(&self, event: &WorkspaceEvent) {
        match event {
            WorkspaceEvent::WorkspaceEmpty {
                duration_ms,
                timestamp: _,
            } => {
                if *duration_ms >= WORKSPACE_EMPTY_THRESHOLD_MS {
                    // Threshold met - trigger epistemic action
                    self.epistemic_action_triggered.store(true, Ordering::SeqCst);
                    tracing::info!(
                        "GWT: Workspace empty for {}ms >= {}ms threshold - epistemic action triggered",
                        duration_ms,
                        WORKSPACE_EMPTY_THRESHOLD_MS
                    );
                } else {
                    // Threshold not met - log but do not trigger
                    tracing::debug!(
                        "GWT: Workspace empty for {}ms < {}ms threshold - waiting",
                        duration_ms,
                        WORKSPACE_EMPTY_THRESHOLD_MS
                    );
                }
            }
            // No-op for other events
            WorkspaceEvent::MemoryEnters { .. } => {}
            WorkspaceEvent::MemoryExits { .. } => {}
            WorkspaceEvent::WorkspaceConflict { .. } => {}
            WorkspaceEvent::IdentityCritical { .. } => {}
        }
    }
}

impl std::fmt::Debug for MetaCognitiveEventListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetaCognitiveEventListener")
            .field(
                "epistemic_action_triggered",
                &self.epistemic_action_triggered.load(Ordering::SeqCst),
            )
            .finish()
    }
}
