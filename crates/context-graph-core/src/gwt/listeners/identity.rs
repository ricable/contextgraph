//! Identity Continuity Event Listener
//!
//! TASK-IDENTITY-P0-006: Wires identity continuity monitoring to GWT workspace attention.
//!
//! Subscribes to `WorkspaceEvent::MemoryEnters` events and computes identity continuity
//! for each memory that enters the Global Workspace.
//!
//! # Algorithm
//!
//! 1. Extract purpose vector from entering memory's teleological fingerprint
//! 2. Compute IC via `IdentityContinuityMonitor::compute_continuity()`
//! 3. Detect crisis state via `detect_crisis()`
//! 4. Execute crisis protocol if status is not Healthy
//! 5. Emit `WorkspaceEvent::IdentityCritical` if critical and cooldown allows
//!
//! # Constitution Reference
//!
//! From constitution.yaml lines 365-392 (gwt.self_ego_node):
//! - IC = cos(PV_t, PV_{t-1}) × r(t)
//! - IC < 0.7: Warning/Degraded (record snapshot)
//! - IC < 0.5: Critical (emit IdentityCritical event, trigger dream)

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::CoreResult;
use crate::gwt::ego_node::{
    CrisisProtocol, IdentityContinuityMonitor, IdentityStatus, SelfEgoNode,
};
use crate::gwt::workspace::{WorkspaceEvent, WorkspaceEventBroadcaster, WorkspaceEventListener};

/// Listener for identity continuity monitoring on workspace events.
///
/// Subscribes to `MemoryEnters` events and computes identity continuity
/// for each memory that enters the Global Workspace.
///
/// # Thread Safety
///
/// All internal state is wrapped in `Arc<RwLock>` for concurrent access.
/// Event processing spawns async tasks to avoid blocking the broadcaster.
///
/// # TASK-IDENTITY-P0-006
pub struct IdentityContinuityListener {
    /// Identity continuity monitor for IC computation
    monitor: Arc<RwLock<IdentityContinuityMonitor>>,
    /// Crisis protocol executor
    protocol: Arc<CrisisProtocol>,
    /// Reference to workspace broadcaster for emitting IdentityCritical events
    broadcaster: Arc<WorkspaceEventBroadcaster>,
}

impl IdentityContinuityListener {
    /// Create new listener with all dependencies.
    ///
    /// # Arguments
    /// * `ego_node` - Arc-wrapped SelfEgoNode for snapshot recording
    /// * `broadcaster` - Arc-wrapped WorkspaceEventBroadcaster for event emission
    ///
    /// # Returns
    /// New `IdentityContinuityListener` instance
    pub fn new(
        ego_node: Arc<RwLock<SelfEgoNode>>,
        broadcaster: Arc<WorkspaceEventBroadcaster>,
    ) -> Self {
        let monitor = Arc::new(RwLock::new(IdentityContinuityMonitor::new()));
        let protocol = Arc::new(CrisisProtocol::new(ego_node));

        Self {
            monitor,
            protocol,
            broadcaster,
        }
    }

    /// Create listener with custom monitor (for testing).
    ///
    /// # Arguments
    /// * `monitor` - Pre-configured IdentityContinuityMonitor
    /// * `ego_node` - Arc-wrapped SelfEgoNode
    /// * `broadcaster` - Arc-wrapped WorkspaceEventBroadcaster
    #[cfg(test)]
    pub fn with_monitor(
        monitor: IdentityContinuityMonitor,
        ego_node: Arc<RwLock<SelfEgoNode>>,
        broadcaster: Arc<WorkspaceEventBroadcaster>,
    ) -> Self {
        let protocol = Arc::new(CrisisProtocol::new(ego_node));

        Self {
            monitor: Arc::new(RwLock::new(monitor)),
            protocol,
            broadcaster,
        }
    }

    /// Get current identity coherence value.
    ///
    /// # Returns
    /// Current IC value (0.0-1.0), or 0.0 if no computation has occurred.
    pub async fn identity_coherence(&self) -> f32 {
        self.monitor
            .read()
            .await
            .identity_coherence()
            .unwrap_or(0.0)
    }

    /// Get current identity status classification.
    ///
    /// # Returns
    /// Current `IdentityStatus`, or `Critical` if no computation has occurred.
    pub async fn identity_status(&self) -> IdentityStatus {
        self.monitor
            .read()
            .await
            .current_status()
            .unwrap_or(IdentityStatus::Critical)
    }

    /// Check if identity is currently in crisis (IC < crisis_threshold).
    ///
    /// # Returns
    /// `true` if IC is below threshold, `false` otherwise.
    pub async fn is_in_crisis(&self) -> bool {
        self.monitor.read().await.is_in_crisis()
    }

    /// Get read access to the internal monitor.
    ///
    /// For advanced use cases that need direct monitor access.
    pub fn monitor(&self) -> Arc<RwLock<IdentityContinuityMonitor>> {
        Arc::clone(&self.monitor)
    }

    /// Get the number of purpose vectors in history.
    pub async fn history_len(&self) -> usize {
        self.monitor.read().await.history_len()
    }

    // === TASK-IDENTITY-P0-007: MCP Tool Exposure Methods ===

    /// Get the last crisis detection result from the monitor.
    ///
    /// Returns `None` if no crisis detection has been performed yet.
    /// This method provides access to cached crisis state for MCP tools
    /// without triggering a new detection cycle.
    ///
    /// # TASK-IDENTITY-P0-007
    pub async fn last_detection(&self) -> Option<crate::gwt::ego_node::CrisisDetectionResult> {
        self.monitor.read().await.last_detection()
    }
}

impl WorkspaceEventListener for IdentityContinuityListener {
    fn on_event(&self, event: &WorkspaceEvent) {
        // Clone Arcs for the spawned async task
        let monitor = Arc::clone(&self.monitor);
        let protocol = Arc::clone(&self.protocol);
        let broadcaster = Arc::clone(&self.broadcaster);
        let event = event.clone();

        // Spawn async task to process event without blocking
        tokio::spawn(async move {
            // Create temporary listener wrapper for process_event
            let temp_listener = IdentityContinuityListenerInner {
                monitor,
                protocol,
                broadcaster,
            };

            if let Err(e) = temp_listener.process_event(&event).await {
                tracing::error!(
                    error = %e,
                    "Failed to process workspace event for identity continuity"
                );
            }
        });
    }
}

/// Inner struct for async processing in spawned tasks.
///
/// This avoids self-reference issues when spawning async tasks from `on_event`.
struct IdentityContinuityListenerInner {
    monitor: Arc<RwLock<IdentityContinuityMonitor>>,
    protocol: Arc<CrisisProtocol>,
    broadcaster: Arc<WorkspaceEventBroadcaster>,
}

impl IdentityContinuityListenerInner {
    /// Process a workspace event asynchronously.
    async fn process_event(&self, event: &WorkspaceEvent) -> CoreResult<()> {
        match event {
            WorkspaceEvent::MemoryEnters {
                id,
                fingerprint,
                order_parameter,
                ..
            } => {
                // Extract purpose vector from fingerprint
                let pv = match fingerprint {
                    Some(fp) => fp.purpose_vector.alignments,
                    None => {
                        tracing::debug!(
                            memory_id = %id,
                            "Memory entered without fingerprint, skipping IC computation"
                        );
                        return Ok(());
                    }
                };

                // Use provided order_parameter as Kuramoto r
                let kuramoto_r = *order_parameter;

                // Compute identity continuity (requires write lock)
                let mut monitor = self.monitor.write().await;
                let ic_result = monitor.compute_continuity(
                    &pv,
                    kuramoto_r,
                    format!("MemoryEnters:{}", id),
                );

                tracing::trace!(
                    ic = %ic_result.identity_coherence,
                    status = ?ic_result.status,
                    memory_id = %id,
                    "Identity continuity computed"
                );

                // Detect crisis state
                let detection = monitor.detect_crisis();

                // Execute crisis protocol if not Healthy
                if detection.current_status != IdentityStatus::Healthy {
                    let protocol_result = self.protocol.execute(detection.clone(), &mut monitor).await?;

                    // Emit IdentityCritical event if critical and cooldown allows
                    if protocol_result.event_emitted {
                        if let Some(crisis_event) = protocol_result.event {
                            let ws_event = crisis_event.to_workspace_event();
                            self.broadcaster.broadcast(ws_event).await;

                            tracing::warn!(
                                ic = %detection.identity_coherence,
                                status = ?detection.current_status,
                                "Identity crisis event emitted"
                            );
                        }
                    }
                }

                Ok(())
            }
            // Ignore all other event types
            _ => Ok(()),
        }
    }
}

impl std::fmt::Debug for IdentityContinuityListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IdentityContinuityListener").finish()
    }
}
