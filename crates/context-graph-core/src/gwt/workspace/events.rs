//! Workspace Events - Event broadcasting and listener system
//!
//! Implements workspace state change events and the event broadcaster
//! for notifying subsystems of consciousness state transitions.

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::types::fingerprint::TeleologicalFingerprint;

/// Events fired by workspace state changes
#[derive(Debug, Clone)]
pub enum WorkspaceEvent {
    /// Memory entered workspace (r crossed 0.8 upward)
    ///
    /// # TASK-IDENTITY-P0-006: Added fingerprint field
    ///
    /// The fingerprint is used by `IdentityContinuityListener` to extract
    /// the purpose vector for identity continuity computation:
    /// - IC = cos(PV_t, PV_{t-1}) × r(t)
    MemoryEnters {
        id: Uuid,
        order_parameter: f32,
        timestamp: DateTime<Utc>,
        /// Teleological fingerprint of entering memory for IC computation.
        /// None if memory doesn't have a fingerprint (legacy compatibility during migration).
        /// Boxed to reduce enum variant size (TeleologicalFingerprint is ~1.6KB).
        fingerprint: Option<Box<TeleologicalFingerprint>>,
    },
    /// Memory exited workspace (r dropped below 0.7)
    MemoryExits {
        id: Uuid,
        order_parameter: f32,
        timestamp: DateTime<Utc>,
    },
    /// Multiple memories competing for workspace (conflict)
    WorkspaceConflict {
        memories: Vec<Uuid>,
        timestamp: DateTime<Utc>,
    },
    /// No memory in workspace for extended time
    WorkspaceEmpty {
        duration_ms: u64,
        timestamp: DateTime<Utc>,
    },
    /// Identity coherence critical (IC < 0.5) - triggers dream consolidation
    /// From constitution.yaml lines 387-392: "dream<0.5"
    ///
    /// # TASK-IDENTITY-P0-005: Added previous_status and current_status
    IdentityCritical {
        identity_coherence: f32,
        /// Status before crisis (e.g., "Healthy", "Warning", "Degraded")
        previous_status: String,
        /// Current status (should be "Critical")
        current_status: String,
        reason: String,
        timestamp: DateTime<Utc>,
    },
}

/// Trait for workspace event listeners
pub trait WorkspaceEventListener: Send + Sync {
    fn on_event(&self, event: &WorkspaceEvent);
}

/// Broadcasts workspace events to subsystems
pub struct WorkspaceEventBroadcaster {
    listeners: std::sync::Arc<tokio::sync::RwLock<Vec<Box<dyn WorkspaceEventListener>>>>,
}

impl std::fmt::Debug for WorkspaceEventBroadcaster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkspaceEventBroadcaster").finish()
    }
}

impl WorkspaceEventBroadcaster {
    pub fn new() -> Self {
        Self {
            listeners: std::sync::Arc::new(tokio::sync::RwLock::new(Vec::new())),
        }
    }

    /// Register a listener for workspace events
    ///
    /// # Arguments
    /// * `listener` - A boxed trait object implementing WorkspaceEventListener
    ///
    /// # Panics
    /// Panics if unable to acquire write lock (indicates deadlock or poisoned lock)
    pub async fn register_listener(&self, listener: Box<dyn WorkspaceEventListener>) {
        let mut listeners = self.listeners.write().await;
        tracing::debug!(
            "Registering workspace event listener (total: {})",
            listeners.len() + 1
        );
        listeners.push(listener);
    }

    /// Get the number of registered listeners
    pub async fn listener_count(&self) -> usize {
        let listeners = self.listeners.read().await;
        listeners.len()
    }

    pub async fn broadcast(&self, event: WorkspaceEvent) {
        let listeners = self.listeners.read().await;
        for listener in listeners.iter() {
            listener.on_event(&event);
        }
    }
}

impl Default for WorkspaceEventBroadcaster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test listener for verification
    struct TestListener {
        event_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl TestListener {
        fn new() -> Self {
            Self {
                event_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }
        }

        #[allow(dead_code)]
        fn count(&self) -> usize {
            self.event_count.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl WorkspaceEventListener for TestListener {
        fn on_event(&self, _event: &WorkspaceEvent) {
            self.event_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn test_register_listener() {
        println!("=== FSV: WorkspaceEventBroadcaster::register_listener ===");

        let broadcaster = WorkspaceEventBroadcaster::new();

        // BEFORE
        let before_count = broadcaster.listener_count().await;
        println!("BEFORE: listener_count = {}", before_count);
        assert_eq!(before_count, 0, "Should start with 0 listeners");

        // EXECUTE
        let listener = TestListener::new();
        broadcaster.register_listener(Box::new(listener)).await;

        // AFTER
        let after_count = broadcaster.listener_count().await;
        println!("AFTER: listener_count = {}", after_count);
        assert_eq!(after_count, 1, "Should have 1 listener after registration");

        // EVIDENCE
        println!("EVIDENCE: Listener correctly registered");
    }

    #[tokio::test]
    async fn test_broadcast_to_registered_listeners() {
        println!("=== FSV: Broadcast reaches registered listeners ===");

        let broadcaster = WorkspaceEventBroadcaster::new();
        let listener = TestListener::new();
        let count_ref = std::sync::Arc::clone(&listener.event_count);

        broadcaster.register_listener(Box::new(listener)).await;

        // BEFORE
        let before = count_ref.load(std::sync::atomic::Ordering::SeqCst);
        println!("BEFORE: event_count = {}", before);

        // EXECUTE
        let event = WorkspaceEvent::MemoryEnters {
            id: Uuid::new_v4(),
            order_parameter: 0.85,
            timestamp: Utc::now(),
            fingerprint: None, // TASK-IDENTITY-P0-006: Test without fingerprint
        };
        broadcaster.broadcast(event).await;

        // AFTER
        let after = count_ref.load(std::sync::atomic::Ordering::SeqCst);
        println!("AFTER: event_count = {}", after);

        assert_eq!(after, 1, "Listener should receive exactly 1 event");
        println!("EVIDENCE: Event correctly broadcast to listener");
    }

    #[tokio::test]
    async fn test_broadcast_empty_listeners() {
        println!("=== EDGE CASE: Broadcast with no listeners ===");

        let broadcaster = WorkspaceEventBroadcaster::new();

        // Should not panic
        let event = WorkspaceEvent::MemoryEnters {
            id: Uuid::new_v4(),
            order_parameter: 0.85,
            timestamp: Utc::now(),
            fingerprint: None, // TASK-IDENTITY-P0-006: Test without fingerprint
        };
        broadcaster.broadcast(event).await;

        println!("EVIDENCE: Broadcast with no listeners does not panic");
    }

    #[tokio::test]
    async fn test_multiple_listeners() {
        println!("=== TEST: Multiple listeners receive events ===");

        let broadcaster = WorkspaceEventBroadcaster::new();

        let listener1 = TestListener::new();
        let count1 = std::sync::Arc::clone(&listener1.event_count);

        let listener2 = TestListener::new();
        let count2 = std::sync::Arc::clone(&listener2.event_count);

        broadcaster.register_listener(Box::new(listener1)).await;
        broadcaster.register_listener(Box::new(listener2)).await;

        assert_eq!(broadcaster.listener_count().await, 2);

        // Broadcast event
        let event = WorkspaceEvent::MemoryEnters {
            id: Uuid::new_v4(),
            order_parameter: 0.85,
            timestamp: Utc::now(),
            fingerprint: None, // TASK-IDENTITY-P0-006: Test without fingerprint
        };
        broadcaster.broadcast(event).await;

        // Both listeners should receive the event
        assert_eq!(count1.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(count2.load(std::sync::atomic::Ordering::SeqCst), 1);

        println!("EVIDENCE: Both listeners received the event");
    }
}
