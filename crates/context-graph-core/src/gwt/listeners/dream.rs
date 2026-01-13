//! Dream Event Listener
//!
//! Queues exiting memories for dream replay consolidation and triggers
//! dream cycles on identity crisis events.
//!
//! # Constitution Compliance
//!
//! - AP-26: "IC<0.5 MUST trigger dream - no silent failures"
//! - AP-38: "IC<0.5 MUST auto-trigger dream"
//! - IDENTITY-007: "IC < 0.5 → auto-trigger dream"

use std::sync::Arc;
use parking_lot::Mutex;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::dream::{ExtendedTriggerReason, TriggerManager};
use crate::gwt::workspace::{WorkspaceEvent, WorkspaceEventListener};

/// Callback type for dream consolidation signaling.
///
/// Invoked when TriggerManager determines a dream cycle should start.
/// The callback receives the trigger reason (e.g., IdentityCritical with IC value).
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use context_graph_core::gwt::listeners::DreamConsolidationCallback;
///
/// let callback: DreamConsolidationCallback = Arc::new(|reason| {
///     println!("Dream triggered: {:?}", reason);
/// });
/// ```
pub type DreamConsolidationCallback = Arc<dyn Fn(ExtendedTriggerReason) + Send + Sync>;

/// Listener that queues exiting memories for dream replay and triggers
/// dream consolidation on identity crisis.
///
/// # Constitution Compliance
///
/// Per AP-26, AP-38, and IDENTITY-007: When IC < 0.5, this listener MUST
/// trigger dream consolidation. GPU monitoring failures during IC crisis
/// are treated as fatal per AP-26 (no silent failures).
///
/// # Usage
///
/// ## Basic (queue-only, backwards compatible):
/// ```ignore
/// use std::sync::Arc;
/// use tokio::sync::RwLock;
/// use context_graph_core::gwt::listeners::DreamEventListener;
///
/// let queue = Arc::new(RwLock::new(Vec::new()));
/// let listener = DreamEventListener::new(queue);
/// ```
///
/// ## With TriggerManager integration:
/// ```ignore
/// use std::sync::Arc;
/// use parking_lot::Mutex;
/// use tokio::sync::RwLock;
/// use context_graph_core::dream::TriggerManager;
/// use context_graph_core::gwt::listeners::DreamEventListener;
///
/// let queue = Arc::new(RwLock::new(Vec::new()));
/// let trigger_manager = Arc::new(Mutex::new(TriggerManager::new()));
/// let callback = Arc::new(|reason| println!("Dream: {:?}", reason));
///
/// let listener = DreamEventListener::new(queue)
///     .with_trigger_manager(trigger_manager)
///     .with_consolidation_callback(callback);
/// ```
pub struct DreamEventListener {
    /// Queue for memories exiting workspace (for dream replay)
    dream_queue: Arc<RwLock<Vec<Uuid>>>,

    /// Optional TriggerManager for IC-based dream triggering
    /// None = backwards-compatible mode (log only for IdentityCritical)
    trigger_manager: Option<Arc<Mutex<TriggerManager>>>,

    /// Optional callback for dream consolidation signaling
    /// Called when TriggerManager returns Some(reason)
    consolidation_callback: Option<DreamConsolidationCallback>,
}

impl DreamEventListener {
    /// Create a new dream event listener with the given queue.
    ///
    /// This creates a listener in backwards-compatible mode:
    /// - MemoryExits: Queued for dream replay
    /// - IdentityCritical: Logged but no trigger (no TriggerManager)
    ///
    /// Use `with_trigger_manager()` to enable IC-based dream triggering.
    pub fn new(dream_queue: Arc<RwLock<Vec<Uuid>>>) -> Self {
        Self {
            dream_queue,
            trigger_manager: None,
            consolidation_callback: None,
        }
    }

    /// Add TriggerManager integration for IC-based dream triggering.
    ///
    /// When wired, IdentityCritical events will:
    /// 1. Update TriggerManager with IC value
    /// 2. Check if triggers fire
    /// 3. Call consolidation callback if trigger activates
    ///
    /// # Arguments
    ///
    /// * `trigger_manager` - Shared TriggerManager instance
    ///
    /// # Example
    ///
    /// ```ignore
    /// let manager = Arc::new(Mutex::new(TriggerManager::new()));
    /// let listener = DreamEventListener::new(queue)
    ///     .with_trigger_manager(manager);
    /// ```
    pub fn with_trigger_manager(mut self, trigger_manager: Arc<Mutex<TriggerManager>>) -> Self {
        self.trigger_manager = Some(trigger_manager);
        self
    }

    /// Add callback for dream consolidation signaling.
    ///
    /// The callback is invoked when TriggerManager determines a dream
    /// cycle should start (e.g., IC < 0.5 threshold).
    ///
    /// # Arguments
    ///
    /// * `callback` - Function to call with trigger reason
    ///
    /// # Example
    ///
    /// ```ignore
    /// let callback = Arc::new(|reason| {
    ///     match reason {
    ///         ExtendedTriggerReason::IdentityCritical { ic_value } => {
    ///             println!("IC crisis at {:.3}, starting dream", ic_value);
    ///         }
    ///         _ => println!("Dream trigger: {:?}", reason),
    ///     }
    /// });
    /// let listener = DreamEventListener::new(queue)
    ///     .with_consolidation_callback(callback);
    /// ```
    pub fn with_consolidation_callback(mut self, callback: DreamConsolidationCallback) -> Self {
        self.consolidation_callback = Some(callback);
        self
    }

    /// Get a clone of the dream queue arc for external access.
    ///
    /// Use this to inspect or drain the queue after events are processed.
    pub fn queue(&self) -> Arc<RwLock<Vec<Uuid>>> {
        Arc::clone(&self.dream_queue)
    }

    /// Handle identity critical event - updates TriggerManager and checks triggers.
    ///
    /// # Constitution Compliance
    ///
    /// Per AP-26, AP-38, IDENTITY-007:
    /// - IC < threshold MUST trigger dream consolidation
    /// - Lock failures are fatal (indicates deadlock/poison)
    ///
    /// # Arguments
    ///
    /// * `identity_coherence` - Current IC value [0.0, 1.0]
    /// * `previous_status` - Status before transition
    /// * `current_status` - Current status
    /// * `reason` - Human-readable reason for crisis
    ///
    /// # Panics
    ///
    /// Panics if lock acquisition fails (AP-26: no silent failures).
    fn handle_identity_critical(
        &self,
        identity_coherence: f32,
        previous_status: &str,
        current_status: &str,
        reason: &str,
    ) {
        // Always log the IC event
        tracing::warn!(
            "Identity critical (IC={:.3}): {} (transition: {} -> {})",
            identity_coherence,
            reason,
            previous_status,
            current_status,
        );

        // Only process through TriggerManager if wired
        if let Some(ref trigger_manager) = self.trigger_manager {
            // Use parking_lot Mutex - never poisons, blocking is fine in sync context
            let mut manager = trigger_manager.lock();

            // Update IC in TriggerManager
            manager.update_identity_coherence(identity_coherence);

            // Check if any trigger fires (IC, entropy, GPU, manual)
            if let Some(trigger_reason) = manager.check_triggers() {
                tracing::info!(
                    "Dream trigger activated: {:?} (IC={:.3})",
                    trigger_reason,
                    identity_coherence
                );

                // Mark as triggered to start cooldown
                manager.mark_triggered(trigger_reason);

                // Invoke consolidation callback if configured
                if let Some(ref callback) = self.consolidation_callback {
                    callback(trigger_reason);
                }
            } else {
                tracing::debug!(
                    "No dream trigger (IC={:.3}, cooldown or above threshold)",
                    identity_coherence
                );
            }
        }
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
                        // AP-26: Lock failure is fatal
                        tracing::error!(
                            "CRITICAL: Failed to acquire dream_queue lock: {:?}",
                            e
                        );
                        panic!("DreamEventListener: Lock poisoned or deadlocked - cannot queue memory {:?}", id);
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
                // Delegate to handler which manages TriggerManager integration
                self.handle_identity_critical(
                    *identity_coherence,
                    previous_status,
                    current_status,
                    reason,
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
        f.debug_struct("DreamEventListener")
            .field("has_trigger_manager", &self.trigger_manager.is_some())
            .field("has_callback", &self.consolidation_callback.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::Duration;

    use crate::dream::TriggerConfig;

    // ============================================================
    // FSV Tests for DreamEventListener
    // ============================================================

    #[tokio::test]
    async fn test_fsv_dream_listener_memory_exits() {
        println!("=== FSV: DreamEventListener - MemoryExits ===");

        // SETUP
        let dream_queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(dream_queue.clone());
        let memory_id = Uuid::new_v4();

        // BEFORE
        let before_len = {
            let queue = dream_queue.read().await;
            queue.len()
        };
        println!("BEFORE: queue.len() = {}", before_len);
        assert_eq!(before_len, 0, "Queue must start empty");

        // EXECUTE
        let event = WorkspaceEvent::MemoryExits {
            id: memory_id,
            order_parameter: 0.65,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        // AFTER - SEPARATE READ
        let after_len = {
            let queue = dream_queue.read().await;
            queue.len()
        };
        let queued_id = {
            let queue = dream_queue.read().await;
            queue.first().cloned()
        };
        println!("AFTER: queue.len() = {}", after_len);

        // VERIFY
        assert_eq!(after_len, 1, "Queue must have exactly 1 item");
        assert_eq!(queued_id, Some(memory_id), "Queued ID must match");

        // EVIDENCE
        println!("EVIDENCE: Memory {:?} correctly queued for dream replay", memory_id);
    }

    #[tokio::test]
    async fn test_dream_listener_ignores_other_events() {
        println!("=== TEST: DreamEventListener ignores non-MemoryExits ===");

        let dream_queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(dream_queue.clone());

        // Send MemoryEnters - should be ignored
        let event = WorkspaceEvent::MemoryEnters {
            id: Uuid::new_v4(),
            order_parameter: 0.85,
            timestamp: Utc::now(),
            fingerprint: None,
        };
        listener.on_event(&event);

        // Send WorkspaceEmpty - should be ignored
        let event = WorkspaceEvent::WorkspaceEmpty {
            duration_ms: 1000,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        let queue_len = {
            let queue = dream_queue.read().await;
            queue.len()
        };

        assert_eq!(queue_len, 0, "Queue should remain empty for non-MemoryExits events");
        println!("EVIDENCE: DreamEventListener correctly ignores non-MemoryExits events");
    }

    #[tokio::test]
    async fn test_dream_listener_identity_critical_without_trigger_manager() {
        println!("=== TEST: DreamEventListener handles IdentityCritical (no TriggerManager) ===");

        let dream_queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(dream_queue.clone());

        // Send IdentityCritical - should log but not queue (no trigger manager)
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.35,
            previous_status: "Warning".to_string(),
            current_status: "Critical".to_string(),
            reason: "Test critical".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        let queue_len = {
            let queue = dream_queue.read().await;
            queue.len()
        };

        assert_eq!(queue_len, 0, "Queue should remain empty for IdentityCritical");
        println!("EVIDENCE: IdentityCritical event handled without queuing (no TriggerManager)");
    }

    // ============================================================
    // TASK-24: TriggerManager Integration Tests
    // ============================================================

    #[test]
    fn test_ic_crisis_triggers_dream_consolidation() {
        println!("=== FSV: IC crisis triggers dream consolidation ===");

        // SETUP: TriggerManager with IC threshold 0.5 (constitution default)
        let config = TriggerConfig::default()
            .with_cooldown(Duration::from_millis(1)); // Short cooldown for test
        let manager = Arc::new(Mutex::new(TriggerManager::with_config(config)));

        // Track callback invocation
        let callback_called = Arc::new(AtomicBool::new(false));
        let callback_ic = Arc::new(AtomicU32::new(0));
        let cb_called = Arc::clone(&callback_called);
        let cb_ic = Arc::clone(&callback_ic);

        let callback: DreamConsolidationCallback = Arc::new(move |reason| {
            cb_called.store(true, Ordering::SeqCst);
            if let ExtendedTriggerReason::IdentityCritical { ic_value } = reason {
                cb_ic.store(ic_value.to_bits(), Ordering::SeqCst);
            }
        });

        // Create listener with TriggerManager
        let queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(queue)
            .with_trigger_manager(manager)
            .with_consolidation_callback(callback);

        // BEFORE
        println!("BEFORE: callback_called = {}", callback_called.load(Ordering::SeqCst));
        assert!(!callback_called.load(Ordering::SeqCst), "Callback should not be called yet");

        // EXECUTE: Emit IC crisis event (IC=0.3 < threshold 0.5)
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.3,
            previous_status: "Stable".to_string(),
            current_status: "Critical".to_string(),
            reason: "Test IC crisis".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        // AFTER: VERIFY callback was invoked
        println!("AFTER: callback_called = {}", callback_called.load(Ordering::SeqCst));
        assert!(
            callback_called.load(Ordering::SeqCst),
            "Consolidation callback MUST be called when IC < threshold"
        );

        // VERIFY: Correct IC value passed
        let stored_ic = f32::from_bits(callback_ic.load(Ordering::SeqCst));
        println!("EVIDENCE: Callback received IC value: {:.3}", stored_ic);
        assert!(
            (stored_ic - 0.3).abs() < 0.001,
            "Callback MUST receive correct IC value, got {}",
            stored_ic
        );
    }

    #[test]
    fn test_ic_above_threshold_no_trigger() {
        println!("=== FSV: IC above threshold does NOT trigger ===");

        let config = TriggerConfig::default()
            .with_cooldown(Duration::from_millis(1));
        let manager = Arc::new(Mutex::new(TriggerManager::with_config(config)));

        let callback_called = Arc::new(AtomicBool::new(false));
        let cb = Arc::clone(&callback_called);
        let callback: DreamConsolidationCallback = Arc::new(move |_| {
            cb.store(true, Ordering::SeqCst);
        });

        let queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(queue)
            .with_trigger_manager(manager)
            .with_consolidation_callback(callback);

        // BEFORE
        println!("BEFORE: callback_called = {}", callback_called.load(Ordering::SeqCst));

        // IC=0.7 > threshold 0.5, should NOT trigger
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.7,
            previous_status: "Stable".to_string(),
            current_status: "Warning".to_string(),
            reason: "Test warning".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        // AFTER
        println!("AFTER: callback_called = {}", callback_called.load(Ordering::SeqCst));
        assert!(
            !callback_called.load(Ordering::SeqCst),
            "Consolidation callback MUST NOT be called when IC >= threshold"
        );

        println!("EVIDENCE: No dream trigger for IC=0.7 (above threshold 0.5)");
    }

    #[test]
    fn test_ic_at_threshold_no_trigger() {
        println!("=== FSV: IC exactly at threshold does NOT trigger ===");

        let config = TriggerConfig::default()
            .with_cooldown(Duration::from_millis(1));
        let manager = Arc::new(Mutex::new(TriggerManager::with_config(config)));

        let callback_called = Arc::new(AtomicBool::new(false));
        let cb = Arc::clone(&callback_called);
        let callback: DreamConsolidationCallback = Arc::new(move |_| {
            cb.store(true, Ordering::SeqCst);
        });

        let queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(queue)
            .with_trigger_manager(manager)
            .with_consolidation_callback(callback);

        // IC=0.5 = threshold 0.5 (not < threshold), should NOT trigger
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.5,
            previous_status: "Stable".to_string(),
            current_status: "Warning".to_string(),
            reason: "Test at threshold".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        assert!(
            !callback_called.load(Ordering::SeqCst),
            "Consolidation callback MUST NOT be called when IC = threshold (need IC < threshold)"
        );

        println!("EVIDENCE: No dream trigger for IC=0.5 (at threshold, not below)");
    }

    #[test]
    fn test_ic_just_below_threshold_triggers() {
        println!("=== FSV: IC just below threshold DOES trigger ===");

        let config = TriggerConfig::default()
            .with_cooldown(Duration::from_millis(1));
        let manager = Arc::new(Mutex::new(TriggerManager::with_config(config)));

        let callback_called = Arc::new(AtomicBool::new(false));
        let cb = Arc::clone(&callback_called);
        let callback: DreamConsolidationCallback = Arc::new(move |_| {
            cb.store(true, Ordering::SeqCst);
        });

        let queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(queue)
            .with_trigger_manager(manager)
            .with_consolidation_callback(callback);

        // IC=0.4999 < threshold 0.5, should trigger
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.4999,
            previous_status: "Stable".to_string(),
            current_status: "Critical".to_string(),
            reason: "Test just below".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        assert!(
            callback_called.load(Ordering::SeqCst),
            "Consolidation callback MUST be called when IC < threshold (even just below)"
        );

        println!("EVIDENCE: Dream trigger for IC=0.4999 (just below threshold 0.5)");
    }

    #[test]
    fn test_ic_zero_triggers() {
        println!("=== FSV: IC=0.0 (minimum) triggers dream ===");

        let config = TriggerConfig::default()
            .with_cooldown(Duration::from_millis(1));
        let manager = Arc::new(Mutex::new(TriggerManager::with_config(config)));

        let callback_called = Arc::new(AtomicBool::new(false));
        let callback_ic = Arc::new(AtomicU32::new(u32::MAX));
        let cb = Arc::clone(&callback_called);
        let cb_ic = Arc::clone(&callback_ic);
        let callback: DreamConsolidationCallback = Arc::new(move |reason| {
            cb.store(true, Ordering::SeqCst);
            if let ExtendedTriggerReason::IdentityCritical { ic_value } = reason {
                cb_ic.store(ic_value.to_bits(), Ordering::SeqCst);
            }
        });

        let queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(queue)
            .with_trigger_manager(manager)
            .with_consolidation_callback(callback);

        // IC=0.0 (minimum possible) should definitely trigger
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.0,
            previous_status: "Stable".to_string(),
            current_status: "Critical".to_string(),
            reason: "Complete identity loss".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        assert!(
            callback_called.load(Ordering::SeqCst),
            "Consolidation callback MUST be called for IC=0.0"
        );

        let stored_ic = f32::from_bits(callback_ic.load(Ordering::SeqCst));
        assert!(
            stored_ic.abs() < 0.001,
            "Callback should receive IC=0.0, got {}",
            stored_ic
        );

        println!("EVIDENCE: Dream trigger for IC=0.0 (minimum, complete identity loss)");
    }

    #[test]
    fn test_listener_without_trigger_manager_still_logs() {
        println!("=== FSV: Listener without TriggerManager still works (logs only) ===");

        // Listener without TriggerManager should still work (just logs)
        let queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(queue);

        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.3,
            previous_status: "Stable".to_string(),
            current_status: "Critical".to_string(),
            reason: "Test".to_string(),
            timestamp: Utc::now(),
        };

        // Should not panic, just log
        listener.on_event(&event);

        println!("EVIDENCE: Listener without TriggerManager handles IC event without panic");
    }

    #[test]
    fn test_queue_functionality_preserved() {
        println!("=== FSV: Existing queue functionality preserved ===");

        // Existing queue behavior must still work with TriggerManager wired
        let queue = Arc::new(RwLock::new(Vec::new()));
        let manager = Arc::new(Mutex::new(TriggerManager::new()));
        let listener = DreamEventListener::new(Arc::clone(&queue))
            .with_trigger_manager(manager);

        let memory_id = Uuid::new_v4();

        // BEFORE
        let before_len = queue.blocking_read().len();
        println!("BEFORE: queue.len() = {}", before_len);

        // EXECUTE
        let event = WorkspaceEvent::MemoryExits {
            id: memory_id,
            order_parameter: 0.65,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        // AFTER
        let q = queue.blocking_read();
        println!("AFTER: queue.len() = {}", q.len());

        assert_eq!(q.len(), 1, "Queue should have 1 memory");
        assert_eq!(q[0], memory_id, "Queued memory should match");

        println!("EVIDENCE: Queue functionality preserved with TriggerManager wired");
    }

    #[test]
    fn test_cooldown_prevents_rapid_triggers() {
        println!("=== FSV: Cooldown prevents rapid IC triggers ===");

        // TriggerManager with 100ms cooldown
        let config = TriggerConfig::default()
            .with_cooldown(Duration::from_millis(100));
        let manager = Arc::new(Mutex::new(TriggerManager::with_config(config)));

        let trigger_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let tc = Arc::clone(&trigger_count);
        let callback: DreamConsolidationCallback = Arc::new(move |_| {
            tc.fetch_add(1, Ordering::SeqCst);
        });

        let queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(queue)
            .with_trigger_manager(manager)
            .with_consolidation_callback(callback);

        // First IC crisis - should trigger
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.3,
            previous_status: "Stable".to_string(),
            current_status: "Critical".to_string(),
            reason: "First crisis".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        // Second IC crisis immediately - should NOT trigger (cooldown)
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.2,
            previous_status: "Critical".to_string(),
            current_status: "Critical".to_string(),
            reason: "Second crisis".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        let count = trigger_count.load(Ordering::SeqCst);
        println!("EVIDENCE: Trigger count = {} (expected 1 due to cooldown)", count);

        assert_eq!(count, 1, "Only first trigger should fire due to cooldown");
    }

    #[test]
    fn test_callback_not_called_when_not_set() {
        println!("=== FSV: No callback = no crash ===");

        let config = TriggerConfig::default()
            .with_cooldown(Duration::from_millis(1));
        let manager = Arc::new(Mutex::new(TriggerManager::with_config(config)));

        // Create listener WITH TriggerManager but WITHOUT callback
        let queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(queue)
            .with_trigger_manager(manager);
        // Note: no .with_consolidation_callback()

        // Should not panic even though IC triggers
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.3,
            previous_status: "Stable".to_string(),
            current_status: "Critical".to_string(),
            reason: "Test".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        println!("EVIDENCE: Trigger fires without callback set (no crash)");
    }

    #[test]
    fn test_debug_impl() {
        let queue = Arc::new(RwLock::new(Vec::new()));

        // Without trigger manager
        let listener = DreamEventListener::new(Arc::clone(&queue));
        let debug_str = format!("{:?}", listener);
        assert!(debug_str.contains("has_trigger_manager: false"));
        assert!(debug_str.contains("has_callback: false"));

        // With trigger manager and callback
        let manager = Arc::new(Mutex::new(TriggerManager::new()));
        let callback: DreamConsolidationCallback = Arc::new(|_| {});
        let listener = DreamEventListener::new(queue)
            .with_trigger_manager(manager)
            .with_consolidation_callback(callback);
        let debug_str = format!("{:?}", listener);
        assert!(debug_str.contains("has_trigger_manager: true"));
        assert!(debug_str.contains("has_callback: true"));

        println!("EVIDENCE: Debug impl shows trigger_manager and callback state");
    }
}
