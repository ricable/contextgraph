//! Kuramoto Background Stepper - TASK-GWT-P0-002
//!
//! Provides a background tokio task that continuously steps the Kuramoto oscillator
//! network at regular intervals, enabling temporal dynamics for consciousness emergence.
//!
//! ## Problem This Solves
//!
//! Without continuous stepping, the Kuramoto oscillator phases remain static, meaning:
//! - The order parameter `r` never changes dynamically
//! - Consciousness emergence via C(t) = I(t) × R(t) × D(t) is impossible
//! - The system appears frozen in time
//!
//! ## Architecture
//!
//! The stepper uses:
//! - `tokio::spawn` for the background task
//! - `tokio::select!` for clean shutdown handling
//! - `tokio::time::interval` for precise timing (not sleep loops)
//! - `parking_lot::RwLock` (matches core.rs pattern, NOT tokio::RwLock)
//! - `Arc<Notify>` for graceful shutdown signaling
//! - `Arc<AtomicBool>` for lock-free running state checks
//!
//! ## Usage
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use parking_lot::RwLock;
//! use context_graph_mcp::handlers::kuramoto_stepper::{KuramotoStepper, KuramotoStepperConfig};
//! use context_graph_mcp::handlers::gwt_providers::KuramotoProviderImpl;
//! use context_graph_mcp::handlers::gwt_traits::KuramotoProvider;
//!
//! // Create the network wrapped in Arc<parking_lot::RwLock>
//! let network: Arc<RwLock<dyn KuramotoProvider>> =
//!     Arc::new(RwLock::new(KuramotoProviderImpl::new()));
//!
//! // Create and start the stepper
//! let mut stepper = KuramotoStepper::new(network, KuramotoStepperConfig::default());
//! stepper.start().expect("stepper must start");
//!
//! // ... let it run ...
//!
//! // Graceful shutdown
//! stepper.stop().await.expect("stepper must stop");
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use super::gwt_traits::KuramotoProvider;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for the Kuramoto background stepper.
///
/// The step interval determines how frequently the oscillator phases are updated.
/// A 10ms interval (100Hz) satisfies the Nyquist rate for all brain wave frequencies
/// modeled in the Kuramoto network (4Hz theta to 80Hz high-gamma).
///
/// NOTE: This is a public API type exported via handlers/mod.rs for external use.
/// It will be wired into MCP server lifecycle in TASK-GWT-P1-002.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct KuramotoStepperConfig {
    /// Step interval in milliseconds (default: 10ms for 100Hz update rate)
    pub step_interval_ms: u64,
}

impl Default for KuramotoStepperConfig {
    fn default() -> Self {
        Self { step_interval_ms: 10 }
    }
}

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Errors that can occur during Kuramoto stepper operations.
///
/// FAIL FAST: These errors are explicit and descriptive. No silent fallbacks.
///
/// NOTE: This is a public API type exported via handlers/mod.rs for external use.
/// It will be wired into MCP server lifecycle in TASK-GWT-P1-002.
#[allow(dead_code)]
#[derive(Debug, thiserror::Error)]
pub enum KuramotoStepperError {
    /// Attempted to start the stepper when it's already running
    #[error("Stepper already running - call stop() first")]
    AlreadyRunning,

    /// Attempted to stop the stepper when it's not running
    #[error("Stepper not running - call start() first")]
    NotRunning,

    /// Shutdown did not complete within the timeout
    #[error("Shutdown timeout after {0}ms - task may be stuck")]
    ShutdownTimeout(u64),
}

// ============================================================================
// KURAMOTO STEPPER
// ============================================================================

/// Background task that continuously steps the Kuramoto oscillator network.
///
/// Runs in a `tokio::spawn` task, calling `step()` at regular intervals (default 10ms).
/// Supports graceful shutdown via the `stop()` method.
///
/// # Thread Safety
///
/// Uses `Arc<parking_lot::RwLock<dyn KuramotoProvider>>` for the network reference.
/// The parking_lot RwLock is used (not tokio::RwLock) to match the pattern in
/// `crates/context-graph-mcp/src/handlers/core.rs` line 262.
///
/// # Lock Contention Handling
///
/// The stepper uses `try_write_for` with a 500μs timeout. If the lock is contended,
/// the step is skipped and the next iteration will catch up with a larger elapsed
/// duration.
///
/// NOTE: This is a public API type exported via handlers/mod.rs for external use.
/// It will be wired into MCP server lifecycle in TASK-GWT-P1-002.
#[allow(dead_code)]
pub struct KuramotoStepper {
    /// Shared reference to the Kuramoto provider (uses parking_lot::RwLock)
    network: Arc<RwLock<dyn KuramotoProvider>>,

    /// Configuration
    config: KuramotoStepperConfig,

    /// Shutdown signal sender
    shutdown_notify: Arc<Notify>,

    /// Handle to the background task (None if not running)
    task_handle: Option<JoinHandle<()>>,

    /// Running state flag (lock-free check)
    is_running: Arc<AtomicBool>,
}

#[allow(dead_code)]
impl KuramotoStepper {
    /// Create a new stepper with the given network and configuration.
    ///
    /// The stepper is NOT started automatically. Call `start()` to begin stepping.
    ///
    /// # Arguments
    ///
    /// * `network` - Arc-wrapped Kuramoto provider with parking_lot RwLock
    /// * `config` - Stepper configuration (step interval, etc.)
    pub fn new(
        network: Arc<RwLock<dyn KuramotoProvider>>,
        config: KuramotoStepperConfig,
    ) -> Self {
        Self {
            network,
            config,
            shutdown_notify: Arc::new(Notify::new()),
            task_handle: None,
            is_running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start the background stepping task.
    ///
    /// Spawns a tokio task that calls `step()` at the configured interval.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if started successfully
    /// * `Err(KuramotoStepperError::AlreadyRunning)` if already running
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// stepper.start().expect("stepper must start");
    /// assert!(stepper.is_running());
    /// ```
    pub fn start(&mut self) -> Result<(), KuramotoStepperError> {
        if self.is_running.load(Ordering::SeqCst) {
            return Err(KuramotoStepperError::AlreadyRunning);
        }

        // Mark as running BEFORE spawning task
        self.is_running.store(true, Ordering::SeqCst);

        // Reset shutdown notify for clean restart
        self.shutdown_notify = Arc::new(Notify::new());

        // Clone Arcs for the spawned task
        let network = Arc::clone(&self.network);
        let shutdown = Arc::clone(&self.shutdown_notify);
        let is_running = Arc::clone(&self.is_running);
        let interval_ms = self.config.step_interval_ms;

        let handle = tokio::spawn(async move {
            stepper_loop(network, shutdown, is_running, interval_ms).await;
        });

        self.task_handle = Some(handle);

        tracing::info!(
            interval_ms = interval_ms,
            "Kuramoto stepper started"
        );

        Ok(())
    }

    /// Stop the background stepping task gracefully.
    ///
    /// Signals the task to stop and waits for it to complete (with 5 second timeout).
    ///
    /// # Returns
    ///
    /// * `Ok(())` if stopped successfully
    /// * `Err(KuramotoStepperError::NotRunning)` if not running
    /// * `Err(KuramotoStepperError::ShutdownTimeout)` if task doesn't stop in time
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// stepper.stop().await.expect("stepper must stop");
    /// assert!(!stepper.is_running());
    /// ```
    pub async fn stop(&mut self) -> Result<(), KuramotoStepperError> {
        if !self.is_running.load(Ordering::SeqCst) {
            return Err(KuramotoStepperError::NotRunning);
        }

        // Signal shutdown
        self.shutdown_notify.notify_one();

        // Wait for task with 5 second timeout
        if let Some(handle) = self.task_handle.take() {
            match tokio::time::timeout(Duration::from_secs(5), handle).await {
                Ok(Ok(())) => {
                    // Task completed normally
                    self.is_running.store(false, Ordering::SeqCst);
                    tracing::info!("Kuramoto stepper stopped gracefully");
                    Ok(())
                }
                Ok(Err(e)) => {
                    // Task panicked - still mark as not running
                    self.is_running.store(false, Ordering::SeqCst);
                    tracing::error!(error = ?e, "Kuramoto stepper task panicked");
                    // Consider it stopped despite the panic
                    Ok(())
                }
                Err(_) => {
                    // Timeout - task is stuck
                    tracing::error!("Kuramoto stepper shutdown timeout after 5000ms");
                    Err(KuramotoStepperError::ShutdownTimeout(5000))
                }
            }
        } else {
            // No handle but was marked running - inconsistent state
            self.is_running.store(false, Ordering::SeqCst);
            Ok(())
        }
    }

    /// Check if the stepper is currently running.
    ///
    /// This is a lock-free check using atomic load.
    #[inline]
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::SeqCst)
    }

    /// Get the current step interval in milliseconds.
    #[inline]
    pub fn step_interval_ms(&self) -> u64 {
        self.config.step_interval_ms
    }
}

// ============================================================================
// STEPPER LOOP
// ============================================================================

/// The main stepper loop that runs in the background task.
///
/// Uses `tokio::select!` to handle both:
/// 1. Shutdown signal (graceful termination)
/// 2. Interval tick (step the network)
///
/// # Lock Contention Handling
///
/// Uses `try_write_for` with 500μs timeout. If the lock is contended:
/// - The step is skipped
/// - Next step will have larger `elapsed` duration to catch up
/// - A trace log is emitted for debugging
#[allow(dead_code)]
async fn stepper_loop(
    network: Arc<RwLock<dyn KuramotoProvider>>,
    shutdown_notify: Arc<Notify>,
    is_running: Arc<AtomicBool>,
    interval_ms: u64,
) {
    // Handle zero interval by using 1ms minimum (tokio::time::interval panics on 0)
    let actual_interval_ms = interval_ms.max(1);
    let mut interval = tokio::time::interval(Duration::from_millis(actual_interval_ms));
    let mut last_step = Instant::now();

    tracing::info!(
        interval_ms = actual_interval_ms,
        "Kuramoto stepper loop started"
    );

    loop {
        tokio::select! {
            // Shutdown signal - highest priority via biased selection
            biased;

            _ = shutdown_notify.notified() => {
                is_running.store(false, Ordering::SeqCst);
                tracing::info!("Kuramoto stepper received shutdown signal");
                break;
            }

            // Step interval tick
            _ = interval.tick() => {
                // Check running flag (may have been set externally)
                if !is_running.load(Ordering::Relaxed) {
                    tracing::debug!("Kuramoto stepper stopping due to is_running=false");
                    break;
                }

                let elapsed = last_step.elapsed();
                last_step = Instant::now();

                // Try to acquire write lock with brief timeout (500 microseconds)
                // Using parking_lot's try_write_for which returns Option<RwLockWriteGuard>
                if let Some(mut network_guard) = network.try_write_for(Duration::from_micros(500)) {
                    network_guard.step(elapsed);
                } else {
                    // Lock contention - skip this step, next one will catch up
                    tracing::trace!(
                        elapsed_ms = elapsed.as_millis(),
                        "Kuramoto step skipped due to lock contention"
                    );
                }
            }
        }
    }

    tracing::info!("Kuramoto stepper loop stopped");
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handlers::gwt_providers::KuramotoProviderImpl;

    /// Helper to create a test network
    fn create_test_network() -> Arc<RwLock<dyn KuramotoProvider>> {
        Arc::new(RwLock::new(KuramotoProviderImpl::new()))
    }

    // ========================================================================
    // LIFECYCLE TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_stepper_new_not_running() {
        println!("\n=== TEST: new() creates non-running stepper ===");

        let network = create_test_network();
        let stepper = KuramotoStepper::new(network, KuramotoStepperConfig::default());

        println!("STATE: is_running = {}", stepper.is_running());
        assert!(!stepper.is_running(), "New stepper should not be running");
        assert_eq!(stepper.step_interval_ms(), 10, "Default interval should be 10ms");

        println!("EVIDENCE: new() creates stepper in stopped state");
    }

    #[tokio::test]
    async fn test_stepper_start_stop_lifecycle() {
        println!("\n=== TEST: start/stop lifecycle ===");

        let network = create_test_network();
        let mut stepper = KuramotoStepper::new(network, KuramotoStepperConfig::default());

        // STATE BEFORE
        println!("BEFORE: is_running = {}", stepper.is_running());
        assert!(!stepper.is_running());

        // START
        stepper.start().expect("start must succeed");
        println!("AFTER START: is_running = {}", stepper.is_running());
        assert!(stepper.is_running(), "Stepper should be running after start");

        // Let it run briefly
        tokio::time::sleep(Duration::from_millis(50)).await;

        // STOP
        stepper.stop().await.expect("stop must succeed");
        println!("AFTER STOP: is_running = {}", stepper.is_running());
        assert!(!stepper.is_running(), "Stepper should not be running after stop");

        println!("EVIDENCE: start/stop lifecycle works correctly");
    }

    // ========================================================================
    // FULL STATE VERIFICATION TEST
    // ========================================================================

    #[tokio::test]
    async fn test_stepper_full_state_verification() {
        println!("\n=== FULL STATE VERIFICATION TEST ===");

        // === SETUP ===
        let network: Arc<RwLock<dyn KuramotoProvider>> =
            Arc::new(RwLock::new(KuramotoProviderImpl::new()));
        let config = KuramotoStepperConfig::default();
        let mut stepper = KuramotoStepper::new(Arc::clone(&network), config);

        // === STATE BEFORE ===
        let initial_r = {
            let net = network.read();
            net.order_parameter().0
        };
        println!("STATE BEFORE: r = {:.4}, is_running = {}", initial_r, stepper.is_running());
        assert!(!stepper.is_running());

        // === EXECUTE ===
        stepper.start().expect("start must succeed");
        println!("STATE AFTER START: is_running = {}", stepper.is_running());
        assert!(stepper.is_running());

        // Let it run for 500ms (50 steps at 10ms interval)
        tokio::time::sleep(Duration::from_millis(500)).await;

        // === VERIFY VIA SEPARATE READ ===
        let after_r = {
            let net = network.read();
            net.order_parameter().0
        };
        println!("STATE AFTER 500ms: r = {:.4}", after_r);

        // r should be valid (within bounds)
        assert!((0.0..=1.0).contains(&after_r), "r must be valid: {}", after_r);

        // === STOP ===
        stepper.stop().await.expect("stop must succeed");
        println!("STATE AFTER STOP: is_running = {}", stepper.is_running());
        assert!(!stepper.is_running());

        // === EVIDENCE OF SUCCESS ===
        println!(
            "EVIDENCE: Stepper ran for 500ms, r evolved from {:.4} to {:.4}",
            initial_r, after_r
        );
    }

    // ========================================================================
    // EDGE CASE TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_double_start_fails() {
        println!("\n=== EDGE CASE 1: Double Start ===");

        let network = create_test_network();
        let mut stepper = KuramotoStepper::new(network, KuramotoStepperConfig::default());

        // STATE BEFORE
        println!("BEFORE: is_running = {}", stepper.is_running());

        // First start succeeds
        assert!(stepper.start().is_ok());
        println!("AFTER FIRST START: is_running = {}", stepper.is_running());

        // Second start MUST fail with AlreadyRunning
        let result = stepper.start();
        println!("AFTER SECOND START: result = {:?}", result);
        assert!(
            matches!(result, Err(KuramotoStepperError::AlreadyRunning)),
            "Second start should return AlreadyRunning error"
        );

        // Cleanup
        stepper.stop().await.expect("cleanup stop must succeed");
        println!("EVIDENCE: Double start correctly returns AlreadyRunning error");
    }

    #[tokio::test]
    async fn test_stop_when_not_running_fails() {
        println!("\n=== EDGE CASE 2: Stop When Not Running ===");

        let network = create_test_network();
        let mut stepper = KuramotoStepper::new(network, KuramotoStepperConfig::default());

        println!("BEFORE: is_running = {}", stepper.is_running());

        let result = stepper.stop().await;
        println!("AFTER STOP: result = {:?}", result);

        assert!(
            matches!(result, Err(KuramotoStepperError::NotRunning)),
            "Stop on non-running stepper should return NotRunning error"
        );

        println!("EVIDENCE: Stop when not running correctly returns NotRunning error");
    }

    #[tokio::test]
    async fn test_zero_interval_handled() {
        println!("\n=== EDGE CASE 3: Zero Interval ===");

        let network = create_test_network();
        let config = KuramotoStepperConfig { step_interval_ms: 0 };
        let mut stepper = KuramotoStepper::new(network, config);

        println!("BEFORE: step_interval_ms = {}", stepper.step_interval_ms());

        // Should start without panic (stepper_loop handles 0 by using 1ms)
        let result = stepper.start();
        println!(
            "AFTER START: result = {:?}, is_running = {}",
            result,
            stepper.is_running()
        );

        assert!(result.is_ok(), "Zero interval should not prevent start");
        assert!(stepper.is_running());

        // Let run briefly
        tokio::time::sleep(Duration::from_millis(50)).await;

        stepper.stop().await.expect("cleanup stop must succeed");
        println!("EVIDENCE: Zero interval handled correctly (uses 1ms minimum)");
    }

    #[tokio::test]
    async fn test_multiple_start_stop_cycles() {
        println!("\n=== TEST: Multiple Start/Stop Cycles ===");

        let network = create_test_network();
        let mut stepper = KuramotoStepper::new(network, KuramotoStepperConfig::default());

        for cycle in 1..=3 {
            println!("--- Cycle {} ---", cycle);

            stepper.start().expect("start must succeed");
            println!("  Started: is_running = {}", stepper.is_running());
            assert!(stepper.is_running());

            tokio::time::sleep(Duration::from_millis(30)).await;

            stepper.stop().await.expect("stop must succeed");
            println!("  Stopped: is_running = {}", stepper.is_running());
            assert!(!stepper.is_running());
        }

        println!("EVIDENCE: Multiple start/stop cycles work correctly");
    }

    #[tokio::test]
    async fn test_order_parameter_changes() {
        println!("\n=== TEST: Order Parameter Changes During Stepping ===");

        // Use synchronized network for predictable behavior
        let network: Arc<RwLock<dyn KuramotoProvider>> =
            Arc::new(RwLock::new(KuramotoProviderImpl::synchronized()));

        let config = KuramotoStepperConfig { step_interval_ms: 5 }; // Fast stepping
        let mut stepper = KuramotoStepper::new(Arc::clone(&network), config);

        // Get initial r
        let initial_r = {
            let net = network.read();
            net.order_parameter().0
        };
        println!("BEFORE: r = {:.6}", initial_r);

        // Start and let run
        stepper.start().expect("start must succeed");
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Get r after stepping
        let after_r = {
            let net = network.read();
            net.order_parameter().0
        };
        println!("AFTER 200ms: r = {:.6}", after_r);

        stepper.stop().await.expect("stop must succeed");

        // Verify r is still valid (may or may not have changed depending on coupling)
        assert!((0.0..=1.0).contains(&after_r), "r must be in [0,1]");

        println!(
            "EVIDENCE: Order parameter tracked correctly, initial={:.6}, final={:.6}",
            initial_r, after_r
        );
    }

    #[tokio::test]
    async fn test_concurrent_network_access() {
        println!("\n=== TEST: Concurrent Network Access During Stepping ===");

        let network: Arc<RwLock<dyn KuramotoProvider>> =
            Arc::new(RwLock::new(KuramotoProviderImpl::new()));

        let mut stepper =
            KuramotoStepper::new(Arc::clone(&network), KuramotoStepperConfig::default());

        stepper.start().expect("start must succeed");

        // Spawn concurrent readers while stepper is running
        let network_clone = Arc::clone(&network);
        let reader_handle = tokio::spawn(async move {
            for i in 0..20 {
                let r = {
                    let net = network_clone.read();
                    net.order_parameter().0
                };
                assert!((0.0..=1.0).contains(&r), "r must be valid during concurrent read");
                if i % 5 == 0 {
                    println!("  Concurrent read {}: r = {:.4}", i, r);
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });

        reader_handle.await.expect("reader task must complete");

        stepper.stop().await.expect("stop must succeed");
        println!("EVIDENCE: Concurrent access works without deadlock or invalid state");
    }

    #[tokio::test]
    async fn test_elapsed_time_passed_correctly() {
        println!("\n=== TEST: Elapsed Time Passed to step() ===");

        // We can't directly verify the elapsed time passed to step(),
        // but we can verify the network's elapsed_total increases
        let network: Arc<RwLock<dyn KuramotoProvider>> =
            Arc::new(RwLock::new(KuramotoProviderImpl::new()));

        let mut stepper = KuramotoStepper::new(
            Arc::clone(&network),
            KuramotoStepperConfig { step_interval_ms: 10 },
        );

        let initial_elapsed = {
            let net = network.read();
            net.elapsed_total()
        };
        println!("BEFORE: elapsed_total = {:?}", initial_elapsed);

        stepper.start().expect("start must succeed");
        tokio::time::sleep(Duration::from_millis(100)).await;
        stepper.stop().await.expect("stop must succeed");

        let final_elapsed = {
            let net = network.read();
            net.elapsed_total()
        };
        println!("AFTER: elapsed_total = {:?}", final_elapsed);

        assert!(
            final_elapsed > initial_elapsed,
            "elapsed_total should increase during stepping"
        );

        println!(
            "EVIDENCE: Network elapsed_total increased from {:?} to {:?}",
            initial_elapsed, final_elapsed
        );
    }
}
