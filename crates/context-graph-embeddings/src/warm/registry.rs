//! Warm Model Registry
//!
//! Tracks the loading state and VRAM handles for all 12 embedding models.
//!
//! # Overview
//!
//! The [`WarmModelRegistry`] is the central state manager for the warm loading system.
//! It maintains entries for each model component and tracks their lifecycle from
//! registration through loading, validation, and warm state.
//!
//! # Thread Safety
//!
//! The [`SharedWarmRegistry`] type alias provides thread-safe access via `Arc<RwLock<_>>`.
//! Multiple readers can access state concurrently, while writers have exclusive access.
//! All state-modifying methods return [`WarmResult`] to handle lock poisoning scenarios.
//!
//! # Model Components
//!
//! The registry tracks 12 embedding models:
//!
//! | Model ID | Description |
//! |----------|-------------|
//! | `E1_Semantic` | Semantic similarity embeddings |
//! | `E2_TemporalRecent` | Recent temporal context |
//! | `E3_TemporalPeriodic` | Periodic temporal patterns |
//! | `E4_TemporalPositional` | Positional temporal encoding |
//! | `E5_Causal` | Causal relationship embeddings |
//! | `E6_Sparse` | Sparse activation embeddings |
//! | `E7_Code` | Code/programming embeddings |
//! | `E8_Graph` | Graph structure embeddings |
//! | `E9_HDC` | Hyperdimensional computing embeddings |
//! | `E10_Multimodal` | Multimodal embeddings (CLIP) |
//! | `E11_Entity` | Named entity embeddings |
//! | `E12_LateInteraction` | Late interaction embeddings |
//!
//! # State Transitions
//!
//! Each model follows a strict state machine with the following valid transitions:
//!
//! ```text
//!                     +-------------+
//!                     |   Pending   |
//!                     +------+------+
//!                            |
//!                     start_loading()
//!                            |
//!                            v
//!                     +------+------+
//!                     |   Loading   |<----+
//!                     +------+------+     |
//!                            |            |
//!            +---------------+------------+
//!            |               |
//!     mark_validating()   update_progress()
//!            |
//!            v
//!     +------+------+
//!     |  Validating |
//!     +------+------+
//!            |
//!     mark_warm()
//!            |
//!            v
//!     +------+------+
//!     |    Warm     |
//!     +-------------+
//!
//! Note: mark_failed() can be called from Loading or Validating states
//!       to transition to Failed state.
//!
//!     Loading ----mark_failed()----> Failed
//!     Validating --mark_failed()---> Failed
//! ```
//!
//! # Requirements Fulfilled
//!
//! - **REQ-WARM-001**: Track all 12 embedding models
//! - **REQ-WARM-004**: Maintain VRAM residency via ModelHandle
//!
//! # Example
//!
//! ```ignore
//! use std::sync::{Arc, RwLock};
//! use context_graph_embeddings::warm::registry::{WarmModelRegistry, SharedWarmRegistry};
//!
//! // Create a shared registry
//! let registry: SharedWarmRegistry = Arc::new(RwLock::new(WarmModelRegistry::new()));
//!
//! // Register a model
//! {
//!     let mut reg = registry.write().unwrap();
//!     reg.register_model("E1_Semantic", 512 * 1024 * 1024, 768)?;
//!     reg.start_loading("E1_Semantic")?;
//!     reg.update_progress("E1_Semantic", 50, 256 * 1024 * 1024)?;
//!     reg.mark_validating("E1_Semantic")?;
//!     reg.mark_warm("E1_Semantic", handle)?;
//! }
//!
//! // Check status
//! {
//!     let reg = registry.read().unwrap();
//!     assert!(reg.get_state("E1_Semantic").map(|s| s.is_warm()).unwrap_or(false));
//! }
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::error::{WarmError, WarmResult};
use super::handle::ModelHandle;
use super::state::WarmModelState;

/// The 12 embedding model IDs in the system.
pub const EMBEDDING_MODEL_IDS: [&str; 12] = [
    "E1_Semantic",
    "E2_TemporalRecent",
    "E3_TemporalPeriodic",
    "E4_TemporalPositional",
    "E5_Causal",
    "E6_Sparse",
    "E7_Code",
    "E8_Graph",
    "E9_HDC",
    "E10_Multimodal",
    "E11_Entity",
    "E12_LateInteraction",
];

/// Total number of model components (12 embeddings).
pub const TOTAL_MODEL_COUNT: usize = 12;

/// Thread-safe shared registry for concurrent access.
///
/// Wraps [`WarmModelRegistry`] in `Arc<RwLock<_>>` for safe multi-threaded access.
/// Use `read()` for shared read access and `write()` for exclusive write access.
///
/// # Lock Poisoning
///
/// If a thread panics while holding the lock, subsequent access attempts will
/// encounter a poisoned lock. Handle this gracefully by returning
/// [`WarmError::RegistryLockPoisoned`].
pub type SharedWarmRegistry = Arc<RwLock<WarmModelRegistry>>;

/// Entry for a single model in the registry.
///
/// Tracks the complete lifecycle state of a model from registration through
/// warm state, including VRAM allocation metadata.
#[derive(Debug)]
pub struct WarmModelEntry {
    /// Current state in the loading lifecycle.
    pub state: WarmModelState,
    /// VRAM handle when model is in Warm state, None otherwise.
    pub handle: Option<ModelHandle>,
    /// Expected size of model weights in bytes.
    pub expected_bytes: usize,
    /// Expected output embedding dimension.
    pub expected_dimension: usize,
    /// Unique model identifier (e.g., "E1_Semantic").
    pub model_id: String,
}

impl WarmModelEntry {
    /// Create a new entry in Pending state.
    fn new(model_id: impl Into<String>, expected_bytes: usize, expected_dimension: usize) -> Self {
        Self {
            state: WarmModelState::Pending,
            handle: None,
            expected_bytes,
            expected_dimension,
            model_id: model_id.into(),
        }
    }
}

/// Registry for tracking warm model loading state.
///
/// Maintains a HashMap of [`WarmModelEntry`] keyed by model ID.
/// All 12 embedding models should be registered before loading begins.
///
/// # Invariants
///
/// - Model IDs are unique within the registry
/// - State transitions follow the documented state machine
/// - `handle` is `Some` only when state is `Warm`
/// - `handle` is `None` for all other states
#[derive(Debug, Default)]
pub struct WarmModelRegistry {
    /// Model entries keyed by model ID.
    entries: HashMap<String, WarmModelEntry>,
}

impl WarmModelRegistry {
    /// Create a new empty registry.
    ///
    /// The registry starts with no registered models. Use [`register_model`](Self::register_model)
    /// to add entries for each of the 12 embedding models.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let registry = WarmModelRegistry::new();
    /// assert_eq!(registry.model_count(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Register a new model in Pending state.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Unique identifier (e.g., "E1_Semantic", "E12_LateInteraction")
    /// * `expected_bytes` - Expected size of model weights in bytes
    /// * `expected_dimension` - Expected output embedding dimension
    ///
    /// # Errors
    ///
    /// Returns [`WarmError::ModelAlreadyRegistered`] if a model with the same ID
    /// already exists in the registry.
    ///
    /// # Example
    ///
    /// ```ignore
    /// registry.register_model("E1_Semantic", 512 * 1024 * 1024, 768)?;
    /// ```
    pub fn register_model(
        &mut self,
        model_id: impl Into<String>,
        expected_bytes: usize,
        expected_dimension: usize,
    ) -> WarmResult<()> {
        let model_id = model_id.into();

        if self.entries.contains_key(&model_id) {
            return Err(WarmError::ModelAlreadyRegistered {
                model_id,
            });
        }

        let entry = WarmModelEntry::new(model_id.clone(), expected_bytes, expected_dimension);
        self.entries.insert(model_id, entry);
        Ok(())
    }

    /// Transition a model from Pending to Loading state.
    ///
    /// Validates that the model exists and is currently in Pending state.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model to start loading
    ///
    /// # Errors
    ///
    /// - [`WarmError::ModelNotRegistered`] if model doesn't exist
    /// - [`WarmError::ModelLoadFailed`] if current state is not Pending
    ///
    /// # State Transition
    ///
    /// `Pending` -> `Loading { progress_percent: 0, bytes_loaded: 0 }`
    pub fn start_loading(&mut self, model_id: &str) -> WarmResult<()> {
        let entry = self.entries.get_mut(model_id).ok_or_else(|| {
            WarmError::ModelNotRegistered {
                model_id: model_id.to_string(),
            }
        })?;

        match &entry.state {
            WarmModelState::Pending => {
                entry.state = WarmModelState::Loading {
                    progress_percent: 0,
                    bytes_loaded: 0,
                };
                Ok(())
            }
            other => Err(WarmError::ModelLoadFailed {
                model_id: model_id.to_string(),
                reason: format!(
                    "Invalid state transition: cannot start loading from {:?} state (expected Pending)",
                    other
                ),
                bytes_read: 0,
                file_size: entry.expected_bytes,
            }),
        }
    }

    /// Update loading progress for a model in Loading state.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model being loaded
    /// * `progress_percent` - Loading progress (0-100)
    /// * `bytes_loaded` - Number of bytes loaded so far
    ///
    /// # Errors
    ///
    /// - [`WarmError::ModelNotRegistered`] if model doesn't exist
    /// - [`WarmError::ModelLoadFailed`] if current state is not Loading
    ///
    /// # State Transition
    ///
    /// `Loading { ... }` -> `Loading { progress_percent, bytes_loaded }`
    pub fn update_progress(
        &mut self,
        model_id: &str,
        progress_percent: u8,
        bytes_loaded: usize,
    ) -> WarmResult<()> {
        let entry = self.entries.get_mut(model_id).ok_or_else(|| {
            WarmError::ModelNotRegistered {
                model_id: model_id.to_string(),
            }
        })?;

        match &entry.state {
            WarmModelState::Loading { .. } => {
                entry.state = WarmModelState::Loading {
                    progress_percent: progress_percent.min(100),
                    bytes_loaded,
                };
                Ok(())
            }
            other => Err(WarmError::ModelLoadFailed {
                model_id: model_id.to_string(),
                reason: format!(
                    "Invalid state transition: cannot update progress in {:?} state (expected Loading)",
                    other
                ),
                bytes_read: bytes_loaded,
                file_size: entry.expected_bytes,
            }),
        }
    }

    /// Transition a model from Loading to Validating state.
    ///
    /// Called when model weights are fully loaded and validation begins.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model to validate
    ///
    /// # Errors
    ///
    /// - [`WarmError::ModelNotRegistered`] if model doesn't exist
    /// - [`WarmError::ModelValidationFailed`] if current state is not Loading
    ///
    /// # State Transition
    ///
    /// `Loading { ... }` -> `Validating`
    pub fn mark_validating(&mut self, model_id: &str) -> WarmResult<()> {
        let entry = self.entries.get_mut(model_id).ok_or_else(|| {
            WarmError::ModelNotRegistered {
                model_id: model_id.to_string(),
            }
        })?;

        match &entry.state {
            WarmModelState::Loading { .. } => {
                entry.state = WarmModelState::Validating;
                Ok(())
            }
            other => Err(WarmError::ModelValidationFailed {
                model_id: model_id.to_string(),
                reason: format!(
                    "Invalid state transition: cannot mark validating from {:?} state (expected Loading)",
                    other
                ),
                expected_output: None,
                actual_output: None,
            }),
        }
    }

    /// Transition a model from Validating to Warm state with a VRAM handle.
    ///
    /// Called when validation succeeds and model is ready for inference.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model that is now warm
    /// * `handle` - VRAM handle for the loaded weights
    ///
    /// # Errors
    ///
    /// - [`WarmError::ModelNotRegistered`] if model doesn't exist
    /// - [`WarmError::ModelValidationFailed`] if current state is not Validating
    ///
    /// # State Transition
    ///
    /// `Validating` -> `Warm` (with handle set)
    pub fn mark_warm(&mut self, model_id: &str, handle: ModelHandle) -> WarmResult<()> {
        let entry = self.entries.get_mut(model_id).ok_or_else(|| {
            WarmError::ModelNotRegistered {
                model_id: model_id.to_string(),
            }
        })?;

        match &entry.state {
            WarmModelState::Validating => {
                entry.state = WarmModelState::Warm;
                entry.handle = Some(handle);
                Ok(())
            }
            other => Err(WarmError::ModelValidationFailed {
                model_id: model_id.to_string(),
                reason: format!(
                    "Invalid state transition: cannot mark warm from {:?} state (expected Validating)",
                    other
                ),
                expected_output: None,
                actual_output: None,
            }),
        }
    }

    /// Transition a model to Failed state.
    ///
    /// Can be called from Loading or Validating states to record a failure.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model that failed
    /// * `error_code` - Numeric error code (e.g., 102 for load failure)
    /// * `error_message` - Human-readable error description
    ///
    /// # Errors
    ///
    /// - [`WarmError::ModelNotRegistered`] if model doesn't exist
    /// - [`WarmError::ModelLoadFailed`] if current state doesn't allow failure transition
    ///
    /// # State Transition
    ///
    /// `Loading` -> `Failed { error_code, error_message }`
    /// `Validating` -> `Failed { error_code, error_message }`
    pub fn mark_failed(
        &mut self,
        model_id: &str,
        error_code: u16,
        error_message: impl Into<String>,
    ) -> WarmResult<()> {
        let entry = self.entries.get_mut(model_id).ok_or_else(|| {
            WarmError::ModelNotRegistered {
                model_id: model_id.to_string(),
            }
        })?;

        match &entry.state {
            WarmModelState::Loading { .. } | WarmModelState::Validating => {
                entry.state = WarmModelState::Failed {
                    error_code,
                    error_message: error_message.into(),
                };
                entry.handle = None; // Ensure no handle on failure
                Ok(())
            }
            other => Err(WarmError::ModelLoadFailed {
                model_id: model_id.to_string(),
                reason: format!(
                    "Invalid state transition: cannot mark failed from {:?} state (expected Loading or Validating)",
                    other
                ),
                bytes_read: 0,
                file_size: entry.expected_bytes,
            }),
        }
    }

    /// Get the current state of a model.
    ///
    /// Returns `None` if the model is not registered.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(state) = registry.get_state("E1_Semantic") {
    ///     if state.is_warm() {
    ///         println!("Model is ready for inference");
    ///     }
    /// }
    /// ```
    #[must_use]
    pub fn get_state(&self, model_id: &str) -> Option<WarmModelState> {
        self.entries.get(model_id).map(|e| e.state.clone())
    }

    /// Get a reference to the VRAM handle for a model.
    ///
    /// Returns `None` if the model is not registered or not in Warm state.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(handle) = registry.get_handle("E1_Semantic") {
    ///     println!("VRAM address: {}", handle.vram_address_hex());
    /// }
    /// ```
    #[must_use]
    pub fn get_handle(&self, model_id: &str) -> Option<&ModelHandle> {
        self.entries.get(model_id).and_then(|e| e.handle.as_ref())
    }

    /// Check if all registered models are in Warm state.
    ///
    /// Returns `true` only if:
    /// - At least one model is registered
    /// - All registered models have `state == Warm`
    ///
    /// # Example
    ///
    /// ```ignore
    /// if registry.all_warm() {
    ///     println!("All {} models ready for inference", registry.model_count());
    /// }
    /// ```
    #[must_use]
    pub fn all_warm(&self) -> bool {
        !self.entries.is_empty() && self.entries.values().all(|e| e.state.is_warm())
    }

    /// Check if any registered model is in Failed state.
    ///
    /// Returns `true` if at least one model has `state == Failed`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if registry.any_failed() {
    ///     for (id, code, msg) in registry.failed_entries() {
    ///         eprintln!("Model {} failed: [{}] {}", id, code, msg);
    ///     }
    /// }
    /// ```
    #[must_use]
    pub fn any_failed(&self) -> bool {
        self.entries.values().any(|e| e.state.is_failed())
    }

    /// Get model IDs sorted by expected_bytes in descending order.
    ///
    /// This determines the optimal loading order: largest models first
    /// to maximize VRAM utilization and catch memory issues early.
    ///
    /// # Returns
    ///
    /// A `Vec<String>` of model IDs sorted from largest to smallest.
    ///
    /// # Example
    ///
    /// ```ignore
    /// for model_id in registry.loading_order() {
    ///     registry.start_loading(&model_id)?;
    ///     // ... load model ...
    /// }
    /// ```
    #[must_use]
    pub fn loading_order(&self) -> Vec<String> {
        let mut entries: Vec<_> = self
            .entries
            .iter()
            .map(|(id, e)| (id.clone(), e.expected_bytes))
            .collect();

        // Sort by expected_bytes descending (largest first)
        entries.sort_by(|a, b| b.1.cmp(&a.1));

        entries.into_iter().map(|(id, _)| id).collect()
    }

    /// Get the total number of registered models.
    ///
    /// Expected to be 12 when all components are registered.
    #[must_use]
    pub fn model_count(&self) -> usize {
        self.entries.len()
    }

    /// Get the number of models in Warm state.
    #[must_use]
    pub fn warm_count(&self) -> usize {
        self.entries.values().filter(|e| e.state.is_warm()).count()
    }

    /// Get information about all failed models.
    ///
    /// # Returns
    ///
    /// A `Vec` of tuples containing `(model_id, error_code, error_message)`
    /// for each model in Failed state.
    ///
    /// # Example
    ///
    /// ```ignore
    /// for (model_id, error_code, error_message) in registry.failed_entries() {
    ///     log::error!("Model {} failed with code {}: {}", model_id, error_code, error_message);
    /// }
    /// ```
    #[must_use]
    pub fn failed_entries(&self) -> Vec<(String, u16, String)> {
        self.entries
            .iter()
            .filter_map(|(id, entry)| {
                if let WarmModelState::Failed {
                    error_code,
                    error_message,
                } = &entry.state
                {
                    Some((id.clone(), *error_code, error_message.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get a reference to an entry by model ID.
    ///
    /// For internal use and testing.
    #[must_use]
    pub fn get_entry(&self, model_id: &str) -> Option<&WarmModelEntry> {
        self.entries.get(model_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a test ModelHandle
    fn test_handle(bytes: usize) -> ModelHandle {
        ModelHandle::new(0x1000_0000, bytes, 0, 0xDEAD_BEEF)
    }

    // ==================== Basic Registration Tests ====================

    #[test]
    fn test_new_registry_is_empty() {
        let registry = WarmModelRegistry::new();
        assert_eq!(registry.model_count(), 0);
        assert!(!registry.all_warm()); // Empty registry is not "all warm"
        assert!(!registry.any_failed());
    }

    #[test]
    fn test_register_model_success() {
        let mut registry = WarmModelRegistry::new();
        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();

        assert_eq!(registry.model_count(), 1);
        let entry = registry.get_entry("E1_Semantic").unwrap();
        assert_eq!(entry.model_id, "E1_Semantic");
        assert_eq!(entry.expected_bytes, 512 * 1024 * 1024);
        assert_eq!(entry.expected_dimension, 768);
        assert!(matches!(entry.state, WarmModelState::Pending));
        assert!(entry.handle.is_none());
    }

    #[test]
    fn test_register_model_duplicate_fails() {
        let mut registry = WarmModelRegistry::new();
        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();

        let err = registry
            .register_model("E1_Semantic", 256 * 1024 * 1024, 512)
            .unwrap_err();

        match err {
            WarmError::ModelAlreadyRegistered { model_id } => {
                assert_eq!(model_id, "E1_Semantic");
            }
            _ => panic!("Expected ModelAlreadyRegistered error"),
        }
    }

    #[test]
    fn test_register_all_models() {
        let mut registry = WarmModelRegistry::new();

        // Register all 12 embedding models
        for (i, model_id) in EMBEDDING_MODEL_IDS.iter().enumerate() {
            registry
                .register_model(*model_id, (i + 1) * 100 * 1024 * 1024, 768)
                .unwrap();
        }

        assert_eq!(registry.model_count(), TOTAL_MODEL_COUNT);
    }

    // ==================== State Transition Tests ====================

    #[test]
    fn test_valid_state_transitions_pending_to_warm() {
        let mut registry = WarmModelRegistry::new();
        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();

        // Pending -> Loading
        assert!(matches!(
            registry.get_state("E1_Semantic"),
            Some(WarmModelState::Pending)
        ));
        registry.start_loading("E1_Semantic").unwrap();
        assert!(matches!(
            registry.get_state("E1_Semantic"),
            Some(WarmModelState::Loading { progress_percent: 0, bytes_loaded: 0 })
        ));

        // Update progress
        registry
            .update_progress("E1_Semantic", 50, 256 * 1024 * 1024)
            .unwrap();
        assert!(matches!(
            registry.get_state("E1_Semantic"),
            Some(WarmModelState::Loading { progress_percent: 50, bytes_loaded: _ })
        ));

        // Loading -> Validating
        registry.mark_validating("E1_Semantic").unwrap();
        assert!(matches!(
            registry.get_state("E1_Semantic"),
            Some(WarmModelState::Validating)
        ));

        // Validating -> Warm
        registry
            .mark_warm("E1_Semantic", test_handle(512 * 1024 * 1024))
            .unwrap();
        assert!(matches!(
            registry.get_state("E1_Semantic"),
            Some(WarmModelState::Warm)
        ));
        assert!(registry.get_handle("E1_Semantic").is_some());
    }

    #[test]
    fn test_invalid_transition_warm_to_loading() {
        let mut registry = WarmModelRegistry::new();
        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();

        // Complete the full cycle to Warm
        registry.start_loading("E1_Semantic").unwrap();
        registry.mark_validating("E1_Semantic").unwrap();
        registry
            .mark_warm("E1_Semantic", test_handle(512 * 1024 * 1024))
            .unwrap();

        // Try to go back to Loading - should fail
        let err = registry.start_loading("E1_Semantic").unwrap_err();
        match err {
            WarmError::ModelLoadFailed { model_id, reason, .. } => {
                assert_eq!(model_id, "E1_Semantic");
                assert!(reason.contains("Invalid state transition"));
                assert!(reason.contains("Warm"));
            }
            _ => panic!("Expected ModelLoadFailed error"),
        }
    }

    #[test]
    fn test_invalid_transition_pending_to_validating() {
        let mut registry = WarmModelRegistry::new();
        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();

        // Cannot go directly from Pending to Validating
        let err = registry.mark_validating("E1_Semantic").unwrap_err();
        match err {
            WarmError::ModelValidationFailed { model_id, reason, .. } => {
                assert_eq!(model_id, "E1_Semantic");
                assert!(reason.contains("Invalid state transition"));
                assert!(reason.contains("Pending"));
            }
            _ => panic!("Expected ModelValidationFailed error"),
        }
    }

    #[test]
    fn test_invalid_transition_pending_to_warm() {
        let mut registry = WarmModelRegistry::new();
        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();

        // Cannot go directly from Pending to Warm
        let err = registry
            .mark_warm("E1_Semantic", test_handle(512 * 1024 * 1024))
            .unwrap_err();
        match err {
            WarmError::ModelValidationFailed { model_id, .. } => {
                assert_eq!(model_id, "E1_Semantic");
            }
            _ => panic!("Expected ModelValidationFailed error"),
        }
    }

    #[test]
    fn test_invalid_transition_loading_to_warm() {
        let mut registry = WarmModelRegistry::new();
        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();
        registry.start_loading("E1_Semantic").unwrap();

        // Cannot go directly from Loading to Warm (must go through Validating)
        let err = registry
            .mark_warm("E1_Semantic", test_handle(512 * 1024 * 1024))
            .unwrap_err();
        match err {
            WarmError::ModelValidationFailed { model_id, .. } => {
                assert_eq!(model_id, "E1_Semantic");
            }
            _ => panic!("Expected ModelValidationFailed error"),
        }
    }

    #[test]
    fn test_mark_failed_from_loading() {
        let mut registry = WarmModelRegistry::new();
        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();
        registry.start_loading("E1_Semantic").unwrap();

        registry
            .mark_failed("E1_Semantic", 102, "CUDA allocation failed")
            .unwrap();

        match registry.get_state("E1_Semantic") {
            Some(WarmModelState::Failed {
                error_code,
                error_message,
            }) => {
                assert_eq!(error_code, 102);
                assert_eq!(error_message, "CUDA allocation failed");
            }
            _ => panic!("Expected Failed state"),
        }
        assert!(registry.get_handle("E1_Semantic").is_none());
    }

    #[test]
    fn test_mark_failed_from_validating() {
        let mut registry = WarmModelRegistry::new();
        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();
        registry.start_loading("E1_Semantic").unwrap();
        registry.mark_validating("E1_Semantic").unwrap();

        registry
            .mark_failed("E1_Semantic", 103, "NaN detected in output")
            .unwrap();

        match registry.get_state("E1_Semantic") {
            Some(WarmModelState::Failed {
                error_code,
                error_message,
            }) => {
                assert_eq!(error_code, 103);
                assert_eq!(error_message, "NaN detected in output");
            }
            _ => panic!("Expected Failed state"),
        }
    }

    #[test]
    fn test_mark_failed_from_pending_fails() {
        let mut registry = WarmModelRegistry::new();
        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();

        // Cannot fail from Pending (must start loading first)
        let err = registry
            .mark_failed("E1_Semantic", 102, "Some error")
            .unwrap_err();
        match err {
            WarmError::ModelLoadFailed { model_id, reason, .. } => {
                assert_eq!(model_id, "E1_Semantic");
                assert!(reason.contains("Pending"));
            }
            _ => panic!("Expected ModelLoadFailed error"),
        }
    }

    #[test]
    fn test_mark_failed_from_warm_fails() {
        let mut registry = WarmModelRegistry::new();
        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();
        registry.start_loading("E1_Semantic").unwrap();
        registry.mark_validating("E1_Semantic").unwrap();
        registry
            .mark_warm("E1_Semantic", test_handle(512 * 1024 * 1024))
            .unwrap();

        // Cannot fail from Warm (model is already successfully loaded)
        let err = registry
            .mark_failed("E1_Semantic", 109, "Context lost")
            .unwrap_err();
        match err {
            WarmError::ModelLoadFailed { model_id, reason, .. } => {
                assert_eq!(model_id, "E1_Semantic");
                assert!(reason.contains("Warm"));
            }
            _ => panic!("Expected ModelLoadFailed error"),
        }
    }

    // ==================== Query Tests ====================

    #[test]
    fn test_get_state_unregistered_model() {
        let registry = WarmModelRegistry::new();
        assert!(registry.get_state("NonExistent").is_none());
    }

    #[test]
    fn test_get_handle_not_warm() {
        let mut registry = WarmModelRegistry::new();
        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();

        // No handle in Pending state
        assert!(registry.get_handle("E1_Semantic").is_none());

        registry.start_loading("E1_Semantic").unwrap();
        // No handle in Loading state
        assert!(registry.get_handle("E1_Semantic").is_none());

        registry.mark_validating("E1_Semantic").unwrap();
        // No handle in Validating state
        assert!(registry.get_handle("E1_Semantic").is_none());
    }

    #[test]
    fn test_get_handle_warm() {
        let mut registry = WarmModelRegistry::new();
        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();
        registry.start_loading("E1_Semantic").unwrap();
        registry.mark_validating("E1_Semantic").unwrap();
        registry
            .mark_warm("E1_Semantic", test_handle(512 * 1024 * 1024))
            .unwrap();

        let handle = registry.get_handle("E1_Semantic").unwrap();
        assert_eq!(handle.vram_address(), 0x1000_0000);
        assert_eq!(handle.allocation_bytes(), 512 * 1024 * 1024);
    }

    // ==================== all_warm and any_failed Tests ====================

    #[test]
    fn test_all_warm_empty_registry() {
        let registry = WarmModelRegistry::new();
        // Empty registry is NOT all warm
        assert!(!registry.all_warm());
    }

    #[test]
    fn test_all_warm_partial() {
        let mut registry = WarmModelRegistry::new();
        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();
        registry
            .register_model("E2_TemporalRecent", 256 * 1024 * 1024, 768)
            .unwrap();

        // Warm first model only
        registry.start_loading("E1_Semantic").unwrap();
        registry.mark_validating("E1_Semantic").unwrap();
        registry
            .mark_warm("E1_Semantic", test_handle(512 * 1024 * 1024))
            .unwrap();

        // Not all warm (E2 is still Pending)
        assert!(!registry.all_warm());
        assert_eq!(registry.warm_count(), 1);
    }

    #[test]
    fn test_all_warm_complete() {
        let mut registry = WarmModelRegistry::new();
        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();
        registry
            .register_model("E2_TemporalRecent", 256 * 1024 * 1024, 768)
            .unwrap();

        // Warm both models
        for model_id in ["E1_Semantic", "E2_TemporalRecent"] {
            registry.start_loading(model_id).unwrap();
            registry.mark_validating(model_id).unwrap();
            registry
                .mark_warm(model_id, test_handle(256 * 1024 * 1024))
                .unwrap();
        }

        assert!(registry.all_warm());
        assert_eq!(registry.warm_count(), 2);
    }

    #[test]
    fn test_any_failed_none() {
        let mut registry = WarmModelRegistry::new();
        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();
        registry.start_loading("E1_Semantic").unwrap();

        assert!(!registry.any_failed());
    }

    #[test]
    fn test_any_failed_one() {
        let mut registry = WarmModelRegistry::new();
        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();
        registry
            .register_model("E2_TemporalRecent", 256 * 1024 * 1024, 768)
            .unwrap();

        registry.start_loading("E1_Semantic").unwrap();
        registry
            .mark_failed("E1_Semantic", 102, "Failed to load")
            .unwrap();

        assert!(registry.any_failed());
    }

    // ==================== loading_order Tests ====================

    #[test]
    fn test_loading_order_descending() {
        let mut registry = WarmModelRegistry::new();

        // Register with different sizes (not in size order)
        registry
            .register_model("Small", 100 * 1024 * 1024, 768)
            .unwrap();
        registry
            .register_model("Large", 500 * 1024 * 1024, 768)
            .unwrap();
        registry
            .register_model("Medium", 250 * 1024 * 1024, 768)
            .unwrap();

        let order = registry.loading_order();

        // Should be sorted largest to smallest
        assert_eq!(order.len(), 3);
        assert_eq!(order[0], "Large");
        assert_eq!(order[1], "Medium");
        assert_eq!(order[2], "Small");
    }

    #[test]
    fn test_loading_order_empty() {
        let registry = WarmModelRegistry::new();
        let order = registry.loading_order();
        assert!(order.is_empty());
    }

    #[test]
    fn test_loading_order_with_all_models() {
        let mut registry = WarmModelRegistry::new();

        // Register models with varying sizes
        let model_sizes = [
            ("E1_Semantic", 500),
            ("E2_TemporalRecent", 200),
            ("E3_TemporalPeriodic", 300),
            ("E4_TemporalPositional", 150),
            ("E10_Multimodal", 800),
        ];

        for (id, size_mb) in model_sizes {
            registry
                .register_model(id, size_mb * 1024 * 1024, 768)
                .unwrap();
        }

        let order = registry.loading_order();

        // E10_Multimodal is largest, should be first
        assert_eq!(order[0], "E10_Multimodal");
        // E1_Semantic is second largest
        assert_eq!(order[1], "E1_Semantic");
        // E4_TemporalPositional is smallest, should be last
        assert_eq!(order[4], "E4_TemporalPositional");
    }

    // ==================== failed_entries Tests ====================

    #[test]
    fn test_failed_entries_empty() {
        let registry = WarmModelRegistry::new();
        assert!(registry.failed_entries().is_empty());
    }

    #[test]
    fn test_failed_entries_multiple() {
        let mut registry = WarmModelRegistry::new();

        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();
        registry
            .register_model("E2_TemporalRecent", 256 * 1024 * 1024, 768)
            .unwrap();
        registry
            .register_model("E3_TemporalPeriodic", 256 * 1024 * 1024, 768)
            .unwrap();

        // Fail E1 and E3
        registry.start_loading("E1_Semantic").unwrap();
        registry
            .mark_failed("E1_Semantic", 102, "CUDA error")
            .unwrap();

        registry.start_loading("E3_TemporalPeriodic").unwrap();
        registry
            .mark_failed("E3_TemporalPeriodic", 104, "VRAM exhausted")
            .unwrap();

        let failed = registry.failed_entries();
        assert_eq!(failed.len(), 2);

        // Find E1 failure
        let e1_failure = failed.iter().find(|(id, _, _)| id == "E1_Semantic");
        assert!(e1_failure.is_some());
        let (_, code, msg) = e1_failure.unwrap();
        assert_eq!(*code, 102);
        assert_eq!(msg, "CUDA error");

        // Find E3 failure
        let e3_failure = failed
            .iter()
            .find(|(id, _, _)| id == "E3_TemporalPeriodic");
        assert!(e3_failure.is_some());
        let (_, code, msg) = e3_failure.unwrap();
        assert_eq!(*code, 104);
        assert_eq!(msg, "VRAM exhausted");
    }

    // ==================== Error Cases Tests ====================

    #[test]
    fn test_operations_on_unregistered_model() {
        let mut registry = WarmModelRegistry::new();

        // All operations should fail with ModelNotRegistered
        let err = registry.start_loading("NonExistent").unwrap_err();
        assert!(matches!(err, WarmError::ModelNotRegistered { .. }));

        let err = registry
            .update_progress("NonExistent", 50, 1000)
            .unwrap_err();
        assert!(matches!(err, WarmError::ModelNotRegistered { .. }));

        let err = registry.mark_validating("NonExistent").unwrap_err();
        assert!(matches!(err, WarmError::ModelNotRegistered { .. }));

        let err = registry
            .mark_warm("NonExistent", test_handle(1000))
            .unwrap_err();
        assert!(matches!(err, WarmError::ModelNotRegistered { .. }));

        let err = registry
            .mark_failed("NonExistent", 102, "Error")
            .unwrap_err();
        assert!(matches!(err, WarmError::ModelNotRegistered { .. }));
    }

    #[test]
    fn test_update_progress_clamps_percent() {
        let mut registry = WarmModelRegistry::new();
        registry
            .register_model("E1_Semantic", 512 * 1024 * 1024, 768)
            .unwrap();
        registry.start_loading("E1_Semantic").unwrap();

        // Progress above 100 should be clamped
        registry.update_progress("E1_Semantic", 150, 1000).unwrap();

        match registry.get_state("E1_Semantic") {
            Some(WarmModelState::Loading {
                progress_percent, ..
            }) => {
                assert_eq!(progress_percent, 100);
            }
            _ => panic!("Expected Loading state"),
        }
    }

    // ==================== Thread Safety Documentation Test ====================

    #[test]
    fn test_shared_registry_type_alias() {
        // Verify SharedWarmRegistry can be created and used
        let registry: SharedWarmRegistry = Arc::new(RwLock::new(WarmModelRegistry::new()));

        // Write access
        {
            let mut reg = registry.write().unwrap();
            reg.register_model("E1_Semantic", 512 * 1024 * 1024, 768)
                .unwrap();
        }

        // Read access (concurrent readers are possible)
        {
            let reg = registry.read().unwrap();
            assert_eq!(reg.model_count(), 1);
        }

        // Verify Arc cloning works
        let registry_clone = Arc::clone(&registry);
        {
            let reg = registry_clone.read().unwrap();
            assert!(reg.get_state("E1_Semantic").is_some());
        }
    }
}
