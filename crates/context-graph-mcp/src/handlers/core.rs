//! Core Handlers struct and dispatch logic.
//!
//! TASK-S001: Updated to use TeleologicalMemoryStore and MultiArrayEmbeddingProvider.
//! TASK-S003: Added GoalAlignmentCalculator for purpose/goal operations.
//! TASK-S004: Added JohariTransitionManager for johari/* handlers.
//! NO BACKWARDS COMPATIBILITY with legacy MemoryStore trait.

use std::sync::Arc;

use parking_lot::RwLock;
use tracing::debug;

use context_graph_core::alignment::GoalAlignmentCalculator;
use context_graph_core::johari::{DynDefaultJohariManager, JohariTransitionManager};
use context_graph_core::purpose::GoalHierarchy;
use context_graph_core::traits::{MultiArrayEmbeddingProvider, TeleologicalMemoryStore, UtlProcessor};

use crate::protocol::{error_codes, methods, JsonRpcRequest, JsonRpcResponse};

/// Request handlers for MCP protocol.
///
/// Uses TeleologicalMemoryStore for 13-embedding fingerprint storage
/// and MultiArrayEmbeddingProvider for generating all 13 embeddings.
/// TASK-S003: Added GoalAlignmentCalculator and GoalHierarchy for purpose operations.
/// TASK-S004: Added JohariTransitionManager for johari/* operations.
pub struct Handlers {
    /// Teleological memory store - stores TeleologicalFingerprint with 13 embeddings.
    /// NO legacy MemoryStore support.
    pub(super) teleological_store: Arc<dyn TeleologicalMemoryStore>,

    /// UTL processor for computing learning metrics.
    pub(super) utl_processor: Arc<dyn UtlProcessor>,

    /// Multi-array embedding provider - generates all 13 embeddings per content.
    /// NO legacy single-embedding support.
    pub(super) multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,

    /// Goal alignment calculator - computes alignment between fingerprints and goal hierarchy.
    /// TASK-S003: Required for purpose/north_star_alignment and purpose/drift_check.
    pub(super) alignment_calculator: Arc<dyn GoalAlignmentCalculator>,

    /// Goal hierarchy - defines North Star and sub-goals.
    /// TASK-S003: RwLock allows runtime updates via purpose/north_star_update.
    pub(super) goal_hierarchy: Arc<RwLock<GoalHierarchy>>,

    /// Johari transition manager - manages Johari quadrant transitions.
    /// TASK-S004: Required for johari/* handlers.
    pub(super) johari_manager: Arc<dyn JohariTransitionManager>,
}

impl Handlers {
    /// Create new handlers with teleological dependencies.
    ///
    /// # Arguments
    /// * `teleological_store` - Store for TeleologicalFingerprint (TASK-F008)
    /// * `utl_processor` - UTL metrics computation
    /// * `multi_array_provider` - 13-embedding generator (TASK-F007)
    /// * `alignment_calculator` - Goal alignment calculator (TASK-S003)
    /// * `goal_hierarchy` - Goal hierarchy with North Star (TASK-S003)
    pub fn new(
        teleological_store: Arc<dyn TeleologicalMemoryStore>,
        utl_processor: Arc<dyn UtlProcessor>,
        multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
        alignment_calculator: Arc<dyn GoalAlignmentCalculator>,
        goal_hierarchy: GoalHierarchy,
    ) -> Self {
        // TASK-S004: Create Johari manager from teleological store
        let johari_manager: Arc<dyn JohariTransitionManager> =
            Arc::new(DynDefaultJohariManager::new(teleological_store.clone()));

        Self {
            teleological_store,
            utl_processor,
            multi_array_provider,
            alignment_calculator,
            goal_hierarchy: Arc::new(RwLock::new(goal_hierarchy)),
            johari_manager,
        }
    }

    /// Create new handlers with shared goal hierarchy reference.
    ///
    /// Use this variant when you need to share the goal hierarchy across
    /// multiple handler instances or access it from outside.
    ///
    /// # Arguments
    /// * `teleological_store` - Store for TeleologicalFingerprint (TASK-F008)
    /// * `utl_processor` - UTL metrics computation
    /// * `multi_array_provider` - 13-embedding generator (TASK-F007)
    /// * `alignment_calculator` - Goal alignment calculator (TASK-S003)
    /// * `goal_hierarchy` - Shared goal hierarchy reference (TASK-S003)
    pub fn with_shared_hierarchy(
        teleological_store: Arc<dyn TeleologicalMemoryStore>,
        utl_processor: Arc<dyn UtlProcessor>,
        multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
        alignment_calculator: Arc<dyn GoalAlignmentCalculator>,
        goal_hierarchy: Arc<RwLock<GoalHierarchy>>,
    ) -> Self {
        // TASK-S004: Create Johari manager from teleological store
        let johari_manager: Arc<dyn JohariTransitionManager> =
            Arc::new(DynDefaultJohariManager::new(teleological_store.clone()));

        Self {
            teleological_store,
            utl_processor,
            multi_array_provider,
            alignment_calculator,
            goal_hierarchy,
            johari_manager,
        }
    }

    /// Create new handlers with explicit Johari manager.
    ///
    /// Use this variant when you need to provide a custom JohariTransitionManager
    /// implementation or share it across multiple handler instances.
    ///
    /// # Arguments
    /// * `teleological_store` - Store for TeleologicalFingerprint (TASK-F008)
    /// * `utl_processor` - UTL metrics computation
    /// * `multi_array_provider` - 13-embedding generator (TASK-F007)
    /// * `alignment_calculator` - Goal alignment calculator (TASK-S003)
    /// * `goal_hierarchy` - Shared goal hierarchy reference (TASK-S003)
    /// * `johari_manager` - Shared Johari manager reference (TASK-S004)
    pub fn with_johari_manager(
        teleological_store: Arc<dyn TeleologicalMemoryStore>,
        utl_processor: Arc<dyn UtlProcessor>,
        multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
        alignment_calculator: Arc<dyn GoalAlignmentCalculator>,
        goal_hierarchy: Arc<RwLock<GoalHierarchy>>,
        johari_manager: Arc<dyn JohariTransitionManager>,
    ) -> Self {
        Self {
            teleological_store,
            utl_processor,
            multi_array_provider,
            alignment_calculator,
            goal_hierarchy,
            johari_manager,
        }
    }

    /// Dispatch a request to the appropriate handler.
    pub async fn dispatch(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        debug!("Dispatching method: {}", request.method);

        match request.method.as_str() {
            // MCP lifecycle methods
            methods::INITIALIZE => self.handle_initialize(request.id).await,
            "notifications/initialized" => self.handle_initialized_notification(),
            methods::SHUTDOWN => self.handle_shutdown(request.id).await,

            // MCP tools protocol
            methods::TOOLS_LIST => self.handle_tools_list(request.id).await,
            methods::TOOLS_CALL => self.handle_tools_call(request.id, request.params).await,

            // Legacy direct methods (kept for backward compatibility)
            methods::MEMORY_STORE => self.handle_memory_store(request.id, request.params).await,
            methods::MEMORY_RETRIEVE => {
                self.handle_memory_retrieve(request.id, request.params)
                    .await
            }
            methods::MEMORY_SEARCH => self.handle_memory_search(request.id, request.params).await,
            methods::MEMORY_DELETE => self.handle_memory_delete(request.id, request.params).await,

            // Search operations (TASK-S002)
            methods::SEARCH_MULTI => self.handle_search_multi(request.id, request.params).await,
            methods::SEARCH_SINGLE_SPACE => {
                self.handle_search_single_space(request.id, request.params)
                    .await
            }
            methods::SEARCH_BY_PURPOSE => {
                self.handle_search_by_purpose(request.id, request.params)
                    .await
            }
            methods::SEARCH_WEIGHT_PROFILES => self.handle_get_weight_profiles(request.id).await,

            // Purpose/goal operations (TASK-S003)
            methods::PURPOSE_QUERY => self.handle_purpose_query(request.id, request.params).await,
            methods::PURPOSE_NORTH_STAR_ALIGNMENT => {
                self.handle_north_star_alignment(request.id, request.params)
                    .await
            }
            methods::GOAL_HIERARCHY_QUERY => {
                self.handle_goal_hierarchy_query(request.id, request.params)
                    .await
            }
            methods::GOAL_ALIGNED_MEMORIES => {
                self.handle_goal_aligned_memories(request.id, request.params)
                    .await
            }
            methods::PURPOSE_DRIFT_CHECK => {
                self.handle_purpose_drift_check(request.id, request.params)
                    .await
            }
            methods::NORTH_STAR_UPDATE => {
                self.handle_north_star_update(request.id, request.params)
                    .await
            }

            // Johari operations (TASK-S004)
            methods::JOHARI_GET_DISTRIBUTION => {
                self.handle_johari_get_distribution(request.id, request.params)
                    .await
            }
            methods::JOHARI_FIND_BY_QUADRANT => {
                self.handle_johari_find_by_quadrant(request.id, request.params)
                    .await
            }
            methods::JOHARI_TRANSITION => {
                self.handle_johari_transition(request.id, request.params)
                    .await
            }
            methods::JOHARI_TRANSITION_BATCH => {
                self.handle_johari_transition_batch(request.id, request.params)
                    .await
            }
            methods::JOHARI_CROSS_SPACE_ANALYSIS => {
                self.handle_johari_cross_space_analysis(request.id, request.params)
                    .await
            }
            methods::JOHARI_TRANSITION_PROBABILITIES => {
                self.handle_johari_transition_probabilities(request.id, request.params)
                    .await
            }

            methods::UTL_COMPUTE => self.handle_utl_compute(request.id, request.params).await,
            methods::UTL_METRICS => self.handle_utl_metrics(request.id, request.params).await,
            methods::SYSTEM_STATUS => self.handle_system_status(request.id).await,
            methods::SYSTEM_HEALTH => self.handle_system_health(request.id).await,
            _ => JsonRpcResponse::error(
                request.id,
                error_codes::METHOD_NOT_FOUND,
                format!("Method not found: {}", request.method),
            ),
        }
    }
}
