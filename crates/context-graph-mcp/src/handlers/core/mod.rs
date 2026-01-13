//! Core Handlers struct and dispatch logic.
//!
//! TASK-S001: Updated to use TeleologicalMemoryStore and MultiArrayEmbeddingProvider.
//! TASK-S003: Added GoalAlignmentCalculator for purpose/goal operations.
//! TASK-S004: Added JohariTransitionManager for johari/* handlers.
//! TASK-S005: Added MetaUtlTracker for meta_utl/* handlers.
//! TASK-GWT-001: Added GWT/Kuramoto provider traits for consciousness operations.
//! NO BACKWARDS COMPATIBILITY with legacy MemoryStore trait.
//!
//! # Module Organization
//!
//! This module is split into submodules for better maintainability:
//! - `types`: Type definitions (PredictionType, Domain, MetaLearningEvent, etc.)
//! - `meta_utl_tracker`: MetaUtlTracker struct and implementation
//! - `handlers`: Handlers struct definition and constructors
//! - `dispatch`: Request dispatch logic

mod dispatch;
pub mod event_log;
mod handlers;
pub mod bayesian_optimizer;
pub mod lambda_correction;
mod meta_utl_tracker;
pub mod meta_utl_service;
mod types;

// Re-export all public types for backwards compatibility
pub use self::handlers::Handlers;
// TASK-METAUTL-P0-002: Lambda correction types (will be used in TASK-METAUTL-P0-005/006)
#[allow(unused_imports)]
pub use self::lambda_correction::{
    AdaptiveLambdaWeights, SelfCorrectingLambda, ACH_BASELINE, ACH_MAX,
};
pub use self::meta_utl_tracker::MetaUtlTracker;
// TASK-METAUTL-P0-001/002: Meta-UTL types (will be used in later tasks)
#[allow(unused_imports)]
pub use self::types::{LambdaAdjustment, PredictionType, SelfCorrectionConfig, StoredPrediction};
// TASK-METAUTL-P0-003: Bayesian optimization escalation types (will be used in TASK-METAUTL-P0-005/006)
#[allow(unused_imports)]
pub use self::bayesian_optimizer::{
    EscalationHandler, EscalationManager, EscalationResult, EscalationStats, EscalationStatus,
    GpObservation, SimpleGaussianProcess, HUMAN_ESCALATION_THRESHOLD, INITIAL_SAMPLES,
    MAX_BO_ITERATIONS,
};
// TASK-METAUTL-P0-004: Event log types (will be used in TASK-METAUTL-P0-005/006)
#[allow(unused_imports)]
pub use self::event_log::{
    EventLogQuery, EventLogStats, EventTypeCount, MetaLearningEventLog, MetaLearningLogger,
    DEFAULT_MAX_EVENTS, DEFAULT_RETENTION_DAYS,
};
// TASK-METAUTL-P0-004: Domain and MetaLearningEvent types
#[allow(unused_imports)]
pub use self::types::{Domain, MetaLearningEvent, MetaLearningEventType};
