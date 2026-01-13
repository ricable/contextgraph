//! Tests for ego_node module
//!
//! Test organization:
//! - tests_basic: Basic SelfEgoNode and IdentityContinuity tests
//! - tests_pv_history: PurposeVectorHistory tests
//! - tests_cosine: cosine_similarity_13d tests
//! - tests_monitor: IdentityContinuityMonitor tests
//! - tests_ic_factory: IdentityContinuity factory and boundary tests
//! - tests_fsv: Full State Verification lifecycle tests
//! - tests_crisis_detection: Crisis detection and state transition tests (TASK-IDENTITY-P0-004)
//! - tests_crisis_protocol: Crisis protocol execution tests (TASK-IDENTITY-P0-005)

mod tests_basic;
mod tests_cosine;
mod tests_crisis_detection;
mod tests_crisis_protocol;
mod tests_fsv;
mod tests_ic_factory;
mod tests_monitor;
mod tests_pv_history;
