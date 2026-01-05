//! Entailment query operations for O(1) IS-A hierarchy checks.
//!
//! This module provides query functions that use EntailmentCone containment
//! to efficiently determine hierarchical relationships between concepts.
//!
//! # Algorithm Overview
//!
//! 1. Use BFS from query node to generate candidate nodes
//! 2. For each candidate, check cone containment with O(1) angle check
//! 3. Filter and rank by membership score
//!
//! # Performance Targets
//!
//! - Single containment check: <1ms (O(1) angle computation)
//! - Batch check (1000 pairs): <100ms
//! - BFS + filter (depth 3): <10ms
//!
//! # Constitution Reference
//!
//! - perf.latency.entailment_check: <1ms
//! - M04-T20: Implement entailment query operations
//!
//! # Module Structure
//!
//! - [`types`]: Core types (EntailmentDirection, EntailmentResult, etc.)
//! - [`conversion`]: Type conversion helpers between storage and domain types
//! - [`single`]: Single-pair entailment operations (is_entailed_by, entailment_score)
//! - [`batch`]: Batch entailment operations
//! - [`traversal`]: BFS traversal for entailment queries
//! - [`lca`]: Lowest Common Ancestor operations

mod batch;
mod conversion;
mod lca;
mod single;
mod traversal;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use types::{
    BatchEntailmentResult, EntailmentDirection, EntailmentQueryParams, EntailmentResult, LcaResult,
};

// Re-export all public functions
pub use batch::entailment_check_batch;
pub use lca::lowest_common_ancestor;
pub use single::{entailment_score, is_entailed_by};
pub use traversal::entailment_query;

// Note: conversion helpers are used by submodules via super::conversion
