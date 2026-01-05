//! Entailment cones for O(1) IS-A hierarchy queries.
//!
//! This module implements entailment cones in hyperbolic space for efficient
//! hierarchical relationship queries. A concept A entails B iff B is
//! contained in A's cone.
//!
//! # Algorithm
//!
//! For a cone with apex `a`, aperture `theta`, and axis `v`:
//! - Point `p` is contained iff angle(p-a, v) <= theta
//! - Ancestors of node = cones that contain the node
//! - Descendants of node = points within node's cone
//!
//! # Performance
//!
//! - Containment check: O(1)
//! - Ancestor lookup: O(k) where k = number of potential ancestors
//! - Descendant lookup: O(n) worst case, O(log n) with spatial index
//!
//! # Components
//!
//! - [`EntailmentCone`]: Cone with apex, aperture, and depth for IS-A queries
//! - [`query`]: Entailment query functions (M04-T20)
//!
//! # Constitution Reference
//!
//! - perf.latency.entailment_check: <1ms
//!
//! # GPU Acceleration
//!
//! CUDA kernels for batch containment checks: TODO: M04-T24

pub mod cones;
pub mod query; // Now a directory module

pub use cones::EntailmentCone;
pub use query::{
    entailment_check_batch, entailment_query, entailment_score, is_entailed_by,
    lowest_common_ancestor, BatchEntailmentResult, EntailmentDirection, EntailmentQueryParams,
    EntailmentResult, LcaResult,
};
