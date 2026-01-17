//! Multi-space clustering types for topic synthesis.
//!
//! This module provides foundational types for clustering memories across
//! the 13 embedding spaces. Used by HDBSCAN (batch) and BIRCH (incremental).
//!
//! # Architecture
//!
//! Per constitution:
//! - ARCH-09: Topic threshold is weighted_agreement >= 2.5
//! - ARCH-04: Temporal embedders (E2-E4) NEVER count toward topic detection
//!
//! # Key Types
//!
//! - [`ClusterMembership`]: Tracks which cluster a memory belongs to per space
//! - [`Cluster`]: Represents a cluster with centroid and quality metrics
//! - [`ClusterError`]: Error types for clustering operations
//! - [`Topic`]: A cross-space concept discovered via weighted clustering
//! - [`TopicProfile`]: Per-space strength profile with weighted agreement
//! - [`TopicPhase`]: Lifecycle phase (Emerging, Stable, Declining, Merging)
//! - [`TopicStability`]: Stability metrics (churn, drift, phase)
//! - [`HDBSCANParams`]: Configuration for HDBSCAN clustering algorithm
//! - [`ClusterSelectionMethod`]: EOM or Leaf cluster selection
//! - [`BIRCHParams`]: Configuration for BIRCH incremental clustering
//! - [`ClusteringFeature`]: CF statistical summary for BIRCH
//! - [`BIRCHTree`]: CF-tree for O(log n) incremental clustering
//! - [`BIRCHNode`]: Internal/leaf node in the CF-tree
//! - [`BIRCHEntry`]: Entry containing CF and optional child pointer
//! - [`MultiSpaceClusterManager`]: Orchestrates clustering across all 13 spaces
//! - [`ManagerParams`]: Configuration for the cluster manager
//! - [`InsertResult`]: Result of inserting a memory into the manager
//! - [`ReclusterResult`]: Result of HDBSCAN batch reclustering

pub mod birch;
pub mod cluster;
pub mod error;
pub mod hdbscan;
pub mod manager;
pub mod membership;
pub mod topic;

pub use birch::{birch_defaults, BIRCHEntry, BIRCHNode, BIRCHParams, BIRCHTree, ClusteringFeature};
pub use cluster::Cluster;
pub use error::ClusterError;
pub use hdbscan::{hdbscan_defaults, ClusterSelectionMethod, HDBSCANClusterer, HDBSCANParams};
pub use manager::{
    manager_defaults, InsertResult, ManagerParams, MultiSpaceClusterManager, ReclusterResult,
    UpdateStatus, DEFAULT_RECLUSTER_THRESHOLD, MAX_WEIGHTED_AGREEMENT, TOPIC_THRESHOLD,
};
pub use membership::ClusterMembership;
pub use topic::{Topic, TopicPhase, TopicProfile, TopicStability};
