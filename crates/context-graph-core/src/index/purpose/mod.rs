//! Purpose Pattern Index for teleological alignment queries.
//!
//! # Overview
//!
//! The Purpose Pattern Index (Stage 4) enables retrieval based on
//! teleological alignment - finding memories that serve the same goals.
//!
//! # Components
//!
//! - `error`: Error types for purpose index operations
//! - `entry`: Entry types (`PurposeMetadata`, `PurposeIndexEntry`)
//! - `query`: Query types (`PurposeQuery`, `PurposeQueryTarget`, `PurposeSearchResult`)
//! - `clustering`: K-means clustering for purpose vectors
//! - `hnsw_purpose`: HNSW-backed purpose index with secondary indexes
//!
//! # Usage
//!
//! ```ignore
//! use context_graph_core::index::purpose::{
//!     HnswPurposeIndex, PurposeIndexOps,
//!     PurposeIndexEntry, PurposeMetadata,
//!     PurposeIndexError, PurposeIndexResult,
//!     PurposeQuery, PurposeQueryTarget, PurposeSearchResult,
//!     KMeansConfig, ClusteringResult, PurposeCluster,
//! };
//! ```

pub mod clustering;
pub mod entry;
pub mod error;
pub mod hnsw_purpose;
pub mod query;

#[cfg(test)]
mod tests;

// Re-exports for convenient access
pub use clustering::{
    ClusteringResult, KMeansConfig, KMeansPurposeClustering, PurposeCluster, StandardKMeans,
};
pub use entry::{GoalId, PurposeIndexEntry, PurposeMetadata};
pub use error::{PurposeIndexError, PurposeIndexResult};
pub use hnsw_purpose::{HnswPurposeIndex, PurposeIndexOps};
pub use query::{PurposeQuery, PurposeQueryBuilder, PurposeQueryTarget, PurposeSearchResult};
