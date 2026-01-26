//! Graph relationship discovery agent using local LLM.
//!
//! This crate provides LLM-based graph relationship discovery for the
//! context-graph system. It mirrors the architecture of `context-graph-causal-agent`
//! but focuses on structural relationships (imports, dependencies, references, etc.)
//! rather than causal relationships.
//!
//! # Architecture
//!
//! The graph agent shares the Qwen2.5-3B LLM with the causal agent via `Arc`,
//! avoiding duplicate VRAM usage. It uses the same ChatML prompt format with
//! different system prompts for graph relationship detection.
//!
//! # E8 Dimension Change
//!
//! E8 has been upgraded from MiniLM (384D) to e5-large-v2 (1024D):
//! - Shares the model with E1, no extra VRAM
//! - Better semantic understanding for graph relationships
//! - Asymmetric source/target embeddings via learned projections
//!
//! # Relationship Types
//!
//! The agent detects 8 relationship types:
//! - **imports**: A imports/uses B
//! - **depends_on**: A depends on B
//! - **references**: A references B
//! - **calls**: A calls B
//! - **implements**: A implements B
//! - **extends**: A extends B
//! - **contains**: A contains B
//! - **used_by**: A is used by B
//!
//! # Pipeline Flow
//!
//! ```text
//! Memories -> Scanner -> Candidates -> LLM Analysis -> Activator -> Graph Edges
//!               |            |              |              |
//!           Heuristics   E1 Similarity   Qwen2.5-3B    E8 1024D
//! ```
//!
//! # Usage
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use context_graph_graph_agent::{GraphDiscoveryService, GraphDiscoveryConfig};
//! use context_graph_causal_agent::CausalDiscoveryLLM;
//!
//! async fn example() {
//!     // Create shared LLM (from causal-agent)
//!     let llm = Arc::new(CausalDiscoveryLLM::new().unwrap());
//!     llm.load().await.unwrap();
//!
//!     // Create graph discovery service
//!     let service = GraphDiscoveryService::new(llm);
//!
//!     // Run discovery cycle
//!     // let result = service.run_discovery_cycle(&memories).await?;
//! }
//! ```

pub mod activator;
pub mod error;
pub mod llm;
pub mod scanner;
pub mod service;
#[cfg(feature = "test-utils")]
pub mod stubs;
pub mod types;

// Re-exports for convenience
pub use activator::{ActivatorConfig, ActivationStats, E8Activator, GraphEdge, GraphStorage};
pub use error::{GraphAgentError, GraphAgentResult};
pub use llm::{prompt::GraphPromptBuilder, GraphRelationshipLLM};
pub use scanner::{MemoryScanner, ScannerConfig};
pub use service::{DiscoveryCycleResult, GraphDiscoveryConfig, GraphDiscoveryService, ServiceStatus};
pub use types::{
    GraphAnalysisResult, GraphCandidate, GraphLinkDirection, GraphMarkers, MemoryForGraphAnalysis,
    RelationshipType,
};

// Test utilities (feature-gated)
#[cfg(feature = "test-utils")]
pub use stubs::create_stub_graph_discovery_service;

/// E8 embedding dimension (upgraded from 384 to 1024).
///
/// Uses e5-large-v2 instead of MiniLM for better graph understanding.
/// Shares the model with E1 for zero additional VRAM.
pub const E8_DIMENSION: usize = 1024;
