//! Quadrant-aware retrieval strategies and suggested actions.
//!
//! This module provides retrieval weighting and action recommendations based on
//! Johari quadrant classification, implementing the UTL constitution specifications
//! for memory retrieval behavior.
//!
//! # Retrieval Strategies
//!
//! Each quadrant has different retrieval characteristics:
//! - **Open**: Direct recall with full weight (1.0)
//! - **Blind**: Discovery-focused with high weight (configurable)
//! - **Hidden**: Private with reduced weight (configurable)
//! - **Unknown**: Frontier exploration with medium weight (configurable)
//!
//! # Suggested Actions
//!
//! Based on the constitution.yaml specifications:
//! - **Open** -> DirectRecall
//! - **Blind** -> EpistemicAction (epistemic_action/dream)
//! - **Hidden** -> GetNeighborhood (get_neighborhood)
//! - **Unknown** -> TriggerDream (frontier exploration)

mod action;
mod functions;
mod quadrant_retrieval;

#[cfg(test)]
mod tests;

// Re-export all public API for backwards compatibility
pub use action::SuggestedAction;
pub use functions::{get_retrieval_weight, get_suggested_action};
pub use quadrant_retrieval::QuadrantRetrieval;
