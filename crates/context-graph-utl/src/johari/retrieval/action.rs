//! Suggested action types for Johari quadrant-based retrieval.
//!
//! This module defines the actions recommended for each quadrant as specified
//! in the UTL constitution.

use serde::{Deserialize, Serialize};

/// Suggested action based on Johari quadrant classification.
///
/// These actions represent the recommended retrieval or exploration strategy
/// for each quadrant, as specified in the UTL constitution.
///
/// # Constitution Reference
///
/// ```text
/// Open:    delta_s < 0.5, delta_c > 0.5 -> direct recall
/// Blind:   delta_s > 0.5, delta_c < 0.5 -> discovery (epistemic_action/dream)
/// Hidden:  delta_s < 0.5, delta_c < 0.5 -> private (get_neighborhood)
/// Unknown: delta_s > 0.5, delta_c > 0.5 -> frontier
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SuggestedAction {
    /// Direct memory recall - for Open quadrant.
    ///
    /// The item is well-known and coherent; retrieve it directly
    /// with full confidence.
    DirectRecall,

    /// Epistemic action or dream-based discovery - for Blind quadrant.
    ///
    /// The item is surprising but not coherent; trigger discovery
    /// mechanisms to understand it better.
    EpistemicAction,

    /// Neighborhood exploration - for Hidden quadrant.
    ///
    /// The item is familiar but not coherent; explore neighboring
    /// nodes to find related context.
    GetNeighborhood,

    /// Frontier exploration via dream consolidation - for Unknown quadrant.
    ///
    /// The item is both surprising and coherent; it represents
    /// frontier knowledge that should be explored and consolidated.
    TriggerDream,
}

impl SuggestedAction {
    /// Returns a human-readable description of this action.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::johari::SuggestedAction;
    ///
    /// let action = SuggestedAction::DirectRecall;
    /// assert!(action.description().contains("Direct memory recall"));
    /// ```
    pub fn description(&self) -> &'static str {
        match self {
            Self::DirectRecall => "Direct memory recall - retrieve with full confidence",
            Self::EpistemicAction => "Epistemic action - trigger discovery mechanisms",
            Self::GetNeighborhood => "Get neighborhood - explore related context",
            Self::TriggerDream => "Trigger dream - consolidate frontier knowledge",
        }
    }

    /// Returns the typical urgency level for this action.
    ///
    /// Higher values indicate more urgent actions that should be
    /// prioritized in retrieval queues.
    ///
    /// # Returns
    ///
    /// A value in range [0.0, 1.0]:
    /// - DirectRecall: 1.0 (immediate)
    /// - EpistemicAction: 0.7 (high priority)
    /// - GetNeighborhood: 0.5 (medium priority)
    /// - TriggerDream: 0.8 (high priority for learning)
    pub fn urgency(&self) -> f32 {
        match self {
            Self::DirectRecall => 1.0,
            Self::EpistemicAction => 0.7,
            Self::GetNeighborhood => 0.5,
            Self::TriggerDream => 0.8,
        }
    }

    /// Returns all action variants.
    pub fn all() -> [SuggestedAction; 4] {
        [
            Self::DirectRecall,
            Self::EpistemicAction,
            Self::GetNeighborhood,
            Self::TriggerDream,
        ]
    }
}

impl std::fmt::Display for SuggestedAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DirectRecall => write!(f, "DirectRecall"),
            Self::EpistemicAction => write!(f, "EpistemicAction"),
            Self::GetNeighborhood => write!(f, "GetNeighborhood"),
            Self::TriggerDream => write!(f, "TriggerDream"),
        }
    }
}
