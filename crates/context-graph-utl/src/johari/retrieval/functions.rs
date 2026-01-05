//! Standalone retrieval functions for Johari quadrant operations.
//!
//! These functions provide simple mappings between quadrants and their
//! corresponding actions and weights.

use context_graph_core::types::JohariQuadrant;

use super::action::SuggestedAction;

/// Returns the suggested action for a given Johari quadrant.
///
/// This is a convenience function that maps quadrants to their corresponding
/// actions as specified in the UTL constitution.
///
/// # Arguments
///
/// * `quadrant` - The Johari quadrant to get the action for
///
/// # Returns
///
/// The appropriate `SuggestedAction` for the quadrant.
///
/// # Example
///
/// ```
/// use context_graph_utl::johari::{get_suggested_action, SuggestedAction, JohariQuadrant};
///
/// assert_eq!(get_suggested_action(JohariQuadrant::Open), SuggestedAction::DirectRecall);
/// assert_eq!(get_suggested_action(JohariQuadrant::Blind), SuggestedAction::EpistemicAction);
/// assert_eq!(get_suggested_action(JohariQuadrant::Hidden), SuggestedAction::GetNeighborhood);
/// assert_eq!(get_suggested_action(JohariQuadrant::Unknown), SuggestedAction::TriggerDream);
/// ```
#[inline]
pub fn get_suggested_action(quadrant: JohariQuadrant) -> SuggestedAction {
    match quadrant {
        JohariQuadrant::Open => SuggestedAction::DirectRecall,
        JohariQuadrant::Blind => SuggestedAction::EpistemicAction,
        JohariQuadrant::Hidden => SuggestedAction::GetNeighborhood,
        JohariQuadrant::Unknown => SuggestedAction::TriggerDream,
    }
}

/// Returns the retrieval weight for a given Johari quadrant.
///
/// This function returns the default retrieval weight as defined by the
/// `JohariQuadrant::default_retrieval_weight()` method in context-graph-core.
///
/// # Arguments
///
/// * `quadrant` - The Johari quadrant to get the weight for
///
/// # Returns
///
/// A weight value in range [0.0, 1.0]:
/// - Open: 1.0 (full weight)
/// - Blind: 0.7 (high discovery weight)
/// - Hidden: 0.3 (reduced private weight)
/// - Unknown: 0.5 (medium frontier weight)
///
/// # Example
///
/// ```
/// use context_graph_utl::johari::{get_retrieval_weight, JohariQuadrant};
///
/// assert_eq!(get_retrieval_weight(JohariQuadrant::Open), 1.0);
/// assert_eq!(get_retrieval_weight(JohariQuadrant::Hidden), 0.3);
/// ```
#[inline]
pub fn get_retrieval_weight(quadrant: JohariQuadrant) -> f32 {
    quadrant.default_retrieval_weight()
}
