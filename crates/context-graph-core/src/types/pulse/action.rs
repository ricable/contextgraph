//! Suggested action types based on cognitive state.

use serde::{Deserialize, Serialize};

/// Action suggestions based on cognitive state.
///
/// These suggest what the agent should consider doing next
/// based on the current entropy/coherence balance.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum SuggestedAction {
    /// System ready for new input - low entropy, high coherence.
    Ready,
    /// Continue current activity - balanced state (DEFAULT).
    #[default]
    Continue,
    /// Explore new knowledge - use epistemic_action or trigger_dream(rem).
    Explore,
    /// Consolidate knowledge - use trigger_dream(nrem) or merge_concepts.
    Consolidate,
    /// Prune redundant information - review curation_tasks.
    Prune,
    /// Stabilize context - use trigger_dream or critique_context.
    Stabilize,
    /// Review context - use critique_context or reflect_on_memory.
    Review,
}

impl SuggestedAction {
    /// Returns a human-readable description with MCP tool guidance.
    ///
    /// Each description includes actionable guidance for which MCP tools
    /// to use based on the current cognitive state.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Ready => "System ready for new input - low entropy, high coherence",
            Self::Continue => "Continue current activity - balanced state",
            Self::Explore => "Explore new knowledge - use epistemic_action or trigger_dream(rem)",
            Self::Consolidate => {
                "Consolidate knowledge - use trigger_dream(nrem) or merge_concepts"
            }
            Self::Prune => "Prune redundant information - review curation_tasks",
            Self::Stabilize => "Stabilize context - use trigger_dream or critique_context",
            Self::Review => "Review context - use critique_context or reflect_on_memory",
        }
    }
}
