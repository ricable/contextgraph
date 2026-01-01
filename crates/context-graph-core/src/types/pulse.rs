//! Cognitive Pulse types for meta-cognitive state tracking.
//!
//! Every MCP tool response includes a Cognitive Pulse header to convey
//! the current system state and suggest next actions.

use serde::{Deserialize, Serialize};

/// Cognitive Pulse header included in all tool responses.
///
/// Provides meta-cognitive state information to help agents
/// understand system state and decide on next actions.
///
/// # Example Response
///
/// ```json
/// {
///   "entropy": 0.45,
///   "coherence": 0.72,
///   "suggested_action": "continue"
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CognitivePulse {
    /// Current entropy level [0.0, 1.0]
    /// Higher values indicate more uncertainty/novelty
    pub entropy: f32,

    /// Current coherence level [0.0, 1.0]
    /// Higher values indicate better integration/understanding
    pub coherence: f32,

    /// Suggested action based on current state
    pub suggested_action: SuggestedAction,
}

impl Default for CognitivePulse {
    fn default() -> Self {
        Self {
            entropy: 0.5,
            coherence: 0.5,
            suggested_action: SuggestedAction::Continue,
        }
    }
}

impl CognitivePulse {
    /// Create a new pulse with the given entropy, coherence, and suggested action.
    ///
    /// If you want the action to be computed automatically based on entropy/coherence,
    /// use `CognitivePulse::computed(entropy, coherence)` instead.
    pub fn new(entropy: f32, coherence: f32, suggested_action: SuggestedAction) -> Self {
        let entropy = entropy.clamp(0.0, 1.0);
        let coherence = coherence.clamp(0.0, 1.0);

        Self {
            entropy,
            coherence,
            suggested_action,
        }
    }

    /// Create a new pulse with the given entropy and coherence.
    /// The suggested action is automatically computed based on the values.
    pub fn computed(entropy: f32, coherence: f32) -> Self {
        let entropy = entropy.clamp(0.0, 1.0);
        let coherence = coherence.clamp(0.0, 1.0);
        let suggested_action = Self::compute_action(entropy, coherence);

        Self {
            entropy,
            coherence,
            suggested_action,
        }
    }

    /// Compute the suggested action based on entropy and coherence.
    fn compute_action(entropy: f32, coherence: f32) -> SuggestedAction {
        match (entropy, coherence) {
            // High entropy, low coherence - needs stabilization
            (e, c) if e > 0.7 && c < 0.4 => SuggestedAction::Stabilize,
            // High entropy, high coherence - exploration frontier
            (e, c) if e > 0.6 && c > 0.5 => SuggestedAction::Explore,
            // Low entropy, high coherence - well understood, ready
            (e, c) if e < 0.4 && c > 0.6 => SuggestedAction::Ready,
            // Low coherence - needs consolidation
            (_, c) if c < 0.4 => SuggestedAction::Consolidate,
            // High entropy - consider pruning
            (e, _) if e > 0.8 => SuggestedAction::Prune,
            // Review needed
            (e, c) if e > 0.5 && c < 0.5 => SuggestedAction::Review,
            // Default: continue
            _ => SuggestedAction::Continue,
        }
    }

    /// Returns true if the system is in a healthy state.
    pub fn is_healthy(&self) -> bool {
        self.entropy < 0.8 && self.coherence > 0.3
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pulse_default() {
        let pulse = CognitivePulse::default();
        assert_eq!(pulse.entropy, 0.5);
        assert_eq!(pulse.coherence, 0.5);
        assert_eq!(pulse.suggested_action, SuggestedAction::Continue);
    }

    #[test]
    fn test_pulse_new_with_action() {
        let pulse = CognitivePulse::new(0.5, 0.7, SuggestedAction::Explore);
        assert_eq!(pulse.entropy, 0.5);
        assert_eq!(pulse.coherence, 0.7);
        assert_eq!(pulse.suggested_action, SuggestedAction::Explore);
    }

    #[test]
    fn test_pulse_computed_stabilize() {
        let pulse = CognitivePulse::computed(0.9, 0.2);
        assert_eq!(pulse.suggested_action, SuggestedAction::Stabilize);
    }

    #[test]
    fn test_pulse_computed_ready() {
        let pulse = CognitivePulse::computed(0.3, 0.8);
        assert_eq!(pulse.suggested_action, SuggestedAction::Ready);
    }

    #[test]
    fn test_is_healthy() {
        let healthy = CognitivePulse::computed(0.5, 0.6);
        assert!(healthy.is_healthy());

        let unhealthy = CognitivePulse::computed(0.9, 0.2);
        assert!(!unhealthy.is_healthy());
    }

    #[test]
    fn test_pulse_clamps_values() {
        let pulse = CognitivePulse::new(1.5, -0.5, SuggestedAction::Continue);
        assert_eq!(pulse.entropy, 1.0);
        assert_eq!(pulse.coherence, 0.0);
    }

    // =======================================================================
    // SuggestedAction Tests (TASK-M02-020)
    // =======================================================================

    #[test]
    fn test_suggested_action_default_is_continue() {
        let action = SuggestedAction::default();
        assert_eq!(action, SuggestedAction::Continue);
    }

    #[test]
    fn test_suggested_action_serde_roundtrip() {
        let actions = [
            SuggestedAction::Ready,
            SuggestedAction::Continue,
            SuggestedAction::Explore,
            SuggestedAction::Consolidate,
            SuggestedAction::Prune,
            SuggestedAction::Stabilize,
            SuggestedAction::Review,
        ];
        for action in actions {
            let json = serde_json::to_string(&action).unwrap();
            let parsed: SuggestedAction = serde_json::from_str(&json).unwrap();
            assert_eq!(action, parsed);
        }
    }

    #[test]
    fn test_suggested_action_serde_snake_case() {
        // Verify snake_case serialization
        let json = serde_json::to_string(&SuggestedAction::Ready).unwrap();
        assert_eq!(json, "\"ready\"");

        let json = serde_json::to_string(&SuggestedAction::Continue).unwrap();
        assert_eq!(json, "\"continue\"");
    }

    #[test]
    fn test_suggested_action_descriptions_not_empty() {
        let actions = [
            SuggestedAction::Ready,
            SuggestedAction::Continue,
            SuggestedAction::Explore,
            SuggestedAction::Consolidate,
            SuggestedAction::Prune,
            SuggestedAction::Stabilize,
            SuggestedAction::Review,
        ];
        for action in actions {
            let desc = action.description();
            assert!(!desc.is_empty(), "{:?} has empty description", action);
            assert!(
                desc.len() > 20,
                "{:?} description too short: {}",
                action,
                desc
            );
        }
    }

    #[test]
    fn test_suggested_action_descriptions_unique() {
        use std::collections::HashSet;
        let actions = [
            SuggestedAction::Ready,
            SuggestedAction::Continue,
            SuggestedAction::Explore,
            SuggestedAction::Consolidate,
            SuggestedAction::Prune,
            SuggestedAction::Stabilize,
            SuggestedAction::Review,
        ];
        let descriptions: HashSet<_> = actions.iter().map(|a| a.description()).collect();
        assert_eq!(
            descriptions.len(),
            actions.len(),
            "Descriptions must be unique"
        );
    }

    #[test]
    fn test_suggested_action_copy_semantics() {
        let action = SuggestedAction::Explore;
        let copied = action; // Copy, not move
        assert_eq!(action, copied);
        assert_eq!(action.description(), copied.description());
    }

    #[test]
    fn test_suggested_action_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SuggestedAction::Ready);
        set.insert(SuggestedAction::Continue);
        set.insert(SuggestedAction::Ready); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_suggested_action_invalid_serde_rejected() {
        // Verify invalid variant is correctly rejected
        let json = "\"unknown_action\"";
        let result: Result<SuggestedAction, _> = serde_json::from_str(json);
        assert!(result.is_err(), "Invalid variant should be rejected");
    }

    #[test]
    fn test_suggested_action_descriptions_contain_mcp_tools() {
        // Verify key actions have MCP tool guidance
        assert!(
            SuggestedAction::Explore
                .description()
                .contains("epistemic_action")
        );
        assert!(SuggestedAction::Explore.description().contains("trigger_dream"));
        assert!(
            SuggestedAction::Consolidate
                .description()
                .contains("trigger_dream")
        );
        assert!(
            SuggestedAction::Consolidate
                .description()
                .contains("merge_concepts")
        );
        assert!(
            SuggestedAction::Stabilize
                .description()
                .contains("critique_context")
        );
        assert!(SuggestedAction::Review.description().contains("reflect_on_memory"));
    }
}
