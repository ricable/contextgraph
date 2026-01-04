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

use context_graph_core::types::JohariQuadrant;

use crate::config::JohariConfig;
use crate::error::{UtlError, UtlResult};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// Configurable quadrant-aware retrieval strategy.
///
/// This struct provides a stateful retrieval strategy that uses configuration-defined
/// weights for each quadrant. It allows customization of retrieval behavior beyond
/// the default weights.
///
/// # Example
///
/// ```
/// use context_graph_utl::johari::{QuadrantRetrieval, JohariQuadrant, SuggestedAction};
/// use context_graph_utl::config::JohariConfig;
///
/// let config = JohariConfig::default();
/// let retrieval = QuadrantRetrieval::new(&config);
///
/// let weight = retrieval.get_weight(JohariQuadrant::Open);
/// assert_eq!(weight, 1.0);
///
/// let action = retrieval.get_action(JohariQuadrant::Blind);
/// assert_eq!(action, SuggestedAction::EpistemicAction);
/// ```
#[derive(Debug, Clone)]
pub struct QuadrantRetrieval {
    /// Weight for Open quadrant items
    open_weight: f32,
    /// Weight for Blind quadrant items
    blind_weight: f32,
    /// Weight for Hidden quadrant items
    hidden_weight: f32,
    /// Weight for Unknown quadrant items
    unknown_weight: f32,
}

impl QuadrantRetrieval {
    /// Creates a new retrieval strategy with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Johari configuration containing quadrant weights
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::johari::QuadrantRetrieval;
    /// use context_graph_utl::config::JohariConfig;
    ///
    /// let config = JohariConfig::default();
    /// let retrieval = QuadrantRetrieval::new(&config);
    /// ```
    pub fn new(config: &JohariConfig) -> Self {
        Self {
            open_weight: config.open_weight,
            blind_weight: config.blind_weight,
            hidden_weight: config.hidden_weight,
            unknown_weight: config.unknown_weight,
        }
    }

    /// Creates a retrieval strategy with default weights.
    ///
    /// Uses the default weights from `JohariQuadrant::default_retrieval_weight()`.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::johari::QuadrantRetrieval;
    ///
    /// let retrieval = QuadrantRetrieval::with_default_weights();
    /// ```
    pub fn with_default_weights() -> Self {
        Self {
            open_weight: JohariQuadrant::Open.default_retrieval_weight(),
            blind_weight: JohariQuadrant::Blind.default_retrieval_weight(),
            hidden_weight: JohariQuadrant::Hidden.default_retrieval_weight(),
            unknown_weight: JohariQuadrant::Unknown.default_retrieval_weight(),
        }
    }

    /// Creates a retrieval strategy with custom weights for all quadrants.
    ///
    /// # Arguments
    ///
    /// * `open` - Weight for Open quadrant [0.0, 2.0]
    /// * `blind` - Weight for Blind quadrant [0.0, 2.0]
    /// * `hidden` - Weight for Hidden quadrant [0.0, 2.0]
    /// * `unknown` - Weight for Unknown quadrant [0.0, 2.0]
    ///
    /// # Returns
    ///
    /// A `Result` containing the retrieval strategy or an error if weights are invalid.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::johari::QuadrantRetrieval;
    ///
    /// let retrieval = QuadrantRetrieval::with_custom_weights(1.0, 0.8, 0.4, 0.6)
    ///     .expect("Valid weights");
    /// ```
    pub fn with_custom_weights(
        open: f32,
        blind: f32,
        hidden: f32,
        unknown: f32,
    ) -> UtlResult<Self> {
        // Validate weights are in reasonable range
        for (name, weight) in [
            ("open", open),
            ("blind", blind),
            ("hidden", hidden),
            ("unknown", unknown),
        ] {
            if !(0.0..=2.0).contains(&weight) {
                return Err(UtlError::JohariError(format!(
                    "Weight for {} quadrant must be in [0.0, 2.0], got {}",
                    name, weight
                )));
            }
        }

        Ok(Self {
            open_weight: open,
            blind_weight: blind,
            hidden_weight: hidden,
            unknown_weight: unknown,
        })
    }

    /// Returns the retrieval weight for a given quadrant.
    ///
    /// # Arguments
    ///
    /// * `quadrant` - The Johari quadrant to get the weight for
    ///
    /// # Returns
    ///
    /// The configured weight for the quadrant.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::johari::{QuadrantRetrieval, JohariQuadrant};
    ///
    /// let retrieval = QuadrantRetrieval::with_default_weights();
    /// assert_eq!(retrieval.get_weight(JohariQuadrant::Open), 1.0);
    /// ```
    #[inline]
    pub fn get_weight(&self, quadrant: JohariQuadrant) -> f32 {
        match quadrant {
            JohariQuadrant::Open => self.open_weight,
            JohariQuadrant::Blind => self.blind_weight,
            JohariQuadrant::Hidden => self.hidden_weight,
            JohariQuadrant::Unknown => self.unknown_weight,
        }
    }

    /// Returns the suggested action for a given quadrant.
    ///
    /// This delegates to the standalone `get_suggested_action` function.
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
    /// use context_graph_utl::johari::{QuadrantRetrieval, JohariQuadrant, SuggestedAction};
    ///
    /// let retrieval = QuadrantRetrieval::with_default_weights();
    /// assert_eq!(retrieval.get_action(JohariQuadrant::Open), SuggestedAction::DirectRecall);
    /// ```
    #[inline]
    pub fn get_action(&self, quadrant: JohariQuadrant) -> SuggestedAction {
        get_suggested_action(quadrant)
    }

    /// Applies the retrieval weight to a base score.
    ///
    /// # Arguments
    ///
    /// * `quadrant` - The Johari quadrant
    /// * `base_score` - The base retrieval score to weight
    ///
    /// # Returns
    ///
    /// The weighted score (base_score * quadrant_weight).
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::johari::{QuadrantRetrieval, JohariQuadrant};
    ///
    /// let retrieval = QuadrantRetrieval::with_default_weights();
    ///
    /// // Open quadrant has weight 1.0, so score unchanged
    /// assert_eq!(retrieval.apply_weight(JohariQuadrant::Open, 0.8), 0.8);
    ///
    /// // Hidden quadrant has weight 0.3, so score reduced
    /// let hidden_score = retrieval.apply_weight(JohariQuadrant::Hidden, 0.8);
    /// assert!((hidden_score - 0.24).abs() < 0.001);
    /// ```
    #[inline]
    pub fn apply_weight(&self, quadrant: JohariQuadrant, base_score: f32) -> f32 {
        base_score * self.get_weight(quadrant)
    }

    /// Returns all weights as a tuple (open, blind, hidden, unknown).
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::johari::QuadrantRetrieval;
    ///
    /// let retrieval = QuadrantRetrieval::with_default_weights();
    /// let (open, blind, hidden, unknown) = retrieval.all_weights();
    /// assert_eq!(open, 1.0);
    /// ```
    pub fn all_weights(&self) -> (f32, f32, f32, f32) {
        (
            self.open_weight,
            self.blind_weight,
            self.hidden_weight,
            self.unknown_weight,
        )
    }

    /// Checks if the retrieval strategy should include items from a given quadrant
    /// in default context retrieval.
    ///
    /// # Arguments
    ///
    /// * `quadrant` - The Johari quadrant to check
    ///
    /// # Returns
    ///
    /// `true` if the quadrant should be included by default.
    /// Delegates to `JohariQuadrant::include_in_default_context()`.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::johari::{QuadrantRetrieval, JohariQuadrant};
    ///
    /// let retrieval = QuadrantRetrieval::with_default_weights();
    /// assert!(retrieval.should_include_by_default(JohariQuadrant::Open));
    /// assert!(!retrieval.should_include_by_default(JohariQuadrant::Hidden));
    /// ```
    #[inline]
    pub fn should_include_by_default(&self, quadrant: JohariQuadrant) -> bool {
        quadrant.include_in_default_context()
    }

    /// Returns the weight for the Open quadrant.
    #[inline]
    pub fn open_weight(&self) -> f32 {
        self.open_weight
    }

    /// Returns the weight for the Blind quadrant.
    #[inline]
    pub fn blind_weight(&self) -> f32 {
        self.blind_weight
    }

    /// Returns the weight for the Hidden quadrant.
    #[inline]
    pub fn hidden_weight(&self) -> f32 {
        self.hidden_weight
    }

    /// Returns the weight for the Unknown quadrant.
    #[inline]
    pub fn unknown_weight(&self) -> f32 {
        self.unknown_weight
    }
}

impl Default for QuadrantRetrieval {
    fn default() -> Self {
        Self::with_default_weights()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suggested_action_mapping() {
        assert_eq!(
            get_suggested_action(JohariQuadrant::Open),
            SuggestedAction::DirectRecall
        );
        assert_eq!(
            get_suggested_action(JohariQuadrant::Blind),
            SuggestedAction::EpistemicAction
        );
        assert_eq!(
            get_suggested_action(JohariQuadrant::Hidden),
            SuggestedAction::GetNeighborhood
        );
        assert_eq!(
            get_suggested_action(JohariQuadrant::Unknown),
            SuggestedAction::TriggerDream
        );
    }

    #[test]
    fn test_retrieval_weight() {
        assert_eq!(get_retrieval_weight(JohariQuadrant::Open), 1.0);
        assert_eq!(get_retrieval_weight(JohariQuadrant::Blind), 0.7);
        assert_eq!(get_retrieval_weight(JohariQuadrant::Hidden), 0.3);
        assert_eq!(get_retrieval_weight(JohariQuadrant::Unknown), 0.5);
    }

    #[test]
    fn test_suggested_action_description() {
        let action = SuggestedAction::DirectRecall;
        assert!(action.description().contains("Direct memory recall"));

        let action = SuggestedAction::EpistemicAction;
        assert!(action.description().contains("Epistemic"));
    }

    #[test]
    fn test_suggested_action_urgency() {
        assert_eq!(SuggestedAction::DirectRecall.urgency(), 1.0);
        assert_eq!(SuggestedAction::EpistemicAction.urgency(), 0.7);
        assert_eq!(SuggestedAction::GetNeighborhood.urgency(), 0.5);
        assert_eq!(SuggestedAction::TriggerDream.urgency(), 0.8);
    }

    #[test]
    fn test_suggested_action_display() {
        assert_eq!(format!("{}", SuggestedAction::DirectRecall), "DirectRecall");
        assert_eq!(
            format!("{}", SuggestedAction::EpistemicAction),
            "EpistemicAction"
        );
        assert_eq!(
            format!("{}", SuggestedAction::GetNeighborhood),
            "GetNeighborhood"
        );
        assert_eq!(format!("{}", SuggestedAction::TriggerDream), "TriggerDream");
    }

    #[test]
    fn test_suggested_action_all() {
        let all = SuggestedAction::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&SuggestedAction::DirectRecall));
        assert!(all.contains(&SuggestedAction::EpistemicAction));
        assert!(all.contains(&SuggestedAction::GetNeighborhood));
        assert!(all.contains(&SuggestedAction::TriggerDream));
    }

    #[test]
    fn test_quadrant_retrieval_new() {
        let config = JohariConfig::default();
        let retrieval = QuadrantRetrieval::new(&config);

        assert_eq!(
            retrieval.get_weight(JohariQuadrant::Open),
            config.open_weight
        );
        assert_eq!(
            retrieval.get_weight(JohariQuadrant::Blind),
            config.blind_weight
        );
    }

    #[test]
    fn test_quadrant_retrieval_default_weights() {
        let retrieval = QuadrantRetrieval::with_default_weights();

        assert_eq!(retrieval.get_weight(JohariQuadrant::Open), 1.0);
        assert_eq!(retrieval.get_weight(JohariQuadrant::Blind), 0.7);
        assert_eq!(retrieval.get_weight(JohariQuadrant::Hidden), 0.3);
        assert_eq!(retrieval.get_weight(JohariQuadrant::Unknown), 0.5);
    }

    #[test]
    fn test_quadrant_retrieval_custom_weights() {
        let retrieval =
            QuadrantRetrieval::with_custom_weights(0.9, 0.8, 0.5, 0.6).expect("Valid weights");

        assert_eq!(retrieval.get_weight(JohariQuadrant::Open), 0.9);
        assert_eq!(retrieval.get_weight(JohariQuadrant::Blind), 0.8);
        assert_eq!(retrieval.get_weight(JohariQuadrant::Hidden), 0.5);
        assert_eq!(retrieval.get_weight(JohariQuadrant::Unknown), 0.6);
    }

    #[test]
    fn test_quadrant_retrieval_invalid_weights() {
        // Negative weight
        let result = QuadrantRetrieval::with_custom_weights(-0.1, 0.5, 0.5, 0.5);
        assert!(result.is_err());

        // Weight > 2.0
        let result = QuadrantRetrieval::with_custom_weights(1.0, 2.5, 0.5, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_quadrant_retrieval_get_action() {
        let retrieval = QuadrantRetrieval::with_default_weights();

        assert_eq!(
            retrieval.get_action(JohariQuadrant::Open),
            SuggestedAction::DirectRecall
        );
        assert_eq!(
            retrieval.get_action(JohariQuadrant::Blind),
            SuggestedAction::EpistemicAction
        );
    }

    #[test]
    fn test_quadrant_retrieval_apply_weight() {
        let retrieval = QuadrantRetrieval::with_default_weights();

        // Open has weight 1.0
        assert_eq!(retrieval.apply_weight(JohariQuadrant::Open, 0.8), 0.8);

        // Hidden has weight 0.3
        let weighted = retrieval.apply_weight(JohariQuadrant::Hidden, 0.8);
        assert!((weighted - 0.24).abs() < 0.001);

        // Blind has weight 0.7
        let weighted = retrieval.apply_weight(JohariQuadrant::Blind, 1.0);
        assert!((weighted - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_quadrant_retrieval_all_weights() {
        let retrieval = QuadrantRetrieval::with_default_weights();
        let (open, blind, hidden, unknown) = retrieval.all_weights();

        assert_eq!(open, 1.0);
        assert_eq!(blind, 0.7);
        assert_eq!(hidden, 0.3);
        assert_eq!(unknown, 0.5);
    }

    #[test]
    fn test_quadrant_retrieval_should_include_by_default() {
        let retrieval = QuadrantRetrieval::with_default_weights();

        assert!(retrieval.should_include_by_default(JohariQuadrant::Open));
        assert!(retrieval.should_include_by_default(JohariQuadrant::Blind));
        assert!(!retrieval.should_include_by_default(JohariQuadrant::Hidden));
        assert!(retrieval.should_include_by_default(JohariQuadrant::Unknown));
    }

    #[test]
    fn test_quadrant_retrieval_individual_weight_getters() {
        let retrieval = QuadrantRetrieval::with_default_weights();

        assert_eq!(retrieval.open_weight(), 1.0);
        assert_eq!(retrieval.blind_weight(), 0.7);
        assert_eq!(retrieval.hidden_weight(), 0.3);
        assert_eq!(retrieval.unknown_weight(), 0.5);
    }

    #[test]
    fn test_quadrant_retrieval_default() {
        let retrieval = QuadrantRetrieval::default();
        assert_eq!(retrieval.open_weight(), 1.0);
    }

    #[test]
    fn test_constitution_compliance() {
        // Verify mappings match constitution.yaml specification
        let retrieval = QuadrantRetrieval::with_default_weights();

        // Open -> direct recall
        assert_eq!(
            retrieval.get_action(JohariQuadrant::Open),
            SuggestedAction::DirectRecall
        );

        // Blind -> discovery (epistemic_action/dream)
        assert_eq!(
            retrieval.get_action(JohariQuadrant::Blind),
            SuggestedAction::EpistemicAction
        );

        // Hidden -> private (get_neighborhood)
        assert_eq!(
            retrieval.get_action(JohariQuadrant::Hidden),
            SuggestedAction::GetNeighborhood
        );

        // Unknown -> frontier
        assert_eq!(
            retrieval.get_action(JohariQuadrant::Unknown),
            SuggestedAction::TriggerDream
        );
    }
}
