//! Configurable quadrant-aware retrieval strategy.
//!
//! This module provides a stateful retrieval strategy that uses configuration-defined
//! weights for each quadrant.

use context_graph_core::types::JohariQuadrant;

use crate::config::JohariConfig;
use crate::error::{UtlError, UtlResult};

use super::action::SuggestedAction;
use super::functions::get_suggested_action;

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
