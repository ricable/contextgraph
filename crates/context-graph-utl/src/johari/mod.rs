//! Johari quadrant classification module.
//!
//! Implements Johari Window classification based on UTL metrics:
//! - Quadrant classification from delta S (surprise) and delta C (coherence)
//! - Retrieval weighting by quadrant
//! - Suggested actions per quadrant
//!
//! # Constitution Reference
//!
//! The Johari quadrants map to UTL states as follows:
//! - **Open**: delta_s < 0.5, delta_c > 0.5 -> direct recall
//! - **Blind**: delta_s > 0.5, delta_c < 0.5 -> discovery (epistemic_action/dream)
//! - **Hidden**: delta_s < 0.5, delta_c < 0.5 -> private (get_neighborhood)
//! - **Unknown**: delta_s > 0.5, delta_c > 0.5 -> frontier
//!
//! # Example
//!
//! ```
//! use context_graph_utl::johari::{JohariClassifier, classify_quadrant, JohariQuadrant};
//! use context_graph_utl::config::JohariConfig;
//!
//! // Using standalone function
//! let quadrant = classify_quadrant(0.3, 0.7);  // Low surprise, high coherence
//! assert_eq!(quadrant, JohariQuadrant::Open);
//!
//! // Using classifier with custom config
//! let config = JohariConfig::default();
//! let classifier = JohariClassifier::new(&config);
//! let quadrant = classifier.classify(0.8, 0.2);  // High surprise, low coherence
//! assert_eq!(quadrant, JohariQuadrant::Blind);
//! ```

mod classifier;
mod retrieval;

pub use classifier::{classify_quadrant, JohariClassifier};
pub use retrieval::{
    get_retrieval_weight, get_suggested_action, QuadrantRetrieval, SuggestedAction,
};

// Re-export JohariQuadrant from core (DO NOT DUPLICATE)
pub use context_graph_core::types::JohariQuadrant;
