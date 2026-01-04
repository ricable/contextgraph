//! Emotional weight (wₑ) computation module.
//!
//! Implements emotional salience and attention modulation for the UTL formula:
//! `L = f((ΔS × ΔC) · wₑ · cos φ)`
//!
//! # Components
//!
//! - [`EmotionalWeightCalculator`]: Main calculator combining sentiment and state
//! - [`SentimentLexicon`]: Sentiment-based lexicon matching for text analysis
//! - [`SentimentScore`]: Positive/negative/neutral sentiment scores
//! - [`EmotionalStateTracker`]: State tracking with temporal decay
//! - [`StateDecay`]: Exponential decay configuration toward neutral state
//!
//! # Constitution Reference
//!
//! - `wₑ` range: `[0.5, 1.5]` representing emotional weight
//! - Default value is `1.0` (neutral)
//! - Values `> 1.0` amplify learning, values `< 1.0` dampen it
//!
//! # Example
//!
//! ```
//! use context_graph_utl::emotional::{
//!     EmotionalWeightCalculator, EmotionalState, SentimentLexicon,
//! };
//! use context_graph_utl::config::EmotionalConfig;
//!
//! let config = EmotionalConfig::default();
//! let calculator = EmotionalWeightCalculator::new(&config);
//!
//! // Compute emotional weight from text and current state
//! let weight = calculator.compute_emotional_weight(
//!     "This is exciting and interesting!",
//!     EmotionalState::Curious,
//! );
//!
//! assert!(weight >= 0.5 && weight <= 1.5);
//! ```

mod calculator;
mod lexicon;
mod state;

pub use calculator::EmotionalWeightCalculator;
pub use lexicon::{SentimentLexicon, SentimentScore};
pub use state::{EmotionalStateTracker, StateDecay};

// Re-export EmotionalState from core (DO NOT DUPLICATE)
pub use context_graph_core::types::EmotionalState;
