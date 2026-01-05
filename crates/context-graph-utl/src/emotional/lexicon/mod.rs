//! Sentiment lexicon for emotional text analysis.
//!
//! Provides word-level sentiment analysis using a lexicon-based approach.
//! The lexicon maps words to sentiment scores in the range `[-1, 1]`.

mod default_words;
mod lexicon_impl;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use lexicon_impl::SentimentLexicon;
pub use types::SentimentScore;
