//! Sentiment lexicon implementation for emotional text analysis.

use std::collections::HashMap;

use super::types::SentimentScore;

/// A sentiment lexicon for analyzing text emotion.
///
/// Maps words to sentiment values and provides text analysis capabilities.
/// Words are matched case-insensitively.
///
/// # Example
///
/// ```
/// use context_graph_utl::emotional::{SentimentLexicon, SentimentScore};
///
/// let lexicon = SentimentLexicon::default();
/// let score = lexicon.analyze("This is wonderful and amazing!");
///
/// assert!(score.positive > score.negative);
/// assert!(score.net_sentiment() > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct SentimentLexicon {
    /// Mapping of words to sentiment values.
    /// Positive values indicate positive sentiment, negative indicate negative.
    pub(crate) words: HashMap<String, f32>,
}

impl SentimentLexicon {
    /// Create a new empty sentiment lexicon.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::emotional::SentimentLexicon;
    ///
    /// let mut lexicon = SentimentLexicon::new();
    /// lexicon.add_positive("fantastic", 0.9);
    /// lexicon.add_negative("terrible", 0.8);
    /// ```
    pub fn new() -> Self {
        Self {
            words: HashMap::new(),
        }
    }

    /// Add a positive sentiment word to the lexicon.
    ///
    /// # Arguments
    ///
    /// * `word` - The word to add (case-insensitive)
    /// * `intensity` - Sentiment intensity in range `[0, 1]`
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::emotional::SentimentLexicon;
    ///
    /// let mut lexicon = SentimentLexicon::new();
    /// lexicon.add_positive("excellent", 0.9);
    /// lexicon.add_positive("good", 0.5);
    /// ```
    pub fn add_positive(&mut self, word: &str, intensity: f32) {
        let intensity = intensity.clamp(0.0, 1.0);
        self.words.insert(word.to_lowercase(), intensity);
    }

    /// Add a negative sentiment word to the lexicon.
    ///
    /// # Arguments
    ///
    /// * `word` - The word to add (case-insensitive)
    /// * `intensity` - Sentiment intensity in range `[0, 1]` (stored as negative)
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::emotional::SentimentLexicon;
    ///
    /// let mut lexicon = SentimentLexicon::new();
    /// lexicon.add_negative("awful", 0.9);
    /// lexicon.add_negative("bad", 0.5);
    /// ```
    pub fn add_negative(&mut self, word: &str, intensity: f32) {
        let intensity = intensity.clamp(0.0, 1.0);
        self.words.insert(word.to_lowercase(), -intensity);
    }

    /// Get the sentiment value for a word.
    ///
    /// # Returns
    ///
    /// The sentiment value in range `[-1, 1]`, or `None` if the word is not in the lexicon.
    pub fn get_sentiment(&self, word: &str) -> Option<f32> {
        self.words.get(&word.to_lowercase()).copied()
    }

    /// Check if a word is in the lexicon.
    pub fn contains(&self, word: &str) -> bool {
        self.words.contains_key(&word.to_lowercase())
    }

    /// Get the number of words in the lexicon.
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Check if the lexicon is empty.
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Analyze text and return a sentiment score.
    ///
    /// Tokenizes the text and looks up each word in the lexicon.
    /// Returns aggregated positive/negative/neutral scores.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to analyze
    ///
    /// # Returns
    ///
    /// A [`SentimentScore`] with positive, negative, and neutral components.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_utl::emotional::SentimentLexicon;
    ///
    /// let lexicon = SentimentLexicon::default();
    /// let score = lexicon.analyze("This is an excellent and wonderful day!");
    ///
    /// assert!(score.positive > 0.0);
    /// ```
    pub fn analyze(&self, text: &str) -> SentimentScore {
        if text.is_empty() {
            return SentimentScore::neutral();
        }

        let mut positive_sum = 0.0;
        let mut negative_sum = 0.0;
        let mut word_count = 0;
        let mut matched_count = 0;

        // Simple tokenization: split on non-alphabetic characters
        for word in text.split(|c: char| !c.is_alphabetic()) {
            if word.is_empty() {
                continue;
            }

            word_count += 1;
            let lower = word.to_lowercase();

            if let Some(&sentiment) = self.words.get(&lower) {
                matched_count += 1;
                if sentiment > 0.0 {
                    positive_sum += sentiment;
                } else {
                    negative_sum += -sentiment; // Convert to positive magnitude
                }
            }
        }

        // Normalize by matched word count (or 1 if no matches)
        let normalizer = if matched_count > 0 {
            matched_count as f32
        } else {
            1.0
        };

        let positive = (positive_sum / normalizer).clamp(0.0, 1.0);
        let negative = (negative_sum / normalizer).clamp(0.0, 1.0);

        // Neutral is based on the proportion of unmatched words
        let neutral = if word_count > 0 {
            (word_count - matched_count) as f32 / word_count as f32
        } else {
            1.0
        };

        SentimentScore::new(positive, negative, neutral)
    }

    /// Analyze text and return a single sentiment value.
    ///
    /// This is a convenience method that returns the net sentiment
    /// (positive - negative) as a single value in range `[-1, 1]`.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to analyze
    ///
    /// # Returns
    ///
    /// Net sentiment in range `[-1, 1]`.
    pub fn analyze_value(&self, text: &str) -> f32 {
        self.analyze(text).net_sentiment()
    }

    /// Merge another lexicon into this one.
    ///
    /// Existing words are overwritten by the other lexicon's values.
    pub fn merge(&mut self, other: &SentimentLexicon) {
        for (word, sentiment) in &other.words {
            self.words.insert(word.clone(), *sentiment);
        }
    }
}
