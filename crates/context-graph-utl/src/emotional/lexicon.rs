//! Sentiment lexicon for emotional text analysis.
//!
//! Provides word-level sentiment analysis using a lexicon-based approach.
//! The lexicon maps words to sentiment scores in the range `[-1, 1]`.

use std::collections::HashMap;

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
    words: HashMap<String, f32>,
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

impl Default for SentimentLexicon {
    /// Create a default sentiment lexicon with common emotional words.
    ///
    /// Includes positive words like: excellent, wonderful, amazing, good, etc.
    /// Includes negative words like: terrible, awful, bad, disappointing, etc.
    fn default() -> Self {
        let mut lexicon = Self::new();

        // Highly positive words (0.8-1.0)
        for word in &[
            "excellent",
            "wonderful",
            "amazing",
            "fantastic",
            "brilliant",
            "outstanding",
            "perfect",
            "exceptional",
            "superb",
            "magnificent",
        ] {
            lexicon.add_positive(word, 0.9);
        }

        // Moderately positive words (0.5-0.7)
        for word in &[
            "good",
            "great",
            "nice",
            "pleasant",
            "lovely",
            "delightful",
            "happy",
            "glad",
            "pleased",
            "satisfied",
            "exciting",
            "interesting",
            "impressive",
            "remarkable",
            "valuable",
            "useful",
            "helpful",
        ] {
            lexicon.add_positive(word, 0.6);
        }

        // Mildly positive words (0.2-0.4)
        for word in &[
            "okay",
            "fine",
            "decent",
            "adequate",
            "acceptable",
            "reasonable",
            "positive",
            "favorable",
            "promising",
            "hopeful",
        ] {
            lexicon.add_positive(word, 0.3);
        }

        // Highly negative words (0.8-1.0)
        for word in &[
            "terrible",
            "awful",
            "horrible",
            "dreadful",
            "atrocious",
            "abysmal",
            "disastrous",
            "catastrophic",
            "devastating",
            "appalling",
        ] {
            lexicon.add_negative(word, 0.9);
        }

        // Moderately negative words (0.5-0.7)
        for word in &[
            "bad",
            "poor",
            "disappointing",
            "frustrating",
            "annoying",
            "unpleasant",
            "difficult",
            "problematic",
            "troublesome",
            "concerning",
            "worrying",
            "upsetting",
            "disturbing",
            "confusing",
            "unclear",
        ] {
            lexicon.add_negative(word, 0.6);
        }

        // Mildly negative words (0.2-0.4)
        for word in &[
            "mediocre",
            "subpar",
            "lacking",
            "insufficient",
            "underwhelming",
            "boring",
            "tedious",
            "dull",
            "unremarkable",
            "forgettable",
        ] {
            lexicon.add_negative(word, 0.3);
        }

        // Learning/discovery related positive words
        for word in &[
            "understand",
            "learned",
            "discovered",
            "realized",
            "grasped",
            "mastered",
            "comprehended",
            "insightful",
            "enlightening",
            "illuminating",
            "clarifying",
            "revealing",
            "informative",
            "educational",
        ] {
            lexicon.add_positive(word, 0.5);
        }

        // Cognitive struggle words (mildly negative)
        for word in &[
            "confused",
            "puzzled",
            "bewildered",
            "perplexed",
            "uncertain",
            "unsure",
            "doubtful",
            "struggling",
            "lost",
        ] {
            lexicon.add_negative(word, 0.4);
        }

        lexicon
    }
}

/// Sentiment score representing positive, negative, and neutral components.
///
/// All components are in the range `[0, 1]` and represent the proportion
/// of sentiment in the analyzed text.
///
/// # Example
///
/// ```
/// use context_graph_utl::emotional::SentimentScore;
///
/// let score = SentimentScore::new(0.6, 0.2, 0.2);
///
/// assert!((score.net_sentiment() - 0.4).abs() < 0.001); // 0.6 - 0.2
/// assert!(score.is_positive());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SentimentScore {
    /// Positive sentiment component in range `[0, 1]`.
    pub positive: f32,

    /// Negative sentiment component in range `[0, 1]`.
    pub negative: f32,

    /// Neutral component in range `[0, 1]`.
    /// Represents the proportion of text with no sentiment.
    pub neutral: f32,
}

impl SentimentScore {
    /// Create a new sentiment score.
    ///
    /// # Arguments
    ///
    /// * `positive` - Positive sentiment in `[0, 1]`
    /// * `negative` - Negative sentiment in `[0, 1]`
    /// * `neutral` - Neutral proportion in `[0, 1]`
    pub fn new(positive: f32, negative: f32, neutral: f32) -> Self {
        Self {
            positive: positive.clamp(0.0, 1.0),
            negative: negative.clamp(0.0, 1.0),
            neutral: neutral.clamp(0.0, 1.0),
        }
    }

    /// Create a neutral sentiment score.
    pub fn neutral() -> Self {
        Self {
            positive: 0.0,
            negative: 0.0,
            neutral: 1.0,
        }
    }

    /// Get the net sentiment (positive - negative).
    ///
    /// # Returns
    ///
    /// Net sentiment in range `[-1, 1]`.
    #[inline]
    pub fn net_sentiment(&self) -> f32 {
        (self.positive - self.negative).clamp(-1.0, 1.0)
    }

    /// Get the arousal (emotional intensity).
    ///
    /// Arousal is the sum of positive and negative sentiment,
    /// representing how emotionally charged the content is.
    ///
    /// # Returns
    ///
    /// Arousal in range `[0, 1]`.
    #[inline]
    pub fn arousal(&self) -> f32 {
        ((self.positive + self.negative) / 2.0).clamp(0.0, 1.0)
    }

    /// Check if the sentiment is predominantly positive.
    #[inline]
    pub fn is_positive(&self) -> bool {
        self.positive > self.negative
    }

    /// Check if the sentiment is predominantly negative.
    #[inline]
    pub fn is_negative(&self) -> bool {
        self.negative > self.positive
    }

    /// Check if the sentiment is neutral (no strong positive or negative).
    #[inline]
    pub fn is_neutral(&self) -> bool {
        self.neutral > 0.5 || (self.positive - self.negative).abs() < 0.1
    }

    /// Get the dominant sentiment type as a string.
    pub fn dominant(&self) -> &'static str {
        if self.is_neutral() {
            "neutral"
        } else if self.is_positive() {
            "positive"
        } else {
            "negative"
        }
    }
}

impl Default for SentimentScore {
    fn default() -> Self {
        Self::neutral()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexicon_creation() {
        let lexicon = SentimentLexicon::new();
        assert!(lexicon.is_empty());
        assert_eq!(lexicon.len(), 0);
    }

    #[test]
    fn test_add_positive_word() {
        let mut lexicon = SentimentLexicon::new();
        lexicon.add_positive("happy", 0.7);

        assert!(lexicon.contains("happy"));
        assert_eq!(lexicon.get_sentiment("happy"), Some(0.7));
        assert_eq!(lexicon.get_sentiment("Happy"), Some(0.7)); // Case insensitive
    }

    #[test]
    fn test_add_negative_word() {
        let mut lexicon = SentimentLexicon::new();
        lexicon.add_negative("sad", 0.6);

        assert!(lexicon.contains("sad"));
        assert_eq!(lexicon.get_sentiment("sad"), Some(-0.6));
    }

    #[test]
    fn test_intensity_clamping() {
        let mut lexicon = SentimentLexicon::new();
        lexicon.add_positive("extreme", 1.5); // Over 1.0
        lexicon.add_negative("also_extreme", 2.0);

        assert_eq!(lexicon.get_sentiment("extreme"), Some(1.0));
        assert_eq!(lexicon.get_sentiment("also_extreme"), Some(-1.0));
    }

    #[test]
    fn test_default_lexicon_has_words() {
        let lexicon = SentimentLexicon::default();

        assert!(!lexicon.is_empty());
        assert!(lexicon.contains("excellent")); // Positive word
        assert!(lexicon.contains("terrible")); // Negative word

        // Check sentiment values
        assert!(lexicon.get_sentiment("excellent").unwrap() > 0.0);
        assert!(lexicon.get_sentiment("terrible").unwrap() < 0.0);
    }

    #[test]
    fn test_analyze_positive_text() {
        let lexicon = SentimentLexicon::default();
        let score = lexicon.analyze("This is excellent and wonderful!");

        assert!(score.positive > 0.0);
        assert!(score.is_positive());
        assert!(score.net_sentiment() > 0.0);
    }

    #[test]
    fn test_analyze_negative_text() {
        let lexicon = SentimentLexicon::default();
        let score = lexicon.analyze("This is terrible and disappointing.");

        assert!(score.negative > 0.0);
        assert!(score.is_negative());
        assert!(score.net_sentiment() < 0.0);
    }

    #[test]
    fn test_analyze_mixed_text() {
        let lexicon = SentimentLexicon::default();
        let score = lexicon.analyze("It was good but also bad.");

        assert!(score.positive > 0.0);
        assert!(score.negative > 0.0);
    }

    #[test]
    fn test_analyze_neutral_text() {
        let lexicon = SentimentLexicon::default();
        let score = lexicon.analyze("The sky is blue and water is wet.");

        assert!(score.is_neutral());
        assert!(score.neutral > 0.5);
    }

    #[test]
    fn test_analyze_empty_text() {
        let lexicon = SentimentLexicon::default();
        let score = lexicon.analyze("");

        assert_eq!(score, SentimentScore::neutral());
    }

    #[test]
    fn test_analyze_value() {
        let lexicon = SentimentLexicon::default();

        let positive_value = lexicon.analyze_value("This is wonderful!");
        let negative_value = lexicon.analyze_value("This is terrible!");

        assert!(positive_value > 0.0);
        assert!(negative_value < 0.0);
    }

    #[test]
    fn test_sentiment_score_creation() {
        let score = SentimentScore::new(0.8, 0.2, 0.0);

        assert_eq!(score.positive, 0.8);
        assert_eq!(score.negative, 0.2);
        assert_eq!(score.neutral, 0.0);
    }

    #[test]
    fn test_sentiment_score_clamping() {
        let score = SentimentScore::new(1.5, -0.5, 2.0);

        assert_eq!(score.positive, 1.0);
        assert_eq!(score.negative, 0.0);
        assert_eq!(score.neutral, 1.0);
    }

    #[test]
    fn test_net_sentiment() {
        let positive = SentimentScore::new(0.8, 0.2, 0.0);
        assert_eq!(positive.net_sentiment(), 0.6);

        let negative = SentimentScore::new(0.2, 0.8, 0.0);
        assert_eq!(negative.net_sentiment(), -0.6);

        let neutral = SentimentScore::new(0.5, 0.5, 0.0);
        assert_eq!(neutral.net_sentiment(), 0.0);
    }

    #[test]
    fn test_arousal() {
        let high_arousal = SentimentScore::new(0.8, 0.8, 0.0);
        let low_arousal = SentimentScore::new(0.1, 0.1, 0.8);

        assert!(high_arousal.arousal() > low_arousal.arousal());
    }

    #[test]
    fn test_dominant_sentiment() {
        let positive = SentimentScore::new(0.8, 0.2, 0.0);
        assert_eq!(positive.dominant(), "positive");

        let negative = SentimentScore::new(0.2, 0.8, 0.0);
        assert_eq!(negative.dominant(), "negative");

        let neutral = SentimentScore::new(0.1, 0.1, 0.8);
        assert_eq!(neutral.dominant(), "neutral");
    }

    #[test]
    fn test_lexicon_merge() {
        let mut lexicon1 = SentimentLexicon::new();
        lexicon1.add_positive("word1", 0.5);
        lexicon1.add_positive("shared", 0.3);

        let mut lexicon2 = SentimentLexicon::new();
        lexicon2.add_positive("word2", 0.6);
        lexicon2.add_positive("shared", 0.9);

        lexicon1.merge(&lexicon2);

        assert!(lexicon1.contains("word1"));
        assert!(lexicon1.contains("word2"));
        assert_eq!(lexicon1.get_sentiment("shared"), Some(0.9)); // Overwritten
    }

    #[test]
    fn test_case_insensitivity() {
        let lexicon = SentimentLexicon::default();

        let lower = lexicon.analyze("excellent");
        let upper = lexicon.analyze("EXCELLENT");
        let mixed = lexicon.analyze("ExCeLLenT");

        assert_eq!(lower.positive, upper.positive);
        assert_eq!(lower.positive, mixed.positive);
    }

    #[test]
    fn test_learning_words() {
        let lexicon = SentimentLexicon::default();

        // Learning-related words should be positive
        let score = lexicon.analyze("I understand and learned something enlightening");
        assert!(score.is_positive());
    }

    #[test]
    fn test_confusion_words() {
        let lexicon = SentimentLexicon::default();

        // Confusion words should be negative
        let score = lexicon.analyze("I am confused and puzzled");
        assert!(score.is_negative());
    }
}
