//! Sentiment score types for emotional text analysis.

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
