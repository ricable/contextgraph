//! Tests for the sentiment lexicon module.

use crate::emotional::lexicon::{SentimentLexicon, SentimentScore};

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
