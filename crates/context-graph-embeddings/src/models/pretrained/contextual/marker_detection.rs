//! Context marker detection for enhanced E10 contextual embedding.
//!
//! This module detects contextual indicator tokens to enable
//! context-aware asymmetric embeddings for intent vs context vectors.
//!
//! # Architecture
//!
//! Context markers fall into three categories:
//! 1. **Continuation markers**: Indicate context is being extended
//!    (e.g., "also", "additionally", "furthermore")
//! 2. **Shift markers**: Indicate topic/context change
//!    (e.g., "however", "but", "instead", "on the other hand")
//! 3. **Reference markers**: Indicate reference to prior context
//!    (e.g., "as mentioned", "previously", "this", "that")
//!
//! # Usage
//!
//! ```ignore
//! let markers = detect_context_markers(text, encoding);
//! let context_type = markers.detected_context_type;
//! ```

use tokenizers::Encoding;

/// Result of context marker detection.
#[derive(Debug, Clone, Default)]
pub struct ContextMarkerResult {
    /// Token indices of continuation markers
    pub continuation_marker_indices: Vec<usize>,
    /// Token indices of shift markers
    pub shift_marker_indices: Vec<usize>,
    /// Token indices of reference markers
    pub reference_marker_indices: Vec<usize>,
    /// All marker indices combined
    pub all_marker_indices: Vec<usize>,
    /// Detected dominant context type
    pub detected_context_type: ContextType,
    /// Context strength score [0.0, 1.0]
    pub context_strength: f32,
}

/// Type of contextual relationship detected in text.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ContextType {
    /// Text continues or extends existing context
    Continuation,
    /// Text shifts to a new context/topic
    Shift,
    /// Text references prior context
    Reference,
    /// Mixed or no clear context type detected
    #[default]
    Unknown,
}

/// Continuation indicator patterns - text extends existing context.
const CONTINUATION_INDICATORS: &[&str] = &[
    // Primary continuation
    "also",
    "additionally",
    "furthermore",
    "moreover",
    "besides",
    "plus",
    // Extension patterns
    "and",
    "as well",
    "too",
    "equally",
    "similarly",
    "likewise",
    // Elaboration patterns
    "specifically",
    "particularly",
    "especially",
    "notably",
    "indeed",
    "in fact",
    // Addition patterns
    "along",
    "together",
    "coupled",
    "combined",
    "including",
    "such as",
    // Sequence patterns
    "then",
    "next",
    "subsequently",
    "afterwards",
    "following",
    "continuing",
    // Reinforcement patterns
    "again",
    "still",
    "yet",
    "even",
    "more",
    "further",
];

/// Shift indicator patterns - text changes context/topic.
const SHIFT_INDICATORS: &[&str] = &[
    // Primary contrast
    "however",
    "but",
    "yet",
    "although",
    "though",
    "whereas",
    "while",
    // Alternative patterns
    "instead",
    "rather",
    "alternatively",
    "otherwise",
    "conversely",
    "contrary",
    // Exception patterns
    "except",
    "unless",
    "excluding",
    "apart",
    "aside",
    // Transition patterns
    "meanwhile",
    "separately",
    "independently",
    "distinctly",
    "differently",
    // Topic change patterns
    "anyway",
    "incidentally",
    "speaking",
    "regarding",
    "concerning",
    "about",
    // Concession patterns
    "nevertheless",
    "nonetheless",
    "still",
    "regardless",
    "despite",
    "notwithstanding",
];

/// Reference indicator patterns - text refers to prior context.
const REFERENCE_INDICATORS: &[&str] = &[
    // Direct reference
    "this",
    "that",
    "these",
    "those",
    "it",
    "they",
    "them",
    "such",
    // Explicit back-reference
    "mentioned",
    "stated",
    "described",
    "discussed",
    "noted",
    "said",
    "above",
    "below",
    "prior",
    "previous",
    "earlier",
    "latter",
    "former",
    // Implicit reference
    "same",
    "similar",
    "like",
    "corresponding",
    "related",
    "relevant",
    // Context pointer patterns
    "here",
    "there",
    "where",
    "which",
    "what",
    "who",
    "whom",
    // Temporal reference
    "recently",
    "previously",
    "before",
    "after",
    "already",
    "just",
];

/// Detect context markers in tokenized text.
///
/// # Arguments
///
/// * `text` - Original text content
/// * `encoding` - Tokenizer encoding with offset mappings
///
/// # Returns
///
/// `ContextMarkerResult` containing marker indices and detected context type
pub fn detect_context_markers(text: &str, encoding: &Encoding) -> ContextMarkerResult {
    let text_lower = text.to_lowercase();
    let tokens = encoding.get_tokens();
    let offsets = encoding.get_offsets();

    let mut continuation_indices = Vec::new();
    let mut shift_indices = Vec::new();
    let mut reference_indices = Vec::new();

    // Iterate through tokens and check for context indicators
    for (idx, token) in tokens.iter().enumerate() {
        // Clean token (remove special prefixes from various tokenizers)
        let clean_token = token
            .trim_start_matches('Ġ') // RoBERTa/GPT
            .trim_start_matches("##") // BERT
            .trim_start_matches("▁") // SentencePiece/MPNet
            .to_lowercase();

        if clean_token.is_empty() || clean_token.len() < 2 {
            continue;
        }

        // Check continuation indicators
        for indicator in CONTINUATION_INDICATORS {
            if clean_token == *indicator
                || clean_token.starts_with(indicator)
                || indicator.starts_with(&clean_token)
            {
                continuation_indices.push(idx);
                break;
            }
        }

        // Check shift indicators
        for indicator in SHIFT_INDICATORS {
            if clean_token == *indicator
                || clean_token.starts_with(indicator)
                || indicator.starts_with(&clean_token)
            {
                shift_indices.push(idx);
                break;
            }
        }

        // Check reference indicators
        for indicator in REFERENCE_INDICATORS {
            if clean_token == *indicator
                || clean_token.starts_with(indicator)
                || indicator.starts_with(&clean_token)
            {
                reference_indices.push(idx);
                break;
            }
        }
    }

    // Check multi-word patterns in original text
    let multi_word_continuation = [
        "as well as",
        "in addition",
        "not only",
        "but also",
        "along with",
        "together with",
        "on top of",
    ];

    let multi_word_shift = [
        "on the other hand",
        "in contrast",
        "on the contrary",
        "at the same time",
        "having said that",
        "that being said",
        "be that as it may",
    ];

    let multi_word_reference = [
        "as mentioned",
        "as stated",
        "as described",
        "as noted",
        "with respect to",
        "in reference to",
        "in regard to",
        "in relation to",
    ];

    // Find token indices for multi-word patterns
    for pattern in multi_word_continuation {
        if let Some(pos) = text_lower.find(pattern) {
            add_pattern_indices(pos, offsets, tokens.len(), &mut continuation_indices);
        }
    }

    for pattern in multi_word_shift {
        if let Some(pos) = text_lower.find(pattern) {
            add_pattern_indices(pos, offsets, tokens.len(), &mut shift_indices);
        }
    }

    for pattern in multi_word_reference {
        if let Some(pos) = text_lower.find(pattern) {
            add_pattern_indices(pos, offsets, tokens.len(), &mut reference_indices);
        }
    }

    // Sort and deduplicate indices
    continuation_indices.sort_unstable();
    continuation_indices.dedup();
    shift_indices.sort_unstable();
    shift_indices.dedup();
    reference_indices.sort_unstable();
    reference_indices.dedup();

    // Combine all markers
    let mut all_indices = continuation_indices.clone();
    all_indices.extend(&shift_indices);
    all_indices.extend(&reference_indices);
    all_indices.sort_unstable();
    all_indices.dedup();

    // Determine dominant context type based on counts
    let cont_count = continuation_indices.len();
    let shift_count = shift_indices.len();
    let ref_count = reference_indices.len();
    let max_count = cont_count.max(shift_count).max(ref_count);

    let detected_context_type = if max_count == 0 {
        ContextType::Unknown
    } else if cont_count == max_count {
        ContextType::Continuation
    } else if shift_count == max_count {
        ContextType::Shift
    } else {
        ContextType::Reference
    };

    // Compute context strength based on marker density
    let word_count = text.split_whitespace().count().max(1) as f32;
    let total_markers = all_indices.len() as f32;
    let context_strength = (total_markers / word_count.sqrt()).min(1.0);

    ContextMarkerResult {
        continuation_marker_indices: continuation_indices,
        shift_marker_indices: shift_indices,
        reference_marker_indices: reference_indices,
        all_marker_indices: all_indices,
        detected_context_type,
        context_strength,
    }
}

/// Add token indices that overlap with a multi-word pattern position.
fn add_pattern_indices(
    pos: usize,
    offsets: &[(usize, usize)],
    num_tokens: usize,
    indices: &mut Vec<usize>,
) {
    for (idx, &(start, end)) in offsets.iter().enumerate() {
        if start <= pos && pos < end {
            if !indices.contains(&idx) {
                indices.push(idx);
            }
            // Include next few tokens for the pattern
            for next_idx in (idx + 1)..=(idx + 3).min(num_tokens - 1) {
                if !indices.contains(&next_idx) {
                    indices.push(next_idx);
                }
            }
            break;
        }
    }
}

/// Create weighted attention for intent-focused embedding.
///
/// Focuses on:
/// - CLS token (captures overall intent)
/// - First few tokens (captures subject/action)
/// - Shift markers (captures intent transitions)
///
/// # Arguments
///
/// * `markers` - Detected context markers
/// * `seq_len` - Sequence length
///
/// # Returns
///
/// Vector of token weights for intent-focused pooling
pub fn intent_pooling_weights(markers: &ContextMarkerResult, seq_len: usize) -> Vec<f32> {
    let mut weights = vec![1.0f32; seq_len];

    // Boost CLS token
    if seq_len > 0 {
        weights[0] = 2.0;
    }

    // Boost first few content tokens (subject/action)
    for i in 1..4.min(seq_len) {
        weights[i] = 1.5;
    }

    // Boost shift markers (intent transitions are important)
    for &idx in &markers.shift_marker_indices {
        if idx < seq_len {
            weights[idx] = 1.8;
        }
    }

    // Slightly boost continuation markers
    for &idx in &markers.continuation_marker_indices {
        if idx < seq_len {
            weights[idx] = 1.3;
        }
    }

    weights
}

/// Create weighted attention for context-focused embedding.
///
/// Focuses on:
/// - Reference markers (captures context dependencies)
/// - Continuation markers (captures context flow)
/// - Later tokens (captures outcomes/conclusions)
///
/// # Arguments
///
/// * `markers` - Detected context markers
/// * `seq_len` - Sequence length
///
/// # Returns
///
/// Vector of token weights for context-focused pooling
pub fn context_pooling_weights(markers: &ContextMarkerResult, seq_len: usize) -> Vec<f32> {
    let mut weights = vec![1.0f32; seq_len];

    // CLS token at standard weight
    if seq_len > 0 {
        weights[0] = 1.0;
    }

    // Boost reference markers (context dependencies)
    for &idx in &markers.reference_marker_indices {
        if idx < seq_len {
            weights[idx] = 2.0;
        }
    }

    // Boost continuation markers (context flow)
    for &idx in &markers.continuation_marker_indices {
        if idx < seq_len {
            weights[idx] = 1.8;
        }
    }

    // Boost later tokens (conclusions/outcomes)
    let late_start = seq_len.saturating_sub(5);
    for i in late_start..seq_len {
        if weights[i] < 1.5 {
            weights[i] = 1.5;
        }
    }

    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuation_indicators_not_empty() {
        assert!(!CONTINUATION_INDICATORS.is_empty());
        assert!(CONTINUATION_INDICATORS.len() > 30);
    }

    #[test]
    fn test_shift_indicators_not_empty() {
        assert!(!SHIFT_INDICATORS.is_empty());
        assert!(SHIFT_INDICATORS.len() > 30);
    }

    #[test]
    fn test_reference_indicators_not_empty() {
        assert!(!REFERENCE_INDICATORS.is_empty());
        assert!(REFERENCE_INDICATORS.len() > 30);
    }

    #[test]
    fn test_context_type_default() {
        assert_eq!(ContextType::default(), ContextType::Unknown);
    }

    #[test]
    fn test_marker_result_default() {
        let result = ContextMarkerResult::default();
        assert!(result.continuation_marker_indices.is_empty());
        assert!(result.shift_marker_indices.is_empty());
        assert!(result.reference_marker_indices.is_empty());
        assert!(result.all_marker_indices.is_empty());
        assert_eq!(result.detected_context_type, ContextType::Unknown);
        assert_eq!(result.context_strength, 0.0);
    }

    #[test]
    fn test_intent_weights_boost_first_tokens() {
        let markers = ContextMarkerResult::default();
        let weights = intent_pooling_weights(&markers, 10);
        assert!(weights[0] > 1.0, "CLS should be boosted");
        assert!(weights[1] > 1.0, "First content token should be boosted");
    }

    #[test]
    fn test_context_weights_boost_late_tokens() {
        let markers = ContextMarkerResult::default();
        let weights = context_pooling_weights(&markers, 10);
        assert!(weights[9] > 1.0, "Last token should be boosted");
    }
}
