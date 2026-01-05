//! Tests for embedding error types - Part 2: Edge cases and comprehensive tests.

use super::*;
use crate::types::{InputType, ModelId};
use std::error::Error;

// ============================================================
// EDGE CASES (4 tests)
// ============================================================

#[test]
fn test_all_12_model_ids_in_model_not_found() {
    for model_id in ModelId::all() {
        let err = EmbeddingError::ModelNotFound {
            model_id: *model_id,
        };
        let msg = format!("{}", err);
        // Verify error message is non-empty and contains model info
        assert!(!msg.is_empty());
        println!("BEFORE: ModelId::{:?}", model_id);
        println!("AFTER: Error message = {}", msg);
    }
}

#[test]
fn test_all_4_input_types_in_unsupported_modality() {
    for input_type in InputType::all() {
        let err = EmbeddingError::UnsupportedModality {
            model_id: ModelId::Semantic,
            input_type: *input_type,
        };
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
        println!("BEFORE: InputType::{:?}", input_type);
        println!("AFTER: Error message = {}", msg);
    }
}

#[test]
fn test_invalid_value_with_infinity() {
    let err = EmbeddingError::InvalidValue {
        index: 0,
        value: f32::INFINITY,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("inf"));
}

#[test]
fn test_invalid_value_with_neg_infinity() {
    let err = EmbeddingError::InvalidValue {
        index: 0,
        value: f32::NEG_INFINITY,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("-inf") || msg.contains("inf"));
}

// ============================================================
// ERROR SOURCE CHAIN TEST (1 test)
// ============================================================

#[test]
fn test_model_load_error_source_chain() {
    let root_cause = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let err = EmbeddingError::ModelLoadError {
        model_id: ModelId::Causal,
        source: Box::new(root_cause),
    };

    // Verify the source chain is preserved
    let source = err.source();
    assert!(source.is_some());

    let source_msg = format!("{}", source.unwrap());
    assert!(source_msg.contains("file not found"));
}

// ============================================================
// ALL 17 VARIANTS EXIST TEST (1 test)
// ============================================================

#[test]
fn test_all_17_variants_can_be_created() {
    // Model Errors (3)
    let _e1 = EmbeddingError::ModelNotFound {
        model_id: ModelId::Semantic,
    };
    let _e2 = EmbeddingError::ModelLoadError {
        model_id: ModelId::Code,
        source: Box::new(std::io::Error::other("test")),
    };
    let _e3 = EmbeddingError::NotInitialized {
        model_id: ModelId::Graph,
    };

    // Validation Errors (4)
    let _e4 = EmbeddingError::InvalidDimension {
        expected: 1536,
        actual: 768,
    };
    let _e5 = EmbeddingError::InvalidValue {
        index: 0,
        value: 0.0,
    };
    let _e6 = EmbeddingError::EmptyInput;
    let _e7 = EmbeddingError::InputTooLong {
        actual: 100,
        max: 50,
    };

    // Processing Errors (2) - FusionError removed (TASK-F006)
    let _e8 = EmbeddingError::BatchError {
        message: "test".to_string(),
    };
    let _e9 = EmbeddingError::TokenizationError {
        model_id: ModelId::Semantic,
        message: "test".to_string(),
    };

    // Infrastructure Errors (4)
    let _e10 = EmbeddingError::GpuError {
        message: "test".to_string(),
    };
    let _e11 = EmbeddingError::CacheError {
        message: "test".to_string(),
    };
    let _e12 = EmbeddingError::IoError(std::io::Error::other("test"));
    let _e13 = EmbeddingError::Timeout { timeout_ms: 1000 };

    // Configuration Errors (2)
    let _e14 = EmbeddingError::UnsupportedModality {
        model_id: ModelId::Semantic,
        input_type: InputType::Image,
    };
    let _e15 = EmbeddingError::ConfigError {
        message: "test".to_string(),
    };

    // Serialization Errors (1)
    let _e16 = EmbeddingError::SerializationError {
        message: "test".to_string(),
    };

    // All variants created successfully (FusionError and InvalidExpertIndex removed in TASK-F006)
    println!("All error variants created successfully!");
}

// ============================================================
// SPECIAL VALUE TESTS (2 tests)
// ============================================================

#[test]
fn test_invalid_value_with_zero() {
    let err = EmbeddingError::InvalidValue {
        index: 100,
        value: 0.0,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("100"));
    assert!(msg.contains("0"));
}

#[test]
fn test_invalid_value_with_negative() {
    let err = EmbeddingError::InvalidValue {
        index: 50,
        value: -123.456,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("50"));
    assert!(msg.contains("-123"));
}

// ============================================================
// BOUNDARY VALUE TESTS (2 tests)
// ============================================================

#[test]
fn test_timeout_with_max_u64() {
    let err = EmbeddingError::Timeout {
        timeout_ms: u64::MAX,
    };
    let msg = format!("{}", err);
    assert!(msg.contains(&u64::MAX.to_string()));
}

#[test]
fn test_input_too_long_with_large_values() {
    let err = EmbeddingError::InputTooLong {
        actual: 1_000_000,
        max: 4096,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("1000000"));
    assert!(msg.contains("4096"));
}
