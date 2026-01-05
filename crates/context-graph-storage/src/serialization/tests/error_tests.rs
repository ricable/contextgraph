//! Tests for SerializationError type.

use crate::serialization::SerializationError;

#[test]
fn test_serialization_error_serialize_failed() {
    let error = SerializationError::SerializeFailed("test error".to_string());
    let msg = error.to_string();
    assert!(msg.contains("Serialization failed"));
    assert!(msg.contains("test error"));
}

#[test]
fn test_serialization_error_deserialize_failed() {
    let error = SerializationError::DeserializeFailed("corrupt data".to_string());
    let msg = error.to_string();
    assert!(msg.contains("Deserialization failed"));
    assert!(msg.contains("corrupt data"));
}

#[test]
fn test_serialization_error_invalid_embedding_size() {
    let error = SerializationError::InvalidEmbeddingSize {
        expected: 6144,
        actual: 100,
    };
    let msg = error.to_string();
    assert!(msg.contains("expected 6144"));
    assert!(msg.contains("got 100"));
}

#[test]
fn test_serialization_error_invalid_uuid_size() {
    let error = SerializationError::InvalidUuidSize { actual: 10 };
    let msg = error.to_string();
    assert!(msg.contains("expected 16"));
    assert!(msg.contains("got 10"));
}

#[test]
fn test_serialization_error_clone() {
    let original = SerializationError::SerializeFailed("test".to_string());
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn test_serialization_error_partial_eq() {
    let a = SerializationError::InvalidEmbeddingSize {
        expected: 100,
        actual: 50,
    };
    let b = SerializationError::InvalidEmbeddingSize {
        expected: 100,
        actual: 50,
    };
    let c = SerializationError::InvalidEmbeddingSize {
        expected: 100,
        actual: 60,
    };
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_serialization_error_debug() {
    let error = SerializationError::InvalidUuidSize { actual: 8 };
    let debug = format!("{:?}", error);
    assert!(debug.contains("InvalidUuidSize"));
    assert!(debug.contains("8"));
}
