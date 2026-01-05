//! Tests for EmbeddingVector serialization.

use context_graph_core::types::{EmbeddingVector, DEFAULT_EMBEDDING_DIM};

use crate::serialization::{deserialize_embedding, serialize_embedding, SerializationError};

/// Create a valid normalized embedding vector.
fn create_normalized_embedding(dim: usize) -> EmbeddingVector {
    let val = 1.0 / (dim as f32).sqrt();
    vec![val; dim]
}

#[test]
fn test_embedding_roundtrip() {
    let embedding = create_normalized_embedding(DEFAULT_EMBEDDING_DIM);
    let bytes = serialize_embedding(&embedding);
    let restored = deserialize_embedding(&bytes).expect("deserialize failed");

    assert_eq!(embedding.len(), restored.len());
    for (orig, rest) in embedding.iter().zip(restored.iter()) {
        assert_eq!(*orig, *rest, "f32 values must be exactly preserved");
    }
}

#[test]
fn test_embedding_size_exact() {
    let embedding = vec![0.5_f32; 1536];
    let bytes = serialize_embedding(&embedding);
    assert_eq!(bytes.len(), 1536 * 4, "Embedding should be dim * 4 bytes");
}

#[test]
fn test_embedding_invalid_size() {
    let bytes = vec![0u8; 13]; // Not divisible by 4
    let result = deserialize_embedding(&bytes);
    assert!(matches!(
        result,
        Err(SerializationError::InvalidEmbeddingSize { .. })
    ));
}

#[test]
fn test_embedding_empty() {
    let embedding: EmbeddingVector = vec![];
    let bytes = serialize_embedding(&embedding);
    assert!(bytes.is_empty());
    let restored = deserialize_embedding(&bytes).unwrap();
    assert!(restored.is_empty());
}

#[test]
fn test_embedding_various_dimensions() {
    for dim in [1, 10, 128, 512, 768, 1024, 1536] {
        let embedding = create_normalized_embedding(dim);
        let bytes = serialize_embedding(&embedding);
        assert_eq!(bytes.len(), dim * 4);
        let restored = deserialize_embedding(&bytes).unwrap();
        assert_eq!(restored.len(), dim);
        assert_eq!(embedding, restored);
    }
}

#[test]
fn edge_case_extreme_floats() {
    let extremes = vec![
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        1e-38_f32,
        1e38_f32,
        0.0_f32,
        -0.0_f32,
    ];

    println!("=== EDGE CASE 3: Extreme Float Values ===");
    println!("BEFORE: {:?}", extremes);

    let bytes = serialize_embedding(&extremes);
    println!("SERIALIZED: {} bytes", bytes.len());

    let restored = deserialize_embedding(&bytes).unwrap();
    println!("AFTER: {:?}", restored);

    for (i, (orig, rest)) in extremes.iter().zip(restored.iter()).enumerate() {
        assert_eq!(orig.to_bits(), rest.to_bits(), "Value {} differs", i);
    }
    println!("RESULT: PASS - All extreme float values preserved exactly");
}

#[test]
fn test_embedding_invalid_sizes() {
    for invalid_len in [1, 2, 3, 5, 7, 9, 11, 13, 15, 17] {
        let bytes = vec![0u8; invalid_len];
        let result = deserialize_embedding(&bytes);
        assert!(
            matches!(result, Err(SerializationError::InvalidEmbeddingSize { .. })),
            "Should fail for length {}",
            invalid_len
        );
    }
}

#[test]
fn test_embedding_valid_sizes() {
    for valid_len in [0, 4, 8, 12, 16, 20, 100, 400, 6144] {
        let bytes = vec![0u8; valid_len];
        let result = deserialize_embedding(&bytes);
        assert!(result.is_ok(), "Should succeed for length {}", valid_len);
        assert_eq!(result.unwrap().len(), valid_len / 4);
    }
}
