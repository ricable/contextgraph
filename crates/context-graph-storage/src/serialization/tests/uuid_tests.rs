//! Tests for UUID serialization.

use uuid::Uuid;

use crate::serialization::{deserialize_uuid, serialize_uuid};

#[test]
fn test_uuid_roundtrip() {
    let id = Uuid::new_v4();
    let bytes = serialize_uuid(&id);
    assert_eq!(bytes.len(), 16);
    let restored = deserialize_uuid(&bytes);
    assert_eq!(id, restored);
}

#[test]
fn test_uuid_nil() {
    let nil = Uuid::nil();
    let bytes = serialize_uuid(&nil);
    let restored = deserialize_uuid(&bytes);
    assert_eq!(nil, restored);
    assert!(restored.is_nil());
}

#[test]
fn test_uuid_max() {
    let max = Uuid::max();
    let bytes = serialize_uuid(&max);
    let restored = deserialize_uuid(&bytes);
    assert_eq!(max, restored);
}

#[test]
fn test_uuid_multiple_roundtrips() {
    for _ in 0..100 {
        let id = Uuid::new_v4();
        let bytes = serialize_uuid(&id);
        let restored = deserialize_uuid(&bytes);
        assert_eq!(id, restored);
    }
}
