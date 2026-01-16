//! Key format functions for autonomous storage.
//!
//! All keys use fixed-size formats for efficient range scans.
//! No variable-length prefixes.
//!
//! # FAIL FAST Policy
//!
//! Key parsing functions panic on invalid input. This ensures:
//! 1. Data corruption is immediately detected
//! 2. No silent degradation of data integrity
//! 3. Clear error messages with full context

use uuid::Uuid;

// =============================================================================
// SINGLETON KEY CONSTANTS
// =============================================================================

/// Key for autonomous_config CF (singleton).
/// Fixed 6-byte string "config".
pub const AUTONOMOUS_CONFIG_KEY: &[u8] = b"config";

/// Key for adaptive_threshold_state CF (singleton).
/// Fixed 5-byte string "state".
pub const ADAPTIVE_THRESHOLD_STATE_KEY: &[u8] = b"state";

// TASK-P0-004: Removed drift_history key functions - old drift detection replaced by topic_stability.churn_rate (ARCH-10)
// TASK-P0-004: Removed goal_activity_metrics key functions - manual goals forbidden by ARCH-03

// =============================================================================
// AUTONOMOUS LINEAGE KEYS (timestamp_ms:uuid = 24 bytes)
// =============================================================================

/// Create key for autonomous_lineage CF.
///
/// # Arguments
/// * `timestamp_ms` - Unix timestamp in milliseconds
/// * `event_id` - UUID of the lineage event
///
/// # Returns
/// Exactly 24 bytes: timestamp_ms (8 bytes BE) + event_uuid (16 bytes)
#[inline]
pub fn autonomous_lineage_key(timestamp_ms: i64, event_id: &Uuid) -> [u8; 24] {
    let mut key = [0u8; 24];
    key[..8].copy_from_slice(&timestamp_ms.to_be_bytes());
    key[8..].copy_from_slice(event_id.as_bytes());
    key
}

/// Parse autonomous_lineage key back to (timestamp_ms, event_uuid).
///
/// # Arguments
/// * `key` - Exactly 24 bytes
///
/// # Returns
/// Tuple of (timestamp_ms, Uuid)
///
/// # Panics
/// Panics if key is not exactly 24 bytes (FAIL FAST).
#[inline]
pub fn parse_autonomous_lineage_key(key: &[u8]) -> (i64, Uuid) {
    if key.len() != 24 {
        panic!(
            "STORAGE ERROR: autonomous_lineage key must be 24 bytes, got {} bytes. \
             Key data: {:02x?}. This indicates corrupted storage or wrong CF access.",
            key.len(),
            key
        );
    }

    let timestamp_ms = i64::from_be_bytes(key[..8].try_into().unwrap_or_else(|_| {
        panic!(
            "STORAGE ERROR: Invalid timestamp bytes in autonomous_lineage key. \
             Key data: {:02x?}.",
            key
        );
    }));

    let uuid = Uuid::from_slice(&key[8..]).unwrap_or_else(|e| {
        panic!(
            "STORAGE ERROR: Invalid UUID bytes in autonomous_lineage key. \
             Error: {}. Key data: {:02x?}.",
            e, key
        );
    });

    (timestamp_ms, uuid)
}

/// Extract timestamp prefix from autonomous_lineage key for range scans.
///
/// # Arguments
/// * `timestamp_ms` - Unix timestamp in milliseconds
///
/// # Returns
/// Exactly 8 bytes (timestamp_ms in big-endian format)
#[inline]
pub fn autonomous_lineage_timestamp_prefix(timestamp_ms: i64) -> [u8; 8] {
    timestamp_ms.to_be_bytes()
}

// =============================================================================
// CONSOLIDATION HISTORY KEYS (timestamp_ms:uuid = 24 bytes)
// =============================================================================

/// Create key for consolidation_history CF.
///
/// # Arguments
/// * `timestamp_ms` - Unix timestamp in milliseconds
/// * `record_id` - UUID of the consolidation record
///
/// # Returns
/// Exactly 24 bytes: timestamp_ms (8 bytes BE) + record_uuid (16 bytes)
#[inline]
pub fn consolidation_history_key(timestamp_ms: i64, record_id: &Uuid) -> [u8; 24] {
    let mut key = [0u8; 24];
    key[..8].copy_from_slice(&timestamp_ms.to_be_bytes());
    key[8..].copy_from_slice(record_id.as_bytes());
    key
}

/// Parse consolidation_history key back to (timestamp_ms, record_uuid).
///
/// # Arguments
/// * `key` - Exactly 24 bytes
///
/// # Returns
/// Tuple of (timestamp_ms, Uuid)
///
/// # Panics
/// Panics if key is not exactly 24 bytes (FAIL FAST).
#[inline]
pub fn parse_consolidation_history_key(key: &[u8]) -> (i64, Uuid) {
    if key.len() != 24 {
        panic!(
            "STORAGE ERROR: consolidation_history key must be 24 bytes, got {} bytes. \
             Key data: {:02x?}. This indicates corrupted storage or wrong CF access.",
            key.len(),
            key
        );
    }

    let timestamp_ms = i64::from_be_bytes(key[..8].try_into().unwrap_or_else(|_| {
        panic!(
            "STORAGE ERROR: Invalid timestamp bytes in consolidation_history key. \
             Key data: {:02x?}.",
            key
        );
    }));

    let uuid = Uuid::from_slice(&key[8..]).unwrap_or_else(|e| {
        panic!(
            "STORAGE ERROR: Invalid UUID bytes in consolidation_history key. \
             Error: {}. Key data: {:02x?}.",
            e, key
        );
    });

    (timestamp_ms, uuid)
}

/// Extract timestamp prefix from consolidation_history key for range scans.
///
/// # Arguments
/// * `timestamp_ms` - Unix timestamp in milliseconds
///
/// # Returns
/// Exactly 8 bytes (timestamp_ms in big-endian format)
#[inline]
pub fn consolidation_history_timestamp_prefix(timestamp_ms: i64) -> [u8; 8] {
    timestamp_ms.to_be_bytes()
}

// =============================================================================
// MEMORY CURATION KEYS (uuid = 16 bytes)
// =============================================================================

/// Key for memory_curation CF: MemoryId UUID as 16 bytes.
///
/// # Arguments
/// * `memory_id` - The memory's UUID
///
/// # Returns
/// Exactly 16 bytes (UUID in big-endian format)
#[inline]
pub fn memory_curation_key(memory_id: &Uuid) -> [u8; 16] {
    *memory_id.as_bytes()
}

/// Parse memory_curation key back to MemoryId UUID.
///
/// # Arguments
/// * `key` - Exactly 16 bytes
///
/// # Returns
/// The parsed UUID
///
/// # Panics
/// Panics if key is not exactly 16 bytes (FAIL FAST).
#[inline]
pub fn parse_memory_curation_key(key: &[u8]) -> Uuid {
    if key.len() != 16 {
        panic!(
            "STORAGE ERROR: memory_curation key must be 16 bytes, got {} bytes. \
             Key data: {:02x?}. This indicates corrupted storage or wrong CF access.",
            key.len(),
            key
        );
    }
    Uuid::from_slice(key).unwrap_or_else(|e| {
        panic!(
            "STORAGE ERROR: Invalid UUID bytes in memory_curation key. \
             Error: {}. Key data: {:02x?}.",
            e, key
        );
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Singleton Key Tests
    // =========================================================================

    #[test]
    fn test_autonomous_config_key_length() {
        assert_eq!(AUTONOMOUS_CONFIG_KEY.len(), 6);
        assert_eq!(AUTONOMOUS_CONFIG_KEY, b"config");
    }

    #[test]
    fn test_adaptive_threshold_state_key_length() {
        assert_eq!(ADAPTIVE_THRESHOLD_STATE_KEY.len(), 5);
        assert_eq!(ADAPTIVE_THRESHOLD_STATE_KEY, b"state");
    }

    // TASK-P0-004: Removed drift_history and goal_activity_metrics tests
    // - drift_history tests removed - old drift detection replaced by topic_stability.churn_rate (ARCH-10)
    // - goal_activity_metrics tests removed - manual goals forbidden by ARCH-03

    // =========================================================================
    // Autonomous Lineage Key Tests
    // =========================================================================

    #[test]
    fn test_autonomous_lineage_key_format() {
        let timestamp_ms: i64 = 1704067200000;
        let uuid = Uuid::new_v4();
        let key = autonomous_lineage_key(timestamp_ms, &uuid);

        assert_eq!(key.len(), 24);
    }

    #[test]
    fn test_autonomous_lineage_key_roundtrip() {
        let timestamp_ms: i64 = 1704067200000;
        let uuid = Uuid::new_v4();

        let key = autonomous_lineage_key(timestamp_ms, &uuid);
        let (parsed_ts, parsed_uuid) = parse_autonomous_lineage_key(&key);

        assert_eq!(parsed_ts, timestamp_ms);
        assert_eq!(parsed_uuid, uuid);
    }

    #[test]
    fn test_autonomous_lineage_timestamp_prefix() {
        let timestamp_ms: i64 = 1704067200000;
        let prefix = autonomous_lineage_timestamp_prefix(timestamp_ms);

        assert_eq!(prefix.len(), 8);
        assert_eq!(prefix, timestamp_ms.to_be_bytes());
    }

    #[test]
    #[should_panic(expected = "STORAGE ERROR: autonomous_lineage key must be 24 bytes")]
    fn test_parse_autonomous_lineage_key_wrong_size() {
        let bad_key = [0u8; 20];
        parse_autonomous_lineage_key(&bad_key);
    }

    // =========================================================================
    // Consolidation History Key Tests
    // =========================================================================

    #[test]
    fn test_consolidation_history_key_format() {
        let timestamp_ms: i64 = 1704067200000;
        let uuid = Uuid::new_v4();
        let key = consolidation_history_key(timestamp_ms, &uuid);

        assert_eq!(key.len(), 24);
    }

    #[test]
    fn test_consolidation_history_key_roundtrip() {
        let timestamp_ms: i64 = 1704067200000;
        let uuid = Uuid::new_v4();

        let key = consolidation_history_key(timestamp_ms, &uuid);
        let (parsed_ts, parsed_uuid) = parse_consolidation_history_key(&key);

        assert_eq!(parsed_ts, timestamp_ms);
        assert_eq!(parsed_uuid, uuid);
    }

    #[test]
    fn test_consolidation_history_timestamp_prefix() {
        let timestamp_ms: i64 = 1704067200000;
        let prefix = consolidation_history_timestamp_prefix(timestamp_ms);

        assert_eq!(prefix.len(), 8);
        assert_eq!(prefix, timestamp_ms.to_be_bytes());
    }

    #[test]
    #[should_panic(expected = "STORAGE ERROR: consolidation_history key must be 24 bytes")]
    fn test_parse_consolidation_history_key_wrong_size() {
        let bad_key = [0u8; 12];
        parse_consolidation_history_key(&bad_key);
    }

    // =========================================================================
    // Memory Curation Key Tests
    // =========================================================================

    #[test]
    fn test_memory_curation_key_format() {
        let uuid = Uuid::new_v4();
        let key = memory_curation_key(&uuid);

        assert_eq!(key.len(), 16);
        assert_eq!(&key[..], uuid.as_bytes());
    }

    #[test]
    fn test_memory_curation_key_roundtrip() {
        let uuid = Uuid::new_v4();

        let key = memory_curation_key(&uuid);
        let parsed_uuid = parse_memory_curation_key(&key);

        assert_eq!(parsed_uuid, uuid);
    }

    #[test]
    #[should_panic(expected = "STORAGE ERROR: memory_curation key must be 16 bytes")]
    fn test_parse_memory_curation_key_wrong_size() {
        let bad_key = [0u8; 10];
        parse_memory_curation_key(&bad_key);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================
    // TASK-P0-004: Updated tests to use autonomous_lineage and memory_curation
    // instead of removed drift_history and goal_activity_metrics functions.

    #[test]
    fn test_timestamp_edge_cases() {
        let uuid = Uuid::new_v4();

        // Minimum timestamp
        let key = autonomous_lineage_key(i64::MIN, &uuid);
        let (parsed_ts, _) = parse_autonomous_lineage_key(&key);
        assert_eq!(parsed_ts, i64::MIN);

        // Maximum timestamp
        let key = autonomous_lineage_key(i64::MAX, &uuid);
        let (parsed_ts, _) = parse_autonomous_lineage_key(&key);
        assert_eq!(parsed_ts, i64::MAX);

        // Zero timestamp
        let key = autonomous_lineage_key(0, &uuid);
        let (parsed_ts, _) = parse_autonomous_lineage_key(&key);
        assert_eq!(parsed_ts, 0);

        // Negative timestamp
        let key = autonomous_lineage_key(-1000, &uuid);
        let (parsed_ts, _) = parse_autonomous_lineage_key(&key);
        assert_eq!(parsed_ts, -1000);
    }

    #[test]
    fn test_nil_uuid() {
        let nil_uuid = Uuid::nil();

        let key = memory_curation_key(&nil_uuid);
        let parsed = parse_memory_curation_key(&key);
        assert_eq!(parsed, nil_uuid);
        assert!(parsed.is_nil());
    }

    #[test]
    fn test_max_uuid() {
        let max_uuid = Uuid::max();

        let key = memory_curation_key(&max_uuid);
        let parsed = parse_memory_curation_key(&key);
        assert_eq!(parsed, max_uuid);
    }

    #[test]
    fn test_deterministic_key_generation() {
        let timestamp_ms: i64 = 1704067200000;
        let uuid = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();

        let key1 = autonomous_lineage_key(timestamp_ms, &uuid);
        let key2 = autonomous_lineage_key(timestamp_ms, &uuid);

        assert_eq!(key1, key2, "Same inputs must produce identical keys");
    }

    #[test]
    fn test_key_uniqueness_with_same_timestamp() {
        let timestamp_ms: i64 = 1704067200000;
        let uuid1 = Uuid::new_v4();
        let uuid2 = Uuid::new_v4();

        let key1 = autonomous_lineage_key(timestamp_ms, &uuid1);
        let key2 = autonomous_lineage_key(timestamp_ms, &uuid2);

        assert_ne!(key1, key2, "Different UUIDs must produce different keys");
    }

    #[test]
    fn test_key_uniqueness_with_same_uuid() {
        let uuid = Uuid::new_v4();
        let ts1: i64 = 1704067200000;
        let ts2: i64 = 1704067200001;

        let key1 = autonomous_lineage_key(ts1, &uuid);
        let key2 = autonomous_lineage_key(ts2, &uuid);

        assert_ne!(
            key1, key2,
            "Different timestamps must produce different keys"
        );
    }
}
