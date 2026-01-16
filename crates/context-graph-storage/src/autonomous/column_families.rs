//! RocksDB column family definitions for autonomous topic-based storage.
//!
//! TASK-P0-004: Reduced from 7 to 5 CFs after North Star removal (TASK-P0-001).
//! Removed CFs:
//! - drift_history: Old drift detection replaced by topic_stability.churn_rate (ARCH-10)
//! - goal_activity_metrics: Manual goals forbidden by ARCH-03 (topics emerge from clustering)
//!
//! # Column Families (5 total)
//! | Name | Purpose | Key Format | Value |
//! |------|---------|------------|-------|
//! | autonomous_config | Singleton AutonomousConfig | "config" (6 bytes) | AutonomousConfig |
//! | adaptive_threshold_state | Singleton threshold state | "state" (5 bytes) | AdaptiveThresholdState |
//! | autonomous_lineage | Lineage events | timestamp_ms:uuid (24 bytes) | LineageEvent |
//! | consolidation_history | Consolidation records | timestamp_ms:uuid (24 bytes) | ConsolidationRecord |
//! | memory_curation | Memory curation state | uuid (16 bytes) | MemoryCurationState |
//!
//! # FAIL FAST Policy
//!
//! All option builders are infallible at construction time. Errors only
//! occur at DB open time, and those are surfaced by RocksDB itself.

use rocksdb::{BlockBasedOptions, Cache, ColumnFamilyDescriptor, Options, SliceTransform};

// =============================================================================
// COLUMN FAMILY NAME CONSTANTS
// =============================================================================

/// Singleton storage for AutonomousConfig.
/// Key: "config" (fixed 6-byte string)
/// Value: AutonomousConfig serialized via bincode
pub const CF_AUTONOMOUS_CONFIG: &str = "autonomous_config";

/// Singleton storage for AdaptiveThresholdState.
/// Key: "state" (fixed 5-byte string)
/// Value: AdaptiveThresholdState serialized via bincode
pub const CF_ADAPTIVE_THRESHOLD_STATE: &str = "adaptive_threshold_state";

// TASK-P0-004: CF_DRIFT_HISTORY removed - old drift detection replaced by topic_stability.churn_rate
// TASK-P0-004: CF_GOAL_ACTIVITY_METRICS removed - manual goals forbidden by ARCH-03

/// Lineage event storage for traceability.
/// Key: timestamp_ms (8 bytes) + event_uuid (16 bytes) = 24 bytes
/// Value: LineageEvent serialized via bincode
pub const CF_AUTONOMOUS_LINEAGE: &str = "autonomous_lineage";

/// Consolidation history records.
/// Key: timestamp_ms (8 bytes) + record_uuid (16 bytes) = 24 bytes
/// Value: ConsolidationRecord serialized via bincode
pub const CF_CONSOLIDATION_HISTORY: &str = "consolidation_history";

/// Memory curation state storage.
/// Key: MemoryId uuid (16 bytes)
/// Value: MemoryCurationState serialized via bincode
pub const CF_MEMORY_CURATION: &str = "memory_curation";

/// All autonomous column family names (5 total after TASK-P0-004).
/// TASK-P0-004: Removed CF_DRIFT_HISTORY and CF_GOAL_ACTIVITY_METRICS.
pub const AUTONOMOUS_CFS: &[&str] = &[
    CF_AUTONOMOUS_CONFIG,
    CF_ADAPTIVE_THRESHOLD_STATE,
    CF_AUTONOMOUS_LINEAGE,
    CF_CONSOLIDATION_HISTORY,
    CF_MEMORY_CURATION,
];

/// Total count of autonomous CFs (5 after TASK-P0-004).
/// TASK-P0-004: Reduced from 7 to 5 (removed drift_history, goal_activity_metrics).
pub const AUTONOMOUS_CF_COUNT: usize = 5;

// =============================================================================
// CF OPTION BUILDERS
// =============================================================================

/// Options for singleton config storage (small, infrequent access).
///
/// # Configuration
/// - No compression (data is small, compression overhead not worth it)
/// - Bloom filter for fast lookups
/// - Optimized for point lookups
///
/// # FAIL FAST Policy
/// No fallback options - let RocksDB error on open if misconfigured.
pub fn autonomous_config_cf_options(cache: &Cache) -> Options {
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_cache_index_and_filter_blocks(true);

    let mut opts = Options::default();
    opts.set_block_based_table_factory(&block_opts);
    opts.set_compression_type(rocksdb::DBCompressionType::None); // Small singleton
    opts.optimize_for_point_lookup(16); // 16MB hint for point lookups
    opts.create_if_missing(true);
    opts
}

/// Options for adaptive threshold state (singleton, moderate size).
///
/// # Configuration
/// - No compression (moderate size, fast access preferred)
/// - Bloom filter for fast lookups
/// - Optimized for point lookups
pub fn adaptive_threshold_state_cf_options(cache: &Cache) -> Options {
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_cache_index_and_filter_blocks(true);

    let mut opts = Options::default();
    opts.set_block_based_table_factory(&block_opts);
    opts.set_compression_type(rocksdb::DBCompressionType::None);
    opts.optimize_for_point_lookup(16);
    opts.create_if_missing(true);
    opts
}

// TASK-P0-004: drift_history_cf_options removed - old drift detection replaced by topic_stability.churn_rate
// TASK-P0-004: goal_activity_metrics_cf_options removed - manual goals forbidden by ARCH-03

/// Options for autonomous lineage (time-series events).
///
/// # Configuration
/// - LZ4 compression (variable size events)
/// - 8-byte prefix extractor for timestamp prefix scans
/// - Cache index and filter blocks
pub fn autonomous_lineage_cf_options(cache: &Cache) -> Options {
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_cache_index_and_filter_blocks(true);

    let mut opts = Options::default();
    opts.set_block_based_table_factory(&block_opts);
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8)); // timestamp_ms prefix
    opts.create_if_missing(true);
    opts
}

/// Options for consolidation history (time-series records).
///
/// # Configuration
/// - LZ4 compression (records can be larger)
/// - 8-byte prefix extractor for timestamp prefix scans
/// - Cache index and filter blocks
pub fn consolidation_history_cf_options(cache: &Cache) -> Options {
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_cache_index_and_filter_blocks(true);

    let mut opts = Options::default();
    opts.set_block_based_table_factory(&block_opts);
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8)); // timestamp_ms prefix
    opts.create_if_missing(true);
    opts
}

/// Options for memory curation state (per-memory, UUID keys).
///
/// # Configuration
/// - No compression (small enum values)
/// - 16-byte prefix extractor for UUID keys
/// - Bloom filter for point lookups
pub fn memory_curation_cf_options(cache: &Cache) -> Options {
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_cache_index_and_filter_blocks(true);

    let mut opts = Options::default();
    opts.set_block_based_table_factory(&block_opts);
    opts.set_compression_type(rocksdb::DBCompressionType::None); // Small state enums
    opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(16)); // UUID prefix
    opts.optimize_for_point_lookup(64); // 64MB hint
    opts.create_if_missing(true);
    opts
}

// =============================================================================
// DESCRIPTOR GETTERS
// =============================================================================

/// Get all 5 autonomous column family descriptors (TASK-P0-004: reduced from 7).
///
/// # Arguments
/// * `cache` - Shared block cache (recommended: 256MB via `Cache::new_lru_cache`)
///
/// # Returns
/// Vector of 5 `ColumnFamilyDescriptor`s for autonomous storage.
/// TASK-P0-004: Removed drift_history and goal_activity_metrics CFs.
///
/// # Example
/// ```ignore
/// use rocksdb::Cache;
/// use context_graph_storage::autonomous::get_autonomous_cf_descriptors;
///
/// let cache = Cache::new_lru_cache(256 * 1024 * 1024); // 256MB
/// let descriptors = get_autonomous_cf_descriptors(&cache);
/// assert_eq!(descriptors.len(), 5);
/// ```
pub fn get_autonomous_cf_descriptors(cache: &Cache) -> Vec<ColumnFamilyDescriptor> {
    vec![
        ColumnFamilyDescriptor::new(CF_AUTONOMOUS_CONFIG, autonomous_config_cf_options(cache)),
        ColumnFamilyDescriptor::new(
            CF_ADAPTIVE_THRESHOLD_STATE,
            adaptive_threshold_state_cf_options(cache),
        ),
        // TASK-P0-004: CF_DRIFT_HISTORY and CF_GOAL_ACTIVITY_METRICS removed
        ColumnFamilyDescriptor::new(CF_AUTONOMOUS_LINEAGE, autonomous_lineage_cf_options(cache)),
        ColumnFamilyDescriptor::new(
            CF_CONSOLIDATION_HISTORY,
            consolidation_history_cf_options(cache),
        ),
        ColumnFamilyDescriptor::new(CF_MEMORY_CURATION, memory_curation_cf_options(cache)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // CF Names Tests
    // =========================================================================

    #[test]
    fn test_cf_count() {
        assert_eq!(
            AUTONOMOUS_CFS.len(),
            AUTONOMOUS_CF_COUNT,
            "AUTONOMOUS_CFS length must match AUTONOMOUS_CF_COUNT"
        );
        // TASK-P0-004: Reduced from 7 to 5 CFs
        assert_eq!(AUTONOMOUS_CF_COUNT, 5, "Must have exactly 5 CFs after TASK-P0-004");
    }

    #[test]
    fn test_cf_names_unique() {
        use std::collections::HashSet;
        let set: HashSet<_> = AUTONOMOUS_CFS.iter().collect();
        assert_eq!(
            set.len(),
            AUTONOMOUS_CF_COUNT,
            "All CF names must be unique"
        );
    }

    #[test]
    fn test_cf_names_snake_case() {
        for name in AUTONOMOUS_CFS {
            assert!(
                name.chars().all(|c| c.is_lowercase() || c == '_'),
                "CF name '{}' should be snake_case",
                name
            );
        }
    }

    #[test]
    fn test_cf_names_non_empty() {
        for name in AUTONOMOUS_CFS {
            assert!(!name.is_empty(), "CF name should not be empty");
        }
    }

    #[test]
    fn test_cf_names_correct_values() {
        assert_eq!(CF_AUTONOMOUS_CONFIG, "autonomous_config");
        assert_eq!(CF_ADAPTIVE_THRESHOLD_STATE, "adaptive_threshold_state");
        // TASK-P0-004: CF_DRIFT_HISTORY and CF_GOAL_ACTIVITY_METRICS removed
        assert_eq!(CF_AUTONOMOUS_LINEAGE, "autonomous_lineage");
        assert_eq!(CF_CONSOLIDATION_HISTORY, "consolidation_history");
        assert_eq!(CF_MEMORY_CURATION, "memory_curation");
    }

    #[test]
    fn test_all_cfs_in_array() {
        assert!(AUTONOMOUS_CFS.contains(&CF_AUTONOMOUS_CONFIG));
        assert!(AUTONOMOUS_CFS.contains(&CF_ADAPTIVE_THRESHOLD_STATE));
        // TASK-P0-004: CF_DRIFT_HISTORY and CF_GOAL_ACTIVITY_METRICS removed
        assert!(AUTONOMOUS_CFS.contains(&CF_AUTONOMOUS_LINEAGE));
        assert!(AUTONOMOUS_CFS.contains(&CF_CONSOLIDATION_HISTORY));
        assert!(AUTONOMOUS_CFS.contains(&CF_MEMORY_CURATION));
    }

    // =========================================================================
    // Option Builders Tests
    // =========================================================================

    #[test]
    fn test_autonomous_config_cf_options_creates_valid_options() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let opts = autonomous_config_cf_options(&cache);
        drop(opts); // No panic = success
    }

    #[test]
    fn test_adaptive_threshold_state_cf_options_creates_valid_options() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let opts = adaptive_threshold_state_cf_options(&cache);
        drop(opts);
    }

    // TASK-P0-004: test_drift_history_cf_options_creates_valid_options removed
    // TASK-P0-004: test_goal_activity_metrics_cf_options_creates_valid_options removed

    #[test]
    fn test_autonomous_lineage_cf_options_creates_valid_options() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let opts = autonomous_lineage_cf_options(&cache);
        drop(opts);
    }

    #[test]
    fn test_consolidation_history_cf_options_creates_valid_options() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let opts = consolidation_history_cf_options(&cache);
        drop(opts);
    }

    #[test]
    fn test_memory_curation_cf_options_creates_valid_options() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let opts = memory_curation_cf_options(&cache);
        drop(opts);
    }

    // =========================================================================
    // Descriptor Tests
    // =========================================================================

    #[test]
    fn test_get_descriptors_returns_5() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let descriptors = get_autonomous_cf_descriptors(&cache);
        assert_eq!(
            descriptors.len(),
            AUTONOMOUS_CF_COUNT,
            "Must return exactly 5 descriptors after TASK-P0-004"
        );
    }

    #[test]
    fn test_descriptors_have_correct_names() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let descriptors = get_autonomous_cf_descriptors(&cache);
        let names: Vec<_> = descriptors.iter().map(|d| d.name()).collect();

        for cf_name in AUTONOMOUS_CFS {
            assert!(names.contains(cf_name), "Missing CF: {}", cf_name);
        }
    }

    #[test]
    fn test_descriptors_in_order() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let descriptors = get_autonomous_cf_descriptors(&cache);

        for (i, cf_name) in AUTONOMOUS_CFS.iter().enumerate() {
            assert_eq!(
                descriptors[i].name(),
                *cf_name,
                "Descriptor {} should be '{}'",
                i,
                cf_name
            );
        }
    }

    #[test]
    fn test_descriptors_unique_names() {
        use std::collections::HashSet;
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let descriptors = get_autonomous_cf_descriptors(&cache);
        let names: HashSet<_> = descriptors.iter().map(|d| d.name()).collect();
        assert_eq!(
            names.len(),
            AUTONOMOUS_CF_COUNT,
            "All descriptor names must be unique"
        );
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn edge_case_multiple_cache_references() {
        println!("=== EDGE CASE: Multiple option builders sharing same cache ===");
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);

        println!("BEFORE: Creating options with shared cache reference");
        let opts1 = autonomous_config_cf_options(&cache);
        let opts2 = adaptive_threshold_state_cf_options(&cache);
        // TASK-P0-004: drift_history_cf_options and goal_activity_metrics_cf_options removed
        let opts3 = autonomous_lineage_cf_options(&cache);
        let opts4 = consolidation_history_cf_options(&cache);
        let opts5 = memory_curation_cf_options(&cache);

        println!("AFTER: All 5 option builders created successfully (TASK-P0-004)");
        drop(opts1);
        drop(opts2);
        drop(opts3);
        drop(opts4);
        drop(opts5);
        println!("RESULT: PASS - Shared cache works across all Options");
    }

    #[test]
    fn edge_case_minimum_cache_size() {
        println!("=== EDGE CASE: Minimum cache size (1MB) ===");
        let cache = Cache::new_lru_cache(1024 * 1024); // 1MB minimum

        println!("BEFORE: Creating descriptors with 1MB cache");
        let descriptors = get_autonomous_cf_descriptors(&cache);

        println!("AFTER: {} descriptors created", descriptors.len());
        assert_eq!(descriptors.len(), AUTONOMOUS_CF_COUNT);
        println!("RESULT: PASS - Works with minimum cache size");
    }

    #[test]
    fn edge_case_zero_cache_size() {
        println!("=== EDGE CASE: Zero cache size ===");
        let cache = Cache::new_lru_cache(0);

        println!("BEFORE: Creating descriptors with 0-byte cache");
        let descriptors = get_autonomous_cf_descriptors(&cache);

        println!("AFTER: {} descriptors created", descriptors.len());
        assert_eq!(descriptors.len(), AUTONOMOUS_CF_COUNT);
        println!("RESULT: PASS - Zero cache handled gracefully");
    }

    #[test]
    fn edge_case_reusable_with_same_cache() {
        println!("=== EDGE CASE: Options can be created multiple times with same cache ===");
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);

        println!("BEFORE: Creating first batch of descriptors");
        let desc1 = get_autonomous_cf_descriptors(&cache);
        println!("  First batch: {} descriptors", desc1.len());

        println!("AFTER: Creating second batch of descriptors with same cache");
        let desc2 = get_autonomous_cf_descriptors(&cache);
        println!("  Second batch: {} descriptors", desc2.len());

        assert_eq!(desc1.len(), AUTONOMOUS_CF_COUNT);
        assert_eq!(desc2.len(), AUTONOMOUS_CF_COUNT);
        println!("RESULT: PASS - Cache can be reused");
    }
}
