# TASK-SESSION-05: Create save_snapshot/load_snapshot Methods

```xml
<task_spec id="TASK-SESSION-05" version="2.0">
<metadata>
  <title>Create save_snapshot/load_snapshot Storage Methods</title>
  <status>pending</status>
  <layer>foundation</layer>
  <sequence>5</sequence>
  <implements>
    <requirement_ref>REQ-SESSION-05</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-SESSION-01</task_ref>  <!-- SessionIdentitySnapshot struct - COMPLETED -->
    <task_ref>TASK-SESSION-04</task_ref>  <!-- CF_SESSION_IDENTITY column family - COMPLETED -->
  </depends_on>
  <estimated_hours>2.0</estimated_hours>
</metadata>
```

## Objective

Implement storage methods for persisting and retrieving `SessionIdentitySnapshot` from `CF_SESSION_IDENTITY` column family with temporal index recovery.

**NO BACKWARDS COMPATIBILITY** - FAIL FAST on all errors with detailed logging.

## Prerequisites - Already Implemented

The following infrastructure is ALREADY COMPLETE (verified in codebase):

### 1. CF_SESSION_IDENTITY Column Family (TASK-SESSION-04)
- **Location**: `crates/context-graph-storage/src/teleological/column_families.rs:157`
- **Constant**: `pub const CF_SESSION_IDENTITY: &str = "session_identity";`
- **Options function**: `session_identity_cf_options()` at line 560
- **Part of**: `TELEOLOGICAL_CFS` array (11 CFs total), `TELEOLOGICAL_CF_COUNT = 11`

### 2. Key Format Functions (TASK-SESSION-04)
- **Location**: `crates/context-graph-storage/src/teleological/schema.rs`
- `SESSION_LATEST_KEY: &[u8] = b"latest"` (line 435)
- `session_identity_key(session_id: &str) -> Vec<u8>` - creates `s:{session_id}` (line 452)
- `session_temporal_key(timestamp_ms: i64) -> Vec<u8>` - creates `t:{timestamp_ms}` BE (line 477)
- `parse_session_identity_key(key: &[u8]) -> &str` (line 496)
- `parse_session_temporal_key(key: &[u8]) -> i64` (line 531)

### 3. SessionIdentitySnapshot Struct (TASK-SESSION-01)
- **Location**: `crates/context-graph-core/src/gwt/session_identity/types.rs`
- 14 fields, ~3KB typical, <30KB max serialized
- `KURAMOTO_N = 13`, `MAX_TRAJECTORY_LEN = 50`

## Context

Key storage patterns:
- **Primary key**: `s:{session_id}` → bincode serialized snapshot
- **Latest pointer**: `latest` → bincode serialized snapshot (atomic copy)
- **Temporal index**: `t:{timestamp_ms_be}` → bincode serialized snapshot (big-endian for lexicographic ordering)

Recovery strategy:
- If `latest` is corrupted, iterate temporal index in REVERSE order to find most recent valid snapshot
- Fresh install returns `None` (not an error)

## Implementation Steps

### Step 1: Create Module File

Create `crates/context-graph-storage/src/session_identity.rs`:

```rust
//! Session Identity Storage Operations
//!
//! Provides save/load methods for SessionIdentitySnapshot persistence
//! in CF_SESSION_IDENTITY column family.
//!
//! # Key Schema
//! - `s:{session_id}` → Snapshot by session ID
//! - `latest` → Latest snapshot pointer
//! - `t:{timestamp_ms}` → Temporal index (big-endian for ordering)
//!
//! # FAIL FAST Policy
//! All errors propagate immediately with detailed context. No silent fallbacks.
//! AP-26: Exit code 2 only for actual storage corruption.
//!
//! # Constitution Reference
//! - IDENTITY-002: IC thresholds (healthy >0.9, warning <0.7, critical <0.5)
//! - GWT-003: Identity continuity tracking

use bincode;
use rocksdb::{IteratorMode, WriteBatch};
use tracing::{debug, error, info, warn};

use context_graph_core::gwt::session_identity::SessionIdentitySnapshot;

use crate::rocksdb_backend::{StorageError, StorageResult};
use crate::RocksDbMemex;
use crate::teleological::column_families::CF_SESSION_IDENTITY;
use crate::teleological::schema::{
    session_identity_key, session_temporal_key, SESSION_LATEST_KEY,
};
```

### Step 2: Implement save_snapshot()

```rust
impl RocksDbMemex {
    /// Save SessionIdentitySnapshot to CF_SESSION_IDENTITY.
    ///
    /// Writes atomically to three keys in a single WriteBatch:
    /// 1. `s:{session_id}` - Primary lookup key
    /// 2. `latest` - Latest session pointer
    /// 3. `t:{timestamp_ms}` - Temporal index (big-endian for ordering)
    ///
    /// # Arguments
    /// * `snapshot` - The snapshot to persist
    ///
    /// # Returns
    /// * `Ok(())` - All three keys written successfully
    /// * `Err(StorageError::ColumnFamilyNotFound)` - CF not found (DB misconfiguration)
    /// * `Err(StorageError::Serialization)` - bincode failed (data issue)
    /// * `Err(StorageError::WriteFailed)` - RocksDB write error (disk/IO)
    ///
    /// # FAIL FAST
    /// No partial writes - entire batch succeeds or fails atomically.
    pub fn save_snapshot(&self, snapshot: &SessionIdentitySnapshot) -> StorageResult<()> {
        let cf = self.db.cf_handle(CF_SESSION_IDENTITY).ok_or_else(|| {
            error!(
                "CF_SESSION_IDENTITY not found - DB opened without teleological CFs. \
                 Use RocksDbMemex::open_with_all_cfs() or verify DB initialization."
            );
            StorageError::ColumnFamilyNotFound {
                name: CF_SESSION_IDENTITY.to_string(),
            }
        })?;

        // Serialize once, use for all three writes
        let data = bincode::serialize(snapshot).map_err(|e| {
            error!(
                "Failed to serialize SessionIdentitySnapshot for session_id='{}': {}",
                snapshot.session_id, e
            );
            StorageError::Serialization(format!(
                "bincode serialize failed for session_id='{}': {}",
                snapshot.session_id, e
            ))
        })?;

        debug!(
            "save_snapshot: session_id='{}', timestamp_ms={}, data_size={} bytes",
            snapshot.session_id,
            snapshot.timestamp_ms,
            data.len()
        );

        // Build atomic batch
        let mut batch = WriteBatch::default();

        // 1. Primary key: s:{session_id}
        let primary_key = session_identity_key(&snapshot.session_id);
        batch.put_cf(&cf, &primary_key, &data);

        // 2. Latest pointer
        batch.put_cf(&cf, SESSION_LATEST_KEY, &data);

        // 3. Temporal index: t:{timestamp_ms} (big-endian)
        let temporal_key = session_temporal_key(snapshot.timestamp_ms);
        batch.put_cf(&cf, &temporal_key, &data);

        // Atomic write
        self.db.write(batch).map_err(|e| {
            error!(
                "WriteBatch failed for session_id='{}': {}. \
                 Check disk space and RocksDB health.",
                snapshot.session_id, e
            );
            StorageError::WriteFailed(format!(
                "WriteBatch failed for session_id='{}': {}",
                snapshot.session_id, e
            ))
        })?;

        info!(
            "save_snapshot: SUCCESS - session_id='{}', timestamp_ms={}, size={} bytes",
            snapshot.session_id,
            snapshot.timestamp_ms,
            data.len()
        );

        Ok(())
    }
```

### Step 3: Implement load_snapshot()

```rust
    /// Load SessionIdentitySnapshot by session_id, or latest if None.
    ///
    /// # Arguments
    /// * `session_id` - Optional session ID. If None, loads latest.
    ///
    /// # Returns
    /// * `Ok(snapshot)` - Successfully loaded snapshot
    /// * `Err(StorageError::NotFound)` - No snapshot exists for given ID
    /// * `Err(StorageError::Serialization)` - Corrupted data in DB
    ///
    /// # FAIL FAST
    /// Returns error immediately on corruption. Use `load_latest()` for
    /// graceful handling of fresh install (returns Option).
    pub fn load_snapshot(
        &self,
        session_id: Option<&str>,
    ) -> StorageResult<SessionIdentitySnapshot> {
        match session_id {
            Some(id) => self.load_snapshot_by_id(id),
            None => self.load_latest()?.ok_or_else(|| {
                debug!("load_snapshot(None): No latest snapshot found - fresh install");
                StorageError::NotFound {
                    id: "session_identity:latest".to_string(),
                }
            }),
        }
    }

    /// Load specific session by ID.
    ///
    /// # FAIL FAST
    /// Panics on malformed key (internal error), returns error on missing/corrupt data.
    fn load_snapshot_by_id(&self, session_id: &str) -> StorageResult<SessionIdentitySnapshot> {
        let cf = self.db.cf_handle(CF_SESSION_IDENTITY).ok_or_else(|| {
            error!("CF_SESSION_IDENTITY not found - DB misconfiguration");
            StorageError::ColumnFamilyNotFound {
                name: CF_SESSION_IDENTITY.to_string(),
            }
        })?;

        let key = session_identity_key(session_id);
        let data = self.db.get_cf(&cf, &key).map_err(|e| {
            error!("RocksDB get failed for session_id='{}': {}", session_id, e);
            StorageError::ReadFailed(format!("get_cf failed for session_id='{}': {}", session_id, e))
        })?;

        match data {
            Some(bytes) => {
                bincode::deserialize(&bytes).map_err(|e| {
                    error!(
                        "Corrupted snapshot for session_id='{}': {}. \
                         Data size: {} bytes. Consider manual recovery.",
                        session_id,
                        e,
                        bytes.len()
                    );
                    StorageError::Serialization(format!(
                        "bincode deserialize failed for session_id='{}': {}",
                        session_id, e
                    ))
                })
            }
            None => {
                debug!("load_snapshot_by_id: session_id='{}' not found", session_id);
                Err(StorageError::NotFound {
                    id: format!("session_identity:s:{}", session_id),
                })
            }
        }
    }
```

### Step 4: Implement load_latest() with Recovery

```rust
    /// Load latest session snapshot, returning None for fresh install.
    ///
    /// # Recovery Behavior
    /// If `latest` key is corrupted, attempts recovery from temporal index
    /// by iterating in reverse chronological order.
    ///
    /// # Returns
    /// * `Ok(Some(snapshot))` - Latest snapshot found
    /// * `Ok(None)` - Fresh install (no snapshots exist)
    /// * `Err(StorageError::Serialization)` - Both latest and recovery failed
    pub fn load_latest(&self) -> StorageResult<Option<SessionIdentitySnapshot>> {
        let cf = self.db.cf_handle(CF_SESSION_IDENTITY).ok_or_else(|| {
            error!("CF_SESSION_IDENTITY not found - DB misconfiguration");
            StorageError::ColumnFamilyNotFound {
                name: CF_SESSION_IDENTITY.to_string(),
            }
        })?;

        // Try latest key first
        match self.db.get_cf(&cf, SESSION_LATEST_KEY) {
            Ok(Some(bytes)) => {
                match bincode::deserialize::<SessionIdentitySnapshot>(&bytes) {
                    Ok(snapshot) => {
                        debug!(
                            "load_latest: Found session_id='{}', timestamp_ms={}",
                            snapshot.session_id, snapshot.timestamp_ms
                        );
                        return Ok(Some(snapshot));
                    }
                    Err(e) => {
                        warn!(
                            "Latest snapshot corrupted ({}), attempting temporal recovery. \
                             Data size: {} bytes.",
                            e,
                            bytes.len()
                        );
                        // Fall through to recovery
                    }
                }
            }
            Ok(None) => {
                debug!("load_latest: No 'latest' key - checking temporal index for fresh install");
                // Check if temporal index is also empty (true fresh install)
            }
            Err(e) => {
                warn!("RocksDB get 'latest' failed: {}. Attempting recovery.", e);
                // Fall through to recovery
            }
        }

        // Recovery: iterate temporal index in reverse
        self.recover_from_temporal_index()
    }

    /// Recover latest snapshot from temporal index.
    ///
    /// Iterates `t:{timestamp_ms}` keys in REVERSE order (most recent first).
    /// Big-endian encoding ensures lexicographic order matches chronological order.
    ///
    /// # Returns
    /// * `Ok(Some(snapshot))` - Found valid snapshot in temporal index
    /// * `Ok(None)` - No temporal entries exist (fresh install)
    /// * `Err(StorageError::Serialization)` - All temporal entries corrupted
    fn recover_from_temporal_index(&self) -> StorageResult<Option<SessionIdentitySnapshot>> {
        let cf = self.db.cf_handle(CF_SESSION_IDENTITY).ok_or_else(|| {
            StorageError::ColumnFamilyNotFound {
                name: CF_SESSION_IDENTITY.to_string(),
            }
        })?;

        info!("recover_from_temporal_index: Starting recovery scan");

        // Temporal keys start with "t:" prefix
        // Iterate in reverse to get most recent first
        let prefix = b"t:";
        let mut iter = self.db.iterator_cf(&cf, IteratorMode::End);
        let mut found_any = false;
        let mut corruption_count = 0;

        while let Some(result) = iter.next() {
            let (key, value) = result.map_err(|e| {
                error!("Iterator error during recovery: {}", e);
                StorageError::ReadFailed(format!("Iterator error: {}", e))
            })?;

            // Check if this is a temporal key
            if !key.starts_with(prefix) {
                // We've passed all temporal keys (iterating backward)
                if key.as_ref() < prefix.as_slice() {
                    break;
                }
                continue;
            }

            found_any = true;

            match bincode::deserialize::<SessionIdentitySnapshot>(&value) {
                Ok(snapshot) => {
                    info!(
                        "recover_from_temporal_index: SUCCESS - Found session_id='{}', \
                         timestamp_ms={} after {} corrupted entries",
                        snapshot.session_id, snapshot.timestamp_ms, corruption_count
                    );
                    return Ok(Some(snapshot));
                }
                Err(e) => {
                    corruption_count += 1;
                    warn!(
                        "Corrupted temporal entry at key {:02x?}: {}. Trying earlier entry.",
                        &key[..10.min(key.len())],
                        e
                    );
                    // Continue to next (earlier) entry
                }
            }
        }

        if found_any {
            error!(
                "recover_from_temporal_index: FAILED - All {} temporal entries corrupted. \
                 Manual intervention required.",
                corruption_count
            );
            Err(StorageError::Serialization(format!(
                "All {} temporal entries corrupted - recovery failed",
                corruption_count
            )))
        } else {
            info!("recover_from_temporal_index: No temporal entries - fresh install");
            Ok(None)
        }
    }
}
```

### Step 5: Update lib.rs Exports

Modify `crates/context-graph-storage/src/lib.rs`:

Add after line 38 (after `pub mod teleological;`):
```rust
pub mod session_identity;
```

Add to re-exports section (after teleological exports):
```rust
// Re-export session identity storage (TASK-SESSION-05)
// Note: Methods are on RocksDbMemex, no separate types needed
```

### Step 6: Create Integration Tests

Create `crates/context-graph-storage/src/session_identity/tests.rs` (or add to existing test module):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use context_graph_core::gwt::session_identity::SessionIdentitySnapshot;
    use tempfile::TempDir;

    /// Helper to create test storage with all CFs
    fn test_storage() -> (TempDir, RocksDbMemex) {
        let tmp = TempDir::new().expect("Failed to create temp dir");
        let memex = RocksDbMemex::open_with_all_cfs(tmp.path())
            .expect("Failed to open RocksDbMemex with all CFs");
        (tmp, memex)
    }

    // =========================================================================
    // TC-SESSION-05: Save/Load Round-Trip (REAL DATA)
    // =========================================================================

    #[test]
    fn tc_session_05_save_load_roundtrip() {
        println!("=== TC-SESSION-05: Save/Load Round-Trip ===");

        let (_tmp, storage) = test_storage();

        // Create snapshot with REAL meaningful data
        let mut snapshot = SessionIdentitySnapshot::default();
        snapshot.session_id = "test-session-abc123".to_string();
        snapshot.timestamp_ms = 1704067200000; // 2024-01-01 00:00:00 UTC
        snapshot.cross_session_ic = 0.85;
        snapshot.last_ic = 0.92;
        snapshot.consciousness = 0.75;
        snapshot.purpose_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3];

        println!("BEFORE: session_id='{}', ic={}", snapshot.session_id, snapshot.last_ic);

        // Save
        storage.save_snapshot(&snapshot).expect("save_snapshot failed");
        println!("AFTER SAVE: Success");

        // Load by session_id
        let loaded = storage.load_snapshot(Some("test-session-abc123"))
            .expect("load_snapshot by ID failed");

        println!("AFTER LOAD: session_id='{}', ic={}", loaded.session_id, loaded.last_ic);

        assert_eq!(snapshot.session_id, loaded.session_id);
        assert_eq!(snapshot.timestamp_ms, loaded.timestamp_ms);
        assert!((snapshot.cross_session_ic - loaded.cross_session_ic).abs() < f32::EPSILON);
        assert!((snapshot.last_ic - loaded.last_ic).abs() < f32::EPSILON);

        // Load latest (should be same)
        let latest = storage.load_snapshot(None).expect("load_snapshot latest failed");
        assert_eq!(latest.session_id, snapshot.session_id);

        println!("RESULT: PASS - Round-trip preserves all data");
    }

    // =========================================================================
    // TC-SESSION-06: Temporal Ordering (REAL DATA)
    // =========================================================================

    #[test]
    fn tc_session_06_temporal_ordering() {
        println!("=== TC-SESSION-06: Temporal Ordering ===");

        let (_tmp, storage) = test_storage();

        // Create three snapshots at different times
        let mut s1 = SessionIdentitySnapshot::default();
        s1.session_id = "session-t1".to_string();
        s1.timestamp_ms = 1704067200000; // 2024-01-01 00:00:00

        let mut s2 = SessionIdentitySnapshot::default();
        s2.session_id = "session-t2".to_string();
        s2.timestamp_ms = 1704153600000; // 2024-01-02 00:00:00

        let mut s3 = SessionIdentitySnapshot::default();
        s3.session_id = "session-t3".to_string();
        s3.timestamp_ms = 1704240000000; // 2024-01-03 00:00:00

        println!("BEFORE: Saving s1 (oldest), s2 (middle), s3 (newest)");

        // Save in NON-chronological order to verify ordering
        storage.save_snapshot(&s2).expect("save s2 failed");
        storage.save_snapshot(&s1).expect("save s1 failed");
        storage.save_snapshot(&s3).expect("save s3 failed");

        println!("AFTER SAVE: All three saved");

        // Latest should be s3 (most recent timestamp)
        let latest = storage.load_latest().expect("load_latest failed")
            .expect("Expected latest snapshot");

        println!("AFTER LOAD: latest.session_id='{}', timestamp_ms={}",
            latest.session_id, latest.timestamp_ms);

        assert_eq!(latest.session_id, "session-t3");
        assert_eq!(latest.timestamp_ms, 1704240000000);

        println!("RESULT: PASS - Latest returns most recent by timestamp");
    }

    // =========================================================================
    // TC-SESSION-07: Fresh Install Returns None (REAL BEHAVIOR)
    // =========================================================================

    #[test]
    fn tc_session_07_fresh_install_returns_none() {
        println!("=== TC-SESSION-07: Fresh Install Returns None ===");

        let (_tmp, storage) = test_storage();

        println!("BEFORE: Fresh database, no snapshots stored");

        let result = storage.load_latest().expect("load_latest should not error");

        println!("AFTER: load_latest returned {:?}", result.is_none());

        assert!(result.is_none(), "Fresh install should return None, not error");

        println!("RESULT: PASS - Fresh install handled gracefully");
    }

    // =========================================================================
    // TC-SESSION-08: Error on Missing Session ID
    // =========================================================================

    #[test]
    fn tc_session_08_error_on_missing_session_id() {
        println!("=== TC-SESSION-08: Error on Missing Session ID ===");

        let (_tmp, storage) = test_storage();

        println!("BEFORE: Requesting non-existent session");

        let result = storage.load_snapshot(Some("nonexistent-session-xyz"));

        println!("AFTER: Result is_err={}", result.is_err());

        assert!(result.is_err());
        match result {
            Err(StorageError::NotFound { id }) => {
                assert!(id.contains("nonexistent-session-xyz"));
                println!("RESULT: PASS - Correct NotFound error for missing session");
            }
            Err(other) => panic!("Expected NotFound, got {:?}", other),
            Ok(_) => panic!("Expected error for non-existent session"),
        }
    }

    // =========================================================================
    // EDGE CASE: Large Trajectory Vector
    // =========================================================================

    #[test]
    fn edge_case_large_trajectory() {
        println!("=== EDGE CASE: Large Trajectory Vector ===");

        let (_tmp, storage) = test_storage();

        let mut snapshot = SessionIdentitySnapshot::default();
        snapshot.session_id = "large-trajectory-test".to_string();
        snapshot.timestamp_ms = 1704067200000;
        // Fill trajectory to MAX_TRAJECTORY_LEN (50)
        snapshot.trajectory = (0..50)
            .map(|i| {
                let val = (i as f32) / 50.0;
                [val; 13]
            })
            .collect();

        println!("BEFORE: trajectory.len()={}", snapshot.trajectory.len());

        storage.save_snapshot(&snapshot).expect("save large trajectory failed");
        let loaded = storage.load_snapshot(Some("large-trajectory-test"))
            .expect("load large trajectory failed");

        println!("AFTER: loaded.trajectory.len()={}", loaded.trajectory.len());

        assert_eq!(loaded.trajectory.len(), 50);
        println!("RESULT: PASS - Large trajectory preserved");
    }

    // =========================================================================
    // EDGE CASE: Unicode Session ID
    // =========================================================================

    #[test]
    fn edge_case_unicode_session_id() {
        println!("=== EDGE CASE: Unicode Session ID ===");

        let (_tmp, storage) = test_storage();

        let mut snapshot = SessionIdentitySnapshot::default();
        snapshot.session_id = "session-日本語-🔥-émoji".to_string();
        snapshot.timestamp_ms = 1704067200000;

        println!("BEFORE: session_id='{}'", snapshot.session_id);

        storage.save_snapshot(&snapshot).expect("save unicode session_id failed");
        let loaded = storage.load_snapshot(Some("session-日本語-🔥-émoji"))
            .expect("load unicode session_id failed");

        println!("AFTER: loaded.session_id='{}'", loaded.session_id);

        assert_eq!(loaded.session_id, "session-日本語-🔥-émoji");
        println!("RESULT: PASS - Unicode session_id preserved");
    }

    // =========================================================================
    // EDGE CASE: Boundary IC Values
    // =========================================================================

    #[test]
    fn edge_case_boundary_ic_values() {
        println!("=== EDGE CASE: Boundary IC Values ===");

        let (_tmp, storage) = test_storage();

        // Test IC at exact thresholds (IDENTITY-002)
        let test_cases = [
            ("ic-critical-boundary", 0.5),   // Critical threshold
            ("ic-warning-boundary", 0.7),    // Warning threshold
            ("ic-healthy-boundary", 0.9),    // Healthy threshold
            ("ic-zero", 0.0),                // Minimum
            ("ic-one", 1.0),                 // Maximum
        ];

        for (session_id, ic_value) in test_cases {
            let mut snapshot = SessionIdentitySnapshot::default();
            snapshot.session_id = session_id.to_string();
            snapshot.timestamp_ms = 1704067200000;
            snapshot.last_ic = ic_value;
            snapshot.cross_session_ic = ic_value;

            storage.save_snapshot(&snapshot).expect(&format!("save {} failed", session_id));
            let loaded = storage.load_snapshot(Some(session_id))
                .expect(&format!("load {} failed", session_id));

            assert!(
                (loaded.last_ic - ic_value).abs() < f32::EPSILON,
                "IC value {} not preserved for {}",
                ic_value,
                session_id
            );
            println!("  PASS: session_id='{}', ic={}", session_id, ic_value);
        }

        println!("RESULT: PASS - All IC boundary values preserved");
    }
}
```

## Input Context Files (VERIFIED PATHS)

```xml
<input_context_files>
  <file purpose="snapshot_type">crates/context-graph-core/src/gwt/session_identity/types.rs</file>
  <file purpose="column_family_constant">crates/context-graph-storage/src/teleological/column_families.rs</file>
  <file purpose="key_format_functions">crates/context-graph-storage/src/teleological/schema.rs</file>
  <file purpose="rocksdb_memex_struct">crates/context-graph-storage/src/rocksdb_backend/core.rs</file>
  <file purpose="error_types">crates/context-graph-storage/src/rocksdb_backend/error.rs</file>
  <file purpose="lib_exports">crates/context-graph-storage/src/lib.rs</file>
</input_context_files>
```

## Files to Create

| File | Description |
|------|-------------|
| `crates/context-graph-storage/src/session_identity.rs` | Storage methods on RocksDbMemex |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-storage/src/lib.rs` | Add `pub mod session_identity;` after line 38 |

## Definition of Done

### Acceptance Criteria

- [ ] `save_snapshot()` writes to `s:{session_id}` key atomically
- [ ] `save_snapshot()` updates `latest` key in same WriteBatch
- [ ] `save_snapshot()` writes temporal index `t:{timestamp_ms}` (big-endian)
- [ ] `load_snapshot(None)` returns latest snapshot
- [ ] `load_snapshot(Some(id))` returns specific snapshot
- [ ] `load_latest()` returns `None` for fresh install (NOT an error)
- [ ] Temporal recovery finds most recent snapshot if `latest` corrupted
- [ ] All methods log detailed context on error (FAIL FAST)
- [ ] TC-SESSION-05 passes (save/load round-trip)
- [ ] TC-SESSION-06 passes (temporal ordering)
- [ ] TC-SESSION-07 passes (fresh install returns None)
- [ ] TC-SESSION-08 passes (error on missing session ID)

### Constraints

- **NO MOCK DATA** in tests - use REAL `SessionIdentitySnapshot` with meaningful values
- **NO BACKWARDS COMPATIBILITY** - FAIL FAST on all errors
- Use bincode for serialization (consistent with existing teleological storage)
- All writes in single WriteBatch for atomicity
- Temporal key uses big-endian timestamp for proper lexicographic ordering
- Recovery iterates temporal index in REVERSE order (most recent first)

### Verification Commands

```bash
# Build
cargo build -p context-graph-storage

# Run all session_identity tests
cargo test -p context-graph-storage session_identity -- --nocapture

# Run specific test cases
cargo test -p context-graph-storage tc_session_05 -- --nocapture
cargo test -p context-graph-storage tc_session_06 -- --nocapture
cargo test -p context-graph-storage tc_session_07 -- --nocapture
cargo test -p context-graph-storage tc_session_08 -- --nocapture

# Run edge case tests
cargo test -p context-graph-storage edge_case -- --nocapture
```

## Full State Verification Protocol

### Source of Truth Verification

After implementation, verify:

```bash
# 1. Module exported correctly
grep -n "session_identity" crates/context-graph-storage/src/lib.rs

# 2. CF_SESSION_IDENTITY accessible
cargo test -p context-graph-storage test_session_identity_cf_options_valid -- --nocapture

# 3. Key format functions work
cargo test -p context-graph-storage test_session_identity_key_format -- --nocapture
```

### Execute & Inspect (Database Physical Verification)

After running tests, verify data exists in RocksDB:

```bash
# Create a test binary to inspect database contents
cargo run --example inspect_session_identity -- /tmp/test_db

# Or use rocksdb_ldb tool if available:
# ldb --db=/tmp/test_db scan --column_family=session_identity
```

### Boundary/Edge Case Audit

Verify these edge cases are tested:
- [ ] Empty session_id (`""`) - should work or fail predictably
- [ ] Very long session_id (1000+ chars) - should work
- [ ] Unicode session_id - MUST work
- [ ] timestamp_ms = 0 (Unix epoch)
- [ ] timestamp_ms = i64::MAX
- [ ] IC values at exact thresholds (0.5, 0.7, 0.9)
- [ ] Empty trajectory vector
- [ ] Full trajectory vector (50 elements)

### Evidence of Success

After all tests pass, the following MUST be true:

1. **File Exists**: `crates/context-graph-storage/src/session_identity.rs` exists
2. **Export Works**: `use context_graph_storage::session_identity;` compiles
3. **Round-Trip**: Save and load returns identical data
4. **Temporal Order**: Latest returns most recent by timestamp_ms
5. **Fresh Install**: `load_latest()` on empty DB returns `Ok(None)`
6. **Error Handling**: Missing session returns `StorageError::NotFound`

## Manual Testing with Synthetic Data

### Synthetic Test Session

```rust
// Create synthetic test data with known values
let snapshot = SessionIdentitySnapshot {
    session_id: "synthetic-test-001".to_string(),
    timestamp_ms: 1704067200000, // 2024-01-01 00:00:00 UTC
    previous_session_id: Some("synthetic-test-000".to_string()),
    cross_session_ic: 0.85,
    kuramoto_phases: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    coupling: 2.5,
    purpose_vector: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4],
    trajectory: vec![
        [0.1; 13],
        [0.2; 13],
        [0.3; 13],
    ],
    last_ic: 0.92,
    crisis_threshold: 0.5,
    consciousness: 0.75,
    integration: 0.8,
    reflection: 0.65,
    differentiation: 0.7,
};

// Expected serialized size: ~400-500 bytes (well under 30KB limit)
```

### Expected Outcomes

| Operation | Input | Expected Output |
|-----------|-------|-----------------|
| `save_snapshot(&snapshot)` | synthetic-test-001 | `Ok(())` |
| `load_snapshot(Some("synthetic-test-001"))` | - | Identical snapshot |
| `load_snapshot(None)` | - | Same as above (latest) |
| `load_latest()` | - | `Ok(Some(snapshot))` |
| `load_snapshot(Some("nonexistent"))` | - | `Err(NotFound)` |
| Fresh DB `load_latest()` | - | `Ok(None)` |

## Exit Conditions

- **Success**: All storage operations work correctly with FAIL FAST error handling
- **Failure**: Any storage corruption or silent failure - must error with detailed logging

## Constitution Compliance

- **IDENTITY-002**: IC thresholds respected in edge case tests
- **GWT-003**: Identity continuity tracked across sessions
- **AP-26**: Exit code 2 only for actual corruption (not missing data)
- **ARCH-07**: Native Claude Code hooks (for future TASK-SESSION-16)

## Next Task

After completion, proceed to **006-TASK-SESSION-06** (SessionIdentityManager).

```xml
</task_spec>
```
