//! Session identity persistence operations for RocksDB backend.
//!
//! Provides save/load operations for SessionIdentitySnapshot with atomic
//! writes and temporal recovery support.
//!
//! # TASK-SESSION-05: Persistence Methods
//!
//! ## Key Schema (CF_SESSION_IDENTITY)
//! - `s:{session_id}` - Primary session data
//! - `latest` - Pointer to most recent session_id
//! - `t:{timestamp_ms_be}` - Temporal index for recovery
//!
//! ## Performance Targets
//! - Read: <5ms p95
//! - Write: <10ms p95
//!
//! # Constitution Reference
//! - IDENTITY-002: IC thresholds (Healthy>0.9, Warning<0.7, Critical<0.5)
//! - GWT-003: Identity continuity tracking
//! - AP-25: Kuramoto must have exactly 13 oscillators
//! - FAIL FAST: No silent failures, robust error logging

use rocksdb::{Direction, IteratorMode, WriteBatch};
use tracing::{debug, error, info, trace, warn};

use context_graph_core::gwt::SessionIdentitySnapshot;

use crate::teleological::{
    parse_session_temporal_key, session_identity_key, session_temporal_key, CF_SESSION_IDENTITY,
    SESSION_LATEST_KEY,
};

use super::core::RocksDbMemex;
use super::error::{StorageError, StorageResult};

impl RocksDbMemex {
    /// Saves a SessionIdentitySnapshot atomically with triple indexing.
    ///
    /// Performs atomic WriteBatch with 3 puts:
    /// 1. `s:{session_id}` - Main snapshot data (bincode serialized)
    /// 2. `latest` - Points to session_id for fast latest lookup
    /// 3. `t:{timestamp_ms_be}` - Temporal index pointing to session_id
    ///
    /// # Arguments
    /// * `snapshot` - The SessionIdentitySnapshot to persist
    ///
    /// # Returns
    /// * `Ok(())` - Successfully saved
    /// * `Err(StorageError::Serialization)` - Serialization failed
    /// * `Err(StorageError::WriteFailed)` - RocksDB write failed
    /// * `Err(StorageError::ColumnFamilyNotFound)` - CF missing (corruption)
    ///
    /// # Performance
    /// Target: <10ms p95
    ///
    /// # FAIL FAST
    /// Any error aborts the operation immediately with detailed logging.
    /// No partial writes due to atomic WriteBatch.
    ///
    /// # Example
    /// ```ignore
    /// let snapshot = SessionIdentitySnapshot::new("my-session-id");
    /// memex.save_snapshot(&snapshot)?;
    /// ```
    pub fn save_snapshot(&self, snapshot: &SessionIdentitySnapshot) -> StorageResult<()> {
        let start = std::time::Instant::now();

        info!(
            session_id = %snapshot.session_id,
            timestamp_ms = snapshot.timestamp_ms,
            trajectory_len = snapshot.trajectory.len(),
            ic = snapshot.last_ic,
            "SESSION_IDENTITY: save_snapshot starting"
        );

        // 1. Get column family - FAIL FAST if missing
        let cf = self.get_cf(CF_SESSION_IDENTITY).map_err(|e| {
            error!(
                cf_name = CF_SESSION_IDENTITY,
                error = %e,
                "STORAGE ERROR: CF_SESSION_IDENTITY not found. Database may be corrupted."
            );
            e
        })?;

        // 2. Serialize snapshot to bincode - FAIL FAST on error
        let serialized = bincode::serialize(snapshot).map_err(|e| {
            error!(
                session_id = %snapshot.session_id,
                error = %e,
                estimated_size = snapshot.estimated_size(),
                "SERIALIZATION ERROR: Failed to serialize SessionIdentitySnapshot"
            );
            StorageError::Serialization(e.to_string())
        })?;

        debug!(
            session_id = %snapshot.session_id,
            serialized_bytes = serialized.len(),
            estimated_bytes = snapshot.estimated_size(),
            "SESSION_IDENTITY: serialization complete"
        );

        // 3. Create atomic WriteBatch
        let mut batch = WriteBatch::default();

        // 4. Put session data: s:{session_id} -> serialized snapshot
        let session_key = session_identity_key(&snapshot.session_id);
        batch.put_cf(cf, &session_key, &serialized);
        trace!(
            key = ?session_key,
            key_len = session_key.len(),
            value_len = serialized.len(),
            "SESSION_IDENTITY: put session key"
        );

        // 5. Put latest pointer: latest -> session_id (as UTF-8 bytes)
        batch.put_cf(cf, SESSION_LATEST_KEY, snapshot.session_id.as_bytes());
        trace!(
            key = ?SESSION_LATEST_KEY,
            value = %snapshot.session_id,
            "SESSION_IDENTITY: put latest pointer"
        );

        // 6. Put temporal index: t:{timestamp_ms_be} -> session_id
        let temporal_key = session_temporal_key(snapshot.timestamp_ms);
        batch.put_cf(cf, &temporal_key, snapshot.session_id.as_bytes());
        trace!(
            key = ?temporal_key,
            timestamp_ms = snapshot.timestamp_ms,
            value = %snapshot.session_id,
            "SESSION_IDENTITY: put temporal index"
        );

        // 7. Execute atomic write - FAIL FAST on error
        self.db.write(batch).map_err(|e| {
            error!(
                session_id = %snapshot.session_id,
                error = %e,
                "WRITE ERROR: Failed to write SessionIdentitySnapshot batch"
            );
            StorageError::WriteFailed(format!(
                "Failed to write session identity batch for {}: {}",
                snapshot.session_id, e
            ))
        })?;

        let elapsed = start.elapsed();
        info!(
            session_id = %snapshot.session_id,
            elapsed_ms = elapsed.as_millis(),
            serialized_bytes = serialized.len(),
            "SESSION_IDENTITY: save_snapshot complete"
        );

        // Performance warning if >10ms
        if elapsed.as_millis() > 10 {
            warn!(
                session_id = %snapshot.session_id,
                elapsed_ms = elapsed.as_millis(),
                "PERFORMANCE WARNING: save_snapshot exceeded 10ms target"
            );
        }

        Ok(())
    }

    /// Loads a SessionIdentitySnapshot by session ID.
    ///
    /// Returns `None` if no snapshot exists with the given session_id.
    /// Use `load_snapshot_by_id` if you need an error on missing.
    ///
    /// # Arguments
    /// * `session_id` - The session UUID string
    ///
    /// # Returns
    /// * `Ok(Some(snapshot))` - Found and deserialized
    /// * `Ok(None)` - No snapshot with this session_id
    /// * `Err(StorageError::Serialization)` - Deserialization failed (corruption)
    /// * `Err(StorageError::ReadFailed)` - RocksDB read failed
    /// * `Err(StorageError::ColumnFamilyNotFound)` - CF missing (corruption)
    ///
    /// # Performance
    /// Target: <5ms p95
    ///
    /// # FAIL FAST
    /// Deserialization errors indicate corruption and fail immediately.
    /// Only returns `None` for genuinely missing data.
    ///
    /// # Example
    /// ```ignore
    /// if let Some(snapshot) = memex.load_snapshot("my-session-id")? {
    ///     println!("Found session with IC: {}", snapshot.last_ic);
    /// }
    /// ```
    pub fn load_snapshot(&self, session_id: &str) -> StorageResult<Option<SessionIdentitySnapshot>> {
        let start = std::time::Instant::now();

        debug!(
            session_id = %session_id,
            "SESSION_IDENTITY: load_snapshot starting"
        );

        // 1. Get column family - FAIL FAST if missing
        let cf = self.get_cf(CF_SESSION_IDENTITY).map_err(|e| {
            error!(
                cf_name = CF_SESSION_IDENTITY,
                error = %e,
                "STORAGE ERROR: CF_SESSION_IDENTITY not found. Database may be corrupted."
            );
            e
        })?;

        // 2. Build key
        let key = session_identity_key(session_id);

        // 3. Read from RocksDB
        let maybe_bytes = self.db.get_cf(cf, &key).map_err(|e| {
            error!(
                session_id = %session_id,
                key = ?key,
                error = %e,
                "READ ERROR: Failed to read session identity"
            );
            StorageError::ReadFailed(format!(
                "Failed to read session identity {}: {}",
                session_id, e
            ))
        })?;

        // 4. Handle missing (return None, not error)
        let bytes = match maybe_bytes {
            Some(b) => b,
            None => {
                debug!(
                    session_id = %session_id,
                    elapsed_ms = start.elapsed().as_millis(),
                    "SESSION_IDENTITY: load_snapshot - not found"
                );
                return Ok(None);
            }
        };

        // 5. Deserialize - FAIL FAST on corruption
        let snapshot: SessionIdentitySnapshot = bincode::deserialize(&bytes).map_err(|e| {
            error!(
                session_id = %session_id,
                bytes_len = bytes.len(),
                error = %e,
                "DESERIALIZATION ERROR: Corrupted SessionIdentitySnapshot data"
            );
            StorageError::Serialization(format!(
                "Failed to deserialize session {}: {}",
                session_id, e
            ))
        })?;

        let elapsed = start.elapsed();
        debug!(
            session_id = %session_id,
            elapsed_ms = elapsed.as_millis(),
            ic = snapshot.last_ic,
            "SESSION_IDENTITY: load_snapshot complete"
        );

        // Performance warning if >5ms
        if elapsed.as_millis() > 5 {
            warn!(
                session_id = %session_id,
                elapsed_ms = elapsed.as_millis(),
                "PERFORMANCE WARNING: load_snapshot exceeded 5ms target"
            );
        }

        Ok(Some(snapshot))
    }

    /// Loads a SessionIdentitySnapshot by session ID, failing if not found.
    ///
    /// Unlike `load_snapshot`, this returns `NotFound` error if missing.
    /// Use when you expect the session to exist.
    ///
    /// # Arguments
    /// * `session_id` - The session UUID string
    ///
    /// # Returns
    /// * `Ok(snapshot)` - Found and deserialized
    /// * `Err(StorageError::NotFound)` - No snapshot with this session_id
    /// * `Err(StorageError::Serialization)` - Deserialization failed (corruption)
    /// * `Err(StorageError::ReadFailed)` - RocksDB read failed
    /// * `Err(StorageError::ColumnFamilyNotFound)` - CF missing (corruption)
    ///
    /// # Performance
    /// Target: <5ms p95
    ///
    /// # FAIL FAST
    /// Both missing data and corruption fail immediately with detailed errors.
    ///
    /// # Example
    /// ```ignore
    /// let snapshot = memex.load_snapshot_by_id("my-session-id")?;
    /// // If we get here, snapshot exists
    /// ```
    pub fn load_snapshot_by_id(&self, session_id: &str) -> StorageResult<SessionIdentitySnapshot> {
        debug!(
            session_id = %session_id,
            "SESSION_IDENTITY: load_snapshot_by_id starting"
        );

        self.load_snapshot(session_id)?.ok_or_else(|| {
            error!(
                session_id = %session_id,
                "NOT FOUND ERROR: Session identity does not exist"
            );
            StorageError::NotFound {
                id: format!("session_identity:{}", session_id),
            }
        })
    }

    /// Loads the most recent SessionIdentitySnapshot with temporal recovery.
    ///
    /// Lookup strategy:
    /// 1. Read `latest` pointer -> get session_id
    /// 2. Load snapshot by that session_id
    /// 3. If step 1 or 2 fails, fall back to temporal index:
    ///    - Iterate `t:*` keys in reverse (most recent first)
    ///    - Try to load each referenced session until one succeeds
    ///
    /// # Returns
    /// * `Ok(Some(snapshot))` - Found latest snapshot
    /// * `Ok(None)` - No snapshots exist (fresh install)
    /// * `Err(StorageError::Serialization)` - Deserialization failed (corruption)
    /// * `Err(StorageError::ReadFailed)` - RocksDB read failed
    /// * `Err(StorageError::ColumnFamilyNotFound)` - CF missing (corruption)
    ///
    /// # Performance
    /// Target: <5ms p95 (typical), may be slower during recovery
    ///
    /// # FAIL FAST
    /// Returns error only on true corruption. Missing data triggers recovery.
    ///
    /// # Example
    /// ```ignore
    /// if let Some(latest) = memex.load_latest()? {
    ///     println!("Restored session: {}", latest.session_id);
    /// } else {
    ///     println!("Fresh install, no previous session");
    /// }
    /// ```
    pub fn load_latest(&self) -> StorageResult<Option<SessionIdentitySnapshot>> {
        let start = std::time::Instant::now();

        info!("SESSION_IDENTITY: load_latest starting");

        // 1. Get column family - FAIL FAST if missing
        let cf = self.get_cf(CF_SESSION_IDENTITY).map_err(|e| {
            error!(
                cf_name = CF_SESSION_IDENTITY,
                error = %e,
                "STORAGE ERROR: CF_SESSION_IDENTITY not found. Database may be corrupted."
            );
            e
        })?;

        // 2. Try fast path: read "latest" pointer
        let maybe_latest_id = self.db.get_cf(cf, SESSION_LATEST_KEY).map_err(|e| {
            error!(
                key = ?SESSION_LATEST_KEY,
                error = %e,
                "READ ERROR: Failed to read latest pointer"
            );
            StorageError::ReadFailed(format!("Failed to read latest pointer: {}", e))
        })?;

        if let Some(latest_id_bytes) = maybe_latest_id {
            // Parse session_id from bytes
            let latest_id = std::str::from_utf8(&latest_id_bytes).map_err(|e| {
                error!(
                    bytes = ?latest_id_bytes,
                    error = %e,
                    "CORRUPTION ERROR: latest pointer contains invalid UTF-8"
                );
                StorageError::Serialization(format!(
                    "Latest pointer contains invalid UTF-8: {}",
                    e
                ))
            })?;

            debug!(
                latest_id = %latest_id,
                "SESSION_IDENTITY: found latest pointer, loading snapshot"
            );

            // Try to load the snapshot
            match self.load_snapshot(latest_id)? {
                Some(snapshot) => {
                    let elapsed = start.elapsed();
                    info!(
                        session_id = %snapshot.session_id,
                        elapsed_ms = elapsed.as_millis(),
                        "SESSION_IDENTITY: load_latest complete (fast path)"
                    );
                    return Ok(Some(snapshot));
                }
                None => {
                    // Latest pointer exists but snapshot is missing - need recovery
                    warn!(
                        latest_id = %latest_id,
                        "SESSION_IDENTITY: latest pointer points to missing snapshot, attempting temporal recovery"
                    );
                }
            }
        } else {
            debug!("SESSION_IDENTITY: no latest pointer found, attempting temporal recovery");
        }

        // 3. Temporal recovery: iterate t:* keys in reverse order
        info!("SESSION_IDENTITY: starting temporal recovery scan");

        // Create iterator starting from the end of t: prefix (largest timestamp)
        // We use a prefix iterator and reverse direction
        let prefix = b"t:";
        let iter = self.db.iterator_cf(
            cf,
            IteratorMode::From(b"t:\xff\xff\xff\xff\xff\xff\xff\xff", Direction::Reverse),
        );

        let mut recovery_attempts = 0;
        for item in iter {
            let (key, value) = item.map_err(|e| {
                error!(
                    error = %e,
                    "READ ERROR: Failed to iterate temporal index"
                );
                StorageError::ReadFailed(format!("Failed to iterate temporal index: {}", e))
            })?;

            // Stop if we've left the t: prefix
            if !key.starts_with(prefix) {
                break;
            }

            recovery_attempts += 1;
            let timestamp = parse_session_temporal_key(&key);

            // Value is session_id as UTF-8
            let session_id = std::str::from_utf8(&value).map_err(|e| {
                error!(
                    timestamp_ms = timestamp,
                    value = ?value,
                    error = %e,
                    "CORRUPTION ERROR: temporal index value contains invalid UTF-8"
                );
                StorageError::Serialization(format!(
                    "Temporal index value at {} contains invalid UTF-8: {}",
                    timestamp, e
                ))
            })?;

            debug!(
                timestamp_ms = timestamp,
                session_id = %session_id,
                attempt = recovery_attempts,
                "SESSION_IDENTITY: temporal recovery attempt"
            );

            // Try to load this snapshot
            match self.load_snapshot(session_id)? {
                Some(snapshot) => {
                    let elapsed = start.elapsed();
                    info!(
                        session_id = %snapshot.session_id,
                        elapsed_ms = elapsed.as_millis(),
                        recovery_attempts = recovery_attempts,
                        "SESSION_IDENTITY: load_latest complete (temporal recovery)"
                    );
                    return Ok(Some(snapshot));
                }
                None => {
                    warn!(
                        session_id = %session_id,
                        timestamp_ms = timestamp,
                        "SESSION_IDENTITY: temporal index points to missing snapshot, trying next"
                    );
                    continue;
                }
            }
        }

        // 4. No snapshots found at all
        let elapsed = start.elapsed();
        info!(
            elapsed_ms = elapsed.as_millis(),
            recovery_attempts = recovery_attempts,
            "SESSION_IDENTITY: load_latest complete - no snapshots found (fresh install)"
        );

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use context_graph_core::gwt::session_identity::{KURAMOTO_N, MAX_TRAJECTORY_LEN};
    use tempfile::TempDir;

    /// Helper to create test database with all CFs
    fn create_test_db() -> (RocksDbMemex, TempDir) {
        let tmp = TempDir::new().expect("Failed to create temp dir");
        let memex = RocksDbMemex::open(tmp.path()).expect("Failed to open RocksDB");
        (memex, tmp)
    }

    // =========================================================================
    // TC-SESSION-05: Save/Load Round-Trip
    // =========================================================================
    #[test]
    fn tc_session_05_save_load_roundtrip() {
        println!("\n=== TC-SESSION-05: Save/Load Round-Trip ===");

        let (memex, _tmp) = create_test_db();

        // SOURCE OF TRUTH: Create snapshot with known values
        let mut snapshot = SessionIdentitySnapshot::new("test-session-05");
        snapshot.last_ic = 0.85;
        snapshot.consciousness = 0.7;
        snapshot.integration = 0.9;
        snapshot.kuramoto_phases = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0];
        snapshot.purpose_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3];
        snapshot.append_to_trajectory([0.5; KURAMOTO_N]);

        println!("BEFORE: Created snapshot");
        println!("  session_id: {}", snapshot.session_id);
        println!("  last_ic: {}", snapshot.last_ic);
        println!("  trajectory.len: {}", snapshot.trajectory.len());

        // EXECUTE: Save
        memex.save_snapshot(&snapshot).expect("save_snapshot must succeed");
        println!("AFTER SAVE: Snapshot written to RocksDB");

        // EXECUTE: Load by session_id
        let loaded = memex
            .load_snapshot(&snapshot.session_id)
            .expect("load_snapshot must succeed")
            .expect("Snapshot must exist");

        println!("AFTER LOAD:");
        println!("  session_id: {}", loaded.session_id);
        println!("  last_ic: {}", loaded.last_ic);
        println!("  trajectory.len: {}", loaded.trajectory.len());

        // VERIFY: Round-trip equality
        assert_eq!(loaded.session_id, snapshot.session_id, "session_id mismatch");
        assert_eq!(loaded.timestamp_ms, snapshot.timestamp_ms, "timestamp_ms mismatch");
        assert_eq!(loaded.last_ic, snapshot.last_ic, "last_ic mismatch");
        assert_eq!(loaded.consciousness, snapshot.consciousness, "consciousness mismatch");
        assert_eq!(loaded.integration, snapshot.integration, "integration mismatch");
        assert_eq!(loaded.kuramoto_phases, snapshot.kuramoto_phases, "kuramoto_phases mismatch");
        assert_eq!(loaded.purpose_vector, snapshot.purpose_vector, "purpose_vector mismatch");
        assert_eq!(loaded.trajectory, snapshot.trajectory, "trajectory mismatch");
        assert_eq!(loaded, snapshot, "Full snapshot equality");

        println!("RESULT: PASS - Round-trip preserves all 14 fields");
    }

    // =========================================================================
    // TC-SESSION-06: Temporal Ordering (load_latest)
    // =========================================================================
    #[test]
    fn tc_session_06_temporal_ordering() {
        println!("\n=== TC-SESSION-06: Temporal Ordering ===");

        let (memex, _tmp) = create_test_db();

        // Create 3 snapshots with different timestamps
        let mut s1 = SessionIdentitySnapshot::new("session-older");
        s1.timestamp_ms = 1000;
        s1.last_ic = 0.5;

        let mut s2 = SessionIdentitySnapshot::new("session-middle");
        s2.timestamp_ms = 2000;
        s2.last_ic = 0.7;

        let mut s3 = SessionIdentitySnapshot::new("session-newest");
        s3.timestamp_ms = 3000;
        s3.last_ic = 0.9;

        println!("BEFORE: Created 3 snapshots");
        println!("  s1: ts={}, ic={}", s1.timestamp_ms, s1.last_ic);
        println!("  s2: ts={}, ic={}", s2.timestamp_ms, s2.last_ic);
        println!("  s3: ts={}, ic={}", s3.timestamp_ms, s3.last_ic);

        // Save in non-chronological order to test ordering
        memex.save_snapshot(&s2).expect("save s2");
        memex.save_snapshot(&s1).expect("save s1");
        memex.save_snapshot(&s3).expect("save s3");
        println!("AFTER SAVE: All 3 saved (order: s2, s1, s3)");

        // EXECUTE: Load latest
        let latest = memex
            .load_latest()
            .expect("load_latest must succeed")
            .expect("Should find latest");

        println!("AFTER LOAD_LATEST:");
        println!("  session_id: {}", latest.session_id);
        println!("  timestamp_ms: {}", latest.timestamp_ms);
        println!("  last_ic: {}", latest.last_ic);

        // VERIFY: Should be s3 (newest based on last save)
        assert_eq!(latest.session_id, "session-newest", "Should return newest session");
        assert_eq!(latest.timestamp_ms, 3000, "Should have newest timestamp");
        assert_eq!(latest.last_ic, 0.9, "Should have newest IC value");

        println!("RESULT: PASS - load_latest returns most recent session");
    }

    // =========================================================================
    // TC-SESSION-07: Fresh Install Returns None
    // =========================================================================
    #[test]
    fn tc_session_07_fresh_install_returns_none() {
        println!("\n=== TC-SESSION-07: Fresh Install Returns None ===");

        let (memex, _tmp) = create_test_db();

        // EXECUTE: Load latest on fresh DB
        let result = memex.load_latest().expect("load_latest must succeed");

        println!("AFTER LOAD_LATEST on fresh DB:");
        println!("  result: {:?}", result.is_none());

        // VERIFY: Should be None
        assert!(result.is_none(), "Fresh install must return None");

        println!("RESULT: PASS - Fresh install returns None");
    }

    // =========================================================================
    // TC-SESSION-08: Error on Missing Session ID
    // =========================================================================
    #[test]
    fn tc_session_08_error_on_missing_session_id() {
        println!("\n=== TC-SESSION-08: Error on Missing Session ID ===");

        let (memex, _tmp) = create_test_db();

        // EXECUTE: Try to load non-existent session
        let result = memex.load_snapshot_by_id("non-existent-session");

        println!("AFTER LOAD_SNAPSHOT_BY_ID for missing session:");
        println!("  result: {:?}", result);

        // VERIFY: Should be NotFound error
        match result {
            Err(StorageError::NotFound { id }) => {
                println!("  error.id: {}", id);
                assert!(id.contains("non-existent-session"), "Error should contain session ID");
                println!("RESULT: PASS - NotFound error for missing session");
            }
            Ok(_) => panic!("Should have returned NotFound error, got Ok"),
            Err(e) => panic!("Should have returned NotFound error, got: {:?}", e),
        }
    }

    // =========================================================================
    // EDGE CASE: Unicode Session ID
    // =========================================================================
    #[test]
    fn edge_case_unicode_session_id() {
        println!("\n=== EDGE CASE: Unicode Session ID ===");

        let (memex, _tmp) = create_test_db();

        let session_id = "test-\u{1F600}-emoji-\u{4E2D}\u{6587}-chinese";
        let snapshot = SessionIdentitySnapshot::new(session_id);

        println!("BEFORE: Created snapshot with unicode session_id");
        println!("  session_id: {}", snapshot.session_id);
        println!("  session_id bytes: {:?}", snapshot.session_id.as_bytes());

        // EXECUTE: Save and load
        memex.save_snapshot(&snapshot).expect("save must succeed");
        let loaded = memex
            .load_snapshot(session_id)
            .expect("load must succeed")
            .expect("Snapshot must exist");

        println!("AFTER: Loaded snapshot");
        println!("  session_id: {}", loaded.session_id);

        // VERIFY
        assert_eq!(loaded.session_id, session_id, "Unicode session_id must round-trip");
        println!("RESULT: PASS - Unicode session ID handled correctly");
    }

    // =========================================================================
    // EDGE CASE: Large Trajectory (MAX_TRAJECTORY_LEN)
    // =========================================================================
    #[test]
    fn edge_case_max_trajectory() {
        println!("\n=== EDGE CASE: Maximum Trajectory Size ===");

        let (memex, _tmp) = create_test_db();

        let mut snapshot = SessionIdentitySnapshot::new("large-trajectory-test");

        // Fill trajectory to max
        for i in 0..MAX_TRAJECTORY_LEN {
            snapshot.append_to_trajectory([i as f32; KURAMOTO_N]);
        }

        println!("BEFORE: Created snapshot with max trajectory");
        println!("  trajectory.len: {}", snapshot.trajectory.len());
        println!("  estimated_size: {} bytes", snapshot.estimated_size());

        // EXECUTE: Save and load
        memex.save_snapshot(&snapshot).expect("save must succeed");
        let loaded = memex
            .load_snapshot(&snapshot.session_id)
            .expect("load must succeed")
            .expect("Snapshot must exist");

        println!("AFTER: Loaded snapshot");
        println!("  trajectory.len: {}", loaded.trajectory.len());

        // VERIFY
        assert_eq!(loaded.trajectory.len(), MAX_TRAJECTORY_LEN, "Trajectory length preserved");
        assert_eq!(loaded.trajectory, snapshot.trajectory, "Trajectory data preserved");
        println!("RESULT: PASS - Max trajectory handled correctly");
    }

    // =========================================================================
    // EDGE CASE: Boundary IC Values
    // =========================================================================
    #[test]
    fn edge_case_boundary_ic_values() {
        println!("\n=== EDGE CASE: Boundary IC Values ===");

        let (memex, _tmp) = create_test_db();

        let test_cases = [
            ("ic-zero", 0.0_f32),
            ("ic-one", 1.0_f32),
            ("ic-healthy", 0.95_f32),
            ("ic-warning", 0.65_f32),
            ("ic-critical", 0.45_f32),
            ("ic-tiny", f32::MIN_POSITIVE),
            ("ic-epsilon", f32::EPSILON),
        ];

        for (session_id, ic_value) in test_cases {
            let mut snapshot = SessionIdentitySnapshot::new(session_id);
            snapshot.last_ic = ic_value;
            snapshot.cross_session_ic = ic_value;

            println!("Testing IC={} ({:.2e})", ic_value, ic_value);

            memex.save_snapshot(&snapshot).expect("save must succeed");
            let loaded = memex
                .load_snapshot(session_id)
                .expect("load must succeed")
                .expect("Snapshot must exist");

            assert_eq!(loaded.last_ic, ic_value, "IC value {} must round-trip", ic_value);
            assert_eq!(
                loaded.cross_session_ic, ic_value,
                "cross_session_ic {} must round-trip",
                ic_value
            );
        }

        println!("RESULT: PASS - All boundary IC values handled correctly");
    }

    // =========================================================================
    // EDGE CASE: Temporal Recovery (corrupt latest pointer)
    // =========================================================================
    #[test]
    fn edge_case_temporal_recovery() {
        println!("\n=== EDGE CASE: Temporal Recovery ===");

        let (memex, _tmp) = create_test_db();

        // Save a valid snapshot
        let mut snapshot = SessionIdentitySnapshot::new("recovery-test");
        snapshot.timestamp_ms = 5000;
        snapshot.last_ic = 0.88;
        memex.save_snapshot(&snapshot).expect("save must succeed");

        println!("BEFORE: Saved snapshot with ts=5000");

        // Corrupt the latest pointer by writing a non-existent session_id
        let cf = memex.get_cf(CF_SESSION_IDENTITY).unwrap();
        memex
            .db
            .put_cf(cf, SESSION_LATEST_KEY, b"non-existent-session")
            .unwrap();

        println!("CORRUPTED: Set latest pointer to non-existent session");

        // EXECUTE: Load latest should recover via temporal index
        let recovered = memex
            .load_latest()
            .expect("load_latest must succeed even with corrupt pointer")
            .expect("Should recover via temporal index");

        println!("AFTER RECOVERY:");
        println!("  session_id: {}", recovered.session_id);
        println!("  timestamp_ms: {}", recovered.timestamp_ms);
        println!("  last_ic: {}", recovered.last_ic);

        // VERIFY: Should have recovered the original snapshot
        assert_eq!(recovered.session_id, "recovery-test", "Should recover correct session");
        assert_eq!(recovered.timestamp_ms, 5000, "Should recover correct timestamp");
        assert_eq!(recovered.last_ic, 0.88, "Should recover correct IC");

        println!("RESULT: PASS - Temporal recovery works with corrupt latest pointer");
    }

    // =========================================================================
    // PERFORMANCE: Measure save/load latency
    // =========================================================================
    #[test]
    fn performance_latency_measurement() {
        println!("\n=== PERFORMANCE: Latency Measurement ===");

        let (memex, _tmp) = create_test_db();

        let mut snapshot = SessionIdentitySnapshot::new("perf-test");
        // Add some trajectory data to make it realistic
        for i in 0..25 {
            snapshot.append_to_trajectory([i as f32; KURAMOTO_N]);
        }

        // Measure save latency
        let save_start = std::time::Instant::now();
        memex.save_snapshot(&snapshot).expect("save must succeed");
        let save_latency = save_start.elapsed();

        // Measure load latency
        let load_start = std::time::Instant::now();
        let _ = memex
            .load_snapshot(&snapshot.session_id)
            .expect("load must succeed");
        let load_latency = load_start.elapsed();

        // Measure load_latest latency
        let latest_start = std::time::Instant::now();
        let _ = memex.load_latest().expect("load_latest must succeed");
        let latest_latency = latest_start.elapsed();

        println!("LATENCY RESULTS:");
        println!("  save_snapshot: {:?}", save_latency);
        println!("  load_snapshot: {:?}", load_latency);
        println!("  load_latest: {:?}", latest_latency);

        // Performance targets (relaxed for CI, actual p95 targets are stricter)
        assert!(
            save_latency.as_millis() < 100,
            "save_snapshot too slow: {:?}",
            save_latency
        );
        assert!(
            load_latency.as_millis() < 50,
            "load_snapshot too slow: {:?}",
            load_latency
        );
        assert!(
            latest_latency.as_millis() < 50,
            "load_latest too slow: {:?}",
            latest_latency
        );

        println!("RESULT: PASS - All operations within acceptable latency");
    }

    // =========================================================================
    // MANUAL VERIFICATION: Raw RocksDB Data Inspection
    // =========================================================================
    #[test]
    fn manual_verification_raw_rocksdb_data() {
        println!("\n=== MANUAL VERIFICATION: Raw RocksDB Data Inspection ===");

        let (memex, _tmp) = create_test_db();

        // Create snapshot with known data
        let mut snapshot = SessionIdentitySnapshot::new("verify-session-xyz");
        snapshot.last_ic = 0.92;
        snapshot.consciousness = 0.78;
        snapshot.kuramoto_phases = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.1, 12.1, 13.1];
        snapshot.append_to_trajectory([0.5; KURAMOTO_N]);

        // Save
        memex.save_snapshot(&snapshot).expect("save must succeed");

        // Get CF handle
        let cf = memex.get_cf(CF_SESSION_IDENTITY).expect("CF must exist");

        // 1. Verify session key exists with data
        let session_key = session_identity_key("verify-session-xyz");
        let session_data = memex.db.get_cf(cf, &session_key).expect("get must succeed");
        println!("\n1. SESSION DATA CHECK:");
        println!("   Key: s:verify-session-xyz");
        println!("   Key bytes: {:?}", &session_key[..std::cmp::min(20, session_key.len())]);
        assert!(session_data.is_some(), "Session data must exist");
        let data = session_data.unwrap();
        println!("   Data exists: YES");
        println!("   Data length: {} bytes", data.len());
        assert!(data.len() > 100, "Data should be >100 bytes (got {})", data.len());

        // 2. Verify latest pointer
        let latest_data = memex.db.get_cf(cf, SESSION_LATEST_KEY).expect("get must succeed");
        println!("\n2. LATEST POINTER CHECK:");
        println!("   Key: {:?}", SESSION_LATEST_KEY);
        assert!(latest_data.is_some(), "Latest pointer must exist");
        let latest = latest_data.unwrap();
        let latest_str = std::str::from_utf8(&latest).expect("UTF-8");
        println!("   Value: {}", latest_str);
        assert_eq!(latest_str, "verify-session-xyz", "Latest must point to our session");

        // 3. Verify temporal index
        let temporal_key = session_temporal_key(snapshot.timestamp_ms);
        let temporal_data = memex.db.get_cf(cf, &temporal_key).expect("get must succeed");
        println!("\n3. TEMPORAL INDEX CHECK:");
        println!("   Key (first 10 bytes): {:?}", &temporal_key[..std::cmp::min(10, temporal_key.len())]);
        println!("   Timestamp: {}", snapshot.timestamp_ms);
        assert!(temporal_data.is_some(), "Temporal index must exist");
        let temporal = temporal_data.unwrap();
        let temporal_str = std::str::from_utf8(&temporal).expect("UTF-8");
        println!("   Value: {}", temporal_str);
        assert_eq!(temporal_str, "verify-session-xyz", "Temporal must point to our session");

        // 4. Verify deserialization matches original
        let loaded = memex
            .load_snapshot("verify-session-xyz")
            .expect("load must succeed")
            .expect("must exist");
        println!("\n4. DESERIALIZATION VERIFICATION:");
        println!("   session_id: {} == {}", loaded.session_id, snapshot.session_id);
        println!("   last_ic: {} == {}", loaded.last_ic, snapshot.last_ic);
        println!("   consciousness: {} == {}", loaded.consciousness, snapshot.consciousness);
        println!("   kuramoto_phases[0]: {} == {}", loaded.kuramoto_phases[0], snapshot.kuramoto_phases[0]);
        println!("   trajectory.len: {} == {}", loaded.trajectory.len(), snapshot.trajectory.len());

        assert_eq!(loaded, snapshot, "Full equality check");

        println!("\n=== RESULT: PASS - All raw RocksDB data verified ===");
        println!("   - Session key exists with {} bytes", data.len());
        println!("   - Latest pointer correctly points to session");
        println!("   - Temporal index correctly points to session");
        println!("   - Deserialization produces identical snapshot");
    }
}
