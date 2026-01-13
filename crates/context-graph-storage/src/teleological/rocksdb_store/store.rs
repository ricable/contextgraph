//! Core RocksDbTeleologicalStore struct and constructor.
//!
//! This module contains the main store struct, its constructor methods,
//! and basic accessor methods. Complex operations are in separate modules:
//! - `index_ops.rs` - HNSW index operations
//! - `inverted_index.rs` - SPLADE inverted index operations

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use bincode;
use rocksdb::{Cache, ColumnFamily, Options, WriteBatch, DB};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use context_graph_core::types::fingerprint::TeleologicalFingerprint;

use crate::teleological::column_families::{
    get_all_teleological_cf_descriptors, CF_CONTENT, CF_E12_LATE_INTERACTION,
    CF_E1_MATRYOSHKA_128, CF_EGO_NODE, CF_FINGERPRINTS, CF_PURPOSE_VECTORS,
    QUANTIZED_EMBEDDER_CFS, TELEOLOGICAL_CFS,
};
use crate::teleological::indexes::EmbedderIndexRegistry;
use crate::teleological::schema::{
    e12_late_interaction_key, e1_matryoshka_128_key, fingerprint_key, purpose_vector_key,
};
use crate::teleological::serialization::{
    serialize_e1_matryoshka_128, serialize_purpose_vector, serialize_teleological_fingerprint,
};

use super::types::{TeleologicalStoreConfig, TeleologicalStoreError, TeleologicalStoreResult};

// ============================================================================
// Main Store Struct
// ============================================================================

/// RocksDB-backed storage for TeleologicalFingerprints.
///
/// Implements the `TeleologicalMemoryStore` trait with persistent storage
/// across 17 column families for efficient indexing and retrieval.
///
/// # Thread Safety
///
/// The store is thread-safe for concurrent access:
/// - RocksDB handles internal locking for reads/writes
/// - HNSW indexes are protected by RwLock
///
/// # Example
///
/// ```ignore
/// use context_graph_storage::teleological::RocksDbTeleologicalStore;
/// use tempfile::TempDir;
///
/// let tmp = TempDir::new().unwrap();
/// let store = RocksDbTeleologicalStore::open(tmp.path()).unwrap();
///
/// // Store a fingerprint
/// let id = store.store(fingerprint).await.unwrap();
///
/// // Retrieve it
/// let retrieved = store.retrieve(id).await.unwrap();
/// ```
pub struct RocksDbTeleologicalStore {
    /// The RocksDB database instance.
    pub(crate) db: Arc<DB>,
    /// Shared block cache across column families.
    #[allow(dead_code)]
    pub(crate) cache: Cache,
    /// Database path.
    pub(crate) path: PathBuf,
    /// In-memory count of fingerprints (cached for performance).
    pub(crate) fingerprint_count: RwLock<Option<usize>>,
    /// Soft-deleted IDs (tracked in memory for filtering).
    pub(crate) soft_deleted: RwLock<HashMap<Uuid, bool>>,
    /// Per-embedder index registry with 12 HNSW indexes for O(log n) ANN search.
    /// E6, E12, E13 use different index types (inverted/MaxSim).
    /// NO FALLBACKS - FAIL FAST on invalid operations.
    pub(crate) index_registry: Arc<EmbedderIndexRegistry>,
}

// ============================================================================
// Constructor and Open Methods
// ============================================================================

impl RocksDbTeleologicalStore {
    /// Open a teleological store at the specified path with default configuration.
    ///
    /// Creates the database and all 17 column families if they don't exist.
    /// **Automatically detects and removes stale lock files.**
    pub fn open<P: AsRef<Path>>(path: P) -> TeleologicalStoreResult<Self> {
        Self::open_with_config(path, TeleologicalStoreConfig::default())
    }

    /// Open a teleological store with custom configuration.
    pub fn open_with_config<P: AsRef<Path>>(
        path: P,
        config: TeleologicalStoreConfig,
    ) -> TeleologicalStoreResult<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let path_str = path_buf.to_string_lossy().to_string();

        info!(
            "Opening RocksDbTeleologicalStore at '{}' with cache_size={}MB",
            path_str,
            config.block_cache_size / (1024 * 1024)
        );

        // STALE LOCK DETECTION: Check for and remove stale lock files before opening.
        if path_buf.exists() {
            match Self::detect_and_remove_stale_lock(&path_buf) {
                Ok(true) => {
                    info!("Removed stale lock at '{}', proceeding with open", path_str);
                }
                Ok(false) => {
                    debug!("No stale lock detected at '{}'", path_str);
                }
                Err(e) => {
                    error!(
                        "FAIL FAST: Stale lock cleanup failed at '{}': {}",
                        path_str, e
                    );
                    return Err(e);
                }
            }
        }

        // Create shared block cache
        let cache = Cache::new_lru_cache(config.block_cache_size);

        // Create DB options
        let mut db_opts = Options::default();
        db_opts.create_if_missing(config.create_if_missing);
        db_opts.create_missing_column_families(true);
        db_opts.set_max_open_files(config.max_open_files);

        if !config.enable_wal {
            db_opts.set_manual_wal_flush(true);
        }

        // Get all 17 teleological column family descriptors
        let cf_descriptors = get_all_teleological_cf_descriptors(&cache);

        debug!(
            "Opening database with {} column families",
            cf_descriptors.len()
        );

        // Open database with all column families
        let db = DB::open_cf_descriptors(&db_opts, &path_str, cf_descriptors).map_err(|e| {
            error!("Failed to open RocksDB at '{}': {}", path_str, e);
            TeleologicalStoreError::OpenFailed {
                path: path_str.clone(),
                message: e.to_string(),
            }
        })?;

        // Create per-embedder index registry (12 HNSW indexes)
        let index_registry = Arc::new(EmbedderIndexRegistry::new());

        info!(
            "Successfully opened RocksDbTeleologicalStore with {} column families and {} per-embedder indexes",
            TELEOLOGICAL_CFS.len() + QUANTIZED_EMBEDDER_CFS.len(),
            index_registry.len()
        );

        Ok(Self {
            db: Arc::new(db),
            cache,
            path: path_buf,
            fingerprint_count: RwLock::new(None),
            soft_deleted: RwLock::new(HashMap::new()),
            index_registry,
        })
    }

    /// Detect and remove stale RocksDB lock files.
    fn detect_and_remove_stale_lock<P: AsRef<Path>>(path: P) -> TeleologicalStoreResult<bool> {
        let lock_path = path.as_ref().join("LOCK");
        let lock_path_str = lock_path.to_string_lossy().to_string();

        if !lock_path.exists() {
            debug!(
                "No LOCK file at '{}' - database not previously opened",
                lock_path_str
            );
            return Ok(false);
        }

        info!(
            "LOCK file exists at '{}' - checking if stale",
            lock_path_str
        );

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;

            let file = match fs::OpenOptions::new().read(true).write(true).open(&lock_path) {
                Ok(f) => f,
                Err(e) => {
                    warn!("Cannot open LOCK file for inspection: {}", e);
                    return Self::try_remove_lock_file(&lock_path, &lock_path_str);
                }
            };

            let fd = file.as_raw_fd();
            let result = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };

            if result == 0 {
                info!(
                    "Acquired exclusive lock - LOCK file at '{}' is STALE",
                    lock_path_str
                );
                unsafe { libc::flock(fd, libc::LOCK_UN) };
                drop(file);
                Self::try_remove_lock_file(&lock_path, &lock_path_str)
            } else {
                let errno = std::io::Error::last_os_error();
                if errno.raw_os_error() == Some(libc::EWOULDBLOCK) {
                    info!(
                        "LOCK file at '{}' is held by another process - NOT stale",
                        lock_path_str
                    );
                    Ok(false)
                } else {
                    warn!("flock() failed with unexpected error: {}", errno);
                    Self::try_remove_lock_file(&lock_path, &lock_path_str)
                }
            }
        }

        #[cfg(windows)]
        {
            Self::try_remove_lock_file(&lock_path, &lock_path_str)
        }

        #[cfg(not(any(unix, windows)))]
        {
            Self::try_remove_lock_file(&lock_path, &lock_path_str)
        }
    }

    /// Attempt to remove a stale lock file.
    fn try_remove_lock_file(lock_path: &Path, lock_path_str: &str) -> TeleologicalStoreResult<bool> {
        match fs::remove_file(lock_path) {
            Ok(()) => {
                info!(
                    "Successfully removed stale LOCK file at '{}'",
                    lock_path_str
                );
                Ok(true)
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                debug!("LOCK file at '{}' was already removed", lock_path_str);
                Ok(true)
            }
            Err(e) => {
                error!(
                    "FAIL FAST: Cannot remove stale LOCK file at '{}': {}",
                    lock_path_str, e
                );
                Err(TeleologicalStoreError::StaleLockCleanupFailed {
                    path: lock_path_str.to_string(),
                    message: format!(
                        "Lock file exists but cannot be removed: {}. \
                        Manual intervention required: remove '{}'",
                        e, lock_path_str
                    ),
                })
            }
        }
    }
}

// ============================================================================
// Column Family Accessors
// ============================================================================

impl RocksDbTeleologicalStore {
    /// Get a column family handle by name.
    pub(crate) fn get_cf(&self, name: &str) -> TeleologicalStoreResult<&ColumnFamily> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| TeleologicalStoreError::ColumnFamilyNotFound {
                name: name.to_string(),
            })
    }

    /// Get the content column family handle (FAIL FAST on missing).
    #[inline]
    pub(crate) fn cf_content(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_CONTENT)
            .expect("CF_CONTENT must exist - database initialization failed")
    }

    /// Get the ego_node column family handle (FAIL FAST on missing).
    #[inline]
    pub(crate) fn cf_ego_node(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_EGO_NODE)
            .expect("CF_EGO_NODE must exist - database initialization failed")
    }

    /// Get the e12_late_interaction column family handle (FAIL FAST on missing).
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn cf_e12_late_interaction(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_E12_LATE_INTERACTION)
            .expect("CF_E12_LATE_INTERACTION must exist - database initialization failed")
    }
}

// ============================================================================
// Core Storage Operations
// ============================================================================

impl RocksDbTeleologicalStore {
    /// Store a fingerprint in all relevant column families.
    ///
    /// Writes to:
    /// 1. `fingerprints` - Full serialized fingerprint
    /// 2. `purpose_vectors` - 13D purpose vector for fast queries
    /// 3. `e1_matryoshka_128` - Truncated E1 embedding for Stage 2
    /// 4. `e13_splade_inverted` - Updates inverted index for Stage 1
    /// 5. `e12_late_interaction` - ColBERT token embeddings for Stage 5
    pub(crate) fn store_fingerprint_internal(
        &self,
        fp: &TeleologicalFingerprint,
    ) -> TeleologicalStoreResult<()> {
        let id = fp.id;
        let key = fingerprint_key(&id);

        let mut batch = WriteBatch::default();

        // 1. Store full fingerprint
        let cf_fingerprints = self.get_cf(CF_FINGERPRINTS)?;
        let serialized = serialize_teleological_fingerprint(fp);
        batch.put_cf(cf_fingerprints, key, &serialized);

        // 2. Store purpose vector
        let cf_purpose = self.get_cf(CF_PURPOSE_VECTORS)?;
        let purpose_key = purpose_vector_key(&id);
        let purpose_bytes = serialize_purpose_vector(&fp.purpose_vector.alignments);
        batch.put_cf(cf_purpose, purpose_key, purpose_bytes);

        // 3. Store E1 Matryoshka 128D truncated vector
        let cf_matryoshka = self.get_cf(CF_E1_MATRYOSHKA_128)?;
        let matryoshka_key = e1_matryoshka_128_key(&id);
        let mut truncated = [0.0f32; 128];
        let e1 = &fp.semantic.e1_semantic;
        let copy_len = std::cmp::min(e1.len(), 128);
        truncated[..copy_len].copy_from_slice(&e1[..copy_len]);
        let matryoshka_bytes = serialize_e1_matryoshka_128(&truncated);
        batch.put_cf(cf_matryoshka, matryoshka_key, matryoshka_bytes);

        // 4. Update E13 SPLADE inverted index
        self.update_splade_inverted_index(&mut batch, &id, &fp.semantic.e13_splade)?;

        // 5. Store E12 late interaction tokens
        if !fp.semantic.e12_late_interaction.is_empty() {
            let cf_e12 = self.get_cf(CF_E12_LATE_INTERACTION)?;
            let e12_key = e12_late_interaction_key(&id);
            let e12_bytes = bincode::serialize(&fp.semantic.e12_late_interaction).map_err(|e| {
                error!(
                    "Failed to serialize E12 tokens for fingerprint {}: {}",
                    id, e
                );
                TeleologicalStoreError::Serialization {
                    id: Some(id),
                    message: format!("E12 token serialization failed: {}", e),
                }
            })?;
            batch.put_cf(cf_e12, e12_key, &e12_bytes);
            debug!(
                "Stored {} E12 tokens ({} bytes) for fingerprint {}",
                fp.semantic.e12_late_interaction.len(),
                e12_bytes.len(),
                id
            );
        }

        // Execute atomic batch write
        self.db.write(batch).map_err(|e| {
            error!("Failed to write fingerprint batch for {}: {}", id, e);
            TeleologicalStoreError::rocksdb_op("write_batch", CF_FINGERPRINTS, Some(id), e)
        })?;

        // Invalidate count cache
        if let Ok(mut count) = self.fingerprint_count.write() {
            *count = None;
        }

        debug!("Stored fingerprint {} ({} bytes)", id, serialized.len());
        Ok(())
    }

    /// Retrieve raw fingerprint bytes from RocksDB.
    pub(crate) fn get_fingerprint_raw(&self, id: Uuid) -> TeleologicalStoreResult<Option<Vec<u8>>> {
        let cf = self.get_cf(CF_FINGERPRINTS)?;
        let key = fingerprint_key(&id);

        self.db
            .get_cf(cf, key)
            .map_err(|e| TeleologicalStoreError::rocksdb_op("get", CF_FINGERPRINTS, Some(id), e))
    }

    /// Check if an ID is soft-deleted.
    pub(crate) fn is_soft_deleted(&self, id: &Uuid) -> bool {
        if let Ok(deleted) = self.soft_deleted.read() {
            deleted.get(id).copied().unwrap_or(false)
        } else {
            false
        }
    }
}

// ============================================================================
// Public API Methods
// ============================================================================

impl RocksDbTeleologicalStore {
    /// Get the database path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get a reference to the underlying RocksDB instance FOR TESTING/DIAGNOSTICS ONLY.
    #[doc(hidden)]
    pub fn db(&self) -> &DB {
        &self.db
    }

    /// Health check: verify all column families are accessible.
    pub fn health_check(&self) -> TeleologicalStoreResult<()> {
        for cf_name in TELEOLOGICAL_CFS {
            self.get_cf(cf_name)?;
        }
        for cf_name in QUANTIZED_EMBEDDER_CFS {
            self.get_cf(cf_name)?;
        }
        Ok(())
    }

    /// Get raw bytes from a specific column family (for debugging).
    pub fn get_raw_bytes(
        &self,
        cf_name: &str,
        key: &[u8],
    ) -> TeleologicalStoreResult<Option<Vec<u8>>> {
        let cf = self.get_cf(cf_name)?;
        self.db.get_cf(cf, key).map_err(|e| {
            TeleologicalStoreError::Internal(format!(
                "RocksDB get_raw failed on CF '{}': {}",
                cf_name, e
            ))
        })
    }
}
