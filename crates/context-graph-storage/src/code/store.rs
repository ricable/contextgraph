//! CodeStore implementation for code entity and E7 embedding storage.

use super::error::{CodeStorageError, CodeStorageResult};
use crate::teleological::column_families::{
    CF_CODE_ENTITIES, CF_CODE_E7_EMBEDDINGS, CF_CODE_FILE_INDEX, CF_CODE_NAME_INDEX,
    CF_CODE_SIGNATURE_INDEX,
};
use context_graph_core::types::{CodeEntity, CodeFileIndexEntry, CodeLanguage, CodeStats};
use rocksdb::{ColumnFamily, IteratorMode, DB};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

/// E7 embedding dimension (Qodo-Embed-1-1.5B).
pub const E7_CODE_DIM: usize = 1536;

/// CodeStore provides storage for code entities and their E7 embeddings.
///
/// # Architecture
/// - Uses separate column families from teleological storage
/// - E7 embeddings are stored directly (not as part of TeleologicalFingerprint)
/// - Secondary indexes for efficient name and signature search
/// - File-level tracking for change detection and cleanup
///
/// # Thread Safety
/// CodeStore uses `Arc<DB>` and is `Send + Sync`.
pub struct CodeStore {
    /// RocksDB database handle.
    db: Arc<DB>,
}

impl CodeStore {
    /// Create a new CodeStore wrapping an existing RocksDB database.
    ///
    /// The database must have been opened with the code column families.
    /// Use `get_code_cf_descriptors()` when opening the database.
    ///
    /// # Arguments
    /// * `db` - Existing RocksDB database with code column families
    ///
    /// # Errors
    /// Returns error if required column families are not found.
    pub fn new(db: Arc<DB>) -> CodeStorageResult<Self> {
        // Verify all required column families exist
        for cf_name in &[
            CF_CODE_ENTITIES,
            CF_CODE_E7_EMBEDDINGS,
            CF_CODE_FILE_INDEX,
            CF_CODE_NAME_INDEX,
            CF_CODE_SIGNATURE_INDEX,
        ] {
            if db.cf_handle(cf_name).is_none() {
                return Err(CodeStorageError::cf_not_found(*cf_name));
            }
        }

        Ok(Self { db })
    }

    /// Open or create a CodeStore at the given path.
    ///
    /// This is a convenience method that opens a RocksDB database with
    /// the code column families.
    ///
    /// # Arguments
    /// * `path` - Path to the database directory
    ///
    /// # Errors
    /// Returns error if database cannot be opened or created.
    pub fn open(path: impl AsRef<Path>) -> CodeStorageResult<Self> {
        use crate::teleological::column_families::get_code_cf_descriptors;
        use rocksdb::{Cache, Options};

        let cache = Cache::new_lru_cache(64 * 1024 * 1024); // 64MB cache
        let cf_descriptors = get_code_cf_descriptors(&cache);

        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);

        let db = DB::open_cf_descriptors(&db_opts, path, cf_descriptors)?;
        Self::new(Arc::new(db))
    }

    // =========================================================================
    // Column Family Helpers
    // =========================================================================

    fn cf_entities(&self) -> CodeStorageResult<&ColumnFamily> {
        self.db
            .cf_handle(CF_CODE_ENTITIES)
            .ok_or_else(|| CodeStorageError::cf_not_found(CF_CODE_ENTITIES))
    }

    fn cf_embeddings(&self) -> CodeStorageResult<&ColumnFamily> {
        self.db
            .cf_handle(CF_CODE_E7_EMBEDDINGS)
            .ok_or_else(|| CodeStorageError::cf_not_found(CF_CODE_E7_EMBEDDINGS))
    }

    fn cf_file_index(&self) -> CodeStorageResult<&ColumnFamily> {
        self.db
            .cf_handle(CF_CODE_FILE_INDEX)
            .ok_or_else(|| CodeStorageError::cf_not_found(CF_CODE_FILE_INDEX))
    }

    fn cf_name_index(&self) -> CodeStorageResult<&ColumnFamily> {
        self.db
            .cf_handle(CF_CODE_NAME_INDEX)
            .ok_or_else(|| CodeStorageError::cf_not_found(CF_CODE_NAME_INDEX))
    }

    fn cf_signature_index(&self) -> CodeStorageResult<&ColumnFamily> {
        self.db
            .cf_handle(CF_CODE_SIGNATURE_INDEX)
            .ok_or_else(|| CodeStorageError::cf_not_found(CF_CODE_SIGNATURE_INDEX))
    }

    // =========================================================================
    // Entity CRUD Operations
    // =========================================================================

    /// Store a code entity with its E7 embedding.
    ///
    /// # Arguments
    /// * `entity` - The code entity to store
    /// * `embedding` - E7 embedding vector (must be 1536D)
    ///
    /// # Errors
    /// Returns error if embedding dimension is wrong or storage fails.
    #[instrument(skip(self, entity, embedding), fields(id = %entity.id, name = %entity.name))]
    pub fn store(
        &self,
        entity: &CodeEntity,
        embedding: &[f32],
    ) -> CodeStorageResult<()> {
        // Validate embedding dimension
        if embedding.len() != E7_CODE_DIM {
            return Err(CodeStorageError::InvalidDimension {
                expected: E7_CODE_DIM,
                actual: embedding.len(),
            });
        }

        let id_bytes = entity.id.as_bytes();

        // Serialize entity
        let entity_bytes = bincode::serialize(entity)
            .map_err(|e| CodeStorageError::serialization(e.to_string()))?;

        // Serialize embedding (native f32 bytes)
        let embedding_bytes = Self::serialize_embedding(embedding);

        // Store entity
        self.db
            .put_cf(&self.cf_entities()?, id_bytes, &entity_bytes)?;

        // Store embedding
        self.db
            .put_cf(&self.cf_embeddings()?, id_bytes, &embedding_bytes)?;

        // Update file index
        self.update_file_index(&entity.file_path, entity.id, &entity.language)?;

        // Update name index
        self.update_name_index(&entity.name, entity.id)?;

        // Update signature index if present
        if let Some(ref sig) = entity.signature {
            self.update_signature_index(sig, entity.id)?;
        }

        debug!(
            id = %entity.id,
            name = %entity.name,
            entity_type = %entity.entity_type,
            file = %entity.file_path,
            "Stored code entity"
        );

        Ok(())
    }

    /// Get a code entity by ID.
    ///
    /// # Arguments
    /// * `id` - Entity UUID
    ///
    /// # Returns
    /// The entity if found, None otherwise.
    pub fn get(&self, id: Uuid) -> CodeStorageResult<Option<CodeEntity>> {
        let id_bytes = id.as_bytes();

        match self.db.get_cf(&self.cf_entities()?, id_bytes)? {
            Some(bytes) => {
                let entity: CodeEntity = bincode::deserialize(&bytes)
                    .map_err(|e| CodeStorageError::deserialization(e.to_string()))?;
                Ok(Some(entity))
            }
            None => Ok(None),
        }
    }

    /// Get a code entity by ID, returning error if not found.
    pub fn get_or_error(&self, id: Uuid) -> CodeStorageResult<CodeEntity> {
        self.get(id)?.ok_or(CodeStorageError::not_found(id))
    }

    /// Get the E7 embedding for an entity.
    ///
    /// # Arguments
    /// * `id` - Entity UUID
    ///
    /// # Returns
    /// The embedding vector if found.
    pub fn get_embedding(&self, id: Uuid) -> CodeStorageResult<Option<Vec<f32>>> {
        let id_bytes = id.as_bytes();

        match self.db.get_cf(&self.cf_embeddings()?, id_bytes)? {
            Some(bytes) => {
                let embedding = Self::deserialize_embedding(&bytes)?;
                Ok(Some(embedding))
            }
            None => Ok(None),
        }
    }

    /// Get entity and its embedding together.
    pub fn get_with_embedding(
        &self,
        id: Uuid,
    ) -> CodeStorageResult<Option<(CodeEntity, Vec<f32>)>> {
        let entity = match self.get(id)? {
            Some(e) => e,
            None => return Ok(None),
        };

        let embedding = self
            .get_embedding(id)?
            .ok_or(CodeStorageError::embedding_not_found(id))?;

        Ok(Some((entity, embedding)))
    }

    /// Delete a code entity and its embedding.
    ///
    /// Also removes from all indexes.
    #[instrument(skip(self), fields(id = %id))]
    pub fn delete(&self, id: Uuid) -> CodeStorageResult<bool> {
        // Get entity first to update indexes
        let entity = match self.get(id)? {
            Some(e) => e,
            None => return Ok(false),
        };

        let id_bytes = id.as_bytes();

        // Delete from entity store
        self.db.delete_cf(&self.cf_entities()?, id_bytes)?;

        // Delete embedding
        self.db.delete_cf(&self.cf_embeddings()?, id_bytes)?;

        // Remove from file index
        self.remove_from_file_index(&entity.file_path, id)?;

        // Remove from name index
        self.remove_from_name_index(&entity.name, id)?;

        // Remove from signature index if present
        if let Some(ref sig) = entity.signature {
            self.remove_from_signature_index(sig, id)?;
        }

        debug!(id = %id, name = %entity.name, "Deleted code entity");
        Ok(true)
    }

    /// Check if an entity exists.
    pub fn exists(&self, id: Uuid) -> CodeStorageResult<bool> {
        let id_bytes = id.as_bytes();
        Ok(self.db.get_cf(&self.cf_entities()?, id_bytes)?.is_some())
    }

    // =========================================================================
    // File-Level Operations
    // =========================================================================

    /// Get all entities for a file.
    ///
    /// # Arguments
    /// * `file_path` - Absolute path to the file
    ///
    /// # Returns
    /// Vector of entities in the file.
    pub fn get_by_file(&self, file_path: &str) -> CodeStorageResult<Vec<CodeEntity>> {
        let index_entry = self.get_file_index(file_path)?;

        match index_entry {
            Some(entry) => {
                let mut entities = Vec::with_capacity(entry.entity_ids.len());
                for id in &entry.entity_ids {
                    if let Some(entity) = self.get(*id)? {
                        entities.push(entity);
                    }
                }
                Ok(entities)
            }
            None => Ok(Vec::new()),
        }
    }

    /// Get the file index entry for a file.
    pub fn get_file_index(&self, file_path: &str) -> CodeStorageResult<Option<CodeFileIndexEntry>> {
        match self.db.get_cf(&self.cf_file_index()?, file_path.as_bytes())? {
            Some(bytes) => {
                let entry: CodeFileIndexEntry = bincode::deserialize(&bytes)
                    .map_err(|e| CodeStorageError::deserialization(e.to_string()))?;
                Ok(Some(entry))
            }
            None => Ok(None),
        }
    }

    /// Delete all entities for a file.
    ///
    /// # Returns
    /// Number of entities deleted.
    #[instrument(skip(self), fields(file = %file_path))]
    pub fn delete_file(&self, file_path: &str) -> CodeStorageResult<usize> {
        let index_entry = match self.get_file_index(file_path)? {
            Some(entry) => entry,
            None => return Ok(0),
        };

        let mut deleted = 0;
        for id in &index_entry.entity_ids {
            if self.delete(*id)? {
                deleted += 1;
            }
        }

        // Delete the file index entry itself
        self.db
            .delete_cf(&self.cf_file_index()?, file_path.as_bytes())?;

        info!(file = %file_path, deleted = deleted, "Deleted all entities for file");
        Ok(deleted)
    }

    /// List all indexed files.
    ///
    /// # Arguments
    /// * `path_filter` - Optional glob pattern to filter paths
    ///
    /// # Returns
    /// Vector of (file_path, entity_count) tuples.
    pub fn list_files(&self, path_filter: Option<&str>) -> CodeStorageResult<Vec<(String, usize)>> {
        let cf = self.cf_file_index()?;
        let iter = self.db.iterator_cf(&cf, IteratorMode::Start);

        let mut files = Vec::new();
        for item in iter {
            let (key, value) = item?;
            let path = String::from_utf8_lossy(&key).to_string();

            // Apply optional filter
            if let Some(filter) = path_filter {
                if !Self::glob_match(&path, filter) {
                    continue;
                }
            }

            let entry: CodeFileIndexEntry = bincode::deserialize(&value)
                .map_err(|e| CodeStorageError::deserialization(e.to_string()))?;

            files.push((path, entry.entity_count()));
        }

        Ok(files)
    }

    // =========================================================================
    // Search Operations
    // =========================================================================

    /// Search entities by name prefix.
    ///
    /// # Arguments
    /// * `prefix` - Name prefix to search for
    /// * `limit` - Maximum results to return
    ///
    /// # Returns
    /// Vector of matching entities.
    pub fn search_by_name(&self, prefix: &str, limit: usize) -> CodeStorageResult<Vec<CodeEntity>> {
        let cf = self.cf_name_index()?;
        let iter = self.db.prefix_iterator_cf(&cf, prefix.as_bytes());

        let mut entities = Vec::new();
        for item in iter {
            if entities.len() >= limit {
                break;
            }

            let (key, value) = item?;
            let name = String::from_utf8_lossy(&key);

            // Verify prefix match (prefix_iterator may return non-matches)
            if !name.starts_with(prefix) {
                break;
            }

            let ids: Vec<Uuid> = bincode::deserialize(&value)
                .map_err(|e| CodeStorageError::deserialization(e.to_string()))?;

            for id in ids {
                if entities.len() >= limit {
                    break;
                }
                if let Some(entity) = self.get(id)? {
                    entities.push(entity);
                }
            }
        }

        Ok(entities)
    }

    /// Search entities by exact name.
    pub fn search_by_exact_name(&self, name: &str) -> CodeStorageResult<Vec<CodeEntity>> {
        match self.db.get_cf(&self.cf_name_index()?, name.as_bytes())? {
            Some(bytes) => {
                let ids: Vec<Uuid> = bincode::deserialize(&bytes)
                    .map_err(|e| CodeStorageError::deserialization(e.to_string()))?;

                let mut entities = Vec::with_capacity(ids.len());
                for id in ids {
                    if let Some(entity) = self.get(id)? {
                        entities.push(entity);
                    }
                }
                Ok(entities)
            }
            None => Ok(Vec::new()),
        }
    }

    /// Search entities by signature pattern.
    ///
    /// # Arguments
    /// * `signature` - Function signature to search for
    ///
    /// # Returns
    /// Vector of matching entities.
    pub fn search_by_signature(&self, signature: &str) -> CodeStorageResult<Vec<CodeEntity>> {
        let hash = Self::hash_signature(signature);

        match self.db.get_cf(&self.cf_signature_index()?, &hash)? {
            Some(bytes) => {
                let ids: Vec<Uuid> = bincode::deserialize(&bytes)
                    .map_err(|e| CodeStorageError::deserialization(e.to_string()))?;

                let mut entities = Vec::with_capacity(ids.len());
                for id in ids {
                    if let Some(entity) = self.get(id)? {
                        entities.push(entity);
                    }
                }
                Ok(entities)
            }
            None => Ok(Vec::new()),
        }
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    /// Get storage statistics.
    pub fn get_stats(&self) -> CodeStorageResult<CodeStats> {
        let files = self.list_files(None)?;
        let total_files = files.len();

        let mut total_entities = 0;
        let mut entities_by_type = HashMap::new();
        let mut entities_by_language = HashMap::new();
        let mut total_lines = 0;

        // Iterate all entities
        let cf = self.cf_entities()?;
        let iter = self.db.iterator_cf(&cf, IteratorMode::Start);

        for item in iter {
            let (_key, value) = item?;
            let entity: CodeEntity = bincode::deserialize(&value)
                .map_err(|e| CodeStorageError::deserialization(e.to_string()))?;

            total_entities += 1;
            *entities_by_type.entry(entity.entity_type).or_insert(0) += 1;
            *entities_by_language.entry(entity.language).or_insert(0) += 1;
            total_lines += entity.line_count();
        }

        let avg_entities_per_file = if total_files > 0 {
            total_entities as f64 / total_files as f64
        } else {
            0.0
        };

        Ok(CodeStats {
            total_files,
            total_entities,
            entities_by_type,
            entities_by_language,
            avg_entities_per_file,
            total_lines,
        })
    }

    /// Count total entities.
    pub fn count(&self) -> CodeStorageResult<usize> {
        let cf = self.cf_entities()?;
        let iter = self.db.iterator_cf(&cf, IteratorMode::Start);
        Ok(iter.count())
    }

    // =========================================================================
    // Index Management Helpers
    // =========================================================================

    fn update_file_index(
        &self,
        file_path: &str,
        entity_id: Uuid,
        language: &CodeLanguage,
    ) -> CodeStorageResult<()> {
        let cf = self.cf_file_index()?;
        let key = file_path.as_bytes();

        let mut entry = match self.db.get_cf(&cf, key)? {
            Some(bytes) => bincode::deserialize(&bytes)
                .map_err(|e| CodeStorageError::deserialization(e.to_string()))?,
            None => CodeFileIndexEntry::new(
                file_path.to_string(),
                *language,
                String::new(), // Hash will be set by watcher
            ),
        };

        entry.add_entity(entity_id);

        let bytes = bincode::serialize(&entry)
            .map_err(|e| CodeStorageError::serialization(e.to_string()))?;
        self.db.put_cf(&cf, key, bytes)?;

        Ok(())
    }

    fn remove_from_file_index(&self, file_path: &str, entity_id: Uuid) -> CodeStorageResult<()> {
        let cf = self.cf_file_index()?;
        let key = file_path.as_bytes();

        if let Some(bytes) = self.db.get_cf(&cf, key)? {
            let mut entry: CodeFileIndexEntry = bincode::deserialize(&bytes)
                .map_err(|e| CodeStorageError::deserialization(e.to_string()))?;

            entry.remove_entity(entity_id);

            if entry.is_empty() {
                self.db.delete_cf(&cf, key)?;
            } else {
                let bytes = bincode::serialize(&entry)
                    .map_err(|e| CodeStorageError::serialization(e.to_string()))?;
                self.db.put_cf(&cf, key, bytes)?;
            }
        }

        Ok(())
    }

    fn update_name_index(&self, name: &str, entity_id: Uuid) -> CodeStorageResult<()> {
        let cf = self.cf_name_index()?;
        let key = name.as_bytes();

        let mut ids: Vec<Uuid> = match self.db.get_cf(&cf, key)? {
            Some(bytes) => bincode::deserialize(&bytes)
                .map_err(|e| CodeStorageError::deserialization(e.to_string()))?,
            None => Vec::new(),
        };

        if !ids.contains(&entity_id) {
            ids.push(entity_id);
            let bytes = bincode::serialize(&ids)
                .map_err(|e| CodeStorageError::serialization(e.to_string()))?;
            self.db.put_cf(&cf, key, bytes)?;
        }

        Ok(())
    }

    fn remove_from_name_index(&self, name: &str, entity_id: Uuid) -> CodeStorageResult<()> {
        let cf = self.cf_name_index()?;
        let key = name.as_bytes();

        if let Some(bytes) = self.db.get_cf(&cf, key)? {
            let mut ids: Vec<Uuid> = bincode::deserialize(&bytes)
                .map_err(|e| CodeStorageError::deserialization(e.to_string()))?;

            if let Some(pos) = ids.iter().position(|&id| id == entity_id) {
                ids.remove(pos);

                if ids.is_empty() {
                    self.db.delete_cf(&cf, key)?;
                } else {
                    let bytes = bincode::serialize(&ids)
                        .map_err(|e| CodeStorageError::serialization(e.to_string()))?;
                    self.db.put_cf(&cf, key, bytes)?;
                }
            }
        }

        Ok(())
    }

    fn update_signature_index(&self, signature: &str, entity_id: Uuid) -> CodeStorageResult<()> {
        let cf = self.cf_signature_index()?;
        let key = Self::hash_signature(signature);

        let mut ids: Vec<Uuid> = match self.db.get_cf(&cf, &key)? {
            Some(bytes) => bincode::deserialize(&bytes)
                .map_err(|e| CodeStorageError::deserialization(e.to_string()))?,
            None => Vec::new(),
        };

        if !ids.contains(&entity_id) {
            ids.push(entity_id);
            let bytes = bincode::serialize(&ids)
                .map_err(|e| CodeStorageError::serialization(e.to_string()))?;
            self.db.put_cf(&cf, &key, bytes)?;
        }

        Ok(())
    }

    fn remove_from_signature_index(
        &self,
        signature: &str,
        entity_id: Uuid,
    ) -> CodeStorageResult<()> {
        let cf = self.cf_signature_index()?;
        let key = Self::hash_signature(signature);

        if let Some(bytes) = self.db.get_cf(&cf, &key)? {
            let mut ids: Vec<Uuid> = bincode::deserialize(&bytes)
                .map_err(|e| CodeStorageError::deserialization(e.to_string()))?;

            if let Some(pos) = ids.iter().position(|&id| id == entity_id) {
                ids.remove(pos);

                if ids.is_empty() {
                    self.db.delete_cf(&cf, &key)?;
                } else {
                    let bytes = bincode::serialize(&ids)
                        .map_err(|e| CodeStorageError::serialization(e.to_string()))?;
                    self.db.put_cf(&cf, &key, bytes)?;
                }
            }
        }

        Ok(())
    }

    // =========================================================================
    // Serialization Helpers
    // =========================================================================

    fn serialize_embedding(embedding: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(embedding.len() * 4);
        for &val in embedding {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    fn deserialize_embedding(bytes: &[u8]) -> CodeStorageResult<Vec<f32>> {
        if bytes.len() % 4 != 0 {
            return Err(CodeStorageError::deserialization(format!(
                "Invalid embedding bytes length: {}",
                bytes.len()
            )));
        }

        let mut embedding = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            embedding.push(f32::from_le_bytes(arr));
        }
        Ok(embedding)
    }

    fn hash_signature(signature: &str) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(signature.as_bytes());
        hasher.finalize().into()
    }

    fn glob_match(path: &str, pattern: &str) -> bool {
        // Simple glob matching - supports * and **
        let pattern_parts: Vec<&str> = pattern.split('/').collect();
        let path_parts: Vec<&str> = path.split('/').collect();

        Self::glob_match_parts(&path_parts, &pattern_parts)
    }

    /// Simple wildcard matching within a single path segment.
    /// Supports * as a wildcard matching any characters.
    fn wildcard_match(text: &str, pattern: &str) -> bool {
        let mut text_idx = 0;
        let mut pattern_idx = 0;
        let mut star_idx: Option<usize> = None;
        let mut match_idx = 0;

        let text_chars: Vec<char> = text.chars().collect();
        let pattern_chars: Vec<char> = pattern.chars().collect();

        while text_idx < text_chars.len() {
            if pattern_idx < pattern_chars.len()
                && (pattern_chars[pattern_idx] == '?'
                    || pattern_chars[pattern_idx] == text_chars[text_idx])
            {
                text_idx += 1;
                pattern_idx += 1;
            } else if pattern_idx < pattern_chars.len() && pattern_chars[pattern_idx] == '*' {
                star_idx = Some(pattern_idx);
                match_idx = text_idx;
                pattern_idx += 1;
            } else if let Some(star) = star_idx {
                pattern_idx = star + 1;
                match_idx += 1;
                text_idx = match_idx;
            } else {
                return false;
            }
        }

        while pattern_idx < pattern_chars.len() && pattern_chars[pattern_idx] == '*' {
            pattern_idx += 1;
        }

        pattern_idx == pattern_chars.len()
    }

    fn glob_match_parts(path: &[&str], pattern: &[&str]) -> bool {
        if pattern.is_empty() {
            return path.is_empty();
        }

        if path.is_empty() {
            return pattern.iter().all(|p| *p == "**");
        }

        match pattern[0] {
            "**" => {
                // Match zero or more path segments
                for i in 0..=path.len() {
                    if Self::glob_match_parts(&path[i..], &pattern[1..]) {
                        return true;
                    }
                }
                false
            }
            "*" => {
                // Match exactly one segment (any content)
                Self::glob_match_parts(&path[1..], &pattern[1..])
            }
            p if p.contains('*') => {
                // Match with wildcards in segment using simple pattern matching
                if Self::wildcard_match(path[0], p) {
                    Self::glob_match_parts(&path[1..], &pattern[1..])
                } else {
                    false
                }
            }
            p => {
                // Exact match
                path[0] == p && Self::glob_match_parts(&path[1..], &pattern[1..])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use context_graph_core::types::CodeEntityType;
    use tempfile::tempdir;

    fn create_test_entity(name: &str) -> CodeEntity {
        CodeEntity::new(
            CodeEntityType::Function,
            name.to_string(),
            format!("fn {}() {{}}", name),
            CodeLanguage::Rust,
            "/test/file.rs".to_string(),
            1,
            3,
        )
    }

    fn create_test_embedding() -> Vec<f32> {
        vec![0.1; E7_CODE_DIM]
    }

    #[test]
    fn test_store_and_get() {
        let dir = tempdir().unwrap();
        let store = CodeStore::open(dir.path()).unwrap();

        let entity = create_test_entity("my_function");
        let embedding = create_test_embedding();
        let id = entity.id;

        store.store(&entity, &embedding).unwrap();

        let retrieved = store.get(id).unwrap().unwrap();
        assert_eq!(retrieved.name, "my_function");
        assert_eq!(retrieved.entity_type, CodeEntityType::Function);

        let retrieved_embedding = store.get_embedding(id).unwrap().unwrap();
        assert_eq!(retrieved_embedding.len(), E7_CODE_DIM);
    }

    #[test]
    fn test_delete() {
        let dir = tempdir().unwrap();
        let store = CodeStore::open(dir.path()).unwrap();

        let entity = create_test_entity("to_delete");
        let embedding = create_test_embedding();
        let id = entity.id;

        store.store(&entity, &embedding).unwrap();
        assert!(store.exists(id).unwrap());

        store.delete(id).unwrap();
        assert!(!store.exists(id).unwrap());
    }

    #[test]
    fn test_get_by_file() {
        let dir = tempdir().unwrap();
        let store = CodeStore::open(dir.path()).unwrap();

        let entity1 = CodeEntity::new(
            CodeEntityType::Function,
            "func1".to_string(),
            "fn func1() {}".to_string(),
            CodeLanguage::Rust,
            "/test/myfile.rs".to_string(),
            1,
            3,
        );
        let entity2 = CodeEntity::new(
            CodeEntityType::Function,
            "func2".to_string(),
            "fn func2() {}".to_string(),
            CodeLanguage::Rust,
            "/test/myfile.rs".to_string(),
            5,
            7,
        );

        store.store(&entity1, &create_test_embedding()).unwrap();
        store.store(&entity2, &create_test_embedding()).unwrap();

        let entities = store.get_by_file("/test/myfile.rs").unwrap();
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn test_search_by_name() {
        let dir = tempdir().unwrap();
        let store = CodeStore::open(dir.path()).unwrap();

        let entity1 = create_test_entity("process_data");
        let entity2 = create_test_entity("process_items");
        let entity3 = create_test_entity("handle_event");

        store.store(&entity1, &create_test_embedding()).unwrap();
        store.store(&entity2, &create_test_embedding()).unwrap();
        store.store(&entity3, &create_test_embedding()).unwrap();

        let results = store.search_by_name("process", 10).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_delete_file() {
        let dir = tempdir().unwrap();
        let store = CodeStore::open(dir.path()).unwrap();

        let entity1 = CodeEntity::new(
            CodeEntityType::Function,
            "func1".to_string(),
            "fn func1() {}".to_string(),
            CodeLanguage::Rust,
            "/test/to_delete.rs".to_string(),
            1,
            3,
        );
        let entity2 = CodeEntity::new(
            CodeEntityType::Struct,
            "MyStruct".to_string(),
            "struct MyStruct {}".to_string(),
            CodeLanguage::Rust,
            "/test/to_delete.rs".to_string(),
            5,
            7,
        );

        store.store(&entity1, &create_test_embedding()).unwrap();
        store.store(&entity2, &create_test_embedding()).unwrap();

        let deleted = store.delete_file("/test/to_delete.rs").unwrap();
        assert_eq!(deleted, 2);

        let entities = store.get_by_file("/test/to_delete.rs").unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_stats() {
        let dir = tempdir().unwrap();
        let store = CodeStore::open(dir.path()).unwrap();

        let entity1 = create_test_entity("func1");
        let entity2 = create_test_entity("func2");

        store.store(&entity1, &create_test_embedding()).unwrap();
        store.store(&entity2, &create_test_embedding()).unwrap();

        let stats = store.get_stats().unwrap();
        assert_eq!(stats.total_entities, 2);
        assert_eq!(stats.total_files, 1);
    }

    #[test]
    fn test_invalid_embedding_dimension() {
        let dir = tempdir().unwrap();
        let store = CodeStore::open(dir.path()).unwrap();

        let entity = create_test_entity("test");
        let wrong_embedding = vec![0.1; 100]; // Wrong dimension

        let result = store.store(&entity, &wrong_embedding);
        assert!(matches!(
            result,
            Err(CodeStorageError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn test_glob_match() {
        assert!(CodeStore::glob_match("/foo/bar/baz.rs", "**/baz.rs"));
        assert!(CodeStore::glob_match("/foo/bar/baz.rs", "/foo/**/*.rs"));
        assert!(CodeStore::glob_match("/foo/bar/baz.rs", "/foo/*/baz.rs"));
        assert!(!CodeStore::glob_match("/foo/bar/baz.rs", "/other/**/*.rs"));
    }
}
