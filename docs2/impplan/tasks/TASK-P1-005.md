# TASK-P1-005: MemoryStore with RocksDB

```xml
<task_spec id="TASK-P1-005" version="1.0">
<metadata>
  <title>MemoryStore with RocksDB</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>10</sequence>
  <phase>1</phase>
  <implements>
    <requirement_ref>REQ-P1-07</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P1-001</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
</metadata>

<context>
Implements the MemoryStore component for persisting Memory structs to RocksDB.
Provides basic CRUD operations and indexing by session_id and source type.

Uses RocksDB column families for organized storage and efficient querying.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE1-MEMORY-CAPTURE.md#component_contracts</file>
  <file purpose="database_schema">docs2/impplan/technical/TECH-PHASE1-MEMORY-CAPTURE.md#database_schema</file>
  <file purpose="memory_type">crates/context-graph-core/src/memory/mod.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P1-001 complete (Memory struct exists)</check>
  <check>rocksdb crate available</check>
  <check>bincode crate available for serialization</check>
</prerequisites>

<scope>
  <in_scope>
    - Create MemoryStore struct
    - Initialize RocksDB with column families
    - Implement store() method
    - Implement get() method
    - Implement get_by_session() method
    - Implement count() method
    - Add StorageError enum
    - Create session_memories index
  </in_scope>
  <out_of_scope>
    - Session storage (TASK-P1-006)
    - File hash tracking (part of watcher)
    - Embedding storage (handled via Memory.teleological_array)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/memory/store.rs">
      pub struct MemoryStore {
          db: Arc&lt;DB&gt;,
      }

      impl MemoryStore {
          pub fn new(path: &amp;Path) -> Result&lt;Self, StorageError&gt;;
          pub async fn store(&amp;self, memory: Memory) -> Result&lt;(), StorageError&gt;;
          pub async fn get(&amp;self, id: Uuid) -> Result&lt;Option&lt;Memory&gt;, StorageError&gt;;
          pub async fn get_by_session(&amp;self, session_id: &amp;str) -> Result&lt;Vec&lt;Memory&gt;, StorageError&gt;;
          pub async fn count(&amp;self) -> Result&lt;u64, StorageError&gt;;
      }
    </signature>
  </signatures>

  <constraints>
    - Use bincode for serialization (fast, compact)
    - Column families: memories, session_index
    - Key for memories: UUID bytes (16 bytes)
    - Fail fast on any write error - no partial writes
    - All operations are atomic within single memory
  </constraints>

  <verification>
    - Store and retrieve round-trips correctly
    - Session index queries work
    - Count returns accurate number
    - Errors propagate correctly
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/memory/store.rs

use rocksdb::{DB, Options, ColumnFamilyDescriptor};
use std::sync::Arc;
use std::path::Path;
use bincode;
use thiserror::Error;
use uuid::Uuid;
use super::Memory;

const CF_MEMORIES: &amp;str = "memories";
const CF_SESSION_INDEX: &amp;str = "session_index";

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Serialization failed: {0}")]
    SerializationFailed(#[from] bincode::Error),
    #[error("Database write failed: {0}")]
    WriteFailed(#[from] rocksdb::Error),
    #[error("Database read failed: {source}")]
    ReadFailed { source: rocksdb::Error },
    #[error("Database initialization failed: {0}")]
    InitFailed(String),
}

pub struct MemoryStore {
    db: Arc&lt;DB&gt;,
}

impl MemoryStore {
    pub fn new(path: &amp;Path) -> Result&lt;Self, StorageError&gt; {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let cf_descriptors = vec![
            ColumnFamilyDescriptor::new(CF_MEMORIES, Options::default()),
            ColumnFamilyDescriptor::new(CF_SESSION_INDEX, Options::default()),
        ];

        let db = DB::open_cf_descriptors(&amp;opts, path, cf_descriptors)
            .map_err(|e| StorageError::InitFailed(e.to_string()))?;

        Ok(Self { db: Arc::new(db) })
    }

    pub async fn store(&amp;self, memory: Memory) -> Result&lt;(), StorageError&gt; {
        let key = memory.id.as_bytes().to_vec();
        let value = bincode::serialize(&amp;memory)?;

        // Store in memories CF
        let cf_memories = self.db.cf_handle(CF_MEMORIES)
            .ok_or_else(|| StorageError::InitFailed("memories CF not found".into()))?;
        self.db.put_cf(&amp;cf_memories, &amp;key, &amp;value)?;

        // Update session index
        let cf_index = self.db.cf_handle(CF_SESSION_INDEX)
            .ok_or_else(|| StorageError::InitFailed("session_index CF not found".into()))?;

        // Get existing session memory list
        let index_key = memory.session_id.as_bytes();
        let mut memory_ids: Vec&lt;Uuid&gt; = match self.db.get_cf(&amp;cf_index, index_key)? {
            Some(data) => bincode::deserialize(&amp;data).unwrap_or_default(),
            None => Vec::new(),
        };
        memory_ids.push(memory.id);

        let index_value = bincode::serialize(&amp;memory_ids)?;
        self.db.put_cf(&amp;cf_index, index_key, &amp;index_value)?;

        Ok(())
    }

    pub async fn get(&amp;self, id: Uuid) -> Result&lt;Option&lt;Memory&gt;, StorageError&gt; {
        let cf = self.db.cf_handle(CF_MEMORIES)
            .ok_or_else(|| StorageError::InitFailed("memories CF not found".into()))?;

        match self.db.get_cf(&amp;cf, id.as_bytes())? {
            Some(data) => {
                let memory: Memory = bincode::deserialize(&amp;data)?;
                Ok(Some(memory))
            }
            None => Ok(None),
        }
    }

    pub async fn get_by_session(&amp;self, session_id: &amp;str) -> Result&lt;Vec&lt;Memory&gt;, StorageError&gt; {
        let cf_index = self.db.cf_handle(CF_SESSION_INDEX)
            .ok_or_else(|| StorageError::InitFailed("session_index CF not found".into()))?;

        let memory_ids: Vec&lt;Uuid&gt; = match self.db.get_cf(&amp;cf_index, session_id.as_bytes())? {
            Some(data) => bincode::deserialize(&amp;data).unwrap_or_default(),
            None => return Ok(Vec::new()),
        };

        let mut memories = Vec::with_capacity(memory_ids.len());
        for id in memory_ids {
            if let Some(memory) = self.get(id).await? {
                memories.push(memory);
            }
        }

        Ok(memories)
    }

    pub async fn count(&amp;self) -> Result&lt;u64, StorageError&gt; {
        let cf = self.db.cf_handle(CF_MEMORIES)
            .ok_or_else(|| StorageError::InitFailed("memories CF not found".into()))?;

        let mut count = 0u64;
        let iter = self.db.iterator_cf(&amp;cf, rocksdb::IteratorMode::Start);
        for _ in iter {
            count += 1;
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_store_and_get() {
        let dir = tempdir().unwrap();
        let store = MemoryStore::new(dir.path()).unwrap();

        // Create test memory (requires placeholder TeleologicalArray)
        let memory = Memory {
            id: Uuid::new_v4(),
            content: "Test content".to_string(),
            source: MemorySource::HookDescription {
                hook_type: HookType::SessionStart,
                tool_name: None
            },
            created_at: Utc::now(),
            session_id: "test-session".to_string(),
            teleological_array: vec![], // Placeholder
            chunk_metadata: None,
            word_count: 2,
        };

        store.store(memory.clone()).await.unwrap();

        let retrieved = store.get(memory.id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, memory.content);
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/memory/store.rs">MemoryStore implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/memory/mod.rs">Add pub mod store and re-export</file>
  <file path="crates/context-graph-core/Cargo.toml">Add rocksdb, bincode dependencies if not present</file>
</files_to_modify>

<validation_criteria>
  <criterion>MemoryStore initializes RocksDB correctly</criterion>
  <criterion>store() persists memory and updates index</criterion>
  <criterion>get() retrieves memory by ID</criterion>
  <criterion>get_by_session() returns all session memories</criterion>
  <criterion>count() returns accurate count</criterion>
  <criterion>Errors propagate correctly</criterion>
  <criterion>Integration tests pass with temp directory</criterion>
</validation_criteria>

<test_commands>
  <command description="Run store tests">cargo test --package context-graph-core store</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Add rocksdb, bincode to Cargo.toml if needed
- [ ] Create store.rs in memory directory
- [ ] Implement StorageError enum
- [ ] Implement MemoryStore struct
- [ ] Implement new() with CF initialization
- [ ] Implement store() with indexing
- [ ] Implement get() method
- [ ] Implement get_by_session() method
- [ ] Implement count() method
- [ ] Write integration tests
- [ ] Run tests to verify
- [ ] Proceed to TASK-P1-006
