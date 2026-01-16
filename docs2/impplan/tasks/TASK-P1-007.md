# TASK-P1-007: MemoryCaptureService

```xml
<task_spec id="TASK-P1-007" version="1.0">
<metadata>
  <title>MemoryCaptureService Implementation</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>12</sequence>
  <phase>1</phase>
  <implements>
    <requirement_ref>REQ-P1-01</requirement_ref>
    <requirement_ref>REQ-P1-02</requirement_ref>
    <requirement_ref>REQ-P1-03</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P1-004</task_ref>
    <task_ref>TASK-P1-005</task_ref>
    <task_ref>TASK-P1-006</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
</metadata>

<context>
Implements the central MemoryCaptureService that coordinates memory capture
from various sources. It orchestrates embedding via MultiArrayProvider and
storage via MemoryStore.

This is the primary interface used by CLI commands and hooks to capture memories.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE1-MEMORY-CAPTURE.md#component_contracts</file>
  <file purpose="memory_store">crates/context-graph-core/src/memory/store.rs</file>
  <file purpose="session_manager">crates/context-graph-core/src/memory/session.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P1-004 complete (TextChunker exists)</check>
  <check>TASK-P1-005 complete (MemoryStore exists)</check>
  <check>TASK-P1-006 complete (SessionManager exists)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create MemoryCaptureService struct
    - Implement capture_hook_description() method
    - Implement capture_claude_response() method
    - Implement capture_md_chunk() method
    - Implement internal capture_memory() method
    - Add CaptureError enum
    - Coordinate with embedding provider (interface)
  </in_scope>
  <out_of_scope>
    - MDFileWatcher (TASK-P1-008)
    - Actual embedding implementation (Phase 2)
    - CLI integration (Phase 6)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/memory/capture.rs">
      pub struct MemoryCaptureService {
          store: Arc&lt;MemoryStore&gt;,
          embedder: Arc&lt;dyn EmbeddingProvider&gt;,
      }

      impl MemoryCaptureService {
          pub fn new(store: Arc&lt;MemoryStore&gt;, embedder: Arc&lt;dyn EmbeddingProvider&gt;) -> Self;
          pub async fn capture_hook_description(&amp;self, content: String, hook_type: HookType, session_id: String, tool_name: Option&lt;String&gt;) -> Result&lt;Uuid, CaptureError&gt;;
          pub async fn capture_claude_response(&amp;self, content: String, response_type: ResponseType, session_id: String) -> Result&lt;Uuid, CaptureError&gt;;
          pub async fn capture_md_chunk(&amp;self, chunk: TextChunk, session_id: String) -> Result&lt;Uuid, CaptureError&gt;;
      }

      #[async_trait]
      pub trait EmbeddingProvider: Send + Sync {
          async fn embed_all(&amp;self, content: &amp;str) -> Result&lt;TeleologicalArray, EmbedderError&gt;;
      }
    </signature>
  </signatures>

  <constraints>
    - All capture methods fail fast on any error
    - Empty content returns CaptureError::EmptyContent
    - Embedding errors propagate immediately
    - Storage errors propagate immediately
    - No partial storage - all or nothing
  </constraints>

  <verification>
    - Hook description capture works
    - Claude response capture works
    - MD chunk capture works
    - Errors propagate correctly
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/memory/capture.rs

use std::sync::Arc;
use async_trait::async_trait;
use thiserror::Error;
use uuid::Uuid;
use chrono::Utc;

use super::{
    Memory, MemorySource, HookType, ResponseType,
    TextChunk, ChunkMetadata,
    store::{MemoryStore, StorageError},
};
use crate::embedding::TeleologicalArray;

#[derive(Debug, Error)]
pub enum CaptureError {
    #[error("Content is empty")]
    EmptyContent,
    #[error("Embedding failed: {0}")]
    EmbeddingFailed(#[from] EmbedderError),
    #[error("Storage failed: {0}")]
    StorageFailed(#[from] StorageError),
    #[error("Session not found: {session_id}")]
    SessionNotFound { session_id: String },
}

#[derive(Debug, Error)]
pub enum EmbedderError {
    #[error("Embedding service unavailable")]
    Unavailable,
    #[error("Embedding failed: {message}")]
    Failed { message: String },
}

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed_all(&amp;self, content: &amp;str) -> Result&lt;TeleologicalArray, EmbedderError&gt;;
}

pub struct MemoryCaptureService {
    store: Arc&lt;MemoryStore&gt;,
    embedder: Arc&lt;dyn EmbeddingProvider&gt;,
}

impl MemoryCaptureService {
    pub fn new(store: Arc&lt;MemoryStore&gt;, embedder: Arc&lt;dyn EmbeddingProvider&gt;) -> Self {
        Self { store, embedder }
    }

    pub async fn capture_hook_description(
        &amp;self,
        content: String,
        hook_type: HookType,
        session_id: String,
        tool_name: Option&lt;String&gt;,
    ) -> Result&lt;Uuid, CaptureError&gt; {
        if content.trim().is_empty() {
            return Err(CaptureError::EmptyContent);
        }

        let source = MemorySource::HookDescription { hook_type, tool_name };
        self.capture_memory(content, source, session_id, None).await
    }

    pub async fn capture_claude_response(
        &amp;self,
        content: String,
        response_type: ResponseType,
        session_id: String,
    ) -> Result&lt;Uuid, CaptureError&gt; {
        if content.trim().is_empty() {
            return Err(CaptureError::EmptyContent);
        }

        let source = MemorySource::ClaudeResponse { response_type };
        self.capture_memory(content, source, session_id, None).await
    }

    pub async fn capture_md_chunk(
        &amp;self,
        chunk: TextChunk,
        session_id: String,
    ) -> Result&lt;Uuid, CaptureError&gt; {
        if chunk.content.trim().is_empty() {
            return Err(CaptureError::EmptyContent);
        }

        let source = MemorySource::MDFileChunk {
            file_path: chunk.metadata.file_path.clone(),
            chunk_index: chunk.metadata.chunk_index,
            total_chunks: chunk.metadata.total_chunks,
        };

        self.capture_memory(chunk.content, source, session_id, Some(chunk.metadata)).await
    }

    async fn capture_memory(
        &amp;self,
        content: String,
        source: MemorySource,
        session_id: String,
        chunk_metadata: Option&lt;ChunkMetadata&gt;,
    ) -> Result&lt;Uuid, CaptureError&gt; {
        // Generate embedding (fail fast if error)
        let teleological_array = self.embedder.embed_all(&amp;content).await?;

        // Create memory
        let word_count = content.split_whitespace().count() as u32;
        let memory = Memory {
            id: Uuid::new_v4(),
            content,
            source,
            created_at: Utc::now(),
            session_id,
            teleological_array,
            chunk_metadata,
            word_count,
        };

        let id = memory.id;

        // Store (fail fast if error)
        self.store.store(memory).await?;

        Ok(id)
    }
}

// Placeholder implementation for testing
pub struct MockEmbeddingProvider;

#[async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed_all(&amp;self, _content: &amp;str) -> Result&lt;TeleologicalArray, EmbedderError&gt; {
        // Return placeholder array
        Ok(vec![0.0; 768]) // Placeholder until Phase 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_capture_hook_description() {
        let dir = tempdir().unwrap();
        let store = Arc::new(MemoryStore::new(dir.path()).unwrap());
        let embedder = Arc::new(MockEmbeddingProvider);
        let service = MemoryCaptureService::new(store, embedder);

        let result = service.capture_hook_description(
            "Test description".to_string(),
            HookType::SessionStart,
            "test-session".to_string(),
            None,
        ).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_capture_empty_content() {
        let dir = tempdir().unwrap();
        let store = Arc::new(MemoryStore::new(dir.path()).unwrap());
        let embedder = Arc::new(MockEmbeddingProvider);
        let service = MemoryCaptureService::new(store, embedder);

        let result = service.capture_hook_description(
            "".to_string(),
            HookType::SessionStart,
            "test-session".to_string(),
            None,
        ).await;

        assert!(matches!(result, Err(CaptureError::EmptyContent)));
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/memory/capture.rs">MemoryCaptureService implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/memory/mod.rs">Add pub mod capture and re-export</file>
  <file path="crates/context-graph-core/Cargo.toml">Add async-trait dependency if not present</file>
</files_to_modify>

<validation_criteria>
  <criterion>capture_hook_description creates memory with correct source</criterion>
  <criterion>capture_claude_response creates memory with correct source</criterion>
  <criterion>capture_md_chunk includes chunk metadata</criterion>
  <criterion>Empty content returns error</criterion>
  <criterion>Embedding errors propagate</criterion>
  <criterion>Storage errors propagate</criterion>
  <criterion>Memory ID returned on success</criterion>
</validation_criteria>

<test_commands>
  <command description="Run capture tests">cargo test --package context-graph-core capture</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Add async-trait to Cargo.toml
- [ ] Create capture.rs in memory directory
- [ ] Implement CaptureError and EmbedderError enums
- [ ] Implement EmbeddingProvider trait
- [ ] Implement MemoryCaptureService struct
- [ ] Implement capture_hook_description() method
- [ ] Implement capture_claude_response() method
- [ ] Implement capture_md_chunk() method
- [ ] Create MockEmbeddingProvider for testing
- [ ] Write unit tests
- [ ] Run tests to verify
- [ ] Proceed to TASK-P1-008
