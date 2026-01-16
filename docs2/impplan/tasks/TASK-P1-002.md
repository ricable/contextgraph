# TASK-P1-002: ChunkMetadata and TextChunk Types

```xml
<task_spec id="TASK-P1-002" version="1.0">
<metadata>
  <title>ChunkMetadata and TextChunk Types</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>7</sequence>
  <phase>1</phase>
  <implements>
    <requirement_ref>REQ-P1-03</requirement_ref>
    <requirement_ref>REQ-P1-04</requirement_ref>
  </implements>
  <depends_on>
    <!-- Can run parallel to P1-001 -->
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<context>
Foundation task defining data types for text chunking. ChunkMetadata tracks
the origin and position of a chunk within its source file. TextChunk combines
content with its metadata.

These types are used by the TextChunker (P1-004) and referenced by Memory struct.
</context>

<input_context_files>
  <file purpose="data_models">docs2/impplan/technical/TECH-PHASE1-MEMORY-CAPTURE.md#data_models</file>
</input_context_files>

<prerequisites>
  <check>crates/context-graph-core/src/memory directory exists</check>
</prerequisites>

<scope>
  <in_scope>
    - Create ChunkMetadata struct
    - Create TextChunk struct
    - Derive necessary traits
    - Export from chunker module
  </in_scope>
  <out_of_scope>
    - TextChunker implementation (TASK-P1-004)
    - Chunking algorithm
    - File reading
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/memory/chunker.rs">
      #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
      pub struct ChunkMetadata {
          pub file_path: String,
          pub chunk_index: u32,
          pub total_chunks: u32,
          pub word_offset: u32,
          pub char_offset: u32,
          pub original_file_hash: String,
      }

      #[derive(Debug, Clone)]
      pub struct TextChunk {
          pub content: String,
          pub word_count: u32,
          pub metadata: ChunkMetadata,
      }
    </signature>
  </signatures>

  <constraints>
    - ChunkMetadata must be Serialize/Deserialize for storage
    - TextChunk is transient (not stored directly), no Serialize needed
    - original_file_hash is SHA256 hex string (64 chars)
    - word_offset and char_offset are 0-indexed
  </constraints>

  <verification>
    - cargo check passes
    - Types exported from memory module
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/memory/chunker.rs

use serde::{Serialize, Deserialize};

/// Metadata about a text chunk's origin and position
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkMetadata {
    /// Path to the source file
    pub file_path: String,
    /// 0-indexed position of this chunk in the file
    pub chunk_index: u32,
    /// Total number of chunks from this file
    pub total_chunks: u32,
    /// Word offset from start of file
    pub word_offset: u32,
    /// Character offset from start of file
    pub char_offset: u32,
    /// SHA256 hash of original file content
    pub original_file_hash: String,
}

/// A chunk of text with associated metadata
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// The chunk content
    pub content: String,
    /// Number of words in content
    pub word_count: u32,
    /// Metadata about chunk origin
    pub metadata: ChunkMetadata,
}

impl TextChunk {
    pub fn new(content: String, metadata: ChunkMetadata) -> Self {
        let word_count = content.split_whitespace().count() as u32;
        Self {
            content,
            word_count,
            metadata,
        }
    }
}

impl ChunkMetadata {
    pub fn new(
        file_path: String,
        chunk_index: u32,
        total_chunks: u32,
        word_offset: u32,
        char_offset: u32,
        original_file_hash: String,
    ) -> Self {
        Self {
            file_path,
            chunk_index,
            total_chunks,
            word_offset,
            char_offset,
            original_file_hash,
        }
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/memory/chunker.rs">ChunkMetadata and TextChunk types</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/memory/mod.rs">Add pub mod chunker and re-export types</file>
</files_to_modify>

<validation_criteria>
  <criterion>ChunkMetadata has all 6 fields per spec</criterion>
  <criterion>TextChunk has content, word_count, metadata fields</criterion>
  <criterion>ChunkMetadata derives Serialize, Deserialize</criterion>
  <criterion>Types exported from memory module</criterion>
  <criterion>cargo check passes</criterion>
</validation_criteria>

<test_commands>
  <command description="Check compilation">cargo check --package context-graph-core</command>
  <command description="Verify exports">grep -r "pub struct ChunkMetadata" crates/context-graph-core/src/memory/</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Create chunker.rs in memory directory
- [ ] Implement ChunkMetadata struct
- [ ] Implement TextChunk struct
- [ ] Add constructors for both types
- [ ] Update mod.rs with module declaration and re-exports
- [ ] Verify with cargo check
- [ ] Proceed to TASK-P1-003
