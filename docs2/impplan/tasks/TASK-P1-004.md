# TASK-P1-004: TextChunker Implementation

```xml
<task_spec id="TASK-P1-004" version="1.0">
<metadata>
  <title>TextChunker Implementation</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>9</sequence>
  <phase>1</phase>
  <implements>
    <requirement_ref>REQ-P1-03</requirement_ref>
    <requirement_ref>REQ-P1-04</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P1-001</task_ref>
    <task_ref>TASK-P1-002</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
</metadata>

<context>
Implements the TextChunker component that splits long text into overlapping
chunks suitable for embedding. Uses 200-word chunks with 50-word overlap
and attempts to preserve sentence boundaries.

This is the core text processing component used for MD file ingestion.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE1-MEMORY-CAPTURE.md#component_contracts</file>
  <file purpose="types">crates/context-graph-core/src/memory/chunker.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P1-002 complete (ChunkMetadata, TextChunk types exist)</check>
  <check>sha2 crate available for hashing</check>
</prerequisites>

<scope>
  <in_scope>
    - Implement TextChunker struct with configuration
    - Implement chunk_text method with sentence boundary detection
    - Implement find_sentence_boundary helper
    - Compute SHA256 hash of input content
    - Create ChunkMetadata for each chunk
    - Add unit tests for chunking logic
  </in_scope>
  <out_of_scope>
    - File I/O (MDFileWatcher handles this)
    - Embedding (Phase 2)
    - Storage (TASK-P1-005)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/memory/chunker.rs">
      pub struct TextChunker {
          chunk_size_words: usize,
          overlap_words: usize,
      }

      impl TextChunker {
          pub const CHUNK_SIZE_WORDS: usize = 200;
          pub const OVERLAP_WORDS: usize = 50;
          pub const MIN_CHUNK_WORDS: usize = 50;

          pub fn new(chunk_size: usize, overlap: usize) -> Result&lt;Self, ChunkerError&gt;;
          pub fn default_config() -> Self;
          pub fn chunk_text(&amp;self, content: &amp;str, file_path: &amp;str) -> Result&lt;Vec&lt;TextChunk&gt;, ChunkerError&gt;;
          fn find_sentence_boundary(&amp;self, words: &amp;[&amp;str], start: usize, end: usize) -> usize;
      }
    </signature>
  </signatures>

  <constraints>
    - Default chunk_size = 200 words, overlap = 50 words
    - chunk_size must be > overlap
    - chunk_size must be >= MIN_CHUNK_WORDS (50)
    - Empty content returns ChunkerError::EmptyContent
    - Sentence boundaries: '.', '!', '?'
    - Adjust to sentence boundary only within 20% of target end
    - SHA256 hash computed with sha2 crate
  </constraints>

  <verification>
    - Unit tests pass for various text lengths
    - Sentence boundary detection works correctly
    - Overlap is correct between consecutive chunks
  </verification>
</definition_of_done>

<pseudo_code>
// Add to crates/context-graph-core/src/memory/chunker.rs

use sha2::{Sha256, Digest};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ChunkerError {
    #[error("Content is empty")]
    EmptyContent,
    #[error("Invalid UTF-8 in content")]
    InvalidUtf8,
    #[error("Configuration error: {message}")]
    ConfigError { message: String },
}

pub struct TextChunker {
    chunk_size_words: usize,
    overlap_words: usize,
}

impl TextChunker {
    pub const CHUNK_SIZE_WORDS: usize = 200;
    pub const OVERLAP_WORDS: usize = 50;
    pub const MIN_CHUNK_WORDS: usize = 50;
    const SENTENCE_TERMINATORS: [char; 3] = ['.', '!', '?'];

    pub fn new(chunk_size: usize, overlap: usize) -> Result&lt;Self, ChunkerError&gt; {
        if chunk_size <= overlap {
            return Err(ChunkerError::ConfigError {
                message: "chunk_size must be > overlap".to_string()
            });
        }
        if chunk_size < Self::MIN_CHUNK_WORDS {
            return Err(ChunkerError::ConfigError {
                message: format!("chunk_size must be >= {}", Self::MIN_CHUNK_WORDS)
            });
        }
        Ok(Self { chunk_size_words: chunk_size, overlap_words: overlap })
    }

    pub fn default_config() -> Self {
        Self {
            chunk_size_words: Self::CHUNK_SIZE_WORDS,
            overlap_words: Self::OVERLAP_WORDS,
        }
    }

    pub fn chunk_text(&amp;self, content: &amp;str, file_path: &amp;str) -> Result&lt;Vec&lt;TextChunk&gt;, ChunkerError&gt; {
        if content.is_empty() {
            return Err(ChunkerError::EmptyContent);
        }

        let words: Vec&lt;&amp;str&gt; = content.split_whitespace().collect();
        if words.is_empty() {
            return Err(ChunkerError::EmptyContent);
        }

        // Compute file hash
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let hash = format!("{:x}", hasher.finalize());

        // If content fits in single chunk
        if words.len() <= self.chunk_size_words {
            let metadata = ChunkMetadata::new(
                file_path.to_string(),
                0, 1, 0, 0, hash
            );
            return Ok(vec![TextChunk::new(content.to_string(), metadata)]);
        }

        let mut chunks = Vec::new();
        let mut current_offset = 0;
        let mut chunk_index = 0;
        let mut char_offset = 0;

        // First pass: count total chunks
        let total_chunks = self.estimate_chunk_count(words.len());

        while current_offset < words.len() {
            let end_offset = std::cmp::min(current_offset + self.chunk_size_words, words.len());
            let adjusted_end = self.find_sentence_boundary(&amp;words, current_offset, end_offset);

            let chunk_words = &amp;words[current_offset..adjusted_end];
            let chunk_content = chunk_words.join(" ");

            let metadata = ChunkMetadata::new(
                file_path.to_string(),
                chunk_index,
                total_chunks,
                current_offset as u32,
                char_offset as u32,
                hash.clone()
            );

            char_offset += chunk_content.len() + 1; // +1 for space
            chunks.push(TextChunk::new(chunk_content, metadata));

            chunk_index += 1;
            current_offset = if adjusted_end >= words.len() {
                words.len()
            } else {
                adjusted_end - self.overlap_words
            };
        }

        Ok(chunks)
    }

    fn find_sentence_boundary(&amp;self, words: &amp;[&amp;str], start: usize, end: usize) -> usize {
        // Only adjust within 20% of target
        let search_start = end - (self.chunk_size_words / 5);
        let search_start = std::cmp::max(search_start, start);

        for i in (search_start..end).rev() {
            if let Some(word) = words.get(i) {
                if word.ends_with('.') || word.ends_with('!') || word.ends_with('?') {
                    return i + 1; // Include the word with terminator
                }
            }
        }
        end // No boundary found, use original end
    }

    fn estimate_chunk_count(&amp;self, total_words: usize) -> u32 {
        if total_words <= self.chunk_size_words {
            return 1;
        }
        let effective_chunk = self.chunk_size_words - self.overlap_words;
        ((total_words - self.overlap_words) / effective_chunk + 1) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_chunk() {
        let chunker = TextChunker::default_config();
        let content = "Short text with few words.";
        let chunks = chunker.chunk_text(content, "test.md").unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, content);
    }

    #[test]
    fn test_empty_content() {
        let chunker = TextChunker::default_config();
        let result = chunker.chunk_text("", "test.md");
        assert!(matches!(result, Err(ChunkerError::EmptyContent)));
    }

    #[test]
    fn test_overlap() {
        let chunker = TextChunker::new(10, 3).unwrap();
        // Create content with 25 words
        let content = (0..25).map(|i| format!("word{}", i)).collect::&lt;Vec&lt;_&gt;&gt;().join(" ");
        let chunks = chunker.chunk_text(&amp;content, "test.md").unwrap();
        assert!(chunks.len() > 1);
        // Verify overlap exists between consecutive chunks
    }
}
</pseudo_code>

<files_to_modify>
  <file path="crates/context-graph-core/src/memory/chunker.rs">Add TextChunker implementation</file>
  <file path="crates/context-graph-core/Cargo.toml">Add sha2 dependency if not present</file>
</files_to_modify>

<validation_criteria>
  <criterion>TextChunker::new validates configuration</criterion>
  <criterion>chunk_text returns error for empty content</criterion>
  <criterion>Single chunk returned for short content</criterion>
  <criterion>Multiple chunks have correct overlap</criterion>
  <criterion>Sentence boundary detection works</criterion>
  <criterion>SHA256 hash computed correctly</criterion>
  <criterion>All unit tests pass</criterion>
</validation_criteria>

<test_commands>
  <command description="Run chunker tests">cargo test --package context-graph-core chunker</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Add sha2 to Cargo.toml if needed
- [ ] Add thiserror dependency for error types
- [ ] Implement ChunkerError enum
- [ ] Implement TextChunker struct
- [ ] Implement new() with validation
- [ ] Implement chunk_text() with algorithm
- [ ] Implement find_sentence_boundary()
- [ ] Write unit tests
- [ ] Run tests to verify
- [ ] Proceed to TASK-P1-005
