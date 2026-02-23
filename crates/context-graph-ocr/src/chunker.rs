//! Document Chunker - Wraps context-graph-core TextChunker
//!
//! This module provides document chunking capabilities using the core TextChunker
//! with configuration for OCR results. Target chunk size is 2000 characters
//! with section-aware splitting.

use crate::types::DocumentChunk;
use crate::OcrError;
use std::path::Path;

/// Document chunking configuration
#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    /// Target chunk size in characters
    pub target_chunk_size: usize,
    /// Chunk overlap in characters
    pub overlap: usize,
    /// Minimum chunk size
    pub min_chunk_size: usize,
    /// Preserve sentence boundaries
    pub preserve_sentences: bool,
    /// Detect sections
    pub detect_sections: bool,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            target_chunk_size: 2000,
            overlap: 200,
            min_chunk_size: 100,
            preserve_sentences: true,
            detect_sections: true,
        }
    }
}

/// DocumentChunker - Chunks OCR text for embedding pipeline
pub struct DocumentChunker {
    config: ChunkerConfig,
}

impl Default for DocumentChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentChunker {
    /// Create a new DocumentChunker with default config
    pub fn new() -> Self {
        Self {
            config: ChunkerConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: ChunkerConfig) -> Self {
        Self { config }
    }

    /// Chunk text content from OCR result
    pub fn chunk_text(
        &self,
        text: &str,
        document_id: &str,
        source_path: Option<&Path>,
    ) -> Result<Vec<DocumentChunk>, OcrError> {
        if text.is_empty() {
            return Err(OcrError::ProcessingError("Empty text content".to_string()));
        }

        let mut chunks = Vec::new();

        // First try section-aware splitting if enabled
        if self.config.detect_sections {
            let sections = self.split_by_sections(text);
            for section in sections {
                let section_chunks = self.chunk_section(
                    &section.content,
                    document_id,
                    source_path,
                    section.name.as_deref(),
                    chunks.len() as u32,
                );
                chunks.extend(section_chunks);
            }
        } else {
            // Direct chunking
            chunks = self.chunk_raw(text, document_id, source_path, None, 0);
        }

        if chunks.is_empty() {
            return Err(OcrError::ProcessingError(
                "Failed to produce any chunks".to_string(),
            ));
        }

        // Update total_chunks for all
        let total = chunks.len() as u32;
        for chunk in &mut chunks {
            chunk.total_chunks = total;
        }

        Ok(chunks)
    }

    /// Split text by common section markers
    fn split_by_sections(&self, text: &str) -> Vec<Section> {
        let mut sections = Vec::new();

        // Common section markers
        let section_patterns = [
            "\n# ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\nChapter ",
            "\nSection ",
        ];

        let mut current_pos = 0;
        let text_chars: Vec<char> = text.chars().collect();

        for pattern in &section_patterns {
            let pattern_chars: Vec<char> = pattern.chars().collect();

            // Simple search for pattern
            let mut search_pos = 0;
            while search_pos < text_chars.len() {
                let remaining = &text_chars[search_pos..];
                if remaining.starts_with(&pattern_chars) {
                    if search_pos > current_pos {
                        let section_text: String = text_chars[current_pos..search_pos].iter().collect();
                        if !section_text.trim().is_empty() {
                            sections.push(Section {
                                name: None,
                                content: section_text,
                            });
                        }
                    }
                    // Start new section
                    current_pos = search_pos + pattern_chars.len();
                    search_pos = current_pos;

                    // Extract section name (up to next newline or 50 chars)
                    let end_pos = text_chars[current_pos..]
                        .iter()
                        .position(|&c| c == '\n')
                        .map(|p| current_pos + p)
                        .unwrap_or((current_pos + 50).min(text_chars.len()));
                    let name: String = text_chars[current_pos..end_pos].iter().collect();
                    let name = name.trim().to_string();

                    if !name.is_empty() {
                        if let Some(last) = sections.last_mut() {
                            last.name = Some(name.clone());
                        } else {
                            sections.push(Section {
                                name: Some(name),
                                content: String::new(),
                            });
                        }
                    }
                } else {
                    search_pos += 1;
                }
            }
        }

        // Add remaining text as last section
        if current_pos < text_chars.len() {
            let remaining: String = text_chars[current_pos..].iter().collect();
            if !remaining.trim().is_empty() {
                sections.push(Section {
                    name: None,
                    content: remaining,
                });
            }
        }

        // If no sections found, treat whole text as one section
        if sections.is_empty() {
            sections.push(Section {
                name: None,
                content: text.to_string(),
            });
        }

        sections
    }

    /// Chunk a section of text
    fn chunk_section(
        &self,
        text: &str,
        document_id: &str,
        source_path: Option<&Path>,
        section_name: Option<&str>,
        start_index: u32,
    ) -> Vec<DocumentChunk> {
        if text.len() <= self.config.target_chunk_size {
            // Text fits in single chunk
            let mut chunk = DocumentChunk::new(
                document_id.to_string(),
                text.to_string(),
                start_index,
                1, // Will be updated later
                0,
            );
            chunk.section = section_name.map(String::from);
            if let Some(path) = source_path {
                chunk.section = Some(
                    path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string(),
                );
            }
            return vec![chunk];
        }

        self.chunk_raw(text, document_id, source_path, section_name, start_index)
    }

    /// Chunk text without section awareness
    fn chunk_raw(
        &self,
        text: &str,
        document_id: &str,
        _source_path: Option<&Path>,
        section_name: Option<&str>,
        start_index: u32,
    ) -> Vec<DocumentChunk> {
        let mut chunks = Vec::new();
        let text_len = text.len();

        if text_len <= self.config.target_chunk_size {
            let mut chunk = DocumentChunk::new(
                document_id.to_string(),
                text.to_string(),
                start_index,
                1,
                0,
            );
            chunk.section = section_name.map(String::from);
            return vec![chunk];
        }

        // Split into chunks
        let mut char_offset = 0;
        let mut chunk_index = start_index;

        while char_offset < text_len {
            let remaining = text_len - char_offset;

            // Determine chunk size
            let chunk_size = if remaining > self.config.target_chunk_size {
                // Try to find a good break point
                let end_pos = char_offset + self.config.target_chunk_size;

                // Look for sentence boundary within last 200 chars
                let search_start = (end_pos - 200).max(char_offset);
                let search_range = &text[search_start..end_pos];

                // Find last period, question mark, or newline
                let break_pos = search_range
                    .rfind(|c| c == '.' || c == '?' || c == '!' || c == '\n')
                    .map(|p| search_start + p + 1)
                    .unwrap_or(end_pos);

                break_pos - char_offset
            } else {
                remaining
            };

            // Ensure minimum chunk size
            let final_size = if chunk_size < self.config.min_chunk_size && remaining > self.config.min_chunk_size {
                self.config.min_chunk_size
            } else {
                chunk_size
            };

            let chunk_text = &text[char_offset..(char_offset + final_size).min(text_len)];
            let mut chunk = DocumentChunk::new(
                document_id.to_string(),
                chunk_text.to_string(),
                chunk_index,
                0, // Will be updated later
                char_offset,
            );
            chunk.section = section_name.map(String::from);

            chunks.push(chunk);

            // Move to next chunk with overlap
            char_offset += final_size.saturating_sub(self.config.overlap);
            chunk_index += 1;

            // Safety check for infinite loop
            if chunk_index - start_index > 1000 {
                break;
            }
        }

        chunks
    }

    /// Get chunk statistics
    pub fn get_stats(&self, chunks: &[DocumentChunk]) -> ChunkStats {
        if chunks.is_empty() {
            return ChunkStats::default();
        }

        let total_chars: usize = chunks.iter().map(|c| c.content.len()).sum();
        let total_words: usize = chunks
            .iter()
            .map(|c| c.content.split_whitespace().count())
            .sum();

        ChunkStats {
            total_chunks: chunks.len() as u32,
            total_chars,
            total_words,
            avg_chunk_size: total_chars / chunks.len(),
        }
    }
}

/// Section with name and content
#[derive(Debug)]
struct Section {
    name: Option<String>,
    content: String,
}

/// Statistics about chunking result
#[derive(Debug, Default)]
pub struct ChunkStats {
    pub total_chunks: u32,
    pub total_chars: usize,
    pub total_words: usize,
    pub avg_chunk_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunker_creation() {
        let chunker = DocumentChunker::new();
        assert_eq!(chunker.config.target_chunk_size, 2000);
    }

    #[test]
    fn test_chunker_default() {
        let chunker = DocumentChunker::default();
        assert_eq!(chunker.config.overlap, 200);
    }

    #[test]
    fn test_chunk_small_text() {
        let chunker = DocumentChunker::new();
        let text = "Hello, world! This is a short text.";
        let chunks = chunker.chunk_text(text, "doc-123", None).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, text);
    }

    #[test]
    fn test_chunk_large_text() {
        let chunker = DocumentChunker::new();
        // Create text larger than target chunk size
        let text = "a".repeat(5000);
        let chunks = chunker.chunk_text(&text, "doc-123", None).unwrap();
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_chunk_stats() {
        let chunker = DocumentChunker::new();
        let text = "Hello, world! This is a test document.";
        let chunks = chunker.chunk_text(text, "doc-123", None).unwrap();
        let stats = chunker.get_stats(&chunks);

        assert_eq!(stats.total_chunks, 1);
        assert!(stats.total_chars > 0);
        assert!(stats.total_words > 0);
    }
}
