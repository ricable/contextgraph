//! OCR Types - RVF-Aligned Type Definitions
//!
//! This module defines types that map to RVF segments:
//! - OcrResult -> OCR_SEG (0x34)
//! - DocumentChunk -> CHUNK_SEG (0x32)
//! - ProvenanceEntry -> WITNESS_SEG (0x0B)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// OCR processing mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OcrMode {
    /// Fast mode - quickest results, lower accuracy
    Fast,
    /// Balanced mode - compromise between speed and accuracy
    Balanced,
    /// Accurate mode - highest accuracy, slower processing
    Accurate,
}

impl Default for OcrMode {
    fn default() -> Self {
        Self::Accurate
    }
}

impl OcrMode {
    /// Convert mode to string for API calls
    pub fn as_str(&self) -> &'static str {
        match self {
            OcrMode::Fast => "fast",
            OcrMode::Balanced => "balanced",
            OcrMode::Accurate => "accurate",
        }
    }
}

/// Operation type for provenance entries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProvenanceOp {
    /// New document created
    Create,
    /// Document updated
    Update,
    /// Document deleted
    Delete,
}

impl ProvenanceOp {
    /// Convert to byte for WITNESS_SEG encoding
    pub fn to_byte(&self) -> u8 {
        match self {
            ProvenanceOp::Create => 0x01,
            ProvenanceOp::Update => 0x02,
            ProvenanceOp::Delete => 0x03,
        }
    }

    /// Convert from byte
    pub fn from_byte(byte: u8) -> Self {
        match byte {
            0x01 => ProvenanceOp::Create,
            0x02 => ProvenanceOp::Update,
            0x03 => ProvenanceOp::Delete,
            _ => ProvenanceOp::Create,
        }
    }
}

/// Page offset information for OCR results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageOffset {
    /// Page number (1-indexed)
    pub page: u32,
    /// Character start position in the full text
    pub char_start: usize,
    /// Character end position in the full text
    pub char_end: usize,
}

/// OCR Result - maps to RVF OCR_SEG (0x34)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrResult {
    /// Unique identifier for this OCR result
    pub id: String,
    /// Associated provenance ID
    pub provenance_id: String,
    /// Document identifier
    pub document_id: String,
    /// Extracted text content
    pub extracted_text: String,
    /// Length of extracted text
    pub text_length: usize,
    /// Datalab request ID
    pub datalab_request_id: Option<String>,
    /// OCR mode used
    pub mode: OcrMode,
    /// Parse quality score (0.0 - 1.0)
    pub parse_quality_score: Option<f64>,
    /// Number of pages processed
    pub page_count: u32,
    /// Cost in cents (if available)
    pub cost_cents: Option<f64>,
    /// SHA-256 content hash
    pub content_hash: String,
    /// Processing start timestamp
    pub processing_started_at: DateTime<Utc>,
    /// Processing completion timestamp
    pub processing_completed_at: DateTime<Utc>,
    /// Processing duration in milliseconds
    pub processing_duration_ms: u64,
    /// Page offsets for text positioning
    pub page_offsets: Vec<PageOffset>,
    /// Extracted images (base64 encoded)
    pub images: Option<Vec<ExtractedImage>>,
    /// JSON blocks from Datalab
    pub json_blocks: Option<serde_json::Value>,
    /// Document metadata
    pub metadata: Option<DocumentMetadata>,
}

/// Extracted image from OCR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedImage {
    /// Image filename
    pub filename: String,
    /// Base64 encoded image data
    pub data: String,
    /// Image format (png, jpg, etc.)
    pub format: String,
    /// Page number where image was found
    pub page: u32,
}

/// Document metadata from OCR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Document title
    pub title: Option<String>,
    /// Document author
    pub author: Option<String>,
    /// Document subject
    pub subject: Option<String>,
    /// Page statistics
    pub page_stats: Option<serde_json::Value>,
    /// Block counts
    pub block_counts: Option<serde_json::Value>,
}

/// Document Chunk - maps to RVF CHUNK_SEG (0x32)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    /// Unique chunk identifier
    pub id: String,
    /// Parent document ID
    pub document_id: String,
    /// Chunk content (text)
    pub content: String,
    /// Chunk index in document
    pub chunk_index: u32,
    /// Total chunks in document
    pub total_chunks: u32,
    /// Character offset in original document
    pub char_offset: usize,
    /// Character end position
    pub char_end: usize,
    /// Page number (1-indexed)
    pub page: Option<u32>,
    /// Section name if detected
    pub section: Option<String>,
    /// Content hash for integrity
    pub content_hash: String,
    /// Timestamp of chunking
    pub created_at: DateTime<Utc>,
}

impl DocumentChunk {
    /// Create a new DocumentChunk
    pub fn new(
        document_id: String,
        content: String,
        chunk_index: u32,
        total_chunks: u32,
        char_offset: usize,
    ) -> Self {
        let char_end = char_offset + content.len();
        let content_hash = compute_hash(&content);

        Self {
            id: Uuid::new_v4().to_string(),
            document_id,
            content,
            chunk_index,
            total_chunks,
            char_offset,
            char_end,
            page: None,
            section: None,
            content_hash,
            created_at: Utc::now(),
        }
    }
}

/// Provenance Entry - maps to RVF WITNESS_SEG (0x0B)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceEntry {
    /// Unique entry ID (16 bytes hex encoded)
    pub id: String,
    /// Document ID this entry belongs to
    pub document_id: String,
    /// SHA-256 content hash
    pub content_hash: String,
    /// Hash of parent entry (for chain linking)
    pub parent_hash: Option<String>,
    /// Unix timestamp in milliseconds
    pub timestamp: i64,
    /// Operation type
    pub operation: ProvenanceOp,
    /// Source file path
    pub source_path: String,
    /// Number of chunks created
    pub chunk_count: u32,
    /// Embedding model used
    pub embedding_model: String,
}

impl ProvenanceEntry {
    /// Create a new provenance entry
    pub fn new(
        document_id: String,
        content_hash: String,
        operation: ProvenanceOp,
        source_path: String,
        chunk_count: u32,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            document_id,
            content_hash,
            parent_hash: None,
            timestamp: Utc::now().timestamp_millis(),
            operation,
            source_path,
            chunk_count,
            embedding_model: "semantic".to_string(), // Default, can be configured
        }
    }

    /// Set parent hash for chain linking
    pub fn with_parent(mut self, parent_hash: String) -> Self {
        self.parent_hash = Some(parent_hash);
        self
    }
}

/// Complete provenance chain for a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceChain {
    /// Document ID
    pub document_id: String,
    /// All entries in chronological order
    pub entries: Vec<ProvenanceEntry>,
    /// Whether chain has been verified
    pub verified: bool,
    /// Root hash of the chain (first entry's content hash)
    pub root_hash: String,
}

/// Result of chain verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether the entire chain is valid
    pub valid: bool,
    /// Number of entries checked
    pub entries_checked: u32,
    /// List of error messages
    pub errors: Vec<String>,
    /// IDs of entries that failed verification
    pub tampered_entries: Vec<String>,
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Compute SHA-256 hash of content
pub fn compute_hash(content: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hex::encode(hasher.finalize())
}

/// Compute content hash with metadata
pub fn compute_content_hash(content: &str, metadata: &serde_json::Value) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hasher.update(metadata.to_string().as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ocr_mode_as_str() {
        assert_eq!(OcrMode::Fast.as_str(), "fast");
        assert_eq!(OcrMode::Balanced.as_str(), "balanced");
        assert_eq!(OcrMode::Accurate.as_str(), "accurate");
    }

    #[test]
    fn test_provenance_op_byte_conversion() {
        assert_eq!(ProvenanceOp::Create.to_byte(), 0x01);
        assert_eq!(ProvenanceOp::Update.to_byte(), 0x02);
        assert_eq!(ProvenanceOp::Delete.to_byte(), 0x03);

        assert_eq!(ProvenanceOp::from_byte(0x01), ProvenanceOp::Create);
        assert_eq!(ProvenanceOp::from_byte(0x02), ProvenanceOp::Update);
        assert_eq!(ProvenanceOp::from_byte(0x03), ProvenanceOp::Delete);
    }

    #[test]
    fn test_document_chunk_creation() {
        let chunk = DocumentChunk::new(
            "doc-123".to_string(),
            "Hello world".to_string(),
            0,
            5,
            0,
        );

        assert_eq!(chunk.document_id, "doc-123");
        assert_eq!(chunk.content, "Hello world");
        assert_eq!(chunk.chunk_index, 0);
        assert_eq!(chunk.total_chunks, 5);
        assert_eq!(chunk.char_offset, 0);
        assert_eq!(chunk.char_end, 11);
    }

    #[test]
    fn test_compute_hash() {
        let hash = compute_hash("Hello world");
        // SHA-256 of "Hello world"
        assert_eq!(hash, "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
    }
}
