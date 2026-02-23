//! Context Graph OCR - Document Ingestion as RVF Producer
//!
//! This crate provides OCR and document ingestion capabilities for the Context Graph system.
//! It integrates with the 13-embedder pipeline to produce RVF segments:
//! - OCR_SEG (0x34): Raw OCR results
//! - CHUNK_SEG (0x32): Text chunks for embedding
//! - WITNESS_SEG (0x0B): Provenance chain for tamper detection
//!
//! # Architecture
//!
//! ```text
//! +-------------+    +-------------+    +-------------+    +-------------+
//! |   Input     | -> |   Datalab   | -> |   Chunker   | -> |  Embedder   |
//! |  Document  |    |    OCR      |    |             |    |  Pipeline   |
//! +-------------+    +-------------+    +-------------+    +-------------+
//!       |                  |                  |                  |
//!       v                  v                  v                  v
//! +-------------+    +-------------+    +-------------+    +-------------+
//! |   PDF/DOCX  |    |  OCRResult |    |   Chunks    |    |   VEC_SEG   |
//! |   Parser    |    |  (OCR_SEG) |    | (CHUNK_SEG) |    |  +INDEX_SEG |
//! +-------------+    +-------------+    +-------------+    +-------------+
//!                                            |
//!                                            v
//!                                     +-------------+
//!                                     |  Witness    |
//!                                     | (WITNESS)   |
//!                                     +-------------+
//! ```
//!
//! # Usage
//!
//! ```rust
//! use context_graph_ocr::{OcrClient, OcrMode};
//!
//! let client = OcrClient::new();
//! let result = client.process_document("/path/to/document.pdf", OcrMode::Accurate).await?;
//! ```

pub mod types;
pub mod datalab;
pub mod pdf;
pub mod docx;
pub mod image;
pub mod witness;
pub mod chunker;

pub use types::{DocumentChunk, ProvenanceEntry, OcrMode, ProvenanceOp};
pub use datalab::DatalabClient;
pub use pdf::PdfExtractor;
pub use docx::DocxExtractor;
pub use image::PdfImageExtractor;
pub use witness::{WitnessChain, create_witness, verify_chain};
pub use chunker::DocumentChunker;

use thiserror::Error;

// =============================================================================
// ERROR TYPES
// =============================================================================

/// OCR processing errors
#[derive(Error, Debug)]
pub enum OcrError {
    #[error("Document processing failed: {0}")]
    ProcessingError(String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Unsupported document format: {0}")]
    UnsupportedFormat(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("PDF error: {0}")]
    PdfError(String),

    #[error("Provenance error: {0}")]
    ProvenanceError(String),
}

/// Result type for OCR operations
pub type OcrResult<T> = Result<T, OcrError>;

// =============================================================================
// RVF SEGMENT TYPE CONSTANTS
// =============================================================================

/// RVF Segment type constants for OCR pipeline
pub mod rvf_segments {
    /// OCR Result segment - raw OCR text extraction
    pub const OCR_SEG: u8 = 0x34;

    /// Text Chunk segment - chunked text for embedding
    pub const CHUNK_SEG: u8 = 0x32;

    /// Witness segment - provenance chain
    pub const WITNESS_SEG: u8 = 0x0B;

    /// Vector segment - embedding output
    pub const VEC_SEG: u8 = 0x01;

    /// Index segment - HNSW index reference
    pub const INDEX_SEG: u8 = 0x10;
}
