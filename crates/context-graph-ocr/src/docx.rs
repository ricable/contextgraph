//! DOCX Text Extraction
//!
//! This module provides DOCX document text extraction capabilities.
//! Uses ZIP archive parsing to extract document.xml and parse WordprocessingML.

use crate::OcrError;
use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;

/// DOCX Extractor - extracts text from DOCX documents
pub struct DocxExtractor;

impl DocxExtractor {
    /// Create a new DocxExtractor
    pub fn new() -> Self {
        Self
    }

    /// Extract text from DOCX bytes (simplified XML parsing)
    pub fn extract_text(&self, docx_bytes: &[u8]) -> Result<String, OcrError> {
        // DOCX files are ZIP archives containing XML
        // Use zip crate to extract document.xml
        let cursor = Cursor::new(docx_bytes);
        let mut archive = zip::ZipArchive::new(cursor)
            .map_err(|e| OcrError::ProcessingError(format!("Failed to read DOCX as ZIP: {}", e)))?;

        // Try to read word/document.xml
        let mut text = String::new();
        if let Ok(mut file) = archive.by_name("word/document.xml") {
            let mut xml_content = String::new();
            file.read_to_string(&mut xml_content)
                .map_err(|e| OcrError::ProcessingError(format!("Failed to read document.xml: {}", e)))?;

            // Extract text between <w:t> tags
            text = extract_text_from_xml(&xml_content);
        }

        if text.is_empty() {
            // Fallback: try reading as raw and extracting <w:t> tags
            let raw = String::from_utf8_lossy(docx_bytes);
            text = extract_text_from_raw(&raw);
        }

        if text.is_empty() {
            return Err(OcrError::ProcessingError(
                "No text content found in DOCX".to_string(),
            ));
        }

        Ok(text)
    }

    /// Extract text from DOCX file path
    pub fn extract_text_from_file(&self, path: &Path) -> Result<String, OcrError> {
        let content = std::fs::read(path)
            .map_err(|e| OcrError::IoError(e))?;
        self.extract_text(&content)
    }

    /// Extract metadata from DOCX
    pub fn extract_metadata(&self, docx_bytes: &[u8]) -> Result<HashMap<String, String>, OcrError> {
        let mut metadata = HashMap::new();

        let cursor = Cursor::new(docx_bytes);
        let mut archive = match zip::ZipArchive::new(cursor) {
            Ok(a) => a,
            Err(_) => return Ok(metadata),
        };

        // Try to read core.xml for metadata
        if let Ok(mut file) = archive.by_name("docProps/core.xml") {
            use std::io::Read;
            let mut xml_content = String::new();
            if file.read_to_string(&mut xml_content).is_ok() {
                // Extract common metadata fields
                if let Some(title) = extract_xml_element(&xml_content, "dc:title") {
                    metadata.insert("title".to_string(), title);
                }
                if let Some(author) = extract_xml_element(&xml_content, "dc:creator") {
                    metadata.insert("author".to_string(), author);
                }
                if let Some(subject) = extract_xml_element(&xml_content, "dc:subject") {
                    metadata.insert("subject".to_string(), subject);
                }
            }
        }

        Ok(metadata)
    }

    /// Get paragraph count (approximate)
    pub fn get_paragraph_count(&self, docx_bytes: &[u8]) -> Result<usize, OcrError> {
        let text = self.extract_text(docx_bytes)?;
        // Count paragraphs by double newlines
        Ok(text.split("\n\n").filter(|s| !s.is_empty()).count())
    }
}

/// Extract text content from XML by finding <w:t> elements
#[allow(unused_variables)]
fn extract_text_from_xml(xml: &str) -> String {
    let mut text = String::new();

    let bytes = xml.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        // Check for <w:t
        if i + 3 < bytes.len() && bytes[i] == b'<' && bytes[i + 1] == b'w' && bytes[i + 2] == b':' && bytes[i + 3] == b't' {
            // Find closing >
            while i < bytes.len() && bytes[i] != b'>' {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'>' {
                // Extract text until </w:t>
                i += 1;
                let text_start = i;
                while i + 5 < bytes.len() {
                    if bytes[i] == b'<' && bytes[i + 1] == b'/' && bytes[i + 2] == b'w' && bytes[i + 3] == b':' && bytes[i + 4] == b't' {
                        break;
                    }
                    i += 1;
                }
                if text_start < i {
                    let segment = String::from_utf8_lossy(&bytes[text_start..i]).to_string();
                    if !segment.trim().is_empty() {
                        if !text.is_empty() && !text.ends_with(' ') && !text.ends_with('\n') {
                            text.push(' ');
                        }
                        text.push_str(&segment);
                    }
                }
            }
        }
        i += 1;
    }

    text.trim().to_string()
}

/// Extract text from raw DOCX content (fallback)
fn extract_text_from_raw(content: &str) -> String {
    extract_text_from_xml(content)
}

/// Extract a specific element value from XML
fn extract_xml_element(xml: &str, element: &str) -> Option<String> {
    let pattern = format!("<{}>", element);
    let end_pattern = format!("</{}>", element);

    if let Some(start) = xml.find(&pattern) {
        let content_start = start + pattern.len();
        if let Some(end) = xml[content_start..].find(&end_pattern) {
            let value = &xml[content_start..content_start + end];
            // Handle XML entities
            let decoded = value
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&amp;", "&")
                .replace("&apos;", "'")
                .replace("&quot;", "\"");
            return Some(decoded.trim().to_string());
        }
    }
    None
}

/// DOCX extraction result
#[derive(Debug)]
pub struct DocxExtractionResult {
    /// Extracted text
    pub text: String,
    /// Paragraph count
    pub paragraph_count: usize,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl DocxExtractor {
    /// Extract complete DOCX information
    pub fn extract(&self, docx_bytes: &[u8]) -> Result<DocxExtractionResult, OcrError> {
        let text = self.extract_text(docx_bytes)?;
        let paragraph_count = self.get_paragraph_count(docx_bytes)?;
        let metadata = self.extract_metadata(docx_bytes)?;

        Ok(DocxExtractionResult {
            text,
            paragraph_count,
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_docx_extractor_creation() {
        let extractor = DocxExtractor::new();
        assert!(std::mem::size_of_val(&extractor) > 0);
    }
}
