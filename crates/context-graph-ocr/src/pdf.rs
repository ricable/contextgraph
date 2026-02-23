//! PDF Text Extraction using lopdf
//!
//! This module provides PDF text extraction capabilities using the lopdf crate.
//! It serves as a fallback when Datalab API is not available for scanned PDFs.

use crate::OcrError;
use lopdf::Document;
use std::collections::HashMap;

/// PDF Extractor - extracts text from PDF documents
pub struct PdfExtractor {
    /// Include images in extraction
    pub include_images: bool,
}

impl Default for PdfExtractor {
    fn default() -> Self {
        Self {
            include_images: false,
        }
    }
}

impl PdfExtractor {
    /// Create a new PdfExtractor
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract text from PDF bytes
    pub fn extract_text(&self, pdf_bytes: &[u8]) -> Result<String, OcrError> {
        let doc = Document::load_mem(pdf_bytes)
            .map_err(|e| OcrError::PdfError(format!("Failed to load PDF: {}", e)))?;

        self.extract_text_from_doc(&doc)
    }

    /// Extract text from PDF file path
    pub fn extract_text_from_file(&self, path: &std::path::Path) -> Result<String, OcrError> {
        let doc = Document::load(path)
            .map_err(|e| OcrError::PdfError(format!("Failed to load PDF: {}", e)))?;

        self.extract_text_from_doc(&doc)
    }

    /// Extract text from a loaded PDF document
    fn extract_text_from_doc(&self, doc: &Document) -> Result<String, OcrError> {
        let mut all_text = String::new();
        let mut page_texts: Vec<(u32, String)> = Vec::new();

        // Get all pages
        let pages = doc.get_pages();
        let mut page_numbers: Vec<u32> = pages.keys().copied().collect();
        page_numbers.sort();

        for page_num in &page_numbers {
            if let Some(_page_id) = pages.get(page_num) {
                if let Ok(text) = doc.extract_text(&[*page_num]) {
                    let trimmed = text.trim().to_string();
                    if !trimmed.is_empty() {
                        page_texts.push((*page_num, trimmed));
                    }
                }
            }
        }

        // Concatenate pages in order
        for (idx, (page_num, text)) in page_texts.iter().enumerate() {
            if idx > 0 && !all_text.is_empty() {
                all_text.push_str(&format!("\n\n--- Page {} ---\n\n", page_num));
            }
            all_text.push_str(text);
        }

        if all_text.is_empty() {
            return Err(OcrError::PdfError(
                "No text content found in PDF. Document may be scanned.".to_string(),
            ));
        }

        Ok(all_text)
    }

    /// Get page count from PDF
    pub fn get_page_count(&self, pdf_bytes: &[u8]) -> Result<u32, OcrError> {
        let doc = Document::load_mem(pdf_bytes)
            .map_err(|e| OcrError::PdfError(format!("Failed to load PDF: {}", e)))?;

        let pages = doc.get_pages();
        Ok(pages.len() as u32)
    }

    /// Check if PDF has text content (vs scanned)
    pub fn has_text_content(&self, pdf_bytes: &[u8]) -> Result<bool, OcrError> {
        let text = self.extract_text(pdf_bytes)?;
        Ok(!text.trim().is_empty())
    }

    /// Extract metadata from PDF
    pub fn extract_metadata(
        &self,
        pdf_bytes: &[u8],
    ) -> Result<HashMap<String, String>, OcrError> {
        let doc = Document::load_mem(pdf_bytes)
            .map_err(|e| OcrError::PdfError(format!("Failed to load PDF: {}", e)))?;

        let mut metadata = HashMap::new();

        // Try to get standard metadata
        if let Ok(info) = doc.trailer.get(b"Info") {
            if let Ok(info_ref) = info.as_reference() {
                if let Ok(info_dict) = doc.get_dictionary(info_ref) {
                    // Title
                    if let Ok(title) = info_dict.get(b"Title") {
                        if let Ok(title_str) = title.as_string() {
                            metadata.insert(
                                "title".to_string(),
                                String::from_utf8_lossy(title_str.as_bytes()).to_string(),
                            );
                        }
                    }

                    // Author
                    if let Ok(author) = info_dict.get(b"Author") {
                        if let Ok(author_str) = author.as_string() {
                            metadata.insert(
                                "author".to_string(),
                                String::from_utf8_lossy(author_str.as_bytes()).to_string(),
                            );
                        }
                    }

                    // Subject
                    if let Ok(subject) = info_dict.get(b"Subject") {
                        if let Ok(subject_str) = subject.as_string() {
                            metadata.insert(
                                "subject".to_string(),
                                String::from_utf8_lossy(subject_str.as_bytes()).to_string(),
                            );
                        }
                    }

                    // Creator
                    if let Ok(creator) = info_dict.get(b"Creator") {
                        if let Ok(creator_str) = creator.as_string() {
                            metadata.insert(
                                "creator".to_string(),
                                String::from_utf8_lossy(creator_str.as_bytes()).to_string(),
                            );
                        }
                    }

                    // Producer
                    if let Ok(producer) = info_dict.get(b"Producer") {
                        if let Ok(producer_str) = producer.as_string() {
                            metadata.insert(
                                "producer".to_string(),
                                String::from_utf8_lossy(producer_str.as_bytes()).to_string(),
                            );
                        }
                    }
                }
            }
        }

        // Add page count
        let pages = doc.get_pages();
        metadata.insert("page_count".to_string(), pages.len().to_string());

        Ok(metadata)
    }
}

/// PDF extraction result
#[derive(Debug)]
pub struct PdfExtractionResult {
    /// Extracted text
    pub text: String,
    /// Page count
    pub page_count: u32,
    /// Metadata
    pub metadata: HashMap<String, String>,
    /// Whether text was extracted (vs scanned)
    pub has_text: bool,
}

impl PdfExtractor {
    /// Extract complete PDF information
    pub fn extract(&self, pdf_bytes: &[u8]) -> Result<PdfExtractionResult, OcrError> {
        let text = self.extract_text(pdf_bytes).unwrap_or_default();
        let page_count = self.get_page_count(pdf_bytes)?;
        let metadata = self.extract_metadata(pdf_bytes)?;
        let has_text = !text.trim().is_empty();

        Ok(PdfExtractionResult {
            text,
            page_count,
            metadata,
            has_text,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdf_extractor_creation() {
        let extractor = PdfExtractor::new();
        assert!(!extractor.include_images);
    }

    #[test]
    fn test_pdf_extractor_default() {
        let extractor = PdfExtractor::default();
        assert!(!extractor.include_images);
    }
}
