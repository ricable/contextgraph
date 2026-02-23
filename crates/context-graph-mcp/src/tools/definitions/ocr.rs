//! OCR tool definitions for MCP server.
//!
//! Provides tools for OCR and document ingestion as RVF producer:
//! - cg_ocr_ingest: Process document through OCR and store in RVF
//! - cg_provenance_verify: Verify provenance chain integrity
//! - cg_provenance_export: Export provenance chain for audit
//! - cg_db_init: Initialize OCR database/storage
//! - cg_memory_list: List memories/embeddings
//! - cg_image_extract: Extract images from document
//! - cg_vlm_analyze: Analyze image with VLM

use crate::tools::types::ToolDefinition;
use serde_json::json;

/// Returns OCR tool definitions (7 tools).
pub fn definitions() -> Vec<ToolDefinition> {
    vec![
        // cg_ocr_ingest - Process document through OCR and store in RVF
        ToolDefinition::new(
            "cg_ocr_ingest",
            "Process a document through OCR and store results in RVF format. \
             Supports PDF, DOCX, TXT, and image files. Creates OCR_SEG, CHUNK_SEG, \
             and WITNESS_SEG entries for full provenance tracking.",
            json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the document file to process"
                    },
                    "document_id": {
                        "type": "string",
                        "description": "Optional custom document ID (generated if not provided)"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["fast", "balanced", "accurate"],
                        "default": "accurate",
                        "description": "OCR processing mode"
                    },
                    "store_embeddings": {
                        "type": "boolean",
                        "default": true,
                        "description": "Whether to generate and store embeddings"
                    },
                    "namespace": {
                        "type": "string",
                        "default": "default",
                        "description": "RVF namespace for storage"
                    }
                },
                "required": ["file_path"],
                "additionalProperties": false
            }),
        ),
        // cg_provenance_verify - Verify provenance chain integrity
        ToolDefinition::new(
            "cg_provenance_verify",
            "Verify the integrity of a document's provenance chain. \
             Checks for tampering by validating hash links between entries.",
            json!({
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Document ID to verify provenance for"
                    }
                },
                "required": ["document_id"],
                "additionalProperties": false
            }),
        ),
        // cg_provenance_export - Export provenance chain for audit
        ToolDefinition::new(
            "cg_provenance_export",
            "Export the complete provenance chain for a document as JSON. \
             Useful for audit trails and compliance documentation.",
            json!({
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Document ID to export provenance for"
                    },
                    "include_binary": {
                        "type": "boolean",
                        "default": false,
                        "description": "Include WITNESS_SEG binary data"
                    }
                },
                "required": ["document_id"],
                "additionalProperties": false
            }),
        ),
        // cg_db_init - Initialize OCR database/storage
        ToolDefinition::new(
            "cg_db_init",
            "Initialize the OCR storage backend. Creates necessary tables/indexes \
             for storing OCR results, chunks, and provenance data.",
            json!({
                "type": "object",
                "properties": {
                    "storage_path": {
                        "type": "string",
                        "description": "Optional custom storage path"
                    },
                    "clear_existing": {
                        "type": "boolean",
                        "default": false,
                        "description": "Clear existing data on initialization"
                    }
                },
                "additionalProperties": false
            }),
        ),
        // cg_memory_list - List memories/embeddings
        ToolDefinition::new(
            "cg_memory_list",
            "List stored memories/embeddings in the OCR namespace. \
             Returns IDs, timestamps, and basic metadata.",
            json!({
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "default": "default",
                        "description": "Namespace to list memories from"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 50,
                        "description": "Maximum number of results"
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "Offset for pagination"
                    }
                },
                "additionalProperties": false
            }),
        ),
        // cg_image_extract - Extract images from document
        ToolDefinition::new(
            "cg_image_extract",
            "Extract embedded images from a PDF or document file. \
             Returns base64-encoded image data with page information.",
            json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the document file"
                    },
                    "min_width": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 100,
                        "description": "Minimum image width to extract"
                    },
                    "min_height": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 100,
                        "description": "Minimum image height to extract"
                    }
                },
                "required": ["file_path"],
                "additionalProperties": false
            }),
        ),
        // cg_vlm_analyze - Analyze image with VLM
        ToolDefinition::new(
            "cg_vlm_analyze",
            "Analyze an image using a Vision-Language Model. \
             Extracts text (OCR), describes visual content, and provides analysis.",
            json!({
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file"
                    },
                    "prompt": {
                        "type": "string",
                        "default": "Describe this image in detail",
                        "description": "Analysis prompt for the VLM"
                    },
                    "extract_text": {
                        "type": "boolean",
                        "default": true,
                        "description": "Extract text from image using OCR"
                    }
                },
                "required": ["image_path"],
                "additionalProperties": false
            }),
        ),
    ]
}
