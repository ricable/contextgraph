//! Witness Chain - Provenance tracking with WITNESS_SEG (0x0B) binary format
//!
//! This module implements tamper-evident chain verification for document provenance.
//! The WITNESS_SEG (0x0B) segment type provides an append-only audit chain.
//!
//! # Binary Format
//!
//! ```text
//! [witness_id: 16][timestamp: 8][operation: 1][document_id_len: 2][document_id: N]
//! [content_hash: 32][parent_hash: 32][metadata_len: 2][metadata: N]
//! ```
//!
//! Total fixed fields: 16 + 8 + 1 + 2 + 32 + 32 + 2 = 93 bytes
//! Variable: document_id (N bytes) + metadata (M bytes)

use crate::types::{ProvenanceChain, ProvenanceEntry, ProvenanceOp, VerificationResult};
use crate::OcrError;
use sha2::{Digest, Sha256};
use uuid::Uuid;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Size of witness ID in bytes
const WITNESS_ID_SIZE: usize = 16;

/// Size of timestamp in bytes
const TIMESTAMP_SIZE: usize = 8;

/// Size of operation byte
const OPERATION_SIZE: usize = 1;

/// Size of length prefix for document ID
const DOC_ID_LEN_SIZE: usize = 2;

/// Size of SHA-256 hash in bytes
const HASH_SIZE: usize = 32;

/// Size of length prefix for metadata
const METADATA_LEN_SIZE: usize = 2;

// =============================================================================
// WITNESS CHAIN
// =============================================================================

/// WitnessChain - Manages provenance chain with WITNESS_SEG encoding
pub struct WitnessChain {
    entries: Vec<ProvenanceEntry>,
}

impl Default for WitnessChain {
    fn default() -> Self {
        Self::new()
    }
}

impl WitnessChain {
    /// Create a new WitnessChain
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Record a new provenance entry
    pub fn record_entry(&mut self, entry: ProvenanceEntry) -> ProvenanceEntry {
        // Get previous entry for chaining
        let parent_hash = self.entries.last().map(|e| compute_entry_hash(e));

        let mut entry_with_parent = entry;
        entry_with_parent.parent_hash = parent_hash;

        self.entries.push(entry_with_parent.clone());
        entry_with_parent
    }

    /// Get the full chain for a document
    pub fn get_chain(&self, document_id: &str) -> ProvenanceChain {
        let entries: Vec<ProvenanceEntry> = self
            .entries
            .iter()
            .filter(|e| e.document_id == document_id)
            .cloned()
            .collect();

        let verification = verify_hash_chain(&entries);

        ProvenanceChain {
            document_id: document_id.to_string(),
            entries,
            verified: verification.valid,
            root_hash: verification.root_hash,
        }
    }

    /// Verify chain integrity
    pub fn verify_chain(&self, document_id: &str) -> VerificationResult {
        let chain = self.get_chain(document_id);
        let entries = &chain.entries;

        if entries.is_empty() {
            return VerificationResult {
                valid: true,
                entries_checked: 0,
                errors: vec![],
                tampered_entries: vec![],
            };
        }

        let hash_verification = verify_hash_chain(entries);
        let mut errors = Vec::new();
        let mut tampered = Vec::new();

        if !hash_verification.valid {
            errors.push(format!(
                "Hash chain broken at index {}",
                hash_verification.break_index.unwrap_or(0)
            ));
            if let Some(idx) = hash_verification.break_index {
                if let Some(entry) = entries.get(idx) {
                    tampered.push(entry.id.clone());
                }
            }
        }

        // Check for duplicate IDs
        let mut seen = std::collections::HashSet::new();
        for entry in entries {
            if !seen.insert(&entry.id) {
                errors.push(format!("Duplicate entry ID: {}", entry.id));
                tampered.push(entry.id.clone());
            }
        }

        VerificationResult {
            valid: errors.is_empty(),
            entries_checked: entries.len() as u32,
            errors,
            tampered_entries: tampered,
        }
    }

    /// Export chain as JSON
    pub fn export_chain(&self, document_id: &str) -> Result<String, OcrError> {
        let chain = self.get_chain(document_id);
        serde_json::to_string_pretty(&chain)
            .map_err(|e| OcrError::ProvenanceError(e.to_string()))
    }

    /// Import chain from JSON
    pub fn import_chain(&mut self, json: &str) -> Result<u32, OcrError> {
        let chain: ProvenanceChain = serde_json::from_str(json)
            .map_err(|e| OcrError::ProvenanceError(e.to_string()))?;

        // Validate chain before import
        let verification = verify_hash_chain(&chain.entries);
        if !verification.valid {
            return Err(OcrError::ProvenanceError(format!(
                "Invalid chain: hash verification failed at index {}",
                verification.break_index.unwrap_or(0)
            )));
        }

        let count = chain.entries.len() as u32;
        self.entries.extend(chain.entries);
        Ok(count)
    }

    /// Get entry count
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all entries
    pub fn entries(&self) -> &[ProvenanceEntry] {
        &self.entries
    }

    /// Encode entry to WITNESS_SEG binary format
    pub fn encode_witness_segment(entry: &ProvenanceEntry) -> Vec<u8> {
        let witness_id = hex::decode(&entry.id).unwrap_or_else(|_| {
            // Generate 16 bytes from UUID if ID is not hex
            let id = Uuid::new_v4();
            let bytes = id.as_bytes();
            bytes.to_vec()
        });

        let timestamp = entry.timestamp;
        let operation = entry.operation.to_byte();

        let doc_id_bytes = entry.document_id.as_bytes();
        let doc_id_len = (doc_id_bytes.len() as u16).to_be_bytes();

        let content_hash = hex::decode(&entry.content_hash).unwrap_or_else(|_| vec![0; 32]);

        let parent_hash = entry
            .parent_hash
            .as_ref()
            .and_then(|h| hex::decode(h).ok())
            .unwrap_or_else(|| vec![0; 32]);

        // Metadata as JSON
        let metadata = serde_json::json!({
            "source_path": entry.source_path,
            "chunk_count": entry.chunk_count,
            "embedding_model": entry.embedding_model,
        });
        let metadata_str = metadata.to_string();
        let metadata_bytes = metadata_str.as_bytes();
        let metadata_len = (metadata_bytes.len() as u16).to_be_bytes();

        // Combine all parts
        let mut result = Vec::with_capacity(
            WITNESS_ID_SIZE
                + TIMESTAMP_SIZE
                + OPERATION_SIZE
                + DOC_ID_LEN_SIZE
                + doc_id_bytes.len()
                + HASH_SIZE
                + HASH_SIZE
                + METADATA_LEN_SIZE
                + metadata_bytes.len(),
        );

        // Witness ID (16 bytes)
        result.extend_from_slice(&witness_id[..16.min(witness_id.len())]);
        if witness_id.len() < WITNESS_ID_SIZE {
            result.resize(result.len() + WITNESS_ID_SIZE - witness_id.len(), 0);
        }

        // Timestamp (8 bytes)
        result.extend_from_slice(&timestamp.to_be_bytes());

        // Operation (1 byte)
        result.push(operation);

        // Document ID length + content
        result.extend_from_slice(&doc_id_len);
        result.extend_from_slice(doc_id_bytes);

        // Content hash (32 bytes)
        result.extend_from_slice(&content_hash[..32.min(content_hash.len())]);
        if content_hash.len() < HASH_SIZE {
            result.resize(result.len() + HASH_SIZE - content_hash.len(), 0);
        }

        // Parent hash (32 bytes)
        result.extend_from_slice(&parent_hash[..32.min(parent_hash.len())]);
        if parent_hash.len() < HASH_SIZE {
            result.resize(result.len() + HASH_SIZE - parent_hash.len(), 0);
        }

        // Metadata length + content
        result.extend_from_slice(&metadata_len);
        result.extend_from_slice(metadata_bytes);

        result
    }

    /// Decode WITNESS_SEG binary format to provenance entry
    pub fn decode_witness_segment(data: &[u8]) -> Result<ProvenanceEntry, OcrError> {
        if data.len() < WITNESS_ID_SIZE
            + TIMESTAMP_SIZE
            + OPERATION_SIZE
            + DOC_ID_LEN_SIZE
            + HASH_SIZE
            + HASH_SIZE
            + METADATA_LEN_SIZE
        {
            return Err(OcrError::ProvenanceError(
                "Data too short for WITNESS_SEG format".to_string(),
            ));
        }

        let mut offset = 0;

        // Witness ID (16 bytes)
        let witness_id = hex::encode(&data[offset..offset + WITNESS_ID_SIZE]);
        offset += WITNESS_ID_SIZE;

        // Timestamp (8 bytes)
        let timestamp = i64::from_be_bytes(
            data[offset..offset + TIMESTAMP_SIZE]
                .try_into()
                .map_err(|_| OcrError::ProvenanceError("Invalid timestamp".to_string()))?,
        );
        offset += TIMESTAMP_SIZE;

        // Operation (1 byte)
        let operation = ProvenanceOp::from_byte(data[offset]);
        offset += OPERATION_SIZE;

        // Document ID length (2 bytes) + Document ID
        let doc_id_len = u16::from_be_bytes(
            data[offset..offset + DOC_ID_LEN_SIZE]
                .try_into()
                .map_err(|_| OcrError::ProvenanceError("Invalid doc ID length".to_string()))?,
        ) as usize;
        offset += DOC_ID_LEN_SIZE;

        let document_id = String::from_utf8(data[offset..offset + doc_id_len].to_vec())
            .map_err(|_| OcrError::ProvenanceError("Invalid document ID".to_string()))?;
        offset += doc_id_len;

        // Content hash (32 bytes)
        let content_hash = hex::encode(&data[offset..offset + HASH_SIZE]);
        offset += HASH_SIZE;

        // Parent hash (32 bytes)
        let parent_hash_bytes = &data[offset..offset + HASH_SIZE];
        let parent_hash = if parent_hash_bytes.iter().all(|&b| b == 0) {
            None
        } else {
            Some(hex::encode(parent_hash_bytes))
        };
        offset += HASH_SIZE;

        // Metadata length (2 bytes) + Metadata
        let metadata_len_bytes: [u8; 2] = data[offset..offset + METADATA_LEN_SIZE]
            .try_into()
            .map_err(|_| OcrError::ProvenanceError("Invalid metadata length".to_string()))?;
        let metadata_len = u16::from_be_bytes(metadata_len_bytes) as usize;
        offset += METADATA_LEN_SIZE;

        let metadata_str = String::from_utf8(data[offset..offset + metadata_len].to_vec())
            .map_err(|_| OcrError::ProvenanceError("Invalid metadata".to_string()))?;
        let metadata: serde_json::Value = serde_json::from_str(&metadata_str)
            .map_err(|_| OcrError::ProvenanceError("Invalid metadata JSON".to_string()))?;

        let source_path = metadata["source_path"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let chunk_count = metadata["chunk_count"].as_u64().unwrap_or(0) as u32;
        let embedding_model = metadata["embedding_model"]
            .as_str()
            .unwrap_or("semantic")
            .to_string();

        Ok(ProvenanceEntry {
            id: witness_id,
            document_id,
            content_hash,
            parent_hash,
            timestamp,
            operation,
            source_path,
            chunk_count,
            embedding_model,
        })
    }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Compute SHA-256 hash of content
fn compute_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hex::encode(hasher.finalize())
}

/// Compute hash for a provenance entry
fn compute_entry_hash(entry: &ProvenanceEntry) -> String {
    let data = format!(
        "{}:{}:{}:{}:{}:{}",
        entry.id,
        entry.document_id,
        entry.content_hash,
        entry.parent_hash.as_deref().unwrap_or(""),
        entry.timestamp,
        entry.operation.to_byte(),
    );

    compute_hash(&data)
}

/// Verify hash chain integrity
fn verify_hash_chain(entries: &[ProvenanceEntry]) -> HashChainVerification {
    if entries.is_empty() {
        return HashChainVerification {
            valid: true,
            break_index: None,
            root_hash: String::new(),
        };
    }

    if entries.len() == 1 {
        return HashChainVerification {
            valid: true,
            break_index: None,
            root_hash: entries[0].content_hash.clone(),
        };
    }

    // Sort by timestamp
    let mut sorted = entries.to_vec();
    sorted.sort_by_key(|e| e.timestamp);

    for i in 1..sorted.len() {
        let current = &sorted[i];
        let previous = &sorted[i - 1];

        let expected_parent = compute_entry_hash(previous);

        if current.parent_hash.as_ref() != Some(&expected_parent) {
            return HashChainVerification {
                valid: false,
                break_index: Some(i),
                root_hash: sorted[0].content_hash.clone(),
            };
        }
    }

    HashChainVerification {
        valid: true,
        break_index: None,
        root_hash: sorted[0].content_hash.clone(),
    }
}

struct HashChainVerification {
    valid: bool,
    break_index: Option<usize>,
    root_hash: String,
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/// Create a new witness entry
pub fn create_witness(
    document_id: String,
    content: String,
    operation: ProvenanceOp,
    source_path: String,
    chunk_count: u32,
) -> ProvenanceEntry {
    let content_hash = compute_hash(&content);

    ProvenanceEntry::new(document_id, content_hash, operation, source_path, chunk_count)
}

/// Verify a chain of witnesses
#[allow(unused_variables)]
pub fn verify_chain(document_id: &str, entries: &[ProvenanceEntry]) -> VerificationResult {
    if entries.is_empty() {
        return VerificationResult {
            valid: true,
            entries_checked: 0,
            errors: vec![],
            tampered_entries: vec![],
        };
    }

    let hash_verification = verify_hash_chain(entries);
    let mut errors = Vec::new();
    let mut tampered = Vec::new();

    if !hash_verification.valid {
        errors.push(format!(
            "Hash chain broken at index {}",
            hash_verification.break_index.unwrap_or(0)
        ));
    }

    // Check duplicates
    let mut seen = std::collections::HashSet::new();
    for entry in entries {
        if !seen.insert(&entry.id) {
            errors.push(format!("Duplicate entry: {}", entry.id));
            tampered.push(entry.id.clone());
        }
    }

    VerificationResult {
        valid: errors.is_empty(),
        entries_checked: entries.len() as u32,
        errors,
        tampered_entries: tampered,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_witness_chain_creation() {
        let chain = WitnessChain::new();
        assert!(chain.is_empty());
    }

    #[test]
    fn test_record_entry() {
        let mut chain = WitnessChain::new();

        let entry = create_witness(
            "doc-123".to_string(),
            "Hello world".to_string(),
            ProvenanceOp::Create,
            "/path/to/doc.pdf".to_string(),
            5,
        );

        let recorded = chain.record_entry(entry);
        assert_eq!(chain.len(), 1);
        assert!(recorded.parent_hash.is_none()); // First entry has no parent
    }

    #[test]
    fn test_chain_linking() {
        let mut chain = WitnessChain::new();

        // First entry
        let entry1 = create_witness(
            "doc-123".to_string(),
            "Content 1".to_string(),
            ProvenanceOp::Create,
            "/path/to/doc.pdf".to_string(),
            5,
        );
        chain.record_entry(entry1);

        // Second entry
        let entry2 = create_witness(
            "doc-123".to_string(),
            "Content 2".to_string(),
            ProvenanceOp::Update,
            "/path/to/doc.pdf".to_string(),
            6,
        );
        let recorded2 = chain.record_entry(entry2);

        assert!(recorded2.parent_hash.is_some()); // Second entry has parent
    }

    #[test]
    fn test_verify_chain() {
        let mut chain = WitnessChain::new();

        let entry1 = create_witness(
            "doc-123".to_string(),
            "Content 1".to_string(),
            ProvenanceOp::Create,
            "/path/to/doc.pdf".to_string(),
            5,
        );
        chain.record_entry(entry1);

        let entry2 = create_witness(
            "doc-123".to_string(),
            "Content 2".to_string(),
            ProvenanceOp::Update,
            "/path/to/doc.pdf".to_string(),
            6,
        );
        chain.record_entry(entry2);

        let result = chain.verify_chain("doc-123");
        assert!(result.valid);
    }

    #[test]
    fn test_encode_decode_witness() {
        let entry = create_witness(
            "doc-123".to_string(),
            "Test content".to_string(),
            ProvenanceOp::Create,
            "/path/to/doc.pdf".to_string(),
            3,
        );

        let encoded = WitnessChain::encode_witness_segment(&entry);
        assert!(encoded.len() > 100); // Should be at least header + data

        let decoded = WitnessChain::decode_witness_segment(&encoded).unwrap();
        assert_eq!(decoded.document_id, "doc-123");
    }
}
