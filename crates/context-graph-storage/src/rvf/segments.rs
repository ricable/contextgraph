//! RVF segment definitions and mappings.
//!
//! This module defines the 15 segment types used in RVF cognitive containers,
//! mapped to their respective byte codes for serialization/deserialization.
//!
//! # Segment Reference
//!
//! | Segment | Code | Purpose |
//! |---------|------|---------|
//! | DOC_SEG | 0x30 | Document container |
//! | PAGE_SEG | 0x31 | Page within document |
//! | CHUNK_SEG | 0x32 | Text chunk within page |
//! | VLM_SEG | 0x33 | Vision-Language Model embedding |
//! | OCR_SEG | 0x34 | OCR text result |
//! | WITNESS_SEG | 0x0B | Append-only witness/audit chain |
//! | VEC_SEG | 0x01 | Raw vector data (f32, f16, bf16, int8) |
//! | INDEX_SEG | 0x02 | HNSW progressive index (Layer A/B/C) |
//! | META_SEG | 0x03 | Vector metadata (JSON, CBOR) |
//! | OVERLAY_SEG | 0x05 | LoRA adapter deltas, MicroLoRA patches |
//! | GRAPH_SEG | 0x06 | Property graph adjacency data |
//! | MODEL_SEG | 0x09 | ML model weights |
//! | CRYPTO_SEG | 0x0A | Signatures and key material (ML-DSA-65/Ed25519) |
//! | WASM_SEG | 0x08 | Embedded WASM modules |
//! | COW_MAP_SEG | 0x20 | Copy-on-write cluster map |

use serde::{Deserialize, Serialize};
use std::fmt;

/// RVF segment types with their byte codes.
///
/// Each variant corresponds to a specific segment type in the RVF specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RvfSegmentType {
    /// Level 0 Root manifest with file metadata (0x00)
    Manifest = 0x00,
    /// Raw vector data (0x01)
    Vec = 0x01,
    /// HNSW progressive index (0x02)
    Index = 0x02,
    /// Vector metadata (0x03)
    Meta = 0x03,
    /// Quantization codebooks (0x04)
    Quant = 0x04,
    /// LoRA adapter deltas (0x05)
    Overlay = 0x05,
    /// Property graph adjacency data (0x06)
    Graph = 0x06,
    /// Dense tensor data (0x07)
    Tensor = 0x07,
    /// Embedded WASM modules (0x08)
    Wasm = 0x08,
    /// ML model weights (0x09)
    Model = 0x09,
    /// Signatures and key material (0x0A)
    Crypto = 0x0A,
    /// Append-only witness/audit chain (0x0B)
    Witness = 0x0B,
    /// Runtime configuration (0x0C)
    Config = 0x0C,
    /// User-defined segment (0x0D)
    Custom = 0x0D,
    /// Linux microkernel image (0x0E)
    Kernel = 0x0E,
    /// eBPF programs (0x0F)
    Ebpf = 0x0F,
    /// Copy-on-write cluster map (0x20)
    CowMap = 0x20,
    /// Cluster reference counts (0x21)
    RefCount = 0x21,
    /// Branch membership filter (0x22)
    Membership = 0x22,
    /// Sparse delta patches (0x23)
    Delta = 0x23,
    // OCR-specific segments (0x30-0x34)
    /// Document container (0x30)
    Doc = 0x30,
    /// Page within document (0x31)
    Page = 0x31,
    /// Text chunk within page (0x32)
    Chunk = 0x32,
    /// Vision-Language Model embedding (0x33)
    Vlm = 0x33,
    /// OCR text result (0x34)
    Ocr = 0x34,
}

impl RvfSegmentType {
    /// Get the byte code for this segment type.
    pub fn byte_code(&self) -> u8 {
        *self as u8
    }

    /// Parse a byte code into a segment type.
    ///
    /// Returns `None` if the byte code is not recognized.
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0x00 => Some(Self::Manifest),
            0x01 => Some(Self::Vec),
            0x02 => Some(Self::Index),
            0x03 => Some(Self::Meta),
            0x04 => Some(Self::Quant),
            0x05 => Some(Self::Overlay),
            0x06 => Some(Self::Graph),
            0x07 => Some(Self::Tensor),
            0x08 => Some(Self::Wasm),
            0x09 => Some(Self::Model),
            0x0A => Some(Self::Crypto),
            0x0B => Some(Self::Witness),
            0x0C => Some(Self::Config),
            0x0D => Some(Self::Custom),
            0x0E => Some(Self::Kernel),
            0x0F => Some(Self::Ebpf),
            0x20 => Some(Self::CowMap),
            0x21 => Some(Self::RefCount),
            0x22 => Some(Self::Membership),
            0x23 => Some(Self::Delta),
            0x30 => Some(Self::Doc),
            0x31 => Some(Self::Page),
            0x32 => Some(Self::Chunk),
            0x33 => Some(Self::Vlm),
            0x34 => Some(Self::Ocr),
            _ => None,
        }
    }

    /// Get the human-readable name for this segment type.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Manifest => "MANIFEST",
            Self::Vec => "VEC",
            Self::Index => "INDEX",
            Self::Meta => "META",
            Self::Quant => "QUANT",
            Self::Overlay => "OVERLAY",
            Self::Graph => "GRAPH",
            Self::Tensor => "TENSOR",
            Self::Wasm => "WASM",
            Self::Model => "MODEL",
            Self::Crypto => "CRYPTO",
            Self::Witness => "WITNESS",
            Self::Config => "CONFIG",
            Self::Custom => "CUSTOM",
            Self::Kernel => "KERNEL",
            Self::Ebpf => "EBPF",
            Self::CowMap => "COW_MAP",
            Self::RefCount => "REFCOUNT",
            Self::Membership => "MEMBERSHIP",
            Self::Delta => "DELTA",
            Self::Doc => "DOC",
            Self::Page => "PAGE",
            Self::Chunk => "CHUNK",
            Self::Vlm => "VLM",
            Self::Ocr => "OCR",
        }
    }

    /// Check if this segment type is data-bearing (contains actual content).
    pub fn is_data_segment(&self) -> bool {
        matches!(
            self,
            Self::Vec
                | Self::Meta
                | Self::Overlay
                | Self::Graph
                | Self::Model
                | Self::Tensor
                | Self::Doc
                | Self::Page
                | Self::Chunk
                | Self::Vlm
                | Self::Ocr
        )
    }

    /// Check if this segment type is index-bearing.
    pub fn is_index_segment(&self) -> bool {
        matches!(self, Self::Index)
    }

    /// Check if this segment type is used for learning/consolidation.
    pub fn is_learning_segment(&self) -> bool {
        matches!(self, Self::Overlay | Self::Graph | Self::Witness)
    }
}

impl fmt::Display for RvfSegmentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (0x{:02X})", self.name(), self.byte_code())
    }
}

impl std::error::Error for RvfSegmentType {}

/// RVF segment header for serialization.
///
/// Contains the segment type, size, and version information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RvfSegmentHeader {
    /// Segment type byte code
    pub segment_type: RvfSegmentType,
    /// Segment version
    pub version: u8,
    /// Segment size in bytes (excluding header)
    pub size: u32,
    /// Reserved for future use
    pub reserved: [u8; 3],
}

impl RvfSegmentHeader {
    /// Create a new segment header.
    pub fn new(segment_type: RvfSegmentType, size: u32) -> Self {
        Self {
            segment_type,
            version: 1,
            size,
            reserved: [0, 0, 0],
        }
    }

    /// Serialize the header to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![
            self.segment_type.byte_code(),
            self.version,
        ];
        bytes.extend_from_slice(&self.size.to_le_bytes());
        bytes.extend_from_slice(&self.reserved);
        bytes
    }

    /// Deserialize header from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 8 {
            return None;
        }
        let segment_type = RvfSegmentType::from_byte(bytes[0])?;
        let version = bytes[1];
        let size = u32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]);
        let reserved = [bytes[6], bytes[7], bytes[8]];
        Some(Self {
            segment_type,
            version,
            size,
            reserved,
        })
    }
}

/// RVF segment containing typed data.
///
/// Generic over the payload type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RvfSegment<T = Vec<u8>> {
    /// Segment header
    pub header: RvfSegmentHeader,
    /// Segment payload
    pub payload: T,
}

impl<T> RvfSegment<T> {
    /// Create a new segment with the given type and payload.
    pub fn new(segment_type: RvfSegmentType, payload: T) -> Self {
        // Size will be calculated when payload is a Vec<u8>
        // Caller must set size manually for other types
        Self {
            header: RvfSegmentHeader::new(segment_type, 0),
            payload,
        }
    }

    /// Get the segment type.
    pub fn segment_type(&self) -> RvfSegmentType {
        self.header.segment_type
    }

    /// Get the segment size.
    pub fn size(&self) -> u32 {
        self.header.size
    }
}

/// Segment statistics for a RVF store.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RvfSegmentStats {
    /// Count of VEC_SEG entries
    pub vec_count: u64,
    /// Count of INDEX_SEG entries
    pub index_count: u64,
    /// Count of META_SEG entries
    pub meta_count: u64,
    /// Count of OVERLAY_SEG entries (LoRA)
    pub overlay_count: u64,
    /// Count of GRAPH_SEG entries
    pub graph_count: u64,
    /// Count of WITNESS_SEG entries
    pub witness_count: u64,
    /// Total size in bytes
    pub total_bytes: u64,
}

impl RvfSegmentStats {
    /// Increment count for a segment type.
    pub fn increment(&mut self, segment_type: RvfSegmentType, bytes: u64) {
        match segment_type {
            RvfSegmentType::Vec => self.vec_count += 1,
            RvfSegmentType::Index => self.index_count += 1,
            RvfSegmentType::Meta => self.meta_count += 1,
            RvfSegmentType::Overlay => self.overlay_count += 1,
            RvfSegmentType::Graph => self.graph_count += 1,
            RvfSegmentType::Witness => self.witness_count += 1,
            _ => {}
        }
        self.total_bytes += bytes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_type_byte_codes() {
        assert_eq!(RvfSegmentType::Vec.byte_code(), 0x01);
        assert_eq!(RvfSegmentType::Index.byte_code(), 0x02);
        assert_eq!(RvfSegmentType::Meta.byte_code(), 0x03);
        assert_eq!(RvfSegmentType::Overlay.byte_code(), 0x05);
        assert_eq!(RvfSegmentType::Graph.byte_code(), 0x06);
        assert_eq!(RvfSegmentType::Wasm.byte_code(), 0x08);
        assert_eq!(RvfSegmentType::Model.byte_code(), 0x09);
        assert_eq!(RvfSegmentType::Crypto.byte_code(), 0x0A);
        assert_eq!(RvfSegmentType::Witness.byte_code(), 0x0B);
        assert_eq!(RvfSegmentType::CowMap.byte_code(), 0x20);
        // OCR segments
        assert_eq!(RvfSegmentType::Doc.byte_code(), 0x30);
        assert_eq!(RvfSegmentType::Page.byte_code(), 0x31);
        assert_eq!(RvfSegmentType::Chunk.byte_code(), 0x32);
        assert_eq!(RvfSegmentType::Vlm.byte_code(), 0x33);
        assert_eq!(RvfSegmentType::Ocr.byte_code(), 0x34);
    }

    #[test]
    fn test_segment_type_roundtrip() {
        for seg_type in [
            RvfSegmentType::Vec,
            RvfSegmentType::Index,
            RvfSegmentType::Graph,
            RvfSegmentType::Overlay,
            RvfSegmentType::Witness,
        ] {
            let parsed = RvfSegmentType::from_byte(seg_type.byte_code());
            assert_eq!(parsed, Some(seg_type), "Roundtrip failed for {:?}", seg_type);
        }
    }

    #[test]
    fn test_segment_type_names() {
        assert_eq!(RvfSegmentType::Vec.name(), "VEC");
        assert_eq!(RvfSegmentType::Index.name(), "INDEX");
        assert_eq!(RvfSegmentType::Graph.name(), "GRAPH");
    }

    #[test]
    fn test_is_data_segment() {
        assert!(RvfSegmentType::Vec.is_data_segment());
        assert!(RvfSegmentType::Meta.is_data_segment());
        assert!(RvfSegmentType::Graph.is_data_segment());
        assert!(!RvfSegmentType::Index.is_data_segment());
        assert!(!RvfSegmentType::Manifest.is_data_segment());
    }

    #[test]
    fn test_header_serialization() {
        let header = RvfSegmentHeader::new(RvfSegmentType::Vec, 1024);
        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), 8);

        let parsed = RvfSegmentHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.segment_type, RvfSegmentType::Vec);
        assert_eq!(parsed.size, 1024);
    }
}
