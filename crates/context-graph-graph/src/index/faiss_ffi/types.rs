//! FAISS type definitions for FFI bindings.
//!
//! This module contains metric type enums and opaque pointer types
//! used by the FAISS C API bindings.
//!
//! # Constitution Reference
//!
//! - TECH-GRAPH-004 Section 3.1: FAISS FFI Bindings

use std::os::raw::c_int;

// ========== Metric Type ==========

/// Metric type for distance computation.
///
/// Determines how similarity is measured between vectors.
/// Must match FAISS MetricType enum values exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MetricType {
    /// Inner product (cosine similarity when normalized).
    /// Higher values = more similar.
    InnerProduct = 0,

    /// L2 (Euclidean) distance.
    /// Lower values = more similar.
    #[default]
    L2 = 1,
}

// ========== Opaque Pointer Types ==========

/// Opaque pointer to FAISS index.
///
/// This type represents any FAISS index (Flat, IVF, PQ, GPU, etc.).
/// The actual type is determined by how the index was created.
#[repr(C)]
pub struct FaissIndex {
    _private: [u8; 0],
}

/// Opaque pointer to FAISS GPU resources provider interface.
///
/// This is the abstract interface that StandardGpuResources implements.
#[repr(C)]
pub struct FaissGpuResourcesProvider {
    _private: [u8; 0],
}

/// Opaque pointer to FAISS standard GPU resources.
///
/// Manages GPU memory allocation for FAISS operations.
/// Must be freed with `faiss_StandardGpuResources_free`.
#[repr(C)]
pub struct FaissStandardGpuResources {
    _private: [u8; 0],
}

// ========== Type Aliases ==========

/// C int type alias for FFI.
pub type CInt = c_int;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_type_values() {
        // FAISS C API requires exact enum values
        assert_eq!(MetricType::InnerProduct as i32, 0);
        assert_eq!(MetricType::L2 as i32, 1);
    }

    #[test]
    fn test_metric_type_default() {
        assert_eq!(MetricType::default(), MetricType::L2);
    }

    #[test]
    fn test_metric_type_debug() {
        assert_eq!(format!("{:?}", MetricType::L2), "L2");
        assert_eq!(format!("{:?}", MetricType::InnerProduct), "InnerProduct");
    }

    #[test]
    fn test_metric_type_clone() {
        let m1 = MetricType::L2;
        let m2 = m1;
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_metric_type_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MetricType::L2);
        set.insert(MetricType::InnerProduct);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_opaque_types_zero_size() {
        // Opaque types should have zero size (for FFI safety)
        assert_eq!(std::mem::size_of::<FaissIndex>(), 0);
        assert_eq!(std::mem::size_of::<FaissGpuResourcesProvider>(), 0);
        assert_eq!(std::mem::size_of::<FaissStandardGpuResources>(), 0);
    }
}
