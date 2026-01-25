//! Integration tests for Module 02 - Core Infrastructure.
//!
//! These tests verify end-to-end functionality of the storage layer.
//! All tests use REAL RocksDB instances via tempfile::TempDir.
//!
//! NO MOCK DATA. NO ASYNC. All assertions verify actual database state.
//!
//! # Test Categories
//! 1. Node lifecycle (CRUD)
//! 2. Marblestone edge features (NT weights, steering, shortcuts)
//! 3. Tag and temporal index operations
//! 4. Concurrent access patterns
//! 5. Performance benchmarks
//! 6. Error handling paths
//! 7. Memex trait compliance
//! 8. Edge cases (empty, limits, NaN)
//!
//! # Full State Verification Protocol
//! After each operation:
//! 1. Execute the storage operation
//! 2. Read back from RocksDB via a separate get operation
//! 3. Compare expected vs actual
//! 4. Log BEFORE/TRIGGER/AFTER/VERIFY/RESULT for evidence

mod common;

mod concurrent;
mod edge_crud;
mod edge_marblestone;
mod embedding_ops;
mod error_handling;
mod index_tags_temporal;
mod memex_trait;
mod node_lifecycle;
mod nt_weights;
mod performance;
