//! Integration tests for RocksDB storage column families.
//!
//! These tests use REAL RocksDB instances - NO MOCKS.
//! Each test creates a temporary database to verify actual functionality.
//!
//! # Constitution Reference
//!
//! - testing.types.integration: tests/integration/ - MCP, graph, session
//! - AP-007: Stub data in prod → use tests/fixtures/
//!
//! # Tasks Tested
//!
//! - M04-T12: Column family definitions and configuration
//! - M04-T13: GraphStorage implementation
//! - M04-T13a: Schema migrations
//!
//! # Module Structure
//!
//! - `constants_tests`: Column family names and StorageConfig validation
//! - `rocksdb_basic_tests`: Basic RocksDB column family and read/write operations
//! - `rocksdb_advanced_tests`: Advanced RocksDB operations (prefix scan, overwrite, etc.)
//! - `graph_storage_crud_tests`: GraphStorage CRUD operations
//! - `graph_storage_batch_tests`: GraphStorage batch and iteration operations
//! - `migration_tests`: Schema versioning and migrations
//! - `edge_cases_tests`: Error handling and concurrent access

#[path = "storage_tests/constants_tests.rs"]
mod constants_tests;

#[path = "storage_tests/edge_cases_tests.rs"]
mod edge_cases_tests;

#[path = "storage_tests/graph_storage_batch_tests.rs"]
mod graph_storage_batch_tests;

#[path = "storage_tests/graph_storage_crud_tests.rs"]
mod graph_storage_crud_tests;

#[path = "storage_tests/migration_tests.rs"]
mod migration_tests;

#[path = "storage_tests/rocksdb_advanced_tests.rs"]
mod rocksdb_advanced_tests;

#[path = "storage_tests/rocksdb_basic_tests.rs"]
mod rocksdb_basic_tests;
