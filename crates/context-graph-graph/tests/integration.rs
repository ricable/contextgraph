//! M04-T25: Integration Tests Entry Point.
//!
//! This file provides the entry point for modular integration tests.
//! Tests are organized into focused modules, each under 300 lines.
//!
//! # Test Modules
//!
//! - `storage_tests` - Storage lifecycle, batch operations, CRUD
//! - `hyperbolic_tests` - Poincare ball geometry and distance properties
//! - `cone_tests` - Entailment cone containment and storage
//! - `traversal_tests` - Graph traversal with NT modulation
//! - `search_tests` - Semantic search and contradiction detection
//! - `workflow_tests` - End-to-end workflow integration
//! - `edge_case_tests` - Boundary conditions and edge cases
//! - `nfr_tests` - NFR timing and M04-T27 canonical formula tests

mod common;
mod integration_tests;
