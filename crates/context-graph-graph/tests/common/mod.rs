//! Common test utilities for M04-T25 Integration Tests.
//!
//! Provides shared test infrastructure with REAL DATA only - NO MOCKS.
//! Per constitution REQ-KG-TEST requirements.
//!
//! # Module Structure
//!
//! - `fixtures`: Real data generators for consistent test data
//! - `helpers`: Test environment setup and verification utilities
//!
//! # Usage
//!
//! ```rust
//! use common::{fixtures, helpers};
//!
//! let storage = helpers::create_test_storage()?;
//! let nodes = fixtures::generate_test_nodes(100);
//! ```

pub mod fixtures;
pub mod helpers;

// Re-exports for convenience
