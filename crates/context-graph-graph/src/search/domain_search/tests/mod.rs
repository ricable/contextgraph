//! Tests for domain-aware search.
//!
//! Split into submodules for maintainability:
//! - `formula_tests` - Net activation and modulation formula tests
//! - `types_tests` - DomainSearchResult and DomainSearchResults tests
//! - `integration` - Integration tests (GPU required)

mod formula_tests;
mod types_tests;

#[cfg(all(test, feature = "faiss-gpu"))]
mod integration;
