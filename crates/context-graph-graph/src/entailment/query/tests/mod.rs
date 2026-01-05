//! Tests for entailment query operations.
//!
//! MUST USE REAL DATA, NO MOCKS (per constitution REQ-KG-TEST)
//!
//! Test modules:
//! - `helpers`: Test utilities and setup functions
//! - `types_tests`: Tests for EntailmentDirection, EntailmentQueryParams, etc.
//! - `single_tests`: Tests for is_entailed_by, entailment_score
//! - `batch_tests`: Tests for entailment_check_batch

mod batch_tests;
mod helpers;
mod single_tests;
mod types_tests;
