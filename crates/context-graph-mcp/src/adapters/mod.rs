//! Adapters bridging external implementations to core traits.
//!
//! This module provides adapter types that bridge real implementations
//! from specialized crates to the core trait interfaces.
//!
//! # Available Adapters
//!
//! - [`LazyMultiArrayProvider`]: Wraps provider for lazy loading on MCP startup
//! - [`CodeStoreAdapter`]: Bridges `CodeStore` to `CodeStorage` trait for code capture
//! - [`LlmCausalHintProvider`]: LLM-based causal hint provider for E5 enhancement

/// LLM-based causal hint provider (requires `llm` feature).
/// Wraps CausalDiscoveryLLM for causal hint generation during store_memory.
#[cfg(feature = "llm")]
pub mod causal_hint;
pub mod code_store_adapter;
pub mod lazy_provider;

// LazyMultiArrayProvider allows immediate MCP startup while models load in background
pub use lazy_provider::LazyMultiArrayProvider;

// CodeStoreAdapter bridges CodeStore to CodeStorage trait for code capture pipeline
pub use code_store_adapter::CodeStoreAdapter;

// LlmCausalHintProvider wraps CausalDiscoveryLLM for causal hint generation
#[cfg(feature = "llm")]
pub use causal_hint::LlmCausalHintProvider;
