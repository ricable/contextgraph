//! Traits for embedding model implementations.
//!
//! This module defines the core trait contract that all 12 embedding models must implement.
//! The `EmbeddingModel` trait provides a unified async interface for embedding generation.
//!
//! # Thread Safety
//!
//! All trait bounds require `Send + Sync` for safe usage in multi-threaded async runtimes.
//! This enables concurrent model execution across worker threads.
//!
//! # Design Principles
//!
//! - **NO FALLBACKS**: Unsupported inputs return `EmbeddingError::UnsupportedModality`
//! - **FAIL FAST**: Invalid state triggers immediate error via `EmbeddingError`
//! - **ASYNC FIRST**: All operations are async for non-blocking I/O

mod embedding_model;

pub use embedding_model::EmbeddingModel;
