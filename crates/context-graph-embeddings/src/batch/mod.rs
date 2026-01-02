//! Batch processing infrastructure for embedding requests.
//!
//! This module provides types and utilities for batching multiple embedding
//! requests together to improve GPU utilization and overall throughput.
//!
//! # Architecture
//!
//! The batch system consists of three main components:
//!
//! - **`BatchRequest`**: Individual embedding request with async response channel
//! - **`BatchQueue`**: Per-model queue that collects and organizes pending requests
//! - **`Batch`**: Assembled batch ready for GPU processing
//!
//! # Example Flow
//!
//! ```text
//! Client 1 ─┬─► BatchQueue ──► Batch ──► Model ──► Results
//! Client 2 ─┤    (collect)    (assemble)  (GPU)   (distribute)
//! Client 3 ─┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use context_graph_embeddings::batch::{BatchQueue, BatchRequest};
//! use context_graph_embeddings::config::BatchConfig;
//! use context_graph_embeddings::types::{ModelId, ModelInput};
//!
//! // Create queue for a model
//! let config = BatchConfig::default();
//! let mut queue = BatchQueue::new(ModelId::Semantic, config);
//!
//! // Submit requests
//! let input = ModelInput::text("Hello, world!").unwrap();
//! let (request, receiver) = BatchRequest::new(input, ModelId::Semantic);
//! queue.push(request);
//!
//! // Process when ready
//! if queue.should_flush() {
//!     if let Some(batch) = queue.drain_batch() {
//!         // Run inference on batch.inputs...
//!         let results = model.embed_batch(&batch.inputs).await?;
//!         batch.complete(results);
//!     }
//! }
//!
//! // Clients await their results
//! let embedding = receiver.await??;
//! ```
//!
//! # Design Principles
//!
//! - **NO FALLBACKS**: Errors propagate immediately with full context
//! - **FAIL FAST**: Invalid state = immediate error
//! - **ASYNC NATIVE**: Uses tokio oneshot channels for response delivery
//! - **THREAD SAFE**: Statistics use atomics for concurrent access

mod types;

pub use types::{
    Batch,
    BatchQueue,
    BatchQueueStats,
    BatchQueueSummary,
    BatchRequest,
};
