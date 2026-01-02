//! Batch queue and request types for asynchronous embedding processing.
//!
//! This module provides the infrastructure for batching multiple embedding
//! requests together to improve GPU utilization and throughput.
//!
//! # Architecture
//!
//! ```text
//! Client             BatchQueue            BatchProcessor
//!   |                    |                      |
//!   |--BatchRequest-->  push()                  |
//!   |--BatchRequest-->  push()                  |
//!   |                    |                      |
//!   |              should_flush()               |
//!   |                    |                      |
//!   |               drain_batch()-->Batch------>|
//!   |                    |                      | (GPU inference)
//!   |<----Result---------+--------complete()----+
//! ```
//!
//! # Design Principles
//!
//! - **NO FALLBACKS**: Errors propagate immediately with full context
//! - **FAIL FAST**: Invalid state = immediate EmbeddingError
//! - **ASYNC NATIVE**: Uses tokio oneshot channels for response delivery
//! - **THREAD SAFE**: Statistics use atomics for concurrent access

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use tokio::sync::oneshot;

use crate::config::BatchConfig;
use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::{ModelEmbedding, ModelId, ModelInput};

// Re-export Uuid for external use
pub use uuid::Uuid;

// ============================================================================
// BATCH REQUEST
// ============================================================================

/// Individual embedding request submitted to the batch system.
///
/// Each request carries its input, target model, and a response channel
/// for asynchronous result delivery.
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_embeddings::batch::BatchRequest;
/// use context_graph_embeddings::types::{ModelId, ModelInput};
///
/// let input = ModelInput::text("Hello, world!").unwrap();
/// let (request, receiver) = BatchRequest::new(input, ModelId::Semantic);
///
/// // Submit request to queue...
/// // queue.push(request);
///
/// // Later, receive result
/// let result = receiver.await.unwrap();
/// ```
#[derive(Debug)]
pub struct BatchRequest {
    /// Unique request identifier for tracking and debugging.
    pub id: Uuid,

    /// Input to embed.
    pub input: ModelInput,

    /// Target model for embedding.
    pub model_id: ModelId,

    /// Channel for returning result.
    /// Consumed when the request is completed.
    pub response_tx: oneshot::Sender<EmbeddingResult<ModelEmbedding>>,

    /// Timestamp when request was submitted.
    /// Used for timeout calculations and metrics.
    pub submitted_at: Instant,

    /// Priority level (higher = more urgent).
    /// Default is 0. Higher values are processed first.
    pub priority: u8,
}

impl BatchRequest {
    /// Create a new batch request with default priority.
    ///
    /// Returns the request and a receiver for the result.
    ///
    /// # Arguments
    /// * `input` - The input to embed
    /// * `model_id` - The model to use for embedding
    ///
    /// # Returns
    /// A tuple of (request, receiver) where:
    /// - `request` should be submitted to a BatchQueue
    /// - `receiver` will receive the embedding result
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let input = ModelInput::text("Hello").unwrap();
    /// let (request, receiver) = BatchRequest::new(input, ModelId::Semantic);
    /// ```
    #[must_use]
    pub fn new(
        input: ModelInput,
        model_id: ModelId,
    ) -> (Self, oneshot::Receiver<EmbeddingResult<ModelEmbedding>>) {
        let (tx, rx) = oneshot::channel();
        let request = Self {
            id: Uuid::new_v4(),
            input,
            model_id,
            response_tx: tx,
            submitted_at: Instant::now(),
            priority: 0,
        };
        (request, rx)
    }

    /// Create a new batch request with specified priority.
    ///
    /// # Arguments
    /// * `input` - The input to embed
    /// * `model_id` - The model to use for embedding
    /// * `priority` - Priority level (higher = more urgent, 0-255)
    #[must_use]
    pub fn with_priority(
        input: ModelInput,
        model_id: ModelId,
        priority: u8,
    ) -> (Self, oneshot::Receiver<EmbeddingResult<ModelEmbedding>>) {
        let (tx, rx) = oneshot::channel();
        let request = Self {
            id: Uuid::new_v4(),
            input,
            model_id,
            response_tx: tx,
            submitted_at: Instant::now(),
            priority,
        };
        (request, rx)
    }

    /// Time elapsed since submission.
    ///
    /// Used for timeout checking and metrics.
    #[inline]
    #[must_use]
    pub fn elapsed(&self) -> std::time::Duration {
        self.submitted_at.elapsed()
    }

    /// Estimated token count for batching decisions.
    ///
    /// This is a rough estimate used for padding calculations:
    /// - Text: ~4 characters per token
    /// - Code: ~3 characters per token (more token-dense)
    /// - Image/Audio: fixed estimate of 100 tokens
    ///
    /// # Returns
    /// Estimated number of tokens for this input.
    #[must_use]
    pub fn estimated_tokens(&self) -> usize {
        match &self.input {
            ModelInput::Text { content, instruction } => {
                // Include instruction in estimate if present
                let total_len = content.len() + instruction.as_ref().map_or(0, |s| s.len());
                // Rough estimate: 4 chars per token, minimum 1
                (total_len / 4).max(1)
            }
            ModelInput::Code { content, .. } => {
                // Code is often more token-dense
                (content.len() / 3).max(1)
            }
            // Non-text inputs get fixed estimate
            ModelInput::Image { .. } | ModelInput::Audio { .. } => 100,
        }
    }
}

// ============================================================================
// BATCH QUEUE STATS
// ============================================================================

/// Queue statistics for monitoring and debugging.
///
/// All fields use atomics for thread-safe concurrent updates.
///
/// # Example
///
/// ```rust,ignore
/// let stats = BatchQueueStats::default();
/// stats.record_request();
/// stats.record_batch(10, 5000); // 10 items, 5ms wait
///
/// let summary = stats.summary();
/// println!("Processed {} batches", summary.batches_processed);
/// ```
#[derive(Debug, Default)]
pub struct BatchQueueStats {
    /// Total requests received.
    pub requests_received: AtomicU64,

    /// Total batches processed.
    pub batches_processed: AtomicU64,

    /// Total requests completed successfully.
    pub requests_completed: AtomicU64,

    /// Total requests that failed.
    pub requests_failed: AtomicU64,

    /// Cumulative wait time in microseconds.
    pub total_wait_time_us: AtomicU64,

    /// Running sum for average batch size calculation.
    /// Stored as (sum * 1000) to preserve precision.
    batch_size_sum: AtomicU64,
}

impl Clone for BatchQueueStats {
    fn clone(&self) -> Self {
        Self {
            requests_received: AtomicU64::new(self.requests_received.load(Ordering::Relaxed)),
            batches_processed: AtomicU64::new(self.batches_processed.load(Ordering::Relaxed)),
            requests_completed: AtomicU64::new(self.requests_completed.load(Ordering::Relaxed)),
            requests_failed: AtomicU64::new(self.requests_failed.load(Ordering::Relaxed)),
            total_wait_time_us: AtomicU64::new(self.total_wait_time_us.load(Ordering::Relaxed)),
            batch_size_sum: AtomicU64::new(self.batch_size_sum.load(Ordering::Relaxed)),
        }
    }
}

impl BatchQueueStats {
    /// Record a new request received.
    #[inline]
    pub fn record_request(&self) {
        self.requests_received.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a batch processed.
    ///
    /// # Arguments
    /// * `size` - Number of requests in the batch
    /// * `wait_time_us` - Average wait time in microseconds
    #[inline]
    pub fn record_batch(&self, size: usize, wait_time_us: u64) {
        self.batches_processed.fetch_add(1, Ordering::Relaxed);
        self.batch_size_sum.fetch_add(size as u64, Ordering::Relaxed);
        self.total_wait_time_us.fetch_add(wait_time_us, Ordering::Relaxed);
    }

    /// Record a request completion.
    ///
    /// # Arguments
    /// * `success` - Whether the request completed successfully
    #[inline]
    pub fn record_completion(&self, success: bool) {
        if success {
            self.requests_completed.fetch_add(1, Ordering::Relaxed);
        } else {
            self.requests_failed.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get a summary snapshot of current statistics.
    #[must_use]
    pub fn summary(&self) -> BatchQueueSummary {
        let batches = self.batches_processed.load(Ordering::Relaxed);
        let size_sum = self.batch_size_sum.load(Ordering::Relaxed);
        let wait_sum = self.total_wait_time_us.load(Ordering::Relaxed);

        BatchQueueSummary {
            requests_received: self.requests_received.load(Ordering::Relaxed),
            batches_processed: batches,
            requests_completed: self.requests_completed.load(Ordering::Relaxed),
            requests_failed: self.requests_failed.load(Ordering::Relaxed),
            avg_batch_size: if batches > 0 {
                (size_sum as f64) / (batches as f64)
            } else {
                0.0
            },
            avg_wait_time_us: if batches > 0 {
                wait_sum / batches
            } else {
                0
            },
        }
    }

    /// Reset all statistics to zero.
    pub fn reset(&self) {
        self.requests_received.store(0, Ordering::Relaxed);
        self.batches_processed.store(0, Ordering::Relaxed);
        self.requests_completed.store(0, Ordering::Relaxed);
        self.requests_failed.store(0, Ordering::Relaxed);
        self.total_wait_time_us.store(0, Ordering::Relaxed);
        self.batch_size_sum.store(0, Ordering::Relaxed);
    }
}

/// Summary snapshot of queue statistics.
///
/// This is a non-atomic copy for reporting purposes.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchQueueSummary {
    /// Total requests received.
    pub requests_received: u64,

    /// Total batches processed.
    pub batches_processed: u64,

    /// Total requests completed successfully.
    pub requests_completed: u64,

    /// Total requests that failed.
    pub requests_failed: u64,

    /// Average batch size (floating point for precision).
    pub avg_batch_size: f64,

    /// Average wait time in microseconds.
    pub avg_wait_time_us: u64,
}

// ============================================================================
// BATCH QUEUE
// ============================================================================

/// Queue of pending requests for a single model.
///
/// Each model has its own BatchQueue, allowing independent batching
/// and timeout behavior per model.
///
/// # Thread Safety
///
/// This struct is NOT Send/Sync due to the oneshot channels.
/// Wrap in Arc<Mutex<>> or use within a single task.
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_embeddings::batch::BatchQueue;
/// use context_graph_embeddings::config::BatchConfig;
/// use context_graph_embeddings::types::ModelId;
///
/// let config = BatchConfig::default();
/// let mut queue = BatchQueue::new(ModelId::Semantic, config);
///
/// // Add requests
/// let (request, _rx) = BatchRequest::new(input, ModelId::Semantic);
/// queue.push(request);
///
/// // Check if ready to process
/// if queue.should_flush() {
///     if let Some(batch) = queue.drain_batch() {
///         // Process batch...
///     }
/// }
/// ```
#[derive(Debug)]
pub struct BatchQueue {
    /// Pending requests ordered by submission time.
    requests: VecDeque<BatchRequest>,

    /// Configuration for batching behavior.
    config: BatchConfig,

    /// Model this queue serves.
    model_id: ModelId,

    /// Statistics.
    stats: BatchQueueStats,
}

impl BatchQueue {
    /// Create a new batch queue for a specific model.
    ///
    /// # Arguments
    /// * `model_id` - The model this queue serves
    /// * `config` - Batching configuration
    #[must_use]
    pub fn new(model_id: ModelId, config: BatchConfig) -> Self {
        Self {
            requests: VecDeque::new(),
            config,
            model_id,
            stats: BatchQueueStats::default(),
        }
    }

    /// Add request to queue.
    ///
    /// The request will be batched with others and processed
    /// when `should_flush()` returns true.
    pub fn push(&mut self, request: BatchRequest) {
        self.stats.record_request();
        self.requests.push_back(request);
    }

    /// Check if queue should be flushed (batch ready).
    ///
    /// Returns true if:
    /// - Queue has reached max_batch_size, OR
    /// - Oldest request has waited >= max_wait_ms
    ///
    /// Returns false if queue is empty.
    #[must_use]
    pub fn should_flush(&self) -> bool {
        if self.requests.is_empty() {
            return false;
        }

        // Flush if reached max batch size
        if self.requests.len() >= self.config.max_batch_size {
            return true;
        }

        // Flush if oldest request waited too long
        if let Some(oldest) = self.requests.front() {
            if oldest.elapsed().as_millis() as u64 >= self.config.max_wait_ms {
                return true;
            }
        }

        false
    }

    /// Extract a batch of requests for processing.
    ///
    /// Drains up to `max_batch_size` requests from the queue.
    /// If `sort_by_length` is enabled, sorts by estimated token count
    /// for padding efficiency.
    ///
    /// # Returns
    /// `Some(Batch)` if there are requests to process, `None` if queue is empty.
    pub fn drain_batch(&mut self) -> Option<Batch> {
        if self.requests.is_empty() {
            return None;
        }

        let batch_size = self.requests.len().min(self.config.max_batch_size);
        let mut batch = Batch::new(self.model_id);

        // Drain requests
        let mut requests: Vec<BatchRequest> = self.requests
            .drain(..batch_size)
            .collect();

        // Calculate average wait time before moving requests
        let avg_wait_us = if !requests.is_empty() {
            let total_wait: u64 = requests.iter()
                .map(|r| r.elapsed().as_micros() as u64)
                .sum();
            total_wait / requests.len() as u64
        } else {
            0
        };

        // Optionally sort by length for padding efficiency
        if self.config.sort_by_length {
            requests.sort_by_key(|r| r.estimated_tokens());
        }

        // Add requests to batch
        for request in requests {
            batch.add(request);
        }

        // Update statistics
        self.stats.record_batch(batch.len(), avg_wait_us);

        Some(batch)
    }

    /// Number of pending requests.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Check if queue is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Oldest request wait time.
    ///
    /// # Returns
    /// `Some(Duration)` if queue is not empty, `None` if empty.
    #[must_use]
    pub fn oldest_wait_time(&self) -> Option<std::time::Duration> {
        self.requests.front().map(|r| r.elapsed())
    }

    /// Clear all pending requests with error.
    ///
    /// Sends a BatchError to all pending request channels and clears the queue.
    /// Used for graceful shutdown or error recovery.
    ///
    /// # Arguments
    /// * `message` - The error message to send to all pending requests
    pub fn cancel_all(&mut self, message: impl Into<String>) {
        let msg = message.into();
        for request in self.requests.drain(..) {
            // Ignore send errors (receiver may have dropped)
            let _ = request.response_tx.send(Err(EmbeddingError::BatchError {
                message: msg.clone(),
            }));
            self.stats.record_completion(false);
        }
    }

    /// Get the model this queue serves.
    #[inline]
    #[must_use]
    pub fn model_id(&self) -> ModelId {
        self.model_id
    }

    /// Get the current configuration.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// Get queue statistics.
    #[inline]
    #[must_use]
    pub fn stats(&self) -> &BatchQueueStats {
        &self.stats
    }

    /// Get a summary of queue statistics.
    #[must_use]
    pub fn stats_summary(&self) -> BatchQueueSummary {
        self.stats.summary()
    }
}

// ============================================================================
// BATCH
// ============================================================================

/// Assembled batch ready for processing.
///
/// Contains all the inputs to be processed together and the channels
/// to send results back to individual requesters.
///
/// # Lifecycle
///
/// 1. Created empty with `Batch::new()`
/// 2. Requests added with `add()`
/// 3. Processed by embedding model
/// 4. Results distributed with `complete()` or `fail()`
///
/// # Example
///
/// ```rust,ignore
/// let mut batch = Batch::new(ModelId::Semantic);
/// batch.add(request1);
/// batch.add(request2);
///
/// // After processing...
/// let results = vec![embedding1, embedding2];
/// batch.complete(results);
/// ```
#[derive(Debug)]
pub struct Batch {
    /// Batch identifier for tracking.
    pub id: Uuid,

    /// Model to use.
    pub model_id: ModelId,

    /// Inputs in this batch.
    pub inputs: Vec<ModelInput>,

    /// Response channels (same order as inputs).
    pub response_txs: Vec<oneshot::Sender<EmbeddingResult<ModelEmbedding>>>,

    /// Original request IDs for tracking.
    pub request_ids: Vec<Uuid>,

    /// When batch was assembled.
    pub assembled_at: Instant,

    /// Total estimated tokens in batch (for padding estimation).
    pub total_tokens: usize,
}

impl Batch {
    /// Create a new empty batch for a model.
    #[must_use]
    pub fn new(model_id: ModelId) -> Self {
        Self {
            id: Uuid::new_v4(),
            model_id,
            inputs: Vec::new(),
            response_txs: Vec::new(),
            request_ids: Vec::new(),
            assembled_at: Instant::now(),
            total_tokens: 0,
        }
    }

    /// Add a request to the batch.
    ///
    /// Consumes the request and stores its components.
    pub fn add(&mut self, request: BatchRequest) {
        self.total_tokens += request.estimated_tokens();
        self.request_ids.push(request.id);
        self.inputs.push(request.input);
        self.response_txs.push(request.response_tx);
    }

    /// Number of items in batch.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    /// Check if batch is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }

    /// Time since batch was assembled.
    #[inline]
    #[must_use]
    pub fn elapsed(&self) -> std::time::Duration {
        self.assembled_at.elapsed()
    }

    /// Send results back to requesters.
    ///
    /// Consumes the batch and sends each result to its corresponding
    /// response channel. Results must be in the same order as the inputs.
    ///
    /// # Arguments
    /// * `results` - Embedding results, one per input
    ///
    /// # Panics
    /// Panics if `results.len() != self.len()` in debug builds.
    pub fn complete(self, results: Vec<EmbeddingResult<ModelEmbedding>>) {
        debug_assert_eq!(
            self.response_txs.len(),
            results.len(),
            "Results count ({}) must match batch size ({})",
            results.len(),
            self.response_txs.len()
        );

        for (tx, result) in self.response_txs.into_iter().zip(results.into_iter()) {
            // Ignore send errors (receiver may have dropped)
            let _ = tx.send(result);
        }
    }

    /// Send error to all requesters.
    ///
    /// Consumes the batch and sends a BatchError to all response channels.
    ///
    /// # Arguments
    /// * `message` - The error message to send to all requesters
    pub fn fail(self, message: impl Into<String>) {
        let msg = message.into();
        for tx in self.response_txs {
            let _ = tx.send(Err(EmbeddingError::BatchError {
                message: msg.clone(),
            }));
        }
    }

    /// Get the maximum estimated tokens across all inputs.
    ///
    /// Useful for determining padding requirements.
    #[must_use]
    pub fn max_tokens(&self) -> usize {
        // We need to estimate from inputs since we don't store per-request tokens
        self.inputs.iter()
            .map(|input| match input {
                ModelInput::Text { content, instruction } => {
                    let total_len = content.len() + instruction.as_ref().map_or(0, |s| s.len());
                    (total_len / 4).max(1)
                }
                ModelInput::Code { content, .. } => (content.len() / 3).max(1),
                ModelInput::Image { .. } | ModelInput::Audio { .. } => 100,
            })
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // BATCH REQUEST TESTS
    // ============================================================

    #[test]
    fn test_batch_request_new_creates_valid_request() {
        let input = ModelInput::text("Hello, world!").unwrap();
        let (request, _rx) = BatchRequest::new(input.clone(), ModelId::Semantic);

        assert_eq!(request.model_id, ModelId::Semantic);
        assert_eq!(request.priority, 0);
        assert!(!request.id.is_nil());
    }

    #[test]
    fn test_batch_request_with_priority() {
        let input = ModelInput::text("Urgent request").unwrap();
        let (request, _rx) = BatchRequest::with_priority(input, ModelId::Semantic, 100);

        assert_eq!(request.priority, 100);
    }

    #[test]
    fn test_batch_request_elapsed_increases() {
        let input = ModelInput::text("Test").unwrap();
        let (request, _rx) = BatchRequest::new(input, ModelId::Semantic);

        let elapsed1 = request.elapsed();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed2 = request.elapsed();

        assert!(elapsed2 > elapsed1);
    }

    #[test]
    fn test_batch_request_estimated_tokens_text() {
        // 12 characters / 4 = 3 tokens
        let input = ModelInput::text("Hello world!").unwrap();
        let (request, _rx) = BatchRequest::new(input, ModelId::Semantic);

        assert_eq!(request.estimated_tokens(), 3);
    }

    #[test]
    fn test_batch_request_estimated_tokens_text_with_instruction() {
        // "Hello" (5) + "query:" (6) = 11 / 4 = 2
        let input = ModelInput::text_with_instruction("Hello", "query:").unwrap();
        let (request, _rx) = BatchRequest::new(input, ModelId::Semantic);

        assert_eq!(request.estimated_tokens(), 2);
    }

    #[test]
    fn test_batch_request_estimated_tokens_code() {
        // 12 characters / 3 = 4 tokens
        let input = ModelInput::code("fn main() {}", "rust").unwrap();
        let (request, _rx) = BatchRequest::new(input, ModelId::Code);

        assert_eq!(request.estimated_tokens(), 4);
    }

    #[test]
    fn test_batch_request_estimated_tokens_image() {
        use crate::types::ImageFormat;
        let input = ModelInput::image(vec![1, 2, 3, 4], ImageFormat::Png).unwrap();
        let (request, _rx) = BatchRequest::new(input, ModelId::Multimodal);

        assert_eq!(request.estimated_tokens(), 100);
    }

    #[test]
    fn test_batch_request_estimated_tokens_minimum_one() {
        // Very short text still returns at least 1
        let input = ModelInput::text("Hi").unwrap();
        let (request, _rx) = BatchRequest::new(input, ModelId::Semantic);

        assert!(request.estimated_tokens() >= 1);
    }

    // ============================================================
    // BATCH QUEUE STATS TESTS
    // ============================================================

    #[test]
    fn test_batch_queue_stats_default() {
        let stats = BatchQueueStats::default();
        let summary = stats.summary();

        assert_eq!(summary.requests_received, 0);
        assert_eq!(summary.batches_processed, 0);
        assert_eq!(summary.requests_completed, 0);
        assert_eq!(summary.requests_failed, 0);
        assert_eq!(summary.avg_batch_size, 0.0);
    }

    #[test]
    fn test_batch_queue_stats_record_request() {
        let stats = BatchQueueStats::default();
        stats.record_request();
        stats.record_request();

        assert_eq!(stats.summary().requests_received, 2);
    }

    #[test]
    fn test_batch_queue_stats_record_batch() {
        let stats = BatchQueueStats::default();
        stats.record_batch(10, 5000);
        stats.record_batch(20, 3000);

        let summary = stats.summary();
        assert_eq!(summary.batches_processed, 2);
        assert!((summary.avg_batch_size - 15.0).abs() < 0.001);
        assert_eq!(summary.avg_wait_time_us, 4000); // (5000 + 3000) / 2
    }

    #[test]
    fn test_batch_queue_stats_record_completion() {
        let stats = BatchQueueStats::default();
        stats.record_completion(true);
        stats.record_completion(true);
        stats.record_completion(false);

        let summary = stats.summary();
        assert_eq!(summary.requests_completed, 2);
        assert_eq!(summary.requests_failed, 1);
    }

    #[test]
    fn test_batch_queue_stats_reset() {
        let stats = BatchQueueStats::default();
        stats.record_request();
        stats.record_batch(10, 5000);
        stats.record_completion(true);

        stats.reset();

        let summary = stats.summary();
        assert_eq!(summary.requests_received, 0);
        assert_eq!(summary.batches_processed, 0);
    }

    #[test]
    fn test_batch_queue_stats_clone() {
        let stats = BatchQueueStats::default();
        stats.record_request();
        stats.record_batch(10, 5000);

        let cloned = stats.clone();
        let summary = cloned.summary();
        assert_eq!(summary.requests_received, 1);
        assert_eq!(summary.batches_processed, 1);
    }

    // ============================================================
    // BATCH QUEUE TESTS
    // ============================================================

    #[test]
    fn test_batch_queue_new() {
        let config = BatchConfig::default();
        let queue = BatchQueue::new(ModelId::Semantic, config);

        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        assert_eq!(queue.model_id(), ModelId::Semantic);
    }

    #[test]
    fn test_batch_queue_push() {
        let config = BatchConfig::default();
        let mut queue = BatchQueue::new(ModelId::Semantic, config);

        let input = ModelInput::text("Test").unwrap();
        let (request, _rx) = BatchRequest::new(input, ModelId::Semantic);
        queue.push(request);

        assert_eq!(queue.len(), 1);
        assert!(!queue.is_empty());
    }

    #[test]
    fn test_batch_queue_should_flush_empty() {
        let config = BatchConfig::default();
        let queue = BatchQueue::new(ModelId::Semantic, config);

        assert!(!queue.should_flush());
    }

    #[test]
    fn test_batch_queue_should_flush_max_size() {
        let mut config = BatchConfig::default();
        config.max_batch_size = 2;

        let mut queue = BatchQueue::new(ModelId::Semantic, config);

        // Add first request - should not flush
        let input1 = ModelInput::text("Test 1").unwrap();
        let (request1, _rx1) = BatchRequest::new(input1, ModelId::Semantic);
        queue.push(request1);
        assert!(!queue.should_flush());

        // Add second request - should flush (max_batch_size = 2)
        let input2 = ModelInput::text("Test 2").unwrap();
        let (request2, _rx2) = BatchRequest::new(input2, ModelId::Semantic);
        queue.push(request2);
        assert!(queue.should_flush());
    }

    #[test]
    fn test_batch_queue_should_flush_timeout() {
        let mut config = BatchConfig::default();
        config.max_wait_ms = 10; // 10ms timeout

        let mut queue = BatchQueue::new(ModelId::Semantic, config);

        let input = ModelInput::text("Test").unwrap();
        let (request, _rx) = BatchRequest::new(input, ModelId::Semantic);
        queue.push(request);

        // Should not flush immediately
        assert!(!queue.should_flush());

        // Wait for timeout
        std::thread::sleep(std::time::Duration::from_millis(15));
        assert!(queue.should_flush());
    }

    #[test]
    fn test_batch_queue_drain_batch_empty() {
        let config = BatchConfig::default();
        let mut queue = BatchQueue::new(ModelId::Semantic, config);

        assert!(queue.drain_batch().is_none());
    }

    #[test]
    fn test_batch_queue_drain_batch_returns_batch() {
        let config = BatchConfig::default();
        let mut queue = BatchQueue::new(ModelId::Semantic, config);

        let input = ModelInput::text("Test").unwrap();
        let (request, _rx) = BatchRequest::new(input, ModelId::Semantic);
        queue.push(request);

        let batch = queue.drain_batch();
        assert!(batch.is_some());

        let batch = batch.unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.model_id, ModelId::Semantic);
    }

    #[test]
    fn test_batch_queue_drain_batch_respects_max_size() {
        let mut config = BatchConfig::default();
        config.max_batch_size = 2;

        let mut queue = BatchQueue::new(ModelId::Semantic, config);

        // Add 3 requests
        for i in 0..3 {
            let input = ModelInput::text(format!("Test {}", i)).unwrap();
            let (request, _rx) = BatchRequest::new(input, ModelId::Semantic);
            queue.push(request);
        }

        // First drain should get 2
        let batch1 = queue.drain_batch().unwrap();
        assert_eq!(batch1.len(), 2);

        // Second drain should get 1
        let batch2 = queue.drain_batch().unwrap();
        assert_eq!(batch2.len(), 1);

        // Third drain should get none
        assert!(queue.drain_batch().is_none());
    }

    #[test]
    fn test_batch_queue_drain_batch_sorts_by_length() {
        let mut config = BatchConfig::default();
        config.sort_by_length = true;

        let mut queue = BatchQueue::new(ModelId::Semantic, config);

        // Add requests with different lengths
        let long_input = ModelInput::text("This is a very long sentence that has many words").unwrap();
        let short_input = ModelInput::text("Short").unwrap();
        let medium_input = ModelInput::text("Medium length text").unwrap();

        let (req1, _) = BatchRequest::new(long_input, ModelId::Semantic);
        let (req2, _) = BatchRequest::new(short_input, ModelId::Semantic);
        let (req3, _) = BatchRequest::new(medium_input, ModelId::Semantic);

        queue.push(req1);
        queue.push(req2);
        queue.push(req3);

        let batch = queue.drain_batch().unwrap();

        // After sorting by tokens, short should be first
        // "Short" (5 chars / 4 = 1 token) < "Medium length text" (18 chars / 4 = 4 tokens)
        // < "This is a very long..." (49 chars / 4 = 12 tokens)
        assert_eq!(batch.inputs.len(), 3);

        // Verify ordering by checking content
        if let ModelInput::Text { content, .. } = &batch.inputs[0] {
            assert_eq!(content, "Short");
        }
    }

    #[test]
    fn test_batch_queue_oldest_wait_time() {
        let config = BatchConfig::default();
        let mut queue = BatchQueue::new(ModelId::Semantic, config);

        // Empty queue
        assert!(queue.oldest_wait_time().is_none());

        // Add request
        let input = ModelInput::text("Test").unwrap();
        let (request, _rx) = BatchRequest::new(input, ModelId::Semantic);
        queue.push(request);

        std::thread::sleep(std::time::Duration::from_millis(10));

        let wait = queue.oldest_wait_time();
        assert!(wait.is_some());
        assert!(wait.unwrap() >= std::time::Duration::from_millis(10));
    }

    #[tokio::test]
    async fn test_batch_queue_cancel_all() {
        let config = BatchConfig::default();
        let mut queue = BatchQueue::new(ModelId::Semantic, config);

        let input = ModelInput::text("Test").unwrap();
        let (request, rx) = BatchRequest::new(input, ModelId::Semantic);
        queue.push(request);

        queue.cancel_all("Shutdown");

        assert!(queue.is_empty());

        // Receiver should get error
        let result = rx.await.unwrap();
        assert!(result.is_err());
    }

    // ============================================================
    // BATCH TESTS
    // ============================================================

    #[test]
    fn test_batch_new() {
        let batch = Batch::new(ModelId::Semantic);

        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert_eq!(batch.model_id, ModelId::Semantic);
        assert!(!batch.id.is_nil());
        assert_eq!(batch.total_tokens, 0);
    }

    #[test]
    fn test_batch_add() {
        let mut batch = Batch::new(ModelId::Semantic);

        let input = ModelInput::text("Hello world!").unwrap();
        let (request, _rx) = BatchRequest::new(input, ModelId::Semantic);

        let expected_tokens = request.estimated_tokens();
        let request_id = request.id;

        batch.add(request);

        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
        assert_eq!(batch.total_tokens, expected_tokens);
        assert_eq!(batch.request_ids[0], request_id);
    }

    #[test]
    fn test_batch_max_tokens() {
        let mut batch = Batch::new(ModelId::Semantic);

        // Short text: 5/4 = 1 token
        let input1 = ModelInput::text("Short").unwrap();
        let (req1, _) = BatchRequest::new(input1, ModelId::Semantic);
        batch.add(req1);

        // Long text: 40/4 = 10 tokens
        let input2 = ModelInput::text("This is a much longer piece of text here").unwrap();
        let (req2, _) = BatchRequest::new(input2, ModelId::Semantic);
        batch.add(req2);

        assert_eq!(batch.max_tokens(), 10);
    }

    #[tokio::test]
    async fn test_batch_complete() {
        let mut batch = Batch::new(ModelId::Semantic);

        let input = ModelInput::text("Test").unwrap();
        let (request, rx) = BatchRequest::new(input, ModelId::Semantic);
        batch.add(request);

        // Create a result
        let embedding = ModelEmbedding::new(ModelId::Semantic, vec![0.1; 1024], 100);

        batch.complete(vec![Ok(embedding.clone())]);

        // Receiver should get the result
        let result = rx.await.unwrap();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().vector, embedding.vector);
    }

    #[tokio::test]
    async fn test_batch_complete_multiple() {
        let mut batch = Batch::new(ModelId::Semantic);

        let input1 = ModelInput::text("Test 1").unwrap();
        let (req1, rx1) = BatchRequest::new(input1, ModelId::Semantic);
        batch.add(req1);

        let input2 = ModelInput::text("Test 2").unwrap();
        let (req2, rx2) = BatchRequest::new(input2, ModelId::Semantic);
        batch.add(req2);

        // Create results
        let emb1 = ModelEmbedding::new(ModelId::Semantic, vec![0.1; 1024], 100);
        let emb2 = ModelEmbedding::new(ModelId::Semantic, vec![0.2; 1024], 200);

        batch.complete(vec![Ok(emb1.clone()), Ok(emb2.clone())]);

        // Both receivers should get results
        let result1 = rx1.await.unwrap().unwrap();
        let result2 = rx2.await.unwrap().unwrap();

        assert_eq!(result1.vector[0], 0.1);
        assert_eq!(result2.vector[0], 0.2);
    }

    #[tokio::test]
    async fn test_batch_fail() {
        let mut batch = Batch::new(ModelId::Semantic);

        let input = ModelInput::text("Test").unwrap();
        let (request, rx) = BatchRequest::new(input, ModelId::Semantic);
        batch.add(request);

        batch.fail("Test error");

        // Receiver should get error
        let result = rx.await.unwrap();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_batch_fail_multiple() {
        let mut batch = Batch::new(ModelId::Semantic);

        let input1 = ModelInput::text("Test 1").unwrap();
        let (req1, rx1) = BatchRequest::new(input1, ModelId::Semantic);
        batch.add(req1);

        let input2 = ModelInput::text("Test 2").unwrap();
        let (req2, rx2) = BatchRequest::new(input2, ModelId::Semantic);
        batch.add(req2);

        batch.fail("Batch failed");

        // Both should get the same error
        assert!(rx1.await.unwrap().is_err());
        assert!(rx2.await.unwrap().is_err());
    }

    #[test]
    fn test_batch_elapsed() {
        let batch = Batch::new(ModelId::Semantic);
        std::thread::sleep(std::time::Duration::from_millis(10));

        assert!(batch.elapsed() >= std::time::Duration::from_millis(10));
    }

    // ============================================================
    // INTEGRATION TESTS
    // ============================================================

    #[tokio::test]
    async fn test_full_batch_workflow() {
        let mut config = BatchConfig::default();
        config.max_batch_size = 3;

        let mut queue = BatchQueue::new(ModelId::Semantic, config);

        // Add 3 requests
        let mut receivers = Vec::new();
        for i in 0..3 {
            let input = ModelInput::text(format!("Request {}", i)).unwrap();
            let (request, rx) = BatchRequest::new(input, ModelId::Semantic);
            queue.push(request);
            receivers.push(rx);
        }

        assert!(queue.should_flush());

        // Drain batch
        let batch = queue.drain_batch().unwrap();
        assert_eq!(batch.len(), 3);

        // Complete with results
        let results: Vec<EmbeddingResult<ModelEmbedding>> = (0..3)
            .map(|i| {
                Ok(ModelEmbedding::new(
                    ModelId::Semantic,
                    vec![i as f32; 1024],
                    100 + i as u64,
                ))
            })
            .collect();

        batch.complete(results);

        // All receivers should get results
        for (i, rx) in receivers.into_iter().enumerate() {
            let result = rx.await.unwrap().unwrap();
            assert_eq!(result.vector[0], i as f32);
        }
    }

    #[test]
    fn test_stats_updated_through_workflow() {
        let mut config = BatchConfig::default();
        config.max_batch_size = 2;

        let mut queue = BatchQueue::new(ModelId::Semantic, config);

        // Add requests
        for _ in 0..2 {
            let input = ModelInput::text("Test").unwrap();
            let (request, _rx) = BatchRequest::new(input, ModelId::Semantic);
            queue.push(request);
        }

        // Check requests received
        assert_eq!(queue.stats_summary().requests_received, 2);

        // Drain batch
        let _batch = queue.drain_batch().unwrap();

        // Check batches processed
        assert_eq!(queue.stats_summary().batches_processed, 1);
        assert!((queue.stats_summary().avg_batch_size - 2.0).abs() < 0.001);
    }
}
