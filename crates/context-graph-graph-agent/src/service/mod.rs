//! Graph discovery service orchestrating the discovery pipeline.
//!
//! The service combines:
//! - Scanner: Finds candidate memory pairs
//! - LLM: Analyzes relationships
//! - Activator: Stores confirmed relationships
//!
//! # Pipeline Flow
//!
//! ```text
//! Memories -> Scanner -> Candidates -> LLM Analysis -> Activator -> Graph Edges
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use context_graph_causal_agent::CausalDiscoveryLLM;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::activator::{ActivatorConfig, E8Activator, GraphStorage};
use crate::error::{GraphAgentError, GraphAgentResult};
use crate::llm::GraphRelationshipLLM;
use crate::scanner::{MemoryScanner, ScannerConfig};
use crate::types::MemoryForGraphAnalysis;

/// Configuration for the graph discovery service.
#[derive(Debug, Clone)]
pub struct GraphDiscoveryConfig {
    /// Interval between discovery cycles (background mode).
    pub interval: Duration,

    /// Maximum candidates to process per cycle.
    pub batch_size: usize,

    /// Minimum confidence for relationships.
    pub min_confidence: f32,

    /// Scanner configuration.
    pub scanner_config: ScannerConfig,

    /// Activator configuration.
    pub activator_config: ActivatorConfig,

    /// Whether to skip already analyzed pairs.
    pub skip_analyzed: bool,
}

impl Default for GraphDiscoveryConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(3600), // 1 hour
            batch_size: 50,
            min_confidence: 0.7,
            scanner_config: ScannerConfig::default(),
            activator_config: ActivatorConfig::default(),
            skip_analyzed: true,
        }
    }
}

/// Result of a single discovery cycle.
#[derive(Debug, Clone)]
pub struct DiscoveryCycleResult {
    /// When the cycle started.
    pub started_at: DateTime<Utc>,
    /// When the cycle completed.
    pub completed_at: DateTime<Utc>,
    /// Total duration.
    pub duration: Duration,
    /// Number of candidate pairs found.
    pub candidates_found: usize,
    /// Number of relationships confirmed by LLM.
    pub relationships_confirmed: usize,
    /// Number of relationships rejected by LLM.
    pub relationships_rejected: usize,
    /// Number of embeddings generated.
    pub embeddings_generated: usize,
    /// Number of graph edges created.
    pub edges_created: usize,
    /// Number of errors encountered.
    pub errors: usize,
    /// Error messages.
    pub error_messages: Vec<String>,
}

/// Service status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceStatus {
    /// Service is stopped.
    Stopped,
    /// Service is starting.
    Starting,
    /// Service is running.
    Running,
    /// Service is stopping.
    Stopping,
}

/// Graph relationship discovery service.
///
/// Orchestrates the discovery pipeline:
/// 1. Scans memories for candidate pairs
/// 2. Analyzes pairs with LLM for relationships
/// 3. Stores confirmed relationships via activator
pub struct GraphDiscoveryService {
    config: GraphDiscoveryConfig,
    llm: Arc<GraphRelationshipLLM>,
    scanner: RwLock<MemoryScanner>,
    activator: Arc<E8Activator>,
    running: AtomicBool,
    status: RwLock<ServiceStatus>,
    last_result: RwLock<Option<DiscoveryCycleResult>>,
    shutdown_tx: RwLock<Option<mpsc::Sender<()>>>,
}

impl GraphDiscoveryService {
    /// Create a new graph discovery service.
    ///
    /// # Arguments
    /// * `shared_llm` - Shared CausalDiscoveryLLM from causal-agent
    pub fn new(shared_llm: Arc<CausalDiscoveryLLM>) -> Self {
        Self::with_config(shared_llm, GraphDiscoveryConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(shared_llm: Arc<CausalDiscoveryLLM>, config: GraphDiscoveryConfig) -> Self {
        let graph_llm = Arc::new(GraphRelationshipLLM::new(shared_llm));
        let activator = Arc::new(E8Activator::with_config(config.activator_config.clone()));
        let scanner = MemoryScanner::with_config(config.scanner_config.clone());

        Self {
            config,
            llm: graph_llm,
            scanner: RwLock::new(scanner),
            activator,
            running: AtomicBool::new(false),
            status: RwLock::new(ServiceStatus::Stopped),
            last_result: RwLock::new(None),
            shutdown_tx: RwLock::new(None),
        }
    }

    /// Check if the service is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get current service status.
    pub fn status(&self) -> ServiceStatus {
        *self.status.read()
    }

    /// Get the last discovery cycle result.
    pub fn last_result(&self) -> Option<DiscoveryCycleResult> {
        self.last_result.read().clone()
    }

    /// Get the graph storage.
    pub fn graph(&self) -> Arc<RwLock<GraphStorage>> {
        self.activator.graph()
    }

    /// Get access to the underlying LLM for direct analysis.
    ///
    /// This allows tools like validate_graph_link to analyze single pairs
    /// without going through the full discovery cycle.
    pub fn llm(&self) -> &Arc<GraphRelationshipLLM> {
        &self.llm
    }

    /// Run a single discovery cycle manually.
    ///
    /// # Arguments
    /// * `memories` - Memories to analyze for relationships
    pub async fn run_discovery_cycle(
        &self,
        memories: &[MemoryForGraphAnalysis],
    ) -> GraphAgentResult<DiscoveryCycleResult> {
        let started_at = Utc::now();
        let mut relationships_confirmed = 0;
        let mut relationships_rejected = 0;
        let mut errors = 0;
        let mut error_messages = Vec::new();

        info!(
            memory_count = memories.len(),
            "Starting graph discovery cycle"
        );

        // Step 1: Find candidate pairs
        let candidates = {
            let mut scanner = self.scanner.write();
            scanner.find_candidates(memories)?
        };
        let candidates_found = candidates.len();

        debug!(candidates_found, "Found candidate pairs");

        if candidates.is_empty() {
            let completed_at = Utc::now();
            let duration = (completed_at - started_at)
                .to_std()
                .unwrap_or(Duration::ZERO);

            let result = DiscoveryCycleResult {
                started_at,
                completed_at,
                duration,
                candidates_found: 0,
                relationships_confirmed: 0,
                relationships_rejected: 0,
                embeddings_generated: 0,
                edges_created: 0,
                errors: 0,
                error_messages: Vec::new(),
            };

            *self.last_result.write() = Some(result.clone());
            return Ok(result);
        }

        // Step 2: Analyze candidates with LLM (batch processing)
        let batch_size = self.config.batch_size.min(candidates.len());
        let batch = &candidates[..batch_size];

        for candidate in batch {
            // Analyze relationship
            let analysis = match self
                .llm
                .analyze_relationship(&candidate.memory_a_content, &candidate.memory_b_content)
                .await
            {
                Ok(result) => result,
                Err(e) => {
                    warn!(
                        memory_a = %candidate.memory_a_id,
                        memory_b = %candidate.memory_b_id,
                        error = %e,
                        "Failed to analyze candidate"
                    );
                    errors += 1;
                    error_messages.push(format!(
                        "Analysis failed for {}-{}: {}",
                        candidate.memory_a_id, candidate.memory_b_id, e
                    ));
                    continue;
                }
            };

            // Mark as analyzed
            if self.config.skip_analyzed {
                let mut scanner = self.scanner.write();
                scanner.mark_analyzed(candidate.memory_a_id, candidate.memory_b_id);
            }

            // Check if relationship was confirmed
            if !analysis.has_connection || analysis.confidence < self.config.min_confidence {
                relationships_rejected += 1;
                debug!(
                    memory_a = %candidate.memory_a_id,
                    memory_b = %candidate.memory_b_id,
                    has_connection = analysis.has_connection,
                    confidence = analysis.confidence,
                    "Relationship rejected"
                );
                continue;
            }

            relationships_confirmed += 1;

            // Step 3: Activate the relationship
            match self
                .activator
                .activate_relationship(
                    candidate.memory_a_id,
                    candidate.memory_b_id,
                    &candidate.memory_a_content,
                    &candidate.memory_b_content,
                    &analysis,
                )
                .await
            {
                Ok(_) => {
                    info!(
                        memory_a = %candidate.memory_a_id,
                        memory_b = %candidate.memory_b_id,
                        relationship_type = ?analysis.relationship_type,
                        direction = ?analysis.direction,
                        confidence = analysis.confidence,
                        "Relationship activated"
                    );
                }
                Err(e) => {
                    warn!(
                        memory_a = %candidate.memory_a_id,
                        memory_b = %candidate.memory_b_id,
                        error = %e,
                        "Failed to activate relationship"
                    );
                    errors += 1;
                    error_messages.push(format!(
                        "Activation failed for {}-{}: {}",
                        candidate.memory_a_id, candidate.memory_b_id, e
                    ));
                }
            }
        }

        let completed_at = Utc::now();
        let duration = (completed_at - started_at)
            .to_std()
            .unwrap_or(Duration::ZERO);

        let stats = self.activator.stats();

        let result = DiscoveryCycleResult {
            started_at,
            completed_at,
            duration,
            candidates_found,
            relationships_confirmed,
            relationships_rejected,
            embeddings_generated: stats.embeddings_generated,
            edges_created: stats.edges_created,
            errors,
            error_messages,
        };

        info!(
            candidates_found,
            relationships_confirmed,
            relationships_rejected,
            edges_created = stats.edges_created,
            duration_ms = duration.as_millis(),
            "Graph discovery cycle complete"
        );

        *self.last_result.write() = Some(result.clone());
        Ok(result)
    }

    /// Start the background discovery loop.
    pub async fn start(self: Arc<Self>) -> GraphAgentResult<()> {
        if self.is_running() {
            return Err(GraphAgentError::ServiceAlreadyRunning);
        }

        *self.status.write() = ServiceStatus::Starting;

        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        *self.shutdown_tx.write() = Some(shutdown_tx);

        self.running.store(true, Ordering::SeqCst);
        *self.status.write() = ServiceStatus::Running;

        info!(
            interval_secs = self.config.interval.as_secs(),
            "Starting graph discovery background loop"
        );

        let service = Arc::clone(&self);

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        info!("Graph discovery service received shutdown signal");
                        break;
                    }
                    _ = tokio::time::sleep(service.config.interval) => {
                        // In background mode, we would load memories from storage
                        // For now, just log that a cycle would run
                        debug!("Background discovery cycle would run here");
                    }
                }
            }

            service.running.store(false, Ordering::SeqCst);
            *service.status.write() = ServiceStatus::Stopped;
        });

        Ok(())
    }

    /// Stop the background discovery loop.
    pub async fn stop(&self) -> GraphAgentResult<()> {
        if !self.is_running() {
            return Ok(());
        }

        *self.status.write() = ServiceStatus::Stopping;

        if let Some(tx) = self.shutdown_tx.write().take() {
            let _ = tx.send(()).await;
        }

        // Wait for shutdown
        while self.is_running() {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        info!("Graph discovery service stopped");
        Ok(())
    }

    /// Get scanner statistics.
    pub fn scanner_analyzed_count(&self) -> usize {
        self.scanner.read().analyzed_count()
    }

    /// Reset scanner's analyzed pairs tracking.
    pub fn reset_scanner(&self) {
        self.scanner.write().clear_analyzed();
    }

    /// Get activator statistics.
    pub fn activator_stats(&self) -> crate::activator::ActivationStats {
        self.activator.stats()
    }

    /// Reset activator statistics.
    pub fn reset_activator_stats(&self) {
        self.activator.reset_stats();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = GraphDiscoveryConfig::default();
        assert_eq!(config.interval, Duration::from_secs(3600));
        assert_eq!(config.batch_size, 50);
        assert!((config.min_confidence - 0.7).abs() < 0.01);
    }

    // Integration tests would require a loaded LLM
    // which we skip in unit tests
}
