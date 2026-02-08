//! Background Causal Discovery Service.
//!
//! Coordinates the causal discovery pipeline:
//! 1. Periodically scans memories for candidate pairs
//! 2. Analyzes candidates using the local LLM
//! 3. Activates E5 embeddings for confirmed relationships
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │              CausalDiscoveryService                         │
//! ├────────────────────────────────────────────────────────────┤
//! │                                                            │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
//! │  │   Scanner   │───▶│     LLM     │───▶│  Activator  │   │
//! │  └─────────────┘    └─────────────┘    └─────────────┘   │
//! │                                                            │
//! │  ┌─────────────────────────────────────────────────────┐  │
//! │  │                 Background Loop                      │  │
//! │  │   sleep(interval) → scan → analyze → activate        │  │
//! │  └─────────────────────────────────────────────────────┘  │
//! │                                                            │
//! └────────────────────────────────────────────────────────────┘
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use context_graph_core::causal::CausalGraph;
use context_graph_embeddings::models::CausalModel;

use crate::activator::{ActivatorConfig, E5EmbedderActivator};
use crate::error::{CausalAgentError, CausalAgentResult};
use crate::llm::{CausalDiscoveryLLM, LlmConfig};
use crate::scanner::{MemoryScanner, ScannerConfig};
use crate::types::{CausalCandidate, MemoryForAnalysis};

/// Configuration for the Causal Discovery Service.
#[derive(Debug, Clone)]
pub struct CausalDiscoveryConfig {
    /// Interval between discovery cycles.
    pub interval: Duration,

    /// Maximum pairs to analyze per cycle.
    pub batch_size: usize,

    /// Minimum LLM confidence to accept a relationship.
    pub min_confidence: f32,

    /// Whether to skip already-analyzed pairs.
    pub skip_analyzed: bool,

    /// LLM configuration (includes model path).
    pub llm_config: LlmConfig,

    /// Scanner configuration.
    pub scanner_config: ScannerConfig,

    /// Activator configuration.
    pub activator_config: ActivatorConfig,
}

impl Default for CausalDiscoveryConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(3600), // 1 hour
            batch_size: 50,
            min_confidence: 0.7,
            skip_analyzed: true,
            llm_config: LlmConfig::default(),
            scanner_config: ScannerConfig::default(),
            activator_config: ActivatorConfig::default(),
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

    /// Duration of the cycle.
    pub duration: Duration,

    /// Number of candidate pairs found.
    pub candidates_found: usize,

    /// Number of relationships confirmed by LLM.
    pub relationships_confirmed: usize,

    /// Number of relationships rejected by LLM.
    pub relationships_rejected: usize,

    /// Number of E5 embeddings generated.
    pub embeddings_generated: usize,

    /// Number of graph edges created.
    pub edges_created: usize,

    /// Number of errors encountered.
    pub errors: usize,

    /// Error messages (if any).
    pub error_messages: Vec<String>,
}

impl Default for DiscoveryCycleResult {
    fn default() -> Self {
        Self {
            started_at: Utc::now(),
            completed_at: Utc::now(),
            duration: Duration::ZERO,
            candidates_found: 0,
            relationships_confirmed: 0,
            relationships_rejected: 0,
            embeddings_generated: 0,
            edges_created: 0,
            errors: 0,
            error_messages: Vec::new(),
        }
    }
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

/// Background service for causal discovery.
pub struct CausalDiscoveryService {
    /// Configuration.
    config: CausalDiscoveryConfig,

    /// Local LLM for causal analysis.
    llm: Arc<CausalDiscoveryLLM>,

    /// Memory scanner.
    scanner: RwLock<MemoryScanner>,

    /// E5 activator.
    activator: Arc<E5EmbedderActivator>,

    /// Causal graph.
    causal_graph: Arc<RwLock<CausalGraph>>,

    /// Whether the service is running.
    running: AtomicBool,

    /// Service status.
    status: RwLock<ServiceStatus>,

    /// Last cycle result.
    last_result: RwLock<Option<DiscoveryCycleResult>>,

    /// Shutdown signal sender.
    shutdown_tx: RwLock<Option<mpsc::Sender<()>>>,

    /// CRIT-05 FIX: Store background task JoinHandle so panics are not silently lost
    /// and stop() can await clean shutdown.
    join_handle: RwLock<Option<tokio::task::JoinHandle<()>>>,
}

impl CausalDiscoveryService {
    /// Create a new service with the given configuration (test mode only).
    ///
    /// # Warning
    ///
    /// This constructor creates an E5EmbedderActivator WITHOUT a CausalModel.
    /// In production (without `test-mode` feature), embedding operations will fail.
    /// Use `with_models()` for production deployments.
    #[cfg_attr(
        not(feature = "test-mode"),
        deprecated(
            since = "0.1.0",
            note = "Use with_models() for production. This constructor creates E5EmbedderActivator without CausalModel."
        )
    )]
    pub async fn new(config: CausalDiscoveryConfig) -> CausalAgentResult<Self> {
        // Create LLM from config
        let llm = CausalDiscoveryLLM::with_config(config.llm_config.clone())?;
        let llm = Arc::new(llm);

        // Create causal graph
        let causal_graph = Arc::new(RwLock::new(CausalGraph::new()));

        // Create scanner
        let scanner = MemoryScanner::with_config(config.scanner_config.clone());

        // Create activator
        let activator_config = ActivatorConfig {
            min_confidence: config.min_confidence,
            ..config.activator_config.clone()
        };
        let activator = E5EmbedderActivator::with_config(causal_graph.clone(), activator_config);
        let activator = Arc::new(activator);

        Ok(Self {
            config,
            llm,
            scanner: RwLock::new(scanner),
            activator,
            causal_graph,
            running: AtomicBool::new(false),
            status: RwLock::new(ServiceStatus::Stopped),
            last_result: RwLock::new(None),
            shutdown_tx: RwLock::new(None),
            join_handle: RwLock::new(None),
        })
    }

    /// Create a new service with models for production use.
    ///
    /// This is the recommended constructor for production deployments.
    /// It properly injects the CausalModel into E5EmbedderActivator for real embeddings.
    ///
    /// # Arguments
    ///
    /// * `shared_llm` - Shared CausalDiscoveryLLM (Qwen2.5-3B) for relationship classification
    /// * `causal_model` - CausalModel for E5 asymmetric embeddings (768D, ~0.4GB VRAM)
    /// * `config` - Service configuration
    ///
    /// # Example
    ///
    /// ```ignore
    /// let causal_model = Arc::new(CausalModel::new(path, config)?);
    /// causal_model.load().await?;
    /// let service = CausalDiscoveryService::with_models(shared_llm, causal_model, config);
    /// ```
    pub fn with_models(
        shared_llm: Arc<CausalDiscoveryLLM>,
        causal_model: Arc<CausalModel>,
        config: CausalDiscoveryConfig,
    ) -> Self {
        // Create causal graph
        let causal_graph = Arc::new(RwLock::new(CausalGraph::new()));

        // Create scanner
        let scanner = MemoryScanner::with_config(config.scanner_config.clone());

        // Create activator with real CausalModel
        let activator_config = ActivatorConfig {
            min_confidence: config.min_confidence,
            ..config.activator_config.clone()
        };
        let activator = E5EmbedderActivator::with_model(
            causal_graph.clone(),
            causal_model,
            activator_config,
        );
        let activator = Arc::new(activator);

        Self {
            config,
            llm: shared_llm,
            scanner: RwLock::new(scanner),
            activator,
            causal_graph,
            running: AtomicBool::new(false),
            status: RwLock::new(ServiceStatus::Stopped),
            last_result: RwLock::new(None),
            shutdown_tx: RwLock::new(None),
            join_handle: RwLock::new(None),
        }
    }

    /// Load the LLM model.
    pub async fn load_model(&self) -> CausalAgentResult<()> {
        info!("Loading LLM model for causal discovery");
        self.llm.load().await
    }

    /// Unload the LLM model.
    pub async fn unload_model(&self) -> CausalAgentResult<()> {
        info!("Unloading LLM model");
        self.llm.unload().await
    }

    /// Check if the LLM model is loaded.
    pub fn is_model_loaded(&self) -> bool {
        self.llm.is_loaded()
    }

    /// Run a single discovery cycle.
    ///
    /// # Arguments
    ///
    /// * `memories` - Memories to scan for causal relationships
    ///
    /// # Returns
    ///
    /// Result of the discovery cycle
    pub async fn run_discovery_cycle(
        &self,
        memories: &[MemoryForAnalysis],
    ) -> CausalAgentResult<DiscoveryCycleResult> {
        let started_at = Utc::now();
        let start = Instant::now();

        let mut result = DiscoveryCycleResult {
            started_at,
            ..Default::default()
        };

        // Ensure model is loaded
        if !self.llm.is_loaded() {
            self.llm.load().await?;
        }

        // 1. Find candidate pairs
        let candidates = {
            let mut scanner = self.scanner.write();
            match scanner.find_candidates(memories) {
                Ok(c) => c,
                Err(CausalAgentError::NoCandidatesFound) => {
                    info!("No candidate pairs found, skipping cycle");
                    result.completed_at = Utc::now();
                    result.duration = start.elapsed();
                    return Ok(result);
                }
                Err(e) => return Err(e),
            }
        };

        result.candidates_found = candidates.len();
        info!(
            candidates = candidates.len(),
            batch_size = self.config.batch_size,
            "Found candidate pairs"
        );

        // 2. Analyze candidates with LLM
        let batch: Vec<_> = candidates
            .iter()
            .take(self.config.batch_size)
            .collect();

        for candidate in batch {
            match self.process_candidate(candidate).await {
                Ok(confirmed) => {
                    if confirmed {
                        result.relationships_confirmed += 1;
                        result.embeddings_generated += 2; // cause + effect
                        result.edges_created += 1;
                    } else {
                        result.relationships_rejected += 1;
                    }
                }
                Err(e) => {
                    warn!(
                        cause = %candidate.cause_memory_id,
                        effect = %candidate.effect_memory_id,
                        error = %e,
                        "Failed to process candidate"
                    );
                    result.errors += 1;
                    result.error_messages.push(e.to_string());
                }
            }
        }

        result.completed_at = Utc::now();
        result.duration = start.elapsed();

        // Store result
        {
            let mut last = self.last_result.write();
            *last = Some(result.clone());
        }

        info!(
            confirmed = result.relationships_confirmed,
            rejected = result.relationships_rejected,
            errors = result.errors,
            duration_ms = result.duration.as_millis(),
            "Discovery cycle complete"
        );

        Ok(result)
    }

    /// Process a single candidate pair.
    async fn process_candidate(&self, candidate: &CausalCandidate) -> CausalAgentResult<bool> {
        debug!(
            cause = %candidate.cause_memory_id,
            effect = %candidate.effect_memory_id,
            initial_score = candidate.initial_score,
            "Processing candidate"
        );

        // Analyze with LLM
        let analysis = self
            .llm
            .analyze_causal_relationship(&candidate.cause_content, &candidate.effect_content)
            .await?;

        if !analysis.has_causal_link || analysis.confidence < self.config.min_confidence {
            debug!(
                has_link = analysis.has_causal_link,
                confidence = analysis.confidence,
                "Rejected candidate"
            );
            return Ok(false);
        }

        // Activate E5 embedding
        self.activator
            .activate_relationship(
                candidate.cause_memory_id,
                candidate.effect_memory_id,
                &candidate.cause_content,
                &candidate.effect_content,
                &analysis,
            )
            .await?;

        Ok(true)
    }

    /// Start the background discovery loop.
    pub async fn start(self: Arc<Self>) -> CausalAgentResult<()> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err(CausalAgentError::ServiceAlreadyRunning);
        }

        {
            let mut status = self.status.write();
            *status = ServiceStatus::Starting;
        }

        info!(
            interval_secs = self.config.interval.as_secs(),
            batch_size = self.config.batch_size,
            "Starting causal discovery service"
        );

        // Create shutdown channel
        let (tx, mut rx) = mpsc::channel(1);
        {
            let mut shutdown = self.shutdown_tx.write();
            *shutdown = Some(tx);
        }

        // Load model
        self.llm.load().await?;

        {
            let mut status = self.status.write();
            *status = ServiceStatus::Running;
        }

        // Background loop
        // CRIT-05 FIX: Store JoinHandle so panics are observable and stop() can await.
        let service = self.clone();
        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = rx.recv() => {
                        info!("Received shutdown signal");
                        break;
                    }
                    _ = tokio::time::sleep(service.config.interval) => {
                        // CRIT-01 FIX: Emit warn! instead of silent debug! so operators
                        // know discovery is NOT running. The background loop is a facade:
                        // it never calls run_discovery_cycle() or fetches memories.
                        warn!(
                            "Causal discovery cycle tick: NOT IMPLEMENTED. \
                             Background loop is running but no discovery occurs. \
                             Use trigger_causal_discovery MCP tool for on-demand analysis."
                        );
                    }
                }
            }

            let mut status = service.status.write();
            *status = ServiceStatus::Stopped;
            service.running.store(false, Ordering::SeqCst);
        });
        *self.join_handle.write() = Some(handle);

        Ok(())
    }

    /// Stop the background discovery loop.
    pub async fn stop(&self) -> CausalAgentResult<()> {
        if !self.running.load(Ordering::SeqCst) {
            return Ok(());
        }

        {
            let mut status = self.status.write();
            *status = ServiceStatus::Stopping;
        }

        info!("Stopping causal discovery service");

        // Send shutdown signal
        // CRIT-04 FIX: Clone sender out of the parking_lot RwLock scope
        // before .await, since parking_lot guards are !Send.
        let tx_clone = {
            let shutdown = self.shutdown_tx.read();
            shutdown.as_ref().cloned()
        };
        if let Some(tx) = tx_clone {
            let _ = tx.send(()).await;
        }

        // Unload model
        self.llm.unload().await?;

        // CRIT-05 FIX: Await the background task JoinHandle with timeout
        // to detect panics and ensure clean shutdown.
        if let Some(handle) = self.join_handle.write().take() {
            match tokio::time::timeout(Duration::from_secs(10), handle).await {
                Ok(Ok(())) => info!("Causal discovery background task completed normally"),
                Ok(Err(e)) => tracing::error!("Causal discovery background task panicked: {:?}", e),
                Err(_) => tracing::error!("Causal discovery background task did not shut down within 10 seconds"),
            }
        }

        Ok(())
    }

    /// Get the current service status.
    pub fn status(&self) -> ServiceStatus {
        *self.status.read()
    }

    /// Check if the service is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get the last cycle result.
    pub fn last_result(&self) -> Option<DiscoveryCycleResult> {
        self.last_result.read().clone()
    }

    /// Get the causal graph.
    pub fn causal_graph(&self) -> &Arc<RwLock<CausalGraph>> {
        &self.causal_graph
    }

    /// Get the configuration.
    pub fn config(&self) -> &CausalDiscoveryConfig {
        &self.config
    }

    /// Get estimated VRAM usage in MB.
    pub fn estimated_vram_mb(&self) -> usize {
        self.llm.estimated_vram_mb()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> CausalDiscoveryConfig {
        // Find workspace root by looking for Cargo.toml
        let mut workspace_root = std::env::current_dir().unwrap();
        while !workspace_root.join("Cargo.toml").exists()
            || !workspace_root.join("models").exists()
        {
            if !workspace_root.pop() {
                // Fall back to relative path if we can't find workspace root
                workspace_root = std::path::PathBuf::from(".");
                break;
            }
        }

        let model_dir = workspace_root.join("models/hermes-2-pro");

        let llm_config = crate::llm::LlmConfig {
            model_path: model_dir.join("Hermes-2-Pro-Mistral-7B.Q5_K_M.gguf"),
            causal_grammar_path: model_dir.join("causal_analysis.gbnf"),
            graph_grammar_path: model_dir.join("graph_relationship.gbnf"),
            validation_grammar_path: model_dir.join("validation.gbnf"),
            ..Default::default()
        };

        CausalDiscoveryConfig {
            interval: Duration::from_secs(1),
            batch_size: 10,
            min_confidence: 0.6,
            llm_config,
            ..Default::default()
        }
    }

    fn create_test_memory(id: u128, content: &str, hours_ago: i64) -> MemoryForAnalysis {
        use chrono::TimeDelta;
        use uuid::Uuid;

        MemoryForAnalysis {
            id: Uuid::from_u128(id),
            content: content.to_string(),
            created_at: Utc::now() - TimeDelta::hours(hours_ago),
            session_id: Some("test-session".to_string()),
            e1_embedding: vec![0.7; 1024],
        }
    }

    #[tokio::test]
    async fn test_service_creation() {
        let config = create_test_config();
        #[allow(deprecated)]
        let service = CausalDiscoveryService::new(config).await;
        assert!(service.is_ok());
    }

    #[tokio::test]
    async fn test_discovery_cycle() {
        let config = create_test_config();

        // Skip test if model not available (e.g., in CI)
        if !config.llm_config.model_path.exists() {
            eprintln!(
                "Skipping test_discovery_cycle: model not found at {:?}",
                config.llm_config.model_path
            );
            return;
        }

        #[allow(deprecated)]
        let service = CausalDiscoveryService::new(config).await.unwrap();

        // Load model
        service.load_model().await.unwrap();

        // Create test memories with causal markers
        let memories = vec![
            create_test_memory(1, "Because of the error, the system crashed", 2),
            create_test_memory(2, "Therefore, users were affected", 1),
        ];

        let result = service.run_discovery_cycle(&memories).await.unwrap();

        // Check that the cycle ran (may find 0 candidates if similarity is too low)
        assert!(result.errors == 0);
    }

    #[tokio::test]
    async fn test_service_status() {
        let config = create_test_config();
        #[allow(deprecated)]
        let service = CausalDiscoveryService::new(config).await.unwrap();

        assert_eq!(service.status(), ServiceStatus::Stopped);
        assert!(!service.is_running());
    }
}
