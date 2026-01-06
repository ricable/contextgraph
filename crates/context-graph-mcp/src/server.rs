//! MCP Server implementation.
//!
//! TASK-S001: Updated to use TeleologicalMemoryStore and MultiArrayEmbeddingProvider.
//! TASK-S003: Added GoalAlignmentCalculator and GoalHierarchy for purpose operations.
//! NO BACKWARDS COMPATIBILITY with legacy MemoryStore.

use std::io::{self, BufRead, Write};
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use context_graph_core::alignment::{DefaultAlignmentCalculator, GoalAlignmentCalculator};
use context_graph_core::config::Config;
use context_graph_core::purpose::GoalHierarchy;
use context_graph_core::stubs::{InMemoryTeleologicalStore, StubMultiArrayProvider, StubUtlProcessor};
use context_graph_core::traits::{MultiArrayEmbeddingProvider, TeleologicalMemoryStore, UtlProcessor};

use crate::handlers::Handlers;
use crate::protocol::{JsonRpcRequest, JsonRpcResponse};

/// MCP Server state.
///
/// TASK-S001: Uses TeleologicalMemoryStore for 13-embedding fingerprint storage.
#[allow(dead_code)]
pub struct McpServer {
    config: Config,
    /// Teleological memory store - stores TeleologicalFingerprint with 13 embeddings.
    teleological_store: Arc<dyn TeleologicalMemoryStore>,
    utl_processor: Arc<dyn UtlProcessor>,
    /// Multi-array embedding provider - generates all 13 embeddings.
    multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>,
    handlers: Handlers,
    initialized: Arc<RwLock<bool>>,
}

impl McpServer {
    /// Create a new MCP server with the given configuration.
    ///
    /// TASK-S001: Creates TeleologicalMemoryStore and MultiArrayEmbeddingProvider.
    pub async fn new(config: Config) -> Result<Self> {
        info!("Initializing MCP Server with TeleologicalMemoryStore...");

        // Create teleological store (in-memory stub for Ghost System phase)
        let teleological_store: Arc<dyn TeleologicalMemoryStore> =
            Arc::new(InMemoryTeleologicalStore::new());
        info!("Created InMemoryTeleologicalStore (13-embedding fingerprint storage)");

        // Create UTL processor
        let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());

        // Create multi-array embedding provider (generates all 13 embeddings)
        let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> =
            Arc::new(StubMultiArrayProvider::new());
        info!("Created StubMultiArrayProvider (13 embedder slots)");

        // TASK-S003: Create alignment calculator and empty goal hierarchy
        let alignment_calculator: Arc<dyn GoalAlignmentCalculator> =
            Arc::new(DefaultAlignmentCalculator::new());
        let goal_hierarchy = GoalHierarchy::new();
        info!("Created DefaultAlignmentCalculator and empty GoalHierarchy");

        let handlers = Handlers::new(
            Arc::clone(&teleological_store),
            Arc::clone(&utl_processor),
            Arc::clone(&multi_array_provider),
            alignment_calculator,
            goal_hierarchy,
        );

        info!("MCP Server initialization complete - TeleologicalFingerprint mode active");

        Ok(Self {
            config,
            teleological_store,
            utl_processor,
            multi_array_provider,
            handlers,
            initialized: Arc::new(RwLock::new(false)),
        })
    }

    /// Run the server, reading from stdin and writing to stdout.
    pub async fn run(&self) -> Result<()> {
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut stdout = stdout.lock();

        info!("Server ready, waiting for requests (TeleologicalMemoryStore mode)...");

        for line in stdin.lock().lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    error!("Error reading stdin: {}", e);
                    break;
                }
            };

            if line.trim().is_empty() {
                continue;
            }

            debug!("Received: {}", line);

            let response = self.handle_request(&line).await;

            // Handle notifications (no response needed)
            if response.id.is_none() && response.result.is_none() && response.error.is_none() {
                debug!("Notification handled, no response needed");
                continue;
            }

            let response_json = serde_json::to_string(&response)?;
            debug!("Sending: {}", response_json);

            // MCP requires newline-delimited JSON on stdout
            writeln!(stdout, "{}", response_json)?;
            stdout.flush()?;

            // Check for shutdown
            if !*self.initialized.read().await {
                // Not initialized yet, continue
            }
        }

        info!("Server shutting down...");
        Ok(())
    }

    /// Handle a single JSON-RPC request.
    async fn handle_request(&self, input: &str) -> JsonRpcResponse {
        // Parse request
        let request: JsonRpcRequest = match serde_json::from_str(input) {
            Ok(r) => r,
            Err(e) => {
                warn!("Failed to parse request: {}", e);
                return JsonRpcResponse::error(
                    None,
                    crate::protocol::error_codes::PARSE_ERROR,
                    format!("Parse error: {}", e),
                );
            }
        };

        // Validate JSON-RPC version
        if request.jsonrpc != "2.0" {
            return JsonRpcResponse::error(
                request.id,
                crate::protocol::error_codes::INVALID_REQUEST,
                "Invalid JSON-RPC version",
            );
        }

        // Dispatch to handler
        self.handlers.dispatch(request).await
    }
}
