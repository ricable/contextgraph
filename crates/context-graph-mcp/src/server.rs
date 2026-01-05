//! MCP Server implementation.

use std::io::{self, BufRead, Write};
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use context_graph_core::config::Config;
use context_graph_core::stubs::{InMemoryStore, StubEmbeddingProvider, StubUtlProcessor};
use context_graph_core::traits::{EmbeddingProvider, MemoryStore, UtlProcessor};

use crate::handlers::Handlers;
use crate::protocol::{JsonRpcRequest, JsonRpcResponse};

/// MCP Server state.
#[allow(dead_code)]
pub struct McpServer {
    config: Config,
    memory_store: Arc<dyn MemoryStore>,
    utl_processor: Arc<dyn UtlProcessor>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    handlers: Handlers,
    initialized: Arc<RwLock<bool>>,
}

impl McpServer {
    /// Create a new MCP server with the given configuration.
    pub async fn new(config: Config) -> Result<Self> {
        info!("Initializing MCP Server components...");

        // Create stub implementations for Ghost System phase
        let memory_store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());

        // BLOCKED: Waiting for TASK-F007 to implement multi-array embedding provider.
        // Using StubEmbeddingProvider until then.
        // TODO(TASK-F007): Replace with real multi-array embedding provider
        let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::new(StubEmbeddingProvider::new());

        let handlers = Handlers::new(
            Arc::clone(&memory_store),
            Arc::clone(&utl_processor),
            Arc::clone(&embedding_provider),
        );

        Ok(Self {
            config,
            memory_store,
            utl_processor,
            embedding_provider,
            handlers,
            initialized: Arc::new(RwLock::new(false)),
        })
    }

    /// Run the server, reading from stdin and writing to stdout.
    pub async fn run(&self) -> Result<()> {
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut stdout = stdout.lock();

        info!("Server ready, waiting for requests...");

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
