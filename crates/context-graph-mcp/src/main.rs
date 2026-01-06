//! Context Graph MCP Server
//!
//! JSON-RPC 2.0 server implementing the Model Context Protocol (MCP)
//! for the Ultimate Context Graph system.
//!
//! # Transport
//!
//! - stdio: Standard input/output (default)
//! - TCP: Future support for networked deployments
//!
//! # Usage
//!
//! ```bash
//! # Run with default configuration
//! context-graph-mcp
//!
//! # Run with custom config
//! context-graph-mcp --config /path/to/config.toml
//!
//! # Run in debug mode
//! RUST_LOG=debug context-graph-mcp
//! ```

mod adapters;
mod handlers;
mod middleware;
mod protocol;
mod server;
mod tools;
mod weights;

use std::env;
use std::io;
use std::path::PathBuf;

use anyhow::Result;
use tracing::info;
use tracing_subscriber::{fmt, EnvFilter};

use context_graph_core::config::Config;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging - CRITICAL: Must write to stderr, not stdout!
    // MCP protocol requires stdout to be exclusively for JSON-RPC messages
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")); // Default to warn to reduce noise

    fmt()
        .with_writer(io::stderr) // CRITICAL: stderr only!
        .with_env_filter(filter)
        .with_target(false) // Cleaner output for MCP
        .init();

    info!("Context Graph MCP Server starting...");

    // Load configuration
    let config_path = env::args()
        .nth(2)
        .filter(|_arg| env::args().nth(1).map(|a| a == "--config").unwrap_or(false))
        .map(PathBuf::from);

    let config = if let Some(path) = config_path {
        info!("Loading configuration from: {:?}", path);
        Config::from_file(&path)?
    } else {
        info!("Using default configuration");
        Config::default()
    };

    info!("Configuration loaded: phase={:?}", config.phase);

    // Create and run server
    let server = server::McpServer::new(config).await?;

    info!("MCP Server initialized, listening on stdio");
    server.run().await?;

    info!("MCP Server shutdown complete");
    Ok(())
}
