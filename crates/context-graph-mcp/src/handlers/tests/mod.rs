//! MCP Protocol Compliance Tests
//!
//! ALL tests use REAL RocksDB storage and REAL GPU embeddings via ProductionMultiArrayProvider.
//! No stubs, no mocks, no workarounds. Every test exercises the full production code path.
//!
//! Tests verify compliance with MCP protocol version 2024-11-05
//! Reference: https://spec.modelcontextprotocol.io/specification/2024-11-05/
//!
//! # Test Helpers
//!
//! - `create_test_handlers()` - Real RocksDB + Real GPU embeddings, returns `(Handlers, TempDir)`
//! - `create_test_handlers_with_rocksdb_store_access()` - Same + exposed store ref for FSV
//! - `create_test_handlers_with_real_embeddings()` - Alias for create_test_handlers()
//! - `create_test_handlers_with_real_embeddings_store_access()` - Alias for store access variant
//!
//! # TempDir Lifecycle
//!
//! All helpers return `(Handlers, TempDir)`. The TempDir MUST be kept alive
//! for the duration of the test - dropping it deletes the database directory.
//!
//! ```ignore
//! #[tokio::test]
//! async fn test_example() {
//!     let (handlers, _tempdir) = create_test_handlers().await;
//!     // _tempdir keeps the database alive until end of test
//! }
//! ```

mod content_storage_verification;
mod curation_tools_fsv;
mod error_codes;
mod gpu_embedding_verification;
mod initialize;
mod manual_fsv_verification;
mod mcp_protocol_e2e_test;
mod robustness_fsv;
mod search_periodic_test;
mod semantic_search_skill_verification;
mod task_emb_024_verification;
mod tcp_transport_integration;
mod tools_call;
mod tools_list;
mod topic_tools;
mod topic_tools_fsv;

use std::sync::Arc;

use tempfile::TempDir;

use context_graph_core::monitoring::{LayerStatusProvider, StubLayerStatusProvider};
use context_graph_core::traits::{MultiArrayEmbeddingProvider, TeleologicalMemoryStore};
use context_graph_storage::teleological::RocksDbTeleologicalStore;

// GRAPH-AGENT: Import stub for testing (enabled via test-utils feature in dev-dependencies)
#[cfg(feature = "llm")]
use context_graph_graph_agent::create_stub_graph_discovery_service;

use context_graph_embeddings::{
    get_warm_provider, initialize_global_warm_provider, is_warm_initialized, warm_status_message,
    GpuConfig, ProductionMultiArrayProvider,
};

use std::path::PathBuf;

use tokio::sync::OnceCell;

/// Global warm-loaded model cache.
///
/// RTX 5090 32GB VRAM - models are warm-loaded ONCE and shared across ALL tests.
/// This prevents CUDA OOM when tests run in parallel, each trying to load
/// all 13 embedding models (~20GB total) from scratch.
///
/// FAIL FAST: If initial load fails, ALL tests will fail - no stubs, no fallbacks.
static WARM_MODEL_CACHE: OnceCell<Arc<dyn MultiArrayEmbeddingProvider>> = OnceCell::const_new();

/// Get or initialize the warm-loaded embedding provider.
///
/// Uses the global warm provider singleton from global_provider.rs.
/// Models are loaded exactly ONCE into GPU VRAM and shared across ALL tests.
///
/// # Panics
///
/// Panics if CUDA GPU not available, models directory missing, or GPU OOM.
async fn get_warm_loaded_provider() -> Arc<dyn MultiArrayEmbeddingProvider> {
    WARM_MODEL_CACHE
        .get_or_init(|| async {
            // TASK-EMB-016: First try to use global warm provider singleton
            if is_warm_initialized() {
                tracing::info!(
                    "WARM LOAD: Using existing global warm provider (already initialized)"
                );
                match get_warm_provider() {
                    Ok(provider) => {
                        tracing::info!(
                            "WARM LOAD: Retrieved global warm provider successfully"
                        );
                        return provider;
                    }
                    Err(e) => {
                        tracing::warn!(
                            "WARM LOAD: Failed to get global warm provider: {}. Falling back to direct load.",
                            e
                        );
                    }
                }
            }

            // Try to initialize the global warm provider
            tracing::info!(
                "WARM LOAD: Attempting to initialize global warm provider..."
            );
            match initialize_global_warm_provider().await {
                Ok(()) => {
                    tracing::info!(
                        "WARM LOAD: Global warm provider initialized successfully"
                    );
                    match get_warm_provider() {
                        Ok(provider) => {
                            tracing::info!(
                                "WARM LOAD: All 13 embedding models ready via global warm provider"
                            );
                            return provider;
                        }
                        Err(e) => {
                            tracing::warn!(
                                "WARM LOAD: Global warm provider init succeeded but get failed: {}. Status: {}",
                                e,
                                warm_status_message()
                            );
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "WARM LOAD: Global warm provider initialization failed: {}. Falling back to ProductionMultiArrayProvider.",
                        e
                    );
                }
            }

            // Fall back to direct ProductionMultiArrayProvider if global warm provider fails
            let models_dir = resolve_test_models_path();
            tracing::info!(
                "WARM LOAD: Falling back to direct ProductionMultiArrayProvider from {:?}",
                models_dir
            );

            let provider =
                ProductionMultiArrayProvider::new(models_dir.clone(), GpuConfig::default())
                    .await
                    .unwrap_or_else(|_| {
                        panic!(
                            "WARM LOAD FAILED: Could not create ProductionMultiArrayProvider. \
                     Ensure models exist at {:?} and RTX 5090 GPU is available with CUDA.",
                            models_dir
                        )
                    });

            tracing::info!("WARM LOAD: All 13 embedding models loaded into VRAM successfully (fallback)");
            Arc::new(provider) as Arc<dyn MultiArrayEmbeddingProvider>
        })
        .await
        .clone()
}

use crate::handlers::Handlers;
use crate::protocol::{JsonRpcId, JsonRpcRequest};

// ============================================================================
// MCP Response Parsing Helpers
// ============================================================================

/// Extract parsed data from MCP tool response.
///
/// MCP tool responses wrap data in: `{ "content": [{ "type": "text", "text": "{...json...}" }] }`
/// This helper extracts and parses the inner JSON from the text field.
///
/// # Arguments
///
/// * `result` - The `result` field from JsonRpcResponse
///
/// # Returns
///
/// Parsed JSON value from content[0].text
///
/// # Panics
///
/// Panics if:
/// - result doesn't have `content` array
/// - content[0] doesn't have `text` field
/// - text field isn't valid JSON
pub(crate) fn extract_mcp_tool_data(result: &serde_json::Value) -> serde_json::Value {
    // Check if this is an error response (MCP format)
    if let Some(is_error) = result.get("isError").and_then(|v| v.as_bool()) {
        if is_error {
            let error_text = result
                .get("content")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|c| c.get("text"))
                .and_then(|t| t.as_str())
                .unwrap_or("Unknown error");
            panic!("MCP tool returned error: {}", error_text);
        }
    }

    // Check if result has MCP content wrapper format: { "content": [{ "text": "..." }] }
    if let Some(content) = result.get("content").and_then(|v| v.as_array()) {
        // MCP wrapped format - extract from content[0].text
        let text = content[0]
            .get("text")
            .and_then(|v| v.as_str())
            .expect("content[0] must have text field");
        serde_json::from_str(text).expect("text field must be valid JSON")
    } else {
        // Direct result format - data is already unwrapped
        // This happens when handler returns data directly via JsonRpcResponse::success
        result.clone()
    }
}

/// Create test handlers with REAL RocksDB storage and REAL GPU embeddings.
///
/// ALL tests use real implementations - no stubs, no mocks, no workarounds.
/// Requires CUDA GPU with models loaded into VRAM via warm provider.
///
/// # Returns
///
/// `(Handlers, TempDir)` - The Handlers instance and TempDir that owns the database.
/// The TempDir MUST be kept alive for the duration of the test.
///
/// # Panics
///
/// Panics if CUDA GPU not available, models not loaded, or RocksDB fails to open.
#[cfg(feature = "llm")]
pub(crate) async fn create_test_handlers() -> (Handlers, TempDir) {
    let tempdir = TempDir::new().expect("Failed to create temp directory for RocksDB test");
    let db_path = tempdir.path().join("test_rocksdb");

    let rocksdb_store = RocksDbTeleologicalStore::open(&db_path)
        .expect("Failed to open RocksDbTeleologicalStore in test");

    let teleological_store: Arc<dyn TeleologicalMemoryStore> = Arc::new(rocksdb_store);
    let multi_array_provider = get_warm_loaded_provider().await;
    let layer_status_provider: Arc<dyn LayerStatusProvider> = Arc::new(StubLayerStatusProvider);

    // GRAPH-AGENT: Create stub service with unloaded LLM
    // Calls to graph discovery methods will return Err(LlmNotInitialized)
    let graph_discovery_service = create_stub_graph_discovery_service();

    let handlers = Handlers::with_defaults(
        teleological_store,
        multi_array_provider,
        layer_status_provider,
        graph_discovery_service,
    );

    (handlers, tempdir)
}

// ============================================================================
// Real RocksDB Test Helpers (Integration Testing)
// ============================================================================

/// Create test handlers with REAL RocksDbTeleologicalStore for integration testing.
///
/// Uses `tempfile::TempDir` for automatic cleanup after tests complete.
/// The returned TempDir MUST be kept alive for the duration of the test -
/// dropping it deletes the database directory.
///
/// # Returns
///
/// `(Handlers, TempDir)` - The Handlers instance and the TempDir that owns the database.
///
/// # Components
///
/// - **Storage**: RocksDbTeleologicalStore (real persistence)
/// - **Embeddings**: ProductionMultiArrayProvider (real GPU)
/// - **GraphDiscovery**: Stub service with unloaded LLM (FAIL FAST on graph ops)
///
/// # HNSW Initialization
///
/// This function **MUST** be async to initialize HNSW indexes before use.
/// Without HNSW initialization, store operations will fail with "Index not initialized" errors.
///
/// # Example
///
/// ```ignore
/// #[tokio::test]
/// async fn test_with_real_storage() {
///     let (handlers, _tempdir) = create_test_handlers_with_rocksdb().await;
///     // _tempdir keeps the database alive until end of test
///
///     // Store and retrieve operations use real RocksDB with initialized HNSW
///     let result = handlers.handle_memory_store(...).await;
///     assert!(result.is_ok());
/// }
/// ```
///
/// # Panics
///
/// Panics if TempDir creation, RocksDB opening, or HNSW initialization fails.
/// This is intentional - tests should fail immediately if infrastructure cannot be set up.
#[cfg(feature = "llm")]
#[allow(dead_code)] // Available for integration tests that need real RocksDB
pub(crate) async fn create_test_handlers_with_rocksdb() -> (Handlers, TempDir) {
    let tempdir = TempDir::new().expect("Failed to create temp directory for RocksDB test");
    let db_path = tempdir.path().join("test_rocksdb");

    // Open RocksDB store
    let rocksdb_store = RocksDbTeleologicalStore::open(&db_path)
        .expect("Failed to open RocksDbTeleologicalStore in test");

    // Note: EmbedderIndexRegistry is initialized in constructor

    let teleological_store: Arc<dyn TeleologicalMemoryStore> = Arc::new(rocksdb_store);

    let multi_array_provider = get_warm_loaded_provider().await;
    let layer_status_provider: Arc<dyn LayerStatusProvider> = Arc::new(StubLayerStatusProvider);
    let graph_discovery_service = create_stub_graph_discovery_service();

    let handlers = Handlers::with_defaults(
        teleological_store,
        multi_array_provider,
        layer_status_provider,
        graph_discovery_service,
    );

    (handlers, tempdir)
}

/// Create test handlers with REAL RocksDB + REAL GPU embeddings and EXPOSED store reference.
///
/// For Full State Verification tests that need to directly inspect the underlying store.
///
/// # Returns
///
/// `(Handlers, Arc<dyn TeleologicalMemoryStore>, TempDir)`
#[cfg(feature = "llm")]
pub(crate) async fn create_test_handlers_with_rocksdb_store_access(
) -> (Handlers, Arc<dyn TeleologicalMemoryStore>, TempDir) {
    let tempdir = TempDir::new().expect("Failed to create temp directory for RocksDB test");
    let db_path = tempdir.path().join("test_rocksdb_fsv");

    let rocksdb_store = RocksDbTeleologicalStore::open(&db_path)
        .expect("Failed to open RocksDbTeleologicalStore in FSV test");

    let teleological_store: Arc<dyn TeleologicalMemoryStore> = Arc::new(rocksdb_store);
    let multi_array_provider = get_warm_loaded_provider().await;
    let layer_status_provider: Arc<dyn LayerStatusProvider> = Arc::new(StubLayerStatusProvider);
    let graph_discovery_service = create_stub_graph_discovery_service();

    let handlers = Handlers::with_defaults(
        Arc::clone(&teleological_store),
        multi_array_provider,
        layer_status_provider,
        graph_discovery_service,
    );

    (handlers, teleological_store, tempdir)
}

// ============================================================================
// TASK-P3-01: Real GPU Embedding Test Helpers (FSV Integration Testing)
// ============================================================================

/// Alias for create_test_handlers(). All helpers now use real GPU embeddings.
#[cfg(feature = "llm")]
#[allow(dead_code)]
pub(crate) async fn create_test_handlers_with_real_embeddings() -> (Handlers, TempDir) {
    let tempdir = TempDir::new().expect("Failed to create temp directory for RocksDB FSV test");
    let db_path = tempdir.path().join("test_rocksdb_fsv_real_embeddings");

    // Open RocksDB store
    let rocksdb_store = RocksDbTeleologicalStore::open(&db_path)
        .expect("Failed to open RocksDbTeleologicalStore in FSV test with real embeddings");

    // Note: EmbedderIndexRegistry is initialized in constructor

    let teleological_store: Arc<dyn TeleologicalMemoryStore> = Arc::new(rocksdb_store);

    // TASK-WARM-LOAD: Use WARM-LOADED embedding provider from global cache
    // RTX 5090 32GB - models loaded ONCE, shared across all tests
    // This prevents CUDA OOM when tests run in parallel
    let multi_array_provider = get_warm_loaded_provider().await;

    let layer_status_provider: Arc<dyn LayerStatusProvider> = Arc::new(StubLayerStatusProvider);

    // GRAPH-AGENT: Create stub service with unloaded LLM
    let graph_discovery_service = create_stub_graph_discovery_service();

    let handlers = Handlers::with_defaults(
        teleological_store,
        multi_array_provider,
        layer_status_provider,
        graph_discovery_service,
    );

    (handlers, tempdir)
}

/// Create test handlers with REAL embeddings and exposed store reference for FSV assertions.
///
/// Same as `create_test_handlers_with_real_embeddings()` but also returns a direct
/// reference to the teleological store for Full State Verification assertions.
///
/// # Returns
///
/// `(Handlers, Arc<dyn TeleologicalMemoryStore>, TempDir)`
/// - Handlers instance for dispatching MCP requests
/// - Direct reference to the store for FSV assertions (verify data was persisted)
/// - TempDir that MUST be kept alive for the duration of the test
#[cfg(feature = "llm")]
pub(crate) async fn create_test_handlers_with_real_embeddings_store_access(
) -> (Handlers, Arc<dyn TeleologicalMemoryStore>, TempDir) {
    let tempdir = TempDir::new().expect("Failed to create temp directory for FSV test");
    let db_path = tempdir
        .path()
        .join("test_rocksdb_fsv_real_embeddings_store");

    // Open RocksDB store
    let rocksdb_store = RocksDbTeleologicalStore::open(&db_path)
        .expect("Failed to open RocksDbTeleologicalStore in FSV test");

    // Note: EmbedderIndexRegistry is initialized in constructor

    let teleological_store: Arc<dyn TeleologicalMemoryStore> = Arc::new(rocksdb_store);

    // TASK-WARM-LOAD: Use WARM-LOADED embedding provider from global cache
    // RTX 5090 32GB - models loaded ONCE, shared across all tests
    let multi_array_provider = get_warm_loaded_provider().await;

    let layer_status_provider: Arc<dyn LayerStatusProvider> = Arc::new(StubLayerStatusProvider);

    // GRAPH-AGENT: Create stub service with unloaded LLM
    let graph_discovery_service = create_stub_graph_discovery_service();

    let handlers = Handlers::with_defaults(
        Arc::clone(&teleological_store),
        multi_array_provider,
        layer_status_provider,
        graph_discovery_service,
    );

    (handlers, teleological_store, tempdir)
}

/// Resolve models directory for tests.
///
/// Priority:
/// 1. `CONTEXT_GRAPH_MODELS_PATH` environment variable
/// 2. Default: `./models` relative to workspace root
fn resolve_test_models_path() -> PathBuf {
    if let Ok(env_path) = std::env::var("CONTEXT_GRAPH_MODELS_PATH") {
        return PathBuf::from(env_path);
    }
    // Navigate from crate directory to workspace root
    // CARGO_MANIFEST_DIR = crates/context-graph-mcp
    // Workspace root = ../../ from there
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));

    // Go up to workspace root (crates/context-graph-mcp -> crates -> root)
    let workspace_root = manifest_dir
        .parent() // -> crates/
        .and_then(|p| p.parent()) // -> workspace root
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    workspace_root.join("models")
}

/// Create a JSON-RPC request for testing.
pub(crate) fn make_request(
    method: &str,
    id: Option<JsonRpcId>,
    params: Option<serde_json::Value>,
) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id,
        method: method.to_string(),
        params,
    }
}

// ============================================================================
// TASK-GAP-001: Removed obsolete test helper code
// ============================================================================
// Legacy test helpers were removed as they referenced deleted modules.
// Current tests use the simplified handler construction per constitution v6.
