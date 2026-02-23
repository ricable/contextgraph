//! RVF Client implementation for context-graph-storage.
//!
//! This is a Rust port of the TypeScript ruvector-client.ts, providing a client
//! for interacting with RVF cognitive containers.
//!
//! # Configuration
//!
//! The client is configured via environment variables:
//! - `RVF_ENDPOINT`: RVF server endpoint (default: "http://localhost:6333")
//! - `RVF_NAMESPACE`: Namespace for vector operations (default: "default")
//! - `RVF_DIMENSION`: Vector dimension (default: 384)
//! - `RVF_API_KEY`: Optional API key for authentication
//!
//! # Features
//!
//! - Create/open RVF stores
//! - Ingest vectors with metadata
//! - Search with progressive recall
//! - COW branch derivation
//! - Store metadata and status queries

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

use super::segments::RvfSegmentStats;

/// Errors that can occur during RVF operations.
#[derive(Debug, Error)]
pub enum RvfClientError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error: {0}")]
    Api(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Store not found: {0}")]
    NotFound(String),

    #[error("Invalid configuration: {0}")]
    Config(String),
}

pub type RvfClientResult<T> = Result<T, RvfClientError>;

/// Configuration for the RVF client.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RvfClientConfig {
    /// RVF server endpoint
    pub endpoint: String,
    /// Namespace for vector operations
    pub namespace: String,
    /// Vector dimension
    pub dimension: usize,
    /// Optional API key
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Connection timeout in seconds
    pub timeout_secs: u64,
    /// Enable dual-write mode (write to both usearch and RVF)
    pub dual_write: bool,
    /// Prefer RVF over usearch for search
    pub prefer_rvf: bool,
}

impl Default for RvfClientConfig {
    fn default() -> Self {
        Self {
            endpoint: std::env::var("RVF_ENDPOINT").unwrap_or_else(|_| "http://localhost:6333".to_string()),
            namespace: std::env::var("RVF_NAMESPACE").unwrap_or_else(|_| "default".to_string()),
            dimension: std::env::var("RVF_DIMENSION")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(384),
            api_key: std::env::var("RVF_API_KEY").ok(),
            timeout_secs: 30,
            dual_write: false,
            prefer_rvf: false,
        }
    }
}

impl RvfClientConfig {
    /// Create config from environment variables.
    pub fn from_env() -> Self {
        Self::default()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> RvfClientResult<()> {
        if self.endpoint.is_empty() {
            return Err(RvfClientError::Config("endpoint cannot be empty".to_string()));
        }
        if self.namespace.is_empty() {
            return Err(RvfClientError::Config("namespace cannot be empty".to_string()));
        }
        if self.dimension == 0 {
            return Err(RvfClientError::Config("dimension must be > 0".to_string()));
        }
        Ok(())
    }
}

/// Search result from RVF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RvfSearchResult {
    /// Vector ID
    pub id: String,
    /// Similarity score
    pub score: f32,
    /// Optional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Status response from RVF store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RvfStoreStatus {
    /// Store file path
    pub path: String,
    /// Vector dimension
    pub dimension: usize,
    /// Number of vectors
    pub count: usize,
    /// Distance metric
    pub metric: String,
    /// Segment statistics
    pub segments: RvfSegmentStats,
}

/// File identity information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RvfFileIdentity {
    /// Unique file ID (UUID)
    pub file_id: String,
    /// Parent file ID (if derived)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    /// Derivation depth
    pub lineage_depth: u32,
}

impl RvfFileIdentity {
    /// Get the file ID as UUID.
    pub fn file_id(&self) -> Option<uuid::Uuid> {
        uuid::Uuid::parse_str(&self.file_id).ok()
    }

    /// Get the parent ID as UUID.
    pub fn parent_id(&self) -> Option<uuid::Uuid> {
        self.parent_id.as_ref().and_then(|s| uuid::Uuid::parse_str(s).ok())
    }
}

/// RVF Client for interacting with RVF stores.
///
/// This is a Rust port of the TypeScript ruvector-client.
pub struct RvfClient {
    config: RvfClientConfig,
    http: Client,
}

impl RvfClient {
    /// Create a new RVF client with the given configuration.
    pub fn new(config: RvfClientConfig) -> RvfClientResult<Self> {
        config.validate()?;

        let http = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| RvfClientError::Config(e.to_string()))?;

        Ok(Self { config, http })
    }

    /// Create a new RVF client from environment variables.
    pub fn from_env() -> RvfClientResult<Self> {
        Self::new(RvfClientConfig::from_env())
    }

    /// Get the configuration.
    pub fn config(&self) -> &RvfClientConfig {
        &self.config
    }

    /// Get the namespace.
    pub fn namespace(&self) -> &str {
        &self.config.namespace
    }

    /// Get the dimension.
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// Build the base URL for API calls.
    fn base_url(&self) -> String {
        format!("{}/{}", self.config.endpoint, self.config.namespace)
    }

    /// Add authorization header if API key is configured.
    fn auth_header(&self) -> Option<(String, String)> {
        self.config.api_key.as_ref().map(|key| {
            ("Authorization".to_string(), format!("Bearer {}", key))
        })
    }

    /// Check if the RVF service is healthy.
    pub async fn health(&self) -> RvfClientResult<bool> {
        let url = format!("{}/health", self.config.endpoint);
        let response = self.http.get(&url).send().await?;
        Ok(response.status().is_success())
    }

    /// Get store status.
    pub async fn status(&self, path: &str) -> RvfClientResult<RvfStoreStatus> {
        let url = format!("{}/status", self.base_url());
        let mut request = self.http.get(&url).query(&[("path", path)]);
        if let Some((key, value)) = self.auth_header() {
            request = request.header(&key, &value);
        }
        let response = request.send().await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(RvfClientError::NotFound(path.to_string()));
        }
        let status: RvfStoreStatus = response.json().await?;
        Ok(status)
    }

    /// Create a new RVF store.
    pub async fn create_store(
        &self,
        path: &Path,
        dimension: Option<usize>,
        metric: Option<&str>,
    ) -> RvfClientResult<RvfFileIdentity> {
        let url = format!("{}/create", self.base_url());
        let dim = dimension.unwrap_or(self.config.dimension);

        let mut body = serde_json::json!({
            "path": path.to_string_lossy(),
            "dimension": dim,
        });
        if let Some(m) = metric {
            body["metric"] = serde_json::json!(m);
        }

        let mut request = self.http.post(&url).json(&body);
        if let Some((key, value)) = self.auth_header() {
            request = request.header(&key, &value);
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            let error: serde_json::Value = response.json().await?;
            return Err(RvfClientError::Api(error.to_string()));
        }

        let identity: RvfFileIdentity = response.json().await?;
        Ok(identity)
    }

    /// Open an existing RVF store.
    pub async fn open_store(&self, path: &Path, read_only: bool) -> RvfClientResult<RvfFileIdentity> {
        let url = format!("{}/open", self.base_url());
        let body = serde_json::json!({
            "path": path.to_string_lossy(),
            "read_only": read_only,
        });

        let mut request = self.http.post(&url).json(&body);
        if let Some((key, value)) = self.auth_header() {
            request = request.header(&key, &value);
        }

        let response = request.send().await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(RvfClientError::NotFound(path.to_string_lossy().to_string()));
        }
        if !response.status().is_success() {
            let error: serde_json::Value = response.json().await?;
            return Err(RvfClientError::Api(error.to_string()));
        }

        let identity: RvfFileIdentity = response.json().await?;
        Ok(identity)
    }

    /// Ingest vectors into the store.
    pub async fn ingest(
        &self,
        vectors: &[Vec<f32>],
        ids: Option<&[String]>,
        metadata: Option<&[serde_json::Value]>,
    ) -> RvfClientResult<usize> {
        let url = format!("{}/ingest", self.base_url());

        let body = serde_json::json!({
            "vectors": vectors,
            "ids": ids,
            "metadata": metadata,
        });

        let mut request = self.http.post(&url).json(&body);
        if let Some((key, value)) = self.auth_header() {
            request = request.header(&key, &value);
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            let error: serde_json::Value = response.json().await?;
            return Err(RvfClientError::Api(error.to_string()));
        }

        #[derive(Deserialize)]
        struct IngestResponse {
            count: usize,
        }
        let result: IngestResponse = response.json().await?;
        Ok(result.count)
    }

    /// Search for similar vectors.
    pub async fn search(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: Option<usize>,
        filter: Option<serde_json::Value>,
    ) -> RvfClientResult<Vec<RvfSearchResult>> {
        let url = format!("{}/search", self.base_url());

        let mut body = serde_json::json!({
            "query": query,
            "k": top_k,
        });

        if let Some(ef) = ef_search {
            body["ef_search"] = serde_json::json!(ef);
        }
        if let Some(f) = filter {
            body["filter"] = f;
        }

        let mut request = self.http.post(&url).json(&body);
        if let Some((key, value)) = self.auth_header() {
            request = request.header(&key, &value);
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            let error: serde_json::Value = response.json().await?;
            return Err(RvfClientError::Api(error.to_string()));
        }

        let results: Vec<RvfSearchResult> = response.json().await?;
        Ok(results)
    }

    /// Delete vectors by ID.
    pub async fn delete(&self, ids: &[String]) -> RvfClientResult<usize> {
        let url = format!("{}/delete", self.base_url());

        let body = serde_json::json!({
            "ids": ids,
        });

        let mut request = self.http.post(&url).json(&body);
        if let Some((key, value)) = self.auth_header() {
            request = request.header(&key, &value);
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            let error: serde_json::Value = response.json().await?;
            return Err(RvfClientError::Api(error.to_string()));
        }

        #[derive(Deserialize)]
        struct DeleteResponse {
            count: usize,
        }
        let result: DeleteResponse = response.json().await?;
        Ok(result.count)
    }

    /// Derive a COW branch from the current store.
    pub async fn derive(&self, path: &Path, derive_type: Option<&str>) -> RvfClientResult<RvfFileIdentity> {
        let url = format!("{}/derive", self.base_url());

        let mut body = serde_json::json!({
            "path": path.to_string_lossy(),
        });
        if let Some(t) = derive_type {
            body["type"] = serde_json::json!(t);
        }

        let mut request = self.http.post(&url).json(&body);
        if let Some((key, value)) = self.auth_header() {
            request = request.header(&key, &value);
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            let error: serde_json::Value = response.json().await?;
            return Err(RvfClientError::Api(error.to_string()));
        }

        let identity: RvfFileIdentity = response.json().await?;
        Ok(identity)
    }

    /// Get file identity information.
    pub async fn file_identity(&self) -> RvfClientResult<RvfFileIdentity> {
        let url = format!("{}/identity", self.base_url());

        let mut request = self.http.get(&url);
        if let Some((key, value)) = self.auth_header() {
            request = request.header(&key, &value);
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            let error: serde_json::Value = response.json().await?;
            return Err(RvfClientError::Api(error.to_string()));
        }

        let identity: RvfFileIdentity = response.json().await?;
        Ok(identity)
    }

    /// Get segment information.
    pub async fn segments(&self) -> RvfClientResult<Vec<SegmentInfo>> {
        let url = format!("{}/segments", self.base_url());

        let mut request = self.http.get(&url);
        if let Some((key, value)) = self.auth_header() {
            request = request.header(&key, &value);
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            let error: serde_json::Value = response.json().await?;
            return Err(RvfClientError::Api(error.to_string()));
        }

        let segments: Vec<SegmentInfo> = response.json().await?;
        Ok(segments)
    }

    /// Compact the store to reclaim space.
    pub async fn compact(&self) -> RvfClientResult<()> {
        let url = format!("{}/compact", self.base_url());

        let mut request = self.http.post(&url);
        if let Some((key, value)) = self.auth_header() {
            request = request.header(&key, &value);
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            let error: serde_json::Value = response.json().await?;
            return Err(RvfClientError::Api(error.to_string()));
        }

        Ok(())
    }

    /// Close the store.
    pub async fn close(&self) -> RvfClientResult<()> {
        let url = format!("{}/close", self.base_url());

        let mut request = self.http.post(&url);
        if let Some((key, value)) = self.auth_header() {
            request = request.header(&key, &value);
        }

        let _ = request.send().await;
        Ok(())
    }
}

/// Information about a segment in the RVF store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentInfo {
    /// Segment type
    #[serde(rename = "type")]
    pub segment_type: String,
    /// Segment ID
    pub id: String,
    /// Size in bytes
    pub size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_env() {
        // Test default config
        let config = RvfClientConfig::default();
        assert_eq!(config.endpoint, "http://localhost:6333");
        assert_eq!(config.namespace, "default");
        assert_eq!(config.dimension, 384);
    }

    #[test]
    fn test_config_validation() {
        let mut config = RvfClientConfig::default();
        assert!(config.validate().is_ok());

        config.endpoint = "".to_string();
        assert!(config.validate().is_err());

        config = RvfClientConfig::default();
        config.namespace = "".to_string();
        assert!(config.validate().is_err());

        config = RvfClientConfig::default();
        config.dimension = 0;
        assert!(config.validate().is_err());
    }
}
