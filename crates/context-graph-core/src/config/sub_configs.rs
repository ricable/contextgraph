//! Sub-configuration structures for Context Graph components.
//!
//! This module contains all the individual configuration structs
//! that make up the main `Config` structure.

use serde::{Deserialize, Serialize};

/// Server configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    pub name: String,
    pub version: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            name: "context-graph".to_string(),
            version: "0.1.0-ghost".to_string(),
        }
    }
}

/// MCP (Model Context Protocol) configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpConfig {
    pub transport: String,
    pub max_payload_size: usize,
    pub request_timeout: u64,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            transport: "stdio".to_string(),
            max_payload_size: 10_485_760, // 10MB
            request_timeout: 30,
        }
    }
}

/// Logging configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub include_location: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "pretty".to_string(),
            include_location: false,
        }
    }
}

/// Storage backend configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StorageConfig {
    pub backend: String,
    pub path: String,
    pub compression: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: "memory".to_string(),
            path: "./data/storage".to_string(),
            compression: true,
        }
    }
}

/// Embedding model configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingConfig {
    pub model: String,
    pub dimension: usize,
    pub max_input_length: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "stub".to_string(),
            dimension: 1536,
            max_input_length: 8191,
        }
    }
}

/// Index backend configuration (HNSW parameters).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IndexConfig {
    pub backend: String,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            backend: "memory".to_string(),
            hnsw_m: 16,
            hnsw_ef_construction: 200,
        }
    }
}

/// UTL (Unified Theory of Learning) configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UtlConfig {
    pub mode: String,
    pub default_emotional_weight: f32,
    pub consolidation_threshold: f32,
}

impl Default for UtlConfig {
    fn default() -> Self {
        Self {
            mode: "stub".to_string(),
            default_emotional_weight: 1.0,
            consolidation_threshold: 0.7,
        }
    }
}

/// Feature flags for enabling/disabling system components.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FeatureFlags {
    pub utl_enabled: bool,
    pub dream_enabled: bool,
    pub neuromodulation_enabled: bool,
    pub active_inference_enabled: bool,
    pub immune_enabled: bool,
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            utl_enabled: true,
            dream_enabled: false,
            neuromodulation_enabled: false,
            active_inference_enabled: false,
            immune_enabled: false,
        }
    }
}

/// CUDA/GPU configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CudaConfig {
    pub enabled: bool,
    pub device_id: u32,
    pub memory_limit_gb: f32,
}

impl Default for CudaConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            device_id: 0,
            memory_limit_gb: 4.0,
        }
    }
}
