//! Configuration management for the Context Graph system.

mod sub_configs;

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::error::{CoreError, CoreResult};

// Re-export all sub-config types for backwards compatibility
pub use sub_configs::{
    CudaConfig, EmbeddingConfig, FeatureFlags, IndexConfig, LoggingConfig, McpConfig,
    ServerConfig, StorageConfig, UtlConfig,
};

/// System development phase.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    /// Ghost system phase - stubs and scaffolding
    #[default]
    Ghost,
    /// Development phase - active implementation
    Development,
    /// Production phase - fully operational
    Production,
}

/// Main configuration structure.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    /// Current system phase
    #[serde(default)]
    pub phase: Phase,
    pub server: ServerConfig,
    pub mcp: McpConfig,
    pub logging: LoggingConfig,
    pub storage: StorageConfig,
    pub embedding: EmbeddingConfig,
    pub index: IndexConfig,
    pub utl: UtlConfig,
    pub features: FeatureFlags,
    pub cuda: CudaConfig,
}

impl Config {
    /// Load configuration from files and environment.
    ///
    /// Configuration is loaded in order:
    /// 1. config/default.toml (base settings)
    /// 2. config/{CONTEXT_GRAPH_ENV}.toml (environment-specific)
    /// 3. Environment variables with CONTEXT_GRAPH_ prefix
    pub fn load() -> CoreResult<Self> {
        let env = std::env::var("CONTEXT_GRAPH_ENV").unwrap_or_else(|_| "development".to_string());

        let builder = config::Config::builder()
            .add_source(config::File::with_name("config/default").required(false))
            .add_source(config::File::with_name(&format!("config/{}", env)).required(false))
            .add_source(config::Environment::with_prefix("CONTEXT_GRAPH").separator("__"));

        let config: Config = builder.build()?.try_deserialize()?;
        config.validate()?;
        Ok(config)
    }

    /// Load configuration with defaults for testing/development.
    pub fn default_config() -> Self {
        Self {
            phase: Phase::default(),
            server: ServerConfig::default(),
            mcp: McpConfig::default(),
            logging: LoggingConfig::default(),
            storage: StorageConfig::default(),
            embedding: EmbeddingConfig::default(),
            index: IndexConfig::default(),
            utl: UtlConfig::default(),
            features: FeatureFlags::default(),
            cuda: CudaConfig::default(),
        }
    }

    /// Load configuration from a TOML file.
    pub fn from_file(path: &std::path::Path) -> CoreResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            CoreError::ConfigError(format!(
                "Failed to read config file {}: {}",
                path.display(),
                e
            ))
        })?;

        let config: Config = toml::from_str(&content)
            .map_err(|e| CoreError::ConfigError(format!("Failed to parse config file: {}", e)))?;

        config.validate()?;
        Ok(config)
    }

    /// Validate configuration values.
    pub fn validate(&self) -> CoreResult<()> {
        if self.mcp.max_payload_size == 0 {
            return Err(CoreError::ConfigError(
                "mcp.max_payload_size must be greater than 0".into(),
            ));
        }

        if self.embedding.dimension == 0 {
            return Err(CoreError::ConfigError(
                "embedding.dimension must be greater than 0".into(),
            ));
        }

        if self.storage.backend != "memory" {
            let path = PathBuf::from(&self.storage.path);
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() && !parent.exists() {
                    return Err(CoreError::ConfigError(format!(
                        "storage.path parent directory does not exist: {}",
                        parent.display()
                    )));
                }
            }
        }

        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::default_config()
    }
}
