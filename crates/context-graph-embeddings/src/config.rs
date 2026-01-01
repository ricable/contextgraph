//! Root configuration for the embedding pipeline.
//!
//! This module defines `EmbeddingConfig`, the top-level configuration struct
//! that aggregates all embedding subsystem configurations.
//!
//! # Loading Configuration
//!
//! ```rust,ignore
//! use context_graph_embeddings::EmbeddingConfig;
//!
//! // Load from file
//! let config = EmbeddingConfig::from_file("embeddings.toml")?;
//!
//! // Or use defaults for development
//! let config = EmbeddingConfig::default();
//!
//! // With environment overrides
//! let config = EmbeddingConfig::default().with_env_overrides();
//! ```
//!
//! # TOML Structure
//!
//! ```toml
//! [models]
//! models_dir = "./models"
//! lazy_loading = true
//! preload_models = ["semantic", "code"]
//!
//! [batch]
//! max_batch_size = 32
//! max_wait_ms = 50
//!
//! [fusion]
//! num_experts = 8
//! top_k = 4
//! output_dim = 1536
//!
//! [cache]
//! enabled = true
//! max_entries = 100000
//!
//! [gpu]
//! enabled = true
//! device_ids = [0]
//! ```
//!
//! # Design Principles
//!
//! - **NO FALLBACKS**: Invalid config returns error, never silently defaults
//! - **FAIL FAST**: File not found or parse error returns immediately
//! - **VALIDATION**: All nested configs are validated together

use std::env;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::ModelId;

// ============================================================================
// MODEL REGISTRY CONFIG
// ============================================================================

/// Configuration for the model registry and loading.
///
/// Controls model paths, lazy loading behavior, and preloaded models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistryConfig {
    /// Directory containing model files.
    /// Relative paths are resolved from working directory.
    #[serde(default = "default_models_dir")]
    pub models_dir: String,

    /// Whether to load models lazily (on first use) or eagerly.
    /// Lazy loading reduces startup time but increases first-request latency.
    #[serde(default = "default_lazy_loading")]
    pub lazy_loading: bool,

    /// Models to preload on startup (by name).
    /// Only effective when lazy_loading is false.
    /// Valid values: "semantic", "temporal_recent", "temporal_periodic",
    /// "temporal_positional", "causal", "sparse", "code", "graph",
    /// "hdc", "multimodal", "entity", "late_interaction"
    #[serde(default)]
    pub preload_models: Vec<String>,

    /// Maximum number of models to keep loaded simultaneously.
    /// When exceeded, least recently used models are unloaded.
    /// 0 means unlimited (all 12 models can be loaded).
    #[serde(default = "default_max_loaded_models")]
    pub max_loaded_models: usize,
}

fn default_models_dir() -> String {
    "./models".to_string()
}

fn default_lazy_loading() -> bool {
    true
}

fn default_max_loaded_models() -> usize {
    12 // All models can be loaded by default
}

impl Default for ModelRegistryConfig {
    fn default() -> Self {
        Self {
            models_dir: default_models_dir(),
            lazy_loading: default_lazy_loading(),
            preload_models: Vec::new(),
            max_loaded_models: default_max_loaded_models(),
        }
    }
}

impl ModelRegistryConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// - `EmbeddingError::ConfigError` if models_dir is empty
    /// - `EmbeddingError::ConfigError` if preload_models contains invalid model names
    pub fn validate(&self) -> EmbeddingResult<()> {
        if self.models_dir.is_empty() {
            return Err(EmbeddingError::ConfigError {
                message: "models_dir cannot be empty".to_string(),
            });
        }

        // Validate preload model names
        let valid_names: Vec<&str> = ModelId::all()
            .iter()
            .map(|id| id.as_str())
            .collect();

        for name in &self.preload_models {
            let normalized = name.to_lowercase().replace('-', "_");
            if !valid_names.iter().any(|v| v.to_lowercase().replace('-', "_") == normalized) {
                return Err(EmbeddingError::ConfigError {
                    message: format!(
                        "Invalid preload model name: '{}'. Valid names: {:?}",
                        name, valid_names
                    ),
                });
            }
        }

        Ok(())
    }
}

// ============================================================================
// PADDING STRATEGY ENUM
// ============================================================================

/// Padding strategy for variable-length sequences in a batch.
///
/// Controls how inputs of different lengths are padded to form uniform batches.
/// Choice affects memory usage and computational efficiency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PaddingStrategy {
    /// Pad all sequences to the model's max_tokens limit.
    /// Most memory-intensive but safest for models with fixed expectations.
    MaxLength,

    /// Pad to the longest sequence in the current batch.
    /// Most memory-efficient for variable-length inputs.
    #[default]
    DynamicMax,

    /// Pad to next power of two (cache-friendly).
    /// Good for GPU memory alignment and tensor core efficiency.
    PowerOfTwo,

    /// Use predefined length buckets (64, 128, 256, 512).
    /// Balances padding efficiency with kernel optimization.
    Bucket,
}

impl PaddingStrategy {
    /// Returns all valid padding strategies.
    pub fn all() -> &'static [PaddingStrategy] {
        &[
            PaddingStrategy::MaxLength,
            PaddingStrategy::DynamicMax,
            PaddingStrategy::PowerOfTwo,
            PaddingStrategy::Bucket,
        ]
    }

    /// Returns the strategy name as snake_case string.
    pub fn as_str(&self) -> &'static str {
        match self {
            PaddingStrategy::MaxLength => "max_length",
            PaddingStrategy::DynamicMax => "dynamic_max",
            PaddingStrategy::PowerOfTwo => "power_of_two",
            PaddingStrategy::Bucket => "bucket",
        }
    }
}

// ============================================================================
// BATCH CONFIG
// ============================================================================

/// Configuration for batch processing.
///
/// Controls how embedding requests are batched for efficient GPU utilization.
/// The batch processor accumulates requests and triggers batch inference when:
/// - Batch reaches `max_batch_size`, OR
/// - `max_wait_ms` timeout expires (if `min_batch_size` is met)
///
/// This enables high throughput (>100 items/sec) by amortizing model invocation overhead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum number of inputs per batch before triggering inference.
    /// Larger batches improve throughput but use more GPU memory.
    /// Constitution spec: max 32
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    /// Minimum batch size to wait for before processing.
    /// If timeout expires and batch size >= min_batch_size, process immediately.
    /// Set to 1 for latency-sensitive applications.
    /// Default: 1
    #[serde(default = "default_min_batch_size")]
    pub min_batch_size: usize,

    /// Maximum time to wait for a full batch (milliseconds).
    /// After this time, partial batch is processed (if >= min_batch_size).
    /// Constitution spec: 50ms (latency-sensitive: 10-100ms range)
    #[serde(default = "default_max_wait_ms")]
    pub max_wait_ms: u64,

    /// Whether to enable dynamic batching based on system load.
    /// When enabled, batch sizes adjust based on queue depth and GPU utilization.
    /// Default: true
    #[serde(default = "default_dynamic_batching")]
    pub dynamic_batching: bool,

    /// Padding strategy for variable-length inputs.
    /// Controls how sequences of different lengths are padded in a batch.
    #[serde(default)]
    pub padding_strategy: PaddingStrategy,

    /// Whether to sort inputs by sequence length before batching.
    /// Reduces padding waste by grouping similar-length sequences.
    /// Can reduce padding overhead by 20-40%.
    /// Default: true
    #[serde(default = "default_sort_by_length")]
    pub sort_by_length: bool,
}

fn default_max_batch_size() -> usize {
    32
}

fn default_min_batch_size() -> usize {
    1
}

fn default_max_wait_ms() -> u64 {
    50
}

fn default_dynamic_batching() -> bool {
    true
}

fn default_sort_by_length() -> bool {
    true
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: default_max_batch_size(),
            min_batch_size: default_min_batch_size(),
            max_wait_ms: default_max_wait_ms(),
            dynamic_batching: default_dynamic_batching(),
            padding_strategy: PaddingStrategy::default(),
            sort_by_length: default_sort_by_length(),
        }
    }
}

impl BatchConfig {
    /// Validate batch configuration values.
    ///
    /// # Errors
    /// - `EmbeddingError::ConfigError` if max_batch_size is 0
    /// - `EmbeddingError::ConfigError` if min_batch_size > max_batch_size
    /// - `EmbeddingError::ConfigError` if max_wait_ms is 0 when min_batch_size > 1
    pub fn validate(&self) -> EmbeddingResult<()> {
        if self.max_batch_size == 0 {
            return Err(EmbeddingError::ConfigError {
                message: "max_batch_size must be > 0".to_string(),
            });
        }

        if self.min_batch_size > self.max_batch_size {
            return Err(EmbeddingError::ConfigError {
                message: format!(
                    "min_batch_size ({}) cannot exceed max_batch_size ({})",
                    self.min_batch_size, self.max_batch_size
                ),
            });
        }

        if self.max_wait_ms == 0 && self.min_batch_size > 1 {
            return Err(EmbeddingError::ConfigError {
                message: "max_wait_ms must be > 0 when min_batch_size > 1".to_string(),
            });
        }

        Ok(())
    }
}

// ============================================================================
// FUSION CONFIG
// ============================================================================

/// Configuration for FuseMoE fusion layer.
///
/// Controls the Mixture-of-Experts fusion that combines 12 model outputs
/// into a unified 1536-dimensional embedding.
///
/// # Architecture
/// ```text
/// Input: Concatenated embeddings (8320D)
///        |
///        v
///   [Gating Network] --> Expert weights (8 values, temperature-scaled softmax)
///        |
///        v
///   [Top-4 Selection] --> Select 4 experts (Serotonin modulates: range [2,8])
///        |
///        v
///   [Expert Networks] --> Each: 8320 -> 4096 -> 4096 -> 1536
///        |
///        v
///   [Weighted Sum] --> Final 1536D embedding
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Number of expert networks in MoE.
    /// Constitution spec: 8 experts
    /// Default: 8
    #[serde(default = "default_num_experts")]
    pub num_experts: usize,

    /// Number of experts to activate per input (top-k routing).
    /// Constitution spec: top_k = 4 (NOT 2)
    /// Neuromodulation range: [2, 8] (Serotonin control)
    /// Default: 4
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Output embedding dimension after fusion.
    /// Constitution spec: 1536D (OpenAI ada-002 compatible)
    /// Default: 1536
    #[serde(default = "default_output_dim")]
    pub output_dim: usize,

    /// Hidden dimension in expert FFN layers.
    /// Architecture: input(8320) -> hidden(4096) -> hidden(4096) -> output(1536)
    /// Required by: M03-L21 (Expert Networks), M03-L30 (Grouped GEMM)
    /// Default: 4096
    #[serde(default = "default_expert_hidden_dim")]
    pub expert_hidden_dim: usize,

    /// Load balance loss coefficient.
    /// Penalizes uneven expert utilization during training.
    /// Set to 0.0 to disable, typical range: [0.01, 0.1]
    /// Default: 0.01
    #[serde(default = "default_load_balance_coef")]
    pub load_balance_coef: f32,

    /// Capacity factor for expert buffers.
    /// 1.25 = 25% overhead above average load.
    /// Must be >= 1.0 (no underprovisioning)
    /// Default: 1.25
    #[serde(default = "default_capacity_factor")]
    pub capacity_factor: f32,

    /// Temperature for gating network softmax.
    /// Lower = sharper expert selection, Higher = more uniform distribution
    /// Range: (0, inf), typical: [0.1, 2.0]
    /// Neuromodulation: Noradrenaline modulates attention.temp in range [0.5, 2.0]
    /// Default: 1.0
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Noise standard deviation for exploration (training only).
    /// Gaussian noise added to gating logits before softmax.
    /// Helps prevent expert collapse during training.
    /// Set to 0.0 for inference (deterministic).
    /// Default: 0.0
    #[serde(default = "default_noise_std")]
    pub noise_std: f32,

    /// Laplace smoothing alpha for stable routing.
    /// Formula: (p + alpha) / (1 + alpha * K)
    /// Prevents zero probabilities in gating.
    /// Default: 0.01
    #[serde(default = "default_laplace_alpha")]
    pub laplace_alpha: f32,
}

fn default_num_experts() -> usize {
    8
}

fn default_top_k() -> usize {
    4 // FIXED: was 2, constitution.yaml says 4
}

fn default_output_dim() -> usize {
    1536
}

fn default_expert_hidden_dim() -> usize {
    4096
}

fn default_load_balance_coef() -> f32 {
    0.01
}

fn default_capacity_factor() -> f32 {
    1.25
}

fn default_temperature() -> f32 {
    1.0
}

fn default_noise_std() -> f32 {
    0.0
}

fn default_laplace_alpha() -> f32 {
    0.01
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            num_experts: default_num_experts(),
            top_k: default_top_k(),
            output_dim: default_output_dim(),
            expert_hidden_dim: default_expert_hidden_dim(),
            load_balance_coef: default_load_balance_coef(),
            capacity_factor: default_capacity_factor(),
            temperature: default_temperature(),
            noise_std: default_noise_std(),
            laplace_alpha: default_laplace_alpha(),
        }
    }
}

impl FusionConfig {
    /// Validate fusion configuration values.
    ///
    /// # Errors
    /// Returns `EmbeddingError::ConfigError` if:
    /// - num_experts == 0
    /// - top_k == 0 or top_k > num_experts
    /// - output_dim == 0
    /// - expert_hidden_dim == 0
    /// - temperature <= 0 or is NaN
    /// - capacity_factor < 1.0 or is NaN
    /// - laplace_alpha < 0 or is NaN
    /// - noise_std < 0 or is NaN
    /// - load_balance_coef < 0 or is NaN
    pub fn validate(&self) -> EmbeddingResult<()> {
        if self.num_experts == 0 {
            return Err(EmbeddingError::ConfigError {
                message: "num_experts must be > 0".to_string(),
            });
        }
        if self.top_k == 0 || self.top_k > self.num_experts {
            return Err(EmbeddingError::ConfigError {
                message: format!(
                    "top_k must be in [1, {}], got {}",
                    self.num_experts, self.top_k
                ),
            });
        }
        if self.output_dim == 0 {
            return Err(EmbeddingError::ConfigError {
                message: "output_dim must be > 0".to_string(),
            });
        }
        if self.expert_hidden_dim == 0 {
            return Err(EmbeddingError::ConfigError {
                message: "expert_hidden_dim must be > 0".to_string(),
            });
        }
        if self.temperature <= 0.0 || self.temperature.is_nan() {
            return Err(EmbeddingError::ConfigError {
                message: "temperature must be > 0 and not NaN".to_string(),
            });
        }
        if self.capacity_factor < 1.0 || self.capacity_factor.is_nan() {
            return Err(EmbeddingError::ConfigError {
                message: "capacity_factor must be >= 1.0 and not NaN".to_string(),
            });
        }
        if self.laplace_alpha < 0.0 || self.laplace_alpha.is_nan() {
            return Err(EmbeddingError::ConfigError {
                message: "laplace_alpha must be >= 0 and not NaN".to_string(),
            });
        }
        if self.noise_std < 0.0 || self.noise_std.is_nan() {
            return Err(EmbeddingError::ConfigError {
                message: "noise_std must be >= 0 and not NaN".to_string(),
            });
        }
        if self.load_balance_coef < 0.0 || self.load_balance_coef.is_nan() {
            return Err(EmbeddingError::ConfigError {
                message: "load_balance_coef must be >= 0 and not NaN".to_string(),
            });
        }
        Ok(())
    }

    /// Create inference configuration (deterministic, no noise).
    ///
    /// Returns config with:
    /// - noise_std = 0.0 (no exploration noise)
    /// - load_balance_coef = 0.0 (no load balancing loss)
    ///
    /// Use this for production inference where determinism is required.
    pub fn for_inference() -> Self {
        Self {
            noise_std: 0.0,
            load_balance_coef: 0.0,
            ..Default::default()
        }
    }

    /// Create training configuration (with exploration noise).
    ///
    /// Returns config with:
    /// - noise_std = 0.1 (Gaussian noise for exploration)
    /// - load_balance_coef = 0.01 (auxiliary loss for load balancing)
    ///
    /// Use this for training to prevent expert collapse.
    pub fn for_training() -> Self {
        Self {
            noise_std: 0.1,
            load_balance_coef: 0.01,
            ..Default::default()
        }
    }

    /// Check if this config is for inference mode.
    pub fn is_inference_mode(&self) -> bool {
        self.noise_std == 0.0
    }
}

// ============================================================================
// CACHE CONFIG
// ============================================================================

/// Configuration for embedding cache.
///
/// Controls caching of computed embeddings to avoid redundant computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Whether caching is enabled.
    #[serde(default = "default_cache_enabled")]
    pub enabled: bool,

    /// Maximum number of cached embeddings.
    /// Constitution spec: 100,000 entries
    #[serde(default = "default_max_entries")]
    pub max_entries: usize,

    /// Time-to-live for cache entries (seconds).
    /// 0 means no expiration.
    #[serde(default = "default_ttl_seconds")]
    pub ttl_seconds: u64,

    /// Whether to persist cache to disk.
    #[serde(default)]
    pub disk_persistence: bool,

    /// Path for disk cache (only used if disk_persistence is true).
    #[serde(default = "default_cache_path")]
    pub cache_path: String,
}

fn default_cache_enabled() -> bool {
    true
}

fn default_max_entries() -> usize {
    100_000
}

fn default_ttl_seconds() -> u64 {
    0 // No expiration by default
}

fn default_cache_path() -> String {
    "./.cache/embeddings".to_string()
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: default_cache_enabled(),
            max_entries: default_max_entries(),
            ttl_seconds: default_ttl_seconds(),
            disk_persistence: false,
            cache_path: default_cache_path(),
        }
    }
}

impl CacheConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// - `EmbeddingError::ConfigError` if enabled but max_entries is 0
    /// - `EmbeddingError::ConfigError` if disk_persistence but cache_path is empty
    pub fn validate(&self) -> EmbeddingResult<()> {
        if self.enabled && self.max_entries == 0 {
            return Err(EmbeddingError::ConfigError {
                message: "max_entries must be > 0 when cache is enabled".to_string(),
            });
        }

        if self.disk_persistence && self.cache_path.is_empty() {
            return Err(EmbeddingError::ConfigError {
                message: "cache_path cannot be empty when disk_persistence is enabled".to_string(),
            });
        }

        Ok(())
    }
}

// ============================================================================
// GPU CONFIG
// ============================================================================

/// Configuration for GPU/CUDA settings.
///
/// Controls GPU device selection and CUDA features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Whether GPU acceleration is enabled.
    #[serde(default = "default_gpu_enabled")]
    pub enabled: bool,

    /// CUDA device IDs to use. Empty means auto-select.
    /// Default: [0] for single RTX 5090
    #[serde(default = "default_device_ids")]
    pub device_ids: Vec<u32>,

    /// Memory limit per GPU (bytes). 0 means no limit.
    /// Constitution spec: <24GB target (32GB available)
    #[serde(default = "default_memory_limit")]
    pub memory_limit: u64,

    /// Whether to enable CUDA graphs for kernel optimization.
    #[serde(default = "default_cuda_graphs")]
    pub cuda_graphs: bool,

    /// Whether to enable GPU Direct Storage (GDS) for fast model loading.
    #[serde(default)]
    pub gds_enabled: bool,

    /// Green Context SM allocation percentage (0-100).
    /// 0 means disabled.
    #[serde(default)]
    pub green_context_percentage: u8,
}

fn default_gpu_enabled() -> bool {
    true
}

fn default_device_ids() -> Vec<u32> {
    vec![0]
}

fn default_memory_limit() -> u64 {
    24 * 1024 * 1024 * 1024 // 24GB
}

fn default_cuda_graphs() -> bool {
    true
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: default_gpu_enabled(),
            device_ids: default_device_ids(),
            memory_limit: default_memory_limit(),
            cuda_graphs: default_cuda_graphs(),
            gds_enabled: false,
            green_context_percentage: 0,
        }
    }
}

impl GpuConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// - `EmbeddingError::ConfigError` if green_context_percentage > 100
    pub fn validate(&self) -> EmbeddingResult<()> {
        if self.green_context_percentage > 100 {
            return Err(EmbeddingError::ConfigError {
                message: "green_context_percentage must be <= 100".to_string(),
            });
        }

        Ok(())
    }
}

// ============================================================================
// ROOT EMBEDDING CONFIG
// ============================================================================

/// Root configuration for the embedding pipeline.
///
/// Aggregates all subsystem configurations.
/// Load from TOML file or use `Default::default()` for development.
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_embeddings::EmbeddingConfig;
///
/// // Load from file
/// let config = EmbeddingConfig::from_file("config/embeddings.toml")?;
///
/// // Validate
/// config.validate()?;
///
/// // With environment overrides
/// let config = EmbeddingConfig::default().with_env_overrides();
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model registry configuration (paths, lazy loading, etc.)
    #[serde(default)]
    pub models: ModelRegistryConfig,

    /// Batch processing configuration
    #[serde(default)]
    pub batch: BatchConfig,

    /// FuseMoE fusion layer configuration
    #[serde(default)]
    pub fusion: FusionConfig,

    /// Embedding cache configuration
    #[serde(default)]
    pub cache: CacheConfig,

    /// GPU configuration
    #[serde(default)]
    pub gpu: GpuConfig,
}


impl EmbeddingConfig {
    /// Load configuration from a TOML file.
    ///
    /// # Arguments
    /// * `path` - Path to the TOML configuration file
    ///
    /// # Errors
    /// - `EmbeddingError::IoError` if file cannot be read
    /// - `EmbeddingError::ConfigError` if TOML parsing fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = EmbeddingConfig::from_file("embeddings.toml")?;
    /// ```
    pub fn from_file(path: impl AsRef<Path>) -> EmbeddingResult<Self> {
        let path = path.as_ref();

        let contents = std::fs::read_to_string(path).map_err(|e| EmbeddingError::ConfigError {
            message: format!("Failed to read config file '{}': {}", path.display(), e),
        })?;

        let config: Self = toml::from_str(&contents).map_err(|e| EmbeddingError::ConfigError {
            message: format!("Failed to parse TOML in '{}': {}", path.display(), e),
        })?;

        Ok(config)
    }

    /// Validate all configuration values.
    ///
    /// Validates all nested configurations and returns the first error found.
    ///
    /// # Errors
    /// - `EmbeddingError::ConfigError` with descriptive message if any config is invalid
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = EmbeddingConfig::default();
    /// config.validate()?; // Should pass for defaults
    /// ```
    pub fn validate(&self) -> EmbeddingResult<()> {
        // Validate each subsystem config, returning first error
        self.models.validate().map_err(|e| EmbeddingError::ConfigError {
            message: format!("[models] {}", e),
        })?;

        self.batch.validate().map_err(|e| EmbeddingError::ConfigError {
            message: format!("[batch] {}", e),
        })?;

        self.fusion.validate().map_err(|e| EmbeddingError::ConfigError {
            message: format!("[fusion] {}", e),
        })?;

        self.cache.validate().map_err(|e| EmbeddingError::ConfigError {
            message: format!("[cache] {}", e),
        })?;

        self.gpu.validate().map_err(|e| EmbeddingError::ConfigError {
            message: format!("[gpu] {}", e),
        })?;

        Ok(())
    }

    /// Create configuration with environment variable overrides.
    ///
    /// Environment variables override TOML values. Prefix: `EMBEDDING_`
    ///
    /// # Supported Variables
    ///
    /// | Variable | Config Path | Type |
    /// |----------|-------------|------|
    /// | `EMBEDDING_MODELS_DIR` | `models.models_dir` | String |
    /// | `EMBEDDING_LAZY_LOADING` | `models.lazy_loading` | bool |
    /// | `EMBEDDING_GPU_ENABLED` | `gpu.enabled` | bool |
    /// | `EMBEDDING_CACHE_ENABLED` | `cache.enabled` | bool |
    /// | `EMBEDDING_CACHE_MAX_ENTRIES` | `cache.max_entries` | usize |
    /// | `EMBEDDING_BATCH_MAX_SIZE` | `batch.max_batch_size` | usize |
    /// | `EMBEDDING_BATCH_MAX_WAIT_MS` | `batch.max_wait_ms` | u64 |
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// std::env::set_var("EMBEDDING_GPU_ENABLED", "false");
    /// let config = EmbeddingConfig::default().with_env_overrides();
    /// assert!(!config.gpu.enabled);
    /// ```
    #[must_use]
    pub fn with_env_overrides(mut self) -> Self {
        // Models config
        if let Ok(val) = env::var("EMBEDDING_MODELS_DIR") {
            self.models.models_dir = val;
        }
        if let Ok(val) = env::var("EMBEDDING_LAZY_LOADING") {
            if let Ok(b) = val.parse::<bool>() {
                self.models.lazy_loading = b;
            }
        }

        // GPU config
        if let Ok(val) = env::var("EMBEDDING_GPU_ENABLED") {
            if let Ok(b) = val.parse::<bool>() {
                self.gpu.enabled = b;
            }
        }

        // Cache config
        if let Ok(val) = env::var("EMBEDDING_CACHE_ENABLED") {
            if let Ok(b) = val.parse::<bool>() {
                self.cache.enabled = b;
            }
        }
        if let Ok(val) = env::var("EMBEDDING_CACHE_MAX_ENTRIES") {
            if let Ok(n) = val.parse::<usize>() {
                self.cache.max_entries = n;
            }
        }

        // Batch config
        if let Ok(val) = env::var("EMBEDDING_BATCH_MAX_SIZE") {
            if let Ok(n) = val.parse::<usize>() {
                self.batch.max_batch_size = n;
            }
        }
        if let Ok(val) = env::var("EMBEDDING_BATCH_MAX_WAIT_MS") {
            if let Ok(n) = val.parse::<u64>() {
                self.batch.max_wait_ms = n;
            }
        }

        self
    }

    /// Create configuration from TOML string (for testing).
    ///
    /// # Errors
    /// - `EmbeddingError::ConfigError` if TOML parsing fails
    pub fn from_toml_str(toml: &str) -> EmbeddingResult<Self> {
        toml::from_str(toml).map_err(|e| EmbeddingError::ConfigError {
            message: format!("Failed to parse TOML: {}", e),
        })
    }

    /// Serialize configuration to TOML string.
    ///
    /// # Errors
    /// - `EmbeddingError::ConfigError` if serialization fails
    pub fn to_toml_string(&self) -> EmbeddingResult<String> {
        toml::to_string_pretty(self).map_err(|e| EmbeddingError::ConfigError {
            message: format!("Failed to serialize to TOML: {}", e),
        })
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // =========================================================================
    // DEFAULT TESTS (5 tests)
    // =========================================================================

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();

        // Verify defaults match constitution.yaml specs
        assert_eq!(config.batch.max_batch_size, 32);
        assert_eq!(config.batch.max_wait_ms, 50);
        assert_eq!(config.fusion.num_experts, 8);
        assert_eq!(config.fusion.top_k, 4); // FIXED: constitution.yaml says 4, not 2
        assert_eq!(config.fusion.output_dim, 1536);
        assert_eq!(config.cache.max_entries, 100_000);
        assert!(config.gpu.enabled);
    }

    #[test]
    fn test_model_registry_config_default() {
        let config = ModelRegistryConfig::default();
        assert_eq!(config.models_dir, "./models");
        assert!(config.lazy_loading);
        assert!(config.preload_models.is_empty());
        assert_eq!(config.max_loaded_models, 12);
    }

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.min_batch_size, 1);
        assert_eq!(config.max_wait_ms, 50);
        assert!(config.dynamic_batching);
        assert!(config.sort_by_length);
        assert_eq!(config.padding_strategy, PaddingStrategy::DynamicMax);
    }

    #[test]
    fn test_fusion_config_default() {
        let config = FusionConfig::default();
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.top_k, 4); // FIXED: constitution.yaml says 4, not 2
        assert_eq!(config.output_dim, 1536);
        assert_eq!(config.expert_hidden_dim, 4096);
        assert!((config.temperature - 1.0).abs() < f32::EPSILON);
        assert!((config.noise_std - 0.0).abs() < f32::EPSILON);
        assert!((config.laplace_alpha - 0.01).abs() < f32::EPSILON);
        assert!((config.capacity_factor - 1.25).abs() < f32::EPSILON);
        assert!((config.load_balance_coef - 0.01).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert!(config.enabled);
        assert_eq!(config.device_ids, vec![0]);
        assert_eq!(config.memory_limit, 24 * 1024 * 1024 * 1024);
        assert!(config.cuda_graphs);
        assert!(!config.gds_enabled);
        assert_eq!(config.green_context_percentage, 0);
    }

    // =========================================================================
    // VALIDATION TESTS (12 tests)
    // =========================================================================

    #[test]
    fn test_default_config_validates() {
        let config = EmbeddingConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_model_registry_empty_dir_fails() {
        let config = ModelRegistryConfig {
            models_dir: "".to_string(),
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("models_dir"));
    }

    #[test]
    fn test_model_registry_invalid_preload_fails() {
        let config = ModelRegistryConfig {
            preload_models: vec!["invalid_model".to_string()],
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid_model"));
    }

    #[test]
    fn test_model_registry_valid_preload_succeeds() {
        let config = ModelRegistryConfig {
            preload_models: vec!["semantic".to_string(), "code".to_string()],
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_batch_zero_size_fails() {
        let config = BatchConfig {
            max_batch_size: 0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_batch_size"));
    }

    #[test]
    fn test_batch_zero_wait_with_min_batch_greater_than_one_fails() {
        // max_wait_ms=0 is only invalid when min_batch_size > 1
        let config = BatchConfig {
            max_wait_ms: 0,
            min_batch_size: 4,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_wait_ms"));
    }

    #[test]
    fn test_batch_zero_wait_with_min_batch_one_succeeds() {
        // Special case: max_wait_ms=0 is OK if min_batch_size=1
        let config = BatchConfig {
            min_batch_size: 1,
            max_wait_ms: 0,
            max_batch_size: 32,
            dynamic_batching: true,
            padding_strategy: PaddingStrategy::DynamicMax,
            sort_by_length: true,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_batch_min_exceeds_max_fails() {
        let config = BatchConfig {
            min_batch_size: 64,
            max_batch_size: 32,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("min_batch_size"));
        assert!(msg.contains("cannot exceed"));
    }

    #[test]
    fn test_fusion_zero_experts_fails() {
        let config = FusionConfig {
            num_experts: 0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("num_experts"));
    }

    #[test]
    fn test_fusion_top_k_exceeds_experts_fails() {
        let config = FusionConfig {
            num_experts: 4,
            top_k: 8,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("top_k"));
    }

    #[test]
    fn test_fusion_negative_laplace_fails() {
        let config = FusionConfig {
            laplace_alpha: -0.1,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("laplace_alpha"));
    }

    #[test]
    fn test_cache_enabled_zero_entries_fails() {
        let config = CacheConfig {
            enabled: true,
            max_entries: 0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_entries"));
    }

    #[test]
    fn test_gpu_green_context_over_100_fails() {
        let config = GpuConfig {
            green_context_percentage: 101,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("green_context_percentage"));
    }

    // =========================================================================
    // SERDE ROUNDTRIP TESTS (5 tests)
    // =========================================================================

    #[test]
    fn test_serde_roundtrip_json() {
        let original = EmbeddingConfig::default();
        let json = serde_json::to_string(&original).unwrap();
        let restored: EmbeddingConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(original.batch.max_batch_size, restored.batch.max_batch_size);
        assert_eq!(original.fusion.num_experts, restored.fusion.num_experts);
    }

    #[test]
    fn test_serde_roundtrip_toml() {
        let original = EmbeddingConfig::default();
        let toml_str = original.to_toml_string().unwrap();
        let restored = EmbeddingConfig::from_toml_str(&toml_str).unwrap();

        assert_eq!(original.batch.max_batch_size, restored.batch.max_batch_size);
        assert_eq!(original.fusion.num_experts, restored.fusion.num_experts);
    }

    #[test]
    fn test_from_toml_str_custom_values() {
        let toml = r#"
[batch]
max_batch_size = 64
max_wait_ms = 100

[fusion]
num_experts = 16
top_k = 4

[cache]
enabled = false
"#;
        let config = EmbeddingConfig::from_toml_str(toml).unwrap();

        assert_eq!(config.batch.max_batch_size, 64);
        assert_eq!(config.batch.max_wait_ms, 100);
        assert_eq!(config.fusion.num_experts, 16);
        assert_eq!(config.fusion.top_k, 4);
        assert!(!config.cache.enabled);
    }

    #[test]
    fn test_from_toml_str_partial_config() {
        // Only specify some values, rest should be defaults
        let toml = r#"
[gpu]
enabled = false
"#;
        let config = EmbeddingConfig::from_toml_str(toml).unwrap();

        assert!(!config.gpu.enabled);
        // Defaults still apply
        assert_eq!(config.batch.max_batch_size, 32);
        assert_eq!(config.fusion.num_experts, 8);
    }

    #[test]
    fn test_from_toml_str_invalid_fails() {
        let toml = "invalid { toml } content";
        let result = EmbeddingConfig::from_toml_str(toml);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("TOML"));
    }

    // =========================================================================
    // FILE LOADING TESTS (4 tests)
    // =========================================================================

    #[test]
    fn test_from_file_success() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "[batch]").unwrap();
        writeln!(file, "max_batch_size = 128").unwrap();

        let config = EmbeddingConfig::from_file(file.path()).unwrap();
        assert_eq!(config.batch.max_batch_size, 128);
    }

    #[test]
    fn test_from_file_missing_returns_config_error() {
        let result = EmbeddingConfig::from_file("/nonexistent/path/config.toml");
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            EmbeddingError::ConfigError { message } => {
                assert!(message.contains("nonexistent"));
            }
            _ => panic!("Expected ConfigError, got {:?}", err),
        }
    }

    #[test]
    fn test_from_file_invalid_toml_returns_config_error() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "not valid toml {{}}").unwrap();

        let result = EmbeddingConfig::from_file(file.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            EmbeddingError::ConfigError { message } => {
                assert!(message.contains("TOML"));
            }
            _ => panic!("Expected ConfigError, got {:?}", err),
        }
    }

    #[test]
    fn test_from_file_empty_uses_defaults() {
        let file = NamedTempFile::new().unwrap();
        // Empty file

        let config = EmbeddingConfig::from_file(file.path()).unwrap();
        assert_eq!(config.batch.max_batch_size, 32); // Default
    }

    // =========================================================================
    // ENVIRONMENT OVERRIDE TESTS (6 tests)
    // =========================================================================

    #[test]
    fn test_env_override_models_dir() {
        env::set_var("EMBEDDING_MODELS_DIR", "/custom/models");
        let config = EmbeddingConfig::default().with_env_overrides();
        env::remove_var("EMBEDDING_MODELS_DIR");

        assert_eq!(config.models.models_dir, "/custom/models");
    }

    #[test]
    fn test_env_override_gpu_enabled() {
        env::set_var("EMBEDDING_GPU_ENABLED", "false");
        let config = EmbeddingConfig::default().with_env_overrides();
        env::remove_var("EMBEDDING_GPU_ENABLED");

        assert!(!config.gpu.enabled);
    }

    #[test]
    fn test_env_override_cache_max_entries() {
        env::set_var("EMBEDDING_CACHE_MAX_ENTRIES", "50000");
        let config = EmbeddingConfig::default().with_env_overrides();
        env::remove_var("EMBEDDING_CACHE_MAX_ENTRIES");

        assert_eq!(config.cache.max_entries, 50000);
    }

    #[test]
    fn test_env_override_batch_max_size() {
        env::set_var("EMBEDDING_BATCH_MAX_SIZE", "64");
        let config = EmbeddingConfig::default().with_env_overrides();
        env::remove_var("EMBEDDING_BATCH_MAX_SIZE");

        assert_eq!(config.batch.max_batch_size, 64);
    }

    #[test]
    fn test_env_override_invalid_value_ignored() {
        env::set_var("EMBEDDING_GPU_ENABLED", "not_a_bool");
        let config = EmbeddingConfig::default().with_env_overrides();
        env::remove_var("EMBEDDING_GPU_ENABLED");

        // Should keep default because "not_a_bool" can't be parsed
        assert!(config.gpu.enabled);
    }

    #[test]
    fn test_env_override_lazy_loading() {
        env::set_var("EMBEDDING_LAZY_LOADING", "false");
        let config = EmbeddingConfig::default().with_env_overrides();
        env::remove_var("EMBEDDING_LAZY_LOADING");

        assert!(!config.models.lazy_loading);
    }

    // =========================================================================
    // CONSTITUTION COMPLIANCE TESTS (5 tests)
    // =========================================================================

    #[test]
    fn test_constitution_batch_defaults() {
        // constitution.yaml: max_batch_size = 32, max_wait_ms = 50
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.max_wait_ms, 50);
    }

    #[test]
    fn test_constitution_fusion_defaults() {
        // constitution.yaml: num_experts = 8, top_k = 4 (NOT 2!), output_dim = 1536
        let config = FusionConfig::default();
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.top_k, 4); // FIXED: constitution.yaml fuse_moe.top_k = 4
        assert_eq!(config.output_dim, 1536);
        assert_eq!(config.expert_hidden_dim, 4096);
    }

    #[test]
    fn test_constitution_cache_defaults() {
        // constitution.yaml: max_entries = 100000
        let config = CacheConfig::default();
        assert_eq!(config.max_entries, 100_000);
    }

    #[test]
    fn test_constitution_gpu_memory_limit() {
        // constitution.yaml: <24GB target
        let config = GpuConfig::default();
        assert_eq!(config.memory_limit, 24 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_fusion_laplace_alpha() {
        // constitution.yaml: fuse_moe.laplace_alpha = 0.01
        let config = FusionConfig::default();
        assert!((config.laplace_alpha - 0.01).abs() < f32::EPSILON);
    }

    // =========================================================================
    // EDGE CASE TESTS (4 tests)
    // =========================================================================

    #[test]
    fn test_fusion_nan_laplace_fails() {
        let config = FusionConfig {
            laplace_alpha: f32::NAN,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_fusion_capacity_factor_below_one_fails() {
        let config = FusionConfig {
            capacity_factor: 0.9,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cache_disabled_with_zero_entries_succeeds() {
        // Zero entries is OK if cache is disabled
        let config = CacheConfig {
            enabled: false,
            max_entries: 0,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_nested_validation_error_includes_section() {
        let mut config = EmbeddingConfig::default();
        config.batch.max_batch_size = 0;

        let result = config.validate();
        assert!(result.is_err());
        // Error message should include [batch] section
        assert!(result.unwrap_err().to_string().contains("[batch]"));
    }

    // =========================================================================
    // PADDING STRATEGY TESTS (6 tests)
    // =========================================================================

    #[test]
    fn test_padding_strategy_default_is_dynamic_max() {
        assert_eq!(PaddingStrategy::default(), PaddingStrategy::DynamicMax);
    }

    #[test]
    fn test_padding_strategy_all_variants() {
        let all = PaddingStrategy::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&PaddingStrategy::MaxLength));
        assert!(all.contains(&PaddingStrategy::DynamicMax));
        assert!(all.contains(&PaddingStrategy::PowerOfTwo));
        assert!(all.contains(&PaddingStrategy::Bucket));
    }

    #[test]
    fn test_padding_strategy_as_str() {
        assert_eq!(PaddingStrategy::MaxLength.as_str(), "max_length");
        assert_eq!(PaddingStrategy::DynamicMax.as_str(), "dynamic_max");
        assert_eq!(PaddingStrategy::PowerOfTwo.as_str(), "power_of_two");
        assert_eq!(PaddingStrategy::Bucket.as_str(), "bucket");
    }

    #[test]
    fn test_padding_strategy_serde_roundtrip() {
        for strategy in PaddingStrategy::all() {
            let json = serde_json::to_string(strategy).unwrap();
            let restored: PaddingStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(*strategy, restored);
        }
    }

    #[test]
    fn test_padding_strategy_serde_snake_case() {
        // Verify snake_case serialization
        let json = serde_json::to_string(&PaddingStrategy::DynamicMax).unwrap();
        assert_eq!(json, "\"dynamic_max\"");

        let json = serde_json::to_string(&PaddingStrategy::PowerOfTwo).unwrap();
        assert_eq!(json, "\"power_of_two\"");
    }

    #[test]
    fn test_padding_strategy_copy() {
        // PaddingStrategy must be Copy for efficiency
        let a = PaddingStrategy::Bucket;
        let b = a; // Copy
        assert_eq!(a, b);
    }

    // =========================================================================
    // BATCH CONFIG NEW FIELD TESTS (3 tests)
    // =========================================================================

    #[test]
    fn test_batch_config_new_defaults() {
        let config = BatchConfig::default();
        assert_eq!(config.min_batch_size, 1);
        assert!(config.dynamic_batching);
        assert_eq!(config.padding_strategy, PaddingStrategy::DynamicMax);
    }

    #[test]
    fn test_batch_config_toml_roundtrip() {
        let original = BatchConfig {
            max_batch_size: 64,
            min_batch_size: 4,
            max_wait_ms: 100,
            dynamic_batching: false,
            padding_strategy: PaddingStrategy::PowerOfTwo,
            sort_by_length: false,
        };

        let toml_str = toml::to_string(&original).unwrap();
        let restored: BatchConfig = toml::from_str(&toml_str).unwrap();

        assert_eq!(original.max_batch_size, restored.max_batch_size);
        assert_eq!(original.min_batch_size, restored.min_batch_size);
        assert_eq!(original.max_wait_ms, restored.max_wait_ms);
        assert_eq!(original.dynamic_batching, restored.dynamic_batching);
        assert_eq!(original.padding_strategy, restored.padding_strategy);
        assert_eq!(original.sort_by_length, restored.sort_by_length);
    }

    #[test]
    fn test_batch_config_partial_toml_uses_defaults() {
        // Only specify max_batch_size, rest should be defaults
        let toml_str = r#"
max_batch_size = 64
"#;
        let config: BatchConfig = toml::from_str(toml_str).unwrap();

        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.min_batch_size, 1); // default
        assert_eq!(config.max_wait_ms, 50); // default
        assert!(config.dynamic_batching); // default
        assert_eq!(config.padding_strategy, PaddingStrategy::DynamicMax); // default
        assert!(config.sort_by_length); // default
    }

    // =========================================================================
    // FUSION CONFIG NEW FIELD TESTS (M03-F14)
    // =========================================================================

    #[test]
    fn test_fusion_config_default_top_k_is_4() {
        // CRITICAL: constitution.yaml specifies top_k = 4, NOT 2
        let config = FusionConfig::default();
        assert_eq!(config.top_k, 4);
    }

    #[test]
    fn test_fusion_config_default_expert_hidden_dim() {
        let config = FusionConfig::default();
        assert_eq!(config.expert_hidden_dim, 4096);
    }

    #[test]
    fn test_fusion_config_default_temperature() {
        let config = FusionConfig::default();
        assert!((config.temperature - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fusion_config_default_noise_std() {
        let config = FusionConfig::default();
        assert!((config.noise_std - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fusion_config_default_load_balance_coef() {
        let config = FusionConfig::default();
        assert!((config.load_balance_coef - 0.01).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fusion_validate_valid_default() {
        let config = FusionConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_fusion_validate_num_experts_zero() {
        let config = FusionConfig {
            num_experts: 0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("num_experts"));
    }

    #[test]
    fn test_fusion_validate_top_k_zero() {
        let config = FusionConfig {
            top_k: 0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("top_k"));
    }

    #[test]
    fn test_fusion_validate_top_k_exceeds_experts() {
        let config = FusionConfig {
            num_experts: 4,
            top_k: 8,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("top_k"));
    }

    #[test]
    fn test_fusion_validate_top_k_equals_experts_succeeds() {
        // Edge case: top_k = num_experts should be valid
        let config = FusionConfig {
            num_experts: 8,
            top_k: 8,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_fusion_validate_output_dim_zero() {
        let config = FusionConfig {
            output_dim: 0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("output_dim"));
    }

    #[test]
    fn test_fusion_validate_expert_hidden_dim_zero() {
        let config = FusionConfig {
            expert_hidden_dim: 0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expert_hidden_dim"));
    }

    #[test]
    fn test_fusion_validate_temperature_zero() {
        let config = FusionConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("temperature"));
    }

    #[test]
    fn test_fusion_validate_temperature_negative() {
        let config = FusionConfig {
            temperature: -1.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("temperature"));
    }

    #[test]
    fn test_fusion_validate_temperature_nan() {
        let config = FusionConfig {
            temperature: f32::NAN,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("temperature"));
    }

    #[test]
    fn test_fusion_validate_temperature_very_small_succeeds() {
        // Edge case: very small temperature (still > 0) should be valid
        let config = FusionConfig {
            temperature: 0.001,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_fusion_validate_noise_std_negative() {
        let config = FusionConfig {
            noise_std: -0.1,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("noise_std"));
    }

    #[test]
    fn test_fusion_validate_noise_std_nan() {
        let config = FusionConfig {
            noise_std: f32::NAN,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("noise_std"));
    }

    #[test]
    fn test_fusion_validate_load_balance_coef_negative() {
        let config = FusionConfig {
            load_balance_coef: -0.01,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("load_balance_coef"));
    }

    #[test]
    fn test_fusion_validate_load_balance_coef_nan() {
        let config = FusionConfig {
            load_balance_coef: f32::NAN,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("load_balance_coef"));
    }

    #[test]
    fn test_fusion_for_inference_no_noise() {
        let config = FusionConfig::for_inference();
        assert!((config.noise_std - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fusion_for_inference_no_load_balance() {
        let config = FusionConfig::for_inference();
        assert!((config.load_balance_coef - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fusion_for_inference_validates() {
        let config = FusionConfig::for_inference();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_fusion_for_training_has_noise() {
        let config = FusionConfig::for_training();
        assert!((config.noise_std - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fusion_for_training_has_load_balance() {
        let config = FusionConfig::for_training();
        assert!((config.load_balance_coef - 0.01).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fusion_for_training_validates() {
        let config = FusionConfig::for_training();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_fusion_is_inference_mode_true() {
        let config = FusionConfig::for_inference();
        assert!(config.is_inference_mode());
    }

    #[test]
    fn test_fusion_is_inference_mode_false() {
        let config = FusionConfig::for_training();
        assert!(!config.is_inference_mode());
    }

    #[test]
    fn test_fusion_is_inference_mode_default() {
        // Default config has noise_std = 0.0, so should be inference mode
        let config = FusionConfig::default();
        assert!(config.is_inference_mode());
    }

    #[test]
    fn test_fusion_serde_roundtrip_json() {
        let original = FusionConfig::default();
        let json = serde_json::to_string(&original).unwrap();
        let restored: FusionConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(original.num_experts, restored.num_experts);
        assert_eq!(original.top_k, restored.top_k);
        assert_eq!(original.output_dim, restored.output_dim);
        assert_eq!(original.expert_hidden_dim, restored.expert_hidden_dim);
        assert!((original.temperature - restored.temperature).abs() < f32::EPSILON);
        assert!((original.noise_std - restored.noise_std).abs() < f32::EPSILON);
        assert!((original.load_balance_coef - restored.load_balance_coef).abs() < f32::EPSILON);
        assert!((original.capacity_factor - restored.capacity_factor).abs() < f32::EPSILON);
        assert!((original.laplace_alpha - restored.laplace_alpha).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fusion_serde_roundtrip_toml() {
        let original = FusionConfig {
            num_experts: 16,
            top_k: 6,
            output_dim: 2048,
            expert_hidden_dim: 8192,
            load_balance_coef: 0.05,
            capacity_factor: 1.5,
            temperature: 0.8,
            noise_std: 0.2,
            laplace_alpha: 0.02,
        };

        let toml_str = toml::to_string(&original).unwrap();
        let restored: FusionConfig = toml::from_str(&toml_str).unwrap();

        assert_eq!(original.num_experts, restored.num_experts);
        assert_eq!(original.top_k, restored.top_k);
        assert_eq!(original.output_dim, restored.output_dim);
        assert_eq!(original.expert_hidden_dim, restored.expert_hidden_dim);
        assert!((original.temperature - restored.temperature).abs() < f32::EPSILON);
        assert!((original.noise_std - restored.noise_std).abs() < f32::EPSILON);
        assert!((original.load_balance_coef - restored.load_balance_coef).abs() < f32::EPSILON);
        assert!((original.capacity_factor - restored.capacity_factor).abs() < f32::EPSILON);
        assert!((original.laplace_alpha - restored.laplace_alpha).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fusion_serde_partial_toml_uses_defaults() {
        // Only specify num_experts and top_k, rest should use defaults
        let toml_str = r#"
num_experts = 16
top_k = 8
"#;
        let config: FusionConfig = toml::from_str(toml_str).unwrap();

        assert_eq!(config.num_experts, 16);
        assert_eq!(config.top_k, 8);
        assert_eq!(config.output_dim, 1536); // default
        assert_eq!(config.expert_hidden_dim, 4096); // default
        assert!((config.temperature - 1.0).abs() < f32::EPSILON); // default
        assert!((config.noise_std - 0.0).abs() < f32::EPSILON); // default
        assert!((config.load_balance_coef - 0.01).abs() < f32::EPSILON); // default
    }

    // =========================================================================
    // EDGE CASE TESTS FOR FUSION CONFIG
    // =========================================================================

    #[test]
    fn test_fusion_edge_case_boundary_values() {
        // Test boundary values that should pass
        let config = FusionConfig {
            num_experts: 1,
            top_k: 1,
            output_dim: 1,
            expert_hidden_dim: 1,
            temperature: 0.0001, // Very small but > 0
            noise_std: 0.0,
            load_balance_coef: 0.0,
            capacity_factor: 1.0, // Exactly 1.0
            laplace_alpha: 0.0, // Exactly 0
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_fusion_edge_case_large_values() {
        // Test large but valid values
        let config = FusionConfig {
            num_experts: 1024,
            top_k: 512,
            output_dim: 65536,
            expert_hidden_dim: 32768,
            temperature: 100.0,
            noise_std: 10.0,
            load_balance_coef: 1.0,
            capacity_factor: 10.0,
            laplace_alpha: 1.0,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_fusion_capacity_factor_nan() {
        let config = FusionConfig {
            capacity_factor: f32::NAN,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("capacity_factor"));
    }

    #[test]
    fn test_fusion_laplace_alpha_nan() {
        let config = FusionConfig {
            laplace_alpha: f32::NAN,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("laplace_alpha"));
    }

    #[test]
    fn test_fusion_laplace_alpha_negative() {
        let config = FusionConfig {
            laplace_alpha: -0.001,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("laplace_alpha"));
    }
}
