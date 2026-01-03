//! Configuration types for Knowledge Graph components.
//!
//! This module provides configuration structures for:
//! - FAISS IVF-PQ vector index (IndexConfig)
//! - Hyperbolic/Poincare ball geometry (HyperbolicConfig)
//! - Entailment cones for IS-A queries (ConeConfig)
//!
//! # Constitution Reference
//!
//! - perf.latency.faiss_1M_k100: <2ms (drives nlist/nprobe defaults)
//! - embeddings.models.E7_Code: 1536D (default dimension)
//!
//! TODO: Full implementation in M04-T01, M04-T02, M04-T03

use serde::{Deserialize, Serialize};

use crate::error::GraphError;

/// Configuration for FAISS IVF-PQ GPU index.
///
/// Configures the FAISS GPU index for 10M+ vector search with <5ms latency.
///
/// # Performance Targets
/// - 10M vectors, k=10: <5ms latency
/// - 10M vectors, k=100: <10ms latency
/// - Memory: ~8GB VRAM for 10M 1536D vectors with PQ64x8
///
/// # Constitution Reference
/// - perf.latency.faiss_1M_k100: <2ms
/// - stack.deps: faiss@0.12+gpu
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexConfig {
    /// Vector dimension (must match embedding dimension).
    /// Default: 1536 per constitution embeddings.models.E7_Code
    pub dimension: usize,

    /// Number of inverted lists (clusters).
    /// Default: 16384 = 4 * sqrt(10M) for optimal recall/speed tradeoff
    pub nlist: usize,

    /// Number of clusters to probe during search.
    /// Default: 128 balances accuracy vs search time
    pub nprobe: usize,

    /// Number of product quantization segments.
    /// Must evenly divide dimension. Default: 64 (1536/64 = 24 bytes per segment)
    pub pq_segments: usize,

    /// Bits per quantization code.
    /// Valid values: 4, 8, 12, 16. Default: 8
    pub pq_bits: u8,

    /// GPU device ID.
    /// Default: 0 (primary GPU)
    pub gpu_id: i32,

    /// Use float16 for reduced memory.
    /// Default: true (halves VRAM usage)
    pub use_float16: bool,

    /// Minimum vectors required for training (256 * nlist).
    /// Default: 4,194,304 (256 * 16384)
    pub min_train_vectors: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dimension: 1536,
            nlist: 16384,
            nprobe: 128,
            pq_segments: 64,
            pq_bits: 8,
            gpu_id: 0,
            use_float16: true,
            min_train_vectors: 4_194_304, // 256 * 16384
        }
    }
}

impl IndexConfig {
    /// Generate FAISS factory string for index creation.
    ///
    /// Returns format: "IVF{nlist},PQ{pq_segments}x{pq_bits}"
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::IndexConfig;
    /// let config = IndexConfig::default();
    /// assert_eq!(config.factory_string(), "IVF16384,PQ64x8");
    /// ```
    pub fn factory_string(&self) -> String {
        format!("IVF{},PQ{}x{}", self.nlist, self.pq_segments, self.pq_bits)
    }

    /// Calculate minimum training vectors based on nlist.
    ///
    /// FAISS requires at least 256 vectors per cluster for quality training.
    ///
    /// # Returns
    /// 256 * nlist
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::IndexConfig;
    /// let config = IndexConfig::default();
    /// assert_eq!(config.calculate_min_train_vectors(), 4_194_304);
    /// ```
    pub fn calculate_min_train_vectors(&self) -> usize {
        256 * self.nlist
    }
}

/// Hyperbolic (Poincare ball) configuration.
///
/// Configures the Poincare ball model for representing hierarchical
/// relationships in hyperbolic space.
///
/// # Mathematics
/// - d(x,y) = arcosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))
/// - Curvature must be negative (typically -1.0)
/// - All points must have norm < 1.0
///
/// # Constitution Reference
/// - edge_model.nt_weights: Neurotransmitter weighting in hyperbolic space
/// - perf.latency.entailment_check: <1ms
///
/// # Example
/// ```
/// use context_graph_graph::config::HyperbolicConfig;
///
/// let config = HyperbolicConfig::default();
/// assert_eq!(config.dim, 64);
/// assert_eq!(config.curvature, -1.0);
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HyperbolicConfig {
    /// Dimension of hyperbolic space (typically 64 for knowledge graphs).
    /// Must be positive.
    pub dim: usize,

    /// Curvature of hyperbolic space. MUST be negative.
    /// Default: -1.0 (unit hyperbolic space)
    /// Validated in validate().
    pub curvature: f32,

    /// Epsilon for numerical stability in hyperbolic operations.
    /// Prevents division by zero and NaN in distance calculations.
    /// Default: 1e-7
    pub eps: f32,

    /// Maximum norm for points (keeps points strictly inside ball boundary).
    /// Points with norm >= max_norm will be projected back inside.
    /// Must be in open interval (0, 1). Default: 1.0 - 1e-5 = 0.99999
    pub max_norm: f32,
}

impl Default for HyperbolicConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            curvature: -1.0,
            eps: 1e-7,
            max_norm: 1.0 - 1e-5, // 0.99999
        }
    }
}

impl HyperbolicConfig {
    /// Create config with custom curvature.
    ///
    /// # Arguments
    /// * `curvature` - Must be negative. Use validate() to check.
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    /// let config = HyperbolicConfig::with_curvature(-0.5);
    /// assert_eq!(config.curvature, -0.5);
    /// assert_eq!(config.dim, 64); // other fields use defaults
    /// ```
    pub fn with_curvature(curvature: f32) -> Self {
        Self {
            curvature,
            ..Default::default()
        }
    }

    /// Get absolute value of curvature.
    ///
    /// Useful for formulas that need |c| rather than c.
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    /// let config = HyperbolicConfig::default();
    /// assert_eq!(config.abs_curvature(), 1.0);
    /// ```
    #[inline]
    pub fn abs_curvature(&self) -> f32 {
        self.curvature.abs()
    }

    /// Scale factor derived from curvature: sqrt(|c|)
    ///
    /// Used in Mobius operations and distance calculations.
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    /// let config = HyperbolicConfig::default();
    /// assert_eq!(config.scale(), 1.0); // sqrt(|-1.0|) = 1.0
    /// ```
    #[inline]
    pub fn scale(&self) -> f32 {
        self.abs_curvature().sqrt()
    }

    /// Validate that all configuration parameters are mathematically valid
    /// for the Poincare ball model.
    ///
    /// # Validation Rules
    /// - `dim` > 0: Dimension must be positive
    /// - `curvature` < 0: Must be negative for hyperbolic space
    /// - `eps` > 0: Must be positive for numerical stability
    /// - `max_norm` in (0, 1): Must be strictly between 0 and 1
    ///
    /// # Errors
    /// Returns `GraphError::InvalidConfig` with descriptive message if any
    /// parameter is invalid. Returns the FIRST error encountered (fail-fast).
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// // Valid config passes
    /// let valid = HyperbolicConfig::default();
    /// assert!(valid.validate().is_ok());
    ///
    /// // Invalid curvature fails
    /// let mut invalid = HyperbolicConfig::default();
    /// invalid.curvature = 1.0; // positive is invalid
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), GraphError> {
        // Check dimension
        if self.dim == 0 {
            return Err(GraphError::InvalidConfig(
                "dim must be positive (got 0)".to_string()
            ));
        }

        // Check curvature - MUST be negative for hyperbolic space
        if self.curvature >= 0.0 {
            return Err(GraphError::InvalidConfig(
                format!(
                    "curvature must be negative for hyperbolic space (got {})",
                    self.curvature
                )
            ));
        }

        // Check for NaN curvature
        if self.curvature.is_nan() {
            return Err(GraphError::InvalidConfig(
                "curvature cannot be NaN".to_string()
            ));
        }

        // Check epsilon
        if self.eps <= 0.0 {
            return Err(GraphError::InvalidConfig(
                format!(
                    "eps must be positive for numerical stability (got {})",
                    self.eps
                )
            ));
        }

        // Check for NaN eps
        if self.eps.is_nan() {
            return Err(GraphError::InvalidConfig(
                "eps cannot be NaN".to_string()
            ));
        }

        // Check max_norm - must be in open interval (0, 1)
        if self.max_norm <= 0.0 || self.max_norm >= 1.0 {
            return Err(GraphError::InvalidConfig(
                format!(
                    "max_norm must be in open interval (0, 1), got {}",
                    self.max_norm
                )
            ));
        }

        // Check for NaN max_norm
        if self.max_norm.is_nan() {
            return Err(GraphError::InvalidConfig(
                "max_norm cannot be NaN".to_string()
            ));
        }

        Ok(())
    }

    /// Create a validated config with custom curvature.
    ///
    /// Returns error if curvature is invalid (>= 0 or NaN).
    ///
    /// # Example
    /// ```
    /// use context_graph_graph::config::HyperbolicConfig;
    ///
    /// let config = HyperbolicConfig::try_with_curvature(-0.5).unwrap();
    /// assert_eq!(config.curvature, -0.5);
    ///
    /// // Invalid curvature returns error
    /// assert!(HyperbolicConfig::try_with_curvature(1.0).is_err());
    /// ```
    pub fn try_with_curvature(curvature: f32) -> Result<Self, GraphError> {
        let config = Self {
            curvature,
            ..Default::default()
        };
        config.validate()?;
        Ok(config)
    }
}

/// Entailment cone configuration.
///
/// Configures entailment cones for O(1) IS-A hierarchy queries.
/// Cones narrow as depth increases (children have smaller apertures).
///
/// # Constitution Reference
/// - perf.latency.entailment_check: <1ms
///
/// TODO: M04-T03 - Add aperture calculation helpers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConeConfig {
    /// Base aperture angle in radians (default: PI/4 = 45 degrees)
    pub base_aperture: f32,
    /// Aperture decay factor per depth level (default: 0.9)
    pub aperture_decay: f32,
    /// Minimum aperture angle (default: 0.1 radians)
    pub min_aperture: f32,
}

impl Default for ConeConfig {
    fn default() -> Self {
        Self {
            base_aperture: std::f32::consts::FRAC_PI_4,
            aperture_decay: 0.9,
            min_aperture: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_config_default_values() {
        let config = IndexConfig::default();
        assert_eq!(config.dimension, 1536);
        assert_eq!(config.nlist, 16384);
        assert_eq!(config.nprobe, 128);
        assert_eq!(config.pq_segments, 64);
        assert_eq!(config.pq_bits, 8);
        assert_eq!(config.gpu_id, 0);
        assert!(config.use_float16);
        assert_eq!(config.min_train_vectors, 4_194_304);
    }

    #[test]
    fn test_index_config_pq_segments_divides_dimension() {
        let config = IndexConfig::default();
        assert_eq!(
            config.dimension % config.pq_segments,
            0,
            "PQ segments must divide dimension evenly"
        );
    }

    #[test]
    fn test_index_config_min_train_vectors_formula() {
        let config = IndexConfig::default();
        assert_eq!(
            config.min_train_vectors,
            256 * config.nlist,
            "min_train_vectors must equal 256 * nlist"
        );
    }

    #[test]
    fn test_factory_string_default() {
        let config = IndexConfig::default();
        assert_eq!(config.factory_string(), "IVF16384,PQ64x8");
    }

    #[test]
    fn test_factory_string_custom() {
        let config = IndexConfig {
            dimension: 768,
            nlist: 4096,
            nprobe: 64,
            pq_segments: 32,
            pq_bits: 4,
            gpu_id: 1,
            use_float16: false,
            min_train_vectors: 256 * 4096,
        };
        assert_eq!(config.factory_string(), "IVF4096,PQ32x4");
    }

    #[test]
    fn test_calculate_min_train_vectors() {
        let config = IndexConfig::default();
        assert_eq!(config.calculate_min_train_vectors(), 4_194_304);

        let custom = IndexConfig {
            nlist: 1024,
            ..Default::default()
        };
        assert_eq!(custom.calculate_min_train_vectors(), 256 * 1024);
    }

    #[test]
    fn test_index_config_serialization_roundtrip() {
        let config = IndexConfig::default();
        let json = serde_json::to_string(&config).expect("Serialization failed");
        let deserialized: IndexConfig =
            serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_index_config_json_format() {
        let config = IndexConfig::default();
        let json = serde_json::to_string_pretty(&config).expect("Serialization failed");
        assert!(json.contains("\"dimension\": 1536"));
        assert!(json.contains("\"nlist\": 16384"));
        assert!(json.contains("\"nprobe\": 128"));
        assert!(json.contains("\"pq_segments\": 64"));
        assert!(json.contains("\"pq_bits\": 8"));
        assert!(json.contains("\"gpu_id\": 0"));
        assert!(json.contains("\"use_float16\": true"));
        assert!(json.contains("\"min_train_vectors\": 4194304"));
    }

    #[test]
    fn test_pq_bits_type_is_u8() {
        let config = IndexConfig::default();
        // This is a compile-time check - if pq_bits is not u8, this won't compile
        let _: u8 = config.pq_bits;
    }

    #[test]
    fn test_hyperbolic_config_default() {
        let config = HyperbolicConfig::default();

        // Verify all 4 fields
        assert_eq!(config.dim, 64, "Default dim must be 64");
        assert_eq!(config.curvature, -1.0, "Default curvature must be -1.0");
        assert_eq!(config.eps, 1e-7, "Default eps must be 1e-7");
        assert!((config.max_norm - 0.99999).abs() < 1e-10, "Default max_norm must be 1.0 - 1e-5");

        // Invariants
        assert!(config.curvature < 0.0, "Curvature must be negative");
        assert!(config.max_norm < 1.0, "Max norm must be < 1.0");
        assert!(config.max_norm > 0.0, "Max norm must be positive");
        assert!(config.eps > 0.0, "Eps must be positive");
    }

    #[test]
    fn test_hyperbolic_config_with_curvature() {
        let config = HyperbolicConfig::with_curvature(-0.5);
        assert_eq!(config.curvature, -0.5);
        assert_eq!(config.dim, 64); // defaults preserved
        assert_eq!(config.eps, 1e-7);
    }

    #[test]
    fn test_hyperbolic_config_abs_curvature() {
        let config = HyperbolicConfig::default();
        assert_eq!(config.abs_curvature(), 1.0);

        let config2 = HyperbolicConfig::with_curvature(-2.5);
        assert_eq!(config2.abs_curvature(), 2.5);
    }

    #[test]
    fn test_hyperbolic_config_scale() {
        let config = HyperbolicConfig::default();
        assert_eq!(config.scale(), 1.0); // sqrt(|-1.0|) = 1.0

        let config2 = HyperbolicConfig::with_curvature(-4.0);
        assert_eq!(config2.scale(), 2.0); // sqrt(|-4.0|) = 2.0
    }

    #[test]
    fn test_hyperbolic_config_serialization_roundtrip() {
        let config = HyperbolicConfig::default();
        let json = serde_json::to_string(&config).expect("Serialization failed");
        let deserialized: HyperbolicConfig = serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_hyperbolic_config_json_fields() {
        let config = HyperbolicConfig::default();
        let json = serde_json::to_string_pretty(&config).expect("Serialization failed");

        // Verify all 4 fields appear in JSON
        assert!(json.contains("\"dim\":"), "JSON must contain dim field");
        assert!(json.contains("\"curvature\":"), "JSON must contain curvature field");
        assert!(json.contains("\"eps\":"), "JSON must contain eps field");
        assert!(json.contains("\"max_norm\":"), "JSON must contain max_norm field");
    }

    // ============ Validation Tests ============

    #[test]
    fn test_validate_default_passes() {
        let config = HyperbolicConfig::default();
        assert!(config.validate().is_ok(), "Default config must be valid");
    }

    #[test]
    fn test_validate_dim_zero_fails() {
        let config = HyperbolicConfig {
            dim: 0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("dim"), "Error should mention 'dim'");
        assert!(err_msg.contains("positive"), "Error should mention 'positive'");
    }

    #[test]
    fn test_validate_curvature_zero_fails() {
        let config = HyperbolicConfig {
            curvature: 0.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("curvature"), "Error should mention 'curvature'");
        assert!(err_msg.contains("negative"), "Error should mention 'negative'");
    }

    #[test]
    fn test_validate_curvature_positive_fails() {
        let config = HyperbolicConfig {
            curvature: 1.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("1"), "Error should include the actual value");
    }

    #[test]
    fn test_validate_curvature_nan_fails() {
        let config = HyperbolicConfig {
            curvature: f32::NAN,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("NaN"), "Error should mention 'NaN'");
    }

    #[test]
    fn test_validate_eps_zero_fails() {
        let config = HyperbolicConfig {
            eps: 0.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("eps"), "Error should mention 'eps'");
    }

    #[test]
    fn test_validate_eps_negative_fails() {
        let config = HyperbolicConfig {
            eps: -1e-7,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_max_norm_zero_fails() {
        let config = HyperbolicConfig {
            max_norm: 0.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("max_norm"), "Error should mention 'max_norm'");
    }

    #[test]
    fn test_validate_max_norm_one_fails() {
        let config = HyperbolicConfig {
            max_norm: 1.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err(), "max_norm=1.0 is ON boundary, not inside ball");
    }

    #[test]
    fn test_validate_max_norm_greater_than_one_fails() {
        let config = HyperbolicConfig {
            max_norm: 1.5,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_max_norm_negative_fails() {
        let config = HyperbolicConfig {
            max_norm: -0.5,
            ..Default::default()
        };
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_custom_valid_curvature() {
        // Various valid negative curvatures
        for c in [-0.1, -0.5, -1.0, -2.0, -10.0] {
            let config = HyperbolicConfig::with_curvature(c);
            assert!(config.validate().is_ok(), "curvature {} should be valid", c);
        }
    }

    #[test]
    fn test_try_with_curvature_valid() {
        let config = HyperbolicConfig::try_with_curvature(-0.5).unwrap();
        assert_eq!(config.curvature, -0.5);
        assert_eq!(config.dim, 64); // default
    }

    #[test]
    fn test_try_with_curvature_invalid() {
        assert!(HyperbolicConfig::try_with_curvature(0.0).is_err());
        assert!(HyperbolicConfig::try_with_curvature(1.0).is_err());
        assert!(HyperbolicConfig::try_with_curvature(f32::NAN).is_err());
    }

    #[test]
    fn test_validate_fail_fast_order() {
        // When multiple fields are invalid, should fail on first check (dim)
        let config = HyperbolicConfig {
            dim: 0,
            curvature: 1.0,  // also invalid
            eps: -1.0,       // also invalid
            max_norm: 2.0,   // also invalid
        };
        let err_msg = config.validate().unwrap_err().to_string();
        assert!(err_msg.contains("dim"), "Should fail on dim first");
    }

    #[test]
    fn test_validate_boundary_values() {
        // Test values very close to boundaries
        let barely_valid = HyperbolicConfig {
            dim: 1,
            curvature: -1e-10,  // tiny but negative
            eps: 1e-10,         // tiny but positive
            max_norm: 0.9999999, // close to 1 but not 1
        };
        assert!(barely_valid.validate().is_ok());
    }

    #[test]
    fn test_cone_config_default() {
        let config = ConeConfig::default();
        assert!(config.base_aperture > 0.0);
        assert!(config.base_aperture < std::f32::consts::PI);
        assert!(config.aperture_decay > 0.0 && config.aperture_decay <= 1.0);
        assert!(config.min_aperture > 0.0);
        assert!(config.min_aperture < config.base_aperture);
    }


    #[test]
    fn test_cone_config_serialization() {
        let config = ConeConfig::default();
        let json = serde_json::to_string(&config).expect("Failed to serialize");
        let deserialized: ConeConfig =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(config.base_aperture, deserialized.base_aperture);
    }
}
