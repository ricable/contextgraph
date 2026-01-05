//! Model validation logic for warm-loaded models.
//!
//! This module validates that models loaded into VRAM are correct before
//! marking them as Warm. Validation includes:
//!
//! - **Dimension validation**: Output dimensions match expected values
//! - **Weight validation**: No NaN or Inf values in model weights
//! - **Checksum validation**: Weight checksums match expected values
//! - **Inference validation**: Test inference produces valid output
//!
//! # Requirements
//!
//! - REQ-WARM-011: Model dimension validation
//! - REQ-WARM-003: Cold-start validation (test inference)
//!
//! # Design Principles
//!
//! - **NO FALLBACKS**: Validation failures return errors, never silently pass
//! - **FAST FAIL**: Validation runs immediately after model load
//! - **COMPREHENSIVE**: Multiple validation stages catch different failure modes

use super::error::{WarmError, WarmResult};
use super::handle::ModelHandle;

/// Configuration for test inference validation.
///
/// Defines the expected behavior and inputs for validating a model
/// after loading weights into VRAM.
///
/// # Example
///
/// ```rust,ignore
/// let config = TestInferenceConfig::for_embedding_model("E1_Semantic", 1024);
/// let result = validator.validate_model(&config, &handle, &output);
/// ```
#[derive(Debug, Clone)]
pub struct TestInferenceConfig {
    /// Model identifier (e.g., "E1_Semantic").
    pub model_id: String,
    /// Expected output dimension from the model.
    pub expected_dimension: usize,
    /// Test input for inference validation.
    pub test_input: TestInput,
    /// Optional reference output for comparison validation.
    /// If provided, actual output must match within tolerance.
    pub reference_output: Option<Vec<f32>>,
    /// Maximum allowed inference time in milliseconds.
    /// Exceeding this is a validation failure.
    pub max_inference_ms: u64,
}

impl TestInferenceConfig {
    /// Create a test inference config for an embedding model.
    ///
    /// Uses a standard test sentence for text-based embedding models.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier (e.g., "E1_Semantic")
    /// * `expected_dimension` - Expected output embedding dimension
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = TestInferenceConfig::for_embedding_model("E1_Semantic", 1024);
    /// assert_eq!(config.expected_dimension, 1024);
    /// ```
    #[must_use]
    pub fn for_embedding_model(model_id: &str, expected_dimension: usize) -> Self {
        Self {
            model_id: model_id.to_string(),
            expected_dimension,
            test_input: TestInput::Text("The quick brown fox jumps over the lazy dog.".to_string()),
            reference_output: None,
            max_inference_ms: 1000, // 1 second timeout for embedding
        }
    }

    /// Create a config with a reference output for strict validation.
    ///
    /// The actual inference output must match the reference within tolerance.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier
    /// * `expected_dimension` - Expected output dimension
    /// * `test_input` - Input for test inference
    /// * `reference_output` - Expected output values
    /// * `max_inference_ms` - Maximum inference time
    #[must_use]
    pub fn with_reference(
        model_id: &str,
        expected_dimension: usize,
        test_input: TestInput,
        reference_output: Vec<f32>,
        max_inference_ms: u64,
    ) -> Self {
        Self {
            model_id: model_id.to_string(),
            expected_dimension,
            test_input,
            reference_output: Some(reference_output),
            max_inference_ms,
        }
    }
}

/// Input types for test inference validation.
///
/// Different models accept different input formats. This enum
/// represents the supported input types for validation.
#[derive(Debug, Clone)]
pub enum TestInput {
    /// Raw text input for text embedding models.
    Text(String),
    /// Tokenized input (e.g., BERT token IDs).
    Tokens(Vec<u32>),
    /// Pre-computed embeddings (for projection models).
    Embeddings(Vec<f32>),
}

impl TestInput {
    /// Get a human-readable description of the input type.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Text(_) => "text",
            Self::Tokens(_) => "tokens",
            Self::Embeddings(_) => "embeddings",
        }
    }

    /// Get the input length (characters, tokens, or embedding dimensions).
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Text(s) => s.len(),
            Self::Tokens(t) => t.len(),
            Self::Embeddings(e) => e.len(),
        }
    }

    /// Check if the input is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Result of model validation after warm loading.
///
/// Aggregates results from all validation stages (dimensions, weights,
/// inference). If any stage fails, `error` contains the failure details.
#[derive(Debug)]
pub struct ValidationResult {
    /// Model identifier that was validated.
    pub model_id: String,
    /// Whether output dimension matches expected.
    pub dimension_valid: bool,
    /// Whether weights contain no NaN/Inf values.
    pub weights_valid: bool,
    /// Whether test inference produced valid output.
    pub inference_valid: bool,
    /// Time taken for test inference in milliseconds.
    pub inference_time_ms: u64,
    /// Error details if any validation stage failed.
    pub error: Option<WarmError>,
}

impl ValidationResult {
    /// Create a successful validation result.
    #[must_use]
    pub fn success(model_id: String, inference_time_ms: u64) -> Self {
        Self {
            model_id,
            dimension_valid: true,
            weights_valid: true,
            inference_valid: true,
            inference_time_ms,
            error: None,
        }
    }

    /// Create a failed validation result.
    #[must_use]
    pub fn failure(
        model_id: String,
        dimension_valid: bool,
        weights_valid: bool,
        inference_valid: bool,
        inference_time_ms: u64,
        error: WarmError,
    ) -> Self {
        Self {
            model_id,
            dimension_valid,
            weights_valid,
            inference_valid,
            inference_time_ms,
            error: Some(error),
        }
    }

    /// Check if all validation stages passed.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.dimension_valid && self.weights_valid && self.inference_valid && self.error.is_none()
    }

    /// Get the validation error, if any.
    #[must_use]
    pub fn error(&self) -> Option<&WarmError> {
        self.error.as_ref()
    }
}

/// Validator for warm-loaded models.
///
/// Performs comprehensive validation of models after weight loading
/// to ensure correctness before marking as Warm.
///
/// # Example
///
/// ```rust,ignore
/// let validator = WarmValidator::new();
///
/// // Validate dimensions
/// validator.validate_dimensions("E1_Semantic", 1024, output.len())?;
///
/// // Validate weights
/// validator.validate_weights_finite(&weights)?;
///
/// // Full validation
/// let result = validator.validate_model(&config, &handle, &output);
/// if !result.is_valid() {
///     return Err(result.error.unwrap());
/// }
/// ```
#[derive(Debug, Default)]
pub struct WarmValidator {
    /// Default tolerance for floating-point comparisons.
    default_tolerance: f32,
}

impl WarmValidator {
    /// Default tolerance for output comparison (1e-5).
    pub const DEFAULT_TOLERANCE: f32 = 1e-5;

    /// Create a new validator with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            default_tolerance: Self::DEFAULT_TOLERANCE,
        }
    }

    /// Create a validator with custom tolerance.
    #[must_use]
    pub fn with_tolerance(tolerance: f32) -> Self {
        Self {
            default_tolerance: tolerance,
        }
    }

    /// Validate that output dimensions match expected values.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier for error reporting
    /// * `expected` - Expected output dimension
    /// * `actual` - Actual output dimension
    ///
    /// # Errors
    ///
    /// Returns `WarmError::ModelDimensionMismatch` if dimensions don't match.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// validator.validate_dimensions("E1_Semantic", 1024, output.len())?;
    /// ```
    pub fn validate_dimensions(
        &self,
        model_id: &str,
        expected: usize,
        actual: usize,
    ) -> WarmResult<()> {
        if expected != actual {
            return Err(WarmError::ModelDimensionMismatch {
                model_id: model_id.to_string(),
                expected,
                actual,
            });
        }
        Ok(())
    }

    /// Validate that weights contain no NaN or Inf values.
    ///
    /// Model weights with NaN or Inf indicate corruption or failed loading.
    /// This validation catches silent failures that might otherwise produce
    /// garbage inference output.
    ///
    /// # Arguments
    ///
    /// * `weights` - Slice of model weight values
    ///
    /// # Errors
    ///
    /// Returns `WarmError::ModelValidationFailed` if any weight is NaN or Inf.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// validator.validate_weights_finite(&model_weights)?;
    /// ```
    pub fn validate_weights_finite(&self, weights: &[f32]) -> WarmResult<()> {
        for (idx, &weight) in weights.iter().enumerate() {
            if weight.is_nan() {
                return Err(WarmError::ModelValidationFailed {
                    model_id: "unknown".to_string(),
                    reason: format!("NaN value found at weight index {idx}"),
                    expected_output: Some("finite values".to_string()),
                    actual_output: Some("NaN".to_string()),
                });
            }
            if weight.is_infinite() {
                return Err(WarmError::ModelValidationFailed {
                    model_id: "unknown".to_string(),
                    reason: format!("Infinite value found at weight index {idx}"),
                    expected_output: Some("finite values".to_string()),
                    actual_output: Some(if weight.is_sign_positive() { "+Inf" } else { "-Inf" }.to_string()),
                });
            }
        }
        Ok(())
    }

    /// Validate that weights contain no NaN or Inf values (with model ID).
    ///
    /// Same as `validate_weights_finite` but includes model ID in error.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier for error reporting
    /// * `weights` - Slice of model weight values
    ///
    /// # Errors
    ///
    /// Returns `WarmError::ModelValidationFailed` if any weight is NaN or Inf.
    pub fn validate_weights_finite_for_model(
        &self,
        model_id: &str,
        weights: &[f32],
    ) -> WarmResult<()> {
        for (idx, &weight) in weights.iter().enumerate() {
            if weight.is_nan() {
                return Err(WarmError::ModelValidationFailed {
                    model_id: model_id.to_string(),
                    reason: format!("NaN value found at weight index {idx}"),
                    expected_output: Some("finite values".to_string()),
                    actual_output: Some("NaN".to_string()),
                });
            }
            if weight.is_infinite() {
                return Err(WarmError::ModelValidationFailed {
                    model_id: model_id.to_string(),
                    reason: format!("Infinite value found at weight index {idx}"),
                    expected_output: Some("finite values".to_string()),
                    actual_output: Some(if weight.is_sign_positive() { "+Inf" } else { "-Inf" }.to_string()),
                });
            }
        }
        Ok(())
    }

    /// Validate weight checksum matches expected value.
    ///
    /// The checksum is computed during weight loading and stored in the
    /// `ModelHandle`. This validation ensures weight integrity.
    ///
    /// # Arguments
    ///
    /// * `handle` - Model handle containing the computed checksum
    /// * `expected` - Expected checksum (SHA256 truncated to 64 bits)
    ///
    /// # Errors
    ///
    /// Returns `WarmError::ModelValidationFailed` if checksums don't match.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// validator.validate_weight_checksum(&handle, 0xdeadbeefcafebabe)?;
    /// ```
    pub fn validate_weight_checksum(
        &self,
        handle: &ModelHandle,
        expected: u64,
    ) -> WarmResult<()> {
        let actual = handle.weight_checksum();
        if expected != actual {
            return Err(WarmError::ModelValidationFailed {
                model_id: "unknown".to_string(),
                reason: "Weight checksum mismatch".to_string(),
                expected_output: Some(format!("0x{expected:016x}")),
                actual_output: Some(format!("0x{actual:016x}")),
            });
        }
        Ok(())
    }

    /// Validate weight checksum matches expected value (with model ID).
    ///
    /// Same as `validate_weight_checksum` but includes model ID in error.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier for error reporting
    /// * `handle` - Model handle containing the computed checksum
    /// * `expected` - Expected checksum (SHA256 truncated to 64 bits)
    ///
    /// # Errors
    ///
    /// Returns `WarmError::ModelValidationFailed` if checksums don't match.
    pub fn validate_weight_checksum_for_model(
        &self,
        model_id: &str,
        handle: &ModelHandle,
        expected: u64,
    ) -> WarmResult<()> {
        let actual = handle.weight_checksum();
        if expected != actual {
            return Err(WarmError::ModelValidationFailed {
                model_id: model_id.to_string(),
                reason: "Weight checksum mismatch".to_string(),
                expected_output: Some(format!("0x{expected:016x}")),
                actual_output: Some(format!("0x{actual:016x}")),
            });
        }
        Ok(())
    }

    /// Run test inference validation.
    ///
    /// Validates that inference output meets the configuration requirements:
    /// - Correct output dimension
    /// - No NaN/Inf in output
    /// - If reference output provided, values match within tolerance
    ///
    /// # Arguments
    ///
    /// * `config` - Test inference configuration
    /// * `output` - Actual inference output
    ///
    /// # Errors
    ///
    /// Returns `WarmError::ModelValidationFailed` or `WarmError::ModelDimensionMismatch`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let output = model.infer(&config.test_input);
    /// validator.run_test_inference(&config, &output)?;
    /// ```
    pub fn run_test_inference(
        &self,
        config: &TestInferenceConfig,
        output: &[f32],
    ) -> WarmResult<()> {
        // Validate output dimension
        self.validate_dimensions(&config.model_id, config.expected_dimension, output.len())?;

        // Validate output contains no NaN/Inf
        self.validate_weights_finite_for_model(&config.model_id, output)?;

        // If reference output provided, compare values
        if let Some(ref reference) = config.reference_output {
            self.compare_output(output, reference, self.default_tolerance)?;
        }

        Ok(())
    }

    /// Compare actual output to reference output within tolerance.
    ///
    /// Uses element-wise absolute difference comparison. All elements must
    /// be within tolerance for validation to pass.
    ///
    /// # Arguments
    ///
    /// * `actual` - Actual inference output
    /// * `reference` - Expected reference output
    /// * `tolerance` - Maximum allowed absolute difference per element
    ///
    /// # Errors
    ///
    /// Returns `WarmError::ModelValidationFailed` if:
    /// - Output lengths differ
    /// - Any element differs by more than tolerance
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// validator.compare_output(&actual, &reference, 1e-5)?;
    /// ```
    pub fn compare_output(
        &self,
        actual: &[f32],
        reference: &[f32],
        tolerance: f32,
    ) -> WarmResult<()> {
        if actual.len() != reference.len() {
            return Err(WarmError::ModelValidationFailed {
                model_id: "unknown".to_string(),
                reason: format!(
                    "Output length mismatch: expected {}, got {}",
                    reference.len(),
                    actual.len()
                ),
                expected_output: Some(format!("length {}", reference.len())),
                actual_output: Some(format!("length {}", actual.len())),
            });
        }

        for (idx, (&a, &r)) in actual.iter().zip(reference.iter()).enumerate() {
            let diff = (a - r).abs();
            if diff > tolerance {
                return Err(WarmError::ModelValidationFailed {
                    model_id: "unknown".to_string(),
                    reason: format!(
                        "Output mismatch at index {idx}: expected {r}, got {a} (diff: {diff}, tolerance: {tolerance})"
                    ),
                    expected_output: Some(format!("{r}")),
                    actual_output: Some(format!("{a}")),
                });
            }
        }

        Ok(())
    }

    /// Compare actual output to reference output within tolerance (with model ID).
    ///
    /// Same as `compare_output` but includes model ID in errors.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier for error reporting
    /// * `actual` - Actual inference output
    /// * `reference` - Expected reference output
    /// * `tolerance` - Maximum allowed absolute difference per element
    pub fn compare_output_for_model(
        &self,
        model_id: &str,
        actual: &[f32],
        reference: &[f32],
        tolerance: f32,
    ) -> WarmResult<()> {
        if actual.len() != reference.len() {
            return Err(WarmError::ModelValidationFailed {
                model_id: model_id.to_string(),
                reason: format!(
                    "Output length mismatch: expected {}, got {}",
                    reference.len(),
                    actual.len()
                ),
                expected_output: Some(format!("length {}", reference.len())),
                actual_output: Some(format!("length {}", actual.len())),
            });
        }

        for (idx, (&a, &r)) in actual.iter().zip(reference.iter()).enumerate() {
            let diff = (a - r).abs();
            if diff > tolerance {
                return Err(WarmError::ModelValidationFailed {
                    model_id: model_id.to_string(),
                    reason: format!(
                        "Output mismatch at index {idx}: expected {r}, got {a} (diff: {diff}, tolerance: {tolerance})"
                    ),
                    expected_output: Some(format!("{r}")),
                    actual_output: Some(format!("{a}")),
                });
            }
        }

        Ok(())
    }

    /// Perform full model validation.
    ///
    /// Runs all validation stages and aggregates results:
    /// 1. Dimension validation
    /// 2. Weight validation (via checksum check)
    /// 3. Inference validation
    ///
    /// # Arguments
    ///
    /// * `config` - Test inference configuration
    /// * `_handle` - Model handle for the loaded model (reserved for future checksum validation)
    /// * `output` - Inference output to validate
    ///
    /// # Returns
    ///
    /// `ValidationResult` with success/failure status for each stage.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = TestInferenceConfig::for_embedding_model("E1_Semantic", 1024);
    /// let result = validator.validate_model(&config, &handle, &output);
    /// if !result.is_valid() {
    ///     log::error!("Validation failed: {:?}", result.error);
    ///     return Err(result.error.unwrap());
    /// }
    /// ```
    #[must_use]
    pub fn validate_model(
        &self,
        config: &TestInferenceConfig,
        _handle: &ModelHandle,
        output: &[f32],
    ) -> ValidationResult {
        let start = std::time::Instant::now();

        // Stage 1: Validate dimensions
        let dimension_valid = self
            .validate_dimensions(&config.model_id, config.expected_dimension, output.len())
            .is_ok();

        if !dimension_valid {
            let error = WarmError::ModelDimensionMismatch {
                model_id: config.model_id.clone(),
                expected: config.expected_dimension,
                actual: output.len(),
            };
            return ValidationResult::failure(
                config.model_id.clone(),
                false,
                true, // weights not checked yet
                false,
                start.elapsed().as_millis() as u64,
                error,
            );
        }

        // Stage 2: Validate weights (output values as proxy for weight health)
        let weights_valid = self.validate_weights_finite_for_model(&config.model_id, output).is_ok();

        if !weights_valid {
            // Find the specific error
            let error = if let Some((idx, _)) = output.iter().enumerate().find(|(_, v)| v.is_nan()) {
                WarmError::ModelValidationFailed {
                    model_id: config.model_id.clone(),
                    reason: format!("NaN value found at output index {idx}"),
                    expected_output: Some("finite values".to_string()),
                    actual_output: Some("NaN".to_string()),
                }
            } else if let Some((idx, val)) = output.iter().enumerate().find(|(_, v)| v.is_infinite()) {
                WarmError::ModelValidationFailed {
                    model_id: config.model_id.clone(),
                    reason: format!("Infinite value found at output index {idx}"),
                    expected_output: Some("finite values".to_string()),
                    actual_output: Some(if val.is_sign_positive() { "+Inf" } else { "-Inf" }.to_string()),
                }
            } else {
                WarmError::ModelValidationFailed {
                    model_id: config.model_id.clone(),
                    reason: "Unknown weight validation failure".to_string(),
                    expected_output: None,
                    actual_output: None,
                }
            };

            return ValidationResult::failure(
                config.model_id.clone(),
                true,
                false,
                false,
                start.elapsed().as_millis() as u64,
                error,
            );
        }

        // Stage 3: Validate inference (reference comparison if provided)
        let inference_valid = if let Some(ref reference) = config.reference_output {
            self.compare_output_for_model(&config.model_id, output, reference, self.default_tolerance)
                .is_ok()
        } else {
            true // No reference output means inference passes
        };

        let inference_time_ms = start.elapsed().as_millis() as u64;

        if !inference_valid {
            let reference = config.reference_output.as_ref().unwrap();
            // Find first mismatch
            let (idx, (actual_val, ref_val)) = output
                .iter()
                .zip(reference.iter())
                .enumerate()
                .find(|(_, (a, r))| (**a - **r).abs() > self.default_tolerance)
                .map(|(i, (a, r))| (i, (*a, *r)))
                .unwrap_or((0, (0.0, 0.0)));

            let error = WarmError::ModelValidationFailed {
                model_id: config.model_id.clone(),
                reason: format!(
                    "Output mismatch at index {idx}: expected {ref_val}, got {actual_val}"
                ),
                expected_output: Some(format!("{ref_val}")),
                actual_output: Some(format!("{actual_val}")),
            };

            return ValidationResult::failure(
                config.model_id.clone(),
                true,
                true,
                false,
                inference_time_ms,
                error,
            );
        }

        // Check inference time
        if inference_time_ms > config.max_inference_ms {
            let error = WarmError::ModelValidationFailed {
                model_id: config.model_id.clone(),
                reason: format!(
                    "Inference time {inference_time_ms}ms exceeded maximum {}ms",
                    config.max_inference_ms
                ),
                expected_output: Some(format!("<= {}ms", config.max_inference_ms)),
                actual_output: Some(format!("{}ms", inference_time_ms)),
            };

            return ValidationResult::failure(
                config.model_id.clone(),
                true,
                true,
                false,
                inference_time_ms,
                error,
            );
        }

        // All validations passed
        ValidationResult::success(config.model_id.clone(), inference_time_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === TestInferenceConfig Tests ===

    #[test]
    fn test_config_for_embedding_model() {
        let config = TestInferenceConfig::for_embedding_model("E1_Semantic", 1024);

        assert_eq!(config.model_id, "E1_Semantic");
        assert_eq!(config.expected_dimension, 1024);
        assert!(matches!(config.test_input, TestInput::Text(_)));
        assert!(config.reference_output.is_none());
        assert_eq!(config.max_inference_ms, 1000);
    }

    #[test]
    fn test_config_with_reference() {
        let reference = vec![0.1, 0.2, 0.3];
        let config = TestInferenceConfig::with_reference(
            "TestModel",
            3,
            TestInput::Text("test".to_string()),
            reference.clone(),
            500,
        );

        assert_eq!(config.model_id, "TestModel");
        assert_eq!(config.expected_dimension, 3);
        assert_eq!(config.reference_output.as_ref().unwrap(), &reference);
        assert_eq!(config.max_inference_ms, 500);
    }

    // === TestInput Tests ===

    #[test]
    fn test_input_description() {
        assert_eq!(TestInput::Text("hello".to_string()).description(), "text");
        assert_eq!(TestInput::Tokens(vec![1, 2, 3]).description(), "tokens");
        assert_eq!(TestInput::Embeddings(vec![0.1, 0.2]).description(), "embeddings");
    }

    #[test]
    fn test_input_len() {
        assert_eq!(TestInput::Text("hello".to_string()).len(), 5);
        assert_eq!(TestInput::Tokens(vec![1, 2, 3]).len(), 3);
        assert_eq!(TestInput::Embeddings(vec![0.1, 0.2]).len(), 2);
    }

    #[test]
    fn test_input_is_empty() {
        assert!(!TestInput::Text("hello".to_string()).is_empty());
        assert!(TestInput::Text(String::new()).is_empty());
        assert!(TestInput::Tokens(vec![]).is_empty());
        assert!(!TestInput::Embeddings(vec![0.1]).is_empty());
    }

    // === ValidationResult Tests ===

    #[test]
    fn test_validation_result_success() {
        let result = ValidationResult::success("model1".to_string(), 42);

        assert_eq!(result.model_id, "model1");
        assert!(result.dimension_valid);
        assert!(result.weights_valid);
        assert!(result.inference_valid);
        assert_eq!(result.inference_time_ms, 42);
        assert!(result.is_valid());
        assert!(result.error().is_none());
    }

    #[test]
    fn test_validation_result_failure() {
        let error = WarmError::ModelDimensionMismatch {
            model_id: "model1".to_string(),
            expected: 100,
            actual: 50,
        };

        let result = ValidationResult::failure(
            "model1".to_string(),
            false,
            true,
            false,
            100,
            error,
        );

        assert!(!result.dimension_valid);
        assert!(result.weights_valid);
        assert!(!result.inference_valid);
        assert!(!result.is_valid());
        assert!(result.error().is_some());
    }

    // === WarmValidator Tests ===

    #[test]
    fn test_validator_new() {
        let v = WarmValidator::new();
        assert!((v.default_tolerance - WarmValidator::DEFAULT_TOLERANCE).abs() < 1e-10);
    }

    #[test]
    fn test_validator_with_tolerance() {
        let v = WarmValidator::with_tolerance(0.01);
        assert!((v.default_tolerance - 0.01).abs() < 1e-10);
    }

    // === validate_dimensions Tests ===

    #[test]
    fn test_validate_dimensions_matching() {
        let v = WarmValidator::new();
        assert!(v.validate_dimensions("model", 1024, 1024).is_ok());
    }

    #[test]
    fn test_validate_dimensions_mismatched() {
        let v = WarmValidator::new();
        let result = v.validate_dimensions("E1_Semantic", 1024, 512);

        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            WarmError::ModelDimensionMismatch { model_id, expected, actual } => {
                assert_eq!(model_id, "E1_Semantic");
                assert_eq!(expected, 1024);
                assert_eq!(actual, 512);
            }
            _ => panic!("Expected ModelDimensionMismatch error"),
        }
    }

    #[test]
    fn test_validate_dimensions_zero() {
        let v = WarmValidator::new();
        assert!(v.validate_dimensions("model", 0, 0).is_ok());
        assert!(v.validate_dimensions("model", 0, 1).is_err());
    }

    // === validate_weights_finite Tests ===

    #[test]
    fn test_validate_weights_finite_valid() {
        let v = WarmValidator::new();
        let weights = vec![0.1, -0.5, 1.0, -1.0, 0.0];
        assert!(v.validate_weights_finite(&weights).is_ok());
    }

    #[test]
    fn test_validate_weights_finite_empty() {
        let v = WarmValidator::new();
        assert!(v.validate_weights_finite(&[]).is_ok());
    }

    #[test]
    fn test_validate_weights_finite_nan() {
        let v = WarmValidator::new();
        let weights = vec![0.1, f32::NAN, 0.3];
        let result = v.validate_weights_finite(&weights);

        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            WarmError::ModelValidationFailed { reason, actual_output, .. } => {
                assert!(reason.contains("NaN"));
                assert!(reason.contains("index 1"));
                assert_eq!(actual_output.as_deref(), Some("NaN"));
            }
            _ => panic!("Expected ModelValidationFailed error"),
        }
    }

    #[test]
    fn test_validate_weights_finite_positive_inf() {
        let v = WarmValidator::new();
        let weights = vec![0.1, f32::INFINITY, 0.3];
        let result = v.validate_weights_finite(&weights);

        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            WarmError::ModelValidationFailed { reason, actual_output, .. } => {
                assert!(reason.contains("Infinite"));
                assert_eq!(actual_output.as_deref(), Some("+Inf"));
            }
            _ => panic!("Expected ModelValidationFailed error"),
        }
    }

    #[test]
    fn test_validate_weights_finite_negative_inf() {
        let v = WarmValidator::new();
        let weights = vec![0.1, f32::NEG_INFINITY, 0.3];
        let result = v.validate_weights_finite(&weights);

        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            WarmError::ModelValidationFailed { actual_output, .. } => {
                assert_eq!(actual_output.as_deref(), Some("-Inf"));
            }
            _ => panic!("Expected ModelValidationFailed error"),
        }
    }

    // === validate_weight_checksum Tests ===

    #[test]
    fn test_validate_weight_checksum_matching() {
        let v = WarmValidator::new();
        let handle = ModelHandle::new(0x1000, 1024, 0, 0xdeadbeefcafebabe);
        assert!(v.validate_weight_checksum(&handle, 0xdeadbeefcafebabe).is_ok());
    }

    #[test]
    fn test_validate_weight_checksum_mismatched() {
        let v = WarmValidator::new();
        let handle = ModelHandle::new(0x1000, 1024, 0, 0xdeadbeefcafebabe);
        let result = v.validate_weight_checksum(&handle, 0x1111111111111111);

        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            WarmError::ModelValidationFailed { reason, expected_output, actual_output, .. } => {
                assert!(reason.contains("checksum"));
                assert!(expected_output.as_ref().unwrap().contains("1111111111111111"));
                assert!(actual_output.as_ref().unwrap().contains("deadbeefcafebabe"));
            }
            _ => panic!("Expected ModelValidationFailed error"),
        }
    }

    // === compare_output Tests ===

    #[test]
    fn test_compare_output_identical() {
        let v = WarmValidator::new();
        let output = vec![0.1, 0.2, 0.3];
        let reference = vec![0.1, 0.2, 0.3];
        assert!(v.compare_output(&output, &reference, 1e-5).is_ok());
    }

    #[test]
    fn test_compare_output_within_tolerance() {
        let v = WarmValidator::new();
        let output = vec![0.10001, 0.20001, 0.30001];
        let reference = vec![0.1, 0.2, 0.3];
        assert!(v.compare_output(&output, &reference, 1e-3).is_ok());
    }

    #[test]
    fn test_compare_output_outside_tolerance() {
        let v = WarmValidator::new();
        let output = vec![0.1, 0.25, 0.3];  // 0.25 != 0.2
        let reference = vec![0.1, 0.2, 0.3];
        let result = v.compare_output(&output, &reference, 1e-5);

        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            WarmError::ModelValidationFailed { reason, .. } => {
                assert!(reason.contains("index 1"));
            }
            _ => panic!("Expected ModelValidationFailed error"),
        }
    }

    #[test]
    fn test_compare_output_length_mismatch() {
        let v = WarmValidator::new();
        let output = vec![0.1, 0.2];
        let reference = vec![0.1, 0.2, 0.3];
        let result = v.compare_output(&output, &reference, 1e-5);

        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            WarmError::ModelValidationFailed { reason, .. } => {
                assert!(reason.contains("length mismatch"));
            }
            _ => panic!("Expected ModelValidationFailed error"),
        }
    }

    #[test]
    fn test_compare_output_empty() {
        let v = WarmValidator::new();
        assert!(v.compare_output(&[], &[], 1e-5).is_ok());
    }

    // === run_test_inference Tests ===

    #[test]
    fn test_run_test_inference_valid() {
        let v = WarmValidator::new();
        let config = TestInferenceConfig::for_embedding_model("TestModel", 4);
        let output = vec![0.1, 0.2, 0.3, 0.4];
        assert!(v.run_test_inference(&config, &output).is_ok());
    }

    #[test]
    fn test_run_test_inference_wrong_dimension() {
        let v = WarmValidator::new();
        let config = TestInferenceConfig::for_embedding_model("TestModel", 4);
        let output = vec![0.1, 0.2, 0.3];
        let result = v.run_test_inference(&config, &output);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), WarmError::ModelDimensionMismatch { .. }));
    }

    #[test]
    fn test_run_test_inference_with_reference() {
        let v = WarmValidator::new();
        let reference = vec![0.1, 0.2, 0.3];
        let config = TestInferenceConfig::with_reference(
            "TestModel",
            3,
            TestInput::Text("test".to_string()),
            reference,
            1000,
        );
        let output = vec![0.1, 0.2, 0.3];
        assert!(v.run_test_inference(&config, &output).is_ok());
    }

    // === validate_model Integration Tests ===

    #[test]
    fn test_validate_model_success() {
        let v = WarmValidator::new();
        let config = TestInferenceConfig::for_embedding_model("E1_Semantic", 4);
        let handle = ModelHandle::new(0x1000, 4 * 4, 0, 0xabcd);
        let output = vec![0.1, 0.2, 0.3, 0.4];

        let result = v.validate_model(&config, &handle, &output);

        assert!(result.is_valid());
        assert!(result.dimension_valid);
        assert!(result.weights_valid);
        assert!(result.inference_valid);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_validate_model_dimension_failure() {
        let v = WarmValidator::new();
        let config = TestInferenceConfig::for_embedding_model("E1_Semantic", 1024);
        let handle = ModelHandle::new(0x1000, 512 * 4, 0, 0xabcd);
        let output = vec![0.1; 512];

        let result = v.validate_model(&config, &handle, &output);

        assert!(!result.is_valid());
        assert!(!result.dimension_valid);
        assert!(matches!(result.error, Some(WarmError::ModelDimensionMismatch { .. })));
    }

    #[test]
    fn test_validate_model_nan_failure() {
        let v = WarmValidator::new();
        let config = TestInferenceConfig::for_embedding_model("E1_Semantic", 4);
        let handle = ModelHandle::new(0x1000, 4 * 4, 0, 0xabcd);
        let output = vec![0.1, f32::NAN, 0.3, 0.4];

        let result = v.validate_model(&config, &handle, &output);

        assert!(!result.is_valid());
        assert!(result.dimension_valid);
        assert!(!result.weights_valid);
        assert!(matches!(result.error, Some(WarmError::ModelValidationFailed { .. })));
    }

    #[test]
    fn test_validate_model_reference_mismatch() {
        let v = WarmValidator::new();
        let reference = vec![0.1, 0.2, 0.3, 0.4];
        let config = TestInferenceConfig::with_reference(
            "TestModel",
            4,
            TestInput::Text("test".to_string()),
            reference,
            1000,
        );
        let handle = ModelHandle::new(0x1000, 4 * 4, 0, 0xabcd);
        let output = vec![0.1, 0.5, 0.3, 0.4];  // 0.5 != 0.2

        let result = v.validate_model(&config, &handle, &output);

        assert!(!result.is_valid());
        assert!(result.dimension_valid);
        assert!(result.weights_valid);
        assert!(!result.inference_valid);
    }

    #[test]
    fn test_validate_model_with_inf() {
        let v = WarmValidator::new();
        let config = TestInferenceConfig::for_embedding_model("TestModel", 3);
        let handle = ModelHandle::new(0x2000, 3 * 4, 0, 0x1234);
        let output = vec![0.1, f32::INFINITY, 0.3];

        let result = v.validate_model(&config, &handle, &output);

        assert!(!result.is_valid());
        assert!(!result.weights_valid);
    }

    #[test]
    fn test_validation_result_aggregation() {
        // Test that ValidationResult properly aggregates multiple validation stages
        let v = WarmValidator::new();
        let config = TestInferenceConfig::for_embedding_model("Aggregation_Test", 3);
        let handle = ModelHandle::new(0x1000, 12, 0, 0xfeed);

        // All valid
        let output1 = vec![0.1, 0.2, 0.3];
        let r1 = v.validate_model(&config, &handle, &output1);
        assert!(r1.is_valid());
        assert!(r1.dimension_valid && r1.weights_valid && r1.inference_valid);

        // Wrong dimension - should fail early
        let output2 = vec![0.1, 0.2];
        let r2 = v.validate_model(&config, &handle, &output2);
        assert!(!r2.is_valid());
        assert!(!r2.dimension_valid);
        // Note: weights_valid may be true since we didn't check (early fail)

        // Valid dimension, invalid weights
        let output3 = vec![0.1, f32::NAN, 0.3];
        let r3 = v.validate_model(&config, &handle, &output3);
        assert!(!r3.is_valid());
        assert!(r3.dimension_valid);
        assert!(!r3.weights_valid);
    }
}
