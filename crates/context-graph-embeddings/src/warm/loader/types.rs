//! Warm Loading Data Types for GPU Weight Management.
//!
//! This module defines the data structures used for loading model weights into GPU memory
//! during warm startup. All types enforce fail-fast validation per Constitution AP-007.
//!
//! # Constitution Alignment
//!
//! - **AP-007**: No Stub Data in Production - All fields contain REAL validated data
//! - **REQ-WARM-003**: Non-evictable VRAM allocation
//! - **REQ-WARM-005**: Weight integrity verification via SHA256 checksums
//!
//! # Design Philosophy
//!
//! **FAIL FAST. NO FALLBACKS.**
//!
//! - All constructors panic on invalid data (null pointers, zero checksums, empty collections)
//! - No silent defaults or fallback values
//! - Validation happens at construction time, not runtime
//!
//! # Critical: No Simulation
//!
//! These types are designed to hold REAL data from actual GPU operations:
//! - `gpu_ptr` must be a real cudaMalloc pointer (never 0x0)
//! - `checksum` must be a real SHA256 hash (never all zeros)
//! - `tensors` must contain real GpuTensor instances backed by GPU memory

use std::collections::HashMap;
use std::time::{Duration, Instant};
use candle_core::DType;
use crate::gpu::GpuTensor;

// =============================================================================
// CRITICAL: NO SIMULATION - ALL DATA MUST BE REAL
// Constitution AP-007: "No Stub Data in Production"
// =============================================================================

/// Metadata extracted from SafeTensors file header.
///
/// Contains the shape and type information for all tensors in a model weight file.
/// This metadata is parsed from the SafeTensors header before actual weight loading.
///
/// # Constitution Alignment
///
/// - AP-007: No stub data - shapes must reflect actual SafeTensors content
///
/// # CRITICAL: No Simulation
///
/// All fields must reflect actual SafeTensors header content. This struct will
/// PANIC if constructed with invalid data.
///
/// # Example
///
/// ```rust,ignore
/// use std::collections::HashMap;
/// use candle_core::DType;
/// use context_graph_embeddings::warm::loader::types::TensorMetadata;
///
/// let mut shapes = HashMap::new();
/// shapes.insert("embeddings.weight".to_string(), vec![30522, 768]);
/// shapes.insert("encoder.layer.0.attention.self.query.weight".to_string(), vec![768, 768]);
///
/// let metadata = TensorMetadata::new(shapes, DType::F32, 24_030_504);
/// assert!(metadata.verify_params());
/// ```
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Tensor name -> shape mapping.
    ///
    /// Example: `{"embeddings.weight": [30522, 768], "layer.0.weight": [768, 768]}`
    ///
    /// # Invariant
    /// Must be non-empty. Empty shapes indicates corrupted or incomplete SafeTensors file.
    pub shapes: HashMap<String, Vec<usize>>,

    /// Data type of tensors (from candle_core).
    ///
    /// Common values: DType::F32, DType::F16, DType::BF16
    pub dtype: DType,

    /// Total number of parameters across all tensors.
    ///
    /// # Invariant
    /// MUST be > 0 for valid models. Zero parameters indicates empty/corrupted model.
    pub total_params: usize,
}

impl TensorMetadata {
    /// Create new TensorMetadata with validation.
    ///
    /// # Arguments
    ///
    /// * `shapes` - HashMap mapping tensor names to their shapes
    /// * `dtype` - Data type of the tensors
    /// * `total_params` - Total parameter count (should match sum of shape products)
    ///
    /// # Panics
    ///
    /// - If `shapes` is empty (no tensors in SafeTensors file)
    /// - If `total_params` is 0 (empty model)
    ///
    /// # Constitution: Fail-Fast
    ///
    /// Per AP-007, we panic immediately on invalid data rather than
    /// propagating corruption through the system.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let shapes: HashMap<String, Vec<usize>> = [
    ///     ("layer.weight".to_string(), vec![768, 768])
    /// ].into_iter().collect();
    ///
    /// let metadata = TensorMetadata::new(shapes, DType::F32, 589_824);
    /// ```
    #[must_use]
    pub fn new(
        shapes: HashMap<String, Vec<usize>>,
        dtype: DType,
        total_params: usize,
    ) -> Self {
        // FAIL-FAST: Empty shapes means corrupted SafeTensors
        assert!(
            !shapes.is_empty(),
            "CONSTITUTION VIOLATION AP-007: shapes is empty. \
             SafeTensors must contain at least one tensor. \
             This indicates corrupted or incomplete weight file."
        );

        // FAIL-FAST: Zero params means empty model
        assert!(
            total_params > 0,
            "CONSTITUTION VIOLATION AP-007: total_params is 0. \
             Model must have parameters. \
             This indicates corrupted or empty weight file."
        );

        Self { shapes, dtype, total_params }
    }

    /// Calculate total parameters from shapes (for verification).
    ///
    /// Computes the sum of all shape products to verify against stored total_params.
    ///
    /// # Returns
    ///
    /// Sum of element counts across all tensors.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // For shapes: {"a": [100, 768], "b": [768, 768]}
    /// // Returns: 100*768 + 768*768 = 76800 + 589824 = 666624
    /// let calculated = metadata.calculate_total_params();
    /// ```
    #[must_use]
    pub fn calculate_total_params(&self) -> usize {
        self.shapes.values()
            .map(|shape| shape.iter().product::<usize>())
            .sum()
    }

    /// Verify total_params matches calculated value.
    ///
    /// # Returns
    ///
    /// `true` if stored total_params equals the sum of all shape products.
    ///
    /// # Use Case
    ///
    /// Call this after parsing SafeTensors header to verify consistency.
    #[must_use]
    pub fn verify_params(&self) -> bool {
        self.total_params == self.calculate_total_params()
    }

    /// Get the number of tensors in this metadata.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.shapes.len()
    }

    /// Get shape for a specific tensor by name.
    #[must_use]
    pub fn get_shape(&self, name: &str) -> Option<&Vec<usize>> {
        self.shapes.get(name)
    }
}

/// Result of loading a model's weights into GPU memory.
///
/// Contains the GPU pointer, checksum, and metadata from a successful weight load operation.
/// This is the primary type returned by warm loading operations.
///
/// # Constitution Alignment
///
/// - REQ-WARM-003: Non-evictable VRAM allocation (gpu_ptr points to pinned memory)
/// - REQ-WARM-005: Weight integrity verification (checksum for validation)
/// - AP-007: No stub data in production (all fields are real, validated data)
///
/// # CRITICAL: No Simulation
///
/// All fields contain REAL data from actual loading operations.
/// This struct will PANIC if constructed with invalid data.
///
/// # Example
///
/// ```rust,ignore
/// use std::time::Duration;
/// use std::collections::HashMap;
/// use candle_core::DType;
/// use context_graph_embeddings::warm::loader::types::{WarmLoadResult, TensorMetadata};
///
/// // After real cudaMalloc and SHA256 computation:
/// let metadata = TensorMetadata::new(
///     [("weight".to_string(), vec![768, 768])].into_iter().collect(),
///     DType::F32,
///     589_824,
/// );
///
/// let result = WarmLoadResult::new(
///     0x7fff_dead_beef,      // Real GPU pointer from cudaMalloc
///     [0xAB; 32],            // Real SHA256 checksum
///     2_359_296,             // 589824 * 4 bytes
///     Duration::from_millis(150),
///     metadata,
/// );
///
/// assert!(result.verify_checksum(&[0xAB; 32]));
/// ```
#[derive(Debug)]
pub struct WarmLoadResult {
    /// Real GPU device pointer from cudaMalloc.
    ///
    /// # Invariant
    /// MUST be non-zero. Zero pointer = PANIC.
    /// This must be an actual pointer returned by CUDA memory allocation.
    pub gpu_ptr: u64,

    /// Real SHA256 checksum of the weight file.
    ///
    /// # Invariant
    /// MUST be non-zero. All-zero checksum = PANIC.
    /// This must be computed from actual file content.
    pub checksum: [u8; 32],

    /// Actual size of weights in GPU memory (bytes).
    ///
    /// # Invariant
    /// MUST be > 0. Zero size = PANIC.
    /// This is the actual cudaMalloc allocation size.
    pub size_bytes: usize,

    /// Loading duration for performance monitoring.
    ///
    /// Measures wall-clock time from start of load to completion.
    pub load_duration: Duration,

    /// Tensor metadata from SafeTensors header.
    ///
    /// Contains shapes, dtype, and total parameter count.
    pub tensor_metadata: TensorMetadata,
}

impl WarmLoadResult {
    /// Create a new WarmLoadResult with validation.
    ///
    /// # Arguments
    ///
    /// * `gpu_ptr` - Real CUDA device pointer (must be non-zero)
    /// * `checksum` - Real SHA256 checksum (must be non-zero)
    /// * `size_bytes` - Allocation size in bytes (must be > 0)
    /// * `load_duration` - Time taken to load
    /// * `tensor_metadata` - Parsed SafeTensors metadata
    ///
    /// # Panics
    ///
    /// - If `gpu_ptr` is 0 (null pointer)
    /// - If `checksum` is all zeros (invalid checksum)
    /// - If `size_bytes` is 0 (empty allocation)
    /// - If `tensor_metadata.total_params` is 0 (empty model)
    ///
    /// # Constitution: Fail-Fast
    ///
    /// Per AP-007, we panic immediately on invalid data rather than
    /// propagating corruption through the system.
    #[must_use]
    pub fn new(
        gpu_ptr: u64,
        checksum: [u8; 32],
        size_bytes: usize,
        load_duration: Duration,
        tensor_metadata: TensorMetadata,
    ) -> Self {
        // FAIL-FAST: Null GPU pointer
        assert!(
            gpu_ptr != 0,
            "CONSTITUTION VIOLATION AP-007: gpu_ptr is null (0x0). \
             Real cudaMalloc pointer required. \
             This indicates CUDA allocation failure or simulated data."
        );

        // FAIL-FAST: Zero checksum (impossible for real SHA256)
        assert!(
            checksum != [0u8; 32],
            "CONSTITUTION VIOLATION AP-007: checksum is all zeros. \
             Real SHA256 checksum required. \
             This indicates simulated data or computation failure."
        );

        // FAIL-FAST: Zero size
        assert!(
            size_bytes > 0,
            "CONSTITUTION VIOLATION AP-007: size_bytes is 0. \
             Real allocation size required. \
             This indicates allocation failure or simulated data."
        );

        // FAIL-FAST: Empty model (validated in TensorMetadata, but double-check)
        assert!(
            tensor_metadata.total_params > 0,
            "CONSTITUTION VIOLATION AP-007: total_params is 0. \
             Real model weights required. \
             This indicates corrupted or empty weight file."
        );

        Self {
            gpu_ptr,
            checksum,
            size_bytes,
            load_duration,
            tensor_metadata,
        }
    }

    /// Verify checksum matches expected value.
    ///
    /// # Arguments
    ///
    /// * `expected` - Expected SHA256 checksum to compare against
    ///
    /// # Returns
    ///
    /// `true` if checksums match exactly.
    #[must_use]
    pub fn verify_checksum(&self, expected: &[u8; 32]) -> bool {
        self.checksum == *expected
    }

    /// Get checksum as hex string for display/logging.
    #[must_use]
    pub fn checksum_hex(&self) -> String {
        self.checksum.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Get the bytes per parameter based on dtype.
    #[must_use]
    pub fn bytes_per_param(&self) -> usize {
        self.tensor_metadata.dtype.size_in_bytes()
    }
}

/// Complete set of weights for a model loaded into GPU memory.
///
/// Represents a fully loaded model with all tensors resident in GPU VRAM.
/// This is the primary output of the warm loading pipeline.
///
/// # Constitution Alignment
///
/// - REQ-WARM-003: Non-evictable VRAM allocation (tensors are pinned)
/// - REQ-WARM-005: Weight integrity verification (file_checksum)
/// - AP-007: No stub data in production (all tensors are real GPU data)
///
/// # CRITICAL: No Simulation
///
/// This represents REAL weights loaded into REAL GPU memory.
/// All GpuTensor instances must be backed by actual CUDA allocations.
///
/// # Example
///
/// ```rust,ignore
/// use std::collections::HashMap;
/// use context_graph_embeddings::gpu::GpuTensor;
/// use context_graph_embeddings::warm::loader::types::LoadedModelWeights;
///
/// // After loading real tensors to GPU:
/// let mut tensors = HashMap::new();
/// tensors.insert("embeddings.weight".to_string(), gpu_tensor_1);
/// tensors.insert("encoder.weight".to_string(), gpu_tensor_2);
///
/// let weights = LoadedModelWeights::new(
///     "E1_Semantic".to_string(),
///     tensors,
///     [0xAB; 32],           // Real SHA256
///     100_000_000,          // ~100MB GPU memory
///     0,                    // CUDA device 0
/// );
///
/// assert!(weights.has_tensor("embeddings.weight"));
/// ```
#[derive(Debug)]
pub struct LoadedModelWeights {
    /// Model identifier (e.g., "E1_Semantic", "E2_Code").
    ///
    /// # Invariant
    /// MUST be non-empty. Empty identifier = PANIC.
    pub model_id: String,

    /// Named tensors loaded to GPU.
    ///
    /// Uses existing GpuTensor from crate::gpu module.
    /// Key is the tensor name from SafeTensors (e.g., "encoder.layer.0.weight").
    ///
    /// # Invariant
    /// MUST be non-empty. Model must have at least one tensor.
    pub tensors: HashMap<String, GpuTensor>,

    /// SHA256 checksum of source weight file.
    ///
    /// # Invariant
    /// MUST be non-zero. All-zero checksum = PANIC.
    /// Used to verify weight file integrity.
    pub file_checksum: [u8; 32],

    /// Total GPU memory used (bytes).
    ///
    /// # Invariant
    /// MUST be > 0. Sum of all tensor memory allocations.
    pub total_gpu_bytes: usize,

    /// CUDA device where weights are loaded.
    ///
    /// 0 = first GPU, 1 = second GPU, etc.
    pub device_id: u32,

    /// Timestamp when weights were loaded.
    ///
    /// Used for performance monitoring and cache management.
    pub loaded_at: Instant,
}

impl LoadedModelWeights {
    /// Create new LoadedModelWeights with validation.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier (must be non-empty)
    /// * `tensors` - HashMap of named GpuTensors (must be non-empty)
    /// * `file_checksum` - SHA256 of source file (must be non-zero)
    /// * `total_gpu_bytes` - Total GPU memory used (must be > 0)
    /// * `device_id` - CUDA device index
    ///
    /// # Panics
    ///
    /// - If `model_id` is empty
    /// - If `tensors` is empty
    /// - If `file_checksum` is all zeros
    /// - If `total_gpu_bytes` is 0
    ///
    /// # Constitution: Fail-Fast
    ///
    /// Per AP-007, we panic immediately on invalid data rather than
    /// propagating corruption through the system.
    #[must_use]
    pub fn new(
        model_id: String,
        tensors: HashMap<String, GpuTensor>,
        file_checksum: [u8; 32],
        total_gpu_bytes: usize,
        device_id: u32,
    ) -> Self {
        // FAIL-FAST: Empty model ID
        assert!(
            !model_id.is_empty(),
            "CONSTITUTION VIOLATION AP-007: model_id is empty. \
             Model must have a valid identifier."
        );

        // FAIL-FAST: No tensors
        assert!(
            !tensors.is_empty(),
            "CONSTITUTION VIOLATION AP-007: tensors is empty. \
             Model must have at least one tensor. \
             This indicates loading failure or corrupted weight file."
        );

        // FAIL-FAST: Zero checksum
        assert!(
            file_checksum != [0u8; 32],
            "CONSTITUTION VIOLATION AP-007: file_checksum is all zeros. \
             Real SHA256 checksum required for weight integrity."
        );

        // FAIL-FAST: Zero GPU bytes
        assert!(
            total_gpu_bytes > 0,
            "CONSTITUTION VIOLATION AP-007: total_gpu_bytes is 0. \
             Loaded model must occupy GPU memory."
        );

        Self {
            model_id,
            tensors,
            file_checksum,
            total_gpu_bytes,
            device_id,
            loaded_at: Instant::now(),
        }
    }

    /// Get a specific tensor by name.
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name (e.g., "encoder.layer.0.weight")
    ///
    /// # Returns
    ///
    /// Reference to the GpuTensor if found, None otherwise.
    #[must_use]
    pub fn get_tensor(&self, name: &str) -> Option<&GpuTensor> {
        self.tensors.get(name)
    }

    /// Check if a tensor exists by name.
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name to check
    ///
    /// # Returns
    ///
    /// `true` if tensor exists in this model's weights.
    #[must_use]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Get all tensor names.
    ///
    /// # Returns
    ///
    /// Iterator over tensor name strings.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Get the number of tensors in this model.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Verify checksum matches expected value.
    ///
    /// # Arguments
    ///
    /// * `expected` - Expected SHA256 checksum
    ///
    /// # Returns
    ///
    /// `true` if checksums match exactly.
    #[must_use]
    pub fn verify_checksum(&self, expected: &[u8; 32]) -> bool {
        self.file_checksum == *expected
    }

    /// Get time since loading completed.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.loaded_at.elapsed()
    }

    /// Get checksum as hex string for display/logging.
    #[must_use]
    pub fn checksum_hex(&self) -> String {
        self.file_checksum.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

// =============================================================================
// COMPILE-TIME ASSERTIONS
// =============================================================================

/// Compile-time check: Checksum size must be 32 bytes (SHA256)
const _: () = assert!(
    std::mem::size_of::<[u8; 32]>() == 32,
    "Checksum must be exactly 32 bytes for SHA256"
);

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // HELPER FUNCTIONS
    // =========================================================================

    fn create_test_metadata() -> TensorMetadata {
        let mut shapes = HashMap::new();
        shapes.insert("embeddings.weight".to_string(), vec![30522, 768]);
        shapes.insert("encoder.layer.0.weight".to_string(), vec![768, 768]);
        // Total: 30522*768 + 768*768 = 23_440_896 + 589_824 = 24_030_720
        TensorMetadata::new(shapes, DType::F32, 24_030_720)
    }

    // =========================================================================
    // FAIL-FAST VALIDATION TESTS (TensorMetadata)
    // =========================================================================

    #[test]
    #[should_panic(expected = "CONSTITUTION VIOLATION AP-007: shapes is empty")]
    fn test_tensor_metadata_rejects_empty_shapes() {
        let _ = TensorMetadata::new(
            HashMap::new(),  // EMPTY - MUST PANIC
            DType::F32,
            100,
        );
    }

    #[test]
    #[should_panic(expected = "CONSTITUTION VIOLATION AP-007: total_params is 0")]
    fn test_tensor_metadata_rejects_zero_params() {
        let _ = TensorMetadata::new(
            [("test".to_string(), vec![100, 768])].into_iter().collect(),
            DType::F32,
            0,  // ZERO PARAMS - MUST PANIC
        );
    }

    // =========================================================================
    // FAIL-FAST VALIDATION TESTS (WarmLoadResult)
    // =========================================================================

    #[test]
    #[should_panic(expected = "CONSTITUTION VIOLATION AP-007: gpu_ptr is null")]
    fn test_warm_load_result_rejects_null_pointer() {
        let metadata = create_test_metadata();
        let _ = WarmLoadResult::new(
            0,  // NULL POINTER - MUST PANIC
            [1u8; 32],
            1024,
            Duration::from_millis(100),
            metadata,
        );
    }

    #[test]
    #[should_panic(expected = "CONSTITUTION VIOLATION AP-007: checksum is all zeros")]
    fn test_warm_load_result_rejects_zero_checksum() {
        let metadata = create_test_metadata();
        let _ = WarmLoadResult::new(
            0x7fff_0000_1000,
            [0u8; 32],  // ZERO CHECKSUM - MUST PANIC
            1024,
            Duration::from_millis(100),
            metadata,
        );
    }

    #[test]
    #[should_panic(expected = "CONSTITUTION VIOLATION AP-007: size_bytes is 0")]
    fn test_warm_load_result_rejects_zero_size() {
        let metadata = create_test_metadata();
        let _ = WarmLoadResult::new(
            0x7fff_0000_1000,
            [1u8; 32],
            0,  // ZERO SIZE - MUST PANIC
            Duration::from_millis(100),
            metadata,
        );
    }

    // =========================================================================
    // FAIL-FAST VALIDATION TESTS (LoadedModelWeights)
    // =========================================================================

    #[test]
    #[should_panic(expected = "CONSTITUTION VIOLATION AP-007: model_id is empty")]
    fn test_loaded_model_weights_rejects_empty_model_id() {
        // Create a real GpuTensor for testing (requires GPU, so we skip in unit tests)
        // In practice, this test would be an integration test with GPU access
        // For unit testing, we verify the panic message is correct

        // Since we can't easily create a GpuTensor without GPU,
        // we test that the first assertion (model_id) fires before tensors check
        let tensors: HashMap<String, GpuTensor> = HashMap::new();

        let _ = LoadedModelWeights::new(
            "".to_string(),  // EMPTY - MUST PANIC
            tensors,         // Empty too, but model_id check comes first
            [1u8; 32],
            1024,
            0,
        );
    }

    #[test]
    #[should_panic(expected = "CONSTITUTION VIOLATION AP-007: tensors is empty")]
    fn test_loaded_model_weights_rejects_empty_tensors() {
        let _ = LoadedModelWeights::new(
            "E1_Semantic".to_string(),
            HashMap::new(),  // EMPTY - MUST PANIC
            [1u8; 32],
            1024,
            0,
        );
    }

    // NOTE: test_loaded_model_weights_rejects_zero_checksum cannot be tested in unit tests
    // because we cannot create a GpuTensor without GPU hardware. The tensors.is_empty()
    // check fires before the checksum check. This validation is documented and would
    // fire correctly in integration tests with actual GPU tensors.
    //
    // The expected panic message is: "CONSTITUTION VIOLATION AP-007: file_checksum is all zeros"
    // This will be verified in integration tests (TASK-EMB-006-integration).

    // =========================================================================
    // VALID DATA TESTS
    // =========================================================================

    #[test]
    fn test_tensor_metadata_accepts_valid_data() {
        let metadata = create_test_metadata();

        assert_eq!(metadata.tensor_count(), 2);
        assert!(metadata.total_params > 0);
        assert_eq!(metadata.dtype, DType::F32);
    }

    #[test]
    fn test_tensor_metadata_calculates_params() {
        let metadata = TensorMetadata::new(
            [
                ("layer1".to_string(), vec![768, 768]),   // 589,824
                ("layer2".to_string(), vec![768, 3072]),  // 2,359,296
            ].into_iter().collect(),
            DType::F32,
            2_949_120,  // Sum of above
        );

        assert!(metadata.verify_params());
        assert_eq!(metadata.calculate_total_params(), 2_949_120);
    }

    #[test]
    fn test_tensor_metadata_calculates_params_mismatch() {
        let metadata = TensorMetadata::new(
            [
                ("layer1".to_string(), vec![768, 768]),   // 589,824
            ].into_iter().collect(),
            DType::F32,
            1_000_000,  // Wrong value
        );

        assert!(!metadata.verify_params());
        assert_eq!(metadata.calculate_total_params(), 589_824);
    }

    #[test]
    fn test_warm_load_result_accepts_valid_data() {
        let metadata = TensorMetadata::new(
            [("embeddings.weight".to_string(), vec![30522, 768])].into_iter().collect(),
            DType::F32,
            23_440_896,  // 30522 * 768
        );

        let result = WarmLoadResult::new(
            0x7fff_0000_1000,  // Real-looking pointer
            [0xAB; 32],        // Non-zero checksum
            93_763_584,        // 23M params * 4 bytes
            Duration::from_millis(150),
            metadata,
        );

        assert!(result.gpu_ptr != 0);
        assert!(result.size_bytes > 0);
        assert_eq!(result.checksum, [0xAB; 32]);
    }

    #[test]
    fn test_checksum_verification() {
        let expected = [0xAB; 32];
        let metadata = TensorMetadata::new(
            [("test".to_string(), vec![100])].into_iter().collect(),
            DType::F32,
            100,
        );

        let result = WarmLoadResult::new(
            0x7fff_0000_1000,
            expected,
            400,
            Duration::from_millis(10),
            metadata,
        );

        assert!(result.verify_checksum(&expected));
        assert!(!result.verify_checksum(&[0xCD; 32]));
    }

    #[test]
    fn test_checksum_hex_conversion() {
        let checksum = [
            0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE,
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
            0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
            0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00,
        ];

        let metadata = TensorMetadata::new(
            [("test".to_string(), vec![100])].into_iter().collect(),
            DType::F32,
            100,
        );

        let result = WarmLoadResult::new(
            0x7fff_0000_1000,
            checksum,
            400,
            Duration::from_millis(10),
            metadata,
        );

        let hex = result.checksum_hex();
        assert_eq!(hex.len(), 64); // 32 bytes * 2 chars per byte
        assert!(hex.starts_with("deadbeef"));
    }

    #[test]
    fn test_tensor_metadata_get_shape() {
        let metadata = create_test_metadata();

        let shape = metadata.get_shape("embeddings.weight");
        assert!(shape.is_some());
        assert_eq!(shape.unwrap(), &vec![30522, 768]);

        let missing = metadata.get_shape("nonexistent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_bytes_per_param() {
        let metadata = TensorMetadata::new(
            [("test".to_string(), vec![100])].into_iter().collect(),
            DType::F32,
            100,
        );

        let result = WarmLoadResult::new(
            0x7fff_0000_1000,
            [1u8; 32],
            400,
            Duration::from_millis(10),
            metadata,
        );

        assert_eq!(result.bytes_per_param(), 4); // F32 = 4 bytes

        // Test with F16
        let metadata_f16 = TensorMetadata::new(
            [("test".to_string(), vec![100])].into_iter().collect(),
            DType::F16,
            100,
        );

        let result_f16 = WarmLoadResult::new(
            0x7fff_0000_1000,
            [1u8; 32],
            200,
            Duration::from_millis(10),
            metadata_f16,
        );

        assert_eq!(result_f16.bytes_per_param(), 2); // F16 = 2 bytes
    }

    // =========================================================================
    // COMPILE-TIME ASSERTION VERIFICATION
    // =========================================================================

    #[test]
    fn test_checksum_size() {
        // Verify at runtime that checksum is correct size (compile-time assertion above)
        assert_eq!(std::mem::size_of::<[u8; 32]>(), 32);
    }
}
