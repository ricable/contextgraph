//! Model loading operations for the warm model loader.
//!
//! Contains the core loading, allocation, and validation logic for individual models.

use std::time::Instant;

use crate::warm::config::WarmConfig;
use crate::warm::error::{WarmError, WarmResult};
use crate::warm::handle::ModelHandle;
use crate::warm::memory_pool::WarmMemoryPools;
use crate::warm::registry::SharedWarmRegistry;
use crate::warm::state::WarmModelState;
use crate::warm::validation::{TestInferenceConfig, WarmValidator};

use super::helpers::format_bytes;

/// Load a single model into VRAM.
pub fn load_single_model(
    model_id: &str,
    config: &WarmConfig,
    registry: &SharedWarmRegistry,
    memory_pools: &mut WarmMemoryPools,
    validator: &WarmValidator,
) -> WarmResult<()> {
    tracing::info!("Loading model: {}", model_id);

    let load_start = Instant::now();

    // Get model metadata from registry
    let (expected_bytes, expected_dimension) = {
        let reg = registry
            .read()
            .map_err(|_| WarmError::RegistryLockPoisoned)?;
        let entry = reg
            .get_entry(model_id)
            .ok_or_else(|| WarmError::ModelNotRegistered {
                model_id: model_id.to_string(),
            })?;
        (entry.expected_bytes, entry.expected_dimension)
    };

    // Transition: Pending -> Loading
    {
        let mut reg = registry
            .write()
            .map_err(|_| WarmError::RegistryLockPoisoned)?;
        reg.start_loading(model_id)?;
    }

    // Allocate VRAM in the model pool
    let vram_ptr = allocate_model_vram(model_id, expected_bytes, memory_pools)?;

    // Update progress
    {
        let mut reg = registry
            .write()
            .map_err(|_| WarmError::RegistryLockPoisoned)?;
        reg.update_progress(model_id, 50, expected_bytes / 2)?;
    }

    // Simulate model weight loading
    let checksum = simulate_weight_loading(model_id, expected_bytes)?;

    // Update progress to 100%
    {
        let mut reg = registry
            .write()
            .map_err(|_| WarmError::RegistryLockPoisoned)?;
        reg.update_progress(model_id, 100, expected_bytes)?;
    }

    // Transition: Loading -> Validating
    {
        let mut reg = registry
            .write()
            .map_err(|_| WarmError::RegistryLockPoisoned)?;
        reg.mark_validating(model_id)?;
    }

    // Run validation
    validate_model(model_id, expected_dimension, config, validator)?;

    // Create model handle
    let handle = ModelHandle::new(vram_ptr, expected_bytes, config.cuda_device_id, checksum);

    // Transition: Validating -> Warm
    {
        let mut reg = registry
            .write()
            .map_err(|_| WarmError::RegistryLockPoisoned)?;
        reg.mark_warm(model_id, handle)?;
    }

    let load_duration = load_start.elapsed();
    tracing::info!(
        "Model {} loaded successfully in {:?} ({} VRAM)",
        model_id,
        load_duration,
        format_bytes(expected_bytes)
    );

    Ok(())
}

/// Allocate VRAM for a model from the model pool.
pub fn allocate_model_vram(
    model_id: &str,
    size_bytes: usize,
    memory_pools: &mut WarmMemoryPools,
) -> WarmResult<u64> {
    // Check if we have enough space in the model pool
    if memory_pools.available_model_bytes() < size_bytes {
        return Err(WarmError::VramAllocationFailed {
            requested_bytes: size_bytes,
            available_bytes: memory_pools.available_model_bytes(),
            error: format!(
                "Model pool exhausted: {} bytes requested, {} bytes available",
                size_bytes,
                memory_pools.available_model_bytes()
            ),
        });
    }

    // Generate a simulated VRAM pointer
    // In a real implementation, this would come from cudaMalloc
    let base_ptr = 0x7f80_0000_0000u64;
    let offset = memory_pools.list_model_allocations().len() as u64 * 0x1_0000_0000;
    let vram_ptr = base_ptr + offset;

    // Record allocation in memory pool
    memory_pools.allocate_model(model_id, size_bytes, vram_ptr)?;

    tracing::debug!(
        "Allocated {} for {} at 0x{:016x}",
        format_bytes(size_bytes),
        model_id,
        vram_ptr
    );

    Ok(vram_ptr)
}

/// Simulate loading model weights.
///
/// In a real implementation, this would:
/// 1. Open SafeTensors file
/// 2. Read tensors
/// 3. Transfer to GPU
/// 4. Compute SHA256 checksum
pub fn simulate_weight_loading(model_id: &str, _size_bytes: usize) -> WarmResult<u64> {
    // Generate a deterministic checksum based on model ID
    let mut checksum = 0u64;
    for (i, byte) in model_id.bytes().enumerate() {
        checksum ^= (byte as u64) << ((i % 8) * 8);
    }
    checksum ^= 0xDEAD_BEEF_CAFE_BABEu64;

    tracing::debug!(
        "Simulated weight loading for {} (checksum: 0x{:016x})",
        model_id,
        checksum
    );

    Ok(checksum)
}

/// Validate a model after loading.
pub fn validate_model(
    model_id: &str,
    expected_dimension: usize,
    config: &WarmConfig,
    validator: &WarmValidator,
) -> WarmResult<()> {
    if !config.enable_test_inference {
        tracing::info!("Skipping validation for {} (disabled in config)", model_id);
        return Ok(());
    }

    tracing::debug!("Validating model {}", model_id);

    // Create test inference config (for future use with actual inference)
    let _test_config = TestInferenceConfig::for_embedding_model(model_id, expected_dimension);

    // Simulate test inference output
    let output: Vec<f32> = (0..expected_dimension)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();

    // Validate dimensions
    validator.validate_dimensions(model_id, expected_dimension, output.len())?;

    // Validate no NaN/Inf
    validator.validate_weights_finite_for_model(model_id, &output)?;

    tracing::debug!("Model {} validation passed", model_id);
    Ok(())
}

/// Mark a model as failed in the registry.
pub fn mark_model_failed(
    model_id: &str,
    error: &WarmError,
    registry: &SharedWarmRegistry,
) -> WarmResult<()> {
    let mut reg = registry
        .write()
        .map_err(|_| WarmError::RegistryLockPoisoned)?;

    // Only mark failed if in Loading or Validating state
    let state = reg.get_state(model_id);
    if matches!(
        state,
        Some(WarmModelState::Loading { .. }) | Some(WarmModelState::Validating)
    ) {
        reg.mark_failed(model_id, error.exit_code() as u16, error.to_string())?;
    }

    Ok(())
}

/// Verify that all models are in Warm state.
pub fn verify_all_warm(
    loading_order: &[String],
    registry: &SharedWarmRegistry,
    total_model_count: usize,
) -> WarmResult<()> {
    let reg = registry
        .read()
        .map_err(|_| WarmError::RegistryLockPoisoned)?;

    if !reg.all_warm() {
        // Find the first non-warm model for error reporting
        for model_id in loading_order {
            let state = reg.get_state(model_id);
            match state {
                Some(WarmModelState::Warm) => continue,
                Some(WarmModelState::Failed {
                    error_code: _,
                    error_message,
                }) => {
                    return Err(WarmError::ModelLoadFailed {
                        model_id: model_id.clone(),
                        reason: error_message,
                        bytes_read: 0,
                        file_size: 0,
                    });
                }
                other => {
                    return Err(WarmError::ModelLoadFailed {
                        model_id: model_id.clone(),
                        reason: format!("Model in unexpected state: {:?}", other),
                        bytes_read: 0,
                        file_size: 0,
                    });
                }
            }
        }

        // Generic error if no specific model found
        return Err(WarmError::ModelValidationFailed {
            model_id: "unknown".to_string(),
            reason: "Not all models are warm after loading".to_string(),
            expected_output: Some(format!("{} models warm", total_model_count)),
            actual_output: Some(format!("{} models warm", reg.warm_count())),
        });
    }

    tracing::info!("All {} models verified warm", reg.warm_count());
    Ok(())
}
