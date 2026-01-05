//! Core loader functionality tests.

use crate::warm::config::WarmConfig;
use crate::warm::cuda_alloc::{REQUIRED_COMPUTE_MAJOR, REQUIRED_COMPUTE_MINOR};
use crate::warm::error::WarmError;
use crate::warm::handle::ModelHandle;
use crate::warm::registry::{WarmModelRegistry, EMBEDDING_MODEL_IDS, TOTAL_MODEL_COUNT};
use crate::warm::state::WarmModelState;

use super::super::constants::GB;
use super::super::engine::WarmLoader;

/// Create a test config that doesn't require real files.
#[allow(clippy::field_reassign_with_default)]
fn test_config() -> WarmConfig {
    let mut config = WarmConfig::default();
    config.enable_test_inference = true;
    config
}

// ============================================================================
// Test 1: Loader Construction
// ============================================================================

#[test]
fn test_loader_construction() {
    let config = test_config();
    let loader = WarmLoader::new(config).expect("Failed to create loader");

    // Verify registry is populated
    let registry = loader.registry().read().unwrap();
    assert_eq!(registry.model_count(), TOTAL_MODEL_COUNT);

    // Verify all models are in Pending state initially
    for model_id in EMBEDDING_MODEL_IDS {
        let state = registry.get_state(model_id);
        assert!(matches!(state, Some(WarmModelState::Pending)));
    }
}

// ============================================================================
// Test 2: Loading Summary Initial State
// ============================================================================

#[test]
fn test_loading_summary_initial() {
    let config = test_config();
    let loader = WarmLoader::new(config).expect("Failed to create loader");

    let summary = loader.loading_summary();

    assert_eq!(summary.total_models, TOTAL_MODEL_COUNT);
    assert_eq!(summary.models_warm, 0);
    assert_eq!(summary.models_failed, 0);
    assert_eq!(summary.models_loading, 0);
    assert_eq!(summary.total_vram_allocated, 0);
    assert!(summary.loading_duration.is_none());
    assert!(!summary.all_warm());
    assert!(!summary.any_failed());
}

// ============================================================================
// Test 3: Loading Order Uses Registry
// ============================================================================

#[test]
fn test_loading_order_uses_registry() {
    let config = test_config();
    let loader = WarmLoader::new(config).expect("Failed to create loader");

    // Verify loading order is non-empty
    let registry = loader.registry().read().unwrap();
    let loading_order = registry.loading_order();
    assert!(!loading_order.is_empty());
    assert_eq!(loading_order.len(), TOTAL_MODEL_COUNT);

    // Verify E10_Multimodal (largest at 800MB) is first in loading order
    assert_eq!(loading_order[0], "E10_Multimodal");
}

// ============================================================================
// Test 4: Preflight Checks GPU Requirements
// ============================================================================

#[test]
fn test_preflight_checks_gpu_requirements() {
    let config = test_config();
    let mut loader = WarmLoader::new(config).expect("Failed to create loader");

    // In non-CUDA mode, preflight should succeed with simulated GPU
    let result = loader.run_preflight_checks();
    assert!(result.is_ok());

    // Verify GPU info is populated
    let gpu_info = loader.gpu_info();
    assert!(gpu_info.is_some());
    let info = gpu_info.unwrap();
    assert!(info.meets_compute_requirement(REQUIRED_COMPUTE_MAJOR, REQUIRED_COMPUTE_MINOR));
}

// ============================================================================
// Test 5: Fail Fast on Allocation Error
// ============================================================================

#[test]
fn test_fail_fast_on_allocation_error() {
    let mut config = test_config();
    // Set a very small VRAM budget to trigger allocation failure
    config.vram_budget_bytes = 1024; // Only 1KB

    let mut loader = WarmLoader::new(config).expect("Failed to create loader");

    // Try to allocate more than available
    let result = loader.allocate_model_vram("test_model", 1024 * 1024 * 1024);
    assert!(result.is_err());

    match result.unwrap_err() {
        WarmError::VramAllocationFailed {
            requested_bytes,
            available_bytes,
            ..
        } => {
            assert_eq!(requested_bytes, 1024 * 1024 * 1024);
            assert!(available_bytes < requested_bytes);
        }
        _ => panic!("Expected VramAllocationFailed error"),
    }
}

// ============================================================================
// Test 6: Fail Fast on Validation Error
// ============================================================================

#[test]
fn test_fail_fast_on_validation_error() {
    let config = test_config();
    let loader = WarmLoader::new(config).expect("Failed to create loader");

    // Test validation with mismatched dimensions via registry access
    let registry = loader.registry().read().unwrap();
    let validator = crate::warm::validation::WarmValidator::new();

    let result = validator.validate_dimensions("E1_Semantic", 1024, 512);
    assert!(result.is_err());

    drop(registry);

    match result.unwrap_err() {
        WarmError::ModelDimensionMismatch {
            model_id,
            expected,
            actual,
        } => {
            assert_eq!(model_id, "E1_Semantic");
            assert_eq!(expected, 1024);
            assert_eq!(actual, 512);
        }
        _ => panic!("Expected ModelDimensionMismatch error"),
    }
}

// ============================================================================
// Test 7: All Warm Check
// ============================================================================

#[test]
fn test_all_warm_check() {
    let config = test_config();
    let loader = WarmLoader::new(config).expect("Failed to create loader");

    // Initially not all warm
    assert!(!loader.all_warm());

    // Manually transition all models to Warm for testing
    {
        let mut registry = loader.registry().write().unwrap();
        for model_id in EMBEDDING_MODEL_IDS.iter() {
            registry.start_loading(model_id).unwrap();
            registry.mark_validating(model_id).unwrap();
            let handle = ModelHandle::new(0x1000, 1024, 0, 0xDEAD);
            registry.mark_warm(model_id, handle).unwrap();
        }
    }

    // Now all should be warm
    assert!(loader.all_warm());
}

// ============================================================================
// Test 8: Loading Summary After Success
// ============================================================================

#[test]
fn test_loading_summary_after_success() {
    let config = test_config();
    let mut loader = WarmLoader::new(config).expect("Failed to create loader");

    // Transition all models through the state machine
    {
        let mut registry = loader.registry().write().unwrap();
        for model_id in EMBEDDING_MODEL_IDS.iter() {
            registry.start_loading(model_id).unwrap();
            registry.mark_validating(model_id).unwrap();
            let handle = ModelHandle::new(0x1000, 1024, 0, 0xCAFE);
            registry.mark_warm(model_id, handle).unwrap();
        }
    }

    // Allocate some memory to simulate VRAM usage
    loader
        .memory_pools_mut()
        .allocate_model("test", GB, 0x1000)
        .unwrap();

    let summary = loader.loading_summary();

    assert_eq!(summary.total_models, TOTAL_MODEL_COUNT);
    assert_eq!(summary.models_warm, TOTAL_MODEL_COUNT);
    assert_eq!(summary.models_failed, 0);
    assert_eq!(summary.models_loading, 0);
    assert!(summary.total_vram_allocated > 0);
    assert!(summary.all_warm());
    assert!(!summary.any_failed());
    assert!((summary.warm_percentage() - 100.0).abs() < 0.01);
}

// ============================================================================
// Test 9: Register All Models
// ============================================================================

#[test]
fn test_register_all_models() {
    let mut registry = WarmModelRegistry::new();
    WarmLoader::register_all_models(&mut registry).expect("Failed to register models");

    assert_eq!(registry.model_count(), TOTAL_MODEL_COUNT);

    // Verify all embedding models are registered
    for model_id in EMBEDDING_MODEL_IDS {
        assert!(
            registry.get_state(model_id).is_some(),
            "Missing model {}",
            model_id
        );
    }
}

// ============================================================================
// Test 10: Simulate Weight Loading
// ============================================================================

#[test]
fn test_simulate_weight_loading() {
    let config = test_config();
    let loader = WarmLoader::new(config).expect("Failed to create loader");

    // Different models should produce different checksums
    let checksum1 = loader.simulate_weight_loading("E1_Semantic", 1024).unwrap();
    let checksum2 = loader
        .simulate_weight_loading("E2_TemporalRecent", 1024)
        .unwrap();

    assert_ne!(checksum1, checksum2);
}

// ============================================================================
// Test 11: Config Accessor
// ============================================================================

#[test]
fn test_config_accessor() {
    let config = test_config();
    let original_budget = config.vram_budget_bytes;

    let loader = WarmLoader::new(config).expect("Failed to create loader");
    assert_eq!(loader.config().vram_budget_bytes, original_budget);
}
