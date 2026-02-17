//! Construction and trait implementation tests for MultimodalModel.

use crate::error::EmbeddingError;
use crate::traits::{EmbeddingModel, SingleModelConfig};
use crate::types::{InputType, ModelId, ModelInput};
use serial_test::serial;

use super::super::{
    MultimodalModel, CLIP_IMAGE_SIZE, CLIP_MEAN, CLIP_STD, MULTIMODAL_DIMENSION,
    MULTIMODAL_LATENCY_BUDGET_MS, MULTIMODAL_MAX_TOKENS, MULTIMODAL_MODEL_NAME,
};
use super::{create_and_load_model, create_test_model, workspace_root};

// ==================== Construction Tests ====================

#[test]
fn test_new_creates_unloaded_model() {
    let model = create_test_model();
    assert!(!model.is_initialized());
}

#[test]
fn test_new_with_zero_batch_size_fails() {
    let config = SingleModelConfig {
        max_batch_size: 0,
        ..Default::default()
    };
    let model_path = workspace_root().join("models/multimodal");
    let result = MultimodalModel::new(&model_path, config);
    assert!(matches!(result, Err(EmbeddingError::ConfigError { .. })));
}

// ==================== Trait Implementation Tests ====================

#[test]
fn test_model_id() {
    let model = create_test_model();
    assert_eq!(model.model_id(), ModelId::Multimodal);
}

#[test]
fn test_native_dimension() {
    let model = create_test_model();
    assert_eq!(model.dimension(), 768);
}

#[test]
fn test_projected_dimension_equals_native() {
    let model = create_test_model();
    assert_eq!(model.projected_dimension(), 768);
}

#[test]
fn test_max_tokens() {
    let model = create_test_model();
    // EMB-1 FIX: e5-base-v2 uses BERT tokenizer (512 tokens), not CLIP (77)
    assert_eq!(model.max_tokens(), 512);
}

#[test]
fn test_latency_budget_ms() {
    let model = create_test_model();
    assert_eq!(model.latency_budget_ms(), 15);
}

#[test]
fn test_is_pretrained() {
    let model = create_test_model();
    assert!(model.is_pretrained());
}

#[test]
fn test_supported_input_types() {
    let model = create_test_model();
    let types = model.supported_input_types();
    assert!(types.contains(&InputType::Text));
    assert!(types.contains(&InputType::Image));
    assert!(!types.contains(&InputType::Code));
    assert!(!types.contains(&InputType::Audio));
}

// ==================== State Transition Tests ====================

#[tokio::test]
async fn test_load_sets_initialized() {
    let model = create_test_model();
    assert!(!model.is_initialized());
    model.load().await.expect("Load should succeed");
    assert!(model.is_initialized());
}

#[tokio::test]
async fn test_unload_clears_initialized() {
    let model = create_and_load_model().await;
    assert!(model.is_initialized());
    model.unload().await.expect("Unload should succeed");
    assert!(!model.is_initialized());
}

#[tokio::test]
async fn test_unload_when_not_loaded_fails() {
    let model = create_test_model();
    let result = model.unload().await;
    assert!(matches!(result, Err(EmbeddingError::NotInitialized { .. })));
}

// Serial: Multiple load/unload cycles require exclusive VRAM access
#[tokio::test]
#[serial]
async fn test_state_transitions_full_cycle() {
    let model = create_test_model();
    assert!(!model.is_initialized());
    model.load().await.unwrap();
    assert!(model.is_initialized());
    model.unload().await.unwrap();
    assert!(!model.is_initialized());
    model.load().await.unwrap();
    assert!(model.is_initialized());
}

// ==================== Constants Tests ====================

#[test]
fn test_constants_are_correct() {
    assert_eq!(MULTIMODAL_DIMENSION, 768);
    assert_eq!(MULTIMODAL_MAX_TOKENS, 77);
    assert_eq!(MULTIMODAL_LATENCY_BUDGET_MS, 15);
    assert_eq!(MULTIMODAL_MODEL_NAME, "openai/clip-vit-large-patch14");
    assert_eq!(CLIP_IMAGE_SIZE, 224);
}

#[test]
fn test_clip_normalization_constants() {
    assert_eq!(CLIP_MEAN, [0.48145466, 0.4578275, 0.40821073]);
    assert_eq!(CLIP_STD, [0.26862954, 0.261_302_6, 0.275_777_1]);
}

#[test]
fn test_model_id_dimension_matches_constant() {
    assert_eq!(ModelId::Multimodal.dimension(), MULTIMODAL_DIMENSION);
    assert_eq!(
        ModelId::Multimodal.projected_dimension(),
        MULTIMODAL_DIMENSION
    );
}

#[tokio::test]
async fn test_embed_before_load_fails() {
    let model = create_test_model();
    let input = ModelInput::text("a photo of a cat").expect("Failed to create input");
    let result = model.embed(&input).await;
    assert!(matches!(result, Err(EmbeddingError::NotInitialized { .. })));
}
