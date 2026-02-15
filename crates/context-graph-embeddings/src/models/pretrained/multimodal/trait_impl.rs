//! EmbeddingModel trait implementation for MultimodalModel.

use std::sync::atomic::Ordering;

use async_trait::async_trait;

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::traits::EmbeddingModel;
use crate::types::{InputType, ModelEmbedding, ModelId, ModelInput};

use super::forward_text::text_forward;
use super::forward_vision::vision_forward;
use crate::models::pretrained::shared::ModelState;
use super::model::MultimodalModel;

#[async_trait]
impl EmbeddingModel for MultimodalModel {
    fn model_id(&self) -> ModelId {
        ModelId::Multimodal
    }

    fn supported_input_types(&self) -> &[InputType] {
        &[InputType::Text, InputType::Image]
    }

    fn is_initialized(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }

    async fn load(&self) -> EmbeddingResult<()> {
        MultimodalModel::load(self).await
    }

    async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding> {
        if !self.is_initialized() {
            return Err(EmbeddingError::NotInitialized {
                model_id: self.model_id(),
            });
        }

        self.validate_input(input)?;

        let start = std::time::Instant::now();

        // Get loaded weights and tokenizer
        let state = self
            .model_state
            .read()
            .map_err(|e| EmbeddingError::InternalError {
                message: format!("MultimodalModel failed to acquire read lock: {}", e),
            })?;

        let (weights, tokenizer) = match &*state {
            ModelState::Loaded { weights, tokenizer } => (weights, tokenizer),
            _ => {
                return Err(EmbeddingError::NotInitialized {
                    model_id: self.model_id(),
                });
            }
        };

        // Run GPU-accelerated forward pass based on input type
        let vector = match input {
            ModelInput::Text {
                content,
                instruction,
            } => {
                // Combine instruction and content if provided
                let full_text = if let Some(inst) = instruction {
                    format!("{} {}", inst, content)
                } else {
                    content.clone()
                };
                text_forward(&full_text, weights, tokenizer)?
            }
            ModelInput::Image { bytes, format } => {
                // Preprocess image
                let tensor = self.image_processor.preprocess(bytes, *format)?;
                vision_forward(&tensor, weights)?
            }
            _ => {
                return Err(EmbeddingError::UnsupportedModality {
                    model_id: self.model_id(),
                    input_type: InputType::from(input),
                });
            }
        };

        let latency_us = start.elapsed().as_micros() as u64;
        Ok(ModelEmbedding::new(ModelId::Multimodal, vector, latency_us))
    }
}
