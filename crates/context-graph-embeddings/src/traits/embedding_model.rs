//! Core trait for embedding model implementations.
//!
//! The `EmbeddingModel` trait defines the contract that all 12 embedding models
//! in the fusion pipeline must implement. Each model (E1-E12) has different
//! input requirements, dimensions, and processing characteristics.
//!
//! # Model Compatibility Matrix
//!
//! | Model | Text | Code | Image | Audio |
//! |-------|------|------|-------|-------|
//! | Semantic (E1) | ✓ | ✓* | ✗ | ✗ |
//! | TemporalRecent (E2) | ✓ | ✓ | ✗ | ✗ |
//! | TemporalPeriodic (E3) | ✓ | ✓ | ✗ | ✗ |
//! | TemporalPositional (E4) | ✓ | ✓ | ✗ | ✗ |
//! | Causal (E5) | ✓ | ✓ | ✗ | ✗ |
//! | Sparse (E6) | ✓ | ✓* | ✗ | ✗ |
//! | Code (E7) | ✓* | ✓ | ✗ | ✗ |
//! | Graph (E8) | ✓ | ✓* | ✗ | ✗ |
//! | HDC (E9) | ✓ | ✓ | ✗ | ✗ |
//! | Multimodal (E10) | ✓ | ✗ | ✓ | ✗ |
//! | Entity (E11) | ✓ | ✓* | ✗ | ✗ |
//! | LateInteraction (E12) | ✓ | ✓* | ✗ | ✗ |
//!
//! *Model can process but is not optimized for this type
//!
//! # Thread Safety
//!
//! The trait requires `Send + Sync` bounds to ensure safe usage in
//! multi-threaded async contexts. All implementations must be thread-safe.
//!
//! # Example Implementation
//!
//! ```rust,ignore
//! use context_graph_embeddings::traits::EmbeddingModel;
//! use context_graph_embeddings::types::{ModelId, ModelEmbedding, ModelInput, InputType};
//! use context_graph_embeddings::error::{EmbeddingError, EmbeddingResult};
//! use async_trait::async_trait;
//!
//! struct SemanticModel {
//!     initialized: bool,
//! }
//!
//! #[async_trait]
//! impl EmbeddingModel for SemanticModel {
//!     fn model_id(&self) -> ModelId {
//!         ModelId::Semantic
//!     }
//!
//!     fn supported_input_types(&self) -> &[InputType] {
//!         &[InputType::Text, InputType::Code]
//!     }
//!
//!     async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding> {
//!         // Implementation...
//!         todo!()
//!     }
//!
//!     fn is_initialized(&self) -> bool {
//!         self.initialized
//!     }
//! }
//! ```

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::{InputType, ModelEmbedding, ModelId, ModelInput};
use async_trait::async_trait;

/// Core trait for embedding model implementations.
///
/// All 12 embedding models in the fusion pipeline must implement this trait.
/// The trait provides a unified async interface for:
/// - Querying model capabilities (ID, supported input types, dimensions)
/// - Generating embeddings from multi-modal inputs
/// - Managing model lifecycle (initialization state)
///
/// # Thread Safety
///
/// This trait requires `Send + Sync` bounds, ensuring implementations
/// can be safely shared across async tasks and threads.
///
/// # Error Handling
///
/// All methods that can fail return `EmbeddingResult<T>`. Implementations
/// must return appropriate error variants:
/// - `EmbeddingError::UnsupportedModality` for incompatible input types
/// - `EmbeddingError::NotInitialized` if model not ready
/// - `EmbeddingError::EmptyInput` for empty content
/// - Other variants as appropriate
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_embeddings::traits::EmbeddingModel;
/// use context_graph_embeddings::types::{ModelInput, InputType};
///
/// async fn generate_embedding(model: &dyn EmbeddingModel, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
///     // Check if model supports text
///     if !model.supports_input_type(InputType::Text) {
///         return Err("Model doesn't support text".into());
///     }
///
///     // Generate embedding
///     let input = ModelInput::text(text)?;
///     let embedding = model.embed(&input).await?;
///     Ok(embedding.vector)
/// }
/// ```
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// Returns the unique identifier for this model.
    ///
    /// Each implementation returns one of the 12 `ModelId` variants (E1-E12).
    /// This ID is used for:
    /// - Routing inputs to appropriate models
    /// - Validating embedding dimensions
    /// - Logging and debugging
    ///
    /// # Returns
    /// The `ModelId` variant identifying this model.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = SemanticModel::new();
    /// assert_eq!(model.model_id(), ModelId::Semantic);
    /// ```
    fn model_id(&self) -> ModelId;

    /// Returns the list of input types this model supports.
    ///
    /// Models should only list types they are designed to handle well.
    /// Attempting to embed an unsupported type should return
    /// `EmbeddingError::UnsupportedModality`.
    ///
    /// # Returns
    /// Static slice of supported `InputType` variants.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = MultimodalModel::new();
    /// let types = model.supported_input_types();
    /// assert!(types.contains(&InputType::Text));
    /// assert!(types.contains(&InputType::Image));
    /// ```
    fn supported_input_types(&self) -> &[InputType];

    /// Generate an embedding for the given input.
    ///
    /// This is the core embedding generation method. Implementations must:
    /// 1. Validate the input type is supported
    /// 2. Process the input through the model
    /// 3. Return a properly dimensioned `ModelEmbedding`
    ///
    /// # Arguments
    /// * `input` - The input to embed (text, code, image, or audio)
    ///
    /// # Returns
    /// - `Ok(ModelEmbedding)` with the generated embedding vector
    /// - `Err(EmbeddingError)` on failure
    ///
    /// # Errors
    /// - `EmbeddingError::UnsupportedModality` if input type not supported
    /// - `EmbeddingError::NotInitialized` if model not initialized
    /// - `EmbeddingError::EmptyInput` if input content is empty
    /// - `EmbeddingError::InputTooLong` if input exceeds max tokens
    /// - `EmbeddingError::Timeout` if processing exceeds time budget
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let input = ModelInput::text("Hello, world!")?;
    /// let embedding = model.embed(&input).await?;
    /// assert_eq!(embedding.model_id, ModelId::Semantic);
    /// assert_eq!(embedding.dimension(), 1024);
    /// ```
    async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding>;

    /// Returns whether the model is initialized and ready for inference.
    ///
    /// Models may require initialization (loading weights, warming up GPU)
    /// before they can process inputs. This method allows checking readiness.
    ///
    /// # Returns
    /// - `true` if model is ready for `embed()` calls
    /// - `false` if model needs initialization
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if !model.is_initialized() {
    ///     return Err(EmbeddingError::NotInitialized { model_id: model.model_id() });
    /// }
    /// ```
    fn is_initialized(&self) -> bool;

    // =========================================================================
    // Default implementations
    // =========================================================================

    /// Check if this model supports the given input type.
    ///
    /// This is a convenience method that checks if the input type
    /// is in the list returned by `supported_input_types()`.
    ///
    /// # Arguments
    /// * `input_type` - The input type to check
    ///
    /// # Returns
    /// `true` if the model supports this input type
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if model.supports_input_type(InputType::Image) {
    ///     let embedding = model.embed(&image_input).await?;
    /// }
    /// ```
    fn supports_input_type(&self, input_type: InputType) -> bool {
        self.supported_input_types().contains(&input_type)
    }

    /// Returns the native output dimension for this model.
    ///
    /// This is a convenience method that delegates to `ModelId::dimension()`.
    /// Returns the raw model output size before any projection.
    ///
    /// # Returns
    /// The embedding dimension for this model.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let dim = model.dimension();
    /// assert_eq!(dim, 1024); // For Semantic model
    /// ```
    fn dimension(&self) -> usize {
        self.model_id().dimension()
    }

    /// Returns the projected dimension for FuseMoE input.
    ///
    /// Some models (Sparse, Code, HDC) project their outputs to
    /// different dimensions for the fusion pipeline.
    ///
    /// # Returns
    /// The projected dimension used in concatenation.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let proj_dim = model.projected_dimension();
    /// assert_eq!(proj_dim, 1536); // For Sparse model
    /// ```
    fn projected_dimension(&self) -> usize {
        self.model_id().projected_dimension()
    }

    /// Returns the latency budget in milliseconds for this model.
    ///
    /// Each model has a performance target from constitution.yaml.
    /// Implementations should aim to complete within this budget.
    ///
    /// # Returns
    /// Maximum expected latency in milliseconds.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let budget = model.latency_budget_ms();
    /// assert_eq!(budget, 5); // Semantic: 5ms
    /// ```
    fn latency_budget_ms(&self) -> u32 {
        self.model_id().latency_budget_ms()
    }

    /// Validate that the input is compatible with this model.
    ///
    /// Checks the input type against supported types and returns
    /// an appropriate error if incompatible.
    ///
    /// # Arguments
    /// * `input` - The input to validate
    ///
    /// # Returns
    /// - `Ok(())` if input is compatible
    /// - `Err(EmbeddingError::UnsupportedModality)` if not supported
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// model.validate_input(&input)?;
    /// let embedding = model.embed(&input).await?;
    /// ```
    fn validate_input(&self, input: &ModelInput) -> EmbeddingResult<()> {
        let input_type = InputType::from(input);
        if self.supports_input_type(input_type) {
            Ok(())
        } else {
            Err(EmbeddingError::UnsupportedModality {
                model_id: self.model_id(),
                input_type,
            })
        }
    }

    /// Returns the maximum input token count for this model.
    ///
    /// Convenience method delegating to `ModelId::max_tokens()`.
    ///
    /// # Returns
    /// Maximum token count (varies by model: 77-4096, or MAX for custom)
    fn max_tokens(&self) -> usize {
        self.model_id().max_tokens()
    }

    /// Returns whether this model uses pretrained weights.
    ///
    /// Convenience method delegating to `ModelId::is_pretrained()`.
    ///
    /// # Returns
    /// - `true` for models with HuggingFace weights (8 models)
    /// - `false` for custom implementations (Temporal*, HDC)
    fn is_pretrained(&self) -> bool {
        self.model_id().is_pretrained()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    // =========================================================================
    // Test Model Implementation
    // =========================================================================

    /// Test implementation of EmbeddingModel for testing the trait.
    struct TestModel {
        model_id: ModelId,
        supported_types: Vec<InputType>,
        initialized: AtomicBool,
    }

    impl TestModel {
        fn new(model_id: ModelId, supported_types: Vec<InputType>) -> Self {
            Self {
                model_id,
                supported_types,
                initialized: AtomicBool::new(true),
            }
        }

        fn set_initialized(&self, initialized: bool) {
            self.initialized.store(initialized, Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl EmbeddingModel for TestModel {
        fn model_id(&self) -> ModelId {
            self.model_id
        }

        fn supported_input_types(&self) -> &[InputType] {
            &self.supported_types
        }

        async fn embed(&self, input: &ModelInput) -> EmbeddingResult<ModelEmbedding> {
            // Check initialization
            if !self.is_initialized() {
                return Err(EmbeddingError::NotInitialized {
                    model_id: self.model_id,
                });
            }

            // Validate input type
            self.validate_input(input)?;

            // Generate a deterministic embedding based on content hash
            let hash = input.content_hash();
            let dim = self.dimension();
            let mut vector = Vec::with_capacity(dim);

            // Generate deterministic values from hash
            let mut state = hash;
            for _ in 0..dim {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = ((state >> 33) as f32) / (u32::MAX as f32) - 0.5;
                vector.push(val);
            }

            Ok(ModelEmbedding::new(self.model_id, vector, 100))
        }

        fn is_initialized(&self) -> bool {
            self.initialized.load(Ordering::SeqCst)
        }
    }

    // =========================================================================
    // MODEL ID TESTS (3 tests)
    // =========================================================================

    #[test]
    fn test_model_id_returns_correct_value() {
        let model = TestModel::new(ModelId::Semantic, vec![InputType::Text]);
        assert_eq!(model.model_id(), ModelId::Semantic);

        let model2 = TestModel::new(ModelId::Code, vec![InputType::Code]);
        assert_eq!(model2.model_id(), ModelId::Code);
    }

    #[test]
    fn test_model_id_for_all_12_models() {
        for model_id in ModelId::all() {
            let model = TestModel::new(*model_id, vec![InputType::Text]);
            assert_eq!(model.model_id(), *model_id);
        }
    }

    #[test]
    fn test_model_id_is_consistent_across_calls() {
        let model = TestModel::new(ModelId::Graph, vec![InputType::Text]);
        assert_eq!(model.model_id(), model.model_id());
        assert_eq!(model.model_id(), model.model_id());
    }

    // =========================================================================
    // SUPPORTED INPUT TYPES TESTS (3 tests)
    // =========================================================================

    #[test]
    fn test_supported_input_types_returns_correct_list() {
        let supported = vec![InputType::Text, InputType::Code];
        let model = TestModel::new(ModelId::Semantic, supported.clone());
        assert_eq!(model.supported_input_types(), supported.as_slice());
    }

    #[test]
    fn test_supports_input_type_true_for_supported() {
        let model = TestModel::new(
            ModelId::Multimodal,
            vec![InputType::Text, InputType::Image],
        );
        assert!(model.supports_input_type(InputType::Text));
        assert!(model.supports_input_type(InputType::Image));
    }

    #[test]
    fn test_supports_input_type_false_for_unsupported() {
        let model = TestModel::new(ModelId::Semantic, vec![InputType::Text]);
        assert!(!model.supports_input_type(InputType::Image));
        assert!(!model.supports_input_type(InputType::Audio));
    }

    // =========================================================================
    // EMBED TESTS (5 tests)
    // =========================================================================

    #[tokio::test]
    async fn test_embed_returns_correct_model_id() {
        let model = TestModel::new(ModelId::Semantic, vec![InputType::Text]);
        let input = ModelInput::text("Hello, world!").unwrap();
        let embedding = model.embed(&input).await.unwrap();
        assert_eq!(embedding.model_id, ModelId::Semantic);
    }

    #[tokio::test]
    async fn test_embed_returns_correct_dimension() {
        let model = TestModel::new(ModelId::Semantic, vec![InputType::Text]);
        let input = ModelInput::text("Test content").unwrap();
        let embedding = model.embed(&input).await.unwrap();
        assert_eq!(embedding.dimension(), ModelId::Semantic.dimension());
        assert_eq!(embedding.dimension(), 1024);
    }

    #[tokio::test]
    async fn test_embed_rejects_unsupported_input_type() {
        let model = TestModel::new(ModelId::Semantic, vec![InputType::Text]);
        let input = ModelInput::image(vec![1, 2, 3, 4], crate::types::ImageFormat::Png).unwrap();
        let result = model.embed(&input).await;
        assert!(result.is_err());
        match result {
            Err(EmbeddingError::UnsupportedModality { model_id, input_type }) => {
                assert_eq!(model_id, ModelId::Semantic);
                assert_eq!(input_type, InputType::Image);
            }
            _ => panic!("Expected UnsupportedModality error"),
        }
    }

    #[tokio::test]
    async fn test_embed_rejects_when_not_initialized() {
        let model = TestModel::new(ModelId::Semantic, vec![InputType::Text]);
        model.set_initialized(false);
        let input = ModelInput::text("Test").unwrap();
        let result = model.embed(&input).await;
        assert!(result.is_err());
        match result {
            Err(EmbeddingError::NotInitialized { model_id }) => {
                assert_eq!(model_id, ModelId::Semantic);
            }
            _ => panic!("Expected NotInitialized error"),
        }
    }

    #[tokio::test]
    async fn test_embed_deterministic_for_same_input() {
        let model = TestModel::new(ModelId::Semantic, vec![InputType::Text]);
        let input = ModelInput::text("Deterministic test").unwrap();

        let embedding1 = model.embed(&input).await.unwrap();
        let embedding2 = model.embed(&input).await.unwrap();

        assert_eq!(embedding1.vector, embedding2.vector);
    }

    // =========================================================================
    // INITIALIZATION STATE TESTS (2 tests)
    // =========================================================================

    #[test]
    fn test_is_initialized_returns_true_by_default() {
        let model = TestModel::new(ModelId::Semantic, vec![InputType::Text]);
        assert!(model.is_initialized());
    }

    #[test]
    fn test_is_initialized_reflects_state_change() {
        let model = TestModel::new(ModelId::Semantic, vec![InputType::Text]);
        assert!(model.is_initialized());

        model.set_initialized(false);
        assert!(!model.is_initialized());

        model.set_initialized(true);
        assert!(model.is_initialized());
    }

    // =========================================================================
    // DEFAULT METHOD TESTS (5 tests)
    // =========================================================================

    #[test]
    fn test_dimension_delegates_to_model_id() {
        for model_id in ModelId::all() {
            let model = TestModel::new(*model_id, vec![InputType::Text]);
            assert_eq!(model.dimension(), model_id.dimension());
        }
    }

    #[test]
    fn test_projected_dimension_delegates_to_model_id() {
        let sparse = TestModel::new(ModelId::Sparse, vec![InputType::Text]);
        assert_eq!(sparse.projected_dimension(), 1536);

        let semantic = TestModel::new(ModelId::Semantic, vec![InputType::Text]);
        assert_eq!(semantic.projected_dimension(), 1024);
    }

    #[test]
    fn test_latency_budget_ms_delegates_to_model_id() {
        let semantic = TestModel::new(ModelId::Semantic, vec![InputType::Text]);
        assert_eq!(semantic.latency_budget_ms(), 5);

        let hdc = TestModel::new(ModelId::Hdc, vec![InputType::Text]);
        assert_eq!(hdc.latency_budget_ms(), 1);
    }

    #[test]
    fn test_max_tokens_delegates_to_model_id() {
        let causal = TestModel::new(ModelId::Causal, vec![InputType::Text]);
        assert_eq!(causal.max_tokens(), 4096);

        let multimodal = TestModel::new(ModelId::Multimodal, vec![InputType::Text]);
        assert_eq!(multimodal.max_tokens(), 77);
    }

    #[test]
    fn test_is_pretrained_delegates_to_model_id() {
        let semantic = TestModel::new(ModelId::Semantic, vec![InputType::Text]);
        assert!(semantic.is_pretrained());

        let temporal = TestModel::new(ModelId::TemporalRecent, vec![InputType::Text]);
        assert!(!temporal.is_pretrained());
    }

    // =========================================================================
    // VALIDATE INPUT TESTS (2 tests)
    // =========================================================================

    #[test]
    fn test_validate_input_accepts_supported_type() {
        let model = TestModel::new(
            ModelId::Code,
            vec![InputType::Text, InputType::Code],
        );

        let text_input = ModelInput::text("Hello").unwrap();
        assert!(model.validate_input(&text_input).is_ok());

        let code_input = ModelInput::code("fn main() {}", "rust").unwrap();
        assert!(model.validate_input(&code_input).is_ok());
    }

    #[test]
    fn test_validate_input_rejects_unsupported_type() {
        let model = TestModel::new(ModelId::Semantic, vec![InputType::Text]);

        let image_input = ModelInput::image(vec![1, 2, 3], crate::types::ImageFormat::Png).unwrap();
        let result = model.validate_input(&image_input);

        assert!(result.is_err());
        match result {
            Err(EmbeddingError::UnsupportedModality { model_id, input_type }) => {
                assert_eq!(model_id, ModelId::Semantic);
                assert_eq!(input_type, InputType::Image);
            }
            _ => panic!("Expected UnsupportedModality error"),
        }
    }

    // =========================================================================
    // SEND + SYNC TESTS (2 tests)
    // =========================================================================

    #[test]
    fn test_embedding_model_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<TestModel>();
    }

    #[test]
    fn test_embedding_model_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<TestModel>();
    }

    // =========================================================================
    // OBJECT SAFETY TEST (1 test)
    // =========================================================================

    #[tokio::test]
    async fn test_trait_is_object_safe() {
        // Test that EmbeddingModel can be used as a trait object
        let model: Box<dyn EmbeddingModel> = Box::new(
            TestModel::new(ModelId::Semantic, vec![InputType::Text])
        );

        assert_eq!(model.model_id(), ModelId::Semantic);
        assert!(model.supports_input_type(InputType::Text));
        assert!(!model.supports_input_type(InputType::Image));

        let input = ModelInput::text("Test").unwrap();
        let embedding = model.embed(&input).await.unwrap();
        assert_eq!(embedding.model_id, ModelId::Semantic);
    }

    // =========================================================================
    // CONCURRENT USAGE TEST (1 test)
    // =========================================================================

    #[tokio::test]
    async fn test_model_can_be_shared_across_tasks() {
        let model = Arc::new(TestModel::new(
            ModelId::Entity,
            vec![InputType::Text, InputType::Code],
        ));

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let m = Arc::clone(&model);
                tokio::spawn(async move {
                    let input = ModelInput::text(format!("Task {}", i)).unwrap();
                    m.embed(&input).await
                })
            })
            .collect();

        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
            assert_eq!(result.unwrap().model_id, ModelId::Entity);
        }
    }

    // =========================================================================
    // ALL 12 MODELS DIMENSION TEST (1 test)
    // =========================================================================

    #[tokio::test]
    async fn test_all_12_models_produce_correct_dimensions() {
        for model_id in ModelId::all() {
            let model = TestModel::new(*model_id, vec![InputType::Text]);
            let input = ModelInput::text("Test embedding").unwrap();
            let embedding = model.embed(&input).await.unwrap();

            assert_eq!(
                embedding.dimension(),
                model_id.dimension(),
                "Model {:?} produced wrong dimension",
                model_id
            );
        }
    }

    // =========================================================================
    // INPUT TYPE COVERAGE TEST (1 test)
    // =========================================================================

    #[test]
    fn test_all_input_types_can_be_supported() {
        let model = TestModel::new(
            ModelId::Multimodal,
            vec![
                InputType::Text,
                InputType::Code,
                InputType::Image,
                InputType::Audio,
            ],
        );

        for input_type in InputType::all() {
            assert!(
                model.supports_input_type(*input_type),
                "Model should support {:?}",
                input_type
            );
        }
    }

    // =========================================================================
    // DYN TRAIT REFERENCE TEST (1 test)
    // =========================================================================

    #[tokio::test]
    async fn test_dyn_trait_reference_works() {
        let model = TestModel::new(ModelId::Graph, vec![InputType::Text]);
        let model_ref: &dyn EmbeddingModel = &model;

        assert_eq!(model_ref.model_id(), ModelId::Graph);
        assert_eq!(model_ref.dimension(), 384);
        assert!(model_ref.is_initialized());

        let input = ModelInput::text("Reference test").unwrap();
        let embedding = model_ref.embed(&input).await.unwrap();
        assert_eq!(embedding.model_id, ModelId::Graph);
    }

    // =========================================================================
    // MULTIPLE MODEL INSTANCES TEST (1 test)
    // =========================================================================

    #[tokio::test]
    async fn test_multiple_model_instances_independent() {
        let semantic = TestModel::new(ModelId::Semantic, vec![InputType::Text]);
        let code_model = TestModel::new(ModelId::Code, vec![InputType::Code]);

        let text_input = ModelInput::text("Hello").unwrap();
        let code_input = ModelInput::code("fn main() {}", "rust").unwrap();

        let sem_emb = semantic.embed(&text_input).await.unwrap();
        let code_emb = code_model.embed(&code_input).await.unwrap();

        assert_eq!(sem_emb.model_id, ModelId::Semantic);
        assert_eq!(code_emb.model_id, ModelId::Code);
        assert_eq!(sem_emb.dimension(), 1024);
        assert_eq!(code_emb.dimension(), 256);
    }

    // =========================================================================
    // HASHSET USAGE FOR INPUT TYPE CHECK (1 test)
    // =========================================================================

    #[test]
    fn test_supported_types_can_use_hashset() {
        let model = TestModel::new(
            ModelId::Multimodal,
            vec![InputType::Text, InputType::Image],
        );

        let supported_set: HashSet<InputType> =
            model.supported_input_types().iter().copied().collect();

        assert!(supported_set.contains(&InputType::Text));
        assert!(supported_set.contains(&InputType::Image));
        assert!(!supported_set.contains(&InputType::Code));
        assert!(!supported_set.contains(&InputType::Audio));
    }
}
