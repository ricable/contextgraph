//! Lifecycle management module (Marblestone lambda weights).
//!
//! Implements lifecycle stage transitions and lambda weight computation for
//! the UTL (Unified Theory of Learning) framework. Knowledge bases evolve
//! through distinct lifecycle stages, each with different learning dynamics.
//!
//! # Lifecycle Stages
//!
//! The Marblestone model defines three stages:
//!
//! - **Infancy** (0-50 interactions): High novelty capture, `lambda_s=0.7, lambda_c=0.3`
//! - **Growth** (50-500 interactions): Balanced learning, `lambda_s=0.5, lambda_c=0.5`
//! - **Maturity** (500+ interactions): Coherence focus, `lambda_s=0.3, lambda_c=0.7`
//!
//! # Constitution Reference
//!
//! ```text
//! Infancy (n=0-50):   lambda_s=0.7, lambda_c=0.3, stance="capture-novelty"
//! Growth (n=50-500):  lambda_s=0.5, lambda_c=0.5, stance="balanced"
//! Maturity (n=500+):  lambda_s=0.3, lambda_c=0.7, stance="curation-coherence"
//! ```
//!
//! # Example
//!
//! ```
//! use context_graph_utl::config::LifecycleConfig;
//! use context_graph_utl::lifecycle::{LifecycleManager, LifecycleStage, LifecycleLambdaWeights};
//!
//! // Create manager from default config
//! let config = LifecycleConfig::default();
//! let mut manager = LifecycleManager::new(&config);
//!
//! // Initial stage is Infancy
//! assert_eq!(manager.current_stage(), LifecycleStage::Infancy);
//!
//! // Get current lambda weights
//! let weights = manager.current_weights();
//! assert!((weights.lambda_s() - 0.7).abs() < 0.001);
//! assert!((weights.lambda_c() - 0.3).abs() < 0.001);
//!
//! // Increment interactions and track stage transitions
//! for _ in 0..60 {
//!     manager.increment();
//! }
//! assert_eq!(manager.current_stage(), LifecycleStage::Growth);
//! ```

mod lambda;
mod manager;
mod stage;

pub use lambda::LifecycleLambdaWeights;
pub use manager::LifecycleManager;
pub use stage::LifecycleStage;
