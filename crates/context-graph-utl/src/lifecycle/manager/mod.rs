//! Lifecycle manager for tracking and transitioning between stages.
//!
//! The `LifecycleManager` tracks the current lifecycle stage of a knowledge
//! base and handles transitions between stages as interactions accumulate.
//! It supports both automatic transitions (based on interaction count) and
//! manual transitions.
//!
//! # Constitution Reference
//!
//! ```text
//! Infancy (n=0-50):   lambda_s=0.7, lambda_c=0.3, stance="capture-novelty"
//! Growth (n=50-500):  lambda_s=0.5, lambda_c=0.5, stance="balanced"
//! Maturity (n=500+):  lambda_s=0.3, lambda_c=0.7, stance="curation-coherence"
//! ```
//!
//! # Module Organization
//!
//! - [`types`] - Core type definitions (`LifecycleManager` struct)
//! - [`core`] - Constructors and basic accessors
//! - [`transitions`] - Stage transition logic
//! - [`progress`] - Progress tracking utilities

mod core;
mod progress;
mod transitions;
mod types;

#[cfg(test)]
mod tests_core;
#[cfg(test)]
mod tests_transitions;

// Re-export the main type for backwards compatibility
pub use types::LifecycleManager;
