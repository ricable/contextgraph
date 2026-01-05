//! Learning computation types and utilities for UTL.
//!
//! This module contains:
//! - [`compute_learning_magnitude`]: Core UTL formula implementation
//! - [`compute_learning_magnitude_validated`]: Validated version with input checking
//! - [`LearningIntensity`]: Quick classification of learning magnitude
//! - [`LearningSignal`]: Complete UTL computation output
//! - [`UtlState`]: Compact storage representation

mod intensity;
mod magnitude;
mod signal;
mod state;

pub use intensity::LearningIntensity;
pub use magnitude::{compute_learning_magnitude, compute_learning_magnitude_validated};
pub use signal::LearningSignal;
pub use state::UtlState;
