//! Main surprise calculator combining KL divergence and embedding distance.
//!
//! This module provides the primary `SurpriseCalculator` struct that computes
//! overall surprise (Delta-S) by combining multiple signals:
//! - KL divergence for distribution comparison
//! - Embedding distance for semantic novelty
//!
//! # Constitution Reference
//!
//! The surprise component (Delta-S) is part of the UTL formula:
//! `L = f((Delta-S x Delta-C) * w_e * cos(phi))`
//!
//! Where Delta-S represents entropy/novelty in range [0, 1].
//!
//! # Numerical Stability
//!
//! Per AP-009, all outputs are clamped to valid ranges with no NaN or Infinity values.

mod compute;
mod constructors;
mod types;

pub use types::SurpriseCalculator;

#[cfg(test)]
mod tests;
