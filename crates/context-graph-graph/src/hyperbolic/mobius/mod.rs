//! PoincareBall implementation with Mobius algebra operations.
//!
//! # Poincare Ball Model
//!
//! The Poincare ball model represents hyperbolic space as the open unit ball.
//! Mobius operations provide the algebra for vector addition, distances, and
//! exponential/logarithmic maps between tangent spaces and the manifold.
//!
//! # Mathematics
//!
//! - Mobius addition: x + y = ((1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y) /
//!   (1 + 2c<x,y> + c^2||x||^2||y||^2)
//! - Distance: d(x,y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-x) + y||)
//! - Exp map: Maps tangent vector at x to point on manifold
//! - Log map: Maps point y to tangent vector at x (inverse of exp_map)
//!
//! # Performance Targets
//!
//! - distance(): <10us per pair
//! - mobius_add(): <5us per operation
//!
//! # Constitution Reference
//!
//! - perf.latency.entailment_check: <1ms (this contributes ~1% budget)
//! - contextprd.md Section 4.4: Poincare Ball d(x,y) formula
//!
//! # Module Structure
//!
//! - `types`: Core `PoincareBall` struct definition
//! - `operations`: Mobius addition and distance calculations
//! - `maps`: Exponential and logarithmic maps

mod maps;
mod operations;
mod types;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_roundtrip;

// Re-export main type for backwards compatibility
pub use self::types::PoincareBall;
