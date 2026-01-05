//! EntailmentCone implementation for O(1) IS-A hierarchy queries.
//!
//! An entailment cone in hyperbolic space enables efficient hierarchical
//! reasoning. A concept's cone contains all concepts it subsumes (entails).
//! Checking if concept A is a subconcept of B is O(1): check if A's position
//! lies within B's cone.
//!
//! # Aperture Decay
//!
//! Aperture decreases with hierarchy depth:
//! - Root concepts have wide cones (capture many descendants)
//! - Leaf concepts have narrow cones (very specific)
//! - Formula: `aperture = base * decay^depth`, clamped to [min, max]
//!
//! # Performance Targets
//!
//! - Cone containment check: <50μs CPU
//! - Entailment check: <1ms total
//! - Target hardware: RTX 5090, CUDA 13.1, Compute 12.0
//!
//! # Constitution Reference
//!
//! - perf.latency.entailment_check: <1ms
//! - Section 9 "HYPERBOLIC ENTAILMENT CONES" in contextprd.md
//!
//! # Module Structure
//!
//! - `types`: Core [`EntailmentCone`] struct definition
//! - `constructors`: Constructor methods (`new`, `with_aperture`)
//! - `operations`: Core operations (containment, membership, aperture update)
//! - `validation`: Validation methods (`is_valid`, `validate`)

mod constructors;
mod operations;
mod types;
mod validation;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_containment;

// Re-export the main type
pub use types::EntailmentCone;
