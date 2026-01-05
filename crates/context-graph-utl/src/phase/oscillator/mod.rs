//! Phase oscillator for learning rhythms.
//!
//! Provides smooth phase oscillation with configurable frequency and coupling
//! strength for the UTL formula phase component `cos(φ)`.

mod core;
mod coupling;
mod types;

#[cfg(test)]
mod tests;

// Re-export the main type for backwards compatibility
pub use types::PhaseOscillator;
