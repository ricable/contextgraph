//! UTL Processor - Main orchestrator for UTL computation pipeline.
//!
//! The `UtlProcessor` integrates all UTL components into a unified pipeline:
//! - SurpriseCalculator for delta_s
//! - CoherenceTracker for delta_c
//! - EmotionalWeightCalculator for w_e
//! - PhaseOscillator for phi
//! - LifecycleManager for lambda weights
//! - JohariClassifier for quadrant classification
//!
//! # Constitution Reference
//! - UTL formula: `L = f((ΔS × ΔC) · wₑ · cos φ)` (constitution.yaml:152)
//! - Lifecycle stages: Infancy/Growth/Maturity (constitution.yaml:165-167)

mod session;
mod utl_processor;

#[cfg(test)]
mod tests;

// Re-export public types for backwards compatibility
pub use self::session::SessionContext;
pub use self::utl_processor::UtlProcessor;
