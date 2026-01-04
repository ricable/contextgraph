//! Phase oscillation (φ) module.
//!
//! Implements phase synchronization and memory consolidation:
//! - Phase oscillator for learning rhythms
//! - Consolidation phase detection (NREM/REM)
//! - Phase alignment computation
//!
//! # Constitution Reference
//!
//! - `φ` range: `[0, π]` representing phase angle
//! - `cos(φ) = 1.0` when fully synchronized (φ = 0)
//! - `cos(φ) = -1.0` when anti-phase (φ = π)
//! - L4 operates at 100Hz reference frequency
//!
//! # Example
//!
//! ```
//! use context_graph_utl::phase::{PhaseOscillator, ConsolidationPhase, PhaseDetector};
//! use context_graph_utl::config::PhaseConfig;
//! use std::time::Duration;
//!
//! // Create phase oscillator
//! let config = PhaseConfig::default();
//! let mut oscillator = PhaseOscillator::new(&config);
//!
//! // Update phase based on elapsed time
//! oscillator.update(Duration::from_millis(10));
//!
//! // Get current phase and cosine
//! let phase = oscillator.phase();
//! let cos_phi = oscillator.cos_phase();
//!
//! assert!(phase >= 0.0 && phase <= std::f32::consts::PI);
//! assert!(cos_phi >= -1.0 && cos_phi <= 1.0);
//!
//! // Detect consolidation phase
//! let detector = PhaseDetector::new();
//! let phase_type = detector.detect_phase(0.3); // Low activity
//! ```

mod consolidation;
mod oscillator;

pub use consolidation::{ConsolidationPhase, PhaseDetector};
pub use oscillator::PhaseOscillator;
