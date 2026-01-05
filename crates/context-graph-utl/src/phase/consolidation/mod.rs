//! Memory consolidation phase detection.
//!
//! Implements consolidation phase detection based on activity levels,
//! inspired by sleep stage dynamics (NREM/REM) for memory consolidation.

mod detector;
mod phase;

#[cfg(test)]
mod tests;

pub use self::detector::PhaseDetector;
pub use self::phase::ConsolidationPhase;
