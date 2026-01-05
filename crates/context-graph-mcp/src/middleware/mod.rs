//! CognitivePulse middleware module.
//!
//! Provides real-time UTL cognitive state injection for MCP responses.
//! Every MCP tool response includes a `_cognitive_pulse` field containing
//! the current system state.
//!
//! # Architecture
//!
//! The middleware extracts UTL metrics from the processor and creates
//! a lightweight pulse struct that:
//! - Reports current entropy (surprise) level
//! - Reports current coherence level
//! - Reports current learning score
//! - Classifies into Johari quadrant
//! - Suggests appropriate action per constitution.yaml
//!
//! # Performance
//!
//! Target: < 1ms per computation.
//! Warning logged if exceeded, but request continues.

mod cognitive_pulse;

pub use cognitive_pulse::{CognitivePulse, CognitivePulseError};
