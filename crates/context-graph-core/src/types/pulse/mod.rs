//! Cognitive Pulse types for meta-cognitive state tracking.
//!
//! Every MCP tool response includes a Cognitive Pulse header to convey
//! the current system state and suggest next actions.

mod action;
mod cognitive_pulse;

#[cfg(test)]
mod tests;

pub use self::action::SuggestedAction;
pub use self::cognitive_pulse::CognitivePulse;
