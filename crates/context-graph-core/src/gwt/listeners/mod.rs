//! Workspace Event Listeners
//!
//! Implements listeners for workspace events that wire to subsystems:
//! - DreamEventListener: Queues exiting memories for dream replay
//! - NeuromodulationEventListener: Boosts dopamine on memory entry
//! - MetaCognitiveEventListener: Triggers epistemic action on workspace empty
//! - IdentityContinuityListener: Monitors IC on memory entry (TASK-IDENTITY-P0-006)
//!
//! ## Constitution Reference
//!
//! From constitution.yaml:
//! - neuromod.Dopamine.trigger: "memory_enters_workspace" (lines 162-170)
//! - gwt.global_workspace step 6: "Inhibit: losing candidates receive dopamine reduction"
//! - gwt.workspace_events: memory_exits → dream replay, workspace_empty → epistemic action
//! - gwt.self_ego_node lines 365-392: IC computation and crisis detection

mod dream;
mod identity;
mod meta_cognitive;
mod neuromod;

pub use dream::{DreamConsolidationCallback, DreamEventListener};
pub use identity::IdentityContinuityListener;
pub use meta_cognitive::{MetaCognitiveEventListener, WORKSPACE_EMPTY_THRESHOLD_MS};
pub use neuromod::NeuromodulationEventListener;

#[cfg(test)]
mod tests;
