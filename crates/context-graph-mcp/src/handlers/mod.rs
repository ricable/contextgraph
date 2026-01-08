//! Request handlers for MCP methods.
//!
//! This module provides the main `Handlers` struct that dispatches JSON-RPC
//! requests to their appropriate handler methods.
//!
//! # Module Organization
//!
//! - `core`: Core Handlers struct and dispatch logic
//! - `lifecycle`: MCP lifecycle handlers (initialize, shutdown)
//! - `tools`: MCP tool call handlers
//! - `memory`: Legacy memory operation handlers
//! - `search`: Multi-embedding weighted search handlers (TASK-S002)
//! - `purpose`: Purpose and goal alignment handlers (TASK-S003)
//! - `johari`: Johari quadrant handlers (TASK-S004)
//! - `utl`: UTL computation handlers
//! - `system`: System status and health handlers
//! - `gwt_traits`: GWT provider traits for consciousness/Kuramoto integration (TASK-GWT-001)
//! - `gwt_providers`: Real GWT provider implementations wrapping actual components (TASK-GWT-001)

mod core;
mod johari;
mod lifecycle;
mod memory;
mod purpose;
mod search;
mod system;
mod tools;
mod utl;
pub mod gwt_traits;
pub mod gwt_providers;

#[cfg(test)]
mod tests;

// Re-export the main Handlers struct for backward compatibility
pub use self::core::Handlers;

// Re-export GWT traits for external use (TASK-GWT-001)
pub use self::gwt_traits::{
    GwtSystemProvider, KuramotoProvider, MetaCognitiveProvider,
    SelfEgoProvider, WorkspaceProvider, NUM_OSCILLATORS,
};

// Re-export GWT provider implementations for wiring (TASK-GWT-001)
pub use self::gwt_providers::{
    GwtSystemProviderImpl, KuramotoProviderImpl, MetaCognitiveProviderImpl,
    SelfEgoProviderImpl, WorkspaceProviderImpl,
};
