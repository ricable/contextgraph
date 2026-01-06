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
//! - `utl`: UTL computation handlers
//! - `system`: System status and health handlers

mod core;
mod lifecycle;
mod memory;
mod search;
mod system;
mod tools;
mod utl;

#[cfg(test)]
mod tests;

// Re-export the main Handlers struct for backward compatibility
pub use self::core::Handlers;
