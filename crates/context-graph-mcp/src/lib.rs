#![deny(deprecated)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::type_complexity)]
#![allow(clippy::result_large_err)]

//! Context Graph MCP Server Library
//!
//! JSON-RPC 2.0 server implementing the Model Context Protocol (MCP)
//! for the 13-embedder semantic memory retrieval system.
//!
//! This library exposes the handlers and protocol types for integration testing.

pub mod adapters;
pub mod gpu_clustering;
pub mod handlers;
pub mod middleware;
pub mod monitoring;
pub mod protocol;
pub mod server;
pub mod tools;
pub mod transport;
pub mod weights;
