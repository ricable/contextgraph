//! Consciousness CLI commands
//!
//! # Commands
//!
//! - `check-identity`: Check IC and trigger dream if crisis (TASK-SESSION-08)
//! - `brief`: Quick consciousness status for PreToolUse hook (TASK-SESSION-11 - placeholder)
//!
//! # Constitution Reference
//! - AP-26: Exit code 1 on error, 2 on corruption
//! - AP-38: IC<0.5 MUST auto-trigger dream
//! - AP-42: entropy>0.7 MUST wire to TriggerManager
//! - IDENTITY-002: IC thresholds (Healthy >= 0.9, Good >= 0.7, Warning >= 0.5, Degraded < 0.5)

mod check_identity;

use clap::Subcommand;

pub use check_identity::CheckIdentityArgs;

/// Consciousness subcommands
#[derive(Subcommand)]
pub enum ConsciousnessCommands {
    /// Check identity continuity and optionally trigger dream on crisis
    CheckIdentity(CheckIdentityArgs),
    /// Quick consciousness brief for PreToolUse hook (TASK-SESSION-11 - placeholder)
    Brief,
}

/// Handle consciousness command dispatch
pub async fn handle_consciousness_command(cmd: ConsciousnessCommands) -> i32 {
    match cmd {
        ConsciousnessCommands::CheckIdentity(args) => {
            check_identity::check_identity_command(args).await
        }
        ConsciousnessCommands::Brief => {
            // TASK-SESSION-11: Placeholder
            use context_graph_core::gwt::session_identity::IdentityCache;
            let brief = IdentityCache::format_brief();
            println!("{}", brief);
            0
        }
    }
}
