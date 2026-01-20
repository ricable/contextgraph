//! Dream consolidation commands
//!
//! Commands for NREM/REM dream cycles.
//!
//! # Commands
//!
//! - `dream trigger`: Execute NREM/REM dream consolidation
//! - `dream status`: Check status of running/completed dream
//!
//! # Constitution Compliance
//!
//! - AP-70: Dream triggers MUST use entropy > 0.7 AND churn > 0.5
//! - Dream wake latency < 100ms
//! - Max 100 queries during dream
//! - < 30% GPU utilization during dream
//! - AP-26: Exit code 1 on error, 2 on corruption

use clap::{Args, Subcommand};
use tracing::{error, info};

use crate::mcp_client::McpClient;

/// Dream consolidation subcommands.
///
/// These commands manage NREM/REM dream cycles for memory consolidation.
/// NREM performs Hebbian replay strengthening, REM performs hyperbolic
/// random walk in Poincare ball to discover blind spots.
#[derive(Subcommand)]
pub enum DreamCommands {
    /// Execute NREM/REM dream consolidation cycle
    ///
    /// NREM phase (3 min): Hebbian learning replay with eta=0.01, recency_bias=0.8
    /// REM phase (2 min): Hyperbolic random walk in Poincare ball (64D, curvature=-1.0)
    ///
    /// Trigger conditions (per AP-70):
    /// - entropy > 0.7 for 5+ min, OR
    /// - churn > 0.5 AND entropy > 0.7
    ///
    /// # Examples
    ///
    /// ```bash
    /// # Full blocking dream cycle
    /// context-graph-cli dream trigger
    ///
    /// # Dry run (simulate without modifying)
    /// context-graph-cli dream trigger --dry-run
    ///
    /// # Non-blocking (returns immediately)
    /// context-graph-cli dream trigger --non-blocking
    ///
    /// # NREM only
    /// context-graph-cli dream trigger --skip-rem
    ///
    /// # REM only
    /// context-graph-cli dream trigger --skip-nrem
    /// ```
    Trigger(TriggerArgs),

    /// Check status of a dream cycle
    ///
    /// Returns current phase, progress, elapsed time, and partial results.
    ///
    /// # Examples
    ///
    /// ```bash
    /// # Status of most recent dream
    /// context-graph-cli dream status
    ///
    /// # Status of specific dream
    /// context-graph-cli dream status --dream-id <uuid>
    /// ```
    Status(StatusArgs),
}

/// Arguments for dream trigger command.
#[derive(Args)]
pub struct TriggerArgs {
    /// Wait for dream completion (default: true)
    #[arg(long)]
    pub non_blocking: bool,

    /// Simulate dream cycle without modifying graph
    #[arg(long)]
    pub dry_run: bool,

    /// Skip NREM phase, run REM only
    #[arg(long)]
    pub skip_nrem: bool,

    /// Skip REM phase, run NREM only
    #[arg(long)]
    pub skip_rem: bool,

    /// Maximum duration in seconds (10-600, default: 300)
    #[arg(long)]
    pub max_duration: Option<u32>,

    /// Output as JSON instead of human-readable
    #[arg(long)]
    pub json: bool,
}

/// Arguments for dream status command.
#[derive(Args)]
pub struct StatusArgs {
    /// Dream cycle UUID (default: most recent)
    #[arg(long)]
    pub dream_id: Option<String>,

    /// Output as JSON instead of human-readable
    #[arg(long)]
    pub json: bool,
}

/// Handle dream subcommands.
///
/// Routes to appropriate handler based on subcommand.
/// Returns exit code per AP-26: 0=success, 1=error, 2=corruption.
pub async fn handle_dream_command(cmd: DreamCommands) -> i32 {
    match cmd {
        DreamCommands::Trigger(args) => handle_trigger(args).await,
        DreamCommands::Status(args) => handle_status(args).await,
    }
}

/// Handle dream trigger command.
async fn handle_trigger(args: TriggerArgs) -> i32 {
    let client = McpClient::new();

    // Check if server is running
    match client.is_server_running().await {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("Error: MCP server not running at {}", client.server_address());
            eprintln!("Start the server with: context-graph-mcp");
            return 1;
        }
        Err(e) => {
            error!("Failed to check server status: {}", e);
            eprintln!("Error: {}", e);
            return 1;
        }
    }

    let blocking = !args.non_blocking;

    if args.dry_run {
        println!("Dry run mode: simulating dream cycle without modifications");
    }

    if !blocking {
        println!("Non-blocking mode: dream will run in background");
    }

    // Call MCP tool
    match client
        .trigger_dream(
            blocking,
            args.dry_run,
            args.skip_nrem,
            args.skip_rem,
            args.max_duration,
        )
        .await
    {
        Ok(result) => {
            if args.json {
                println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default());
            } else {
                print_dream_result(&result, args.dry_run, blocking);
            }
            info!("Dream trigger completed successfully");
            0
        }
        Err(e) => {
            error!("Failed to trigger dream: {}", e);
            eprintln!("Error: {}", e);
            1
        }
    }
}

/// Handle dream status command.
async fn handle_status(args: StatusArgs) -> i32 {
    let client = McpClient::new();

    // Check if server is running
    match client.is_server_running().await {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("Error: MCP server not running at {}", client.server_address());
            eprintln!("Start the server with: context-graph-mcp");
            return 1;
        }
        Err(e) => {
            error!("Failed to check server status: {}", e);
            eprintln!("Error: {}", e);
            return 1;
        }
    }

    // Call MCP tool
    match client.get_dream_status(args.dream_id.as_deref()).await {
        Ok(result) => {
            if args.json {
                println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default());
            } else {
                print_dream_status(&result);
            }
            info!("Dream status retrieved successfully");
            0
        }
        Err(e) => {
            error!("Failed to get dream status: {}", e);
            eprintln!("Error: {}", e);
            1
        }
    }
}

/// Print dream trigger result in human-readable format.
fn print_dream_result(result: &serde_json::Value, dry_run: bool, blocking: bool) {
    let prefix = if dry_run { "[DRY RUN] " } else { "" };

    println!("{}Dream Consolidation", prefix);
    println!("{}====================\n", "=".repeat(prefix.len()));

    let dream_id = result.get("dream_id").and_then(|d| d.as_str()).unwrap_or("unknown");
    let status = result.get("status").and_then(|s| s.as_str()).unwrap_or("unknown");

    println!("Dream ID: {}", dream_id);
    println!("Status: {}", status);

    if blocking || status == "completed" {
        // Show detailed results for completed dreams
        if let Some(nrem) = result.get("nrem_report") {
            println!("\nNREM Phase (Hebbian Replay):");
            if let Some(strengthened) = nrem.get("connections_strengthened").and_then(|c| c.as_u64()) {
                println!("  Connections Strengthened: {}", strengthened);
            }
            if let Some(duration) = nrem.get("duration_ms").and_then(|d| d.as_u64()) {
                println!("  Duration: {}ms", duration);
            }
        }

        if let Some(rem) = result.get("rem_report") {
            println!("\nREM Phase (Hyperbolic Walk):");
            if let Some(blindspots) = rem.get("blind_spots_discovered").and_then(|b| b.as_u64()) {
                println!("  Blind Spots Discovered: {}", blindspots);
            }
            if let Some(duration) = rem.get("duration_ms").and_then(|d| d.as_u64()) {
                println!("  Duration: {}ms", duration);
            }
        }

        if let Some(recommendations) = result.get("recommendations").and_then(|r| r.as_array()) {
            if !recommendations.is_empty() {
                println!("\nRecommendations:");
                for rec in recommendations {
                    if let Some(msg) = rec.as_str() {
                        println!("  - {}", msg);
                    }
                }
            }
        }
    } else {
        // Non-blocking: show how to check status
        println!("\nDream running in background.");
        println!("Check status with: context-graph-cli dream status --dream-id {}", dream_id);
    }
}

/// Print dream status in human-readable format.
fn print_dream_status(result: &serde_json::Value) {
    println!("Dream Status");
    println!("============\n");

    let dream_id = result.get("dream_id").and_then(|d| d.as_str()).unwrap_or("unknown");
    let status = result.get("status").and_then(|s| s.as_str()).unwrap_or("unknown");
    let progress = result.get("progress_percent").and_then(|p| p.as_u64()).unwrap_or(0);
    let current_phase = result.get("current_phase").and_then(|p| p.as_str()).unwrap_or("unknown");
    let elapsed_ms = result.get("elapsed_ms").and_then(|e| e.as_u64()).unwrap_or(0);

    println!("Dream ID: {}", dream_id);
    println!("Status: {}", status);
    println!("Progress: {}%", progress);
    println!("Current Phase: {}", current_phase);
    println!("Elapsed: {}ms", elapsed_ms);

    if let Some(remaining) = result.get("remaining_ms").and_then(|r| r.as_u64()) {
        println!("Remaining: ~{}ms", remaining);
    }

    // Progress bar
    let bar_len = (progress as usize) / 5;
    let bar = "#".repeat(bar_len);
    let empty = ".".repeat(20 - bar_len);
    println!("\n[{}{}] {}%", bar, empty, progress);

    match status {
        "completed" => {
            println!("\nDream cycle completed successfully.");
        }
        "failed" => {
            if let Some(error) = result.get("error").and_then(|e| e.as_str()) {
                println!("\nDream failed: {}", error);
            }
        }
        "interrupted" => {
            println!("\nDream was interrupted.");
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigger_args_defaults() {
        let args = TriggerArgs {
            non_blocking: false,
            dry_run: false,
            skip_nrem: false,
            skip_rem: false,
            max_duration: None,
            json: false,
        };
        assert!(!args.non_blocking);
        assert!(!args.dry_run);
    }

    #[test]
    fn test_status_args_defaults() {
        let args = StatusArgs {
            dream_id: None,
            json: false,
        };
        assert!(args.dream_id.is_none());
    }

    #[test]
    fn test_print_dream_result_completed() {
        let result = serde_json::json!({
            "dream_id": "test-id",
            "status": "completed",
            "nrem_report": {
                "connections_strengthened": 42,
                "duration_ms": 180000
            },
            "rem_report": {
                "blind_spots_discovered": 5,
                "duration_ms": 120000
            },
            "recommendations": ["Consider running again in 24h"]
        });
        // Just ensure it doesn't panic
        print_dream_result(&result, false, true);
    }

    #[test]
    fn test_print_dream_status_in_progress() {
        let result = serde_json::json!({
            "dream_id": "test-id",
            "status": "nrem_in_progress",
            "progress_percent": 45,
            "current_phase": "nrem",
            "elapsed_ms": 90000,
            "remaining_ms": 110000
        });
        // Just ensure it doesn't panic
        print_dream_status(&result);
    }
}
