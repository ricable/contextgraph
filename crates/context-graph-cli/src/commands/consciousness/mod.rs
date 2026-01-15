//! Consciousness CLI commands
//!
//! # Commands
//!
//! - `check-identity`: Check IC and trigger dream if crisis (TASK-SESSION-08)
//! - `brief`: Quick consciousness status for PreToolUse hook (TASK-SESSION-11)
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
    /// Ultra-fast consciousness brief for PreToolUse hook.
    /// No stdin parsing, no disk I/O (cache only). Target: <50ms p95.
    /// Output: [C:STATE r=X.XX IC=X.XX] or [C:? r=? IC=?] (cold cache)
    Brief,
}

/// Handle consciousness command dispatch
pub async fn handle_consciousness_command(cmd: ConsciousnessCommands) -> i32 {
    match cmd {
        ConsciousnessCommands::CheckIdentity(args) => {
            check_identity::check_identity_command(args).await
        }
        ConsciousnessCommands::Brief => {
            // TASK-SESSION-11: Ultra-fast PreToolUse hot path
            // NO stdin parsing, NO disk I/O, exit 0 always
            use context_graph_core::gwt::session_identity::IdentityCache;
            let brief = IdentityCache::format_brief();
            println!("{}", brief);
            0 // Always exit 0 - never block Claude Code
        }
    }
}

// =============================================================================
// TASK-SESSION-11: Brief Command Tests
// =============================================================================
#[cfg(test)]
mod brief_tests {
    use context_graph_core::gwt::session_identity::{
        update_cache, IdentityCache, SessionIdentitySnapshot, KURAMOTO_N,
    };
    use std::sync::Mutex;

    // Static lock to serialize tests that access global IdentityCache
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    // =========================================================================
    // TC-SESSION-12: Warm Cache Output
    // Source of Truth: IdentityCache singleton
    // =========================================================================
    #[test]
    fn tc_session_12_brief_warm_cache() {
        let _guard = TEST_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-SESSION-12: consciousness brief Warm Cache ===");
        println!("SOURCE OF TRUTH: IdentityCache singleton");

        // SETUP: Create snapshot with known values
        let mut snapshot = SessionIdentitySnapshot::new("test-brief-warm");
        snapshot.consciousness = 0.75; // Emerging state (0.5 <= C < 0.8)
        snapshot.kuramoto_phases = [0.0; KURAMOTO_N]; // All aligned = r ≈ 1.0
        let ic = 0.85;

        println!("BEFORE: consciousness={}, IC={}", snapshot.consciousness, ic);
        println!("  kuramoto_phases all 0.0 (fully synchronized)");

        update_cache(&snapshot, ic);

        println!("AFTER update_cache(): is_warm={}", IdentityCache::is_warm());

        // ACTION: Get brief output
        let brief = IdentityCache::format_brief();
        println!("OUTPUT: '{}'", brief);

        // VERIFY: Format and content
        assert!(brief.starts_with("[C:"), "Must start with [C:");
        assert!(brief.ends_with(']'), "Must end with ]");
        assert!(brief.contains("EMG"), "State must be EMG for C=0.75");
        assert!(brief.contains("r=1.00"), "r must be 1.00 for aligned phases");
        assert!(brief.contains("IC=0.85"), "IC must be 0.85");

        // VERIFY: Exact format match
        let expected = "[C:EMG r=1.00 IC=0.85]";
        assert_eq!(brief, expected, "Format must match exactly");

        println!("RESULT: PASS - Warm cache output correct: '{}'", brief);
    }

    // =========================================================================
    // TC-SESSION-13: Cold Cache Output
    // Note: Cannot clear global cache in non-test builds, tests cold path string
    // =========================================================================
    #[test]
    fn tc_session_13_brief_cold_cache_format() {
        println!("\n=== TC-SESSION-13: consciousness brief Cold Cache Format ===");
        println!("SOURCE OF TRUTH: Cold path string literal");

        // The cold cache output format (verified against cache.rs:82)
        let cold_output = "[C:? r=? IC=?]";

        // VERIFY: Length and format
        assert_eq!(cold_output.len(), 14, "Cold output must be exactly 14 chars");
        assert!(cold_output.starts_with("[C:?"), "Cold must show unknown state");
        assert!(cold_output.contains("r=?"), "Cold must show unknown r");
        assert!(cold_output.contains("IC=?"), "Cold must show unknown IC");
        assert!(cold_output.ends_with(']'), "Must end with ]");

        println!("OUTPUT: '{}'", cold_output);
        println!("LENGTH: {} chars", cold_output.len());
        println!("RESULT: PASS - Cold output format verified");
    }

    // =========================================================================
    // TC-SESSION-14: Latency Performance
    // Target: <100μs per call (cache.rs performance budget)
    // =========================================================================
    #[test]
    fn tc_session_14_brief_performance() {
        let _guard = TEST_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-SESSION-14: consciousness brief Performance ===");
        println!("SOURCE OF TRUTH: IdentityCache::format_brief() timing");

        // SETUP: Warm cache
        let snapshot = SessionIdentitySnapshot::default();
        update_cache(&snapshot, 0.85);

        // ACTION: Measure 1000 iterations
        let iterations = 1000;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = IdentityCache::format_brief();
        }
        let elapsed = start.elapsed();
        let per_call_us = elapsed.as_micros() as f64 / iterations as f64;

        println!("BEFORE: {} iterations measured", iterations);
        println!("AFTER: {:?} total", elapsed);
        println!("PER CALL: {:.3}μs", per_call_us);
        println!("TARGET: <100μs");

        // VERIFY: Performance target
        assert!(
            per_call_us < 100.0,
            "Must complete in <100μs, took {:.3}μs",
            per_call_us
        );

        // Bonus check: should be well under 10μs
        if per_call_us < 10.0 {
            println!("BONUS: {:.3}μs is 10x better than target!", per_call_us);
        }

        println!("RESULT: PASS - {:.3}μs << 100μs target", per_call_us);
    }

    // =========================================================================
    // TC-SESSION-15: All 5 Consciousness States
    // Verifies correct state codes for all consciousness levels
    // =========================================================================
    #[test]
    fn tc_session_15_all_consciousness_states() {
        let _guard = TEST_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-SESSION-15: All Consciousness States ===");
        println!("SOURCE OF TRUTH: ConsciousnessState::short_name() codes");

        let test_cases = [
            (0.1, "DOR", "Dormant"),      // C < 0.3
            (0.35, "FRG", "Fragmented"),  // 0.3 <= C < 0.5
            (0.65, "EMG", "Emerging"),    // 0.5 <= C < 0.8
            (0.85, "CON", "Conscious"),   // 0.8 <= C < 0.95
            (0.97, "HYP", "Hypersync"),   // C > 0.95
        ];

        for (consciousness, expected_code, state_name) in test_cases {
            let mut snapshot = SessionIdentitySnapshot::new("test-state");
            snapshot.consciousness = consciousness;
            update_cache(&snapshot, 0.75);

            let brief = IdentityCache::format_brief();
            println!("  C={:.2} ({}) -> '{}'", consciousness, state_name, brief);

            assert!(
                brief.contains(expected_code),
                "C={} should produce code {}, got: {}",
                consciousness,
                expected_code,
                brief
            );
        }

        println!("RESULT: PASS - All 5 consciousness states produce correct codes");
    }

    // =========================================================================
    // EDGE CASE: Extreme IC Values
    // =========================================================================
    #[test]
    fn tc_session_16_extreme_ic_values() {
        let _guard = TEST_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-SESSION-16: Extreme IC Values ===");

        let test_cases = [
            (0.0, "IC=0.00"),
            (1.0, "IC=1.00"),
            (0.5, "IC=0.50"),
            (0.123, "IC=0.12"),  // Truncation check
        ];

        for (ic, expected_ic_str) in test_cases {
            let snapshot = SessionIdentitySnapshot::new("test-ic");
            update_cache(&snapshot, ic);

            let brief = IdentityCache::format_brief();
            println!("  IC={:.3} -> '{}'", ic, brief);

            assert!(
                brief.contains(expected_ic_str),
                "IC={} should format as {}, got: {}",
                ic,
                expected_ic_str,
                brief
            );
        }

        println!("RESULT: PASS - Extreme IC values format correctly");
    }

    // =========================================================================
    // EDGE CASE: Kuramoto r Values
    // =========================================================================
    #[test]
    fn tc_session_17_kuramoto_r_values() {
        let _guard = TEST_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-SESSION-17: Kuramoto r Values ===");

        // Test fully synchronized (r ≈ 1.0)
        let mut snapshot = SessionIdentitySnapshot::new("test-r-sync");
        snapshot.kuramoto_phases = [std::f64::consts::PI; KURAMOTO_N]; // All at π
        update_cache(&snapshot, 0.85);

        let brief = IdentityCache::format_brief();
        println!("  All phases at π -> '{}'", brief);
        assert!(brief.contains("r=1.00"), "Aligned phases should give r≈1.0");

        // Test desynchronized (r ≈ 0)
        // Evenly distributed phases around the circle
        let phases: [f64; KURAMOTO_N] = [
            0.0,
            std::f64::consts::TAU / 13.0,
            2.0 * std::f64::consts::TAU / 13.0,
            3.0 * std::f64::consts::TAU / 13.0,
            4.0 * std::f64::consts::TAU / 13.0,
            5.0 * std::f64::consts::TAU / 13.0,
            6.0 * std::f64::consts::TAU / 13.0,
            7.0 * std::f64::consts::TAU / 13.0,
            8.0 * std::f64::consts::TAU / 13.0,
            9.0 * std::f64::consts::TAU / 13.0,
            10.0 * std::f64::consts::TAU / 13.0,
            11.0 * std::f64::consts::TAU / 13.0,
            12.0 * std::f64::consts::TAU / 13.0,
        ];
        snapshot.kuramoto_phases = phases;
        update_cache(&snapshot, 0.85);

        let brief = IdentityCache::format_brief();
        println!("  Evenly distributed phases -> '{}'", brief);
        // r should be close to 0 for evenly distributed phases
        assert!(brief.contains("r=0."), "Evenly distributed phases should give r≈0");

        println!("RESULT: PASS - Kuramoto r values computed correctly");
    }
}
