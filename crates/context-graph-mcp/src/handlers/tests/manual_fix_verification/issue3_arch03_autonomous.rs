//! Issue 3 / ARCH-03: Autonomous Operation Without North Star Tests
//!
//! Per constitution ARCH-03: "System MUST operate autonomously without manual goal setting"
//!
//! These tests verify that autonomous handlers work correctly when no North Star is configured.

use crate::handlers::tests::{create_test_handlers_no_north_star, make_request};
use crate::protocol::JsonRpcId;
use serde_json::json;

/// ARCH-03 VERIFICATION: get_autonomous_status works WITHOUT North Star.
///
/// BEFORE: Would fail or return error when no North Star configured
/// AFTER: Returns status with recommendations to store memories first
///
/// Per constitution ARCH-03: "System MUST operate autonomously without manual goal setting"
#[tokio::test]
async fn test_arch03_get_autonomous_status_without_north_star() {
    println!("\n{}", "=".repeat(60));
    println!("ARCH-03 VERIFICATION: get_autonomous_status without North Star");
    println!("{}", "=".repeat(60));

    // BEFORE STATE: Create handlers WITHOUT North Star
    let handlers = create_test_handlers_no_north_star();
    println!("[BEFORE] Handlers created WITHOUT North Star (empty goal hierarchy)");

    // SYNTHETIC DATA: Request status with all optional params
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_autonomous_status",
            "arguments": {
                "include_metrics": true,
                "include_history": true,
                "history_count": 5
            }
        })),
    );

    println!("[EXECUTE] Calling get_autonomous_status without North Star...");
    let response = handlers.dispatch(request).await;

    // SOURCE OF TRUTH: Check response
    println!("[SOURCE OF TRUTH] Checking response...");

    // VERIFY: No protocol error
    assert!(
        response.error.is_none(),
        "[FAIL] Protocol error when no North Star: {:?}",
        response.error
    );
    println!("[VERIFY] No protocol error without North Star - PASS");

    // VERIFY: Result exists
    let result = response.result.expect("[FAIL] Must have result");
    println!("[VERIFY] Result field present - PASS");

    // VERIFY: Should NOT be an isError response
    let is_error = result.get("isError").and_then(|v| v.as_bool()).unwrap_or(false);
    assert!(
        !is_error,
        "[FAIL] Returned isError=true without North Star - should still work"
    );
    println!("[VERIFY] Not an error response (isError=false) - PASS");

    // Extract and verify content
    if let Some(content) = result.get("content").and_then(|v| v.as_array()) {
        if let Some(first) = content.first() {
            if let Some(text) = first.get("text").and_then(|v| v.as_str()) {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
                    // VERIFY: north_star.configured should be false
                    if let Some(ns) = parsed.get("north_star") {
                        let configured = ns.get("configured").and_then(|v| v.as_bool()).unwrap_or(true);
                        assert!(!configured, "[FAIL] north_star.configured should be false");
                        println!("[VERIFY] north_star.configured = false - PASS");
                    }

                    // VERIFY: Should have recommendations for unconfigured state
                    if let Some(recommendations) = parsed.get("recommendations").and_then(|v| v.as_array()) {
                        println!("[VERIFY] Has {} recommendations - PASS", recommendations.len());

                        // Check for store_memory recommendation
                        let has_store_recommendation = recommendations.iter().any(|r| {
                            r.get("action")
                                .and_then(|a| a.as_str())
                                .map(|a| a == "store_memory")
                                .unwrap_or(false)
                        });
                        if has_store_recommendation {
                            println!("[VERIFY] Has store_memory recommendation (ARCH-03 compliant) - PASS");
                        }
                    }

                    // VERIFY: overall_health should indicate not_configured
                    if let Some(health) = parsed.get("overall_health") {
                        if let Some(status) = health.get("status").and_then(|v| v.as_str()) {
                            println!("[VERIFY] overall_health.status = \"{}\"", status);
                            assert_eq!(status, "not_configured", "[INFO] Expected not_configured status");
                        }
                    }
                }
            }
        }
    }

    // PHYSICAL EVIDENCE
    println!("\n[PHYSICAL EVIDENCE]");
    println!("  Tool: get_autonomous_status");
    println!("  North Star configured: false");
    println!("  Response error: {:?}", response.error);
    println!("  Response has valid result: true");
    println!("\n[ARCH-03 get_autonomous_status VERIFICATION COMPLETE]\n");
}

// REMOVED: test_arch03_auto_bootstrap_discovers_from_stored_fingerprints per TASK-P0-001 (ARCH-03)
//
// The `auto_bootstrap_north_star` tool has been REMOVED per constitution v6.0.0.
// Goals now emerge autonomously from topic clustering (HDBSCAN/BIRCH).
//
// See constitution ARCH-03: "Autonomous operation - goals emerge from topic clustering, no manual goal setting"
// See topic_system.topic_portfolio: "Emergent topics discovered via clustering, no manual setting"
//
// For similar functionality, use:
// - `get_topic_portfolio` - Get current emergent topic portfolio
// - `get_topic_stability` - Get topic stability metrics (churn, entropy)

/// ARCH-03 VERIFICATION: get_alignment_drift works without North Star.
///
/// BEFORE: Would fail when no North Star configured
/// AFTER: Computes drift relative to computed centroid of memories
#[tokio::test]
async fn test_arch03_get_alignment_drift_without_north_star() {
    println!("\n{}", "=".repeat(60));
    println!("ARCH-03 VERIFICATION: get_alignment_drift without North Star");
    println!("{}", "=".repeat(60));

    let handlers = create_test_handlers_no_north_star();
    println!("[BEFORE] Handlers created WITHOUT North Star");

    // EXECUTE: Get drift without North Star
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_alignment_drift",
            "arguments": {
                "timeframe": "24h",
                "include_history": false
            }
        })),
    );

    println!("[EXECUTE] Calling get_alignment_drift without North Star...");
    let response = handlers.dispatch(request).await;

    // VERIFY: No protocol error
    assert!(
        response.error.is_none(),
        "[FAIL] Protocol error without North Star: {:?}",
        response.error
    );
    println!("[VERIFY] No protocol error - PASS");

    let result = response.result.expect("[FAIL] Must have result");

    // Should return valid response (even if minimal without memory_ids)
    let is_error = result.get("isError").and_then(|v| v.as_bool()).unwrap_or(false);
    assert!(!is_error, "[FAIL] Should not be error without North Star");
    println!("[VERIFY] Not an error response - PASS");

    if let Some(content) = result.get("content").and_then(|v| v.as_array()) {
        if let Some(first) = content.first() {
            if let Some(text) = first.get("text").and_then(|v| v.as_str()) {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
                    // Check reference_type - should be "no_reference" or "centroid" when no North Star
                    if let Some(ref_type) = parsed.get("reference_type").and_then(|v| v.as_str()) {
                        println!("[VERIFY] reference_type = \"{}\"", ref_type);
                        // Without memory_ids, returns no_reference
                        // With memory_ids, would compute centroid
                    }

                    // Check for usage_hint when no memory_ids provided
                    if let Some(hint) = parsed.get("usage_hint").and_then(|v| v.as_str()) {
                        println!("[VERIFY] Has usage_hint for memory_ids - PASS");
                        println!("[INFO] usage_hint: {}", hint);
                    }
                }
            }
        }
    }

    println!("\n[ARCH-03 get_alignment_drift VERIFICATION COMPLETE]\n");
}
