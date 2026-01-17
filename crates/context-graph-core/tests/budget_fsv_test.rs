//! TASK-P5-002 Full State Verification Manual Tests
//!
//! These tests verify TokenBudget behavior with edge cases and synthetic data.

use context_graph_core::{TokenBudget, BRIEF_BUDGET, InjectionCategory};

// =============================================================================
// Edge Case 1: Empty/Zero Budget Detection
// =============================================================================

#[test]
fn fsv_edge_case_1_invalid_budget_detection() {
    println!("\n=== FSV Edge Case 1: Invalid Budget Detection ===");

    // Create an invalid budget where allocations don't sum to total
    let invalid = TokenBudget {
        total: 1200,
        divergence_budget: 0,
        cluster_budget: 0,
        single_space_budget: 0,
        session_budget: 0,
        reserved: 0,
    };

    println!("Input: TokenBudget with all zero allocations");
    println!("  total: {}", invalid.total);
    println!("  divergence_budget: {}", invalid.divergence_budget);
    println!("  cluster_budget: {}", invalid.cluster_budget);
    println!("  single_space_budget: {}", invalid.single_space_budget);
    println!("  session_budget: {}", invalid.session_budget);
    println!("  reserved: {}", invalid.reserved);

    let is_valid = invalid.is_valid();
    println!("Output: is_valid() = {}", is_valid);
    println!("Expected: false");

    assert!(!is_valid, "Zero allocations should be invalid");
    println!("[PASS] Invalid budget correctly detected\n");
}

// =============================================================================
// Edge Case 2: Minimum Total (100)
// =============================================================================

#[test]
fn fsv_edge_case_2_minimum_total() {
    println!("\n=== FSV Edge Case 2: Minimum Total (100) ===");

    let budget = TokenBudget::with_total(100);

    println!("Input: TokenBudget::with_total(100)");
    println!("Output:");
    println!("  total: {}", budget.total);
    println!("  divergence_budget: {}", budget.divergence_budget);
    println!("  cluster_budget: {}", budget.cluster_budget);
    println!("  single_space_budget: {}", budget.single_space_budget);
    println!("  session_budget: {}", budget.session_budget);
    println!("  reserved: {}", budget.reserved);
    println!("  is_valid: {}", budget.is_valid());

    // Verify expected behavior
    assert_eq!(budget.total, 100, "Total should be 100");
    assert!(budget.divergence_budget > 0, "Divergence budget should be positive");
    assert!(budget.cluster_budget > 0, "Cluster budget should be positive");
    assert!(budget.is_valid(), "Budget should be valid");

    // Verify allocations sum to total
    let sum = budget.divergence_budget
        + budget.cluster_budget
        + budget.single_space_budget
        + budget.session_budget
        + budget.reserved;
    println!("  sum of parts: {}", sum);
    assert_eq!(sum, budget.total, "Parts must sum to total");

    println!("[PASS] Minimum total creates valid budget with positive allocations\n");
}

// =============================================================================
// Edge Case 3: Large Total (1,000,000)
// =============================================================================

#[test]
fn fsv_edge_case_3_large_total() {
    println!("\n=== FSV Edge Case 3: Large Total (1,000,000) ===");

    let budget = TokenBudget::with_total(1_000_000);

    println!("Input: TokenBudget::with_total(1_000_000)");
    println!("Output:");
    println!("  total: {}", budget.total);
    println!("  divergence_budget: {}", budget.divergence_budget);
    println!("  cluster_budget: {}", budget.cluster_budget);
    println!("  single_space_budget: {}", budget.single_space_budget);
    println!("  session_budget: {}", budget.session_budget);
    println!("  reserved: {}", budget.reserved);
    println!("  is_valid: {}", budget.is_valid());

    // Verify no overflow
    assert_eq!(budget.total, 1_000_000, "Total should be 1,000,000");
    assert!(budget.is_valid(), "Budget should be valid");

    // Verify proportions are roughly maintained
    // cluster should be largest (1/3)
    assert!(budget.cluster_budget > budget.divergence_budget);
    assert!(budget.cluster_budget > budget.single_space_budget);

    println!("[PASS] Large total creates valid budget without overflow\n");
}

// =============================================================================
// Constitution Compliance Verification
// =============================================================================

#[test]
fn fsv_constitution_compliance() {
    println!("\n=== FSV Constitution Compliance ===");

    let budget = TokenBudget::default();

    println!("DEFAULT_TOKEN_BUDGET verification:");
    println!("  Constitution specifies: Total=1200, Div=200, Cluster=400, Single=300, Session=200, Reserved=100");
    println!("  Actual values:");
    println!("    Total: {} (expected 1200)", budget.total);
    println!("    Divergence: {} (expected 200)", budget.divergence_budget);
    println!("    Cluster: {} (expected 400)", budget.cluster_budget);
    println!("    SingleSpace: {} (expected 300)", budget.single_space_budget);
    println!("    Session: {} (expected 200)", budget.session_budget);
    println!("    Reserved: {} (expected 100)", budget.reserved);

    assert_eq!(budget.total, 1200, "Total should be 1200 per constitution");
    assert_eq!(budget.divergence_budget, 200, "Divergence should be 200 per constitution");
    assert_eq!(budget.cluster_budget, 400, "Cluster should be 400 per constitution");
    assert_eq!(budget.single_space_budget, 300, "SingleSpace should be 300 per constitution");
    assert_eq!(budget.session_budget, 200, "Session should be 200 per constitution");
    assert_eq!(budget.reserved, 100, "Reserved should be 100 per constitution");

    println!("\nBRIEF_BUDGET verification:");
    println!("  Constitution specifies: 200 tokens for PreToolUse");
    println!("  Actual value: {}", BRIEF_BUDGET);
    assert_eq!(BRIEF_BUDGET, 200, "BRIEF_BUDGET should be 200 per constitution");

    println!("[PASS] All constitution values match\n");
}

// =============================================================================
// Budget-Category Integration
// =============================================================================

#[test]
fn fsv_budget_category_integration() {
    println!("\n=== FSV Budget-Category Integration ===");

    let budget = TokenBudget::default();

    println!("Verifying budget_for_category() returns correct values:");

    for cat in InjectionCategory::all() {
        let category_budget = budget.budget_for_category(cat);
        let expected = cat.token_budget();

        println!("  {}: budget_for_category={}, InjectionCategory::token_budget()={}",
            cat, category_budget, expected);

        assert_eq!(
            category_budget, expected,
            "budget_for_category({}) should match InjectionCategory::token_budget()", cat
        );
    }

    println!("[PASS] Budget-Category integration verified\n");
}

// =============================================================================
// Usable Budget Calculation
// =============================================================================

#[test]
fn fsv_usable_budget_calculation() {
    println!("\n=== FSV Usable Budget Calculation ===");

    let budget = TokenBudget::default();
    let usable = budget.usable();
    let expected = budget.total - budget.reserved;

    println!("Input: DEFAULT_TOKEN_BUDGET");
    println!("  total: {}", budget.total);
    println!("  reserved: {}", budget.reserved);
    println!("Output: usable() = {}", usable);
    println!("Expected: {} - {} = {}", budget.total, budget.reserved, expected);

    assert_eq!(usable, expected, "usable should be total - reserved");
    assert_eq!(usable, 1100, "usable should be 1100 for default budget");

    println!("[PASS] Usable budget calculation correct\n");
}

// =============================================================================
// Synthetic Data Test
// =============================================================================

#[test]
fn fsv_synthetic_data_pipeline() {
    println!("\n=== FSV Synthetic Data Pipeline Test ===");

    // Simulate a context injection budget tracking scenario
    let budget = TokenBudget::default();

    println!("Simulating injection budget tracking:");
    println!("Initial budget: {:?}", budget);

    // Synthetic candidates with token counts
    let synthetic_candidates = vec![
        ("Divergence alert about code changes", 45, InjectionCategory::DivergenceAlert),
        ("High relevance: Rust embedding patterns", 120, InjectionCategory::HighRelevanceCluster),
        ("Single space: Code similarity match", 80, InjectionCategory::SingleSpaceMatch),
        ("Recent session summary", 95, InjectionCategory::RecentSession),
    ];

    let mut spent: std::collections::HashMap<InjectionCategory, u32> = std::collections::HashMap::new();

    for (content, tokens, category) in &synthetic_candidates {
        let cat_budget = budget.budget_for_category(*category);
        let current_spent = *spent.get(category).unwrap_or(&0);
        let remaining = cat_budget.saturating_sub(current_spent);
        let fits = *tokens <= remaining;

        println!("\nCandidate: \"{}\"", content);
        println!("  tokens: {}, category: {}", tokens, category);
        println!("  category_budget: {}, spent: {}, remaining: {}", cat_budget, current_spent, remaining);
        println!("  fits: {}", fits);

        if fits {
            *spent.entry(*category).or_insert(0) += tokens;
        }
    }

    println!("\nFinal spending:");
    for cat in InjectionCategory::all() {
        let s = *spent.get(&cat).unwrap_or(&0);
        let b = budget.budget_for_category(cat);
        println!("  {}: spent {}/{}", cat, s, b);
        assert!(s <= b, "Should not exceed category budget");
    }

    println!("[PASS] Synthetic pipeline test passed\n");
}
