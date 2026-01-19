//! Full State Verification (FSV) tests for TokenBudgetManager.
//!
//! These tests verify the TokenBudgetManager implementation using synthetic data
//! with known expected outputs. They follow the verification protocol:
//! 1. Define expected output before running
//! 2. Execute with synthetic input
//! 3. Verify actual output matches expected
//! 4. Log evidence of success

use context_graph_core::injection::{
    estimate_tokens, InjectionCandidate, InjectionCategory, PriorityRanker,
    TokenBudget, TokenBudgetManager,
};
use context_graph_core::teleological::Embedder;
use chrono::Utc;
use uuid::Uuid;

/// Create test candidate with known token count
fn make_test_candidate(
    id: &str,
    tokens: u32,
    category: InjectionCategory,
    relevance: f32,
    weighted_agreement: f32,
) -> InjectionCandidate {
    // Generate content that will have approximately the desired token count
    let word_count = ((tokens as f32) / 1.3).ceil() as usize;
    let content = format!("[{}] {}", id, "word ".repeat(word_count));

    let mut c = InjectionCandidate::new(
        Uuid::new_v4(),
        content,
        relevance,
        weighted_agreement,
        vec![Embedder::Semantic, Embedder::Code],
        category,
        Utc::now(),
    );
    // Override token count to exact value for deterministic testing
    c.token_count = tokens;
    c
}

// =============================================================================
// FSV Test 1: Synthetic Selection - Happy Path
// =============================================================================

#[test]
fn fsv_happy_path_selection() {
    println!("\n=== FSV TEST 1: Happy Path Selection ===");
    println!("SYNTHETIC INPUT:");
    println!("  Budget: total=1200, divergence=200, cluster=400, single=300, session=200, reserved=100");
    println!("  Candidates:");
    println!("    - C1: DivergenceAlert, 150 tokens");
    println!("    - C2: HighRelevanceCluster, 200 tokens");
    println!("    - C3: HighRelevanceCluster, 150 tokens");
    println!("    - C4: SingleSpaceMatch, 100 tokens");
    println!("    - C5: RecentSession, 100 tokens");

    let budget = TokenBudget::default();

    let candidates = vec![
        make_test_candidate("C1", 150, InjectionCategory::DivergenceAlert, 0.9, 0.5),
        make_test_candidate("C2", 200, InjectionCategory::HighRelevanceCluster, 0.85, 3.0),
        make_test_candidate("C3", 150, InjectionCategory::HighRelevanceCluster, 0.80, 3.0),
        make_test_candidate("C4", 100, InjectionCategory::SingleSpaceMatch, 0.75, 2.0),
        make_test_candidate("C5", 100, InjectionCategory::RecentSession, 0.70, 1.5),
    ];

    println!("\nEXPECTED OUTPUT:");
    println!("  All 5 candidates selected (total 700 tokens < 1200)");
    println!("  Each candidate within its category budget");

    let (_selected, stats) = TokenBudgetManager::select_with_stats(&candidates, &budget);

    println!("\nACTUAL OUTPUT:");
    println!("  selected_count: {}", stats.selected_count);
    println!("  rejected_count: {}", stats.rejected_count);
    println!("  tokens_used: {}", stats.tokens_used);
    println!("  tokens_available: {}", stats.tokens_available);
    println!("  by_category:");
    for (cat, tokens) in &stats.by_category {
        println!("    {:?}: {} tokens", cat, tokens);
    }

    // Verify expectations
    assert_eq!(
        stats.selected_count, 5,
        "All 5 candidates should be selected"
    );
    assert_eq!(stats.rejected_count, 0, "No candidates should be rejected");
    assert_eq!(stats.tokens_used, 700, "Total should be 700 tokens");
    assert_eq!(stats.tokens_available, 500, "Remaining should be 500");

    // Verify category spending
    assert_eq!(
        *stats
            .by_category
            .get(&InjectionCategory::DivergenceAlert)
            .unwrap(),
        150,
        "DivergenceAlert should use 150"
    );
    assert_eq!(
        *stats
            .by_category
            .get(&InjectionCategory::HighRelevanceCluster)
            .unwrap(),
        350,
        "HighRelevanceCluster should use 350"
    );
    assert_eq!(
        *stats
            .by_category
            .get(&InjectionCategory::SingleSpaceMatch)
            .unwrap(),
        100,
        "SingleSpaceMatch should use 100"
    );
    assert_eq!(
        *stats
            .by_category
            .get(&InjectionCategory::RecentSession)
            .unwrap(),
        100,
        "RecentSession should use 100"
    );

    println!("\nVERIFICATION: PASSED");
    println!("  Evidence: All 5 candidates selected, total 700 tokens used");
    println!("  Category spending matches expected values");
}

// =============================================================================
// FSV Test 2: High-Priority Overflow Allowed
// =============================================================================

#[test]
fn fsv_high_priority_overflow() {
    println!("\n=== FSV TEST 2: High-Priority Overflow ===");
    println!("SYNTHETIC INPUT:");
    println!("  Budget: total=500, divergence=100, cluster=200, single=100, session=50, reserved=50");
    println!("  Candidates:");
    println!("    - C1: DivergenceAlert, 150 tokens (EXCEEDS divergence_budget of 100)");
    println!("    - C2: DivergenceAlert, 100 tokens (WOULD put total at 250)");

    let budget = TokenBudget {
        total: 500,
        divergence_budget: 100,
        cluster_budget: 200,
        single_space_budget: 100,
        session_budget: 50,
        reserved: 50,
    };

    let candidates = vec![
        make_test_candidate("C1", 150, InjectionCategory::DivergenceAlert, 0.9, 0.5),
        make_test_candidate("C2", 100, InjectionCategory::DivergenceAlert, 0.85, 0.5),
    ];

    println!("\nEXPECTED OUTPUT:");
    println!("  Both candidates selected (overflow allowed for DivergenceAlert, priority=1)");
    println!("  Total tokens: 250 (within total budget 500)");

    let (_selected, stats) = TokenBudgetManager::select_with_stats(&candidates, &budget);

    println!("\nACTUAL OUTPUT:");
    println!("  selected_count: {}", stats.selected_count);
    println!("  tokens_used: {}", stats.tokens_used);
    println!("  DivergenceAlert spending: {:?}", stats.by_category.get(&InjectionCategory::DivergenceAlert));

    assert_eq!(
        stats.selected_count, 2,
        "Both high-priority candidates should be selected (overflow allowed)"
    );
    assert_eq!(stats.tokens_used, 250, "Total should be 250 tokens");
    assert_eq!(
        *stats
            .by_category
            .get(&InjectionCategory::DivergenceAlert)
            .unwrap(),
        250,
        "DivergenceAlert should use 250 (overflow)"
    );

    println!("\nVERIFICATION: PASSED");
    println!("  Evidence: High-priority overflow allowed, 250 tokens > 100 budget");
}

// =============================================================================
// FSV Test 3: Low-Priority Overflow Blocked
// =============================================================================

#[test]
fn fsv_low_priority_no_overflow() {
    println!("\n=== FSV TEST 3: Low-Priority Overflow Blocked ===");
    println!("SYNTHETIC INPUT:");
    println!("  Budget: total=500, divergence=100, cluster=100, single=100, session=100, reserved=100");
    println!("  Candidates:");
    println!("    - C1: RecentSession, 150 tokens (EXCEEDS session_budget of 100)");
    println!("    - C2: SingleSpaceMatch, 150 tokens (EXCEEDS single_space_budget of 100)");

    let budget = TokenBudget {
        total: 500,
        divergence_budget: 100,
        cluster_budget: 100,
        single_space_budget: 100,
        session_budget: 100,
        reserved: 100,
    };

    let candidates = vec![
        make_test_candidate("C1", 150, InjectionCategory::RecentSession, 0.7, 1.5),
        make_test_candidate("C2", 150, InjectionCategory::SingleSpaceMatch, 0.75, 2.0),
    ];

    println!("\nEXPECTED OUTPUT:");
    println!("  NO candidates selected (low priority cannot overflow)");
    println!("  RecentSession priority=4, SingleSpaceMatch priority=3");

    let (_selected, stats) = TokenBudgetManager::select_with_stats(&candidates, &budget);

    println!("\nACTUAL OUTPUT:");
    println!("  selected_count: {}", stats.selected_count);
    println!("  rejected_count: {}", stats.rejected_count);
    println!("  tokens_used: {}", stats.tokens_used);

    assert_eq!(
        stats.selected_count, 0,
        "No candidates should be selected (overflow not allowed)"
    );
    assert_eq!(stats.rejected_count, 2, "Both should be rejected");
    assert_eq!(stats.tokens_used, 0, "No tokens used");

    println!("\nVERIFICATION: PASSED");
    println!("  Evidence: Low-priority candidates cannot overflow their budgets");
}

// =============================================================================
// FSV Test 4: Total Budget Hard Limit
// =============================================================================

#[test]
fn fsv_total_budget_hard_limit() {
    println!("\n=== FSV TEST 4: Total Budget Hard Limit ===");
    println!("SYNTHETIC INPUT:");
    println!("  Budget: total=300, divergence=300, cluster=300, single=300, session=300, reserved=0");
    println!("    (Category budgets exceed total - tests total budget enforcement)");
    println!("  Candidates:");
    println!("    - C1: DivergenceAlert, 150 tokens");
    println!("    - C2: DivergenceAlert, 100 tokens");
    println!("    - C3: DivergenceAlert, 100 tokens (WOULD exceed total)");

    let budget = TokenBudget {
        total: 300,
        divergence_budget: 300, // Artificially high to test total budget
        cluster_budget: 300,
        single_space_budget: 300,
        session_budget: 300,
        reserved: 0,
    };

    let candidates = vec![
        make_test_candidate("C1", 150, InjectionCategory::DivergenceAlert, 0.9, 0.5),
        make_test_candidate("C2", 100, InjectionCategory::DivergenceAlert, 0.85, 0.5),
        make_test_candidate("C3", 100, InjectionCategory::DivergenceAlert, 0.80, 0.5),
    ];

    println!("\nEXPECTED OUTPUT:");
    println!("  C1 and C2 selected (250 tokens)");
    println!("  C3 rejected (would make 350 > 300 total)");

    let (_selected, stats) = TokenBudgetManager::select_with_stats(&candidates, &budget);

    println!("\nACTUAL OUTPUT:");
    println!("  selected_count: {}", stats.selected_count);
    println!("  rejected_count: {}", stats.rejected_count);
    println!("  tokens_used: {}", stats.tokens_used);
    println!("  tokens_available: {}", stats.tokens_available);

    assert_eq!(stats.selected_count, 2, "Only 2 candidates should fit");
    assert_eq!(stats.rejected_count, 1, "1 candidate rejected");
    assert_eq!(stats.tokens_used, 250, "250 tokens used");
    assert_eq!(stats.tokens_available, 50, "50 remaining");

    // CRITICAL: Verify total never exceeded
    assert!(
        stats.tokens_used <= budget.total,
        "INVARIANT VIOLATED: tokens_used {} > budget.total {}",
        stats.tokens_used,
        budget.total
    );

    println!("\nVERIFICATION: PASSED");
    println!("  Evidence: Total budget hard limit enforced (250 < 300)");
}

// =============================================================================
// FSV Test 5: estimate_tokens Function
// =============================================================================

#[test]
fn fsv_estimate_tokens() {
    println!("\n=== FSV TEST 5: estimate_tokens Function ===");

    let test_cases = vec![
        ("", 0, "Empty string"),
        ("hello", 2, "Single word (1 × 1.3 → ceil 2)"),
        ("one two three", 4, "Three words (3 × 1.3 = 3.9 → ceil 4)"),
        ("a b c d e f g h i j", 13, "Ten words (10 × 1.3 → 13)"),
        ("word word word word word word word word word word word word word word word word word word word word", 26, "Twenty words (20 × 1.3 → 26)"),
    ];

    println!("SYNTHETIC INPUT & EXPECTED:");
    for (input, expected, desc) in &test_cases {
        println!("  '{}' -> {} tokens ({})",
            if input.len() > 30 { &input[..30] } else { input },
            expected, desc);
    }

    println!("\nACTUAL OUTPUT:");
    for (input, expected, desc) in &test_cases {
        let actual = estimate_tokens(input);
        println!("  '{}' -> {} tokens (expected {})",
            if input.len() > 30 { &input[..30] } else { input },
            actual, expected);
        assert_eq!(
            actual, *expected,
            "estimate_tokens failed for '{}': got {}, expected {}",
            desc, actual, expected
        );
    }

    println!("\nVERIFICATION: PASSED");
    println!("  Evidence: All estimate_tokens calculations match formula (words × 1.3 → ceil)");
}

// =============================================================================
// FSV Test 6: Category Priority Order Enforcement
// =============================================================================

#[test]
fn fsv_category_priority_order() {
    println!("\n=== FSV TEST 6: Category Priority Order ===");
    println!("SYNTHETIC INPUT:");
    println!("  Budget: default (1200 total)");
    println!("  Candidates (INTENTIONALLY UNSORTED):");
    println!("    - C1: RecentSession (priority=4), 50 tokens");
    println!("    - C2: DivergenceAlert (priority=1), 50 tokens");
    println!("    - C3: SingleSpaceMatch (priority=3), 50 tokens");
    println!("    - C4: HighRelevanceCluster (priority=2), 50 tokens");

    // budget is used only for documentation
    let _budget = TokenBudget::default();

    // Intentionally create unsorted candidates
    let mut candidates = vec![
        make_test_candidate("C1", 50, InjectionCategory::RecentSession, 0.95, 1.5),
        make_test_candidate("C2", 50, InjectionCategory::DivergenceAlert, 0.60, 0.5),
        make_test_candidate("C3", 50, InjectionCategory::SingleSpaceMatch, 0.80, 2.0),
        make_test_candidate("C4", 50, InjectionCategory::HighRelevanceCluster, 0.70, 3.0),
    ];

    println!("\nEXPECTED OUTPUT:");
    println!("  After sorting (by PriorityRanker): C2, C4, C3, C1");
    println!("  Order: DivergenceAlert(1) -> HighRelevanceCluster(2) -> SingleSpaceMatch(3) -> RecentSession(4)");

    // Simulate what the full pipeline does
    PriorityRanker::rank_candidates(&mut candidates);

    println!("\nACTUAL OUTPUT (after PriorityRanker::rank_candidates):");
    for (i, c) in candidates.iter().enumerate() {
        println!("  [{}] {:?} (priority={})", i, c.category, c.category.priority());
    }

    assert_eq!(
        candidates[0].category,
        InjectionCategory::DivergenceAlert,
        "First should be DivergenceAlert"
    );
    assert_eq!(
        candidates[1].category,
        InjectionCategory::HighRelevanceCluster,
        "Second should be HighRelevanceCluster"
    );
    assert_eq!(
        candidates[2].category,
        InjectionCategory::SingleSpaceMatch,
        "Third should be SingleSpaceMatch"
    );
    assert_eq!(
        candidates[3].category,
        InjectionCategory::RecentSession,
        "Fourth should be RecentSession"
    );

    println!("\nVERIFICATION: PASSED");
    println!("  Evidence: Categories sorted by priority (1 -> 2 -> 3 -> 4)");
}

// =============================================================================
// FSV Test 7: Edge Case - Zero Token Candidate
// =============================================================================

#[test]
fn fsv_zero_token_candidate() {
    println!("\n=== FSV TEST 7: Zero Token Candidate ===");
    println!("SYNTHETIC INPUT:");
    println!("  Budget: default (1200 total)");
    println!("  Candidate: DivergenceAlert with 0 tokens");

    let budget = TokenBudget::default();

    let mut candidate = make_test_candidate("C1", 0, InjectionCategory::DivergenceAlert, 0.9, 0.5);
    candidate.token_count = 0; // Force zero tokens

    let candidates = vec![candidate];

    println!("\nEXPECTED OUTPUT:");
    println!("  Candidate selected (0 tokens fits in any budget)");

    let (_selected, stats) = TokenBudgetManager::select_with_stats(&candidates, &budget);

    println!("\nACTUAL OUTPUT:");
    println!("  selected_count: {}", stats.selected_count);
    println!("  tokens_used: {}", stats.tokens_used);

    assert_eq!(
        stats.selected_count, 1,
        "Zero-token candidate should be selected"
    );
    assert_eq!(stats.tokens_used, 0, "Zero tokens used");

    println!("\nVERIFICATION: PASSED");
    println!("  Evidence: Zero-token candidates handled correctly");
}

// =============================================================================
// FSV Test 8: Comprehensive Integration
// =============================================================================

#[test]
fn fsv_comprehensive_integration() {
    println!("\n=== FSV TEST 8: Comprehensive Integration ===");
    println!("This test simulates a real injection pipeline scenario.");

    let budget = TokenBudget::default();

    println!("SYNTHETIC INPUT:");
    println!("  Budget: {:?}", budget);
    println!("  Candidates (simulating post-retrieval candidates):");

    // Simulate realistic candidate distribution
    let mut candidates = vec![
        // Divergence alerts (should be processed first)
        make_test_candidate("DIV1", 100, InjectionCategory::DivergenceAlert, 0.9, 0.5),
        make_test_candidate("DIV2", 80, InjectionCategory::DivergenceAlert, 0.85, 0.5),
        // High relevance clusters (processed second)
        make_test_candidate("CLU1", 150, InjectionCategory::HighRelevanceCluster, 0.88, 4.0),
        make_test_candidate("CLU2", 120, InjectionCategory::HighRelevanceCluster, 0.82, 3.5),
        make_test_candidate("CLU3", 100, InjectionCategory::HighRelevanceCluster, 0.75, 3.0),
        // Single space matches (processed third)
        make_test_candidate("SNG1", 100, InjectionCategory::SingleSpaceMatch, 0.7, 2.0),
        make_test_candidate("SNG2", 80, InjectionCategory::SingleSpaceMatch, 0.65, 1.5),
        make_test_candidate("SNG3", 150, InjectionCategory::SingleSpaceMatch, 0.60, 1.2),
        // Recent session (processed last)
        make_test_candidate("SES1", 100, InjectionCategory::RecentSession, 0.5, 1.0),
    ];

    for c in &candidates {
        println!("    - {:?}, {} tokens", c.category, c.token_count);
    }

    // Step 1: Rank candidates (simulate PriorityRanker)
    PriorityRanker::rank_candidates(&mut candidates);

    println!("\nAfter ranking:");
    for (i, c) in candidates.iter().enumerate() {
        println!(
            "  [{}] {:?}, {} tokens, priority={}",
            i, c.category, c.token_count, c.priority
        );
    }

    // Step 2: Select within budget
    let (selected, stats) = TokenBudgetManager::select_with_stats(&candidates, &budget);

    println!("\nSELECTION RESULT:");
    println!("  Selected: {} / Rejected: {}", stats.selected_count, stats.rejected_count);
    println!("  Tokens: {} used / {} available", stats.tokens_used, stats.tokens_available);
    println!("  By category:");
    for (cat, tokens) in &stats.by_category {
        println!("    {:?}: {} tokens", cat, tokens);
    }

    println!("\nSELECTED CANDIDATES:");
    for c in &selected {
        println!("  - {:?}, {} tokens", c.category, c.token_count);
    }

    // Verify invariants
    assert!(
        stats.tokens_used <= budget.total,
        "INVARIANT: Total budget must not be exceeded"
    );
    assert!(
        *stats
            .by_category
            .get(&InjectionCategory::SingleSpaceMatch)
            .unwrap()
            <= budget.single_space_budget,
        "INVARIANT: SingleSpaceMatch cannot overflow"
    );
    assert!(
        *stats
            .by_category
            .get(&InjectionCategory::RecentSession)
            .unwrap()
            <= budget.session_budget,
        "INVARIANT: RecentSession cannot overflow"
    );

    // Verify category ordering in selected candidates
    let mut last_priority = 0;
    for c in &selected {
        assert!(
            c.category.priority() >= last_priority,
            "INVARIANT: Selected candidates must be in priority order"
        );
        last_priority = c.category.priority();
    }

    println!("\nVERIFICATION: PASSED");
    println!("  Evidence: All invariants verified:");
    println!("    - Total budget not exceeded");
    println!("    - Low-priority categories within limits");
    println!("    - Category priority order maintained");
}
