//! GPU OOM detection and recovery tests.
//!
//! Constitution Reference:
//! - AP-001: Fail fast - no panics, proper error propagation
//! - perf.memory.gpu: <24GB budget tracking
//!
//! These tests verify:
//! 1. OOM detected via Result::Err(GraphError::GpuResourceAllocation), not panic
//! 2. Recovery works after deallocation
//! 3. Category isolation prevents cascading failures
//! 4. Error messages include useful context (requested size, available)

use context_graph_graph::error::GraphError;
use context_graph_graph::index::gpu_memory::{GpuMemoryConfig, GpuMemoryManager, MemoryCategory};

/// Test that GPU OOM is detected via Result::Err, not panic.
///
/// SYNTHETIC INPUT: 10KB budget, attempt 20KB allocation
/// EXPECTED: Allocation returns Err(GraphError::GpuResourceAllocation)
/// SOURCE OF TRUTH: manager.used() unchanged, result.is_err()
#[test]
#[ignore] // Chaos test - run with --ignored
fn test_gpu_oom_detection() {
    println!("\n=== CHAOS TEST: GPU OOM Detection ===");

    // SYNTHETIC INPUT: 10KB budget, attempt 20KB allocation
    let budget = 10 * 1024; // 10KB
    let manager = GpuMemoryManager::new(GpuMemoryConfig::with_budget(budget))
        .expect("Manager creation should succeed");

    // BEFORE state
    println!(
        "BEFORE: budget={}, used={}, available={}",
        manager.budget(),
        manager.used(),
        manager.available()
    );

    // SOURCE OF TRUTH: manager.used() and allocation result
    // EXPECTED: Allocation of 20KB should fail with GpuResourceAllocation error
    let result = manager.allocate(20 * 1024, MemoryCategory::WorkingMemory);

    // AFTER state
    println!(
        "AFTER: used={}, available={}, result_is_err={}",
        manager.used(),
        manager.available(),
        result.is_err()
    );

    // VERIFY: Error type, not panic
    match result {
        Err(GraphError::GpuResourceAllocation(msg)) => {
            println!("SUCCESS: Got expected GpuResourceAllocation error");
            println!("Error message: {}", msg);
            assert!(
                msg.contains("exceed") || msg.contains("20480") || msg.contains("bytes"),
                "Message should mention exceeding budget or size, got: {}",
                msg
            );
        }
        Err(other) => panic!("Wrong error type: {:?}", other),
        Ok(_) => panic!("Should have failed but got Ok"),
    }

    // VERIFY: State unchanged after failed allocation
    assert_eq!(
        manager.used(),
        0,
        "Failed allocation should not change used memory"
    );

    println!("=== PASSED: test_gpu_oom_detection ===\n");
}

/// Test OOM recovery after deallocation.
///
/// SYNTHETIC INPUT: 1MB budget, allocate 90%, fail at 110%, free, retry
/// EXPECTED: Retry succeeds after freeing memory
/// SOURCE OF TRUTH: manager.used() == 0 after drop, retry succeeds
#[test]
#[ignore]
fn test_gpu_oom_recovery_with_deallocation() {
    println!("\n=== CHAOS TEST: OOM Recovery After Deallocation ===");

    // SYNTHETIC INPUT: 1MB budget
    let budget = 1024 * 1024;
    let manager =
        GpuMemoryManager::new(GpuMemoryConfig::with_budget(budget)).expect("Manager creation");

    // STEP 1: Allocate 90% of budget
    println!("BEFORE: Allocating 90% of {} bytes", budget);
    let h1 = manager
        .allocate(900 * 1024, MemoryCategory::WorkingMemory)
        .expect("First allocation should succeed");
    println!(
        "AFTER FIRST: used={}, available={}",
        manager.used(),
        manager.available()
    );
    assert_eq!(manager.used(), 900 * 1024, "Should have allocated 900KB");

    // STEP 2: Attempt allocation that would exceed budget
    let result = manager.allocate(200 * 1024, MemoryCategory::WorkingMemory);
    println!("OVER-ALLOCATION: result_is_err={}", result.is_err());
    assert!(result.is_err(), "Should fail when budget exceeded");

    // STEP 3: Drop first handle to free memory
    drop(h1);
    println!(
        "AFTER DROP: used={}, available={}",
        manager.used(),
        manager.available()
    );

    // SOURCE OF TRUTH: manager.used() should be 0 after drop
    assert_eq!(manager.used(), 0, "Memory should be freed after drop");

    // STEP 4: Retry allocation - should succeed now
    let h2 = manager.allocate(200 * 1024, MemoryCategory::WorkingMemory);
    assert!(h2.is_ok(), "Allocation should succeed after recovery");
    println!("RECOVERY: Allocation succeeded after freeing memory");

    drop(h2);
    println!("=== PASSED: test_gpu_oom_recovery_with_deallocation ===\n");
}

/// Test that exhausting one category doesn't cascade to others.
///
/// SYNTHETIC INPUT: 10MB total, 2MB FaissIndex limit, 2MB WorkingMemory limit
/// EXPECTED: FaissIndex exhausted, WorkingMemory still works
/// SOURCE OF TRUTH: stats.category_usage shows isolation
#[test]
#[ignore]
fn test_gpu_cascading_oom_prevention() {
    println!("\n=== CHAOS TEST: Cascading OOM Prevention ===");

    // SYNTHETIC INPUT: Set per-category budgets
    let config = GpuMemoryConfig::with_budget(10 * 1024 * 1024) // 10MB total
        .category_budget(MemoryCategory::FaissIndex, 2 * 1024 * 1024) // 2MB
        .category_budget(MemoryCategory::WorkingMemory, 2 * 1024 * 1024); // 2MB

    let manager = GpuMemoryManager::new(config).expect("Manager creation");

    // STEP 1: Exhaust FaissIndex category
    println!("BEFORE: Exhausting FaissIndex category");
    let h1 = manager
        .allocate(2 * 1024 * 1024, MemoryCategory::FaissIndex)
        .expect("FaissIndex allocation");

    // VERIFY: FaissIndex exhausted
    let result = manager.allocate(1, MemoryCategory::FaissIndex);
    assert!(result.is_err(), "FaissIndex should be exhausted");
    if let Err(GraphError::GpuResourceAllocation(msg)) = &result {
        println!("FaissIndex exhausted: {}", msg);
    }

    // STEP 2: WorkingMemory should still work
    println!("AFTER FaissIndex exhaustion: Testing WorkingMemory");
    let h2 = manager.allocate(1024 * 1024, MemoryCategory::WorkingMemory);
    assert!(h2.is_ok(), "WorkingMemory should still work");
    println!("WorkingMemory allocation succeeded: {}", h2.is_ok());

    // SOURCE OF TRUTH: Stats should show category isolation
    let stats = manager.stats();
    println!(
        "STATS: total_allocated={}, category_usage={:?}",
        stats.total_allocated, stats.category_usage
    );

    // Verify category isolation
    let faiss_usage = stats
        .category_usage
        .get(&MemoryCategory::FaissIndex)
        .copied()
        .unwrap_or(0);
    let working_usage = stats
        .category_usage
        .get(&MemoryCategory::WorkingMemory)
        .copied()
        .unwrap_or(0);

    assert_eq!(faiss_usage, 2 * 1024 * 1024, "FaissIndex should be 2MB");
    assert_eq!(
        working_usage,
        1024 * 1024,
        "WorkingMemory should be 1MB"
    );

    drop(h1);
    drop(h2);
    println!("=== PASSED: test_gpu_cascading_oom_prevention ===\n");
}

/// Test that OOM error messages include useful context.
///
/// SYNTHETIC INPUT: 100 byte budget, request 1000 bytes
/// EXPECTED: Error message includes requested size or available info
/// SOURCE OF TRUTH: Error message content
#[test]
#[ignore]
fn test_gpu_oom_error_propagation() {
    println!("\n=== CHAOS TEST: OOM Error Propagation ===");

    // SYNTHETIC INPUT: Tiny budget
    let manager =
        GpuMemoryManager::new(GpuMemoryConfig::with_budget(100)).expect("Manager creation");

    println!(
        "BEFORE: budget={}, requesting 1000 bytes",
        manager.budget()
    );

    // TRIGGER: Request more than available
    let result = manager.allocate(1000, MemoryCategory::Other);

    // VERIFY: Error type and message content
    match result {
        Err(GraphError::GpuResourceAllocation(msg)) => {
            println!("ERROR MESSAGE: {}", msg);
            // Check error includes useful context
            // The message should contain info about the allocation or budget
            let has_context = msg.contains("1000")
                || msg.contains("100")
                || msg.contains("bytes")
                || msg.contains("exceed")
                || msg.contains("budget");
            assert!(
                has_context,
                "Error should include allocation context, got: {}",
                msg
            );
            println!("Error message includes allocation context: OK");
        }
        Err(e) => panic!("Wrong error type: {:?}", e),
        Ok(_) => panic!("Should have failed"),
    }

    println!("=== PASSED: test_gpu_oom_error_propagation ===\n");
}
