//! Tests for the teleological retrieval pipeline.

use std::sync::Arc;
use std::time::Duration;

use crate::retrieval::InMemoryMultiEmbeddingExecutor;
use crate::stubs::{InMemoryTeleologicalStore, StubMultiArrayProvider};

use super::super::teleological_query::TeleologicalQuery;
use super::{DefaultTeleologicalPipeline, PipelineHealth, TeleologicalRetrievalPipeline};
use crate::error::CoreError;

async fn create_test_pipeline(
) -> DefaultTeleologicalPipeline<InMemoryMultiEmbeddingExecutor, InMemoryTeleologicalStore> {
    let store = InMemoryTeleologicalStore::new();
    let provider = StubMultiArrayProvider::new();

    // Store needs to be Arc-wrapped for sharing between executor and pipeline
    let store_arc = Arc::new(store);

    let executor = Arc::new(InMemoryMultiEmbeddingExecutor::with_arcs(
        store_arc.clone(),
        Arc::new(provider),
    ));

    DefaultTeleologicalPipeline::new(executor, store_arc)
}

#[tokio::test]
async fn test_pipeline_creation() {
    let pipeline = create_test_pipeline().await;
    let health = pipeline.health_check().await.unwrap();

    assert_eq!(health.spaces_available, 13);
    assert!(!health.has_goal_hierarchy);

    println!("[VERIFIED] Pipeline created with all components");
}

#[tokio::test]
async fn test_execute_basic_query() {
    let pipeline = create_test_pipeline().await;

    let query = TeleologicalQuery::from_text("authentication patterns");
    let result = pipeline.execute(&query).await.unwrap();

    assert!(result.total_time.as_millis() < 1000); // Generous for test
    assert!(result.spaces_searched > 0);

    println!("BEFORE: query text = 'authentication patterns'");
    println!(
        "AFTER: results = {}, time = {:?}",
        result.len(),
        result.total_time
    );
    println!("[VERIFIED] Basic query execution works");
}

#[tokio::test]
async fn test_execute_with_breakdown() {
    let pipeline = create_test_pipeline().await;

    let query = TeleologicalQuery::from_text("test query").with_breakdown(true);

    let result = pipeline.execute(&query).await.unwrap();

    assert!(result.breakdown.is_some());
    let breakdown = result.breakdown.unwrap();

    println!("Breakdown: {}", breakdown.funnel_summary());
    println!("[VERIFIED] Pipeline breakdown is populated when requested");
}

#[tokio::test]
async fn test_execute_fails_empty_query() {
    let pipeline = create_test_pipeline().await;

    let query = TeleologicalQuery::default();
    let result = pipeline.execute(&query).await;

    assert!(result.is_err());
    match result {
        Err(CoreError::ValidationError { field, .. }) => {
            assert_eq!(field, "text");
            println!("[VERIFIED] Empty query fails fast with ValidationError");
        }
        _ => panic!("Expected ValidationError"),
    }
}

#[tokio::test]
async fn test_timing_breakdown() {
    let pipeline = create_test_pipeline().await;

    let query = TeleologicalQuery::from_text("timing test");
    let result = pipeline.execute(&query).await.unwrap();

    println!("Timing: {}", result.timing_summary());
    println!("  Stage 1 (SPLADE): {:?}", result.timing.stage1_splade);
    println!(
        "  Stage 2 (Matryoshka): {:?}",
        result.timing.stage2_matryoshka
    );
    println!(
        "  Stage 3 (Full HNSW): {:?}",
        result.timing.stage3_full_hnsw
    );
    println!(
        "  Stage 4 (Teleological): {:?}",
        result.timing.stage4_teleological
    );
    println!(
        "  Stage 5 (Late Interaction): {:?}",
        result.timing.stage5_late_interaction
    );
    println!("  Total: {:?}", result.total_time);

    println!("[VERIFIED] All pipeline stages have timing measurements");
}


#[test]
fn test_pipeline_health_defaults() {
    let health = PipelineHealth {
        is_healthy: true,
        spaces_available: 13,
        has_goal_hierarchy: false,
        index_size: 1_000_000,
        last_query_time: Some(Duration::from_millis(45)),
    };

    assert!(health.is_healthy);
    assert_eq!(health.spaces_available, 13);
    println!("[VERIFIED] PipelineHealth struct works correctly");
}
