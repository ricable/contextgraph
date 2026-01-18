//! Tests for search/weight_profiles handler.
//!
//! Tests that weight profiles endpoint returns all profiles with 13 weights each.

use crate::protocol::JsonRpcId;
use crate::weights::{get_profile_names, WEIGHT_PROFILES};
use context_graph_core::types::fingerprint::NUM_EMBEDDERS;

use crate::handlers::tests::{create_test_handlers, make_request};

/// Test search/weight_profiles returns all profiles.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_weight_profiles_returns_all() {
    let handlers = create_test_handlers();

    let request = make_request("search/weight_profiles", Some(JsonRpcId::Number(1)), None);
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "search/weight_profiles should succeed"
    );
    let result = response.result.expect("Should have result");

    // Verify profiles array
    let profiles = result.get("profiles").and_then(|v| v.as_array());
    assert!(profiles.is_some(), "Should have profiles array");

    let profiles = profiles.unwrap();
    assert_eq!(
        profiles.len(),
        WEIGHT_PROFILES.len(),
        "Should return all weight profiles"
    );

    // Verify each profile has required fields
    for profile in profiles {
        assert!(profile.get("name").is_some(), "Profile must have name");
        assert!(
            profile.get("weights").is_some(),
            "Profile must have weights"
        );
        assert!(
            profile.get("primary_spaces").is_some(),
            "Profile must have primary_spaces"
        );
        assert!(
            profile.get("description").is_some(),
            "Profile must have description"
        );

        // Verify weights array has 13 elements
        let weights = profile.get("weights").and_then(|v| v.as_array());
        assert!(weights.is_some(), "Weights must be array");
        assert_eq!(
            weights.unwrap().len(),
            NUM_EMBEDDERS,
            "Each profile must have exactly 13 weights"
        );
    }
}

/// Test search/weight_profiles returns correct embedding spaces.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_weight_profiles_embedding_spaces() {
    let handlers = create_test_handlers();

    let request = make_request("search/weight_profiles", Some(JsonRpcId::Number(1)), None);
    let response = handlers.dispatch(request).await;

    let result = response.result.expect("Should have result");

    // Verify embedding_spaces
    let spaces = result.get("embedding_spaces").and_then(|v| v.as_array());
    assert!(spaces.is_some(), "Should have embedding_spaces array");
    assert_eq!(
        spaces.unwrap().len(),
        NUM_EMBEDDERS,
        "Should have exactly 13 embedding spaces"
    );

    // Verify total_spaces
    assert_eq!(
        result.get("total_spaces").and_then(|v| v.as_u64()),
        Some(NUM_EMBEDDERS as u64),
        "total_spaces should be 13"
    );

    // Verify default values
    assert_eq!(
        result.get("default_aggregation").and_then(|v| v.as_str()),
        Some("rrf"),
        "default_aggregation should be rrf"
    );
    assert_eq!(
        result.get("default_rrf_k").and_then(|v| v.as_f64()),
        Some(60.0),
        "default_rrf_k should be 60.0"
    );
}

/// Test search/weight_profiles includes expected profile names.
#[tokio::test]
#[ignore = "Uses removed PRD v6 API - use tools/call"]
async fn test_weight_profiles_expected_names() {
    let handlers = create_test_handlers();

    let request = make_request("search/weight_profiles", Some(JsonRpcId::Number(1)), None);
    let response = handlers.dispatch(request).await;

    let result = response.result.expect("Should have result");
    let profiles = result
        .get("profiles")
        .and_then(|v| v.as_array())
        .expect("Should have profiles");

    let profile_names: Vec<&str> = profiles
        .iter()
        .filter_map(|p| p.get("name").and_then(|v| v.as_str()))
        .collect();

    // Verify expected profiles from get_profile_names()
    let expected_names = get_profile_names();
    for expected in expected_names {
        assert!(
            profile_names.contains(&expected),
            "Should contain profile: {}",
            expected
        );
    }
}
