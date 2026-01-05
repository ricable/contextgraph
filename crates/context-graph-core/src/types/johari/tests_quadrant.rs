//! Tests for JohariQuadrant enum.

use super::*;

#[test]
fn test_is_self_aware() {
    assert!(
        JohariQuadrant::Open.is_self_aware(),
        "Open should be self-aware"
    );
    assert!(
        JohariQuadrant::Hidden.is_self_aware(),
        "Hidden should be self-aware"
    );
    assert!(
        !JohariQuadrant::Blind.is_self_aware(),
        "Blind should NOT be self-aware"
    );
    assert!(
        !JohariQuadrant::Unknown.is_self_aware(),
        "Unknown should NOT be self-aware"
    );
}

#[test]
fn test_is_other_aware() {
    assert!(
        JohariQuadrant::Open.is_other_aware(),
        "Open should be other-aware"
    );
    assert!(
        !JohariQuadrant::Hidden.is_other_aware(),
        "Hidden should NOT be other-aware"
    );
    assert!(
        JohariQuadrant::Blind.is_other_aware(),
        "Blind should be other-aware"
    );
    assert!(
        !JohariQuadrant::Unknown.is_other_aware(),
        "Unknown should NOT be other-aware"
    );
}

#[test]
fn test_default_retrieval_weight() {
    assert_eq!(JohariQuadrant::Open.default_retrieval_weight(), 1.0);
    assert_eq!(JohariQuadrant::Hidden.default_retrieval_weight(), 0.3);
    assert_eq!(JohariQuadrant::Blind.default_retrieval_weight(), 0.7);
    assert_eq!(JohariQuadrant::Unknown.default_retrieval_weight(), 0.5);
}

#[test]
fn test_retrieval_weights_in_valid_range() {
    for quadrant in JohariQuadrant::all() {
        let weight = quadrant.default_retrieval_weight();
        assert!(
            weight >= 0.0,
            "Weight {} for {:?} below 0.0",
            weight,
            quadrant
        );
        assert!(
            weight <= 1.0,
            "Weight {} for {:?} above 1.0",
            weight,
            quadrant
        );
    }
}

#[test]
fn test_include_in_default_context() {
    assert!(
        JohariQuadrant::Open.include_in_default_context(),
        "Open should be in default context"
    );
    assert!(
        !JohariQuadrant::Hidden.include_in_default_context(),
        "Hidden should NOT be in default context"
    );
    assert!(
        JohariQuadrant::Blind.include_in_default_context(),
        "Blind should be in default context"
    );
    assert!(
        JohariQuadrant::Unknown.include_in_default_context(),
        "Unknown should be in default context"
    );
}

#[test]
fn test_column_family() {
    assert_eq!(JohariQuadrant::Open.column_family(), "johari_open");
    assert_eq!(JohariQuadrant::Hidden.column_family(), "johari_hidden");
    assert_eq!(JohariQuadrant::Blind.column_family(), "johari_blind");
    assert_eq!(JohariQuadrant::Unknown.column_family(), "johari_unknown");
}

#[test]
fn test_all_variants() {
    let all = JohariQuadrant::all();
    assert_eq!(all.len(), 4);
    assert!(all.contains(&JohariQuadrant::Open));
    assert!(all.contains(&JohariQuadrant::Hidden));
    assert!(all.contains(&JohariQuadrant::Blind));
    assert!(all.contains(&JohariQuadrant::Unknown));
}

#[test]
fn test_default_is_open() {
    assert_eq!(JohariQuadrant::default(), JohariQuadrant::Open);
}

#[test]
fn test_display() {
    assert_eq!(format!("{}", JohariQuadrant::Open), "Open");
    assert_eq!(format!("{}", JohariQuadrant::Hidden), "Hidden");
    assert_eq!(format!("{}", JohariQuadrant::Blind), "Blind");
    assert_eq!(format!("{}", JohariQuadrant::Unknown), "Unknown");
}

#[test]
fn test_from_str_valid() {
    assert_eq!(
        "open".parse::<JohariQuadrant>().unwrap(),
        JohariQuadrant::Open
    );
    assert_eq!(
        "OPEN".parse::<JohariQuadrant>().unwrap(),
        JohariQuadrant::Open
    );
    assert_eq!(
        "Open".parse::<JohariQuadrant>().unwrap(),
        JohariQuadrant::Open
    );
    assert_eq!(
        "hidden".parse::<JohariQuadrant>().unwrap(),
        JohariQuadrant::Hidden
    );
    assert_eq!(
        "blind".parse::<JohariQuadrant>().unwrap(),
        JohariQuadrant::Blind
    );
    assert_eq!(
        "unknown".parse::<JohariQuadrant>().unwrap(),
        JohariQuadrant::Unknown
    );
}

#[test]
fn test_from_str_invalid() {
    let result = "invalid".parse::<JohariQuadrant>();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("Invalid JohariQuadrant"));
    assert!(err.contains("invalid"));
}

#[test]
fn test_serde_roundtrip() {
    for quadrant in JohariQuadrant::all() {
        let json = serde_json::to_string(&quadrant).expect("serialize failed");
        let parsed: JohariQuadrant = serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(quadrant, parsed, "Roundtrip failed for {:?}", quadrant);
    }
}

#[test]
fn test_serde_snake_case() {
    // Verify snake_case serialization per constitution.yaml requirement
    assert_eq!(
        serde_json::to_string(&JohariQuadrant::Open).unwrap(),
        "\"open\""
    );
    assert_eq!(
        serde_json::to_string(&JohariQuadrant::Hidden).unwrap(),
        "\"hidden\""
    );
    assert_eq!(
        serde_json::to_string(&JohariQuadrant::Blind).unwrap(),
        "\"blind\""
    );
    assert_eq!(
        serde_json::to_string(&JohariQuadrant::Unknown).unwrap(),
        "\"unknown\""
    );
}

#[test]
fn test_description_not_empty() {
    for quadrant in JohariQuadrant::all() {
        let desc = quadrant.description();
        assert!(!desc.is_empty(), "Description empty for {:?}", quadrant);
        assert!(desc.len() > 10, "Description too short for {:?}", quadrant);
    }
}

#[test]
fn test_clone_and_copy() {
    let original = JohariQuadrant::Open;
    let cloned = original;
    let copied = original;
    assert_eq!(original, cloned);
    assert_eq!(original, copied);
}

#[test]
fn test_hash_consistency() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    for quadrant in JohariQuadrant::all() {
        assert!(set.insert(quadrant), "Duplicate hash for {:?}", quadrant);
    }
    assert_eq!(set.len(), 4);
}

#[test]
fn test_valid_transitions_count() {
    assert_eq!(JohariQuadrant::Open.valid_transitions().len(), 1);
    assert_eq!(JohariQuadrant::Hidden.valid_transitions().len(), 1);
    assert_eq!(JohariQuadrant::Blind.valid_transitions().len(), 2);
    assert_eq!(JohariQuadrant::Unknown.valid_transitions().len(), 4);
}

#[test]
fn test_can_transition_to_false_for_self() {
    for quadrant in JohariQuadrant::all() {
        assert!(
            !quadrant.can_transition_to(quadrant),
            "{:?} should not be able to transition to itself",
            quadrant
        );
    }
}

#[test]
fn test_can_transition_to_valid_targets() {
    // Open can only go to Hidden
    assert!(JohariQuadrant::Open.can_transition_to(JohariQuadrant::Hidden));
    assert!(!JohariQuadrant::Open.can_transition_to(JohariQuadrant::Blind));
    assert!(!JohariQuadrant::Open.can_transition_to(JohariQuadrant::Unknown));

    // Hidden can only go to Open
    assert!(JohariQuadrant::Hidden.can_transition_to(JohariQuadrant::Open));
    assert!(!JohariQuadrant::Hidden.can_transition_to(JohariQuadrant::Blind));
    assert!(!JohariQuadrant::Hidden.can_transition_to(JohariQuadrant::Unknown));

    // Blind can go to Open or Hidden
    assert!(JohariQuadrant::Blind.can_transition_to(JohariQuadrant::Open));
    assert!(JohariQuadrant::Blind.can_transition_to(JohariQuadrant::Hidden));
    assert!(!JohariQuadrant::Blind.can_transition_to(JohariQuadrant::Unknown));

    // Unknown can go to Open, Hidden, or Blind
    assert!(JohariQuadrant::Unknown.can_transition_to(JohariQuadrant::Open));
    assert!(JohariQuadrant::Unknown.can_transition_to(JohariQuadrant::Hidden));
    assert!(JohariQuadrant::Unknown.can_transition_to(JohariQuadrant::Blind));
}

#[test]
fn test_boundary_minimum_transitions() {
    assert_eq!(JohariQuadrant::Open.valid_transitions().len(), 1);
    for target in JohariQuadrant::all() {
        let result = JohariQuadrant::Open.can_transition_to(target);
        if target == JohariQuadrant::Hidden {
            assert!(result, "Open should transition to Hidden");
        } else {
            assert!(!result, "Open should NOT transition to {:?}", target);
        }
    }
}
