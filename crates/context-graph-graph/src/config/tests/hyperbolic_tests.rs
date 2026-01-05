//! Tests for HyperbolicConfig.

use crate::config::HyperbolicConfig;

#[test]
fn test_hyperbolic_config_default() {
    let config = HyperbolicConfig::default();

    // Verify all 4 fields
    assert_eq!(config.dim, 64, "Default dim must be 64");
    assert_eq!(config.curvature, -1.0, "Default curvature must be -1.0");
    assert_eq!(config.eps, 1e-7, "Default eps must be 1e-7");
    assert!(
        (config.max_norm - 0.99999).abs() < 1e-10,
        "Default max_norm must be 1.0 - 1e-5"
    );

    // Invariants
    assert!(config.curvature < 0.0, "Curvature must be negative");
    assert!(config.max_norm < 1.0, "Max norm must be < 1.0");
    assert!(config.max_norm > 0.0, "Max norm must be positive");
    assert!(config.eps > 0.0, "Eps must be positive");
}

#[test]
fn test_hyperbolic_config_with_curvature() {
    let config = HyperbolicConfig::with_curvature(-0.5);
    assert_eq!(config.curvature, -0.5);
    assert_eq!(config.dim, 64); // defaults preserved
    assert_eq!(config.eps, 1e-7);
}

#[test]
fn test_hyperbolic_config_abs_curvature() {
    let config = HyperbolicConfig::default();
    assert_eq!(config.abs_curvature(), 1.0);

    let config2 = HyperbolicConfig::with_curvature(-2.5);
    assert_eq!(config2.abs_curvature(), 2.5);
}

#[test]
fn test_hyperbolic_config_scale() {
    let config = HyperbolicConfig::default();
    assert_eq!(config.scale(), 1.0); // sqrt(|-1.0|) = 1.0

    let config2 = HyperbolicConfig::with_curvature(-4.0);
    assert_eq!(config2.scale(), 2.0); // sqrt(|-4.0|) = 2.0
}

#[test]
fn test_hyperbolic_config_serialization_roundtrip() {
    let config = HyperbolicConfig::default();
    let json = serde_json::to_string(&config).expect("Serialization failed");
    let deserialized: HyperbolicConfig =
        serde_json::from_str(&json).expect("Deserialization failed");
    assert_eq!(config, deserialized);
}

#[test]
fn test_hyperbolic_config_json_fields() {
    let config = HyperbolicConfig::default();
    let json = serde_json::to_string_pretty(&config).expect("Serialization failed");

    // Verify all 4 fields appear in JSON
    assert!(json.contains("\"dim\":"), "JSON must contain dim field");
    assert!(
        json.contains("\"curvature\":"),
        "JSON must contain curvature field"
    );
    assert!(json.contains("\"eps\":"), "JSON must contain eps field");
    assert!(
        json.contains("\"max_norm\":"),
        "JSON must contain max_norm field"
    );
}

// ============ Validation Tests ============

#[test]
fn test_validate_default_passes() {
    let config = HyperbolicConfig::default();
    assert!(config.validate().is_ok(), "Default config must be valid");
}

#[test]
fn test_validate_dim_zero_fails() {
    let config = HyperbolicConfig {
        dim: 0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("dim"), "Error should mention 'dim'");
    assert!(
        err_msg.contains("positive"),
        "Error should mention 'positive'"
    );
}

#[test]
fn test_validate_curvature_zero_fails() {
    let config = HyperbolicConfig {
        curvature: 0.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("curvature"),
        "Error should mention 'curvature'"
    );
    assert!(
        err_msg.contains("negative"),
        "Error should mention 'negative'"
    );
}

#[test]
fn test_validate_curvature_positive_fails() {
    let config = HyperbolicConfig {
        curvature: 1.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("1"),
        "Error should include the actual value"
    );
}

#[test]
fn test_validate_curvature_nan_fails() {
    let config = HyperbolicConfig {
        curvature: f32::NAN,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("NaN"), "Error should mention 'NaN'");
}

#[test]
fn test_validate_eps_zero_fails() {
    let config = HyperbolicConfig {
        eps: 0.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("eps"), "Error should mention 'eps'");
}

#[test]
fn test_validate_eps_negative_fails() {
    let config = HyperbolicConfig {
        eps: -1e-7,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn test_validate_max_norm_zero_fails() {
    let config = HyperbolicConfig {
        max_norm: 0.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("max_norm"),
        "Error should mention 'max_norm'"
    );
}

#[test]
fn test_validate_max_norm_one_fails() {
    let config = HyperbolicConfig {
        max_norm: 1.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(
        result.is_err(),
        "max_norm=1.0 is ON boundary, not inside ball"
    );
}

#[test]
fn test_validate_max_norm_greater_than_one_fails() {
    let config = HyperbolicConfig {
        max_norm: 1.5,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn test_validate_max_norm_negative_fails() {
    let config = HyperbolicConfig {
        max_norm: -0.5,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn test_validate_custom_valid_curvature() {
    // Various valid negative curvatures
    for c in [-0.1, -0.5, -1.0, -2.0, -10.0] {
        let config = HyperbolicConfig::with_curvature(c);
        assert!(config.validate().is_ok(), "curvature {} should be valid", c);
    }
}

#[test]
fn test_try_with_curvature_valid() {
    let config = HyperbolicConfig::try_with_curvature(-0.5).unwrap();
    assert_eq!(config.curvature, -0.5);
    assert_eq!(config.dim, 64); // default
}

#[test]
fn test_try_with_curvature_invalid() {
    assert!(HyperbolicConfig::try_with_curvature(0.0).is_err());
    assert!(HyperbolicConfig::try_with_curvature(1.0).is_err());
    assert!(HyperbolicConfig::try_with_curvature(f32::NAN).is_err());
}

#[test]
fn test_validate_fail_fast_order() {
    // When multiple fields are invalid, should fail on first check (dim)
    let config = HyperbolicConfig {
        dim: 0,
        curvature: 1.0, // also invalid
        eps: -1.0,      // also invalid
        max_norm: 2.0,  // also invalid
    };
    let err_msg = config.validate().unwrap_err().to_string();
    assert!(err_msg.contains("dim"), "Should fail on dim first");
}

#[test]
fn test_validate_boundary_values() {
    // Test values very close to boundaries
    let barely_valid = HyperbolicConfig {
        dim: 1,
        curvature: -1e-10,   // tiny but negative
        eps: 1e-10,          // tiny but positive
        max_norm: 0.9999999, // close to 1 but not 1
    };
    assert!(barely_valid.validate().is_ok());
}
