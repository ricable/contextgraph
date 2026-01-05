//! Tests for configuration validation.

use crate::config::Config;

#[test]
fn test_validation_passes() {
    let config = Config::default_config();
    assert!(config.validate().is_ok());
}

#[test]
fn test_validation_fails_zero_payload() {
    let mut config = Config::default_config();
    config.mcp.max_payload_size = 0;
    assert!(config.validate().is_err());
}

// =========================================================================
// TC-GHOST-006: Configuration & Infrastructure Tests - Validation
// =========================================================================

#[test]
fn test_config_validation_embedding_dimension_zero() {
    // TC-GHOST-006: Embedding dimension 0 must fail validation
    let mut config = Config::default_config();
    config.embedding.dimension = 0;

    let result = config.validate();
    assert!(
        result.is_err(),
        "Embedding dimension 0 must fail validation"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("embedding.dimension"),
        "Error must mention embedding.dimension"
    );
}
