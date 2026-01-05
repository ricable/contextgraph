# TASK-S005: New MCP Handlers for Meta-UTL Operations

```yaml
metadata:
  id: "TASK-S005"
  title: "New MCP Handlers for Meta-UTL Operations"
  layer: "surface"
  priority: "P1"
  estimated_hours: 6
  created: "2026-01-04"
  status: "pending"
  dependencies:
    - "TASK-L002"  # Purpose Vector Computation (for learning trajectory)
    - "TASK-L003"  # Goal Alignment Calculator (for predictions)
  traces_to:
    - "FR-501"  # Self-Aware Learning Monitoring
    - "FR-502"  # Learning Trajectory Tracking
    - "FR-503"  # System Health Metrics
```

## Problem Statement

Create new MCP handlers for Meta-UTL (meta-learning about unified theory of learning) operations including learning trajectory queries, system health metrics, per-space accuracy tracking, and prediction validation.

## Context

Meta-UTL enables the system to learn about its own learning. Key capabilities:
1. Predict storage impact before committing memories
2. Predict retrieval quality before executing queries
3. Track per-space learning accuracy over time
4. Self-adjust UTL parameters based on prediction outcomes
5. Expose system health metrics for monitoring

**These are NEW handlers - no legacy equivalents exist.**

## Technical Specification

### MCP Handler Function Signatures

```rust
/// Get learning trajectory per embedding space
pub async fn handle_meta_utl_learning_trajectory(
    request: LearningTrajectoryRequest,
    meta_utl: Arc<MetaUTL>,
) -> Result<LearningTrajectoryResponse, McpError>;

/// Get system health metrics
pub async fn handle_meta_utl_health_metrics(
    request: HealthMetricsRequest,
    meta_utl: Arc<MetaUTL>,
) -> Result<HealthMetricsResponse, McpError>;

/// Predict storage impact
pub async fn handle_meta_utl_predict_storage(
    request: StoragePredictionRequest,
    meta_utl: Arc<MetaUTL>,
) -> Result<StoragePredictionResponse, McpError>;

/// Predict retrieval quality
pub async fn handle_meta_utl_predict_retrieval(
    request: RetrievalPredictionRequest,
    meta_utl: Arc<MetaUTL>,
) -> Result<RetrievalPredictionResponse, McpError>;

/// Validate prediction against actual outcome
pub async fn handle_meta_utl_validate_prediction(
    request: PredictionValidationRequest,
    meta_utl: Arc<MetaUTL>,
) -> Result<PredictionValidationResponse, McpError>;

/// Get optimized weights from meta-learning
pub async fn handle_meta_utl_get_optimized_weights(
    meta_utl: Arc<MetaUTL>,
) -> Result<OptimizedWeightsResponse, McpError>;
```

### Request/Response JSON Schemas

#### LearningTrajectoryRequest

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "embedder_indices": {
      "type": "array",
      "items": { "type": "integer", "minimum": 0, "maximum": 11 },
      "description": "Specific embedders to query (null for all)"
    },
    "history_window": {
      "type": "integer",
      "minimum": 10,
      "maximum": 1000,
      "default": 100,
      "description": "Number of recent predictions to include"
    },
    "include_accuracy_trend": {
      "type": "boolean",
      "default": true
    }
  }
}
```

#### LearningTrajectoryResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["trajectories"],
  "properties": {
    "trajectories": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["embedder_index", "current_weight", "recent_accuracy"],
        "properties": {
          "embedder_index": { "type": "integer" },
          "embedder_name": { "type": "string" },
          "current_weight": {
            "type": "number",
            "description": "Current weight (learned from accuracy)"
          },
          "initial_weight": { "type": "number" },
          "weight_delta": {
            "type": "number",
            "description": "Change from initial weight"
          },
          "alignment_threshold": {
            "type": "number",
            "description": "Space-specific alignment threshold"
          },
          "recent_accuracy": {
            "type": "number",
            "description": "Rolling window prediction accuracy [0, 1]"
          },
          "prediction_count": { "type": "integer" },
          "correct_predictions": { "type": "integer" },
          "accuracy_trend": {
            "type": "string",
            "enum": ["improving", "stable", "degrading"],
            "description": "Direction of accuracy change"
          },
          "accuracy_history": {
            "type": "array",
            "items": { "type": "number" },
            "description": "Recent accuracy values for trend visualization"
          }
        }
      },
      "minItems": 1,
      "maxItems": 12
    },
    "system_summary": {
      "type": "object",
      "properties": {
        "overall_accuracy": { "type": "number" },
        "best_performing_space": { "type": "integer" },
        "worst_performing_space": { "type": "integer" },
        "spaces_above_target": { "type": "integer" },
        "spaces_below_target": { "type": "integer" }
      }
    }
  }
}
```

#### HealthMetricsRequest

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "include_targets": {
      "type": "boolean",
      "default": true,
      "description": "Include target thresholds for comparison"
    },
    "include_recommendations": {
      "type": "boolean",
      "default": false,
      "description": "Include actionable recommendations"
    }
  }
}
```

#### HealthMetricsResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["metrics"],
  "properties": {
    "metrics": {
      "type": "object",
      "required": ["learning_score", "coherence_recovery_time_ms", "attack_detection_rate", "false_positive_rate"],
      "properties": {
        "learning_score": {
          "type": "number",
          "description": "Overall learning score (UTL avg, target > 0.6)"
        },
        "learning_score_target": { "type": "number" },
        "learning_score_status": {
          "type": "string",
          "enum": ["passing", "warning", "failing"]
        },
        "coherence_recovery_time_ms": {
          "type": "number",
          "description": "Time to recover coherence (target < 10000ms)"
        },
        "coherence_recovery_target_ms": { "type": "number" },
        "coherence_recovery_status": {
          "type": "string",
          "enum": ["passing", "warning", "failing"]
        },
        "attack_detection_rate": {
          "type": "number",
          "description": "Attack detection rate (target > 0.95)"
        },
        "attack_detection_target": { "type": "number" },
        "attack_detection_status": {
          "type": "string",
          "enum": ["passing", "warning", "failing"]
        },
        "false_positive_rate": {
          "type": "number",
          "description": "False positive rate (target < 0.02)"
        },
        "false_positive_target": { "type": "number" },
        "false_positive_status": {
          "type": "string",
          "enum": ["passing", "warning", "failing"]
        },
        "per_space_accuracy": {
          "type": "array",
          "items": { "type": "number" },
          "minItems": 12,
          "maxItems": 12
        }
      }
    },
    "overall_status": {
      "type": "string",
      "enum": ["healthy", "degraded", "unhealthy"],
      "description": "Overall system health status"
    },
    "failed_targets": {
      "type": "array",
      "items": { "type": "string" }
    },
    "recommendations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "metric": { "type": "string" },
          "issue": { "type": "string" },
          "action": { "type": "string" },
          "priority": { "type": "string", "enum": ["high", "medium", "low"] }
        }
      }
    }
  }
}
```

#### StoragePredictionRequest

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["fingerprint"],
  "properties": {
    "fingerprint": {
      "type": "object",
      "description": "TeleologicalFingerprint to predict impact for"
    },
    "include_confidence": {
      "type": "boolean",
      "default": true
    }
  }
}
```

#### StoragePredictionResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["predictions"],
  "properties": {
    "predictions": {
      "type": "object",
      "properties": {
        "coherence_delta": {
          "type": "number",
          "description": "Predicted change in system coherence [-1, 1]"
        },
        "alignment_delta": {
          "type": "number",
          "description": "Predicted change in average alignment [-1, 1]"
        },
        "storage_impact_bytes": {
          "type": "integer",
          "description": "Predicted storage impact in bytes"
        },
        "index_rebuild_required": {
          "type": "boolean",
          "description": "Whether indexes need rebuilding"
        }
      }
    },
    "confidence": {
      "type": "number",
      "description": "Prediction confidence [0, 1]"
    },
    "prediction_id": {
      "type": "string",
      "format": "uuid",
      "description": "ID for later validation"
    }
  }
}
```

#### RetrievalPredictionRequest

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["query_fingerprint"],
  "properties": {
    "query_fingerprint": {
      "type": "object",
      "description": "Query fingerprint for prediction"
    },
    "target_top_k": {
      "type": "integer",
      "default": 10
    }
  }
}
```

#### RetrievalPredictionResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["predictions"],
  "properties": {
    "predictions": {
      "type": "object",
      "properties": {
        "expected_relevance": {
          "type": "number",
          "description": "Predicted average relevance of top-k [0, 1]"
        },
        "expected_alignment": {
          "type": "number",
          "description": "Predicted alignment of results to query goal [0, 1]"
        },
        "expected_result_count": {
          "type": "integer",
          "description": "Predicted number of results above threshold"
        },
        "per_space_expected_contribution": {
          "type": "array",
          "items": { "type": "number" },
          "minItems": 12,
          "maxItems": 12
        }
      }
    },
    "confidence": { "type": "number" },
    "prediction_id": { "type": "string", "format": "uuid" }
  }
}
```

#### PredictionValidationRequest

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["prediction_id", "actual_outcome"],
  "properties": {
    "prediction_id": {
      "type": "string",
      "format": "uuid"
    },
    "actual_outcome": {
      "type": "object",
      "description": "Actual measured outcome for comparison"
    }
  }
}
```

#### OptimizedWeightsResponse

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["weights"],
  "properties": {
    "weights": {
      "type": "array",
      "items": { "type": "number" },
      "minItems": 12,
      "maxItems": 12,
      "description": "Meta-learned optimized weights per embedder"
    },
    "confidence": {
      "type": "number",
      "description": "Confidence in weight optimization"
    },
    "training_samples": {
      "type": "integer",
      "description": "Number of samples used for optimization"
    },
    "last_updated": {
      "type": "string",
      "format": "date-time"
    }
  }
}
```

### Error Handling

```rust
#[derive(Debug, Clone, Serialize)]
pub enum McpMetaUtlError {
    /// Prediction not found for validation
    PredictionNotFound { id: Uuid },
    /// Meta-UTL not initialized
    MetaUtlNotInitialized,
    /// Insufficient data for prediction
    InsufficientData { required: usize, available: usize },
    /// Invalid outcome format
    InvalidOutcomeFormat { message: String },
}
```

## Implementation Requirements

### Prerequisites

- [ ] TASK-L002 complete (Purpose Vector Computation for learning tracking)
- [ ] TASK-L003 complete (Goal Alignment Calculator for predictions)

### Scope

#### In Scope

- Learning trajectory queries per embedder
- System health metrics with targets
- Storage impact prediction before commit
- Retrieval quality prediction before query
- Prediction validation and feedback loop
- Meta-learned optimized weights

#### Out of Scope

- Direct memory operations (TASK-S001)
- Search execution (TASK-S002)
- Purpose/goal operations (TASK-S003)

### Constraints

- Predictions must include confidence scores
- Validation updates learning trajectories
- Health metrics align with constitution.yaml targets
- Storage prediction accuracy target > 0.85
- Retrieval prediction accuracy target > 0.80

## Definition of Done

### Implementation Checklist

- [ ] `handle_meta_utl_learning_trajectory` with trends
- [ ] `handle_meta_utl_health_metrics` with status
- [ ] `handle_meta_utl_predict_storage` with confidence
- [ ] `handle_meta_utl_predict_retrieval` with confidence
- [ ] `handle_meta_utl_validate_prediction` with feedback
- [ ] `handle_meta_utl_get_optimized_weights` from learning
- [ ] All error cases with context

### Testing Requirements

Tests MUST use REAL prediction outcomes from actual operations.

```rust
#[cfg(test)]
mod tests {
    use crate::test_fixtures::load_meta_utl_test_data;

    #[tokio::test]
    async fn test_learning_trajectory() {
        // Pre-populate with real prediction history
        let meta_utl = create_meta_utl_with_history(100);

        let request = LearningTrajectoryRequest {
            embedder_indices: None, // All
            history_window: 50,
            include_accuracy_trend: true,
        };

        let response = handle_meta_utl_learning_trajectory(request, meta_utl.clone()).await.unwrap();

        assert_eq!(response.trajectories.len(), 12);

        for traj in &response.trajectories {
            assert!(traj.recent_accuracy >= 0.0 && traj.recent_accuracy <= 1.0);
            assert!(traj.current_weight >= 0.0);
            assert!(["improving", "stable", "degrading"].contains(&traj.accuracy_trend.as_str()));
        }
    }

    #[tokio::test]
    async fn test_health_metrics_targets() {
        let meta_utl = create_test_meta_utl();

        let request = HealthMetricsRequest {
            include_targets: true,
            include_recommendations: true,
        };

        let response = handle_meta_utl_health_metrics(request, meta_utl.clone()).await.unwrap();

        // Verify targets match constitution.yaml
        assert_eq!(response.metrics.learning_score_target, 0.6);
        assert_eq!(response.metrics.coherence_recovery_target_ms, 10_000);
        assert_eq!(response.metrics.attack_detection_target, 0.95);
        assert_eq!(response.metrics.false_positive_target, 0.02);

        // Verify status calculation
        if response.metrics.learning_score >= 0.6 {
            assert_eq!(response.metrics.learning_score_status, "passing");
        }
    }

    #[tokio::test]
    async fn test_prediction_validation_updates_trajectory() {
        let meta_utl = create_test_meta_utl();

        // Make a prediction
        let fp = create_test_fingerprint();
        let pred_resp = handle_meta_utl_predict_storage(
            StoragePredictionRequest { fingerprint: fp, include_confidence: true },
            meta_utl.clone()
        ).await.unwrap();

        let initial_accuracy = meta_utl.system_accuracy();

        // Validate with actual outcome
        let validation = PredictionValidationRequest {
            prediction_id: pred_resp.prediction_id,
            actual_outcome: serde_json::json!({
                "coherence_delta": pred_resp.predictions.coherence_delta + 0.05, // Slight error
                "alignment_delta": pred_resp.predictions.alignment_delta,
            }),
        };

        handle_meta_utl_validate_prediction(validation, meta_utl.clone()).await.unwrap();

        // Accuracy should update based on prediction error
        let new_accuracy = meta_utl.system_accuracy();
        // Accuracy tracking is updated (may go up or down based on error)
        assert!(new_accuracy != initial_accuracy || meta_utl.prediction_count() > 0);
    }

    #[tokio::test]
    async fn test_optimized_weights_sum_to_one() {
        let meta_utl = create_meta_utl_with_training(500);

        let response = handle_meta_utl_get_optimized_weights(meta_utl.clone()).await.unwrap();

        let sum: f32 = response.weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Weights should sum to 1.0, got {}", sum);

        // All weights should be positive
        for w in &response.weights {
            assert!(*w >= 0.0);
        }
    }
}
```

### Verification Commands

```bash
# Run Meta-UTL handler tests
cargo test -p context-graph-mcp meta_utl_handlers

# Verify health metrics
cargo test -p context-graph-mcp health_metrics

# Test prediction validation
cargo test -p context-graph-mcp prediction_validation
```

## Files to Create

| File | Description |
|------|-------------|
| `crates/context-graph-mcp/src/handlers/meta_utl.rs` | Meta-UTL handlers |
| `crates/context-graph-mcp/src/schemas/learning_trajectory.json` | Trajectory schema |
| `crates/context-graph-mcp/src/schemas/health_metrics.json` | Health schema |

## Files to Modify

| File | Change |
|------|--------|
| `crates/context-graph-mcp/src/handlers/mod.rs` | Export meta_utl handlers |
| `crates/context-graph-mcp/src/router.rs` | Register meta_utl routes |
| `crates/context-graph-mcp/src/error.rs` | Add McpMetaUtlError |

## Traceability

| Requirement | Source | Coverage |
|-------------|--------|----------|
| FR-501 | FUNC-SPEC-001 | Self-aware learning |
| FR-502 | FUNC-SPEC-001 | Learning trajectory |
| FR-503 | FUNC-SPEC-001 | System health metrics |
| AC-501.1 | FUNC-SPEC-001 | Storage prediction > 0.85 |
| AC-501.2 | FUNC-SPEC-001 | Retrieval prediction > 0.80 |

---

*Task created: 2026-01-04*
*Layer: Surface*
*Priority: P1 - Meta-learning features*
