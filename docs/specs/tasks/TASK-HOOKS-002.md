# TASK-HOOKS-002: Create HookInput and HookOutput Types

```xml
<task_spec id="TASK-HOOKS-002" version="1.0">
<metadata>
  <title>Create HookInput and HookOutput Types</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>2</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-10</requirement_ref>
    <requirement_ref>REQ-HOOKS-13</requirement_ref>
    <requirement_ref>REQ-HOOKS-14</requirement_ref>
    <requirement_ref>REQ-HOOKS-15</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-001</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>1.0</estimated_hours>
</metadata>

<context>
This task creates the HookInput and HookOutput structs that define the JSON contract between
Claude Code and context-graph-cli. HookInput is received via stdin from Claude Code's hook
executor. HookOutput is returned via stdout and may include consciousness state, IC classification,
and optional context injection content.

These types form the core communication protocol for the native hooks integration.
</context>

<input_context_files>
  <file purpose="type_definitions">crates/context-graph-cli/src/commands/hooks/types.rs</file>
  <file purpose="technical_spec">docs/specs/technical/TECH-HOOKS.md#section-2.2</file>
  <file purpose="json_schema">docs/specs/technical/TECH-HOOKS.md#section-3.3</file>
</input_context_files>

<prerequisites>
  <check>TASK-HOOKS-001 completed (HookEventType exists)</check>
  <check>serde, serde_json dependencies available</check>
</prerequisites>

<scope>
  <in_scope>
    - Create HookInput struct with hook_type, session_id, timestamp_ms, payload
    - Create HookOutput struct with success, error, consciousness_state, ic_classification, context_injection, execution_time_ms
    - Create ConsciousnessState struct for hook output
    - Create ICClassification struct with value, level, crisis_triggered
    - Create ICLevel enum (Healthy, Normal, Warning, Critical)
    - Create JohariQuadrant enum (Open, Blind, Hidden, Unknown)
    - Add unit tests
  </in_scope>
  <out_of_scope>
    - HookPayload variants (TASK-HOOKS-003)
    - CLI argument types (TASK-HOOKS-004)
    - Error handling types (TASK-HOOKS-005)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/commands/hooks/types.rs">
/// Input received from Claude Code hook system via stdin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookInput {
    pub hook_type: HookEventType,
    pub session_id: String,
    pub timestamp_ms: i64,
    pub payload: HookPayload,
}

/// Output returned to Claude Code hook system via stdout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookOutput {
    pub success: bool,
    pub error: Option&lt;String&gt;,
    pub consciousness_state: Option&lt;ConsciousnessState&gt;,
    pub ic_classification: Option&lt;ICClassification&gt;,
    pub context_injection: Option&lt;String&gt;,
    pub execution_time_ms: u64,
}

/// Consciousness state for hook output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessState {
    pub consciousness: f32,
    pub integration: f32,
    pub reflection: f32,
    pub differentiation: f32,
    pub identity_continuity: f32,
    pub johari_quadrant: JohariQuadrant,
}

/// Identity Continuity classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ICClassification {
    pub value: f32,
    pub level: ICLevel,
    pub crisis_triggered: bool,
}

/// IC level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ICLevel {
    Healthy,   // IC >= 0.9
    Normal,    // 0.7 <= IC < 0.9
    Warning,   // 0.5 <= IC < 0.7
    Critical,  // IC < 0.5
}

/// Johari window quadrant classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JohariQuadrant {
    Open,    // Known to self and others
    Blind,   // Unknown to self, known to others
    Hidden,  // Known to self, unknown to others
    Unknown, // Unknown to self and others
}
    </signature>
  </signatures>
  <constraints>
    - All structs MUST derive Serialize, Deserialize
    - ICLevel thresholds: Healthy>=0.9, Normal>=0.7, Warning>=0.5, Critical<0.5
    - HookOutput MUST match JSON schema in TECH-HOOKS.md section 3.3
    - execution_time_ms MUST be u64 (not i64)
  </constraints>
  <verification>
    - cargo test --package context-graph-cli hook_input_output
    - Verify JSON round-trip for all types
    - Verify IC level classification thresholds
  </verification>
</definition_of_done>

<pseudo_code>
1. Add to types.rs after HookEventType:

2. Create ICLevel enum:
   - Healthy (>= 0.9)
   - Normal (>= 0.7)
   - Warning (>= 0.5)
   - Critical (< 0.5)
   - Implement from_value(ic: f32) -> Self

3. Create JohariQuadrant enum:
   - Open (high consciousness, high integration)
   - Blind (low consciousness, high integration)
   - Hidden (high consciousness, low integration)
   - Unknown (low consciousness, low integration)

4. Create ConsciousnessState struct:
   - consciousness: f32
   - integration: f32
   - reflection: f32
   - differentiation: f32
   - identity_continuity: f32
   - johari_quadrant: JohariQuadrant

5. Create ICClassification struct:
   - value: f32
   - level: ICLevel
   - crisis_triggered: bool

6. Create HookInput struct (placeholder payload as serde_json::Value):
   - hook_type: HookEventType
   - session_id: String
   - timestamp_ms: i64
   - payload: serde_json::Value (temporary, replaced in TASK-HOOKS-003)

7. Create HookOutput struct:
   - success: bool
   - error: Option<String>
   - consciousness_state: Option<ConsciousnessState>
   - ic_classification: Option<ICClassification>
   - context_injection: Option<String>
   - execution_time_ms: u64

8. Implement Default for HookOutput (success: true, all others None/0)

9. Add tests for JSON serialization, IC classification
</pseudo_code>

<files_to_create>
  <!-- types.rs already exists from TASK-HOOKS-001 -->
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/commands/hooks/types.rs">Add HookInput, HookOutput, ConsciousnessState, ICClassification, ICLevel, JohariQuadrant</file>
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli hook_input_output</command>
</test_commands>
</task_spec>
```

## Implementation

### Add to types.rs (after HookEventType)

```rust
// ============================================================================
// IC Level Classification
// ============================================================================

/// IC level classification
/// Thresholds per constitution IDENTITY-002: Healthy>0.9, Warning<0.7, Critical<0.5
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ICLevel {
    /// IC >= 0.9 - Identity is stable and coherent
    Healthy,
    /// 0.7 <= IC < 0.9 - Normal operation
    Normal,
    /// 0.5 <= IC < 0.7 - Identity drift detected
    Warning,
    /// IC < 0.5 - Crisis state, auto-dream may trigger
    Critical,
}

impl ICLevel {
    /// Classify IC value into level
    ///
    /// # Arguments
    /// * `ic` - Identity continuity value [0.0, 1.0]
    ///
    /// # Returns
    /// ICLevel classification based on IDENTITY-002 thresholds
    #[inline]
    pub fn from_value(ic: f32) -> Self {
        if ic >= 0.9 {
            Self::Healthy
        } else if ic >= 0.7 {
            Self::Normal
        } else if ic >= 0.5 {
            Self::Warning
        } else {
            Self::Critical
        }
    }

    /// Check if this level indicates a crisis state
    #[inline]
    pub const fn is_crisis(&self) -> bool {
        matches!(self, Self::Critical)
    }

    /// Check if this level requires attention
    #[inline]
    pub const fn needs_attention(&self) -> bool {
        matches!(self, Self::Warning | Self::Critical)
    }
}

// ============================================================================
// Johari Window Classification
// ============================================================================

/// Johari window quadrant classification
/// Implements REQ-HOOKS-16
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JohariQuadrant {
    /// Known to self and others - high consciousness, high external awareness
    Open,
    /// Unknown to self, known to others - external feedback needed
    Blind,
    /// Known to self, unknown to others - internal knowledge not shared
    Hidden,
    /// Unknown to self and others - unconscious/unexplored territory
    Unknown,
}

impl JohariQuadrant {
    /// Johari quadrant thresholds
    pub const HIGH_THRESHOLD: f32 = 0.7;
    pub const LOW_THRESHOLD: f32 = 0.3;

    /// Classify from consciousness and integration values
    ///
    /// # Arguments
    /// * `consciousness` - Consciousness level C(t) [0.0, 1.0]
    /// * `integration` - Integration factor (Kuramoto r) [0.0, 1.0]
    ///
    /// # Returns
    /// Johari quadrant based on thresholds
    pub fn classify(consciousness: f32, integration: f32) -> Self {
        let high_c = consciousness >= Self::HIGH_THRESHOLD;
        let high_i = integration >= Self::HIGH_THRESHOLD;

        match (high_c, high_i) {
            (true, true) => Self::Open,
            (false, true) => Self::Blind,
            (true, false) => Self::Hidden,
            (false, false) => Self::Unknown,
        }
    }
}

// ============================================================================
// Consciousness State
// ============================================================================

/// Consciousness state for hook output
/// Implements REQ-HOOKS-14, REQ-HOOKS-15
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessState {
    /// Current consciousness level C(t) [0.0, 1.0]
    pub consciousness: f32,
    /// Integration (Kuramoto r) [0.0, 1.0]
    pub integration: f32,
    /// Reflection (meta-cognitive) [0.0, 1.0]
    pub reflection: f32,
    /// Differentiation (purpose entropy) [0.0, 1.0]
    pub differentiation: f32,
    /// Identity continuity score [0.0, 1.0]
    pub identity_continuity: f32,
    /// Johari quadrant classification
    pub johari_quadrant: JohariQuadrant,
}

impl Default for ConsciousnessState {
    fn default() -> Self {
        Self {
            consciousness: 0.0,
            integration: 0.0,
            reflection: 0.0,
            differentiation: 0.0,
            identity_continuity: 1.0,
            johari_quadrant: JohariQuadrant::Unknown,
        }
    }
}

// ============================================================================
// IC Classification
// ============================================================================

/// Identity Continuity classification
/// Constitution Reference: IDENTITY-002
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ICClassification {
    /// IC value [0.0, 1.0]
    pub value: f32,
    /// Classification level
    pub level: ICLevel,
    /// Whether crisis threshold was breached
    pub crisis_triggered: bool,
}

impl ICClassification {
    /// Create new IC classification from value
    ///
    /// # Arguments
    /// * `value` - IC value [0.0, 1.0]
    /// * `crisis_threshold` - Threshold for crisis trigger (default 0.5)
    pub fn new(value: f32, crisis_threshold: f32) -> Self {
        let level = ICLevel::from_value(value);
        Self {
            value,
            level,
            crisis_triggered: value < crisis_threshold,
        }
    }
}

// ============================================================================
// Hook Input/Output
// ============================================================================

/// Input received from Claude Code hook system via stdin
/// Implements REQ-HOOKS-03, REQ-HOOKS-10
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookInput {
    /// Hook event type
    pub hook_type: HookEventType,
    /// Session identifier from Claude Code
    pub session_id: String,
    /// Unix timestamp in milliseconds
    pub timestamp_ms: i64,
    /// Event-specific payload (typed in TASK-HOOKS-003)
    pub payload: serde_json::Value,
}

/// Output returned to Claude Code hook system via stdout
/// Implements REQ-HOOKS-04, REQ-HOOKS-13
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookOutput {
    /// Whether hook execution succeeded
    pub success: bool,
    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Consciousness state snapshot
    #[serde(skip_serializing_if = "Option::is_none")]
    pub consciousness_state: Option<ConsciousnessState>,
    /// Identity continuity classification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ic_classification: Option<ICClassification>,
    /// Content to inject into context (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_injection: Option<String>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
}

impl Default for HookOutput {
    fn default() -> Self {
        Self {
            success: true,
            error: None,
            consciousness_state: None,
            ic_classification: None,
            context_injection: None,
            execution_time_ms: 0,
        }
    }
}

impl HookOutput {
    /// Create successful output with execution time
    pub fn success(execution_time_ms: u64) -> Self {
        Self {
            success: true,
            execution_time_ms,
            ..Default::default()
        }
    }

    /// Create error output
    pub fn error(message: impl Into<String>, execution_time_ms: u64) -> Self {
        Self {
            success: false,
            error: Some(message.into()),
            execution_time_ms,
            ..Default::default()
        }
    }

    /// Add consciousness state to output
    pub fn with_consciousness_state(mut self, state: ConsciousnessState) -> Self {
        self.consciousness_state = Some(state);
        self
    }

    /// Add IC classification to output
    pub fn with_ic_classification(mut self, classification: ICClassification) -> Self {
        self.ic_classification = Some(classification);
        self
    }

    /// Add context injection to output
    pub fn with_context_injection(mut self, content: impl Into<String>) -> Self {
        self.context_injection = Some(content.into());
        self
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod hook_input_output_tests {
    use super::*;

    #[test]
    fn test_ic_level_from_value() {
        assert_eq!(ICLevel::from_value(0.95), ICLevel::Healthy);
        assert_eq!(ICLevel::from_value(0.90), ICLevel::Healthy);
        assert_eq!(ICLevel::from_value(0.85), ICLevel::Normal);
        assert_eq!(ICLevel::from_value(0.70), ICLevel::Normal);
        assert_eq!(ICLevel::from_value(0.65), ICLevel::Warning);
        assert_eq!(ICLevel::from_value(0.50), ICLevel::Warning);
        assert_eq!(ICLevel::from_value(0.49), ICLevel::Critical);
        assert_eq!(ICLevel::from_value(0.0), ICLevel::Critical);
    }

    #[test]
    fn test_johari_classification() {
        assert_eq!(JohariQuadrant::classify(0.8, 0.8), JohariQuadrant::Open);
        assert_eq!(JohariQuadrant::classify(0.2, 0.8), JohariQuadrant::Blind);
        assert_eq!(JohariQuadrant::classify(0.8, 0.2), JohariQuadrant::Hidden);
        assert_eq!(JohariQuadrant::classify(0.2, 0.2), JohariQuadrant::Unknown);
    }

    #[test]
    fn test_hook_output_default() {
        let output = HookOutput::default();
        assert!(output.success);
        assert!(output.error.is_none());
        assert!(output.consciousness_state.is_none());
        assert_eq!(output.execution_time_ms, 0);
    }

    #[test]
    fn test_hook_output_json_roundtrip() {
        let output = HookOutput::success(42)
            .with_consciousness_state(ConsciousnessState::default())
            .with_ic_classification(ICClassification::new(0.85, 0.5));

        let json = serde_json::to_string(&output).unwrap();
        let parsed: HookOutput = serde_json::from_str(&json).unwrap();

        assert_eq!(output.success, parsed.success);
        assert_eq!(output.execution_time_ms, parsed.execution_time_ms);
    }

    #[test]
    fn test_hook_output_matches_json_schema() {
        // Verify output matches TECH-HOOKS.md section 3.3 schema
        let output = HookOutput {
            success: true,
            error: None,
            consciousness_state: Some(ConsciousnessState {
                consciousness: 0.73,
                integration: 0.85,
                reflection: 0.78,
                differentiation: 0.82,
                identity_continuity: 0.92,
                johari_quadrant: JohariQuadrant::Open,
            }),
            ic_classification: Some(ICClassification {
                value: 0.92,
                level: ICLevel::Healthy,
                crisis_triggered: false,
            }),
            context_injection: None,
            execution_time_ms: 15,
        };

        let json = serde_json::to_value(&output).unwrap();

        // Check required fields
        assert!(json.get("success").is_some());
        assert!(json.get("execution_time_ms").is_some());

        // Check consciousness_state structure
        let cs = json.get("consciousness_state").unwrap();
        assert!(cs.get("consciousness").is_some());
        assert!(cs.get("integration").is_some());
        assert!(cs.get("johari_quadrant").is_some());
    }

    #[test]
    fn test_ic_classification_crisis_trigger() {
        let classification = ICClassification::new(0.45, 0.5);
        assert!(classification.crisis_triggered);
        assert_eq!(classification.level, ICLevel::Critical);

        let classification = ICClassification::new(0.55, 0.5);
        assert!(!classification.crisis_triggered);
        assert_eq!(classification.level, ICLevel::Warning);
    }
}
```

## Verification Checklist

- [ ] HookInput struct has all 4 required fields
- [ ] HookOutput struct matches JSON schema in TECH-HOOKS.md
- [ ] ICLevel::from_value() uses correct thresholds
- [ ] JohariQuadrant::classify() uses correct thresholds (0.7 high, 0.3 low)
- [ ] All types derive Serialize, Deserialize
- [ ] JSON round-trip tests pass
