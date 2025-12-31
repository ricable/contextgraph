//! Johari Window quadrant classification for memory categorization.
//!
//! The Johari Window is a psychological model adapted for knowledge graph
//! classification. Each quadrant determines how memories are retrieved and
//! weighted in the UTL (Unified Theory of Learning) system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Represents the four quadrants of the Johari Window model for memory classification.
///
/// # UTL Integration
/// From constitution.yaml, Johari quadrants map to UTL states:
/// - **Open**: ΔS<0.5, ΔC>0.5 → direct recall
/// - **Blind**: ΔS>0.5, ΔC<0.5 → discovery (epistemic_action/dream)
/// - **Hidden**: ΔS<0.5, ΔC<0.5 → private (get_neighborhood)
/// - **Unknown**: ΔS>0.5, ΔC>0.5 → frontier
///
/// # Performance
/// All methods are O(1) with no allocations per constitution.yaml requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JohariQuadrant {
    /// Known to self and others - direct recall
    Open,
    /// Known to self, hidden from others - private knowledge
    Hidden,
    /// Unknown to self, known to others - discovered patterns
    Blind,
    /// Unknown to both - frontier knowledge
    Unknown,
}

impl JohariQuadrant {
    /// Returns true if this quadrant represents self-aware knowledge.
    /// Open and Hidden quadrants are self-aware.
    ///
    /// # Returns
    /// - `true` for Open, Hidden
    /// - `false` for Blind, Unknown
    #[inline]
    pub fn is_self_aware(&self) -> bool {
        matches!(self, Self::Open | Self::Hidden)
    }

    /// Returns true if this quadrant represents knowledge visible to others.
    /// Open and Blind quadrants are other-aware.
    ///
    /// # Returns
    /// - `true` for Open, Blind
    /// - `false` for Hidden, Unknown
    #[inline]
    pub fn is_other_aware(&self) -> bool {
        matches!(self, Self::Open | Self::Blind)
    }

    /// Returns the default retrieval weight for this quadrant.
    ///
    /// # Returns
    /// - Open: 1.0 (full weight, always retrieve)
    /// - Hidden: 0.3 (reduced weight, private)
    /// - Blind: 0.7 (high weight, discovery)
    /// - Unknown: 0.5 (medium weight, frontier)
    ///
    /// # Constraint
    /// All values in range [0.0, 1.0]
    #[inline]
    pub fn default_retrieval_weight(&self) -> f32 {
        match self {
            Self::Open => 1.0,
            Self::Hidden => 0.3,
            Self::Blind => 0.7,
            Self::Unknown => 0.5,
        }
    }

    /// Returns whether this quadrant should be included in default context retrieval.
    ///
    /// # Returns
    /// - `true` for Open, Blind, Unknown
    /// - `false` for Hidden (private knowledge requires explicit request)
    #[inline]
    pub fn include_in_default_context(&self) -> bool {
        matches!(self, Self::Open | Self::Blind | Self::Unknown)
    }

    /// Returns a human-readable description of this quadrant.
    ///
    /// # Returns
    /// Static string describing quadrant semantics.
    #[inline]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Open => "Known to self and others - direct recall",
            Self::Hidden => "Known to self, hidden from others - private knowledge",
            Self::Blind => "Unknown to self, known to others - discovered patterns",
            Self::Unknown => "Unknown to both - frontier knowledge",
        }
    }

    /// Returns the RocksDB column family name for this quadrant.
    ///
    /// # Column Family Names
    /// - "johari_open"
    /// - "johari_hidden"
    /// - "johari_blind"
    /// - "johari_unknown"
    #[inline]
    pub fn column_family(&self) -> &'static str {
        match self {
            Self::Open => "johari_open",
            Self::Hidden => "johari_hidden",
            Self::Blind => "johari_blind",
            Self::Unknown => "johari_unknown",
        }
    }

    /// Returns all quadrant variants as a fixed-size array.
    ///
    /// # Returns
    /// Array containing [Open, Hidden, Blind, Unknown] in canonical order.
    #[inline]
    pub fn all() -> [JohariQuadrant; 4] {
        [Self::Open, Self::Hidden, Self::Blind, Self::Unknown]
    }

    /// Get all valid transitions from this quadrant.
    ///
    /// Returns a static slice of (target_quadrant, trigger) pairs representing
    /// all legal state transitions from the current quadrant.
    ///
    /// # Transition Rules (from constitution.yaml)
    /// - Open → Hidden (Privatize)
    /// - Hidden → Open (ExplicitShare)
    /// - Blind → Open (SelfRecognition), Hidden (SelfRecognition)
    /// - Unknown → Open (DreamConsolidation, PatternDiscovery), Hidden (DreamConsolidation), Blind (ExternalObservation)
    pub fn valid_transitions(&self) -> &'static [(JohariQuadrant, TransitionTrigger)] {
        use TransitionTrigger::*;
        static OPEN_TRANSITIONS: [(JohariQuadrant, TransitionTrigger); 1] =
            [(JohariQuadrant::Hidden, Privatize)];
        static HIDDEN_TRANSITIONS: [(JohariQuadrant, TransitionTrigger); 1] =
            [(JohariQuadrant::Open, ExplicitShare)];
        static BLIND_TRANSITIONS: [(JohariQuadrant, TransitionTrigger); 2] = [
            (JohariQuadrant::Open, SelfRecognition),
            (JohariQuadrant::Hidden, SelfRecognition),
        ];
        static UNKNOWN_TRANSITIONS: [(JohariQuadrant, TransitionTrigger); 4] = [
            (JohariQuadrant::Open, DreamConsolidation),
            (JohariQuadrant::Open, PatternDiscovery),
            (JohariQuadrant::Hidden, DreamConsolidation),
            (JohariQuadrant::Blind, ExternalObservation),
        ];

        match self {
            Self::Open => &OPEN_TRANSITIONS,
            Self::Hidden => &HIDDEN_TRANSITIONS,
            Self::Blind => &BLIND_TRANSITIONS,
            Self::Unknown => &UNKNOWN_TRANSITIONS,
        }
    }

    /// Check if a transition to the target quadrant is valid.
    ///
    /// Returns false for self-transitions (from == to).
    pub fn can_transition_to(&self, target: JohariQuadrant) -> bool {
        if *self == target {
            return false; // No self-transitions allowed
        }
        self.valid_transitions().iter().any(|(t, _)| *t == target)
    }

    /// Attempt to transition to a target quadrant with the given trigger.
    ///
    /// # Returns
    /// - `Ok(JohariTransition)` if the transition is valid for this trigger
    /// - `Err(String)` with descriptive message if transition is invalid
    ///
    /// # Errors
    /// - Self-transitions (from == to)
    /// - Invalid target quadrant for this source
    /// - Wrong trigger for the source→target pair
    pub fn transition_to(
        &self,
        target: JohariQuadrant,
        trigger: TransitionTrigger,
    ) -> Result<JohariTransition, String> {
        if *self == target {
            return Err(format!("Cannot transition to same quadrant: {:?}", self));
        }

        let is_valid = self
            .valid_transitions()
            .iter()
            .any(|(t, tr)| *t == target && *tr == trigger);

        if is_valid {
            Ok(JohariTransition::new(*self, target, trigger))
        } else {
            Err(format!(
                "Invalid transition: {:?} -> {:?} via {:?}. Valid transitions from {:?}: {:?}",
                self,
                target,
                trigger,
                self,
                self.valid_transitions()
            ))
        }
    }
}

/// Triggers that cause Johari quadrant transitions.
///
/// Each trigger represents a specific event that moves knowledge
/// between quadrants in the Johari Window model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionTrigger {
    /// User explicitly shares hidden knowledge (Hidden → Open).
    ExplicitShare,
    /// Agent recognizes pattern in blind spot (Blind → Open/Hidden).
    SelfRecognition,
    /// Dream consolidation discovers new patterns (Unknown → Open).
    PatternDiscovery,
    /// User marks knowledge as private (Open → Hidden).
    Privatize,
    /// External observation reveals blind spot (Unknown → Blind).
    ExternalObservation,
    /// Dream consolidation surfaces unknown knowledge (Unknown → Open/Hidden).
    DreamConsolidation,
}

impl TransitionTrigger {
    /// Returns a human-readable description of this trigger.
    pub fn description(&self) -> &'static str {
        match self {
            Self::ExplicitShare => "User explicitly shares hidden knowledge",
            Self::SelfRecognition => "Agent recognizes pattern in blind spot",
            Self::PatternDiscovery => "Dream consolidation discovers new patterns",
            Self::Privatize => "User marks knowledge as private",
            Self::ExternalObservation => "External observation reveals blind spot",
            Self::DreamConsolidation => "Dream consolidation surfaces unknown knowledge",
        }
    }

    /// Returns all trigger variants as a fixed-size array.
    pub fn all() -> [TransitionTrigger; 6] {
        [
            Self::ExplicitShare,
            Self::SelfRecognition,
            Self::PatternDiscovery,
            Self::Privatize,
            Self::ExternalObservation,
            Self::DreamConsolidation,
        ]
    }
}

impl fmt::Display for TransitionTrigger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExplicitShare => write!(f, "ExplicitShare"),
            Self::SelfRecognition => write!(f, "SelfRecognition"),
            Self::PatternDiscovery => write!(f, "PatternDiscovery"),
            Self::Privatize => write!(f, "Privatize"),
            Self::ExternalObservation => write!(f, "ExternalObservation"),
            Self::DreamConsolidation => write!(f, "DreamConsolidation"),
        }
    }
}

/// Record of a Johari quadrant transition.
///
/// Captures the complete context of a knowledge reclassification event,
/// including source/target quadrants, trigger, and timestamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JohariTransition {
    /// Starting quadrant.
    pub from: JohariQuadrant,
    /// Ending quadrant.
    pub to: JohariQuadrant,
    /// What triggered this transition.
    pub trigger: TransitionTrigger,
    /// When this transition occurred.
    pub timestamp: DateTime<Utc>,
}

impl JohariTransition {
    /// Create a new transition record with current UTC timestamp.
    ///
    /// # Arguments
    /// * `from` - Source quadrant
    /// * `to` - Target quadrant
    /// * `trigger` - Event that caused the transition
    ///
    /// # Example
    /// ```
    /// use context_graph_core::types::{JohariQuadrant, TransitionTrigger, JohariTransition};
    /// let t = JohariTransition::new(
    ///     JohariQuadrant::Hidden,
    ///     JohariQuadrant::Open,
    ///     TransitionTrigger::ExplicitShare
    /// );
    /// assert_eq!(t.from, JohariQuadrant::Hidden);
    /// assert_eq!(t.to, JohariQuadrant::Open);
    /// ```
    pub fn new(from: JohariQuadrant, to: JohariQuadrant, trigger: TransitionTrigger) -> Self {
        Self {
            from,
            to,
            trigger,
            timestamp: Utc::now(),
        }
    }
}

impl Default for JohariQuadrant {
    /// Default quadrant is Open (most accessible).
    fn default() -> Self {
        Self::Open
    }
}

impl fmt::Display for JohariQuadrant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open => write!(f, "Open"),
            Self::Hidden => write!(f, "Hidden"),
            Self::Blind => write!(f, "Blind"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

impl FromStr for JohariQuadrant {
    type Err = String;

    /// Parses a string into a JohariQuadrant (case-insensitive).
    ///
    /// # Accepted Values
    /// "open", "OPEN", "Open", etc. for each variant
    ///
    /// # Errors
    /// Returns error string if input doesn't match any variant.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "open" => Ok(Self::Open),
            "hidden" => Ok(Self::Hidden),
            "blind" => Ok(Self::Blind),
            "unknown" => Ok(Self::Unknown),
            _ => Err(format!(
                "Invalid JohariQuadrant: '{}'. Valid values: open, hidden, blind, unknown",
                s
            )),
        }
    }
}

/// Input modality classification.
///
/// Classifies the type of content stored in a memory node.
/// Used for embedding model selection and content type detection.
///
/// # Embedding Model Mapping
/// From constitution.yaml Section 12-MODEL EMBEDDING:
/// - Text: E1_Semantic (1024D)
/// - Code: E7_Code (1536D)
/// - Image: E10_Multimodal (1024D)
/// - Audio: E10_Multimodal (1024D)
/// - Structured: E1_Semantic (1024D)
/// - Mixed: E1_Semantic (1024D)
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    /// Plain text content
    #[default]
    Text,
    /// Source code
    Code,
    /// Image data
    Image,
    /// Audio data
    Audio,
    /// Structured data (JSON, XML, etc.)
    Structured,
    /// Mixed modalities
    Mixed,
}

impl Modality {
    /// Detect modality from content string by analyzing patterns.
    ///
    /// # Detection Order (most specific first):
    /// 1. Code patterns (fn, def, class, import, etc.)
    /// 2. Structured data (JSON/YAML markers)
    /// 3. Data URIs (image/audio)
    /// 4. Default: Text
    ///
    /// # Examples
    /// ```
    /// use context_graph_core::types::Modality;
    /// assert_eq!(Modality::detect("fn main() {}"), Modality::Code);
    /// assert_eq!(Modality::detect("{\"key\": 1}"), Modality::Structured);
    /// assert_eq!(Modality::detect("Hello world"), Modality::Text);
    /// ```
    pub fn detect(content: &str) -> Self {
        // Code patterns - case-sensitive, must include space after keyword
        const CODE_PATTERNS: &[&str] = &[
            "fn ",
            "def ",
            "class ",
            "import ",
            "function ",
            "const ",
            "let ",
            "var ",
            "pub ",
            "async ",
            "impl ",
            "struct ",
            "enum ",
            "mod ",
            "use ",
            "package ",
            "func ",
            "export ",
            "from ",
            "#include",
            "#define",
        ];

        for pattern in CODE_PATTERNS {
            if content.contains(pattern) {
                return Self::Code;
            }
        }

        // Structured data detection
        let trimmed = content.trim();
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            return Self::Structured;
        }

        // YAML detection: lines starting with word followed by colon
        if content.lines().any(|line| {
            let t = line.trim();
            !t.is_empty() && !t.starts_with('#') && t.contains(": ")
        }) {
            return Self::Structured;
        }

        // Data URI detection
        if content.starts_with("data:image") {
            return Self::Image;
        }
        if content.starts_with("data:audio") {
            return Self::Audio;
        }

        Self::Text
    }

    /// Returns common file extensions for this modality (lowercase, no dots).
    ///
    /// # Returns
    /// Static slice of extension strings. Empty slice for Mixed modality.
    pub fn file_extensions(&self) -> &'static [&'static str] {
        match self {
            Self::Text => &["txt", "md", "rst", "adoc"],
            Self::Code => &[
                "rs", "py", "js", "ts", "go", "java", "c", "cpp", "h", "rb", "php",
            ],
            Self::Image => &["png", "jpg", "jpeg", "gif", "svg", "webp", "bmp"],
            Self::Audio => &["mp3", "wav", "ogg", "flac", "m4a", "aac"],
            Self::Structured => &["json", "yaml", "yml", "toml", "xml"],
            Self::Mixed => &[],
        }
    }

    /// Returns the primary embedding model ID per PRD spec.
    ///
    /// # Model Mapping (from constitution.yaml Section 12-MODEL EMBEDDING)
    /// - Text: E1_Semantic (1024D)
    /// - Code: E7_Code (1536D)
    /// - Image: E10_Multimodal (1024D)
    /// - Audio: E10_Multimodal (1024D)
    /// - Structured: E1_Semantic (1024D)
    /// - Mixed: E1_Semantic (1024D)
    pub fn primary_embedding_model(&self) -> &'static str {
        match self {
            Self::Text => "E1_Semantic",
            Self::Code => "E7_Code",
            Self::Image => "E10_Multimodal",
            Self::Audio => "E10_Multimodal",
            Self::Structured => "E1_Semantic",
            Self::Mixed => "E1_Semantic",
        }
    }

    /// Returns all modality variants as a fixed-size array.
    pub fn all() -> [Modality; 6] {
        [
            Self::Text,
            Self::Code,
            Self::Image,
            Self::Audio,
            Self::Structured,
            Self::Mixed,
        ]
    }
}

impl fmt::Display for Modality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text => write!(f, "Text"),
            Self::Code => write!(f, "Code"),
            Self::Image => write!(f, "Image"),
            Self::Audio => write!(f, "Audio"),
            Self::Structured => write!(f, "Structured"),
            Self::Mixed => write!(f, "Mixed"),
        }
    }
}

#[cfg(test)]
mod tests {
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
        let cloned = original.clone();
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
    fn test_modality_default() {
        assert_eq!(Modality::default(), Modality::Text);
    }

    // =========================================================================
    // TASK-M02-002: Modality Enum Tests
    // =========================================================================

    #[test]
    fn test_modality_detect_rust_code() {
        assert_eq!(
            Modality::detect("fn main() { println!(\"hello\"); }"),
            Modality::Code
        );
        assert_eq!(
            Modality::detect("pub struct Foo { bar: i32 }"),
            Modality::Code
        );
        assert_eq!(Modality::detect("impl Default for Foo {}"), Modality::Code);
        assert_eq!(
            Modality::detect("use std::collections::HashMap;"),
            Modality::Code
        );
    }

    #[test]
    fn test_modality_detect_python_code() {
        assert_eq!(Modality::detect("def hello(): pass"), Modality::Code);
        assert_eq!(Modality::detect("class Foo: pass"), Modality::Code);
        assert_eq!(Modality::detect("import os"), Modality::Code);
        assert_eq!(Modality::detect("from typing import List"), Modality::Code);
        assert_eq!(Modality::detect("async def fetch(): pass"), Modality::Code);
    }

    #[test]
    fn test_modality_detect_javascript_code() {
        assert_eq!(Modality::detect("function foo() {}"), Modality::Code);
        assert_eq!(Modality::detect("const x = 5;"), Modality::Code);
        assert_eq!(Modality::detect("let y = 10;"), Modality::Code);
        assert_eq!(Modality::detect("var z = 'hello';"), Modality::Code);
        assert_eq!(
            Modality::detect("export default function() {}"),
            Modality::Code
        );
    }

    #[test]
    fn test_modality_detect_structured_json() {
        assert_eq!(
            Modality::detect("{\"key\": \"value\"}"),
            Modality::Structured
        );
        assert_eq!(Modality::detect("[1, 2, 3]"), Modality::Structured);
        assert_eq!(
            Modality::detect("  { \"nested\": { } }"),
            Modality::Structured
        );
    }

    #[test]
    fn test_modality_detect_structured_yaml() {
        assert_eq!(
            Modality::detect("key: value\nother: data"),
            Modality::Structured
        );
        assert_eq!(
            Modality::detect("name: John\nage: 30"),
            Modality::Structured
        );
    }

    #[test]
    fn test_modality_detect_data_uri() {
        assert_eq!(
            Modality::detect("data:image/png;base64,iVBORw0KGg..."),
            Modality::Image
        );
        assert_eq!(
            Modality::detect("data:audio/mp3;base64,SUQz..."),
            Modality::Audio
        );
    }

    #[test]
    fn test_modality_detect_plain_text() {
        assert_eq!(Modality::detect("Hello, world!"), Modality::Text);
        assert_eq!(Modality::detect("This is just a sentence."), Modality::Text);
        assert_eq!(Modality::detect("The quick brown fox"), Modality::Text);
        assert_eq!(Modality::detect(""), Modality::Text);
    }

    #[test]
    fn test_modality_file_extensions() {
        assert!(Modality::Text.file_extensions().contains(&"txt"));
        assert!(Modality::Text.file_extensions().contains(&"md"));

        assert!(Modality::Code.file_extensions().contains(&"rs"));
        assert!(Modality::Code.file_extensions().contains(&"py"));
        assert!(Modality::Code.file_extensions().contains(&"js"));

        assert!(Modality::Image.file_extensions().contains(&"png"));
        assert!(Modality::Image.file_extensions().contains(&"jpg"));

        assert!(Modality::Audio.file_extensions().contains(&"mp3"));
        assert!(Modality::Audio.file_extensions().contains(&"wav"));

        assert!(Modality::Structured.file_extensions().contains(&"json"));
        assert!(Modality::Structured.file_extensions().contains(&"yaml"));

        assert!(Modality::Mixed.file_extensions().is_empty());
    }

    #[test]
    fn test_modality_file_extensions_no_dots() {
        for modality in Modality::all() {
            for ext in modality.file_extensions() {
                assert!(
                    !ext.starts_with('.'),
                    "Extension '{}' should not start with dot",
                    ext
                );
                assert_eq!(
                    *ext,
                    ext.to_lowercase(),
                    "Extension '{}' should be lowercase",
                    ext
                );
            }
        }
    }

    #[test]
    fn test_modality_primary_embedding_model() {
        assert_eq!(Modality::Text.primary_embedding_model(), "E1_Semantic");
        assert_eq!(Modality::Code.primary_embedding_model(), "E7_Code");
        assert_eq!(Modality::Image.primary_embedding_model(), "E10_Multimodal");
        assert_eq!(Modality::Audio.primary_embedding_model(), "E10_Multimodal");
        assert_eq!(
            Modality::Structured.primary_embedding_model(),
            "E1_Semantic"
        );
        assert_eq!(Modality::Mixed.primary_embedding_model(), "E1_Semantic");
    }

    #[test]
    fn test_modality_display() {
        assert_eq!(format!("{}", Modality::Text), "Text");
        assert_eq!(format!("{}", Modality::Code), "Code");
        assert_eq!(format!("{}", Modality::Image), "Image");
        assert_eq!(format!("{}", Modality::Audio), "Audio");
        assert_eq!(format!("{}", Modality::Structured), "Structured");
        assert_eq!(format!("{}", Modality::Mixed), "Mixed");
    }

    #[test]
    fn test_modality_serde_roundtrip() {
        for modality in Modality::all() {
            let json = serde_json::to_string(&modality).expect("serialize failed");
            let parsed: Modality = serde_json::from_str(&json).expect("deserialize failed");
            assert_eq!(modality, parsed, "Roundtrip failed for {:?}", modality);
        }
    }

    #[test]
    fn test_modality_all_variants() {
        let all = Modality::all();
        assert_eq!(all.len(), 6);
        assert!(all.contains(&Modality::Text));
        assert!(all.contains(&Modality::Code));
        assert!(all.contains(&Modality::Image));
        assert!(all.contains(&Modality::Audio));
        assert!(all.contains(&Modality::Structured));
        assert!(all.contains(&Modality::Mixed));
    }

    #[test]
    fn test_modality_clone_copy() {
        let original = Modality::Code;
        let cloned = original.clone();
        let copied = original;
        assert_eq!(original, cloned);
        assert_eq!(original, copied);
    }

    #[test]
    fn test_modality_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        for modality in Modality::all() {
            assert!(set.insert(modality), "Duplicate hash for {:?}", modality);
        }
        assert_eq!(set.len(), 6);
    }

    #[test]
    fn test_modality_detect_empty_string() {
        // Empty string should return Text (default)
        assert_eq!(Modality::detect(""), Modality::Text);
        println!("BEFORE: empty string input");
        println!("AFTER: Modality::Text returned");
    }

    #[test]
    fn test_modality_detect_whitespace_only() {
        assert_eq!(Modality::detect("   \n\t  "), Modality::Text);
        println!("BEFORE: whitespace-only input");
        println!("AFTER: Modality::Text returned");
    }

    #[test]
    fn test_modality_detect_keyword_no_space() {
        // "function" without trailing space should NOT match
        // "functionality" is not code
        assert_eq!(Modality::detect("functionality test"), Modality::Text);
        println!("BEFORE: 'functionality test' (keyword embedded)");
        println!("AFTER: Modality::Text returned (not Code)");
    }

    // =========================================================================
    // TASK-M02-012: Johari Transition Logic Tests
    // =========================================================================

    #[test]
    fn test_transition_trigger_all_variants() {
        let all = TransitionTrigger::all();
        assert_eq!(
            all.len(),
            6,
            "TransitionTrigger should have exactly 6 variants"
        );
        assert!(all.contains(&TransitionTrigger::ExplicitShare));
        assert!(all.contains(&TransitionTrigger::SelfRecognition));
        assert!(all.contains(&TransitionTrigger::PatternDiscovery));
        assert!(all.contains(&TransitionTrigger::Privatize));
        assert!(all.contains(&TransitionTrigger::ExternalObservation));
        assert!(all.contains(&TransitionTrigger::DreamConsolidation));
    }

    #[test]
    fn test_transition_trigger_description_not_empty() {
        for trigger in TransitionTrigger::all() {
            let desc = trigger.description();
            assert!(!desc.is_empty(), "Description empty for {:?}", trigger);
            assert!(desc.len() > 10, "Description too short for {:?}", trigger);
            println!("Trigger {:?} -> '{}'", trigger, desc);
        }
    }

    #[test]
    fn test_transition_trigger_display() {
        assert_eq!(
            format!("{}", TransitionTrigger::ExplicitShare),
            "ExplicitShare"
        );
        assert_eq!(
            format!("{}", TransitionTrigger::SelfRecognition),
            "SelfRecognition"
        );
        assert_eq!(
            format!("{}", TransitionTrigger::PatternDiscovery),
            "PatternDiscovery"
        );
        assert_eq!(format!("{}", TransitionTrigger::Privatize), "Privatize");
        assert_eq!(
            format!("{}", TransitionTrigger::ExternalObservation),
            "ExternalObservation"
        );
        assert_eq!(
            format!("{}", TransitionTrigger::DreamConsolidation),
            "DreamConsolidation"
        );
    }

    #[test]
    fn test_transition_trigger_serde_roundtrip() {
        for trigger in TransitionTrigger::all() {
            let json = serde_json::to_string(&trigger).expect("serialize failed");
            let parsed: TransitionTrigger =
                serde_json::from_str(&json).expect("deserialize failed");
            assert_eq!(trigger, parsed, "Roundtrip failed for {:?}", trigger);
        }
    }

    #[test]
    fn test_transition_trigger_serde_snake_case() {
        // Verify snake_case serialization per constitution.yaml requirement
        assert_eq!(
            serde_json::to_string(&TransitionTrigger::ExplicitShare).unwrap(),
            "\"explicit_share\""
        );
        assert_eq!(
            serde_json::to_string(&TransitionTrigger::SelfRecognition).unwrap(),
            "\"self_recognition\""
        );
        assert_eq!(
            serde_json::to_string(&TransitionTrigger::PatternDiscovery).unwrap(),
            "\"pattern_discovery\""
        );
        assert_eq!(
            serde_json::to_string(&TransitionTrigger::Privatize).unwrap(),
            "\"privatize\""
        );
        assert_eq!(
            serde_json::to_string(&TransitionTrigger::ExternalObservation).unwrap(),
            "\"external_observation\""
        );
        assert_eq!(
            serde_json::to_string(&TransitionTrigger::DreamConsolidation).unwrap(),
            "\"dream_consolidation\""
        );
    }

    #[test]
    fn test_transition_trigger_copy() {
        let original = TransitionTrigger::ExplicitShare;
        let copied = original; // Copy trait
        let cloned = original.clone();
        assert_eq!(original, copied);
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_open_to_hidden_via_privatize() {
        let result = JohariQuadrant::Open
            .transition_to(JohariQuadrant::Hidden, TransitionTrigger::Privatize);
        assert!(
            result.is_ok(),
            "Open -> Hidden via Privatize should succeed"
        );
        let transition = result.unwrap();
        assert_eq!(transition.from, JohariQuadrant::Open);
        assert_eq!(transition.to, JohariQuadrant::Hidden);
        assert_eq!(transition.trigger, TransitionTrigger::Privatize);
        println!("Open -> Hidden: {:?}", transition);
    }

    #[test]
    fn test_open_cannot_go_to_blind() {
        let result = JohariQuadrant::Open
            .transition_to(JohariQuadrant::Blind, TransitionTrigger::SelfRecognition);
        assert!(result.is_err(), "Open -> Blind should fail");
        let err = result.unwrap_err();
        assert!(
            err.contains("Invalid transition"),
            "Error should mention 'Invalid transition': {}",
            err
        );
        println!("Open -> Blind error: {}", err);
    }

    #[test]
    fn test_open_cannot_go_to_unknown() {
        let result = JohariQuadrant::Open.transition_to(
            JohariQuadrant::Unknown,
            TransitionTrigger::DreamConsolidation,
        );
        assert!(result.is_err(), "Open -> Unknown should fail");
        let err = result.unwrap_err();
        assert!(err.contains("Invalid transition"));
        println!("Open -> Unknown error: {}", err);
    }

    #[test]
    fn test_hidden_to_open_via_explicit_share() {
        let result = JohariQuadrant::Hidden
            .transition_to(JohariQuadrant::Open, TransitionTrigger::ExplicitShare);
        assert!(
            result.is_ok(),
            "Hidden -> Open via ExplicitShare should succeed"
        );
        let transition = result.unwrap();
        assert_eq!(transition.from, JohariQuadrant::Hidden);
        assert_eq!(transition.to, JohariQuadrant::Open);
        assert_eq!(transition.trigger, TransitionTrigger::ExplicitShare);
        println!("Hidden -> Open: {:?}", transition);
    }

    #[test]
    fn test_hidden_cannot_go_to_blind() {
        let result = JohariQuadrant::Hidden
            .transition_to(JohariQuadrant::Blind, TransitionTrigger::SelfRecognition);
        assert!(result.is_err(), "Hidden -> Blind should fail");
        println!("Hidden -> Blind error: {}", result.unwrap_err());
    }

    #[test]
    fn test_hidden_cannot_go_to_unknown() {
        let result = JohariQuadrant::Hidden.transition_to(
            JohariQuadrant::Unknown,
            TransitionTrigger::DreamConsolidation,
        );
        assert!(result.is_err(), "Hidden -> Unknown should fail");
        println!("Hidden -> Unknown error: {}", result.unwrap_err());
    }

    #[test]
    fn test_blind_to_open_via_self_recognition() {
        let result = JohariQuadrant::Blind
            .transition_to(JohariQuadrant::Open, TransitionTrigger::SelfRecognition);
        assert!(
            result.is_ok(),
            "Blind -> Open via SelfRecognition should succeed"
        );
        let transition = result.unwrap();
        assert_eq!(transition.from, JohariQuadrant::Blind);
        assert_eq!(transition.to, JohariQuadrant::Open);
        assert_eq!(transition.trigger, TransitionTrigger::SelfRecognition);
        println!("Blind -> Open: {:?}", transition);
    }

    #[test]
    fn test_blind_to_hidden_via_self_recognition() {
        let result = JohariQuadrant::Blind
            .transition_to(JohariQuadrant::Hidden, TransitionTrigger::SelfRecognition);
        assert!(
            result.is_ok(),
            "Blind -> Hidden via SelfRecognition should succeed"
        );
        let transition = result.unwrap();
        assert_eq!(transition.from, JohariQuadrant::Blind);
        assert_eq!(transition.to, JohariQuadrant::Hidden);
        assert_eq!(transition.trigger, TransitionTrigger::SelfRecognition);
        println!("Blind -> Hidden: {:?}", transition);
    }

    #[test]
    fn test_blind_cannot_go_to_unknown() {
        let result = JohariQuadrant::Blind.transition_to(
            JohariQuadrant::Unknown,
            TransitionTrigger::DreamConsolidation,
        );
        assert!(result.is_err(), "Blind -> Unknown should fail");
        println!("Blind -> Unknown error: {}", result.unwrap_err());
    }

    #[test]
    fn test_unknown_to_open_via_dream_consolidation() {
        let result = JohariQuadrant::Unknown
            .transition_to(JohariQuadrant::Open, TransitionTrigger::DreamConsolidation);
        assert!(
            result.is_ok(),
            "Unknown -> Open via DreamConsolidation should succeed"
        );
        let transition = result.unwrap();
        assert_eq!(transition.from, JohariQuadrant::Unknown);
        assert_eq!(transition.to, JohariQuadrant::Open);
        assert_eq!(transition.trigger, TransitionTrigger::DreamConsolidation);
        println!("Unknown -> Open via DreamConsolidation: {:?}", transition);
    }

    #[test]
    fn test_unknown_to_open_via_pattern_discovery() {
        let result = JohariQuadrant::Unknown
            .transition_to(JohariQuadrant::Open, TransitionTrigger::PatternDiscovery);
        assert!(
            result.is_ok(),
            "Unknown -> Open via PatternDiscovery should succeed"
        );
        let transition = result.unwrap();
        assert_eq!(transition.from, JohariQuadrant::Unknown);
        assert_eq!(transition.to, JohariQuadrant::Open);
        assert_eq!(transition.trigger, TransitionTrigger::PatternDiscovery);
        println!("Unknown -> Open via PatternDiscovery: {:?}", transition);
    }

    #[test]
    fn test_unknown_to_hidden_via_dream_consolidation() {
        let result = JohariQuadrant::Unknown.transition_to(
            JohariQuadrant::Hidden,
            TransitionTrigger::DreamConsolidation,
        );
        assert!(
            result.is_ok(),
            "Unknown -> Hidden via DreamConsolidation should succeed"
        );
        let transition = result.unwrap();
        assert_eq!(transition.from, JohariQuadrant::Unknown);
        assert_eq!(transition.to, JohariQuadrant::Hidden);
        assert_eq!(transition.trigger, TransitionTrigger::DreamConsolidation);
        println!("Unknown -> Hidden via DreamConsolidation: {:?}", transition);
    }

    #[test]
    fn test_unknown_to_blind_via_external_observation() {
        let result = JohariQuadrant::Unknown.transition_to(
            JohariQuadrant::Blind,
            TransitionTrigger::ExternalObservation,
        );
        assert!(
            result.is_ok(),
            "Unknown -> Blind via ExternalObservation should succeed"
        );
        let transition = result.unwrap();
        assert_eq!(transition.from, JohariQuadrant::Unknown);
        assert_eq!(transition.to, JohariQuadrant::Blind);
        assert_eq!(transition.trigger, TransitionTrigger::ExternalObservation);
        println!("Unknown -> Blind via ExternalObservation: {:?}", transition);
    }

    #[test]
    fn test_no_self_transitions() {
        for quadrant in JohariQuadrant::all() {
            // Try all triggers for self-transition
            for trigger in TransitionTrigger::all() {
                let result = quadrant.transition_to(quadrant, trigger);
                assert!(
                    result.is_err(),
                    "Self-transition {:?} -> {:?} should fail",
                    quadrant,
                    quadrant
                );
                let err = result.unwrap_err();
                assert!(
                    err.contains("Cannot transition to same quadrant"),
                    "Error should mention 'same quadrant': {}",
                    err
                );
            }
            println!("VERIFIED: {:?} cannot transition to itself", quadrant);
        }
    }

    #[test]
    fn test_invalid_trigger_rejected() {
        // Valid target (Hidden) but wrong trigger (not Privatize)
        let result = JohariQuadrant::Open
            .transition_to(JohariQuadrant::Hidden, TransitionTrigger::ExplicitShare);
        assert!(
            result.is_err(),
            "Open -> Hidden via ExplicitShare should fail (wrong trigger)"
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("Invalid transition"),
            "Error should mention 'Invalid transition': {}",
            err
        );
        println!("Wrong trigger rejected: {}", err);
    }

    #[test]
    fn test_transition_creates_record() {
        let transition = JohariTransition::new(
            JohariQuadrant::Hidden,
            JohariQuadrant::Open,
            TransitionTrigger::ExplicitShare,
        );
        assert_eq!(transition.from, JohariQuadrant::Hidden);
        assert_eq!(transition.to, JohariQuadrant::Open);
        assert_eq!(transition.trigger, TransitionTrigger::ExplicitShare);
        // Timestamp should be roughly now (within 1 second)
        let now = chrono::Utc::now();
        let diff = (now - transition.timestamp).num_seconds().abs();
        assert!(
            diff < 2,
            "Timestamp should be within 2 seconds of now, got diff={}",
            diff
        );
        println!("Transition record: {:?}", transition);
    }

    #[test]
    fn test_valid_transitions_count() {
        // Open: 1 (-> Hidden)
        assert_eq!(
            JohariQuadrant::Open.valid_transitions().len(),
            1,
            "Open should have 1 valid transition"
        );

        // Hidden: 1 (-> Open)
        assert_eq!(
            JohariQuadrant::Hidden.valid_transitions().len(),
            1,
            "Hidden should have 1 valid transition"
        );

        // Blind: 2 (-> Open, -> Hidden)
        assert_eq!(
            JohariQuadrant::Blind.valid_transitions().len(),
            2,
            "Blind should have 2 valid transitions"
        );

        // Unknown: 4 (-> Open×2, -> Hidden, -> Blind)
        assert_eq!(
            JohariQuadrant::Unknown.valid_transitions().len(),
            4,
            "Unknown should have 4 valid transitions"
        );

        println!(
            "Transition counts: Open={}, Hidden={}, Blind={}, Unknown={}",
            JohariQuadrant::Open.valid_transitions().len(),
            JohariQuadrant::Hidden.valid_transitions().len(),
            JohariQuadrant::Blind.valid_transitions().len(),
            JohariQuadrant::Unknown.valid_transitions().len()
        );
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
    fn test_johari_transition_serde_roundtrip() {
        let transition = JohariTransition::new(
            JohariQuadrant::Hidden,
            JohariQuadrant::Open,
            TransitionTrigger::ExplicitShare,
        );
        let json = serde_json::to_string(&transition).expect("serialize failed");
        let parsed: JohariTransition = serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(transition.from, parsed.from);
        assert_eq!(transition.to, parsed.to);
        assert_eq!(transition.trigger, parsed.trigger);
        // Timestamps might differ slightly due to precision, just check they're close
        let diff = (transition.timestamp - parsed.timestamp)
            .num_milliseconds()
            .abs();
        assert!(diff < 1000, "Timestamps should be within 1 second");
        println!("JohariTransition roundtrip JSON: {}", json);
    }

    #[test]
    fn test_johari_transition_is_clone_not_copy() {
        let original = JohariTransition::new(
            JohariQuadrant::Open,
            JohariQuadrant::Hidden,
            TransitionTrigger::Privatize,
        );
        let cloned = original.clone();
        assert_eq!(original.from, cloned.from);
        assert_eq!(original.to, cloned.to);
        assert_eq!(original.trigger, cloned.trigger);
        // Note: JohariTransition is Clone but NOT Copy (contains DateTime)
    }

    #[test]
    fn test_transition_trigger_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        for trigger in TransitionTrigger::all() {
            assert!(set.insert(trigger), "Duplicate hash for {:?}", trigger);
        }
        assert_eq!(set.len(), 6);
    }

    #[test]
    fn test_all_valid_unknown_transitions() {
        // Unknown has the most transitions (4), test all of them work
        let transitions = JohariQuadrant::Unknown.valid_transitions();
        for (target, trigger) in transitions.iter() {
            let result = JohariQuadrant::Unknown.transition_to(*target, *trigger);
            assert!(
                result.is_ok(),
                "Unknown -> {:?} via {:?} should succeed",
                target,
                trigger
            );
            println!("Unknown -> {:?} via {:?}: OK", target, trigger);
        }
    }

    #[test]
    fn test_boundary_minimum_transitions() {
        // Open and Hidden have minimum (1) valid transition
        let open_transitions = JohariQuadrant::Open.valid_transitions();
        assert_eq!(open_transitions.len(), 1);

        // Try all 4 possible targets from Open
        for target in JohariQuadrant::all() {
            let result = JohariQuadrant::Open.can_transition_to(target);
            if target == JohariQuadrant::Hidden {
                assert!(result, "Open should be able to transition to Hidden");
            } else {
                assert!(
                    !result,
                    "Open should NOT be able to transition to {:?}",
                    target
                );
            }
            println!(
                "Open -> {:?}: {}",
                target,
                if result { "VALID" } else { "INVALID" }
            );
        }
    }
}
