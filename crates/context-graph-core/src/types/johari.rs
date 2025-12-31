//! Johari Window quadrant classification for memory categorization.
//!
//! The Johari Window is a psychological model adapted for knowledge graph
//! classification. Each quadrant determines how memories are retrieved and
//! weighted in the UTL (Unified Theory of Learning) system.

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
}
