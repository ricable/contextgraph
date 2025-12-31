//! Johari Window quadrant classification and modality types.

use serde::{Deserialize, Serialize};

/// Johari Window quadrant classification.
///
/// The Johari Window is a psychological model for understanding self-awareness
/// and interpersonal relationships. In the context graph, it classifies memories
/// based on their visibility and awareness states.
///
/// # Quadrants
///
/// - **Open**: Known to self and others - readily accessible knowledge
/// - **Blind**: Known to others, unknown to self - insights from external feedback
/// - **Hidden**: Known to self, hidden from others - private knowledge
/// - **Unknown**: Unknown to both self and others - exploration frontier
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum JohariQuadrant {
    /// Known to self and others
    Open,
    /// Known to others, unknown to self
    Blind,
    /// Known to self, hidden from others
    Hidden,
    /// Unknown to both self and others
    #[default]
    Unknown,
}

impl JohariQuadrant {
    /// Returns true if this quadrant represents known information.
    pub fn is_known(&self) -> bool {
        matches!(self, Self::Open | Self::Hidden)
    }

    /// Returns true if this quadrant is visible to others.
    pub fn is_visible(&self) -> bool {
        matches!(self, Self::Open | Self::Blind)
    }

    /// Returns true if quadrant is self-aware (Open or Hidden).
    pub fn is_self_aware(&self) -> bool {
        matches!(self, Self::Open | Self::Hidden)
    }

    /// Returns true if quadrant is observable by others (Open or Blind).
    pub fn is_other_aware(&self) -> bool {
        matches!(self, Self::Open | Self::Blind)
    }

    /// Default retrieval weight for search operations.
    /// Open=1.0 (full weight), Hidden=0.3 (reduced), Blind=0.7, Unknown=0.5
    pub fn default_retrieval_weight(&self) -> f32 {
        match self {
            Self::Open => 1.0,
            Self::Hidden => 0.3,
            Self::Blind => 0.7,
            Self::Unknown => 0.5,
        }
    }

    /// Whether to include in default context retrieval.
    pub fn include_in_default_context(&self) -> bool {
        matches!(self, Self::Open | Self::Blind)
    }

    /// Human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Open => "Known to self and others - readily accessible knowledge",
            Self::Hidden => "Known to self, hidden from others - private knowledge",
            Self::Blind => "Known to others, unknown to self - insights from external feedback",
            Self::Unknown => "Unknown to both self and others - exploration frontier",
        }
    }

    /// RocksDB column family name for storage partitioning.
    pub fn column_family(&self) -> &'static str {
        match self {
            Self::Open => "johari_open",
            Self::Hidden => "johari_hidden",
            Self::Blind => "johari_blind",
            Self::Unknown => "johari_unknown",
        }
    }

    /// All quadrants for iteration.
    pub fn all() -> &'static [JohariQuadrant] {
        &[Self::Open, Self::Hidden, Self::Blind, Self::Unknown]
    }
}

impl std::fmt::Display for JohariQuadrant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open => write!(f, "Open"),
            Self::Hidden => write!(f, "Hidden"),
            Self::Blind => write!(f, "Blind"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

impl std::str::FromStr for JohariQuadrant {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "open" => Ok(Self::Open),
            "hidden" => Ok(Self::Hidden),
            "blind" => Ok(Self::Blind),
            "unknown" => Ok(Self::Unknown),
            _ => Err(format!("Unknown quadrant: {}", s)),
        }
    }
}

/// Input modality classification.
///
/// Classifies the type of content stored in a memory node.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_johari_default() {
        assert_eq!(JohariQuadrant::default(), JohariQuadrant::Unknown);
    }

    #[test]
    fn test_johari_is_known() {
        assert!(JohariQuadrant::Open.is_known());
        assert!(JohariQuadrant::Hidden.is_known());
        assert!(!JohariQuadrant::Blind.is_known());
        assert!(!JohariQuadrant::Unknown.is_known());
    }

    #[test]
    fn test_modality_default() {
        assert_eq!(Modality::default(), Modality::Text);
    }
}
