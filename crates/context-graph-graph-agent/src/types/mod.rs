//! Core types for graph relationship discovery.
//!
//! # Graph Analysis Types
//!
//! This module defines the core types used in the graph relationship
//! discovery pipeline:
//!
//! - [`GraphAnalysisResult`]: LLM output for relationship analysis
//! - [`GraphLinkDirection`]: Direction of the graph connection
//! - [`RelationshipType`]: The 8 supported relationship types
//! - [`GraphCandidate`]: Candidate pair for LLM analysis
//! - [`MemoryForGraphAnalysis`]: Memory with graph-relevant metadata

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Result of LLM-based graph relationship analysis.
///
/// Contains the analysis of whether two memories have a structural
/// relationship and what type of relationship it is.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAnalysisResult {
    /// Whether a graph connection exists between the two memories.
    pub has_connection: bool,

    /// Direction of the graph connection.
    pub direction: GraphLinkDirection,

    /// Type of relationship detected.
    pub relationship_type: RelationshipType,

    /// Confidence score [0.0, 1.0].
    pub confidence: f32,

    /// Human-readable description of the relationship.
    pub description: String,

    /// Raw LLM response (for debugging).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<String>,
}

impl Default for GraphAnalysisResult {
    fn default() -> Self {
        Self {
            has_connection: false,
            direction: GraphLinkDirection::NoConnection,
            relationship_type: RelationshipType::None,
            confidence: 0.0,
            description: String::new(),
            raw_response: None,
        }
    }
}

/// Direction of graph relationship between two entities.
///
/// Following the E8 asymmetric pattern (ARCH-15):
/// - Source: Entity that points to another (outgoing relationship)
/// - Target: Entity that is pointed to (incoming relationship)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GraphLinkDirection {
    /// A connects to B (A is source, B is target).
    /// Example: "Module A imports Module B"
    AConnectsB,

    /// B connects to A (B is source, A is target).
    /// Example: "Module B imports Module A"
    BConnectsA,

    /// Bidirectional connection.
    /// Example: "Module A and Module B reference each other"
    Bidirectional,

    /// No connection detected.
    #[default]
    NoConnection,
}

impl GraphLinkDirection {
    /// Parse direction from string (case-insensitive).
    pub fn from_str(s: &str) -> Self {
        let lower = s.to_lowercase();
        let trimmed = lower.trim();

        match trimmed {
            "a_connects_b" | "a connects b" | "a->b" | "a_to_b" | "forward" | "source" => {
                Self::AConnectsB
            }
            "b_connects_a" | "b connects a" | "b->a" | "b_to_a" | "reverse" | "target" => {
                Self::BConnectsA
            }
            "bidirectional" | "both" | "mutual" | "symmetric" | "a<->b" => Self::Bidirectional,
            _ => Self::NoConnection,
        }
    }

    /// Check if this direction indicates a connection exists.
    pub fn is_connected(&self) -> bool {
        !matches!(self, Self::NoConnection)
    }

    /// Get the direction as a string for LLM output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AConnectsB => "a_connects_b",
            Self::BConnectsA => "b_connects_a",
            Self::Bidirectional => "bidirectional",
            Self::NoConnection => "none",
        }
    }
}

/// The 8 supported relationship types for graph connections.
///
/// These cover both code relationships and documentation patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum RelationshipType {
    /// Import/use relationship.
    /// Example: "use crate::module;"
    Imports,

    /// Dependency relationship.
    /// Example: "requires PostgreSQL"
    DependsOn,

    /// Reference relationship.
    /// Example: "See also: [link]"
    References,

    /// Function call relationship.
    /// Example: "function A calls function B"
    Calls,

    /// Implementation relationship.
    /// Example: "impl Trait for Struct"
    Implements,

    /// Extension/inheritance relationship.
    /// Example: "extends BaseClass"
    Extends,

    /// Containment relationship.
    /// Example: "module contains function"
    Contains,

    /// Usage relationship.
    /// Example: "used by client code"
    UsedBy,

    /// No relationship detected.
    #[default]
    None,
}

impl RelationshipType {
    /// Parse relationship type from string (case-insensitive).
    pub fn from_str(s: &str) -> Self {
        let lower = s.to_lowercase();
        let trimmed = lower.trim();

        match trimmed {
            "imports" | "import" | "uses" | "use" => Self::Imports,
            "depends_on" | "depends on" | "dependency" | "requires" => Self::DependsOn,
            "references" | "reference" | "refs" | "see" | "links" | "links_to" => Self::References,
            "calls" | "call" | "invokes" | "invoke" => Self::Calls,
            "implements" | "implement" | "impl" => Self::Implements,
            "extends" | "extend" | "inherits" | "inherit" => Self::Extends,
            "contains" | "contain" | "has" | "includes" => Self::Contains,
            "used_by" | "used by" | "consumed_by" | "consumed by" => Self::UsedBy,
            _ => Self::None,
        }
    }

    /// Get the relationship type as a string for LLM output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Imports => "imports",
            Self::DependsOn => "depends_on",
            Self::References => "references",
            Self::Calls => "calls",
            Self::Implements => "implements",
            Self::Extends => "extends",
            Self::Contains => "contains",
            Self::UsedBy => "used_by",
            Self::None => "none",
        }
    }

    /// All valid relationship types (excluding None).
    pub fn all() -> &'static [Self] {
        &[
            Self::Imports,
            Self::DependsOn,
            Self::References,
            Self::Calls,
            Self::Implements,
            Self::Extends,
            Self::Contains,
            Self::UsedBy,
        ]
    }
}

/// Candidate pair for graph relationship analysis.
///
/// Represents two memories that may have a structural relationship,
/// identified by the scanner and awaiting LLM analysis.
#[derive(Debug, Clone)]
pub struct GraphCandidate {
    /// First memory ID.
    pub memory_a_id: Uuid,

    /// First memory content.
    pub memory_a_content: String,

    /// Second memory ID.
    pub memory_b_id: Uuid,

    /// Second memory content.
    pub memory_b_content: String,

    /// Initial heuristic score from scanner [0.0, 1.0].
    pub initial_score: f32,

    /// Timestamp of memory A.
    pub memory_a_timestamp: DateTime<Utc>,

    /// Timestamp of memory B.
    pub memory_b_timestamp: DateTime<Utc>,

    /// Suspected relationship types based on heuristics.
    pub suspected_types: Vec<RelationshipType>,
}

impl GraphCandidate {
    /// Create a new graph candidate.
    pub fn new(
        memory_a_id: Uuid,
        memory_a_content: String,
        memory_b_id: Uuid,
        memory_b_content: String,
    ) -> Self {
        Self {
            memory_a_id,
            memory_a_content,
            memory_b_id,
            memory_b_content,
            initial_score: 0.0,
            memory_a_timestamp: Utc::now(),
            memory_b_timestamp: Utc::now(),
            suspected_types: Vec::new(),
        }
    }
}

/// Memory with metadata for graph analysis.
#[derive(Debug, Clone)]
pub struct MemoryForGraphAnalysis {
    /// Memory UUID.
    pub id: Uuid,

    /// Memory content text.
    pub content: String,

    /// Creation timestamp.
    pub created_at: DateTime<Utc>,

    /// Session ID if available.
    pub session_id: Option<String>,

    /// E1 semantic embedding for similarity clustering.
    pub e1_embedding: Vec<f32>,

    /// Source type (e.g., "HookDescription", "CodeEntity").
    pub source_type: Option<String>,

    /// File path if this memory is from code.
    pub file_path: Option<String>,
}

/// Markers for detecting graph relationships heuristically.
pub struct GraphMarkers;

impl GraphMarkers {
    /// Import/use markers.
    pub const IMPORT_MARKERS: &'static [&'static str] = &[
        "import",
        "use",
        "require",
        "include",
        "from",
        "extern crate",
        "mod",
    ];

    /// Dependency markers.
    pub const DEPENDENCY_MARKERS: &'static [&'static str] =
        &["depends on", "requires", "needs", "dependency", "prerequisite"];

    /// Reference markers.
    pub const REFERENCE_MARKERS: &'static [&'static str] = &[
        "see:",
        "ref:",
        "see also",
        "links to",
        "references",
        "related to",
        "http://",
        "https://",
    ];

    /// Call markers.
    pub const CALL_MARKERS: &'static [&'static str] =
        &["calls", "invokes", "executes", "runs", "triggers", "fn ", "()"];

    /// Implementation markers.
    pub const IMPL_MARKERS: &'static [&'static str] =
        &["impl", "implements", "implementation", "trait"];

    /// Extension markers.
    pub const EXTEND_MARKERS: &'static [&'static str] =
        &["extends", "inherits", "derives", "subclass", "child of"];

    /// Containment markers.
    pub const CONTAIN_MARKERS: &'static [&'static str] =
        &["contains", "includes", "has", "module", "struct", "enum", "pub mod"];

    /// Used-by markers.
    pub const USED_BY_MARKERS: &'static [&'static str] =
        &["used by", "consumed by", "client of", "caller"];

    /// Count markers in content for a specific relationship type.
    pub fn count_markers_for_type(content: &str, rel_type: RelationshipType) -> usize {
        let markers = match rel_type {
            RelationshipType::Imports => Self::IMPORT_MARKERS,
            RelationshipType::DependsOn => Self::DEPENDENCY_MARKERS,
            RelationshipType::References => Self::REFERENCE_MARKERS,
            RelationshipType::Calls => Self::CALL_MARKERS,
            RelationshipType::Implements => Self::IMPL_MARKERS,
            RelationshipType::Extends => Self::EXTEND_MARKERS,
            RelationshipType::Contains => Self::CONTAIN_MARKERS,
            RelationshipType::UsedBy => Self::USED_BY_MARKERS,
            RelationshipType::None => return 0,
        };

        let lower = content.to_lowercase();
        markers.iter().filter(|m| lower.contains(*m)).count()
    }

    /// Get suspected relationship types based on content markers.
    pub fn detect_suspected_types(content: &str) -> Vec<RelationshipType> {
        let mut types = Vec::new();

        for rel_type in RelationshipType::all() {
            if Self::count_markers_for_type(content, *rel_type) > 0 {
                types.push(*rel_type);
            }
        }

        types
    }

    /// Count all graph-related markers in content.
    pub fn count_all_markers(content: &str) -> usize {
        RelationshipType::all()
            .iter()
            .map(|t| Self::count_markers_for_type(content, *t))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direction_from_str() {
        assert_eq!(
            GraphLinkDirection::from_str("a_connects_b"),
            GraphLinkDirection::AConnectsB
        );
        assert_eq!(
            GraphLinkDirection::from_str("B->A"),
            GraphLinkDirection::BConnectsA
        );
        assert_eq!(
            GraphLinkDirection::from_str("bidirectional"),
            GraphLinkDirection::Bidirectional
        );
        assert_eq!(
            GraphLinkDirection::from_str("invalid"),
            GraphLinkDirection::NoConnection
        );
    }

    #[test]
    fn test_relationship_type_from_str() {
        assert_eq!(
            RelationshipType::from_str("imports"),
            RelationshipType::Imports
        );
        assert_eq!(
            RelationshipType::from_str("DEPENDS_ON"),
            RelationshipType::DependsOn
        );
        assert_eq!(
            RelationshipType::from_str("calls"),
            RelationshipType::Calls
        );
        assert_eq!(
            RelationshipType::from_str("invalid"),
            RelationshipType::None
        );
    }

    #[test]
    fn test_marker_detection() {
        let content = "use crate::module; impl Trait for Struct";
        let types = GraphMarkers::detect_suspected_types(content);
        assert!(types.contains(&RelationshipType::Imports));
        assert!(types.contains(&RelationshipType::Implements));
    }
}
