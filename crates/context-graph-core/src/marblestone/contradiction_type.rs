//! Contradiction type classification for graph edges.
//!
//! # M04-T26: NT Modulation for Contradiction Detection
//!
//! Defines the different types of contradictions that can exist
//! between knowledge nodes, with severity scoring for prioritization.
//!
//! # Constitution Reference
//! - edge_model.attrs: type includes Contradicts
//! - Contradiction edges use inhibitory-heavy NT weights
//!
//! # Example
//!
//! ```rust
//! use context_graph_core::marblestone::ContradictionType;
//!
//! let ct = ContradictionType::DirectOpposition;
//! assert_eq!(ct.severity(), 1.0);
//! assert!(ct.severity() > ContradictionType::LogicalInconsistency.severity());
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

/// Classification of contradiction types.
///
/// Each type has a severity score for prioritization during conflict resolution.
/// Higher severity indicates more critical contradictions that should be
/// addressed with higher priority.
///
/// # Severity Scores
///
/// - DirectOpposition (1.0): Most severe - direct negation
/// - LogicalInconsistency (0.8): Indirect logical conflict
/// - TemporalConflict (0.7): Timeline inconsistencies
/// - CausalConflict (0.6): Conflicting cause-effect chains
///
/// # Constitution Reference
///
/// Used in conjunction with `EdgeType::Contradicts` to classify
/// the nature of contradictions between knowledge nodes.
///
/// # Example
///
/// ```rust
/// use context_graph_core::marblestone::ContradictionType;
///
/// // Create and inspect contradiction types
/// let direct = ContradictionType::DirectOpposition;
/// let logical = ContradictionType::LogicalInconsistency;
///
/// // Compare severities
/// assert!(direct.severity() > logical.severity());
///
/// // Get description
/// assert!(direct.description().contains("opposition"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContradictionType {
    /// Direct logical opposition (A vs not-A).
    ///
    /// The most severe contradiction type - one statement
    /// directly negates another.
    ///
    /// Examples:
    /// - "The file exists" vs "The file does not exist"
    /// - "User is authenticated" vs "User is not authenticated"
    ///
    /// Severity: 1.0
    DirectOpposition,

    /// Logical inconsistency (A implies B, but B is false).
    ///
    /// Indirect contradiction through logical inference.
    /// Requires reasoning to detect the conflict.
    ///
    /// Examples:
    /// - "All users must be authenticated" + "Anonymous access is allowed"
    /// - "Function always returns positive" + "Function can return -1 on error"
    ///
    /// Severity: 0.8
    LogicalInconsistency,

    /// Temporal conflict (event order inconsistency).
    ///
    /// Timeline conflicts where events cannot coexist
    /// in the stated temporal order.
    ///
    /// Examples:
    /// - "File created at T1" + "File accessed at T0 (before creation)"
    /// - "User registered before first login" + "First login predates registration"
    ///
    /// Severity: 0.7
    TemporalConflict,

    /// Causal conflict (conflicting cause-effect chains).
    ///
    /// Two statements imply contradictory causal relationships.
    ///
    /// Examples:
    /// - "A causes B" + "B prevents A"
    /// - "Cache miss causes DB query" + "DB query happens before cache check"
    ///
    /// Severity: 0.6
    CausalConflict,
}

impl ContradictionType {
    /// Get human-readable description of this contradiction type.
    ///
    /// # Returns
    ///
    /// Static string describing the contradiction type.
    #[inline]
    pub fn description(&self) -> &'static str {
        match self {
            Self::DirectOpposition => "Direct logical opposition (A vs not-A)",
            Self::LogicalInconsistency => "Logical inconsistency (implies contradiction)",
            Self::TemporalConflict => "Temporal sequence conflict",
            Self::CausalConflict => "Causal relationship conflict",
        }
    }

    /// Get severity weight for this contradiction type.
    ///
    /// Higher severity = more critical contradiction that should
    /// be prioritized for resolution.
    ///
    /// # Returns
    ///
    /// - DirectOpposition: 1.0
    /// - LogicalInconsistency: 0.8
    /// - TemporalConflict: 0.7
    /// - CausalConflict: 0.6
    ///
    /// # Example
    ///
    /// ```rust
    /// use context_graph_core::marblestone::ContradictionType;
    ///
    /// assert_eq!(ContradictionType::DirectOpposition.severity(), 1.0);
    /// assert_eq!(ContradictionType::CausalConflict.severity(), 0.6);
    /// ```
    #[inline]
    pub fn severity(&self) -> f32 {
        match self {
            Self::DirectOpposition => 1.0,
            Self::LogicalInconsistency => 0.8,
            Self::TemporalConflict => 0.7,
            Self::CausalConflict => 0.6,
        }
    }

    /// Get all contradiction types as an array.
    ///
    /// # Returns
    ///
    /// Array containing all 4 contradiction type variants.
    #[inline]
    pub fn all() -> [ContradictionType; 4] {
        [
            Self::DirectOpposition,
            Self::LogicalInconsistency,
            Self::TemporalConflict,
            Self::CausalConflict,
        ]
    }

    /// Convert to u8 for compact storage.
    ///
    /// # Returns
    ///
    /// - DirectOpposition: 0
    /// - LogicalInconsistency: 1
    /// - TemporalConflict: 2
    /// - CausalConflict: 3
    #[inline]
    pub fn as_u8(&self) -> u8 {
        match self {
            Self::DirectOpposition => 0,
            Self::LogicalInconsistency => 1,
            Self::TemporalConflict => 2,
            Self::CausalConflict => 3,
        }
    }

    /// Convert from u8.
    ///
    /// # Arguments
    ///
    /// * `value` - The u8 representation (0-3)
    ///
    /// # Returns
    ///
    /// - `Some(ContradictionType)` if value is 0-3
    /// - `None` if value is out of range (FAIL FAST)
    #[inline]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::DirectOpposition),
            1 => Some(Self::LogicalInconsistency),
            2 => Some(Self::TemporalConflict),
            3 => Some(Self::CausalConflict),
            _ => None,
        }
    }

    /// Check if this is a high-severity contradiction.
    ///
    /// High severity is defined as severity >= 0.8.
    ///
    /// # Returns
    ///
    /// `true` for DirectOpposition and LogicalInconsistency.
    #[inline]
    pub fn is_high_severity(&self) -> bool {
        self.severity() >= 0.8
    }

    /// Get the recommended action for this contradiction type.
    ///
    /// # Returns
    ///
    /// Static string with recommended resolution approach.
    #[inline]
    pub fn recommended_action(&self) -> &'static str {
        match self {
            Self::DirectOpposition => "Immediate resolution required - choose one statement",
            Self::LogicalInconsistency => "Review logical chain for errors",
            Self::TemporalConflict => "Verify timestamps and event ordering",
            Self::CausalConflict => "Analyze causal relationships for correctness",
        }
    }
}

impl Default for ContradictionType {
    /// Default to DirectOpposition (most common and severe case).
    #[inline]
    fn default() -> Self {
        Self::DirectOpposition
    }
}

impl fmt::Display for ContradictionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::DirectOpposition => "direct_opposition",
            Self::LogicalInconsistency => "logical_inconsistency",
            Self::TemporalConflict => "temporal_conflict",
            Self::CausalConflict => "causal_conflict",
        };
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Severity Tests ==========

    #[test]
    fn test_severity_ordering() {
        // DirectOpposition should be most severe
        assert!(
            ContradictionType::DirectOpposition.severity()
                > ContradictionType::LogicalInconsistency.severity()
        );
        assert!(
            ContradictionType::LogicalInconsistency.severity()
                > ContradictionType::TemporalConflict.severity()
        );
        assert!(
            ContradictionType::TemporalConflict.severity()
                > ContradictionType::CausalConflict.severity()
        );
    }

    #[test]
    fn test_severity_values() {
        assert!((ContradictionType::DirectOpposition.severity() - 1.0).abs() < 1e-6);
        assert!((ContradictionType::LogicalInconsistency.severity() - 0.8).abs() < 1e-6);
        assert!((ContradictionType::TemporalConflict.severity() - 0.7).abs() < 1e-6);
        assert!((ContradictionType::CausalConflict.severity() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_severity_in_valid_range() {
        for ct in ContradictionType::all() {
            let severity = ct.severity();
            assert!(
                (0.0..=1.0).contains(&severity),
                "Severity {} for {:?} out of range",
                severity,
                ct
            );
        }
    }

    // ========== All Variants Tests ==========

    #[test]
    fn test_all_variants() {
        let all = ContradictionType::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&ContradictionType::DirectOpposition));
        assert!(all.contains(&ContradictionType::LogicalInconsistency));
        assert!(all.contains(&ContradictionType::TemporalConflict));
        assert!(all.contains(&ContradictionType::CausalConflict));
    }

    #[test]
    fn test_all_unique() {
        use std::collections::HashSet;
        let all = ContradictionType::all();
        let unique: HashSet<_> = all.iter().collect();
        assert_eq!(unique.len(), 4);
    }

    // ========== u8 Conversion Tests ==========

    #[test]
    fn test_as_u8() {
        assert_eq!(ContradictionType::DirectOpposition.as_u8(), 0);
        assert_eq!(ContradictionType::LogicalInconsistency.as_u8(), 1);
        assert_eq!(ContradictionType::TemporalConflict.as_u8(), 2);
        assert_eq!(ContradictionType::CausalConflict.as_u8(), 3);
    }

    #[test]
    fn test_from_u8_valid() {
        assert_eq!(
            ContradictionType::from_u8(0),
            Some(ContradictionType::DirectOpposition)
        );
        assert_eq!(
            ContradictionType::from_u8(1),
            Some(ContradictionType::LogicalInconsistency)
        );
        assert_eq!(
            ContradictionType::from_u8(2),
            Some(ContradictionType::TemporalConflict)
        );
        assert_eq!(
            ContradictionType::from_u8(3),
            Some(ContradictionType::CausalConflict)
        );
    }

    #[test]
    fn test_from_u8_invalid() {
        assert_eq!(ContradictionType::from_u8(4), None);
        assert_eq!(ContradictionType::from_u8(5), None);
        assert_eq!(ContradictionType::from_u8(255), None);
    }

    #[test]
    fn test_u8_roundtrip() {
        for ct in ContradictionType::all() {
            let u8_val = ct.as_u8();
            let recovered = ContradictionType::from_u8(u8_val).expect("valid u8 should convert");
            assert_eq!(recovered, ct);
        }
    }

    // ========== Default Tests ==========

    #[test]
    fn test_default() {
        assert_eq!(
            ContradictionType::default(),
            ContradictionType::DirectOpposition
        );
    }

    #[test]
    fn test_default_is_most_severe() {
        let default = ContradictionType::default();
        for ct in ContradictionType::all() {
            assert!(default.severity() >= ct.severity());
        }
    }

    // ========== Display Tests ==========

    #[test]
    fn test_display() {
        assert_eq!(
            ContradictionType::DirectOpposition.to_string(),
            "direct_opposition"
        );
        assert_eq!(
            ContradictionType::LogicalInconsistency.to_string(),
            "logical_inconsistency"
        );
        assert_eq!(
            ContradictionType::TemporalConflict.to_string(),
            "temporal_conflict"
        );
        assert_eq!(
            ContradictionType::CausalConflict.to_string(),
            "causal_conflict"
        );
    }

    #[test]
    fn test_display_is_snake_case() {
        for ct in ContradictionType::all() {
            let s = ct.to_string();
            assert!(
                s.chars().all(|c| c.is_lowercase() || c == '_'),
                "Display for {:?} is not snake_case: {}",
                ct,
                s
            );
        }
    }

    // ========== Serde Tests ==========

    #[test]
    fn test_serde_roundtrip() {
        for ct in ContradictionType::all() {
            let json = serde_json::to_string(&ct).expect("serialize failed");
            let recovered: ContradictionType =
                serde_json::from_str(&json).expect("deserialize failed");
            assert_eq!(recovered, ct, "Roundtrip failed for {:?}", ct);
        }
    }

    #[test]
    fn test_serde_snake_case() {
        let json = serde_json::to_string(&ContradictionType::DirectOpposition).unwrap();
        assert_eq!(json, r#""direct_opposition""#);

        let json = serde_json::to_string(&ContradictionType::LogicalInconsistency).unwrap();
        assert_eq!(json, r#""logical_inconsistency""#);
    }

    #[test]
    fn test_serde_deserialize() {
        let ct: ContradictionType = serde_json::from_str(r#""temporal_conflict""#).unwrap();
        assert_eq!(ct, ContradictionType::TemporalConflict);
    }

    #[test]
    fn test_serde_invalid_fails() {
        let result: Result<ContradictionType, _> = serde_json::from_str(r#""invalid_type""#);
        assert!(result.is_err());
    }

    // ========== Description Tests ==========

    #[test]
    fn test_description_non_empty() {
        for ct in ContradictionType::all() {
            let desc = ct.description();
            assert!(!desc.is_empty(), "Description for {:?} is empty", ct);
            assert!(
                desc.len() > 10,
                "Description for {:?} too short: {}",
                ct,
                desc
            );
        }
    }

    #[test]
    fn test_description_contains_key_words() {
        assert!(ContradictionType::DirectOpposition
            .description()
            .to_lowercase()
            .contains("opposition"));
        assert!(ContradictionType::LogicalInconsistency
            .description()
            .to_lowercase()
            .contains("logical"));
        assert!(ContradictionType::TemporalConflict
            .description()
            .to_lowercase()
            .contains("temporal"));
        assert!(ContradictionType::CausalConflict
            .description()
            .to_lowercase()
            .contains("causal"));
    }

    // ========== High Severity Tests ==========

    #[test]
    fn test_is_high_severity() {
        assert!(ContradictionType::DirectOpposition.is_high_severity());
        assert!(ContradictionType::LogicalInconsistency.is_high_severity());
        assert!(!ContradictionType::TemporalConflict.is_high_severity());
        assert!(!ContradictionType::CausalConflict.is_high_severity());
    }

    // ========== Recommended Action Tests ==========

    #[test]
    fn test_recommended_action_non_empty() {
        for ct in ContradictionType::all() {
            let action = ct.recommended_action();
            assert!(!action.is_empty(), "Action for {:?} is empty", ct);
        }
    }

    // ========== Trait Tests ==========

    #[test]
    fn test_clone() {
        let ct = ContradictionType::TemporalConflict;
        let cloned = ct;
        assert_eq!(ct, cloned);
    }

    #[test]
    fn test_copy() {
        let ct = ContradictionType::CausalConflict;
        let copied = ct;
        assert_eq!(ct, copied);
        let _still_valid = ct;
    }

    #[test]
    fn test_debug() {
        let debug = format!("{:?}", ContradictionType::LogicalInconsistency);
        assert!(debug.contains("LogicalInconsistency"));
    }

    #[test]
    fn test_partial_eq() {
        assert_eq!(
            ContradictionType::DirectOpposition,
            ContradictionType::DirectOpposition
        );
        assert_ne!(
            ContradictionType::DirectOpposition,
            ContradictionType::CausalConflict
        );
    }

    #[test]
    fn test_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ContradictionType::DirectOpposition);
        set.insert(ContradictionType::LogicalInconsistency);
        set.insert(ContradictionType::DirectOpposition); // Duplicate
        assert_eq!(set.len(), 2);
    }
}
