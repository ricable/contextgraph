//! Goal hierarchy types for teleological alignment.
//!
//! Provides the goal tree structure that defines the North Star and sub-goals
//! for purpose vector computation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for a goal in the hierarchy.
///
/// Used to reference goals in the tree structure and for parent-child relationships.
///
/// # Example
///
/// ```
/// use context_graph_core::purpose::GoalId;
///
/// let id = GoalId::new("north_star_ml");
/// assert_eq!(id.as_str(), "north_star_ml");
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GoalId(String);

impl GoalId {
    /// Create a new GoalId from any string-like type.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the ID as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for GoalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for GoalId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for GoalId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

/// Goal hierarchy level.
///
/// Defines the position of a goal in the hierarchical tree structure.
/// Each level has a different propagation weight for alignment computation.
///
/// From constitution.yaml:
/// - NorthStar: 1.0 weight (global goal)
/// - Strategic: 0.7 weight (mid-level)
/// - Tactical: 0.4 weight (short-term)
/// - Immediate: 0.2 weight (per-operation)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum GoalLevel {
    /// Top-level aspirational goal (the "North Star").
    /// Only one allowed per hierarchy.
    NorthStar = 0,

    /// Mid-term strategic objectives.
    /// Children of NorthStar.
    Strategic = 1,

    /// Short-term tactical goals.
    /// Children of Strategic goals.
    Tactical = 2,

    /// Immediate context goals.
    /// Lowest level, most specific.
    Immediate = 3,
}

impl GoalLevel {
    /// Weight factor for hierarchical propagation.
    ///
    /// From constitution.yaml:
    /// - NorthStar: 1.0
    /// - Strategic: 0.7
    /// - Tactical: 0.4
    /// - Immediate: 0.2
    #[inline]
    pub fn propagation_weight(&self) -> f32 {
        match self {
            GoalLevel::NorthStar => 1.0,
            GoalLevel::Strategic => 0.7,
            GoalLevel::Tactical => 0.4,
            GoalLevel::Immediate => 0.2,
        }
    }

    /// Get numeric depth (0 = NorthStar, 3 = Immediate).
    #[inline]
    pub fn depth(&self) -> u8 {
        *self as u8
    }
}

/// A node in the goal hierarchy tree.
///
/// Represents a single goal with its semantic embedding, keywords for
/// SPLADE alignment, and position in the hierarchy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoalNode {
    /// Unique goal identifier.
    pub id: GoalId,

    /// Human-readable goal description.
    pub description: String,

    /// Level in the hierarchy.
    pub level: GoalLevel,

    /// Parent goal ID (None for NorthStar).
    pub parent: Option<GoalId>,

    /// Goal's semantic embedding (1024D for projection to E1).
    ///
    /// This embedding represents the goal's meaning in semantic space.
    /// It is projected to match each embedding space's dimensions during
    /// alignment computation.
    pub embedding: Vec<f32>,

    /// Importance weight [0.0, 1.0].
    ///
    /// Higher weight means this goal contributes more to the overall
    /// purpose vector computation.
    pub weight: f32,

    /// Keywords for E13 SPLADE matching.
    ///
    /// These keywords are matched against the SPLADE embedding to compute
    /// keyword-based alignment for Stage 1 sparse pre-filtering.
    pub keywords: Vec<String>,
}

impl GoalNode {
    /// Create a new North Star goal.
    ///
    /// North Star is the top-level goal with no parent and weight 1.0.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the goal
    /// * `description` - Human-readable description
    /// * `embedding` - 1024D semantic embedding
    /// * `keywords` - Keywords for SPLADE matching
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::purpose::{GoalNode, GoalLevel};
    ///
    /// let goal = GoalNode::north_star(
    ///     "master_ml",
    ///     "Master machine learning fundamentals",
    ///     vec![0.5; 1024],
    ///     vec!["machine".into(), "learning".into()],
    /// );
    ///
    /// assert_eq!(goal.level, GoalLevel::NorthStar);
    /// assert!(goal.parent.is_none());
    /// assert_eq!(goal.weight, 1.0);
    /// ```
    pub fn north_star(
        id: impl Into<String>,
        description: impl Into<String>,
        embedding: Vec<f32>,
        keywords: Vec<String>,
    ) -> Self {
        Self {
            id: GoalId::new(id),
            description: description.into(),
            level: GoalLevel::NorthStar,
            parent: None,
            embedding,
            weight: 1.0,
            keywords,
        }
    }

    /// Create a child goal with a parent.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the goal
    /// * `description` - Human-readable description
    /// * `level` - Must NOT be NorthStar
    /// * `parent` - Parent goal ID
    /// * `embedding` - 1024D semantic embedding
    /// * `weight` - Importance weight [0.0, 1.0]
    /// * `keywords` - Keywords for SPLADE matching
    ///
    /// # Panics
    ///
    /// Panics if `level` is `GoalLevel::NorthStar`.
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::purpose::{GoalNode, GoalLevel, GoalId};
    ///
    /// let child = GoalNode::child(
    ///     "learn_pytorch",
    ///     "Learn PyTorch framework",
    ///     GoalLevel::Strategic,
    ///     GoalId::new("master_ml"),
    ///     vec![0.4; 1024],
    ///     0.8,
    ///     vec!["pytorch".into(), "tensors".into()],
    /// );
    ///
    /// assert_eq!(child.level, GoalLevel::Strategic);
    /// assert!(child.parent.is_some());
    /// ```
    pub fn child(
        id: impl Into<String>,
        description: impl Into<String>,
        level: GoalLevel,
        parent: GoalId,
        embedding: Vec<f32>,
        weight: f32,
        keywords: Vec<String>,
    ) -> Self {
        assert!(level != GoalLevel::NorthStar, "Child cannot be NorthStar level");
        Self {
            id: GoalId::new(id),
            description: description.into(),
            level,
            parent: Some(parent),
            embedding,
            weight: weight.clamp(0.0, 1.0),
            keywords,
        }
    }

    /// Check if this is a North Star goal.
    #[inline]
    pub fn is_north_star(&self) -> bool {
        self.level == GoalLevel::NorthStar
    }
}

/// Goal hierarchy tree structure.
///
/// Manages a tree of goals with a single North Star at the root.
/// Used for hierarchical alignment propagation in purpose vector computation.
///
/// # Invariants
///
/// - At most one North Star goal
/// - All child goals have valid parent references
/// - No cycles in the hierarchy
#[derive(Clone, Debug, Default)]
pub struct GoalHierarchy {
    nodes: HashMap<GoalId, GoalNode>,
    north_star: Option<GoalId>,
}

impl GoalHierarchy {
    /// Create a new empty goal hierarchy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a goal to the hierarchy.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Adding a second North Star goal
    /// - Child goal's parent doesn't exist
    ///
    /// # Example
    ///
    /// ```
    /// use context_graph_core::purpose::{GoalHierarchy, GoalNode, GoalLevel, GoalId};
    ///
    /// let mut hierarchy = GoalHierarchy::new();
    ///
    /// let ns = GoalNode::north_star("ns", "North Star", vec![0.5; 1024], vec![]);
    /// hierarchy.add_goal(ns).unwrap();
    ///
    /// let child = GoalNode::child(
    ///     "child",
    ///     "Child Goal",
    ///     GoalLevel::Strategic,
    ///     GoalId::new("ns"),
    ///     vec![0.4; 1024],
    ///     0.8,
    ///     vec![],
    /// );
    /// hierarchy.add_goal(child).unwrap();
    /// ```
    pub fn add_goal(&mut self, goal: GoalNode) -> Result<(), GoalHierarchyError> {
        // Validate parent exists (except for NorthStar)
        if let Some(ref parent_id) = goal.parent {
            if !self.nodes.contains_key(parent_id) {
                return Err(GoalHierarchyError::ParentNotFound(parent_id.clone()));
            }
        }

        // Only one North Star allowed
        if goal.level == GoalLevel::NorthStar {
            if self.north_star.is_some() {
                return Err(GoalHierarchyError::MultipleNorthStars);
            }
            self.north_star = Some(goal.id.clone());
        }

        self.nodes.insert(goal.id.clone(), goal);
        Ok(())
    }

    /// Get the North Star goal.
    ///
    /// Returns None if no North Star has been added.
    pub fn north_star(&self) -> Option<&GoalNode> {
        self.north_star.as_ref().and_then(|id| self.nodes.get(id))
    }

    /// Check if a North Star goal exists.
    #[inline]
    pub fn has_north_star(&self) -> bool {
        self.north_star.is_some()
    }

    /// Get a goal by ID.
    pub fn get(&self, id: &GoalId) -> Option<&GoalNode> {
        self.nodes.get(id)
    }

    /// Get direct children of a goal.
    pub fn children(&self, parent_id: &GoalId) -> Vec<&GoalNode> {
        self.nodes
            .values()
            .filter(|n| n.parent.as_ref() == Some(parent_id))
            .collect()
    }

    /// Get all goals at a specific level.
    pub fn at_level(&self, level: GoalLevel) -> Vec<&GoalNode> {
        self.nodes.values().filter(|n| n.level == level).collect()
    }

    /// Total number of goals in the hierarchy.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the hierarchy is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Iterate over all goals.
    pub fn iter(&self) -> impl Iterator<Item = &GoalNode> {
        self.nodes.values()
    }

    /// Get path from a goal to the North Star.
    ///
    /// Returns the sequence of goal IDs from the given goal up to (and including)
    /// the North Star. Returns empty vec if goal not found.
    pub fn path_to_north_star(&self, goal_id: &GoalId) -> Vec<GoalId> {
        let mut path = Vec::new();
        let mut current = self.nodes.get(goal_id);

        while let Some(node) = current {
            path.push(node.id.clone());
            current = node.parent.as_ref().and_then(|pid| self.nodes.get(pid));
        }

        path
    }

    /// Validate hierarchy integrity.
    ///
    /// Checks:
    /// - North Star exists if hierarchy is not empty
    /// - All parent references are valid
    pub fn validate(&self) -> Result<(), GoalHierarchyError> {
        if self.north_star.is_none() && !self.nodes.is_empty() {
            return Err(GoalHierarchyError::NoNorthStar);
        }

        // Check all parents exist
        for node in self.nodes.values() {
            if let Some(ref parent_id) = node.parent {
                if !self.nodes.contains_key(parent_id) {
                    return Err(GoalHierarchyError::ParentNotFound(parent_id.clone()));
                }
            }
        }

        Ok(())
    }
}

/// Errors for goal hierarchy operations.
#[derive(Debug, thiserror::Error)]
pub enum GoalHierarchyError {
    /// No North Star goal has been defined.
    #[error("No North Star goal defined")]
    NoNorthStar,

    /// Attempted to add a second North Star goal.
    #[error("Multiple North Star goals not allowed")]
    MultipleNorthStars,

    /// Referenced parent goal does not exist.
    #[error("Parent goal not found: {0}")]
    ParentNotFound(GoalId),

    /// Referenced goal does not exist.
    #[error("Goal not found: {0}")]
    GoalNotFound(GoalId),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goal_id_creation_and_display() {
        let id = GoalId::new("north_star_ml");
        assert_eq!(id.as_str(), "north_star_ml");
        assert_eq!(format!("{}", id), "north_star_ml");
        println!("[VERIFIED] GoalId creation and display works correctly");
    }

    #[test]
    fn test_goal_id_from_traits() {
        let from_str: GoalId = "test_goal".into();
        let from_string: GoalId = String::from("test_goal").into();
        assert_eq!(from_str, from_string);
        println!("[VERIFIED] GoalId From traits work correctly");
    }

    #[test]
    fn test_goal_level_propagation_weights() {
        assert_eq!(GoalLevel::NorthStar.propagation_weight(), 1.0);
        assert_eq!(GoalLevel::Strategic.propagation_weight(), 0.7);
        assert_eq!(GoalLevel::Tactical.propagation_weight(), 0.4);
        assert_eq!(GoalLevel::Immediate.propagation_weight(), 0.2);
        println!("[VERIFIED] GoalLevel propagation weights match constitution.yaml");
    }

    #[test]
    fn test_goal_level_depth() {
        assert_eq!(GoalLevel::NorthStar.depth(), 0);
        assert_eq!(GoalLevel::Strategic.depth(), 1);
        assert_eq!(GoalLevel::Tactical.depth(), 2);
        assert_eq!(GoalLevel::Immediate.depth(), 3);
        println!("[VERIFIED] GoalLevel depth values are correct");
    }

    #[test]
    fn test_goal_node_north_star_creation() {
        let embedding = vec![0.1; 1024];
        let keywords = vec!["machine".into(), "learning".into()];

        let goal = GoalNode::north_star(
            "ml_mastery",
            "Master machine learning fundamentals",
            embedding.clone(),
            keywords.clone(),
        );

        assert_eq!(goal.level, GoalLevel::NorthStar);
        assert!(goal.parent.is_none());
        assert_eq!(goal.weight, 1.0);
        assert_eq!(goal.embedding.len(), 1024);
        assert_eq!(goal.keywords.len(), 2);
        assert!(goal.is_north_star());
        println!("[VERIFIED] GoalNode::north_star creates correct structure");
    }

    #[test]
    fn test_goal_node_child_creation() {
        let child = GoalNode::child(
            "learn_pytorch",
            "Learn PyTorch framework",
            GoalLevel::Strategic,
            GoalId::new("master_ml"),
            vec![0.4; 1024],
            0.8,
            vec!["pytorch".into()],
        );

        assert_eq!(child.level, GoalLevel::Strategic);
        assert!(child.parent.is_some());
        assert_eq!(child.parent.as_ref().unwrap().as_str(), "master_ml");
        assert_eq!(child.weight, 0.8);
        assert!(!child.is_north_star());
        println!("[VERIFIED] GoalNode::child creates correct structure");
    }

    #[test]
    fn test_goal_node_weight_clamping() {
        let child_over = GoalNode::child(
            "test1",
            "Test",
            GoalLevel::Strategic,
            GoalId::new("parent"),
            vec![],
            1.5, // Over max
            vec![],
        );
        assert_eq!(child_over.weight, 1.0);

        let child_under = GoalNode::child(
            "test2",
            "Test",
            GoalLevel::Strategic,
            GoalId::new("parent"),
            vec![],
            -0.5, // Under min
            vec![],
        );
        assert_eq!(child_under.weight, 0.0);
        println!("[VERIFIED] GoalNode weight is clamped to [0.0, 1.0]");
    }

    #[test]
    #[should_panic(expected = "Child cannot be NorthStar")]
    fn test_goal_node_child_cannot_be_north_star() {
        let _ = GoalNode::child(
            "bad",
            "Bad goal",
            GoalLevel::NorthStar, // Should panic
            GoalId::new("parent"),
            vec![],
            0.5,
            vec![],
        );
    }

    #[test]
    fn test_goal_hierarchy_single_north_star() {
        let mut hierarchy = GoalHierarchy::new();

        let ns1 = GoalNode::north_star("ns1", "Goal 1", vec![0.1; 1024], vec![]);
        let ns2 = GoalNode::north_star("ns2", "Goal 2", vec![0.2; 1024], vec![]);

        assert!(hierarchy.add_goal(ns1).is_ok());
        let result = hierarchy.add_goal(ns2);
        assert!(matches!(result, Err(GoalHierarchyError::MultipleNorthStars)));
        println!("[VERIFIED] GoalHierarchy enforces single North Star");
    }

    #[test]
    fn test_goal_hierarchy_parent_validation() {
        let mut hierarchy = GoalHierarchy::new();

        // Try to add child without parent
        let child = GoalNode::child(
            "orphan",
            "Orphan goal",
            GoalLevel::Strategic,
            GoalId::new("nonexistent"),
            vec![0.1; 1024],
            0.8,
            vec![],
        );

        let result = hierarchy.add_goal(child);
        assert!(matches!(
            result,
            Err(GoalHierarchyError::ParentNotFound(_))
        ));
        println!("[VERIFIED] GoalHierarchy validates parent existence");
    }

    #[test]
    fn test_goal_hierarchy_full_tree() {
        let mut hierarchy = GoalHierarchy::new();

        // Add North Star
        let ns = GoalNode::north_star(
            "master_ml",
            "Master ML",
            vec![0.5; 1024],
            vec!["machine".into(), "learning".into()],
        );
        hierarchy.add_goal(ns).unwrap();

        // Add Strategic child
        let strategic = GoalNode::child(
            "learn_pytorch",
            "Learn PyTorch",
            GoalLevel::Strategic,
            GoalId::new("master_ml"),
            vec![0.4; 1024],
            0.8,
            vec!["pytorch".into(), "tensors".into()],
        );
        hierarchy.add_goal(strategic).unwrap();

        // Add Tactical child
        let tactical = GoalNode::child(
            "complete_tutorial",
            "Complete tutorial",
            GoalLevel::Tactical,
            GoalId::new("learn_pytorch"),
            vec![0.3; 1024],
            0.6,
            vec!["tutorial".into()],
        );
        hierarchy.add_goal(tactical).unwrap();

        assert_eq!(hierarchy.len(), 3);
        assert!(!hierarchy.is_empty());
        assert!(hierarchy.has_north_star());
        assert!(hierarchy.north_star().is_some());
        assert_eq!(hierarchy.at_level(GoalLevel::Strategic).len(), 1);
        assert_eq!(hierarchy.at_level(GoalLevel::Tactical).len(), 1);
        assert_eq!(hierarchy.children(&GoalId::new("master_ml")).len(), 1);
        assert!(hierarchy.validate().is_ok());

        println!("[VERIFIED] GoalHierarchy full tree structure works correctly");
    }

    #[test]
    fn test_goal_hierarchy_path_to_north_star() {
        let mut hierarchy = GoalHierarchy::new();

        hierarchy
            .add_goal(GoalNode::north_star("ns", "NS", vec![], vec![]))
            .unwrap();
        hierarchy
            .add_goal(GoalNode::child(
                "s1",
                "S1",
                GoalLevel::Strategic,
                GoalId::new("ns"),
                vec![],
                0.8,
                vec![],
            ))
            .unwrap();
        hierarchy
            .add_goal(GoalNode::child(
                "t1",
                "T1",
                GoalLevel::Tactical,
                GoalId::new("s1"),
                vec![],
                0.6,
                vec![],
            ))
            .unwrap();

        let path = hierarchy.path_to_north_star(&GoalId::new("t1"));
        assert_eq!(path.len(), 3);
        assert_eq!(path[0].as_str(), "t1");
        assert_eq!(path[1].as_str(), "s1");
        assert_eq!(path[2].as_str(), "ns");

        println!("[VERIFIED] path_to_north_star returns correct path");
    }

    #[test]
    fn test_goal_hierarchy_validate_no_north_star() {
        let mut hierarchy = GoalHierarchy::new();

        // Manually insert a node without North Star (bypass add_goal)
        hierarchy.nodes.insert(
            GoalId::new("orphan"),
            GoalNode {
                id: GoalId::new("orphan"),
                description: "Orphan".into(),
                level: GoalLevel::Strategic,
                parent: None,
                embedding: vec![],
                weight: 0.5,
                keywords: vec![],
            },
        );

        let result = hierarchy.validate();
        assert!(matches!(result, Err(GoalHierarchyError::NoNorthStar)));
        println!("[VERIFIED] validate detects missing North Star");
    }

    #[test]
    fn test_goal_hierarchy_iter() {
        let mut hierarchy = GoalHierarchy::new();
        hierarchy
            .add_goal(GoalNode::north_star("ns", "NS", vec![], vec![]))
            .unwrap();
        hierarchy
            .add_goal(GoalNode::child(
                "c1",
                "C1",
                GoalLevel::Strategic,
                GoalId::new("ns"),
                vec![],
                0.8,
                vec![],
            ))
            .unwrap();

        let count = hierarchy.iter().count();
        assert_eq!(count, 2);
        println!("[VERIFIED] GoalHierarchy iter works correctly");
    }
}
