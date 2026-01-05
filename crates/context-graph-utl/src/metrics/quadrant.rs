//! Johari quadrant distribution tracking for UTL classifications.

use serde::{Deserialize, Serialize};

use crate::johari::JohariQuadrant;

/// Distribution of classifications across Johari Window quadrants.
///
/// Maps to the UTL theory's Johari-DS x DC plane:
/// - Open: Low entropy, High coherence (known to self & others)
/// - Blind: High entropy, Low coherence (unknown to self, known to others)
/// - Hidden: Medium entropy, High coherence (known to self, hidden from others)
/// - Unknown: High entropy, Unknown coherence (unknown to all)
///
/// # Example
///
/// ```
/// use context_graph_utl::metrics::QuadrantDistribution;
/// use context_graph_utl::johari::JohariQuadrant;
///
/// let mut dist = QuadrantDistribution::new();
/// dist.increment(JohariQuadrant::Open);
/// dist.increment(JohariQuadrant::Open);
/// dist.increment(JohariQuadrant::Blind);
///
/// assert_eq!(dist.total(), 3);
/// assert_eq!(dist.dominant(), JohariQuadrant::Open);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct QuadrantDistribution {
    /// Count of Open quadrant classifications (low DS, high DC)
    pub open: u32,

    /// Count of Blind quadrant classifications (high DS, low DC)
    pub blind: u32,

    /// Count of Hidden quadrant classifications (medium DS, high DC)
    pub hidden: u32,

    /// Count of Unknown quadrant classifications (high DS, unknown DC)
    pub unknown: u32,
}

impl QuadrantDistribution {
    /// Create new empty distribution.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get total count across all quadrants.
    ///
    /// Uses checked arithmetic to prevent overflow, returning `u32::MAX` if overflow would occur.
    pub fn total(&self) -> u32 {
        self.open
            .checked_add(self.blind)
            .and_then(|sum| sum.checked_add(self.hidden))
            .and_then(|sum| sum.checked_add(self.unknown))
            .unwrap_or(u32::MAX)
    }

    /// Get percentages for each quadrant.
    ///
    /// Returns `[open_pct, blind_pct, hidden_pct, unknown_pct]`.
    /// Returns uniform `[0.25, 0.25, 0.25, 0.25]` when empty.
    ///
    /// # Invariant
    /// Sum of returned values equals 1.0 (within floating point tolerance).
    pub fn percentages(&self) -> [f32; 4] {
        let total = self.total();
        if total == 0 {
            return [0.25, 0.25, 0.25, 0.25];
        }

        let total_f = total as f32;
        [
            self.open as f32 / total_f,
            self.blind as f32 / total_f,
            self.hidden as f32 / total_f,
            self.unknown as f32 / total_f,
        ]
    }

    /// Get the dominant (most frequent) quadrant.
    ///
    /// Returns `JohariQuadrant::Open` on tie or when empty.
    pub fn dominant(&self) -> JohariQuadrant {
        // Use explicit comparison to ensure Open wins ties (first in order)
        let max_count = self.open.max(self.blind).max(self.hidden).max(self.unknown);

        if self.open == max_count {
            JohariQuadrant::Open
        } else if self.blind == max_count {
            JohariQuadrant::Blind
        } else if self.hidden == max_count {
            JohariQuadrant::Hidden
        } else {
            JohariQuadrant::Unknown
        }
    }

    /// Increment count for a specific quadrant.
    ///
    /// Saturates at `u32::MAX` to prevent overflow.
    pub fn increment(&mut self, quadrant: JohariQuadrant) {
        match quadrant {
            JohariQuadrant::Open => self.open = self.open.saturating_add(1),
            JohariQuadrant::Blind => self.blind = self.blind.saturating_add(1),
            JohariQuadrant::Hidden => self.hidden = self.hidden.saturating_add(1),
            JohariQuadrant::Unknown => self.unknown = self.unknown.saturating_add(1),
        }
    }

    /// Get count for a specific quadrant.
    pub fn count(&self, quadrant: JohariQuadrant) -> u32 {
        match quadrant {
            JohariQuadrant::Open => self.open,
            JohariQuadrant::Blind => self.blind,
            JohariQuadrant::Hidden => self.hidden,
            JohariQuadrant::Unknown => self.unknown,
        }
    }

    /// Reset all counts to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadrant_distribution_default() {
        let dist = QuadrantDistribution::default();

        assert_eq!(dist.open, 0);
        assert_eq!(dist.blind, 0);
        assert_eq!(dist.hidden, 0);
        assert_eq!(dist.unknown, 0);
        assert_eq!(dist.total(), 0);
    }

    #[test]
    fn test_quadrant_distribution_percentages_empty() {
        let dist = QuadrantDistribution::default();
        let pcts = dist.percentages();

        // Uniform distribution when empty
        assert_eq!(pcts, [0.25, 0.25, 0.25, 0.25]);

        // Sum must equal 1.0
        let sum: f32 = pcts.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_quadrant_distribution_percentages_with_data() {
        let dist = QuadrantDistribution {
            open: 50,
            blind: 25,
            hidden: 15,
            unknown: 10,
        };

        let pcts = dist.percentages();

        assert!((pcts[0] - 0.50).abs() < 0.001); // open
        assert!((pcts[1] - 0.25).abs() < 0.001); // blind
        assert!((pcts[2] - 0.15).abs() < 0.001); // hidden
        assert!((pcts[3] - 0.10).abs() < 0.001); // unknown

        // Sum must equal 1.0
        let sum: f32 = pcts.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_quadrant_distribution_dominant() {
        let dist = QuadrantDistribution {
            open: 10,
            blind: 50, // Most frequent
            hidden: 20,
            unknown: 15,
        };

        assert_eq!(dist.dominant(), JohariQuadrant::Blind);
    }

    #[test]
    fn test_quadrant_distribution_dominant_empty() {
        let dist = QuadrantDistribution::default();

        // Default to Open when empty
        assert_eq!(dist.dominant(), JohariQuadrant::Open);
    }

    #[test]
    fn test_quadrant_distribution_dominant_tie() {
        let dist = QuadrantDistribution {
            open: 25,
            blind: 25,
            hidden: 25,
            unknown: 25,
        };

        // On tie, max_by_key returns first max (Open)
        assert_eq!(dist.dominant(), JohariQuadrant::Open);
    }

    #[test]
    fn test_quadrant_distribution_increment() {
        let mut dist = QuadrantDistribution::default();

        dist.increment(JohariQuadrant::Open);
        dist.increment(JohariQuadrant::Open);
        dist.increment(JohariQuadrant::Blind);
        dist.increment(JohariQuadrant::Hidden);

        assert_eq!(dist.open, 2);
        assert_eq!(dist.blind, 1);
        assert_eq!(dist.hidden, 1);
        assert_eq!(dist.unknown, 0);
        assert_eq!(dist.total(), 4);
    }

    #[test]
    fn test_quadrant_distribution_increment_saturation() {
        let mut dist = QuadrantDistribution {
            open: u32::MAX,
            blind: 0,
            hidden: 0,
            unknown: 0,
        };

        dist.increment(JohariQuadrant::Open);

        // Should saturate, not overflow
        assert_eq!(dist.open, u32::MAX);
    }

    #[test]
    fn test_quadrant_distribution_count() {
        let dist = QuadrantDistribution {
            open: 10,
            blind: 20,
            hidden: 30,
            unknown: 40,
        };

        assert_eq!(dist.count(JohariQuadrant::Open), 10);
        assert_eq!(dist.count(JohariQuadrant::Blind), 20);
        assert_eq!(dist.count(JohariQuadrant::Hidden), 30);
        assert_eq!(dist.count(JohariQuadrant::Unknown), 40);
    }

    #[test]
    fn test_quadrant_distribution_total_overflow_protection() {
        let dist = QuadrantDistribution {
            open: u32::MAX,
            blind: u32::MAX,
            hidden: u32::MAX,
            unknown: u32::MAX,
        };

        // Should return MAX, not panic or wrap
        assert_eq!(dist.total(), u32::MAX);
    }

    #[test]
    fn test_quadrant_distribution_reset() {
        let mut dist = QuadrantDistribution {
            open: 100,
            blind: 200,
            hidden: 300,
            unknown: 400,
        };

        dist.reset();

        assert_eq!(dist.total(), 0);
    }

    #[test]
    fn test_quadrant_distribution_serialization() {
        let dist = QuadrantDistribution {
            open: 10,
            blind: 20,
            hidden: 30,
            unknown: 40,
        };

        let json = serde_json::to_string(&dist).expect("serialize");
        let parsed: QuadrantDistribution = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(dist, parsed);
    }
}
