//! Metrics and statistics for SearchResult.
//!
//! Methods for computing aggregate statistics over search results.

use std::cmp::Ordering;

use super::types::SearchResult;

impl SearchResult {
    /// Total number of valid results across all queries.
    ///
    /// Counts all IDs that are not -1 sentinel values.
    #[inline]
    pub fn total_valid_results(&self) -> usize {
        self.ids.iter().filter(|&&id| id != -1).count()
    }

    /// Check if result is empty (no valid matches for any query).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_valid_results() == 0
    }

    /// Get the minimum distance found across all queries.
    ///
    /// Returns `None` if no valid results exist.
    pub fn min_distance(&self) -> Option<f32> {
        self.ids
            .iter()
            .zip(&self.distances)
            .filter(|(&id, _)| id != -1)
            .map(|(_, &dist)| dist)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
    }

    /// Get the maximum distance found across all queries.
    ///
    /// Returns `None` if no valid results exist.
    pub fn max_distance(&self) -> Option<f32> {
        self.ids
            .iter()
            .zip(&self.distances)
            .filter(|(&id, _)| id != -1)
            .map(|(_, &dist)| dist)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
    }
}
