//! SessionContext - Context management for UTL computation sessions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Context for a single UTL computation session.
///
/// Maintains state for multiple related computations within a session,
/// including recent embeddings for context comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionContext {
    /// Unique session identifier.
    pub session_id: Uuid,

    /// Recent embeddings for context (sliding window).
    pub recent_embeddings: Vec<Vec<f32>>,

    /// Maximum context window size.
    pub max_window_size: usize,

    /// Number of interactions in this session.
    pub interaction_count: u64,

    /// Session start time.
    pub started_at: DateTime<Utc>,

    /// Last activity timestamp.
    pub last_activity: DateTime<Utc>,
}

impl SessionContext {
    /// Create a new session context.
    pub fn new(session_id: Uuid, max_window_size: usize) -> Self {
        let now = Utc::now();
        Self {
            session_id,
            recent_embeddings: Vec::with_capacity(max_window_size),
            max_window_size,
            interaction_count: 0,
            started_at: now,
            last_activity: now,
        }
    }

    /// Create a new session with auto-generated ID.
    pub fn new_with_generated_id(max_window_size: usize) -> Self {
        Self::new(Uuid::new_v4(), max_window_size)
    }

    /// Default session with 50-embedding window.
    pub fn default_session() -> Self {
        Self::new_with_generated_id(50)
    }

    /// Add an embedding to the context window.
    ///
    /// Maintains a sliding window - oldest embeddings are removed when
    /// the window exceeds `max_window_size`.
    pub fn add_embedding(&mut self, embedding: Vec<f32>) {
        if self.recent_embeddings.len() >= self.max_window_size {
            self.recent_embeddings.remove(0);
        }
        self.recent_embeddings.push(embedding);
        self.interaction_count += 1;
        self.last_activity = Utc::now();
    }

    /// Get context embeddings as slice.
    #[inline]
    pub fn context_embeddings(&self) -> &[Vec<f32>] {
        &self.recent_embeddings
    }

    /// Check if session has sufficient context for meaningful comparison.
    pub fn has_sufficient_context(&self) -> bool {
        self.recent_embeddings.len() >= 2
    }

    /// Check if session is stale.
    pub fn is_stale(&self, max_age_seconds: i64) -> bool {
        let age = Utc::now() - self.last_activity;
        age.num_seconds() > max_age_seconds
    }

    /// Get session age in seconds.
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.started_at).num_seconds()
    }

    /// Clear context but preserve session metadata.
    pub fn clear_context(&mut self) {
        self.recent_embeddings.clear();
        self.interaction_count = 0;
        self.last_activity = Utc::now();
    }
}

impl Default for SessionContext {
    fn default() -> Self {
        Self::default_session()
    }
}
