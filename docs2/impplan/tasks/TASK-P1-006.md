# TASK-P1-006: SessionManager

```xml
<task_spec id="TASK-P1-006" version="1.0">
<metadata>
  <title>SessionManager Implementation</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>11</sequence>
  <phase>1</phase>
  <implements>
    <requirement_ref>REQ-P1-06</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P1-003</task_ref>
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<context>
Implements the SessionManager component for managing memory capture sessions.
Tracks session lifecycle (start, end, abandon) and provides current session lookup.

Uses file-based tracking for current session ID to survive process restarts.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE1-MEMORY-CAPTURE.md#component_contracts</file>
  <file purpose="session_types">crates/context-graph-core/src/memory/session.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P1-003 complete (Session, SessionStatus types exist)</check>
</prerequisites>

<scope>
  <in_scope>
    - Create SessionManager struct
    - Implement start_session() method
    - Implement end_session() method
    - Implement get_current_session() method
    - Track current session in file
    - Store sessions in RocksDB
    - Add SessionError enum
  </in_scope>
  <out_of_scope>
    - Memory capture (TASK-P1-007)
    - Session summary generation
    - Automatic session abandonment detection
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/memory/session.rs">
      pub struct SessionManager {
          db: Arc&lt;DB&gt;,
          session_file: PathBuf,
      }

      impl SessionManager {
          pub fn new(db: Arc&lt;DB&gt;, data_dir: &amp;Path) -> Result&lt;Self, SessionError&gt;;
          pub async fn start_session(&amp;self) -> Result&lt;String, SessionError&gt;;
          pub async fn end_session(&amp;self, session_id: &amp;str) -> Result&lt;(), SessionError&gt;;
          pub async fn get_current_session(&amp;self) -> Result&lt;Option&lt;String&gt;, SessionError&gt;;
          pub async fn get_session(&amp;self, session_id: &amp;str) -> Result&lt;Option&lt;Session&gt;, SessionError&gt;;
          pub async fn abandon_session(&amp;self, session_id: &amp;str) -> Result&lt;(), SessionError&gt;;
      }
    </signature>
  </signatures>

  <constraints>
    - Current session ID stored in file for persistence
    - Session data stored in RocksDB sessions CF
    - end_session is idempotent (no error if already ended)
    - start_session fails if active session exists
  </constraints>

  <verification>
    - Start/end session lifecycle works
    - Current session survives process restart
    - Idempotent end_session works
  </verification>
</definition_of_done>

<pseudo_code>
// Add to crates/context-graph-core/src/memory/session.rs

use rocksdb::DB;
use std::sync::Arc;
use std::path::{Path, PathBuf};
use std::fs;
use thiserror::Error;

const CF_SESSIONS: &amp;str = "sessions";
const CURRENT_SESSION_FILE: &amp;str = "current_session";

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("Session not found: {session_id}")]
    NotFound { session_id: String },
    #[error("Session already active: {session_id}")]
    AlreadyActive { session_id: String },
    #[error("Storage failed: {0}")]
    StorageFailed(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub struct SessionManager {
    db: Arc&lt;DB&gt;,
    session_file: PathBuf,
}

impl SessionManager {
    pub fn new(db: Arc&lt;DB&gt;, data_dir: &amp;Path) -> Result&lt;Self, SessionError&gt; {
        let session_file = data_dir.join(CURRENT_SESSION_FILE);
        Ok(Self { db, session_file })
    }

    pub async fn start_session(&amp;self) -> Result&lt;String, SessionError&gt; {
        // Check if session already active
        if let Some(existing) = self.get_current_session().await? {
            return Err(SessionError::AlreadyActive { session_id: existing });
        }

        // Create new session
        let session = Session::new();
        let session_id = session.id.clone();

        // Store session in DB
        self.store_session(&amp;session).await?;

        // Write current session file
        fs::write(&amp;self.session_file, &amp;session_id)?;

        Ok(session_id)
    }

    pub async fn end_session(&amp;self, session_id: &amp;str) -> Result&lt;(), SessionError&gt; {
        // Load session
        let mut session = match self.get_session(session_id).await? {
            Some(s) => s,
            None => return Ok(()), // Idempotent - already gone
        };

        // Already completed?
        if session.status != SessionStatus::Active {
            return Ok(()); // Idempotent
        }

        // Mark as completed
        session.complete();
        self.store_session(&amp;session).await?;

        // Clear current session file if this is the current one
        if let Some(current) = self.get_current_session().await? {
            if current == session_id {
                fs::remove_file(&amp;self.session_file).ok(); // Ignore if already gone
            }
        }

        Ok(())
    }

    pub async fn get_current_session(&amp;self) -> Result&lt;Option&lt;String&gt;, SessionError&gt; {
        match fs::read_to_string(&amp;self.session_file) {
            Ok(content) => {
                let session_id = content.trim().to_string();
                if session_id.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(session_id))
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(SessionError::IoError(e)),
        }
    }

    pub async fn get_session(&amp;self, session_id: &amp;str) -> Result&lt;Option&lt;Session&gt;, SessionError&gt; {
        let cf = self.db.cf_handle(CF_SESSIONS)
            .ok_or_else(|| SessionError::StorageFailed("sessions CF not found".into()))?;

        match self.db.get_cf(&amp;cf, session_id.as_bytes()) {
            Ok(Some(data)) => {
                let session: Session = bincode::deserialize(&amp;data)
                    .map_err(|e| SessionError::StorageFailed(e.to_string()))?;
                Ok(Some(session))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(SessionError::StorageFailed(e.to_string())),
        }
    }

    pub async fn abandon_session(&amp;self, session_id: &amp;str) -> Result&lt;(), SessionError&gt; {
        let mut session = match self.get_session(session_id).await? {
            Some(s) => s,
            None => return Ok(()),
        };

        if session.status != SessionStatus::Active {
            return Ok(());
        }

        session.abandon();
        self.store_session(&amp;session).await?;

        // Clear current session file
        if let Some(current) = self.get_current_session().await? {
            if current == session_id {
                fs::remove_file(&amp;self.session_file).ok();
            }
        }

        Ok(())
    }

    async fn store_session(&amp;self, session: &amp;Session) -> Result&lt;(), SessionError&gt; {
        let cf = self.db.cf_handle(CF_SESSIONS)
            .ok_or_else(|| SessionError::StorageFailed("sessions CF not found".into()))?;

        let value = bincode::serialize(session)
            .map_err(|e| SessionError::StorageFailed(e.to_string()))?;

        self.db.put_cf(&amp;cf, session.id.as_bytes(), &amp;value)
            .map_err(|e| SessionError::StorageFailed(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use rocksdb::Options;

    fn create_test_db(path: &amp;Path) -> Arc&lt;DB&gt; {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let cf = ColumnFamilyDescriptor::new(CF_SESSIONS, Options::default());
        Arc::new(DB::open_cf_descriptors(&amp;opts, path, vec![cf]).unwrap())
    }

    #[tokio::test]
    async fn test_session_lifecycle() {
        let dir = tempdir().unwrap();
        let db = create_test_db(dir.path());
        let manager = SessionManager::new(db, dir.path()).unwrap();

        // Start session
        let session_id = manager.start_session().await.unwrap();

        // Should be current
        let current = manager.get_current_session().await.unwrap();
        assert_eq!(current, Some(session_id.clone()));

        // End session
        manager.end_session(&amp;session_id).await.unwrap();

        // Should have no current
        let current = manager.get_current_session().await.unwrap();
        assert_eq!(current, None);
    }
}
</pseudo_code>

<files_to_modify>
  <file path="crates/context-graph-core/src/memory/session.rs">Add SessionManager implementation</file>
</files_to_modify>

<validation_criteria>
  <criterion>start_session creates new session and tracks as current</criterion>
  <criterion>end_session marks session complete and clears current</criterion>
  <criterion>get_current_session returns correct session ID</criterion>
  <criterion>Session survives across process restarts (file-based)</criterion>
  <criterion>end_session is idempotent</criterion>
  <criterion>start_session fails if session already active</criterion>
</validation_criteria>

<test_commands>
  <command description="Run session tests">cargo test --package context-graph-core session</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Add SessionError enum
- [ ] Implement SessionManager struct
- [ ] Implement new() constructor
- [ ] Implement start_session() method
- [ ] Implement end_session() method
- [ ] Implement get_current_session() method
- [ ] Implement get_session() method
- [ ] Implement abandon_session() method
- [ ] Write unit tests
- [ ] Run tests to verify
- [ ] Proceed to TASK-P1-007
