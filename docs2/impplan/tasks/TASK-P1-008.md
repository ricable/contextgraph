# TASK-P1-008: MDFileWatcher

```xml
<task_spec id="TASK-P1-008" version="1.0">
<metadata>
  <title>MDFileWatcher Implementation</title>
  <status>ready</status>
  <layer>surface</layer>
  <sequence>13</sequence>
  <phase>1</phase>
  <implements>
    <requirement_ref>REQ-P1-05</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-P1-007</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
</metadata>

<context>
Implements the MDFileWatcher component that monitors directories for markdown
file changes and automatically chunks and captures them as memories.

Uses the notify crate for filesystem watching with debouncing to handle
rapid file changes efficiently.
</context>

<input_context_files>
  <file purpose="component_spec">docs2/impplan/technical/TECH-PHASE1-MEMORY-CAPTURE.md#component_contracts</file>
  <file purpose="chunker">crates/context-graph-core/src/memory/chunker.rs</file>
  <file purpose="capture_service">crates/context-graph-core/src/memory/capture.rs</file>
</input_context_files>

<prerequisites>
  <check>TASK-P1-007 complete (MemoryCaptureService exists)</check>
  <check>notify crate available</check>
</prerequisites>

<scope>
  <in_scope>
    - Create MDFileWatcher struct
    - Implement file system watching with notify
    - Handle Create/Modify events for .md files
    - Debounce rapid changes (1000ms)
    - Track file hashes to detect actual changes
    - Process files through TextChunker
    - Store via MemoryCaptureService
    - Add WatcherError enum
  </in_scope>
  <out_of_scope>
    - CLI integration (Phase 6)
    - Configuration file parsing
    - Recursive directory watching (optional enhancement)
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/memory/watcher.rs">
      pub struct MDFileWatcher {
          watcher: RecommendedWatcher,
          capture_service: Arc&lt;MemoryCaptureService&gt;,
          chunker: TextChunker,
          file_hashes: Arc&lt;RwLock&lt;HashMap&lt;PathBuf, String&gt;&gt;&gt;,
      }

      impl MDFileWatcher {
          pub fn new(watch_paths: Vec&lt;PathBuf&gt;, capture_service: Arc&lt;MemoryCaptureService&gt;, session_id: String) -> Result&lt;Self, WatcherError&gt;;
          pub async fn start(&amp;mut self) -> Result&lt;(), WatcherError&gt;;
          pub fn stop(&amp;mut self);
      }
    </signature>
  </signatures>

  <constraints>
    - Only process .md files
    - Debounce events for 1000ms
    - Skip files that haven't changed (hash comparison)
    - Handle errors without crashing watcher
    - Log all processed files and errors
  </constraints>

  <verification>
    - Watcher detects new MD files
    - Watcher detects modified MD files
    - Hash comparison prevents duplicate processing
    - Debouncing prevents rapid reprocessing
  </verification>
</definition_of_done>

<pseudo_code>
File: crates/context-graph-core/src/memory/watcher.rs

use notify::{Watcher, RecommendedWatcher, Config, RecursiveMode, Event, EventKind};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use sha2::{Sha256, Digest};
use thiserror::Error;
use std::time::Duration;

use super::{
    TextChunker, MemoryCaptureService,
    chunker::ChunkerError,
    capture::CaptureError,
};

#[derive(Debug, Error)]
pub enum WatcherError {
    #[error("Path not found: {path:?}")]
    PathNotFound { path: PathBuf },
    #[error("Watch failed: {0}")]
    WatchFailed(#[from] notify::Error),
    #[error("Read failed for {path:?}: {source}")]
    ReadFailed { path: PathBuf, source: std::io::Error },
    #[error("Capture failed: {0}")]
    CaptureFailed(#[from] CaptureError),
    #[error("Chunking failed: {0}")]
    ChunkingFailed(#[from] ChunkerError),
}

pub struct MDFileWatcher {
    watcher: Option&lt;RecommendedWatcher&gt;,
    capture_service: Arc&lt;MemoryCaptureService&gt;,
    chunker: TextChunker,
    file_hashes: Arc&lt;RwLock&lt;HashMap&lt;PathBuf, String&gt;&gt;&gt;,
    session_id: String,
    watch_paths: Vec&lt;PathBuf&gt;,
    event_rx: Option&lt;std::sync::mpsc::Receiver&lt;Result&lt;Event, notify::Error&gt;&gt;&gt;,
}

impl MDFileWatcher {
    pub fn new(
        watch_paths: Vec&lt;PathBuf&gt;,
        capture_service: Arc&lt;MemoryCaptureService&gt;,
        session_id: String,
    ) -> Result&lt;Self, WatcherError&gt; {
        // Validate all paths exist
        for path in &amp;watch_paths {
            if !path.exists() {
                return Err(WatcherError::PathNotFound { path: path.clone() });
            }
        }

        let chunker = TextChunker::default_config();

        Ok(Self {
            watcher: None,
            capture_service,
            chunker,
            file_hashes: Arc::new(RwLock::new(HashMap::new())),
            session_id,
            watch_paths,
            event_rx: None,
        })
    }

    pub async fn start(&amp;mut self) -> Result&lt;(), WatcherError&gt; {
        let (tx, rx) = std::sync::mpsc::channel();

        let config = Config::default()
            .with_poll_interval(Duration::from_millis(1000));

        let mut watcher = RecommendedWatcher::new(
            move |res| { tx.send(res).ok(); },
            config,
        )?;

        for path in &amp;self.watch_paths {
            watcher.watch(path, RecursiveMode::NonRecursive)?;
        }

        self.watcher = Some(watcher);
        self.event_rx = Some(rx);

        // Initial scan of existing files
        for path in &amp;self.watch_paths {
            self.scan_directory(path).await?;
        }

        Ok(())
    }

    pub async fn process_events(&amp;mut self) -> Result&lt;(), WatcherError&gt; {
        if let Some(ref rx) = self.event_rx {
            while let Ok(result) = rx.try_recv() {
                match result {
                    Ok(event) => {
                        self.handle_event(event).await?;
                    }
                    Err(e) => {
                        tracing::error!("Watch error: {}", e);
                    }
                }
            }
        }
        Ok(())
    }

    async fn handle_event(&amp;self, event: Event) -> Result&lt;(), WatcherError&gt; {
        match event.kind {
            EventKind::Create(_) | EventKind::Modify(_) => {
                for path in event.paths {
                    if self.is_markdown(&amp;path) {
                        self.process_file(&amp;path).await?;
                    }
                }
            }
            _ => {} // Ignore other events
        }
        Ok(())
    }

    async fn process_file(&amp;self, path: &amp;Path) -> Result&lt;Vec&lt;uuid::Uuid&gt;, WatcherError&gt; {
        // Read file content
        let content = std::fs::read_to_string(path)
            .map_err(|e| WatcherError::ReadFailed {
                path: path.to_path_buf(),
                source: e
            })?;

        // Compute hash
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let hash = format!("{:x}", hasher.finalize());

        // Check if file changed
        {
            let hashes = self.file_hashes.read().await;
            if let Some(existing_hash) = hashes.get(path) {
                if existing_hash == &amp;hash {
                    tracing::debug!("File unchanged, skipping: {:?}", path);
                    return Ok(Vec::new());
                }
            }
        }

        // Update hash
        {
            let mut hashes = self.file_hashes.write().await;
            hashes.insert(path.to_path_buf(), hash);
        }

        // Chunk the content
        let path_str = path.to_string_lossy().to_string();
        let chunks = self.chunker.chunk_text(&amp;content, &amp;path_str)?;

        tracing::info!("Processing file {:?}: {} chunks", path, chunks.len());

        // Capture each chunk
        let mut memory_ids = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let id = self.capture_service
                .capture_md_chunk(chunk, self.session_id.clone())
                .await?;
            memory_ids.push(id);
        }

        Ok(memory_ids)
    }

    async fn scan_directory(&amp;self, dir: &amp;Path) -> Result&lt;(), WatcherError&gt; {
        if dir.is_file() {
            if self.is_markdown(dir) {
                self.process_file(dir).await?;
            }
            return Ok(());
        }

        let entries = std::fs::read_dir(dir)
            .map_err(|e| WatcherError::ReadFailed {
                path: dir.to_path_buf(),
                source: e
            })?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() &amp;&amp; self.is_markdown(&amp;path) {
                self.process_file(&amp;path).await?;
            }
        }

        Ok(())
    }

    fn is_markdown(&amp;self, path: &amp;Path) -> bool {
        path.extension()
            .map(|ext| ext == "md" || ext == "markdown")
            .unwrap_or(false)
    }

    pub fn stop(&amp;mut self) {
        self.watcher = None;
        self.event_rx = None;
        tracing::info!("MDFileWatcher stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::fs;

    #[tokio::test]
    async fn test_watcher_processes_md_file() {
        // Setup test environment
        let dir = tempdir().unwrap();
        let md_path = dir.path().join("test.md");
        fs::write(&amp;md_path, "# Test\n\nThis is test content.").unwrap();

        // Would need full setup with mocked capture service
        // Abbreviated for space
    }
}
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-core/src/memory/watcher.rs">MDFileWatcher implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/memory/mod.rs">Add pub mod watcher and re-export</file>
  <file path="crates/context-graph-core/Cargo.toml">Add notify dependency if not present</file>
</files_to_modify>

<validation_criteria>
  <criterion>Watcher validates paths exist</criterion>
  <criterion>Watcher detects new .md files</criterion>
  <criterion>Watcher detects modified .md files</criterion>
  <criterion>Hash comparison prevents duplicate processing</criterion>
  <criterion>Files are chunked and captured correctly</criterion>
  <criterion>Errors are logged but don't crash watcher</criterion>
  <criterion>stop() cleanly shuts down watching</criterion>
</validation_criteria>

<test_commands>
  <command description="Run watcher tests">cargo test --package context-graph-core watcher</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
</test_commands>

<notes>
  <note category="debouncing">
    The notify crate's poll interval provides debouncing.
    Set to 1000ms to batch rapid file saves.
  </note>
  <note category="hash_tracking">
    File hashes are kept in memory for the watcher session.
    Could be persisted to DB for cross-session duplicate detection.
  </note>
</notes>
</task_spec>
```

## Execution Checklist

- [ ] Add notify crate to Cargo.toml
- [ ] Create watcher.rs in memory directory
- [ ] Implement WatcherError enum
- [ ] Implement MDFileWatcher struct
- [ ] Implement new() with path validation
- [ ] Implement start() with notify setup
- [ ] Implement process_file() with hash checking
- [ ] Implement scan_directory() for initial scan
- [ ] Implement stop() for clean shutdown
- [ ] Write integration tests
- [ ] Run tests to verify
- [ ] Phase 1 complete!
