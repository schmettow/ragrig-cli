# Changelog

All notable changes to ragrig-cli are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0]

### Added

- **`--demo`** — start in demo mode: the small `llama3.2:3b` chat model with
  a 4096-token context (fits an 8 GB GPU), memory off, the embedded HTML
  fixture book (Martin Schmettow, *New Statistics for Design Researchers*,
  https://schmettow.github.io/New_Stats/) as the document corpus, a greeting
  listing the required Ollama models, and the first question prefilled in
  the prompt line.  Behind the `test-fixtures` feature (on by default).
- **`/chunker [name]`** — show or hot-swap the chunking strategy.  Every
  stored chunk records which (parser, chunker, embedder) pipeline built it;
  swapping to a pipeline that has no chunks in the store prints a warning
  and never re-embeds automatically (`/embed index` does).
- **Named document corpora** — `--corpus-dir name=dir` (repeatable) and
  `--corpus-urls name=url` (repeatable; same name accumulates one curated
  URL list).  Several corpora share one vector store, each keeping its own
  provenance identity.
- **`/corpus <name> on|off`** — toggle a named corpus (syncs it in, or
  removes its chunks) and **`/corpus dyn on|off`** — dynamic routing for
  web downloads into the first active URL corpus, else the first active
  directory corpus, else the main folder.
- **`--workspace <DIR>`** — state directory for store, history, sessions,
  and profiles.  `--folder <DIR>` remains as a shortcut for
  `--workspace <DIR> --corpus-dir folder=<DIR>`.
- **Chunker feature passthroughs** — `chunker-textsplitter` and
  `chunker-cognigraph` make the feature-gated chunkers available to
  `/chunker` (both are included in `all`).

### Changed

- **Aligned with ragrig 1.0.0** — the REPL builds against the current
  library API: the `Corpus` trait (`source` terminology retired; legacy
  profiles with `"source_dirs"`/`"source_urls"` still load), rig-core's
  completion API, the vendored chunkedrs fork with its overlap fixes, and
  the dimension-agnostic LanceDB store.
- **Chat turns route through `AgentSession`** — the REPL no longer keeps
  its own transcript/persistence bookkeeping (`prompt_memory`,
  `session_store`, `memory_enabled` are gone).  Queries go through
  `AgentSession::chat_detailed(_with_attachments)`, which accumulates
  turns, diffuses history, and auto-saves; `/memory log|summary|transcript|off|purge`
  map onto `set_history_strategy` / `set_use_transcript` / `clear_turns`,
  `/hist` onto `load_session` / `list_sessions` / `delete_session`, and
  session ids come from `SessionId::new()`.
- **Streaming answers with a token counter and ESC-cancel** — queries now
  stream token-by-token (`chat_streaming_detailed(_with_attachments)`);
  the info header reports the received-token count, and pressing **ESC**
  cancels the in-flight generation (a raw-mode stdin watcher flips a
  `CancellationToken`; the terminal is restored on drop, and non-TTY
  stdin degrades to a no-op).
- **Embedding progress bar** — bootstrap indexing, `/embed index`, and
  `/corpus on` render a live progress bar (files processed, chunks
  embedded, failures) on stderr via the library's `Progress` events;
  ESC cancels indexing between documents (`Cancelled` is handled
  gracefully).
- **Ingestion routes through the agent** — bootstrap, `/embed index`, and
  `/corpus on` now call the agent's own `sync_corpus` / `reindex_corpus`
  (which carry the parser registry, chunk config, and chunker) instead of
  bypassing `RagAgent` with `vector::ingest_corpus`/`sync_corpus`;
  `/parser` swaps keep the agent's registry in sync.
- **Default chat model `qwen3.5:9b`** and **context budget 4096 tokens**
  (previously `gemma2:latest` and 8192).

## [0.9.8] — 2026-07-11

### Added

- **`/attach <file>` command** — attach a PDF, EPUB, DOCX, HTML, or Markdown
  file for the next RAG query.  The file is parsed inline (using the active
  document parsers) and its full text is injected into the prompt alongside
  vector store results via the `PrependAttach` strategy.  Attachments are
  one-shot — they are cleared automatically after each query.
  - `/attach` — list currently attached files.
  - `/attach clear` — remove all attachments.
- **`/search by <file>` command** — use a document file as the search query.
  Parses the file, chunks it, embeds each chunk, and finds similar documents
  in the database via `search_by_document()`.  Results are stored in
  `last_results` so `/refs` and `/get` work automatically.

### Changed

- **Aligned with ragrig v0.9.8 API** — all breaking changes from the library
  migration are reflected:

  - `ChatAgentSpec::ollama()` and `ChatAgentSpec::deepseek()` now pass
    `request_timeout_secs: None` as the final argument.
  - `EmbedderSpec::Ollama { … }` struct literals include the new
    `request_timeout_secs: None` field.
  - `ChatConfig` and `EmbedConfig` struct literals include
    `request_timeout_secs: None`.
  - `ChunkConfig { size, overlap }` struct literals replaced with the
    validated `ChunkConfig::new(size, overlap)?` constructor.
  - `download_and_ingest_url()` calls pass `Some(DEFAULT_MAX_DOWNLOAD_BYTES)`
    (50 MiB cap) as the new final parameter.  The constant is now
    re-exported from `ragrig`.
  - `RagAgentBuilder::build()` now returns `Result` — call sites
    unwrap with `?`.
