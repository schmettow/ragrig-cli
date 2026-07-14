# Changelog

All notable changes to ragrig-cli are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
