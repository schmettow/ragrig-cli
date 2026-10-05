# Roadmap

Planned work for upcoming versions.  Entries describe intended
non-breaking changes until they ship in a release.


# Changelog

All notable changes to ragrig-cli are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## v1.2.0

### Added

- **`/embed update`** — incremental re-index: runs the GROBID pre-pass for
  new PDFs (when `--embed-rename` is on) and embeds only new and changed
  documents, reported with the same per-file table as `/embed index`.
  Documents that disappeared from a corpus are dropped from the store.
  `/embed index` keeps its full re-embed semantics.
- **`/chat show`**, **`/embed show`**, **`/memory show`** — print the full
  settings of the running pipeline: chat backend, model, context window and
  mode, temperature, top-p, `max_tokens`, seed; embedding backend, model,
  top-k, similarity threshold, chunker, chunk size/overlap, parser, store
  size, GROBID pre-pass; memory mode, history diffusion, rewriter, and turn
  count.  The generation parameters are mirrored from `/chat` hot-swaps and
  profile loads, and changing one sampling parameter no longer resets the
  others.

### Changed

- **GROBID pre-pass on `grobid-bibtex`** — the pre-pass now uses the shared
  `grobid-bibtex` crate for extraction, OpenAlex completion and renaming
  (its `Manifest` and keyed-name policy moved there), instead of depending
  on `grobid` directly. Behavior is unchanged; OpenAlex lookups are now
  paced at the keyless pool's 10 requests/s like the other tools. The
  `grobid` cargo feature keeps its name.
- **GROBID rename format** — renamed PDFs are now
  `<key> - <full authors> - <title> - <year>`, e.g.
  `Kahle2000 - Brewster Kahle - The Barc model for continuous variables - 2000.pdf`:
  the citation key comes first (always ASCII), authors keep their full given
  and family names, the title is complete and stripped of syntax characters,
  and the year moves to the end.  Colliding names still number the year
  (`... - 2000 - Title`, `... - 2000-1 - Title`) so authors and title keep
  their place.  Needs the `grobid` cargo feature (`grobid-bibtex` 0.1).

## v1.2.0

### Added

- **`--embed-rename`** (with the new `grobid` cargo feature) — before a
  directory corpus is indexed, every new PDF is parsed by a GROBID server,
  its header metadata is completed against OpenAlex, and the file is renamed
  to `Author1, Author2, ... - Year - Full title`: all authors, the year, and
  the complete title with punctuation stripped.  Since chunk provenance
  includes the file name, the chat agent can cite documents by their real
  title.  `--grobid-url` (default `http://localhost:8070`) and
  `--grobid-workers` (default 4) configure the pre-pass; processed files are
  fingerprinted in `.ragrig_grobid.json` so the parse/lookup pass only runs
  for new or changed PDFs.  Colliding names are deduplicated by numbering
  the year (`... - 2020-1 - Title.pdf`).  The pre-pass also runs after
  `/download` and `/get` save a file into the main folder, and waits up to
  15 s for a cold GROBID container.  Build with
  `cargo install ragrig-cli --features grobid`; without the feature,
  `--embed-rename` reports how to enable it.  The README documents the
  container setup with links to the GROBID instructions.

## [1.1.0]

### Added

- **`--memory-strategy <MODE>`** — set the memory handling mode from the
  CLI: `rewrite` (default), `transcript`, `log`, `summary`, or `off`.
  The default value leaves a profile's stored strategy untouched when
  CLI overrides are merged on top.

### Changed

- **Cross-platform REPL UI (crossterm + indicatif)** — the hand-rolled
  terminal plumbing was replaced by purpose-built, cross-platform crates:

  - **ESC-cancel watcher** now reads key events via crossterm (raw mode
    plus `event::poll`/`event::read`), replacing the Unix-only `nix`
    termios/poll implementation and its Windows no-op stub.  ESC-cancel
    therefore works on native Windows again, and the `nix` dependency
    is gone.
  - **Embedding progress bar** is now rendered by indicatif with the
    same `[####----] 7/12 files | 341 chunks | … (ESC: cancel)` format,
    and it auto-hides when the output is not a TTY (no more `\r`
    garbage in piped/CI logs).

  Non-breaking: same commands, output format, and cancellation
  semantics; history and session files are unchanged.

- **Profile management backed by the library** — profiles now persist
  the full memory state through ragrig's new `MemoryConfig.strategy`
  field, so the CLI-side `ProfileWrapper` serialisation workaround is
  gone.  `/profile save|load` and `--profile` use the library's
  `save_to_profile` / `load_from_profile`, which no longer write the
  machine-specific workspace into the file (profiles are portable
  across machines) and force the caller's workspace on load.  The
  runtime strategy is synced back via the library's
  `AgentSession::memory_strategy()`.

### Fixed

- **`/memory` misreported transcript mode as off** — the status line
  derived the state from the rewriter alone, so a transcript-strategy
  profile (no rewriter, transcript on) displayed `Memory: off` after a
  `--profile` restart or `/profile load`, although the conversation was
  passed to the model and turns accumulated correctly.  The status now
  reports `transcript` via `use_transcript()`.

## v1.0.2

### Added

- **`/attach` accepts any UTF-8 text file** — files whose extension has no
  registered document parser (BibTeX `.bib`, `.txt`, `.csv`, …) are now read
  directly as UTF-8 text and injected into the next query, instead of being
  rejected with "No document parser registered".  Formats with a registered
  parser (PDF, EPUB, DOCX, HTML, Markdown) keep going through the parser
  pipeline, and a parse failure there remains a hard error — there is no
  silent fallback to raw bytes.

## [1.0.1]

### Fixed

- **Windows MSVC builds** — the ESC-cancel watcher used `nix` (termios +
  poll) and `std::os::fd`, which do not exist on Windows, so the crate
  failed to compile for `x86_64-pc-windows-msvc`.  The watcher is now
  compiled on Unix only; native Windows builds get an inert no-op stub
  and simply run operations to completion (ESC cancellation is
  currently unavailable there — noted in the README).  `nix` is now a
  `cfg(unix)`-only dependency.

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
