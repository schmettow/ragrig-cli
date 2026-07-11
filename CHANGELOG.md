# Changelog

All notable changes to ragrig-cli are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.8] — 2026-07-11

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
