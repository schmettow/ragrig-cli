//! Optional GROBID pre-pass for directory corpora.
//!
//! With `--embed-rename` and/or `--bibtex-merge` (and the `grobid` cargo
//! feature), every new or changed PDF in a directory corpus is parsed by a
//! GROBID server, its header metadata is completed against OpenAlex, and
//! then:
//!
//! * with `--embed-rename`, the file is renamed to
//!   `Key - Full authors - Full title - Year` (citation key, all authors with
//!   full names, punctuation-stripped title, year) before it is indexed;
//! * with `--bibtex-merge <FILE>`, its record is merged into `<FILE>` with
//!   the duplicate detection of `pdf2bibtex --merge` (normalized field
//!   content, identifiers, PDF file name). New entries are appended; the file
//!   is never rewritten or pruned. Each entry records its PDF's path in a
//!   `file` field unless `--bibtex-no-link` is given.
//!
//! ragrig embeds chunk provenance including the file name, so meaningful
//! names give the chat agent stable, human-readable citations; the BibTeX
//! file gives the corpus a bibliography that survives re-indexing and lets
//! the chat attach a paper when a query mentions its key (`@Key`).
//!
//! The orchestration is delegated to the [`grobid_bibtex::prepass`] pipeline:
//! the manifest and rename policy are the single source of truth shared with
//! the other hosts of the crate (see `UPSTREAM.md`, US-2).
//!
//! The pre-pass is resilient by design:
//!
//! * a PDF GROBID cannot parse keeps its name and is still indexed;
//! * an OpenAlex lookup that fails leaves the GROBID header unchanged;
//! * a failed rename keeps the original path.
//!
//! Only an unreachable GROBID server — or a corpus in which every PDF fails —
//! aborts startup, because the user explicitly asked for the pre-pass. When
//! there is nothing new to parse, cached records are merged into the
//! bibliography without contacting the server at all.

use std::path::{Path, PathBuf};

#[cfg(feature = "grobid")]
use anyhow::Context;
use anyhow::Result;

/// Runtime configuration for the GROBID pre-pass, built from
/// `--embed-rename`, `--bibtex-merge`, `--grobid-url` and `--grobid-workers`.
#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "grobid"), allow(dead_code))]
pub struct GrobidPrepassConfig {
    /// Base URL of the GROBID server.
    pub url: String,
    /// Maximum number of concurrent GROBID/OpenAlex requests.
    pub workers: usize,
    /// Rename processed PDFs to `Key - Full authors - Full title - Year`.
    pub rename: bool,
    /// Merge processed records into this BibTeX file.
    pub bibtex: Option<BibtexMergeConfig>,
}

/// The BibTeX file the pre-pass merges records into.
#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "grobid"), allow(dead_code))]
pub struct BibtexMergeConfig {
    /// The `.bib` file; created when the first entry is merged.
    pub path: PathBuf,
    /// Record each PDF's path in the entry's `file` field.
    pub link: bool,
}

/// Resolve the pre-pass CLI flags.
///
/// Returns `None` when neither `--embed-rename` nor `--bibtex-merge` was
/// given, making the pre-pass a no-op. `feature_enabled` is the caller's
/// `cfg!(feature = "grobid")`: asking for the pre-pass without the feature is
/// an error, as is an empty `--bibtex-merge` path.
pub fn resolve_prepass(
    embed_rename: bool,
    bibtex_merge: Option<&Path>,
    bibtex_link: bool,
    url: &str,
    workers: usize,
    feature_enabled: bool,
) -> Result<Option<GrobidPrepassConfig>> {
    let bibtex = match bibtex_merge {
        Some(path) if path.as_os_str().is_empty() => {
            anyhow::bail!("`--bibtex-merge` needs a non-empty .bib file path");
        }
        Some(path) => Some(BibtexMergeConfig {
            path: path.to_path_buf(),
            link: bibtex_link,
        }),
        None => None,
    };
    if !embed_rename && bibtex.is_none() {
        return Ok(None);
    }
    if !feature_enabled {
        anyhow::bail!(
            "`--embed-rename` and `--bibtex-merge` need a build with the `grobid` cargo feature: \
             cargo install ragrig-cli --features grobid (or cargo build --features grobid)"
        );
    }
    Ok(Some(GrobidPrepassConfig {
        url: url.to_string(),
        workers: workers.max(1),
        rename: embed_rename,
        bibtex,
    }))
}

/// Run the GROBID pre-pass for one directory `folder`.
///
/// `cfg` is `None` when neither `--embed-rename` nor `--bibtex-merge` was
/// given, making this a no-op.
pub async fn prepass(cfg: Option<&GrobidPrepassConfig>, folder: &Path) -> Result<()> {
    let Some(cfg) = cfg else {
        return Ok(());
    };
    #[cfg(feature = "grobid")]
    {
        use grobid_bibtex::prepass::PrepassError;

        let config = pipeline_config(cfg, folder);
        let result = grobid_bibtex::prepass::prepass(&config)
            .await
            .map(|_report| ());
        result.map_err(|error| match error {
            PrepassError::NotReady { url, source } => anyhow::Error::new(source).context(format!(
                "GROBID server at {url} is not ready while preparing {} \u{2014} start a \
                     GROBID server or drop --embed-rename/--bibtex-merge",
                folder.display()
            )),
            other => anyhow::Error::new(other)
                .context(format!("GROBID pre-pass failed for {}", folder.display())),
        })
    }
    #[cfg(not(feature = "grobid"))]
    {
        let _ = (cfg, folder);
        anyhow::bail!(
            "`--embed-rename` and `--bibtex-merge` need a build with the `grobid` cargo feature \
             (cargo build --features grobid)"
        )
    }
}

/// Translate the CLI config for one folder into the library pipeline config.
#[cfg(feature = "grobid")]
fn pipeline_config(
    cfg: &GrobidPrepassConfig,
    folder: &Path,
) -> grobid_bibtex::prepass::PrepassConfig {
    use grobid_bibtex::prepass;

    let mut config = prepass::PrepassConfig::new(folder);
    config.grobid_url = cfg.url.clone();
    config.workers = cfg.workers.max(1);
    config.rename = cfg.rename;
    config.bibtex = cfg.bibtex.as_ref().map(|bibtex| prepass::BibtexTarget {
        path: bibtex.path.clone(),
        link: bibtex.link,
    });
    config
}

/// Parse the configured BibTeX target and return its entry count; a missing
/// file counts as empty.
///
/// # Errors
///
/// Returns an error when the file exists but is not a valid bibliography, or
/// when this build has no `grobid` feature.
pub fn bibtex_entry_count(cfg: &BibtexMergeConfig) -> Result<usize> {
    #[cfg(feature = "grobid")]
    {
        grobid_bibtex::Collection::load_or_new(&cfg.path)
            .map(|collection| collection.len())
            .with_context(|| format!("cannot read the BibTeX file {}", cfg.path.display()))
    }
    #[cfg(not(feature = "grobid"))]
    {
        let _ = cfg;
        anyhow::bail!(
            "`--bibtex-merge` needs a build with the `grobid` cargo feature \
             (cargo build --features grobid)"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_prepass_is_none_without_flags() {
        let cfg =
            resolve_prepass(false, None, true, "http://localhost:8070", 4, true).expect("no flags");
        assert!(cfg.is_none());
    }

    #[test]
    fn resolve_prepass_rename_only() {
        let cfg = resolve_prepass(true, None, true, "http://grobid:8070", 0, true)
            .expect("rename")
            .expect("configured");
        assert!(cfg.rename);
        assert!(cfg.bibtex.is_none());
        assert_eq!(cfg.url, "http://grobid:8070");
        // The worker count is clamped to at least one.
        assert_eq!(cfg.workers, 1);
    }

    #[test]
    fn resolve_prepass_bibtex_only() {
        let cfg = resolve_prepass(
            false,
            Some(Path::new("refs.bib")),
            false,
            "http://localhost:8070",
            4,
            true,
        )
        .expect("bibtex")
        .expect("configured");
        // BibTeX maintenance runs the pre-pass without renaming files.
        assert!(!cfg.rename);
        let bibtex = cfg.bibtex.expect("target");
        assert_eq!(bibtex.path, PathBuf::from("refs.bib"));
        assert!(!bibtex.link);
    }

    #[test]
    fn resolve_prepass_both_flags() {
        let cfg = resolve_prepass(
            true,
            Some(Path::new("out/refs.bib")),
            true,
            "http://localhost:8070",
            2,
            true,
        )
        .expect("both")
        .expect("configured");
        assert!(cfg.rename);
        assert_eq!(cfg.workers, 2);
        let bibtex = cfg.bibtex.expect("target");
        assert_eq!(bibtex.path, PathBuf::from("out/refs.bib"));
        assert!(bibtex.link);
    }

    #[test]
    fn resolve_prepass_without_feature_errors() {
        let error = resolve_prepass(false, Some(Path::new("refs.bib")), true, "u", 1, false)
            .expect_err("feature missing");
        assert!(error.to_string().contains("grobid"), "{error}");
        let error = resolve_prepass(true, None, true, "u", 1, false).expect_err("feature missing");
        assert!(error.to_string().contains("grobid"), "{error}");
    }

    #[test]
    fn resolve_prepass_rejects_empty_path() {
        let error = resolve_prepass(false, Some(Path::new("")), true, "u", 1, true)
            .expect_err("empty path");
        assert!(error.to_string().contains("non-empty"), "{error}");
    }

    #[cfg(feature = "grobid")]
    #[test]
    fn pipeline_config_keeps_the_rename_policy_and_merge_target() {
        let folder = Path::new("/papers");
        let cfg = GrobidPrepassConfig {
            url: "http://grobid:8070".to_string(),
            workers: 2,
            rename: true,
            bibtex: Some(BibtexMergeConfig {
                path: PathBuf::from("/papers/refs.bib"),
                link: false,
            }),
        };
        let pipeline = pipeline_config(&cfg, folder);
        assert_eq!(pipeline.folder, PathBuf::from(folder));
        assert!(pipeline.sources.is_none());
        assert!(pipeline.rename);
        assert!(pipeline.openalex);
        assert_eq!(pipeline.workers, 2);
        let bibtex = pipeline.bibtex.expect("target");
        assert_eq!(bibtex.path, PathBuf::from("/papers/refs.bib"));
        assert!(!bibtex.link);
    }
}
