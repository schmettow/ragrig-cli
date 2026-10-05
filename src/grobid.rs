//! Optional GROBID pre-pass for directory corpora.
//!
//! With `--embed-rename` (and the `grobid` cargo feature), every new PDF in a
//! directory corpus is parsed by a GROBID server, its header metadata is
//! completed against OpenAlex, and the file is renamed to
//! `Key - Full authors - Full title - Year` (citation key, all authors with
//! full names, punctuation-stripped title, year) before it is indexed.
//! ragrig embeds chunk provenance including the file name, so meaningful
//! names give the chat agent stable, human-readable citations.
//!
//! The work is delegated to the [`grobid_bibtex`] crate:
//! [`grobid_bibtex::files::collect_pdfs`] and
//! [`grobid_bibtex::files::Manifest`] (a size/mtime fingerprint sidecar so
//! unchanged files are skipped), [`grobid_bibtex::extract::headers`] (bounded
//! GROBID batch), [`grobid_bibtex::complete::biblios`] (OpenAlex completion)
//! and [`grobid_bibtex::files::rename_pdfs_with`] (keyed file stems with year
//! collision numbering).
//!
//! The pre-pass is resilient by design:
//!
//! * a PDF GROBID cannot parse keeps its name and is still indexed;
//! * an OpenAlex lookup that fails leaves the GROBID header unchanged;
//! * a failed rename keeps the original path.
//!
//! Only an unreachable GROBID server — or a corpus in which every PDF fails —
//! aborts startup, because the user explicitly asked for the pre-pass.

use std::path::Path;

use anyhow::Result;

/// Runtime configuration for the GROBID pre-pass, built from
/// `--embed-rename`, `--grobid-url`, and `--grobid-workers`.
#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "grobid"), allow(dead_code))]
pub struct GrobidRenameConfig {
    /// Base URL of the GROBID server.
    pub url: String,
    /// Maximum number of concurrent GROBID/OpenAlex requests.
    pub workers: usize,
}

/// Parse, complete, and rename the PDFs in `folder`.
///
/// `cfg` is `None` when `--embed-rename` was not given, making this a no-op.
pub async fn rename_pdfs(cfg: Option<&GrobidRenameConfig>, folder: &Path) -> Result<()> {
    let Some(cfg) = cfg else {
        return Ok(());
    };
    #[cfg(feature = "grobid")]
    {
        imp::rename_pdfs(cfg, folder).await
    }
    #[cfg(not(feature = "grobid"))]
    {
        let _ = (cfg, folder);
        anyhow::bail!(
            "`--embed-rename` needs a build with the `grobid` cargo feature \
             (cargo build --features grobid)"
        )
    }
}

#[cfg(feature = "grobid")]
mod imp {
    use std::path::Path;
    use std::time::Duration;

    use anyhow::{Context, Result, bail};
    use grobid_bibtex::bibtex::{FileStemOptions, FileStemStyle};
    use grobid_bibtex::files::{self, Collision, Manifest, Rename};
    use grobid_bibtex::openalex::Completer;
    use grobid_bibtex::{GrobidClient, ProcessOptions, complete, extract};
    use log::{info, warn};

    use super::GrobidRenameConfig;

    /// Sidecar manifest with per-file fingerprints of everything already
    /// processed, so unchanged PDFs are not parsed again.
    const MANIFEST_NAME: &str = ".ragrig_grobid.json";

    /// Number of liveness probes before giving up on the server.
    const PROBE_ATTEMPTS: usize = 5;

    /// Delay between liveness probes; a cold GROBID container can take a
    /// while to preload its models.
    const PROBE_DELAY: Duration = Duration::from_secs(3);

    /// Per-request timeout for OpenAlex lookups, so an unreachable API host
    /// cannot stall a worker for the OS TCP timeout.
    const OPENALEX_TIMEOUT: Duration = Duration::from_secs(15);

    /// Rename policy: `Key - Full authors - Full title - Year`; syntax
    /// characters are stripped, Unicode letters are kept (the citation key
    /// itself is always ASCII).
    const RENAME_OPTIONS: FileStemOptions = FileStemOptions {
        style: FileStemStyle::Keyed,
        // `title_words` only applies to `Compact`; `Keyed` keeps the full title.
        title_words: 10,
        ascii_only: false,
    };

    pub(super) async fn rename_pdfs(cfg: &GrobidRenameConfig, folder: &Path) -> Result<()> {
        let manifest_path = folder.join(MANIFEST_NAME);
        let mut manifest = match Manifest::load(&manifest_path) {
            Ok(manifest) => manifest,
            Err(err) => {
                warn!("GROBID: ignoring corrupt manifest: {err}");
                Manifest::empty(&manifest_path)
            }
        };
        let pdfs: Vec<_> = match files::collect_pdfs(folder) {
            Ok(pdfs) => pdfs,
            // A missing folder is reported by the sync that follows; the
            // pre-pass itself has nothing to do.
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(err) => {
                return Err(err).with_context(|| format!("scanning {} for PDFs", folder.display()));
            }
        };
        let pdfs: Vec<_> = pdfs
            .into_iter()
            .filter(|path| !manifest.is_unchanged(folder, path))
            .collect();
        if pdfs.is_empty() {
            info!(
                "GROBID pre-pass: no new or changed PDFs in {}.",
                folder.display()
            );
            return Ok(());
        }
        let total = pdfs.len();

        info!(
            "GROBID pre-pass: {} PDF(s) in {} — parsing headers, completing against \
             OpenAlex, renaming to Key - Full authors - Full title - Year ({} worker(s)).",
            total,
            folder.display(),
            cfg.workers.max(1)
        );

        let client = GrobidClient::new(&cfg.url)
            .with_context(|| format!("invalid GROBID server URL {:?}", cfg.url))?;
        info!("GROBID pre-pass: waiting for the server at {} ...", cfg.url);
        client
            .wait_until_ready(PROBE_ATTEMPTS, PROBE_DELAY)
            .await
            .with_context(|| {
                format!(
                    "GROBID server at {} is not ready — start a GROBID server or drop \
                     --embed-rename",
                    cfg.url
                )
            })?;
        let completer =
            Completer::with_timeout(OPENALEX_TIMEOUT).context("building the OpenAlex completer")?;

        // Parse the headers with a bounded number of concurrent requests.
        let outcomes =
            extract::headers(&client, pdfs, ProcessOptions::default(), cfg.workers.max(1)).await;
        let mut results = Vec::new();
        let mut failed = 0usize;
        for outcome in outcomes {
            match outcome {
                Ok(extract::Extracted { path, record }) => results.push((path, record)),
                Err(err) => {
                    failed += 1;
                    warn!("GROBID: skipping {}: {}", err.path.display(), err.source);
                }
            }
        }
        if results.is_empty() {
            bail!(
                "GROBID pre-pass: all {failed} PDF(s) failed — is the server at {} running?",
                cfg.url
            );
        }

        // Complete the parsed headers against OpenAlex; failures keep the
        // GROBID header as parsed.
        let report = complete::biblios(&completer, results, cfg.workers.max(1)).await;
        for failure in &report.failures {
            warn!(
                "OpenAlex lookup failed for {}: {}",
                report.records[failure.position - 1].0.display(),
                failure.source
            );
        }
        let completed = report.matched;
        let mut results = report.records;

        // Rename in deterministic path order; collisions are numbered on the
        // year, so authors and title keep their place in the name. Every
        // processed file is recorded under its final name, including failed
        // renames and files that already had the right name.
        let mut renamed = 0usize;
        for event in files::rename_pdfs_with(&mut results, &RENAME_OPTIONS, Collision::Year) {
            match event {
                Rename::Renamed { from, to } => {
                    info!("GROBID: renamed {} → {}", from.display(), to.display());
                    renamed += 1;
                    manifest.record(folder, &to);
                }
                Rename::Unchanged { path } => manifest.record(folder, &path),
                Rename::Unnamed { path } => {
                    info!(
                        "GROBID: no usable author/year/title for {}; keeping name",
                        path.display()
                    );
                    manifest.record(folder, &path);
                }
                Rename::Failed { from, source, .. } => {
                    warn!("GROBID: cannot rename {}: {source}", from.display());
                    manifest.record(folder, &from);
                }
            }
        }
        manifest.prune(folder);
        if let Err(err) = manifest.save() {
            warn!("GROBID: {err}");
        }

        info!(
            "GROBID pre-pass done: {} of {} PDF(s) renamed, {} OpenAlex-completed, {} failed.",
            renamed, total, completed, failed
        );
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use grobid_bibtex::{Author, Biblio};

        fn biblio(author: Option<&str>, year: Option<&str>, title: Option<&str>) -> Biblio {
            Biblio {
                authors: author
                    .map(|surname| Author {
                        surname: Some(surname.to_string()),
                        ..Author::default()
                    })
                    .into_iter()
                    .collect(),
                date: year.map(str::to_string),
                title: title.map(str::to_string),
                ..Biblio::default()
            }
        }

        #[test]
        fn target_is_key_full_authors_title_year() {
            let biblio = biblio(
                Some("Kahle"),
                Some("2000-03-01"),
                Some("The Barc model for continuous variables with extra words beyond ten"),
            );
            let target = grobid_bibtex::bibtex::suggest_file_name_with(
                Path::new("/papers/orig.pdf"),
                &biblio,
                &RENAME_OPTIONS,
            )
            .unwrap();
            assert_eq!(
                target.file_name().unwrap(),
                "Kahle2000 - Kahle - The Barc model for continuous variables with extra words beyond ten - 2000.pdf"
            );
        }

        #[test]
        fn target_omits_missing_parts() {
            let biblio = biblio(None, Some("2019"), Some("A lone title"));
            let target = grobid_bibtex::bibtex::suggest_file_name_with(
                Path::new("x.pdf"),
                &biblio,
                &RENAME_OPTIONS,
            )
            .unwrap();
            // Without an author or year there is no key; the title leads.
            assert_eq!(target.file_name().unwrap(), "A lone title - 2019.pdf");
            assert!(
                grobid_bibtex::bibtex::suggest_file_name_with(
                    Path::new("x.pdf"),
                    &Biblio::default(),
                    &RENAME_OPTIONS
                )
                .is_none()
            );
        }
    }
}
