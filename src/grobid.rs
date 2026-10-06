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
//! file gives the corpus a bibliography that survives re-indexing.
//!
//! The work is delegated to the [`grobid_bibtex`] crate:
//! [`grobid_bibtex::files::collect_pdfs`] and
//! [`grobid_bibtex::files::Manifest`] (a size/mtime fingerprint sidecar that
//! also caches each file's extracted record, so unchanged PDFs are neither
//! re-parsed nor lost from a rebuilt bibliography),
//! [`grobid_bibtex::extract::headers`] (bounded GROBID batch),
//! [`grobid_bibtex::complete::biblios`] (OpenAlex completion),
//! [`grobid_bibtex::files::rename_pdfs_with`] (keyed file stems with year
//! collision numbering) and [`grobid_bibtex::collection::merge_file`] (merge
//! into a `.bib` file, appending only new entries).
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
        imp::prepass(cfg, folder).await
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
        imp::bibtex_entry_count(cfg)
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
}

#[cfg(feature = "grobid")]
mod imp {
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use anyhow::{Context, Result, bail};
    use grobid_bibtex::bibtex::{FileStemOptions, FileStemStyle};
    use grobid_bibtex::collection::{self, Collection};
    use grobid_bibtex::files::{self, Collision, Manifest, Rename};
    use grobid_bibtex::openalex::Completer;
    use grobid_bibtex::{Biblio, GrobidClient, ProcessOptions, complete, extract};
    use log::{info, warn};

    use super::{BibtexMergeConfig, GrobidPrepassConfig};

    /// Sidecar manifest with per-file fingerprints of everything already
    /// processed, so unchanged PDFs are not parsed again. It also caches each
    /// file's extracted record, so a bibliography can be rebuilt from it.
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

    /// Parse the configured BibTeX target and return its entry count; a
    /// missing file counts as empty.
    pub(super) fn bibtex_entry_count(cfg: &BibtexMergeConfig) -> Result<usize> {
        let collection = Collection::load_or_new(&cfg.path)
            .with_context(|| format!("cannot read the BibTeX file {}", cfg.path.display()))?;
        Ok(collection.len())
    }

    pub(super) async fn prepass(cfg: &GrobidPrepassConfig, folder: &Path) -> Result<()> {
        // A corrupt bibliography must fail before any work: the pre-pass
        // could not record what it parses.
        if let Some(bibtex) = &cfg.bibtex {
            bibtex_entry_count(bibtex)?;
        }

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

        // Partition the PDFs: new and changed files need GROBID; unchanged
        // files with a cached record can be merged as they are. Unchanged
        // files without a cached record — e.g. after a run that only
        // renamed, or a manifest written before records were cached — are
        // parsed once so the cache and the bibliography catch up.
        let (needed, cached) = partition_pdfs(&manifest, cfg, folder, pdfs);
        if needed.is_empty() && cached.is_empty() {
            info!(
                "GROBID pre-pass: no new or changed PDFs in {}.",
                folder.display()
            );
            return Ok(());
        }

        // Cached records alone need no server: top up the bibliography
        // offline instead of probing GROBID.
        if needed.is_empty() {
            info!(
                "GROBID pre-pass: merging {} cached record(s) from {} (no PDFs to parse).",
                cached.len(),
                folder.display()
            );
            merge_records(cfg, &cached)?;
            return Ok(());
        }

        let total = needed.len();
        let mut notes = Vec::new();
        if cfg.rename {
            notes.push("renaming to Key - Full authors - Full title - Year");
        }
        if cfg.bibtex.is_some() {
            notes.push("merging into the BibTeX file");
        }
        let notes = if notes.is_empty() {
            String::new()
        } else {
            format!(", {}", notes.join(", "))
        };
        info!(
            "GROBID pre-pass: {} PDF(s) in {} — parsing headers, completing against \
             OpenAlex{notes} ({} worker(s)).",
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
                     --embed-rename/--bibtex-merge",
                    cfg.url
                )
            })?;
        let completer =
            Completer::with_timeout(OPENALEX_TIMEOUT).context("building the OpenAlex completer")?;

        // Parse the headers with a bounded number of concurrent requests.
        let outcomes = extract::headers(
            &client,
            needed,
            ProcessOptions::default(),
            cfg.workers.max(1),
        )
        .await;
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
        let processed = results.len();

        // Rename in deterministic path order; collisions are numbered on the
        // year, so authors and title keep their place in the name.
        let mut renamed = 0usize;
        if cfg.rename {
            for event in files::rename_pdfs_with(&mut results, &RENAME_OPTIONS, Collision::Year) {
                match event {
                    Rename::Renamed { from, to } => {
                        info!("GROBID: renamed {} → {}", from.display(), to.display());
                        renamed += 1;
                    }
                    Rename::Unnamed { path } => {
                        info!(
                            "GROBID: no usable author/year/title for {}; keeping name",
                            path.display()
                        );
                    }
                    Rename::Failed { from, source, .. } => {
                        warn!("GROBID: cannot rename {}: {source}", from.display());
                    }
                    Rename::Unchanged { .. } => {}
                }
            }
        }

        // Record every processed file under its final name, including failed
        // renames and files that already had the right name; the cached
        // record lets later runs merge without GROBID.
        for (path, biblio) in &results {
            manifest.record_with_biblio(folder, path, biblio);
        }
        manifest.prune(folder);
        if let Err(err) = manifest.save() {
            warn!("GROBID: {err}");
        }

        // Merge after renaming, so `file` fields point at the final names.
        let mut records = cached;
        records.extend(results);
        merge_records(cfg, &records)?;

        if cfg.rename {
            info!(
                "GROBID pre-pass done: {renamed} of {processed} PDF(s) renamed, \
                 {completed} OpenAlex-completed, {failed} failed."
            );
        } else {
            info!("GROBID pre-pass done: {completed} OpenAlex-completed, {failed} failed.");
        }
        Ok(())
    }

    /// Split `pdfs` into the files that need GROBID and the files whose
    /// cached record can be merged directly.
    ///
    /// A changed file always needs processing; an unchanged file with a
    /// cached record is merged from the cache; an unchanged file without one
    /// is processed once to fill the cache (when merging is configured).
    /// Rename-only runs skip unchanged files entirely.
    fn partition_pdfs(
        manifest: &Manifest,
        cfg: &GrobidPrepassConfig,
        folder: &Path,
        pdfs: Vec<PathBuf>,
    ) -> (Vec<PathBuf>, Vec<(PathBuf, Biblio)>) {
        let mut needed: Vec<PathBuf> = Vec::new();
        let mut cached: Vec<(PathBuf, Biblio)> = Vec::new();
        for path in pdfs {
            if !manifest.is_unchanged(folder, &path) {
                needed.push(path);
                continue;
            }
            if cfg.bibtex.is_none() {
                continue;
            }
            match manifest.biblio(folder, &path) {
                Some(biblio) => cached.push((path, biblio.clone())),
                None => needed.push(path),
            }
        }
        (needed, cached)
    }

    /// Merge `records` into the configured BibTeX file, if one is configured.
    fn merge_records(cfg: &GrobidPrepassConfig, records: &[(PathBuf, Biblio)]) -> Result<()> {
        let Some(bibtex) = &cfg.bibtex else {
            return Ok(());
        };
        if records.is_empty() {
            return Ok(());
        }
        let report = collection::merge_file(&bibtex.path, records, bibtex.link)
            .with_context(|| format!("cannot update the BibTeX file {}", bibtex.path.display()))?;
        info!(
            "BibTeX: merged {} into {} ({} duplicate(s) skipped, {} total).",
            entry_count(report.entries.len()),
            bibtex.path.display(),
            report.duplicates,
            report.total
        );
        Ok(())
    }

    /// `1 entry`, `2 entries`, ...
    fn entry_count(count: usize) -> String {
        format!("{count} entr{}", if count == 1 { "y" } else { "ies" })
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

        /// A corpus folder with one fake PDF, plus a manifest that records it
        /// as already processed with `biblio` as its extracted record.
        fn cached_corpus(dir: &Path, name: &str, biblio: &Biblio) -> PathBuf {
            let folder = dir.join(name);
            std::fs::create_dir_all(&folder).expect("create corpus folder");
            let pdf = folder.join("scan_0012.pdf");
            std::fs::write(&pdf, b"%PDF-1.4 fake").expect("write pdf");
            let mut manifest = Manifest::empty(folder.join(MANIFEST_NAME));
            manifest.record_with_biblio(&folder, &pdf, biblio);
            manifest.save().expect("save manifest");
            folder
        }

        /// A pre-pass that must never contact the server: nothing listens on
        /// the URL, so a probe would fail the test.
        fn offline_config(bibtex: Option<BibtexMergeConfig>) -> GrobidPrepassConfig {
            GrobidPrepassConfig {
                url: "http://127.0.0.1:1".to_string(),
                workers: 1,
                rename: false,
                bibtex,
            }
        }

        #[tokio::test]
        async fn cached_records_merge_without_a_server() {
            let dir = tempfile::tempdir().expect("temp dir");
            let record = biblio(Some("Kahle"), Some("2000"), Some("The Barc model"));
            let folder = cached_corpus(dir.path(), "papers", &record);
            let bib = dir.path().join("refs.bib");
            let cfg = offline_config(Some(BibtexMergeConfig {
                path: bib.clone(),
                link: true,
            }));

            prepass(&cfg, &folder).await.expect("prepass");
            let text = std::fs::read_to_string(&bib).expect("read bib");
            assert!(text.starts_with("@misc{Kahle2000,"), "{text}");
            assert!(text.contains("file = {"), "{text}");
            assert!(text.contains("scan_0012.pdf"), "{text}");

            // Idempotent: nothing new, the file is left as it is.
            prepass(&cfg, &folder).await.expect("second prepass");
            assert_eq!(std::fs::read_to_string(&bib).expect("read"), text);
        }

        #[tokio::test]
        async fn deleted_bibtex_is_rebuilt_from_cache() {
            let dir = tempfile::tempdir().expect("temp dir");
            let record = biblio(Some("Kahle"), Some("2000"), Some("The Barc model"));
            let folder = cached_corpus(dir.path(), "papers", &record);
            let bib = dir.path().join("refs.bib");
            let cfg = offline_config(Some(BibtexMergeConfig {
                path: bib.clone(),
                link: true,
            }));

            prepass(&cfg, &folder).await.expect("first prepass");
            let text = std::fs::read_to_string(&bib).expect("read bib");
            std::fs::remove_file(&bib).expect("delete bib");

            // The unchanged PDF is not re-parsed; the record comes from the
            // manifest cache and the file is recreated.
            prepass(&cfg, &folder).await.expect("second prepass");
            assert_eq!(std::fs::read_to_string(&bib).expect("read bib"), text);
        }

        #[tokio::test]
        async fn merge_without_link_omits_file_field() {
            let dir = tempfile::tempdir().expect("temp dir");
            let record = biblio(Some("Kahle"), Some("2000"), Some("The Barc model"));
            let folder = cached_corpus(dir.path(), "papers", &record);
            let bib = dir.path().join("refs.bib");
            let cfg = offline_config(Some(BibtexMergeConfig {
                path: bib.clone(),
                link: false,
            }));

            prepass(&cfg, &folder).await.expect("prepass");
            let text = std::fs::read_to_string(&bib).expect("read bib");
            assert!(text.starts_with("@misc{Kahle2000,"), "{text}");
            assert!(!text.contains("file = "), "{text}");
        }

        #[test]
        fn partition_upgrades_unchanged_files_without_cached_records() {
            let dir = tempfile::tempdir().expect("temp dir");
            let folder = dir.path().join("papers");
            std::fs::create_dir_all(&folder).expect("create folder");
            let pdf = folder.join("a.pdf");
            std::fs::write(&pdf, b"pdf").expect("write pdf");
            let mut manifest = Manifest::empty(folder.join(MANIFEST_NAME));
            // Fingerprint only, as a rename-only run or an older manifest
            // would leave it.
            manifest.record(&folder, &pdf);

            let cfg = offline_config(Some(BibtexMergeConfig {
                path: dir.path().join("refs.bib"),
                link: true,
            }));
            let (needed, cached) = partition_pdfs(&manifest, &cfg, &folder, vec![pdf.clone()]);
            assert_eq!(needed, vec![pdf.clone()], "must be parsed once");
            assert!(cached.is_empty());

            // A rename-only run still skips unchanged files entirely.
            let rename_only = offline_config(None);
            let (needed, cached) =
                partition_pdfs(&manifest, &rename_only, &folder, vec![pdf.clone()]);
            assert!(needed.is_empty());
            assert!(cached.is_empty());

            // With a cached record, the file is merged without parsing.
            manifest.record_with_biblio(
                &folder,
                &pdf,
                &biblio(Some("Kahle"), Some("2000"), Some("The Barc model")),
            );
            let (needed, cached) = partition_pdfs(&manifest, &cfg, &folder, vec![pdf.clone()]);
            assert!(needed.is_empty());
            assert_eq!(cached.len(), 1);
            assert_eq!(cached[0].0, pdf);

            // A changed file needs processing even with a cached record.
            std::fs::write(&pdf, b"pdf changed").expect("rewrite pdf");
            let (needed, cached) = partition_pdfs(&manifest, &cfg, &folder, vec![pdf.clone()]);
            assert_eq!(needed, vec![pdf]);
            assert!(cached.is_empty());
        }

        #[tokio::test]
        async fn corrupt_bibtex_fails_before_grobid() {
            let dir = tempfile::tempdir().expect("temp dir");
            let folder = dir.path().join("papers");
            std::fs::create_dir_all(&folder).expect("create corpus folder");
            // A PDF that needs processing, so the server would be contacted
            // if the pre-pass got that far.
            std::fs::write(folder.join("new.pdf"), b"%PDF-1.4 fake").expect("write pdf");
            let bib = dir.path().join("refs.bib");
            std::fs::write(&bib, "@misc{broken,\n  title = {Unclosed").expect("write bib");
            let cfg = offline_config(Some(BibtexMergeConfig {
                path: bib.clone(),
                link: true,
            }));

            let error = prepass(&cfg, &folder)
                .await
                .expect_err("corrupt bibliography must fail");
            let message = format!("{error:#}");
            assert!(message.contains("cannot read the BibTeX file"), "{message}");
            assert!(message.contains("cannot parse"), "{message}");
            // The corrupt file is left untouched.
            assert_eq!(
                std::fs::read_to_string(&bib).expect("read bib"),
                "@misc{broken,\n  title = {Unclosed"
            );
        }

        // ── Mock GROBID server ────────────────────────────────────────────

        /// The title "Tiny." stays below the OpenAlex search threshold, so
        /// nothing in this test reaches the OpenAlex API.
        const MOCK_TEI: &str = r#"<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader>
        <encodingDesc><appInfo>
            <application version="0.9.1" when="2026-01-01T00:00+0000"/>
        </appInfo></encodingDesc>
        <fileDesc>
            <titleStmt><title level="a" type="main">Tiny</title></titleStmt>
            <publicationStmt><publisher/></publicationStmt>
            <sourceDesc>
                <biblStruct>
                    <analytic>
                        <title level="a" type="main">Tiny.</title>
                        <author><persName>
                            <forename type="first">Jane</forename>
                            <surname>Smith</surname>
                        </persName></author>
                    </analytic>
                    <monogr><imprint><date type="published" when="2020"/></imprint></monogr>
                </biblStruct>
            </sourceDesc>
        </fileDesc>
    </teiHeader>
    <text/>
</TEI>"#;

        /// A minimal mock of the two GROBID endpoints the pre-pass uses.
        async fn spawn_mock_grobid(tei: &'static str) -> std::net::SocketAddr {
            use tokio::io::AsyncWriteExt;

            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind mock server");
            let addr = listener.local_addr().expect("local addr");
            tokio::spawn(async move {
                loop {
                    let Ok((mut stream, _)) = listener.accept().await else {
                        break;
                    };
                    tokio::spawn(async move {
                        let Some(head) = read_request_head(&mut stream).await else {
                            return;
                        };
                        let request_line = head.lines().next().unwrap_or("");
                        let (status, body) = if request_line.contains("/api/isalive") {
                            ("200 OK", "true".to_string())
                        } else if request_line.contains("/api/processHeaderDocument") {
                            ("200 OK", tei.to_string())
                        } else {
                            ("404 Not Found", String::new())
                        };
                        let response = format!(
                            "HTTP/1.1 {status}\r\nContent-Type: text/plain\r\n\
                             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        );
                        let _ = stream.write_all(response.as_bytes()).await;
                    });
                }
            });
            addr
        }

        /// Read a full request (headers and body, by `Content-Length`) and
        /// return the header block. Keeping the whole body in the read path
        /// avoids closing the connection while the client is still sending.
        async fn read_request_head(stream: &mut tokio::net::TcpStream) -> Option<String> {
            use tokio::io::AsyncReadExt;

            let mut buf = Vec::new();
            let mut tmp = [0u8; 4096];
            loop {
                let n = stream.read(&mut tmp).await.ok()?;
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
                let Some(header_end) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
                    continue;
                };
                let header_end = header_end + 4;
                let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
                let content_length = head
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        if name.eq_ignore_ascii_case("content-length") {
                            value.trim().parse::<usize>().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                if buf.len() >= header_end + content_length {
                    return Some(head);
                }
            }
            Some(String::from_utf8_lossy(&buf).into_owned())
        }

        #[tokio::test]
        async fn prepass_extracts_renames_and_merges_against_mock_server() {
            let dir = tempfile::tempdir().expect("temp dir");
            let folder = dir.path().join("papers");
            std::fs::create_dir_all(&folder).expect("create corpus folder");
            let pdf = folder.join("scan_0012.pdf");
            std::fs::write(&pdf, b"%PDF-1.4 fake").expect("write pdf");

            let addr = spawn_mock_grobid(MOCK_TEI).await;
            let bib = dir.path().join("refs.bib");
            let cfg = GrobidPrepassConfig {
                url: format!("http://{addr}"),
                workers: 2,
                rename: true,
                bibtex: Some(BibtexMergeConfig {
                    path: bib.clone(),
                    link: true,
                }),
            };
            prepass(&cfg, &folder).await.expect("prepass");

            // The PDF was renamed from the extracted, completed header.
            let pdfs = files::collect_pdfs(&folder).expect("collect");
            assert_eq!(pdfs.len(), 1, "{pdfs:?}");
            let name = pdfs[0].file_name().unwrap().to_string_lossy().into_owned();
            assert!(name.starts_with("Smith2020 - "), "{name}");
            assert!(name.contains("Tiny"), "{name}");
            assert!(name.ends_with(" - 2020.pdf"), "{name}");

            // The entry points at the final name.
            let text = std::fs::read_to_string(&bib).expect("read bib");
            assert!(text.starts_with("@misc{Smith2020,"), "{text}");
            assert!(
                text.contains(&format!("file = {{{}}}", pdfs[0].display())),
                "{text}"
            );

            // The manifest caches the record under the final name.
            let manifest = Manifest::load(&folder.join(MANIFEST_NAME)).expect("manifest");
            assert!(manifest.biblio(&folder, &pdfs[0]).is_some());
            assert!(manifest.is_unchanged(&folder, &pdfs[0]));

            // A second run is served from the cache and changes nothing.
            prepass(&cfg, &folder).await.expect("second prepass");
            assert_eq!(std::fs::read_to_string(&bib).expect("read bib"), text);
        }
    }
}
