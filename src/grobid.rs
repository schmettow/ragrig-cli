//! Optional GROBID pre-pass for directory corpora.
//!
//! With `--embed-rename` (and the `grobid` cargo feature), every new PDF in a
//! directory corpus is parsed by a GROBID server, its header metadata is
//! completed against OpenAlex, and the file is renamed to `Author_Year_Title`
//! (title truncated to ten words) before it is indexed.  ragrig embeds chunk
//! provenance including the file name, so meaningful names give the chat
//! agent stable, human-readable citations.
//!
//! The pre-pass is resilient by design:
//!
//! * a PDF GROBID cannot parse keeps its name and is still indexed;
//! * an OpenAlex lookup that fails leaves the GROBID header unchanged;
//! * a failed rename keeps the original path.
//!
//! Only an unreachable GROBID server — or a corpus in which every PDF fails —
//! aborts startup, because the user explicitly asked for the pre-pass.
//!
//! Unchanged files are skipped on later runs via a small fingerprint manifest
//! (`.ragrig_grobid.json` in the corpus folder), so the expensive parse and
//! lookup pass is paid once per file, not on every startup.

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
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::{Duration, UNIX_EPOCH};

    use anyhow::{Context, Result, bail};
    use grobid::bibtex;
    use grobid::openalex::Completer;
    use grobid::{Biblio, GrobidClient, PdfInput, ProcessOptions};
    use log::{info, warn};
    use serde::{Deserialize, Serialize};
    use tokio::sync::Semaphore;
    use tokio::task::JoinSet;

    use super::GrobidRenameConfig;

    /// Title words kept in the new file name.
    const TITLE_WORDS: usize = 10;

    /// Sidecar manifest with per-file fingerprints of everything already
    /// processed, so unchanged PDFs are not parsed again.
    const MANIFEST_NAME: &str = ".ragrig_grobid.json";

    /// Number of liveness probes before giving up on the server.
    const PROBE_ATTEMPTS: usize = 3;

    /// Delay between liveness probes.
    const PROBE_DELAY: Duration = Duration::from_secs(1);

    /// A parsed PDF, tracked by its current path while it is renamed.
    struct Processed {
        path: PathBuf,
        biblio: Biblio,
    }

    /// What one worker returns: the PDF path plus its parsed record (or a
    /// per-file error message).  The `bool` marks an OpenAlex completion.
    type WorkerResult = (PathBuf, Result<(Biblio, bool), String>);

    pub(super) async fn rename_pdfs(cfg: &GrobidRenameConfig, folder: &Path) -> Result<()> {
        let mut manifest = Manifest::load(folder);
        let pdfs: Vec<PathBuf> = match collect_pdfs(folder) {
            Ok(pdfs) => pdfs,
            // A missing folder is reported by the sync that follows; the
            // pre-pass itself has nothing to do.
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(err) => {
                return Err(err).with_context(|| format!("scanning {} for PDFs", folder.display()));
            }
        };
        let pdfs: Vec<PathBuf> = pdfs
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

        info!(
            "GROBID pre-pass: {} PDF(s) in {} — parsing headers, completing against \
             OpenAlex, renaming to Author_Year_Title ({} worker(s)).",
            pdfs.len(),
            folder.display(),
            cfg.workers.max(1)
        );

        let client = GrobidClient::new(&cfg.url)
            .with_context(|| format!("invalid GROBID server URL {:?}", cfg.url))?;
        wait_for_server(&client).await?;
        let completer = Completer::new();
        let options = ProcessOptions::default();
        let semaphore = Arc::new(Semaphore::new(cfg.workers.max(1)));

        let mut tasks: JoinSet<WorkerResult> = JoinSet::new();
        for pdf in pdfs {
            let client = client.clone();
            let completer = completer.clone();
            let options = options.clone();
            let semaphore = Arc::clone(&semaphore);
            tasks.spawn(async move {
                // Bound the number of in-flight GROBID/OpenAlex requests.
                let _permit = semaphore
                    .acquire_owned()
                    .await
                    .expect("semaphore not closed");
                let outcome = process_one(&client, &completer, &pdf, &options).await;
                (pdf, outcome)
            });
        }

        let mut results: Vec<Processed> = Vec::new();
        let mut completed = 0usize;
        let mut failed = 0usize;
        while let Some(joined) = tasks.join_next().await {
            let (path, outcome) = joined.expect("GROBID worker task panicked");
            match outcome {
                Ok((biblio, was_completed)) => {
                    if was_completed {
                        completed += 1;
                    }
                    results.push(Processed { path, biblio });
                }
                Err(message) => {
                    failed += 1;
                    warn!("GROBID: skipping {}: {message}", path.display());
                }
            }
        }

        if results.is_empty() {
            bail!(
                "GROBID pre-pass: all {} PDF(s) failed — is the server at {} running?",
                failed,
                cfg.url
            );
        }

        let renamed = rename_all(&mut results, folder, &mut manifest);
        manifest.prune(folder);
        manifest.save(folder);

        info!(
            "GROBID pre-pass done: {} of {} PDF(s) renamed, {} OpenAlex-completed, {} failed.",
            renamed,
            results.len() + failed,
            completed,
            failed
        );
        Ok(())
    }

    /// Parse one PDF's header and complete it against OpenAlex.  A failed
    /// lookup only warns: the GROBID header is kept as parsed.
    async fn process_one(
        client: &GrobidClient,
        completer: &Completer,
        path: &Path,
        options: &ProcessOptions,
    ) -> Result<(Biblio, bool), String> {
        let document = client
            .process_header_document(PdfInput::from(path), options)
            .await
            .map_err(|err| err.to_string())?;
        let mut biblio = document.header.biblio;
        let mut completed = false;
        match completer.complete(&biblio).await {
            Ok(Some(completion)) => {
                biblio = completion.biblio;
                completed = true;
            }
            Ok(None) => {}
            Err(err) => warn!("OpenAlex lookup failed for {}: {err}", path.display()),
        }
        Ok((biblio, completed))
    }

    /// Wait for the GROBID server to come up, with a bounded number of
    /// probes; GROBID can take a while to preload its models.
    async fn wait_for_server(client: &GrobidClient) -> Result<()> {
        for attempt in 1..=PROBE_ATTEMPTS {
            let remaining = PROBE_ATTEMPTS - attempt;
            match client.ping().await {
                Ok(true) => return Ok(()),
                Ok(false) if remaining > 0 => {
                    info!(
                        "GROBID at {} not alive yet; retrying in {}s ({attempt}/{PROBE_ATTEMPTS}).",
                        client.base_url(),
                        PROBE_DELAY.as_secs()
                    );
                }
                Ok(false) => {
                    bail!(
                        "GROBID at {} is up, but reports it is not alive",
                        client.base_url()
                    );
                }
                Err(err) if remaining > 0 => {
                    info!(
                        "GROBID at {} not responding ({err}); retrying in {}s ({attempt}/{PROBE_ATTEMPTS}).",
                        client.base_url(),
                        PROBE_DELAY.as_secs()
                    );
                }
                Err(err) => {
                    bail!(
                        "cannot reach GROBID server at {} after {PROBE_ATTEMPTS} attempts: {err}. \
                         Start a server (e.g. docker run --rm -p 8070:8070 grobid/grobid:0.9.1-crf) \
                         or drop --embed-rename.",
                        client.base_url()
                    );
                }
            }
            tokio::time::sleep(PROBE_DELAY).await;
        }
        unreachable!("the loop returns on its final attempt")
    }

    /// Rename parsed PDFs in deterministic path order (collision suffixes are
    /// stable across runs), record fingerprints of the final paths, and
    /// return the number of files actually renamed.
    fn rename_all(results: &mut [Processed], folder: &Path, manifest: &mut Manifest) -> usize {
        let mut order: Vec<usize> = (0..results.len()).collect();
        order.sort_by(|&a, &b| results[a].path.cmp(&results[b].path));
        let mut renamed = 0usize;
        for index in order {
            let original = results[index].path.clone();
            let final_path = match rename_target(&original, &results[index].biblio) {
                None => {
                    info!(
                        "GROBID: no usable author/year/title for {}; keeping name",
                        original.display()
                    );
                    original
                }
                Some(target) if target == original => original,
                Some(target) => {
                    let target = first_free(target);
                    match fs::rename(&original, &target) {
                        Ok(()) => {
                            info!(
                                "GROBID: renamed {} → {}",
                                original.display(),
                                target.display()
                            );
                            renamed += 1;
                            target
                        }
                        Err(err) => {
                            warn!("GROBID: cannot rename {}: {err}", original.display());
                            original
                        }
                    }
                }
            };
            results[index].path = final_path.clone();
            manifest.record(folder, &final_path);
        }
        renamed
    }

    /// Build the new file name `Author_Year_<first 10 title words>` for a
    /// parsed PDF, keeping the original file extension.  Parts are filtered
    /// to ASCII alphanumerics (as [`bibtex::suggest_key`] does for citation
    /// keys); missing parts are omitted.  Returns `None` when neither author,
    /// year nor a title word is available.
    fn rename_target(path: &Path, biblio: &Biblio) -> Option<PathBuf> {
        let author = biblio
            .authors
            .first()
            .and_then(|author| author.surname.as_deref())
            .map(sanitize)
            .unwrap_or_default();
        let year = bibtex::year(biblio).unwrap_or_default();
        let title = biblio
            .title
            .as_deref()
            .unwrap_or_default()
            .split_whitespace()
            .take(TITLE_WORDS)
            .map(sanitize)
            .filter(|word| !word.is_empty())
            .collect::<Vec<_>>()
            .join("_");
        let stem = [author, year, title]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("_");
        if stem.is_empty() {
            return None;
        }
        let mut target = path.with_file_name(stem);
        if let Some(extension) = path.extension() {
            target.set_extension(extension);
        }
        Some(target)
    }

    /// The first free variant of `target`, extended with a `-2`, `-3`, ...
    /// suffix on collisions.
    fn first_free(target: PathBuf) -> PathBuf {
        let mut candidate = target.clone();
        let mut suffix = 2usize;
        while candidate.exists() {
            candidate = with_suffix(&target, suffix);
            suffix += 1;
        }
        candidate
    }

    /// Insert a `-<suffix>` marker before the file extension.
    fn with_suffix(path: &Path, suffix: usize) -> PathBuf {
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        match path.extension() {
            Some(extension) => {
                path.with_file_name(format!("{stem}-{suffix}.{}", extension.to_string_lossy()))
            }
            None => path.with_file_name(format!("{stem}-{suffix}")),
        }
    }

    /// Filter `text` down to ASCII alphanumerics.
    fn sanitize(text: &str) -> String {
        text.chars().filter(char::is_ascii_alphanumeric).collect()
    }

    /// Recursively collect all PDFs under `dir`, in sorted order.
    fn collect_pdfs(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
            for entry in fs::read_dir(dir)? {
                let path = entry?.path();
                if path.is_dir() {
                    walk(&path, out)?;
                } else if path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
                {
                    out.push(path);
                }
            }
            Ok(())
        }
        let mut out = Vec::new();
        walk(dir, &mut out)?;
        out.sort();
        Ok(out)
    }

    /// Fingerprints of processed files, persisted as `.ragrig_grobid.json`
    /// next to the documents (the same pattern as ragrig's own
    /// `.ragrig_embeddings.json`).
    #[derive(Default, Serialize, Deserialize)]
    struct Manifest {
        /// Path relative to the corpus folder → fingerprint after processing.
        #[serde(default)]
        files: HashMap<String, Fingerprint>,
    }

    #[derive(PartialEq, Eq, Serialize, Deserialize)]
    struct Fingerprint {
        size: u64,
        mtime: u64,
    }

    impl Manifest {
        fn load(folder: &Path) -> Self {
            let path = folder.join(MANIFEST_NAME);
            let Ok(text) = fs::read_to_string(&path) else {
                return Self::default();
            };
            match serde_json::from_str(&text) {
                Ok(manifest) => manifest,
                Err(err) => {
                    warn!("GROBID: ignoring corrupt {}: {err}", path.display());
                    Self::default()
                }
            }
        }

        fn key(folder: &Path, path: &Path) -> String {
            path.strip_prefix(folder)
                .unwrap_or(path)
                .to_string_lossy()
                .into_owned()
        }

        fn is_unchanged(&self, folder: &Path, path: &Path) -> bool {
            let Some(fingerprint) = fingerprint(path) else {
                return false;
            };
            self.files.get(&Self::key(folder, path)) == Some(&fingerprint)
        }

        fn record(&mut self, folder: &Path, path: &Path) {
            if let Some(fingerprint) = fingerprint(path) {
                self.files.insert(Self::key(folder, path), fingerprint);
            }
        }

        /// Drop entries whose file no longer exists (e.g. pre-rename names).
        fn prune(&mut self, folder: &Path) {
            self.files.retain(|key, _| folder.join(key).exists());
        }

        fn save(&self, folder: &Path) {
            let path = folder.join(MANIFEST_NAME);
            match serde_json::to_string_pretty(self) {
                Ok(json) => {
                    if let Err(err) = fs::write(&path, json) {
                        warn!("GROBID: cannot write {}: {err}", path.display());
                    }
                }
                Err(err) => warn!("GROBID: cannot serialize {}: {err}", path.display()),
            }
        }
    }

    fn fingerprint(path: &Path) -> Option<Fingerprint> {
        let metadata = path.metadata().ok()?;
        let mtime = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        Some(Fingerprint {
            size: metadata.len(),
            mtime,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use grobid::Author;

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
        fn target_is_author_year_and_ten_title_words() {
            let biblio = biblio(
                Some("Kahle"),
                Some("2000-03-01"),
                Some("The Barc model for continuous variables with extra words beyond ten"),
            );
            let target = rename_target(Path::new("/papers/orig.pdf"), &biblio).unwrap();
            assert_eq!(
                target.file_name().unwrap(),
                "Kahle_2000_The_Barc_model_for_continuous_variables_with_extra_words_beyond.pdf"
            );
        }

        #[test]
        fn target_omits_missing_parts() {
            let biblio = biblio(None, Some("2019"), Some("A lone title"));
            let target = rename_target(Path::new("x.pdf"), &biblio).unwrap();
            assert_eq!(target.file_name().unwrap(), "2019_A_lone_title.pdf");
            assert!(rename_target(Path::new("x.pdf"), &Biblio::default()).is_none());
        }

        #[test]
        fn collision_gets_suffix() {
            let dir = tempfile::tempdir().unwrap();
            let target = dir.path().join("Smith_2020_Title.pdf");
            fs::write(&target, b"").unwrap();
            let free = first_free(target);
            assert_eq!(free.file_name().unwrap(), "Smith_2020_Title-2.pdf");
        }

        #[test]
        fn rename_all_moves_file_and_records_manifest() {
            let dir = tempfile::tempdir().unwrap();
            let original = dir.path().join("orig.pdf");
            fs::write(&original, b"pdf").unwrap();
            let mut results = vec![Processed {
                path: original.clone(),
                biblio: biblio(Some("Smith"), Some("2020"), Some("Tiny")),
            }];
            let mut manifest = Manifest::default();

            assert_eq!(rename_all(&mut results, dir.path(), &mut manifest), 1);
            let renamed = dir.path().join("Smith_2020_Tiny.pdf");
            assert!(renamed.exists());
            assert!(!original.exists());
            assert_eq!(results[0].path, renamed);
            assert!(manifest.is_unchanged(dir.path(), &renamed));

            // The old path's entry is dropped once the rename is recorded.
            manifest.prune(dir.path());
            assert_eq!(manifest.files.len(), 1);
        }

        #[test]
        fn manifest_skips_only_unchanged_files() {
            let dir = tempfile::tempdir().unwrap();
            let file = dir.path().join("paper.pdf");
            fs::write(&file, b"one").unwrap();
            let mut manifest = Manifest::default();
            assert!(!manifest.is_unchanged(dir.path(), &file));

            manifest.record(dir.path(), &file);
            assert!(manifest.is_unchanged(dir.path(), &file));

            // A different size means the content changed and is rescanned.
            fs::write(&file, b"much longer").unwrap();
            assert!(!manifest.is_unchanged(dir.path(), &file));
        }
    }
}
