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
    use grobid::bibtex::{self, FileStemOptions, FileStemStyle};
    use grobid::openalex::Completer;
    use grobid::{Biblio, GrobidClient, PdfInput, ProcessOptions};
    use log::{info, warn};
    use serde::{Deserialize, Serialize};
    use tokio::sync::Semaphore;
    use tokio::task::JoinSet;

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
             OpenAlex, renaming to Key - Full authors - Full title - Year ({} worker(s)).",
            pdfs.len(),
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

    /// Rename parsed PDFs in deterministic path order (collision suffixes are
    /// stable across runs), record fingerprints of the final paths, and
    /// return the number of files actually renamed.
    fn rename_all(results: &mut [Processed], folder: &Path, manifest: &mut Manifest) -> usize {
        let mut order: Vec<usize> = (0..results.len()).collect();
        order.sort_by(|&a, &b| results[a].path.cmp(&results[b].path));
        let mut renamed = 0usize;
        for index in order {
            let original = results[index].path.clone();
            let final_path = match bibtex::suggest_file_name_with(
                &original,
                &results[index].biblio,
                &RENAME_OPTIONS,
            ) {
                None => {
                    info!(
                        "GROBID: no usable author/year/title for {}; keeping name",
                        original.display()
                    );
                    original
                }
                Some(target) if target == original => original,
                Some(target) => {
                    // Colliding names get a counter on the year, so authors
                    // and title keep their place in the name.
                    let target = bibtex::unique_path_with_year(
                        target,
                        bibtex::year(&results[index].biblio).as_deref(),
                    );
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
        fn target_is_key_full_authors_title_year() {
            let biblio = biblio(
                Some("Kahle"),
                Some("2000-03-01"),
                Some("The Barc model for continuous variables with extra words beyond ten"),
            );
            let target = bibtex::suggest_file_name_with(
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
            let target =
                bibtex::suggest_file_name_with(Path::new("x.pdf"), &biblio, &RENAME_OPTIONS)
                    .unwrap();
            // Without an author or year there is no key; the title leads.
            assert_eq!(target.file_name().unwrap(), "A lone title - 2019.pdf");
            assert!(
                bibtex::suggest_file_name_with(
                    Path::new("x.pdf"),
                    &Biblio::default(),
                    &RENAME_OPTIONS
                )
                .is_none()
            );
        }

        #[test]
        fn collision_gets_year_suffix() {
            let dir = tempfile::tempdir().unwrap();
            let target = dir.path().join("Smith - 2020 - Title.pdf");
            fs::write(&target, b"").unwrap();
            let free = bibtex::unique_path_with_year(target, Some("2020"));
            assert_eq!(free.file_name().unwrap(), "Smith - 2020-1 - Title.pdf");
        }

        #[test]
        fn rename_all_numbers_the_year_on_collisions() {
            // Two records with the same authors, year and title must not
            // overwrite each other: the second gets `2020-1`.
            let dir = tempfile::tempdir().unwrap();
            let first = dir.path().join("a.pdf");
            let second = dir.path().join("b.pdf");
            fs::write(&first, b"pdf").unwrap();
            fs::write(&second, b"pdf").unwrap();
            let mut results = vec![
                Processed {
                    path: first,
                    biblio: biblio(Some("Smith"), Some("2020"), Some("Same title")),
                },
                Processed {
                    path: second,
                    biblio: biblio(Some("Smith"), Some("2020"), Some("Same title")),
                },
            ];
            let mut manifest = Manifest::default();
            assert_eq!(rename_all(&mut results, dir.path(), &mut manifest), 2);

            let mut names: Vec<String> = fs::read_dir(dir.path())
                .unwrap()
                .map(|entry| entry.unwrap().file_name().into_string().unwrap())
                .collect();
            names.sort();
            assert_eq!(
                names,
                vec![
                    "Smith2020 - Smith - Same title - 2020-1.pdf",
                    "Smith2020 - Smith - Same title - 2020.pdf"
                ]
            );
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
            let renamed = dir.path().join("Smith2020 - Smith - Tiny - 2020.pdf");
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

        /// A minimal synchronous HTTP mock of the two GROBID endpoints the
        /// pre-pass uses, to exercise the whole chain (client, parser,
        /// OpenAlex skip, rename, manifest) without a real server.
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
                    let tei = tei.to_string();
                    tokio::spawn(async move {
                        let Some(head) = read_request_head(&mut stream).await else {
                            return;
                        };
                        let request_line = head.lines().next().unwrap_or("");
                        let (status, body) = if request_line.contains("/api/isalive") {
                            ("200 OK", "true".to_string())
                        } else if request_line.contains("/api/processHeaderDocument") {
                            ("200 OK", tei)
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

        /// The title "Tiny." stays below the OpenAlex search threshold, so
        /// the pre-pass makes no OpenAlex call in this test.  Its trailing
        /// period must be stripped by the rename policy.
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
                        <author><persName>
                            <forename type="first">Ann</forename>
                            <surname>Jones</surname>
                        </persName></author>
                    </analytic>
                    <monogr><imprint><date type="published" when="2020"/></imprint></monogr>
                </biblStruct>
            </sourceDesc>
        </fileDesc>
    </teiHeader>
    <text/>
</TEI>"#;

        #[tokio::test]
        async fn rename_pdfs_end_to_end_against_mock_server() {
            let dir = tempfile::tempdir().unwrap();
            let pdf = dir.path().join("scan.pdf");
            fs::write(&pdf, b"%PDF-1.4 fake").unwrap();
            // A second PDF with the same metadata forces the collision path.
            let colliding = dir.path().join("scan2.pdf");
            fs::write(&colliding, b"%PDF-1.4 fake").unwrap();

            let addr = spawn_mock_grobid(MOCK_TEI).await;
            let cfg = GrobidRenameConfig {
                url: format!("http://{addr}"),
                workers: 2,
            };
            rename_pdfs(&cfg, dir.path()).await.expect("pre-pass");

            let renamed = dir
                .path()
                .join("Smith2020 - Jane Smith, Ann Jones - Tiny - 2020.pdf");
            assert!(renamed.exists(), "expected {renamed:?} to exist");
            // The collision is resolved by numbering the year at the end of
            // the name.
            let numbered = dir
                .path()
                .join("Smith2020 - Jane Smith, Ann Jones - Tiny - 2020-1.pdf");
            assert!(numbered.exists(), "expected {numbered:?} to exist");
            assert!(!pdf.exists(), "the first original file must be renamed");
            assert!(
                !colliding.exists(),
                "the second original file must be renamed"
            );

            // The fingerprint manifest turns the second run into a no-op; it
            // returns before contacting the server again.
            let manifest = Manifest::load(dir.path());
            assert!(manifest.is_unchanged(dir.path(), &renamed));
            assert!(manifest.is_unchanged(dir.path(), &numbered));
            rename_pdfs(&cfg, dir.path())
                .await
                .expect("second pre-pass");
            let pdf_count = fs::read_dir(dir.path())
                .unwrap()
                .filter(|entry| {
                    entry
                        .as_ref()
                        .unwrap()
                        .path()
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
                })
                .count();
            assert_eq!(pdf_count, 2);
        }
    }
}
