//! Chat-side citation resolution: `@key` mentions become PDF attachments.
//!
//! When a session runs with the GROBID pre-pass configured
//! (`--embed-rename` and/or `--bibtex-merge`), a plain query may mention
//! citation keys, e.g. `Compare @Kahle2000 with the rest of the corpus`.
//! For every mention this module:
//!
//! 1. resolves the key against the maintained BibTeX file — exact match
//!    first, then ASCII case-insensitive (a missing or corrupt file just
//!    disables the bibliography view; the file-name fallback still works),
//! 2. locates the paper's PDF: the entry's `file` field first, then a
//!    best-effort scan of the active directory corpora for a keyed file
//!    name (`Key - ...`, the naming of `--embed-rename`), and
//! 3. hands the located files to the caller, which reads them with the
//!    session's parser pipeline and injects them into the prompt as
//!    one-shot attachments.
//!
//! The work is data-first and I/O-light: [`cite_keys`] is pure,
//! [`CiteState::locate`] maps mentions to [`Located`] values (with the
//! bibliography view cached by file size and modification time), and
//! [`load_documents`] turns located files into attachments with the caller
//! supplying the loader — which keeps the module testable without real
//! PDFs.  Every failure is a value, never an error: an unknown key, a paper
//! without a local PDF, or an unparseable PDF is reported and the query
//! proceeds without it.
//!
//! Only files inside the active corpus directories are used.  A hand-edited
//! `file` field must not turn an `@mention` into permission to read
//! arbitrary files into a (possibly remote) model prompt.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::Result;
use grobid_bibtex::collection::{CiteEntry, Collection};
use log::{debug, warn};
use ragrig::AttachedDocument;

/// Scan `text` for citation-key mentions.
///
/// A mention is `@` at the start of the input or after a character that is
/// not alphanumeric (so `user@example.com` never matches), followed by at
/// least one key character.  A key character is an ASCII letter or digit or
/// one of `_ . : + -`; trailing `.`, `:`, `+` and `-` are treated as
/// sentence punctuation and trimmed, and the first character must be a
/// letter or digit.  Mentions are returned deduplicated, in order of first
/// appearance.
pub fn cite_keys(text: &str) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();
    let mut prev: Option<char> = None;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if c == '@' && !prev.is_some_and(char::is_alphanumeric) {
            let start = i + 1;
            let mut end = start;
            while let Some(&(j, next)) = chars.peek() {
                if !is_key_char(next) {
                    break;
                }
                end = j + next.len_utf8();
                chars.next();
            }
            let token = text[start..end].trim_end_matches(['.', ':', '+', '-']);
            let well_formed = token
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphanumeric());
            if well_formed && !keys.iter().any(|key| key == token) {
                keys.push(token.to_string());
            }
        }
        prev = Some(c);
    }
    keys
}

/// Whether `c` may appear in a citation-key mention.
fn is_key_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | ':' | '+' | '-')
}

/// Where one `@key` mention points, before the document is read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Located {
    /// A paper's PDF in an active corpus.
    Found {
        /// The canonical citation key (the bibliography spelling, when the
        /// key was in the bibliography).
        key: String,
        /// The PDF to read.
        path: PathBuf,
        /// Whether the path came from the entry's `file` field (rather than
        /// a corpus file name matching the key).
        linked: bool,
    },
    /// The key is neither in the bibliography nor a file name.
    UnknownKey {
        /// The mentioned key.
        key: String,
    },
    /// The key is in the bibliography, but no PDF was found.
    NoDocument {
        /// The canonical citation key.
        key: String,
        /// Whether the entry records a `file` path at all.
        linked: bool,
    },
    /// The entry's recorded PDF exists but lies outside every active corpus.
    OutsideCorpora {
        /// The canonical citation key.
        key: String,
        /// The recorded, rejected path.
        path: PathBuf,
    },
    /// Several corpus files match a collision key; without a `file` link the
    /// right one cannot be chosen.
    Ambiguous {
        /// The canonical citation key.
        key: String,
        /// The matching files.
        candidates: Vec<PathBuf>,
    },
}

impl Located {
    /// A one-line, user-facing description of the outcome.
    pub fn describe(&self) -> String {
        match self {
            Located::Found {
                key,
                path,
                linked: true,
            } => format!("@{key} \u{2192} {}", path.display()),
            Located::Found {
                key,
                path,
                linked: false,
            } => format!("@{key} \u{2192} {} (matched by file name)", path.display()),
            Located::UnknownKey { key } => {
                format!("@{key} \u{2014} not found in the bibliography or the active corpora")
            }
            Located::NoDocument { key, linked: true } => {
                format!("@{key} \u{2014} the recorded PDF was not found in the active corpora")
            }
            Located::NoDocument { key, linked: false } => format!(
                "@{key} \u{2014} the entry has no file link and no matching file name in the \
                 active corpora"
            ),
            Located::OutsideCorpora { key, path } => format!(
                "@{key} \u{2014} {} is outside the active corpora; skipped",
                path.display()
            ),
            Located::Ambiguous { key, candidates } => {
                let names: Vec<String> = candidates
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect();
                format!(
                    "@{key} \u{2014} {} files match (add a file link with --bibtex-merge \
                     without --bibtex-no-link to disambiguate): {}",
                    names.len(),
                    names.join(", ")
                )
            }
        }
    }
}

/// The final outcome of one mention after loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CiteOutcome {
    /// The paper's text was attached to the prompt.
    Attached {
        /// The canonical citation key.
        key: String,
        /// The PDF the text was read from.
        path: PathBuf,
        /// Bytes of parsed text.
        chars: usize,
    },
    /// The PDF was located but could not be read.
    Unreadable {
        /// The canonical citation key.
        key: String,
        /// The PDF that failed.
        path: PathBuf,
        /// The loader's error.
        error: String,
    },
    /// The mention was not resolved to a document.
    Missed(Located),
}

impl CiteOutcome {
    /// A one-line, user-facing description of the outcome.
    pub fn describe(&self) -> String {
        match self {
            CiteOutcome::Attached { key, path, chars } => format!(
                "@{key} \u{2192} {} ({chars} chars \u{2248} {} tokens)",
                path.display(),
                chars / ATTACH_CHARS_PER_TOKEN
            ),
            CiteOutcome::Unreadable { key, path, error } => {
                format!("@{key} \u{2014} cannot parse {}: {error}", path.display())
            }
            CiteOutcome::Missed(reason) => reason.describe(),
        }
    }
}

/// The characters-per-token estimate ragrig uses for prompt sizes, so the
/// citation size report speaks the same units as the context budget.
const ATTACH_CHARS_PER_TOKEN: usize = 3;

/// Read the located documents and build the attachments.
///
/// `load` is the session's document pipeline (any registered parser, for
/// example `resolve_attachment_text`); it is injected so the mapping from
/// mentions to attachments is testable without real PDFs.  Mentions that
/// point at the same file (aliases) attach it once.
pub fn load_documents(
    located: &[Located],
    load: &dyn Fn(&Path) -> Result<String>,
) -> (Vec<AttachedDocument>, Vec<CiteOutcome>) {
    let mut documents = Vec::new();
    let mut outcomes = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();
    for item in located {
        match item {
            Located::Found { key, path, .. } => {
                if !seen.insert(path.clone()) {
                    debug!(
                        "Citations: skipping @{key}, {} is already attached",
                        path.display()
                    );
                    continue;
                }
                match load(path) {
                    Ok(content) => {
                        outcomes.push(CiteOutcome::Attached {
                            key: key.clone(),
                            path: path.clone(),
                            chars: content.len(),
                        });
                        documents.push(AttachedDocument {
                            name: format!("@{key}"),
                            content,
                        });
                    }
                    Err(error) => outcomes.push(CiteOutcome::Unreadable {
                        key: key.clone(),
                        path: path.clone(),
                        error: format!("{error:#}"),
                    }),
                }
            }
            other => outcomes.push(CiteOutcome::Missed(other.clone())),
        }
    }
    (documents, outcomes)
}

/// File size and modification time; `None` means the file does not exist.
type Fingerprint = Option<(u64, Option<SystemTime>)>;

/// A cached view of the bibliography: canonical keys with their linked PDFs.
#[derive(Debug)]
struct CiteIndex {
    fingerprint: Fingerprint,
    /// Canonical key to entry.
    exact: HashMap<String, CiteEntry>,
    /// Lowercased key to canonical key, for the forgiving lookup.
    folded: HashMap<String, String>,
}

impl CiteIndex {
    /// Index `entries`, remembering the file `fingerprint` they were read at.
    fn from_entries(entries: Vec<CiteEntry>, fingerprint: Fingerprint) -> Self {
        let mut exact = HashMap::with_capacity(entries.len());
        let mut folded = HashMap::with_capacity(entries.len());
        for entry in entries {
            folded.insert(entry.key.to_ascii_lowercase(), entry.key.clone());
            exact.insert(entry.key.clone(), entry);
        }
        Self {
            fingerprint,
            exact,
            folded,
        }
    }

    /// Resolve a key exactly, then ASCII case-insensitively.
    fn get(&self, key: &str) -> Option<&CiteEntry> {
        if let Some(entry) = self.exact.get(key) {
            return Some(entry);
        }
        let canonical = self.folded.get(&key.to_ascii_lowercase())?;
        self.exact.get(canonical)
    }
}

/// The bibliography view: parsed, being parsed, or remembered as broken.
#[derive(Debug)]
enum Index {
    /// A parsed bibliography.
    Ready(CiteIndex),
    /// A file whose read or parse failed at this fingerprint; resolution
    /// falls back to file names and does not warn again until the file
    /// changes.
    Invalid(Fingerprint),
}

/// Session-side citation resolver state.
///
/// The bibliography view loads lazily on the first mention and refreshes
/// when its file size or modification time changes, so merges during a
/// session (the pre-pass, `/bibtex merge`) are picked up without a restart.
#[derive(Debug)]
pub struct CiteState {
    /// Whether the session resolves `@key` mentions (`/bibtex attach`).
    enabled: bool,
    /// The file the cached index was built from.
    source: Option<PathBuf>,
    /// The cached index, if loaded for the current source.
    index: Option<Index>,
}

impl Default for CiteState {
    /// Citation attach is on; the bibliography loads on the first mention.
    fn default() -> Self {
        Self {
            enabled: true,
            source: None,
            index: None,
        }
    }
}

impl CiteState {
    /// Whether `@key` mentions are resolved at all.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Turn mention resolution on or off (`/bibtex attach on|off`).
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Resolve every mention in `keys` to a file.
    ///
    /// `bib` is the configured BibTeX merge target (if any) and
    /// `corpus_dirs` the folders of the active directory corpora.  One
    /// [`Located`] is returned per key, in order; misses explain themselves.
    pub fn locate(
        &mut self,
        keys: &[String],
        bib: Option<&Path>,
        corpus_dirs: &[&Path],
    ) -> Vec<Located> {
        if keys.is_empty() {
            return Vec::new();
        }
        self.refresh(bib);
        let index = match &self.index {
            Some(Index::Ready(index)) => Some(index),
            _ => None,
        };

        // The corpus scan is built once per call, and only when a mention
        // actually needs the file-name fallback.
        let mut scan: Option<Vec<(String, PathBuf)>> = None;
        let mut located = Vec::with_capacity(keys.len());
        for key in keys {
            let entry = index.and_then(|index| index.get(key));
            let canonical = entry.map_or_else(|| key.clone(), |entry| entry.key.clone());
            let linked = entry.and_then(|entry| entry.file.clone());

            // 1. The PDF recorded in the entry's `file` field.
            let mut outside: Option<PathBuf> = None;
            if let Some(recorded) = linked.as_deref() {
                if let Some(path) = resolve_recorded(recorded, corpus_dirs) {
                    located.push(Located::Found {
                        key: canonical,
                        path,
                        linked: true,
                    });
                    continue;
                }
                if recorded.is_file() {
                    outside = Some(recorded.to_path_buf());
                }
            }

            // 2. A corpus file whose name matches the key.
            let files = scan.get_or_insert_with(|| corpus_files(corpus_dirs));
            let mut matches = candidates_for(&canonical, files);
            match matches.len() {
                0 => {
                    let reason = if let Some(path) = outside {
                        Located::OutsideCorpora {
                            key: canonical,
                            path,
                        }
                    } else if linked.is_some() {
                        Located::NoDocument {
                            key: canonical,
                            linked: true,
                        }
                    } else if entry.is_some() {
                        Located::NoDocument {
                            key: canonical,
                            linked: false,
                        }
                    } else {
                        Located::UnknownKey { key: canonical }
                    };
                    located.push(reason);
                }
                1 => {
                    let path = matches.pop().expect("one candidate");
                    located.push(Located::Found {
                        key: canonical,
                        path,
                        linked: false,
                    });
                }
                _ => located.push(Located::Ambiguous {
                    key: canonical,
                    candidates: matches,
                }),
            }
        }
        located
    }

    /// Bring the bibliography view up to date for `bib`.
    fn refresh(&mut self, bib: Option<&Path>) {
        if self.source.as_deref() != bib {
            self.source = bib.map(Path::to_path_buf);
            self.index = None;
        }
        let Some(path) = bib else {
            return;
        };
        let fingerprint = fingerprint(path);
        let up_to_date = match &self.index {
            Some(Index::Ready(index)) => index.fingerprint == fingerprint,
            Some(Index::Invalid(loaded)) => *loaded == fingerprint,
            None => false,
        };
        if up_to_date {
            return;
        }
        match Collection::load_or_new(path) {
            Ok(collection) => {
                let entries: Vec<CiteEntry> = collection.cite_entries().collect();
                debug!(
                    "Citations: indexed {} key(s) from {}",
                    entries.len(),
                    path.display()
                );
                self.index = Some(Index::Ready(CiteIndex::from_entries(entries, fingerprint)));
            }
            Err(error) => {
                warn!(
                    "Citations: ignoring {} until it changes: {error}",
                    path.display()
                );
                self.index = Some(Index::Invalid(fingerprint));
            }
        }
    }
}

/// File size and modification time of `path`; `None` when it does not exist.
fn fingerprint(path: &Path) -> Fingerprint {
    std::fs::metadata(path)
        .ok()
        .map(|meta| (meta.len(), meta.modified().ok()))
}

/// Try the path recorded in a `file` field: as recorded — but only when it
/// stays inside an active corpus — and otherwise under any corpus by file
/// name (the corpus may have moved, or the path may be from another host).
fn resolve_recorded(recorded: &Path, corpus_dirs: &[&Path]) -> Option<PathBuf> {
    if recorded.is_file() && inside_corpora(recorded, corpus_dirs) {
        return Some(recorded.to_path_buf());
    }
    let name = recorded.file_name()?;
    corpus_dirs
        .iter()
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

/// Whether `path` lies inside one of the corpus directories.
///
/// Both sides are canonicalized, so symlinked corpus roots and `..` in the
/// recorded path cannot escape the containment check.
fn inside_corpora(path: &Path, corpus_dirs: &[&Path]) -> bool {
    let Ok(path) = path.canonicalize() else {
        return false;
    };
    corpus_dirs
        .iter()
        .any(|dir| dir.canonicalize().is_ok_and(|dir| path.starts_with(dir)))
}

/// Every PDF in the active corpora as (lowercased file stem, path).
///
/// `collect_pdfs` sorts per corpus, so the result is deterministic.
/// Unreadable directories are skipped with a warning.
fn corpus_files(corpus_dirs: &[&Path]) -> Vec<(String, PathBuf)> {
    let mut files = Vec::new();
    for dir in corpus_dirs {
        match grobid_bibtex::files::collect_pdfs(dir) {
            Ok(paths) => files.extend(paths.into_iter().filter_map(|path| {
                let stem = path.file_stem()?.to_str()?.to_lowercase();
                Some((stem, path))
            })),
            Err(error) => warn!("Citations: cannot scan {}: {error}", dir.display()),
        }
    }
    files
}

/// Corpus files whose name matches `key`.
///
/// A file named exactly `Key.pdf` wins alone.  Otherwise `Key - ...` names
/// match (the naming of `--embed-rename`); and only when no file bears the
/// full key does the base of a collision key such as `Smith2020-2` match
/// `Smith2020 - ...` — the rename numbers the year and not the key, so the
/// files of a collision group keep the base name, and more than one
/// candidate then means the key alone cannot identify the file.
fn candidates_for(key: &str, files: &[(String, PathBuf)]) -> Vec<PathBuf> {
    let lower = key.to_ascii_lowercase();
    let base = collision_base(&lower);
    let mut exact = Vec::new();
    let mut prefixed = Vec::new();
    let mut based = Vec::new();
    for (stem, path) in files {
        if stem == &lower {
            exact.push(path.clone());
        } else if keyed(stem, &lower) {
            prefixed.push(path.clone());
        } else if let Some(base) = base
            && keyed(stem, base)
        {
            based.push(path.clone());
        }
    }
    if !exact.is_empty() {
        exact
    } else if !prefixed.is_empty() {
        prefixed
    } else {
        based
    }
}

/// Whether `stem` is a keyed file name for `base` (`Base - ...`).
fn keyed(stem: &str, base: &str) -> bool {
    stem.strip_prefix(base)
        .is_some_and(|rest| rest.starts_with(" - "))
}

/// The key without a `-2`-style collision suffix.
fn collision_base(key: &str) -> Option<&str> {
    let (base, suffix) = key.rsplit_once('-')?;
    (!base.is_empty() && suffix.chars().all(|c| c.is_ascii_digit())).then_some(base)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn write(path: &Path, content: impl AsRef<[u8]>) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create parent");
        }
        std::fs::write(path, content.as_ref()).expect("write file");
    }

    fn keys(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_string()).collect()
    }

    #[test]
    fn cite_keys_finds_mentions_at_word_boundaries() {
        assert_eq!(
            cite_keys("Compare (@Kano2022) and [@Smith2020], see \"@Jones2019\"."),
            keys(&["Kano2022", "Smith2020", "Jones2019"])
        );
        // Sentence punctuation after a mention is not part of the key.
        assert_eq!(
            cite_keys("@Kano2022, then @Smith2020."),
            keys(&["Kano2022", "Smith2020"])
        );
        // Keys may contain periods, colons, plus and hyphen-minus.
        assert_eq!(
            cite_keys("repeat @doe.2020:x+y-2 now"),
            keys(&["doe.2020:x+y-2"])
        );
    }

    #[test]
    fn cite_keys_ignores_email_addresses_and_stray_at_signs() {
        assert!(cite_keys("mail m.schmettow@example.com about it").is_empty());
        assert!(cite_keys("@ email, @, @: mixed").is_empty());
        assert!(cite_keys("2 @ 3").is_empty());
    }

    #[test]
    fn cite_keys_deduplicates_and_keeps_order() {
        assert_eq!(cite_keys("@B @A more @B @A"), keys(&["B", "A"]));
    }

    #[test]
    fn locate_uses_the_recorded_file_field() {
        let dir = tempfile::tempdir().expect("temp dir");
        let corpus = dir.path().join("papers");
        let pdf = corpus.join("Kano2022 - Kano - A title - 2022.pdf");
        write(&pdf, b"%PDF-1.4 fake");
        let bib = dir.path().join("refs.bib");
        write(
            &bib,
            format!("@misc{{Kano2022,\n  file = {{{}}},\n}}\n", pdf.display()),
        );

        let mut state = CiteState::default();
        let located = state.locate(&keys(&["Kano2022"]), Some(&bib), &[&corpus]);
        assert_eq!(
            located,
            vec![Located::Found {
                key: "Kano2022".to_string(),
                path: pdf,
                linked: true,
            }]
        );
    }

    #[test]
    fn locate_resolves_case_insensitively_to_the_canonical_key() {
        let dir = tempfile::tempdir().expect("temp dir");
        let corpus = dir.path().join("papers");
        let pdf = corpus.join("Kano2022 - Kano - A title - 2022.pdf");
        write(&pdf, b"%PDF-1.4 fake");
        let bib = dir.path().join("refs.bib");
        write(
            &bib,
            format!("@misc{{Kano2022,\n  file = {{{}}},\n}}\n", pdf.display()),
        );

        let mut state = CiteState::default();
        let located = state.locate(&keys(&["kano2022"]), Some(&bib), &[&corpus]);
        assert!(
            matches!(&located[0], Located::Found { key, .. } if key == "Kano2022"),
            "{located:?}"
        );
    }

    #[test]
    fn locate_falls_back_to_keyed_file_names() {
        let dir = tempfile::tempdir().expect("temp dir");
        let corpus = dir.path().join("papers");
        let keyed = corpus.join("Kano2022 - Kano - A title - 2022.pdf");
        let plain = corpus.join("Smith2020.pdf");
        write(&keyed, b"%PDF-1.4 fake");
        write(&plain, b"%PDF-1.4 fake");

        let mut state = CiteState::default();
        let located = state.locate(&keys(&["Kano2022", "Smith2020"]), None, &[&corpus]);
        assert_eq!(
            located,
            vec![
                Located::Found {
                    key: "Kano2022".to_string(),
                    path: keyed,
                    linked: false,
                },
                Located::Found {
                    key: "Smith2020".to_string(),
                    path: plain,
                    linked: false,
                },
            ]
        );
    }

    #[test]
    fn locate_reports_unknown_keys_and_missing_documents() {
        let dir = tempfile::tempdir().expect("temp dir");
        let corpus = dir.path().join("papers");
        std::fs::create_dir_all(&corpus).expect("corpus");
        let bib = dir.path().join("refs.bib");
        write(&bib, "@misc{Kano2022,\n  title = {A title},\n}\n");

        let mut state = CiteState::default();
        let located = state.locate(&keys(&["Kano2022", "Nope2024"]), Some(&bib), &[&corpus]);
        assert_eq!(
            located,
            vec![
                Located::NoDocument {
                    key: "Kano2022".to_string(),
                    linked: false,
                },
                Located::UnknownKey {
                    key: "Nope2024".to_string(),
                },
            ]
        );

        // A recorded path that is gone keeps the `linked` distinction.
        write(
            &bib,
            format!(
                "@misc{{Kano2022,\n  file = {{{}}},\n}}\n",
                corpus.join("gone.pdf").display()
            ),
        );
        let located = state.locate(&keys(&["Kano2022"]), Some(&bib), &[&corpus]);
        assert_eq!(
            located,
            vec![Located::NoDocument {
                key: "Kano2022".to_string(),
                linked: true,
            }]
        );
    }

    #[test]
    fn locate_rejects_paths_outside_the_corpora() {
        let dir = tempfile::tempdir().expect("temp dir");
        let corpus = dir.path().join("papers");
        std::fs::create_dir_all(&corpus).expect("corpus");
        let elsewhere = dir.path().join("private").join("secret.pdf");
        write(&elsewhere, b"%PDF-1.4 fake");
        let bib = dir.path().join("refs.bib");
        write(
            &bib,
            format!(
                "@misc{{Kano2022,\n  file = {{{}}},\n}}\n",
                elsewhere.display()
            ),
        );

        let mut state = CiteState::default();
        let located = state.locate(&keys(&["Kano2022"]), Some(&bib), &[&corpus]);
        assert_eq!(
            located,
            vec![Located::OutsideCorpora {
                key: "Kano2022".to_string(),
                path: elsewhere,
            }]
        );
    }

    #[test]
    fn locate_flags_collision_ambiguity_and_prefers_exact_names() {
        let dir = tempfile::tempdir().expect("temp dir");
        let corpus = dir.path().join("papers");
        let first = corpus.join("Smith2020 - Smith - First - 2020.pdf");
        let second = corpus.join("Smith2020 - Smith - Second - 2020-1.pdf");
        write(&first, b"%PDF-1.4 fake");
        write(&second, b"%PDF-1.4 fake");

        // The rename numbers the year, not the key, so the collision key
        // `Smith2020-2` matches both files.
        let mut state = CiteState::default();
        let located = state.locate(&keys(&["Smith2020-2"]), None, &[&corpus]);
        match &located[0] {
            Located::Ambiguous { key, candidates } => {
                assert_eq!(key, "Smith2020-2");
                assert_eq!(candidates.len(), 2, "{candidates:?}");
            }
            other => panic!("expected ambiguity, got {other:?}"),
        }

        // An exact stem wins over the keyed names.
        let exact = corpus.join("Smith2020-2 - Smith - Second - 2020.pdf");
        write(&exact, b"%PDF-1.4 fake");
        let located = state.locate(&keys(&["Smith2020-2"]), None, &[&corpus]);
        assert_eq!(
            located,
            vec![Located::Found {
                key: "Smith2020-2".to_string(),
                path: exact,
                linked: false,
            }]
        );
    }

    #[test]
    fn locate_refreshes_the_index_when_the_bibliography_changes() {
        let dir = tempfile::tempdir().expect("temp dir");
        let corpus = dir.path().join("papers");
        let alpha = corpus.join("Alpha2020 - Alpha - First - 2020.pdf");
        let beta = corpus.join("Beta2021 - Beta - Second - 2021.pdf");
        write(&alpha, b"%PDF-1.4 fake");
        write(&beta, b"%PDF-1.4 fake");
        let bib = dir.path().join("refs.bib");
        write(
            &bib,
            format!("@misc{{Alpha2020,\n  file = {{{}}},\n}}\n", alpha.display()),
        );

        let mut state = CiteState::default();
        let located = state.locate(&keys(&["Alpha2020", "Beta2021"]), Some(&bib), &[&corpus]);
        assert!(matches!(located[0], Located::Found { linked: true, .. }));
        // Not in the bibliography yet: the keyed file name resolves it.
        assert!(
            matches!(&located[1], Located::Found { key, linked: false, .. } if key == "Beta2021")
        );

        // A merge during the session is picked up: `Beta2021` now comes from
        // the bibliography's file link, not from the scanned file name.
        write(
            &bib,
            format!(
                "@misc{{Alpha2020,\n  file = {{{}}},\n}}\n\
                 @misc{{Beta2021,\n  file = {{{}}},\n}}\n",
                alpha.display(),
                beta.display()
            ),
        );
        let located = state.locate(&keys(&["Beta2021"]), Some(&bib), &[&corpus]);
        assert_eq!(
            located,
            vec![Located::Found {
                key: "Beta2021".to_string(),
                path: beta,
                linked: true,
            }]
        );
    }

    #[test]
    fn locate_ignores_a_corrupt_bibliography_but_still_scans_names() {
        let dir = tempfile::tempdir().expect("temp dir");
        let corpus = dir.path().join("papers");
        let pdf = corpus.join("Kano2022 - Kano - A title - 2022.pdf");
        write(&pdf, b"%PDF-1.4 fake");
        let bib = dir.path().join("refs.bib");
        write(&bib, "@misc{broken,\n  title = {Unclosed");

        let mut state = CiteState::default();
        let located = state.locate(&keys(&["Kano2022"]), Some(&bib), &[&corpus]);
        assert_eq!(
            located,
            vec![Located::Found {
                key: "Kano2022".to_string(),
                path: pdf,
                linked: false,
            }]
        );
    }

    #[test]
    fn load_documents_attaches_reads_reports_and_dedupes() {
        let located = vec![
            Located::Found {
                key: "A2020".to_string(),
                path: PathBuf::from("/papers/a.pdf"),
                linked: true,
            },
            // An alias of the same file is skipped.
            Located::Found {
                key: "A2020-alias".to_string(),
                path: PathBuf::from("/papers/a.pdf"),
                linked: false,
            },
            Located::Found {
                key: "B2020".to_string(),
                path: PathBuf::from("/papers/b.pdf"),
                linked: false,
            },
            Located::UnknownKey {
                key: "C2020".to_string(),
            },
        ];
        let load = |path: &Path| {
            if path.ends_with("a.pdf") {
                Ok("full text of A".to_string())
            } else {
                Err(anyhow::anyhow!("no parser for b.pdf"))
            }
        };

        let (documents, outcomes) = load_documents(&located, &load);
        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].name, "@A2020");
        assert_eq!(documents[0].content, "full text of A");
        assert_eq!(
            outcomes,
            vec![
                CiteOutcome::Attached {
                    key: "A2020".to_string(),
                    path: PathBuf::from("/papers/a.pdf"),
                    chars: "full text of A".len(),
                },
                CiteOutcome::Unreadable {
                    key: "B2020".to_string(),
                    path: PathBuf::from("/papers/b.pdf"),
                    error: "no parser for b.pdf".to_string(),
                },
                CiteOutcome::Missed(Located::UnknownKey {
                    key: "C2020".to_string(),
                }),
            ]
        );
    }

    #[test]
    fn describe_renders_misses_and_attachments() {
        let mut state = CiteState::default();
        assert!(state.is_enabled(), "citation attach defaults to on");
        state.set_enabled(false);
        assert!(!state.is_enabled());

        let located = state.locate(&keys(&["Nope2024"]), None, &[]);
        let (_, outcomes) = load_documents(&located, &|_| Ok(String::new()));
        let missed = outcomes[0].describe();
        assert!(missed.contains("Nope2024"), "{missed}");
        assert!(missed.contains("not found"), "{missed}");

        let attached = CiteOutcome::Attached {
            key: "Kano2022".to_string(),
            path: PathBuf::from("Kano2022.pdf"),
            chars: 900,
        };
        let text = attached.describe();
        assert!(text.contains("@Kano2022"), "{text}");
        assert!(text.contains("300 tokens"), "{text}");
    }
}
