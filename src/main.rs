use anyhow::{Context, Result};
use clap::Parser;
use log::{debug, error, info, trace, warn};
use ragrig::types::{
    ChatConfig, ContextSizeMode, EmbedConfig, EmbeddingProvider, MemoryConfig, ParseConfig,
    PdfParserBackend, Provider, RagrigConfig,
};
use ragrig::{
    AgentSession, AttachedDocument, CancellationToken, ChatAgentSpec, ChunkConfig, Corpus,
    DEFAULT_MAX_DOWNLOAD_BYTES, DocumentParser, DocumentParsers, EmbedderSpec, EpubParserBackend,
    FileIndexResult, FolderCorpus, FsSessionStore, GenerationParams, HistoryStrategy,
    HybridRrfRanker, LlmReranker, LogHistory, MmrDiversityRanker, PaperResult, PipelineFilter,
    PrependAttach, ProgressEvent, RagAgent, RagrigError, Ranker, ScoredChunk, SessionId,
    SessionStore, SummaryHistory, UrlCorpus, WeightedFusionRanker, available_chunkers,
    scan_document_files, search_by_document,
};
use ragrig::{parsers, store};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use std::fs;
use std::io::{Write, stdout};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use tracing::Level;
use tracing_appender::rolling;
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::prelude::*;
use tracing_subscriber::{self, EnvFilter, fmt};

mod search;
use search::{search_arxiv, search_semantic_scholar};

// ── CLI parsing (binary-only) ──────────────────────────────────────────────

/// CLI arguments for the chat / generation sub-system.
#[derive(clap::Args, Debug)]
struct CliChatConfig {
    #[arg(long, default_value = "ollama")]
    pub provider: String,
    #[arg(short, long, default_value = "qwen3.5:9b")]
    pub model: String,
    #[arg(long, env = "DEEPSEEK_API_KEY")]
    pub deepseek_api_key: Option<String>,
    #[arg(long, default_value = "deepseek-v4-pro")]
    pub deepseek_model: String,
    #[arg(long)]
    pub temperature: Option<f64>,
    #[arg(long)]
    pub top_p: Option<f64>,
    #[arg(long)]
    pub max_tokens: Option<usize>,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long, default_value = "4096")]
    pub context_tokens: usize,
    #[arg(long, default_value = "auto")]
    pub context_size_mode: String,
    #[arg(long)]
    pub prompt_chat: Option<PathBuf>,
}

/// CLI arguments for the embedding / retrieval sub-system.
#[derive(clap::Args, Debug)]
struct CliEmbedConfig {
    #[arg(long = "embedding-provider", default_value = "ollama")]
    pub embedding_provider: String,
    #[arg(
        short = 'e',
        long = "embedding-model",
        default_value = "nomic-embed-text:latest"
    )]
    pub embedding_model: String,
    #[arg(long, default_value = "50")]
    pub top_k: usize,
    #[arg(long, default_value = "0.04")]
    pub similarity_threshold: f64,
}

/// CLI arguments for the document parsing / chunking sub-system.
#[derive(clap::Args, Debug)]
struct CliParseConfig {
    #[arg(long = "pdf-parser", default_value = "extract")]
    pub pdf_parser: String,
    #[arg(long)]
    pub sloppy_pdf: bool,
    #[arg(long, default_value = "1024")]
    pub chunk_size: usize,
    #[arg(long, default_value = "128")]
    pub chunk_overlap: usize,
}

/// CLI arguments for the memory / query-rewrite sub-system.
#[derive(clap::Args, Debug)]
struct CliMemoryConfig {
    #[arg(long = "memory-model", default_value = "qwen2.5:1.5b")]
    pub memory_model: String,
    #[arg(long)]
    pub prompt_rewrite: Option<PathBuf>,
}

/// CLI arguments — thin `clap` wrapper that converts into the library `RagrigConfig`.
#[derive(Parser, Debug)]
#[command(about = "Pure Rust local RAG — chunkedrs + rig + Ollama/DeepSeek/Fastembed")]
struct Cli {
    /// Workspace directory for the vector store, history, sessions, and
    /// profiles.  Defaults to the current directory.
    #[arg(long = "workspace", value_name = "DIR")]
    pub workspace: Option<PathBuf>,
    /// Shortcut for `--workspace <DIR> --corpus-dir folder=<DIR>`.
    #[arg(short, long, value_name = "DIR", conflicts_with = "workspace")]
    pub folder: Option<PathBuf>,
    /// Load a named profile (JSON) from `.ragrig/profiles/` before applying
    /// CLI overrides.  Use `/profile save <name>` in the REPL to create one.
    #[arg(short = 'p', long)]
    pub profile: Option<String>,
    /// Additional document corpora: a named directory.  Repeatable; format
    /// `NAME=DIR`, e.g. `--corpus-dir papers=/data/papers`.
    #[arg(long = "corpus-dir", value_name = "NAME=DIR")]
    pub corpus_dirs: Vec<String>,
    /// Additional document corpora: a named URL.  Repeatable; format
    /// `NAME=URL`, e.g. `--corpus-urls arxiv=https://arxiv.org/pdf/2410.0.pdf`.
    /// Repeating the same NAME accumulates URLs into one curated list.
    #[arg(long = "corpus-urls", value_name = "NAME=URL")]
    pub corpus_urls: Vec<String>,
    #[arg(long, env = "SEMANTIC_SCHOLAR_API_KEY")]
    pub semantic_scholar_api_key: Option<String>,

    /// Start in demo mode: the small llama3.2:3b chat model with a 4096-token
    /// context (fits an 8 GB GPU), memory off, and the embedded HTML fixture
    /// book as the document corpus, with a prefilled first question.
    #[cfg(feature = "test-fixtures")]
    #[arg(long)]
    pub demo: bool,

    #[command(flatten)]
    pub chat: CliChatConfig,
    #[command(flatten)]
    pub embed: CliEmbedConfig,
    #[command(flatten)]
    pub parse: CliParseConfig,
    #[command(flatten)]
    pub memory: CliMemoryConfig,
}

impl From<CliChatConfig> for ChatConfig {
    fn from(c: CliChatConfig) -> Self {
        ChatConfig {
            provider: match c.provider.as_str() {
                "deepseek" => Provider::Deepseek,
                _ => Provider::Ollama,
            },
            model: c.model,
            deepseek_api_key: c.deepseek_api_key,
            deepseek_model: c.deepseek_model,
            params: GenerationParams {
                temperature: c.temperature,
                top_p: c.top_p,
                max_tokens: c.max_tokens,
                seed: c.seed,
            },
            context_tokens: c.context_tokens,
            context_size_mode: match c.context_size_mode.as_str() {
                "forced" => ContextSizeMode::Forced,
                _ => ContextSizeMode::Auto,
            },
            system_prompt_path: c.prompt_chat,
            request_timeout_secs: None,
        }
    }
}

impl From<CliEmbedConfig> for EmbedConfig {
    fn from(c: CliEmbedConfig) -> Self {
        EmbedConfig {
            provider: match c.embedding_provider.as_str() {
                #[cfg(feature = "internal-embed")]
                "fastembed" => EmbeddingProvider::Fastembed,
                _ => EmbeddingProvider::Ollama,
            },
            model: c.embedding_model,
            top_k: c.top_k,
            similarity_threshold: c.similarity_threshold,
            request_timeout_secs: None,
        }
    }
}

impl From<CliParseConfig> for ParseConfig {
    fn from(c: CliParseConfig) -> Self {
        ParseConfig {
            pdf_parser: match c.pdf_parser.as_str() {
                "unpdf" => PdfParserBackend::Unpdf,
                "sink" => PdfParserBackend::Sink,
                "internal" => PdfParserBackend::Internal,
                "vision" => PdfParserBackend::Vision,
                #[cfg(feature = "kreuzberg")]
                "kreuzberg" => PdfParserBackend::Kreuzberg,
                _ => PdfParserBackend::Extract,
            },
            sloppy_pdf: c.sloppy_pdf,
            chunk_size: c.chunk_size,
            chunk_overlap: c.chunk_overlap,
        }
    }
}

impl From<CliMemoryConfig> for MemoryConfig {
    fn from(c: CliMemoryConfig) -> Self {
        MemoryConfig {
            model: c.memory_model,
            rewrite_prompt_path: c.prompt_rewrite,
        }
    }
}

/// Resolve the workspace and implicit-corpus handling from the CLI args:
///
/// - `--folder X`          → workspace X plus a `folder=X` dir corpus
/// - `--workspace X`       → workspace X, no implicit corpus
/// - neither, no corpora   → workspace `.` plus a `folder=.` dir corpus
/// - explicit corpora      → workspace `.`, no implicit corpus
fn resolve_workspace_and_corpora(
    folder: Option<&PathBuf>,
    workspace: Option<&PathBuf>,
    mut corpus_dirs: Vec<String>,
    corpus_urls: &[String],
) -> (PathBuf, Vec<String>) {
    let explicit_workspace = workspace.is_some();
    let workspace = match (folder, workspace) {
        (Some(f), _) => f.clone(),
        (None, Some(ws)) => ws.clone(),
        (None, None) => PathBuf::from("."),
    };
    if let Some(f) = folder {
        // --folder is sugar for --workspace <dir> --corpus-dir folder=<dir>.
        corpus_dirs.push(format!("folder={}", f.display()));
    } else if !explicit_workspace && corpus_dirs.is_empty() && corpus_urls.is_empty() {
        // Bare invocation: --workspace . --corpus-dir folder=.
        corpus_dirs.push("folder=.".to_string());
    }
    (workspace, corpus_dirs)
}

impl From<Cli> for RagrigConfig {
    fn from(c: Cli) -> Self {
        let (workspace, corpus_dirs) = resolve_workspace_and_corpora(
            c.folder.as_ref(),
            c.workspace.as_ref(),
            c.corpus_dirs,
            &c.corpus_urls,
        );
        RagrigConfig {
            workspace,
            chat: c.chat.into(),
            embed: c.embed.into(),
            parse: c.parse.into(),
            memory: c.memory.into(),
            corpus_dirs,
            corpus_urls: c.corpus_urls,
            semantic_scholar_api_key: c.semantic_scholar_api_key,
        }
    }
}

/// Choose the web-download destination: `dyn` on → first active URL corpus,
/// else first active directory corpus; `dyn` off or no active named corpora
/// → the main folder (legacy behaviour).
fn pick_web_route(dyn_corpora: bool, corpora: &[CorpusEntry]) -> WebRoute {
    if dyn_corpora {
        if let Some(e) = corpora
            .iter()
            .find(|e| e.active && matches!(&e.kind, CorpusKind::Urls(_)))
        {
            return WebRoute::Url {
                name: e.name.clone(),
            };
        }
        if let Some(e) = corpora
            .iter()
            .find(|e| e.active && matches!(&e.kind, CorpusKind::Dir(_)))
        {
            return WebRoute::Dir {
                name: e.name.clone(),
            };
        }
    }
    WebRoute::Folder
}

// ── Session: carries all context between REPL cycles ──────────────────────

/// Concrete kind of a named document corpus.  Kept concrete (rather than
/// `Box<dyn Corpus>` alone) so the REPL can call type-specific
/// methods: `UrlCorpus::add_url` for dynamic routing and
/// `FolderCorpus::folder` for saving downloads into a directory corpus.
#[derive(Debug, Clone)]
enum CorpusKind {
    Dir(FolderCorpus),
    Urls(UrlCorpus),
}

impl CorpusKind {
    /// `"dir"` or `"urls"` — for display.
    fn label(&self) -> &'static str {
        match self {
            CorpusKind::Dir(_) => "dir",
            CorpusKind::Urls(_) => "urls",
        }
    }
}

/// A named document corpus registered from the command line
/// (`--corpus-dir` / `--corpus-urls`), plus its on/off state, toggled with
/// `/corpus <name> on|off`.
#[derive(Debug, Clone)]
struct CorpusEntry {
    name: String,
    kind: CorpusKind,
    active: bool,
}

impl CorpusEntry {
    /// The corpus as a trait object, for ingestion.
    fn as_corpus(&self) -> &dyn Corpus {
        match &self.kind {
            CorpusKind::Dir(folder) => folder,
            CorpusKind::Urls(urls) => urls,
        }
    }
}

/// Where a runtime web download is routed by the `dyn` rules.
#[derive(Debug)]
enum WebRoute {
    /// First active URL corpus: the URL joins its curated list.
    Url { name: String },
    /// First active directory corpus: the file is saved into its directory.
    Dir { name: String },
    /// Main folder (legacy behaviour; also when `dyn` is off).
    Folder,
}

/// Persistent state shared across the REPL loop.
///
/// Holds trait-object agents for every pipeline stage — chat, memory,
/// and embeddings — plus the vector store, parser registry, and
/// conversation log.  Agents are `Box<dyn Trait>` so they can be
/// hot-swapped at runtime via `/chat`, `/memory`, and `/embed`.
///
/// # Construction
///
/// Sessions are built by [`bootstrap`], which takes a [`RagrigConfig`], creates
/// all agents from their spec enums, indexes documents, and opens or
/// creates the vector store.
///
/// ```ignore
/// let config = RagrigConfig::from(Cli::parse());
/// let session = bootstrap(config).await?;
/// // session enters the REPL loop
/// ```
struct Session {
    config: RagrigConfig,
    /// Stateful chat session: owns the agent, the transcript, persistence,
    /// and cross-session history diffusion.  All chat turns go through it.
    session: AgentSession,
    last_results: Vec<ScoredChunk>,
    last_search_results: Vec<PaperResult>,
    rl: DefaultEditor,
    history_path: PathBuf,
    http_client: reqwest::Client,
    /// Document parser registry — dispatches `.parse()` to the right
    /// backend based on file extension.  Built once at startup.
    doc_parsers: DocumentParsers,
    /// Currently active PDF parser backend.
    pdf_parser: PdfParserBackend,
    /// EPUB parser backend (currently only one option).
    epub_parser: EpubParserBackend,
    /// Controls context-overflow behaviour: `Auto` retries with fewer chunks,
    /// `Forced` treats overflow as a fatal error.
    context_size_forced: ContextSizeMode,
    /// Externally attached documents — parsed text injected into the next
    /// RAG query.  Cleared after each query (one-shot by default).
    attached_docs: Vec<AttachedDocument>,
    /// Named document corpora (`--corpus-dir` / `--corpus-urls`), toggled
    /// with `/corpus <name> on|off`.
    corpora: Vec<CorpusEntry>,
    /// Dynamic routing for web downloads (`/corpus dyn on|off`).  When on,
    /// `/download` and `/get` route documents into the first active URL
    /// corpus (else the first active directory corpus) instead of the main
    /// folder.
    dyn_corpora: bool,
    /// Shared log-level string for the interactive (stderr) output.
    /// Read by the `FilterFn` on every log event, written by `/log`.
    /// Valid values: `"off"`, `"error"`, `"warn"`, `"info"`,
    /// `"ragrig=debug,info"`, `"ragrig=trace,info"`.
    log_level: Arc<RwLock<String>>,
}

// ── Command: parsed user input ────────────────────────────────────────────

/// Commands recognized by the REPL.  Plain text without a `/` prefix is
/// treated as a RAG query (`RagQuery`).
enum Command {
    #[allow(dead_code)]
    Attach(String),
    Download(String),
    GetPapers(String),
    Help,
    Scholar(String),
    SearchArxiv(String),
    ExtractRefs(String),
    Chat(String),
    Search(String),
    Embed(String),
    Chunker(String),
    Memory(String),
    Hist(String),
    Parser(String),
    Profile(String),
    Prompt(String),
    Corpus(String),
    Log(String),
    RagQuery(String),
    Unknown(String),
    Exit,
}

// ── Bootstrap: build agents, index documents, enter REPL ───────────────────

/// The PDF backend names compiled into this build — drives the `/parser`
/// usage strings so they never advertise a backend that is not there.
fn pdf_backend_names() -> Vec<&'static str> {
    let names = vec!["unpdf", "sink", "extract", "internal"];
    #[cfg(feature = "kreuzberg")]
    let names = [names.as_slice(), &["kreuzberg"]].concat();
    #[cfg(feature = "vision-pdf")]
    let names = [names.as_slice(), &["vision"]].concat();
    names
}

/// Filter the parser list to include the selected PDF backend as primary,
/// plus a panic-fallback (kreuzberg when available, otherwise sloppy-pdf).
fn filtered_parsers(pdf: &PdfParserBackend, _sloppy_pdf: bool) -> Vec<Box<dyn DocumentParser>> {
    let selected_pdf = match pdf {
        #[cfg(feature = "kreuzberg")]
        PdfParserBackend::Kreuzberg => "kreuzberg",
        PdfParserBackend::Unpdf => "unpdf",
        PdfParserBackend::Sink => "pdfsink",
        PdfParserBackend::Extract => "pdf-extract",
        PdfParserBackend::Internal => "sloppy-pdf",
        #[cfg(feature = "vision-pdf")]
        PdfParserBackend::Vision => "vision-pdf",
        // Vision selected but the parser is not compiled in: fall back to
        // the legacy default instead of leaving PDFs unparseable.
        #[cfg(not(feature = "vision-pdf"))]
        PdfParserBackend::Vision => {
            log::warn!(
                "vision-pdf selected but ragrig was built without the `vision-pdf` feature — falling back to pdf-extract"
            );
            "pdf-extract"
        }
        // New backends added upstream: fall back to the legacy default.
        _ => "pdf-extract",
    };
    let fallback = {
        #[cfg(feature = "kreuzberg")]
        {
            "kreuzberg"
        }
        #[cfg(not(feature = "kreuzberg"))]
        {
            "sloppy-pdf"
        }
    };
    let mut list = parsers::build_parsers();
    list.retain(|p| {
        if p.extensions().contains(&"pdf") {
            // Keep the selected parser plus the fallback (don't duplicate if selected == fallback).
            p.name() == selected_pdf || p.name() == fallback
        } else {
            true
        }
    });
    list
}

/// Parse the `name=value` command-line corpus specs into document corpora.
///
/// - `--corpus-dir name=path`  → a [`FolderCorpus`] named `name`.
/// - `--corpus-urls name=url`  → adds `url` to the [`UrlCorpus`] named `name`;
///   repeating a name builds one curated list.
///
/// Duplicate names are rejected so every corpus keeps a unique provenance
/// identity in the store.  Corpora start **active** and are synced into the
/// store at startup.
fn parse_corpora(
    corpus_dirs: &[String],
    corpus_urls: &[String],
    http_client: &reqwest::Client,
) -> Result<Vec<CorpusEntry>> {
    let mut dirs: Vec<(String, FolderCorpus)> = Vec::new();
    let mut urls: Vec<(String, UrlCorpus)> = Vec::new();
    let mut names: Vec<String> = Vec::new();

    for spec in corpus_dirs {
        let Some((name, dir)) = spec.split_once('=') else {
            anyhow::bail!("Invalid --corpus-dir '{spec}': expected NAME=/path/to/dir");
        };
        let (name, dir) = (name.trim(), dir.trim());
        if name.is_empty() || dir.is_empty() {
            anyhow::bail!("Invalid --corpus-dir '{spec}': expected NAME=/path/to/dir");
        }
        if names.iter().any(|n| n == name) {
            anyhow::bail!("Duplicate corpus name '{name}'");
        }
        names.push(name.to_string());
        dirs.push((name.to_string(), FolderCorpus::named(name, dir)));
    }

    for spec in corpus_urls {
        let Some((name, url)) = spec.split_once('=') else {
            anyhow::bail!("Invalid --corpus-urls '{spec}': expected NAME=URL");
        };
        let (name, url) = (name.trim(), url.trim());
        if name.is_empty() || url.is_empty() {
            anyhow::bail!("Invalid --corpus-urls '{spec}': expected NAME=URL");
        }
        if let Some((_, corpus)) = urls.iter_mut().find(|(n, _)| n == name) {
            // Repeating a name accumulates URLs into one curated list.
            corpus.add_url(url);
            continue;
        }
        if names.iter().any(|n| n == name) {
            anyhow::bail!("Duplicate corpus name '{name}' (already a --corpus-dir corpus)");
        }
        names.push(name.to_string());
        let corpus = UrlCorpus::new(name, http_client.clone())
            .with_max_download_bytes(Some(DEFAULT_MAX_DOWNLOAD_BYTES));
        corpus.add_url(url);
        urls.push((name.to_string(), corpus));
    }

    let mut entries: Vec<CorpusEntry> = dirs
        .into_iter()
        .map(|(name, corpus)| CorpusEntry {
            name,
            kind: CorpusKind::Dir(corpus),
            active: true,
        })
        .collect();
    entries.extend(urls.into_iter().map(|(name, corpus)| CorpusEntry {
        name,
        kind: CorpusKind::Urls(corpus),
        active: true,
    }));
    Ok(entries)
}

/// Linear initialisation of the entire RAG session.
///
/// 1. Builds the chat agent, embedding backend, and memory agent from
///    configuration via their `*Spec` factories.
/// 2. Opens or creates the vector store in the workspace directory.
/// 3. Parses the named document corpora (`--corpus-dir` / `--corpus-urls`,
///    plus the implicit `folder` corpus from `--folder` or the bare default)
///    and syncs each active one into the store.
/// 4. Constructs a [`Session`] carrying all state needed by the REPL.
///
/// This is the only place where the full pipeline is assembled —
/// downstream code just calls `session.execute(cmd).await`.
async fn bootstrap(config: RagrigConfig, log_level: Arc<RwLock<String>>) -> Result<Session> {
    // Build generation params from config.
    let chat_params = config.chat.params.clone();
    debug!(
        "Generation params: temperature={:?} top_p={:?} max_tokens={:?} seed={:?}",
        chat_params.temperature, chat_params.top_p, chat_params.max_tokens, chat_params.seed,
    );

    // Build the initial chat agent from config.
    let initial_spec = match config.chat.provider {
        Provider::Ollama => {
            ChatAgentSpec::ollama(config.chat.model.clone(), chat_params.clone(), None)
        }
        Provider::Deepseek => ChatAgentSpec::deepseek(
            config.chat.deepseek_model.clone(),
            config.chat.deepseek_api_key.clone(),
            chat_params.clone(),
            None,
        ),
        // New providers added upstream: fall back to Ollama.
        _ => ChatAgentSpec::ollama(config.chat.model.clone(), chat_params.clone(), None),
    };
    let chat_agent = initial_spec.build()?;
    info!(
        "Chat: {} ({})  |  chunk_size={}, chunk_overlap={}",
        chat_agent.backend_name(),
        chat_agent.model_name(),
        config.parse.chunk_size,
        config.parse.chunk_overlap
    );

    // Build the initial embedding backend from config.
    let embedder_spec = match config.embed.provider {
        EmbeddingProvider::Ollama => EmbedderSpec::Ollama {
            model: config.embed.model.clone(),
            request_timeout_secs: None,
        },
        #[cfg(feature = "internal-embed")]
        EmbeddingProvider::Fastembed => EmbedderSpec::Fastembed,
        // New embedding backends added upstream: fall back to Ollama.
        _ => EmbedderSpec::Ollama {
            model: config.embed.model.clone(),
            request_timeout_secs: None,
        },
    };
    let embedder = embedder_spec.build()?;
    info!(
        "Embed: {} ({})",
        embedder.backend_name(),
        embedder.model_name()
    );

    // Build the document parser registry (needed before store setup).
    let doc_parsers = DocumentParsers::new(filtered_parsers(
        &config.parse.pdf_parser,
        config.parse.sloppy_pdf,
    ));
    info!(
        "Parsers: {}  |  Active PDF: {:?}  |  Chunker: markdown (default; /chunker to swap)",
        doc_parsers.names().join(", "),
        config.parse.pdf_parser
    );

    // Open or create the vector store in the workspace.  Documents come
    // exclusively from the named corpora; the workspace is state only.
    info!("Workspace: {}", config.workspace.display());
    let store = store::open_store(&config.workspace).await?;
    let chunk_cfg = ChunkConfig::new(config.parse.chunk_size, config.parse.chunk_overlap)?;

    // Build the rewrite (memory) agent.
    let memory_spec = ChatAgentSpec::ollama(config.memory.model.clone(), chat_params.clone(), None);
    let memory_agent = memory_spec.build()?;
    info!(
        "Memory: {} ({})",
        memory_agent.backend_name(),
        memory_agent.model_name()
    );

    // Build the RagAgent.
    let mut agent_builder = RagAgent::builder()
        .chat(chat_agent)
        .embed(embedder)
        .store(store)
        // The agent owns the parser registry and chunk config its ingestion
        // methods use; the session keeps a matching registry for direct
        // parse operations (search-by-document, attachments).
        .parsers(DocumentParsers::new(filtered_parsers(
            &config.parse.pdf_parser,
            config.parse.sloppy_pdf,
        )))
        .chunk_config(chunk_cfg.clone())
        .rewriter(memory_agent)
        .attach_strategy(Box::new(PrependAttach))
        .context_tokens(config.chat.context_tokens)
        .top_k(config.embed.top_k)
        .similarity_threshold(config.embed.similarity_threshold);

    // Optional prompt overrides from CLI.
    if let Some(ref path) = config.chat.system_prompt_path {
        let prompt_text = fs::read_to_string(path)?;
        agent_builder = agent_builder.system_prompt(prompt_text);
    }
    if let Some(ref path) = config.memory.rewrite_prompt_path {
        let rewrite_text = fs::read_to_string(path)?;
        agent_builder = agent_builder.rewrite_prompt(rewrite_text);
    }

    let agent = agent_builder.build()?;

    // ── Named document corpora (--corpus-dir / --corpus-urls) ───────
    let http_client = reqwest::Client::new();
    let corpora = parse_corpora(&config.corpus_dirs, &config.corpus_urls, &http_client)?;
    for entry in &corpora {
        info!(
            "Corpus '{}' ({}) registered.",
            entry.name,
            entry.kind.label()
        );
    }
    for entry in corpora.iter().filter(|e| e.active) {
        info!("Indexing corpus '{}' ({}).", entry.name, entry.kind.label());
        let token = CancellationToken::new();
        let watcher = EscWatcher::spawn(token.clone());
        let state = Arc::new(Mutex::new(EmbedProgress::default()));
        let sink = embed_progress_sink(state);
        match agent
            .sync_corpus_with_progress(entry.as_corpus(), Some(&sink), Some(&token))
            .await
        {
            Ok(_) => {}
            Err(e) if e.downcast_ref::<ragrig::Cancelled>().is_some() => {
                eprint!("\r\x1b[2K");
                eprintln!("Indexing cancelled — exiting.");
                return Err(e);
            }
            Err(e) => return Err(e),
        }
        drop(watcher);
        eprint!("\r\x1b[2K");
    }

    let row_count = agent.store().len();
    if row_count == 0 {
        return Err(anyhow::anyhow!(ragrig::RagrigError::NoDocumentsFound {
            folder: config.workspace.to_string_lossy().into_owned(),
        }));
    }
    info!("Vector store initialized with {} total entries.", row_count);

    let pdf_parser = config.parse.pdf_parser.clone();
    let context_size_forced = config.chat.context_size_mode;

    let mut rl = DefaultEditor::new()?;
    let history_path = config.workspace.join(".ragrig_history");
    if history_path.exists()
        && let Err(e) = rl.load_history(&history_path)
    {
        warn!("Could not load history: {}", e);
    }

    info!("RAG System Online. Commands: /download <url> | /get <nums> | /help | exit");
    info!("Ask questions based on your loaded documents (Arrow-Up for history, Ctrl+C to exit):");

    // ── Stateful chat session (filesystem‑backed store, fresh id) ──
    let sessions_dir = config.workspace.join(".ragrig").join("sessions");
    let session_store: Box<dyn SessionStore> = Box::new(FsSessionStore::new(sessions_dir)?);
    let session = AgentSession::new(agent, session_store);
    info!("Session: {}", session.session_id().0);

    Ok(Session {
        config,
        session,
        last_results: Vec::new(),
        last_search_results: Vec::new(),
        rl,
        history_path,
        http_client,
        attached_docs: Vec::new(),
        corpora,
        dyn_corpora: true,
        doc_parsers,
        pdf_parser,
        epub_parser: EpubParserBackend::Epub,
        context_size_forced,
        log_level,
    })
}

// ── Command dispatch ──────────────────────────────────────────────────────

impl From<&str> for Command {
    fn from(input: &str) -> Self {
        let input = input.trim();

        // Non‑slash input is always a RAG query.
        if !input.starts_with('/') {
            return Command::RagQuery(input.to_string());
        }

        if input == "exit" || input == "quit" || input == "/exit" || input == "/bye" {
            return Command::Exit;
        }
        if input == "/help" {
            return Command::Help;
        }

        // Helper: safe substring after a known prefix.
        let after = |prefix: &str| -> &str {
            if input.len() > prefix.len() + 1 {
                &input[prefix.len()..]
            } else {
                ""
            }
        };

        if input.starts_with("/attach") {
            let arg = after("/attach").trim().to_string();
            return Command::Attach(arg);
        }
        if input.starts_with("/download ") {
            let url = strip_ansi(after("/download ")).trim().to_string();
            return Command::Download(url);
        }
        if input.starts_with("/get ") {
            return Command::GetPapers(after("/get ").trim().to_string());
        }
        if input.starts_with("/scholar ") {
            return Command::Scholar(after("/scholar ").trim().to_string());
        }
        if input.eq_ignore_ascii_case("/scholar") {
            return Command::Scholar(String::new());
        }
        if input.eq_ignore_ascii_case("/search") {
            return Command::Search(String::new());
        }
        if input.starts_with("/search ") {
            return Command::Search(after("/search ").trim().to_string());
        }
        if input.starts_with("/arxiv ") {
            return Command::SearchArxiv(after("/arxiv ").trim().to_string());
        }
        if input.starts_with("/refs") {
            return Command::ExtractRefs(after("/refs").trim().to_string());
        }
        if input.starts_with("/chat") {
            return Command::Chat(after("/chat").trim().to_string());
        }
        if input.starts_with("/embed") {
            return Command::Embed(after("/embed").trim().to_string());
        }
        if input.starts_with("/chunker") {
            return Command::Chunker(after("/chunker").trim().to_string());
        }
        if input.starts_with("/memory") {
            return Command::Memory(after("/memory").trim().to_string());
        }
        if input.starts_with("/hist") {
            return Command::Hist(after("/hist").trim().to_string());
        }
        if input.starts_with("/prompt") {
            return Command::Prompt(after("/prompt").trim().to_string());
        }
        if input.starts_with("/log") {
            return Command::Log(after("/log").trim().to_string());
        }
        if input.starts_with("/parser") {
            return Command::Parser(after("/parser").trim().to_string());
        }
        if input.starts_with("/corpus") {
            return Command::Corpus(after("/corpus").trim().to_string());
        }
        if input.starts_with("/profile") {
            return Command::Profile(after("/profile").trim().to_string());
        }

        // Any other slash‑prefixed input is an unknown command, not a query.
        Command::Unknown(input.to_string())
    }
}

impl Session {
    async fn execute(&mut self, cmd: Command) -> Result<()> {
        match cmd {
            Command::Attach(arg) => self.cmd_attach(&arg).await,
            Command::Download(url) => self.cmd_download(&url).await,
            Command::GetPapers(range) => self.cmd_get_papers(&range).await,
            Command::Help => {
                self.cmd_help();
                Ok(())
            }
            Command::Scholar(q) => self.cmd_search_scholar(&q).await,
            Command::SearchArxiv(q) => self.cmd_search_arxiv(&q).await,
            Command::ExtractRefs(filter) => self.cmd_extract_refs(&filter).await,
            Command::Chat(args_str) => self.cmd_chat(&args_str).await,
            Command::Search(args) => self.cmd_search(&args).await,
            Command::Embed(args_str) => self.cmd_embed(&args_str).await,
            Command::Chunker(args_str) => self.cmd_chunker(&args_str).await,
            Command::Memory(args_str) => self.cmd_memory(&args_str).await,
            Command::Hist(args_str) => self.cmd_hist(&args_str).await,
            Command::Prompt(args_str) => self.cmd_prompt(&args_str).await,
            Command::Log(args_str) => self.cmd_log(&args_str).await,
            Command::Parser(args_str) => self.cmd_parser(&args_str).await,
            Command::Profile(args_str) => self.cmd_profile(&args_str).await,
            Command::Corpus(args_str) => self.cmd_corpus(&args_str).await,
            Command::RagQuery(q) => self.cmd_rag_query(&q).await,
            Command::Unknown(cmd) => {
                println!("Unknown command: '{}'", cmd);
                Ok(())
            }
            Command::Exit => Ok(()),
        }
    }

    // ── /attach <file> ─────────────────────────────────────────────────

    /// Parse a file into text and store it as an attachment for the next
    /// RAG query.  The attachment is one-shot: it is cleared after the
    /// next query (unless re-attached).
    ///
    /// Usage:
    ///   /attach <file>            — attach a PDF, EPUB, DOCX, HTML, or MD file
    ///   /attach                  — show currently attached files
    ///   /attach clear            — clear all attachments
    async fn cmd_attach(&mut self, file_path: &str) -> Result<()> {
        if file_path.is_empty() {
            if self.attached_docs.is_empty() {
                println!("No files attached. Usage: /attach <file>  |  /attach clear");
            } else {
                println!("Attached files:");
                for doc in &self.attached_docs {
                    println!("  {} ({} chars)", doc.name, doc.content.len());
                }
                println!("Use /attach clear to remove all attachments.");
            }
            return Ok(());
        }

        if file_path.eq_ignore_ascii_case("clear") {
            let count = self.attached_docs.len();
            self.attached_docs.clear();
            println!("Cleared {} attached document(s).", count);
            return Ok(());
        }

        let path = std::path::Path::new(file_path);
        if !path.exists() {
            error!("File not found: {}", file_path);
            return Ok(());
        }
        if !path.is_file() {
            error!("Not a regular file: {}", file_path);
            return Ok(());
        }

        // Parse the file using the existing document parsers.
        match ragrig::extract_text(&self.doc_parsers, path) {
            Ok(text) => {
                let name = path
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| file_path.to_string());
                let chars = text.len();
                self.attached_docs.push(AttachedDocument {
                    name: name.clone(),
                    content: text,
                });
                println!(
                    "Attached: {} ({} chars, {} total attached)",
                    name,
                    chars,
                    self.attached_docs.len()
                );
            }
            Err(e) => {
                error!("Failed to parse {}: {}", file_path, e);
            }
        }
        Ok(())
    }

    // ── /download <url> ───────────────────────────────────────────────

    /// Choose the destination for a runtime web download:
    /// `dyn` on → first active URL corpus, else first active directory
    /// corpus; `dyn` off or no active corpora → main folder.
    fn choose_web_route(&self) -> WebRoute {
        pick_web_route(self.dyn_corpora, &self.corpora)
    }

    /// Download and index one web document, routed by the `dyn` rules.
    /// Returns a human-readable summary of where it went.
    async fn ingest_web_url(&mut self, url: &str) -> Result<String> {
        match self.choose_web_route() {
            WebRoute::Url { name } => {
                let idx = self
                    .corpora
                    .iter()
                    .position(|e| e.name == name)
                    .ok_or_else(|| {
                        anyhow::anyhow!("internal: web route target '{name}' missing")
                    })?;
                match &self.corpora[idx].kind {
                    CorpusKind::Urls(corpus) => corpus.add_url(url),
                    CorpusKind::Dir(_) => {
                        anyhow::bail!(
                            "internal: web route chose a URL corpus, got directory '{name}'"
                        )
                    }
                }
                let indexed = self.sync_corpus_entry(idx).await?;
                Ok(format!(
                    "Added '{url}' to URL corpus '{name}' ({indexed} documents indexed)."
                ))
            }
            WebRoute::Dir { name } => {
                let idx = self
                    .corpora
                    .iter()
                    .position(|e| e.name == name)
                    .ok_or_else(|| {
                        anyhow::anyhow!("internal: web route target '{name}' missing")
                    })?;
                let folder = match &self.corpora[idx].kind {
                    CorpusKind::Dir(corpus) => corpus.folder().to_path_buf(),
                    CorpusKind::Urls(_) => {
                        anyhow::bail!(
                            "internal: web route chose a directory corpus, got URLs '{name}'"
                        )
                    }
                };
                let (bytes, filename, _content_type) =
                    ragrig::fetch_url(&self.http_client, url, Some(DEFAULT_MAX_DOWNLOAD_BYTES))
                        .await?;
                let dest = folder.join(&filename);
                std::fs::write(&dest, &bytes).with_context(|| {
                    format!(
                        "failed to save '{}' into dir corpus '{name}'",
                        dest.display()
                    )
                })?;
                let indexed = self.sync_corpus_entry(idx).await?;
                Ok(format!(
                    "Saved '{filename}' into dir corpus '{name}' ({indexed} documents indexed)."
                ))
            }
            WebRoute::Folder => {
                if self.dyn_corpora && !self.corpora.is_empty() {
                    println!("Note: no active named corpora — adding to the main folder.");
                }
                let chunk_cfg = self.session.agent().chunk_config();
                ragrig::download_and_ingest_url_with_chunker(
                    self.session.agent().embedder(),
                    self.session.agent().parsers(),
                    &self.config.workspace,
                    &chunk_cfg,
                    &self.http_client,
                    self.session.agent().store(),
                    url,
                    Some(DEFAULT_MAX_DOWNLOAD_BYTES),
                    self.session.agent().chunker(),
                )
                .await
            }
        }
    }

    async fn cmd_download(&mut self, url: &str) -> Result<()> {
        if url.is_empty() {
            println!("Usage: /download <url>");
            return Ok(());
        }
        // Guard: the pipeline that would index this document must already
        // exist in the store.  Adding to a non-indexed provenance is an error.
        let ext = std::path::Path::new(url.split('?').next().unwrap_or(url))
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("pdf");
        if let Err(e) = self.ensure_pipeline_indexed_for(ext, url).await {
            error!("{e}");
            println!("Error: {e}");
            return Ok(());
        }
        info!("Downloading and ingesting: {} ...", url);
        debug!("URL bytes: {:?}", url.as_bytes());
        match self.ingest_web_url(url).await {
            Ok(summary) => {
                println!("{summary}");
            }
            Err(e) => RagrigError::log_or(&e, "Download failed"),
        }
        Ok(())
    }

    // ── /get <nums> ──────────────────────────────────────────────────

    async fn cmd_get_papers(&mut self, range_str: &str) -> Result<()> {
        if self.last_search_results.is_empty() {
            println!("No search results available. Run /scholar or /arxiv first.");
            return Ok(());
        }
        if range_str.is_empty() {
            println!("Usage: /get 1,2,3-4,8");
            return Ok(());
        }

        let indices = match parse_number_range(range_str) {
            Ok(ids) => ids,
            Err(e) => {
                println!("Invalid range: {}", e);
                return Ok(());
            }
        };

        // Guard: papers are PDFs — the (pdf parser, chunker, embedder)
        // pipeline must already exist in the store before adding documents.
        if let Err(e) = self.ensure_pipeline_indexed_for("pdf", "papers").await {
            error!("{e}");
            println!("Error: {e}");
            return Ok(());
        }

        let mut downloaded = 0;
        let mut failed = 0;
        for idx in &indices {
            if *idx >= self.last_search_results.len() {
                println!(
                    "  Skipping [{}]: out of range (max {})",
                    idx + 1,
                    self.last_search_results.len()
                );
                failed += 1;
                continue;
            }
            let paper = &self.last_search_results[*idx];
            let url = strip_ansi(&paper.best_pdf_url());

            if url.is_empty() {
                println!(
                    "  [{:2}] {} — no download URL available",
                    idx + 1,
                    paper.title
                );
                failed += 1;
                continue;
            }

            print!("  [{:2}] {} ... ", idx + 1, paper.title);
            stdout().flush()?;
            match self.ingest_web_url(&url).await {
                Ok(summary) => {
                    println!("done — {summary}");
                    downloaded += 1;
                }
                Err(e) => {
                    error!("Paper download failed: {}", e);
                    println!("failed: {}", e);
                    failed += 1;
                }
            }
        }

        println!(
            "Download complete: {} added, {} failed, {} skipped.",
            downloaded,
            failed,
            indices.len().saturating_sub(downloaded + failed)
        );

        Ok(())
    }

    // ── /help ────────────────────────────────────────────────────────

    fn cmd_help(&self) {
        println!("/attach <file>  — attach a document for the next query (one-shot, not indexed)");
        println!("/attach         — show currently attached files");
        println!("/attach clear   — clear all attachments");
        println!(
            "/download <url>  — download and ingest a PDF (dyn routing: first active URL corpus, else first active dir corpus)"
        );
        println!("/scholar <q>   — search Semantic Scholar (free API key for higher limits)");
        println!("/arxiv <q>      — search arXiv (no API key needed, no rate limits)");
        println!("/search         — show / adjust search parameters (topk, threshold, rank, by)");
        println!("/get 1,2,3-4    — download papers by number from last search");
        println!(
            "/refs [topic]   — extract references from last query results (optionally filtered by topic)"
        );
        println!(
            "/chat <backend> [model] [api_key] | context <N> — hot-swap chat engine or adjust context window"
        );
        println!(
            "/embed <backend> [model] | purge | index — hot-swap embedding backend; index (re)builds the current pipeline"
        );
        println!(
            "/chunker [name] — show or hot-swap the chunking strategy (warns when the pipeline is not indexed)"
        );
        println!(
            "/memory <backend> [model] [key] | transcript | log | summary | off | purge — hot-swap memory + history diffusion"
        );
        println!("/hist [list | load <id> | delete <id>] — manage saved sessions");
        println!("/prompt chat|rewrite <file> | reset — load custom system prompts");
        println!("/log [off|error|warn|info|debug|trace] — show or change log verbosity");
        #[cfg(feature = "kreuzberg")]
        println!(
            "/parser pdf unpdf|sink|extract|internal|kreuzberg | epub epub — hot-swap parser per format"
        );
        #[cfg(not(feature = "kreuzberg"))]
        println!(
            "/parser pdf unpdf|sink|extract|internal | epub epub — hot-swap parser per format"
        );
        println!("/profile save|show|load|list [name] — manage configuration profiles");
        println!(
            "/corpus <name> on|off — toggle a named document corpus (--corpus-dir / --corpus-urls); /corpus dyn on|off — dynamic web-download routing"
        );
        println!("exit / quit     — end the session");
    }

    // ── /scholar <q> ──────────────────────────────────────────────────

    async fn cmd_search_scholar(&mut self, q: &str) -> Result<()> {
        if q.is_empty() {
            println!("Usage: /scholar <query>");
            return Ok(());
        }
        info!("Searching Semantic Scholar for: {} ...", q);
        match search_semantic_scholar(
            self.config.semantic_scholar_api_key.as_deref(),
            &self.http_client,
            q,
            20,
        )
        .await
        {
            Ok(papers) if papers.is_empty() => {
                println!("No papers found.");
            }
            Ok(papers) => {
                println!("\nResults:");
                for (i, p) in papers.iter().enumerate() {
                    println!(
                        "  [{:2}] {} — {}{}",
                        i + 1,
                        p.title,
                        p.format_authors(),
                        p.format_year()
                    );
                    let url = p.best_pdf_url();
                    if !url.is_empty() {
                        println!("       /download {}", url);
                    }
                }
                self.last_search_results = papers.clone();
                println!("\nUse /download <url> to ingest any paper.");
            }
            Err(e) => error!("Scholar search error: {}", e),
        }
        Ok(())
    }

    // ── /arxiv <q> ───────────────────────────────────────────────────

    async fn cmd_search_arxiv(&mut self, q: &str) -> Result<()> {
        if q.is_empty() {
            println!("Usage: /arxiv <query>");
            return Ok(());
        }
        info!("Searching arXiv for: {} ...", q);
        match search_arxiv(&self.http_client, q, 20).await {
            Ok(papers) if papers.is_empty() => {
                println!("No papers found.");
            }
            Ok(papers) => {
                println!("\nResults (arXiv):");
                for (i, p) in papers.iter().enumerate() {
                    println!(
                        "  [{:2}] {} — {}{}",
                        i + 1,
                        p.title,
                        p.format_authors(),
                        p.format_year()
                    );
                    let url = p.best_pdf_url();
                    if !url.is_empty() {
                        println!("       /download {}", url);
                    }
                }
                self.last_search_results = papers;
                println!("\nUse /download <url> to ingest any paper.");
            }
            Err(e) => error!("arXiv search error: {}", e),
        }
        Ok(())
    }

    // ── /search [topk|threshold] ─────────────────────────────────────

    async fn cmd_search(&mut self, args_str: &str) -> Result<()> {
        let mut parts = args_str.split_whitespace();
        let sub = parts.next().unwrap_or("");

        if sub.is_empty() {
            println!("Vector search parameters:");
            println!(
                "  top-k:     {}  (change: /search topk <N>)",
                self.session.agent().top_k()
            );
            println!(
                "  threshold: {:.3}  (change: /search threshold <F>)",
                self.session.agent().similarity_threshold()
            );
            if let Some(name) = self.session.agent().ranker_name() {
                println!(
                    "  ranker:    {}  (change: /search rank <name> [key value]*)",
                    name
                );
            } else {
                println!("  ranker:    (opaque — store backend handles ranking)");
            }
            println!("  /search by <file> — use a document as the search query");
            return Ok(());
        }

        if sub == "topk" {
            match parts.next().and_then(|s| s.parse::<usize>().ok()) {
                Some(n) if n > 0 => {
                    self.session.agent_mut().set_top_k(n);
                    info!("Top-k set to {}.", n);
                }
                _ => println!(
                    "Usage: /search topk <N>  (current: {})",
                    self.session.agent().top_k()
                ),
            }
            return Ok(());
        }

        if sub == "threshold" {
            match parts.next().and_then(|s| s.parse::<f64>().ok()) {
                Some(f) if f >= 0.0 => {
                    self.session.agent_mut().set_similarity_threshold(f);
                    info!("Similarity threshold set to {:.3}.", f);
                }
                _ => println!(
                    "Usage: /search threshold <F>  (current: {:.3})",
                    self.session.agent().similarity_threshold()
                ),
            }
            return Ok(());
        }

        if sub == "rank" {
            let name = parts.next().unwrap_or("");
            if name.is_empty() {
                if let Some(current) = self.session.agent().ranker_name() {
                    println!("Current ranker: {}", current);
                    println!("Available: RRFFusion, Cosine, BM25, Weighted, MMR, LLM");
                    println!("Usage: /search rank <name> [key value]*");
                    println!("  /search rank RRFFusion [k <F>]");
                    println!("  /search rank Cosine");
                    println!("  /search rank BM25");
                    println!("  /search rank Weighted [alpha <F>]");
                    println!("  /search rank MMR [lambda <F>] [inner <name>]");
                    println!("  /search rank LLM [inner <name>] [provider <p>] [model <m>]");
                } else {
                    println!("This store backend handles ranking internally.");
                }
                return Ok(());
            }

            // Collect remaining key-value pairs.
            let mut params: Vec<(&str, &str)> = Vec::new();
            while let (Some(key), Some(val)) = (parts.next(), parts.next()) {
                params.push((key, val));
            }

            // Helper: construct a default-initialised ranker by name.
            fn build_default_ranker(name: &str) -> Option<Box<dyn Ranker>> {
                match name {
                    "RRFFusion" | "rrffusion" => Some(Box::new(HybridRrfRanker::default())),
                    // Hardcoded alphas are always valid.
                    "Cosine" | "cosine" => Some(Box::new(
                        WeightedFusionRanker::new(1.0).expect("alpha 1.0 is valid"),
                    )),
                    "BM25" | "bm25" => Some(Box::new(
                        WeightedFusionRanker::new(0.0).expect("alpha 0.0 is valid"),
                    )),
                    "Weighted" | "weighted" => Some(Box::new(WeightedFusionRanker::default())),
                    _ => None,
                }
            }

            let ranker: Box<dyn Ranker> = match name {
                "RRFFusion" | "rrffusion" => {
                    let mut k: f64 = 60.0;
                    for (key, val) in &params {
                        if *key == "k" {
                            k = val.parse::<f64>().unwrap_or(60.0);
                        }
                    }
                    match HybridRrfRanker::new(k) {
                        Ok(r) => Box::new(r),
                        Err(e) => {
                            println!("Invalid k: {e}");
                            return Ok(());
                        }
                    }
                }
                "Cosine" | "cosine" => {
                    Box::new(WeightedFusionRanker::new(1.0).expect("alpha 1.0 is valid"))
                }
                "BM25" | "bm25" => {
                    Box::new(WeightedFusionRanker::new(0.0).expect("alpha 0.0 is valid"))
                }
                "Weighted" | "weighted" => {
                    let mut alpha: f64 = 0.5;
                    for (key, val) in &params {
                        if *key == "alpha" {
                            alpha = val.parse::<f64>().unwrap_or(0.5);
                        }
                    }
                    match WeightedFusionRanker::new(alpha) {
                        Ok(r) => Box::new(r),
                        Err(e) => {
                            println!("Invalid alpha: {e}");
                            return Ok(());
                        }
                    }
                }
                "MMR" | "mmr" => {
                    let mut lambda: f64 = 0.5;
                    let mut inner_name: Option<String> = None;
                    for (key, val) in &params {
                        if *key == "lambda" {
                            lambda = val.parse::<f64>().unwrap_or(0.5);
                        } else if *key == "inner" {
                            inner_name = Some(val.to_string());
                        }
                    }
                    let inner = match inner_name.as_deref() {
                        Some(n) => match build_default_ranker(n) {
                            Some(r) => r,
                            None => {
                                println!("Unknown inner ranker: '{}'", n);
                                return Ok(());
                            }
                        },
                        None => {
                            // Default inner: use current ranker name, fall back to RRFFusion.
                            let cur = self.session.agent().ranker_name().unwrap_or_default();
                            build_default_ranker(&cur)
                                .unwrap_or_else(|| Box::new(HybridRrfRanker::default()))
                        }
                    };
                    match MmrDiversityRanker::new(lambda, inner) {
                        Ok(r) => Box::new(r),
                        Err(e) => {
                            println!("Invalid lambda: {e}");
                            return Ok(());
                        }
                    }
                }
                "LLM" | "llm" => {
                    let mut inner_name: Option<String> = None;
                    let mut provider: String = "ollama".into();
                    let mut model: String = self.config.memory.model.clone();
                    let mut api_key: Option<String> = None;
                    for (key, val) in &params {
                        match *key {
                            "inner" => inner_name = Some(val.to_string()),
                            "provider" => provider = val.to_string(),
                            "model" => model = val.to_string(),
                            "key" => api_key = Some(val.to_string()),
                            _ => {}
                        }
                    }
                    let inner = match inner_name.as_deref() {
                        Some(n) => match build_default_ranker(n) {
                            Some(r) => r,
                            None => {
                                println!("Unknown inner ranker: '{}'", n);
                                return Ok(());
                            }
                        },
                        None => Box::new(HybridRrfRanker::default()),
                    };
                    let spec = match ChatAgentSpec::parse(
                        &provider,
                        Some(&model),
                        api_key.as_deref(),
                        None,
                    ) {
                        Ok(s) => s,
                        Err(e) => {
                            println!("Could not build LLM reranker generator: {e}");
                            return Ok(());
                        }
                    };
                    let generator: Box<dyn ragrig::Generator> = match spec.try_into() {
                        Ok(g) => g,
                        Err(e) => {
                            println!("Could not build LLM reranker generator: {e}");
                            return Ok(());
                        }
                    };
                    info!(
                        "LLM reranker using {} ({})",
                        generator.backend_name(),
                        generator.model_name()
                    );
                    Box::new(LlmReranker {
                        generator,
                        inner,
                        prompt_template: String::new(),
                    })
                }
                _ => {
                    println!(
                        "Unknown ranker: '{}'. Available: RRFFusion, Cosine, BM25, Weighted, MMR, LLM",
                        name
                    );
                    return Ok(());
                }
            };

            match self.session.agent().set_ranker(ranker) {
                Ok(()) => {
                    let current = self.session.agent().ranker_name().unwrap_or_default();
                    info!("Ranker set to {}.", current);
                }
                Err(e) => {
                    println!("Could not set ranker: {}", e);
                }
            }
            return Ok(());
        }

        if sub == "by" {
            let file = parts.next().unwrap_or("");
            if file.is_empty() {
                println!("Usage: /search by <file>  — use a document as the search query");
                println!(
                    "  Parses the file, chunks it, and finds similar documents in the database."
                );
                return Ok(());
            }
            return self.cmd_search_by_doc(file).await;
        }

        println!(
            "Unknown subcommand: '{}'. Use /search, /search topk <N>, /search threshold <F>, /search rank <name>, or /search by <file>.",
            sub
        );
        Ok(())
    }

    /// Search the database using a document file as the query.
    /// Parses the file, chunks it, embeds each chunk, and finds similar
    /// documents in the vector store.
    async fn cmd_search_by_doc(&mut self, file_path: &str) -> Result<()> {
        let path = std::path::Path::new(file_path);
        if !path.exists() {
            error!("File not found: {}", file_path);
            return Ok(());
        }
        if !path.is_file() {
            error!("Not a regular file: {}", file_path);
            return Ok(());
        }

        let chunk_cfg = ChunkConfig::new(
            self.config.parse.chunk_size,
            self.config.parse.chunk_overlap,
        )?;

        info!(
            "Search-by-document: '{}' (k={}, threshold={:.3})",
            file_path,
            self.session.agent().top_k(),
            self.session.agent().similarity_threshold()
        );

        match search_by_document(
            self.session.agent().embedder(),
            self.session.agent().store(),
            &self.doc_parsers,
            path,
            &chunk_cfg,
            self.session.agent().top_k(),
            self.session.agent().similarity_threshold(),
        )
        .await
        {
            Ok(results) => {
                if results.is_empty() {
                    println!("No similar documents found.");
                } else {
                    println!(
                        "Found {} similar chunks (from document '{}'):",
                        results.len(),
                        file_path
                    );
                    for (i, sc) in results.iter().enumerate() {
                        println!(
                            "  [{:2}] {:.4}  {} — {:.100}",
                            i + 1,
                            sc.score,
                            sc.chunk.document,
                            sc.chunk.text.trim()
                        );
                    }
                }
                self.last_results = results;
                if !self.last_results.is_empty() {
                    println!("\nUse /refs to extract references from these results.");
                }
            }
            Err(e) => {
                RagrigError::log_or(&e, "Search-by-document failed");
            }
        }
        Ok(())
    }

    // ── /refs [topic] ────────────────────────────────────────────────

    async fn cmd_extract_refs(&mut self, filter: &str) -> Result<()> {
        if self.last_results.is_empty() {
            println!("No previous query results. Ask a question first, then use /refs.");
            return Ok(());
        }

        let filter_hint = if filter.is_empty() {
            String::new()
        } else {
            format!(
                " Focus specifically on references related to: \"{}\".",
                filter
            )
        };

        let mut context = String::new();
        for (i, sc) in self.last_results.iter().take(5).enumerate() {
            context.push_str(&format!(
                "[Document {} | Corpus: {}]\n{}\n\n",
                i + 1,
                sc.chunk.document,
                sc.chunk.text
            ));
        }

        let extract_prompt = format!(
            "Extract all academic paper references (cited works with title, authors, year) from the documents below.{}\n\n\
            Return ONLY a numbered list. For each reference, include:\n\
            - Title of the cited paper\n\
            - Authors (last name of first author + et al. if multiple)\n\
            - Year\n\
            - If an arXiv ID or DOI is visible, include it as a URL.\n\n\
            Documents:\n{}",
            filter_hint, context
        );

        info!("Extracting references...");
        print!("Assistant > ");
        stdout().flush()?;

        let got_response = AtomicBool::new(false);
        match self
            .session
            .agent()
            .chat_agent()
            .generate_stream(&extract_prompt, &|text: String| {
                print!("{}", text);
                let _ = stdout().flush();
                got_response.store(true, Ordering::Relaxed);
            })
            .await
        {
            Ok(()) => {}
            Err(e) => RagrigError::log_or(&e, "Reference extraction failed"),
        }
        if !got_response.load(Ordering::Relaxed) {
            println!("(no references found)");
        }
        println!();

        Ok(())
    }

    // ── /chat <backend> [model] [api_key] ─────────────────────────────

    /// Hot-swap the chat agent or adjust the context budget.
    ///
    /// # Agent swap
    ///
    /// Builds a new `Box<dyn Generator>` from a `ChatAgentSpec` and
    /// replaces the session's chat agent without touching the vector
    /// store, conversation memory, or document index.
    ///
    /// ```text
    /// /chat ollama qwen3.5:9b           # switch to local model
    /// /chat deepseek deepseek-chat sk-…  # switch to cloud
    /// ```
    ///
    /// # Context budget
    ///
    /// `context <N>` adjusts the prompt-truncation budget in tokens
    /// without changing the chat engine.
    ///
    /// ```text
    /// /chat context 4096                 # shrink for 4K-window models
    /// /chat context 131072               # expand for cloud models
    /// ```
    async fn cmd_chat(&mut self, args_str: &str) -> Result<()> {
        let mut parts = args_str.split_whitespace();
        let backend = parts.next().unwrap_or("");
        if backend == "context" {
            match parts.next().and_then(|s| s.parse::<usize>().ok()) {
                Some(n) if n > 0 => {
                    self.session.agent_mut().set_context_tokens(n);
                    let ctx = self.session.agent().context_tokens();
                    info!(
                        "Context window set to {} tokens (prompt budget ~{} chars).",
                        ctx,
                        (ctx.saturating_sub(1024)).saturating_mul(3)
                    );
                }
                _ => println!(
                    "Usage: /chat context <tokens>  (current: {})",
                    self.session.agent().context_tokens()
                ),
            }
            return Ok(());
        }
        if backend == "temperature" {
            match parts.next().and_then(|s| s.parse::<f64>().ok()) {
                Some(t) if t >= 0.0 => {
                    self.rebuild_chat_with_param(|p| p.temperature = Some(t));
                }
                _ => println!("Usage: /chat temperature <F>  (0.0 = deterministic)"),
            }
            return Ok(());
        }
        if backend == "top_p" {
            match parts.next().and_then(|s| s.parse::<f64>().ok()) {
                Some(p) if (0.0..=1.0).contains(&p) => {
                    self.rebuild_chat_with_param(|gp| gp.top_p = Some(p));
                }
                _ => println!("Usage: /chat top_p <F>  (0.0–1.0)"),
            }
            return Ok(());
        }
        if backend == "max_tokens" {
            match parts.next().and_then(|s| s.parse::<usize>().ok()) {
                Some(n) if n > 0 => {
                    self.rebuild_chat_with_param(|p| p.max_tokens = Some(n));
                }
                _ => println!("Usage: /chat max_tokens <N>"),
            }
            return Ok(());
        }
        if backend == "seed" {
            match parts.next().and_then(|s| s.parse::<u64>().ok()) {
                Some(s) => {
                    self.rebuild_chat_with_param(|p| p.seed = Some(s));
                }
                _ => println!("Usage: /chat seed <N>"),
            }
            return Ok(());
        }
        if backend.is_empty() {
            println!(
                "Chat: {} ({}) — context window: {} tokens",
                self.session.agent().chat_agent().backend_name(),
                self.session.agent().chat_agent().model_name(),
                self.session.agent().context_tokens(),
            );
            println!(
                "Usage: /chat <backend> [model] [api_key]  |  context <N>  |  temperature <F>  |  top_p <F>  |  max_tokens <N>  |  seed <N>"
            );
            println!("  backends: ollama, deepseek");
            return Ok(());
        }

        let model = parts.next();
        let api_key = parts.next();

        let spec = match ChatAgentSpec::parse(backend, model, api_key, None) {
            Ok(s) => s,
            Err(e) => {
                error!("Chat agent spec parse error: {}", e);
                return Ok(());
            }
        };

        match spec.build() {
            Ok(new_agent) => {
                let old_backend = self.session.agent().chat_agent().backend_name();
                let old_model = self.session.agent().chat_agent().model_name().to_string();
                self.session.agent_mut().set_chat_agent(new_agent);
                info!(
                    "Chat agent swapped: {} ({}) → {} ({})",
                    old_backend,
                    old_model,
                    self.session.agent().chat_agent().backend_name(),
                    self.session.agent().chat_agent().model_name()
                );
            }
            Err(e) => RagrigError::log_or(&e, "Failed to build chat agent"),
        }
        Ok(())
    }

    // ── /embed <backend> [model] ──────────────────────────────────────
    /// Rebuild the current chat agent with a modified `GenerationParams`, keeping
    /// the same backend and model.
    fn rebuild_chat_with_param(&mut self, f: impl FnOnce(&mut GenerationParams)) {
        let current = self.session.agent().chat_agent();
        let backend = current.backend_name();
        let model = current.model_name();

        // We need to reconstruct the spec with updated params.
        // Since we can't introspect the existing generator's params,
        // we start fresh and apply the mutation.
        let mut params = GenerationParams::default();
        f(&mut params);

        let spec = match ChatAgentSpec::parse(backend, Some(model), None, Some(params)) {
            Ok(spec) => spec,
            Err(e) => {
                error!("Cannot rebuild unknown backend: {}", e);
                return;
            }
        };

        match spec.build() {
            Ok(new_agent) => {
                self.session.agent_mut().set_chat_agent(new_agent);
                let agent = self.session.agent().chat_agent();
                info!(
                    "Chat params updated: {} ({})",
                    agent.backend_name(),
                    agent.model_name()
                );
            }
            Err(e) => RagrigError::log_or(&e, "Failed to rebuild chat agent"),
        }
    }

    // ── /embed <backend> [model] ──────────────────────────────────────

    async fn cmd_embed(&mut self, args_str: &str) -> Result<()> {
        let mut parts = args_str.split_whitespace();
        let backend = parts.next().unwrap_or("");
        if backend.is_empty() {
            println!(
                "Embed: {} ({}) — top‑k: {}, threshold: {}",
                self.session.agent().embedder().backend_name(),
                self.session.agent().embedder().model_name(),
                self.session.agent().top_k(),
                self.session.agent().similarity_threshold(),
            );
            println!(
                "Usage: /embed <backend> [model]  |  purge  |  index  |  topk <N>  |  threshold <F>"
            );
            println!(
                "  index — (re)build the index for the current parser+chunker+embedder pipeline"
            );
            println!(
                "  backends: {}",
                EmbedderSpec::available_backends().join(", ")
            );
            return Ok(());
        }

        if backend.eq_ignore_ascii_case("purge") {
            let store = self.session.agent().store();
            let count = store.len();
            let docs: Vec<_> = store.document_ids().into_iter().collect();
            for doc in &docs {
                store.delete_by_document(&doc.0).await?;
            }
            info!(
                "Vector store purged ({} chunks across {} documents).",
                count,
                docs.len()
            );
            return Ok(());
        }

        if backend.eq_ignore_ascii_case("index") {
            info!("Re-indexing all active document corpora...");

            let token = CancellationToken::new();
            let watcher = EscWatcher::spawn(token.clone());
            let state = Arc::new(Mutex::new(EmbedProgress::default()));
            let sink = embed_progress_sink(state);

            // Full re-ingest of every active corpus, stats aggregated.
            let mut all_stats: Vec<FileIndexResult> = Vec::new();
            for entry in self.corpora.iter().filter(|e| e.active) {
                info!(
                    "Re-indexing corpus '{}' ({}).",
                    entry.name,
                    entry.kind.label()
                );
                match self
                    .session
                    .agent()
                    .reindex_corpus_with_progress(entry.as_corpus(), Some(&sink), Some(&token))
                    .await
                {
                    Ok(stats) => all_stats.extend(stats),
                    Err(e) if e.downcast_ref::<ragrig::Cancelled>().is_some() => {
                        eprint!("\r\x1b[2K");
                        info!("Indexing cancelled.");
                        return Ok(());
                    }
                    Err(e) => return Err(e),
                }
            }
            drop(watcher);
            eprint!("\r\x1b[2K");
            info!(
                "Re-indexing complete. Store size: {} chunks.",
                self.session.agent().store().len()
            );
            // Print the aggregated per-file result table.
            let stats = &all_stats;
            let ok_count = stats.iter().filter(|s| s.ok).count();
            let fail_count = stats.len() - ok_count;
            let total_chunks: usize = stats.iter().map(|s| s.chunks).sum();
            let total_chars: usize = stats.iter().map(|s| s.chars).sum();
            let total_kb: u64 = stats.iter().map(|s| s.file_size_kb).sum();
            println!(
                "\n{} files processed ({} ok, {} failed), {} chunks, {} chars, {} KB total.\n",
                stats.len(),
                ok_count,
                fail_count,
                total_chunks,
                total_chars,
                total_kb
            );
            if !stats.is_empty() {
                println!(
                    "{:<44} {:>6} {:>7} {:>8} {:>6}",
                    "File", "KB", "Chunks", "Chars", "Avg/Ch"
                );
                println!("{}", "─".repeat(78));
                for s in stats {
                    let name = if s.file_name.0.len() > 42 {
                        format!("{}…", &s.file_name.0[..41])
                    } else {
                        s.file_name.0.clone()
                    };
                    if s.ok {
                        println!(
                            "{:<44} {:>6} {:>7} {:>8} {:>6.0}",
                            name,
                            s.file_size_kb,
                            s.chunks,
                            s.chars,
                            s.avg_chars_per_chunk()
                        );
                    } else {
                        println!(
                            "{:<44} {:>6} {:>7} {:>8} {:>6}  FAIL",
                            name, s.file_size_kb, "—", "—", "—"
                        );
                    }
                }
            }
            return Ok(());
        }

        if backend == "topk" {
            match parts.next().and_then(|s| s.parse::<usize>().ok()) {
                Some(n) if n > 0 => {
                    self.session.agent_mut().set_top_k(n);
                    info!("Top-k set to {}.", n);
                }
                _ => println!(
                    "Usage: /embed topk <N>  (current: {})",
                    self.session.agent().top_k()
                ),
            }
            return Ok(());
        }

        if backend == "threshold" {
            match parts.next().and_then(|s| s.parse::<f64>().ok()) {
                Some(f) if f >= 0.0 => {
                    self.session.agent_mut().set_similarity_threshold(f);
                    info!("Similarity threshold set to {:.3}.", f);
                }
                _ => println!(
                    "Usage: /embed threshold <F>  (current: {:.3})",
                    self.session.agent().similarity_threshold()
                ),
            }
            return Ok(());
        }

        let model = parts.next();

        let spec = match EmbedderSpec::parse(backend, model) {
            Ok(s) => s,
            Err(e) => {
                error!("Embed spec parse error: {}", e);
                return Ok(());
            }
        };

        match spec.build() {
            Ok(new_embedder) => {
                let old_backend = self.session.agent().embedder().backend_name();
                let old_model = self.session.agent().embedder().model_name().to_string();
                self.session.agent_mut().set_embedder(new_embedder);
                info!(
                    "Embedder swapped: {} ({}) → {} ({})",
                    old_backend,
                    old_model,
                    self.session.agent().embedder().backend_name(),
                    self.session.agent().embedder().model_name()
                );
                // Provenance check: warn when the new embedder has no chunks
                // in the store yet — the user must run /embed index explicitly.
                self.pipeline_indexed().await;
            }
            Err(e) => RagrigError::log_or(&e, "Failed to build embedder"),
        }
        Ok(())
    }

    // ── /chunker [name] ──────────────────────────────────────────────

    /// Show or hot-swap the chunking strategy.
    ///
    /// Changing the chunker does **not** re-embed automatically.  If the
    /// resulting (parser, chunker, embedder) pipeline has no chunks in the
    /// store yet, a warning is printed and the user must run `/embed index`
    /// to build it.
    async fn cmd_chunker(&mut self, args_str: &str) -> Result<()> {
        let available = available_chunkers();
        let name = args_str.trim();
        if name.is_empty() {
            println!("Chunker: {}", self.session.agent().chunker().name());
            println!(
                "Available: {}",
                available
                    .iter()
                    .map(|c| c.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            println!("Usage: /chunker <name>");
            return Ok(());
        }

        let Some(new_chunker) = available
            .into_iter()
            .find(|c| c.name().eq_ignore_ascii_case(name))
        else {
            let names = available_chunkers()
                .iter()
                .map(|c| c.name())
                .collect::<Vec<_>>()
                .join(", ");
            println!("Unknown chunker: '{}'. Available: {}", name, names);
            return Ok(());
        };

        let old = self.session.agent().chunker().name();
        self.session.agent_mut().set_chunker(new_chunker);
        info!(
            "Chunker: {} → {}",
            old,
            self.session.agent().chunker().name()
        );
        // Provenance check: warn when the new pipeline is not indexed yet.
        self.pipeline_indexed().await;
        Ok(())
    }

    // ── Pipeline provenance checks ──────────────────────────────────────

    /// Check whether the store already holds chunks for every
    /// (parser, chunker, embedder) pipeline that indexing the current
    /// document folder would produce.
    ///
    /// Returns `true` when every file format present in the folder has
    /// chunks with the current provenance.  Prints a warning (but never
    /// embeds) when some or all pipelines are missing.
    async fn pipeline_indexed(&self) -> bool {
        // Collect the distinct extensions present in the active directory
        // corpora.
        let mut formats: Vec<String> = Vec::new();
        for entry in self.corpora.iter().filter(|e| e.active) {
            if let CorpusKind::Dir(folder) = &entry.kind {
                for (doc, _name) in scan_document_files(folder.folder()) {
                    if let Some(ext) = doc.path().extension().and_then(|e| e.to_str())
                        && !formats.iter().any(|f| f == ext)
                    {
                        formats.push(ext.to_string());
                    }
                }
            }
        }

        let embedder_id = self.session.agent().embedder().metadata().id();
        let chunker = self.session.agent().chunker().name();
        let mut missing: Vec<String> = Vec::new();
        for ext in &formats {
            let parser = self.doc_parsers.primary_name_for(ext).unwrap_or_default();
            let filter = PipelineFilter {
                corpus: None,
                parser: Some(parser.to_string()),
                chunker: Some(chunker.to_string()),
                embedder: Some(embedder_id.clone()),
                pipeline: None,
            };
            if self.session.agent().store().count_matching(&filter).await == 0 {
                missing.push(format!("{ext} ({parser})"));
            }
        }

        if missing.is_empty() {
            return true;
        }
        let msg = format!(
            "Warning: no chunks indexed for {} with chunker={}, embedder={}. \
             Run '/embed index' to build the index.",
            missing.join(", "),
            chunker,
            embedder_id
        );
        warn!("{msg}");
        println!("{msg}");
        false
    }

    /// Guard for adding documents: the pipeline that would index the new
    /// document must already exist in the store.
    ///
    /// Returns an error when no chunk with the matching
    /// (parser, chunker, embedder) provenance exists, telling the user to
    /// run `/embed index` first.
    async fn ensure_pipeline_indexed_for(&self, ext: &str, what: &str) -> Result<()> {
        let Some(parser) = self.doc_parsers.primary_name_for(ext) else {
            anyhow::bail!("No parser registered for .{ext} files");
        };
        let embedder_id = self.session.agent().embedder().metadata().id();
        let chunker = self.session.agent().chunker().name();
        let filter = PipelineFilter {
            corpus: None,
            parser: Some(parser.to_string()),
            chunker: Some(chunker.to_string()),
            embedder: Some(embedder_id.clone()),
            pipeline: None,
        };
        if self.session.agent().store().count_matching(&filter).await > 0 {
            return Ok(());
        }
        anyhow::bail!(
            "Cannot add {what}: no index exists for the current pipeline \
             (parser={parser}, chunker={chunker}, embedder={embedder_id}). \
             Run '/embed index' first."
        )
    }

    // ── /hist [list | load <id> | delete <id>] ─────────────────────

    async fn cmd_hist(&mut self, args_str: &str) -> Result<()> {
        let arg = args_str.trim();
        if arg.is_empty() || arg == "list" {
            match self.session.list_sessions().await {
                Ok(manifests) if manifests.is_empty() => {
                    println!("No saved sessions.");
                }
                Ok(manifests) => {
                    println!("{} saved session(s):", manifests.len());
                    for m in &manifests {
                        println!("  {} — {} turns — {:?}", m.id.0, m.turn_count, m.created);
                    }
                }
                Err(e) => error!("Error listing sessions: {}", e),
            }
            return Ok(());
        }
        let mut parts = arg.split_whitespace();
        let sub = parts.next().unwrap_or("");
        let id = parts.next().unwrap_or("");
        match sub {
            "load" if !id.is_empty() => {
                let sid = SessionId(id.to_string());
                match self.session.load_session(&sid).await {
                    Ok(true) => {
                        info!(
                            "Loaded session {} ({} turns).",
                            id,
                            self.session.turns().len()
                        );
                    }
                    Ok(false) => println!("Session '{}' not found.", id),
                    Err(e) => error!("Error loading session: {}", e),
                }
            }
            "delete" if !id.is_empty() => {
                let sid = SessionId(id.to_string());
                match self.session.delete_session(&sid).await {
                    Ok(()) => info!("Deleted session '{}'.", id),
                    Err(e) => error!("Error deleting session: {}", e),
                }
            }
            _ => {
                println!("Usage: /hist [list | load <id> | delete <id>]");
            }
        }
        Ok(())
    }

    // ── /memory <backend> [model] [api_key] | off ───────────────────

    async fn cmd_memory(&mut self, args_str: &str) -> Result<()> {
        let arg = args_str.trim();
        if arg.is_empty() {
            // ── Current config ──────────────────────────────────────
            let mem = if self.session.agent().rewriter().is_some() {
                "rewrite"
            } else {
                "off"
            };
            let diff = match self.session.history_strategy() {
                Some(s) => s.name(),
                None => "off",
            };
            println!(
                "Memory: {} — {} turns  |  history diffusion: {}",
                mem,
                self.session.turns().len(),
                diff,
            );
            // ── Usage ──────────────────────────────────────────────
            println!(
                "Usage: /memory <backend> [model] [api_key]  |  transcript  |  log  |  summary  |  off  |  purge"
            );
            println!("  backends: ollama, deepseek");
            println!("  modes:    transcript — raw memory, no query rewriting");
            println!("            log       — enable history diffusion (raw last session)");
            println!("            summary   — enable history diffusion (LLM summarisation)");
            return Ok(());
        }

        if arg.eq_ignore_ascii_case("purge") {
            let count = self.session.turns().len();
            self.session.clear_turns();
            if let Some(rewriter) = self.session.agent().rewriter()
                && let Err(e) = rewriter.clear_memory().await
            {
                warn!("Memory clear failed: {}", e);
            }
            info!("Conversation memory purged ({} entries removed).", count);
            return Ok(());
        }

        if arg.eq_ignore_ascii_case("off") || arg.eq_ignore_ascii_case("none") {
            let was = self.session.agent().rewriter().is_some();
            self.session.agent_mut().set_rewriter(None);
            self.session.set_use_transcript(false);
            let cleared = self.session.turns().len();
            self.session.clear_turns();
            if was {
                info!("Memory disabled ({} turns cleared).", cleared);
            } else if cleared > 0 {
                info!("Memory off — {} turns cleared.", cleared);
            } else {
                info!("Memory already off.");
            }
            return Ok(());
        }

        if arg.eq_ignore_ascii_case("log") {
            self.session.set_use_transcript(true);
            let old = self.session.history_strategy().map(|s| s.name());
            self.session
                .set_history_strategy(Some(Box::new(LogHistory)));
            match old {
                Some("log") => {
                    info!("History diffusion unchanged: log");
                }
                Some(o) => {
                    info!("History diffusion: {} → log", o);
                }
                None => info!("History diffusion enabled: log"),
            }
            return Ok(());
        }

        if arg.eq_ignore_ascii_case("summary") {
            self.session.set_use_transcript(true);
            let summary_spec = ChatAgentSpec::ollama(
                self.config.memory.model.clone(),
                GenerationParams::default(),
                None,
            );
            match summary_spec.build() {
                Ok(summary_agent) => {
                    let strat: Box<dyn HistoryStrategy> =
                        Box::new(SummaryHistory::new(summary_agent));
                    let old = self.session.history_strategy().map(|s| s.name());
                    self.session.set_history_strategy(Some(strat));
                    match old {
                        Some("summary") => {
                            info!("History diffusion unchanged: summary");
                        }
                        Some(o) => {
                            info!("History diffusion: {} → summary", o);
                        }
                        None => info!("History diffusion enabled: summary"),
                    }
                }
                Err(e) => RagrigError::log_or(&e, "Failed to build summary agent"),
            }
            return Ok(());
        }

        if arg.eq_ignore_ascii_case("transcript") {
            let was = self.session.agent().rewriter().is_some();
            self.session.agent_mut().set_rewriter(None);
            self.session.set_use_transcript(true);
            if was {
                info!("Memory strategy: rewrite → transcript");
            } else {
                info!("Memory unchanged: transcript");
            }
            return Ok(());
        }

        // ── LLM-backed memory (rewrite mode) ───────────────────────

        self.session.set_use_transcript(true);
        let mut parts = arg.split_whitespace();
        let backend = parts.next().unwrap_or("");
        let model = parts.next();
        let api_key = parts.next();

        let spec = match ChatAgentSpec::parse(backend, model, api_key, None) {
            Ok(s) => s,
            Err(e) => {
                error!("Memory spec parse error: {}", e);
                return Ok(());
            }
        };

        match spec.build() {
            Ok(new_rewriter) => {
                let new_backend = new_rewriter.backend_name();
                let new_model = new_rewriter.model_name().to_string();
                let was = self.session.agent().rewriter().is_some();
                self.session.agent_mut().set_rewriter(Some(new_rewriter));
                if was {
                    info!("Memory agent: {} ({})", new_backend, new_model);
                } else {
                    info!("Memory enabled: rewrite — {} ({})", new_backend, new_model);
                }
            }
            Err(e) => RagrigError::log_or(&e, "Failed to build memory agent"),
        }
        Ok(())
    }

    // ── /prompt [chat|rewrite|reset] [file] ──────────────────────────

    async fn cmd_prompt(&mut self, args_str: &str) -> Result<()> {
        let mut parts = args_str.split_whitespace();
        let sub = parts.next().unwrap_or("");
        if sub.is_empty() {
            println!("Current prompts:");
            println!(
                "  chat (docs):    {:.80}",
                self.session.agent().system_prompt().trim()
            );
            println!(
                "  chat (no docs): {:.80}",
                self.session.agent().chat_without_docs_prompt().trim()
            );
            println!(
                "  rewrite:        {:.80}",
                self.session.agent().rewrite_prompt().trim()
            );
            println!("Usage: /prompt chat|rewrite <file>  or  /prompt reset");
            return Ok(());
        }

        match sub {
            "reset" => {
                self.session.agent_mut().set_system_prompt(
                    "You are a helpful document assistant. Answer the user's question \
                     explicitly using the provided Context snippets.\n\
                     \n\
                     Context:\n{context}\n"
                        .to_string(),
                );
                self.session.agent_mut().set_rewrite_prompt(
                    "You are a query rewriter. Given the conversation and the \
                     latest question, produce a single self-contained search query \
                     that captures all relevant context. Output ONLY the rewritten \
                     query, nothing else.\n\n\
                     Latest question: {question}"
                        .to_string(),
                );
                info!("Prompts reset to defaults.");
            }
            "chat" => {
                let file = parts.next();
                let Some(file) = file else {
                    println!("Usage: /prompt chat <file>");
                    return Ok(());
                };
                match fs::read_to_string(file) {
                    Ok(text) => {
                        self.session.agent_mut().set_system_prompt(text);
                        info!("Chat prompt loaded from {}", file);
                    }
                    Err(e) => error!("Failed to load chat prompt from {}: {}", file, e),
                }
            }
            "rewrite" => {
                let file = parts.next();
                let Some(file) = file else {
                    println!("Usage: /prompt rewrite <file>");
                    return Ok(());
                };
                match fs::read_to_string(file) {
                    Ok(text) => {
                        self.session.agent_mut().set_rewrite_prompt(text);
                        info!("Rewrite prompt loaded from {}", file);
                    }
                    Err(e) => error!("Failed to load rewrite prompt from {}: {}", file, e),
                }
            }
            other => {
                println!(
                    "Unknown sub-command: {}. Use chat, rewrite, or reset.",
                    other
                );
            }
        }
        Ok(())
    }

    // ── /parser [pdf|epub] [sink|extract|internal|epub] ────────────

    async fn cmd_parser(&mut self, args_str: &str) -> Result<()> {
        let mut parts = args_str.split_whitespace();
        let format = parts.next().unwrap_or("");
        if format.is_empty() {
            println!("PDF:  {:?}", self.pdf_parser);
            println!("EPUB: {:?}", self.epub_parser);
            println!("Usage: /parser pdf {}", pdf_backend_names().join("|"));
            println!("       /parser epub epub");
            return Ok(());
        }

        let choice = parts.next().unwrap_or("");
        if choice.is_empty() {
            println!("Usage: /parser {} <backend>", format);
            return Ok(());
        }

        match format.to_lowercase().as_str() {
            "pdf" => {
                let new = match choice.to_lowercase().as_str() {
                    #[cfg(feature = "kreuzberg")]
                    "kreuzberg" => PdfParserBackend::Kreuzberg,
                    "unpdf" => PdfParserBackend::Unpdf,
                    "sink" => PdfParserBackend::Sink,
                    "extract" => PdfParserBackend::Extract,
                    "internal" => PdfParserBackend::Internal,
                    #[cfg(feature = "vision-pdf")]
                    "vision" => PdfParserBackend::Vision,
                    other => {
                        println!(
                            "Unknown PDF parser: {other}. Use {}.",
                            pdf_backend_names().join(", ")
                        );
                        return Ok(());
                    }
                };
                let old = std::mem::replace(&mut self.pdf_parser, new.clone());
                info!("PDF parser: {:?} → {:?}", old, new);
                // Rebuild the parser registry so the selected backend takes
                // effect — both for the session and for the agent's
                // ingestion methods.
                self.doc_parsers =
                    DocumentParsers::new(filtered_parsers(&new, self.config.parse.sloppy_pdf));
                self.session
                    .agent_mut()
                    .set_parsers(DocumentParsers::new(filtered_parsers(
                        &new,
                        self.config.parse.sloppy_pdf,
                    )));
                info!("Active parsers: {}", self.doc_parsers.names().join(", "));
                // Provenance check: warn when the new parser has no chunks in
                // the store yet — the user must run /embed index explicitly.
                self.pipeline_indexed().await;
            }
            "epub" => {
                let new = match choice.to_lowercase().as_str() {
                    "epub" => EpubParserBackend::Epub,
                    other => {
                        println!("Unknown EPUB parser: {}. The only option is 'epub'.", other);
                        return Ok(());
                    }
                };
                let old = std::mem::replace(&mut self.epub_parser, new.clone());
                info!("EPUB parser: {:?} → {:?}", old, new);
            }
            other => {
                println!("Unknown format: {}. Use pdf or epub.", other);
            }
        }
        Ok(())
    }

    // ── /corpus [<name> [on|off]] ─────────────────────────────

    /// Toggle a named document corpus on or off, or control dynamic routing.
    ///
    /// - `/corpus`                  — list all configured corpora + the `dyn` route
    /// - `/corpus <name>`           — show one corpus's state
    /// - `/corpus <name> on`        — index the corpus into the store (sync)
    /// - `/corpus <name> off`       — remove the corpus's chunks from the store
    /// - `/corpus dyn on|off`       — toggle dynamic routing for web downloads
    async fn cmd_corpus(&mut self, args_str: &str) -> Result<()> {
        let mut parts = args_str.split_whitespace();
        let Some(name) = parts.next() else {
            if self.corpora.is_empty() {
                println!(
                    "No named corpora configured. Use --corpus-dir NAME=DIR or \
                     --corpus-urls NAME=URL."
                );
            } else {
                println!("Document corpora:");
                for e in &self.corpora {
                    let state = if e.active { "on" } else { "off" };
                    println!("  {:<16} ({:<4}) {}", e.name, e.kind.label(), state);
                }
            }
            let state = if self.dyn_corpora { "on" } else { "off" };
            println!("  {:<16} ({:<4}) {}", "dyn", "route", state);
            println!("Usage: /corpus <name> on|off   (or /corpus to list)");
            return Ok(());
        };

        // `dyn` is the virtual dynamic-routing entry, not a real corpus.
        if name == "dyn" {
            match parts.next() {
                None => {
                    let state = if self.dyn_corpora { "on" } else { "off" };
                    println!(
                        "Dynamic routing (dyn) is {state}: /download and /get route into the first active URL corpus, else the first active directory corpus."
                    );
                }
                Some("on") => {
                    self.dyn_corpora = true;
                    println!(
                        "Dynamic routing is on — web downloads go to the first active URL corpus (else the first active directory corpus)."
                    );
                }
                Some("off") => {
                    self.dyn_corpora = false;
                    println!("Dynamic routing is off — web downloads go to the main folder.");
                }
                Some(other) => println!("Usage: /corpus dyn on|off (got '{other}')"),
            }
            return Ok(());
        }

        match parts.next() {
            None => match self.corpora.iter().find(|e| e.name == name) {
                Some(e) => {
                    let state = if e.active { "on" } else { "off" };
                    println!("Corpus '{}' ({}) is {}", e.name, e.kind.label(), state);
                }
                None => println!("Unknown corpus '{name}'. Run /corpus to list corpora."),
            },
            Some("on") => self.corpus_on(name).await?,
            Some("off") => self.corpus_off(name).await?,
            Some(other) => println!("Usage: /corpus <name> on|off (got '{other}')"),
        }
        Ok(())
    }

    /// Sync one named corpus into the store with the current pipeline.
    /// Returns the number of documents indexed.
    async fn sync_corpus_entry(&mut self, idx: usize) -> Result<usize> {
        let token = CancellationToken::new();
        let watcher = EscWatcher::spawn(token.clone());
        let state = Arc::new(Mutex::new(EmbedProgress::default()));
        let sink = embed_progress_sink(state);
        let result = self
            .session
            .agent()
            .sync_corpus_with_progress(self.corpora[idx].as_corpus(), Some(&sink), Some(&token))
            .await;
        drop(watcher);
        eprint!("\r\x1b[2K");
        Ok(result?.len())
    }

    /// Enable a corpus: sync its documents into the store.
    ///
    /// Calling `on` for an already-active corpus re-syncs it, so documents
    /// added to the corpus at runtime are picked up.
    async fn corpus_on(&mut self, name: &str) -> Result<()> {
        let Some(idx) = self.corpora.iter().position(|e| e.name == name) else {
            anyhow::bail!("Unknown corpus '{name}'. Run /corpus to list corpora.");
        };
        if self.corpora[idx].active {
            println!("Corpus '{name}' is already on — syncing for changes.");
        } else {
            println!(
                "Indexing corpus '{name}' ({})...",
                self.corpora[idx].kind.label()
            );
        }
        let indexed = self.sync_corpus_entry(idx).await?;
        self.corpora[idx].active = true;
        println!(
            "Corpus '{name}' is on ({} documents indexed; store: {} chunks).",
            indexed,
            self.session.agent().store().len()
        );
        Ok(())
    }

    /// Disable a corpus: remove its chunks from the store.
    async fn corpus_off(&mut self, name: &str) -> Result<()> {
        let Some(idx) = self.corpora.iter().position(|e| e.name == name) else {
            anyhow::bail!("Unknown corpus '{name}'. Run /corpus to list corpora.");
        };
        if !self.corpora[idx].active {
            println!("Corpus '{name}' is already off.");
            return Ok(());
        }
        self.corpora[idx].active = false;
        self.session.agent().store().delete_corpus(name).await?;
        println!("Corpus '{name}' is off; its chunks were removed from the store.");
        Ok(())
    }

    // ── /profile [save|show|load|list] [name] ────────────────────

    /// Save, show, load, or list configuration profiles.
    ///
    /// Profiles are stored as JSON files in `.ragrig/profiles/`.
    /// Use `--profile <name>` at startup to load a profile automatically.
    async fn cmd_profile(&mut self, args_str: &str) -> Result<()> {
        let mut parts = args_str.split_whitespace();
        let sub = parts.next().unwrap_or("");
        let name = parts.next().unwrap_or("default");

        match sub {
            "" | "list" => {
                let profiles = RagrigConfig::list_profiles(&self.config.workspace)?;
                if profiles.is_empty() {
                    println!("No saved profiles. Use /profile save <name> to create one.");
                } else {
                    println!("Saved profiles:");
                    for p in &profiles {
                        let marker = if p == name || (profiles.len() == 1 && sub.is_empty()) {
                            " *"
                        } else {
                            ""
                        };
                        println!("  {}{marker}", p);
                    }
                    if !profiles.contains(&name.to_string()) && !sub.is_empty() {
                        println!("  (profile '{}' not found)", name);
                    }
                }
            }
            "save" => {
                // Update the in-memory config from the running agent state
                // before saving, so the profile reflects current runtime settings.
                self.sync_config_from_agent();
                self.config.save_to_profile(&self.config.workspace, name)?;
                println!("Profile '{}' saved.", name);
            }
            "show" => {
                let config = if name == "current"
                    || parts.next().is_none()
                        && name == "default"
                        && RagrigConfig::list_profiles(&self.config.workspace)?.is_empty()
                {
                    self.sync_config_from_agent();
                    self.config.clone()
                } else {
                    match RagrigConfig::load_from_profile(&self.config.workspace, name) {
                        Ok(c) => c,
                        Err(e) => {
                            // Try showing the current in-memory config if named profile not found.
                            if name == "current" {
                                self.sync_config_from_agent();
                                self.config.clone()
                            } else {
                                println!("{}", e);
                                return Ok(());
                            }
                        }
                    }
                };
                println!("{}", serde_json::to_string_pretty(&config)?);
            }
            "load" => {
                let profile = RagrigConfig::load_from_profile(&self.config.workspace, name)?;
                // Keep the current workspace — profiles don't override it.
                let workspace = self.config.workspace.clone();
                self.config = profile;
                self.config.workspace = workspace;
                println!(
                    "Profile '{}' loaded. Use /chat, /embed, /memory to apply.",
                    name
                );
                info!(
                    "Loaded profile '{}': chat={} embed={} memory={}",
                    name, self.config.chat.model, self.config.embed.model, self.config.memory.model,
                );
            }
            _ => {
                println!("Usage: /profile [save|show|load|list] [name]");
                println!("  save <name>  — save current config as a profile");
                println!(
                    "  show [name]  — display a profile as JSON (or 'current' for running state)"
                );
                println!("  load <name>  — load a profile (use /chat, /embed, /memory to apply)");
                println!("  list         — list saved profiles");
            }
        }
        Ok(())
    }

    /// Copy the running agent state back into `self.config` so profile
    /// saves reflect any runtime hot-swaps.
    fn sync_config_from_agent(&mut self) {
        self.config.chat.model = self.session.agent().chat_agent().model_name().to_string();
        self.config.embed.model = self.session.agent().embedder().model_name().to_string();
        self.config.embed.top_k = self.session.agent().top_k();
        self.config.embed.similarity_threshold = self.session.agent().similarity_threshold();
        self.config.chat.context_tokens = self.session.agent().context_tokens();
        if let Some(rw) = self.session.agent().rewriter() {
            self.config.memory.model = rw.model_name().to_string();
        }
    }

    // ── /log [level] ───────────────────────────────────────────────

    /// Show or change the log verbosity at runtime.
    ///
    /// Without an argument prints the current filter.  With an argument
    /// sets it: `off`, `error`, `warn`, `info`, `debug`, or `trace`.
    async fn cmd_log(&mut self, args_str: &str) -> Result<()> {
        let level = args_str.trim().to_lowercase();
        if level.is_empty() {
            println!(
                "Interactive log level: {} — change with /log <off|error|warn|info|debug|trace>",
                self.log_level.read().unwrap_or_else(|e| e.into_inner())
            );
            return Ok(());
        }

        // `debug` and `trace` use target-filtered directives to
        // avoid keystroke‑level noise from `rustyline` and other
        // dependencies.  Only the `ragrig` crate (binary + library)
        // gets the elevated level; everything else stays at `info`.
        // (The log file always records at `ragrig=debug` independently.)
        let filter_str = match level.as_str() {
            "off" => "off",
            "error" => "error",
            "warn" => "warn",
            "info" => "info",
            "debug" => "ragrig=debug,info",
            "trace" => "ragrig=trace,info",
            other => {
                println!(
                    "Unknown log level '{}'. Use off, error, warn, info, debug, or trace.",
                    other
                );
                return Ok(());
            }
        };

        *self.log_level.write().unwrap_or_else(|e| e.into_inner()) = filter_str.to_string();
        info!("Interactive log level set to {}.", level);
        Ok(())
    }

    // ── Normal RAG query ─────────────────────────────────────────────

    /// Execute a full RAG pipeline: rewrite → embed → search → prompt → generate.
    ///
    /// 1. **Rewrite** — the memory agent expands pronouns and implicit
    ///    context into a self-contained search query (skipped if `/memory off`).
    /// 2. **Embed + Search** — the embedder vectorises the rewritten query;
    ///    the vector store performs hybrid BM25 + cosine RRF retrieval.
    /// 3. **Prompt construction** — system prompt + retrieved context +
    ///    conversation Memory (when enabled) + current question, formatted
    ///    as a single string with chat-template tokens.
    /// 4. **Generate** — the chat agent streams the response token by token.
    ///
    /// Retrieved context is truncated to `(model_ctx_tokens − 1024) × 3`
    /// chars to avoid exceeding the model's context window.
    async fn cmd_rag_query(&mut self, query: &str) -> Result<()> {
        trace!("Query: {:?}", query);

        trace!("Agent config: {:?}", self.session.agent());
        debug!(
            "Provider: {} | Model: {}",
            self.session.agent().chat_agent().backend_name(),
            self.session.agent().chat_agent().model_name()
        );

        let has_attachments = !self.attached_docs.is_empty();
        if has_attachments {
            debug!(
                "Attachments: {} document(s) — {}",
                self.attached_docs.len(),
                self.attached_docs
                    .iter()
                    .map(|d| format!("{} ({} chars)", d.name, d.content.len()))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        print!("Assistant > ");
        stdout().flush()?;

        let start = std::time::Instant::now();
        // Attachments are one-shot — take them out and let the session
        // consume them for this query only.
        let attached = std::mem::take(&mut self.attached_docs);

        // ESC cancels: a raw-mode stdin watcher flips the token, the
        // generator polls it between streamed tokens.
        let token = CancellationToken::new();
        let watcher = EscWatcher::spawn(token.clone());

        let token_count = Arc::new(AtomicUsize::new(0));
        let on_token = {
            let token_count = token_count.clone();
            move |t: String| {
                token_count.fetch_add(1, Ordering::Relaxed);
                print!("{}", t);
                let _ = stdout().flush();
            }
        };

        let response = if has_attachments {
            self.session
                .chat_streaming_detailed_with_attachments(query, &attached, &on_token, Some(&token))
                .await
        } else {
            self.session
                .chat_streaming_detailed(query, &on_token, Some(&token))
                .await
        };
        drop(watcher);

        match response {
            Ok(resp) => {
                println!(); // terminate the streamed answer line

                // ── Trace: pipeline metadata ────────────────────
                if let Some(ref rw) = resp.rewritten_query
                    && rw != query
                {
                    trace!("Rewritten query: {:?}", rw);
                }
                trace!(
                    "Chunks retrieved: {:?}  |  Documents: {:?}",
                    resp.chunks_retrieved, resp.documents
                );
                trace!(
                    "System prompt ({} chars): {:.300}...",
                    resp.system_prompt.len(),
                    &resp.system_prompt[..300.min(resp.system_prompt.len())]
                );
                trace!(
                    "User prompt ({} chars): {}",
                    resp.user_prompt.len(),
                    resp.user_prompt
                );
                trace!("Elapsed: {:?}", resp.elapsed);

                // Info header.
                let chunks = resp.chunks_retrieved.unwrap_or(0);
                let tokens = token_count.load(Ordering::Relaxed);
                let secs = start.elapsed().as_secs_f64();
                let attach_hint = if has_attachments {
                    format!(" + {} attached doc(s)", attached.len())
                } else {
                    String::new()
                };
                if let Some(ref documents) = resp.documents {
                    let names: Vec<&str> = documents.iter().map(|s| s.0.as_str()).collect();
                    println!(
                        "--- {} chunks | {} tokens from [{}]{} in {:.1}s ---",
                        chunks,
                        tokens,
                        names.join(", "),
                        attach_hint,
                        secs
                    );
                } else {
                    println!(
                        "--- {} chunks | {} tokens{} in {:.1}s ---",
                        chunks, tokens, attach_hint, secs
                    );
                }
            }
            Err(e) if e.downcast_ref::<ragrig::Cancelled>().is_some() => {
                println!();
                let tokens = token_count.load(Ordering::Relaxed);
                println!("[cancelled after {} tokens]", tokens);
            }
            Err(e) => {
                println!();
                if self.context_size_forced == ContextSizeMode::Auto
                    && self.session.agent_mut().try_recover_from_error(&e)
                {
                    if let Some(re) = e.downcast_ref::<RagrigError>() {
                        error!(
                            "Context overflow: model allows {} tokens, prompt needed {}. Budget auto-adjusted to {}.",
                            re.max_size(),
                            re.current_size(),
                            self.session.agent().context_tokens(),
                        );
                        eprintln!(
                            "\n*** Context overflow: model allows {} tokens, prompt needed {}. ***",
                            re.max_size(),
                            re.current_size()
                        );
                        eprintln!(
                            "*** Budget auto-adjusted to {} tokens. Use `/chat context {}` to override. ***",
                            self.session.agent().context_tokens(),
                            re.max_size().saturating_sub(512)
                        );
                    }
                } else {
                    RagrigError::log_or(&e, "Generation failed");
                }
            }
        }
        println!();

        Ok(())
    }
}

// ── Demo mode ───────────────────────────────────────────────────────────────

/// The chat model `--demo` uses: ~2.2 GB at Q4, so it sits on an 8 GB GPU
/// with plenty of headroom for the 4096-token context and KV cache
/// (https://localaimaster.com/vram/best-ollama-models-8gb-vram).
const DEMO_CHAT_MODEL: &str = "llama3.2:3b";

/// The first question `--demo` prefills into the prompt line.
const DEMO_FIRST_QUESTION: &str = "What is Bayesian Statistics?";

/// `--demo` startup: pin the small chat model with a 4096-token context and
/// swap the implicit corpus for the embedded HTML fixture book.  Returns the
/// fixture `TempDir`, kept alive for the lifetime of the process.
#[cfg(feature = "test-fixtures")]
fn apply_demo_setup(
    demo: bool,
    had_explicit_corpora: bool,
    config: &mut RagrigConfig,
) -> Result<Option<tempfile::TempDir>> {
    if !demo {
        return Ok(None);
    }
    let (fixture_dir, tmp) = ragrig::fixtures::extract_fixtures("html")?;
    let mut corpora = vec![format!("book={}", fixture_dir.display())];
    if had_explicit_corpora {
        corpora.append(&mut config.corpus_dirs);
    }
    config.corpus_dirs = corpora;
    config.chat.model = DEMO_CHAT_MODEL.into();
    config.chat.context_tokens = 4096;
    Ok(Some(tmp))
}

// ── main: parse → bootstrap → central match loop ──────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    #[cfg(feature = "test-fixtures")]
    let demo = cli.demo;
    #[cfg(not(feature = "test-fixtures"))]
    let demo = false;
    // Explicit corpora survive demo mode; the implicit `folder=.` corpus does not.
    let had_explicit_corpora = !cli.corpus_dirs.is_empty() || !cli.corpus_urls.is_empty();
    let profile_name = cli.profile.clone();
    let mut cli_config: RagrigConfig = cli.into();
    #[cfg(feature = "test-fixtures")]
    let _demo_fixtures = apply_demo_setup(demo, had_explicit_corpora, &mut cli_config)?;

    // If --profile was given, load it and merge CLI overrides on top.
    let config = if let Some(ref name) = profile_name {
        match RagrigConfig::load_from_profile(&cli_config.workspace, name) {
            Ok(mut profile) => {
                profile.override_with(&cli_config);
                info!("Loaded profile '{}' with CLI overrides applied.", name);
                profile
            }
            Err(e) => {
                // Profile not found or corrupt — warn and use CLI values only.
                warn!("Profile '{}' could not be loaded: {}", name, e);
                cli_config
            }
        }
    } else {
        cli_config
    };

    // ── File logging (always debug level, daily rotation) ───────────
    let log_dir = config.workspace.join(".ragrig");
    let _ = std::fs::create_dir_all(&log_dir);
    let file_appender = rolling::daily(&log_dir, "ragrig.log");
    let (non_blocking, _log_guard) = tracing_appender::non_blocking(file_appender);

    // ── Interactive (stderr) filter: changed at runtime via /log ────
    let initial_level = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into());
    let stderr_level: Arc<RwLock<String>> = Arc::new(RwLock::new(initial_level));

    let stderr_filter = {
        let lvl = stderr_level.clone();
        filter_fn(move |meta| {
            let filter_str = lvl.read().unwrap_or_else(|e| e.into_inner());
            match filter_str.as_str() {
                "off" => false,
                "error" => *meta.level() <= Level::ERROR,
                "warn" => *meta.level() <= Level::WARN,
                "info" => *meta.level() <= Level::INFO,
                "debug" => *meta.level() <= Level::DEBUG,
                "trace" => true,
                // Target-filtered: only ragrig at elevated level
                "ragrig=debug,info" => {
                    if meta.target().starts_with("ragrig") {
                        *meta.level() <= Level::DEBUG
                    } else {
                        *meta.level() <= Level::INFO
                    }
                }
                "ragrig=trace,info" => {
                    if meta.target().starts_with("ragrig") {
                        true
                    } else {
                        *meta.level() <= Level::INFO
                    }
                }
                _ => *meta.level() <= Level::INFO,
            }
        })
    };

    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .without_time()
                .with_target(false)
                .with_filter(stderr_filter),
        )
        .with(
            fmt::layer()
                .without_time()
                .with_target(false)
                .with_writer(non_blocking)
                .with_filter(EnvFilter::new("ragrig=debug")),
        )
        .init();

    let mut session = bootstrap(config, stderr_level).await?;

    if demo {
        // Memory off: no query rewriting, no transcript accumulation — every
        // question stands alone (the `/memory off` behaviour).
        session.session.agent_mut().set_rewriter(None);
        session.session.set_use_transcript(false);
        session.session.clear_turns();
        println!("Welcome to ragrig demo mode!");
        println!();
        println!("This session answers questions about the book:");
        println!(
            "  Martin Schmettow, New Statistics for Design Researchers. A Bayesian workflow in tidy R."
        );
        println!("  https://schmettow.github.io/New_Stats/");
        println!();
        println!(
            "Demo mode requires the Ollama models 'nomic-embed-text:latest' and '{DEMO_CHAT_MODEL}'."
        );
        println!("Memory is off — every question is independent.");
        println!();
        println!("Press Enter to run the prefilled question, or edit it first.");
    }

    let mut first_prompt = demo;
    loop {
        let readline = if first_prompt {
            first_prompt = false;
            session
                .rl
                .readline_with_initial("Query > ", (DEMO_FIRST_QUESTION, ""))
        } else {
            session.rl.readline("Query > ")
        };

        let cmd = match readline {
            Ok(line) => {
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    continue;
                }
                session.rl.add_history_entry(&trimmed)?;
                Command::from(trimmed.as_str())
            }
            Err(ReadlineError::Interrupted) => {
                info!("Session interrupted.");
                break;
            }
            Err(ReadlineError::Eof) => {
                info!("Session ended via Ctrl+D.");
                break;
            }
            Err(err) => {
                error!("Error reading input: {}", err);
                break;
            }
        };

        match cmd {
            Command::Exit => break,
            Command::RagQuery(ref q) if q.is_empty() => continue,
            _ => {
                if let Err(e) = session.execute(cmd).await {
                    RagrigError::log_or(&e, "Command execution error");
                }
            }
        }
    }

    // Auto‑save the session before exiting (skip empty transcripts so we
    // don't litter the store with blank session files).
    if !session.session.turns().is_empty()
        && let Err(e) = session.session.save().await
    {
        warn!("Failed to save session on exit: {}", e);
    }
    session.rl.save_history(&session.history_path)?;
    Ok(())
}

// ── ESC watcher + embedding progress bar ──────────────────────────────────

/// Watches stdin for the ESC byte (0x1b) while an operation runs and flips
/// the cancellation token when it arrives.  Puts the terminal into raw mode
/// for its lifetime and restores it on drop.
///
/// rustyline owns the terminal while reading a line, so the watcher is only
/// ever active between `readline` calls — the two never overlap.  When stdin
/// is not a TTY (piped input), the watcher degrades to an inactive no-op.
struct EscWatcher {
    handle: Option<std::thread::JoinHandle<()>>,
    original: Option<nix::sys::termios::Termios>,
    shutdown: Arc<AtomicBool>,
}

impl EscWatcher {
    /// Spawn the watcher.  Never fails: without a TTY it just stays inactive.
    fn spawn(token: CancellationToken) -> Self {
        use nix::sys::termios::{LocalFlags, SetArg, tcgetattr, tcsetattr};
        use std::os::fd::{AsRawFd, BorrowedFd};

        let stdin_fd = std::io::stdin().as_raw_fd();
        let stdin = unsafe { BorrowedFd::borrow_raw(stdin_fd) };
        let shutdown = Arc::new(AtomicBool::new(false));

        // Switch to raw mode (no canonical line buffering, no echo) so ESC
        // arrives immediately instead of waiting for a newline.
        let original = tcgetattr(stdin)
            .and_then(|orig| {
                let mut raw = orig.clone();
                raw.local_flags
                    .remove(LocalFlags::ICANON | LocalFlags::ECHO);
                tcsetattr(stdin, SetArg::TCSANOW, &raw).map(|_| orig)
            })
            .ok();

        let handle = original.as_ref().map(|_| {
            let shutdown = shutdown.clone();
            std::thread::spawn(move || {
                use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
                let mut fds = [PollFd::new(stdin, PollFlags::POLLIN)];
                loop {
                    if shutdown.load(Ordering::Relaxed) {
                        return;
                    }
                    match poll(&mut fds, PollTimeout::from(100u16)) {
                        Ok(0) => continue, // timeout — re-check shutdown
                        Ok(_) => {
                            let mut buf = [0u8; 16];
                            match nix::unistd::read(stdin, &mut buf) {
                                Ok(0) => return, // EOF — nothing to watch
                                Ok(n) => {
                                    if buf[..n].contains(&0x1b) {
                                        token.cancel();
                                        return;
                                    }
                                }
                                Err(_) => return,
                            }
                        }
                        Err(_) => return,
                    }
                }
            })
        });

        Self {
            handle,
            original,
            shutdown,
        }
    }
}

impl Drop for EscWatcher {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take()
            && handle.join().is_err()
        {
            // The watcher thread cannot panic by construction; ignore.
        }
        if let Some(original) = &self.original {
            use std::os::fd::{AsRawFd, BorrowedFd};
            let stdin = unsafe { BorrowedFd::borrow_raw(std::io::stdin().as_raw_fd()) };
            let _ =
                nix::sys::termios::tcsetattr(stdin, nix::sys::termios::SetArg::TCSANOW, original);
        }
    }
}

/// Aggregate state for the embedding progress bar.
#[derive(Default)]
struct EmbedProgress {
    total: usize,
    started: usize,
    done: usize,
    chunks: usize,
    failed: usize,
    current: String,
}

/// Draw the one-line embedding progress bar to stderr (overwritten with
/// `\r` on the next event; cleared by printing `\r\x1b[2K`).
fn render_embed_progress(state: &EmbedProgress) {
    const WIDTH: usize = 30;
    let frac = if state.total > 0 {
        (state.started as f64 / state.total as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let filled = (frac * WIDTH as f64) as usize;
    let bar = format!(
        "[{}{}]",
        "#".repeat(filled),
        "-".repeat(WIDTH.saturating_sub(filled))
    );
    let failed = if state.failed > 0 {
        format!(" | {} failed", state.failed)
    } else {
        String::new()
    };
    eprint!(
        "\r\x1b[2K{bar} {}/{} files | {} chunks{failed} | {} (ESC: cancel)",
        state.started, state.total, state.chunks, state.current
    );
}

/// Build the closure-based [`ragrig::Progress`] reporter for one indexing
/// run, sharing `state` between events.
fn embed_progress_sink(state: Arc<Mutex<EmbedProgress>>) -> impl Fn(&ProgressEvent) + Send + Sync {
    move |event: &ProgressEvent| {
        let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
        match event {
            ProgressEvent::FileStarted {
                index,
                total,
                corpus,
                document,
            } => {
                st.total = *total;
                st.started = *index + 1;
                st.current = format!("{corpus}/{document}");
            }
            ProgressEvent::FileChunked { .. } | ProgressEvent::FileEmbedded { .. } => {}
            ProgressEvent::ChunksEmbedded { done, .. } => st.chunks = *done,
            ProgressEvent::FileStored => st.done += 1,
            ProgressEvent::FileFailed { .. } => st.failed += 1,
            // `ProgressEvent` is #[non_exhaustive] — future variants are ignored.
            _ => {}
        }
        render_embed_progress(&st);
    }
}

// ── Utility functions ─────────────────────────────────────────────────────

/// Strip ANSI escape sequences (bracketed paste, colors, etc.) from a string.
fn strip_ansi(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            // Consume the escape sequence: ESC[... until a letter
            chars.next(); // skip '['
            while let Some(&nc) = chars.peek() {
                chars.next();
                if nc.is_alphabetic() || nc == '~' {
                    break;
                }
            }
        } else {
            result.push(c);
        }
    }
    result
}

/// Parse "1,2,3-4,8" into zero-based indices [0,1,2,3,7]
fn parse_number_range(input: &str) -> Result<Vec<usize>, String> {
    let mut indices = Vec::new();
    for part in input.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            let s: usize = start
                .trim()
                .parse()
                .map_err(|_| format!("invalid number: {}", start))?;
            let e: usize = end
                .trim()
                .parse()
                .map_err(|_| format!("invalid number: {}", end))?;
            if s == 0 || e == 0 {
                return Err("Indices start at 1".to_string());
            }
            if s > e {
                return Err(format!("invalid range: {}-{}", s, e));
            }
            for n in s..=e {
                indices.push(n - 1); // convert to zero-based
            }
        } else {
            let n: usize = part
                .parse()
                .map_err(|_| format!("invalid number: {}", part))?;
            if n == 0 {
                return Err("Indices start at 1".to_string());
            }
            indices.push(n - 1);
        }
    }
    Ok(indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ragrig::{Cancel, TurnRole};

    // ── Command::from ─────────────────────────────────────────────────

    #[test]
    fn parse_unknown_slash_command_is_not_query() {
        // Any input starting with / that doesn't match a known command
        // must be Unknown, never RagQuery.
        let cmd = Command::from("/foobar");
        assert!(matches!(cmd, Command::Unknown(_)));
    }

    #[test]
    fn parse_unknown_slash_with_args() {
        let cmd = Command::from("/bogus arg1 arg2");
        assert!(matches!(cmd, Command::Unknown(c) if c.contains("arg1")));
    }

    #[test]
    fn parse_plain_text_is_rag_query() {
        let cmd = Command::from("What is RAG?");
        assert!(matches!(cmd, Command::RagQuery(q) if q == "What is RAG?"));
    }

    // ── /corpus command ───────────────────────────────────────────────

    #[test]
    fn parse_corpus_no_args() {
        let cmd = Command::from("/corpus");
        assert!(matches!(cmd, Command::Corpus(s) if s.is_empty()));
    }

    #[test]
    fn parse_corpus_toggle() {
        let cmd = Command::from("/corpus papers on");
        assert!(matches!(cmd, Command::Corpus(s) if s == "papers on"));
    }

    #[test]
    fn corpus_command_parses_dyn_toggle() {
        let on = Command::from("/corpus dyn on");
        assert!(matches!(on, Command::Corpus(s) if s == "dyn on"));
        let off = Command::from("/corpus dyn off");
        assert!(matches!(off, Command::Corpus(s) if s == "dyn off"));
    }

    // ── CLI clap parsing (--corpus-dir / --corpus-urls) ───────────────

    #[test]
    fn cli_parses_corpus_flags() {
        let cli = Cli::try_parse_from([
            "ragrig",
            "--corpus-dir",
            "papers=/tmp/papers",
            "--corpus-dir",
            "books=/tmp/books",
            "--corpus-urls",
            "urls=https://example.com/a.pdf,https://example.com/b.pdf",
        ])
        .expect("CLI args should parse");

        // The repeatable flags land in the Cli struct in order.
        assert_eq!(
            cli.corpus_dirs,
            vec!["papers=/tmp/papers", "books=/tmp/books"]
        );
        assert_eq!(
            cli.corpus_urls,
            vec!["urls=https://example.com/a.pdf,https://example.com/b.pdf"]
        );

        // And they survive the conversion into the library config.
        let config = RagrigConfig::from(cli);
        assert_eq!(
            config.corpus_dirs,
            vec!["papers=/tmp/papers", "books=/tmp/books"]
        );
        assert_eq!(
            config.corpus_urls,
            vec!["urls=https://example.com/a.pdf,https://example.com/b.pdf"]
        );
    }

    // ── parse_corpora ─────────────────────────────────────────────────

    #[test]
    fn parse_corpora_dir_and_urls() {
        let client = reqwest::Client::new();
        let entries = parse_corpora(
            &["papers=/data/papers".to_string()],
            &["arxiv=https://arxiv.org/pdf/a.pdf".to_string()],
            &client,
        )
        .unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "papers");
        assert_eq!(entries[0].kind.label(), "dir");
        assert!(entries[0].active);
        assert_eq!(entries[0].as_corpus().name(), "papers");
        assert_eq!(entries[1].name, "arxiv");
        assert_eq!(entries[1].kind.label(), "urls");
        assert!(entries[1].active);
        assert_eq!(entries[1].as_corpus().name(), "arxiv");
    }

    #[test]
    fn parse_corpora_repeated_url_name_merges() {
        let client = reqwest::Client::new();
        let entries = parse_corpora(
            &[],
            &[
                "arxiv=https://arxiv.org/pdf/a.pdf".to_string(),
                "arxiv=https://arxiv.org/pdf/b.pdf".to_string(),
            ],
            &client,
        )
        .unwrap();
        // Both URLs land in one curated corpus, not two clashing ones.
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "arxiv");
        assert_eq!(entries[0].kind.label(), "urls");
        assert_eq!(entries[0].as_corpus().name(), "arxiv");
    }

    #[test]
    fn parse_corpora_duplicate_dir_name_errors() {
        let client = reqwest::Client::new();
        let err = parse_corpora(
            &["docs=/a".to_string(), "docs=/b".to_string()],
            &[],
            &client,
        )
        .unwrap_err();
        assert!(err.to_string().contains("Duplicate corpus name 'docs'"));
    }

    #[test]
    fn parse_corpora_name_clash_between_kinds_errors() {
        let client = reqwest::Client::new();
        let err = parse_corpora(
            &["docs=/a".to_string()],
            &["docs=https://example.com/x.pdf".to_string()],
            &client,
        )
        .unwrap_err();
        assert!(err.to_string().contains("Duplicate corpus name 'docs'"));
    }

    #[test]
    fn parse_corpora_missing_equals_errors() {
        let client = reqwest::Client::new();
        assert!(parse_corpora(&["nodir".to_string()], &[], &client).is_err());
        assert!(parse_corpora(&[], &["nourl".to_string()], &client).is_err());
    }

    // ── resolve_workspace_and_corpora (--folder / --workspace) ──────

    #[test]
    fn resolve_folder_is_workspace_plus_corpus() {
        let (ws, dirs) =
            resolve_workspace_and_corpora(Some(&PathBuf::from("/data/docs")), None, vec![], &[]);
        assert_eq!(ws, PathBuf::from("/data/docs"));
        assert_eq!(dirs, vec!["folder=/data/docs".to_string()]);
    }

    #[test]
    fn resolve_workspace_alone_has_no_implicit_corpus() {
        let (ws, dirs) =
            resolve_workspace_and_corpora(None, Some(&PathBuf::from("/data/ws")), vec![], &[]);
        assert_eq!(ws, PathBuf::from("/data/ws"));
        assert!(dirs.is_empty());
    }

    #[test]
    fn resolve_bare_invocation_defaults_to_cwd() {
        let (ws, dirs) = resolve_workspace_and_corpora(None, None, vec![], &[]);
        assert_eq!(ws, PathBuf::from("."));
        assert_eq!(dirs, vec!["folder=.".to_string()]);
    }

    #[test]
    fn resolve_explicit_corpora_suppress_implicit_dir() {
        let (ws, dirs) = resolve_workspace_and_corpora(
            None,
            None,
            vec![],
            &["arxiv=https://arxiv.org/pdf/a.pdf".to_string()],
        );
        assert_eq!(ws, PathBuf::from("."));
        assert!(dirs.is_empty());
    }

    // ── pick_web_route (dyn routing rules) ─────────────────────────────

    fn dir_entry(name: &str) -> CorpusEntry {
        CorpusEntry {
            name: name.into(),
            kind: CorpusKind::Dir(FolderCorpus::named(name, "/tmp")),
            active: true,
        }
    }

    fn url_entry(name: &str) -> CorpusEntry {
        CorpusEntry {
            name: name.into(),
            kind: CorpusKind::Urls(UrlCorpus::new(name, reqwest::Client::new())),
            active: true,
        }
    }

    #[test]
    fn route_dyn_off_goes_to_folder() {
        let corpora = vec![dir_entry("papers"), url_entry("arxiv")];
        assert!(matches!(pick_web_route(false, &corpora), WebRoute::Folder));
    }

    #[test]
    fn route_without_corpora_goes_to_folder() {
        assert!(matches!(pick_web_route(true, &[]), WebRoute::Folder));
    }

    #[test]
    fn route_prefers_first_active_url_corpus() {
        let corpora = vec![dir_entry("papers"), url_entry("arxiv")];
        match pick_web_route(true, &corpora) {
            WebRoute::Url { name } => assert_eq!(name, "arxiv"),
            other => panic!("expected Url route, got {other:?}"),
        }
    }

    #[test]
    fn route_falls_back_to_first_active_dir_corpus() {
        let corpora = vec![dir_entry("papers"), dir_entry("books")];
        match pick_web_route(true, &corpora) {
            WebRoute::Dir { name } => assert_eq!(name, "papers"),
            other => panic!("expected Dir route, got {other:?}"),
        }
    }

    #[test]
    fn route_skips_inactive_corpora() {
        let mut url = url_entry("arxiv");
        url.active = false;
        let mut dir = dir_entry("papers");
        dir.active = false;
        // Inactive URL + active dir → dir wins.
        let corpora = vec![url.clone(), dir_entry("books")];
        match pick_web_route(true, &corpora) {
            WebRoute::Dir { name } => assert_eq!(name, "books"),
            other => panic!("expected Dir route, got {other:?}"),
        }
        // Everything inactive → folder.
        let corpora = vec![url, dir];
        assert!(matches!(pick_web_route(true, &corpora), WebRoute::Folder));
    }

    #[test]
    fn parse_memory_no_args_does_not_panic() {
        // Regression: /memory with no trailing content used to panic
        // on input[8..] when input was only 7 chars.
        let cmd = Command::from("/memory");
        assert!(matches!(cmd, Command::Memory(s) if s.is_empty()));
    }

    #[test]
    fn parse_memory_with_args() {
        let cmd = Command::from("/memory transcript");
        assert!(matches!(cmd, Command::Memory(s) if s == "transcript"));
    }

    #[test]
    fn parse_hist_no_args_does_not_panic() {
        let cmd = Command::from("/hist");
        assert!(matches!(cmd, Command::Hist(s) if s.is_empty()));
    }

    #[test]
    fn parse_chat_command_recognised() {
        // Regression: /chat was broken and fell through to RagQuery.
        let cmd = Command::from("/chat ollama");
        assert!(matches!(cmd, Command::Chat(s) if s == "ollama"));
    }

    #[test]
    fn parse_embed_no_args_does_not_panic() {
        let cmd = Command::from("/embed");
        assert!(matches!(cmd, Command::Embed(s) if s.is_empty()));
    }

    #[test]
    fn parse_refs_no_args_does_not_panic() {
        let cmd = Command::from("/refs");
        assert!(matches!(cmd, Command::ExtractRefs(s) if s.is_empty()));
    }

    #[test]
    fn parse_slash_exit() {
        assert!(matches!(Command::from("/exit"), Command::Exit));
    }

    #[test]
    fn parse_slash_bye() {
        assert!(matches!(Command::from("/bye"), Command::Exit));
    }

    /// The ESC watcher degrades to an inert no-op without a TTY (piped
    /// stdin, e.g. CI) — spawn and drop must both be safe.
    #[test]
    fn esc_watcher_without_tty_is_inert() {
        let token = CancellationToken::new();
        let watcher = EscWatcher::spawn(token.clone());
        assert!(!token.is_cancelled());
        drop(watcher);
        assert!(!token.is_cancelled());
    }

    // ── Integration test ─────────────────────────────────────────────

    /// Full RAG integration test — requires a running Ollama server
    /// with gemma4:e4b pulled, and tests/fixtures/formats/pdf indexed.
    ///
    /// Run with: cargo test --features ollama-embed -- --ignored
    #[tokio::test]
    #[ignore = "requires Ollama with gemma4:e4b and tests/fixtures/formats/pdf"]
    async fn gemma4_rag_answer_exceeds_20_words() {
        let log_level = Arc::new(RwLock::new("warn".into()));

        let config = RagrigConfig {
            workspace: "tests/fixtures/formats/pdf".into(),
            // Equivalent to the old `--folder` shortcut: the workspace is
            // state-only; documents come from named corpora.
            corpus_dirs: vec!["folder=tests/fixtures/formats/pdf".to_string()],
            chat: ChatConfig {
                model: "gemma4:e4b".into(),
                ..Default::default()
            },
            embed: EmbedConfig {
                model: "nomic-embed-text:latest".into(),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut session = match bootstrap(config, log_level).await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("bootstrap failed (Ollama not running?): {}", e);
                return;
            }
        };
        let question = "I have used a 7-item Likert scale in my research. What should I do?";
        match session.cmd_rag_query(question).await {
            Ok(()) => {
                let memory = session.session.turns();
                let answer = memory
                    .last()
                    .filter(|t| t.role == TurnRole::Assistant)
                    .map(|t| t.text.as_str())
                    .unwrap_or("");
                let word_count = answer.split_whitespace().count();
                eprintln!("Answer ({} words): {}", word_count, answer);
                assert!(
                    word_count > 20,
                    "Expected >20 words, got {}: '{}'",
                    word_count,
                    answer
                );
            }
            Err(e) => {
                eprintln!("RAG query failed: {}", e);
                // Don't panic on API errors, but do report them.
                // The test still fails if we get here because the
                // assertion above never runs.
                panic!("cmd_rag_query returned error: {}", e);
            }
        }
    }
}
