# Ragrig — Interactive RAG Terminal

![ragrig logo](https://raw.githubusercontent.com/schmettow/ragrig/main/docs/assets/ragrig_logo.png)

A terminal-based Retrieval-Augmented Generation REPL built on the
[`ragrig`](https://crates.io/crates/ragrig) library.  Index your document
collection once, then chat with it — PDF, EPUB, DOCX, and HTML supported.

**Designed for students.**  
  Easy to install. Default build compiles with zero external dependencies. Install Ollama, run `cargo install ragrig-cli`, and call `ragrig-cli --folder mypaper/literature`.

> This is the **binary crate** providing a RAG chat console.  If you want to roll your own RAG application
> see [`ragrig`](https://crates.io/crates/ragrig).

- **Zero extra dependencies** — default build is pure Rust; Ollama provides
  models at runtime
- **Deepseek** support, if you need some heavy lifting
- **Paper search** — built-in arXiv and Semantic Scholar integration with
  one-command PDF download
- **Session persistence** — auto-save conversations, reload past sessions
- **Profile management** — save/load complete configs as JSON for different
  project contexts
- **Cross-platform** — Linux, macOS, WSL, and Windows (MSVC / MinGW)
- - **Hot-swappable** — switch chat, memory, or embedding engines mid-session
  without losing document index or conversation context
- **Hybrid retrieval** — BM25 + cosine vector similarity with Reciprocal
  Rank Fusion.  Hot-swap ranking algorithms (RRF, weighted, MMR, LLM re-rank)

---

## Quick Start

### You need three things

1. **Rust** — [rustup.rs](https://rustup.rs)
2. **Ollama** — [ollama.com/download](https://ollama.com/download)
3. **Three models** (run these once):

```bash
ollama pull qwen3.5:9b             # chat
ollama pull nomic-embed-text        # embeddings
ollama pull qwen2.5:1.5b           # memory / query-rewriting
```

### Install

```bash
cargo install ragrig-cli
```

This downloads and compiles the latest release from
[crates.io](https://crates.io/crates/ragrig-cli).  The binary ends up in
`~/.cargo/bin/ragrig-cli` — make sure that directory is on your `PATH`.

### Or get it from Github

ragrig-cli builds against the `ragrig` library, which must sit next to it:

```bash
git clone https://github.com/schmettow/ragrig
cd ragrig
cargo build --release           # optional — validates the library first
cd ..
git clone https://github.com/schmettow/ragrig-cli
cd ragrig-cli
cargo build --release
./target/release/ragrig-cli --folder ~/Documents/papers
```

### Index and query

```bash
ragrig-cli --folder ~/Documents/papers
```

First launch indexes all PDFs, EPUBs, DOCXs, and HTMLs in the folder.
Subsequent launches are instant — only changed files are re-indexed.

```
Query > What are the key findings about forced-choice paradigms?
```

### Demo mode

Try ragrig against a real book without any of your own documents:

```bash
ragrig-cli --demo
```

Demo mode pins the small `llama3.2:3b` chat model with a 4096-token context
(comfortably fits an 8 GB GPU), disables memory, and indexes the embedded
HTML fixture book — Martin Schmettow, *New Statistics for Design
Researchers. A Bayesian workflow in tidy R.*
(https://schmettow.github.io/New_Stats/) — then prefills the first question
("What is Bayesian Statistics?").  It requires the Ollama models
`nomic-embed-text:latest` and `llama3.2:3b`.

### Hybrid Search Tuning

The vector store uses **Reciprocal Rank Fusion** (RRF, k=60) by default to combine
cosine vector similarity with BM25 full-text search.  Two parameters
control retrieval quality:

| Parameter | Default | What it does |
|---|---|---|
| `top_k` | 50 | Maximum chunks injected into the prompt |
| `similarity_threshold` | 0.04 | Cosine pre‑filter — chunks with cosine < threshold are excluded from RRF fusion |

**Understanding the threshold**:

- The threshold operates on **cosine similarity** (range: 0.0–1.0).
- RRF fusion produces scores in the **0.0–0.03** range (rank‑based, not
  similarity‑based).  The trace output shows RRF scores, not cosine scores.
- A threshold of `0.0` passes everything; `0.04` filters out chunks with
  negligible vector overlap while letting BM25 keyword matches through.
- Values above ~0.05 will aggressively prune — use when you have
  high‑quality embeddings and want strictly semantic results.

Tune at runtime:

```
Query > /search                      # show current values
Query > /search topk 10              # fewer chunks, tighter context
Query > /search threshold 0.08       # stricter semantic filter
```

### Streaming, cancellation, and progress

- Answers **stream token-by-token**; the info header reports the received
  token count (`--- 5 chunks | 132 tokens from [a.pdf] in 2.1s ---`).
- Press **ESC** while an answer is generating (or while documents are being
  indexed) to cancel the operation; the terminal is restored afterwards.
- Indexing (bootstrap, `/embed index`, `/corpus <name> on`) renders a live
  progress bar with files processed, chunks embedded, and failures:
  `[####----] 7/12 files | 341 chunks | papers/a.pdf (ESC: cancel)`.

### Hot-Swap Examples

**Start with everything local, switch chat to cloud mid-session:**

```
Query > /chat deepseek deepseek-chat sk-...
Chat agent swapped: Ollama (qwen3.5:9b) → DeepSeek (deepseek-chat)
```

**Forgetful mode — ask Alice's name, then make her forget:**

```
Query > My name is Alice
Assistant > Nice to meet you, Alice!

Query > /memory off
Memory disabled (was: Ollama qwen2.5:1.5b)

Query > What's my name?
Assistant > I don't know — you haven't told me yet.
```

**Raw transcript — no query rewriting, test context-window pressure:**

```
Query > /memory transcript
Memory strategy: rewrite → transcript

Query > What is a vector database?
Assistant > A vector database stores embeddings ...

Query > Can you summarize that?
# "that" is NOT rewritten — the raw transcript in the prompt
# provides context.  Good for testing how models handle growing
# context windows with full conversation memory appended.
```

**Session persistence — exit, restart, and recall past context:**

```
Query > What are random effects in meta-analysis?
Assistant > Random effects models assume that the true effect size
varies across studies, as opposed to a single fixed effect …

Query > /exit
# next day …

$ ragrig --folder ~/papers
Session: 1718400000

Query > /memory log
History diffusion: off → log

Query > What was I asking about yesterday?
# The chat prompt now includes the raw transcript of the previous
# session, so the model can pick up the thread without you
# repeating yourself.
Assistant > Yesterday you asked about random effects in
meta-analysis.  We discussed how they differ from fixed-effect
models …
```

**One shot questions — no document search, no memory, cloud-only:**

```
Query > /embed none
Query > /memory off
Query > /chat deepseek deepseek-v4-pro
Query > Explain quantum entanglement in one paragraph.
```

**Switch embeddings to CPU-only (no network):**

```
Query > /embed fastembed
Embedder swapped: Ollama (nomic-embed-text) → Fastembed (Nomic-Embed-Text-v1.5)
```

**Experiment with ranking algorithms — same index, different retrieval:**

```
Query > /search rank Cosine
Ranker set to Cosine.

Query > /search rank BM25
Ranker set to BM25.

Query > /search rank Weighted alpha 0.7
Ranker set to Weighted.

Query > /search rank MMR lambda 0.7 inner Cosine
Ranker set to MMR.

Query > /search rank LLM inner Cosine model qwen2.5:0.5b
LLM reranker using Ollama (qwen2.5:0.5b)
Ranker set to LLM.
```

**Swap the chunking strategy — pipeline-aware hot-swap:**

```
Query > /chunker
Chunker: markdown
Available: markdown, token, chunkedrs-recursive, chunkedrs-markdown, ...

Query > /chunker chunkedrs-markdown
Chunker: markdown → chunkedrs-markdown
Warning: no chunks indexed for pdf (unpdf), md (markdown) with
chunker=chunkedrs-markdown, embedder=Ollama/nomic-embed-text:latest.
Run '/embed index' to build the index.

Query > /embed index      # nothing is embedded automatically — the user decides
...
```

Every stored chunk records which (parser, chunker, embedder) pipeline built
it.  When the chunker, parser, or embedder changes, the REPL checks whether
the resulting pipeline already exists in the database and warns if it does
not — but never re-embeds on its own.  Adding documents (`/download`, `/get`)
under a pipeline that has not been indexed yet is an error.



---

## Commands

| Command | Action |
|---|---|
| Any text | RAG query against your document pool |
| `/attach <file>` | Attach a document for the next query (one-shot, not indexed) |
| `/attach` | Show currently attached files |
| `/attach clear` | Clear all attachments |
| `/download <url>` | Download and ingest a document by URL |
| `/get 1,2,3-4,8` | Bulk-download papers from last search results |
| `/scholar <query>` | Search Semantic Scholar |
| `/arxiv <query>` | Search arXiv (no rate limits) |
| `/refs [topic]` | Extract references from last RAG results |
| `/chat <b> [model] [key] \| context <N>` | Hot-swap chat engine, set context window |
| `/embed <b> [model] \| purge \| index \| topk <N> \| threshold <F>` | Hot-swap embedding, clear store, re-index, tune search |
| `/memory <b> [model] [key] \| transcript \| log \| summary \| off \| purge` | Hot-swap memory, history diffusion, or clear |
| `/prompt chat\|rewrite <file> \| reset` | Load custom system prompts |
| `/parser pdf unpdf\|sink\|extract\|internal\|vision \| epub epub` | Hot-swap document parser per format |
| `/chunker [name]` | Show or hot-swap the chunking strategy (warns when the pipeline is not indexed) |
| `/log [off\|error\|warn\|info\|debug\|trace]` | Show or change log verbosity at runtime |
| `/profile save\|show\|load\|list [name]` | Manage configuration profiles |
| `/corpus <name> on\|off \| dyn on\|off` | Toggle a named corpus (index/remove its chunks) or dynamic web-download routing |
| `/search topk <N> \| threshold <F> \| rank <name> \| by <file>` | Tune retrieval or search by document |
| `/hist [list \| load <id> \| delete <id>]` | Manage saved sessions |
| `/help` | Show available commands |
| `exit` / `quit` | End session |

### Model parameters

Fine-tune generation at startup or mid-session:

```bash
# From the command line:
ragrig-cli --folder ~/Documents/papers --temperature 0.1 --seed 42

# Or hot-swap at runtime from the REPL:
Query > /chat temperature 0.1
Query > /chat seed 42
Query > /chat top_p 0.9
Query > /chat max_tokens 2048
```

### Runtime log levels

By default ragrig-cli prints informational messages (`info` level) — agent
swaps, indexing progress, and errors.  You can change the verbosity at any
time without restarting:

```text
/log           # show current level
/log debug     # enable diagnostic output from the ragrig crate only
/log trace     # enable full pipeline observability (rewrite, chunks, tokens)
/log warn      # back to quiet operation
```

At `trace` level the REPL logs every pipeline stage:

```text
[TRACE] Query: "What is RAG?"
[TRACE] Rewrite: "What is RAG?" → "retrieval augmented generation definition"
[TRACE] Retrieved 5 chunks (cosine threshold 0.040):
[TRACE]   [0.0164] intro.pdf — "Retrieval-Augmented Generation (RAG) is a..."
[TRACE]   [0.0141] survey.pdf — "RAG systems have become the standard..."
[TRACE] Context budget: 4096 tokens (~12288 chars)  |  Full prompt: 3142 chars (~1047 tokens)
[TRACE] Elapsed: Some(1.234s)
```

The initial level can also be set via the `RUST_LOG` environment variable:

```bash
RUST_LOG=debug ragrig-cli --folder ~/papers
```

### Profile management

Profiles let you save and reload complete configurations — chat model,
embedding backend, chunk size, search tuning, and more — as JSON files.
Use them to switch between project contexts without memorising long CLI
invocations.

**At startup** — load a profile and optionally override individual flags:

```bash
# Save a profile from the REPL first:
ragrig-cli --folder ~/papers --model qwen3.5:9b --chunk-size 512 --top-k 20
> /profile save physics

# Reload it later — chunk-size and top-k come from the profile,
# but you can still override on the CLI:
ragrig-cli --folder ~/papers --profile physics --model gemma4:e4b
```

**In the REPL:**

| Command | Action |
|---|---|
| `/profile save <name>` | Serialise current config (including runtime hot-swaps) to `.ragrig/profiles/<name>.json` |
| `/profile show [name]`  | Pretty-print a profile as JSON.  `show current` prints the live running state. |
| `/profile load <name>`   | Load a profile into memory without restarting agents — use `/chat`, `/embed`, `/memory` afterwards to apply it. |
| `/profile list`          | List all saved profile names. |

Profiles are stored under `<workspace>/.ragrig/profiles/`.  The JSON is
hand-editable if you prefer typing values over REPL commands.

---

## CLI Flags

```
Usage: ragrig-cli [OPTIONS]

Options:
      --workspace <DIR>             State directory for store, history, sessions, profiles [default: .]
  -f, --folder <DIR>                Shortcut for --workspace <DIR> --corpus-dir folder=<DIR>
      --corpus-dir <NAME=DIR>       Named document corpus: a directory.  Repeatable.
      --corpus-urls <NAME=URL>      Named document corpus: a URL.  Repeatable; same NAME
                                    accumulates URLs into one curated list.
  -p, --profile <NAME>              Load a saved profile from .ragrig/profiles/ (JSON)
      --provider <PROVIDER>        Chat backend: ollama (default) or deepseek
      --deepseek-api-key <KEY>     DeepSeek API key [env: DEEPSEEK_API_KEY]
      --deepseek-model <MODEL>     DeepSeek model [default: deepseek-v4-pro]
  -m, --model <MODEL>              Ollama chat model [default: qwen3.5:9b]
      --embedding-provider <P>     Embedding: ollama (default) or fastembed
  -e, --embedding-model <MODEL>    Ollama embedding model [default: nomic-embed-text]
      --memory-model <MODEL>       Memory/rewrite model [default: qwen2.5:1.5b]
      --prompt-chat <FILE>         Custom system prompt for chat agent
      --prompt-rewrite <FILE>      Custom system prompt for rewrite agent
      --pdf-parser <BACKEND>       PDF parser: unpdf (default), sink, extract, internal
      --sloppy-pdf                 Enable sloppy binary text extraction as fallback
      --chunk-size <TOKENS>        Max tokens per chunk [default: 1024]
      --chunk-overlap <TOKENS>     Overlap between chunks [default: 128]
      --top-k <N>                  Chunks per query [default: 50]
      --similarity-threshold <FL>  Cosine similarity pre‑filter [default: 0.04]
      --context-tokens <N>         Context window budget for prompt truncation [default: 4096]
      --context-size-mode <MODE>   Context overflow handling: auto (default) or forced
      --temperature <F>            Sampling temperature
      --top-p <F>                  Nucleus sampling top-p
      --max-tokens <N>             Max output tokens
      --seed <N>                   Random seed for reproducibility
      --semantic-scholar-api-key <K>  API key [env: SEMANTIC_SCHOLAR_API_KEY]
```

---

## Q & A

### What is unique about ragrig and why should I use it?

Ragrig tries to be a flexible and zero-friction prototyping tool for
researchers and students, not an enterprise-grade framework with all bells
and whistles. Here are the points that distinguish Ragrig from other crates:

1. **Zero native dependencies in default build.** Every other crate needs
   at minimum a C compiler (for tokenizers, ONNX runtime, tree-sitter, etc.)
   or an API key. Ragrig builds with `cargo build --release` and nothing else.
   This is a **genuinely unique** selling point for students, workshops, and
   quick-start scenarios.

2. **Runtime hot-swapping via trait objects.** Every other crate uses
   compile-time feature flags to select backends. Ragrig lets you switch
   chat/embed/memory engines *mid-session* without losing state.
   `/chat deepseek`, `/embed fastembed`, `/memory off` commands have no
   equivalent in any competitor.

3. **Panic-safe multi-parser PDF pipeline.** Multiple PDF parsers with
   `catch_unwind` wrapping. No other crate does this — they pick one parser
   and crash on malformed PDFs.

4. **Token-efficient cloud usage pattern.** Use a tiny local model for query
   rewriting, only send the final prompt + context to the cloud.

5. **Student-focused UX.** The quick-start is 3 commands. The REPL has 15+
   slash commands. Session persistence works out of the box.

### When should I not use it?

Ragrig is designed as an accessible framework to build multi-agent interactive
prototypes. It is not intended for production use or highly scalable
deployments.

### I am a Python programmer. How can I use Ragrig?

For version 2.0, we plan to provide Python and possibly R bindings.

### Embeddings model not found

If you see *Embedding model nomic-embed-text not found*, although you have pulled it with Ollama,
you are most likely using an older version of Ollama, which needs the full name of the model: *nomic-embed-text:latest*

Update your Ollama installation to the latest version to avoid this issue. 
Or use the full model name in a profile configuration (`-p` or `--profile`).
Or use command line argument (`--embedding_model nomic-embed-text:latest`).

### Ollama is unreachable — what should I check?

If ragrig-cli reports `OllamaUnreachable`, work through these in order:

1. **Ollama isn't running.**  Start it in a terminal:
   ```bash
   ollama serve
   ```
   On macOS and Windows, launching the Ollama desktop app also starts the
   server.

2. **Ollama is running on a non-default port.**  By default ragrig-cli connects
   to `localhost:11434`.  If you changed the port (e.g. via `OLLAMA_HOST`),
   set the same variable:
   ```bash
   export OLLAMA_HOST=127.0.0.1:11435
   ragrig-cli --folder ./my_docs
   ```

3. **A model pull was interrupted.**  Partial downloads can leave the Ollama
   registry in a broken state.  Re-pull the model:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen3.5:9b
   ```
   If that fails, remove the partial model and pull fresh:
   ```bash
   ollama rm nomic-embed-text && ollama pull nomic-embed-text
   ```

4. **Firewall or port conflict.**  Ensure port 11434 is not blocked:
   ```bash
   # Linux / macOS / WSL
   lsof -i :11434
   # Windows (PowerShell)
   netstat -ano | findstr :11434
   ```

5. **WSL → Windows networking.**  When Ollama runs on Windows and ragrig-cli
   runs inside WSL, `localhost` does not automatically forward.  Find the
   Windows host IP from inside WSL and set `OLLAMA_HOST`:
   ```bash
   export OLLAMA_HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):11434
   ```
   Alternatively, install Ollama directly inside WSL.

### I adjusted the context size and now Ragrig produces empty answers, or answers that clearly come from general knowledge, not the indexed corpora.

This happens, when the context size of local Ollama models exceeds the hard VRAM limits. See below for closer explanations.

If you really need a large context size and a powerful model, you have the following option: 

+ switch to a cloud model
+ get a GPU with 24GB+ VRAM
+ get a computer with good amounts of unified memory

### When the context size exceeds the model's maximum, how can I adjust this?

Context-size errors happen for two reasons:

1. **Hardware VRAM limits** — Ollama caps the context window at 4096 tokens
   on GPUs with less than 24 GB VRAM to prevent out-of-memory crashes.
2. **Architectural limits** — some distilled reasoning models (e.g. DeepSeek
   R1 8B/14B) have a hard-coded 4096-token maximum.

Ragrig detects context overflows automatically.  By default, when the model
reports a `ContextSizeExceeded` error, the binary auto-adjusts its budget to
the model's actual maximum, rebuilds the prompt with fewer chunks, and retries
once.  You see:

```
[INFO] Context overflow — shrinking budget to 9216 chars, retrying.
```

If the retry also fails, pass `--context-size-mode forced` to keep the
original error path, then set a manual budget:

```bash
ragrig-cli --folder ~/papers --context-tokens 4096 --context-size-mode forced
# or mid-session:
Query > /chat context 4096
```

---

## License

MIT License — see [LICENSE](https://github.com/schmettow/ragrig/blob/main/LICENSE).
