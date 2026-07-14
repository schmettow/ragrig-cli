# Ragrig — Interactive RAG Terminal

![ragrig logo](https://raw.githubusercontent.com/schmettow/ragrig/main/docs/assets/ragrig_logo.png)

A terminal-based Retrieval-Augmented Generation REPL built on the
[`ragrig`](https://crates.io/crates/ragrig) library.  Index your document
collection once, then chat with it — PDF, EPUB, DOCX, and HTML supported.

**Designed for students.**  The default build compiles with zero external
dependencies — no C++ toolchain, no `cmake`, no `protoc`.  Install Rust,
install Ollama, run `cargo install ragrig-bin`, and you're done.

> This is the **binary crate**.  If you want the library for building your
> own application, see [`ragrig`](https://crates.io/crates/ragrig).

- **Zero extra dependencies** — default build is pure Rust; Ollama provides
  models at runtime
- **Hot-swappable** — switch chat, memory, or embedding engines mid-session
  without losing document index or conversation context
- **Hybrid retrieval** — BM25 + cosine vector similarity with Reciprocal
  Rank Fusion.  Hot-swap ranking algorithms (RRF, weighted, MMR, LLM re-rank)
- **Paper search** — built-in arXiv and Semantic Scholar integration with
  one-command PDF download
- **Session persistence** — auto-save conversations, reload past sessions
- **Profile management** — save/load complete configs as JSON for different
  project contexts
- **Cross-platform** — Linux, macOS, WSL, and Windows (MSVC / MinGW)

---

## Quick Start

### You need three things

1. **Rust** — [rustup.rs](https://rustup.rs)
2. **Ollama** — [ollama.com/download](https://ollama.com/download)
3. **Three models** (run these once):

```bash
ollama pull gemma2:latest           # chat
ollama pull nomic-embed-text        # embeddings
ollama pull qwen2.5:1.5b           # memory / query-rewriting
```

### Install

```bash
cargo install ragrig-bin
```

This downloads and compiles the latest release from
[crates.io](https://crates.io/crates/ragrig-bin).  The binary ends up in
`~/.cargo/bin/ragrig-bin` — make sure that directory is on your `PATH`.

### Or build from source

```bash
git clone https://github.com/schmettow/ragrig
cd ragrig-bin
cargo build --release
./target/release/ragrig-bin --folder ~/Documents/papers
```

### Index and query

```bash
ragrig-bin --folder ~/Documents/papers
```

First launch indexes all PDFs, EPUBs, DOCXs, and HTMLs in the folder.
Subsequent launches are instant — only changed files are re-indexed.

```
Query > What are the key findings about forced-choice paradigms?
```

> **Students:** if you only have Rust and Ollama installed, you already have
> everything you need.  The default build adds nothing else.

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
| `/log [off\|error\|warn\|info\|debug\|trace]` | Show or change log verbosity at runtime |
| `/profile save\|show\|load\|list [name]` | Manage configuration profiles |
| `/search topk <N> \| threshold <F> \| rank <name> \| by <file>` | Tune retrieval or search by document |
| `/hist [list \| load <id> \| delete <id>]` | Manage saved sessions |
| `/help` | Show available commands |
| `exit` / `quit` | End session |

### Model parameters

Fine-tune generation at startup or mid-session:

```bash
# From the command line:
ragrig-bin --folder ~/Documents/papers --temperature 0.1 --seed 42

# Or hot-swap at runtime from the REPL:
Query > /chat temperature 0.1
Query > /chat seed 42
Query > /chat top_p 0.9
Query > /chat max_tokens 2048
```

### Runtime log levels

By default ragrig-bin prints informational messages (`info` level) — agent
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
[TRACE] Context budget: 8192 tokens (~24576 chars)  |  Full prompt: 3142 chars (~1047 tokens)
[TRACE] Elapsed: Some(1.234s)
```

The initial level can also be set via the `RUST_LOG` environment variable:

```bash
RUST_LOG=debug ragrig-bin --folder ~/papers
```

### Profile management

Profiles let you save and reload complete configurations — chat model,
embedding backend, chunk size, search tuning, and more — as JSON files.
Use them to switch between project contexts without memorising long CLI
invocations.

**At startup** — load a profile and optionally override individual flags:

```bash
# Save a profile from the REPL first:
ragrig-bin --folder ~/papers --model gemma2:latest --chunk-size 512 --top-k 20
> /profile save physics

# Reload it later — chunk-size and top-k come from the profile,
# but you can still override on the CLI:
ragrig-bin --folder ~/papers --profile physics --model gemma4:e4b
```

**In the REPL:**

| Command | Action |
|---|---|
| `/profile save <name>` | Serialise current config (including runtime hot-swaps) to `.ragrig/profiles/<name>.json` |
| `/profile show [name]`  | Pretty-print a profile as JSON.  `show current` prints the live running state. |
| `/profile load <name>`   | Load a profile into memory without restarting agents — use `/chat`, `/embed`, `/memory` afterwards to apply it. |
| `/profile list`          | List all saved profile names. |

Profiles are stored under `<folder>/.ragrig/profiles/`.  The JSON is
hand-editable if you prefer typing values over REPL commands.

---

## CLI Flags

```
Usage: ragrig-bin --folder <FOLDER>

Options:
  -f, --folder <FOLDER>            Document directory (PDFs, EPUBs, DOCXs, HTMLs)
  -p, --profile <NAME>             Load a saved profile from .ragrig/profiles/ (JSON)
      --provider <PROVIDER>        Chat backend: ollama (default) or deepseek
      --deepseek-api-key <KEY>     DeepSeek API key [env: DEEPSEEK_API_KEY]
      --deepseek-model <MODEL>     DeepSeek model [default: deepseek-v4-pro]
  -m, --model <MODEL>              Ollama chat model [default: gemma2:latest]
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
      --context-tokens <N>         Context window budget for prompt truncation [default: 8192]
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

If ragrig-bin reports `OllamaUnreachable`, work through these in order:

1. **Ollama isn't running.**  Start it in a terminal:
   ```bash
   ollama serve
   ```
   On macOS and Windows, launching the Ollama desktop app also starts the
   server.

2. **Ollama is running on a non-default port.**  By default ragrig-bin connects
   to `localhost:11434`.  If you changed the port (e.g. via `OLLAMA_HOST`),
   set the same variable:
   ```bash
   export OLLAMA_HOST=127.0.0.1:11435
   ragrig-bin --folder ./my_docs
   ```

3. **A model pull was interrupted.**  Partial downloads can leave the Ollama
   registry in a broken state.  Re-pull the model:
   ```bash
   ollama pull nomic-embed-text
   ollama pull gemma2:latest
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

5. **WSL → Windows networking.**  When Ollama runs on Windows and ragrig-bin
   runs inside WSL, `localhost` does not automatically forward.  Find the
   Windows host IP from inside WSL and set `OLLAMA_HOST`:
   ```bash
   export OLLAMA_HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):11434
   ```
   Alternatively, install Ollama directly inside WSL.

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
ragrig-bin --folder ~/papers --context-tokens 4096 --context-size-mode forced
# or mid-session:
Query > /chat context 4096
```

---

## License

MIT License — see [LICENSE](https://github.com/schmettow/ragrig/blob/main/LICENSE).
