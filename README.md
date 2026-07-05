<p align="center"><img src="assets/coelho_genai_logo.png" alt="COELHO GenAI" width="220"></p>

<p align="center"><strong>A privacy-first Generative AI platform — one LLM substrate, five live tools, your choice of local or hosted inference.</strong></p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="https://streamlit.io/"><img alt="Streamlit" src="https://img.shields.io/badge/streamlit-app-FF4B4B?logo=streamlit&logoColor=white"></a>
  <a href="https://python.langchain.com/"><img alt="LangChain" src="https://img.shields.io/badge/LangChain-agents%20%2B%20tools-1C3C3C?logo=langchain&logoColor=white"></a>
  <a href="https://qdrant.tech/"><img alt="Qdrant" src="https://img.shields.io/badge/Qdrant-vector%20search-DC244C?logo=qdrant&logoColor=white"></a>
  <a href="https://ds4sd.github.io/docling/"><img alt="Docling" src="https://img.shields.io/badge/Docling-document%20parsing-0F62FE"></a>
  <a href="https://go.dev/"><img alt="Go" src="https://img.shields.io/badge/go-1.23-00ADD8?logo=go&logoColor=white"></a>
  <img alt="Providers" src="https://img.shields.io/badge/LLM%20providers-5-success">
</p>

<p align="center">
  <a href="https://rafaelcoelho.pages.dev/work/coelho-genai">Portfolio Page</a> ·
  <a href="./COELHOGenAI.pdf">PDF Presentation</a>
</p>

---

## Table of Contents

- [What is this?](#what-is-this)
- [The five tools](#the-five-tools)
- [How it's built](#how-its-built)
- [Tech Stack](#tech-stack)
- [Key Components](#key-components)
- [Shipped but not wired in](#shipped-but-not-wired-in)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Author](#author)

---

## What is this?

COELHO GenAI connects a single Streamlit interface to open-source and hosted Large Language Models — picking where your tokens go (local on Ollama, or hosted on Groq / SambaNova / Scaleway / OpenAI) is a first-class setting, not an afterthought. Five tools sit on top of that shared substrate: a conversational assistant, an online-tool-calling research agent, an autonomous data-science agent, a document-RAG assistant, and a plan-then-execute agent.

The design principle is **LLM choice + privacy**: every tool works identically regardless of which provider is active, so switching from a fully local Ollama model to a hosted one is a settings change, not a rewrite. Five providers means five completely different cost/latency/privacy trade-offs available from the same UI.

## The five tools

| Tool | What it does | Notable detail |
|---|---|---|
| **Assistant** | General-purpose conversational chatbot with persistent memory | A `ConversationChain` over `ConversationBufferMemory` — every tool in this app shares the same chat history object, so switching tools mid-session doesn't lose context |
| **Information Retrieval** | Grounds the LLM in a live external tool, one at a time | 9 selectable tools: **Arxiv, DuckDuckGo, PubMed, StackExchange, Wikidata, Wikipedia, Yahoo Finance News, Shell, Python REPL** — a `ZERO_SHOT_REACT_DESCRIPTION` agent reasons about when to call the selected tool |
| **Data Science** | Autonomous pandas agent over an uploaded CSV/XLSX | LangChain's `create_pandas_dataframe_agent` with `allow_dangerous_code=True` — the agent writes and executes real pandas code against your data, not canned analysis |
| **Document Assistant** | RAG over a file, a URL, or virtually any LangChain document source | Two independent loader paths (see [Key Components](#key-components)) — Docling for structure-aware extraction, or any of LangChain's `document_loaders` picked dynamically at runtime |
| **Plan & Solve** | Decomposes a request into a plan, then executes it step by step | LangChain's `plan_and_execute` pattern — a chat planner drafts the steps, a separate agent executor (armed with Wikipedia search) carries each one out |

## How it's built

Streamlit's native multi-page navigation (`st.navigation`) wires each tool into its own page under `applications/`, all sharing one `app.py` shell that owns:

- **The provider/model switch** — a `Settings` dialog lets you pick a framework (Groq / Ollama / SambaNova / Scaleway / OpenAI) and a specific model, with API keys autofilled from the environment when present. Ollama's model list is queried live from the local `ollama` daemon (`ollama.list()`), including which model is currently loaded in memory (`ollama.ps()`).
- **Shared memory** — one `ConversationBufferMemory` and one `StreamlitChatMessageHistory` instance, initialized once and reused across every tool, plus a `MemorySaver` for LangGraph-based flows.
- **A common per-provider LLM factory** — every tool's model-loading code resolves the active framework to the right LangChain chat model class (`ChatGroq`, `ChatOllama`, `ChatSambaNovaCloud`, `ChatOpenAI` for both Scaleway and OpenAI proper), so adding a tool never means re-solving "which provider is this."

## Tech Stack

### LLM providers & models

| Provider | Needs a key? | Representative models |
|---|---|---|
| **Groq** | Yes | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`, `qwen-2.5-32b`, `deepseek-r1-distill-llama-70b`, `gemma2-9b-it`, +8 more |
| **Ollama** | No (local) | Whatever's installed locally — model list and active/loaded state queried live from the Ollama daemon |
| **SambaNova** | Yes | `Meta-Llama-3.1-405B-Instruct`, `DeepSeek-R1`, `Qwen2.5-72B-Instruct`, `QwQ-32B-Preview`, +7 more |
| **Scaleway** | Yes | `llama-3.3-70b-instruct`, `deepseek-r1`, `mistral-nemo-instruct-2407`, `pixtral-12b-2409`, `qwen2.5-coder-32b-instruct`, +4 more |
| **OpenAI** | Yes | `gpt-4o`, `gpt-4o-mini`, `o1`, `o1-mini`, `o3-mini`, `o1-preview` |

Five providers, six model families (Llama, Gemma, Mixtral/Mistral, Qwen, DeepSeek, GPT/o-series) — swapping between a fully local model and a frontier hosted one is one dropdown, not a code change.

### Core libraries

| Technology | Purpose |
|---|---|
| **Streamlit** | Multi-page app shell, chat UI, file uploads |
| **LangChain / LangChain Community / Experimental** | Agents, tools, document loaders, memory |
| **LangGraph** | Checkpointed memory (`MemorySaver`) for graph-based flows |
| **Qdrant** (`qdrant-client`, `langchain-qdrant`) | Vector store backing Document Assistant's RAG mode |
| **Docling** | Structure-aware document parsing — OCR, table-cell matching, layout-aware markdown export |
| **pandas / openpyxl** | Data Science tool's dataframe ingestion (CSV/XLSX) |

## Key Components

### Document Assistant's dual ingestion path

- **Docling loader** — accepts a file (PDF, image, DOCX, HTML, PPTX, AsciiDoc, Markdown) or a URL, runs OCR and table-structure detection with cell matching, and can persist the parsed result as Markdown (with embedded or referenced images), JSON, HTML, and per-table CSVs.
- **LangChain loader** — dynamically lists *every* loader in `langchain_community.document_loaders.__all__` (Wikipedia, YouTube, web pages, dozens more) via `inspect.signature`, builds a form from that loader's actual constructor arguments, and instantiates it at runtime. If the chosen loader needs a package that isn't installed, the app catches the `ImportError`, `uv pip install`s the missing dependency on the spot, and reruns — no manual dependency wrangling.
- Either path can be embedded into Qdrant (per-provider embeddings: `OllamaEmbeddings`, `HuggingFaceEmbeddings` for Groq/SambaNova, `OpenAIEmbeddings` for Scaleway/OpenAI) and retrieved with `similarity_search` on every chat turn, or used directly as unembedded context.

### Information Retrieval's tool roster

Nine tools are live behind a single-select dropdown — Arxiv, DuckDuckGo, PubMed, StackExchange, Wikidata, Wikipedia, Yahoo Finance News, Shell, and a Python REPL — each wrapped as a single LangChain `Tool` and handed to a `ZERO_SHOT_REACT_DESCRIPTION` agent. Two more (SearxNG, YouTube Search) are implemented in code but commented out — SearxNG needs a local instance running, YouTube Search was disabled without an in-code reason given.

### The LangChain Hub prompt scraper (Go + Python)

Two independent implementations of the same job — paginate `smith.langchain.com/hub`, scrape every prompt's name and description, save to `prompts.json` — feeding the Prompt Engineering tool's "which prompt name do I actually type?" problem. The Python version (`langchain_hub.py`) is a straightforward sequential `asyncio` + Playwright scraper; the Go version (`langchain_hub.go`) does the same job concurrently, capping parallelism at 25 goroutines with a semaphore. Same result, one written for simplicity and one for throughput.

## Shipped but not wired in

Two tools have complete implementations in `functions.py` and their own page under `applications/`, but are commented out of `app.py`'s navigation — the code runs if re-enabled, but neither ships in the default experience:

- **Prompt Engineering** — pulls any named prompt from LangChain Hub (`hub.pull(...)`), introspects its system/human input variables, and builds a form to fill them in before running it through Ollama specifically (this tool doesn't follow the global provider switch).
- **Software Development** — a `ZERO_SHOT_REACT_DESCRIPTION` agent armed with `ShellTool` and `PythonREPLTool`, i.e. autonomous shell + Python execution. Kept out of the default navigation given the obvious blast radius of unrestricted code execution.

Data Science's `PandasAI` framework option and Information Retrieval's `SearxNG` tool follow the same pattern — implemented, currently disabled by a commented-out line rather than deleted.

## Prerequisites

| Tool | Purpose |
|---|---|
| **Python** 3.12+ | Runtime |
| **[uv](https://docs.astral.sh/uv/getting-started/installation/)** | Python package/venv management |
| **Docker** | Runs Qdrant for Document Assistant's RAG mode |
| **[Ollama](https://ollama.com/download)** | Optional — only needed for fully local inference |
| At least one provider API key | Groq, SambaNova, Scaleway, or OpenAI — required unless running Ollama-only |

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/rafaelcoelho1409/COELHOGenAI
cd COELHOGenAI

# 2. Start Qdrant (required for Document Assistant's RAG mode)
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant

# 3. Set up the environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Ollama models are managed separately — install any model via `ollama pull <model>` ([library](https://ollama.com/library)) or pull one from [Hugging Face](https://huggingface.co/models) before selecting it in-app.

## Configuration

API keys are entered directly in the app's **Settings** dialog (autofilled from the environment if already set as `GROQ_API_KEY`, `SAMBANOVA_API_KEY`, `SCW_GENERATIVE_APIs_ENDPOINT` / `SCW_ACCESS_KEY` / `SCW_SECRET_KEY`, or `OPENAI_API_KEY`) — no `.env` file is required to get started, though setting one saves re-entering keys every session.

## Project Structure

```
COELHOGenAI/
├── app.py                      # Streamlit shell — navigation, settings, shared memory
├── functions.py                 # LLM factory per provider, all tool classes, shared dialogs
├── applications/
│   ├── home.py                   # Landing page
│   ├── assistant.py               # Tool 1 — conversational chat
│   ├── information_retrieval.py   # Tool 2 — single-tool-calling agent
│   ├── data_science.py            # Tool 3 — pandas dataframe agent
│   ├── document_assistant.py      # Tool 4 — Docling / LangChain RAG
│   ├── plan_and_solve.py           # Tool 5 — planner + executor
│   ├── prompt_engineering.py       # Shipped, not wired in — see above
│   └── software_development.py    # Shipped, not wired in — see above
│
├── langchain_hub.py             # LangChain Hub prompt scraper (Python/asyncio)
├── langchain_hub.go              # Same scraper, concurrent (Go)
├── go.mod / go.sum
│
├── assets/                      # Logo + presentation screenshots
├── databases/                   # Per-model local vector store working directories
├── qdrant_storage/               # Qdrant's persisted collections
│
├── COELHOGenAI.pdf               # Deployment record / demo deck
├── requirements.txt
└── style.css                    # Streamlit UI theming
```

## Author

**Rafael Coelho** — [rafaelcoelho.pages.dev](https://rafaelcoelho.pages.dev/)
