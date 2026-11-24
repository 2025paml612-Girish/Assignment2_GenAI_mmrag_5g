<<<<<<< HEAD
# mmrag-5g - Multimodal Document Retrieval System (PCAM ZG516, Assignment 2, Part A)

A retrieval-augmented question-answering system over **PDFs and PowerPoint decks** from the 5G NR / O-RAN
domain. It extracts meaning from **text, tables and images** (OCR + layout detection + native chart data),
answers with **grounded citations (file + page + modality)**, and is **guarded, measurable and self-improving**.

Full instructions: `docs/1_Run_Guide.pdf` | architecture: `docs/2_Architecture.pdf` | code walkthrough: `docs/3_Code_Walkthrough.pdf`

## Quick start (Windows / Linux / macOS)

```bash
# 1. install uv (once)          Windows PowerShell:  irm https://astral.sh/uv/install.ps1 | iex
#                               Linux / macOS:       curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. install everything (Python 3.11 is downloaded by uv automatically)
uv sync --extra all
# 3. (recommended) local open-weight LLM
ollama pull qwen2.5:3b-instruct
# 4. check the environment
uv run mmrag doctor
# 5. everything end to end: corpus -> ingestion -> index -> E1..E10 -> outputs/REPORT_TABLES.md
uv run mmrag run-all
```

Then try it:

```bash
uv run mmrag query "What PCI is assigned to cell C955-2?"
uv run mmrag chat                                   # multi-turn with memory
uv run streamlit run app/streamlit_app.py           # demo UI
```

## Using your group's real corpus

1. Put >= 100 public / sanitised PDFs and PPT/PPTX files in `data/corpus/raw/` (sub-folders are fine).
2. `uv run mmrag ingest --force`
3. `uv run mmrag golden-template` -> fill `outputs/golden_template.csv` (>= 30 questions, text/table/image)
   -> `uv run mmrag golden-from-csv outputs/golden_template.csv`
4. Write `data/golden/multiturn_dialog.json` (3 follow-up turns) and keep `guardrail_tests.jsonl`.
5. `uv run mmrag experiments`

Without your own documents, `run-all` generates a synthetic but realistic 120-document corpus
(71 PDF + 49 PPTX: test reports, configuration guides, KPI analyses, release notes, RIC architecture notes,
training decks, weekly KPI reviews) whose image facts exist **only** inside figures/charts.

## Graceful fallbacks (the pipeline always runs)

| Component  | Preferred                               | Fallback used automatically                  |
|------------|-----------------------------------------|----------------------------------------------|
| LLM        | Ollama (qwen2.5) / OpenAI-compatible    | deterministic extractive answerer            |
| Embeddings | BAAI/bge-small-en-v1.5 (sentence-transformers) | TF-IDF + LSA                          |
| Re-ranker  | cross-encoder ms-marco-MiniLM-L-6-v2    | off                                          |
| Vector DB  | ChromaDB (persistent)                   | exact NumPy cosine store                     |
| OCR        | RapidOCR (pip only)                     | Tesseract -> captions only                   |
| Layout     | heuristic (raster + vector regions)     | YOLO doc-layout if weights configured        |
| Tokenizer  | tiktoken cl100k_base                    | regex approximation                          |

All settings live in `config/config.yaml`; the metadata ontology is `config/ontology.yaml`.
=======
# Assignment2_GenAI_mmrag_5g
Multimodal Document Retrieval System for 5G Network
>>>>>>> a566b3397609c659f16b42dd316cdc583bd6cde8
