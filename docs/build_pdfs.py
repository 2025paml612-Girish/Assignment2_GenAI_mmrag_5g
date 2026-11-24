"""Build the three project PDFs from the code and the latest results in outputs/.

    uv run python docs/diagrams.py      # diagrams -> docs/img
    uv run python docs/build_pdfs.py    # PDFs     -> docs/
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdfkit import (OUT, ROOT, H1, H2, H3, Doc, KeepTogether, NextPageTemplate, P, PageBreak, S, Spacer,  # noqa: E402
                    bullets, cm, code, df_table, image, load_csv, load_json, note, snippet, table, toc)

W = 17.4 * cm           # portrait frame width
LW, LH = 26.6 * cm, 17.2 * cm


def cover(title: str, subtitle: str, items: list[str]):
    st = load_json("e1_ingestion_stats.json") or {}
    return [Spacer(1, 3.2 * cm), P(title, "title"), Spacer(1, 0.3 * cm), P(subtitle, "subtitle"), Spacer(1, 1.2 * cm),
            P("PCAM ZG516 - Applied Generative AI  |  Group Major Project / Assignment 2  |  Part A (12 marks)", "subtitle"),
            Spacer(1, 0.5 * cm),
            P("Multimodal Document Retrieval System over a 5G NR / O-RAN corpus of PDFs and PowerPoint decks", "subtitle"),
            Spacer(1, 1.5 * cm),
            table([["This document covers"]] + [[i] for i in items], [W], 9),
            Spacer(1, 1.2 * cm),
            P(f"Reference run: {st.get('documents_parsed', '-')} documents, {st.get('pages_or_slides', '-')} pages/slides, "
              f"{st.get('tables', '-')} tables, {st.get('images', '-')} images, {st.get('chunks_total', '-')} chunks.",
              "subtitle"),
            PageBreak(), P("Contents", "h1"), toc(), PageBreak()]


def ref_run_note():
    return note("<b>About the numbers in this document.</b> They come from a reference run of the unmodified project "
                "in a sandbox with <b>no internet model downloads and no local LLM</b>, so the automatic fallbacks were "
                "active: TF-IDF/LSA embeddings instead of bge-small, the deterministic <i>extractive</i> answerer instead "
                "of an LLM, a regex token counter instead of tiktoken, and no re-ranker. On your machine, with "
                "<font face='Courier'>uv sync --extra all</font> and Ollama running, the same commands use the neural "
                "components and the tables are regenerated with real LLM numbers (latency will be seconds, not "
                "milliseconds). Always report the numbers from your own run.")


# =============================================================================================
# 1. RUN GUIDE
# =============================================================================================
def run_guide(path: Path):
    doc = Doc(path, "1 - How to run, start to end")
    f = cover("How to Run - Start to End", "Installation, configuration, every command, outputs and troubleshooting",
              ["1. What the project delivers and how it maps to the assignment",
               "2. Prerequisites and installation with uv (Windows / Linux / macOS)",
               "3. Choosing the LLM (Ollama, OpenAI-compatible endpoint, or offline fallback)",
               "4. Step-by-step run: corpus, ingestion, query, chat, UI, experiments E1-E10",
               "5. Switching to your group's real corpus and hand-curated golden set",
               "6. Outputs map, configuration reference, runtimes, troubleshooting, cheat sheet"])

    f += H1("1. What you get")
    f += [P("The zip contains a complete Python project, <b>mmrag-5g</b>, that implements Part A of the assignment end to "
            "end: it ingests at least 100 PDFs and PPT/PPTX files, extracts meaning from <b>text, tables and images</b> "
            "(OCR, layout detection, native chart data), indexes everything in a vector store with BM25, answers questions "
            "with <b>citations to file, page and modality</b>, protects itself with DoS and input/output guardrails, keeps "
            "conversation memory, and runs all ten required experiments with one command, writing every table you need "
            "for the report to <font face='Courier'>outputs/</font>."),
          table([["Assignment item", "Where it is produced"],
                 ["FR1 >= 100 PDFs/PPTs ingested", "mmrag ingest -> outputs/e1_corpus_inventory.csv, e1_ingestion_stats.json"],
                 ["FR2 text, tables, images searchable", "outputs/e3_modality_examples.json, e3_retrieval_design.csv"],
                 ["FR3 P95 latency < 90 s", "outputs/e6_nonfunctional.json (latency block)"],
                 ["FR4 DoS + guardrails < 6 s", "outputs/e6_guardrail_overhead.csv, e6_nonfunctional.json"],
                 ["FR5 self-improving prompts", "outputs/e8_selfimprove_log.md / .json, e8_optimized_prompt.json"],
                 ["NFR1 consistency (5 runs)", "outputs/e6_consistency.csv"],
                 ["NFR2 token optimisation vs naive", "outputs/e6_token_comparison.csv, e10_benchmark.csv"],
                 ["A1 preprocessing record", "data/work/preprocessing_record.jsonl, elements.jsonl"],
                 ["A2 vector store / memory / retrieval", "data/index/*, outputs/e2_*.csv, e3_*.csv, e4_memory_transcript.md"],
                 ["A3 HLD / LLD", "docs/2_Architecture.pdf (this zip), docs/img/*.png"],
                 ["A4 golden set, auto eval, failures", "data/golden/golden_set.jsonl, outputs/e5_*"],
                 ["A5 ablation, optimiser, guardrails, ontology", "outputs/e7_*, e8_*, e9_*, config/ontology.yaml"],
                 ["E10 integrated system / demo", "mmrag chat, Streamlit UI, notebooks/PartA_end_to_end.ipynb, e10_*"],
                 ["All report tables in one file", "outputs/REPORT_TABLES.md"]], [6.2 * cm, 11.2 * cm], 8)]

    f += H1("2. Prerequisites and installation")
    f += H2("2.1 Machine")
    f += bullets(["Windows 10/11, Linux or macOS; 8 GB RAM minimum (16 GB recommended with a local LLM); ~3 GB free disk "
                  "(Python packages + embedding models + an Ollama model).",
                  "Internet for the first install (Python packages, the bge-small embedding model ~130 MB, the cross-encoder "
                  "~90 MB, the tiktoken vocabulary). Afterwards everything runs offline.",
                  "Optional: <b>LibreOffice</b> (only to convert legacy <font face='Courier'>.ppt</font> files), "
                  "<b>Tesseract</b> (only if RapidOCR cannot be installed), a GPU (not required)."])
    f += H2("2.2 Install uv")
    f += [P("uv installs Python itself (3.11, pinned in <font face='Courier'>.python-version</font>) and all packages in "
            "an isolated <font face='Courier'>.venv</font> inside the project."),
          code("""
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
# check (open a new terminal first)
uv --version
""")]
    f += H2("2.3 Unpack and install the project")
    f += [code("""
# unzip mmrag_5g.zip, e.g. to D:\\Projects\\mmrag_5g  (Windows) or ~/mmrag_5g (Linux/macOS)
cd mmrag_5g
uv sync --extra all          # core + sentence-transformers + rapidocr + chromadb + streamlit
# optional extras:
uv sync --extra all --extra notebook     # Jupyter for notebooks/PartA_end_to_end.ipynb
uv sync --extra all --extra yolo         # ultralytics, for YOLO layout detection
uv sync --group dev                      # pytest
"""),
          table([["Extra", "Packages", "What you lose without it (automatic fallback)"],
                 ["(core)", "pymupdf, python-pptx, numpy, pandas, scikit-learn, rank-bm25, jinja2, requests, tiktoken, "
                            "pyyaml, reportlab, matplotlib", "- (always installed)"],
                 ["ml", "sentence-transformers (+ torch CPU)", "bge-small embeddings + cross-encoder re-ranker -> TF-IDF/LSA, no re-rank"],
                 ["ocr", "rapidocr-onnxruntime", "pip-only OCR -> Tesseract if installed, else captions only"],
                 ["chroma", "chromadb", "persistent HNSW vector DB -> exact NumPy store"],
                 ["ui", "streamlit", "web demo UI -> use the CLI chat"],
                 ["yolo / notebook", "ultralytics / jupyter", "YOLO layout detection / notebook"]],
                [2.2 * cm, 7 * cm, 8.2 * cm], 7.8),
          note("<b>Windows tip:</b> run the commands in PowerShell or Windows Terminal from the project folder. "
               "If corporate SSL inspection blocks downloads, set <font face='Courier'>UV_NATIVE_TLS=1</font> "
               "(uses the Windows certificate store).")]

    f += H1("3. Choose the LLM")
    f += [P("Part A has no model restriction; the project defaults to <b>open-weight</b> models so it runs locally and "
            "reproducibly (temperature 0, fixed seed). <font face='Courier'>llm.backend: auto</font> picks the first "
            "available option in this order:"),
          table([["Option", "Setup", "When to use"],
                 ["A. Ollama (default)", "Install from ollama.com, then: ollama pull qwen2.5:3b-instruct  (or qwen2.5:7b-instruct "
                  "for better quality). Ollama must be running (tray app or `ollama serve`).", "Recommended for the report and demo."],
                 ["B. OpenAI-compatible endpoint", "Copy .env.example to .env and set OPENAI_COMPAT_BASE_URL, _API_KEY, _MODEL "
                  "(Groq, Together, OpenRouter, vLLM, LM Studio ...).", "No local GPU/CPU budget; faster runs."],
                 ["C. Extractive fallback", "Nothing. Used automatically if A and B are unavailable, or force with --llm extractive.",
                  "Smoke tests, offline machines. Prompt wording does not change its answers."]],
                [3.6 * cm, 9 * cm, 4.8 * cm], 8),
          code("""
ollama pull qwen2.5:3b-instruct
uv run mmrag --model qwen2.5:7b-instruct query "..."   # override the model per command
uv run mmrag --llm extractive run-all                     # force the offline fallback
""")]

    f += H1("4. Run it step by step")
    f += H2("Step 0 - Check the environment")
    f += [code("uv run mmrag doctor"), P("Reference output (sandbox without model downloads / Ollama):"),
          code("""
  sentence-transformers        missing -> TF-IDF/LSA fallback (uv sync --extra ml)
  chromadb                     installed
  OCR engine                   rapidocr
  Ollama                       not reachable at http://localhost:11434
  OpenAI-compatible endpoint   not configured
  tokenizer                    regex-approx
  documents in corpus          120 in .../data/corpus/raw
  index                        present
"""),
          P("On a fully installed machine you should see <i>installed</i>, <i>reachable</i> and <i>tiktoken/cl100k_base</i>.")]
    f += H2("Step 1 - One command for everything")
    f += [code("uv run mmrag run-all            # add --quick for a 12-question subset (fast with a local LLM)"),
          P("This (1) generates the synthetic 120-document corpus if <font face='Courier'>data/corpus/raw</font> is empty, "
            "(2) parses, OCRs, chunks and indexes it, (3) runs E1-E10 and (4) writes "
            "<font face='Courier'>outputs/REPORT_TABLES.md</font>. The individual steps below do the same thing piece by piece.")]
    f += H2("Step 2 - Corpus")
    f += [code("uv run mmrag generate-corpus --n 120"),
          P("Creates 71 PDFs + 49 PPTX in <font face='Courier'>data/corpus/raw</font> across nine document kinds "
            "(fronthaul test reports, gNB cell configuration guides, handover KPI analyses, 5G core PDU-session reports, "
            "O-DU release notes, Near-RT RIC architecture notes, PTP sync validations, Robot Framework training decks, weekly "
            "KPI reviews with native charts). Each document has a fact in text, a fact in a table and a fact that exists "
            "<b>only inside a figure or chart</b>, which makes image retrieval testable. It also writes "
            "<font face='Courier'>data/golden/golden_set.jsonl</font> (36 questions: 12 text, 12 table, 12 image), "
            "<font face='Courier'>multiturn_dialog.json</font> and <font face='Courier'>guardrail_tests.jsonl</font>. "
            "For the submission, use your group's real corpus (Section 5).")]
    f += H2("Step 3 - Ingestion (Task A1, E1)")
    st = load_json("e1_ingestion_stats.json") or {}
    f += [code("uv run mmrag ingest            # --force re-parses everything (parse results are cached)"),
          table([["metric", "value"]] + [[k, st.get(k)] for k in
                                         ["documents_parsed", "documents_failed", "pdf", "pptx_ppt", "pages_or_slides", "tables",
                                          "images", "ocr_characters", "chunks_total", "chunks_text", "chunks_table",
                                          "chunks_image", "avg_chunk_tokens", "parse_seconds_total", "ocr_engine"]],
                [6 * cm, 5 * cm], 8),
          P("Files written: <font face='Courier'>data/work/elements.jsonl</font> (every parsed element), "
            "<font face='Courier'>data/work/preprocessing_record.jsonl</font> (per document: pages, blocks, tables, images, "
            "OCR characters, seconds, errors), <font face='Courier'>data/index/</font> (Chroma DB, vectors.npy, bm25.pkl, "
            "chunks.jsonl, embedder.pkl, manifest.json) and the inventory CSV/JSON in "
            "<font face='Courier'>outputs/</font>. A file that fails to parse is logged and skipped, never fatal.")]
    f += H2("Step 4 - Ask a question")
    f += [code("""
uv run mmrag query "Which VLAN carries the fronthaul traffic in the topology diagram of Site-754?"

Answer: Fronthaul VLAN 2754 NG 25GE [1].
  [1] fh_test_site754.pptx  page 4  (image)
  tokens in/out: 421/10  latency: 0.13s
"""),
          P("Flags: <font face='Courier'>--variant zero_shot|fstring|three_shot|jinja2|create|react|optimized</font>, "
            "<font face='Courier'>--json</font> (full Answer with timings), <font face='Courier'>--no-cache</font>, "
            "<font face='Courier'>--session NAME</font> (memory scope). The answer above comes only from OCR text in the "
            "figure: the VLAN appears nowhere else in the document.")]
    f += H2("Step 5 - Multi-turn chat (memory, E4) and the demo UI")
    f += [code("""
uv run mmrag chat
you> What eCPRI one-way latency was measured in the fronthaul tests of Site-754?
Answer: Measured for FH-754-01 is 76 us [1].
you> Which VLAN carries its fronthaul traffic in the topology diagram?
Answer: Fronthaul VLAN 2754 NG 25GE [1].
  (standalone query: Which VLAN carries its fronthaul traffic in the topology diagram (regarding Site-754)?)
you> :reset      (clear memory)      :q  (quit)

uv run streamlit run app/streamlit_app.py      # browser UI: variants, citations, chunks, timings
""")]
    f += H2("Step 6 - Experiments E1-E10")
    f += [code("""
uv run mmrag experiments                 # all ten, then REPORT_TABLES.md
uv run mmrag experiments --only E5,E7    # a subset
uv run mmrag experiments --quick         # 12-question balanced subset for E6-E8/E10
uv run mmrag evaluate                    # golden-set evaluation only (E5)
uv run mmrag report                      # rebuild REPORT_TABLES.md from existing CSV/JSON
"""),
          image("experiments_flow.png", W, 4 * cm, "Order of the experiments in run_all()"),
          table([["Exp.", "What runs", "Main output files"],
                 ["E1", "parse + index statistics", "e1_corpus_inventory.csv, e1_ingestion_stats.json"],
                 ["E2", "3 chunking/embedding configs x (dense, hybrid+filters)", "e2_embedding_chunking.csv"],
                 ["E3", "text-only baseline vs multimodal / routing / metadata filter; answer accuracy", "e3_retrieval_design.csv, e3_modality_examples.json"],
                 ["E4", "3-turn dialog with and without memory", "e4_memory_transcript.md, e4_memory.json"],
                 ["E5", "golden set, synthetic QG, reference-free judge, agreement, failures", "e5_*.csv/json/md"],
                 ["E6", "P50/P95, 5-run consistency, guard overhead, DoS burst, tokens vs naive", "e6_*.csv, e6_nonfunctional.json"],
                 ["E7", "6 prompt variants + naive baseline (quality, tokens, latency, consistency)", "e7_prompt_ablation.csv"],
                 ["E8", "self-improving prompt, dev/test split, >= 3 iterations", "e8_selfimprove_log.md/json, e8_optimized_prompt.json"],
                 ["E9", "15 guardrail tests, ontology routing, rendered prompt", "e9_guardrail_tests.csv, e9_ontology.yaml, e9_rendered_prompt_example.txt"],
                 ["E10", "final benchmark incl. optimised prompt + demo answers", "e10_benchmark.csv, e10_demo_answers.json"]],
                [1.2 * cm, 9 * cm, 7.2 * cm], 7.8)]
    f += H2("Step 7 - Notebook (Colab-equivalent deliverable)")
    f += [code("""
uv sync --extra all --extra notebook
uv run jupyter notebook notebooks/PartA_end_to_end.ipynb
# Google Colab: upload the zip, then in the first cell:
#   !pip install -q uv && unzip -q mmrag_5g.zip && cd mmrag_5g && uv pip install --system -e '.[all]'
"""), P("The notebook walks through corpus, ingestion, one question per modality, memory, guardrails and run_all(), "
       "and displays REPORT_TABLES.md inline.")]
    f += H2("Step 8 - Tests and regenerating these PDFs")
    f += [code("""
uv run --group dev pytest -q                 # 6 unit tests (guards, router, metrics, extractive, chunker, memory)
uv run python docs/diagrams.py               # needs Graphviz 'dot' on PATH
uv run python docs/build_pdfs.py             # rebuilds the three PDFs from your latest outputs/
""")]

    f += H1("5. Using your group's real corpus")
    f += bullets(["<b>Collect >= 100 distinct public / sanitised documents</b> from your domain (for 5G: 3GPP TS/TR PDFs, O-RAN "
                  "Alliance white papers, vendor-neutral training decks, conference slides, public test reports). Include "
                  "documents with tables and with figures. No duplicates, no confidential material.",
                  "Delete the synthetic files (<font face='Courier'>data/corpus/raw/*</font>, "
                  "<font face='Courier'>data/golden/*</font>) and copy yours into <font face='Courier'>data/corpus/raw/</font> "
                  "(sub-folders allowed; .pdf, .pptx, .ppt).",
                  "<font face='Courier'>uv run mmrag ingest --force</font> and check <font face='Courier'>outputs/e1_*</font>.",
                  "<font face='Courier'>uv run mmrag golden-template --per-modality 15</font> samples 45 chunks into "
                  "<font face='Courier'>outputs/golden_template.csv</font>. For each row you keep, write a question, the "
                  "answer and a short <i>key</i> (the token that must appear in a correct answer, e.g. 2754 or R24.2.8). "
                  "Keep >= 30 rows covering text, table and image questions; check each against the page.",
                  "<font face='Courier'>uv run mmrag golden-from-csv outputs/golden_template.csv</font> writes "
                  "<font face='Courier'>data/golden/golden_set.jsonl</font>.",
                  "Write <font face='Courier'>data/golden/multiturn_dialog.json</font> (3 turns, the 2nd and 3rd as follow-ups "
                  "with 'it/its/there'), and adapt the 4 benign questions in "
                  "<font face='Courier'>guardrail_tests.jsonl</font> to your corpus.",
                  "Extend <font face='Courier'>config/ontology.yaml</font> (doc types, topics, keywords) and the identifier "
                  "pattern <font face='Courier'>ENTITY</font> in <font face='Courier'>src/mmrag/retrieval/router.py</font> "
                  "for your document identifiers (e.g. TS 38.211, clause numbers).",
                  "<font face='Courier'>uv run mmrag experiments</font> - then copy tables from REPORT_TABLES.md into the report."])
    f += [code("""
{"qid": "Q001", "question": "Which subcarrier spacings does TS 38.211 allow for SSB?",
 "answer": "15, 30, 120 and 240 kHz", "key": "240 kHz", "modality": "table",
 "source": "ts_138211.pdf", "doc_id": "ts_138211", "page": 37, "difficulty": "medium"}
"""), P("doc_id is the file stem with non-alphanumerics replaced by '_' (first 48 characters); page is 1-based "
       "(slide number for PPTX). Retrieval metrics count a hit when a chunk from that document, page and modality is retrieved.")]

    f += H1("6. Reference")
    f += H2("6.1 Most useful configuration keys (config/config.yaml)")
    f += [table([["Key", "Default", "Effect"],
                 ["chunking.strategy / chunk_tokens / overlap_tokens", "recursive / 220 / 40", "text chunking (E2 compares page)"],
                 ["embedding.model / alt_model", "bge-small-en-v1.5 / all-MiniLM-L6-v2", "primary and E2 comparison model"],
                 ["vector_store.backend", "auto", "chroma | numpy"],
                 ["retrieval.top_k / candidate_k", "4 / 30", "chunks sent to LLM / candidates per retriever"],
                 ["retrieval.hybrid, modality_routing, ontology_filter", "true", "switch retrieval features (E3)"],
                 ["retrieval.rerank", "auto", "cross-encoder when sentence-transformers is present"],
                 ["context.max_tokens_per_chunk / max_context_tokens", "140 / 700", "token budget (NFR2)"],
                 ["llm.backend / ollama.model / temperature / seed", "auto / qwen2.5:3b-instruct / 0 / 42", "generator"],
                 ["llm.timeout_s", "80", "hard cap per LLM call (keeps P95 < 90 s)"],
                 ["prompting.default_variant", "jinja2", "variant used by query/chat/UI"],
                 ["guardrails.dos.*", "1000 chars, 30 rpm, 4 concurrent", "DoS limits"],
                 ["guardrails.output.min_grounding", "0.35", "refuse answers less grounded than this"],
                 ["evaluation.consistency_runs / optimizer_iterations", "5 / 4", "NFR1 runs / E8 iterations"]],
                [6.4 * cm, 4.8 * cm, 6.2 * cm], 7.6)]
    f += H2("6.2 Expected runtimes")
    f += [table([["Step", "Reference sandbox (fallbacks)", "Laptop CPU + Ollama qwen2.5:3b (estimate)"],
                 ["Ingestion of 120 docs (OCR dominates)", "~100 s", "2-4 min (+ first model download)"],
                 ["One query", "~0.01 s", "2-8 s"],
                 ["run-all (full)", "~2 min", "45-90 min (~700 LLM calls)"],
                 ["run-all --quick", "~2 min", "15-25 min"]], [6.4 * cm, 5 * cm, 6 * cm], 8),
          P("With a GPU or a hosted endpoint, LLM time drops by 5-20x. The answer cache makes repeated queries instant; "
            "evaluation runs disable it so measurements stay honest.")]
    f += H2("6.3 Troubleshooting")
    f += [table([["Symptom", "Cause / fix"],
                 ["'No index at data/index' ", "Run uv run mmrag ingest first."],
                 ["'extractive fallback LLM' warning", "Ollama not running / model not pulled: start Ollama, ollama pull qwen2.5:3b-instruct; or set .env."],
                 ["'TF-IDF/LSA fallback' warning", "uv sync --extra ml; first run needs internet to download bge-small."],
                 ["Only N documents found (FR1 warning)", "Fewer than 100 files in data/corpus/raw; add documents."],
                 ["legacy .ppt could not be converted", "Install LibreOffice (soffice on PATH) or save the deck as .pptx."],
                 ["OCR engine: none", "uv sync --extra ocr (RapidOCR) or install Tesseract."],
                 ["Changed documents not picked up", "uv run mmrag ingest --force (parse cache keyed on file names + sizes)."],
                 ["LLM timeouts / slow E7", "Use --quick, a smaller model, or an OpenAI-compatible endpoint; llm.timeout_s caps each call."],
                 ["Chroma errors on Windows", "Set vector_store.backend: numpy (exact search, same results)."],
                 ["Streamlit not found", "uv sync --extra ui"],
                 ["Proxy / SSL errors during uv sync", "UV_NATIVE_TLS=1, or configure HTTPS_PROXY."]], [6 * cm, 11.4 * cm], 8)]
    f += H2("6.4 Cheat sheet")
    f += [code("""
uv sync --extra all                         install
uv run mmrag doctor                         environment check
uv run mmrag run-all [--quick]              everything end to end
uv run mmrag generate-corpus --n 120        synthetic corpus + golden set
uv run mmrag ingest [--force]               parse + OCR + chunk + index
uv run mmrag query "..." [--variant react] [--json]
uv run mmrag chat                           multi-turn with memory
uv run streamlit run app/streamlit_app.py   UI
uv run mmrag experiments [--only E3,E7] [--quick]
uv run mmrag golden-template | golden-from-csv FILE
uv run mmrag report                         rebuild outputs/REPORT_TABLES.md
""")]

    f += H1("7. Results of the reference run")
    f += [ref_run_note()]
    e5 = load_json("e5_summary.json") or {}
    e6 = load_json("e6_nonfunctional.json") or {}
    g = e5.get("golden", {})
    f += [table([["Measure", "Value", "Requirement"],
                 ["Documents ingested (PDF + PPTX)", f"{st.get('documents_parsed')} ({st.get('pdf')} + {st.get('pptx_ppt')})", "FR1 >= 100"],
                 ["Golden Recall@1 / @5 / MRR", f"{g.get('recall@1')} / {g.get('recall@5')} / {g.get('mrr')}", "A4"],
                 ["Answer accuracy / faithfulness / citation accuracy",
                  f"{g.get('answer_accuracy')} / {g.get('faithfulness')} / {g.get('citation_accuracy')}", "A4"],
                 ["P50 / P95 latency (s)", f"{e6.get('latency', {}).get('p50_s')} / {e6.get('latency', {}).get('p95_s')}", "FR3 P95 < 90 s"],
                 ["Guard time P95 (ms)", f"{e6.get('guardrail_overhead', {}).get('p95_guard_time_ms')}", "FR4 < 6 s"],
                 ["Identical retrieval / answers over 5 runs",
                  f"{e6.get('consistency', {}).get('identical_retrieval_share')} / {e6.get('consistency', {}).get('identical_answer_share')}", "NFR1"],
                 ["Input tokens: naive -> final", f"{e6['tokens'][0]['avg_input_tokens']:.0f} -> {e6['tokens'][2]['avg_input_tokens']:.0f} "
                  f"({e6['tokens'][2]['token_reduction_vs_naive']} less)" if e6.get("tokens") else "-", "NFR2"]],
                [7.4 * cm, 5.6 * cm, 4.4 * cm], 8.5)]
    doc.multiBuild(f)


# =============================================================================================
# 2. ARCHITECTURE
# =============================================================================================
def architecture(path: Path):
    doc = Doc(path, "2 - Architecture, diagrams and call flow")
    f = cover("Project Architecture", "HLD, LLD, data and call flow, module dependencies, constraint-to-design mapping",
              ["1. Scenario, users and why retrieval-augmented generation",
               "2. High-level design: ingestion path and query path",
               "3. Ingestion design: parsing, OCR, layout detection, chunking, metadata, embeddings",
               "4. Index, metadata schema, retrieval strategy, multimodal fusion and memory",
               "5. Query path, sequence diagram of one query, error handling",
               "6. Low-level design: classes, interfaces, schemas, module call graph (which module calls which)",
               "7. Prompting layer, guardrails, ontology, self-improvement, evaluation design",
               "8. Mapping of every constraint (FR1-FR5, NFR1-NFR2) to design decisions and evidence"])

    f += H1("1. Scenario")
    f += [P("<b>Target users:</b> RAN test and automation engineers (5G gNodeB / O-RAN integration, regression and "
            "acceptance testing). Their knowledge lives in test reports, configuration guides, KPI analyses, release notes, "
            "architecture notes and training decks, as PDFs and PowerPoints full of tables (parameters, KPIs, verdicts) and "
            "diagrams (topologies, call flows, trend charts)."),
          P("<b>Usage scenario:</b> during a test campaign an engineer asks 'Which VLAN carries the fronthaul for Site-754?', "
            "'What PCI is configured on C955-2?', 'Which defect does build R24.2.8 fix?' and follows up conversationally. "
            "The answer must be exact, cite the page, and be fast and cheap enough for daily use."),
          P("<b>Why retrieval and not fine-tuning:</b> the facts are numerous, change with every release and must be cited "
            "verbatim; retrieval keeps the model frozen, updates instantly by re-indexing, gives page-level provenance and "
            "lets the model refuse when there is no evidence. Fine-tuning would memorise stale numbers without citations and "
            "needs labelled data and GPUs. Prompting alone cannot see the documents.")]

    f += [NextPageTemplate("landscape"), PageBreak()]
    f += H1("2. High-level design (HLD)")
    f += [image("hld.png", LW, LH - 2.2 * cm, "Figure 1 - Components and data flow of the ingestion path (top), the index "
                "(middle) and the query path (bottom)."),
          NextPageTemplate("portrait"), PageBreak()]
    f += [P("<b>Ingestion path (offline).</b> Every file in the corpus is parsed once. PDFs go through PyMuPDF (text blocks with "
            "font-size heading detection, <font face='Courier'>find_tables</font> for tables), PPTX through python-pptx "
            "(text frames, tables, pictures, native chart data, speaker notes). A layout detector finds figure regions; only "
            "those crops are OCR-ed. Elements are chunked per modality, tagged with ontology metadata and embedded; vectors go "
            "into ChromaDB, tokens into BM25, and chunk text + metadata into a JSONL chunk store."),
          P("<b>Query path (online).</b> A query passes DoS and input guards, is rewritten if it is a follow-up, checked "
            "against the answer cache, routed through the ontology (entities, topics, modality), retrieved with hybrid "
            "metadata-filtered search, checked for scope, compressed into a small context, rendered into a prompt variant, "
            "answered by the LLM and verified by the output guard (citation + grounding) before it is returned and stored "
            "in memory.")]

    f += H1("3. Ingestion design (Task A1)")
    f += [image("ingestion_flow.png", W, 12 * cm, "Figure 2 - Ingestion call flow (functions in src/mmrag/ingest and index)")]
    f += [table([["Concern", "Design decision", "Why"],
                 ["PDF text", "PyMuPDF text blocks in reading order; heading = font >= 1.15 x body size, sets 'section'; blocks "
                              "inside tables/figures skipped", "fast, layout-aware, no duplicate text"],
                 ["PDF tables", "find_tables() for the grid, cell text rebuilt from word boxes, Markdown with header; caption "
                                "'Table n:' attached", "Table.extract() drops spaces; Markdown is compact and LLM-readable"],
                 ["PPTX", "title, text frames, tables, group shapes (recursive), notes; charts read from embedded series data",
                  "chart data is exact and far cheaper than OCR"],
                 ["Figures", "heuristic layout: raster image rects + clusters of vector drawings; optional YOLO doc-layout "
                             "(figure/picture/chart classes); repeated logos (same hash > 2x) and icons < 80 px skipped",
                  "process only the region that carries information"],
                 ["OCR", "RapidOCR (pip only, onnxruntime) -> Tesseract -> none; images upscaled to 900 px; boxes grouped into "
                         "lines; glued-word repair (FronthaulVLAN2180 -> Fronthaul VLAN 2180)", "works on Windows without "
                                                                                              "system installs"],
                 ["Legacy .ppt", "LibreOffice headless conversion to .pptx", "one parser for all decks"],
                 ["Chunking", "text: per section, sentence-packed to 220 tokens, 40 overlap, tiny heading groups merged; "
                              "tables: own chunk, split every 12 rows with header repeated; figures: caption + OCR text",
                  "modality becomes a filterable field; small chunks = fewer tokens"],
                 ["Metadata", "doc_type, topic(s) from the ontology keyword tagger, page, section, date ('Document date:' or file "
                              "metadata), modality, source, sensitivity, tokens", "routing, filters, prompt placeholders"],
                 ["Embeddings", "BAAI/bge-small-en-v1.5 (384-d, strong MTEB retrieval, CPU-fast, query instruction prefix); "
                                "E2 compares all-MiniLM-L6-v2 and page chunking; offline fallback TF-IDF(1-2 gram)+LSA",
                  "quality / speed balance; always runnable"],
                 ["Embedded text", "'title | section | modality: text' (contextual header)", "disambiguates near-identical "
                                                                                             "templates across sites"],
                 ["Robustness", "one failing file is logged in the preprocessing record and skipped; parse cache keyed on file "
                                "names + sizes", "large corpora re-index quickly"]],
                [2.4 * cm, 10 * cm, 5 * cm], 7.6)]

    f += H1("4. Index, retrieval and memory (Task A2)")
    f += H2("4.1 Vector store and metadata schema")
    f += [P("ChromaDB persistent collection (cosine HNSW) with the chunk metadata stored alongside each vector, so "
            "<font face='Courier'>where</font> filters run inside the store. An exact NumPy copy of the vectors is always "
            "saved: it is the deterministic fallback and is used for the in-memory E2 experiments. Both implement the same "
            "interface <font face='Courier'>add / query(vector, k, where) / count</font>."),
          table([["Field", "Type", "Example", "Used for"],
                 ["chunk_id", "str", "fh_test_site754-p4-i0", "identity, citation, consistency checks"],
                 ["doc_id / source", "str", "fh_test_site754 / fh_test_site754.pptx", "entity filter, citation"],
                 ["doc_type", "enum", "test_report", "routing, prompt placeholder"],
                 ["title / section", "str", "O-RAN Fronthaul ... / 1. System Under Test", "embedded header, citation"],
                 ["page", "int", "4", "citation, golden-set relevance"],
                 ["modality", "enum", "text | table | image", "modality routing / guarantee, E3"],
                 ["topic / topics", "enum / csv", "oran / oran,transport", "topic pre-filter"],
                 ["date / sensitivity", "str", "2025-02-26 / public", "ontology, governance"],
                 ["tokens", "int", "47", "token budgeting"]], [3 * cm, 1.6 * cm, 6 * cm, 6.8 * cm], 7.8)]
    f += H2("4.2 Retrieval strategy and multimodal fusion")
    f += bullets(["<b>Metadata pre-filter.</b> Identifiers in the query (Site-754, C955-2, R24.2.8, O-RU-328B ...) select the "
                  "documents that mention them; otherwise the routed topics select documents tagged with those topics "
                  "(relaxed automatically if too few candidates). This narrows candidates before retrieval (token hint 3).",
                  "<b>Hybrid search.</b> Dense cosine top-30 and BM25 top-30 on the same filter; BM25 catches exact codes that "
                  "embeddings blur.",
                  "<b>Reciprocal rank fusion.</b> score = sum 1/(60 + rank) over both lists, normalised to [0,1].",
                  "<b>Multimodal fusion.</b> All modalities share one embedding space (tables as Markdown, figures as caption + "
                  "OCR text), so they compete in one ranked list. Modality cues in the query ('figure', 'diagram', 'chart' -> image; "
                  "'table', 'row', 'assigned to' ... -> table) add +0.25 to chunks of that modality, and a <i>modality "
                  "guarantee</i> swaps the best chunk of the requested modality into the last slot if none made the top-k.",
                  "<b>Memory boost.</b> Documents cited in the previous turn get +0.15 for follow-ups.",
                  "<b>Optional re-ranking.</b> A cross-encoder (ms-marco-MiniLM-L-6-v2) re-scores the top 12 when available.",
                  "<b>Determinism.</b> Ties are broken by chunk_id; temperature 0 and a fixed seed for the LLM (NFR1)."])
    e3 = load_csv("e3_retrieval_design.csv")
    if e3 is not None:
        f += [P("Effect of each step on the golden set (E3, reference run):"),
              df_table(e3, ["setup", "recall@1", "recall@5", "mrr", "recall@5_image", "recall@5_table", "answer_accuracy"],
                       {"recall@5_image": "R@5 image", "recall@5_table": "R@5 table", "answer_accuracy": "answer acc."},
                       [6.6 * cm, 1.6 * cm, 1.6 * cm, 1.4 * cm, 1.9 * cm, 1.9 * cm, 2.4 * cm], 7.4)]
    f += H2("4.3 Memory design")
    f += [table([["Layer", "Implementation", "Behaviour"],
                ["Short-term conversational", "ConversationMemory: deque of the last 4 turns per session (query, standalone query, "
                                              "answer, entities, cited doc_ids)",
                 "follow-up detection (it/its/there/that, leading 'and / what about', or no own entity) -> rewrite by carrying "
                 "the previous entities; optional LLM rewrite (memory.rewrite: llm); last 2 turns shown in the Jinja2 prompt"],
                ["Long-term answer cache", "AnswerCache: SQLite, key = sha1(normalised standalone query | variant | index version | "
                                           "llm)", "repeated questions answered with 0 LLM tokens; invalidated by re-ingestion"],
                ["Query-embedding cache", "lru_cache on Retriever._qvec", "repeated retrievals skip the embedder"]],
               [3.2 * cm, 6.8 * cm, 7.4 * cm], 7.8)]

    f += H1("5. Query path")
    f += [image("query_flow.png", W, 15.5 * cm, "Figure 3 - Call flow of RAGSystem.answer() with the four refusal exits")]
    f += [image("sequence.png", W, 12 * cm, "Figure 4 - Sequence diagram of one query (solid = call, dashed = return)")]
    f += H2("5.1 Error handling")
    f += [table([["Failure", "Where", "Handling"],
                 ["Corrupt / encrypted / unsupported file", "ingest.pipeline.parse_corpus", "caught per file, status=error in the "
                                                                                          "preprocessing record, ingestion continues"],
                 ["Legacy .ppt without LibreOffice", "pptx_parser.convert_ppt", "warning, file marked failed"],
                 ["No OCR engine / no YOLO weights", "OCREngine / LayoutDetector", "degrade to captions / heuristic layout"],
                 ["Embedding model download fails", "Embedder", "TF-IDF/LSA fallback, logged"],
                 ["ChromaDB missing / filter matches nothing", "vector_store", "NumPy store / empty list -> filter relaxed"],
                 ["Over-long, flooding or repeated queries", "DoSGuard", "immediate refusal 'Request rejected: dos ...'"],
                 ["Injection / PII", "InputGuard", "refusal with reason (PII can be redacted instead: block_pii: false)"],
                 ["Out-of-scope question", "InputGuard.out_of_scope", "refusal before any LLM call"],
                 ["LLM unreachable / timeout (80 s)", "RAGSystem.answer", "refusal 'llm: <error>', no crash; P95 stays < 90 s"],
                 ["Answer without citation / ungrounded / leaks prompt", "OutputGuard", "citation repair if grounded, else "
                                                                                        "refusal text"],
                 ["Experiment failure", "Experiments.run_all", "logged in outputs/run_status.json, next experiment runs"]],
                [5.2 * cm, 4.4 * cm, 7.8 * cm], 7.8)]

    f += [NextPageTemplate("landscape"), PageBreak()]
    f += H1("6. Low-level design (LLD)")
    f += [image("class_diagram.png", LW, LH - 2.2 * cm, "Figure 5 - Main classes, their key methods and who uses whom"),
          NextPageTemplate("portrait"), PageBreak()]
    f += H2("6.1 Modules and interfaces")
    f += [table([["Module (src/mmrag/...)", "Responsibility", "Key interface"],
                 ["config.py", "load YAML + .env, dotted access", "load_config() -> Config; cfg.get('a.b'); cfg.path(key)"],
                 ["schemas.py", "shared dataclasses", "Element, Chunk, Hit, Answer"],
                 ["ingest/pdf_parser.py", "PDF text/tables/figures", "parse_pdf(path, doc_id, ocr, layout) -> (elements, stats)"],
                 ["ingest/pptx_parser.py", "PPTX/PPT parsing", "parse_pptx(path, doc_id, ocr); convert_ppt()"],
                 ["ingest/ocr.py, layout.py", "OCR engine, figure regions", "OCREngine.read(img) -> (text, conf); LayoutDetector.figure_regions(page)"],
                 ["ingest/chunker.py, metadata.py", "chunking, ontology tags", "chunk_document(...) -> [Chunk]; OntologyTagger"],
                 ["ingest/pipeline.py", "orchestration, records", "parse_corpus(), make_chunks(), write_inventory(), run_ingest()"],
                 ["index/__init__.py", "build/load index", "build_index(chunks, dir, ...) -> Index; Index.load(dir)"],
                 ["index/embedder.py, vector_store.py, bm25.py", "vectors, stores, sparse", "Embedder.encode(); store.query(v, k, where); BM25Index.query()"],
                 ["retrieval/router.py", "ontology routing", "QueryRouter.route(q) -> Route"],
                 ["retrieval/retriever.py", "hybrid retrieval", "Retriever.search(q, route, k, ...) -> (hits, info)"],
                 ["memory/conversation.py, cache.py", "memory, cache", "ConversationMemory.rewrite(); AnswerCache.get/put"],
                 ["llm/backends.py", "LLM backends", "make_llm(cfg) -> LLM; LLM.generate(messages, ...) -> LLMResponse"],
                 ["prompting/context.py", "context assembly", "build_context(hits, q, cfg, mode) -> ([ContextItem], stats)"],
                 ["prompting/variants.py", "7 prompt variants", "PromptBuilder.build(variant, q, contexts, route, ...)"],
                 ["prompting/optimizer.py", "self-improving prompt", "PromptOptimizer.run(out_dir); split_dev_test()"],
                 ["guardrails/*.py", "DoS, input, output", "DoSGuard.check(); InputGuard.check()/out_of_scope(); OutputGuard.check()"],
                 ["pipeline/rag.py", "end-to-end query path", "RAGSystem.answer(query, session, variant, ...) -> Answer"],
                 ["eval/*.py", "metrics, runners, auto eval, failures, E1-E10", "run_questions(); summarise(); Experiments.run_all()"],
                 ["corpus/synth.py", "synthetic corpus + golden set", "generate(out_dir, golden_dir, n_docs)"],
                 ["cli.py", "commands", "main(): doctor, ingest, query, chat, experiments, run-all, ..."]],
                [4.6 * cm, 4 * cm, 8.8 * cm], 7.4)]
    f += H2("6.2 Data schemas")
    f += [code(snippet("src/mmrag/schemas.py", None, 75, start="@dataclass\nclass Element") if False else
               "\n".join(l for l in (ROOT / "src/mmrag/schemas.py").read_text().splitlines()[12:75] if not l.strip().startswith('"""')))]
    f += H2("6.3 Module call graph - which module calls which")
    f += [P("Computed from the source with Python's <font face='Courier'>ast</font> (relative imports), so it is exactly the "
            "dependency structure of the code. Arrows point from caller to callee; utility modules (config, schemas, "
            "utils.log, utils.tokens) are used everywhere and omitted for readability."),
          image("module_graph.png", W, 20.5 * cm, "Figure 6 - Module dependency (call) graph of src/mmrag")]
    edges = (ROOT / "docs/img/module_edges.txt").read_text().splitlines()
    core = [e for e in edges if not any(x in e for x in ("utils.", "-> config", "-> schemas"))]
    callers: dict[str, list[str]] = {}
    for e in core:
        a, b = e.split(" -> ")
        callers.setdefault(a, []).append(b)
    f += [P("The same graph as a table (caller -> callees):"),
          table([["Caller module", "Calls / imports"]] + [[a, ", ".join(sorted(bs))] for a, bs in sorted(callers.items())],
                [4.4 * cm, 13 * cm], 7.4)]
    f += H2("6.4 Runtime call chain for one query")
    f += [code("""
cli.main -> cmd_query -> RAGSystem(cfg)                              # pipeline/rag.py
  Index.load -> Embedder.load, ChromaStore | NumpyStore.load, BM25Index.load
  Retriever(index, cfg, QueryRouter) ; make_llm(cfg) ; PromptBuilder(cfg)
  DoSGuard, InputGuard, OutputGuard, ConversationMemory, AnswerCache
RAGSystem.answer(q, session, variant)
  DoSGuard.check -> InputGuard.check
  QueryRouter.route -> ConversationMemory.rewrite -> AnswerCache.get
  Retriever.search
      entity_docs | doc-topic filter -> store.query (dense) + BM25Index.query (sparse)
      -> RRF -> modality boost -> memory boost -> [CrossEncoder.predict] -> modality guarantee
  InputGuard.out_of_scope
  context.build_context -> compress (per chunk)  |  naive whole pages
  PromptBuilder.build(variant) -> Jinja2 templates in prompts/
  LLM.generate  (react: loop with Retriever.search for search[...] actions)
  OutputGuard.check -> grounding_score
  AnswerCache.put ; ConversationMemory.add -> Answer
""")]

    f += H1("7. Prompting layer, guardrails and ontology (Task A5)")
    f += H2("7.1 Prompt variants")
    f += [table([["Variant", "Construction", "Ontology use"],
                 ["zero_shot", "context + question", "-"],
                 ["fstring", "Python f-string: role, grounding + citation rule, refusal text", "-"],
                 ["three_shot", "f-string + 3 demos (table cell, figure OCR, refusal) about documents not in the corpus", "-"],
                 ["jinja2", "prompts/answer_jinja2.j2", "topic labels, routed modality hint, per-chunk source/page/modality/section, memory"],
                 ["create", "prompts/create.j2 - Character, Request, Examples, Adjustments, Type, Extras", "modality, doc_type, topics"],
                 ["react", "prompts/react.j2 - Thought/Action/Observation, tool search[...], <= 2 extra searches", "-"],
                 ["optimized", "jinja2 + instruction and demos found by the optimiser (E8)", "as jinja2"]],
                [2.2 * cm, 9 * cm, 6.2 * cm], 7.8)]
    f += H2("7.2 Self-improving loop")
    f += bullets(["Golden set split per modality into DEV and held-out TEST (seeded).",
                  "Objective = 0.7 accuracy + 0.3 faithfulness - 0.0005 x input tokens (rewards fewer tokens at equal quality).",
                  "Each iteration proposes: an LLM critique-and-rewrite of the instruction from the dev failures; a failure-driven "
                  "rule edit chosen from the failing modalities; bootstrapped few-shot demos from correctly answered dev items.",
                  "Candidates are scored on DEV; the best is accepted only if the objective improves. Final prompt is reported on "
                  "TEST and saved to outputs/e8_optimized_prompt.json, which the 'optimized' variant loads."])
    f += H2("7.3 Guardrail specification")
    f += [table([["Check", "Rule", "Action"],
                 ["DoS: size", "query > 1000 characters", "reject"],
                 ["DoS: rate", "token bucket 30 requests/min per session", "reject"],
                 ["DoS: burst", "same query >= 5 times in 10 s", "reject"],
                 ["DoS: concurrency", "bounded semaphore of 4", "wait max 5 s"],
                 ["Injection", "16 patterns: ignore/disregard instructions, reveal system prompt, 'you are now', DAN, "
                               "developer mode, fake &lt;context&gt;/SYSTEM: tags, 'answer without citations' ...", "reject"],
                 ["PII", "email, Indian mobile, Aadhaar, PAN, Luhn-valid card numbers", "reject (or redact)"],
                 ["Out of scope", "no ontology/entity vocabulary AND weak retrieval (dense < 0.12 or BM25 < 2)", "reject"],
                 ["Citation", "answer must cite an existing [n]; repaired only if grounding >= threshold", "refuse"],
                 ["Grounding", "share of answer tokens present in the cited chunks >= 0.35", "refuse"],
                 ["Refusal / leakage", "model says no evidence -> standard refusal text; prompt scaffolding in output -> refuse",
                  "refuse"]], [2.6 * cm, 11 * cm, 3.8 * cm], 7.8)]
    f += H2("7.4 Ontology (config/ontology.yaml)")
    f += [P("Defines doc_types (6), topics (8: RAN, O-RAN, 5G Core, Mobility, Transport, Test Automation, Security, "
            "Performance) with keywords, modalities, modality cues, sensitivity levels and the chunk schema. It is consumed "
            "by the tagger (metadata at ingestion), the router (filters and modality routing), the Jinja2/CREATE prompts "
            "(placeholders) and the out-of-scope guard (domain vocabulary).")]
    rp = (OUT / "e9_rendered_prompt_example.txt")
    if rp.exists():
        f += [P("Rendered Jinja2 prompt for a table question (from outputs/e9_rendered_prompt_example.txt):"),
              code(rp.read_text(encoding="utf-8"), 112)]

    f += H1("8. Evaluation design (Task A4)")
    f += bullets(["<b>Golden set:</b> 36 questions (12 text / 12 table / 12 image) with answer, key, source document and page. "
                  "Retrieval: Recall@1/3/5 and MRR, relevance = same document, page and modality. Answers: accuracy (key "
                  "present, word-boundary match), faithfulness (grounding of the answer in the cited chunks), citation accuracy.",
                  "<b>Automatic approach 1 - synthetic QG:</b> 30 questions generated from chunks of documents not in the golden "
                  "set (LLM-written when a generative model is present, templates otherwise); the source chunk is the label.",
                  "<b>Automatic approach 2 - reference-free judge:</b> rubric GROUNDED / CITED / RESPONSIVE, applied by the LLM or "
                  "a deterministic proxy, compared with the golden labels (agreement, Cohen's kappa, confusion counts).",
                  "<b>Failure analysis:</b> every golden miss is diagnosed automatically (retrieval_miss, ocr_miss, wrong_evidence, "
                  "false_refusal, generation_error) with a proposed fix."])

    f += H1("9. Constraint-to-design mapping (Task A3)")
    e6 = load_json("e6_nonfunctional.json") or {}
    st = load_json("e1_ingestion_stats.json") or {}
    tok = e6.get("tokens", [{}, {}, {}])
    f += [table([["ID", "Requirement", "Design decisions", "Evidence (reference run)"],
                 ["FR1", ">= 100 PDFs/PPTs", "recursive corpus walk, .ppt conversion, per-file error isolation, parse cache",
                  f"{st.get('documents_parsed')} parsed, {st.get('documents_failed')} failed (e1_*)"],
                 ["FR2", "text, tables, images searchable", "table Markdown chunks, figure OCR chunks, chart data, modality field + "
                                                           "routing + guarantee", "e3_modality_examples.json; image R@5 0 -> 1.0 (E3)"],
                 ["FR3", "P95 < 90 s", "top-k 4, compressed context, max_tokens 220, 80 s LLM timeout, local models, caches",
                  f"P95 = {e6.get('latency', {}).get('p95_s')} s"],
                 ["FR4", "guards < 6 s", "regex/O(1) checks only, no LLM calls in guards", f"P95 guard time = "
                                                                                         f"{e6.get('guardrail_overhead', {}).get('p95_guard_time_ms')} ms"],
                 ["FR5", "self-improving prompts", "PromptOptimizer: critique-rewrite, rule edits, bootstrapped demos, dev/test",
                  "e8_selfimprove_log.md"],
                 ["NFR1", "consistency over 5 runs", "temperature 0 + seed, deterministic tie-breaks, exact NumPy fallback, cache "
                                                    "off during tests", f"identical answers "
                                                                        f"{e6.get('consistency', {}).get('identical_answer_share')}"],
                 ["NFR2", "token optimisation", "OCR text instead of images, layout crops, entity/topic filters, top-k, sentence/row "
                                                "compression, answer cache",
                  f"{tok[0].get('avg_input_tokens', 0):.0f} -> {tok[2].get('avg_input_tokens', 0):.0f} input tokens "
                  f"({tok[2].get('token_reduction_vs_naive', '')} less)" if len(tok) > 2 else "-"]],
                [1.2 * cm, 3.2 * cm, 7.6 * cm, 5.4 * cm], 7.8),
          Spacer(1, 6), ref_run_note()]
    doc.multiBuild(f)


# =============================================================================================
# 3. CODE WALKTHROUGH
# =============================================================================================
def tree_text() -> str:
    skip = {".venv", "__pycache__", "data", "outputs", ".git", "img", ".pytest_cache"}
    lines = ["mmrag_5g/"]

    def walk(d: Path, prefix: str):
        items = sorted([p for p in d.iterdir() if p.name not in skip and not p.name.endswith((".pdf", ".lock"))
                        and not p.name.startswith(".")], key=lambda p: (p.is_file(), p.name))
        for i, p in enumerate(items):
            last = i == len(items) - 1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{p.name}{'/' if p.is_dir() else ''}")
            if p.is_dir():
                walk(p, prefix + ("    " if last else "│   "))

    walk(ROOT, "")
    lines += ["├── data/  corpus/raw (documents), golden/ (golden set, dialog, guard tests),",
              "│         work/ (parse cache, preprocessing record), index/ (vector DB, BM25), cache/",
              "└── outputs/  all experiment results + REPORT_TABLES.md"]
    return "\n".join(lines)


def walkthrough(path: Path):
    doc = Doc(path, "3 - Complete code walkthrough")
    f = cover("Complete Code Walkthrough", "Every module, what it does, how it works and how the pieces call each other",
              ["1. Repository layout and entry points",
               "2. Three execution traces: ingest, query, run-all",
               "3. Module-by-module walkthrough with the real code",
               "4. Extension points: new parsers, models, prompt variants, guards"])

    f += H1("1. Repository layout")
    f += [code(tree_text(), 112)]
    f += [table([["Entry point", "How to call", "What it starts"],
                 ["CLI", "uv run mmrag <command>  (pyproject [project.scripts])", "src/mmrag/cli.py: main()"],
                 ["Python API", "from mmrag.pipeline.rag import RAGSystem", "RAGSystem(load_config()).answer(q)"],
                 ["Experiments", "Experiments(cfg).run_all()", "src/mmrag/eval/experiments.py"],
                 ["UI", "uv run streamlit run app/streamlit_app.py", "Streamlit chat around RAGSystem"],
                 ["Notebook", "notebooks/PartA_end_to_end.ipynb", "same API, step by step"]], [2.6 * cm, 7.6 * cm, 7.2 * cm], 8)]

    f += H1("2. Execution traces")
    f += H2("2.1 uv run mmrag ingest")
    f += [code("""
cli.cmd_ingest(args)
 └─ ingest.pipeline.run_ingest(cfg)
     ├─ parse_corpus(cfg)                       # cached in data/work/elements.jsonl
     │   ├─ ocr.get_ocr()  -> OCREngine(rapidocr | tesseract | none)
     │   ├─ layout.LayoutDetector(heuristic | yolo)
     │   └─ for each file:
     │       ├─ .ppt  -> pptx_parser.convert_ppt()            (LibreOffice)
     │       ├─ .pdf  -> pdf_parser.parse_pdf()
     │       │          ├─ page.find_tables() -> _table_rows() -> rows_to_markdown()
     │       │          ├─ layout.figure_regions() -> extract_image | crop -> OCREngine.read() -> _find_caption()
     │       │          └─ text blocks (heading detection, skip table/figure areas)
     │       └─ .pptx -> pptx_parser.parse_pptx()   (tables, pictures->OCR, charts->_chart_to_text, notes)
     ├─ make_chunks()  -> OntologyTagger.doc_type() + chunker.chunk_document()
     ├─ write_inventory()  -> outputs/e1_corpus_inventory.csv, e1_ingestion_stats.json
     └─ index.build_index()
         ├─ Embedder(auto).fit().encode()       # bge-small or TF-IDF/LSA
         ├─ vector_store.make_store().add()     # Chroma (+ NumPy copy)
         ├─ BM25Index(ids, texts)
         └─ chunks.jsonl, embedder.pkl, vectors.npy, bm25.pkl, manifest.json
""")]
    f += H2("2.2 uv run mmrag query \"...\"")
    f += [code("""
cli.cmd_query -> RAGSystem(cfg).answer(question, session, variant)
  1 DoSGuard.check            size / rate / burst                       -> refusal?
  2 InputGuard.check          injection, PII (block or redact)          -> refusal?
  3 QueryRouter.route + ConversationMemory.rewrite (follow-ups)
  4 AnswerCache.get           hit -> Answer(from_cache=True, 0 tokens)
  5 Retriever.search          filter -> dense + BM25 -> RRF -> boosts -> [rerank] -> guarantee
  6 InputGuard.out_of_scope                                              -> refusal?
  7 context.build_context     compressed (default) | naive
  8 PromptBuilder.build       zero_shot | fstring | three_shot | jinja2 | create | react | optimized
  9 LLM.generate              Ollama | OpenAI-compatible | Extractive (ReAct loop if variant=react)
 10 OutputGuard.check         citation, grounding, refusal, leakage      -> refusal?
 11 AnswerCache.put, ConversationMemory.add -> Answer
cli._print_answer(answer)
""")]
    f += H2("2.3 uv run mmrag run-all")
    f += [code("""
cli.cmd_run_all
 ├─ corpus.synth.generate()   (only if data/corpus/raw is empty)
 └─ eval.experiments.Experiments(cfg).run_all(force_ingest=True)
     E1 run_ingest            E2 make_chunks x3 + build_index(persist=False) + _retrieval_eval
     E3 Retriever.search with feature switches + run_questions (baseline vs final)
     E4 RAGSystem.answer over the dialog, use_memory True/False
     E5 run_questions(golden) -> summarise/by_modality; generate_synthetic; judge_answers -> agreement; failure_table
     E6 consistency (5 runs), guard on/off, DoS burst, naive vs compressed tokens, cache
     E7 run_questions per prompt variant + naive baseline + 5-run consistency
     E8 PromptOptimizer(dev, test).run()
     E9 guardrail_tests.jsonl, ontology routing, rendered prompt
     E10 optimized variant + benchmark table + demo answers
     build_report_tables() -> outputs/REPORT_TABLES.md ; run_status.json
""")]

    f += H1("3. Module-by-module walkthrough")
    sections = [
        ("3.1 config.py - configuration", "src/mmrag/config.py",
         "Loads config/config.yaml and config/ontology.yaml, reads a .env file without extra dependencies and expands "
         "${ENV_VAR} placeholders (used for the OpenAI-compatible endpoint). <font face='Courier'>Config.get('a.b')</font> "
         "gives dotted access; <font face='Courier'>clone(context__compress=False)</font> creates a modified copy for "
         "experiments.", [("Config.get", 10), ("load_config", 10)]),
        ("3.2 schemas.py - data structures", "src/mmrag/schemas.py",
         "Element (parsed unit), Chunk (indexed unit + metadata), Hit (chunk + scores) and Answer (final response with "
         "citations, tokens, latency, per-stage timings). Answer.to_dict() is what --json prints and what the cache stores.",
         [("Answer", 22)]),
        ("3.3 ingest/ocr.py - OCR", "src/mmrag/ingest/ocr.py",
         "OCREngine picks RapidOCR, then Tesseract, then none. Recognised boxes are sorted into reading-order lines "
         "(_join_lines) and glued words are split by repair_spacing.", [("OCREngine._rapid", 16), ("repair_spacing", 8)]),
        ("3.4 ingest/layout.py - figure regions", "src/mmrag/ingest/layout.py",
         "The heuristic returns raster-image rectangles (ignoring icons) plus one region that unions vector drawings when "
         "they cover 8-90 % of the page and are not already an image. With YOLO weights, figure/picture/chart boxes from "
         "the model are used instead. crop() renders only the region.", [("LayoutDetector._heuristic", 26)]),
        ("3.5 ingest/pdf_parser.py - PDFs", "src/mmrag/ingest/pdf_parser.py",
         "Per page: tables first (their rectangles are remembered), then figure regions (original image bytes via xref, "
         "duplicate-logo filter, upscale, OCR, caption), then text blocks that do not overlap tables or figures. A block "
         "whose font is >= 1.15 x the page's body size becomes the current section.",
         [("_table_rows", 20), ("parse_pdf", 40)]),
        ("3.6 ingest/pptx_parser.py - PowerPoint", "src/mmrag/ingest/pptx_parser.py",
         "Iterates shapes recursively (groups). Tables -> Markdown, charts -> series data text, pictures -> OCR with the "
         "slide title as caption, text frames -> text, notes -> 'Speaker notes:' text.", [("_chart_to_text", 16)]),
        ("3.7 ingest/chunker.py and metadata.py - chunks and tags", "src/mmrag/ingest/chunker.py",
         "_pack builds ~220-token chunks from sentences with a 40-token overlap; tiny groups (lone headings) are merged; "
         "tables are split by rows with the header repeated. Chunk ids look like <font face='Courier'>doc-p4-i0</font> "
         "(x = text, t = table, i = image). OntologyTagger counts keyword hits (substring for long words/phrases, whole "
         "word for short tokens) to assign doc_type and topics.", [("_pack", 22), ("_split_table", 12)]),
        ("3.8 ingest/pipeline.py - orchestration", "src/mmrag/ingest/pipeline.py",
         "parse_corpus caches parsed elements keyed by a corpus signature (file names + sizes), so experiments re-chunk "
         "without re-OCR. One failing file never stops ingestion.", [("run_ingest", 14)]),
        ("3.9 index/ - embeddings, vector store, BM25", "src/mmrag/index/__init__.py",
         "build_index embeds 'title | section | modality: text', writes to Chroma (or NumPy), keeps an exact NumPy copy, "
         "builds BM25 over text + title + section and stores a manifest whose version hash keys the answer cache.",
         [("build_index", 34)]),
        ("", "src/mmrag/index/vector_store.py", "NumpyStore: exact cosine with metadata masks and deterministic ordering.",
         [("NumpyStore.query", 14)]),
        ("", "src/mmrag/index/bm25.py", "Tokenizer shared by BM25, the extractive answerer and grounding: lower-case "
                                        "identifiers kept whole and also split on hyphens, light stemming of plain words.",
         [("tokenize", 14)]),
        ("3.10 retrieval/router.py - ontology routing", "src/mmrag/retrieval/router.py",
         "Extracts identifiers with the ENTITY regex, scores topics and doc types with the ontology keywords and picks a "
         "modality from cue words (image cues win over table cues).", [("QueryRouter.route", 22)]),
        ("3.11 retrieval/retriever.py - hybrid retrieval", "src/mmrag/retrieval/retriever.py",
         "The heart of Task A2: metadata pre-filter, dense + sparse search, reciprocal-rank fusion, modality and memory "
         "boosts, optional cross-encoder and the modality guarantee.", [("Retriever.search", 95)]),
        ("3.12 memory/ - conversation and cache", "src/mmrag/memory/conversation.py",
         "Follow-up detection and rule-based rewriting; the LLM rewrite is used when memory.rewrite is 'llm' and a "
         "generative model is available.", [("ConversationMemory.rewrite", 22)]),
        ("3.13 llm/backends.py - generators", "src/mmrag/llm/backends.py",
         "OllamaLLM calls /api/chat with temperature, seed, top_k=1 and num_predict; token counts come from Ollama's "
         "prompt_eval_count / eval_count. OpenAICompatLLM calls /chat/completions. ExtractiveLLM selects the best "
         "sentence, OCR line or table cell (row by identifier, column by the header the question names), down-weighting "
         "tokens that appear in most contexts.", [("OllamaLLM.generate", 14), ("make_llm", 18)]),
        ("3.14 prompting/context.py - context compression", "src/mmrag/prompting/context.py",
         "For each hit keeps only the query-relevant sentences, table rows or OCR lines (top 3/3/4), caps tokens per "
         "chunk and per context. The naive mode sends whole pages and charges image tokens.", [("compress", 30)]),
        ("3.15 prompting/variants.py - prompt variants", "src/mmrag/prompting/variants.py",
         "One builder renders all variants; Jinja2 templates live in prompts/ and receive the route (ontology), history "
         "and demos.", [("PromptBuilder.build", 34)]),
        ("", "prompts/answer_jinja2.j2", "The ontology-driven template:", [(None, 34)]),
        ("3.16 prompting/optimizer.py - self-improving prompt", "src/mmrag/prompting/optimizer.py",
         "Iterative search over instructions and demos scored on the dev split; see Architecture section 7.2.",
         [("PromptOptimizer.run", 45)]),
        ("3.17 guardrails/ - DoS, input, output", "src/mmrag/guardrails/dos.py", "Token bucket + burst detection:",
         [("DoSGuard.check", 22)]),
        ("", "src/mmrag/guardrails/input_guard.py", "Injection patterns, PII with Luhn check, out-of-scope rule:",
         [("InputGuard.check", 14), ("InputGuard.out_of_scope", 6)]),
        ("", "src/mmrag/guardrails/output_guard.py", "Citation validation/repair and grounding:",
         [("OutputGuard.check", 40)]),
        ("3.18 pipeline/rag.py - the end-to-end query path", "src/mmrag/pipeline/rag.py",
         "Orchestrates everything above and records per-stage timings (guard_dos, guard_input, memory_rewrite, retrieval, "
         "guard_oos, context_prompt, llm, guard_output, total) that E6 uses for FR3/FR4.", [("RAGSystem.answer", 150)]),
        ("3.19 eval/ - metrics, runner, automatic evaluation, failures", "src/mmrag/eval/metrics.py",
         "answer_correct: word-boundary key match tolerant to unit spacing; retrieval_summary: Recall@k and MRR.",
         [("answer_correct", 16), ("retrieval_summary", 8)]),
        ("", "src/mmrag/eval/runner.py", "run_questions measures retrieval on the top-5 list and answers with cache and "
                                         "memory disabled, per question:", [("run_questions", 34)]),
        ("", "src/mmrag/eval/auto_eval.py", "Reference-free heuristic judge (the LLM judge uses the same rubric):",
         [("heuristic_judge", 26)]),
        ("", "src/mmrag/eval/failure.py", "Automatic diagnosis of misses:", [("diagnose", 16)]),
        ("3.20 eval/experiments.py - E1 to E10", "src/mmrag/eval/experiments.py",
         "Each experiment is a method writing CSV/JSON/Markdown; run_all executes them in order, isolates failures and "
         "builds REPORT_TABLES.md. Example - the non-functional tests:", [("Experiments.e6", 60)]),
        ("3.21 corpus/synth.py - synthetic corpus", "src/mmrag/corpus/synth.py",
         "Each document kind returns a DocSpec with three facts (text / table / image). render_figure draws diagrams or "
         "charts whose labels carry the image fact; write_pdf / write_pptx lay them out on known pages so the golden set "
         "knows document, page and modality.", [("kind_fronthaul", 40)]),
        ("3.22 cli.py - commands", "src/mmrag/cli.py", "argparse sub-commands mapped to functions:",
         [("main", 30)]),
    ]
    for title, rel, text, snips in sections:
        if title:
            f += H2(title)
        f += [P(f"<font face='Courier' size='8.5'>{rel}</font> - {text}")]
        for name, n in snips:
            try:
                src = snippet(rel, name, n)
            except StopIteration:
                continue
            f += [code(src)]
    f += H1("4. Extension points")
    f += [table([["To ...", "Change"],
                 ["add a file type (e.g. .docx)", "write parse_docx() returning (elements, stats) like parse_pdf; register the "
                                                 "suffix in ingest/pipeline.py SUPPORTED and the dispatch in parse_corpus"],
                 ["use YOLO layout detection", "uv sync --extra yolo; set ingest.layout.detector: yolo and yolo_weights: <path>.pt"],
                 ["add a VLM caption for diagrams", "in pdf_parser figure loop, call a vision model (e.g. Ollama qwen2.5vl) and "
                                                     "append the caption to the element content"],
                 ["change the embedding model", "embedding.model in config.yaml, then mmrag ingest --force"],
                 ["add a prompt variant", "add a template in prompts/, a branch in PromptBuilder.build and its name to VARIANTS / "
                                          "ABLATION_VARIANTS"],
                 ["add a guardrail", "add a pattern to INJECTION / PII in input_guard.py or a rule in OutputGuard.check; add a "
                                     "row to data/golden/guardrail_tests.jsonl"],
                 ["recognise new identifiers", "extend ENTITY in retrieval/router.py (e.g. r'TS ?\\d{2}\\.\\d{3}')"],
                 ["new domain", "edit config/ontology.yaml (doc_types, topics, cues) and the DOMAIN / USER_ROLE constants in "
                                "prompting/variants.py"]], [5 * cm, 12.4 * cm], 8)]
    doc.multiBuild(f)


if __name__ == "__main__":
    out = ROOT / "docs"
    run_guide(out / "1_Run_Guide.pdf")
    architecture(out / "2_Architecture.pdf")
    walkthrough(out / "3_Code_Walkthrough.pdf")
    for p in sorted(out.glob("*.pdf")):
        print(p, os.path.getsize(p) // 1024, "KB")
