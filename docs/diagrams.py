"""Generate the architecture diagrams used in the PDFs (needs Graphviz `dot`).

    uv run python docs/diagrams.py
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, GREEN, ORANGE, GREY, PURPLE, RED = "#dbe8f6", "#e3f1e3", "#fdebd3", "#eeeeee", "#ece3f5", "#f8dede"
FONT = 'fontname="Helvetica"'


def dot(name: str, src: str) -> Path:
    p = OUT / f"{name}.png"
    subprocess.run(["dot", "-Tpng", "-Gdpi=170", "-o", str(p)], input=src.encode(), check=True)
    return p


def hld():
    return dot("hld", f"""
digraph G {{ rankdir=TB; compound=true; newrank=true; {FONT}; fontsize=12; nodesep=0.25; ranksep=0.45;
 node [shape=box, style="rounded,filled", {FONT}, fontsize=11, fillcolor="{BLUE}"]; edge [{FONT}, fontsize=9];
 subgraph cluster_ing {{ label="INGESTION PATH (offline, uv run mmrag ingest)"; style="rounded,dashed"; color="#1f4e79";
   corpus [label="Corpus\\n>=100 PDF / PPT / PPTX", shape=folder, fillcolor="{GREY}"];
   parse [label="Parsers\\nPyMuPDF (text, find_tables)\\npython-pptx (text, tables,\\npictures, charts, notes)"];
   layout [label="Layout detector\\nraster + vector regions\\n(optional YOLO)", fillcolor="{ORANGE}"];
   ocr [label="OCR\\nRapidOCR / Tesseract\\n(only figure crops)", fillcolor="{ORANGE}"];
   chunk [label="Chunker\\nrecursive 220 tok / 40 overlap\\ntable rows / figure chunks"];
   meta [label="Ontology tagger\\ndoc_type, topic, modality,\\npage, section, date", fillcolor="{PURPLE}"];
   emb [label="Embedder\\nbge-small-en-v1.5\\n(TF-IDF/LSA fallback)"];
   corpus -> parse -> layout -> ocr -> chunk -> meta -> emb; parse -> chunk [constraint=false];
   {{rank=same; corpus; parse; layout; ocr; chunk; meta; emb;}}
 }}
 subgraph cluster_store {{ label="INDEX (data/index)"; style="rounded"; color="#2e7d32";
   vdb [label="Vector store\\nChromaDB cosine HNSW\\n(+ NumPy exact copy)", shape=cylinder, fillcolor="{GREEN}"];
   bm25 [label="BM25 index", shape=cylinder, fillcolor="{GREEN}"];
   cstore [label="Chunk store\\nchunks.jsonl + manifest", shape=cylinder, fillcolor="{GREEN}"];
 }}
 emb -> vdb; emb -> bm25 [label="tokens"]; emb -> cstore; {{rank=same; vdb; bm25; cstore;}}
 subgraph cluster_q {{ label="QUERY PATH (online, mmrag query / chat / UI)"; style="rounded,dashed"; color="#b8860b";
   user [label="User query", shape=oval, fillcolor="{GREY}"];
   dos [label="DoS guard\\nsize, rate, burst,\\nconcurrency", fillcolor="{RED}"];
   ig [label="Input guard\\ninjection, PII", fillcolor="{RED}"];
   mem [label="Memory\\nfollow-up rewrite +\\nanswer cache", fillcolor="{PURPLE}"];
   route [label="Ontology router\\nentities, topics,\\nmodality, doc_type", fillcolor="{PURPLE}"];
   ret [label="Hybrid retriever\\nfilter -> dense + BM25 -> RRF\\n-> modality boost -> (re-rank)"];
   oos [label="Out-of-scope\\ncheck", fillcolor="{RED}"];
   ctx [label="Context compression\\nrelevant sentences / rows /\\nOCR lines, token budget"];
   prm [label="Prompt builder\\nJinja2 + ontology\\n(7 variants)"];
   llm [label="LLM\\nOllama / OpenAI-compat\\n(extractive fallback)", fillcolor="{ORANGE}"];
   og [label="Output guard\\ncitation, grounding,\\nrefusal, leakage", fillcolor="{RED}"];
   ans [label="Answer + citations\\n(file, page, modality)", shape=oval, fillcolor="{GREY}"];
   user -> dos -> ig -> mem -> route -> ret; ret -> oos; ans -> og -> llm -> prm -> ctx -> oos [dir=back];
   og -> mem [style=dashed, label="store turn", constraint=false];
   {{rank=same; user; dos; ig; mem; route; ret;}}
   {{rank=same; ans; og; llm; prm; ctx; oos;}}
 }}
 vdb -> ret [style=bold]; bm25 -> ret [style=bold]; cstore -> ret [style=bold];
}}""")


def ingestion_flow():
    return dot("ingestion_flow", f"""
digraph G {{ rankdir=TB; {FONT}; node [shape=box, style="rounded,filled", fillcolor="{BLUE}", {FONT}, fontsize=10]; edge [{FONT}, fontsize=9];
 a [label="cli.cmd_ingest / Experiments.e1"]; b [label="ingest.pipeline.run_ingest(cfg)"];
 c [label="parse_corpus(): walk corpus_dir, cache check (corpus_signature.txt)"];
 d1 [label="pdf_parser.parse_pdf()\\nfind_tables -> word-box cells -> Markdown\\nLayoutDetector.figure_regions -> crop -> OCREngine.read\\ntext blocks (skip table / figure areas), headings"];
 d2 [label="pptx_parser.parse_pptx()\\ntitle, text frames, tables, pictures (OCR),\\nnative charts (series data), speaker notes\\n(.ppt -> .pptx via LibreOffice)"];
 e [label="elements.jsonl + preprocessing_record.jsonl", shape=note, fillcolor="{GREY}"];
 f [label="make_chunks(): OntologyTagger.doc_type + chunker.chunk_document()\\n(text: recursive / page, tables: row groups, images: caption + OCR)"];
 g [label="write_inventory(): e1_corpus_inventory.csv, e1_ingestion_stats.json", shape=note, fillcolor="{GREY}"];
 h [label="index.build_index(): Embedder.fit/encode -> make_store().add ->\\nBM25Index -> chunks.jsonl, embedder.pkl, vectors.npy, bm25.pkl, manifest.json"];
 a -> b -> c; c -> d1 [label=".pdf"]; c -> d2 [label=".pptx/.ppt"]; d1 -> e; d2 -> e; e -> f -> g; f -> h;
}}""")


def query_flow():
    return dot("query_flow", f"""
digraph G {{ rankdir=TB; {FONT}; node [shape=box, style="rounded,filled", fillcolor="{BLUE}", {FONT}, fontsize=10]; edge [{FONT}, fontsize=9];
 s [label="RAGSystem.answer(query, session, variant)", fillcolor="{GREY}"];
 g1 [label="DoSGuard.check()", fillcolor="{RED}"]; g2 [label="InputGuard.check()  injection / PII", fillcolor="{RED}"];
 m [label="QueryRouter.route() + ConversationMemory.rewrite()", fillcolor="{PURPLE}"];
 c [label="AnswerCache.get(key)", fillcolor="{PURPLE}"];
 r [label="Retriever.search(): entity/topic filter -> store.query + bm25.query -> RRF\\n-> modality boost -> memory doc boost -> CrossEncoder (opt.) -> modality guarantee"];
 o [label="InputGuard.out_of_scope(domain_hits, max_dense, max_sparse)", fillcolor="{RED}"];
 x [label="context.build_context()  (compressed | naive)"];
 p [label="PromptBuilder.build(variant, ...)  zero_shot | fstring | three_shot | jinja2 | create | react | optimized"];
 l [label="LLM.generate()   (ReAct: up to 2 search[...] tool calls)", fillcolor="{ORANGE}"];
 og [label="OutputGuard.check(): citations, grounding >= 0.35, refusal, leakage", fillcolor="{RED}"];
 a [label="Answer(answer, citations, hits, tokens, latency, timings)", fillcolor="{GREY}"];
 rf [label="refusal Answer", shape=octagon, fillcolor="{RED}"];
 s -> g1 -> g2 -> m -> c; c -> a [label="hit (0 tokens)", style=dashed]; c -> r [label="miss"]; r -> o -> x -> p -> l -> og -> a;
 g1 -> rf [style=dotted]; g2 -> rf [style=dotted]; o -> rf [style=dotted]; og -> rf [style=dotted, label="no evidence"];
 a -> m [label="memory.add + cache.put", style=dashed];
}}""")


def module_graph():
    """Import graph of src/mmrag computed from the source (AST) - which module calls which."""
    pkg = ROOT / "src" / "mmrag"
    mods = {}
    for f in pkg.rglob("*.py"):
        rel = f.relative_to(pkg).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        mods[".".join(parts) or "mmrag"] = f
    edges = set()
    for name, f in mods.items():
        tree = ast.parse(f.read_text(encoding="utf-8"))
        base = name.split(".")
        is_pkg = f.name == "__init__.py"
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                anchor = base if is_pkg else base[:-1]
                anchor = anchor[: len(anchor) - (node.level - 1)] if node.level > 1 else anchor
                target = ".".join(anchor + (node.module.split(".") if node.module else []))
                cands = [target] + [f"{target}.{a.name}" for a in node.names]
                for t in cands:
                    if t in mods and t != name:
                        edges.add((name, t))
    skip = {"mmrag", "utils", "utils.log", "utils.tokens", "schemas", "config"}
    groups: dict[str, list[str]] = {}
    for m in mods:
        if m in skip or m.endswith("__init__"):
            continue
        groups.setdefault(m.split(".")[0], []).append(m)
    colors = {"ingest": ORANGE, "index": GREEN, "retrieval": BLUE, "memory": PURPLE, "llm": ORANGE, "prompting": BLUE,
              "guardrails": RED, "pipeline": GREY, "eval": GREEN, "corpus": GREY, "cli": GREY}
    lines = [f'digraph G {{ rankdir=LR; {FONT}; ranksep=0.9; nodesep=0.12; node [shape=box, style="rounded,filled", {FONT}, fontsize=9];'
             f' edge [color="#555555", arrowsize=0.6];']
    used = {x for e in edges for x in e if x not in skip}
    groups = {g: [m for m in ms if m in used] for g, ms in groups.items()}
    groups = {g: ms for g, ms in groups.items() if ms}
    for g, ms in sorted(groups.items()):
        lines.append(f'subgraph cluster_{g} {{ label="{g}"; style="rounded"; color="#999999"; fontsize=10;')
        for m in sorted(ms):
            lines.append(f'"{m}" [label="{m.split(".")[-1] if "." in m else m}", fillcolor="{colors.get(g, BLUE)}"];')
        lines.append("}")
    for a, b in sorted(edges):
        if a in skip or b in skip:
            continue
        if any(a in v for v in groups.values()) and any(b in v for v in groups.values()):
            lines.append(f'"{a}" -> "{b}";')
    lines.append("}")
    return dot("module_graph", "\n".join(lines)), sorted(edges)


def class_diagram():
    return dot("class_diagram", f"""
digraph G {{ rankdir=TB; {FONT}; node [shape=record, style=filled, fillcolor="{BLUE}", {FONT}, fontsize=9]; edge [{FONT}, fontsize=8, arrowhead=vee];
 RAG [label="{{RAGSystem|cfg, index, router, retriever, llm, prompts,\\ldos, inguard, outguard, memory, cache\\l|answer(query, session, variant, ...) : Answer\\l_remember(...)\\l}}", fillcolor="{GREY}"];
 IDX [label="{{Index|chunks: dict, embedder, store, bm25, manifest\\l|load(index_dir) : Index\\lversion : str\\l}}", fillcolor="{GREEN}"];
 EMB [label="{{Embedder|backend: st \\| tfidf, model\\l|fit(texts, dims)\\lencode(texts, is_query) : ndarray\\lsave / load\\l}}", fillcolor="{GREEN}"];
 VS [label="{{ChromaStore \\| NumpyStore|add(ids, vectors, metas)\\lquery(vec, k, where) : [(id, score)]\\lcount()\\l}}", fillcolor="{GREEN}"];
 BM [label="{{BM25Index|query(text, k, allowed) : [(id, score)]\\l}}", fillcolor="{GREEN}"];
 RET [label="{{Retriever|index, router, reranker\\l|search(query, route, k, modalities,\\l  use_filters, use_sparse, boost_docs)\\l  : (hits, info)\\lentity_docs(entities)\\l}}"];
 RT [label="{{QueryRouter|route(query) : Route\\l}}|{{Route|topics, doc_type, modality,\\lentities, domain_hits\\l}}", fillcolor="{PURPLE}"];
 MEM [label="{{ConversationMemory|history / add / clear\\lrewrite(session, q, entities, llm)\\l  : (standalone, boost_docs, followup)\\l}}", fillcolor="{PURPLE}"];
 CAC [label="{{AnswerCache (SQLite)|key(q, variant, index_ver, llm)\\lget / put / clear\\l}}", fillcolor="{PURPLE}"];
 PB [label="{{PromptBuilder|build(variant, question, contexts,\\l  route, history, instruction, demos)\\l  : messages\\l}}"];
 CTX [label="{{context|build_context(hits, q, cfg, mode)\\l  : ([ContextItem], stats)\\lcompress(hit, q, max_tokens)\\l}}"];
 LLM [label="{{LLM (Ollama \\| OpenAICompat \\| Extractive)|generate(messages, max_tokens, temp,\\l  stop, **meta) : LLMResponse\\l}}", fillcolor="{ORANGE}"];
 G [label="{{DoSGuard|check(session, q)\\l}}|{{InputGuard|check(q)\\lout_of_scope(...)\\l}}|{{OutputGuard|check(answer, contexts)\\l}}", fillcolor="{RED}"];
 OPT [label="{{PromptOptimizer|evaluate(instruction, demos, qs)\\lrun(out_dir) : log\\l}}"];
 EXP [label="{{Experiments|e1 ... e10()\\lrun_all()\\l}}", fillcolor="{GREY}"];
 RAG -> IDX; RAG -> RET; RAG -> MEM; RAG -> CAC; RAG -> PB; RAG -> CTX; RAG -> LLM; RAG -> G;
 RET -> IDX; RET -> RT; IDX -> EMB; IDX -> VS; IDX -> BM; EXP -> RAG; EXP -> OPT; OPT -> RAG;
}}""")


def sequence_diagram():
    actors = ["User", "RAGSystem", "Guards", "Memory", "Router", "Retriever", "Index", "Prompt", "LLM"]
    msgs = [
        (0, 1, "answer(q, session)"), (1, 2, "DoS.check + Input.check"), (2, 1, "allowed / redacted q"),
        (1, 3, "rewrite(q) + cache.get"), (3, 1, "standalone q, boost docs"), (1, 4, "route(q)"),
        (4, 1, "Route(entities, topics, modality)"), (1, 5, "search(q, route)"), (5, 6, "where filter; dense + BM25"),
        (6, 5, "candidates"), (5, 1, "top-k hits (RRF, boosts)"), (1, 2, "out_of_scope()"),
        (1, 7, "build_context + build(variant)"), (7, 1, "messages"), (1, 8, "generate(messages)"),
        (8, 1, "text + token usage"), (1, 2, "OutputGuard.check()"), (2, 1, "cited, grounded answer"),
        (1, 3, "memory.add + cache.put"), (1, 0, "Answer(answer, citations, tokens, latency)"),
    ]
    fig, ax = plt.subplots(figsize=(12, 8.2), dpi=150)
    xs = [i * 1.35 for i in range(len(actors))]
    top = len(msgs) + 1
    for x, a in zip(xs, actors):
        ax.add_patch(plt.Rectangle((x - 0.55, top), 1.1, 0.7, fc=BLUE, ec="#1f4e79"))
        ax.text(x, top + 0.35, a, ha="center", va="center", fontsize=8.5, weight="bold")
        ax.plot([x, x], [0.3, top], color="#999999", ls="--", lw=0.8)
    for i, (a, b, label) in enumerate(msgs):
        y = top - 0.6 - i
        ax.annotate("", xy=(xs[b], y), xytext=(xs[a], y),
                    arrowprops=dict(arrowstyle="->", lw=1.1, color="#b22222" if b < a else "#1f4e79",
                                    ls="--" if b < a else "-"))
        ax.text((xs[a] + xs[b]) / 2, y + 0.12, f"{i + 1}. {label}", ha="center", fontsize=7)
    ax.set_xlim(-0.8, xs[-1] + 0.8)
    ax.set_ylim(0, top + 0.9)
    ax.axis("off")
    fig.tight_layout()
    p = OUT / "sequence.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def experiments_flow():
    return dot("experiments_flow", f"""
digraph G {{ rankdir=TB; {FONT}; node [shape=box, style="rounded,filled", fillcolor="{BLUE}", {FONT}, fontsize=9]; edge [arrowsize=0.6];
 ra [label="mmrag run-all\\nExperiments.run_all()", fillcolor="{GREY}"];
 e1 [label="E1 corpus +\\ningestion"]; e2 [label="E2 embedding +\\nchunking configs"]; e3 [label="E3 text-only vs\\nmultimodal retrieval"];
 e4 [label="E4 memory\\nmulti-turn"]; e5 [label="E5 golden + auto eval\\n+ failure analysis"]; e6 [label="E6 consistency, P95,\\nguard overhead, tokens"];
 e7 [label="E7 prompt ablation\\n(7 variants)"]; e8 [label="E8 self-improving\\nprompt (dev/test)"]; e9 [label="E9 guardrails +\\nontology"]; e10 [label="E10 benchmark +\\ndemo"];
 rep [label="outputs/REPORT_TABLES.md", shape=note, fillcolor="{GREEN}"];
 ra -> e1 -> e2 -> e3 -> e4 -> e5; e5 -> e6; rep -> e10 -> e9 -> e8 -> e7 -> e6 [dir=back];
 {{rank=same; ra; e1; e2; e3; e4; e5;}} {{rank=same; rep; e10; e9; e8; e7; e6;}}
}}""")


def main():
    hld()
    ingestion_flow()
    query_flow()
    p, edges = module_graph()
    class_diagram()
    sequence_diagram()
    experiments_flow()
    (OUT / "module_edges.txt").write_text("\n".join(f"{a} -> {b}" for a, b in edges))
    print("diagrams written to", OUT)


if __name__ == "__main__":
    main()
