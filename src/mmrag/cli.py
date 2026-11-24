"""Command line interface:  uv run mmrag <command>

  doctor           check which optional components are available
  generate-corpus  create the synthetic 120-document 5G corpus + golden set
  ingest           parse PDFs/PPTX, OCR, chunk, embed, build the index (E1)
  query "..."      ask one question
  chat             interactive multi-turn chat (short-term memory)
  evaluate         golden-set evaluation only (E5)
  experiments      run E1-E10 (or --only E3,E7) and build outputs/REPORT_TABLES.md
  run-all          generate corpus (if empty) + ingest + all experiments
  golden-template  sample chunks into a CSV to help curate a golden set for a real corpus
  report           rebuild outputs/REPORT_TABLES.md from existing results
"""
from __future__ import annotations

import argparse
import json
import random
import sys

from .config import load_config


def _cfg(args):
    cfg = load_config(args.config)
    if getattr(args, "llm", None):
        cfg.set("llm.backend", args.llm)
    if getattr(args, "model", None):
        cfg.set("llm.ollama.model", args.model)
    return cfg


def cmd_doctor(args):
    cfg = _cfg(args)
    from .ingest.ocr import OCREngine
    from .llm.backends import OllamaLLM
    from .utils.tokens import tokenizer_name

    rows = []
    try:
        import sentence_transformers  # noqa: F401
        rows.append(("sentence-transformers", "installed"))
    except Exception:
        rows.append(("sentence-transformers", "missing -> TF-IDF/LSA fallback (uv sync --extra ml)"))
    try:
        import chromadb  # noqa: F401
        rows.append(("chromadb", "installed"))
    except Exception:
        rows.append(("chromadb", "missing -> numpy store (uv sync --extra chroma)"))
    rows.append(("OCR engine", OCREngine(cfg.get("ingest.ocr.engine")).name))
    host = cfg.get("llm.ollama.host")
    rows.append(("Ollama", f"reachable at {host}" if OllamaLLM.available(host) else f"not reachable at {host}"))
    rows.append(("OpenAI-compatible endpoint", cfg.get("llm.openai_compat.base_url") or "not configured"))
    rows.append(("tokenizer", tokenizer_name()))
    corpus = cfg.path("corpus_dir")
    n = len([p for p in corpus.rglob("*") if p.suffix.lower() in (".pdf", ".pptx", ".ppt")]) if corpus.exists() else 0
    rows.append(("documents in corpus", f"{n} in {corpus}"))
    rows.append(("index", "present" if (cfg.path("index_dir") / "manifest.json").exists() else "not built"))
    for k, v in rows:
        print(f"  {k:<28} {v}")


def cmd_generate(args):
    cfg = _cfg(args)
    from .corpus.synth import generate

    print(generate(cfg.path("corpus_dir"), cfg.path("golden_set").parent, args.n, cfg.get("project.seed", 42)))


def cmd_ingest(args):
    cfg = _cfg(args)
    from .ingest.pipeline import run_ingest

    _, stats = run_ingest(cfg, force=args.force)
    print(json.dumps({k: v for k, v in stats.items() if k != "index"}, indent=2))


def _print_answer(a, as_json=False):
    if as_json:
        print(json.dumps(a.to_dict(), indent=2))
        return
    print(f"\nAnswer: {a.answer}")
    for c in a.citations:
        print(f"  [{c['n']}] {c['source']}  page {c['page']}  ({c['modality']})")
    if a.rewritten_query != a.query:
        print(f"  (standalone query: {a.rewritten_query})")
    print(f"  tokens in/out: {a.input_tokens}/{a.output_tokens}  latency: {a.latency_s:.2f}s"
          f"{'  [cache]' if a.from_cache else ''}{'  [refused: ' + a.refusal_reason + ']' if a.refused else ''}")


def cmd_query(args):
    cfg = _cfg(args)
    from .pipeline.rag import RAGSystem

    rag = RAGSystem(cfg)
    _print_answer(rag.answer(args.question, session=args.session, variant=args.variant, use_cache=not args.no_cache),
                  args.json)


def cmd_chat(args):
    cfg = _cfg(args)
    from .pipeline.rag import RAGSystem

    rag = RAGSystem(cfg)
    print("Multimodal 5G document assistant. Type a question; ':reset' clears memory; ':q' quits.")
    session = "chat"
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in (":q", "quit", "exit"):
            break
        if q == ":reset":
            rag.memory.clear(session)
            print("memory cleared")
            continue
        if q:
            _print_answer(rag.answer(q, session=session, variant=args.variant))


def cmd_evaluate(args):
    cfg = _cfg(args)
    from .eval.experiments import Experiments

    ex = Experiments(cfg, quick=args.quick)
    print(json.dumps(ex.e5(), indent=2, default=str))


def cmd_experiments(args):
    cfg = _cfg(args)
    from .eval.experiments import Experiments

    only = [s.strip().upper() for s in args.only.split(",")] if args.only else None
    Experiments(cfg, quick=args.quick).run_all(force_ingest=args.force_ingest, only=only)


def cmd_run_all(args):
    cfg = _cfg(args)
    corpus = cfg.path("corpus_dir")
    has_docs = corpus.exists() and any(p.suffix.lower() in (".pdf", ".pptx", ".ppt") for p in corpus.rglob("*"))
    if not has_docs and not args.no_generate:
        from .corpus.synth import generate

        print("Corpus folder is empty -> generating the synthetic 5G corpus")
        print(generate(corpus, cfg.path("golden_set").parent, 120, cfg.get("project.seed", 42)))
    from .eval.experiments import Experiments

    Experiments(cfg, quick=args.quick).run_all(force_ingest=True)


def cmd_golden_template(args):
    cfg = _cfg(args)
    import pandas as pd

    from .index import Index

    idx = Index.load(cfg.path("index_dir"), cfg.get("vector_store.backend"), cfg.get("vector_store.collection"))
    rnd = random.Random(42)
    rows = []
    for m in ("text", "table", "image"):
        pool = [c for c in idx.chunks.values() if c.modality == m]
        for c in rnd.sample(pool, min(args.per_modality, len(pool))):
            rows.append({"qid": "", "question": "", "answer": "", "key": "", "modality": m, "source": c.source,
                         "doc_id": c.doc_id, "page": c.page, "difficulty": "medium", "chunk_text": c.text[:500]})
    out = cfg.path("outputs_dir") / "golden_template.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}. Fill question/answer/key, then convert with: uv run mmrag golden-from-csv {out}")


def cmd_golden_from_csv(args):
    cfg = _cfg(args)
    import pandas as pd

    df = pd.read_csv(args.csv).fillna("")
    df = df[df.question.str.len() > 0]
    out = cfg.path("golden_set")
    with open(out, "w", encoding="utf-8") as f:
        for i, r in enumerate(df.itertuples(), 1):
            f.write(json.dumps({"qid": r.qid or f"Q{i:03d}", "question": r.question, "answer": r.answer,
                                "key": r.key or r.answer, "modality": r.modality, "source": r.source,
                                "doc_id": r.doc_id, "page": int(r.page), "difficulty": r.difficulty}) + "\n")
    print(f"Wrote {len(df)} golden questions to {out}")


def cmd_report(args):
    cfg = _cfg(args)
    from .eval.experiments import build_report_tables

    print(build_report_tables(cfg))


def main(argv=None):
    p = argparse.ArgumentParser(prog="mmrag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--llm", default=None, choices=["auto", "ollama", "openai_compat", "extractive"])
    p.add_argument("--model", default=None, help="Ollama model name, e.g. qwen2.5:7b-instruct")
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("doctor").set_defaults(fn=cmd_doctor)
    g = sp.add_parser("generate-corpus"); g.add_argument("--n", type=int, default=120); g.set_defaults(fn=cmd_generate)
    g = sp.add_parser("ingest"); g.add_argument("--force", action="store_true"); g.set_defaults(fn=cmd_ingest)
    g = sp.add_parser("query"); g.add_argument("question"); g.add_argument("--variant", default=None)
    g.add_argument("--session", default="cli"); g.add_argument("--json", action="store_true")
    g.add_argument("--no-cache", action="store_true"); g.set_defaults(fn=cmd_query)
    g = sp.add_parser("chat"); g.add_argument("--variant", default=None); g.set_defaults(fn=cmd_chat)
    g = sp.add_parser("evaluate"); g.add_argument("--quick", action="store_true"); g.set_defaults(fn=cmd_evaluate)
    g = sp.add_parser("experiments"); g.add_argument("--only", default=None); g.add_argument("--quick", action="store_true")
    g.add_argument("--force-ingest", action="store_true"); g.set_defaults(fn=cmd_experiments)
    g = sp.add_parser("run-all"); g.add_argument("--quick", action="store_true"); g.add_argument("--no-generate", action="store_true")
    g.set_defaults(fn=cmd_run_all)
    g = sp.add_parser("golden-template"); g.add_argument("--per-modality", type=int, default=15)
    g.set_defaults(fn=cmd_golden_template)
    g = sp.add_parser("golden-from-csv"); g.add_argument("csv"); g.set_defaults(fn=cmd_golden_from_csv)
    sp.add_parser("report").set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
