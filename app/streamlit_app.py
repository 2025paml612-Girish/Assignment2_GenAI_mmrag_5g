"""Demo UI:  uv run streamlit run app/streamlit_app.py   (needs: uv sync --extra ui)"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from mmrag.config import load_config  # noqa: E402
from mmrag.ingest.figures import figure_png  # noqa: E402
from mmrag.pipeline.rag import RAGSystem  # noqa: E402
from mmrag.prompting.variants import VARIANTS  # noqa: E402

st.set_page_config(page_title="5G Multimodal Document Assistant", layout="wide")


@st.cache_resource
def get_rag():
    return RAGSystem(load_config())


rag = get_rag()


@st.cache_data(show_spinner=False)
def get_figure(chunk_id: str) -> bytes | None:
    """Render the figure behind an image chunk from its source PDF/PPTX (cached per chunk)."""
    chunk = rag.index.chunks.get(chunk_id)
    return figure_png(rag.cfg, chunk) if chunk is not None else None


def show_figures(chunk_ids: list[str]) -> None:
    for cid in dict.fromkeys(chunk_ids):            # keep order, drop duplicates
        png = get_figure(cid)
        if png:
            ch = rag.index.chunks[cid]
            st.image(png, caption=f"{ch.source} - page {ch.page}", width=700)
st.title("5G / O-RAN Multimodal Document Assistant")
st.caption(f"LLM: {rag.llm.name} | embedder: {rag.index.embedder.name} | chunks: {len(rag.index.chunks)}")

with st.sidebar:
    variant = st.selectbox("Prompt variant", VARIANTS, index=VARIANTS.index("jinja2"))
    use_cache = st.checkbox("Answer cache", value=True)
    guards = st.checkbox("Guardrails", value=True)
    if st.button("Reset conversation"):
        rag.memory.clear("ui")
        st.session_state["log"] = []

st.session_state.setdefault("log", [])
for turn in st.session_state["log"]:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])
        show_figures(turn.get("figs", []))

q = st.chat_input("Ask about the indexed PDFs / PPTs ...")
if q:
    a = rag.answer(q, session="ui", variant=variant, use_cache=use_cache, use_guardrails=guards)
    cited_figs = [c["chunk_id"] for c in a.citations if c.get("modality") == "image" and c.get("chunk_id")]
    st.session_state["log"].append({"q": q, "a": a.answer, "figs": cited_figs})
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a.answer)
        for c in a.citations:
            st.markdown(f"- **[{c['n']}]** `{c['source']}` page {c['page']} ({c['modality']})")
        show_figures(cited_figs)                     # the actual figure(s) the answer cites
        cols = st.columns(4)
        cols[0].metric("input tokens", a.input_tokens)
        cols[1].metric("output tokens", a.output_tokens)
        cols[2].metric("latency (s)", f"{a.latency_s:.2f}")
        cols[3].metric("grounding", f"{a.grounding:.2f}")
        if a.rewritten_query != q:
            st.info(f"Follow-up rewritten to: {a.rewritten_query}")
        if a.refused:
            st.warning(a.refusal_reason)
        with st.expander("Retrieved chunks"):
            for h in a.hits:
                st.markdown(f"**{h.chunk.chunk_id}** - {h.chunk.modality} - score {h.score:.3f}")
                st.code(h.chunk.text[:700])
                if h.chunk.modality == "image":
                    png = get_figure(h.chunk.chunk_id)
                    if png:
                        st.image(png, width=420)
        with st.expander("Timings"):
            st.json({k: round(v, 4) for k, v in a.timings.items()})
