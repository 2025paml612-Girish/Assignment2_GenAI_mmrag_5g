"""Demo UI:  uv run streamlit run app/streamlit_app.py   (needs: uv sync --extra ui)"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from mmrag.config import load_config  # noqa: E402
from mmrag.pipeline.rag import RAGSystem  # noqa: E402
from mmrag.prompting.variants import VARIANTS  # noqa: E402

st.set_page_config(page_title="5G Multimodal Document Assistant", layout="wide")


@st.cache_resource
def get_rag():
    return RAGSystem(load_config())


rag = get_rag()
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

q = st.chat_input("Ask about the indexed PDFs / PPTs ...")
if q:
    a = rag.answer(q, session="ui", variant=variant, use_cache=use_cache, use_guardrails=guards)
    st.session_state["log"].append({"q": q, "a": a.answer})
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a.answer)
        for c in a.citations:
            st.markdown(f"- **[{c['n']}]** `{c['source']}` page {c['page']} ({c['modality']})")
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
        with st.expander("Timings"):
            st.json({k: round(v, 4) for k, v in a.timings.items()})
