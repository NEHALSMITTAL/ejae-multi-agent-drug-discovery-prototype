import streamlit as st
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
import streamlit as st
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Set
import datetime
import re
import networkx as nx
from pyvis.network import Network

# Backend Imports
from backend.agents import literature_agent, evidence_scorer, experiment_recommender
from backend.active_learning import save_feedback, rerank_hypotheses
from backend.chem_utils import compute_similarity
from backend.knowledge_graph import (
    load_knowledge_graph,
    save_knowledge_graph,
    add_hypothesis_to_kg,
    build_dynamic_pathway_graph,
    draw_knowledge_graph_html,
    draw_pathway_graph_html
)

# ============================================================
# ROOT DIRECTORY + DATA FOLDER
# ============================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "kg" not in st.session_state:
    st.session_state.kg = load_knowledge_graph()

if "last_query" not in st.session_state:
    st.session_state.last_query = None

if "last_results" not in st.session_state:
    st.session_state.last_results = {}

# ============================================================
# CACHING FOR SPEED
# ============================================================

@st.cache_data(show_spinner=False)
def cached_literature_agent(query: str):
    return literature_agent(query)

@st.cache_data(show_spinner=False)
def cached_similarity(smi1: str, smi2: str):
    return compute_similarity(smi1, smi2)

# ============================================================
# STREAMLIT PAGE SETUP
# ============================================================
st.set_page_config(page_title="Ejae Drug Discovery Prototype", layout="wide")
st.title("🧬 Ejae Multi-Agent Drug Discovery Prototype")

query = st.text_input("Enter disease, pathway, or molecule:")
smiles_input = st.text_input("SMILES (optional):")

run = st.button("🔍 Run Analysis")

# ============================================================
# MAIN EXECUTION
# ============================================================
if run:

    st.session_state.last_query = query

    parsed, docs = cached_literature_agent(query)

    st.subheader("📄 Retrieved Documents")
    for d in docs:
        with st.expander(f"{d['id']} — {d['meta']['title']}"):
            st.write(d["text"])

    # =======================================================
    # MOLECULAR SIMILARITY
    # =======================================================
    if smiles_input.strip():
        st.subheader("🧪 Molecular Similarity (RDKit)")

        refs = {
            "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C",
            "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2)C"
        }

        for name, ref_smi in refs.items():
            sim = cached_similarity(smiles_input, ref_smi)
            if sim is not None:
                st.write(f"{name}: **{sim:.3f}**")

    # =======================================================
    # HYPOTHESES SECTION
    # =======================================================
    if "hypotheses" in parsed:

        parsed["hypotheses"] = rerank_hypotheses(parsed["hypotheses"])

        st.subheader("🧠 Hypotheses")

        for i, h in enumerate(parsed["hypotheses"]):
            st.markdown(f"### Hypothesis {i + 1}")
            st.write(h["text"])

            evidence_snips = [
                e.get("snippet", "")[:300] for e in h.get("evidence", [])
            ]

            for s in evidence_snips:
                st.write(f"- {s}")

            # Score
            score = evidence_scorer(h["text"], evidence_snips)
            st.info(f"Evidence Score: {score}")

            # Experiment Recommendation
            exp_text = experiment_recommender(h["text"], evidence_snips)
            st.success(exp_text)

            # Buttons
            col1, col2, col3 = st.columns(3)

            # Feedback buttons
            if col1.button(f"Accept {i+1}", key=f"acc_{i}_{h['text']}"):
                save_feedback(h["text"], True)
                st.success("Feedback saved ✔")

            if col2.button(f"Reject {i+1}", key=f"rej_{i}_{h['text']}"):
                save_feedback(h["text"], False)
                st.error("Feedback saved ✘")

            # Add to KG button
            if col3.button(f"➕ Add to KG {i+1}", key=f"kg_{i}_{h['text']}"):
                add_hypothesis_to_kg(st.session_state.kg, h["text"], h["evidence"])
                save_knowledge_graph(st.session_state.kg)
                st.success("Added to Knowledge Graph!")

    # =======================================================
    # DYNAMIC PATHWAY GRAPH
    # =======================================================
    st.subheader("🧬 Dynamic Pathway Graph")

    try:
        pathway_path = draw_pathway_graph_html(
            build_dynamic_pathway_graph(parsed.get("hypotheses", []), docs)
        )
        st.components.v1.html(open(pathway_path).read(), height=420)
    except Exception as e:
        st.error(f"Pathway graph error: {e}")

    # =======================================================
    # KNOWLEDGE GRAPH VISUALIZATION
    # =======================================================
    st.subheader("🧠 Knowledge Graph")

    if len(st.session_state.kg["nodes"]) > 0:
        kg_path = draw_knowledge_graph_html(st.session_state.kg)
        st.components.v1.html(open(kg_path).read(), height=420)
    else:
        st.info("No Knowledge Graph entries yet. Add hypotheses using the ➕ button!")

    st.success("✅ Analysis completed. Knowledge Graph & Pathway Graph updated.")

