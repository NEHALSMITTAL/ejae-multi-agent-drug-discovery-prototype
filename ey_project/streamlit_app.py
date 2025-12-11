import streamlit as st
import json
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any

# backend imports
from backend.agents import (
    literature_agent,
    evidence_scorer,
    experiment_recommender
)
from backend.active_learning import save_feedback, load_feedback
from backend.chem_utils import compute_similarity

# Visualization
from pyvis.network import Network
import networkx as nx

# PDF export
from fpdf import FPDF
import datetime

# ============================================================
# 1. Correct ROOT Directory (VERY IMPORTANT)
# ============================================================
ROOT = Path(__file__).resolve().parent         # ey_project/
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

KG_FILE = DATA_DIR / "knowledge_graph.json"
FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"

# ============================================================
# 2. Load & Save Knowledge Graph
# ============================================================
def load_knowledge_graph() -> Dict[str, Any]:
    if KG_FILE.exists():
        try:
            return json.loads(KG_FILE.read_text(encoding="utf-8"))
        except:
            return {"nodes": [], "edges": []}
    return {"nodes": [], "edges": []}

def save_knowledge_graph(kg: Dict[str, Any]):
    KG_FILE.write_text(json.dumps(kg, indent=2), encoding="utf-8")

def kg_add_hypothesis(kg, hypothesis_text, evidence_list):
    hid = f"HYP_{len(kg['nodes'])+1}"

    kg["nodes"].append({
        "id": hid, "label": hypothesis_text, "type": "hypothesis"
    })

    # Add evidence as nodes + edges
    for ev in evidence_list:
        doc_id = ev.get("doc_id", "doc_unknown")

        # create doc node if not exists
        if not any(n for n in kg["nodes"] if n["id"] == doc_id):
            kg["nodes"].append({
                "id": doc_id,
                "label": doc_id,
                "type": "document"
            })

        kg["edges"].append({
            "source": doc_id,
            "target": hid,
            "label": "supports"
        })

    save_knowledge_graph(kg)

# ============================================================
# 3. Draw Knowledge Graph
# ============================================================
def draw_kg_pyvis(kg: Dict[str, Any], height="420px"):
    G = nx.DiGraph()

    for n in kg.get("nodes", []):
        G.add_node(n["id"],
            label=n.get("label", n["id"]),
            title=n.get("label",""),
        )
    for e in kg.get("edges", []):
        G.add_edge(e["source"], e["target"], label=e.get("label",""))

    net = Network(height=height, width="100%", directed=True)
    net.from_nx(G)

    # Node colors
    for nd in net.nodes:
        if nd["id"].startswith("HYP_"):
            nd["color"] = "#6c5ce7"
            nd["shape"] = "box"
        elif nd["id"].startswith("doc"):
            nd["color"] = "#00b894"
            nd["shape"] = "ellipse"

    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, "kg_graph.html")
    net.save_graph(path)
    return path

# ============================================================
# 4. PDF Generation
# ============================================================
def generate_protocol_pdf(hypothesis, evidence_snips, exp_text):
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = DATA_DIR / f"protocol_{timestamp}.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Experiment Protocol", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.multi_cell(0, 8, f"Hypothesis:\n{hypothesis}")

    pdf.set_font("Arial", "B", 12)
    pdf.ln(3)
    pdf.cell(0, 8, "Evidence:", ln=True)

    pdf.set_font("Arial", "", 11)
    for s in evidence_snips:
        pdf.multi_cell(0, 7, f"- {s}")

    pdf.set_font("Arial", "B", 12)
    pdf.ln(3)
    pdf.cell(0, 8, "Recommended Experiment:", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, exp_text)

    pdf.output(str(filename))
    return filename

# ============================================================
# 5. Simple Pathway Graph
# ============================================================
def draw_pathway_graph():
    G = nx.DiGraph()
    edges = [("IL6","JAK"),("JAK","STAT3"),("EGFR","VEGF"),("TNFα","STAT3")]

    for s,t in edges:
        G.add_edge(s,t)

    net = Network(height="420px", width="100%", directed=True)
    net.from_nx(G)

    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, "pathway_graph.html")
    net.save_graph(path)
    return path

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Ejae Drug Discovery Prototype", layout="wide")

st.title("🧬 Ejae Multi-Agent Drug Discovery Prototype")

kg = load_knowledge_graph()

query = st.text_input("Enter disease, pathway, or molecule:")
smiles_input = st.text_input("SMILES (optional):")
run = st.button("Run analysis")

# ============================================================
# RUN PIPELINE
# ============================================================
if run:

    parsed, docs = literature_agent(query)

    st.subheader("📄 Retrieved Documents")
    for d in docs:
        with st.expander(f"{d['id']} — {d['meta']['title']}"):
            st.write(d["text"])

    # ----------------- Similarity -----------------
    if smiles_input.strip():
        st.subheader("🧪 Molecular Similarity (RDKit)")
        refs = {
            "Gefitinib": "COCCOc1ccc2nc(N3CCC(N)CC3)nc(N)c2c1",
            "Erlotinib": "CN(C)CCOc1c(OC)nc(Nc2cccc(Cl)c2)n1"
        }
        for name, smi in refs.items():
            sim = compute_similarity(smiles_input, smi)
            if sim:
                st.write(f"{name}: {sim:.3f}")

    # ----------------- Hypotheses -----------------
    if "hypotheses" in parsed:
        st.subheader("🧠 Hypotheses")

        for i, h in enumerate(parsed["hypotheses"]):

            st.markdown(f"### Hypothesis {i+1}")
            st.write(h["text"])

            evidence_snips = []
            for e in h["evidence"]:
                snippet = e["snippet"][:300]
                st.write(f"- {snippet}")
                evidence_snips.append(snippet)

            score = evidence_scorer(h["text"], evidence_snips)
            st.info(f"Evidence Score: {score}")

            exp_text = experiment_recommender(h["text"], evidence_snips)
            st.success(exp_text)

            # Feedback
            col1, col2 = st.columns(2)
            if col1.button(f"Accept {i+1}"):
                save_feedback(h["text"], True)
                st.success("Saved ✔")
            if col2.button(f"Reject {i+1}"):
                save_feedback(h["text"], False)
                st.error("Saved ✘")

            # Add to KG
            if st.checkbox(f"Add to Knowledge Graph {i+1}"):
                kg_add_hypothesis(kg, h["text"], h["evidence"])
                st.success("Added to Knowledge Graph!")

            # PDF
            if st.button(f"Export PDF {i+1}"):
                path = generate_protocol_pdf(h["text"], evidence_snips, exp_text)
                with open(path, "rb") as f:
                    st.download_button("Download PDF", data=f, file_name=path.name)

    # Pathway graph
    st.subheader("🧬 Pathway Graph")
    pg = draw_pathway_graph()
    st.components.v1.html(open(pg).read(), height=420)

    # Knowledge graph
    st.subheader("🧠 Knowledge Graph")
    if kg["nodes"]:
        kg_path = draw_kg_pyvis(kg)
        st.components.v1.html(open(kg_path).read(), height=420)
    else:
        st.info("No KG entries yet.")

    save_knowledge_graph(kg)
