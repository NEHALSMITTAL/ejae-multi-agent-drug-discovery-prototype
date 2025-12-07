import streamlit as st
import json
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any

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

# -----------------------
# Paths & Data dir setup
# -----------------------
ROOT = Path(".")
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

KG_FILE = DATA_DIR / "knowledge_graph.json"      # persisted knowledge graph
FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"      # created by save_feedback

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(
    page_title="🧬 Ejae Drug Discovery Prototype",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧬 Ejae Multi-Agent Drug Discovery Prototype")
st.write("Enter a disease name, biological process, or molecule description to generate scientific insights.")

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Local Model (Ollama)",
        ["llama3", "llama3:8b", "phi3", "mistral", "qwen"],
        index=0
    )
    k_docs = st.slider("Number of documents to retrieve", 1, 10, 5)
    st.markdown("**Session tools**")
    if st.button("Show saved feedback"):
        entries = load_feedback()
        st.write(f"Total feedback entries: {len(entries)}")
        for e in entries[-20:]:
            st.write(e)
    st.info("Make sure the model is pulled in Ollama (if using).")

# -------------------------
# Inputs
# -------------------------
query = st.text_input("🔍 Enter Disease, Pathway, or SMILES Molecule:")
smiles_input = st.text_input("🧪 Enter SMILES for Molecule Similarity (optional):")
run = st.button("🚀 Run Analysis")

# -------------------------
# Knowledge Graph helpers
# -------------------------
def load_knowledge_graph() -> Dict[str, Any]:
    if KG_FILE.exists():
        try:
            return json.loads(KG_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"nodes": [], "edges": []}
    return {"nodes": [], "edges": []}

def save_knowledge_graph(kg: Dict[str, Any]):
    KG_FILE.write_text(json.dumps(kg, indent=2), encoding="utf-8")

def kg_add_hypothesis(kg: Dict[str, Any], hypothesis_text: str, evidence_list: List[Dict[str,str]]):
    # create a node for hypothesis
    hid = f"HYP_{len(kg['nodes'])+1}"
    kg['nodes'].append({"id": hid, "label": hypothesis_text, "type": "hypothesis"})
    # add evidence nodes and edges
    for ev in evidence_list:
        doc_id = ev.get("doc_id", "doc_unknown")
        # ensure doc node exists
        if not any(n for n in kg['nodes'] if n.get("id") == doc_id):
            kg['nodes'].append({"id": doc_id, "label": doc_id, "type": "document"})
        # add edge doc -> hypothesis
        kg['edges'].append({"source": doc_id, "target": hid, "label": "supports"})
    save_knowledge_graph(kg)

def draw_kg_pyvis(kg: Dict[str, Any], height="420px"):
    G = nx.DiGraph()
    for n in kg.get("nodes", []):
        G.add_node(n["id"], label=n.get("label", n["id"]), title=n.get("label",""), type=n.get("type",""))
    for e in kg.get("edges", []):
        G.add_edge(e["source"], e["target"], label=e.get("label",""))
    net = Network(height=height, width="100%", directed=True)
    net.from_nx(G)
    # style nodes a bit
    for node in net.nodes:
        if node.get("id", "").startswith("HYP_"):
            node["color"] = "#6c5ce7"
            node["shape"] = "box"
        elif node.get("id","").startswith("doc"):
            node["color"] = "#00b894"
            node["shape"] = "ellipse"
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, "kg_graph.html")
    net.save_graph(path)
    return path

# -------------------------
# PDF Generation helper
# -------------------------
def generate_protocol_pdf(hypothesis: str, evidence_snippets: List[str], experiment_text: str, filename: str = None):
    """
    Create a simple PDF that contains the hypothesis, evidence, and experiment steps.
    Returns the path to the generated PDF.
    """
    if filename is None:
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"experiment_protocol_{timestamp}.pdf"
    out_path = DATA_DIR / filename

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Experiment Protocol (Ejae Prototype)", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"Hypothesis: {hypothesis}")
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Evidence:", ln=True)
    pdf.set_font("Arial", "", 11)
    for s in evidence_snippets:
        pdf.multi_cell(0, 7, f"- {s}")
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Recommended Experiment:", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7, experiment_text)
    pdf.ln(6)
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(0, 6, "Disclaimer: This protocol is generated by a prototype LLM-powered system for demonstration only. Do not perform experiments without institutional approvals and proper biosafety measures.")
    pdf.output(str(out_path))
    return out_path

# -------------------------
# Pathway graph helper (simple demo)
# -------------------------
def draw_pathway_graph():
    G = nx.DiGraph()
    nodes = ["IL6", "JAK", "STAT3", "EGFR", "VEGF", "TNFα"]
    edges = [("IL6", "JAK"), ("JAK", "STAT3"), ("EGFR", "VEGF"), ("TNFα", "STAT3")]
    for n in nodes:
        G.add_node(n)
    for e in edges:
        G.add_edge(e[0], e[1])
    net = Network(height="420px", width="100%", directed=True)
    net.from_nx(G)
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, "pathway_graph.html")
    net.save_graph(path)
    return path

# -------------------------
# Run main pipeline
# -------------------------
kg = load_knowledge_graph()  # load on start (session memory)

if run:
    if not query.strip():
        st.error("⚠️ Please enter a valid query.")
        st.stop()

    with st.spinner("🔬 Running multi-agent biomedical reasoning pipeline…"):
        parsed, docs = literature_agent(query)

    # -------------------------
    # Display retrieved documents
    # -------------------------
    st.subheader("📄 Retrieved Scientific Literature")
    for d in docs:
        with st.expander(f"📘 Document {d['id']} — {d['meta'].get('title','')}"):
            st.write(d["text"])

    # -------------------------
    # Molecule similarity
    # -------------------------
    if smiles_input.strip():
        st.subheader("🧪 Molecule Similarity (RDKit)")
        reference_molecules = {
            "Gefitinib": "COCCOc1ccc2nc(N3CCC(N)CC3)nc(N)c2c1",
            "Erlotinib": "CN(C)CCOc1c(OC)nc(Nc2cccc(Cl)c2)n1",
            "Imatinib": "CCOc1ccc(cc1)C(=O)Nc2ccc(cc2)N3CCN(CC3)C"
        }
        for name, ref_smiles in reference_molecules.items():
            sim = compute_similarity(smiles_input, ref_smiles)
            if sim is not None:
                st.write(f"🔹 Similarity with **{name}**: `{sim:.3f}`")
            else:
                st.warning(f"⚠️ Could not compute similarity with {name} (invalid SMILES).")

    # -------------------------
    # Hypotheses & experiments
    # -------------------------
    if "hypotheses" in parsed:
        st.subheader("🧠 Generated Hypotheses")
        for i, h in enumerate(parsed["hypotheses"]):
            st.markdown(f"### Hypothesis {i+1}")
            st.write(h.get("text", ""))

            # evidence list and simple display
            evidence_snips = []
            st.markdown("**Evidence:**")
            for e in h.get("evidence", []):
                docid = e.get("doc_id", "doc?")
                snippet = e.get("snippet", "")[:400]
                st.write(f"- ({docid}) {snippet}...")
                evidence_snips.append(snippet)

            # score
            score = evidence_scorer(h.get("text", ""), evidence_snips)
            st.markdown("**Evidence score**")
            st.info(score)

            # experiment recommendation
            exp_text = experiment_recommender(h.get("text", ""), evidence_snips)
            st.markdown("**Recommended experiment**")
            st.success(exp_text)

            # Save to knowledge graph (session memory)
            kg_add = st.checkbox(f"Add Hypothesis {i+1} to Knowledge Graph", key=f"kgadd_{i}")
            if kg_add:
                kg_add_hypothesis(kg, h.get("text",""), h.get("evidence", []))
                st.success("Added to knowledge graph (saved).")

            # PDF export button
            if st.button(f"Export Protocol PDF (Hypothesis {i+1})", key=f"pdf_{i}"):
                pdf_path = generate_protocol_pdf(h.get("text",""), evidence_snips, exp_text)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download Protocol PDF",
                        data=f,
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf"
                    )

            # Feedback buttons
            col1, col2 = st.columns(2)
            if col1.button(f"👍 Accept Hypothesis {i+1}", key=f"accept_{i}"):
                save_feedback(h.get("text",""), True)
                st.success(f"Hypothesis {i+1} accepted ✔")
            if col2.button(f"👎 Reject Hypothesis {i+1}", key=f"reject_{i}"):
                save_feedback(h.get("text",""), False)
                st.error(f"Hypothesis {i+1} rejected ✘")

            st.markdown("---")
    else:
        st.error("❌ Model did not return valid JSON or no hypotheses generated.")
        st.code(parsed, language="json")

    # -------------------------
    # Pathway graph display
    # -------------------------
    st.subheader("🧬 Biological Pathway Graph")
    pg = draw_pathway_graph()
    st.components.v1.html(open(pg, "r", encoding="utf-8").read(), height=420)

    # -------------------------
    # Knowledge graph display (session memory)
    # -------------------------
    st.subheader("🧠 Session Knowledge Graph")

    if kg.get("nodes"):
        kg_path = draw_kg_pyvis(kg)
        st.components.v1.html(open(kg_path, "r", encoding="utf-8").read(), height=420)
    else:
        st.info("No knowledge graph data yet. Select 'Add Hypothesis to Knowledge Graph' above to populate this graph.")

    # persist KG after run (if updated)
    save_knowledge_graph(kg)
