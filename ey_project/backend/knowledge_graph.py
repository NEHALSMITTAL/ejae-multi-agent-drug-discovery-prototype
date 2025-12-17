import json
from pathlib import Path
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import re

# ============================================================
# ROOT & DATA PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]      # ey_project/
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

KG_JSON = DATA_DIR / "knowledge_graph.json"


# ============================================================
# LOAD KNOWLEDGE GRAPH
# ============================================================
def load_knowledge_graph():
    """
    Loads stored knowledge graph JSON into memory.
    Returns: {"nodes": [...], "edges": [...]}
    """
    if KG_JSON.exists():
        try:
            return json.loads(KG_JSON.read_text())
        except Exception:
            return {"nodes": [], "edges": []}
    return {"nodes": [], "edges": []}


# ============================================================
# SAVE KNOWLEDGE GRAPH
# ============================================================
def save_knowledge_graph(kg):
    KG_JSON.write_text(json.dumps(kg, indent=2))


# ============================================================
# ADD HYPOTHESIS + EVIDENCE TO KG
# ============================================================
def add_hypothesis_to_kg(kg, hypothesis_text, evidence_list):
    """
    Adds hypothesis node + evidence doc nodes + edges.
    """

    # Create unique ID
    hid = f"HYP_{len(kg['nodes']) + 1}"

    kg["nodes"].append({
        "id": hid,
        "label": hypothesis_text,
        "type": "hypothesis"
    })

    # Add evidence nodes
    for ev in evidence_list:
        doc_id = ev.get("doc_id") or ev.get("source") or f"doc_{len(kg['nodes'])}"

        # Create document node only once
        if not any(n["id"] == doc_id for n in kg["nodes"]):
            kg["nodes"].append({
                "id": doc_id,
                "label": doc_id,
                "type": "document"
            })

        # Edge: evidence → hypothesis
        kg["edges"].append({
            "source": doc_id,
            "target": hid,
            "label": "supports"
        })

    return kg


# ============================================================
# RENDER KNOWLEDGE GRAPH AS HTML (PyVis)
# ============================================================
def draw_knowledge_graph_html(kg):
    """
    Creates an interactive PyVis HTML for display in Streamlit.
    """

    G = nx.DiGraph()

    # Add nodes
    for n in kg.get("nodes", []):
        G.add_node(
            n["id"],
            label=n.get("label", n["id"]),
            title=n.get("label", "")
        )

    # Add edges
    for e in kg.get("edges", []):
        G.add_edge(e["source"], e["target"], label=e.get("label", ""))

    # PyVis rendering
    net = Network(height="420px", width="100%", directed=True, bgcolor="#FFFFFF")
    net.barnes_hut()
    net.from_nx(G)

    # Style
    for node in net.nodes:
        if node["id"].startswith("HYP_"):
            node["color"] = "#6c5ce7"
            node["shape"] = "box"
            node["borderWidth"] = 2
        else:
            node["color"] = "#00b894"
            node["shape"] = "ellipse"

    # Save temporary HTML
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, "knowledge_graph.html")
    net.save_graph(path)
    return path


# ============================================================
# ENTITY EXTRACTION FOR PATHWAY GRAPH
# ============================================================
KNOWN_GENES = {
    "IL6", "JAK", "JAK1", "JAK2", "STAT3", "STAT1", "STAT5",
    "EGFR", "VEGF", "TNFA", "TNF", "PI3K", "AKT", "MTOR"
}

GENE_REGEX = r"\b[A-Z0-9\-]{3,8}\b"


def extract_entities(text: str):
    """
    Extracts gene-like tokens using regex + whitelist.
    """
    if not text:
        return set()

    found = set(re.findall(GENE_REGEX, text))
    return {g for g in found if g.upper() in KNOWN_GENES}


# ============================================================
# BUILD DYNAMIC PATHWAY GRAPH (CO-OCCURRENCE MODEL)
# ============================================================
def build_dynamic_pathway_graph(hypotheses, docs):
    """
    Builds a rough gene–gene pathway graph from hypotheses & docs.
    """

    G = nx.DiGraph()
    entities = []

    # Extract from hypotheses
    for h in hypotheses:
        entities += list(extract_entities(h.get("text", "")))

        for ev in h.get("evidence", []):
            entities += list(extract_entities(ev.get("snippet", "")))

    # Extract from docs
    for d in docs:
        entities += list(extract_entities(d.get("text", "")))

    uniq = list(set(entities))

    # Build simple pairwise edges
    for i in range(len(uniq) - 1):
        G.add_edge(uniq[i], uniq[i + 1])

    # Fallback graph if empty
    if len(G.nodes()) == 0:
        fallback = [("IL6", "JAK"), ("JAK", "STAT3"), ("EGFR", "VEGF")]
        for s, t in fallback:
            G.add_edge(s, t)

    return G


# ============================================================
# DRAW PATHWAY GRAPH AS HTML
# ============================================================
def draw_pathway_graph_html(G):
    """
    Renders dynamic pathway graph to HTML.
    """

    net = Network(height="420px", width="100%", directed=True, bgcolor="#FFFFFF")
    net.barnes_hut()
    net.from_nx(G)

    # Style
    for node in net.nodes:
        node["shape"] = "dot"
        node["color"] = "#0984e3"
        node["size"] = 18
        node["borderWidth"] = 2

    for edge in net.edges:
        edge["color"] = "#636e72"
        edge["width"] = 2
        edge["arrows"] = "to"

    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, "dynamic_pathway.html")
    net.save_graph(path)
    return path
