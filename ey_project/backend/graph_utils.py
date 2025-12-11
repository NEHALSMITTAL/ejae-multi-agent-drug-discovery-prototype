# ey_project/backend/graph_utils.py

from pyvis.network import Network
import networkx as nx
from pathlib import Path
import tempfile, os
import json

# -----------------------
# Correct Data directory
# -----------------------
ROOT = Path(__file__).resolve().parents[1]     # ey_project/
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

KG_HTML = DATA_DIR / "knowledge_graph.html"    # Force save inside ey_project/data/


# -----------------------
# Build PyVis Graph
# -----------------------
def make_pyvis_graph(nodes, edges, output_html_path=None,
                     height="450px", width="100%"):

    """
    nodes: list of dicts → {id, label, title(optional)}
    edges: list of dicts → {source, target, label(optional)}
    """

    G = nx.DiGraph()

    # Add nodes
    for n in nodes:
        G.add_node(
            n["id"],
            label=n.get("label", n["id"]),
            title=n.get("title", n["label"]),
            color=("#6c5ce7" if n.get("type") == "hypothesis"
                   else "#00b894" if n.get("type") == "document"
                   else "#0984e3")
        )

    # Add edges
    for e in edges:
        G.add_edge(e["source"], e["target"], label=e.get("label", ""))

    # PyVis network
    net = Network(height=height, width=width, directed=True)
    net.from_nx(G)
    net.toggle_physics(True)

    # Final output path
    if output_html_path is None:
        output_html_path = KG_HTML

    net.save_graph(str(output_html_path))
    return str(output_html_path)


# -----------------------
# Reset / Delete KG file
# -----------------------
def reset_knowledge_graph():
    if KG_HTML.exists():
        KG_HTML.unlink()
