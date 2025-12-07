from pyvis.network import Network
import networkx as nx
import tempfile, os

def make_pyvis_graph(nodes, edges, output_html_path=None, height="450px", width="100%"):
    """
    nodes: list of dicts {'id':'GeneA', 'label':'GeneA', 'title':'info'}
    edges: list of tuples (source, target)
    returns path to HTML
    """
    G = nx.Graph()
    for n in nodes:
        G.add_node(n['id'], label=n.get('label', n['id']), title=n.get('title',''))
    for s,t in edges:
        G.add_edge(s,t)

    net = Network(height=height, width=width, notebook=False)
    net.from_nx(G)
    net.toggle_physics(True)
    if output_html_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        output_html_path = tmp.name
    net.show(output_html_path)
    return output_html_path
