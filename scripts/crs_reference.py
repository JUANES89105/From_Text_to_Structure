"""CRS reference functions copied unchanged from notebooks 2, 5, and 3.

The experiment validates their syntax trees before every invocation.
"""
import ast
import itertools
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def parse_keywords(x):
    if pd.isna(x) or x == "" or x == "[]":
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            return val if isinstance(val, list) else []
        except Exception:
            return []
    return []

def build_crs_for_tau(docs_keywords, embeddings_dict, tau):
    G = nx.Graph()

    for idx, kws in enumerate(docs_keywords, start=1):

        # Add nodes and document frequency
        for k in kws:
            if G.has_node(k):
                G.nodes[k]["doc_freq"] += 1
            else:
                G.add_node(k, doc_freq=1)

        if len(kws) < 2:
            continue

        # Local embedding matrix
        local_emb = np.array([embeddings_dict[k] for k in kws])
        S = cosine_similarity(local_emb)

        # Local edges filtered by tau
        for i, j in itertools.combinations(range(len(kws)), 2):
            sim = float(S[i, j])

            if sim >= tau:
                a, b = kws[i], kws[j]

                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                    G[a][b]["sim_sum"] += sim
                    G[a][b]["sim_count"] += 1
                else:
                    G.add_edge(
                        a,
                        b,
                        weight=1,
                        sim_sum=sim,
                        sim_count=1
                    )

    # Mean similarity
    for u, v, d in G.edges(data=True):
        d["sim_mean"] = d["sim_sum"] / d["sim_count"]

    return G

def build_backbone(G, wmin):
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from([
        (u, v, d)
        for u, v, d in G.edges(data=True)
        if float(d.get("weight", 1)) >= wmin
    ])
    H.remove_nodes_from([n for n in list(H.nodes()) if H.degree(n) == 0])
    return H
