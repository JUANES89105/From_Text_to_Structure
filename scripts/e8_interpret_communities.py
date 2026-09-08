from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e8_topic_modeling"

MAIN_COMMUNITIES = [0, 1, 2, 4]
TOP_N_CONCEPTS = 15
TOP_N_TOPICS = 5
TOP_N_BRIDGES = 20

print("=" * 78)
print("E8 - Interpretación estructural CRS vs LDA (K=50)")
print("=" * 78)

# ============================================================
# 1. Cargar resultados ya calculados
# ============================================================

print("\n[1/6] Cargando resultados...", flush=True)

docs = pd.read_csv(OUT / "crs_document_assignments.csv")
communities = pd.read_csv(OUT / "crs_keyword_communities.csv")
edges = pd.read_csv(OUT / "crs_backbone_edges.csv")
topics = pd.read_csv(OUT / "lda_topics_all_k.csv")

print(f"Documentos: {len(docs):,}")
print(f"Conceptos backbone: {len(communities):,}")
print(f"Aristas backbone: {len(edges):,}")
print(f"Filas de tópicos: {len(topics):,}")

print("\nColumnas communities:", communities.columns.tolist())
print("Columnas edges:", edges.columns.tolist())
print("Columnas topics:", topics.columns.tolist())

# ============================================================
# 2. Detectar columnas
# ============================================================

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"No se encontró ninguna de {candidates}. "
        f"Columnas disponibles: {df.columns.tolist()}"
    )

keyword_col = find_col(
    communities,
    ["keyword", "node", "concept", "term"]
)

community_col = find_col(
    communities,
    ["community", "louvain_community", "cluster"]
)

source_col = find_col(
    edges,
    ["source", "u", "node1", "keyword1"]
)

target_col = find_col(
    edges,
    ["target", "v", "node2", "keyword2"]
)

weight_col = find_col(
    edges,
    ["weight", "support", "edge_weight"]
)

# ============================================================
# 3. Métricas de conceptos dentro del backbone
# ============================================================

print("\n[2/6] Calculando centralidad descriptiva del backbone...",
      flush=True)

degree = Counter()
weighted_degree = Counter()

for _, row in edges.iterrows():
    u = row[source_col]
    v = row[target_col]
    w = float(row[weight_col])

    degree[u] += 1
    degree[v] += 1

    weighted_degree[u] += w
    weighted_degree[v] += w

communities["degree_backbone"] = (
    communities[keyword_col].map(degree).fillna(0).astype(int)
)

communities["weighted_degree_backbone"] = (
    communities[keyword_col].map(weighted_degree).fillna(0.0)
)

communities = communities.sort_values(
    [community_col, "weighted_degree_backbone", "degree_backbone"],
    ascending=[True, False, False]
)

communities.to_csv(
    OUT / "crs_community_concept_metrics.csv",
    index=False
)

# ============================================================
# 4. Descripción de cada comunidad
# ============================================================

print("\n[3/6] Resumiendo comunidades principales...", flush=True)

community_summary_rows = []
community_concept_rows = []

assigned_docs = docs[
    docs["crs_status"] == "assigned"
].copy()

assigned_docs["crs_community"] = (
    assigned_docs["crs_community"].astype(int)
)

assigned_total = len(assigned_docs)

for community_id in sorted(
    communities[community_col].unique()
):

    subset_nodes = communities[
        communities[community_col] == community_id
    ].copy()

    subset_docs = assigned_docs[
        assigned_docs["crs_community"] == community_id
    ].copy()

    top_nodes = subset_nodes.head(TOP_N_CONCEPTS)

    top_concepts = "; ".join(
        top_nodes[keyword_col].astype(str).tolist()
    )

    for rank, (_, row) in enumerate(
        top_nodes.iterrows(),
        start=1
    ):
        community_concept_rows.append({
            "community": int(community_id),
            "rank": rank,
            "keyword": row[keyword_col],
            "degree_backbone": row["degree_backbone"],
            "weighted_degree_backbone":
                row["weighted_degree_backbone"]
        })

    topic_counts = (
        subset_docs["lda_topic_k50"]
        .value_counts()
        .head(TOP_N_TOPICS)
    )

    top_topics_string = "; ".join(
        [
            f"T{int(topic)}: {int(count)} "
            f"({count / len(subset_docs):.2%})"
            for topic, count in topic_counts.items()
        ]
    ) if len(subset_docs) else ""

    community_summary_rows.append({
        "community": int(community_id),
        "n_backbone_concepts": len(subset_nodes),
        "n_assigned_documents": len(subset_docs),
        "proportion_of_assigned_documents":
            len(subset_docs) / assigned_total,
        "top_backbone_concepts": top_concepts,
        "top_lda50_topics": top_topics_string
    })

community_summary = pd.DataFrame(community_summary_rows)

community_concepts = pd.DataFrame(community_concept_rows)

community_summary.to_csv(
    OUT / "crs_lda50_community_summary.csv",
    index=False
)

community_concepts.to_csv(
    OUT / "crs_top_concepts_by_community.csv",
    index=False
)

# ============================================================
# 5. Distribución completa CRS x LDA50
# ============================================================

print("\n[4/6] Construyendo tabla CRS x LDA50...", flush=True)

cross = pd.crosstab(
    assigned_docs["crs_community"],
    assigned_docs["lda_topic_k50"]
)

cross.to_csv(
    OUT / "crs_lda50_crosstab_counts.csv"
)

cross_prop = cross.div(
    cross.sum(axis=1),
    axis=0
)

cross_prop.to_csv(
    OUT / "crs_lda50_crosstab_row_proportions.csv"
)

# Formato largo más fácil para paper/análisis
long_rows = []

for community_id in cross.index:
    total = cross.loc[community_id].sum()

    for topic_id in cross.columns:
        count = int(cross.loc[community_id, topic_id])

        if count > 0:
            long_rows.append({
                "crs_community": int(community_id),
                "lda_topic_k50": int(topic_id),
                "n_documents": count,
                "proportion_within_crs_community":
                    count / total
            })

pd.DataFrame(long_rows).to_csv(
    OUT / "crs_lda50_overlap_long.csv",
    index=False
)

# ============================================================
# 6. Aristas intercomunidad / conceptos puente
# ============================================================

print("\n[5/6] Identificando conexiones entre comunidades...",
      flush=True)

keyword_to_comm = dict(
    zip(
        communities[keyword_col],
        communities[community_col]
    )
)

bridge_rows = []

for _, row in edges.iterrows():
    u = row[source_col]
    v = row[target_col]

    cu = keyword_to_comm.get(u)
    cv = keyword_to_comm.get(v)

    if cu is None or cv is None:
        continue

    if cu != cv:
        bridge_rows.append({
            "source": u,
            "source_community": int(cu),
            "target": v,
            "target_community": int(cv),
            "weight": float(row[weight_col])
        })

bridges = pd.DataFrame(bridge_rows)

if len(bridges):
    bridges = bridges.sort_values(
        "weight",
        ascending=False
    )

bridges.to_csv(
    OUT / "crs_intercommunity_edges.csv",
    index=False
)

# Resumen por par de comunidades
if len(bridges):
    pair_summary = (
        bridges
        .assign(
            community_a=lambda x:
                x[
                    ["source_community", "target_community"]
                ].min(axis=1),
            community_b=lambda x:
                x[
                    ["source_community", "target_community"]
                ].max(axis=1)
        )
        .groupby(
            ["community_a", "community_b"],
            as_index=False
        )
        .agg(
            n_intercommunity_edges=("weight", "size"),
            total_intercommunity_weight=("weight", "sum"),
            mean_intercommunity_weight=("weight", "mean"),
            max_intercommunity_weight=("weight", "max")
        )
        .sort_values(
            [
                "total_intercommunity_weight",
                "n_intercommunity_edges"
            ],
            ascending=False
        )
    )
else:
    pair_summary = pd.DataFrame()

pair_summary.to_csv(
    OUT / "crs_intercommunity_pair_summary.csv",
    index=False
)

# Conceptos involucrados en aristas intercomunidad
bridge_concept_stats = Counter()
bridge_concept_weight = Counter()

for _, row in bridges.iterrows():
    for node in [row["source"], row["target"]]:
        bridge_concept_stats[node] += 1
        bridge_concept_weight[node] += row["weight"]

bridge_concepts = []

for node, n_edges in bridge_concept_stats.items():
    bridge_concepts.append({
        "keyword": node,
        "community": int(keyword_to_comm[node]),
        "n_intercommunity_edges": n_edges,
        "intercommunity_weight":
            bridge_concept_weight[node],
        "degree_backbone":
            degree[node],
        "weighted_degree_backbone":
            weighted_degree[node]
    })

bridge_concepts = pd.DataFrame(bridge_concepts)

if len(bridge_concepts):
    bridge_concepts = bridge_concepts.sort_values(
        [
            "intercommunity_weight",
            "n_intercommunity_edges"
        ],
        ascending=False
    )

bridge_concepts.to_csv(
    OUT / "crs_bridge_concepts.csv",
    index=False
)

# ============================================================
# 7. Mostrar resumen interpretable
# ============================================================

print("\n[6/6] RESULTADOS")
print("=" * 78)

print("\nCOMUNIDADES PRINCIPALES")
print("-" * 78)

for c in MAIN_COMMUNITIES:

    row = community_summary[
        community_summary["community"] == c
    ]

    if row.empty:
        continue

    row = row.iloc[0]

    print(f"\nCRS COMMUNITY {c}")
    print(
        f"Conceptos backbone: "
        f"{row['n_backbone_concepts']}"
    )
    print(
        f"Documentos asignados: "
        f"{row['n_assigned_documents']:,} "
        f"({row['proportion_of_assigned_documents']:.2%})"
    )
    print("Top conceptos:")
    print(row["top_backbone_concepts"])
    print("Top LDA-50:")
    print(row["top_lda50_topics"])

print("\n" + "=" * 78)
print("CONEXIONES ENTRE COMUNIDADES")
print("=" * 78)

print(
    f"\nNúmero de aristas intercomunidad: "
    f"{len(bridges):,}"
)

if len(pair_summary):
    print("\nPares de comunidades:")
    print(pair_summary.to_string(index=False))

if len(bridges):
    print(f"\nTop {TOP_N_BRIDGES} aristas intercomunidad:")
    print(
        bridges.head(TOP_N_BRIDGES).to_string(
            index=False
        )
    )

if len(bridge_concepts):
    print(f"\nTop {TOP_N_BRIDGES} conceptos puente:")
    print(
        bridge_concepts.head(TOP_N_BRIDGES).to_string(
            index=False
        )
    )

print("\nArchivos generados:")
for name in [
    "crs_community_concept_metrics.csv",
    "crs_lda50_community_summary.csv",
    "crs_top_concepts_by_community.csv",
    "crs_lda50_crosstab_counts.csv",
    "crs_lda50_crosstab_row_proportions.csv",
    "crs_lda50_overlap_long.csv",
    "crs_intercommunity_edges.csv",
    "crs_intercommunity_pair_summary.csv",
    "crs_bridge_concepts.csv"
]:
    print(" ", OUT / name)

print("\nE8 qualitative/structural comparison complete.")
