from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import networkx as nx
import community as community_louvain

# Permitir importar los scripts locales
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import e3_scalability as e3
import crs_reference as ref

OUT = ROOT / "results" / "e8_topic_modeling"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("E8 - Reconstrucción CRS y exportación de comunidades Louvain")
print("=" * 70)

# ------------------------------------------------------------
# 1. Cargar exactamente el corpus analizable de E3
# ------------------------------------------------------------
print("\n[1/7] Cargando corpus...", flush=True)

corpus, corpus_audit = e3.load_corpus()

print("Documentos analizables:", len(corpus))

assert len(corpus) == 52946, (
    f"Se esperaban 52946 documentos y se obtuvieron {len(corpus)}"
)

docs = corpus["keywords"].tolist()

# ------------------------------------------------------------
# 2. Configuración determinista idéntica a E3
# ------------------------------------------------------------
print("\n[2/7] Configurando ejecución determinista...", flush=True)

np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

print("Modelo:", e3.MODEL)
print("Revisión:", e3.REVISION)
print("tau:", e3.TAU)
print("backbone:", e3.BACKBONE)
print("Louvain seed:", e3.LOUVAIN_SEED)

# ------------------------------------------------------------
# 3. Cargar modelo local
# ------------------------------------------------------------
print("\n[3/7] Cargando SentenceTransformer...", flush=True)

t0 = time.perf_counter()

model = e3.SentenceTransformer(
    e3.MODEL,
    revision=e3.REVISION,
    device="cpu",
    local_files_only=True
)

print(
    f"Modelo cargado en {time.perf_counter() - t0:.2f} s",
    flush=True
)

# ------------------------------------------------------------
# 4. Vocabulario y embeddings
# ------------------------------------------------------------
print("\n[4/7] Construyendo vocabulario...", flush=True)

vocab = sorted(set(k for kws in docs for k in kws))

print("Vocabulario único:", len(vocab))

assert len(vocab) == 56635, (
    f"Se esperaban 56635 keywords y se obtuvieron {len(vocab)}"
)

print("\nCalculando embeddings...", flush=True)

t0 = time.perf_counter()

embedding_matrix = model.encode(
    vocab,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(
    f"Embeddings terminados en {time.perf_counter() - t0:.2f} s",
    flush=True
)

embeddings = {
    kw: embedding_matrix[i]
    for i, kw in enumerate(vocab)
}

# ------------------------------------------------------------
# 5. Reconstruir CRS
# ------------------------------------------------------------
print("\n[5/7] Construyendo CRS...", flush=True)

t0 = time.perf_counter()

graph = ref.build_crs_for_tau(
    docs,
    embeddings,
    e3.TAU
)

print(
    f"CRS construido en {time.perf_counter() - t0:.2f} s",
    flush=True
)

print("CRS nodes:", graph.number_of_nodes())
print("CRS edges:", graph.number_of_edges())

# ------------------------------------------------------------
# 6. Validación exacta contra E3
# ------------------------------------------------------------
print("\n[6/7] Validando contra E3...", flush=True)

result = e3.metrics(graph)
result["n_documents"] = len(corpus)
gate = e3.full_reference_gate(result)

print("Full graph:")
print("  nodes:", result["nodes_full_graph"])
print("  edges:", result["edges_full_graph"])
print("  components:", result["n_components_full_graph"])

print("Backbone:")
print("  nodes:", result["backbone_nodes"])
print("  edges:", result["backbone_edges"])
print("  components:", result["backbone_components"])
print("  modularity:", result["modularity"])
print("  communities:", result["n_communities"])

if not gate["passed"]:
    print("\nERROR: la reconstrucción NO reproduce la referencia E3.")
    print(gate)
    raise RuntimeError(
        "E8 detenido: la reconstrucción CRS no supera el full_reference_gate."
    )

print("\nREFERENCE GATE: PASSED")

# ------------------------------------------------------------
# 7. Backbone, Louvain y exportación
# ------------------------------------------------------------
print("\n[7/7] Exportando backbone y comunidades...", flush=True)

backbone = ref.build_backbone(graph, e3.BACKBONE)

assert backbone.number_of_nodes() == 408
assert backbone.number_of_edges() == 608

partition = community_louvain.best_partition(
    backbone,
    weight="weight",
    random_state=e3.LOUVAIN_SEED
)

modularity = community_louvain.modularity(
    partition,
    backbone,
    weight="weight"
)

assert len(set(partition.values())) == 7
assert abs(modularity - 0.3560443171946043) < 1e-6

# Keyword -> comunidad
df_comm = pd.DataFrame({
    "keyword": list(partition.keys()),
    "community": list(partition.values())
}).sort_values(
    ["community", "keyword"]
).reset_index(drop=True)

community_path = OUT / "crs_keyword_communities.csv"
df_comm.to_csv(community_path, index=False)

# Aristas del backbone
edge_rows = []

for u, v, d in backbone.edges(data=True):
    edge_rows.append({
        "source": u,
        "target": v,
        "weight": d.get("weight"),
        "sim_count": d.get("sim_count"),
        "sim_mean": d.get("sim_mean")
    })

df_edges = pd.DataFrame(edge_rows).sort_values(
    ["source", "target"]
).reset_index(drop=True)

edges_path = OUT / "crs_backbone_edges.csv"
df_edges.to_csv(edges_path, index=False)

# Resumen de comunidades
df_sizes = (
    df_comm.groupby("community")
    .size()
    .rename("n_keywords")
    .reset_index()
    .sort_values("n_keywords", ascending=False)
)

sizes_path = OUT / "crs_community_sizes.csv"
df_sizes.to_csv(sizes_path, index=False)

print("\n" + "=" * 70)
print("E8 CRS EXPORT COMPLETE")
print("=" * 70)

print("Backbone nodes:", backbone.number_of_nodes())
print("Backbone edges:", backbone.number_of_edges())
print("Communities:", len(set(partition.values())))
print("Modularity:", modularity)

print("\nCommunity sizes:")
print(df_sizes.to_string(index=False))

print("\nArchivos:")
print(community_path)
print(edges_path)
print(sizes_path)
