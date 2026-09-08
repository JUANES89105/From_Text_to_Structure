from pathlib import Path
import ast
import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e8_topic_modeling"

KS = [5, 10, 15, 20, 30, 40, 50]

print("=" * 72)
print("E8 - Comparación LDA vs CRS")
print("=" * 72)

# ============================================================
# 1. Cargar datos
# ============================================================

print("\n[1/6] Cargando dataset y comunidades CRS...", flush=True)

t0 = time.perf_counter()

df = pd.read_pickle(OUT / "e8_matched_dataset.pkl")
comm = pd.read_csv(OUT / "crs_keyword_communities.csv")

print(
    f"Dataset: {len(df):,} documentos | "
    f"keywords backbone: {len(comm):,} | "
    f"{time.perf_counter()-t0:.2f}s",
    flush=True
)

assert len(df) == 52922
assert df["EID_clean"].is_unique
assert len(comm) == 408
assert comm["keyword"].is_unique
assert comm["community"].nunique() == 7

keyword_to_community = dict(
    zip(comm["keyword"], comm["community"])
)

# ============================================================
# 2. Normalización compatible con CRS
# ============================================================

def parse_keywords(value):
    if isinstance(value, (list, tuple, set, np.ndarray)):
        values = list(value)
    elif pd.isna(value):
        values = []
    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, set)):
                values = list(parsed)
            else:
                values = [value]
        except Exception:
            values = [value]
    else:
        values = []

    # Misma lógica fundamental del CRS:
    # strip + lowercase + deduplicación intradocumento
    return sorted({
        str(x).strip().lower()
        for x in values
        if str(x).strip()
    })

# ============================================================
# 3. Asignación documental CRS
# ============================================================

print("\n[2/6] Asignando comunidades CRS a documentos...", flush=True)

def assign_crs(value):
    kws = parse_keywords(value)

    mapped = [
        keyword_to_community[k]
        for k in kws
        if k in keyword_to_community
    ]

    if not mapped:
        return pd.Series({
            "crs_status": "unassigned",
            "crs_community": np.nan,
            "crs_mapped_keywords": 0,
            "crs_total_keywords": len(kws)
        })

    counts = Counter(mapped)
    max_count = max(counts.values())

    winners = [
        community
        for community, count in counts.items()
        if count == max_count
    ]

    if len(winners) > 1:
        return pd.Series({
            "crs_status": "ambiguous",
            "crs_community": np.nan,
            "crs_mapped_keywords": len(mapped),
            "crs_total_keywords": len(kws)
        })

    return pd.Series({
        "crs_status": "assigned",
        "crs_community": int(winners[0]),
        "crs_mapped_keywords": len(mapped),
        "crs_total_keywords": len(kws)
    })

t0 = time.perf_counter()

assignments = df["keywords_llm"].apply(assign_crs)
df = pd.concat([df, assignments], axis=1)

print(
    f"Asignación CRS completada en {time.perf_counter()-t0:.2f}s",
    flush=True
)

status_counts = df["crs_status"].value_counts()

n_total = len(df)
n_assigned = int((df["crs_status"] == "assigned").sum())
n_ambiguous = int((df["crs_status"] == "ambiguous").sum())
n_unassigned = int((df["crs_status"] == "unassigned").sum())

coverage = n_assigned / n_total

print("\nEstado CRS:")
print(f"  Total:       {n_total:,}")
print(f"  Assigned:    {n_assigned:,} ({n_assigned/n_total:.4%})")
print(f"  Ambiguous:   {n_ambiguous:,} ({n_ambiguous/n_total:.4%})")
print(f"  Unassigned:  {n_unassigned:,} ({n_unassigned/n_total:.4%})")

# ============================================================
# 4. Vectorización LDA
# ============================================================

print("\n[3/6] Cargando vectorizador y reconstruyendo matriz LDA...",
      flush=True)

t0 = time.perf_counter()

vectorizer = joblib.load(OUT / "lda_vectorizer.joblib")

X = vectorizer.transform(
    df["text_lda"].fillna("").astype(str)
)

print(
    f"Matriz LDA: {X.shape} | nnz={X.nnz:,} | "
    f"{time.perf_counter()-t0:.2f}s",
    flush=True
)

assert X.shape[0] == 52922
assert X.shape[1] == 30000

# Documentos válidos para NMI/ARI
mask = df["crs_status"].eq("assigned")

y_crs = (
    df.loc[mask, "crs_community"]
    .astype(int)
    .to_numpy()
)

# ============================================================
# 5. Comparación para cada K
# ============================================================

print("\n[4/6] Calculando asignaciones LDA y NMI/ARI...", flush=True)

results = []

for k in KS:

    print(f"\n  K={k}: cargando modelo...", flush=True)

    t0 = time.perf_counter()

    model = joblib.load(OUT / f"lda_k{k}.joblib")

    print(f"  K={k}: calculando P(topic|document)...", flush=True)

    theta = model.transform(X)

    dominant_topic = np.argmax(theta, axis=1)
    dominant_probability = np.max(theta, axis=1)

    df[f"lda_topic_k{k}"] = dominant_topic
    df[f"lda_probability_k{k}"] = dominant_probability

    y_lda = dominant_topic[mask.to_numpy()]

    nmi = normalized_mutual_info_score(y_crs, y_lda)
    ari = adjusted_rand_score(y_crs, y_lda)

    n_topics_observed = int(np.unique(y_lda).size)

    result = {
        "K": k,
        "n_documents_total": n_total,
        "n_crs_assigned": n_assigned,
        "n_crs_ambiguous": n_ambiguous,
        "n_crs_unassigned": n_unassigned,
        "crs_assignment_coverage": coverage,
        "n_documents_compared": len(y_crs),
        "n_crs_communities_observed": int(np.unique(y_crs).size),
        "n_lda_topics_observed": n_topics_observed,
        "NMI": nmi,
        "ARI": ari,
        "mean_dominant_topic_probability_all": float(
            dominant_probability.mean()
        ),
        "mean_dominant_topic_probability_compared": float(
            dominant_probability[mask.to_numpy()].mean()
        ),
        "runtime_transform_seconds": time.perf_counter() - t0
    }

    results.append(result)

    print(
        f"  K={k}: NMI={nmi:.6f} | "
        f"ARI={ari:.6f} | "
        f"topics observados={n_topics_observed} | "
        f"{result['runtime_transform_seconds']:.2f}s",
        flush=True
    )

# ============================================================
# 6. Guardar resultados
# ============================================================

print("\n[5/6] Guardando resultados...", flush=True)

comparison = pd.DataFrame(results)

# Dataset documental compacto:
doc_columns = [
    "EID_clean",
    "crs_status",
    "crs_community",
    "crs_mapped_keywords",
    "crs_total_keywords"
]

for k in KS:
    doc_columns.extend([
        f"lda_topic_k{k}",
        f"lda_probability_k{k}"
    ])

doc_output = df[doc_columns].copy()

doc_path = OUT / "crs_document_assignments.csv"
comparison_path = OUT / "lda_crs_comparison.csv"

doc_output.to_csv(doc_path, index=False)
comparison.to_csv(comparison_path, index=False)

# Resumen adicional de asignaciones CRS
crs_distribution = (
    df.loc[df["crs_status"] == "assigned", "crs_community"]
    .astype(int)
    .value_counts()
    .sort_index()
    .rename_axis("community")
    .reset_index(name="n_documents")
)

crs_distribution["proportion_assigned"] = (
    crs_distribution["n_documents"] / n_assigned
)

crs_distribution.to_csv(
    OUT / "crs_document_community_distribution.csv",
    index=False
)

print("\n[6/6] E8 DOCUMENT-LEVEL COMPARISON COMPLETE")
print("=" * 72)

print("\nComparación LDA–CRS:")
print(
    comparison[
        [
            "K",
            "n_documents_compared",
            "crs_assignment_coverage",
            "NMI",
            "ARI",
            "mean_dominant_topic_probability_all"
        ]
    ].to_string(index=False)
)

print("\nDistribución documental CRS:")
print(crs_distribution.to_string(index=False))

print("\nArchivos:")
print(doc_path)
print(comparison_path)
print(OUT / "crs_document_community_distribution.csv")

print("\nIMPORTANTE:")
print(
    "La selección de K permanece independiente de CRS: "
    "K=50 fue seleccionado previamente por el mayor NPMI "
    "entre los valores evaluados."
)
