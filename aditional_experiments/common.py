"""Shared utilities for the additional revision experiments (E2, E4, E5, E6, E7, E9).

Design rules
------------
* Reference semantics are imported, never re-implemented: graph construction comes from
  ``scripts/crs_reference.py`` (verbatim copies of notebooks 2, 3 and 5) and phrase
  normalisation / soft-matching from ``scripts/inspec_evaluation.py`` (notebook 6).
* Every script writes ``metadata.json`` (environment, input hashes, timings) and
  ``validation.json`` (reproduction gates) next to its results.
* Scopus text (the ``insumo`` records) is licensed and never written to disk here; only
  aggregates and per-document scores keyed by Scopus EID are produced.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

AE_ROOT = Path(__file__).resolve().parent          # aditional_experiments/
REPO_ROOT = AE_ROOT.parent
RESULTS = AE_ROOT / "results"
DATA = AE_ROOT / "data"
sys.path.insert(0, str(REPO_ROOT / "scripts"))     # crs_reference, inspec_evaluation

# Private inputs (Scopus records, Elsevier licence) are read from this directory and are
# never copied into the repository. Override with FTTS_PRIVATE_DIR.
PRIVATE_DIR = Path(os.environ.get("FTTS_PRIVATE_DIR", REPO_ROOT.parent))

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
TAU_EDGE = 0.40       # CRS semantic edge threshold (paper)
W_BACKBONE = 20       # CRS backbone support threshold (paper)
TAU_SOFT = 0.70       # soft-match threshold for keyword-level comparisons (paper, notebook 6)
SEED = 42
FIELD_SEP = " • "
TRUNCATE_CHARS = 3000  # notebook 1: article text inserted into the prompt is cut here

# Reference values of the published CRS (notebooks 2-5; also the E3 gate).
CRS_REFERENCE = {
    "analyzed_documents": 52946,
    "nodes": 56635,
    "edges": 109022,
    "components": 19856,
    "lcc_nodes": 34316,
    "backbone_nodes": 408,
    "backbone_edges": 608,
    "backbone_modularity": 0.3560443171946043,
    "backbone_communities": 7,
}

# Reference Inspec means for LLaMA 3.1 8B (notebook 6; E1 gate).
INSPEC_REFERENCE = {
    "jaccard_lex": 0.1422, "soft_precision": 0.7939, "soft_recall": 0.4670,
    "soft_f1": 0.5652, "soft_mean_max": 0.7522, "global_sem_sim": 0.8042,
}


# --------------------------------------------------------------------------- utilities
def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seeds(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:
        return None


def env_metadata(**extra) -> dict:
    pkgs = {}
    for m in ["numpy", "pandas", "scipy", "sklearn", "networkx", "community",
              "sentence_transformers", "torch", "transformers", "langdetect", "openai"]:
        try:
            mod = importlib.import_module(m)
            pkgs[m] = getattr(mod, "__version__", "unknown")
        except Exception:
            pkgs[m] = None
    return {
        "started_utc": now_utc(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "git_commit": git_commit(),
        "packages": pkgs,
        "embedding_model": MODEL_NAME,
        "embedding_model_revision": MODEL_REVISION,
        **extra,
    }


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serialisable: {type(o)}")


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.marks = {}

    def mark(self, name):
        self.marks[name] = round(time.perf_counter() - self.t0, 3)
        return self.marks[name]


# --------------------------------------------------------------------------- embeddings
def load_embedder(threads: int | None = None):
    import torch
    from sentence_transformers import SentenceTransformer
    if threads:
        torch.set_num_threads(threads)
    model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device="cpu")
    return model


def embed_unique(model, phrases, batch_size: int = 128, normalize: bool = True,
                 desc: str = "embedding") -> dict:
    """Embed each unique string once; returns {phrase: vector}."""
    uniq = sorted(set(p for p in phrases if p))
    if not uniq:
        return {}
    vecs = model.encode(uniq, batch_size=batch_size, show_progress_bar=True,
                        normalize_embeddings=normalize, convert_to_numpy=True)
    return dict(zip(uniq, vecs))


# --------------------------------------------------------------------------- records
_ws = re.compile(r"\s+")
_sent_split = re.compile(r"(?<=[.!?])\s+")


def parse_record(insumo: str) -> dict:
    """Split one ``insumo`` record into its six Scopus fields.

    Notebook 1 concatenates: authors • title • year • source • abstract • original keywords.
    A small number of records contain the separator inside the abstract; the first four
    fields and the last field are positional, the abstract is everything in between.
    """
    s = "" if insumo is None or (isinstance(insumo, float) and np.isnan(insumo)) else str(insumo)
    parts = s.split(FIELD_SEP)
    n = len(parts)
    if n < 6:
        parts = parts + [""] * (6 - n)
    # The year is the only purely numeric field; when the separator also occurs inside the title
    # (5 records) or the abstract (183 records) it anchors the split: title = fields before it,
    # source = the field after it, abstract = everything between source and the last field.
    k = next((i for i in range(2, max(3, n - 3)) if re.fullmatch(r"\d{4}", parts[i].strip())), 2)
    rec = {
        "authors": parts[0].strip(),
        "title": FIELD_SEP.join(parts[1:k]).strip(),
        "year": parts[k].strip(),
        "source": parts[k + 1].strip() if k + 1 < n else "",
        "abstract": FIELD_SEP.join(parts[k + 2:-1]).strip() if n >= 6 else parts[4].strip(),
        "original_keywords_raw": parts[-1].strip() if n >= 6 else "",
        "n_fields": n,
        "length": len(s),
    }
    # Character offset at which the original-keyword field starts (for truncation analysis).
    rec["keyword_field_start"] = s.rfind(FIELD_SEP) + len(FIELD_SEP) if n >= 6 else len(s)
    return rec


def split_original_keywords(raw: str) -> list[str]:
    """Scopus author + index keywords are '; '-separated in the record."""
    from inspec_evaluation import normalize_phrase
    out, seen = [], set()
    for k in str(raw).split(";"):
        t = normalize_phrase(k)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def split_sentences(text: str) -> list[str]:
    text = _ws.sub(" ", str(text)).strip()
    if not text:
        return []
    return [s for s in _sent_split.split(text) if s.strip()]


def load_keywords_xlsx(path: Path, notebook_semantics: bool = True) -> pd.DataFrame:
    """Load an EID/keyword workbook. With ``notebook_semantics`` the lists are cleaned as in
    notebook 2 (non-empty strings, strip().lower(), sorted(set())) so graphs reproduce exactly."""
    from crs_reference import parse_keywords
    df = pd.read_excel(path)
    eid_col = [c for c in df.columns if "eid" in c.lower()][0]
    kw_col = [c for c in df.columns if "palabra" in c.lower() or "keyword" in c.lower()][0]
    parsed = [parse_keywords(x) for x in df[kw_col]]
    if notebook_semantics:
        cleaned = [sorted(set(str(k).strip().lower() for k in kws if isinstance(k, str) and k.strip()))
                   for kws in parsed]
    else:
        cleaned = [[str(k) for k in kws] for kws in parsed]
    return pd.DataFrame({"eid": df[eid_col].astype(str), "keywords_raw": df[kw_col],
                         "keywords": cleaned, "source_row": np.arange(len(df))})


def load_alignment(path: Path | None = None, one_row_per_eid: bool = True) -> pd.DataFrame:
    """Row index of the private record file -> Scopus EID, restricted to rows whose saved output is
    identical to the published one (built by ``build_alignment.py``)."""
    df = pd.read_csv(path or (DATA / "insumo_row_to_eid.csv"))
    df = df[df.matches_published & df.eid.notna()].copy()
    if one_row_per_eid:
        df = df.drop_duplicates("eid", keep="first")
    df["insumo_row"] = df.insumo_row.astype(int)
    return df.reset_index(drop=True)


def load_published_keywords(path: Path | None = None, notebook_semantics: bool = False) -> dict:
    """{EID: keyword list} from the published EID_KEYWORDS.xlsx."""
    kw = load_keywords_xlsx(path or (REPO_ROOT / "EID_KEYWORDS.xlsx"), notebook_semantics=notebook_semantics)
    return dict(zip(kw.eid, kw.keywords))


# --------------------------------------------------------------------------- graphs
def graph_summary(G) -> dict:
    import networkx as nx
    n, m = G.number_of_nodes(), G.number_of_edges()
    comps = list(nx.connected_components(G)) if n else []
    lcc = max(comps, key=len) if comps else set()
    return {
        "nodes": n, "edges": m,
        "density": nx.density(G) if n > 1 else 0.0,
        "components": len(comps),
        "lcc_nodes": len(lcc),
        "lcc_fraction": (len(lcc) / n) if n else 0.0,
        "lcc_edges": G.subgraph(lcc).number_of_edges() if lcc else 0,
        "isolated_nodes": sum(1 for _, d in G.degree() if d == 0),
    }


def louvain_partition(H, seed: int = SEED):
    import community as community_louvain
    if H.number_of_edges() == 0:
        return {}, float("nan")
    part = community_louvain.best_partition(H, weight="weight", random_state=seed)
    q = community_louvain.modularity(part, H, weight="weight")
    return part, q


def partition_agreement(p1: dict, p2: dict) -> dict:
    """NMI / ARI on the nodes present in both partitions."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    shared = sorted(set(p1) & set(p2))
    if len(shared) < 2:
        return {"shared_nodes": len(shared), "nmi": float("nan"), "ari": float("nan")}
    a = [p1[n] for n in shared]
    b = [p2[n] for n in shared]
    return {"shared_nodes": len(shared),
            "nmi": float(normalized_mutual_info_score(a, b)),
            "ari": float(adjusted_rand_score(a, b))}
