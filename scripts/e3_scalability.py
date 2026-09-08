"""E3: independent corpus-size replicates, gated by full-corpus reproduction.

Run: .venv/bin/python -B -u scripts/e3_scalability.py
Use --reference-only to stop after the full-corpus gate, or --resume to reuse
checksummed completed runs with identical code/configuration/inputs/versions.
No network/model downloads, LLM calls, or original-file writes are performed.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/e3_scalability"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MPLCONFIGDIR"] = str(OUT / ".matplotlib")

from e3_runtime import activate
RUNTIME_ISOLATION = activate()

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import psutil
import torch
from sentence_transformers import SentenceTransformer
import crs_reference as ref

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
TAU = 0.40
BACKBONE = 20
LOUVAIN_SEED = 42
SEEDS = list(range(42, 52))
SIZES = [500, 1000, 5000, 10000, 25000]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def protected():
    paths = list(ROOT.glob("[1-7]. *.ipynb")) + [ROOT / name for name in
             ["EID_KEYWORDS.xlsx", "dataset_inspec.csv", "inspec_llama-3.1-8b-EN.csv", "human_eval_M1_8b.csv"]]
    return {p.name: digest(p) for p in sorted(paths)}


def reference_audit():
    checks = {}
    for filename, function in [("2. CRS.ipynb", "parse_keywords"),
                               ("3. w_THRESHOLDS.ipynb", "build_backbone"),
                               ("5. Tau_SENSITIVITY.ipynb", "build_crs_for_tau")]:
        nb = json.loads((ROOT / filename).read_text())
        source = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == function)
        actual = ast.parse(inspect.getsource(getattr(ref, function))).body[0]
        assert ast.dump(node, include_attributes=False) == ast.dump(actual, include_attributes=False)
        checks[function] = {"notebook": filename, "ast_identical": True}
    return checks


def load_corpus():
    raw = pd.read_excel(ROOT / "EID_KEYWORDS.xlsx")
    assert raw.columns.tolist() == ["EID_o_identificador", "palabras_clave"]
    assert raw.EID_o_identificador.notna().all() and raw.EID_o_identificador.is_unique
    raw["source_row_abs"] = np.arange(len(raw))
    parsed = raw.palabras_clave.apply(ref.parse_keywords)
    retained = parsed.apply(len) > 0
    # Exactly the notebook's string filtering, strip/lower, and sorted-set deduplication.
    cleaned = parsed.apply(lambda kws: sorted(set(k.strip().lower() for k in kws if isinstance(k, str) and k.strip())))
    corpus = raw.loc[retained, ["source_row_abs", "EID_o_identificador"]].copy().reset_index(drop=True)
    corpus["keywords"] = cleaned.loc[retained].tolist()
    corpus["corpus_position"] = np.arange(len(corpus))
    audit = raw[["source_row_abs", "EID_o_identificador"]].copy()
    audit["parsed_keyword_count"] = parsed.apply(len)
    audit["clean_keyword_count"] = cleaned.apply(len)
    audit["included_in_reference_corpus"] = retained
    audit["exclusion_reason"] = np.where(retained, "", "empty parsed list: original notebook 2 filter")
    return corpus, audit


def sampled_positions(total, size, seed):
    # Independent size-specific RNG streams, never slices of a shared permutation.
    # Seed is still 42..51; both entropy components are explicitly recorded.
    rng = np.random.default_rng(np.random.SeedSequence([seed, size]))
    return np.sort(rng.choice(total, size=size, replace=False))


class MemorySampler:
    """Approximate per-run resident-memory peak, sampled every 50 ms."""
    def __enter__(self):
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(0.05):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def component_stats(graph):
    n, e = graph.number_of_nodes(), graph.number_of_edges()
    components = list(nx.connected_components(graph)) if n else []
    # Notebook 5 convention: an edgeless graph reports LCC size/fraction as zero.
    if n and e:
        largest = graph.subgraph(max(components, key=len))
        return len(components), largest.number_of_nodes(), largest.number_of_edges(), largest.number_of_nodes() / n
    return len(components), 0, 0, 0.0


def metrics(graph):
    n, e = graph.number_of_nodes(), graph.number_of_edges()
    components, ln, le, lf = component_stats(graph)
    backbone = ref.build_backbone(graph, BACKBONE)
    hn, he = backbone.number_of_nodes(), backbone.number_of_edges()
    hc, hln, hle, hlf = component_stats(backbone)
    assert nx.number_of_selfloops(graph) == 0
    assert all(d["weight"] >= BACKBONE for _, _, d in backbone.edges(data=True))
    assert all(backbone.degree(node) > 0 for node in backbone)
    assert all(isinstance(d["weight"], int) and d["weight"] == d["sim_count"] for _, _, d in graph.edges(data=True))
    if hn and he:
        partition = community_louvain.best_partition(backbone, weight="weight", random_state=LOUVAIN_SEED)
        modularity = community_louvain.modularity(partition, backbone, weight="weight")
        communities = len(set(partition.values()))
    else:
        modularity, communities = float("nan"), 0
    return {"nodes_full_graph": n, "edges_full_graph": e, "density_full_graph": nx.density(graph),
            "n_components_full_graph": components, "lcc_nodes": ln, "lcc_edges": le,
            "lcc_fraction_nodes": lf, "modularity": modularity, "n_communities": communities,
            "mean_degree": float(np.mean([d for _, d in graph.degree()])) if n else float("nan"),
            "mean_weighted_degree": float(np.mean([d for _, d in graph.degree(weight="weight")])) if n else float("nan"),
            "clustering_coefficient": nx.average_clustering(graph, weight=None, count_zeros=True) if n else float("nan"),
            "backbone_nodes": hn, "backbone_edges": he, "backbone_components": hc,
            "backbone_density": nx.density(backbone), "backbone_lcc_nodes": hln,
            "backbone_lcc_edges": hle, "backbone_lcc_fraction_nodes": hlf,
            "backbone_mean_degree": float(np.mean([d for _, d in backbone.degree()])) if hn else float("nan"),
            "backbone_mean_weighted_degree": float(np.mean([d for _, d in backbone.degree(weight="weight")])) if hn else float("nan"),
            "backbone_empty": hn == 0}


def execute_run(corpus, positions, replicate, seed, model):
    sample = corpus.iloc[positions]
    n = len(positions)
    assert sample.source_row_abs.is_unique and sample.EID_o_identificador.is_unique
    label = f"n{n}_r{replicate}_s{seed}"
    sample[["corpus_position", "source_row_abs", "EID_o_identificador"]].to_csv(OUT / "samples" / f"{label}.csv", index=False)
    gc.collect()
    started = time.perf_counter()
    with MemorySampler() as memory:
        docs = sample.keywords.tolist()
        vocab = sorted(set(k for kws in docs for k in kws))
        before_embedding = time.perf_counter()
        embedding_matrix = model.encode(vocab, batch_size=32, normalize_embeddings=True,
                                        show_progress_bar=False, convert_to_numpy=True)
        embedding_seconds = time.perf_counter() - before_embedding
        print(f"{label}: {len(vocab)} embeddings en {embedding_seconds:.2f}s", flush=True)
        before_graph = time.perf_counter()
        embeddings = {kw: embedding_matrix[i] for i, kw in enumerate(vocab)}
        graph = ref.build_crs_for_tau(docs, embeddings, TAU)
        graph_seconds = time.perf_counter() - before_graph
        before_metrics = time.perf_counter()
        result = metrics(graph)
        metrics_seconds = time.perf_counter() - before_metrics
        total_seconds = time.perf_counter() - started
    result.update({"n_documents": n, "replicate": replicate, "seed": seed, "tau": TAU,
                   "backbone_threshold": BACKBONE, "louvain_seed": LOUVAIN_SEED,
                   "vocabulary_size": len(vocab), "runtime_total_seconds": total_seconds,
                   "runtime_embedding_seconds": embedding_seconds,
                   "runtime_graph_construction_seconds": graph_seconds,
                   "runtime_metrics_seconds": metrics_seconds,
                   "peak_memory_rss_mib": memory.peak / 2**20,
                   "baseline_memory_rss_mib": memory.baseline / 2**20,
                   "peak_memory_increase_mib": (memory.peak - memory.baseline) / 2**20,
                   "sample_file": f"samples/{label}.csv",
                   "sample_sha256": digest(OUT / "samples" / f"{label}.csv")})
    print(f"{label}: V={result['nodes_full_graph']} E={result['edges_full_graph']} "
          f"backbone={result['backbone_nodes']}/{result['backbone_edges']} total={total_seconds:.2f}s", flush=True)
    return result


def full_reference_gate(result):
    # Values are stored outputs of notebooks 2, 3, 4 and tau=0.4 in notebook 5.
    expected = {"n_documents": 52946, "nodes_full_graph": 56635, "edges_full_graph": 109022,
                "n_components_full_graph": 19856, "lcc_nodes": 34316, "lcc_edges": 106297,
                "lcc_fraction_nodes": 34316 / 56635,
                "density_full_graph": 2 * 109022 / (56635 * 56634),
                "backbone_nodes": 408, "backbone_edges": 608, "backbone_components": 2,
                "backbone_lcc_nodes": 406, "backbone_lcc_edges": 607,
                "backbone_lcc_fraction_nodes": 406 / 408,
                "backbone_density": 2 * 608 / (408 * 407),
                "backbone_mean_degree": 2.980392156862745,
                "backbone_mean_weighted_degree": 188.84803921568627,
                "modularity": 0.3560443171946043, "n_communities": 7}
    checks = []
    for key, value in expected.items():
        tolerance = 0 if isinstance(value, int) else (1e-6 if key == "modularity" else 1e-10)
        actual = result[key]
        checks.append({"metric": key, "expected": value, "actual": actual,
                       "absolute_tolerance": tolerance, "passed": abs(actual - value) <= tolerance})
    return {"passed": all(c["passed"] for c in checks), "checks": checks,
            "sources": ["2. CRS.ipynb stored output", "3. w_THRESHOLDS.ipynb stored table",
                        "4. METRICS.ipynb stored summary", "5. Tau_SENSITIVITY.ipynb tau=0.4 stored table"]}


def summarize(runs):
    excluded = {"n_documents", "replicate", "seed", "tau", "backbone_threshold", "louvain_seed"}
    numeric = [c for c in runs.select_dtypes(include=["number", "bool"]).columns if c not in excluded]
    rows = []
    for n, group in runs.groupby("n_documents", sort=True):
        for metric in numeric:
            values = group[metric].astype(float)
            mean, sd = values.mean(), values.std(ddof=1)
            rows.append({"n_documents": n, "metric": metric, "n_runs": len(group),
                         "n_valid": int(values.notna().sum()), "mean": mean, "sd": sd,
                         "min": values.min(), "max": values.max(),
                         "cv": sd / mean if mean > 0 else float("nan")})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "summary.csv", index=False)
    table[table.metric.str.startswith("runtime_")].to_csv(OUT / "runtime_summary.csv", index=False)
    return table


def validate_samples(runs, corpus, full_required):
    assert (runs.tau == 0.40).all() and (runs.backbone_threshold == 20).all()
    assert (runs.louvain_seed == 42).all()
    assert not runs.duplicated(["n_documents", "replicate"]).any()
    details = []
    for row in runs.itertuples(index=False):
        path = OUT / row.sample_file
        assert digest(path) == row.sample_sha256
        saved = pd.read_csv(path)
        expected = np.arange(len(corpus)) if row.n_documents == len(corpus) else sampled_positions(len(corpus), row.n_documents, row.seed)
        assert len(saved) == row.n_documents
        assert saved.corpus_position.is_unique and saved.EID_o_identificador.is_unique
        assert saved.corpus_position.tolist() == expected.tolist()
        assert saved.source_row_abs.tolist() == corpus.iloc[expected].source_row_abs.tolist()
        assert saved.EID_o_identificador.tolist() == corpus.iloc[expected].EID_o_identificador.tolist()
        assert row.seed == (42 if row.n_documents == len(corpus) else 41 + row.replicate)
        details.append({"n_documents": row.n_documents, "replicate": row.replicate, "seed": row.seed, "passed": True})
    if full_required:
        assert len(runs) == 51
        for n in SIZES:
            group = runs[runs.n_documents == n]
            assert len(group) == 10 and sorted(group.seed.tolist()) == SEEDS
        assert (runs.n_documents == len(corpus)).sum() == 1
    return details


def figures(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    definitions = [("runtime", [("runtime_total_seconds", "Total"), ("runtime_embedding_seconds", "Embeddings")], "Seconds"),
                   ("nodes_edges", [("nodes_full_graph", "Nodes"), ("edges_full_graph", "Edges")], "Count"),
                   ("lcc_fraction", [("lcc_fraction_nodes", "Full graph"), ("backbone_lcc_fraction_nodes", "Backbone")], "LCC / graph nodes"),
                   ("modularity", [("modularity", "Backbone Louvain modularity")], "Weighted modularity"),
                   ("backbone_nodes_edges", [("backbone_nodes", "Nodes"), ("backbone_edges", "Edges")], "Backbone count")]
    (OUT / "figures").mkdir(exist_ok=True)
    for filename, series, ylabel in definitions:
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        for metric, label in series:
            values = summary[summary.metric == metric].sort_values("n_documents")
            line, = ax.plot(values.n_documents, values["mean"], "o-", label=label)
            valid = values.sd.notna()
            ax.errorbar(values.loc[valid, "n_documents"], values.loc[valid, "mean"],
                        yerr=values.loc[valid, "sd"], fmt="none", capsize=3, color=line.get_color())
        ax.set(xlabel="Documents in analyzed corpus", ylabel=ylabel,
               title="Fixed CRS τ=0.40; backbone support ≥20")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.text(0.5, 0.005, "Error bars: ±1 sample SD (10 replicates); full corpus: one run, no SD.", ha="center", fontsize=8)
        for extension in ("png", "pdf"):
            fig.savefig(OUT / "figures" / f"{filename}.{extension}", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "ieee-revision-experiments":
        raise RuntimeError(f"Rama inesperada: {branch}; no se cambia automáticamente")
    hashes = protected()
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), ROOT / "scripts/crs_reference.py", ROOT / "scripts/e3_runtime.py"]}
    reference_checks = reference_audit()
    print("Leyendo corpus original...", flush=True)
    before_setup = time.perf_counter()
    corpus, corpus_audit = load_corpus()
    print(f"Filas originales={len(corpus_audit)}, N_total analizable={len(corpus)}", flush=True)
    assert all(n < len(corpus) for n in SIZES)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "samples").mkdir(exist_ok=True)
    corpus_audit.to_csv(OUT / "corpus_audit.csv", index=False)
    versions = {name: importlib.metadata.version(name) for name in
                ["numpy", "pandas", "scipy", "scikit-learn", "networkx", "python-louvain", "torch",
                 "sentence-transformers", "transformers", "huggingface-hub", "openpyxl", "matplotlib", "psutil"]}
    configuration = {"model": MODEL, "model_revision": REVISION, "tau": TAU, "backbone_threshold": BACKBONE,
                     "louvain_seed": LOUVAIN_SEED, "louvain_weight": "weight", "louvain_resolution": 1.0,
                     "sizes": SIZES + [len(corpus)], "replicate_seeds": SEEDS, "full_corpus_runs": 1,
                     "sampling": "default_rng(SeedSequence([seed,n])).choice(N,n,replace=False); sort original row order",
                     "sampling_nested": False, "embedding_batch_size": 32, "normalize_embeddings": True,
                     "embedding_reuse_between_runs": False, "embedding_strategy": "unique sorted vocabulary once per run, as notebook 5",
                     "device": "cpu", "torch_threads": 4, "random_seed": 42,
                     "modularity_and_n_communities_scope": "backbone, exactly as notebooks 4/5",
                     "mean_degree_and_mean_weighted_degree_scope": "full graph; same degree formulas, additional scope",
                     "clustering_coefficient": "additional descriptive nx.average_clustering(full_graph,weight=None,count_zeros=True); not in original notebooks",
                     "empty_backbone": "retain zero counts/fraction; modularity and mean degrees undefined (NaN), no lowered threshold",
                     "edgeless_lcc_convention": "0 nodes/edges/fraction, as notebook 5",
                     "runtime_scope": "per-run vocabulary, embedding, graph aggregation, backbone and metrics; shared loading and sample CSV I/O excluded",
                     "sd_ddof": 1, "singleton_sd": None, "peak_memory": "RSS sampled every 50ms; includes shared loaded model; not isolated allocation peak"}
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "branch": branch,
                "python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
                "packages": versions, "raw_excel_rows": len(corpus_audit), "analyzed_corpus_rows": len(corpus),
                "excluded_empty_parsed_lists": int((~corpus_audit.included_in_reference_corpus).sum()),
                "protected_sha256": hashes, "source_sha256": source_hashes,
                "runtime_isolation": RUNTIME_ISOLATION,
                "llm_calls": 0, "paid_api_calls": 0, "model_downloads": 0, "status": "running"}
    completed = []
    if (OUT / "runs.csv").exists():
        if not args.resume:
            raise RuntimeError("Ya existe runs.csv; usar --resume para reanudar validando integridad")
        previous = json.loads((OUT / "metadata.json").read_text())
        assert previous["protected_sha256"] == hashes and previous["source_sha256"] == source_hashes
        assert previous["packages"] == versions
        assert json.loads((OUT / "configuration.json").read_text()) == configuration
        completed = pd.read_csv(OUT / "runs.csv").to_dict("records")
        validate_samples(pd.DataFrame(completed), corpus, full_required=False)
        if not full_reference_gate(completed[0])["passed"]:
            raise RuntimeError("La ejecución completa guardada no supera la referencia; diagnosticar antes de reanudar")
        metadata["resumed_from_started_utc"] = previous["started_utc"]
    json_write(OUT / "configuration.json", configuration)
    json_write(OUT / "metadata.json", metadata)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    try:
        model = SentenceTransformer(MODEL, revision=REVISION, device="cpu", local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f"No se pudo cargar el modelo local {MODEL}@{REVISION}; no se descargará automáticamente: {exc}") from exc
    metadata["shared_setup_seconds"] = time.perf_counter() - before_setup
    metadata["model_max_seq_length"] = model.max_seq_length
    json_write(OUT / "metadata.json", metadata)
    schedule = [(len(corpus), 1, 42)] + [(n, i + 1, seed) for n in SIZES for i, seed in enumerate(SEEDS)]
    for n, replicate, seed in schedule:
        if args.reference_only and n != len(corpus):
            break
        if any(r["n_documents"] == n and r["replicate"] == replicate for r in completed):
            continue
        positions = np.arange(len(corpus)) if n == len(corpus) else sampled_positions(len(corpus), n, seed)
        assert len(positions) == n and len(np.unique(positions)) == n
        print(f"Ejecutando n={n}, réplica={replicate}, seed={seed}", flush=True)
        result = execute_run(corpus, positions, replicate, seed, model)
        completed.append(result)
        runs = pd.DataFrame(completed)
        runs.to_csv(OUT / "runs.csv", index=False)
        gate = full_reference_gate(completed[0])
        validation = {"full_corpus_reference": gate, "reference_functions": reference_checks,
                      "protected_artifacts_unchanged": protected() == hashes,
                      "completed_runs": len(completed), "tau_exact": True, "backbone_threshold_exact": True}
        json_write(OUT / "validation.json", validation)
        assert validation["protected_artifacts_unchanged"]
        if not gate["passed"]:
            metadata["status"] = "stopped_reference_mismatch"
            json_write(OUT / "metadata.json", metadata)
            print(json.dumps(gate, indent=2), flush=True)
            raise RuntimeError("DETENIDO: CRS completo no reproduce referencia; no se ejecutan réplicas ni se interpreta E3")
    runs = pd.DataFrame(completed)
    sample_checks = validate_samples(runs, corpus, full_required=not args.reference_only)
    summary = summarize(runs)
    validation = {"passed": True, "full_corpus_reference": full_reference_gate(completed[0]),
                  "reference_functions": reference_checks, "sample_checks": sample_checks,
                  "protected_artifacts_unchanged": protected() == hashes, "tau_exact": bool((runs.tau == 0.4).all()),
                  "backbone_threshold_exact": bool((runs.backbone_threshold == 20).all()),
                  "completed_runs": len(runs), "expected_runs": 51,
                  "experiment_complete": len(runs) == 51,
                  "empty_backbone_runs": int(runs.backbone_empty.sum()),
                  "undefined_modularity_runs": int(runs.modularity.isna().sum())}
    assert validation["protected_artifacts_unchanged"]
    json_write(OUT / "validation.json", validation)
    metadata["status"] = "complete" if len(runs) == 51 else "reference_validated"
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    json_write(OUT / "metadata.json", metadata)
    if not args.reference_only:
        figures(summary)
    print(f"Finalizado: {metadata['status']}, {len(runs)} ejecuciones", flush=True)


if __name__ == "__main__":
    main()
