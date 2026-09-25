"""E3 library: corpus loading, sampling, graph metrics and the reproduction gate of the
corpus-size experiment (Section IV-D2 and Table 8 of the manuscript).

Functions only. The experiment driver is inlined in ``13. CORPUS_SIZE_SCALABILITY.ipynb``;
``14. LDA_COMPARISON.ipynb`` reuses ``load_corpus``, ``metrics`` and ``full_reference_gate``
to rebuild the CRS before exporting its backbone partition. Graph construction, backbone
filtering and keyword parsing are imported from ``crs_reference`` (verbatim copies of
notebooks 2, 3 and 5); ``reference_audit`` compares their syntax trees with the notebooks.
No network access, model download, LLM call or write to an original file happens here.
"""
from __future__ import annotations

import ast
import gc
import hashlib
import inspect
import json
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import psutil
import crs_reference as ref

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/e3_scalability"

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
TAU = 0.40
BACKBONE = 20
LOUVAIN_SEED = 42
SEEDS = list(range(42, 52))
SIZES = [500, 1000, 5000, 10000, 25000]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def protected():
    """SHA-256 of the original notebooks and public inputs that the experiment must not alter."""
    paths = list(ROOT.glob("[1-7]. *.ipynb")) + [ROOT / name for name in
             ["EID_KEYWORDS.xlsx", "dataset_inspec.csv", "inspec_llama-3.1-8b-EN.csv", "human_eval_M1_8b.csv"]]
    return {p.name: digest(p) for p in sorted(paths)}


def reference_audit():
    """Check that the functions of crs_reference are AST-identical to the notebook cells they copy."""
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
    """Analysed corpus of notebook 2 (documents with a non-empty parsed keyword list) and a row audit."""
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
    """One timed run: vocabulary, embeddings, CRS, backbone and metrics on the given corpus positions."""
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
        print(f"{label}: {len(vocab)} embeddings in {embedding_seconds:.2f}s", flush=True)
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
    """Long-form summary (mean, sample SD, min, max, CV per size and metric); writes summary.csv and runtime_summary.csv."""
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
