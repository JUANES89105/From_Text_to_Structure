"""E9 - Ablation of the semantic filter: co-word CRS (no tau) versus the published CRS (tau = 0.40).

Addresses: Editor E-1 (novelty against co-word / bibliometric mapping), R3-5 (what the
semantic constraint adds to the original CRS of Gaona 2024), R2-3 (systematic justification
of tau and of the backbone support threshold w).

The original CRS formulation connects every pair of concepts that co-occur in a document.
The published pipeline keeps a co-occurring pair only if cos(e_u, e_v) >= 0.40. Running the
unchanged reference constructor with tau = -1 (cosine is bounded below by -1, so no pair is
dropped) yields the pure co-word network on the same keywords, same corpus, same aggregation.
Everything downstream (backbone w >= 20, Louvain seed 42) is held fixed. The difference is
therefore the isolated contribution of the semantic constraint.

Inputs (public): EID_KEYWORDS.xlsx. No LLM calls, no paid API.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from crs_reference import build_backbone, build_crs_for_tau  # noqa: E402

OUT = C.RESULTS / "e9_semantic_filter_ablation"
W_SWEEP = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50]
LOUVAIN_SEEDS = list(range(42, 52))
SIM_BINS = np.round(np.arange(-0.2, 1.0001, 0.05), 2)


def backbone_block(G, wmin, seeds=(C.SEED,)):
    H = build_backbone(G, wmin)
    s = C.graph_summary(H)
    parts, qs, ncs = {}, [], []
    for sd in seeds:
        p, q = C.louvain_partition(H, sd)
        parts[sd] = p
        qs.append(q)
        ncs.append(len(set(p.values())) if p else 0)
    s.update({
        "w_min": wmin,
        "modularity_seed42": qs[0],
        "communities_seed42": ncs[0],
        "modularity_mean": float(np.nanmean(qs)) if len(qs) > 1 else qs[0],
        "modularity_sd": float(np.nanstd(qs, ddof=1)) if len(qs) > 1 else float("nan"),
        "communities_mean": float(np.mean(ncs)) if len(ncs) > 1 else ncs[0],
        "total_weight": float(sum(d["weight"] for _, _, d in H.edges(data=True))),
    })
    return H, s, parts[seeds[0]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keywords", default=str(C.REPO_ROOT / "EID_KEYWORDS.xlsx"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-w1", action="store_true",
                    help="skip Louvain on the w=1 graphs (slow on the unfiltered network)")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    C.set_seeds()
    T = C.Timer()
    meta = C.env_metadata(experiment="E9 semantic filter ablation",
                          inputs={"keywords": {"path": args.keywords, "sha256": C.sha256(args.keywords)}},
                          llm_calls=0, paid_api_calls=0)

    # ------------------------------------------------------------------ data
    kw = C.load_keywords_xlsx(Path(args.keywords), notebook_semantics=True)
    docs = [k for k in kw["keywords"] if len(k) > 0]           # notebook 2 drops empty lists
    n_docs = len(docs)
    vocab = sorted(set(itertools.chain.from_iterable(docs)))
    print(f"documents={n_docs} vocabulary={len(vocab)}", flush=True)
    T.mark("load")

    model = C.load_embedder(threads=args.threads)
    emb = C.embed_unique(model, vocab, batch_size=256, normalize=True, desc="vocabulary")
    T.mark("embeddings")

    # ------------------------------------------------------------------ graphs
    G40 = build_crs_for_tau(docs, emb, C.TAU_EDGE)     # published CRS
    T.mark("build_tau040")
    G00 = build_crs_for_tau(docs, emb, -1.0)           # co-word: every co-occurring pair
    T.mark("build_coword")

    summ40, summ00 = C.graph_summary(G40), C.graph_summary(G00)
    H40, bb40, part40 = backbone_block(G40, C.W_BACKBONE, LOUVAIN_SEEDS)
    H00, bb00, part00 = backbone_block(G00, C.W_BACKBONE, LOUVAIN_SEEDS)
    T.mark("backbones")

    # ------------------------------------------------------------------ reproduction gate
    ref = C.CRS_REFERENCE
    gate = {
        "analyzed_documents": (n_docs, ref["analyzed_documents"]),
        "nodes": (summ40["nodes"], ref["nodes"]),
        "edges": (summ40["edges"], ref["edges"]),
        "components": (summ40["components"], ref["components"]),
        "lcc_nodes": (summ40["lcc_nodes"], ref["lcc_nodes"]),
        "backbone_nodes": (bb40["nodes"], ref["backbone_nodes"]),
        "backbone_edges": (bb40["edges"], ref["backbone_edges"]),
        "backbone_communities": (bb40["communities_seed42"], ref["backbone_communities"]),
    }
    checks = {k: {"observed": o, "reference": r, "pass": o == r} for k, (o, r) in gate.items()}
    checks["backbone_modularity"] = {
        "observed": bb40["modularity_seed42"], "reference": ref["backbone_modularity"],
        "pass": abs(bb40["modularity_seed42"] - ref["backbone_modularity"]) < 1e-6}
    gate_pass = all(v["pass"] for v in checks.values())
    print("REPRODUCTION GATE (tau=0.40):", "PASS" if gate_pass else "FAIL", flush=True)
    for k, v in checks.items():
        print(f"  {k:22s} observed={v['observed']} reference={v['reference']} {'ok' if v['pass'] else 'MISMATCH'}")

    # ------------------------------------------------------------------ filter effect
    # Every edge of G00 carries the mean cosine of its instances; G40 keeps instances with cos >= 0.40.
    rows = []
    for u, v, d in G00.edges(data=True):
        kept = G40.get_edge_data(u, v)
        rows.append((u, v, d["weight"], d["sim_mean"], kept["weight"] if kept else 0))
    E = pd.DataFrame(rows, columns=["u", "v", "weight_coword", "sim_mean", "weight_tau040"])
    E["removed_instances"] = E["weight_coword"] - E["weight_tau040"]
    total_inst = int(E["weight_coword"].sum())
    kept_inst = int(E["weight_tau040"].sum())
    fully_removed = E[E["weight_tau040"] == 0]
    filter_effect = {
        "cooccurrence_edge_instances_total": total_inst,
        "edge_instances_kept_tau040": kept_inst,
        "edge_instances_removed": total_inst - kept_inst,
        "share_instances_removed": 1 - kept_inst / total_inst,
        "unique_edges_coword": int(len(E)),
        "unique_edges_tau040": int(summ40["edges"]),
        "unique_edges_fully_removed": int(len(fully_removed)),
        "share_unique_edges_removed": len(fully_removed) / len(E),
        "removed_edges_with_weight_ge_20": int((fully_removed["weight_coword"] >= 20).sum()),
        "removed_edges_with_weight_ge_5": int((fully_removed["weight_coword"] >= 5).sum()),
    }
    # Highest-support co-occurrences that the semantic filter discards (examples for the paper).
    fully_removed.sort_values("weight_coword", ascending=False).head(60).to_csv(
        OUT / "removed_edges_top60.csv", index=False)
    # Highest-support co-occurrences that survive (contrast).
    E[E["weight_tau040"] > 0].sort_values("weight_tau040", ascending=False).head(60).to_csv(
        OUT / "kept_edges_top60.csv", index=False)
    # Similarity distribution of all co-occurring pairs (unique edges and edge instances).
    cats = pd.cut(E["sim_mean"], SIM_BINS, include_lowest=True)
    dist = E.groupby(cats, observed=False).agg(unique_edges=("u", "size"),
                                              edge_instances=("weight_coword", "sum")).reset_index()
    dist.columns = ["sim_mean_bin", "unique_edges", "edge_instances"]
    dist["cum_share_instances"] = dist["edge_instances"].cumsum() / total_inst
    dist.to_csv(OUT / "similarity_distribution.csv", index=False)
    T.mark("filter_effect")

    # ------------------------------------------------------------------ backbone comparison
    nodes40, nodes00 = set(H40.nodes()), set(H00.nodes())
    bb_compare = {
        "coword_backbone_nodes": len(nodes00), "tau040_backbone_nodes": len(nodes40),
        "shared_nodes": len(nodes00 & nodes40),
        "nodes_only_in_coword_backbone": len(nodes00 - nodes40),
        "nodes_only_in_tau040_backbone": len(nodes40 - nodes00),
        "coword_backbone_edges_removed_by_filter": int(sum(
            1 for u, v in H00.edges() if not H40.has_edge(u, v))),
        "coword_backbone_edges_total": H00.number_of_edges(),
        "partition_agreement_shared_nodes": C.partition_agreement(part00, part40),
    }
    # Hub dominance: how much of each backbone hangs on its most connected concept, and how many of
    # the co-occurrence instances discarded by the filter involve that hub.
    def hub_stats(H):
        if H.number_of_nodes() == 0:
            return {}
        hub = max(H.degree(), key=lambda t: t[1])[0]
        return {"hub": hub, "hub_degree": H.degree(hub), "hub_share_of_edges": H.degree(hub) / H.number_of_edges(),
                "hub_weighted_degree_share": H.degree(hub, weight="weight") / sum(d["weight"] for _, _, d in H.edges(data=True)),
                "edges_not_incident_to_hub": H.number_of_edges() - H.degree(hub)}
    hub00, hub40 = hub_stats(H00), hub_stats(H40)
    hub_name = hub00.get("hub")
    inc = E[(E.u == hub_name) | (E.v == hub_name)]
    bb_compare["hub_dominance"] = {
        "coword_backbone": hub00, "tau040_backbone": hub40,
        "share_removed_instances_incident_to_hub": float(inc.removed_instances.sum() / max(1, E.removed_instances.sum())),
        "share_hub_instances_removed": float(inc.removed_instances.sum() / max(1, inc.weight_coword.sum())),
        "share_removed_instances_not_incident_to_hub": float(1 - inc.removed_instances.sum() / max(1, E.removed_instances.sum())),
    }
    bb_compare["null_placeholder_node"] = {
        "in_global_graph": G00.has_node("null"),
        "doc_freq": G00.nodes["null"]["doc_freq"] if G00.has_node("null") else 0,
        "in_coword_backbone": H00.has_node("null"), "in_tau040_backbone": H40.has_node("null"),
        "coword_backbone_degree": H00.degree("null") if H00.has_node("null") else 0,
    }
    # Removed edges whose endpoints are both outside the hub (are non-hub relations also discarded?).
    fully_removed[(fully_removed.u != hub_name) & (fully_removed.v != hub_name)].sort_values(
        "weight_coword", ascending=False).head(60).to_csv(OUT / "removed_edges_top60_excluding_hub.csv", index=False)
    pd.DataFrame(sorted(nodes00 - nodes40)).to_csv(OUT / "nodes_only_in_coword_backbone.csv",
                                                  index=False, header=["keyword"])
    # Community sizes and top nodes by weighted degree for both backbones.
    for tag, H, part in (("tau040", H40, part40), ("coword", H00, part00)):
        wdeg = dict(H.degree(weight="weight"))
        top = (pd.DataFrame({"keyword": list(H.nodes()),
                             "community": [part.get(n) for n in H.nodes()],
                             "degree": [H.degree(n) for n in H.nodes()],
                             "weighted_degree": [wdeg[n] for n in H.nodes()]})
               .sort_values("weighted_degree", ascending=False))
        top.to_csv(OUT / f"backbone_nodes_{tag}.csv", index=False)
        sizes = pd.Series(part).value_counts().sort_values(ascending=False)
        sizes.rename_axis("community").reset_index(name="n_keywords").to_csv(
            OUT / f"community_sizes_{tag}.csv", index=False)
    T.mark("backbone_compare")

    # ------------------------------------------------------------------ w sweep (both graphs)
    sweep_rows, prev = [], {}
    for tag, G in (("tau040", G40), ("coword", G00)):
        prev_part = None
        for w in W_SWEEP:
            if w == 1 and args.skip_w1:
                continue
            H = build_backbone(G, w)
            s = C.graph_summary(H)
            part, q = C.louvain_partition(H, C.SEED)
            agree = C.partition_agreement(prev_part, part) if prev_part else {"nmi": np.nan, "ari": np.nan, "shared_nodes": 0}
            sweep_rows.append({"graph": tag, "w_min": w, **s, "modularity": q,
                               "communities": len(set(part.values())) if part else 0,
                               "nmi_vs_previous_w": agree["nmi"], "ari_vs_previous_w": agree["ari"],
                               "shared_nodes_vs_previous_w": agree["shared_nodes"]})
            prev_part = part
            print(f"  sweep {tag} w>={w}: nodes={s['nodes']} edges={s['edges']} lcc={s['lcc_fraction']:.3f} Q={q:.4f}", flush=True)
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT / "w_sweep.csv", index=False)
    T.mark("w_sweep")

    # ------------------------------------------------------------------ figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        for tag, mk in (("tau040", "o"), ("coword", "s")):
            d = sweep[sweep.graph == tag]
            ax[0].plot(d.w_min, d.nodes, marker=mk, label=tag); ax[0].set_yscale("log")
            ax[1].plot(d.w_min, d.lcc_fraction, marker=mk, label=tag)
            ax[2].plot(d.w_min, d.modularity, marker=mk, label=tag)
        for a, t in zip(ax, ["backbone nodes", "LCC fraction (backbone)", "modularity (Louvain, seed 42)"]):
            a.set_xlabel("w (minimum edge support)"); a.set_title(t); a.axvline(20, ls="--", c="gray", lw=0.8)
        ax[0].legend(); fig.tight_layout(); fig.savefig(OUT / "w_sweep.png", dpi=150); fig.savefig(OUT / "w_sweep.pdf")
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.bar(dist.index, dist.edge_instances, width=0.9)
        ax.set_xticks(dist.index[::2]); ax.set_xticklabels([str(b) for b in dist.sim_mean_bin[::2]], rotation=60, fontsize=7)
        ax.axvline(dist.index[np.searchsorted(SIM_BINS, 0.40) - 1] + 0.5, ls="--", c="red", lw=1)
        ax.set_ylabel("co-occurrence instances"); ax.set_title("Cosine similarity of co-occurring keyword pairs")
        fig.tight_layout(); fig.savefig(OUT / "similarity_distribution.png", dpi=150); fig.savefig(OUT / "similarity_distribution.pdf")
    except Exception as exc:  # figures are optional
        print("figure generation skipped:", exc)

    # ------------------------------------------------------------------ summary
    comparison = pd.DataFrame([
        {"graph": "coword (no semantic filter)", **summ00, **{f"backbone_{k}": v for k, v in bb00.items()}},
        {"graph": "tau=0.40 (published CRS)", **summ40, **{f"backbone_{k}": v for k, v in bb40.items()}},
    ])
    comparison.to_csv(OUT / "comparison.csv", index=False)
    summary = {"documents": n_docs, "vocabulary": len(vocab),
               "global_graph": {"coword": summ00, "tau040": summ40},
               "backbone_w20": {"coword": bb00, "tau040": bb40},
               "filter_effect": filter_effect, "backbone_comparison": bb_compare,
               "louvain_seeds": LOUVAIN_SEEDS, "w_sweep_values": W_SWEEP}
    C.write_json(summary, OUT / "summary.json")
    C.write_json({"gate": "tau=0.40 CRS reproduction (notebooks 2-5)", "pass": gate_pass,
                  "checks": checks}, OUT / "validation.json")
    meta.update({"completed_utc": C.now_utc(), "timings_seconds": T.marks, "status": "complete",
                 "gate_pass": gate_pass})
    C.write_json(meta, OUT / "metadata.json")
    print("done", T.marks, flush=True)
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
