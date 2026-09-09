"""E4 (step 2) - Compare the CRS built from k = 5 keywords with the CRS built from k = 10 (or other k)
on the same 5,000-document sample.

The k = 5 arm uses the published keywords (EID_KEYWORDS.xlsx) restricted to the sample; other arms
use ``predictions_k{k}.csv`` produced by ``run_extraction.py``. If no prediction file is present
the script still reports the k = 5 arm and marks the comparison as pending.

Reported per arm: vocabulary, keywords per document, global CRS (tau = 0.40) size and connectivity,
backbone at w >= 20 (paper) and at w >= 5 (adapted to a 5,000-document sample, see E3), Louvain
modularity and communities. Between arms: overlap of backbone concepts, NMI/ARI of the community
partitions on shared concepts, and how many of the extra keywords are new concepts versus exact or
soft (cos >= 0.70) restatements of the document's k = 5 keywords.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from crs_reference import build_backbone, build_crs_for_tau, parse_keywords  # noqa: E402

OUT = C.RESULTS / "e4_k_sensitivity"
W_LEVELS = [20, 5]


def notebook_clean(lst):
    return sorted(set(str(k).strip().lower() for k in lst if isinstance(k, str) and k.strip()))


def arm_metrics(docs, emb, label):
    G = build_crs_for_tau(docs, emb, C.TAU_EDGE)
    res = {"arm": label, "documents": len(docs), "keywords_total": int(sum(len(d) for d in docs)),
           "keywords_per_document_mean": float(np.mean([len(d) for d in docs])),
           "vocabulary": len(set(itertools.chain.from_iterable(docs))),
           "null_tokens": int(sum(d.count("null") for d in docs)), **{f"global_{k}": v for k, v in C.graph_summary(G).items()}}
    parts = {}
    for w in W_LEVELS:
        H = build_backbone(G, w)
        s = C.graph_summary(H)
        part, q = C.louvain_partition(H, C.SEED)
        parts[w] = part
        res.update({f"bb{w}_{k}": v for k, v in s.items()})
        res[f"bb{w}_modularity"] = q
        res[f"bb{w}_communities"] = len(set(part.values())) if part else 0
    return res, G, parts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", default=str(C.REPO_ROOT / "results/e3_scalability/samples/n5000_r1_s42.csv"))
    ap.add_argument("--keywords", default=str(C.REPO_ROOT / "EID_KEYWORDS.xlsx"))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    C.set_seeds()
    T = C.Timer()
    meta = C.env_metadata(experiment="E4 k sensitivity (analysis)", llm_calls=0, paid_api_calls=0,
                          inputs={"sample": {"path": args.sample, "sha256": C.sha256(args.sample)},
                                  "keywords": {"path": args.keywords, "sha256": C.sha256(args.keywords)}})
    sample = pd.read_csv(args.sample)
    kw = C.load_keywords_xlsx(Path(args.keywords), notebook_semantics=True)
    by_eid = dict(zip(kw.eid, kw.keywords))
    eids = [str(e) for e in sample["EID_o_identificador"]]
    arms = {5: [by_eid[e] for e in eids]}
    pred_files = sorted(OUT.glob("predictions_k*.csv"))
    for pf in pred_files:
        k = int(pf.stem.split("_k")[1])
        p = pd.read_csv(pf)
        p["EID_o_identificador"] = p["EID_o_identificador"].astype(str)
        p = p.drop_duplicates("EID_o_identificador").set_index("EID_o_identificador")
        arms[k] = [notebook_clean(parse_keywords(p.loc[e, "keywords_llm"])) if e in p.index and isinstance(p.loc[e, "keywords_llm"], str) else [] for e in eids]
        meta.setdefault("prediction_files", {})[str(pf)] = {"sha256": C.sha256(pf), "rows": int(len(p)),
                                                            "status_counts": pd.read_csv(pf)["status"].value_counts().to_dict()}
    vocab = sorted(set(itertools.chain.from_iterable(itertools.chain.from_iterable(arms.values()))))
    model = C.load_embedder(threads=args.threads)
    emb = C.embed_unique(model, vocab, batch_size=256)
    T.mark("embeddings")

    rows, graphs, parts = [], {}, {}
    for k, docs in sorted(arms.items()):
        docs_nonempty = [d for d in docs if d]
        r, G, pk = arm_metrics(docs_nonempty, emb, f"k={k}")
        rows.append(r); graphs[k] = G; parts[k] = pk
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "arms_comparison.csv", index=False)
    print(table.T.to_string())

    # Gate: the k=5 arm must coincide with the E3 record of the same sample (same code path, same seed).
    e3 = pd.read_csv(C.REPO_ROOT / "results/e3_scalability/runs.csv")
    ref = e3[e3.sample_file.str.endswith(Path(args.sample).name)]
    k5 = table[table.arm == "k=5"].iloc[0]
    checks = {}
    if len(ref) == 1:
        ref = ref.iloc[0]
        for ours, theirs in [("global_nodes", "nodes_full_graph"), ("global_edges", "edges_full_graph"),
                             ("global_components", "n_components_full_graph"), ("global_lcc_nodes", "lcc_nodes"),
                             ("bb20_nodes", "backbone_nodes"), ("bb20_edges", "backbone_edges"),
                             ("bb20_communities", "n_communities")]:
            checks[ours] = {"observed": int(k5[ours]), "e3_reference": int(ref[theirs]), "pass": int(k5[ours]) == int(ref[theirs])}
        checks["bb20_modularity"] = {"observed": float(k5.bb20_modularity), "e3_reference": float(ref.modularity),
                                     "pass": abs(float(k5.bb20_modularity) - float(ref.modularity)) < 1e-6}
    gate_pass = bool(checks) and all(v["pass"] for v in checks.values())
    C.write_json({"gate": "k=5 arm reproduces the E3 run on the same sample", "pass": gate_pass, "checks": checks},
                 OUT / "validation.json")
    print("E3 CROSS-CHECK GATE:", "PASS" if gate_pass else "FAIL")

    comparison = {}
    for k in [k for k in arms if k != 5]:
        comp = {}
        for w in W_LEVELS:
            p5, pk = parts[5][w], parts[k][w]
            n5, nk = set(p5), set(pk)
            comp[f"bb{w}"] = {"nodes_k5": len(n5), f"nodes_k{k}": len(nk), "shared": len(n5 & nk),
                              "jaccard_nodes": len(n5 & nk) / max(1, len(n5 | nk)), **C.partition_agreement(p5, pk)}
        # Novelty of the extra keywords with respect to the document's k=5 set.
        exact, soft, new, total = 0, 0, 0, 0
        for d5, dk in zip(arms[5], arms[k]):
            s5 = set(d5)
            for t in dk:
                if not t or t == "null":
                    continue
                total += 1
                if t in s5:
                    exact += 1
                elif d5 and max(float(emb[t] @ emb[u]) for u in d5) >= C.TAU_SOFT:
                    soft += 1
                else:
                    new += 1
        comp["extra_keyword_novelty"] = {"keywords": total, "exact_restatement_of_k5": exact / max(1, total),
                                         "soft_restatement_of_k5": soft / max(1, total), "new_concept": new / max(1, total)}
        comparison[f"k5_vs_k{k}"] = comp
    status = "complete" if len(arms) > 1 else "k=5 arm only; run run_extraction.py (needs OPENROUTER_API_KEY) to add other k"
    C.write_json({"status": status, "arms": rows, "comparison": comparison, "w_levels": W_LEVELS}, OUT / "summary.json")
    meta.update({"completed_utc": C.now_utc(), "timings_seconds": T.marks, "status": status})
    C.write_json(meta, OUT / "metadata_analysis.json")
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
