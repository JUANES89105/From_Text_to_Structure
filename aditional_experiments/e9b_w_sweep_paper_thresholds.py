"""E9b - Backbone sweep restricted to the support thresholds used in the manuscript.

The manuscript's Table 7 reports the backbone for w in {1, 3, 5, 8, 10, 20, 30} (notebook 3). E9 swept
twelve thresholds; this script recomputes, for the manuscript's seven values only, the modularity,
the number of communities and the agreement (NMI, ARI) between the Louvain partitions of consecutive
thresholds on shared nodes, so that the numbers quoted in the paper refer to the same threshold set
as its table. Same graph (tau = 0.40), same constructor, same Louvain seed as E9; the tau = 0.40 graph
must reproduce the published CRS before anything is written.
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
W_PAPER = [1, 3, 5, 8, 10, 20, 30]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keywords", default=str(C.REPO_ROOT / "EID_KEYWORDS.xlsx"))
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args(argv)
    C.set_seeds()
    T = C.Timer()
    kw = C.load_keywords_xlsx(Path(args.keywords), notebook_semantics=True)
    docs = [k for k in kw["keywords"] if len(k) > 0]
    vocab = sorted(set(itertools.chain.from_iterable(docs)))
    model = C.load_embedder(threads=args.threads)
    emb = C.embed_unique(model, vocab, batch_size=256)
    G = build_crs_for_tau(docs, emb, C.TAU_EDGE)
    s = C.graph_summary(G)
    ref = C.CRS_REFERENCE
    gate = s["nodes"] == ref["nodes"] and s["edges"] == ref["edges"] and len(docs) == ref["analyzed_documents"]
    print("REPRODUCTION GATE (tau=0.40):", "PASS" if gate else "FAIL", flush=True)
    rows, prev = [], None
    for w in W_PAPER:
        H = build_backbone(G, w)
        hs = C.graph_summary(H)
        part, q = C.louvain_partition(H, C.SEED)
        agree = C.partition_agreement(prev, part) if prev else {"nmi": np.nan, "ari": np.nan, "shared_nodes": 0}
        rows.append({"w_min": w, "nodes": hs["nodes"], "edges": hs["edges"], "components": hs["components"],
                     "lcc_nodes": hs["lcc_nodes"], "lcc_fraction": hs["lcc_fraction"], "modularity": q,
                     "communities": len(set(part.values())) if part else 0,
                     "nmi_vs_previous_w": agree["nmi"], "ari_vs_previous_w": agree["ari"],
                     "shared_nodes_vs_previous_w": agree["shared_nodes"]})
        print(f"  w>={w}: nodes={hs['nodes']} edges={hs['edges']} comps={hs['components']} lcc={hs['lcc_fraction']:.4f} "
              f"Q={q:.4f} comm={rows[-1]['communities']} NMI_prev={agree['nmi']:.3f}", flush=True)
        prev = part
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "w_sweep_paper_thresholds.csv", index=False)
    gate_ok = bool(gate) and all(abs(r["modularity"] - m) < 1e-6 for r, m in [(rows[5], ref["backbone_modularity"])])
    C.write_json({"gate": "tau=0.40 graph and w=20 backbone reproduce the published CRS", "pass": gate_ok,
                  "thresholds": W_PAPER, "nodes": s["nodes"], "edges": s["edges"],
                  "backbone_w20": {k: rows[5][k] for k in ("nodes", "edges", "modularity", "communities")}},
                 OUT / "w_sweep_paper_thresholds_validation.json")
    meta = C.env_metadata(experiment="E9b w sweep on the manuscript thresholds", llm_calls=0, paid_api_calls=0,
                          inputs={"keywords": {"path": args.keywords, "sha256": C.sha256(args.keywords)}})
    meta.update({"completed_utc": C.now_utc(), "timings_seconds": T.marks | {"total": T.mark("total")}, "status": "complete"})
    C.write_json(meta, OUT / "w_sweep_paper_thresholds_metadata.json")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
