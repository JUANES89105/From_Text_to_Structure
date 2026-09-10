"""E10 - Baseline: the same CRS built from the Scopus author/index keywords instead of the LLM keywords.

Addresses: Editor E-4 (what the method adds over conventional approaches), R2-1 (advantage of the LLM
extraction for downstream graph construction), R3-5/R3-6 (baseline for the contribution). The grounding
audit (E5) showed that 45% of the LLM keywords copy a record keyword and 86% appear verbatim in the
record, and every record carries author or index keywords (E6). The natural question is what the LLM
stage contributes at the graph level compared with a co-word network built directly from those keywords.

Three arms, same 52,946 documents, same constructor (``crs_reference.build_crs_for_tau``), same
tau = 0.40, same backbone rule (w >= 20) and Louvain seed:

  llm_k5        the published LLM keywords (gate: must reproduce the published CRS exactly)
  scopus_all    every author/index keyword of the record, normalised as in notebook 2
  scopus_first5 the first five keywords of the record (author keywords come first in the export),
                which holds the cardinality fixed and isolates the effect of the LLM's selection

For ``scopus_all`` the unconstrained co-word network (no tau) is also built, since that is the classical
bibliometric co-word map. Arms are compared on vocabulary size and lexical fragmentation, isolated nodes,
connectivity, backbone size, modularity, hub dominance, English residue, backbone-concept overlap and
partition agreement (on shared concepts and at the document level, as in E8).

Inputs: EID_KEYWORDS.xlsx (public), data/insumo_row_to_eid.csv (public) and the private record file for
the Scopus keywords. Outputs contain only aggregates, vocabularies and per-document community labels.
"""
from __future__ import annotations

import argparse
import itertools
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from crs_reference import build_backbone, build_crs_for_tau  # noqa: E402

OUT = C.RESULTS / "e10_scopus_keyword_baseline"
LOUVAIN_SEEDS = list(range(42, 52))
SPANISH_STOP = {"de", "la", "el", "en", "y", "para", "del", "las", "los", "con", "una", "un", "por"}
_nonalnum = re.compile(r"[^a-z0-9 ]+")
_ws = re.compile(r"\s+")


def notebook_clean(items):
    """Notebook 2 semantics: non-empty strings, strip().lower(), sorted(set())."""
    return sorted(set(str(k).strip().lower() for k in items if isinstance(k, str) and k.strip()))


def canon(phrase: str) -> str:
    """Aggressive canonical form used only to measure lexical fragmentation: strip punctuation and
    hyphens, collapse spaces, naive singular."""
    t = _ws.sub(" ", _nonalnum.sub(" ", phrase.lower())).strip()
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in t.split()]
    return " ".join(words)


def fragmentation(vocab):
    groups = defaultdict(list)
    for v in vocab:
        groups[canon(v)].append(v)
    multi = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    return {"vocabulary": len(vocab), "canonical_forms": len(groups),
            "share_vocabulary_that_collapses": 1 - len(groups) / max(1, len(vocab)),
            "variant_groups": len(multi),
            "largest_groups": sorted(multi.values(), key=len, reverse=True)[:15]}


def english_residue(flat):
    non_ascii = sum(any(ord(ch) > 127 for ch in k) for k in flat)
    spanish = sum(any(w in SPANISH_STOP for w in k.split()) for k in flat)
    return {"keywords": len(flat), "share_non_ascii": non_ascii / max(1, len(flat)),
            "share_spanish_function_words": spanish / max(1, len(flat))}


def hub_stats(H):
    if H.number_of_edges() == 0:
        return {}
    hub = max(H.degree(), key=lambda t: t[1])[0]
    return {"hub": hub, "hub_degree": int(H.degree(hub)), "hub_share_of_edges": H.degree(hub) / H.number_of_edges()}


def build_arm(name, docs, emb, tau, T):
    G = build_crs_for_tau(docs, emb, tau)
    T.mark(f"build_{name}")
    H = build_backbone(G, C.W_BACKBONE)
    parts, qs, ncs = {}, [], []
    for sd in LOUVAIN_SEEDS:
        p, q = C.louvain_partition(H, sd)
        parts[sd] = p; qs.append(q); ncs.append(len(set(p.values())) if p else 0)
    s = {"arm": name, "tau": tau, "documents": len(docs),
         "keywords_total": int(sum(len(d) for d in docs)),
         "keywords_per_document_mean": float(np.mean([len(d) for d in docs])),
         "keywords_per_document_median": float(np.median([len(d) for d in docs])),
         **{f"global_{k}": v for k, v in C.graph_summary(G).items()},
         **{f"backbone_{k}": v for k, v in C.graph_summary(H).items()},
         "backbone_modularity_seed42": qs[0],
         "backbone_modularity_mean": float(np.nanmean(qs)), "backbone_modularity_sd": float(np.nanstd(qs, ddof=1)),
         "backbone_communities_seed42": ncs[0], "backbone_communities_mean": float(np.mean(ncs)),
         **{f"backbone_{k}": v for k, v in hub_stats(H).items()}}
    # documents covered by the backbone (at least one keyword among backbone concepts)
    bb = set(H.nodes())
    s["share_documents_with_backbone_concept"] = float(np.mean([any(k in bb for k in d) for d in docs]))
    T.mark(f"backbone_{name}")
    return s, G, H, parts[LOUVAIN_SEEDS[0]]


def document_assignments(docs, part):
    """Majority community of a document's backbone keywords; None if no backbone keyword, 'tie' on ties."""
    out = []
    for d in docs:
        cnt = Counter(part[k] for k in d if k in part)
        if not cnt:
            out.append(None); continue
        top = cnt.most_common()
        out.append("tie" if len(top) > 1 and top[0][1] == top[1][1] else top[0][0])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--insumo", default=str(C.PRIVATE_DIR / "corpus_insumo_DEFINITIVO.csv"))
    ap.add_argument("--keywords", default=str(C.REPO_ROOT / "EID_KEYWORDS.xlsx"))
    ap.add_argument("--alignment", default=str(C.DATA / "insumo_row_to_eid.csv"))
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    C.set_seeds()
    T = C.Timer()
    meta = C.env_metadata(experiment="E10 Scopus keyword baseline", llm_calls=0, paid_api_calls=0,
                          inputs={"insumo": {"path": args.insumo, "sha256": C.sha256(args.insumo), "redistributed": False},
                                  "keywords": {"path": args.keywords, "sha256": C.sha256(args.keywords)},
                                  "alignment": {"path": args.alignment, "sha256": C.sha256(args.alignment)}})

    # ------------------------------------------------------------------ documents and keyword sets
    ins = pd.read_csv(args.insumo)["insumo"].astype(str).tolist()
    align = C.load_alignment(Path(args.alignment))
    pub = C.load_published_keywords(Path(args.keywords), notebook_semantics=True)
    eids, llm_docs, sc_all, sc_first5 = [], [], [], []
    n_no_scopus = 0
    for eid, row in zip(align.eid, align.insumo_row):
        kws = pub.get(eid, [])
        if not kws:                      # notebook 2 drops empty lists: 52,946 analysed documents
            continue
        rec = C.parse_record(ins[row])
        raw = [k for k in rec["original_keywords_raw"].split(";") if k.strip()]
        ordered = []                     # record order, de-duplicated after normalisation
        for k in raw:
            t = str(k).strip().lower()
            if t and t not in ordered:
                ordered.append(t)
        if not ordered:
            n_no_scopus += 1
        eids.append(eid); llm_docs.append(kws)
        sc_all.append(sorted(ordered)); sc_first5.append(sorted(ordered[:5]))
    n = len(eids)
    print(f"documents={n} (records without Scopus keywords: {n_no_scopus})", flush=True)
    T.mark("load")

    vocab_llm = sorted(set(itertools.chain.from_iterable(llm_docs)))
    vocab_sc = sorted(set(itertools.chain.from_iterable(sc_all)))
    vocab_sc5 = sorted(set(itertools.chain.from_iterable(sc_first5)))
    print(f"vocabulary: llm={len(vocab_llm)} scopus_all={len(vocab_sc)} scopus_first5={len(vocab_sc5)}", flush=True)
    frag = {"llm_k5": fragmentation(vocab_llm), "scopus_all": fragmentation(vocab_sc), "scopus_first5": fragmentation(vocab_sc5)}
    residue = {"llm_k5": english_residue(list(itertools.chain.from_iterable(llm_docs))),
               "scopus_all": english_residue(list(itertools.chain.from_iterable(sc_all)))}
    T.mark("vocabulary")

    model = C.load_embedder(threads=args.threads)
    emb = C.embed_unique(model, set(vocab_llm) | set(vocab_sc), batch_size=256)
    T.mark("embeddings")

    # ------------------------------------------------------------------ arms
    arms, graphs, backbones, parts = [], {}, {}, {}
    for name, docs, tau in (("llm_k5", llm_docs, C.TAU_EDGE), ("scopus_all", sc_all, C.TAU_EDGE),
                            ("scopus_first5", sc_first5, C.TAU_EDGE), ("scopus_all_coword", sc_all, -1.0)):
        s, G, H, part = build_arm(name, docs, emb, tau, T)
        arms.append(s); graphs[name] = G; backbones[name] = H; parts[name] = part
        print(f"  {name}: nodes={s['global_nodes']} edges={s['global_edges']} isolated={s['global_isolated_nodes']} "
              f"bb={s['backbone_nodes']}/{s['backbone_edges']} Q={s['backbone_modularity_seed42']:.4f} "
              f"comm={s['backbone_communities_seed42']} hub={s.get('backbone_hub')} ({s.get('backbone_hub_share_of_edges', 0):.3f})", flush=True)
        wdeg = dict(H.degree(weight="weight"))
        pd.DataFrame({"keyword": list(H.nodes()), "community": [part.get(k) for k in H.nodes()],
                      "degree": [H.degree(k) for k in H.nodes()], "weighted_degree": [wdeg[k] for k in H.nodes()]}
                     ).sort_values("weighted_degree", ascending=False).to_csv(OUT / f"backbone_nodes_{name}.csv", index=False)
        pd.Series(part).value_counts().rename_axis("community").reset_index(name="n_keywords").to_csv(
            OUT / f"community_sizes_{name}.csv", index=False)
        pd.DataFrame([(u, v, d["weight"], d.get("sim_mean")) for u, v, d in H.edges(data=True)],
                     columns=["u", "v", "weight", "sim_mean"]).sort_values("weight", ascending=False).head(60).to_csv(
            OUT / f"backbone_top_edges_{name}.csv", index=False)
    table = pd.DataFrame(arms)
    table.to_csv(OUT / "arms_comparison.csv", index=False)

    # ------------------------------------------------------------------ gate
    ref, g = C.CRS_REFERENCE, table[table.arm == "llm_k5"].iloc[0]
    checks = {k: {"observed": int(g[o]), "reference": ref[k], "pass": int(g[o]) == ref[k]} for k, o in
              [("analyzed_documents", "documents"), ("nodes", "global_nodes"), ("edges", "global_edges"),
               ("components", "global_components"), ("lcc_nodes", "global_lcc_nodes"),
               ("backbone_nodes", "backbone_nodes"), ("backbone_edges", "backbone_edges"),
               ("backbone_communities", "backbone_communities_seed42")]}
    checks["backbone_modularity"] = {"observed": float(g.backbone_modularity_seed42), "reference": ref["backbone_modularity"],
                                     "pass": abs(float(g.backbone_modularity_seed42) - ref["backbone_modularity"]) < 1e-6}
    gate_pass = all(v["pass"] for v in checks.values())
    print("REPRODUCTION GATE (llm_k5):", "PASS" if gate_pass else "FAIL", flush=True)

    # ------------------------------------------------------------------ comparisons between arms
    comp = {}
    for a in ("scopus_all", "scopus_first5", "scopus_all_coword"):
        na, nl = set(backbones[a].nodes()), set(backbones["llm_k5"].nodes())
        # document-level agreement (documents with an unambiguous community in both arms)
        da = document_assignments(sc_all if a.startswith("scopus_all") else sc_first5, parts[a])
        dl = document_assignments(llm_docs, parts["llm_k5"])
        both = [(x, y) for x, y in zip(da, dl) if x not in (None, "tie") and y not in (None, "tie")]
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        comp[f"llm_k5_vs_{a}"] = {
            "backbone_nodes_llm": len(nl), f"backbone_nodes_{a}": len(na), "shared_backbone_concepts": len(na & nl),
            "jaccard_backbone_concepts": len(na & nl) / max(1, len(na | nl)),
            "concept_partition_agreement_on_shared": C.partition_agreement(parts["llm_k5"], parts[a]),
            "documents_unambiguous_llm": int(sum(1 for y in dl if y not in (None, "tie"))),
            f"documents_unambiguous_{a}": int(sum(1 for x in da if x not in (None, "tie"))),
            "documents_compared": len(both),
            "document_nmi": float(normalized_mutual_info_score([y for _, y in both], [x for x, _ in both])) if len(both) > 1 else None,
            "document_ari": float(adjusted_rand_score([y for _, y in both], [x for x, _ in both])) if len(both) > 1 else None,
            "llm_backbone_concepts_absent_from_scopus_vocabulary": int(sum(1 for k in nl if k not in set(vocab_sc))),
        }
        pd.DataFrame(sorted(nl - na)).to_csv(OUT / f"llm_backbone_concepts_not_in_{a}_backbone.csv", index=False, header=["keyword"])
    # per-document labels for both main arms (no text)
    pd.DataFrame({"eid": eids, "community_llm_k5": document_assignments(llm_docs, parts["llm_k5"]),
                  "community_scopus_all": document_assignments(sc_all, parts["scopus_all"]),
                  "community_scopus_first5": document_assignments(sc_first5, parts["scopus_first5"])}).to_csv(
        OUT / "document_communities.csv", index=False)
    # vocabulary overlap
    overlap = {"llm_vocabulary": len(vocab_llm), "scopus_vocabulary": len(vocab_sc),
               "shared_terms": len(set(vocab_llm) & set(vocab_sc)),
               "share_llm_terms_present_in_scopus_vocabulary": len(set(vocab_llm) & set(vocab_sc)) / len(vocab_llm),
               "share_llm_keyword_instances_that_are_a_scopus_keyword_of_same_record": float(np.mean(
                   [k in set(s) for d, s in zip(llm_docs, sc_all) for k in d]))}
    T.mark("comparisons")

    summary = {"documents": n, "records_without_scopus_keywords": n_no_scopus, "arms": arms,
               "vocabulary_fragmentation": frag, "english_residue": residue, "vocabulary_overlap": overlap,
               "comparisons": comp, "louvain_seeds": LOUVAIN_SEEDS}
    C.write_json(summary, OUT / "summary.json")
    C.write_json({"gate": "llm_k5 arm reproduces the published CRS (notebooks 2-5)", "pass": gate_pass, "checks": checks},
                 OUT / "validation.json")
    pd.DataFrame([{"arm": k, **{kk: vv for kk, vv in v.items() if kk != "largest_groups"}} for k, v in frag.items()]).to_csv(
        OUT / "vocabulary_fragmentation.csv", index=False)
    meta.update({"completed_utc": C.now_utc(), "timings_seconds": T.marks, "status": "complete", "gate_pass": gate_pass})
    C.write_json(meta, OUT / "metadata.json")
    print(table[["arm", "keywords_per_document_mean", "global_nodes", "global_edges", "global_isolated_nodes", "global_lcc_fraction",
                 "backbone_nodes", "backbone_edges", "backbone_modularity_seed42", "backbone_communities_seed42",
                 "share_documents_with_backbone_concept"]].round(4).to_string(index=False))
    for k, v in comp.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "concept_partition_agreement_on_shared"},
              "shared-concept NMI/ARI:", {kk: round(vv, 4) for kk, vv in v["concept_partition_agreement_on_shared"].items()})
    print("done", T.marks, flush=True)
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
