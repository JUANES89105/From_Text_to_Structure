"""E2 - Robustness of the extracted representation to the language model.

Addresses: R2-7 (a single LLM configuration), R3-1 (dependence of the representation on the model).

Two complementary measurements, neither of which requires a new LLM call:

1. Inspec benchmark (2,000 documents): the same prompt run with LLaMA 3.1 8B (published) and with
   LLaMA 3.3 70B. We report (a) agreement between the two models' keyword sets per document and
   (b) both models against the expert gold keyphrases with the six metrics of notebook 6.
2. Mathematics-education validation sample (182 documents): agreement between the two models.

The 8B Inspec means are reproduced first as a gate (must match notebook 6 within 5e-5).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import inspec_evaluation as IE  # noqa: E402

OUT = C.RESULTS / "e2_model_agreement"


def make_embed_fn(model):
    cache = {}

    def embed(phrases):
        missing = [p for p in phrases if p not in cache]
        if missing:
            vecs = model.encode(missing, batch_size=IE.BATCH_SIZE, show_progress_bar=False,
                                normalize_embeddings=False, convert_to_numpy=True)
            cache.update(zip(missing, vecs))
        return np.vstack([cache[p] for p in phrases])
    return embed


def score_pair(ref, pred, embed):
    """Six notebook-6 metrics of ``pred`` against ``ref`` (both cleaned lists)."""
    m = IE.soft_matching_metrics(ref, pred, embed, IE.TAU)
    m["jaccard_lex"] = IE.jaccard(ref, pred)
    m["global_sem_sim"] = IE.global_concat_similarity(ref, pred, embed)
    m["identical_set"] = float(set(ref) == set(pred))
    return m


def agreement_table(list_a, list_b, embed, ids, label_a, label_b):
    rows = []
    for i, a, b in zip(ids, list_a, list_b):
        ca, cb = IE.clean_list(a), IE.clean_list(b)
        m = score_pair(ca, cb, embed)
        rows.append({"doc": i, f"n_{label_a}": len(ca), f"n_{label_b}": len(cb), **m})
    df = pd.DataFrame(rows)
    summ = {k: {"mean": float(df[k].mean()), "sd": float(df[k].std(ddof=1))} for k in IE.METRICS}
    summ["identical_set_share"] = float(df.identical_set.mean())
    summ["documents"] = int(len(df))
    return df, summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inspec", default=str(C.REPO_ROOT / "dataset_inspec.csv"))
    ap.add_argument("--inspec-8b", default=str(C.REPO_ROOT / "inspec_llama-3.1-8b-EN.csv"))
    ap.add_argument("--inspec-70b", default=str(C.PRIVATE_DIR / "inspec_llama-3.3-70b-EN.csv"))
    ap.add_argument("--sample-8b", default=str(C.DATA / "keywords_llm_llama-3.1-8b-EN.csv"))
    ap.add_argument("--sample-70b", default=str(C.PRIVATE_DIR / "keywords_llm_llama-3.3-70b-EN.csv"))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    C.set_seeds()
    T = C.Timer()
    inputs = {}
    for k in ("inspec", "inspec_8b", "inspec_70b", "sample_8b", "sample_70b"):
        p = Path(getattr(args, k))
        inputs[k] = {"path": str(p), "present": p.exists(), "sha256": C.sha256(p) if p.exists() else None,
                     "redistributed": str(C.REPO_ROOT) in str(p.resolve())}
    meta = C.env_metadata(experiment="E2 model agreement", inputs=inputs, llm_calls=0, paid_api_calls=0)
    model = C.load_embedder(threads=args.threads)
    embed = make_embed_fn(model)
    summary, validation = {}, {}

    # ------------------------------------------------------------------ 1. Inspec
    insp = pd.read_csv(args.inspec)
    gold = [IE.clean_list(IE.safe_parse_list(x)) for x in insp["keywords_gt"]]
    p8 = pd.read_csv(args.inspec_8b).sort_values("row_abs")
    pred8 = [IE.clean_list(IE.safe_parse_list(x)) for x in p8["keywords_llm"]]
    rows = [{"row_abs": r, **score_pair(g, p, embed)} for r, g, p in zip(p8.row_abs, gold, pred8)]
    g8 = pd.DataFrame(rows)
    means8 = {k: round(float(g8[k].mean()), 4) for k in IE.METRICS}
    gate = {k: {"observed": means8[k], "reference": IE.EXPECTED[k], "pass": abs(means8[k] - IE.EXPECTED[k]) <= 5e-5} for k in IE.METRICS}
    gate_pass = all(v["pass"] for v in gate.values())
    validation["inspec_8b_reproduction"] = {"pass": gate_pass, "checks": gate}
    print("INSPEC 8B GATE:", "PASS" if gate_pass else "FAIL", means8, flush=True)
    gold_table = [{"model": "llama-3.1-8b-instruct", **{k: float(g8[k].mean()) for k in IE.METRICS},
                   **{f"{k}_sd": float(g8[k].std(ddof=1)) for k in IE.METRICS}}]
    T.mark("inspec_8b")

    if inputs["inspec_70b"]["present"]:
        p70 = pd.read_csv(args.inspec_70b).sort_values("row_abs")
        assert list(p70.row_abs) == list(p8.row_abs)
        pred70 = [IE.clean_list(IE.safe_parse_list(x)) for x in p70["keywords_llm"]]
        g70 = pd.DataFrame([{"row_abs": r, **score_pair(g, p, embed)} for r, g, p in zip(p70.row_abs, gold, pred70)])
        gold_table.append({"model": "llama-3.3-70b-instruct", **{k: float(g70[k].mean()) for k in IE.METRICS},
                           **{f"{k}_sd": float(g70[k].std(ddof=1)) for k in IE.METRICS}})
        agree_df, agree = agreement_table(pred8, pred70, embed, p8.row_abs, "8b", "70b")
        agree_df.to_csv(OUT / "inspec_agreement_8b_vs_70b_per_document.csv", index=False)
        # Paired difference of gold metrics between models (per document), for a CI without new runs.
        diff = {k: {"mean_70b_minus_8b": float((g70[k] - g8[k]).mean()),
                    "ci95_low": float((g70[k] - g8[k]).mean() - 1.96 * (g70[k] - g8[k]).std(ddof=1) / np.sqrt(len(g8))),
                    "ci95_high": float((g70[k] - g8[k]).mean() + 1.96 * (g70[k] - g8[k]).std(ddof=1) / np.sqrt(len(g8)))}
                for k in IE.METRICS}
        summary["inspec"] = {"documents": int(len(g8)), "agreement_8b_vs_70b": agree,
                             "gold_metrics_paired_difference": diff,
                             "empty_predictions": {"8b": int(sum(len(p) == 0 for p in pred8)), "70b": int(sum(len(p) == 0 for p in pred70))}}
        print("Inspec 8B vs 70B agreement:", {k: round(v["mean"], 4) for k, v in agree.items() if isinstance(v, dict)}, flush=True)
    else:
        summary["inspec"] = {"documents": int(len(g8)), "agreement_8b_vs_70b": "70B prediction file not available"}
    pd.DataFrame(gold_table).to_csv(OUT / "inspec_gold_metrics_by_model.csv", index=False)
    T.mark("inspec_70b")

    # ------------------------------------------------------------------ 2. Mathematics-education sample
    s8 = pd.read_csv(args.sample_8b).sort_values("row_abs")
    if inputs["sample_70b"]["present"]:
        s70 = pd.read_csv(args.sample_70b).sort_values("row_abs")
        assert list(s70.row_abs) == list(s8.row_abs)
        a8 = [IE.safe_parse_list(x) for x in s8.keywords_llm]
        a70 = [IE.safe_parse_list(x) for x in s70.keywords_llm]
        sdf, sagree = agreement_table(a8, a70, embed, s8.row_abs, "8b", "70b")
        sdf.to_csv(OUT / "matheduc_sample_agreement_8b_vs_70b_per_document.csv", index=False)
        summary["mathematics_education_sample"] = {"documents": int(len(sdf)), "agreement_8b_vs_70b": sagree}
        print("Math-ed sample 8B vs 70B agreement:", {k: round(v["mean"], 4) for k, v in sagree.items() if isinstance(v, dict)}, flush=True)
    else:
        summary["mathematics_education_sample"] = {"documents": int(len(s8)), "agreement_8b_vs_70b": "70B prediction file not available"}
    T.mark("sample")

    C.write_json(summary, OUT / "summary.json")
    C.write_json(validation, OUT / "validation.json")
    meta.update({"completed_utc": C.now_utc(), "timings_seconds": T.marks, "status": "complete", "gate_pass": gate_pass})
    C.write_json(meta, OUT / "metadata.json")
    print("done", T.marks, flush=True)
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
