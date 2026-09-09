"""E6 - Corpus profile: coverage, temporal distribution, sources, and language.

Addresses: R3-4 (representativeness and bias of the corpus, in particular language bias) and
Editor E-2 (corpus selection).

All figures are derived from the six Scopus fields of each record (authors, title, year, source,
abstract, original keywords). Language is identified with ``langdetect`` (deterministic seed) on
the title and on the first 1,000 characters of the abstract. The generated keywords are audited
for non-English residue with two transparent heuristics (non-ASCII characters and Spanish
function words), because the prompt forces English output.

Input: ``corpus_insumo_DEFINITIVO.csv`` (private, Elsevier licence). Only aggregates are written.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from inspec_evaluation import normalize_phrase  # noqa: E402

OUT = C.RESULTS / "e6_corpus_profile"
SPANISH_STOP = {"de", "la", "el", "en", "y", "para", "del", "las", "los", "con", "una", "un", "por"}


def detect(text: str, n: int = 1000) -> str:
    from langdetect import detect
    t = str(text).strip()[:n]
    if len(t) < 20:
        return "too_short"
    try:
        return detect(t)
    except Exception:
        return "undetermined"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--insumo", default=str(C.PRIVATE_DIR / "corpus_insumo_DEFINITIVO.csv"))
    ap.add_argument("--keywords", default=str(C.REPO_ROOT / "EID_KEYWORDS.xlsx"))
    ap.add_argument("--alignment", default=str(C.DATA / "insumo_row_to_eid.csv"))
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    from langdetect import DetectorFactory
    DetectorFactory.seed = C.SEED
    T = C.Timer()
    meta = C.env_metadata(experiment="E6 corpus profile",
                          inputs={"insumo": {"path": args.insumo, "sha256": C.sha256(args.insumo), "redistributed": False},
                                  "keywords": {"path": args.keywords, "sha256": C.sha256(args.keywords)}},
                          llm_calls=0, paid_api_calls=0)

    ins = pd.read_csv(args.insumo)["insumo"].astype(str).tolist()
    recs = pd.DataFrame([C.parse_record(s) for s in ins])
    recs["year_int"] = pd.to_numeric(recs.year, errors="coerce")
    recs["n_original_keywords"] = [len(C.split_original_keywords(r)) for r in recs.original_keywords_raw]
    recs["abstract_chars"] = recs.abstract.str.len()
    recs["n_authors"] = recs.authors.apply(lambda a: max(1, len([x for x in a.split(";") if x.strip()]) // 3) if a else 0)
    T.mark("parse")

    recs["lang_title"] = [detect(t, 300) for t in recs.title]
    recs["lang_abstract"] = [detect(a, 1000) for a in recs.abstract]
    T.mark("langdetect")

    # ------------------------------------------------------------------ tables
    year = recs.year_int.value_counts().sort_index().rename_axis("year").reset_index(name="documents")
    year["share"] = year.documents / len(recs)
    year.to_csv(OUT / "year_distribution.csv", index=False)
    src = recs.source.replace("", "(missing)").value_counts().head(40).rename_axis("source").reset_index(name="documents")
    src["share"] = src.documents / len(recs)
    src.to_csv(OUT / "sources_top40.csv", index=False)
    lt = recs.lang_title.value_counts().rename_axis("language").reset_index(name="documents"); lt["share"] = lt.documents / len(recs)
    la = recs.lang_abstract.value_counts().rename_axis("language").reset_index(name="documents"); la["share"] = la.documents / len(recs)
    lt.to_csv(OUT / "language_title.csv", index=False); la.to_csv(OUT / "language_abstract.csv", index=False)
    top_langs = [l for l in la.language.head(6)]
    ct = pd.crosstab(recs.lang_title.where(recs.lang_title.isin(top_langs), "other"),
                     recs.lang_abstract.where(recs.lang_abstract.isin(top_langs), "other"))
    ct.to_csv(OUT / "language_crosstab_title_vs_abstract.csv")
    # Language by period (is the non-English share changing over time?)
    recs["period"] = pd.cut(recs.year_int, [1999, 2005, 2010, 2015, 2020, 2027],
                            labels=["2000-2005", "2006-2010", "2011-2015", "2016-2020", "2021-2026"])
    lp = pd.crosstab(recs.period, recs.lang_abstract.where(recs.lang_abstract.isin(top_langs), "other"), normalize="index")
    lp.to_csv(OUT / "language_abstract_by_period.csv")

    # ------------------------------------------------------------------ generated keyword audit
    kw = C.load_keywords_xlsx(Path(args.keywords), notebook_semantics=False)      # published outputs
    align = C.load_alignment(Path(args.alignment))
    flat = [str(k) for kws in kw.keywords for k in kws]
    norm = [normalize_phrase(k) for k in flat]
    non_ascii = [k for k in flat if any(ord(c) > 127 for c in k)]
    spanish = [n for n in norm if any(w in SPANISH_STOP for w in n.split())]
    kw_audit = {
        "keywords_total": len(flat),
        "keywords_with_non_ascii_characters": len(non_ascii),
        "share_non_ascii": len(non_ascii) / len(flat),
        "examples_non_ascii": Counter(non_ascii).most_common(15),
        "keywords_with_spanish_function_words": len(spanish),
        "share_spanish_function_words": len(spanish) / len(flat),
        "examples_spanish_function_words": Counter(spanish).most_common(20),
    }
    # Do non-English abstracts produce more non-English residue in the keywords?
    row_by_eid = dict(zip(align.eid, align.insumo_row))
    kw["lang_abstract"] = [recs.lang_abstract.iloc[row_by_eid[e]] if e in row_by_eid else "unlinked" for e in kw.eid]
    kw["has_spanish_stop"] = [any(any(w in SPANISH_STOP for w in normalize_phrase(k).split()) for k in kws) for kws in kw.keywords]
    by_lang = kw.groupby(kw.lang_abstract.where(kw.lang_abstract.isin(top_langs), "other")).has_spanish_stop.agg(["mean", "size"])
    by_lang.rename(columns={"mean": "share_documents_with_spanish_function_word_in_keywords", "size": "documents"}).to_csv(
        OUT / "keyword_language_residue_by_abstract_language.csv")

    summary = {
        "documents": int(len(recs)),
        "records_with_6_fields": int((recs.n_fields == 6).sum()),
        "records_with_more_than_6_fields": int((recs.n_fields > 6).sum()),
        "year": {"min": int(recs.year_int.min()), "max": int(recs.year_int.max()), "median": float(recs.year_int.median()),
                 "share_2016_or_later": float((recs.year_int >= 2016).mean()), "missing": int(recs.year_int.isna().sum())},
        "abstract": {"missing_or_under_50_chars": int((recs.abstract_chars < 50).sum()), "mean_chars": float(recs.abstract_chars.mean()),
                     "median_chars": float(recs.abstract_chars.median())},
        "original_keywords": {"documents_without_original_keywords": int((recs.n_original_keywords == 0).sum()),
                              "mean_per_document": float(recs.n_original_keywords.mean()),
                              "median_per_document": float(recs.n_original_keywords.median())},
        "sources": {"unique_sources": int(recs.source.nunique()), "top10_share": float(src.head(10).documents.sum() / len(recs))},
        "language_title": lt.set_index("language").share.round(5).head(10).to_dict(),
        "language_abstract": la.set_index("language").share.round(5).head(10).to_dict(),
        "share_abstract_english": float((recs.lang_abstract == "en").mean()),
        "share_title_english": float((recs.lang_title == "en").mean()),
        "generated_keyword_language_audit": kw_audit,
        "search_query_note": "Records retrieved from Scopus with the query reported in the repository README; "
                             "the prompt forces English output, so the CRS vocabulary is English regardless of the source language.",
    }
    C.write_json(summary, OUT / "summary.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ax[0].bar(year.year, year.documents, color="#4C72B0"); ax[0].set_title("Documents per year"); ax[0].set_xlabel("year")
        la_top = la.head(8)
        ax[1].barh(la_top.language[::-1], la_top.share[::-1], color="#55A868"); ax[1].set_xscale("log")
        ax[1].set_title("Abstract language (langdetect), share of documents"); ax[1].set_xlabel("share (log)")
        fig.tight_layout(); fig.savefig(OUT / "corpus_profile.png", dpi=150); fig.savefig(OUT / "corpus_profile.pdf")
    except Exception as exc:
        print("figure skipped:", exc)

    meta.update({"completed_utc": C.now_utc(), "timings_seconds": T.marks, "status": "complete"})
    C.write_json(meta, OUT / "metadata.json")
    C.write_json({"gate": "all records parsed into >= 6 fields", "pass": bool((recs.n_fields >= 6).all()),
                  "records_with_fewer_than_6_fields": int((recs.n_fields < 6).sum())}, OUT / "validation.json")
    print(year.tail(8).to_string(index=False)); print(la.head(8).to_string(index=False)); print(kw_audit["examples_spanish_function_words"][:10])
    print("done", T.marks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
