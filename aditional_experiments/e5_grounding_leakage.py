"""E5 - Grounding of the generated keywords and leakage from the original Scopus keywords.

Addresses: R2-4 (to what extent the generated keywords reproduce or paraphrase the author- and
index-assigned keywords that were part of the LLM input) and Editor E-2 (hallucination control).

Every generated keyword is classified against the record the model actually saw (the
``insumo`` truncated to 3,000 characters, as in notebook 1), in a fixed priority order:

  null_token          the literal placeholder "null" requested by the prompt
  exact_original      equal (after normalisation) to one of the original Scopus keywords
  verbatim_text       appears verbatim in the title or abstract
  verbatim_metadata   appears verbatim in the authors / year / source fields only
  soft_original       cosine >= 0.70 with an original keyword (paraphrase of the record keywords)
  soft_text           cosine >= 0.70 with at least one sentence of title + abstract
  ungrounded          none of the above (abstractive keyword or potential hallucination)

Because the original keywords are the last field of the record, the 3,000-character truncation
removes them entirely for some documents and partially for others. Those documents form a
natural control: the model produced keywords without seeing any original keyword.

Inputs: ``corpus_insumo_DEFINITIVO.csv`` (private, Elsevier licence; never written here), the
row -> EID index ``data/insumo_row_to_eid.csv`` and the published keywords ``EID_KEYWORDS.xlsx``.
Only records whose published output could be linked to a record are analysed (one row per EID).
Outputs contain only aggregates and per-document scores keyed by EID.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from inspec_evaluation import normalize_phrase  # noqa: E402

OUT = C.RESULTS / "e5_grounding_leakage"
CATEGORIES = ["null_token", "exact_original", "verbatim_text", "verbatim_metadata",
              "soft_original", "soft_text", "ungrounded"]
GENERIC_TERMS = {"article", "study", "analysis", "research", "work", "keyword", "keywords", "paper"}
SPANISH_STOP = {"de", "la", "el", "en", "y", "para", "del", "las", "los", "con", "una", "un", "por"}


def norm_text(s: str) -> str:
    """Same normalisation as phrases, applied to running text, so substring tests are consistent."""
    return normalize_phrase(s)


def contains_phrase(text_norm: str, phrase: str) -> bool:
    if not phrase:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text_norm) is not None


def visible_original_keywords(rec: dict, insumo: str) -> tuple[list[str], str]:
    """Original keywords whose text lies inside the first TRUNCATE_CHARS characters."""
    start = rec["keyword_field_start"]
    if start >= C.TRUNCATE_CHARS:
        return [], "not_visible"
    seen_raw = insumo[start:C.TRUNCATE_CHARS]
    full_raw = insumo[start:]
    status = "fully_visible" if len(insumo) <= C.TRUNCATE_CHARS else "partially_visible"
    # Keep only keywords completely inside the visible span (a cut keyword was not seen whole).
    kws = []
    for k in seen_raw.split(";"):
        t = normalize_phrase(k)
        if t:
            kws.append(t)
    if status == "partially_visible" and kws and not full_raw.rstrip().endswith(seen_raw.rstrip()):
        # The last visible fragment may be a truncated keyword: drop it unless it also occurs in full.
        full_set = set(C.split_original_keywords(full_raw))
        if kws[-1] not in full_set:
            kws = kws[:-1]
    return list(dict.fromkeys(kws)), status


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--insumo", default=str(C.PRIVATE_DIR / "corpus_insumo_DEFINITIVO.csv"))
    ap.add_argument("--keywords", default=str(C.REPO_ROOT / "EID_KEYWORDS.xlsx"))
    ap.add_argument("--alignment", default=str(C.DATA / "insumo_row_to_eid.csv"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="debug: process only the first N rows")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    C.set_seeds()
    T = C.Timer()
    meta = C.env_metadata(experiment="E5 grounding and leakage",
                          inputs={"insumo": {"path": args.insumo, "sha256": C.sha256(args.insumo), "redistributed": False},
                                  "keywords": {"path": args.keywords, "sha256": C.sha256(args.keywords)},
                                  "alignment": {"path": args.alignment, "sha256": C.sha256(args.alignment)}},
                          llm_calls=0, paid_api_calls=0, truncate_chars=C.TRUNCATE_CHARS, tau_soft=C.TAU_SOFT)

    # ------------------------------------------------------------------ load and align
    ins_all = pd.read_csv(args.insumo)["insumo"].astype(str).tolist()
    align = C.load_alignment(Path(args.alignment))
    pub = C.load_published_keywords(Path(args.keywords), notebook_semantics=False)
    if args.limit:
        align = align.head(args.limit)
    # Work on the linked records only: ``ins[i]`` is the record of published output ``kw.keywords[i]``.
    ins = [ins_all[r] for r in align.insumo_row]
    kw = pd.DataFrame({"eid": align.eid.values, "keywords": [pub[e] for e in align.eid]})
    n = len(kw)
    print(f"record rows={len(ins_all)} published outputs linked to a record={n}", flush=True)

    # Alignment gate: share of generated keywords found verbatim in the same row versus shifted rows.
    def literal_share(offset, rows=3000):
        vals = []
        for i in range(0, min(rows, n)):
            j = i + offset
            if j < 0 or j >= len(ins):
                continue
            kws = [normalize_phrase(k) for k in kw["keywords"][i]]
            kws = [k for k in kws if k and k != "null"]
            if kws:
                t = norm_text(ins[j])
                vals.append(np.mean([contains_phrase(t, k) for k in kws]))
        return float(np.mean(vals)) if vals else float("nan")
    align = {"offset_0": literal_share(0), "offset_plus1": literal_share(1), "offset_minus1": literal_share(-1)}
    align_pass = align["offset_0"] > 0.6 and align["offset_0"] - max(align["offset_plus1"], align["offset_minus1"]) > 0.3
    print("ALIGNMENT GATE:", "PASS" if align_pass else "FAIL", align, flush=True)
    unmatched_tail = {"record_rows_total": len(ins_all), "records_analyzed": n,
                      "records_not_linked_to_a_published_output": len(ins_all) - n}
    T.mark("load")

    # ------------------------------------------------------------------ per keyword classification (lexical part)
    recs, gen_rows = [], []
    all_gen, all_orig, all_sent = set(), set(), []
    doc_sentences = []
    for i in range(n):
        s = ins[i]
        seen = s[:C.TRUNCATE_CHARS]
        rec = C.parse_record(s)
        rec_seen = C.parse_record(seen)          # fields as the model saw them
        orig_seen, vis = visible_original_keywords(rec, s)
        orig_full = C.split_original_keywords(rec["original_keywords_raw"])
        text_seen_norm = norm_text(rec_seen["title"] + " . " + rec_seen["abstract"])
        meta_norm = norm_text(rec_seen["authors"] + " " + rec_seen["year"] + " " + rec_seen["source"])
        sents = C.split_sentences(rec_seen["title"] + ". " + rec_seen["abstract"])
        doc_sentences.append(sents)
        all_sent.extend(sents)
        all_orig.update(orig_seen)
        gen_raw = kw["keywords"][i]
        gen = []
        for g in gen_raw:
            t = normalize_phrase(g)
            gen.append(t)
        recs.append({"row": i, "eid": kw["eid"][i], "visibility": vis, "record_length": rec["length"],
                     "n_generated": len(gen_raw), "n_original_full": len(orig_full), "n_original_seen": len(orig_seen),
                     "orig_seen": orig_seen, "orig_full": orig_full})
        for pos, (raw, t) in enumerate(zip(gen_raw, gen)):
            if not t or t == "null":
                cat = "null_token"
            elif t in set(orig_seen):
                cat = "exact_original"
            elif contains_phrase(text_seen_norm, t):
                cat = "verbatim_text"
            elif contains_phrase(meta_norm, t):
                cat = "verbatim_metadata"
            else:
                cat = None                          # decided semantically below
            gen_rows.append({"row": i, "eid": kw["eid"][i], "position": pos, "keyword": t, "raw": str(raw),
                             "lexical_category": cat,
                             "in_original_seen": bool(t) and t != "null" and t in set(orig_seen),
                             "in_text": bool(t) and t != "null" and contains_phrase(text_seen_norm, t), "in_original_full_not_seen": (t in set(orig_full)) and (t not in set(orig_seen)),
                             "n_words": len(t.split()), "non_ascii": any(ord(ch) > 127 for ch in str(raw)),
                             "generic_term": t in GENERIC_TERMS, "spanish_stopword": any(w in SPANISH_STOP for w in t.split())})
            if t and t != "null":
                all_gen.add(t)
        if i % 10000 == 0:
            print(f"  lexical pass {i}/{n}", flush=True)
    G = pd.DataFrame(gen_rows)
    D = pd.DataFrame(recs)
    T.mark("lexical")

    # ------------------------------------------------------------------ semantic part
    model = C.load_embedder(threads=args.threads)
    pending = G[G.lexical_category.isna()]
    print(f"keywords needing semantic check: {len(pending)} of {len(G)}", flush=True)
    emb_gen = C.embed_unique(model, set(pending.keyword), batch_size=256, desc="generated keywords")
    emb_orig = C.embed_unique(model, all_orig, batch_size=256, desc="original keywords")
    T.mark("embed_phrases")
    # Sentences only for documents with pending keywords.
    need_docs = set(pending.row)
    sent_list = sorted(set(s for i in need_docs for s in doc_sentences[i]))
    print(f"sentences to embed: {len(sent_list)} from {len(need_docs)} documents", flush=True)
    emb_sent = C.embed_unique(model, sent_list, batch_size=256, desc="sentences")
    T.mark("embed_sentences")

    max_sim_orig = np.full(len(G), np.nan)
    max_sim_sent = np.full(len(G), np.nan)
    final = G.lexical_category.copy()
    for idx, r in pending.iterrows():
        v = emb_gen[r.keyword]
        orig = recs[r.row]["orig_seen"]
        so = max((float(v @ emb_orig[o]) for o in orig if o in emb_orig), default=np.nan)
        sents = doc_sentences[r.row]
        ss = max((float(v @ emb_sent[s]) for s in sents if s in emb_sent), default=np.nan)
        max_sim_orig[idx], max_sim_sent[idx] = so, ss
        if not np.isnan(so) and so >= C.TAU_SOFT:
            final[idx] = "soft_original"
        elif not np.isnan(ss) and ss >= C.TAU_SOFT:
            final[idx] = "soft_text"
        else:
            final[idx] = "ungrounded"
    G["category"] = final
    G["max_sim_original_seen"] = max_sim_orig
    G["max_sim_sentence"] = max_sim_sent
    T.mark("semantic")

    # ------------------------------------------------------------------ document-level adequacy by stratum
    # Global semantic similarity between the concatenated generated keywords and title + abstract
    # (the Table 2 metric; the embedder truncates the document at 128 tokens, as disclosed in the paper).
    doc_texts = [C.parse_record(ins[i][:C.TRUNCATE_CHARS]) for i in range(n)]
    doc_strings = [(d["title"] + ". " + d["abstract"]).strip() for d in doc_texts]
    kw_strings = [" ; ".join(k for k in G[G.row == i].keyword if k and k != "null") for i in range(n)]
    E_doc = model.encode(doc_strings, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    E_kw = model.encode(kw_strings, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
    D["global_sem_sim"] = np.einsum("ij,ij->i", E_doc, E_kw)
    T.mark("doc_similarity")

    # ------------------------------------------------------------------ aggregates
    G["is_leak_exact"] = G.category.eq("exact_original")
    G["is_leak_any"] = G.category.isin(["exact_original", "soft_original"])
    G["is_grounded"] = ~G.category.isin(["ungrounded", "null_token"])
    per_doc = G.groupby("row").agg(n_keywords=("keyword", "size"),
                                   n_null=("category", lambda c: (c == "null_token").sum()),
                                   n_exact_original=("is_leak_exact", "sum"),
                                   n_leak_any=("is_leak_any", "sum"),
                                   n_verbatim_text=("category", lambda c: (c == "verbatim_text").sum()),
                                   n_ungrounded=("category", lambda c: (c == "ungrounded").sum()),
                                   n_unique=("keyword", "nunique"))
    D = D.join(per_doc, on="row")
    D["share_exact_original"] = D.n_exact_original / D.n_keywords.clip(lower=1)
    D["share_leak_any"] = D.n_leak_any / D.n_keywords.clip(lower=1)
    # Reverse direction: how many of the visible original keywords were reproduced exactly.
    def reproduced(row):
        s = set(G[(G.row == row.row)].keyword)
        return sum(1 for o in row.orig_seen if o in s)
    D["n_original_seen_reproduced"] = [reproduced(r) for r in D.itertuples()]
    D["recall_original_seen"] = np.where(D.n_original_seen > 0, D.n_original_seen_reproduced / D.n_original_seen.clip(lower=1), np.nan)

    cat_share = G.category.value_counts(normalize=True).reindex(CATEGORIES).fillna(0)
    cat_count = G.category.value_counts().reindex(CATEGORIES).fillna(0).astype(int)
    shares = pd.DataFrame({"category": CATEGORIES, "keywords": cat_count.values, "share": cat_share.values})
    shares.to_csv(OUT / "category_shares.csv", index=False)

    strat = (G.merge(D[["row", "visibility"]], on="row")
             .groupby(["visibility", "category"]).size().unstack(fill_value=0).reindex(columns=CATEGORIES, fill_value=0))
    strat_share = strat.div(strat.sum(axis=1), axis=0)
    strat_share["n_keywords"] = strat.sum(axis=1)
    strat_share["n_documents"] = D.groupby("visibility").size()
    strat_share["mean_global_sem_sim"] = D.groupby("visibility").global_sem_sim.mean()
    strat_share["sd_global_sem_sim"] = D.groupby("visibility").global_sem_sim.std(ddof=1)
    strat_share["mean_record_length"] = D.groupby("visibility").record_length.mean()
    strat_share["mean_share_exact_original"] = D.groupby("visibility").share_exact_original.mean()
    strat_share["mean_recall_original_seen"] = D.groupby("visibility").recall_original_seen.mean()
    strat_share.reset_index().to_csv(OUT / "by_visibility_stratum.csv", index=False)

    # Ungrounded keywords: most frequent and a random sample for qualitative inspection.
    ung = G[G.category == "ungrounded"]
    ung.keyword.value_counts().head(50).rename_axis("keyword").reset_index(name="count").to_csv(
        OUT / "ungrounded_top50.csv", index=False)
    ung.sample(min(150, len(ung)), random_state=C.SEED)[["eid", "keyword", "max_sim_original_seen", "max_sim_sentence"]].to_csv(
        OUT / "ungrounded_sample150.csv", index=False)
    # Distribution of the best sentence similarity for ungrounded vs soft_text keywords.
    bins = np.round(np.arange(0, 1.0001, 0.05), 2)
    for cat in ("ungrounded", "soft_text", "soft_original"):
        sub = G[G.category == cat]
        col = "max_sim_original_seen" if cat == "soft_original" else "max_sim_sentence"
        h = pd.cut(sub[col], bins, include_lowest=True).value_counts().sort_index()
        h.rename_axis("bin").reset_index(name="count").to_csv(OUT / f"similarity_hist_{cat}.csv", index=False)

    adherence = {
        "documents": int(n),
        "keywords_total": int(len(G)),
        "documents_with_exactly_5_keywords": int((D.n_generated == 5).sum()),
        "documents_with_fewer_than_5": int((D.n_generated < 5).sum()),
        "documents_with_more_than_5": int((D.n_generated > 5).sum()),
        "null_tokens": int(cat_count["null_token"]),
        "null_token_rate": float(cat_count["null_token"] / len(G)),
        "documents_with_duplicate_keywords": int((D.n_unique < D.n_generated).sum()),
        "keywords_with_non_ascii_characters": int(G.non_ascii.sum()),
        "keywords_equal_to_forbidden_generic_terms": int(G.generic_term.sum()),
        "keywords_longer_than_4_words": int((G.n_words > 4).sum()),
        "keywords_with_spanish_function_words": int(G.spanish_stopword.sum()),
        "examples_spanish_function_words": G[G.spanish_stopword].keyword.value_counts().head(15).to_dict(),
    }

    valid = G[G.category != "null_token"]
    attribution = {
        "in_original_only": float((valid.in_original_seen & ~valid.in_text).mean()),
        "in_original_and_text": float((valid.in_original_seen & valid.in_text).mean()),
        "in_text_only": float((~valid.in_original_seen & valid.in_text).mean()),
        "in_neither": float((~valid.in_original_seen & ~valid.in_text).mean()),
        "keywords": int(len(valid)),
        "note": "exact lexical presence in the record the model saw; categories above are hierarchical, this table is not",
    }
    pd.DataFrame([attribution]).to_csv(OUT / "attribution_2x2.csv", index=False)
    summary = {
        "documents_analyzed": int(n),
        "attribution_2x2_exact_presence": attribution,
        "keywords_classified": int(len(G)),
        "category_shares": shares.set_index("category")["share"].round(5).to_dict(),
        "category_counts": shares.set_index("category")["keywords"].to_dict(),
        "leakage": {
            "share_keywords_exact_copy_of_visible_original": float(G.is_leak_exact.mean()),
            "share_keywords_exact_or_paraphrase_of_visible_original": float(G.is_leak_any.mean()),
            "share_keywords_matching_original_hidden_by_truncation": float(G.in_original_full_not_seen.mean()),
            "mean_per_document_share_exact_original": float(D.share_exact_original.mean()),
            "median_per_document_share_exact_original": float(D.share_exact_original.median()),
            "documents_with_zero_exact_copies": int((D.n_exact_original == 0).sum()),
            "documents_with_all_keywords_exact_copies": int((D.n_exact_original == D.n_keywords).sum()),
            "mean_recall_of_visible_original_keywords": float(D.recall_original_seen.mean()),
            "mean_n_original_keywords_visible": float(D.n_original_seen.mean()),
        },
        "grounding": {
            "share_grounded_any_level": float(G.is_grounded.mean()),
            "share_verbatim_in_record": float(G.category.isin(["exact_original", "verbatim_text", "verbatim_metadata"]).mean()),
            "share_ungrounded": float(G.category.eq("ungrounded").mean()),
            "documents_with_at_least_one_ungrounded": int((D.n_ungrounded > 0).sum()),
            "mean_max_sentence_similarity_ungrounded": float(ung.max_sim_sentence.mean()),
            "median_max_sentence_similarity_ungrounded": float(ung.max_sim_sentence.median()),
        },
        "visibility_of_original_keywords": D.visibility.value_counts().to_dict(),
        "by_visibility_stratum": strat_share.round(5).reset_index().to_dict(orient="records"),
        "prompt_adherence": adherence,
        "alignment": {**align, "pass": bool(align_pass), **unmatched_tail},
    }
    C.write_json(summary, OUT / "summary.json")
    D.drop(columns=["orig_seen", "orig_full"]).to_csv(OUT / "per_document.csv", index=False)
    G[["eid", "position", "keyword", "category", "max_sim_original_seen", "max_sim_sentence"]].to_csv(
        OUT / "per_keyword.csv.gz", index=False, compression="gzip")
    C.write_json({"gate": "row alignment insumo <-> keyword workbook", "pass": bool(align_pass), **align,
                  **unmatched_tail}, OUT / "validation.json")
    meta.update({"completed_utc": C.now_utc(), "timings_seconds": T.marks, "status": "complete"})
    C.write_json(meta, OUT / "metadata.json")
    print(shares.to_string(index=False))
    print(strat_share[["n_documents", "exact_original", "soft_original", "verbatim_text", "ungrounded", "mean_global_sem_sim"]].round(4).to_string())
    print("done", T.marks, flush=True)
    return 0 if align_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
