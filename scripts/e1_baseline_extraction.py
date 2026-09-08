"""E1: frozen Inspec evaluation and extraction baselines (no LLM calls).

Run from the repository: .venv/bin/python scripts/e1_baseline_extraction.py
Use --validate-only to reproduce the reference LLM scores first.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import warnings

# Inference uses the already cached model, without network access.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

import inspec_evaluation as ev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e1_baselines"
SEED = 42
TOLERANCE = 0.00005  # half a unit at the reference's four-decimal precision
TOKEN_PATTERN = r"(?u)\b\w\w+\b"
VECTORIZER_CONFIG = {"ngram_range": (1, 4), "stop_words": "english", "min_df": 1,
                     "max_df": 1.0, "lowercase": True, "strip_accents": None,
                     "token_pattern": TOKEN_PATTERN, "analyzer": "word"}
BASELINE_CONFIG = {
    "target_k": 5, "gold_label_tuning": False,
    "tfidf": {**VECTORIZER_CONFIG, "norm": "l2", "use_idf": True,
              "smooth_idf": True, "sublinear_tf": False, "binary": False,
              "max_features": None, "dtype": "float64",
              "fit_scope": "all 2000 truncated insumo inputs; transductive",
              "ranking": "descending nonzero document TF-IDF; lexicographic phrase on exact ties",
              "deduplication": "unique vectorizer features; no fuzzy deduplication"},
    "yake": {"lan": "en", "n": 4, "dedup_lim": 0.9, "dedup_func": "seqm",
             "window_size": 1, "top": 5, "features": None, "lemmatize": False,
             "stopwords": "YAKE bundled English stopwords",
             "ranking": "native ascending score; stable candidate insertion order for ties",
             "top_k": "native top=5 after similarity deduplication; no padding"},
    "keybert": {**VECTORIZER_CONFIG, "model_name": ev.MODEL_NAME,
                "model_revision": ev.MODEL_REVISION, "top_n": 5,
                "use_mmr": False, "use_maxsum": False,
                "diversity": 0.5, "nr_candidates": 20,
                "diversity_settings_active": False, "seed_keywords": None,
                "llm": None, "embedding_batch_size": 128,
                "ranking": "native cosine argsort descending; NumPy default tie order; returned scores rounded to 4 decimals",
                "deduplication": "unique vectorizer features; no fuzzy deduplication",
                "fit_scope": "candidate vocabulary from all inputs; each document ranked only against its own candidates"},
    "output_policy": "keep native ranked top-five phrases, including post-cleaning collisions; never pad, drop documents, or alter saved LLM lists",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def protected_hashes():
    names = [*sorted(ROOT.glob("[1-7]. *.ipynb")),
             ROOT / "dataset_inspec.csv", ROOT / "inspec_llama-3.1-8b-EN.csv",
             ROOT / "EID_KEYWORDS.xlsx", ROOT / "human_eval_M1_8b.csv"]
    return {p.name: sha256(p) for p in names}


def check_reference_functions():
    """Compare syntax trees against reference, not just expected summary values."""
    nb = json.loads((ROOT / "6. GROUND_TRUTH_INSPEC.ipynb").read_text())
    source = "\n".join("".join(nb["cells"][i]["source"]) for i in (3, 4))
    reference = ast.parse(source)
    module = ast.parse(inspect.getsource(ev))
    reference_nodes = [ast.dump(node, include_attributes=False) for node in reference.body]
    module_nodes = [ast.dump(node, include_attributes=False) for node in module.body]
    assert all(node in module_nodes for node in reference_nodes), "Reference evaluator drift"
    return {"reference_cells_zero_based": [3, 4], "ast_identical": True}


def load_data():
    gold = pd.read_csv(ROOT / "dataset_inspec.csv", dtype={"doc_id": str})
    gold["row_abs"] = np.arange(len(gold))
    llm = pd.read_csv(ROOT / "inspec_llama-3.1-8b-EN.csv")
    assert len(gold) == len(llm) == 2000
    assert gold.doc_id.nunique() == 2000
    assert llm.row_abs.is_unique and set(llm.row_abs) == set(gold.row_abs)
    assert gold[["doc_id", "insumo", "keywords_gt"]].notna().all().all()
    # This is the ONLY table passed to extraction functions: no gold labels.
    inputs = gold[["row_abs", "doc_id", "insumo"]].copy()
    inputs["input_chars_original"] = inputs.insumo.str.len()
    inputs["insumo"] = inputs.insumo.str.slice(stop=3000)
    inputs["input_chars_used"] = inputs.insumo.str.len()
    inputs["input_sha256"] = inputs.insumo.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    llm = gold[["row_abs", "doc_id"]].merge(llm, on="row_abs", validate="one_to_one", how="left")
    return gold, inputs, llm


def make_embedder(model):
    def embed(texts):
        return model.encode(texts, batch_size=ev.BATCH_SIZE, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=False)
    return embed


def evaluate(gold, predictions, embed, method):
    assert predictions.row_abs.is_unique and set(predictions.row_abs) == set(gold.row_abs)
    joined = gold[["row_abs", "doc_id", "keywords_gt"]].merge(
        predictions[["row_abs", "keywords"]], on="row_abs", how="left", validate="one_to_one")
    # Missing field values follow notebook 6's safe_parse_list -> [] convention;
    # the row-identity check above prevents accidentally dropping observations.
    assert len(joined) == 2000
    records = []
    started = time.perf_counter()
    for row in joined.itertuples(index=False):
        gt = ev.clean_list(ev.safe_parse_list(row.keywords_gt), keep_null=False)
        pred = ev.clean_list(ev.safe_parse_list(row.keywords), keep_null=False)
        records.append({"method": method, "row_abs": row.row_abs, "doc_id": row.doc_id,
                        "gt_n": len(gt), "pred_n": len(pred),
                        "jaccard_lex": ev.jaccard(gt, pred),
                        **ev.soft_matching_metrics(gt, pred, embed, ev.TAU),
                        "global_sem_sim": ev.global_concat_similarity(gt, pred, embed)})
        if len(records) % 100 == 0:
            print(f"{method}: evaluated {len(records)}/2000 ({time.perf_counter()-started:.1f}s)", flush=True)
    result = pd.DataFrame(records)
    assert np.isfinite(result[ev.METRICS].to_numpy()).all()
    return result, time.perf_counter() - started


def summary(scores):
    return pd.DataFrame([{"method": method, "metric": metric, "n": len(group),
                          "mean": group[metric].mean(), "sd": group[metric].std(ddof=1)}
                         for method, group in scores.groupby("method", sort=False)
                         for metric in ev.METRICS])


def llm_predictions(llm):
    result = llm[["row_abs", "doc_id"]].copy()
    result["method"] = "llama_3.1_8b"
    # Preserve the saved string and list ordering; never enforce k on this file.
    result["keywords"] = llm.keywords_llm
    result["raw_n"] = result.keywords.map(lambda s: len(ev.safe_parse_list(s)))
    result["clean_n"] = result.keywords.map(lambda s: len(ev.clean_list(ev.safe_parse_list(s))))
    result["missing_prediction"] = llm.keywords_llm.isna()
    result["missing_raw_response"] = llm.raw_response.isna()
    result["status"] = np.where(result.missing_prediction, "missing_saved_prediction",
                                 np.where(result.raw_n == 0, "empty_saved_prediction", "saved_prediction"))
    result["error"] = ""
    result["extraction_seconds"] = np.nan  # historical runtime is unavailable
    return result


def prediction_record(row, method, ranked, seconds, error="", warning_text="", candidate_n=None):
    phrases = [str(phrase) for phrase, _ in ranked]
    clean = ev.clean_list(phrases)
    status = "failure" if error else ("empty_output" if not phrases else "ok")
    return {"method": method, "row_abs": int(row.row_abs), "doc_id": row.doc_id,
            "keywords": json.dumps(phrases, ensure_ascii=False),
            "scores": json.dumps([float(score) for _, score in ranked]),
            "raw_n": len(phrases), "clean_n": len(clean), "candidate_n": candidate_n,
            "status": status, "error": error, "warnings": warning_text,
            "missing_prediction": False, "missing_raw_response": None,
            "extraction_seconds": seconds}


class CaptureWarnings(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def extract_tfidf(inputs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    started = time.perf_counter()
    vectorizer = TfidfVectorizer(**VECTORIZER_CONFIG, norm="l2", use_idf=True,
                                 smooth_idf=True, sublinear_tf=False, binary=False,
                                 max_features=None, dtype=np.float64)
    matrix = vectorizer.fit_transform(inputs.insumo.tolist())
    features = vectorizer.get_feature_names_out()
    fit_seconds = time.perf_counter() - started
    records = []
    for i, row in enumerate(inputs.itertuples(index=False)):
        begin = time.perf_counter()
        sparse = matrix.getrow(i)
        ranked = sorted(zip(features[sparse.indices], sparse.data), key=lambda pair: (-pair[1], pair[0]))[:5]
        records.append(prediction_record(row, "tfidf", ranked, time.perf_counter() - begin,
                                         candidate_n=sparse.nnz))
    return pd.DataFrame(records), {"extraction_seconds": time.perf_counter() - started,
                                    "fit_seconds": fit_seconds, "vocabulary_size": len(features),
                                    "per_document_timing": "ranking only; corpus fit reported separately"}


def extract_yake(inputs):
    import yake
    extractor = yake.KeywordExtractor(lan="en", n=4, dedup_lim=0.9, dedup_func="seqm",
                                      window_size=1, top=5, features=None, lemmatize=False)
    write_json(OUT / "yake_stopwords.json", sorted(extractor.stopword_set))
    records = []
    started = time.perf_counter()
    for row in inputs.itertuples(index=False):
        capture = CaptureWarnings()
        logging.getLogger().addHandler(capture)
        begin = time.perf_counter()
        error = ""
        ranked = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ranked = extractor.extract_keywords(row.insumo)
            capture.messages.extend(str(w.message) for w in caught)
            # YAKE catches exceptions internally and emits a warning before returning [].
            errors = [message for message in capture.messages if "Exception during keyword extraction" in message]
            error = " | ".join(errors)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            logging.getLogger().removeHandler(capture)
        records.append(prediction_record(row, "yake", ranked, time.perf_counter() - begin,
                                         error, " | ".join(capture.messages)))
        if len(records) % 200 == 0:
            print(f"yake: extracted {len(records)}/2000", flush=True)
    return pd.DataFrame(records), {"extraction_seconds": time.perf_counter() - started,
                                    "per_document_timing": "full extraction, excluding extractor initialization"}


def extract_keybert(inputs, model):
    from keybert import KeyBERT
    from keybert.backend import BaseEmbedder
    from sklearn.feature_extraction.text import CountVectorizer

    class BatchedMiniLM(BaseEmbedder):
        """Standard sentence embeddings, batched for CPU memory and progress reporting."""
        def __init__(self):
            super().__init__(embedding_model=model)
            self.timings = []

        def embed(self, documents, verbose=False):
            begin = time.perf_counter()
            parts = []
            for start in range(0, len(documents), 4096):
                parts.append(model.encode(list(documents[start:start + 4096]), batch_size=128,
                                          show_progress_bar=False, convert_to_numpy=True,
                                          normalize_embeddings=False))
                print(f"keybert embeddings: {min(start + 4096, len(documents))}/{len(documents)}", flush=True)
            self.timings.append({"text_count": len(documents), "seconds": time.perf_counter() - begin})
            return np.concatenate(parts)

    started = time.perf_counter()
    backend = BatchedMiniLM()
    extractor = KeyBERT(model=backend)
    assert extractor.llm is None
    vectorizer = CountVectorizer(**VECTORIZER_CONFIG)
    ranked_lists = extractor.extract_keywords(inputs.insumo.tolist(), vectorizer=vectorizer,
                                              top_n=5, use_mmr=False, use_maxsum=False,
                                              diversity=0.5, nr_candidates=20, seed_keywords=None)
    if len(ranked_lists) != len(inputs):
        raise RuntimeError(f"KeyBERT returned {len(ranked_lists)} results for {len(inputs)} inputs")
    counts = vectorizer.transform(inputs.insumo.tolist()).getnnz(axis=1)
    records = []
    for row, ranked, candidate_n in zip(inputs.itertuples(index=False), ranked_lists, counts, strict=True):
        # Native KeyBERT catches ValueError and returns []; expose this if candidates existed.
        error = "KeyBERT returned empty output despite available candidates; possible internal ValueError" if candidate_n and not ranked else ""
        records.append(prediction_record(row, "keybert", ranked, None, error=error,
                                         candidate_n=int(candidate_n)))
    return pd.DataFrame(records), {"extraction_seconds": time.perf_counter() - started,
                                    "embedding_stages": backend.timings,
                                    "vocabulary_size": len(vectorizer.get_feature_names_out()),
                                    "per_document_timing": "unavailable: batched corpus extraction; no imputed per-document times"}


def save_aggregate_outputs(predictions, scores, runtimes):
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_scores = pd.concat(scores, ignore_index=True)
    assert len(all_predictions) == len(all_scores) == 8000
    assert not all_predictions.duplicated(["method", "row_abs"]).any()
    assert not all_scores.duplicated(["method", "row_abs"]).any()
    all_predictions.to_csv(OUT / "predictions_all.csv", index=False)
    all_scores.to_csv(OUT / "metrics_all.csv", index=False)
    table = summary(all_scores)
    table.to_csv(OUT / "summary.csv", index=False)
    distributions = []
    for field in ("raw_n", "clean_n"):
        dist = all_predictions.groupby(["method", field], dropna=False).size().reset_index(name="documents")
        dist = dist.rename(columns={field: "phrase_count"})
        dist["count_type"] = field
        distributions.append(dist)
    pd.concat(distributions, ignore_index=True).to_csv(OUT / "output_length_distributions.csv", index=False)
    quality = []
    for method, group in all_predictions.groupby("method", sort=False):
        method_scores = all_scores[all_scores.method == method]
        quality.append({"method": method, "documents": len(group),
                        "raw_empty": int((group.raw_n == 0).sum()),
                        "clean_empty": int((group.clean_n == 0).sum()),
                        "raw_not_five": int((group.raw_n != 5).sum()),
                        "clean_not_five": int((group.clean_n != 5).sum()),
                        "reported_failures": int(group.error.fillna("").ne("").sum()),
                        "historical_failure_status": "unavailable" if method == "llama_3.1_8b" else "not_applicable",
                        "missing_predictions": int(group.missing_prediction.fillna(False).sum()),
                        "missing_raw_responses": int(group.missing_raw_response.fillna(False).sum()) if method == "llama_3.1_8b" else None,
                        "warning_documents": int(group.warnings.fillna("").ne("").sum()),
                        "missing_metric_values": int(method_scores[ev.METRICS].isna().sum().sum())})
    pd.DataFrame(quality).to_csv(OUT / "quality_counts.csv", index=False)
    pd.DataFrame(runtimes).to_csv(OUT / "runtimes.csv", index=False)
    lines = ["# E1 extraction baseline results", "", "All scores are document means ± sample SD (ddof=1), n=2,000 per method.", "",
             "| Method | " + " | ".join(ev.METRICS) + " |",
             "|---|" + "---|" * len(ev.METRICS)]
    for method in all_scores.method.unique():
        rows = table[table.method == method].set_index("metric")
        lines.append("| " + method + " | " + " | ".join(
            f"{rows.loc[m, 'mean']:.4f} ± {rows.loc[m, 'sd']:.4f}" for m in ev.METRICS) + " |")
    lines += ["", "See README.md for configurations, timing scope, and methodological limitations.", ""]
    (OUT / "comparison.md").write_text("\n".join(lines))


def verify_completed_artifacts():
    """Read back results and independently check identities, counts and summaries."""
    check_reference_functions()
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert protected_hashes() == json.loads((OUT / "protected_artifact_hashes.json").read_text())
    for name, checksum in metadata["source_sha256"].items():
        assert sha256(ROOT / name) == checksum, f"Source changed since run: {name}"
    gold, inputs, llm = load_data()
    predictions = pd.read_csv(OUT / "predictions_all.csv", dtype={"doc_id": str})
    scores = pd.read_csv(OUT / "metrics_all.csv", dtype={"doc_id": str})
    methods = {"llama_3.1_8b", "tfidf", "yake", "keybert"}
    assert set(predictions.method) == set(scores.method) == methods
    assert len(predictions) == len(scores) == 8000
    assert not predictions.duplicated(["method", "row_abs"]).any()
    assert not scores.duplicated(["method", "row_abs"]).any()
    assert np.isfinite(scores[ev.METRICS].to_numpy()).all()
    gold_lists = [ev.clean_list(ev.safe_parse_list(s)) for s in gold.keywords_gt]
    for method in sorted(methods):
        p = predictions[predictions.method == method].sort_values("row_abs").reset_index(drop=True)
        s = scores[scores.method == method].sort_values("row_abs").reset_index(drop=True)
        assert len(p) == len(s) == 2000
        assert p.row_abs.tolist() == s.row_abs.tolist() == list(range(2000))
        assert p.doc_id.tolist() == s.doc_id.tolist() == gold.doc_id.tolist()
        raw = [ev.safe_parse_list(value) for value in p.keywords]
        clean = [ev.clean_list(value) for value in raw]
        assert p.raw_n.tolist() == [len(value) for value in raw]
        assert p.clean_n.tolist() == s.pred_n.tolist() == [len(value) for value in clean]
        assert s.gt_n.tolist() == [len(value) for value in gold_lists]
        lexical = [ev.jaccard(g, candidate) for g, candidate in zip(gold_lists, clean, strict=True)]
        np.testing.assert_allclose(s.jaccard_lex, lexical, rtol=0, atol=1e-15)
        single = pd.read_csv(OUT / f"metrics_{method}.csv", dtype={"doc_id": str})
        pd.testing.assert_frame_equal(s, single.reset_index(drop=True), check_exact=False, atol=1e-15, rtol=0)
        if method == "llama_3.1_8b":
            assert p.keywords.tolist() == llm.keywords_llm.tolist(), "Saved LLM predictions altered"
    expected_summary = summary(scores).sort_values(["method", "metric"]).reset_index(drop=True)
    saved_summary = pd.read_csv(OUT / "summary.csv").sort_values(["method", "metric"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected_summary, saved_summary, check_exact=False, atol=1e-15, rtol=0)
    manifest = pd.read_csv(OUT / "input_manifest.csv", dtype={"doc_id": str})
    pd.testing.assert_frame_equal(manifest, inputs.drop(columns="insumo"), check_dtype=False)
    validation = json.loads((OUT / "llm_validation.json").read_text())
    assert validation["passed"]
    for name, checksum in validation["artifact_sha256"].items():
        assert sha256(ROOT / name) == checksum
    report = {"passed": True, "methods": sorted(methods), "prediction_rows": len(predictions),
              "metric_rows": len(scores), "metric_values": len(scores) * len(ev.METRICS),
              "missing_metric_values": 0, "row_identity_and_counts_checked": True,
              "saved_llm_predictions_identical": True, "lexical_scores_recomputed": True,
              "summary_means_and_sample_sds_recomputed": True,
              "protected_artifacts_unchanged": True, "input_manifest_verified": True,
              "verified_utc": datetime.now(timezone.utc).isoformat()}
    write_json(OUT / "artifact_validation.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--reuse-validated-llm", action="store_true",
                        help="Reuse a checksummed successful reproduction with identical inputs, evaluator and packages")
    parser.add_argument("--verify-results", action="store_true", help="Verify completed result files without model inference")
    args = parser.parse_args()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "ieee-revision-experiments":
        raise RuntimeError(f"Unexpected branch: {branch}")
    if args.verify_results:
        verify_completed_artifacts()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    initial_hashes = protected_hashes()
    reference_check = check_reference_functions()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    gold, inputs, llm = load_data()
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    cached_validation = None
    if args.reuse_validated_llm:
        cached_validation = json.loads((OUT / "llm_validation.json").read_text())
        assert cached_validation["passed"]
        assert json.loads((OUT / "protected_artifact_hashes.json").read_text()) == initial_hashes
        assert json.loads((OUT / "package_versions.json").read_text()) == versions
        for name, checksum in cached_validation["artifact_sha256"].items():
            assert sha256(ROOT / name) == checksum, f"Validated artifact changed: {name}"
    write_json(OUT / "package_versions.json", dict(sorted(versions.items())))
    write_json(OUT / "protected_artifact_hashes.json", initial_hashes)
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "branch": branch,
                "seed": SEED, "python": platform.python_version(), "platform": platform.platform(),
                "machine": platform.machine(), "cpu_count": os.cpu_count(), "torch_threads": 4,
                "device": "cpu", "model_name": ev.MODEL_NAME, "model_revision": ev.MODEL_REVISION,
                "tau": ev.TAU, "batch_size": ev.BATCH_SIZE, "sd_ddof": 1,
                "input_count": len(inputs), "input_truncation_chars": 3000,
                "truncated_document_ids": inputs.loc[inputs.input_chars_original > 3000, ["row_abs", "doc_id"]].to_dict("records"),
                "paid_api_calls": 0, "historical_llm_runtime_seconds": None,
                "historical_llm_cost": None, "reference_check": reference_check}
    metadata["llm_validation_reused"] = args.reuse_validated_llm
    metadata["baseline_configurations"] = BASELINE_CONFIG
    metadata["hardware_detail_unavailable"] = "CPU brand and physical RAM sysctl query denied by local sandbox; platform and logical CPU count measured"
    inputs.drop(columns="insumo").to_csv(OUT / "input_manifest.csv", index=False)
    started = time.perf_counter()
    model = SentenceTransformer(ev.MODEL_NAME, revision=ev.MODEL_REVISION,
                                device="cpu", local_files_only=True)
    metadata["model_load_seconds"] = time.perf_counter() - started
    metadata["model_max_seq_length"] = model.max_seq_length
    metadata["model_dtype"] = str(next(model.parameters()).dtype)
    write_json(OUT / "metadata.json", metadata)
    write_json(OUT / "configuration.json", {"seed": SEED, "input_column": "insumo",
                                           "truncate_chars": 3000,
                                           "evaluation": {"model": ev.MODEL_NAME, "revision": ev.MODEL_REVISION,
                                                          "tau": ev.TAU, "batch_size": ev.BATCH_SIZE,
                                                          "keep_null": False, "ddof": 1,
                                                          "normalize_embeddings": False,
                                                          "reference_function_cells": [3, 4]},
                                           "baselines": BASELINE_CONFIG})
    embed = make_embedder(model)
    preds = llm_predictions(llm)
    if args.reuse_validated_llm:
        scores = pd.read_csv(OUT / "metrics_llama_3.1_8b.csv", dtype={"doc_id": str})
        assert len(scores) == 2000 and scores.row_abs.is_unique
        assert scores.doc_id.tolist() == gold.doc_id.tolist()
        assert scores.row_abs.tolist() == gold.row_abs.tolist()
        elapsed = cached_validation["evaluation_seconds"]
    else:
        preds.to_csv(OUT / "predictions_llama_3.1_8b.csv", index=False)
        scores, elapsed = evaluate(gold, preds, embed, "llama_3.1_8b")
        scores.to_csv(OUT / "metrics_llama_3.1_8b.csv", index=False)
    checks = []
    for metric in ev.METRICS:
        actual = float(scores[metric].mean())
        checks.append({"metric": metric, "expected": ev.EXPECTED[metric], "reproduced": actual,
                       "difference": actual - ev.EXPECTED[metric],
                       "passed": abs(actual - ev.EXPECTED[metric]) <= TOLERANCE})
    validation = {"checks": checks, "absolute_tolerance": TOLERANCE,
                  "passed": all(c["passed"] for c in checks),
                  "evaluation_seconds": elapsed, **reference_check}
    validation["artifact_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in
                                      [ROOT / "scripts/inspec_evaluation.py",
                                       OUT / "predictions_llama_3.1_8b.csv", OUT / "metrics_llama_3.1_8b.csv"]}
    write_json(OUT / "llm_validation.json", validation)
    summary(scores).to_csv(OUT / "summary.csv", index=False)
    print(json.dumps(validation, indent=2), flush=True)
    assert protected_hashes() == initial_hashes, "Protected input changed"
    if not validation["passed"]:
        raise RuntimeError("LLM reproduction differs from saved reference; STOP before baseline execution")
    if args.validate_only:
        return
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    write_json(OUT / "sklearn_english_stopwords.json", sorted(ENGLISH_STOP_WORDS))
    all_predictions, all_scores = [preds], [scores]
    runtimes = [{"method": "llama_3.1_8b", "extraction_seconds": None,
                 "evaluation_seconds": elapsed, "extraction_source": "historical; unavailable",
                 "evaluation_source": "independent local reproduction"}]
    metadata["method_details"] = {}
    for method, extractor in [("tfidf", extract_tfidf), ("yake", extract_yake),
                              ("keybert", lambda frame: extract_keybert(frame, model))]:
        print(f"Starting {method} extraction (no gold columns in extractor input)", flush=True)
        method_predictions, detail = extractor(inputs)
        assert len(method_predictions) == 2000 and method_predictions.row_abs.is_unique
        method_predictions.to_csv(OUT / f"predictions_{method}.csv", index=False)
        method_scores, eval_seconds = evaluate(gold, method_predictions, embed, method)
        method_scores.to_csv(OUT / f"metrics_{method}.csv", index=False)
        all_predictions.append(method_predictions)
        all_scores.append(method_scores)
        runtimes.append({"method": method, "extraction_seconds": detail["extraction_seconds"],
                         "evaluation_seconds": eval_seconds, "extraction_source": "measured locally",
                         "evaluation_source": "measured locally"})
        metadata["method_details"][method] = detail
        write_json(OUT / "metadata.json", metadata)
        print(f"Completed {method}: extraction={detail['extraction_seconds']:.2f}s evaluation={eval_seconds:.2f}s", flush=True)
    save_aggregate_outputs(all_predictions, all_scores, runtimes)
    assert protected_hashes() == initial_hashes, "Protected input changed"
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["status"] = "complete"
    metadata["protected_artifacts_unchanged"] = True
    metadata["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "scripts").glob("*.py"))}
    write_json(OUT / "metadata.json", metadata)
    print((OUT / "comparison.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
