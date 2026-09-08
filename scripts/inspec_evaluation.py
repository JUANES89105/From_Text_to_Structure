"""Inspec evaluator: reference notebook 6 cells 3–4 copied verbatim.

No extraction, downloads, or file writes occur on import. Scoring functions
retain the reference semantics, including phrase order and empty lists.
"""
import ast
import re
import numpy as np

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
TAU = 0.70
BATCH_SIZE = 128
METRICS = ["jaccard_lex", "soft_precision", "soft_recall", "soft_f1",
           "soft_mean_max", "global_sem_sim"]
EXPECTED = dict(zip(METRICS, [0.1422, 0.7939, 0.4670, 0.5652, 0.7522, 0.8042]))

_whitespace_re = re.compile(r"\s+")
_non_alnum_space_re = re.compile(r"[^a-z0-9 ]+")


def normalize_phrase(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = _whitespace_re.sub(" ", s)
    s = _non_alnum_space_re.sub(" ", s)
    return _whitespace_re.sub(" ", s).strip()


def safe_parse_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    s = str(x).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        return [str(i) for i in v] if isinstance(v, list) else []
    except Exception:
        s = s.strip("[]")
        return [p.strip().strip('"').strip("'") for p in s.split(",") if p.strip()]


def clean_list(items, keep_null=False):
    out = []
    for it in items:
        t = normalize_phrase(it)
        if not t or (t == "null" and not keep_null):
            continue
        out.append(t)
    seen, dedup = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            dedup.append(t)
    return dedup


def jaccard(a, b):
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def cosine_sim_matrix(X, Y):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    return Xn @ Yn.T


def soft_matching_metrics(gt_phrases, llm_phrases, embed_fn, tau):
    """Soft precision/recall/F1 (coverage-style) plus soft mean-max."""
    if not gt_phrases and not llm_phrases:
        return {"soft_recall": 1.0, "soft_precision": 1.0, "soft_f1": 1.0, "soft_mean_max": 1.0}
    if not gt_phrases or not llm_phrases:
        return {"soft_recall": 0.0, "soft_precision": 0.0, "soft_f1": 0.0, "soft_mean_max": 0.0}

    E_gt = embed_fn(gt_phrases)
    E_llm = embed_fn(llm_phrases)
    S = cosine_sim_matrix(E_gt, E_llm)

    gt_best = S.max(axis=1)
    llm_best = S.max(axis=0)

    soft_recall = float(np.mean(gt_best >= tau))
    soft_precision = float(np.mean(llm_best >= tau))
    soft_f1 = 0.0 if (soft_recall + soft_precision) == 0 else float(
        2 * soft_recall * soft_precision / (soft_recall + soft_precision)
    )
    soft_mean_max = float(0.5 * (gt_best.mean() + llm_best.mean()))

    return {
        "soft_recall": soft_recall,
        "soft_precision": soft_precision,
        "soft_f1": soft_f1,
        "soft_mean_max": soft_mean_max,
    }


def global_concat_similarity(gt_phrases, llm_phrases, embed_fn):
    """Cosine similarity between the two keyphrase sets, each concatenated with " ; "."""
    if not gt_phrases and not llm_phrases:
        return 1.0
    if not gt_phrases or not llm_phrases:
        return 0.0
    gt_text = " ; ".join(gt_phrases)
    llm_text = " ; ".join(llm_phrases)
    E = embed_fn([gt_text, llm_text])
    x, y = E[0], E[1]
    return float(np.dot(x, y) / ((np.linalg.norm(x) * np.linalg.norm(y)) + 1e-12))

