"""E7 - Computational cost of the LLM extraction stage: tokens, list price, wall-clock bound.

Addresses: R2-5 (computational requirements, inference cost, processing time, hardware).

The original extraction (notebook 1) was run through the OpenRouter API without logging usage,
so cost and time are reconstructed rather than read from logs:

* Input tokens: the exact chat prompt of notebook 1 (system message + user template + record
  truncated to 3,000 characters) is rebuilt for every record and tokenised with the Llama 3.1
  tokenizer, including the chat-template control tokens.
* Output tokens: the model was instructed to return ``{"keywords": [...]}``. The response is
  rebuilt from the saved keyword lists and tokenised. The reconstruction is calibrated against
  the raw responses that were preserved for the 182-document validation sample and for the 2,000
  Inspec documents (token count of the real response vs. the reconstruction).
* Cost: tokens x OpenRouter list price for ``meta-llama/llama-3.1-8b-instruct`` fetched at run
  time and stored in ``pricing_snapshot.json`` (prices at the time of the original run may differ).
* Time: the client waited 0.6 s between calls (notebook 1), which gives a strict lower bound on
  wall-clock time; per-call latency was not recorded.

Graph-stage timings and hardware are already reported in results/e3_scalability.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import inspec_evaluation as IE  # noqa: E402

OUT = C.RESULTS / "e7_extraction_cost"
MODEL_ID = "meta-llama/llama-3.1-8b-instruct"
TOKENIZER = "unsloth/Meta-Llama-3.1-8B-Instruct"   # ungated copy of the Llama 3.1 tokenizer
SLEEP_BETWEEN_CALLS = 0.6
MAX_ATTEMPTS = 4
MAX_OUTPUT_TOKENS = 700

# Verbatim from notebook 1 (`1. LLMS.ipynb`, cell 0).
SYSTEM_MSG = (
    'Respond ONLY with a valid JSON in the form {"keywords":[...]}. '
    'Do not include explanations, markdown, apologies, or code blocks. '
    'STRICT LANGUAGE RULE: '
    '- The output MUST contain ONLY ENGLISH WORDS. '
    '- Translate foreign terms into English; if translation is impossible, use a English descriptive equivalent. '
    '- Before responding, internally verify that EVERY keyword is fully in English. '
)
PROMPT_TMPL = Template("""
You are an expert assistant in bibliographic analysis.
From the following article record (authors, title, year, journal, abstract, original keywords),
extract EXACTLY 5 terms that represent the main concepts of the article.

Rules:
- ALWAYS return: {"keywords": ["term1", "term2", ...]}.
- Output MUST be ONLY in ENGLISH; no other languages under any circumstance.
- Normalize to lowercase, without accents or special characters (ej. "mathematics education").
- Accept short multiword phrases (2–4 words).
- Avoid generic terms such as: "article", "study", "analysis", "research", "work", "keyword".
- Prioritize disciplinary concepts, population, context, method, theory.
- ALWAYS return exactly 5 items in the list; if a concept cannot be expressed in English, use the string "null" as a placeholder, which still counts toward the five terms.

Article text:
$texto

Before responding, SELF-CHECK: "all the keywords are in English without exception?"
""")


def fetch_pricing() -> dict:
    try:
        with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as r:
            data = json.load(r)["data"]
        for m in data:
            if m["id"] == MODEL_ID:
                return {"model": MODEL_ID, "fetched_utc": C.now_utc(),
                        "usd_per_prompt_token": float(m["pricing"]["prompt"]),
                        "usd_per_completion_token": float(m["pricing"]["completion"]),
                        "context_length": m.get("context_length"), "source": "https://openrouter.ai/api/v1/models"}
    except Exception as exc:  # offline fallback: snapshot taken on 2026-09-08
        print("pricing fetch failed, using stored snapshot:", exc)
    return {"model": MODEL_ID, "fetched_utc": "2026-09-08 (stored snapshot; live fetch failed)",
            "usd_per_prompt_token": 5e-8, "usd_per_completion_token": 8e-8, "context_length": 131072,
            "source": "https://openrouter.ai/api/v1/models"}


def reconstruct_response(kws) -> str:
    return json.dumps({"keywords": [str(k) for k in kws]}, ensure_ascii=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--insumo", default=str(C.PRIVATE_DIR / "corpus_insumo_DEFINITIVO.csv"))
    ap.add_argument("--keywords", default=str(C.REPO_ROOT / "EID_KEYWORDS.xlsx"))
    ap.add_argument("--alignment", default=str(C.DATA / "insumo_row_to_eid.csv"))
    ap.add_argument("--sample-8b", default=str(C.DATA / "keywords_llm_llama-3.1-8b-EN.csv"))
    ap.add_argument("--inspec-8b", default=str(C.REPO_ROOT / "inspec_llama-3.1-8b-EN.csv"))
    ap.add_argument("--inspec", default=str(C.REPO_ROOT / "dataset_inspec.csv"))
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    T = C.Timer()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    meta = C.env_metadata(experiment="E7 extraction cost reconstruction", tokenizer=TOKENIZER,
                          tokenizer_vocab=len(tok),
                          inputs={k: {"path": getattr(args, k), "sha256": C.sha256(getattr(args, k))}
                                  for k in ("insumo", "keywords", "alignment", "sample_8b", "inspec_8b", "inspec")},
                          llm_calls=0, paid_api_calls=0)

    def prompt_tokens(texts):
        """Token count of the full chat prompt for each record text (batched)."""
        # Constant part: chat template + system + user template with an empty article text.
        base = tok.apply_chat_template([{"role": "system", "content": SYSTEM_MSG},
                                        {"role": "user", "content": PROMPT_TMPL.substitute(texto="")}],
                                       add_generation_prompt=True, tokenize=True)
        enc = tok(list(texts), add_special_tokens=False)["input_ids"]
        # Joining the article text inside the template can merge/split one token at each boundary.
        return np.array([len(base) + len(e) for e in enc]), len(base)

    # ------------------------------------------------------------------ calibration of the output reconstruction
    calib_rows = []
    for name, path in (("validation_sample_182", args.sample_8b), ("inspec_2000", args.inspec_8b)):
        df = pd.read_csv(path)
        real = [str(x) for x in df["raw_response"] if isinstance(x, str)]
        recon = [reconstruct_response(IE.safe_parse_list(k)) for k, r in zip(df["keywords_llm"], df["raw_response"]) if isinstance(r, str)]
        n_real = np.array([len(x) for x in tok(real, add_special_tokens=False)["input_ids"]])
        n_recon = np.array([len(x) for x in tok(recon, add_special_tokens=False)["input_ids"]])
        exact_str = float(np.mean([a.strip() == b for a, b in zip(real, recon)]))
        calib_rows.append({"set": name, "responses": len(real), "mean_tokens_real": float(n_real.mean()),
                           "mean_tokens_reconstructed": float(n_recon.mean()),
                           "ratio_real_over_reconstructed": float(n_real.sum() / n_recon.sum()),
                           "share_string_identical": exact_str,
                           "max_tokens_real": int(n_real.max()), "p99_tokens_real": float(np.percentile(n_real, 99))})
    calib = pd.DataFrame(calib_rows)
    calib.to_csv(OUT / "output_reconstruction_calibration.csv", index=False)
    ratio = float(calib.loc[calib.set == "validation_sample_182", "ratio_real_over_reconstructed"].iloc[0])
    print(calib.round(4).to_string(index=False), flush=True)
    T.mark("calibration")

    # ------------------------------------------------------------------ Inspec run (2,000 calls; same prompt)
    insp = pd.read_csv(args.inspec)
    insp_tokens, base_len = prompt_tokens([str(t)[:C.TRUNCATE_CHARS] for t in insp["insumo"]])
    insp_out = np.array([len(x) for x in tok([reconstruct_response(IE.safe_parse_list(k)) for k in pd.read_csv(args.inspec_8b)["keywords_llm"]],
                                              add_special_tokens=False)["input_ids"]])
    T.mark("inspec_tokens")

    # ------------------------------------------------------------------ full corpus (53,130 calls)
    ins = pd.read_csv(args.insumo)["insumo"].astype(str)
    align = C.load_alignment(Path(args.alignment))
    pub = C.load_published_keywords(Path(args.keywords), notebook_semantics=False)
    seen = [s[:C.TRUNCATE_CHARS] for s in ins]          # every record was sent once: input tokens for all rows
    in_tokens, base_len = prompt_tokens(seen)
    out_recon = [reconstruct_response(pub[e]) for e in align.eid]
    out_tokens = np.array([len(x) for x in tok(out_recon, add_special_tokens=False)["input_ids"]])
    # Records without a linked published output (85 of 53,130) are charged the mean output length.
    out_full = np.full(len(ins), out_tokens.mean())
    out_full[align.insumo_row.to_numpy()] = out_tokens
    eid_by_row = pd.Series(align.eid.values, index=align.insumo_row.values).reindex(range(len(ins)))
    lengths = ins.str.len().to_numpy()
    T.mark("corpus_tokens")

    pricing = fetch_pricing()
    C.write_json(pricing, OUT / "pricing_snapshot.json")
    p_in, p_out = pricing["usd_per_prompt_token"], pricing["usd_per_completion_token"]
    n_calls = len(ins)
    total_in = int(in_tokens.sum())
    total_out_recon = float(out_full.sum())
    total_out_cal = total_out_recon * ratio
    cost = {
        "calls_minimum": n_calls,
        "input_tokens_total": total_in,
        "input_tokens_mean_per_call": float(in_tokens.mean()),
        "input_tokens_max_per_call": int(in_tokens.max()),
        "prompt_constant_tokens": int(base_len),
        "output_tokens_total_reconstructed": total_out_recon,
        "output_tokens_total_calibrated": total_out_cal,
        "output_tokens_mean_per_call_calibrated": total_out_cal / n_calls,
        "usd_input_at_list_price": total_in * p_in,
        "usd_output_at_list_price": total_out_cal * p_out,
        "usd_total_at_list_price": total_in * p_in + total_out_cal * p_out,
        "usd_per_1000_documents": (total_in * p_in + total_out_cal * p_out) / n_calls * 1000,
        "usd_total_if_every_call_retried_max_attempts": (total_in * p_in + total_out_cal * p_out) * MAX_ATTEMPTS,
        "wall_clock_lower_bound_hours_from_inter_call_delay": n_calls * SLEEP_BETWEEN_CALLS / 3600,
        "wall_clock_hours_if_mean_latency_1s": n_calls * (SLEEP_BETWEEN_CALLS + 1.0) / 3600,
        "wall_clock_hours_if_mean_latency_2s": n_calls * (SLEEP_BETWEEN_CALLS + 2.0) / 3600,
        "max_output_tokens_setting": MAX_OUTPUT_TOKENS,
        "records_truncated_at_3000_chars": int((lengths > C.TRUNCATE_CHARS).sum()),
        "share_records_truncated": float((lengths > C.TRUNCATE_CHARS).mean()),
        "record_chars_mean": float(lengths.mean()), "record_chars_median": float(np.median(lengths)),
    }
    inspec_cost = {"calls": int(len(insp)), "input_tokens_total": int(insp_tokens.sum()),
                   "output_tokens_total_calibrated": float(insp_out.sum() * ratio),
                   "usd_total_at_list_price": float(insp_tokens.sum() * p_in + insp_out.sum() * ratio * p_out)}
    summary = {"model": MODEL_ID, "pricing": pricing, "output_calibration_ratio_used": ratio,
               "full_corpus": cost, "inspec_benchmark": inspec_cost,
               "notes": ["Token counts use the Llama 3.1 tokenizer with the chat template control tokens; "
                         "OpenRouter providers may add a few tokens of their own.",
                         "Prices are the list prices at fetch time; the original run took place in 2025.",
                         "Retries were not logged; the upper bound assumes every call used all 4 attempts."]}
    C.write_json(summary, OUT / "summary.json")
    pd.DataFrame({"row": np.arange(len(ins)), "eid": eid_by_row.values,
                  "record_chars": lengths, "input_tokens": in_tokens,
                  "output_tokens_reconstructed": out_full}).to_csv(OUT / "per_document_tokens.csv.gz", index=False, compression="gzip")
    meta.update({"completed_utc": C.now_utc(), "timings_seconds": T.marks, "status": "complete"})
    C.write_json(meta, OUT / "metadata.json")
    C.write_json({"gate": "output reconstruction calibrated on saved raw responses",
                  "pass": bool(0.8 <= ratio <= 1.25), "ratio_real_over_reconstructed": ratio,
                  "calibration": calib.to_dict(orient="records")}, OUT / "validation.json")
    print(json.dumps(cost, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
