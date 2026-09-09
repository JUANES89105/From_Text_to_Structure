"""E4 (step 1) - Re-extract keywords with a different cardinality k on a fixed 5,000-document sample.

Addresses: R3-2 (k = 5 is arbitrary and may omit content of complex documents).

The sample is the first replicate of the 5,000-document size of experiment E3
(``results/e3_scalability/samples/n5000_r1_s42.csv``), so the k = 5 arm is already available from
the published keywords and its sampling variability is known from E3's ten replicates.

The prompt is the notebook-1 prompt with the cardinality changed (``EXACTLY 5`` -> ``EXACTLY k``);
model, decoding (temperature 0, max 700 output tokens), retry policy (4 attempts, 0.6 s between
calls) and 3,000-character truncation are unchanged. Usage returned by the API is logged per call,
so the actual token cost of this run is measured rather than reconstructed.

Requires OPENROUTER_API_KEY unless --dry-run is given. --dry-run builds every prompt, writes the
sample <-> record mapping and a cost estimate, and performs no network call.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from string import Template

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

OUT = C.RESULTS / "e4_k_sensitivity"
MODEL_ID = "meta-llama/llama-3.1-8b-instruct"
MAX_ATTEMPTS, SLEEP_BETWEEN_CALLS, MAX_OUTPUT_TOKENS, TEMPERATURE = 4, 0.6, 700, 0.0
ACCEPT_KEYS = ["keywords", "keyword", "palabras_clave", "palabras", "términos", "terminos"]

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
extract EXACTLY $k terms that represent the main concepts of the article.

Rules:
- ALWAYS return: {"keywords": ["term1", "term2", ...]}.
- Output MUST be ONLY in ENGLISH; no other languages under any circumstance.
- Normalize to lowercase, without accents or special characters (ej. "mathematics education").
- Accept short multiword phrases (2–4 words).
- Avoid generic terms such as: "article", "study", "analysis", "research", "work", "keyword".
- Prioritize disciplinary concepts, population, context, method, theory.
- ALWAYS return exactly $k items in the list; if a concept cannot be expressed in English, use the string "null" as a placeholder, which still counts toward the $k terms.

Article text:
$texto

Before responding, SELF-CHECK: "all the keywords are in English without exception?"
""")


def extract_json(text: str) -> list | None:
    """Fault-tolerant parsing, following the logic of notebook 1 (object, then bare list)."""
    t = str(text).strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            for key in ACCEPT_KEYS:
                if isinstance(obj, dict) and key in obj and isinstance(obj[key], list):
                    return [str(x).strip().lower() for x in obj[key]]
        except Exception:
            pass
    m = re.search(r"\[.*\]", t, flags=re.S)
    if m:
        try:
            lst = json.loads(m.group(0))
            if isinstance(lst, list):
                return [str(x).strip().lower() for x in lst]
        except Exception:
            items = re.findall(r'"([^"]+)"', m.group(0))
            if items:
                return [x.strip().lower() for x in items]
    return None


def build_mapping(sample_path: Path, alignment_path: Path, insumo_path: Path) -> pd.DataFrame:
    sample = pd.read_csv(sample_path)
    align = C.load_alignment(alignment_path)
    first_row = dict(zip(align.eid, align.insumo_row))
    sample["insumo_row"] = [first_row.get(str(e)) for e in sample["EID_o_identificador"]]
    missing = sample.insumo_row.isna().sum()
    if missing:
        raise SystemExit(f"{missing} sample EIDs could not be linked to a record")
    ins = pd.read_csv(insumo_path)["insumo"].astype(str)
    sample["record_chars"] = [len(ins.iloc[int(r)]) for r in sample.insumo_row]
    sample["text_seen"] = [ins.iloc[int(r)][:C.TRUNCATE_CHARS] for r in sample.insumo_row]
    return sample


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, action="append", help="cardinality (repeatable); default 10")
    ap.add_argument("--sample", default=str(C.REPO_ROOT / "results/e3_scalability/samples/n5000_r1_s42.csv"))
    ap.add_argument("--alignment", default=str(C.DATA / "insumo_row_to_eid.csv"))
    ap.add_argument("--insumo", default=str(C.PRIVATE_DIR / "corpus_insumo_DEFINITIVO.csv"))
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    ks = args.k or [10]
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = build_mapping(Path(args.sample), Path(args.alignment), Path(args.insumo))
    if args.max_docs:
        mapping = mapping.head(args.max_docs)
    mapping.drop(columns=["text_seen"]).to_csv(OUT / "sample_mapping.csv", index=False)

    manifest = {
        "created_utc": C.now_utc(), "model": args.model, "k_values": ks, "documents": int(len(mapping)),
        "sample_file": args.sample, "sample_sha256": C.sha256(args.sample),
        "decoding": {"temperature": TEMPERATURE, "max_tokens": MAX_OUTPUT_TOKENS},
        "retry_policy": {"max_attempts": MAX_ATTEMPTS, "sleep_between_calls_s": SLEEP_BETWEEN_CALLS},
        "truncate_chars": C.TRUNCATE_CHARS, "records_truncated": int((mapping.record_chars > C.TRUNCATE_CHARS).sum()),
        "system_message": SYSTEM_MSG, "prompt_template": PROMPT_TMPL.template,
        "expected_calls": int(len(mapping) * len(ks)),
        "wall_clock_lower_bound_hours": len(mapping) * len(ks) * SLEEP_BETWEEN_CALLS / 3600,
    }
    price_file = C.RESULTS / "e7_extraction_cost" / "pricing_snapshot.json"
    if price_file.exists():
        pr = json.load(open(price_file))
        est_in = 640 * len(mapping) * len(ks)             # mean prompt length measured in E7
        est_out = 27 * len(mapping) * len(ks) * (max(ks) / 5)
        manifest["estimated_cost_usd_at_list_price"] = est_in * pr["usd_per_prompt_token"] + est_out * pr["usd_per_completion_token"]
    C.write_json(manifest, OUT / ("dry_run_manifest.json" if args.dry_run else "run_manifest.json"))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("system_message", "prompt_template")}, indent=1))
    if args.dry_run:
        print("dry run: no API call performed")
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set; use --dry-run to validate the setup")
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    for k in ks:
        out_path = OUT / f"predictions_k{k}.csv"
        done = set()
        if out_path.exists():
            done = set(pd.read_csv(out_path)["EID_o_identificador"].astype(str))
            print(f"resuming k={k}: {len(done)} documents already extracted")
        rows = []
        for _, r in mapping.iterrows():
            if str(r.EID_o_identificador) in done:
                continue
            prompt = PROMPT_TMPL.substitute(k=k, texto=r.text_seen)
            raw, kws, status, usage, t0 = None, None, "error", {}, time.perf_counter()
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    resp = client.chat.completions.create(
                        model=args.model, temperature=TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS,
                        messages=[{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": prompt}])
                    raw = resp.choices[0].message.content
                    usage = {"prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                             "completion_tokens": getattr(resp.usage, "completion_tokens", None)}
                    kws = extract_json(raw)
                    status = "ok" if kws is not None else "unparseable"
                    break
                except Exception as exc:  # transient errors: retry, as in notebook 1
                    status = f"error: {type(exc).__name__}"
                    time.sleep(SLEEP_BETWEEN_CALLS * attempt)
            rows.append({"corpus_position": r.corpus_position, "EID_o_identificador": r.EID_o_identificador,
                         "insumo_row": r.insumo_row, "k": k, "keywords_llm": str(kws) if kws is not None else None,
                         "n_keywords": len(kws) if kws else 0, "raw_response": raw, "status": status,
                         "attempts": attempt, "latency_s": round(time.perf_counter() - t0, 3), **usage})
            if len(rows) % 100 == 0:
                pd.DataFrame(rows).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
                rows = []
                print(f"  k={k}: {len(done) + _ + 1} documents", flush=True)
            time.sleep(SLEEP_BETWEEN_CALLS)
        if rows:
            pd.DataFrame(rows).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
