"""Build ``data/insumo_row_to_eid.csv``: the index that links each line of the private record file
(``corpus_insumo_DEFINITIVO.csv``, the input of notebook 1) to the Scopus EID of the published output.

The record file has no identifier column. The row order of the extraction log that the authors kept
locally follows the record file, but that log also contains rows from a discarded, differently
configured pass and was therefore never published. This script uses the log **only** to recover the
row -> EID correspondence, verifies the correspondence independently (share of published keywords
found verbatim in the record of the same row versus neighbouring rows), and keeps a row only when
the keyword list stored in the log is identical to the published list in ``EID_KEYWORDS.xlsx``.
No keyword from the log is written anywhere; downstream scripts read keywords from the published file.

Output columns: ``insumo_row`` (0-based line in the record file), ``eid``, ``matches_published``.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from crs_reference import parse_keywords  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--insumo", default=str(C.PRIVATE_DIR / "corpus_insumo_DEFINITIVO.csv"))
    ap.add_argument("--extraction-log", default=str(C.PRIVATE_DIR / "EID_KEYWORDS_with_duplicates.xlsx"),
                    help="local, unpublished extraction log in record order (used only for the row -> EID index)")
    ap.add_argument("--published", default=str(C.REPO_ROOT / "EID_KEYWORDS.xlsx"))
    args = ap.parse_args(argv)
    ins = pd.read_csv(args.insumo)["insumo"].astype(str).tolist()
    log = pd.read_excel(args.extraction_log)
    pub = pd.read_excel(args.published)
    eid_l = log.iloc[:, 0].astype(str); kw_l = log.iloc[:, 1]
    eid_p = pub.iloc[:, 0].astype(str); kw_p = pub.iloc[:, 1]
    norm = lambda x: tuple(sorted(str(k).strip().lower() for k in parse_keywords(x)))
    published = dict(zip(eid_p, (norm(x) for x in kw_p)))
    n = min(len(ins), len(log))
    match = np.array([eid_l[i] in published and norm(kw_l[i]) == published[eid_l[i]] for i in range(n)])

    # Independent alignment check with the *published* keywords only.
    def literal(i, off):
        j = i + off
        if j < 0 or j >= len(ins) or not match[i]:
            return np.nan
        t = re.sub(r"[^a-z0-9 ]+", " ", ins[j].lower())
        ks = [k for k in published[eid_l[i]] if k != "null"]
        ks = [re.sub(r"[^a-z0-9 ]+", " ", k) for k in ks]
        return np.mean([k in t for k in ks]) if ks else np.nan
    idx = np.linspace(0, n - 1, 4000).astype(int)
    align = {f"offset_{o:+d}": float(np.nanmean([literal(i, o) for i in idx])) for o in (-1, 0, 1)}
    gate = align["offset_+0"] > 0.6 and align["offset_+0"] - max(align["offset_-1"], align["offset_+1"]) > 0.3
    out = pd.DataFrame({"insumo_row": np.arange(n), "eid": eid_l[:n].values, "matches_published": match})
    # Rows of the record file beyond the log (no saved output) are listed with no EID.
    if len(ins) > n:
        out = pd.concat([out, pd.DataFrame({"insumo_row": np.arange(n, len(ins)), "eid": None, "matches_published": False})], ignore_index=True)
    out.loc[~out.matches_published, "eid"] = None
    C.DATA.mkdir(parents=True, exist_ok=True)
    out.to_csv(C.DATA / "insumo_row_to_eid.csv", index=False)
    dup_eids = out[out.matches_published].eid.duplicated().sum()
    summary = {"record_rows": len(ins), "log_rows": len(log), "published_rows": len(pub),
               "rows_matching_published_output": int(match.sum()), "rows_not_matching": int(n - match.sum()) + (len(ins) - n),
               "published_eids_covered": int(out[out.matches_published].eid.nunique()), "duplicate_eids_among_matched": int(dup_eids),
               "alignment_check_literal_share": align, "alignment_gate_pass": bool(gate),
               "inputs": {"insumo_sha256": C.sha256(args.insumo), "published_sha256": C.sha256(args.published),
                          "extraction_log_sha256": C.sha256(args.extraction_log), "extraction_log_redistributed": False}}
    C.write_json(summary, C.DATA / "insumo_row_to_eid.build.json")
    print(summary)
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
