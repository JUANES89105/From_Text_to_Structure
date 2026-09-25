# Additional experiments (not reported in the manuscript)

The two analyses in this directory were run during the IEEE Access revision (Access-2026-36745) with the same reference implementation of the published pipeline (`scripts/crs_reference.py`, `scripts/common.py`) and the same reproduction gates as the notebooks in the repository root. Their numbers are **not** reported in the manuscript; they are provided for completeness and for readers who want to extend the comparison.

| Notebook | Question | Inputs | Output folder |
|---|---|---|---|
| `A1. SCOPUS_KEYWORD_BASELINE.ipynb` | What does the LLM extraction stage contribute at the graph level, compared with the author and index keywords that Scopus already provides for the same records? | `EID_KEYWORDS.xlsx`, `data/insumo_row_to_eid.csv`, private record file via `FTTS_PRIVATE_DIR` | `results/e10_scopus_keyword_baseline/` |
| `A2. K_SENSITIVITY.ipynb` | How does the graph change with the number of keywords per document (k)? | `EID_KEYWORDS.xlsx`; the k = 10 arm needs new inference | `results/e4_k_sensitivity/` |

## A1. CRS built from the Scopus author and index keywords

Four arms are built with the same constructor, tau = 0.40, backbone w >= 20 and Louvain seed 42, on the same 52,946 documents:

- `llm_k5`: the published LLM keywords (gate: exact reproduction of the published CRS);
- `scopus_all`: every author and index keyword of the record, normalised as in notebook 2;
- `scopus_first5`: the first five keywords in record order (fixed cardinality);
- `scopus_all_coword`: the unconstrained co-word network of `scopus_all`, the classical bibliometric map.

Arms are compared on vocabulary size and lexical fragmentation, isolated nodes, connectivity, backbone size and modularity, hub dominance, non-English residue, overlap of backbone concepts, and agreement of community partitions on shared concepts and at the document level. Outputs: `arms_comparison.csv`, `summary.json`, `vocabulary_fragmentation.csv`, per-arm `backbone_nodes_*.csv`, `community_sizes_*.csv`, `backbone_top_edges_*.csv`, `llm_backbone_concepts_not_in_*_backbone.csv`, `document_communities.csv` (EID and community label per arm, no text), `validation.json`, `metadata.json`.

## A2. Sensitivity to the number of keywords k

`e4_k_sensitivity/run_extraction.py` re-extracts the 5,000 documents of the corpus-size sample `n5000_r1_s42` (notebook 13) with the notebook-1 prompt and `EXACTLY k` terms, same model, decoding, truncation and retry policy, logging the usage reported by the API. `--dry-run` writes `dry_run_manifest.json` and `sample_mapping.csv` without any call. The notebook rebuilds the sample and the dry-run manifest, runs the k = 5 arm (gate: reproduces the notebook-13 record of the same sample exactly) and documents the comparison that `analyze` performs once a k = 10 arm exists (CRS at tau = 0.40 for each arm; backbones at w >= 20 and w >= 5; NMI and ARI of Louvain partitions on shared concepts; classification of the extra keywords as exact or soft restatements of the k = 5 set or as new concepts).

Status: the k = 10 extraction has not been run. It requires an OpenRouter key (about 5,000 calls, about US$0.20 at list price, at least 50 minutes):

```sh
OPENROUTER_API_KEY=... .venv/bin/python aditional_experiments/e4_k_sensitivity/run_extraction.py --k 10
```

## Private data

Private inputs (Scopus records under Elsevier licence) are read from the directory given by `FTTS_PRIVATE_DIR` (default: the parent directory of the repository) and are never copied into results.
