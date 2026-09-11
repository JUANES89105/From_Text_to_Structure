# Additional experiments for the IEEE Access revision (Access-2026-36745)

This directory extends `results/e1_baselines`, `results/e3_scalability` and `results/e8_topic_modeling`
with the experiments that the remaining reviewer and editor comments required. Every experiment
reuses the reference implementations of the published pipeline (`scripts/crs_reference.py`,
verbatim copies of notebooks 2, 3 and 5; `scripts/inspec_evaluation.py`, verbatim copy of notebook 6)
and starts by reproducing a published result as a gate. None of E2, E5, E6, E7 or E9 calls an LLM or
a paid API; E4 is the only experiment that needs new inference and is delivered ready to run.

| Code | Script | Reviewer / editor item | New LLM calls |
|---|---|---|---|
| E9 | `e9_semantic_filter_ablation.py` | Editor E-1 (co-word / bibliometric mapping), R3-5 (what the semantic constraint adds to the CRS of Gaona 2024), R2-3 (choice of tau and w) | none |
| E5 | `e5_grounding_leakage.py` | R2-4 (leakage of author/index keywords), Editor E-2 (hallucination control) | none |
| E2 | `e2_model_agreement.py` | R2-7 (single LLM configuration), R3-1 (variability of LLM outputs) | none |
| E7 | `e7_extraction_cost.py` | R2-5 (inference cost, processing time, hardware) | none |
| E6 | `e6_corpus_profile.py` | R3-4 (corpus representativeness, language bias), Editor E-2 (corpus selection) | none |
| E4 | `e4_k_sensitivity/run_extraction.py`, `analyze.py` | R3-2 (k = 5 is arbitrary) | 5,000 calls per additional k |
| E10 | `e10_scopus_keyword_baseline.py` | Editor E-4, R2-1, R3-5, R3-6 (what the LLM stage adds over a co-word network built from the Scopus keywords) | none |

Results live in `results/<code>/` with, for each experiment, `summary.json` (all reported numbers),
`validation.json` (reproduction gate and its outcome), `metadata.json` (environment, package versions,
SHA-256 of every input, timings) and the tables and figures listed in each section below.
`results/SUMMARY.md` collects the headline numbers.

## Data

Public inputs are in the repository root (`EID_KEYWORDS.xlsx`, `dataset_inspec.csv`,
`inspec_llama-3.1-8b-EN.csv`, `human_eval_M1_8b.csv`, the E3 sample lists) and in `data/`:

* `data/insumo_row_to_eid.csv`: index linking each line of the private record file (the input of
  notebook 1, which carries no identifier column) to the Scopus EID of the published output, with a
  flag `matches_published`. It is produced by `build_alignment.py` from a local extraction log that the
  authors discarded (it mixes in rows from a differently configured pass) and that is therefore not
  redistributed: the log is used only to recover the row order, a row is kept only when its stored
  list is identical to the published one, and the correspondence is verified independently (86.7% of
  published keywords occur verbatim in the record of the same row, against 9% in the neighbouring
  rows; `data/insumo_row_to_eid.build.json`). 53,045 rows link to the 52,947 published EIDs (98 EIDs
  occur on two rows with identical output; the loaders keep one row per EID); 85 rows have no
  published counterpart. All keywords used downstream come from `EID_KEYWORDS.xlsx`.
* `data/keywords_llm_llama-3.1-8b-EN.csv`: the 182 outputs of the human-validation sample.

Private inputs (Scopus records under Elsevier licence) are read from the directory given by the
environment variable `FTTS_PRIVATE_DIR` (default: the parent directory of the repository) and are
never copied into results: `corpus_insumo_DEFINITIVO.csv` (the 53,130 `insumo` records consumed by
notebook 1, one per line) and, for E2, the LLaMA 3.3 70B prediction files
`keywords_llm_llama-3.3-70b-EN.csv` and `inspec_llama-3.3-70b-EN.csv`. E9 and the k = 5 arm of E4
use only public inputs. E5, E6, E7 and part of E2 need the private records; their outputs contain
only aggregates and per-document scores keyed by EID.

## Reproduce

```sh
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r aditional_experiments/requirements-additional.txt
export FTTS_PRIVATE_DIR=/path/to/private/records      # only for E2 (full), E5, E6, E7, E4 extraction

.venv/bin/python aditional_experiments/build_alignment.py        # needs the private record file and the local extraction log
.venv/bin/python aditional_experiments/e9_semantic_filter_ablation.py
.venv/bin/python aditional_experiments/e5_grounding_leakage.py
.venv/bin/python aditional_experiments/e2_model_agreement.py
.venv/bin/python aditional_experiments/e7_extraction_cost.py
.venv/bin/python aditional_experiments/e6_corpus_profile.py
.venv/bin/python aditional_experiments/e10_scopus_keyword_baseline.py
.venv/bin/python aditional_experiments/e4_k_sensitivity/run_extraction.py --dry-run --k 10
OPENROUTER_API_KEY=... .venv/bin/python aditional_experiments/e4_k_sensitivity/run_extraction.py --k 10
.venv/bin/python aditional_experiments/e4_k_sensitivity/analyze.py
```

The embedding model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` at revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, the same revision used in E1 and E3; it is downloaded once
by `sentence-transformers`. All scripts run on CPU; seeds are fixed (Python, NumPy, Torch, Louvain).
Exit code 0 means the experiment ran and its gate passed.

## E9 - Semantic filter ablation (co-word CRS vs. tau = 0.40)

The original CRS (Gaona 2024) links every pair of keywords that co-occur in a document. The
published pipeline keeps a pair only when the cosine similarity of the keyword embeddings is at
least 0.40. Calling the unchanged reference constructor with `tau = -1` (cosine is bounded below by
-1) yields the pure co-word network on the same keywords; everything downstream (backbone
w >= 20, Louvain with seed 42) is identical. The difference isolates the semantic constraint.

Gate: the tau = 0.40 graph must reproduce the paper (56,635 nodes, 109,022 edges, 19,856 components,
LCC 34,316; backbone 408 nodes, 608 edges, modularity 0.356044, 7 communities).

Outputs: `comparison.csv` (both graphs and backbones side by side), `summary.json`
(`filter_effect`, `backbone_comparison` with NMI/ARI of the two community partitions and hub
dominance), `removed_edges_top60.csv` and `removed_edges_top60_excluding_hub.csv` (highest-support
co-occurrences that the filter discards), `kept_edges_top60.csv`, `similarity_distribution.csv`
(cosine of all co-occurring pairs, with the cumulative share below any threshold), `w_sweep.csv`
(backbone size, connectivity, modularity and NMI between consecutive thresholds for
w in {1,2,3,5,8,10,15,20,25,30,40,50}, for both graphs), `backbone_nodes_*.csv`,
`community_sizes_*.csv`, `nodes_only_in_coword_backbone.csv`, figures `w_sweep.*`,
`similarity_distribution.*`. `e9b_w_sweep_paper_thresholds.py` recomputes the sweep for the seven
thresholds reported in the manuscript's Table 7 (w in {1, 3, 5, 8, 10, 20, 30}) with the agreement between
consecutive partitions defined on that set (`w_sweep_paper_thresholds.csv`).

## E5 - Grounding and leakage

Each of the 265,491 generated keywords is compared with the record the model actually saw (the
`insumo` cut at 3,000 characters, as in notebook 1). Hierarchical categories: `null_token`,
`exact_original` (equal to a visible author/index keyword after normalisation), `verbatim_text`
(present in title or abstract), `verbatim_metadata` (present only in authors/year/source),
`soft_original` (cosine >= 0.70 with a visible original keyword), `soft_text` (cosine >= 0.70 with a
sentence of title + abstract), `ungrounded`. `attribution_2x2.csv` gives the non-hierarchical
cross-tabulation of exact presence in the original keywords and in the text.

Because the original keywords are the last field of the record, the truncation hides them completely
for 1,170 documents and partially for 840. `by_visibility_stratum.csv` reports every category,
the global semantic similarity between keyword set and document (the Table 2 metric) and the recall of
the visible original keywords for the three strata (`fully_visible`, `partially_visible`,
`not_visible`), which is a natural control for the leakage question without new inference.

Gate: row alignment between records and outputs (share of keywords found verbatim in their own
record at offset 0 versus offsets +1/-1). Other outputs: `category_shares.csv`, `per_document.csv`,
`per_keyword.csv.gz`, `ungrounded_top50.csv`, `ungrounded_sample150.csv`, `similarity_hist_*.csv`,
and the prompt-adherence audit inside `summary.json` (lists of exactly five items, `null`
placeholders, duplicates, non-ASCII residue, forbidden generic terms).

## E2 - Model agreement

1. Inspec (2,000 documents): agreement between LLaMA 3.1 8B and LLaMA 3.3 70B keyword sets per
   document (lexical Jaccard, soft precision/recall/F1 at 0.70, Soft Mean-Max, global similarity,
   share of identical sets) and both models against the gold keyphrases
   (`inspec_gold_metrics_by_model.csv`, with the paired per-document difference and its 95% CI in
   `summary.json`). Gate: the published 8B means are reproduced within 5e-5.
2. Mathematics-education validation sample (182 documents): the same agreement metrics.

The extraction used greedy decoding (temperature 0) and is deterministic in the authors' setup, so
no repeat-run comparison is included. The 70B prediction files are optional inputs; without them the
script reports the 8B gold metrics only and states that the 70B comparison was not run.

## E7 - Extraction cost

The exact notebook-1 chat prompt is rebuilt for all 53,130 records and tokenised with the Llama 3.1
tokenizer (chat-template control tokens included). The output `{"keywords": [...]}` is rebuilt from
the saved lists; on the 182 + 1,999 saved raw responses the reconstruction is token-exact
(`output_reconstruction_calibration.csv`). Cost uses the OpenRouter list price fetched at run time
(`pricing_snapshot.json`); the wall-clock lower bound follows from the 0.6 s inter-call delay of
notebook 1. `per_document_tokens.csv.gz` has per-record token counts.

## E6 - Corpus profile

Year distribution, top sources, language of title and abstract (`langdetect`, seeded), language by
period, number of original keywords per record, and an audit of non-English residue in the generated
keywords (non-ASCII characters; Spanish function words) overall and by abstract language.

## E4 - Sensitivity to the number of keywords k

`run_extraction.py` re-extracts the 5,000 documents of E3 sample `n5000_r1_s42` with the notebook-1
prompt and `EXACTLY k` terms (default k = 10), same model, decoding, truncation and retry policy,
logging the usage reported by the API. `--dry-run` writes `dry_run_manifest.json` and
`sample_mapping.csv` without any call. `analyze.py` builds the CRS (tau = 0.40) for every arm and
compares backbones at w >= 20 and w >= 5 (NMI/ARI of Louvain partitions on shared concepts, node
overlap) and classifies the extra keywords as exact or soft restatements of the k = 5 set or as new
concepts. Gate: the k = 5 arm reproduces the E3 record of the same sample exactly.

Status: the dry run and the k = 5 arm are included; the k = 10 extraction requires an OpenRouter key
(estimated 5,000 calls, about US$0.18 at list price, at least 50 minutes) and is the only pending
computation of this directory.

## E10 - Baseline: CRS built from the Scopus author/index keywords

The grounding audit (E5) showed that 45% of the LLM keywords copy a record keyword and 86% appear
verbatim in the record, and every record carries author or index keywords (E6). E10 asks what the LLM
stage contributes at the graph level by building the same CRS from the Scopus keywords of the same
52,946 documents with the same constructor, tau = 0.40, w >= 20 and Louvain seed. Three arms:
`llm_k5` (published keywords; gate: exact reproduction of the published CRS), `scopus_all` (every
author/index keyword of the record, normalised as in notebook 2) and `scopus_first5` (the first five
keywords in record order, which holds the cardinality fixed). The unconstrained co-word network of
`scopus_all` is also built, since that is the classical bibliometric map. Arms are compared on
vocabulary size and lexical fragmentation (share of terms that collapse under punctuation and plural
normalisation), isolated nodes, connectivity, backbone size and modularity, hub dominance, non-English
residue, overlap of backbone concepts, and agreement of community partitions on shared concepts and at
the document level (majority community of a document's backbone keywords, as in E8).

Outputs: `arms_comparison.csv`, `summary.json`, `vocabulary_fragmentation.csv`, per-arm
`backbone_nodes_*.csv`, `community_sizes_*.csv`, `backbone_top_edges_*.csv`,
`llm_backbone_concepts_not_in_*_backbone.csv`, `document_communities.csv` (EID and community label
per arm, no text), `validation.json`, `metadata.json`.
