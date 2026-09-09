# Additional experiments: headline results

Generated 2026-09-09T02:52:21.722597+00:00 from `aditional_experiments/results/*/summary.json`. Commit `e75f027baa8cd4701291cb4303f6b38a2234ec9e`.

| Experiment | Reviewer / editor item | Gate |
|---|---|---|
| E9 semantic filter ablation | Editor E-1, R3-5, R2-3 | PASS |
| E5 grounding and leakage | R2-4, Editor E-2 | PASS |
| E2 model agreement | R2-7, R3-1 | PASS |
| E7 extraction cost | R2-5 | PASS |
| E6 corpus profile | R3-4, Editor E-2 | PASS |
| E4 k sensitivity | R3-2 | PASS (k = 5 arm; k = 10 extraction pending, needs API key) |

## E9. Semantic filter ablation: co-word CRS (no tau) vs. published CRS (tau = 0.40)

Same 52,946 documents, same keywords, same aggregation, same backbone rule (w >= 20) and Louvain seed; only the semantic constraint differs.

| | Co-word (no filter) | tau = 0.40 (paper) |
|---|---|---|
| Global nodes / edges | 56,635 / 377,144 | 56,635 / 109,022 |
| Global LCC fraction | 0.991 | 0.606 |
| Backbone nodes / edges | 658 / 1108 | 408 / 608 |
| Backbone components, LCC fraction | 1, 1.000 | 2, 0.995 |
| Backbone modularity (seed 42; mean +- sd over 10 seeds) | 0.3098 (0.3096 +- 0.0005) | 0.3560 (0.3559 +- 0.0003) |
| Backbone communities (seed 42) | 7 | 7 |
| Hub ('mathematics education') share of backbone edges | 52.4% | 51.5% |
| 'null' placeholder in backbone | yes (degree 3) | no |

* The filter removes 65.3% of the 529,000 co-occurrence instances and 71.1% of the 377,144 unique co-word edges; 500 discarded edges had support >= 20 (they would have been backbone edges).
* All 408 concepts of the published backbone are also in the co-word backbone (658 concepts); on the shared concepts the two community partitions agree with NMI = 0.703 and ARI = 0.834.
* 85.8% of the removed instances do not involve the hub: the filter mostly discards topical co-occurrences between semantically distant concepts (see `removed_edges_top60_excluding_hub.csv`), not hub noise.
* Stability of the backbone threshold: see `w_sweep.csv` (NMI between consecutive w and modularity plateau); w = 20 sits on the plateau where modularity is 0.35-0.36 and the LCC holds >= 99% of backbone nodes.

## E5. Grounding of generated keywords and leakage from the original Scopus keywords

264,586 keywords from 52,947 documents, classified against the 3,000-character record the model saw.

| Category (hierarchical) | Share |
|---|---|
| exact_original | 45.26% |
| verbatim_text | 40.18% |
| verbatim_metadata | 0.56% |
| soft_original | 7.26% |
| soft_text | 0.41% |
| ungrounded | 6.26% |
| null_token | 0.07% |

Non-hierarchical attribution (exact presence): in original keywords only 12.9%, in original keywords and in the text 32.3%, in the text only 40.2%, in neither 14.5%.

* Exact copies of a visible original keyword: 45.3% of keywords; exact or paraphrase (cos >= 0.70): 52.5%. Per document, the median share of exact copies is 40.0%; 4,817 documents had all keywords copied, 10,363 had none.
* Reverse direction: the model reproduced on average 30.9% of the visible original keywords (mean 9.5 visible originals per record versus 5 outputs).
* Grounded at some level: 93.7%; verbatim in the record: 86.0%; ungrounded (neither lexical nor cos >= 0.70 support): 6.26%, with median best-sentence similarity 0.522. See `ungrounded_top50.csv` for what these terms are.
* The most frequent 'ungrounded' terms are field-level labels rather than fabricated content: *mathematics education* (3,709), *stem education* (1,823), *science education* (492) together account for 36.4% of the 16,563 ungrounded keywords.
* Natural control (original keywords hidden by truncation; these are also the longest records, whose abstracts were cut as well):

| Stratum | Documents | exact_original | soft_original | verbatim_text | ungrounded | global sem. sim. (keywords vs. document) |
|---|---|---|---|---|---|---|
| fully_visible | 50,939 | 46.4% | 7.5% | 39.2% | 6.0% | 0.720 +- 0.091 |
| not_visible | 1,173 | 0.0% | 0.0% | 79.2% | 18.7% | 0.698 +- 0.094 |
| partially_visible | 835 | 39.1% | 5.5% | 47.4% | 7.5% | 0.703 +- 0.096 |

* Prompt adherence: 99.6% of documents returned exactly five keywords; 192 `null` placeholders (0.073%); 276 keywords equal to a forbidden generic term; 38 with non-ASCII characters; 24 containing Spanish function words.

## E2. Robustness of the representation to the language model

* Inspec, 8B vs 70B on the same 2,000 documents: identical sets 5.8%; Jaccard 0.395; soft F1 0.756; Soft Mean-Max 0.838; global similarity 0.899.
  Against the gold keyphrases (paired difference 70B minus 8B, 95% CI): soft_f1 +0.0158 [+0.0102, +0.0213]; soft_recall +0.0119 [+0.0069, +0.0170]; soft_precision +0.0201 [+0.0125, +0.0278]; global_sem_sim +0.0068 [+0.0038, +0.0098]. Full table: `inspec_gold_metrics_by_model.csv`.
* Mathematics-education sample (182 documents), 8B vs 70B: identical sets 4.9%; Jaccard 0.440; soft F1 0.785; global similarity 0.906.

## E7. Cost of the extraction stage (reconstructed)

* 53,130 calls; 33,975,989 input tokens (mean 639 per call, of which 317 are the fixed prompt) and about 1,399,261 output tokens (mean 26.3).
* At the OpenRouter list price fetched on 2026-09-09 (US$0.050/M input, US$0.080/M output) the whole corpus costs US$1.81 (US$0.034 per 1,000 documents); upper bound if every call used the four retries: US$7.24.
* Wall clock: at least 8.9 h from the 0.6 s inter-call delay alone; 24 h to 38 h for mean latencies of 1 s to 2 s.
* 2,010 records (3.78%) exceeded the 3,000-character truncation.

## E6. Corpus profile

* 53,130 records, 1996-2026, median year 2019; 67.2% published in 2016 or later.
* Abstract language (langdetect): English 99.6%; other languages: too_short 0.09%, es 0.09%, de 0.06%, fr 0.04%, pt 0.02%. Title in English: 93.5%.
* Original author/index keywords: mean 10.0 per record, 0 records without any.
* Generated keywords with non-ASCII characters: 0.014%; containing Spanish function words: 0.009%.
* 4,813 distinct sources; the ten most frequent account for 31.4% of records.

## E4. Sensitivity to k

* Status: k=5 arm only; run run_extraction.py (needs OPENROUTER_API_KEY) to add other k.
* k = 5 arm on the 5,000-document sample: 10,249 concepts, 13,247 edges, backbone w >= 20: 31 nodes / 35 edges (Q = 0.340); backbone w >= 5: 184 nodes / 244 edges (Q = 0.370, 7 communities).
