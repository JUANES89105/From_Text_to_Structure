# E3 — CRS structure and computational cost by corpus size

This experiment uses existing extracted keywords only. It does not invoke an LLM, use a paid API, change any original notebook/dataset, or implement E2/E4–E8. Consult `metadata.json` and `validation.json` for execution status; results must not be interpreted unless the full-corpus reproduction gate passes.

## Audit and reference reproduction

`EID_KEYWORDS.xlsx` contains 52,947 data rows, with columns `EID_o_identificador` and `palabras_clave`, and unique nonmissing document identifiers. One row (zero-based source row 22,794) contains an empty keyword list. Notebook 2 excludes empty parsed lists, leaving **52,946 analyzed documents**, the E3 `N_total`. `corpus_audit.csv` records every source row and its inclusion/exclusion explicitly. No original file is changed.

The original parsing uses `ast.literal_eval`, accepts lists, and returns an empty list for null, empty, malformed, or non-list values. It filters empty parsed lists before normalization. Normalization keeps nonempty strings, applies `strip().lower()`, then `sorted(set(...))` within each document. It does not remove punctuation, stem, collapse internal whitespace, or specially remove the literal `null`. In the supplied Excel, parsed and cleaned list lengths agree for every row.

The undirected graph has one node per normalized phrase, with `doc_freq` increased once per supporting document. The model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Normalized phrase embeddings are compared using `sklearn.metrics.pairwise.cosine_similarity` within each document. Only co-occurring pairs with cosine **>=0.40** add edges. Edge `weight` counts supporting documents; cosine is kept separately as `sim_sum`, `sim_count`, and `sim_mean`.

The backbone retains edges with **weight>=20**, then removes isolated nodes. Global graph isolates remain. LCC is the largest connected component by node count, with node fraction calculated using the node count of the relevant graph. For empty/edgeless graphs, the notebook 5 convention reports LCC size and fraction as zero. Louvain is `python-louvain.best_partition(backbone, weight="weight", random_state=42)`, resolution 1.0. Modularity is weighted and calculated on the backbone, not the full graph.

`scripts/crs_reference.py` contains unchanged copies of `parse_keywords` (notebook 2), `build_backbone` (notebook 3), and `build_crs_for_tau` (notebook 5). Their ASTs are compared to the notebooks at startup. The full corpus is run **first and once**. Required reference values include:

| Quantity | Stored reference |
|---|---:|
| Analyzed documents | 52,946 |
| Global nodes / edges | 56,635 / 109,022 |
| Global components | 19,856 |
| Global LCC nodes / edges | 34,316 / 106,297 |
| Backbone nodes / edges | 408 / 608 |
| Backbone components | 2 |
| Backbone LCC nodes / edges | 406 / 607 |
| Backbone Louvain modularity | 0.3560443171946043 |
| Backbone communities | 7 |

Reference sources are saved outputs in notebooks 2–5. Counts must match exactly, modularity within absolute 1e-6, other full-precision derived values within 1e-10. The script stops before all subsamples if the gate fails. No tolerance or threshold is selected from E3 performance.

## Sampling and fixed settings

Sizes: 500, 1,000, 5,000, 10,000, 25,000, and 52,946. Each smaller size has ten replicates, with seeds 42–51. Each sample is drawn directly without replacement from the complete analyzed corpus using `default_rng(SeedSequence([seed,n])).choice(N,n,replace=False)`. The size-specific entropy component avoids shared-permutation nested samples while retaining the requested replicate seeds. Selected positions are sorted to preserve original relative document order, including graph insertion order. Per-run sample CSVs preserve both corpus positions and original Excel positions/identifiers.

Model revision: `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, already cached from the project. CPU float32, four Torch threads, fixed Python/NumPy/Torch seed 42, deterministic Torch algorithms, embedding batch size 32 (reference default), normalized embeddings. Hugging Face and Transformers are offline; `local_files_only=True` prevents silent model downloads.

As in notebook 5, each run embeds its sorted unique phrase vocabulary once. Each run recomputes its own embeddings; no full-corpus embedding cache is reused across sample sizes or replicates. Graph construction still computes every document's pairwise cosine matrix and aggregates support counts using the unchanged reference function. This measures the vocabulary-cached implementation in notebook 5; timings are not presented as equivalent to notebook 2's repeated per-document inference.

## Metrics and scope

- Full graph: nodes, edges, density, components, LCC nodes/edges/fraction, mean degree and mean weighted degree.
- Backbone: nodes, edges, density, components, LCC nodes/edges/fraction, mean degree and weighted degree, weighted Louvain modularity and community count.
- `modularity` and `n_communities` refer **only to the backbone**, consistent with notebooks 4/5. `mean_degree` and `mean_weighted_degree` refer to the full graph; explicitly prefixed counterparts reproduce the original backbone scope.
- `clustering_coefficient` is an additional descriptive metric: unweighted `nx.average_clustering(full_graph, weight=None, count_zeros=True)`. It was not implemented in notebooks 2–5 and is not claimed as an original-reference metric. It does not change the graph.
- An empty backbone retains zero node/edge/component/community counts and LCC fraction zero, as in notebook 5. Modularity and mean degrees are undefined (CSV NaN), not imputed as zero. The absolute threshold is never lowered.
- Existing but unrequested expensive metrics (exact betweenness, shortest paths, connectivity, etc.) are not repeated for E3. Consequently E3 timings cover the declared metrics, not every operation in notebook 4.

## Timings, memory, and summary statistics

`runtime_total_seconds` covers per-run vocabulary preparation, embedding, graph construction, backbone creation and metrics. Embedding, construction and metrics are timed separately. Shared Excel/model loading is recorded separately in metadata; sample CSV I/O and plotting are excluded. No historical LLM extraction cost is inferred.

Memory is approximate process RSS sampled every 50ms, with per-run baseline, peak, and peak-minus-baseline. The shared model and allocator-retained memory are included; these are not isolated allocation peaks. Wall-clock results are measurements on an active host, not controlled cross-hardware benchmarks. macOS initially had dependency files marked `dataless`; environment recovery belongs to setup, not the timed graph runs. No new packages were required.

`summary.csv` is long-form: size, metric, number of runs, number of valid values, mean, sample SD (`ddof=1`), minimum, maximum, and CV where the mean is positive. The singleton full corpus has undefined SD/CV; no artificial zero SD is inserted. Undefined metrics are identified through `n_valid`. `runtime_summary.csv` contains the timing subset.

## Reproduction and outputs

```sh
.venv/bin/python -B -m unittest discover -s scripts -p 'test_e3_*.py' -v
.venv/bin/python -B -u scripts/e3_scalability.py
```

Optional `--reference-only` runs just the required full-corpus gate. `--resume` continues completed, checksummed runs only with identical code, configuration, protected inputs, and package versions. Do not launch two concurrent E3 processes into the same results directory.

Required outputs: `runs.csv`, `summary.csv`, `configuration.json`, `metadata.json`, `validation.json`, and this README. Additional outputs: `corpus_audit.csv`, `samples/*.csv`, `runtime_summary.csv`, and five separate PNG/PDF figure pairs under `figures/` (runtime, full nodes/edges, LCC fractions, backbone modularity, and backbone nodes/edges). Figures show mean ± one sample SD for replicated sizes; the complete corpus has a single point without an SD bar.

Interpret computational cost separately from structural stability. A high LCC fraction in a tiny surviving backbone does not mean the full corpus is structurally represented. Changes in community count/modularity do not establish identical community membership. Neither completion of the script nor within-corpus subsampling demonstrates generalization to other corpora.
