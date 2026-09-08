# E1 — Inspec extraction baseline comparison

This experiment compares saved LLaMA 3.1 8B predictions with TF-IDF, YAKE, and KeyBERT on the same 2,000 Inspec documents. See `comparison.md` for means and sample SDs, `summary.csv` for full precision, and `llm_validation.json` for independent reproduction of notebook 6's saved means.

## Reproduce

From the repository root on `ieee-revision-experiments`:

```sh
.venv/bin/python -m pip install -r requirements-e1.txt
.venv/bin/python -B -m unittest discover -s scripts -p 'test_*.py' -v
.venv/bin/python -B -u scripts/e1_baseline_extraction.py --validate-only
.venv/bin/python -B -u scripts/e1_baseline_extraction.py --reuse-validated-llm
.venv/bin/python -B scripts/e1_baseline_extraction.py --verify-results
```

Alternatively, omit both command-line flags to reproduce the LLM scores and then run all baselines in one invocation. A mismatch larger than 0.00005 in any LLM mean stops execution before baseline extraction. Cached validation reuse requires identical protected inputs and package versions, checksummed evaluator/prediction/metric artifacts, unchanged reference syntax trees, and a successful score check.

The script requires the model revision below to be cached locally. It enables Hugging Face/Transformers offline mode, uses `local_files_only=True`, and never invokes an LLM or paid API. If the cache is absent, it fails rather than downloading silently. Rerunning writes the generated files in this results directory; original datasets and notebooks are never written.

## Fixed input and evaluation protocol

- Exactly 2,000 records from `dataset_inspec.csv`, preserving positional `row_abs` and string `doc_id`.
- Extraction functions receive a separate table containing IDs, `insumo`, character counts, and input hashes. They receive no `keywords_gt` column. No gold-based parameter search or validation tuning is performed.
- Use only `insumo[:3000]`, matching notebook 1's character truncation. This affects one document: `row_abs=164`, `doc_id=1150`, whose original input is 3,100 characters. The input manifest records the truncated text hashes and lengths.
- Reuse the saved `keywords_llm` strings and their list order unchanged. The prediction file joins by `row_abs`; no document ID or input hash in that historical file independently proves its generation provenance.
- `scripts/inspec_evaluation.py` copies notebook 6's cells 3–4 verbatim, with an AST equivalence check at runtime. No notebook was refactored in place.
- Evaluator: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, CPU float32, batch size 128, `normalize_embeddings=False`; the reference scoring functions normalize for cosine themselves.
- Semantic evaluation threshold is **0.70**. Neither CRS edge threshold 0.40 nor backbone weight 20 is used in this experiment.
- Normalize phrases to lowercase, collapse whitespace, replace non-ASCII-alphanumeric/non-space characters with spaces, remove empty/`null` phrases, and deduplicate preserving first occurrence order. No stemming or lemmatization.
- Lexical Jaccard operates on whole normalized phrase sets. Soft precision and recall measure many-to-one best-match coverage at cosine >=0.70; soft F1 is their per-document harmonic mean. Soft mean-max averages both directional mean maxima without thresholding. Global semantic similarity embeds ordered phrase lists concatenated with `" ; "`.
- Both lists empty gives 1.0; one list empty gives 0.0 for every metric. All documents remain in the averages. Report document means and sample SD (`ddof=1`), including the mean of per-document F1 values.

## Baseline configurations

These settings were fixed before baseline scoring, without inspecting gold-based baseline performance. The common maximum phrase length of four accommodates the reference prompt's short multiword phrases; unigrams remain eligible. No claim is made that these are optimized configurations.

### TF-IDF

- scikit-learn `TfidfVectorizer`, fit on all 2,000 truncated inputs only (a transductive corpus fit).
- Word n-grams 1–4; lowercase; scikit-learn's bundled English stopwords; token pattern `(?u)\b\w\w+\b` (Unicode word tokens of at least two characters); no accent stripping, stemming, or custom tokenizer.
- `min_df=1`, `max_df=1.0`, `max_features=None`, `binary=False`, `sublinear_tf=False`, `use_idf=True`, `smooth_idf=True`, `norm="l2"`, float64.
- Rank each document's nonzero features by descending TF-IDF, then lexicographically by phrase on exact ties. Return the first five, or all available if fewer. Features are unique; no fuzzy deduplication or diversity reranking.

### YAKE

- YAKE 0.7.3, `lan="en"`, `n=4`, `dedup_lim=0.9`, `dedup_func="seqm"`, `window_size=1`, `top=5`, `features=None`, `lemmatize=False`.
- Bundled English stopwords and native tokenization/candidate validity rules. The exact stopword set is saved in `yake_stopwords.json`.
- Native ascending score order; tied scores retain candidate insertion order. Native similarity deduplication rejects a candidate when similarity to a selected candidate is greater than 0.9. Return at most five surviving candidates without padding.
- Exceptions caught internally by YAKE are captured from its warning logs and recorded as failures; other per-document warnings are retained.

### KeyBERT

- KeyBERT 0.9.0 using the same named/revision-pinned multilingual MiniLM model as evaluation. No LLM backend or seed keywords.
- `CountVectorizer` candidates: word n-grams 1–4, English stopwords, `min_df=1`, `max_df=1.0`, lowercase, token pattern `(?u)\b\w\w+\b`, no accent stripping/custom tokenizer; unique features.
- Vocabulary constructed from all truncated inputs; each document is ranked only against candidates present in that document. Candidate embeddings are shared across documents for efficiency.
- Native cosine ranking, `top_n=5`, `use_mmr=False`, `use_maxsum=False`. `diversity=0.5` and `nr_candidates=20` are passed but inactive because both rerankers are disabled.
- Preserve native descending NumPy argsort order (including its tie behavior). Returned scores are rounded to four decimals by KeyBERT; selection uses unrounded similarities. No extra fuzzy deduplication.
- The embedding backend only batches standard `SentenceTransformer.encode` calls: up to 4,096 texts per outer chunk, inference batches of 128, no normalized output vectors. Candidate ordering is retained. No training occurs.

All three baselines target five native ranked phrases. Actual raw and evaluator-cleaned counts are saved separately. Post-cleaning collisions are reported without padding or selecting replacement candidates. Saved LLM predictions are never capped, padded, or otherwise changed.

## Artifacts

| File | Contents |
|---|---|
| `predictions_<method>.csv` | IDs, ordered phrases, raw/clean counts, extraction status and timing; baseline scores and warnings where applicable |
| `predictions_all.csv` | All four methods, 8,000 rows |
| `metrics_<method>.csv`, `metrics_all.csv` | All six per-document scores, IDs, and gold/prediction phrase counts |
| `summary.csv`, `comparison.md` | Six means and sample SDs for each method |
| `output_length_distributions.csv` | Raw and cleaned phrase-count distributions |
| `quality_counts.csv` | Empty lists, deviations from five, missing predictions/responses, reported failures, warnings, missing metric values |
| `runtimes.csv` | Extraction and evaluation wall-clock seconds by method |
| `configuration.json`, `metadata.json` | Fixed configurations, model revision, seeds, environment, timing details, truncation, provenance |
| `input_manifest.csv` | IDs, original/used input lengths, and truncated-input SHA-256 hashes |
| `llm_validation.json` | Expected versus reproduced means, tolerance, pass/fail, validated artifact hashes |
| `artifact_validation.json` | Post-run verification of all row identities/counts, lexical scores, summaries, input hashes and protected artifacts |
| `protected_artifact_hashes.json` | SHA-256 checksums of all seven notebooks and four original data files |
| `package_versions.json` | All installed distribution versions in the measured environment |
| `sklearn_english_stopwords.json`, `yake_stopwords.json` | Exact stopword sets used |
| `yake_repeatability_check.json` | Independent-process check of all 2,000 ordered YAKE phrase lists and scores |
| `keybert_native_check.json` | Fixed rows 0, 999 and 1999 compared against the unmodified native SentenceTransformer KeyBERT backend; identical ordered phrases and scores |
| `token_length_diagnostics.json` | Untruncated tokenizer lengths for document inputs and gold concatenations; no change to inference |

The `keywords` field for saved LLM predictions retains the original `keywords_llm` content. Historical `raw_response` content remains in the original CSV; missing-response indicators are carried into the result records. Historical failure causes are unknown and are not inferred from empty lists.

## Timing and reproducibility limits

- Fixed Python, NumPy, and Torch seed 42; deterministic Torch algorithms; four CPU threads. CPU platform, logical core count, package versions, model revision/dtype, and maximum sequence length are recorded. CPU-brand and physical-RAM queries were denied by the local sandbox and are explicitly unavailable.
- TF-IDF extraction time includes corpus fitting and ranking; per-document times cover ranking only. YAKE times cover extraction (excluding extractor initialization). KeyBERT time includes vocabulary construction, document/candidate embedding and ranking; per-document times are unavailable because extraction is batched. Embedding-stage times are recorded separately.
- Evaluation times are separate from extraction. Model loading is recorded once and is not charged repeatedly to methods. Historical LLM extraction time, tokens, costs, retries, and hardware are unavailable; no values are imputed.
- These are single-run wall-clock measurements on an active host, not isolated performance benchmarks. Separate validation processes ran during parts of the experiment and can affect timings. The independent YAKE repeatability check is separate from the reported method extraction run.
- YAKE 0.7.3 and KeyBERT 0.9.0 plus required dependencies were installed in the existing `.venv`. `requirements-e1.txt` adds those two pinned packages, pins the already installed scikit-learn 1.9.0, and includes the existing `requirements.txt`, which was not modified. `package_versions.json` records the complete environment; `pip check` found no broken requirements.
- The full reproduction runs inference from the original function semantics without an embedding cache or altered metric definitions. Notebook outputs alone were not treated as reproduced evidence.

## Methodological concerns

- Saved LLM output lengths vary, including one empty output and one six-phrase output after cleaning. Preserve them and report count distributions; this is a target-five comparison, not an artificially equalized output budget.
- MiniLM has a built-in maximum sequence length of 128 tokens. That applies to KeyBERT document embeddings and to concatenated phrase-list embeddings in the unchanged evaluator. After the 3,000-character limit, 1,429 input documents still exceed 128 tokens; 110 gold phrase-list concatenations also exceed 128 tokens. Counts include special tokens. The character limit does not override the model's token limit. The diagnostic tokenizer warning for sequences above 512 tokens was produced while counting untruncated tokens, not during model inference; actual inference retains the reference 128-token limit.
- Using the same embedding model for KeyBERT extraction and semantic evaluation creates model alignment that may benefit KeyBERT on semantic metrics. Lexical Jaccard is also reported; no universal superiority claim follows from one benchmark.
- TF-IDF/KeyBERT remove stopwords before forming n-grams, so adjacent retained words can span removed words. Their candidates need not be contiguous substrings of the original input. YAKE has its own boundary/stopword rules.
- The model can generate phrases absent from the input; the three baselines use input-derived candidates. Interpret differences in light of these candidate-space differences and the fixed, non-tuned settings.
- The gold list has more than five phrases for many documents. Coverage-style recall depends on the output budget and permits multiple gold phrases to match one prediction. Global similarity also depends on retained phrase order.
- These are descriptive means/SDs, not significance tests or evidence of cross-corpus generalization. No CRS graph reconstruction, E2–E8, or new LLM extraction was performed.

## Completed-run checks and observations

All six reproduced LLM means match the saved four-decimal reference values within the predeclared absolute tolerance of 0.00005. Ten unit/adapter tests passed. Final artifact verification checked 8,000 prediction rows, 8,000 metric rows and 48,000 finite metric values, including row identity, unchanged saved LLM strings, independently recomputed lexical scores, means/sample SDs, input manifests and protected-file hashes.

All baseline methods returned five raw phrases for every document, with zero reported extraction failures and no missing metric values. One YAKE list becomes four phrases after the reference cleaning step. The LLM retains its one empty prediction, 19 cleaned four-phrase lists, 1,979 five-phrase lists and one six-phrase list. One historical raw response is missing; its cause remains unknown.

The LLM has the highest mean soft F1 (0.5652) in this fixed comparison. KeyBERT has slightly higher mean soft precision (0.7941 versus 0.7939), but lower mean recall (0.2026) and F1 (0.3034). Its fixed native configuration can return closely related or nested phrases: for row 0, examples include `separate account industry`, `separate account industry supplying`, and `shaking separate account industry`. No diversity reranking or gold-based retuning was introduced in response to these results. An independent native-backend check on preselected rows 0, 999 and 1999 reproduced the ordered phrases and rounded scores exactly.
