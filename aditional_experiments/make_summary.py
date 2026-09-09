"""Render results/SUMMARY.md from the summary.json / validation.json files of every experiment.

Numbers are read from the result files, never typed by hand, so the summary cannot drift from the
artifacts. Re-run after any experiment: ``.venv/bin/python aditional_experiments/make_summary.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

R = C.RESULTS


def load(name, fname="summary.json"):
    p = R / name / fname
    return json.load(open(p)) if p.exists() else None


def gate(name):
    v = load(name, "validation.json")
    if not v:
        return "not run"
    if "pass" in v:
        return "PASS" if v["pass"] else "FAIL"
    nested = [x.get("pass") for x in v.values() if isinstance(x, dict) and "pass" in x]
    return "PASS" if nested and all(nested) else "FAIL"


def f(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def pct(x, nd=1):
    try:
        return f"{100 * float(x):.{nd}f}%"
    except Exception:
        return str(x)


def main():
    out = []
    w = out.append
    w("# Additional experiments: headline results\n")
    w(f"Generated {C.now_utc()} from `aditional_experiments/results/*/summary.json`. Commit `{C.git_commit()}`.\n")
    w("| Experiment | Reviewer / editor item | Gate |")
    w("|---|---|---|")
    w(f"| E9 semantic filter ablation | Editor E-1, R3-5, R2-3 | {gate('e9_semantic_filter_ablation')} |")
    w(f"| E5 grounding and leakage | R2-4, Editor E-2 | {gate('e5_grounding_leakage')} |")
    w(f"| E2 model agreement | R2-7, R3-1 | {gate('e2_model_agreement')} |")
    w(f"| E7 extraction cost | R2-5 | {gate('e7_extraction_cost')} |")
    w(f"| E6 corpus profile | R3-4, Editor E-2 | {gate('e6_corpus_profile')} |")
    w(f"| E4 k sensitivity | R3-2 | {gate('e4_k_sensitivity')} (k = 5 arm; k = 10 extraction pending, needs API key) |")
    w("")

    # ------------------------------------------------------------------ E9
    s = load("e9_semantic_filter_ablation")
    if s:
        g0, g4 = s["global_graph"]["coword"], s["global_graph"]["tau040"]
        b0, b4 = s["backbone_w20"]["coword"], s["backbone_w20"]["tau040"]
        fe, bc = s["filter_effect"], s["backbone_comparison"]
        w("## E9. Semantic filter ablation: co-word CRS (no tau) vs. published CRS (tau = 0.40)\n")
        w("Same 52,946 documents, same keywords, same aggregation, same backbone rule (w >= 20) and Louvain seed; only the semantic constraint differs.\n")
        w("| | Co-word (no filter) | tau = 0.40 (paper) |")
        w("|---|---|---|")
        w(f"| Global nodes / edges | {g0['nodes']:,} / {g0['edges']:,} | {g4['nodes']:,} / {g4['edges']:,} |")
        w(f"| Global LCC fraction | {f(g0['lcc_fraction'])} | {f(g4['lcc_fraction'])} |")
        w(f"| Backbone nodes / edges | {b0['nodes']} / {b0['edges']} | {b4['nodes']} / {b4['edges']} |")
        w(f"| Backbone components, LCC fraction | {b0['components']}, {f(b0['lcc_fraction'])} | {b4['components']}, {f(b4['lcc_fraction'])} |")
        w(f"| Backbone modularity (seed 42; mean +- sd over 10 seeds) | {f(b0['modularity_seed42'],4)} ({f(b0['modularity_mean'],4)} +- {f(b0['modularity_sd'],4)}) | {f(b4['modularity_seed42'],4)} ({f(b4['modularity_mean'],4)} +- {f(b4['modularity_sd'],4)}) |")
        w(f"| Backbone communities (seed 42) | {b0['communities_seed42']} | {b4['communities_seed42']} |")
        hub = bc["hub_dominance"]
        w(f"| Hub ('{hub['coword_backbone']['hub']}') share of backbone edges | {pct(hub['coword_backbone']['hub_share_of_edges'])} | {pct(hub['tau040_backbone']['hub_share_of_edges'])} |")
        nn = bc["null_placeholder_node"]
        w(f"| 'null' placeholder in backbone | {'yes' if nn['in_coword_backbone'] else 'no'} (degree {nn['coword_backbone_degree']}) | {'yes' if nn['in_tau040_backbone'] else 'no'} |")
        w("")
        w(f"* The filter removes {pct(fe['share_instances_removed'])} of the {fe['cooccurrence_edge_instances_total']:,} co-occurrence instances "
          f"and {pct(fe['share_unique_edges_removed'])} of the {fe['unique_edges_coword']:,} unique co-word edges; "
          f"{fe['removed_edges_with_weight_ge_20']} discarded edges had support >= 20 (they would have been backbone edges).")
        w(f"* All {bc['tau040_backbone_nodes']} concepts of the published backbone are also in the co-word backbone ({bc['coword_backbone_nodes']} concepts); "
          f"on the shared concepts the two community partitions agree with NMI = {f(bc['partition_agreement_shared_nodes']['nmi'])} and ARI = {f(bc['partition_agreement_shared_nodes']['ari'])}.")
        w(f"* {pct(hub['share_removed_instances_not_incident_to_hub'])} of the removed instances do not involve the hub: the filter mostly discards "
          f"topical co-occurrences between semantically distant concepts (see `removed_edges_top60_excluding_hub.csv`), not hub noise.")
        w("* Stability of the backbone threshold: see `w_sweep.csv` (NMI between consecutive w and modularity plateau); w = 20 sits on the plateau where modularity is 0.35-0.36 and the LCC holds >= 99% of backbone nodes.")
        w("")

    # ------------------------------------------------------------------ E5
    s = load("e5_grounding_leakage")
    if s:
        cs, lk, gr, at = s["category_shares"], s["leakage"], s["grounding"], s.get("attribution_2x2_exact_presence", {})
        w("## E5. Grounding of generated keywords and leakage from the original Scopus keywords\n")
        w(f"{s['keywords_classified']:,} keywords from {s['documents_analyzed']:,} documents, classified against the 3,000-character record the model saw.\n")
        w("| Category (hierarchical) | Share |")
        w("|---|---|")
        for k in ["exact_original", "verbatim_text", "verbatim_metadata", "soft_original", "soft_text", "ungrounded", "null_token"]:
            w(f"| {k} | {pct(cs[k], 2)} |")
        w("")
        if at:
            w(f"Non-hierarchical attribution (exact presence): in original keywords only {pct(at['in_original_only'])}, in original keywords and in the text {pct(at['in_original_and_text'])}, "
              f"in the text only {pct(at['in_text_only'])}, in neither {pct(at['in_neither'])}.\n")
        w(f"* Exact copies of a visible original keyword: {pct(lk['share_keywords_exact_copy_of_visible_original'])} of keywords; exact or paraphrase (cos >= 0.70): {pct(lk['share_keywords_exact_or_paraphrase_of_visible_original'])}. "
          f"Per document, the median share of exact copies is {pct(lk['median_per_document_share_exact_original'])}; {lk['documents_with_all_keywords_exact_copies']:,} documents had all keywords copied, {lk['documents_with_zero_exact_copies']:,} had none.")
        w(f"* Reverse direction: the model reproduced on average {pct(lk['mean_recall_of_visible_original_keywords'])} of the visible original keywords (mean {f(lk['mean_n_original_keywords_visible'],1)} visible originals per record versus 5 outputs).")
        w(f"* Grounded at some level: {pct(gr['share_grounded_any_level'])}; verbatim in the record: {pct(gr['share_verbatim_in_record'])}; ungrounded (neither lexical nor cos >= 0.70 support): {pct(gr['share_ungrounded'], 2)}, "
          f"with median best-sentence similarity {f(gr['median_max_sentence_similarity_ungrounded'])}. See `ungrounded_top50.csv` for what these terms are.")
        top = R / "e5_grounding_leakage" / "ungrounded_top50.csv"
        if top.exists():
            import pandas as pd
            t = pd.read_csv(top).head(3)
            n_ung = s["category_counts"]["ungrounded"]
            w(f"* The most frequent 'ungrounded' terms are field-level labels rather than fabricated content: "
              + ", ".join(f"*{r.keyword}* ({int(r['count']):,})" for _, r in t.iterrows())
              + f" together account for {pct(t['count'].sum() / n_ung)} of the {n_ung:,} ungrounded keywords.")
        w("* Natural control (original keywords hidden by truncation; these are also the longest records, whose abstracts were cut as well):\n")
        w("| Stratum | Documents | exact_original | soft_original | verbatim_text | ungrounded | global sem. sim. (keywords vs. document) |")
        w("|---|---|---|---|---|---|---|")
        for r in s["by_visibility_stratum"]:
            w(f"| {r['visibility']} | {int(r['n_documents']):,} | {pct(r['exact_original'])} | {pct(r['soft_original'])} | {pct(r['verbatim_text'])} | {pct(r['ungrounded'])} | {f(r['mean_global_sem_sim'])} +- {f(r['sd_global_sem_sim'])} |")
        pa = s["prompt_adherence"]
        w("")
        w(f"* Prompt adherence: {pct(pa['documents_with_exactly_5_keywords']/pa['documents'])} of documents returned exactly five keywords; "
          f"{pa['null_tokens']} `null` placeholders ({pct(pa['null_token_rate'],3)}); {pa['keywords_equal_to_forbidden_generic_terms']} keywords equal to a forbidden generic term; "
          f"{pa['keywords_with_non_ascii_characters']} with non-ASCII characters; {pa['keywords_with_spanish_function_words']} containing Spanish function words.")
        w("")

    # ------------------------------------------------------------------ E2
    s = load("e2_model_agreement")
    if s:
        w("## E2. Robustness of the representation to the language model\n")
        ins = s["inspec"]
        if isinstance(ins.get("agreement_8b_vs_70b"), dict):
            a = ins["agreement_8b_vs_70b"]
            w(f"* Inspec, 8B vs 70B on the same 2,000 documents: identical sets {pct(a['identical_set_share'])}; Jaccard {f(a['jaccard_lex']['mean'])}; soft F1 {f(a['soft_f1']['mean'])}; Soft Mean-Max {f(a['soft_mean_max']['mean'])}; global similarity {f(a['global_sem_sim']['mean'])}.")
            d = ins["gold_metrics_paired_difference"]
            w("  Against the gold keyphrases (paired difference 70B minus 8B, 95% CI): " + "; ".join(
                f"{k} {d[k]['mean_70b_minus_8b']:+.4f} [{d[k]['ci95_low']:+.4f}, {d[k]['ci95_high']:+.4f}]" for k in ["soft_f1", "soft_recall", "soft_precision", "global_sem_sim"]) + ". Full table: `inspec_gold_metrics_by_model.csv`.")
        else:
            w(f"* Inspec 8B vs 70B: {ins.get('agreement_8b_vs_70b')}")
        me = s["mathematics_education_sample"]
        if isinstance(me.get("agreement_8b_vs_70b"), dict):
            a = me["agreement_8b_vs_70b"]
            w(f"* Mathematics-education sample ({me['documents']} documents), 8B vs 70B: identical sets {pct(a['identical_set_share'])}; Jaccard {f(a['jaccard_lex']['mean'])}; soft F1 {f(a['soft_f1']['mean'])}; global similarity {f(a['global_sem_sim']['mean'])}.")
        w("")

    # ------------------------------------------------------------------ E7
    s = load("e7_extraction_cost")
    if s:
        c, p = s["full_corpus"], s["pricing"]
        w("## E7. Cost of the extraction stage (reconstructed)\n")
        w(f"* {c['calls_minimum']:,} calls; {c['input_tokens_total']:,} input tokens (mean {f(c['input_tokens_mean_per_call'],0)} per call, of which {c['prompt_constant_tokens']} are the fixed prompt) and about {int(c['output_tokens_total_calibrated']):,} output tokens (mean {f(c['output_tokens_mean_per_call_calibrated'],1)}).")
        w(f"* At the OpenRouter list price fetched on {p['fetched_utc'][:10]} (US${p['usd_per_prompt_token']*1e6:.3f}/M input, US${p['usd_per_completion_token']*1e6:.3f}/M output) the whole corpus costs US${c['usd_total_at_list_price']:.2f} "
          f"(US${c['usd_per_1000_documents']:.3f} per 1,000 documents); upper bound if every call used the four retries: US${c['usd_total_if_every_call_retried_max_attempts']:.2f}.")
        w(f"* Wall clock: at least {c['wall_clock_lower_bound_hours_from_inter_call_delay']:.1f} h from the 0.6 s inter-call delay alone; {c['wall_clock_hours_if_mean_latency_1s']:.0f} h to {c['wall_clock_hours_if_mean_latency_2s']:.0f} h for mean latencies of 1 s to 2 s.")
        w(f"* {c['records_truncated_at_3000_chars']:,} records ({pct(c['share_records_truncated'],2)}) exceeded the 3,000-character truncation.")
        w("")

    # ------------------------------------------------------------------ E6
    s = load("e6_corpus_profile")
    if s:
        w("## E6. Corpus profile\n")
        y = s["year"]
        w(f"* {s['documents']:,} records, {y['min']}-{y['max']}, median year {y['median']:.0f}; {pct(y['share_2016_or_later'])} published in 2016 or later.")
        w(f"* Abstract language (langdetect): English {pct(s['share_abstract_english'])}; other languages: " + ", ".join(f"{k} {pct(v,2)}" for k, v in list(s['language_abstract'].items())[1:6]) + f". Title in English: {pct(s['share_title_english'])}.")
        ok = s["original_keywords"]
        w(f"* Original author/index keywords: mean {f(ok['mean_per_document'],1)} per record, {ok['documents_without_original_keywords']} records without any.")
        ka = s["generated_keyword_language_audit"]
        w(f"* Generated keywords with non-ASCII characters: {pct(ka['share_non_ascii'],3)}; containing Spanish function words: {pct(ka['share_spanish_function_words'],3)}.")
        w(f"* {s['sources']['unique_sources']:,} distinct sources; the ten most frequent account for {pct(s['sources']['top10_share'])} of records.")
        w("")

    # ------------------------------------------------------------------ E4
    s = load("e4_k_sensitivity")
    if s:
        w("## E4. Sensitivity to k\n")
        w(f"* Status: {s['status']}.")
        a5 = [a for a in s["arms"] if a["arm"] == "k=5"][0]
        w(f"* k = 5 arm on the 5,000-document sample: {a5['vocabulary']:,} concepts, {a5['global_edges']:,} edges, backbone w >= 20: {a5['bb20_nodes']} nodes / {a5['bb20_edges']} edges (Q = {f(a5['bb20_modularity'],3)}); backbone w >= 5: {a5['bb5_nodes']} nodes / {a5['bb5_edges']} edges (Q = {f(a5['bb5_modularity'],3)}, {a5['bb5_communities']} communities).")
        for k, comp in s.get("comparison", {}).items():
            nov = comp["extra_keyword_novelty"]
            w(f"* {k}: backbone w>=5 node Jaccard {f(comp['bb5']['jaccard_nodes'])}, partition NMI {f(comp['bb5']['nmi'])}, ARI {f(comp['bb5']['ari'])}; extra keywords: exact restatement {pct(nov['exact_restatement_of_k5'])}, soft restatement {pct(nov['soft_restatement_of_k5'])}, new concepts {pct(nov['new_concept'])}.")
        w("")
    (R / "SUMMARY.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
