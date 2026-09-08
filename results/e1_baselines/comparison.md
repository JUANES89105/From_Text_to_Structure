# E1 extraction baseline results

All scores are document means ± sample SD (ddof=1), n=2,000 per method.

| Method | jaccard_lex | soft_precision | soft_recall | soft_f1 | soft_mean_max | global_sem_sim |
|---|---|---|---|---|---|---|
| llama_3.1_8b | 0.1422 ± 0.1118 | 0.7939 ± 0.2069 | 0.4670 ± 0.1799 | 0.5652 ± 0.1626 | 0.7522 ± 0.0791 | 0.8042 ± 0.0785 |
| tfidf | 0.0469 ± 0.0588 | 0.6211 ± 0.2549 | 0.2777 ± 0.1604 | 0.3631 ± 0.1734 | 0.6389 ± 0.0897 | 0.6623 ± 0.1250 |
| yake | 0.0496 ± 0.0580 | 0.6789 ± 0.2798 | 0.2416 ± 0.1564 | 0.3328 ± 0.1738 | 0.6360 ± 0.0989 | 0.6921 ± 0.1269 |
| keybert | 0.0119 ± 0.0287 | 0.7941 ± 0.2744 | 0.2026 ± 0.1330 | 0.3034 ± 0.1623 | 0.6411 ± 0.0718 | 0.7325 ± 0.0991 |

See README.md for configurations, timing scope, and methodological limitations.
