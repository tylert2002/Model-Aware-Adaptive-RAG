# Table 2b: Original automatic silver-label validity

## Validation

- Gemma 12B: 1,500 unique model/question/strategy records, 500 unique question IDs, and exactly one A/B/C record per question.
- Gemma 31B: 1,500 unique model/question/strategy records, 500 unique question IDs, and exactly one A/B/C record per question.
- The two models use the same 500 question IDs. Every record has a numeric F1 and `total_tokens`; failed or duplicate records: 0.
- Oracle mean F1, including all-zero questions: 0.593700 (Gemma 12B) and 0.671060 (Gemma 31B).

## Label provenance and coverage

Labels were read without extraction or modification from `adaptive-rag/data.tar.gz`, under `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/<source>/silver/train.json`. These are original automatically generated silver labels, not classifier predictions and not empirical labels derived from the Gemma runs.

| Source | Archive-member SHA-256 | Verified labels | Unmatched / 500 | A | B | C |
|---|---|---:|---:|---:|---:|---:|
| FLAN-T5 XL | `1e30e8ba60ad3e019767f6b3a4ecc4f1cd617b7848ce65bdba685fcc3d91920b` | 241 | 259 | 80 | 111 | 50 |
| FLAN-T5 XXL | `db81463a63a555493909922b5d21461144645a1758d17a4216058d40f0b53e95` | 254 | 246 | 82 | 124 | 48 |
| GPT | `3c2768efa3be066cc716c63f1191f6943d698fbfe9e817e150aa6c4d7513a44e` | 262 | 238 | 167 | 58 | 37 |

An unmatched question is excluded from every label-dependent comparison. The binary HotpotQA dataset-prior artifact (which assigns C by construction) was not substituted for the silver labels.

## Verified results

| Target model | Label source | N | Tie-aware optimal agreement (%) | Auto-label mean F1 | Matched-subset oracle F1 | Mean F1 regret | F1-first token-tiebreak agreement (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemma 12B | FLAN-T5 XL | 241 | 75.934 | 0.640705 | 0.813029 | 0.172324 | 55.602 |
| Gemma 12B | FLAN-T5 XXL | 254 | 78.346 | 0.668780 | 0.821535 | 0.152756 | 61.811 |
| Gemma 12B | GPT | 262 | 67.176 | 0.563397 | 0.807366 | 0.243969 | 59.924 |
| Gemma 31B | FLAN-T5 XL | 241 | 81.328 | 0.753817 | 0.889087 | 0.135270 | 60.166 |
| Gemma 31B | FLAN-T5 XXL | 254 | 82.677 | 0.749055 | 0.884843 | 0.135787 | 62.205 |
| Gemma 31B | GPT | 262 | 70.992 | 0.655573 | 0.887672 | 0.232099 | 63.359 |

Table 2b reports the matched-subset oracle shown here: every metric within a row is computed from exactly the same question IDs. The full-500 Oracle mean F1 is retained only as a separate descriptive statistic in `table2b_metrics.json`: 0.593700 for Gemma 12B and 0.671060 for Gemma 31B. The F1-first token-tiebreak metric forms the complete F1-optimal strategy set for each matched question, retains all members attaining the minimum observed `total_tokens`, and counts agreement exactly when the verified silver label belongs to that retained set.

## Missing inputs

- The 238--259 unmatched HotpotQA questions, depending on source, have no original silver label because none of that source model's original A/B/C outputs entered the correct-prediction artifacts used by the labeling procedure.
- SQuAD lacks complete paired Gemma A/B/C outcomes and aligned verified original silver labels, so its Table 2b fields remain `--`.
