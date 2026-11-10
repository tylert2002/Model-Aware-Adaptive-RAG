# Probability-Space Retrieval Gain Calibration: Repository Audit

Audit date: 2026-11-05. This is a planning and artifact-inventory document. No
inference was launched, and no result record or completed numerical table entry
was changed.

## Status definitions

- **Completed:** every required output for the stated scope is present and valid.
- **Partial:** usable artifacts exist, but at least one required output or split is missing.
- **Pending:** no qualifying result artifact exists yet.
- **Optional:** useful analysis that is not required for the primary claim.

The 30-question HotpotQA runs are pilot-only feasibility evidence. They cannot be
promoted to formal held-out results. The earlier one-question runs are compatibility
checks and are not pooled with the 30-question pilots.

## Updated experiment inventory

| Experiment | Model | Dataset | Strategy/Method | Available Data | Missing Data | Status | Output Table |
|---|---|---|---|---|---|---|---|
| Pilot A/B/C baselines | Gemma 12B | HotpotQA pilot | A, B, C | 30 shared IDs; 90/90 complete rows; answer, F1, EM, retrieval calls, LLM calls, input/output/total tokens, generation and end-to-end latency, traces, run manifest | Nothing for pilot scope; formal split remains separate | Completed | P1 only |
| Pilot A/B/C baselines | Gemma 31B | HotpotQA pilot | A, B, C | 30 shared IDs; 90/90 complete rows with the same fields and IDs as Gemma 12B | Nothing for pilot scope; formal split remains separate | Completed | P1 only |
| SQuAD pilot A/B/C | Gemma 12B / 31B | SQuAD pilot | A, B, C | Frozen 30-ID manifest and processed questions | Verified Wikipedia corpus/index and all A/B/C outputs | Partial | P1 only |
| Formal A/B/C baselines | Gemma 12B / 31B | Frozen target test set | A, B, C | Pilot pipeline and metric implementation; upstream archived predictions are from different target models/settings | Disjoint formal calibration/validation/test manifests and valid A/B/C outcomes | Pending | T2a, T3, T4 |
| Archived upstream A/B/C | FLAN-T5 XL, FLAN-T5 XXL, GPT | Six upstream datasets | A, B, C | `predictions.tar.gz` contains predictions, aggregate metrics, ground truth, configs, and some chains for dev-500 and test runs | Provenance validation and conversion to aligned per-question F1/cost records; these are not Gemma results | Partial | Reuse analysis only; do not mix into Gemma T4 |
| Original Adaptive-RAG router | Original T5-large classifier + each target LLM | Formal target test set | Original router baseline | Classifier source and label-only postprocessing are present; archived A/B/C answers may support upstream-model reconstruction | Exact original classifier checkpoint, verified predictions for the frozen IDs, and target-LLM A/B/C answer selection/evaluation | Pending | T4, T5 |
| Probability-Space Calibration | Gemma 12B | Formal target split | Distance-weighted KNN | Complete pilot A/B/C gains can exercise offline code later | Classifier probability vectors; disjoint calibration/validation/test A/B/C records; tuned KNN and held-out evaluation | Pending | T4, T5 |
| Probability-Space Calibration | Gemma 31B | Formal target split | Distance-weighted KNN | Complete pilot A/B/C gains can exercise offline code later | Same as above, retained separately for Gemma 31B | Pending | T4, T5 |
| Cross-LLM comparison | Gemma 12B vs 31B | HotpotQA pilot | Fixed A/B/C | Same 30 IDs and compatible settings; paired descriptive comparisons exist | Formal disjoint splits, probability vectors, original-router and KNN routing outputs | Partial | P1 now; T5 formal |
| Probability-space diagnostic | Each target LLM separately | Calibration/validation only | Neighbor distance vs retrieval-gain similarity | Per-question pilot gains are derivable offline | Classifier probability vectors and a pre-registered diagnostic/statistic | Pending | T3 / diagnostic figure |
| Calibration-size ablation | Each target LLM separately | Calibration/validation/test | Distance-weighted KNN | A/B/C reuse design is feasible | Sufficient disjoint data; fixed size grid/seeds; validation-only tuning; held-out evaluation | Optional | T6 / F4 |
| Oracle upper bound | Each target LLM separately | Paired A/B/C set | Best observed F1 per question | Pilot oracle is derivable offline from existing rows | Formal paired A/B/C test records and frozen tie policy | Optional | T2a/T2b context; analytical bound only |

## Artifact audit

### Pilot and expanded-run coverage

| Run | Purpose | Unique questions | Complete A/B/C questions | Field coverage |
|---|---|---:|---:|---|
| `20261105T172646Z_gemma-small_5946a862` | Failed compatibility attempt | 1 | 0 | Retrieval/LLM-call counts and elapsed time only; F1, EM, and tokens are null |
| `20261105T180039Z_gemma-small_69d950f0` | Successful compatibility check | 1 | 1 | F1, EM, retrieval/LLM calls, tokens, latency, and traces |
| `20261105T181253Z_gemma-small_90f19188` | Gemma 12B HotpotQA pilot | 30 | 30 | All 90 rows contain F1, EM, retrieval/LLM calls, tokens, latency, and traces |
| `20261105T182407Z_gemma_8248a157` | Gemma 31B HotpotQA pilot | 30 | 30 | All 90 rows contain F1, EM, retrieval/LLM calls, tokens, latency, and traces |

The two 30-question runs have identical question-ID sets, no duplicate
question/strategy rows, identical dataset hash, identical A/B/C configuration
hashes, and the same decoding settings. Their target model IDs differ as intended.
The one-question compatibility ID is contained in the later pilot and must not be
counted as a 31st question. Thus the repository has 30 unique HotpotQA pilot
questions with complete A/B/C results for each target LLM, not 31.

The pilot result schema contains generated answers, reference answers, F1, EM,
retrieval-call counts, downstream LLM-request counts, input/output/total tokens,
generation latency, end-to-end latency, completion/error status, reasoning chains,
and trace paths. Retrieval trace files preserve individual retrieval requests and
responses. The saved aggregate summary and P1 table already report the completed
pilot values; those values were not recalculated or altered in this audit.

### Original archive reuse boundary

`adaptive-rag/predictions.tar.gz` contains upstream dev-500 and test artifacts for
A/B/C with FLAN-T5 XL, FLAN-T5 XXL, and GPT configuration families across the six
supported datasets. These artifacts are valuable for reproducing upstream aggregate
baselines and checking label construction, but they are not interchangeable with the
Gemma pilot or future formal Gemma runs. The archive does not contain probability or
logit records from the routing classifier. `adaptive-rag/data.tar.gz` contains
classifier training/validation examples and `predict.json`, but not a trained
classifier checkpoint.

## Classifier probability assessment

The original classifier is a T5-large sequence-to-sequence model with labels A/B/C.
In `adaptive-rag/classifier/run_classifier.py`, evaluation calls `model.generate`
with `return_dict_in_generate=True` and `output_scores=True`, extracts the first
decoder-step scores for the tokenizer IDs of A, B, and C, applies softmax across
those three scores, and selects the argmax. Therefore a usable three-coordinate
probability-space representation is available in principle.

Important qualifications:

1. The current code saves only `prediction`, `answer`, and `dataset_name` in
   `dict_id_pred_results.json`; logits and probabilities are discarded.
2. These are first-step label-token scores normalized over A/B/C, not an already
   archived probability vector and not a downstream LLM confidence score.
3. Before extraction, verify with the recovered original tokenizer that A, B, and C
   each map to the intended distinct single label token at the first decoder step.
4. Preserve both raw A/B/C logits and the restricted softmax vector, plus checkpoint,
   tokenizer, input, code-revision, and question-ID provenance in a new derived file.
5. Do not silently substitute a newly trained classifier or another checkpoint.

Minimum future classifier-only work: recover and checksum the exact original
classifier checkpoint; run its existing prediction inputs once with frozen ordering;
save a separate immutable record keyed by question ID containing `logits_A_B_C`,
`probabilities_A_B_C`, and the argmax label. This does not require classifier
retraining, fine-tuning, retrieval, or target-LLM inference. It was not run here.

## Paper table mapping

- **P1:** already populated by the two completed 30-question HotpotQA pilot runs.
  The SQuAD rows correctly remain missing.
- **T1:** pending formal split counts and frozen corpus/index/configuration metadata.
  The pilot manifest is evidence for feasibility, not a formal split.
- **T2a:** the pilot A/B/C rows are sufficient to derive a pilot-only optimal-strategy
  distribution, but the current table is a formal-result shell. Do not fill it with
  pilot values. Formal paired A/B/C records are still missing.
- **T2b:** additionally requires verified original automatic labels. Classifier
  argmax predictions are not substitutes for the label-generation artifacts.
- **T3:** per-question gains can be derived for the pilot, but probability-space and
  documented-category analyses remain unavailable.
- **T4:** fixed-strategy pilot values exist only in P1. Formal fixed A/B/C, original
  router, and proposed KNN results are pending.
- **T5:** paired pilot A/B/C comparisons are descriptive; formal router-transfer and
  model-specific calibration results are pending.
- **T6/F4:** optional until the primary distance-weighted KNN evaluation is complete.
- **Oracle:** report only as a retrospective analytical upper bound. It must never
  route a held-out test question or supply its label to KNN.

## Exact missing-data checklist

### Fixed A/B/C and original router

- Freeze mutually disjoint calibration, validation, and held-out test manifests,
  excluding all pilot IDs; check duplicate IDs within and across splits.
- For each target LLM separately, obtain valid A/B/C answer, F1, EM, retrieval-call,
  token, latency, trace, and configuration records on every included question.
- Recover the exact original classifier checkpoint and tokenizer; archive checksums.
- Extract original-router labels and evaluate selection only from already generated
  A/B/C answers whenever those aligned answers exist.

### Proposed distance-weighted KNN

- Obtain one verified `[P(A), P(B), P(C)]` vector per included question.
- Join calibration IDs to target-LLM-specific `F1_A`, `F1_B`, and `F1_C`; derive
  `gain_B = F1_B - F1_A` and `gain_C = F1_C - F1_A` in a separate file.
- Pre-register distance metric, weight function and zero-distance behavior, candidate
  values of `k`, utility/cost units, deterministic tie handling, and missing-row policy.
- Tune all hyperparameters and any cost coefficient on calibration/validation only.
- Evaluate once on held-out test. Exclude self-neighbors by ID and assert no test ID
  appears in the calibration or validation neighbor pools.
- Keep calibration outcomes and fitted choices separate by target LLM.

### Diagnostics and optional analyses

- Probability-space diagnostic: neighbor-distance bins or rank correlation versus
  absolute gain differences, defined before examining test outcomes.
- Calibration-size ablation: fixed nested calibration subsets and seeds, with the test
  set untouched and hyperparameters selected without test access.
- Oracle: compute `max(F1_A,F1_B,F1_C)` and exact tie sets offline after all three
  outcomes exist; never expose this choice to a deployable router.

## Minimum implementation sequence (future work)

1. Freeze and validate disjoint ID manifests, with pilot-ID exclusion assertions.
2. Recover the exact classifier checkpoint and export probability vectors to a new
   derived artifact without changing original classifier outputs.
3. Normalize existing A/B/C JSONL into read-only joined views keyed by
   `(dataset, target_model, question_id)`; never rewrite source JSONL.
4. Build target-LLM-specific calibration rows containing probabilities, F1 values,
   gains, costs, and provenance.
5. On calibration/validation only, fit the single proposed method: distance-weighted
   KNN in the three-dimensional classifier probability simplex.
6. Freeze `k`, distance weighting, utility/cost rule, and tie policy.
7. Route held-out test questions without using their outcomes or allowing self-neighbors;
   select the corresponding precomputed A/B/C answer and evaluate offline.
8. Produce separate derived routing decisions, metrics, diagnostics, and table inputs.

There is no Hard-label Gain Matrix baseline, no separate interpolation method, and no
test-time oracle in this plan.
