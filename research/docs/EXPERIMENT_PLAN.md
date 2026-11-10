# Experiment Plan

This is the operational roadmap for Probability-Space Retrieval Gain Calibration. The stable publication-facing table shells live in `../paper/tables/`; this document records dependencies and expected artifacts without claiming results. The evidence-backed repository inventory and missing-data checklist are in `PROBABILITY_SPACE_CALIBRATION_AUDIT.md`.

## Pilot prerequisite — feasibility and pipeline validation

| ID | Experiment | Expected deliverable | Result table / figure | Status |
|---|---|---|---|---|
| EP.1 | Pilot Model and Pipeline Validation | **Prerequisite:** NRP API compatibility and pipeline validation; preliminary A/B/C F1; API success/failure statistics; runtime and token-usage summary | P1 | Completed for HotpotQA; SQuAD Partial |

EP.1 uses approximately 30–50 questions that are identical across candidate models and strategies A (No Retrieval), B (Single-step Retrieval), and C (Multi-step Retrieval). It verifies NRP-hosted LLM integration, strategy execution, answer generation, F1 evaluation, failure and timeout handling, retrieval errors, and runtime/resource feasibility before formal Phase 0 begins.

Pilot questions must be disjoint from the final held-out test set, and pilot measurements must never be reused as formal evaluation results. Models and configurations must not be selected solely because pilot outcomes favor the research hypothesis. Retain every per-question and per-strategy output, including answers, errors, token usage, and latency. P1 is a feasibility record, not a final research claim.

### EP.1 integration readiness checklist

- [x] Audit the original local Hugging Face, custom `/generate`, and legacy OpenAI call paths.
- [x] Verify the official NRP base URL, chat endpoint, authentication scheme, and candidate aliases (`gemma-small`, `gemma`).
- [x] Implement a research-owned chat adapter with exact-model enforcement, request accounting, timeouts, and bounded transient retries.
- [x] Add a credential-gated sequential smoke-test script and `.env.example`; no authenticated request has been sent.
- [ ] User configures `NRP_API_KEY` locally and optionally a private `NRP_CACHE_SALT`.
- [ ] Run the two-model smoke test after explicit approval; confirm alias availability, response schema, usage fields, and thinking-off behavior.
- [ ] Verify which optional decoding parameters are accepted and freeze one comparable generation configuration.
- [ ] Resolve hosted-tokenizer prompt fitting without changing prompt content or silently downloading candidate model weights.
- [ ] Implement and review a research-owned generator-factory injection; do not patch upstream algorithms.
- [ ] Freeze 30–50 pilot-only question IDs and verify zero overlap with formal held-out test IDs.
- [ ] Run A/B/C sequentially or at conservative concurrency, retaining every success and failure artifact.

The NRP adapter defaults to one request at a time even though the current documented short-context ceiling is higher. Strategy C is expected to make multiple LLM calls per question, so its request count, latency, and failure exposure are not directly comparable to the one-answer-call A/B paths without per-execution accounting.

## Phase 0 — Reproduction and routing-label validity

| ID | Experiment | Expected deliverable | Result table / figure | Status |
|---|---|---|---|---|
| E0.0 | Dataset and Evaluation Setup | **Direct:** frozen disjoint calibration/validation/test splits, corpus/index versions, and model, generation, retrieval, and evaluation configuration | T1 | Partial |
| E0.1 | Evaluate the original Adaptive-RAG router | **Direct:** original-router metrics and routing decisions with exact classifier checkpoint and run manifest | T4 | Pending |
| E0.2 | Evaluate A/B/C on the same questions | **Intermediate:** aligned per-question predictions, EM/F1, retrieval counts, tokens, latency, and traces | Feeds T2–T4 | **Completed:** HotpotQA 500-question paired A/B/C evaluation for Gemma 12B and 31B |
| E0.3 | Compute retrospective oracle upper bound | **Analytical only:** exact F1-optimal tie sets, oracle quality, and regret; never used for test routing | T2 | **Completed:** HotpotQA 500-question retrospective analysis |
| E0.4 | Analyze original-router disagreement, ties, and F1 regret | **Direct:** disagreement categories and per-question/aggregate regret analysis | T2; F1 | Pending |

Phase 0 must freeze question identifiers and use the same evaluator for all three strategies. Calibration, validation, and test questions must not overlap. The quality oracle contains every strategy tied for highest per-question F1; the cost-aware oracle selects the least expensive quality-equivalent strategy under a pre-registered cost definition.

## Phase 1 — Model profiling

| ID | Experiment | Expected deliverable | Result table / figure | Status |
|---|---|---|---|---|
| E1.1 | Profile retrieval gains for Gemma 12B | **Direct:** per-question `F1_B-F1_A` and `F1_C-F1_A` with provenance | T3; F2 | Partial: pilot gains derivable offline |
| E1.2 | Profile retrieval gains for Gemma 31B | **Direct:** per-question `F1_B-F1_A` and `F1_C-F1_A` with provenance | T3; F2 | Partial: pilot gains derivable offline |
| E1.3 | Diagnose probability-space locality | **Direct:** whether nearby classifier probability vectors have similar target-LLM retrieval gains | T3; F2 | Pending |

Both models must use identical held-out questions and compatible retrieval settings. Question categories must come from available annotations or a documented, frozen classification procedure.

## Phase 2 — Method construction

| ID | Experiment | Expected deliverable | Result table / figure | Status |
|---|---|---|---|---|
| E2.1 | Export original-classifier A/B/C probability vectors | **Intermediate:** separate immutable per-question logits/probabilities with checkpoint and tokenizer provenance | Feeds E2.2 and T3–T6 | Pending |
| E2.2 | Implement Probability-Space Calibration | **Intermediate:** distance-weighted KNN over `[P(A),P(B),P(C)]`, using target-LLM-specific observed retrieval gains | Feeds T4–T6 | Pending |

Shared research code belongs under `../src/`, while experiment-specific code belongs in the corresponding directory under `../experiments/`; the original Adaptive-RAG implementation remains unchanged.

## Phase 3 — Main evaluation

| ID | Experiment | Expected deliverable | Result table / figure | Status |
|---|---|---|---|---|
| E3.1 | Evaluate fixed A/B/C and original-router baselines | **Direct:** quality, regret, retrieval-call, latency, token/cost, and routing outputs | T4–T5 | Pending |
| E3.2 | Evaluate distance-weighted KNN across target LLMs | **Direct:** held-out proposed-method metrics and model-specific calibration comparisons | T4–T5 | Pending |
| E3.3 | Compare answer quality and retrieval cost | **Direct:** quality–efficiency analysis with itemized and dated cost scope | T4; F3 | Pending |

Use one frozen evaluation protocol. Retrieval cost, model-inference cost, and total API cost are separate quantities and must not be conflated.

## Phase 4 — Calibration and ablations

| ID | Experiment | Expected deliverable | Result table / figure | Status |
|---|---|---|---|---|
| E4.1 | Conduct calibration-size ablation | **Direct:** KNN quality, regret, and retrieval learning curves using nested calibration subsets | T6; F4 | Optional |
| E4.2 | Evaluate cost-aware utility selection | **Direct:** same KNN gain estimates with a pre-registered retrieval-cost penalty | T6 | Optional |

## Result index

- **P1:** Pilot model and pipeline validation; excluded from formal test results.
- **T1:** Dataset statistics and experimental configuration.
- **T2:** Validity of automatically generated routing labels.
- **T3:** Model-specific retrieval gain analysis.
- **T4:** Main answer-quality and retrieval-efficiency results.
- **T5:** Cross-model generalization and adaptation.
- **T6:** Model-aware routing ablations.
- **F1:** Label disagreement, ties, and F1-regret analysis.
- **F2:** Model-specific retrieval-gain profiles.
- **F3:** Answer-quality versus retrieval-efficiency trade-off.
- **F4:** Calibration sample-efficiency curves.

“Direct” deliverables populate a result table or planned figure. “Intermediate” deliverables are versioned artifacts required by later experiments and are not standalone result claims.

## Artifact and update policy

For every completed row, archive or reference the code revision, immutable model identifier, prompt and generation settings, retriever/index version, split manifest, random seed where applicable, predictions, routing decisions, per-question metrics, retrieval calls, latency protocol, and cost scope. Update the corresponding file under `../paper/tables/` only after verifying the artifact. Missing measurements remain `--`.

Audit status values are **Completed**, **Partial**, **Pending**, and **Optional**. Status describes artifact readiness, not whether a hypothesis was supported. A pilot completion never implies completion of the formal held-out experiment.
