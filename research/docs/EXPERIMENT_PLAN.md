# Experiment Plan

This is the operational roadmap for Model-Aware Adaptive-RAG. The stable publication-facing table shells live in `../paper/tables/`; this document records dependencies and expected artifacts without claiming results.

## Pilot prerequisite — feasibility and pipeline validation

| ID | Experiment | Expected deliverable | Result table / figure | Status |
|---|---|---|---|---|
| EP.1 | Pilot Model and Pipeline Validation | **Prerequisite:** NRP API compatibility and pipeline validation; preliminary A/B/C F1; API success/failure statistics; runtime and token-usage summary | P1 | Complete |

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
| E0.0 | Dataset and Evaluation Setup | **Direct:** frozen splits, corpus/index versions, and model, generation, retrieval, and evaluation configuration | T1 | Not Started |
| E0.1 | Reproduce the original Adaptive-RAG evaluation | **Direct:** baseline metrics with exact command, environment, configuration, and run manifest | T4 | Not Started |
| E0.2 | Evaluate A/B/C on the same questions | **Intermediate:** aligned per-question predictions, EM/F1, retrieval counts, and traces | Feeds T2–T4 | Not Started |
| E0.3 | Compare automatic labels with empirical oracle strategies | **Direct:** quality-oracle sets, cost-aware choices, agreement, regret, and tie policy | T2 | Not Started |
| E0.4 | Analyze disagreement, ties, and F1 regret | **Direct:** disagreement categories and per-question/aggregate regret analysis | T2; F1 | Not Started |

Phase 0 must freeze question identifiers and use the same evaluator for all three strategies. Calibration, validation, and test questions must not overlap. The quality oracle contains every strategy tied for highest per-question F1; the cost-aware oracle selects the least expensive quality-equivalent strategy under a pre-registered cost definition.

## Phase 1 — Model profiling

| ID | Experiment | Expected deliverable | Result table / figure | Status |
|---|---|---|---|---|
| E1.1 | Profile retrieval gains for LLM 1 | **Direct:** per-question and category-level retrieval-gain profile for LLM 1 | T3; F2 | Not Started |
| E1.2 | Profile retrieval gains for LLM 2 | **Direct:** per-question and category-level retrieval-gain profile for LLM 2 | T3; F2 | Not Started |

Both models must use identical held-out questions and compatible retrieval settings. Question categories must come from available annotations or a documented, frozen classification procedure.

## Phase 2 — Method construction

| ID | Experiment | Expected deliverable | Result table / figure | Status |
|---|---|---|---|---|
| E2.1 | Construct the Retrieval Gain Matrix | **Intermediate:** versioned gain matrix with question, model, strategy, and provenance metadata | Feeds E2.2 and T4–T6 | Not Started |
| E2.2 | Implement the model-aware retrieval router | **Intermediate:** tested router, specification, versioned profiles, and routing decisions | Feeds T4–T6 | Not Started |

Shared research code belongs under `../src/`, while experiment-specific code belongs in the corresponding directory under `../experiments/`; the original Adaptive-RAG implementation remains unchanged.

## Phase 3 — Main evaluation

| ID | Experiment | Expected deliverable | Result table / figure | Status |
|---|---|---|---|---|
| E3.1 | Evaluate all baseline routing strategies | **Direct:** baseline quality, regret, retrieval-call, latency, cost, and routing outputs | T4–T5 | Not Started |
| E3.2 | Evaluate the proposed method across LLMs | **Direct:** proposed-method metrics, routing traces, and cross-model calibration comparisons | T4–T5 | Not Started |
| E3.3 | Compare answer quality and retrieval cost | **Direct:** quality–efficiency analysis with itemized and dated cost scope | T4; F3 | Not Started |

Use one frozen evaluation protocol. Retrieval cost, model-inference cost, and total API cost are separate quantities and must not be conflated.

## Phase 4 — Calibration and ablations

| ID | Experiment | Expected deliverable | Result table / figure | Status |
|---|---|---|---|---|
| E4.1 | Conduct calibration sample-efficiency experiments | **Direct:** quality, regret, and retrieval learning curves by calibration-set size | F4 | Not Started |
| E4.2 | Perform ablation studies | **Direct:** component-level quality, regret, and retrieval metrics | T6 | Not Started |

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

Allowed status values are **Not Started**, **In Progress**, **Blocked**, and **Complete**. Status describes artifact readiness, not whether a hypothesis was supported.
