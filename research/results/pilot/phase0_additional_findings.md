# Phase 0 Pilot: Additional Findings

This report records supplementary Pilot observations that are not part of the primary P1 result table. These are feasibility and engineering findings, not formal test-set claims. HotpotQA values below are derived from the saved per-question records and traces; SQuAD remains pending.

## 1. Experimental configuration and sample size

- Dataset: HotpotQA processed development subset (`dev_500_subsampled.jsonl`).
- Selection: 30 deterministic question IDs sampled with seed 100. The question manifest is identical for both models and all strategies.
- Models:
  - Gemma 12B: `gemma-small` → `google/gemma-4-12B-it-qat-w4a16-ct`
  - Gemma 31B: `gemma` → `google/gemma-4-31B-it-qat-w4a16-ct`
- Strategies: A (No Retrieval), B (Single-step Retrieval), and C (Multi-step Retrieval).
- Expected executions: 90 per model and 180 overall. All 180 completed, for 100% coverage.
- Inference policy: sequential execution, concurrency 1, and zero automatic retries.
- Evaluation: the original Adaptive-RAG in-memory HotpotQA answer metric (`DropAnswerEmAndF1`), with its upstream normalization and scoring logic unchanged.

Primary artifacts:

- Aggregate: [`phase0_hotpotqa_pilot_summary.json`](phase0_hotpotqa_pilot_summary.json)
- Gemma 12B run: [`20261105T181253Z_gemma-small_90f19188/`](20261105T181253Z_gemma-small_90f19188/)
- Gemma 31B run: [`20261105T182407Z_gemma_8248a157/`](20261105T182407Z_gemma_8248a157/)

Each run directory contains its immutable configuration and question manifest in `run_metadata.json`, per-question measurements in `results.jsonl`, generation traces, and retrieval traces.

## 2. Answer quality by model and retrieval strategy

| Model | Strategy | Completed | Mean F1 | Mean EM |
|---|---|---:|---:|---:|
| Gemma 12B | A | 30/30 | 0.137 | 0.100 |
| Gemma 12B | B | 30/30 | 0.278 | 0.233 |
| Gemma 12B | C | 30/30 | 0.446 | 0.367 |
| Gemma 31B | A | 30/30 | 0.167 | 0.133 |
| Gemma 31B | B | 30/30 | 0.272 | 0.233 |
| Gemma 31B | C | 30/30 | 0.486 | 0.400 |

Strategy C has the highest mean F1 and EM for both models in this Pilot. Relative to B, C increases mean F1 by 0.168 for Gemma 12B and 0.214 for Gemma 31B. These differences are descriptive; no uncertainty estimate or significance test is reported for this 30-question feasibility sample.

## 3. Retrieval calls and NRP inference calls

| Model | Strategy | Mean retrieval calls | Total retrieval calls | Mean NRP calls | Total NRP calls |
|---|---|---:|---:|---:|---:|
| Gemma 12B | A | 0.000 | 0 | 1.000 | 30 |
| Gemma 12B | B | 1.000 | 30 | 1.000 | 30 |
| Gemma 12B | C | 9.500 | 285 | 10.500 | 315 |
| Gemma 31B | A | 0.000 | 0 | 1.000 | 30 |
| Gemma 31B | B | 1.000 | 30 | 1.000 | 30 |
| Gemma 31B | C | 7.433 | 223 | 8.433 | 253 |

Across all strategies, Gemma 12B used 375 NRP requests and Gemma 31B used 313, for 688 total requests. In C, one final answer-generation call follows the iterative retrieval/reasoning calls, which accounts for the mean NRP-call count being one greater than the mean retrieval-call count.

## 4. Input, output, and total token usage

| Model | Strategy | Mean input | Mean output | Mean total | Total tokens |
|---|---|---:|---:|---:|---:|
| Gemma 12B | A | 1,486.167 | 66.467 | 1,552.633 | 46,579 |
| Gemma 12B | B | 3,107.967 | 51.400 | 3,159.367 | 94,781 |
| Gemma 12B | C | 32,032.000 | 364.233 | 32,396.233 | 971,887 |
| Gemma 31B | A | 1,486.167 | 61.767 | 1,547.933 | 46,438 |
| Gemma 31B | B | 3,107.967 | 59.633 | 3,167.600 | 95,028 |
| Gemma 31B | C | 25,816.267 | 375.833 | 26,192.100 | 785,763 |

The two models used 2,011,096 input tokens and 29,380 output tokens, or 2,040,476 total tokens. Strategy C accounts for 1,757,650 tokens (86.1% of all reported tokens). Its repeated prompts make input tokens the dominant cost component.

## 5. End-to-end latency

| Model | Strategy | Mean latency (s) | Total latency (s) |
|---|---|---:|---:|
| Gemma 12B | A | 1.983 | 59.503 |
| Gemma 12B | B | 2.365 | 70.953 |
| Gemma 12B | C | 17.006 | 510.166 |
| Gemma 31B | A | 1.385 | 41.554 |
| Gemma 31B | B | 1.715 | 51.440 |
| Gemma 31B | C | 11.427 | 342.806 |

These are end-to-end wall-clock measurements per strategy execution, including retrieval and generation. C was 8.57× as slow as A and 7.19× as slow as B for Gemma 12B. For Gemma 31B, C was 8.25× as slow as A and 6.66× as slow as B. Latency is provider- and load-dependent and should not be interpreted as a stable model-serving benchmark.

## 6. Strategy C maximum-step rates and stopping behavior

The preserved upstream configuration permits at most ten iterative retrieval steps in C.

| Model | Retrieval-call distribution in C | Hit ten-step cap | Cap rate |
|---|---|---:|---:|
| Gemma 12B | 2 calls: 1; 3 calls: 1; 10 calls: 28 | 28/30 | 93.3% |
| Gemma 31B | 2 calls: 3; 3 calls: 5; 4 calls: 3; 10 calls: 19 | 19/30 | 63.3% |

Gemma 31B stopped before the cap on 11 questions, compared with 2 for Gemma 12B. This explains much of the difference in C request count, token consumption, and latency. Reaching the cap indicates that the upstream control flow did not accept an earlier stopping decision; it does not by itself establish that the additional retrievals were useful or harmful.

## 7. Cross-model observations

Gemma 31B does not outperform Gemma 12B under every strategy:

- A: Gemma 31B is higher by 0.030 mean F1.
- B: Gemma 12B is higher by 0.006 mean F1; mean EM is tied.
- C: Gemma 31B is higher by 0.040 mean F1 and 0.033 mean EM.

Paired per-question F1 comparisons reinforce that neither model dominates consistently:

| Strategy | Gemma 31B wins | Gemma 12B wins | Ties |
|---|---:|---:|---:|
| A | 3 | 2 | 25 |
| B | 2 | 4 | 24 |
| C | 4 | 5 | 21 |

The high number of ties and small sample size make broad model-ranking claims inappropriate. The more notable operational difference is that Gemma 31B required fewer C iterations on this sample.

## 8. Retrieval efficiency and cost-quality trade-offs

For Gemma 12B, moving from B to C increases mean F1 by 0.168 while multiplying mean tokens by 10.25× and latency by 7.19×. For Gemma 31B, the corresponding F1 increase is 0.214, with 8.27× the tokens and 6.66× the latency.

Compared with A, C raises mean F1 by 0.309 for Gemma 12B and 0.319 for Gemma 31B, but uses 20.87× and 16.92× as many tokens, respectively. B occupies a middle operating point: it improves mean F1 over A at far lower cost than C, although its quality remains below C in aggregate.

These ratios characterize measured resource use, not monetary cost. A monetary comparison requires a verified and dated NRP pricing or accounting policy; no USD estimate is inferred here.

## 9. Potential implications for Model-Aware Adaptive-RAG

The Pilot supports investigating a router that selects strategies jointly by expected answer-quality gain and execution cost:

- Always using C would obtain the highest aggregate quality here but would incur most of the token and latency burden.
- A and B may be preferable on questions where C offers little or no per-question F1 improvement.
- Model identity may be relevant to routing because C stopping behavior and cost differed substantially between the two models.
- A cost-aware objective should account for retrieval calls, NRP calls, input tokens, and latency separately rather than collapsing them prematurely into one unverified monetary measure.

These are hypotheses for later experiments. The Pilot does not demonstrate that a learned model-aware router will outperform the baselines.

## 10. Limitations and follow-up questions

- The sample contains only 30 HotpotQA development questions and is not a formal held-out evaluation.
- Pilot IDs must be excluded from future formal test sets.
- No confidence intervals, significance tests, or multiple-seed analyses were conducted.
- Provider load can affect latency; the two model runs were sequential rather than simultaneous.
- The high C cap rate warrants per-question analysis of whether late steps add useful evidence, even though the stopping mechanism was intentionally left unchanged for this Pilot.
- Token counts are provider-reported and use the hosted model tokenizer, while prompt fitting uses the upstream GPT-2 tokenizer. Those tokenizations are not identical.
- The in-memory upstream answer metric was used. This report does not include supporting-fact metrics from the separate official HotpotQA evaluator.
- Future work should measure per-question F1 gain versus marginal retrieval, token, and latency cost and define a pre-registered cost-aware tie policy.
- Question-category analyses require a documented annotation or classification procedure rather than post hoc categories.

## 11. SQuAD findings (pending)

The SQuAD Pilot has not been executed. No SQuAD quality, retrieval, token, latency, stopping, or cross-model values are available, and none are inferred here. When SQuAD artifacts become available, this section should use the same reporting structure and explicitly record any dataset-specific evaluation differences.

## Reproduction note

All aggregate means use completed executions only; coverage is reported separately. Values can be reproduced by grouping each run's `results.jsonl` by `strategy`, computing arithmetic means for `f1`, `exact_match`, `retrieval_calls`, `llm_request_count`, `input_tokens`, `output_tokens`, `total_tokens`, and `end_to_end_latency`, and checking C's cap rate as the fraction of completed C rows with `retrieval_calls == 10`. The saved summary provides rounded reference values, while the JSONL files retain per-question precision.
