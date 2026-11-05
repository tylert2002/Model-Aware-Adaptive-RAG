# Phase 0 Repository Notes

## Scope

Phase 0 will test whether Adaptive-RAG's automatically generated routing labels
select the retrieval strategy that maximizes downstream QA performance. The
strategy mapping used throughout the original repository is:

- **A — No Retrieval:** `nor_qa`
- **B — Single-step Retrieval:** `oner_qa`
- **C — Multi-step Retrieval:** `ircot_qa`

This file documents the existing implementation only. No experiments were run
and no research method was changed during repository initialization.

All baseline paths below are relative to `adaptive-rag/`. Run original commands
with that directory as the working directory.

## 1. Automatic label generation

Primary paths:

- `classifier/preprocess/preprocess_silver_train.py`
- `classifier/preprocess/preprocess_silver_valid.py`
- `classifier/preprocess/preprocess_silver_train_gpt.py`
- `classifier/preprocess/preprocess_silver_valid_gpt.py`
- `classifier/preprocess/preprocess_utils.py`
- `classifier/preprocess/preprocess_binary_train.py`
- `classifier/preprocess/concat_binary_silver_train.py`

`preprocess_silver_train.py` and `preprocess_silver_valid.py` locate prediction
files for `nor_qa`, `oner_qa`, and `ircot_qa`, then call
`label_complexity(...)` from `preprocess_utils.py`. That function assigns C when
the multi-step system's correct-prediction file contains the question, then B
for the single-step system, then A for no retrieval. Because these are separate
`if` statements, later matches overwrite earlier ones; the effective priority
for questions answered correctly by multiple strategies is A over B over C.
The `total_answer` field preserves every strategy that answered correctly.

The binary-label path supplies dataset-inductive-bias labels: single-hop data
are labeled B, while MuSiQue, HotpotQA, and 2WikiMultiHopQA are labeled C.
`concat_binary_silver_train.py` combines binary and silver training examples.

Phase 0 implication: distinguish the final scalar `answer` label from
`total_answer`; the latter is needed to study ties and whether the label is a
true downstream-performance optimum rather than merely a correct/incorrect
heuristic.

## 2. Question complexity classifier

Primary paths:

- `classifier/run_classifier.py`
- `classifier/utils.py`
- `classifier/run/run_large_train_xl.sh`
- `classifier/run/run_large_train_xxl.sh`
- `classifier/run/run_large_train_gpt.sh`
- `classifier/preprocess/preprocess_predict.py`
- `classifier/postprocess/predict_complexity_on_classification_results.py`
- `classifier/postprocess/postprocess_utils.py`

`run_classifier.py` is a Hugging Face/Accelerate seq2seq training and inference
script. It maps A/B/C to class IDs 0/1/2 and trains the original T5-large
classifier using the prepared JSON files. The postprocessing code reads the
classifier prediction for each question, selects the corresponding strategy's
saved QA prediction, and records the chosen option and retrieval-step count.

## 3. Retrieval strategy execution

Entry points and orchestration:

- `run_retrieval_dev.sh`
- `run_retrieval_test.sh`
- `predict.py`
- `commaqa/inference/configurable_inference.py`
- `commaqa/inference/model_search.py`
- `commaqa/inference/participant_qa.py`
- `retriever_server/serve.py`
- `retriever_server/unified_retriever.py`
- `retriever_server/elasticsearch_retriever.py`

Representative strategy configurations:

- A: `base_configs/nor_qa_flan_t5_xl_nq.jsonnet`
- B: `base_configs/oner_qa_flan_t5_xl_nq.jsonnet`
- C: `base_configs/ircot_qa_flan_t5_xl_nq.jsonnet`

The `nor_qa` configuration sends the question directly to the QA model. The
`oner_qa` configuration performs one BM25 retrieval before answering. The
`ircot_qa` configuration alternates retrieval and chain-of-thought generation
through an exit controller, then answers using the accumulated context.
Equivalent configuration families exist for each supported dataset and model.

## 4. Downstream QA evaluation

Primary paths:

- `evaluate.py`
- `evaluate_final_acc.py`
- `metrics/squad_answer_em_f1.py`
- `metrics/drop_answer_em_f1.py`
- `metrics/drop_eval.py`
- `official_evaluation/hotpotqa/`
- `official_evaluation/2wikimultihopqa/`
- `official_evaluation/musique/`

`predict.py` invokes `evaluate.py` after inference unless evaluation is skipped.
`evaluate.py` supports local EM/F1 metrics and dataset-specific official
evaluators. `evaluate_final_acc.py` evaluates the classifier-routed predictions
assembled by the postprocessing step. Phase 0 should retain per-question
strategy predictions and compute comparable per-question EM/F1 before deriving
an optimal-strategy target.

## 5. Existing datasets and prediction artifacts

Tracked archives:

- `data.tar.gz` — prepared classifier data
- `processed_data.tar.gz` — processed QA datasets
- `predictions.tar.gz` — prediction outputs for the retrieval strategies

Download/preparation scripts:

- `download/raw_data.sh`
- `download/processed_data.sh`
- `download/official_eval.sh`
- `processing_scripts/`

Expected extracted paths referenced by the code are `raw_data/`,
`processed_data/`, `predictions/dev_500/`, `predictions/test/`, and
`classifier/data/`. The archives were preserved exactly as cloned and were not
extracted during initialization.

## Component interaction

1. Each QA dataset is run through A, B, and C using the corresponding Jsonnet
   configuration family.
2. Evaluation produces correct-prediction artifacts for each strategy.
3. Silver preprocessing merges membership in those artifacts into A/B/C labels;
   binary dataset-prior labels may also be concatenated.
4. T5-large learns question-to-complexity-label prediction.
5. Postprocessing uses the predicted label to select an already generated QA
   answer from A, B, or C.
6. `evaluate_final_acc.py` reports downstream performance of the routed answer.

## Dependencies and operational cautions

- The original setup targets Python 3.8, PyTorch 1.13.1/CUDA 11.7, and the
  packages in `requirements.txt`.
- Retrieval requires Elasticsearch 7.10.2, an indexed corpus, and the local
  FastAPI retriever service.
- FLAN-T5 execution requires model weights and the local LLM server; GPT paths
  require an external API key. Neither was started or downloaded here.
- Some scripts contain experiment-specific relative paths and dated classifier
  output directories; validate these before Phase 0 execution.
- Do not commit credentials, `.env` files, downloaded model checkpoints, or
  newly extracted large corpora without an explicit data/versioning policy.

## 6. NRP-hosted downstream LLM audit

The pilot candidates are the NRP API aliases `gemma-small` and `gemma`. The
original Adaptive-RAG code cannot use the current NRP endpoint directly.

### Original model backends

- `commaqa/execution/llm_qa_model.py::LLMQAModel` selects either
  `GPT3Generator` (`gen_model="gpt3"`) or `LLMClientGenerator`
  (`gen_model="llm_api"`). The same generator classes are also constructed by
  the iterative participants in `commaqa/inference/ircot.py`.
- `commaqa/models/llm_client_generator.py::LLMClientGenerator` calls a custom
  local `GET /generate` service, accepts only a hard-coded model-name list, and
  expects `generated_texts` plus a matching local `model_name`. It is not an
  OpenAI-compatible client.
- `llm_server/serve.py::get_model_and_tokenizer` loads Hugging Face models and
  tokenizers into local GPU memory. Its `/generate` route returns the custom
  fields `generated_num_tokens`, `generated_texts`,
  `run_time_in_seconds`, and `model_name`.
- `commaqa/models/gpt3generator.py::GPT3Generator` uses the legacy
  `openai.Completion.create` interface. NRP documents the modern
  `/v1/chat/completions` interface, so changing only the API base is
  insufficient. The original requirements do not pin a compatible OpenAI SDK
  version.
- `commaqa/models/generator.py::LMGenerator` is a separate local Transformers
  path that loads weights and sequence scores; it is not the configured
  `llm_api` path in the representative A/B/C configurations.

No configured retrieval strategy depends on LLM hidden states. The local
server explicitly disables output scores. There are, however, tokenizer and
text-processing dependencies: `commaqa/inference/prompt_reader.py` uses a Hugging
Face tokenizer to fit demonstrations within a model-length budget;
`LLMClientGenerator.generate_text_sequence` passes its model name into that
logic; and iterative C uses spaCy sentence segmentation. An NRP alias is not a
Hugging Face tokenizer identifier, and the hosted API does not expose tokenizer
objects. Prompt-length handling must therefore be frozen and validated rather
than silently substituted.

### A/B/C interaction with the LLM

- A (`nor_qa`) constructs the no-context QA prompt and makes one downstream LLM
  answer call.
- B (`oner_qa`) performs the configured BM25 retrieval first, constructs the
  context-bearing prompt, and makes one downstream LLM answer call.
- C (`ircot_qa`) alternates BM25 retrieval with
  `StepByStepCOTGenParticipant` generation. It can make multiple downstream LLM
  calls per question before making the final answer call; its upper bounds,
  paragraph limits, controller behavior, and sentence parsing must remain
  unchanged.

The prediction/evaluation boundary is reusable. `configurable_inference.py`
writes per-question predictions and reasoning chains, while `evaluate.py`,
`evaluate_final_acc.py`, and the metric modules consume prediction artifacts
without requiring local model weights or hidden states. Backend provenance must
be recorded, and NRP Gemma results must not be described as exact reproduction
of the original FLAN-T5/GPT results.

### Verified NRP API contract (2026-11-05)

Official NRP documentation currently specifies:

- base URL `https://ellm.nrp-nautilus.io/v1`;
- `POST /chat/completions` with Bearer-token authentication;
- compatibility with the modern OpenAI Python client;
- exact model aliases `gemma-small` (12B, evaluating) and `gemma` (31B,
  generally supported);
- reasoning enabled by default for both, with
  `chat_template_kwargs.enable_thinking` documented as the disable/enable
  control;
- an 8-request short-context per-user concurrency ceiling for these aliases,
  one concurrent request at 35% or more of context, and a gateway limit of
  48,000 output tokens/minute per API token and model; and
- HTTP 429 behavior at the token-rate limit plus guidance to retry with
  increasing intervals and reduce concurrency when latency rises.

The documentation recognizes optional `max_tokens`, warns that it is an output
limit rather than total context, and does not enumerate every decoding
parameter for each deployment. Exact support for temperature, top-p, stop
sequences, usage accounting, request identifiers, and response latency remains
verified by one authenticated connectivity request per candidate model.
Client-observed wall-clock latency remains the authoritative pilot latency
measurement; API token usage is recorded when present and otherwise marked
unavailable. Those connectivity measurements are not RAG performance results.

Sources:

- `https://nrp.ai/documentation/userdocs/ai/llm-managed/api-access/`
- `https://nrp.ai/documentation/userdocs/ai/llm-managed/models/`
- `https://nrp.ai/documentation/userdocs/ai/llm-managed/fair-use/`
- `https://nrp.ai/documentation/userdocs/ai/llm-managed/lifecycle/`

### Research-owned adapter architecture

Integration code is isolated under `research/experiments/pilot/`:

1. `nrp_adapter.py` sends chat-completion requests, requires an exact candidate
   alias, records client latency and returned token usage, and handles timeouts,
   429s, and transient server errors with bounded exponential backoff. It never
   falls back to NRP's `service` alias or another LLM.
2. `adaptive_rag_bridge.py` exposes the original
   `generate_text_sequence(prompt)` result shape and appends request metadata to
   JSONL. The research-owned pilot runner now injects it at the instantiated
   upstream generator boundary; it is not an upstream patch.
3. `smoke_test.py` requires `--confirm-network`, then sequentially checks both
   aliases and prints sanitized response metadata without response text or
   credentials.

The research-owned `pipeline.py` now builds the unchanged upstream state graphs
and replaces only their instantiated generator objects. The fixed mapping is
`nor_qa_gpt_<dataset>` for A, `oner_qa_gpt_<dataset>` for B, and
`ircot_qa_gpt_<dataset>` for C. It asserts the corresponding upstream start
states (`generate_question`, `generate_titles`, and
`step_by_step_bm25_retriever`) before execution.

The reused upstream boundaries are:

- `build_decomposer_and_models` and `load_reader` in
  `commaqa.inference.configurable_inference`;
- `BestFirstDecomposer.return_qid_prediction` and the original model controller;
- `MultiParaRCReader` for processed JSONL examples;
- `RetrieveAndResetParagraphsParticipant` for BM25/Elasticsearch retrieval;
- `CopyQuestionParticipant`, `LLMQAParticipantModel`, and `AnswerExtractor` for
  final answer generation;
- `StepByStepCOTGenParticipant` and `StepByStepExitControllerParticipant` for
  C's repeated retrieval/reasoning loop and stopping rules; and
- `DropAnswerEmAndF1` for HotpotQA/2WikiMultiHopQA/MuSiQue and
  `SquadAnswerEmF1Metric` for NQ/TriviaQA/SQuAD.

`run.py` requires exactly one of `--dry-run` or `--execute`. Dry-run does not
load `.env`; execute loads it only inside the runner process. Deterministic
pilot IDs come from `processed_data/<dataset>/dev_subsampled.jsonl`. Unique run
directories under `research/results/pilot/` contain `run_metadata.json`,
`results.jsonl`, and per-question/per-strategy request traces. No existing run
directory is overwritten.

The upstream GPT generator normally performs GPT-2-tokenizer prompt fitting.
The NRP bridge receives the prompt built by the upstream participant directly
and therefore does not reproduce that model-specific fitting step or require a
local tokenizer download. Current GPT-config decoding defaults and the upstream
strategy graph are retained. This difference, plus hosted chat formatting and
quantization, prevents claims of exact GPT/FLAN-T5 reproduction.

Real execution is presently blocked in this checkout by absent processed
datasets, Elasticsearch corpora/indexes, several packages from the upstream
requirements, and the local `en_core_web_sm` spaCy checkpoint used by C. The
offline suite verifies orchestration with mocks only. Before the first real
one-question run, install/prepare those original resources, start the matching
retriever, rerun dry-run, freeze pilot-only IDs, and obtain explicit execution
approval.

Required environment variables and controls are documented in
`research/experiments/pilot/.env.example`. The required secret is
`NRP_API_KEY`; `NRP_API_BASE` is configurable and defaults to the documented
endpoint. Optional `NRP_CACHE_SALT` provides NRP cache isolation and must also
be treated as secret. `.env` is ignored by Git.
