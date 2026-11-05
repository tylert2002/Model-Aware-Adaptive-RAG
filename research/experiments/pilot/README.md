# NRP Pilot Adapter

## Experiment status

- Experiment name: Pilot Model and Pipeline Validation
- Research question: Can the NRP-hosted candidate models run the intended A/B/C pipeline reliably before formal Phase 0 evaluation?
- Related LaTeX table: P1
- Expected outputs: API compatibility, preliminary A/B/C F1, success/failure statistics, and runtime and token-usage summaries.
- Status: API Adapter and Pilot Runner Ready; Real End-to-End Execution Pending Resources and Approval

This directory contains research-owned integration code for the NRP-hosted `gemma-small` and `gemma` aliases. It does not modify or replace the original Adaptive-RAG retrieval, prompting, controller, or evaluation algorithms.

## Verified NRP contract

As of 2026-11-05, the official NRP documentation verifies:

- API base: `https://ellm.nrp-nautilus.io/v1`
- Chat endpoint: `POST /chat/completions`
- Authentication: `Authorization: Bearer <token>`
- OpenAI Python compatibility: `OpenAI(api_key=..., base_url=...)` and `client.chat.completions.create(...)`
- Candidate API aliases: `gemma-small` and `gemma`
- Both aliases accept the documented thinking toggle `chat_template_kwargs.enable_thinking`; reasoning is on by default.
- `gemma` is a generally supported 31B deployment; `gemma-small` is an evaluating 12B deployment whose availability/configuration may change.
- Current fair-use limits for short requests are 8 concurrent requests per user for these models and 48,000 output tokens/minute per API token and model. Requests using at least 35% of model context are limited to one concurrent request.

Sources: [API access](https://nrp.ai/documentation/userdocs/ai/llm-managed/api-access/), [available models](https://nrp.ai/documentation/userdocs/ai/llm-managed/models/), and [fair use](https://nrp.ai/documentation/userdocs/ai/llm-managed/fair-use/).

The documentation does not enumerate deployment-specific support for every generic OpenAI decoding parameter, response-side token accounting, response headers, or latency fields. The adapter therefore requires only `model` and `messages`, treats `max_tokens` as optional, sends temperature/top-p/stop only when explicitly requested, measures latency client-side, and records token usage only when returned. Exact behavior remains pending the authenticated smoke test.

## Why an adapter is required

Adaptive-RAG's local path calls a custom `GET /generate` endpoint and expects `generated_texts`, `generated_num_tokens`, `run_time_in_seconds`, and a whitelisted local model name. Its legacy OpenAI path uses `openai.Completion.create` rather than chat completions. Neither directly accepts NRP's current chat-completions contract.

`nrp_adapter.py` supplies one prompt-to-text interface for A/B/C and request metadata. `adaptive_rag_bridge.py` exposes the original generator method shape, `generate_text_sequence(prompt)`. `pipeline.py` injects that bridge into the instantiated upstream generator objects without changing upstream source. `run.py` provides the guarded command-line entry point. Retrieval depth, stopping rules, prompts, corpora, splits, and evaluation remain controlled by the original pipeline.

## Upstream execution reused by the runner

The strategy mapping is fixed and checked at load time:

| Strategy | Upstream config | Expected start state |
|---|---|---|
| A — No Retrieval | `nor_qa_gpt_<dataset>.jsonnet` | `generate_question` |
| B — Single-step Retrieval | `oner_qa_gpt_<dataset>.jsonnet` | `generate_titles` |
| C — Multi-step Retrieval | `ircot_qa_gpt_<dataset>.jsonnet` | `step_by_step_bm25_retriever` |

The runner reuses `build_decomposer_and_models` and `load_reader` from `commaqa.inference.configurable_inference`, `BestFirstDecomposer.return_qid_prediction`, and `MultiParaRCReader`. Strategy A follows `CopyQuestionParticipant` to `LLMQAParticipantModel` and `AnswerExtractor`. Strategy B first calls `RetrieveAndResetParagraphsParticipant`, then follows the same answer path. Strategy C retains the original loop among `RetrieveAndResetParagraphsParticipant`, `StepByStepCOTGenParticipant`, and `StepByStepExitControllerParticipant` before the final answer path. Retrieval still uses the upstream Elasticsearch `/retrieve` contract and corpus checks.

For evaluation, HotpotQA, 2WikiMultiHopQA, and MuSiQue use the upstream `DropAnswerEmAndF1`; NQ, TriviaQA, and SQuAD use `SquadAnswerEmF1Metric`. This preserves the original dataset-specific numeric and multi-span handling rather than forcing one metric across all datasets.

The only generation-backend substitution is the injected NRP object. The upstream graph still constructs prompts and C still uses the upstream spaCy sentence segmentation and exit rules. The NRP bridge carries the current GPT-config decoding defaults (`max_tokens=300`, temperature 0, top-p 1, newline stop, thinking disabled). It does not run the legacy GPT-2 tokenizer-based prompt-fitting step because that would require a local tokenizer/model download; prompt text produced by the upstream participants is sent directly to NRP. This backend difference must be recorded and means the pilot is a compatibility study, not an exact reproduction of GPT or FLAN-T5 outputs.

## Environment and secure setup

Copy the example locally and edit only the ignored `.env` file:

```sh
cd research/experiments/pilot
cp .env.example .env
```

Set `NRP_API_KEY` to the NRP token. Never place it in a command line, source file, notebook, result artifact, or Git-tracked file. `NRP_API_BASE` defaults to the documented endpoint. `NRP_CACHE_SALT` is optional but recommended by NRP for cache isolation; treat it as a secret too.

This project deliberately does not auto-load `.env`. Export it into the current shell when ready:

```sh
set -a
source .env
set +a
```

## Smoke test

The script refuses network access unless the explicit confirmation flag is present. After configuring the environment, run:

```sh
python3 smoke_test.py --confirm-network
```

It sends one sequential request to each exact alias, validates the chat response shape, and prints sanitized JSON containing model, request status, client-observed latency, attempts, finish reason, and token usage when supplied. It never prints the API key or generated response text.

Offline verification requires no credentials or network:

```sh
python3 -m unittest -v \
  research.experiments.pilot.test_nrp_adapter \
  research.experiments.pilot.test_pipeline
python3 -m py_compile research/experiments/pilot/*.py
```

## Pilot runner

Run preflight from the repository root. It never loads `.env` and never makes an API request:

```sh
python -m research.experiments.pilot.run \
  --model gemma-small \
  --dataset hotpotqa \
  --limit 1 \
  --seed 100 \
  --dry-run
```

Only `--execute` loads `research/experiments/pilot/.env` and permits inference. A real run processes the same deterministically selected question IDs through A, B, and C sequentially. Each run receives a unique timestamp/UUID directory under `research/results/pilot/`; existing artifacts are never overwritten. `run_metadata.json` records configuration and provenance, `results.jsonl` stores one question/strategy record, and `traces/` stores sanitized request accounting. Credentials and authorization headers are never written.

The default pilot input is `adaptive-rag/processed_data/<dataset>/dev_subsampled.jsonl`, deliberately not the final held-out test file. Before formal work, freeze the selected pilot IDs and ensure they are excluded from every final test split.

## Required local resources and remaining work

Faithful execution requires:

- the original Python dependencies in `adaptive-rag/requirements.txt`, including Jsonnet, spaCy, the local `en_core_web_sm` package, retrieval dependencies, and metric dependencies;
- `processed_data/<dataset>/dev_subsampled.jsonl` in the upstream tree;
- the original prompts and all three GPT strategy configs for the selected dataset;
- `.retriever_address.jsonnet`, a running upstream retriever service, and the matching Elasticsearch corpus/index (`hotpotqa`, `2wikimultihopqa`, `musique`, or `wiki` for NQ/TriviaQA/SQuAD);
- the ignored `.env` file only when `--execute` is intentionally used.

This checkout currently lacks the processed datasets and several upstream Python dependencies, so real end-to-end execution has not been verified. The offline tests use mock responses and synthetic temporary records solely to verify software behavior; they are not research results. After resources are installed without changing upstream algorithms, run dry-run again, confirm the retriever/index, review the frozen IDs and decoding configuration, and request explicit approval before the first one-question execution.

## Pilot safeguards

Use identical pilot-only questions for both models and A/B/C; do not overlap the formal held-out test split. Never use the `service` alias because it can fall back to a different model.

Chat formatting and hosted quantization differ from the original FLAN-T5/GPT backends, so this can validate pipeline feasibility but cannot reproduce original downstream-model results exactly. Multi-step strategy C makes repeated LLM calls, magnifying latency, transient-failure, and nondeterminism risks. Preserve all answers, errors, usage, latency, model alias, configuration, and request identifiers.
