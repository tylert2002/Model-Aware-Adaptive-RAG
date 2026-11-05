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
