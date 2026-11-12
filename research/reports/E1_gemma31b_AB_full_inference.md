# E1 — Full Gemma 31B A/B inference and held-out evaluation

**Owner:** Tyler Tran. **Status: COMPLETE — all 450 records independently validated.**
Model alias `gemma` resolved to `google/gemma-4-31B-it-qat-w4a16-ct` for every record.
Exactly 225 frozen unique HotpotQA questions were evaluated with A and B. No
Strategy C, Gemma 12B, router/classifier training, resampling, or Test-based tuning
was performed. The four compatible pilot records were reused, so 446 new question–strategy
executions were required. The original pilot files remain unchanged.

## Environment and provenance

- Execution branch: `tyler/gemma31b-formal-ab`. Base commit: `755ac6b49dc2d3c611438bbeec1663564764ac83`.
  Preflight ran immediately before the separate local branch was created; it
  recorded branch `tylertran` at the same commit. Official shared main
  at dataset verification: `8d5009f9dbac38161f5605ac4b4e2f9d8deaf922`.
- Platform: `macOS-15.7.4-arm64-arm-64bit`. Python: `3.9.6 (default, Apr 30 2025, 02:07:17) `.
  Reused `adaptive-rag/.venv`; no dependencies were installed for E1.
- Dependencies: jsonnet 0.22.0, diskcache 5.6.3, nltk 3.9.2, transformers 4.26.0.dev0, requests 2.32.5, torch 1.13.1, spacy 3.4.1, openai 1.5.0, elasticsearch 7.9.1, fastapi 0.103.2, uvicorn 0.39.0, ftfy 6.3.1, rapidfuzz 3.13.0.
- Existing Elasticsearch: `http://127.0.0.1:9201`, OSS 7.10.2, index `hotpotqa`,
  **5,233,329** documents. Existing `hotpotqa-es-restore` container and volumes
  were reused. Replica-only yellow: one active primary and one unassigned replica.
  Existing upstream retriever: `http://localhost:8000/retrieve`.
- NRP: `https://ellm.nrp-nautilus.io/v1/chat/completions`. Preflight confirmed the
  exact resolved model with one tiny connectivity request; all 450 scored
  records independently match it. Credentials remain private and are excluded.
- `preflight.json` records environment, shared GitHub verification, all frozen
  hashes, config/prompt/implementation hashes, and sanitized API evidence.
  `execution_provenance.json` records the executed runner source hash and exact commands; `validation_report.json` records the final evaluator hash.

Dataset root: `research/data/splits/hotpotqa_v1/`. Hashes match metadata and
byte-for-byte files from the fetched official GitHub main; no manifests changed.

| Manifest relative to dataset root | Records | SHA-256 |
|---|---:|---|
| `calibration.jsonl` | 50 | `d387583b768319c91edb832502cfa78349a4957e6596c433b51d8c6179d59d9f` |
| `inference/calibration.jsonl` | 50 | `0917e208cdaea7a429e3f922a12557c93982ccf1b1a1ab83f89df3ae67e9f585` |
| `validation.jsonl` | 25 | `738d6508d57dae28179c039ae7611111ec27c01859ead046104bd7cad09b72d0` |
| `inference/validation.jsonl` | 25 | `bcffb56de3a6b15eb0b138c0c007ce320334b0c1749c634e293058f1e8f5bbe1` |
| `test.jsonl` | 150 | `be530a7228aa8e0964ed70b56d4752f20d63f93ad30f0db769ca7726fdc0d4c2` |
| `inference/test.jsonl` | 150 | `39383a6c70643e2465a65eea19e96714c0695f5a0fe9e7d18e7a1262ddc12c1f` |

The ordered 50/25/150 inference question IDs and text match their evaluation-only
counterparts. Splits are disjoint, with 225 unique IDs and **zero overlap with the
500-question Phase 0 manifest**. Pilot IDs `5a737e4655429908901be2cc` and
`5a746a4455429974ef308bf6` are the first two Calibration entries in frozen order.
The shared metadata's historical candidate-frozen label is preserved; the user
explicitly authorized these existing manifests after live retrieval validation.

## Unchanged experimental protocol

The original upstream A/B graphs, prompts, adapter, answer extraction, and metric
implementation match the pilot source bytes. A performs no retrieval. B uses
single-step BM25, depth 5, `hotpotqa`, `title_paragraph_text`, original-question
query source and global maximum 15 paragraphs. The unchanged upstream retriever
removes WH words, skips paragraphs over 600 words and near duplicates; prompt
formatting keeps up to 350 words per paragraph. Retrieved-versus-included document
IDs and titles are recorded explicitly; depth 5 does not imply every hit must
survive those existing filters.

Generation: max_tokens 300, temperature 0, top_p 1, newline stop, thinking disabled.
GPT-2 cache-only prompt fitting: 3,500-token limit, 300 reserved generation tokens,
triple-newline demonstration delimiter, remove first demonstration, retain the
test example. The original loader warning “Dropped 7 / 20 examples” is retained.
API timeout 300 s, max retries 2, concurrency
1, initial/max retry backoff 1.0/16.0 s with existing
jitter. No model fallback. Existing project-local disk/NLTK/tokenizer caches were
reused; no infrastructure setup, snapshot restore or index rebuild was repeated.

## Completion summary

| Split | A completed | B completed | Expected |
|---|---:|---:|---|
| Calibration | 50 | 50 | 50 per strategy |
| Validation | 25 | 25 | 25 per strategy |
| Test | 150 | 150 | 150 per strategy |
| **Total** | **225** | **225** | **450 records / 225 unique questions** |

Calibration precedes Validation, which precedes Test. Frozen order is preserved
within each split, interleaving A then B. Four pilot records count toward the
Calibration total, with explicit reuse provenance. Each missing combination ran
sequentially with one process under an exclusive lock. Successful checkpoints
were saved incrementally and never replaced. The runner skips completed records
on resume, bounds transient API retries and stops after three consecutive API
failures or immediately on model/protocol/integration mismatch.

## Split-level results

F1 and EM below use the original normalized metric on a 0–1 scale. API latency
is the adapter's request time, including any retry waits. End-to-end latency
additionally includes retrieval, prompt preparation and controller execution.

| Split | Strategy | N | Mean F1 | Mean EM | Mean API latency (s) | Mean end-to-end (s) | Mean retrieval calls | Mean LLM calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Calibration | A | 50 | 0.341200 | 0.200000 | 1.458 | 1.506 | 0.00 | 1.00 |
| Calibration | B | 50 | 0.363000 | 0.240000 | 1.718 | 1.958 | 1.00 | 1.00 |
| Validation | A | 25 | 0.512000 | 0.400000 | 1.419 | 1.469 | 0.00 | 1.00 |
| Validation | B | 25 | 0.432000 | 0.360000 | 1.683 | 1.871 | 1.00 | 1.00 |
| Test | A | 150 | 0.340200 | 0.246667 | 1.635 | 1.681 | 0.00 | 1.00 |
| Test | B | 150 | 0.335267 | 0.246667 | 1.835 | 1.987 | 1.00 | 1.00 |

## Official Held-out Test Results — Gemma 31B (n = 150)

**Only the 150 frozen Test IDs** contribute to each official row. Calibration,
Validation and pilot records are excluded from these averages. No Test F1/EM
was used to train, calibrate, tune or select any router or experimental setting.
The inference process never reads gold references; evaluation ran separately
after all 450 generation checkpoints were present.

| Strategy | Test N | Mean F1 | Mean EM | Mean API latency (s) | Mean end-to-end (s) | Mean retrieval calls | Total LLM calls | Input / output / total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | 150 | 34.02% | 24.67% | 1.635 | 1.681 | 0.00 | 150 | 223007 / 7258 / 230265 |
| B | 150 | 33.53% | 24.67% | 1.835 | 1.987 | 1.00 | 150 | 463377 / 8958 / 472335 |

The mean Test F1 denominator is exactly 150 per strategy. The same 150 IDs are
used for Test EM, latency, retrieval and resource metrics. These are fixed
baselines, not router or Strategy C results. No monetary cost was inferred.

## Execution statistics

| Split | Strategy | N | LLM calls | HTTP attempts | Retries | Failed executions | Input / output / total tokens | Total API latency (s) |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Calibration | A | 50 | 50 | 50 | 0 | 0 | 74396 / 2796 / 77192 | 72.888 |
| Calibration | B | 50 | 50 | 50 | 0 | 0 | 152784 / 3318 / 156102 | 85.890 |
| Validation | A | 25 | 25 | 25 | 0 | 0 | 37170 / 1291 / 38461 | 35.463 |
| Validation | B | 25 | 25 | 25 | 0 | 0 | 76877 / 1558 / 78435 | 42.076 |
| Test | A | 150 | 150 | 151 | 1 | 0 | 223007 / 7258 / 230265 | 245.286 |
| Test | B | 150 | 150 | 150 | 0 | 0 | 463377 / 8958 / 472335 | 275.201 |

Across the included 450 records: **450 LLM calls**,
451 HTTP attempts, 1 retries,
1027611 input tokens, 25179 output tokens,
**1052790 total tokens**, 756.805 s API latency
and 806.916 s summed end-to-end latency.
Failed executions: 0; failed formal HTTP attempts observed:
1. These latency sums include original pilot
latencies for reused records and are not uninterrupted wall-clock duration.

One Test A request (`5abf64265542997ec76fd3eb`) returned HTTP 500 and
succeeded on the existing bounded retry. No separate failed execution remained.
Four A responses ended with `finish_reason=length` at the preserved 300-token
limit: Calibration `5a7c8e7955429935c91b521a`, Validation
`5ab93781554299131ca422c7`, and Test `5ac2a288554299657fa28ff5` /
`5a7b1e22554299042af8f6e4`. These are valid protocol-limited API completions,
scored exactly as generated; no token limit was increased and none were rerun.
All B responses ended with `stop`. Token totals reflect reported successful
responses; no token usage was reported for the failed HTTP 500 attempt.

The separate E1 connectivity request used 17 input /
2 output tokens and 0.677 s; it is
excluded from inference counts and metrics. Raw per-attempt ledgers preserve
formal HTTP outcomes without recording authorization headers or credentials.

## Independent validation and leakage boundary

The separate evaluator requires all 450 unique checkpoints before loading any
gold evaluation JSONL. It checks all manifest hashes, split assignments, paired
IDs, unexpected/missing/duplicate keys, model identities, token sums, attempts,
latencies and retrieval counts. It independently derives the final answer from
the saved raw API text through the original extraction behavior, then recomputes
EM/F1 directly with `DropAnswerEmAndF1`, independently of runner scores. Reused
pilot scores are identical to their original values. All aggregate denominators
and statistics come from the exported per-question rows.

For every record, the evaluator reconstructs the **entire original and fitted
prompt exactly** from the hashed fixed demonstration template, inference-safe
question and actual retrieved passages. A has no test-question context. B's
context contains only upstream-selected retrieval results with the unchanged
formatting/truncation. Input-example snapshots contain no answer/gold contexts,
oracle labels or scores. This provenance check is stronger than searching for
answer strings, which may legitimately occur in the question or retrieved text.
Gold supporting paragraphs were never supplied from evaluation files. Standard
fixed few-shot demonstration answers remain part of the unchanged protocol.

No previous output or Docker volume was overwritten. Frozen and pilot hashes
were rechecked. `validation_report.json` records COMPLETE, empty missing and
duplicate lists, and export hashes; `raw_evidence_hashes.json` inventories the
locally retained evidence. The offline unit suite passed **31 tests**, including
immutable checkpoints, A-context rejection, B evidence/depth requirements,
evaluation refusal before complete generation, and non-overwriting exports.

## Handoff artifacts and exact commands

All result paths below are relative to `research/results/hotpotqa_v1/gemma31b/`:

- `predictions_A.jsonl`: 225 A predictions with original IDs, references, EM/F1 and resources.
- `predictions_B.jsonl`: 225 B predictions with IDs, references, EM/F1, retrieval document IDs/titles and resources.
- `calibration_AB.jsonl`: 100 rows, 50 paired questions; suitable for later joining with Strategy C and router calibration.
- `validation_AB.jsonl`: 50 rows, 25 paired questions.
- `test_AB.jsonl`: 300 rows, 150 paired questions; reserved for held-out evaluation, not router training.
- `summary_metrics.json`: split-specific n, F1/EM, latency, calls, tokens, failures/retries and official Test sample size.
- `preflight.json`, `execution_provenance.json`, `validation_report.json`, `compact_validation.json`: reproducibility and independent validation.
- `raw_evidence_hashes.json`: local raw evidence inventory; raw traces themselves are not part of Git handoff.
- `checkpoints/<split>/<question_id>_<strategy>.json`: immutable resumable generation records (local only).
- `raw/<split>/<question_id>_<strategy>/<attempt>/`: fitted/original prompts, API responses, HTTP-attempt ledgers, retrieval traces, input snapshots and chains (local only).
- `raw/pilot_reuse/`: byte-preserved copies of the compatible pilot evidence (local only).
- `events.jsonl`, `run.lock`, and any `failures.jsonl`: local execution state, excluded from Git.

Scripts live in `research/experiments/formal/`:

```sh
# Original execution; preflight refuses to overwrite its existing evidence.
adaptive-rag/.venv/bin/python research/experiments/formal/gemma31b_ab.py preflight
# Safe resume; skips complete checkpoints and makes no request when all 450 exist.
adaptive-rag/.venv/bin/python research/experiments/formal/gemma31b_ab.py run
# Separate offline evaluation; existing identical exports are checked, not overwritten.
PYTHONPYCACHEPREFIX=.pilot-cache/pycache adaptive-rag/.venv/bin/python research/experiments/formal/evaluate_gemma31b_ab.py
PYTHONPYCACHEPREFIX=.pilot-cache/pycache adaptive-rag/.venv/bin/python -m unittest -q research.experiments.formal.test_gemma31b_ab research.experiments.pilot.test_nrp_adapter research.experiments.pilot.test_pipeline
PYTHONPYCACHEPREFIX=.pilot-cache/pycache adaptive-rag/.venv/bin/python research/experiments/formal/validate_gemma31b_exports.py --write-report
python3 research/experiments/formal/build_gemma31b_handoff.py
```

Reproducing fresh inference requires a new dedicated output location; the
existing directory is a completed immutable experiment, not a place to rerun it.
Offline full evidence verification requires the locally retained raw directory.
The portable `validate_gemma31b_exports.py` command checks the compact files without raw traces. The compact per-question files and manifests suffice for downstream joins and
independent EM/F1 recomputation; raw data can be shared separately if needed.

## Paper integration and safe transfer

Shared Table 0/Table 4 and the generated paper PDF were deliberately left
unchanged to avoid overwriting simultaneous lead/Suzy work. An exact proposed
patch is supplied at `proposed_table_updates.patch` and below. Table 0 adds a
completed Gemma 31B A/B milestone attributed to Tyler Tran. Table 4 fills only
the Gemma 31B Always-A/B rows using Test n=150, with end-to-end latency and no
invented dollar cost. Strategy C/original-router/KNN cells remain pending. This
patch targets the local table schema; review/rebase it against newer shared
tables rather than choosing an entire older table. No Phase 0 or unrelated table
is modified. No LaTeX compiler (`latexmk`/`pdflatex`) was available, so compilation
was not performed.

```diff
--- a/research/paper/tables/table0_progress.tex
+++ b/research/paper/tables/table0_progress.tex
@@ -21,6 +21,7 @@
 E0.3 & Oracle upper-bound analysis & \textit{Analytical only:} retrospective best-F1 tie sets, quality, and regret; never used for test routing & T2a & Completed & Suzy Huang \\
 E0.4 & Audit silver labels, ties, and F1 regret & \textit{Direct:} provenance, overlap, agreement, and matched-subset regret analysis & T2b & Completed & Suzy Huang \\
 \midrule
+E1.AB & Gemma 31B frozen A/B inference & \textit{Direct:} 450 validated records on 225 frozen questions; held-out A/B baselines ($n=150$ each) & T4; E1 report & Completed (A/B only) & Tyler Tran \\
 E1.1 & Profile retrieval gains for LLM 1 & \textit{Direct:} per-question $F1_B-F1_A$ and $F1_C-F1_A$ & T3 & Completed (Phase~0) & Suzy Huang \\
 E1.2 & Profile retrieval gains for LLM 2 & \textit{Direct:} per-question $F1_B-F1_A$ and $F1_C-F1_A$ & T3 & Completed (Phase~0) & Suzy Huang \\
 E1.3 & Diagnose probability-space locality & \textit{Direct:} classifier-probability distance versus retrieval-gain similarity & T3; F2 & Planned & \\
--- a/research/paper/tables/table4_main_results.tex
+++ b/research/paper/tables/table4_main_results.tex
@@ -17,14 +17,14 @@
 & Original Adaptive-RAG & \NA & \NA & \NA & \NA & \NA \\
 & \ours{} (distance-weighted KNN) & \NA & \NA & \NA & \NA & \NA \\
 \midrule
-\multirow{5}{*}{LLM 2}
-& No Retrieval & \NA & \NA & \NA & \NA & \NA \\
-& Always Single-step Retrieval & \NA & \NA & \NA & \NA & \NA \\
+\multirow{5}{*}{Gemma 31B}
+& No Retrieval & 34.02 & 24.67 & 0.00 & 1.681 & \NA \\
+& Always Single-step Retrieval & 33.53 & 24.67 & 1.00 & 1.987 & \NA \\
 & Always Multi-step Retrieval & \NA & \NA & \NA & \NA & \NA \\
 & Original Adaptive-RAG & \NA & \NA & \NA & \NA & \NA \\
 & \ours{} (distance-weighted KNN) & \NA & \NA & \NA & \NA & \NA \\
 \bottomrule
 \end{tabular}
 \end{adjustbox}
-\TableNote{Pending: dataset, held-out split, and unique-question $N$ are not yet frozen; no displayed row has results. Future rows within an LLM block must use the identical held-out IDs and report that common $N$; question counts must remain distinct from method inference records. The Oracle is excluded as non-deployable. Cost scope, price date, and latency protocol must be fixed before reporting.}
+\TableNote{Gemma 31B Always-A/B use the identical 150 frozen HotpotQA v1 held-out test IDs ($n=150$ per strategy), excluding calibration and validation. Latency is recorded client end-to-end time including retrieval and prompt processing; cost is not monetized. Other methods and LLM blocks remain pending. Phase~0 results are not changed. Oracle and held-out labels are never used for routing or calibration.}
 \end{table}
```

The isolated local branch is `tyler/gemma31b-formal-ab`. Only the explicit compact
handoff allowlist is eligible for a local commit. Credentials, root `env.example`,
`.env` files, raw traces, checkpoints, runtime state, caches, environments, model
weights and the Elasticsearch archive/volumes remain excluded. No Git push,
PR creation, merge or repository-permission change was performed for E1.

Recommended transfer: review the local formal commit, then share the prepared
`.pilot-cache/E1_gemma31b_AB_handoff.bundle` and its SHA-256 with the project lead
through the team's approved storage. It contains only the formal commit on top
of the previously shared pilot commit; raw inference state is excluded. The
lead can verify and fetch without force or overwriting shared branches:

```sh
git bundle verify /path/E1_gemma31b_AB_handoff.bundle
git fetch /path/E1_gemma31b_AB_handoff.bundle tyler/gemma31b-formal-ab:tyler/gemma31b-formal-ab-review
```

Any later push or PR should follow explicit authorization and review of the
compact file list. **Final status: COMPLETE, 450/450 validated; no missing records.**
