"""Build E1 report and a proposed table patch from independently validated exports."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'research/results/hotpotqa_v1/gemma31b'
REPORT=ROOT/'research/reports/E1_gemma31b_AB_full_inference.md'


def save(path,text):
    if path.exists():
        assert path.read_text()==text, 'Refuse to overwrite different handoff: '+str(path)
    else:
        with path.open('x') as f:f.write(text)


def build():
    summary=json.loads((OUT/'summary_metrics.json').read_text())
    validation=json.loads((OUT/'validation_report.json').read_text())
    meta=json.loads((OUT/'preflight.json').read_text())
    execution=json.loads((OUT/'execution_provenance.json').read_text())
    assert summary['status']==validation['status']=='COMPLETE' and summary['valid_records']==450
    test=summary['splits']['test']
    table0=ROOT/'research/paper/tables/table0_progress.tex'
    table4=ROOT/'research/paper/tables/table4_main_results.tex'
    # Supply a separate patch: never edit the lead's shared scientific tables.
    original0=table0.read_text()
    lines0=original0.splitlines(keepends=True)
    position=next(i for i,line in enumerate(lines0) if line.startswith('E1.1 &'))
    lines0.insert(position,'E1.AB & Gemma 31B frozen A/B inference & \\textit{Direct:} 450 validated records on 225 frozen questions; held-out A/B baselines ($n=150$ each) & T4; E1 report & Completed (A/B only) & Tyler Tran \\\\\n')
    original4=table4.read_text()
    lines4=original4.splitlines(keepends=True)
    start=next(i for i,line in enumerate(lines4) if 'multirow{5}{*}{LLM 2}' in line)
    lines4[start]='\\multirow{5}{*}{Gemma 31B}\n'
    for offset,strategy,label in [(1,'A','No Retrieval'),(2,'B','Always Single-step Retrieval')]:
        x=test[strategy]
        # End-to-end latency includes retrieval and prompt processing.
        lines4[start+offset]=f"& {label} & {x['mean_f1']*100:.2f} & {x['mean_em']*100:.2f} & {x['mean_retrieval_calls']:.2f} & {x['mean_end_to_end_latency_seconds']:.3f} & \\NA \\\\\n"
    note=next(i for i,line in enumerate(lines4) if line.startswith('\\TableNote{'))
    lines4[note]='\\TableNote{Gemma 31B Always-A/B use the identical 150 frozen HotpotQA v1 held-out test IDs ($n=150$ per strategy), excluding calibration and validation. Latency is recorded client end-to-end time including retrieval and prompt processing; cost is not monetized. Other methods and LLM blocks remain pending. Phase~0 results are not changed. Oracle and held-out labels are never used for routing or calibration.}\n'
    patch=''.join(difflib.unified_diff(original0.splitlines(keepends=True),lines0,fromfile='a/research/paper/tables/table0_progress.tex',tofile='b/research/paper/tables/table0_progress.tex'))
    patch+=''.join(difflib.unified_diff(original4.splitlines(keepends=True),lines4,fromfile='a/research/paper/tables/table4_main_results.tex',tofile='b/research/paper/tables/table4_main_results.tex'))
    save(OUT/'proposed_table_updates.patch',patch)
    coverage='\n'.join(f"| {split.title()} | {summary['splits'][split]['A']['n']} | {summary['splits'][split]['B']['n']} | {n} per strategy |" for split,n in [('calibration',50),('validation',25),('test',150)])
    results=[]
    for split,strategies in summary['splits'].items():
        for code,x in strategies.items():
            results.append(f"| {split.title()} | {code} | {x['n']} | {x['mean_f1']:.6f} | {x['mean_em']:.6f} | {x['mean_latency_seconds']:.3f} | {x['mean_end_to_end_latency_seconds']:.3f} | {x['mean_retrieval_calls']:.2f} | {x['mean_llm_api_calls']:.2f} |")
    official=[]
    resources=[]
    for code,x in test.items():
        official.append(f"| {code} | {x['n']} | {x['mean_f1']*100:.2f}% | {x['mean_em']*100:.2f}% | {x['mean_latency_seconds']:.3f} | {x['mean_end_to_end_latency_seconds']:.3f} | {x['mean_retrieval_calls']:.2f} | {x['total_api_calls']} | {x['total_input_tokens']} / {x['total_output_tokens']} / {x['total_tokens']} |")
    for split,strategies in summary['splits'].items():
        for code,x in strategies.items():
            resources.append(f"| {split.title()} | {code} | {x['n']} | {x['total_api_calls']} | {x['total_http_attempts']} | {x['total_retries']} | {x['failed_executions']} | {x['total_input_tokens']} / {x['total_output_tokens']} / {x['total_tokens']} | {x['total_latency_seconds']:.3f} |")
    hashes='\n'.join(f"| `{name}` | {value['records']} | `{value['sha256']}` |" for name,value in meta['manifests'].items())
    deps=', '.join(f"{name} {version}" for name,version in meta['dependencies'].items())
    stats=summary['execution_statistics']
    text=f'''# E1 — Full Gemma 31B A/B inference and held-out evaluation

**Owner:** Tyler Tran. **Status: COMPLETE — all 450 records independently validated.**
Model alias `gemma` resolved to `{summary['model']}` for every record.
Exactly 225 frozen unique HotpotQA questions were evaluated with A and B. No
Strategy C, Gemma 12B, router/classifier training, resampling, or Test-based tuning
was performed. The four compatible pilot records were reused, so 446 new question–strategy
executions were required. The original pilot files remain unchanged.

## Environment and provenance

- Execution branch: `{execution['branch']}`. Base commit: `{execution['base_commit']}`.
  Preflight ran immediately before the separate local branch was created; it
  recorded branch `{meta['branch']}` at the same commit. Official shared main
  at dataset verification: `{meta['shared_main']}`.
- Platform: `{meta['platform']}`. Python: `{meta['python'].splitlines()[0]}`.
  Reused `adaptive-rag/.venv`; no dependencies were installed for E1.
- Dependencies: {deps}.
- Existing Elasticsearch: `http://127.0.0.1:9201`, OSS 7.10.2, index `hotpotqa`,
  **5,233,329** documents. Existing `hotpotqa-es-restore` container and volumes
  were reused. Replica-only yellow: one active primary and one unassigned replica.
  Existing upstream retriever: `http://localhost:8000/retrieve`.
- NRP: `{meta['adapter']['api_base']}/chat/completions`. Preflight confirmed the
  exact resolved model with one tiny connectivity request; all 450 scored
  records independently match it. Credentials remain private and are excluded.
- `preflight.json` records environment, shared GitHub verification, all frozen
  hashes, config/prompt/implementation hashes, and sanitized API evidence.
  `execution_provenance.json` records the executed runner source hash and exact commands; `validation_report.json` records the final evaluator hash.

Dataset root: `research/data/splits/hotpotqa_v1/`. Hashes match metadata and
byte-for-byte files from the fetched official GitHub main; no manifests changed.

| Manifest relative to dataset root | Records | SHA-256 |
|---|---:|---|
{hashes}

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
API timeout {meta['adapter']['timeout_seconds']:.0f} s, max retries {meta['adapter']['max_retries']}, concurrency
{meta['adapter']['max_concurrency']}, initial/max retry backoff {meta['adapter']['backoff_initial_seconds']}/{meta['adapter']['backoff_max_seconds']} s with existing
jitter. No model fallback. Existing project-local disk/NLTK/tokenizer caches were
reused; no infrastructure setup, snapshot restore or index rebuild was repeated.

## Completion summary

| Split | A completed | B completed | Expected |
|---|---:|---:|---|
{coverage}
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
{chr(10).join(results)}

## Official Held-out Test Results — Gemma 31B (n = 150)

**Only the 150 frozen Test IDs** contribute to each official row. Calibration,
Validation and pilot records are excluded from these averages. No Test F1/EM
was used to train, calibrate, tune or select any router or experimental setting.
The inference process never reads gold references; evaluation ran separately
after all 450 generation checkpoints were present.

| Strategy | Test N | Mean F1 | Mean EM | Mean API latency (s) | Mean end-to-end (s) | Mean retrieval calls | Total LLM calls | Input / output / total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(official)}

The mean Test F1 denominator is exactly 150 per strategy. The same 150 IDs are
used for Test EM, latency, retrieval and resource metrics. These are fixed
baselines, not router or Strategy C results. No monetary cost was inferred.

## Execution statistics

| Split | Strategy | N | LLM calls | HTTP attempts | Retries | Failed executions | Input / output / total tokens | Total API latency (s) |
|---|---|---:|---:|---:|---:|---:|---|---:|
{chr(10).join(resources)}

Across the included 450 records: **{stats['total_included_api_calls']} LLM calls**,
{stats['total_included_http_attempts']} HTTP attempts, {stats['retries_in_successful_records']} retries,
{stats['input_tokens']} input tokens, {stats['output_tokens']} output tokens,
**{stats['total_tokens']} total tokens**, {stats['total_recorded_api_latency_seconds']:.3f} s API latency
and {stats['total_recorded_end_to_end_latency_seconds']:.3f} s summed end-to-end latency.
Failed executions: {stats['failed_executions']}; failed formal HTTP attempts observed:
{stats['failed_http_attempts_observed']}. These latency sums include original pilot
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

The separate E1 connectivity request used {meta['connectivity']['input_tokens']} input /
{meta['connectivity']['output_tokens']} output tokens and {meta['connectivity']['latency_seconds']:.3f} s; it is
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
{patch.rstrip()}
```

The isolated local branch is `{execution['branch']}`. Only the explicit compact
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
'''
    save(REPORT,text)
    print('Wrote',REPORT.relative_to(ROOT),'and exact proposed table patch; shared tables unchanged.')


if __name__=='__main__':
    build()
