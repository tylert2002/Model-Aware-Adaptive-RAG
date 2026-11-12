# Gemma 31B A/B pilot handoff

Only the verified two-question pilot is complete. This submission contains four
new real inference records from the saved November 12, 2026 UTC run. It does not
claim completion of the formal 225-question / 450-record experiment. No model
inference, retrieval, or Elasticsearch operation was run during Git synchronization.

| Split | Pilot A | Pilot B | Formal A completed/target | Formal B completed/target |
|---|---:|---:|---:|---:|
| Calibration | 2 | 2 | 0/50 | 0/50 |
| Validation | 0 | 0 | 0/25 | 0/25 |
| Held-out Test | 0 | 0 | 0/150 | 0/150 |
| Total | 2 | 2 | 0/225 | 0/225 |

The pilot IDs, in frozen order, are `5a737e4655429908901be2cc` and
`5a746a4455429974ef308bf6`. Both are in the frozen Calibration manifest.
Previously tracked 500-question historical analysis exports are unchanged and
are not evidence that this formal experiment was completed.

| Strategy | Mean EM | Mean F1 | Retrieval calls | API requests / attempts | Input / output tokens | Total API / end-to-end seconds |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.50 | 0.50 | 0 | 2 / 2 | 2,967 / 137 | 4.662 / 4.742 |
| B | 0.50 | 0.665 | 2 | 2 / 2 | 6,102 / 150 | 4.005 / 4.933 |

B's incorrect first answer receives 0.33 F1 from the original scorer's word
overlap with “Cashin' In.” The second question has EM/F1 1 for both strategies.
This tiny pilot supports pipeline validation only.

## Artifacts available to the project lead

Run directory: [`../results/pilot/20261112T054914Z_gemma_b89bcbfd/`](../results/pilot/20261112T054914Z_gemma_b89bcbfd/).

- [`predictions.jsonl`](../results/pilot/20261112T054914Z_gemma_b89bcbfd/predictions.jsonl):
  four compact rows keyed by `question_id` and `strategy`, including predictions,
  references, F1, EM, model, split, status, token usage, latency, attempts, and
  retrieved document IDs/titles. These are accessible for later KNN integration;
  they cover only two IDs, so they cannot supply the complete KNN training set.
- `summary_metrics.json`: recomputed aggregate metrics and explicit coverage.
- `run_metadata.json`: original execution commit, config/prompt hashes, frozen
  inference manifest hash, model and decoding provenance. Its inherited stop
  metadata escapes a newline as literal backslash-n; execution used a newline.
  `validation_summary.json` preserves that clarification.
- `local_evidence_manifest.json`: hashes/sizes of the original locally retained
  responses, retrieval traces and fitted prompts. All five retrieved passages
  appeared verbatim in each B fitted test-question context.
- [`../../PILOT_REPORT_GEMMA_31B.md`](../../PILOT_REPORT_GEMMA_31B.md): complete
  report, live environment verification from the pilot, and historical diagnosis.
- [`run_frozen_ab.py`](../results/pilot/gemma31b_assignment_20261111/run_frozen_ab.py):
  scoped wrapper around the unchanged upstream A/B graphs, generator bridge,
  fitting and scorer. It enforces the existing pair and frozen manifest hashes.
- `serve_restored_retriever.py`, `verify_restored_index.py`, and the selected
  successful connectivity/index evidence under `gemma31b_assignment_20261111/`.
  The index verifier now reads the delivered manifest next to that directory
  rather than requiring a machine-specific Downloads location.

## Validation and protocol

Saved responses resolve `google/gemma-4-31B-it-qat-w4a16-ct` through alias `gemma`.
Each record is complete, unique, and paired on both strategies. Model-facing
input comes from the frozen inference-safe Calibration manifest: no gold
answers or dataset contexts are supplied to the model. References are joined
only after generation using the original reader and `DropAnswerEmAndF1` scorer.
Retrieved passages supply B's test-question context. Existing demonstrations,
answer extraction and prompt fitting are unchanged.

The runner retains max_tokens 300, temperature 0, top_p 1, newline stop,
thinking disabled, original GPT-2 prompt fitting, BM25 depth 5 and HotpotQA
corpus. Runtime repairs only redirected disk caches into `.pilot-cache` and
loaded the standard NLTK stopwords resource locally. No experimental settings,
frozen data, shared tables, or PDF were changed.

Offline checks, from the repository root, with the existing compatible environment:

```sh
adaptive-rag/.venv/bin/python research/results/pilot/gemma31b_assignment_20261111/validate_saved_pilot.py
adaptive-rag/.venv/bin/python -m unittest -q research.experiments.pilot.test_nrp_adapter research.experiments.pilot.test_pipeline
```

Both passed: four artifact records validated and 26 unit tests passed. The local
validation additionally passed `--local-evidence`, checking raw evidence hashes,
API status/attempts, compact-to-original equality and B passage inclusion.
These commands send no API inference requests. The portable validator can run
from the PR checkout without excluded raw files. Frozen IDs/hashes, config and
prompt provenance, original-score recomputation, duplicates, completion counts,
token sums and all aggregate metrics are checked.

To reproduce the inference later (not part of synchronization), provide local
authorized credentials using the existing environment/configuration loader,
the cache-only GPT-2 tokenizer and NLTK stopwords resource, and the already
restored index/retriever. Then run `run_frozen_ab.py` above. It makes four paid
or quota-consuming requests and creates a new directory. Do not use the generic
runner unchanged: it selects questions and includes Strategy C. No reproduction
inference was run for this PR.

## Git audit and exclusions

Initially `main` tracked Tyler's fork (`origin`, `tylert2002/...`) at `f692efa`,
with no ahead commits and no tracked modifications. Untracked pilot artifacts,
instructions, credential-bearing local configuration and snapshot delivery were
preserved. No other member's tracked edits were present. The official repository
was added as `team`, preserving `origin`; official `main` and existing
`tylertran` both pointed to `8d5009f`. The two newer main commits only delete and
re-add README with no final tree difference from the local base. A local
`tylertran` tracking branch was created from `team/tylertran` and reused for
this submission. Official push permission was verified. No changes were made to
repository permissions or ownership.

The shared `table0_progress.tex`, `table1_datasets.tex`, and
`table4_main_results.tex` match the fetched official main and are excluded from
this change. `main.pdf` is ignored/untracked. Suzy's results are not overwritten;
the report's pilot metrics are proposed information for review, not formal
table replacements.

Intentionally excluded and preserved locally: root `env.example` containing
credentials; snapshot archive (~1.94 GB), volumes and runtime state; environments
and caches; raw prompts, reasoning, retrieval logs and original inference files;
failed pre-inference attempt; historical troubleshooting diagnostics and local
assignment instructions. The archive is ignored explicitly and its provenance
is retained in the small delivery manifest/checksum. No secrets or private
configuration values are included. Only selected small evidence is committed.
The large corpus and raw traces require separate sharing if later requested.

No full inference, Strategy C, alternate model, Elasticsearch rebuild, force
push, branch deletion, or automatic merge is part of this handoff.
