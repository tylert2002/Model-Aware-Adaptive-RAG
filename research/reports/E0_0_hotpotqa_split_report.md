# E0.0 — HotpotQA split freeze report

## Recommendation: NEEDS REVIEW

The deterministic candidate manifests pass count, schema, uniqueness,
disjointness, Phase 0 exclusion, and reproducibility checks. They are not marked
fully ready because the configured local Elasticsearch service was unavailable,
so live evidence retrieval from the `hotpotqa` index was not verified. Do not
start Robbie or Tyler's inference until that check and user approval are complete.

## Source and provenance

- Source archive: `adaptive-rag/processed_data.tar.gz`
  - SHA-256: `6482d1c567f7a9f169e96f01c1c5c85328026c8c8c88f2116f48482c2a1c7db9`
- Source member: `./processed_data/hotpotqa/test_subsampled.jsonl`
  - SHA-256: `729d8987560092c8476523488069e6fb9bd1527157dc0d7bfebf3ddc336c558e`
  - 500 unique HotpotQA records: 412 bridge, 88 comparison; all 500 have
    `question_id`, question text, level, type, answer objects, and contexts.
- Phase 0 exclusion manifest:
  `research/results/hotpotqa_500_question_manifest_seed100.json`
  - SHA-256: `a09e0e212ca1e18ba569d7fdf05de5af60691b5a0bc060d697448ca55d3a98d5`
  - 500 unique IDs; overlap with the source member and all generated splits: 0.

The upstream subsampling code identifies `test_subsampled.jsonl` as the test
sample and avoids it when creating `dev_500_subsampled.jsonl`. It also attempts
to remap non-single-hop contexts through the dataset-named retrieval corpus.
The research pipeline maps HotpotQA to the `hotpotqa` corpus alias.

## Deterministic selection

- Seed: `20261109`.
- Stratification field: documented HotpotQA `type` (`bridge`, `comparison`).
- Allocation: Hamilton rounding against the 500-record source distribution,
  followed by one seeded shuffle per sorted category. Split order is calibration,
  validation, then test; emitted records are sorted by exact question ID.

| Split | N | Bridge | Comparison | Phase 0 overlap |
|---|---:|---:|---:|---:|
| Calibration | 50 | 41 | 9 | 0 |
| Validation | 25 | 21 | 4 | 0 |
| Held-out test | 150 | 124 | 26 | 0 |

The three split ID sets are pairwise disjoint. The same files are intended for
Gemma 12B and Gemma 31B under A/B/C; no model-specific sampling occurs.

## Generated manifest hashes

| File | Records | SHA-256 |
|---|---:|---|
| `calibration.jsonl` | 50 | `d387583b768319c91edb832502cfa78349a4957e6596c433b51d8c6179d59d9f` |
| `validation.jsonl` | 25 | `738d6508d57dae28179c039ae7611111ec27c01859ead046104bd7cad09b72d0` |
| `test.jsonl` | 150 | `be530a7228aa8e0964ed70b56d4752f20d63f93ad30f0db769ca7726fdc0d4c2` |
| `inference/calibration.jsonl` | 50 | `0917e208cdaea7a429e3f922a12557c93982ccf1b1a1ab83f89df3ae67e9f585` |
| `inference/validation.jsonl` | 25 | `bcffb56de3a6b15eb0b138c0c007ce320334b0c1749c634e293058f1e8f5bbe1` |
| `inference/test.jsonl` | 150 | `39383a6c70643e2465a65eea19e96714c0695f5a0fe9e7d18e7a1262ddc12c1f` |
| `split_metadata.json` | 1 | `725ff6f37647cb511897395ce9420f2738bb872201109e165f5065a6a38ee6f6` |

Top-level manifests preserve complete evaluation records. The inference-safe
copies omit both `answers_objects` and `contexts`, including gold-support flags.

## Validation results

- Expected counts and unique-ID counts: pass.
- Pairwise split disjointness: pass.
- Exclusion of all 500 Phase 0 IDs: pass.
- Exact membership in the verified source: pass.
- Source schema, answers, types, and contexts: pass.
- Same manifests for both target LLMs: pass.
- Byte-for-byte rerun with the same seed using `--verify-existing`: pass.
- Full-record schema matches the Phase 0 processed schema. An attempted offline
  A/B/C reader construction was blocked because the local Python environment
  lacks the `_jsonnet` dependency; no inference was attempted.
- Live retrieval/index compatibility: **not verified**. Requests to
  `localhost:9200` failed because Elasticsearch was not running.
- The broader pilot unit suite ran 24 tests: 23 passed and one existing mock
  orchestration assertion failed on LLM request counts (`[0,0,0]` observed
  versus `[1,1,2]` expected). This does not change split membership, but it
  should be resolved before reusing that orchestration path for inference.

The archived records contain contexts, and the source-generation code attempted
HotpotQA-corpus remapping; this is provenance evidence, not a substitute for a
live index check. No retrieval service, LLM, or external API was invoked.

## Blocker and required review action

Restore the existing pipeline environment (including `_jsonnet`), start the
existing Elasticsearch/retriever services without rebuilding the index, verify
that the A/B/C readers consume all 50/25/150 records, and verify representative
or all selected questions against the expected `hotpotqa` index while recording
its identity/count. Until then, E0.0 remains In Progress and the manifests are
candidate-frozen, not approved.

## Created or modified files

- Created `research/experiments/splits/freeze_hotpotqa_v1.py`.
- Created `research/data/splits/hotpotqa_v1/` manifests, metadata, README, and
  inference-safe copies.
- Created this report.
- Modified Table 0 to reference this report while retaining In Progress and no owner.
- Modified Table 1 with the validated candidate counts and retrieval-check caveat.
- Recompiled `research/paper/main.pdf`; Tables 2–6 were not modified.
