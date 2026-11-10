# E0.0-B — HotpotQA retrieval and A/B/C compatibility report

## Final recommendation: NEEDS REVIEW

The frozen full-record calibration, validation, and held-out-test manifests are
schema-compatible with the existing A/B/C reader, the existing HotpotQA index is
present and usable, and a one-question calibration smoke test completed A, B,
and C against the configured NRP API. E0.0 should nevertheless remain **In
Progress**: the inference-safe files correctly omit answers and gold contexts,
but the upstream `MultiParaRCReader` cannot load them because it unconditionally
requires `answers_objects` and `contexts`. Running held-out inference from the
full manifest would load gold fields into pipeline state even though the current
configs do not add them to model prompts. A safe reader/evaluation merge boundary
must be verified before held-out inference.

No held-out-test inference, routing, retrieval query, or API call was performed.
No frozen question ID or manifest byte was modified.

## Elasticsearch and index status

- The repository documents a native Elasticsearch 7.10.2 installation; it has
  no project Docker Compose file for Elasticsearch. The existing local runtime
  is a manually created Docker container, `model-aware-pilot-es`, using
  `docker.elastic.co/elasticsearch/elasticsearch-oss:7.10.2-arm64` with
  `discovery.type=single-node`, a 2 GiB heap, and persistent volume
  `model-aware-pilot-es-data` mounted at `/usr/share/elasticsearch/data`.
- The apparent `localhost:9200` outage was an execution-sandbox artifact. Docker
  Desktop owned `127.0.0.1:9200`, but sandboxed HTTP and Docker-socket access were
  denied/refused. The same endpoint queried outside that sandbox responded
  normally. Container logs show Elasticsearch bound to port 9200 and recovered
  the persisted index; no rebuild or download was performed.
- Elasticsearch version: 7.10.2 OSS; cluster UUID:
  `m3ZE9uKqRM2MIn5MDOultw`; index UUID:
  `8N5vCmxTSx6zcyJpkFlCgg`.
- Cluster health was `yellow`, expected for one active primary and one unassigned
  replica on a single-node cluster. There were no pending tasks and the primary
  shard was usable.
- The sole index was `hotpotqa`, open with exactly **5,233,329** documents, zero
  deleted documents, and approximately 2.1 GiB store size. This matches the
  upstream documented count and the pilot validator's exact requirement.
- The repository retriever was started locally from
  `adaptive-rag/retriever_server/serve.py` on `127.0.0.1:8000`; its root health
  request and `/retrieve/` requests succeeded.

## Dependency resolution

- System Python 3.9.6 does not provide `_jsonnet`.
- The project-convention isolated runtime, `.venv-pilot` (Python 3.9.6), already
  provides `_jsonnet` from `jsonnet==0.21.0`, matching
  `research/config/pilot-runtime-py39.lock.txt` and the `jsonnet` dependency in
  `adaptive-rag/requirements.txt`.
- The same environment contains the other checked upstream dependencies,
  `en_core_web_sm`, and the cache-only GPT-2 tokenizer assets required by prompt
  fitting. No package was installed and no system-wide change was made.

## Manifest-reader compatibility

The frozen outputs were reproduced byte-for-byte with
`freeze_hotpotqa_v1.py --verify-existing`. Using absolute manifest paths (the
upstream import context changes the working directory), each full manifest was
passed through the existing dry-run preflight. For every split, all three
Jsonnet graphs were instantiated and all records were read by each graph's
unchanged `MultiParaRCReader`:

| Manifest | Records read per A/B/C graph | IDs changed | Result |
|---|---:|---:|---|
| `calibration.jsonl` | 50 | 0 | Pass |
| `validation.jsonl` | 25 | 0 | Pass |
| `test.jsonl` | 150 | 0 | Pass; no routing or inference executed |

The start states also matched the fixed strategy contract: A
`generate_question`, B `generate_titles`, and C
`step_by_step_bm25_retriever`. The relevant pilot adapter/pipeline unit suite ran
22 tests with all passing.

The inference-safe files preserve the exact record count, ID order, and
question-metadata projection of the full files. Their only fields are `dataset`,
`question_id`, `question_text`, `level`, and `type`; they contain no
`answers_objects`, `contexts`, supporting facts, pinned contexts, or other gold
fields. Their frozen SHA-256 hashes remain:

| Inference-safe file | Records | SHA-256 |
|---|---:|---|
| `inference/calibration.jsonl` | 50 | `0917e208cdaea7a429e3f922a12557c93982ccf1b1a1ab83f89df3ae67e9f585` |
| `inference/validation.jsonl` | 25 | `bcffb56de3a6b15eb0b138c0c007ce320334b0c1749c634e293058f1e8f5bbe1` |
| `inference/test.jsonl` | 150 | `39383a6c70643e2465a65eea19e96714c0695f5a0fe9e7d18e7a1262ddc12c1f` |

However, these safe projections are not directly consumable by the existing
reader: it unconditionally reads `answers_objects` and `contexts`, including
gold metadata, before execution. The active GPT HotpotQA configs set both
`add_paras` and `add_gold_paras` to false, and the live calibration traces showed
only retrieved evidence reaching B/C, so no gold answer or gold context was sent
to the model in this smoke test. This config-level non-exposure is not an
adequate held-out safety boundary because the gold fields are still resident in
pipeline state when a full manifest is used.

## Retrieval smoke test

Two questions were read only from `inference/calibration.jsonl` and sent to the
existing `/retrieve/` contract with `corpus_name="hotpotqa"`,
`document_type="title_paragraph_text"`, and five requested hits. Gold calibration
contexts were used only after retrieval to count supporting-title overlap; no
answer content was printed or sent.

| Calibration question ID | HTTP | Hits | Returned corpus tags | Supporting-title overlap |
|---|---:|---:|---|---:|
| `5a737e4655429908901be2cc` | 200 | 5 | `hotpotqa` only | 1/5 |
| `5a746a4455429974ef308bf6` | 200 | 5 | `hotpotqa` only | 2/5 |

The exact index count, request corpus parameter, response corpus tags, and
supporting-title overlaps jointly confirm that retrieval used the intended
HotpotQA corpus rather than `wiki` or another index.

## A/B/C smoke test

One calibration question (`5a7a5b3a5542996c55b2dd71`) was run sequentially
through A, B, and C using the frozen full calibration manifest, the existing
local index/retriever, and the configured `gemma-small` API alias. The resolved
model was `google/gemma-4-12B-it-qat-w4a16-ct`. All strategies completed. This
was a compatibility smoke test, not a research result; the observed F1/EM was
0 for all three and should not be interpreted statistically.

| Strategy | Status | Retrieval calls | LLM calls | Input tokens | Output tokens | Total tokens |
|---|---|---:|---:|---:|---:|---:|
| A | Complete | 0 | 1 | 1,484 | 49 | 1,533 |
| B | Complete | 1 | 1 | 2,926 | 40 | 2,966 |
| C | Complete | 10 | 11 | 31,925 | 285 | 32,210 |
| **Total** | **Complete** | **11** | **13** | **36,335** | **374** | **36,709** |

Every one of B's one and C's ten retrieval responses was HTTP 200 and tagged
only `hotpotqa`. All 13 API trace records had status `ok` and request IDs. The
API response supplied token counts but no monetary-cost field, and the repository
contains no dated price mapping for this NRP deployment; therefore monetary cost
is recorded as **not reported / not derivable**, not estimated.

Artifacts are under
`research/results/pilot/20261110T022642Z_gemma-small_e2048ade/`. They include
calibration reference answers and generated reasoning for evaluation, so they
must not be used as inference input.

## File changes

- Created this compatibility report.
- Created the one-question calibration smoke artifact directory named above.
- Updated Table 1's note to record verified HotpotQA retrieval compatibility and
  the remaining inference-safe-reader limitation.
- Retained E0.0 as `In Progress` in Table 0 and updated its deliverable/note to
  identify the remaining safe-reader boundary; ownership remains unset under the
  table's rule that only completed rows have owners.
- Recompiled `research/paper/main.pdf` after the table edits.
- Tables 2--6 were not modified. Frozen manifests and metadata were not modified.

## Remaining blocker

Before E0.0 can be marked complete, provide or verify a held-out-safe execution
path that consumes the inference-safe record for routing/model prompts and joins
answers/gold contexts only after inference for scoring. The path must preserve
the frozen IDs and A/B/C algorithms and should fail closed if gold fields are
present at the model-facing boundary. Then repeat a no-API reader/preflight check
on all three safe files. Do not run held-out-test inference until that boundary
is in place.
