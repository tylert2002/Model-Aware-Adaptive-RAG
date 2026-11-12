> **GitHub handoff:** Only the verified two-question pilot is submitted. The compact, per-question A/B predictions and EM/F1 are in [`predictions.jsonl`](research/results/pilot/20261112T054914Z_gemma_b89bcbfd/predictions.jsonl); aggregate metrics are in [`summary_metrics.json`](research/results/pilot/20261112T054914Z_gemma_b89bcbfd/summary_metrics.json). See [`GEMMA_31B_PILOT_HANDOFF.md`](research/reports/GEMMA_31B_PILOT_HANDOFF.md) for coverage, offline validation, and exclusions. Raw prompts, reasoning/retrieval traces, original scored records, failed attempts, and historical diagnostics referenced below are preserved locally; only the selected lightweight evidence is committed. Formal Calibration/Validation/Test completion is **0/0/0 per strategy**; this pilot covers two Calibration IDs per strategy and is not a full formal run. No new inference was performed during synchronization.

# Completed frozen Gemma 31B A/B pilot — 2026-11-12 05:49 UTC

**COMPLETE: A 2/2, B 2/2; four valid inference records.** This section supersedes all blocked/missing-ID statements in the historical report below. This was a two-question pipeline validation, not an accuracy study. Work stopped after this pilot and report.

| Strategy | Completed | Mean EM | Mean F1 | Input/output tokens | Total API latency | Total end-to-end latency |
|---|---:|---:|---:|---:|---:|---:|
| A — No Retrieval | 2/2 | 50% | 50% | 2,967 / 137 | 4.662 s | 4.742 s |
| B — Single-step Retrieval | 2/2 | 50% | 66.5% | 6,102 / 150 | 4.005 s | 4.933 s |

| Frozen question ID (in order) | Reference | A answer | A EM/F1 | B answer | B EM/F1 |
|---|---|---|---|---|---|
| `5a737e4655429908901be2cc` | Cashin' In | Varney & Co | 0 / 0 | Not mentioned in the text | 0 / 0.33 |
| `5a746a4455429974ef308bf6` | Simone Bolelli | Simone Bolelli | 1 / 1 | Simone Bolelli | 1 / 1 |

Scores use the unchanged `pipeline.score_answer` → `DropAnswerEmAndF1`. The scorer awards B partial overlap credit for “in” in the first answer; this is not a correct answer. Its rounded per-question F1 is 0.33. No scorer adjustment was made. All four requests resolved `google/gemma-4-31B-it-qat-w4a16-ct`, returned status `ok` and finish reason `stop`, and used one attempt each. A made no retrieval calls; each B record made one retrieval call and one LLM call. Five actual retrieved paragraphs were verified verbatim in each saved fitted B prompt.

## Verified permissions, Docker, ports, and API

The new session exposes `sandbox_permissions=require_escalated` with a justification. The initial sandbox Docker request failed with socket permission denied and the localhost probe failed to connect. A supported approval request for the existing read-only diagnostics succeeded, confirming that this session can request and obtain the required access. Separate approved requests covered the missing NLTK resource and scoped pilot. Safety controls remained enabled.

Docker Desktop 4.99.0 / Engine 29.9.3 is running on Linux arm64, with 7.748 GiB Docker memory. Host: macOS Darwin 24.6.0 arm64, 16 GiB RAM, 263 GiB free. Existing container `81662e01cbee` / `hotpotqa-es-restore` is running, not restarting, and not OOM-killed; heap remains 2 GiB. Its original data/snapshot volumes and read-only snapshot mount were reused.

- **9200:** existing Java listener accepts connections; plaintext HTTP returns an empty reply. HTTPS completes enough TLS negotiation to present a certificate, then fails normal validation with “self signed certificate in certificate chain.” This confirms a TLS endpoint; its ES version and authenticated index contents were not established. No TLS verification was disabled and this service was not changed.
- **9201:** existing isolated Elasticsearch **OSS 7.10.2** responds HTTP 200. Fresh exact `hotpotqa` count: **5,233,329**. One active primary, one unassigned replica, no initializing/relocating shards; replica-only yellow is expected on one node. Full read-only verifier passed recovery, mappings/settings, sample IDs, and manifest BM25 result order.
- **8000:** existing actual upstream retriever is running and successfully returns real HotpotQA passages through 9201. No duplicate retriever was started.
- **NRP:** fresh connectivity returned `OK` at `https://ellm.nrp-nautilus.io/v1/chat/completions`, alias `gemma`, resolved model `google/gemma-4-31B-it-qat-w4a16-ct`, one attempt, 17 input / 2 output tokens, 0.729 s. Credentials were loaded in memory using the existing root configuration and never printed.

Fresh evidence: `research/results/pilot/gemma31b_assignment_20261111/restore_verification_20261112T054723651637Z.json` and `connectivity_20261112T054723198254Z.json`. Docker/port diagnostics also appear in the approved command output. No restore, repository registration, rebuild, volume deletion, model substitution, frozen-data modification, or service restart occurred. The previously verified delivered archive remains `hotpotqa_elasticsearch_7.10.2_snapshot_20261110t0630z.tar.gz`, SHA-256 `48ca9131be591611a725390ddcce7b83a0f3d62ddd17920675803c3cac9d5e54`; reuse was based on fresh live index verification rather than another archive import.

## Scope and local repairs

Branch `main`, commit `f692efabd34d6bbcef24b62737c280951f219ef5`; existing uncommitted work preserved. The scoped research wrapper `research/results/pilot/gemma31b_assignment_20261111/run_frozen_ab.py` restricts the original pipeline to A/B and the exact ordered IDs above, without sampling. It uses `research/data/splits/hotpotqa_v1/inference/calibration.jsonl` for model-facing input, with empty initial contexts and no gold answers. Both frozen Calibration hashes are asserted before execution: evaluation `d387583b768319c91edb832502cfa78349a4957e6596c433b51d8c6179d59d9f`; inference `0917e208cdaea7a429e3f922a12557c93982ccf1b1a1ab83f89df3ae67e9f585`. Gold references are read through the original reader only after all inference and used solely for scoring.

Upstream home disk caches are redirected to ignored project-local `.pilot-cache/{gpt3calls,llmcalls,title_queries}` by the wrapper during imports, preserving cache semantics and HOME. The first graph-construction attempt stopped before inference because standard NLTK `stopwords` was absent; downloaded only that resource into `.pilot-cache/nltk_data`, then reran successfully. The failed pre-inference directory `research/results/pilot/20261112T054852Z_gemma_4a4be204/` is preserved and is not counted as an inference record. No packages or model weights were installed. Existing cache-only GPT-2 tokenizer loading passed.

Prompts, controller graphs, original answer extraction, and scoring were retained. Generation: max_tokens 300, temperature 0, top_p 1, newline stop, thinking disabled; timeout 300 s, max retries 2, concurrency 1. GPT-2 fitting retains the original 3,500-token limit, 300-token reservation and demonstration-removal algorithm. B retains BM25 depth 5, original-question query, `hotpotqa`, `title_paragraph_text`, maximum 15 paragraphs. The original prompt loader warned “Dropped 7 / 20 examples”; its configured filtering/fitting was preserved. Strategy C and Gemma 12B were never executed.

## Artifacts and reproduction

Run directory: `research/results/pilot/20261112T054914Z_gemma_b89bcbfd/`.

- **`scored_results.jsonl`**: authoritative four complete records, answers/references, original EM/F1, model, tokens, latency, retrieval counts and verified context document IDs.
- `results.jsonl`: preserved inference-only records before reference joining; null scores there are intentional.
- `traces/*.jsonl`: actual API responses including attempts and token usage; `*_retrieval.jsonl`: actual document IDs, titles, passages and retrieval requests; `*.prompt.txt`: exact fitted prompts sent to the adapter.
- `run_metadata.json`: commit, input/config/prompt hashes and settings. Its inherited upstream metadata serializer represents stop as a literal backslash-n; actual generator execution uses a newline. `validation_summary.json` records that distinction and verified aggregate results without altering the original artifact.

For the first question, B retrieved Fox News Specialists, Fox News Talk, Fox News Watch, Fox News Live, and Fox News; none supplied the required program. For the second, it retrieved Simone Bolelli, Bolelli, John Peers, 2015 Trofeo Faip–Perrel – Singles, and Simon Youl. All ten retrieved passage IDs and full contents are saved, and all five passages per B record were found in the fitted prompt. A's fitted prompts contain no retrieved test-question context.

From the repository root, with approved localhost/NRP access and the existing services running:

```sh
adaptive-rag/.venv/bin/python research/results/pilot/gemma31b_assignment_20261111/verify_restored_index.py
adaptive-rag/.venv/bin/python research/results/pilot/gemma31b_assignment_20261111/run_frozen_ab.py
```

The second command would issue four new requests and create a distinct result directory; it was not repeated after success. Four records, exact paired IDs, model identity, one attempt per request, A/B retrieval counts, and all B passage-to-prompt matches were checked from saved artifacts. This completes the authorized pilot; no expanded experiment or push was performed.

---

# Latest permission recheck — 2026-11-12 05:42 UTC

**BLOCKED: A 0/2, B 0/2, valid records 0/4; EM/F1 unavailable.** Current permissions still prohibit the operations required for actual inference. This section supersedes earlier current-status claims; prior evidence is preserved below.

- Docker CLI runs, but Engine socket access returns permission denied.
- Both `127.0.0.1:9200` and `:9201` return `Operation not permitted` before an HTTP connection. No current Elasticsearch version, index count or retrieval success can be claimed. Prior verified OSS 7.10.2 restore remains on 9201; no restore was repeated.
- Fresh authorized Gemma adapter request failed with `network_error`, no HTTP status. Direct credential-free HTTPS probe failed at DNS: `curl (6) Could not resolve host: ellm.nrp-nautilus.io`. Thus this attempt did not reach authentication or API HTTP handling. Restricted sandbox networking is confirmed; independent host DNS failure or service outage is not established. Host comparison is required.
- Project files and existing experiment outputs are readable; writes to this report and diagnostic JSON succeeded. Both frozen manifest hashes were checked again against metadata, and IDs/question texts match the inference-safe reader. First pair: `5a737e4655429908901be2cc`, `5a746a4455429974ef308bf6`.
- Existing retriever supports host/port constructor arguments; the preserved `serve_restored_retriever.py` uses port 9201. No endpoint configuration was changed without being able to verify that endpoint. No running service was stopped or duplicated.

## Minimal authorized-session action

This session has restricted networking and granular approvals with `sandbox_approval=false`. User authorization cannot change that runtime policy; no unsupported escalation or sandbox bypass was attempted. From **regular Terminal**, start a separate approval-enabled local session:

```bash
cd /Users/tylert2002/Desktop/CPSC-481/Model-Aware-Adaptive-RAG
codex --sandbox workspace-write --ask-for-approval on-request
```

Then ask it: `Read PILOT_REPORT_GEMMA_31B.md and complete only the frozen two-question Gemma A/B pilot. Request approval for Docker socket, localhost and NRP operations when blocked. Preserve all data and do not run C or push.`

These flags were verified with the installed `codex --help`. Keep the sandbox enabled and approve only the required operations. If managed organization policy disables escalation there too, its administrator must permit these specific operations; another prompt in this same restricted session cannot grant access. Official guidance: [Agent approvals and security](https://learn.chatgpt.com/docs/agent-approvals-security).

Alternatively, the existing `host_diagnostics.sh` command documented below collects host evidence without changing services. Current host API access and current cluster health remain unknown. The new session must verify them before constructing/executing a scoped A/B runner; the generic runner includes C and must not be executed unchanged.

New artifacts: `permissions_recheck_20261112T0542Z.json` and `connectivity_20261112T054216993082Z.json` under `research/results/pilot/gemma31b_assignment_20261111/`. Report updated; no source/configuration changes, index mutations, model substitutions, scored requests or fabricated metrics. No push performed.

---

# Current diagnosis — 2026-11-12 05:36 UTC

**Final status: BLOCKED. Strategy A 0/2; Strategy B 0/2; valid inference records 0/4.** No generated pilot answers, EM/F1, token usage, or inference latency are available. This section supersedes all historical status statements below, especially claims that frozen question IDs are missing. Historical evidence remains preserved below.

## Environment and diagnostic results

macOS Darwin 24.6.0, arm64; UID 501 (staff/admin). Docker CLI 29.9.3 (API 1.56), context `desktop-linux`, socket `/Users/tylert2002/.docker/run/docker.sock`; socket owned by UID 501, mode srwxr-xr-x. Docker Desktop plugin 4.99.0 is installed; current Engine version and Desktop running state are unverified. Host RAM query and process listing are sandbox-denied; historical RAM is 16 GiB, not freshly verified. Disk has 263 GiB available. Branch `main`, commit `f692efabd34d6bbcef24b62737c280951f219ef5`. Existing untracked work preserved.

| Component | Status | Evidence |
|---|---|---|
| Docker Engine | Current state unverified | `docker version`, `info`, `ps -a`, and `volume ls` fail at Unix socket permission denied; client/context inspection works |
| Codex sandbox access | Confirmed blocker | localhost IPv4/IPv6 connect returns `Operation not permitted`; sandbox approval disabled by session policy |
| Elasticsearch HTTP | Live verification blocked | All requested port 9200 probes denied before connection; 9201 also denied, including with proxy bypass |
| Snapshot integrity | Verified locally | All four delivery files present; archive 1,942,360,297 bytes, SHA-256 `48ca9131be591611a725390ddcce7b83a0f3d62ddd17920675803c3cac9d5e54`; checksum verification repeated from delivery directory |
| HotpotQA index | Historical verification only | Existing `restore_verification_20261112T042645968096Z.json`: OSS 7.10.2, count 5,233,329, recovery DONE, replica-only yellow, mappings/settings/samples and BM25 order verified on 9201 |
| Retriever | Live access blocked | Port 8000 sandbox-denied; historical evidence contains five upstream retrieval hits from restored index |
| Frozen question IDs | Verified | First two Calibration entries below; both manifests hash-match metadata and local `origin/suzyhuang` tree |
| NRP API | Fresh request failed | `connectivity_20261112T053630060745Z.json`: `network_error`, no HTTP status; authorized credential loader succeeded; historical successful model identity only |

## Frozen pilot pair

Use exactly this order, for both A and B:

1. `5a737e4655429908901be2cc` — This Fox News Specialist Host also took over what Fox Business Channel news program in 2013?
2. `5a746a4455429974ef308bf6` — Which tennis player is older out of John Peers and Simone Bolelli?

Calibration manifest SHA-256: `d387583b768319c91edb832502cfa78349a4957e6596c433b51d8c6179d59d9f`.
Inference Calibration SHA-256: `0917e208cdaea7a429e3f922a12557c93982ccf1b1a1ab83f89df3ae67e9f585`.
All 50 IDs/question texts match in order; both files match the existing remote-tracking branch bytes. No fetch, checkout, sampling, or manifest change occurred. Metadata calls the split `candidate_frozen_pending_retrieval_runtime_verification`; the supplied task authorizes this existing Calibration order. `read_inference_safe_examples` accepts all 50 records with empty titles/paragraphs and rejects gold fields by implementation. Actual retrieved-evidence inclusion in a B generation prompt remains unverified.

## Root cause analysis

Confirmed: sandbox denies Docker socket and local service connections. This session disables `sandbox_approval`; no escalation or bypass attempted. Socket ownership alone gives no evidence of a host filesystem permission fault. Docker daemon failure, OOM, disk watermarks, bootstrap errors, mount permissions, allocation errors and restart state cannot be diagnosed without host logs.

Confirmed: Java PID 95464 listens on `*:9200`; historical restore used isolated `127.0.0.1:9201:9200`. The expected port 9200 does not identify the historically restored OSS node. Preserve both services and verify port 9201 rather than moving/deleting either. The reported host `Empty reply from server` was not reproduced here: current sandbox failures happen before HTTP. HTTPS/security on the Java service is a hypothesis until host protocol/log output confirms it. No proxy environment variables are set; proxy-bypassed localhost requests are still denied.

NRP configured endpoint is `https://ellm.nrp-nautilus.io/v1/chat/completions`, alias `gemma`. Credentials are available through the existing root configuration loader and were never printed. Fresh sandbox request failed with `network_error`; this does not prove invalid credentials or API outage. Current host API access remains untested. Historical response resolved `google/gemma-4-31B-it-qat-w4a16-ct`; no current model identity is claimed.

## Changes and safe repairs

Updated this report; added `diagnosis_20261112.json` with manifest/reader evidence and `host_diagnostics.sh` with read-only host checks; existing connectivity helper created one new sanitized failure record. Corrected the missing-ID diagnosis. No application configuration, prompts, retriever algorithm, containers, volumes, index or frozen manifests changed. No new restore, extraction, download, model substitution, Strategy C or bulk inference occurred. The initial checksum command used the wrong working directory and could not find the companion; it was rerun from `research/results/pilot/`.

## Remaining blockers and exact host steps

Run from a regular Terminal:

```bash
cd /Users/tylert2002/Desktop/CPSC-481/Model-Aware-Adaptive-RAG
bash research/results/pilot/gemma31b_assignment_20261111/host_diagnostics.sh > /tmp/gemma-host-diagnostics.txt 2>&1
```

Return `/tmp/gemma-host-diagnostics.txt` (review logs for private data before sharing). The script inventories real container IDs before reading logs; it makes no Docker mutations. It probes both ports, HTTP/TLS, count, resources, NRP connectivity and the existing full restore/retriever verifier. Docker Desktop memory allocation should also be read from Settings → Resources; the historical node used 2 GiB heap, but current limits and available RAM require host confirmation. Do not apply memory changes until logs/resources identify a need.

If port 9201 verifies but 8000 is absent, start the existing helper in a separate regular Terminal only after confirming no current retriever listener:

```bash
adaptive-rag/.venv/bin/python research/results/pilot/gemma31b_assignment_20261111/serve_restored_retriever.py
```

Do not rerun restore or create replacement volumes. If the existing node is stopped, inspect its actual state/logs first and start only the verified existing container. The original generic runner samples and includes C; do not use it unchanged for this assignment. A scoped A/B runner using the verified pair and original bridge/scorer is still required when API access permits execution. Historical home diskcache write failures remain a possible runtime blocker, not newly tested or repaired here. No scored API request was attempted after the minimal connectivity prerequisite failed. Unit tests are not evidence of completed inference.

---

# Gemma 31B Strategies A + B pilot — partial report

## Snapshot setup follow-up

### Restore and live retrieval completed

Docker Desktop is now installed and running (Linux aarch64 runtime). Pulled `docker.elastic.co/elasticsearch/elasticsearch-oss:7.10.2-arm64`; digest `sha256:fbd0d103d52b35df1cffeb8011f9006f51ee7a2447f062d6232c05dfa999fc3b` matches the source manifest. Created new volumes `hotpotqa-restore-snapshots` and `hotpotqa-restore-data`, imported the complete archive into the snapshot volume, and started container `hotpotqa-es-restore` with `127.0.0.1:9201:9200`, 2 GiB JVM heap, and read-only snapshot mount. Existing port 9200 service preserved.

Registered `hotpotqa_fs_repository` read-only at `/mnt/es_snapshots/hotpotqa_repository`; confirmed snapshot SUCCESS and index `hotpotqa`. Verified target index absent before restoring only `hotpotqa` with `include_global_state: false`. Primary recovery completed. Exact count: **5,233,329**. Health yellow with one active primary and one unassigned replica, no initializing shards; expected for this single-node setup. Mappings, creation version `7100299`, one primary/one replica, sample IDs `12001`/`12002` and their titles matched. Manifest multi_match query produced IDs `4795782`, `12001`, `1247980` in order.

Source index UUID `8N5vCmxTSx6zcyJpkFlCgg` differs from restored UUID `UxEda6-2SJWaj2ey6sI_wQ`. This is expected for restore into a new index: Elasticsearch 7.10.2 [RestoreService source](https://github.com/elastic/elasticsearch/blob/v7.10.2/server/src/main/java/org/elasticsearch/snapshots/RestoreService.java#L302-L314) explicitly generates a new index UUID. Both are retained in evidence; no index was recreated to force a match.

Started actual upstream FastAPI retrieval service on `127.0.0.1:8000` via the research helper `serve_restored_retriever.py`, which changes only the backend connection to port 9201 and leaves upstream files/algorithms intact. A real `/retrieve/` call for `Luxembourg Football Federation`, document type `title_paragraph_text`, corpus `hotpotqa`, max hits 5 returned five nonempty passages. First result: `Luxembourg Football Federation`, source document ID `4FQnxQP9koKD2ZhNvKKBUmL3jn7pVwWQ`. Full returned IDs, titles, passages, and settings are saved in `research/results/pilot/gemma31b_assignment_20261111/restore_verification_20261112T042645968096Z.json`. Retriever ranking differs from the manifest's direct multi_match query because the unchanged upstream retriever uses its own query; the manifest's direct query was separately verified.

Reproduction from repository root, after Docker Desktop starts:

```sh
docker start hotpotqa-es-restore
adaptive-rag/.venv/bin/python research/results/pilot/gemma31b_assignment_20261111/serve_restored_retriever.py
# In another terminal:
adaptive-rag/.venv/bin/python research/results/pilot/gemma31b_assignment_20261111/verify_restored_index.py
```

Do not run a second retriever instance while the current port 8000 service is running. Snapshot/index/retrieval blockers are now resolved. The ordered two team question IDs remain missing, so no assigned A/B inference was performed and retrieved-context inclusion in a B generation prompt has not yet been tested. No NRP inference was sent during restore verification.

The four delivery files are now present in the local Downloads folder. The original `.tar.gz` is exactly 1,942,360,297 bytes and its SHA-256 matches both manifest and checksum: `48ca9131be591611a725390ddcce7b83a0f3d62ddd17920675803c3cac9d5e54`.

Read the supplied restore guide. Archive inspection completed without extraction: 89 entries, all regular files/directories under `hotpotqa_repository`, no absolute paths, parent traversal, or links. Internal `index-0` identifies snapshot `hotpotqa_20261110t0630z`, version 7.10.2, and index `hotpotqa`, matching the manifest. This does not yet establish a successful local restore.

Disk space is sufficient (277 GiB available). A Java process occupies host port 9200; preserve it and plan to use host port 9201 for the isolated restore. Docker and Docker Desktop were absent. An authorized `brew install --cask docker` downloaded Docker Desktop but failed when Homebrew needed an interactive administrator password to link binaries. Homebrew rolled back the app installation and ran its automatic update/cleanup. The user must complete installation in their own interactive terminal and launch Docker Desktop before container setup can continue. No snapshot was extracted, no Docker data volume created, and no Elasticsearch index changed.

The original report below records the earlier inspection; its snapshot-file availability statements are superseded by this follow-up. The shared two-question IDs remain outstanding.

Date: November 11, 2026 (America/Los_Angeles). Connectivity timestamps are stored in UTC (November 12).

## Outcome

**Connectivity completed; assigned HotpotQA pilot blocked.** No HotpotQA inference or EM/F1 results were produced. Strategy A: 0/2 executed. Strategy B: 0/2 executed. No question IDs were selected arbitrarily, and no historical results were reused as new executions. Strategy C and Gemma 12B were not executed.

| Check | Status | Evidence |
|---|---|---|
| Gemma 31B API connectivity | Completed | Real response `OK`, resolved model `google/gemma-4-31B-it-qat-w4a16-ct` |
| Original development dataset | Prepared | 500 genuine records restored from bundled archive; SHA-256 verified against repository manifest |
| Shared ordered two-question IDs | Blocked | No assignment-specific two-ID manifest found |
| Strategy A, two shared questions | Blocked | Shared IDs and cached GPT-2 tokenizer unavailable |
| Snapshot integrity and restore | Blocked | Four delivery files missing; compatible local ES not verified |
| Retrieval smoke test and context verification | Blocked | Retriever unavailable; snapshot not restored |
| Strategy B, same two questions | Blocked | Shared IDs, tokenizer, verified snapshot/index, and retrieval service required |

There is no A-vs-B accuracy table because neither strategy has actual pilot results.

## Repository and environment

- Branch: `main`; commit: `f692efabd34d6bbcef24b62737c280951f219ef5`.
- Initially untracked: assignment instructions and root `env.example`. Both preserved. No existing tracked source was modified and no commit was created.
- Inspected root/upstream/pilot READMEs, experiment plan, pipeline, adapter, generator bridge, metric implementation, strategy configurations, manifests, and locally available `origin/suzyhuang` tree/README. No checkout or remote fetch was performed. No applicable AGENTS.md was found in the repository scan.
- macOS 15.7.4, Apple Silicon `arm64`, 16 GiB RAM; approximately 305 GB free at inspection.
- System Python: 3.9.6. Reused `adaptive-rag/.venv/bin/python` (3.9.6); no packages installed.
- Docker executable absent from PATH; `/Applications/Docker.app` absent. No Compose configuration or compatible Docker installation was found, so container/volume inspection and OSS 7.10.2 image compatibility verification could not proceed.
- An Elasticsearch 9.5.6 distribution was found under Downloads. It was not used: it does not satisfy OSS 7.10.2. HTTP on `127.0.0.1:9200` returned an empty reply outside the sandbox, so version, indices, and health remain unverified. No local index was deleted or changed.
- Actual retriever `127.0.0.1:8000` refused connection outside the sandbox.
- Installed packages: jsonnet 0.22.0, diskcache 5.6.3, ftfy 6.3.1, openai 1.5.0, rapidfuzz 3.13.0, spaCy 3.4.1, tqdm 4.70.1, transformers 4.26.0.dev0, elasticsearch 7.9.1, FastAPI 0.103.2, uvicorn 0.39.0, requests 2.32.5, torch 1.13.1. Availability does not prove full pipeline compatibility.
- Existing dry-run reported missing dataset (subsequently restored), missing local GPT-2 tokenizer, and missing `en_core_web_sm`. The latter is required by the current all-strategy validator for C and is not itself an A/B requirement.

## Data and shared protocol

Dataset: `adaptive-rag/processed_data/hotpotqa/dev_500_subsampled.jsonl`, from the exact member `./processed_data/hotpotqa/dev_500_subsampled.jsonl` of the existing `adaptive-rag/processed_data.tar.gz`. Only this regular file was copied with exclusive creation; no general archive extraction was used.

Records: 500. SHA-256: `af878f9767eaf3ee0284240e02d105b198f051971fab831f3877270c2c3711ef`, matching `research/results/hotpotqa_500_question_manifest_seed100.json` (development dev_500_subsampled).

**Exact assigned question IDs: unavailable.** Found a 500-ID HotpotQA manifest (including a preserved 30-ID pilot subset), a 30-ID SQuAD manifest, and a historical one-question run. None specifies the team's ordered two HotpotQA IDs. Neither the first two manifest entries nor a seed-selected pair was treated as the agreed pair.

## API test and intended pipeline settings

Used the already-present local configuration in root `env.example` without printing credential values or copying them into artifacts. The normal runner instead expects `research/experiments/pilot/.env`, which is absent; shell `NRP_API_KEY` was also absent. The connectivity helper explicitly loads the existing root configuration in memory.

- Endpoint: `https://ellm.nrp-nautilus.io/v1/chat/completions`.
- Requested alias: `gemma`; actual response model: `google/gemma-4-31B-it-qat-w4a16-ct`.
- Connectivity prompt: `Reply with OK.`; max_tokens 16, temperature 0, top_p 1, thinking disabled; no stop sequence supplied for this connectivity-only request.
- Configured timeout 300 seconds; max retries 2; successful request used 1 attempt.
- Response: `OK`; finish reason `stop`; 17 input tokens, 2 output tokens, 19 total; adapter latency 0.641 seconds.
- First sandboxed attempt failed with `network_error`; rerun with network permission succeeded. Both records retained. This is a technical connectivity test, not a scored pilot execution.

Existing intended pilot decoding: max_tokens 300, temperature 0, top_p 1, newline stop, thinking disabled. Bridge retains original GPT-2 fitting (3,500-token limit, 300 reserved tokens, triple-newline demonstrations, remove first demonstration, retain test example). B config uses BM25 retrieval_count 5 against `hotpotqa`, original question as query, `title_paragraph_text`, maximum 15 paragraphs, and original prompt/answer extraction. No retrieval parameters or prompts changed.

Scoring remains `pipeline.score_answer` → `DropAnswerEmAndF1` for HotpotQA, including its original normalization, numeric/multi-span behavior, and best-reference handling. No alternative evaluator was introduced.

The existing `run.py` selects IDs by seed and `run_pilot` iterates A/B/C unconditionally. Do not execute it unchanged for this assignment. Once the ordered team IDs are provided, a narrowly scoped research wrapper must reuse the original A/B backends, bridge, traces, and scorer while enforcing only those two IDs and strategies A/B. No such inference wrapper was needed while IDs remained unavailable.

## Snapshot and retrieval

Searched repository, Downloads, Desktop, and local `Library/CloudStorage` for delivery files/manifests/restore guidance. None of the required four files was found; no OneDrive share URL was provided in the inspected assignment. No archive download, index rebuild, snapshot extraction, repository registration, or restore was attempted.

Required delivery:

- `hotpotqa_elasticsearch_7.10.2_snapshot_20261110t0630z.tar.gz`
- Matching `.sha256` file
- Companion manifest JSON
- `RESTORE_HOTPOTQA_7.10.2.md`

Expected archive: 1,942,360,297 bytes; SHA-256 `48ca9131be591611a725390ddcce7b83a0f3d62ddd17920675803c3cac9d5e54`. These are required values from the assignment, **not locally verified measurements**.

Required snapshot: `hotpotqa_20261110t0630z`, state SUCCESS, index `hotpotqa`, 5,233,329 documents on Elasticsearch OSS 7.10.2. Archive integrity, snapshot contents, mappings/settings, shard recovery, exact count, sample IDs, BM25 matches, and retrieved context inclusion remain unverified. There are no real retrieved document IDs/titles to report.

## Saved artifacts and reproduction

Artifacts: `research/results/pilot/gemma31b_assignment_20261111/` contains exclusively created connectivity JSON records, `dependency_versions.json`, and the connectivity helper. No per-question result file exists because no shared question was executed.

From repository root, repeat only the connectivity test (sends one small Gemma 31B request; creates a new timestamped record):

```sh
adaptive-rag/.venv/bin/python research/results/pilot/gemma31b_assignment_20261111/connectivity_check.py
```

Original diagnostic command (no inference; all-strategy validator, not the assigned A/B execution):

```sh
adaptive-rag/.venv/bin/python -m research.experiments.pilot.run \
  --model gemma --dataset hotpotqa --limit 2 --seed 100 --dry-run
```

Dataset preparation used Python `tarfile` to read only the named archive member and write its bytes to the expected path with `open('xb')`. The destination now exists; do not overwrite it when repeating setup.

## Inputs needed to resume

1. Provide the **two agreed HotpotQA IDs in order**, or the authoritative manifest identifying exactly that pair. A can then proceed once its local runtime/tokenizer is ready, independently of snapshot availability.
2. Provide the four snapshot delivery files or an accessible OneDrive share/download location.
3. Make a compatible OSS 7.10.2 runtime available, using the team's restore guide and a separate destination data volume; verify Apple Silicon image compatibility before installation/restore. Preserve any existing service/data on port 9200.
4. Cache the GPT-2 tokenizer assets required by the original fitting algorithm; validate A/B graph compatibility in the existing environment. No model weights or spaCy C checkpoint are required merely for this assigned scope.

Once unblocked: execute the two genuine A runs, verify/restore snapshot and retrieval, then execute the two paired B runs, preserving non-overwriting per-question answers, references, EM/F1, model/settings, latency, attempts/tokens, and B retrieval traces. Current report is an accurate partial deliverable, not a claim of completed pilot inference.


## Latest local reinspection — 2026-11-12T05:21:25.013796+00:00

This section supersedes earlier current-status statements above. Existing report text and all pre-existing uncommitted files were preserved. No tracked code, prompt, index, container, volume, or results were overwritten.

**Partial: A 0/2, B 0/2.** The ordered two-question assignment remains absent from the searched repository, local manifests, assignment instructions, Desktop/Downloads filenames, and locally available branch guidance. A request for the authoritative ordered IDs is pending. No arbitrary IDs, reference answers, pilot rows, or accuracy metrics were created. No A-vs-B results table is warranted. C and Gemma 12B were not executed.

Completed afresh: archive integrity (1,942,360,297 bytes; SHA-256 `48ca9131be591611a725390ddcce7b83a0f3d62ddd17920675803c3cac9d5e54`), companion checksum/manifest agreement, safe archive inspection (89 regular-file/directory entries), dataset validation (500 rows, hash `af878f9767eaf3ee0284240e02d105b198f051971fab831f3877270c2c3711ef`), and 26 passing adapter/pipeline unit tests. All four delivery files are present under `research/results/pilot/`; no OneDrive download or second extraction/restore was needed. Dataset path remains `adaptive-rag/processed_data/hotpotqa/dev_500_subsampled.jsonl`, the original bundled development subset rather than a newly selected split. Exact team IDs: unavailable.

Current host: macOS 15.7.4 arm64, 262 GiB available. RAM probing is sandbox-denied; the earlier inspection recorded 16 GiB. Reused Python 3.9.6 in `adaptive-rag/.venv`; installed no packages. Compose is v5.7.0. Branch/commit remain `main` / `f692efabd34d6bbcef24b62737c280951f219ef5`. No Compose configuration was located; prior isolated Docker setup is documented above.

Fresh checks and precise limitations:

- Gemma-only connectivity was actually attempted with the existing authorized credential loader and failed with `network_error`; sanitized output is in the newest `connectivity_*.json`. The endpoint is `https://ellm.nrp-nautilus.io/v1/chat/completions`, requested alias `gemma`. The current deployment ID could not be reconfirmed. Prior successful evidence resolved `google/gemma-4-31B-it-qat-w4a16-ct`; this is historical evidence, not a fresh response.
- Read-only live index verification was attempted and stopped at the first localhost request with `Operation not permitted`. Docker inspection also failed with socket permission denied. The session policy disallows sandbox escalation; current cluster/container/index health cannot be confirmed here. These errors do not establish that the services are down.
- Existing `restore_verification_20261112T042645968096Z.json` records OSS 7.10.2, exact count 5,233,329, completed recovery, replica-only yellow, matching mappings/settings/samples and manifest BM25 order, and five actual upstream retriever hits. Those completed checks are preserved as prior evidence. B prompt context inclusion has not been demonstrated.
- GPT-2 cache-only tokenizer loading now passes. The default runner's dry-run fails only its C checkpoint check (`en_core_web_sm`); that checkpoint is outside this assignment. A separate in-memory A/B-only graph-construction diagnostic, without question selection or inference, fails earlier at upstream diskcache initialization (`attempt to write a readonly database`), which uses home-directory caches outside this session's writable roots. No source was modified to bypass it.

Pilot decoding/scoring/retrieval settings remain those documented above: max_tokens 300, temperature 0, top_p 1, newline stop, thinking disabled; original GPT-2 prompt fitting; B BM25 depth 5; HotpotQA `DropAnswerEmAndF1`. The existing runner always builds/executes A/B/C and selects IDs by seed; its `--execute` command is not an acceptable reproduction command for this two-ID A/B assignment. An A/B-only wrapper enforcing the supplied ordered IDs is still needed before inference. No assigned pilot execution command is claimed to exist yet.

Machine-readable reinspection: `research/results/pilot/gemma31b_assignment_20261111/inspection_20261112T052125013531Z.json`. There are no new per-question files because no assigned questions were run. Fresh diagnostics, from repository root:

```sh
adaptive-rag/.venv/bin/python research/results/pilot/gemma31b_assignment_20261111/connectivity_check.py
adaptive-rag/.venv/bin/python research/results/pilot/gemma31b_assignment_20261111/verify_restored_index.py
adaptive-rag/.venv/bin/python -m unittest -q research.experiments.pilot.test_nrp_adapter research.experiments.pilot.test_pipeline
```

Resume requires the authoritative ordered two IDs and an execution context permitting NRP, localhost/Docker access, and upstream home-cache writes (or a scoped project-local cache redirection). Preserve the already restored cluster and its volumes; do not re-index or restore over the existing index. Then validate only A/B, confirm the current Gemma model ID, run two A executions, verify retrieval and prompt-context inclusion, and run the paired two B executions with original evaluation settings.


## Shared-snapshot correction follow-up — 2026-11-12T05:28:25.484135+00:00

All four delivered files are local under `research/results/pilot/`; no download request is needed. Reread `RESTORE_HOTPOTQA_7.10.2.md` and reran its checksum command from that directory:

```sh
shasum -a 256 -c hotpotqa_elasticsearch_7.10.2_snapshot_20261110t0630z.tar.gz.sha256
```

Result: **OK**, SHA-256 `48ca9131be591611a725390ddcce7b83a0f3d62ddd17920675803c3cac9d5e54`. No rebuild or re-index was attempted.

Docker Desktop exists at `/Applications/Docker.app`, the CLI exists at `/usr/local/bin/docker`, and its Unix socket exists and is owned by the current UID 501. Both `docker info` and `docker ps` fail with permission denied connecting to that socket. Process inspection is also denied. Fresh requests to the isolated ES port 9201 `_count` and the existing project retriever on port 8000 both fail with `[Errno 1] Operation not permitted`. This is a **Codex sandbox access blocker**; it is not evidence that Docker Desktop is absent, the archive is missing, or Elasticsearch is broken. Whether Docker Desktop is currently running cannot be confirmed within these restrictions; no host permission problem has been established.

The user requested execution permission if sandbox access is denied. This session explicitly sets `sandbox_approval` to false, so escalation requests are unavailable and automatically rejected by policy; no permission request was issued. An execution context allowing Docker sockets and local/network connections is required to continue live verification.

Preserved the prior restore and evidence documenting OSS 7.10.2, snapshot `hotpotqa_20261110t0630z`, exact 5,233,329 count and real upstream retrieval. Did not register/restore again over the existing index, start duplicate services, or mutate any container/volume. Current verification is blocked, not newly passed. B remains **0/2, blocked**, also awaiting the authoritative ordered two IDs; A remains **0/2, blocked** by those IDs and current API access. No substitute corpus, retriever, model, or ES version was used.

Machine-readable follow-up: `research/results/pilot/gemma31b_assignment_20261111/snapshot_followup_20261112T052825484135Z.json`. No pilot answers or EM/F1 rows exist for this assignment. Work stopped after recording the actual partial outcome.
