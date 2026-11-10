"""Resume the validated HotpotQA pilot artifacts to the full 500-question source split.

This command appends only newly completed question/strategy records to an existing
validated run. Failed attempts are recorded separately and can be retried safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".pilot-cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(REPO_ROOT / ".pilot-cache" / "huggingface" / "transformers"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from .adaptive_rag_bridge import AdaptiveRAGNRPGenerator
from .nrp_adapter import NRPChatAdapter, NRPConfig, NRPRequestError
from .pipeline import (
    RESOLVED_MODEL_IDS,
    STRATEGIES,
    ResourceValidationError,
    aggregate_trace,
    append_jsonl,
    build_upstream_backends,
    capture_retrieval_trace,
    classify_error,
    default_paths,
    directory_sha256s,
    display_path,
    file_sha256,
    git_commit,
    load_question_ids,
    read_selected_examples,
    retrieval_corpus,
    score_answer,
    upstream_import_context,
    validate_resources,
)
from .run import load_env_file

DATASET = "hotpotqa"
SEED = 100
EXPECTED_SOURCE_SHA256 = "af878f9767eaf3ee0284240e02d105b198f051971fab831f3877270c2c3711ef"
EXPECTED_PREFIXES = {
    "gemma-small": {
        "run_dir": "20261105T181253Z_gemma-small_90f19188",
        "results_sha256": "6499ff1835d99de3357eed1deb3938940cec196c86b699940f84ecdf7cae9298",
        "metadata_sha256": "cb1ff58b5bd06b4c8f4f9a37a1cdccde0100eb5ce2de7fd7ac947b167a9aff63",
    },
    "gemma": {
        "run_dir": "20261105T182407Z_gemma_8248a157",
        "results_sha256": "5b403396164f04f04d00f6af096f32d5ef99a74a816a9dfbca2ef513c09ac7f0",
        "metadata_sha256": "eddf1fc14fb72c197729f863cebba617e84b5714a62984b1384cff0442382ba6",
    },
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", choices=tuple(EXPECTED_PREFIXES), required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--execute", action="store_true")
    return result


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def verify_original_prefix(run_dir: Path, model: str, rows: list[dict[str, Any]]) -> None:
    expected = EXPECTED_PREFIXES[model]
    metadata_path = run_dir / "run_metadata.json"
    if file_sha256(metadata_path) != expected["metadata_sha256"]:
        raise ResourceValidationError("original run_metadata.json hash changed")
    if len(rows) < 90:
        raise ResourceValidationError("results file no longer contains the original 90-row pilot prefix")
    prefix_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in rows[:90]
    )
    # JSON round-tripping should be byte-identical for the repository's original rows.
    if hashlib.sha256(prefix_bytes).hexdigest() != expected["results_sha256"]:
        raise ResourceValidationError("original 90-row pilot prefix hash changed")
    keys = [(row.get("question_id"), row.get("strategy")) for row in rows[:90]]
    if len(set(keys)) != 90 or any(row.get("completion_status") != "complete" for row in rows[:90]):
        raise ResourceValidationError("original pilot prefix is not 90 unique completed combinations")


def load_manifest(path: Path, dataset_path: Path) -> tuple[dict[str, Any], list[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    ids = [str(value) for value in manifest.get("question_ids", [])]
    source_ids = load_question_ids(dataset_path)
    if manifest.get("dataset") != DATASET or manifest.get("seed") != SEED:
        raise ResourceValidationError("manifest dataset or seed is incompatible")
    if manifest.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise ResourceValidationError("manifest source hash is incompatible")
    if len(ids) != 500 or len(set(ids)) != 500 or set(ids) != set(source_ids):
        raise ResourceValidationError("manifest must contain exactly the source split's 500 unique IDs")
    return manifest, ids


def next_trace_paths(traces_dir: Path, question_id: str, strategy: str) -> tuple[Path, Path]:
    safe_qid = hashlib.sha256(question_id.encode()).hexdigest()[:16]
    attempt = 1
    while True:
        suffix = "" if attempt == 1 else f"_attempt{attempt}"
        generation = traces_dir / f"{safe_qid}_{strategy}{suffix}.jsonl"
        retrieval = traces_dir / f"{safe_qid}_{strategy}{suffix}_retrieval.jsonl"
        if not generation.exists() and not retrieval.exists():
            return generation, retrieval
        attempt += 1


def expansion_metadata(paths, run_dir: Path, model: str, manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": "hotpotqa_500_abc_expansion",
        "model_alias": model,
        "expected_resolved_model_id": RESOLVED_MODEL_IDS[model],
        "dataset": DATASET,
        "dataset_path": display_path(paths.dataset_path, paths.repo_root),
        "dataset_sha256": file_sha256(paths.dataset_path),
        "manifest_path": display_path(manifest_path, paths.repo_root),
        "manifest_sha256": file_sha256(manifest_path),
        "question_count": 500,
        "seed": SEED,
        "original_completed_records": 90,
        "expected_new_records": 1410,
        "original_results_prefix_sha256": EXPECTED_PREFIXES[model]["results_sha256"],
        "original_metadata_sha256": EXPECTED_PREFIXES[model]["metadata_sha256"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(paths.repo_root),
        "strategy_configs": {
            spec.code: display_path(spec.config_path(paths.upstream_root, DATASET), paths.repo_root)
            for spec in STRATEGIES
        },
        "strategy_config_sha256": {
            spec.code: file_sha256(spec.config_path(paths.upstream_root, DATASET)) for spec in STRATEGIES
        },
        "retriever_address_config_sha256": file_sha256(paths.retriever_address_path),
        "retrieval_corpus": retrieval_corpus(DATASET),
        "prompt_file_sha256": directory_sha256s(paths.upstream_root / "prompts" / DATASET),
        "decoding": {"max_tokens": 300, "temperature": 0, "top_p": 1, "stop": ["\\n"], "enable_thinking": False},
        "manifest": manifest,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.execute:
        raise SystemExit("Refusing inference without --execute")
    paths = default_paths(REPO_ROOT, DATASET)
    validate_resources(paths, DATASET)
    if file_sha256(paths.dataset_path) != EXPECTED_SOURCE_SHA256:
        raise ResourceValidationError("HotpotQA source dataset hash changed")
    manifest, selected_ids = load_manifest(args.manifest.resolve(), paths.dataset_path)
    expected = EXPECTED_PREFIXES[args.model]
    run_dir = paths.results_root / expected["run_dir"]
    results_path = run_dir / "results.jsonl"
    rows = load_rows(results_path)
    verify_original_prefix(run_dir, args.model, rows)
    completed = {
        (str(row["question_id"]), str(row["strategy"]))
        for row in rows
        if row.get("completion_status") == "complete"
    }
    if len(completed) != len(rows):
        raise ResourceValidationError("canonical results contains incomplete or duplicate records")
    if not completed <= {(qid, spec.code) for qid in selected_ids for spec in STRATEGIES}:
        raise ResourceValidationError("canonical results contains a question outside the 500-ID manifest")

    metadata_path = run_dir / "expansion_500_metadata.json"
    if metadata_path.exists():
        prior = json.loads(metadata_path.read_text(encoding="utf-8"))
        if prior.get("manifest_sha256") != file_sha256(args.manifest.resolve()):
            raise ResourceValidationError("existing expansion metadata references a different manifest")
    else:
        metadata_path.write_text(
            json.dumps(expansion_metadata(paths, run_dir, args.model, args.manifest.resolve(), manifest), indent=2) + "\n",
            encoding="utf-8",
        )

    load_env_file(Path(__file__).with_name(".env"))
    adapter = NRPChatAdapter(NRPConfig.from_env())
    attempts_path = run_dir / "expansion_500_attempts.jsonl"
    traces_dir = run_dir / "traces"

    with upstream_import_context(paths.upstream_root):
        backends = build_upstream_backends(paths, DATASET)
        example_maps = {
            code: {example["qid"]: example for example in read_selected_examples(backend, paths.dataset_path, selected_ids)}
            for code, backend in backends.items()
        }
        for question_id in selected_ids:
            for spec in STRATEGIES:
                key = (question_id, spec.code)
                if key in completed:
                    continue
                backend = backends[spec.code]
                example = example_maps[spec.code][question_id]
                trace_path, retrieval_trace_path = next_trace_paths(traces_dir, question_id, spec.code)
                generator = AdaptiveRAGNRPGenerator(
                    adapter,
                    model=args.model,
                    metadata_path=trace_path,
                    max_tokens=300,
                    temperature=0,
                    top_p=1,
                    stop=["\n"],
                    enable_thinking=False,
                )
                started = time.perf_counter()
                answer: Any = ""
                chain = ""
                retrieval_calls = 0
                error_category = None
                try:
                    with capture_retrieval_trace(retrieval_trace_path):
                        answer, chain, retrieval_calls = backend.execute(example, generator)
                    f1, exact_match = score_answer(DATASET, answer, example["answer"])
                except NRPRequestError as error:
                    error_category = f"api:{error.category}"
                    f1 = exact_match = None
                except Exception as error:
                    error_category = classify_error(error)
                    f1 = exact_match = None
                retrieval_calls = int(getattr(backend, "last_retrieval_calls", retrieval_calls))
                trace = aggregate_trace(trace_path)
                references = list(example["answer"]) if isinstance(example["answer"], (list, tuple)) else [example["answer"]]
                row = {
                    "question_id": question_id,
                    "dataset": DATASET,
                    "model_alias": args.model,
                    "resolved_model_id": trace["resolved_model_id"],
                    "strategy": spec.code,
                    "strategy_name": spec.name,
                    "generated_answer": answer,
                    "reference_answers": references,
                    "f1": f1,
                    "exact_match": exact_match,
                    "retrieval_calls": retrieval_calls,
                    "llm_request_count": trace["llm_request_count"],
                    "input_tokens": trace["input_tokens"],
                    "output_tokens": trace["output_tokens"],
                    "total_tokens": trace["total_tokens"],
                    "latency": trace["latency"],
                    "end_to_end_latency": time.perf_counter() - started,
                    "completion_status": "complete" if error_category is None else "incomplete",
                    "error_category": error_category,
                    "trace_file": str(trace_path.relative_to(run_dir)),
                    "retrieval_trace_file": str(retrieval_trace_path.relative_to(run_dir)),
                    "reasoning_chain": chain,
                }
                required_present = all(row[name] is not None for name in ("f1", "exact_match", "total_tokens"))
                if error_category is None and required_present:
                    append_jsonl(results_path, row)
                    completed.add(key)
                    print(f"completed {args.model} {question_id} {spec.code} ({len(completed)}/1500)", flush=True)
                else:
                    if error_category is None:
                        error_category = "missing_required_metrics"
                        row["error_category"] = error_category
                        row["completion_status"] = "incomplete"
                    append_jsonl(attempts_path, row)
                    print(f"incomplete {args.model} {question_id} {spec.code}: {error_category}", file=sys.stderr, flush=True)
                    raise ResourceValidationError(
                        "expansion stopped after the first incomplete attempt; fix the external or pipeline error and resume"
                    )
    return 0 if len(completed) == 1500 else 1


if __name__ == "__main__":
    raise SystemExit(main())
