"""Run one real Pilot example through upstream A/B/C with a mock generator.

This entry point never reads API credentials and never performs LLM inference. It
requires the complete upstream dataset-specific Elasticsearch index and retriever service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".pilot-cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(REPO_ROOT / ".pilot-cache" / "huggingface" / "transformers"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from .nrp_adapter import GenerationResult
from .pipeline import (
    STRATEGIES,
    PilotPaths,
    ResourceValidationError,
    build_upstream_backends,
    default_paths,
    load_question_ids,
    read_selected_examples,
    run_pilot,
    select_question_ids,
    upstream_import_context,
    validate_resources,
)

NOTICE = "OFFLINE MOCK VALIDATION — NOT RESEARCH RESULTS"
EXPECTED_HOTPOTQA_DOCUMENTS = 5_233_329
EXPECTED_WIKI_DOCUMENTS = 21_015_324


@dataclass(frozen=True)
class OfflineConfig:
    api_base: str = "offline://no-network"
    timeout_seconds: float = 0.0
    max_retries: int = 0
    max_concurrency: int = 1


class DeterministicMockAdapter:
    """Five-call fixture: A answer, B answer, C evidence, C answer, C final."""

    def __init__(self, reference_answer: str) -> None:
        self.config = OfflineConfig()
        final = f"Therefore, the answer is: {reference_answer}."
        self._responses = [
            ("A:final-answer", final),
            ("B:final-answer", final),
            ("C:reasoning-step-1", "The retrieved evidence identifies a related entity."),
            ("C:reasoning-step-2", final),
            ("C:final-answer", final),
        ]
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, *, model: str, **_: Any) -> GenerationResult:
        if model != "offline-mock":
            raise RuntimeError(f"offline adapter refuses model {model!r}")
        call_index = len(self.calls)
        if call_index >= len(self._responses):
            raise RuntimeError("offline mock received more generation calls than expected")
        stage, text = self._responses[call_index]
        self.calls.append(
            {
                "call_index": call_index + 1,
                "expected_stage": stage,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_character_count": len(prompt),
            }
        )
        return GenerationResult(
            model="offline/mock-deterministic-v1",
            text=text,
            status="ok",
            latency_seconds=0.0,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            finish_reason="mock_fixture",
            request_id=f"offline-mock-{call_index + 1}",
            attempts=1,
        )

    def assert_complete(self) -> None:
        if len(self.calls) != len(self._responses):
            raise RuntimeError(
                f"offline mock expected {len(self._responses)} calls but observed {len(self.calls)}"
            )


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def verify_retrieval_services(dataset: str) -> dict[str, Any]:
    index = "wiki" if dataset == "squad" else "hotpotqa"
    expected_count = EXPECTED_WIKI_DOCUMENTS if dataset == "squad" else EXPECTED_HOTPOTQA_DOCUMENTS
    try:
        count = int(_get_json(f"http://127.0.0.1:9200/{index}/_count")["count"])
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise ResourceValidationError(
                f"missing Elasticsearch index {index!r}; build it from the official upstream corpus first"
            ) from error
        raise ResourceValidationError(f"Elasticsearch {index} count check failed with HTTP {error.code}") from error
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise ResourceValidationError(f"could not validate localhost Elasticsearch: {error}") from error
    if count != expected_count:
        raise ResourceValidationError(
            f"{index} index has {count:,} documents; expected exactly "
            f"{expected_count:,} from the upstream documentation"
        )
    try:
        retriever = _get_json("http://127.0.0.1:8000/")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ResourceValidationError(f"could not validate upstream retriever on localhost:8000: {error}") from error
    return {"elasticsearch_url": "http://127.0.0.1:9200", "document_count": count, "retriever": retriever}


@contextmanager
def capture_retrieval_trace(output_path: Path) -> Iterator[list[dict[str, Any]]]:
    import requests

    original_post = requests.post
    records: list[dict[str, Any]] = []

    def traced_post(url: str, *args: Any, **kwargs: Any):
        started = time.perf_counter()
        response = original_post(url, *args, **kwargs)
        record: dict[str, Any] = {
            "url": url,
            "request": kwargs.get("json"),
            "http_status": response.status_code,
            "latency_seconds": time.perf_counter() - started,
        }
        try:
            body = response.json()
            record["retrieval"] = [
                {key: item.get(key) for key in ("id", "title", "score", "corpus_name")}
                for item in body.get("retrieval", [])
            ]
        except Exception as error:
            record["response_error"] = error.__class__.__name__
        records.append(record)
        return response

    requests.post = traced_post
    try:
        yield records
    finally:
        requests.post = original_post
        output_path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run a real upstream Pilot pipeline with an offline mock LLM.")
    result.add_argument("--dataset", choices=("hotpotqa", "squad"), default="hotpotqa")
    result.add_argument("--seed", type=int, default=100)
    result.add_argument("--dataset-path", type=Path)
    result.add_argument("--execute", action="store_true", help="Required confirmation; never authorizes network LLM calls.")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.execute:
        raise SystemExit("Refusing to run without --execute (this command still never calls an LLM API).")

    paths = default_paths(REPO_ROOT, args.dataset, args.dataset_path)
    paths = PilotPaths(
        paths.repo_root,
        paths.upstream_root,
        paths.dataset_path,
        paths.retriever_address_path,
        REPO_ROOT / "research" / "results" / "pilot" / "offline_validation" / args.dataset,
    )
    validate_resources(paths, args.dataset)
    service_metadata = verify_retrieval_services(args.dataset)
    selected_id = select_question_ids(load_question_ids(paths.dataset_path), 1, args.seed)[0]

    with upstream_import_context(paths.upstream_root):
        backends = build_upstream_backends(paths, args.dataset)
        example = read_selected_examples(backends["A"], paths.dataset_path, [selected_id])[0]
    reference = example["answer"]
    reference_answer = str(reference[0] if isinstance(reference, (list, tuple)) else reference)
    adapter = DeterministicMockAdapter(reference_answer)

    paths.results_root.mkdir(parents=True, exist_ok=True)
    provisional_trace = paths.results_root / ".retrieval_trace_in_progress.jsonl"
    with capture_retrieval_trace(provisional_trace) as retrieval_records:
        run_dir = run_pilot(
            paths=paths,
            dataset=args.dataset,
            model="offline-mock",
            limit=1,
            seed=args.seed,
            adapter=adapter,
            backends=backends,
        )
    adapter.assert_complete()
    provisional_trace.replace(run_dir / "retrieval_trace.jsonl")

    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    if [row["strategy"] for row in rows] != [spec.code for spec in STRATEGIES]:
        raise RuntimeError("A/B/C result order changed")
    if any(row["completion_status"] != "complete" for row in rows):
        raise RuntimeError("one or more upstream strategies did not complete")
    if rows[1]["retrieval_calls"] < 1 or rows[2]["retrieval_calls"] < 2:
        raise RuntimeError("real retrieval counts did not exercise B and multi-step C")

    (run_dir / "VALIDATION_NOTICE.txt").write_text(NOTICE + "\n", encoding="utf-8")
    (run_dir / "mock_generation_trace.json").write_text(
        json.dumps({"notice": NOTICE, "calls": adapter.calls}, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "offline_environment.json").write_text(
        json.dumps(
            {
                "notice": NOTICE,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "selected_question_id": selected_id,
                "dataset_sha256": hashlib.sha256(paths.dataset_path.read_bytes()).hexdigest(),
                "services": service_metadata,
                "retrieval_request_count": len(retrieval_records),
                "nrp_credentials_loaded": False,
                "nrp_requests_made": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{NOTICE}: {run_dir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResourceValidationError as error:
        print(f"Offline validation preflight failed: {error}", file=sys.stderr)
        raise SystemExit(2)
