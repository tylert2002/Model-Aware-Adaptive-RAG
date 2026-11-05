"""Research-owned orchestration around the unchanged Adaptive-RAG pipeline."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from .adaptive_rag_bridge import AdaptiveRAGNRPGenerator
from .nrp_adapter import NRPChatAdapter, NRPConfigurationError, NRPRequestError


SUPPORTED_DATASETS = (
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "nq",
    "trivia",
    "squad",
)
RESOLVED_MODEL_IDS = {
    "gemma-small": "google/gemma-4-12B-it-qat-w4a16-ct",
    "gemma": "google/gemma-4-31B-it-qat-w4a16-ct",
}


def retrieval_corpus(dataset: str) -> str:
    return "wiki" if dataset in {"nq", "trivia", "squad"} else dataset


@dataclass(frozen=True)
class StrategySpec:
    code: str
    name: str
    config_prefix: str
    expected_start_state: str

    def config_path(self, upstream_root: Path, dataset: str) -> Path:
        return upstream_root / "base_configs" / f"{self.config_prefix}_{dataset}.jsonnet"


STRATEGIES = (
    StrategySpec("A", "No Retrieval", "nor_qa_gpt", "generate_question"),
    StrategySpec("B", "Single-step Retrieval", "oner_qa_gpt", "generate_titles"),
    StrategySpec("C", "Multi-step Retrieval", "ircot_qa_gpt", "step_by_step_bm25_retriever"),
)

REQUIRED_UPSTREAM_MODULES = (
    "_jsonnet",
    "diskcache",
    "ftfy",
    "openai",
    "rapidfuzz",
    "spacy",
    "tqdm",
)


class ResourceValidationError(RuntimeError):
    """Raised before inference when faithful upstream execution is unavailable."""


class PilotExecutionError(RuntimeError):
    """Categorized failure for one question/strategy execution."""

    def __init__(self, category: str, message: str):
        self.category = category
        super().__init__(message)


class StrategyBackend(Protocol):
    spec: StrategySpec

    def execute(self, example: Mapping[str, Any], generator: AdaptiveRAGNRPGenerator) -> tuple[Any, str, int]: ...


@dataclass(frozen=True)
class PilotPaths:
    repo_root: Path
    upstream_root: Path
    dataset_path: Path
    retriever_address_path: Path
    results_root: Path


def default_paths(repo_root: Path, dataset: str) -> PilotPaths:
    upstream = repo_root / "adaptive-rag"
    return PilotPaths(
        repo_root=repo_root,
        upstream_root=upstream,
        dataset_path=upstream / "processed_data" / dataset / "dev_subsampled.jsonl",
        retriever_address_path=upstream / ".retriever_address.jsonnet",
        results_root=repo_root / "research" / "results" / "pilot",
    )


def validate_resources(paths: PilotPaths, dataset: str, *, check_dependencies: bool = True) -> list[str]:
    errors: list[str] = []
    if dataset not in SUPPORTED_DATASETS:
        errors.append(f"unsupported dataset {dataset!r}")
    if not paths.upstream_root.is_dir():
        errors.append(f"missing upstream directory: {paths.upstream_root}")
    if not paths.dataset_path.is_file():
        errors.append(
            f"missing processed pilot dataset: {paths.dataset_path} "
            "(prepare it with the original Adaptive-RAG data scripts; do not substitute synthetic questions)"
        )
    if not paths.retriever_address_path.is_file():
        errors.append(f"missing retriever address configuration: {paths.retriever_address_path}")
    for spec in STRATEGIES:
        config = spec.config_path(paths.upstream_root, dataset)
        if not config.is_file():
            errors.append(f"missing strategy {spec.code} config: {config}")
    prompt_dir = paths.upstream_root / "prompts" / dataset
    if not prompt_dir.is_dir():
        errors.append(f"missing prompt directory: {prompt_dir}")
    if check_dependencies:
        missing = [name for name in REQUIRED_UPSTREAM_MODULES if importlib.util.find_spec(name) is None]
        if missing:
            errors.append("missing upstream Python dependencies: " + ", ".join(missing))
        if importlib.util.find_spec("spacy") is not None:
            try:
                import spacy

                if not spacy.util.is_package("en_core_web_sm"):
                    errors.append("missing local spaCy checkpoint: en_core_web_sm (required by strategy C)")
            except Exception as error:  # pragma: no cover - dependency-specific
                errors.append(f"could not validate spaCy checkpoint: {error.__class__.__name__}")
    if errors:
        raise ResourceValidationError("\n".join(f"- {error}" for error in errors))
    return [str(paths.dataset_path), str(paths.retriever_address_path), str(prompt_dir)]


def load_question_ids(dataset_path: Path) -> list[str]:
    question_ids: list[str] = []
    with dataset_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                datum = json.loads(line)
                question_id = str(datum["question_id"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ResourceValidationError(
                    f"invalid processed dataset record at {dataset_path}:{line_number}"
                ) from error
            question_ids.append(question_id)
    if not question_ids:
        raise ResourceValidationError(f"processed dataset contains no questions: {dataset_path}")
    if len(question_ids) != len(set(question_ids)):
        raise ResourceValidationError(f"processed dataset has duplicate question IDs: {dataset_path}")
    return question_ids


def select_question_ids(question_ids: Iterable[str], limit: int, seed: int) -> list[str]:
    ordered = sorted(str(question_id) for question_id in question_ids)
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit > len(ordered):
        raise ResourceValidationError(f"requested limit {limit} exceeds available questions {len(ordered)}")
    selected = random.Random(seed).sample(ordered, limit)
    return sorted(selected)


@contextlib.contextmanager
def upstream_import_context(upstream_root: Path):
    original_cwd = Path.cwd()
    upstream_text = str(upstream_root)
    inserted = upstream_text not in sys.path
    if inserted:
        sys.path.insert(0, upstream_text)
    os.chdir(upstream_root)
    try:
        yield
    finally:
        os.chdir(original_cwd)
        if inserted and sys.path and sys.path[0] == upstream_text:
            sys.path.pop(0)


class AdaptiveRAGStrategyBackend:
    """An exact upstream model graph with only its generator object replaced."""

    def __init__(self, spec: StrategySpec, decomposer: Any, model_map: Mapping[str, Any], reader: Any):
        self.spec = spec
        self.decomposer = decomposer
        self.model_map = dict(model_map)
        self.reader = reader
        self.last_retrieval_calls = 0

    def inject_generator(self, generator: AdaptiveRAGNRPGenerator) -> int:
        injected = 0
        for model in self.model_map.values():
            if hasattr(model, "qa_model") and hasattr(model.qa_model, "generator"):
                model.qa_model.generator = generator
                injected += 1
            elif hasattr(model, "generator"):
                model.generator = generator
                injected += 1
        if injected == 0:
            raise PilotExecutionError("integration", f"strategy {self.spec.code} has no generator boundary")
        return injected

    def execute(self, example: Mapping[str, Any], generator: AdaptiveRAGNRPGenerator) -> tuple[Any, str, int]:
        self.inject_generator(generator)
        before = self._retrieval_calls()
        try:
            _qid, answer, chain = self.decomposer.return_qid_prediction(dict(example), silent=True)
        finally:
            self.last_retrieval_calls = self._retrieval_calls() - before
        return answer, chain, self.last_retrieval_calls

    def _retrieval_calls(self) -> int:
        total = 0
        for model in self.model_map.values():
            try:
                calls = model.return_model_calls()
            except (AttributeError, NotImplementedError):
                continue
            total += int(calls.get("paragraph_retrieve_and_reset", 0))
        return total


def build_upstream_backends(paths: PilotPaths, dataset: str) -> dict[str, AdaptiveRAGStrategyBackend]:
    import _jsonnet
    from commaqa.inference.configurable_inference import build_decomposer_and_models, load_reader

    address = json.loads(_jsonnet.evaluate_file(str(paths.retriever_address_path)))
    try:
        ext_vars = {"RETRIEVER_HOST": str(address["host"]), "RETRIEVER_PORT": str(address["port"])}
    except KeyError as error:
        raise ResourceValidationError(f"invalid retriever address configuration: missing {error.args[0]}") from error

    class ReaderArgs:
        example_reader = "multi_para_rc"

    backends: dict[str, AdaptiveRAGStrategyBackend] = {}
    for spec in STRATEGIES:
        config_path = spec.config_path(paths.upstream_root, dataset)
        config_map = json.loads(_jsonnet.evaluate_file(str(config_path), ext_vars=ext_vars))
        if config_map.get("start_state") != spec.expected_start_state:
            raise ResourceValidationError(
                f"strategy {spec.code} start state changed: {config_map.get('start_state')!r} "
                f"!= {spec.expected_start_state!r}"
            )
        decomposer, model_map = build_decomposer_and_models(config_map)
        reader = load_reader(ReaderArgs(), config_map)
        backends[spec.code] = AdaptiveRAGStrategyBackend(spec, decomposer, model_map, reader)
    return backends


def read_selected_examples(backend: AdaptiveRAGStrategyBackend, dataset_path: Path, selected_ids: list[str]) -> list[dict[str, Any]]:
    selected_set = set(selected_ids)
    examples = {example["qid"]: example for example in backend.reader.read_examples(str(dataset_path)) if example["qid"] in selected_set}
    missing = selected_set - examples.keys()
    if missing:
        raise ResourceValidationError("upstream reader omitted selected IDs: " + ", ".join(sorted(missing)))
    return [examples[question_id] for question_id in selected_ids]


def dry_run_preflight(paths: PilotPaths, dataset: str, limit: int, seed: int) -> list[str]:
    """Fully instantiate upstream graphs and read selected examples without inference."""
    validate_resources(paths, dataset)
    selected_ids = select_question_ids(load_question_ids(paths.dataset_path), limit, seed)
    with upstream_import_context(paths.upstream_root):
        backends = build_upstream_backends(paths, dataset)
        for backend in backends.values():
            read_selected_examples(backend, paths.dataset_path, selected_ids)
    return selected_ids


def score_answer(dataset: str, prediction: Any, reference: Any) -> tuple[float, float]:
    predicted = prediction[0] if isinstance(prediction, list) and prediction else prediction
    predicted = str(predicted or "")
    references = list(reference) if isinstance(reference, (list, tuple)) else [str(reference)]
    references = [str(item) for item in references]
    if dataset in {"hotpotqa", "2wikimultihopqa", "musique"}:
        from metrics.drop_answer_em_f1 import DropAnswerEmAndF1

        metric = DropAnswerEmAndF1()
        metric([predicted], [references])
    else:
        from metrics.squad_answer_em_f1 import SquadAnswerEmF1Metric

        metric = SquadAnswerEmF1Metric()
        metric(predicted, references)
    result = metric.get_metric()
    return float(result["f1"]), float(result["em"])


def aggregate_trace(trace_path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if trace_path.exists():
        with trace_path.open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
    successful = [record for record in records if record.get("status") == "ok"]

    def sum_optional(key: str) -> int | None:
        values = [record.get(key) for record in successful]
        return sum(int(value) for value in values) if values and all(value is not None for value in values) else None

    return {
        "llm_request_count": len(records),
        "input_tokens": sum_optional("input_tokens"),
        "output_tokens": sum_optional("output_tokens"),
        "total_tokens": sum_optional("total_tokens"),
        "latency": sum(float(record.get("latency_seconds", 0.0)) for record in successful),
        "resolved_model_id": successful[-1].get("model") if successful else None,
        "request_statuses": [record.get("status") for record in records],
    }


def create_run_directory(results_root: Path, model: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_root / f"{timestamp}_{model}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)
        output.write("\n")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256s(path: Path) -> dict[str, str]:
    return {
        str(file.relative_to(path)): file_sha256(file)
        for file in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    }


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def run_pilot(
    *,
    paths: PilotPaths,
    dataset: str,
    model: str,
    limit: int,
    seed: int,
    adapter: NRPChatAdapter,
    backends: Mapping[str, StrategyBackend] | None = None,
) -> Path:
    selected_ids = select_question_ids(load_question_ids(paths.dataset_path), limit, seed)
    run_dir = create_run_directory(paths.results_root, model)
    traces_dir = run_dir / "traces"
    traces_dir.mkdir()
    metadata = {
        "experiment": "pilot_model_and_pipeline_validation",
        "model_alias": model,
        "expected_resolved_model_id": RESOLVED_MODEL_IDS[model],
        "dataset": dataset,
        "dataset_path": display_path(paths.dataset_path, paths.repo_root),
        "dataset_sha256": file_sha256(paths.dataset_path),
        "selected_question_ids": selected_ids,
        "limit": limit,
        "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(paths.repo_root),
        "strategy_configs": {
            spec.code: display_path(spec.config_path(paths.upstream_root, dataset), paths.repo_root)
            for spec in STRATEGIES
        },
        "strategy_config_sha256": {
            spec.code: file_sha256(spec.config_path(paths.upstream_root, dataset)) for spec in STRATEGIES
        },
        "retriever_address_config": display_path(paths.retriever_address_path, paths.repo_root),
        "retriever_address_config_sha256": file_sha256(paths.retriever_address_path),
        "retrieval_corpus": retrieval_corpus(dataset),
        "prompt_directory": display_path(paths.upstream_root / "prompts" / dataset, paths.repo_root),
        "prompt_file_sha256": directory_sha256s(paths.upstream_root / "prompts" / dataset),
        "decoding": {"max_tokens": 300, "temperature": 0, "top_p": 1, "stop": ["\\n"], "enable_thinking": False},
        "adapter": {
            "api_base": adapter.config.api_base,
            "timeout_seconds": adapter.config.timeout_seconds,
            "max_retries": adapter.config.max_retries,
            "max_concurrency": adapter.config.max_concurrency,
        },
    }
    write_json_exclusive(run_dir / "run_metadata.json", metadata)
    output_path = run_dir / "results.jsonl"

    with upstream_import_context(paths.upstream_root):
        active_backends = dict(backends or build_upstream_backends(paths, dataset))
        example_maps = {
            code: {example["qid"]: example for example in read_selected_examples(backend, paths.dataset_path, selected_ids)}
            for code, backend in active_backends.items()
        }
        for question_id in selected_ids:
            for spec in STRATEGIES:
                backend = active_backends[spec.code]
                example = example_maps[spec.code][question_id]
                safe_qid = hashlib.sha256(question_id.encode()).hexdigest()[:16]
                trace_path = traces_dir / f"{safe_qid}_{spec.code}.jsonl"
                generator = AdaptiveRAGNRPGenerator(
                    adapter,
                    model=model,
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
                status = "complete"
                error_category = None
                try:
                    answer, chain, retrieval_calls = backend.execute(example, generator)
                    f1, exact_match = score_answer(dataset, answer, example["answer"])
                except NRPRequestError as error:
                    retrieval_calls = int(getattr(backend, "last_retrieval_calls", retrieval_calls))
                    status, error_category, f1, exact_match = "incomplete", f"api:{error.category}", None, None
                except Exception as error:
                    retrieval_calls = int(getattr(backend, "last_retrieval_calls", retrieval_calls))
                    status, error_category, f1, exact_match = "incomplete", classify_error(error), None, None
                trace = aggregate_trace(trace_path)
                references = list(example["answer"]) if isinstance(example["answer"], (list, tuple)) else [example["answer"]]
                row = {
                    "question_id": question_id,
                    "dataset": dataset,
                    "model_alias": model,
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
                    "completion_status": status,
                    "error_category": error_category,
                    "trace_file": str(trace_path.relative_to(run_dir)),
                    "reasoning_chain": chain,
                }
                append_jsonl(output_path, row)
    return run_dir


def classify_error(error: Exception) -> str:
    if isinstance(error, PilotExecutionError):
        return error.category
    if isinstance(error, ResourceValidationError):
        return "resource"
    if isinstance(error, NRPConfigurationError):
        return "configuration"
    message = str(error).lower()
    if "retriev" in message or "elasticsearch" in message:
        return "retrieval"
    if "metric" in message or "evaluat" in message:
        return "evaluation"
    return f"pipeline:{error.__class__.__name__}"
