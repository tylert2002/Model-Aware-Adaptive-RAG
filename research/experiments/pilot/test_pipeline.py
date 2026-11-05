from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from . import run
from .nrp_adapter import GenerationResult, NRPChatAdapter, NRPConfig, NRPRequestError
from .pipeline import (
    STRATEGIES,
    AdaptiveRAGStrategyBackend,
    PilotPaths,
    ResourceValidationError,
    aggregate_trace,
    run_pilot,
    score_answer,
    select_question_ids,
    upstream_import_context,
    validate_resources,
)


class FakeReader:
    def read_examples(self, path):
        with open(path, encoding="utf-8") as source:
            for line in source:
                datum = json.loads(line)
                yield {
                    "qid": datum["question_id"],
                    "question": datum["question_text"],
                    "query": datum["question_text"],
                    "answer": ("answer",),
                    "titles": [],
                    "paras": [],
                    "metadata": {},
                }


class FakeBackend:
    def __init__(self, spec, calls, retrieval_calls, *, fail=False):
        self.spec = spec
        self.calls = calls
        self.retrieval_calls = retrieval_calls
        self.fail = fail
        self.reader = FakeReader()
        self.executions = []

    def execute(self, example, generator):
        self.executions.append(example["qid"])
        for index in range(self.calls):
            generator.generate_text_sequence(f"prompt {self.spec.code} {index}")
        if self.fail:
            raise RuntimeError("retriever unavailable")
        return "answer", f"chain-{self.spec.code}", self.retrieval_calls


def fake_transport(request, timeout):
    payload = json.loads(request.data.decode())
    body = {
        "model": "resolved/test-model",
        "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }
    return 200, {}, json.dumps(body).encode()


class PilotPipelineTests(unittest.TestCase):
    def test_strategy_mapping_targets_exact_upstream_graphs(self):
        self.assertEqual([spec.config_prefix for spec in STRATEGIES], ["nor_qa_gpt", "oner_qa_gpt", "ircot_qa_gpt"])
        self.assertEqual(
            [spec.expected_start_state for spec in STRATEGIES],
            ["generate_question", "generate_titles", "step_by_step_bm25_retriever"],
        )

    def test_generator_injection_uses_upstream_boundaries(self):
        qa = types.SimpleNamespace(qa_model=types.SimpleNamespace(generator=None))
        cot = types.SimpleNamespace(generator=None)
        backend = AdaptiveRAGStrategyBackend(STRATEGIES[2], None, {"qa": qa, "cot": cot}, None)
        sentinel = object()
        self.assertEqual(backend.inject_generator(sentinel), 2)
        self.assertIs(qa.qa_model.generator, sentinel)
        self.assertIs(cot.generator, sentinel)

    def test_deterministic_question_selection(self):
        ids = [f"q{index}" for index in range(10)]
        self.assertEqual(select_question_ids(ids, 4, 17), select_question_ids(reversed(ids), 4, 17))
        self.assertNotEqual(select_question_ids(ids, 4, 17), select_question_ids(ids, 4, 18))

    def test_end_to_end_orchestration_schema_counts_and_incomplete_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            upstream = root / "adaptive-rag"
            dataset_path = upstream / "processed_data" / "squad" / "dev_subsampled.jsonl"
            dataset_path.parent.mkdir(parents=True)
            dataset_path.write_text(
                json.dumps({"question_id": "q1", "question_text": "question", "answers_objects": []}) + "\n",
                encoding="utf-8",
            )
            (upstream / "base_configs").mkdir()
            (upstream / "prompts" / "squad").mkdir(parents=True)
            for spec in STRATEGIES:
                spec.config_path(upstream, "squad").write_text("{}", encoding="utf-8")
            retriever = upstream / ".retriever_address.jsonnet"
            retriever.write_text("{}", encoding="utf-8")
            results = root / "research" / "results" / "pilot"
            results.mkdir(parents=True)
            paths = PilotPaths(root, upstream, dataset_path, retriever, results)
            backends = {
                "A": FakeBackend(STRATEGIES[0], 1, 0),
                "B": FakeBackend(STRATEGIES[1], 1, 1, fail=True),
                "C": FakeBackend(STRATEGIES[2], 2, 2),
            }
            adapter = NRPChatAdapter(NRPConfig(api_key="placeholder", max_retries=0), transport=fake_transport)
            with patch("research.experiments.pilot.pipeline.score_answer", return_value=(1.0, 1.0)):
                run_dir = run_pilot(
                    paths=paths,
                    dataset="squad",
                    model="gemma-small",
                    limit=1,
                    seed=5,
                    adapter=adapter,
                    backends=backends,
                )
            rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
            self.assertEqual([row["strategy"] for row in rows], ["A", "B", "C"])
            self.assertEqual([row["llm_request_count"] for row in rows], [1, 1, 2])
            self.assertEqual([row["retrieval_calls"] for row in rows], [0, 0, 2])
            self.assertEqual(rows[2]["total_tokens"], 10)
            self.assertEqual(rows[1]["completion_status"], "incomplete")
            self.assertEqual(rows[1]["error_category"], "retrieval")
            expected = {
                "question_id", "dataset", "model_alias", "resolved_model_id", "strategy",
                "generated_answer", "reference_answers", "f1", "exact_match", "retrieval_calls",
                "llm_request_count", "input_tokens", "output_tokens", "total_tokens", "latency",
                "completion_status", "error_category",
            }
            self.assertTrue(expected.issubset(rows[0]))
            self.assertEqual(backends["C"].executions, ["q1"])

    def test_trace_aggregation_preserves_unavailable_usage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = Path(temp_dir) / "trace.jsonl"
            trace.write_text(json.dumps({"status": "ok", "input_tokens": None, "latency_seconds": 0.4}) + "\n")
            aggregate = aggregate_trace(trace)
            self.assertIsNone(aggregate["input_tokens"])
            self.assertEqual(aggregate["llm_request_count"], 1)

    def test_f1_and_exact_match_use_upstream_squad_metric(self):
        repo_root = Path(__file__).resolve().parents[3]
        fake_ftfy = types.SimpleNamespace(fix_text=lambda text: text)
        with patch.dict("sys.modules", {"ftfy": fake_ftfy}), upstream_import_context(repo_root / "adaptive-rag"):
            f1, exact_match = score_answer("squad", "The blue car", ["blue car"])
        self.assertEqual(f1, 1.0)
        self.assertEqual(exact_match, 1.0)

    def test_dry_run_does_not_load_credentials_or_run_pipeline(self):
        with patch.object(run, "dry_run_preflight", return_value=["q1"]), patch.object(
            run, "load_env_file", side_effect=AssertionError("credentials loaded")
        ), patch.object(run, "run_pilot", side_effect=AssertionError("pipeline ran")):
            status = run.main(["--model", "gemma-small", "--dataset", "squad", "--limit", "1", "--dry-run"])
        self.assertEqual(status, 0)

    def test_missing_dataset_corpus_and_checkpoint_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            upstream = root / "adaptive-rag"
            (upstream / "base_configs").mkdir(parents=True)
            (upstream / "prompts" / "squad").mkdir(parents=True)
            for spec in STRATEGIES:
                spec.config_path(upstream, "squad").touch()
            paths = PilotPaths(root, upstream, upstream / "missing.jsonl", upstream / ".retriever_address.jsonnet", root / "results")
            with self.assertRaises(ResourceValidationError) as context:
                validate_resources(paths, "squad", check_dependencies=False)
            message = str(context.exception)
            self.assertIn("missing processed pilot dataset", message)
            self.assertIn("missing retriever address configuration", message)

            paths.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            paths.dataset_path.write_text('{}\n')
            paths.retriever_address_path.touch()
            fake_spacy = types.SimpleNamespace(util=types.SimpleNamespace(is_package=lambda _: False))
            with patch("importlib.util.find_spec", return_value=object()), patch.dict("sys.modules", {"spacy": fake_spacy}):
                with self.assertRaises(ResourceValidationError) as checkpoint_context:
                    validate_resources(paths, "squad", check_dependencies=True)
            self.assertIn("missing local spaCy checkpoint", str(checkpoint_context.exception))


if __name__ == "__main__":
    unittest.main()
