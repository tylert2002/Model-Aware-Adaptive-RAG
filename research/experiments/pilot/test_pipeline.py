from __future__ import annotations

import json
import os
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
    default_paths,
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
    def test_hotpotqa_default_path_uses_upstream_archive_filename(self):
        root = Path("/repo")
        paths = default_paths(root, "hotpotqa")
        self.assertEqual(
            paths.dataset_path,
            root / "adaptive-rag" / "processed_data" / "hotpotqa" / "dev_500_subsampled.jsonl",
        )

    def test_squad_default_path_uses_upstream_archive_filename(self):
        root = Path("/repo")
        paths = default_paths(root, "squad")
        self.assertEqual(
            paths.dataset_path,
            root / "adaptive-rag" / "processed_data" / "squad" / "dev_500_subsampled.jsonl",
        )

    def test_dataset_path_override_is_preserved(self):
        override = Path("/data/pilot.jsonl")
        self.assertEqual(default_paths(Path("/repo"), "hotpotqa", override).dataset_path, override)

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

    def test_bridge_uses_original_prompt_limit_parameters(self):
        captured = {}

        def fitter(**kwargs):
            captured.update(kwargs)
            return "fitted"

        adapter = NRPChatAdapter(NRPConfig(api_key="placeholder", max_retries=0), transport=fake_transport)
        with tempfile.TemporaryDirectory() as temp_dir:
            from .adaptive_rag_bridge import AdaptiveRAGNRPGenerator

            generator = AdaptiveRAGNRPGenerator(
                adapter,
                model="gemma-small",
                metadata_path=Path(temp_dir) / "trace.jsonl",
                max_tokens=300,
                prompt_fitter=fitter,
            )
            generator.generate_text_sequence("prompt   ")
        self.assertEqual(captured["original_prompt"], "prompt")
        self.assertEqual(captured["model_length_limit"], 3500)
        self.assertEqual(captured["estimated_generation_length"], 300)
        self.assertEqual(captured["demonstration_delimiter"], "\n\n\n")
        self.assertEqual(captured["remove_method"], "first")
        self.assertEqual(captured["tokenizer_model_name"], "gpt2")
        self.assertTrue(captured["last_is_test_example"])

    def test_upstream_prompt_fitter_removes_first_demonstration(self):
        repo_root = Path(__file__).resolve().parents[3]

        class Tokenizer:
            def tokenize(self, text):
                return text.split()

        with upstream_import_context(repo_root / "adaptive-rag"):
            from commaqa.inference import prompt_reader

            with patch.object(prompt_reader, "get_tokenizer", return_value=Tokenizer()):
                fitted = prompt_reader.fit_prompt_into_given_limit(
                    original_prompt="one two three\n\n\nfour five six\n\n\ntest question",
                    model_length_limit=8,
                    estimated_generation_length=2,
                    remove_method="first",
                    tokenizer_model_name="gpt2",
                )
        self.assertEqual(fitted, "four five six\n\n\ntest question")

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
            with patch("research.experiments.pilot.pipeline.score_answer", return_value=(1.0, 1.0)), patch(
                "research.experiments.pilot.adaptive_rag_bridge.fit_prompt_with_upstream",
                side_effect=lambda **kwargs: kwargs["original_prompt"],
            ):
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

    def test_env_file_overrides_stale_process_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("NRP_API_KEY=file-key\nNRP_API_BASE=https://example.test/v1\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"NRP_API_KEY": "stale-key", "NRP_API_BASE": "https://stale.test/v1"},
                clear=False,
            ):
                run.load_env_file(env_path)
                self.assertEqual(os.environ["NRP_API_KEY"], "file-key")
                self.assertEqual(os.environ["NRP_API_BASE"], "https://example.test/v1")

    def test_execute_returns_nonzero_when_all_strategies_fail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_root = Path(temp_dir)
            run_dir = fake_root / "run"
            run_dir.mkdir()
            (run_dir / "results.jsonl").write_text(
                "\n".join(
                    json.dumps({"strategy": strategy, "completion_status": "incomplete"})
                    for strategy in ("A", "B", "C")
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.object(run, "REPO_ROOT", fake_root), patch.object(run, "validate_resources"), patch.object(run, "load_env_file"), patch.object(
                run.NRPConfig, "from_env", return_value=object()
            ), patch.object(run, "NRPChatAdapter", return_value=object()), patch.object(
                run, "run_pilot", return_value=run_dir
            ):
                status = run.main(
                    ["--model", "gemma-small", "--dataset", "hotpotqa", "--limit", "1", "--execute"]
                )
        self.assertEqual(status, 1)

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
            self.assertIn(str(paths.dataset_path), message)

            paths.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            paths.dataset_path.write_text('{}\n')
            paths.retriever_address_path.touch()
            fake_spacy = types.SimpleNamespace(util=types.SimpleNamespace(is_package=lambda _: False))
            with patch("importlib.util.find_spec", return_value=object()), patch.dict("sys.modules", {"spacy": fake_spacy}), patch(
                "research.experiments.pilot.pipeline.load_local_gpt2_tokenizer", return_value=object()
            ):
                with self.assertRaises(ResourceValidationError) as checkpoint_context:
                    validate_resources(paths, "squad", check_dependencies=True)
            self.assertIn("missing local spaCy checkpoint", str(checkpoint_context.exception))

    def test_missing_local_gpt2_tokenizer_fails_preflight_clearly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            upstream = root / "adaptive-rag"
            dataset = upstream / "processed_data" / "squad" / "dev_subsampled.jsonl"
            dataset.parent.mkdir(parents=True)
            dataset.write_text('{"question_id":"q1"}\n')
            (upstream / "base_configs").mkdir()
            (upstream / "prompts" / "squad").mkdir(parents=True)
            for spec in STRATEGIES:
                spec.config_path(upstream, "squad").touch()
            retriever = upstream / ".retriever_address.jsonnet"
            retriever.touch()
            paths = PilotPaths(root, upstream, dataset, retriever, root / "results")
            fake_spacy = types.SimpleNamespace(util=types.SimpleNamespace(is_package=lambda _: True))
            with patch("importlib.util.find_spec", return_value=object()), patch.dict(
                "sys.modules", {"spacy": fake_spacy}
            ), patch(
                "research.experiments.pilot.pipeline.load_local_gpt2_tokenizer",
                side_effect=RuntimeError("GPT-2 tokenizer assets are not available locally"),
            ):
                with self.assertRaises(ResourceValidationError) as context:
                    validate_resources(paths, "squad")
            self.assertIn("GPT-2 tokenizer assets are not available locally", str(context.exception))


if __name__ == "__main__":
    unittest.main()
