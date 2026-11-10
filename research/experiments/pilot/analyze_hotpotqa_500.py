"""Strictly validate completed HotpotQA A/B/C runs and derive Table 2a artifacts."""

from __future__ import annotations

import hashlib
import json
import statistics
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "research/results/hotpotqa_500_question_manifest_seed100.json"
OUTPUT = REPO / "research/results/hotpotqa_500_analysis"
STRATEGIES = ("A", "B", "C")
MODELS = {
    "gemma-small": {
        "name": "Gemma 12B",
        "resolved": "google/gemma-4-12B-it-qat-w4a16-ct",
        "run": REPO / "research/results/pilot/20261105T181253Z_gemma-small_90f19188",
        "prefix": "6499ff1835d99de3357eed1deb3938940cec196c86b699940f84ecdf7cae9298",
    },
    "gemma": {
        "name": "Gemma 31B",
        "resolved": "google/gemma-4-31B-it-qat-w4a16-ct",
        "run": REPO / "research/results/pilot/20261105T182407Z_gemma_8248a157",
        "prefix": "5b403396164f04f04d00f6af096f32d5ef99a74a816a9dfbca2ef513c09ac7f0",
    },
}
REQUIRED = (
    "generated_answer", "f1", "exact_match", "end_to_end_latency", "total_tokens",
    "retrieval_calls", "llm_request_count", "trace_file", "retrieval_trace_file",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "mean_f1": mean(rows, "f1"),
        "exact_match": mean(rows, "exact_match"),
        "mean_latency_seconds": mean(rows, "end_to_end_latency"),
        "mean_total_tokens": mean(rows, "total_tokens"),
        "mean_retrieval_calls": mean(rows, "retrieval_calls"),
        "mean_api_requests": mean(rows, "llm_request_count"),
        "mean_input_tokens": mean(rows, "input_tokens"),
        "mean_output_tokens": mean(rows, "output_tokens"),
    }


def replace_exact(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        if new in text:
            return
        raise ValueError(f"expected progress-document text not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def validate_model(alias: str, manifest_ids: set[str]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    spec = MODELS[alias]
    rows = load_jsonl(spec["run"] / "results.jsonl")
    if len(rows) != 1500:
        raise ValueError(f"{alias}: expected exactly 1500 rows, found {len(rows)}")
    prefix = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows[:90])
    if hashlib.sha256(prefix).hexdigest() != spec["prefix"]:
        raise ValueError(f"{alias}: immutable 90-row pilot prefix changed")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        qid, strategy = str(row.get("question_id")), str(row.get("strategy"))
        key = (qid, strategy)
        if key in indexed:
            raise ValueError(f"{alias}: duplicate key {key}")
        if qid not in manifest_ids or strategy not in STRATEGIES:
            raise ValueError(f"{alias}: key outside manifest/strategies {key}")
        if row.get("completion_status") != "complete" or row.get("error_category") is not None:
            raise ValueError(f"{alias}: incomplete canonical row {key}")
        if row.get("dataset") != "hotpotqa" or row.get("model_alias") != alias:
            raise ValueError(f"{alias}: provenance mismatch {key}")
        if row.get("resolved_model_id") != spec["resolved"]:
            raise ValueError(f"{alias}: resolved model mismatch {key}")
        missing = [field for field in REQUIRED if row.get(field) is None]
        if missing:
            raise ValueError(f"{alias}: missing {missing} for {key}")
        indexed[key] = row
    expected = {(qid, strategy) for qid in manifest_ids for strategy in STRATEGIES}
    if set(indexed) != expected:
        raise ValueError(f"{alias}: canonical key set is not the 500 x A/B/C product")
    return rows, indexed


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ids = [str(value) for value in manifest["question_ids"]]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError("manifest does not contain exactly 500 unique question IDs")
    manifest_ids = set(ids)
    aggregates: dict[str, Any] = {}
    canonical_rows: dict[str, list[dict[str, Any]]] = {}
    per_question: list[dict[str, Any]] = []
    for alias, spec in MODELS.items():
        rows, indexed = validate_model(alias, manifest_ids)
        canonical_rows[alias] = rows
        fixed = {strategy: metrics([row for row in rows if row["strategy"] == strategy]) for strategy in STRATEGIES}
        best_fixed = max(STRATEGIES, key=lambda strategy: (fixed[strategy]["mean_f1"], -STRATEGIES.index(strategy)))
        selected_rows: list[dict[str, Any]] = []
        selection = {strategy: 0 for strategy in STRATEGIES}
        tie_count = 0
        strictly_better = 0
        optimal_set_counts = {key: 0 for key in ("A", "B", "C", "AB", "AC", "BC", "ABC")}
        all_failed_count = 0
        for qid in ids:
            qrows = {strategy: indexed[(qid, strategy)] for strategy in STRATEGIES}
            top = max(float(row["f1"]) for row in qrows.values())
            ties = [strategy for strategy in STRATEGIES if float(qrows[strategy]["f1"]) == top]
            optimal_set = "".join(ties)
            optimal_set_counts[optimal_set] += 1
            all_failed_count += optimal_set == "ABC" and top == 0
            chosen = ties[0]
            selected_rows.append(qrows[chosen])
            selection[chosen] += 1
            tie_count += len(ties) > 1
            strictly_better += top > float(qrows[best_fixed]["f1"])
            per_question.append({
                "model_alias": alias, "question_id": qid,
                "f1": {strategy: qrows[strategy]["f1"] for strategy in STRATEGIES},
                "gain_b_vs_a": float(qrows["B"]["f1"]) - float(qrows["A"]["f1"]),
                "gain_c_vs_a": float(qrows["C"]["f1"]) - float(qrows["A"]["f1"]),
                "maximizing_strategies": ties, "oracle_selected_strategy": chosen,
            })
        oracle = metrics(selected_rows)
        aggregates[alias] = {
            "display_name": spec["name"], "question_count": 500, "execution_count": 1500,
            "fixed_strategies": fixed, "best_fixed_strategy": best_fixed,
            "best_fixed_mean_f1": fixed[best_fixed]["mean_f1"], "oracle": oracle,
            "oracle_gap": oracle["mean_f1"] - fixed[best_fixed]["mean_f1"],
            "oracle_selection_counts": selection,
            "oracle_selection_percent": {key: value / 5 for key, value in selection.items()},
            "highest_f1_tie_count": tie_count, "highest_f1_tie_percent": tie_count / 5,
            "oracle_strictly_better_than_best_fixed_count": strictly_better,
            "oracle_strictly_better_than_best_fixed_percent": strictly_better / 5,
            "optimal_strategy_set_counts": optimal_set_counts,
            "optimal_strategy_set_percent": {key: value / 5 for key, value in optimal_set_counts.items()},
            "all_failed_count": all_failed_count,
            "all_failed_percent": all_failed_count / 5,
        }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "validation": {
            "passed": True, "models": 2, "questions_per_model": 500, "executions": 3000,
            "question_id_sets_identical": True, "duplicate_model_question_strategy_keys": 0,
            "missing_model_question_strategy_keys": 0, "failed_canonical_records": 0,
            "historical_incomplete_attempts_excluded": {
                alias: len(load_jsonl(MODELS[alias]["run"] / "expansion_500_attempts.jsonl")) for alias in MODELS
            },
        },
        "models": aggregates,
    }
    (OUTPUT / "table2a_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT / "oracle_per_question.jsonl").open("w", encoding="utf-8") as stream:
        for row in per_question:
            stream.write(json.dumps(row) + "\n")
    tex_rows: list[str] = []
    report = [
        "# HotpotQA 500 Offline A/B/C Analysis",
        "",
        "The two canonical result files contain exactly 1,500 complete, unique question/strategy records each.",
        "Both models use the same 500 manifest question IDs. Historical incomplete attempts remain isolated",
        "outside the canonical JSONL files and are not included in any aggregate.",
        "",
    ]
    for alias in MODELS:
        model = aggregates[alias]
        percentages = model["optimal_strategy_set_percent"]
        tex_rows.append(
            f'HotpotQA & {model["display_name"]} & 500 & '
            + " & ".join(f'{percentages[key]:.1f}' for key in ("A", "B", "C", "AB", "AC", "BC", "ABC"))
            + f' & {model["all_failed_percent"]:.1f} \\tabularnewline'
        )
        report.extend([
            f'## {model["display_name"]}', "",
            "| Strategy | Mean F1 | Mean EM | Retrieval calls | API requests | Tokens | Latency (s) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for strategy in STRATEGIES:
            value = model["fixed_strategies"][strategy]
            report.append(
                f'| {strategy} | {value["mean_f1"]:.6f} | {value["exact_match"]:.6f} | '
                f'{value["mean_retrieval_calls"]:.6f} | {value["mean_api_requests"]:.6f} | '
                f'{value["mean_total_tokens"]:.3f} | {value["mean_latency_seconds"]:.6f} |'
            )
        report.extend(["", "Optimal-set counts (percent): " + ", ".join(
            f'{key} {model["optimal_strategy_set_counts"][key]} ({model["optimal_strategy_set_percent"][key]:.1f}%)'
            for key in ("A", "B", "C", "AB", "AC", "BC", "ABC")
        ), f'All-failed subset of ABC: {model["all_failed_count"]} ({model["all_failed_percent"]:.1f}%)', ""])
    (OUTPUT / "table2a_rows.tex").write_text("\n".join(tex_rows) + "\n", encoding="utf-8")
    (OUTPUT / "hotpotqa_500_offline_analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    csv_path = OUTPUT / "hotpotqa_500_tyler.csv"
    csv_fields = [
        "model_alias", "model", "strategy", "n", "mean_f1", "mean_em", "mean_retrieval_calls",
        "mean_api_requests", "mean_total_tokens", "mean_latency_seconds", "optimal_A_count", "optimal_B_count",
        "optimal_C_count", "optimal_AB_count", "optimal_AC_count", "optimal_BC_count", "optimal_ABC_count",
        "all_failed_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for alias, model in aggregates.items():
            for strategy in STRATEGIES:
                value = model["fixed_strategies"][strategy]
                writer.writerow({
                    "model_alias": alias, "model": model["display_name"], "strategy": strategy, "n": 500,
                    "mean_f1": f'{value["mean_f1"]:.6f}', "mean_em": f'{value["exact_match"]:.6f}',
                    "mean_retrieval_calls": f'{value["mean_retrieval_calls"]:.6f}',
                    "mean_api_requests": f'{value["mean_api_requests"]:.6f}',
                    "mean_total_tokens": f'{value["mean_total_tokens"]:.3f}',
                    "mean_latency_seconds": f'{value["mean_latency_seconds"]:.6f}',
                    **{f"optimal_{key}_count": model["optimal_strategy_set_counts"][key]
                       for key in ("A", "B", "C", "AB", "AC", "BC", "ABC")},
                    "all_failed_count": model["all_failed_count"],
                })
    summary = {
        "label": "HOTPOTQA EXPANDED A/B/C EVALUATION — 500 SHARED DEVELOPMENT QUESTIONS",
        "dataset": "hotpotqa", "seed": 100, "questions_per_model": 500,
        "question_ids_identical": True, "canonical_executions": 3000,
        "manifest_sha256": result["manifest_sha256"],
        "runs": {alias: MODELS[alias]["run"].name for alias in MODELS},
        "models": {},
    }
    for alias, model in aggregates.items():
        rows = canonical_rows[alias]
        attempts_path = MODELS[alias]["run"] / "expansion_500_attempts.jsonl"
        strategies = {}
        for strategy in STRATEGIES:
            value = model["fixed_strategies"][strategy]
            strategies[strategy] = {
                "completed": 500, "failed_canonical": 0, "mean_f1": value["mean_f1"],
                "mean_em": value["exact_match"], "mean_retrieval_calls": value["mean_retrieval_calls"],
                "mean_llm_calls": value["mean_api_requests"], "mean_input_tokens": value["mean_input_tokens"],
                "mean_output_tokens": value["mean_output_tokens"], "mean_total_tokens": value["mean_total_tokens"],
                "mean_end_to_end_latency_seconds": value["mean_latency_seconds"],
            }
        summary["models"][alias] = {
            "resolved_model": MODELS[alias]["resolved"], "coverage_percent": 100.0,
            "canonical_records": 1500, "historical_incomplete_attempts": len(load_jsonl(attempts_path)),
            "pooled_mean_end_to_end_latency_seconds": mean(rows, "end_to_end_latency"),
            "total_requests": sum(int(row["llm_request_count"]) for row in rows),
            "input_tokens": sum(int(row["input_tokens"]) for row in rows),
            "output_tokens": sum(int(row["output_tokens"]) for row in rows),
            "total_tokens": sum(int(row["total_tokens"]) for row in rows),
            "strategies": strategies,
            "optimal_strategy_sets": {
                key: {"count": model["optimal_strategy_set_counts"][key],
                      "percent": model["optimal_strategy_set_percent"][key]}
                for key in ("A", "B", "C", "AB", "AC", "BC", "ABC")
            },
            "all_failed": {"count": model["all_failed_count"], "percent": model["all_failed_percent"]},
        }
    all_rows = [row for rows in canonical_rows.values() for row in rows]
    summary["totals"] = {
        "canonical_records": len(all_rows),
        "historical_incomplete_attempts": sum(value["historical_incomplete_attempts"] for value in summary["models"].values()),
        "requests": sum(int(row["llm_request_count"]) for row in all_rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in all_rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in all_rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in all_rows),
    }
    (REPO / "research/results/pilot/phase0_hotpotqa_pilot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "validation.json").write_text(
        json.dumps({"generated_utc": result["generated_utc"], **result["validation"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    table_path = REPO / "research/paper/tables/table2_label_validity.tex"
    table_text = table_path.read_text(encoding="utf-8")
    begin, end = "% TABLE2A_ROWS_BEGIN", "% TABLE2A_ROWS_END"
    if table_text.count(begin) != 1 or table_text.count(end) != 1:
        raise ValueError("Table 2a generated-row markers are missing or duplicated")
    before, remainder = table_text.split(begin, 1)
    _old_rows, after = remainder.split(end, 1)
    table_path.write_text(
        before + begin + "\n" + "\n".join(tex_rows) + "\n" + end + after,
        encoding="utf-8",
    )
    plan_path = REPO / "research/docs/EXPERIMENT_PLAN.md"
    replace_exact(
        plan_path,
        "| E0.2 | Evaluate A/B/C on the same questions | **Intermediate:** aligned per-question predictions, EM/F1, retrieval counts, tokens, latency, and traces | Feeds T2–T4 | Completed for 30-question HotpotQA pilot; formal evaluation Pending |",
        "| E0.2 | Evaluate A/B/C on the same questions | **Intermediate:** aligned per-question predictions, EM/F1, retrieval counts, tokens, latency, and traces | Feeds T2–T4 | **Completed:** HotpotQA 500-question paired A/B/C evaluation for Gemma 12B and 31B |",
    )
    replace_exact(
        plan_path,
        "| E0.3 | Compute retrospective oracle upper bound | **Analytical only:** exact F1-optimal tie sets, oracle quality, and regret; never used for test routing | T2 | Optional |",
        "| E0.3 | Compute retrospective oracle upper bound | **Analytical only:** exact F1-optimal tie sets, oracle quality, and regret; never used for test routing | T2 | **Completed:** HotpotQA 500-question retrospective analysis |",
    )
    progress_path = REPO / "research/paper/tables/table0_progress.tex"
    replace_exact(
        progress_path,
        "E0.2 & Evaluate A/B/C on shared questions & \\textit{Intermediate:} aligned predictions, EM/F1, retrieval calls, tokens, latency, and traces & Feeds T2--T4 & Pilot Completed; Formal Pending & \\NA \\\\",
        "E0.2 & Evaluate A/B/C on shared questions & \\textit{Intermediate:} aligned predictions, EM/F1, retrieval calls, tokens, latency, and traces & Feeds T2--T4 & HotpotQA 500 Completed & \\NA \\\\",
    )
    replace_exact(
        progress_path,
        "E0.3 & Oracle upper-bound analysis & \\textit{Analytical only:} retrospective best-F1 tie sets, quality, and regret; never used for test routing & T2 & Optional & \\NA \\\\",
        "E0.3 & Oracle upper-bound analysis & \\textit{Analytical only:} retrospective best-F1 tie sets, quality, and regret; never used for test routing & T2 & HotpotQA 500 Completed & \\NA \\\\",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
