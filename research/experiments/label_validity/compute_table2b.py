"""Compute Table 2b from immutable HotpotQA A/B/C results and archived silver labels."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "research/results/hotpotqa_500_question_manifest_seed100.json"
DATASET = REPO / "adaptive-rag/processed_data/hotpotqa/dev_500_subsampled.jsonl"
LABEL_ARCHIVE = REPO / "adaptive-rag/data.tar.gz"
OUTPUT = REPO / "research/results/hotpotqa_500_analysis/table2b_metrics.json"
ROWS_OUTPUT = REPO / "research/results/hotpotqa_500_analysis/table2b_rows.tex"
TABLE = REPO / "research/paper/tables/table2_label_validity.tex"
STRATEGIES = ("A", "B", "C")
MODELS = {
    "gemma-small": ("Gemma 12B", REPO / "research/results/pilot/20261105T181253Z_gemma-small_90f19188/results.jsonl"),
    "gemma": ("Gemma 31B", REPO / "research/results/pilot/20261105T182407Z_gemma_8248a157/results.jsonl"),
}
LABEL_SOURCES = {
    "flan_t5_xl": ("FLAN-T5 XL", "./classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xl/silver/train.json"),
    "flan_t5_xxl": ("FLAN-T5 XXL", "./classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xxl/silver/train.json"),
    "gpt": ("GPT", "./classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/gpt/silver/train.json"),
}


def id_digest(ids: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def assert_same_question_ids(metric_ids: dict[str, set[str]]) -> None:
    if not metric_ids:
        raise ValueError("no metric question-ID sets supplied")
    first_name, first_ids = next(iter(metric_ids.items()))
    for name, ids in metric_ids.items():
        if ids != first_ids:
            raise ValueError(f"metric question-set mismatch: {name} differs from {first_name}")


def load_results(path: Path, manifest_ids: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["question_id"]), str(row["strategy"]))
        if key in indexed:
            raise ValueError(f"duplicate result {key} in {path}")
        if row.get("completion_status") != "complete" or row.get("f1") is None or row.get("total_tokens") is None:
            raise ValueError(f"invalid result {key} in {path}")
        indexed[key] = row
    expected = {(qid, strategy) for qid in manifest_ids for strategy in STRATEGIES}
    if set(indexed) != expected:
        raise ValueError(f"result key set in {path} is not the 500 x A/B/C product")
    return indexed


def compute_row(indexed: dict[tuple[str, str], dict[str, Any]], labels: dict[str, str]) -> dict[str, Any]:
    matched_ids = set(labels)
    if not matched_ids:
        raise ValueError("silver-label overlap is empty")
    auto_values: list[float] = []
    oracle_values: list[float] = []
    regrets: list[float] = []
    agreements: list[float] = []
    cost_agreements: list[float] = []
    metric_ids = {name: set() for name in ("auto_label_f1", "oracle_f1", "regret", "agreement", "cost_agreement")}
    for qid in sorted(matched_ids):
        qrows = {strategy: indexed[(qid, strategy)] for strategy in STRATEGIES}
        f1 = {strategy: float(qrows[strategy]["f1"]) for strategy in STRATEGIES}
        label = labels[qid]
        if label not in STRATEGIES:
            raise ValueError(f"invalid label {label!r} for {qid}")
        oracle = max(f1.values())
        optimal = {strategy for strategy in STRATEGIES if f1[strategy] == oracle}
        cheapest_tokens = min(float(qrows[strategy]["total_tokens"]) for strategy in optimal)
        cheapest = {strategy for strategy in optimal if float(qrows[strategy]["total_tokens"]) == cheapest_tokens}
        auto_values.append(f1[label])
        oracle_values.append(oracle)
        regrets.append(oracle - f1[label])
        agreements.append(float(label in optimal))
        cost_agreements.append(float(label in cheapest))
        for ids in metric_ids.values():
            ids.add(qid)
    assert_same_question_ids(metric_ids)
    auto_mean = statistics.fmean(auto_values)
    oracle_mean = statistics.fmean(oracle_values)
    regret_mean = statistics.fmean(regrets)
    if oracle_mean + 1e-12 < auto_mean:
        raise AssertionError("oracle mean F1 is below auto-label mean F1")
    if not math.isclose(regret_mean, oracle_mean - auto_mean, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("mean regret does not equal oracle mean minus auto-label mean")
    digest = id_digest(matched_ids)
    return {
        "n": len(matched_ids),
        "question_id_sha256": digest,
        "metric_question_id_sha256": {name: id_digest(ids) for name, ids in metric_ids.items()},
        "optimal_agreement_percent": 100.0 * statistics.fmean(agreements),
        "auto_label_mean_f1": auto_mean,
        "oracle_mean_f1": oracle_mean,
        "mean_f1_regret": regret_mean,
        "cost_aware_agreement_percent": 100.0 * statistics.fmean(cost_agreements),
        "all_zero_questions": sum(value == 0.0 for value in oracle_values),
    }


def replace_rows(table: Path, rows: str) -> None:
    text = table.read_text(encoding="utf-8")
    begin, end = "% TABLE2B_ROWS_BEGIN", "% TABLE2B_ROWS_END"
    if begin not in text or end not in text:
        raise ValueError("Table 2b row markers are missing")
    prefix, remainder = text.split(begin, 1)
    _, suffix = remainder.split(end, 1)
    table.write_text(prefix + begin + "\n" + rows.rstrip() + "\n" + end + suffix, encoding="utf-8")


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_ids = {str(qid) for qid in manifest["question_ids"]}
    dataset_rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset_text = {str(row["question_id"]): str(row["question_text"]) for row in dataset_rows}
    if len(manifest_ids) != 500 or manifest_ids != set(dataset_text):
        raise ValueError("manifest and local HotpotQA development subset are not the same 500 IDs")
    results = {alias: load_results(path, manifest_ids) for alias, (_, path) in MODELS.items()}
    labels: dict[str, dict[str, str]] = {}
    with tarfile.open(LABEL_ARCHIVE, "r:gz") as archive:
        for source, (_, member) in LABEL_SOURCES.items():
            records = json.load(archive.extractfile(member))
            hotpot = [row for row in records if row.get("dataset_name") == "hotpotqa"]
            by_id = {str(row["id"]): row for row in hotpot}
            if len(by_id) != len(hotpot) or not set(by_id) <= manifest_ids:
                raise ValueError(f"invalid or out-of-manifest silver IDs in {member}")
            if any(str(row["question"]) != dataset_text[qid] for qid, row in by_id.items()):
                raise ValueError(f"silver-label question text mismatch in {member}")
            labels[source] = {qid: str(row["answer"]) for qid, row in by_id.items()}
    output: dict[str, Any] = {"scope": "source-specific exact-ID matched subsets", "rows": [], "full_500_oracle_mean_f1": {}}
    tex_rows: list[str] = []
    for alias, (model_name, _) in MODELS.items():
        indexed = results[alias]
        output["full_500_oracle_mean_f1"][alias] = statistics.fmean(
            max(float(indexed[(qid, strategy)]["f1"]) for strategy in STRATEGIES) for qid in manifest_ids
        )
        for source, (source_name, member) in LABEL_SOURCES.items():
            row = compute_row(indexed, labels[source])
            row.update({"model_alias": alias, "model": model_name, "label_source": source, "label_source_name": source_name, "label_archive_member": member})
            output["rows"].append(row)
            tex_rows.append(
                f'HotpotQA & {model_name} & {source_name} & {row["n"]} & '
                f'{row["optimal_agreement_percent"]:.1f} & {row["auto_label_mean_f1"]:.3f} & '
                f'{row["oracle_mean_f1"]:.3f} & {row["mean_f1_regret"]:.3f} & '
                f'{row["cost_aware_agreement_percent"]:.1f} \\\\'
            )
    rows_text = "\n".join(tex_rows)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    ROWS_OUTPUT.write_text(rows_text + "\n", encoding="utf-8")
    replace_rows(TABLE, rows_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
