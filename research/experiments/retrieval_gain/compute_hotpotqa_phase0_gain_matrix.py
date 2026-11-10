"""Validate HotpotQA Phase 0 A/B/C results and generate Table 3 plus its gain matrix."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "research/results/hotpotqa_500_question_manifest_seed100.json"
DATASET = REPO / "adaptive-rag/processed_data/hotpotqa/dev_500_subsampled.jsonl"
OUTPUT = REPO / "research/results/hotpotqa_500_analysis/table3_gain_matrix.json"
ROWS_OUTPUT = REPO / "research/results/hotpotqa_500_analysis/table3_rows.tex"
TABLE = REPO / "research/paper/tables/table3_retrieval_gain.tex"
STRATEGIES = ("A", "B", "C")
MODELS = {
    "gemma-small": ("Gemma 12B", REPO / "research/results/pilot/20261105T181253Z_gemma-small_90f19188/results.jsonl"),
    "gemma": ("Gemma 31B", REPO / "research/results/pilot/20261105T182407Z_gemma_8248a157/results.jsonl"),
}


def load_index(path: Path, ids: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["question_id"]), str(row["strategy"]))
        if key in indexed:
            raise ValueError(f"duplicate A/B/C record: {key}")
        if row.get("completion_status") != "complete" or row.get("f1") is None:
            raise ValueError(f"invalid A/B/C F1 record: {key}")
        indexed[key] = row
    expected = {(qid, strategy) for qid in ids for strategy in STRATEGIES}
    if set(indexed) != expected:
        raise ValueError(f"{path} does not contain exactly 500 x A/B/C records")
    return indexed


def digest(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def replace_rows(rows: str) -> None:
    text = TABLE.read_text(encoding="utf-8")
    begin, end = "% TABLE3_ROWS_BEGIN", "% TABLE3_ROWS_END"
    if begin not in text or end not in text:
        raise ValueError("Table 3 generated-row markers are missing")
    prefix, remainder = text.split(begin, 1)
    _, suffix = remainder.split(end, 1)
    TABLE.write_text(prefix + begin + "\n" + rows.rstrip() + "\n" + end + suffix, encoding="utf-8")


def main() -> int:
    manifest_ids = {str(qid) for qid in json.loads(MANIFEST.read_text(encoding="utf-8"))["question_ids"]}
    dataset_rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset = {str(row["question_id"]): row for row in dataset_rows}
    if len(manifest_ids) != 500 or manifest_ids != set(dataset):
        raise ValueError("manifest and documented-category source are not the same 500 exact IDs")
    categories = sorted({str(dataset[qid]["type"]) for qid in manifest_ids})
    if categories != ["bridge", "comparison"]:
        raise ValueError(f"unexpected HotpotQA categories: {categories}")

    payload: dict[str, Any] = {
        "dataset": "hotpotqa",
        "phase": "Phase 0",
        "split": "local development dev_500_subsampled",
        "category_field": "type",
        "unique_questions": 500,
        "models": {},
    }
    tex_rows: list[str] = []
    for alias, (model_name, path) in MODELS.items():
        indexed = load_index(path, manifest_ids)
        model_rows: dict[str, Any] = {}
        for category in categories + ["Overall"]:
            qids = sorted(manifest_ids if category == "Overall" else [qid for qid in manifest_ids if dataset[qid]["type"] == category])
            means = {
                strategy: statistics.fmean(float(indexed[(qid, strategy)]["f1"]) for qid in qids)
                for strategy in STRATEGIES
            }
            gain_b = statistics.fmean(
                float(indexed[(qid, "B")]["f1"]) - float(indexed[(qid, "A")]["f1"]) for qid in qids
            )
            gain_c = statistics.fmean(
                float(indexed[(qid, "C")]["f1"]) - float(indexed[(qid, "A")]["f1"]) for qid in qids
            )
            if not math.isclose(gain_b, means["B"] - means["A"], abs_tol=1e-12):
                raise AssertionError("mean B-A gain is inconsistent")
            if not math.isclose(gain_c, means["C"] - means["A"], abs_tol=1e-12):
                raise AssertionError("mean C-A gain is inconsistent")
            best = max(means.values())
            preferred = [strategy for strategy in STRATEGIES if math.isclose(means[strategy], best, abs_tol=1e-12)]
            model_rows[category.lower()] = {
                "n": len(qids),
                "question_id_sha256": digest(qids),
                "mean_f1": means,
                "mean_gain_b_minus_a": gain_b,
                "mean_gain_c_minus_a": gain_c,
                "preferred_strategies": preferred,
            }
            display_category = category.capitalize() if category != "Overall" else category
            tex_rows.append(
                f'{model_name} & {display_category} & {len(qids)} & {means["A"]:.3f} & {means["B"]:.3f} & '
                f'{means["C"]:.3f} & {gain_b:+.3f} & {gain_c:+.3f} & {"/".join(preferred)} \\\\'
            )
        payload["models"][alias] = {"display_name": model_name, "inference_records": 1500, "categories": model_rows}
    rows_text = "\n".join(tex_rows)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    ROWS_OUTPUT.write_text(rows_text + "\n", encoding="utf-8")
    replace_rows(rows_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
