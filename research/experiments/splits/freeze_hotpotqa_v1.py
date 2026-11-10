"""Generate or verify the deterministic HotpotQA v1 experimental splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
SOURCE_ARCHIVE = REPO / "adaptive-rag/processed_data.tar.gz"
SOURCE_MEMBER = "./processed_data/hotpotqa/test_subsampled.jsonl"
PHASE0_MANIFEST = REPO / "research/results/hotpotqa_500_question_manifest_seed100.json"
OUTPUT = REPO / "research/data/splits/hotpotqa_v1"
SEED = 20261109
SPLIT_SIZES = {"calibration": 50, "validation": 25, "test": 150}
REQUIRED_FIELDS = {"dataset", "question_id", "question_text", "level", "type", "answers_objects", "contexts"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode() for row in rows)


def load_source() -> tuple[bytes, list[dict[str, Any]]]:
    with tarfile.open(SOURCE_ARCHIVE, "r:gz") as archive:
        raw = archive.extractfile(SOURCE_MEMBER).read()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    ids = [str(row.get("question_id")) for row in rows]
    if len(rows) != 500 or len(ids) != len(set(ids)):
        raise ValueError("source must contain exactly 500 unique questions")
    for row in rows:
        if set(row) != REQUIRED_FIELDS or row["dataset"] != "hotpotqa":
            raise ValueError(f"unexpected source schema for {row.get('question_id')}")
        if row["type"] not in {"bridge", "comparison"} or not row["answers_objects"] or not row["contexts"]:
            raise ValueError(f"incomplete source record {row['question_id']}")
    return raw, rows


def category_allocations(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = Counter(str(row["type"]) for row in rows)
    total = sum(counts.values())
    allocations: dict[str, dict[str, int]] = {}
    for split, size in SPLIT_SIZES.items():
        ideal = {category: size * count / total for category, count in counts.items()}
        assigned = {category: int(value) for category, value in ideal.items()}
        remainder = size - sum(assigned.values())
        order = sorted(counts, key=lambda category: (-(ideal[category] - assigned[category]), category))
        for category in order[:remainder]:
            assigned[category] += 1
        allocations[split] = assigned
    return allocations


def generate() -> tuple[dict[str, Any], dict[str, bytes]]:
    source_raw, source_rows = load_source()
    phase0 = json.loads(PHASE0_MANIFEST.read_text(encoding="utf-8"))
    phase0_ids = {str(qid) for qid in phase0["question_ids"]}
    source_ids = {str(row["question_id"]) for row in source_rows}
    if len(phase0_ids) != 500 or source_ids & phase0_ids:
        raise ValueError("source overlaps Phase 0 or Phase 0 does not contain 500 IDs")
    allocations = category_allocations(source_rows)
    pools: dict[str, list[dict[str, Any]]] = {}
    rng = random.Random(SEED)
    for category in sorted({str(row["type"]) for row in source_rows}):
        pool = sorted((row for row in source_rows if row["type"] == category), key=lambda row: row["question_id"])
        rng.shuffle(pool)
        pools[category] = pool
    offsets = {category: 0 for category in pools}
    selected: dict[str, list[dict[str, Any]]] = {}
    for split in SPLIT_SIZES:
        chosen: list[dict[str, Any]] = []
        for category in sorted(pools):
            count = allocations[split][category]
            start = offsets[category]
            chosen.extend(pools[category][start:start + count])
            offsets[category] += count
        selected[split] = sorted(chosen, key=lambda row: row["question_id"])
    split_sets = {name: {row["question_id"] for row in rows} for name, rows in selected.items()}
    if any(len(selected[name]) != size or len(split_sets[name]) != size for name, size in SPLIT_SIZES.items()):
        raise AssertionError("split size validation failed")
    if any(split_sets[a] & split_sets[b] for i, a in enumerate(SPLIT_SIZES) for b in list(SPLIT_SIZES)[i + 1:]):
        raise AssertionError("split overlap validation failed")
    if set().union(*split_sets.values()) & phase0_ids:
        raise AssertionError("Phase 0 exclusion failed")
    files: dict[str, bytes] = {}
    for split, rows in selected.items():
        files[f"{split}.jsonl"] = canonical_jsonl(rows)
        safe = [
            {key: row[key] for key in ("dataset", "question_id", "question_text", "level", "type")}
            for row in rows
        ]
        files[f"inference/{split}.jsonl"] = canonical_jsonl(safe)
    metadata = {
        "schema_version": 1,
        "dataset": "hotpotqa",
        "version": "hotpotqa_v1",
        "status": "candidate_frozen_pending_retrieval_runtime_verification",
        "seed": SEED,
        "selection": "Hamilton allocation by source question-type proportion; one seeded shuffle per sorted category; split order calibration, validation, test; records sorted by question_id",
        "split_sizes": SPLIT_SIZES,
        "question_type_allocations": allocations,
        "source": {
            "archive_path": str(SOURCE_ARCHIVE.relative_to(REPO)),
            "archive_sha256": sha256_bytes(SOURCE_ARCHIVE.read_bytes()),
            "member": SOURCE_MEMBER,
            "member_sha256": sha256_bytes(source_raw),
            "records": len(source_rows),
            "question_types": dict(sorted(Counter(row["type"] for row in source_rows).items())),
        },
        "phase0_exclusion": {
            "manifest_path": str(PHASE0_MANIFEST.relative_to(REPO)),
            "manifest_sha256": sha256_bytes(PHASE0_MANIFEST.read_bytes()),
            "question_ids": len(phase0_ids),
            "overlap": 0,
        },
        "retrieval": {
            "intended_corpus_alias": "hotpotqa",
            "source_records_have_contexts": True,
            "live_index_verified": False,
            "reason": "localhost Elasticsearch was unavailable during generation",
        },
        "files": {},
        "validation": {
            "counts_valid": True,
            "unique_ids_valid": True,
            "splits_disjoint": True,
            "phase0_disjoint": True,
            "source_schema_valid": True,
            "answers_present_in_evaluation_files": True,
            "answers_absent_from_inference_safe_files": True,
            "same_manifests_for_both_target_llms": True,
            "live_retrieval_compatibility_valid": False,
        },
    }
    metadata["files"] = {name: {"sha256": sha256_bytes(data), "records": len(data.splitlines())} for name, data in files.items()}
    files["split_metadata.json"] = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    return metadata, files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    metadata, files = generate()
    if args.verify_existing:
        for name, expected in files.items():
            path = OUTPUT / name
            if not path.is_file() or path.read_bytes() != expected:
                raise SystemExit(f"verification failed: {path}")
        print("verified", OUTPUT)
        return 0
    conflicts = [OUTPUT / name for name in files if (OUTPUT / name).exists()]
    if conflicts:
        raise SystemExit("refusing to overwrite frozen outputs; use --verify-existing or obtain approval")
    for name, data in files.items():
        path = OUTPUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(json.dumps({"output": str(OUTPUT), "status": metadata["status"], "splits": SPLIT_SIZES}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
