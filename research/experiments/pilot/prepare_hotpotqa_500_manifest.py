"""Create the deterministic 500-question expansion manifest without inference."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .pipeline import file_sha256, load_question_ids

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = REPO_ROOT / "adaptive-rag" / "processed_data" / "hotpotqa" / "dev_500_subsampled.jsonl"
PILOT_METADATA = REPO_ROOT / "research" / "results" / "pilot" / "20261105T181253Z_gemma-small_90f19188" / "run_metadata.json"
OUTPUT = REPO_ROOT / "research" / "results" / "hotpotqa_500_question_manifest_seed100.json"


def main() -> None:
    source_ids = load_question_ids(DATASET_PATH)
    pilot = json.loads(PILOT_METADATA.read_text(encoding="utf-8"))
    pilot_ids = [str(value) for value in pilot["selected_question_ids"]]
    if len(pilot_ids) != 30 or not set(pilot_ids) <= set(source_ids):
        raise RuntimeError("original 30-ID pilot manifest is invalid")
    remaining = sorted(set(source_ids) - set(pilot_ids))
    added_ids = random.Random(100).sample(remaining, 470)
    question_ids = sorted(pilot_ids + added_ids)
    payload = {
        "dataset": "hotpotqa",
        "source_split": "development dev_500_subsampled",
        "source_file": str(DATASET_PATH.relative_to(REPO_ROOT)),
        "source_sha256": file_sha256(DATASET_PATH),
        "seed": 100,
        "selection_method": "preserve audited 30 IDs; random.Random(100).sample(sorted(remaining IDs), 470); sort union",
        "preserved_pilot_count": 30,
        "added_count": 470,
        "limit": 500,
        "preserved_pilot_question_ids": sorted(pilot_ids),
        "question_ids": question_ids,
    }
    encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    OUTPUT.write_bytes(encoded)
    digest_path = OUTPUT.with_suffix(OUTPUT.suffix + ".sha256")
    digest_path.write_text(f"{hashlib.sha256(encoded).hexdigest()}  {OUTPUT.name}\n", encoding="utf-8")
    print(OUTPUT)
    print(digest_path)


if __name__ == "__main__":
    main()
