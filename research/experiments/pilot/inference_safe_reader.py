"""Model-facing reader for answer-free, context-free question manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


REQUIRED_FIELDS = {"dataset", "question_id", "question_text", "level", "type"}
FORBIDDEN_GOLD_FIELDS = {
    "answer",
    "answers",
    "answers_objects",
    "context",
    "contexts",
    "gold_contexts",
    "pinned_contexts",
    "supporting_facts",
}


class InferenceSafeManifestError(ValueError):
    """Raised when a purported inference-safe record is incomplete or contains gold data."""


def is_inference_safe_manifest(path: str | Path) -> bool:
    """Return true when the first record omits both upstream evaluation fields."""
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise InferenceSafeManifestError(f"invalid JSON at {path}:{line_number}") from error
            return "answers_objects" not in record and "contexts" not in record
    raise InferenceSafeManifestError(f"manifest contains no records: {path}")


def read_inference_safe_examples(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield the minimal state required by the unchanged A/B/C model graphs.

    The compatibility layer deliberately creates no answer, reference, real PID,
    pinned paragraph, or gold metadata fields. ``titles`` and ``paras`` start
    empty and can only be populated by the existing retrieval participants.
    """
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise InferenceSafeManifestError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise InferenceSafeManifestError(f"record is not an object at {path}:{line_number}")
            missing = REQUIRED_FIELDS - record.keys()
            forbidden = FORBIDDEN_GOLD_FIELDS & record.keys()
            if missing:
                raise InferenceSafeManifestError(
                    f"missing inference-safe fields at {path}:{line_number}: {', '.join(sorted(missing))}"
                )
            if forbidden:
                raise InferenceSafeManifestError(
                    f"gold fields are forbidden at the model boundary {path}:{line_number}: "
                    + ", ".join(sorted(forbidden))
                )
            question_id = str(record["question_id"])
            if question_id in seen_ids:
                raise InferenceSafeManifestError(f"duplicate question ID at {path}:{line_number}: {question_id}")
            seen_ids.add(question_id)
            question = str(record["question_text"])
            yield {
                "qid": question_id,
                "query": question,
                "question": question,
                "titles": [],
                "paras": [],
                "pids": [],
                "backup_titles": [],
                "backup_paras": [],
                "metadata": {
                    "level": record["level"],
                    "type": record["type"],
                    "answer_type": None,
                    "simplified_answer_type": None,
                    "pin_position": "no_op",
                },
            }
