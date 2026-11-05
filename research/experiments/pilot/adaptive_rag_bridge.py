"""Research-owned generator facade matching Adaptive-RAG's text interface.

This does not patch or import the upstream implementation. A future pilot runner can
inject this object at the narrow generator boundary while retaining upstream prompt,
retrieval, controller, and evaluation behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

try:
    from .nrp_adapter import NRPChatAdapter, NRPRequestError
except ImportError:  # Support direct execution from this directory.
    from nrp_adapter import NRPChatAdapter, NRPRequestError


class AdaptiveRAGNRPGenerator:
    """Expose ``generate_text_sequence`` and append request metadata as JSONL."""

    def __init__(
        self,
        adapter: NRPChatAdapter,
        *,
        model: str,
        metadata_path: str | Path,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        enable_thinking: bool | None = False,
    ) -> None:
        self.adapter = adapter
        self.model = model
        self.metadata_path = Path(metadata_path)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop
        self.enable_thinking = enable_thinking
        self._write_lock = Lock()

    def generate_text_sequence(self, prompt: str) -> list[tuple[str, float]]:
        try:
            result = self.adapter.generate(
                prompt,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=self.stop,
                enable_thinking=self.enable_thinking,
            )
        except NRPRequestError as error:
            self._append_metadata(
                {
                    "model": self.model,
                    "status": "error",
                    "error_type": error.category,
                    "status_code": error.status_code,
                    "retryable": error.retryable,
                }
            )
            raise
        self._append_metadata(result.to_dict())
        return [(result.text, 0.0)]

    def _append_metadata(self, record: dict[str, Any]) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock, self.metadata_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
