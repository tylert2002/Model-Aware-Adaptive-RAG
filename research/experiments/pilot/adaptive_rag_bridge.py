"""Research-owned generator facade matching Adaptive-RAG's text interface.

This does not patch or import the upstream implementation. A future pilot runner can
inject this object at the narrow generator boundary while retaining upstream prompt,
retrieval, controller, and evaluation behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Callable

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
        prompt_fitter: Callable[..., str] | None = None,
    ) -> None:
        self.adapter = adapter
        self.model = model
        self.metadata_path = Path(metadata_path)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop
        self.enable_thinking = enable_thinking
        self.prompt_fitter = prompt_fitter or fit_prompt_with_upstream
        self._write_lock = Lock()

    def generate_text_sequence(self, prompt: str) -> list[tuple[str, float]]:
        prompt = prompt.rstrip()
        prompt = self.prompt_fitter(
            original_prompt=prompt,
            model_length_limit=3500,
            estimated_generation_length=self.max_tokens if self.max_tokens is not None else 300,
            demonstration_delimiter="\n\n\n",
            shuffle=False,
            remove_method="first",
            tokenizer_model_name="gpt2",
            last_is_test_example=True,
        )
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
                    "diagnostic": error.diagnostic,
                }
            )
            raise
        self._append_metadata(result.to_dict())
        return [(result.text, 0.0)]

    def _append_metadata(self, record: dict[str, Any]) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock, self.metadata_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_local_gpt2_tokenizer() -> Any:
    """Load tokenizer assets from cache only; never initiate a download."""
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "GPT-2 prompt truncation requires the transformers package and a locally cached gpt2 tokenizer"
        ) from error
    try:
        return AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except Exception as error:
        raise RuntimeError(
            "GPT-2 tokenizer assets are not available locally; install/cache the gpt2 tokenizer before running "
            "the pilot (automatic download is disabled)"
        ) from error


def fit_prompt_with_upstream(**kwargs: Any) -> str:
    """Run the unchanged upstream fitting algorithm with a cache-only tokenizer."""
    from commaqa.inference import prompt_reader

    tokenizer = load_local_gpt2_tokenizer()
    original_get_tokenizer = prompt_reader.get_tokenizer
    try:
        prompt_reader.get_tokenizer = lambda model_name: tokenizer
        return prompt_reader.fit_prompt_into_given_limit(**kwargs)
    finally:
        prompt_reader.get_tokenizer = original_get_tokenizer
