"""Minimal NRP chat-completions adapter for research-owned pilot code.

This module deliberately has no dependency on the original Adaptive-RAG package and
never falls back to a different model. It uses the documented OpenAI-compatible
chat-completions HTTP contract directly so it can coexist with Adaptive-RAG's legacy
OpenAI SDK dependency.
"""

from __future__ import annotations

import json
import os
import random
import socket
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence


DEFAULT_API_BASE = "https://ellm.nrp-nautilus.io/v1"
SUPPORTED_PILOT_MODELS = ("gemma-small", "gemma")
TRANSIENT_HTTP_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


class NRPConfigurationError(ValueError):
    """Raised before a request when required configuration is absent or unsafe."""


class NRPRequestError(RuntimeError):
    """A sanitized request failure that never includes authorization headers."""

    def __init__(
        self,
        category: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        diagnostic: str | None = None,
    ):
        self.category = category
        self.status_code = status_code
        self.retryable = retryable
        self.diagnostic = diagnostic
        status = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"NRP request failed: {category}{status}")


@dataclass(frozen=True)
class NRPConfig:
    api_key: str
    api_base: str = DEFAULT_API_BASE
    timeout_seconds: float = 120.0
    max_retries: int = 3
    backoff_initial_seconds: float = 1.0
    backoff_max_seconds: float = 16.0
    max_concurrency: int = 1
    cache_salt: str | None = None

    @classmethod
    def from_env(cls) -> "NRPConfig":
        api_key = os.environ.get("NRP_API_KEY", "").strip()
        if not api_key:
            raise NRPConfigurationError("NRP_API_KEY is required")

        config = cls(
            api_key=api_key,
            api_base=os.environ.get("NRP_API_BASE", DEFAULT_API_BASE).strip().rstrip("/"),
            timeout_seconds=_env_float("NRP_TIMEOUT_SECONDS", 120.0, minimum=0.1),
            max_retries=_env_int("NRP_MAX_RETRIES", 3, minimum=0, maximum=10),
            backoff_initial_seconds=_env_float("NRP_BACKOFF_INITIAL_SECONDS", 1.0, minimum=0.0),
            backoff_max_seconds=_env_float("NRP_BACKOFF_MAX_SECONDS", 16.0, minimum=0.0),
            max_concurrency=_env_int("NRP_MAX_CONCURRENCY", 1, minimum=1, maximum=8),
            cache_salt=os.environ.get("NRP_CACHE_SALT") or None,
        )
        if not config.api_base.startswith("https://"):
            raise NRPConfigurationError("NRP_API_BASE must use HTTPS")
        if config.backoff_max_seconds < config.backoff_initial_seconds:
            raise NRPConfigurationError("NRP_BACKOFF_MAX_SECONDS must be >= NRP_BACKOFF_INITIAL_SECONDS")
        return config


@dataclass(frozen=True)
class GenerationResult:
    model: str
    text: str
    status: str
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    finish_reason: str | None
    request_id: str | None
    attempts: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Transport = Callable[[urllib.request.Request, float], tuple[int, Mapping[str, str], bytes]]


class NRPChatAdapter:
    """Generate text through NRP while retaining request-level accounting.

    The same ``generate`` method is intended for prompts constructed by strategy A,
    B, or C. Retrieval, prompt construction, stopping rules, and answer evaluation
    remain outside this adapter.
    """

    def __init__(
        self,
        config: NRPConfig,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self._transport = transport or _urlopen_transport
        self._sleep = sleep
        self._clock = clock

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | Sequence[str] | None = None,
        enable_thinking: bool | None = None,
    ) -> GenerationResult:
        if model not in SUPPORTED_PILOT_MODELS:
            raise NRPConfigurationError(
                f"Unsupported pilot model {model!r}; expected one of {SUPPORTED_PILOT_MODELS}"
            )
        if not prompt or not prompt.strip():
            raise NRPConfigurationError("prompt must not be empty")
        if max_tokens is not None and max_tokens <= 0:
            raise NRPConfigurationError("max_tokens must be positive when provided")

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        # The NRP docs explicitly recognize max_tokens. Other decoding parameters
        # are sent only when the caller opts in, pending deployment-specific smoke tests.
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if stop is not None:
            payload["stop"] = stop
        if self.config.cache_salt:
            payload["cache_salt"] = self.config.cache_salt
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        attempts_allowed = self.config.max_retries + 1
        started = self._clock()
        last_error: NRPRequestError | None = None

        for attempt in range(1, attempts_allowed + 1):
            request = urllib.request.Request(
                f"{self.config.api_base}/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                status_code, headers, response_body = self._transport(request, self.config.timeout_seconds)
                if status_code < 200 or status_code >= 300:
                    raise NRPRequestError(
                        "http_error",
                        status_code=status_code,
                        retryable=status_code in TRANSIENT_HTTP_STATUSES,
                        diagnostic=_sanitize_error_detail(response_body, self.config.api_key),
                    )
                parsed = _parse_chat_completion(response_body)
                latency = self._clock() - started
                usage = parsed.get("usage") or {}
                choice = parsed["choices"][0]
                return GenerationResult(
                    model=str(parsed.get("model") or model),
                    text=choice["message"]["content"],
                    status="ok",
                    latency_seconds=latency,
                    input_tokens=_optional_int(usage.get("prompt_tokens")),
                    output_tokens=_optional_int(usage.get("completion_tokens")),
                    total_tokens=_optional_int(usage.get("total_tokens")),
                    finish_reason=choice.get("finish_reason"),
                    request_id=parsed.get("id") or _header_value(headers, "x-request-id"),
                    attempts=attempt,
                )
            except NRPRequestError as error:
                last_error = error
            except (TimeoutError, socket.timeout):
                last_error = NRPRequestError("timeout", retryable=True)
            except urllib.error.HTTPError as error:
                try:
                    error_body = error.read(4096)
                except Exception:
                    error_body = b""
                last_error = NRPRequestError(
                    "http_error",
                    status_code=error.code,
                    retryable=error.code in TRANSIENT_HTTP_STATUSES,
                    diagnostic=_sanitize_error_detail(error_body, self.config.api_key),
                )
            except urllib.error.URLError:
                last_error = NRPRequestError("network_error", retryable=True)

            if last_error is None or not last_error.retryable or attempt == attempts_allowed:
                break
            delay = min(
                self.config.backoff_max_seconds,
                self.config.backoff_initial_seconds * (2 ** (attempt - 1)),
            )
            # Small jitter prevents synchronized retries while preserving a strict bound.
            self._sleep(delay * random.uniform(0.8, 1.2))

        assert last_error is not None
        raise last_error


def _parse_chat_completion(body: bytes) -> dict[str, Any]:
    try:
        parsed = json.loads(body.decode("utf-8"))
        choices = parsed["choices"]
        content = choices[0]["message"]["content"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise NRPRequestError("invalid_response", retryable=False) from error
    if not isinstance(content, str):
        raise NRPRequestError("invalid_response", retryable=False)
    return parsed


def _urlopen_transport(
    request: urllib.request.Request, timeout: float
) -> tuple[int, Mapping[str, str], bytes]:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, dict(response.headers.items()), response.read()


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


def _sanitize_error_detail(body: bytes, secret: str) -> str | None:
    """Extract a bounded server diagnostic without retaining credentials or raw bodies."""
    if not body:
        return None
    try:
        parsed = json.loads(body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        value = body.decode("utf-8", errors="replace")
    else:
        value: Any = parsed
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict):
                value = error.get("message") or error.get("detail") or error.get("type") or ""
            else:
                value = error or parsed.get("message") or parsed.get("detail") or ""
        if not isinstance(value, str):
            value = ""
    value = " ".join(value.split())[:500]
    if secret:
        value = value.replace(secret, "[REDACTED]")
    return value or None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise NRPConfigurationError(f"{name} must be an integer") from error
    if value < minimum or value > maximum:
        raise NRPConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value


def _env_float(name: str, default: float, *, minimum: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise NRPConfigurationError(f"{name} must be numeric") from error
    if value < minimum:
        raise NRPConfigurationError(f"{name} must be >= {minimum}")
    return value
