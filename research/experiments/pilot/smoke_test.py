#!/usr/bin/env python3
"""Credential-gated, sequential smoke test for the two NRP pilot aliases."""

from __future__ import annotations

import argparse
import json
import os
import sys

from nrp_adapter import (
    NRPChatAdapter,
    NRPConfigurationError,
    NRPConfig,
    NRPRequestError,
    SUPPORTED_PILOT_MODELS,
)


SMOKE_PROMPT = "Reply with exactly: NRP pilot connection successful"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential NRP compatibility smoke test; no request is sent without --confirm-network."
    )
    parser.add_argument(
        "--confirm-network",
        action="store_true",
        help="Confirm that authenticated requests may be sent to the configured NRP endpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_network:
        print("Refusing to send requests without --confirm-network.", file=sys.stderr)
        return 2

    try:
        config = NRPConfig.from_env()
    except NRPConfigurationError as error:
        print(json.dumps({"status": "configuration_error", "error": str(error)}), file=sys.stderr)
        return 2

    max_tokens = _optional_positive_int("NRP_SMOKE_MAX_TOKENS")
    enable_thinking = _optional_bool("NRP_ENABLE_THINKING")
    adapter = NRPChatAdapter(config)
    failures = 0

    # Sequential by design: conservative even though NRP currently documents a
    # higher short-context concurrency ceiling for both candidate aliases.
    for model in SUPPORTED_PILOT_MODELS:
        try:
            result = adapter.generate(
                SMOKE_PROMPT,
                model=model,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
            record = result.to_dict()
            record["response_nonempty"] = bool(result.text.strip())
            # Do not print generated text: the smoke report tests structure and accounting.
            record.pop("text", None)
            print(json.dumps(record, sort_keys=True))
            if not result.text.strip():
                failures += 1
                print(
                    json.dumps({"model": model, "status": "error", "error_type": "empty_response"}),
                    file=sys.stderr,
                )
        except (NRPRequestError, NRPConfigurationError) as error:
            failures += 1
            print(
                json.dumps(
                    {
                        "model": model,
                        "status": "error",
                        "error_type": error.__class__.__name__,
                        "status_code": getattr(error, "status_code", None),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
    return 1 if failures else 0


def _optional_positive_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise NRPConfigurationError(f"{name} must be an integer") from error
    if value <= 0:
        raise NRPConfigurationError(f"{name} must be positive")
    return value


def _optional_bool(name: str) -> bool | None:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise NRPConfigurationError(f"{name} must be true or false")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NRPConfigurationError as error:
        print(json.dumps({"status": "configuration_error", "error": str(error)}), file=sys.stderr)
        raise SystemExit(2)
