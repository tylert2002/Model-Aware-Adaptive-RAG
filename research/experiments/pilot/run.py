"""CLI for the faithful Adaptive-RAG A/B/C pilot pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".pilot-cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(REPO_ROOT / ".pilot-cache" / "huggingface" / "transformers"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from .nrp_adapter import NRPChatAdapter, NRPConfig, NRPConfigurationError, SUPPORTED_PILOT_MODELS
from .pipeline import (
    SUPPORTED_DATASETS,
    ResourceValidationError,
    default_paths,
    dry_run_preflight,
    run_pilot,
    validate_resources,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run or validate the NRP-backed Adaptive-RAG pilot.")
    result.add_argument("--model", required=True, choices=SUPPORTED_PILOT_MODELS)
    result.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    result.add_argument("--limit", type=int, default=3)
    result.add_argument("--seed", type=int, default=100)
    result.add_argument(
        "--dataset-path",
        type=Path,
        help="Override the processed dataset JSONL path (defaults to the upstream dataset-specific filename).",
    )
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate local resources without credentials or network.")
    mode.add_argument("--execute", action="store_true", help="Explicitly authorize NRP inference and result creation.")
    return result


def load_env_file(path: Path) -> None:
    if not path.is_file():
        raise ResourceValidationError(f"missing API environment file: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ResourceValidationError(f"invalid environment assignment in {path}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.startswith("NRP_"):
            raise ResourceValidationError(f"unexpected environment variable {key!r} in {path}")
        # The task-local file is authoritative; stale shell values must not
        # override the credential or endpoint selected for an experiment run.
        os.environ[key] = value.strip().strip('"').strip("'")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = REPO_ROOT
    paths = default_paths(repo_root, args.dataset, args.dataset_path)
    try:
        if args.limit <= 0:
            raise ResourceValidationError("--limit must be positive")
        if args.dry_run:
            selected_ids = dry_run_preflight(paths, args.dataset, args.limit, args.seed)
            print(
                f"Dry-run validation passed for {args.dataset} with {len(selected_ids)} selected question(s); "
                "no credentials loaded and no requests sent."
            )
            return 0
        validate_resources(paths, args.dataset)
        load_env_file(Path(__file__).with_name(".env"))
        config = NRPConfig.from_env()
        adapter = NRPChatAdapter(config)
        run_dir = run_pilot(
            paths=paths,
            dataset=args.dataset,
            model=args.model,
            limit=args.limit,
            seed=args.seed,
            adapter=adapter,
        )
        print(f"Pilot artifacts written to {run_dir.relative_to(repo_root)}")
        rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        return 1 if rows and all(row.get("completion_status") != "complete" for row in rows) else 0
    except (ResourceValidationError, NRPConfigurationError, ImportError) as error:
        print(f"Pilot preflight failed:\n{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
