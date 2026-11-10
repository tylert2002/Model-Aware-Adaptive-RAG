"""Keep the append-only HotpotQA 500 expansion alive across terminal loss.

This supervisor never starts a second runner for a model whose command is already
present.  A resumed runner revalidates the immutable pilot prefix and skips every
canonical question/strategy key already completed.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
RESULTS = {
    "gemma-small": REPO / "research/results/pilot/20261105T181253Z_gemma-small_90f19188/results.jsonl",
    "gemma": REPO / "research/results/pilot/20261105T182407Z_gemma_8248a157/results.jsonl",
}
MANIFEST = REPO / "research/results/hotpotqa_500_question_manifest_seed100.json"
STATE_DIR = REPO / "research/results/hotpotqa_500_supervisor"
PYTHON = REPO / ".venv-pilot/bin/python"
POLL_SECONDS = 30
MAX_RESUME_ATTEMPTS_WITHOUT_PROGRESS = 5


class CriticalSupervisorError(RuntimeError):
    pass


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    with (STATE_DIR / "supervisor.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{stamp} {message}\n")
        stream.flush()


def completed_count(model: str) -> int:
    count = 0
    keys: set[tuple[str, str]] = set()
    with RESULTS[model].open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["question_id"]), str(row["strategy"]))
            if row.get("completion_status") != "complete" or key in keys:
                raise RuntimeError(f"invalid canonical record for {model}: {key}")
            keys.add(key)
            count += 1
    return count


def matching_pids(fragment: str) -> list[int]:
    result = subprocess.run(["pgrep", "-f", fragment], text=True, capture_output=True, check=False)
    return [int(value) for value in result.stdout.split() if int(value) != os.getpid()]


def runner_fragment(model: str) -> str:
    return f"research.experiments.pilot.expand_hotpotqa_500 --model {model} --manifest"


def ensure_retriever() -> None:
    if matching_pids("uvicorn serve:app --port 8000 --app-dir retriever_server"):
        return
    log("retriever absent; launching preserved port-8000 configuration")
    output = (STATE_DIR / "retriever.log").open("a", encoding="utf-8")
    subprocess.Popen(
        [str(PYTHON), "-m", "uvicorn", "serve:app", "--port", "8000", "--app-dir", "retriever_server"],
        cwd=REPO / "adaptive-rag",
        stdin=subprocess.DEVNULL,
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def ensure_runner(model: str, retry_state: dict[str, dict[str, int]]) -> None:
    count = completed_count(model)
    state = retry_state.setdefault(model, {"last_count": count, "attempts_without_progress": 0})
    if count > state["last_count"]:
        state["last_count"] = count
        state["attempts_without_progress"] = 0
    if count == 1500:
        return
    if count > 1500:
        raise RuntimeError(f"{model} has {count} canonical records; expected at most 1500")
    pids = matching_pids(runner_fragment(model))
    if pids:
        log(f"{model}: active pids={pids}, completed={count}/1500")
        return
    state["attempts_without_progress"] += 1
    attempt = state["attempts_without_progress"]
    if attempt > MAX_RESUME_ATTEMPTS_WITHOUT_PROGRESS:
        raise CriticalSupervisorError(
            f"{model}: runner stopped {attempt - 1} times without appending a valid record; manual intervention required"
        )
    backoff = min(60, 2 ** (attempt - 1))
    log(f"{model}: bounded resume attempt {attempt}/{MAX_RESUME_ATTEMPTS_WITHOUT_PROGRESS} after {backoff}s backoff")
    time.sleep(backoff)
    log(f"{model}: runner absent at {count}/1500; launching append-only resume")
    output = (STATE_DIR / f"{model}.log").open("a", encoding="utf-8")
    subprocess.Popen(
        [
            str(PYTHON),
            "-m",
            "research.experiments.pilot.expand_hotpotqa_500",
            "--model",
            model,
            "--manifest",
            str(MANIFEST.relative_to(REPO)),
            "--execute",
        ],
        cwd=REPO,
        stdin=subprocess.DEVNULL,
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def finalize() -> None:
    log("all canonical executions present; running strict validation and Table 2a analysis")
    analysis_log = (STATE_DIR / "finalization.log").open("a", encoding="utf-8")
    analysis = subprocess.run(
        [str(PYTHON), "-m", "research.experiments.pilot.analyze_hotpotqa_500"],
        cwd=REPO,
        stdin=subprocess.DEVNULL,
        stdout=analysis_log,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if analysis.returncode != 0:
        raise CriticalSupervisorError(f"strict validation/Table 2a analysis failed with exit code {analysis.returncode}")
    latexmk = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"],
        cwd=REPO / "research/paper",
        stdin=subprocess.DEVNULL,
        stdout=analysis_log,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if latexmk.returncode != 0:
        raise CriticalSupervisorError(f"paper compilation failed with exit code {latexmk.returncode}")
    log("strict validation, analysis, and paper compilation completed successfully")


def main() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_stream = (STATE_DIR / "supervisor.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("HotpotQA 500 supervisor is already running", file=sys.stderr)
        return 2
    lock_stream.write(str(os.getpid()))
    lock_stream.flush()
    stopping = False

    def stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log(f"supervisor started pid={os.getpid()}")
    retry_state_path = STATE_DIR / "retry_state.json"
    retry_state = (
        json.loads(retry_state_path.read_text(encoding="utf-8"))
        if retry_state_path.exists() else {}
    )
    try:
        while not stopping:
            ensure_retriever()
            for model in RESULTS:
                ensure_runner(model, retry_state)
            retry_state_path.write_text(json.dumps(retry_state, indent=2) + "\n", encoding="utf-8")
            counts = {model: completed_count(model) for model in RESULTS}
            (STATE_DIR / "checkpoint.json").write_text(
                json.dumps({"updated_utc": datetime.now(timezone.utc).isoformat(), "counts": counts}, indent=2) + "\n",
                encoding="utf-8",
            )
            if all(count == 1500 for count in counts.values()):
                log("all canonical executions complete: 3000/3000")
                finalize()
                return 0
            time.sleep(POLL_SECONDS)
    except CriticalSupervisorError as error:
        message = {"updated_utc": datetime.now(timezone.utc).isoformat(), "error": str(error)}
        (STATE_DIR / "critical_error.json").write_text(json.dumps(message, indent=2) + "\n", encoding="utf-8")
        log(f"CRITICAL: {error}")
        return 0
    log("supervisor stopped by signal")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
