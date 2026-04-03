#!/usr/bin/env python3
"""Monitor V14 runs and trigger report generation after completion."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis")
RESULTS_DIR = BASE_DIR / "results"
STATUS_JSON = RESULTS_DIR / "v14_monitor_status.json"
STATUS_MD = RESULTS_DIR / "v14_monitor_status.md"
AUTOMATION_LOG = RESULTS_DIR / "v14_automation.log"
LOCK_PATH = RESULTS_DIR / ".v14_monitor.lock"
PYTHON = Path("/home/v-seungplee/miniconda3/envs/llm-addiction/bin/python")
READY_MD_PATH = RESULTS_DIR / "v14_READY_FOR_APPROVAL.md"
NEXT_PROMPT_PATH = RESULTS_DIR / "v14_NEXT_CODEX_PROMPT.txt"

EXPECTED_FILES = {
    "exp1": "v14_exp1_llama_sm_perm20_*.json",
    "exp2a": "v14_exp2a_llama_ic_n100_*.json",
    "exp2b": "v14_exp2b_llama_mw_n100_*.json",
    "exp4": "v14_exp4_gemma_mw_n100_*.json",
    "exp5": "v14_exp5_gemma_sm_n100_*.json",
    "exp6": "v14_exp6_gemma_ic_n100_*.json",
    "exp3_mw2ic": "v14_exp3_mw2ic_*.json",
    "exp3_mw2sm": "v14_exp3_mw2sm_*.json",
    "exp3_ic2sm": "v14_exp3_ic2sm_*.json",
}


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def append_log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with AUTOMATION_LOG.open("a") as f:
        f.write(f"[{timestamp}] {message}\n")


def list_running_processes() -> list[str]:
    proc = run(["pgrep", "-af", "run_v14_(experiments|parallel)\\.py"])
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def latest_result_map() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    json_dir = RESULTS_DIR / "json"
    for key, pattern in EXPECTED_FILES.items():
        matches = sorted(json_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        out[key] = str(matches[-1]) if matches else None
    return out


def write_status(status: dict) -> None:
    STATUS_JSON.write_text(json.dumps(status, indent=2))
    lines = [
        "# V14 Monitor Status",
        "",
        f"- Timestamp: {status['timestamp']}",
        f"- State: {status['state']}",
        f"- Running processes: {len(status['running_processes'])}",
        "",
        "## Processes",
        "",
    ]
    if status["running_processes"]:
        lines.extend(f"- `{line}`" for line in status["running_processes"])
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Result Files",
        "",
    ])
    for key, value in status["result_files"].items():
        lines.append(f"- `{key}`: `{value or 'missing'}`")
    if status.get("assessment"):
        lines.extend([
            "",
            "## Assessment",
            "",
            f"- Recommendation: `{status['assessment'].get('recommendation')}`",
            f"- Note: {status['assessment'].get('recommendation_text')}",
        ])
    if READY_MD_PATH.exists():
        lines.extend([
            "",
            "## Artifacts",
            "",
            f"- Ready file: `{READY_MD_PATH}`",
        ])
    if NEXT_PROMPT_PATH.exists():
        lines.append(f"- Next prompt: `{NEXT_PROMPT_PATH}`")
    STATUS_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    LOCK_PATH.touch(exist_ok=True)
    with LOCK_PATH.open("r+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            append_log("Skipped run because another monitor instance is active.")
            return

        running = list_running_processes()
        result_files = latest_result_map()
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": "running" if running else "waiting_for_results",
            "running_processes": running,
            "result_files": result_files,
        }

        if running:
            write_status(status)
            append_log(f"V14 still running with {len(running)} processes.")
            return

        if any(path is None for path in result_files.values()):
            write_status(status)
            append_log("V14 processes are gone but not all result JSON files exist yet.")
            return

        append_log("All expected V14 JSON files found. Generating report summary.")
        gen = run([str(PYTHON), str(RESULTS_DIR / "generate_v14_report.py")])
        if gen.returncode != 0:
            append_log(f"generate_v14_report.py failed: {gen.stderr.strip()}")
            write_status(status)
            return

        summary_path = RESULTS_DIR / "v14_latest_summary.json"
        if not summary_path.exists():
            append_log("Summary JSON was not created.")
            write_status(status)
            return

        summary = json.loads(summary_path.read_text())
        status["state"] = "completed"
        status["assessment"] = summary["assessment"]
        write_status(status)

        recommendation = summary["assessment"]["recommendation"]
        if recommendation == "positive":
            append_log("Positive recommendation detected. Building PDF report.")
            build = run([str(PYTHON), str(RESULTS_DIR / "build_v14_pdf.py")])
            if build.returncode != 0:
                append_log(f"build_v14_pdf.py failed: {build.stderr.strip()}")
                return
            append_log("V14 PDF report generated successfully.")
        else:
            append_log(f"Recommendation is {recommendation}. PDF auto-build skipped.")


if __name__ == "__main__":
    main()
