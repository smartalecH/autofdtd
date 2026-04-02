#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

STATUS_ACTIVE = {"QUEUED", "RUNNING"}
STATUS_TERMINAL = {"COMPLETED", "FAILED", "BLOCKED"}


@dataclass
class Task:
    task_id: str
    status: str
    title: str
    success: str
    attempts: int
    last_run: str
    last_duration: str
    completed_at: str
    notes: str


HEADING_RE = re.compile(r"^## \[(?P<status>[A-Z]+)\] (?P<task_id>[a-z0-9-]+) - (?P<title>.+)$")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def now_human() -> str:
    return datetime.now().astimezone().strftime("%B %d, %Y at %I:%M:%S %p %Z")


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def parse_spec(path: Path) -> tuple[str, list[Task]]:
    lines = path.read_text().splitlines()
    prefix: list[str] = []
    tasks: list[Task] = []
    i = 0
    while i < len(lines):
        match = HEADING_RE.match(lines[i])
        if not match:
            if not tasks:
                prefix.append(lines[i])
            i += 1
            continue
        block = [lines[i]]
        i += 1
        while i < len(lines) and not HEADING_RE.match(lines[i]):
            block.append(lines[i])
            i += 1
        data = {
            "status": match.group("status"),
            "task_id": match.group("task_id"),
            "title": match.group("title"),
            "success": "",
            "attempts": "0",
            "last_run": "-",
            "last_duration": "-",
            "completed_at": "-",
            "notes": "",
        }
        for line in block[1:]:
            if line.startswith("Success:"):
                data["success"] = line.split(":", 1)[1].strip()
            elif line.startswith("Attempts:"):
                data["attempts"] = line.split(":", 1)[1].strip()
            elif line.startswith("Last Run:"):
                data["last_run"] = line.split(":", 1)[1].strip()
            elif line.startswith("Last Duration:"):
                data["last_duration"] = line.split(":", 1)[1].strip()
            elif line.startswith("Completed At:"):
                data["completed_at"] = line.split(":", 1)[1].strip()
            elif line.startswith("Notes:"):
                data["notes"] = line.split(":", 1)[1].strip()
        tasks.append(
            Task(
                task_id=data["task_id"],
                status=data["status"],
                title=data["title"],
                success=data["success"],
                attempts=int(data["attempts"]) if data["attempts"].isdigit() else 0,
                last_run=data["last_run"],
                last_duration=data["last_duration"],
                completed_at=data["completed_at"],
                notes=data["notes"],
            )
        )
    return "\n".join(prefix).rstrip() + "\n\n", tasks


def write_spec(path: Path, header: str, tasks: Iterable[Task]) -> None:
    sections = [header.rstrip(), ""]
    for task in tasks:
        sections.extend(
            [
                f"## [{task.status}] {task.task_id} - {task.title}",
                f"Success: {task.success}",
                f"Attempts: {task.attempts}",
                f"Last Run: {task.last_run}",
                f"Last Duration: {task.last_duration}",
                f"Completed At: {task.completed_at}",
                f"Notes: {task.notes}",
                "",
            ]
        )
    path.write_text("\n".join(sections).rstrip() + "\n")


def append_event(log_path: Path, payload: dict) -> None:
    with log_path.open("a") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def append_console(console_path: Path, message: str = "") -> None:
    print(message, flush=True)
    with console_path.open("a") as fh:
        fh.write(message + "\n")


def extract_agent_text(event: dict) -> str | None:
    msg = event.get("msg")
    if isinstance(msg, str) and msg.strip():
        return msg.strip()
    text = event.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    delta = event.get("delta")
    if isinstance(delta, str) and delta.strip():
        return delta.strip()
    message = event.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text_val = item.get("text")
                    if isinstance(text_val, str) and text_val.strip():
                        parts.append(text_val.strip())
            if parts:
                return " ".join(parts)
    return None


def render_structured_event(event: dict) -> str | None:
    event_type = event.get("type")
    detail = event.get("detail")
    step = event.get("step")
    if isinstance(event_type, str) and event_type.strip():
        normalized = event_type.strip().lower().replace("-", "_")
        hidden_types = {
            "token_usage",
            "usage",
            "metrics",
            "trace",
            "debug",
            "delta",
            "raw_event",
            "heartbeat",
            "keepalive",
        }
        if normalized in hidden_types:
            return None
        label = event_type.strip().replace("_", " ").capitalize()
        parts = [label]
        if isinstance(step, int):
            parts.append(f"(step {step})")
        prefix = " ".join(parts)
        if isinstance(detail, str) and detail.strip():
            return f"{prefix}: {detail.strip()}"
        return prefix
    return None


def render_stream_line(raw_line: str) -> str:
    stripped = raw_line.rstrip("\n")
    if not stripped:
        return ""
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    text = extract_agent_text(payload)
    if text:
        return text
    structured = render_structured_event(payload)
    if structured:
        return structured

    if isinstance(payload, dict):
        noisy_keys = {"id", "uuid", "created_at", "timestamp", "ts", "latency_ms"}
        meaningful_keys = [key for key in payload.keys() if key not in noisy_keys]
        if not meaningful_keys:
            return ""
        if meaningful_keys == ["type"]:
            return ""
        compact = {key: payload[key] for key in meaningful_keys}
        if len(compact) <= 3:
            parts = []
            for key, value in compact.items():
                if isinstance(value, (str, int, float, bool)):
                    parts.append(f"{key}={value}")
            if parts:
                return "Event: " + ", ".join(parts)
        return ""

    return stripped


def render_task_prompt(prompt_path: Path, task: Task) -> str:
    base = prompt_path.read_text().rstrip()
    working_memory_path = prompt_path.parent / "working-memory.md"
    working_memory = ""
    if working_memory_path.exists():
        working_memory = working_memory_path.read_text().rstrip()
    return (
        f"{base}\n\n"
        "## Working Memory\n\n"
        f"{working_memory}\n\n"
        "## Current Task\n\n"
        f"Task ID: {task.task_id}\n"
        f"Title: {task.title}\n"
        f"Success: {task.success}\n"
        f"Notes: {task.notes}\n"
    )


def load_result(result_path: Path, task_id: str) -> dict:
    if not result_path.exists():
        return {
            "task_id": task_id,
            "status": "failed",
            "summary": "Codex did not produce a result.json payload.",
            "key_findings": [],
            "artifacts": [],
            "error_summary": "Missing result.json output.",
            "follow_up_notes": "",
            "next_action": "human_review",
        }
    with result_path.open() as fh:
        return json.load(fh)


def run_codex(
    bundle_dir: Path,
    working_dir: str,
    permission_mode: str,
    prompt_text: str,
    run_dir: Path,
    console_path: Path,
) -> int:
    cmd = [
        "codex",
        "exec",
        "-C",
        working_dir,
        "--json",
        "--sandbox",
        permission_mode,
        "--output-schema",
        str(bundle_dir / "output.schema.json"),
        "-o",
        str(run_dir / "result.json"),
        "-",
    ]
    env = os.environ.copy()
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=bundle_dir,
        env=env,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(prompt_text)
    process.stdin.close()

    with (run_dir / "codex.jsonl").open("w") as jsonl_fh, (run_dir / "codex.stdout.txt").open("w") as stdout_fh:
        for line in process.stdout:
            jsonl_fh.write(line)
            stdout_fh.write(line)
            rendered = render_stream_line(line)
            if rendered:
                append_console(console_path, rendered)
    return process.wait()


def clear_terminal_metadata(task: Task) -> None:
    task.attempts = 0
    task.last_run = "-"
    task.last_duration = "-"
    task.completed_at = "-"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    config_path = bundle_dir / "state" / "bundle-config.json"
    config = json.loads(config_path.read_text())
    max_iterations = args.max_iterations or int(config.get("max_iterations", 100))
    if max_iterations <= 0:
        raise SystemExit("max_iterations must be a positive integer")
    spec_path = bundle_dir / "loop.spec.md"
    prompt_path = bundle_dir / "loop.prompt.md"
    logs_dir = bundle_dir / "logs"
    runs_dir = bundle_dir / "runs"
    state_dir = bundle_dir / "state"
    logs_dir.mkdir(exist_ok=True)
    runs_dir.mkdir(exist_ok=True)
    state_dir.mkdir(exist_ok=True)
    events_path = logs_dir / "events.jsonl"
    console_path = logs_dir / "console.log"

    append_console(console_path, "Captain Wiggum loop starting")
    append_console(console_path, f"Max iterations this run: {max_iterations}")
    started = False
    iteration_count = 0

    while True:
        if iteration_count >= max_iterations:
            append_console(console_path, "")
            append_console(console_path, "============================================")
            append_console(console_path, f"Reached max iterations ({max_iterations}) at {now_human()}")
            append_console(console_path, "Stop reason: iteration limit reached")
            append_event(
                events_path,
                {
                    "ts": now_iso(),
                    "run_id": "loop",
                    "event": "loop_finished",
                    "task_id": None,
                    "task_title": None,
                    "attempt": None,
                    "status": "finished",
                    "duration_s": None,
                    "tasks_remaining": len([task for task in parse_spec(spec_path)[1] if task.status in STATUS_ACTIVE]),
                    "paths": {},
                    "message": f"max iterations reached: {max_iterations}",
                },
            )
            return 0

        header, tasks = parse_spec(spec_path)
        active_tasks = [task for task in tasks if task.status in STATUS_ACTIVE]
        total_tasks = len(tasks)

        if not started:
            append_event(
                events_path,
                {
                    "ts": now_iso(),
                    "run_id": "loop",
                    "event": "loop_started",
                    "task_id": None,
                    "task_title": None,
                    "attempt": None,
                    "status": "started",
                    "duration_s": None,
                    "tasks_remaining": len(active_tasks),
                    "paths": {},
                    "message": f"Loop started with {len(active_tasks)} active tasks.",
                },
            )
            started = True

        if not active_tasks:
            exhaustion_mode = config["exhaustion_mode"]
            if exhaustion_mode == "recycle the list":
                for task in tasks:
                    if task.status in STATUS_TERMINAL:
                        task.status = "QUEUED"
                        clear_terminal_metadata(task)
                write_spec(spec_path, header, tasks)
                append_console(console_path, "")
                append_console(console_path, "============================================")
                append_console(console_path, f"Recycling task list at {now_human()}")
                append_console(console_path, "============================================")
                continue
            append_console(console_path, "")
            append_console(console_path, "============================================")
            append_console(console_path, f"All tasks complete at {now_human()}")
            append_console(console_path, f"Failure policy: {config['failure_mode']}")
            append_console(console_path, f"Permission mode: {config['permission_mode']}")
            append_console(console_path, f"Exhaustion mode: {exhaustion_mode}")
            append_console(console_path, f"Total tasks: {total_tasks}")
            append_event(
                events_path,
                {
                    "ts": now_iso(),
                    "run_id": "loop",
                    "event": "loop_finished",
                    "task_id": None,
                    "task_title": None,
                    "attempt": None,
                    "status": "finished",
                    "duration_s": None,
                    "tasks_remaining": 0,
                    "paths": {},
                    "message": exhaustion_mode,
                },
            )
            return 0

        task = active_tasks[0]
        iteration_count += 1
        task_index = next(index for index, item in enumerate(tasks, start=1) if item.task_id == task.task_id)
        task.status = "RUNNING"
        task.attempts += 1
        run_id = f"{task.task_id}-attempt-{task.attempts:03d}"
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        task.last_run = run_id
        write_spec(spec_path, header, tasks)

        prompt_text = render_task_prompt(prompt_path, task)
        (run_dir / "prompt.txt").write_text(prompt_text)
        append_console(console_path, "")
        append_console(console_path, "============================================")
        append_console(
            console_path,
            f"Beginning task {task_index}/{total_tasks} at time {now_human()}",
        )
        append_console(console_path, f"Loop iteration: {iteration_count}/{max_iterations}")
        append_console(console_path, f"Task: {task.task_id} - {task.title}")
        append_console(console_path, f"Attempt: {task.attempts}")
        append_console(console_path, "============================================")
        append_event(
            events_path,
            {
                "ts": now_iso(),
                "run_id": run_id,
                "event": "task_selected",
                "task_id": task.task_id,
                "task_title": task.title,
                "attempt": task.attempts,
                "status": task.status,
                "duration_s": None,
                "tasks_remaining": len(active_tasks) - 1,
                "paths": {"run_dir": str(run_dir)},
                "message": "Task selected and marked RUNNING.",
            },
        )
        start = datetime.now(timezone.utc)
        exit_code = run_codex(
            bundle_dir,
            config["codex_working_dir"],
            config["permission_mode"],
            prompt_text,
            run_dir,
            console_path,
        )
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        result = load_result(run_dir / "result.json", task.task_id)
        result_status = str(result.get("status", "failed"))
        summary = str(result.get("summary", "")).strip() or "No summary provided."
        key_findings = result.get("key_findings", [])
        follow_up = str(result.get("follow_up_notes", "")).strip()
        error_summary = str(result.get("error_summary", "")).strip()
        next_action = str(result.get("next_action", "none")).strip() or "none"
        artifact_list = result.get("artifacts", [])

        if result_status == "completed":
            task.status = "COMPLETED"
            task.completed_at = now_iso()
        elif result_status == "blocked":
            task.status = "BLOCKED"
            task.completed_at = "-"
        elif result_status == "needs_retry":
            task.status = "QUEUED"
            task.completed_at = "-"
        else:
            if config["failure_mode"] == "retry":
                task.status = "QUEUED"
            elif config["failure_mode"] == "pause for review":
                task.status = "BLOCKED"
            else:
                task.status = "FAILED"
            task.completed_at = "-"

        task.last_duration = f"{elapsed:.1f}s"
        write_spec(spec_path, header, tasks)

        remaining = len([item for item in tasks if item.status in STATUS_ACTIVE])
        append_event(
            events_path,
            {
                "ts": now_iso(),
                "run_id": run_id,
                "event": "codex_finished",
                "task_id": task.task_id,
                "task_title": task.title,
                "attempt": task.attempts,
                "status": task.status,
                "duration_s": round(elapsed, 1),
                "tasks_remaining": remaining,
                "paths": {
                    "run_dir": str(run_dir),
                    "result_json": str(run_dir / "result.json"),
                    "codex_jsonl": str(run_dir / "codex.jsonl"),
                    "codex_stdout": str(run_dir / "codex.stdout.txt"),
                },
                "message": summary,
                "exit_code": exit_code,
                "result_status": result_status,
                "artifacts": artifact_list,
                "key_findings": key_findings,
                "error_summary": error_summary,
                "follow_up_notes": follow_up,
                "next_action": next_action,
            },
        )
        if summary:
            append_console(console_path, "")
            append_console(console_path, f"Summary: {summary}")
        if isinstance(key_findings, list) and key_findings:
            append_console(console_path, "Key findings:")
            for finding in key_findings:
                append_console(console_path, f"- {finding}")
        if error_summary:
            append_console(console_path, f"Error summary: {error_summary}")
        if follow_up:
            append_console(console_path, f"Follow-up: {follow_up}")
        if isinstance(artifact_list, list) and artifact_list:
            append_console(console_path, "Reported artifacts:")
            for artifact in artifact_list:
                if isinstance(artifact, dict):
                    path = artifact.get("path", "")
                    description = artifact.get("description", "")
                    if path and description:
                        append_console(console_path, f"- {path}: {description}")
                    elif path:
                        append_console(console_path, f"- {path}")
        append_console(console_path, f"Next action: {next_action}")
        append_console(console_path, f"Run artifacts: {run_dir / 'result.json'}, {run_dir / 'codex.jsonl'}, {run_dir / 'codex.stdout.txt'}")

        if task.status == "COMPLETED":
            append_console(
                console_path,
                f"Task complete in {format_duration(elapsed)} at {now_human()}",
            )
            append_console(console_path, f"Tasks remaining: {remaining}")
            continue

        if result_status == "needs_retry":
            append_console(
                console_path,
                f"Task re-queued after {format_duration(elapsed)} at {now_human()}",
            )
            append_console(console_path, f"Tasks remaining: {remaining}")
            continue

        if config["failure_mode"] == "retry":
            append_console(
                console_path,
                f"Task failed in {format_duration(elapsed)} at {now_human()} and was re-queued by failure policy",
            )
            append_console(console_path, f"Tasks remaining: {remaining}")
            continue

        if config["failure_mode"] == "skip and continue":
            append_console(
                console_path,
                f"Task failed in {format_duration(elapsed)} at {now_human()} and was marked {task.status}",
            )
            append_console(console_path, f"Tasks remaining: {remaining}")
            continue

        append_console(
            console_path,
            f"Task stopped the loop in {format_duration(elapsed)} at {now_human()}",
        )
        append_console(console_path, f"Failure policy: {config['failure_mode']}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
