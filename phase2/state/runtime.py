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
    blocked_by: str = ""  # comma-separated list of task IDs that block this task


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
            "blocked_by": "",
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
            elif line.startswith("Blocked By:"):
                data["blocked_by"] = line.split(":", 1)[1].strip()
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
                blocked_by=data["blocked_by"],
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
                f"Blocked By: {task.blocked_by}" if task.blocked_by else f"Blocked By: —",
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


def is_warning_line(stripped: str) -> bool:
    return stripped.lower().startswith("warning:")


def is_progress_line(stripped: str) -> bool:
    return stripped.startswith("  →") or stripped.startswith("  •")


def should_resume_after_prompt_echo(stripped: str) -> bool:
    return (
        stripped == "exec"
        or stripped.startswith("mcp startup:")
        or stripped.startswith("ERROR:")
        or is_warning_line(stripped)
        or is_progress_line(stripped)
        or stripped.startswith("OpenAI Codex")
        or stripped.startswith("Claude")
        or stripped.startswith("--------")
        or bool(re.match(r"^\d{4}-\d{2}-\d{2}T", stripped))
        or stripped in {"assistant", "tool", "reasoning", "final"}
    )


def is_top_level_stream_marker(stripped: str) -> bool:
    return (
        stripped == "exec"
        or stripped.startswith("mcp startup:")
        or stripped.startswith("ERROR:")
        or is_warning_line(stripped)
        or is_progress_line(stripped)
        or stripped.startswith("OpenAI Codex")
        or stripped.startswith("Claude")
        or stripped.startswith("--------")
        or bool(re.match(r"^\d{4}-\d{2}-\d{2}T", stripped))
        or stripped in {"assistant", "tool", "reasoning", "final"}
    )


def is_file_read_command(stripped: str) -> bool:
    read_markers = [
        "pwd",
        "sed -n",
        "cat ",
        "rg --files",
        "ls -R",
        "find ",
    ]
    return any(marker in stripped for marker in read_markers)


def tee_filtered_output(console_path: Path, stdout_fh, raw_line: str, state: dict) -> None:
    stdout_fh.write(raw_line)
    stdout_fh.flush()

    stripped = raw_line.rstrip("\n")
    if state["suppress_prompt_echo"]:
        if should_resume_after_prompt_echo(stripped):
            state["suppress_prompt_echo"] = False
            if not state["prompt_notice_emitted"]:
                notice = "[stdin prompt echo suppressed]"
                print(notice, flush=True)
                with console_path.open("a") as fh:
                    fh.write(notice + "\n")
                state["prompt_notice_emitted"] = True
            if not stripped:
                return
        else:
            return

    if stripped == "user":
        state["suppress_prompt_echo"] = True
        return

    if state["suppress_exec_payload"]:
        if is_top_level_stream_marker(stripped):
            state["suppress_exec_payload"] = False
        else:
            return

    if stripped == "exec":
        state["awaiting_exec_command"] = True
        state["last_exec_is_file_read"] = False
    elif state["awaiting_exec_command"]:
        state["awaiting_exec_command"] = False
        state["last_exec_is_file_read"] = is_file_read_command(stripped)
        if state["last_exec_is_file_read"] and " succeeded in " in stripped:
            state["last_exec_is_file_read"] = False
            state["suppress_exec_payload"] = True
    elif state["last_exec_is_file_read"] and " succeeded in " in stripped:
        state["last_exec_is_file_read"] = False
        state["suppress_exec_payload"] = True

    sys.stdout.write(raw_line)
    sys.stdout.flush()
    with console_path.open("a") as fh:
        fh.write(raw_line)

    if state["suppress_exec_payload"]:
        notice = "[file-read output suppressed]"
        print(notice, flush=True)
        with console_path.open("a") as fh:
            fh.write(notice + "\n")


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
    try:
        with result_path.open() as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return {
            "task_id": task_id,
            "status": "failed",
            "summary": "Codex produced an invalid or empty result.json payload.",
            "key_findings": [],
            "artifacts": [],
            "error_summary": "Invalid or empty result.json output.",
            "follow_up_notes": "",
            "next_action": "human_review",
        }


PERMISSION_MODE_MAP = {
    "danger-full-access": "bypassPermissions",
}


def run_claude(
    bundle_dir: Path,
    working_dir: str,
    permission_mode: str,
    prompt_text: str,
    run_dir: Path,
    console_path: Path,
    task_id: str = "",
) -> int:
    """Run Claude Code in headless (-p) mode using stream-json for reliable I/O."""
    env = os.environ.copy()
    # IS_SANDBOX=1 allows --dangerously-skip-permissions even as root
    env["IS_SANDBOX"] = "1"

    claude_mode = PERMISSION_MODE_MAP.get(permission_mode, permission_mode)

    # Resolve working_dir to absolute path and add reference repos
    working_path = Path(working_dir)
    if not working_path.is_absolute():
        working_path = (bundle_dir / working_dir).resolve()

    # Reference repos that the loop prompt references
    reference_repos = [
        "/workspace/tidy3d",
        "/workspace/meep",
        "/workspace/GeometryPrimitives.jl",
        "/workspace/libctl",
        "/workspace/VectorModesolver.jl",
        "/workspace/fdtdx",
        "/workspace/Khronos.jl",
        "/workspace/warp",
    ]

    # Build stream-json input
    input_data = json.dumps({
        "type": "user",
        "message": {"role": "user", "content": prompt_text}
    })

    cmd = [
        "claude", "-p",
        "--dangerously-skip-permissions",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
        "--add-dir", str(working_path),
    ]
    # Add reference repos
    for repo in reference_repos:
        if Path(repo).exists():
            cmd.extend(["--add-dir", repo])
        cmd.insert(3, claude_mode)

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=bundle_dir,
        env=env,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(input_data)
    process.stdin.close()

    # Capture stdout lines and look for result
    stdout_lines = []
    result_data = None
    last_message_type = None

    with (run_dir / "codex.stdout.txt").open("w") as stdout_fh:
        for line in process.stdout:
            stdout_fh.write(line)
            stdout_fh.flush()
            stdout_lines.append(line)

            # Try to parse as JSON and extract meaningful info
            try:
                data = json.loads(line)
                msg_type = data.get("type", "")

                # Only write important events to console
                if msg_type == "system":
                    # Skip system init messages
                    pass
                elif msg_type == "assistant" and data.get("message"):
                    content = data["message"].get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "thinking":
                                # Show thinking briefly
                                thought = item.get("thinking", "")[:100]
                                if thought:
                                    append_console(console_path, f"  → {thought}...")
                            elif item.get("type") == "tool_use":
                                tool_name = item.get("name", "unknown")
                                append_console(console_path, f"  • Using {tool_name}...")
                elif msg_type == "result":
                    result_data = data
                    append_console(console_path, f"\nResult received: {len(data.get('result', ''))} chars")

                last_message_type = msg_type
            except json.JSONDecodeError:
                pass

    exit_code = process.wait()

    # Prefer the agent's actual result from phase2/results/ if it exists
    result_path = run_dir / "result.json"
    agent_result_path = bundle_dir / "results" / f"{task_id}-result.json"
    if agent_result_path.exists():
        try:
            agent_result = json.loads(agent_result_path.read_text())
            # Validate it has the expected schema fields
            if isinstance(agent_result, dict) and "status" in agent_result:
                append_console(console_path, f"  → Using agent result from {agent_result_path}")
                result_path.write_text(json.dumps(agent_result, indent=2))
                return exit_code
        except (json.JSONDecodeError, OSError):
            pass  # Fall through to fabricated result

    # Build result.json in the expected schema format (fallback)
    if result_data:
        is_error = result_data.get("is_error", False)
        stop_reason = result_data.get("stop_reason", "")
        result_text = result_data.get("result", "")
        result = {
            "task_id": "",
            "status": "completed" if not is_error and stop_reason == "end_turn" else "failed",
            "summary": result_text[:4000] if result_text else "No result produced",
            "key_findings": [],
            "artifacts": [],
            "error_summary": result_data.get("error", "") if is_error else "",
            "follow_up_notes": f"stop_reason: {stop_reason}, duration_ms: {result_data.get('duration_ms', 0)}",
            "next_action": "none",
        }
    else:
        # No result found - timeout or error
        full_output = "".join(stdout_lines).strip()
        # Try to extract useful text from partial output
        summary_lines = []
        for line in stdout_lines[-50:]:
            try:
                data = json.loads(line)
                if data.get("type") == "assistant" and data.get("message", {}).get("content"):
                    content = data["message"]["content"]
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                text = item.get("text", "")[:200]
                                if text:
                                    summary_lines.append(text)
            except:
                pass

        summary = "\n".join(summary_lines[-3:]) if summary_lines else full_output[:500]
        result = {
            "task_id": "",
            "status": "needs_retry",
            "summary": f"Task did not complete. Exit code: {exit_code}. Partial output:\n{summary}",
            "key_findings": [],
            "artifacts": [],
            "error_summary": "",
            "follow_up_notes": f"exit_code: {exit_code}",
            "next_action": "retry",
        }
    result_path.write_text(json.dumps(result, indent=2))

    return exit_code


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
        bufsize=1,
        cwd=bundle_dir,
        env=env,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(prompt_text)
    process.stdin.close()

    filter_state = {
        "suppress_prompt_echo": False,
        "prompt_notice_emitted": False,
        "awaiting_exec_command": False,
        "last_exec_is_file_read": False,
        "suppress_exec_payload": False,
    }
    with (run_dir / "codex.stdout.txt").open("w") as stdout_fh:
        for line in process.stdout:
            tee_filtered_output(console_path, stdout_fh, line, filter_state)
    return process.wait()


def clear_terminal_metadata(task: Task) -> None:
    task.attempts = 0
    task.last_run = "-"
    task.last_duration = "-"
    task.completed_at = "-"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--agent", default="codex", choices=["codex", "claude"])
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
        working_dir = config["codex_working_dir"]
        if args.agent == "claude":
            exit_code = run_claude(
                bundle_dir,
                working_dir,
                config["permission_mode"],
                prompt_text,
                run_dir,
                console_path,
                task_id=task.task_id,
            )
        else:
            exit_code = run_codex(
                bundle_dir,
                working_dir,
                config["permission_mode"],
                prompt_text,
                run_dir,
                console_path,
            )
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        # Prefer agent-written honest result from bundle_dir/results/ over
        # the loop-copied fallback in run_dir/result.json
        agent_result_path = bundle_dir / "results" / f"task-{task.task_id}-result.json"
        if agent_result_path.exists():
            result = json.loads(agent_result_path.read_text())
        else:
            result = load_result(run_dir / "result.json", task.task_id)

        # Ground-truth guard: example tasks must compare FDTD output to analytical/semi-analytical
        # reference, not just validate multi-GPU self-consistency. If ground_truth_error is missing
        # or the comparison is self-consistency-only (single-vs-multi-GPU), force needs_retry.
        # Also check provenance to detect fabricated results.
        task_id_str = task.task_id or ""
        if task_id_str.startswith("task-") and task_id_str[5:].isdigit():
            task_num = int(task_id_str[5:])
            if 101 <= task_num <= 118:
                gt = result.get("ground_truth_error", {})
                if not gt or not isinstance(gt, dict):
                    append_console(console_path, f"  ⚠ ground_truth_error missing for {task.task_id} — forcing needs_retry")
                    result_status = "needs_retry"
                    result["status"] = "needs_retry"
                    result["summary"] = (result.get("summary") or "") + " [GROUND TRUTH MISSING]"
                elif not gt.get("reference_source"):
                    append_console(console_path, f"  ⚠ ground_truth_error.reference_source missing for {task.task_id} — forcing needs_retry")
                    result_status = "needs_retry"
                    result["status"] = "needs_retry"
                    result["summary"] = (result.get("summary") or "") + " [GROUND TRUTH MISSING]"

                # Provenance guard: check that physics actually ran (not fabricated)
                prov = result.get("provenance", {})
                e_max = prov.get("E_max", None)
                total_cells = prov.get("total_cells", 0)
                num_steps = prov.get("num_steps", 0)
                runtime = prov.get("runtime_seconds", 0)

                if e_max is None or total_cells == 0 or num_steps == 0:
                    append_console(console_path, f"  ⚠ provenance missing for {task.task_id} — forcing needs_retry")
                    result_status = "needs_retry"
                    result["status"] = "needs_retry"
                    result["summary"] = (result.get("summary") or "") + " [PROVENANCE MISSING]"
                elif e_max < 1e-10 and result.get("status") == "completed":
                    # Source injection bug — zero fields mean ground-truth comparison is meaningless
                    append_console(console_path, f"  ⚠ E_max={e_max:.2e} for {task.task_id} — source injection failed, forcing needs_retry")
                    result_status = "needs_retry"
                    result["status"] = "needs_retry"
                    result["summary"] = (result.get("summary") or "") + " [ZERO FIELDS — SOURCE INJECTION BUG]"
                elif runtime < 0.1 and total_cells > 1000:
                    # Simulation that large shouldn't finish in < 100ms — likely fabricated
                    append_console(console_path, f"  ⚠ runtime={runtime:.2f}s for {total_cells} cells — suspiciously fast, forcing needs_retry")
                    result_status = "needs_retry"
                    result["status"] = "needs_retry"
                    result["summary"] = (result.get("summary") or "") + " [RUNTIME SUSPICIOUS]"

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
            # Auto-unblock: any task that was blocked by this one is now runnable
            completed_id = task.task_id
            unblocked = []
            for t in tasks:
                if t.status == "BLOCKED" and t.blocked_by:
                    blockers = [b.strip() for b in t.blocked_by.split(",")]
                    if completed_id in blockers:
                        # Check all blockers are now COMPLETED
                        all_done = all(
                            any(x.task_id == b and x.status == "COMPLETED" for x in tasks)
                            for b in blockers
                        )
                        if all_done:
                            t.status = "QUEUED"
                            unblocked.append(t.task_id)
            if unblocked:
                append_console(
                    console_path,
                    f"  → Auto-unblocked: {', '.join(unblocked)}",
                )
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
        append_console(console_path, f"Run artifacts: {run_dir / 'result.json'}, {run_dir / 'codex.stdout.txt'}")

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
