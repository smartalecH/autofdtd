from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DOCS = [
    "framework-decision.md",
    "meep-gpu-core-spec.md",
    "geometry-gpu-core-spec.md",
    "geometry-and-domain-initialization.md",
    "fdtdx-and-systolic-notes.md",
    "vector-mode-solver-spec.md",
    "performance-benchmarking-spec.md",
    "gaps-and-investigation-areas.md",
    "tidy3d-compatibility-and-benchmarks.md",
    "mathematical-kernel-spec.md",
    "adjoint-and-discrepancy-notes.md",
]

RESULTS_HEADER = "timestamp\titem_id\ttrack\tstatus\tartifact\tmetric\tvalue\tnote\n"
VALID_STATUSES = {"pending", "active", "blocked", "complete"}


def repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "work_items.json").exists() and (candidate / "program.md").exists():
            return candidate
    raise SystemExit(
        "Could not locate the AutoFDTD repo root. Run this command from the repository or pass --root."
    )


def workspace_paths(root: Path) -> dict[str, Path]:
    workspace = root / "workspace"
    return {
        "workspace": workspace,
        "state": workspace / "orchestration_state.json",
        "index": workspace / "research_index.json",
        "results": root / "results.tsv",
        "work_items": root / "work_items.json",
        "references": root / "references.json",
    }


def ensure_workspace(root: Path) -> None:
    workspace = workspace_paths(root)["workspace"]
    for path in [
        workspace,
        workspace / "logs",
        workspace / "artifacts",
        workspace / "generated",
        workspace / "benchmarks",
        workspace / "reports",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_work_items(root: Path) -> dict:
    return json.loads(workspace_paths(root)["work_items"].read_text())


def load_references(root: Path) -> list[dict]:
    path = workspace_paths(root)["references"]
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return payload.get("references", [])


def ensure_state(root: Path) -> None:
    paths = workspace_paths(root)
    if paths["state"].exists():
        return
    work_items = load_work_items(root)
    state = {
        "items": {
            item["id"]: {
                "status": "pending",
                "history": [],
            }
            for item in work_items["items"]
        }
    }
    paths["state"].write_text(json.dumps(state, indent=2) + "\n")


def ensure_results(root: Path) -> None:
    results = workspace_paths(root)["results"]
    if not results.exists():
        results.write_text(RESULTS_HEADER)


def build_index(root: Path) -> dict:
    docs = []
    for rel in DOCS:
        path = root / rel
        docs.append(
            {
                "path": rel,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )

    references = []
    for ref in load_references(root):
        hinted = (root / ref["path_hint"]).resolve()
        references.append(
            {
                "id": ref["id"],
                "kind": ref["kind"],
                "role": ref.get("role", ""),
                "path_hint": ref["path_hint"],
                "available_locally": hinted.exists(),
            }
        )

    work_items = load_work_items(root)
    return {
        "root": ".",
        "documents": docs,
        "references": references,
        "work_items": {
            "count": len(work_items["items"]),
            "tracks": sorted({item["track"] for item in work_items["items"]}),
        },
    }


def prepare(root: Path) -> None:
    ensure_workspace(root)
    ensure_state(root)
    ensure_results(root)

    paths = workspace_paths(root)
    index = build_index(root)
    paths["index"].write_text(json.dumps(index, indent=2) + "\n")

    missing_docs = [doc["path"] for doc in index["documents"] if not doc["exists"]]
    local_refs = sum(1 for ref in index["references"] if ref["available_locally"])

    print("AutoFDTD bootstrap prepared")
    print("root: .")
    print("workspace: workspace")
    print(f"documents: {len(index['documents'])}")
    print(f"work_items: {index['work_items']['count']}")
    print(f"local_references_found: {local_refs}/{len(index['references'])}")
    print("results: results.tsv")
    print("index: workspace/research_index.json")
    if missing_docs:
        print("missing documents:")
        for path in missing_docs:
            print(f"  - {path}")


def load_state(root: Path) -> dict:
    state = workspace_paths(root)["state"]
    if not state.exists():
        raise SystemExit("Run `python3 prepare.py` or `autofdtd-prepare` first.")
    return json.loads(state.read_text())


def save_state(root: Path, state: dict) -> None:
    workspace_paths(root)["state"].write_text(json.dumps(state, indent=2) + "\n")


def current_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def indexed_items(root: Path) -> tuple[dict, dict]:
    work = load_work_items(root)
    state = load_state(root)
    items = {item["id"]: item for item in work["items"]}
    return items, state


def deps_satisfied(item: dict, state: dict) -> bool:
    return all(state["items"][dep]["status"] == "complete" for dep in item.get("depends_on", []))


def append_result(root: Path, item: dict, status: str, note: str, artifact: str, metric: str, value: str) -> None:
    row = "\t".join(
        [
            current_timestamp(),
            item["id"],
            item["track"],
            status,
            artifact,
            metric,
            value,
            note.replace("\t", " ").strip(),
        ]
    )
    with workspace_paths(root)["results"].open("a") as fh:
        fh.write(row + "\n")


def cmd_status(root: Path) -> None:
    items, state = indexed_items(root)
    counts = {status: 0 for status in VALID_STATUSES}
    for record in state["items"].values():
        counts[record["status"]] += 1

    print("queue status")
    for status in ["pending", "active", "blocked", "complete"]:
        print(f"  {status}: {counts[status]}")

    print("\nactive items")
    active = [item_id for item_id, record in state["items"].items() if record["status"] == "active"]
    if not active:
        print("  none")
    else:
        for item_id in active:
            print(f"  {item_id}: {items[item_id]['title']}")


def cmd_next(root: Path) -> None:
    items, state = indexed_items(root)
    pending = []
    for item in items.values():
        record = state["items"][item["id"]]
        if record["status"] != "pending":
            continue
        if not deps_satisfied(item, state):
            continue
        pending.append(item)
    pending.sort(key=lambda item: (item["priority"], item["track"], item["id"]))
    if not pending:
        print("DONE")
        return
    item = pending[0]
    print(item["id"])
    print(f"title: {item['title']}")
    print(f"track: {item['track']}")
    print(f"priority: {item['priority']}")
    print("outputs:")
    for output in item["outputs"]:
        print(f"  - {output}")


def cmd_list(root: Path, status_filter: str | None) -> None:
    items, state = indexed_items(root)
    rows = sorted(items.values(), key=lambda item: (item["priority"], item["track"], item["id"]))
    for item in rows:
        status = state["items"][item["id"]]["status"]
        if status_filter and status != status_filter:
            continue
        print(f"{item['id']}\t{status}\tP{item['priority']}\t{item['track']}\t{item['title']}")


def cmd_start(root: Path, item_id: str, note: str) -> None:
    items, state = indexed_items(root)
    item = items[item_id]
    if not deps_satisfied(item, state):
        raise SystemExit(f"Dependencies not complete for {item_id}")
    state["items"][item_id]["status"] = "active"
    state["items"][item_id]["history"].append(
        {"timestamp": current_timestamp(), "event": "start", "note": note}
    )
    save_state(root, state)
    print(f"STARTED {item_id}")


def cmd_record(root: Path, item_id: str, status: str, note: str, artifact: str, metric: str, value: str) -> None:
    if status not in VALID_STATUSES:
        raise SystemExit(f"Invalid status: {status}")
    items, state = indexed_items(root)
    item = items[item_id]
    state["items"][item_id]["status"] = status
    state["items"][item_id]["history"].append(
        {"timestamp": current_timestamp(), "event": "record", "note": note, "status": status}
    )
    save_state(root, state)
    append_result(root, item, status, note, artifact, metric, value)
    print(f"RECORDED {item_id} -> {status}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoFDTD bootstrap CLI")
    parser.add_argument("--root", default=None, help="Path to the AutoFDTD repo root")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare")
    sub.add_parser("status")
    sub.add_parser("next")

    list_parser = sub.add_parser("list")
    list_parser.add_argument("--status", choices=sorted(VALID_STATUSES))

    start_parser = sub.add_parser("start")
    start_parser.add_argument("item_id")
    start_parser.add_argument("--note", default="")

    record_parser = sub.add_parser("record")
    record_parser.add_argument("item_id")
    record_parser.add_argument("status", choices=sorted(VALID_STATUSES))
    record_parser.add_argument("--note", default="")
    record_parser.add_argument("--artifact", default="")
    record_parser.add_argument("--metric", default="")
    record_parser.add_argument("--value", default="")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else repo_root()

    if args.command == "prepare":
        prepare(root)
    elif args.command == "status":
        cmd_status(root)
    elif args.command == "next":
        cmd_next(root)
    elif args.command == "list":
        cmd_list(root, args.status)
    elif args.command == "start":
        cmd_start(root, args.item_id, args.note)
    elif args.command == "record":
        cmd_record(root, args.item_id, args.status, args.note, args.artifact, args.metric, args.value)
    else:
        raise SystemExit(f"Unhandled command: {args.command}")
