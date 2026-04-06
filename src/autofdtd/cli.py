"""Command-line entrypoints for package planning metadata."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence

from autofdtd.planning import feature_matrix


def _table_lines() -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for entry in feature_matrix():
        grouped[entry.family].append(f"{entry.feature} [{entry.status}]")

    lines = ["AutoFDTD Phase 1 feature matrix:"]
    for family in sorted(grouped):
        lines.append(f"- {family}:")
        for item in grouped[family]:
            lines.append(f"  - {item}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Render the bundled feature matrix in either JSON or a plain-text table."""

    parser = argparse.ArgumentParser(prog="autofdtd-feature-matrix")
    parser.add_argument("--format", choices=("table", "json"), default="table")
    args = parser.parse_args(list(argv) if argv is not None else None)

    entries = [
        {
            "family": entry.family,
            "feature": entry.feature,
            "status": entry.status,
            "notes": entry.notes,
        }
        for entry in feature_matrix()
    ]

    if args.format == "json":
        print(json.dumps(entries, indent=2))
    else:
        print("\n".join(_table_lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
