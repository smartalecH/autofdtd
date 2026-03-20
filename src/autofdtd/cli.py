from __future__ import annotations

from .bootstrap import main


def prepare_main() -> None:
    main(["prepare"])


def orchestrate_main() -> None:
    main()
