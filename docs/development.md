# Development

## Install

```bash
python -m pip install -e .[dev]
```

## Entry Points

```bash
pytest
ruff check .
mypy src
mkdocs build
```

## Packaging Notes

- `pyproject.toml` uses `hatchling`.
- The wheel is built from `src/autofdtd`.
- `py.typed` is included so downstream type checking can consume typed modules.
