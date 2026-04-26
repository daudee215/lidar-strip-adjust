# Contributing

Thank you for considering contributing to `lidar-strip-adjust`.

## Quick start

```bash
git clone https://github.com/daudee215/lidar-strip-adjust.git
cd lidar-strip-adjust
pip install uv
uv sync --all-extras
pre-commit install
```

## Development workflow

1. Open an issue describing the change before starting large work.
2. Fork the repository and create a feature branch from `main`.
3. Write code. Run the test suite locally before pushing:

```bash
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run mypy src/
```

4. All pre-commit hooks must pass (`pre-commit run --all-files`).
5. Open a pull request against `main`. Describe what the change does and why.
6. A maintainer will review within 7 days.

## Commit style

Use the imperative mood in commit subjects: "Add strip graph builder", not
"Added" or "Adding". Keep the subject under 72 characters. Reference issues
with `Fixes #N` in the body where applicable.

## Testing requirements

- New features must have unit tests covering the happy path and at least one
  edge case (empty input, wrong shape, non-overlapping strips).
- Tests live in `tests/unit/` (fast, no I/O) or `tests/integration/` (uses
  synthetic LAS files in `tests/data/`).
- Keep CI wall time under 3 minutes. Use small synthetic point clouds in tests.

## Code style

- Python ≥ 3.10. All public functions and classes must have docstrings.
- Type annotations on all public APIs (`mypy --strict`).
- Line length 99 characters (ruff default in `pyproject.toml`).
- No third-party imports outside the declared dependency list in
  `pyproject.toml`.

## Reporting bugs

Open a GitHub issue with:
- Python version and OS
- `lidar-strip-adjust --version` output
- Minimal reproducible example (ideally a tiny synthetic LAS pair)
- Full traceback

## Security issues

Do not open a public issue. See [SECURITY.md](security.md).
