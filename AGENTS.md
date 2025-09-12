# Repository Guidelines

## Repository Info
- Name: `specbox`
- Description: A simple tool to manipulate and visualize optical spectra for astronomical research.
- Author: Yuming Fu
- URL: https://github.com/rudolffu/specbox
- Email: fuympku@outlook.com

## Project Structure & Module Organization
- `specbox/` – Python package with modules and assets:
  - `basemodule/` core spectrum readers and utilities.
  - `qtmodule/` Qt/pyqtgraph UI components and threads.
  - `auxmodule/` small helpers.
  - `data/` packaged templates and example FITS/text files.
  - `docs/` guides (see `docs/VITUTORIAL.md`).
  - `examples/` runnable snippets (e.g., `examples/example_enhanced_viewer.py`).
- Root contains `setup.py`, `README.md`, and build artifacts (`build/`, `dist/`).

## Build, Test, and Development Commands
- Install (user): `python -m pip install .`
- Install (editable dev): `python -m pip install -e .`
- Run example (GUI): `python specbox/examples/example_enhanced_viewer.py`
- Package (wheel/sdist): `python -m pip install build && python -m build`

## Coding Style & Naming Conventions
- Python 3.7+; follow PEP 8, 4‑space indentation.
- Names: packages/modules `snake_case`, classes `CamelCase`, functions/vars `snake_case`.
- Prefer type hints and docstrings for public APIs.
- Keep Qt code responsive; avoid blocking the UI thread (use threads provided in `qtmodule`).

## Testing Guidelines
- No formal test suite yet; use examples as smoke tests.
- Add new tests under `tests/` with `pytest` when contributing features/bugfixes.
- Name tests `test_*.py`; include sample inputs and minimal fixtures.

## Commit & Pull Request Guidelines
- Commits: clear, imperative subject (<= 72 chars); include concise body explaining why.
- Reference issues (`Fixes #123`) when relevant.
- PRs must include: summary, behavior/UX screenshots for UI changes, usage notes, and doc updates (README or `docs/`).
- If adding packaged data, update `package_data` in `setup.py` and keep files small.

## Security & Configuration Tips
- Input files are user‑provided FITS/CSV; validate paths and handle I/O errors gracefully.
- GUI dependencies: PySide6 and pyqtgraph; document platform‑specific quirks in PRs.
- Avoid bundling large proprietary datasets in the repo.

## Agent Notes
- These guidelines apply repo‑wide. When editing files, preserve the existing layout and public APIs under `specbox/` unless a change is documented and versioned.
- Keep docs in sync: whenever updating `README.md` (especially dependencies, install or usage), mirror relevant changes in `specbox/docs/VITUTORIAL.md` in the same PR.
