# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

This repository holds the homework solutions for the MLZoomCamp machine learning course (pandas, numpy, scikit-learn, pytorch).

## Commands

- Run scripts **from the repo root** — they read data via relative paths like `data/course_lead_scoring_2026.csv`.
- Run tests: `pytest` (single test: `pytest tests/Homework3/test_ClassificationUtils.py::test_prepare_target`).
- There is no lint/build config; `pyproject.toml` only declares the package metadata (no dependencies listed). The package `MLzoomcamp` is installed in editable mode (see `src/MLzoomcamp.egg-info`).

## Structure

- `src/HomeworkN/main.py` — one script per course homework, executed top-to-bottom (no `__main__` guard). Shared helpers live next to them in `RegressionUtils.py` (Homework2) and `ClassificationUtils.py` (Homework3, a class of static methods such as `split_data`, `ohe_data`, `train`, `predict`).
- `src/Classification`, `src/LinearRegression` — lecture notebooks/scripts.
- `data/` — CSV datasets shared across homeworks.
- `tests/Homework3/` — pytest tests for `ClassificationUtils`.

## Gotchas

- Import style is inconsistent across homeworks: Homework2/3 `main.py` use `from RegressionUtils import *` / `from ClassificationUtils import *` (works only when the script's own directory is on `sys.path`), while Homework4 and the tests use the package-style `from Homework3.ClassificationUtils import ...` (requires `src` on the path, e.g. via the editable install). Match the style of the file you are editing.
- Homework4 reuses Homework3's utilities rather than copying them.
