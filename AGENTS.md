# Repository Guidelines

## Project Structure & Module Organization
- Pipelines, data/model/task/trainer factories live in `src/`; add new logic via the matching factory registry to keep wiring modular.
- Experiment configs sit in `configs/` (start from `configs/demo/Single_DG/CWRU.yaml`); clone templates in `configs/experiments/` and keep local variants in their own subfolder.
- Runtime assets stay out of Git: raw data in `data/`, outputs in `save/`, visuals in `pic/`, docs in `docs/`; active tests in `test/` (legacy stress suites in `tests/`).
- See `src/*_factory/CLAUDE.md` and `src/configs/CLAUDE.md` for deeper architecture notes.

## Build, Test, and Development Commands
- `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt` — set up the environment (add `dev/test_history/requirements-test.txt` when extending pytest suites).
- `python main.py --config configs/demo/Single_DG/CWRU.yaml` — run the default CWRU baseline; swap in other YAMLs or use `load_config` overrides.
- `python scripts/hse_synthetic_demo.py` — quick validation of the HSE pipeline.
- `python -m pytest test/` — run maintained tests; add `--cov=src --cov-report=term` for coverage or use `python dev/test_history/run_tests.py --unit` to mirror history.
- `streamlit run streamlit_app.py` — launch the monitoring UI for manual QA.

## Coding Style & Naming Conventions
- Follow PEP 8, 100-character lines, grouped imports, NumPy-style docstrings for public APIs.
- Naming: classes `PascalCase`, functions/vars `snake_case`, constants `UPPER_CASE`; config folders use `task_dataset_variant`.
- Keep YAML keys lowercase with hyphen-separated values. Avoid hard-coded paths; prefer config-driven defaults.
- Format before committing: `black src/ test/` and `isort src/ test/`; lint with `flake8` and static-check with `mypy src/`.

## Testing Guidelines
- Tests live in `test/`; name files `test_<feature>.py`. Mark long runs with `@pytest.mark.slow` and GPU paths with `@pytest.mark.gpu`.
- Target coverage on critical pipelines (`pytest --cov=src --cov-report=term`). Document accuracy/latency changes alongside commands when altering models.

## Commit & Pull Request Guidelines
- Commit messages: imperative mood, optional scoped prefixes (e.g., `docs(hse):`, `refactor:`); English unless updating Chinese-only docs.
- Keep commits focused (separate config, factory, and docs changes). Include problem statement, dataset/model impact, reproduction commands, and artifact paths under `save/<metadata>/<model>/<experiment>`.
- Link tracked issues; add screenshots or UI notes when relevant. Ensure any data-source change follows `SECURITY.md`.

## Architecture & Configuration Tips
- Factory pattern with registries for data, models, tasks, and trainers; pipelines include `Pipeline_01_default`, `Pipeline_02_pretrain_fewshot`, `Pipeline_03_multitask_pretrain_finetune`, and `Pipeline_ID`.
- Use `load_config('isfm', {'model.d_model': 512})` to apply presets with dot-notation overrides; prefer `copy().update()` for chained configs.
- Save outputs under `save/{metadata}/{model}/{task_trainer_timestamp}/` (checkpoints, metrics, logs, figures, config backup).
