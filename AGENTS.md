# Repository Guidelines

## Project Structure & Module Organization
Core libraries live in `src/`, organized into factories (`data_factory`, `model_factory`, `task_factory`, `trainer_factory`) that assemble experiments from YAML configs. Experiment definitions sit in `configs/`, with ready-to-run demos under `configs/demo`. Tests and regression suites are in `test/`; keep new cases close to the module they exercise. Visualization/UI assets stay in `app/` with the Streamlit entrypoint in `streamlit_app.py`. Use `data/` for local datasets and `save/` for outputs—both are ignored by git. Notebooks, research artefacts, and helper utilities are staged in `dev/`, `docs/`, and `script/`.

## Build, Test, and Development Commands
Set up dependencies inside a virtual environment: `pip install -r requirements.txt`. Launch a standard experiment via `python main.py --config configs/demo/Single_DG/CWRU.yaml`; swap in other YAMLs or add `--pipeline Pipeline_02_pretrain_fewshot --config_path ...` for multi-stage runs. The GUI prototype starts with `streamlit run streamlit_app.py`. Run the full test pack using `pytest test/`, and scope runs when iterating, e.g. `pytest test/test_model_factory.py::TestNeuralOperators`. Capture metrics from `save/` and reference the relevant config in review notes.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents and a 100-character soft limit. Classes use `PascalCase`, functions and variables use `snake_case`, and module names should describe their role (`contrastive_dg_task.py`, `resnet_backbone.py`). Before submitting, tidy imports and formatting with `black src/ test/` and `isort src/ test/`; lint via `flake8` and type-check touched modules with `mypy src/`. Public APIs and complex utilities need NumPy-style docstrings and inline comments only where logic is non-obvious.

## Testing Guidelines
Mirror the existing parametrized style when covering new datasets, models, or trainers. Unit-level checks belong in the corresponding `test_*factory.py`, while cross-component flows go into the integration suites (for example `test/test_integration_multi_task.py`). Prefer lightweight fixtures and synthetic tensors to keep runtime manageable. Mark GPU-only or slow cases with `pytest.mark.skipif` guards and document any external dependencies in the test docstring.

## Commit & Pull Request Guidelines
Use concise, imperative commit subjects and the prevailing `type: summary` convention (`feat: add contrastive DG sampler`). Separate unrelated changes to simplify review. Pull requests should explain motivation, list the configs you touched, and attach key logs or plots (paths under `save/`). Reference related issues, flag breaking changes, and note any required dataset preparation so reviewers can rerun `python main.py` with the same YAML.
