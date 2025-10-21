# Repository Guidelines

## Project Structure & Module Organization
- `analysis/`: Standalone Python scripts for post-hoc analytics; run them with `python analysis/<script>.py`.
- `causal_feature_discovery/`: Main experimentation stack—`src/` contains runners, `Language-Model-SAEs/` provides the SAE library and UI, while `results/` and `logs/` store outputs.
- `gpt_experiments/`, `claude_experiment/`, and peers mirror that layout for model-specific pipelines.
- Shared assets sit in `figures/` and `writing/`; bulky raw data stays inside the existing `experiment_*` archives, and unit tests live under `causal_feature_discovery/Language-Model-SAEs/tests`.

## Build, Test, and Development Commands
- Install Python 3.11 dependencies from `causal_feature_discovery/Language-Model-SAEs/` with `pdm install`.
- Lint via `pdm run ruff check src tests`, adding `--fix` when appropriate.
- Format with `pdm run ruff format` to enforce 4-space indents, 120-character lines, and double quotes.
- Run the core test suite using `pdm run pytest tests/unit`; narrow scope with `-k` or node IDs during iteration.

## Coding Style & Naming Conventions
- Follow the Ruff configuration: snake_case modules, PascalCase classes, lowercase_with_underscores functions, double-quoted strings, and target Python 3.11.
- Prefer configuration files or CLI flags over hard-coded paths, and keep logs suffixed `.log` with data as `.json` or `.csv`.
- Add type hints on shared utilities and reserve inline comments for genuinely non-obvious control flow or numerical choices.

## Testing Guidelines
- Place new tests under `tests/unit` as `test_<module>.py`, grouping reusable fixtures at the top of each file.
- Cover success and failure paths with `pytest.raises` and mocks; skip GPU-heavy checks by default to preserve quick feedback.
- Refresh fixtures in `tests/data/` when formats change and note which `pytest` command you executed in the PR description.

## Commit & Pull Request Guidelines
- Git metadata is absent, so write short imperative commit subjects (e.g., `Add multiround GPT parser`) with optional body context.
- Keep commits tightly scoped and exclude generated artifacts or raw datasets from version control.
- Pull requests should state motivation, list verification commands (`pdm run pytest`, key scripts), link to the driving plan or issue, and attach updated outputs when they change.

## Experiment & Data Handling Tips
- Use existing launchers such as `launch_all_experiments.sh` or `gpt_experiments/src/run_gpt_50reps.sh`, noting overrides in `EXPERIMENT_STATUS_REPORT.md`.
- Store large checkpoints and transcripts in archival directories or external storage, referencing them by relative path, and keep secrets in untracked `.env` files documented alongside the consuming script.
