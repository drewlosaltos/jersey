# Repository Guidelines

## Project Structure & Module Organization

`jersey_sdg` handles SkyBall jersey visibility labeling, reporting, and uncertainty-aware jersey-number training. `cli.py` exposes labeling/report commands, `jersey_number_train.py` contains the model and training loop, and `data.py`, `crops.py`, `sampling.py`, `vlm.py`, `reports.py`, and `io_utils.py` provide supporting utilities. Long-running operational scripts live in `scripts/`. Tests live in `tests/`. Generated local artifacts under `outputs/` are samples only; large data and training outputs should live under `/mnt/t/...`.

## Build, Test, and Development Commands

Run package commands from `/home/atao/vsdevel/skyball/jersey`:

```bash
conda run -n pt5090new python -m pytest jersey_sdg/tests
conda run -n pt5090new python -m py_compile jersey_sdg/jersey_number_train.py
```

Example training entry point:

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n pt5090new \
  python -m jersey_sdg.jersey_number_train --help
```

Use `vllm5090` for Qwen/vLLM labeling workflows.

## Coding Style & Naming Conventions

Use Python 3.12-compatible code, four-space indentation, `snake_case` names, and type hints where useful. Prefer `Path`, JSON/JSONL helpers, and structured parsing. Keep comments short and only where they clarify non-obvious behavior.

## Testing Guidelines

Tests use `pytest` and should be named `tests/test_*.py`. Add focused tests for parsing, crop geometry, sampling, and VLM response handling. For training changes, at minimum run `py_compile`; use a smoke run only when runtime and GPUs allow.

## Operational Rules

Do not overwrite completed experiment outputs. Use versioned names such as `..._v2_4x_bs128_lr2e4_tda_mb_e15_gpus23_augfix`. Record completed metrics in `../workflows/results.jsonl`.

