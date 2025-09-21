"""Helper utilities for TSPN hyperparameter sweeps."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import pandas as pd
import yaml

from src.configs.config_utils import load_config, save_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "hparam_eval"


@dataclass
class LaunchResult:
    """Container for subprocess execution outcomes."""

    returncode: int
    runtime: float
    timed_out: bool = False
    error: Optional[str] = None


def _ensure_cache_dir(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _merge_dict(target: MutableMapping[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if key in target and isinstance(target[key], MutableMapping) and isinstance(value, Mapping):
            _merge_dict(target[key], value)
        else:
            target[key] = value


def _normalize_overrides(overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert dotted keys to nested dictionaries while preserving existing mappings."""
    normalized: Dict[str, Any] = {}
    for key, value in overrides.items():
        if isinstance(value, Mapping) and not isinstance(value, dict):  # e.g., SimpleNamespace
            value = dict(value)  # type: ignore[arg-type]
        if isinstance(value, Mapping):
            value = _normalize_overrides(value)

        if isinstance(key, str) and "." in key:
            parts = key.split(".")
            cursor = normalized
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})  # type: ignore[assignment]
            cursor[parts[-1]] = value
        else:
            if isinstance(value, Mapping):
                existing = normalized.get(key, {})
                if isinstance(existing, MutableMapping):
                    _merge_dict(existing, value)
                    normalized[key] = existing
                    continue
            normalized[key] = value
    return normalized


def build_override(
    base_config: Path,
    overrides: Mapping[str, Any],
    run_name: str,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Materialise a temporary YAML config by merging overrides with the base file."""
    cache_root = _ensure_cache_dir(cache_dir or DEFAULT_CACHE_DIR)
    override_dict = _normalize_overrides(dict(overrides))
    config = load_config(str(base_config), override_dict)
    output_path = cache_root / f"{run_name}.yaml"
    save_config(config, output_path)
    return output_path


def launch_experiment(
    config_path: Path,
    *,
    log_path: Path,
    timeout: Optional[int] = None,
    extra_env: Optional[Mapping[str, str]] = None,
    pipeline: str | None = None,
    dry_run: bool = False,
) -> LaunchResult:
    """Execute ``main.py`` with the provided configuration."""
    start = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        log_path.write_text(
            f"[DRY-RUN] Would launch main.py with config {config_path}\n",
            encoding="utf-8",
        )
        return LaunchResult(returncode=0, runtime=0.0)

    command = [sys.executable or "python", "main.py", "--config_path", str(config_path)]
    if pipeline and pipeline != "Pipeline_01_default":
        command.extend(["--pipeline", pipeline])

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    # Run subprocess, streaming logs to file
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = _run_subprocess(command, log_file, timeout, env)
    except TimeoutError as exc:  # pragma: no cover - handled for safety
        runtime = time.time() - start
        return LaunchResult(returncode=-1, runtime=runtime, timed_out=True, error=str(exc))

    runtime = time.time() - start
    return LaunchResult(returncode=completed.returncode, runtime=runtime)


def _run_subprocess(command: Sequence[str], log_file, timeout: Optional[int], env: Mapping[str, str]):
    import subprocess

    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return completed
    except subprocess.TimeoutExpired as exc:
        log_file.write(f"\n[ERROR] Process timed out after {timeout} seconds.\n")
        raise TimeoutError(str(exc)) from exc


def load_metrics(log_dir: Path) -> Dict[str, Any]:
    """Read Lightning metric outputs and compute final values."""
    metrics_path = log_dir / "metrics.csv"
    if not metrics_path.exists():
        return {"summary": {}, "metrics_path": metrics_path, "data": None}

    df = pd.read_csv(metrics_path)
    summary: Dict[str, float] = {}
    for column in df.columns:
        if column.lower() in {"step", "epoch", "time"}:
            continue
        series = df[column].dropna()
        if not series.empty:
            summary[column] = float(series.iloc[-1])
    return {"summary": summary, "metrics_path": metrics_path, "data": df}


def export_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def export_yaml(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(data, fp, allow_unicode=True, sort_keys=False)


__all__ = [
    "LaunchResult",
    "build_override",
    "launch_experiment",
    "load_metrics",
    "export_json",
    "export_yaml",
]
