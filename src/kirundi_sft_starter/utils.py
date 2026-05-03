from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_dir(path: str | Path) -> Path:
    path = project_path(path)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    if os.environ.get("HUGGING_FACE_HUB_TOKEN") and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]

    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    if os.environ.get("TINKER_TOKEN") and not os.environ.get("TINKER_API_KEY"):
        os.environ["TINKER_API_KEY"] = os.environ["TINKER_TOKEN"]

    if os.environ.get("ADAPTION_TOKEN") and not os.environ.get("ADAPTION_API_KEY"):
        os.environ["ADAPTION_API_KEY"] = os.environ["ADAPTION_TOKEN"]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = project_path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = ensure_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def require_file(path: str | Path, hint: str = "") -> Path:
    path = project_path(path)
    if not path.exists():
        message = f"Missing required file: {path}"
        if hint:
            message += f"\nHint: {hint}"
        raise FileNotFoundError(message)
    return path
