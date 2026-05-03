from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .utils import read_jsonl

KIRUNDI_MARKERS = {
    "ivyo",
    "kandi",
    "nivyo",
    "umuntu",
    "abantu",
    "mu",
    "ku",
    "cane",
    "neza",
    "amakuru",
    "uburyo",
    "ikintu",
}


def normalize_label(text: str, labels: list[str]) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z\s_-]", " ", text)
    tokens = set(text.replace("_", " ").replace("-", " ").split())
    for label in labels:
        if label.lower() in tokens or text.startswith(label.lower()):
            return label
    return "unknown"


def is_kirundi_label(label: str, prefixes: tuple[str, ...] = ("run", "rn")) -> bool:
    normalized = (label or "").lower()
    return normalized.startswith(prefixes) or "kirundi" in normalized


def fallback_kirundi_heuristic(text: str) -> tuple[str, float]:
    words = re.findall(r"[A-Za-z']+", (text or "").lower())
    if not words:
        return "unknown", 0.0
    marker_hits = sum(1 for word in words if word in KIRUNDI_MARKERS)
    score = marker_hits / max(1, min(len(words), 50))
    label = "heuristic_kirundi" if score >= 0.06 else "unknown"
    return label, min(1.0, score * 10)


def classify_language(text: str, lid_pipeline: Any | None = None) -> dict[str, Any]:
    if lid_pipeline is None:
        label, score = fallback_kirundi_heuristic(text)
        return {
            "language_label": label,
            "language_score": score,
            "is_kirundi": is_kirundi_label(label),
            "method": "fallback_heuristic",
        }

    result = lid_pipeline(text[:1000])
    if isinstance(result, list):
        result = result[0]
    label = result.get("label", "unknown")
    score = float(result.get("score", 0.0))
    return {
        "language_label": label,
        "language_score": score,
        "is_kirundi": is_kirundi_label(label),
        "method": "afrolid",
    }


def load_response_table(path: str | Path, model_key: str) -> pd.DataFrame:
    rows = read_jsonl(path)
    df = pd.DataFrame(rows)
    if "response" not in df.columns and "output" in df.columns:
        df = df.rename(columns={"output": "response"})
    df["model"] = model_key
    return df


def summarize_language_adherence(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("model")
        .agg(
            num_prompts=("prompt_id", "count"),
            pct_kirundi=("is_kirundi", lambda s: round(100 * float(s.mean()), 2)),
            notes=("method", lambda s: ", ".join(sorted(set(s)))),
        )
        .reset_index()
    )


def score_kirnews_predictions(df: pd.DataFrame, labels: list[str]) -> dict[str, Any]:
    y_true = [normalize_label(x, labels) for x in df["gold_label"]]
    y_pred = [normalize_label(x, labels) for x in df["prediction"]]
    scored_labels = labels + ["unknown"]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=scored_labels, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=scored_labels),
        "labels": scored_labels,
    }
