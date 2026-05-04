from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from .utils import ensure_dir, project_path, write_jsonl

THINKING_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def strip_reasoning_traces(text: str) -> str:
    text = THINKING_RE.sub("", text or "")
    return re.sub(r"\s+", " ", text).strip()


def parse_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    return []


def messages_to_pair(messages: list[dict[str, str]]) -> tuple[str, str]:
    user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]
    assistant_messages = [m.get("content", "") for m in messages if m.get("role") == "assistant"]

    instruction = user_messages[0].strip() if user_messages else ""
    response = strip_reasoning_traces(assistant_messages[-1]) if assistant_messages else ""
    return instruction, response


def truncate_text(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + " ... [truncated]"


def prepare_kakugo_subset(config: dict[str, Any]) -> pd.DataFrame:
    ds_cfg = config["datasets"]["sft"]
    dataset = load_dataset(ds_cfg["id"], split=ds_cfg["split"])
    dataset = dataset.shuffle(seed=config["project"]["random_seed"])
    dataset = dataset.select(range(min(ds_cfg["sample_size"], len(dataset))))

    max_chars = int(ds_cfg.get("max_chars_per_field", 4000))
    rows: list[dict[str, Any]] = []

    for idx, row in enumerate(dataset):
        instruction, response = messages_to_pair(parse_messages(row.get("messages")))
        if not instruction or not response:
            continue

        rows.append(
            {
                "example_id": f"kakugo_{idx:05d}",
                "instruction": truncate_text(instruction, max_chars),
                "response": truncate_text(response, max_chars),
                "generation_method": row.get("generation_method"),
                "prompt_type": row.get("prompt_type"),
                "topic": row.get("topic"),
                "scenario": row.get("scenario"),
            }
        )

    df = pd.DataFrame(rows)
    raw_sample_path = project_path(ds_cfg["raw_sample_path"])
    ensure_dir(raw_sample_path)
    df.to_json(raw_sample_path, orient="records", lines=True, force_ascii=False)

    adaption_input_path = project_path(ds_cfg["adaption_input_path"])
    ensure_dir(adaption_input_path)
    df[["example_id", "instruction", "response"]].to_csv(adaption_input_path, index=False)

    save_sft_jsonl(df, ds_cfg["raw_sft_path"])
    return df


def save_sft_jsonl(df: pd.DataFrame, path: str | Path) -> Path:
    rows = []
    for row in df.to_dict(orient="records"):
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": str(row["instruction"]).strip()},
                    {"role": "assistant", "content": str(row["response"]).strip()},
                ],
                "metadata": {"example_id": row.get("example_id")},
            }
        )
    return write_jsonl(rows, path)


def load_adapted_table(path: str | Path) -> pd.DataFrame:
    path = project_path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported adapted dataset format: {path}")


def convert_adapted_to_sft(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    df = load_adapted_table(input_path)

    prompt_candidates = ["instruction", "prompt", "input", "question"]
    response_candidates = ["response", "completion", "output", "answer"]
    prompt_col = next((c for c in prompt_candidates if c in df.columns), None)
    response_col = next((c for c in response_candidates if c in df.columns), None)

    if not prompt_col or not response_col:
        raise ValueError(
            "Could not find prompt/response columns in adapted data. "
            f"Available columns: {list(df.columns)}"
        )

    normalized = pd.DataFrame(
        {
            "example_id": df.get("example_id", pd.Series([f"adapted_{i:05d}" for i in range(len(df))])),
            "instruction": df[prompt_col].astype(str),
            "response": df[response_col].astype(str).map(strip_reasoning_traces),
        }
    )
    save_sft_jsonl(normalized, output_path)
    return normalized
