from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def final_sampler_path(checkpoints_path: str | Path) -> str:
    checkpoints = [json.loads(line) for line in Path(checkpoints_path).read_text().splitlines() if line]
    for checkpoint in checkpoints:
        if checkpoint.get("name") == "final":
            return checkpoint["sampler_path"]
    if checkpoints:
        return checkpoints[-1]["sampler_path"]
    raise ValueError(f"No checkpoints found in {checkpoints_path}")


def training_plan(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": config["run_name"],
        "base_model": config["base_model"],
        "data_path": config["data_path"],
        "output_dir": config["output_dir"],
        "learning_rate": config["learning_rate"],
        "num_epochs": config["num_epochs"],
        "batch_size": config["batch_size"],
        "lora_rank": config["lora_rank"],
        "renderer_name": config["renderer_name"],
    }
