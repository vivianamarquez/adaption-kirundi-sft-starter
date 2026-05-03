from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .utils import ensure_dir, load_env, project_path, require_file, write_jsonl


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


def generate_model_responses(
    project: dict[str, Any],
    model_key: str,
    prompts: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> Path:
    load_env()
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("Missing TINKER_TOKEN or TINKER_API_KEY. Add it to .env before sampling.")

    import tinker
    from tinker import types
    from tinker_cookbook import renderers
    from transformers import AutoTokenizer

    model_config = project["models"]["registry"][model_key]
    output_path = project_path(output_path or model_config["response_path"])

    base_model = project["models"]["base_model"]
    renderer_name = project["models"]["renderer_name"]
    service_client = tinker.ServiceClient()

    if model_key == "base":
        client = service_client.create_sampling_client(base_model=model_config["model_name"])
    else:
        checkpoint_path = require_file(
            model_config["checkpoint_path"],
            f"Run the SFT notebook for {model_key} before sampling.",
        )
        sampler_path = final_sampler_path(checkpoint_path)
        client = service_client.create_sampling_client(model_path=sampler_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=os.environ.get("HF_TOKEN"))
    renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
    sampling_params = types.SamplingParams(
        max_tokens=int(project["models"]["max_new_tokens"]),
        temperature=float(project["models"]["temperature"]),
        stop=renderer.get_stop_sequences(),
    )

    rows = []
    for row in prompts:
        messages = [renderers.Message(role="user", content=row["prompt"])]
        model_input = renderer.build_generation_prompt(messages)
        result = client.sample(prompt=model_input, sampling_params=sampling_params, num_samples=1).result()
        content = renderer.parse_response(result.sequences[0].tokens)[0]["content"]
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") for block in content if block.get("type") == "text"
            ).strip()
        rows.append({**row, "model": model_key, "response": content})

    ensure_dir(output_path)
    return write_jsonl(rows, output_path)
