from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from kirundi_sft_starter.tinker_utils import final_sampler_path
from kirundi_sft_starter.utils import (
    PROJECT_ROOT,
    ensure_dir,
    load_env,
    load_yaml,
    read_jsonl,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate responses for one configured model.")
    parser.add_argument("--model-key", required=True, choices=["base", "sft_raw", "sft_adapted"])
    parser.add_argument("--prompts", default="configs/language_adherence_prompts.jsonl")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env()
    project = load_yaml(PROJECT_ROOT / args.config)
    model_cfg = project["models"]["registry"][args.model_key]
    output_path = PROJECT_ROOT / (args.output or model_cfg["response_path"])
    prompts = read_jsonl(PROJECT_ROOT / args.prompts)

    if args.dry_run or not os.environ.get("TINKER_API_KEY"):
        rows = [
            {
                **row,
                "model": args.model_key,
                "response": "[dry run] Tinker credentials required to generate this response.",
            }
            for row in prompts
        ]
        write_jsonl(rows, output_path)
        print(f"Wrote dry-run responses to {output_path}")
        return

    import tinker
    from dotenv import load_dotenv
    from tinker import types
    from tinker_cookbook import renderers
    from transformers import AutoTokenizer

    load_dotenv()
    base_model = project["models"]["base_model"]
    renderer_name = project["models"]["renderer_name"]
    service_client = tinker.ServiceClient()

    if args.model_key == "base":
        client = service_client.create_sampling_client(base_model=model_cfg["model_name"])
    else:
        sampler_path = final_sampler_path(PROJECT_ROOT / model_cfg["checkpoint_path"])
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
            content = " ".join(block.get("text", "") for block in content if block.get("type") == "text").strip()
        rows.append({**row, "model": args.model_key, "response": content})

    ensure_dir(Path(output_path).parent)
    write_jsonl(rows, output_path)
    print(f"Wrote {len(rows)} responses to {output_path}")


if __name__ == "__main__":
    main()
