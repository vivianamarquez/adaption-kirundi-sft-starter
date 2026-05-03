from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from kirundi_sft_starter.tinker_utils import training_plan
from kirundi_sft_starter.utils import PROJECT_ROOT, ensure_dir, load_env, load_yaml, require_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Tinker SFT run from a config file.")
    parser.add_argument("--config", required=True, help="Example: configs/tinker_sft_raw.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env()
    config = load_yaml(PROJECT_ROOT / args.config)
    require_file(PROJECT_ROOT / config["data_path"], "Prepare the SFT JSONL before training.")
    ensure_dir(PROJECT_ROOT / config["output_dir"])

    plan = training_plan(config)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        print("\nDry run only. Rerun without --dry-run to launch training.")
        return

    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("Missing TINKER_TOKEN or TINKER_API_KEY. Add it to .env before launching training.")

    import nest_asyncio
    from huggingface_hub import login
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.train import Config
    from tinker_cookbook.supervised.train import main as train_main
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    nest_asyncio.apply()
    if os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=config["base_model"],
        renderer_name=config["renderer_name"],
        max_length=int(config["max_length"]),
        batch_size=int(config["batch_size"]),
        train_on_what=config["train_on_what"],
    )
    dataset_builder = FromConversationFileBuilder(
        file_path=str(PROJECT_ROOT / config["data_path"]),
        common_config=common_config,
    )
    sft_config = Config(
        log_path=str(PROJECT_ROOT / config["output_dir"]),
        model_name=config["base_model"],
        dataset_builder=dataset_builder,
        learning_rate=float(config["learning_rate"]),
        lora_rank=int(config["lora_rank"]),
        num_epochs=int(config["num_epochs"]),
    )

    print("\nSFT Config:")
    print(f"  Run:           {config['run_name']}")
    print(f"  Model:         {sft_config.model_name}")
    print(f"  Renderer:      {config['renderer_name']}")
    print(f"  Data:          {PROJECT_ROOT / config['data_path']}")
    print(f"  Output:        {sft_config.log_path}")
    print(f"  Learning rate: {sft_config.learning_rate}")
    print(f"  LoRA rank:     {sft_config.lora_rank}")
    print(f"  Epochs:        {sft_config.num_epochs}")
    print("\nStarting SFT training. Watch train_mean_nll in the logs.")
    result = asyncio.get_event_loop().run_until_complete(train_main(sft_config))
    print(result)


if __name__ == "__main__":
    main()
