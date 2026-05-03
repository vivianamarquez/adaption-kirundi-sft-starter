from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from kirundi_sft_starter.data import prepare_kakugo_subset
from kirundi_sft_starter.utils import PROJECT_ROOT, load_env, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small Kakugo Kirundi SFT subset.")
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    load_env()
    config = load_yaml(PROJECT_ROOT / args.config)
    df = prepare_kakugo_subset(config)

    sft_cfg = config["datasets"]["sft"]
    print(f"Prepared {len(df)} examples")
    print(f"Adaption input: {sft_cfg['adaption_input_path']}")
    print(f"Raw SFT JSONL:  {sft_cfg['raw_sft_path']}")


if __name__ == "__main__":
    main()
