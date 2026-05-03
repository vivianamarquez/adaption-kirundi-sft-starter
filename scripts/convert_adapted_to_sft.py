from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from kirundi_sft_starter.data import convert_adapted_to_sft
from kirundi_sft_starter.utils import PROJECT_ROOT, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Adaption output to chat SFT JSONL.")
    parser.add_argument("--project-config", default="configs/project.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    project = load_yaml(PROJECT_ROOT / args.project_config)
    sft_cfg = project["datasets"]["sft"]
    input_path = args.input or sft_cfg["adapted_output_path"]
    output_path = args.output or sft_cfg["adapted_sft_path"]

    df = convert_adapted_to_sft(PROJECT_ROOT / input_path, PROJECT_ROOT / output_path)
    print(f"Converted {len(df)} adapted examples")
    print(f"Adapted SFT JSONL: {output_path}")


if __name__ == "__main__":
    main()
