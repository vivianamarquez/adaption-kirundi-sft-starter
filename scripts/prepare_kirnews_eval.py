from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from kirundi_sft_starter.data import load_kirnews_prompts
from kirundi_sft_starter.utils import PROJECT_ROOT, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare KIRNEWS classification prompts.")
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    config = load_yaml(PROJECT_ROOT / args.config)
    df = load_kirnews_prompts(config)
    print(f"Prepared {len(df)} KIRNEWS prompts")
    print(f"Output: {config['datasets']['kirnews']['prompt_path']}")


if __name__ == "__main__":
    main()
