from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from kirundi_sft_starter.evals import (
    classify_language,
    load_response_table,
    summarize_language_adherence,
)
from kirundi_sft_starter.utils import PROJECT_ROOT, ensure_dir, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate whether model outputs are classified as Kirundi/Rundi.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--use-afrolid", action="store_true")
    args = parser.parse_args()

    project = load_yaml(PROJECT_ROOT / args.config)
    frames = []
    for model_key, model_cfg in project["models"]["registry"].items():
        path = PROJECT_ROOT / model_cfg["response_path"]
        if path.exists():
            frames.append(load_response_table(path, model_key))

    if not frames:
        raise FileNotFoundError("No response files found. Run scripts/generate_model_responses.py first.")

    lid = None
    if args.use_afrolid:
        from transformers import pipeline

        lid = pipeline("text-classification", model=project["evaluation"]["language_id_model"])

    df = pd.concat(frames, ignore_index=True)
    detected = df["response"].apply(lambda text: classify_language(text, lid)).apply(pd.Series)
    scored = pd.concat([df, detected], axis=1)
    summary = summarize_language_adherence(scored)

    output_path = PROJECT_ROOT / project["evaluation"]["language_results_path"]
    ensure_dir(output_path)
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
