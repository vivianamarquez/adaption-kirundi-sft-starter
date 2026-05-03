from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from kirundi_sft_starter.evals import normalize_label, score_kirnews_predictions
from kirundi_sft_starter.utils import PROJECT_ROOT, ensure_dir, load_yaml, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Score KIRNEWS classification predictions.")
    parser.add_argument("--predictions", required=True, help="JSONL with model,prompt_id,gold_label,prediction.")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    args = parser.parse_args()

    cfg = load_yaml(PROJECT_ROOT / args.config)
    labels = cfg["kirnews_classification"]["labels"]
    df = pd.DataFrame(read_jsonl(PROJECT_ROOT / args.predictions))
    if "prediction" not in df.columns and "response" in df.columns:
        df = df.rename(columns={"response": "prediction"})

    rows = []
    for model, group in df.groupby("model"):
        scores = score_kirnews_predictions(group, labels)
        rows.append(
            {
                "model": model,
                "accuracy": round(scores["accuracy"], 4),
                "macro_f1": round(scores["macro_f1"], 4),
                "num_examples": len(group),
                "notes": "proxy automatic classification eval",
            }
        )

    summary = pd.DataFrame(rows)
    output_path = PROJECT_ROOT / cfg["kirnews_classification"]["output_path"]
    ensure_dir(output_path)
    summary.to_csv(output_path, index=False)

    df["gold_label_normalized"] = df["gold_label"].map(lambda x: normalize_label(x, labels))
    df["prediction_normalized"] = df["prediction"].map(lambda x: normalize_label(x, labels))
    print(summary.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
