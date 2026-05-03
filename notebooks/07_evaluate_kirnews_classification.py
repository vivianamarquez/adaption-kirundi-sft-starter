import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Notebook 07 - Evaluate KIRNEWS Classification

    This notebook evaluates whether each model can classify Kirundi news articles from KIRNEWS into English category labels.

    This is still a proxy task. It is useful because KIRNEWS is labeled, but classification accuracy does not prove broad language quality.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT / "src"))

    from kirundi_sft_starter.data import load_kirnews_prompts
    from kirundi_sft_starter.evals import score_kirnews_predictions
    from kirundi_sft_starter.utils import ensure_dir, load_yaml, read_jsonl

    project_config = load_yaml(ROOT / "configs/project.yaml")
    eval_config = load_yaml(ROOT / "configs/evaluation.yaml")
    kirnews_labels = eval_config["kirnews_classification"]["labels"]
    return (
        ROOT,
        ensure_dir,
        eval_config,
        kirnews_labels,
        load_kirnews_prompts,
        pd,
        project_config,
        read_jsonl,
        score_kirnews_predictions,
    )


@app.cell
def _(kirnews_labels):
    print(kirnews_labels)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Prepare KIRNEWS prompts

    The prompt asks for one English label only. The default label set uses the labels from the dataset card, including `history`, `tourism`, and `fashion`.
    """)
    return


@app.cell
def _(load_kirnews_prompts, project_config):
    kirnews_prompt_df = load_kirnews_prompts(project_config)
    kirnews_prompt_df[["prompt_id", "gold_label", "title"]].head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Generate predictions

    Run these from the repo root. Add `--dry-run` if you only want placeholder files.

    ```bash
    python scripts/generate_model_responses.py --model-key base --prompts data/eval/kirnews_prompts.jsonl --output results/responses/kirnews_base.jsonl
    python scripts/generate_model_responses.py --model-key sft_raw --prompts data/eval/kirnews_prompts.jsonl --output results/responses/kirnews_sft_raw.jsonl
    python scripts/generate_model_responses.py --model-key sft_adapted --prompts data/eval/kirnews_prompts.jsonl --output results/responses/kirnews_sft_adapted.jsonl
    ```
    """)
    return


@app.cell
def _(ROOT, pd, read_jsonl):
    kirnews_prediction_files = {
        "base": ROOT / "results/responses/kirnews_base.jsonl",
        "sft_raw": ROOT / "results/responses/kirnews_sft_raw.jsonl",
        "sft_adapted": ROOT / "results/responses/kirnews_sft_adapted.jsonl",
    }

    kirnews_frames = []
    for _prediction_model_key, prediction_path in kirnews_prediction_files.items():
        if prediction_path.exists():
            prediction_df = pd.DataFrame(read_jsonl(prediction_path))
            prediction_df["model"] = _prediction_model_key
            prediction_df = prediction_df.rename(columns={"response": "prediction"})
            kirnews_frames.append(prediction_df)

    if not kirnews_frames:
        raise FileNotFoundError("No KIRNEWS prediction files found. Generate predictions first.")

    kirnews_predictions = pd.concat(kirnews_frames, ignore_index=True)
    kirnews_predictions[["model", "prompt_id", "gold_label", "prediction"]].head()
    return (kirnews_predictions,)


@app.cell
def _(
    ROOT,
    ensure_dir,
    eval_config,
    kirnews_labels,
    kirnews_predictions,
    pd,
    score_kirnews_predictions,
):
    kirnews_rows = []
    for _score_model_key, model_group in kirnews_predictions.groupby("model"):
        scores = score_kirnews_predictions(model_group, kirnews_labels)
        kirnews_rows.append(
            {
                "model": _score_model_key,
                "accuracy": round(scores["accuracy"], 4),
                "macro_f1": round(scores["macro_f1"], 4),
                "num_examples": len(model_group),
                "notes": "proxy automatic classification eval",
            }
        )

    kirnews_summary = pd.DataFrame(kirnews_rows)
    kirnews_output_path = ROOT / eval_config["kirnews_classification"]["output_path"]
    ensure_dir(kirnews_output_path)
    kirnews_summary.to_csv(kirnews_output_path, index=False)
    print("Saved", kirnews_output_path)
    print(kirnews_summary.to_string(index=False))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Suggested qualitative table

    After scoring, inspect examples where `sft_adapted` differs from `sft_raw`. Those rows are the most useful for understanding whether Adaption changed the downstream behavior or merely changed surface form.
    """)
    return


if __name__ == "__main__":
    app.run()
