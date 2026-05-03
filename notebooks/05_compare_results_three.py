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
    # Notebook 05 - Compare Base / Raw SFT / Adapted SFT

    This notebook creates a qualitative comparison table for the three model conditions.

    The important experimental discipline: use the same prompts and sampling settings for all three models.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT / "src"))

    from kirundi_sft_starter.utils import load_yaml, read_jsonl

    project_config = load_yaml(ROOT / "configs/project.yaml")
    return ROOT, pd, project_config, read_jsonl


@app.cell
def _(project_config):
    print(project_config["models"]["registry"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Generate or load responses

    Use the CLI to generate comparable outputs. Add `--dry-run` if you want placeholder files while reading the notebook.

    ```bash
    python scripts/generate_model_responses.py --model-key base
    python scripts/generate_model_responses.py --model-key sft_raw
    python scripts/generate_model_responses.py --model-key sft_adapted
    ```
    """)
    return


@app.cell
def _(ROOT, pd, project_config, read_jsonl):
    response_frames = []
    for model_key, model_config in project_config["models"]["registry"].items():
        response_path = ROOT / model_config["response_path"]
        if response_path.exists():
            model_responses = pd.DataFrame(read_jsonl(response_path))
            model_responses["model"] = model_key
            response_frames.append(model_responses)

    if response_frames:
        comparison_responses = pd.concat(response_frames, ignore_index=True)
        comparison_responses.head()
    else:
        comparison_responses = pd.DataFrame()
        print("No response files yet. Run the generation commands above.")
    return comparison_responses, response_frames


@app.cell
def _(comparison_responses, response_frames):
    if response_frames:
        qualitative_wide = comparison_responses.pivot_table(
            index=["prompt_id", "prompt"],
            columns="model",
            values="response",
            aggfunc="first",
        ).reset_index()
        print(qualitative_wide.to_string(index=False))
    return


@app.cell
def _(comparison_responses, response_frames):
    if response_frames:
        length_summary = (
            comparison_responses.assign(response_chars=comparison_responses["response"].str.len())
            .groupby("model")["response_chars"]
            .mean()
            .reset_index()
        )
        print(length_summary.to_string(index=False))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Qualitative review questions

    - Did the answer stay in Kirundi?
    - Did it answer the question directly?
    - Did it copy unwanted reasoning traces?
    - Did Adaption appear to improve clarity or formatting?
    - Are there examples that need native speaker review before any conclusion?
    """)
    return


if __name__ == "__main__":
    app.run()
