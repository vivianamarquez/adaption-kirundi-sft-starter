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
    # Notebook 01 - Prepare Kirundi SFT Dataset

    This notebook samples `ptrdvn/kakugo-run`, extracts user/assistant pairs, removes visible `<think>...</think>` reasoning traces from assistant responses, and writes two files:

    - `data/processed/kakugo_adaption_input.csv` for Adaption
    - `data/processed/kakugo_raw_sft.jsonl` for the raw SFT run
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT / "src"))

    from kirundi_sft_starter.data import prepare_kakugo_subset
    from kirundi_sft_starter.utils import load_yaml

    project_config = load_yaml(ROOT / "configs/project.yaml")
    sft_config = project_config["datasets"]["sft"]
    return ROOT, prepare_kakugo_subset, project_config, sft_config


@app.cell
def _(sft_config):
    print(sft_config)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Load and normalize a small subset

    Keep this small at first. The point of the starter repo is to validate the workflow before spending money or time on larger adaptation/training runs.
    """)
    return


@app.cell
def _(prepare_kakugo_subset, project_config):
    sft_df = prepare_kakugo_subset(project_config)
    sft_df.head()
    return (sft_df,)


@app.cell
def _(ROOT, sft_config, sft_df):
    print(f"Prepared examples: {len(sft_df)}")
    print("Adaption CSV:", ROOT / sft_config["adaption_input_path"])
    print("Raw SFT JSONL:", ROOT / sft_config["raw_sft_path"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## CLI Equivalent

    ```bash
    python scripts/prepare_sft_dataset.py
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
