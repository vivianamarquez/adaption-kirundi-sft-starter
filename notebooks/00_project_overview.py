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
    # Notebook 00 - Project Overview

    This repo is a starter workflow for testing whether improving Kirundi/Rundi SFT data with Adaption changes post-training outcomes.

    The goal is not to prove that a model understands Kirundi perfectly. The goal is to make a repeatable experiment that compares three conditions:

    1. Base model
    2. Same base model SFT'd on raw `ptrdvn/kakugo-run` examples
    3. Same base model SFT'd on Adaption-improved versions of those examples
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Workflow

    ```text
    ptrdvn/kakugo-run
            |
            v
    small clean experiment subset
            |
            +--------------------------+
            |                          |
            v                          v
    raw SFT JSONL              Adaption input CSV
            |                          |
            |                          v
            |                  adapted dataset
            |                          |
            v                          v
    Tinker SFT run A           Tinker SFT run B
            |                          |
            +------------+-------------+
                         v
          compare base vs raw SFT vs adapted SFT
                         |
                         v
     language adherence + KIRNEWS classification
    ```
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import yaml

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT / "src"))

    with open(ROOT / "configs/project.yaml", encoding="utf-8") as f:
        project_config = yaml.safe_load(f)
    return (project_config,)


@app.cell
def _(project_config):
    project_config["project"]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Notebook Map

    | Notebook | Purpose | Requires credentials? |
    |---|---|---|
    | `01_prepare_kirundi_sft_dataset.py` | Sample and clean Kakugo Rundi SFT data | No |
    | `02_adapt_dataset_with_adaption.py` | Run estimate, pilot, download adapted data | Adaption key for API cells |
    | `03_sft_without_adaption.py` | Train SFT on raw data | Tinker key |
    | `04_sft_with_adaption.py` | Train SFT on adapted data | Tinker key |
    | `05_compare_results_three.py` | Compare base/raw/adapted outputs | Tinker key if generating fresh outputs |
    | `06_evaluate_language_adherence.py` | Check whether outputs are classified as Kirundi/Rundi | Optional HF download for AfroLID |
    | `07_evaluate_kirnews_classification.py` | Score KIRNEWS classification accuracy/F1 | Tinker key if generating fresh predictions |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Responsible Framing

    Automatic language ID and KIRNEWS classification are proxy evaluations. They are useful for iteration, but they do not replace native speaker review, cultural review, or task-specific human evaluation.
    """)
    return


if __name__ == "__main__":
    app.run()
