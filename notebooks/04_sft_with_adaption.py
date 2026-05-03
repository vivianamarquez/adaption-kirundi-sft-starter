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
    # Notebook 04 - SFT With Adaption-Improved Data

    This is the adapted-data SFT condition. It trains the same base model with the same training settings as Notebook 03, but uses `data/processed/kakugo_adapted_sft.jsonl`.

    Keeping these notebooks parallel makes the comparison easier to reason about.
    """)
    return


@app.cell
def _():
    import subprocess
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT / "src"))

    from kirundi_sft_starter.utils import load_yaml

    adapted_sft_config = load_yaml(ROOT / "configs/tinker_sft_adapted.yaml")
    return ROOT, adapted_sft_config, subprocess


@app.cell
def _(adapted_sft_config):
    print(adapted_sft_config)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Dry run first

    This checks the config and data path without launching a paid or remote training job.
    """)
    return


@app.cell
def _(ROOT, subprocess):
    dry_run = subprocess.run(
        [
            "python",
            str(ROOT / "scripts/train_tinker_sft.py"),
            "--config",
            "configs/tinker_sft_adapted.yaml",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    print(dry_run.stdout)
    if dry_run.stderr:
        print(dry_run.stderr)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Launch training

    Run this only when the adapted SFT JSONL exists and your Tinker credentials are set.

    ```bash
    python scripts/train_tinker_sft.py --config configs/tinker_sft_adapted.yaml
    ```

    Expected output directory: `results/models/sft_adapted/`.
    """)
    return


if __name__ == "__main__":
    app.run()
