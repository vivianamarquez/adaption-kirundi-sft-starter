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
    # Notebook 03 - SFT Without Adaption

    This is the raw-data SFT condition. It trains the base model on `data/processed/kakugo_raw_sft.jsonl`.

    Notebook 04 mirrors this notebook. The only intended difference is the training data path.
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

    raw_sft_config = load_yaml(ROOT / "configs/tinker_sft_raw.yaml")
    return ROOT, raw_sft_config, subprocess


@app.cell
def _(raw_sft_config):
    print(raw_sft_config)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Dry run first

    This prints the training plan and verifies that the data file exists. It does not launch Tinker training.
    """)
    return


@app.cell
def _(ROOT, subprocess):
    dry_run = subprocess.run(
        [
            "python",
            str(ROOT / "scripts/train_tinker_sft.py"),
            "--config",
            "configs/tinker_sft_raw.yaml",
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

    Run this only when `TINKER_TOKEN` or `TINKER_API_KEY` is set in `.env`.

    ```bash
    python scripts/train_tinker_sft.py --config configs/tinker_sft_raw.yaml
    ```

    Expected output directory: `results/models/sft_raw/`.
    """)
    return


if __name__ == "__main__":
    app.run()
