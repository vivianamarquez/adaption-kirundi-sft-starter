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
    # Notebook 02 - Adapt Dataset With Adaption

    This notebook follows the Adaption SDK flow from the docs: install/import the SDK, load credentials, upload the prepared file, confirm column mapping, run an estimate, run a small pilot, poll for completion, download the adapted output, and inspect before/after rows.

    API cells are guarded by `RUN_ADAPTION_API = False` by default.
    """)
    return


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT / "src"))
    sys.path.append(str(ROOT))

    from kirundi_sft_starter.utils import load_env, load_yaml
    from scripts.adapt_with_adaption import blueprint_text

    load_env()
    adaption_run_config = load_yaml(ROOT / "configs/adaption_run.yaml")
    blueprint_config = load_yaml(ROOT / "configs/adaption_blueprint.yaml")

    RUN_ADAPTION_API = False
    has_adaption_key = bool(os.environ.get("ADAPTION_API_KEY"))
    print("Adaption key loaded:", has_adaption_key)
    return (
        ROOT,
        RUN_ADAPTION_API,
        adaption_run_config,
        blueprint_config,
        blueprint_text,
        has_adaption_key,
        os,
        pd,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Inspect the input and column mapping

    Adaption needs to know which columns contain the prompt and completion. This repo maps `instruction` to `prompt` and `response` to `completion`.
    """)
    return


@app.cell
def _(ROOT, adaption_run_config, pd):
    adaption_input_path = ROOT / adaption_run_config["input_path"]
    adaption_input_df = pd.read_csv(adaption_input_path)
    adaption_input_df.head()
    return (adaption_input_path,)


@app.cell
def _(adaption_run_config):
    adaption_run_config["column_mapping"]
    return


@app.cell
def _(blueprint_config, blueprint_text):
    brand_controls = dict(blueprint_config["adaption_brand_controls"])
    brand_controls["blueprint"] = blueprint_text(blueprint_config)

    print(brand_controls["blueprint"][:1200])
    return (brand_controls,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Run estimate and pilot

    Leave `RUN_ADAPTION_API = False` until you have reviewed the input, column mapping, and blueprint. The CLI version below also supports `--dry-run`.
    """)
    return


@app.cell
def _(
    RUN_ADAPTION_API,
    adaption_input_path,
    adaption_run_config,
    blueprint_config,
    brand_controls,
    has_adaption_key,
    os,
):
    if RUN_ADAPTION_API and has_adaption_key:
        from adaption import Adaption

        client = Adaption(api_key=os.environ["ADAPTION_API_KEY"])
        upload = client.datasets.upload_file(
            str(adaption_input_path),
            name=adaption_run_config["dataset_name"],
        )
        dataset_id = upload.dataset_id
        print("Dataset ID:", dataset_id)

        estimate = client.datasets.run(
            dataset_id,
            column_mapping=adaption_run_config["column_mapping"],
            brand_controls=brand_controls,
            recipe_specification=blueprint_config["adaption_recipe_specification"],
            job_specification={"max_rows": adaption_run_config["pilot"]["max_rows"]},
            estimate=True,
        )
        print("Estimated credits:", estimate.estimated_credits_consumed)
    else:
        print("Dry run. Review configs/adaption_run.yaml, then run the CLI command below when ready.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## CLI Equivalent

    ```bash
    python scripts/adapt_with_adaption.py --dry-run
    python scripts/adapt_with_adaption.py
    # Optional full run after the pilot looks good:
    python scripts/adapt_with_adaption.py --full
    ```

    After download, convert the adapted output into the same SFT format as the raw data:

    ```bash
    python scripts/convert_adapted_to_sft.py
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
