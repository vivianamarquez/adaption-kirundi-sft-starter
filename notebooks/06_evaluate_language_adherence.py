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
    # Notebook 06 - Evaluate Language Adherence

    Question: did the model actually answer in Kirundi/Rundi?

    The preferred automatic evaluator is AfroLID (`UBC-NLP/afrolid_1.5`), an African-language-aware language identification model. If that model is too heavy for your machine, this notebook can use a transparent heuristic fallback. The fallback is not a real language ID model.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT / "src"))

    from kirundi_sft_starter.evals import (
        classify_language,
        load_response_table,
        summarize_language_adherence,
    )
    from kirundi_sft_starter.utils import ensure_dir, load_yaml

    project_config = load_yaml(ROOT / "configs/project.yaml")
    USE_AFROLID = False
    return (
        ROOT,
        USE_AFROLID,
        classify_language,
        ensure_dir,
        load_response_table,
        pd,
        project_config,
        summarize_language_adherence,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Generate responses for the language prompts

    ```bash
    python scripts/generate_model_responses.py --model-key base
    python scripts/generate_model_responses.py --model-key sft_raw
    python scripts/generate_model_responses.py --model-key sft_adapted
    ```

    Use `--dry-run` to create placeholder files without Tinker credentials.
    """)
    return


@app.cell
def _(ROOT, load_response_table, pd, project_config):
    language_frames = []
    for model_key, model_config in project_config["models"]["registry"].items():
        response_path = ROOT / model_config["response_path"]
        if response_path.exists():
            language_frames.append(load_response_table(response_path, model_key))

    if not language_frames:
        raise FileNotFoundError("No response files found. Generate responses first.")

    language_responses = pd.concat(language_frames, ignore_index=True)
    language_responses.head()
    return (language_responses,)


@app.cell
def _(
    ROOT,
    USE_AFROLID,
    classify_language,
    ensure_dir,
    language_responses,
    pd,
    project_config,
    summarize_language_adherence,
):
    language_id_pipeline = None
    if USE_AFROLID:
        from transformers import pipeline

        language_id_pipeline = pipeline(
            "text-classification",
            model=project_config["evaluation"]["language_id_model"],
        )

    detected_languages = (
        language_responses["response"]
        .apply(lambda text: classify_language(text, language_id_pipeline))
        .apply(pd.Series)
    )
    language_scored = pd.concat([language_responses, detected_languages], axis=1)
    language_summary = summarize_language_adherence(language_scored)

    language_output_path = ROOT / project_config["evaluation"]["language_results_path"]
    ensure_dir(language_output_path)
    language_summary.to_csv(language_output_path, index=False)
    print("Saved", language_output_path)
    print(language_summary.to_string(index=False))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Report table

    | model | num_prompts | % Kirundi/Rundi responses | notes |
    |---|---:|---:|---|
    | base | filled by notebook | filled by notebook | automatic LID or fallback |
    | sft_raw | filled by notebook | filled by notebook | automatic LID or fallback |
    | sft_adapted | filled by notebook | filled by notebook | automatic LID or fallback |
    """)
    return


if __name__ == "__main__":
    app.run()
