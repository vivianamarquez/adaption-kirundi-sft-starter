# Adaption Kirundi SFT Starter

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status: Beta](https://img.shields.io/badge/status-beta-orange.svg)]()
[![Purpose: Educational](https://img.shields.io/badge/purpose-educational-green.svg)]()

An open-source starter repo for improving low-resource Kirundi SFT data with [Adaption](https://adaptionlabs.ai/) and evaluating post-training outcomes.

> If we improve low-resource SFT data before training, do we see measurable changes in model behavior?

## Why This Project Exists

In 2023, I taught AI in Burundi and saw the AI gap up close. At the time, many of the most widely used LLM tools were not accessible from the country. But the gap was bigger than platform access alone. It was also a language gap, a data gap, and a design gap.

Many AI workflows assume abundant English data, large evaluation sets, and easy access to native-language reviewers. That is not the reality for many low-resource languages. For communities whose languages are underrepresented in training data, AI often does not meet people where they are.

This repo explores one practical slice of that problem: can adaptive data improvement make a Kirundi SFT workflow easier to build, test, compare, and improve?

The broader question is even more important: what would it look like for AI systems to adapt to the world, instead of expecting the world to adapt to them?

The framing is intentionally cautious. Automatic metrics are only proxies. Native speaker review matters. This repo does not claim to solve low-resource language AI. Instead, it offers a clear starter workflow that developers, researchers, and product teams can extend responsibly.

## What This Repo Builds

This repo walks through an end-to-end SFT experiment:

1. Load a small subset of `ptrdvn/kakugo-run` from Hugging Face.
2. Clean and normalize the data into an instruction/response format.
3. Send the raw SFT data through Adaption.
4. Download the Adaption-improved dataset.
5. Convert raw and adapted data into the same chat SFT JSONL format.
6. Fine-tune the same base model twice with Tinker:
   - SFT without Adaption
   - SFT with Adaption-improved data
7. Compare three model conditions:
   - base model
   - raw-data SFT model
   - adapted-data SFT model
8. Evaluate with two proxy metrics:
   - language adherence: did the model answer in Kirundi/Rundi?
   - KIRNEWS classification: did the model classify labeled Kirundi news articles correctly?

## Workflow Diagram

```text
Hugging Face: ptrdvn/kakugo-run
        |
        v
Small clean experiment subset
        |
        +-----------------------------+
        |                             |
        v                             v
Raw chat SFT JSONL              Adaption input CSV
        |                             |
        |                             v
        |                    Adaption estimate
        |                             |
        |                             v
        |                    Adaption pilot run
        |                             |
        |                             v
        |                    Adapted output CSV
        |                             |
        v                             v
Tinker SFT: raw data          Tinker SFT: adapted data
        |                             |
        +-------------+---------------+
                      |
                      v
       Compare base vs raw SFT vs adapted SFT
                      |
                      v
      Language adherence + KIRNEWS classification
```

## Experiment Design

| Condition | Training data | Purpose |
|---|---|---|
| Base model | none | Baseline behavior before post-training |
| SFT without Adaption | cleaned raw `ptrdvn/kakugo-run` subset | Measures effect of ordinary SFT data |
| SFT with Adaption | same subset after Adaption improvement | Measures effect of data improvement before SFT |

The two SFT notebooks use matching training settings. The intended experimental difference is the dataset, not the model, learning rate, renderer, or sampling setup.

## Datasets

| Dataset | Repo ID | Used for |
|---|---|---|
| Kakugo Rundi SFT data | [`ptrdvn/kakugo-run`](https://huggingface.co/datasets/ptrdvn/kakugo-run) | Raw SFT data for both training conditions |
| KIRNEWS/KINNEWS | [`andreniyongabo/kinnews_kirnews`](https://huggingface.co/datasets/andreniyongabo/kinnews_kirnews) | Labeled Kirundi news classification evaluation |

The KIRNEWS evaluation asks the model to classify an article into English labels such as `politics`, `sport`, `economy`, `health`, `technology`, `education`, and related categories.

## Evaluation Methodology

### 1. Language adherence

Question:

> Did the model actually answer in Kirundi/Rundi?

The preferred automatic evaluator is an African-language-aware language ID model such as [`UBC-NLP/afrolid_1.5`](https://huggingface.co/UBC-NLP/afrolid_1.5). If that is too heavy or unavailable, the notebook includes a transparent fallback heuristic and marks it as a fallback.

Report table:

| model | num_prompts | % Kirundi/Rundi responses | notes |
|---|---:|---:|---|
| base | generated by notebook | generated by notebook | automatic LID or fallback |
| sft_raw | generated by notebook | generated by notebook | automatic LID or fallback |
| sft_adapted | generated by notebook | generated by notebook | automatic LID or fallback |

### 2. KIRNEWS labeled classification

Prompt shape:

```text
Classify this Kirundi article into one of the following English labels:
politics, sport, economy, health, entertainment, history, technology,
tourism, culture, fashion, religion, environment, education, relationship.
Return only the English label.
```

Report table:

| model | accuracy | macro_f1 | num_examples | notes |
|---|---:|---:|---:|---|
| base | generated by notebook | generated by notebook | generated by notebook | proxy classification eval |
| sft_raw | generated by notebook | generated by notebook | generated by notebook | proxy classification eval |
| sft_adapted | generated by notebook | generated by notebook | generated by notebook | proxy classification eval |

### Optional qualitative review

The comparison notebook also builds a qualitative table:

| prompt | base_output | sft_raw_output | sft_adapted_output | language_id | notes |
|---|---|---|---|---|---|

This is where native speaker review should eventually enter the loop.

## Repo Structure

```text
adaption-kirundi-sft-starter/
├── README.md
├── LICENSE
├── .env.example
├── .gitignore
├── environment.yml
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── project.yaml
│   ├── adaption_blueprint.yaml
│   ├── adaption_run.yaml
│   ├── tinker_sft_raw.yaml
│   ├── tinker_sft_adapted.yaml
│   ├── evaluation.yaml
│   └── language_adherence_prompts.jsonl
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   ├── adapted/
│   └── eval/
├── docs/
│   ├── product_testing_notes.md
│   └── onboarding_feedback.md
├── notebooks/
│   ├── 00_project_overview.ipynb
│   ├── 01_prepare_kirundi_sft_dataset.ipynb
│   ├── 02_adapt_dataset_with_adaption.ipynb
│   ├── 03_sft_without_adaption.ipynb
│   ├── 04_sft_with_adaption.ipynb
│   ├── 05_compare_results_three.ipynb
│   ├── 06_evaluate_language_adherence.ipynb
│   └── 07_evaluate_kirnews_classification.ipynb
├── scripts/
└── src/kirundi_sft_starter/
```

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate adaption-kirundi-sft
```

If you already created the environment before this scaffold was added, update it:

```bash
conda env update -f environment.yml --prune
```

If you prefer pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name adaption-kirundi-sft --display-name "Python (adaption-kirundi-sft)"
```

### 3. Configure environment variables

Copy the example file:

```bash
cp .env.example .env
```

Fill in your own credentials:

```bash
HF_TOKEN=your_huggingface_token
TINKER_TOKEN=your_tinker_api_token
ADAPTION_API_KEY=your_adaption_api_key
```

The notebooks and scripts also support these local aliases:

```bash
TINKER_API_KEY=your_tinker_api_token
ADAPTION_TOKEN=your_adaption_api_key
```

Never commit `.env`.

## How To Run The Notebooks

Run notebooks in order:

| # | Notebook | What it does | External key needed |
|---|---|---|---|
| 00 | `notebooks/00_project_overview.ipynb` | Explains the project and workflow | no |
| 01 | `notebooks/01_prepare_kirundi_sft_dataset.ipynb` | Builds raw SFT and Adaption input files | no |
| 02 | `notebooks/02_adapt_dataset_with_adaption.ipynb` | Runs Adaption estimate/pilot/download flow | Adaption for API cells |
| 03 | `notebooks/03_sft_without_adaption.ipynb` | Tinker SFT on raw data | Tinker |
| 04 | `notebooks/04_sft_with_adaption.ipynb` | Tinker SFT on adapted data | Tinker |
| 05 | `notebooks/05_compare_results_three.ipynb` | Qualitative three-model comparison | Tinker if generating outputs |
| 06 | `notebooks/06_evaluate_language_adherence.ipynb` | Language ID summary | optional HF model download |
| 07 | `notebooks/07_evaluate_kirnews_classification.ipynb` | KIRNEWS accuracy/F1 | Tinker if generating predictions |

Launch Jupyter:

```bash
jupyter lab
```

## How To Run From The CLI

Prepare the raw SFT data:

```bash
python scripts/prepare_sft_dataset.py
```

Review the Adaption run without calling the API:

```bash
python scripts/adapt_with_adaption.py --dry-run
```

Run the Adaption pilot:

```bash
python scripts/adapt_with_adaption.py
```

Run the optional full Adaption job:

```bash
python scripts/adapt_with_adaption.py --full
```

Convert the adapted dataset to SFT JSONL:

```bash
python scripts/convert_adapted_to_sft.py
```

Train the two SFT models:

```bash
python scripts/train_tinker_sft.py --config configs/tinker_sft_raw.yaml
python scripts/train_tinker_sft.py --config configs/tinker_sft_adapted.yaml
```

Generate language-adherence responses:

```bash
python scripts/generate_model_responses.py --model-key base
python scripts/generate_model_responses.py --model-key sft_raw
python scripts/generate_model_responses.py --model-key sft_adapted
```

Evaluate language adherence:

```bash
python scripts/evaluate_language_adherence.py --use-afrolid
```

Prepare KIRNEWS prompts:

```bash
python scripts/prepare_kirnews_eval.py
```

Generate KIRNEWS predictions:

```bash
python scripts/generate_model_responses.py --model-key base --prompts data/eval/kirnews_prompts.jsonl --output results/responses/kirnews_base.jsonl
python scripts/generate_model_responses.py --model-key sft_raw --prompts data/eval/kirnews_prompts.jsonl --output results/responses/kirnews_sft_raw.jsonl
python scripts/generate_model_responses.py --model-key sft_adapted --prompts data/eval/kirnews_prompts.jsonl --output results/responses/kirnews_sft_adapted.jsonl
```

Score KIRNEWS predictions:

```bash
python scripts/evaluate_kirnews_classification.py --predictions results/responses/kirnews_base.jsonl
```

For a combined three-model KIRNEWS score, concatenate prediction JSONL files first or score them in Notebook 07.

## Expected Outputs

```text
data/
├── raw/kakugo_run_sample.jsonl
├── processed/kakugo_adaption_input.csv
├── processed/kakugo_raw_sft.jsonl
├── adapted/kakugo_adapted.csv
├── processed/kakugo_adapted_sft.jsonl
└── eval/kirnews_prompts.jsonl

results/
├── models/sft_raw/
├── models/sft_adapted/
├── responses/
│   ├── base.jsonl
│   ├── sft_raw.jsonl
│   ├── sft_adapted.jsonl
│   ├── kirnews_base.jsonl
│   ├── kirnews_sft_raw.jsonl
│   └── kirnews_sft_adapted.jsonl
└── eval/
    ├── language_adherence.csv
    ├── kirnews_classification.csv
    └── comparison_summary.csv
```

Generated data, responses, and model outputs are ignored by git.

## TODOs Before A Real Run

- Add real `ADAPTION_API_KEY`, `TINKER_TOKEN`, and `HF_TOKEN` values to local `.env`.
- Run Notebook 01 and manually inspect the sampled SFT rows before sending anything to Adaption.
- Run the Adaption dry run, then a small pilot, before any full adaptation job.
- Ask a Kirundi/Rundi speaker to review a sample of raw and adapted rows.
- Confirm that the configured base model is supported by your Tinker account.
- Start with the small default sample size, then increase it only after the pipeline works end to end.
- Treat automatic language ID and KIRNEWS metrics as proxy signals, not final quality judgments.

## Adaption Blueprint

The Adaption data-improvement instructions live in [`configs/adaption_blueprint.yaml`](configs/adaption_blueprint.yaml). The core goal is:

> Improve this dataset for supervised fine-tuning a small assistant that can answer beginner-friendly questions in Rundi/Kirundi.

The constraints emphasize preserving meaning, keeping responses in Rundi/Kirundi, avoiding unsupported local facts, removing malformed formatting and reasoning traces, and keeping explanations simple.

API usage follows the Adaption documentation:

- [Getting started](https://docs.adaptionlabs.ai/introduction/getting-started/)
- [API reference](https://docs.adaptionlabs.ai/api)
- [Processing large datasets](https://docs.adaptionlabs.ai/guides/processing-large-datasets/)
- [Reasoning traces guide](https://docs.adaptionlabs.ai/guides/reasoning-traces/)

## Product Testing Notes

This repo includes two docs written from the perspective of building with Adaption:

- [`docs/product_testing_notes.md`](docs/product_testing_notes.md)
- [`docs/onboarding_feedback.md`](docs/onboarding_feedback.md)

Themes captured there:

- what was clear in the docs
- where a new developer may get stuck
- what examples or presets would help
- why "run a pilot first" should be part of the default workflow
- what SFT export formats would make post-training easier

## Limitations And Responsible Framing

- This repo is a starter workflow, not a definitive Kirundi benchmark.
- Automatic language identification can be wrong, especially for short outputs.
- KIRNEWS classification is a useful labeled proxy task, not a complete evaluation of language quality.
- Native speaker review is necessary before drawing strong conclusions.
- The default sample size is intentionally small for cost and iteration speed.
- Tinker and Adaption API calls require credentials and may incur usage costs.
- The base model choice is configurable. Use a model that your training provider supports and document any change.

## Future Work

- Add native speaker review of prompts, outputs, and adapted data.
- Run larger training jobs with multiple random seeds.
- Add better low-resource language evaluation sets.
- Compare multiple base models.
- Add before/after Adaption diff visualizations.
- Add a human review rubric for fluency, correctness, cultural grounding, and task success.
- Invite Kirundi speakers and Burundian AI practitioners to contribute examples and evaluation criteria.
- Extend the workflow to other low-resource languages with similarly cautious framing.

## Important Notice

This project is provided as-is for educational and research purposes.

- It is not production-ready.
- It may contain incomplete API examples while services evolve.
- It does not commit API keys, datasets, or model artifacts.
- It should not be used to make claims about real-world Kirundi language performance without human review.

Contributions and careful critique are welcome.
