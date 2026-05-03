# Data Directory

This repo intentionally does not commit generated datasets.

The marimo notebooks and scripts create the following local files:

| Directory | Purpose | Tracked? |
|---|---|---|
| `data/raw/` | Small local samples pulled from Hugging Face | no |
| `data/processed/` | Clean JSONL/CSV files ready for SFT or Adaption | no |
| `data/adapted/` | Files downloaded from Adaption | no |
| `data/eval/` | Evaluation prompts, responses, and intermediate predictions | no |

Each directory contains a `.gitkeep` so the folder structure is visible without committing data.
