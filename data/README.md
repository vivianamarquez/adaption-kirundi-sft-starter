# Data Directory

This repo intentionally does not commit generated datasets.

The Jupyter notebooks create the following local files:

| Directory | Purpose | Tracked? |
|---|---|---|
| `data/raw/` | Local raw rows pulled from Hugging Face | no |
| `data/processed/` | Clean JSONL/CSV files ready for SFT or Adaption | no |
| `data/adapted/` | Files downloaded from Adaption | no |

Each directory contains a `.gitkeep` so the folder structure is visible without committing data.
