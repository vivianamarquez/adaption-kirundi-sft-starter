# Results Directory

Generated model checkpoints, metrics, response tables, plots, and evaluation summaries should live here.

This directory is ignored by git except for this README and `.gitkeep`, because training outputs and model artifacts can become large quickly.

Recommended layout:

```text
results/
├── models/
│   ├── sft_raw/
│   └── sft_adapted/
├── responses/
│   ├── base.jsonl
│   ├── sft_raw.jsonl
│   └── sft_adapted.jsonl
└── eval/
    ├── language_adherence.csv
    ├── kirnews_classification.csv
    └── comparison_summary.csv
```
