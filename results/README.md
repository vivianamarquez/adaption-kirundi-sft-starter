# Results Directory

Generated model checkpoints, metrics, response tables, plots, and evaluation summaries should live here.

This directory is ignored by git except for this README and `.gitkeep`, because training outputs and model artifacts can become large quickly.

Notebook 02 also writes `adaption_dataset_diagnosis.json`, which captures the API-visible Adaption dataset description, status, and evaluation fields before the pilot run starts.

Recommended layout:

```text
results/
├── adaption_dataset_diagnosis.json
├── adaption_run_metadata.json
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
