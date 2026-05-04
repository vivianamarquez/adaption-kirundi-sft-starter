# Results Directory

Generated Adaption metadata, Tinker training outputs, and model response files should live here.

This directory is ignored by git except for this README and `.gitkeep`, because training outputs and model artifacts can become large quickly.

Current notebook outputs:

- Notebook 02 writes Adaption metadata such as `adaption_dataset_diagnosis.json`, `adaption_dataset_evaluation.json`, and `adaption_run_metadata.json`.
- Notebook 03 writes the raw-data SFT run under `models/sft_raw/`.
- Notebook 04 writes the adapted-data SFT run under `models/sft_adapted/`.
- Notebook 05 writes comparable model responses under `responses/`.

Recommended layout:

```text
results/
├── adaption_dataset_diagnosis.json
├── adaption_dataset_evaluation.json
├── adaption_run_metadata.json
├── models/
│   ├── sft_raw/
│   └── sft_adapted/
└── responses/
    ├── base.jsonl
    ├── sft_raw.jsonl
    └── sft_adapted.jsonl
```
