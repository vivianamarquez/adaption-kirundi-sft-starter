# Results Directory

Generated model checkpoints, training metrics, and response tables should live here.

This directory is ignored by git except for this README and `.gitkeep`, because training outputs and model artifacts can become large quickly.

Notebook 02 also writes `adaption_dataset_diagnosis.json`, which captures the API-visible Adaption dataset description and status before the pilot run starts.

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
```
