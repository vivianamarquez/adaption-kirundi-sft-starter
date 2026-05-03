# Onboarding Feedback

This document translates the build experience into onboarding suggestions for developers who are new to Adaption, SFT, or low-resource language evaluation.

## Ideal First-Run Path

1. Start with a tiny dataset and a documented schema.
2. Upload the file.
3. Confirm column mapping.
4. Run an estimate.
5. Run a pilot on 10-25 rows.
6. Inspect before/after examples.
7. Download the adapted output.
8. Export to a training-ready format.
9. Train a small SFT run.
10. Evaluate on at least one automatic metric and one qualitative sample table.

## Onboarding Improvements For Adaption

- Official SFT starter repo with notebooks and scripts.
- Sample dataset that requires no customer data.
- Before/after table UI that highlights row-level changes.
- SFT export presets for Tinker, TRL, OpenAI fine-tuning JSONL, and generic chat JSONL.
- Low-resource language walkthrough showing how to avoid overclaiming.
- Evaluation rubric templates:
  - language adherence
  - task accuracy
  - factuality risk
  - formatting consistency
  - native speaker review
- A "pilot first" checklist that appears before large runs.
- Clearer explanations of how blueprint instructions interact with recipe toggles.

## Helpful Defaults

For an SFT-focused workflow, useful defaults would be:

| Setting | Suggested default | Why |
|---|---|---|
| `estimate_first` | true | Avoids accidental spend |
| `pilot_max_rows` | 25 | Makes review fast |
| `reasoning_traces` | false | Avoids training on hidden reasoning by accident |
| `length` | concise | Keeps examples consistent |
| export format | chat JSONL | Common denominator for SFT |
| before/after sample size | 10 rows | Enough to spot obvious failures |

## Notes For This Repo

- The notebooks are intentionally readable even when credentials are missing.
- Cells that call paid or external APIs are marked as credential-required.
- The default flow avoids RLHF, DPO, and preference optimization so the learning goal stays focused on SFT data quality.
- The evaluation is a proxy. It should guide iteration, not replace native speaker review.
