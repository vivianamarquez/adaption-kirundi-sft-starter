# Onboarding Feedback

This document translates the build experience into onboarding suggestions for developers who are new to Adaption, SFT, or low-resource language evaluation.

## Ideal First-Run Path

1. Start with a tiny dataset and a documented schema.
2. Normalize the upload file so each prompt/completion pair is easy for the platform to ingest.
3. Upload the file.
4. Confirm the platform-ingested row count matches the expected local row count.
5. Confirm column mapping.
6. Run an estimate.
7. Run a small pilot before scaling up.
8. Inspect before/after examples.
9. Download the adapted output.
10. Export to a training-ready format.
11. Train a small SFT run.
12. Compare model outputs on a small qualitative sample table before adding heavier automatic evaluations.

## Onboarding Improvements For Adaption

- Official notebook-first SFT starter repo.
- Sample dataset that requires no customer data.
- Pre-upload CSV validation guidance, especially for embedded newlines inside prompt/completion fields.
- Rejected-row diagnostics when the uploaded file row count and API-ingested row count do not match.
- Before/after table UI that highlights row-level changes.
- SFT export presets for Tinker, TRL, OpenAI fine-tuning JSONL, and generic chat JSONL.
- Low-resource language walkthrough showing how to avoid overclaiming.
- Review rubric templates:
  - language quality
  - task completion
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
| `pilot_max_rows` | small review sample | Makes review fast |
| upload row-count check | required | Prevents requesting more rows than Adaption accepted |
| newline normalization | recommended | Avoids CSV ingestion surprises |
| `reasoning_traces` | false | Avoids training on hidden reasoning by accident |
| `length` | concise | Keeps examples consistent |
| export format | chat JSONL | Common denominator for SFT |
| before/after sample size | 10 rows | Enough to spot obvious failures |

## Notes For This Repo

- The Jupyter notebooks are intentionally runnable as real workflows when credentials are configured.
- Cells that call paid or external APIs should fail clearly if credentials, input files, or service access are missing.
- The default flow avoids RLHF, DPO, and preference optimization so the learning goal stays focused on SFT data quality.
- The qualitative comparison is a proxy. It should guide iteration, not replace native speaker review.
