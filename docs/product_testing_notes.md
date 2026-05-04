# Product Testing Notes

These notes capture observations from building this educational starter workflow around Adaption for a low-resource language SFT experiment.

## What Felt Clear

- The Adaption docs present the lifecycle in the same order a developer needs it: install, authenticate, ingest, adapt, wait, download.
- The SDK examples make the happy path easy to recognize: `Adaption(...)`, `datasets.upload_file(...)`, `datasets.run(...)`, `wait_for_completion(...)`, and `download(...)`.
- The docs explicitly recommend `estimate=True` before starting a real run, which is useful for educational projects and budget-conscious pilots.
- The `job_specification.max_rows` pattern is a good fit for low-risk pilots because a developer can test column mapping and behavior on a small subset first.
- The `brand_controls.blueprint` field is a flexible place to encode qualitative constraints such as language, tone, and data-cleaning goals.

## Where A New Developer May Get Stuck

- It may not be obvious what an ideal SFT input file should look like before upload.
- Column mapping is simple once understood, but beginners may need concrete examples for common training formats like chat JSONL, instruction/completion CSV, and preference data.
- The difference between dataset ingestion status, adaptation run status, and evaluation status can be confusing on the first pass.
- The local row count can differ from the Adaption-ingested row count. In this repo, pandas parsed 200 CSV rows locally, while Adaption reported 182 rows after upload. The likely issue was ingestion-sensitive CSV formatting, especially embedded newlines inside quoted `instruction` fields. A developer may see the dataset in the UI and still be blocked because the API accepted fewer rows than the configured run size.
- Downloaded output columns may vary depending on the original file and run configuration, so downstream conversion examples are important.
- Low-resource language users may not know whether Adaption is preserving language, improving style, or translating content unless examples make that explicit.

## What Examples Were Missing Or Would Help

- An official SFT starter repo with a small public dataset and before/after output.
- A low-resource language example with clear responsible framing and qualitative before/after review.
- A before/after diff view for adapted rows, including what changed and why.
- A pre-upload validation checklist for CSV files, including embedded newlines, blank prompt/completion fields, very short completions, malformed quoting, and the difference between local parser row count and platform-ingested row count.
- Export presets for common post-training formats:
  - chat JSONL
  - instruction/completion CSV
  - Hugging Face `datasets` upload
  - Tinker supervised fine-tuning format
- A Jupyter notebook that runs `estimate=True`, then a pilot, then a full run only after manual confirmation.

## Suggested Product Improvements

- Add an "SFT data improvement" preset that asks for prompt/completion columns and exports chat JSONL.
- Add a "run a pilot first" checklist in the app and docs.
- Include a small sample dataset in the docs so users can test without bringing their own data.
- After upload, show both "rows received" and "rows accepted for processing," with downloadable rejected-row details when those counts differ.
- Add CSV normalization guidance or an SDK helper that can flatten multiline fields before upload when a product flow expects one logical example per physical row.
- Add clearer examples of `brand_controls.blueprint` for language preservation and low-resource settings.
- Show a structured explanation of what Adaption changed in each row:
  - formatting fix
  - language consistency
  - deduplication
  - length adjustment
  - hallucination risk reduction
- Add review rubric templates for dataset quality, language quality, formatting, and native speaker review.

## Responsible Product Framing

This starter treats Adaption as a data-improvement step, not as proof that a model is linguistically correct or culturally grounded. For Kirundi or any low-resource language, automatic metrics should be paired with native speaker review before drawing strong conclusions.
