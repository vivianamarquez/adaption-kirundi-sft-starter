from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from kirundi_sft_starter.utils import PROJECT_ROOT, ensure_dir, load_env, load_yaml, require_file


def blueprint_text(blueprint: dict) -> str:
    lines = ["Goal:", str(blueprint["goal"]).strip()]
    language_policy = blueprint.get("language_policy", {})
    if language_policy:
        lines.append("")
        lines.append("Language policy:")
        if language_policy.get("target_language"):
            lines.append(f"- Target language: {language_policy['target_language']}")
        if language_policy.get("source_language_issue"):
            lines.append(f"- Source language issue: {language_policy['source_language_issue'].strip()}")
        lines.extend(f"- {item}" for item in language_policy.get("instructions", []))
    lines.append("")
    lines.append("Constraints:")
    lines.extend(f"- {item}" for item in blueprint.get("constraints", []))
    lines.append("")
    lines.append("Quality checks:")
    lines.extend(f"- {item}" for item in blueprint.get("quality_checks", []))
    return "\n".join(lines)


def wait_until_ingested(client, dataset_id: str, timeout_seconds: int | None = None) -> None:
    started = time.time()
    while True:
        status = client.datasets.get_status(dataset_id)
        if getattr(status, "row_count", None) is not None:
            return
        if timeout_seconds is not None and time.time() - started > timeout_seconds:
            raise TimeoutError(f"Dataset ingestion still pending after {timeout_seconds}s")
        time.sleep(2)


def format_elapsed(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def wait_for_completion(
    client,
    dataset_id: str,
    timeout_seconds: int | None,
    heartbeat_seconds: int = 60,
):
    from adaption import DatasetTimeout

    started = time.time()
    print("Waiting for Adaption run to finish...")

    while True:
        elapsed = time.time() - started
        if timeout_seconds is not None:
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                raise TimeoutError(f"Adaption run still pending after {timeout_seconds}s")
            wait_seconds = max(1, min(heartbeat_seconds, int(remaining)))
        else:
            wait_seconds = heartbeat_seconds

        try:
            return client.datasets.wait_for_completion(dataset_id, timeout=wait_seconds)
        except DatasetTimeout:
            print(f"Still waiting for Adaption run... elapsed {format_elapsed(time.time() - started)}")


def download_to_file(client, dataset_id: str, output_path: Path) -> None:
    download_result = client.datasets.download(
        dataset_id,
        file_format=output_path.suffix.lstrip(".") or "csv",
    )
    ensure_dir(output_path)

    if isinstance(download_result, bytes):
        output_path.write_bytes(download_result)
        return

    if isinstance(download_result, str):
        if download_result.startswith(("http://", "https://")):
            response = httpx.get(download_result, timeout=120)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            return
        output_path.write_text(download_result, encoding="utf-8")
        return

    content = getattr(download_result, "content", None)
    if isinstance(content, bytes):
        output_path.write_bytes(content)
        return

    text = getattr(download_result, "text", None)
    if isinstance(text, str):
        output_path.write_text(text, encoding="utf-8")
        return

    raise TypeError(f"Unsupported Adaption download response type: {type(download_result)!r}")


def to_plain_data(value):
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json", exclude_none=True)
        except TypeError:
            return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    return value


def capture_dataset_diagnosis(client, dataset_id: str, dataset_name: str) -> dict:
    """Capture the documented API-visible fields closest to the UI diagnosis view."""
    dataset_record = client.datasets.get(dataset_id)
    dataset_status = client.datasets.get_status(dataset_id)

    dataset_listing = client.datasets.list(q=dataset_name, limit=10)
    listed_datasets = getattr(dataset_listing, "datasets", None) or getattr(dataset_listing, "data", None)
    if listed_datasets is None:
        listed_datasets = list(dataset_listing)
    listed_dataset = next(
        (item for item in listed_datasets if getattr(item, "dataset_id", None) == dataset_id),
        None,
    )

    try:
        pre_pilot_evaluation = client.datasets.get_evaluation(dataset_id)
        evaluation_error = None
    except Exception as exc:
        pre_pilot_evaluation = None
        evaluation_error = repr(exc)

    return {
        "dataset_id": dataset_id,
        "capture_stage": "after_upload_before_pilot_run",
        "api_docs_note": (
            "Public Adaption docs expose dataset list/get/status/evaluation APIs. "
            "The UI Data Diagnosis page may include additional fields that are not documented as a separate endpoint."
        ),
        "dataset_record": to_plain_data(dataset_record),
        "dataset_status": to_plain_data(dataset_status),
        "listed_dataset": to_plain_data(listed_dataset),
        "pre_pilot_evaluation": to_plain_data(pre_pilot_evaluation),
        "pre_pilot_evaluation_error": evaluation_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Adaption on the prepared Kirundi SFT file.")
    parser.add_argument("--config", default="configs/adaption_run.yaml")
    parser.add_argument("--blueprint", default="configs/adaption_blueprint.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print the run plan without calling Adaption.")
    parser.add_argument("--full", action="store_true", help="Run the full adaptation after the pilot.")
    args = parser.parse_args()

    load_env()
    run_cfg = load_yaml(PROJECT_ROOT / args.config)
    blueprint_cfg = load_yaml(PROJECT_ROOT / args.blueprint)
    input_path = require_file(PROJECT_ROOT / run_cfg["input_path"], "Run scripts/prepare_sft_dataset.py first.")

    brand_controls = dict(blueprint_cfg.get("adaption_brand_controls", {}))
    brand_controls["blueprint"] = blueprint_text(blueprint_cfg)
    recipe_specification = blueprint_cfg.get("adaption_recipe_specification", {})

    plan = {
        "input_path": str(input_path),
        "output_path": run_cfg["output_path"],
        "diagnosis_path": run_cfg.get("diagnosis_path", "results/adaption_dataset_diagnosis.json"),
        "column_mapping": run_cfg["column_mapping"],
        "brand_controls": brand_controls,
        "recipe_specification": recipe_specification,
        "pilot": run_cfg["pilot"],
        "ingestion_timeout_seconds": run_cfg.get("ingestion_timeout_seconds"),
        "full_requested": args.full,
    }

    if args.dry_run:
        print(json.dumps(plan, indent=2))
        print("\nDry run only. Rerun without --dry-run to call Adaption.")
        return

    api_key = os.environ.get("ADAPTION_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ADAPTION_API_KEY. Add it to .env before running Adaption.")

    from adaption import Adaption

    client = Adaption(api_key=api_key)
    upload = client.datasets.upload_file(str(input_path), name=run_cfg["dataset_name"])
    dataset_id = upload.dataset_id
    wait_until_ingested(client, dataset_id, timeout_seconds=run_cfg.get("ingestion_timeout_seconds"))

    diagnosis_path = PROJECT_ROOT / run_cfg.get("diagnosis_path", "results/adaption_dataset_diagnosis.json")
    ensure_dir(diagnosis_path)
    diagnosis = capture_dataset_diagnosis(client, dataset_id, run_cfg["dataset_name"])
    diagnosis_path.write_text(json.dumps(diagnosis, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Saved API-visible dataset diagnosis metadata to {diagnosis_path}")

    estimate = client.datasets.run(
        dataset_id,
        column_mapping=run_cfg["column_mapping"],
        brand_controls=brand_controls,
        recipe_specification=recipe_specification,
        job_specification={"max_rows": run_cfg["pilot"]["max_rows"]},
        estimate=True,
    )
    print(f"Pilot estimate: {estimate.estimated_credits_consumed} credits, ~{estimate.estimated_minutes} min")

    pilot = client.datasets.run(
        dataset_id,
        column_mapping=run_cfg["column_mapping"],
        brand_controls=brand_controls,
        recipe_specification=recipe_specification,
        job_specification={"max_rows": run_cfg["pilot"]["max_rows"]},
    )
    print(f"Pilot run started: {pilot.run_id}")

    final = wait_for_completion(client, dataset_id, timeout_seconds=run_cfg["pilot"]["timeout_seconds"])
    print(f"Pilot finished: {final.status}")
    if getattr(final, "error", None):
        raise RuntimeError(final.error.message)

    if args.full or run_cfg["run_controls"].get("run_full_by_default", False):
        full_job = {}
        if run_cfg["full_run"].get("max_rows"):
            full_job["max_rows"] = run_cfg["full_run"]["max_rows"]
        full_run = client.datasets.run(
            dataset_id,
            column_mapping=run_cfg["column_mapping"],
            brand_controls=brand_controls,
            recipe_specification=recipe_specification,
            job_specification=full_job or None,
        )
        print(f"Full run started: {full_run.run_id}")
        wait_for_completion(client, dataset_id, timeout_seconds=run_cfg["full_run"]["timeout_seconds"])

    output_path = PROJECT_ROOT / run_cfg["output_path"]
    download_to_file(client, dataset_id, output_path)

    metadata_path = PROJECT_ROOT / run_cfg["metadata_path"]
    ensure_dir(metadata_path)
    metadata_path.write_text(json.dumps({"dataset_id": dataset_id, "plan": plan}, indent=2), encoding="utf-8")
    print(f"Downloaded adapted data to {output_path}")


if __name__ == "__main__":
    main()
