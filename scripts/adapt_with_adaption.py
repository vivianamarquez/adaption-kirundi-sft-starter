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
    lines = ["Goal:", str(blueprint["goal"]).strip(), "", "Constraints:"]
    lines.extend(f"- {item}" for item in blueprint.get("constraints", []))
    lines.append("")
    lines.append("Quality checks:")
    lines.extend(f"- {item}" for item in blueprint.get("quality_checks", []))
    return "\n".join(lines)


def wait_until_ingested(client, dataset_id: str, timeout_seconds: int = 600) -> None:
    started = time.time()
    while True:
        status = client.datasets.get_status(dataset_id)
        if getattr(status, "row_count", None) is not None:
            return
        if time.time() - started > timeout_seconds:
            raise TimeoutError(f"Dataset ingestion still pending after {timeout_seconds}s")
        time.sleep(2)


def download_to_file(client, dataset_id: str, output_path: Path) -> None:
    url = client.datasets.download(dataset_id, file_format=output_path.suffix.lstrip(".") or "csv")
    response = httpx.get(url, timeout=120)
    response.raise_for_status()
    ensure_dir(output_path)
    output_path.write_bytes(response.content)


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
        "column_mapping": run_cfg["column_mapping"],
        "brand_controls": brand_controls,
        "recipe_specification": recipe_specification,
        "pilot": run_cfg["pilot"],
        "full_requested": args.full,
    }

    if args.dry_run or not os.environ.get("ADAPTION_API_KEY"):
        print(json.dumps(plan, indent=2))
        print("\nDry run only. Set ADAPTION_API_KEY and rerun without --dry-run to call Adaption.")
        return

    from adaption import Adaption, DatasetTimeout

    client = Adaption(api_key=os.environ["ADAPTION_API_KEY"])
    upload = client.datasets.upload_file(str(input_path), name=run_cfg["dataset_name"])
    dataset_id = upload.dataset_id
    wait_until_ingested(client, dataset_id)

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

    try:
        final = client.datasets.wait_for_completion(dataset_id, timeout=run_cfg["pilot"]["timeout_seconds"])
        print(f"Pilot finished: {final.status}")
        if getattr(final, "error", None):
            raise RuntimeError(final.error.message)
    except DatasetTimeout as exc:
        raise TimeoutError(f"Pilot timed out after {exc.timeout}s") from exc

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
        client.datasets.wait_for_completion(dataset_id, timeout=run_cfg["full_run"]["timeout_seconds"])

    output_path = PROJECT_ROOT / run_cfg["output_path"]
    download_to_file(client, dataset_id, output_path)

    metadata_path = PROJECT_ROOT / run_cfg["metadata_path"]
    ensure_dir(metadata_path)
    metadata_path.write_text(json.dumps({"dataset_id": dataset_id, "plan": plan}, indent=2), encoding="utf-8")
    print(f"Downloaded adapted data to {output_path}")


if __name__ == "__main__":
    main()
