from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx

from .utils import ensure_dir


def blueprint_text(blueprint: dict[str, Any]) -> str:
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


def get_adaptation_job_config(run_config: dict[str, Any]) -> dict[str, Any]:
    job_config = run_config.get("adaptation_job")
    if not job_config:
        raise KeyError("configs/adaption_run.yaml must define adaptation_job.")
    return job_config


def wait_until_ingested(
    client: Any,
    dataset_id: str,
    timeout_seconds: int | None = None,
) -> Any:
    started = time.time()
    while True:
        status = client.datasets.get_status(dataset_id)
        status_name = str(getattr(status, "status", "") or "").lower()
        row_count = getattr(status, "row_count", None)

        if status_name in {"failed", "error"}:
            raise RuntimeError(f"Dataset ingestion failed with status: {status_name}")

        if row_count is not None:
            return status

        if timeout_seconds is not None and time.time() - started > timeout_seconds:
            row_note = f" Current row_count={row_count}."
            raise TimeoutError(f"Dataset ingestion still pending after {timeout_seconds}s.{row_note}")
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
    client: Any,
    dataset_id: str,
    timeout_seconds: int | None,
    heartbeat_seconds: int = 60,
) -> Any:
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


def download_to_file(client: Any, dataset_id: str, output_path: Path) -> None:
    download_result = client.datasets.download(
        dataset_id,
        file_format=output_path.suffix.lstrip(".") or "csv",
    )
    output_path = ensure_dir(output_path)

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


def to_plain_data(value: Any) -> Any:
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


def capture_dataset_diagnosis(
    client: Any,
    dataset_id: str,
    dataset_name: str,
    include_evaluation: bool = False,
) -> dict[str, Any]:
    """Capture the API-visible fields closest to the UI diagnosis view."""
    dataset_record = client.datasets.get(dataset_id)
    dataset_status = client.datasets.get_status(dataset_id)

    dataset_listing = client.datasets.list(q=dataset_name, limit=10)
    listed_datasets = getattr(dataset_listing, "datasets", None) or getattr(
        dataset_listing, "data", None
    )
    if listed_datasets is None:
        listed_datasets = list(dataset_listing)
    listed_dataset = next(
        (item for item in listed_datasets if getattr(item, "dataset_id", None) == dataset_id),
        None,
    )

    pre_run_evaluation = None
    evaluation_error = None
    if include_evaluation:
        try:
            pre_run_evaluation = client.datasets.get_evaluation(dataset_id)
        except Exception as exc:
            evaluation_error = repr(exc)

    return {
        "dataset_id": dataset_id,
        "capture_stage": "after_upload_before_adaptation_run",
        "api_docs_note": (
            "Public Adaption docs expose dataset list/get/status/evaluation APIs. "
            "The UI Data Diagnosis page may include additional fields that are not documented as a separate endpoint."
        ),
        "dataset_record": to_plain_data(dataset_record),
        "dataset_status": to_plain_data(dataset_status),
        "listed_dataset": to_plain_data(listed_dataset),
        "pre_run_evaluation": to_plain_data(pre_run_evaluation),
        "pre_run_evaluation_error": evaluation_error,
        "pre_run_evaluation_skipped": not include_evaluation,
    }
