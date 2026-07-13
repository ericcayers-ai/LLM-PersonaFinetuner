from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from app.core import dataset_builder, parsers
from app.core.parsers import DetectedFormat, get_format_meta
from app.core.synthetic_qa import generate_synthetic_qa
from app.config import settings
from app.errors import api_error
from app.models.schemas import (
    DatasetBuildRequest,
    DatasetPreviewExample,
    DatasetPreviewResponse,
    UploadResponse,
)
from app.storage import store

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

SUPPORTED_EXTENSIONS = parsers.SUPPORTED_EXTENSIONS


def _validate_upload_content(content: bytes, filename: str) -> None:
    if not content:
        raise api_error(
            400,
            "empty_file",
            "The uploaded file is empty.",
            hint="Choose a file that contains writing samples or chat messages.",
        )
    if len(content) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes // (1024 * 1024)
        raise api_error(
            400,
            "file_too_large",
            f"File exceeds the {limit_mb} MB upload limit.",
            hint="Split large exports into smaller files or remove attachments.",
        )
    if filename.lower().endswith(".json"):
        try:
            import json

            json.loads(content.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            raise api_error(
                400,
                "invalid_json",
                "The JSON file could not be parsed.",
                detail=str(e),
                hint="Export a fresh copy from Discord, Instagram, or validate JSON syntax.",
            ) from e


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise api_error(
            400,
            "missing_filename",
            "No filename was provided with the upload.",
            hint="Select a file using the file picker or drag-and-drop.",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise api_error(
            400,
            "unsupported_format",
            f"Unsupported file type: {ext or '(none)'}",
            hint=f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    content = await file.read()
    _validate_upload_content(content, file.filename)

    record = store.save_upload(file.filename, content)
    path = Path(record["path"])
    preview_text = content.decode("utf-8", errors="replace")[:8000]
    detected = parsers.detect_format(path, preview_text)
    meta = get_format_meta(detected)

    store.update_upload_meta(
        record["id"],
        detected_format=detected.value,
        format_label=meta["label"],
    )

    return UploadResponse(
        upload_id=record["id"],
        filename=record["filename"],
        file_type=record["file_type"],
        size_bytes=record["size_bytes"],
        detected_format=detected.value,
        format_label=meta["label"],
        requires_target_name=meta["requires_target_name"],
        sample_hint=meta["hint"],
    )


@router.post("/build", response_model=DatasetPreviewResponse)
def build_dataset(req: DatasetBuildRequest) -> DatasetPreviewResponse:
    if not req.upload_ids:
        raise api_error(
            400,
            "no_uploads",
            "At least one uploaded file is required.",
            hint="Upload writing samples or chat exports in the Source step first.",
        )
    if not req.persona_name.strip():
        raise api_error(400, "missing_persona", "Persona name is required.")
    if not req.system_prompt.strip():
        raise api_error(400, "missing_prompt", "System prompt is required.")

    parsed_items: list[dict] = []
    all_warnings: list[str] = []

    for upload_id in req.upload_ids:
        upload = store.get_upload(upload_id)
        if not upload:
            raise api_error(
                404,
                "upload_not_found",
                f"Upload {upload_id} was not found.",
                hint="Re-upload your files — uploads are stored locally and may have been cleared.",
            )

        path = Path(upload["path"])
        stored_fmt = upload.get("detected_format")
        fmt = DetectedFormat(stored_fmt) if stored_fmt else parsers.detect_format(path)

        try:
            items = parsers.parse_upload(path, fmt=fmt, target_name=req.target_name)
            all_warnings.extend(parsers.extract_meta_warnings(items))
            parsed_items.extend(parsers.strip_meta_items(items))
        except ValueError as e:
            raise api_error(
                400,
                "parse_error",
                str(e),
                hint="Check the target username/contact name matches the export exactly.",
            ) from e

    if not parsed_items:
        raise api_error(
            400,
            "no_content",
            "No usable content found in the uploaded files.",
            hint="Verify the export format and target name, then try again.",
        )

    if req.generate_synthetic_qa:
        text_samples = [
            item["content"] for item in parsed_items if item.get("type") == "text"
        ]
        if not text_samples:
            for item in parsed_items:
                if item.get("type") == "chat":
                    for msg in item.get("messages", []):
                        if msg.get("role") == "assistant":
                            text_samples.append(msg["content"])
        qa_pairs, qa_warnings = generate_synthetic_qa(
            text_samples,
            base_model=req.synthetic_base_model,
        )
        parsed_items.extend(qa_pairs)
        all_warnings.extend(qa_warnings)

    persona = store.create_persona(
        name=req.persona_name.strip(),
        system_prompt=req.system_prompt.strip(),
    )

    output_dir = Path("data/datasets") / persona["id"]
    train_path, val_path, meta = dataset_builder.build_dataset_files(
        parsed_items,
        req.system_prompt.strip(),
        output_dir,
        chunk_min_tokens=req.chunk_min_tokens,
        chunk_max_tokens=req.chunk_max_tokens,
    )

    meta_warnings = meta.get("warnings", [])
    meta_warnings.extend(all_warnings)
    meta["warnings"] = meta_warnings

    dataset = store.save_dataset(
        persona_id=persona["id"],
        train_path=train_path,
        val_path=val_path,
        meta=meta,
    )

    store.update_persona(persona["id"], dataset_id=dataset["id"])

    preview_rows = store.read_jsonl_preview(str(train_path), limit=10)
    examples = [DatasetPreviewExample(messages=row["messages"]) for row in preview_rows]

    return DatasetPreviewResponse(
        dataset_id=dataset["id"],
        persona_id=persona["id"],
        total_examples=meta["total_examples"],
        train_examples=meta["train_examples"],
        val_examples=meta["val_examples"],
        warnings=meta.get("warnings", []),
        examples=examples,
    )


@router.get("/{dataset_id}/preview", response_model=DatasetPreviewResponse)
def preview_dataset(dataset_id: str, limit: int = 10) -> DatasetPreviewResponse:
    dataset = store.get_dataset(dataset_id)
    if not dataset:
        raise api_error(404, "dataset_not_found", "Dataset not found.")

    preview_rows = store.read_jsonl_preview(dataset["train_path"], limit=limit)
    examples = [DatasetPreviewExample(messages=row["messages"]) for row in preview_rows]

    return DatasetPreviewResponse(
        dataset_id=dataset["id"],
        persona_id=dataset["persona_id"],
        total_examples=dataset.get("total_examples", len(preview_rows)),
        train_examples=dataset.get("train_examples", 0),
        val_examples=dataset.get("val_examples", 0),
        warnings=dataset.get("warnings", []),
        examples=examples,
    )
