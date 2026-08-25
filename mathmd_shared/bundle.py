"""Common MathMD problem-bundle format for images, PDFs, and videos."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .markdown import (
    atomic_write_json,
    atomic_write_text,
    build_chunks,
    build_quality_report,
    extract_title,
    sanitize_stem,
)


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BundleAsset:
    path: Path
    label: str = "原始图像"
    timestamp: float | None = None


@dataclass(frozen=True)
class ProblemDocument:
    title: str
    markdown: str
    source: str
    source_kind: str
    source_id: str
    page: int | None = None
    t_start: float | None = None
    t_end: float | None = None
    assets: Sequence[BundleAsset] = field(default_factory=tuple)


def _inject_assets(markdown: str, image_lines: list[str]) -> str:
    if not image_lines:
        return markdown
    section = "## 参考图像\n\n" + "\n\n".join(image_lines) + "\n\n"
    match = re.search(r"(?m)^##\s+解答(?:（[^）]+）)?\s*$", markdown or "")
    if match:
        return markdown[: match.start()] + section + markdown[match.start() :]
    return (markdown or "").rstrip() + "\n\n" + section.rstrip() + "\n"


def _source_comment(document: ProblemDocument) -> str:
    fields: dict[str, object] = {
        "kind": document.source_kind,
        "source": document.source,
        "id": document.source_id,
    }
    if document.page is not None:
        fields["page"] = document.page
    if document.t_start is not None:
        fields["t_start"] = document.t_start
    if document.t_end is not None:
        fields["t_end"] = document.t_end
    return f"<!-- mathmd: {json.dumps(fields, ensure_ascii=False, separators=(',', ':'))} -->"


def _reset_problem_outputs(bundle_dir: Path) -> None:
    problems_dir = Path(bundle_dir) / "problems"
    if not problems_dir.exists():
        return
    for path in problems_dir.glob("p*.md"):
        path.unlink()
    assets_dir = problems_dir / "assets"
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    fusion_manifest = Path(bundle_dir) / "fusion_manifest.json"
    fusion_manifest.unlink(missing_ok=True)


def _source_records(markdown: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for raw in re.findall(r"(?m)^<!-- mathmd:\s*(.*?)\s*-->$", markdown or ""):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            records.append({"raw": raw})
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def write_problem_bundle(
    bundle_dir: Path,
    documents: Sequence[ProblemDocument],
    *,
    generator: str,
    model: str,
) -> dict[str, object]:
    bundle_dir = Path(bundle_dir)
    _reset_problem_outputs(bundle_dir)
    problems_dir = bundle_dir / "problems"
    assets_dir = problems_dir / "assets"
    problems_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    manifest_documents: list[dict[str, object]] = []
    for number, document in enumerate(documents, start=1):
        title = document.title or extract_title(document.markdown, document.source_id)
        filename = f"p{number:03d}_{sanitize_stem(title)[:60]}.md"
        image_lines: list[str] = []
        asset_records: list[dict[str, object]] = []
        for asset_number, asset in enumerate(document.assets, start=1):
            source_path = Path(asset.path)
            suffix = source_path.suffix.lower() or ".png"
            asset_name = f"p{number:03d}_{asset_number:02d}_{sanitize_stem(source_path.stem)}{suffix}"
            destination = assets_dir / asset_name
            shutil.copy2(source_path, destination)
            alt = asset.label or f"参考图像 {asset_number}"
            image_lines.append(f"![{alt}](assets/{asset_name})")
            asset_records.append(
                {
                    "file": f"problems/assets/{asset_name}",
                    "label": alt,
                    "timestamp": asset.timestamp,
                }
            )

        content = document.markdown.strip()
        if not re.search(r"(?m)^#\s+", content):
            content = f"# {title}\n\n{content}"
        content = _inject_assets(content, image_lines)
        content = f"{_source_comment(document)}\n\n{content.rstrip()}\n"
        atomic_write_text(problems_dir / filename, content)
        manifest_documents.append(
            {
                "number": number,
                "file": f"problems/{filename}",
                "title": title,
                "source": document.source,
                "source_kind": document.source_kind,
                "source_id": document.source_id,
                "page": document.page,
                "t_start": document.t_start,
                "t_end": document.t_end,
                "assets": asset_records,
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )

    return refresh_bundle_metadata(
        bundle_dir,
        generator=generator,
        model=model,
        document_metadata=manifest_documents,
    )


def _problem_number(path: Path) -> int:
    match = re.match(r"p(\d+)_", path.name)
    return int(match.group(1)) if match else 10**9


def list_problem_files(bundle_dir: Path) -> list[Path]:
    problems_dir = Path(bundle_dir) / "problems"
    return sorted(
        (
            path
            for path in problems_dir.glob("p*.md")
            if not path.name.endswith(".raw.md")
        ),
        key=lambda path: (_problem_number(path), path.name),
    )


def refresh_bundle_metadata(
    bundle_dir: Path,
    *,
    generator: str,
    model: str,
    document_metadata: Sequence[dict[str, object]] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    bundle_dir = Path(bundle_dir)
    problem_files = list_problem_files(bundle_dir)
    if document_metadata is None:
        prior_manifest = bundle_dir / "mathmd_manifest.json"
        if prior_manifest.is_file():
            try:
                prior_data = json.loads(prior_manifest.read_text(encoding="utf-8"))
                prior_documents = prior_data.get("documents", [])
                if isinstance(prior_documents, list):
                    document_metadata = prior_documents
            except (json.JSONDecodeError, OSError, TypeError):
                document_metadata = None
    metadata_by_file = {
        str(item.get("file", "")).removeprefix("problems/"): item
        for item in (document_metadata or [])
    }
    records: list[dict[str, object]] = []
    index_lines = ["# MathMD 内容索引", ""]
    for number, path in enumerate(problem_files, start=1):
        content = path.read_text(encoding="utf-8")
        source_kind = (extra or {}).get("source_kind")
        source = (extra or {}).get("source")
        if source_kind and not _source_records(content):
            source_payload = {
                "kind": source_kind,
                "source": source or "",
                "id": path.stem,
            }
            source_comment = f"<!-- mathmd: {json.dumps(source_payload, ensure_ascii=False, separators=(',', ':'))} -->"
            content = f"{source_comment}\n\n{content.lstrip()}"
            atomic_write_text(path, content)
        title = extract_title(content, path.stem)
        index_lines.append(f"- [{number}. {title}](problems/{path.name})")
        record = dict(metadata_by_file.get(path.name, {}))
        record.update(
            {
                "number": number,
                "file": f"problems/{path.name}",
                "title": title,
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )
        sources = _source_records(content)
        if sources:
            record["sources"] = sources
        linked_assets = []
        for label, target in re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", content):
            file_target = target
            if not re.match(r"^[a-z]+://", target) and not target.startswith("problems/"):
                file_target = f"problems/{target}"
            linked_assets.append({"file": file_target, "label": label})
        if linked_assets:
            record["linked_assets"] = linked_assets
        records.append(record)
    index_lines.extend(["", f"_共 {len(problem_files)} 个内容单元_", ""])
    atomic_write_text(bundle_dir / "index.md", "\n".join(index_lines))

    quality: dict[str, object] = {"summary": {"issue_count": 0}, "issues": []}
    for path in problem_files:
        file_report = build_quality_report(
            path.read_text(encoding="utf-8"),
            check_image_links=True,
            base_dir=path.parent,
        )
        for issue in file_report["issues"]:
            quality["issues"].append({"file": f"problems/{path.name}", **issue})
    if not problem_files:
        quality["issues"].append({"kind": "empty_bundle", "message": "Bundle has no problem Markdown files."})
    quality["summary"]["issue_count"] = len(quality["issues"])
    chunks = [
        {"file": f"problems/{path.name}", **chunk}
        for path in problem_files
        for chunk in build_chunks(path.read_text(encoding="utf-8"))
    ]
    manifest: dict[str, object] = {
        "schema": "mathmd-bundle",
        "schema_version": SCHEMA_VERSION,
        "generator": generator,
        "model": model,
        "documents": records,
        "quality_summary": quality["summary"],
    }
    if extra:
        manifest.update(extra)
    atomic_write_json(bundle_dir / "mathmd_manifest.json", manifest)
    atomic_write_json(bundle_dir / "quality_report.json", quality)
    atomic_write_json(bundle_dir / "chunks.json", chunks)
    return manifest


def merge_problem_bundles(
    bundle_dirs: Sequence[Path],
    destination: Path,
    *,
    generator: str = "math2md",
    model: str,
) -> dict[str, object]:
    destination = Path(destination)
    _reset_problem_outputs(destination)
    problems_dir = destination / "problems"
    assets_dir = problems_dir / "assets"
    problems_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    number = 0
    source_bundles: list[str] = []
    for bundle_number, bundle_dir in enumerate(bundle_dirs, start=1):
        bundle_dir = Path(bundle_dir)
        source_bundles.append(str(bundle_dir))
        prefix = f"{bundle_number:03d}_{sanitize_stem(bundle_dir.name)[:36]}"
        asset_mapping: dict[str, str] = {}
        source_assets = bundle_dir / "problems" / "assets"
        if source_assets.exists():
            for asset in sorted(source_assets.iterdir()):
                if not asset.is_file():
                    continue
                destination_name = f"{prefix}_{asset.name}"
                shutil.copy2(asset, assets_dir / destination_name)
                asset_mapping[f"assets/{asset.name}"] = f"assets/{destination_name}"

        for source_problem in list_problem_files(bundle_dir):
            number += 1
            content = source_problem.read_text(encoding="utf-8")
            for old, new in asset_mapping.items():
                content = content.replace(f"]({old})", f"]({new})")
            title = extract_title(content, source_problem.stem)
            filename = f"p{number:03d}_{prefix}_{sanitize_stem(title)[:45]}.md"
            atomic_write_text(problems_dir / filename, content)

    return refresh_bundle_metadata(
        destination,
        generator=generator,
        model=model,
        extra={"source_bundles": source_bundles},
    )
