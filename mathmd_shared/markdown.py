"""Shared Markdown cleanup, quality, and chunking helpers."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Sequence


DEFAULT_FORBIDDEN_PROVENANCE_TERMS = [
    "板书体面",
    "老师口述",
    "OCR",
    "根据图片",
    "根据板书",
]


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, data: object) -> None:
    atomic_write_text(Path(path), json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text or "", flags=re.DOTALL).lstrip()


def strip_code_fence(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json|markdown|md)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def replace_inline_math_delimiters(line: str) -> str:
    parts = re.split(r"(`[^`]*`)", line)
    for index, part in enumerate(parts):
        if part.startswith("`") and part.endswith("`"):
            continue
        parts[index] = part.replace(r"\(", "$").replace(r"\)", "$")
    return "".join(parts)


def normalize_markdown_for_obsidian(text: str) -> str:
    lines = (text or "").splitlines()
    converted: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            converted.append(line)
            in_fence = not in_fence
            continue
        quote_match = re.match(r"^(\s*>+\s*)(\\\[|\\\])\s*$", line)
        if not in_fence and quote_match:
            converted.append(f"{quote_match.group(1)}$$")
            continue
        if not in_fence and stripped in (r"\[", r"\]"):
            indent = line[: len(line) - len(line.lstrip())]
            converted.append(f"{indent}$$")
            continue
        if not in_fence:
            line = replace_inline_math_delimiters(line)
        converted.append(line)
    trailing = "\n" if text and text.endswith("\n") else ""
    return "\n".join(converted) + trailing


def sanitize_stem(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", text or "").strip("._")
    return cleaned or "document"


def extract_title(markdown: str, fallback: str = "数学内容") -> str:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", markdown or "")
    return match.group(1).strip() if match else fallback


def build_quality_report(
    markdown: str,
    *,
    forbidden_terms: Sequence[str] | None = None,
    check_image_links: bool = False,
    base_dir: Path | None = None,
) -> dict[str, object]:
    issues: list[dict[str, object]] = []
    terms = DEFAULT_FORBIDDEN_PROVENANCE_TERMS if forbidden_terms is None else forbidden_terms
    for term in terms:
        if term and term in markdown:
            issues.append(
                {
                    "kind": "forbidden_provenance",
                    "term": term,
                    "message": f"Final output contains process/provenance term: {term}",
                }
            )
    if markdown.count("$") % 2 == 1:
        issues.append(
            {
                "kind": "math_delimiter",
                "message": "Odd number of dollar signs may indicate an unclosed math delimiter.",
            }
        )
    if "```" in markdown:
        issues.append(
            {
                "kind": "code_fence",
                "message": "Final Markdown still contains a code fence.",
            }
        )
    if not re.search(r"(?m)^#", markdown or ""):
        issues.append({"kind": "heading", "message": "Markdown has no heading."})
    if check_image_links and base_dir is not None:
        for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown or ""):
            if re.match(r"^[a-z]+://", target):
                continue
            if not (Path(base_dir) / target).exists():
                issues.append(
                    {
                        "kind": "missing_image",
                        "target": target,
                        "message": f"Referenced image is missing: {target}",
                    }
                )
    return {"summary": {"issue_count": len(issues)}, "issues": issues}


def build_chunks(markdown: str) -> list[dict[str, object]]:
    lines = (markdown or "").splitlines()
    start = 0
    while start < len(lines):
        stripped = lines[start].strip()
        if stripped and not stripped.startswith(("<!-- mathmd:", "<!-- board2md:")):
            break
        start += 1
    next_content = start + 1
    while next_content < len(lines) and not lines[next_content].strip():
        next_content += 1
    # board2md wraps each model result in a source-level ## heading. Treat the
    # model's following # heading as the real chunk boundary for compatibility.
    if (
        start < len(lines)
        and next_content < len(lines)
        and lines[start].startswith("## ")
        and lines[next_content].startswith("#")
    ):
        start = next_content

    chunks: list[dict[str, object]] = []
    heading = "Document"
    current: list[str] = []
    for line in lines[start:]:
        if line.strip().startswith("<!-- mathmd:") or line.strip().startswith("<!-- board2md:"):
            continue
        if line.startswith("#"):
            if current and any(part.strip() for part in current):
                chunks.append({"heading": heading, "text": "\n".join(current).strip()})
            heading = line.strip()
            current = [line]
        else:
            current.append(line)
    if current and any(line.strip() for line in current):
        chunks.append({"heading": heading, "text": "\n".join(current).strip()})
    return chunks
