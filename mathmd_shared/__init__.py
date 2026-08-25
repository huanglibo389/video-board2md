"""Shared local-Qwen and Markdown infrastructure for board2md/video2md."""

from .alignment import align_cross_source_problems
from .bundle import (
    BundleAsset,
    ProblemDocument,
    merge_problem_bundles,
    refresh_bundle_metadata,
    write_problem_bundle,
)
from .fusion import fuse_problem_directory
from .markdown import (
    atomic_write_json,
    atomic_write_text,
    build_chunks,
    build_quality_report,
    extract_title,
    normalize_markdown_for_obsidian,
    sanitize_stem,
    strip_code_fence,
    strip_think_blocks,
)
from .qwen import DEFAULT_BASE_URL, DEFAULT_MODEL, QwenClient, QwenConfig, resolve_qwen_config

__all__ = [
    "BundleAsset",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "ProblemDocument",
    "QwenClient",
    "QwenConfig",
    "align_cross_source_problems",
    "atomic_write_json",
    "atomic_write_text",
    "build_chunks",
    "build_quality_report",
    "extract_title",
    "fuse_problem_directory",
    "merge_problem_bundles",
    "normalize_markdown_for_obsidian",
    "refresh_bundle_metadata",
    "resolve_qwen_config",
    "sanitize_stem",
    "strip_code_fence",
    "strip_think_blocks",
    "write_problem_bundle",
]
