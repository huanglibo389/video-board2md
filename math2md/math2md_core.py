#!/usr/bin/env python3
"""Unified images/PDF/video to MathMD course-bundle orchestrator."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


TOOLS_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from mathmd_shared import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    QwenClient,
    align_cross_source_problems,
    fuse_problem_directory,
    merge_problem_bundles,
    resolve_qwen_config,
    sanitize_stem,
)
from mathmd_shared.markdown import atomic_write_json, atomic_write_text  # noqa: E402


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm", ".m4v"}
BOARD_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".pdf"}
SUPPORTED_SUFFIXES = VIDEO_SUFFIXES | BOARD_SUFFIXES
SPLIT_FORMAT_RE = re.compile(r"\.f\d+\.(?:mp4|webm)$", flags=re.IGNORECASE)
BOARD2MD = TOOLS_ROOT / "board2md" / "board2md"
VIDEO2MD = TOOLS_ROOT / "video2md_review" / "video2md"


@dataclass(frozen=True)
class RoutedInputs:
    board: tuple[Path, ...]
    video: tuple[Path, ...]
    ignored: tuple[Path, ...] = ()


@dataclass(frozen=True)
class PipelineCommand:
    kind: str
    source: tuple[Path, ...]
    bundle_dir: Path
    argv: tuple[str, ...]


def _is_complete_media(path: Path) -> bool:
    name = path.name.lower()
    return not name.endswith(".part") and not SPLIT_FORMAT_RE.search(name)


def expand_and_route_inputs(raw_inputs: Sequence[str | Path], *, recursive: bool = True) -> RoutedInputs:
    candidates: list[Path] = []
    ignored: list[Path] = []
    for raw_input in raw_inputs:
        path = Path(os.path.expanduser(str(raw_input))).resolve()
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            candidates.extend(sorted(item for item in iterator if item.is_file()))
        elif path.is_file():
            candidates.append(path)
        else:
            ignored.append(path)

    board: list[Path] = []
    video: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        suffix = path.suffix.lower()
        if suffix in BOARD_SUFFIXES:
            board.append(path)
        elif suffix in VIDEO_SUFFIXES and _is_complete_media(path):
            video.append(path)
        elif suffix not in SUPPORTED_SUFFIXES or not _is_complete_media(path):
            ignored.append(path)
    return RoutedInputs(tuple(board), tuple(video), tuple(ignored))


def build_pipeline_commands(
    routed: RoutedInputs,
    *,
    out_root: Path,
    profile: str,
    interval: int,
    fuse_mode: str,
    fuse_engine: str,
    fuse_rounds: int,
    fusion_threshold: float,
    base_url: str,
    model: str,
    continue_on_error: bool,
) -> list[PipelineCommand]:
    out_root = Path(out_root)
    commands: list[PipelineCommand] = []
    if routed.board:
        bundle_dir = out_root / "board"
        argv = [
            sys.executable,
            str(BOARD2MD),
            *(str(path) for path in routed.board),
            "--output",
            str(bundle_dir / "board_notes.md"),
            "--profile",
            profile,
            "--bundle-dir",
            str(bundle_dir),
            "--bundle-fuse",
            fuse_mode,
            "--bundle-fuse-engine",
            fuse_engine,
            "--bundle-fuse-rounds",
            str(fuse_rounds),
            "--bundle-fuse-threshold",
            str(fusion_threshold),
            "--base-url",
            base_url,
            "--model",
            model,
        ]
        if continue_on_error:
            argv.append("--continue-on-error")
        commands.append(PipelineCommand("board", routed.board, bundle_dir, tuple(argv)))

    for number, source in enumerate(routed.video, start=1):
        name = f"{number:03d}_{sanitize_stem(source.stem)[:70]}"
        bundle_dir = out_root / "videos" / name
        argv = (
            sys.executable,
            str(VIDEO2MD),
            str(source),
            "--out",
            str(bundle_dir),
            "--provider",
            "qwen",
            "--interval",
            str(interval),
            "--embed-images",
            "auto",
            "--fuse-problems",
            fuse_mode,
            "--fuse-engine",
            fuse_engine,
            "--fuse-rounds",
            str(fuse_rounds),
            "--fusion-threshold",
            str(fusion_threshold),
            "--base-url",
            base_url,
            "--model",
            model,
        )
        commands.append(PipelineCommand("video", (source,), bundle_dir, argv))
    return commands


def bundle_is_complete(bundle_dir: Path) -> bool:
    bundle_dir = Path(bundle_dir)
    return (
        (bundle_dir / "index.md").is_file()
        and (bundle_dir / "mathmd_manifest.json").is_file()
        and any((bundle_dir / "problems").glob("p*.md"))
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统一处理视频、板书图片和 PDF，并融合为一个 MathMD 课程内容包"
    )
    parser.add_argument("inputs", nargs="+", help="视频、图片、PDF 或包含这些文件的目录")
    parser.add_argument("--out-root", type=Path, default=Path("math2md_out"), help="输出根目录")
    parser.add_argument(
        "--profile",
        choices=["custom", "quick", "verified", "lecture"],
        default="verified",
        help="图片/PDF 的 board2md 处理深度",
    )
    parser.add_argument("--interval", type=int, default=60, help="视频固定截帧间隔秒数")
    parser.add_argument("--fuse", choices=["never", "auto", "always"], default="auto")
    parser.add_argument("--fuse-engine", choices=["semantic", "rules"], default="semantic")
    parser.add_argument("--fuse-rounds", type=int, default=3)
    parser.add_argument("--fusion-threshold", type=float, default=0.72)
    parser.add_argument("--base-url", help=f"本地 Qwen 地址；默认 {DEFAULT_BASE_URL} 或环境变量")
    parser.add_argument("--model", help=f"本地 Qwen 模型；默认 {DEFAULT_MODEL} 或环境变量")
    parser.add_argument("--no-recursive", action="store_true", help="目录只扫描一层")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true", help="即使已有完整内容包也重新处理")
    parser.add_argument("--dry-run", action="store_true", help="只显示分流和命令，不执行模型或媒体处理")
    return parser


def _result_code(result: object) -> int:
    return int(getattr(result, "returncode", 0) or 0)


def main(
    argv: Sequence[str] | None = None,
    *,
    run_command: Callable[..., object] = subprocess.run,
) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    routed = expand_and_route_inputs(args.inputs, recursive=not args.no_recursive)
    if not routed.board and not routed.video:
        print("ERROR: 没有找到可处理的视频、图片或 PDF", file=sys.stderr)
        return 1

    config = resolve_qwen_config(base_url=args.base_url, model=args.model)
    out_root = args.out_root.expanduser().resolve()
    commands = build_pipeline_commands(
        routed,
        out_root=out_root,
        profile=args.profile,
        interval=args.interval,
        fuse_mode=args.fuse,
        fuse_engine=args.fuse_engine,
        fuse_rounds=max(0, args.fuse_rounds),
        fusion_threshold=args.fusion_threshold,
        base_url=config.base_url,
        model=config.model,
        continue_on_error=args.continue_on_error,
    )

    print(f"Math2MD: {len(routed.video)} 个视频，{len(routed.board)} 个图片/PDF 文件", flush=True)
    if routed.ignored:
        print(f"忽略 {len(routed.ignored)} 个不存在、不完整或不支持的文件")

    if args.dry_run:
        for command in commands:
            print(f"[{command.kind}] {shlex.join(command.argv)}")
        print(f"[course] merge -> {out_root / 'course'}")
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    completed_bundles: list[Path] = []
    failures: list[dict[str, object]] = []
    for number, command in enumerate(commands, start=1):
        if bundle_is_complete(command.bundle_dir) and not args.force:
            print(f"[{number}/{len(commands)}] 复用已有内容包: {command.bundle_dir}", flush=True)
            completed_bundles.append(command.bundle_dir)
            continue
        print(
            f"[{number}/{len(commands)}] 处理 {command.kind}: {', '.join(path.name for path in command.source)}",
            flush=True,
        )
        error_message = ""
        try:
            result = run_command(list(command.argv))
            code = _result_code(result)
        except OSError as exc:
            code = 2
            error_message = str(exc)
        bundle_ready = bundle_is_complete(command.bundle_dir)
        if bundle_ready:
            completed_bundles.append(command.bundle_dir)
        if code or not bundle_ready:
            failure = {
                "kind": command.kind,
                "sources": [str(path) for path in command.source],
                "returncode": code or 2,
                "error": error_message or ("output bundle was not created" if not bundle_ready else ""),
            }
            failures.append(failure)
            if not args.continue_on_error:
                atomic_write_json(out_root / "run_report.json", {"status": "failed", "failures": failures})
                return code or 2

    if not completed_bundles:
        print("ERROR: 没有成功生成可融合的内容包", file=sys.stderr)
        atomic_write_json(out_root / "run_report.json", {"status": "failed", "failures": failures})
        return 2

    course_dir = out_root / "course"
    merge_problem_bundles(completed_bundles, course_dir, generator="math2md", model=config.model)
    alignment_report: dict[str, object] | None = None
    fusion_report: dict[str, object] | None = None
    if args.fuse != "never":
        complete = None
        if args.fuse_engine == "semantic":
            client = QwenClient(config)
            complete = lambda prompt, limit: client.complete(prompt, max_tokens=limit)
        alignment_report = align_cross_source_problems(
            course_dir,
            complete=complete,
            engine=args.fuse_engine,
            threshold=args.fusion_threshold,
            generator="math2md",
            model=config.model,
        )
        fusion_report = fuse_problem_directory(
            course_dir,
            complete=complete,
            mode=args.fuse,
            engine=args.fuse_engine,
            rounds=max(0, args.fuse_rounds),
            threshold=args.fusion_threshold,
            generator="math2md",
            model=config.model,
        )

    top_index = [
        "# Math2MD 统一课程包",
        "",
        "- [打开融合后的课程索引](course/index.md)",
        "- [查看质量报告](course/quality_report.json)",
        "- [查看机器可读清单](course/mathmd_manifest.json)",
    ]
    if alignment_report is not None:
        top_index.append("- [查看跨来源对齐记录](course/alignment_manifest.json)")
    top_index.extend(
        [
            "",
            "## 来源内容包",
            "",
            *(f"- `{bundle}`" for bundle in completed_bundles),
            "",
        ]
    )
    atomic_write_text(out_root / "index.md", "\n".join(top_index))
    report = {
        "status": "partial" if failures else "complete",
        "elapsed_seconds": round(time.time() - started, 3),
        "board_inputs": [str(path) for path in routed.board],
        "video_inputs": [str(path) for path in routed.video],
        "ignored": [str(path) for path in routed.ignored],
        "bundles": [str(path) for path in completed_bundles],
        "course": str(course_dir),
        "alignment": alignment_report,
        "fusion": fusion_report,
        "failures": failures,
    }
    atomic_write_json(out_root / "run_report.json", report)
    print(f"完成: {course_dir / 'index.md'}")
    return 3 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
