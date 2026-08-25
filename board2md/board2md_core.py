#!/usr/bin/env python3
"""Core implementation for board2md."""

from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import requests

TOOLS_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from mathmd_shared import (  # noqa: E402
    DEFAULT_BASE_URL as SHARED_DEFAULT_BASE_URL,
    DEFAULT_MODEL as SHARED_DEFAULT_MODEL,
    BundleAsset,
    ProblemDocument,
    QwenClient,
    QwenConfig,
    build_chunks as shared_build_chunks,
    build_quality_report as shared_build_quality_report,
    extract_title,
    fuse_problem_directory,
    strip_code_fence as shared_strip_code_fence,
    strip_think_blocks,
    write_problem_bundle,
)

DEFAULT_BASE_URL = SHARED_DEFAULT_BASE_URL
DEFAULT_MODEL = SHARED_DEFAULT_MODEL
DEFAULT_PROVIDER = "qwen"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
SUPPORTED_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".pdf"}
PARTIAL_FAILURE_EXIT = 3
FATAL_PROCESSING_EXIT = 4
QUALITY_FAILURE_EXIT = 5
FORBIDDEN_PROVENANCE_TERMS = ["板书体面", "老师口述", "OCR", "根据图片", "根据板书", "源文件", "来源时间"]

PROMPTS = {
    "problem": """你是数学老师。这是一张题目/板书照片。

任务：识别图中所有题目，**只列出题面**（不要解答）。每题用 LaTeX 写公式。

**严禁**：英文双引号 `"`，用中文「」或单引号 '。

输出 markdown（不要 ```markdown 包裹）：
# [题号/主题]

**来源**: 板书
**关键词**: [...]

## 题面

[题目 1，LaTeX]

## 题面 2

[题目 2，LaTeX]""",
    "full": """你是数学竞赛教练。这是一张题目/板书照片。

任务：识别图中所有题目，并整理为 Markdown。每道题保留题面；如果图中含有解答、提示或推导，也一并转写。公式用 LaTeX。

**严禁**：英文双引号 `"`，用中文「」或单引号 '。

输出 markdown（不要 ```markdown 包裹）：
# [题号/主题]

**来源**: 板书
**关键词**: [...]

## 题面

[题面]

## 解答

[若图中没有解答，写：图中未见解答。]""",
    "raw": """你是严谨的 OCR 转写员。这是一张板书、讲义或题目照片。

任务：尽量忠实转写画面中的文字、公式、题号、列表和明显结构。不要补充图中没有的信息。公式用 LaTeX。

输出 markdown（不要 ```markdown 包裹）。""",
}

CROSS_VALIDATE_PROMPTS = {
    "pass_a": """你是板书 OCR 的独立识别 A。

任务：从图片中尽量完整识别数学板书信息，包括题面、定义、公式、证明片段、箭头关系、补充说明和位置关系。

要求：
1. 不要参考其他识别结果。
2. 保留零碎信息，但要说明它可能属于哪一个主题或步骤。
3. 输出 Markdown，不要写代码块。""",
    "pass_b": """你是板书 OCR 的独立识别 B。

任务：从图片中重新识别数学板书信息，重点检查 A 类系统容易遗漏的内容：角落补充、箭头、上下文承接、公式条件、证明中的隐含步骤。

要求：
1. 不要假设已有识别正确。
2. 对不确定内容用“疑似”标记。
3. 输出 Markdown，不要写代码块。""",
    "judge": """你是交叉验证裁判。请比较同一页板书的两份独立识别结果，合并可靠内容，并标注冲突。

输出必须是 JSON，不要代码块，字段如下：
{
  "status": "agree | conflict | partial",
  "confidence": 0.0 到 1.0,
  "conflicts": ["冲突或疑点"],
  "merged_markdown": "合并后的 Markdown",
  "audit_markdown": "给人类看的审校记录"
}

只在两份识别都支持时把内容放入 merged_markdown；不确定内容写进 audit_markdown。""",
}

LECTURE_SYNTHESIS_PROMPT = """你是数学竞赛教练和讲义编辑。请基于下面的板书识别片段，生成正式讲义。

目标：把人类板书中较为零碎的信息串联起来，形成完整、可教学、可发布的 Markdown 讲义。

要求：
1. 正文不要出现“板书体面”“老师口述”“OCR”“根据图片”等来源痕迹。
2. 可以补充必要定义、过渡句、漏写步骤和标准证明细节。
3. 不确定、冲突、无法确认的内容不要硬写进正文，应留给审校记录。
4. 使用 $...$ 和 $$...$$ 数学定界。
5. 输出正式讲义 Markdown，不要代码块。
"""


@dataclass(frozen=True)
class BoardInput:
    path: Path
    source_path: Path
    page: int | None = None
    original_path: Path | None = None

    @property
    def source_display(self) -> str:
        return str(self.source_path)

    @property
    def safe_id(self) -> str:
        stem = sanitize_stem(self.source_path.stem)
        if self.page is not None:
            return f"{stem}_p{self.page:03d}"
        return sanitize_stem(self.path.stem)


def sanitize_stem(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", text).strip("._")
    return cleaned or "page"


def image_mime_type(path: Path) -> str:
    """Return an API-friendly MIME type for a local image path."""
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".heic":
        return "image/heic"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def parse_page_range(value: str | None) -> list[int] | None:
    if not value:
        return None
    pages: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError(f"invalid page range: {part}")
            start = int(start_text)
            end = int(end_text)
            if start < 1 or end < 1 or end < start:
                raise ValueError(f"invalid page range: {part}")
            pages.update(range(start, end + 1))
        else:
            if not part.isdigit():
                raise ValueError(f"invalid page number: {part}")
            page = int(part)
            if page < 1:
                raise ValueError(f"invalid page number: {part}")
            pages.add(page)
    return sorted(pages)


def page_ranges(pages: Sequence[int] | None) -> list[tuple[int | None, int | None]]:
    if not pages:
        return [(None, None)]
    ranges: list[tuple[int | None, int | None]] = []
    start = pages[0]
    prev = pages[0]
    for page in pages[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append((start, prev))
        start = page
        prev = page
    ranges.append((start, prev))
    return ranges


def page_number_from_path(path: Path) -> int | None:
    match = re.search(r"-(\d+)\.png$", path.name)
    if not match:
        return None
    return int(match.group(1))


def load_prompt(mode: str, prompt_file: str | None) -> str:
    if prompt_file:
        return Path(os.path.expanduser(prompt_file)).read_text(encoding="utf-8")
    return PROMPTS[mode]


def apply_note_type(prompt: str, note_type: str) -> str:
    if note_type == "generic":
        return prompt
    if note_type == "whiteboard":
        return (
            prompt
            + "\n\n补充要求：这是白板/黑板类板书。请特别理解空间布局、角落补充、箭头连接、分栏、擦写痕迹、临时插入说明，并把零碎内容归并到合适的数学结构中。"
        )
    if note_type == "paper":
        return (
            prompt
            + "\n\n补充要求：这是纸面手写笔记。请按从上到下、从左到右的自然阅读顺序整理，并留意边注、小表格和旁批。"
        )
    return prompt + "\n\n补充要求：请先判断这是白板、纸面笔记还是通用图片，再采用合适的阅读顺序和版面理解策略。"


def m3_vision(
    image_paths: Sequence[Path],
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int = 4000,
    timeout: int = 180,
    retries: int = 3,
    request_post: Callable[..., object] = requests.post,
) -> str:
    """Call an OpenAI-compatible vision chat endpoint on local image files."""
    client = QwenClient(
        QwenConfig(
            base_url=base_url.rstrip("/"),
            model=model,
            api_key=api_key or None,
            timeout=timeout,
            retries=max(1, retries),
        ),
        request_post=request_post,
    )
    return client.vision(image_paths, prompt, max_tokens=max_tokens)


def build_auth_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def strip_think_block(text: str) -> str:
    """Remove model reasoning blocks from the beginning of a response."""
    return strip_think_blocks(text)


def strip_code_fence(text: str) -> str:
    return shared_strip_code_fence(text)


def parse_json_response(text: str) -> dict[str, object]:
    cleaned = strip_code_fence(strip_think_block(text))
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "status": "partial",
            "confidence": 0.0,
            "conflicts": ["judge did not return valid JSON"],
            "merged_markdown": cleaned,
            "audit_markdown": "交叉验证裁判未返回合法 JSON，已保留原始输出。",
        }
    if not isinstance(data, dict):
        return {
            "status": "partial",
            "confidence": 0.0,
            "conflicts": ["judge returned non-object JSON"],
            "merged_markdown": cleaned,
            "audit_markdown": "交叉验证裁判返回的 JSON 不是对象。",
        }
    return data


def clamp_confidence(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def ensure_text(value: object, fallback: str = "") -> str:
    if value is None:
        return fallback
    return str(value)


def expand_inputs(
    patterns: Sequence[str],
    pdf_tmp_root: Path,
    *,
    convert_pdfs: bool = True,
    page_range: Sequence[int] | None = None,
    render_dpi: int = 150,
    recursive: bool = False,
    run_command: Callable[..., object] = subprocess.run,
) -> list[BoardInput]:
    """Expand paths/globs/directories and optionally convert PDFs into PNG pages."""
    items: list[BoardInput] = []
    for pat in patterns:
        expanded_pat = os.path.expanduser(pat)
        matches = (
            [Path(p) for p in sorted(glob.glob(expanded_pat))]
            if glob.has_magic(expanded_pat)
            else [Path(expanded_pat)]
        )

        if not matches or not any(p.exists() for p in matches):
            print(f"  WARN skip (not found): {pat}", file=sys.stderr)
            continue

        for path in matches:
            items.extend(
                expand_path(
                    path,
                    pdf_tmp_root,
                    convert_pdfs=convert_pdfs,
                    page_range=page_range,
                    render_dpi=render_dpi,
                    recursive=recursive,
                    run_command=run_command,
                )
            )
    return items


def expand_path(
    path: Path,
    pdf_tmp_root: Path,
    *,
    convert_pdfs: bool,
    page_range: Sequence[int] | None,
    render_dpi: int,
    recursive: bool,
    run_command: Callable[..., object],
) -> list[BoardInput]:
    if not path.exists():
        print(f"  WARN skip (not found): {path}", file=sys.stderr)
        return []
    if path.is_dir():
        iterator = path.rglob("*") if recursive else path.glob("*")
        files = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
        return [
            BoardInput(path=file, source_path=file, original_path=file)
            for file in files
            if file.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ] + [
            item
            for file in files
            if file.suffix.lower() == ".pdf"
            for item in (
                convert_pdf(
                    file,
                    pdf_tmp_root,
                    page_range=page_range,
                    render_dpi=render_dpi,
                    run_command=run_command,
                )
                if convert_pdfs
                else [BoardInput(path=file, source_path=file, original_path=file)]
            )
        ]
    if not path.is_file():
        print(f"  WARN skip (not a file): {path}", file=sys.stderr)
        return []

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        print(f"  WARN skip (unsupported): {path}", file=sys.stderr)
        return []

    if suffix == ".pdf" and convert_pdfs:
        return convert_pdf(
            path,
            pdf_tmp_root,
            page_range=page_range,
            render_dpi=render_dpi,
            run_command=run_command,
        )

    return [BoardInput(path=path, source_path=path, original_path=path)]


def convert_pdf(
    path: Path,
    pdf_tmp_root: Path,
    *,
    page_range: Sequence[int] | None = None,
    render_dpi: int = 150,
    run_command: Callable[..., object] = subprocess.run,
) -> list[BoardInput]:
    """Convert a PDF to temporary PNG pages using pdftoppm."""
    pdf_key = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    out_dir = pdf_tmp_root / f"{path.stem}_{pdf_key}"
    out_prefix = out_dir / "page"
    out_dir.mkdir(parents=True, exist_ok=True)
    before = set(out_dir.glob("page-*.png"))

    for start, end in page_ranges(page_range):
        command = ["pdftoppm", "-r", str(render_dpi), "-png"]
        if start is not None and end is not None:
            command.extend(["-f", str(start), "-l", str(end)])
        command.extend([str(path), str(out_prefix)])
        try:
            run_command(command, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError(
                "pdftoppm not found. Install poppler first, e.g. `brew install poppler`."
            ) from None
        except subprocess.CalledProcessError as exc:
            msg = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(f"pdftoppm failed for {path}: {msg}") from exc

    converted = sorted(out_dir.glob("page-*.png"), key=lambda p: page_number_from_path(p) or 0)
    if page_range:
        wanted = set(page_range)
        converted = [p for p in converted if (page_number_from_path(p) in wanted and p not in before)]
    if not converted:
        print(f"  WARN skip (pdf produced no pages): {path}", file=sys.stderr)
    return [
        BoardInput(path=page_path, source_path=path, page=page_number_from_path(page_path), original_path=path)
        for page_path in converted
    ]


def prepare_inputs(
    items: Sequence[BoardInput],
    tmp_root: Path,
    *,
    max_side: int | None = None,
    run_command: Callable[..., object] = subprocess.run,
) -> list[BoardInput]:
    if not max_side:
        return list(items)
    prepared: list[BoardInput] = []
    tmp_root.mkdir(parents=True, exist_ok=True)
    for item in items:
        if item.path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            prepared.append(item)
            continue
        out_path = tmp_root / f"{item.safe_id}{item.path.suffix.lower()}"
        try:
            run_command(["sips", "-Z", str(max_side), str(item.path), "--out", str(out_path)], check=True)
        except FileNotFoundError:
            raise RuntimeError("sips not found. On macOS it should be available by default.") from None
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"sips failed for {item.path}: {exc}") from exc
        prepared.append(replace(item, path=out_path))
    return prepared


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="板书图片/PDF -> 题面 .md")
    parser.add_argument("images", nargs="+", help="板书图片/PDF/目录路径（支持 glob）")
    parser.add_argument("-o", "--output", default="board2md-out.md", help="输出 .md 路径")
    parser.add_argument(
        "--profile",
        choices=["custom", "quick", "verified", "lecture"],
        default="custom",
        help="预设流程：quick/verified/lecture",
    )
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--timeout", type=int, default=300, help="单次 API 请求超时秒数")
    parser.add_argument("--dry-run", action="store_true", help="只展开并列出输入文件，不调用 API 或 PDF 转换")

    parser.add_argument("--page-range", help="PDF 页码范围，1-based，例如 1,3-5,9")
    parser.add_argument("--render-dpi", type=int, default=150, help="PDF 转图 DPI，默认 150")
    parser.add_argument("--mode", choices=sorted(PROMPTS), default="problem", help="内置 prompt 模式")
    parser.add_argument("--prompt-file", help="使用自定义 prompt 文件覆盖 --mode")
    parser.add_argument(
        "--note-type",
        choices=["auto", "whiteboard", "paper", "generic"],
        default="generic",
        help="按图片类型调整 prompt，whiteboard 会强化空间布局理解",
    )
    parser.add_argument("--debug-dir", help="保存每页图片、原始模型输出和 meta.json")
    parser.add_argument("--obsidian", action="store_true", help="输出 Obsidian source callout")
    parser.add_argument("--atomic-dir", help="把每页结果另存为独立 Obsidian/Atomic Markdown 文件")
    parser.add_argument(
        "--cross-validate",
        choices=["off", "fast", "balanced", "strict"],
        default="off",
        help="交叉验证强度；balanced 为双路独立识别 + judge 合并",
    )
    parser.add_argument(
        "--lecture-synthesis",
        choices=["off", "teaching"],
        default="off",
        help="把零碎板书片段合成为正式讲义",
    )
    parser.add_argument("--lecture-audit", help="讲义合成审校记录输出路径")
    parser.add_argument("--verification-report", help="交叉验证 JSON 报告输出路径")
    parser.add_argument("--quality-report", help="质量检查 JSON 输出路径")
    parser.add_argument("--fail-on-quality", action="store_true", help="质量检查发现问题时返回非零退出码")
    parser.add_argument(
        "--output-format",
        choices=["markdown", "json", "both"],
        default="markdown",
        help="输出 Markdown、JSON 或两者",
    )
    parser.add_argument("--chunks-output", help="把输出按标题切成 chunks JSON")
    parser.add_argument("--run-report", help="写入运行摘要 JSON")
    parser.add_argument(
        "--bundle-dir",
        help="同时输出统一 MathMD 内容包（problems/index/manifest/quality/chunks）",
    )
    parser.add_argument(
        "--bundle-fuse",
        choices=["never", "auto", "always"],
        default="never",
        help="对内容包中的相邻 Markdown 做语义融合；默认关闭以保持旧行为",
    )
    parser.add_argument(
        "--bundle-fuse-engine",
        choices=["semantic", "rules"],
        default="semantic",
        help="内容包融合引擎；semantic 使用本地 Qwen 判断逻辑关联",
    )
    parser.add_argument("--bundle-fuse-rounds", type=int, default=3, help="内容包最多融合轮数")
    parser.add_argument(
        "--bundle-fuse-threshold",
        type=float,
        default=0.72,
        help="语义融合关联阈值，默认 0.72",
    )

    parser.add_argument("--continue-on-error", action="store_true", help="单页失败时继续处理后续页面")
    parser.add_argument("--retries", type=int, default=3, help="每页 API 重试次数")
    parser.add_argument("--resume", action="store_true", help="配合 --debug-dir 复用已存在的 .raw.md")
    parser.add_argument(
        "--provider",
        choices=["qwen"],
        default=DEFAULT_PROVIDER,
        help="provider 标签，仅支持 qwen（本地 127.0.0.1:8085）且兼容 OpenAI-compatible chat completions",
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible base URL，默认读取 QWEN_BASE_URL / VIDEO2MD_BASE_URL 或本地默认值",
    )
    parser.add_argument(
        "--model",
        help=(
            "模型名或本地路径，默认读取 QWEN_MODEL / VIDEO2MD_MODEL，"
            "否则使用本地 Qwen3.8-27B-MLX-Mixed-4.95bpw"
        ),
    )
    parser.add_argument("--api-key-env", default="QWEN_API_KEY", help="API key 环境变量名")

    parser.add_argument("--recursive", action="store_true", help="目录输入时递归查找图片/PDF")
    parser.add_argument("--max-side", type=int, help="调用 sips 把发送给 API 的图片最长边缩到指定像素")
    parser.add_argument("--max-pages", type=int, help="安全阀：超过指定页/图数量则停止")
    parser.add_argument("--force", action="store_true", help="超过 --max-pages 时仍继续")
    parser.add_argument("--cache-dir", help="缓存单页普通识别 raw 输出，避免重复 API 调用")
    return parser


def resolve_api_config(args: argparse.Namespace, env: Mapping[str, str]) -> tuple[str, str, str]:
    provider = (args.provider or DEFAULT_PROVIDER).strip().lower()
    if provider != "qwen":
        raise RuntimeError("仅支持 provider=qwen；当前仅对接本地 Qwen 接口")

    base_url = (
        args.base_url
        or env.get("QWEN_BASE_URL")
        or env.get("VIDEO2MD_BASE_URL")
        or DEFAULT_BASE_URL
    )
    model = args.model or env.get("QWEN_MODEL") or env.get("VIDEO2MD_MODEL") or DEFAULT_MODEL

    key_envs = [args.api_key_env, "QWEN_API_KEY"]

    api_key = None
    for key_env in key_envs:
        if key_env:
            candidate = env.get(key_env)
            if candidate:
                api_key = candidate
                break

    return api_key or "", base_url, model


def apply_profile_defaults(args: argparse.Namespace) -> None:
    if args.profile == "quick":
        if args.cross_validate == "off":
            args.cross_validate = "off"
        if args.retries == 3:
            args.retries = 1
    elif args.profile == "verified":
        if args.cross_validate == "off":
            args.cross_validate = "balanced"
    elif args.profile == "lecture":
        if args.cross_validate == "off":
            args.cross_validate = "balanced"
        if args.lecture_synthesis == "off":
            args.lecture_synthesis = "teaching"
        if args.note_type == "generic":
            args.note_type = "whiteboard"


def cache_key_for_item(item: BoardInput, prompt: str, model: str, mode: str) -> str:
    stat = item.path.stat()
    payload = {
        "path": str(item.source_path.resolve()),
        "page": item.page,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
        "model": model,
        "mode": mode,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def read_cached_raw(cache_dir: Path | None, item: BoardInput, prompt: str, model: str, mode: str) -> str | None:
    if cache_dir is None:
        return None
    cache_path = cache_dir / f"{cache_key_for_item(item, prompt, model, mode)}.raw.md"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    return None


def write_cached_raw(cache_dir: Path | None, item: BoardInput, prompt: str, model: str, mode: str, raw: str) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key_for_item(item, prompt, model, mode)}.raw.md"
    cache_path.write_text(raw, encoding="utf-8")


def provenance_comment(item: BoardInput) -> str:
    page = "" if item.page is None else f" page={item.page}"
    return f"<!-- board2md: source={item.source_display}{page} image={item.path.name} -->"


def source_callout(item: BoardInput) -> str:
    lines = [
        "> [!source] Board2MD",
        f"> 来源: `{item.source_display}`",
    ]
    if item.page is not None:
        lines.append(f"> 页码: {item.page}")
    return "\n".join(lines)


def format_part(item: BoardInput, content: str, *, obsidian: bool = False) -> str:
    pieces = [provenance_comment(item), f"## {item.safe_id}"]
    if obsidian:
        pieces.append(source_callout(item))
    pieces.append(content)
    return "\n\n".join(pieces)


def debug_paths(debug_dir: Path, item: BoardInput) -> tuple[Path, Path, Path]:
    suffix = item.path.suffix.lower() or ".png"
    return (
        debug_dir / f"{item.safe_id}.raw.md",
        debug_dir / f"{item.safe_id}.meta.json",
        debug_dir / f"{item.safe_id}{suffix}",
    )


def write_debug(debug_dir: Path, item: BoardInput, raw: str, meta: dict[str, object]) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    raw_path, meta_path, image_path = debug_paths(debug_dir, item)
    raw_path.write_text(raw, encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if item.path.exists() and item.path.is_file():
        shutil.copy2(item.path, image_path)


def read_resume_raw(debug_dir: Path, item: BoardInput) -> str | None:
    raw_path, _meta_path, _image_path = debug_paths(debug_dir, item)
    if raw_path.exists():
        return raw_path.read_text(encoding="utf-8")
    return None


def write_atomic_note(atomic_dir: Path, item: BoardInput, content: str, *, obsidian: bool) -> None:
    atomic_dir.mkdir(parents=True, exist_ok=True)
    note = format_part(item, content, obsidian=obsidian)
    (atomic_dir / f"{item.safe_id}.md").write_text(note, encoding="utf-8")


def process_item(
    item: BoardInput,
    *,
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    args: argparse.Namespace,
    debug_dir: Path | None,
    cache_dir: Path | None,
    request_post: Callable[..., object],
) -> str:
    started = time.time()
    raw: str | None = None
    if args.resume and debug_dir:
        raw = read_resume_raw(debug_dir, item)
    if raw is None:
        raw = read_cached_raw(cache_dir, item, prompt, model, args.mode)
    if raw is None:
        raw = m3_vision(
            [item.path],
            prompt,
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            request_post=request_post,
        )
        write_cached_raw(cache_dir, item, prompt, model, args.mode, raw)
    elapsed = time.time() - started
    if debug_dir:
        write_debug(
            debug_dir,
            item,
            raw,
            {
                "source": item.source_display,
                "page": item.page,
                "image": str(item.path),
                "provider": args.provider,
                "base_url": base_url,
                "model": model,
                "elapsed_seconds": round(elapsed, 3),
                "retries": args.retries,
                "resumed": args.resume and raw is not None,
            },
        )
    return strip_think_block(raw)


def call_text_model(
    *,
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    args: argparse.Namespace,
    request_post: Callable[..., object],
) -> str:
    client = QwenClient(
        QwenConfig(
            base_url=base_url.rstrip("/"),
            model=model,
            api_key=api_key or None,
            timeout=args.timeout,
            retries=max(1, args.retries),
        ),
        request_post=request_post,
    )
    return client.complete(prompt, max_tokens=args.max_tokens)


def cross_validate_item(
    item: BoardInput,
    *,
    api_key: str,
    base_url: str,
    model: str,
    args: argparse.Namespace,
    debug_dir: Path | None,
    request_post: Callable[..., object],
) -> tuple[str, dict[str, object]]:
    pass_a = m3_vision(
        [item.path],
        CROSS_VALIDATE_PROMPTS["pass_a"],
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        request_post=request_post,
    )
    pass_b = m3_vision(
        [item.path],
        CROSS_VALIDATE_PROMPTS["pass_b"],
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        request_post=request_post,
    )
    judge_prompt = (
        f"{CROSS_VALIDATE_PROMPTS['judge']}\n\n"
        f"## 来源\n{item.source_display}\n\n"
        f"## 识别 A\n{strip_think_block(pass_a)}\n\n"
        f"## 识别 B\n{strip_think_block(pass_b)}\n"
    )
    judge_raw = call_text_model(
        prompt=judge_prompt,
        api_key=api_key,
        base_url=base_url,
        model=model,
        args=args,
        request_post=request_post,
    )
    judge = parse_json_response(judge_raw)
    content = ensure_text(judge.get("merged_markdown"), strip_think_block(pass_a))
    item_report = {
        "id": item.safe_id,
        "source": item.source_display,
        "page": item.page,
        "status": ensure_text(judge.get("status"), "partial"),
        "confidence": clamp_confidence(judge.get("confidence")),
        "conflicts": judge.get("conflicts") if isinstance(judge.get("conflicts"), list) else [],
        "audit_markdown": ensure_text(judge.get("audit_markdown")),
    }
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / f"{item.safe_id}.pass-a.md").write_text(strip_think_block(pass_a), encoding="utf-8")
        (debug_dir / f"{item.safe_id}.pass-b.md").write_text(strip_think_block(pass_b), encoding="utf-8")
        (debug_dir / f"{item.safe_id}.judge.json").write_text(
            json.dumps(judge, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_debug(
            debug_dir,
            item,
            content,
            {
                "source": item.source_display,
                "page": item.page,
                "image": str(item.path),
                "provider": args.provider,
                "base_url": base_url,
                "model": model,
                "cross_validate": args.cross_validate,
                "confidence": item_report["confidence"],
                "status": item_report["status"],
            },
        )
    return strip_think_block(content), item_report


def build_verification_report(item_reports: Sequence[dict[str, object]]) -> dict[str, object]:
    conflicts = sum(1 for item in item_reports if item.get("status") == "conflict")
    partial = sum(1 for item in item_reports if item.get("status") == "partial")
    confidences = [float(item.get("confidence", 0.0)) for item in item_reports]
    avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    return {
        "summary": {
            "items": len(item_reports),
            "conflicts": conflicts,
            "partial": partial,
            "average_confidence": avg_confidence,
        },
        "items": list(item_reports),
    }


def write_verification_report(
    report: dict[str, object],
    *,
    debug_dir: Path | None,
    report_path: str | None,
) -> None:
    targets: list[Path] = []
    if debug_dir:
        targets.append(debug_dir / "verification.json")
    if report_path:
        targets.append(Path(os.path.expanduser(report_path)))
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def synthesize_lecture(
    parts: Sequence[str],
    items: Sequence[BoardInput],
    *,
    api_key: str,
    base_url: str,
    model: str,
    args: argparse.Namespace,
    debug_dir: Path | None,
    request_post: Callable[..., object],
) -> tuple[str, str, dict[str, object]]:
    source_map = [
        {
            "id": item.safe_id,
            "source": item.source_display,
            "page": item.page,
        }
        for item in items
    ]
    joined_parts = "\n\n---\n\n".join(
        f"## Fragment {i + 1}: {items[i].safe_id}\n\n{part}"
        for i, part in enumerate(parts)
    )
    prompt = (
        f"{LECTURE_SYNTHESIS_PROMPT}\n\n"
        f"## Source Map JSON\n{json.dumps(source_map, ensure_ascii=False, indent=2)}\n\n"
        f"## Board Fragments\n{joined_parts}"
    )
    lecture = strip_think_block(
        call_text_model(
            prompt=prompt,
            api_key=api_key,
            base_url=base_url,
            model=model,
            args=args,
            request_post=request_post,
        )
    )
    plan = {
        "mode": args.lecture_synthesis,
        "items": source_map,
        "sections_detected": extract_markdown_headings(lecture),
    }
    audit = render_lecture_audit(source_map, parts, plan)
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "lecture_prompt.md").write_text(prompt, encoding="utf-8")
        (debug_dir / "lecture_plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return lecture, audit, plan


def extract_markdown_headings(markdown: str) -> list[str]:
    return [line.strip() for line in markdown.splitlines() if line.startswith("#")]


def render_lecture_audit(
    source_map: Sequence[dict[str, object]],
    parts: Sequence[str],
    plan: dict[str, object],
) -> str:
    lines = [
        "# Board2MD Lecture Audit",
        "",
        "## Source Map",
        "",
    ]
    for source in source_map:
        page = "" if source.get("page") is None else f", page {source.get('page')}"
        lines.append(f"- `{source.get('id')}`: {source.get('source')}{page}")
    lines.extend(["", "## Synthesis Notes", ""])
    lines.append("- 正文已按正式讲义要求生成；来源痕迹与不确定内容应保留在本审校记录中。")
    headings = plan.get("sections_detected") or []
    if headings:
        lines.append(f"- 讲义标题/小节: {', '.join(str(h) for h in headings)}")
    lines.extend(["", "## Raw Fragment Index", ""])
    for i, part in enumerate(parts, 1):
        preview = " ".join(part.split())[:160]
        lines.append(f"{i}. {preview}")
    return "\n".join(lines).rstrip() + "\n"


def build_quality_report(markdown: str) -> dict[str, object]:
    return shared_build_quality_report(markdown, forbidden_terms=FORBIDDEN_PROVENANCE_TERMS)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_document_json(
    markdown: str,
    items: Sequence[BoardInput],
    *,
    verification: dict[str, object] | None,
    quality: dict[str, object] | None,
    lecture_plan: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "markdown": markdown,
        "items": [
            {"id": item.safe_id, "source": item.source_display, "page": item.page, "image": str(item.path)}
            for item in items
        ],
        "verification": verification,
        "quality": quality,
        "lecture_plan": lecture_plan,
    }


def build_chunks(markdown: str) -> list[dict[str, object]]:
    return shared_build_chunks(markdown)


def build_run_report(
    *,
    args: argparse.Namespace,
    items: Sequence[BoardInput],
    failures: Sequence[tuple[BoardInput, Exception]],
    started: float,
    output_path: Path,
) -> dict[str, object]:
    multiplier = 3 if args.cross_validate in {"fast", "balanced", "strict"} else 1
    if args.lecture_synthesis != "off":
        multiplier += 1
    return {
        "items": len(items),
        "failures": len(failures),
        "elapsed_seconds": round(time.time() - started, 3),
        "output": str(output_path),
        "profile": args.profile,
        "cross_validate": args.cross_validate,
        "lecture_synthesis": args.lecture_synthesis,
        "estimated_model_calls": len(items) * multiplier,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    run_command: Callable[..., object] = subprocess.run,
    request_post: Callable[..., object] = requests.post,
    env: Mapping[str, str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    env = os.environ if env is None else env
    apply_profile_defaults(args)
    started = time.time()

    try:
        page_range = parse_page_range(args.page_range)
    except ValueError as exc:
        parser.error(str(exc))

    if args.resume and not args.debug_dir:
        print("ERROR: --resume requires --debug-dir", file=sys.stderr)
        return 1

    prompt = apply_note_type(load_prompt(args.mode, args.prompt_file), args.note_type)

    with tempfile.TemporaryDirectory(prefix="board2md_pdf_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        inputs = expand_inputs(
            args.images,
            tmp_root / "pdf",
            convert_pdfs=not args.dry_run,
            page_range=page_range,
            render_dpi=args.render_dpi,
            recursive=args.recursive,
            run_command=run_command,
        )

        if not inputs:
            print("ERROR: no valid images", file=sys.stderr)
            return 1

        if args.max_pages is not None and len(inputs) > args.max_pages and not args.force:
            print(
                f"ERROR: input count {len(inputs)} exceeds --max-pages {args.max_pages}; pass --force to continue",
                file=sys.stderr,
            )
            return 1

        if args.dry_run:
            print(f"Would process {len(inputs)} inputs:")
            if args.cross_validate != "off":
                print(f"  cross_validate={args.cross_validate}")
            if args.lecture_synthesis != "off":
                print(f"  lecture_synthesis={args.lecture_synthesis}")
            if args.profile != "custom":
                print(f"  profile={args.profile}")
            if args.bundle_dir:
                print(f"  bundle_dir={args.bundle_dir} fuse={args.bundle_fuse}")
            for item in inputs:
                page = "" if item.page is None else f" page={item.page}"
                print(f"  {item.path}{page}")
            return 0

        try:
            inputs = prepare_inputs(
                inputs,
                tmp_root / "preprocessed",
                max_side=args.max_side,
                run_command=run_command,
            )
            api_key, base_url, model = resolve_api_config(args, env)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        debug_dir = Path(os.path.expanduser(args.debug_dir)) if args.debug_dir else None
        atomic_dir = Path(os.path.expanduser(args.atomic_dir)) if args.atomic_dir else None
        cache_dir = Path(os.path.expanduser(args.cache_dir)) if args.cache_dir else None
        print(f"Processing {len(inputs)} images with {model}...")

        all_md_parts = []
        clean_contents = []
        processed_items = []
        item_reports = []
        failures = []
        verification_report = None
        lecture_plan = None
        for i, item in enumerate(inputs, 1):
            print(f"  [{i}/{len(inputs)}] {item.safe_id}")
            try:
                if args.cross_validate in {"fast", "balanced", "strict"}:
                    if args.cross_validate != "balanced":
                        print(
                            f"  WARN cross_validate={args.cross_validate} currently uses balanced pipeline",
                            file=sys.stderr,
                        )
                    content, report = cross_validate_item(
                        item,
                        api_key=api_key,
                        base_url=base_url,
                        model=model,
                        args=args,
                        debug_dir=debug_dir,
                        request_post=request_post,
                    )
                    item_reports.append(report)
                else:
                    content = process_item(
                        item,
                        prompt=prompt,
                        api_key=api_key,
                        base_url=base_url,
                        model=model,
                        args=args,
                        debug_dir=debug_dir,
                        cache_dir=cache_dir,
                        request_post=request_post,
                    )
            except Exception as exc:
                failures.append((item, exc))
                if not args.continue_on_error:
                    print(f"ERROR: {item.safe_id}: {exc}", file=sys.stderr)
                    return FATAL_PROCESSING_EXIT
                content = f"> [!failure] board2md error\n> {exc}"

            part = format_part(item, content, obsidian=args.obsidian)
            all_md_parts.append(part)
            clean_contents.append(content)
            processed_items.append(item)
            if atomic_dir:
                write_atomic_note(atomic_dir, item, content, obsidian=args.obsidian)

        if item_reports:
            verification_report = build_verification_report(item_reports)
            write_verification_report(
                verification_report,
                debug_dir=debug_dir,
                report_path=args.verification_report,
            )

        if args.lecture_synthesis == "teaching":
            output_md, lecture_audit, lecture_plan = synthesize_lecture(
                clean_contents,
                processed_items,
                api_key=api_key,
                base_url=base_url,
                model=model,
                args=args,
                debug_dir=debug_dir,
                request_post=request_post,
            )
            if args.lecture_audit:
                audit_path = Path(os.path.expanduser(args.lecture_audit))
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                audit_path.write_text(lecture_audit, encoding="utf-8")
        else:
            output_md = "\n\n---\n\n".join(all_md_parts)

        out_path = Path(os.path.expanduser(args.output))
        quality_report = build_quality_report(output_md)
        if args.quality_report:
            write_json(Path(os.path.expanduser(args.quality_report)), quality_report)
        document_json = build_document_json(
            output_md,
            processed_items,
            verification=verification_report,
            quality=quality_report,
            lecture_plan=lecture_plan,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.output_format in {"markdown", "both"}:
            out_path.write_text(output_md, encoding="utf-8")
        if args.output_format in {"json", "both"}:
            json_path = out_path if args.output_format == "json" else out_path.with_suffix(".json")
            write_json(json_path, document_json)
        if args.chunks_output:
            write_json(Path(os.path.expanduser(args.chunks_output)), build_chunks(output_md))
        if args.run_report:
            write_json(
                Path(os.path.expanduser(args.run_report)),
                build_run_report(
                    args=args,
                    items=processed_items,
                    failures=failures,
                    started=started,
                    output_path=out_path,
                ),
            )
        if args.bundle_dir:
            bundle_dir = Path(os.path.expanduser(args.bundle_dir))
            bundle_documents = [
                ProblemDocument(
                    title=extract_title(content, item.safe_id),
                    markdown=content,
                    source=item.source_display,
                    source_kind="board",
                    source_id=item.safe_id,
                    page=item.page,
                    assets=(BundleAsset(path=item.path, label="板书或讲义原图"),),
                )
                for content, item in zip(clean_contents, processed_items)
            ]
            write_problem_bundle(
                bundle_dir,
                bundle_documents,
                generator="board2md",
                model=model,
            )
            if args.bundle_fuse != "never":
                fusion_client = QwenClient(
                    QwenConfig(
                        base_url=base_url.rstrip("/"),
                        model=model,
                        api_key=api_key or None,
                        timeout=args.timeout,
                        retries=max(1, args.retries),
                    ),
                    request_post=request_post,
                )
                complete = None
                if args.bundle_fuse_engine == "semantic":
                    complete = lambda text, limit: fusion_client.complete(text, max_tokens=limit)
                fuse_problem_directory(
                    bundle_dir,
                    complete=complete,
                    mode=args.bundle_fuse,
                    engine=args.bundle_fuse_engine,
                    rounds=args.bundle_fuse_rounds,
                    threshold=args.bundle_fuse_threshold,
                    generator="board2md",
                    model=model,
                )
            print(f"MathMD bundle written to {bundle_dir}")
        print(f"Written to {out_path} ({len(output_md)} bytes)")
        if args.fail_on_quality and quality_report["summary"]["issue_count"]:
            print(
                f"WARN: quality check found {quality_report['summary']['issue_count']} issue(s)",
                file=sys.stderr,
            )
            return QUALITY_FAILURE_EXIT
        if failures:
            print(f"WARN: {len(failures)} item(s) failed", file=sys.stderr)
            return PARTIAL_FAILURE_EXIT
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
