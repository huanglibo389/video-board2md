"""Markdown-only multi-round semantic fusion for MathMD problem bundles."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .bundle import list_problem_files, refresh_bundle_metadata
from .markdown import atomic_write_json, atomic_write_text, extract_title, strip_code_fence, strip_think_blocks


TextComplete = Callable[[str, int], str]
CONTINUATION_RE = re.compile(r"记号|事实|引理|情形|分析|继续|证明|构造|关系链|回到|推广|置换|分解")


@dataclass(frozen=True)
class FusionDecision:
    related: bool
    score: float
    reason: str


def _extract_json(text: str) -> dict[str, object]:
    cleaned = strip_code_fence(strip_think_blocks(text))
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("fusion decision is not a JSON object")
    return data


def _rule_decision(left: str, right: str) -> FusionDecision:
    left_title = extract_title(left, "")
    right_title = extract_title(right, "")
    normalize = lambda value: re.sub(r"\s+|第?\d+[题问讲节部分页]|继续|题目|讲解", "", value or "")
    a = normalize(left_title)
    b = normalize(right_title)
    if a and b and (a == b or a in b or b in a):
        return FusionDecision(True, 0.85, "标题高度相似")
    if CONTINUATION_RE.search(right_title):
        return FusionDecision(True, 0.72, "后一标题显示为连续推导")
    body = "\n".join(right.splitlines()[:12])
    if re.search(r"继续|接下来|因此|于是|回到原", body):
        return FusionDecision(True, 0.72, "后一片段显式承接前文")
    return FusionDecision(False, 0.2, "规则未发现明确承接")


def _semantic_decision(left: str, right: str, complete: TextComplete, threshold: float) -> FusionDecision:
    prompt = f"""你是数学讲义的结构编辑。只根据下面两份相邻 Markdown 判断它们是否属于同一道题、同一证明或同一连续概念讲解。

判断标准：
1. 共享题设、变量、目标或证明链，且后一份依赖前一份，才算强关联。
2. 仅仅主题相同、都属于代数/几何、或出现“证明”“情形”等泛词，不足以融合。
3. 若是两道独立例题，即使方法相同，也不得融合。

只返回 JSON：{{"related": true/false, "score": 0.0到1.0, "reason": "一句话理由"}}

## Markdown A
{left}

## Markdown B
{right}
"""
    data = _extract_json(complete(prompt, 1000))
    score = max(0.0, min(1.0, float(data.get("score", 0.0))))
    related_value = data.get("related", False)
    related = related_value is True or str(related_value).strip().lower() == "true"
    return FusionDecision(related and score >= threshold, score, str(data.get("reason", "")))


def _markdown_images(text: str) -> list[str]:
    return re.findall(r"(?m)^!\[[^\n]+\]\([^)]+\)\s*$", text or "")


def _source_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if (
            stripped.startswith("<!-- mathmd:")
            or stripped.startswith("**来源**")
            or stripped.startswith("- **来源**")
            or stripped.startswith("- 视频")
            or stripped.startswith("> 来源")
        ):
            lines.append(stripped)
    return lines


def _math_expressions(text: str) -> list[str]:
    expressions = re.findall(r"\$\$.*?\$\$|(?<!\$)\$[^$\n]+\$(?!\$)", text or "", flags=re.DOTALL)
    return list(dict.fromkeys(expression.strip() for expression in expressions if expression.strip()))


def _ensure_section(markdown: str, heading: str, lines: list[str]) -> str:
    missing = [line for line in dict.fromkeys(lines) if line and line not in markdown]
    if not missing:
        return markdown
    return markdown.rstrip() + f"\n\n## {heading}\n\n" + "\n\n".join(missing) + "\n"


def _fallback_merge(left: str, right: str, left_name: str, right_name: str) -> str:
    title = extract_title(left, Path(left_name).stem)
    return (
        f"# {title}\n\n"
        f"## 第一片段\n\n{left.strip()}\n\n"
        f"## 后续片段\n\n{right.strip()}\n\n"
        f"## 原始片段索引\n\n- {left_name}\n- {right_name}\n"
    )


def _semantic_merge(
    left: str,
    right: str,
    left_name: str,
    right_name: str,
    complete: TextComplete,
) -> str:
    prompt = f"""你是数学竞赛讲义编辑。把下面两份属于同一逻辑单元的 Markdown 融合为一份完整 Markdown。

硬性要求：
1. 删除语义重复的题面、解答和说明，但不得丢失任何条件、公式、分类、证明步骤、结论、来源时间或图片链接。
2. 若两片段存在前后推进关系，按数学逻辑重排为连贯解答，不要简单标成“片段1/片段2”。
3. 保留 Markdown 图片链接的原始相对路径。
4. 第一行必须是一个 # 标题；使用 $...$ 和 $$...$$；不要输出代码围栏或解释。
5. 不确定是否重复的内容宁可保留。
6. 只做重排、去重和衔接，不得引入 A/B 中没有出现的新定义、新公式、新数值、新结论或自行补写的解法。

## A ({left_name})
{left}

## B ({right_name})
{right}
"""
    merge_budget = min(6000, max(1200, int((len(left) + len(right)) * 1.5)))
    merged = strip_code_fence(strip_think_blocks(complete(prompt, merge_budget))).strip()
    if len(merged) < max(180, int(max(len(left), len(right)) * 0.55)):
        merged = _fallback_merge(left, right, left_name, right_name)
    if not re.search(r"(?m)^#\s+", merged):
        merged = f"# {extract_title(left, Path(left_name).stem)}\n\n{merged}"
    audit_prompt = f"""你是数学内容无损融合审校员。检查融合稿是否遗漏 A 或 B 中任何独立条件、分类、证明步骤或结论，并检查是否新增了 A/B 都没有的信息。公式、来源和图片另有确定性检查，不必重复报告。

只返回 JSON，不要代码块：
{{"complete": true/false, "missing": ["可直接补回讲义的遗漏原文或准确改写"], "unsupported": ["融合稿擅自新增的信息"]}}

## A
{left}

## B
{right}

## 融合稿
{merged}
"""
    try:
        audit_budget = min(2000, max(700, (len(left) + len(right)) // 2))
        audit = _extract_json(complete(audit_prompt, audit_budget))
        complete_value = audit.get("complete", False)
        is_complete = complete_value is True or str(complete_value).strip().lower() == "true"
        unsupported = audit.get("unsupported", [])
        if isinstance(unsupported, list) and unsupported:
            merged = _fallback_merge(left, right, left_name, right_name)
        elif not is_complete:
            missing = audit.get("missing", [])
            if isinstance(missing, list) and missing:
                merged = _ensure_section(merged, "无损审校补回", [f"- {item}" for item in missing])
    except Exception:
        # Deterministic checks below still preserve formulas, images, sources,
        # and original segment references if the audit response is malformed.
        pass
    if not re.search(r"(?m)^#\s+", merged):
        merged = f"# {extract_title(left, Path(left_name).stem)}\n\n{merged}"
    merged = _ensure_section(merged, "来源片段", _source_lines(left) + _source_lines(right))
    merged = _ensure_section(merged, "图像保留", _markdown_images(left) + _markdown_images(right))
    merged = _ensure_section(merged, "未合入的公式保留", _math_expressions(left) + _math_expressions(right))
    merged = _ensure_section(merged, "原始片段索引", [f"- {left_name}", f"- {right_name}"])
    return merged.rstrip() + "\n"


def _backup_originals(out_dir: Path, files: list[Path]) -> Path:
    backup_dir = Path(out_dir) / "problems_raw_segments"
    backup_dir.mkdir(parents=True, exist_ok=True)
    history_dir = backup_dir / "history"
    for path in files:
        destination = backup_dir / path.name
        if destination.exists() and destination.read_bytes() != path.read_bytes():
            history_dir.mkdir(parents=True, exist_ok=True)
            old_hash = hashlib.sha256(destination.read_bytes()).hexdigest()[:10]
            historical = history_dir / f"{destination.stem}_{old_hash}{destination.suffix}"
            if not historical.exists():
                shutil.copy2(destination, historical)
        shutil.copy2(path, destination)
    return backup_dir


def fuse_problem_directory(
    out_dir: Path,
    *,
    complete: TextComplete | None = None,
    mode: str = "auto",
    engine: str = "semantic",
    rounds: int = 3,
    threshold: float = 0.72,
    generator: str = "mathmd-shared",
    model: str = "",
) -> dict[str, object]:
    out_dir = Path(out_dir)
    original_files = list_problem_files(out_dir)
    round_reports: list[dict[str, object]] = []
    total_removed = 0

    if mode == "never" or rounds <= 0:
        refresh_bundle_metadata(out_dir, generator=generator, model=model)
        report = {"engine": engine, "mode": mode, "rounds": [], "removed": 0, "files": len(original_files)}
        atomic_write_json(out_dir / "fusion_manifest.json", report)
        return report
    _backup_originals(out_dir, original_files)
    if engine == "semantic" and complete is None:
        raise ValueError("semantic fusion requires a text completion callback")

    for round_number in range(1, max(1, rounds) + 1):
        files = list_problem_files(out_dir)
        decisions: list[dict[str, object]] = []
        removed = 0
        index = 0
        while index + 1 < len(files):
            left_path = files[index]
            right_path = files[index + 1]
            left = left_path.read_text(encoding="utf-8")
            right = right_path.read_text(encoding="utf-8")
            try:
                if mode == "always":
                    decision = FusionDecision(True, 1.0, "mode=always")
                elif engine == "semantic":
                    decision = _semantic_decision(left, right, complete, threshold)
                else:
                    decision = _rule_decision(left, right)
            except Exception as exc:
                decision = FusionDecision(False, 0.0, f"decision error: {exc}")

            record = {
                "left": left_path.name,
                "right": right_path.name,
                "related": decision.related,
                "score": decision.score,
                "reason": decision.reason,
            }
            if decision.related:
                if engine == "semantic":
                    try:
                        merged = _semantic_merge(
                            left,
                            right,
                            left_path.name,
                            right_path.name,
                            complete,
                        )
                    except Exception as exc:
                        record["related"] = False
                        record["reason"] = f"merge error: {exc}"
                        decisions.append(record)
                        index += 1
                        continue
                else:
                    merged = _fallback_merge(left, right, left_path.name, right_path.name)
                record["before_sha256"] = [
                    hashlib.sha256(left.encode("utf-8")).hexdigest(),
                    hashlib.sha256(right.encode("utf-8")).hexdigest(),
                ]
                record["after_sha256"] = hashlib.sha256(merged.encode("utf-8")).hexdigest()
                atomic_write_text(left_path, merged)
                right_path.unlink()
                removed += 1
                total_removed += 1
                index += 2
            else:
                index += 1
            decisions.append(record)

        round_report = {
            "round": round_number,
            "removed": removed,
            "decisions": decisions,
            "remaining": len(list_problem_files(out_dir)),
        }
        round_reports.append(round_report)
        if removed == 0:
            break

    refresh_bundle_metadata(
        out_dir,
        generator=generator,
        model=model,
        extra={"fusion_engine": engine, "fusion_rounds": len(round_reports)},
    )
    report = {
        "engine": engine,
        "mode": mode,
        "threshold": threshold,
        "rounds": round_reports,
        "removed": total_removed,
        "files": len(list_problem_files(out_dir)),
    }
    atomic_write_json(out_dir / "fusion_manifest.json", report)
    return report
