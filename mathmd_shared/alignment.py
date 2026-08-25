"""Cross-source semantic alignment for board/PDF and video problem units."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .bundle import list_problem_files, refresh_bundle_metadata
from .markdown import atomic_write_json, extract_title, strip_code_fence, strip_think_blocks


TextComplete = Callable[[str, int], str]


@dataclass(frozen=True)
class AlignmentCandidate:
    board: Path
    video: Path
    lexical_score: float


def _source_kind(markdown: str) -> str:
    for raw in re.findall(r"(?m)^<!-- mathmd:\s*(.*?)\s*-->$", markdown or ""):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("kind") in {"board", "video"}:
            return str(data["kind"])
    if re.search(r"\*\*来源\*\*\s*:\s*视频|视频\s*\[t=", markdown or ""):
        return "video"
    if re.search(r"\*\*来源\*\*\s*:\s*板书|板书或讲义原图", markdown or ""):
        return "board"
    return "unknown"


def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", (text or "").lower())
    return {compact[index : index + 2] for index in range(max(0, len(compact) - 1))}


def _formulas(text: str) -> set[str]:
    values = re.findall(r"\$\$.*?\$\$|(?<!\$)\$[^$\n]+\$(?!\$)", text or "", flags=re.DOTALL)
    return {
        re.sub(r"\s+", "", value).strip("$")
        for value in values
        if len(re.sub(r"\s+", "", value).strip("$")) >= 4
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _lexical_score(left: str, right: str) -> float:
    left_title = extract_title(left, "")
    right_title = extract_title(right, "")
    title_score = _jaccard(_bigrams(left_title), _bigrams(right_title))
    body_score = _jaccard(_bigrams(left[:1800]), _bigrams(right[:1800]))
    formula_score = _jaccard(_formulas(left), _formulas(right))
    exact_title = 1.0 if left_title and right_title and left_title == right_title else 0.0
    score = min(1.0, 0.45 * title_score + 0.2 * body_score + 0.25 * formula_score + 0.1 * exact_title)
    generic_titles = {"题目", "例题", "解答", "证明", "概念", "未命名"}
    if exact_title and left_title not in generic_titles:
        score = max(score, 0.82)
    return score


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
        raise ValueError("alignment result is not a JSON object")
    return data


def _semantic_match(board: str, video: str, complete: TextComplete) -> tuple[bool, float, str]:
    prompt = f"""你是数学课程资料对齐编辑。判断下面的板书/PDF 内容与视频内容是否讲解同一道题、同一命题或同一连续知识单元。

要求：
1. 题设、变量、目标、关键公式或证明结构实质一致，才可配对。
2. 仅主题或方法相似不足以配对；两道独立例题必须判为 false。
3. 视频可以比板书多出讲解步骤，板书也可以只有题面或图形。

只返回 JSON：{{"same_unit": true/false, "score": 0.0到1.0, "reason": "一句话理由"}}

## 板书/PDF
{board[:3500]}

## 视频
{video[:3500]}
"""
    data = _extract_json(complete(prompt, 1200))
    value = data.get("same_unit", False)
    same = value is True or str(value).strip().lower() == "true"
    score = max(0.0, min(1.0, float(data.get("score", 0.0))))
    return same, score, str(data.get("reason", ""))


def _candidate_pairs(
    board_files: list[Path],
    video_files: list[Path],
    contents: dict[Path, str],
    *,
    candidates_per_board: int,
    minimum_lexical_score: float,
) -> list[AlignmentCandidate]:
    candidates: list[AlignmentCandidate] = []
    for board in board_files:
        ranked = sorted(
            (
                AlignmentCandidate(board, video, _lexical_score(contents[board], contents[video]))
                for video in video_files
            ),
            key=lambda candidate: candidate.lexical_score,
            reverse=True,
        )
        candidates.extend(
            candidate
            for candidate in ranked[: max(1, candidates_per_board)]
            if candidate.lexical_score >= minimum_lexical_score
        )
    return candidates


def _renumber_files(out_dir: Path, ordered: list[Path]) -> list[dict[str, str]]:
    problems_dir = Path(out_dir) / "problems"
    staged: list[tuple[Path, Path]] = []
    for index, path in enumerate(ordered, start=1):
        temporary = problems_dir / f".mathmd_align_{index:05d}_{path.name}"
        path.replace(temporary)
        staged.append((path, temporary))

    mapping: list[dict[str, str]] = []
    for number, (original, temporary) in enumerate(staged, start=1):
        slug = re.sub(r"^p\d+_", "", original.name)
        destination = problems_dir / f"p{number:03d}_{slug}"
        temporary.replace(destination)
        mapping.append({"from": original.name, "to": destination.name})
    return mapping


def align_cross_source_problems(
    out_dir: Path,
    *,
    complete: TextComplete | None = None,
    engine: str = "semantic",
    threshold: float = 0.72,
    candidates_per_board: int = 2,
    minimum_lexical_score: float = 0.06,
    generator: str = "math2md",
    model: str = "",
) -> dict[str, object]:
    out_dir = Path(out_dir)
    files = list_problem_files(out_dir)
    contents = {path: path.read_text(encoding="utf-8") for path in files}
    board_files = [path for path in files if _source_kind(contents[path]) == "board"]
    video_files = [path for path in files if _source_kind(contents[path]) == "video"]
    unknown_files = [path for path in files if path not in board_files and path not in video_files]

    if not board_files or not video_files:
        report = {
            "engine": engine,
            "matches": [],
            "decisions": [],
            "reordered": False,
            "reason": "both board and video sources are required",
        }
        atomic_write_json(out_dir / "alignment_manifest.json", report)
        return report
    if engine == "semantic" and complete is None:
        raise ValueError("semantic cross-source alignment requires a text completion callback")

    candidates = _candidate_pairs(
        board_files,
        video_files,
        contents,
        candidates_per_board=candidates_per_board,
        minimum_lexical_score=minimum_lexical_score,
    )
    decisions: list[dict[str, object]] = []
    accepted: list[tuple[float, float, Path, Path]] = []
    for candidate in candidates:
        try:
            if engine == "semantic":
                same, score, reason = _semantic_match(
                    contents[candidate.board],
                    contents[candidate.video],
                    complete,
                )
            else:
                score = candidate.lexical_score
                same = score >= threshold
                reason = "deterministic title/body/formula similarity"
        except Exception as exc:
            same, score, reason = False, 0.0, f"alignment error: {exc}"
        record = {
            "board": candidate.board.name,
            "video": candidate.video.name,
            "lexical_score": round(candidate.lexical_score, 4),
            "same_unit": bool(same and score >= threshold),
            "score": score,
            "reason": reason,
        }
        decisions.append(record)
        if record["same_unit"]:
            accepted.append((score, candidate.lexical_score, candidate.board, candidate.video))

    matched_boards: set[Path] = set()
    matched_videos: set[Path] = set()
    matches: list[dict[str, object]] = []
    board_to_video: dict[Path, Path] = {}
    for score, lexical_score, board, video in sorted(accepted, reverse=True, key=lambda item: (item[0], item[1])):
        if board in matched_boards or video in matched_videos:
            continue
        matched_boards.add(board)
        matched_videos.add(video)
        board_to_video[board] = video
        matches.append(
            {
                "board": board.name,
                "video": video.name,
                "score": score,
                "lexical_score": round(lexical_score, 4),
            }
        )

    ordered: list[Path] = []
    for board in board_files:
        ordered.append(board)
        if board in board_to_video:
            ordered.append(board_to_video[board])
    ordered.extend(video for video in video_files if video not in matched_videos)
    ordered.extend(unknown_files)

    mapping: list[dict[str, str]] = []
    if matches and ordered != files:
        mapping = _renumber_files(out_dir, ordered)
        refresh_bundle_metadata(
            out_dir,
            generator=generator,
            model=model,
            extra={"cross_source_matches": len(matches)},
        )
    report = {
        "engine": engine,
        "threshold": threshold,
        "matches": matches,
        "decisions": decisions,
        "reordered": bool(mapping),
        "order_mapping": mapping,
    }
    atomic_write_json(out_dir / "alignment_manifest.json", report)
    return report
