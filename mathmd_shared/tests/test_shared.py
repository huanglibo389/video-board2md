from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from mathmd_shared import (  # noqa: E402
    BundleAsset,
    ProblemDocument,
    QwenClient,
    QwenConfig,
    align_cross_source_problems,
    fuse_problem_directory,
    merge_problem_bundles,
    refresh_bundle_metadata,
    write_problem_bundle,
)


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, content: str):
        self.content = content

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


class MathMdSharedTests(unittest.TestCase):
    def test_qwen_client_supports_keyless_text_and_real_image_payload(self):
        with tempfile.TemporaryDirectory() as td:
            image = Path(td) / "diagram.png"
            image.write_bytes(b"png-bytes")
            calls = []

            def fake_post(**kwargs):
                calls.append(kwargs)
                return FakeResponse("# 识别结果")

            client = QwenClient(
                QwenConfig(base_url="http://127.0.0.1:8085/v1", model="qwen-test", api_key=None),
                request_post=fake_post,
            )
            self.assertEqual(client.complete("hello"), "# 识别结果")
            self.assertEqual(client.vision([image], "read"), "# 识别结果")
            self.assertNotIn("Authorization", calls[0]["headers"])
            content = calls[1]["json"]["messages"][0]["content"]
            self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))
            self.assertEqual(calls[1]["json"]["model"], "qwen-test")

    def test_problem_bundle_preserves_assets_and_writes_canonical_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / "board.jpg"
            image.write_bytes(b"image")
            bundle = root / "bundle"
            manifest = write_problem_bundle(
                bundle,
                [
                    ProblemDocument(
                        title="几何题",
                        markdown="# 几何题\n\n## 题面\n\n证明 $AB=AC$。\n\n## 解答\n\n作辅助线。",
                        source=str(image),
                        source_kind="board",
                        source_id="board-1",
                        assets=(BundleAsset(image, "原始几何图"),),
                    )
                ],
                generator="test",
                model="qwen-test",
            )
            problem = next((bundle / "problems").glob("p*.md"))
            text = problem.read_text(encoding="utf-8")
            self.assertIn("## 参考图像", text)
            self.assertIn("![原始几何图](assets/", text)
            self.assertTrue(next((bundle / "problems" / "assets").iterdir()).is_file())
            self.assertEqual(manifest["schema"], "mathmd-bundle")
            self.assertTrue((bundle / "index.md").is_file())
            self.assertTrue((bundle / "quality_report.json").is_file())
            self.assertTrue((bundle / "chunks.json").is_file())

    def test_refresh_adds_machine_readable_source_to_legacy_video_markdown(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = Path(td)
            problems = bundle / "problems"
            problems.mkdir()
            problem = problems / "p001_legacy.md"
            problem.write_text(
                "# 视频题\n\n**来源**: 视频 [t=0s-60s]\n\n内容 $a=b$。\n",
                encoding="utf-8",
            )
            refresh_bundle_metadata(
                bundle,
                generator="video2md",
                model="qwen-test",
                extra={"source_kind": "video", "source": "/tmp/lesson with spaces.mp4"},
            )
            content = problem.read_text(encoding="utf-8")
            self.assertTrue(content.startswith("<!-- mathmd:"))
            manifest = json.loads((bundle / "mathmd_manifest.json").read_text(encoding="utf-8"))
            source = manifest["documents"][0]["sources"][0]
            self.assertEqual(source["kind"], "video")
            self.assertEqual(source["source"], "/tmp/lesson with spaces.mp4")

    def test_semantic_fusion_merges_only_related_neighbors_without_losing_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = Path(td) / "bundle"
            asset = Path(td) / "figure.png"
            asset.write_bytes(b"figure")
            write_problem_bundle(
                bundle,
                [
                    ProblemDocument(
                        title="同一题",
                        markdown="# 同一题\n\n## 题面\n\n求证 $a+b=c$。",
                        source="video-a",
                        source_kind="video",
                        source_id="a",
                        assets=(BundleAsset(asset, "图 A"),),
                    ),
                    ProblemDocument(
                        title="同一题续证",
                        markdown="# 同一题续证\n\n## 解答\n\n由引理得到 $c=d$。",
                        source="board-b",
                        source_kind="board",
                        source_id="b",
                    ),
                    ProblemDocument(
                        title="独立例题",
                        markdown="# 独立例题\n\n证明 $x>0$。",
                        source="video-c",
                        source_kind="video",
                        source_id="c",
                    ),
                ],
                generator="test",
                model="qwen-test",
            )

            budgets = []

            def complete(prompt: str, max_tokens: int) -> str:
                budgets.append((prompt, max_tokens))
                if "只返回 JSON" in prompt:
                    if "# 独立例题" in prompt:
                        return '{"related": false, "score": 0.08, "reason": "独立题"}'
                    return '{"related": true, "score": 0.96, "reason": "同题续证"}'
                return (
                    "# 同一题\n\n## 题面\n\n求证 $a+b=c$。\n\n"
                    "## 解答\n\n由引理得到 $c=d$，结合题设完成证明。\n\n"
                    "## 说明\n\n以上保留题设、推导与结论，且两份来源共同构成完整证明。"
                )

            result = fuse_problem_directory(
                bundle,
                complete=complete,
                mode="auto",
                engine="semantic",
                rounds=3,
                threshold=0.72,
                generator="test",
                model="qwen-test",
            )
            files = sorted((bundle / "problems").glob("p*.md"))
            merged = files[0].read_text(encoding="utf-8")
            self.assertEqual(result["removed"], 1)
            self.assertEqual(len(files), 2)
            self.assertIn("$a+b=c$", merged)
            self.assertIn("$c=d$", merged)
            self.assertIn("![图 A](assets/", merged)
            self.assertEqual(len(list((bundle / "problems_raw_segments").glob("p*.md"))), 3)
            self.assertEqual(len(result["rounds"]), 2)
            audit_budgets = [limit for prompt, limit in budgets if "无损融合审校员" in prompt]
            self.assertTrue(audit_budgets)
            self.assertTrue(all(limit <= 2000 for limit in audit_budgets))

    def test_semantic_fusion_falls_back_when_audit_finds_unsupported_additions(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = Path(td) / "bundle"
            write_problem_bundle(
                bundle,
                [
                    ProblemDocument(
                        title="同一题题面",
                        markdown="# 同一题题面\n\n求证 $a=b$。",
                        source="board-a",
                        source_kind="board",
                        source_id="a",
                    ),
                    ProblemDocument(
                        title="同一题解答",
                        markdown="# 同一题解答\n\n由条件可得 $a=b$。",
                        source="video-b",
                        source_kind="video",
                        source_id="b",
                    ),
                ],
                generator="test",
                model="qwen-test",
            )

            def complete(prompt: str, _max_tokens: int) -> str:
                if "结构编辑" in prompt:
                    return '{"related": true, "score": 0.98, "reason": "同一题"}'
                if "无损融合审校员" in prompt:
                    return (
                        '{"complete": false, "missing": [], '
                        '"unsupported": ["新增错误事实"]}'
                    )
                return "# 同一题\n\n求证 $a=b$。\n\n新增错误事实。"

            result = fuse_problem_directory(
                bundle,
                complete=complete,
                mode="auto",
                engine="semantic",
                rounds=1,
                generator="test",
                model="qwen-test",
            )
            merged = next((bundle / "problems").glob("p*.md")).read_text(encoding="utf-8")
            self.assertEqual(result["removed"], 1)
            self.assertIn("## 第一片段", merged)
            self.assertIn("## 后续片段", merged)
            self.assertNotIn("新增错误事实", merged)

    def test_merge_problem_bundles_relinks_copied_assets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundles = []
            for number in (1, 2):
                image = root / f"image-{number}.jpg"
                image.write_bytes(str(number).encode())
                bundle = root / f"bundle-{number}"
                write_problem_bundle(
                    bundle,
                    [
                        ProblemDocument(
                            title=f"题 {number}",
                            markdown=f"# 题 {number}\n\n内容 $x={number}$。",
                            source=str(image),
                            source_kind="board" if number == 1 else "video",
                            source_id=str(number),
                            assets=(BundleAsset(image),),
                        )
                    ],
                    generator="test",
                    model="qwen-test",
                )
                bundles.append(bundle)
            course = root / "course"
            merge_problem_bundles(bundles, course, model="qwen-test")
            self.assertEqual(len(list((course / "problems").glob("p*.md"))), 2)
            self.assertEqual(len(list((course / "problems" / "assets").iterdir())), 2)
            for problem in (course / "problems").glob("p*.md"):
                targets = [part.split(")", 1)[0] for part in problem.read_text().split("](assets/")[1:]]
                for target in targets:
                    self.assertTrue((course / "problems" / "assets" / target).exists())
            data = json.loads((course / "mathmd_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data["documents"]), 2)

    def test_cross_source_alignment_places_matching_board_before_video(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = Path(td) / "course"
            write_problem_bundle(
                bundle,
                [
                    ProblemDocument(
                        "勾股定理",
                        "# 勾股定理\n\n## 题面\n\n证明 $a^2+b^2=c^2$。",
                        "board-1",
                        "board",
                        "b1",
                    ),
                    ProblemDocument("独立组合题", "# 独立组合题\n\n计算染色数。", "board-2", "board", "b2"),
                    ProblemDocument(
                        "勾股定理视频讲解",
                        "# 勾股定理视频讲解\n\n## 解答\n\n利用相似三角形证明 $a^2+b^2=c^2$。",
                        "video-1",
                        "video",
                        "v1",
                    ),
                    ProblemDocument("独立数论题", "# 独立数论题\n\n求素数。", "video-2", "video", "v2"),
                ],
                generator="test",
                model="qwen-test",
            )

            def complete(prompt: str, _max_tokens: int) -> str:
                if "勾股定理视频讲解" in prompt and "# 勾股定理" in prompt:
                    return '{"same_unit": true, "score": 0.94, "reason": "同一命题"}'
                return '{"same_unit": false, "score": 0.1, "reason": "独立内容"}'

            report = align_cross_source_problems(
                bundle,
                complete=complete,
                engine="semantic",
                threshold=0.72,
                generator="test",
                model="qwen-test",
            )
            files = sorted((bundle / "problems").glob("p*.md"))
            self.assertEqual(len(report["matches"]), 1)
            self.assertTrue(report["reordered"])
            self.assertIn("# 勾股定理\n", files[0].read_text(encoding="utf-8"))
            self.assertIn("# 勾股定理视频讲解\n", files[1].read_text(encoding="utf-8"))
            self.assertTrue((bundle / "alignment_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
