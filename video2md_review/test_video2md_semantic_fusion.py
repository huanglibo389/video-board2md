from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parent / "video2md"
SPEC = importlib.util.spec_from_file_location("video2md_semantic_test_module", SCRIPT)
if SPEC is None or SPEC.loader is None:
    from importlib.machinery import SourceFileLoader

    SPEC = importlib.util.spec_from_loader(
        "video2md_semantic_test_module",
        SourceFileLoader("video2md_semantic_test_module", str(SCRIPT)),
    )
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class Video2MdSemanticFusionTests(unittest.TestCase):
    def test_frame_extraction_uses_ffmpeg8_jpeg_format_and_first_frame_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "short.mp4"
            video.write_bytes(b"video")
            commands = []

            def fake_run(cmd, **_kwargs):
                commands.append(cmd)
                if "-frames:v" in cmd:
                    Path(cmd[-1]).write_bytes(b"jpg")
                return SimpleNamespace(returncode=0, stderr="")

            original_run = MODULE.run
            MODULE.run = fake_run
            try:
                frames = MODULE.step_extract_frames(video, root / "out", interval_s=60)
            finally:
                MODULE.run = original_run
            self.assertEqual(len(frames), 1)
            self.assertIn("format=yuvj420p", commands[0][commands[0].index("-vf") + 1])
            self.assertIn("-frames:v", commands[1])
            self.assertIn("format=yuvj420p", commands[2][commands[2].index("-vf") + 1])

    def test_video_adapter_uses_shared_semantic_fusion_and_writes_bundle_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            problems = out / "problems"
            problems.mkdir()
            (problems / "p001_同题.md").write_text(
                "# 同题\n\n**来源**: 视频 [t=0s-60s]\n\n## 题面\n\n证明 $a=b$。\n",
                encoding="utf-8",
            )
            (problems / "p002_续证.md").write_text(
                "# 同题续证\n\n**来源**: 视频 [t=60s-120s]\n\n## 解答\n\n由 $b=c$ 得证。\n",
                encoding="utf-8",
            )

            def complete(prompt: str, _max_tokens: int) -> str:
                if "结构编辑" in prompt:
                    return '{"related": true, "score": 0.95, "reason": "连续证明"}'
                if "无损融合审校员" in prompt:
                    return '{"complete": true, "missing": [], "revised_markdown": ""}'
                return "# 同题\n\n## 题面\n\n证明 $a=b$。\n\n## 解答\n\n由 $b=c$ 得证。"

            result = MODULE.fuse_problem_markdowns(
                out,
                mode="auto",
                rounds=3,
                engine="semantic",
                complete=complete,
                threshold=0.72,
            )
            self.assertEqual(result["removed"], 1)
            self.assertEqual(result["files"], 1)
            self.assertEqual(result["engine"], "semantic")
            self.assertTrue((out / "mathmd_manifest.json").is_file())
            self.assertTrue((out / "quality_report.json").is_file())
            self.assertTrue((out / "chunks.json").is_file())
            content = next(problems.glob("p*.md")).read_text(encoding="utf-8")
            self.assertIn("$a=b$", content)
            self.assertIn("$b=c$", content)


if __name__ == "__main__":
    unittest.main()
