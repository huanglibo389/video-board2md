from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


MATH2MD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = MATH2MD_ROOT.parent
for path in (str(MATH2MD_ROOT), str(TOOLS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from math2md_core import (  # noqa: E402
    RoutedInputs,
    build_pipeline_commands,
    expand_and_route_inputs,
    main,
)
from mathmd_shared import ProblemDocument, write_problem_bundle  # noqa: E402


class Result:
    returncode = 0


class Math2MdCoreTests(unittest.TestCase):
    def test_expand_and_route_excludes_partial_and_split_video_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("lecture.mp4", "board.png", "notes.pdf", "lecture.f137.mp4", "next.webm.part", "x.txt"):
                (root / name).write_bytes(b"x")
            routed = expand_and_route_inputs([root])
            self.assertEqual([path.name for path in routed.video], ["lecture.mp4"])
            self.assertEqual([path.name for path in routed.board], ["board.png", "notes.pdf"])
            self.assertEqual({path.name for path in routed.ignored}, {"lecture.f137.mp4", "next.webm.part", "x.txt"})

    def test_commands_route_board_and_video_through_shared_semantic_settings(self):
        routed = RoutedInputs(
            board=(Path("/tmp/board.png"), Path("/tmp/notes.pdf")),
            video=(Path("/tmp/lesson.mp4"),),
        )
        commands = build_pipeline_commands(
            routed,
            out_root=Path("/tmp/out"),
            profile="verified",
            interval=60,
            fuse_mode="auto",
            fuse_engine="semantic",
            fuse_rounds=3,
            fusion_threshold=0.72,
            base_url="http://127.0.0.1:8085/v1",
            model="qwen-test",
            continue_on_error=False,
        )
        self.assertEqual([command.kind for command in commands], ["board", "video"])
        board = list(commands[0].argv)
        video = list(commands[1].argv)
        self.assertEqual(Path(board[1]).name, "board2md")
        self.assertEqual(Path(video[1]).name, "video2md")
        self.assertEqual(board[board.index("--bundle-fuse-engine") + 1], "semantic")
        self.assertEqual(video[video.index("--fuse-engine") + 1], "semantic")
        self.assertEqual(video[video.index("--interval") + 1], "60")
        self.assertIn("qwen-test", board)
        self.assertIn("qwen-test", video)

    def test_dry_run_does_not_invoke_subprocess(self):
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "lesson.mp4"
            video.write_bytes(b"video")

            def forbidden_runner(_cmd):
                raise AssertionError("dry-run must not invoke subprocess")

            code = main([str(video), "--out-root", str(Path(td) / "out"), "--dry-run"], run_command=forbidden_runner)
            self.assertEqual(code, 0)

    def test_success_exit_without_bundle_is_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / "board.png"
            image.write_bytes(b"image")
            out = root / "out"
            code = main([str(image), "--out-root", str(out)], run_command=lambda _cmd: Result())
            self.assertEqual(code, 2)
            report = json.loads((out / "run_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failures"][0]["error"], "output bundle was not created")

    def test_main_aligns_and_fuses_matching_board_and_video_bundles(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            board = root / "board.png"
            video = root / "lesson.mp4"
            board.write_bytes(b"board")
            video.write_bytes(b"video")
            out = root / "out"

            def fake_runner(cmd):
                if "--bundle-dir" in cmd:
                    bundle = Path(cmd[cmd.index("--bundle-dir") + 1])
                    source_kind = "board"
                else:
                    bundle = Path(cmd[cmd.index("--out") + 1])
                    source_kind = "video"
                write_problem_bundle(
                    bundle,
                    [
                        ProblemDocument(
                            title="共同题",
                            markdown="# 共同题\n\n完整数学信息 $x=1$。",
                            source=source_kind,
                            source_kind=source_kind,
                            source_id=source_kind,
                        )
                    ],
                    generator=source_kind,
                    model="qwen-test",
                )
                return Result()

            code = main(
                [
                    str(board),
                    str(video),
                    "--out-root",
                    str(out),
                    "--model",
                    "qwen-test",
                    "--fuse-engine",
                    "rules",
                ],
                run_command=fake_runner,
            )
            self.assertEqual(code, 0)
            self.assertEqual(len(list((out / "course" / "problems").glob("p*.md"))), 1)
            self.assertTrue((out / "course" / "mathmd_manifest.json").is_file())
            self.assertTrue((out / "course" / "alignment_manifest.json").is_file())
            self.assertTrue((out / "course" / "fusion_manifest.json").is_file())
            report = json.loads((out / "run_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete")
            self.assertEqual(len(report["bundles"]), 2)
            self.assertEqual(len(report["alignment"]["matches"]), 1)
            self.assertEqual(report["fusion"]["removed"], 1)


if __name__ == "__main__":
    unittest.main()
