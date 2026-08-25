import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import board2md_core


class FakeResponse:
    def __init__(self, text="# 题面\n\n内容", status_code=200):
        self.status_code = status_code
        self.text = text
        self._text = text

    def json(self):
        return {"choices": [{"message": {"content": self._text}}]}


class Board2MdCoreTests(unittest.TestCase):
    def test_expand_inputs_dry_run_keeps_pdf_without_converting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "lesson.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")

            def fail_if_called(*_args, **_kwargs):
                raise AssertionError("pdftoppm should not run during dry-run expansion")

            items = board2md_core.expand_inputs(
                [str(pdf)],
                root / "pdf_tmp",
                convert_pdfs=False,
                run_command=fail_if_called,
            )

            self.assertEqual([item.path for item in items], [pdf])
            self.assertEqual(items[0].source_path, pdf)
            self.assertIsNone(items[0].page)

    def test_expand_inputs_converts_selected_pdf_pages_with_render_dpi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "lesson.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            calls = []

            def fake_run(command, **_kwargs):
                calls.append(command)
                out_prefix = Path(command[-1])
                out_prefix.parent.mkdir(parents=True, exist_ok=True)
                if "-f" in command:
                    start = int(command[command.index("-f") + 1])
                    end = int(command[command.index("-l") + 1])
                else:
                    start = 1
                    end = 2
                for page in range(start, end + 1):
                    (out_prefix.parent / f"page-{page:02d}.png").write_bytes(b"png")

            items = board2md_core.expand_inputs(
                [str(pdf)],
                root / "pdf_tmp",
                convert_pdfs=True,
                page_range=[1, 3, 4],
                render_dpi=200,
                run_command=fake_run,
            )

            self.assertEqual([item.page for item in items], [1, 3, 4])
            self.assertEqual([command[2] for command in calls], ["200", "200"])
            self.assertIn("-f", calls[0])
            self.assertIn("-l", calls[1])

    def test_parse_page_range_accepts_commas_and_ranges(self):
        self.assertEqual(board2md_core.parse_page_range("3,1-2,2,5"), [1, 2, 3, 5])
        with self.assertRaises(ValueError):
            board2md_core.parse_page_range("3-1")

    def test_main_uses_prompt_file_and_cli_provider_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "board.png"
            img.write_bytes(b"png")
            prompt = root / "prompt.md"
            prompt.write_text("CUSTOM PROMPT", encoding="utf-8")
            out = root / "out.md"
            seen = {}

            def fake_post(url, headers, json, timeout):
                seen["url"] = url
                seen["auth"] = headers["Authorization"]
                seen["prompt"] = json["messages"][0]["content"][-1]["text"]
                seen["model"] = json["model"]
                seen["timeout"] = timeout
                return FakeResponse("# 自定义\n\n题面")

            env = {"MY_KEY": "secret"}
            exit_code = board2md_core.main(
                [
                    str(img),
                    "-o",
                    str(out),
                    "--prompt-file",
                    str(prompt),
                    "--base-url",
                    "https://example.test/v1",
                    "--model",
                    "vision-test",
                    "--api-key-env",
                    "MY_KEY",
                ],
                request_post=fake_post,
                env=env,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(seen["url"], "https://example.test/v1/chat/completions")
            self.assertEqual(seen["auth"], "Bearer secret")
            self.assertEqual(seen["prompt"], "CUSTOM PROMPT")
            self.assertEqual(seen["model"], "vision-test")
            self.assertIn("source=", out.read_text(encoding="utf-8"))

    def test_main_writes_debug_raw_meta_and_atomic_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "board.png"
            img.write_bytes(b"png")
            out = root / "out.md"
            debug_dir = root / "debug"
            atomic_dir = root / "atomic"

            exit_code = board2md_core.main(
                [
                    str(img),
                    "-o",
                    str(out),
                    "--debug-dir",
                    str(debug_dir),
                    "--atomic-dir",
                    str(atomic_dir),
                    "--obsidian",
                ],
                request_post=lambda **_kwargs: FakeResponse("<think>x</think>\n# T\n\nA"),
                env={"MINIMAX_API_KEY": "secret"},
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((debug_dir / "board.raw.md").exists())
            meta = json.loads((debug_dir / "board.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["source"], str(img))
            self.assertTrue((debug_dir / "board.png").exists())
            atomic_files = list(atomic_dir.glob("*.md"))
            self.assertEqual(len(atomic_files), 1)
            self.assertIn("[!source]", atomic_files[0].read_text(encoding="utf-8"))

    def test_continue_on_error_keeps_successful_pages_and_returns_partial_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.png"
            second = root / "b.png"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            out = root / "out.md"

            def fake_post(**kwargs):
                first_image = kwargs["json"]["messages"][0]["content"][0]["image_url"]["url"]
                if "Yg==" in first_image:
                    raise RuntimeError("boom")
                return FakeResponse("# A")

            exit_code = board2md_core.main(
                [str(first), str(second), "-o", str(out), "--continue-on-error", "--retries", "1"],
                request_post=fake_post,
                env={"MINIMAX_API_KEY": "secret"},
            )

            text = out.read_text(encoding="utf-8")
            self.assertEqual(exit_code, 3)
            self.assertIn("# A", text)
            self.assertIn("board2md error", text)

    def test_resume_reuses_debug_raw_without_calling_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "board.png"
            img.write_bytes(b"png")
            debug_dir = root / "debug"
            debug_dir.mkdir()
            (debug_dir / "board.raw.md").write_text("# cached", encoding="utf-8")
            out = root / "out.md"

            def fail_if_called(**_kwargs):
                raise AssertionError("resume should reuse debug raw output")

            exit_code = board2md_core.main(
                [str(img), "-o", str(out), "--debug-dir", str(debug_dir), "--resume"],
                request_post=fail_if_called,
                env={"MINIMAX_API_KEY": "secret"},
            )

            self.assertEqual(exit_code, 0)
            self.assertIn("# cached", out.read_text(encoding="utf-8"))

    def test_directory_inputs_and_sips_max_side_preprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "boards"
            folder.mkdir()
            (folder / "a.png").write_bytes(b"a")
            nested = folder / "nested"
            nested.mkdir()
            (nested / "b.jpg").write_bytes(b"b")
            preprocessed = []

            def fake_run(command, **_kwargs):
                self.assertEqual(command[0], "sips")
                out_path = Path(command[-1])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"small")
                preprocessed.append(out_path)

            items = board2md_core.expand_inputs(
                [str(folder)],
                root / "pdf_tmp",
                recursive=True,
            )
            items = board2md_core.prepare_inputs(
                items,
                root / "prep",
                max_side=1200,
                run_command=fake_run,
            )

            self.assertEqual([item.source_path.name for item in items], ["a.png", "b.jpg"])
            self.assertEqual(len(preprocessed), 2)
            self.assertEqual(items[0].path.read_bytes(), b"small")

    def test_main_dry_run_lists_pdf_source_without_api_or_pdf_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "board.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            out = root / "out.md"

            def fail_if_called(*_args, **_kwargs):
                raise AssertionError("external work should not run during dry-run")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = board2md_core.main(
                    [str(pdf), "--dry-run", "-o", str(out)],
                    run_command=fail_if_called,
                    request_post=fail_if_called,
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("Would process 1 inputs:", stdout.getvalue())
            self.assertIn(str(pdf), stdout.getvalue())
            self.assertFalse(out.exists())

    def test_dry_run_reports_validation_and_lecture_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "board.png"
            img.write_bytes(b"png")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = board2md_core.main(
                    [
                        str(img),
                        "--dry-run",
                        "--cross-validate",
                        "balanced",
                        "--lecture-synthesis",
                        "teaching",
                    ],
                    request_post=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("no API")),
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("cross_validate=balanced", stdout.getvalue())
            self.assertIn("lecture_synthesis=teaching", stdout.getvalue())

    def test_strip_think_block_removes_model_reasoning_prefix(self):
        text = "<think>hidden</think>\n\n# 题目\n\n内容"

        self.assertEqual(board2md_core.strip_think_block(text), "# 题目\n\n内容")

    def test_cross_validate_balanced_runs_two_passes_and_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "board.png"
            img.write_bytes(b"png")
            out = root / "out.md"
            debug_dir = root / "debug"
            calls = []

            def fake_post(**kwargs):
                content = kwargs["json"]["messages"][0]["content"]
                prompt = content[-1]["text"] if isinstance(content, list) else content
                calls.append(prompt)
                if "独立识别 A" in prompt:
                    return FakeResponse("# A\n\n题面：$a+b$")
                if "独立识别 B" in prompt:
                    return FakeResponse("# B\n\n题面：$a+b$")
                if "交叉验证裁判" in prompt:
                    return FakeResponse(
                        json.dumps(
                            {
                                "status": "agree",
                                "confidence": 0.92,
                                "conflicts": [],
                                "merged_markdown": "# 合并题面\n\n$a+b$",
                                "audit_markdown": "## 审校\n\n两路一致。",
                            },
                            ensure_ascii=False,
                        )
                    )
                raise AssertionError(prompt)

            exit_code = board2md_core.main(
                [
                    str(img),
                    "-o",
                    str(out),
                    "--cross-validate",
                    "balanced",
                    "--debug-dir",
                    str(debug_dir),
                ],
                request_post=fake_post,
                env={"MINIMAX_API_KEY": "secret"},
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(calls), 3)
            self.assertIn("# 合并题面", out.read_text(encoding="utf-8"))
            report = json.loads((debug_dir / "verification.json").read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["items"], 1)
            self.assertEqual(report["items"][0]["status"], "agree")
            self.assertTrue((debug_dir / "board.pass-a.md").exists())
            self.assertTrue((debug_dir / "board.pass-b.md").exists())
            self.assertTrue((debug_dir / "board.judge.json").exists())

    def test_lecture_synthesis_keeps_clean_handout_and_separate_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "p1.png"
            second = root / "p2.png"
            first.write_bytes(b"1")
            second.write_bytes(b"2")
            out = root / "lecture.md"
            audit = root / "lecture_audit.md"
            debug_dir = root / "debug"

            def fake_post(**kwargs):
                content = kwargs["json"]["messages"][0]["content"]
                prompt = content[-1]["text"] if isinstance(content, list) else content
                if "生成正式讲义" in prompt:
                    return FakeResponse(
                        "# 完整讲义\n\n## 定义\n\n正式定义。\n\n## 例题\n\n完整证明。"
                    )
                return FakeResponse("# 片段\n\n板书体面：不要进入正文。")

            exit_code = board2md_core.main(
                [
                    str(first),
                    str(second),
                    "-o",
                    str(out),
                    "--lecture-synthesis",
                    "teaching",
                    "--lecture-audit",
                    str(audit),
                    "--debug-dir",
                    str(debug_dir),
                ],
                request_post=fake_post,
                env={"MINIMAX_API_KEY": "secret"},
            )

            self.assertEqual(exit_code, 0)
            lecture = out.read_text(encoding="utf-8")
            self.assertIn("# 完整讲义", lecture)
            self.assertNotIn("板书体面", lecture)
            audit_text = audit.read_text(encoding="utf-8")
            self.assertIn("Source Map", audit_text)
            self.assertIn("p1", audit_text)
            self.assertTrue((debug_dir / "lecture_plan.json").exists())

    def test_profile_lecture_sets_validation_and_synthesis_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "board.png"
            img.write_bytes(b"png")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = board2md_core.main([str(img), "--profile", "lecture", "--dry-run"])

            self.assertEqual(exit_code, 0)
            self.assertIn("cross_validate=balanced", stdout.getvalue())
            self.assertIn("lecture_synthesis=teaching", stdout.getvalue())

    def test_note_type_whiteboard_changes_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "board.png"
            img.write_bytes(b"png")
            out = root / "out.md"
            seen = {}

            def fake_post(**kwargs):
                content = kwargs["json"]["messages"][0]["content"]
                seen["prompt"] = content[-1]["text"]
                return FakeResponse("# ok")

            exit_code = board2md_core.main(
                [str(img), "-o", str(out), "--note-type", "whiteboard"],
                request_post=fake_post,
                env={"MINIMAX_API_KEY": "secret"},
            )

            self.assertEqual(exit_code, 0)
            self.assertIn("白板", seen["prompt"])
            self.assertIn("空间布局", seen["prompt"])

    def test_quality_report_detects_provenance_and_can_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "board.png"
            img.write_bytes(b"png")
            out = root / "out.md"
            report = root / "quality.json"

            exit_code = board2md_core.main(
                [
                    str(img),
                    "-o",
                    str(out),
                    "--quality-report",
                    str(report),
                    "--fail-on-quality",
                ],
                request_post=lambda **_kwargs: FakeResponse("# 讲义\n\n根据图片可知，板书体面如下。"),
                env={"MINIMAX_API_KEY": "secret"},
            )

            self.assertEqual(exit_code, board2md_core.QUALITY_FAILURE_EXIT)
            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertGreater(data["summary"]["issue_count"], 0)
            self.assertIn("forbidden_provenance", data["issues"][0]["kind"])

    def test_output_format_both_writes_markdown_json_chunks_and_run_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "board.png"
            img.write_bytes(b"png")
            out = root / "out.md"
            chunks = root / "chunks.json"
            run_report = root / "run.json"

            exit_code = board2md_core.main(
                [
                    str(img),
                    "-o",
                    str(out),
                    "--output-format",
                    "both",
                    "--chunks-output",
                    str(chunks),
                    "--run-report",
                    str(run_report),
                ],
                request_post=lambda **_kwargs: FakeResponse("# 标题\n\n## 小节\n\n内容"),
                env={"MINIMAX_API_KEY": "secret"},
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(out.exists())
            self.assertTrue(out.with_suffix(".json").exists())
            self.assertEqual(json.loads(chunks.read_text(encoding="utf-8"))[0]["heading"], "# 标题")
            self.assertEqual(json.loads(run_report.read_text(encoding="utf-8"))["items"], 1)

    def test_max_pages_guard_blocks_large_runs_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.png"
            second = root / "b.png"
            first.write_bytes(b"a")
            second.write_bytes(b"b")

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = board2md_core.main(
                    [str(first), str(second), "--max-pages", "1"],
                    env={"MINIMAX_API_KEY": "secret"},
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("exceeds --max-pages", stderr.getvalue())

    def test_cache_dir_reuses_cached_raw_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "board.png"
            img.write_bytes(b"png")
            out1 = root / "out1.md"
            out2 = root / "out2.md"
            cache_dir = root / "cache"
            calls = {"count": 0}

            def fake_post(**_kwargs):
                calls["count"] += 1
                return FakeResponse("# cached result")

            first_code = board2md_core.main(
                [str(img), "-o", str(out1), "--cache-dir", str(cache_dir)],
                request_post=fake_post,
                env={"MINIMAX_API_KEY": "secret"},
            )
            second_code = board2md_core.main(
                [str(img), "-o", str(out2), "--cache-dir", str(cache_dir)],
                request_post=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
                env={"MINIMAX_API_KEY": "secret"},
            )

            self.assertEqual(first_code, 0)
            self.assertEqual(second_code, 0)
            self.assertEqual(calls["count"], 1)
            self.assertIn("# cached result", out2.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
