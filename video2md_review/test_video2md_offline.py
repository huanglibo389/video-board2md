import importlib.util
import importlib.machinery
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_video2md():
    os.environ.pop("MINIMAX_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("VIDEO2MD_PROVIDER", None)
    os.environ.pop("VIDEO2MD_CODEX_MODE", None)
    os.environ.pop("OPENAI_MODEL", None)
    os.environ.pop("CODEX_MODEL", None)
    os.environ.pop("QWEN_BASE_URL", None)
    os.environ.pop("QWEN_MODEL", None)
    loader = importlib.machinery.SourceFileLoader("video2md_under_test", str(ROOT / "video2md"))
    spec = importlib.util.spec_from_loader("video2md_under_test", loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_v2md_queue2():
    loader = importlib.machinery.SourceFileLoader("v2md_queue2_under_test", str(ROOT / "v2md-queue2"))
    spec = importlib.util.spec_from_loader("v2md_queue2_under_test", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Video2MdOfflineTests(unittest.TestCase):
    def test_import_without_api_key(self):
        module = load_video2md()
        self.assertIsNone(module.API_KEY)

    def test_provider_defaults_to_qwen(self):
        module = load_video2md()

        self.assertEqual(module.resolve_provider(None), "qwen")

    def test_default_model_is_local_qwen38_27b(self):
        module = load_video2md()

        self.assertEqual(
            module.MODEL,
            str(Path.home() / ".omlx/models/Qwen3.8-27B-MLX-Mixed-4.95bpw"),
        )

    def test_provider_accepts_qwen_alias_and_cli(self):
        module = load_video2md()
        os.environ["VIDEO2MD_PROVIDER"] = "qwen3.8-27b"

        self.assertEqual(module.resolve_provider(None), "qwen")
        self.assertEqual(module.resolve_provider("qwen"), "qwen")

    def test_provider_rejects_non_qwen(self):
        module = load_video2md()
        os.environ["VIDEO2MD_PROVIDER"] = "openai"

        with self.assertRaises(ValueError):
            module.resolve_provider(None)

    def test_codex_mode_defaults_to_final_and_accepts_full(self):
        module = load_video2md()
        os.environ["VIDEO2MD_CODEX_MODE"] = "full"

        self.assertEqual(module.resolve_codex_mode(None), "full")
        self.assertEqual(module.resolve_codex_mode("final"), "final")

    def test_codex_complete_uses_codex_exec_and_output_file(self):
        module = load_video2md()
        calls = []

        def fake_run(cmd, **kw):
            calls.append((cmd, kw))
            out_path = Path(cmd[cmd.index("-o") + 1])
            out_path.write_text("CODEX OK\n", encoding="utf-8")

            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        module.run = fake_run

        result = module.codex_complete("只输出 OK")

        cmd, kw = calls[0]
        self.assertEqual(result, "CODEX OK")
        self.assertEqual(cmd[:2], ["codex", "exec"])
        self.assertIn("--ephemeral", cmd)
        self.assertIn("--ignore-user-config", cmd)
        self.assertIn("--ignore-rules", cmd)
        self.assertIn("--skip-git-repo-check", cmd)
        self.assertEqual(cmd[cmd.index("-m") + 1], "gpt-5.4")
        self.assertEqual(cmd[-1], "-")
        self.assertEqual(kw["input"], "只输出 OK")
        self.assertEqual(kw["timeout"], 1800)

    def test_codex_exec_retries_transient_failure(self):
        module = load_video2md()
        calls = []

        def fake_run(cmd, **kw):
            calls.append((cmd, kw))

            class Result:
                stdout = ""

            if len(calls) == 1:
                Result.returncode = 1
                Result.stderr = "temporary network failure"
                return Result()

            out_path = Path(cmd[cmd.index("-o") + 1])
            out_path.write_text("RECOVERED\n", encoding="utf-8")
            Result.returncode = 0
            Result.stderr = ""
            return Result()

        module.run = fake_run
        module.time.sleep = lambda _: None

        result = module.codex_exec("retry please", max_retries=2)

        self.assertEqual(result, "RECOVERED")
        self.assertEqual(len(calls), 2)

    def test_codex_exec_raises_quota_without_retry(self):
        module = load_video2md()
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)

            class Result:
                returncode = 1
                stdout = ""
                stderr = "ERROR: You've hit your usage limit. try again at 12:23 PM."

            return Result()

        module.run = fake_run

        with self.assertRaises(module.CodexUsageLimitError):
            module.codex_exec("quota", max_retries=3)

        self.assertEqual(len(calls), 1)

    def test_describe_frames_aborts_on_codex_quota(self):
        module = load_video2md()

        def fake_vision(*args, **kwargs):
            raise module.CodexUsageLimitError("quota")

        module.llm_vision = fake_vision
        module.INTERVAL_S = 60
        with tempfile.TemporaryDirectory() as td:
            frame = Path(td) / "frame_000000.jpg"
            frame.write_bytes(b"fake image")

            with self.assertRaises(module.CodexUsageLimitError):
                module.step_describe_frames([frame], batch_size=1)

            self.assertEqual((Path(td) / "descriptions.json").read_text(encoding="utf-8"), "{}")

    def test_qwen_provider_uses_two_final_workers(self):
        module = load_video2md()
        module.LLM_PROVIDER = "qwen"

        self.assertEqual(module.final_generation_workers(), 2)

    def test_codex_vision_attaches_images(self):
        module = load_video2md()
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            out_path = Path(cmd[cmd.index("-o") + 1])
            out_path.write_text("VISION CODEX\n", encoding="utf-8")

            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        module.run = fake_run
        module.prepare_image_for_vision = lambda p: Path(p)
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "frame.jpg"
            img.write_bytes(b"not-real-but-attached")

            result = module.codex_vision([img], "看图")

        cmd = calls[0]
        self.assertEqual(result, "VISION CODEX")
        self.assertIn("--image", cmd)
        self.assertEqual(Path(cmd[cmd.index("--image") + 1]), img)

    def test_qwen_vision_dispatches_to_local_qwen(self):
        module = load_video2md()
        module.LLM_PROVIDER = "qwen"
        calls = []

        def fake_qwen_vision(image_paths, prompt, max_tokens=2000):
            calls.append((image_paths, prompt, max_tokens))
            return "QWEN VISION"

        module.qwen_vision = fake_qwen_vision

        result = module.llm_vision([Path("frame.jpg")], "看图", max_tokens=123)

        self.assertEqual(result, "QWEN VISION")
        self.assertEqual(calls, [([Path("frame.jpg")], "看图", 123)])

    def test_openai_complete_sends_responses_payload(self):
        module = load_video2md()
        os.environ["OPENAI_API_KEY"] = "sk-test"
        calls = []

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"output_text": "OK"}

        def fake_post(url, headers, json, timeout):
            calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
            return FakeResponse()

        module.requests.post = fake_post

        result = module.openai_complete("hello", max_tokens=123, temperature=0.2, max_retries=1)

        self.assertEqual(result, "OK")
        self.assertEqual(calls[0]["url"], "https://api.openai.com/v1/responses")
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(calls[0]["json"]["model"], "gpt-5.5")
        self.assertEqual(calls[0]["json"]["input"], "hello")
        self.assertEqual(calls[0]["json"]["max_output_tokens"], 123)

    def test_openai_vision_sends_input_images(self):
        module = load_video2md()
        os.environ["OPENAI_API_KEY"] = "sk-test"
        calls = []

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"output_text": "VISION OK"}

        def fake_post(url, headers, json, timeout):
            calls.append(json)
            return FakeResponse()

        module.requests.post = fake_post
        module.prepare_image_for_vision = lambda p: Path(p)
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "frame.jpg"
            img.write_bytes(b"not-real-but-base64-ok")

            result = module.openai_vision([img], "describe", max_tokens=456)

        self.assertEqual(result, "VISION OK")
        content = calls[0]["input"][0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "describe"})
        self.assertEqual(content[1]["type"], "input_image")
        self.assertTrue(content[1]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(calls[0]["max_output_tokens"], 456)

    def test_parse_vtt_multiline_cue(self):
        module = load_video2md()
        with tempfile.TemporaryDirectory() as td:
            vtt = Path(td) / "sample.vtt"
            vtt.write_text(
                "WEBVTT\n\n"
                "00:00:01.000 --> 00:00:03.500\n"
                "第一行\n"
                "第二行\n\n"
                "00:00:04.000 --> 00:00:05.000\n"
                "第三行\n",
                encoding="utf-8",
            )
            self.assertEqual(
                module.parse_vtt(vtt),
                [
                    (1.0, 3.5, "第一行 第二行"),
                    (4.0, 5.0, "第三行"),
                ],
            )

    def test_missing_batches_keep_sparse_indices(self):
        module = load_video2md()
        frames = [Path(f"frame_{i}.jpg") for i in range(5)]
        batches = module.make_missing_batches([0, 2, 3], frames, batch_size=2)
        self.assertEqual(
            batches,
            [
                [(0, frames[0])],
                [(2, frames[2]), (3, frames[3])],
            ],
        )

    def test_prepare_image_for_vision_resizes_large_frames(self):
        try:
            from PIL import Image
        except Exception as e:
            self.skipTest(f"Pillow unavailable: {e}")

        module = load_video2md()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "large.jpg"
            Image.new("RGB", (2400, 1200), "white").save(src, quality=95)

            prepared = module.prepare_image_for_vision(src, max_side=1600, quality=80)

            self.assertNotEqual(src, prepared)
            self.assertTrue(prepared.exists())
            with Image.open(prepared) as im:
                self.assertLessEqual(max(im.size), 1600)

    def test_problem_segments_merge_short_similar_fragments(self):
        module = load_video2md()
        problems = [
            {"start_idx": 0, "end_idx": 0, "title": "函数单调性"},
            {"start_idx": 1, "end_idx": 1, "title": "函数单调性"},
            {"start_idx": 2, "end_idx": 4, "title": "新例题：圆"},
        ]

        merged = module.merge_problem_segments(problems, min_frames=2)

        self.assertEqual(
            merged,
            [
                {"start_idx": 0, "end_idx": 1, "title": "函数单调性"},
                {"start_idx": 2, "end_idx": 4, "title": "新例题：圆"},
            ],
        )

    def test_segment_boundaries_ignore_duplicate_first_new_topic(self):
        module = load_video2md()
        descriptions = {
            "0": {"is_new_topic": True, "topic_title": "第一题"},
            "1": {"is_new_topic": False, "topic_title": ""},
            "2": {"is_new_topic": True, "topic_title": "第二题"},
        }

        problems = module.step_segment_problems(descriptions, cues=[])

        self.assertEqual(problems[0]["start_idx"], 0)
        self.assertEqual(problems[0]["end_idx"], 1)
        self.assertEqual(problems[1]["start_idx"], 2)

    def test_manifest_sorted_before_index(self):
        module = load_video2md()
        manifest = [
            {"idx": 3, "file": "p003.md", "title": "三"},
            {"idx": 1, "file": "p001.md", "title": "一"},
            {"idx": 2, "file": "p002.md", "title": "二"},
        ]
        with tempfile.TemporaryDirectory() as td:
            module.write_index(Path(td), manifest)
            text = (Path(td) / "index.md").read_text(encoding="utf-8")
            self.assertLess(text.index("[1. 一]"), text.index("[2. 二]"))
            self.assertLess(text.index("[2. 二]"), text.index("[3. 三]"))

    def test_atomic_write_json_replaces_complete_file(self):
        module = load_video2md()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.json"
            path.write_text('{"old": true}', encoding="utf-8")

            module.atomic_write_json(path, {"new": [1, 2, 3]})

            self.assertEqual(path.read_text(encoding="utf-8"), '{\n  "new": [\n    1,\n    2,\n    3\n  ]\n}')
            self.assertEqual(list(Path(td).glob("*.tmp")), [])

    def test_dedupe_frame_entries_prefers_scene_near_same_timestamp(self):
        module = load_video2md()
        entries = [
            {"path": Path("interval.jpg"), "timestamp": 20.0, "source": "interval"},
            {"path": Path("scene.jpg"), "timestamp": 20.4, "source": "scene"},
            {"path": Path("later.jpg"), "timestamp": 25.0, "source": "interval"},
        ]

        deduped = module.dedupe_frame_entries(entries, tolerance_s=0.75)

        self.assertEqual([item["path"].name for item in deduped], ["scene.jpg", "later.jpg"])

    def test_finalize_frames_removes_dropped_duplicate_sources(self):
        module = load_video2md()
        with tempfile.TemporaryDirectory() as td:
            frames_dir = Path(td)
            interval = frames_dir / "f0000000001.jpg"
            scene = frames_dir / "scene_0000000001.jpg"
            interval.write_bytes(b"interval")
            scene.write_bytes(b"scene")

            frames = module.finalize_frames_with_manifest(frames_dir, [
                {"path": interval, "timestamp": 10.0, "source": "interval"},
                {"path": scene, "timestamp": 10.2, "source": "scene"},
            ])

            self.assertEqual([p.name for p in frames], ["frame_000000.jpg"])
            self.assertFalse(interval.exists())
            self.assertFalse(scene.exists())
            self.assertEqual(sorted(p.name for p in frames_dir.glob("*.jpg")), ["frame_000000.jpg"])

    def test_clear_directory_removes_nested_cache_dirs(self):
        module = load_video2md()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache = root / ".vision_cache"
            cache.mkdir()
            (cache / "cached.jpg").write_bytes(b"cache")
            (root / "frames_manifest.json").write_text("{}", encoding="utf-8")

            module.clear_directory(root)

            self.assertEqual(list(root.iterdir()), [])

    def test_extract_frames_json_from_fenced_object(self):
        module = load_video2md()
        raw = '前言\n```json\n{"frames": [{"idx": 0, "description": "A"}]}\n```\n后记'

        payload = module.extract_frames_payload(raw)

        self.assertEqual(payload["frames"][0]["description"], "A")

    def test_extract_frames_repairs_latex_backslashes(self):
        module = load_video2md()
        raw = '{"frames": [{"idx": 0, "description": "公式 \\ln x + \\frac{1}{2} + \\_ + \\{x\\}"}]}'

        payload = module.extract_frames_payload(raw)

        self.assertEqual(payload["frames"][0]["description"], "公式 \\ln x + \\frac{1}{2} + \\_ + \\{x\\}")

    def test_extract_frames_preserves_existing_escaped_latex_commands(self):
        module = load_video2md()
        raw = '{"frames": [{"idx": 0, "description": "公式 \\(n\\\\to \\\\sqrt{x}\\)"}]}'

        payload = module.extract_frames_payload(raw)

        self.assertEqual(payload["frames"][0]["description"], "公式 \\(n\\to \\sqrt{x}\\)")

    def test_extract_frames_repairs_latex_commands_that_look_like_json_escapes(self):
        module = load_video2md()
        raw = '{"frames": [{"idx": 0, "description": "公式 \\theta + \\nabla + \\tan x"}]}'

        payload = module.extract_frames_payload(raw)

        self.assertEqual(payload["frames"][0]["description"], "公式 \\theta + \\nabla + \\tan x")

    def test_obsidian_math_delimiters(self):
        module = load_video2md()
        text = (
            "行内公式 \\(a+b\\)，代码 `\\(not math\\)`\n"
            "\\[\n"
            "a^2+b^2=c^2\n"
            "\\]\n"
        )

        converted = module.normalize_markdown_for_obsidian(text)

        self.assertIn("行内公式 $a+b$，代码 `\\(not math\\)`", converted)
        self.assertIn("$$\na^2+b^2=c^2\n$$", converted)

    def test_obsidian_math_delimiters_in_blockquote(self):
        module = load_video2md()
        text = "> \\[\n> a+b=c\n> \\]\n"

        converted = module.normalize_markdown_for_obsidian(text)

        self.assertEqual(converted, "> $$\n> a+b=c\n> $$\n")

    def test_select_problem_visual_frames_prefers_explicit_diagrams(self):
        module = load_video2md()
        frames = [Path(f"frame_{i:06d}.jpg") for i in range(4)]
        descriptions = {
            "0": {"description": "纯文字板书", "has_visual_diagram": False},
            "1": {
                "description": "画面中有三角形 ABC 和外接圆",
                "has_visual_diagram": True,
                "visual_type": "geometry",
                "visual_importance": 5,
            },
            "2": {
                "description": "棋盘染色示意图",
                "has_visual_diagram": True,
                "visual_type": "combinatorics",
                "visual_importance": 4,
            },
            "3": {"description": "继续讲解文字"},
        }
        problem = {"start_idx": 0, "end_idx": 3, "title": "几何组合题"}

        selected = module.select_problem_visual_frames(
            problem,
            frames,
            descriptions,
            mode="auto",
            max_images=2,
        )

        self.assertEqual([idx for idx, _ in selected], [1, 2])

    def test_auto_visual_selection_uses_geometry_keyword_fallback(self):
        module = load_video2md()
        frames = [Path(f"frame_{i:06d}.jpg") for i in range(2)]
        descriptions = {
            "0": {"description": "此页只有代数公式"},
            "1": {"description": "屏幕显示圆、切线和角平分线的几何图形"},
        }
        problem = {"start_idx": 0, "end_idx": 1, "title": "几何题"}

        selected = module.select_problem_visual_frames(
            problem,
            frames,
            descriptions,
            mode="auto",
            max_images=1,
        )

        self.assertEqual(selected, [(1, frames[1])])

    def test_materialize_visual_assets_and_inject_markdown_section(self):
        module = load_video2md()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            frames_dir = root / "frames"
            problems_dir = root / "problems"
            frames_dir.mkdir()
            problems_dir.mkdir()
            frame = frames_dir / "frame_000042.jpg"
            frame.write_bytes(b"fake-jpeg")

            assets = module.materialize_problem_visual_assets(
                [(42, frame)],
                problems_dir,
                problem_number=3,
            )
            content = (
                "# 几何题\n\n"
                "**来源**: 视频 [t=10s-20s]\n"
                "**关键词**: [几何]\n\n"
                "## 题面（板书/PPT 提取）\n"
                "如图，求证。\n\n"
                "## 解答\n"
                "证明如下。\n"
            )

            updated = module.inject_visual_section(content, assets)

            self.assertEqual(len(assets), 1)
            self.assertEqual(assets[0]["relative_path"], "assets/p003_frame_000042.jpg")
            self.assertTrue((problems_dir / "assets" / "p003_frame_000042.jpg").exists())
            self.assertIn("## 视频截图", updated)
            self.assertIn("![视频截图 1（t≈??:??）](assets/p003_frame_000042.jpg)", updated)
            self.assertLess(updated.index("## 视频截图"), updated.index("## 解答"))
            self.assertEqual(module.inject_visual_section(updated, assets), updated)

    def test_frame_description_requires_visual_metadata_for_complete_cache(self):
        module = load_video2md()

        self.assertFalse(module.frame_description_has_visual_metadata({"description": "旧缓存"}))
        self.assertTrue(module.frame_description_has_visual_metadata({
            "description": "有三角形",
            "has_visual_diagram": True,
            "visual_type": "geometry",
            "visual_importance": 5,
        }))

    def test_describe_frames_upgrades_legacy_cache_missing_visual_metadata(self):
        module = load_video2md()
        with tempfile.TemporaryDirectory() as td:
            frames_dir = Path(td) / "frames"
            frames_dir.mkdir()
            frames = []
            for idx in range(3):
                frame = frames_dir / f"frame_{idx:06d}.jpg"
                frame.write_bytes(b"fake-jpeg")
                frames.append(frame)

            desc_path = frames_dir / "descriptions.json"
            desc_path.write_text(
                """{
  "0": {"description": "旧缓存，有几何图", "is_new_topic": true, "topic_title": "几何"},
  "1": {"description": "新缓存", "is_new_topic": false, "topic_title": "", "has_visual_diagram": false, "visual_type": "text", "visual_importance": 0}
}""",
                encoding="utf-8",
            )

            calls = []
            module.INTERVAL_S = 60

            def fake_vision(image_paths, prompt, max_tokens=2000):
                calls.append([Path(path).name for path in image_paths])
                items = []
                for path in image_paths:
                    idx = int(Path(path).stem.rsplit("_", 1)[1])
                    items.append({
                        "idx": idx,
                        "description": f"补全 frame {idx}",
                        "is_new_topic": idx == 0,
                        "topic_title": "补全",
                        "has_visual_diagram": idx == 0,
                        "visual_type": "geometry" if idx == 0 else "text",
                        "visual_importance": 4 if idx == 0 else 0,
                    })
                return json.dumps({"frames": items}, ensure_ascii=False)

            module.llm_vision = fake_vision
            descriptions = module.step_describe_frames(frames)

            self.assertEqual(calls, [["frame_000000.jpg"], ["frame_000002.jpg"]])
            self.assertEqual(descriptions["0"]["visual_type"], "geometry")
            self.assertEqual(descriptions["1"]["description"], "新缓存")
            self.assertIn("visual_importance", descriptions["2"])

    def test_resolve_problem_markdown_path_reuses_existing_problem_number(self):
        module = load_video2md()
        with tempfile.TemporaryDirectory() as td:
            problems_dir = Path(td)
            existing = problems_dir / "p001_旧标题.md"
            existing.write_text("# old\n", encoding="utf-8")

            resolved = module.resolve_problem_markdown_path(
                problems_dir,
                problem_number=1,
                preferred_name="p001_新标题.md",
            )

            self.assertEqual(resolved, existing)

    def test_retrofit_existing_problem_markdowns_only_updates_existing_files(self):
        module = load_video2md()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            frames_dir = root / "frames"
            problems_dir = root / "problems"
            frames_dir.mkdir()
            problems_dir.mkdir()
            frame = frames_dir / "frame_000001.jpg"
            frame.write_bytes(b"fake-jpeg")
            existing = problems_dir / "p001_旧标题.md"
            existing.write_text("# 旧题\n\n## 解答\n已有解答\n", encoding="utf-8")
            (problems_dir / "p001_旧标题.raw.md").write_text("# raw\n", encoding="utf-8")
            problems = [{"start_idx": 0, "end_idx": 1, "title": "新标题"}]
            descriptions = {
                "1": {
                    "description": "圆与切线图",
                    "has_visual_diagram": True,
                    "visual_type": "geometry",
                    "visual_importance": 5,
                }
            }

            results = module.retrofit_existing_problem_markdowns(
                problems,
                [frames_dir / "frame_000000.jpg", frame],
                descriptions,
                problems_dir,
                mode="auto",
                max_images=1,
            )

            self.assertEqual(results, [(existing, "updated")])
            text = existing.read_text(encoding="utf-8")
            self.assertIn("## 视频截图", text)
            self.assertTrue((problems_dir / "assets" / "p001_frame_000001.jpg").exists())
            self.assertFalse((problems_dir / "p001_新标题.md").exists())

    def test_fuse_problem_markdowns_preserves_sources_images_and_backups(self):
        module = load_video2md()
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "lesson"
            problems_dir = out_dir / "problems"
            assets_dir = problems_dir / "assets"
            assets_dir.mkdir(parents=True)
            (assets_dir / "p001_frame_000001.jpg").write_bytes(b"img1")
            (assets_dir / "p002_frame_000002.jpg").write_bytes(b"img2")
            (assets_dir / "p003_frame_000003.jpg").write_bytes(b"img3")
            (problems_dir / "p001_拉丁方矩形题.md").write_text(
                "# 拉丁方矩形题\n\n"
                "**来源**: 视频 [t=0s–60s]\n\n"
                "## 题面（板书/PPT 提取）\n"
                "设有一个拉丁方矩形。\n\n"
                "## 视频截图\n\n"
                "![视频截图 1（t≈01:00）](assets/p001_frame_000001.jpg)\n\n"
                "## 解答\n"
                "先建立目标。\n",
                encoding="utf-8",
            )
            (problems_dir / "p002_置换化记号.md").write_text(
                "# 置换化记号\n\n"
                "**来源**: 视频 [t=60s–120s]\n\n"
                "## 题面（板书/PPT 提取）\n"
                "继续把行列写成置换。\n\n"
                "## 视频截图\n\n"
                "![视频截图 1（t≈02:00）](assets/p002_frame_000002.jpg)\n\n"
                "## 解答\n"
                "引入置换记号。\n",
                encoding="utf-8",
            )
            (problems_dir / "p003_置换分解事实.md").write_text(
                "# 置换分解事实\n\n"
                "**来源**: 视频 [t=120s–180s]\n\n"
                "## 题面（板书/PPT 提取）\n"
                "继续证明置换分解事实。\n\n"
                "## 视频截图\n\n"
                "![视频截图 1（t≈03:00）](assets/p003_frame_000003.jpg)\n\n"
                "## 解答\n"
                "完成关键事实。\n",
                encoding="utf-8",
            )
            (out_dir / "index.md").write_text(
                "# 视频题目索引\n\n"
                "- [1. 拉丁方矩形题](problems/p001_拉丁方矩形题.md)\n"
                "- [2. 置换化记号](problems/p002_置换化记号.md)\n"
                "- [3. 置换分解事实](problems/p003_置换分解事实.md)\n\n"
                "_共 3 道题_\n",
                encoding="utf-8",
            )

            result = module.fuse_problem_markdowns(out_dir, mode="auto", rounds=3)

            final_files = sorted(path.name for path in problems_dir.glob("p*.md"))
            self.assertEqual(final_files, ["p001_拉丁方矩形题.md"])
            fused = (problems_dir / "p001_拉丁方矩形题.md").read_text(encoding="utf-8")
            self.assertIn("## 来源片段", fused)
            self.assertIn("p002_置换化记号.md", fused)
            self.assertIn("视频 [t=120s–180s]", fused)
            self.assertIn("![视频截图 1（t≈01:00）](assets/p001_frame_000001.jpg)", fused)
            self.assertIn("![视频截图 2（t≈02:00）](assets/p002_frame_000002.jpg)", fused)
            self.assertIn("![视频截图 3（t≈03:00）](assets/p003_frame_000003.jpg)", fused)
            self.assertTrue((out_dir / "problems_raw_segments" / "p002_置换化记号.md").exists())
            self.assertTrue((out_dir / "fusion_manifest.json").exists())
            self.assertIn("p001_拉丁方矩形题.md", (out_dir / "index.md").read_text(encoding="utf-8"))
            self.assertEqual(result["groups"], 2)
            self.assertEqual(result["removed"], 2)
            self.assertEqual(result["rounds"], 3)

    def test_merge_supports_legacy_root_problem_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "lesson"
            source.mkdir()
            (source / "p001_旧结构.md").write_text("# 旧结构\n", encoding="utf-8")
            output = root / "merged.md"
            subprocess.run(
                [sys.executable, str(ROOT / "video2md-merge"), str(source), "-o", str(output)],
                check=True,
            )
            self.assertIn("旧结构", output.read_text(encoding="utf-8"))

    def test_merge_skips_raw_fallback_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "lesson"
            problems = source / "problems"
            problems.mkdir(parents=True)
            (problems / "p001_题目.md").write_text("# 正常版本\n", encoding="utf-8")
            (problems / "p001_题目.raw.md").write_text("# Raw 版本\n", encoding="utf-8")
            output = root / "merged.md"

            subprocess.run(
                [sys.executable, str(ROOT / "video2md-merge"), str(source), "-o", str(output)],
                check=True,
            )

            text = output.read_text(encoding="utf-8")
            self.assertIn("正常版本", text)
            self.assertNotIn("Raw 版本", text)

    def test_queue_output_dir_uses_provider_mode_and_interval_suffix(self):
        module = load_v2md_queue2()

        out_dir = module.build_output_dir(
            Path("/videos/第七讲 数学归纳法1.mp4"),
            Path("/out"),
            provider="qwen",
            codex_mode="full",
            interval=60,
        )

        self.assertEqual(out_dir, Path("/out/第七讲_数学归纳法1_qwen_i60"))

    def test_queue_detects_completed_output_from_index(self):
        module = load_v2md_queue2()
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "lesson_i60"
            out_dir.mkdir(parents=True)
            (out_dir / "index.md").write_text("# done\n", encoding="utf-8")

            self.assertTrue(module.is_completed_output(out_dir))

    def test_queue_builds_sleep_run_command_with_explicit_out_dir(self):
        module = load_v2md_queue2()

        cmd = module.build_run_command(
            video=Path("/videos/第八讲 数列1.mp4"),
            out_dir=Path("/out/第八讲_数列1_i60"),
            provider="qwen",
            codex_mode="full",
            interval=60,
            extra_args=["--limit", "600"],
        )

        self.assertEqual(cmd[0], str(ROOT / "v2md-sleep-run"))
        self.assertIn("--out", cmd)
        self.assertEqual(cmd[cmd.index("--out") + 1], "/out/第八讲_数列1_i60")
        self.assertEqual(cmd[-2:], ["--limit", "600"])

    def test_queue_parse_args_allows_forwarded_args_after_separator(self):
        module = load_v2md_queue2()

        args, extra = module.parse_args([
            "--jobs", "2",
            "/videos",
            "--",
            "--retrofit-existing-problems",
        ])

        self.assertEqual(args.inputs, ["/videos"])
        self.assertEqual(extra, ["--retrofit-existing-problems"])

    def test_sleep_run_defaults_to_project_video2md_entrypoint(self):
        script = (ROOT / "v2md-sleep-run").read_text(encoding="utf-8")

        self.assertIn('VIDEO2MD_BIN="${VIDEO2MD_BIN:-$SCRIPT_DIR/video2md}"', script)
        self.assertIn('"$VIDEO2MD_BIN" "$video" "$@"', script)
        self.assertNotIn('$HOME/.local/bin/video2md "$video"', script)

    def test_queue_scheduler_keeps_at_most_two_active_jobs(self):
        module = load_v2md_queue2()
        launch_order = []
        active_counts = []

        class FakeProcess:
            def __init__(self, name):
                self.name = name
                self.returncode = None
                self.polls_remaining = 2

            def poll(self):
                if self.returncode is not None:
                    return self.returncode
                self.polls_remaining -= 1
                if self.polls_remaining <= 0:
                    self.returncode = 0
                return self.returncode

        def fake_launcher(job):
            launch_order.append(job.video.name)
            return FakeProcess(job.video.name)

        jobs = [
            module.Job(Path("/videos/a.mp4"), Path("/out/a")),
            module.Job(Path("/videos/b.mp4"), Path("/out/b")),
            module.Job(Path("/videos/c.mp4"), Path("/out/c")),
        ]

        def fake_on_tick(active):
            active_counts.append(len(active))

        results = module.run_job_queue(
            jobs,
            jobs_limit=2,
            launcher=fake_launcher,
            poll_interval=0.0,
            on_tick=fake_on_tick,
        )

        self.assertEqual(launch_order, ["a.mp4", "b.mp4", "c.mp4"])
        self.assertTrue(active_counts)
        self.assertLessEqual(max(active_counts), 2)
        self.assertEqual([result.status for result in results], ["done", "done", "done"])

    def test_queue_parse_args_keeps_options_out_of_passthrough(self):
        module = load_v2md_queue2()

        args, extra = module.parse_args([
            "/videos",
            "--jobs",
            "2",
            "--provider",
            "qwen",
            "--codex-mode",
            "full",
            "--interval",
            "60",
        ])

        self.assertEqual(args.inputs, ["/videos"])
        self.assertEqual(args.jobs, 2)
        self.assertEqual(args.provider, "qwen")
        self.assertEqual(extra, [])


if __name__ == "__main__":
    unittest.main()
